import taichi as ti

from simulator.simulator_utils import *


# ============================================================================
# UNIFIED REWARD CONSTANTS
# These are the single source of truth for reward shaping. Both the simulator's
# autodiff loss (_accumulate_total) and the env's _calculate_reward MUST use these.
# If you change them here, they change everywhere.
# ============================================================================
REWARD_K_SIGMOID = 10.0         # sigmoid steepness for SDF-based soft indicators
REWARD_K_IDLE    = 10.0         # sigmoid steepness for idle gate (on normalized total)
REWARD_W_GOOD    = 10.0
REWARD_W_BAD     = 10.0
REWARD_W_BOUND   = 5.0
REWARD_W_PROG    = 1.0
REWARD_W_IDLE    = 1.0
REWARD_W_HOLDER  = 1.0
BOUNDARY_SIGMA   = 8.0          # exp(-sigma * target_dist^2) falloff near target surface
BOUNDARY_STOCK_OFFSET = 0.05    # stock-presence sigmoid offset
IDLE_THRESHOLD   = 0.1          # normalized cut threshold under which idle kicks in


@ti.data_oriented
class CNCSimulator:
    _ti_initialized = False

    def __init__(self, resolution=128, debug=False):
        # Initialize Taichi (only on first instantiation)
        if not CNCSimulator._ti_initialized:
            if ti._lib.core.with_cuda():
                ti.init(arch=ti.gpu, debug=debug)
            else:
                ti.init(arch=ti.cpu, debug=debug)
            CNCSimulator._ti_initialized = True

        self.res = resolution
        self.dx = 1.0 / self.res
        # Normalization factor used to put "total_removed" on a scale-invariant
        # basis before feeding the idle sigmoid. Using res**3 makes IDLE_THRESHOLD
        # a fraction-of-grid quantity, independent of resolution.
        self.grid_norm = float(self.res ** 3)

        # Define the SDF fields for stock and target geometry
        self.sdf_stock = ti.field(dtype=ti.f32, shape=(self.res, self.res, self.res))
        self.sdf_target = ti.field(dtype=ti.f32, shape=(self.res, self.res, self.res))

        # Stock snapshot BEFORE the cut — needed so autodiff differentiates the
        # same function the env evaluates: reward(stock_before, stock_after(tool_pos), ...)
        # Must NOT need grad (it's a constant w.r.t. tool_pos during the tape run).
        self.sdf_stock_before = ti.field(dtype=ti.f32, shape=(self.res, self.res, self.res))

        # Define the tool
        self.tool_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
        self.tool_radius = ti.field(dtype=ti.f32, shape=())
        self.tool_height = ti.field(dtype=ti.f32, shape=())

        # Loss fields (all need grad — tape writes into their .grad)
        self.loss             = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.loss_good_cuts   = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.loss_bad_cuts    = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.loss_boundary    = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.loss_progress    = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.loss_idle        = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.loss_holder      = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        # ----- Visualization fields  -----
        self.stock_points = ti.Vector.field(3, dtype=ti.f32, shape=self.res**3)
        self.stock_count = ti.field(dtype=ti.i32, shape=())

        self.tool_points = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.tool_template = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.tool_count = ti.field(dtype=ti.i32, shape=())

        self.holder_points = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.holder_template = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.holder_count = ti.field(dtype=ti.i32, shape=())

        self.target_points = ti.Vector.field(3, dtype=ti.f32, shape=self.res**3)
        self.target_count = ti.field(dtype=ti.i32, shape=())

        self.slice_xy = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
        self.slice_xz = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
        self.slice_yz = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))

        self.debug_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(341, 2 * self.res))
        self.raymarch_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(1024, 768))

        # Scalar "volume removed" counter (voxel-count proxy, not true volume)
        self.removed_vol_field = ti.field(dtype=ti.f32, shape=())
        self.excess_field = ti.field(dtype=ti.f32, shape=())

    # ========================================================================
    # Initialization kernels
    # ========================================================================

    @ti.kernel
    def initialize_target_sphere(self, radius: ti.f32):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            self.sdf_target[i, j, k] = sphere_sdf(p, center, radius)

    @ti.kernel
    def initialize_target_cube(self, half_size: ti.f32):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            self.sdf_target[i, j, k] = box_sdf(p, center, half_size)

    @ti.kernel
    def initialize_target_pyramid(self, half_base: ti.f32, height: ti.f32):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5])
            base_z = 0.5 - height * 0.5
            self.sdf_target[i, j, k] = pyramid_sdf(
                p, center.x, center.y, base_z, half_base, height
            )

    @ti.kernel
    def initialize_target_cylinder(self, radius: ti.f32, height: ti.f32):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5])
            cz = 0.5 - height * 0.5
            self.sdf_target[i, j, k] = cylinder_sdf(
                p, center.x, center.y, cz, radius, height * 0.5
            )

    @ti.kernel
    def initialize_stock(self, half_size: ti.f32):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            self.sdf_stock[i, j, k] = box_sdf(p, center, half_size)

    def initialize_tool(self, pos, radius, height):
        self.tool_pos[None] = ti.Vector(pos)
        self.tool_radius[None] = radius
        self.tool_height[None] = height

    # ========================================================================
    # Tool / holder SDFs
    # ========================================================================

    @ti.func
    def dist_from_tool(self, p):
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        dx = p.x - tool_pos.x
        dy = p.y - tool_pos.y
        d_h = ti.sqrt(dx * dx + dy * dy + 1e-12) - tool_radius   # explicit, with eps for grad stability

        d_z_bottom = tool_pos.z - p.z
        d_z_top    = p.z - (tool_pos.z + tool_height)
        d_z        = ti.max(d_z_bottom, d_z_top)
        return ti.max(d_h, d_z)

    @ti.func
    def dist_from_holder(self, p):
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        holder_radius = tool_radius * 2.0
        holder_height = tool_height * 0.5
        holder_z_start = tool_pos.z + tool_height

        dx = p.x - tool_pos.x
        dy = p.y - tool_pos.y
        d_h = ti.sqrt(dx * dx + dy * dy + 1e-12) - holder_radius

        d_z_bottom = holder_z_start - p.z
        d_z_top    = p.z - (holder_z_start + holder_height)
        d_z        = ti.max(d_z_bottom, d_z_top)
        return ti.max(d_h, d_z)
    # ========================================================================
    # Cutting
    # ========================================================================

    @ti.kernel
    def snapshot_stock(self):
        """Copy current sdf_stock into sdf_stock_before. Call this BEFORE apply_cut
        so the gradient computation can see both pre- and post-cut fields."""
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            self.sdf_stock_before[i, j, k] = self.sdf_stock[i, j, k]

    def apply_cut(self) -> float:
        """
        Boolean subtraction: stock = max(stock, -tool)
        Returns voxel-count proxy for volume removed (not true volume).
        """
        self.removed_vol_field[None] = 0.0
        self._apply_cut_kernel()
        return float(self.removed_vol_field[None])

    @ti.kernel
    def _apply_cut_kernel(self):
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        min_x = int(ti.floor((tool_pos.x - tool_radius) / self.dx - 4.0))
        max_x = int(ti.ceil((tool_pos.x + tool_radius) / self.dx + 4.0))
        min_y = int(ti.floor((tool_pos.y - tool_radius) / self.dx - 4.0))
        max_y = int(ti.ceil((tool_pos.y + tool_radius) / self.dx + 4.0))
        min_z = int(ti.floor((tool_pos.z) / self.dx - 4.0))
        max_z = int(ti.ceil((tool_pos.z + tool_height) / self.dx + 4.0))

        min_x = ti.max(min_x, 0); max_x = ti.min(max_x, self.res)
        min_y = ti.max(min_y, 0); max_y = ti.min(max_y, self.res)
        min_z = ti.max(min_z, 0); max_z = ti.min(max_z, self.res)

        for i, j, k in ti.ndrange((min_x, max_x), (min_y, max_y), (min_z, max_z)):
            p = ti.Vector([i, j, k]) * self.dx
            tool_dist = self.dist_from_tool(p)
            stock_dist = self.sdf_stock[i, j, k]
            new_dist = ti.max(stock_dist, -tool_dist)

            # Safety mask: don't cut below target surface
            target_dist = self.sdf_target[i, j, k]
            if target_dist < 0.0:
                new_dist = ti.min(new_dist, target_dist)

            if new_dist != stock_dist:
                ti.atomic_add(self.removed_vol_field[None], 1.0)
                self.sdf_stock[i, j, k] = new_dist

    @ti.kernel
    def move_tool_one_unit(self, dir: ti.types.vector(3, ti.f32)):
        """Moves the tool one voxel in a unit direction."""
        # NOTE: removed the dead `valid_dir` check from the previous version —
        # it computed a flag and never used it. If direction validation matters,
        # enforce it on the Python side before calling this kernel.
        new_pos = self.tool_pos[None]
        for i in ti.static(range(3)):
            new_pos[i] = ti.max(
                0.0, ti.min(1.0 - self.dx, new_pos[i] + dir[i] * self.dx)
            )
        self.tool_pos[None] = new_pos

    @ti.kernel
    def check_holder_collision(self) -> ti.i32:
        """Returns 1 if holder intersects stock, 0 otherwise."""
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        holder_radius = tool_radius * 2.0
        holder_height = tool_height * 0.5
        holder_z_start = tool_pos.z + tool_height
        holder_z_end = holder_z_start + holder_height

        collision = 0
        min_x = ti.max(0, ti.cast(ti.floor((tool_pos.x - holder_radius) / self.dx) - 1, ti.i32))
        max_x = ti.min(self.res, ti.cast(ti.ceil((tool_pos.x + holder_radius) / self.dx) + 1, ti.i32))
        min_y = ti.max(0, ti.cast(ti.floor((tool_pos.y - holder_radius) / self.dx) - 1, ti.i32))
        max_y = ti.min(self.res, ti.cast(ti.ceil((tool_pos.y + holder_radius) / self.dx) + 1, ti.i32))
        min_z = ti.max(0, ti.cast(ti.floor(holder_z_start / self.dx) - 1, ti.i32))
        max_z = ti.min(self.res, ti.cast(ti.ceil(holder_z_end / self.dx) + 1, ti.i32))

        for i, j, k in ti.ndrange((min_x, max_x), (min_y, max_y), (min_z, max_z)):
            p = ti.Vector([i, j, k]) * self.dx
            if self.dist_from_holder(p) < 0 and self.sdf_stock[i, j, k] < 0:
                collision = 1
        return collision

    @ti.kernel
    def check_tool_intersects_target(self) -> ti.i32:
        """Returns 1 if tool intersects target geometry, 0 otherwise."""
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]
        cuts_target = 0
        step = self.dx * 0.5

        n_x = ti.cast(ti.ceil((2.0 * tool_radius + 2.0 * self.dx) / step), ti.i32) + 1
        n_y = ti.cast(ti.ceil((2.0 * tool_radius + 2.0 * self.dx) / step), ti.i32) + 1
        n_z = ti.cast(ti.ceil((tool_height + 2.0 * self.dx) / step), ti.i32) + 1

        origin_x = tool_pos.x - tool_radius - self.dx
        origin_y = tool_pos.y - tool_radius - self.dx
        origin_z = tool_pos.z - self.dx

        for i, j, k in ti.ndrange(n_x, n_y, n_z):
            p = ti.Vector([origin_x + i * step, origin_y + j * step, origin_z + k * step])
            if (0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 and 0.0 <= p.z <= 1.0):
                if self.dist_from_tool(p) <= 0:
                    target_val = self.interpolate_sdf(self.sdf_target, p)
                    if target_val <= 0:
                        cuts_target = 1
        return cuts_target

    def compute_excess(self) -> float:
        """Excess volume: sum(max(min(-sdf_stock, sdf_target), 0)) over full grid.
        Zero = stock is a subset of target (no excess to remove)."""
        self.excess_field[None] = 0.0
        self._compute_excess_kernel()
        return float(self.excess_field[None])

    @ti.kernel
    def _compute_excess_kernel(self):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            stock = self.sdf_stock[i, j, k]
            target = self.sdf_target[i, j, k]
            val = ti.max(ti.min(-stock, target), 0.0)
            if val > 0.0:
                ti.atomic_add(self.excess_field[None], val)

    # ========================================================================
    # Interpolation / normals
    # ========================================================================

    @ti.func
    def interpolate_sdf(self, field: ti.template(), p: ti.template()) -> ti.f32:
        p_grid = p * self.res
        x, y, z = p_grid.x, p_grid.y, p_grid.z

        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        z0 = ti.cast(ti.floor(z), ti.i32)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        tx, ty, tz = x - x0, y - y0, z - z0

        x0 = ti.max(0, ti.min(self.res - 1, x0)); x1 = ti.max(0, ti.min(self.res - 1, x1))
        y0 = ti.max(0, ti.min(self.res - 1, y0)); y1 = ti.max(0, ti.min(self.res - 1, y1))
        z0 = ti.max(0, ti.min(self.res - 1, z0)); z1 = ti.max(0, ti.min(self.res - 1, z1))

        c000 = field[x0, y0, z0]; c100 = field[x1, y0, z0]
        c010 = field[x0, y1, z0]; c110 = field[x1, y1, z0]
        c001 = field[x0, y0, z1]; c101 = field[x1, y0, z1]
        c011 = field[x0, y1, z1]; c111 = field[x1, y1, z1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    @ti.func
    def compute_surface_normal(self, field: ti.template(), p: ti.template()) -> ti.math.vec3:
        eps = self.dx * 1.5
        dx = ti.Vector([eps, 0.0, 0.0])
        dy = ti.Vector([0.0, eps, 0.0])
        dz = ti.Vector([0.0, 0.0, eps])
        nx = self.interpolate_sdf(field, p + dx) - self.interpolate_sdf(field, p - dx)
        ny = self.interpolate_sdf(field, p + dy) - self.interpolate_sdf(field, p - dy)
        nz = self.interpolate_sdf(field, p + dz) - self.interpolate_sdf(field, p - dz)
        return ti.math.normalize(ti.Vector([nx, ny, nz]))

    # ========================================================================
    # UNIFIED REWARD
    # ========================================================================
    #
    # Reward formulation (single source of truth):
    #
    #   good_cuts(x,y,z) = sum over voxels of:
    #       [sigmoid(k*stock_before) - sigmoid(k*cut_dist(tool_pos))] * sigmoid(-k*target)
    #   bad_cuts = same but with sigmoid(k*target) (inside-target mask)
    #
    #   progress = sum of [sigmoid(k*stock_before) * sigmoid(-k*target)]
    #            - sum of [sigmoid(k*cut_dist) * sigmoid(-k*target)]
    #
    #   boundary = near_target_surface(tool_tip) * stock_presence(tool_tip)
    #              * cutting_mask(total_removed)   [scalar, evaluated at tool tip]
    #
    #   total_removed_normalized = good_cuts + bad_cuts   (grid-normalized sums,
    #                                                      already in [0, ~1])
    #   idle = -0.2 + 0.2 * sigmoid(k_idle * (total_removed_norm - IDLE_THRESHOLD))
    #
    #   holder = - sigmoid(k * (-stock_at_holder))
    #          = - sigmoid(-k * stock_at_holder)   [soft, differentiable version of
    #                                               "holder inside stock"]
    #
    #   reward = W_GOOD*good - W_BAD*bad + W_BOUND*boundary + W_PROG*progress
    #          + W_IDLE*idle + W_HOLDER*holder
    #
    # ========================================================================

    @ti.kernel
    def _accumulate_cuts(self):
        for i, j, kk in ti.ndrange(self.res, self.res, self.res):
            k = REWARD_K_SIGMOID
            norm = 1.0 / self.grid_norm

            tool_pos = self.tool_pos[None]
            tool_radius = self.tool_radius[None]
            tool_height = self.tool_height[None]

            px = i * self.dx
            py = j * self.dx
            pz = kk * self.dx

            dx_t = px - tool_pos.x
            dy_t = py - tool_pos.y
            d_h = ti.sqrt(dx_t * dx_t + dy_t * dy_t + 1e-12) - tool_radius
            d_z_bottom = tool_pos.z - pz
            d_z_top = pz - (tool_pos.z + tool_height)
            d_z = ti.max(d_z_bottom, d_z_top)
            tool_dist = ti.max(d_h, d_z)

            stock_before = self.sdf_stock_before[i, j, kk]
            target       = self.sdf_target[i, j, kk]
            cut_dist     = ti.max(stock_before, -tool_dist)

            inside_stock_before = 1.0 / (1.0 + ti.exp( k * stock_before))
            inside_stock_after  = 1.0 / (1.0 + ti.exp( k * cut_dist))
            inside_target       = 1.0 / (1.0 + ti.exp( k * target))
            outside_target      = 1.0 / (1.0 + ti.exp(-k * target))

            removed = inside_stock_before - inside_stock_after

            # CHANGED: += instead of ti.atomic_add
            self.loss_good_cuts[None] += removed * outside_target * norm
            self.loss_bad_cuts[None]  += removed * inside_target  * norm

    @ti.kernel
    def _accumulate_progress(self):
        """Progress is just the change in "progress" from pre- to post-cut, so we
        can compute it as a separate kernel that reads the already-updated
        sdf_stock. This is cheaper than recomputing the post-cut SDF analytically
        as in _accumulate_reward_components, and it also lets us use the true
        post-cut SDF for better reward shaping (instead of the analytic max)."""

        for i, j, kk in ti.ndrange(self.res, self.res, self.res):
            k = REWARD_K_SIGMOID
            norm = 1.0 / self.grid_norm

            p = ti.Vector([i, j, kk]) * self.dx
            stock_before = self.sdf_stock_before[i, j, kk]
            stock_after  = self.sdf_stock[i, j, kk]
            target       = self.sdf_target[i, j, kk]

            inside_target       = 1.0 / (1.0 + ti.exp( k * target))
            outside_target      = 1.0 / (1.0 + ti.exp(-k * target))

            ti.atomic_add(self.loss_progress[None],
                (inside_target * norm) * (stock_before - stock_after))

    @ti.kernel
    def _accumulate_boundary(self):
        for _ in ti.ndrange(1):
            k = REWARD_K_SIGMOID
            tool_pos = self.tool_pos[None]
            tool_height = self.tool_height[None]

            target_at_tool = self.interpolate_sdf(self.sdf_target, tool_pos)

            stock_before_at_tool = self.interpolate_sdf(self.sdf_stock_before, tool_pos)
            tool_dist_at_tool = self.dist_from_tool(tool_pos)
            stock_after_at_tool = ti.max(stock_before_at_tool, -tool_dist_at_tool)

            near_target_surface = ti.exp(-BOUNDARY_SIGMA * target_at_tool * target_at_tool)
            stock_presence = 1.0 / (1.0 + ti.exp(k * (stock_after_at_tool - BOUNDARY_STOCK_OFFSET)))

            total_removed = self.loss_good_cuts[None] + self.loss_bad_cuts[None]
            cutting_mask = 1.0 / (1.0 + ti.exp(-REWARD_K_IDLE * (total_removed - IDLE_THRESHOLD)))

            boundary = near_target_surface * stock_presence * cutting_mask
            ti.atomic_add(self.loss_boundary[None], boundary)

    @ti.kernel
    def _accumulate_holder(self):
        for _ in ti.ndrange(1):
            k = REWARD_K_SIGMOID
            tool_pos = self.tool_pos[None]
            tool_height = self.tool_height[None]

            holder_base = ti.Vector([tool_pos.x, tool_pos.y, tool_pos.z + tool_height])
            stock_at_holder = self.interpolate_sdf(self.sdf_stock_before, holder_base)

            holder_inside = 1.0 / (1.0 + ti.exp(k * stock_at_holder))
            ti.atomic_add(self.loss_holder[None], -holder_inside)

    @ti.kernel
    def _accumulate_idle(self):
        """Idle penalty as a function of already-accumulated good+bad cuts.
        Runs after _accumulate_reward_components. Writes ONLY loss_idle."""

        for _ in ti.ndrange(1):
            total_removed = self.loss_good_cuts[None] + self.loss_bad_cuts[None]
            idle = -0.2 + 0.2 * (1.0 / (1.0 + ti.exp(-REWARD_K_IDLE * (total_removed - IDLE_THRESHOLD))))
            ti.atomic_add(self.loss_idle[None], idle)

    @ti.kernel
    def _accumulate_total(self):
        """Combines components into full reward. Runs last.
        Writes ONLY loss[None]."""
        for _ in range(1): # did not with with ti.ndrange
            ti.atomic_add(self.loss[None],
                REWARD_W_GOOD   * self.loss_good_cuts[None]
                - REWARD_W_BAD    * self.loss_bad_cuts[None]
                + REWARD_W_BOUND  * self.loss_boundary[None]
                + REWARD_W_PROG   * self.loss_progress[None]
                + REWARD_W_IDLE   * self.loss_idle[None]
                + REWARD_W_HOLDER * self.loss_holder[None])

    def clear_losses(self):
        self.loss[None] = 0.0
        self.loss_good_cuts[None] = 0.0
        self.loss_bad_cuts[None] = 0.0
        self.loss_boundary[None] = 0.0
        self.loss_progress[None] = 0.0
        self.loss_idle[None] = 0.0
        self.loss_holder[None] = 0.0

    def _forward_reward(self):
        """Full forward pass producing all loss fields.
        Order matters: idle reads loss_good_cuts/loss_bad_cuts; boundary does too."""
        self._accumulate_cuts()
        self._accumulate_progress()
        self._accumulate_boundary()
        self._accumulate_holder()
        self._accumulate_idle()
        self._accumulate_total()

    def compute_reward_and_gradients(self) -> dict:
        """
        Runs autodiff through the unified reward forward pass.
        MUST be called AFTER snapshot_stock() so sdf_stock_before holds the
        pre-cut snapshot. Safe to call before or after apply_cut since the
        forward pass reads only sdf_stock_before (not sdf_stock).

        Returns a dict with:
            "grad":       np.ndarray shape (3,) = d(full_reward)/d(tool_pos)
            "reward":     float, full reward value
            "good_cuts":  float
            "bad_cuts":   float
            "boundary":   float
            "progress":   float
            "idle":       float
            "holder":     float
        """
        self.clear_losses()
        self.tool_pos.grad[None] = ti.Vector([0.0, 0.0, 0.0])

        with ti.ad.Tape(loss=self.loss):
            self._forward_reward()

        return {
            "grad":      self.tool_pos.grad[None].to_numpy().copy(),
            "reward":    float(self.loss[None]),
            "good_cuts": float(self.loss_good_cuts[None]),
            "bad_cuts":  float(self.loss_bad_cuts[None]),
            "boundary":  float(self.loss_boundary[None]),
            "progress":  float(self.loss_progress[None]),
            "idle":      float(self.loss_idle[None]),
            "holder":    float(self.loss_holder[None]),
        }
    
    def compute_reward(self) -> dict:
        """Forward-only (no gradients). Much cheaper than compute_reward_and_gradients.
        Use this when you need the reward value but not its gradient."""
        self.clear_losses()
        self._forward_reward()
        return {
            "reward":        float(self.loss[None]),
            "good_cuts":     float(self.loss_good_cuts[None]),
            "bad_cuts":      float(self.loss_bad_cuts[None]),
            "boundary":      float(self.loss_boundary[None]),
            "progress":      float(self.loss_progress[None]),
            "idle":          float(self.loss_idle[None]),
            "holder":        float(self.loss_holder[None]),
        }

    # ========================================================================
    # Visualization kernels (unchanged from prior version)
    # ========================================================================

    @ti.kernel
    def generate_stock_visualization_mesh(self):
        self.stock_count[None] = 0
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            val = self.sdf_stock[i, j, k]
            if ti.abs(val) < self.dx * 1.5:
                idx = ti.atomic_add(self.stock_count[None], 1)
                if idx < self.stock_points.shape[0]:
                    self.stock_points[idx] = ti.Vector([i, j, k]) * self.dx

    @ti.kernel
    def generate_target_visualization_mesh(self):
        self.target_count[None] = 0
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            val = self.sdf_target[i, j, k]
            if ti.abs(val) < self.dx * 1.5:
                idx = ti.atomic_add(self.target_count[None], 1)
                if idx < self.target_points.shape[0]:
                    self.target_points[idx] = ti.Vector([i, j, k]) * self.dx

    @ti.kernel
    def init_tool_template(self):
        radius = self.tool_radius[None]
        height = self.tool_height[None]
        visual_r = radius - 0.005

        self.tool_count[None] = 0
        r_steps = int(ti.ceil(visual_r / self.dx))
        h_steps = int(ti.ceil(height / self.dx))

        for i, j, k in ti.ndrange((-r_steps, r_steps + 1), (-r_steps, r_steps + 1), (0, h_steps + 1)):
            x = i * self.dx; y = j * self.dx; z = k * self.dx
            if x * x + y * y <= visual_r * visual_r and z <= height:
                idx = ti.atomic_add(self.tool_count[None], 1)
                if idx < 100000:
                    self.tool_template[idx] = ti.Vector([x, y, z])

        self.holder_count[None] = 0
        holder_r = visual_r * 2.0
        holder_h = height * 0.5
        hr_steps = int(ti.ceil(holder_r / self.dx))
        hh_steps = int(ti.ceil(holder_h / self.dx))

        for i, j, k in ti.ndrange((-hr_steps, hr_steps + 1), (-hr_steps, hr_steps + 1), (0, hh_steps + 1)):
            x = i * self.dx; y = j * self.dx; z = k * self.dx
            if x * x + y * y <= holder_r * holder_r and z <= holder_h:
                idx = ti.atomic_add(self.holder_count[None], 1)
                if idx < 100000:
                    self.holder_template[idx] = ti.Vector([x, y, z])

    @ti.kernel
    def update_tool(self, tool_pos: ti.types.vector(3, ti.f32)):
        for i in self.tool_points:
            self.tool_points[i] = self.tool_template[i] + tool_pos
        holder_offset = ti.Vector([0.0, 0.0, self.tool_height[None]])
        for i in self.holder_points:
            self.holder_points[i] = self.holder_template[i] + tool_pos + holder_offset

    @ti.kernel
    def generate_slices(self):
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        z_idx = int(tool_pos.z / self.dx); z_idx = ti.max(0, ti.min(z_idx, self.res - 1))
        for i, j in ti.ndrange(self.res, self.res):
            val = self.sdf_stock[i, j, z_idx]
            color = ti.Vector([0.0, 0.0, 0.0])
            grid_check = ((i // 4) + (j // 4)) % 2
            if val < 0:
                color = ti.Vector([0.3, 0.3, 0.9])
            else:
                bg_col = 0.8 if grid_check == 0 else 0.7
                color = ti.Vector([bg_col, bg_col, bg_col])
            p = ti.Vector([i, j, z_idx]) * self.dx
            dist_to_center = (ti.Vector([p.x, p.y]) - ti.Vector([tool_pos.x, tool_pos.y])).norm()
            if abs(dist_to_center - tool_radius) < self.dx:
                color = ti.Vector([0.0, 1.0, 0.0])
            self.slice_xy[i, j] = color

        y_idx = int(tool_pos.y / self.dx); y_idx = ti.max(0, ti.min(y_idx, self.res - 1))
        for i, k in ti.ndrange(self.res, self.res):
            val = self.sdf_stock[i, y_idx, k]
            color = ti.Vector([0.3, 0.3, 0.9]) if val < 0 else ti.Vector([0.8, 0.8, 0.8])
            p = ti.Vector([i, y_idx, k]) * self.dx
            if abs(abs(p.x - tool_pos.x) - tool_radius) < self.dx:
                color = ti.Vector([0.0, 1.0, 0.0])
            self.slice_xz[i, k] = color

        x_idx = int(tool_pos.x / self.dx); x_idx = ti.max(0, ti.min(x_idx, self.res - 1))
        for j, k in ti.ndrange(self.res, self.res):
            val = self.sdf_stock[x_idx, j, k]
            color = ti.Vector([0.3, 0.3, 0.9]) if val < 0 else ti.Vector([0.8, 0.8, 0.8])
            p = ti.Vector([x_idx, j, k]) * self.dx
            if abs(abs(p.y - tool_pos.y) - tool_radius) < self.dx:
                color = ti.Vector([0.0, 1.0, 0.0])
            self.slice_yz[j, k] = color

    @ti.kernel
    def compose_debug_view(self):
        for i, j in ti.ndrange(341, 2 * self.res):
            self.debug_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])
        for i, j in ti.ndrange(self.res, self.res):
            self.debug_buffer[i, j + self.res] = self.slice_xy[i, j]
            self.debug_buffer[i + self.res, j + self.res] = self.slice_xz[i, j]
            self.debug_buffer[i, j] = self.slice_yz[i, j]

    @ti.kernel
    def render_raymarch(
        self,
        cam_pos: ti.types.vector(3, ti.f32),
        cam_up: ti.types.vector(3, ti.f32),
        cam_dir: ti.types.vector(3, ti.f32),
        show_stock: ti.i32,
        show_part: ti.i32,
        show_tool: ti.i32,
    ):
        cam_right = cam_dir.cross(cam_up).normalized()
        cam_up_actual = cam_right.cross(cam_dir).normalized()
        fov_scale = ti.tan(3.14159 / 4.0)
        width = self.raymarch_buffer.shape[0]
        height = self.raymarch_buffer.shape[1]
        aspect_ratio = float(width) / float(height)

        for i, j in self.raymarch_buffer:
            u = (2.0 * (i + 0.5) / float(width) - 1.0) * aspect_ratio * fov_scale
            v = (2.0 * (j + 0.5) / float(height) - 1.0) * fov_scale
            ray_dir = (cam_dir + cam_right * u + cam_up_actual * v).normalized()
            t = 0.0
            max_t = 10.0
            max_steps = 150
            color = ti.Vector([0.1, 0.1, 0.1])

            for step in range(max_steps):
                p = cam_pos + ray_dir * t
                d_stock = 1e6; d_target = 1e6; d_tool = 1e6

                if (0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 and 0.0 <= p.z <= 1.0):
                    if show_stock == 1:
                        d_stock = self.interpolate_sdf(self.sdf_stock, p)
                    if show_part == 1:
                        d_target = self.interpolate_sdf(self.sdf_target, p)
                else:
                    d_box = p - ti.Vector([0.5, 0.5, 0.5])
                    d_aabb = ti.max(ti.abs(d_box.x), ti.max(ti.abs(d_box.y), ti.abs(d_box.z))) - 0.5
                    if show_stock == 1:
                        d_stock = ti.max(d_aabb, 2e-3)
                    if show_part == 1:
                        d_target = ti.max(d_aabb, 2e-3)

                if show_tool == 1:
                    d_tool = ti.min(self.dist_from_tool(p), self.dist_from_holder(p))

                d = ti.min(d_stock, ti.min(d_target, d_tool))

                if d < 1e-3:
                    norm = ti.Vector([0.0, 0.0, 1.0])
                    mat_color = ti.Vector([0.8, 0.8, 0.8])
                    if d == d_tool:
                        mat_color = (ti.Vector([1.0, 0.2, 0.2])
                            if self.dist_from_tool(p) < self.dist_from_holder(p)
                            else ti.Vector([0.2, 0.2, 0.2]))
                        eps = 1e-3
                        dx = ti.Vector([eps, 0.0, 0.0])
                        dy = ti.Vector([0.0, eps, 0.0])
                        dz = ti.Vector([0.0, 0.0, eps])
                        if self.dist_from_tool(p) < self.dist_from_holder(p):
                            nx = self.dist_from_tool(p + dx) - self.dist_from_tool(p - dx)
                            ny = self.dist_from_tool(p + dy) - self.dist_from_tool(p - dy)
                            nz = self.dist_from_tool(p + dz) - self.dist_from_tool(p - dz)
                            norm = ti.math.normalize(ti.Vector([nx, ny, nz]))
                        else:
                            nx = self.dist_from_holder(p + dx) - self.dist_from_holder(p - dx)
                            ny = self.dist_from_holder(p + dy) - self.dist_from_holder(p - dy)
                            nz = self.dist_from_holder(p + dz) - self.dist_from_holder(p - dz)
                            norm = ti.math.normalize(ti.Vector([nx, ny, nz]))
                    elif d == d_stock:
                        mat_color = ti.Vector([0.2, 0.8, 0.2])
                        norm = self.compute_surface_normal(self.sdf_stock, p)
                        grid_p = p * float(self.res)
                        cx = int(grid_p.x) % 2; cy = int(grid_p.y) % 2; cz = int(grid_p.z) % 2
                        if (cx + cy + cz) % 2 == 0:
                            mat_color = mat_color * 0.8
                    elif d == d_target:
                        mat_color = ti.Vector([0.5, 0.5, 1.0])
                        norm = self.compute_surface_normal(self.sdf_target, p)

                    light_dir = ti.Vector([1.0, 1.0, 1.0]).normalized()
                    diffuse = ti.max(0.0, norm.dot(light_dir))
                    ambient = 0.2
                    color = mat_color * (diffuse * 0.8 + ambient)
                    break

                t += d
                if t > max_t:
                    break
            self.raymarch_buffer[i, j] = color


def main():
    sim = CNCSimulator(resolution=64)
    print("Simulator initialized!")


if __name__ == "__main__":
    main()