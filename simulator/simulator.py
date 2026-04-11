import taichi as ti

from cam_env.physics_config import (
    STOCK_HALF_SIZE,
    TARGET_RADIUS,
    TOOL_START_POS, TOOL_RADIUS, TOOL_HEIGHT,
)



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

        # Define the SDF fields for stock and target geometry
        self.sdf_stock = ti.field(dtype=ti.f32, shape=(self.res, self.res, self.res))
        self.sdf_target = ti.field(dtype=ti.f32, shape=(self.res, self.res, self.res))

        # Define the tool
        self.tool_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.tool_radius = ti.field(dtype=ti.f32, shape=())
        self.tool_height = ti.field(dtype=ti.f32, shape=())

        # Visualization fields (for GGUI)

        # Stock visualization
        self.stock_points = ti.Vector.field(
            3, dtype=ti.f32, shape=self.res**3
        )  # Full resolution size to be safe
        self.stock_count = ti.field(dtype=ti.i32, shape=())

        # Tool Visualization
        self.tool_points = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.tool_template = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.tool_count = ti.field(dtype=ti.i32, shape=())

        # Holder Visualization
        self.holder_points = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.holder_template = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.holder_count = ti.field(dtype=ti.i32, shape=())

        # Part (Target) Visualization
        self.target_points = ti.Vector.field(3, dtype=ti.f32, shape=self.res**3)
        self.target_count = ti.field(dtype=ti.i32, shape=())

        # Debug Slices
        self.slice_xy = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
        self.slice_xz = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
        self.slice_yz = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))

        # Combined debug view using 2x2 grid with padding to match window Aspect Ratio (4:3)
        # Window: 1024x768. AR = 1.333
        # Buffer Height: 2 * res = 256.
        # Buffer Width Target: 256 * 1.333 = 341.33 -> 341
        self.debug_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(341, 2 * self.res))

        # Raymarching buffer for high-quality alternative render
        self.raymarch_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(1024, 768))
        self.removed_vol_field = ti.field(dtype=ti.f32, shape=())
        self.excess_field = ti.field(dtype=ti.f32, shape=())

        # Observation buffer: [tool_pos(3), grad_stock(3), grad_diff(3), sdf_stock(res³), sdf_target(res³)]
        self.obs_size = 9 + 2 * resolution ** 3
        self.obs_buffer = ti.field(dtype=ti.f32, shape=(self.obs_size,))

    # ── Excess volume (GPU-only) ────────────────────────────────────
    def get_excess(self) -> float:
        """Compute total excess material volume entirely on the GPU.
        Returns a Python float — no array transfer involved."""
        self.excess_field[None] = 0.0
        self._compute_excess_kernel()
        return float(self.excess_field[None])

    @ti.kernel
    def _compute_excess_kernel(self):
        """sum(max(min(-sdf_stock, sdf_target), 0)) over the full grid."""
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            stock = self.sdf_stock[i, j, k]
            target = self.sdf_target[i, j, k]
            val = ti.max(ti.min(-stock, target), 0.0)
            if val > 0.0:
                ti.atomic_add(self.excess_field[None], val)

    # ── Fused observation builder (GPU-only) ────────────────────────
    def build_obs(self):
        """Build the full observation vector AND compute excess volume in one
        kernel launch.  After this call:
          - obs_buffer contains the flat observation (transfer with .to_numpy())
          - excess_field[None] holds the current excess volume (scalar read)
        """
        self.excess_field[None] = 0.0
        self._build_obs_kernel()

    @ti.kernel
    def _build_obs_kernel(self):
        # ── Serial: tool position + gradients at tool cell ──────────
        tp = self.tool_pos[None]
        self.obs_buffer[0] = tp.x
        self.obs_buffer[1] = tp.y
        self.obs_buffer[2] = tp.z

        res = self.res
        ix = ti.max(1, ti.min(res - 2, int(tp.x * res)))
        iy = ti.max(1, ti.min(res - 2, int(tp.y * res)))
        iz = ti.max(1, ti.min(res - 2, int(tp.z * res)))

        inv_2dx = float(res) * 0.5  # 1 / (2 * dx)

        # ∇φ_stock (normalized)
        gs_x = (self.sdf_stock[ix+1, iy, iz] - self.sdf_stock[ix-1, iy, iz]) * inv_2dx
        gs_y = (self.sdf_stock[ix, iy+1, iz] - self.sdf_stock[ix, iy-1, iz]) * inv_2dx
        gs_z = (self.sdf_stock[ix, iy, iz+1] - self.sdf_stock[ix, iy, iz-1]) * inv_2dx
        gs = ti.Vector([gs_x, gs_y, gs_z])
        gs_norm = gs.norm()
        if gs_norm > 1e-8:
            gs = gs / gs_norm
        self.obs_buffer[3] = gs.x
        self.obs_buffer[4] = gs.y
        self.obs_buffer[5] = gs.z

        # ∇(φ_target − φ_stock) — direction toward excess material
        gd_x = ((self.sdf_target[ix+1, iy, iz] - self.sdf_stock[ix+1, iy, iz])
               - (self.sdf_target[ix-1, iy, iz] - self.sdf_stock[ix-1, iy, iz])) * inv_2dx
        gd_y = ((self.sdf_target[ix, iy+1, iz] - self.sdf_stock[ix, iy+1, iz])
               - (self.sdf_target[ix, iy-1, iz] - self.sdf_stock[ix, iy-1, iz])) * inv_2dx
        gd_z = ((self.sdf_target[ix, iy, iz+1] - self.sdf_stock[ix, iy, iz+1])
               - (self.sdf_target[ix, iy, iz-1] - self.sdf_stock[ix, iy, iz-1])) * inv_2dx
        self.obs_buffer[6] = gd_x
        self.obs_buffer[7] = gd_y
        self.obs_buffer[8] = gd_z

        # ── Parallel: clipped SDF copy + excess accumulation ───────
        for i, j, k in ti.ndrange(res, res, res):
            idx = i * res * res + j * res + k
            stock_val = self.sdf_stock[i, j, k]
            target_val = self.sdf_target[i, j, k]

            # Clipped SDF values → obs_buffer
            self.obs_buffer[9 + idx] = ti.max(-1.0, ti.min(1.0, stock_val))
            self.obs_buffer[9 + res * res * res + idx] = ti.max(-1.0, ti.min(1.0, target_val))

            # Excess accumulation (free — already visiting every voxel)
            excess_val = ti.max(ti.min(-stock_val, target_val), 0.0)
            if excess_val > 0.0:
                ti.atomic_add(self.excess_field[None], excess_val)

    # Initialization Kernels
    @ti.kernel
    def initialize_stock_primitive(self):
        """Initializes stock as a solid block (SDF < 0 inside)"""
        hs = float(STOCK_HALF_SIZE)
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            d = ti.abs(p - center) - hs
            dist = ti.max(d.x, ti.max(d.y, d.z))
            self.sdf_stock[i, j, k] = dist


    @ti.kernel
    def initialize_target_sphere_primitive(self, radius: ti.f32):
        """Initializes target as a smaller sphere/box for visualization"""
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            # Sphere target
            d = (p - center).norm() - radius
            self.sdf_target[i, j, k] = d

    @ti.kernel
    def initialize_target_cube(self, half_size: ti.f32):
        """
        Initializes target as a cube.
        SDF for a box: max(|x - cx| - s, |y - cy| - s, |z - cz| - s)
        Same formula your stock uses, just with variable center and size.
        """
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            d = ti.abs(p - center) - half_size
            dist = ti.max(d.x, ti.max(d.y, d.z))
            self.sdf_target[i, j, k] = dist


    @ti.kernel
    def initialize_target_cylinder(self, radius: ti.f32, half_height: ti.f32):
        """
        Initializes target as a cylinder aligned with the Z axis.
        The cylinder is centered at (cx, cy, cz) with given radius and half_height.
        SDF = max(horizontal_dist - radius, |z - cz| - half_height)
        """
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            cx, cy, cz = 0.5, 0.5, 0.5

            p = ti.Vector([i, j, k]) * self.dx
            d_horiz = ti.Vector([p.x - cx, p.y - cy]).norm() - radius
            d_vert = ti.abs(p.z - cz) - half_height
            self.sdf_target[i, j, k] = ti.max(d_horiz, d_vert)


    @ti.kernel
    def initialize_target_pyramid(self, half_base: ti.f32, height: ti.f32):
        """
        Initializes target as a square pyramid.
        Base is centered at (cx, cy, base_z) with half-width half_base.
        Apex is at (cx, cy, base_z + height).
 
        The pyramid is the intersection of:
        - A bottom plane (z >= base_z)
        - Four sloping planes that taper from the base to the apex
 
        Each sloping plane says: at height z, the allowed half-width is
        half_base * (1 - (z - base_z) / height). The SDF at each point
        is the max of all plane distances.
        """
        cx, cy, base_z = 0.5, 0.5, 0.2

        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
 
            # How far up the pyramid are we (0 at base, 1 at apex)
            t = (p.z - base_z) / height
 
            # Below the base
            d_bottom = base_z - p.z
 
            # Above the apex
            d_top = p.z - (base_z + height)
 
            # At height z, the allowed half-width shrinks linearly
            # from half_base at the bottom to 0 at the apex
            allowed = half_base * (1.0 - t)
 
            # Box SDF in XY at current height
            dx = ti.abs(p.x - cx) - allowed
            dy = ti.abs(p.y - cy) - allowed
            d_sides = ti.max(dx, dy)
 
            # Final SDF: max of all constraints
            dist = ti.max(d_bottom, ti.max(d_top, d_sides))
            self.sdf_target[i, j, k] = dist

    def initialize_tool_primitive(self, pos, radius=0.1, height=0.3):
        """Initializes the tool position, tool radius, and tool height"""

        self.tool_pos[None] = ti.Vector(pos)
        self.tool_radius[None] = radius
        self.tool_height[None] = height

    # Methods to generate tool and holder sdfs
    @ti.func
    def tool_sdf(self, p):
        """Analytic SDF for a cylindrical cutter (aligned with Z-axis)"""
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        d_h = ti.Vector([p.x - tool_pos.x, p.y - tool_pos.y]).norm() - tool_radius

        # Finite height cylinder
        # bottom = tool_pos.z
        # top = tool_pos.z + tool_height
        d_z_bottom = tool_pos.z - p.z
        d_z_top = p.z - (tool_pos.z + tool_height)
        d_z = ti.max(d_z_bottom, d_z_top)

        return ti.max(d_h, d_z)

    @ti.func
    def holder_sdf(self, p):
        """Analytic SDF for the holder cylinder (based on tool position)"""
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        holder_radius = tool_radius * 2.0
        holder_height = tool_height * 0.5
        holder_z_start = tool_pos.z + tool_height

        # Horizontal distance from holder axis
        d_h = ti.Vector([p.x - tool_pos.x, p.y - tool_pos.y]).norm() - holder_radius

        # Vertical distance (finite height cylinder)
        d_z_bottom = holder_z_start - p.z
        d_z_top = p.z - (holder_z_start + holder_height)
        d_z = ti.max(d_z_bottom, d_z_top)

        return ti.max(d_h, d_z)

    # Other kernel utilities
    def apply_cut(self) -> float:
        """
        Boolean Subtraction: Stock = max(Stock, -Tool)
        Returns the approximate volume removed (useful for force calculation)
        """
        self.removed_vol_field[None] = 0.0
        self._apply_cut_kernel()
        return float(self.removed_vol_field[None])

    @ti.kernel
    def _apply_cut_kernel(self):
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        # Optimization: Only iterate over the bounding box of the tool
        # Convert tool position and radius to index space

        # Calculate bounds in integer coordinates
        # X and Y are bounded by tool radius
        min_x = int(ti.floor((tool_pos.x - tool_radius) / self.dx - 4.0))
        max_x = int(ti.ceil((tool_pos.x + tool_radius) / self.dx + 4.0))

        min_y = int(ti.floor((tool_pos.y - tool_radius) / self.dx - 4.0))
        max_y = int(ti.ceil((tool_pos.y + tool_radius) / self.dx + 4.0))

        # Z is bounded at bottom by tool tip, and top by tool tip + height
        min_z = int(ti.floor((tool_pos.z) / self.dx - 4.0))
        max_z = int(ti.ceil((tool_pos.z + tool_height) / self.dx + 4.0))

        # Clamp to domain size
        min_x = ti.max(min_x, 0)
        max_x = ti.min(max_x, self.res)
        min_y = ti.max(min_y, 0)
        max_y = ti.min(max_y, self.res)
        min_z = ti.max(min_z, 0)
        max_z = ti.min(max_z, self.res)

        # Iterate over the bounding box
        for i, j, k in ti.ndrange((min_x, max_x), (min_y, max_y), (min_z, max_z)):
            p = ti.Vector([i, j, k]) * self.dx

            tool_dist = self.tool_sdf(p)
            stock_dist = self.sdf_stock[i, j, k]

            # The Cut Logic:
            # We want the intersection of the Stock and the NOT(Tool)
            # In SDF math: max(A, -B)
            new_dist = ti.max(stock_dist, -tool_dist)

            # Safety mask: prevent cutting inside the target.
            # Where the target is solid (target_dist < 0), the stock must remain
            # at least as solid — clamp new_dist so it can't exceed target_dist.
            target_dist = self.sdf_target[i, j, k]
            if target_dist < 0.0:
                new_dist = ti.min(new_dist, target_dist)

            if new_dist != stock_dist:
                # If values changed, we removed material
                ti.atomic_add(self.removed_vol_field[None], 1.0)  # simplistic volume proxy
                self.sdf_stock[i, j, k] = new_dist


    @ti.kernel
    def move_tool_one_unit(self, dir: ti.types.vector(3, ti.f32)):
        """Moves the tool one unit in the direction of a unit vector"""
        valid_dir = True
        for i in ti.static(range(3)):
            if not (dir[i] == -1.0 or dir[i] == 0.0 or dir[i] == 1.0):
                valid_dir = False

        if not valid_dir:
            print(
                "Error: Direction must be a unit step vector with components in {-1, 0, 1}"
            )

        new_pos = self.tool_pos[None]
        for i in ti.static(range(3)):
            new_pos[i] = ti.max(0.0, ti.min(1.0 - self.dx, new_pos[i] + dir[i] * self.dx))
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

        # Bounding box of holder
        min_x = ti.max(0, int(ti.floor((tool_pos.x - holder_radius) / self.dx) - 1))
        max_x = ti.min(
            self.res, int(ti.ceil((tool_pos.x + holder_radius) / self.dx) + 1)
        )
        min_y = ti.max(0, int(ti.floor((tool_pos.y - holder_radius) / self.dx) - 1))
        max_y = ti.min(
            self.res, int(ti.ceil((tool_pos.y + holder_radius) / self.dx) + 1)
        )
        min_z = ti.max(0, int(ti.floor(holder_z_start / self.dx) - 1))
        max_z = ti.min(self.res, int(ti.ceil(holder_z_end / self.dx) + 1))

        for i, j, k in ti.ndrange((min_x, max_x), (min_y, max_y), (min_z, max_z)):
            p = ti.Vector([i, j, k]) * self.dx

            # If point is inside holder AND inside stock → collision
            if self.holder_sdf(p) < 0 and self.sdf_stock[i, j, k] < 0:
                collision = 1

        return collision

    @ti.kernel
    def check_tool_intersects_target(self) -> ti.i32:
        """ Returns 1 if tool intersects target geometry, 0 otherwise.

        Uses sub-grid sampling (half-dx steps) with trilinear interpolation
        of the target SDF for robust detection even at low resolution.
        """
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        cuts_target = 0

        # Sample at half-dx spacing for sub-grid accuracy
        step = self.dx * 0.5

        # Number of samples in each dimension (covers tool bounding box + 1dx pad)
        n_x = int(ti.ceil((2.0 * tool_radius + 2.0 * self.dx) / step)) + 1
        n_y = int(ti.ceil((2.0 * tool_radius + 2.0 * self.dx) / step)) + 1
        n_z = int(ti.ceil((tool_height + 2.0 * self.dx) / step)) + 1

        origin_x = tool_pos.x - tool_radius - self.dx
        origin_y = tool_pos.y - tool_radius - self.dx
        origin_z = tool_pos.z - self.dx

        for i, j, k in ti.ndrange(n_x, n_y, n_z):
            p = ti.Vector([
                origin_x + i * step,
                origin_y + j * step,
                origin_z + k * step,
            ])

            # Stay within the simulation domain
            if (p.x >= 0.0 and p.x <= 1.0
                    and p.y >= 0.0 and p.y <= 1.0
                    and p.z >= 0.0 and p.z <= 1.0):
                # tool_sdf <= 0  : point is on or inside tool (includes bottom face)
                # target_sdf <= 0: point is on or inside target (interpolated)
                if self.tool_sdf(p) <= 0:
                    target_val = self.sample_sdf(self.sdf_target, p)
                    if target_val <= 0:
                        cuts_target = 1

        return cuts_target

    # Visualization Kernels
    @ti.kernel
    def generate_stock_visualization_mesh(self):
        """
        Extracts a point cloud of the surface (SDF approx 0) for GGUI.
        Real implementations might use Marching Cubes or Raymarching.
        """
        self.stock_count[None] = 0

        # Iterate over the entire domain safely
        # Sparse iterator was causing crashes, likely due to internal structural updates in apply_cut
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            val = self.sdf_stock[i, j, k]
            # If close to surface
            if ti.abs(val) < self.dx * 1.5:
                idx = ti.atomic_add(self.stock_count[None], 1)
                # Bounds check to prevent CUDA invalid address
                if idx < self.stock_points.shape[0]:
                    self.stock_points[idx] = ti.Vector([i, j, k]) * self.dx

    @ti.kernel
    def generate_target_visualization_mesh(self):
        """Extracts points for the target/part"""
        self.target_count[None] = 0

        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            val = self.sdf_target[i, j, k]
            if ti.abs(val) < self.dx * 1.5:
                idx = ti.atomic_add(self.target_count[None], 1)
                if idx < self.target_points.shape[0]:
                    self.target_points[idx] = ti.Vector([i, j, k]) * self.dx

    @ti.kernel
    def init_tool_template(self):
        # Grab radius and height
        radius = self.tool_radius[None]
        height = self.tool_height[None]

        # Generate a point cloud for the cylinder ONCE
        # We subtract the visualization particle radius (0.005) so the visual edge matches physics
        visual_r = radius - 0.005

        self.tool_count[None] = 0

        # Grid based generation
        r_steps = int(ti.ceil(visual_r / self.dx))
        h_steps = int(ti.ceil(height / self.dx))

        for i, j, k in ti.ndrange(
            (-r_steps, r_steps + 1), (-r_steps, r_steps + 1), (0, h_steps + 1)
        ):
            x = i * self.dx
            y = j * self.dx
            z = k * self.dx

            if x * x + y * y <= visual_r * visual_r and z <= height:
                idx = ti.atomic_add(self.tool_count[None], 1)
                if idx < 100000:
                    self.tool_template[idx] = ti.Vector([x, y, z])

        # Generate Holder Template (Wider cylinder on top)
        self.holder_count[None] = 0
        holder_r = visual_r * 2.0
        holder_h = height * 0.5
        hr_steps = int(ti.ceil(holder_r / self.dx))
        hh_steps = int(ti.ceil(holder_h / self.dx))

        for i, j, k in ti.ndrange(
            (-hr_steps, hr_steps + 1), (-hr_steps, hr_steps + 1), (0, hh_steps + 1)
        ):
            x = i * self.dx
            y = j * self.dx
            z = k * self.dx

            if x * x + y * y <= holder_r * holder_r and z <= holder_h:
                idx = ti.atomic_add(self.holder_count[None], 1)
                if idx < 100000:
                    # Offset slightly up so it sits on top of the tool, but we'll handle placement in update
                    self.holder_template[idx] = ti.Vector([x, y, z])

    @ti.kernel
    def update_tool(self, tool_pos: ti.types.vector(3, ti.f32)):
        # Fast update: just add position
        for i in self.tool_points:
            self.tool_points[i] = self.tool_template[i] + tool_pos

        # Update holder position (Attach to top of tool)
        # Holder sits at tool_pos.z + tool_height (0.3)
        # We need to know tool height here or assume it's checked elsewhere.
        # For this simple viz, let's assume a fix offset or pass it in.
        holder_offset = ti.Vector([0.0, 0.0, self.tool_height[None]])
        for i in self.holder_points:
            self.holder_points[i] = self.holder_template[i] + tool_pos + holder_offset

    @ti.kernel
    def generate_slices(self):
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        # 1. XY Slice at tool Z
        z_idx = int(tool_pos.z / self.dx)
        z_idx = ti.max(0, ti.min(z_idx, self.res - 1))

        for i, j in ti.ndrange(self.res, self.res):
            # SDF Color
            val = self.sdf_stock[i, j, z_idx]
            color = ti.Vector([0.0, 0.0, 0.0])

            # Helper for checkerboard pattern to see grid
            grid_check = ((i // 4) + (j // 4)) % 2

            if val < 0:  # Inside stock
                color = ti.Vector([0.3, 0.3, 0.9])
            else:  # Outside
                bg_col = 0.8 if grid_check == 0 else 0.7
                color = ti.Vector([bg_col, bg_col, bg_col])

            # Tool Overlay
            p = ti.Vector([i, j, z_idx]) * self.dx
            dist_to_center = (
                ti.Vector([p.x, p.y]) - ti.Vector([tool_pos.x, tool_pos.y])
            ).norm()

            # Green outline for tool radius
            if abs(dist_to_center - tool_radius) < self.dx:
                color = ti.Vector([0.0, 1.0, 0.0])

            self.slice_xy[i, j] = color

        # 2. XZ Slice at tool Y
        y_idx = int(tool_pos.y / self.dx)
        y_idx = ti.max(0, ti.min(y_idx, self.res - 1))

        for i, k in ti.ndrange(self.res, self.res):
            val = self.sdf_stock[i, y_idx, k]
            color = ti.Vector([0.0, 0.0, 0.0])
            if val < 0:
                color = ti.Vector([0.3, 0.3, 0.9])
            else:
                color = ti.Vector([0.8, 0.8, 0.8])

            p = ti.Vector([i, y_idx, k]) * self.dx
            dist_x = abs(p.x - tool_pos.x)
            if abs(dist_x - tool_radius) < self.dx:
                color = ti.Vector([0.0, 1.0, 0.0])

            self.slice_xz[i, k] = color

        # 3. YZ Slice at tool X
        x_idx = int(tool_pos.x / self.dx)
        x_idx = ti.max(0, ti.min(x_idx, self.res - 1))

        for j, k in ti.ndrange(self.res, self.res):
            val = self.sdf_stock[x_idx, j, k]
            color = ti.Vector([0.0, 0.0, 0.0])
            if val < 0:
                color = ti.Vector([0.3, 0.3, 0.9])
            else:
                color = ti.Vector([0.8, 0.8, 0.8])

            p = ti.Vector([x_idx, j, k]) * self.dx
            dist_y = abs(p.y - tool_pos.y)
            if abs(dist_y - tool_radius) < self.dx:
                color = ti.Vector([0.0, 1.0, 0.0])

            self.slice_yz[j, k] = color

    @ti.kernel
    def compose_debug_view(self):
        # Clear buffer (black)
        for i, j in ti.ndrange(341, 2 * self.res):
            self.debug_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])

        # Stitch slices into 2x2 grid
        # Top-Left: XY
        # Top-Right: XZ
        # Bottom-Left: YZ
        # Botom-Right: Legend area (empty)

        for i, j in ti.ndrange(self.res, self.res):
            # 1. XY at (0, res) to (res, 2*res) -- Top Left?
            # Canvas coords: (0,0) is bottom-left usually in Taichi?
            # Let's verify. standard graphical convention is usually bottom-left origin.

            # Place XY at Top-Left: x=[0, res], y=[res, 2*res]
            self.debug_buffer[i, j + self.res] = self.slice_xy[i, j]

            # Place XZ at Top-Right: x=[res, 2*res], y=[res, 2*res]
            self.debug_buffer[i + self.res, j + self.res] = self.slice_xz[i, j]

            # Place YZ at Bottom-Left: x=[0, res], y=[0, res]
            self.debug_buffer[i, j] = self.slice_yz[i, j]


    @ti.func
    def sample_sdf(self, field: ti.template(), p: ti.template()) -> ti.f32:
        p_grid = p * self.res
        x = p_grid.x
        y = p_grid.y
        z = p_grid.z
        
        x0 = int(ti.floor(x))
        y0 = int(ti.floor(y))
        z0 = int(ti.floor(z))
        
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1
        
        tx = x - x0
        ty = y - y0
        tz = z - z0
        
        x0 = ti.max(0, ti.min(self.res - 1, x0))
        y0 = ti.max(0, ti.min(self.res - 1, y0))
        z0 = ti.max(0, ti.min(self.res - 1, z0))
        x1 = ti.max(0, ti.min(self.res - 1, x1))
        y1 = ti.max(0, ti.min(self.res - 1, y1))
        z1 = ti.max(0, ti.min(self.res - 1, z1))
        
        c000 = field[x0, y0, z0]
        c100 = field[x1, y0, z0]
        c010 = field[x0, y1, z0]
        c110 = field[x1, y1, z0]
        c001 = field[x0, y0, z1]
        c101 = field[x1, y0, z1]
        c011 = field[x0, y1, z1]
        c111 = field[x1, y1, z1]
        
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        
        c = c0 * (1 - tz) + c1 * tz
        return c

    @ti.func
    def normal_sdf(self, field: ti.template(), p: ti.template()) -> ti.math.vec3:
        eps = self.dx * 1.5
        dx = ti.Vector([eps, 0.0, 0.0])
        dy = ti.Vector([0.0, eps, 0.0])
        dz = ti.Vector([0.0, 0.0, eps])
        
        nx = self.sample_sdf(field, p + dx) - self.sample_sdf(field, p - dx)
        ny = self.sample_sdf(field, p + dy) - self.sample_sdf(field, p - dy)
        nz = self.sample_sdf(field, p + dz) - self.sample_sdf(field, p - dz)
        
        return ti.math.normalize(ti.Vector([nx, ny, nz]))

    @ti.kernel
    def render_raymarch(self, cam_pos: ti.types.vector(3, ti.f32), cam_up: ti.types.vector(3, ti.f32), cam_dir: ti.types.vector(3, ti.f32), show_stock: ti.i32, show_part: ti.i32, show_tool: ti.i32):
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
                d_stock = 1e6
                d_target = 1e6
                d_tool = 1e6
                
                if p.x >= 0.0 and p.x <= 1.0 and p.y >= 0.0 and p.y <= 1.0 and p.z >= 0.0 and p.z <= 1.0:
                    if show_stock == 1:
                        d_stock = self.sample_sdf(self.sdf_stock, p)
                    if show_part == 1:
                        d_target = self.sample_sdf(self.sdf_target, p)
                else:
                    d_box = p - ti.Vector([0.5, 0.5, 0.5])
                    d_aabb = ti.max(ti.abs(d_box.x), ti.max(ti.abs(d_box.y), ti.abs(d_box.z))) - 0.5
                    if show_stock == 1:
                        d_stock = ti.max(d_aabb, 2e-3)
                    if show_part == 1:
                        d_target = ti.max(d_aabb, 2e-3)
                
                if show_tool == 1:
                    d_tool = ti.min(self.tool_sdf(p), self.holder_sdf(p))
                
                d = ti.min(d_stock, ti.min(d_target, d_tool))
                
                if d < 1e-3:
                    norm = ti.Vector([0.0, 0.0, 1.0])
                    mat_color = ti.Vector([0.8, 0.8, 0.8])
                    
                    if d == d_tool:
                        mat_color = ti.Vector([1.0, 0.2, 0.2]) if self.tool_sdf(p) < self.holder_sdf(p) else ti.Vector([0.2, 0.2, 0.2])
                        eps = 1e-3
                        dx = ti.Vector([eps, 0.0, 0.0]); dy = ti.Vector([0.0, eps, 0.0]); dz = ti.Vector([0.0, 0.0, eps])
                        if self.tool_sdf(p) < self.holder_sdf(p):
                            nx = self.tool_sdf(p + dx) - self.tool_sdf(p - dx)
                            ny = self.tool_sdf(p + dy) - self.tool_sdf(p - dy)
                            nz = self.tool_sdf(p + dz) - self.tool_sdf(p - dz)
                            norm = ti.math.normalize(ti.Vector([nx, ny, nz]))
                        else:
                            nx = self.holder_sdf(p + dx) - self.holder_sdf(p - dx)
                            ny = self.holder_sdf(p + dy) - self.holder_sdf(p - dy)
                            nz = self.holder_sdf(p + dz) - self.holder_sdf(p - dz)
                            norm = ti.math.normalize(ti.Vector([nx, ny, nz]))
                    elif d == d_stock:
                        mat_color = ti.Vector([0.2, 0.8, 0.2])
                        norm = self.normal_sdf(self.sdf_stock, p)
                        grid_p = p * float(self.res)
                        cx = int(grid_p.x) % 2; cy = int(grid_p.y) % 2; cz = int(grid_p.z) % 2
                        if (cx + cy + cz) % 2 == 0: mat_color = mat_color * 0.8
                    elif d == d_target:
                        mat_color = ti.Vector([0.5, 0.5, 1.0])
                        norm = self.normal_sdf(self.sdf_target, p)
                    
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
    sim = CNCSimulator(resolution=128)
    print("Simulator initialized!")


if __name__ == "__main__":
    main()
