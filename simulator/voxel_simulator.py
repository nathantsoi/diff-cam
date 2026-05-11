import os
import random

import taichi as ti

from simulator.simulator_utils import *


_TI_INIT_PID = None

def ensure_taichi_initialized(arch="gpu", debug=False):
    global _TI_INIT_PID
    if _TI_INIT_PID == os.getpid():
        return
    if arch == "gpu":
        try:
            ti.init(arch=ti.gpu, debug=debug)
        except Exception:
            ti.init(arch=ti.cpu, debug=debug)
    else:
        ti.init(arch=ti.cpu, debug=debug)
    _TI_INIT_PID = os.getpid()


@ti.data_oriented
class CNCSimulator:

    def __init__(self, resolution=32, shape="sphere"):
        # Initialize Taichi (only on first instantiation)
        ensure_taichi_initialized()

        self.res = resolution
        self.dx = 1.0 / self.res
        self.grid_norm = float(self.res ** 3)

        # ----- Stock  -----
        self.sdf_stock = ti.field(dtype=ti.f32, shape=(self.res, self.res, self.res))
        self.sdf_stock_before = ti.field(dtype=ti.f32, shape=(self.res, self.res, self.res)) # for reward calculation

        # ----- Target  -----
        shape_options = ["box", "cylinder", "sphere", "pyramid"]
        self.sdf_target = ti.field(dtype=ti.f32, shape=(self.res, self.res, self.res))

        if shape is None:
            shape = random.choice(shape_options)
        self.shape = shape
        
        # ----- Tool  -----
        self.tool_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.tool_radius = ti.field(dtype=ti.f32, shape=())
        self.tool_height = ti.field(dtype=ti.f32, shape=())

        # ----- Reward -----
        self.reward_components = ti.field(dtype=ti.f32, shape=6)  # good, bad, prog, idle, bdry, holder

        # ----- Visualization -----
        self.initialize_visualization()

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
    def initialize_target_box(self, half_size: ti.f32):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            self.sdf_target[i, j, k] = box_sdf(p, center, half_size)

    @ti.kernel
    def initialize_target_pyramid(self, half_base: ti.f32, height: ti.f32):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            base_z = 0.5 - height * 0.5
            self.sdf_target[i, j, k] = pyramid_sdf(
                p, center.x, center.y, base_z, half_base, height
            )

    @ti.kernel
    def initialize_target_cylinder(self, radius: ti.f32, height: ti.f32):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            cz = 0.5 - height * 0.5
            self.sdf_target[i, j, k] = cylinder_sdf(
                p, center.x, center.y, cz, radius, height
            )

    @ti.kernel
    def initialize_stock(self, half_size: ti.f32):
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            self.sdf_stock[i, j, k] = box_sdf(p, center, half_size)


    def initialize_tool(self, tool_pos: list, radius: float, height: float):
        self.tool_pos[None] = ti.Vector(tool_pos)
        self.tool_radius[None] = radius
        self.tool_height[None] = height

    
    def initialize_visualization(self):
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

        self.debug_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(2 * self.res, 2 * self.res))
        self.raymarch_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(1024, 768))

    # ========================================================================
    # SDFs that aren't stored as class variables
    # ========================================================================
    @ti.func
    def tool_sdf(self, p, tool_pos):
         # cylindrical tool aligned with z
        r = self.tool_radius[None]
        h = self.tool_height[None]

        d_xy = ti.Vector([p.x - tool_pos.x, p.y - tool_pos.y]).norm() - r

        d_z_bottom = tool_pos.z - p.z
        d_z_top = p.z - (tool_pos.z + h)
        d_z = ti.max(d_z_bottom, d_z_top)

        return ti.max(d_xy, d_z)

    @ti.func
    def holder_sdf(self, p):
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
        d_z_top = p.z - (holder_z_start + holder_height)
        d_z = ti.max(d_z_bottom, d_z_top)
        return ti.max(d_h, d_z)

    # ========================================================================
    # Cutting
    # ========================================================================

    @ti.kernel
    def apply_cut(self) -> ti.types.vector(2, ti.f32):
        """
        Apply one tool cut to the stock.
        Returns:
            [0] vol_removed:      world-volume of material carved out this step
            [1] target_violations: number of protected voxels removed
        """
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

        voxels_removed = 0
        target_violations = 0

        for i, j, k in ti.ndrange((min_x, max_x), (min_y, max_y), (min_z, max_z)):
            p = ti.Vector([i, j, k]) * self.dx
            tool_dist = self.tool_sdf(p, tool_pos)
            stock_dist = self.sdf_stock[i, j, k]
            target_dist = self.sdf_target[i, j, k]
            new_dist = ti.max(stock_dist, -tool_dist)

            if stock_dist < 0.0 and new_dist >= 0.0:
                ti.atomic_add(voxels_removed, 1)
                if target_dist < 0.0:
                    ti.atomic_add(target_violations, 1)

            self.sdf_stock[i, j, k] = new_dist

        vol_removed = ti.cast(voxels_removed, ti.f32) * (self.dx ** 3)
        return ti.Vector([vol_removed, ti.cast(target_violations, ti.f32)])

    @ti.kernel
    def move_tool_one_unit(self, dir: ti.types.vector(3, ti.f32)):
        """Moves the tool one voxel in a unit direction."""
        # NOTE: removed the dead `valid_dir` check from the previous version —
        # it computed a flag and never used it. If direction validation matters,
        # enforce it on the Python side before calling this kernel.
        new_pos = self.tool_pos[None]
        for i in ti.static(range(3)):
            new_pos[i] = ti.max(
                0.0, ti.min(1.0, new_pos[i] + dir[i] * self.dx)
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
            if self.holder_sdf(p) < 0 and self.sdf_stock[i, j, k] < 0:
                collision = 1
        return collision

    @ti.kernel
    def set_sdf_stock_before(self):
        for I in ti.grouped(self.sdf_stock):
            self.sdf_stock_before[I] = self.sdf_stock[I]

    @ti.kernel
    def compute_reward_components(self, k_sdf: ti.f32, boundary_sigma: ti.f32, k_idle: ti.f32, idle_thresh: ti.f32):
        good = 0.0; bad = 0.0; excess_before = 0.0; excess_after = 0.0
        inv_norm = 1.0 / self.grid_norm

        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            sdf_stock_before = self.sdf_stock_before[i, j, k]
            sdf_stock_after = self.sdf_stock[i, j, k]
            sdf_target = self.sdf_target[i, j, k]

            # numerically-safe sigmoid
            in_sb = 1.0 / (1.0 + ti.exp(ti.max(-50.0, ti.min(50.0,  k_sdf * sdf_stock_before))))
            in_sa = 1.0 / (1.0 + ti.exp(ti.max(-50.0, ti.min(50.0,  k_sdf * sdf_stock_after))))
            in_t  = 1.0 / (1.0 + ti.exp(ti.max(-50.0, ti.min(50.0,  k_sdf * sdf_target))))
            out_t = 1.0 - in_t  # sigmoid(-x) = 1 - sigmoid(x), saves an exp

            removed = in_sb - in_sa
            good += removed * out_t
            bad  += removed * in_t
            excess_before += in_sb * out_t
            excess_after  += in_sa * out_t

        # good + bad
        good *= inv_norm
        bad *= inv_norm

        # progress
        progress = (excess_before - excess_after) * inv_norm

        # idle
        total_removed = good + bad
        idle_gate = 1.0 / (1.0 + ti.exp(ti.max(-50.0, ti.min(50.0, -k_idle * (total_removed - idle_thresh)))))
        idle = 1.0 - idle_gate

        # boundary
        tp = self.tool_pos[None]
        ix = ti.max(0, ti.min(self.res - 1, int(tp.x * self.res)))
        iy = ti.max(0, ti.min(self.res - 1, int(tp.y * self.res)))
        iz = ti.max(0, ti.min(self.res - 1, int(tp.z * self.res)))
        t_at_tool = self.sdf_target[ix, iy, iz]
        s_at_tool = self.sdf_stock_before[ix, iy, iz]
        near_surf = ti.exp(-boundary_sigma * t_at_tool * t_at_tool)
        stock_pres = 1.0 / (1.0 + ti.exp(ti.max(-50.0, ti.min(50.0, k_sdf * s_at_tool))))
        boundary = near_surf * stock_pres * idle_gate

        # holder
        th = self.tool_height[None]
        z_lo = ti.max(0, ti.min(self.res - 1, int(ti.floor((tp.z + th) * self.res))))
        z_hi = ti.max(z_lo + 1, ti.min(self.res, int(ti.ceil((tp.z + 1.5 * th) * self.res))))
        holder_sum = 0.0
        for kk in range(z_lo, z_hi):
            v = self.sdf_stock[ix, iy, kk]
            holder_sum += 1.0 / (1.0 + ti.exp(ti.max(-50.0, ti.min(50.0, k_sdf * v))))
        holder = holder_sum / ti.cast(z_hi - z_lo, ti.f32)

        self.reward_components[0] = good
        self.reward_components[1] = bad
        self.reward_components[2] = progress
        self.reward_components[3] = idle
        self.reward_components[4] = boundary
        self.reward_components[5] = holder

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
    def update_tool(self):
        """Refresh tool_points and holder_points to reflect the current tool_pos.

        Reads tool_pos / tool_height from their 0-D fields. The env should call
        this with no arguments after any change to tool position, radius, or
        height (radius/height changes also require re-running init_tool_template).
        """
        tp = self.tool_pos[None]
        th = self.tool_height[None]
        holder_offset = ti.Vector([0.0, 0.0, th])

        n_tool = self.tool_count[None]
        for i in range(n_tool):
            self.tool_points[i] = self.tool_template[i] + tp

        n_holder = self.holder_count[None]
        for i in range(n_holder):
            self.holder_points[i] = self.holder_template[i] + tp + holder_offset

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
        for i, j in ti.ndrange(2 * self.res, 2 * self.res):
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

                if (0.0 <= p.x and p.x <= 1.0 and 0.0 <= p.y and p.y <= 1.0 and 0.0 <= p.z and p.z <= 1.0):
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
                    d_tool = ti.min(self.tool_sdf(p, self.tool_pos[None]), self.holder_sdf(p))

                d = ti.min(d_stock, ti.min(d_target, d_tool))

                if d < 1e-3:
                    norm = ti.Vector([0.0, 0.0, 1.0])
                    mat_color = ti.Vector([0.8, 0.8, 0.8])
                    if d == d_tool:
                        mat_color = (ti.Vector([1.0, 0.2, 0.2])
                            if self.tool_sdf(p, self.tool_pos[None]) < self.holder_sdf(p)
                            else ti.Vector([0.2, 0.2, 0.2]))
                        eps = 1e-3
                        dx = ti.Vector([eps, 0.0, 0.0])
                        dy = ti.Vector([0.0, eps, 0.0])
                        dz = ti.Vector([0.0, 0.0, eps])
                        if self.tool_sdf(p, self.tool_pos[None]) < self.holder_sdf(p):
                            nx = self.tool_sdf(p + dx, self.tool_pos[None]) - self.tool_sdf(p - dx, self.tool_pos[None])
                            ny = self.tool_sdf(p + dy, self.tool_pos[None]) - self.tool_sdf(p - dy, self.tool_pos[None])
                            nz = self.tool_sdf(p + dz, self.tool_pos[None]) - self.tool_sdf(p - dz, self.tool_pos[None])
                            norm = ti.math.normalize(ti.Vector([nx, ny, nz]))
                        else:
                            nx = self.holder_sdf(p + dx) - self.holder_sdf(p - dx)
                            ny = self.holder_sdf(p + dy) - self.holder_sdf(p - dy)
                            nz = self.holder_sdf(p + dz) - self.holder_sdf(p - dz)
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
    sim = CNCSimulator(resolution=32, shape="sphere")
    sim.initialize_target_sphere(radius=0.3)
    sim.initialize_stock(half_size=0.4)
    sim.initialize_tool(tool_pos=[0.5, 0.5, 0.8], radius=0.1, height=0.2)
    print("Simulator initialized!")


if __name__ == "__main__":
    main()