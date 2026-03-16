import taichi as ti
import numpy as np

from cam_env.physics_config import (
    STOCK_CENTER, STOCK_HALF_SIZE,
    TARGET_CENTER, TARGET_RADIUS,
    TOOL_START_POS, TOOL_RADIUS, TOOL_HEIGHT,
)

@ti.data_oriented
class BatchedCNCSimulator:
    _ti_initialized = False

    def __init__(self, num_envs: int, resolution: int = 8, step_size: float = 0.05, debug: bool = False):
        if not BatchedCNCSimulator._ti_initialized:
            if ti._lib.core.with_cuda():
                ti.init(arch=ti.gpu, debug=debug)
            else:
                ti.init(arch=ti.cpu, debug=debug)
            BatchedCNCSimulator._ti_initialized = True

        self.num_envs = num_envs
        self.res = resolution
        self.dx = 1.0 / self.res
        self.step_size = step_size

        # --- Vectorized Fields ---
        # The target geometry is identical across all environments, so it is just 3D.
        # The stock geometry, however, is modified independently in each environment, so it is 4D.
        
        # Dense storage to avoid sparse pointer hierarchy overhead for small grids (e.g. res=8,16)
        # Using dense is much faster for batched kernel dispatches.
        self.sdf_stock = ti.field(dtype=ti.f32, shape=(self.num_envs, self.res, self.res, self.res))
        self.sdf_target = ti.field(dtype=ti.f32, shape=(self.res, self.res, self.res))

        # Tool dimensions (same for all envs)
        self.tool_radius = ti.field(dtype=ti.f32, shape=())
        self.tool_height = ti.field(dtype=ti.f32, shape=())

        # Per-environment tool state
        self.tool_pos = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_envs,))
        
        # Per-environment metric fields
        self.excess_field = ti.field(dtype=ti.f32, shape=(self.num_envs,))
        self.cut_vol_field = ti.field(dtype=ti.f32, shape=(self.num_envs,))
        self.move_blocked = ti.field(dtype=ti.i32, shape=(self.num_envs,))
        self.target_violation = ti.field(dtype=ti.f32, shape=(self.num_envs,))  # fail-safe

        # Observation buffer: [tool_pos(3), grad_stock(3), grad_diff(3), sdf_stock(res³), sdf_target(res³)]
        self.obs_size = 9 + 2 * (self.res ** 3)
        self.obs_buffer = ti.field(dtype=ti.f32, shape=(self.num_envs, self.obs_size))

    # --- Initialization ---
    
    @ti.kernel
    def initialize_stock_primitive(self):
        """Initializes stock as a solid block — matches simulator.py."""
        cx = float(STOCK_CENTER[0])
        cy = float(STOCK_CENTER[1])
        cz = float(STOCK_CENTER[2])
        hs = float(STOCK_HALF_SIZE)
        for env_id, i, j, k in ti.ndrange(self.num_envs, self.res, self.res, self.res):
            p = ti.Vector([(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx])
            d = ti.abs(p - ti.Vector([cx, cy, cz])) - hs
            dist = ti.max(d.x, ti.max(d.y, d.z))
            self.sdf_stock[env_id, i, j, k] = dist

    @ti.kernel
    def initialize_target_primitive(self):
        """Initializes target as a sphere — matches simulator.py."""
        center = ti.Vector([float(TARGET_CENTER[0]), float(TARGET_CENTER[1]), float(TARGET_CENTER[2])])
        radius = float(TARGET_RADIUS)
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            pos = ti.Vector([(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx])
            dist = (pos - center).norm() - radius
            self.sdf_target[i, j, k] = dist

    @ti.kernel
    def initialize_tool_primitive(self):
        """Resets all tools to their start positions — matches simulator.py."""
        start_pos = ti.Vector([float(TOOL_START_POS[0]), float(TOOL_START_POS[1]), float(TOOL_START_POS[2])])
        for env_id in range(self.num_envs):
            self.tool_pos[env_id] = start_pos

        self.tool_radius[None] = float(TOOL_RADIUS)
        self.tool_height[None] = float(TOOL_HEIGHT)

    @ti.func
    def _tool_sdf(self, p: ti.template(), tool_pos: ti.template(), h: ti.f32, r: ti.f32) -> ti.f32:
        """SDF for a cylinder extending UPWARD from tool_pos.z to tool_pos.z + h.
        Matches the original simulator.py tool_sdf convention."""
        d_h = ti.Vector([p.x - tool_pos.x, p.y - tool_pos.y]).norm() - r
        d_z_bottom = tool_pos.z - p.z
        d_z_top = p.z - (tool_pos.z + h)
        d_z = ti.max(d_z_bottom, d_z_top)
        return ti.max(d_h, d_z)

    @ti.func
    def _sample_target_sdf(self, p: ti.template()) -> ti.f32:
        """Trilinear interpolation of the target SDF at an arbitrary point.
        Matches simulator.py's sample_sdf for sub-grid accuracy."""
        res = self.res
        gx = p.x * res - 0.5
        gy = p.y * res - 0.5
        gz = p.z * res - 0.5

        x0 = int(ti.floor(gx));  x1 = x0 + 1
        y0 = int(ti.floor(gy));  y1 = y0 + 1
        z0 = int(ti.floor(gz));  z1 = z0 + 1

        tx = gx - x0;  ty = gy - y0;  tz = gz - z0

        x0 = ti.max(0, ti.min(res - 1, x0));  x1 = ti.max(0, ti.min(res - 1, x1))
        y0 = ti.max(0, ti.min(res - 1, y0));  y1 = ti.max(0, ti.min(res - 1, y1))
        z0 = ti.max(0, ti.min(res - 1, z0));  z1 = ti.max(0, ti.min(res - 1, z1))

        c00 = self.sdf_target[x0,y0,z0]*(1-tx) + self.sdf_target[x1,y0,z0]*tx
        c10 = self.sdf_target[x0,y1,z0]*(1-tx) + self.sdf_target[x1,y1,z0]*tx
        c01 = self.sdf_target[x0,y0,z1]*(1-tx) + self.sdf_target[x1,y0,z1]*tx
        c11 = self.sdf_target[x0,y1,z1]*(1-tx) + self.sdf_target[x1,y1,z1]*tx
        c0 = c00*(1-ty) + c10*ty
        c1 = c01*(1-ty) + c11*ty
        return c0*(1-tz) + c1*tz

    @ti.func
    def _sample_stock_sdf(self, env_id: ti.i32, p: ti.template()) -> ti.f32:
        """Trilinear interpolation of the stock SDF at an arbitrary point."""
        res = self.res
        gx = p.x * res - 0.5
        gy = p.y * res - 0.5
        gz = p.z * res - 0.5

        x0 = int(ti.floor(gx));  x1 = x0 + 1
        y0 = int(ti.floor(gy));  y1 = y0 + 1
        z0 = int(ti.floor(gz));  z1 = z0 + 1

        tx = gx - x0;  ty = gy - y0;  tz = gz - z0

        x0 = ti.max(0, ti.min(res - 1, x0));  x1 = ti.max(0, ti.min(res - 1, x1))
        y0 = ti.max(0, ti.min(res - 1, y0));  y1 = ti.max(0, ti.min(res - 1, y1))
        z0 = ti.max(0, ti.min(res - 1, z0));  z1 = ti.max(0, ti.min(res - 1, z1))

        c00 = self.sdf_stock[env_id,x0,y0,z0]*(1-tx) + self.sdf_stock[env_id,x1,y0,z0]*tx
        c10 = self.sdf_stock[env_id,x0,y1,z0]*(1-tx) + self.sdf_stock[env_id,x1,y1,z0]*tx
        c01 = self.sdf_stock[env_id,x0,y0,z1]*(1-tx) + self.sdf_stock[env_id,x1,y0,z1]*tx
        c11 = self.sdf_stock[env_id,x0,y1,z1]*(1-tx) + self.sdf_stock[env_id,x1,y1,z1]*tx
        c0 = c00*(1-ty) + c10*ty
        c1 = c01*(1-ty) + c11*ty
        return c0*(1-tz) + c1*tz

    @ti.kernel
    def reset_envs(self, reset_mask: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        """Resets specific environments to initial states."""
        start_pos = ti.Vector([float(TOOL_START_POS[0]), float(TOOL_START_POS[1]), float(TOOL_START_POS[2])])
        cx = float(STOCK_CENTER[0])
        cy = float(STOCK_CENTER[1])
        cz = float(STOCK_CENTER[2])
        hs = float(STOCK_HALF_SIZE)
        for env_id in range(self.num_envs):
            if reset_mask[env_id]:
                self.tool_pos[env_id] = start_pos
                
        # Reset stock grid for just those envs
        for env_id, i, j, k in ti.ndrange(self.num_envs, self.res, self.res, self.res):
            if reset_mask[env_id]:
                p = ti.Vector([(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx])
                d = ti.abs(p - ti.Vector([cx, cy, cz])) - hs
                dist = ti.max(d.x, ti.max(d.y, d.z))
                self.sdf_stock[env_id, i, j, k] = dist

    # --- Batched Stepping ---

    def batched_step(self, actions_np):
        """Executes a full step vector for all environments.
        
        Args:
            actions_np: [num_envs, 3] integer array of local relative actions {-1, 0, 1}
        """
        # 1. Update tool pos and check collisions
        self.move_blocked.fill(0)
        self._batched_move_and_collide(actions_np)
        
        # 2. Apply cuts for valid moves
        self.cut_vol_field.fill(0.0)
        self.target_violation.fill(0.0)
        self._batched_apply_cut()
        
        # 3. Build observations and update excess fields
        self.excess_field.fill(0.0)
        self._batched_build_obs()

    @ti.kernel
    def _batched_move_and_collide(self, actions: ti.types.ndarray(dtype=ti.i32, ndim=2)):
        """Moves tools and blocks moves that would cut into the target.

        Instead of only checking tool-body overlap with the target, this
        pre-simulates the SDF subtraction (max(stock, -tool)) on every voxel
        in the tool's bounding box.  If any voxel inside the target
        (target_sdf < 0) would be modified by the cut, the move is blocked.
        """
        tr = self.tool_radius[None]
        th = self.tool_height[None]
        limit = 1.0

        for env_id in range(self.num_envs):
            # 1. Tentatively move tool
            old_pos = self.tool_pos[env_id]
            dx_m = float(actions[env_id, 0]) * self.step_size
            dy_m = float(actions[env_id, 1]) * self.step_size
            dz_m = float(actions[env_id, 2]) * self.step_size

            new_pos = ti.Vector([
                ti.max(0.0, ti.min(limit, old_pos.x + dx_m)),
                ti.max(0.0, ti.min(limit, old_pos.y + dy_m)),
                ti.max(0.0, ti.min(limit, old_pos.z + dz_m))
            ])
            self.tool_pos[env_id] = new_pos

            # 2. Pre-simulate the cut on the grid and check for target damage
            res = self.res
            ix = ti.max(0, ti.min(res - 1, int(new_pos.x * res)))
            iy = ti.max(0, ti.min(res - 1, int(new_pos.y * res)))
            iz = ti.max(0, ti.min(res - 1, int(new_pos.z * res)))

            rx = int(ti.ceil(tr / self.dx))
            rz = int(ti.ceil(th / self.dx))
            margin = 2

            intersects = 0
            for i in range(ix - rx - margin, ix + rx + margin + 1):
                for j in range(iy - rx - margin, iy + rx + margin + 1):
                    for k in range(iz - rz - margin, iz + rz + margin + 1 + rz):
                        if 0 <= i < res and 0 <= j < res and 0 <= k < res:
                            target_dist = self.sdf_target[i, j, k]
                            if target_dist < 0.0:
                                voxel_pos = ti.Vector([
                                    (i + 0.5) * self.dx,
                                    (j + 0.5) * self.dx,
                                    (k + 0.5) * self.dx])
                                tool_dist = self._tool_sdf(voxel_pos, new_pos, th, tr)
                                stock_dist = self.sdf_stock[env_id, i, j, k]
                                would_be = ti.max(stock_dist, -tool_dist)
                                if would_be > stock_dist:
                                    intersects = 1

            if intersects == 1:
                self.move_blocked[env_id] = 1
                self.tool_pos[env_id] = old_pos
                
    @ti.kernel
    def _batched_apply_cut(self):
        """Applies tool subtraction to stock grid for all unblocked environments."""
        tr = self.tool_radius[None]
        th = self.tool_height[None]
        res = self.res

        for env_id in range(self.num_envs):
            # Only apply cut if move was legal
            if self.move_blocked[env_id] == 0:
                tpos = self.tool_pos[env_id]
                ix = ti.max(0, ti.min(res - 1, int(tpos.x * res)))
                iy = ti.max(0, ti.min(res - 1, int(tpos.y * res)))
                iz = ti.max(0, ti.min(res - 1, int(tpos.z * res)))

                rx = int(ti.ceil(tr / self.dx))
                rz = int(ti.ceil(th / self.dx))
                margin = 2

                for i in range(ix - rx - margin, ix + rx + margin + 1):
                    for j in range(iy - rx - margin, iy + rx + margin + 1):
                        for k in range(iz - rz - margin, iz + rz + margin + 1 + rz):
                            if 0 <= i < res and 0 <= j < res and 0 <= k < res:
                                voxel_pos = ti.Vector([(i + 0.5) * self.dx,
                                                       (j + 0.5) * self.dx,
                                                       (k + 0.5) * self.dx])
                                tool_dist = self._tool_sdf(voxel_pos, tpos, th, tr)

                                old_dist = self.sdf_stock[env_id, i, j, k]
                                new_dist = ti.max(old_dist, -tool_dist)

                                if new_dist > old_dist:
                                    ti.atomic_add(self.cut_vol_field[env_id], new_dist - old_dist)
                                    self.sdf_stock[env_id, i, j, k] = new_dist

                                    # Diagnostic: detect if cut damaged the target
                                    # Should be zero if collision check works correctly.
                                    target_dist = self.sdf_target[i, j, k]
                                    if target_dist < 0.0 and new_dist > target_dist:
                                        ti.atomic_add(self.target_violation[env_id], new_dist - target_dist)

    @ti.kernel
    def _batched_build_obs(self):
        """Computes gradients, SDF grids, and total excess volume across all environments."""
        res = self.res

        # 1. Setup per-environment tool position and gradients (Serial over grid, parallel over envs)
        for env_id in range(self.num_envs):
            tp = self.tool_pos[env_id]
            self.obs_buffer[env_id, 0] = tp.x
            self.obs_buffer[env_id, 1] = tp.y
            self.obs_buffer[env_id, 2] = tp.z

            eps = 1e-3
            dx_vec = ti.Vector([eps, 0.0, 0.0])
            dy_vec = ti.Vector([0.0, eps, 0.0])
            dz_vec = ti.Vector([0.0, 0.0, eps])

            # ∇φ_stock
            gs_x = (self._sample_stock_sdf(env_id, tp + dx_vec) - self._sample_stock_sdf(env_id, tp - dx_vec)) / (2.0 * eps)
            gs_y = (self._sample_stock_sdf(env_id, tp + dy_vec) - self._sample_stock_sdf(env_id, tp - dy_vec)) / (2.0 * eps)
            gs_z = (self._sample_stock_sdf(env_id, tp + dz_vec) - self._sample_stock_sdf(env_id, tp - dz_vec)) / (2.0 * eps)
            gs = ti.Vector([gs_x, gs_y, gs_z])
            gs_norm = gs.norm()
            if gs_norm > 1e-8:
                gs = gs / gs_norm
            self.obs_buffer[env_id, 3] = gs.x
            self.obs_buffer[env_id, 4] = gs.y
            self.obs_buffer[env_id, 5] = gs.z

            # ∇(φ_target − φ_stock)
            gd_x = ((self._sample_target_sdf(tp + dx_vec) - self._sample_stock_sdf(env_id, tp + dx_vec))
                   - (self._sample_target_sdf(tp - dx_vec) - self._sample_stock_sdf(env_id, tp - dx_vec))) / (2.0 * eps)
            gd_y = ((self._sample_target_sdf(tp + dy_vec) - self._sample_stock_sdf(env_id, tp + dy_vec))
                   - (self._sample_target_sdf(tp - dy_vec) - self._sample_stock_sdf(env_id, tp - dy_vec))) / (2.0 * eps)
            gd_z = ((self._sample_target_sdf(tp + dz_vec) - self._sample_stock_sdf(env_id, tp + dz_vec))
                   - (self._sample_target_sdf(tp - dz_vec) - self._sample_stock_sdf(env_id, tp - dz_vec))) / (2.0 * eps)
            self.obs_buffer[env_id, 6] = gd_x
            self.obs_buffer[env_id, 7] = gd_y
            self.obs_buffer[env_id, 8] = gd_z

        # 2. Extract clipped grids and compute excess volume (Parallel over full 4D grid)
        for env_id, i, j, k in ti.ndrange(self.num_envs, res, res, res):
            idx = i * res * res + j * res + k
            stock_val = self.sdf_stock[env_id, i, j, k]
            target_val = self.sdf_target[i, j, k]

            # Copy to 1D obs buffer per env
            self.obs_buffer[env_id, 9 + idx] = ti.max(-1.0, ti.min(1.0, stock_val))
            self.obs_buffer[env_id, 9 + res * res * res + idx] = ti.max(-1.0, ti.min(1.0, target_val))

            # Excess Volume accumulation
            excess_val = ti.max(ti.min(-stock_val, target_val), 0.0)
            if excess_val > 0.0:
                ti.atomic_add(self.excess_field[env_id], excess_val)
