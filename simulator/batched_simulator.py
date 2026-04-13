import taichi as ti
import numpy as np

from simulator.simulator_utils import *

@ti.data_oriented
class BatchedCNCSimulator:
    _ti_initialized = False

    def __init__(self, num_envs: int, resolution: int = 8, debug: bool = False):
        if not BatchedCNCSimulator._ti_initialized:
            if ti._lib.core.with_cuda():
                ti.init(arch=ti.gpu, debug=debug)
            else:
                ti.init(arch=ti.cpu, debug=debug)
            BatchedCNCSimulator._ti_initialized = True

        self.num_envs = num_envs
        self.res = resolution
        self.dx = 1.0 / self.res

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

    # --- Initialization ---

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
        for env_id, i, j, k in ti.ndrange(self.num_envs, self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            self.sdf_stock[env_id, i, j, k] = box_sdf(p, center, half_size)

    def initialize_tool(self, pos, radius, height):
        self._initialize_tool_kernel(ti.Vector(pos))
        self.tool_radius[None] = radius
        self.tool_height[None] = height

    @ti.kernel
    def _initialize_tool_kernel(self, pos: ti.types.vector(3, ti.f32)):
        for env_id in range(self.num_envs):
            self.tool_pos[env_id] = pos

    @ti.kernel
    def reset_envs(self, reset_mask: ti.types.ndarray(dtype=ti.i32, ndim=1), start_pos: ti.types.vector(3, ti.f32), half_size: ti.f32):
        """Resets specific environments to initial states."""
        for env_id in range(self.num_envs):
            if reset_mask[env_id]:
                self.tool_pos[env_id] = start_pos
                
        # Reset stock grid for just those envs
        for env_id, i, j, k in ti.ndrange(self.num_envs, self.res, self.res, self.res):
            if reset_mask[env_id]:
                p = ti.Vector([i, j, k]) * self.dx
                center = ti.Vector([0.5, 0.5, 0.5])
                self.sdf_stock[env_id, i, j, k] = box_sdf(p, center, half_size)

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

    def compute_excess(self):
        """Computes per-environment excess volume: sum(max(min(-stock, target), 0))."""
        self.excess_field.fill(0.0)
        self._compute_excess_kernel()

    @ti.kernel
    def _compute_excess_kernel(self):
        res = self.res
        for env_id, i, j, k in ti.ndrange(self.num_envs, res, res, res):
            stock_val = self.sdf_stock[env_id, i, j, k]
            target_val = self.sdf_target[i, j, k]
            excess_val = ti.max(ti.min(-stock_val, target_val), 0.0)
            if excess_val > 0.0:
                ti.atomic_add(self.excess_field[env_id], excess_val)

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
        limit = 1.0 - self.dx
        res = self.res

        for env_id in range(self.num_envs):
            # 1. Tentatively move tool
            old_pos = self.tool_pos[env_id]
            dx_m = float(actions[env_id, 0]) * self.dx
            dy_m = float(actions[env_id, 1]) * self.dx
            dz_m = float(actions[env_id, 2]) * self.dx

            new_pos = ti.Vector([
                ti.max(0.0, ti.min(limit, old_pos.x + dx_m)),
                ti.max(0.0, ti.min(limit, old_pos.y + dy_m)),
                ti.max(0.0, ti.min(limit, old_pos.z + dz_m))
            ])
            self.tool_pos[env_id] = new_pos

            # 2. Pre-simulate the cut on the grid and check for target damage
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
                                    i * self.dx,
                                    j * self.dx,
                                    k * self.dx])
                                tool_dist = cylinder_sdf(voxel_pos, new_pos.x, new_pos.y, new_pos.z, tr, th)
                                stock_dist = self.sdf_stock[env_id, i, j, k]
                                would_be = boolean_subtract(stock_dist, tool_dist)
                                if would_be > target_dist:
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
                                voxel_pos = ti.Vector([i * self.dx, j * self.dx, k * self.dx])
                                tool_dist = cylinder_sdf(voxel_pos, tpos.x, tpos.y, tpos.z, tr, th)

                                old_dist = self.sdf_stock[env_id, i, j, k]
                                new_dist = boolean_subtract(old_dist, tool_dist)

                                if new_dist > old_dist:
                                    ti.atomic_add(self.cut_vol_field[env_id], new_dist - old_dist)
                                    self.sdf_stock[env_id, i, j, k] = new_dist

                                    # Diagnostic: detect if cut damaged the target
                                    # Should be zero if collision check works correctly.
                                    target_dist = self.sdf_target[i, j, k]
                                    if target_dist < 0.0 and new_dist > target_dist:
                                        ti.atomic_add(self.target_violation[env_id], new_dist - target_dist)


