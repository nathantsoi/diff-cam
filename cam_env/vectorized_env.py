from __future__ import annotations

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import taichi as ti
import pufferlib

from simulator.batched_voxel_simulator import BatchedCNCSimulator
from cam_env.physics_config import (
    TIME_PENALTY, COMPLETION_BONUS, COMPLETION_THRESHOLD,
    PROXIMITY_COEF_INITIAL, PROXIMITY_ANNEAL_STEPS, PROXIMITY_CLIP,
    STOCK_HALF_SIZE, TARGET_RADIUS, TOOL_START_POS, TOOL_RADIUS, TOOL_HEIGHT,
)


@ti.data_oriented
class PufferBatchedCamEnv(pufferlib.PufferEnv):
    """Batched diff-cam environment using PufferLib PufferEnv API.
    
    This environment manages N diff-cam instances simultaneously using a single
    BatchedCNCSimulator running 4D Taichi fields, eliminating per-env kernel
    launch overhead.

    Observations are built in this environment class using Taichi kernels that
    read directly from the simulator's SDF fields, keeping the simulator as a
    pure physics engine.
    """

    def __init__(
        self,
        num_envs: int,
        resolution: int = 8,
        max_steps: int = 1000,
        buf=None,
        seed=None,
    ):
        self.num_envs = num_envs
        self.resolution = resolution
        self.max_steps = max_steps

        # Action space: 3x3x3 discrete movements
        self.single_action_space = spaces.Discrete(27)
        
        # Observation space: pos(3) + gs(3) + gd(3) + stock(res³) + target(res³)
        self.obs_size = 9 + 2 * (resolution ** 3)
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )
        
        # PufferLib configuration
        self.render_mode = "ansi"
        self.num_agents = num_envs
        super().__init__(buf=buf)

        # --- Internal State ---
        self.sim = BatchedCNCSimulator(num_envs=num_envs, resolution=resolution)

        # --- Observation Taichi Fields (owned by the environment) ---
        self.obs_buffer = ti.field(dtype=ti.f32, shape=(num_envs, self.obs_size))

        # Global track of progress for shaping coefficients
        self.global_step = 0

        # Python-side state per environment
        self._step_counts = np.zeros(num_envs, dtype=np.int32)
        self._prev_excess = np.zeros(num_envs, dtype=np.float32)

        # Array of directions for fast vector lookup
        self._action_lookup = np.zeros((27, 3), dtype=np.int32)
        for a in range(27):
            self._action_lookup[a] = [(a // 9) - 1, ((a // 3) % 3) - 1, (a % 3) - 1]

    # --- Observation Kernel (owned by env, reads from sim fields) ---

    @ti.kernel
    def _build_obs(self):
        """Computes gradients, SDF grids, and total excess volume across all environments.
        
        Reads from simulator fields (sdf_stock, sdf_target, tool_pos) and writes
        to the environment's obs_buffer. Also accumulates excess volume into
        the simulator's excess_field as a side-effect.
        """
        res = self.resolution
        inv_2dx = float(res) * 0.5

        # 1. Per-environment tool position and gradients
        for env_id in range(self.num_envs):
            tp = self.sim.tool_pos[env_id]
            self.obs_buffer[env_id, 0] = tp.x
            self.obs_buffer[env_id, 1] = tp.y
            self.obs_buffer[env_id, 2] = tp.z

            ix = ti.max(1, ti.min(res - 2, int(tp.x * res)))
            iy = ti.max(1, ti.min(res - 2, int(tp.y * res)))
            iz = ti.max(1, ti.min(res - 2, int(tp.z * res)))

            # ∇φ_stock
            gs_x = (self.sim.sdf_stock[env_id, ix+1, iy, iz] - self.sim.sdf_stock[env_id, ix-1, iy, iz]) * inv_2dx
            gs_y = (self.sim.sdf_stock[env_id, ix, iy+1, iz] - self.sim.sdf_stock[env_id, ix, iy-1, iz]) * inv_2dx
            gs_z = (self.sim.sdf_stock[env_id, ix, iy, iz+1] - self.sim.sdf_stock[env_id, ix, iy, iz-1]) * inv_2dx
            gs = ti.Vector([gs_x, gs_y, gs_z])
            gs_norm = gs.norm()
            if gs_norm > 1e-8:
                gs = gs / gs_norm
            self.obs_buffer[env_id, 3] = gs.x
            self.obs_buffer[env_id, 4] = gs.y
            self.obs_buffer[env_id, 5] = gs.z

            # ∇(φ_target − φ_stock)
            gd_x = ((self.sim.sdf_target[ix+1, iy, iz] - self.sim.sdf_stock[env_id, ix+1, iy, iz])
                   - (self.sim.sdf_target[ix-1, iy, iz] - self.sim.sdf_stock[env_id, ix-1, iy, iz])) * inv_2dx
            gd_y = ((self.sim.sdf_target[ix, iy+1, iz] - self.sim.sdf_stock[env_id, ix, iy+1, iz])
                   - (self.sim.sdf_target[ix, iy-1, iz] - self.sim.sdf_stock[env_id, ix, iy-1, iz])) * inv_2dx
            gd_z = ((self.sim.sdf_target[ix, iy, iz+1] - self.sim.sdf_stock[env_id, ix, iy, iz+1])
                   - (self.sim.sdf_target[ix, iy, iz-1] - self.sim.sdf_stock[env_id, ix, iy, iz-1])) * inv_2dx
            self.obs_buffer[env_id, 6] = gd_x
            self.obs_buffer[env_id, 7] = gd_y
            self.obs_buffer[env_id, 8] = gd_z

        # 2. Extract clipped grids and compute excess volume (Parallel over full 4D grid)
        for env_id, i, j, k in ti.ndrange(self.num_envs, res, res, res):
            idx = i * res * res + j * res + k
            stock_val = self.sim.sdf_stock[env_id, i, j, k]
            target_val = self.sim.sdf_target[i, j, k]

            # Copy to 1D obs buffer per env
            self.obs_buffer[env_id, 9 + idx] = ti.max(-1.0, ti.min(1.0, stock_val))
            self.obs_buffer[env_id, 9 + res * res * res + idx] = ti.max(-1.0, ti.min(1.0, target_val))

            # Excess Volume accumulation
            excess_val = ti.max(ti.min(-stock_val, target_val), 0.0)
            if excess_val > 0.0:
                ti.atomic_add(self.sim.excess_field[env_id], excess_val)

    def _sync_observations(self):
        """Builds observations on GPU and syncs to PufferLib buffer."""
        self.sim.excess_field.fill(0.0)
        self._build_obs()
        obs_np = self.obs_buffer.to_numpy()
        self.observations[:] = obs_np

    def reset(self, seed: int | None = None) -> tuple:
        """Resets all environments to their initial states."""
        if seed is not None:
            np.random.seed(seed)

        self.sim.initialize_stock(STOCK_HALF_SIZE)
        self.sim.initialize_target_sphere(TARGET_RADIUS)
        self.sim.initialize_tool(TOOL_START_POS, TOOL_RADIUS, TOOL_HEIGHT)

        self._step_counts.fill(0)
        self.global_step = 0

        # Compute initial observations and excess on GPU
        self._sync_observations()
        self._prev_excess[:] = self.sim.excess_field.to_numpy()

        # Reset puffer buffers
        self.rewards.fill(0.0)
        self.terminals.fill(False)
        self.truncations.fill(False)

        return self.observations, []

    def step(self, actions):
        """Executes a batched step across all environments."""
        flat_actions = actions.reshape(-1)
        
        # 1. Map 0-26 discrete actions to (x,y,z) directions
        move_dirs = self._action_lookup[flat_actions]
        
        # 2. Execute batched physics/simulation step
        self.sim.batched_step(move_dirs)
        
        # 3. Build observations (also computes excess as side-effect)
        self._sync_observations()
        excess_after = self.sim.excess_field.to_numpy()
        cut_vol = self.sim.cut_vol_field.to_numpy()
        move_blocked = self.sim.move_blocked.to_numpy()

        # 4. Compute batched rewards and terminals
        infos = []
        target_violations = self.sim.target_violation.to_numpy()

        # Anneal proximity coefficient linearly to zero
        anneal_frac = max(0.0, 1.0 - self.global_step / PROXIMITY_ANNEAL_STEPS)
        proximity_coef = PROXIMITY_COEF_INITIAL * anneal_frac
        self.global_step += self.num_envs
        
        obs_flat = self.observations
        
        for env_id in range(self.num_envs):
            self._step_counts[env_id] += 1
            
            e_before = self._prev_excess[env_id]
            e_after = excess_after[env_id]
            
            progress = e_before - e_after
            
            completed = e_after < COMPLETION_THRESHOLD
            completion_bonus = COMPLETION_BONUS if completed else 0.0

            # Proximity Shaping
            grad_diff = obs_flat[env_id, 6:9]
            move_dir = move_dirs[env_id].astype(np.float32)
            move_norm = np.linalg.norm(move_dir)
            if move_norm > 1e-8:
                move_dir = move_dir / move_norm
            
            prox_bonus = proximity_coef * float(np.dot(move_dir, grad_diff))
            prox_bonus = np.clip(prox_bonus, -PROXIMITY_CLIP, PROXIMITY_CLIP)

            reward = progress + TIME_PENALTY + completion_bonus + prox_bonus
            
            truncated = self._step_counts[env_id] >= self.max_steps
            terminated = completed

            self.rewards[env_id] = reward
            self.terminals[env_id] = terminated
            self.truncations[env_id] = truncated

            # Warn on target violations
            violation = float(target_violations[env_id])
            if violation > 0 and not move_blocked[env_id]:
                print(f"[WARN] Env {env_id} step {self._step_counts[env_id]}: target violation = {violation:.4f}")

            infos.append({
                "step": self._step_counts[env_id],
                "action": move_dirs[env_id].tolist(),
                "vol": float(cut_vol[env_id]),
                "excess": float(e_after),
                "completed": completed,
                "move_blocked": bool(move_blocked[env_id]),
                "proximity_coef": proximity_coef,
                "target_violation": violation,
            })
            
            # Auto-reset logic
            if terminated or truncated:
                self._step_counts[env_id] = 0

        # Update cached excess for next step
        self._prev_excess[:] = excess_after

        reset_mask = np.logical_or(self.terminals, self.truncations).astype(np.int32)
        if np.any(reset_mask):
            self.sim.reset_envs(reset_mask, ti.Vector(TOOL_START_POS), STOCK_HALF_SIZE)
            # Re-build obs for reset envs
            self._sync_observations()
            self._prev_excess[:] = self.sim.excess_field.to_numpy()

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        pass
    
    def render(self):
        return None
