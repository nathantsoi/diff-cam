from __future__ import annotations

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pufferlib

from simulator.batched_simulator import BatchedCNCSimulator
from cam_env.physics_config import (
    TIME_PENALTY, COMPLETION_BONUS, COMPLETION_THRESHOLD,
    PROXIMITY_COEF_INITIAL, PROXIMITY_ANNEAL_FRACTION, PROXIMITY_CLIP,
    BLOCKED_PENALTY,
)


class PufferBatchedCamEnv(pufferlib.PufferEnv):
    """Batched diff-cam environment using PufferLib PufferEnv API.
    
    This environment manages N diff-cam instances simultaneously using a single
    BatchedCNCSimulator running 4D Taichi fields, eliminating per-env kernel
    launch overhead.
    """

    def __init__(
        self,
        num_envs: int,
        resolution: int = 8,
        step_size: float = 0.05,
        max_steps: int = 1000,
        total_timesteps: int = 1000000,
        buf=None,
        seed=None,
        **kwargs
    ):
        self.num_envs = num_envs
        self.resolution = resolution
        self.step_size = step_size
        self.max_steps = max_steps
        self.total_timesteps = total_timesteps

        # Action space: 3x3x3 discrete movements
        self.single_action_space = spaces.Discrete(27)
        
        # Observation space: pos(3) + gs(3) + gd(3) + stock(res³) + target(res³)
        obs_size = 9 + 2 * (resolution ** 3)
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # PufferLib configuration
        self.render_mode = "ansi"
        # Since our single PufferEnv simulates N independent games, we tell
        # PufferLib there are N agents.
        self.num_agents = num_envs
        # Standard PufferEnv initialization allocates self.observations, 
        # self.rewards, self.terminals, self.truncations
        super().__init__(buf=buf)

        # --- Internal State ---
        self.sim = BatchedCNCSimulator(num_envs=num_envs, resolution=resolution, step_size=step_size)

        # Global track of progress for shaping coefficients
        self.global_step = 0
        # Anneal proximity over the first PROXIMITY_ANNEAL_FRACTION of training
        self.proximity_anneal_steps = int(total_timesteps * PROXIMITY_ANNEAL_FRACTION)

        # Python-side state per environment
        self._step_counts = np.zeros(num_envs, dtype=np.int32)
        self._prev_excess = np.zeros(num_envs, dtype=np.float32)

        # Array of directions for fast vector lookup
        self._action_lookup = np.zeros((27, 3), dtype=np.int32)
        for a in range(27):
            self._action_lookup[a] = [(a // 9) - 1, ((a // 3) % 3) - 1, (a % 3) - 1]

    def _sync_observations(self):
        """Copies batched Taichi observation fields to PufferLib buffer."""
        # Transfer the (num_envs, obs_size) shaped array from GPU to CPU
        obs_np = self.sim.obs_buffer.to_numpy()
        # PufferEnv stores observations as (num_envs * num_agents, obs_size)
        self.observations[:] = obs_np

    def reset(self, seed: int | None = None) -> tuple:
        """Resets all environments to their initial states.
        
        In PufferEnv, reset() is typically called once at the start.
        Auto-resetting during steps handles individual environment resets.
        """
        if seed is not None:
            np.random.seed(seed)

        self.sim.initialize_stock_primitive()
        self.sim.initialize_target_primitive()
        self.sim.initialize_tool_primitive()

        self._step_counts.fill(0)
        self.global_step = 0

        # Compute initial observations and excess on GPU
        self.sim.excess_field.fill(0.0)
        self.sim._batched_build_obs()

        # Sync to CPU
        self._sync_observations()
        self._prev_excess[:] = self.sim.excess_field.to_numpy()

        # Reset pufferv buffers
        self.rewards.fill(0.0)
        self.terminals.fill(False)
        self.truncations.fill(False)

        return self.observations, []

    def step(self, actions):
        """Executes a batched step across all environments.
        
        Args:
            actions: (num_envs, num_agents) array. For us, num_agents=1.
        """
        # Flatten actions to (num_envs,) shape
        flat_actions = actions.reshape(-1)
        
        # 1. Map 0-26 discrete actions to (x,y,z) directions
        move_dirs = self._action_lookup[flat_actions]
        
        # 2. Execute batched physics/simulation step
        self.sim.batched_step(move_dirs)
        
        # 3. Synchronize necessary fields from GPU -> CPU
        self._sync_observations()
        excess_after = self.sim.excess_field.to_numpy()
        cut_vol = self.sim.cut_vol_field.to_numpy()
        move_blocked = self.sim.move_blocked.to_numpy()

        # 4. Compute batched rewards and terminals
        infos = []
        target_violations = self.sim.target_violation.to_numpy()

        # Anneal proximity coefficient linearly to zero
        anneal_frac = max(0.0, 1.0 - self.global_step / self.proximity_anneal_steps)
        proximity_coef = PROXIMITY_COEF_INITIAL * anneal_frac
        self.global_step += self.num_envs
        
        obs_flat = self.observations # Shape: (N, obs_size)
        
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
            
            if move_blocked[env_id]:
                reward += BLOCKED_PENALTY
            
            truncated = self._step_counts[env_id] >= self.max_steps
            terminated = completed

            self.rewards[env_id] = reward
            self.terminals[env_id] = terminated
            self.truncations[env_id] = truncated

            # Warn on target violations (only for unblocked moves — blocked ones are expected)
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
                "reward_progress": float(progress),
                "reward_time_penalty": float(TIME_PENALTY),
                "reward_completion_bonus": float(completion_bonus),
                "reward_prox_bonus": float(prox_bonus),
                "reward_blocked_penalty": float(BLOCKED_PENALTY if move_blocked[env_id] else 0.0),
                 "reward_total": float(reward),
            })
            
            # Auto-reset logic
            if terminated or truncated:
                self._step_counts[env_id] = 0

        # Update cached excess for next step (for envs that didn't reset)
        self._prev_excess[:] = excess_after

        reset_mask = np.logical_or(self.terminals, self.truncations).astype(np.int32)
        if np.any(reset_mask):
            self.sim.reset_envs(reset_mask)
            # Re-run build_obs to update observation buffer and excess for reset envs
            self.sim.excess_field.fill(0.0)
            self.sim._batched_build_obs()
            self._sync_observations()
            # Only update prev_excess for envs that actually reset —
            # mid-episode envs already have the correct value from line 204.
            new_excess = self.sim.excess_field.to_numpy()
            reset_bool = reset_mask.astype(bool)
            self._prev_excess[reset_bool] = new_excess[reset_bool]

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        pass
    
    def render(self):
        return None
