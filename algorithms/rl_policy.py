# SPDX-License-Identifier: MIT
"""T7.1 First-Order Differentiable vs. Zero-Order RL & Trajectory Optimization.

Implements controlled, MIT-licensed solvers sharing the exact ``CandidateEvaluator``,
``ComputeLedger``, and ``HardSelectionPolicy`` contracts:
1. ``ResidualMillingPolicy`` + ``DirectSegmentActionMap``: Shared 2-layer tanh MLP policy
   mapping 18-D geometry/kinematic segment features to bounded 2.5D XY + log-feedrate actions,
   with exact analytical backpropagation (pure vectorized float64 NumPy).
2. ``train_diff_policy_shac``: First-order differentiable policy optimization (SHAC-style
   analytical policy gradient through smooth/STE Taichi CSG + blended kinematics).
3. ``train_hard_ppo``: Zero-order Clipped Proximal Policy Optimization (PPO + GAE + value
   network critic) trained exclusively on discontinuous ``forward_hard`` rollouts.
4. ``solve_ste_continuation_adam``: First-order trajectory optimization over ``JointPathFeedSpline``
   with Straight-Through Estimator (STE), k-continuation, and hard-feasible line-search guard.
5. ``solve_hard_cem``: Zero-order Cross-Entropy Method (CEM) elite-population optimizer.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from algorithms import blackbox as B
from algorithms import objectives as O
from benchmarks import contracts as C
from benchmarks.milling_suite import compute_waypoint_xy_weights

STATE_DIM = 18
ACTION_DIM = 3


def _precondition_policy_action_grad(grad_mu: np.ndarray) -> np.ndarray:
    """Balance XY position vs. log-feedrate gradient scales before MLP backpropagation."""
    g = np.asarray(grad_mu, dtype=np.float64).copy()
    n_xy = float(np.linalg.norm(g[:, :2]))
    n_f = float(np.linalg.norm(g[:, 2]))
    if n_xy > 1e-12:
        g[:, :2] = g[:, :2] / max(n_xy, 1e-3)
    if n_f > 1e-12:
        g[:, 2] = g[:, 2] / max(n_f, 1e-3)
    return g


class DirectSegmentActionMap:
    """Maps raw policy pre-activations ``u in R^(T*3)`` to 2.5D waypoint shifts & feed multipliers.

    Implements the ``Parameterization`` interface so ``CandidateEvaluator`` scores policy
    actions identically to spline parameterizations:
    - ``a = tanh(u) in (-1, 1)^(T, 3)``
    - Interior waypoint XY shifts ``delta_P[i] = scale_mm * w[i] * a[i, :2]`` (for ``i = 0..T-2``,
      with ``delta_P[-1] = delta_P[T-1] = 0``) map to segment delta residuals via exact first
      differences ``d_xy[t] = delta_P[t] - delta_P[t-1]``. Every waypoint displacement is strictly
      ``O(1)``-bounded by ``scale_mm`` regardless of horizon ``T``, start/end coordinates are
      exact invariants, and vertical plunges/retracts remain strictly vertical.
    - Cutting feed multipliers map ``a[:, 2]`` via ``min(exp(log_feed_scale * a[:, 2]), max_mult)``
      with rapids pinned identically to ``1.0``.
    """

    def __init__(
        self,
        task: C.TaskSpec,
        motion: C.MotionPlan,
        *,
        pos_scale_mm: float = 1.25,
        log_feed_scale: float = 1.25,
        u_bound: float = 6.0,
        waypoint_weights: Optional[np.ndarray] = None,
    ) -> None:
        inv = task.work_to_machine.inverse().apply
        pts_work = np.asarray(
            [inv(motion.initial_state.position_mm)] + [inv(s.end_mm) for s in motion.segments],
            dtype=np.float64,
        )
        self._init_pos = pts_work[0].copy()
        self._init_pos.flags.writeable = False
        self._ref_deltas = np.diff(pts_work, axis=0)
        self._ref_deltas.flags.writeable = False
        self._ref_positions = pts_work.copy()
        self._ref_positions.flags.writeable = False

        self.num_steps = len(motion.segments)
        self.num_params = self.num_steps * ACTION_DIM
        self.parameter_shape = (self.num_params,)
        self.scale_mm = float(pos_scale_mm)
        self.log_feed_scale = float(log_feed_scale)
        self.u_bound = float(u_bound)

        is_rapid = np.array([s.kind == C.SegmentKind.RAPID for s in motion.segments], dtype=bool)
        self._is_rapid = is_rapid
        self._cut_mask = (~is_rapid).astype(np.float64)

        # Interior waypoint weights w in [0, 1]^T (last entry T-1 is terminal P_T, pinned to 0.0)
        wp_w = np.zeros(self.num_steps, dtype=np.float64)
        if self.num_steps >= 2:
            if waypoint_weights is not None:
                w_in = np.asarray(waypoint_weights, dtype=np.float64).reshape(-1)
                wp_w[: self.num_steps - 1] = w_in[: self.num_steps - 1]
            else:
                wp_w[: self.num_steps - 1] = compute_waypoint_xy_weights(
                    task, motion, scale_mm=self.scale_mm
                )
        self._wp_weights = wp_w
        self._wp_weights.flags.writeable = False

        self._max_mult = np.array(
            [
                1.0 if s.kind == C.SegmentKind.RAPID else (task.limits.max_feed_mm_s * 0.995) / s.feed_mm_s
                for s in motion.segments
            ],
            dtype=np.float64,
        )
        self._lo = np.full(self.num_params, -self.u_bound, dtype=np.float64)
        self._hi = np.full(self.num_params, self.u_bound, dtype=np.float64)

    @property
    def initial_position_mm(self) -> np.ndarray:
        return self._init_pos

    @property
    def reference_deltas_mm(self) -> np.ndarray:
        return self._ref_deltas

    @property
    def reference_positions_mm(self) -> np.ndarray:
        return self._ref_positions

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lo, self._hi

    def in_bounds(self, theta: Any, tolerance: float = 0.0) -> bool:
        arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if arr.shape != (self.num_params,) or not np.isfinite(arr).all():
            return False
        return bool(np.all(arr >= -self.u_bound - tolerance) and np.all(arr <= self.u_bound + tolerance))

    def clip(self, theta: Any) -> np.ndarray:
        arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        return np.clip(arr, -self.u_bound, self.u_bound)

    def _u_matrix(self, theta: Any) -> np.ndarray:
        arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if arr.shape != (self.num_params,) or not np.isfinite(arr).all():
            raise ValueError(f"expected finite ({self.num_params},) action vector")
        if np.any(np.abs(arr) > self.u_bound + 1e-12):
            raise ValueError("action parameter outside u_bound")
        return arr.reshape(self.num_steps, ACTION_DIM)

    def _waypoint_xy_shifts(self, u: np.ndarray) -> np.ndarray:
        a_xy = np.tanh(u[:, :2])
        return self.scale_mm * self._wp_weights[:, None] * a_xy

    def forward(self, theta: Any) -> np.ndarray:
        u = self._u_matrix(theta)
        dp_xy = self._waypoint_xy_shifts(u)
        d_xy = np.empty_like(dp_xy)
        d_xy[0] = dp_xy[0]
        if self.num_steps > 1:
            d_xy[1:] = dp_xy[1:] - dp_xy[:-1]
        deltas = self._ref_deltas.copy()
        deltas[:, :2] += d_xy
        return deltas

    def positions(self, theta: Any) -> np.ndarray:
        u = self._u_matrix(theta)
        dp_xy = self._waypoint_xy_shifts(u)
        pos = self._ref_positions.copy()
        pos[1:, :2] += dp_xy
        return pos

    def forward_feed_multipliers(self, theta: Any) -> np.ndarray:
        u = self._u_matrix(theta)
        a_f = np.tanh(u[:, 2])
        raw_mult = np.exp(self.log_feed_scale * a_f)
        capped = np.minimum(raw_mult, self._max_mult)
        return 1.0 + self._cut_mask * (capped - 1.0)

    def pullback(self, grad_deltas: Any) -> np.ndarray:
        raise NotImplementedError("use pullback_joint with theta for DirectSegmentActionMap")

    def pullback_joint(self, grad_deltas: Any, grad_feed_mult: Any, theta: Any) -> np.ndarray:
        u = self._u_matrix(theta)
        a = np.tanh(u)
        sech2 = 1.0 - a * a

        gd = np.asarray(grad_deltas, dtype=np.float64)
        g_dp = np.zeros((self.num_steps, 2), dtype=np.float64)
        if self.num_steps >= 2:
            g_dp[:-1] = gd[:-1, :2] - gd[1:, :2]
        g_dp[-1] = gd[-1, :2]
        da_xy = self.scale_mm * self._wp_weights[:, None] * g_dp
        du_xy = da_xy * sech2[:, :2]

        gm = np.asarray(grad_feed_mult, dtype=np.float64).reshape(self.num_steps)
        raw_mult = np.exp(self.log_feed_scale * a[:, 2])
        d_cap = (raw_mult < self._max_mult).astype(np.float64)
        da_f = gm * self._cut_mask * d_cap * raw_mult * self.log_feed_scale
        du_f = da_f * sech2[:, 2]

        return np.column_stack([du_xy, du_f]).reshape(-1)


def extract_segment_states(
    task: C.TaskSpec,
    motion: C.MotionPlan,
    target_sdf_mm: np.ndarray,
    nominal_engagement: np.ndarray,
    mrr_limit_mm3_s: float,
) -> np.ndarray:
    """Build (T, 18) dimensionless geometric & kinematic state features for every segment."""
    inv = task.work_to_machine.inverse().apply
    pts = np.asarray(
        [inv(motion.initial_state.position_mm)] + [inv(s.end_mm) for s in motion.segments],
        dtype=np.float64,
    )
    deltas = np.diff(pts, axis=0)
    steps = len(motion.segments)

    lo = np.asarray(task.stock.params["min_mm"], dtype=np.float64)
    hi = np.asarray(task.stock.params["max_mm"], dtype=np.float64)
    centre = 0.5 * (lo + hi)
    half_ext = np.maximum(0.5 * (hi - lo), 1.0)
    r_tool = max(float(task.tool.cutter_radius_mm), 0.5)
    grid = task.grid
    nx, ny, nz = grid.shape
    origin = np.asarray(grid.origin_mm, dtype=np.float64)
    vox = float(grid.voxel_mm)

    def _sample_sdf_and_grad_xy(xyz: np.ndarray) -> Tuple[float, float, float]:
        idx_f = (xyz - origin) / vox - 0.5
        ix = int(np.clip(round(idx_f[0]), 0, nx - 1))
        iy = int(np.clip(round(idx_f[1]), 0, ny - 1))
        iz = int(np.clip(round(idx_f[2]), 0, nz - 1))
        val = float(target_sdf_mm[ix, iy, iz])
        ix_p, ix_m = min(nx - 1, ix + 1), max(0, ix - 1)
        iy_p, iy_m = min(ny - 1, iy + 1), max(0, iy - 1)
        gx = float(target_sdf_mm[ix_p, iy, iz] - target_sdf_mm[ix_m, iy, iz]) / max((ix_p - ix_m) * vox, vox)
        gy = float(target_sdf_mm[ix, iy_p, iz] - target_sdf_mm[ix, iy_m, iz]) / max((iy_p - iy_m) * vox, vox)
        gnorm = max(1.0, math.hypot(gx, gy))
        return float(np.clip(val / (3.0 * r_tool), -2.0, 2.0)), gx / gnorm, gy / gnorm

    lengths = np.linalg.norm(deltas, axis=1)
    unit_dirs = deltas / np.maximum(lengths[:, None], 1e-6)
    stock_vol_mm3 = float( np.prod(hi - lo) )
    durations = np.array([s.duration_s for s in motion.segments], dtype=np.float64)

    states = np.zeros((steps, STATE_DIM), dtype=np.float64)
    for t, seg in enumerate(motion.segments):
        p0, p1 = pts[t], pts[t + 1]
        pmid = 0.5 * (p0 + p1)
        # 0..3: normalized start/end XY
        states[t, 0] = (p0[0] - centre[0]) / half_ext[0]
        states[t, 1] = (p0[1] - centre[1]) / half_ext[1]
        states[t, 2] = (p1[0] - centre[0]) / half_ext[0]
        states[t, 3] = (p1[1] - centre[1]) / half_ext[1]
        # 4..6: chord vector & length
        states[t, 4] = deltas[t, 0] / half_ext[0]
        states[t, 5] = deltas[t, 1] / half_ext[1]
        states[t, 6] = lengths[t] / max(half_ext[0], half_ext[1])
        # 7..8: incoming & outgoing corner cosine angles
        cos_in = float(np.dot(unit_dirs[t - 1], unit_dirs[t])) if t > 0 else 1.0
        cos_out = float(np.dot(unit_dirs[t], unit_dirs[t + 1])) if t + 1 < steps else 1.0
        states[t, 7] = cos_in
        states[t, 8] = cos_out
        # 9..13: target SDF clearance (start, mid, end) + midpoint target SDF XY gradient
        d0, _, _ = _sample_sdf_and_grad_xy(p0)
        dm, gx, gy = _sample_sdf_and_grad_xy(pmid)
        d1, _, _ = _sample_sdf_and_grad_xy(p1)
        states[t, 9] = d0
        states[t, 10] = dm
        states[t, 11] = d1
        states[t, 12] = gx
        states[t, 13] = gy
        # 14..17: nominal engagement, nominal MRR ratio, segment regime flags, progress
        eng_t = float(nominal_engagement[t]) if t < len(nominal_engagement) else 0.0
        is_rapid = 1.0 if seg.kind == C.SegmentKind.RAPID else 0.0
        vol_mm3 = eng_t * stock_vol_mm3 * (1.0 - is_rapid)
        mrr_nom = vol_mm3 / max(durations[t], 1e-6)
        states[t, 14] = float(np.clip(vol_mm3 / max(r_tool ** 3, 1.0), 0.0, 5.0))
        states[t, 15] = float(np.clip(mrr_nom / max(mrr_limit_mm3_s, 1.0), 0.0, 3.0))
        is_horiz_cut = float(seg.kind != C.SegmentKind.RAPID and math.hypot(deltas[t, 0], deltas[t, 1]) > 1e-6)
        states[t, 16] = is_horiz_cut - is_rapid
        states[t, 17] = t / max(1, steps - 1)
    return states


class ResidualMillingPolicy:
    """2-layer tanh MLP Gaussian policy ``pi_psi(u_t | s_t)`` in pure float64 NumPy.

    Initialized with small output weights so ``mu_psi(s_t) ≈ 0`` at iteration 0
    (recovering the safe nominal CAM seed trajectory at initialization).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 64,
        action_dim: int = ACTION_DIM,
        *,
        seed: int = 0,
        init_log_std: float = -1.6,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)

        s1 = math.sqrt(2.0 / (self.state_dim + self.hidden_dim))
        s2 = math.sqrt(2.0 / (2 * self.hidden_dim))
        self.W1 = rng.normal(0.0, s1, size=(self.state_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0.0, s2, size=(self.hidden_dim, self.hidden_dim))
        self.b2 = np.zeros(self.hidden_dim, dtype=np.float64)
        # Small output head so initial deterministic action is close to nominal CAM while passing clear hidden grads
        self.W3 = rng.normal(0.0, 0.015, size=(self.hidden_dim, self.action_dim))
        self.b3 = np.zeros(self.action_dim, dtype=np.float64)
        self.log_std = np.full(self.action_dim, float(init_log_std), dtype=np.float64)

    def copy(self) -> "ResidualMillingPolicy":
        clone = ResidualMillingPolicy(self.state_dim, self.hidden_dim, self.action_dim)
        clone.set_flat_params(self.get_flat_params())
        clone.log_std = self.log_std.copy()
        return clone

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate(
            [self.W1.ravel(), self.b1, self.W2.ravel(), self.b2, self.W3.ravel(), self.b3]
        )

    def set_flat_params(self, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        idx = 0
        n = self.state_dim * self.hidden_dim
        self.W1 = v[idx : idx + n].reshape(self.state_dim, self.hidden_dim).copy()
        idx += n
        self.b1 = v[idx : idx + self.hidden_dim].copy()
        idx += self.hidden_dim
        n = self.hidden_dim * self.hidden_dim
        self.W2 = v[idx : idx + n].reshape(self.hidden_dim, self.hidden_dim).copy()
        idx += n
        self.b2 = v[idx : idx + self.hidden_dim].copy()
        idx += self.hidden_dim
        n = self.hidden_dim * self.action_dim
        self.W3 = v[idx : idx + n].reshape(self.hidden_dim, self.action_dim).copy()
        idx += n
        self.b3 = v[idx : idx + self.action_dim].copy()

    def forward_with_cache(self, states: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        X = np.asarray(states, dtype=np.float64)
        z1 = X @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.tanh(z2)
        mu = h2 @ self.W3 + self.b3
        return mu, {"X": X, "h1": h1, "h2": h2}

    def forward(self, states: np.ndarray) -> np.ndarray:
        mu, _ = self.forward_with_cache(states)
        return mu

    def backward(self, cache: Dict[str, np.ndarray], grad_mu: np.ndarray) -> np.ndarray:
        """Exact analytical gradient dL / d(flat_params) given dL / d(mu) of shape (N, action_dim)."""
        X, h1, h2 = cache["X"], cache["h1"], cache["h2"]
        G3 = np.asarray(grad_mu, dtype=np.float64)
        dW3 = h2.T @ G3
        db3 = np.sum(G3, axis=0)

        dh2 = G3 @ self.W3.T
        dz2 = dh2 * (1.0 - h2 * h2)
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (1.0 - h1 * h1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2, dW3.ravel(), db3])

    def sample(
        self, states: np.ndarray, rng: np.random.Generator, u_bound: float = 6.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample Gaussian pre-activations ``u ~ N(mu(s), std^2)`` and return ``(u, mu, log_prob)``."""
        mu = self.forward(states)
        std = np.exp(np.clip(self.log_std, -4.0, 0.5))
        eps = rng.normal(0.0, 1.0, size=mu.shape)
        u = np.clip(mu + eps * std[None, :], -u_bound, u_bound)
        diff = (u - mu) / std[None, :]
        log_prob = -0.5 * np.sum(
            diff * diff + 2.0 * np.log(std)[None, :] + math.log(2.0 * math.pi), axis=-1
        )
        return u, mu, log_prob

    def log_prob_and_grads(
        self, states: np.ndarray, actions_u: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Return per-step log_prob(N,), forward cache, and std(A,) for PPO updates."""
        mu, cache = self.forward_with_cache(states)
        std = np.exp(np.clip(self.log_std, -4.0, 0.5))
        diff = (actions_u - mu) / std[None, :]
        log_prob = -0.5 * np.sum(
            diff * diff + 2.0 * np.log(std)[None, :] + math.log(2.0 * math.pi), axis=-1
        )
        cache["diff"] = diff
        cache["std"] = std
        return log_prob, cache, std


class ValueCritic:
    """2-layer tanh MLP state-value baseline ``V_omega(s_t)`` for PPO + GAE."""

    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = 64, *, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        s1 = math.sqrt(2.0 / (self.state_dim + self.hidden_dim))
        s2 = math.sqrt(2.0 / (2 * self.hidden_dim))
        self.W1 = rng.normal(0.0, s1, size=(self.state_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0.0, s2, size=(self.hidden_dim, self.hidden_dim))
        self.b2 = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W3 = np.zeros((self.hidden_dim, 1), dtype=np.float64)
        self.b3 = np.zeros(1, dtype=np.float64)

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate(
            [self.W1.ravel(), self.b1, self.W2.ravel(), self.b2, self.W3.ravel(), self.b3]
        )

    def set_flat_params(self, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        idx = 0
        n = self.state_dim * self.hidden_dim
        self.W1 = v[idx : idx + n].reshape(self.state_dim, self.hidden_dim).copy()
        idx += n
        self.b1 = v[idx : idx + self.hidden_dim].copy()
        idx += self.hidden_dim
        n = self.hidden_dim * self.hidden_dim
        self.W2 = v[idx : idx + n].reshape(self.hidden_dim, self.hidden_dim).copy()
        idx += n
        self.b2 = v[idx : idx + self.hidden_dim].copy()
        idx += self.hidden_dim
        self.W3 = v[idx : idx + self.hidden_dim].reshape(self.hidden_dim, 1).copy()
        idx += self.hidden_dim
        self.b3 = v[idx : idx + 1].copy()

    def forward_with_cache(self, states: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        X = np.asarray(states, dtype=np.float64)
        h1 = np.tanh(X @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        val = (h2 @ self.W3 + self.b3).reshape(-1)
        return val, {"X": X, "h1": h1, "h2": h2}

    def backward(self, cache: Dict[str, np.ndarray], grad_val: np.ndarray) -> np.ndarray:
        X, h1, h2 = cache["X"], cache["h1"], cache["h2"]
        G3 = np.asarray(grad_val, dtype=np.float64).reshape(-1, 1)
        dW3 = h2.T @ G3
        db3 = np.sum(G3, axis=0)
        dz2 = (G3 @ self.W3.T) * (1.0 - h2 * h2)
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        dz1 = (dz2 @ self.W2.T) * (1.0 - h1 * h1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2, dW3.ravel(), db3])


class _AdamUpdater:
    def __init__(self, dim: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m = np.zeros(dim, dtype=np.float64)
        self.v = np.zeros(dim, dtype=np.float64)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray, *, lr_scale: float = 1.0, max_norm: float = 1.0) -> np.ndarray:
        g = np.asarray(grad, dtype=np.float64)
        gnorm = float(np.linalg.norm(g))
        if gnorm > max_norm and gnorm > 0:
            g = g * (max_norm / gnorm)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return params - (self.lr * lr_scale) * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class SolverRunSummary:
    method: str
    run_id: str
    seed: int
    evaluations: int
    forward_calls: int
    backward_calls: int
    initial_result: C.CandidateResult
    best_result: C.CandidateResult
    best_theta: np.ndarray
    history: List[Dict[str, float]] = field(default_factory=list)
    policy: Optional[ResidualMillingPolicy] = None


def _better_hard_candidate(
    cand: C.CandidateResult,
    incumbent: Optional[C.CandidateResult],
    resolved_sel: B.ResolvedHardSelection,
) -> bool:
    if incumbent is None:
        return cand.valid
    r_cand = B._hard_rank(cand, resolved_sel)
    r_inc = B._hard_rank(incumbent, resolved_sel)
    if r_cand is None:
        return False
    if r_inc is None:
        return True
    return r_cand < r_inc


def train_diff_policy_shac(
    evaluator: O.CandidateEvaluator,
    states: np.ndarray,
    *,
    max_evaluations: int = 80,
    lr: float = 8e-3,
    k_start: float = 25.0,
    k_end: float = 85.0,
    hidden_dim: int = 64,
    seed: int = 0,
    run_id: str = "diffpolicy-shac",
    initial_policy: Optional[ResidualMillingPolicy] = None,
    selection: Optional[B.HardSelectionPolicy] = None,
) -> SolverRunSummary:
    """First-order analytical policy optimization (SHAC-style) through smooth/STE CSG.

    Each step evaluates ``u = pi_psi(states)`` via ``evaluator.evaluate_ste_with_gradient``,
    obtaining BOTH the exact hard ``CandidateResult`` (for monotonic best-checkpoint tracking)
    and the analytical Jacobian ``d L_smooth / d u`` backpropagated directly into ``psi``.
    """
    sel_policy = selection or B.HardSelectionPolicy()
    policy = (
        initial_policy.copy()
        if initial_policy is not None
        else ResidualMillingPolicy(states.shape[1], hidden_dim, ACTION_DIM, seed=seed)
    )
    params = policy.get_flat_params()
    opt = _AdamUpdater(params.size, lr=lr)

    # Evaluate exact zero-action nominal baseline first (1 hard eval)
    u_zero = np.zeros(evaluator.parameterization.num_params, dtype=np.float64)
    init_res = evaluator.evaluate_candidate(u_zero, f"{run_id}:init", "hard")
    resolved_sel = B._resolve_hard_selection(evaluator, sel_policy)

    best_res = init_res
    best_theta = u_zero.copy()
    best_policy = policy.copy()
    history: List[Dict[str, float]] = [
        {
            "eval": 1.0,
            "objective": float(init_res.objective or 0.0),
            "best_objective": float(best_res.objective or 0.0),
            "cycle_time_s": float(init_res.components.get("cycle_time_s", 0.0)),
            "peak_mrr_mm3_s": float(init_res.components.get("peak_mrr_mm3_s", 0.0)),
            "feasible": float(init_res.components.get("feasible", 0.0)),
        }
    ]

    n_iters = max(1, max_evaluations - 1)
    for step in range(n_iters):
        frac = step / max(1, n_iters - 1)
        k_cur = k_start + (k_end - k_start) * frac
        evaluator.set_evaluation_k(k_cur)

        mu, cache = policy.forward_with_cache(states)
        u_vec = evaluator.parameterization.clip(mu.reshape(-1))
        ste_eval = evaluator.evaluate_ste_with_gradient(u_vec, f"{run_id}:step:{step:03d}")
        res = ste_eval.result

        if res.valid and _better_hard_candidate(res, best_res, resolved_sel):
            best_res = res
            best_theta = u_vec.copy()
            best_policy = policy.copy()

        history.append(
            {
                "eval": float(step + 2),
                "objective": float(res.objective if res.objective is not None else 1e3),
                "best_objective": float(best_res.objective or 0.0),
                "cycle_time_s": float(res.components.get("cycle_time_s", 0.0)),
                "best_cycle_time_s": float(best_res.components.get("cycle_time_s", 0.0)),
                "peak_mrr_mm3_s": float(res.components.get("peak_mrr_mm3_s", 0.0)),
                "feasible": float(res.components.get("feasible", 0.0)),
            }
        )

        if ste_eval.gradient_theta is not None and np.all(np.isfinite(ste_eval.gradient_theta)):
            grad_mu = _precondition_policy_action_grad(
                ste_eval.gradient_theta.reshape(states.shape[0], ACTION_DIM)
            )
            grad_mu = grad_mu + 1e-4 * mu
            grad_psi = policy.backward(cache, grad_mu)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * frac))
            params = opt.step(params, grad_psi, lr_scale=max(0.2, lr_scale), max_norm=3.0)
            policy.set_flat_params(params)

    evaluator.clear_evaluation_k()
    snap = evaluator.ledger.snapshot()
    return SolverRunSummary(
        method="DiffPolicy-SHAC",
        run_id=run_id,
        seed=seed,
        evaluations=int(snap["candidate_evaluations"]),
        forward_calls=int(snap["forward_calls"]),
        backward_calls=int(snap["backward_calls"]),
        initial_result=init_res,
        best_result=best_res,
        best_theta=best_theta,
        history=history,
        policy=best_policy,
    )


def _per_segment_hard_rewards(
    evaluator: O.CandidateEvaluator,
    theta: np.ndarray,
    hard_res: C.CandidateResult,
) -> np.ndarray:
    """Decompose exact hard rollout cost into dense per-segment rewards + terminal feasibility penalty."""
    steps = evaluator.parameterization.num_steps
    if not hard_res.valid or hard_res.objective is None:
        return np.full(steps, -2.0 / max(1, steps), dtype=np.float64)

    motion = evaluator.candidate_motion(theta)
    durations = np.array([s.duration_s for s in motion.segments], dtype=np.float64)
    engagement = np.asarray(evaluator.sim.seg_engage.to_numpy(), dtype=np.float64).reshape(-1)[:steps]
    cells = evaluator.sim.Nx * evaluator.sim.Ny * evaluator.sim.Nz
    air_mask = (engagement <= 0.5 / max(1, cells)).astype(np.float64)
    is_rapid = np.array([s.kind == C.SegmentKind.RAPID for s in motion.segments], dtype=bool)
    stock_box_mm3 = float(evaluator.sim.Lx * evaluator.sim.Ly * evaluator.sim.Lz)
    vol_mm3 = engagement * stock_box_mm3 * (~is_rapid).astype(np.float64)
    mrr_seg = vol_mm3 / np.maximum(durations, 1e-6)

    spec = evaluator.spec
    ref_t = max(evaluator.reference_cycle_time_s, 1e-6)
    step_cost = (
        spec.time_weight * (durations / ref_t)
        + spec.air_time_weight * (durations * air_mask / ref_t)
    )
    if spec.mrr_weight > 0:
        excess = np.maximum(0.0, mrr_seg / spec.mrr_limit_mm3_s - 1.0)
        step_cost = step_cost + (spec.mrr_weight / steps) * (excess ** 2)

    rewards = -step_cost
    # Terminal geometry + hard constraint shaping so sum(rewards) == -hard_objective - feasibility_penalty
    geom_and_safety = float(hard_res.components.get("geometry", 0.0)) + float(
        hard_res.components.get("holder_proxy", 0.0)
    )
    infeasible_pen = 0.5 * (1.0 - float(hard_res.components.get("feasible", 0.0))) + 5.0 * float(
        hard_res.components.get("constraint_violation", 0.0)
    )
    rewards[-1] -= geom_and_safety + infeasible_pen
    return rewards


def train_hard_ppo(
    evaluator: O.CandidateEvaluator,
    states: np.ndarray,
    *,
    max_evaluations: int = 80,
    batch_episodes: int = 8,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    actor_lr: float = 3e-3,
    critic_lr: float = 5e-3,
    init_log_std: float = -1.35,
    hidden_dim: int = 64,
    seed: int = 0,
    run_id: str = "hard-ppo",
    initial_policy: Optional[ResidualMillingPolicy] = None,
    selection: Optional[B.HardSelectionPolicy] = None,
) -> SolverRunSummary:
    """Zero-order Clipped PPO + GAE + Value Critic trained on discontinuous ``forward_hard`` rollouts."""
    rng = np.random.default_rng(seed)
    sel_policy = selection or B.HardSelectionPolicy()
    policy = (
        initial_policy.copy()
        if initial_policy is not None
        else ResidualMillingPolicy(
            states.shape[1], hidden_dim, ACTION_DIM, seed=seed, init_log_std=init_log_std
        )
    )
    critic = ValueCritic(states.shape[1], hidden_dim, seed=seed + 101)
    actor_opt = _AdamUpdater(policy.get_flat_params().size, lr=actor_lr)
    critic_opt = _AdamUpdater(critic.get_flat_params().size, lr=critic_lr)

    u_zero = np.zeros(evaluator.parameterization.num_params, dtype=np.float64)
    init_res = evaluator.evaluate_candidate(u_zero, f"{run_id}:init", "hard")
    resolved_sel = B._resolve_hard_selection(evaluator, sel_policy)

    best_res = init_res
    best_theta = u_zero.copy()
    best_policy = policy.copy()
    eval_count = 1
    history: List[Dict[str, float]] = [
        {
            "eval": 1.0,
            "objective": float(init_res.objective or 0.0),
            "best_objective": float(best_res.objective or 0.0),
            "cycle_time_s": float(init_res.components.get("cycle_time_s", 0.0)),
            "peak_mrr_mm3_s": float(init_res.components.get("peak_mrr_mm3_s", 0.0)),
            "feasible": float(init_res.components.get("feasible", 0.0)),
        }
    ]

    steps = states.shape[0]
    u_bound = evaluator.parameterization.u_bound

    while eval_count < max_evaluations:
        # Reserve 1 final deterministic evaluation at the end of training if possible
        rem = max_evaluations - eval_count
        cur_batch = min(batch_episodes, max(1, rem - 1)) if rem > 1 else 1
        if rem == 1:
            u_det = evaluator.parameterization.clip(policy.forward(states).reshape(-1))
            det_res = evaluator.evaluate_candidate(u_det, f"{run_id}:det-final", "hard")
            eval_count += 1
            if det_res.valid and _better_hard_candidate(det_res, best_res, resolved_sel):
                best_res = det_res
                best_theta = u_det.copy()
                best_policy = policy.copy()
            history.append(
                {
                    "eval": float(eval_count),
                    "objective": float(det_res.objective if det_res.objective is not None else 1e3),
                    "best_objective": float(best_res.objective or 0.0),
                    "cycle_time_s": float(det_res.components.get("cycle_time_s", 0.0)),
                    "best_cycle_time_s": float(best_res.components.get("cycle_time_s", 0.0)),
                    "peak_mrr_mm3_s": float(det_res.components.get("peak_mrr_mm3_s", 0.0)),
                    "feasible": float(det_res.components.get("feasible", 0.0)),
                }
            )
            break

        batch_states: List[np.ndarray] = []
        batch_actions: List[np.ndarray] = []
        batch_old_logp: List[np.ndarray] = []
        batch_adv: List[np.ndarray] = []
        batch_ret: List[np.ndarray] = []

        vals_pred, _ = critic.forward_with_cache(states)
        for ep in range(cur_batch):
            if ep == 0 and eval_count > 1:
                # Include one deterministic mean rollout per batch to evaluate current policy cleanly
                mu = policy.forward(states)
                u_mat = np.clip(mu, -u_bound, u_bound)
                old_lp, _, _ = policy.log_prob_and_grads(states, u_mat)
            else:
                u_mat, _, old_lp = policy.sample(states, rng, u_bound=u_bound)

            u_flat = u_mat.reshape(-1)
            res = evaluator.evaluate_candidate(u_flat, f"{run_id}:ep:{eval_count:03d}", "hard")
            eval_count += 1

            if res.valid and _better_hard_candidate(res, best_res, resolved_sel):
                best_res = res
                best_theta = u_flat.copy()
                best_policy = policy.copy()

            history.append(
                {
                    "eval": float(eval_count),
                    "objective": float(res.objective if res.objective is not None else 1e3),
                    "best_objective": float(best_res.objective or 0.0),
                    "cycle_time_s": float(res.components.get("cycle_time_s", 0.0)),
                    "best_cycle_time_s": float(best_res.components.get("cycle_time_s", 0.0)),
                    "peak_mrr_mm3_s": float(res.components.get("peak_mrr_mm3_s", 0.0)),
                    "feasible": float(res.components.get("feasible", 0.0)),
                }
            )

            rews = _per_segment_hard_rewards(evaluator, u_flat, res)
            adv = np.zeros(steps, dtype=np.float64)
            gae = 0.0
            for t in range(steps - 1, -1, -1):
                next_v = vals_pred[t + 1] if t + 1 < steps else 0.0
                delta = rews[t] + gamma * next_v - vals_pred[t]
                gae = delta + gamma * gae_lambda * gae
                adv[t] = gae
            ret = adv + vals_pred

            batch_states.append(states)
            batch_actions.append(u_mat)
            batch_old_logp.append(old_lp)
            batch_adv.append(adv)
            batch_ret.append(ret)

        S_all = np.vstack(batch_states)
        A_all = np.vstack(batch_actions)
        LP_old = np.concatenate(batch_old_logp)
        ADV = np.concatenate(batch_adv)
        RET = np.concatenate(batch_ret)
        adv_std = float(np.std(ADV))
        if adv_std > 1e-8:
            ADV = (ADV - np.mean(ADV)) / adv_std

        n_samples = S_all.shape[0]
        actor_params = policy.get_flat_params()
        critic_params = critic.get_flat_params()

        for _ in range(ppo_epochs):
            lp_new, p_cache, std = policy.log_prob_and_grads(S_all, A_all)
            ratio = np.exp(np.clip(lp_new - LP_old, -10.0, 10.0))
            unclipped = ratio * ADV
            clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * ADV
            use_unclipped = (
                (ADV >= 0) & (ratio <= 1.0 + clip_eps)
                | (ADV < 0) & (ratio >= 1.0 - clip_eps)
            ).astype(np.float64)
            # Maximize PPO surrogate => minimize negative surrogate
            d_lp = -(use_unclipped * ratio * ADV) / max(1, n_samples)
            # d(log_prob_i) / d(mu_i) = (a_i - mu_i) / std^2 = diff_i / std
            grad_mu = d_lp[:, None] * (p_cache["diff"] / std[None, :])
            grad_actor = policy.backward(p_cache, grad_mu)
            actor_params = actor_opt.step(actor_params, grad_actor, max_norm=0.5)
            policy.set_flat_params(actor_params)

            # Update log_std gently with entropy bonus
            d_log_std = np.sum(
                d_lp[:, None] * (p_cache["diff"] ** 2 - 1.0), axis=0
            ) - 1e-3 * np.ones_like(policy.log_std)
            policy.log_std = np.clip(policy.log_std - 0.01 * d_log_std, -3.0, -0.5)

            v_pred, c_cache = critic.forward_with_cache(S_all)
            grad_v = 2.0 * (v_pred - RET) / max(1, n_samples)
            grad_critic = critic.backward(c_cache, grad_v)
            critic_params = critic_opt.step(critic_params, grad_critic, max_norm=1.0)
            critic.set_flat_params(critic_params)

    snap = evaluator.ledger.snapshot()
    return SolverRunSummary(
        method="Hard-PPO",
        run_id=run_id,
        seed=seed,
        evaluations=int(snap["candidate_evaluations"]),
        forward_calls=int(snap["forward_calls"]),
        backward_calls=int(snap["backward_calls"]),
        initial_result=init_res,
        best_result=best_res,
        best_theta=best_theta,
        history=history,
        policy=best_policy,
    )


def solve_ste_continuation_adam(
    evaluator: O.CandidateEvaluator,
    *,
    max_evaluations: int = 80,
    lr: float = 0.045,
    k_start: float = 25.0,
    k_end: float = 90.0,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    run_id: str = "ste-cont-adam",
    selection: Optional[B.HardSelectionPolicy] = None,
) -> SolverRunSummary:
    """First-order STE-Continuation-Adam over ``JointPathFeedSpline`` with feasibility guard."""
    sel_policy = selection or B.HardSelectionPolicy()
    dim = evaluator.parameterization.num_params
    theta = (
        evaluator.parameterization.clip(x0)
        if x0 is not None
        else np.zeros(dim, dtype=np.float64)
    )
    opt = _AdamUpdater(dim, lr=lr)

    init_res = evaluator.evaluate_candidate(theta, f"{run_id}:init", "hard")
    resolved_sel = B._resolve_hard_selection(evaluator, sel_policy)
    best_res = init_res
    best_theta = theta.copy()

    history: List[Dict[str, float]] = [
        {
            "eval": 1.0,
            "objective": float(init_res.objective or 0.0),
            "best_objective": float(best_res.objective or 0.0),
            "cycle_time_s": float(init_res.components.get("cycle_time_s", 0.0)),
            "peak_mrr_mm3_s": float(init_res.components.get("peak_mrr_mm3_s", 0.0)),
            "feasible": float(init_res.components.get("feasible", 0.0)),
        }
    ]

    n_iters = max(1, max_evaluations - 1)
    consec_infeasible = 0
    for step in range(n_iters):
        frac = step / max(1, n_iters - 1)
        k_cur = k_start + (k_end - k_start) * frac
        evaluator.set_evaluation_k(k_cur)

        ste_eval = evaluator.evaluate_ste_with_gradient(theta, f"{run_id}:step:{step:03d}")
        res = ste_eval.result
        is_feas = res.valid and float(res.components.get("feasible", 0.0)) == 1.0

        if res.valid and _better_hard_candidate(res, best_res, resolved_sel):
            best_res = res
            best_theta = theta.copy()
            consec_infeasible = 0
        elif not is_feas:
            consec_infeasible += 1
        else:
            consec_infeasible = 0

        history.append(
            {
                "eval": float(step + 2),
                "objective": float(res.objective if res.objective is not None else 1e3),
                "best_objective": float(best_res.objective or 0.0),
                "cycle_time_s": float(res.components.get("cycle_time_s", 0.0)),
                "best_cycle_time_s": float(best_res.components.get("cycle_time_s", 0.0)),
                "peak_mrr_mm3_s": float(res.components.get("peak_mrr_mm3_s", 0.0)),
                "feasible": float(res.components.get("feasible", 0.0)),
            }
        )

        # Pull spatial parameters back toward best feasible incumbent if step crossed hard boundary
        n_pos = getattr(evaluator.parameterization, "num_pos_params", dim)
        if not is_feas and consec_infeasible >= 2:
            theta[:n_pos] = 0.75 * best_theta[:n_pos] + 0.25 * theta[:n_pos]
            opt.m[:n_pos] *= 0.4
            consec_infeasible = 0

        if ste_eval.gradient_theta is not None and np.all(np.isfinite(ste_eval.gradient_theta)):
            g = ste_eval.gradient_theta.copy()
            if 0 < n_pos < dim:
                gn_p = float(np.linalg.norm(g[:n_pos]))
                gn_f = float(np.linalg.norm(g[n_pos:]))
                if gn_p > 1e-12:
                    g[:n_pos] = g[:n_pos] / max(gn_p, 1e-3)
                if gn_f > 1e-12:
                    g[n_pos:] = g[n_pos:] / max(gn_f, 1e-3)
            lr_scale = max(0.2, 0.5 * (1.0 + math.cos(math.pi * frac)))
            theta = evaluator.parameterization.clip(
                opt.step(theta, g, lr_scale=lr_scale, max_norm=5.0)
            )

    evaluator.clear_evaluation_k()
    snap = evaluator.ledger.snapshot()
    return SolverRunSummary(
        method="STE-Continuation-Adam",
        run_id=run_id,
        seed=seed,
        evaluations=int(snap["candidate_evaluations"]),
        forward_calls=int(snap["forward_calls"]),
        backward_calls=int(snap["backward_calls"]),
        initial_result=init_res,
        best_result=best_res,
        best_theta=best_theta,
        history=history,
    )


def solve_hard_cem(
    evaluator: O.CandidateEvaluator,
    *,
    max_evaluations: int = 80,
    population_size: int = 12,
    elite_frac: float = 0.25,
    sigma0: float = 0.25,
    smoothing: float = 0.7,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    run_id: str = "hard-cem",
    selection: Optional[B.HardSelectionPolicy] = None,
) -> SolverRunSummary:
    """Zero-order Cross-Entropy Method (CEM) on exact ``forward_hard`` evaluations."""
    rng = np.random.default_rng(seed)
    sel_policy = selection or B.HardSelectionPolicy()
    dim = evaluator.parameterization.num_params
    mean = (
        evaluator.parameterization.clip(x0)
        if x0 is not None
        else np.zeros(dim, dtype=np.float64)
    )
    std = np.full(dim, float(sigma0), dtype=np.float64)

    init_res = evaluator.evaluate_candidate(mean, f"{run_id}:init", "hard")
    resolved_sel = B._resolve_hard_selection(evaluator, sel_policy)
    best_res = init_res
    best_theta = mean.copy()
    eval_count = 1
    history: List[Dict[str, float]] = [
        {
            "eval": 1.0,
            "objective": float(init_res.objective or 0.0),
            "best_objective": float(best_res.objective or 0.0),
            "cycle_time_s": float(init_res.components.get("cycle_time_s", 0.0)),
            "peak_mrr_mm3_s": float(init_res.components.get("peak_mrr_mm3_s", 0.0)),
            "feasible": float(init_res.components.get("feasible", 0.0)),
        }
    ]

    gen = 0
    while eval_count < max_evaluations:
        cur_pop = min(population_size, max_evaluations - eval_count)
        samples: List[np.ndarray] = []
        ranks: List[Tuple[float, ...]] = []
        for idx in range(cur_pop):
            cand_theta = evaluator.parameterization.clip(mean + rng.normal(0.0, 1.0, size=dim) * std)
            res = evaluator.evaluate_candidate(
                cand_theta, f"{run_id}:g{gen:02d}:m{idx:02d}", "hard"
            )
            eval_count += 1
            if res.valid and _better_hard_candidate(res, best_res, resolved_sel):
                best_res = res
                best_theta = cand_theta.copy()
            rank = B._hard_rank(res, resolved_sel)
            ranks.append(rank if rank is not None else (2.0, 1e6, 1e6, 1e6, 1e6))
            samples.append(cand_theta)
            history.append(
                {
                    "eval": float(eval_count),
                    "objective": float(res.objective if res.objective is not None else 1e3),
                    "best_objective": float(best_res.objective or 0.0),
                    "cycle_time_s": float(res.components.get("cycle_time_s", 0.0)),
                    "best_cycle_time_s": float(best_res.components.get("cycle_time_s", 0.0)),
                    "peak_mrr_mm3_s": float(res.components.get("peak_mrr_mm3_s", 0.0)),
                    "feasible": float(res.components.get("feasible", 0.0)),
                }
            )

        n_elite = max(2, int(math.ceil(elite_frac * len(samples))))
        if len(samples) >= 2:
            order = sorted(range(len(samples)), key=lambda i: ranks[i])[:n_elite]
            elites = np.stack([samples[i] for i in order], axis=0)
            new_mean = np.mean(elites, axis=0)
            new_std = np.maximum(np.std(elites, axis=0), 0.02)
            mean = evaluator.parameterization.clip(smoothing * new_mean + (1.0 - smoothing) * mean)
            std = smoothing * new_std + (1.0 - smoothing) * std
        gen += 1

    snap = evaluator.ledger.snapshot()
    return SolverRunSummary(
        method="Hard-CEM",
        run_id=run_id,
        seed=seed,
        evaluations=int(snap["candidate_evaluations"]),
        forward_calls=int(snap["forward_calls"]),
        backward_calls=int(snap["backward_calls"]),
        initial_result=init_res,
        best_result=best_res,
        best_theta=best_theta,
        history=history,
    )


def train_multitask_policy(
    task_contexts: Sequence[Tuple[O.CandidateEvaluator, np.ndarray]],
    *,
    mode: str,
    epochs: int = 18,
    lr: float = 4e-3,
    hidden_dim: int = 64,
    seed: int = 0,
) -> Tuple[ResidualMillingPolicy, List[Dict[str, float]]]:
    """Multi-task training across ``dev`` tasks for Exp3 zero-shot & few-shot transfer.

    ``mode == 'shac'``: analytical STE policy gradient averaged across dev tasks per epoch.
    ``mode == 'ppo'``: zero-order Clipped PPO + GAE across dev-task rollouts per epoch.
    """
    if mode not in ("shac", "ppo"):
        raise ValueError("mode must be 'shac' or 'ppo'")
    rng = np.random.default_rng(seed)
    state_dim = task_contexts[0][1].shape[1]
    policy = ResidualMillingPolicy(state_dim, hidden_dim, ACTION_DIM, seed=seed, init_log_std=-1.4)
    params = policy.get_flat_params()
    opt = _AdamUpdater(params.size, lr=lr)
    critic = ValueCritic(state_dim, hidden_dim, seed=seed + 77)
    critic_opt = _AdamUpdater(critic.get_flat_params().size, lr=5e-3)

    curve: List[Dict[str, float]] = []
    for ep in range(epochs):
        frac = ep / max(1, epochs - 1)
        k_cur = 25.0 + 60.0 * frac
        if mode == "shac":
            grad_accum = np.zeros_like(params)
            objs: List[float] = []
            feas_list: List[float] = []
            for idx, (ev, st) in enumerate(task_contexts):
                ev.set_evaluation_k(k_cur)
                mu, cache = policy.forward_with_cache(st)
                u_vec = ev.parameterization.clip(mu.reshape(-1))
                ste = ev.evaluate_ste_with_gradient(u_vec, f"mt-shac-e{ep:02d}-t{idx:02d}")
                ev.clear_evaluation_k()
                if ste.result.valid and ste.result.objective is not None:
                    objs.append(float(ste.result.objective))
                    feas_list.append(float(ste.result.components.get("feasible", 0.0)))
                if ste.gradient_theta is not None and np.all(np.isfinite(ste.gradient_theta)):
                    g_mu = _precondition_policy_action_grad(
                        ste.gradient_theta.reshape(st.shape[0], ACTION_DIM)
                    ) + 1e-4 * mu
                    grad_accum += policy.backward(cache, g_mu) / len(task_contexts)
            lr_scale = max(0.2, 0.5 * (1.0 + math.cos(math.pi * frac)))
            params = opt.step(params, grad_accum, lr_scale=lr_scale, max_norm=3.0)
            policy.set_flat_params(params)
            curve.append(
                {
                    "epoch": float(ep),
                    "mean_dev_objective": float(np.mean(objs)) if objs else 1e3,
                    "mean_dev_feasible": float(np.mean(feas_list)) if feas_list else 0.0,
                }
            )
        else:
            S_list, A_list, LP_list, ADV_list, RET_list = [], [], [], [], []
            objs = []
            feas_list = []
            for idx, (ev, st) in enumerate(task_contexts):
                u_bound = ev.parameterization.u_bound
                vals, _ = critic.forward_with_cache(st)
                # 1 trajectory sample per task so evaluation budget matches SHAC epoch-for-epoch!
                u_mat, _, old_lp = policy.sample(st, rng, u_bound=u_bound)
                u_flat = u_mat.reshape(-1)
                res = ev.evaluate_candidate(u_flat, f"mt-ppo-e{ep:02d}-t{idx:02d}", "hard")
                if res.valid and res.objective is not None:
                    objs.append(float(res.objective))
                    feas_list.append(float(res.components.get("feasible", 0.0)))
                rews = _per_segment_hard_rewards(ev, u_flat, res)
                adv = np.zeros(st.shape[0], dtype=np.float64)
                gae = 0.0
                for t in range(st.shape[0] - 1, -1, -1):
                    nv = vals[t + 1] if t + 1 < st.shape[0] else 0.0
                    delta = rews[t] + 0.99 * nv - vals[t]
                    gae = delta + 0.99 * 0.95 * gae
                    adv[t] = gae
                S_list.append(st)
                A_list.append(u_mat)
                LP_list.append(old_lp)
                ADV_list.append(adv)
                RET_list.append(adv + vals)

            S_all = np.vstack(S_list)
            A_all = np.vstack(A_list)
            LP_old = np.concatenate(LP_list)
            ADV = np.concatenate(ADV_list)
            RET = np.concatenate(RET_list)
            if np.std(ADV) > 1e-8:
                ADV = (ADV - np.mean(ADV)) / np.std(ADV)

            c_params = critic.get_flat_params()
            for _ in range(4):
                lp_new, p_cache, std = policy.log_prob_and_grads(S_all, A_all)
                ratio = np.exp(np.clip(lp_new - LP_old, -10.0, 10.0))
                use_unclipped = (
                    ((ADV >= 0) & (ratio <= 1.2)) | ((ADV < 0) & (ratio >= 0.8))
                ).astype(np.float64)
                d_lp = -(use_unclipped * ratio * ADV) / max(1, S_all.shape[0])
                grad_mu = d_lp[:, None] * (p_cache["diff"] / std[None, :])
                params = opt.step(params, policy.backward(p_cache, grad_mu), max_norm=0.5)
                policy.set_flat_params(params)

                vp, c_cache = critic.forward_with_cache(S_all)
                c_params = critic_opt.step(
                    c_params,
                    critic.backward(c_cache, 2.0 * (vp - RET) / max(1, S_all.shape[0])),
                    max_norm=1.0,
                )
                critic.set_flat_params(c_params)

            curve.append(
                {
                    "epoch": float(ep),
                    "mean_dev_objective": float(np.mean(objs)) if objs else 1e3,
                    "mean_dev_feasible": float(np.mean(feas_list)) if feas_list else 0.0,
                }
            )
    return policy, curve


__all__ = [
    "STATE_DIM",
    "ACTION_DIM",
    "DirectSegmentActionMap",
    "extract_segment_states",
    "ResidualMillingPolicy",
    "ValueCritic",
    "SolverRunSummary",
    "train_diff_policy_shac",
    "train_hard_ppo",
    "solve_ste_continuation_adam",
    "solve_hard_cem",
    "train_multitask_policy",
]
