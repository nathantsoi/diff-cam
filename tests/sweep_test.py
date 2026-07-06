"""Tests for the one-shot swept-volume carve (simulator/sweep.py).

Success measures:
1. The B-spline basis is a partition of unity and interpolates its endpoints
   (so pinning P[0] pins the sampled path start).
2. The swept carve max(stock0, -min_s seg_sdf) equals the simulator's
   sequential hard carve (forward_hard) voxel-for-voxel — material removal is
   order-independent, so the one-shot union must reproduce the evaluator's
   geometry exactly when the speed clip is inert.
3. Taichi autodiff through the argmin pass matches central finite differences
   for meaningful-magnitude gradients.
"""

import numpy as np
import pytest
import taichi as ti

from simulator.sweep import (SweepCarve, bspline_basis, fit_control_points,
                             init_reference_path)


def _make_sim_and_path(T=24, seed=0):
    from simulator.csg_simulator import CSGSimulatorDelta
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    sim = CSGSimulatorDelta(max_steps=T, target_shape="sphere", init_taichi=False,
                            stock_size_in=(1.0, 1.0, 1.0), voxel_size_mm=1.0,
                            dt=0.45)
    sim.set_target_params(radius_mm=11.43)
    sim.tool_radius[None] = 3.175
    sim.tool_height[None] = 25.0
    sim.bake_target_grid()
    rng = np.random.default_rng(seed)
    start = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    # Steps well under the 1.9 mm feed cap at dt=0.45 so the clip is inert.
    steps = rng.uniform(-0.03, 0.03, size=(T - 1, 3)).astype(np.float32)
    steps[:, 2] -= 0.015
    X = np.vstack([start, start + np.cumsum(steps, axis=0)]).astype(np.float32)
    return sim, np.clip(X, 0.02, 1.05)


def test_bspline_basis_partition_and_endpoints():
    B = bspline_basis(12, 64)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-5)
    P = np.random.default_rng(0).random((12, 3)).astype(np.float32)
    X = B @ P
    assert np.allclose(X[0], P[0], atol=1e-5)
    assert np.allclose(X[-1], P[-1], atol=1e-5)


def test_swept_carve_matches_sequential_hard_carve():
    T = 24
    sim, X = _make_sim_and_path(T)
    full = np.zeros((T, 3), dtype=np.float32)
    full[:T - 1] = np.diff(X, axis=0)
    sim.tool_start[None] = ti.Vector([float(X[0, 0]), float(X[0, 1]), float(X[0, 2])])
    sim.tool_delta.from_numpy(full)
    sim.forward_hard(T)
    seq_mask = sim.stock.to_numpy()[T - 1] < 0.0
    # Speed clip must have been inert for the comparison to be meaningful.
    assert np.abs(sim.tool_pos.to_numpy()[:T] - X).max() < 1e-5

    sweep = SweepCarve(sim, n_points=T)
    swept_mask = sweep.hard_carve_mask(X)
    assert (seq_mask != swept_mask).sum() == 0


def test_gradient_matches_finite_differences():
    T = 24
    sim, X = _make_sim_and_path(T)
    sweep = SweepCarve(sim, n_points=T)
    _, gX = sweep.loss_and_grad(X)
    eps = 2e-3
    rng = np.random.default_rng(1)
    ok = 0
    for _ in range(6):
        t = int(rng.integers(1, T))
        c = int(rng.integers(0, 3))
        losses = []
        for sign in (+1, -1):
            Xp = X.copy()
            Xp[t, c] += sign * eps
            sweep.path.from_numpy(Xp)
            sweep.find_argmin(T - 1)
            sweep.loss[None] = 0.0
            sweep.compute_loss()
            losses.append(float(sweep.loss[None]))
        fd = (losses[0] - losses[1]) / (2 * eps)
        g = float(gX[t, c])
        rel = abs(fd - g) / max(1e-8, abs(fd) + abs(g))
        # f32 loss + hard-min subgradients: below ~2e-4 the central difference
        # is rounding noise / argmin-reassignment kinks.
        if rel < 0.05 or (abs(fd) < 2e-4 and abs(g) < 2e-4):
            ok += 1
    assert ok >= 5


def test_init_fit_starts_at_tool_start():
    T = 64
    sim, _ = _make_sim_and_path(T)
    start = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    B = bspline_basis(16, T)
    for mode in ("raster", "helix", "random"):
        ref = init_reference_path(sim, start, T, mode=mode)
        P = fit_control_points(B, ref)
        P[0] = start
        X = B @ P
        assert np.allclose(X[0], start, atol=1e-5)
