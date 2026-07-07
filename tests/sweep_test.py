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
    for mode in ("raster", "raster_arc", "helix", "random"):
        ref = init_reference_path(sim, start, T, mode=mode)
        P = fit_control_points(B, ref)
        P[0] = start
        X = B @ P
        assert np.allclose(X[0], start, atol=1e-5)


def test_raster_arc_steps_uniform_and_capped():
    """Arc-length resampling: every physical step equals total_len/(T-1) up to
    corner-chord shortening — never longer — so choosing T >= len/cap + 1
    makes the whole init feed-feasible by construction."""
    from simulator.sweep import raster_arc_waypoints, _resample_arc_length
    sim, _ = _make_sim_and_path(24)
    start = (0.5, 0.5, 1.0)
    wps, total_len = raster_arc_waypoints(sim, start)
    L_mm = np.array([sim.Lx, sim.Ly, sim.Lz])
    T = int(np.ceil(total_len / 1.905)) + 1
    X = _resample_arc_length(wps, L_mm, T)
    step = np.linalg.norm(np.diff(X, axis=0) * L_mm, axis=1)
    ds = total_len / (T - 1)
    assert step.max() <= ds + 1e-6          # never exceeds the uniform pitch
    assert step.min() > 0.0                 # corner/U-turn chords only shorten
    assert np.allclose(X[0], start, atol=1e-6)


def test_stale_argmin_matches_exact_for_small_motion():
    """amin-refresh: with an unchanged path the cached argmin gives the exact
    loss; after a sub-voxel perturbation the stale loss stays close (the
    winner index is stable under small motion)."""
    sim, X = _make_sim_and_path(T=24)
    sweep = SweepCarve(sim, n_points=X.shape[0])
    l_exact, g_exact = sweep.loss_and_grad(X)
    l_stale, g_stale = sweep.loss_and_grad(X, refresh_argmin=False)
    assert np.isclose(l_exact, l_stale, rtol=1e-6)
    assert np.allclose(g_exact, g_stale, rtol=1e-5, atol=1e-8)
    X2 = X + np.float32(1e-4)               # ~0.003 voxels at 32^3
    l2_exact, _ = sweep.loss_and_grad(X2)
    l2_stale, _ = sweep.loss_and_grad(X2, refresh_argmin=False)
    assert np.isclose(l2_exact, l2_stale, rtol=1e-3)


def test_raster_terrain_init_carve_is_gouge_free():
    """The terrain-following init climbs over part (legal tool-base height),
    so hard-carving the raw resampled polyline must remove (essentially) no
    part voxels — the whole point of the mode vs plain raster_arc."""
    from simulator.sweep import init_reference_path
    sim, _ = _make_sim_and_path(T=24)
    n = 600
    X = init_reference_path(sim, (0.5, 0.5, 1.0), n, mode="raster_terrain")
    sweep = SweepCarve(sim, n_points=n)
    remaining = sweep.hard_carve_mask(X.astype(np.float32))
    part = sim.target.to_numpy() <= 0.0
    gouged = int((part & ~remaining).sum())
    assert gouged <= 0.002 * part.sum(), gouged


def test_raster_arc_covers_bbox_footprint():
    """The serpentine must reach all four footprint corners of the target bbox
    and descend to its bottom (coverage precondition for any carve)."""
    from simulator.sweep import raster_arc_waypoints, target_bbox
    sim, _ = _make_sim_and_path(24)
    wps, _ = raster_arc_waypoints(sim, (0.5, 0.5, 1.0))
    lo, hi = target_bbox(sim)
    body = wps[2:]                          # skip start + lead-in
    for d in range(2):
        assert body[:, d].min() <= lo[d] + 1e-6
        assert body[:, d].max() >= hi[d] - 1e-6
    assert body[:, 2].min() <= lo[2] + 1e-6
