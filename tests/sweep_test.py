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


# ---------------------------------------------------------------------------
# Physical-plausibility terms (jul13-phys-plausible): sequential chip
# attribution, force penalties, fragility field, ramped layer entry.
# ---------------------------------------------------------------------------

def _seg_sdf_np(P, a, b, r_vox, h_vox):
    """Numpy mirror of SweepCarve._seg_sdf (voxel space) for brute-force checks."""
    pa = P[:, :2] - a[:2]
    ba = b[:2] - a[:2]
    denom = float(ba @ ba) + 1e-12
    t = np.clip((pa @ ba) / denom, 0.0, 1.0)
    closest = a[:2] + np.outer(t, ba)
    d_xy = np.sqrt(((P[:, :2] - closest) ** 2).sum(1) + 1e-8) - r_vox
    z_base = a[2] + (b[2] - a[2]) * t
    z_center = z_base + 0.5 * h_vox
    d_z = np.sqrt((P[:, 2] - z_center) ** 2 + 1e-8) - 0.5 * h_vox
    outside = np.sqrt(np.maximum(d_xy, 0) ** 2 + np.maximum(d_z, 0) ** 2 + 1e-8)
    inside = -np.maximum(-np.maximum(d_xy, d_z), 0.0)
    return outside + inside


def test_cut_seg_is_first_covering_segment():
    T = 24
    sim, X = _make_sim_and_path(T)
    sweep = SweepCarve(sim, n_points=T)
    sweep.path.from_numpy(X)
    sweep.find_argmin(T - 1)
    cut = sweep.cut_seg.to_numpy()
    N = np.array([sweep.Nx, sweep.Ny, sweep.Nz])
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in N], indexing="ij")
    P = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3) + 0.5
    r_vox = sim.tool_radius[None] / sim.v
    h_vox = sim.tool_height[None] / sim.v
    Xv = X * N  # voxel-space path
    first = np.full(P.shape[0], -1, dtype=np.int32)
    for s in range(T - 1):
        d = _seg_sdf_np(P, Xv[s], Xv[s + 1], r_vox, h_vox)
        hit = (d < 0.0) & (first == -1)
        first[hit] = s
    half = 0.5 * N
    stock0 = np.max(np.abs(P - half) - half, axis=1)
    first[stock0 >= 0.0] = -1
    assert (first.reshape(cut.shape) == cut).all()


def test_seg_chip_tracks_sequential_removal():
    """Per-segment soft chip vs brute-force sequential hard removal (the
    voxels each segment covers FIRST). The soft volume is biased low (surface
    sigmoid, one-sided attribution band) — a calibratable scale factor — but
    it must (a) sit in a sane band of the hard truth, (b) rank segments
    consistently (heavy vs idle), (c) be ~zero on segments that remove
    nothing."""
    T = 24
    sim, X = _make_sim_and_path(T)
    sweep = SweepCarve(sim, n_points=T)
    sweep.w_force[None] = 1.0  # activate physics kernels
    sweep.loss_and_grad(X)
    chip = sweep.seg_chip_np.astype(np.float64)
    assert (chip >= 0).all()
    # Brute-force sequential hard chips from the cut_seg attribution already
    # verified exact above.
    cut = sweep.cut_seg.to_numpy()
    hard = np.bincount(cut[cut >= 0].ravel(), minlength=T - 1) * sim.v ** 3
    total_ratio = chip.sum() / max(hard.sum(), 1e-9)
    assert 0.4 < total_ratio <= 1.1, total_ratio
    idle = hard == 0
    if idle.any():
        assert chip[idle].max() < 0.1 * max(chip.max(), 1e-9)
    if (~idle).sum() >= 3:
        c = np.corrcoef(chip[~idle], hard[~idle])[0, 1]
        assert c > 0.9, c


def test_force_penalty_gradient_matches_fd():
    T = 24
    sim, X = _make_sim_and_path(T)
    sweep = SweepCarve(sim, n_points=T)
    sweep.w_force[None] = 5.0
    sweep.f_cap[None] = 2.0   # tiny cap so the penalty is active
    _, gX = sweep.loss_and_grad(X)

    def loss_at(Xp):
        # FIXED attribution (no find_argmin): the FD probes the surrogate the
        # optimizer actually descends between refreshes. Refreshing inside the
        # FD would cross first-cover attribution flips, which are genuine O(1)
        # handoffs of boundary voxels between segments (bounded jumps absorbed
        # by the amin-refresh cadence, not part of the smooth surrogate).
        S = T - 1
        sweep.path.from_numpy(Xp)
        sweep.loss[None] = 0.0
        sweep.compute_loss()
        sweep.zero_seg_chip(S)
        sweep.accum_seg_chip()
        sweep.add_force_penalties(S)
        return float(sweep.loss[None])

    eps = 2e-3
    rng = np.random.default_rng(3)
    ok = 0
    for _ in range(6):
        t = int(rng.integers(1, T))
        c = int(rng.integers(0, 3))
        Xp, Xm = X.copy(), X.copy()
        Xp[t, c] += eps
        Xm[t, c] -= eps
        fd = (loss_at(Xp) - loss_at(Xm)) / (2 * eps)
        g = float(gX[t, c])
        rel = abs(fd - g) / max(1e-8, abs(fd) + abs(g))
        if rel < 0.05 or (abs(fd) < 2e-4 and abs(g) < 2e-4):
            ok += 1
    assert ok >= 5


def test_fragility_detects_pin():
    from utils.fragility import compute_fragility, F_ALLOW_SAFE
    Nx, Ny, Nz = 40, 40, 44
    occ = np.zeros((Nx, Ny, Nz), bool)
    occ[:, :, :20] = True                      # plate: 10 mm thick at 0.5 mm/vox
    xx, yy = np.mgrid[:Nx, :Ny]
    pin = (xx - 20) ** 2 + (yy - 20) ** 2 <= 2.0 ** 2
    occ[pin, 20:36] = True                     # pin: r=1 mm, h=8 mm
    sdf = np.where(occ, -1.0, 1.0).astype(np.float32)
    frag = compute_fragility(sdf, voxel_mm=0.5, sigma_y_mpa=10.0,
                             tool_radius_mm=3.175, contact_mm=1.0)
    feats = frag["features"]
    assert len(feats) == 1, feats
    f = feats[0]
    # t ~ 2 mm, h ~ 8 mm -> F = 10 * 2^3 / (6 * 8) = 1.67 N (edt quantization slack)
    assert 0.8 < f["f_allow_n"] < 3.5, f
    # waste voxel right beside the pin top carries the cap; far corner is safe
    assert frag["f_allow_vox"][23, 20, 30] < 5.0
    assert frag["f_allow_vox"][2, 2, 40] == F_ALLOW_SAFE
    # tool-center field is capped anywhere the cutter overlaps the pin band
    assert frag["f_allow_tool"][26, 20, 30] < 5.0


def test_ramp_entry_has_no_engaged_plunges():
    """With ramp_deg on, every descending step below the stock top moves
    laterally at >= the ramp slope (plain mode: exact by construction)."""
    from simulator.sweep import raster_arc_waypoints
    sim, _ = _make_sim_and_path(24)
    wps, _ = raster_arc_waypoints(sim, (0.5, 0.5, 1.0), ramp_deg=3.0)
    L = np.array([sim.Lx, sim.Ly, sim.Lz])
    seg = np.diff(np.asarray(wps), axis=0) * L
    below = np.minimum(wps[:-1, 2], wps[1:, 2]) < 1.0  # touches material zone
    dz, dxy = seg[:, 2], np.linalg.norm(seg[:, :2], axis=1)
    desc = below & (dz < -1e-9)
    tan3 = np.tan(np.radians(3.0))
    assert (-dz[desc] <= tan3 * dxy[desc] + 1e-6).all(), (
        -dz[desc] - tan3 * dxy[desc]).max()
