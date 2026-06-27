"""Tests for the cam G-code export / G-code -> trajectory round trip.

Success measure: the trajectory similarity between an original trajectory and the
trajectory produced by exporting it to G-code and re-planning ("executing") that
G-code. Exact-stop (G61) planning means the executed path passes through every
original waypoint, so similarity should be near machine precision. A secondary
end-to-end check carves both trajectories in the simulator and compares the
resulting stock.
"""

import os

import numpy as np
import pytest

from cam import (
    MachineConfig,
    trajectory_to_gcode,
    save_gcode,
    parse_gcode,
    segment_waypoints,
    plan_trajectory,
    gcode_to_trajectory,
    discrete_frechet,
    dtw_distance,
    resampled_rmse,
    waypoint_roundtrip_error,
)
from cam.gcode_parser import PLANE_XY

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)
REAL_TRAJ = os.path.join(REPO, "trajectory.npy")


# ---------------------------------------------------------------------------
# Sample trajectories
# ---------------------------------------------------------------------------

def _line():
    return np.array(
        [[0.1, 0.1, 0.5], [0.9, 0.5, 0.5]], dtype=np.float64
    )


def _square():
    return np.array(
        [
            [0.2, 0.2, 0.5],
            [0.8, 0.2, 0.5],
            [0.8, 0.8, 0.5],
            [0.2, 0.8, 0.5],
            [0.2, 0.2, 0.5],
        ],
        dtype=np.float64,
    )


def _zigzag():
    xs = np.linspace(0.1, 0.9, 12)
    ys = np.where(np.arange(12) % 2 == 0, 0.2, 0.8)
    zs = np.full(12, 0.5)
    return np.stack([xs, ys, zs], axis=1).astype(np.float64)


def _circle_polyline(n=40):
    t = np.linspace(0, 2 * np.pi, n)
    x = 0.5 + 0.3 * np.cos(t)
    y = 0.5 + 0.3 * np.sin(t)
    z = np.full(n, 0.5)
    return np.stack([x, y, z], axis=1).astype(np.float64)


ALL_PATHS = {
    "line": _line(),
    "square": _square(),
    "zigzag": _zigzag(),
    "circle": _circle_polyline(),
}


# ---------------------------------------------------------------------------
# 1. Export format
# ---------------------------------------------------------------------------

def test_export_format():
    cfg = MachineConfig()
    g = trajectory_to_gcode(_square(), cfg)
    lines = [ln for ln in g.splitlines() if ln and not ln.startswith("(")]

    assert "G21" in lines          # mm
    assert "G90" in lines          # absolute
    assert "G61" in lines          # exact stop
    assert lines[-1] == "M2"       # program end

    g0 = [ln for ln in lines if ln.startswith("G0")]
    g1 = [ln for ln in lines if ln.startswith("G1")]
    assert len(g0) == 1            # single rapid to the start
    assert len(g1) == len(_square()) - 1
    assert any(ln.startswith("F") for ln in lines)


def test_export_units_inch():
    cfg = MachineConfig(units="inch")
    g = trajectory_to_gcode(_line(), cfg)
    assert "G20" in g and "G21" not in g


def test_export_rejects_bad_shape():
    with pytest.raises(ValueError):
        trajectory_to_gcode(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        trajectory_to_gcode(np.zeros((0, 3)))


# ---------------------------------------------------------------------------
# 2. Parser round-trip (waypoint recovery)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ALL_PATHS))
def test_waypoint_roundtrip(name):
    cfg = MachineConfig()
    P = ALL_PATHS[name]
    segs = parse_gcode(trajectory_to_gcode(P, cfg), cfg)
    wp = segment_waypoints(segs)
    assert wp.shape == P.shape
    err_mm = waypoint_roundtrip_error(P, wp, cfg.workspace_mm)
    # Limited only by the G-code coordinate precision.
    assert err_mm < 1e-4


def test_save_gcode_file(tmp_path):
    cfg = MachineConfig()
    path = tmp_path / "prog.ngc"
    save_gcode(_square(), str(path), cfg)
    assert path.exists()
    text = path.read_text()
    segs = parse_gcode(text, cfg)
    assert segment_waypoints(segs).shape == _square().shape


# ---------------------------------------------------------------------------
# 3. Arc parsing (parser generality vs LinuxCNC G2/G3)
# ---------------------------------------------------------------------------

def test_arc_ccw_offsets_on_circle():
    cfg = MachineConfig()
    # Quarter circle CCW, radius 10 mm about (50, 50) mm -> 0.1 about (0.5,0.5).
    gc = "G21\nG90\nG17\nG0 X60 Y50 Z0\nG3 X50 Y60 I-10 J0\nM2\n"
    segs = parse_gcode(gc, cfg)
    assert len(segs) == 1 and segs[0].kind == "arc" and not segs[0].cw
    assert segs[0].plane == PLANE_XY

    pts, _ = plan_trajectory(segs, cfg)
    center = np.array([0.5, 0.5])
    radii = np.linalg.norm(pts[:, :2] - center, axis=1)
    assert np.allclose(radii, 0.1, atol=1e-4)
    assert np.allclose(pts[0], [0.6, 0.5, 0.0], atol=1e-6)
    assert np.allclose(pts[-1], [0.5, 0.6, 0.0], atol=1e-6)


def test_arc_radius_word():
    cfg = MachineConfig()
    gc = "G21\nG90\nG17\nG0 X60 Y50 Z0\nG2 X50 Y40 R10\nM2\n"
    segs = parse_gcode(gc, cfg)
    assert len(segs) == 1 and segs[0].cw
    pts, _ = plan_trajectory(segs, cfg)
    radii = np.linalg.norm(pts[:, :2] - np.array([0.5, 0.5]), axis=1)
    assert np.allclose(radii, 0.1, atol=1e-3)


# ---------------------------------------------------------------------------
# 4. End-to-end geometric trajectory similarity (primary success measure)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ALL_PATHS))
def test_roundtrip_similarity_synthetic(name):
    cfg = MachineConfig()
    P = ALL_PATHS[name]
    executed, times = gcode_to_trajectory(trajectory_to_gcode(P, cfg), cfg)

    scale = cfg.workspace_mm
    # Exact-stop planning => executed points lie on the original segments.
    assert discrete_frechet(P, executed, scale) < 1e-2     # mm
    assert resampled_rmse(P, executed, scale=scale) < 1e-3  # mm
    assert dtw_distance(P, executed, scale) < 1e-2          # mm (mean)
    # Executed trajectory begins exactly at the first waypoint.
    assert np.allclose(executed[0], P[0], atol=1e-6)
    assert np.allclose(executed[-1], P[-1], atol=1e-6)


def test_roundtrip_similarity_real_trajectory():
    cfg = MachineConfig()
    P = np.load(REAL_TRAJ).astype(np.float64)
    executed, _ = gcode_to_trajectory(trajectory_to_gcode(P, cfg), cfg)

    scale = cfg.workspace_mm
    frechet = discrete_frechet(P, executed, scale)
    rmse = resampled_rmse(P, executed, scale=scale)
    # Sub-micron agreement on a real 64-point optimized trajectory.
    assert frechet < 1e-2, f"frechet={frechet} mm"
    assert rmse < 1e-3, f"rmse={rmse} mm"


# ---------------------------------------------------------------------------
# 5. Velocity profile sanity (trapezoidal, exact-stop, accel-limited)
# ---------------------------------------------------------------------------

def test_velocity_profile_trapezoidal():
    # Fine sampling so discrete endpoint speeds genuinely approach zero.
    cfg = MachineConfig(dt=0.002)
    # Single long straight feed move so a cruise phase exists.
    P = np.array([[0.05, 0.5, 0.5], [0.95, 0.5, 0.5]], dtype=np.float64)
    pts, times = plan_trajectory(parse_gcode(trajectory_to_gcode(P, cfg), cfg), cfg)

    assert np.all(np.diff(times) > 0)                       # monotonic time

    step = np.diff(pts, axis=0) * cfg.workspace_mm          # mm
    dt = np.diff(times)
    speed = np.linalg.norm(step, axis=1) / dt               # mm/s

    vmax = cfg.feed_mm_per_s
    assert speed.max() <= vmax * 1.02                       # never exceeds feed
    assert speed.max() > 0.5 * vmax                         # actually cruises
    # Exact stop: ramps up from near rest and back down.
    assert speed[0] < 0.1 * vmax
    assert speed[-1] < 0.1 * vmax
    assert speed[len(speed) // 2] > speed[0]               # ramps up

    accel = np.diff(speed) / dt[1:]
    assert np.max(np.abs(accel)) <= cfg.max_accel * 1.5     # accel-limited


def test_exact_stop_at_each_waypoint():
    cfg = MachineConfig()
    P = _zigzag()
    pts, times = plan_trajectory(parse_gcode(trajectory_to_gcode(P, cfg), cfg), cfg)
    dt = np.diff(times)
    speed = np.linalg.norm(np.diff(pts, axis=0) * cfg.workspace_mm, axis=1) / dt
    vmax = cfg.feed_mm_per_s
    # Each interior waypoint should produce a near-zero-speed sample.
    near_zero = np.sum(speed < 0.1 * vmax)
    assert near_zero >= len(P) - 1


# ---------------------------------------------------------------------------
# 6. End-to-end carved-stock similarity (secondary success measure)
# ---------------------------------------------------------------------------

def test_carved_stock_matches_real_trajectory():
    # Heavy: imports Taichi and carves twice. Coarse dt keeps the executed
    # trajectory modest; the hard CSG carve is step-count invariant regardless.
    from cam.sim_exec import carve_stock
    from simulator.csg_metrics import sdf_to_mask, dice_score

    cfg = MachineConfig(dt=0.1)
    P = np.load(REAL_TRAJ).astype(np.float64)
    executed, _ = gcode_to_trajectory(trajectory_to_gcode(P, cfg), cfg)

    stock_orig = carve_stock(P, resolution=24, target_shape="sphere")
    mask_orig = sdf_to_mask(stock_orig)

    stock_exec = carve_stock(executed, resolution=24, target_shape="sphere")
    mask_exec = sdf_to_mask(stock_exec)

    dice = dice_score(mask_orig, mask_exec)
    assert dice > 0.97, f"carved-stock dice={dice}"
