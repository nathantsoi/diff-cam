"""Plan a time-sampled trajectory from parsed G-code segments.

This is the diff-cam analogue of LinuxCNC's trajectory planner (``tp.c``): it
turns geometric segments into time-parameterised motion. Each segment runs an
acceleration-limited **trapezoidal velocity profile** and, in the exact-stop
(G61) mode used here, decelerates to a full stop at every waypoint. The tool
therefore passes exactly through each original point, sampled at the configured
servo period ``dt``.

Output is an ``(M, 3)`` array of unit-cube positions plus an ``(M,)`` array of
times (seconds). ``M`` generally differs from the number of waypoints, so the
trajectory-similarity metrics in ``cam.trajectory_metrics`` are
parameterisation-invariant.
"""

import math

import numpy as np

from benchmarks.contracts import (
    Frame,
    MotionPlan,
    TaskSpec,
    TimingRule,
    arc_geometry,
)
from .config import MachineConfig
from .gcode_parser import parse_gcode, _plane_axes, _typed_plane_axes


def _trapezoid_distances(length, vmax, accel, dt):
    """Return sample distances ``s`` (0..length) and their times for an
    acceleration-limited trapezoidal profile that starts and ends at rest."""
    if length <= 1e-12:
        return np.array([0.0]), np.array([0.0])
    vmax = max(vmax, 1e-9)
    accel = max(accel, 1e-9)

    d_acc = 0.5 * vmax * vmax / accel       # distance to reach vmax
    if 2.0 * d_acc >= length:
        # Triangular profile: never reaches vmax.
        t_peak = np.sqrt(length / accel)
        total = 2.0 * t_peak
        t_acc = t_peak
        t_cruise = 0.0
        d_acc = 0.5 * length
        d_cruise = 0.0
    else:
        t_acc = vmax / accel
        d_cruise = length - 2.0 * d_acc
        t_cruise = d_cruise / vmax
        total = 2.0 * t_acc + t_cruise

    # Sample times: 0, dt, 2dt, ..., and the exact end time.
    n = max(int(np.ceil(total / dt)), 1)
    times = np.arange(n) * dt
    times = times[times < total]
    times = np.append(times, total)

    s = np.empty_like(times)
    for idx, t in enumerate(times):
        if t <= t_acc:
            s[idx] = 0.5 * accel * t * t
        elif t <= t_acc + t_cruise:
            s[idx] = d_acc + vmax * (t - t_acc)
        else:
            td = t - (t_acc + t_cruise)
            s[idx] = d_acc + d_cruise + vmax * td - 0.5 * accel * td * td
    s = np.clip(s, 0.0, length)
    s[-1] = length
    return s, times


def _sample_linear(seg, config):
    start = np.asarray(seg.start, dtype=np.float64)
    end = np.asarray(seg.end, dtype=np.float64)
    length_mm = np.linalg.norm((end - start) * config.stock_size_vec)
    vmax = config.rapid_mm_per_s if seg.kind == "rapid" else config.feed_mm_per_s
    s, times = _trapezoid_distances(length_mm, vmax, config.max_accel, config.dt)
    if length_mm <= 1e-12:
        return end[None, :].copy(), times
    frac = (s / length_mm)[:, None]
    pts = start[None, :] + frac * (end - start)[None, :]
    return pts, times


def _sample_arc(seg, config):
    a0, a1, ax = _plane_axes(seg.plane)
    start = np.asarray(seg.start, dtype=np.float64)
    end = np.asarray(seg.end, dtype=np.float64)
    center = np.asarray(seg.center, dtype=np.float64)

    r0 = start[[a0, a1]] - center[[a0, a1]]
    r1 = end[[a0, a1]] - center[[a0, a1]]
    radius = np.linalg.norm(r0)

    ang0 = np.arctan2(r0[1], r0[0])
    ang1 = np.arctan2(r1[1], r1[0])
    sweep = ang1 - ang0
    # Normalise sweep to the correct direction. CCW (G3) is positive.
    if seg.cw:
        while sweep >= 0:
            sweep -= 2.0 * np.pi
        # full circle when start == end
        if abs(sweep) < 1e-9:
            sweep = -2.0 * np.pi
    else:
        while sweep <= 0:
            sweep += 2.0 * np.pi
        if abs(sweep) < 1e-9:
            sweep = 2.0 * np.pi

    # Arcs are circular in normalized coords; under an anisotropic stock box they
    # warp to ellipses. Approximate the physical arc length with the in-plane
    # axes' mean scale (exact for a cubic/near-cubic stock).
    a0i, a1i, _ = _plane_axes(seg.plane)
    ws = config.stock_size_vec
    plane_scale = 0.5 * (ws[a0i] + ws[a1i])
    out_delta = end[ax] - start[ax]
    plane_len = abs(radius * sweep) * plane_scale
    out_len = abs(out_delta) * ws[ax]
    length_mm = np.hypot(plane_len, out_len)

    vmax = config.feed_mm_per_s
    s, times = _trapezoid_distances(length_mm, vmax, config.max_accel, config.dt)
    if length_mm <= 1e-12:
        return end[None, :].copy(), times
    frac = s / length_mm

    pts = np.empty((len(frac), 3), dtype=np.float64)
    ang = ang0 + frac * sweep
    pts[:, a0] = center[a0] + radius * np.cos(ang)
    pts[:, a1] = center[a1] + radius * np.sin(ang)
    pts[:, ax] = start[ax] + frac * out_delta
    pts[-1] = end  # snap exact endpoint
    return pts, times


def plan_trajectory(segments, config: MachineConfig = MachineConfig()):
    """Plan a time-sampled ``(positions, times)`` trajectory from segments.

    Consecutive segments share an endpoint (exact stop), so the duplicated
    junction sample is dropped to keep a clean point sequence.
    """
    if not segments:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    all_pts = []
    all_times = []
    t_offset = 0.0
    for i, seg in enumerate(segments):
        if seg.kind == "arc":
            pts, times = _sample_arc(seg, config)
        else:
            pts, times = _sample_linear(seg, config)

        if i == 0:
            all_pts.append(pts)
            all_times.append(times + t_offset)
        else:
            # Drop the first sample (duplicate of previous segment's endpoint).
            all_pts.append(pts[1:])
            all_times.append(times[1:] + t_offset)
        t_offset += times[-1]

    positions = np.concatenate(all_pts, axis=0)
    time_arr = np.concatenate(all_times, axis=0)
    return positions, time_arr


def gcode_to_trajectory(text: str, config: MachineConfig = MachineConfig()):
    """Convenience: parse a G-code program and plan its executed trajectory."""
    segments = parse_gcode(text, config)
    return plan_trajectory(segments, config)


# ---------------------------------------------------------------------------
# Shared-contract exact-stop constant-speed sampling
# ---------------------------------------------------------------------------


def _sample_typed_segment(segment, local_times: np.ndarray) -> np.ndarray:
    start = np.asarray(segment.start_mm, dtype=np.float64)
    end = np.asarray(segment.end_mm, dtype=np.float64)
    if segment.duration_s == 0.0:
        fraction = np.zeros(1, dtype=np.float64)
        local_times = np.zeros(1, dtype=np.float64)
    else:
        fraction = local_times / segment.duration_s

    if segment.arc is None:
        return start[None, :] + fraction[:, None] * (end - start)[None, :]

    arc = segment.arc
    a, b, normal = _typed_plane_axes(arc.plane)
    centre = np.asarray(arc.centre_mm, dtype=np.float64)
    r0 = math.hypot(start[a] - centre[a], start[b] - centre[b])
    theta0 = math.atan2(start[b] - centre[b], start[a] - centre[a])
    geometry = arc_geometry(start, end, arc, segment.id)
    angles = theta0 + geometry["sweep_rad"] * fraction
    points = np.repeat(start[None, :], len(fraction), axis=0)
    points[:, a] = centre[a] + r0 * np.cos(angles)
    points[:, b] = centre[b] + r0 * np.sin(angles)
    points[:, normal] = start[normal] + fraction * (end[normal] - start[normal])
    points[-1] = end
    return points


def plan_motion_samples(
    plan: MotionPlan,
    task: TaskSpec,
    max_dt_s: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an exact-stop constant-speed MotionPlan in machine millimetres.

    Each non-zero segment uses linear time along its true chord/helix and includes
    its exact endpoint. At most ``max_dt_s`` elapses between samples. Junctions
    are exact stops, but duplicate junction samples are omitted from this numeric
    diagnostic export; evaluators must retain typed segments to model stationary
    cuts and per-segment chronology. No acceleration/blending profile is added.

    This deliberately does not alter the legacy normalized-array trapezoidal
    planner above, which remains available for backwards compatibility only.
    """
    if not np.isfinite(max_dt_s) or max_dt_s <= 0.0:
        raise ValueError("max_dt_s must be finite and positive")
    if plan.frame != Frame.MACHINE:
        raise ValueError("typed motion sampling requires a machine-frame MotionPlan")
    if plan.timing_rule != TimingRule.EXACT_STOP_CONSTANT_SPEED:
        raise ValueError("only exact_stop_constant_speed can use this sampler")
    if task.motion_semantics.timing_rule != plan.timing_rule:
        raise ValueError("plan and task timing rules differ")
    plan.validate_for_task(task)
    if not plan.segments:
        if plan.initial_state.known:
            return (
                np.asarray([plan.initial_state.position_mm], dtype=np.float64),
                np.zeros(1, dtype=np.float64),
            )
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

    all_points: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    offset = 0.0
    for index, segment in enumerate(plan.segments):
        intervals = max(1, int(np.ceil(segment.duration_s / max_dt_s)))
        local_times = (
            np.zeros(1, dtype=np.float64)
            if segment.duration_s == 0.0
            else np.linspace(0.0, segment.duration_s, intervals + 1)
        )
        points = _sample_typed_segment(segment, local_times)
        times = local_times + offset
        if index:
            points = points[1:]
            times = times[1:]
        all_points.append(points)
        all_times.append(times)
        offset += segment.duration_s
    return np.concatenate(all_points), np.concatenate(all_times)
