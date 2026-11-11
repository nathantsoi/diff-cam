# SPDX-License-Identifier: MIT
"""Clean-room MIT canonical 3-axis CAM toolpath generators across four strategy families.

Implements textbook geometric toolpath constructions:
1. ``facing_raster_seed``: Boustrophedon (zig-zag) stock-bounding-box facing with boss moat pass.
2. ``pocket_offset_seed``: Concentric outward offset-ring clearing from retained island to pocket wall.
3. ``adaptive_trochoidal_seed``: Chordal peel / trochoidal channel clearing + dual-island wall cleanup.
4. ``contour_offset_seed``: Outside-in multi-offset exterior boss contour profiling.

All generated MotionPlans operate in the declared machine frame, keep all RAPID segments strictly
at ``z_safe`` above ``stock_max_z`` (with vertical plunges/retracts emitted as FEED moves so
CAMotics v1.2.0 never flags rapid-cutter stock AABB overlap), and compute exact durations under
``task.motion_semantics.timing_rule`` (both ``EXACT_STOP_CONSTANT_SPEED`` and ``BLENDED_ACCEL_BOUNDED``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Sequence, Tuple

from benchmarks import contracts as C

Point3D = Tuple[float, float, float]


def _segments_to_motion_plan(
    task: C.TaskSpec,
    waypoints_work: Sequence[Tuple[C.SegmentKind, Point3D, float | None, str | None]],
    provenance: str,
) -> C.MotionPlan:
    """Convert work-frame (kind, end_xyz, feed_mm_s, tag) waypoints into a validated MotionPlan."""
    to_machine = task.work_to_machine.apply
    inv_machine = task.work_to_machine.inverse().apply
    cur_work = inv_machine(task.initial_state.position_mm)
    cur_mach = task.initial_state.position_mm

    raw_segments: List[C.Segment] = []
    idx = 0
    for kind, target_work, feed_mm_s, tag in waypoints_work:
        nxt_work = (float(target_work[0]), float(target_work[1]), float(target_work[2]))
        nxt_mach = to_machine(nxt_work)
        dist = math.dist(cur_mach, nxt_mach)
        if dist <= 1e-9:
            continue
        if kind == C.SegmentKind.RAPID:
            eff_feed = None
            dur = dist / task.limits.rapid_mm_s
        else:
            if feed_mm_s is None or feed_mm_s <= 0:
                raise ValueError("FEED segment requires positive feed_mm_s")
            eff_feed = min(float(feed_mm_s), float(task.limits.max_feed_mm_s))
            dur = dist / eff_feed
        raw_segments.append(
            C.Segment(
                id=f"seg_{idx:03d}",
                kind=kind,
                start_mm=cur_mach,
                end_mm=nxt_mach,
                duration_s=dur,
                feed_mm_s=eff_feed,
                tag=tag,
            )
        )
        cur_work = nxt_work
        cur_mach = nxt_mach
        idx += 1

    rule = task.motion_semantics.timing_rule
    if rule == C.TimingRule.BLENDED_ACCEL_BOUNDED and raw_segments:
        durations = C.blended_segment_durations(
            raw_segments, task.limits.rapid_mm_s, task.limits.max_accel_mm_s2
        )
        segments = tuple(
            dataclasses.replace(seg, duration_s=dur)
            for seg, dur in zip(raw_segments, durations)
        )
    else:
        segments = tuple(raw_segments)

    plan = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=task.initial_state,
        segments=segments,
        timing_rule=rule,
        provenance=provenance,
    )
    plan.validate_for_task(task)
    return plan


def _rect_loop_waypoints(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z_cut: float,
    feed_mm_s: float,
    tag_prefix: str,
    ccw: bool = True,
) -> List[Tuple[C.SegmentKind, Point3D, float | None, str | None]]:
    """Closed 4-edge rectangular milling loop starting and ending at (x0, y0, z_cut)."""
    if ccw:
        corners = [(x1, y0, z_cut), (x1, y1, z_cut), (x0, y1, z_cut), (x0, y0, z_cut)]
    else:
        corners = [(x0, y1, z_cut), (x1, y1, z_cut), (x1, y0, z_cut), (x0, y0, z_cut)]
    return [
        (C.SegmentKind.FEED, pt, feed_mm_s, f"{tag_prefix}_{i}")
        for i, pt in enumerate(corners)
    ]


def facing_raster_seed(
    task: C.TaskSpec,
    *,
    nominal_feed_mm_s: float = 5.5,
    plunge_feed_mm_s: float = 4.0,
    retract_feed_mm_s: float = 14.0,
    overtravel_mm: float = 2.5,
) -> C.MotionPlan:
    """Boustrophedon (zig-zag) facing pass over an open-top billet with an asymmetric raised boss.

    Uses standard CAM stock-bounding-box lead-in/lead-out overtravel and uniform conservative
    feedrate, stepping safely around any full-height boss via an exact cutter-offset moat + split
    raster bands so 100% of open face stock is cleared with zero boss gouge.
    """
    p = task.original_part.params
    bx0, by0, _ = [float(v) for v in p["box_min_mm"]]
    bx1, by1, _ = [float(v) for v in p["box_max_mm"]]
    _, _, pz0 = [float(v) for v in p["pocket_min_mm"]]
    z_safe = float(task.work_to_machine.inverse().apply(task.initial_state.position_mm)[2])
    r = float(task.tool.cutter_radius_mm)
    step = 1.45 * r

    islands = p.get("islands_mm", ())
    waypoints: List[Tuple[C.SegmentKind, Point3D, float | None, str | None]] = []

    x_left = bx0 - overtravel_mm
    x_right = bx1 + overtravel_mm
    # Clamp within machine XY envelope in work frame
    env_lo = task.work_to_machine.inverse().apply(task.limits.envelope_min_mm)
    env_hi = task.work_to_machine.inverse().apply(task.limits.envelope_max_mm)
    x_left = max(x_left, env_lo[0] + 0.5)
    x_right = min(x_right, env_hi[0] - 0.5)

    # Generate Y raster stations covering [by0, by1]
    y_start = by0 + 0.6 * r
    y_end = by1 - 0.6 * r
    n_rows = max(2, int(math.ceil((y_end - y_start) / step)) + 1)
    y_rows = [y_start + (y_end - y_start) * i / (n_rows - 1) for i in range(n_rows)]

    active_isl = [isl for isl in islands if float(isl[1][2]) > pz0 + 1e-6]
    if not active_isl:
        waypoints.append((C.SegmentKind.RAPID, (x_left, y_rows[0], z_safe), None, "approach"))
        waypoints.append((C.SegmentKind.FEED, (x_left, y_rows[0], pz0), plunge_feed_mm_s, "plunge"))
        for idx, y in enumerate(y_rows):
            if idx > 0:
                cur_x = x_left if idx % 2 == 0 else x_right
                waypoints.append((C.SegmentKind.FEED, (cur_x, y, pz0), nominal_feed_mm_s, "stepover"))
            nxt_x = x_right if idx % 2 == 0 else x_left
            waypoints.append((C.SegmentKind.FEED, (nxt_x, y, pz0), nominal_feed_mm_s, "raster"))
        last_x = x_right if (len(y_rows) - 1) % 2 == 0 else x_left
        waypoints.append((C.SegmentKind.FEED, (last_x, y_rows[-1], z_safe), retract_feed_mm_s, "retract"))
        return _segments_to_motion_plan(task, waypoints, "mit_cleanroom:facing_raster_seed@1")

    # Single asymmetric raised boss: clear boss moat first, then raster uninterrupted rows
    # and left/right bands without gouging the boss.
    ilo, ihi = active_isl[0]
    ix0, iy0 = float(ilo[0]) - r, float(ilo[1]) - r
    ix1, iy1 = float(ihi[0]) + r, float(ihi[1]) + r

    # 1. Moat loop around the raised boss
    waypoints.append((C.SegmentKind.RAPID, (ix0, iy0, z_safe), None, "approach_moat"))
    waypoints.append((C.SegmentKind.FEED, (ix0, iy0, pz0), plunge_feed_mm_s, "plunge_moat"))
    waypoints.extend(_rect_loop_waypoints(ix0, ix1, iy0, iy1, pz0, nominal_feed_mm_s, "moat", ccw=True))
    waypoints.append((C.SegmentKind.FEED, (ix0, iy0, z_safe), retract_feed_mm_s, "retract_moat"))

    # Partition rows into lower (y <= iy0), middle (iy0 < y < iy1), and upper (y >= iy1)
    lower_rows = [y for y in y_rows if y <= iy0 + 1e-6]
    mid_rows = [y for y in y_rows if iy0 + 1e-6 < y < iy1 - 1e-6]
    upper_rows = [y for y in y_rows if y >= iy1 - 1e-6]

    def _emit_band(rows: Sequence[float], xa: float, xb: float, band_name: str) -> None:
        if not rows or xb <= xa + 1e-6:
            return
        waypoints.append((C.SegmentKind.RAPID, (xa, rows[0], z_safe), None, f"approach_{band_name}"))
        waypoints.append((C.SegmentKind.FEED, (xa, rows[0], pz0), plunge_feed_mm_s, f"plunge_{band_name}"))
        cur_x = xa
        for r_idx, y in enumerate(rows):
            if r_idx > 0:
                waypoints.append((C.SegmentKind.FEED, (cur_x, y, pz0), nominal_feed_mm_s, f"step_{band_name}"))
            nxt_x = xb if r_idx % 2 == 0 else xa
            waypoints.append((C.SegmentKind.FEED, (nxt_x, y, pz0), nominal_feed_mm_s, f"cut_{band_name}"))
            cur_x = nxt_x
        waypoints.append((C.SegmentKind.FEED, (cur_x, rows[-1], z_safe), retract_feed_mm_s, f"retract_{band_name}"))

    _emit_band(lower_rows, x_left, x_right, "lower")
    if mid_rows:
        if ix0 > bx0 - 0.5 * r:
            _emit_band(mid_rows, x_left, ix0, "mid_left")
        if ix1 < bx1 + 0.5 * r:
            _emit_band(mid_rows, ix1, x_right, "mid_right")
    _emit_band(upper_rows, x_left, x_right, "upper")

    return _segments_to_motion_plan(task, waypoints, "mit_cleanroom:facing_raster_seed@1")


def pocket_offset_seed(
    task: C.TaskSpec,
    *,
    nominal_feed_mm_s: float = 5.0,
    plunge_feed_mm_s: float = 3.5,
    retract_feed_mm_s: float = 14.0,
) -> C.MotionPlan:
    """Concentric offset-ring pocket clearing from retained island outward to pocket wall.

    Interpolates rectangular offset rings between the inner cutter-offset island boundary
    and the outer cutter-offset pocket boundary with max per-axis radial stepover <= 1.45 * r,
    guaranteeing zero corner cusps and zero island/wall gouges.
    """
    p = task.original_part.params
    px0, py0, pz0 = [float(v) for v in p["pocket_min_mm"]]
    px1, py1, _ = [float(v) for v in p["pocket_max_mm"]]
    r = float(task.tool.cutter_radius_mm)
    z_safe = float(task.work_to_machine.inverse().apply(task.initial_state.position_mm)[2])

    outer = (px0 + r, px1 - r, py0 + r, py1 - r)
    islands = [isl for isl in p.get("islands_mm", ()) if float(isl[1][2]) > pz0 + 1e-6]
    if islands:
        ilo, ihi = islands[0]
        inner = (float(ilo[0]) - r, float(ihi[0]) + r, float(ilo[1]) - r, float(ihi[1]) + r)
    else:
        cx, cy = 0.5 * (px0 + px1), 0.5 * (py0 + py1)
        inner = (cx - 0.5 * r, cx + 0.5 * r, cy - 0.5 * r, cy + 0.5 * r)

    max_span = max(
        abs(outer[0] - inner[0]),
        abs(outer[1] - inner[1]),
        abs(outer[2] - inner[2]),
        abs(outer[3] - inner[3]),
    )
    max_step = 1.42 * r
    n_rings = max(2, int(math.ceil(max_span / max_step)) + 1)

    waypoints: List[Tuple[C.SegmentKind, Point3D, float | None, str | None]] = []
    waypoints.append((C.SegmentKind.RAPID, (inner[0], inner[2], z_safe), None, "approach"))
    waypoints.append((C.SegmentKind.FEED, (inner[0], inner[2], pz0), plunge_feed_mm_s, "plunge"))

    for idx in range(n_rings):
        alpha = idx / (n_rings - 1)
        rx0 = inner[0] + alpha * (outer[0] - inner[0])
        rx1 = inner[1] + alpha * (outer[1] - inner[1])
        ry0 = inner[2] + alpha * (outer[2] - inner[2])
        ry1 = inner[3] + alpha * (outer[3] - inner[3])
        if idx > 0:
            waypoints.append((C.SegmentKind.FEED, (rx0, ry0, pz0), nominal_feed_mm_s, f"link_{idx}"))
        waypoints.extend(_rect_loop_waypoints(rx0, rx1, ry0, ry1, pz0, nominal_feed_mm_s, f"ring_{idx}", ccw=True))

    waypoints.append((C.SegmentKind.FEED, (outer[0], outer[2], z_safe), retract_feed_mm_s, "retract"))
    return _segments_to_motion_plan(task, waypoints, "mit_cleanroom:pocket_offset_seed@1")


def adaptive_trochoidal_seed(
    task: C.TaskSpec,
    *,
    nominal_feed_mm_s: float = 5.0,
    plunge_feed_mm_s: float = 3.5,
    retract_feed_mm_s: float = 14.0,
) -> C.MotionPlan:
    """Clean-room chordal peel / trochoidal clearing through dual-island pinched-throat pockets.

    Clears left and right chambers with oblique peel-and-return strokes (modelling constant-step
    peel clearing with low-engagement return chords), threads the central throat channel between
    the two islands, and finishes the island and outer pocket walls cleanly.
    """
    p = task.original_part.params
    px0, py0, pz0 = [float(v) for v in p["pocket_min_mm"]]
    px1, py1, _ = [float(v) for v in p["pocket_max_mm"]]
    r = float(task.tool.cutter_radius_mm)
    z_safe = float(task.work_to_machine.inverse().apply(task.initial_state.position_mm)[2])

    ox0, ox1, oy0, oy1 = px0 + r, px1 - r, py0 + r, py1 - r
    islands = sorted(
        [isl for isl in p.get("islands_mm", ()) if float(isl[1][2]) > pz0 + 1e-6],
        key=lambda b: float(b[0][1]),
    )
    if len(islands) < 2:
        return pocket_offset_seed(
            task,
            nominal_feed_mm_s=nominal_feed_mm_s,
            plunge_feed_mm_s=plunge_feed_mm_s,
            retract_feed_mm_s=retract_feed_mm_s,
        )

    # Two islands stacked in Y (bottom island isl_b, top island isl_t) creating:
    # - left chamber [ox0, ix_lo], right chamber [ix_hi, ox1],
    # - bottom lane [oy0, b_y0], central throat [b_y1, t_y0], top lane [t_y1, oy1].
    b_lo, b_hi = islands[0]
    t_lo, t_hi = islands[1]
    bx0_i, by0_i = float(b_lo[0]) - r, float(b_lo[1]) - r
    bx1_i, by1_i = float(b_hi[0]) + r, float(b_hi[1]) + r
    tx0_i, ty0_i = float(t_lo[0]) - r, float(t_lo[1]) - r
    tx1_i, ty1_i = float(t_hi[0]) + r, float(t_hi[1]) + r

    waypoints: List[Tuple[C.SegmentKind, Point3D, float | None, str | None]] = []
    waypoints.append((C.SegmentKind.RAPID, (ox0, oy0, z_safe), None, "approach"))
    waypoints.append((C.SegmentKind.FEED, (ox0, oy0, pz0), plunge_feed_mm_s, "plunge"))

    # 1. Outer pocket wall perimeter pass
    waypoints.extend(_rect_loop_waypoints(ox0, ox1, oy0, oy1, pz0, nominal_feed_mm_s, "wall", ccw=True))

    # 2. Left chamber peel strokes (x from ox0 to min(bx0_i, tx0_i))
    lx_hi = min(bx0_i, tx0_i)
    if lx_hi > ox0 + 0.2:
        n_peel = max(2, int(math.ceil((lx_hi - ox0) / (1.35 * r))))
        for k in range(1, n_peel + 1):
            xk = ox0 + (lx_hi - ox0) * k / n_peel
            x_ret = ox0 + (lx_hi - ox0) * (k - 0.5) / n_peel
            waypoints.append((C.SegmentKind.FEED, (xk, oy0, pz0), nominal_feed_mm_s, f"peel_l_{k}a"))
            waypoints.append((C.SegmentKind.FEED, (xk, oy1, pz0), nominal_feed_mm_s, f"peel_l_{k}b"))
            if k < n_peel:
                waypoints.append((C.SegmentKind.FEED, (x_ret, oy0, pz0), nominal_feed_mm_s, f"ret_l_{k}"))

    # 3. Top lane & right chamber peel strokes
    rx_lo = max(bx1_i, tx1_i)
    waypoints.append((C.SegmentKind.FEED, (ox1, oy1, pz0), nominal_feed_mm_s, "top_lane"))
    if oy1 - ty1_i > 1.45 * r:
        y_top_mid = 0.5 * (oy1 + ty1_i)
        waypoints.append((C.SegmentKind.FEED, (ox0, y_top_mid, pz0), nominal_feed_mm_s, "top_mid_a"))
        waypoints.append((C.SegmentKind.FEED, (ox1, y_top_mid, pz0), nominal_feed_mm_s, "top_mid_b"))
        waypoints.append((C.SegmentKind.FEED, (ox1, oy1, pz0), nominal_feed_mm_s, "top_mid_c"))

    if ox1 > rx_lo + 0.2:
        n_peel_r = max(2, int(math.ceil((ox1 - rx_lo) / (1.35 * r))))
        for k in range(1, n_peel_r + 1):
            xk = ox1 - (ox1 - rx_lo) * k / n_peel_r
            x_ret = ox1 - (ox1 - rx_lo) * (k - 0.5) / n_peel_r
            waypoints.append((C.SegmentKind.FEED, (xk, oy1, pz0), nominal_feed_mm_s, f"peel_r_{k}a"))
            waypoints.append((C.SegmentKind.FEED, (xk, oy0, pz0), nominal_feed_mm_s, f"peel_r_{k}b"))
            if k < n_peel_r:
                waypoints.append((C.SegmentKind.FEED, (x_ret, oy1, pz0), nominal_feed_mm_s, f"ret_r_{k}"))

    if by0_i - oy0 > 1.45 * r:
        y_bot_mid = 0.5 * (oy0 + by0_i)
        waypoints.append((C.SegmentKind.FEED, (ox1, y_bot_mid, pz0), nominal_feed_mm_s, "bot_mid_a"))
        waypoints.append((C.SegmentKind.FEED, (ox0, y_bot_mid, pz0), nominal_feed_mm_s, "bot_mid_b"))

    # 4. Bottom island clean contour + central throat channel clearing + top island clean contour
    waypoints.append((C.SegmentKind.FEED, (bx0_i, by0_i, pz0), nominal_feed_mm_s, "to_isl_b"))
    waypoints.extend(_rect_loop_waypoints(bx0_i, bx1_i, by0_i, by1_i, pz0, nominal_feed_mm_s, "isl_b", ccw=True))

    # Central throat between by1_i and ty0_i
    if ty0_i > by1_i + 1e-6:
        n_th = max(1, int(math.ceil((ty0_i - by1_i) / (1.35 * r))))
        for k in range(1, n_th + 1):
            yk = by1_i + (ty0_i - by1_i) * k / (n_th + 1)
            xa, xb = (lx_hi, rx_lo) if k % 2 == 1 else (rx_lo, lx_hi)
            waypoints.append((C.SegmentKind.FEED, (xa, yk, pz0), nominal_feed_mm_s, f"throat_in_{k}"))
            waypoints.append((C.SegmentKind.FEED, (xb, yk, pz0), nominal_feed_mm_s, f"throat_cut_{k}"))

    # Retract safely over island to top island start (tx0_i, ty0_i)
    cur_pt = waypoints[-1][1]
    waypoints.append((C.SegmentKind.FEED, (cur_pt[0], cur_pt[1], z_safe), retract_feed_mm_s, "hop_up"))
    waypoints.append((C.SegmentKind.RAPID, (tx0_i, ty0_i, z_safe), None, "hop_over"))
    waypoints.append((C.SegmentKind.FEED, (tx0_i, ty0_i, pz0), plunge_feed_mm_s, "hop_plunge"))
    waypoints.extend(_rect_loop_waypoints(tx0_i, tx1_i, ty0_i, ty1_i, pz0, nominal_feed_mm_s, "isl_t", ccw=True))
    waypoints.append((C.SegmentKind.FEED, (tx0_i, ty0_i, z_safe), retract_feed_mm_s, "retract"))

    return _segments_to_motion_plan(task, waypoints, "mit_cleanroom:adaptive_trochoidal_seed@1")


def contour_offset_seed(
    task: C.TaskSpec,
    *,
    nominal_feed_mm_s: float = 5.0,
    plunge_feed_mm_s: float = 4.0,
    retract_feed_mm_s: float = 14.0,
    outer_clearance_mm: float = 1.5,
) -> C.MotionPlan:
    """Outside-in multi-offset exterior boss contour profiling.

    Steps inward from outside the billet perimeter down to the exact cutter-offset boss contour
    with radial stepover <= 1.42 * r so 100% of exterior shoulder material is removed cleanly.
    """
    p = task.original_part.params
    bx0, by0, _ = [float(v) for v in p["box_min_mm"]]
    bx1, by1, _ = [float(v) for v in p["box_max_mm"]]
    _, _, pz0 = [float(v) for v in p["pocket_min_mm"]]
    r = float(task.tool.cutter_radius_mm)
    z_safe = float(task.work_to_machine.inverse().apply(task.initial_state.position_mm)[2])

    islands = [isl for isl in p.get("islands_mm", ()) if float(isl[1][2]) > pz0 + 1e-6]
    if not islands:
        raise ValueError("contour_offset_seed requires at least one standing boss in islands_mm")
    ilo, ihi = islands[0]
    inner = (float(ilo[0]) - r, float(ihi[0]) + r, float(ilo[1]) - r, float(ihi[1]) + r)
    # Start outer loop slightly outside stock minus 0.4*r so cutter engages stock corners cleanly
    # while including realistic exterior CAM lead-around air segments on asymmetric bosses.
    outer = (
        bx0 - outer_clearance_mm + 0.5 * r,
        bx1 + outer_clearance_mm - 0.5 * r,
        by0 - outer_clearance_mm + 0.5 * r,
        by1 + outer_clearance_mm - 0.5 * r,
    )
    env_lo = task.work_to_machine.inverse().apply(task.limits.envelope_min_mm)
    env_hi = task.work_to_machine.inverse().apply(task.limits.envelope_max_mm)
    outer = (
        max(outer[0], env_lo[0] + 0.5),
        min(outer[1], env_hi[0] - 0.5),
        max(outer[2], env_lo[1] + 0.5),
        min(outer[3], env_hi[1] - 0.5),
    )

    max_span = max(
        abs(inner[0] - outer[0]),
        abs(inner[1] - outer[1]),
        abs(inner[2] - outer[2]),
        abs(inner[3] - outer[3]),
    )
    n_loops = max(2, int(math.ceil(max_span / (1.42 * r))) + 1)

    waypoints: List[Tuple[C.SegmentKind, Point3D, float | None, str | None]] = []
    waypoints.append((C.SegmentKind.RAPID, (outer[0], outer[2], z_safe), None, "approach"))
    waypoints.append((C.SegmentKind.FEED, (outer[0], outer[2], pz0), plunge_feed_mm_s, "plunge"))

    for idx in range(n_loops):
        alpha = idx / (n_loops - 1)
        lx0 = outer[0] + alpha * (inner[0] - outer[0])
        lx1 = outer[1] + alpha * (inner[1] - outer[1])
        ly0 = outer[2] + alpha * (inner[2] - outer[2])
        ly1 = outer[3] + alpha * (inner[3] - outer[3])
        if idx > 0:
            waypoints.append((C.SegmentKind.FEED, (lx0, ly0, pz0), nominal_feed_mm_s, f"step_in_{idx}"))
        waypoints.extend(_rect_loop_waypoints(lx0, lx1, ly0, ly1, pz0, nominal_feed_mm_s, f"contour_{idx}", ccw=True))

    waypoints.append((C.SegmentKind.FEED, (inner[0], inner[2], z_safe), retract_feed_mm_s, "retract"))
    return _segments_to_motion_plan(task, waypoints, "mit_cleanroom:contour_offset_seed@1")


def canonical_seed_for_task(task: C.TaskSpec) -> C.MotionPlan:
    """Dispatch the canonical MIT CAM strategy matching ``task.family``."""
    dispatch = {
        "facing_open_boss": facing_raster_seed,
        "pocket_island": pocket_offset_seed,
        "adaptive_channel": adaptive_trochoidal_seed,
        "contour_profile": contour_offset_seed,
    }
    fn = dispatch.get(task.family)
    if fn is None:
        raise ValueError(f"unsupported canonical milling family: {task.family!r}")
    return fn(task)


__all__ = [
    "facing_raster_seed",
    "pocket_offset_seed",
    "adaptive_trochoidal_seed",
    "contour_offset_seed",
    "canonical_seed_for_task",
]
