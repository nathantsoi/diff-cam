# SPDX-License-Identifier: MIT
"""General-purpose 2.5D clean-room open-perimeter facing toolpath generator.

Supports:
- ``0..N`` standing rectangular bosses/islands of arbitrary count and placement
- Configurable radial stepover fraction, arbitrary raster angle in ``[0, 180)`` deg,
  boustrophedon ``"zigzag"`` or true ``"climb_one_way"`` cutting modes, and
  optional multi-depth axial stepdown layers from ``stock_max_z`` down to ``pz0``
- Exact cutter-offset moat contour passes around active standing bosses, followed
  by analytically clipped scanlines decomposed into unobstructed boustrophedon
  sub-bands with safe ``z_safe`` retracts/rapids across bosses and sub-bands
- Machine-frame ``MotionPlan`` output under both ``EXACT_STOP_CONSTANT_SPEED`` and
  ``BLENDED_ACCEL_BOUNDED`` (plus ``FIXED_DT``) timing rules.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks import contracts as C

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
Box2D = Tuple[float, float, float, float]  # (x0, x1, y0, y1)
Box3D = Tuple[Point3D, Point3D]

PROVENANCE = "mit_cleanroom:facing_strategy@2"

# Guard margins (mm) exceeding 4-decimal NC post quantization (5e-5 mm) while staying
# orders of magnitude within the R-wide moat buffer cleared around every standing boss.
_SCAN_BOSS_GUARD_MM = 1e-3
_CHORD_BOSS_GUARD_MM = 0.5e-3
_MIN_CUT_SEG_LEN_MM = 1e-2
_ENVELOPE_MARGIN_MM = 0.5


@dataclass(frozen=True)
class FacingConfig:
    """Configuration parameters for 2.5D open-perimeter billet facing."""

    stepover_fraction: float = 0.70
    raster_angle_deg: float = 0.0
    direction_mode: str = "zigzag"
    axial_stepdown_mm: float | None = None
    overtravel_mm: float = 2.5
    nominal_feed_mm_s: float = 5.5
    plunge_feed_mm_s: float = 4.0
    retract_feed_mm_s: float = 14.0

    def __post_init__(self) -> None:
        sf = float(self.stepover_fraction)
        if not (math.isfinite(sf) and 0.1 < sf < 0.9):
            raise ValueError(
                f"stepover_fraction must be finite and strictly in (0.1, 0.9), got {self.stepover_fraction!r}"
            )
        object.__setattr__(self, "stepover_fraction", sf)

        ang = float(self.raster_angle_deg)
        if not (math.isfinite(ang) and 0.0 <= ang < 180.0):
            raise ValueError(
                f"raster_angle_deg must be finite and in [0.0, 180.0), got {self.raster_angle_deg!r}"
            )
        object.__setattr__(self, "raster_angle_deg", ang)

        if self.direction_mode not in ("zigzag", "climb_one_way"):
            raise ValueError(
                f"direction_mode must be 'zigzag' or 'climb_one_way', got {self.direction_mode!r}"
            )

        if self.axial_stepdown_mm is not None:
            sd = float(self.axial_stepdown_mm)
            if not (math.isfinite(sd) and sd > 0.0):
                raise ValueError(
                    f"axial_stepdown_mm must be finite and > 0 when provided, got {self.axial_stepdown_mm!r}"
                )
            object.__setattr__(self, "axial_stepdown_mm", sd)

        ot = float(self.overtravel_mm)
        if not (math.isfinite(ot) and ot >= 0.0):
            raise ValueError(f"overtravel_mm must be finite and >= 0, got {self.overtravel_mm!r}")
        object.__setattr__(self, "overtravel_mm", ot)

        for field_name in ("nominal_feed_mm_s", "plunge_feed_mm_s", "retract_feed_mm_s"):
            val = float(getattr(self, field_name))
            if not (math.isfinite(val) and val > 0.0):
                raise ValueError(f"{field_name} must be finite and > 0, got {getattr(self, field_name)!r}")
            object.__setattr__(self, field_name, val)


def _extract_facing_geometry(
    task: C.TaskSpec,
) -> Tuple[Point3D, Point3D, float, Tuple[Box3D, ...]]:
    """Extract (stock_min_mm, stock_max_mm, target_face_z_mm, islands_mm) in work frame."""
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        raise ValueError("generate_facing_toolpath requires a flat cutter (CutterKind.FLAT)")
    if not task.initial_state.known or task.initial_state.position_mm is None:
        raise ValueError("generate_facing_toolpath requires a known initial_state")
    if task.stock.kind != "analytic" or task.stock.params.get("primitive") != "box":
        raise ValueError("generate_facing_toolpath requires analytic box stock")

    s_min = tuple(float(v) for v in task.stock.params["min_mm"])
    s_max = tuple(float(v) for v in task.stock.params["max_mm"])
    if len(s_min) != 3 or len(s_max) != 3:
        raise ValueError("stock min_mm/max_mm must be 3D coordinates")

    part = task.original_part
    if part.kind != "analytic":
        raise ValueError("generate_facing_toolpath requires analytic original_part geometry")
    p = part.params
    prim = p.get("primitive")

    if prim == "box_minus_multi_pocket":
        pz0 = float(p["pocket_min_mm"][2])
        raw_islands = p.get("islands_mm", ())
        islands = tuple(
            (
                (float(isl[0][0]), float(isl[0][1]), float(isl[0][2])),
                (float(isl[1][0]), float(isl[1][1]), float(isl[1][2])),
            )
            for isl in raw_islands
        )
    elif prim == "box_minus_pocket":
        pz0 = float(p["pocket_min_mm"][2])
        if "island_min_mm" in p and "island_max_mm" in p:
            ilo = p["island_min_mm"]
            ihi = p["island_max_mm"]
            islands = (
                (
                    (float(ilo[0]), float(ilo[1]), float(ilo[2])),
                    (float(ihi[0]), float(ihi[1]), float(ihi[2])),
                ),
            )
        else:
            islands = ()
    elif prim == "box":
        pz0 = float(p["max_mm"][2])
        islands = ()
    else:
        raise ValueError(f"unsupported original_part primitive for facing: {prim!r}")

    return (
        (s_min[0], s_min[1], s_min[2]),
        (s_max[0], s_max[1], s_max[2]),
        pz0,
        islands,
    )


def _compute_z_layers(
    stock_max_z: float,
    pz0: float,
    axial_stepdown_mm: Optional[float],
) -> List[float]:
    """Compute cutting Z depths from stock_max_z down to exact final floor pz0."""
    if axial_stepdown_mm is None or pz0 >= stock_max_z - 1e-9:
        return [pz0]
    total_depth = stock_max_z - pz0
    n_layers = max(1, int(math.ceil((total_depth - 1e-9) / axial_stepdown_mm)))
    layers = [stock_max_z - total_depth * (k / n_layers) for k in range(1, n_layers)]
    layers.append(pz0)
    return layers


def _clip_line_to_closed_box(
    px: float,
    py: float,
    dx: float,
    dy: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    t_min: float,
    t_max: float,
) -> Optional[Tuple[float, float]]:
    """Liang-Barsky parameter clip of P(t) = (px + t*dx, py + t*dy) to closed [x0, x1] x [y0, y1]."""
    lo, hi = t_min, t_max
    for p_coord, d_coord, b0, b1 in ((px, dx, x0, x1), (py, dy, y0, y1)):
        if abs(d_coord) <= 1e-12:
            if p_coord < b0 - 1e-12 or p_coord > b1 + 1e-12:
                return None
        else:
            t1 = (b0 - p_coord) / d_coord
            t2 = (b1 - p_coord) / d_coord
            t_enter = min(t1, t2)
            t_exit = max(t1, t2)
            lo = max(lo, t_enter)
            hi = min(hi, t_exit)
            if lo > hi:
                return None
    return (lo, hi)


def _line_open_box_interval(
    px: float,
    py: float,
    dx: float,
    dy: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> Optional[Tuple[float, float]]:
    """Return parameter interval (t_enter, t_exit) where P(t) enters open (x0, x1) x (y0, y1)."""
    if x1 <= x0 or y1 <= y0:
        return None
    lo, hi = -math.inf, math.inf
    for p_coord, d_coord, b0, b1 in ((px, dx, x0, x1), (py, dy, y0, y1)):
        if abs(d_coord) <= 1e-12:
            if not (b0 < p_coord < b1):
                return None
        else:
            t1 = (b0 - p_coord) / d_coord
            t2 = (b1 - p_coord) / d_coord
            lo = max(lo, min(t1, t2))
            hi = min(hi, max(t1, t2))
            if hi <= lo:
                return None
    if hi <= lo + 1e-12:
        return None
    return (lo, hi)


def _subtract_open_intervals(
    t0: float,
    t1: float,
    forbidden: Sequence[Tuple[float, float]],
    min_len: float = 1e-6,
) -> List[Tuple[float, float]]:
    """Subtract a collection of open 1D intervals from closed [t0, t1]."""
    if t1 <= t0 + min_len:
        return []
    kept: List[Tuple[float, float]] = [(t0, t1)]
    for flo, fhi in forbidden:
        if fhi <= flo:
            continue
        nxt: List[Tuple[float, float]] = []
        for a, b in kept:
            if fhi <= a + 1e-12 or flo >= b - 1e-12:
                nxt.append((a, b))
                continue
            if flo > a + min_len:
                nxt.append((a, min(b, flo)))
            if fhi < b - min_len:
                nxt.append((max(a, fhi), b))
        kept = nxt
        if not kept:
            break
    return [(a, b) for a, b in kept if b - a >= min_len]


def _clip_work_line_to_machine_envelope(
    task: C.TaskSpec,
    px: float,
    py: float,
    dx: float,
    dy: float,
    z_work: float,
    t0: float,
    t1: float,
    margin_mm: float = _ENVELOPE_MARGIN_MM,
) -> Optional[Tuple[float, float]]:
    """Clip a work-frame XY parametric line P(t) for t in [t0, t1] inside machine XY limits."""
    to_mach = task.work_to_machine.apply
    m0 = to_mach((px, py, z_work))
    m1 = to_mach((px + dx, py + dy, z_work))
    mdx = m1[0] - m0[0]
    mdy = m1[1] - m0[1]

    elo_x, elo_y, _ = task.limits.envelope_min_mm
    ehi_x, ehi_y, _ = task.limits.envelope_max_mm
    mx = min(margin_mm, 0.1 * (ehi_x - elo_x))
    my = min(margin_mm, 0.1 * (ehi_y - elo_y))
    return _clip_line_to_closed_box(
        m0[0],
        m0[1],
        mdx,
        mdy,
        elo_x + mx,
        ehi_x - mx,
        elo_y + my,
        ehi_y - my,
        t0,
        t1,
    )


def _segment_is_clear(
    p0: Point2D,
    p1: Point2D,
    forbidden_boxes: Sequence[Box2D],
) -> bool:
    """Return True iff closed segment p0 -> p1 never enters any open box in forbidden_boxes."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    for bx0, bx1, by0, by1 in forbidden_boxes:
        hit = _line_open_box_interval(p0[0], p0[1], dx, dy, bx0, bx1, by0, by1)
        if hit is not None:
            t_enter, t_exit = hit
            if min(1.0, t_exit) > max(0.0, t_enter) + 1e-12:
                return False
    return True


def _build_moat_polylines_for_boss(
    task: C.TaskSpec,
    boss_idx: int,
    active_bosses: Sequence[Box3D],
    cutter_radius_mm: float,
    z_layer: float,
) -> List[List[Point2D]]:
    """Construct exact cutter-offset moat contour loop(s) around active_bosses[boss_idx].

    Edges lie along [ix0 - R, ix1 + R] x [iy0 - R, iy1 + R], clipped safely to the machine XY
    envelope and against open guarded zones of any other active standing bosses so adjacent or
    overlapping boss moats never gouge a neighboring boss.
    """
    ilo, ihi = active_bosses[boss_idx]
    mx0 = ilo[0] - cutter_radius_mm
    mx1 = ihi[0] + cutter_radius_mm
    my0 = ilo[1] - cutter_radius_mm
    my1 = ihi[1] + cutter_radius_mm

    other_boxes: List[Box2D] = []
    for j, (jlo, jhi) in enumerate(active_bosses):
        if j == boss_idx:
            continue
        other_boxes.append(
            (
                jlo[0] - cutter_radius_mm - _SCAN_BOSS_GUARD_MM,
                jhi[0] + cutter_radius_mm + _SCAN_BOSS_GUARD_MM,
                jlo[1] - cutter_radius_mm - _SCAN_BOSS_GUARD_MM,
                jhi[1] + cutter_radius_mm + _SCAN_BOSS_GUARD_MM,
            )
        )

    ccw_edges: Tuple[Tuple[Point2D, Point2D], ...] = (
        ((mx0, my0), (mx1, my0)),
        ((mx1, my0), (mx1, my1)),
        ((mx1, my1), (mx0, my1)),
        ((mx0, my1), (mx0, my0)),
    )

    pieces: List[Tuple[Point2D, Point2D]] = []
    for a, b in ccw_edges:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        seg_len = math.hypot(dx, dy)
        if seg_len <= _MIN_CUT_SEG_LEN_MM:
            continue
        env_clip = _clip_work_line_to_machine_envelope(
            task, a[0], a[1], dx, dy, z_layer, 0.0, 1.0
        )
        if env_clip is None:
            continue
        t_lo, t_hi = env_clip
        forb: List[Tuple[float, float]] = []
        for ox0, ox1, oy0, oy1 in other_boxes:
            hit = _line_open_box_interval(a[0], a[1], dx, dy, ox0, ox1, oy0, oy1)
            if hit is not None:
                forb.append(hit)
        sub_t = _subtract_open_intervals(t_lo, t_hi, forb, min_len=_MIN_CUT_SEG_LEN_MM / seg_len)
        for ta, tb in sub_t:
            pa = (a[0] + ta * dx, a[1] + ta * dy)
            pb = (a[0] + tb * dx, a[1] + tb * dy)
            if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) >= _MIN_CUT_SEG_LEN_MM:
                pieces.append((pa, pb))

    if not pieces:
        return []

    polylines: List[List[Point2D]] = []
    for pa, pb in pieces:
        if polylines and math.hypot(polylines[-1][-1][0] - pa[0], polylines[-1][-1][1] - pa[1]) <= 1e-6:
            polylines[-1].append(pb)
        else:
            polylines.append([pa, pb])

    # Merge wrap-around seam if an open clipped moat splits mid-loop across (mx0, my0).
    if (
        len(polylines) > 1
        and math.hypot(
            polylines[-1][-1][0] - polylines[0][0][0],
            polylines[-1][-1][1] - polylines[0][0][1],
        )
        <= 1e-6
    ):
        merged = polylines[-1] + polylines[0][1:]
        polylines = [merged] + polylines[1:-1]

    return polylines


def _build_clipped_scanlines(
    task: C.TaskSpec,
    stock_min: Point3D,
    stock_max: Point3D,
    active_bosses: Sequence[Box3D],
    config: FacingConfig,
    z_layer: float,
) -> Tuple[
    Point2D,
    Point2D,
    Point2D,
    List[Tuple[int, float, List[Tuple[float, float]]]],
]:
    """Generate rotated scanline stations and clip each 1D u-interval against all active bosses."""
    r = float(task.tool.cutter_radius_mm)
    step = config.stepover_fraction * (2.0 * r)
    theta = math.radians(config.raster_angle_deg)
    eu: Point2D = (math.cos(theta), math.sin(theta))
    ev: Point2D = (-math.sin(theta), math.cos(theta))

    bx0, by0 = stock_min[0], stock_min[1]
    bx1, by1 = stock_max[0], stock_max[1]
    cx = 0.5 * (bx0 + bx1)
    cy = 0.5 * (by0 + by1)
    hx = 0.5 * (bx1 - bx0)
    hy = 0.5 * (by1 - by0)

    v_max = hx * abs(ev[0]) + hy * abs(ev[1])
    v_min = -v_max
    edge_inset = min(0.5 * r, 0.5 * step)

    if v_max - v_min <= 2.0 * edge_inset + 1e-9:
        v_rows = [0.0]
    else:
        v_start = v_min + edge_inset
        v_end = v_max - edge_inset
        n_rows = max(2, int(math.ceil((v_end - v_start - 1e-9) / step)) + 1)
        v_rows = [v_start + (v_end - v_start) * i / (n_rows - 1) for i in range(n_rows)]

    # In climb_one_way with +eu feed direction, stepping from +v down to -v keeps uncut stock
    # consistently on the -ev (right-hand) side of travel for true M03 CW climb milling.
    if config.direction_mode == "climb_one_way":
        v_rows = list(reversed(v_rows))

    # Expand stock projection box by 0.5*step along |ev| so every stock point's orthogonal
    # projection onto its nearest scanline lies inside [u_stock_lo, u_stock_hi] at any angle.
    proj_mx = 0.5 * step * abs(ev[0])
    proj_my = 0.5 * step * abs(ev[1])
    sx0, sx1 = bx0 - proj_mx, bx1 + proj_mx
    sy0, sy1 = by0 - proj_my, by1 + proj_my

    scan_boss_boxes: List[Box2D] = [
        (
            ilo[0] - r - _SCAN_BOSS_GUARD_MM,
            ihi[0] + r + _SCAN_BOSS_GUARD_MM,
            ilo[1] - r - _SCAN_BOSS_GUARD_MM,
            ihi[1] + r + _SCAN_BOSS_GUARD_MM,
        )
        for ilo, ihi in active_bosses
    ]

    u_bound = math.hypot(sx1 - sx0, sy1 - sy0) + config.overtravel_mm + 10.0
    rows_out: List[Tuple[int, float, List[Tuple[float, float]]]] = []

    for r_idx, v in enumerate(v_rows):
        px = cx + v * ev[0]
        py = cy + v * ev[1]
        stock_span = _clip_line_to_closed_box(
            px, py, eu[0], eu[1], sx0, sx1, sy0, sy1, -u_bound, u_bound
        )
        if stock_span is None:
            continue
        u0 = stock_span[0] - config.overtravel_mm
        u1 = stock_span[1] + config.overtravel_mm

        env_span = _clip_work_line_to_machine_envelope(
            task, px, py, eu[0], eu[1], z_layer, u0, u1
        )
        if env_span is None:
            continue

        forbidden_u: List[Tuple[float, float]] = []
        for gbx0, gbx1, gby0, gby1 in scan_boss_boxes:
            hit = _line_open_box_interval(px, py, eu[0], eu[1], gbx0, gbx1, gby0, gby1)
            if hit is not None:
                forbidden_u.append(hit)

        clipped = _subtract_open_intervals(
            env_span[0], env_span[1], forbidden_u, min_len=_MIN_CUT_SEG_LEN_MM
        )
        if clipped:
            rows_out.append((r_idx, v, clipped))

    return (cx, cy), eu, ev, rows_out


def _decompose_into_sub_bands(
    center: Point2D,
    eu: Point2D,
    ev: Point2D,
    rows: Sequence[Tuple[int, float, List[Tuple[float, float]]]],
    chord_boss_boxes: Sequence[Box2D],
) -> List[List[Tuple[float, float, float]]]:
    """Group scanline sub-segments into unobstructed 1-to-1 boustrophedon sub-bands."""
    if not rows:
        return []

    def _pt(v: float, u: float) -> Point2D:
        return (
            center[0] + v * ev[0] + u * eu[0],
            center[1] + v * ev[1] + u * eu[1],
        )

    bands: List[List[Tuple[float, float, float]]] = []
    prev_r_idx: Optional[int] = None
    prev_v: float = 0.0
    prev_intervals: List[Tuple[float, float]] = []
    active_map: Dict[int, int] = {}

    for r_idx, v, intervals in rows:
        next_map: Dict[int, int] = {}
        if prev_r_idx is not None and r_idx == prev_r_idx + 1 and prev_intervals:
            fwd_edges: Dict[int, List[int]] = {j: [] for j in range(len(prev_intervals))}
            bwd_edges: Dict[int, List[int]] = {i: [] for i in range(len(intervals))}
            for j, (ua0, ub0) in enumerate(prev_intervals):
                for i, (ua1, ub1) in enumerate(intervals):
                    if min(ub0, ub1) <= max(ua0, ua1):
                        continue
                    if _segment_is_clear(
                        _pt(prev_v, ua0), _pt(v, ua1), chord_boss_boxes
                    ) and _segment_is_clear(
                        _pt(prev_v, ub0), _pt(v, ub1), chord_boss_boxes
                    ):
                        fwd_edges[j].append(i)
                        bwd_edges[i].append(j)

            for j, targets in fwd_edges.items():
                if len(targets) == 1:
                    i = targets[0]
                    if len(bwd_edges[i]) == 1:
                        b_id = active_map[j]
                        ua1, ub1 = intervals[i]
                        bands[b_id].append((v, ua1, ub1))
                        next_map[i] = b_id

        for i, (ua1, ub1) in enumerate(intervals):
            if i not in next_map:
                b_id = len(bands)
                bands.append([(v, ua1, ub1)])
                next_map[i] = b_id

        prev_r_idx = r_idx
        prev_v = v
        prev_intervals = intervals
        active_map = next_map

    return bands


def _waypoints_to_motion_plan(
    task: C.TaskSpec,
    waypoints_work: Sequence[Tuple[C.SegmentKind, Point3D, Optional[float], str]],
    provenance: str,
) -> C.MotionPlan:
    """Convert work-frame motion commands into a task-validated machine-frame MotionPlan."""
    to_machine = task.work_to_machine.apply
    cur_mach = task.initial_state.position_mm
    assert cur_mach is not None

    # Leave a tiny margin below max_feed_mm_s when clamping so 4-decimal G-code F-word
    # formatting never rounds above task.limits.max_feed_mm_s on re-parse.
    feed_cap = float(task.limits.max_feed_mm_s)
    safe_cap = feed_cap * (1.0 - 1e-5)

    raw_segments: List[C.Segment] = []
    seg_idx = 0

    for kind, target_work, feed_mm_s, tag in waypoints_work:
        nxt_work = (float(target_work[0]), float(target_work[1]), float(target_work[2]))
        nxt_mach = to_machine(nxt_work)
        dist = math.dist(cur_mach, nxt_mach)
        if dist <= 1e-5:
            continue
        if kind == C.SegmentKind.RAPID:
            eff_feed = None
            dur = dist / float(task.limits.rapid_mm_s)
        else:
            if feed_mm_s is None or feed_mm_s <= 0.0:
                raise ValueError("FEED segment requires positive feed_mm_s")
            eff_feed = float(feed_mm_s) if float(feed_mm_s) < safe_cap else safe_cap
            dur = dist / eff_feed

        raw_segments.append(
            C.Segment(
                id=f"seg_{seg_idx:04d}",
                kind=kind,
                start_mm=cur_mach,
                end_mm=nxt_mach,
                duration_s=dur,
                feed_mm_s=eff_feed,
                tag=tag,
            )
        )
        cur_mach = nxt_mach
        seg_idx += 1

    rule = task.motion_semantics.timing_rule
    dt_s: Optional[float] = None
    if rule == C.TimingRule.BLENDED_ACCEL_BOUNDED and raw_segments:
        durations = C.blended_segment_durations(
            raw_segments, task.limits.rapid_mm_s, task.limits.max_accel_mm_s2
        )
        segments = tuple(
            dataclasses.replace(seg, duration_s=dur)
            for seg, dur in zip(raw_segments, durations)
        )
    elif rule == C.TimingRule.FIXED_DT:
        dt_s = 0.01
        segments = tuple(
            dataclasses.replace(seg, duration_s=max(1, round(seg.duration_s / dt_s)) * dt_s)
            for seg in raw_segments
        )
    else:
        segments = tuple(raw_segments)

    plan = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=task.initial_state,
        segments=segments,
        timing_rule=rule,
        dt_s=dt_s,
        provenance=provenance,
    )
    plan.validate_for_task(task)
    return plan


def generate_facing_toolpath(
    task: C.TaskSpec,
    config: FacingConfig = FacingConfig(),
) -> C.MotionPlan:
    """Generate a validated 2.5D open-perimeter facing MotionPlan in ``Frame.MACHINE``.

    At each Z layer:
    1. Emits exact cutter-offset moat contour loops ``[ix0 - R, ix1 + R] x [iy0 - R, iy1 + R]``
       around every active standing boss (``top_z > z_layer``).
    2. Generates rotated scanlines across the stock XY box with clamped lead-in/lead-out
       overtravel, analytically clips every scanline against all active expanded boss
       rectangles, chains unobstructed boustrophedon sub-bands (in ``"zigzag"`` mode) or
       one-way climb passes (in ``"climb_one_way"`` mode), and executes safe ``z_safe``
       hops (``FEED`` retract, ``RAPID`` at ``z_safe``, ``FEED`` plunge) across bosses
       and between sub-bands.
    """
    stock_min, stock_max, pz0, islands = _extract_facing_geometry(task)
    inv_machine = task.work_to_machine.inverse().apply
    assert task.initial_state.position_mm is not None
    init_work = inv_machine(task.initial_state.position_mm)

    stock_max_z = stock_min[2] if stock_max[2] < stock_min[2] else stock_max[2]
    z_safe = max(float(init_work[2]), stock_max_z + 2.0)
    r = float(task.tool.cutter_radius_mm)

    waypoints: List[Tuple[C.SegmentKind, Point3D, Optional[float], str]] = []
    cur_xy: Point2D = (float(init_work[0]), float(init_work[1]))
    cur_z: float = float(init_work[2])

    def _retract_to_safe(tag: str = "retract") -> None:
        nonlocal cur_z
        if abs(cur_z - z_safe) > 1e-9:
            waypoints.append(
                (C.SegmentKind.FEED, (cur_xy[0], cur_xy[1], z_safe), config.retract_feed_mm_s, tag)
            )
            cur_z = z_safe

    def _rapid_and_plunge(target_xy: Point2D, z_cut: float, prefix: str) -> None:
        nonlocal cur_xy, cur_z
        _retract_to_safe(tag=f"{prefix}_retract")
        if math.hypot(target_xy[0] - cur_xy[0], target_xy[1] - cur_xy[1]) > 1e-6:
            waypoints.append(
                (C.SegmentKind.RAPID, (target_xy[0], target_xy[1], z_safe), None, f"{prefix}_approach")
            )
            cur_xy = target_xy
        waypoints.append(
            (C.SegmentKind.FEED, (target_xy[0], target_xy[1], z_cut), config.plunge_feed_mm_s, f"{prefix}_plunge")
        )
        cur_z = z_cut

    def _cut_to(target_xy: Point2D, z_cut: float, tag: str) -> None:
        nonlocal cur_xy, cur_z
        waypoints.append(
            (C.SegmentKind.FEED, (target_xy[0], target_xy[1], z_cut), config.nominal_feed_mm_s, tag)
        )
        cur_xy = target_xy
        cur_z = z_cut

    if cur_z < z_safe - 1e-9:
        _retract_to_safe(tag="initial_retract")

    z_layers = _compute_z_layers(stock_max_z, pz0, config.axial_stepdown_mm)

    for layer_idx, z_layer in enumerate(z_layers):
        active_bosses = [isl for isl in islands if isl[1][2] > z_layer + 1e-6]

        # 1. Exact cutter-offset moat contour pass around every active standing boss
        for b_idx in range(len(active_bosses)):
            moat_polylines = _build_moat_polylines_for_boss(
                task, b_idx, active_bosses, r, z_layer
            )
            for p_idx, poly in enumerate(moat_polylines):
                if len(poly) < 2:
                    continue
                _rapid_and_plunge(poly[0], z_layer, f"L{layer_idx}_moat{b_idx}_{p_idx}")
                for e_idx, pt in enumerate(poly[1:]):
                    _cut_to(pt, z_layer, f"L{layer_idx}_moat{b_idx}_{p_idx}_{e_idx}")
                _retract_to_safe(tag=f"L{layer_idx}_moat{b_idx}_{p_idx}_retract")

        # 2. Rotated scanlines clipped against all active expanded boss rectangles
        center, eu, ev, rows = _build_clipped_scanlines(
            task, stock_min, stock_max, active_bosses, config, z_layer
        )
        if not rows:
            continue

        def _uv_to_xy(v_val: float, u_val: float) -> Point2D:
            return (
                center[0] + v_val * ev[0] + u_val * eu[0],
                center[1] + v_val * ev[1] + u_val * eu[1],
            )

        if config.direction_mode == "climb_one_way":
            pass_idx = 0
            for _, v_val, intervals in rows:
                for ua, ub in intervals:
                    p_start = _uv_to_xy(v_val, ua)
                    p_end = _uv_to_xy(v_val, ub)
                    _rapid_and_plunge(p_start, z_layer, f"L{layer_idx}_climb{pass_idx}")
                    _cut_to(p_end, z_layer, f"L{layer_idx}_climb{pass_idx}_cut")
                    _retract_to_safe(tag=f"L{layer_idx}_climb{pass_idx}_retract")
                    pass_idx += 1
        else:
            chord_boss_boxes: List[Box2D] = [
                (
                    ilo[0] - r - _CHORD_BOSS_GUARD_MM,
                    ihi[0] + r + _CHORD_BOSS_GUARD_MM,
                    ilo[1] - r - _CHORD_BOSS_GUARD_MM,
                    ihi[1] + r + _CHORD_BOSS_GUARD_MM,
                )
                for ilo, ihi in active_bosses
            ]
            sub_bands = _decompose_into_sub_bands(center, eu, ev, rows, chord_boss_boxes)
            for band_idx, band in enumerate(sub_bands):
                if not band:
                    continue
                v0, ua0, ub0 = band[0]
                pa0 = _uv_to_xy(v0, ua0)
                pb0 = _uv_to_xy(v0, ub0)
                start_pos = (
                    math.hypot(pa0[0] - cur_xy[0], pa0[1] - cur_xy[1])
                    <= math.hypot(pb0[0] - cur_xy[0], pb0[1] - cur_xy[1]) + 1e-9
                )
                for k, (vk, uak, ubk) in enumerate(band):
                    go_pos = start_pos if (k % 2 == 0) else (not start_pos)
                    p_start = _uv_to_xy(vk, uak) if go_pos else _uv_to_xy(vk, ubk)
                    p_end = _uv_to_xy(vk, ubk) if go_pos else _uv_to_xy(vk, uak)
                    if k == 0:
                        _rapid_and_plunge(p_start, z_layer, f"L{layer_idx}_band{band_idx}")
                    else:
                        _cut_to(p_start, z_layer, f"L{layer_idx}_band{band_idx}_step{k}")
                    _cut_to(p_end, z_layer, f"L{layer_idx}_band{band_idx}_cut{k}")
                _retract_to_safe(tag=f"L{layer_idx}_band{band_idx}_retract")

    _retract_to_safe(tag="final_retract")
    return _waypoints_to_motion_plan(task, waypoints, PROVENANCE)


__all__ = [
    "FacingConfig",
    "PROVENANCE",
    "generate_facing_toolpath",
]
