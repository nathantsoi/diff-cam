# SPDX-License-Identifier: MIT
"""General-purpose 2.5D Multi-Boss Exterior & Island Contour Profiling strategy.

Implements a clean-room MIT 2.5D exterior and multi-boss contour generator:
1. Outside-in shoulder roughing down to ``R + finish_allowance_mm``:
   - Single-boss billets use clamped constant-offset ``C^1`` rounded-corner
     contour loops stepping inward at ``<= rough_stepover_fraction * 2 * R``.
   - Multi-boss billets use exact analytical scanline free-interval clearing
     chained into monotone bands plus per-boss ``R + finish_allowance_mm``
     roughing moat passes, guaranteeing 100% shoulder stock removal with zero
     boss collision.
2. High-precision finishing contour pass around every standing boss at exact
   offset ``R`` with discretized ``C^1`` rounded convex corners
   (``corner_roll_arc_segments``) and smooth tangential quarter-circle
   horizontal lead-in / lead-out arcs (``lead_in_radius_mm``) in cleared
   shoulder space at ``finish_feed_mm_s``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from benchmarks import contracts as C

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
WaypointSpec = Tuple[C.SegmentKind, Point3D, Optional[float], str]

PROVENANCE_ID = "mit_cleanroom:contour_strategy@2"


@dataclass(frozen=True)
class ContourConfig:
    """Configuration for 2.5D multi-boss exterior & island contour profiling."""

    rough_stepover_fraction: float = 0.65
    finish_allowance_mm: float = 0.25
    corner_roll_arc_segments: int = 5
    lead_in_radius_mm: float = 1.2
    milling_direction: str = "climb"
    axial_stepdown_mm: float | None = None
    nominal_feed_mm_s: float = 5.5
    finish_feed_mm_s: float = 4.5
    plunge_feed_mm_s: float = 4.0
    retract_feed_mm_s: float = 14.0

    def __post_init__(self) -> None:
        sf = float(self.rough_stepover_fraction)
        if not math.isfinite(sf) or not (0.0 < sf < 1.0):
            raise ValueError(
                f"rough_stepover_fraction must be finite in (0, 1), got {self.rough_stepover_fraction!r}"
            )
        fa = float(self.finish_allowance_mm)
        if not math.isfinite(fa) or fa < 0.0:
            raise ValueError(
                f"finish_allowance_mm must be finite and >= 0, got {self.finish_allowance_mm!r}"
            )
        if (
            isinstance(self.corner_roll_arc_segments, bool)
            or not isinstance(self.corner_roll_arc_segments, int)
            or self.corner_roll_arc_segments < 1
        ):
            raise ValueError(
                f"corner_roll_arc_segments must be an integer >= 1, got {self.corner_roll_arc_segments!r}"
            )
        lr = float(self.lead_in_radius_mm)
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError(
                f"lead_in_radius_mm must be finite and >= 0, got {self.lead_in_radius_mm!r}"
            )
        if self.milling_direction not in ("climb", "conventional"):
            raise ValueError(
                f"milling_direction must be 'climb' or 'conventional', got {self.milling_direction!r}"
            )
        if self.axial_stepdown_mm is not None:
            sd = float(self.axial_stepdown_mm)
            if not math.isfinite(sd) or sd <= 0.0:
                raise ValueError(
                    f"axial_stepdown_mm must be None or finite > 0, got {self.axial_stepdown_mm!r}"
                )
        for name in (
            "nominal_feed_mm_s",
            "finish_feed_mm_s",
            "plunge_feed_mm_s",
            "retract_feed_mm_s",
        ):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {getattr(self, name)!r}")


@dataclass(frozen=True)
class _Boss2D:
    idx: int
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float


def _point_boss_dist(x: float, y: float, boss: _Boss2D) -> float:
    dx = max(0.0, boss.x0 - x, x - boss.x1)
    dy = max(0.0, boss.y0 - y, y - boss.y1)
    return math.hypot(dx, dy)


def _seg_point_dist(
    ax: float, ay: float, bx: float, by: float, px: float, py: float
) -> float:
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-18:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _seg_intersects_box(
    ax: float, ay: float, bx: float, by: float, x0: float, x1: float, y0: float, y1: float
) -> bool:
    """Liang-Barsky exact closed-rectangle intersection test for a 2D line segment."""
    dx = bx - ax
    dy = by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if abs(p) <= 1e-15:
            if q < 0.0:
                return False
        else:
            r = q / p
            if p < 0.0:
                if r > t1:
                    return False
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                if r < t1:
                    t1 = r
    return t0 <= t1


def _seg_boss_dist(ax: float, ay: float, bx: float, by: float, boss: _Boss2D) -> float:
    """Exact 2D Euclidean distance between segment [(ax, ay), (bx, by)] and boss AABB."""
    if _seg_intersects_box(ax, ay, bx, by, boss.x0, boss.x1, boss.y0, boss.y1):
        return 0.0
    d = min(_point_boss_dist(ax, ay, boss), _point_boss_dist(bx, by, boss))
    for cx in (boss.x0, boss.x1):
        for cy in (boss.y0, boss.y1):
            d = min(d, _seg_point_dist(ax, ay, bx, by, cx, cy))
    return d


def _min_seg_clearance_to_bosses(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    bosses: Sequence[_Boss2D],
    skip_idx: Optional[int] = None,
) -> float:
    d_min = math.inf
    for b in bosses:
        if skip_idx is not None and b.idx == skip_idx:
            continue
        d_min = min(d_min, _seg_boss_dist(ax, ay, bx, by, b))
    return d_min


def _in_machine_envelope(
    task: C.TaskSpec, x: float, y: float, z: float, margin_mm: float = 0.25
) -> bool:
    mx, my, mz = task.work_to_machine.apply((x, y, z))
    lo = task.limits.envelope_min_mm
    hi = task.limits.envelope_max_mm
    return (
        (lo[0] + margin_mm <= mx <= hi[0] - margin_mm)
        and (lo[1] + margin_mm <= my <= hi[1] - margin_mm)
        and (lo[2] + margin_mm <= mz <= hi[2] - margin_mm)
    )


def _work_xy_envelope_bounds(task: C.TaskSpec, z_work: float, margin_mm: float = 0.5) -> Tuple[float, float, float, float]:
    """Conservative axis-aligned work-frame XY bounds inscribed in the machine envelope."""
    inv = task.work_to_machine.inverse()
    m = inv.matrix
    lo = task.limits.envelope_min_mm
    hi = task.limits.envelope_max_mm
    # If work_to_machine is an XY permutation/translation (standard horizontal setup),
    # map the machine box corners directly; otherwise compute inscribed box around stock.
    if (abs(abs(m[0][0]) - 1.0) <= 1e-9 and abs(m[1][0]) <= 1e-9) or (
        abs(m[0][0]) <= 1e-9 and abs(abs(m[1][0]) - 1.0) <= 1e-9
    ):
        corners = [
            inv.apply((mx, my, lo[2] + 0.5 * (hi[2] - lo[2])))
            for mx in (lo[0] + margin_mm, hi[0] - margin_mm)
            for my in (lo[1] + margin_mm, hi[1] - margin_mm)
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), max(xs), min(ys), max(ys)

    # Rotated XY setup: shrink from mapped AABB until all 4 corners lie inside machine envelope.
    raw_corners = [
        inv.apply((mx, my, lo[2] + 0.5 * (hi[2] - lo[2])))
        for mx in (lo[0], hi[0])
        for my in (lo[1], hi[1])
    ]
    cx = 0.5 * (min(c[0] for c in raw_corners) + max(c[0] for c in raw_corners))
    cy = 0.5 * (min(c[1] for c in raw_corners) + max(c[1] for c in raw_corners))
    hx = 0.5 * (max(c[0] for c in raw_corners) - min(c[0] for c in raw_corners))
    hy = 0.5 * (max(c[1] for c in raw_corners) - min(c[1] for c in raw_corners))
    lo_s, hi_s = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo_s + hi_s)
        ok = all(
            _in_machine_envelope(task, cx + sx * mid * hx, cy + sy * mid * hy, z_work, margin_mm)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        )
        if ok:
            lo_s = mid
        else:
            hi_s = mid
    return cx - lo_s * hx, cx + lo_s * hx, cy - lo_s * hy, cy + lo_s * hy


def _extract_contour_geometry(
    task: C.TaskSpec,
) -> Tuple[Tuple[float, float, float, float], float, float, List[_Boss2D]]:
    """Extract billet XY bounds, z_top, z_floor, and all boss specifications."""
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        raise ValueError("generate_contour_toolpath supports flat endmills only")
    if task.stock.kind != "analytic" or task.stock.params.get("primitive") != "box":
        raise ValueError("generate_contour_toolpath requires analytic box stock")
    part = task.original_part
    if part.kind != "analytic":
        raise ValueError("generate_contour_toolpath requires analytic original_part")

    p = part.params
    prim = p.get("primitive")
    if prim == "box_minus_multi_pocket":
        raw_islands = list(p.get("islands_mm", ()))
    elif prim == "box_minus_pocket":
        raw_islands = [(p["island_min_mm"], p["island_max_mm"])]
    else:
        raise ValueError(f"unsupported analytic primitive for contour profiling: {prim!r}")

    smin = [float(v) for v in task.stock.params["min_mm"]]
    smax = [float(v) for v in task.stock.params["max_mm"]]
    bmin = [float(v) for v in p["box_min_mm"]]
    bmax = [float(v) for v in p["box_max_mm"]]
    pmin = [float(v) for v in p["pocket_min_mm"]]
    pmax = [float(v) for v in p["pocket_max_mm"]]

    if (
        pmin[0] > bmin[0] + 1e-6
        or pmax[0] < bmax[0] - 1e-6
        or pmin[1] > bmin[1] + 1e-6
        or pmax[1] < bmax[1] - 1e-6
    ):
        raise ValueError(
            "generate_contour_toolpath requires an open-shoulder billet "
            "(pocket_min_mm..pocket_max_mm covering box_min_mm..box_max_mm in XY)"
        )

    z_top = float(smax[2])
    z_floor = float(pmin[2])
    if z_top <= z_floor + 1e-6:
        raise ValueError(f"contour depth must be positive (z_top={z_top}, z_floor={z_floor})")
    if float(pmax[2]) < z_top - 1e-6:
        raise ValueError("pocket_max_mm[2] must reach stock top z_top for open-shoulder contouring")

    bosses: List[_Boss2D] = []
    for idx, isl in enumerate(raw_islands):
        ilo, ihi = isl[0], isl[1]
        x0, y0, iz0 = float(ilo[0]), float(ilo[1]), float(ilo[2])
        x1, y1, iz1 = float(ihi[0]), float(ihi[1]), float(ihi[2])
        if x1 <= x0 + 1e-9 or y1 <= y0 + 1e-9:
            raise ValueError(f"island[{idx}] has non-positive XY extent")
        if iz1 > z_floor + 1e-6:
            bosses.append(_Boss2D(idx=idx, x0=x0, x1=x1, y0=y0, y1=y1, z0=iz0, z1=iz1))

    if not bosses:
        raise ValueError("generate_contour_toolpath requires at least one standing boss above z_floor")

    sx0 = min(smin[0], bmin[0])
    sx1 = max(smax[0], bmax[0])
    sy0 = min(smin[1], bmin[1])
    sy1 = max(smax[1], bmax[1])
    return (sx0, sx1, sy0, sy1), z_top, z_floor, bosses


def _boss_wall_corner_defs(
    boss: _Boss2D, offset_d: float, climb: bool
) -> List[
    Tuple[
        str,
        Point2D,
        Point2D,
        Point2D,
        Point2D,
        str,
        Point2D,
        Point2D,
        Point2D,
    ]
]:
    """Ordered (wall_name, p_start, p_end, t_hat, n_hat, corner_name, vertex, e_in, e_out) tuples.

    ``climb=True`` traverses CW around the exterior boss (for right-hand M03 spindle);
    ``climb=False`` traverses CCW around the exterior boss.
    """
    x0, x1, y0, y1 = boss.x0, boss.x1, boss.y0, boss.y1
    d = offset_d
    if climb:
        return [
            (
                "bottom",
                (x1, y0 - d),
                (x0, y0 - d),
                (-1.0, 0.0),
                (0.0, -1.0),
                "bl",
                (x0, y0),
                (0.0, -1.0),
                (-1.0, 0.0),
            ),
            (
                "left",
                (x0 - d, y0),
                (x0 - d, y1),
                (0.0, 1.0),
                (-1.0, 0.0),
                "tl",
                (x0, y1),
                (-1.0, 0.0),
                (0.0, 1.0),
            ),
            (
                "top",
                (x0, y1 + d),
                (x1, y1 + d),
                (1.0, 0.0),
                (0.0, 1.0),
                "tr",
                (x1, y1),
                (0.0, 1.0),
                (1.0, 0.0),
            ),
            (
                "right",
                (x1 + d, y1),
                (x1 + d, y0),
                (0.0, -1.0),
                (1.0, 0.0),
                "br",
                (x1, y0),
                (1.0, 0.0),
                (0.0, -1.0),
            ),
        ]
    return [
        (
            "bottom",
            (x0, y0 - d),
            (x1, y0 - d),
            (1.0, 0.0),
            (0.0, -1.0),
            "br",
            (x1, y0),
            (0.0, -1.0),
            (1.0, 0.0),
        ),
        (
            "right",
            (x1 + d, y0),
            (x1 + d, y1),
            (0.0, 1.0),
            (1.0, 0.0),
            "tr",
            (x1, y1),
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        (
            "top",
            (x1, y1 + d),
            (x0, y1 + d),
            (-1.0, 0.0),
            (0.0, 1.0),
            "tl",
            (x0, y1),
            (0.0, 1.0),
            (-1.0, 0.0),
        ),
        (
            "left",
            (x0 - d, y1),
            (x0 - d, y0),
            (0.0, -1.0),
            (-1.0, 0.0),
            "bl",
            (x0, y0),
            (-1.0, 0.0),
            (0.0, -1.0),
        ),
    ]


def _quarter_arc_targets(
    vertex: Point2D,
    radius: float,
    e_in: Point2D,
    e_out: Point2D,
    n_segs: int,
) -> List[Point2D]:
    """Return the ``n_segs`` target points along a 90 deg corner arc (excluding start, including end)."""
    vx, vy = vertex
    pts: List[Point2D] = []
    for k in range(1, n_segs + 1):
        if k == n_segs:
            pts.append((vx + radius * e_out[0], vy + radius * e_out[1]))
        else:
            phi = 0.5 * math.pi * (k / n_segs)
            c = math.cos(phi)
            s = math.sin(phi)
            ux = c * e_in[0] + s * e_out[0]
            uy = c * e_in[1] + s * e_out[1]
            pts.append((vx + radius * ux, vy + radius * uy))
    return pts


def _boss_loop_targets_from_anchor(
    boss: _Boss2D,
    offset_d: float,
    n_corner_segs: int,
    climb: bool,
    anchor_wall_idx: int,
    anchor_u: float,
    tag_prefix: str,
) -> Tuple[Point2D, Point2D, Point2D, List[Tuple[Point2D, str]]]:
    """Build closed C^1 constant-offset boss loop starting & ending at wall anchor ``p0``.

    Returns ``(p0, t_hat, n_hat, [(target_xy, tag), ...])``.
    """
    defs = _boss_wall_corner_defs(boss, offset_d, climb)
    w0_name, w0_start, w0_end, t0, n0, c0_name, c0_v, c0_ein, c0_eout = defs[anchor_wall_idx]
    p0 = (
        w0_start[0] + anchor_u * (w0_end[0] - w0_start[0]),
        w0_start[1] + anchor_u * (w0_end[1] - w0_start[1]),
    )

    loop_steps: List[Tuple[Point2D, str]] = []
    # Second part of anchor wall -> its outgoing corner
    if math.hypot(w0_end[0] - p0[0], w0_end[1] - p0[1]) > 1e-9:
        loop_steps.append((w0_end, f"{tag_prefix}_wall_{w0_name}_a"))
    for k, pt in enumerate(_quarter_arc_targets(c0_v, offset_d, c0_ein, c0_eout, n_corner_segs)):
        loop_steps.append((pt, f"{tag_prefix}_corner_{c0_name}_{k}"))

    # Remaining 3 walls and corners
    for shift in (1, 2, 3):
        w_name, _, w_end, _, _, c_name, c_v, c_ein, c_eout = defs[(anchor_wall_idx + shift) % 4]
        loop_steps.append((w_end, f"{tag_prefix}_wall_{w_name}"))
        for k, pt in enumerate(_quarter_arc_targets(c_v, offset_d, c_ein, c_eout, n_corner_segs)):
            loop_steps.append((pt, f"{tag_prefix}_corner_{c_name}_{k}"))

    # First part of anchor wall closing back to p0
    if math.hypot(p0[0] - w0_start[0], p0[1] - w0_start[1]) > 1e-9:
        loop_steps.append((p0, f"{tag_prefix}_wall_{w0_name}_b"))

    return p0, t0, n0, loop_steps


def _emit_single_boss_roughing(
    task: C.TaskSpec,
    boss: _Boss2D,
    stock_xy: Tuple[float, float, float, float],
    env_xy: Tuple[float, float, float, float],
    z_layer: float,
    z_safe: float,
    r_tool: float,
    config: ContourConfig,
    layer_idx: int,
    waypoints: List[WaypointSpec],
) -> None:
    """Outside-in clamped constant-offset C^1 rounded-corner roughing for a single boss."""
    sx0, sx1, sy0, sy1 = stock_xy
    ex0, ex1, ey0, ey1 = env_xy
    climb = config.milling_direction == "climb"
    d_inner = r_tool + config.finish_allowance_mm
    step_mm = config.rough_stepover_fraction * (2.0 * r_tool)

    # Safe clamping box: non-expansive projection containing stock + d_inner boss moat,
    # bounded strictly within machine XY envelope.
    x_min = max(ex0, min(boss.x0 - d_inner, sx0 - 0.35 * r_tool))
    x_max = min(ex1, max(boss.x1 + d_inner, sx1 + 0.35 * r_tool))
    y_min = max(ey0, min(boss.y0 - d_inner, sy0 - 0.35 * r_tool))
    y_max = min(ey1, max(boss.y1 + d_inner, sy1 + 0.35 * r_tool))

    d_corners = [
        _point_boss_dist(sx, sy, boss)
        for sx in (sx0, sx1)
        for sy in (sy0, sy1)
    ]
    d_outer = max(d_inner, max(d_corners) - 0.30 * r_tool)

    if d_outer <= d_inner + 1e-6:
        offsets = [d_inner]
    else:
        n_steps = max(1, int(math.ceil((d_outer - d_inner) / step_mm)))
        offsets = [d_outer - (d_outer - d_inner) * (i / n_steps) for i in range(n_steps + 1)]

    # Pick anchor wall with largest exterior stock shoulder so step-in moves occur in wide shoulder.
    defs_ref = _boss_wall_corner_defs(boss, d_inner, climb)
    shoulder_widths = {
        "bottom": boss.y0 - sy0,
        "top": sy1 - boss.y1,
        "left": boss.x0 - sx0,
        "right": sx1 - boss.x1,
    }
    anchor_wall_idx = max(range(4), key=lambda idx: shoulder_widths[defs_ref[idx][0]])

    def _clip_xy(pt: Point2D) -> Point2D:
        return (
            min(max(pt[0], x_min), x_max),
            min(max(pt[1], y_min), y_max),
        )

    started = False
    for loop_idx, d_cur in enumerate(offsets):
        # Refine roughing corner discretization on large outer offsets so clipped arcs stay dense.
        n_arc = max(
            config.corner_roll_arc_segments,
            int(math.ceil((0.5 * math.pi * d_cur) / max(0.5 * r_tool, 0.25))),
        )
        p0_raw, _, _, raw_steps = _boss_loop_targets_from_anchor(
            boss=boss,
            offset_d=d_cur,
            n_corner_segs=n_arc,
            climb=climb,
            anchor_wall_idx=anchor_wall_idx,
            anchor_u=0.5,
            tag_prefix=f"L{layer_idx}_rough_{loop_idx}",
        )
        p0 = _clip_xy(p0_raw)
        if not started:
            waypoints.append(
                (C.SegmentKind.RAPID, (p0[0], p0[1], z_safe), None, f"L{layer_idx}_rough_approach")
            )
            waypoints.append(
                (
                    C.SegmentKind.FEED,
                    (p0[0], p0[1], z_layer),
                    config.plunge_feed_mm_s,
                    f"L{layer_idx}_rough_plunge",
                )
            )
            started = True
        else:
            waypoints.append(
                (
                    C.SegmentKind.FEED,
                    (p0[0], p0[1], z_layer),
                    config.nominal_feed_mm_s,
                    f"L{layer_idx}_rough_step_in_{loop_idx}",
                )
            )

        clipped_chain: List[Point2D] = [p0]
        for target_raw, _ in raw_steps:
            target_xy = _clip_xy(target_raw)
            if math.hypot(target_xy[0] - clipped_chain[-1][0], target_xy[1] - clipped_chain[-1][1]) > 1e-5:
                clipped_chain.append(target_xy)
        simplified_chain = _simplify_collinear_2d(clipped_chain)
        for seg_i, target_xy in enumerate(simplified_chain[1:]):
            waypoints.append(
                (
                    C.SegmentKind.FEED,
                    (target_xy[0], target_xy[1], z_layer),
                    config.nominal_feed_mm_s,
                    f"L{layer_idx}_rough_{loop_idx}_seg_{seg_i}",
                )
            )

    if started:
        last_xyz = waypoints[-1][1]
        waypoints.append(
            (
                C.SegmentKind.FEED,
                (last_xyz[0], last_xyz[1], z_safe),
                config.retract_feed_mm_s,
                f"L{layer_idx}_rough_retract",
            )
        )


def _subtract_intervals(
    base_lo: float, base_hi: float, blocked: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Subtract open intervals ``(blo, bhi)`` from ``[base_lo, base_hi]``."""
    if base_hi < base_lo - 1e-9:
        return []
    clipped = [
        (max(base_lo, blo), min(base_hi, bhi))
        for blo, bhi in blocked
        if bhi > base_lo + 1e-9 and blo < base_hi - 1e-9 and bhi > blo + 1e-9
    ]
    if not clipped:
        return [(base_lo, base_hi)]
    clipped.sort(key=lambda iv: iv[0])
    merged: List[Tuple[float, float]] = [clipped[0]]
    for blo, bhi in clipped[1:]:
        mlo, mhi = merged[-1]
        if blo <= mhi + 1e-9:
            merged[-1] = (mlo, max(mhi, bhi))
        else:
            merged.append((blo, bhi))

    free: List[Tuple[float, float]] = []
    cursor = base_lo
    for blo, bhi in merged:
        if blo > cursor + 1e-6:
            free.append((cursor, blo))
        cursor = max(cursor, bhi)
    if base_hi > cursor + 1e-6:
        free.append((cursor, base_hi))
    return free


def _free_x_intervals_at_y(
    y: float,
    x_lo: float,
    x_hi: float,
    bosses: Sequence[_Boss2D],
    d_rough: float,
) -> List[Tuple[float, float]]:
    blocked: List[Tuple[float, float]] = []
    for b in bosses:
        dy = max(0.0, b.y0 - y, y - b.y1)
        if dy < d_rough - 1e-12:
            dx_pad = math.sqrt(max(0.0, d_rough * d_rough - dy * dy))
            blocked.append((b.x0 - dx_pad, b.x1 + dx_pad))
    return _subtract_intervals(x_lo, x_hi, blocked)


def _free_x_intervals_on_y_slab(
    ya: float,
    yb: float,
    x_lo: float,
    x_hi: float,
    bosses: Sequence[_Boss2D],
    d_rough: float,
) -> List[Tuple[float, float]]:
    """X intervals guaranteed collision-free across the entire vertical slab y in [ya, yb]."""
    ylo, yhi = (ya, yb) if ya <= yb else (yb, ya)
    blocked: List[Tuple[float, float]] = []
    for b in bosses:
        dy_min = max(0.0, b.y0 - yhi, ylo - b.y1)
        if dy_min < d_rough - 1e-12:
            dx_pad = math.sqrt(max(0.0, d_rough * d_rough - dy_min * dy_min))
            blocked.append((b.x0 - dx_pad, b.x1 + dx_pad))
    return _subtract_intervals(x_lo, x_hi, blocked)


def _simplify_collinear_2d(pts: Sequence[Point2D], tol: float = 1e-7) -> List[Point2D]:
    if len(pts) <= 2:
        return list(pts)
    out: List[Point2D] = [pts[0]]
    for i in range(1, len(pts) - 1):
        ax, ay = out[-1]
        bx, by = pts[i]
        cx, cy = pts[i + 1]
        ux, uy = bx - ax, by - ay
        vx, vy = cx - bx, cy - by
        lu = math.hypot(ux, uy)
        lv = math.hypot(vx, vy)
        if lu <= 1e-7:
            continue
        if lv <= 1e-7:
            continue
        cross = abs(ux * vy - uy * vx) / (lu + lv)
        dot = ux * vx + uy * vy
        if cross <= tol and dot > 0.0:
            continue
        out.append(pts[i])
    if math.hypot(pts[-1][0] - out[-1][0], pts[-1][1] - out[-1][1]) > 1e-7:
        out.append(pts[-1])
    return out


def _emit_multi_boss_roughing(
    task: C.TaskSpec,
    bosses: Sequence[_Boss2D],
    stock_xy: Tuple[float, float, float, float],
    env_xy: Tuple[float, float, float, float],
    z_layer: float,
    z_safe: float,
    r_tool: float,
    config: ContourConfig,
    layer_idx: int,
    waypoints: List[WaypointSpec],
) -> None:
    """Multi-boss shoulder & channel roughing down to ``R + finish_allowance_mm``.

    Combines:
    1. Exact analytical scanline free-interval decomposition chained into monotone
       boustrophedon bands across the entire billet shoulder and inter-boss channels.
    2. Per-boss constant-offset ``R + finish_allowance_mm`` rounded-corner roughing
       moat loops (clipped safely against adjacent bosses and machine envelope).
    """
    sx0, sx1, sy0, sy1 = stock_xy
    ex0, ex1, ey0, ey1 = env_xy
    climb = config.milling_direction == "climb"
    d_rough = r_tool + config.finish_allowance_mm
    step_mm = min(config.rough_stepover_fraction * (2.0 * r_tool), 1.35 * r_tool)

    x_left = max(ex0, sx0 - 0.35 * r_tool)
    x_right = min(ex1, sx1 + 0.35 * r_tool)
    y_lo = max(ey0, sy0 + 0.35 * r_tool)
    y_hi = min(ey1, sy1 - 0.35 * r_tool)
    if y_hi < y_lo:
        y_lo = y_hi = 0.5 * (max(ey0, sy0) + min(ey1, sy1))

    # Build Y stations including exact tangent stations just outside horizontal boss moats
    # so horizontal walls never have an uncovered strip between scanlines and moats.
    critical_ys = [y_lo, y_hi]
    for b in bosses:
        for yc in (b.y0 - d_rough - 1e-4, b.y1 + d_rough + 1e-4):
            if y_lo + 1e-3 < yc < y_hi - 1e-3:
                critical_ys.append(yc)
    critical_ys = sorted(critical_ys)
    dedup_crit: List[float] = [critical_ys[0]]
    for yc in critical_ys[1:]:
        if yc - dedup_crit[-1] > 0.15 * r_tool:
            dedup_crit.append(yc)
    if abs(dedup_crit[-1] - y_hi) > 1e-6:
        dedup_crit[-1] = y_hi

    y_rows: List[float] = [dedup_crit[0]]
    for k in range(len(dedup_crit) - 1):
        ya, yb = dedup_crit[k], dedup_crit[k + 1]
        n_sub = max(1, int(math.ceil((yb - ya) / step_mm)))
        for s in range(1, n_sub + 1):
            y_rows.append(ya + (yb - ya) * (s / n_sub))

    row_intervals: List[List[Tuple[float, float]]] = [
        _free_x_intervals_at_y(y, x_left, x_right, bosses, d_rough) for y in y_rows
    ]
    visited: List[List[bool]] = [[False] * len(ivs) for ivs in row_intervals]

    # 1. Monotone band chaining of scanline intervals
    chain_id = 0
    for r0 in range(len(y_rows)):
        for s0 in range(len(row_intervals[r0])):
            if visited[r0][s0]:
                continue
            visited[r0][s0] = True
            xa, xb = row_intervals[r0][s0]
            y_cur = y_rows[r0]
            waypoints.append(
                (
                    C.SegmentKind.RAPID,
                    (xa, y_cur, z_safe),
                    None,
                    f"L{layer_idx}_band{chain_id}_approach",
                )
            )
            waypoints.append(
                (
                    C.SegmentKind.FEED,
                    (xa, y_cur, z_layer),
                    config.plunge_feed_mm_s,
                    f"L{layer_idx}_band{chain_id}_plunge",
                )
            )
            if xb - xa > 1e-5:
                waypoints.append(
                    (
                        C.SegmentKind.FEED,
                        (xb, y_cur, z_layer),
                        config.nominal_feed_mm_s,
                        f"L{layer_idx}_band{chain_id}_cut_0",
                    )
                )
            cur_x = xb
            r_cur = r0
            step_idx = 1

            while r_cur + 1 < len(y_rows):
                y_nxt = y_rows[r_cur + 1]
                slab_free = _free_x_intervals_on_y_slab(
                    y_cur, y_nxt, x_left, x_right, bosses, d_rough
                )
                best_choice: Optional[Tuple[int, float, float, float, bool]] = None
                for s_nxt, (nxa, nxb) in enumerate(row_intervals[r_cur + 1]):
                    if visited[r_cur + 1][s_nxt]:
                        continue
                    # Direct diagonal link check (choose nearer endpoint of next interval)
                    near_x = nxa if abs(cur_x - nxa) <= abs(cur_x - nxb) else nxb
                    far_x = nxb if near_x == nxa else nxa
                    if (
                        _min_seg_clearance_to_bosses(cur_x, y_cur, near_x, y_nxt, bosses)
                        >= d_rough - 1e-5
                    ):
                        best_choice = (s_nxt, near_x, far_x, cur_x, True)
                        break
                    # Safe vertical bridge through a slab-free X interval overlapping both rows
                    for s_lo, s_hi in slab_free:
                        ov_lo = max(xa, nxa, s_lo)
                        ov_hi = min(xb, nxb, s_hi)
                        if ov_hi >= ov_lo - 1e-6:
                            x_bridge = min(max(cur_x, ov_lo), ov_hi)
                            go_right = abs(x_bridge - nxa) <= abs(x_bridge - nxb)
                            nx_start = nxa if go_right else nxb
                            nx_end = nxb if go_right else nxa
                            best_choice = (s_nxt, nx_start, nx_end, x_bridge, False)
                            break
                    if best_choice is not None:
                        break

                if best_choice is None:
                    break

                s_nxt, nx_start, nx_end, x_bridge, is_direct = best_choice
                visited[r_cur + 1][s_nxt] = True
                if is_direct:
                    waypoints.append(
                        (
                            C.SegmentKind.FEED,
                            (nx_start, y_nxt, z_layer),
                            config.nominal_feed_mm_s,
                            f"L{layer_idx}_band{chain_id}_step_{step_idx}",
                        )
                    )
                else:
                    if abs(x_bridge - cur_x) > 1e-5:
                        waypoints.append(
                            (
                                C.SegmentKind.FEED,
                                (x_bridge, y_cur, z_layer),
                                config.nominal_feed_mm_s,
                                f"L{layer_idx}_band{chain_id}_jog_{step_idx}a",
                            )
                        )
                    waypoints.append(
                        (
                            C.SegmentKind.FEED,
                            (x_bridge, y_nxt, z_layer),
                            config.nominal_feed_mm_s,
                            f"L{layer_idx}_band{chain_id}_bridge_{step_idx}",
                        )
                    )
                    if abs(nx_start - x_bridge) > 1e-5:
                        waypoints.append(
                            (
                                C.SegmentKind.FEED,
                                (nx_start, y_nxt, z_layer),
                                config.nominal_feed_mm_s,
                                f"L{layer_idx}_band{chain_id}_jog_{step_idx}b",
                            )
                        )

                if abs(nx_end - nx_start) > 1e-5:
                    waypoints.append(
                        (
                            C.SegmentKind.FEED,
                            (nx_end, y_nxt, z_layer),
                            config.nominal_feed_mm_s,
                            f"L{layer_idx}_band{chain_id}_cut_{step_idx}",
                        )
                    )
                cur_x = nx_end
                xa, xb = row_intervals[r_cur + 1][s_nxt]
                y_cur = y_nxt
                r_cur += 1
                step_idx += 1

            waypoints.append(
                (
                    C.SegmentKind.FEED,
                    (cur_x, y_cur, z_safe),
                    config.retract_feed_mm_s,
                    f"L{layer_idx}_band{chain_id}_retract",
                )
            )
            chain_id += 1

    # 2. Per-boss roughing moat pass at exact offset d_rough = R + finish_allowance_mm
    for b in bosses:
        unclipped_ok = (
            b.x0 - d_rough >= ex0
            and b.x1 + d_rough <= ex1
            and b.y0 - d_rough >= ey0
            and b.y1 + d_rough <= ey1
        )
        if unclipped_ok:
            _, _, _, check_steps = _boss_loop_targets_from_anchor(
                boss=b,
                offset_d=d_rough,
                n_corner_segs=config.corner_roll_arc_segments,
                climb=climb,
                anchor_wall_idx=0,
                anchor_u=0.5,
                tag_prefix=f"L{layer_idx}_moat_b{b.idx}",
            )
            p_prev = check_steps[-1][0]
            for pt_nxt, _ in check_steps:
                if (
                    _min_seg_clearance_to_bosses(
                        p_prev[0], p_prev[1], pt_nxt[0], pt_nxt[1], bosses, skip_idx=b.idx
                    )
                    < d_rough - 1e-5
                ):
                    unclipped_ok = False
                    break
                p_prev = pt_nxt

        if unclipped_ok:
            p0, _, _, moat_steps = _boss_loop_targets_from_anchor(
                boss=b,
                offset_d=d_rough,
                n_corner_segs=config.corner_roll_arc_segments,
                climb=climb,
                anchor_wall_idx=0,
                anchor_u=0.5,
                tag_prefix=f"L{layer_idx}_moat_b{b.idx}",
            )
            waypoints.append(
                (
                    C.SegmentKind.RAPID,
                    (p0[0], p0[1], z_safe),
                    None,
                    f"L{layer_idx}_moat_b{b.idx}_approach",
                )
            )
            waypoints.append(
                (
                    C.SegmentKind.FEED,
                    (p0[0], p0[1], z_layer),
                    config.plunge_feed_mm_s,
                    f"L{layer_idx}_moat_b{b.idx}_plunge",
                )
            )
            for pt, seg_tag in moat_steps:
                waypoints.append(
                    (
                        C.SegmentKind.FEED,
                        (pt[0], pt[1], z_layer),
                        config.nominal_feed_mm_s,
                        seg_tag,
                    )
                )
            waypoints.append(
                (
                    C.SegmentKind.FEED,
                    (p0[0], p0[1], z_safe),
                    config.retract_feed_mm_s,
                    f"L{layer_idx}_moat_b{b.idx}_retract",
                )
            )
            continue

        # Fine sampling + clipping against adjacent boss d_rough zones and machine envelope
        p0, _, _, coarse_steps = _boss_loop_targets_from_anchor(
            boss=b,
            offset_d=d_rough,
            n_corner_segs=max(config.corner_roll_arc_segments, 12),
            climb=climb,
            anchor_wall_idx=0,
            anchor_u=0.5,
            tag_prefix=f"L{layer_idx}_moat_b{b.idx}",
        )
        dense_pts: List[Point2D] = [p0]
        cur_p = p0
        for nxt_p, _ in coarse_steps:
            dist = math.hypot(nxt_p[0] - cur_p[0], nxt_p[1] - cur_p[1])
            n_div = max(1, int(math.ceil(dist / 0.2)))
            for j in range(1, n_div + 1):
                alpha = j / n_div
                dense_pts.append(
                    (
                        cur_p[0] + alpha * (nxt_p[0] - cur_p[0]),
                        cur_p[1] + alpha * (nxt_p[1] - cur_p[1]),
                    )
                )
            cur_p = nxt_p

        def _pt_safe(pt: Point2D) -> bool:
            if not (ex0 <= pt[0] <= ex1 and ey0 <= pt[1] <= ey1):
                return False
            for ob in bosses:
                if ob.idx == b.idx:
                    continue
                if _point_boss_dist(pt[0], pt[1], ob) < d_rough - 1e-5:
                    return False
            return True

        arcs: List[List[Point2D]] = []
        cur_arc: List[Point2D] = []
        for pt in dense_pts:
            if _pt_safe(pt):
                if (
                    cur_arc
                    and _min_seg_clearance_to_bosses(
                        cur_arc[-1][0], cur_arc[-1][1], pt[0], pt[1], bosses, skip_idx=b.idx
                    )
                    < r_tool + 1e-4
                ):
                    if len(cur_arc) >= 2:
                        arcs.append(_simplify_collinear_2d(cur_arc))
                    cur_arc = [pt]
                else:
                    cur_arc.append(pt)
            else:
                if len(cur_arc) >= 2:
                    arcs.append(_simplify_collinear_2d(cur_arc))
                cur_arc = []
        if len(cur_arc) >= 2:
            arcs.append(_simplify_collinear_2d(cur_arc))

        for a_idx, arc_pts in enumerate(arcs):
            if len(arc_pts) < 2:
                continue
            s_pt = arc_pts[0]
            waypoints.append(
                (
                    C.SegmentKind.RAPID,
                    (s_pt[0], s_pt[1], z_safe),
                    None,
                    f"L{layer_idx}_moat_b{b.idx}_{a_idx}_approach",
                )
            )
            waypoints.append(
                (
                    C.SegmentKind.FEED,
                    (s_pt[0], s_pt[1], z_layer),
                    config.plunge_feed_mm_s,
                    f"L{layer_idx}_moat_b{b.idx}_{a_idx}_plunge",
                )
            )
            for p_idx, m_pt in enumerate(arc_pts[1:]):
                waypoints.append(
                    (
                        C.SegmentKind.FEED,
                        (m_pt[0], m_pt[1], z_layer),
                        config.nominal_feed_mm_s,
                        f"L{layer_idx}_moat_b{b.idx}_{a_idx}_cut_{p_idx}",
                    )
                )
            e_pt = arc_pts[-1]
            waypoints.append(
                (
                    C.SegmentKind.FEED,
                    (e_pt[0], e_pt[1], z_safe),
                    config.retract_feed_mm_s,
                    f"L{layer_idx}_moat_b{b.idx}_{a_idx}_retract",
                )
            )


def _build_quarter_lead_points(
    p0: Point2D,
    t_hat: Point2D,
    n_hat: Point2D,
    radius: float,
    n_segs: int,
) -> Tuple[List[Point2D], List[Point2D]]:
    """Build tangential quarter-circle horizontal lead-in and lead-out polylines at ``p0``."""
    cx = p0[0] + radius * n_hat[0]
    cy = p0[1] + radius * n_hat[1]

    lead_in: List[Point2D] = []
    for k in range(n_segs + 1):
        if k == 0:
            lead_in.append((cx - radius * t_hat[0], cy - radius * t_hat[1]))
        elif k == n_segs:
            lead_in.append(p0)
        else:
            phi = 0.5 * math.pi * (k / n_segs)
            c = math.cos(phi)
            s = math.sin(phi)
            lead_in.append(
                (
                    cx - radius * (c * t_hat[0] + s * n_hat[0]),
                    cy - radius * (c * t_hat[1] + s * n_hat[1]),
                )
            )

    lead_out: List[Point2D] = []
    for k in range(n_segs + 1):
        if k == 0:
            lead_out.append(p0)
        elif k == n_segs:
            lead_out.append((cx + radius * t_hat[0], cy + radius * t_hat[1]))
        else:
            phi = 0.5 * math.pi * (k / n_segs)
            c = math.cos(phi)
            s = math.sin(phi)
            lead_out.append(
                (
                    cx - radius * c * n_hat[0] + radius * s * t_hat[0],
                    cy - radius * c * n_hat[1] + radius * s * t_hat[1],
                )
            )
    return lead_in, lead_out


def _select_finish_entry_and_leads(
    task: C.TaskSpec,
    boss: _Boss2D,
    all_bosses: Sequence[_Boss2D],
    stock_xy: Tuple[float, float, float, float],
    z_layer: float,
    r_tool: float,
    config: ContourConfig,
) -> Tuple[int, float, List[Point2D], List[Point2D]]:
    """Select the safest wall anchor and tangential quarter-circle (or shortened/off-wall) leads."""
    climb = config.milling_direction == "climb"
    defs = _boss_wall_corner_defs(boss, r_tool, climb)
    n_lead_segs = max(4, config.corner_roll_arc_segments)
    sx0, sx1, sy0, sy1 = stock_xy
    shoulder_map = {
        "bottom": boss.y0 - sy0,
        "top": sy1 - boss.y1,
        "left": boss.x0 - sx0,
        "right": sx1 - boss.x1,
    }

    def _lead_chain_valid(pts: Sequence[Point2D], min_other_clear: float) -> Tuple[bool, float]:
        min_clear = math.inf
        for pt in pts:
            if not _in_machine_envelope(task, pt[0], pt[1], z_layer, margin_mm=0.35):
                return False, 0.0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if _seg_boss_dist(a[0], a[1], b[0], b[1], boss) < r_tool - 1e-7:
                return False, 0.0
            other_d = _min_seg_clearance_to_bosses(
                a[0], a[1], b[0], b[1], all_bosses, skip_idx=boss.idx
            )
            if other_d < min_other_clear:
                return False, 0.0
            min_clear = min(min_clear, other_d)
        return True, min_clear

    best_score = (-1.0, -1.0, -math.inf, -math.inf)
    best_result: Optional[Tuple[int, float, List[Point2D], List[Point2D]]] = None

    req_other_clear = r_tool + max(0.02, min(0.20, config.finish_allowance_mm))
    for min_clear_target in (req_other_clear, r_tool + 0.01):
        for w_idx in range(4):
            w_name, w_start, w_end, t_hat, n_hat, _, _, _, _ = defs[w_idx]
            ext_shoulder = shoulder_map[w_name]
            for u in (0.5, 0.3, 0.7):
                p0 = (
                    w_start[0] + u * (w_end[0] - w_start[0]),
                    w_start[1] + u * (w_end[1] - w_start[1]),
                )
                if not _in_machine_envelope(task, p0[0], p0[1], z_layer, margin_mm=0.35):
                    continue
                if any(
                    _point_boss_dist(p0[0], p0[1], ob) < r_tool - 1e-6
                    for ob in all_bosses
                    if ob.idx != boss.idx
                ):
                    continue

                r_req = config.lead_in_radius_mm
                achieved_r = 0.0
                chosen_in = [p0]
                chosen_out = [p0]
                achieved_clear = math.inf

                if r_req > 1e-6:
                    li_full, lo_full = _build_quarter_lead_points(
                        p0, t_hat, n_hat, r_req, n_lead_segs
                    )
                    ok_in, cl_in = _lead_chain_valid(li_full, min_clear_target)
                    ok_out, cl_out = _lead_chain_valid(lo_full, min_clear_target)
                    if ok_in and ok_out:
                        achieved_r = r_req
                        chosen_in = li_full
                        chosen_out = lo_full
                        achieved_clear = min(cl_in, cl_out)
                    else:
                        lo_r, hi_r = 0.0, r_req
                        for _ in range(18):
                            mid_r = 0.5 * (lo_r + hi_r)
                            li_m, lo_m = _build_quarter_lead_points(
                                p0, t_hat, n_hat, mid_r, n_lead_segs
                            )
                            ok_i, cl_i = _lead_chain_valid(li_m, min_clear_target)
                            ok_o, cl_o = _lead_chain_valid(lo_m, min_clear_target)
                            if ok_i and ok_o:
                                lo_r = mid_r
                                achieved_r = mid_r
                                chosen_in = li_m
                                chosen_out = lo_m
                                achieved_clear = min(cl_i, cl_o)
                            else:
                                hi_r = mid_r

                if achieved_r < 0.12:
                    # Fallback: safe off-wall normal entry/exit in cleared shoulder space
                    max_norm = max(0.35, config.finish_allowance_mm)
                    lo_n, hi_n = 0.0, max_norm
                    best_n = 0.0
                    for _ in range(14):
                        mid_n = 0.5 * (lo_n + hi_n)
                        p_off = (p0[0] + mid_n * n_hat[0], p0[1] + mid_n * n_hat[1])
                        ok_n, cl_n = _lead_chain_valid([p_off, p0], min_clear_target)
                        if ok_n:
                            lo_n = mid_n
                            best_n = mid_n
                            achieved_clear = cl_n
                        else:
                            hi_n = mid_n
                    if best_n > 1e-4:
                        p_off = (p0[0] + best_n * n_hat[0], p0[1] + best_n * n_hat[1])
                        chosen_in = [p_off, p0]
                        chosen_out = [p0, p_off]
                    else:
                        chosen_in = [p0]
                        chosen_out = [p0]

                mid_bonus = 1.0 if abs(u - 0.5) < 1e-9 else 0.0
                eff_other = min(achieved_clear, 50.0)
                score = (round(achieved_r, 6), mid_bonus, round(eff_other, 4), ext_shoulder)
                if score > best_score:
                    best_score = score
                    best_result = (w_idx, u, chosen_in, chosen_out)

        if best_result is not None and best_score[0] >= min(0.15, config.lead_in_radius_mm):
            break

    if best_result is None:
        raise ValueError(f"no collision-free finishing contour entry found for boss {boss.idx}")
    return best_result


def _emit_boss_finishing_pass(
    task: C.TaskSpec,
    boss: _Boss2D,
    all_bosses: Sequence[_Boss2D],
    stock_xy: Tuple[float, float, float, float],
    z_layer: float,
    z_safe: float,
    r_tool: float,
    config: ContourConfig,
    layer_idx: int,
    waypoints: List[WaypointSpec],
) -> None:
    """Emit dedicated finishing contour pass with exact radius R corners and tangential leads."""
    climb = config.milling_direction == "climb"
    w_idx, u_anchor, lead_in_pts, lead_out_pts = _select_finish_entry_and_leads(
        task=task,
        boss=boss,
        all_bosses=all_bosses,
        stock_xy=stock_xy,
        z_layer=z_layer,
        r_tool=r_tool,
        config=config,
    )
    _, _, _, finish_steps = _boss_loop_targets_from_anchor(
        boss=boss,
        offset_d=r_tool,
        n_corner_segs=config.corner_roll_arc_segments,
        climb=climb,
        anchor_wall_idx=w_idx,
        anchor_u=u_anchor,
        tag_prefix=f"L{layer_idx}_finish_b{boss.idx}",
    )

    # Verify the finishing pass around boss does not gouge any other standing boss
    prev_chk = lead_in_pts[-1]
    for tgt_xy, _ in finish_steps:
        if (
            _min_seg_clearance_to_bosses(
                prev_chk[0], prev_chk[1], tgt_xy[0], tgt_xy[1], all_bosses, skip_idx=boss.idx
            )
            < r_tool - 1e-6
        ):
            raise ValueError(
                f"standing bosses are separated by less than cutter diameter {2.0 * r_tool} mm; "
                f"cannot finish boss {boss.idx} without gouging an adjacent boss"
            )
        prev_chk = tgt_xy

    p_plunge = lead_in_pts[0]
    waypoints.append(
        (
            C.SegmentKind.RAPID,
            (p_plunge[0], p_plunge[1], z_safe),
            None,
            f"L{layer_idx}_finish_b{boss.idx}_approach",
        )
    )
    waypoints.append(
        (
            C.SegmentKind.FEED,
            (p_plunge[0], p_plunge[1], z_layer),
            config.plunge_feed_mm_s,
            f"L{layer_idx}_finish_b{boss.idx}_plunge",
        )
    )

    for k, pt in enumerate(lead_in_pts[1:]):
        waypoints.append(
            (
                C.SegmentKind.FEED,
                (pt[0], pt[1], z_layer),
                config.finish_feed_mm_s,
                f"L{layer_idx}_finish_b{boss.idx}_lead_in_{k}",
            )
        )

    for pt, seg_tag in finish_steps:
        waypoints.append(
            (
                C.SegmentKind.FEED,
                (pt[0], pt[1], z_layer),
                config.finish_feed_mm_s,
                seg_tag,
            )
        )

    for k, pt in enumerate(lead_out_pts[1:]):
        waypoints.append(
            (
                C.SegmentKind.FEED,
                (pt[0], pt[1], z_layer),
                config.finish_feed_mm_s,
                f"L{layer_idx}_finish_b{boss.idx}_lead_out_{k}",
            )
        )

    p_retract = lead_out_pts[-1]
    waypoints.append(
        (
            C.SegmentKind.FEED,
            (p_retract[0], p_retract[1], z_safe),
            config.retract_feed_mm_s,
            f"L{layer_idx}_finish_b{boss.idx}_retract",
        )
    )


def _waypoints_to_motion_plan(
    task: C.TaskSpec,
    waypoints_work: Sequence[WaypointSpec],
    provenance: str,
) -> C.MotionPlan:
    """Convert work-frame waypoints into a validated machine-frame MotionPlan."""
    if not task.initial_state.known or task.initial_state.position_mm is None:
        raise C.ContractError("generate_contour_toolpath requires a known initial_state.position_mm")

    to_machine = task.work_to_machine.apply
    cur_mach = task.initial_state.position_mm
    max_feed = float(task.limits.max_feed_mm_s)
    rapid_speed = float(task.limits.rapid_mm_s)

    raw_segments: List[C.Segment] = []
    idx = 0
    for kind, target_work, feed_mm_s, tag in waypoints_work:
        nxt_work = (float(target_work[0]), float(target_work[1]), float(target_work[2]))
        nxt_mach = to_machine(nxt_work)
        dist = math.dist(cur_mach, nxt_mach)
        if dist <= 1e-8:
            continue
        if kind == C.SegmentKind.RAPID:
            eff_feed = None
            dur = dist / rapid_speed
        else:
            if feed_mm_s is None or feed_mm_s <= 0.0:
                raise ValueError(f"segment {tag!r} requires positive feed_mm_s")
            eff_feed = min(float(feed_mm_s), max_feed)
            dur = dist / eff_feed
        raw_segments.append(
            C.Segment(
                id=f"seg_{idx:04d}",
                kind=kind,
                start_mm=cur_mach,
                end_mm=nxt_mach,
                duration_s=dur,
                feed_mm_s=eff_feed,
                tag=tag,
            )
        )
        cur_mach = nxt_mach
        idx += 1

    rule = task.motion_semantics.timing_rule
    if rule == C.TimingRule.BLENDED_ACCEL_BOUNDED and raw_segments:
        durations = C.blended_segment_durations(
            raw_segments, rapid_speed, float(task.limits.max_accel_mm_s2)
        )
        segments = tuple(
            dataclasses.replace(seg, duration_s=dur)
            for seg, dur in zip(raw_segments, durations)
        )
    elif rule == C.TimingRule.FIXED_DT and raw_segments:
        raise C.ContractError("FIXED_DT timing_rule is unsupported by generate_contour_toolpath")
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


def generate_contour_toolpath(
    task: C.TaskSpec,
    config: ContourConfig = ContourConfig(),
) -> C.MotionPlan:
    """Generate a 2.5D multi-boss exterior & island contour MotionPlan for ``task``."""
    stock_xy, z_top, z_floor, all_bosses = _extract_contour_geometry(task)
    r_tool = float(task.tool.cutter_radius_mm)

    inv_machine = task.work_to_machine.inverse().apply
    init_work = inv_machine(task.initial_state.position_mm)
    env_z_max = inv_machine(task.limits.envelope_max_mm)[2]
    if init_work[2] >= z_top + 1.0:
        z_safe = float(init_work[2])
        need_initial_lift = False
    else:
        z_safe = min(z_top + 5.0, float(env_z_max) - 0.5)
        if z_safe <= z_top + 0.5:
            raise ValueError("machine Z envelope cannot accommodate safe retract height above stock")
        need_initial_lift = True

    total_depth = z_top - z_floor
    if config.axial_stepdown_mm is None or config.axial_stepdown_mm >= total_depth - 1e-9:
        z_layers = [z_floor]
    else:
        n_layers = max(1, int(math.ceil((total_depth - 1e-9) / float(config.axial_stepdown_mm))))
        z_layers = [z_top - total_depth * (k / n_layers) for k in range(1, n_layers + 1)]

    waypoints: List[WaypointSpec] = []
    if need_initial_lift:
        waypoints.append(
            (
                C.SegmentKind.FEED,
                (float(init_work[0]), float(init_work[1]), z_safe),
                config.retract_feed_mm_s,
                "initial_safe_retract",
            )
        )

    for layer_idx, z_layer in enumerate(z_layers):
        active_bosses = [b for b in all_bosses if b.z1 > z_layer + 1e-6]
        env_xy = _work_xy_envelope_bounds(task, z_layer, margin_mm=0.5)

        if not active_bosses:
            # Pure open slab above shorter boss tops: clear via multi-boss scanline with zero bosses
            _emit_multi_boss_roughing(
                task=task,
                bosses=(),
                stock_xy=stock_xy,
                env_xy=env_xy,
                z_layer=z_layer,
                z_safe=z_safe,
                r_tool=r_tool,
                config=config,
                layer_idx=layer_idx,
                waypoints=waypoints,
            )
            continue

        if len(active_bosses) == 1:
            _emit_single_boss_roughing(
                task=task,
                boss=active_bosses[0],
                stock_xy=stock_xy,
                env_xy=env_xy,
                z_layer=z_layer,
                z_safe=z_safe,
                r_tool=r_tool,
                config=config,
                layer_idx=layer_idx,
                waypoints=waypoints,
            )
        else:
            _emit_multi_boss_roughing(
                task=task,
                bosses=active_bosses,
                stock_xy=stock_xy,
                env_xy=env_xy,
                z_layer=z_layer,
                z_safe=z_safe,
                r_tool=r_tool,
                config=config,
                layer_idx=layer_idx,
                waypoints=waypoints,
            )

        for boss in active_bosses:
            _emit_boss_finishing_pass(
                task=task,
                boss=boss,
                all_bosses=active_bosses,
                stock_xy=stock_xy,
                z_layer=z_layer,
                z_safe=z_safe,
                r_tool=r_tool,
                config=config,
                layer_idx=layer_idx,
                waypoints=waypoints,
            )

    return _waypoints_to_motion_plan(task, waypoints, PROVENANCE_ID)


__all__ = [
    "ContourConfig",
    "generate_contour_toolpath",
]
