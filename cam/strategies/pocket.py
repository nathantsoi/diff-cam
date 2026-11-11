# SPDX-License-Identifier: MIT
"""2.5D Distance-Field Contour-Parallel Pocket Clearing strategy (MIT clean-room).

Generates collision-free, gouge-free 2.5D pocket clearing MotionPlans for rectangular
cavities containing ``0..N`` arbitrary standing rectangular islands:

1. Computes a 2D signed Euclidean Distance Field ``D(x, y)`` of the machinable cavity
   (distance to the nearest pocket outer wall or active standing island wall at each Z
   layer) via ``scipy.ndimage.distance_transform_edt``.
2. Extracts closed iso-distance contour loops from the innermost medial cores (``d_max``
   down to ``R + finish_margin``) using a topological 2D Marching Squares stitcher and
   Ramer-Douglas-Peucker (``tol = 0.04 mm``) polyline simplification.
3. Orders contour loops inside-out per 4-connected component (post-order traversal of the
   level-set component tree) so every pocket chamber enters its central core first via a
   gentle sloped contour ramp (or vertical plunge) and expands outward.
4. Follows iso-offset roughing with exact analytical cutter-offset finishing passes along
   the outer pocket boundary and every active standing island boundary (clipped against
   walls and neighboring islands) for exact dimensional accuracy.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, label, maximum_filter

from benchmarks import contracts as C

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
Box2D = Tuple[float, float, float, float]  # (x0, y0, x1, y1)
Waypoint = Tuple[C.SegmentKind, Point3D, Optional[float], str]

RDP_TOL_MM = 0.04
PROVENANCE_ID = "mit_cleanroom:pocket_strategy@2"


@dataclass(frozen=True)
class PocketConfig:
    """Configuration for 2.5D distance-field contour-parallel pocket clearing."""

    stepover_fraction: float = 0.65
    grid_pitch_mm: float = 0.10
    entry_mode: str = "ramp"
    ramp_angle_deg: float = 6.0
    axial_stepdown_mm: float | None = None
    finish_spring_pass: bool = True
    nominal_feed_mm_s: float = 5.0
    plunge_feed_mm_s: float = 3.5
    retract_feed_mm_s: float = 14.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.stepover_fraction) or not (0.05 <= self.stepover_fraction <= 0.95):
            raise ValueError(f"stepover_fraction must lie in [0.05, 0.95], got {self.stepover_fraction!r}")
        if not math.isfinite(self.grid_pitch_mm) or self.grid_pitch_mm <= 0.0:
            raise ValueError(f"grid_pitch_mm must be positive, got {self.grid_pitch_mm!r}")
        if self.entry_mode not in ("ramp", "plunge"):
            raise ValueError(f"entry_mode must be 'ramp' or 'plunge', got {self.entry_mode!r}")
        if not math.isfinite(self.ramp_angle_deg) or not (0.5 <= self.ramp_angle_deg < 89.0):
            raise ValueError(f"ramp_angle_deg must lie in [0.5, 89.0), got {self.ramp_angle_deg!r}")
        if self.axial_stepdown_mm is not None and (
            not math.isfinite(self.axial_stepdown_mm) or self.axial_stepdown_mm <= 0.0
        ):
            raise ValueError(f"axial_stepdown_mm must be positive or None, got {self.axial_stepdown_mm!r}")
        for fname in ("nominal_feed_mm_s", "plunge_feed_mm_s", "retract_feed_mm_s"):
            val = getattr(self, fname)
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{fname} must be positive, got {val!r}")


# ---------------------------------------------------------------------------
# Analytical 2D obstacle distance & interval clipping helpers
# ---------------------------------------------------------------------------
def _point_to_box_dist(p: Point2D, box: Box2D) -> float:
    dx = max(0.0, box[0] - p[0], p[0] - box[2])
    dy = max(0.0, box[1] - p[1], p[1] - box[3])
    return math.hypot(dx, dy)


def _point_to_seg_dist(p: Point2D, a: Point2D, b: Point2D) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    len2 = dx * dx + dy * dy
    if len2 <= 1e-18:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _seg_intersects_box(a: Point2D, b: Point2D, box: Box2D) -> bool:
    """Exact Liang-Barsky intersection check between segment AB and closed AABB."""
    t0, t1 = 0.0, 1.0
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    checks = (
        (-dx, a[0] - box[0]),
        (dx, box[2] - a[0]),
        (-dy, a[1] - box[1]),
        (dy, box[3] - a[1]),
    )
    for p_val, q_val in checks:
        if abs(p_val) <= 1e-15:
            if q_val < 0.0:
                return False
        else:
            r = q_val / p_val
            if p_val < 0.0:
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


def _seg_to_box_dist(a: Point2D, b: Point2D, box: Box2D) -> float:
    if _seg_intersects_box(a, b, box):
        return 0.0
    corners = (
        (box[0], box[1]),
        (box[2], box[1]),
        (box[2], box[3]),
        (box[0], box[3]),
    )
    return min(
        _point_to_box_dist(a, box),
        _point_to_box_dist(b, box),
        *(_point_to_seg_dist(c, a, b) for c in corners),
    )


def _segment_wall_and_island_clearance(
    a: Point2D,
    b: Point2D,
    cavity_box: Box2D,
    islands: Sequence[Box2D],
) -> float:
    """Exact minimum Euclidean clearance from segment AB to cavity outer walls and islands."""
    cx0, cy0, cx1, cy1 = cavity_box
    d_wall = min(
        a[0] - cx0, b[0] - cx0,
        cx1 - a[0], cx1 - b[0],
        a[1] - cy0, b[1] - cy0,
        cy1 - a[1], cy1 - b[1],
    )
    if d_wall <= 0.0:
        return d_wall
    d_min = d_wall
    for isl in islands:
        d_isl = _seg_to_box_dist(a, b, isl)
        if d_isl < d_min:
            d_min = d_isl
    return d_min


def _subtract_open_interval(
    intervals: List[Tuple[float, float]],
    cut_lo: float,
    cut_hi: float,
    min_len: float = 1e-5,
) -> List[Tuple[float, float]]:
    if cut_hi <= cut_lo:
        return intervals
    out: List[Tuple[float, float]] = []
    for u0, u1 in intervals:
        if cut_hi <= u0 or cut_lo >= u1:
            out.append((u0, u1))
            continue
        if cut_lo - u0 > min_len:
            out.append((u0, cut_lo))
        if u1 - cut_hi > min_len:
            out.append((cut_hi, u1))
    return out


def _clip_axis_segment_to_free_space(
    a: Point2D,
    b: Point2D,
    outer_offset_box: Box2D,
    islands: Sequence[Box2D],
    cutter_radius: float,
) -> List[Tuple[Point2D, Point2D]]:
    """Analytically clip a horizontal or vertical segment against outer walls and islands."""
    ox0, oy0, ox1, oy1 = outer_offset_box
    r_strict = cutter_radius - 1e-7

    if abs(a[1] - b[1]) <= 1e-9:
        y = 0.5 * (a[1] + b[1])
        if y < oy0 - 1e-7 or y > oy1 + 1e-7:
            return []
        y = min(max(y, oy0), oy1)
        lo = max(min(a[0], b[0]), ox0)
        hi = min(max(a[0], b[0]), ox1)
        if hi - lo <= 1e-5:
            return []
        spans = [(lo, hi)]
        for ix0, iy0, ix1, iy1 in islands:
            dy = max(0.0, iy0 - y, y - iy1)
            if dy < r_strict:
                dx = math.sqrt(max(0.0, r_strict * r_strict - dy * dy))
                spans = _subtract_open_interval(spans, ix0 - dx, ix1 + dx)
                if not spans:
                    return []
        if a[0] <= b[0]:
            return [((u0, y), (u1, y)) for u0, u1 in spans]
        return [((u1, y), (u0, y)) for u0, u1 in reversed(spans)]

    if abs(a[0] - b[0]) <= 1e-9:
        x = 0.5 * (a[0] + b[0])
        if x < ox0 - 1e-7 or x > ox1 + 1e-7:
            return []
        x = min(max(x, ox0), ox1)
        lo = max(min(a[1], b[1]), oy0)
        hi = min(max(a[1], b[1]), oy1)
        if hi - lo <= 1e-5:
            return []
        spans = [(lo, hi)]
        for ix0, iy0, ix1, iy1 in islands:
            dx = max(0.0, ix0 - x, x - ix1)
            if dx < r_strict:
                dy = math.sqrt(max(0.0, r_strict * r_strict - dx * dx))
                spans = _subtract_open_interval(spans, iy0 - dy, iy1 + dy)
                if not spans:
                    return []
        if a[1] <= b[1]:
            return [((x, u0), (x, u1)) for u0, u1 in spans]
        return [((x, u1), (x, u0)) for u0, u1 in reversed(spans)]

    raise ValueError("Expected axis-aligned analytical boundary segment")


def _stitch_directed_segments(segments: Sequence[Tuple[Point2D, Point2D]]) -> List[List[Point2D]]:
    """Stitch contiguous directed segments into closed loops or maximal open chains."""
    if not segments:
        return []
    chains: List[List[Point2D]] = []
    for s_pt, e_pt in segments:
        if chains and math.hypot(chains[-1][-1][0] - s_pt[0], chains[-1][-1][1] - s_pt[1]) <= 1e-5:
            chains[-1].append(e_pt)
        else:
            chains.append([s_pt, e_pt])
    if (
        len(chains) > 1
        and math.hypot(chains[-1][-1][0] - chains[0][0][0], chains[-1][-1][1] - chains[0][0][1]) <= 1e-5
    ):
        chains[0] = chains[-1][:-1] + chains[0]
        chains.pop()
    return chains


# ---------------------------------------------------------------------------
# 2D Marching Squares (4-connected foreground) + Ramer-Douglas-Peucker
# ---------------------------------------------------------------------------
EdgeId = Tuple[int, int, int]  # (axis: 0=horiz/1=vert, i, j)


def _edge_point(
    edge: EdgeId,
    field: np.ndarray,
    level: float,
    x_origin: float,
    y_origin: float,
    pitch: float,
) -> Tuple[Point2D, Tuple[int, int]]:
    """Linearly interpolate iso-crossing on a grid edge and return (xy, inside_pixel_ij)."""
    axis, i, j = edge
    if axis == 0:
        v0 = float(field[i, j])
        v1 = float(field[i + 1, j])
        dv = v1 - v0
        t = 0.5 if abs(dv) <= 1e-15 else max(0.0, min(1.0, (level - v0) / dv))
        pt = (x_origin + (i + 0.5 + t) * pitch, y_origin + (j + 0.5) * pitch)
        inside_ij = (i, j) if v0 >= level else (i + 1, j)
        return pt, inside_ij
    v0 = float(field[i, j])
    v1 = float(field[i, j + 1])
    dv = v1 - v0
    t = 0.5 if abs(dv) <= 1e-15 else max(0.0, min(1.0, (level - v0) / dv))
    pt = (x_origin + (i + 0.5) * pitch, y_origin + (j + 0.5 + t) * pitch)
    inside_ij = (i, j) if v0 >= level else (i, j + 1)
    return pt, inside_ij


# Exact 16-case lookup for 4-connected foreground (inside = field >= level on LEFT of directed edge).
# Edges of cell (i, j): 0=bottom (0,i,j), 1=right (1,i+1,j), 2=top (0,i,j+1), 3=left (1,i,j).
_MS_TABLE: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    (),                 # 0: 0000
    ((0, 3),),          # 1: 0001 (BL)
    ((1, 0),),          # 2: 0010 (BR)
    ((1, 3),),          # 3: 0011 (BL+BR)
    ((2, 1),),          # 4: 0100 (TR)
    ((0, 3), (2, 1)),   # 5: 0101 (BL, TR separate under 4-connectivity)
    ((2, 0),),          # 6: 0110 (BR+TR)
    ((2, 3),),          # 7: 0111 (~TL)
    ((3, 2),),          # 8: 1000 (TL)
    ((0, 2),),          # 9: 1001 (BL+TL)
    ((1, 0), (3, 2)),   # 10: 1010 (BR, TL separate under 4-connectivity)
    ((1, 2),),          # 11: 1011 (~TR)
    ((3, 1),),          # 12: 1100 (TL+TR)
    ((0, 1),),          # 13: 1101 (~BR)
    ((3, 0),),          # 14: 1110 (~BL)
    (),                 # 15: 1111
)


def _rdp_open(points: Sequence[Point2D], tol: float) -> List[Point2D]:
    """Iterative Ramer-Douglas-Peucker polyline simplification on an open chain."""
    n = len(points)
    if n <= 2:
        return list(points)
    keep = [False] * n
    keep[0] = True
    keep[-1] = True
    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 <= 1:
            continue
        a = points[i0]
        b = points[i1]
        max_d = -1.0
        best_k = -1
        for k in range(i0 + 1, i1):
            d = _point_to_seg_dist(points[k], a, b)
            if d > max_d:
                max_d = d
                best_k = k
        if max_d > tol and best_k != -1:
            keep[best_k] = True
            stack.append((i0, best_k))
            stack.append((best_k, i1))
    return [pt for pt, flag in zip(points, keep) if flag]


def _rdp_closed_loop(ring: Sequence[Point2D], tol: float = RDP_TOL_MM) -> List[Point2D]:
    """Simplify a closed ring (without duplicate endpoint) into a closed loop [p0, ..., p0]."""
    m = len(ring)
    if m < 3:
        return []
    idx_a = min(range(m), key=lambda k: (ring[k][0], ring[k][1]))
    pa = ring[idx_a]
    idx_b = max(range(m), key=lambda k: math.hypot(ring[k][0] - pa[0], ring[k][1] - pa[1]))
    if idx_a == idx_b or math.hypot(ring[idx_b][0] - pa[0], ring[idx_b][1] - pa[1]) < 0.05:
        return []

    if idx_a < idx_b:
        chain1 = list(ring[idx_a : idx_b + 1])
        chain2 = list(ring[idx_b:]) + list(ring[: idx_a + 1])
    else:
        chain1 = list(ring[idx_a:]) + list(ring[: idx_b + 1])
        chain2 = list(ring[idx_b : idx_a + 1])

    s1 = _rdp_open(chain1, tol)
    s2 = _rdp_open(chain2, tol)
    merged = s1[:-1] + s2[:-1]
    cleaned: List[Point2D] = []
    for pt in merged:
        if not cleaned or math.hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) > 2e-3:
            cleaned.append(pt)
    if len(cleaned) >= 2 and math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) <= 2e-3:
        cleaned.pop()
    if len(cleaned) < 3:
        return []
    cleaned.append(cleaned[0])
    return cleaned


def _signed_area(closed_loop: Sequence[Point2D]) -> float:
    area2 = 0.0
    for i in range(len(closed_loop) - 1):
        x0, y0 = closed_loop[i]
        x1, y1 = closed_loop[i + 1]
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def _polyline_length(pts: Sequence[Point2D]) -> float:
    return sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    )


def _extract_iso_loops_by_component(
    field: np.ndarray,
    inside_mask: np.ndarray,
    comp_labels: np.ndarray,
    level: float,
    x_origin: float,
    y_origin: float,
    pitch: float,
    rdp_tol: float = RDP_TOL_MM,
) -> Dict[int, List[List[Point2D]]]:
    """Extract closed, RDP-simplified Marching Squares loops grouped by 4-connected component ID."""
    b0 = inside_mask[:-1, :-1].astype(np.uint8)
    b1 = inside_mask[1:, :-1].astype(np.uint8)
    b2 = inside_mask[1:, 1:].astype(np.uint8)
    b3 = inside_mask[:-1, 1:].astype(np.uint8)
    masks = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)

    active_i, active_j = np.nonzero((masks != 0) & (masks != 15))
    next_edge: Dict[EdgeId, EdgeId] = {}
    for ii, jj in zip(active_i.tolist(), active_j.tolist()):
        m = int(masks[ii, jj])
        cell_edges: Tuple[EdgeId, EdgeId, EdgeId, EdgeId] = (
            (0, ii, jj),
            (1, ii + 1, jj),
            (0, ii, jj + 1),
            (1, ii, jj),
        )
        for src_local, dst_local in _MS_TABLE[m]:
            next_edge[cell_edges[src_local]] = cell_edges[dst_local]

    visited: set[EdgeId] = set()
    by_comp: Dict[int, List[List[Point2D]]] = {}
    for start_edge in sorted(next_edge.keys()):
        if start_edge in visited:
            continue
        ring_pts: List[Point2D] = []
        comp_id = 0
        cur = start_edge
        while cur not in visited:
            visited.add(cur)
            pt, inside_ij = _edge_point(cur, field, level, x_origin, y_origin, pitch)
            ring_pts.append(pt)
            if comp_id == 0:
                comp_id = int(comp_labels[inside_ij[0], inside_ij[1]])
            cur = next_edge[cur]
        loop = _rdp_closed_loop(ring_pts, tol=rdp_tol)
        if loop and _polyline_length(loop) >= 0.20 and comp_id > 0:
            by_comp.setdefault(comp_id, []).append(loop)

    for cid, loops in by_comp.items():
        loops.sort(key=lambda lp: (-_signed_area(lp), lp[0][0], lp[0][1]))
    return by_comp


# ---------------------------------------------------------------------------
# Loop rotation, entry ramping, and linking helpers
# ---------------------------------------------------------------------------
def _rotate_closed_loop_nearest(closed_loop: Sequence[Point2D], ref: Point2D) -> List[Point2D]:
    """Split/rotate a closed loop [v0, ..., v_{m-1}, v0] to start at the point closest to ref."""
    ring = list(closed_loop[:-1])
    m = len(ring)
    if m == 0:
        return list(closed_loop)

    best_dist = math.inf
    best_i = 0
    best_t = 0.0
    best_q = ring[0]
    for i in range(m):
        a = ring[i]
        b = ring[(i + 1) % m]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        len2 = dx * dx + dy * dy
        t = 0.0 if len2 <= 1e-18 else max(0.0, min(1.0, ((ref[0] - a[0]) * dx + (ref[1] - a[1]) * dy) / len2))
        q = (a[0] + t * dx, a[1] + t * dy)
        d = math.hypot(q[0] - ref[0], q[1] - ref[1])
        if d < best_dist:
            best_dist = d
            best_i = i
            best_t = t
            best_q = q

    if best_t <= 1e-4 or math.hypot(best_q[0] - ring[best_i][0], best_q[1] - ring[best_i][1]) <= 2e-3:
        ordered = ring[best_i:] + ring[:best_i]
        return ordered + [ordered[0]]
    nxt_i = (best_i + 1) % m
    if best_t >= 1.0 - 1e-4 or math.hypot(best_q[0] - ring[nxt_i][0], best_q[1] - ring[nxt_i][1]) <= 2e-3:
        ordered = ring[nxt_i:] + ring[:nxt_i]
        return ordered + [ordered[0]]
    tail = ring[nxt_i:] + ring[: nxt_i]
    return [best_q] + tail + [best_q]


def _orient_chain_nearest(chain: Sequence[Point2D], ref: Point2D) -> List[Point2D]:
    """Orient an open chain or rotate a closed loop to start closest to ref."""
    if len(chain) >= 3 and math.hypot(chain[0][0] - chain[-1][0], chain[0][1] - chain[-1][1]) <= 1e-5:
        return _rotate_closed_loop_nearest(chain, ref)
    d_start = math.hypot(chain[0][0] - ref[0], chain[0][1] - ref[1])
    d_end = math.hypot(chain[-1][0] - ref[0], chain[-1][1] - ref[1])
    return list(chain) if d_start <= d_end else list(reversed(chain))


def _emit_entry_and_traverse(
    waypoints: List[Waypoint],
    chain: Sequence[Point2D],
    z_safe: float,
    z_stock_top: float,
    z_cut: float,
    config: PocketConfig,
    force_plunge: bool = False,
) -> Point2D:
    """Approach at z_safe, enter to z_cut (ramped or plunged), and cut one full pass of chain."""
    start_xy = chain[0]
    waypoints.append((C.SegmentKind.RAPID, (start_xy[0], start_xy[1], z_safe), None, "approach"))

    is_closed = len(chain) >= 3 and math.hypot(chain[0][0] - chain[-1][0], chain[0][1] - chain[-1][1]) <= 1e-5
    total_len = _polyline_length(chain)

    if config.entry_mode == "plunge" or force_plunge or total_len <= 1e-3:
        waypoints.append((C.SegmentKind.FEED, (start_xy[0], start_xy[1], z_cut), config.plunge_feed_mm_s, "plunge"))
        for pt in chain[1:]:
            waypoints.append((C.SegmentKind.FEED, (pt[0], pt[1], z_cut), config.nominal_feed_mm_s, "cut"))
        return chain[-1]

    # Sloped contour/shuttle ramp from just above uncut stock top down to z_cut
    z_ramp_top = min(z_safe, z_stock_top + 0.15)
    if z_safe > z_ramp_top + 1e-6:
        waypoints.append(
            (C.SegmentKind.FEED, (start_xy[0], start_xy[1], z_ramp_top), config.retract_feed_mm_s, "approach_z")
        )

    dz_total = z_ramp_top - z_cut
    if dz_total <= 1e-6:
        for pt in chain[1:]:
            waypoints.append((C.SegmentKind.FEED, (pt[0], pt[1], z_cut), config.nominal_feed_mm_s, "cut"))
        return chain[-1]

    slope = math.tan(math.radians(config.ramp_angle_deg))
    ramp_xy_needed = dz_total / slope
    s_accum = 0.0

    if is_closed:
        ring = list(chain[:-1])
        m = len(ring)
        edge_idx = 0
        while True:
            a = ring[edge_idx % m]
            b = ring[(edge_idx + 1) % m]
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg_len <= 1e-9:
                edge_idx += 1
                continue
            if s_accum + seg_len < ramp_xy_needed - 1e-6:
                s_accum += seg_len
                z_now = z_ramp_top - s_accum * slope
                waypoints.append((C.SegmentKind.FEED, (b[0], b[1], z_now), config.plunge_feed_mm_s, "ramp"))
                edge_idx += 1
            else:
                frac = max(0.0, min(1.0, (ramp_xy_needed - s_accum) / seg_len))
                land_xy = (a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1]))
                waypoints.append(
                    (C.SegmentKind.FEED, (land_xy[0], land_xy[1], z_cut), config.plunge_feed_mm_s, "ramp")
                )
                # Complete one full flat 360-degree sweep around the ring from land_xy back to land_xy
                if math.hypot(b[0] - land_xy[0], b[1] - land_xy[1]) > 2e-3:
                    waypoints.append((C.SegmentKind.FEED, (b[0], b[1], z_cut), config.nominal_feed_mm_s, "cut"))
                cur_v = (edge_idx + 1) % m
                for _ in range(m - 1):
                    nxt_v = (cur_v + 1) % m
                    p_nxt = ring[nxt_v]
                    waypoints.append((C.SegmentKind.FEED, (p_nxt[0], p_nxt[1], z_cut), config.nominal_feed_mm_s, "cut"))
                    cur_v = nxt_v
                if math.hypot(land_xy[0] - a[0], land_xy[1] - a[1]) > 2e-3:
                    waypoints.append(
                        (C.SegmentKind.FEED, (land_xy[0], land_xy[1], z_cut), config.nominal_feed_mm_s, "cut")
                    )
                return land_xy

    # Open chain shuttle ramp: walk back and forth along chain until z_cut is reached, then sweep chain
    fwd_pts = list(chain)
    rev_pts = list(reversed(chain))
    direction_fwd = True
    while True:
        seq = fwd_pts if direction_fwd else rev_pts
        for k in range(len(seq) - 1):
            a = seq[k]
            b = seq[k + 1]
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg_len <= 1e-9:
                continue
            if s_accum + seg_len < ramp_xy_needed - 1e-6:
                s_accum += seg_len
                z_now = z_ramp_top - s_accum * slope
                waypoints.append((C.SegmentKind.FEED, (b[0], b[1], z_now), config.plunge_feed_mm_s, "ramp"))
            else:
                frac = max(0.0, min(1.0, (ramp_xy_needed - s_accum) / seg_len))
                land_xy = (a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1]))
                waypoints.append(
                    (C.SegmentKind.FEED, (land_xy[0], land_xy[1], z_cut), config.plunge_feed_mm_s, "ramp")
                )
                # Finish walking to the end of current direction, then sweep the entire chain cleanly
                for rem in seq[k + 1 :]:
                    waypoints.append((C.SegmentKind.FEED, (rem[0], rem[1], z_cut), config.nominal_feed_mm_s, "cut"))
                full_sweep = rev_pts if direction_fwd else fwd_pts
                for pt in full_sweep[1:]:
                    waypoints.append((C.SegmentKind.FEED, (pt[0], pt[1], z_cut), config.nominal_feed_mm_s, "cut"))
                return full_sweep[-1]
        direction_fwd = not direction_fwd


# ---------------------------------------------------------------------------
# 2D rasterized swept-coverage tracking (for smart plunging & zero-residual)
# ---------------------------------------------------------------------------
def _rasterize_segment_on_grid(
    mask: np.ndarray,
    a: Point2D,
    b: Point2D,
    x_origin: float,
    y_origin: float,
    pitch: float,
) -> None:
    nx, ny = mask.shape
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    n_steps = max(1, int(math.ceil(dist / (0.5 * pitch))))
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    xs = a[0] + ts * (b[0] - a[0])
    ys = a[1] + ts * (b[1] - a[1])
    ii = np.clip(np.floor((xs - x_origin) / pitch).astype(np.intp), 0, nx - 1)
    jj = np.clip(np.floor((ys - y_origin) / pitch).astype(np.intp), 0, ny - 1)
    mask[ii, jj] = True


def _record_chain_on_grid(
    mask: np.ndarray,
    chain: Sequence[Point2D],
    x_origin: float,
    y_origin: float,
    pitch: float,
) -> None:
    for k in range(len(chain) - 1):
        _rasterize_segment_on_grid(mask, chain[k], chain[k + 1], x_origin, y_origin, pitch)


def _is_point_cleared_on_grid(
    swept_centers: np.ndarray,
    pt: Point2D,
    radius_check: float,
    x_origin: float,
    y_origin: float,
    pitch: float,
) -> bool:
    if not np.any(swept_centers):
        return False
    nx, ny = swept_centers.shape
    ci = int(math.floor((pt[0] - x_origin) / pitch))
    cj = int(math.floor((pt[1] - y_origin) / pitch))
    rad_cells = int(math.ceil(radius_check / pitch)) + 1
    i0, i1 = max(0, ci - rad_cells), min(nx, ci + rad_cells + 1)
    j0, j1 = max(0, cj - rad_cells), min(ny, cj + rad_cells + 1)
    sub = swept_centers[i0:i1, j0:j1]
    if not np.any(sub):
        return False
    si, sj = np.nonzero(sub)
    xs = x_origin + (i0 + si + 0.5) * pitch
    ys = y_origin + (j0 + sj + 0.5) * pitch
    return bool(np.any((xs - pt[0]) ** 2 + (ys - pt[1]) ** 2 <= radius_check * radius_check))


# ---------------------------------------------------------------------------
# Task geometry extraction & MotionPlan assembly
# ---------------------------------------------------------------------------
def _extract_pocket_geometry(
    task: C.TaskSpec,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]:
    part = task.original_part
    if part.kind != "analytic":
        raise ValueError(f"pocket strategy requires analytic original_part, got {part.kind!r}")
    p = part.params
    prim = p.get("primitive")
    if prim == "box_minus_multi_pocket":
        bmin = tuple(float(v) for v in p["box_min_mm"])
        bmax = tuple(float(v) for v in p["box_max_mm"])
        pmin = tuple(float(v) for v in p["pocket_min_mm"])
        pmax = tuple(float(v) for v in p["pocket_max_mm"])
        islands = [
            (tuple(float(v) for v in isl[0]), tuple(float(v) for v in isl[1]))
            for isl in p.get("islands_mm", ())
        ]
        return bmin, bmax, pmin, pmax, islands  # type: ignore[return-value]
    if prim == "box_minus_pocket":
        bmin = tuple(float(v) for v in p["box_min_mm"])
        bmax = tuple(float(v) for v in p["box_max_mm"])
        pmin = tuple(float(v) for v in p["pocket_min_mm"])
        pmax = tuple(float(v) for v in p["pocket_max_mm"])
        islands = []
        if "island_min_mm" in p and "island_max_mm" in p:
            islands.append(
                (tuple(float(v) for v in p["island_min_mm"]), tuple(float(v) for v in p["island_max_mm"]))
            )
        return bmin, bmax, pmin, pmax, islands  # type: ignore[return-value]
    raise ValueError(f"Unsupported analytic primitive for pocket clearing: {prim!r}")


def _waypoints_to_motion_plan(
    task: C.TaskSpec,
    waypoints_work: Sequence[Waypoint],
    provenance: str,
) -> C.MotionPlan:
    """Convert work-frame waypoints into a contract-validated machine-frame MotionPlan."""
    to_machine = task.work_to_machine.apply
    cur_mach = task.initial_state.position_mm
    if cur_mach is None:
        raise ValueError("TaskSpec.initial_state.position_mm must be known")

    rule = task.motion_semantics.timing_rule
    dt_s = 0.01 if rule == C.TimingRule.FIXED_DT else None

    raw_segments: List[C.Segment] = []
    idx = 0
    for kind, target_work, feed_mm_s, tag in waypoints_work:
        nxt_work = (float(target_work[0]), float(target_work[1]), float(target_work[2]))
        nxt_mach = to_machine(nxt_work)
        dist = math.dist(cur_mach, nxt_mach)
        # Filter sub-micron duplicates so 4-decimal G-code posting never collapses distinct points
        if dist <= 1e-3:
            continue
        if kind == C.SegmentKind.RAPID:
            eff_feed = None
            dur = dist / float(task.limits.rapid_mm_s)
        else:
            if feed_mm_s is None or feed_mm_s <= 0.0:
                raise ValueError("FEED segment requires positive feed_mm_s")
            eff_feed = min(float(feed_mm_s), float(task.limits.max_feed_mm_s))
            dur = dist / eff_feed
        if rule == C.TimingRule.FIXED_DT and dt_s is not None:
            dur = max(1, round(dur / dt_s)) * dt_s
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
        dt_s=dt_s,
        provenance=provenance,
    )
    plan.validate_for_task(task)
    return plan


# ---------------------------------------------------------------------------
# Single Z-layer clearing planner
# ---------------------------------------------------------------------------
def _plan_single_z_layer(
    waypoints: List[Waypoint],
    z_cut: float,
    z_stock_top: float,
    z_safe: float,
    cavity_box: Box2D,
    outer_offset_box: Box2D,
    active_islands: Sequence[Box2D],
    cutter_radius: float,
    config: PocketConfig,
) -> None:
    eff_px0, eff_py0, eff_px1, eff_py1 = cavity_box
    ox0, oy0, ox1, oy1 = outer_offset_box
    h = float(config.grid_pitch_mm)

    # 1. Construct 2D cavity grid (padded by 2 obstacle cells on every border) & signed EDT
    pad = 2
    nx = int(math.ceil((eff_px1 - eff_px0) / h - 1e-9)) + 2 * pad
    ny = int(math.ceil((eff_py1 - eff_py0) / h - 1e-9)) + 2 * pad
    x_origin = eff_px0 - pad * h
    y_origin = eff_py0 - pad * h
    xs = x_origin + (np.arange(nx, dtype=np.float64) + 0.5) * h
    ys = y_origin + (np.arange(ny, dtype=np.float64) + 0.5) * h

    cavity_mask = (
        (xs[:, None] > eff_px0)
        & (xs[:, None] < eff_px1)
        & (ys[None, :] > eff_py0)
        & (ys[None, :] < eff_py1)
    )
    for ix0, iy0, ix1, iy1 in active_islands:
        cavity_mask &= ~(
            (xs[:, None] >= ix0)
            & (xs[:, None] <= ix1)
            & (ys[None, :] >= iy0)
            & (ys[None, :] <= iy1)
        )

    edt_in = distance_transform_edt(cavity_mask, sampling=h)
    edt_out = distance_transform_edt(~cavity_mask, sampling=h)
    dist_field = np.where(cavity_mask, edt_in - 0.5 * h, -(edt_out - 0.5 * h))

    swept_centers = np.zeros((nx, ny), dtype=bool)
    cur_xy: Optional[Point2D] = None  # None iff tool is currently retracted at z_safe

    def _link_or_enter_chain(chain_in: Sequence[Point2D]) -> None:
        nonlocal cur_xy
        if cur_xy is None:
            oriented = list(chain_in)
            already_cleared = _is_point_cleared_on_grid(
                swept_centers, oriented[0], 0.35 * cutter_radius, x_origin, y_origin, h
            )
            cur_xy = _emit_entry_and_traverse(
                waypoints,
                oriented,
                z_safe=z_safe,
                z_stock_top=z_stock_top,
                z_cut=z_cut,
                config=config,
                force_plunge=already_cleared,
            )
            _record_chain_on_grid(swept_centers, oriented, x_origin, y_origin, h)
            return

        oriented = _orient_chain_nearest(chain_in, cur_xy)
        start_xy = oriented[0]
        link_clear = _segment_wall_and_island_clearance(cur_xy, start_xy, cavity_box, active_islands)
        if link_clear >= cutter_radius - 1e-6:
            if math.hypot(start_xy[0] - cur_xy[0], start_xy[1] - cur_xy[1]) > 1e-3:
                waypoints.append(
                    (C.SegmentKind.FEED, (start_xy[0], start_xy[1], z_cut), config.nominal_feed_mm_s, "link")
                )
                _rasterize_segment_on_grid(swept_centers, cur_xy, start_xy, x_origin, y_origin, h)
            for pt in oriented[1:]:
                waypoints.append((C.SegmentKind.FEED, (pt[0], pt[1], z_cut), config.nominal_feed_mm_s, "cut"))
            _record_chain_on_grid(swept_centers, oriented, x_origin, y_origin, h)
            cur_xy = oriented[-1]
            return

        # Obstacle blocks direct XY link -> retract to z_safe, rapid across, and re-enter cleanly
        waypoints.append(
            (C.SegmentKind.FEED, (cur_xy[0], cur_xy[1], z_safe), config.retract_feed_mm_s, "hop_retract")
        )
        already_cleared = _is_point_cleared_on_grid(
            swept_centers, start_xy, 0.35 * cutter_radius, x_origin, y_origin, h
        )
        cur_xy = _emit_entry_and_traverse(
            waypoints,
            oriented,
            z_safe=z_safe,
            z_stock_top=z_stock_top,
            z_cut=z_cut,
            config=config,
            force_plunge=already_cleared,
        )
        _record_chain_on_grid(swept_centers, oriented, x_origin, y_origin, h)

    # 2. Compute iso-distance contour levels from innermost medial core down to R + finish_margin
    if config.finish_spring_pass:
        finish_margin = max(0.20, 2.0 * h)
    else:
        finish_margin = max(0.075, 0.75 * h)
    d_min_rough = cutter_radius + finish_margin
    d_max = float(dist_field.max())
    # Cap effective diagonal-safe stepover at min(configured, 1.60*R) so 90-deg corners never leave cusps
    radial_step = min(float(config.stepover_fraction) * (2.0 * cutter_radius), 1.60 * cutter_radius)

    levels: List[float] = []
    if d_max >= d_min_rough:
        d_top = max(d_min_rough, d_max - 0.60 * cutter_radius)
        if d_top <= d_min_rough + 1e-6:
            raw_levels = [d_min_rough]
        else:
            n_steps = max(1, int(math.ceil((d_top - d_min_rough) / radial_step)))
            raw_levels = [d_top - i * (d_top - d_min_rough) / n_steps for i in range(n_steps + 1)]

        # Also catch secondary chamber local maxima that might sit below d_top between two levels
        win = max(3, int(round(cutter_radius / h)))
        local_peaks = (maximum_filter(dist_field, size=win) == dist_field) & (
            dist_field > d_min_rough + 0.75 * cutter_radius
        )
        if np.any(local_peaks):
            peak_vals = sorted({round(float(v), 3) for v in dist_field[local_peaks]}, reverse=True)
            for pk in peak_vals:
                lower_lvls = [lv for lv in raw_levels if lv <= pk]
                nearest_lower = max(lower_lvls) if lower_lvls else d_min_rough
                if pk - nearest_lower > 0.85 * cutter_radius:
                    raw_levels.append(max(d_min_rough, pk - 0.55 * cutter_radius))

        raw_levels.sort(reverse=True)
        for lv in raw_levels:
            if not levels or levels[-1] - lv >= 0.15 * cutter_radius:
                levels.append(lv)
        if abs(levels[-1] - d_min_rough) > 1e-6:
            if levels[-1] - d_min_rough < 0.12 * cutter_radius and len(levels) > 1:
                levels[-1] = d_min_rough
            else:
                levels.append(d_min_rough)

    # 3. Extract Marching Squares contours at every level and build the 4-connected component tree
    if levels:
        num_lvls = len(levels)
        lvl_labels: List[np.ndarray] = []
        lvl_counts: List[int] = []
        lvl_rep_ij: List[Dict[int, Tuple[int, int]]] = []
        lvl_centroids: List[Dict[int, Point2D]] = []
        lvl_loops: List[Dict[int, List[List[Point2D]]]] = []

        for l_idx, lv in enumerate(levels):
            ins = dist_field >= lv
            lbl, n_c = label(ins)
            lvl_labels.append(lbl)
            lvl_counts.append(int(n_c))
            reps: Dict[int, Tuple[int, int]] = {}
            cents: Dict[int, Point2D] = {}
            comp_peaks: Dict[int, float] = {}
            for cid in range(1, int(n_c) + 1):
                mask_c = lbl == cid
                ii, jj = np.nonzero(mask_c)
                reps[cid] = (int(ii[0]), int(jj[0]))
                cents[cid] = (
                    float(x_origin + (ii.mean() + 0.5) * h),
                    float(y_origin + (jj.mean() + 0.5) * h),
                )
                comp_peaks[cid] = float(dist_field[mask_c].max())
            lvl_rep_ij.append(reps)
            lvl_centroids.append(cents)
            extracted = _extract_iso_loops_by_component(dist_field, ins, lbl, lv, x_origin, y_origin, h)
            # Suppress accidental peak-grazing micro-loops when the next level down still covers the peak
            if l_idx + 1 < num_lvls:
                nxt_lv = levels[l_idx + 1]
                for cid in list(extracted.keys()):
                    pk = comp_peaks[cid]
                    if pk - lv < 0.25 * cutter_radius and pk - nxt_lv <= 0.90 * cutter_radius:
                        extracted.pop(cid, None)
            lvl_loops.append(extracted)

        children_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for l_idx in range(num_lvls - 1):
            parent_lbl = lvl_labels[l_idx + 1]
            for cid, rep_ij in lvl_rep_ij[l_idx].items():
                pid = int(parent_lbl[rep_ij[0], rep_ij[1]])
                if pid > 0:
                    children_map.setdefault((l_idx + 1, pid), []).append((l_idx, cid))

        for key in children_map:
            children_map[key].sort(key=lambda node: lvl_centroids[node[0]][node[1]])

        def _traverse_component_inside_out(l_idx: int, cid: int) -> bool:
            """Post-order inside-out clearing of component (l_idx, cid); returns True if any loop was cut."""
            nonlocal cur_xy
            kids = children_map.get((l_idx, cid), [])
            emitted_any_child = False
            for k_node in kids:
                if emitted_any_child and cur_xy is not None:
                    # Separate core chamber before saddle merge: retract to z_safe and ramp into new chamber core
                    waypoints.append(
                        (C.SegmentKind.FEED, (cur_xy[0], cur_xy[1], z_safe), config.retract_feed_mm_s, "chamber_retract")
                    )
                    cur_xy = None
                if _traverse_component_inside_out(k_node[0], k_node[1]):
                    emitted_any_child = True

            comp_loops = list(lvl_loops[l_idx].get(cid, []))
            if not comp_loops:
                return emitted_any_child

            # Order multiple loops of the same component (e.g., outer loop + island hole loop) greedily
            remaining_loops = comp_loops
            while remaining_loops:
                if cur_xy is None:
                    chosen = remaining_loops.pop(0)
                else:
                    best_k = min(
                        range(len(remaining_loops)),
                        key=lambda k: math.hypot(
                            _rotate_closed_loop_nearest(remaining_loops[k], cur_xy)[0][0] - cur_xy[0],
                            _rotate_closed_loop_nearest(remaining_loops[k], cur_xy)[0][1] - cur_xy[1],
                        ),
                    )
                    chosen = remaining_loops.pop(best_k)
                _link_or_enter_chain(chosen)
            return True

        root_lvl = num_lvls - 1
        root_cids = sorted(
            range(1, lvl_counts[root_lvl] + 1),
            key=lambda c: lvl_centroids[root_lvl][c],
        )
        prev_root_cut = False
        for r_cid in root_cids:
            if prev_root_cut and cur_xy is not None:
                waypoints.append(
                    (C.SegmentKind.FEED, (cur_xy[0], cur_xy[1], z_safe), config.retract_feed_mm_s, "region_retract")
                )
                cur_xy = None
            if _traverse_component_inside_out(root_lvl, r_cid):
                prev_root_cut = True

    # 4. Exact analytical cutter-offset boundary finishing passes (outer wall + every active island)
    if config.finish_spring_pass and ox1 >= ox0 - 1e-9 and oy1 >= oy0 - 1e-9:
        finish_chains: List[List[Point2D]] = []

        # 4a. Outer pocket wall rectangle [ox0, ox1] x [oy0, oy1] (CCW)
        outer_raw_edges: List[Tuple[Point2D, Point2D]] = []
        if ox1 - ox0 > 1e-5 and oy1 - oy0 > 1e-5:
            outer_raw_edges = [
                ((ox0, oy0), (ox1, oy0)),
                ((ox1, oy0), (ox1, oy1)),
                ((ox1, oy1), (ox0, oy1)),
                ((ox0, oy1), (ox0, oy0)),
            ]
        elif ox1 - ox0 > 1e-5:
            y_mid = 0.5 * (oy0 + oy1)
            outer_raw_edges = [((ox0, y_mid), (ox1, y_mid))]
        elif oy1 - oy0 > 1e-5:
            x_mid = 0.5 * (ox0 + ox1)
            outer_raw_edges = [((x_mid, oy0), (x_mid, oy1))]

        outer_clipped: List[Tuple[Point2D, Point2D]] = []
        for seg_a, seg_b in outer_raw_edges:
            outer_clipped.extend(
                _clip_axis_segment_to_free_space(seg_a, seg_b, outer_offset_box, active_islands, cutter_radius)
            )
        finish_chains.extend(_stitch_directed_segments(outer_clipped))

        # 4b. Every active standing island rectangle [ix0 - R, ix1 + R] x [iy0 - R, iy1 + R], clipped
        for isl_idx, (ix0, iy0, ix1, iy1) in enumerate(active_islands):
            jx0, jy0 = ix0 - cutter_radius, iy0 - cutter_radius
            jx1, jy1 = ix1 + cutter_radius, iy1 + cutter_radius
            isl_edges = [
                ((jx0, jy0), (jx1, jy0)),
                ((jx1, jy0), (jx1, jy1)),
                ((jx1, jy1), (jx0, jy1)),
                ((jx0, jy1), (jx0, jy0)),
            ]
            isl_clipped: List[Tuple[Point2D, Point2D]] = []
            for seg_a, seg_b in isl_edges:
                isl_clipped.extend(
                    _clip_axis_segment_to_free_space(seg_a, seg_b, outer_offset_box, active_islands, cutter_radius)
                )
            finish_chains.extend(_stitch_directed_segments(isl_clipped))

        # Execute analytical finishing chains in nearest-neighbor order from cur_xy
        while finish_chains:
            if cur_xy is None:
                nxt_chain = finish_chains.pop(0)
            else:
                best_idx = min(
                    range(len(finish_chains)),
                    key=lambda k: math.hypot(
                        _orient_chain_nearest(finish_chains[k], cur_xy)[0][0] - cur_xy[0],
                        _orient_chain_nearest(finish_chains[k], cur_xy)[0][1] - cur_xy[1],
                    ),
                )
                nxt_chain = finish_chains.pop(best_idx)
            _link_or_enter_chain(nxt_chain)

    # 5. Sub-grid 2D swept-disk residual check (catches any narrow pinch/cusp missed when finish_spring_pass=False)
    valid_centers = cavity_mask & (dist_field >= cutter_radius + 0.25 * h)
    if np.any(valid_centers):
        edt_valid = distance_transform_edt(~valid_centers, sampling=h)
        reachable_cavity = cavity_mask & (edt_valid <= cutter_radius - 0.5 * h)
        for _ in range(8):
            if not np.any(swept_centers):
                uncut = reachable_cavity
            else:
                edt_swept = distance_transform_edt(~swept_centers, sampling=h)
                uncut = reachable_cavity & (edt_swept > cutter_radius - 0.5 * h)
            if not np.any(uncut):
                break
            u_lbl, u_cnt = label(uncut)
            vc_i, vc_j = np.nonzero(valid_centers)
            vc_x = x_origin + (vc_i + 0.5) * h
            vc_y = y_origin + (vc_j + 0.5) * h
            for u_id in range(1, int(u_cnt) + 1):
                ui, uj = np.nonzero(u_lbl == u_id)
                if len(ui) == 0:
                    continue
                ux = float(x_origin + (ui.mean() + 0.5) * h)
                uy = float(y_origin + (uj.mean() + 0.5) * h)
                d2 = (vc_x - ux) ** 2 + (vc_y - uy) ** 2
                k_min = int(np.argmin(d2))
                tx, ty = float(vc_x[k_min]), float(vc_y[k_min])
                # verify exact analytical clearance before emitting cleanup stroke
                if _segment_wall_and_island_clearance((tx, ty), (tx, ty), cavity_box, active_islands) < cutter_radius:
                    continue
                # Find farthest valid point in same uncut neighborhood for a short cleanup segment
                u_x0 = float(x_origin + (ui.min() + 0.5) * h)
                u_y0 = float(y_origin + (uj.min() + 0.5) * h)
                u_x1 = float(x_origin + (ui.max() + 0.5) * h)
                u_y1 = float(y_origin + (uj.max() + 0.5) * h)
                k_a = int(np.argmin((vc_x - u_x0) ** 2 + (vc_y - u_y0) ** 2))
                k_b = int(np.argmin((vc_x - u_x1) ** 2 + (vc_y - u_y1) ** 2))
                pa = (float(vc_x[k_a]), float(vc_y[k_a]))
                pb = (float(vc_x[k_b]), float(vc_y[k_b]))
                if (
                    math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 0.05
                    and _segment_wall_and_island_clearance(pa, pb, cavity_box, active_islands) >= cutter_radius - 1e-6
                ):
                    _link_or_enter_chain([pa, pb])
                else:
                    # Short micro-jog inside valid center
                    for dx_j, dy_j in ((0.15, 0.0), (-0.15, 0.0), (0.0, 0.15), (0.0, -0.15)):
                        cand = (tx + dx_j, ty + dy_j)
                        if _segment_wall_and_island_clearance((tx, ty), cand, cavity_box, active_islands) >= cutter_radius:
                            _link_or_enter_chain([(tx, ty), cand])
                            break

    # 6. Retract vertically as FEED up to z_safe at the end of the Z layer
    if cur_xy is not None:
        waypoints.append(
            (C.SegmentKind.FEED, (cur_xy[0], cur_xy[1], z_safe), config.retract_feed_mm_s, "retract")
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_pocket_toolpath(
    task: C.TaskSpec,
    config: PocketConfig = PocketConfig(),
) -> C.MotionPlan:
    """Generate a 2.5D Distance-Field Contour-Parallel Pocket Clearing MotionPlan."""
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        raise ValueError(f"generate_pocket_toolpath requires a FLAT cutter, got {task.tool.cutter_kind!r}")

    bmin, bmax, pmin, pmax, raw_islands = _extract_pocket_geometry(task)
    bx0, by0, _ = bmin
    bx1, by1, _ = bmax
    px0, py0, pz0 = pmin
    px1, py1, pz1 = pmax

    stock_max_z = float(task.stock.params["max_mm"][2])
    z_top = min(pz1, stock_max_z)
    z_bot = pz0
    total_depth = z_top - z_bot
    if total_depth <= 1e-9:
        raise ValueError(f"Pocket has non-positive cutting depth: z_top={z_top}, z_bot={z_bot}")

    r = float(task.tool.cutter_radius_mm)
    inv_machine = task.work_to_machine.inverse().apply
    if not task.initial_state.known or task.initial_state.position_mm is None:
        raise ValueError("TaskSpec.initial_state must have a known starting position")
    init_work = inv_machine(task.initial_state.position_mm)
    z_safe = max(float(init_work[2]), stock_max_z + 2.0)

    env_lo = inv_machine(task.limits.envelope_min_mm)
    env_hi = inv_machine(task.limits.envelope_max_mm)

    walled_x0 = px0 > bx0 + 1e-9
    walled_x1 = px1 < bx1 - 1e-9
    walled_y0 = py0 > by0 + 1e-9
    walled_y1 = py1 < by1 - 1e-9

    eff_px0 = px0 if walled_x0 else max(bx0 - r - 0.5, env_lo[0] + r + 0.5)
    eff_px1 = px1 if walled_x1 else min(bx1 + r + 0.5, env_hi[0] - r - 0.5)
    eff_py0 = py0 if walled_y0 else max(by0 - r - 0.5, env_lo[1] + r + 0.5)
    eff_py1 = py1 if walled_y1 else min(by1 + r + 0.5, env_hi[1] - r - 0.5)

    if eff_px1 - eff_px0 < 2.0 * r - 1e-9 or eff_py1 - eff_py0 < 2.0 * r - 1e-9:
        raise ValueError(
            f"Pocket XY dimensions ({eff_px1 - eff_px0:.3f} x {eff_py1 - eff_py0:.3f} mm) "
            f"cannot accommodate tool diameter {2.0 * r:.3f} mm"
        )

    cavity_box: Box2D = (eff_px0, eff_py0, eff_px1, eff_py1)
    outer_offset_box: Box2D = (eff_px0 + r, eff_py0 + r, eff_px1 - r, eff_py1 - r)

    if config.axial_stepdown_mm is None or config.axial_stepdown_mm >= total_depth - 1e-9:
        z_layers = [z_bot]
    else:
        n_layers = int(math.ceil((total_depth - 1e-9) / float(config.axial_stepdown_mm)))
        z_layers = [z_top - total_depth * (k / n_layers) for k in range(1, n_layers + 1)]
        z_layers[-1] = z_bot

    waypoints: List[Waypoint] = []
    if float(init_work[2]) < z_safe - 1e-6:
        waypoints.append(
            (C.SegmentKind.FEED, (float(init_work[0]), float(init_work[1]), z_safe), config.retract_feed_mm_s, "init_retract")
        )

    for layer_idx, z_cut in enumerate(z_layers):
        z_stock_top = z_top if layer_idx == 0 else z_layers[layer_idx - 1]
        active_islands: List[Box2D] = []
        for ilo, ihi in raw_islands:
            if float(ihi[2]) > z_cut + 1e-6 and float(ilo[2]) < z_top - 1e-9:
                ix0 = max(eff_px0, float(ilo[0]))
                iy0 = max(eff_py0, float(ilo[1]))
                ix1 = min(eff_px1, float(ihi[0]))
                iy1 = min(eff_py1, float(ihi[1]))
                if ix1 > ix0 + 1e-6 and iy1 > iy0 + 1e-6:
                    active_islands.append((ix0, iy0, ix1, iy1))

        _plan_single_z_layer(
            waypoints=waypoints,
            z_cut=z_cut,
            z_stock_top=z_stock_top,
            z_safe=z_safe,
            cavity_box=cavity_box,
            outer_offset_box=outer_offset_box,
            active_islands=active_islands,
            cutter_radius=r,
            config=config,
        )

    if not waypoints:
        raise ValueError("Pocket cavity contains no reachable tool-centre region")

    return _waypoints_to_motion_plan(task, waypoints, provenance=PROVENANCE_ID)


__all__ = [
    "PocketConfig",
    "generate_pocket_toolpath",
]
