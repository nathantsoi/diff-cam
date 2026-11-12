# SPDX-License-Identifier: MIT
"""Clean-room MIT 2.5D engagement-bounded adaptive & trochoidal pocket clearing.

Implements genuine 2.5D high-efficiency milling (HEM) mechanics for pockets with
``0..N`` standing islands, narrow pinched-throat channels, and open/walled chambers:

1. Safe cutter-center domain ``F = {x : dist_to_walls(x) >= R + finish_stock}``
   with maximal-clearance entry discovery via ``scipy.ndimage.distance_transform_edt``
   and helical spiral bore + Archimedean radial pocket opening down to ``pz0``.
2. Weighted geodesic wavefront propagation that switches naturally between:
   - **Circular trochoidal loops** (engaged forward cutting arc at ``nominal_feed_mm_s``
     + disengaged interior return chord/arc at ``return_feed_multiplier * nominal_feed_mm_s``)
     through narrow channels and pinched throats, and
   - **Constant-stepover engagement-bounded offset peel passes** (climb-oriented peel
     strokes + fast cleared-IPW returns) across wider chambers,
   while continuously updating an internal 2D In-Process Workpiece (IPW) mask.
3. 2D IPW residual verification & cleanup followed by exact analytical wall-finishing
   loops around the outer pocket boundary and every standing island (analytically
   clipped against walls/islands and linked via ``z_safe`` hops when crossing obstacles).
"""

from __future__ import annotations

import dataclasses
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from benchmarks import contracts as C

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
WaypointSpec = Tuple[C.SegmentKind, Point3D, Optional[float], str]
Rect2D = Tuple[Point2D, Point2D]

PROVENANCE_ID = "mit_cleanroom:adaptive_strategy@2"


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for 2.5D engagement-bounded adaptive / trochoidal clearing."""

    optimal_load_fraction: float = 0.35
    trochoid_pitch_fraction: float = 0.32
    helix_bore_radius_fraction: float = 0.35
    return_feed_multiplier: float = 1.8
    ipw_pitch_mm: float = 0.20
    nominal_feed_mm_s: float = 5.5
    plunge_feed_mm_s: float = 3.5
    retract_feed_mm_s: float = 14.0
    chord_step_mm: float = 0.75
    simplify_tol_mm: float = 0.05
    max_doc_mm: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.05 <= self.optimal_load_fraction <= 0.95):
            raise ValueError("optimal_load_fraction must be in [0.05, 0.95]")
        if not (0.05 <= self.trochoid_pitch_fraction <= 0.95):
            raise ValueError("trochoid_pitch_fraction must be in [0.05, 0.95]")
        if not (0.05 <= self.helix_bore_radius_fraction <= 0.95):
            raise ValueError("helix_bore_radius_fraction must be in [0.05, 0.95]")
        if self.return_feed_multiplier < 1.0:
            raise ValueError("return_feed_multiplier must be >= 1.0")
        if self.ipw_pitch_mm <= 0.0 or not math.isfinite(self.ipw_pitch_mm):
            raise ValueError("ipw_pitch_mm must be finite and positive")
        if self.chord_step_mm <= 0.05 or not math.isfinite(self.chord_step_mm):
            raise ValueError("chord_step_mm must be finite and > 0.05")
        if self.simplify_tol_mm <= 0.0 or not math.isfinite(self.simplify_tol_mm):
            raise ValueError("simplify_tol_mm must be finite and positive")
        if self.max_doc_mm is not None and (self.max_doc_mm <= 0.0 or not math.isfinite(self.max_doc_mm)):
            raise ValueError("max_doc_mm must be finite and positive")
        for name in ("nominal_feed_mm_s", "plunge_feed_mm_s", "retract_feed_mm_s"):
            val = getattr(self, name)
            if val <= 0.0 or not math.isfinite(val):
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class _PocketGeometry:
    box_min: Point3D
    box_max: Point3D
    pocket_min: Point3D
    pocket_max: Point3D
    islands_2d: Tuple[Rect2D, ...]
    walled_x0: bool
    walled_x1: bool
    walled_y0: bool
    walled_y1: bool
    env_xy_min: Point2D
    env_xy_max: Point2D


@dataclass
class _WaveContour:
    level_idx: int
    points: List[Point2D]
    is_closed: bool
    length_mm: float
    peak_pt: Point2D
    peak_clearance_mm: float
    channel_clearance_mm: float


# ---------------------------------------------------------------------------
# Post-quantization safe coordinate rounding helpers
# ---------------------------------------------------------------------------
def _safe_hi(val: float, decimals: int = 4) -> float:
    """Round upward only if needed so 4-decimal NC quantization never crosses below ``val``."""
    scale = 10.0 ** decimals
    if abs(round(val, decimals) - val) <= 1e-11:
        return float(round(val, decimals))
    return math.ceil(val * scale - 1e-9) / scale


def _safe_lo(val: float, decimals: int = 4) -> float:
    """Round downward only if needed so 4-decimal NC quantization never crosses above ``val``."""
    scale = 10.0 ** decimals
    if abs(round(val, decimals) - val) <= 1e-11:
        return float(round(val, decimals))
    return math.floor(val * scale + 1e-9) / scale


# ---------------------------------------------------------------------------
# Geometry parsing & analytical 2D wall/island distance functions
# ---------------------------------------------------------------------------
def _extract_pocket_geometry(task: C.TaskSpec) -> _PocketGeometry:
    part = task.original_part
    if part.kind != "analytic":
        raise ValueError(f"adaptive strategy requires analytic original_part, got {part.kind!r}")
    p = part.params
    prim = p.get("primitive")
    if prim == "box_minus_multi_pocket":
        box_min = tuple(float(v) for v in p["box_min_mm"])
        box_max = tuple(float(v) for v in p["box_max_mm"])
        pocket_min = tuple(float(v) for v in p["pocket_min_mm"])
        pocket_max = tuple(float(v) for v in p["pocket_max_mm"])
        raw_islands = p.get("islands_mm", ())
    elif prim == "box_minus_pocket":
        box_min = tuple(float(v) for v in p["box_min_mm"])
        box_max = tuple(float(v) for v in p["box_max_mm"])
        pocket_min = tuple(float(v) for v in p["pocket_min_mm"])
        pocket_max = tuple(float(v) for v in p["pocket_max_mm"])
        raw_islands = ((p["island_min_mm"], p["island_max_mm"]),)
    else:
        raise ValueError(f"unsupported original_part primitive {prim!r}")

    pz0 = pocket_min[2]
    islands_2d: List[Rect2D] = []
    for isl in raw_islands:
        ilo, ihi = isl[0], isl[1]
        if float(ihi[2]) > pz0 + 1e-6:
            islands_2d.append(
                (
                    (float(ilo[0]), float(ilo[1])),
                    (float(ihi[0]), float(ihi[1])),
                )
            )

    inv = task.work_to_machine.inverse()
    c0 = inv.apply(task.limits.envelope_min_mm)
    c1 = inv.apply(task.limits.envelope_max_mm)
    env_xy_min = (min(c0[0], c1[0]) + 0.25, min(c0[1], c1[1]) + 0.25)
    env_xy_max = (max(c0[0], c1[0]) - 0.25, max(c0[1], c1[1]) - 0.25)

    return _PocketGeometry(
        box_min=box_min,  # type: ignore[arg-type]
        box_max=box_max,  # type: ignore[arg-type]
        pocket_min=pocket_min,  # type: ignore[arg-type]
        pocket_max=pocket_max,  # type: ignore[arg-type]
        islands_2d=tuple(islands_2d),
        walled_x0=pocket_min[0] > box_min[0] + 1e-9,
        walled_x1=pocket_max[0] < box_max[0] - 1e-9,
        walled_y0=pocket_min[1] > box_min[1] + 1e-9,
        walled_y1=pocket_max[1] < box_max[1] - 1e-9,
        env_xy_min=env_xy_min,
        env_xy_max=env_xy_max,
    )


def _outer_cutter_bounds(geom: _PocketGeometry, radius: float, margin: float = 0.0) -> Tuple[float, float, float, float]:
    """Safe cutter-center rectangle [ox0, ox1] x [oy0, oy1] respecting walled vs open sides."""
    px0, py0, _ = geom.pocket_min
    px1, py1, _ = geom.pocket_max
    r_eff = radius + margin
    # On open sides, allow slight overtravel (0.35 * R) clamped to machine envelope so open edges clear cleanly.
    overtravel = 0.35 * radius
    ox0 = (px0 + r_eff) if geom.walled_x0 else max(geom.env_xy_min[0], px0 - overtravel + margin)
    ox1 = (px1 - r_eff) if geom.walled_x1 else min(geom.env_xy_max[0], px1 + overtravel - margin)
    oy0 = (py0 + r_eff) if geom.walled_y0 else max(geom.env_xy_min[1], py0 - overtravel + margin)
    oy1 = (py1 - r_eff) if geom.walled_y1 else min(geom.env_xy_max[1], py1 + overtravel - margin)
    if margin == 0.0:
        if geom.walled_x0:
            ox0 = _safe_hi(ox0)
        if geom.walled_x1:
            ox1 = _safe_lo(ox1)
        if geom.walled_y0:
            oy0 = _safe_hi(oy0)
        if geom.walled_y1:
            oy1 = _safe_lo(oy1)
    return ox0, ox1, oy0, oy1


def _wall_distance_grid(
    gx: np.ndarray,
    gy: np.ndarray,
    geom: _PocketGeometry,
    radius: float,
) -> np.ndarray:
    """Exact signed 2D distance to pocket walls and standing islands (positive in cavity)."""
    px0, py0, _ = geom.pocket_min
    px1, py1, _ = geom.pocket_max
    over = 0.35 * radius
    x_lo = px0 if geom.walled_x0 else (max(geom.env_xy_min[0], px0 - over) - radius)
    x_hi = px1 if geom.walled_x1 else (min(geom.env_xy_max[0], px1 + over) + radius)
    y_lo = py0 if geom.walled_y0 else (max(geom.env_xy_min[1], py0 - over) - radius)
    y_hi = py1 if geom.walled_y1 else (min(geom.env_xy_max[1], py1 + over) + radius)

    d = np.minimum.reduce([gx - x_lo, x_hi - gx, gy - y_lo, y_hi - gy])
    for (ix0, iy0), (ix1, iy1) in geom.islands_2d:
        qx = np.maximum(ix0 - gx, gx - ix1)
        qy = np.maximum(iy0 - gy, gy - iy1)
        outside_d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        inside_d = np.minimum(np.maximum(qx, qy), 0.0)
        d = np.minimum(d, outside_d + inside_d)
    return d


def _wall_distance_point(pt: Point2D, geom: _PocketGeometry, radius: float) -> float:
    x, y = pt
    px0, py0, _ = geom.pocket_min
    px1, py1, _ = geom.pocket_max
    over = 0.35 * radius
    x_lo = px0 if geom.walled_x0 else (max(geom.env_xy_min[0], px0 - over) - radius)
    x_hi = px1 if geom.walled_x1 else (min(geom.env_xy_max[0], px1 + over) + radius)
    y_lo = py0 if geom.walled_y0 else (max(geom.env_xy_min[1], py0 - over) - radius)
    y_hi = py1 if geom.walled_y1 else (min(geom.env_xy_max[1], py1 + over) + radius)

    d = min(x - x_lo, x_hi - x, y - y_lo, y_hi - y)
    for (ix0, iy0), (ix1, iy1) in geom.islands_2d:
        qx = max(ix0 - x, x - ix1)
        qy = max(iy0 - y, y - iy1)
        if qx <= 0.0 and qy <= 0.0:
            d_isl = max(qx, qy)
        else:
            d_isl = math.hypot(max(qx, 0.0), max(qy, 0.0))
        if d_isl < d:
            d = d_isl
    return d


# ---------------------------------------------------------------------------
# Exact analytical 1D interval clipping of a 2D line segment against walls/islands
# ---------------------------------------------------------------------------
def _subtract_interval(
    intervals: List[Tuple[float, float]],
    bad_lo: float,
    bad_hi: float,
) -> List[Tuple[float, float]]:
    if bad_hi <= bad_lo + 1e-12:
        return intervals
    out: List[Tuple[float, float]] = []
    for a, b in intervals:
        if bad_hi <= a + 1e-12 or bad_lo >= b - 1e-12:
            out.append((a, b))
        else:
            if bad_lo > a + 1e-9:
                out.append((a, bad_lo))
            if bad_hi < b - 1e-9:
                out.append((bad_hi, b))
    return out


def _open_box_hit_interval(
    ax: float,
    ay: float,
    dx: float,
    dy: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> Optional[Tuple[float, float]]:
    """Parameter interval t in (0, 1) where a + t*d lies strictly inside (x0, x1) x (y0, y1)."""
    t_lo, t_hi = 0.0, 1.0
    for p0, dp, lo, hi in ((ax, dx, x0, x1), (ay, dy, y0, y1)):
        if abs(dp) <= 1e-12:
            if not (lo < p0 < hi):
                return None
        else:
            t1 = (lo - p0) / dp
            t2 = (hi - p0) / dp
            if t1 > t2:
                t1, t2 = t2, t1
            t_lo = max(t_lo, t1)
            t_hi = min(t_hi, t2)
            if t_hi <= t_lo + 1e-12:
                return None
    return (t_lo, t_hi)


def _open_disk_hit_interval(
    ax: float,
    ay: float,
    dx: float,
    dy: float,
    cx: float,
    cy: float,
    r: float,
) -> Optional[Tuple[float, float]]:
    """Parameter interval t in (0, 1) where ||a + t*d - c|| < r."""
    fx, fy = ax - cx, ay - cy
    qa = dx * dx + dy * dy
    if qa <= 1e-15:
        return (0.0, 1.0) if (fx * fx + fy * fy < r * r - 1e-12) else None
    qb = 2.0 * (fx * dx + fy * dy)
    qc = fx * fx + fy * fy - r * r
    disc = qb * qb - 4.0 * qa * qc
    if disc <= 1e-14:
        return None
    sq = math.sqrt(disc)
    t1 = (-qb - sq) / (2.0 * qa)
    t2 = (-qb + sq) / (2.0 * qa)
    t_lo = max(0.0, t1)
    t_hi = min(1.0, t2)
    return (t_lo, t_hi) if t_hi > t_lo + 1e-12 else None


def _clip_segment_safe(
    p0: Point2D,
    p1: Point2D,
    geom: _PocketGeometry,
    radius: float,
    margin: float = 0.0,
) -> List[Tuple[Point2D, Point2D]]:
    """Analytically clip segment p0 -> p1 to the closed safe cutter-center domain."""
    ox0, ox1, oy0, oy1 = _outer_cutter_bounds(geom, radius, margin=margin)
    if ox1 < ox0 - 1e-9 or oy1 < oy0 - 1e-9:
        return []
    ax, ay = p0
    dx, dy = p1[0] - ax, p1[1] - ay

    # Clip to outer closed rectangle [ox0, ox1] x [oy0, oy1]
    t_lo, t_hi = 0.0, 1.0
    for q0, dq, lo, hi in ((ax, dx, ox0, ox1), (ay, dy, oy0, oy1)):
        if abs(dq) <= 1e-12:
            if q0 < lo - 1e-9 or q0 > hi + 1e-9:
                return []
        else:
            u1 = (lo - q0) / dq
            u2 = (hi - q0) / dq
            if u1 > u2:
                u1, u2 = u2, u1
            t_lo = max(t_lo, u1)
            t_hi = min(t_hi, u2)
            if t_hi < t_lo - 1e-12:
                return []

    t_lo = min(max(t_lo, 0.0), 1.0)
    t_hi = min(max(t_hi, 0.0), 1.0)
    intervals: List[Tuple[float, float]] = [(t_lo, t_hi)]
    r_eff = radius + margin - 1e-7

    for (ix0, iy0), (ix1, iy1) in geom.islands_2d:
        # Subtract open Minkowski sum of island with disk of radius r_eff:
        # 2 open rectangles + 4 open corner disks + core open rectangle
        for bx0, bx1, by0, by1 in (
            (ix0 - r_eff, ix1 + r_eff, iy0, iy1),
            (ix0, ix1, iy0 - r_eff, iy1 + r_eff),
            (ix0 - 1e-7, ix1 + 1e-7, iy0 - 1e-7, iy1 + 1e-7),
        ):
            hit = _open_box_hit_interval(ax, ay, dx, dy, bx0, bx1, by0, by1)
            if hit is not None:
                intervals = _subtract_interval(intervals, hit[0], hit[1])
                if not intervals:
                    return []
        for cx, cy in ((ix0, iy0), (ix1, iy0), (ix1, iy1), (ix0, iy1)):
            hit = _open_disk_hit_interval(ax, ay, dx, dy, cx, cy, r_eff)
            if hit is not None:
                intervals = _subtract_interval(intervals, hit[0], hit[1])
                if not intervals:
                    return []

    segs: List[Tuple[Point2D, Point2D]] = []
    for u0, u1 in intervals:
        qa = (ax + u0 * dx, ay + u0 * dy)
        qb = (ax + u1 * dx, ay + u1 * dy)
        if math.hypot(qb[0] - qa[0], qb[1] - qa[1]) > 1e-4:
            segs.append((qa, qb))
    return segs


def _is_segment_safe(
    p0: Point2D,
    p1: Point2D,
    geom: _PocketGeometry,
    radius: float,
    margin: float = 0.0,
) -> bool:
    """True iff the entire closed line segment p0 -> p1 stays in the safe domain."""
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if dist <= 1e-6:
        return _wall_distance_point(p0, geom, radius) >= radius + margin - 1e-6
    clipped = _clip_segment_safe(p0, p1, geom, radius, margin=margin)
    if len(clipped) != 1:
        return False
    qa, qb = clipped[0]
    return (
        math.hypot(qa[0] - p0[0], qa[1] - p0[1]) <= 1e-4
        and math.hypot(qb[0] - p1[0], qb[1] - p1[1]) <= 1e-4
    )


# ---------------------------------------------------------------------------
# 2D In-Process Workpiece (IPW) grid tracker
# ---------------------------------------------------------------------------
class _IPWGrid2D:
    """Boolean 2D material-remaining grid supporting fast exact capsule sweeps."""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        pitch_mm: float,
        initial_mask_fn=None,
    ) -> None:
        self.pitch = float(pitch_mm)
        self.nx = max(2, int(math.ceil((x_max - x_min) / self.pitch)))
        self.ny = max(2, int(math.ceil((y_max - y_min) / self.pitch)))
        self.x0 = float(x_min)
        self.y0 = float(y_min)
        self.xs = self.x0 + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.pitch
        self.ys = self.y0 + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.pitch
        self.gx, self.gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        if initial_mask_fn is None:
            self.remaining = (self.gx <= x_max) & (self.gy <= y_max)
        else:
            self.remaining = np.asarray(initial_mask_fn(self.gx, self.gy), dtype=bool)
        self.cell_area = self.pitch * self.pitch

    def _bbox_slice(self, xa: float, ya: float, xb: float, yb: float, r: float) -> Tuple[slice, slice]:
        ix0 = max(0, int(math.floor((min(xa, xb) - r - self.x0) / self.pitch)))
        ix1 = min(self.nx, int(math.ceil((max(xa, xb) + r - self.x0) / self.pitch)) + 1)
        iy0 = max(0, int(math.floor((min(ya, yb) - r - self.y0) / self.pitch)))
        iy1 = min(self.ny, int(math.ceil((max(ya, yb) + r - self.y0) / self.pitch)) + 1)
        return slice(ix0, ix1), slice(iy0, iy1)

    def sweep_segment(
        self,
        p0: Point2D,
        p1: Point2D,
        radius: float,
        *,
        mutate: bool = True,
    ) -> float:
        """Return newly swept 2D material area (mm^2) along capsule(p0, p1, radius)."""
        xa, ya = p0
        xb, yb = p1
        sx, sy = self._bbox_slice(xa, ya, xb, yb, radius)
        if sx.start >= sx.stop or sy.start >= sy.stop:
            return 0.0
        sub_rem = self.remaining[sx, sy]
        if not np.any(sub_rem):
            return 0.0
        sub_x = self.gx[sx, sy]
        sub_y = self.gy[sx, sy]
        dx, dy = xb - xa, yb - ya
        l2 = dx * dx + dy * dy
        if l2 <= 1e-14:
            dist2 = (sub_x - xa) ** 2 + (sub_y - ya) ** 2
        else:
            t = np.clip(((sub_x - xa) * dx + (sub_y - ya) * dy) / l2, 0.0, 1.0)
            dist2 = (sub_x - (xa + t * dx)) ** 2 + (sub_y - (ya + t * dy)) ** 2
        hit = dist2 <= (radius * radius)
        newly_cut = int(np.count_nonzero(sub_rem & hit))
        if mutate and newly_cut > 0:
            sub_rem[hit] = False
        return newly_cut * self.cell_area

    def sweep_polyline(
        self,
        pts: Sequence[Point2D],
        radius: float,
        *,
        mutate: bool = True,
    ) -> float:
        if not pts:
            return 0.0
        if len(pts) == 1:
            return self.sweep_segment(pts[0], pts[0], radius, mutate=mutate)
        if not mutate:
            # Non-mutating union query over bounding box
            xs_p = [p[0] for p in pts]
            ys_p = [p[1] for p in pts]
            sx, sy = self._bbox_slice(min(xs_p), min(ys_p), max(xs_p), max(ys_p), radius)
            if sx.start >= sx.stop or sy.start >= sy.stop:
                return 0.0
            sub_rem = self.remaining[sx, sy]
            if not np.any(sub_rem):
                return 0.0
            sub_x = self.gx[sx, sy]
            sub_y = self.gy[sx, sy]
            union_hit = np.zeros_like(sub_rem)
            r2 = radius * radius
            for k in range(len(pts) - 1):
                xa, ya = pts[k]
                xb, yb = pts[k + 1]
                dx, dy = xb - xa, yb - ya
                l2 = dx * dx + dy * dy
                if l2 <= 1e-14:
                    d2 = (sub_x - xa) ** 2 + (sub_y - ya) ** 2
                else:
                    t = np.clip(((sub_x - xa) * dx + (sub_y - ya) * dy) / l2, 0.0, 1.0)
                    d2 = (sub_x - (xa + t * dx)) ** 2 + (sub_y - (ya + t * dy)) ** 2
                union_hit |= d2 <= r2
            return int(np.count_nonzero(sub_rem & union_hit)) * self.cell_area
        total = 0.0
        for k in range(len(pts) - 1):
            total += self.sweep_segment(pts[k], pts[k + 1], radius, mutate=True)
        return total


# ---------------------------------------------------------------------------
# Weighted Geodesic Wavefront + Marching Squares Contour Extraction
# ---------------------------------------------------------------------------
def _compute_weighted_geodesic(
    comp_mask: np.ndarray,
    seed_mask: np.ndarray,
    weights: np.ndarray,
    pitch: float,
) -> np.ndarray:
    """8-connected Dijkstra Eikonal approximation on ``comp_mask`` from ``seed_mask``."""
    nx, ny = comp_mask.shape
    dist = np.full((nx, ny), np.inf, dtype=np.float64)
    heap: List[Tuple[float, int, int]] = []

    seed_indices = np.argwhere(comp_mask & seed_mask)
    for idx in seed_indices:
        i, j = int(idx[0]), int(idx[1])
        dist[i, j] = 0.0
        heapq.heappush(heap, (0.0, i, j))

    diag = math.sqrt(2.0) * pitch
    neighbors = (
        (-1, 0, pitch),
        (1, 0, pitch),
        (0, -1, pitch),
        (0, 1, pitch),
        (-1, -1, diag),
        (-1, 1, diag),
        (1, -1, diag),
        (1, 1, diag),
    )

    while heap:
        d_cur, i, j = heapq.heappop(heap)
        if d_cur > dist[i, j] + 1e-12:
            continue
        w_cur = float(weights[i, j])
        for di, dj, step_len in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny and comp_mask[ni, nj]:
                # Prevent diagonal corner-cutting across 4-disconnected pinches
                if di != 0 and dj != 0 and not (comp_mask[i + di, j] and comp_mask[i, j + dj]):
                    continue
                w_avg = 0.5 * (w_cur + float(weights[ni, nj]))
                nd = d_cur + step_len * w_avg
                if nd < dist[ni, nj] - 1e-12:
                    dist[ni, nj] = nd
                    heapq.heappush(heap, (nd, ni, nj))
    return dist


EdgeKey = Tuple[str, int, int]


def _marching_squares_oriented(
    field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    level: float,
) -> List[Tuple[List[Point2D], bool]]:
    """Extract oriented iso-contours at ``field == level`` with ``field > level`` on the RIGHT."""
    v = field - level
    # Avoid exact zero on grid vertices so edge crossings are non-degenerate
    v = np.where(v == 0.0, -1e-12, v)
    nx, ny = v.shape

    edge_pts: Dict[EdgeKey, Point2D] = {}
    next_edge: Dict[EdgeKey, EdgeKey] = {}
    in_deg: Dict[EdgeKey, int] = {}

    def _interp_h(i: int, j: int) -> EdgeKey:
        key: EdgeKey = ("h", i, j)
        if key not in edge_pts:
            va, vb = float(v[i, j]), float(v[i + 1, j])
            t = 0.5 if abs(va - vb) <= 1e-15 else va / (va - vb)
            t = min(max(t, 0.0), 1.0)
            edge_pts[key] = (float(xs[i] + t * (xs[i + 1] - xs[i])), float(ys[j]))
        return key

    def _interp_v(i: int, j: int) -> EdgeKey:
        key: EdgeKey = ("v", i, j)
        if key not in edge_pts:
            va, vb = float(v[i, j]), float(v[i, j + 1])
            t = 0.5 if abs(va - vb) <= 1e-15 else va / (va - vb)
            t = min(max(t, 0.0), 1.0)
            edge_pts[key] = (float(xs[i]), float(ys[j] + t * (ys[j + 1] - ys[j])))
        return key

    def _add_dir(ea: EdgeKey, eb: EdgeKey) -> None:
        next_edge[ea] = eb
        in_deg[eb] = in_deg.get(eb, 0) + 1
        in_deg.setdefault(ea, 0)

    pos = v > 0.0
    m_grid = (
        pos[:-1, :-1].astype(np.int32)
        + (pos[1:, :-1].astype(np.int32) << 1)
        + (pos[1:, 1:].astype(np.int32) << 2)
        + (pos[:-1, 1:].astype(np.int32) << 3)
    )
    active_cells = np.argwhere((m_grid > 0) & (m_grid < 15))

    for cell in active_cells:
        i, j = int(cell[0]), int(cell[1])
        mask = int(m_grid[i, j])
        if mask == 1:
            _add_dir(_interp_v(i, j), _interp_h(i, j))
        elif mask == 2:
            _add_dir(_interp_h(i, j), _interp_v(i + 1, j))
        elif mask == 3:
            _add_dir(_interp_v(i, j), _interp_v(i + 1, j))
        elif mask == 4:
            _add_dir(_interp_v(i + 1, j), _interp_h(i, j + 1))
        elif mask == 5:
            vc = 0.25 * float(v[i, j] + v[i + 1, j] + v[i + 1, j + 1] + v[i, j + 1])
            if vc > 0.0:
                _add_dir(_interp_v(i, j), _interp_h(i, j + 1))
                _add_dir(_interp_v(i + 1, j), _interp_h(i, j))
            else:
                _add_dir(_interp_v(i, j), _interp_h(i, j))
                _add_dir(_interp_v(i + 1, j), _interp_h(i, j + 1))
        elif mask == 6:
            _add_dir(_interp_h(i, j), _interp_h(i, j + 1))
        elif mask == 7:
            _add_dir(_interp_v(i, j), _interp_h(i, j + 1))
        elif mask == 8:
            _add_dir(_interp_h(i, j + 1), _interp_v(i, j))
        elif mask == 9:
            _add_dir(_interp_h(i, j + 1), _interp_h(i, j))
        elif mask == 10:
            vc = 0.25 * float(v[i, j] + v[i + 1, j] + v[i + 1, j + 1] + v[i, j + 1])
            if vc > 0.0:
                _add_dir(_interp_h(i, j), _interp_v(i, j))
                _add_dir(_interp_h(i, j + 1), _interp_v(i + 1, j))
            else:
                _add_dir(_interp_h(i, j), _interp_v(i + 1, j))
                _add_dir(_interp_h(i, j + 1), _interp_v(i, j))
        elif mask == 11:
            _add_dir(_interp_h(i, j + 1), _interp_v(i + 1, j))
        elif mask == 12:
            _add_dir(_interp_v(i + 1, j), _interp_v(i, j))
        elif mask == 13:
            _add_dir(_interp_v(i + 1, j), _interp_h(i, j))
        elif mask == 14:
            _add_dir(_interp_h(i, j), _interp_v(i, j))

    visited: set[EdgeKey] = set()
    polylines: List[Tuple[List[Point2D], bool]] = []

    # 1. Open chains starting at in_deg == 0
    for start_key in sorted(k for k, deg in in_deg.items() if deg == 0 and k in next_edge):
        if start_key in visited:
            continue
        pts = [edge_pts[start_key]]
        visited.add(start_key)
        cur = start_key
        while cur in next_edge:
            nxt = next_edge[cur]
            pts.append(edge_pts[nxt])
            if nxt in visited:
                break
            visited.add(nxt)
            cur = nxt
        if len(pts) >= 2:
            polylines.append((pts, False))

    # 2. Closed cycles
    for start_key in sorted(next_edge.keys()):
        if start_key in visited:
            continue
        pts = [edge_pts[start_key]]
        visited.add(start_key)
        cur = start_key
        is_closed = False
        while cur in next_edge:
            nxt = next_edge[cur]
            if nxt == start_key:
                pts.append(edge_pts[start_key])
                is_closed = True
                break
            pts.append(edge_pts[nxt])
            if nxt in visited:
                break
            visited.add(nxt)
            cur = nxt
        if len(pts) >= 3:
            polylines.append((pts, is_closed))

    return polylines


def _clip_polyline_to_domain(
    pts: Sequence[Point2D],
    is_closed: bool,
    geom: _PocketGeometry,
    radius: float,
    margin: float,
) -> List[Tuple[List[Point2D], bool]]:
    """Clip a Marching Squares polyline analytically to the safe roughing domain."""
    pieces: List[List[Point2D]] = []
    cur: List[Point2D] = []
    all_intact = True

    for idx in range(len(pts) - 1):
        p0, p1 = pts[idx], pts[idx + 1]
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= 1e-5:
            continue
        segs = _clip_segment_safe(p0, p1, geom, radius, margin=margin)
        if not segs:
            all_intact = False
            if len(cur) >= 2:
                pieces.append(cur)
            cur = []
            continue
        if len(segs) > 1:
            all_intact = False
        for s_idx, (qa, qb) in enumerate(segs):
            if not cur:
                cur = [qa, qb]
            elif math.hypot(qa[0] - cur[-1][0], qa[1] - cur[-1][1]) <= 1e-3:
                cur.append(qb)
            else:
                all_intact = False
                if len(cur) >= 2:
                    pieces.append(cur)
                cur = [qa, qb]
    if len(cur) >= 2:
        pieces.append(cur)

    if not pieces:
        return []

    if is_closed and all_intact and len(pieces) == 1:
        loop = pieces[0]
        if math.hypot(loop[0][0] - loop[-1][0], loop[0][1] - loop[-1][1]) <= 1e-2:
            loop[-1] = loop[0]
            return [(loop, True)]

    # If a originally closed loop got clipped across boundary segments so piece[-1] meets piece[0], merge them!
    if is_closed and len(pieces) >= 2:
        if math.hypot(pieces[-1][-1][0] - pieces[0][0][0], pieces[-1][-1][1] - pieces[0][0][1]) <= 1e-2:
            merged = pieces[-1] + pieces[0][1:]
            pieces = [merged] + pieces[1:-1]

    return [(p, False) for p in pieces]


def _smooth_and_simplify_polyline(
    pts: Sequence[Point2D],
    is_closed: bool,
    geom: _PocketGeometry,
    radius: float,
    margin: float,
    tol_mm: float = 0.05,
) -> List[Point2D]:
    """Smooth Marching Squares staircase jitter and RDP-simplify while preserving exact safety."""
    if len(pts) <= 2:
        return list(pts)

    work = list(pts[:-1] if is_closed else pts)
    n = len(work)
    if n >= 4:
        for _ in range(2):
            nxt = list(work)
            for idx in range(0 if is_closed else 1, n if is_closed else n - 1):
                prev_pt = work[(idx - 1) % n]
                cur_pt = work[idx]
                next_pt = work[(idx + 1) % n]
                cand = (
                    0.25 * prev_pt[0] + 0.5 * cur_pt[0] + 0.25 * next_pt[0],
                    0.25 * prev_pt[1] + 0.5 * cur_pt[1] + 0.25 * next_pt[1],
                )
                if _wall_distance_point(cand, geom, radius) >= radius + margin:
                    nxt[idx] = cand
            work = nxt

    if is_closed:
        work.append(work[0])

    # Safety-checked Ramer-Douglas-Peucker simplification
    def _rdp(sub: List[Point2D]) -> List[Point2D]:
        if len(sub) <= 2:
            return sub
        a, b = sub[0], sub[-1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        l2 = dx * dx + dy * dy
        max_d = -1.0
        split_idx = 1
        for k in range(1, len(sub) - 1):
            px, py = sub[k]
            if l2 <= 1e-12:
                d = math.hypot(px - a[0], py - a[1])
            else:
                t = min(max(((px - a[0]) * dx + (py - a[1]) * dy) / l2, 0.0), 1.0)
                d = math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))
            if d > max_d:
                max_d = d
                split_idx = k
        if max_d <= tol_mm and _is_segment_safe(a, b, geom, radius, margin=margin):
            return [a, b]
        left = _rdp(sub[: split_idx + 1])
        right = _rdp(sub[split_idx:])
        return left[:-1] + right

    if is_closed and len(work) >= 4:
        mid = len(work) // 2
        h1 = _rdp(work[: mid + 1])
        h2 = _rdp(work[mid:])
        res = h1[:-1] + h2
        res[-1] = res[0]
        return res
    return _rdp(work)


def _polyline_length(pts: Sequence[Point2D]) -> float:
    return sum(
        math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
        for k in range(len(pts) - 1)
    )


def _min_polyline_dist(a: Sequence[Point2D], b: Sequence[Point2D]) -> Tuple[float, Point2D, Point2D]:
    best = math.inf
    pa_best = a[0]
    pb_best = b[0]
    step_a = max(1, len(a) // 12)
    step_b = max(1, len(b) // 12)
    idxs_a = sorted(set(list(range(0, len(a), step_a)) + [len(a) - 1]))
    idxs_b = sorted(set(list(range(0, len(b), step_b)) + [len(b) - 1]))
    for ia in idxs_a:
        xa, ya = a[ia]
        for ib in idxs_b:
            xb, yb = b[ib]
            d = math.hypot(xb - xa, yb - ya)
            if d < best:
                best = d
                pa_best = a[ia]
                pb_best = b[ib]
    return best, pa_best, pb_best


# ---------------------------------------------------------------------------
# Toolpath Builder with Engagement-Aware Safe Linking
# ---------------------------------------------------------------------------
_STRUCT_8 = np.ones((3, 3), dtype=np.int32)


class _AdaptivePathBuilder:
    def __init__(
        self,
        task: C.TaskSpec,
        config: AdaptiveConfig,
        geom: _PocketGeometry,
        ipw: _IPWGrid2D,
        z_safe: float,
        pz0: float,
        stock_max_z: float,
        d_wall_grid: Optional[np.ndarray] = None,
    ) -> None:
        self.task = task
        self.config = config
        self.geom = geom
        self.ipw = ipw
        self.z_safe = z_safe
        self.pz0 = pz0
        self.cur_pass_z: float = pz0
        self.stock_max_z = stock_max_z
        self.d_wall_grid = d_wall_grid
        self.r = float(task.tool.cutter_radius_mm)
        self.d = 2.0 * self.r
        self.ae_target = config.optimal_load_fraction * self.d

        max_f = float(task.limits.max_feed_mm_s)
        self.cut_feed = min(float(config.nominal_feed_mm_s), max_f)
        self.return_feed = min(float(config.return_feed_multiplier * config.nominal_feed_mm_s), max_f)
        self.plunge_feed = min(float(config.plunge_feed_mm_s), max_f)
        self.retract_feed = min(float(config.retract_feed_mm_s), max_f)

        inv = task.work_to_machine.inverse()
        init_w = inv.apply(task.initial_state.position_mm)
        self.cur_xyz: Point3D = (float(init_w[0]), float(init_w[1]), float(init_w[2]))
        self.waypoints: List[WaypointSpec] = []
        self.last_peel_pts: Optional[List[Point2D]] = None

        # Ensure initial Z is at or above z_safe before any XY rapid
        if self.cur_xyz[2] < self.z_safe - 1e-6:
            self._emit(C.SegmentKind.FEED, (self.cur_xyz[0], self.cur_xyz[1], self.z_safe), self.retract_feed, "lift_init")

    def _emit(
        self,
        kind: C.SegmentKind,
        target: Point3D,
        feed: Optional[float],
        tag: str,
        *,
        stamp_ipw: bool = True,
    ) -> None:
        dist = math.dist(self.cur_xyz, target)
        if dist <= 1e-3:
            return
        if kind != C.SegmentKind.RAPID and stamp_ipw:
            if min(self.cur_xyz[2], target[2]) < self.stock_max_z - 1e-6:
                self.ipw.sweep_segment(
                    (self.cur_xyz[0], self.cur_xyz[1]),
                    (target[0], target[1]),
                    self.r,
                    mutate=True,
                )
        self.waypoints.append((kind, target, feed, tag))
        self.cur_xyz = target

    def retract_to_safe(self) -> None:
        if self.cur_xyz[2] < self.z_safe - 1e-6:
            self._emit(
                C.SegmentKind.FEED,
                (self.cur_xyz[0], self.cur_xyz[1], self.z_safe),
                self.retract_feed,
                "retract",
                stamp_ipw=False,
            )

    def _emit_link_chord(
        self,
        target_xy: Point2D,
        return_tag: str,
    ) -> None:
        cur_xy = (self.cur_xyz[0], self.cur_xyz[1])
        seg_len = math.hypot(target_xy[0] - cur_xy[0], target_xy[1] - cur_xy[1])
        if seg_len <= 1e-3:
            return
        eff_ae = self.ipw.sweep_segment(cur_xy, target_xy, self.r, mutate=False) / max(seg_len, 1e-6)
        ret_ae_cap = min(0.08, 0.12 * self.ae_target)
        if eff_ae <= ret_ae_cap:
            self._emit(C.SegmentKind.FEED, (target_xy[0], target_xy[1], self.cur_pass_z), self.return_feed, return_tag)
        else:
            feed = (
                self.cut_feed
                if eff_ae <= 0.72 * self.d
                else max(0.35 * self.cut_feed, self.cut_feed * (0.60 * self.d / max(eff_ae, 1e-6)))
            )
            self._emit(C.SegmentKind.FEED, (target_xy[0], target_xy[1], self.cur_pass_z), feed, "step_link")

    def helical_bore_entry(
        self,
        center: Point2D,
        r_bore: float,
        r_open: float,
        margin: float,
        *,
        z_target: Optional[float] = None,
        stay_down_from_z: Optional[float] = None,
        reset_ipw_to: Optional[np.ndarray] = None,
    ) -> None:
        """Enter at ``center`` via G1 discretized helical spiral bore + radial expansion."""
        z_cut = self.pz0 if z_target is None else float(z_target)
        cx, cy = center
        eff_bore = max(0.0, r_bore)
        start_xy = (cx + eff_bore, cy)

        can_stay_down = (
            stay_down_from_z is not None
            and abs(self.cur_xyz[2] - stay_down_from_z) <= 1e-4
        )
        if can_stay_down:
            assert stay_down_from_z is not None
            self.cur_pass_z = float(stay_down_from_z)
            self.link_to(start_xy, margin=margin, allow_cut_step=False, return_tag="peel_ret")
            if abs(self.cur_xyz[2] - stay_down_from_z) <= 1e-4 and math.hypot(
                self.cur_xyz[0] - start_xy[0], self.cur_xyz[1] - start_xy[1]
            ) <= 2e-3:
                z_top = float(stay_down_from_z)
            else:
                can_stay_down = False

        if not can_stay_down:
            self.retract_to_safe()
            self._emit(C.SegmentKind.RAPID, (start_xy[0], start_xy[1], self.z_safe), None, "approach", stamp_ipw=False)
            z_top = min(self.z_safe, self.stock_max_z + 0.5)
            if z_top < self.z_safe - 1e-4:
                self._emit(C.SegmentKind.FEED, (start_xy[0], start_xy[1], z_top), self.retract_feed, "approach_z", stamp_ipw=False)

        if reset_ipw_to is not None:
            self.ipw.remaining[:, :] = reset_ipw_to
        self.cur_pass_z = z_cut

        c_step = self.config.chord_step_mm
        dz = max(0.0, z_top - z_cut)
        if eff_bore >= 0.04 and dz > 1e-5:
            z_pitch = min(1.40, max(0.75, 0.28 * self.d))
            n_turns = max(1.0, dz / z_pitch)
            min_bore_steps = 8 if c_step <= 0.80 else 5
            steps_per_turn = max(min_bore_steps, min(10, int(math.ceil(2.0 * math.pi * eff_bore / c_step))))
            total_steps = max(min_bore_steps, int(math.ceil(n_turns * steps_per_turn)))
            total_angle = 2.0 * math.pi * n_turns
            for k in range(1, total_steps + 1):
                frac = k / total_steps
                th = frac * total_angle
                zk = z_top - frac * dz
                pt = (cx + eff_bore * math.cos(th), cy + eff_bore * math.sin(th), zk)
                self._emit(C.SegmentKind.FEED, pt, self.plunge_feed, "helix_bore")

            # One flat cleanup turn at z_cut so the entry bore floor is completely flat
            th0 = total_angle
            for k in range(1, steps_per_turn + 1):
                th = th0 + 2.0 * math.pi * (k / steps_per_turn)
                pt = (cx + eff_bore * math.cos(th), cy + eff_bore * math.sin(th), z_cut)
                self._emit(C.SegmentKind.FEED, pt, self.cut_feed, "helix_flat")
            cur_angle = th0 + 2.0 * math.pi
        else:
            self._emit(C.SegmentKind.FEED, (cx, cy, z_cut), self.plunge_feed, "helix_bore")
            cur_angle = 0.0

        # Archimedean radial spiral pocket opening from eff_bore out to r_open
        if r_open > eff_bore + 0.04:
            radial_pitch = max(0.30, 0.85 * self.ae_target)
            exp_turns = max(0.5, (r_open - eff_bore) / radial_pitch)
            min_open_steps = 8 if c_step <= 0.80 else 6
            steps_per_turn = max(min_open_steps, min(12, int(math.ceil(2.0 * math.pi * r_open / (1.13 * c_step)))))
            exp_steps = max(min_open_steps, int(math.ceil(exp_turns * steps_per_turn)))
            exp_angle = 2.0 * math.pi * exp_turns
            for k in range(1, exp_steps + 1):
                frac = k / exp_steps
                rk = eff_bore + frac * (r_open - eff_bore)
                th = cur_angle + frac * exp_angle
                cand_xy = (cx + rk * math.cos(th), cy + rk * math.sin(th))
                if _wall_distance_point(cand_xy, self.geom, self.r) >= self.r + margin - 1e-5:
                    self._emit(C.SegmentKind.FEED, (cand_xy[0], cand_xy[1], z_cut), self.cut_feed, "helix_expand")
            cur_angle += exp_angle
            # Full cleanup circle at r_open
            for k in range(1, steps_per_turn + 1):
                th = cur_angle + 2.0 * math.pi * (k / steps_per_turn)
                cand_xy = (cx + r_open * math.cos(th), cy + r_open * math.sin(th))
                if _wall_distance_point(cand_xy, self.geom, self.r) >= self.r + margin - 1e-5:
                    self._emit(C.SegmentKind.FEED, (cand_xy[0], cand_xy[1], z_cut), self.cut_feed, "helix_expand")

        self.last_peel_pts = None

    def _find_cleared_plunge_lead(self, target_xy: Point2D, margin: float) -> Point2D:
        """Find a nearby already-cleared safe point behind ``target_xy`` for zero-load air plunge."""
        if self.ipw.sweep_segment(target_xy, target_xy, self.r, mutate=False) <= 0.02 * self.d:
            return target_xy
        search_r = min(0.85 * self.ae_target, 1.5)
        best_pt = target_xy
        best_area = self.ipw.sweep_segment(target_xy, target_xy, self.r, mutate=False)
        for rad in (0.45 * search_r, 0.85 * search_r, 1.25 * search_r):
            for k in range(12):
                th = 2.0 * math.pi * k / 12.0
                cand = (target_xy[0] + rad * math.cos(th), target_xy[1] + rad * math.sin(th))
                if not _is_segment_safe(cand, target_xy, self.geom, self.r, margin=margin):
                    continue
                area = self.ipw.sweep_segment(cand, cand, self.r, mutate=False)
                if area < best_area - 1e-6:
                    best_area = area
                    best_pt = cand
                    if best_area <= 1e-6:
                        return best_pt
        return best_pt

    def _route_stay_down_ipw(
        self,
        cur_xy: Point2D,
        target_xy: Point2D,
        link_margin: float,
        allow_cut_step: bool,
    ) -> Optional[List[Point2D]]:
        """Route a wall-safe stay-down polyline across cleared 2D IPW around islands/corners."""
        if self.d_wall_grid is None:
            return None
        xs = self.ipw.xs
        ys = self.ipw.ys
        pitch = self.ipw.pitch
        nx, ny = self.d_wall_grid.shape
        if nx < 3 or ny < 3:
            return None

        walk_mask = self.d_wall_grid >= (self.r + link_margin)

        def _snap_valid(pt: Point2D, from_pt: bool) -> Optional[Tuple[int, int]]:
            ci = min(max(int(round((pt[0] - xs[0]) / pitch)), 0), nx - 1)
            cj = min(max(int(round((pt[1] - ys[0]) / pitch)), 0), ny - 1)
            best_ij: Optional[Tuple[int, int]] = None
            best_d2 = float("inf")
            for di in range(-2, 3):
                ii = ci + di
                if ii < 0 or ii >= nx:
                    continue
                for dj in range(-2, 3):
                    jj = cj + dj
                    if jj < 0 or jj >= ny:
                        continue
                    if not walk_mask[ii, jj] and self.d_wall_grid[ii, jj] < self.r + link_margin - 1e-5:
                        continue
                    g_pt = (float(xs[ii]), float(ys[jj]))
                    seg_ok = (
                        _is_segment_safe(pt, g_pt, self.geom, self.r, margin=link_margin)
                        if from_pt
                        else _is_segment_safe(g_pt, pt, self.geom, self.r, margin=link_margin)
                    )
                    if not seg_ok:
                        continue
                    d2 = (g_pt[0] - pt[0]) ** 2 + (g_pt[1] - pt[1]) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_ij = (ii, jj)
            return best_ij

        s_ij = _snap_valid(cur_xy, from_pt=True)
        g_ij = _snap_valid(target_xy, from_pt=False)
        if s_ij is None or g_ij is None:
            return None

        dist_uncut = ndimage.distance_transform_edt(~self.ipw.remaining) * pitch
        cleared = dist_uncut >= max(0.0, self.r - 0.85 * pitch)
        near_start = ((self.ipw.gx - cur_xy[0]) ** 2 + (self.ipw.gy - cur_xy[1]) ** 2) <= (
            max(1.2 * pitch, 0.35 * self.r) ** 2
        )
        step_r = max(2.25 * self.ae_target, 1.5 * pitch) if allow_cut_step else max(1.2 * pitch, 0.25 * self.ae_target)
        near_goal = ((self.ipw.gx - target_xy[0]) ** 2 + (self.ipw.gy - target_xy[1]) ** 2) <= (step_r * step_r)

        base_safe = walk_mask.copy()
        base_safe[s_ij[0], s_ij[1]] = True
        base_safe[g_ij[0], g_ij[1]] = True
        passable = base_safe & (cleared | near_start | near_goal)

        labeled_p, _ = ndimage.label(passable, structure=_STRUCT_8)
        comp_id = int(labeled_p[s_ij[0], s_ij[1]])
        if comp_id == 0 or int(labeled_p[g_ij[0], g_ij[1]]) != comp_id:
            return None

        in_comp = labeled_p == comp_id
        gi, gj = g_ij
        si, sj = s_ij

        if (si, sj) == (gi, gj):
            raw_pts = [cur_xy, (float(xs[si]), float(ys[sj])), target_xy]
        else:
            g_cost = np.full((nx, ny), np.inf, dtype=np.float64)
            parent_i = np.full((nx, ny), -1, dtype=np.int32)
            parent_j = np.full((nx, ny), -1, dtype=np.int32)
            g_cost[si, sj] = 0.0

            diag = math.sqrt(2.0) * pitch
            h0 = pitch * (abs(gi - si) + abs(gj - sj) + (math.sqrt(2.0) - 2.0) * min(abs(gi - si), abs(gj - sj)))
            heap: List[Tuple[float, float, int, int]] = [(h0, 0.0, si, sj)]
            neighbors = (
                (-1, 0, pitch),
                (1, 0, pitch),
                (0, -1, pitch),
                (0, 1, pitch),
                (-1, -1, diag),
                (-1, 1, diag),
                (1, -1, diag),
                (1, 1, diag),
            )
            found = False
            while heap:
                _, cur_g, ci, cj = heapq.heappop(heap)
                if cur_g > g_cost[ci, cj] + 1e-12:
                    continue
                if ci == gi and cj == gj:
                    found = True
                    break
                for di, dj, step_d in neighbors:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nx and 0 <= nj < ny and in_comp[ni, nj]:
                        # Prevent diagonal corner cutting across wall pixels
                        if di != 0 and dj != 0 and not (in_comp[ci + di, cj] and in_comp[ci, cj + dj]):
                            continue
                        w_mult = 1.0 if cleared[ni, nj] else 7.5
                        cand_g = cur_g + step_d * w_mult
                        if cand_g + 1e-12 < g_cost[ni, nj]:
                            g_cost[ni, nj] = cand_g
                            parent_i[ni, nj] = ci
                            parent_j[ni, nj] = cj
                            dx_g = abs(gi - ni)
                            dy_g = abs(gj - nj)
                            h_val = 1.001 * pitch * (dx_g + dy_g + (math.sqrt(2.0) - 2.0) * min(dx_g, dy_g))
                            heapq.heappush(heap, (cand_g + h_val, cand_g, ni, nj))
            if not found:
                return None

            grid_chain: List[Point2D] = []
            ci, cj = gi, gj
            while ci >= 0 and cj >= 0:
                grid_chain.append((float(xs[ci]), float(ys[cj])))
                if ci == si and cj == sj:
                    break
                pi, pj = int(parent_i[ci, cj]), int(parent_j[ci, cj])
                ci, cj = pi, pj
            grid_chain.reverse()
            raw_pts = [cur_xy] + grid_chain + [target_xy]

        # Deduplicate near-identical consecutive points
        dedup: List[Point2D] = [raw_pts[0]]
        for p in raw_pts[1:]:
            if math.hypot(p[0] - dedup[-1][0], p[1] - dedup[-1][1]) > 1e-4:
                dedup.append(p)
        if len(dedup) < 2:
            return None

        ret_ae_cap = min(0.08, 0.12 * self.ae_target)
        simplified: List[Point2D] = [dedup[0]]
        idx = 0
        n_pts = len(dedup)
        engaged_len = 0.0
        while idx < n_pts - 1:
            chosen: Optional[int] = None
            for probe in range(n_pts - 1, idx, -1):
                seg_len = math.hypot(dedup[probe][0] - dedup[idx][0], dedup[probe][1] - dedup[idx][1])
                if seg_len <= 1e-4:
                    chosen = probe
                    break
                if not _is_segment_safe(dedup[idx], dedup[probe], self.geom, self.r, margin=link_margin):
                    continue
                eff_ae = self.ipw.sweep_segment(dedup[idx], dedup[probe], self.r, mutate=False) / seg_len
                if eff_ae <= ret_ae_cap:
                    chosen = probe
                    break
                if (
                    allow_cut_step
                    and probe == n_pts - 1
                    and seg_len <= 2.35 * self.ae_target
                    and eff_ae <= 1.08 * self.d
                ):
                    chosen = probe
                    break
            if chosen is None:
                nxt = idx + 1
                seg_len = math.hypot(dedup[nxt][0] - dedup[idx][0], dedup[nxt][1] - dedup[idx][1])
                rem_goal = math.hypot(dedup[-1][0] - dedup[idx][0], dedup[-1][1] - dedup[idx][1])
                if (
                    allow_cut_step
                    and rem_goal <= 2.5 * self.ae_target
                    and engaged_len + seg_len <= 2.6 * self.ae_target
                    and _is_segment_safe(dedup[idx], dedup[nxt], self.geom, self.r, margin=link_margin)
                ):
                    eff_ae = self.ipw.sweep_segment(dedup[idx], dedup[nxt], self.r, mutate=False) / max(seg_len, 1e-6)
                    if eff_ae <= 1.08 * self.d:
                        chosen = nxt
                        engaged_len += seg_len
            if chosen is None:
                return None
            simplified.append(dedup[chosen])
            idx = chosen

        if _polyline_length(simplified) > 35.0 * self.d:
            return None
        return simplified

    def link_to(
        self,
        target_xy: Point2D,
        margin: float,
        *,
        allow_cut_step: bool = True,
        return_tag: str = "peel_ret",
    ) -> None:
        """Link safely from current position to ``target_xy`` at ``cur_pass_z``."""
        cur_xy = (self.cur_xyz[0], self.cur_xyz[1])
        dist = math.hypot(target_xy[0] - cur_xy[0], target_xy[1] - cur_xy[1])

        if self.cur_xyz[2] > self.cur_pass_z + 1e-5:
            # Currently retracted at z_safe: rapid above pre-lead point, plunge, feed in
            lead_xy = self._find_cleared_plunge_lead(target_xy, margin)
            self._emit(C.SegmentKind.RAPID, (lead_xy[0], lead_xy[1], self.z_safe), None, "hop_over", stamp_ipw=False)
            self._emit(C.SegmentKind.FEED, (lead_xy[0], lead_xy[1], self.cur_pass_z), self.plunge_feed, "hop_plunge")
            if math.hypot(target_xy[0] - lead_xy[0], target_xy[1] - lead_xy[1]) > 1e-3:
                self._emit(C.SegmentKind.FEED, (target_xy[0], target_xy[1], self.cur_pass_z), self.cut_feed, "lead_in")
            return

        if dist <= 1e-3:
            return

        link_margin = max(1e-4, 0.85 * margin) if margin > 1e-4 else 0.0
        ret_ae_cap = min(0.08, 0.12 * self.ae_target)

        # Case 1: Direct straight chord is safe
        if dist <= 12.0 * self.d and _is_segment_safe(cur_xy, target_xy, self.geom, self.r, margin=link_margin):
            swept = self.ipw.sweep_segment(cur_xy, target_xy, self.r, mutate=False)
            eff_ae = swept / max(dist, 1e-6)
            if eff_ae <= ret_ae_cap:
                self._emit(C.SegmentKind.FEED, (target_xy[0], target_xy[1], self.cur_pass_z), self.return_feed, return_tag)
                return
            if allow_cut_step and dist <= 2.25 * self.ae_target and eff_ae <= 0.75 * self.d:
                self._emit(C.SegmentKind.FEED, (target_xy[0], target_xy[1], self.cur_pass_z), self.cut_feed, "step_link")
                return
            if allow_cut_step and dist <= 1.35 * self.ae_target and eff_ae <= 1.15 * self.d:
                # Narrow-throat forward stepover: stay down at proportionally moderated feedrate
                step_feed = max(0.35 * self.cut_feed, self.cut_feed * (0.60 * self.d / max(eff_ae, 1e-6)))
                self._emit(C.SegmentKind.FEED, (target_xy[0], target_xy[1], self.cur_pass_z), step_feed, "step_link")
                return

        # Case 2: Disengaged return to start of previous peel pass (100% cleared IPW) + short step-in
        if self.last_peel_pts is not None and len(self.last_peel_pts) >= 2:
            q_start = self.last_peel_pts[0]
            end_step = math.hypot(target_xy[0] - q_start[0], target_xy[1] - q_start[1])
            if (
                end_step <= 2.25 * self.ae_target
                and _is_segment_safe(q_start, target_xy, self.geom, self.r, margin=link_margin)
            ):
                end_ae = self.ipw.sweep_segment(q_start, target_xy, self.r, mutate=False) / max(end_step, 1e-6)
                if end_ae <= 0.78 * self.d:
                    # Try 1-chord oblique return through cleared IPW directly to q_start first
                    ret_dist = math.hypot(q_start[0] - cur_xy[0], q_start[1] - cur_xy[1])
                    if (
                        ret_dist <= 12.0 * self.d
                        and _is_segment_safe(cur_xy, q_start, self.geom, self.r, margin=link_margin)
                        and self.ipw.sweep_segment(cur_xy, q_start, self.r, mutate=False) / max(ret_dist, 1e-6) <= ret_ae_cap
                    ):
                        self._emit(C.SegmentKind.FEED, (q_start[0], q_start[1], self.cur_pass_z), self.return_feed, return_tag)
                        if end_step > 1e-3:
                            self._emit_link_chord(target_xy, return_tag)
                        return

                    # Otherwise route back along simplified reversed peel corridor
                    rev = list(reversed(self.last_peel_pts))
                    if math.hypot(rev[0][0] - cur_xy[0], rev[0][1] - cur_xy[1]) <= 0.5 * self.d:
                        back_pts: List[Point2D] = [cur_xy] + rev
                        simplified = [back_pts[0]]
                        idx = 0
                        ok_route = True
                        while idx < len(back_pts) - 1:
                            jumped = idx + 1
                            for probe in range(len(back_pts) - 1, idx + 1, -1):
                                seg_len = math.hypot(back_pts[probe][0] - back_pts[idx][0], back_pts[probe][1] - back_pts[idx][1])
                                if seg_len > 1e-4 and _is_segment_safe(back_pts[idx], back_pts[probe], self.geom, self.r, margin=link_margin):
                                    if self.ipw.sweep_segment(back_pts[idx], back_pts[probe], self.r, mutate=False) / seg_len <= ret_ae_cap:
                                        jumped = probe
                                        break
                            if not _is_segment_safe(back_pts[idx], back_pts[jumped], self.geom, self.r, margin=link_margin):
                                ok_route = False
                                break
                            simplified.append(back_pts[jumped])
                            idx = jumped

                        if ok_route and _polyline_length(simplified) <= 15.0 * self.d:
                            for pt in simplified[1:]:
                                self._emit_link_chord(pt, return_tag)
                            if end_step > 1e-3:
                                self._emit_link_chord(target_xy, return_tag)
                            return

        # Case 2.5: 2D A* stay-down route across cleared IPW around standing islands / corners
        routed = self._route_stay_down_ipw(cur_xy, target_xy, link_margin, allow_cut_step)
        if routed is not None and len(routed) >= 2:
            for pt in routed[1:]:
                self._emit_link_chord(pt, return_tag)
            return

        # Case 3: Safe z_safe retract + rapid + zero-load plunge into cleared floor (disconnected chambers only)
        self.retract_to_safe()
        lead_xy = self._find_cleared_plunge_lead(target_xy, margin)
        self._emit(C.SegmentKind.RAPID, (lead_xy[0], lead_xy[1], self.z_safe), None, "hop_over", stamp_ipw=False)
        self._emit(C.SegmentKind.FEED, (lead_xy[0], lead_xy[1], self.cur_pass_z), self.plunge_feed, "hop_plunge")
        if math.hypot(target_xy[0] - lead_xy[0], target_xy[1] - lead_xy[1]) > 1e-3:
            self._emit(C.SegmentKind.FEED, (target_xy[0], target_xy[1], self.cur_pass_z), self.cut_feed, "lead_in")

    def execute_trochoid_loop(
        self,
        center: Point2D,
        tangent: Point2D,
        loop_radius: float,
        margin: float,
    ) -> None:
        """Emit one circular trochoidal loop: CCW cutting semicircle + disengaged return arc."""
        tx, ty = tangent
        nx, ny = -ty, tx
        cx, cy = center
        rk = max(0.04, loop_radius)

        min_cut = 4 if self.config.chord_step_mm <= 0.80 else 3
        n_cut = max(min_cut, min(7, int(math.ceil(math.pi * rk / (0.87 * self.config.chord_step_mm)))))
        cut_pts: List[Point2D] = []
        for k in range(n_cut + 1):
            phi = -0.5 * math.pi + math.pi * (k / n_cut)
            cp, sp = math.cos(phi), math.sin(phi)
            px = cx + rk * (cp * tx + sp * nx)
            py = cy + rk * (cp * ty + sp * ny)
            cut_pts.append((px, py))

        # Check if this loop actually removes any remaining IPW
        if self.ipw.sweep_polyline(cut_pts, self.r, mutate=False) <= 1e-6:
            return

        self.link_to(cut_pts[0], margin, allow_cut_step=True, return_tag="trochoid_ret")
        for pt in cut_pts[1:]:
            self._emit(C.SegmentKind.FEED, (pt[0], pt[1], self.cur_pass_z), self.cut_feed, "trochoid_cut")

        # Disengaged return inside closed disk B_rk(center) (100% cleared & wall-safe)
        if rk >= 0.70:
            mid_ret = (cx - 0.25 * rk * tx, cy - 0.25 * rk * ty)
            self._emit(C.SegmentKind.FEED, (mid_ret[0], mid_ret[1], self.cur_pass_z), self.return_feed, "trochoid_ret")
        self._emit(C.SegmentKind.FEED, (cut_pts[0][0], cut_pts[0][1], self.cur_pass_z), self.return_feed, "trochoid_ret")
        self.last_peel_pts = None

    def execute_peel_contour(
        self,
        pts: List[Point2D],
        is_closed: bool,
        margin: float,
    ) -> None:
        """Emit a constant-stepover engagement-bounded offset peel pass."""
        if len(pts) < 2:
            return
        if self.ipw.sweep_polyline(pts, self.r, mutate=False) <= 1e-6:
            return

        if is_closed and len(pts) >= 4:
            cur_xy = (self.cur_xyz[0], self.cur_xyz[1])
            ring = pts[:-1]
            best_i = min(
                range(len(ring)),
                key=lambda i: math.hypot(ring[i][0] - cur_xy[0], ring[i][1] - cur_xy[1]),
            )
            ordered = ring[best_i:] + ring[:best_i] + [ring[best_i]]
            self.link_to(ordered[0], margin, allow_cut_step=True, return_tag="peel_ret")
            for pt in ordered[1:]:
                self._emit(C.SegmentKind.FEED, (pt[0], pt[1], self.cur_pass_z), self.cut_feed, "peel_cut")
            self.last_peel_pts = None
        else:
            self.link_to(pts[0], margin, allow_cut_step=True, return_tag="peel_ret")
            for pt in pts[1:]:
                self._emit(C.SegmentKind.FEED, (pt[0], pt[1], self.cur_pass_z), self.cut_feed, "peel_cut")
            self.last_peel_pts = list(pts)


# ---------------------------------------------------------------------------
# Stage 2 & Stage 3 Orchestration
# ---------------------------------------------------------------------------
def _extract_component_contours(
    comp_mask: np.ndarray,
    seed_mask: np.ndarray,
    d_wall_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    pitch: float,
    geom: _PocketGeometry,
    radius: float,
    finish_stock: float,
    config: AdaptiveConfig,
) -> List[_WaveContour]:
    d_tool = 2.0 * radius
    ae_target = config.optimal_load_fraction * d_tool
    a_step = 0.88 * ae_target
    r_rough = radius + finish_stock

    c_grid = np.where(comp_mask, np.maximum(0.0, d_wall_grid - r_rough), 0.0)
    win_px = max(3, 2 * int(round(1.0 * radius / pitch)) + 1)
    chan_c_grid = ndimage.maximum_filter(c_grid, size=win_px, mode="constant", cval=0.0)

    # Trochoid pitch law derived from area conservation: 2*(R + c)*p <= ae*(pi*c + p)
    p_cap = config.trochoid_pitch_fraction * d_tool
    p_floor = max(0.38, 0.10 * d_tool)
    ae_eff = 0.92 * ae_target
    denom = np.maximum(0.2 * d_tool, 2.0 * (radius + chan_c_grid) - ae_eff)
    p_troch = np.clip((math.pi * chan_c_grid * ae_eff) / denom, p_floor, p_cap)

    is_narrow = chan_c_grid <= 0.48 * d_tool
    weights = np.where(is_narrow, np.maximum(1.0, a_step / p_troch), 1.0)

    g_field = _compute_weighted_geodesic(comp_mask, seed_mask, weights, pitch)
    finite_vals = g_field[comp_mask & np.isfinite(g_field)]
    if finite_vals.size == 0:
        return []
    g_max = float(finite_vals.max())
    if g_max <= 0.12 * a_step:
        return []

    # Neumann extension outside comp_mask via nearest-neighbor EDT indices so iso-contours
    # terminate orthogonally on dF instead of wrapping around already-cleared boundaries.
    valid = comp_mask & np.isfinite(g_field)
    nearest_idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    g_ext = g_field[nearest_idx[0], nearest_idx[1]]

    n_levels = max(1, int(math.ceil(g_max / a_step)))
    levels = [m * a_step for m in range(1, n_levels + 1)]
    if g_max - levels[-1] > 0.2 * a_step or levels[-1] > g_max:
        levels[-1] = max(0.25 * a_step, g_max - 0.08 * a_step)

    contours: List[_WaveContour] = []
    nx, ny = comp_mask.shape

    for lvl_idx, lvl in enumerate(levels, start=1):
        raw_polys = _marching_squares_oriented(g_ext, xs, ys, lvl)
        for raw_pts, raw_closed in raw_polys:
            clipped_list = _clip_polyline_to_domain(raw_pts, raw_closed, geom, radius, margin=finish_stock)
            for c_pts, c_closed in clipped_list:
                simp = _smooth_and_simplify_polyline(
                    c_pts, c_closed, geom, radius, margin=finish_stock, tol_mm=config.simplify_tol_mm
                )
                plen = _polyline_length(simp)
                if plen < 0.08:
                    continue
                # Evaluate peak wall clearance along simplified contour
                sample_pts: List[Point2D] = []
                for k in range(len(simp) - 1):
                    seg_d = math.hypot(simp[k + 1][0] - simp[k][0], simp[k + 1][1] - simp[k][1])
                    n_s = max(1, int(math.ceil(seg_d / 0.25)))
                    for s_i in range(n_s):
                        u = s_i / n_s
                        sample_pts.append(
                            (
                                simp[k][0] + u * (simp[k + 1][0] - simp[k][0]),
                                simp[k][1] + u * (simp[k + 1][1] - simp[k][1]),
                            )
                        )
                sample_pts.append(simp[-1])
                peak_pt = max(sample_pts, key=lambda q: _wall_distance_point(q, geom, radius))
                peak_c = max(0.0, _wall_distance_point(peak_pt, geom, radius) - r_rough)
                ix = min(max(int(round((peak_pt[0] - xs[0]) / pitch)), 0), nx - 1)
                iy = min(max(int(round((peak_pt[1] - ys[0]) / pitch)), 0), ny - 1)
                chan_c = float(chan_c_grid[ix, iy])

                contours.append(
                    _WaveContour(
                        level_idx=lvl_idx,
                        points=simp,
                        is_closed=c_closed,
                        length_mm=plen,
                        peak_pt=peak_pt,
                        peak_clearance_mm=peak_c,
                        channel_clearance_mm=chan_c,
                    )
                )
    return contours


def _schedule_and_execute_contours(
    contours: List[_WaveContour],
    builder: _AdaptivePathBuilder,
    geom: _PocketGeometry,
    finish_stock: float,
) -> None:
    """Execute wavefront contours in prerequisite-safe nearest-continuation order."""
    if not contours:
        return
    d_tool = 2.0 * builder.r
    a_step = 0.88 * builder.ae_target
    n_c = len(contours)

    by_level: Dict[int, List[int]] = {}
    for idx, wc in enumerate(contours):
        by_level.setdefault(wc.level_idx, []).append(idx)

    parents: List[List[int]] = [[] for _ in range(n_c)]
    children: List[List[int]] = [[] for _ in range(n_c)]
    sorted_levels = sorted(by_level.keys())

    sched_margin = max(1e-4, 0.85 * finish_stock) if finish_stock > 1e-4 else 0.0

    def _contours_coupled(wc_a: _WaveContour, wc_b: _WaveContour, max_d: float) -> bool:
        dist, pa, pb = _min_polyline_dist(wc_a.points, wc_b.points)
        if dist > max_d:
            return False
        cand_pairs = [(pa, pb), (wc_a.points[0], wc_b.points[0]), (wc_a.points[-1], wc_b.points[-1])]
        for qa, qb in cand_pairs:
            if math.hypot(qb[0] - qa[0], qb[1] - qa[1]) <= max_d and _is_segment_safe(
                qa, qb, geom, builder.r, margin=0.0
            ):
                return True
        return False

    for l_pos in range(1, len(sorted_levels)):
        prev_lvl = sorted_levels[l_pos - 1]
        cur_lvl = sorted_levels[l_pos]
        for c_idx in by_level[cur_lvl]:
            wc_cur = contours[c_idx]
            for p_idx in by_level[prev_lvl]:
                if _contours_coupled(contours[p_idx], wc_cur, 2.35 * a_step):
                    parents[c_idx].append(p_idx)
                    children[p_idx].append(c_idx)
            if not parents[c_idx] and l_pos >= 2:
                prev2_lvl = sorted_levels[l_pos - 2]
                for p_idx in by_level[prev2_lvl]:
                    if _contours_coupled(contours[p_idx], wc_cur, 3.40 * a_step):
                        parents[c_idx].append(p_idx)
                        children[p_idx].append(c_idx)

    rem_deps = [len(p) for p in parents]
    done = [False] * n_c
    ret_ae_cap = min(0.08, 0.12 * builder.ae_target)

    def _entry_point(wc: _WaveContour) -> Point2D:
        if not wc.is_closed and wc.length_mm <= 1.25 * d_tool and wc.channel_clearance_mm <= 0.48 * d_tool:
            return wc.peak_pt
        if wc.is_closed:
            cur_xy = (builder.cur_xyz[0], builder.cur_xyz[1])
            return min(wc.points, key=lambda q: math.hypot(q[0] - cur_xy[0], q[1] - cur_xy[1]))
        return wc.points[0]

    remaining = n_c
    while remaining > 0:
        ready = [i for i in range(n_c) if not done[i] and rem_deps[i] == 0]
        if not ready:
            # Fallback: break any rare tie by lowest level_idx
            min_lvl = min(contours[i].level_idx for i in range(n_c) if not done[i])
            ready = [i for i in range(n_c) if not done[i] and contours[i].level_idx == min_lvl]

        cur_xy = (builder.cur_xyz[0], builder.cur_xyz[1])
        peel_origin = builder.last_peel_pts[0] if builder.last_peel_pts else None

        def _score(idx: int) -> Tuple[int, float, int]:
            ep = _entry_point(contours[idx])
            d_cur = math.hypot(ep[0] - cur_xy[0], ep[1] - cur_xy[1])
            wall_ok = d_cur <= 6.5 * d_tool and _is_segment_safe(cur_xy, ep, geom, builder.r, margin=sched_margin)
            can_cur = wall_ok and (
                d_cur <= 2.35 * a_step
                or (builder.ipw.sweep_segment(cur_xy, ep, builder.r, mutate=False) / max(d_cur, 1e-6)) <= ret_ae_cap
            )
            if peel_origin is not None:
                d_orig = math.hypot(ep[0] - peel_origin[0], ep[1] - peel_origin[1])
                if d_orig <= 2.2 * a_step and _is_segment_safe(peel_origin, ep, geom, builder.r, margin=sched_margin):
                    return (0, d_orig, contours[idx].level_idx)
            same_branch = 0 if can_cur else 1
            return (same_branch, d_cur, contours[idx].level_idx)

        pick = min(ready, key=_score)
        done[pick] = True
        remaining -= 1
        for ch in children[pick]:
            rem_deps[ch] = max(0, rem_deps[ch] - 1)

        wc = contours[pick]
        use_trochoid = (
            not wc.is_closed
            and wc.length_mm <= 1.25 * d_tool
            and wc.channel_clearance_mm <= 0.48 * d_tool
            and wc.peak_clearance_mm >= 0.04
        )
        if use_trochoid:
            p0, p1 = wc.points[0], wc.points[-1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            chord = math.hypot(dx, dy)
            if chord > 1e-6:
                # Contour has uncut stock on RIGHT -> forward tangent is right-hand normal (dy, -dx)
                tangent = (dy / chord, -dx / chord)
            else:
                tangent = (1.0, 0.0)
            rk = min(wc.peak_clearance_mm, 0.55 * d_tool)
            builder.execute_trochoid_loop(wc.peak_pt, tangent, rk, margin=finish_stock)
        else:
            builder.execute_peel_contour(wc.points, wc.is_closed, margin=finish_stock)


def _build_analytical_finishing_chains(
    geom: _PocketGeometry,
    radius: float,
) -> List[List[Point2D]]:
    """Build exact analytical wall-finishing polylines for outer pocket and all standing islands."""
    chains: List[List[Point2D]] = []
    ox0, ox1, oy0, oy1 = _outer_cutter_bounds(geom, radius, margin=0.0)

    def _append_clipped_edges(edges: Sequence[Tuple[Point2D, Point2D]]) -> None:
        cur: List[Point2D] = []
        for p0, p1 in edges:
            segs = _clip_segment_safe(p0, p1, geom, radius, margin=0.0)
            if not segs:
                if len(cur) >= 2:
                    chains.append(cur)
                cur = []
                continue
            for qa, qb in segs:
                if not cur:
                    cur = [qa, qb]
                elif math.hypot(qa[0] - cur[-1][0], qa[1] - cur[-1][1]) <= 1e-3:
                    cur.append(qb)
                else:
                    if len(cur) >= 2:
                        chains.append(cur)
                    cur = [qa, qb]
        if len(cur) >= 2:
            chains.append(cur)

    # 1. Outer pocket boundary (CCW climb milling along outer walls)
    if ox1 >= ox0 - 1e-9 and oy1 >= oy0 - 1e-9:
        if ox1 > ox0 + 1e-4 and oy1 > oy0 + 1e-4:
            outer_edges = [
                ((ox0, oy0), (ox1, oy0)),
                ((ox1, oy0), (ox1, oy1)),
                ((ox1, oy1), (ox0, oy1)),
                ((ox0, oy1), (ox0, oy0)),
            ]
            _append_clipped_edges(outer_edges)
        elif ox1 > ox0 + 1e-4:
            _append_clipped_edges([((ox0, 0.5 * (oy0 + oy1)), (ox1, 0.5 * (oy0 + oy1)))])
        elif oy1 > oy0 + 1e-4:
            _append_clipped_edges([((0.5 * (ox0 + ox1), oy0), (0.5 * (ox0 + ox1), oy1))])

    # 2. Every standing island [ix0 - R, ix1 + R] x [iy0 - R, iy1 + R] (CW climb milling around island)
    for (ix0, iy0), (ix1, iy1) in geom.islands_2d:
        lx0 = _safe_lo(ix0 - radius)
        lx1 = _safe_hi(ix1 + radius)
        ly0 = _safe_lo(iy0 - radius)
        ly1 = _safe_hi(iy1 + radius)
        isl_edges = [
            ((lx0, ly0), (lx0, ly1)),
            ((lx0, ly1), (lx1, ly1)),
            ((lx1, ly1), (lx1, ly0)),
            ((lx1, ly0), (lx0, ly0)),
        ]
        _append_clipped_edges(isl_edges)

    return chains


def _sweep_residual_ipw(
    builder: _AdaptivePathBuilder,
    safe_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    margin: float,
    exclude_mask: Optional[np.ndarray] = None,
) -> None:
    """Verify 2D IPW and sweep clean any remaining uncut pockets reachable by ``safe_mask``."""
    r_check = max(0.2 * builder.r, builder.r - 0.55 * builder.ipw.pitch)

    for _ in range(64):
        target_rem = builder.ipw.remaining if exclude_mask is None else (builder.ipw.remaining & ~exclude_mask)
        if not np.any(target_rem):
            break

        # Find safe cutter centers within r_check of any remaining uncut target pixel
        dist_to_rem = ndimage.distance_transform_edt(~target_rem) * builder.ipw.pitch
        candidates = safe_mask & (dist_to_rem <= r_check + 1e-6)
        if not np.any(candidates):
            break

        labeled, n_comp = ndimage.label(candidates)
        swept_any = False

        # Order residual components nearest-first to current cutter position to keep links short & stay-down
        comp_strokes: List[List[Tuple[Point2D, Point2D]]] = []
        for c_id in range(1, n_comp + 1):
            idxs = np.argwhere(labeled == c_id)
            if idxs.size == 0:
                continue
            unique_i = np.unique(idxs[:, 0])
            step_i = max(1, int(math.floor(0.65 * builder.ae_target / builder.ipw.pitch)))
            sel_i = sorted(set(list(unique_i[::step_i]) + [int(unique_i[-1])]))

            strokes: List[Tuple[Point2D, Point2D]] = []
            for rank, i_val in enumerate(sel_i):
                js = np.sort(idxs[idxs[:, 0] == i_val, 1])
                splits = np.where(np.diff(js) > 1)[0] + 1
                runs = np.split(js, splits)
                if rank % 2 == 1:
                    runs = list(reversed(runs))
                for run in runs:
                    pa = (float(xs[i_val]), float(ys[int(run[0])]))
                    pb = (float(xs[i_val]), float(ys[int(run[-1])]))
                    if rank % 2 == 1:
                        pa, pb = pb, pa
                    strokes.append((pa, pb))
            if strokes:
                comp_strokes.append(strokes)

        while comp_strokes:
            cur_xy = (builder.cur_xyz[0], builder.cur_xyz[1])
            best_k = min(
                range(len(comp_strokes)),
                key=lambda k: min(
                    math.hypot(comp_strokes[k][0][0][0] - cur_xy[0], comp_strokes[k][0][0][1] - cur_xy[1]),
                    math.hypot(comp_strokes[k][-1][1][0] - cur_xy[0], comp_strokes[k][-1][1][1] - cur_xy[1]),
                ),
            )
            strokes = comp_strokes.pop(best_k)
            if math.hypot(strokes[-1][1][0] - cur_xy[0], strokes[-1][1][1] - cur_xy[1]) + 1e-6 < math.hypot(
                strokes[0][0][0] - cur_xy[0], strokes[0][0][1] - cur_xy[1]
            ):
                strokes = [(pb, pa) for pa, pb in reversed(strokes)]

            for pa, pb in strokes:
                if builder.ipw.sweep_segment(pa, pb, builder.r, mutate=False) <= 1e-6:
                    continue
                builder.link_to(pa, margin, allow_cut_step=True, return_tag="cleanup_ret")
                if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-3:
                    builder._emit(C.SegmentKind.FEED, (pb[0], pb[1], builder.cur_pass_z), builder.cut_feed, "ipw_cleanup")
                else:
                    builder.ipw.sweep_segment(pa, pa, builder.r, mutate=True)
                swept_any = True

        if not swept_any:
            break


def _waypoints_to_motion_plan(
    task: C.TaskSpec,
    waypoints_work: Sequence[WaypointSpec],
    provenance: str,
) -> C.MotionPlan:
    to_machine = task.work_to_machine.apply
    cur_mach = task.initial_state.position_mm
    assert cur_mach is not None

    raw_segments: List[C.Segment] = []
    idx = 0
    for kind, target_work, feed_mm_s, tag in waypoints_work:
        nxt_mach = to_machine((float(target_work[0]), float(target_work[1]), float(target_work[2])))
        dist = math.dist(cur_mach, nxt_mach)
        if dist <= 1e-4:
            continue
        if kind == C.SegmentKind.RAPID:
            eff_feed = None
            dur = dist / float(task.limits.rapid_mm_s)
        else:
            if feed_mm_s is None or feed_mm_s <= 0.0:
                raise ValueError("FEED segment requires positive feed_mm_s")
            eff_feed = min(float(feed_mm_s), float(task.limits.max_feed_mm_s))
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
            raw_segments, task.limits.rapid_mm_s, task.limits.max_accel_mm_s2
        )
        segments = tuple(
            dataclasses.replace(seg, duration_s=dur)
            for seg, dur in zip(raw_segments, durations)
        )
    elif rule == C.TimingRule.FIXED_DT and raw_segments:
        dt = 0.01
        segments = tuple(
            dataclasses.replace(seg, duration_s=max(1, round(seg.duration_s / dt)) * dt)
            for seg in raw_segments
        )
        plan = C.MotionPlan(
            frame=C.Frame.MACHINE,
            initial_state=task.initial_state,
            segments=segments,
            timing_rule=rule,
            dt_s=dt,
            provenance=provenance,
        )
        plan.validate_for_task(task)
        return plan
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_adaptive_toolpath(
    task: C.TaskSpec,
    config: AdaptiveConfig = AdaptiveConfig(),
) -> C.MotionPlan:
    """Generate a 2.5D engagement-bounded adaptive/trochoidal clearing MotionPlan."""
    if not task.initial_state.known or task.initial_state.position_mm is None:
        raise ValueError("generate_adaptive_toolpath requires a known initial_state")
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        raise ValueError("generate_adaptive_toolpath requires a flat endmill")

    geom = _extract_pocket_geometry(task)
    r = float(task.tool.cutter_radius_mm)
    d = 2.0 * r
    px0, py0, pz0 = geom.pocket_min
    px1, py1, _ = geom.pocket_max
    bx0, by0, _ = geom.box_min
    bx1, by1, _ = geom.box_max
    stock_max_z = float(task.stock.params["max_mm"][2])

    inv = task.work_to_machine.inverse()
    init_work = inv.apply(task.initial_state.position_mm)
    z_safe = max(float(init_work[2]), stock_max_z + 2.0)

    pitch = float(config.ipw_pitch_mm)
    pad = r + 2.0 * pitch

    def _cavity_target_mask(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        in_stock = (gx >= bx0) & (gx <= bx1) & (gy >= by0) & (gy <= by1)
        in_pocket = (gx >= px0) & (gx <= px1) & (gy >= py0) & (gy <= py1)
        mask = in_stock & in_pocket
        for (ix0, iy0), (ix1, iy1) in geom.islands_2d:
            mask &= ~((gx >= ix0) & (gx <= ix1) & (gy >= iy0) & (gy <= iy1))
        # Exclude unreachable concave walled outer-pocket corner fillets of radius R
        for x_side, xw in ((0, geom.walled_x0), (1, geom.walled_x1)):
            if not xw:
                continue
            for y_side, yw in ((0, geom.walled_y0), (1, geom.walled_y1)):
                if not yw:
                    continue
                cx = (px0 + r) if x_side == 0 else (px1 - r)
                cy = (py0 + r) if y_side == 0 else (py1 - r)
                x_lo = px0 if x_side == 0 else (px1 - r)
                x_hi = (px0 + r) if x_side == 0 else px1
                y_lo = py0 if y_side == 0 else (py1 - r)
                y_hi = (py0 + r) if y_side == 0 else py1
                in_c = (gx >= x_lo) & (gx <= x_hi) & (gy >= y_lo) & (gy <= y_hi)
                mask &= ~(in_c & (((gx - cx) ** 2 + (gy - cy) ** 2) >= r * r))
        return mask

    ipw = _IPWGrid2D(px0 - pad, px1 + pad, py0 - pad, py1 + pad, pitch, _cavity_target_mask)
    d_wall_grid = _wall_distance_grid(ipw.gx, ipw.gy, geom, r)

    # Cavity EDT via scipy.ndimage.distance_transform_edt for entry clearance analysis
    cavity_mask = d_wall_grid > 0.0
    cavity_edt_mm = ndimage.distance_transform_edt(cavity_mask) * pitch

    max_d = float(d_wall_grid.max()) if d_wall_grid.size else 0.0
    if max_d >= r + 0.30:
        finish_stock = min(0.15, 0.03 * d)
    elif max_d > r + 0.04:
        finish_stock = 0.25 * (max_d - r)
    else:
        finish_stock = 0.0
    rough_margin = finish_stock + 1e-4

    f_mask = cavity_mask & (d_wall_grid >= r + rough_margin)
    labeled_f, n_comp = ndimage.label(f_mask)

    builder = _AdaptivePathBuilder(task, config, geom, ipw, z_safe, pz0, stock_max_z, d_wall_grid=d_wall_grid)

    total_depth = max(0.0, stock_max_z - pz0)
    max_doc = (
        float(config.max_doc_mm)
        if config.max_doc_mm is not None
        else min(0.85 * float(task.tool.flute_length_mm), 2.2 * d)
    )
    n_z_passes = max(1, int(math.ceil((total_depth - 1e-6) / max(max_doc, 0.25))))
    z_passes = [
        pz0 if m == n_z_passes else (stock_max_z - (m / n_z_passes) * total_depth)
        for m in range(1, n_z_passes + 1)
    ]

    # Order roughing components from largest to smallest (cleared depth-first per component)
    comp_order = sorted(
        range(1, n_comp + 1),
        key=lambda cid: int(np.count_nonzero(labeled_f == cid)),
        reverse=True,
    )

    for cid in comp_order:
        comp_mask = labeled_f == cid
        if int(np.count_nonzero(comp_mask)) < 2:
            continue

        # Maximal-clearance entry discovery via distance_transform_edt on safe domain F_c
        f_edt_mm = ndimage.distance_transform_edt(comp_mask) * pitch
        combined_clearance = 0.5 * f_edt_mm + 0.5 * np.maximum(0.0, cavity_edt_mm - (r + rough_margin))
        max_c = float(combined_clearance[comp_mask].max())
        d_comp_max = float(d_wall_grid[comp_mask].max())

        # Select extremity of maximal-clearance plateau so narrow slots enter at one end
        plateau = comp_mask & (
            d_wall_grid >= (r + rough_margin) + 0.96 * max(0.0, d_comp_max - (r + rough_margin))
        )
        if not np.any(plateau):
            plateau = comp_mask & (combined_clearance >= 0.95 * max_c)
        cand_ij = np.argwhere(plateau)
        pick_idx = min(
            range(len(cand_ij)),
            key=lambda k: (float(ipw.xs[cand_ij[k, 0]]), float(ipw.ys[cand_ij[k, 1]])),
        )
        ei, ej = int(cand_ij[pick_idx, 0]), int(cand_ij[pick_idx, 1])
        entry_xy: Point2D = (float(ipw.xs[ei]), float(ipw.ys[ej]))

        c_free = max(0.0, _wall_distance_point(entry_xy, geom, r) - (r + rough_margin))
        r_bore = min(config.helix_bore_radius_fraction * c_free, 0.45 * d)
        r_open = min(c_free, 0.90 * d)

        # Skip tiny components whose entire reachable stock was already cleared
        reach_check = ndimage.distance_transform_edt(~comp_mask) * pitch <= (r + pitch)
        if not np.any(ipw.remaining & reach_check):
            continue

        seed_r = max(pitch * 0.8, r_open)
        seed_mask = ((ipw.gx - entry_xy[0]) ** 2 + (ipw.gy - entry_xy[1]) ** 2) <= (seed_r * seed_r)
        if not np.any(comp_mask & seed_mask):
            seed_mask = np.zeros_like(comp_mask)
            seed_mask[ei, ej] = True

        contours = _extract_component_contours(
            comp_mask,
            seed_mask,
            d_wall_grid,
            ipw.xs,
            ipw.ys,
            pitch,
            geom,
            r,
            rough_margin,
            config,
        )

        pre_comp_rem = ipw.remaining.copy() if n_z_passes > 1 else None
        for m_idx, z_pass in enumerate(z_passes):
            prev_z = z_passes[m_idx - 1] if m_idx > 0 else None
            # Stage 1: Helical spiral bore + Archimedean radial pocket expansion (stay-down between passes)
            builder.helical_bore_entry(
                entry_xy,
                r_bore,
                r_open,
                margin=rough_margin,
                z_target=z_pass,
                stay_down_from_z=prev_z,
                reset_ipw_to=pre_comp_rem if m_idx > 0 else None,
            )
            # Stage 2: Geodesic wavefront circular trochoids (channels) + offset peels (chambers)
            _schedule_and_execute_contours(contours, builder, geom, rough_margin)

    builder.cur_pass_z = pz0

    # Stage 3a: Pre-compute coverage of analytical wall/island finishing loops and sweep any
    # remaining uncut interior pockets before running the wall finishing passes.
    finish_chains = _build_analytical_finishing_chains(geom, r)
    finish_sim = _IPWGrid2D(px0 - pad, px1 + pad, py0 - pad, py1 + pad, pitch, lambda gx, gy: np.ones_like(gx, dtype=bool))
    for chain in finish_chains:
        finish_sim.sweep_polyline(chain, max(0.1 * r, r - 0.5 * pitch), mutate=True)
    will_finish_cover = ~finish_sim.remaining

    exact_safe_mask = cavity_mask & (d_wall_grid >= r + 1e-4)
    _sweep_residual_ipw(
        builder,
        f_mask if np.any(f_mask) else exact_safe_mask,
        ipw.xs,
        ipw.ys,
        margin=rough_margin if np.any(f_mask) else 1e-4,
        exclude_mask=will_finish_cover,
    )

    # Stage 3b: Exact analytical wall-finishing loops around outer pocket & standing islands
    remaining_chains = list(finish_chains)

    def _chain_entry_dist(ch: List[Point2D], cxy: Point2D) -> float:
        if len(ch) >= 4 and math.hypot(ch[0][0] - ch[-1][0], ch[0][1] - ch[-1][1]) <= 1e-4:
            return min(math.hypot(v[0] - cxy[0], v[1] - cxy[1]) for v in ch[:-1])
        return math.hypot(ch[0][0] - cxy[0], ch[0][1] - cxy[1])

    while remaining_chains:
        cur_xy = (builder.cur_xyz[0], builder.cur_xyz[1])
        best_c_idx = min(
            range(len(remaining_chains)),
            key=lambda i: _chain_entry_dist(remaining_chains[i], cur_xy),
        )
        chain = remaining_chains.pop(best_c_idx)
        # Rotate closed finishing loop to start at the corner nearest to the current cutter XY
        if len(chain) >= 4 and math.hypot(chain[0][0] - chain[-1][0], chain[0][1] - chain[-1][1]) <= 1e-4:
            ring = chain[:-1]
            best_v = min(
                range(len(ring)),
                key=lambda k: math.hypot(ring[k][0] - cur_xy[0], ring[k][1] - cur_xy[1]),
            )
            chain = ring[best_v:] + ring[:best_v] + [ring[best_v]]

        builder.link_to(chain[0], margin=0.0, allow_cut_step=True, return_tag="finish_link")
        for pt in chain[1:]:
            builder._emit(C.SegmentKind.FEED, (pt[0], pt[1], pz0), builder.cut_feed, "wall_finish")
        builder.last_peel_pts = None

    # Stage 3c: Final verification on 2D IPW at exact cutter radius R to guarantee 0% reachable residual
    _sweep_residual_ipw(builder, exact_safe_mask, ipw.xs, ipw.ys, margin=1e-4, exclude_mask=None)

    builder.retract_to_safe()
    return _waypoints_to_motion_plan(task, builder.waypoints, PROVENANCE_ID)


def estimate_radial_engagement_profile(
    plan: C.MotionPlan,
    task: C.TaskSpec,
    pitch_mm: float = 0.25,
) -> np.ndarray:
    """Estimate per-segment effective radial engagement ``ae_mm = swept_2d_area / xy_length``.

    Evaluates chronologically over a 2D boolean In-Process Workpiece (IPW) grid of
    resolution ``pitch_mm`` covering ``task.stock`` in the work frame. Rapid segments,
    above-stock air segments, and pure vertical plunges report ``0.0`` radial engagement
    (while plunges below ``stock_max_z`` still update the 2D IPW bore).
    """
    if pitch_mm <= 0.0 or not math.isfinite(pitch_mm):
        raise ValueError("pitch_mm must be finite and positive")

    s_min = [float(v) for v in task.stock.params["min_mm"]]
    s_max = [float(v) for v in task.stock.params["max_mm"]]
    sx0, sy0 = s_min[0], s_min[1]
    sx1, sy1, stock_max_z = s_max[0], s_max[1], s_max[2]
    r = float(task.tool.cutter_radius_mm)

    ipw = _IPWGrid2D(
        sx0,
        sx1,
        sy0,
        sy1,
        pitch_mm,
        lambda gx, gy: (gx >= sx0) & (gx <= sx1) & (gy >= sy0) & (gy <= sy1),
    )

    to_work = (lambda p: tuple(p)) if plan.frame == C.Frame.WORK else task.work_to_machine.inverse().apply
    profile = np.zeros(len(plan.segments), dtype=np.float64)

    for idx, seg in enumerate(plan.segments):
        if seg.kind == C.SegmentKind.RAPID:
            continue

        # Sample segment into work-frame 3D stations (exact line for FEED, chordal arc for ARC)
        if seg.kind == C.SegmentKind.ARC and seg.arc is not None:
            n_sub = max(4, int(math.ceil(seg.path_length_mm / max(0.1, 0.5 * pitch_mm))))
            arc_w = seg.arc
            if plan.frame == C.Frame.MACHINE:
                arc_w = dataclasses.replace(arc_w, centre_mm=to_work(arc_w.centre_mm))
            sw = to_work(seg.start_mm)
            ew = to_work(seg.end_mm)
            geom_arc = C.arc_geometry(sw, ew, arc_w, seg.id)
            if arc_w.plane == C.ArcPlane.XY:
                cx, cy, _ = arc_w.centre_mm
                th0 = math.atan2(sw[1] - cy, sw[0] - cx)
                rad = geom_arc["radius_mm"]
                sweep = geom_arc["sweep_rad"]
                pts_3d = [
                    (
                        cx + rad * math.cos(th0 + (k / n_sub) * sweep),
                        cy + rad * math.sin(th0 + (k / n_sub) * sweep),
                        sw[2] + (k / n_sub) * (ew[2] - sw[2]),
                    )
                    for k in range(n_sub + 1)
                ]
            else:
                pts_3d = [sw, ew]
        else:
            pts_3d = [to_work(seg.start_mm), to_work(seg.end_mm)]

        swept_area = 0.0
        xy_len = 0.0
        for k in range(len(pts_3d) - 1):
            a = pts_3d[k]
            b = pts_3d[k + 1]
            if a[2] >= stock_max_z - 1e-9 and b[2] >= stock_max_z - 1e-9:
                continue
            # Clip sub-segment to z <= stock_max_z
            p0, p1 = a, b
            if a[2] > stock_max_z and b[2] < stock_max_z:
                t = (stock_max_z - a[2]) / (b[2] - a[2])
                p0 = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), stock_max_z)
            elif b[2] > stock_max_z and a[2] < stock_max_z:
                t = (stock_max_z - a[2]) / (b[2] - a[2])
                p1 = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), stock_max_z)

            sub_xy = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            xy_len += sub_xy
            swept_area += ipw.sweep_segment((p0[0], p0[1]), (p1[0], p1[1]), r, mutate=True)

        if xy_len > 1e-6:
            profile[idx] = swept_area / xy_len

    return profile


__all__ = [
    "AdaptiveConfig",
    "PROVENANCE_ID",
    "generate_adaptive_toolpath",
    "estimate_radial_engagement_profile",
]
