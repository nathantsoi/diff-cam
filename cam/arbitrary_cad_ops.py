# SPDX-License-Identifier: MIT
"""Arbitrary-CAD 2.5D/3D toolpath generators and persistent-IPW CWE trajectory optimizer.

Provides clean-room MIT toolpath synthesis for arbitrary STEP/STL geometries:
  - Distance-transform multi-level inward/outward contour & rest-pocket clearing
  - Exact ball-endmill cutter-location (``z_CL``) 3D scallop surfacing
  - Helical bore & counterbore interpolation (enabling large bores with standard cribs)
  - 82°/90° countersinking, 45° perimeter chamfer milling, and climb helical thread milling
  - Differentiable persistent-IPW cutter-workpiece-engagement (CWE) feed & shortcut optimization
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from algorithms.chipload_opt import (
    _sweep_removes_or_updates_dexels,
    blended_durations_with_pullback,
)
from benchmarks import contracts as C
from cam.step_brep import BRepChamferFeature, BRepFreeformRegion, BRepHoleFeature


# ---------------------------------------------------------------------------
# 1. Marching-Squares 2D Iso-Contour Extraction & Simplification
# ---------------------------------------------------------------------------
def _rdp_simplify_2d(pts: np.ndarray, eps: float = 0.12) -> np.ndarray:
    """Iterative Ramer-Douglas-Peucker polyline simplifier in 2D."""
    if len(pts) <= 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: List[Tuple[int, int]] = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 <= 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        seg = p1 - p0
        slen2 = float(np.dot(seg, seg))
        sub = pts[i0 + 1 : i1]
        if slen2 <= 1e-12:
            dists = np.linalg.norm(sub - p0[None, :], axis=1)
        else:
            t = np.clip(((sub - p0[None, :]) @ seg) / slen2, 0.0, 1.0)
            proj = p0[None, :] + t[:, None] * seg[None, :]
            dists = np.linalg.norm(sub - proj, axis=1)
        imax = int(np.argmax(dists))
        if float(dists[imax]) > eps:
            split = i0 + 1 + imax
            keep[split] = True
            stack.append((i0, split))
            stack.append((split, i1))
    return pts[keep]


def trace_iso_contours_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    field_2d: np.ndarray,
    level: float,
    simplify_tol_mm: float = 0.14,
) -> List[np.ndarray]:
    """Extract ordered 2D iso-contours ``field_2d == level`` via marching squares."""
    nx, ny = field_2d.shape
    if nx < 2 or ny < 2:
        return []

    v00 = field_2d[:-1, :-1]
    v10 = field_2d[1:, :-1]
    v11 = field_2d[1:, 1:]
    v01 = field_2d[:-1, 1:]

    b0 = (v00 >= level).astype(np.int8)
    b1 = (v10 >= level).astype(np.int8)
    b2 = (v11 >= level).astype(np.int8)
    b3 = (v01 >= level).astype(np.int8)
    code = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)

    active_ix, active_iy = np.where((code > 0) & (code < 15))
    if len(active_ix) == 0:
        return []

    # Marching-squares directed edge table (keeps >= level interior on the left)
    # Edges: 0=bottom(00-10), 1=right(10-11), 2=top(01-11), 3=left(00-01)
    ms_table: Dict[int, Tuple[Tuple[int, int], ...]] = {
        1: ((3, 0),),
        2: ((0, 1),),
        3: ((3, 1),),
        4: ((1, 2),),
        5: ((3, 2), (1, 0)),
        6: ((0, 2),),
        7: ((3, 2),),
        8: ((2, 3),),
        9: ((2, 0),),
        10: ((0, 3), (2, 1)),
        11: ((2, 1),),
        12: ((1, 3),),
        13: ((1, 0),),
        14: ((0, 3),),
    }

    def _edge_pt_and_key(ix: int, iy: int, edge_id: int) -> Tuple[Tuple[int, int, int], Tuple[float, float]]:
        x0, x1 = float(xs[ix]), float(xs[ix + 1])
        y0, y1 = float(ys[iy]), float(ys[iy + 1])
        if edge_id == 0:  # bottom: (ix, iy) - (ix+1, iy)
            fa, fb = float(field_2d[ix, iy]), float(field_2d[ix + 1, iy])
            t = 0.5 if abs(fb - fa) < 1e-12 else max(0.0, min(1.0, (level - fa) / (fb - fa)))
            return (0, ix, iy), (x0 + t * (x1 - x0), y0)
        if edge_id == 1:  # right: (ix+1, iy) - (ix+1, iy+1)
            fa, fb = float(field_2d[ix + 1, iy]), float(field_2d[ix + 1, iy + 1])
            t = 0.5 if abs(fb - fa) < 1e-12 else max(0.0, min(1.0, (level - fa) / (fb - fa)))
            return (1, ix + 1, iy), (x1, y0 + t * (y1 - y0))
        if edge_id == 2:  # top: (ix, iy+1) - (ix+1, iy+1)
            fa, fb = float(field_2d[ix, iy + 1]), float(field_2d[ix + 1, iy + 1])
            t = 0.5 if abs(fb - fa) < 1e-12 else max(0.0, min(1.0, (level - fa) / (fb - fa)))
            return (0, ix, iy + 1), (x0 + t * (x1 - x0), y1)
        # left: (ix, iy) - (ix, iy+1)
        fa, fb = float(field_2d[ix, iy]), float(field_2d[ix, iy + 1])
        t = 0.5 if abs(fb - fa) < 1e-12 else max(0.0, min(1.0, (level - fa) / (fb - fa)))
        return (1, ix, iy), (x0, y0 + t * (y1 - y0))

    adj: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}
    in_deg: Dict[Tuple[int, int, int], int] = {}
    pt_map: Dict[Tuple[int, int, int], Tuple[float, float]] = {}

    for ix_np, iy_np in zip(active_ix, active_iy):
        ix, iy = int(ix_np), int(iy_np)
        c_val = int(code[ix, iy])
        for ea, eb in ms_table.get(c_val, ()):
            ka, pa = _edge_pt_and_key(ix, iy, ea)
            kb, pb = _edge_pt_and_key(ix, iy, eb)
            pt_map[ka] = pa
            pt_map[kb] = pb
            adj.setdefault(ka, []).append(kb)
            in_deg[kb] = in_deg.get(kb, 0) + 1
            in_deg.setdefault(ka, 0)

    contours: List[np.ndarray] = []
    # Trace open chains first (in_deg == 0), then closed rings
    starts = [k for k, deg in in_deg.items() if deg == 0 and adj.get(k)]
    for pass_open in (True, False):
        seed_keys = starts if pass_open else list(adj.keys())
        for sk in seed_keys:
            while adj.get(sk):
                chain = [pt_map[sk]]
                cur = sk
                while adj.get(cur):
                    nxt = adj[cur].pop(0)
                    chain.append(pt_map[nxt])
                    cur = nxt
                    if cur == sk:
                        break
                if len(chain) >= 3:
                    arr = _rdp_simplify_2d(np.asarray(chain, dtype=np.float64), eps=simplify_tol_mm)
                    if len(arr) >= 2 and float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))) >= 0.8:
                        contours.append(arr)
    return contours


# ---------------------------------------------------------------------------
# 2. Arbitrary 2.5D Level-Set Clearing, 3D Ball Scallop, & Hole/Thread Builders
# ---------------------------------------------------------------------------
def _make_motion_plan_from_segs(
    segs: List[C.Segment],
    z_safe_mm: float,
    rapid_mm_s: float = 100.0,
    max_accel_mm_s2: float = 150.0,
    provenance: str = "diffcam:arbitrary_cad",
) -> C.MotionPlan:
    if not segs:
        segs = [
            C.Segment(
                id="noop0",
                kind=C.SegmentKind.RAPID,
                start_mm=(0.0, 0.0, z_safe_mm),
                end_mm=(0.0, 0.0, z_safe_mm),
                feed_mm_s=None,
                duration_s=0.001,
            )
        ]
    else:
        durs = C.blended_segment_durations(segs, rapid_mm_s, max_accel_mm_s2)
        segs = [dataclasses.replace(s, duration_s=max(1e-5, float(d))) for s, d in zip(segs, durs)]
    return C.MotionPlan(
        frame=C.Frame.MACHINE,
        timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
        initial_state=C.MachineState(position_mm=(0.0, 0.0, z_safe_mm), frame=C.Frame.MACHINE),
        segments=tuple(segs),
        provenance=provenance,
    )


def build_facing_motion_plan(
    stock_min_mm: Tuple[float, float, float],
    stock_max_mm: Tuple[float, float, float],
    z_face_target_mm: float,
    z_start_top_mm: float,
    cutter_radius_mm: float,
    *,
    doc_mm: float,
    woc_mm: float,
    base_feed_mm_s: float,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 6.0,
) -> C.MotionPlan:
    """Multi-pass bidirectional zig-zag facing across stock XY with outside-air lead-in/out."""
    x0, y0, _ = stock_min_mm
    x1, y1, _ = stock_max_mm
    segs: List[C.Segment] = []
    cur = (0.0, 0.0, z_safe_mm)
    sid = 0

    def _push(pt: Tuple[float, float, float], kind: C.SegmentKind, feed: Optional[float]) -> None:
        nonlocal cur, sid
        dist = math.dist(cur, pt)
        if dist <= 1e-7:
            return
        spd = rapid_mm_s if kind == C.SegmentKind.RAPID or feed is None else feed
        segs.append(
            C.Segment(
                id=f"fc{sid:04d}",
                kind=kind,
                start_mm=cur,
                end_mm=pt,
                feed_mm_s=None if kind == C.SegmentKind.RAPID else round(float(spd), 4),
                duration_s=dist / max(spd, 1e-6),
            )
        )
        cur = pt
        sid += 1

    total_dz = max(0.0, z_start_top_mm - z_face_target_mm)
    if total_dz <= 1e-4:
        return _make_motion_plan_from_segs([], z_safe_mm, rapid_mm_s, provenance="diffcam:facing")

    n_z = max(1, int(math.ceil(total_dz / max(doc_mm, 0.25))))
    z_levels = [z_start_top_mm - (k + 1) * (total_dz / n_z) for k in range(n_z)]
    z_levels[-1] = z_face_target_mm

    lead_x = cutter_radius_mm + 1.5
    span_y = max(0.1, (y1 - y0) - 0.5 * cutter_radius_mm)
    n_y = max(1, int(math.ceil((y1 - y0) / max(woc_mm, 0.5))))
    ys_pass = np.linspace(y0 + 0.30 * woc_mm, y1 - 0.30 * woc_mm, n_y) if n_y > 1 else np.array([0.5 * (y0 + y1)])
    _ = span_y

    for z_cut in z_levels:
        x_left = x0 - lead_x
        x_right = x1 + lead_x
        _push((x_left, float(ys_pass[0]), z_safe_mm), C.SegmentKind.RAPID, None)
        _push((x_left, float(ys_pass[0]), z_cut), C.SegmentKind.FEED, 0.65 * base_feed_mm_s)
        for idx, yp in enumerate(ys_pass):
            yf = float(yp)
            if idx > 0:
                _push((cur[0], yf, z_cut), C.SegmentKind.FEED, base_feed_mm_s)
            x_end = x_right if (idx % 2 == 0) else x_left
            _push((x_end, yf, z_cut), C.SegmentKind.FEED, base_feed_mm_s)
        _push((cur[0], cur[1], z_safe_mm), C.SegmentKind.RAPID, None)

    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    return _make_motion_plan_from_segs(segs, z_safe_mm, rapid_mm_s, provenance="diffcam:facing")


def build_levelset_clearing_motion_plan(
    xs: np.ndarray,
    ys: np.ndarray,
    z_target_safe: np.ndarray,
    z_ipw: np.ndarray,
    cutter_radius_mm: float,
    *,
    z_top_start_mm: float,
    z_bottom_limit_mm: float,
    doc_mm: float,
    woc_mm: float,
    base_feed_mm_s: float,
    radial_stock_mm: float = 0.04,
    is_rest: bool = False,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 6.0,
) -> C.MotionPlan:
    """Arbitrary-silhouette 2.5D multi-level offset pocket/boss/rest clearing via Euclidean EDT."""
    pitch = float(xs[1] - xs[0]) if len(xs) > 1 else 0.5
    nx, ny = z_target_safe.shape
    segs: List[C.Segment] = []
    cur = (0.0, 0.0, z_safe_mm)
    sid = 0

    def _push(pt: Tuple[float, float, float], kind: C.SegmentKind, feed: Optional[float]) -> None:
        nonlocal cur, sid
        dist = math.dist(cur, pt)
        if dist <= 1e-6:
            return
        spd = rapid_mm_s if kind == C.SegmentKind.RAPID or feed is None else feed
        segs.append(
            C.Segment(
                id=f"ls{sid:05d}",
                kind=kind,
                start_mm=cur,
                end_mm=pt,
                feed_mm_s=None if kind == C.SegmentKind.RAPID else round(float(spd), 4),
                duration_s=dist / max(spd, 1e-6),
            )
        )
        cur = pt
        sid += 1

    # Determine distinct Z floors + intermediate axial stepdowns
    z_hi = float(z_top_start_mm)
    z_lo = max(float(z_bottom_limit_mm), float(np.min(z_target_safe)))
    if z_hi - z_lo <= 0.12:
        return _make_motion_plan_from_segs([], z_safe_mm, rapid_mm_s, provenance="diffcam:levelset_25d")

    step_z = max(0.5, float(doc_mm))
    n_steps = max(1, int(math.ceil((z_hi - z_lo) / step_z)))
    # Also snap to modal horizontal pocket/step floors in z_target_safe so floors have zero axial residual!
    quant_floors = np.unique(np.round(z_target_safe[(z_target_safe >= z_lo - 0.05) & (z_target_safe <= z_hi - 0.12)] * 20.0) / 20.0)
    uniform_levels = [round(z_hi - (k + 1) * ((z_hi - z_lo) / n_steps), 4) for k in range(n_steps)]
    cand_levels = sorted(set(uniform_levels + [round(float(zf), 4) for zf in quant_floors if z_lo - 0.02 <= zf <= z_hi - 0.12]), reverse=True)
    # Coalesce levels closer than 0.22 mm (preserving exact floors)
    z_levels: List[float] = []
    floor_set = {round(float(zf), 4) for zf in quant_floors}
    for zv in cand_levels:
        if not z_levels or (z_levels[-1] - zv) >= 0.25 or zv in floor_set:
            if z_levels and (z_levels[-1] - zv) < 0.18 and z_levels[-1] not in floor_set:
                z_levels[-1] = zv
            else:
                z_levels.append(zv)
    if len(z_levels) > 14:
        # Keep runtime tight on very deep parts while preserving lowest floor
        idx_keep = np.unique(np.round(np.linspace(0, len(z_levels) - 1, 12)).astype(int))
        z_levels = [z_levels[i] for i in idx_keep]

    # Pad grid by 2 cells around outer perimeter so exterior open contours close outside stock cleanly
    pad = max(2, int(math.ceil((cutter_radius_mm + 1.0) / pitch)))
    xs_pad = np.concatenate(
        [
            xs[0] - np.arange(pad, 0, -1, dtype=np.float64) * pitch,
            xs,
            xs[-1] + np.arange(1, pad + 1, dtype=np.float64) * pitch,
        ]
    )
    ys_pad = np.concatenate(
        [
            ys[0] - np.arange(pad, 0, -1, dtype=np.float64) * pitch,
            ys,
            ys[-1] + np.arange(1, pad + 1, dtype=np.float64) * pitch,
        ]
    )

    step_xy = max(0.35, float(woc_mm))
    r_req = cutter_radius_mm + max(0.05, radial_stock_mm) + 0.55 * pitch

    def _local_ipw_ceil(px_c: float, py_c: float) -> float:
        rad_c = cutter_radius_mm + 1.5 * pitch
        ix0_c = max(0, int(math.floor((px_c - rad_c - xs[0]) / pitch)))
        ix1_c = min(nx, int(math.ceil((px_c + rad_c - xs[0]) / pitch)) + 1)
        iy0_c = max(0, int(math.floor((py_c - rad_c - ys[0]) / pitch)))
        iy1_c = min(ny, int(math.ceil((py_c + rad_c - ys[0]) / pitch)) + 1)
        if ix0_c < ix1_c and iy0_c < iy1_c:
            return float(np.max(z_ipw[ix0_c:ix1_c, iy0_c:iy1_c]))
        return float(np.max(z_ipw))

    for z_cut in z_levels:
        part_solid = np.zeros((nx + 2 * pad, ny + 2 * pad), dtype=bool)
        part_solid[pad : pad + nx, pad : pad + ny] = z_target_safe > (z_cut + 0.035)

        ipw_solid = np.zeros((nx + 2 * pad, ny + 2 * pad), dtype=bool)
        ipw_solid[pad : pad + nx, pad : pad + ny] = z_ipw > (z_cut + 0.06)
        if not np.any(ipw_solid):
            continue

        dist_part = (
            distance_transform_edt(~part_solid) * pitch
            if np.any(part_solid)
            else np.full_like(part_solid, 999.0, dtype=np.float64)
        )
        dist_ipw = distance_transform_edt(~ipw_solid) * pitch

        reach_r = cutter_radius_mm * (0.82 if is_rest else 0.96)
        safe_mask = dist_part >= r_req
        active = safe_mask & (dist_ipw <= reach_r)
        if int(np.sum(active)) < 4:
            continue

        # Inward distance from boundary of active cutting zone
        d_in = distance_transform_edt(active) * pitch
        d_max = float(np.max(d_in))
        if d_max <= 0.15 * pitch:
            continue

        # Generate offset iso-levels spaced by step_xy (starting >= 0.65 * pitch inside active mask)
        iso_vals: List[float] = [max(0.65 * pitch, min(0.30 * step_xy, 0.55 * d_max))]
        lv = iso_vals[0] + step_xy
        while lv < d_max - 0.20 * pitch and len(iso_vals) < 36:
            iso_vals.append(lv)
            lv += step_xy

        # Order inner-to-outer so enclosed pockets ramp near center and spiral out; open borders enter gently
        ordered_loops: List[np.ndarray] = []
        simp_tol = min(0.10, 0.18 * pitch)
        for iv in reversed(iso_vals):
            loops = trace_iso_contours_2d(xs_pad, ys_pad, d_in, iv, simplify_tol_mm=simp_tol)
            ordered_loops.extend(loops)

        for loop in ordered_loops:
            sx, sy = float(loop[0, 0]), float(loop[0, 1])
            # Check if we can stay down safely between adjacent offset rings without crossing part_solid
            can_link_direct = False
            if abs(cur[2] - z_cut) <= 1e-4 and math.hypot(sx - cur[0], sy - cur[1]) <= 2.4 * step_xy + cutter_radius_mm:
                can_link_direct = True
                for alpha in (0.25, 0.5, 0.75):
                    mx = cur[0] + alpha * (sx - cur[0])
                    my = cur[1] + alpha * (sy - cur[1])
                    ix_m = int(round((mx - xs_pad[0]) / pitch))
                    iy_m = int(round((my - ys_pad[0]) / pitch))
                    if not (0 <= ix_m < len(xs_pad) and 0 <= iy_m < len(ys_pad) and safe_mask[ix_m, iy_m]):
                        can_link_direct = False
                        break

            if can_link_direct:
                _push((sx, sy, z_cut), C.SegmentKind.FEED, 0.85 * base_feed_mm_s)
            else:
                if cur[2] < z_safe_mm - 1e-5:
                    _push((cur[0], cur[1], z_safe_mm), C.SegmentKind.RAPID, None)
                _push((sx, sy, z_safe_mm), C.SegmentKind.RAPID, None)
                # Never rapid below local remaining IPW stock ceiling + 0.65 mm
                z_loc_top = _local_ipw_ceil(sx, sy)
                if len(loop) >= 2:
                    z_loc_top = max(z_loc_top, _local_ipw_ceil(float(loop[1, 0]), float(loop[1, 1])))
                z_ramp_top = z_cut + max(1.0, doc_mm + 0.5)
                z_above = min(z_safe_mm, max(z_ramp_top, z_loc_top + 0.65))
                _push((sx, sy, z_above), C.SegmentKind.RAPID, None)
                if z_above > z_ramp_top + 0.4:
                    _push((sx, sy, z_ramp_top), C.SegmentKind.FEED, 1.25 * base_feed_mm_s)
                if len(loop) >= 2 and math.hypot(float(loop[1, 0]) - sx, float(loop[1, 1]) - sy) >= 0.6:
                    nx_p, ny_p = float(loop[1, 0]), float(loop[1, 1])
                    z_mid = 0.5 * (min(z_above, z_ramp_top) + z_cut)
                    _push((nx_p, ny_p, z_mid), C.SegmentKind.FEED, 0.60 * base_feed_mm_s)
                    _push((sx, sy, z_cut), C.SegmentKind.FEED, 0.60 * base_feed_mm_s)
                else:
                    _push((sx, sy, z_cut), C.SegmentKind.FEED, 0.50 * base_feed_mm_s)

            for k in range(1, len(loop)):
                _push((float(loop[k, 0]), float(loop[k, 1]), z_cut), C.SegmentKind.FEED, base_feed_mm_s)

        if cur[2] < z_safe_mm - 1e-5:
            _push((cur[0], cur[1], z_safe_mm), C.SegmentKind.RAPID, None)

    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    return _make_motion_plan_from_segs(segs, z_safe_mm, rapid_mm_s, provenance="diffcam:levelset_25d")


def build_3d_ball_scallop_motion_plan(
    xs: np.ndarray,
    ys: np.ndarray,
    z_target_safe: np.ndarray,
    regions: Sequence[BRepFreeformRegion],
    cutter_radius_mm: float,
    corner_radius_mm: float,
    *,
    stepover_mm: float,
    base_feed_mm_s: float,
    stock_leave_z_mm: float = 0.06,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 6.0,
    z_ipw: Optional[np.ndarray] = None,
    z_min_clamp_mm: Optional[float] = None,
) -> C.MotionPlan:
    """Exact gouge-free 3D ball/bull-nose cutter-location (``z_CL``) scallop surfacing."""
    pitch = float(xs[1] - xs[0]) if len(xs) > 1 else 0.5
    nx, ny = z_target_safe.shape
    r_cut = float(cutter_radius_mm)
    r_tip = min(r_cut, max(0.0, float(corner_radius_mm)))

    # Precompute circular cutter-offset stencil delta_z(dx, dy) so tip never gouges 3D surface
    rad_cells = max(1, int(math.ceil(r_cut / pitch)))
    stencil: List[Tuple[int, int, float]] = []
    r_flat = max(0.0, r_cut - r_tip)
    for di in range(-rad_cells, rad_cells + 1):
        for dj in range(-rad_cells, rad_cells + 1):
            rr = math.hypot(di * pitch, dj * pitch)
            if rr <= r_cut + 1e-6:
                if rr <= r_flat or r_tip <= 1e-5:
                    dz_off = 0.0
                else:
                    u = min(r_tip, rr - r_flat)
                    dz_off = r_tip - math.sqrt(max(0.0, r_tip * r_tip - u * u))
                stencil.append((di, dj, -dz_off))

    segs: List[C.Segment] = []
    cur = (0.0, 0.0, z_safe_mm)
    sid = 0

    def _push(pt: Tuple[float, float, float], kind: C.SegmentKind, feed: Optional[float]) -> None:
        nonlocal cur, sid
        dist = math.dist(cur, pt)
        if dist <= 1e-6:
            return
        spd = rapid_mm_s if kind == C.SegmentKind.RAPID or feed is None else feed
        segs.append(
            C.Segment(
                id=f"sc{sid:05d}",
                kind=kind,
                start_mm=cur,
                end_mm=pt,
                feed_mm_s=None if kind == C.SegmentKind.RAPID else round(float(spd), 4),
                duration_s=dist / max(spd, 1e-6),
            )
        )
        cur = pt
        sid += 1

    ae = max(0.45, float(stepover_mm))
    for reg in regions:
        rx0, ry0 = reg.min_xy_mm
        rx1, ry1 = reg.max_xy_mm
        ix0 = max(0, int(math.floor((rx0 - xs[0]) / pitch)))
        ix1 = min(nx - 1, int(math.ceil((rx1 - xs[0]) / pitch)))
        iy0 = max(0, int(math.floor((ry0 - ys[0]) / pitch)))
        iy1 = min(ny - 1, int(math.ceil((ry1 - ys[0]) / pitch)))
        if ix1 <= ix0 + 1 or iy1 <= iy0 + 1:
            continue

        # Clamp z_target_safe floor to this sculptured region's Z range (and vise jaw limit) so
        # the ball cutter never plunges off curved exterior silhouettes or down through-bores!
        reg_floor = float(reg.z_min_mm) - 0.08
        if z_min_clamp_mm is not None:
            reg_floor = max(reg_floor, float(z_min_clamp_mm))
        z_surf_reg = np.maximum(z_target_safe, reg_floor)

        z_cl = z_surf_reg + stock_leave_z_mm
        for di, dj, neg_rise in stencil:
            ix_src0 = max(0, di)
            ix_src1 = min(nx, nx + di)
            iy_src0 = max(0, dj)
            iy_src1 = min(ny, ny + dj)
            ix_dst0 = ix_src0 - di
            ix_dst1 = ix_src1 - di
            iy_dst0 = iy_src0 - dj
            iy_dst1 = iy_src1 - dj
            if ix_src0 < ix_src1 and iy_src0 < iy_src1:
                z_cl[ix_dst0:ix_dst1, iy_dst0:iy_dst1] = np.maximum(
                    z_cl[ix_dst0:ix_dst1, iy_dst0:iy_dst1],
                    z_surf_reg[ix_src0:ix_src1, iy_src0:iy_src1] + neg_rise + stock_leave_z_mm,
                )

        n_lines = max(2, min(64, int(math.ceil((ys[iy1] - ys[iy0]) / ae)) + 1))
        iy_indices = np.unique(np.round(np.linspace(iy0, iy1, n_lines)).astype(int))
        ix_seq_full = np.arange(ix0, ix1 + 1, 1)

        first_line = True
        prev_iy = int(iy_indices[0])
        prev_ix_end = int(ix_seq_full[0])
        row_dir = 0
        for iy_c_np in iy_indices:
            iy_c = int(iy_c_np)
            # Trim exterior silhouette air margins outside the sculptured face
            on_face = z_target_safe[ix_seq_full, iy_c] >= (float(reg.z_min_mm) - 0.45)
            valid_pos = np.where(on_face)[0]
            if len(valid_pos) < 2:
                continue
            p0 = max(0, int(valid_pos[0]) - 1)
            p1 = min(len(ix_seq_full) - 1, int(valid_pos[-1]) + 1)
            ix_sub = ix_seq_full[p0 : p1 + 1]
            ix_seq = ix_sub if (row_dir % 2 == 0) else ix_sub[::-1]
            row_dir += 1

            y_val = float(ys[iy_c])
            ix_s0 = int(ix_seq[0])
            x_start = float(xs[ix_s0])
            z_start = float(z_cl[ix_s0, iy_c])
            if first_line or math.hypot(x_start - cur[0], y_val - cur[1]) > max(12.0, 5.0 * ae):
                if not first_line and cur[2] < z_safe_mm - 1e-5:
                    _push((cur[0], cur[1], z_safe_mm), C.SegmentKind.RAPID, None)
                _push((x_start, y_val, z_safe_mm), C.SegmentKind.RAPID, None)
                z_ipw_loc = z_start
                if z_ipw is not None:
                    rx_c = max(1, int(math.ceil((r_cut + pitch) / pitch)))
                    z_ipw_loc = float(
                        np.max(
                            z_ipw[
                                max(0, ix_s0 - rx_c) : min(nx, ix_s0 + rx_c + 1),
                                max(0, iy_c - rx_c) : min(ny, iy_c + rx_c + 1),
                            ]
                        )
                    )
                z_rap_to = min(z_safe_mm, max(z_start + 1.5, z_ipw_loc + 0.65))
                _push((x_start, y_val, z_rap_to), C.SegmentKind.RAPID, None)
                _push((x_start, y_val, z_start), C.SegmentKind.FEED, 0.70 * base_feed_mm_s)
                first_line = False
            else:
                ix_lo = max(0, min(prev_ix_end, ix_s0) - 1)
                ix_hi = min(nx, max(prev_ix_end, ix_s0) + 2)
                iy_lo = max(0, min(prev_iy, iy_c) - 1)
                iy_hi = min(ny, max(prev_iy, iy_c) + 2)
                z_link = max(cur[2], z_start, float(np.max(z_cl[ix_lo:ix_hi, iy_lo:iy_hi]))) + 0.45
                _push((cur[0], cur[1], z_link), C.SegmentKind.FEED, 0.90 * base_feed_mm_s)
                _push((x_start, y_val, z_link), C.SegmentKind.FEED, 0.90 * base_feed_mm_s)
                _push((x_start, y_val, z_start), C.SegmentKind.FEED, 0.70 * base_feed_mm_s)

            for ix_c in ix_seq[1:]:
                _push((float(xs[int(ix_c)]), y_val, float(z_cl[int(ix_c), iy_c])), C.SegmentKind.FEED, base_feed_mm_s)
            prev_iy = iy_c
            prev_ix_end = int(ix_seq[-1])

        if cur[2] < z_safe_mm - 1e-5:
            _push((cur[0], cur[1], z_safe_mm), C.SegmentKind.RAPID, None)

    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    return _make_motion_plan_from_segs(segs, z_safe_mm, rapid_mm_s, provenance="diffcam:3d_ball_scallop")


def build_helical_bore_motion_plan(
    bores: Sequence[Tuple[Tuple[float, float], float, float, float]],  # ((cx, cy), bore_diam, z_top, z_bot)
    cutter_radius_mm: float,
    *,
    doc_mm: float,
    woc_mm: float,
    base_feed_mm_s: float,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 6.0,
) -> C.MotionPlan:
    """Helical ramp + radial spiral-out + circular finish pass for bores & counterbores."""
    segs: List[C.Segment] = []
    cur = (0.0, 0.0, z_safe_mm)
    sid = 0
    diam = 2.0 * cutter_radius_mm

    def _push(pt: Tuple[float, float, float], kind: C.SegmentKind, feed: Optional[float]) -> None:
        nonlocal cur, sid
        dist = math.dist(cur, pt)
        if dist <= 1e-6:
            return
        spd = rapid_mm_s if kind == C.SegmentKind.RAPID or feed is None else feed
        segs.append(
            C.Segment(
                id=f"hb{sid:05d}",
                kind=kind,
                start_mm=cur,
                end_mm=pt,
                feed_mm_s=None if kind == C.SegmentKind.RAPID else round(float(spd), 4),
                duration_s=dist / max(spd, 1e-6),
            )
        )
        cur = pt
        sid += 1

    for (cx, cy), bore_d, z_top, z_bot in bores:
        r_max_orbit = 0.5 * bore_d - cutter_radius_mm
        if r_max_orbit <= 0.04 or z_top <= z_bot + 0.05:
            continue
        r_helix = min(r_max_orbit, max(0.15, 0.36 * diam))
        pitch_rev = max(0.4, min(1.6, 0.28 * diam, doc_mm))
        z_entry = z_top + 0.5
        total_dz = z_entry - z_bot
        n_revs = max(1.0, total_dz / pitch_rev)
        pts_per_rev = 16
        n_helix_pts = max(12, int(math.ceil(n_revs * pts_per_rev)))

        _push((cx + r_helix, cy, z_safe_mm), C.SegmentKind.RAPID, None)
        _push((cx + r_helix, cy, min(z_safe_mm, z_entry + 1.0)), C.SegmentKind.RAPID, None)
        # Helical ramp down to z_bot
        for k in range(1, n_helix_pts + 1):
            frac = k / n_helix_pts
            ang = 2.0 * math.pi * n_revs * frac
            zk = z_entry - frac * total_dz
            _push((cx + r_helix * math.cos(ang), cy + r_helix * math.sin(ang), zk), C.SegmentKind.FEED, 0.72 * base_feed_mm_s)

        # If bore is wider than core helix, step outward in axial layers of doc_mm
        if r_max_orbit > r_helix + 0.08:
            n_z_layers = max(1, int(math.ceil((z_top - z_bot) / max(doc_mm, 0.8))))
            for lz_i in range(n_z_layers):
                z_layer = z_top - (lz_i + 1) * ((z_top - z_bot) / n_z_layers)
                if lz_i > 0 or abs(cur[2] - z_layer) > 0.2:
                    _push((cx, cy, cur[2]), C.SegmentKind.FEED, base_feed_mm_s)
                    _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)
                    _push((cx + r_helix, cy, z_safe_mm), C.SegmentKind.RAPID, None)
                    _push((cx + r_helix, cy, min(z_safe_mm, z_entry + 1.0)), C.SegmentKind.RAPID, None)
                    _push((cx + r_helix, cy, z_layer), C.SegmentKind.FEED, 1.25 * base_feed_mm_s)
                n_rad = max(1, int(math.ceil((r_max_orbit - r_helix) / max(woc_mm, 0.4))))
                for r_i in range(1, n_rad + 1):
                    rk = r_helix + (r_i / n_rad) * (r_max_orbit - r_helix)
                    n_circ = max(16, min(36, int(math.ceil(2.0 * math.pi * rk / 1.2))))
                    for c_i in range(n_circ + 1):
                        th = 2.0 * math.pi * (c_i / n_circ)
                        _push((cx + rk * math.cos(th), cy + rk * math.sin(th), z_layer), C.SegmentKind.FEED, base_feed_mm_s)
        else:
            # Cleanup 360° circle at z_bot
            for c_i in range(1, 19):
                th = 2.0 * math.pi * (c_i / 18.0)
                _push((cx + r_max_orbit * math.cos(th), cy + r_max_orbit * math.sin(th), z_bot), C.SegmentKind.FEED, base_feed_mm_s)

        # Lead to bore center before rapid retract so wall has zero dwell mark
        _push((cx, cy, cur[2]), C.SegmentKind.FEED, base_feed_mm_s)
        _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)

    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    return _make_motion_plan_from_segs(segs, z_safe_mm, rapid_mm_s, provenance="diffcam:helical_bore")


def build_countersink_motion_plan(
    holes: Sequence[BRepHoleFeature],
    tool_diameter_mm: float,
    tip_angle_deg: float,
    *,
    plunge_feed_mm_s: float,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 6.0,
) -> C.MotionPlan:
    """Countersink plunge/circular chamfering for 82° / 90° countersunk holes."""
    segs: List[C.Segment] = []
    cur = (0.0, 0.0, z_safe_mm)
    sid = 0
    half_rad = math.radians(max(25.0, min(65.0, 0.5 * tip_angle_deg)))
    tan_h = math.tan(half_rad)

    def _push(pt: Tuple[float, float, float], kind: C.SegmentKind, feed: Optional[float]) -> None:
        nonlocal cur, sid
        dist = math.dist(cur, pt)
        if dist <= 1e-6:
            return
        spd = rapid_mm_s if kind == C.SegmentKind.RAPID or feed is None else feed
        segs.append(
            C.Segment(
                id=f"cs{sid:04d}",
                kind=kind,
                start_mm=cur,
                end_mm=pt,
                feed_mm_s=None if kind == C.SegmentKind.RAPID else round(float(spd), 4),
                duration_s=dist / max(spd, 1e-6),
            )
        )
        cur = pt
        sid += 1

    for hl in holes:
        cs_d = hl.countersink_diameter_mm or (hl.diameter_mm + 1.0)
        cx, cy = hl.center_xy_mm
        _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)
        _push((cx, cy, min(z_safe_mm, hl.z_top_mm + 1.0)), C.SegmentKind.RAPID, None)
        if cs_d <= tool_diameter_mm + 0.05:
            # Direct axial countersink plunge: z_tip = z_top - (0.5 * cs_d) / tan(theta/2)
            z_tip = max(hl.z_bottom_mm, hl.z_top_mm - (0.5 * cs_d) / max(tan_h, 0.25))
            _push((cx, cy, z_tip), C.SegmentKind.FEED, plunge_feed_mm_s)
        else:
            # Circular chamfer mill orbit when countersink mouth exceeds tool diameter
            tip_engage = 0.32 * tool_diameter_mm
            r_orbit = max(0.05, 0.5 * cs_d - tip_engage)
            z_tip = max(hl.z_bottom_mm, hl.z_top_mm - tip_engage / max(tan_h, 0.25))
            _push((cx, cy, z_tip), C.SegmentKind.FEED, plunge_feed_mm_s)
            for k in range(19):
                th = 2.0 * math.pi * (k / 18.0)
                _push((cx + r_orbit * math.cos(th), cy + r_orbit * math.sin(th), z_tip), C.SegmentKind.FEED, 1.5 * plunge_feed_mm_s)
            _push((cx, cy, z_tip), C.SegmentKind.FEED, 1.5 * plunge_feed_mm_s)
        _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)

    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    return _make_motion_plan_from_segs(segs, z_safe_mm, rapid_mm_s, provenance="diffcam:countersink")


def build_chamfer_contour_motion_plan(
    xs: np.ndarray,
    ys: np.ndarray,
    z_target_safe: np.ndarray,
    chamfers: Sequence[BRepChamferFeature],
    tool_diameter_mm: float,
    tip_angle_deg: float,
    *,
    base_feed_mm_s: float,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 6.0,
) -> C.MotionPlan:
    """Collision-checked 45° perimeter/boss chamfer trace with flute-offset tip engagement."""
    pitch = float(xs[1] - xs[0]) if len(xs) > 1 else 0.5
    nx, ny = z_target_safe.shape
    half_rad = math.radians(max(30.0, min(60.0, 0.5 * tip_angle_deg)))
    tip_past = min(0.8, 0.18 * tool_diameter_mm)
    r_engage = tip_past * math.tan(half_rad)

    segs: List[C.Segment] = []
    cur = (0.0, 0.0, z_safe_mm)
    sid = 0

    def _push(pt: Tuple[float, float, float], kind: C.SegmentKind, feed: Optional[float]) -> None:
        nonlocal cur, sid
        dist = math.dist(cur, pt)
        if dist <= 1e-6:
            return
        spd = rapid_mm_s if kind == C.SegmentKind.RAPID or feed is None else feed
        segs.append(
            C.Segment(
                id=f"ch{sid:04d}",
                kind=kind,
                start_mm=cur,
                end_mm=pt,
                feed_mm_s=None if kind == C.SegmentKind.RAPID else round(float(spd), 4),
                duration_s=dist / max(spd, 1e-6),
            )
        )
        cur = pt
        sid += 1

    for cf in chamfers[:24]:
        if len(cf.polyline_xy_mm) < 2:
            continue
        p0 = np.asarray(cf.polyline_xy_mm[0], dtype=np.float64)
        p1 = np.asarray(cf.polyline_xy_mm[-1], dtype=np.float64)
        chord = p1 - p0
        clen = float(np.linalg.norm(chord))
        if clen < 0.8:
            continue
        tang = chord / clen
        n_cand_a = np.array([-tang[1], tang[0]])
        n_cand_b = -n_cand_a
        mid = 0.5 * (p0 + p1)

        # Pick normal pointing toward LOWER z_target_safe (away from higher solid boss!)
        def _sample_z(pt_xy: np.ndarray) -> float:
            ix = min(nx - 1, max(0, int(round((pt_xy[0] - xs[0]) / pitch))))
            iy = min(ny - 1, max(0, int(round((pt_xy[1] - ys[0]) / pitch))))
            return float(z_target_safe[ix, iy])

        za = _sample_z(mid + (r_engage + 0.8) * n_cand_a)
        zb = _sample_z(mid + (r_engage + 0.8) * n_cand_b)
        out_n = n_cand_a if za <= zb else n_cand_b
        z_tip = cf.z_bottom_mm - tip_past

        # Verify cutter center along shifted segment stays clear of z_target_safe above z_tip
        q0 = p0 + r_engage * out_n
        q1 = p1 + r_engage * out_n
        safe_ok = True
        for alpha in (0.0, 0.5, 1.0):
            qm = q0 + alpha * (q1 - q0)
            if _sample_z(qm) > cf.z_bottom_mm + 0.15:
                safe_ok = False
                break
        if not safe_ok:
            continue

        _push((float(q0[0]), float(q0[1]), z_safe_mm), C.SegmentKind.RAPID, None)
        _push((float(q0[0]), float(q0[1]), min(z_safe_mm, cf.z_top_mm + 1.0)), C.SegmentKind.RAPID, None)
        _push((float(q0[0]), float(q0[1]), z_tip), C.SegmentKind.FEED, 0.65 * base_feed_mm_s)
        _push((float(q1[0]), float(q1[1]), z_tip), C.SegmentKind.FEED, base_feed_mm_s)
        _push((float(q1[0]), float(q1[1]), z_safe_mm), C.SegmentKind.RAPID, None)

    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    return _make_motion_plan_from_segs(segs, z_safe_mm, rapid_mm_s, provenance="diffcam:chamfer")


def build_thread_mill_motion_plan(
    holes: Sequence[BRepHoleFeature],
    tool_diameter_mm: float,
    entity_type: str,
    *,
    base_feed_mm_s: float,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 6.0,
) -> C.MotionPlan:
    """Bottom-up climb helical internal thread milling (or synchronized rigid tap cycle)."""
    segs: List[C.Segment] = []
    cur = (0.0, 0.0, z_safe_mm)
    sid = 0
    is_tap = "tap" in entity_type.lower()

    def _push(pt: Tuple[float, float, float], kind: C.SegmentKind, feed: Optional[float]) -> None:
        nonlocal cur, sid
        dist = math.dist(cur, pt)
        if dist <= 1e-6:
            return
        spd = rapid_mm_s if kind == C.SegmentKind.RAPID or feed is None else feed
        segs.append(
            C.Segment(
                id=f"tm{sid:05d}",
                kind=kind,
                start_mm=cur,
                end_mm=pt,
                feed_mm_s=None if kind == C.SegmentKind.RAPID else round(float(spd), 4),
                duration_s=dist / max(spd, 1e-6),
            )
        )
        cur = pt
        sid += 1

    for hl in holes:
        cx, cy = hl.center_xy_mm
        maj_d = hl.thread_major_mm or (hl.diameter_mm + 1.0)
        pitch = hl.thread_pitch_mm or 1.0
        z_bot_th = max(hl.z_bottom_mm + 0.8, hl.z_top_mm - min(hl.z_top_mm - hl.z_bottom_mm - 0.5, 2.2 * maj_d))
        if z_bot_th >= hl.z_top_mm - 0.5:
            continue

        _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)
        _push((cx, cy, min(z_safe_mm, hl.z_top_mm + 1.5)), C.SegmentKind.RAPID, None)

        if is_tap or tool_diameter_mm >= maj_d - 0.05:
            # Synchronized rigid tap plunge & feed retract along hole axis
            _push((cx, cy, z_bot_th), C.SegmentKind.FEED, base_feed_mm_s)
            _push((cx, cy, min(z_safe_mm, hl.z_top_mm + 1.5)), C.SegmentKind.FEED, base_feed_mm_s)
            _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)
        else:
            # Bottom-up right-hand internal climb helical thread mill (CCW from bottom up!)
            r_orbit = max(0.05, 0.5 * (maj_d - tool_diameter_mm))
            _push((cx, cy, z_bot_th), C.SegmentKind.RAPID, None)
            # Radial lead-in to thread wall
            _push((cx + r_orbit, cy, z_bot_th), C.SegmentKind.FEED, 0.60 * base_feed_mm_s)
            n_revs = max(1.0, (hl.z_top_mm + 0.5 * pitch - z_bot_th) / max(pitch, 0.25))
            n_pts = max(16, int(math.ceil(n_revs * 18)))
            for k in range(1, n_pts + 1):
                frac = k / n_pts
                ang = 2.0 * math.pi * n_revs * frac
                zk = z_bot_th + frac * n_revs * pitch
                _push((cx + r_orbit * math.cos(ang), cy + r_orbit * math.sin(ang), zk), C.SegmentKind.FEED, base_feed_mm_s)
            # Return to hole centerline before Z retract so teeth clear internal threads!
            _push((cx, cy, cur[2]), C.SegmentKind.FEED, base_feed_mm_s)
            _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)

    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    return _make_motion_plan_from_segs(segs, z_safe_mm, rapid_mm_s, provenance="diffcam:thread_mill")


def build_wips_probe_motion_plan(
    stock_min_mm: Tuple[float, float, float],
    stock_max_mm: Tuple[float, float, float],
    stylus_radius_mm: float = 3.0,
    z_top_datum_mm: float = 0.0,
    probe_feed_mm_s: float = 25.0,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 8.0,
) -> C.MotionPlan:
    """Spindle touch-probe (T99) top-Z + 4-side XY web WCS calibration cycle for 3D visualization."""
    x0, y0, _ = stock_min_mm
    x1, y1, _ = stock_max_mm
    z_side = z_top_datum_mm - 3.5
    clear = stylus_radius_mm + 4.0
    touch = stylus_radius_mm + 0.02

    segs: List[C.Segment] = []
    cur = (0.0, 0.0, z_safe_mm)
    sid = 0

    def _push(pt: Tuple[float, float, float], kind: C.SegmentKind, feed: Optional[float]) -> None:
        nonlocal cur, sid
        dist = math.dist(cur, pt)
        if dist <= 1e-6:
            return
        spd = rapid_mm_s if kind == C.SegmentKind.RAPID or feed is None else feed
        segs.append(
            C.Segment(
                id=f"pr{sid:04d}",
                kind=kind,
                start_mm=cur,
                end_mm=pt,
                feed_mm_s=None if kind == C.SegmentKind.RAPID else round(float(spd), 4),
                duration_s=dist / max(spd, 1e-6),
            )
        )
        cur = pt
        sid += 1

    # 1. Center Z datum touch
    _push((0.0, 0.0, z_top_datum_mm + 4.0), C.SegmentKind.RAPID, None)
    _push((0.0, 0.0, z_top_datum_mm + 0.02), C.SegmentKind.FEED, probe_feed_mm_s)
    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    # 2. -X / +X / -Y / +Y stock web touches
    for (ax, ay), (tx, ty) in (
        ((x0 - clear, 0.0), (x0 - touch, 0.0)),
        ((x1 + clear, 0.0), (x1 + touch, 0.0)),
        ((0.0, y0 - clear), (0.0, y0 - touch)),
        ((0.0, y1 + clear), (0.0, y1 + touch)),
    ):
        _push((ax, ay, z_safe_mm), C.SegmentKind.RAPID, None)
        _push((ax, ay, z_side), C.SegmentKind.RAPID, None)
        _push((tx, ty, z_side), C.SegmentKind.FEED, probe_feed_mm_s)
        _push((ax, ay, z_side), C.SegmentKind.FEED, probe_feed_mm_s)
        _push((ax, ay, z_safe_mm), C.SegmentKind.RAPID, None)

    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    return _make_motion_plan_from_segs(segs, z_safe_mm, rapid_mm_s, provenance="diffcam:wips_probe")


# ---------------------------------------------------------------------------
# 3. Persistent-IPW Differentiable CWE Trajectory Optimizer for Arbitrary CAD
# ---------------------------------------------------------------------------
def optimize_arbitrary_cwe_plan(
    nominal_plan: C.MotionPlan,
    xs: np.ndarray,
    ys: np.ndarray,
    z_ipw: np.ndarray,
    z_target_safe: np.ndarray,
    stock_min_mm: Tuple[float, float, float],
    stock_max_mm: Tuple[float, float, float],
    *,
    cutter_radius_mm: float,
    flutes: int,
    spindle_rpm: float,
    base_feed_mm_s: float,
    target_hex_mm: float,
    max_feed_mm_s: float = 65.0,
    mrr_limit_mm3_s: float = 450.0,
    opt_iters: int = 32,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 6.0,
    naive_nominal_override: Optional[C.MotionPlan] = None,
    rapid_check_radius_mm: Optional[float] = None,
) -> Tuple[C.MotionPlan, C.MotionPlan, C.MotionPlan, float]:
    """Subdivide, shortcut verified air moves, and Adam-optimize per-segment chipload on persistent IPW.

    Mutates ``z_ipw`` in place to reflect the post-operation carved stock and returns
    ``(naive_plan, same_speed_plan, opt_chipload_plan, max_xy_shift_mm)``.
    """
    pitch = float(xs[1] - xs[0]) if len(xs) > 1 else 0.5
    cell_area = pitch * pitch
    x0_g, y0_g, z_min_s = stock_min_mm
    z_max_s = stock_max_mm[2]
    diam = 2.0 * cutter_radius_mm
    r_rapid_chk = max(cutter_radius_mm, float(rapid_check_radius_mm or cutter_radius_mm)) + 0.06
    max_sub_mm = max(1.0, min(3.0, 0.55 * diam))

    # 1. Subdivide long cutting segments for spatial CWE resolution
    sub_segs: List[C.Segment] = []
    sid = 0
    for seg in nominal_plan.segments:
        ln = math.dist(seg.start_mm, seg.end_mm)
        if seg.kind == C.SegmentKind.RAPID or ln <= max_sub_mm:
            sub_segs.append(
                dataclasses.replace(
                    seg,
                    id=f"s{sid:05d}",
                    feed_mm_s=None if seg.kind == C.SegmentKind.RAPID else float(seg.feed_mm_s or base_feed_mm_s),
                )
            )
            sid += 1
        else:
            n_k = max(2, int(math.ceil(ln / max_sub_mm)))
            pa, pb = seg.start_mm, seg.end_mm
            for k in range(n_k):
                a0, a1 = k / n_k, (k + 1) / n_k
                p0 = (pa[0] + a0 * (pb[0] - pa[0]), pa[1] + a0 * (pb[1] - pa[1]), pa[2] + a0 * (pb[2] - pa[2]))
                p1 = (pa[0] + a1 * (pb[0] - pa[0]), pa[1] + a1 * (pb[1] - pa[1]), pa[2] + a1 * (pb[2] - pa[2]))
                sub_segs.append(
                    C.Segment(
                        id=f"s{sid:05d}",
                        kind=C.SegmentKind.FEED,
                        start_mm=p0,
                        end_mm=p1,
                        feed_mm_s=float(seg.feed_mm_s or base_feed_mm_s),
                        duration_s=(ln / n_k) / max(base_feed_mm_s, 1e-6),
                    )
                )
                sid += 1

    base_sub_plan = _make_motion_plan_from_segs(
        [
            dataclasses.replace(s, feed_mm_s=None if s.kind == C.SegmentKind.RAPID else round(base_feed_mm_s, 4))
            for s in sub_segs
        ],
        z_safe_mm,
        rapid_mm_s,
        provenance="diffcam:naive_arbitrary",
    )
    if naive_nominal_override is not None and len(naive_nominal_override.segments) > 0:
        nv_segs: List[C.Segment] = []
        nsid = 0
        for seg in naive_nominal_override.segments:
            ln = math.dist(seg.start_mm, seg.end_mm)
            if seg.kind == C.SegmentKind.RAPID or ln <= max_sub_mm:
                nv_segs.append(
                    dataclasses.replace(
                        seg,
                        id=f"nv{nsid:05d}",
                        feed_mm_s=None if seg.kind == C.SegmentKind.RAPID else round(base_feed_mm_s, 4),
                    )
                )
                nsid += 1
            else:
                n_k = max(2, int(math.ceil(ln / max_sub_mm)))
                pa, pb = seg.start_mm, seg.end_mm
                for k in range(n_k):
                    a0, a1 = k / n_k, (k + 1) / n_k
                    p0 = (pa[0] + a0 * (pb[0] - pa[0]), pa[1] + a0 * (pb[1] - pa[1]), pa[2] + a0 * (pb[2] - pa[2]))
                    p1 = (pa[0] + a1 * (pb[0] - pa[0]), pa[1] + a1 * (pb[1] - pa[1]), pa[2] + a1 * (pb[2] - pa[2]))
                    nv_segs.append(
                        C.Segment(
                            id=f"nv{nsid:05d}",
                            kind=C.SegmentKind.FEED,
                            start_mm=p0,
                            end_mm=p1,
                            feed_mm_s=round(base_feed_mm_s, 4),
                            duration_s=(ln / n_k) / max(base_feed_mm_s, 1e-6),
                        )
                    )
                    nsid += 1
        naive_plan = _make_motion_plan_from_segs(nv_segs, z_safe_mm, rapid_mm_s, provenance="diffcam:naive_arbitrary")
    else:
        naive_plan = base_sub_plan

    if len(sub_segs) <= 1:
        return naive_plan, base_sub_plan, base_sub_plan, 0.0

    # 2. Simulate sequentially over z_ipw to measure exact per-segment CWE (rem_v, a_p, a_e)
    #    and convert any nominal rapid that approaches residual IPW stock into a safe feed move.
    z_work = z_ipw.copy()
    n_s = len(sub_segs)
    rem_vols = np.zeros(n_s, dtype=np.float64)
    ap_arr = np.zeros(n_s, dtype=np.float64)
    ae_arr = np.zeros(n_s, dtype=np.float64)
    len_arr = np.zeros(n_s, dtype=np.float64)
    is_feed = np.zeros(n_s, dtype=bool)
    had_rapid_upgrade = False

    for k, seg in enumerate(sub_segs):
        ln = math.dist(seg.start_mm, seg.end_mm)
        len_arr[k] = ln
        if ln <= 1e-8:
            continue
        if seg.kind == C.SegmentKind.RAPID:
            rap_probe = _sweep_removes_or_updates_dexels(
                xs, ys, z_work, x0_g, y0_g, pitch, r_rapid_chk, z_min_s, z_max_s,
                seg.start_mm, seg.end_mm, mutate=False,
            )
            if rap_probe <= 0.005:
                continue
            sub_segs[k] = dataclasses.replace(
                seg,
                kind=C.SegmentKind.FEED,
                feed_mm_s=round(base_feed_mm_s, 4),
                duration_s=ln / max(base_feed_mm_s, 1e-6),
            )
            seg = sub_segs[k]
            had_rapid_upgrade = True
        is_feed[k] = True
        z_before = z_work.copy()
        _sweep_removes_or_updates_dexels(
            xs, ys, z_work, x0_g, y0_g, pitch, cutter_radius_mm, z_min_s, z_max_s,
            seg.start_mm, seg.end_mm, mutate=True,
        )
        dz = np.maximum(0.0, z_before - z_work)
        rv = float(np.sum(dz)) * cell_area
        rem_vols[k] = rv
        if rv > 1e-4:
            ap_est = float(np.percentile(dz[dz > 1e-5], 80))
            len_xy = math.hypot(seg.end_mm[0] - seg.start_mm[0], seg.end_mm[1] - seg.start_mm[1])
            ae_est = min(diam, rv / max(ap_est * (len_xy + 0.60 * diam), 1e-6)) if len_xy > 0.12 else diam
            ap_arr[k] = ap_est
            ae_arr[k] = ae_est

    if had_rapid_upgrade:
        base_sub_plan = _make_motion_plan_from_segs(
            [
                dataclasses.replace(s, feed_mm_s=None if s.kind == C.SegmentKind.RAPID else round(base_feed_mm_s, 4))
                for s in sub_segs
            ],
            z_safe_mm,
            rapid_mm_s,
            provenance="diffcam:naive_arbitrary",
        )
        if naive_nominal_override is None:
            naive_plan = base_sub_plan

    # 3. Shortcut verified contiguous zero-engagement air-feed chains via full-radius safe rapid bridge!
    #    (Checks against BOTH initial z_ipw and evolved z_work with full physical cutter radius r_rapid_chk!)
    z_guard = np.maximum(z_ipw, z_work)
    shortcut_segs: List[C.Segment] = []
    idx = 0
    while idx < n_s:
        if is_feed[idx] and rem_vols[idx] <= 1e-4:
            j = idx
            air_len = 0.0
            while j < n_s and is_feed[j] and rem_vols[j] <= 1e-4:
                air_len += len_arr[j]
                j += 1
            p_start = sub_segs[idx].start_mm
            p_end = sub_segs[j - 1].end_mm
            if air_len >= 5.5:
                orig_air_t = air_len / max(base_feed_mm_s, 1e-6)
                direct_probe_v = _sweep_removes_or_updates_dexels(
                    xs, ys, z_guard, x0_g, y0_g, pitch, r_rapid_chk, z_min_s, z_max_s,
                    p_start, p_end, mutate=False,
                )
                d_chord = math.dist(p_start, p_end)
                if direct_probe_v <= 0.005 and d_chord > 1e-6 and (d_chord / rapid_mm_s + 0.18) < orig_air_t:
                    shortcut_segs.append(
                        C.Segment(
                            id=f"sc_r{idx:05d}",
                            kind=C.SegmentKind.RAPID,
                            start_mm=p_start,
                            end_mm=p_end,
                            feed_mm_s=None,
                            duration_s=d_chord / rapid_mm_s,
                        )
                    )
                else:
                    p_up_a = (p_start[0], p_start[1], z_safe_mm)
                    p_up_b = (p_end[0], p_end[1], z_safe_mm)
                    up_probe = _sweep_removes_or_updates_dexels(
                        xs, ys, z_guard, x0_g, y0_g, pitch, r_rapid_chk, z_min_s, z_max_s,
                        p_start, p_up_a, mutate=False,
                    )
                    top_probe = _sweep_removes_or_updates_dexels(
                        xs, ys, z_guard, x0_g, y0_g, pitch, r_rapid_chk, z_min_s, z_max_s,
                        p_up_a, p_up_b, mutate=False,
                    )
                    bridge_t = (
                        (math.dist(p_start, p_up_a) + math.dist(p_up_a, p_up_b)) / rapid_mm_s
                        + math.dist(p_up_b, p_end) / max(base_feed_mm_s, 1e-6)
                        + 0.45
                    )
                    if air_len >= 22.0 and up_probe <= 0.005 and top_probe <= 0.005 and bridge_t < orig_air_t:
                        for pa_r, pb_r, tag_s in ((p_start, p_up_a, "u"), (p_up_a, p_up_b, "b"), (p_up_b, p_end, "d")):
                            if math.dist(pa_r, pb_r) > 1e-6:
                                shortcut_segs.append(
                                    C.Segment(
                                        id=f"sc_{tag_s}{idx:05d}",
                                        kind=C.SegmentKind.RAPID if tag_s != "d" else C.SegmentKind.FEED,
                                        start_mm=pa_r,
                                        end_mm=pb_r,
                                        feed_mm_s=None if tag_s != "d" else base_feed_mm_s,
                                        duration_s=math.dist(pa_r, pb_r) / (rapid_mm_s if tag_s != "d" else base_feed_mm_s),
                                    )
                                )
                    else:
                        for m_i in range(idx, j):
                            shortcut_segs.append(sub_segs[m_i])
            else:
                for m_i in range(idx, j):
                    shortcut_segs.append(sub_segs[m_i])
            idx = j
        else:
            shortcut_segs.append(sub_segs[idx])
            idx += 1

    cand_same_plan = _make_motion_plan_from_segs(
        [
            dataclasses.replace(s, feed_mm_s=None if s.kind == C.SegmentKind.RAPID else round(base_feed_mm_s, 4))
            for s in shortcut_segs
        ],
        z_safe_mm,
        rapid_mm_s,
        provenance="diffcam:same_speed_arbitrary",
    )
    if sum(s.duration_s for s in cand_same_plan.segments) <= sum(s.duration_s for s in base_sub_plan.segments) + 1e-6:
        same_speed_plan = cand_same_plan
    else:
        shortcut_segs = list(sub_segs)
        same_speed_plan = base_sub_plan

    # Re-measure exact CWE on shortcut_segs and advance persistent z_ipw
    m_sc = len(shortcut_segs)
    sc_len = np.zeros(m_sc, dtype=np.float64)
    sc_rem = np.zeros(m_sc, dtype=np.float64)
    sc_ae = np.zeros(m_sc, dtype=np.float64)
    sc_feed_mask = np.zeros(m_sc, dtype=bool)

    for k, seg in enumerate(shortcut_segs):
        ln = math.dist(seg.start_mm, seg.end_mm)
        sc_len[k] = ln
        if seg.kind == C.SegmentKind.RAPID or ln <= 1e-8:
            continue
        sc_feed_mask[k] = True
        z_before = z_ipw.copy()
        _sweep_removes_or_updates_dexels(
            xs, ys, z_ipw, x0_g, y0_g, pitch, cutter_radius_mm, z_min_s, z_max_s,
            seg.start_mm, seg.end_mm, mutate=True,
        )
        dz = np.maximum(0.0, z_before - z_ipw)
        rv = float(np.sum(dz)) * cell_area
        sc_rem[k] = rv
        if rv > 1e-4:
            ap_e = float(np.percentile(dz[dz > 1e-5], 80))
            lxy = math.hypot(seg.end_mm[0] - seg.start_mm[0], seg.end_mm[1] - seg.start_mm[1])
            sc_ae[k] = min(diam, rv / max(ap_e * (lxy + 0.60 * diam), 1e-6)) if lxy > 0.12 else diam

    # 4. Differentiable CWE Feed/Chipload Optimization via Adam over log-feed eta_k
    tooth_freq = max(1.0, (spindle_rpm / 60.0) * max(1, flutes))
    u_ae = np.clip(sc_ae / max(diam, 1e-6), 0.0, 1.0)
    # Radial chip-thinning effective sin(phi_eff): sqrt(1 - (1 - 2 u)^2) for u < 0.5, 1.0 for u >= 0.5
    sin_eff = np.ones(m_sc, dtype=np.float64)
    thin_mask = (u_ae > 1e-4) & (u_ae < 0.5)
    uu = 1.0 - 2.0 * u_ae[thin_mask]
    sin_eff[thin_mask] = np.maximum(0.32, np.sqrt(np.maximum(1.0 - uu * uu, 1e-6)))

    v_lo = max(1.5, 0.45 * base_feed_mm_s)
    v_hi = max(base_feed_mm_s * 1.40, min(max_feed_mm_s * 1.35, 2.65 * base_feed_mm_s))
    v_air = min(rapid_mm_s * 0.72, max(v_hi, 2.8 * base_feed_mm_s))

    # Precompute exact segment tangent vectors u_vec & effective lengths for G187/G64 blended timing gradient
    eps_l = C.BLENDED_EPS_LEN_MM
    eps_v = C.BLENDED_EPS_VEL_MM_S
    max_accel = 150.0
    inv_2a = 0.5 / max_accel
    d_xyz = np.array(
        [[s.end_mm[0] - s.start_mm[0], s.end_mm[1] - s.start_mm[1], s.end_mm[2] - s.start_mm[2]] for s in shortcut_segs],
        dtype=np.float64,
    )
    norm_eps = np.sqrt(np.sum(d_xyz * d_xyz, axis=1) + eps_l * eps_l)
    eff_len = norm_eps - eps_l
    u_vec = d_xyz / norm_eps[:, None]

    eta = np.zeros(m_sc, dtype=np.float64)
    m_adam = np.zeros_like(eta)
    v_adam = np.zeros_like(eta)
    eng_mask = sc_feed_mask & (sc_rem > 1e-4)
    air_feed_mask = sc_feed_mask & (~eng_mask)

    h_nom_ref = (base_feed_mm_s / tooth_freq) * 0.86
    h_star = max(0.005, float(target_hex_mm), h_nom_ref)
    w_chip = 0.85
    w_overload = 3.0
    w_mrr = 2.8
    w_smooth = 0.04
    lr = 0.12

    for it in range(1, max(1, int(opt_iters)) + 1):
        spd_all = np.where(sc_feed_mask, np.clip(base_feed_mm_s * np.exp(eta), v_lo, v_air), rapid_mm_s)
        spd_all[eng_mask] = np.minimum(spd_all[eng_mask], v_hi)

        # Exact d(T_blended) / d(spd_k) across all junctions (Haas G187 / RS274 G64)
        vel_vec = spd_all[:, None] * u_vec
        v_prev = np.zeros_like(vel_vec)
        if m_sc > 1:
            v_prev[1:] = vel_vec[:-1]
        dv = vel_vec - v_prev
        dn = np.sqrt(np.sum(dv * dv, axis=1) + eps_v * eps_v)

        d_tblend_d_spd = -eff_len / np.maximum(spd_all * spd_all, 1e-9)
        # Contribution from junction k: ||V_k - V_{k-1}||
        d_tblend_d_spd += inv_2a * np.sum(dv * u_vec, axis=1) / dn
        # Contribution from junction k+1: ||V_{k+1} - V_k||
        if m_sc > 1:
            d_tblend_d_spd[:-1] -= inv_2a * np.sum(dv[1:] * u_vec[:-1], axis=1) / dn[1:]
        # Final segment stop deceleration
        vn_last = math.sqrt(float(np.sum(vel_vec[-1] * vel_vec[-1])) + eps_v * eps_v)
        d_tblend_d_spd[-1] += inv_2a * float(np.sum(vel_vec[-1] * u_vec[-1])) / vn_last

        # Chain rule d/d(eta_k) = spd_k * d/d(spd_k)
        grad = np.zeros(m_sc, dtype=np.float64)
        grad[sc_feed_mask] = spd_all[sc_feed_mask] * d_tblend_d_spd[sc_feed_mask]

        if np.any(eng_mask):
            vk = spd_all[eng_mask]
            lk = sc_len[eng_mask]
            rk = sc_rem[eng_mask]
            sk = sin_eff[eng_mask]

            # Asymmetric chipload tracking: gentle tracking below h_star, strong protection above 1.10 * h_star
            hex_k = (vk / tooth_freq) * sk
            diff_h = (hex_k - h_star) / h_star
            over_h = np.maximum(0.0, hex_k / (1.08 * h_star) - 1.0)
            g_chip = (
                w_chip * (lk / max(base_feed_mm_s, 1e-6)) * 2.0 * diff_h * (hex_k / h_star)
                + w_overload * (lk / max(base_feed_mm_s, 1e-6)) * 2.0 * over_h * (hex_k / (1.08 * h_star))
            )

            # MRR spindle-load cap (accounting for swept circular endcap length lk + 0.60 * D)
            mrr_k = (rk / np.maximum(lk + 0.60 * diam, 1e-6)) * vk
            over_mrr = np.maximum(0.0, mrr_k / max(mrr_limit_mm3_s, 1.0) - 1.0)
            g_mrr = w_mrr * 2.0 * over_mrr * (mrr_k / max(mrr_limit_mm3_s, 1.0))

            grad[eng_mask] += g_chip + g_mrr
        _ = air_feed_mask

        # Feed smoothness penalty along consecutive feed segments
        if m_sc >= 2:
            both_f = sc_feed_mask[:-1] & sc_feed_mask[1:]
            d_eta = (eta[1:] - eta[:-1]) * both_f
            grad[:-1] -= 2.0 * w_smooth * d_eta
            grad[1:] += 2.0 * w_smooth * d_eta

        m_adam = 0.9 * m_adam + 0.1 * grad
        v_adam = 0.999 * v_adam + 0.001 * (grad * grad)
        mh = m_adam / (1.0 - 0.9 ** it)
        vh = v_adam / (1.0 - 0.999 ** it)
        eta = np.clip(
            eta - lr * mh / (np.sqrt(vh) + 1e-8),
            math.log(v_lo / max(base_feed_mm_s, 1e-6)),
            math.log(v_air / max(base_feed_mm_s, 1e-6)),
        )

    v_opt = np.clip(base_feed_mm_s * np.exp(eta), v_lo, v_air)
    v_opt[eng_mask] = np.minimum(v_opt[eng_mask], v_hi)

    opt_segs = [
        dataclasses.replace(
            s,
            feed_mm_s=None if s.kind == C.SegmentKind.RAPID else round(float(v_opt[k]), 4),
        )
        for k, s in enumerate(shortcut_segs)
    ]
    opt_plan = _make_motion_plan_from_segs(opt_segs, z_safe_mm, rapid_mm_s, provenance="diffcam:opt_chipload_arbitrary")
    _ = z_target_safe
    return naive_plan, same_speed_plan, opt_plan, 0.0
