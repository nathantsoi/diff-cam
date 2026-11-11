# SPDX-License-Identifier: MIT
"""Clean-room differentiable 2.5D Cutter-Workpiece Engagement (CWE) & Chipload Co-Optimization.

Addresses two fundamental limitations of coarse segment-level MRR optimization:
1. **Geometric Trajectory Locking**: Un-subdivided long CAM passes (20-39 mm) with
   pinned plunge/retract columns and tight spline bounds cannot locally round 90 deg
   roughing corners, retract wasted exterior air-overtravel loops, or split a pass
   between high-speed air traverse and regulated cutting.
2. **Radial Chip Thinning & Local Engagement Spikes**: Segment-averaged MRR misses
   local corner immersion spikes (``a_e -> D``) and radial chip thinning at light
   peripheral stepovers (``a_e < 0.5 D``), where effective maximum undeformed chip
   thickness is ``h_ex = f_z * 2 * sqrt((a_e/D) * (1 - a_e/D))``.

Provides:
- ``analyze_plan_engagement``: Exact 0.25 mm 2.5D dexel + sub-segment station analyzer
  computing engaged vs. air-cutting G1 time/distance, rapid G0 time/distance, removed
  volume, radial engagement ``a_e`` (%D), wrap angle ``phi_eng``, ``f_z``, and ``h_ex``.
- ``SubsegmentedCWEModel``: Sub-segmented trajectory parameterization with coupled
  vertical-column XY mobility, triangle-inequality blueprint wall safety, and exact
  chronological reverse-mode AD through smooth/STE 2.5D dexel removal + blended timing.
- ``optimize_chipload_trajectory`` (``DiffCAM-Chipload``): First-order STE-Adam joint
  spatial + feed co-optimizer regulating ``h_ex -> h_target`` while minimizing blended
  cycle time, air travel, and high-engagement corner spikes.
- ``LocalCWEPolicy``, ``train_chipload_shac``, ``train_chipload_ppo``: Controlled
  first-order (SHAC) vs. zero-order (PPO) RL policies conditioned on local IPW CWE
  features for zero-shot cross-geometry chipload & cycle-time optimization.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmarks import contracts as C
from benchmarks.milling_suite import MillingTaskBundle, multi_pocket_fields
from cam.feeds_and_speeds import (
    STOCK_MATERIALS,
    CutterGeometrySpec,
    MachineEnvelopeSpec,
    compute_feeds_and_speeds,
    infer_case_nominal_doc_woc_mm,
)


@dataclass(frozen=True)
class ChiploadSpec:
    """Physical spindle, endmill, and target chipload specification."""

    spindle_rpm: float = 8000.0
    flutes: int = 3
    cutter_diameter_mm: float = 5.0
    target_hex_mm: float = 0.036
    min_hex_mm: float = 0.020
    max_hex_mm: float = 0.048
    ideal_woc_frac: float = 0.35
    slot_ipt_factor: float = 0.80
    base_feed_mm_s: float = 16.0
    stock_material_id: str = "al_6061_t6"
    tool_material_id: str = "carbide_coated"
    aggressiveness: float = 1.0
    subdivide_max_mm: float = 2.5
    dexel_pitch_mm: float = 0.25
    station_step_mm: float = 0.20
    max_xy_shift_mm: float = 3.5
    wall_safety_margin_mm: float = 0.16
    smooth_footprint_mm: float = 0.28
    u_spike_cap: float = 0.58
    w_time: float = 1.0
    w_matched_time: float = 0.85
    w_chipload: float = 2.20
    w_hex_overload: float = 4.00
    w_spike: float = 0.65
    w_mrr: float = 0.80
    w_smooth_xy: float = 0.04
    w_residual: float = 2.50

    @property
    def tooth_freq_hz(self) -> float:
        return (self.spindle_rpm * float(self.flutes)) / 60.0

    @classmethod
    def from_feeds_and_speeds(
        cls,
        bundle: MillingTaskBundle,
        stock_material_id: str = "al_6061_t6",
        tool_material_id: str = "carbide_coated",
        aggressiveness: float = 1.0,
        flutes: int = 3,
    ) -> "ChiploadSpec":
        """Construct physics-grounded ``ChiploadSpec`` from tool, stock, DOC, and WOC."""
        tool = bundle.task.tool
        d_mm = 2.0 * float(tool.cutter_radius_mm)
        cutter = CutterGeometrySpec(
            diameter_mm=d_mm,
            flutes=flutes,
            flute_length_mm=float(tool.flute_length_mm),
            stickout_mm=float(tool.holder_bottom_mm),
            shank_diameter_mm=2.0 * float(tool.shank_radius_mm),
            corner_radius_mm=float(getattr(bundle.case, "tool_corner_radius_mm", 0.0)),
            helix_deg=35.0,
            slot_ipt_factor=0.80,
        )
        machine = MachineEnvelopeSpec(
            max_rpm=8000.0,
            max_feed_mm_s=float(bundle.task.limits.max_feed_mm_s),
            rapid_xy_mm_s=float(bundle.task.limits.rapid_mm_s),
        )
        doc_mm, woc_mm = infer_case_nominal_doc_woc_mm(
            float(bundle.case.stock_max_mm[2]),
            float(bundle.case.pocket_min_mm[2]),
            d_mm,
            bundle.case.family,
        )
        fs = compute_feeds_and_speeds(
            stock_material_id=stock_material_id,
            tool_material_id=tool_material_id,
            cutter=cutter,
            machine=machine,
            doc_mm=doc_mm,
            woc_mm=woc_mm,
            aggressiveness=aggressiveness,
        )
        stock_spec = STOCK_MATERIALS.get(stock_material_id, STOCK_MATERIALS["al_6061_t6"])
        h_target = max(0.004, float(fs.chip_thickness_mm))
        return cls(
            spindle_rpm=float(fs.spindle_rpm),
            flutes=int(flutes),
            cutter_diameter_mm=d_mm,
            target_hex_mm=h_target,
            min_hex_mm=round(0.56 * h_target, 6),
            max_hex_mm=round(1.26 * h_target, 6),
            ideal_woc_frac=float(stock_spec.ideal_woc_frac),
            slot_ipt_factor=float(cutter.slot_ipt_factor),
            base_feed_mm_s=float(min(fs.feed_mm_s, bundle.task.limits.max_feed_mm_s * 0.99)),
            stock_material_id=stock_spec.id,
            tool_material_id=fs.tool_material_id,
            aggressiveness=float(fs.aggressiveness),
        )


def _slot_factor_and_grad(
    u_ae_over_d: np.ndarray,
    ideal_woc_frac: float = 0.35,
    slot_ipt_factor: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Differentiable slotting IPT reduction ``slot_factor(u_ae)`` and ``d_slot / d_u``."""
    u = np.clip(np.asarray(u_ae_over_d, dtype=np.float64), 0.0, 1.0)
    u0 = float(min(max(ideal_woc_frac, 0.02), 0.95))
    span = max(1.0 - u0, 1e-6)
    t = np.clip((u - u0) / span, 0.0, 1.0)
    sf = 1.0 + t * (float(slot_ipt_factor) - 1.0)
    dsf = np.where((u > u0) & (u < 1.0), (float(slot_ipt_factor) - 1.0) / span, 0.0)
    return sf, dsf


def radial_chip_thinning_factor(
    u_ae_over_d: np.ndarray | float,
    eps: float = 1e-3,
) -> np.ndarray:
    """Smooth radial chip-thinning factor ``psi_thin(a_e / D) in [0, 1]``.

    For ``u = a_e / D < 0.5``, ``h_ex / f_z = 2 * sqrt(u * (1 - u))``;
    for ``u >= 0.5`` (slotting / >90 deg wrap), ``h_ex / f_z = 1.0``.
    """
    u = np.clip(np.asarray(u_ae_over_d, dtype=np.float64), 0.0, 1.0)
    u_eff = np.minimum(u, 0.5)
    return 2.0 * np.sqrt(np.maximum(u_eff * (1.0 - u_eff), 0.0) + eps * eps) / math.sqrt(
        1.0 + 4.0 * eps * eps
    )


def _radial_chip_thinning_and_grad(
    u_ae_over_d: np.ndarray,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(psi_thin(u), d_psi_thin / d_u)`` for ``u in [0, 1]^M``."""
    u = np.clip(np.asarray(u_ae_over_d, dtype=np.float64), 0.0, 1.0)
    norm_c = math.sqrt(1.0 + 4.0 * eps * eps)
    u_eff = np.minimum(u, 0.5)
    rad = np.maximum(u_eff * (1.0 - u_eff), 0.0) + eps * eps
    sqrt_rad = np.sqrt(rad)
    psi = (2.0 * sqrt_rad) / norm_c
    dpsi = np.where(
        (u > 0.0) & (u < 0.5),
        (1.0 - 2.0 * u_eff) / ( sqrt_rad * norm_c ),
        0.0,
    )
    return psi, dpsi


def build_target_heightfield_25d(
    bundle: MillingTaskBundle,
    pitch_mm: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact 2.5D cell centers ``(xs, ys)`` and ``(z_init, z_target)`` in ``C.Frame.WORK``."""
    smin = bundle.case.stock_min_mm
    smax = bundle.case.stock_max_mm
    nx = max(4, int(round((smax[0] - smin[0]) / pitch_mm)))
    ny = max(4, int(round((smax[1] - smin[1]) / pitch_mm)))
    nz = max(4, int(round((smax[2] - smin[2]) / pitch_mm)))
    grid = C.GridSpec(
        shape=(nx, ny, nz),
        origin_mm=smin,
        voxel_mm=float(pitch_mm),
        frame=C.Frame.WORK,
    )
    _, target_sdf = multi_pocket_fields(
        bundle.task.stock, bundle.task.operation_target.geometry, grid
    )
    xs = smin[0] + (np.arange(nx, dtype=np.float64) + 0.5) * pitch_mm
    ys = smin[1] + (np.arange(ny, dtype=np.float64) + 0.5) * pitch_mm
    occ = target_sdf <= 0.0  # (nx, ny, nz)
    has_mat = np.any(occ, axis=2)
    # Highest occupied voxel top z = smin[2] + (k_max + 1) * pitch_mm
    rev_idx = np.argmax(occ[:, :, ::-1], axis=2)
    k_top = (nz - 1) - rev_idx
    z_target = np.where(has_mat, smin[2] + (k_top + 1).astype(np.float64) * pitch_mm, smin[2])
    z_init = np.full((nx, ny), float(smax[2]), dtype=np.float64)
    return xs, ys, z_init, z_target


def analyze_plan_engagement(
    bundle: MillingTaskBundle,
    plan: C.MotionPlan,
    spec: Optional[ChiploadSpec] = None,
) -> Dict[str, Any]:
    """Compute high-resolution per-segment & summary engaged/air/CWE/chipload telemetry.

    Evaluates exact closed-form 3D flat-endmill sweeps over a ``0.25 mm`` 2.5D dexel grid
    with ``0.20 mm`` sub-segment stations so partial air lead-in/lead-out on long passes
    is separated accurately from material-removing travel.
    """
    cfg = spec or ChiploadSpec(cutter_diameter_mm=2.0 * float(bundle.task.tool.cutter_radius_mm))
    pitch = float(cfg.dexel_pitch_mm)
    cell_area = pitch * pitch
    xs, ys, z_stock, z_target = build_target_heightfield_25d(bundle, pitch_mm=pitch)
    nx, ny = z_stock.shape
    x0_grid = float(bundle.case.stock_min_mm[0])
    y0_grid = float(bundle.case.stock_min_mm[1])
    z_min_stock = float(bundle.case.stock_min_mm[2])

    r_cut = float(bundle.task.tool.cutter_radius_mm)
    diam = 2.0 * r_cut
    r2 = r_cut * r_cut
    tooth_freq = cfg.tooth_freq_hz
    inv = bundle.task.work_to_machine.inverse().apply

    seg_records: List[Dict[str, Any]] = []
    total_engaged_time = 0.0
    total_air_cut_time = 0.0
    total_rapid_time = 0.0
    total_engaged_dist = 0.0
    total_air_cut_dist = 0.0
    total_rapid_dist = 0.0
    total_removed_mm3 = 0.0

    rubbing_time = 0.0
    optimal_hex_time = 0.0
    overload_hex_time = 0.0
    weighted_hex_sum = 0.0
    weighted_hex_sq_err = 0.0
    weighted_ae_sum = 0.0
    weighted_ap_sum = 0.0
    peak_ae_mm = 0.0
    peak_ap_mm = 0.0
    peak_hex_mm = 0.0
    peak_inst_mrr = 0.0

    for seg in plan.segments:
        sw = inv(seg.start_mm)
        ew = inv(seg.end_mm)
        ax, ay, az = float(sw[0]), float(sw[1]), float(sw[2])
        bx, by, bz = float(ew[0]), float(ew[1]), float(ew[2])
        dx, dy, dz = bx - ax, by - ay, bz - az
        len_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        len_xy = math.hypot(dx, dy)
        dur = float(seg.duration_s)
        is_rapid = seg.kind == C.SegmentKind.RAPID
        feed_s = float(
            bundle.task.limits.rapid_mm_s
            if is_rapid or seg.feed_mm_s is None
            else seg.feed_mm_s
        )

        if is_rapid or len_3d <= 1e-9:
            if is_rapid:
                total_rapid_time += dur
                total_rapid_dist += len_3d
            else:
                total_air_cut_time += dur
            seg_records.append(
                {
                    "engaged": False,
                    "engaged_fraction": 0.0,
                    "engaged_time_s": 0.0,
                    "air_cut_time_s": 0.0 if is_rapid else round(dur, 6),
                    "rapid_time_s": round(dur, 6) if is_rapid else 0.0,
                    "removed_mm3": 0.0,
                    "mrr_mm3_s": 0.0,
                    "ap_mm": 0.0,
                    "ae_mm": 0.0,
                    "ae_pct": 0.0,
                    "phi_eng_deg": 0.0,
                    "fz_mm": 0.0,
                    "hex_mm": 0.0,
                }
            )
            continue

        # Sub-step along the cutting segment to separate engaged vs. air stations accurately
        n_sub = max(1, int(math.ceil(len_3d / max(cfg.station_step_mm, 0.10))))
        sub_dur = dur / n_sub
        sub_len_3d = len_3d / n_sub
        sub_len_xy = len_xy / n_sub

        seg_removed = 0.0
        seg_eng_subs = 0
        seg_ae_vals: List[float] = []
        seg_ap_vals: List[float] = []
        fz = feed_s / max(tooth_freq, 1e-6)

        for m in range(n_sub):
            u0 = m / n_sub
            u1 = (m + 1) / n_sub
            sx0, sy0, sz0 = ax + u0 * dx, ay + u0 * dy, az + u0 * dz
            sx1, sy1, sz1 = ax + u1 * dx, ay + u1 * dy, az + u1 * dz

            z_lo_seg = min(sz0, sz1)
            if z_lo_seg >= float(bundle.case.stock_max_mm[2]) - 1e-7:
                continue

            ix0 = max(0, int(math.floor((min(sx0, sx1) - r_cut - x0_grid) / pitch)))
            ix1 = min(nx, int(math.ceil((max(sx0, sx1) + r_cut - x0_grid) / pitch)) + 1)
            iy0 = max(0, int(math.floor((min(sy0, sy1) - r_cut - y0_grid) / pitch)))
            iy1 = min(ny, int(math.ceil((max(sy0, sy1) + r_cut - y0_grid) / pitch)) + 1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue

            px = xs[ix0:ix1, None]
            py = ys[None, iy0:iy1]
            sdx = sx1 - sx0
            sdy = sy1 - sy0
            sdz = sz1 - sz0
            lxy2 = sdx * sdx + sdy * sdy
            if lxy2 <= 1e-12:
                dist2 = (px - sx0) ** 2 + (py - sy0) ** 2
                inside = dist2 <= r2
                z_swept = np.full(dist2.shape, max(min(sz0, sz1), z_min_stock), dtype=np.float64)
            else:
                uc = ((px - sx0) * sdx + (py - sy0) * sdy) / lxy2
                d_perp2 = (px - sx0 - uc * sdx) ** 2 + (py - sy0 - uc * sdy) ** 2
                valid_perp = d_perp2 <= r2
                du = np.sqrt(np.maximum(r2 - d_perp2, 0.0) / lxy2)
                u_lo = np.maximum(0.0, uc - du)
                u_hi = np.minimum(1.0, uc + du)
                inside = valid_perp & (u_lo <= u_hi)
                u_best = u_hi if sdz < 0.0 else u_lo
                z_swept = np.maximum(sz0 + u_best * sdz, z_min_stock)

            cur_patch = z_stock[ix0:ix1, iy0:iy1]
            dz_cut = np.where(inside & (z_swept < cur_patch), cur_patch - z_swept, 0.0)
            dv = float(np.sum(dz_cut)) * cell_area
            if dv > 1e-6:
                z_stock[ix0:ix1, iy0:iy1] = cur_patch - dz_cut
                seg_removed += dv
                seg_eng_subs += 1
                # Effective axial depth & radial width on this station
                max_ap = float(np.max(dz_cut))
                if sub_len_xy > 1e-4 and max_ap > 1e-4:
                    ae_sub = min(diam, dv / (max_ap * max(sub_len_xy, 0.20)))
                else:
                    # Plunge / pure axial entry: full diameter engagement
                    ae_sub = diam
                seg_ae_vals.append(ae_sub)
                seg_ap_vals.append(max_ap)
                peak_ae_mm = max(peak_ae_mm, ae_sub)
                peak_ap_mm = max(peak_ap_mm, max_ap)
                inst_mrr = dv / max(sub_dur, 1e-7)
                peak_inst_mrr = max(peak_inst_mrr, inst_mrr)

        eng_frac = seg_eng_subs / float(n_sub)
        eng_t = dur * eng_frac
        air_t = dur * (1.0 - eng_frac)
        eng_d = len_3d * eng_frac
        air_d = len_3d * (1.0 - eng_frac)

        total_engaged_time += eng_t
        total_air_cut_time += air_t
        total_engaged_dist += eng_d
        total_air_cut_dist += air_d
        total_removed_mm3 += seg_removed

        if seg_ae_vals:
            ae_mean = float(np.mean(seg_ae_vals))
            ap_mean = float(np.mean(seg_ap_vals))
            u_ae = min(1.0, max(0.0, ae_mean / diam))
            phi_deg = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * u_ae))))
            psi = float(radial_chip_thinning_factor(u_ae))
            hex_val = fz * psi
            peak_hex_mm = max(peak_hex_mm, hex_val)

            weighted_ae_sum += ae_mean * eng_t
            weighted_ap_sum += ap_mean * eng_t
            weighted_hex_sum += hex_val * eng_t
            weighted_hex_sq_err += ((hex_val - cfg.target_hex_mm) ** 2) * eng_t
            if hex_val < cfg.min_hex_mm:
                rubbing_time += eng_t
            elif hex_val > cfg.max_hex_mm:
                overload_hex_time += eng_t
            else:
                optimal_hex_time += eng_t
        else:
            ae_mean = 0.0
            ap_mean = 0.0
            u_ae = 0.0
            phi_deg = 0.0
            hex_val = 0.0

        seg_mrr = seg_removed / max(dur, 1e-6)
        seg_records.append(
            {
                "engaged": seg_eng_subs > 0,
                "engaged_fraction": round(eng_frac, 4),
                "engaged_time_s": round(eng_t, 5),
                "air_cut_time_s": round(air_t, 5),
                "rapid_time_s": 0.0,
                "removed_mm3": round(seg_removed, 4),
                "mrr_mm3_s": round(seg_mrr, 3),
                "ap_mm": round(ap_mean, 4),
                "ae_mm": round(ae_mean, 4),
                "ae_pct": round(100.0 * u_ae, 2),
                "phi_eng_deg": round(phi_deg, 2),
                "fz_mm": round(fz, 5),
                "hex_mm": round(hex_val, 5),
            }
        )

    total_time = total_engaged_time + total_air_cut_time + total_rapid_time
    denom_t = max(total_time, 1e-9)
    denom_eng = max(total_engaged_time, 1e-9)
    mean_ae_mm = weighted_ae_sum / denom_eng if total_engaged_time > 1e-6 else 0.0
    mean_ap_mm = weighted_ap_sum / denom_eng if total_engaged_time > 1e-6 else 0.0
    mean_hex_mm = weighted_hex_sum / denom_eng if total_engaged_time > 1e-6 else 0.0
    hex_rmse_mm = math.sqrt(weighted_hex_sq_err / denom_eng) if total_engaged_time > 1e-6 else 0.0
    dexel_residual_mm3 = float(np.sum(np.maximum(0.0, z_stock - z_target))) * cell_area
    dexel_gouge_mm3 = float(np.sum(np.maximum(0.0, z_target - z_stock))) * cell_area

    summary = {
        "total_time_s": round(total_time, 4),
        "engaged_time_s": round(total_engaged_time, 4),
        "air_cut_time_s": round(total_air_cut_time, 4),
        "rapid_time_s": round(total_rapid_time, 4),
        "engaged_pct": round(100.0 * total_engaged_time / denom_t, 2),
        "air_cut_pct": round(100.0 * total_air_cut_time / denom_t, 2),
        "rapid_pct": round(100.0 * total_rapid_time / denom_t, 2),
        "non_cutting_pct": round(100.0 * (total_air_cut_time + total_rapid_time) / denom_t, 2),
        "engaged_dist_mm": round(total_engaged_dist, 3),
        "air_cut_dist_mm": round(total_air_cut_dist, 3),
        "rapid_dist_mm": round(total_rapid_dist, 3),
        "removed_volume_mm3": round(total_removed_mm3, 3),
        "mean_ap_mm": round(mean_ap_mm, 4),
        "peak_ap_mm": round(peak_ap_mm, 4),
        "mean_ae_mm": round(mean_ae_mm, 4),
        "mean_ae_pct": round(100.0 * mean_ae_mm / diam, 2),
        "peak_ae_mm": round(peak_ae_mm, 4),
        "peak_ae_pct": round(100.0 * peak_ae_mm / diam, 2),
        "mean_hex_mm": round(mean_hex_mm, 5),
        "peak_hex_mm": round(peak_hex_mm, 5),
        "target_hex_mm": round(cfg.target_hex_mm, 5),
        "hex_rmse_mm": round(hex_rmse_mm, 5),
        "optimal_hex_pct": round(100.0 * optimal_hex_time / denom_eng, 2) if total_engaged_time > 1e-6 else 0.0,
        "rubbing_pct": round(100.0 * rubbing_time / denom_eng, 2) if total_engaged_time > 1e-6 else 0.0,
        "overload_hex_pct": round(100.0 * overload_hex_time / denom_eng, 2) if total_engaged_time > 1e-6 else 0.0,
        "peak_inst_mrr_mm3_s": round(peak_inst_mrr, 2),
        "dexel_residual_mm3": round(dexel_residual_mm3, 3),
        "dexel_gouge_mm3": round(dexel_gouge_mm3, 4),
        "spindle_rpm": cfg.spindle_rpm,
        "flutes": cfg.flutes,
    }
    return {"summary": summary, "segments": seg_records}


def subdivide_motion_plan(
    task: C.TaskSpec,
    motion: C.MotionPlan,
    max_cut_seg_mm: float = 2.5,
) -> C.MotionPlan:
    """Subdivide long horizontal/ramping ``G1`` segments into ``<= max_cut_seg_mm`` stations."""
    new_segs: List[C.Segment] = []
    sid = 0
    for seg in motion.segments:
        sx, sy, sz = seg.start_mm
        ex, ey, ez = seg.end_mm
        xy_len = math.hypot(ex - sx, ey - sy)
        if seg.kind == C.SegmentKind.RAPID or xy_len <= max_cut_seg_mm + 1e-6:
            new_segs.append(
                dataclasses.replace(
                    seg,
                    id=f"s{sid:04d}",
                )
            )
            sid += 1
        else:
            n_sub = max(2, int(math.ceil(xy_len / max_cut_seg_mm)))
            for m in range(n_sub):
                u0 = m / n_sub
                u1 = (m + 1) / n_sub
                p0 = (sx + u0 * (ex - sx), sy + u0 * (ey - sy), sz + u0 * (ez - sz))
                p1 = (sx + u1 * (ex - sx), sy + u1 * (ey - sy), sz + u1 * (ez - sz))
                sub_len = math.dist(p0, p1)
                feed = float(seg.feed_mm_s)
                new_segs.append(
                    C.Segment(
                        id=f"s{sid:04d}",
                        kind=seg.kind,
                        start_mm=p0,
                        end_mm=p1,
                        duration_s=max(sub_len / feed, 1e-9),
                        feed_mm_s=feed,
                        tag=seg.tag,
                    )
                )
                sid += 1

    if motion.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        durs = C.blended_segment_durations(
            new_segs, task.limits.rapid_mm_s, task.limits.max_accel_mm_s2
        )
        new_segs = [dataclasses.replace(s, duration_s=float(d)) for s, d in zip(new_segs, durs)]

    out = C.MotionPlan(
        frame=motion.frame,
        initial_state=motion.initial_state,
        segments=tuple(new_segs),
        timing_rule=motion.timing_rule,
        provenance=f"{motion.provenance}:subdivided",
    )
    out.validate_for_task(task)
    return out


def _pt_to_seg_2d(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 1e-18:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _seg_intersects_box_2d(
    ax: float, ay: float, bx: float, by: float, x0: float, y0: float, x1: float, y1: float
) -> bool:
    t0, t1 = 0.0, 1.0
    dx, dy = bx - ax, by - ay
    for p_val, q_val in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
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


def _segment_wall_clearance_mm(task: C.TaskSpec, pa: np.ndarray, pb: np.ndarray) -> float:
    """Exact 2D distance from segment ``pa->pb`` to vertical blueprint walls minus cutter radius."""
    r_tool = float(task.tool.cutter_radius_mm)
    part = task.original_part
    if not (part.kind == "analytic" and part.params.get("primitive") == "box_minus_multi_pocket"):
        return 0.0
    smin = part.params["box_min_mm"]
    smax = part.params["box_max_mm"]
    pmin = part.params["pocket_min_mm"]
    pmax = part.params["pocket_max_mm"]
    islands = part.params.get("islands_mm", ())

    ax, ay, az = float(pa[0]), float(pa[1]), float(pa[2])
    bx, by, bz = float(pb[0]), float(pb[1]), float(pb[2])
    z_low = min(az, bz)
    if z_low >= float(smax[2]) - 1e-6:
        return 100.0

    d_min = 100.0
    if pmin[0] > smin[0] + 1e-6:
        d_min = min(d_min, max(0.0, min(ax, bx) - float(pmin[0])))
    if pmax[0] < smax[0] - 1e-6:
        d_min = min(d_min, max(0.0, float(pmax[0]) - max(ax, bx)))
    if pmin[1] > smin[1] + 1e-6:
        d_min = min(d_min, max(0.0, min(ay, by) - float(pmin[1])))
    if pmax[1] < smax[1] - 1e-6:
        d_min = min(d_min, max(0.0, float(pmax[1]) - max(ay, by)))

    for imin, imax in islands:
        if float(imax[2]) > z_low + 1e-6:
            ix0, iy0 = float(imin[0]), float(imin[1])
            ix1, iy1 = float(imax[0]), float(imax[1])
            if _seg_intersects_box_2d(ax, ay, bx, by, ix0, iy0, ix1, iy1):
                return -r_tool
            for qx, qy in ((ax, ay), (bx, by)):
                dx = max(0.0, ix0 - qx, qx - ix1)
                dy = max(0.0, iy0 - qy, qy - iy1)
                d_min = min(d_min, math.hypot(dx, dy))
            for cx, cy in ((ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1)):
                d_min = min(d_min, _pt_to_seg_2d(cx, cy, ax, ay, bx, by))

    return d_min - r_tool


def blended_durations_with_pullback(
    deltas_mm: np.ndarray,
    speeds_mm_s: np.ndarray,
    max_accel_mm_s2: float,
    g_tau: Optional[np.ndarray] = None,
    *,
    eps_len_mm: float = C.BLENDED_EPS_LEN_MM,
    eps_vel_mm_s: float = C.BLENDED_EPS_VEL_MM_S,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute ``tau in R^M`` matching ``C.blended_segment_durations`` and exact ``(g_deltas, g_speeds)``."""
    d = np.asarray(deltas_mm, dtype=np.float64)
    s = np.asarray(speeds_mm_s, dtype=np.float64)
    m = d.shape[0]
    if m == 0:
        return np.zeros(0, dtype=np.float64), None, None

    inv_2a = 0.5 / float(max_accel_mm_s2)
    norm_eps = np.sqrt(np.sum(d * d, axis=1) + eps_len_mm * eps_len_mm)  # (M,)
    eff_len = norm_eps - eps_len_mm
    cruise = eff_len / s
    vel = (s / norm_eps)[:, None] * d  # (M, 3)

    v_prev = np.zeros_like(vel)
    v_prev[1:] = vel[:-1]
    dv = vel - v_prev
    j_norm = np.sqrt(np.sum(dv * dv, axis=1) + eps_vel_mm_s * eps_vel_mm_s)
    junction = (j_norm - eps_vel_mm_s) * inv_2a

    end_norm = math.sqrt(float(np.sum(vel[-1] ** 2)) + eps_vel_mm_s * eps_vel_mm_s)
    junction[-1] += (end_norm - eps_vel_mm_s) * inv_2a
    tau = cruise + junction

    if g_tau is None:
        return tau, None, None

    gt = np.asarray(g_tau, dtype=np.float64)
    # Gradient through junction velocities vel[k]
    g_dv = (gt * inv_2a / j_norm)[:, None] * dv  # (M, 3)
    g_vel = g_dv.copy()
    g_vel[:-1] -= g_dv[1:]
    g_vel[-1] += (gt[-1] * inv_2a / end_norm) * vel[-1]

    # vel[k] = (s[k] / norm_eps[k]) * d[k]
    dot_gv_d = np.sum(g_vel * d, axis=1)
    g_speeds = -eff_len / (s * s) * gt + dot_gv_d / norm_eps

    # d(cruise)/d(norm_eps) = gt / s; d(1/norm_eps)/d(norm_eps) = -1 / norm_eps^2
    g_norm_eps = gt / s - (s * dot_gv_d) / (norm_eps ** 2)
    g_deltas = (s / norm_eps)[:, None] * g_vel + (g_norm_eps / norm_eps)[:, None] * d
    return tau, g_deltas, g_speeds


class SubsegmentedCWEModel:
    """Chronological 2.5D STE Differentiable Cutter-Workpiece Engagement & Chipload Model.

    Parameter vector ``theta in R^(2 * G + M)``:
    - ``u_xy in R^(G, 2)``: pre-tanh XY shifts for ``G`` vertical-column waypoint groups,
      mapped via ``r_g * tanh(u_xy[g])`` with ``r_g`` bounded by exact triangle-inequality
      wall clearance so ``original_part`` gouge is identically zero and vertical plunges
      stay strictly vertical while open-air lead-in/lead-out columns can retract freely.
    - ``u_f in R^M``: pre-tanh feed controls per sub-segment ``k``, mapped smoothly to
      ``[f_min, f_max]`` on cutting segments and pinned to ``rapid_mm_s`` on rapids.
    """

    def __init__(
        self,
        bundle: MillingTaskBundle,
        spec: Optional[ChiploadSpec] = None,
    ) -> None:
        self.bundle = bundle
        self.task = bundle.task
        self.spec = spec or ChiploadSpec.from_feeds_and_speeds(bundle)
        self.base_motion = subdivide_motion_plan(
            self.task, bundle.nominal_motion, max_cut_seg_mm=self.spec.subdivide_max_mm
        )
        self.m_segs = len(self.base_motion.segments)

        inv = self.task.work_to_machine.inverse().apply
        pts = np.asarray(
            [inv(self.base_motion.initial_state.position_mm)]
            + [inv(s.end_mm) for s in self.base_motion.segments],
            dtype=np.float64,
        )
        self.ref_pts = pts  # (M+1, 3)
        self.ref_deltas = np.diff(pts, axis=0)  # (M, 3)
        self.is_rapid = np.array(
            [s.kind == C.SegmentKind.RAPID for s in self.base_motion.segments], dtype=bool
        )
        self.cut_mask = (~self.is_rapid).astype(np.float64)
        self.nom_feeds = np.array(
            [
                float(self.task.limits.rapid_mm_s)
                if s.kind == C.SegmentKind.RAPID or s.feed_mm_s is None
                else float(s.feed_mm_s)
                for s in self.base_motion.segments
            ],
            dtype=np.float64,
        )
        self.f_min = 3.0
        self.f_max = float(self.task.limits.max_feed_mm_s) * 0.995
        self.rapid_speed = float(self.task.limits.rapid_mm_s)
        self.max_accel = float(self.task.limits.max_accel_mm_s2)
        self.mrr_limit = float(bundle.mrr_limit_mm3_s)

        # Build coupled vertical-column groups over interior waypoints 1..M-1
        # Waypoint 0 (initial_state) and Waypoint M (terminal retract) are pinned to group -1.
        wp_group = np.full(self.m_segs + 1, -1, dtype=np.int64)
        g_cursor = 0
        for wp_idx in range(1, self.m_segs):
            prev_seg_xy = math.hypot(
                float(self.ref_deltas[wp_idx - 1, 0]), float(self.ref_deltas[wp_idx - 1, 1])
            )
            if prev_seg_xy <= 1e-6 and wp_idx > 1 and wp_group[wp_idx - 1] >= 0:
                # Pure vertical plunge/retract: share exact same XY shift as top/bottom mate!
                wp_group[wp_idx] = wp_group[wp_idx - 1]
            else:
                wp_group[wp_idx] = g_cursor
                g_cursor += 1
        # If the last interior waypoint connects via a vertical move to terminal waypoint M, pin that group
        if self.m_segs >= 2:
            last_xy = math.hypot(
                float(self.ref_deltas[self.m_segs - 1, 0]),
                float(self.ref_deltas[self.m_segs - 1, 1]),
            )
            if last_xy <= 1e-6 and wp_group[self.m_segs - 1] >= 0:
                bad_g = wp_group[self.m_segs - 1]
                wp_group[wp_group == bad_g] = -1

        # Compact active group IDs
        unique_g = sorted(int(g) for g in set(wp_group.tolist()) if g >= 0)
        remap = {old_g: new_i for new_i, old_g in enumerate(unique_g)}
        self.wp_group = np.array([remap.get(int(g), -1) for g in wp_group], dtype=np.int64)
        self.n_groups = len(unique_g)

        # Compute exact triangle-inequality safe XY shift radius per column group
        seg_clear = np.array(
            [
                100.0
                if self.is_rapid[k]
                else _segment_wall_clearance_mm(self.task, self.ref_pts[k], self.ref_pts[k + 1])
                for k in range(self.m_segs)
            ],
            dtype=np.float64,
        )
        group_radii = np.full(self.n_groups, float(self.spec.max_xy_shift_mm), dtype=np.float64)
        for k in range(self.m_segs):
            if self.is_rapid[k]:
                continue
            allow_k = max(0.0, float(seg_clear[k]) - float(self.spec.wall_safety_margin_mm))
            for wp_i in (k, k + 1):
                gid = int(self.wp_group[wp_i])
                if gid >= 0:
                    group_radii[gid] = min(group_radii[gid], allow_k)
        self.group_radii = np.clip(group_radii, 0.0, float(self.spec.max_xy_shift_mm))

        # Envelope clamp margins per group
        env_lo = self.task.limits.envelope_min_mm
        env_hi = self.task.limits.envelope_max_mm
        for wp_i in range(1, self.m_segs):
            gid = int(self.wp_group[wp_i])
            if gid >= 0:
                x_ref, y_ref = float(self.ref_pts[wp_i, 0]), float(self.ref_pts[wp_i, 1])
                env_room = min(
                    x_ref - float(env_lo[0]) - 0.5,
                    float(env_hi[0]) - x_ref - 0.5,
                    y_ref - float(env_lo[1]) - 0.5,
                    float(env_hi[1]) - y_ref - 0.5,
                )
                self.group_radii[gid] = min(self.group_radii[gid], max(0.0, env_room))

        # 2.5D dexel grid for differentiable CWE forward/backward
        self.pitch = float(self.spec.dexel_pitch_mm)
        self.cell_area = self.pitch * self.pitch
        self.xs, self.ys, self.z_init, self.z_target = build_target_heightfield_25d(
            bundle, pitch_mm=self.pitch
        )
        self.nx, self.ny = self.z_init.shape
        self.x0_grid = float(bundle.case.stock_min_mm[0])
        self.y0_grid = float(bundle.case.stock_min_mm[1])
        self.z_min_stock = float(bundle.case.stock_min_mm[2])
        self.z_max_stock = float(bundle.case.stock_max_mm[2])
        self.r_cut = float(self.task.tool.cutter_radius_mm)
        self.diam = 2.0 * self.r_cut

        # Inverse-map nominal feeds into initial u_f0 so theta=0 reproduces base_motion exactly
        self.u_f0 = np.zeros(self.m_segs, dtype=np.float64)
        for k in range(self.m_segs):
            if not self.is_rapid[k]:
                f_clamped = np.clip(self.nom_feeds[k], self.f_min + 0.05, self.f_max - 0.05)
                frac = (f_clamped - self.f_min) / (self.f_max - self.f_min)
                self.u_f0[k] = 0.5 * math.log(frac / (1.0 - frac))

        self.num_xy_params = 2 * self.n_groups
        self.num_params = self.num_xy_params + self.m_segs
        self.nom_duration_s = float(sum(s.duration_s for s in self.base_motion.segments))

        # Precompute nominal local CWE observation features (M, 12) for RL policies
        self.nominal_obs = self._build_local_cwe_observations()

    def zero_theta(self) -> np.ndarray:
        return np.zeros(self.num_params, dtype=np.float64)

    def decode(self, theta: np.ndarray, xy_scale_override: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode ``theta`` into ``(pts_work [M+1,3], deltas_work [M,3], speeds_mm_s [M])``."""
        th = np.asarray(theta, dtype=np.float64).reshape(-1)
        u_xy = th[: self.num_xy_params].reshape(self.n_groups, 2)
        u_f = th[self.num_xy_params :]

        group_shifts = (xy_scale_override * self.group_radii)[:, None] * np.tanh(u_xy)  # (G, 2)
        pts = self.ref_pts.copy()
        valid = self.wp_group >= 0
        if np.any(valid):
            pts[valid, :2] += group_shifts[self.wp_group[valid]]
        deltas = np.diff(pts, axis=0)

        # Feed mapping: f = f_min + (f_max - f_min) * sigmoid(2 * (u_f0 + u_f))
        arg = np.clip(2.0 * (self.u_f0 + u_f), -30.0, 30.0)
        sig = 1.0 / (1.0 + np.exp(-arg))
        cut_speeds = self.f_min + (self.f_max - self.f_min) * sig
        speeds = np.where(self.is_rapid, self.rapid_speed, cut_speeds)
        return pts, deltas, speeds

    def to_motion_plan(
        self,
        theta: np.ndarray,
        *,
        xy_scale_override: float = 1.0,
        provenance: str = "diffcam:chipload_ste",
    ) -> C.MotionPlan:
        """Convert parameter vector ``theta`` into a contract-validated ``C.MotionPlan``."""
        pts, _, speeds = self.decode(theta, xy_scale_override=xy_scale_override)
        to_mach = self.task.work_to_machine.apply
        raw_segs: List[C.Segment] = []
        for k, orig in enumerate(self.base_motion.segments):
            sm = to_mach((float(pts[k, 0]), float(pts[k, 1]), float(pts[k, 2])))
            em = to_mach((float(pts[k + 1, 0]), float(pts[k + 1, 1]), float(pts[k + 1, 2])))
            ln = math.dist(sm, em)
            is_r = self.is_rapid[k]
            f_val = None if is_r else float(speeds[k])
            dur = ln / (self.rapid_speed if is_r else float(speeds[k]))
            raw_segs.append(
                C.Segment(
                    id=orig.id,
                    kind=orig.kind,
                    start_mm=sm,
                    end_mm=em,
                    duration_s=max(dur, 1e-9),
                    feed_mm_s=f_val,
                    tag=orig.tag,
                )
            )
        if self.task.motion_semantics.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
            durs = C.blended_segment_durations(raw_segs, self.rapid_speed, self.max_accel)
            raw_segs = [dataclasses.replace(s, duration_s=float(d)) for s, d in zip(raw_segs, durs)]

        plan = C.MotionPlan(
            frame=C.Frame.MACHINE,
            initial_state=self.task.initial_state,
            segments=tuple(raw_segs),
            timing_rule=self.task.motion_semantics.timing_rule,
            provenance=provenance,
        )
        plan.validate_for_task(self.task)
        return plan

    def loss_and_grad(
        self,
        theta: np.ndarray,
        *,
        use_ste: bool = True,
        xy_scale_override: float = 1.0,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """Evaluate chronological 2.5D CWE + Chipload objective and exact reverse-mode gradient."""
        th = np.asarray(theta, dtype=np.float64).reshape(-1)
        u_xy = th[: self.num_xy_params].reshape(self.n_groups, 2)
        u_f = th[self.num_xy_params :]

        pts, deltas, speeds = self.decode(th, xy_scale_override=xy_scale_override)
        tau, _, _ = blended_durations_with_pullback(deltas, speeds, self.max_accel, g_tau=None)

        z_cur = self.z_init.copy()
        tau_r = float(self.spec.smooth_footprint_mm)
        pad_r = self.r_cut + 8.0 * tau_r
        eps_seg = 1e-6

        # Forward pass caches for exact chronological reverse-mode AD
        patches: List[Optional[Tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]] = []
        dv_arr = np.zeros(self.m_segs, dtype=np.float64)
        ap_arr = np.full(self.m_segs, 1.0, dtype=np.float64)
        lxy_arr = np.hypot(deltas[:, 0], deltas[:, 1])

        for k in range(self.m_segs):
            if self.is_rapid[k]:
                patches.append(None)
                continue
            ax, ay, az = float(pts[k, 0]), float(pts[k, 1]), float(pts[k, 2])
            bx, by, bz = float(pts[k + 1, 0]), float(pts[k + 1, 1]), float(pts[k + 1, 2])
            z_cut = max(min(az, bz), self.z_min_stock)
            if z_cut >= self.z_max_stock - 1e-7:
                patches.append(None)
                continue

            ix0 = max(0, int(math.floor((min(ax, bx) - pad_r - self.x0_grid) / self.pitch)))
            ix1 = min(self.nx, int(math.ceil((max(ax, bx) + pad_r - self.x0_grid) / self.pitch)) + 1)
            iy0 = max(0, int(math.floor((min(ay, by) - pad_r - self.y0_grid) / self.pitch)))
            iy1 = min(self.ny, int(math.ceil((max(ay, by) + pad_r - self.y0_grid) / self.pitch)) + 1)
            if ix0 >= ix1 or iy0 >= iy1:
                patches.append(None)
                continue

            px = self.xs[ix0:ix1, None]
            py = self.ys[None, iy0:iy1]
            dx = bx - ax
            dy = by - ay
            l2 = dx * dx + dy * dy + eps_seg * eps_seg
            t_raw = ((px - ax) * dx + (py - ay) * dy) / l2
            t_proj = np.clip(t_raw, 0.0, 1.0)
            rx = px - (ax + t_proj * dx)
            ry = py - (ay + t_proj * dy)
            dist = np.sqrt(rx * rx + ry * ry + eps_seg * eps_seg)

            arg_w = np.clip((self.r_cut - dist) / tau_r, -35.0, 35.0)
            w_soft = 1.0 / (1.0 + np.exp(-arg_w))
            w_hard = (dist <= self.r_cut).astype(np.float64)
            w_fwd = w_hard if use_ste else w_soft

            z_pre = z_cur[ix0:ix1, iy0:iy1].copy()
            depth_avail = np.maximum(0.0, z_pre - z_cut)
            dz_removed = depth_avail * w_fwd
            z_cur[ix0:ix1, iy0:iy1] = z_pre - dz_removed

            dv_k = float(np.sum(dz_removed)) * self.cell_area
            dv_arr[k] = dv_k
            # Smooth axial depth scale (Z is fixed per pass; multi-depth layers step by ~1.5-2.0 mm)
            z_ref_top = float(np.max(z_pre)) if z_pre.size > 0 else self.z_max_stock
            ap_nom = min(max(self.z_max_stock - z_cut, 0.50), 2.50)
            ap_arr[k] = ap_nom if use_ste is False else max(min(z_ref_top - z_cut, 2.50), 0.50)

            patches.append((ix0, ix1, iy0, iy1, z_pre, w_soft, w_fwd, t_proj, rx, ry, z_cut))

        # Per-segment CWE & Radial Chip-Thinning Chipload Objective Terms
        eff_span = np.maximum(lxy_arr, 0.35 * self.r_cut)
        denom_ae = ap_arr * eff_span
        raw_ae = dv_arr / denom_ae
        ae_mm = np.clip(raw_ae, 0.0, self.diam)
        u_ae = ae_mm / self.diam  # (M,) in [0, 1]

        psi, dpsi_du = _radial_chip_thinning_and_grad(u_ae)
        slot_fac, dslot_du = _slot_factor_and_grad(
            u_ae, self.spec.ideal_woc_frac, self.spec.slot_ipt_factor
        )
        tooth_freq = self.spec.tooth_freq_hz
        fz = speeds / tooth_freq
        hex_mm = fz * psi

        # Smooth engagement indicator eta_k in (0, 1): 1 when removing material, 0 in air
        v_thresh = 0.08  # mm^3 threshold per sub-segment
        v_temp = 0.04
        arg_eta = np.clip((dv_arr - v_thresh) / v_temp, -35.0, 35.0)
        eta = (1.0 / (1.0 + np.exp(-arg_eta))) * self.cut_mask

        inv_tnom = 1.0 / max(self.nom_duration_s, 1e-6)
        h_star = float(self.spec.target_hex_mm)
        h_star_vec = h_star * slot_fac

        # 1a. Adaptive cycle time term
        loss_time = float(self.spec.w_time) * float(np.sum(tau)) * inv_tnom

        # 1b. Fair matched-speed geometric path-time term (prevents path lengthening)
        base_cut = float(np.clip(self.spec.base_feed_mm_s, self.f_min, self.f_max))
        matched_speeds = np.where(self.is_rapid, self.rapid_speed, base_cut)
        tau_matched, _, _ = blended_durations_with_pullback(
            deltas, matched_speeds, self.max_accel, g_tau=None
        )
        loss_matched = float(self.spec.w_matched_time) * float(np.sum(tau_matched)) * inv_tnom

        # 2. Optimal chipload tracking (with slotting derating) + tooth-overload barrier
        h_max = float(self.spec.max_hex_mm)
        hex_rel_err = (hex_mm - h_star_vec) / h_star
        hex_over = np.maximum(0.0, (hex_mm - h_max) / h_star)
        chip_pen_per_seg = (
            float(self.spec.w_chipload) * (hex_rel_err ** 2)
            + float(self.spec.w_hex_overload) * (hex_over ** 2)
        )
        loss_chip = float(np.sum(eta * tau * inv_tnom * chip_pen_per_seg))

        # 3. High-engagement corner spike barrier (a_e / D > u_spike_cap)
        spike_excess = np.maximum(0.0, u_ae - float(self.spec.u_spike_cap)) * self.cut_mask
        loss_spike = float(self.spec.w_spike) * float(np.sum(spike_excess ** 2)) / max(self.m_segs, 1)

        # 4. Volumetric MRR barrier
        mrr_k = (dv_arr / np.maximum(tau, 1e-6)) * self.cut_mask
        mrr_excess = np.maximum(0.0, mrr_k / self.mrr_limit - 1.0)
        loss_mrr = float(self.spec.w_mrr) * float(np.sum(mrr_excess ** 2)) / max(self.m_segs, 1)

        # 5. Terminal target completion residual barrier
        res_grid = np.maximum(0.0, z_cur - self.z_target)
        inv_cells = 1.0 / float(self.nx * self.ny)
        loss_res = float(self.spec.w_residual) * float(np.sum(res_grid ** 2)) * inv_cells

        # 6. Gentle XY spatial regularization on waypoint second differences
        dp_xy = pts[:, :2] - self.ref_pts[:, :2]
        lap_xy = dp_xy[2:] - 2.0 * dp_xy[1:-1] + dp_xy[:-2]
        loss_smooth = (
            float(self.spec.w_smooth_xy) * float(np.sum(lap_xy ** 2)) / max(self.m_segs, 1)
            if self.m_segs >= 2
            else 0.0
        )

        total_loss = (
            loss_time
            + loss_matched
            + loss_chip
            + loss_spike
            + loss_mrr
            + loss_res
            + loss_smooth
        )

        # ---------------- Reverse-Mode Analytical Adjoint Pass ----------------
        g_tau = np.full(self.m_segs, float(self.spec.w_time) * inv_tnom, dtype=np.float64)
        g_tau += eta * inv_tnom * chip_pen_per_seg

        # Gradient wrt speeds (direct from hex_mm = (speeds / tooth_freq) * psi)
        over_active = (hex_mm > h_max).astype(np.float64)
        d_chip_pen_dhex = (
            float(self.spec.w_chipload) * (2.0 * hex_rel_err / h_star)
            + float(self.spec.w_hex_overload) * (2.0 * hex_over / h_star) * over_active
        )
        g_hex = eta * tau * inv_tnom * d_chip_pen_dhex
        g_speeds_direct = g_hex * (psi / tooth_freq) * self.cut_mask

        # Gradient wrt u_ae -> ae_mm -> (dv_arr, lxy_arr)
        g_u_ae = g_hex * fz * dpsi_du
        g_u_ae -= (
            eta
            * tau
            * inv_tnom
            * (2.0 * float(self.spec.w_chipload) * hex_rel_err * dslot_du)
        )
        g_u_ae += (
            float(self.spec.w_spike)
            * (2.0 * spike_excess / max(self.m_segs, 1))
            * self.cut_mask
        )
        ae_active = ((raw_ae > 0.0) & (raw_ae < self.diam)).astype(np.float64)
        g_raw_ae = (g_u_ae / self.diam) * ae_active

        g_dv = g_raw_ae / denom_ae
        # eta dependence on dv_arr
        d_eta_ddv = eta * (1.0 - np.clip(eta, 0.0, 1.0)) / v_temp
        g_dv += d_eta_ddv * tau * inv_tnom * chip_pen_per_seg

        # MRR dependence on dv_arr and tau
        g_mrr_k = (
            float(self.spec.w_mrr)
            * (2.0 * mrr_excess / (self.mrr_limit * max(self.m_segs, 1)))
            * self.cut_mask
        )
        inv_tau = 1.0 / np.maximum(tau, 1e-6)
        g_dv += g_mrr_k * inv_tau
        g_tau -= g_mrr_k * dv_arr * (inv_tau ** 2)

        # Pullback blended timing wrt deltas and speeds
        _, g_deltas_time, g_speeds_time = blended_durations_with_pullback(
            deltas, speeds, self.max_accel, g_tau=g_tau
        )
        g_tau_matched = np.full(
            self.m_segs, float(self.spec.w_matched_time) * inv_tnom, dtype=np.float64
        )
        _, g_deltas_matched, _ = blended_durations_with_pullback(
            deltas, matched_speeds, self.max_accel, g_tau=g_tau_matched
        )
        assert (
            g_deltas_time is not None
            and g_speeds_time is not None
            and g_deltas_matched is not None
        )
        g_speeds_total = (g_speeds_direct + g_speeds_time) * self.cut_mask

        # raw_ae dependence on lxy_arr (where lxy_arr > 0.35 * r_cut)
        lxy_active = (lxy_arr > 0.35 * self.r_cut).astype(np.float64)
        g_lxy = -g_raw_ae * (raw_ae / eff_span) * lxy_active
        inv_lxy = np.where(lxy_arr > 1e-8, 1.0 / np.maximum(lxy_arr, 1e-8), 0.0)
        g_deltas = g_deltas_time + g_deltas_matched
        g_deltas[:, 0] += g_lxy * deltas[:, 0] * inv_lxy
        g_deltas[:, 1] += g_lxy * deltas[:, 1] * inv_lxy

        # Convert g_deltas (M, 3) into waypoint gradients g_pts (M+1, 2)
        g_pts_xy = np.zeros((self.m_segs + 1, 2), dtype=np.float64)
        g_pts_xy[:-1] -= g_deltas[:, :2]
        g_pts_xy[1:] += g_deltas[:, :2]

        # Laplacian smoothness gradient wrt pts[:, :2]
        if self.m_segs >= 2:
            c_sm = 2.0 * float(self.spec.w_smooth_xy) / max(self.m_segs, 1)
            g_pts_xy[:-2] += c_sm * lap_xy
            g_pts_xy[1:-1] -= 2.0 * c_sm * lap_xy
            g_pts_xy[2:] += c_sm * lap_xy

        # Chronological reverse sweep over 2.5D dexel patches (k = M-1 .. 0)
        adj_z = (2.0 * float(self.spec.w_residual) * inv_cells) * res_grid  # dL / dZ^(M)
        for k in range(self.m_segs - 1, -1, -1):
            patch = patches[k]
            if patch is None:
                continue
            ix0, ix1, iy0, iy1, z_pre, w_soft, w_fwd, t_proj, rx, ry, z_cut = patch
            adj_patch = adj_z[ix0:ix1, iy0:iy1]
            depth_avail = np.maximum(0.0, z_pre - z_cut)
            has_depth = (z_pre > z_cut).astype(np.float64)

            # Z_post = Z_pre - dz_removed, dv_k = sum(dz_removed) * cell_area
            g_dz = -adj_patch + g_dv[k] * self.cell_area
            g_w = g_dz * depth_avail

            # STE surrogate derivative: d(w_soft)/d(dist) = -w_soft*(1 - w_soft)/tau_r
            dw_ddist = -(w_soft * (1.0 - w_soft)) / tau_r
            dist = np.sqrt(rx * rx + ry * ry + eps_seg * eps_seg)
            coeff = (g_w * dw_ddist) / dist

            # At clamped projection t_proj, envelope theorem cancels interior dt/dA, dt/dB!
            # d(dist)/d(ax) = -(1 - t_proj) * rx / dist, d(dist)/d(bx) = -t_proj * rx / dist
            one_minus_t = 1.0 - t_proj
            g_ax = -float(np.sum(coeff * one_minus_t * rx))
            g_ay = -float(np.sum(coeff * one_minus_t * ry))
            g_bx = -float(np.sum(coeff * t_proj * rx))
            g_by = -float(np.sum(coeff * t_proj * ry))

            g_pts_xy[k, 0] += g_ax
            g_pts_xy[k, 1] += g_ay
            g_pts_xy[k + 1, 0] += g_bx
            g_pts_xy[k + 1, 1] += g_by

            # Propagate heightfield adjoint backward:
            # Z_post = Z_pre - dz_removed and dv_k = cell_area * sum(dz_removed),
            # where g_dz = -adj_patch + g_dv[k] * cell_area and d(dz_removed)/d(Z_pre) = has_depth * w_fwd.
            adj_z[ix0:ix1, iy0:iy1] = adj_patch + g_dz * has_depth * w_fwd

        # Pullback waypoint XY gradients to group parameters u_xy
        g_group = np.zeros((self.n_groups, 2), dtype=np.float64)
        for wp_i in range(1, self.m_segs):
            gid = int(self.wp_group[wp_i])
            if gid >= 0:
                g_group[gid] += g_pts_xy[wp_i]
        sech2_xy = 1.0 - np.tanh(u_xy) ** 2
        g_u_xy = g_group * (xy_scale_override * self.group_radii)[:, None] * sech2_xy

        # Pullback speed gradients to u_f: df/du_f = 2 * (f_max - f_min) * sig * (1 - sig)
        arg_f = np.clip(2.0 * (self.u_f0 + u_f), -30.0, 30.0)
        sig_f = 1.0 / (1.0 + np.exp(-arg_f))
        df_duf = 2.0 * (self.f_max - self.f_min) * sig_f * (1.0 - sig_f)
        g_u_f = g_speeds_total * df_duf

        grad_theta = np.concatenate([g_u_xy.reshape(-1), g_u_f])
        diagnostics = {
            "loss": float(total_loss),
            "cycle_time_s": float(np.sum(tau)),
            "loss_time": float(loss_time),
            "loss_chipload": float(loss_chip),
            "loss_spike": float(loss_spike),
            "loss_mrr": float(loss_mrr),
            "loss_residual": float(loss_res),
            "dexel_residual_mm3": float(np.sum(res_grid)) * self.cell_area,
            "mean_xy_shift_mm": float(np.mean(np.hypot(dp_xy[:, 0], dp_xy[:, 1]))),
            "max_xy_shift_mm": float(np.max(np.hypot(dp_xy[:, 0], dp_xy[:, 1]))),
        }
        return float(total_loss), grad_theta, diagnostics

    def _build_local_cwe_observations(self) -> np.ndarray:
        """Build 12-D normalized per-subsegment CWE + kinematic features for RL policies."""
        eng = analyze_plan_engagement(self.bundle, self.base_motion, self.spec)
        segs_e = eng["segments"]

        # Junction cosines at segment start & end
        d = self.ref_deltas
        lens = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        dirs = d / lens[:, None]
        cos_in = np.ones(self.m_segs, dtype=np.float64)
        cos_out = np.ones(self.m_segs, dtype=np.float64)
        if self.m_segs >= 2:
            cos_in[1:] = np.sum(dirs[1:] * dirs[:-1], axis=1)
            cos_out[:-1] = cos_in[1:]

        obs = np.zeros((self.m_segs, 12), dtype=np.float64)
        for k in range(self.m_segs):
            se = segs_e[k]
            u_ae = float(se["ae_pct"]) / 100.0
            psi = float(radial_chip_thinning_factor(u_ae))
            gid = int(self.wp_group[min(k + 1, self.m_segs)])
            r_norm = float(self.group_radii[gid]) / max(self.spec.max_xy_shift_mm, 1e-6) if gid >= 0 else 0.0
            turn_vec = dirs[min(k + 1, self.m_segs - 1), :2] - dirs[k, :2]
            obs[k] = [
                1.0 if self.is_rapid[k] else 0.0,
                1.0 if se["engaged"] else 0.0,
                float(se["engaged_fraction"]),
                u_ae,
                psi,
                min(float(se["mrr_mm3_s"]) / max(self.mrr_limit, 1e-6), 2.0),
                lens[k] / 5.0,
                self.nom_feeds[k] / self.f_max if not self.is_rapid[k] else 1.0,
                cos_in[k],
                cos_out[k],
                r_norm,
                float(np.clip(turn_vec[0] + turn_vec[1], -2.0, 2.0)) * 0.5,
            ]
        return obs


@dataclass
class ChiploadOptResult:
    plan: C.MotionPlan
    theta: np.ndarray
    telemetry: Dict[str, Any]
    mean_xy_shift_mm: float
    max_xy_shift_mm: float
    wall_time_s: float
    shortcut_mode: str = "none"
    posted_eval: Optional[Dict[str, Any]] = None
    history: List[float] = field(default_factory=list)


def retime_motion_plan_constant_speed(
    task: C.TaskSpec,
    plan: C.MotionPlan,
    cut_feed_mm_s: float,
    *,
    provenance_suffix: str = "matched_speed",
) -> C.MotionPlan:
    """Return a copy of ``plan`` with every ``G1`` cutting segment set to ``cut_feed_mm_s``."""
    f_cut = float(np.clip(cut_feed_mm_s, 0.2, float(task.limits.max_feed_mm_s) * 0.999))
    rapid_s = float(task.limits.rapid_mm_s)
    new_segs: List[C.Segment] = []
    for s in plan.segments:
        ln = s.path_length_mm
        if s.kind == C.SegmentKind.RAPID:
            new_segs.append(dataclasses.replace(s, duration_s=max(ln / rapid_s, 1e-9), feed_mm_s=None))
        else:
            new_segs.append(dataclasses.replace(s, duration_s=max(ln / f_cut, 1e-9), feed_mm_s=f_cut))
    if plan.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        durs = C.blended_segment_durations(new_segs, rapid_s, float(task.limits.max_accel_mm_s2))
        new_segs = [dataclasses.replace(s, duration_s=float(d)) for s, d in zip(new_segs, durs)]
    out = C.MotionPlan(
        frame=plan.frame,
        initial_state=plan.initial_state,
        segments=tuple(new_segs),
        timing_rule=plan.timing_rule,
        provenance=f"{plan.provenance}:{provenance_suffix}",
    )
    out.validate_for_task(task)
    return out


def _sweep_removes_or_updates_dexels(
    xs: np.ndarray,
    ys: np.ndarray,
    z_stock: np.ndarray,
    x0_grid: float,
    y0_grid: float,
    pitch: float,
    r_eff: float,
    z_min_stock: float,
    z_max_stock: float,
    pa: Tuple[float, float, float],
    pb: Tuple[float, float, float],
    *,
    mutate: bool,
) -> float:
    """Return volume ``mm^3`` below ``z_stock`` swept by flat endmill of radius ``r_eff``."""
    ax, ay, az = pa
    bx, by, bz = pb
    if min(az, bz) >= z_max_stock - 1e-7:
        return 0.0
    nx, ny = z_stock.shape
    r2 = r_eff * r_eff
    len_3d = math.dist(pa, pb)
    n_sub = max(1, int(math.ceil(len_3d / 0.20)))
    dx, dy, dz = bx - ax, by - ay, bz - az
    removed = 0.0
    cell_area = pitch * pitch
    for m in range(n_sub):
        u0 = m / n_sub
        u1 = (m + 1) / n_sub
        sx0, sy0, sz0 = ax + u0 * dx, ay + u0 * dy, az + u0 * dz
        sx1, sy1, sz1 = ax + u1 * dx, ay + u1 * dy, az + u1 * dz
        ix0 = max(0, int(math.floor((min(sx0, sx1) - r_eff - x0_grid) / pitch)))
        ix1 = min(nx, int(math.ceil((max(sx0, sx1) + r_eff - x0_grid) / pitch)) + 1)
        iy0 = max(0, int(math.floor((min(sy0, sy1) - r_eff - y0_grid) / pitch)))
        iy1 = min(ny, int(math.ceil((max(sy0, sy1) + r_eff - y0_grid) / pitch)) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        px = xs[ix0:ix1, None]
        py = ys[None, iy0:iy1]
        sdx, sdy, sdz = sx1 - sx0, sy1 - sy0, sz1 - sz0
        lxy2 = sdx * sdx + sdy * sdy
        if lxy2 <= 1e-12:
            dist2 = (px - sx0) ** 2 + (py - sy0) ** 2
            inside = dist2 <= r2
            z_swept = np.full(dist2.shape, max(min(sz0, sz1), z_min_stock), dtype=np.float64)
        else:
            uc = ((px - sx0) * sdx + (py - sy0) * sdy) / lxy2
            d_perp2 = (px - sx0 - uc * sdx) ** 2 + (py - sy0 - uc * sdy) ** 2
            valid_perp = d_perp2 <= r2
            du = np.sqrt(np.maximum(r2 - d_perp2, 0.0) / lxy2)
            u_lo = np.maximum(0.0, uc - du)
            u_hi = np.minimum(1.0, uc + du)
            inside = valid_perp & (u_lo <= u_hi)
            u_best = u_hi if sdz < 0.0 else u_lo
            z_swept = np.maximum(sz0 + u_best * sdz, z_min_stock)
        cur_patch = z_stock[ix0:ix1, iy0:iy1]
        dz_cut = np.where(inside & (z_swept < cur_patch), cur_patch - z_swept, 0.0)
        dv = float(np.sum(dz_cut)) * cell_area
        if dv > 0.0:
            removed += dv
            if mutate:
                z_stock[ix0:ix1, iy0:iy1] = cur_patch - dz_cut
            elif dv > 1e-7:
                return removed
    return removed


def shortcut_verified_air_detours(
    bundle: MillingTaskBundle,
    plan: C.MotionPlan,
    spec: Optional[ChiploadSpec] = None,
    *,
    mode: str = "rapid_bridge",
) -> C.MotionPlan:
    """Collapse stock-sim-verified empty-air ``G1`` detours into straight chords or safe ``G0`` bridges."""
    if mode == "none" or len(plan.segments) < 4:
        return plan

    cfg = spec or ChiploadSpec.from_feeds_and_speeds(bundle)
    pitch = float(cfg.dexel_pitch_mm)
    xs, ys, z_stock, _ = build_target_heightfield_25d(bundle, pitch_mm=pitch)
    nx, ny = z_stock.shape
    x0_grid = float(bundle.case.stock_min_mm[0])
    y0_grid = float(bundle.case.stock_min_mm[1])
    z_min_stock = float(bundle.case.stock_min_mm[2])
    z_max_stock = float(bundle.case.stock_max_mm[2])
    z_safe = min(
        z_max_stock + 2.0,
        float(bundle.task.limits.envelope_max_mm[2]) - 0.5,
    )
    r_cut = float(bundle.task.tool.cutter_radius_mm)
    rapid_s = float(bundle.task.limits.rapid_mm_s)
    base_f = max(4.0, float(cfg.base_feed_mm_s))

    inv = bundle.task.work_to_machine.inverse().apply
    to_mach = bundle.task.work_to_machine.apply
    segs = list(plan.segments)
    n_segs = len(segs)

    def _vertical_rapid_safe(pt_w: Tuple[float, float, float]) -> bool:
        arr_p = np.asarray(pt_w, dtype=np.float64)
        margin_mm = max(0.85, math.sqrt(3.0) * 0.5 * float(bundle.task.grid.voxel_mm) + 0.35)
        if _segment_wall_clearance_mm(bundle.task, arr_p, arr_p) < margin_mm:
            return False
        r_chk = r_cut + margin_mm
        ix0 = max(0, int(math.floor((pt_w[0] - r_chk - x0_grid) / pitch)))
        ix1 = min(nx, int(math.ceil((pt_w[0] + r_chk - x0_grid) / pitch)) + 1)
        iy0 = max(0, int(math.floor((pt_w[1] - r_chk - y0_grid) / pitch)))
        iy1 = min(ny, int(math.ceil((pt_w[1] + r_chk - y0_grid) / pitch)) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            return True
        px = xs[ix0:ix1, None]
        py = ys[None, iy0:iy1]
        disk = ((px - pt_w[0]) ** 2 + (py - pt_w[1]) ** 2) <= (r_chk * r_chk)
        if not np.any(disk):
            return True
        return bool(np.all(z_stock[ix0:ix1, iy0:iy1][disk] <= pt_w[2] + 1e-5))

    out_segs: List[C.Segment] = []
    sid = 0
    i = 0
    while i < n_segs:
        s_i = segs[i]
        pa_i = tuple(float(v) for v in inv(s_i.start_mm))
        pb_i = tuple(float(v) for v in inv(s_i.end_mm))
        if s_i.kind == C.SegmentKind.RAPID:
            out_segs.append(dataclasses.replace(s_i, id=f"s{sid:04d}"))
            sid += 1
            i += 1
            continue

        # Probe whether s_i is a zero-removal horizontal G1 segment
        is_horiz = abs(pb_i[2] - pa_i[2]) <= 1e-5 and math.hypot(pb_i[0] - pa_i[0], pb_i[1] - pa_i[1]) > 1e-5
        if not is_horiz:
            _sweep_removes_or_updates_dexels(
                xs, ys, z_stock, x0_grid, y0_grid, pitch, r_cut, z_min_stock, z_max_stock, pa_i, pb_i, mutate=True
            )
            out_segs.append(dataclasses.replace(s_i, id=f"s{sid:04d}"))
            sid += 1
            i += 1
            continue

        dv_i = _sweep_removes_or_updates_dexels(
            xs, ys, z_stock, x0_grid, y0_grid, pitch, r_cut, z_min_stock, z_max_stock, pa_i, pb_i, mutate=False
        )
        if dv_i > 1e-6:
            _sweep_removes_or_updates_dexels(
                xs, ys, z_stock, x0_grid, y0_grid, pitch, r_cut, z_min_stock, z_max_stock, pa_i, pb_i, mutate=True
            )
            out_segs.append(dataclasses.replace(s_i, id=f"s{sid:04d}"))
            sid += 1
            i += 1
            continue

        # Extend contiguous zero-removal horizontal G1 run [i .. j-1]
        z_level = pa_i[2]
        j = i + 1
        while j < n_segs and segs[j].kind != C.SegmentKind.RAPID:
            pa_j = tuple(float(v) for v in inv(segs[j].start_mm))
            pb_j = tuple(float(v) for v in inv(segs[j].end_mm))
            if abs(pa_j[2] - z_level) > 1e-5 or abs(pb_j[2] - z_level) > 1e-5:
                break
            if math.hypot(pb_j[0] - pa_j[0], pb_j[1] - pa_j[1]) <= 1e-5:
                break
            dv_j = _sweep_removes_or_updates_dexels(
                xs, ys, z_stock, x0_grid, y0_grid, pitch, r_cut, z_min_stock, z_max_stock, pa_j, pb_j, mutate=False
            )
            if dv_j > 1e-6:
                break
            j += 1

        run_len = j - i
        replaced = False
        if run_len >= 2:
            # Candidate sub-spans: full [i, j-1] first, then 1-station buffered [i+1, j-2]
            cand_bounds: List[Tuple[int, int]] = [(i, j - 1)]
            if run_len >= 4:
                cand_bounds.append((i + 1, j - 2))

            for a_idx, b_idx in cand_bounds:
                pa_w = tuple(float(v) for v in inv(segs[a_idx].start_mm))
                pb_w = tuple(float(v) for v in inv(segs[b_idx].end_mm))
                orig_len = sum(segs[k].path_length_mm for k in range(a_idx, b_idx + 1))
                chord_xy = math.hypot(pb_w[0] - pa_w[0], pb_w[1] - pa_w[1])

                # Option 1: Safe G0 rapid bridge above z_safe
                if (
                    mode == "rapid_bridge"
                    and orig_len >= 5.0
                    and z_safe > z_level + 0.2
                    and _vertical_rapid_safe(pa_w)
                    and _vertical_rapid_safe(pb_w)
                ):
                    lift_d = max(0.0, z_safe - pa_w[2])
                    plunge_d = max(0.0, z_safe - pb_w[2])
                    bridge_time = (lift_d + chord_xy + plunge_d) / rapid_s + 0.04
                    if bridge_time < 0.80 * (orig_len / base_f):
                        for pre_k in range(i, a_idx):
                            out_segs.append(dataclasses.replace(segs[pre_k], id=f"s{sid:04d}"))
                            sid += 1
                        p_top_a = (pa_w[0], pa_w[1], z_safe)
                        p_top_b = (pb_w[0], pb_w[1], z_safe)
                        for w0, w1 in ((pa_w, p_top_a), (p_top_a, p_top_b), (p_top_b, pb_w)):
                            m0, m1 = to_mach(w0), to_mach(w1)
                            d_m = math.dist(m0, m1)
                            if d_m > 1e-6:
                                out_segs.append(
                                    C.Segment(
                                        id=f"s{sid:04d}",
                                        kind=C.SegmentKind.RAPID,
                                        start_mm=m0,
                                        end_mm=m1,
                                        duration_s=max(d_m / rapid_s, 1e-9),
                                        feed_mm_s=None,
                                        tag="air_bridge_rapid",
                                    )
                                )
                                sid += 1
                        for post_k in range(b_idx + 1, j):
                            out_segs.append(dataclasses.replace(segs[post_k], id=f"s{sid:04d}"))
                            sid += 1
                        replaced = True
                        break

                # Option 2: Direct straight G1 chord across verified empty cavity
                if chord_xy > 1e-4 and (orig_len - chord_xy) >= 0.08:
                    pa_np = np.asarray(pa_w, dtype=np.float64)
                    pb_np = np.asarray(pb_w, dtype=np.float64)
                    if _segment_wall_clearance_mm(bundle.task, pa_np, pb_np) >= 0.24:
                        dv_chord = _sweep_removes_or_updates_dexels(
                            xs,
                            ys,
                            z_stock,
                            x0_grid,
                            y0_grid,
                            pitch,
                            r_cut + 0.10,
                            z_min_stock,
                            z_max_stock,
                            pa_w,
                            pb_w,
                            mutate=False,
                        )
                        if dv_chord <= 1e-7:
                            for pre_k in range(i, a_idx):
                                out_segs.append(dataclasses.replace(segs[pre_k], id=f"s{sid:04d}"))
                                sid += 1
                            max_f_span = max(
                                float(segs[k].feed_mm_s or base_f) for k in range(a_idx, b_idx + 1)
                            )
                            n_ch = max(1, int(math.ceil(chord_xy / max(cfg.subdivide_max_mm, 1.0))))
                            for m in range(n_ch):
                                u0, u1 = m / n_ch, (m + 1) / n_ch
                                w0 = (
                                    pa_w[0] + u0 * (pb_w[0] - pa_w[0]),
                                    pa_w[1] + u0 * (pb_w[1] - pa_w[1]),
                                    z_level,
                                )
                                w1 = (
                                    pa_w[0] + u1 * (pb_w[0] - pa_w[0]),
                                    pa_w[1] + u1 * (pb_w[1] - pa_w[1]),
                                    z_level,
                                )
                                m0, m1 = to_mach(w0), to_mach(w1)
                                d_m = math.dist(m0, m1)
                                out_segs.append(
                                    C.Segment(
                                        id=f"s{sid:04d}",
                                        kind=segs[a_idx].kind,
                                        start_mm=m0,
                                        end_mm=m1,
                                        duration_s=max(d_m / max_f_span, 1e-9),
                                        feed_mm_s=max_f_span,
                                        tag="air_chord_cut",
                                    )
                                )
                                sid += 1
                            for post_k in range(b_idx + 1, j):
                                out_segs.append(dataclasses.replace(segs[post_k], id=f"s{sid:04d}"))
                                sid += 1
                            replaced = True
                            break

        if not replaced:
            for k in range(i, j):
                out_segs.append(dataclasses.replace(segs[k], id=f"s{sid:04d}"))
                sid += 1
        i = j

    if plan.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        durs = C.blended_segment_durations(
            out_segs, rapid_s, float(bundle.task.limits.max_accel_mm_s2)
        )
        out_segs = [dataclasses.replace(s, duration_s=float(d)) for s, d in zip(out_segs, durs)]

    out_plan = C.MotionPlan(
        frame=plan.frame,
        initial_state=plan.initial_state,
        segments=tuple(out_segs),
        timing_rule=plan.timing_rule,
        provenance=f"{plan.provenance}:{mode}",
    )
    out_plan.validate_for_task(bundle.task)
    return out_plan


def optimize_chipload_trajectory(
    bundle: MillingTaskBundle,
    spec: Optional[ChiploadSpec] = None,
    *,
    num_iters: int = 90,
    lr_xy: float = 0.08,
    lr_feed: float = 0.14,
    verify_posted: bool = True,
) -> ChiploadOptResult:
    """Run ``DiffCAM-Chipload`` 1st-order STE-Adam joint CWE + Chipload trajectory optimization."""
    t0 = time.perf_counter()
    cfg = spec or ChiploadSpec.from_feeds_and_speeds(bundle)
    model = SubsegmentedCWEModel(bundle, spec=cfg)
    theta = model.zero_theta()

    # Pre-initialize u_f toward feeds_and_speeds radial-chip-thinning + slotting target feed
    nom_eng = analyze_plan_engagement(bundle, model.base_motion, model.spec)
    tooth_freq = model.spec.tooth_freq_hz
    for k, se in enumerate(nom_eng["segments"]):
        if model.is_rapid[k]:
            continue
        if not se["engaged"]:
            f_init = model.f_max * 0.99
        else:
            u_ae = max(0.02, float(se["ae_pct"]) / 100.0)
            psi = float(radial_chip_thinning_factor(u_ae))
            sf, _ = _slot_factor_and_grad(
                np.asarray([u_ae]), model.spec.ideal_woc_frac, model.spec.slot_ipt_factor
            )
            h_eff = model.spec.target_hex_mm * float(sf[0])
            f_chip = (tooth_freq * h_eff) / max(psi, 0.34)
            if float(se["removed_mm3"]) > 1e-5:
                seg_len = max(float(np.linalg.norm(model.ref_deltas[k])), 0.2)
                f_mrr = (model.mrr_limit * 0.92 * seg_len) / float(se["removed_mm3"])
                f_chip = min(f_chip, f_mrr)
            f_init = float(np.clip(f_chip, model.f_min + 0.2, model.f_max - 0.05))
        frac = np.clip((f_init - model.f_min) / (model.f_max - model.f_min), 1e-3, 1.0 - 1e-3)
        theta[model.num_xy_params + k] = 0.5 * math.log(frac / (1.0 - frac)) - model.u_f0[k]

    m_adam = np.zeros_like(theta)
    v_adam = np.zeros_like(theta)
    beta1, beta2, eps_a = 0.9, 0.999, 1e-8
    lr_vec = np.empty_like(theta)
    lr_vec[: model.num_xy_params] = lr_xy
    lr_vec[model.num_xy_params :] = lr_feed

    best_theta = theta.copy()
    best_score = float("inf")
    history: List[float] = []
    res_tol = float(bundle.task.tolerances.completion_residual_mm3)

    for it in range(1, num_iters + 1):
        loss, grad, diag = model.loss_and_grad(theta, use_ste=True)
        history.append(loss)
        if diag["dexel_residual_mm3"] <= res_tol * 0.92 and loss < best_score:
            best_score = loss
            best_theta = theta.copy()

        g_clip = np.clip(grad, -10.0, 10.0)
        m_adam = beta1 * m_adam + (1.0 - beta1) * g_clip
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g_clip * g_clip)
        m_hat = m_adam / (1.0 - beta1 ** it)
        v_hat = v_adam / (1.0 - beta2 ** it)
        decay = 0.5 * (1.0 + math.cos(math.pi * (it - 1) / max(num_iters, 1)))
        step_lr = lr_vec * (0.25 + 0.75 * decay)
        theta = np.clip(theta - step_lr * m_hat / (np.sqrt(v_hat) + eps_a), -4.5, 4.5)

    # Certified feasibility guard on exact 0.125 mm posted G-code evaluator
    chosen_plan: Optional[C.MotionPlan] = None
    chosen_theta = best_theta.copy()
    chosen_shortcut = "none"
    posted_dict: Optional[Dict[str, Any]] = None

    from cam.config import MachineConfig
    from cam.posted_evaluator import evaluate_posted_gcode
    from cam.posts import post_motion_plan

    m_cfg = MachineConfig(
        units="mm",
        tool_number=bundle.task.tool.tool_number,
        length_offset=bundle.task.tool.tool_number,
        spindle_rpm=int(round(model.spec.spindle_rpm)),
        coolant=True,
    )

    for xy_scale in (1.0, 0.75, 0.50, 0.25, 0.0):
        raw_plan = model.to_motion_plan(
            best_theta, xy_scale_override=xy_scale, provenance="diffcam:chipload_ste"
        )
        if not verify_posted:
            chosen_plan = shortcut_verified_air_detours(
                bundle, raw_plan, model.spec, mode="rapid_bridge"
            )
            chosen_shortcut = "rapid_bridge"
            break
        for sc_mode in ("rapid_bridge", "g1_chord", "none"):
            cand_plan = shortcut_verified_air_detours(
                bundle, raw_plan, model.spec, mode=sc_mode
            )
            nc_str = post_motion_plan(cand_plan, bundle.task, dialect=C.Dialect.HAAS, config=m_cfg)
            rep = evaluate_posted_gcode(
                bundle.task,
                nc_str.encode("utf-8"),
                dialect=C.Dialect.HAAS,
                settings={"max_step_mm": 0.125, "compute_bounds": False, "safety_screen": True},
            )
            orig_m = rep.metrics.get("original_part_gouge_volume_mm3")
            orig_gouge = (
                float(orig_m.value) if orig_m is not None and orig_m.value is not None else 0.0
            )
            res_vol = float(rep.metrics["residual_volume_mm3"].value)
            if rep.geometry_pass and rep.safety_pass and orig_gouge <= 1e-6 and res_vol <= res_tol:
                chosen_plan = cand_plan
                chosen_shortcut = sc_mode
                if xy_scale < 1.0:
                    u_xy = best_theta[: model.num_xy_params]
                    scaled_tanh = np.clip(xy_scale * np.tanh(u_xy), -0.999, 0.999)
                    chosen_theta[: model.num_xy_params] = np.arctanh(scaled_tanh)
                posted_air = rep.metrics.get("air_cut_time_s")
                posted_rapid = rep.metrics.get("rapid_time_s")
                posted_dict = {
                    "geometry_pass": bool(rep.geometry_pass),
                    "safety_pass": bool(rep.safety_pass),
                    "original_part_gouge_mm3": round(orig_gouge, 5),
                    "residual_mm3": round(res_vol, 4),
                    "posted_air_cut_time_s": round(float(posted_air.value), 4)
                    if posted_air and posted_air.value is not None
                    else None,
                    "posted_rapid_time_s": round(float(posted_rapid.value), 4)
                    if posted_rapid and posted_rapid.value is not None
                    else None,
                    "xy_scale_applied": xy_scale,
                    "shortcut_mode": sc_mode,
                }
                break
        if chosen_plan is not None:
            break

    if chosen_plan is None:
        chosen_plan = model.to_motion_plan(
            np.zeros_like(best_theta), xy_scale_override=0.0, provenance="diffcam:chipload_ste"
        )

    pts_opt, _, _ = model.decode(chosen_theta)
    xy_shifts = np.hypot(
        pts_opt[:, 0] - model.ref_pts[:, 0], pts_opt[:, 1] - model.ref_pts[:, 1]
    )
    telemetry = analyze_plan_engagement(bundle, chosen_plan, model.spec)
    wall_s = time.perf_counter() - t0
    return ChiploadOptResult(
        plan=chosen_plan,
        theta=chosen_theta,
        telemetry=telemetry,
        mean_xy_shift_mm=float(np.mean(xy_shifts)),
        max_xy_shift_mm=float(np.max(xy_shifts)),
        wall_time_s=wall_s,
        shortcut_mode=chosen_shortcut,
        posted_eval=posted_dict,
        history=history,
    )


class LocalCWEPolicy:
    """Compact 2-layer tanh MLP ``(12 -> 32 -> 3)`` mapping local CWE features to ``(dx, dy, df)``."""

    def __init__(self, obs_dim: int = 12, hidden_dim: int = 32, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.w1 = rng.normal(0.0, 0.08, size=(obs_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.w2 = rng.normal(0.0, 0.02, size=(hidden_dim, 3))
        self.b2 = np.zeros(3, dtype=np.float64)

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])

    def set_flat_params(self, flat: np.ndarray) -> None:
        idx = 0
        s1 = self.obs_dim * self.hidden_dim
        self.w1 = flat[idx : idx + s1].reshape(self.obs_dim, self.hidden_dim).copy()
        idx += s1
        self.b1 = flat[idx : idx + self.hidden_dim].copy()
        idx += self.hidden_dim
        s2 = self.hidden_dim * 3
        self.w2 = flat[idx : idx + s2].reshape(self.hidden_dim, 3).copy()
        idx += s2
        self.b2 = flat[idx : idx + 3].copy()

    def forward_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Return ``actions in (-2.5, 2.5)^(M, 3)`` and backprop cache."""
        z1 = obs @ self.w1 + self.b1
        h1 = np.tanh(z1)
        out = 2.2 * np.tanh(h1 @ self.w2 + self.b2)
        return out, {"obs": obs, "h1": h1, "raw_out": h1 @ self.w2 + self.b2}

    def backward_params(self, grad_out: np.ndarray, cache: Mapping[str, np.ndarray]) -> np.ndarray:
        sech2_out = 1.0 - np.tanh(cache["raw_out"]) ** 2
        g_z2 = grad_out * 2.2 * sech2_out  # (M, 3)
        h1 = cache["h1"]
        obs = cache["obs"]
        gw2 = h1.T @ g_z2
        gb2 = np.sum(g_z2, axis=0)
        gh1 = g_z2 @ self.w2.T
        gz1 = gh1 * (1.0 - h1 ** 2)
        gw1 = obs.T @ gz1
        gb1 = np.sum(gz1, axis=0)
        return np.concatenate([gw1.ravel(), gb1, gw2.ravel(), gb2])

    def map_to_model_theta(self, model: SubsegmentedCWEModel) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Map ``(M, 3)`` sub-segment actions onto ``model``'s ``(2*G + M)`` parameter vector."""
        act, cache = self.forward_actions(model.nominal_obs)
        u_xy = np.zeros((model.n_groups, 2), dtype=np.float64)
        counts = np.zeros(model.n_groups, dtype=np.float64)
        for wp_i in range(1, model.m_segs):
            gid = int(model.wp_group[wp_i])
            if gid >= 0:
                u_xy[gid] += act[wp_i, :2]
                counts[gid] += 1.0
        nz = counts > 0
        u_xy[nz] /= counts[nz, None]
        theta = np.concatenate([u_xy.reshape(-1), act[:, 2]])
        return theta, {"mlp": cache, "counts": counts}

    def pullback_theta_grad(
        self, model: SubsegmentedCWEModel, grad_theta: np.ndarray, map_cache: Mapping[str, Any]
    ) -> np.ndarray:
        g_u_xy = grad_theta[: model.num_xy_params].reshape(model.n_groups, 2)
        g_u_f = grad_theta[model.num_xy_params :]
        counts = map_cache["counts"]
        grad_act = np.zeros((model.m_segs, 3), dtype=np.float64)
        grad_act[:, 2] = g_u_f
        for wp_i in range(1, model.m_segs):
            gid = int(model.wp_group[wp_i])
            if gid >= 0 and counts[gid] > 0:
                grad_act[wp_i, :2] += g_u_xy[gid] / counts[gid]
        return self.backward_params(grad_act, map_cache["mlp"])


def train_chipload_shac(
    dev_bundles: Sequence[MillingTaskBundle],
    spec: Optional[ChiploadSpec] = None,
    *,
    epochs: int = 45,
    lr: float = 0.035,
    seed: int = 0,
) -> LocalCWEPolicy:
    """Train a zero-shot local-CWE neural policy via 1st-order analytical SHAC gradients."""
    models = [SubsegmentedCWEModel(b, spec=spec) for b in dev_bundles]
    policy = LocalCWEPolicy(seed=seed)
    params = policy.get_flat_params()
    m_adam = np.zeros_like(params)
    v_adam = np.zeros_like(params)

    for ep in range(1, epochs + 1):
        grad_acc = np.zeros_like(params)
        for mdl in models:
            theta, mcache = policy.map_to_model_theta(mdl)
            _, g_theta, _ = mdl.loss_and_grad(theta, use_ste=True)
            grad_acc += policy.pullback_theta_grad(mdl, g_theta, mcache)
        grad_acc /= max(len(models), 1)
        gnorm = float(np.linalg.norm(grad_acc))
        if gnorm > 5.0:
            grad_acc *= 5.0 / gnorm
        m_adam = 0.9 * m_adam + 0.1 * grad_acc
        v_adam = 0.999 * v_adam + 0.001 * (grad_acc ** 2)
        m_hat = m_adam / (1.0 - 0.9 ** ep)
        v_hat = v_adam / (1.0 - 0.999 ** ep)
        params = params - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        policy.set_flat_params(params)

    return policy


def train_chipload_ppo(
    dev_bundles: Sequence[MillingTaskBundle],
    spec: Optional[ChiploadSpec] = None,
    *,
    epochs: int = 45,
    pop_per_epoch: int = 8,
    sigma: float = 0.25,
    lr: float = 0.02,
    seed: int = 0,
) -> LocalCWEPolicy:
    """Train the identical ``LocalCWEPolicy`` via zero-order clipped PPO / score-function rollouts."""
    rng = np.random.default_rng(seed)
    models = [SubsegmentedCWEModel(b, spec=spec) for b in dev_bundles]
    policy = LocalCWEPolicy(seed=seed)
    params = policy.get_flat_params()
    m_adam = np.zeros_like(params)
    v_adam = np.zeros_like(params)

    for ep in range(1, epochs + 1):
        grad_est = np.zeros_like(params)
        mdl = models[(ep - 1) % len(models)]
        mu_act, mcache = policy.forward_actions(mdl.nominal_obs)

        losses: List[float] = []
        eps_samples: List[np.ndarray] = []
        for _ in range(pop_per_epoch):
            eps = rng.normal(0.0, sigma, size=mu_act.shape)
            act_pert = np.clip(mu_act + eps, -3.5, 3.5)
            # Map perturbed action to model theta
            u_xy = np.zeros((mdl.n_groups, 2), dtype=np.float64)
            counts = np.zeros(mdl.n_groups, dtype=np.float64)
            for wp_i in range(1, mdl.m_segs):
                gid = int(mdl.wp_group[wp_i])
                if gid >= 0:
                    u_xy[gid] += act_pert[wp_i, :2]
                    counts[gid] += 1.0
            nz = counts > 0
            u_xy[nz] /= counts[nz, None]
            th_sample = np.concatenate([u_xy.reshape(-1), act_pert[:, 2]])
            l_val, _, _ = mdl.loss_and_grad(th_sample, use_ste=True)
            losses.append(l_val)
            eps_samples.append(eps)

        arr_l = np.asarray(losses, dtype=np.float64)
        adv = (arr_l - np.mean(arr_l)) / (np.std(arr_l) + 1e-6)  # minimize loss
        grad_mu = np.zeros_like(mu_act)
        for a_i, eps in zip(adv, eps_samples):
            grad_mu += (a_i / (sigma * sigma * pop_per_epoch)) * eps
        grad_est = policy.backward_params(grad_mu, mcache)
        gnorm = float(np.linalg.norm(grad_est))
        if gnorm > 5.0:
            grad_est *= 5.0 / gnorm
        m_adam = 0.9 * m_adam + 0.1 * grad_est
        v_adam = 0.999 * v_adam + 0.001 * (grad_est ** 2)
        m_hat = m_adam / (1.0 - 0.9 ** ep)
        v_hat = v_adam / (1.0 - 0.999 ** ep)
        params = params - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        policy.set_flat_params(params)

    return policy


def rollout_chipload_policy(
    bundle: MillingTaskBundle,
    policy: LocalCWEPolicy,
    spec: Optional[ChiploadSpec] = None,
    *,
    provenance: str = "diffcam:shac_chipload",
    verify_posted: bool = False,
) -> ChiploadOptResult:
    """Zero-shot rollout of a trained ``LocalCWEPolicy`` on ``bundle`` with optional posted safety check."""
    t0 = time.perf_counter()
    model = SubsegmentedCWEModel(bundle, spec=spec)
    theta, _ = policy.map_to_model_theta(model)
    res_tol = float(bundle.task.tolerances.completion_residual_mm3)

    from cam.config import MachineConfig
    from cam.posted_evaluator import evaluate_posted_gcode
    from cam.posts import post_motion_plan

    m_cfg = MachineConfig(
        units="mm",
        tool_number=bundle.task.tool.tool_number,
        length_offset=bundle.task.tool.tool_number,
        spindle_rpm=int(round(model.spec.spindle_rpm)),
        coolant=True,
    )

    chosen_plan: Optional[C.MotionPlan] = None
    chosen_theta = theta.copy()
    posted_dict: Optional[Dict[str, Any]] = None
    for xy_scale in (1.0, 0.5, 0.0):
        _, _, diag = model.loss_and_grad(theta, use_ste=True, xy_scale_override=xy_scale)
        if diag["dexel_residual_mm3"] > res_tol * 0.95 and xy_scale > 0.0:
            continue
        cand_plan = model.to_motion_plan(
            theta, xy_scale_override=xy_scale, provenance=provenance
        )
        if verify_posted:
            nc_str = post_motion_plan(cand_plan, bundle.task, dialect=C.Dialect.HAAS, config=m_cfg)
            rep = evaluate_posted_gcode(
                bundle.task,
                nc_str.encode("utf-8"),
                dialect=C.Dialect.HAAS,
                settings={"max_step_mm": 0.125, "compute_bounds": False, "safety_screen": True},
            )
            orig_m = rep.metrics.get("original_part_gouge_volume_mm3")
            orig_gouge = float(orig_m.value) if orig_m is not None and orig_m.value is not None else 0.0
            res_vol = float(rep.metrics["residual_volume_mm3"].value)
            if not (rep.geometry_pass and rep.safety_pass and orig_gouge <= 1e-6 and res_vol <= res_tol) and xy_scale > 0.0:
                continue
            posted_dict = {
                "geometry_pass": bool(rep.geometry_pass),
                "safety_pass": bool(rep.safety_pass),
                "original_part_gouge_mm3": round(orig_gouge, 5),
                "residual_mm3": round(res_vol, 4),
                "xy_scale_applied": xy_scale,
            }
        chosen_plan = cand_plan
        if xy_scale < 1.0:
            u_xy = theta[: model.num_xy_params]
            chosen_theta[: model.num_xy_params] = np.arctanh(
                np.clip(xy_scale * np.tanh(u_xy), -0.999, 0.999)
            )
        break

    assert chosen_plan is not None
    pts_opt, _, _ = model.decode(chosen_theta)
    xy_shifts = np.hypot(
        pts_opt[:, 0] - model.ref_pts[:, 0], pts_opt[:, 1] - model.ref_pts[:, 1]
    )
    telemetry = analyze_plan_engagement(bundle, chosen_plan, model.spec)
    return ChiploadOptResult(
        plan=chosen_plan,
        theta=chosen_theta,
        telemetry=telemetry,
        mean_xy_shift_mm=float(np.mean(xy_shifts)),
        max_xy_shift_mm=float(np.max(xy_shifts)),
        wall_time_s=time.perf_counter() - t0,
        posted_eval=posted_dict,
    )


__all__ = [
    "ChiploadSpec",
    "ChiploadOptResult",
    "LocalCWEPolicy",
    "SubsegmentedCWEModel",
    "radial_chip_thinning_factor",
    "build_target_heightfield_25d",
    "analyze_plan_engagement",
    "subdivide_motion_plan",
    "blended_durations_with_pullback",
    "retime_motion_plan_constant_speed",
    "shortcut_verified_air_detours",
    "optimize_chipload_trajectory",
    "train_chipload_shac",
    "train_chipload_ppo",
    "rollout_chipload_policy",
]
