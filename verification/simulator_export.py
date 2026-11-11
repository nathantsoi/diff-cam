# SPDX-License-Identifier: MIT
"""Clean-room MIT simulator bundle exporters (.rustcnc, .camotics, WebGL viewer).

Exports optimized and baseline DiffCAM toolpaths + resulting stock models into:
1. ``.rustcnc`` Manufacture Workspace containers (``rustcnc-container/1``) that
   open directly on the RustCNC Manufacture page with the 3D parametric CAD
   target part, billet ``FixedBox`` stock, ``T1`` 5.0 mm carbide flat endmill,
   and pre-populated ``rustcnc-toolpath/1`` streams + domain-tagged SHA-256
   ``manifest.json`` cache entries so clicking **Simulate** immediately displays
   the 3D toolpath and carves the dexel stock (``StockSim``).
2. ``.camotics`` project files pairing billet bounds, ``T1`` 5.0 mm flat endmill,
   resolution, and posted ``.nc`` G-code for one-click opening in CAMotics GUI.
3. Compact JSON + binary ``.stl`` bundles consumed by the local ``127.0.0.1``
   WebGL 3D Toolpath & Resulting Stock Viewer (``viewer/``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import shutil
import struct
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from algorithms.chipload_opt import (
    SubsegmentedCWEModel,
    analyze_plan_engagement,
    shortcut_verified_air_detours,
    subdivide_motion_plan,
)
from algorithms.rl_policy import DirectSegmentActionMap
from benchmarks import contracts as C
from benchmarks.milling_suite import (
    CORNER_RADIUS_KEY,
    MillingTaskBundle,
    make_joint_parameterization,
    multi_pocket_fields,
)
from cam.config import MachineConfig
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from verification.camotics_adapter import (
    CamoticsAdapter,
    _prepare_camotics_nc,
    build_camotics_project,
)

RUSTCNC_CONTAINER_FORMAT = "rustcnc-container/1"
RUSTCNC_DESIGN_FORMAT = "rustcnc/22"
RUSTCNC_CAM_FORMAT = "rustcnc-cam/1"
RUSTCNC_TOOL_LIB_FORMAT = "rustcnc-tool-library/1"
RUSTCNC_VIEW_FORMAT = "rustcnc-cam-view/1"
RUSTCNC_TOOLPATH_FORMAT = "rustcnc-toolpath/1"
RUSTCNC_CACHE_FORMAT = "rustcnc-cam-cache/1"

_SETUP_STAMP_TAG = b"rustcnc-cam/setup-inputs/1\n"
_OP_STAMP_TAG = b"rustcnc-cam/op-inputs/1\n"

_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

_FAMILY_TO_RUSTCNC_STRATEGY: Dict[str, Tuple[str, str]] = {
    "facing_open_boss": ("Face", "cam.face"),
    "pocket_island": ("Pocket2D", "cam.pocket2d"),
    "adaptive_channel": ("Adaptive2D", "cam.adaptive2d"),
    "contour_profile": ("Contour2D", "cam.contour2d"),
}


def safe_resolve_under(root: Path, relative: str) -> Path:
    """Resolve ``relative`` strictly inside ``root`` (CWE-22 path traversal guard)."""
    if not isinstance(relative, str) or not relative or "\x00" in relative or "\\" in relative:
        raise ValueError("Invalid relative path.")
    rel_path = Path(relative)
    if rel_path.is_absolute() or any(part in ("", ".", "..") for part in rel_path.parts):
        raise ValueError(f"Unsafe relative path rejected: {relative!r}")
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / rel_path).resolve()
    root_prefix = str(root_resolved) + os.sep
    if not str(candidate).startswith(root_prefix):
        raise ValueError(f"Path escapes sandbox root: {relative!r}")
    return candidate


def _deterministic_uuid(key: str) -> uuid.UUID:
    """Mint a deterministic v5 UUID so container exports are 100% reproducible."""
    return uuid.uuid5(_UUID_NAMESPACE, f"diffcam.rustcnc.v1:{key}")


def narrow_uuid_u64(u: uuid.UUID) -> int:
    """Upper 64 bits of ``u``, matching ``rustcnc_cam::generation_adapter::narrow_id``."""
    return (u.int >> 64) & 0xFFFFFFFFFFFFFFFF


def _fmt_mm(val: float) -> str:
    """Format a finite length value with explicit ``mm`` unit suffix."""
    if not math.isfinite(val):
        raise ValueError(f"Non-finite millimetre value: {val!r}")
    rounded = round(float(val), 6)
    if abs(rounded) < 1e-12:
        rounded = 0.0
    text = f"{rounded:g}"
    return f"{text} mm"


def reconstruct_motion_plan(
    bundle: MillingTaskBundle,
    method_name: str,
    best_theta: Optional[Sequence[float]],
    *,
    shortcut_mode: str = "none",
) -> C.MotionPlan:
    """Reconstruct a validated ``MotionPlan`` on ``bundle.task`` from ``best_theta``."""
    if method_name in ("Nominal-CAM", "Nominal") or best_theta is None:
        return bundle.nominal_motion

    theta_arr = np.asarray(best_theta, dtype=np.float64).reshape(-1)
    if method_name in ("DiffCAM-Chipload", "Chipload-STE", "SHAC-Chipload", "PPO-Chipload"):
        slug = method_name.lower().replace("-", "_")
        raw_plan = SubsegmentedCWEModel(bundle).to_motion_plan(
            theta_arr, provenance=f"diffcam:{slug}"
        )
        if shortcut_mode and shortcut_mode != "none":
            return shortcut_verified_air_detours(bundle, raw_plan, mode=shortcut_mode)
        return raw_plan
    task = bundle.task
    nom = bundle.nominal_motion

    if method_name in ("DiffPolicy-SHAC", "Hard-PPO"):
        param_map: Any = DirectSegmentActionMap(task, nom)
    else:
        param_map = make_joint_parameterization(task, nom)

    positions = param_map.positions(theta_arr)
    feed_mult = param_map.forward_feed_multipliers(theta_arr)
    to_mach = task.work_to_machine.apply

    raw_segs: List[C.Segment] = []
    for orig, sw, ew, mult in zip(nom.segments, positions[:-1], positions[1:], feed_mult):
        s_m = to_mach((float(sw[0]), float(sw[1]), float(sw[2])))
        e_m = to_mach((float(ew[0]), float(ew[1]), float(ew[2])))
        length = math.dist(s_m, e_m)
        eff_f = None if orig.kind == C.SegmentKind.RAPID else float(orig.feed_mm_s * float(mult))
        dur = length / (task.limits.rapid_mm_s if eff_f is None else eff_f)
        raw_segs.append(
            C.Segment(
                id=orig.id,
                kind=orig.kind,
                start_mm=s_m,
                end_mm=e_m,
                duration_s=max(dur, 1e-9),
                feed_mm_s=eff_f,
                tag=orig.tag,
            )
        )

    if task.motion_semantics.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        durs = C.blended_segment_durations(
            raw_segs, task.limits.rapid_mm_s, task.limits.max_accel_mm_s2
        )
        segs = tuple(dataclasses.replace(s, duration_s=float(d)) for s, d in zip(raw_segs, durs))
    else:
        segs = tuple(raw_segs)

    slug = method_name.lower().replace("-", "_").replace(" ", "_")
    plan = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=task.initial_state,
        segments=segs,
        timing_rule=task.motion_semantics.timing_rule,
        provenance=f"diffcam:{slug}",
    )
    plan.validate_for_task(task)
    return plan


def build_target_cad_document(bundle: MillingTaskBundle) -> Dict[str, Any]:
    """Construct a parametric ``rustcnc/22`` CAD ``Document`` for ``bundle.task``."""
    task = bundle.task
    geom = task.operation_target.geometry
    p = geom.params

    smin = [float(v) for v in p["box_min_mm"]]
    smax = [float(v) for v in p["box_max_mm"]]
    pmin = [float(v) for v in p["pocket_min_mm"]]
    pmax = [float(v) for v in p["pocket_max_mm"]]
    islands = p.get("islands_mm", ())
    corner_radius = float(p.get(CORNER_RADIUS_KEY, 0.0))

    features: List[Dict[str, Any]] = []
    fid = 1

    # 1. Base stock-envelope block (Body1)
    features.append(
        {
            "id": fid,
            "name": "BilletEnvelope",
            "suppressed": False,
            "body_names": ["Body1"],
            "type": "box",
            "corner": [smin[0], smin[1], smin[2]],
            "length": _fmt_mm(smax[0] - smin[0]),
            "width": _fmt_mm(smax[1] - smin[1]),
            "height": _fmt_mm(smax[2] - smin[2]),
            "operation": "new_body",
        }
    )
    fid += 1

    # 2. Subtract pocket / facing / outer-contour cavity
    cav_lx = pmax[0] - pmin[0]
    cav_ly = pmax[1] - pmin[1]
    cav_lz = (pmax[2] + 0.5) - pmin[2]
    if cav_lx > 1e-6 and cav_ly > 1e-6 and pmax[2] - pmin[2] > 1e-6:
        features.append(
            {
                "id": fid,
                "name": "MachinedCavity",
                "suppressed": False,
                "type": "box",
                "corner": [pmin[0], pmin[1], pmin[2]],
                "length": _fmt_mm(cav_lx),
                "width": _fmt_mm(cav_ly),
                "height": _fmt_mm(cav_lz),
                "operation": "cut",
            }
        )
        fid += 1

    # 3. Join retained island / standing boss features
    for idx, (imin, imax) in enumerate(islands, start=1):
        ilx = float(imax[0]) - float(imin[0])
        ily = float(imax[1]) - float(imin[1])
        ilz = float(imax[2]) - float(imin[2])
        if ilx > 1e-6 and ily > 1e-6 and ilz > 1e-6:
            features.append(
                {
                    "id": fid,
                    "name": f"RetainedFeature{idx}",
                    "suppressed": False,
                    "type": "box",
                    "corner": [float(imin[0]), float(imin[1]), float(imin[2])],
                    "length": _fmt_mm(ilx),
                    "width": _fmt_mm(ily),
                    "height": _fmt_mm(ilz),
                    "operation": "join",
                }
            )
            fid += 1

    # 4. Exact concave pocket corner fillets (matching derive_multi_pocket_target)
    if corner_radius > 1e-6 and (pmax[2] - pmin[2]) > 1e-6:
        depth = pmax[2] - pmin[2]
        c_idx = 1
        for x_side in (0, 1):
            x_walled = (pmin[0] > smin[0] + 1e-9) if x_side == 0 else (pmax[0] < smax[0] - 1e-9)
            if not x_walled:
                continue
            for y_side in (0, 1):
                y_walled = (pmin[1] > smin[1] + 1e-9) if y_side == 0 else (pmax[1] < smax[1] - 1e-9)
                if not y_walled:
                    continue
                cx = pmin[0] + corner_radius if x_side == 0 else pmax[0] - corner_radius
                cy = pmin[1] + corner_radius if y_side == 0 else pmax[1] - corner_radius
                bx0 = pmin[0] if x_side == 0 else pmax[0] - corner_radius
                by0 = pmin[1] if y_side == 0 else pmax[1] - corner_radius
                features.append(
                    {
                        "id": fid,
                        "name": f"CornerBlock{c_idx}",
                        "suppressed": False,
                        "type": "box",
                        "corner": [bx0, by0, pmin[2]],
                        "length": _fmt_mm(corner_radius),
                        "width": _fmt_mm(corner_radius),
                        "height": _fmt_mm(depth),
                        "operation": "join",
                    }
                )
                fid += 1
                features.append(
                    {
                        "id": fid,
                        "name": f"CornerFillet{c_idx}",
                        "suppressed": False,
                        "type": "cylinder",
                        "base": [cx, cy, pmin[2] - 0.1],
                        "axis": [0.0, 0.0, 1.0],
                        "radius": _fmt_mm(corner_radius),
                        "height": _fmt_mm(depth + 0.5),
                        "operation": "cut",
                    }
                )
                fid += 1
                c_idx += 1

    return {
        "format": RUSTCNC_DESIGN_FORMAT,
        "name": f"DiffCAM {bundle.case.case_id}",
        "units": "mm",
        "params": [],
        "features": features,
        "next_id": fid,
        "next_component_color_index": 0,
    }


def build_rustcnc_tool_library(tool: C.ToolAssembly, tool_uuid: uuid.UUID) -> Dict[str, Any]:
    """Construct ``cam/tools.json`` (``rustcnc-tool-library/1``) for ``tool``."""
    diam = 2.0 * float(tool.cutter_radius_mm)
    shank_diam = 2.0 * float(tool.shank_radius_mm)
    holder_diam = 2.0 * float(tool.holder_radius_mm)
    flute_len = float(tool.flute_length_mm)
    below_holder = max(float(tool.holder_bottom_mm), flute_len + 2.0)
    holder_h = max(float(tool.holder_height_mm), 20.0)
    overall_len = below_holder + holder_h

    return {
        "format": RUSTCNC_TOOL_LIB_FORMAT,
        "schema_version": 1,
        "name": "DiffCAM Project Tools",
        "unit": "mm",
        "notes": "Clean-room MIT 5.0 mm flat carbide endmill for 3-axis milling simulation.",
        "tools": [
            {
                "id": str(tool_uuid),
                "tool_type": "flat_end_mill",
                "description": f"T{tool.tool_number} {diam:g} mm Flat Carbide Endmill",
                "vendor": "DiffCAM Clean-Room",
                "product_id": f"DC-FLAT-{diam:g}MM",
                "product_link": "",
                "body_material": "carbide",
                "grade": None,
                "reference_guid": None,
                "last_modified": None,
                "geometry": {
                    "diameter": diam,
                    "corner_radius": 0.0,
                    "flute_length": flute_len,
                    "length_below_holder": below_holder,
                    "overall_length": overall_len,
                    "shank_diameter": shank_diam,
                    "flutes": 3,
                },
                "shank": {
                    "segments": [
                        {
                            "height": max(below_holder - flute_len, 1.0),
                            "lower_diameter": shank_diam,
                            "upper_diameter": shank_diam,
                        }
                    ]
                },
                "holder": {
                    "description": "CAT40 / ER20 Collet Chuck",
                    "vendor": "DiffCAM Clean-Room",
                    "gauge_length": holder_h,
                    "segments": [
                        {
                            "height": holder_h * 0.4,
                            "lower_diameter": max(shank_diam * 1.8, holder_diam * 0.75),
                            "upper_diameter": holder_diam,
                        },
                        {
                            "height": holder_h * 0.6,
                            "lower_diameter": holder_diam,
                            "upper_diameter": holder_diam,
                        },
                    ],
                },
                "turning_holder": None,
                "post": {
                    "number": int(tool.tool_number),
                    "length_offset": int(tool.tool_number),
                    "diameter_offset": int(tool.tool_number),
                    "turret": 0,
                    "compensation_offset": 0,
                    "comment": "5mm 3FL Carbide Endmill",
                    "manual_tool_change": False,
                    "break_control": False,
                    "live": True,
                },
                "turning_setup": None,
                "presets": [
                    {
                        "name": "Aluminum 6061",
                        "description": "Haas Mini Mill NGC 8000 RPM Flood",
                        "guid": None,
                        "material": {"category": "metal", "query": "aluminum", "use_hardness": False},
                        "spindle_rpm": 8000.0,
                        "ramp_rpm": 8000.0,
                        "feed_cutting": 360.0,
                        "feed_plunge": 180.0,
                        "feed_ramp": 240.0,
                        "feed_lead_in": 300.0,
                        "feed_lead_out": 300.0,
                        "feed_retract": 600.0,
                        "feed_transition": 600.0,
                        "coolant": "flood",
                    }
                ],
            }
        ],
    }


def _cam_put_json(hasher: "hashlib._Hash", value: Any) -> None:
    """Length-prefixed compact JSON update matching ``rustcnc_cam::cache::put``."""
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    hasher.update(struct.pack("<Q", len(raw)))
    hasher.update(raw)


def _cam_put_str(hasher: "hashlib._Hash", text: str) -> None:
    raw = text.encode("utf-8")
    hasher.update(struct.pack("<Q", len(raw)))
    hasher.update(raw)


def compute_rustcnc_setup_stamp(setup: Mapping[str, Any]) -> bytes:
    """Compute exact 32-byte ``setup_toolpath_stamp`` matching ``rustcnc-cam/src/cache.rs``."""
    h = hashlib.sha256(_SETUP_STAMP_TAG)
    _cam_put_json(h, setup["operation_type"])
    _cam_put_json(h, setup["models"])
    _cam_put_json(h, setup["wcs"])
    _cam_put_json(h, setup["stock"])
    _cam_put_json(h, setup["fixture"])
    _cam_put_json(h, setup.get("rotary_axis", None))
    _cam_put_json(h, setup.get("machine", None))
    uses_prev = bool(setup.get("uses_previous_setup", False))
    h.update(bytes([1 if uses_prev else 0]))
    if uses_prev:
        h.update(b"previous:none")
    return h.digest()


def compute_rustcnc_op_stamp_hex(setup: Mapping[str, Any], op: Mapping[str, Any]) -> str:
    """Compute exact 64-hex-char ``op_stamp`` matching ``rustcnc-cam/src/cache.rs``."""
    h = hashlib.sha256(_OP_STAMP_TAG)
    _cam_put_str(h, str(op["strategy"]))
    h.update(uuid.UUID(str(op["tool"])).bytes)
    h.update(bytes([1 if op.get("suppressed", False) else 0]))
    _cam_put_json(h, op["parameters"])
    _cam_put_json(h, op["geometry"])
    h.update(uuid.UUID(str(op["setup"])).bytes)
    h.update(compute_rustcnc_setup_stamp(setup))
    return h.hexdigest()


def _classify_rustcnc_move(seg: C.Segment, z_top_mm: float = 0.0) -> str:
    """Map a ``C.Segment`` into ``rustcnc_cam_gen::stream::MotionClass``."""
    if seg.kind == C.SegmentKind.RAPID:
        return "Rapid"
    tag = (seg.tag or "").lower()
    if "helix" in tag:
        return "RampHelix"
    if "ramp" in tag:
        return "Ramp"
    if "lead_in" in tag or "leadin" in tag:
        return "LeadIn"
    if "lead_out" in tag or "leadout" in tag:
        return "LeadOut"
    if "finish" in tag:
        return "FinishCutting"
    if "link" in tag or "retract" in tag:
        return "LinkTransition"
    dx = seg.end_mm[0] - seg.start_mm[0]
    dy = seg.end_mm[1] - seg.start_mm[1]
    dz = seg.end_mm[2] - seg.start_mm[2]
    xy = math.hypot(dx, dy)
    if dz < -1e-6 and xy <= 1e-5:
        return "Plunge"
    if dz < -1e-6 and xy > 1e-5:
        return "Ramp"
    if dz > 1e-6 and seg.end_mm[2] > z_top_mm + 0.25:
        return "LinkTransition"
    return "Cutting"


def _build_toolpath_part(
    bundle: MillingTaskBundle,
    plan: C.MotionPlan,
    op_uuid: uuid.UUID,
    tool_uuid: uuid.UUID,
    strategy_stream_id: str,
    spindle_rpm: float = 8000.0,
) -> Dict[str, Any]:
    """Serialize ``plan`` into a ``rustcnc-toolpath/1`` stream JSON object."""
    inv = bundle.task.work_to_machine.inverse().apply
    z_top = float(bundle.case.stock_max_mm[2])
    tool_u64 = narrow_uuid_u64(tool_uuid)

    records: List[Dict[str, Any]] = [
        {"Spindle": {"rpm": float(spindle_rpm), "direction": "Clockwise"}},
        {"Coolant": {"flood": True, "mist": False}},
    ]

    for seg in plan.segments:
        sw = inv(seg.start_mm)
        ew = inv(seg.end_mm)
        if math.dist(sw, ew) <= 1e-9:
            continue
        m_class = _classify_rustcnc_move(seg, z_top_mm=z_top)
        if seg.kind == C.SegmentKind.RAPID or seg.feed_mm_s is None:
            feed_spec: Any = "Rapid"
        else:
            feed_spec = {"MmPerMin": round(float(seg.feed_mm_s) * 60.0, 4)}
        records.append(
            {
                "Move": {
                    "class": m_class,
                    "start": [round(float(sw[0]), 6), round(float(sw[1]), 6), round(float(sw[2]), 6)],
                    "end": [round(float(ew[0]), 6), round(float(ew[1]), 6), round(float(ew[2]), 6)],
                    "arc": None,
                    "feed": feed_spec,
                    "pass": 0,
                }
            }
        )

    records.append({"Coolant": {"flood": False, "mist": False}})
    records.append({"Spindle": {"rpm": 0.0, "direction": "Off"}})

    # Identity work->model 4x4 matrix because CAD features in document.json are authored in C.Frame.WORK.
    identity_frame = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    input_hash = int.from_bytes(
        hashlib.sha256(f"{bundle.case.case_id}:{op_uuid}".encode("utf-8")).digest()[:8],
        byteorder="little",
    )
    return {
        "format": RUSTCNC_TOOLPATH_FORMAT,
        "op": str(op_uuid),
        "tool_id": tool_u64,
        "tool_number": int(bundle.task.tool.tool_number),
        "stamp": {
            "artifact_version": 1,
            "input_hash": input_hash,
            "strategy_id": strategy_stream_id,
            "strategy_version": 1,
            "op_param_hash": 0,
            "tools": [[tool_u64, 1]],
            "geometry_hashes": [],
            "stock_hash": None,
            "frame_hash": 1,
        },
        "frame": identity_frame,
        "records": records,
        "messages": [
            {
                "severity": "Info",
                "text": f"DiffCAM clean-room stream ({plan.provenance}), {len(plan.segments)} segments.",
            }
        ],
    }


def _build_rustcnc_setup(
    bundle: MillingTaskBundle,
    setup_uuid: uuid.UUID,
    setup_name: str,
    op_uuid: uuid.UUID,
    op_name: str,
    tool_uuid: uuid.UUID,
    strategy_name: str,
    notes: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Construct one ``Setup`` + ``Operation`` in exact ``rustcnc-cam/src/model.rs`` field order."""
    smin = bundle.case.stock_min_mm
    smax = bundle.case.stock_max_mm
    lx = smax[0] - smin[0]
    ly = smax[1] - smin[1]
    lz = smax[2] - smin[2]

    op_dict: Dict[str, Any] = {
        "id": str(op_uuid),
        "setup": str(setup_uuid),
        "strategy": strategy_name,
        "parameters": {},
        "tool": str(tool_uuid),
        "geometry": {},
        "suppressed": False,
        "optional": False,
        "protected": False,
        "name": op_name,
        "tool_preset": "Aluminum 6061",
    }

    setup_dict: Dict[str, Any] = {
        "id": str(setup_uuid),
        "name": setup_name,
        "operation_type": "milling",
        "models": {"refs": []},
        "wcs": {
            "orientation": {"kind": "model_orientation"},
            "origin": {"mode": "stock_box_point", "box_point": "top_center"},
        },
        "stock": {
            "mode": {"default": "'default'", "override_expr": "'fixedbox'"},
            "body": {
                "mode": "fixed_box",
                "x": {"default": "", "override_expr": _fmt_mm(lx)},
                "y": {"default": "", "override_expr": _fmt_mm(ly)},
                "z": {"default": "", "override_expr": _fmt_mm(lz)},
                "x_mode": "model",
                "y_mode": "model",
                "z_mode": "model",
                "x_offset": {"default": "0 mm", "override_expr": _fmt_mm(smin[0])},
                "y_offset": {"default": "0 mm", "override_expr": _fmt_mm(smin[1])},
                "z_offset": {"default": "0 mm", "override_expr": _fmt_mm(smin[2])},
                "rounding": {"default": "0 mm"},
                "ground_at_model_origin": True,
            },
            "axis_enabled": False,
            "specific_cutting_force": {"default": "1500 N/mm2"},
            "initial_tool_wear": {"default": "0 J"},
            "bounding_solid_wcs": False,
        },
        "fixture": {
            "bodies": {"refs": []},
            "use_fixture": False,
            "radial_clearance": {"default": "5 mm"},
            "axial_clearance": {"default": "5 mm"},
            "position": {
                "mode": "fixture_point",
                "box_point": "bottom_center",
                "x_offset": {"default": "0 mm"},
                "y_offset": {"default": "0 mm"},
                "z_offset": {"default": "0 mm"},
            },
        },
        "program": {
            "program_name": "1001",
            "comment": f"DiffCAM {bundle.case.case_id}",
            "work_offset": 1,
            "multiple_work_offsets": False,
            "work_offset_duplicates": 1,
            "work_offset_increment": 1,
            "safe_z_offset": {"default": "stockZHigh + 5 mm"},
            "operation_order": "default",
        },
        "uses_previous_setup": False,
        "notes": notes,
        "operations": [op_dict],
    }
    return setup_dict, op_dict


def _validate_zip_entry_name(name: str) -> None:
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError(f"Invalid archive entry name: {name!r}")


def export_rustcnc_project(
    output_path: Path,
    bundle: MillingTaskBundle,
    variant_plans: Sequence[Tuple[str, str, C.MotionPlan]],
) -> Dict[str, Any]:
    """Write a complete ``.rustcnc`` Manufacture Workspace container.

    Parameters
    ----------
    output_path : Path
        Destination ``.rustcnc`` file path.
    bundle : MillingTaskBundle
        Benchmark case bundle containing ``TaskSpec`` and ``MillingCaseSpec``.
    variant_plans : Sequence[Tuple[str, str, C.MotionPlan]]
        Ordered ``(variant_id, display_label, motion_plan)`` tuples. Each variant
        is placed in its own independent ``Setup`` (with the first setup active
        by default) so clicking **Simulate** in the RustCNC Manufacture page
        carves fresh billet stock for the selected method without interference.
    """
    if not variant_plans:
        raise ValueError("variant_plans must contain at least one (id, label, MotionPlan).")

    cid = bundle.case.case_id
    strategy_name, strategy_stream_id = _FAMILY_TO_RUSTCNC_STRATEGY.get(
        bundle.case.family, ("Pocket2D", "cam.pocket2d")
    )
    tool_uuid = _deterministic_uuid(f"{cid}:tool:T1")

    doc_json_obj = build_target_cad_document(bundle)
    doc_bytes = json.dumps(doc_json_obj, indent=2).encode("utf-8")
    design_sha256 = hashlib.sha256(doc_bytes).hexdigest()

    tools_json_obj = build_rustcnc_tool_library(bundle.task.tool, tool_uuid)
    tools_bytes = json.dumps(tools_json_obj, indent=2).encode("utf-8")

    setups: List[Dict[str, Any]] = []
    nc_programs: List[Dict[str, Any]] = []
    op_records: List[Tuple[uuid.UUID, Dict[str, Any], Dict[str, Any], C.MotionPlan]] = []

    for idx, (var_id, var_label, plan) in enumerate(variant_plans, start=1):
        setup_uuid = _deterministic_uuid(f"{cid}:setup:{var_id}")
        op_uuid = _deterministic_uuid(f"{cid}:op:{var_id}")
        nc_uuid = _deterministic_uuid(f"{cid}:nc:{var_id}")
        dur_s = sum(s.duration_s for s in plan.segments)
        setup_dict, op_dict = _build_rustcnc_setup(
            bundle=bundle,
            setup_uuid=setup_uuid,
            setup_name=f"Setup {idx} - {var_label} ({dur_s:.2f}s)",
            op_uuid=op_uuid,
            op_name=f"{strategy_name} - {var_label}",
            tool_uuid=tool_uuid,
            strategy_name=strategy_name,
            notes=f"{cid} | {var_label} | {len(plan.segments)} segments | {dur_s:.3f} s cycle time",
        )
        setups.append(setup_dict)
        nc_programs.append(
            {
                "id": str(nc_uuid),
                "name": f"NC_{cid}_{var_id}",
                "entries": [{"kind": "setup", "id": str(setup_uuid)}],
                "output": {
                    "program_name": str(1000 + idx),
                    "file_name": f"{cid}_{var_id}",
                    "extension": "nc",
                },
            }
        )
        op_records.append((op_uuid, setup_dict, op_dict, plan))

    cam_doc_obj: Dict[str, Any] = {
        "format": RUSTCNC_CAM_FORMAT,
        "schema_version": 1,
        "name": f"CAM Root ({cid})",
        "design": {"kind": "embedded"},
        "setups": setups,
        "nc_programs": nc_programs,
        "folders": [],
        "tool_assignments": [
            {"tool": str(tool_uuid), "number": int(bundle.task.tool.tool_number)}
        ],
        "metadata": {
            "generator": "diffcam.verification.simulator_export",
            "license": "MIT",
            "case_id": cid,
            "family": bundle.case.family,
        },
    }
    cam_bytes = json.dumps(cam_doc_obj, indent=2).encode("utf-8")
    cam_sha256 = hashlib.sha256(cam_bytes).hexdigest()

    active_setup_id = setups[0]["id"]
    hidden_ops = [str(rec[0]) for rec in op_records[1:]]
    view_obj = {
        "format": RUSTCNC_VIEW_FORMAT,
        "active_setup": active_setup_id,
        "hidden_toolpaths": hidden_ops,
    }
    view_bytes = json.dumps(view_obj, indent=2).encode("utf-8")

    manifest_entries: List[Dict[str, Any]] = []
    toolpath_parts: List[Tuple[str, bytes]] = []
    for op_uuid, setup_dict, op_dict, plan in op_records:
        tp_obj = _build_toolpath_part(
            bundle=bundle,
            plan=plan,
            op_uuid=op_uuid,
            tool_uuid=tool_uuid,
            strategy_stream_id=strategy_stream_id,
        )
        tp_bytes = json.dumps(tp_obj, separators=(",", ":")).encode("utf-8")
        tp_digest_u64 = hashlib.sha256(tp_bytes).hexdigest()[:16]
        tp_part_name = f"cache/cam/{op_uuid}.{tp_digest_u64}.toolpath"
        toolpath_parts.append((tp_part_name, tp_bytes))

        op_stamp_hex = compute_rustcnc_op_stamp_hex(setup_dict, op_dict)
        manifest_entries.append(
            {
                "id": str(op_uuid),
                "kind": "toolpath",
                "inputs": {
                    "design": design_sha256,
                    "cam": cam_sha256,
                    "op": op_stamp_hex,
                    "strategy": strategy_name,
                    "tool": str(tool_uuid),
                },
                "outputs": [tp_part_name],
            }
        )

    manifest_obj = {
        "format": RUSTCNC_CACHE_FORMAT,
        "entries": manifest_entries,
    }
    manifest_bytes = json.dumps(manifest_obj, indent=2).encode("utf-8")
    container_bytes = json.dumps({"format": RUSTCNC_CONTAINER_FORMAT}, separators=(",", ":")).encode(
        "utf-8"
    )

    # Write non-ZIP64 archive with unique validated entries in canonical RustCNC order.
    ordered_entries: List[Tuple[str, bytes, int]] = [
        ("document.json", doc_bytes, zipfile.ZIP_DEFLATED),
        ("container.json", container_bytes, zipfile.ZIP_STORED),
        ("cam.json", cam_bytes, zipfile.ZIP_DEFLATED),
        ("cam/tools.json", tools_bytes, zipfile.ZIP_DEFLATED),
        ("cam/view.json", view_bytes, zipfile.ZIP_DEFLATED),
        ("cache/cam/manifest.json", manifest_bytes, zipfile.ZIP_DEFLATED),
    ]
    for tp_name, tp_payload in toolpath_parts:
        ordered_entries.append((tp_name, tp_payload, zipfile.ZIP_DEFLATED))

    seen_names = set()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", allowZip64=False) as zf:
        for entry_name, payload, compression in ordered_entries:
            _validate_zip_entry_name(entry_name)
            if entry_name in seen_names:
                raise ValueError(f"Duplicate archive entry: {entry_name}")
            seen_names.add(entry_name)
            zinfo = zipfile.ZipInfo(filename=entry_name, date_time=(2026, 11, 11, 0, 0, 0))
            zinfo.compress_type = compression
            zinfo.external_attr = 0o644 << 16
            zf.writestr(zinfo, payload)

    archive_bytes = buf.getvalue()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(archive_bytes)

    return {
        "path": str(output_path),
        "sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "size_bytes": len(archive_bytes),
        "active_setup": active_setup_id,
        "setups": [s["name"] for s in setups],
        "toolpath_parts": [p[0] for p in toolpath_parts],
    }


def export_camotics_project(
    output_dir: Path,
    bundle: MillingTaskBundle,
    variant_id: str,
    plan: C.MotionPlan,
    *,
    resolution_mm: float = 0.25,
) -> Dict[str, Any]:
    """Write a self-contained ``<case_id>_<variant_id>.camotics`` project + RS-274 ``.nc`` file."""
    cid = bundle.case.case_id
    nc_name = f"{cid}_{variant_id}.nc"
    proj_name = f"{cid}_{variant_id}.camotics"

    nc_path = safe_resolve_under(output_dir, nc_name)
    proj_path = safe_resolve_under(output_dir, proj_name)
    nc_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = MachineConfig(
        units="mm",
        tool_number=bundle.task.tool.tool_number,
        length_offset=bundle.task.tool.tool_number,
        spindle_rpm=8000,
        coolant=True,
    )
    raw_rs274_nc = post_motion_plan(plan, bundle.task, dialect=C.Dialect.RS274, config=cfg)
    prepared_nc_bytes, _ = _prepare_camotics_nc(
        bundle.task,
        raw_rs274_nc.encode("utf-8"),
        dialect=C.Dialect.RS274,
    )
    nc_path.write_bytes(prepared_nc_bytes)

    camotics_doc = build_camotics_project(
        bundle.task,
        float(resolution_mm),
        program_name=nc_name,
        tool_number=int(bundle.task.tool.tool_number),
    )
    proj_path.write_text(json.dumps(camotics_doc, indent=2), encoding="utf-8")
    return {
        "project_file": proj_name,
        "nc_file": nc_name,
        "resolution_mm": float(resolution_mm),
    }


def voxel_mask_to_binary_stl(
    mask: np.ndarray,
    grid: C.GridSpec,
    *,
    header_label: str = "DiffCAM Clean-Room Voxel Mesh",
) -> bytes:
    """Convert a 3D boolean occupancy ``mask`` into a watertight outward-oriented binary STL."""
    occ = np.asarray(mask, dtype=bool)
    if occ.ndim != 3:
        raise ValueError(f"Expected 3D boolean occupancy array, got shape {occ.shape}")

    nx, ny, nz = occ.shape
    vox = float(grid.voxel_mm)
    ox, oy, oz = (float(v) for v in grid.origin_mm)

    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    tris_list: List[np.ndarray] = []
    norms_list: List[np.ndarray] = []

    # 6 axis-aligned exposed face directions with outward right-hand winding
    face_specs = (
        # (-X): x = x0, normal = (-1, 0, 0)
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        # (+X): x = x0 + vox, normal = (+1, 0, 0)
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        # (-Y): y = y0, normal = (0, -1, 0)
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        # (+Y): y = y0 + vox, normal = (0, +1, 0)
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        # (-Z): z = z0, normal = (0, 0, -1)
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        # (+Z): z = z0 + vox, normal = (0, 0, +1)
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )

    for (dx, dy, dz), corners in face_specs:
        exposed = occ & ~padded[
            1 + dx : 1 + dx + nx,
            1 + dy : 1 + dy + ny,
            1 + dz : 1 + dz + nz,
        ]
        ix, iy, iz = np.nonzero(exposed)
        if len(ix) == 0:
            continue
        x0 = ox + ix.astype(np.float32) * vox
        y0 = oy + iy.astype(np.float32) * vox
        z0 = oz + iz.astype(np.float32) * vox
        pts = [
            np.stack([x0 + c[0] * vox, y0 + c[1] * vox, z0 + c[2] * vox], axis=1)
            for c in corners
        ]
        t1 = np.stack([pts[0], pts[1], pts[2]], axis=1)
        t2 = np.stack([pts[0], pts[2], pts[3]], axis=1)
        n_vec = np.broadcast_to(
            np.array([dx, dy, dz], dtype=np.float32), (len(ix) * 2, 3)
        ).copy()
        tris_list.append(np.concatenate([t1, t2], axis=0))
        norms_list.append(n_vec)

    if tris_list:
        all_tris = np.concatenate(tris_list, axis=0).astype("<f4")
        all_norms = np.concatenate(norms_list, axis=0).astype("<f4")
    else:
        all_tris = np.zeros((0, 3, 3), dtype="<f4")
        all_norms = np.zeros((0, 3), dtype="<f4")

    count = len(all_tris)
    header = header_label.encode("ascii", errors="ignore")[:80].ljust(80, b" ")
    stl_dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    records = np.zeros(count, dtype=stl_dtype)
    if count > 0:
        records["normal"] = all_norms
        records["vertices"] = all_tris
    return header + struct.pack("<I", count) + records.tobytes()


def build_target_part_stl_bytes(bundle: MillingTaskBundle) -> bytes:
    """Build a crisp binary STL mesh of ``bundle.task.operation_target`` in ``C.Frame.WORK``."""
    _, target_sdf = multi_pocket_fields(
        bundle.task.stock, bundle.task.operation_target.geometry, bundle.task.grid
    )
    return voxel_mask_to_binary_stl(
        target_sdf <= 0.0,
        bundle.task.grid,
        header_label=f"DiffCAM Target {bundle.case.case_id}",
    )


def carve_resulting_stock_stl_bytes(
    bundle: MillingTaskBundle,
    plan: C.MotionPlan,
    *,
    use_camotics_docker: bool = False,
    scratch_dir: Optional[Path] = None,
) -> Tuple[bytes, str]:
    """Generate resulting cut-stock binary STL in ``C.Frame.WORK`` coordinates.

    When ``use_camotics_docker=True``, invokes ``CamoticsAdapter`` (``diffcam-camotics:1.2.0``)
    and shifts the emitted STL vertices by ``+initial_state`` into ``C.Frame.WORK``. Otherwise
    uses the exact ``0.125 mm`` station-swept reference voxel carve.
    """
    cfg = MachineConfig(
        units="mm",
        tool_number=bundle.task.tool.tool_number,
        length_offset=bundle.task.tool.tool_number,
        spindle_rpm=8000,
        coolant=True,
    )
    haas_nc = post_motion_plan(plan, bundle.task, dialect=C.Dialect.HAAS, config=cfg)

    if use_camotics_docker and scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        adapter = CamoticsAdapter(
            default_resolution_mm=float(bundle.task.grid.voxel_mm),
            scratch_root=scratch_dir,
        )
        adapter.evaluate_posted_gcode(
            bundle.task,
            haas_nc.encode("utf-8"),
            dialect=C.Dialect.HAAS,
            settings={"resolution_mm": float(bundle.task.grid.voxel_mm), "retain_workdir": True},
        )
        stl_candidates = sorted(scratch_dir.glob("*/cut_workpiece.stl"))
        if stl_candidates:
            sim_to_work_offset = bundle.task.work_to_machine.inverse().apply(
                bundle.task.initial_state.position_mm
            )
            return translate_binary_stl_bytes(
                stl_candidates[0].read_bytes(),
                offset_mm=sim_to_work_offset,
            ), "camotics_v1.2.0"

    # Exact CPU reference carve fallback (fast subgrid oracle on bundle.task.grid)
    from cam.posted_evaluator import RefinementSettings, _execute

    exec_res = _execute(
        bundle.task,
        plan,
        RefinementSettings(max_step_mm=0.125, compute_bounds=False, safety_screen=False),
    )
    return (
        voxel_mask_to_binary_stl(
            exec_res.remaining,
            bundle.task.grid,
            header_label=f"DiffCAM Stock {bundle.case.case_id}",
        ),
        "reference_voxel_0.25mm",
    )


def translate_binary_stl_bytes(stl_bytes: bytes, offset_mm: Sequence[float]) -> bytes:
    """Translate binary STL vertices by ``offset_mm`` (e.g. CAMotics frame -> ``C.Frame.WORK``)."""
    if len(stl_bytes) < 84:
        raise ValueError("Invalid binary STL payload (< 84 bytes).")
    count = struct.unpack_from("<I", stl_bytes, 80)[0]
    if 84 + count * 50 != len(stl_bytes):
        raise ValueError("Expected binary STL payload matching declared facet count.")
    stl_dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    recs = np.frombuffer(stl_bytes, dtype=stl_dtype, count=count, offset=84).copy()
    if count > 0:
        shift = np.asarray(offset_mm, dtype="<f4").reshape(1, 1, 3)
        recs["vertices"] = recs["vertices"] + shift
    return stl_bytes[:84] + recs.tobytes()


def _compute_xy_shift_vs_nominal(bundle: MillingTaskBundle, plan: C.MotionPlan) -> Tuple[float, float]:
    """Compute ``(mean_xy_shift_mm, max_xy_shift_mm)`` of ``plan`` cutting waypoints vs. nominal."""
    if plan == bundle.nominal_motion:
        return 0.0, 0.0
    inv = bundle.task.work_to_machine.inverse().apply
    nom_segs = [
        (inv(s.start_mm), inv(s.end_mm))
        for s in bundle.nominal_motion.segments
        if s.kind != C.SegmentKind.RAPID
    ]
    if not nom_segs:
        return 0.0, 0.0
    a_xy = np.asarray([[p0[0], p0[1]] for p0, _ in nom_segs], dtype=np.float64)
    b_xy = np.asarray([[p1[0], p1[1]] for _, p1 in nom_segs], dtype=np.float64)
    d_xy = b_xy - a_xy
    l2 = np.sum(d_xy * d_xy, axis=1)

    dists: List[float] = []
    for seg in plan.segments:
        if seg.kind == C.SegmentKind.RAPID:
            continue
        for pt in (inv(seg.start_mm), inv(seg.end_mm)):
            px, py = float(pt[0]), float(pt[1])
            q = np.array([px, py], dtype=np.float64)
            diff = q[None, :] - a_xy
            proj = np.where(
                l2 > 1e-12,
                np.clip(np.sum(diff * d_xy, axis=1) / np.maximum(l2, 1e-12), 0.0, 1.0),
                0.0,
            )
            closest = a_xy + proj[:, None] * d_xy
            d_min = float(np.min(np.hypot(q[0] - closest[:, 0], q[1] - closest[:, 1])))
            dists.append(d_min)
    if not dists:
        return 0.0, 0.0
    return round(float(np.mean(dists)), 4), round(float(np.max(dists)), 4)


def serialize_plan_for_viewer(
    bundle: MillingTaskBundle,
    plan: C.MotionPlan,
    *,
    subdivide_for_cwe: bool = False,
) -> Dict[str, Any]:
    """Serialize ``plan`` segments, cumulative timing, CWE, and chipload telemetry for WebGL playback."""
    inv = bundle.task.work_to_machine.inverse().apply
    mean_xy_shift, max_xy_shift = _compute_xy_shift_vs_nominal(bundle, plan)

    cwe_plan = plan
    if subdivide_for_cwe:
        max_xy = max(
            (
                math.hypot(
                    float(inv(s.end_mm)[0] - inv(s.start_mm)[0]),
                    float(inv(s.end_mm)[1] - inv(s.start_mm)[1]),
                )
                for s in plan.segments
                if s.kind != C.SegmentKind.RAPID
            ),
            default=0.0,
        )
        if max_xy > 3.2:
            cwe_plan = subdivide_motion_plan(bundle.task, plan, max_cut_seg_mm=2.5)

    eng_report = analyze_plan_engagement(bundle, cwe_plan)
    eng_segs = eng_report["segments"]
    eng_summary = eng_report["summary"]

    t_cursor = 0.0
    cum_removed = 0.0
    segments_json: List[Dict[str, Any]] = []
    cut_feeds: List[float] = []
    cut_len = 0.0
    rapid_len = 0.0

    for seg, se in zip(cwe_plan.segments, eng_segs):
        sw = inv(seg.start_mm)
        ew = inv(seg.end_mm)
        length = math.dist(sw, ew)
        is_rapid = seg.kind == C.SegmentKind.RAPID
        feed_mm_s = float(bundle.task.limits.rapid_mm_s if is_rapid or seg.feed_mm_s is None else seg.feed_mm_s)
        dur = float(seg.duration_s)
        if is_rapid:
            rapid_len += length
        else:
            cut_len += length
            cut_feeds.append(feed_mm_s)
        rem_k = float(se["removed_mm3"])
        cum_removed += rem_k
        segments_json.append(
            {
                "id": seg.id,
                "kind": "rapid" if is_rapid else "cut",
                "class": _classify_rustcnc_move(seg, z_top_mm=float(bundle.case.stock_max_mm[2])),
                "start": [round(float(sw[0]), 4), round(float(sw[1]), 4), round(float(sw[2]), 4)],
                "end": [round(float(ew[0]), 4), round(float(ew[1]), 4), round(float(ew[2]), 4)],
                "length_mm": round(length, 4),
                "feed_mm_s": round(feed_mm_s, 4),
                "feed_mm_min": round(feed_mm_s * 60.0, 2),
                "t0_s": round(t_cursor, 5),
                "t1_s": round(t_cursor + dur, 5),
                "duration_s": round(dur, 5),
                "engaged": bool(se["engaged"]),
                "engaged_fraction": float(se["engaged_fraction"]),
                "engaged_time_s": float(se["engaged_time_s"]),
                "air_cut_time_s": float(se["air_cut_time_s"]),
                "removed_mm3": round(rem_k, 4),
                "cum_removed_mm3": round(cum_removed, 3),
                "mrr_mm3_s": float(se["mrr_mm3_s"]),
                "ap_mm": float(se["ap_mm"]),
                "ae_mm": float(se["ae_mm"]),
                "ae_pct": float(se["ae_pct"]),
                "phi_eng_deg": float(se["phi_eng_deg"]),
                "fz_mm": float(se["fz_mm"]),
                "hex_mm": float(se["hex_mm"]),
            }
        )
        t_cursor += dur

    return {
        "provenance": plan.provenance,
        "num_segments": len(segments_json),
        "total_duration_s": round(t_cursor, 4),
        "cutting_length_mm": round(cut_len, 3),
        "rapid_length_mm": round(rapid_len, 3),
        "min_cut_feed_mm_s": round(min(cut_feeds), 3) if cut_feeds else 0.0,
        "max_cut_feed_mm_s": round(max(cut_feeds), 3) if cut_feeds else 0.0,
        "mean_cut_feed_mm_s": round(sum(cut_feeds) / len(cut_feeds), 3) if cut_feeds else 0.0,
        "mean_xy_shift_mm": mean_xy_shift,
        "max_xy_shift_mm": max_xy_shift,
        "engagement_summary": eng_summary,
        "initial_pos": [
            round(float(inv(plan.initial_state.position_mm)[0]), 4),
            round(float(inv(plan.initial_state.position_mm)[1]), 4),
            round(float(inv(plan.initial_state.position_mm)[2]), 4),
        ],
        "segments": segments_json,
    }


__all__ = [
    "RUSTCNC_CONTAINER_FORMAT",
    "RUSTCNC_DESIGN_FORMAT",
    "RUSTCNC_CAM_FORMAT",
    "RUSTCNC_TOOL_LIB_FORMAT",
    "RUSTCNC_VIEW_FORMAT",
    "RUSTCNC_TOOLPATH_FORMAT",
    "RUSTCNC_CACHE_FORMAT",
    "safe_resolve_under",
    "narrow_uuid_u64",
    "reconstruct_motion_plan",
    "build_target_cad_document",
    "build_rustcnc_tool_library",
    "compute_rustcnc_setup_stamp",
    "compute_rustcnc_op_stamp_hex",
    "export_rustcnc_project",
    "export_camotics_project",
    "voxel_mask_to_binary_stl",
    "build_target_part_stl_bytes",
    "carve_resulting_stock_stl_bytes",
    "translate_binary_stl_bytes",
    "serialize_plan_for_viewer",
]
