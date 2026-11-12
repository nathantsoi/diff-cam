# SPDX-License-Identifier: MIT
"""Line-mapped 6-dialect multi-tool and multi-setup CNC G-code post-processor.

Compiles ordered sequences of :class:`MultiToolPostOpInput` operations across
multiple tools (``Tn M6``, ``G43 Hn``) and workholding setups (``G54`` Op1 ->
``M00`` flip -> ``G55`` Op2, with optional Renishaw WIPS / Inspection Plus /
RS274 spindle touch-probe WCS calibration ``T99``) into shop-ready NC programs
for all six controller families:
  - ``haas_ngc``  : Haas Next Generation Control (``.nc``, 5-digit ``O``, ``G187 P2 E0.05``)
  - ``rs274``     : LinuxCNC / NIST RS274NGC (``.ngc``, ``G91.1``, ``G64 P0.05``)
  - ``fanuc``     : FANUC 0i / 30i / 31i (``.nc``, 4-digit ``O``, ``G05.1 Q1 R5`` / ``Q0``, ``G28``)
  - ``robodrill`` : FANUC Robodrill Alpha 31i-B5 (``.nc``, 4-digit ``O``, ``G05.1 Q1 R5`` / ``Q0``, ``G53``)
  - ``makino``    : Makino Professional 5 / 6 Super GI (``.nc``, 4-digit ``O``, ``M252`` / ``M250``, ``G28``)
  - ``mazak``     : Mazak Mazatrol Smooth / Matrix EIA/ISO (``.eia``, no leading ``%``, ``G61.1 P2`` / ``G64``, ``G53``)

Every motion segment in ``op.motion_plan.segments`` is mapped to its exact
1-based G-code line number in :attr:`PostedMultiToolProgram.segment_to_line`
and :attr:`PostedMultiToolProgram.per_op_segment_lines` for synchronized 3D
WebGL playback and G-code scrubbers.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks import contracts as C
from cam.config import MachineConfig
from cam.dialects import (
    ControllerDialect,
    get_dialect_spec,
)
from cam.dialects.base import (
    arc_offset_words,
    axis_words,
    feed_out,
    fmt_int,
    fmt_real,
    len_out,
    posted_work_vector,
    sanitize_comment,
)

SUPPORTED_MULTI_DIALECTS: Tuple[str, ...] = (
    "haas_ngc",
    "rs274",
    "fanuc",
    "robodrill",
    "makino",
    "mazak",
)

_DIALECT_ALIASES: Dict[str, str] = {
    "haas_ngc": "haas_ngc",
    "haas": "haas_ngc",
    "ngc": "haas_ngc",
    "rs274": "rs274",
    "rs274ngc": "rs274",
    "linuxcnc": "rs274",
    "fanuc": "fanuc",
    "robodrill": "robodrill",
    "fanuc_robodrill": "robodrill",
    "makino": "makino",
    "mazak": "mazak",
    "mazak_eia": "mazak",
}

_TO_CONTROLLER_DIALECT: Dict[str, ControllerDialect] = {
    "haas_ngc": ControllerDialect.HAAS,
    "rs274": ControllerDialect.RS274,
    "fanuc": ControllerDialect.FANUC,
    "robodrill": ControllerDialect.ROBODRILL,
    "makino": ControllerDialect.MAKINO,
    "mazak": ControllerDialect.MAZAK,
}

MULTI_DIALECT_METADATA: Dict[str, Dict[str, Any]] = {
    "haas_ngc": {
        "id": "haas_ngc",
        "label": "Haas Next-Gen Control (NGC)",
        "file_extension": ".nc",
        "smoothing": "G187 P2 E0.05",
    },
    "rs274": {
        "id": "rs274",
        "label": "LinuxCNC / NIST RS274NGC",
        "file_extension": ".ngc",
        "smoothing": "G64 P0.05",
    },
    "fanuc": {
        "id": "fanuc",
        "label": "FANUC 0i / 30i / 31i (AICC II)",
        "file_extension": ".nc",
        "smoothing": "G05.1 Q1 R5",
    },
    "robodrill": {
        "id": "robodrill",
        "label": "FANUC Robodrill Alpha 31i-B5",
        "file_extension": ".nc",
        "smoothing": "G05.1 Q1 R5",
    },
    "makino": {
        "id": "makino",
        "label": "Makino Pro 5 / 6 (Super GI)",
        "file_extension": ".nc",
        "smoothing": "M252 / AICC G05.1 Q1",
    },
    "mazak": {
        "id": "mazak",
        "label": "Mazak Mazatrol Smooth EIA/ISO",
        "file_extension": ".eia",
        "smoothing": "G61.1 P2",
    },
}

_G54_TO_G59_RE = re.compile(r"^G(5[4-9])$")


def normalize_multi_dialect(dialect: str) -> str:
    """Normalize a dialect identifier into one of :data:`SUPPORTED_MULTI_DIALECTS`."""
    key = str(dialect).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _DIALECT_ALIASES:
        return _DIALECT_ALIASES[key]
    raise ValueError(
        f"Unsupported multi-tool post dialect {dialect!r}; expected one of {SUPPORTED_MULTI_DIALECTS}"
    )


def _normalize_output_units(unit_output: str) -> str:
    u = str(unit_output).strip().lower()
    if u in ("mm", "millimeters", "millimeter", "millimetres", "metric"):
        return "mm"
    if u in ("inch", "inches", "in", "us", "imperial", "sae"):
        return "inch"
    raise ValueError(f"Unsupported unit_output {unit_output!r}; expected 'mm' or 'inch'")


def _normalize_work_offset(work_offset: str, setup_index: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(work_offset).strip().upper())
    if not cleaned:
        return "G55" if int(setup_index) >= 2 else "G54"
    if re.match(r"^G(?:5[4-9]|54\.1\s+P[1-9][0-9]*)$", cleaned):
        return cleaned
    raise ValueError(f"Unsupported work_offset {work_offset!r}; expected G54..G59 or G54.1 Pn")


def _wcs_number_for_probe(work_offset: str) -> Tuple[int, int]:
    """Return ``(haas_w_number, iso_p_or_s_index)`` for ``work_offset`` (e.g. ``G54`` -> ``(54, 1)``)."""
    m = _G54_TO_G59_RE.match(work_offset)
    if m is not None:
        g_num = int(m.group(1))
        return g_num, g_num - 53
    return 54, 1


@dataclass(frozen=True)
class MultiToolPostOpInput:
    """Single operation input for multi-tool, multi-setup G-code post-processing."""

    op_index: int
    op_id: str
    op_kind: str
    setup_index: int  # 1 for G54 Op1, 2 for G55 Op2
    work_offset: str  # "G54", "G55", ...
    tool_number: int
    tool_label: str
    diameter_mm: float
    spindle_rpm: float
    base_feed_mm_s: float
    motion_plan: C.MotionPlan
    notes: str = ""


@dataclass(frozen=True)
class PostedMultiToolProgram:
    """Compiled multi-tool CNC shop program with exact 1-based segment-to-line mapping."""

    dialect: str
    program_name: str
    text: str
    lines: Tuple[str, ...]
    segment_to_line: Tuple[int, ...]
    per_op_segment_lines: Tuple[Tuple[int, ...], ...]
    op_header_lines: Dict[str, int]
    viewer_segment_to_line: Tuple[int, ...] = ()
    setup_flip_lines: Dict[int, int] = field(default_factory=dict)


def _home_z_retract_blocks(dialect_id: str) -> Tuple[str, ...]:
    if dialect_id in ("fanuc", "makino"):
        return ("G28 G91 Z0.0", "G90")
    return ("G90 G53 G0 Z0.0",)


def _smoothing_blocks(
    dialect_id: str,
    *,
    units: str,
    xyz_precision: int,
    blend_tolerance_mm: float = 0.05,
) -> Tuple[str, Optional[str]]:
    """Return ``(enable_block, cancel_block_or_None)`` for ``dialect_id``."""
    cfg_tmp = MachineConfig(units=units)
    tol_out = float(len_out(blend_tolerance_mm, cfg_tmp))
    tol_str = fmt_real(tol_out, xyz_precision, force_decimal=True)
    if dialect_id == "haas_ngc":
        return f"G187 P2 E{tol_str}", "G187"
    if dialect_id == "rs274":
        return f"G64 P{tol_str}", None
    if dialect_id in ("fanuc", "robodrill"):
        return "G05.1 Q1 R5", "G05.1 Q0"
    if dialect_id == "makino":
        return "M252 (SUPER GI SGI.4 - AICC G05.1 Q1 EQUIV)", "M250 (CANCEL SGI)"
    if dialect_id == "mazak":
        return "G61.1 P2", "G64"
    raise ValueError(f"Unhandled dialect {dialect_id!r}")


def _emit_probing_cycle(
    lines: List[str],
    *,
    dialect_id: str,
    work_offset: str,
    setup_index: int,
    stock_dims_mm: Tuple[float, float, float],
    cfg: MachineConfig,
    xyz_p: int,
    feed_p: int,
) -> None:
    """Append authentic controller-specific spindle probe WCS calibration blocks (``T99``)."""
    w_num, p_idx = _wcs_number_for_probe(work_offset)
    sx_out = fmt_real(float(len_out(max(1.0, float(stock_dims_mm[0])), cfg)), xyz_p, force_decimal=True)
    sy_out = fmt_real(float(len_out(max(1.0, float(stock_dims_mm[1])), cfg)), xyz_p, force_decimal=True)
    z_safe_out = fmt_real(float(len_out(25.0, cfg)), xyz_p, force_decimal=True)
    z_approach_out = fmt_real(float(len_out(10.0, cfg)), xyz_p, force_decimal=True)
    z_web_depth_out = fmt_real(float(len_out(-5.0, cfg)), xyz_p, force_decimal=True)
    f_probe = fmt_real(float(feed_out(1500.0, cfg)), feed_p, force_decimal=True)

    lines.append(
        f"(--- SETUP {setup_index} SPINDLE PROBE WCS DATUM {work_offset} ---)"
    )
    lines.append("T99 M6")
    lines.append(work_offset)
    lines.append("G90 G0 X0.0 Y0.0")
    lines.append(f"G43 H99 Z{z_safe_out}")

    if dialect_id == "haas_ngc":
        lines.extend(
            [
                "G65 P9832 (WIPS PROBE ON)",
                f"G65 P9810 Z{z_approach_out} F{f_probe} (PROTECTED POSITIONING)",
                f"G65 P9995 W{w_num}. A20. X{sx_out} Y{sy_out} Z{z_web_depth_out} (RECTANGULAR STOCK + Z DATUM)",
                "G65 P9833 (WIPS PROBE OFF)",
            ]
        )
    elif dialect_id == "rs274":
        z_plunge = fmt_real(float(len_out(-10.0, cfg)), xyz_p, force_decimal=True)
        f_slow = fmt_real(float(feed_out(150.0, cfg)), feed_p, force_decimal=True)
        lines.extend(
            [
                f"(RS274 G38.2 TOUCH PROBE CALIBRATION FOR {work_offset})",
                f"G0 Z{z_approach_out}",
                f"G38.2 Z{z_plunge} F{f_slow}",
                f"G10 L20 P{fmt_int(p_idx, name='P')} Z0.0",
                f"G0 Z{z_safe_out}",
            ]
        )
    else:
        # FANUC / Robodrill / Makino / Mazak Renishaw Inspection Plus / MMS macro convention
        sys_tag = "MMS" if dialect_id == "mazak" else "RENISHAW INSPECTION PLUS"
        lines.extend(
            [
                f"G65 P9832 ({sys_tag} PROBE ON)",
                f"G65 P9810 Z{z_approach_out} F{f_probe}",
                f"G65 P9811 Z0.0 S{fmt_int(p_idx, name='S')} (TOP Z DATUM -> {work_offset})",
                f"G65 P9812 X{sx_out} Z{z_web_depth_out} S{fmt_int(p_idx, name='S')} (X WEB CENTER)",
                f"G65 P9812 Y{sy_out} Z{z_web_depth_out} S{fmt_int(p_idx, name='S')} (Y WEB CENTER)",
                f"G65 P9833 ({sys_tag} PROBE OFF)",
            ]
        )

    lines.extend(_home_z_retract_blocks(dialect_id))


def format_multitool_program(
    ops: Sequence[MultiToolPostOpInput],
    *,
    dialect: str = "haas_ngc",
    program_number: int = 2001,
    title: str = "DiffCAM Part",
    unit_output: str = "mm",
    enable_probing: bool = False,
    stock_dims_mm: Tuple[float, float, float] = (100.0, 60.0, 20.0),
    work_to_machine: Optional[C.RigidTransform] = None,
    include_atc_pseudo_segments: bool = False,
) -> PostedMultiToolProgram:
    """Compile a multi-tool, multi-setup CNC G-code program with exact 1-based line mapping."""
    from cam.gcode_parser import _mapped_arc_orientation

    dialect_id = normalize_multi_dialect(dialect)
    ctrl_dialect = _TO_CONTROLLER_DIALECT[dialect_id]
    spec = get_dialect_spec(ctrl_dialect)
    units = _normalize_output_units(unit_output)
    cfg = MachineConfig(units=units)

    xyz_p = spec.shop_xyz_decimals(units)
    feed_p = spec.shop_feed_decimals(units)
    clean_title = sanitize_comment(title)[:48].upper()

    if isinstance(program_number, bool) or not isinstance(program_number, int) or program_number < 1:
        raise ValueError(f"program_number must be a positive integer; got {program_number!r}")

    if dialect_id == "haas_ngc":
        p_num = ((program_number - 1) % 99999) + 1
        o_word = f"O{p_num:05d}"
        prog_filename = f"{o_word}{spec.file_extension}"
    elif dialect_id == "rs274":
        p_num = ((program_number - 1) % 99999) + 1
        o_word = ""
        prog_filename = f"program_{p_num:04d}{spec.file_extension}"
    elif dialect_id == "mazak":
        p_num = ((program_number - 1) % 9999) + 1
        o_word = f"O{p_num:04d}"
        prog_filename = f"{o_word}{spec.file_extension}"
    else:
        p_num = ((program_number - 1) % 8999) + 1
        o_word = f"O{p_num:04d}"
        prog_filename = f"{o_word}{spec.file_extension}"

    lines: List[str] = []
    if dialect_id != "mazak":
        lines.append("%")

    banner_ctrl = sanitize_comment(MULTI_DIALECT_METADATA[dialect_id]["label"]).upper()
    if dialect_id == "rs274":
        lines.append(f"(DIFFCAM MIT {banner_ctrl} - {clean_title})")
    else:
        lines.append(f"{o_word} (DIFFCAM MIT {banner_ctrl} - {clean_title})")

    lines.append("(DIFFCAM AUTOMATED MULTI-TOOL MULTI-SETUP CAM - MIT CLEAN-ROOM)")

    # Emit shop tool-crib table comments
    seen_tools: Dict[int, Tuple[float, str]] = {}
    for op in ops:
        if op.tool_number not in seen_tools:
            seen_tools[int(op.tool_number)] = (float(op.diameter_mm), op.tool_label)
    for t_num, (d_mm, t_lbl) in seen_tools.items():
        d_str = fmt_real(float(len_out(d_mm, cfg)), xyz_p, force_decimal=True)
        lines.append(
            f"(T{fmt_int(t_num, name='T')} D={d_str} - {sanitize_comment(t_lbl).upper()})"
        )
    if enable_probing and 99 not in seen_tools:
        d_prb = fmt_real(float(len_out(4.0, cfg)), xyz_p, force_decimal=True)
        lines.append(f"(T99 D={d_prb} - RUBY SPINDLE TOUCH PROBE)")

    # Safe modal initialization block + unit selection
    if dialect_id == "rs274":
        lines.append("G90 G94 G17 G91.1 G40 G49 G80")
    else:
        lines.append("G90 G94 G17 G40 G49 G80")
    lines.append(cfg.units_code)
    lines.extend(_home_z_retract_blocks(dialect_id))

    smooth_on, smooth_off = _smoothing_blocks(
        dialect_id, units=units, xyz_precision=xyz_p, blend_tolerance_mm=0.05
    )

    inverse = work_to_machine.inverse() if work_to_machine is not None else None
    plane_codes = {C.ArcPlane.XY: "G17", C.ArcPlane.XZ: "G18", C.ArcPlane.YZ: "G19"}

    current_tool: Optional[int] = None
    current_wcs: Optional[str] = None
    current_rpm_int: Optional[int] = None
    prev_setup_index: Optional[int] = None
    prev_viewer_tool: Optional[int] = None
    smoothing_active = False
    coolant_active = False
    spindle_active = False
    output_plane = C.ArcPlane.XY

    op_header_lines: Dict[str, int] = {}
    setup_flip_lines: Dict[int, int] = {}
    per_op_segment_lines: List[Tuple[int, ...]] = []
    flat_segment_lines: List[int] = []
    viewer_segment_lines: List[int] = []

    def _shutdown_active_spindle_and_smoothing() -> None:
        nonlocal smoothing_active, coolant_active, spindle_active, output_plane
        if output_plane != C.ArcPlane.XY:
            lines.append("G17")
            output_plane = C.ArcPlane.XY
        if smoothing_active and smooth_off is not None:
            lines.append(smooth_off)
            smoothing_active = False
        if coolant_active:
            lines.append("M9")
            coolant_active = False
        if spindle_active:
            lines.append("M5")
            spindle_active = False
        lines.extend(_home_z_retract_blocks(dialect_id))

    if enable_probing and ops:
        first_wcs = _normalize_work_offset(ops[0].work_offset, ops[0].setup_index)
        _emit_probing_cycle(
            lines,
            dialect_id=dialect_id,
            work_offset=first_wcs,
            setup_index=int(ops[0].setup_index),
            stock_dims_mm=stock_dims_mm,
            cfg=cfg,
            xyz_p=xyz_p,
            feed_p=feed_p,
        )

    for op in ops:
        wcs = _normalize_work_offset(op.work_offset, op.setup_index)
        t_num = int(op.tool_number)
        rpm_int = max(0, int(round(float(op.spindle_rpm))))

        # Inter-setup part flip (e.g. Setup 1 G54 -> Setup 2 G55)
        if prev_setup_index is not None and int(op.setup_index) != prev_setup_index:
            _shutdown_active_spindle_and_smoothing()
            flip_comment = sanitize_comment(
                f"OP{int(op.setup_index)} FLIP PART IN {wcs} SOFT JAWS"
            )
            lines.append(f"M00 ({flip_comment})")
            flip_line_no = len(lines)
            setup_flip_lines[int(op.setup_index)] = flip_line_no
            if enable_probing:
                _emit_probing_cycle(
                    lines,
                    dialect_id=dialect_id,
                    work_offset=wcs,
                    setup_index=int(op.setup_index),
                    stock_dims_mm=stock_dims_mm,
                    cfg=cfg,
                    xyz_p=xyz_p,
                    feed_p=feed_p,
                )
            current_tool = None
            current_wcs = None
            current_rpm_int = None
        prev_setup_index = int(op.setup_index)

        header_comment = sanitize_comment(
            f"--- OP {op.op_index}: {op.op_id.upper()} | {wcs} | {op.tool_label.upper()} ---"
        )
        lines.append(f"({header_comment})")
        op_header_line = len(lines)
        op_header_lines[op.op_id] = op_header_line
        if op.notes:
            lines.append(f"({sanitize_comment(op.notes)})")

        init_pos_mm = op.motion_plan.initial_state.position_mm or (0.0, 0.0, 5.0)
        init_work = inverse.apply(init_pos_mm) if inverse is not None else init_pos_mm
        init_out = len_out(np.asarray(init_work, dtype=np.float64), cfg)
        z_init_str = fmt_real(float(init_out[2]), xyz_p, force_decimal=True)

        is_probe_op = op.op_kind == "wips_probe" or t_num == 99
        atc_action_line = op_header_line
        if current_tool != t_num or current_wcs != wcs:
            if current_tool is not None or spindle_active or coolant_active or smoothing_active:
                _shutdown_active_spindle_and_smoothing()
            lines.append(f"T{fmt_int(t_num, name='T')} M6")
            atc_action_line = len(lines)
            lines.append(wcs)
            if rpm_int > 0 and not is_probe_op:
                lines.append(f"S{fmt_int(rpm_int, name='S')} M3")
                spindle_active = True
            lines.append(f"G43 H{fmt_int(t_num, name='H')} Z{z_init_str}")
            if is_probe_op:
                if dialect_id != "rs274":
                    lines.append("G65 P9832 (SPINDLE PROBE ON)")
            else:
                lines.append("M8")
                coolant_active = True
                lines.append(smooth_on)
                smoothing_active = True
            current_tool = t_num
            current_wcs = wcs
            current_rpm_int = rpm_int
        else:
            if rpm_int > 0 and rpm_int != current_rpm_int and not is_probe_op:
                lines.append(f"S{fmt_int(rpm_int, name='S')} M3")
                spindle_active = True
                current_rpm_int = rpm_int
            if not smoothing_active and not is_probe_op:
                lines.append(smooth_on)
                smoothing_active = True

        # Track optional ATC tool-swap pseudo-segment line (matching viewer timeline segments)
        if prev_viewer_tool is not None and prev_viewer_tool != t_num:
            viewer_segment_lines.append(atc_action_line)
            if include_atc_pseudo_segments:
                flat_segment_lines.append(atc_action_line)
        prev_viewer_tool = t_num

        # Rapid to initial state XYZ
        lines.append(f"G0 {axis_words(init_out, xyz_p, force_decimal=True)}")

        op_seg_lines: List[int] = []
        last_linear_kind: Optional[C.SegmentKind] = C.SegmentKind.RAPID
        last_f_str: Optional[str] = None

        for seg in op.motion_plan.segments:
            is_home = seg.tag == "home"
            if seg.kind == C.SegmentKind.ARC and seg.arc is not None:
                arc = seg.arc
                if inverse is not None:
                    work_plane, work_dir = _mapped_arc_orientation(
                        inverse, arc.plane, arc.direction
                    )
                    start_w = inverse.apply(seg.start_mm)
                    end_w = inverse.apply(seg.end_mm)
                    centre_w = inverse.apply(arc.centre_mm)
                else:
                    work_plane, work_dir = arc.plane, arc.direction
                    start_w = seg.start_mm
                    end_w = seg.end_mm
                    centre_w = arc.centre_mm

                if work_plane != output_plane:
                    lines.append(plane_codes[work_plane])
                    output_plane = work_plane

                end_out = posted_work_vector(end_w, cfg, xyz_p)
                words = [axis_words(end_out, xyz_p, force_decimal=True)]
                words.extend(
                    arc_offset_words(
                        start_w,
                        end_w,
                        centre_w,
                        work_plane,
                        cfg,
                        xyz_p,
                        seg.id,
                        force_decimal=True,
                    )
                )
                if arc.turns:
                    words.append(f"P{fmt_int(arc.turns + 1, name='P')}")
                f_mm_min = float(seg.feed_mm_s or op.base_feed_mm_s) * 60.0
                f_str = fmt_real(float(feed_out(f_mm_min, cfg)), feed_p, force_decimal=True)
                words.append(f"F{f_str}")
                last_f_str = f_str
                last_linear_kind = None
                g_arc = "G2" if work_dir == C.ArcDirection.CW else "G3"
                lines.append(f"{g_arc} {' '.join(words)}")
            elif seg.kind == C.SegmentKind.RAPID:
                target_w = seg.end_mm if is_home else (
                    inverse.apply(seg.end_mm) if inverse is not None else seg.end_mm
                )
                target_out = len_out(np.asarray(target_w, dtype=np.float64), cfg)
                coords = axis_words(target_out, xyz_p, force_decimal=True)
                if is_home:
                    lines.append(f"G53 G0 {coords}")
                    last_linear_kind = None
                else:
                    prefix = "G0 " if last_linear_kind != C.SegmentKind.RAPID else ""
                    lines.append(f"{prefix}{coords}")
                    last_linear_kind = C.SegmentKind.RAPID
            else:
                target_w = seg.end_mm if is_home else (
                    inverse.apply(seg.end_mm) if inverse is not None else seg.end_mm
                )
                target_out = len_out(np.asarray(target_w, dtype=np.float64), cfg)
                coords = axis_words(target_out, xyz_p, force_decimal=True)
                f_mm_min = float(seg.feed_mm_s or op.base_feed_mm_s) * 60.0
                f_str = fmt_real(float(feed_out(f_mm_min, cfg)), feed_p, force_decimal=True)
                f_suffix = ""
                if last_f_str != f_str:
                    f_suffix = f" F{f_str}"
                    last_f_str = f_str
                if is_home:
                    lines.append(f"G53 G1 {coords}{f_suffix}")
                    last_linear_kind = None
                elif is_probe_op:
                    if dialect_id == "rs274":
                        lines.append(f"G38.2 {coords} F{f_str}")
                    else:
                        lines.append(f"G65 P9810 {coords} F{f_str}")
                    last_linear_kind = None
                else:
                    prefix = "G1 " if last_linear_kind != C.SegmentKind.FEED else ""
                    lines.append(f"{prefix}{coords}{f_suffix}")
                    last_linear_kind = C.SegmentKind.FEED

            seg_line_no = len(lines)
            op_seg_lines.append(seg_line_no)
            flat_segment_lines.append(seg_line_no)
            viewer_segment_lines.append(seg_line_no)

        if is_probe_op and dialect_id != "rs274":
            w_num, p_idx = _wcs_number_for_probe(wcs)
            if dialect_id == "haas_ngc":
                lines.append(f"G65 P9995 W{w_num}. A20. (UPDATE {wcs} WCS DATUM)")
            else:
                lines.append(f"G65 P9811 Z0.0 S{fmt_int(p_idx, name='S')} (UPDATE {wcs} WCS DATUM)")
            lines.append("G65 P9833 (SPINDLE PROBE OFF)")

        per_op_segment_lines.append(tuple(op_seg_lines))

    # Safe program epilogue
    _shutdown_active_spindle_and_smoothing()
    if dialect_id == "haas_ngc":
        lines.append("G90 G53 G0 X0.0 Y0.0")
    lines.append("M30")
    lines.append("%")

    lines_tuple = tuple(lines)
    nc_text = "\n".join(lines_tuple) + "\n"

    return PostedMultiToolProgram(
        dialect=dialect_id,
        program_name=prog_filename,
        text=nc_text,
        lines=lines_tuple,
        segment_to_line=tuple(flat_segment_lines),
        per_op_segment_lines=tuple(per_op_segment_lines),
        op_header_lines=op_header_lines,
        viewer_segment_to_line=tuple(viewer_segment_lines),
        setup_flip_lines=setup_flip_lines,
    )


def format_all_dialects_multitool(
    ops: Sequence[MultiToolPostOpInput],
    *,
    program_number: int = 2001,
    title: str = "DiffCAM Part",
    unit_output: str = "mm",
    enable_probing: bool = False,
    stock_dims_mm: Tuple[float, float, float] = (100.0, 60.0, 20.0),
    work_to_machine: Optional[C.RigidTransform] = None,
    include_atc_pseudo_segments: bool = False,
) -> Dict[str, PostedMultiToolProgram]:
    """Post ``ops`` across all six supported CNC controller dialects."""
    return {
        d: format_multitool_program(
            ops,
            dialect=d,
            program_number=program_number,
            title=title,
            unit_output=unit_output,
            enable_probing=enable_probing,
            stock_dims_mm=stock_dims_mm,
            work_to_machine=work_to_machine,
            include_atc_pseudo_segments=include_atc_pseudo_segments,
        )
        for d in SUPPORTED_MULTI_DIALECTS
    }


__all__ = [
    "MULTI_DIALECT_METADATA",
    "MultiToolPostOpInput",
    "PostedMultiToolProgram",
    "SUPPORTED_MULTI_DIALECTS",
    "format_all_dialects_multitool",
    "format_multitool_program",
    "normalize_multi_dialect",
]
