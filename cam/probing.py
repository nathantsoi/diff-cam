# SPDX-License-Identifier: MIT
"""Flexible CNC spindle-probing and tool-setting subsystem with Haas WIPS-R implementation.

Provides a pluggable, clean-room MIT driver architecture (:class:`ProbingDriver`
and :data:`PROBING_DRIVERS`) for:
  1. **Automatic Tool Setting** prior to cutting (e.g. **Renishaw OTS** optical
     table tool setter measuring length ``H`` and rotating cutter diameter ``D``
     via ``G65 P9995`` on Haas NGC or ``G65 P9857`` on ISO Inspection Plus).
  2. **First-Step & Post-Flip Workpiece WCS Probing** (e.g. **Renishaw OMP40-2**
     optical spindle probe executing as Operation 1 in ``G54`` and executing
     again immediately after an ``M00`` part flip in ``G55`` soft jaws).
  3. **Operator Setup Sheet Synthesis** (:func:`build_setup_sheet`) describing
     exact part-to-probe pre-positioning, Kurt DX6 jaw datum references, 5-point
     touch coordinates, and per-tool OTS calibration schedules for the webapp.
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmarks import contracts as C
from cam.config import MachineConfig
from cam.dialects.base import (
    feed_out,
    fmt_int,
    fmt_real,
    len_out,
    sanitize_comment,
)

INCH_MM: float = 25.4

ROTATING_MILL_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "flat end mill",
        "bull nose end mill",
        "ball end mill",
        "face mill",
        "slot mill",
        "lollipop mill",
        "dovetail mill",
        "tapered mill",
    }
)


@dataclass(frozen=True)
class SpindleProbeSpec:
    """Hardware & cycle geometry specification for a 3-axis spindle touch probe."""

    model_id: str = "renishaw_omp40_2"
    label: str = "Renishaw OMP40-2 Optical Spindle Probe"
    vendor_package: str = "Haas WIPS-R"
    transmission: str = "Optical Infrared (OMI-2H / OMI-2T)"
    tool_number: int = 99
    stylus_diameter_mm: float = 6.0
    stylus_length_mm: float = 50.0
    body_diameter_mm: float = 40.0
    body_length_mm: float = 50.0
    z_safe_mm: float = 8.0
    pre_position_z_mm: float = 10.0
    xy_jog_tolerance_mm: float = 5.0
    z_jog_tolerance_mm: float = 4.0
    radial_clearance_mm: float = 6.0
    side_touch_depth_mm: float = 3.5
    overtravel_mm: float = 10.0
    protected_feed_mm_min: float = 1500.0
    touch_feed_mm_s: float = 25.0


@dataclass(frozen=True)
class ToolSetterSpec:
    """Hardware & calibration policy for an automated table-mounted tool setter."""

    model_id: str = "renishaw_ots"
    label: str = "Renishaw OTS Optical Table Tool Setter"
    vendor_package: str = "Haas WIPS-R"
    transmission: str = "Optical Infrared (OMI-2H / OMI-2T)"
    stylus_disc_diameter_mm: float = 12.7
    measure_length: bool = True
    measure_rotating_diameter: bool = True
    check_breakage_after_cut: bool = False
    table_mount_zone: str = "Right table (+X) clear of Kurt DX6 vise"
    min_cutter_diameter_for_auto_dia_mm: float = 1.5
    max_cutter_diameter_for_auto_dia_mm: float = 65.0


@dataclass(frozen=True)
class ProbingConfig:
    """Top-level configuration selecting probing hardware and active cycle stages."""

    enabled: bool = True
    profile_id: str = "haas_wips_r"
    enable_spindle_wcs_probe: bool = True
    enable_tool_setter: bool = True
    probe_after_flip: bool = True
    spindle_probe: SpindleProbeSpec = field(default_factory=SpindleProbeSpec)
    tool_setter: ToolSetterSpec = field(default_factory=ToolSetterSpec)


@dataclass(frozen=True)
class ToolSettingInput:
    """Metadata for a single tool scheduled for automatic tool-setter calibration."""

    tool_number: int
    tool_label: str
    entity_type: str = "flat end mill"
    diameter_mm: float = 6.0
    stickout_mm: float = 30.0
    flutes: int = 3
    holder_label: str = ""


def classify_ots_tool_mode(
    tool: ToolSettingInput,
    config: ProbingConfig,
) -> Tuple[str, str]:
    """Return ``(ots_mode_id, human_label)`` for ``tool`` under ``config``."""
    etype = str(tool.entity_type or "").strip().lower()
    if etype == "probe" or int(tool.tool_number) == int(config.spindle_probe.tool_number):
        return (
            "spindle_probe_master",
            "Master Spindle Probe (H99 / Stylus Sphere Calibrated)",
        )
    if not config.enabled or not config.enable_tool_setter:
        return ("manual_offset", "Manual Tool Length / Diameter Offset")

    ts = config.tool_setter
    d_mm = float(tool.diameter_mm)
    if (
        ts.measure_rotating_diameter
        and etype in ROTATING_MILL_ENTITY_TYPES
        and ts.min_cutter_diameter_for_auto_dia_mm <= d_mm <= ts.max_cutter_diameter_for_auto_dia_mm
    ):
        return (
            "ots_length_and_diameter",
            "OTS Auto Length (H) + Rotating Diameter (D)",
        )
    if d_mm > ts.stylus_disc_diameter_mm and etype in ROTATING_MILL_ENTITY_TYPES:
        return (
            "ots_rotating_length",
            "OTS Off-Center Rotating Length (H)",
        )
    return (
        "ots_centerline_length",
        "OTS On-Center Tool Length (H)",
    )


def compute_safe_side_touch_z_mm(
    z_top_datum_mm: float,
    side_touch_depth_mm: float,
    jaw_top_z_mm: Optional[float] = None,
    stock_min_z_mm: Optional[float] = None,
) -> float:
    """Compute a side-web touch Z coordinate that stays clear of vise hard/soft jaws."""
    desired = float(z_top_datum_mm) - abs(float(side_touch_depth_mm))
    floor_limits: List[float] = []
    if jaw_top_z_mm is not None and math.isfinite(jaw_top_z_mm):
        floor_limits.append(float(jaw_top_z_mm) + 0.55)
    if stock_min_z_mm is not None and math.isfinite(stock_min_z_mm):
        floor_limits.append(float(stock_min_z_mm) + 0.50)
    if floor_limits:
        safe_floor = max(floor_limits)
        desired = max(desired, safe_floor)
    return round(min(float(z_top_datum_mm) - 0.60, desired), 4)


def build_wcs_probe_motion_plan(
    target_min_mm: Tuple[float, float, float],
    target_max_mm: Tuple[float, float, float],
    *,
    spec: Optional[SpindleProbeSpec] = None,
    stylus_radius_mm: Optional[float] = None,
    z_top_datum_mm: float = 0.0,
    jaw_top_z_mm: Optional[float] = None,
    rapid_mm_s: float = 100.0,
    provenance: str = "diffcam:wips_r_omp40_2",
) -> C.MotionPlan:
    """Synthesize a collision-checked 5-point (Top Z0 + 4 XY web) spindle probe MotionPlan."""
    sp = spec or SpindleProbeSpec()
    r_styl = float(stylus_radius_mm) if stylus_radius_mm is not None else 0.5 * float(sp.stylus_diameter_mm)
    r_styl = max(0.5, r_styl)
    z_safe = max(float(sp.z_safe_mm), float(z_top_datum_mm) + 5.0)
    z_side = compute_safe_side_touch_z_mm(
        z_top_datum_mm=z_top_datum_mm,
        side_touch_depth_mm=sp.side_touch_depth_mm,
        jaw_top_z_mm=jaw_top_z_mm,
        stock_min_z_mm=target_min_mm[2],
    )
    clear = r_styl + max(2.5, float(sp.radial_clearance_mm))
    touch = r_styl + 0.02
    feed_s = max(2.0, float(sp.touch_feed_mm_s))

    x0, y0, _ = target_min_mm
    x1, y1, _ = target_max_mm
    cx = 0.5 * (float(x0) + float(x1))
    cy = 0.5 * (float(y0) + float(y1))

    segs: List[C.Segment] = []
    cur = (cx, cy, z_safe)
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

    # 1. Protected approach & single-surface top Z0 datum touch at (cx, cy)
    _push((cx, cy, float(z_top_datum_mm) + 4.0), C.SegmentKind.RAPID, None)
    _push((cx, cy, float(z_top_datum_mm) + 0.02), C.SegmentKind.FEED, feed_s)
    _push((cx, cy, z_safe), C.SegmentKind.RAPID, None)

    # 2. Four-side (-X, +X, -Y, +Y) web centering touches safely above vise jaws
    for (ax, ay), (tx, ty) in (
        ((float(x0) - clear, cy), (float(x0) - touch, cy)),
        ((float(x1) + clear, cy), (float(x1) + touch, cy)),
        ((cx, float(y0) - clear), (cx, float(y0) - touch)),
        ((cx, float(y1) + clear), (cx, float(y1) + touch)),
    ):
        _push((ax, ay, z_safe), C.SegmentKind.RAPID, None)
        _push((ax, ay, z_side), C.SegmentKind.RAPID, None)
        _push((tx, ty, z_side), C.SegmentKind.FEED, feed_s)
        _push((ax, ay, z_side), C.SegmentKind.FEED, feed_s)
        _push((ax, ay, z_safe), C.SegmentKind.RAPID, None)

    _push((cx, cy, z_safe), C.SegmentKind.RAPID, None)
    durs = C.blended_segment_durations(segs, rapid_mm_s, 150.0)
    segs = [dataclasses.replace(s, duration_s=float(d)) for s, d in zip(segs, durs)]
    return C.MotionPlan(
        frame=C.Frame.MACHINE,
        timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
        initial_state=C.MachineState(position_mm=(cx, cy, z_safe), frame=C.Frame.MACHINE),
        segments=tuple(segs),
        provenance=provenance,
    )


def _wcs_codes(work_offset: str) -> Tuple[int, int]:
    """Return ``(haas_w_code, iso_s_index)`` for ``work_offset`` (e.g. ``G54`` -> ``(54, 1)``)."""
    cleaned = str(work_offset).strip().upper()
    if len(cleaned) == 3 and cleaned.startswith("G5") and cleaned[2].isdigit():
        g_num = int(cleaned[1:])
        if 54 <= g_num <= 59:
            return g_num, g_num - 53
    return 54, 1


class ProbingDriver(ABC):
    """Pluggable clean-room interface for machine probing and tool-setting packages."""

    profile_id: str
    package_label: str
    package_url: str

    def default_config(self) -> ProbingConfig:
        """Return default :class:`ProbingConfig` for this probing package."""
        return ProbingConfig(profile_id=self.profile_id)

    @abstractmethod
    def emit_tool_setting_preamble(
        self,
        tools: Sequence[ToolSettingInput],
        *,
        config: ProbingConfig,
        dialect_id: str,
        machine_cfg: MachineConfig,
        xyz_p: int,
        feed_p: int,
        home_z_blocks: Sequence[str],
    ) -> List[str]:
        """Emit automatic tool-setter calibration NC lines prior to cutting operations."""

    @abstractmethod
    def emit_wcs_probe_on_lines(
        self,
        *,
        config: ProbingConfig,
        dialect_id: str,
        work_offset: str,
        setup_index: int,
    ) -> List[str]:
        """Emit spindle-probe activation lines immediately after ``G43 H99``."""

    @abstractmethod
    def format_probe_Protected_move(
        self,
        coords_str: str,
        feed_str: str,
        *,
        config: ProbingConfig,
        dialect_id: str,
    ) -> str:
        """Format a single protected positioning / touch move block."""

    @abstractmethod
    def emit_wcs_probe_off_and_update_lines(
        self,
        *,
        config: ProbingConfig,
        dialect_id: str,
        work_offset: str,
        setup_index: int,
        stock_dims_mm: Tuple[float, float, float],
        side_touch_z_mm: float,
        machine_cfg: MachineConfig,
        xyz_p: int,
    ) -> List[str]:
        """Emit WCS offset update + spindle-probe deactivation lines at the end of a probe op."""

    def emit_standalone_wcs_probe_cycle(
        self,
        *,
        config: ProbingConfig,
        dialect_id: str,
        work_offset: str,
        setup_index: int,
        stock_dims_mm: Tuple[float, float, float],
        side_touch_z_mm: float,
        machine_cfg: MachineConfig,
        xyz_p: int,
        feed_p: int,
        home_z_blocks: Sequence[str],
    ) -> List[str]:
        """Emit a self-contained WCS spindle-probe block sequence (when not represented as an op)."""
        sp = config.spindle_probe
        t_prb = fmt_int(sp.tool_number, name="T")
        z_safe_out = fmt_real(float(len_out(max(20.0, sp.z_safe_mm), machine_cfg)), xyz_p, force_decimal=True)
        z_app_out = fmt_real(float(len_out(sp.pre_position_z_mm, machine_cfg)), xyz_p, force_decimal=True)
        f_prot = fmt_real(float(feed_out(sp.protected_feed_mm_min, machine_cfg)), feed_p, force_decimal=True)

        out: List[str] = [
            f"({sanitize_comment(f'--- SETUP {setup_index} {sp.model_id.upper()} WCS PROBE {work_offset} ---')})",
            f"T{t_prb} M6",
            work_offset,
            "G90 G0 X0.0 Y0.0",
            f"G43 H{t_prb} Z{z_safe_out}",
        ]
        out.extend(
            self.emit_wcs_probe_on_lines(
                config=config,
                dialect_id=dialect_id,
                work_offset=work_offset,
                setup_index=setup_index,
            )
        )
        out.append(
            self.format_probe_Protected_move(
                f"Z{z_app_out}",
                f_prot,
                config=config,
                dialect_id=dialect_id,
            )
        )
        out.extend(
            self.emit_wcs_probe_off_and_update_lines(
                config=config,
                dialect_id=dialect_id,
                work_offset=work_offset,
                setup_index=setup_index,
                stock_dims_mm=stock_dims_mm,
                side_touch_z_mm=side_touch_z_mm,
                machine_cfg=machine_cfg,
                xyz_p=xyz_p,
            )
        )
        out.extend(home_z_blocks)
        return out

    def format_ots_macro_preview(
        self,
        tool: ToolSettingInput,
        *,
        config: ProbingConfig,
        dialect_id: str = "haas_ngc",
    ) -> str:
        """Return a concise single-line preview of the tool-setter macro for the Setup Sheet."""
        mode_id, _ = classify_ots_tool_mode(tool, config)
        t_num = int(tool.tool_number)
        if mode_id == "spindle_probe_master":
            return f"T{t_num} Master OMP40-2 (G65 P9832 / P9811 / P9812)"
        if mode_id == "manual_offset":
            return f"Manual H{t_num} / D{t_num}"
        if dialect_id == "haas_ngc":
            b_code = "B3." if mode_id == "ots_length_and_diameter" else ("B2." if mode_id == "ots_rotating_length" else "B1.")
            return (
                f"G65 P9995 A0. {b_code} C2. T{t_num}. "
                f"E{tool.stickout_mm:.1f} D{tool.diameter_mm:.2f}"
            )
        if dialect_id == "rs274":
            return f"G38.2 OTS Touch -> G10 L1 P{t_num}"
        b_code = "B3." if mode_id == "ots_length_and_diameter" else "B1."
        return f"G65 P9857 {b_code} T{t_num}. D{tool.diameter_mm:.2f}"


class HaasWipsRDriver(ProbingDriver):
    """Haas WIPS-R package driver (Renishaw OMP40-2 Spindle Probe + OTS Tool Setter)."""

    profile_id = "haas_wips_r"
    package_label = "Haas WIPS-R (Renishaw OMP40-2 Spindle Probe + OTS Tool Setter)"
    package_url = "https://haascnc.com/productivity/probe-system/wips-r"

    def emit_tool_setting_preamble(
        self,
        tools: Sequence[ToolSettingInput],
        *,
        config: ProbingConfig,
        dialect_id: str,
        machine_cfg: MachineConfig,
        xyz_p: int,
        feed_p: int,
        home_z_blocks: Sequence[str],
    ) -> List[str]:
        if not config.enabled or not config.enable_tool_setter:
            return []
        cutting_tools = [
            t
            for t in tools
            if str(t.entity_type).strip().lower() != "probe"
            and int(t.tool_number) != int(config.spindle_probe.tool_number)
        ]
        if not cutting_tools:
            return []

        ts = config.tool_setter
        out: List[str] = [
            f"({sanitize_comment(f'--- STEP 0: {ts.vendor_package.upper()} OTS AUTOMATIC TOOL SETTING ---')})"
        ]
        for tool in cutting_tools:
            t_num = fmt_int(tool.tool_number, name="T")
            mode_id, mode_lbl = classify_ots_tool_mode(tool, config)
            d_str = fmt_real(float(len_out(max(0.2, tool.diameter_mm), machine_cfg)), xyz_p, force_decimal=True)
            e_str = fmt_real(float(len_out(max(5.0, tool.stickout_mm), machine_cfg)), xyz_p, force_decimal=True)
            out.append(f"T{t_num} M6 ({sanitize_comment(f'{tool.tool_label} - {mode_lbl}')})")
            if dialect_id == "haas_ngc":
                b_code = (
                    "B3."
                    if mode_id == "ots_length_and_diameter"
                    else ("B2." if mode_id == "ots_rotating_length" else "B1.")
                )
                out.append(
                    f"G65 P9995 A0. {b_code} C2. T{t_num}. E{e_str} D{d_str} "
                    f"({sanitize_comment(mode_lbl.upper())})"
                )
            elif dialect_id == "rs274":
                z_ots = fmt_real(float(len_out(-25.0, machine_cfg)), xyz_p, force_decimal=True)
                f_ots = fmt_real(float(feed_out(150.0, machine_cfg)), feed_p, force_decimal=True)
                r_str = fmt_real(float(len_out(0.5 * max(0.2, tool.diameter_mm), machine_cfg)), xyz_p, force_decimal=True)
                out.append(f"G38.2 Z{z_ots} F{f_ots} (OTS TOOL LENGTH TOUCH)")
                out.append(f"G10 L1 P{t_num} Z0.0 R{r_str}")
            else:
                out.append("G65 P9832 (OTS PROBE ON)")
                if mode_id == "ots_length_and_diameter":
                    out.append(
                        f"G65 P9857 B3. T{t_num}. D{d_str} ({sanitize_comment(mode_lbl.upper())})"
                    )
                else:
                    out.append(
                        f"G65 P9857 B1. T{t_num}. ({sanitize_comment(mode_lbl.upper())})"
                    )
                out.append("G65 P9833 (OTS PROBE OFF)")
            out.extend(home_z_blocks)
        return out

    def emit_wcs_probe_on_lines(
        self,
        *,
        config: ProbingConfig,
        dialect_id: str,
        work_offset: str,
        setup_index: int,
    ) -> List[str]:
        _ = config
        stage_tag = "POST-FLIP RE-PROBE" if setup_index >= 2 else "INITIAL WCS PROBE"
        if dialect_id == "rs274":
            return [f"({sanitize_comment(f'OMP40-2 {stage_tag} FOR {work_offset}')})"]
        if dialect_id == "haas_ngc":
            return [f"G65 P9832 ({sanitize_comment(f'WIPS-R OMP40-2 OPTICAL PROBE ON - {stage_tag}')})"]
        return [f"G65 P9832 ({sanitize_comment(f'RENISHAW OMP40-2 PROBE ON - {stage_tag}')})"]

    def format_probe_Protected_move(
        self,
        coords_str: str,
        feed_str: str,
        *,
        config: ProbingConfig,
        dialect_id: str,
    ) -> str:
        _ = config
        if dialect_id == "rs274":
            return f"G38.2 {coords_str} F{feed_str}"
        return f"G65 P9810 {coords_str} F{feed_str}"

    def emit_wcs_probe_off_and_update_lines(
        self,
        *,
        config: ProbingConfig,
        dialect_id: str,
        work_offset: str,
        setup_index: int,
        stock_dims_mm: Tuple[float, float, float],
        side_touch_z_mm: float,
        machine_cfg: MachineConfig,
        xyz_p: int,
    ) -> List[str]:
        sp = config.spindle_probe
        w_num, p_idx = _wcs_codes(work_offset)
        sx_out = fmt_real(float(len_out(max(1.0, float(stock_dims_mm[0])), machine_cfg)), xyz_p, force_decimal=True)
        sy_out = fmt_real(float(len_out(max(1.0, float(stock_dims_mm[1])), machine_cfg)), xyz_p, force_decimal=True)
        z_side_out = fmt_real(float(len_out(float(side_touch_z_mm), machine_cfg)), xyz_p, force_decimal=True)
        q_out = fmt_real(float(len_out(max(2.0, sp.overtravel_mm), machine_cfg)), xyz_p, force_decimal=True)
        r_out = fmt_real(float(len_out(max(2.0, sp.radial_clearance_mm), machine_cfg)), xyz_p, force_decimal=True)
        stage_note = "POST-FLIP " if setup_index >= 2 else ""

        if dialect_id == "rs274":
            return [
                f"G10 L20 P{fmt_int(p_idx, name='P')} Z0.0 "
                f"({sanitize_comment(f'OMP40-2 {stage_note}SET {work_offset} Z0 DATUM')})"
            ]
        if dialect_id == "haas_ngc":
            return [
                f"G65 P9811 Z0.0 Q{q_out} S{p_idx}. "
                f"({sanitize_comment(f'OMP40-2 {stage_note}SET {work_offset} TOP Z0 DATUM')})",
                f"G65 P9812 X{sx_out} Z{z_side_out} Q{q_out} R{r_out} S{p_idx}. "
                f"({sanitize_comment(f'OMP40-2 {stage_note}SET {work_offset} X WEB CENTER')})",
                f"G65 P9812 Y{sy_out} Z{z_side_out} Q{q_out} R{r_out} S{p_idx}. "
                f"({sanitize_comment(f'OMP40-2 {stage_note}SET {work_offset} Y WEB CENTER')})",
                f"G65 P9995 W{w_num}. A20. X{sx_out} Y{sy_out} Z{z_side_out} "
                f"({sanitize_comment(f'WIPS-R UPDATE {work_offset} WORK OFFSET')})",
                "G65 P9833 (WIPS-R OMP40-2 OPTICAL PROBE OFF)",
            ]
        return [
            f"G65 P9811 Z0.0 Q{q_out} S{p_idx}. "
            f"({sanitize_comment(f'OMP40-2 {stage_note}SET {work_offset} TOP Z0 DATUM')})",
            f"G65 P9812 X{sx_out} Z{z_side_out} Q{q_out} R{r_out} S{p_idx}. "
            f"({sanitize_comment(f'OMP40-2 {stage_note}SET {work_offset} X WEB CENTER')})",
            f"G65 P9812 Y{sy_out} Z{z_side_out} Q{q_out} R{r_out} S{p_idx}. "
            f"({sanitize_comment(f'OMP40-2 {stage_note}SET {work_offset} Y WEB CENTER')})",
            "G65 P9833 (OMP40-2 PROBE OFF)",
        ]


class RenishawInspectionPlusDriver(HaasWipsRDriver):
    """Generic ISO Renishaw Inspection Plus + OTS/TS27R probing driver."""

    profile_id = "renishaw_inspection_plus"
    package_label = "Renishaw Inspection Plus (OMP40-2 / OMP60 + OTS / TS27R)"
    package_url = "https://www.renishaw.com"

    def default_config(self) -> ProbingConfig:
        return ProbingConfig(
            profile_id=self.profile_id,
            spindle_probe=SpindleProbeSpec(vendor_package="Renishaw Inspection Plus"),
            tool_setter=ToolSetterSpec(vendor_package="Renishaw Inspection Plus"),
        )


class RS274TouchProbeDriver(HaasWipsRDriver):
    """Clean-room NIST RS274NGC / LinuxCNC ``G38.2`` + ``G10`` touch-probe driver."""

    profile_id = "rs274_g38"
    package_label = "RS274NGC G38.2 Spindle Probe + G10 L1 Tool Setter"
    package_url = "https://linuxcnc.org/docs/html/gcode/g-code.html#gcode:g38"

    def default_config(self) -> ProbingConfig:
        return ProbingConfig(
            profile_id=self.profile_id,
            spindle_probe=SpindleProbeSpec(
                model_id="rs274_touch_probe",
                label="3D Touch Trigger Spindle Probe (G38.2)",
                vendor_package="RS274NGC",
                transmission="Hardwired / Optical Skip",
            ),
            tool_setter=ToolSetterSpec(
                model_id="rs274_tool_setter",
                label="Table Tool Touch-Off Pad (G38.2 + G10 L1)",
                vendor_package="RS274NGC",
                transmission="Hardwired Skip",
                measure_rotating_diameter=False,
            ),
        )


PROBING_DRIVERS: Dict[str, ProbingDriver] = {
    HaasWipsRDriver.profile_id: HaasWipsRDriver(),
    RenishawInspectionPlusDriver.profile_id: RenishawInspectionPlusDriver(),
    RS274TouchProbeDriver.profile_id: RS274TouchProbeDriver(),
}

ALLOWED_PROBING_PROFILES: frozenset[str] = frozenset(PROBING_DRIVERS.keys())


def register_probing_driver(driver: ProbingDriver) -> None:
    """Register a custom :class:`ProbingDriver` implementation by ``driver.profile_id``."""
    key = str(driver.profile_id).strip().lower()
    if not key:
        raise ValueError("ProbingDriver.profile_id must be non-empty")
    PROBING_DRIVERS[key] = driver


def get_probing_driver(profile_id: str = "haas_wips_r") -> ProbingDriver:
    """Resolve a registered :class:`ProbingDriver` by ``profile_id``."""
    key = str(profile_id or "haas_wips_r").strip().lower()
    if key in PROBING_DRIVERS:
        return PROBING_DRIVERS[key]
    raise ValueError(
        f"Unknown probing profile {profile_id!r}; registered profiles: {sorted(PROBING_DRIVERS)}"
    )


def make_probing_config(
    *,
    enabled: bool = True,
    profile_id: str = "haas_wips_r",
    enable_spindle_wcs_probe: bool = True,
    enable_tool_setter: bool = True,
    probe_after_flip: bool = True,
) -> ProbingConfig:
    """Construct a validated :class:`ProbingConfig` for ``profile_id``."""
    drv = get_probing_driver(profile_id)
    base = drv.default_config()
    return dataclasses.replace(
        base,
        enabled=bool(enabled),
        enable_spindle_wcs_probe=bool(enable_spindle_wcs_probe),
        enable_tool_setter=bool(enable_tool_setter),
        probe_after_flip=bool(probe_after_flip),
    )


def build_setup_sheet(
    *,
    part_id: str,
    part_title: str,
    program_number: int,
    stock_material_id: str,
    primary_dialect: str,
    stock_min_mm: Tuple[float, float, float],
    stock_max_mm: Tuple[float, float, float],
    part_min_mm: Tuple[float, float, float],
    part_max_mm: Tuple[float, float, float],
    workholding: Dict[str, Any],
    tools_table: Dict[str, Dict[str, Any]],
    operations_table: Sequence[Dict[str, Any]],
    probing_config: Optional[ProbingConfig] = None,
) -> Dict[str, Any]:
    """Build a complete operator Setup Sheet & Per-Tool OTS schedule for the webapp."""
    cfg = probing_config if probing_config is not None else make_probing_config(enabled=True)
    drv = get_probing_driver(cfg.profile_id)
    sp = cfg.spindle_probe
    ts = cfg.tool_setter

    lx = round(float(stock_max_mm[0] - stock_min_mm[0]), 3)
    ly = round(float(stock_max_mm[1] - stock_min_mm[1]), 3)
    lz = round(float(stock_max_mm[2] - stock_min_mm[2]), 3)
    px = round(float(part_max_mm[0] - part_min_mm[0]), 3)
    py = round(float(part_max_mm[1] - part_min_mm[1]), 3)
    pz = round(float(part_max_mm[2] - part_min_mm[2]), 3)
    cx = round(0.5 * float(stock_min_mm[0] + stock_max_mm[0]), 3)
    cy = round(0.5 * float(stock_min_mm[1] + stock_max_mm[1]), 3)

    clamp_axis = str(workholding.get("clamping_axis") or "Y").upper()
    wh_setups = workholding.get("setups") or {}
    s1_wh = wh_setups.get("1") or {}
    s2_wh = wh_setups.get("2") or {}
    has_op2 = bool(
        workholding.get("has_op2_soft_jaws")
        or any(int(op.get("setup_index") or 1) >= 2 for op in operations_table)
    )

    jaw_top_s1 = float(s1_wh.get("jaw_top_z_mm", float(stock_min_mm[2]) + 2.0))
    z_side_s1 = compute_safe_side_touch_z_mm(0.0, sp.side_touch_depth_mm, jaw_top_s1, stock_min_mm[2])
    r_styl = 0.5 * float(sp.stylus_diameter_mm)

    fixed_jaw_desc = (
        "Fixed Rear Jaw (+Y / Stationary Vise Body) — set X workstop on left (-X) end of billet."
        if clamp_axis == "Y"
        else "Fixed Left Jaw (-X / Stationary Vise Body) — set Y workstop on front (-Y) end of billet."
    )

    setups_out: List[Dict[str, Any]] = [
        {
            "setup_index": 1,
            "work_offset": "G54",
            "stage_badge": "INITIAL PROBE · STEP 1",
            "title": "Setup 1 (G54) — Raw Billet Clamping & OMP40-2 Pre-Start Position",
            "vise_configuration": str(
                s1_wh.get("label")
                or f"Kurt DX6 Hardened Jaws + {float(s1_wh.get('parallel_height_mm', 42.45)):.2f} mm Parallels"
            ),
            "clamping_axis": clamp_axis,
            "jaw_opening_mm": round(float(s1_wh.get("jaw_opening_mm", ly if clamp_axis == "Y" else lx)), 2),
            "grip_z_mm": round(float(s1_wh.get("grip_z_mm", 2.0)), 2),
            "parallel_height_mm": round(float(s1_wh.get("parallel_height_mm", 42.45)), 2),
            "jaw_top_z_mm": round(jaw_top_s1, 3),
            "fixed_jaw_reference": fixed_jaw_desc,
            "wcs_datum_description": "Top-Center of Raw Stock Billet (X = 0.000, Y = 0.000, Z = 0.000 at unmachined top face)",
            "probe_start_xyz_mm": [cx, cy, round(float(sp.pre_position_z_mm), 2)],
            "probe_start_xyz_in": [
                round(cx / INCH_MM, 4),
                round(cy / INCH_MM, 4),
                round(float(sp.pre_position_z_mm) / INCH_MM, 4),
            ],
            "xy_jog_tolerance_mm": round(float(sp.xy_jog_tolerance_mm), 2),
            "z_jog_tolerance_mm": round(float(sp.z_jog_tolerance_mm), 2),
            "probed_span_xy_mm": [lx, ly],
            "side_touch_z_mm": round(z_side_s1, 3),
            "radial_clearance_mm": round(float(sp.radial_clearance_mm), 2),
            "touch_points_mm": [
                {"id": "Z_TOP", "label": "1. Top Face Z0 Datum", "xyz_mm": [cx, cy, 0.0]},
                {"id": "X_NEG", "label": "2. -X Billet Wall", "xyz_mm": [round(float(stock_min_mm[0]) - r_styl, 3), cy, round(z_side_s1, 3)]},
                {"id": "X_POS", "label": "3. +X Billet Wall", "xyz_mm": [round(float(stock_max_mm[0]) + r_styl, 3), cy, round(z_side_s1, 3)]},
                {"id": "Y_NEG", "label": "4. -Y Billet Wall", "xyz_mm": [cx, round(float(stock_min_mm[1]) - r_styl, 3), round(z_side_s1, 3)]},
                {"id": "Y_POS", "label": "5. +Y Billet Wall", "xyz_mm": [cx, round(float(stock_max_mm[1]) + r_styl, 3), round(z_side_s1, 3)]},
            ],
            "checklist": [
                f"Verify Renishaw OMP40-2 (T{sp.tool_number}, Ø{sp.stylus_diameter_mm:g} mm ruby ball) in spindle/ATC and Renishaw OTS clear on right (+X) table.",
                f"Seat {lx:.1f} × {ly:.1f} × {lz:.1f} mm billet on {float(s1_wh.get('parallel_height_mm', 42.45)):.2f} mm parallels in Kurt DX6 ({float(s1_wh.get('grip_z_mm', 2.0)):.1f} mm grip bite); tap down flush.",
                f"Handle-jog T{sp.tool_number} OMP40-2 ruby tip over billet top-center within ±{sp.xy_jog_tolerance_mm:g} mm XY and Z = +{sp.pre_position_z_mm:g} mm (+{sp.pre_position_z_mm / INCH_MM:.2f} in) above top surface; set rough initial G54 XYZ.",
                "Press Cycle Start: OTS automatically sets length (H) & rotating diameter (D) for all cutting tools, then OMP40-2 probes exact G54 Z0 and XY web center before Op1 cutting.",
            ],
        }
    ]

    if has_op2:
        jaw_top_s2 = float(s2_wh.get("jaw_top_z_mm", -3.2))
        z_side_s2 = compute_safe_side_touch_z_mm(0.0, sp.side_touch_depth_mm, jaw_top_s2, stock_min_mm[2])
        setups_out.append(
            {
                "setup_index": 2,
                "work_offset": "G55",
                "stage_badge": "POST-FLIP RE-PROBE · AFTER M00",
                "title": "Setup 2 (G55) — 180° Part Flip in Soft Jaws & OMP40-2 Re-Probing",
                "vise_configuration": str(
                    s2_wh.get("label") or "Kurt DX6 Machined 6061-T6 Contoured Soft Jaws"
                ),
                "clamping_axis": clamp_axis,
                "jaw_opening_mm": round(float(s2_wh.get("jaw_opening_mm", py if clamp_axis == "Y" else px)), 2),
                "grip_z_mm": round(float(s2_wh.get("grip_z_mm", 4.0)), 2),
                "seat_z_mm": round(float(s2_wh.get("seat_z_mm", -6.0)), 2),
                "jaw_top_z_mm": round(jaw_top_s2, 3),
                "fixed_jaw_reference": "Contoured G55 6061-T6 Soft Jaws — flip part 180° around X-axis (Y -> -Y, Z -> -Z) so carrier hat faces +Z.",
                "wcs_datum_description": "Top-Center of Flipped Carrier Hat / Part in G55 Soft Jaws (X = 0.000, Y = 0.000, Z = 0.000 at flipped top)",
                "probe_start_xyz_mm": [cx, cy, round(float(sp.pre_position_z_mm), 2)],
                "probe_start_xyz_in": [
                    round(cx / INCH_MM, 4),
                    round(cy / INCH_MM, 4),
                    round(float(sp.pre_position_z_mm) / INCH_MM, 4),
                ],
                "xy_jog_tolerance_mm": round(float(sp.xy_jog_tolerance_mm), 2),
                "z_jog_tolerance_mm": round(float(sp.z_jog_tolerance_mm), 2),
                "probed_span_xy_mm": [lx, ly],
                "side_touch_z_mm": round(z_side_s2, 3),
                "radial_clearance_mm": round(float(sp.radial_clearance_mm), 2),
                "touch_points_mm": [
                    {"id": "Z_TOP", "label": "1. Flipped Top Z0 Datum", "xyz_mm": [cx, cy, 0.0]},
                    {"id": "X_NEG", "label": "2. -X Flipped Hat Wall", "xyz_mm": [round(float(stock_min_mm[0]) - r_styl, 3), cy, round(z_side_s2, 3)]},
                    {"id": "X_POS", "label": "3. +X Flipped Hat Wall", "xyz_mm": [round(float(stock_max_mm[0]) + r_styl, 3), cy, round(z_side_s2, 3)]},
                    {"id": "Y_NEG", "label": "4. -Y Flipped Hat Wall", "xyz_mm": [cx, round(float(stock_min_mm[1]) - r_styl, 3), round(z_side_s2, 3)]},
                    {"id": "Y_POS", "label": "5. +Y Flipped Hat Wall", "xyz_mm": [cx, round(float(stock_max_mm[1]) + r_styl, 3), round(z_side_s2, 3)]},
                ],
                "checklist": [
                    "At M00 program stop, open Kurt DX6 jaws, clear chips with air blast, and install/clean G55 contoured 6061 soft jaws.",
                    "Flip Op1-machined part 180° about the X-axis (unmachined Op1 grip hat now faces +Z) and torque into G55 soft-jaw pocket.",
                    f"Ensure G55 rough seed is within ±{sp.xy_jog_tolerance_mm:g} mm of flipped center (Z = 0 at flipped hat top, side touches clear soft jaws at Z = {z_side_s2:.2f} mm > jaw top {jaw_top_s2:.2f} mm).",
                    f"Press Cycle Start: T{sp.tool_number} OMP40-2 automatically re-probes exact G55 Z0 and XY center before facing off the carrier hat and finishing Op2 features.",
                ],
            }
        )

    # Assemble per-tool schedule (OMP40-2 T99 first when probing is enabled, then T1..Tn in numeric order)
    ops_by_tnum: Dict[int, List[str]] = {}
    for op in operations_table:
        tn = int(op.get("tool_number") or 1)
        wcs_s = str(op.get("work_offset") or ("G55" if int(op.get("setup_index") or 1) >= 2 else "G54"))
        ops_by_tnum.setdefault(tn, []).append(f"Op{op.get('op_index', '?')} ({wcs_s})")

    raw_tool_entries: List[Tuple[int, Dict[str, Any]]] = []
    for k, tdict in tools_table.items():
        try:
            tn = int(tdict.get("tool_number") or k)
        except (TypeError, ValueError):
            tn = 1
        raw_tool_entries.append((tn, tdict))

    raw_tool_entries.sort(key=lambda pair: (0 if pair[0] == sp.tool_number else 1, pair[0]))

    tools_schedule: List[Dict[str, Any]] = []
    for tn, tdict in raw_tool_entries:
        d_mm = float(tdict.get("diameter_mm", 2.0 * float(tdict.get("cutter_radius_mm", 3.0))))
        stick_mm = float(tdict.get("stickout_mm", tdict.get("holder_bottom_mm", 30.0)))
        flute_len_mm = float(tdict.get("flute_length_mm", 0.5 * stick_mm))
        flutes = int(tdict.get("flutes", 1 if str(tdict.get("entity_type")) == "probe" else 3))
        etype = str(tdict.get("entity_type") or ("probe" if tn == sp.tool_number else "flat end mill"))
        lbl = str(tdict.get("nominal_label") or tdict.get("label") or f"Tool T{tn}")
        hld_lbl = str(tdict.get("holder_description") or tdict.get("holder_id") or "BT30 / CAT40 Collet")

        ts_in = ToolSettingInput(
            tool_number=tn,
            tool_label=lbl,
            entity_type=etype,
            diameter_mm=d_mm,
            stickout_mm=stick_mm,
            flutes=flutes,
            holder_label=hld_lbl,
        )
        ots_mode_id, ots_mode_lbl = classify_ots_tool_mode(ts_in, cfg)
        ots_macro = drv.format_ots_macro_preview(ts_in, config=cfg, dialect_id=primary_dialect)

        tools_schedule.append(
            {
                "tool_number": tn,
                "offset_registers": f"H{tn}" + (f" / D{tn}" if ots_mode_id == "ots_length_and_diameter" else ""),
                "tool_id": str(tdict.get("tool_id") or f"t{tn}"),
                "label": lbl,
                "entity_type": etype,
                "diameter_mm": round(d_mm, 3),
                "diameter_in": round(d_mm / INCH_MM, 4),
                "flutes": flutes,
                "flute_length_mm": round(flute_len_mm, 2),
                "stickout_mm": round(stick_mm, 2),
                "stickout_in": round(stick_mm / INCH_MM, 3),
                "holder_label": hld_lbl,
                "ots_mode": ots_mode_id,
                "ots_mode_label": ots_mode_lbl,
                "ots_macro_preview": ots_macro,
                "used_in_ops": ops_by_tnum.get(tn, []),
            }
        )

    return {
        "schema": "diffcam-setup-sheet/1",
        "part_id": part_id,
        "part_title": part_title,
        "program_number": int(program_number),
        "stock_material_id": stock_material_id,
        "primary_dialect": primary_dialect,
        "probing_enabled": bool(cfg.enabled),
        "probe_after_flip": bool(cfg.enabled and cfg.probe_after_flip and has_op2),
        "ots_enabled": bool(cfg.enabled and cfg.enable_tool_setter),
        "probe_package": {
            "profile_id": drv.profile_id,
            "package_label": drv.package_label,
            "package_url": drv.package_url,
            "spindle_probe": asdict(sp),
            "tool_setter": asdict(ts),
        },
        "stock_dims_mm": [lx, ly, lz],
        "stock_min_mm": [round(float(v), 3) for v in stock_min_mm],
        "stock_max_mm": [round(float(v), 3) for v in stock_max_mm],
        "part_dims_mm": [px, py, pz],
        "part_min_mm": [round(float(v), 3) for v in part_min_mm],
        "part_max_mm": [round(float(v), 3) for v in part_max_mm],
        "has_op2_flip": has_op2,
        "setups": setups_out,
        "tools_schedule": tools_schedule,
    }


__all__ = [
    "ALLOWED_PROBING_PROFILES",
    "HaasWipsRDriver",
    "PROBING_DRIVERS",
    "ProbingConfig",
    "ProbingDriver",
    "RS274TouchProbeDriver",
    "RenishawInspectionPlusDriver",
    "SpindleProbeSpec",
    "ToolSetterSpec",
    "ToolSettingInput",
    "build_setup_sheet",
    "build_wcs_probe_motion_plan",
    "classify_ots_tool_mode",
    "compute_safe_side_touch_z_mm",
    "get_probing_driver",
    "make_probing_config",
    "register_probing_driver",
]
