# SPDX-License-Identifier: MIT
"""Clean-room CNC controller dialect specifications for Makino and Mazak.

Implements two industrial EIA/ISO post-processor and syntax-verification specs:
- :class:`MakinoDialectSpec` -- Makino Professional 5 / 6 with Super Geometric
  Intelligence (SGI.4 ``M252`` / ``M250`` smoothing, ``%`` framing, 4-digit
  ``O0001``..``O8999``, and ``G28 G91 Z0.0`` + ``G90`` machine Z retracts).
- :class:`MazakDialectSpec` -- Mazak Mazatrol Smooth / Matrix EIA/ISO (``.eia``
  extension, **no** leading ``%`` + trailing ``%``, ``O0001``..``O9999`` first
  line, ``G61.1 P2`` + ``G64`` cancel smoothing, and ``G90 G53 G0 Z0.0``
  machine Z retracts).
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np

from benchmarks.contracts import Frame, MotionPlan, TaskSpec, TimingRule
from cam.config import MachineConfig
from cam.dialects.base import (
    ControllerDialect,
    ControllerDialectSpec,
    HardwareSyntaxError,
    axis_words,
    build_motion_lines,
    check_real_word_decimals,
    feed_out,
    fmt_int,
    fmt_real,
    format_program_number,
    len_out,
    sanitize_comment,
    strip_nc_comments,
    tool_table_comment,
    xy_words,
)

_TOKEN_RE = re.compile(r"([A-Za-z])\s*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))")
_WORK_OFFSET_RE = re.compile(r"^G(?:5[4-9]|54\.1\s+P[1-9][0-9]*)$")
_G54_TO_G59_RE = re.compile(r"^G(5[4-9])$")
_PROG_NUMBER_RE = re.compile(r"^O([0-9]{4})$")
_MAZAK_FIRST_LINE_RE = re.compile(r"^O([0-9]{4})\b")

_INT_LETTERS = frozenset({"O", "T", "H", "S", "M", "N", "Q", "L"})


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    """Validate a ``(T, 3)`` finite trajectory array in normalized stock coordinates."""
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must be (T, 3); got {arr.shape}")
    if len(arr) == 0:
        raise ValueError("positions is empty")
    if not np.isfinite(arr).all():
        raise ValueError("positions must contain only finite values")
    return arr


def _validate_work_offset(work_offset: str) -> str:
    """Validate and normalize a modal work-offset code (``G54``..``G59`` or ``G54.1 Pn``)."""
    normalized = re.sub(r"\s+", " ", str(work_offset).strip().upper())
    if not _WORK_OFFSET_RE.match(normalized):
        raise ValueError(
            f"Unsupported work_offset {work_offset!r}; expected G54..G59 or G54.1 Pn"
        )
    return normalized


def _fixture_offset_p_register(work_offset: str) -> int:
    """Map ``G54``..``G59`` to ``G10 L2 P1``..``P6``."""
    match = _G54_TO_G59_RE.match(_validate_work_offset(work_offset))
    if match is None:
        raise ValueError(
            f"G10 L2 work-offset programming requires G54..G59; got {work_offset!r}"
        )
    return int(match.group(1)) - 53


def _validate_plan_for_shop(plan: MotionPlan, blend_tolerance_mm: float) -> None:
    """Validate structural preconditions shared by ``format_shop_program``."""
    from cam.posts import UnsupportedPostError

    if not math.isfinite(blend_tolerance_mm) or blend_tolerance_mm <= 0.0:
        raise ValueError(f"blend_tolerance_mm must be finite and > 0; got {blend_tolerance_mm!r}")
    if plan.frame != Frame.MACHINE:
        raise UnsupportedPostError("shop posting requires a machine-frame MotionPlan")
    if not plan.initial_state.known or plan.initial_state.position_mm is None:
        raise UnsupportedPostError("cannot emit shop program from an unknown initial state")
    if plan.timing_rule not in (
        TimingRule.EXACT_STOP_CONSTANT_SPEED,
        TimingRule.BLENDED_ACCEL_BOUNDED,
    ):
        raise UnsupportedPostError(
            "NC posting preserves exact_stop_constant_speed or blended_accel_bounded, not fixed_dt"
        )


def _validate_plan_timing_for_typed(plan: MotionPlan) -> None:
    """Validate timing rule supported by ``format_typed_program``."""
    from cam.posts import UnsupportedPostError

    if plan.timing_rule not in (
        TimingRule.EXACT_STOP_CONSTANT_SPEED,
        TimingRule.BLENDED_ACCEL_BOUNDED,
    ):
        raise UnsupportedPostError(
            "NC posting preserves exact_stop_constant_speed or blended_accel_bounded, not fixed_dt"
        )


def _check_comment_delimiters(raw_line: str, line_no: int) -> None:
    """Reject nested, unmatched, or unterminated parenthesized comments."""
    depth = 0
    for ch in raw_line:
        if ch == "(":
            if depth > 0:
                raise HardwareSyntaxError(
                    f"Line {line_no}: nested parenthesized comments are invalid in EIA/ISO blocks"
                )
            depth = 1
        elif ch == ")":
            if depth == 0:
                raise HardwareSyntaxError(
                    f"Line {line_no}: unmatched closing ')' comment delimiter"
                )
            depth = 0
        elif ch == ";" and depth == 0:
            break
    if depth != 0:
        raise HardwareSyntaxError(f"Line {line_no}: unterminated '(' comment")


def _tokenize_nc_block(raw_line: str, line_no: int) -> list[tuple[str, str]]:
    """Tokenize executable EIA/ISO words on ``raw_line``, rejecting malformed text."""
    _check_comment_delimiters(raw_line, line_no)
    code = strip_nc_comments(raw_line)
    if not code or code == "%":
        return []

    tokens: list[tuple[str, str]] = []
    offset = 0
    while offset < len(code):
        while offset < len(code) and code[offset].isspace():
            offset += 1
        if offset >= len(code):
            break
        match = _TOKEN_RE.match(code, offset)
        if match is None:
            raise HardwareSyntaxError(
                f"Line {line_no}: invalid EIA/ISO token near {code[offset:]!r}"
            )
        tokens.append((match.group(1).upper(), match.group(2)))
        offset = match.end()
    return tokens


def _validate_syntax_units(units: str) -> None:
    if units not in ("mm", "inch"):
        raise HardwareSyntaxError(
            f"Unsupported units {units!r} for syntax check; expected 'mm' or 'inch'"
        )


class MakinoDialectSpec(ControllerDialectSpec):
    """Makino Professional 5 / 6 controller dialect with Super GI (SGI.4)."""

    dialect = ControllerDialect.MAKINO
    display_name = "Makino Professional 5 / 6 (Super GI)"
    file_extension = ".nc"
    force_decimal = True
    program_digits = 4
    max_program_number = 8999
    leading_percent = True
    trailing_percent = True
    shop_xyz_decimals_mm = 3
    shop_xyz_decimals_inch = 4
    shop_feed_decimals_mm = 1
    shop_feed_decimals_inch = 2

    def format_typed_program(
        self,
        motion_lines: Sequence[str],
        *,
        plan: MotionPlan,
        task: TaskSpec,
        config: MachineConfig,
    ) -> str:
        """Wrap ``motion_lines`` in Makino Pro 5/6 prologue, SGI smoothing, and epilogue."""
        del task  # validated upstream; signature fixed by ControllerDialectSpec
        _validate_plan_timing_for_typed(plan)
        prog_num = format_program_number(
            config.program_number, digits=4, max_number=self.max_program_number
        )
        prog_label = sanitize_comment(config.program_name)
        units_code = config.units_code
        work_offset = _validate_work_offset(config.work_offset)
        blended = plan.timing_rule == TimingRule.BLENDED_ACCEL_BOUNDED
        offset_line = (
            f"G43 H{fmt_int(config.h_register, name='H')}"
            if plan.initial_state.length_offset_applied
            else "G49"
        )

        lines: list[str] = [
            "%",
            f"{prog_num} ({prog_label})",
            "(Generated by diff-cam typed MotionPlan Makino Pro 5/6 post)",
            "G90 G94 G17 G49 G40 G80",
            units_code,
            work_offset if blended else f"{work_offset} G61",
            f"T{fmt_int(config.tool_number, name='T')} M06",
            offset_line,
            f"S{fmt_int(config.spindle_rpm, name='S')} M03",
        ]
        if config.coolant:
            lines.append("M08")
        if blended:
            lines.append("M252")
        lines.extend(motion_lines)
        if blended:
            lines.append("M250")
        if config.coolant:
            lines.append("M09")
        lines.extend(("M05", "M30", "%"))
        return "\n".join(lines) + "\n"

    def format_shop_program(
        self,
        plan: MotionPlan,
        task: TaskSpec,
        config: MachineConfig,
        *,
        case_label: str | None = None,
        blend_tolerance_mm: float = 0.05,
    ) -> str:
        """Emit a standalone Makino Pro 5/6 shop-floor program."""
        _validate_plan_for_shop(plan, blend_tolerance_mm)
        prog_num = format_program_number(
            config.program_number, digits=4, max_number=self.max_program_number
        )
        label = sanitize_comment(case_label or config.program_name)
        units_code = config.units_code
        work_offset = _validate_work_offset(config.work_offset)
        xyz_p = self.shop_xyz_decimals(config.units)
        feed_p = self.shop_feed_decimals(config.units)

        assert plan.initial_state.position_mm is not None
        init_work = task.work_to_machine.inverse().apply(plan.initial_state.position_mm)
        init_out = len_out(np.asarray(init_work, dtype=np.float64), config)
        blended = plan.timing_rule == TimingRule.BLENDED_ACCEL_BOUNDED
        offset_line = (
            f"G43 H{fmt_int(config.h_register, name='H')}"
            if plan.initial_state.length_offset_applied
            else "G49"
        )
        motion_lines = build_motion_lines(
            plan,
            task,
            config,
            xyz_precision=xyz_p,
            feed_precision=feed_p,
            force_decimal=True,
        )

        lines: list[str] = [
            "%",
            f"{prog_num} (DIFFCAM MIT MAKINO PRO - {label})",
            tool_table_comment(task, config),
            "G90 G94 G17 G49 G40 G80",
            units_code,
            f"T{fmt_int(config.tool_number, name='T')} M06",
            f"S{fmt_int(config.spindle_rpm, name='S')} M03",
            work_offset,
            offset_line,
            "G28 G91 Z0.0",
            "G90",
            f"G0 {xy_words(float(init_out[0]), float(init_out[1]), xyz_p, force_decimal=True)}",
            f"G0 Z{fmt_real(float(init_out[2]), xyz_p, force_decimal=True)}",
        ]
        if config.coolant:
            lines.append("M08")
        lines.append("M252" if blended else "G61")
        lines.extend(motion_lines)
        if blended:
            lines.append("M250")
        if config.coolant:
            lines.append("M09")
        lines.extend(("M05", "G28 G91 Z0.0", "G90", "M30", "%"))
        return "\n".join(lines) + "\n"

    def format_legacy_positions_program(
        self,
        positions: np.ndarray,
        config: MachineConfig = MachineConfig(),
    ) -> str:
        """Emit a legacy ``(T, 3)`` trajectory program for Makino Professional 5/6."""
        pts = _validate_positions(positions)
        pts_out = len_out(config.to_wcs(pts), config)
        f_cut = feed_out(config.feed, config)
        f_plunge = feed_out(config.plunge_feed, config)
        units_code = config.units_code
        work_offset = _validate_work_offset(config.work_offset)
        p = config.precision
        safe = fmt_real(float(len_out(config.safe_z_mm, config)), p, force_decimal=True)
        prog_num = format_program_number(
            config.program_number, digits=4, max_number=self.max_program_number
        )
        prog_label = sanitize_comment(config.program_name)

        lines: list[str] = [
            "%",
            f"{prog_num} ({prog_label})",
            "(Post: Makino Professional 5/6 Super GI -- generated by diff-cam)",
            "G90 G94 G17 G49 G40 G80",
            units_code,
            "G28 G91 Z0.0",
            "G90",
            f"T{fmt_int(config.tool_number, name='T')} M06",
            f"S{fmt_int(config.spindle_rpm, name='S')} M03",
            work_offset,
        ]
        if config.stock_origin_in is not None:
            ox, oy, oz = len_out(config.stock_origin_vec, config)
            p_reg = _fixture_offset_p_register(work_offset)
            lines.append(
                f"G10 L2 P{p_reg} {axis_words((ox, oy, oz), p, force_decimal=True)}"
            )

        x0, y0, z0 = pts_out[0]
        lines.append(f"G00 {xy_words(float(x0), float(y0), p, force_decimal=True)}")
        lines.append(f"G43 H{fmt_int(config.h_register, name='H')} Z{safe}")
        if config.coolant:
            lines.append("M08")

        lines.append(
            f"G01 Z{fmt_real(float(z0), p, force_decimal=True)} "
            f"F{fmt_real(float(f_plunge), p, force_decimal=True)}"
        )
        if len(pts_out) > 1:
            lines.append(f"F{fmt_real(float(f_cut), p, force_decimal=True)}")
            for pt in pts_out[1:]:
                lines.append(f"G01 {axis_words(pt, p, force_decimal=True)}")

        lines.append(f"G00 Z{safe}")
        if config.coolant:
            lines.append("M09")
        lines.extend(("M05", "G28 G91 Z0.0", "G90", "M30", "%"))
        return "\n".join(lines) + "\n"

    def verify_hardware_syntax(
        self,
        nc_text: str,
        *,
        units: str = "mm",
        shop_mode: bool = False,
    ) -> None:
        """Verify Makino Pro 5/6 EIA/ISO framing, SGI pairing, and decimal rules."""
        _validate_syntax_units(units)
        raw_lines = nc_text.splitlines()
        non_empty = [(idx, raw.strip()) for idx, raw in enumerate(raw_lines, start=1) if raw.strip()]
        if not non_empty:
            raise HardwareSyntaxError("Makino NC program is empty")

        if non_empty[0][1] != "%":
            raise HardwareSyntaxError(
                f"Line {non_empty[0][0]}: Makino NC program must begin with leading '%'"
            )
        if len(non_empty) < 3 or non_empty[-1][1] != "%":
            raise HardwareSyntaxError(
                "Makino NC program must end with trailing '%' delimiter"
            )
        for line_no, stripped in non_empty[1:-1]:
            if stripped == "%" or strip_nc_comments(stripped) == "%":
                raise HardwareSyntaxError(
                    f"Line {line_no}: unexpected '%' inside Makino NC program body"
                )

        # Verify exact 4-digit O0001..O8999 on the program header line after '%'.
        header_line_no, header_raw = non_empty[1]
        _check_comment_delimiters(header_raw, header_line_no)
        header_code = strip_nc_comments(header_raw)
        prog_match = _PROG_NUMBER_RE.match(header_code)
        if prog_match is None:
            raise HardwareSyntaxError(
                f"Line {header_line_no}: Makino program header must be an exact 4-digit "
                f"O0001..O8999 number after '%', got {header_code!r}"
            )
        prog_val = int(prog_match.group(1))
        if not (1 <= prog_val <= self.max_program_number):
            raise HardwareSyntaxError(
                f"Line {header_line_no}: Makino program number O{prog_match.group(1)} "
                f"must be in O0001..O{self.max_program_number:04d}"
            )

        check_real_word_decimals(
            nc_text,
            int_letters=_INT_LETTERS,
            max_xyz_decimals=self.shop_xyz_decimals(units) if shop_mode else None,
            max_feed_decimals=self.shop_feed_decimals(units) if shop_mode else None,
        )

        sgi_active = False
        saw_m30 = False
        for line_no, raw in enumerate(raw_lines, start=1):
            tokens = _tokenize_nc_block(raw, line_no)
            if not tokens:
                continue

            block_g: list[float] = []
            block_m: list[int] = []
            for ltr, num_str in tokens:
                if ltr == "O" and line_no != header_line_no:
                    raise HardwareSyntaxError(
                        f"Line {line_no}: duplicate O-number O{num_str} in Makino program body"
                    )
                if ltr == "G":
                    g_val = float(num_str)
                    block_g.append(g_val)
                    if abs(g_val - 64.0) <= 1e-9:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: G64 is forbidden on Makino Pro 5/6 (use SGI M252/M250)"
                        )
                    if abs(g_val - 187.0) <= 1e-9:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: Haas G187 is forbidden on Makino Pro 5/6"
                        )
                    if abs(g_val - 5.1) <= 1e-9 or abs(g_val - 5.0) <= 1e-9:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: FANUC G05/G05.1 is forbidden on Makino Pro 5/6 (use SGI M252/M250)"
                        )
                    if abs(g_val - 61.1) <= 1e-9:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: Mazak G61.1 is forbidden on Makino Pro 5/6 (use SGI M252/M250)"
                        )
                    if abs(g_val - 91.1) <= 1e-9:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: RS274 G91.1 is forbidden on Makino Pro 5/6"
                        )
                elif ltr == "M":
                    block_m.append(int(num_str))

            has_sgi_on = any(m in (251, 252, 253) for m in block_m)
            has_sgi_off = 250 in block_m
            if has_sgi_on and has_sgi_off:
                raise HardwareSyntaxError(
                    f"Line {line_no}: conflicting SGI activation and M250 cancel on same block"
                )
            if has_sgi_on and any(abs(g - 61.0) <= 1e-9 for g in block_g):
                raise HardwareSyntaxError(
                    f"Line {line_no}: conflicting G61 exact-stop and Makino SGI mode on same block"
                )

            for m_code in block_m:
                if m_code in (251, 252, 253):
                    if saw_m30:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: Makino SGI M{m_code} cannot appear after M30"
                        )
                    sgi_active = True
                elif m_code == 250:
                    if saw_m30:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: Makino SGI cancel M250 must appear before M30"
                        )
                    if not sgi_active:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: M250 (SGI cancel) without active M251/M252/M253"
                        )
                    sgi_active = False
                elif m_code == 30:
                    if sgi_active:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: active Makino SGI mode (M251/M252/M253) must be "
                            "cancelled with M250 before M30"
                        )
                    saw_m30 = True

        if not saw_m30:
            raise HardwareSyntaxError("Makino NC program is missing M30 program end")
        if sgi_active:
            raise HardwareSyntaxError(
                "Makino SGI mode (M251/M252/M253) was not cancelled with M250 before program end"
            )


class MazakDialectSpec(ControllerDialectSpec):
    """Mazak Mazatrol Smooth / Matrix (EIA/ISO) controller dialect."""

    dialect = ControllerDialect.MAZAK
    display_name = "Mazak Mazatrol Smooth / Matrix (EIA/ISO)"
    file_extension = ".eia"
    force_decimal = True
    program_digits = 4
    max_program_number = 9999
    leading_percent = False
    trailing_percent = True
    shop_xyz_decimals_mm = 3
    shop_xyz_decimals_inch = 4
    shop_feed_decimals_mm = 2
    shop_feed_decimals_inch = 3

    def format_typed_program(
        self,
        motion_lines: Sequence[str],
        *,
        plan: MotionPlan,
        task: TaskSpec,
        config: MachineConfig,
    ) -> str:
        """Wrap ``motion_lines`` in Mazak EIA/ISO prologue, G61.1/G64 smoothing, and epilogue."""
        del task  # validated upstream; signature fixed by ControllerDialectSpec
        _validate_plan_timing_for_typed(plan)
        prog_num = format_program_number(
            config.program_number, digits=4, max_number=self.max_program_number
        )
        prog_label = sanitize_comment(config.program_name)
        units_code = config.units_code
        work_offset = _validate_work_offset(config.work_offset)
        blended = plan.timing_rule == TimingRule.BLENDED_ACCEL_BOUNDED
        offset_line = (
            f"G43 H{fmt_int(config.h_register, name='H')}"
            if plan.initial_state.length_offset_applied
            else "G49"
        )

        # CRITICAL: Mazak EIA/ISO files MUST NOT begin with leading '%'.
        lines: list[str] = [
            f"{prog_num} ({prog_label})",
            "(Generated by diff-cam typed MotionPlan Mazak EIA/ISO post)",
            "G90 G94 G17 G49 G40 G80",
            units_code,
            work_offset if blended else f"{work_offset} G61",
            f"T{fmt_int(config.tool_number, name='T')} M06",
            offset_line,
            f"S{fmt_int(config.spindle_rpm, name='S')} M03",
        ]
        if config.coolant:
            lines.append("M08")
        if blended:
            lines.append("G61.1 P2")
        lines.extend(motion_lines)
        if blended:
            lines.append("G64")
        if config.coolant:
            lines.append("M09")
        lines.extend(("M05", "M30", "%"))
        return "\n".join(lines) + "\n"

    def format_shop_program(
        self,
        plan: MotionPlan,
        task: TaskSpec,
        config: MachineConfig,
        *,
        case_label: str | None = None,
        blend_tolerance_mm: float = 0.05,
    ) -> str:
        """Emit a standalone Mazak Mazatrol Smooth / Matrix EIA/ISO shop-floor program."""
        _validate_plan_for_shop(plan, blend_tolerance_mm)
        prog_num = format_program_number(
            config.program_number, digits=4, max_number=self.max_program_number
        )
        label = sanitize_comment(case_label or config.program_name)
        units_code = config.units_code
        work_offset = _validate_work_offset(config.work_offset)
        xyz_p = self.shop_xyz_decimals(config.units)
        feed_p = self.shop_feed_decimals(config.units)

        assert plan.initial_state.position_mm is not None
        init_work = task.work_to_machine.inverse().apply(plan.initial_state.position_mm)
        init_out = len_out(np.asarray(init_work, dtype=np.float64), config)
        blended = plan.timing_rule == TimingRule.BLENDED_ACCEL_BOUNDED
        offset_line = (
            f"G43 H{fmt_int(config.h_register, name='H')}"
            if plan.initial_state.length_offset_applied
            else "G49"
        )
        motion_lines = build_motion_lines(
            plan,
            task,
            config,
            xyz_precision=xyz_p,
            feed_precision=feed_p,
            force_decimal=True,
        )

        # CRITICAL: No leading '%' on Mazak EIA/ISO; ends with M30 followed by '%'.
        lines: list[str] = [
            f"{prog_num} (DIFFCAM MIT MAZAK EIA - {label})",
            tool_table_comment(task, config),
            "G90 G94 G17 G49 G40 G80",
            units_code,
            f"T{fmt_int(config.tool_number, name='T')} M06",
            f"S{fmt_int(config.spindle_rpm, name='S')} M03",
            work_offset,
            offset_line,
            "G90 G53 G0 Z0.0",
            f"G0 {xy_words(float(init_out[0]), float(init_out[1]), xyz_p, force_decimal=True)}",
            f"G0 Z{fmt_real(float(init_out[2]), xyz_p, force_decimal=True)}",
        ]
        if config.coolant:
            lines.append("M08")
        lines.append("G61.1 P2" if blended else "G61")
        lines.extend(motion_lines)
        if blended:
            lines.append("G64")
        if config.coolant:
            lines.append("M09")
        lines.extend(("M05", "G90 G53 G0 Z0.0", "M30", "%"))
        return "\n".join(lines) + "\n"

    def format_legacy_positions_program(
        self,
        positions: np.ndarray,
        config: MachineConfig = MachineConfig(),
    ) -> str:
        """Emit a legacy ``(T, 3)`` trajectory program for Mazak EIA/ISO."""
        pts = _validate_positions(positions)
        pts_out = len_out(config.to_wcs(pts), config)
        f_cut = feed_out(config.feed, config)
        f_plunge = feed_out(config.plunge_feed, config)
        units_code = config.units_code
        work_offset = _validate_work_offset(config.work_offset)
        p = config.precision
        safe = fmt_real(float(len_out(config.safe_z_mm, config)), p, force_decimal=True)
        prog_num = format_program_number(
            config.program_number, digits=4, max_number=self.max_program_number
        )
        prog_label = sanitize_comment(config.program_name)

        # CRITICAL: Starts directly with O0001 (...) without leading '%'.
        lines: list[str] = [
            f"{prog_num} ({prog_label})",
            "(Post: Mazak Mazatrol Smooth / Matrix EIA/ISO -- generated by diff-cam)",
            "G90 G94 G17 G49 G40 G80",
            units_code,
            "G90 G53 G00 Z0.0",
            f"T{fmt_int(config.tool_number, name='T')} M06",
            f"S{fmt_int(config.spindle_rpm, name='S')} M03",
            work_offset,
        ]
        if config.stock_origin_in is not None:
            ox, oy, oz = len_out(config.stock_origin_vec, config)
            p_reg = _fixture_offset_p_register(work_offset)
            lines.append(
                f"G10 L2 P{p_reg} {axis_words((ox, oy, oz), p, force_decimal=True)}"
            )

        x0, y0, z0 = pts_out[0]
        lines.append(f"G00 {xy_words(float(x0), float(y0), p, force_decimal=True)}")
        lines.append(f"G43 H{fmt_int(config.h_register, name='H')} Z{safe}")
        if config.coolant:
            lines.append("M08")

        lines.append(
            f"G01 Z{fmt_real(float(z0), p, force_decimal=True)} "
            f"F{fmt_real(float(f_plunge), p, force_decimal=True)}"
        )
        if len(pts_out) > 1:
            lines.append(f"F{fmt_real(float(f_cut), p, force_decimal=True)}")
            for pt in pts_out[1:]:
                lines.append(f"G01 {axis_words(pt, p, force_decimal=True)}")

        lines.append(f"G00 Z{safe}")
        if config.coolant:
            lines.append("M09")
        lines.extend(("M05", "G90 G53 G00 Z0.0", "M30", "%"))
        return "\n".join(lines) + "\n"

    def verify_hardware_syntax(
        self,
        nc_text: str,
        *,
        units: str = "mm",
        shop_mode: bool = False,
    ) -> None:
        """Verify Mazak EIA/ISO framing (no leading '%'), G61.1/G64 pairing, and decimals."""
        _validate_syntax_units(units)
        raw_lines = nc_text.splitlines()
        non_empty = [(idx, raw.strip()) for idx, raw in enumerate(raw_lines, start=1) if raw.strip()]
        if not non_empty:
            raise HardwareSyntaxError("Mazak EIA/ISO program is empty")

        first_line_no, first_raw = non_empty[0]
        if first_raw.startswith("%"):
            raise HardwareSyntaxError(
                f"Line {first_line_no}: Mazak EIA/ISO programs must NOT begin with leading '%'"
            )
        _check_comment_delimiters(first_raw, first_line_no)
        first_match = _MAZAK_FIRST_LINE_RE.match(first_raw)
        first_code = strip_nc_comments(first_raw)
        if first_match is None or _PROG_NUMBER_RE.match(first_code) is None:
            raise HardwareSyntaxError(
                f"Line {first_line_no}: Mazak EIA/ISO first non-empty line must be an exact "
                f"4-digit O0001..O9999 header, got {first_raw!r}"
            )
        prog_val = int(first_match.group(1))
        if not (1 <= prog_val <= self.max_program_number):
            raise HardwareSyntaxError(
                f"Line {first_line_no}: Mazak program number O{first_match.group(1)} "
                f"must be in O0001..O{self.max_program_number:04d}"
            )

        if len(non_empty) < 2 or non_empty[-1][1] != "%":
            raise HardwareSyntaxError(
                "Mazak EIA/ISO program must end with trailing '%' delimiter"
            )
        for line_no, stripped in non_empty[:-1]:
            if stripped == "%" or strip_nc_comments(stripped) == "%":
                raise HardwareSyntaxError(
                    f"Line {line_no}: unexpected '%' before end of Mazak EIA/ISO program"
                )

        check_real_word_decimals(
            nc_text,
            int_letters=_INT_LETTERS,
            max_xyz_decimals=self.shop_xyz_decimals(units) if shop_mode else None,
            max_feed_decimals=self.shop_feed_decimals(units) if shop_mode else None,
        )

        g61_1_active = False
        saw_m30 = False
        for line_no, raw in enumerate(raw_lines, start=1):
            tokens = _tokenize_nc_block(raw, line_no)
            if not tokens:
                continue

            block_g: list[float] = []
            block_m: list[int] = []
            block_p: list[str] = []
            for ltr, num_str in tokens:
                if ltr == "O" and line_no != first_line_no:
                    raise HardwareSyntaxError(
                        f"Line {line_no}: duplicate O-number O{num_str} in Mazak program body"
                    )
                if ltr == "G":
                    g_val = float(num_str)
                    block_g.append(g_val)
                    if abs(g_val - 187.0) <= 1e-9:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: Haas G187 is forbidden on Mazak EIA/ISO"
                        )
                    if abs(g_val - 5.1) <= 1e-9 or abs(g_val - 5.0) <= 1e-9:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: FANUC G05/G05.1 is forbidden on Mazak EIA/ISO "
                            "(use G61.1 P2 / G64)"
                        )
                    if abs(g_val - 91.1) <= 1e-9:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: RS274 G91.1 is forbidden on Mazak EIA/ISO"
                        )
                    if shop_mode and abs(g_val - 28.0) <= 1e-9:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: G28 is forbidden in Mazak shop mode "
                            "(use G90 G53 G0 Z0.0)"
                        )
                elif ltr == "M":
                    m_val = int(num_str)
                    block_m.append(m_val)
                    if m_val in (250, 251, 252, 253):
                        raise HardwareSyntaxError(
                            f"Line {line_no}: Makino SGI M{m_val} is forbidden on Mazak EIA/ISO"
                        )
                elif ltr == "P":
                    block_p.append(num_str)

            has_g61_1 = any(abs(g - 61.1) <= 1e-9 for g in block_g)
            has_g64 = any(abs(g - 64.0) <= 1e-9 for g in block_g)
            has_g61 = any(abs(g - 61.0) <= 1e-9 for g in block_g)

            if (has_g61_1 and has_g64) or (has_g61_1 and has_g61) or (has_g64 and has_g61):
                raise HardwareSyntaxError(
                    f"Line {line_no}: conflicting cutting/smoothing modes on same Mazak block"
                )

            if has_g61_1:
                if saw_m30:
                    raise HardwareSyntaxError(
                        f"Line {line_no}: Mazak G61.1 cannot appear after M30"
                    )
                for p_str in block_p:
                    if "." in p_str or not (0 <= int(p_str) <= 7):
                        raise HardwareSyntaxError(
                            f"Line {line_no}: Mazak G61.1 P level must be an integer in 0..7, "
                            f"got P{p_str}"
                        )
                g61_1_active = True

            if has_g61 and g61_1_active:
                raise HardwareSyntaxError(
                    f"Line {line_no}: Mazak G61.1 must be cancelled with G64, not G61"
                )

            if has_g64:
                if saw_m30:
                    raise HardwareSyntaxError(
                        f"Line {line_no}: Mazak G64 cancel must appear before M30"
                    )
                if not g61_1_active:
                    raise HardwareSyntaxError(
                        f"Line {line_no}: Mazak G64 is only valid to cancel an active G61.1 mode"
                    )
                g61_1_active = False

            for m_code in block_m:
                if m_code == 30:
                    if g61_1_active:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: active Mazak G61.1 smoothing mode must be "
                            "cancelled with G64 before M30"
                        )
                    saw_m30 = True

        if not saw_m30:
            raise HardwareSyntaxError("Mazak EIA/ISO program is missing M30 program end")
        if g61_1_active:
            raise HardwareSyntaxError(
                "Mazak G61.1 smoothing mode was not cancelled with G64 before program end"
            )


MAKINO_SPEC = MakinoDialectSpec()
MAZAK_SPEC = MazakDialectSpec()

__all__ = [
    "MakinoDialectSpec",
    "MazakDialectSpec",
    "MAKINO_SPEC",
    "MAZAK_SPEC",
]
