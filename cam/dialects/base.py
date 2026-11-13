# SPDX-License-Identifier: MIT
"""Clean-room controller dialect base classes and EIA/ISO formatting helpers.

Provides shared, dependency-light primitives for emitting and syntax-checking
typed and shop-floor CNC G-code programs across six controller families:
- ``rs274``     (LinuxCNC / NIST RS274NGC)
- ``haas``      (Haas Next Generation Control / NGC)
- ``fanuc``     (FANUC 0i / 30i / 31i & Compact)
- ``robodrill`` (FANUC Robodrill Alpha 31i-B5)
- ``makino``    (Makino Professional 5 / 6 Super GI)
- ``mazak``     (Mazak Mazatrol Smooth / Matrix EIA/ISO)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import math
import re
from typing import Sequence

import numpy as np

from benchmarks.contracts import (
    ARC_RADIUS_TOL_MM,
    ArcDirection,
    ArcPlane,
    Dialect,
    Frame,
    MotionPlan,
    SegmentKind,
    TaskSpec,
    TimingRule,
)
from cam.config import MachineConfig
from cam.units import mm_per_min_to_ipm, mm_to_inch


class ControllerDialect(str, Enum):
    """Supported CNC post-processor and round-trip parser controller dialects."""

    RS274 = "rs274"
    HAAS = "haas"
    FANUC = "fanuc"
    ROBODRILL = "robodrill"
    MAKINO = "makino"
    MAZAK = "mazak"


INDUSTRIAL_EIA_DIALECTS: frozenset[ControllerDialect] = frozenset(
    {
        ControllerDialect.HAAS,
        ControllerDialect.FANUC,
        ControllerDialect.ROBODRILL,
        ControllerDialect.MAKINO,
        ControllerDialect.MAZAK,
    }
)


class HardwareSyntaxError(ValueError):
    """Raised when emitted G-code violates a physical controller's EIA/ISO rules."""


def coerce_controller_dialect(value: Dialect | ControllerDialect | str) -> ControllerDialect:
    """Normalize a ``contracts.Dialect``, ``ControllerDialect``, or string name."""
    if isinstance(value, ControllerDialect):
        return value
    if isinstance(value, Dialect):
        return ControllerDialect(value.value)
    if isinstance(value, str):
        try:
            return ControllerDialect(value.strip().lower())
        except ValueError:
            pass
    raise ValueError(
        f"Unknown controller dialect {value!r}; expected one of {[d.value for d in ControllerDialect]}"
    )


def is_dialect_allowed_by_task(task: TaskSpec, dialect: ControllerDialect) -> bool:
    """Check whether ``task.motion_semantics.dialects`` permits ``dialect``.

    ``TaskSpec.motion_semantics.dialects`` defaults to ``(Dialect.RS274, Dialect.HAAS)``
    in the frozen benchmark contract. When ``Dialect.HAAS`` is enabled on a task,
    all industrial EIA/ISO dialects (``haas``, ``fanuc``, ``robodrill``, ``makino``,
    ``mazak``) are permitted; tasks restricted to ``(Dialect.RS274,)`` reject them.
    """
    if dialect == ControllerDialect.RS274:
        return Dialect.RS274 in task.motion_semantics.dialects
    return Dialect.HAAS in task.motion_semantics.dialects


def len_out(value_mm, config: MachineConfig):
    """Convert mm scalar or array to ``config.units`` (``mm`` or ``inch``)."""
    return mm_to_inch(value_mm) if config.units == "inch" else value_mm


def feed_out(feed_mm_per_min: float, config: MachineConfig) -> float:
    """Convert feedrate from ``mm/min`` to ``config.units`` (``mm/min`` or ``in/min``)."""
    return mm_per_min_to_ipm(feed_mm_per_min) if config.units == "inch" else feed_mm_per_min


def fmt_real(value: float, precision: int, *, force_decimal: bool = True) -> str:
    """Format a real G-code coordinate/feed word.

    When ``force_decimal=True`` (mandatory on industrial EIA/ISO controllers so
    integer-valued coordinates like ``Z6.0`` or ``F1194.0`` are never interpreted
    in least-input increments such as ``0.006 mm``):
    - ``precision >= 1`` always retains a decimal point and at least one digit
      after ``.`` (e.g. ``6.0``, ``-10.5``, ``0.0``) with at most ``precision``
      fractional digits.
    - ``precision == 0`` rounds to the nearest integer and appends a trailing
      decimal point ``.`` (e.g. ``1194.``, ``0.``) so zero fractional digits are
      emitted while still explicitly marking a floating-point register value.
    When ``force_decimal=False`` (legacy ``RS274``), trailing zeros and trailing
    ``.`` are stripped (``6``, ``-10.5``, ``0``).
    """
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"numeric G-code words must be finite; got {value!r}")
    if precision < 0:
        raise ValueError(f"precision must be >= 0; got {precision}")

    if precision == 0:
        rounded = int(round(val))
        s_int = "0" if rounded == 0 else str(rounded)
        return f"{s_int}." if force_decimal else s_int

    s = f"{val:.{precision}f}"
    if force_decimal:
        s = s.rstrip("0")
        if s.endswith("."):
            s = f"{s}0"
        if s in ("-0.0", "-0."):
            return "0.0"
        return s

    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def fmt_int(value: int | float, *, name: str = "register") -> str:
    """Format an integer register word (``O``, ``T``, ``H``, ``S``, ``P``, ``Q``) with no decimal."""
    val = float(value)
    if not math.isfinite(val) or int(val) != val:
        raise ValueError(f"{name} requires an exact finite integer; got {value!r}")
    return str(int(val))


def sanitize_comment(text: str) -> str:
    """Sanitize a comment payload so it cannot break ``( ... )`` EIA/ISO blocks."""
    cleaned = re.sub(r"[();%\r\n]+", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "DIFFCAM"


def format_program_number(program_number: int, *, digits: int, max_number: int) -> str:
    """Format an ``O`` program number with exact zero-padded width."""
    if isinstance(program_number, bool) or not isinstance(program_number, int):
        raise ValueError(f"program_number must be an integer; got {program_number!r}")
    if not (1 <= program_number <= max_number):
        raise ValueError(f"program_number must be in 1..{max_number}; got {program_number}")
    return f"O{program_number:0{digits}d}"


def axis_words(point_out, precision: int, *, force_decimal: bool) -> str:
    """Format ``X.. Y.. Z..`` axis words in output units."""
    x, y, z = point_out
    return (
        f"X{fmt_real(float(x), precision, force_decimal=force_decimal)} "
        f"Y{fmt_real(float(y), precision, force_decimal=force_decimal)} "
        f"Z{fmt_real(float(z), precision, force_decimal=force_decimal)}"
    )


def xy_words(x_out: float, y_out: float, precision: int, *, force_decimal: bool) -> str:
    """Format ``X.. Y..`` words in output units."""
    return (
        f"X{fmt_real(float(x_out), precision, force_decimal=force_decimal)} "
        f"Y{fmt_real(float(y_out), precision, force_decimal=force_decimal)}"
    )


def posted_work_vector(point_work, config: MachineConfig, precision: int) -> np.ndarray:
    """Return work-frame coordinates quantized in output units at ``precision``."""
    converted = len_out(np.asarray(point_work, dtype=np.float64), config)
    return np.asarray(
        [float(fmt_real(float(v), precision, force_decimal=True)) for v in converted],
        dtype=np.float64,
    )


def tool_table_comment(task: TaskSpec, config: MachineConfig) -> str:
    """Return a clean-room shop-floor tool description comment line."""
    diam_out = float(len_out(2.0 * float(task.tool.cutter_radius_mm), config))
    prec = 4 if config.units == "inch" else 3
    cr_out = float(len_out(float(task.tool.cutter_radius_mm), config)) if task.tool.cutter_kind.value == "ball" else 0.0
    kind_label = "BALL END MILL" if task.tool.cutter_kind.value == "ball" else "FLAT END MILL"
    d_str = fmt_real(diam_out, prec, force_decimal=True)
    cr_str = fmt_real(cr_out, prec, force_decimal=True)
    return f"(T{fmt_int(config.tool_number, name='T')} D={d_str} CR={cr_str} - {kind_label})"


def arc_offset_words(
    start_work,
    end_work,
    centre_work,
    plane: ArcPlane,
    config: MachineConfig,
    precision: int,
    segment_id: str,
    *,
    force_decimal: bool,
) -> list[str]:
    """Quantize incremental ``I/J/K`` arc centre offsets within ``ARC_RADIUS_TOL_MM``."""
    from cam.gcode_parser import _typed_plane_axes
    from cam.posts import UnsupportedPostError

    start_out = posted_work_vector(start_work, config, precision)
    end_out = posted_work_vector(end_work, config, precision)
    if math.dist(start_work, end_work) > 0 and np.array_equal(start_out, end_out):
        raise UnsupportedPostError(
            f"segment {segment_id}: distinct arc endpoints collapse to the same posted point at "
            f"{config.units} precision {precision}; rejecting a partial arc that would become a full turn"
        )
    centre_out = len_out(np.asarray(centre_work, dtype=np.float64), config)
    a, b, _ = _typed_plane_axes(plane)
    axis_names = {0: "I", 1: "J", 2: "K"}
    unit_to_mm = 25.4 if config.units == "inch" else 1.0
    step = 10.0 ** (-precision)
    native_mean_radius = 0.5 * (
        np.linalg.norm(np.asarray(start_work)[[a, b]] - np.asarray(centre_work)[[a, b]])
        + np.linalg.norm(np.asarray(end_work)[[a, b]] - np.asarray(centre_work)[[a, b]])
    )
    best = (math.inf, math.inf, math.inf, [])
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            centre = start_out.copy()
            candidate = []
            for axis_index, shift in zip((a, b), (dx, dy)):
                offset = centre_out[axis_index] - start_out[axis_index] + shift * step
                word = fmt_real(float(offset), precision, force_decimal=force_decimal)
                centre[axis_index] += float(word)
                candidate.append(f"{axis_names[axis_index]}{word}")
            r_start = np.linalg.norm((start_out[[a, b]] - centre[[a, b]]) * unit_to_mm)
            r_end = np.linalg.norm((end_out[[a, b]] - centre[[a, b]]) * unit_to_mm)
            if min(r_start, r_end) <= ARC_RADIUS_TOL_MM:
                continue
            radius_error = abs(r_start - r_end)
            mean_error = abs(0.5 * (r_start + r_end) - native_mean_radius)
            score = (radius_error, mean_error, dx * dx + dy * dy)
            if score < best[:3]:
                best = (*score, candidate)
    error, _, _, words = best
    if not words:
        raise UnsupportedPostError(
            f"segment {segment_id}: post quantization collapses the arc to zero/near-zero radius at "
            f"{config.units} precision {precision}; increase post precision or reject this incompatible arc"
        )
    if error > ARC_RADIUS_TOL_MM + 1e-12:
        raise UnsupportedPostError(
            f"segment {segment_id}: no nearest-grid arc offset preserves radii at {config.units} "
            f"precision {precision} (best error {error:g} mm > {ARC_RADIUS_TOL_MM:g} mm). "
            "Increase post precision or reject this incompatible arc; it was not emitted"
        )
    return words


def build_motion_lines(
    plan: MotionPlan,
    task: TaskSpec,
    config: MachineConfig,
    *,
    xyz_precision: int,
    feed_precision: int,
    force_decimal: bool,
) -> list[str]:
    """Emit ``G0``/``G1``/``G2``/``G3`` blocks for ``plan.segments``."""
    from cam.gcode_parser import _mapped_arc_orientation
    from cam.posts import UnsupportedPostError

    inverse = task.work_to_machine.inverse()
    output_plane = ArcPlane.XY
    lines: list[str] = []
    plane_codes = {ArcPlane.XY: "G17", ArcPlane.XZ: "G18", ArcPlane.YZ: "G19"}

    for segment in plan.segments:
        home = segment.tag == "home"
        if segment.kind == SegmentKind.ARC:
            if home:
                raise UnsupportedPostError("G53 does not support arc motion")
            arc = segment.arc
            try:
                work_plane, work_direction = _mapped_arc_orientation(
                    inverse, arc.plane, arc.direction
                )
            except ValueError as error:
                raise UnsupportedPostError(str(error)) from error
            if work_plane != output_plane:
                lines.append(plane_codes[work_plane])
                output_plane = work_plane
            end_work = inverse.apply(segment.end_mm)
            centre_work = inverse.apply(arc.centre_mm)
            start_work = inverse.apply(segment.start_mm)
            end_out = posted_work_vector(end_work, config, xyz_precision)
            words = [axis_words(end_out, xyz_precision, force_decimal=force_decimal)]
            words.extend(
                arc_offset_words(
                    start_work,
                    end_work,
                    centre_work,
                    work_plane,
                    config,
                    xyz_precision,
                    segment.id,
                    force_decimal=force_decimal,
                )
            )
            if arc.turns:
                words.append(f"P{fmt_int(arc.turns + 1, name='P')}")
            f_val = feed_out(segment.feed_mm_s * 60.0, config)
            words.append(f"F{fmt_real(f_val, feed_precision, force_decimal=force_decimal)}")
            code = "G2" if work_direction == ArcDirection.CW else "G3"
            lines.append(f"{code} {' '.join(words)}")
            continue

        code = "G0" if segment.kind == SegmentKind.RAPID else "G1"
        target = segment.end_mm if home else inverse.apply(segment.end_mm)
        target_out = len_out(np.asarray(target, dtype=np.float64), config)
        words = [
            f"G53 {code}" if home else code,
            axis_words(target_out, xyz_precision, force_decimal=force_decimal),
        ]
        if segment.kind == SegmentKind.FEED:
            f_val = feed_out(segment.feed_mm_s * 60.0, config)
            words.append(f"F{fmt_real(f_val, feed_precision, force_decimal=force_decimal)}")
        lines.append(" ".join(words))
    return lines


_CODE_TOKEN_RE = re.compile(r"([A-Za-z])\s*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))")


def strip_nc_comments(raw_line: str) -> str:
    """Return executable NC text on ``raw_line`` with ``(...)`` and ``;...`` removed."""
    out: list[str] = []
    depth = 0
    for ch in raw_line:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == ";" and depth == 0:
            break
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()


def check_real_word_decimals(
    nc_text: str,
    *,
    real_letters: frozenset[str] = frozenset({"X", "Y", "Z", "I", "J", "K", "R", "F", "E"}),
    int_letters: frozenset[str] = frozenset({"O", "T", "H", "S", "M", "N", "Q"}),
    max_xyz_decimals: int | None = None,
    max_feed_decimals: int | None = None,
    g05_r_is_integer: bool = False,
) -> None:
    """Verify decimal-point presence and fractional-digit bounds on emitted G-code."""
    xyz_like = {"X", "Y", "Z", "I", "J", "K", "E"}
    for line_no, raw in enumerate(nc_text.splitlines(), start=1):
        code = strip_nc_comments(raw)
        if not code or code == "%":
            continue
        tokens = _CODE_TOKEN_RE.findall(code)
        is_g05_block = any(
            ltr.upper() == "G" and abs(float(num) - 5.1) <= 1e-9 for ltr, num in tokens
        )
        is_g65_block = any(
            ltr.upper() == "G" and abs(float(num) - 65.0) <= 1e-9 for ltr, num in tokens
        )
        if is_g65_block:
            for ltr_raw, num_str in tokens:
                if ltr_raw.upper() == "P" and "." in num_str:
                    raise HardwareSyntaxError(
                        f"Line {line_no}: G65 macro program number P{num_str} must be an integer"
                    )
            continue
        for ltr_raw, num_str in tokens:
            ltr = ltr_raw.upper()
            if g05_r_is_integer and is_g05_block and ltr == "R":
                if "." in num_str:
                    raise HardwareSyntaxError(
                        f"Line {line_no}: AICC G05.1 R level must be an integer, got R{num_str}"
                    )
                continue
            if ltr in int_letters:
                if "." in num_str:
                    raise HardwareSyntaxError(
                        f"Line {line_no}: integer register {ltr}{num_str} must not contain a decimal point"
                    )
                continue
            if ltr in real_letters:
                if "." not in num_str:
                    raise HardwareSyntaxError(
                        f"Line {line_no}: real word {ltr}{num_str} is missing a mandatory decimal point"
                    )
                frac = num_str.split(".", 1)[1]
                if max_xyz_decimals is not None and (ltr in xyz_like or ltr == "R"):
                    if len(frac) > max_xyz_decimals:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: {ltr}{num_str} has {len(frac)} decimals (max {max_xyz_decimals})"
                        )
                if max_feed_decimals is not None and ltr == "F":
                    if len(frac) > max_feed_decimals:
                        raise HardwareSyntaxError(
                            f"Line {line_no}: F{num_str} has {len(frac)} decimals (max {max_feed_decimals})"
                        )


class ControllerDialectSpec(ABC):
    """Abstract clean-room specification for a physical CNC controller dialect."""

    dialect: ControllerDialect
    display_name: str
    file_extension: str
    force_decimal: bool
    program_digits: int | None
    max_program_number: int
    leading_percent: bool
    trailing_percent: bool
    shop_xyz_decimals_mm: int = 3
    shop_xyz_decimals_inch: int = 4
    shop_feed_decimals_mm: int = 2
    shop_feed_decimals_inch: int = 3

    def shop_xyz_decimals(self, units: str) -> int:
        return self.shop_xyz_decimals_inch if units == "inch" else self.shop_xyz_decimals_mm

    def shop_feed_decimals(self, units: str) -> int:
        return self.shop_feed_decimals_inch if units == "inch" else self.shop_feed_decimals_mm

    @abstractmethod
    def format_typed_program(
        self,
        motion_lines: Sequence[str],
        *,
        plan: MotionPlan,
        task: TaskSpec,
        config: MachineConfig,
    ) -> str:
        """Wrap ``motion_lines`` in the controller's typed prologue/smoothing/epilogue."""

    @abstractmethod
    def format_shop_program(
        self,
        plan: MotionPlan,
        task: TaskSpec,
        config: MachineConfig,
        *,
        case_label: str | None = None,
        blend_tolerance_mm: float = 0.05,
    ) -> str:
        """Emit a standalone shop-floor program with safe home retracts and decimal caps."""

    @abstractmethod
    def format_legacy_positions_program(
        self,
        positions: np.ndarray,
        config: MachineConfig,
    ) -> str:
        """Emit a ``PostProcessor.program(positions, config)`` G-code string."""

    @abstractmethod
    def verify_hardware_syntax(
        self,
        nc_text: str,
        *,
        units: str = "mm",
        shop_mode: bool = False,
    ) -> None:
        """Raise ``HardwareSyntaxError`` if ``nc_text`` violates controller syntax rules."""
