"""Parse RS274/NGC G-code into a list of motion segments.

This mirrors the front of LinuxCNC's interpreter: a modal state machine that
turns G-code blocks into canonical moves. We support the subset relevant to
diff-cam toolpaths:

  * G0 / G1                 -- rapid / feed linear moves
  * G2 / G3                 -- clockwise / counter-clockwise arcs (I/J/K or R)
  * G17 / G18 / G19         -- XY / XZ / YZ plane select (for arcs)
  * G20 / G21               -- inch / mm
  * G90 / G91               -- absolute / incremental distance mode
  * F                       -- feed rate

The parser is intentionally more general than the exporter (which only emits
G0/G1) so the system stays faithful to LinuxCNC and can be exercised with
hand-written arcs. All emitted coordinates are converted back to the unit cube
``[0, 1]^3`` via ``MachineConfig.workspace_mm``.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from benchmarks.contracts import (
    ArcData,
    ArcDirection,
    ArcPlane,
    Dialect,
    Frame,
    MotionPlan,
    Segment,
    SegmentKind,
    TaskSpec,
    TimingRule,
    arc_geometry,
    sha256_text,
)
from .config import MachineConfig

# A word is a letter followed by a (signed, optional-decimal) number.
_WORD_RE = re.compile(r"([A-Za-z])\s*([-+]?[0-9]*\.?[0-9]+)")

# Plane constants.
PLANE_XY, PLANE_XZ, PLANE_YZ = 17, 18, 19


@dataclass
class MotionSegment:
    """A single canonical move in unit-cube coordinates."""
    kind: str                       # "rapid" | "feed" | "arc"
    start: np.ndarray               # (3,)
    end: np.ndarray                 # (3,)
    feed: float = 0.0               # mm/min (0 for rapids)
    # Arc-only fields:
    center: np.ndarray = None       # (3,) arc centre, unit-cube coords
    cw: bool = False                # True for G2, False for G3
    plane: int = PLANE_XY


def _strip_comments(text: str) -> str:
    """Remove ``( ... )`` and ``; ...`` comments line by line."""
    out_lines = []
    for line in text.splitlines():
        line = re.sub(r"\([^)]*\)", "", line)   # parenthesised comments
        line = line.split(";", 1)[0]            # semicolon to end of line
        out_lines.append(line)
    return "\n".join(out_lines)


def _plane_axes(plane):
    """Return the two in-plane axis indices and the out-of-plane axis index."""
    if plane == PLANE_XY:
        return 0, 1, 2
    if plane == PLANE_XZ:
        return 0, 2, 1
    return 1, 2, 0  # PLANE_YZ


def _arc_center_from_offsets(start_mm, words, plane):
    """Centre (mm) from I/J/K offsets relative to the start point."""
    i = words.get("I", 0.0)
    j = words.get("J", 0.0)
    k = words.get("K", 0.0)
    return start_mm + np.array([i, j, k], dtype=np.float64)


def _signed_sweep(p0, p1, center2d, cw):
    """Sweep angle (signed, CCW positive) from p0 to p1 about center2d for the
    given direction. Magnitude in (0, 2*pi]."""
    a0 = np.arctan2(p0[1] - center2d[1], p0[0] - center2d[0])
    a1 = np.arctan2(p1[1] - center2d[1], p1[0] - center2d[0])
    sweep = a1 - a0
    if cw:
        while sweep >= 0:
            sweep -= 2.0 * np.pi
    else:
        while sweep <= 0:
            sweep += 2.0 * np.pi
    return sweep


def _arc_center_from_radius(start_mm, end_mm, radius, cw, plane):
    """Centre (mm) from an R-word, using the standard RS274 sign convention:
    R > 0 selects the arc <= 180 deg, R < 0 selects the arc > 180 deg."""
    a0, a1, _ = _plane_axes(plane)
    p0 = np.array([start_mm[a0], start_mm[a1]])
    p1 = np.array([end_mm[a0], end_mm[a1]])
    mid = 0.5 * (p0 + p1)
    chord = p1 - p0
    chord_len = np.linalg.norm(chord)
    if chord_len < 1e-12:
        raise ValueError("Arc with R-word has coincident start and end points")
    h_sq = radius * radius - (chord_len / 2.0) ** 2
    if h_sq < -1e-9:
        raise ValueError("Arc radius too small for the given endpoints")
    h = np.sqrt(max(h_sq, 0.0))
    # Unit normal to the chord; the two candidate centres lie at mid +/- h*n.
    n = np.array([-chord[1], chord[0]]) / chord_len
    cand_plus = mid + h * n
    cand_minus = mid - h * n
    # R > 0 -> minor arc (sweep <= pi); R < 0 -> major arc (sweep > pi).
    want_minor = radius > 0
    sweep_plus = abs(_signed_sweep(p0, p1, cand_plus, cw))
    sweep_minus = abs(_signed_sweep(p0, p1, cand_minus, cw))
    if want_minor:
        center2d = cand_plus if sweep_plus <= sweep_minus else cand_minus
    else:
        center2d = cand_plus if sweep_plus >= sweep_minus else cand_minus
    center = np.array(start_mm, dtype=np.float64)
    center[a0] = center2d[0]
    center[a1] = center2d[1]
    return center


def parse_gcode(text: str, config: MachineConfig = MachineConfig()):
    """Parse a G-code program into a list of :class:`MotionSegment`.

    Returns segments in execution order with all coordinates in the unit cube.
    """
    text = _strip_comments(text)

    pos_mm = np.zeros(3, dtype=np.float64)   # current position, machine mm
    have_pos = False
    motion_mode = 0          # 0=G0, 1=G1, 2=G2, 3=G3
    absolute = True          # G90
    plane = PLANE_XY         # G17
    feed = config.feed       # mm/min
    unit_scale = 1.0         # mm per program unit (25.4 under G20)

    segments = []

    for raw in text.splitlines():
        words = _WORD_RE.findall(raw)
        if not words:
            continue
        block = {}
        gcodes = []
        for letter, num in words:
            L = letter.upper()
            val = float(num)
            if L == "G":
                gcodes.append(val)
            else:
                block[L] = val

        # --- Apply modal G-codes in this block ---
        for g in gcodes:
            if g in (0.0, 1.0, 2.0, 3.0):
                motion_mode = int(g)
            elif g == 20.0:
                unit_scale = 25.4
            elif g == 21.0:
                unit_scale = 1.0
            elif g == 90.0:
                absolute = True
            elif g == 91.0:
                absolute = False
            elif g == 17.0:
                plane = PLANE_XY
            elif g == 18.0:
                plane = PLANE_XZ
            elif g == 19.0:
                plane = PLANE_YZ
            # other G-codes (G61/G64/etc.) carry no geometry here

        # G10 (programmable data input -- e.g. setting the G54 work-offset table)
        # carries X/Y/Z words but is NOT a motion; skip the block entirely.
        if 10.0 in gcodes:
            continue

        if "F" in block:
            feed = block["F"] * unit_scale

        # --- Resolve target position if any axis word is present ---
        axis_present = any(a in block for a in ("X", "Y", "Z"))
        arc_present = any(a in block for a in ("I", "J", "K", "R"))
        if not (axis_present or (motion_mode in (2, 3) and arc_present)):
            continue

        new_pos = pos_mm.copy()
        for idx, axis in enumerate(("X", "Y", "Z")):
            if axis in block:
                v = block[axis] * unit_scale
                new_pos[idx] = v if absolute else pos_mm[idx] + v

        if not have_pos:
            # The machine's pre-program position is undefined; treat the first
            # positioning move as establishing the start point. This makes the
            # executed trajectory begin exactly at the first commanded point
            # (no spurious travel from the origin).
            pos_mm = new_pos
            have_pos = True
            continue

        # Coordinate words are in the work coordinate system (mm) relative to the
        # top-centre G54 origin; invert the export mapping to normalized stock
        # coords [0,1]^3.
        if motion_mode in (0, 1):
            kind = "rapid" if motion_mode == 0 else "feed"
            seg = MotionSegment(
                kind=kind,
                start=config.wcs_to_unit(pos_mm.copy()),
                end=config.wcs_to_unit(new_pos.copy()),
                feed=0.0 if kind == "rapid" else feed,
            )
            segments.append(seg)
        else:  # arc: G2 (cw) or G3 (ccw)
            cw = (motion_mode == 2)
            offset_words = {k: block[k] * unit_scale for k in ("I", "J", "K") if k in block}
            if "R" in block:
                center_mm = _arc_center_from_radius(
                    pos_mm, new_pos, block["R"] * unit_scale, cw, plane
                )
            else:
                center_mm = _arc_center_from_offsets(pos_mm, offset_words, plane)
            seg = MotionSegment(
                kind="arc",
                start=config.wcs_to_unit(pos_mm.copy()),
                end=config.wcs_to_unit(new_pos.copy()),
                feed=feed,
                center=config.wcs_to_unit(center_mm),
                cw=cw,
                plane=plane,
            )  # NOTE: arcs assume near-isotropic scale; anisotropic envelopes warp circles
            segments.append(seg)

        pos_mm = new_pos
        have_pos = True

    return segments


# ---------------------------------------------------------------------------
# Strict RS274/Haas -> contracts.MotionPlan adapter
# ---------------------------------------------------------------------------
# The legacy adapter above deliberately retains its historical array-oriented
# behaviour (including omitting first-position establishment).  The API below is
# the task-aware interface for evaluation.  It never discards an unrecognised
# word and never guesses a missing initial position.


class GCodeError(ValueError):
    """An NC syntax or semantics error, with its physical source line."""


class UnsupportedGCodeError(GCodeError):
    """A valid NC feature outside this parser's declared, task-bounded subset."""


def _nc_error(line: int, message: str, unsupported: bool = False) -> GCodeError:
    cls = UnsupportedGCodeError if unsupported else GCodeError
    return cls(f"NC line {line}: {message}")


_STRICT_WORD_RE = re.compile(r"([A-Za-z])\s*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))")


def _strict_line_words(raw: str, line: int) -> list[tuple[str, float]]:
    """Tokenise one complete block, retaining no unexamined trailing text."""
    text = raw.split(";", 1)[0]
    chars: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            if depth:
                raise _nc_error(line, "nested parenthesised comments are unsupported", True)
            depth = 1
        elif char == ")":
            if not depth:
                raise _nc_error(line, "unmatched closing comment delimiter")
            depth = 0
        elif not depth:
            chars.append(char)
    if depth:
        raise _nc_error(line, "unterminated parenthesised comment")
    text = "".join(chars).strip()
    if not text or text == "%":
        return []

    words: list[tuple[str, float]] = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        match = _STRICT_WORD_RE.match(text, offset)
        if match is None:
            raise _nc_error(line, f"invalid or unsupported NC syntax near {text[offset:]!r}")
        words.append((match.group(1).upper(), float(match.group(2))))
        offset = match.end()
    return words


def _single_modal(gcodes: list[float], members: tuple[float, ...], line: int, name: str) -> None:
    if sum(g in members for g in gcodes) > 1:
        raise _nc_error(line, f"conflicting {name} modes: {gcodes}")


def _integer_word(block: dict[str, float], letter: str, line: int, minimum: int = 0) -> int:
    value = block[letter]
    integer = int(value)
    if integer != value or integer < minimum:
        raise _nc_error(line, f"{letter} requires an integer >= {minimum}; got {value}")
    return integer


def _typed_plane_axes(plane: ArcPlane) -> tuple[int, int, int]:
    """Right-handed bases shared with contracts.arc_geometry()."""
    if plane == ArcPlane.XY:
        return 0, 1, 2
    if plane == ArcPlane.XZ:
        return 2, 0, 1
    return 1, 2, 0


def _strict_sweep(start, end, centre, plane: ArcPlane, direction: ArcDirection) -> float:
    a, b, _ = _typed_plane_axes(plane)
    theta0 = math.atan2(start[b] - centre[b], start[a] - centre[a])
    theta1 = math.atan2(end[b] - centre[b], end[a] - centre[a])
    ccw = (theta1 - theta0) % (2.0 * math.pi)
    if direction == ArcDirection.CCW:
        return (ccw if ccw > 1e-12 else 2.0 * math.pi)
    cw = (2.0 * math.pi - ccw) % (2.0 * math.pi)
    return -(cw if cw > 1e-12 else 2.0 * math.pi)


def _radius_centre(start, end, radius: float, plane: ArcPlane, direction: ArcDirection, line: int):
    """Resolve R in the work plane; sign selects minor/major as in RS274."""
    if radius == 0.0:
        raise _nc_error(line, "arc R must be non-zero")
    a, b, _ = _typed_plane_axes(plane)
    p0 = np.array([start[a], start[b]], dtype=np.float64)
    p1 = np.array([end[a], end[b]], dtype=np.float64)
    chord = p1 - p0
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-12:
        raise _nc_error(line, "R cannot define a full circle; use I/J/K")
    h_squared = radius * radius - (chord_length / 2.0) ** 2
    if h_squared < -1e-8 * max(1.0, radius * radius):
        raise _nc_error(line, "arc radius is too small for its endpoint chord")
    h = math.sqrt(max(0.0, h_squared))
    normal = np.array([-chord[1], chord[0]], dtype=np.float64) / chord_length
    midpoint = 0.5 * (p0 + p1)
    candidates = (midpoint + h * normal, midpoint - h * normal)
    resolved = []
    for candidate in candidates:
        centre = np.asarray(start, dtype=np.float64).copy()
        centre[a], centre[b] = candidate
        sweep = _strict_sweep(start, end, centre, plane, direction)
        resolved.append((abs(sweep), centre))
    if radius > 0.0:
        return min(resolved, key=lambda item: item[0])[1]
    return max(resolved, key=lambda item: item[0])[1]


def _mapped_arc_orientation(transform, plane: ArcPlane, direction: ArcDirection) -> tuple[ArcPlane, ArcDirection]:
    """Map a right-handed arc normal through a rigid transform.

    Axis-aligned plane mappings preserve or reverse CCW as appropriate.  A
    rotation that tilts an arc away from every machine principal plane cannot be
    represented by contracts.ArcPlane and is therefore rejected explicitly.
    """
    _, _, normal_index = _typed_plane_axes(plane)
    normal = [0.0, 0.0, 0.0]
    normal[normal_index] = 1.0
    rotation = [row[:3] for row in transform.matrix[:3]]
    mapped = [sum(rotation[i][j] * normal[j] for j in range(3)) for i in range(3)]
    for candidate in ArcPlane:
        _, _, candidate_index = _typed_plane_axes(candidate)
        projection = mapped[candidate_index]
        remainder = mapped.copy()
        remainder[candidate_index] = 0.0
        if abs(abs(projection) - 1.0) <= 1e-9 and math.hypot(*remainder) <= 1e-9:
            if projection < 0.0:
                direction = ArcDirection.CW if direction == ArcDirection.CCW else ArcDirection.CCW
            return candidate, direction
    raise UnsupportedGCodeError(
        f"work_to_machine rotates {plane.value} arcs outside every principal machine plane"
    )


def _motion_tag(kind: SegmentKind, start, end, index: int, home: bool) -> str:
    if home:
        return "home"
    dx, dy, dz = (end[i] - start[i] for i in range(3))
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9 and abs(dz) > 1e-9:
        return "retract" if dz > 0.0 else "plunge"
    if kind == SegmentKind.FEED and abs(dz) > 1e-9:
        return "ramp"
    if index == 0 and kind == SegmentKind.RAPID:
        return "approach"
    return ""


def _strict_arc(
    start_work,
    end_work,
    block: dict[str, float],
    offsets_present: bool,
    plane: ArcPlane,
    direction: ArcDirection,
    transform,
    line: int,
):
    a, b, normal = _typed_plane_axes(plane)
    axis_names = ("X", "Y", "Z")
    offset_names = {"X": "I", "Y": "J", "Z": "K"}
    allowed_offsets = {offset_names[axis_names[a]], offset_names[axis_names[b]]}
    supplied_offsets = {letter for letter in ("I", "J", "K") if letter in block}
    if "R" in block and supplied_offsets:
        raise _nc_error(line, "R and I/J/K centre forms are mutually exclusive")
    if not offsets_present and "R" not in block:
        raise _nc_error(line, "G2/G3 requires R or I/J/K centre data")
    outside = supplied_offsets - allowed_offsets
    if outside:
        raise _nc_error(
            line,
            f"out-of-plane arc offset(s) {sorted(outside)} are unsupported for {plane.value}",
            True,
        )

    if "R" in block:
        centre_work = _radius_centre(start_work, end_work, block["R"], plane, direction, line)
    else:
        centre_work = np.asarray(start_work, dtype=np.float64).copy()
        centre_work[a] = start_work[a] + block.get(offset_names[axis_names[a]], 0.0)
        centre_work[b] = start_work[b] + block.get(offset_names[axis_names[b]], 0.0)
        centre_work[normal] = start_work[normal]

    turns = 0
    if "P" in block:
        turns = _integer_word(block, "P", line, minimum=1) - 1
        if "R" in block and turns:
            raise _nc_error(line, "multi-turn R arcs are ambiguous; use I/J/K", True)

    centre_machine = transform.apply(centre_work)
    machine_plane, machine_direction = _mapped_arc_orientation(transform, plane, direction)
    arc = ArcData(
        centre_mm=centre_machine,
        plane=machine_plane,
        direction=machine_direction,
        turns=turns,
    )
    return arc


def parse_motion_plan(
    text: str,
    task: TaskSpec,
    dialect: Dialect | str = Dialect.RS274,
) -> MotionPlan:
    """Parse a declared RS274/Haas subset into a machine-frame MotionPlan.

    G54 positions are transformed by ``task.work_to_machine``.  G53 is a
    single-block absolute machine move (G91+G53 and G53 arcs are rejected).
    I/J/K are always start-relative offsets, including in G91.  All three
    principal arc planes and helical/multi-turn arcs use ``arc_geometry`` from
    the shared contract.

    Tool-length register tables are not modeled.  G43 Hn declares that the H
    register is calibrated so coordinates refer to the cutter tip; G49 declares
    the equivalent zero-length/gauge-tip convention.  The caller must provision
    those registers for physical execution.  Tool changes are limited to one
    repeated tool number because TaskSpec describes one fixed tool assembly.

    The plan uses the task's declared timing rule; G61/G61.1 are accepted and
    G64/other blended path-control modes are unsupported.  Initial units,
    distance mode, motion mode, and feed must be explicit.  Most importantly,
    an unknown initial position is an error at the first motion: the approach
    is never silently omitted.
    """
    dialect = Dialect(dialect)
    if dialect not in task.motion_semantics.dialects:
        raise UnsupportedGCodeError(f"dialect {dialect.value!r} is disabled by this TaskSpec")
    if task.initial_state.work_offset != "G54":
        raise UnsupportedGCodeError("only the G54 work offset is represented by TaskSpec")
    if task.initial_state.known and task.initial_state.frame != Frame.MACHINE:
        raise GCodeError("TaskSpec.initial_state must use the machine frame")
    if task.initial_state.length_offset_applied not in (True, False):
        raise GCodeError("invalid initial tool-length mode")

    transform = task.work_to_machine
    inverse = transform.inverse()
    position_machine = task.initial_state.position_mm
    segments: list[Segment] = []
    units_scale: Optional[float] = None
    absolute: Optional[bool] = None
    motion_mode: Optional[int] = None
    plane = ArcPlane.XY
    feed_mm_s: Optional[float] = None
    length_offset = task.initial_state.length_offset_applied
    tool_number: Optional[int] = None
    program_ended = False

    supported_g = {
        0.0, 1.0, 2.0, 3.0, 17.0, 18.0, 19.0, 20.0, 21.0,
        40.0, 43.0, 49.0, 53.0, 54.0, 61.0, 61.1, 80.0, 90.0, 91.0, 94.0,
    }
    supported_m = {0, 1, 2, 3, 4, 5, 6, 8, 9, 30}
    allowed_words = set("GMXYZIJKRFPSTHNOL")

    for line, raw in enumerate(text.splitlines(), start=1):
        try:
            words = _strict_line_words(raw, line)
            if not words:
                continue
            if program_ended:
                raise _nc_error(line, "content follows M2/M30 program end")
            gcodes: list[float] = []
            mcodes: list[int] = []
            block: dict[str, float] = {}
            for letter, value in words:
                if letter not in allowed_words:
                    raise _nc_error(line, f"unsupported geometry/setup word {letter}{value:g}", True)
                if letter == "G":
                    gcodes.append(value)
                elif letter == "M":
                    if int(value) != value:
                        raise _nc_error(line, f"M code must be an integer; got {value}")
                    mcodes.append(int(value))
                else:
                    if letter in block:
                        raise _nc_error(line, f"duplicate {letter} word")
                    block[letter] = value

            for group in ((0.0, 1.0, 2.0, 3.0), (17.0, 18.0, 19.0),
                          (20.0, 21.0), (43.0, 49.0), (90.0, 91.0)):
                _single_modal(gcodes, group, line, "modal")
            for code in gcodes:
                if code in (10.0, 28.0, 30.0):
                    raise _nc_error(
                        line,
                        f"G{code:g} is unsupported: no programmable-offset/home register exists in TaskSpec; "
                        "use task.work_to_machine or an explicit absolute G53 move",
                        True,
                    )
                if code in (64.0, 93.0, 95.0):
                    raise _nc_error(line, f"G{code:g} changes timing/blending semantics", True)
                if code not in supported_g:
                    raise _nc_error(line, f"unsupported G{code:g}", True)
            if 53.0 in gcodes and 54.0 in gcodes:
                raise _nc_error(line, "G53 and G54 cannot share a block")
            if any(code not in supported_m for code in mcodes):
                raise _nc_error(line, f"unsupported M code(s): {mcodes}", True)
            if "L" in block:
                raise _nc_error(line, f"unsupported setup word L{block['L']:g}", True)

            if "N" in block:
                _integer_word(block, "N", line)
            if "O" in block:
                _integer_word(block, "O", line)
            if "T" in block:
                selected_tool = _integer_word(block, "T", line, minimum=1)
                if tool_number is not None and selected_tool != tool_number:
                    raise _nc_error(line, "multiple tool numbers are incompatible with fixed TaskSpec.tool", True)
                tool_number = selected_tool
            if "H" in block and 43.0 not in gcodes:
                raise _nc_error(line, "H is only accepted on a G43 block")

            if 20.0 in gcodes:
                units_scale = 25.4
            if 21.0 in gcodes:
                units_scale = 1.0
            if 90.0 in gcodes:
                absolute = True
            if 91.0 in gcodes:
                absolute = False
            if 17.0 in gcodes:
                plane = ArcPlane.XY
            if 18.0 in gcodes:
                plane = ArcPlane.XZ
            if 19.0 in gcodes:
                plane = ArcPlane.YZ
            for mode in (0, 1, 2, 3):
                if float(mode) in gcodes:
                    motion_mode = mode
            if 43.0 in gcodes:
                if "H" not in block:
                    raise _nc_error(line, "G43 requires an explicit H register (e.g. G43 H1)")
                _integer_word(block, "H", line)
                length_offset = True
            if 49.0 in gcodes:
                length_offset = False

            if "F" in block:
                if units_scale is None:
                    raise _nc_error(line, "G20/G21 must precede feed interpretation")
                if block["F"] <= 0.0:
                    raise _nc_error(line, "F must be positive")
                feed_mm_s = block["F"] * units_scale / 60.0

            axis_present = any(axis in block for axis in ("X", "Y", "Z"))
            offsets_present = any(word in block for word in ("I", "J", "K"))
            arc_words_present = offsets_present or "R" in block or "P" in block
            # G49 is emitted in a no-motion safety prologue even when the declared
            # mode is G43; G43 re-establishes it before motion. A no-motion register
            # declaration does not alter trajectory, but every MOTION block must
            # match the task's single declared initial state.
            if (axis_present or arc_words_present) and length_offset != task.initial_state.length_offset_applied:
                raise _nc_error(
                    line,
                    f"G43/G49 changes tool-length state to {length_offset}, incompatible with "
                    f"task.initial_state.length_offset_applied={task.initial_state.length_offset_applied}; "
                    "intermediate register changes outside the single declared initial state are unsupported",
                    True,
                )
            requested_offset_commands = {43.0, 49.0}.intersection(gcodes)
            expected_offset_command = {43.0} if task.initial_state.length_offset_applied else {49.0}
            if ((axis_present or arc_words_present) and requested_offset_commands
                    and requested_offset_commands != expected_offset_command):
                raise _nc_error(
                    line,
                    "tool-length-changing G43/G49 block with motion cannot represent the one declared initial state; "
                    f"motion may contain only {sorted(expected_offset_command)} for this task",
                    True,
                )
            if not axis_present and not arc_words_present:
                if 2 in mcodes or 30 in mcodes:
                    program_ended = True
                continue
            if units_scale is None or absolute is None:
                raise _nc_error(line, "G20/G21 and G90/G91 must precede motion")
            if motion_mode is None:
                raise _nc_error(line, "motion mode is unknown; declare G0/G1/G2/G3")
            if motion_mode not in (2, 3) and arc_words_present:
                raise _nc_error(line, "I/J/K/R/P are only valid on an arc block")
            if not task.initial_state.known or position_machine is None:
                raise _nc_error(
                    line,
                    "initial machine position is unknown; refusing to omit or invent the approach",
                    True,
                )
            if motion_mode in (1, 2, 3) and feed_mm_s is None:
                raise _nc_error(line, "feed/arc motion requires an explicit positive F word")

            start_machine = tuple(position_machine)
            start_work = inverse.apply(start_machine)
            machine_move = 53.0 in gcodes
            if machine_move and (not absolute or motion_mode in (2, 3)):
                raise _nc_error(line, "G53 supports only absolute G0/G1 moves", True)

            if machine_move:
                target = list(start_machine)
                for index, axis in enumerate(("X", "Y", "Z")):
                    if axis in block:
                        target[index] = block[axis] * units_scale
                end_machine = tuple(target)
                end_work = inverse.apply(end_machine)
            else:
                target_work = list(start_work)
                for index, axis in enumerate(("X", "Y", "Z")):
                    if axis in block:
                        value = block[axis] * units_scale
                        target_work[index] = value if absolute else start_work[index] + value
                end_work = tuple(target_work)
                end_machine = transform.apply(end_work)

            scaled_block = {
                key: value * units_scale
                for key, value in block.items()
                if key in ("I", "J", "K", "R")
            }
            if "P" in block:
                scaled_block["P"] = block["P"]
            segment_id = f"nc-{line:06d}"
            home = machine_move
            if motion_mode == 0:
                kind = SegmentKind.RAPID
                duration = math.dist(start_machine, end_machine) / task.limits.rapid_mm_s
                feed = None
                arc = None
            elif motion_mode == 1:
                kind = SegmentKind.FEED
                duration = math.dist(start_machine, end_machine) / feed_mm_s
                feed = feed_mm_s
                arc = None
            else:
                if machine_move:
                    raise _nc_error(line, "G53 arcs are unsupported", True)
                kind = SegmentKind.ARC
                direction = ArcDirection.CW if motion_mode == 2 else ArcDirection.CCW
                arc = _strict_arc(
                    start_work,
                    end_work,
                    scaled_block,
                    offsets_present,
                    plane,
                    direction,
                    transform,
                    line,
                )
                end_machine = transform.apply(end_work)
                duration = arc_geometry(start_machine, end_machine, arc, segment_id)["length_mm"] / feed_mm_s
                feed = feed_mm_s

            segments.append(
                Segment(
                    id=segment_id,
                    kind=kind,
                    start_mm=start_machine,
                    end_mm=end_machine,
                    duration_s=duration,
                    feed_mm_s=feed,
                    arc=arc,
                    tag=_motion_tag(kind, start_machine, end_machine, len(segments), home),
                    source_line=line,
                )
            )
            position_machine = end_machine
            if 2 in mcodes or 30 in mcodes:
                program_ended = True
        except (GCodeError, ValueError) as error:
            if isinstance(error, GCodeError):
                raise
            raise _nc_error(line, str(error)) from error

    plan = MotionPlan(
        frame=Frame.MACHINE,
        initial_state=task.initial_state,
        segments=tuple(segments),
        timing_rule=task.motion_semantics.timing_rule,
        provenance=f"parsed:{dialect.value}:{sha256_text(text)}",
    )
    try:
        plan.validate_for_task(task)
    except ValueError as error:
        raise GCodeError(f"typed NC does not satisfy TaskSpec: {error}") from error
    return plan


def segment_waypoints(segments):
    """Return the ordered waypoints implied by the segments (start of the first
    segment followed by every segment end), in unit-cube coords. Useful for
    verifying that export/parse recovers the original points."""
    if not segments:
        return np.zeros((0, 3), dtype=np.float64)
    pts = [segments[0].start]
    pts.extend(seg.end for seg in segments)
    return np.asarray(pts, dtype=np.float64)
