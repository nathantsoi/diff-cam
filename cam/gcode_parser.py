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

import re
from dataclasses import dataclass, field

import numpy as np

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

        scale = config.workspace_mm

        if motion_mode in (0, 1):
            kind = "rapid" if motion_mode == 0 else "feed"
            seg = MotionSegment(
                kind=kind,
                start=pos_mm.copy() / scale,
                end=new_pos.copy() / scale,
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
                start=pos_mm.copy() / scale,
                end=new_pos.copy() / scale,
                feed=feed,
                center=center_mm / scale,
                cw=cw,
                plane=plane,
            )
            segments.append(seg)

        pos_mm = new_pos
        have_pos = True

    return segments


def segment_waypoints(segments):
    """Return the ordered waypoints implied by the segments (start of the first
    segment followed by every segment end), in unit-cube coords. Useful for
    verifying that export/parse recovers the original points."""
    if not segments:
        return np.zeros((0, 3), dtype=np.float64)
    pts = [segments[0].start]
    pts.extend(seg.end for seg in segments)
    return np.asarray(pts, dtype=np.float64)
