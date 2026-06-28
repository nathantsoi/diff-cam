"""Unit conversions for the diff-cam pipeline.

The simulator and the whole CAM pipeline do every scale-related calculation in
**millimetres** (see ``MachineConfig.workspace_mm`` and the speed enforcement in
``simulator.csg_simulator.CSGSimulatorDelta``). Inches only appear at the I/O
boundary:

* **Inputs** -- any parameter a caller wants to specify in inches (feeds, safe
  distances, tool sizes) is converted to mm with :func:`inch_to_mm` /
  :func:`ipm_to_mm_per_s` *before* it enters the simulator.
* **Outputs** -- when G-code is requested in inches (``G20``), the post-processor
  converts the internal mm coordinates and feeds back to inches with
  :func:`mm_to_inch` / :func:`mm_per_min_to_ipm` *just prior* to emitting words.

This mirrors how LinuxCNC and CAMotics keep one canonical internal unit and only
scale by 25.4 at the edges (cf. ``CAMotics/src/gcode/Units.h``).

Every function accepts a Python scalar or a NumPy array (the arithmetic is plain
multiplication/division, so arrays broadcast naturally).
"""

# Exact inch definition, identical to the constant used by LinuxCNC / CAMotics.
MM_PER_INCH = 25.4


# ---------------------------------------------------------------------------
# Lengths
# ---------------------------------------------------------------------------
def inch_to_mm(value):
    """Length in inches -> millimetres."""
    return value * MM_PER_INCH


def mm_to_inch(value):
    """Length in millimetres -> inches."""
    return value / MM_PER_INCH


# ---------------------------------------------------------------------------
# Feeds / speeds
# ---------------------------------------------------------------------------
# Internally the simulator stores speeds in mm/s (distance per second); G-code
# feed words are mm/min or in/min. Provide both so callers never hand-roll a
# stray 25.4 or 60.

def ipm_to_mm_per_min(value):
    """Feed in inches/min -> millimetres/min."""
    return value * MM_PER_INCH


def mm_per_min_to_ipm(value):
    """Feed in millimetres/min -> inches/min."""
    return value / MM_PER_INCH


def ipm_to_mm_per_s(value):
    """Speed in inches/min -> millimetres/second (the simulator's internal unit)."""
    return value * MM_PER_INCH / 60.0


def mm_per_s_to_ipm(value):
    """Speed in millimetres/second -> inches/min."""
    return value * 60.0 / MM_PER_INCH


def mm_per_min_to_mm_per_s(value):
    """Feed in millimetres/min -> millimetres/second."""
    return value / 60.0


def mm_per_s_to_mm_per_min(value):
    """Speed in millimetres/second -> millimetres/min."""
    return value * 60.0
