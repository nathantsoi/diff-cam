"""Machine / CAM configuration.

A single dataclass holds every parameter that the G-code export, parser, and
trajectory planner must agree on. Most importantly it owns ``workspace_mm``, the
physical edge length (mm) of the simulator's unit cube ``[0, 1]^3``. Export
multiplies unit-cube coordinates by this scale; the parser divides by it. Keeping
the scale in one place is what makes the round trip
``trajectory -> G-code -> trajectory`` reproduce the original path exactly.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MachineConfig:
    # --- Coordinate mapping ---------------------------------------------------
    # The trajectory lives in the normalized box [0,1]^3. ``workspace_mm`` is the
    # legacy scalar edge length (cube). For a non-cubic machine envelope (e.g.
    # the Haas Mini Mill, 16x12x10 in) set ``workspace_in`` to a per-axis
    # (x, y, z) tuple in inches, which overrides the scalar. ``workspace_vec``
    # resolves either form to a (3,) mm vector used everywhere downstream.
    workspace_mm: float = 100.0   # physical edge length of the unit cube [0,1]^3
    workspace_in: Optional[Tuple[float, float, float]] = None  # per-axis envelope, inches

    # --- Feeds & speeds -------------------------------------------------------
    feed: float = 600.0           # cutting feed rate, mm/min (G1)
    rapid: float = 3000.0         # rapid traverse rate, mm/min (G0)
    max_accel: float = 500.0      # acceleration limit, mm/s^2

    # --- Planner --------------------------------------------------------------
    dt: float = 0.01              # planner sample period, seconds

    # --- Formatting -----------------------------------------------------------
    units: str = "mm"             # "mm" -> G21, "inch" -> G20
    precision: int = 6            # decimal places for coordinate words

    # --- Machine / post-processor setup --------------------------------------
    # Used by machine-specific posts (e.g. the Haas post). The generic RS274
    # post ignores everything below.
    program_number: int = 1       # Haas O-number (O00001)
    program_name: str = "DIFFCAM" # program comment / name
    tool_number: int = 1          # Txx for the tool change
    length_offset: int = 0        # G43 Hxx register; 0 -> mirror tool_number
    spindle_rpm: float = 5000.0   # Sxxxx M03
    coolant: bool = True          # flood coolant (M08 / M09)
    work_offset: str = "G54"      # work coordinate system
    retract_mm: float = 10.0      # clearance plane above the top of the workspace
    plunge_feed: float = 200.0    # Z plunge feed rate, mm/min

    @property
    def workspace_vec(self):
        """Per-axis envelope as a (3,) mm vector (resolves scalar or inch form)."""
        import numpy as np
        from .units import inch_to_mm
        if self.workspace_in is not None:
            return np.asarray([inch_to_mm(c) for c in self.workspace_in], dtype=np.float64)
        return np.asarray([self.workspace_mm] * 3, dtype=np.float64)

    @property
    def units_code(self) -> str:
        """Modal G-code for the configured units."""
        return "G21" if self.units == "mm" else "G20"

    @property
    def safe_z_mm(self) -> float:
        """Clearance Z (mm): ``retract_mm`` above the top of the envelope (Z axis)."""
        return float(self.workspace_vec[2]) + self.retract_mm

    @property
    def h_register(self) -> int:
        """Tool-length-offset register for G43 (defaults to the tool number)."""
        return self.length_offset or self.tool_number

    @property
    def feed_mm_per_s(self) -> float:
        return self.feed / 60.0

    @property
    def rapid_mm_per_s(self) -> float:
        return self.rapid / 60.0

    def to_machine(self, p):
        """normalized [0,1] coords -> machine coords (mm), per-axis."""
        import numpy as np
        return np.asarray(p, dtype=np.float64) * self.workspace_vec

    def to_unit(self, p):
        """machine coords (mm) -> normalized [0,1] coords, per-axis."""
        import numpy as np
        return np.asarray(p, dtype=np.float64) / self.workspace_vec
