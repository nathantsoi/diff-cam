"""Machine / CAM configuration.

A single dataclass holds every parameter that the G-code export, parser, and
trajectory planner must agree on. Most importantly it owns ``workspace_mm``, the
physical edge length (mm) of the simulator's unit cube ``[0, 1]^3``. Export
multiplies unit-cube coordinates by this scale; the parser divides by it. Keeping
the scale in one place is what makes the round trip
``trajectory -> G-code -> trajectory`` reproduce the original path exactly.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineConfig:
    # --- Coordinate mapping ---------------------------------------------------
    workspace_mm: float = 100.0   # physical edge length of the unit cube [0,1]^3

    # --- Feeds & speeds -------------------------------------------------------
    feed: float = 600.0           # cutting feed rate, mm/min (G1)
    rapid: float = 3000.0         # rapid traverse rate, mm/min (G0)
    max_accel: float = 500.0      # acceleration limit, mm/s^2

    # --- Planner --------------------------------------------------------------
    dt: float = 0.01              # planner sample period, seconds

    # --- Formatting -----------------------------------------------------------
    units: str = "mm"             # "mm" -> G21, "inch" -> G20
    precision: int = 6            # decimal places for coordinate words

    @property
    def units_code(self) -> str:
        """Modal G-code for the configured units."""
        return "G21" if self.units == "mm" else "G20"

    @property
    def feed_mm_per_s(self) -> float:
        return self.feed / 60.0

    @property
    def rapid_mm_per_s(self) -> float:
        return self.rapid / 60.0

    def to_machine(self, p):
        """unit-cube coords -> machine coords (mm)."""
        import numpy as np
        return np.asarray(p, dtype=np.float64) * self.workspace_mm

    def to_unit(self, p):
        """machine coords (mm) -> unit-cube coords."""
        import numpy as np
        return np.asarray(p, dtype=np.float64) / self.workspace_mm
