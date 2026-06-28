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
    # The trajectory lives in the normalized box [0,1]^3, and that box is the
    # STOCK bounding box (only the stock is voxelized -- RAM scales with the
    # part). ``stock_size_in``/``stock_size_mm`` give the stock box (REQUIRED for
    # export); ``stock_size_vec`` resolves either to a (3,) mm vector used to map
    # normalized coords to physical millimetres.
    #
    # The machine WORK VOLUME (toolhead limits, e.g. the Haas Mini Mill 16x12x10
    # in) is separate metadata: ``workspace_mm`` is the legacy scalar edge length
    # (cube), or set ``workspace_in`` to a per-axis (x, y, z) tuple in inches.
    # ``workspace_vec`` resolves either to a (3,) mm vector.
    #
    # The work origin (G54 offset) is the stock's TOP-CENTRE in machine coords;
    # set ``stock_origin_in`` (inches) to place the stock in the envelope.
    workspace_mm: float = 100.0   # work-volume edge length (cube), mm
    workspace_in: Optional[Tuple[float, float, float]] = None  # work volume, inches
    stock_size_in: Optional[Tuple[float, float, float]] = None  # stock box, inches
    stock_size_mm: Optional[Tuple[float, float, float]] = None  # stock box, mm (scalar ok)
    stock_origin_in: Optional[Tuple[float, float, float]] = None  # G54 top-centre, inches

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
        """Machine work volume as a (3,) mm vector (resolves scalar or inch form)."""
        import numpy as np
        from .units import inch_to_mm
        if self.workspace_in is not None:
            return np.asarray([inch_to_mm(c) for c in self.workspace_in], dtype=np.float64)
        return np.asarray([self.workspace_mm] * 3, dtype=np.float64)

    @property
    def stock_size_vec(self):
        """Stock box as a (3,) mm vector. The normalized cube [0,1]^3 spans it."""
        import numpy as np
        from .units import inch_to_mm
        if self.stock_size_mm is not None:
            v = self.stock_size_mm
            if np.isscalar(v):
                return np.asarray([float(v)] * 3, dtype=np.float64)
            return np.asarray([float(c) for c in v], dtype=np.float64)
        if self.stock_size_in is not None:
            return np.asarray([inch_to_mm(c) for c in self.stock_size_in], dtype=np.float64)
        raise ValueError(
            "stock_size_in (or stock_size_mm) must be set on MachineConfig; the "
            "normalized cube [0,1]^3 is the stock box, not the work volume"
        )

    @property
    def stock_origin_vec(self):
        """Work origin (G54) = stock TOP-CENTRE in machine coords, (3,) mm.

        Defaults to zeros when unset (machine origin == work origin)."""
        import numpy as np
        from .units import inch_to_mm
        if self.stock_origin_in is not None:
            return np.asarray([inch_to_mm(c) for c in self.stock_origin_in], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    @property
    def units_code(self) -> str:
        """Modal G-code for the configured units."""
        return "G21" if self.units == "mm" else "G20"

    @property
    def safe_z_mm(self) -> float:
        """Clearance Z in the work coordinate system (mm): ``retract_mm`` above the
        stock top face. With the top-centre G54, Z=0 is the stock top, so the
        retract plane is simply ``+retract_mm``."""
        return float(self.retract_mm)

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

    # The normalized cube is the STOCK box, with the work origin (G54) at the
    # stock's top-centre: normalized (0.5, 0.5, 1.0) maps to WCS (0, 0, 0).
    _WCS_REF = (0.5, 0.5, 1.0)

    def to_wcs(self, p):
        """normalized stock coords [0,1] -> work coordinate system (mm).

        Top-centre G54: X/Y are relative to the stock's XY centre, Z relative to
        the stock's top face (Z=0 at the top, negative down into the stock)."""
        import numpy as np
        ref = np.asarray(self._WCS_REF, dtype=np.float64)
        return (np.asarray(p, dtype=np.float64) - ref) * self.stock_size_vec

    def wcs_to_unit(self, p_wcs):
        """work coordinate system (mm) -> normalized stock coords [0,1]."""
        import numpy as np
        ref = np.asarray(self._WCS_REF, dtype=np.float64)
        return np.asarray(p_wcs, dtype=np.float64) / self.stock_size_vec + ref

    def to_machine(self, p):
        """normalized stock coords [0,1] -> absolute machine coords (mm)."""
        return self.stock_origin_vec + self.to_wcs(p)

    def to_unit(self, p):
        """absolute machine coords (mm) -> normalized stock coords [0,1]."""
        import numpy as np
        return self.wcs_to_unit(np.asarray(p, dtype=np.float64) - self.stock_origin_vec)
