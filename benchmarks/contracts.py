"""DiffCAM shared contracts, schema ``diffcam-contracts/1`` (T0.0).

Typed, validated, JSON-serializable records shared by every task card:
``TaskSpec``, ``MotionPlan``, ``EvaluationResult``, ``CandidateResult`` and
``RunRecord``, plus the evaluator / parameterization / artifact-writer
interfaces. Semantics are documented in ``docs/contracts.md``; this module is
the executable form. Changing a field's meaning requires a schema version bump
and a coordinator-approved handoff.

Conventions (all enforced or documented here):

* Lengths are millimetres, times seconds, speeds mm/s, volumes mm^3. Inch
  values are converted at the I/O boundary (``cam.units``) and never stored.
* Frames: ``work`` is the G54 frame (origin at stock top-centre, Z up);
  ``machine`` is the machine frame. ``TaskSpec.work_to_machine`` is a rigid
  4x4 homogeneous transform mapping work coordinates to machine coordinates.
* Tool reference point is the cutter bottom-centre (tip); the tool extends +Z.
* Grids are cell-centred: cell (i, j, k) has centre
  ``origin_mm + (index + 0.5) * voxel_mm`` in the grid's frame; voxels are cubic.
  Signed fields are negative inside the solid.
* Membership: material removal uses the closed tool set (signed distance
  <= 0); collision uses strict interior overlap (< 0). Touching is not a collision.
* Hashes are ``sha256:<hex>`` over canonical JSON (sorted keys, no whitespace,
  shortest round-trip floats). NC hashes are over raw file bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

SCHEMA_VERSION = "diffcam-contracts/1.1"
LENGTH_UNIT = "mm"

REMOVAL_MEMBERSHIP = "closed"     # phi <= 0 removes material
COLLISION_MEMBERSHIP = "strict"   # phi < 0 overlaps count as collision


class ContractError(ValueError):
    """Invalid or inconsistent contract data. Messages name the offending field."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Frame(str, Enum):
    WORK = "work"
    MACHINE = "machine"


class Split(str, Enum):
    DEV = "dev"
    TEST = "test"
    EXTERNAL = "external"


class SegmentKind(str, Enum):
    RAPID = "rapid"   # G0: never removes material; checks cutter + assembly clearance
    FEED = "feed"     # G1 linear interpolation at feed_mm_s
    ARC = "arc"       # G2/G3 circular (helical) interpolation at feed_mm_s


class ArcPlane(str, Enum):
    XY = "XY"  # G17
    XZ = "XZ"  # G18
    YZ = "YZ"  # G19


class ArcDirection(str, Enum):
    CW = "cw"    # G2, viewed from the plane's positive normal
    CCW = "ccw"  # G3


class TimingRule(str, Enum):
    # Each segment runs at its declared constant speed with exact stop at both
    # endpoints (G61-like); duration_s = path length / speed.
    EXACT_STOP_CONSTANT_SPEED = "exact_stop_constant_speed"
    # Segments are sampled at a fixed controller period ``dt_s``; duration_s is
    # stated per segment and must be a positive multiple of dt_s.
    FIXED_DT = "fixed_dt"


class CutterKind(str, Enum):
    FLAT = "flat"
    BALL = "ball"


class Coverage(str, Enum):
    COMPLETE = "complete"          # every geometry-affecting element modeled at declared resolution
    INCOMPLETE = "incomplete"      # some elements modeled; reasons list the gaps
    UNSUPPORTED = "unsupported"    # input outside the evaluator's declared scope
    UNKNOWN = "unknown"            # evaluation did not establish coverage


class RunStatus(str, Enum):
    COMPLETE = "complete"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    INTERRUPTED = "interrupted"
    BLOCKED = "blocked"            # a required dependency/tool is unavailable


class Dialect(str, Enum):
    RS274 = "rs274"
    HAAS = "haas"


class PostStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"    # dialect cannot represent the motion; no NC/result written
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Helpers: validation, canonical JSON, hashing
# ---------------------------------------------------------------------------
def _finite(name: str, x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ContractError(f"{name}: expected a number, got {x!r}") from None
    if not math.isfinite(v):
        raise ContractError(f"{name}: must be finite, got {v!r}")
    return v


def _positive(name: str, x: Any, allow_zero: bool = False) -> float:
    v = _finite(name, x)
    if v < 0 or (v == 0 and not allow_zero):
        raise ContractError(f"{name}: must be {'>= 0' if allow_zero else '> 0'} (mm/s/mm^3 units), got {v}")
    return v


def _vec3(name: str, x: Any) -> Tuple[float, float, float]:
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise ContractError(f"{name}: expected a length-3 vector in mm, got {x!r}")
    return tuple(_finite(f"{name}[{i}]", c) for i, c in enumerate(x))  # type: ignore[return-value]


def _enum(cls, name: str, x: Any):
    try:
        return cls(x)
    except ValueError:
        raise ContractError(f"{name}: {x!r} is not one of {[e.value for e in cls]}") from None


def _require_mm(name: str, unit: str) -> None:
    if unit != LENGTH_UNIT:
        raise ContractError(
            f"{name}: length unit must be 'mm' (got {unit!r}); convert inches with cam.units.inch_to_mm "
            "at the I/O boundary before constructing contract records"
        )


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ContractError(f"non-finite float {obj!r} cannot be serialized; use null with a reason")
        return obj
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing (sorted keys, compact, round-trip floats)."""
    return json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def nc_hash(nc_bytes: bytes) -> str:
    """Hash of a posted NC program over its exact bytes (no newline normalisation)."""
    return sha256_bytes(nc_bytes)


def _check_hash(name: str, h: Optional[str], allow_none: bool = False) -> Optional[str]:
    if h is None and allow_none:
        return None
    if not isinstance(h, str) or not h.startswith("sha256:") or len(h) != 71:
        raise ContractError(f"{name}: expected 'sha256:<64 hex>', got {h!r}")
    try:
        int(h[7:], 16)
    except ValueError:
        raise ContractError(f"{name}: non-hex digest {h!r}") from None
    return h


# ---------------------------------------------------------------------------
# Geometry and setup
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Transform:
    """Rigid 4x4 homogeneous transform (row-major), mm."""

    matrix: Tuple[Tuple[float, ...], ...]

    def __post_init__(self):
        m = self.matrix
        if not isinstance(m, (list, tuple)) or len(m) != 4 or any(len(r) != 4 for r in m):
            raise ContractError("work_to_machine.matrix: expected 4x4 nested list")
        m = tuple(tuple(_finite(f"work_to_machine.matrix[{i}][{j}]", v) for j, v in enumerate(r)) for i, r in enumerate(m))
        object.__setattr__(self, "matrix", m)
        if any(abs(a - b) > 1e-12 for a, b in zip(m[3], (0.0, 0.0, 0.0, 1.0))):
            raise ContractError("work_to_machine.matrix: last row must be [0,0,0,1]")
        R = [r[:3] for r in m[:3]]
        for i in range(3):
            for j in range(3):
                dot = sum(R[k][i] * R[k][j] for k in range(3))
                if abs(dot - (1.0 if i == j else 0.0)) > 1e-9:
                    raise ContractError("work_to_machine.matrix: rotation block must be orthonormal (rigid, no scaling)")
        det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
               - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
               + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
        if det < 0:
            raise ContractError("work_to_machine.matrix: rotation must be proper (det=+1), not a reflection")

    @classmethod
    def translation(cls, offset_mm: Sequence[float]) -> "Transform":
        x, y, z = _vec3("translation", list(offset_mm))
        return cls(((1.0, 0.0, 0.0, x), (0.0, 1.0, 0.0, y), (0.0, 0.0, 1.0, z), (0.0, 0.0, 0.0, 1.0)))

    def apply(self, p: Sequence[float]) -> Tuple[float, float, float]:
        m = self.matrix
        return tuple(m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3] for i in range(3))  # type: ignore[return-value]

    def inverse(self) -> "Transform":
        m = self.matrix
        Rt = [[m[j][i] for j in range(3)] for i in range(3)]
        t = [-sum(Rt[i][k] * m[k][3] for k in range(3)) for i in range(3)]
        return Transform(tuple(tuple(Rt[i]) + (t[i],) for i in range(3)) + ((0.0, 0.0, 0.0, 1.0),))


@dataclass(frozen=True)
class GridSpec:
    """Cell-centred cubic-voxel grid. ``origin_mm`` is the min corner of cell (0,0,0)."""

    shape: Tuple[int, int, int]
    origin_mm: Tuple[float, float, float]
    voxel_mm: float
    frame: Frame = Frame.WORK
    sign_convention: str = "negative_inside"

    def __post_init__(self):
        if not isinstance(self.shape, (list, tuple)) or len(self.shape) != 3:
            raise ContractError("grid.shape: expected (Nx, Ny, Nz)")
        shape = tuple(int(n) for n in self.shape)
        if any(n < 1 for n in shape) or any(int(n) != n for n in self.shape):
            raise ContractError(f"grid.shape: positive integers required, got {self.shape!r}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "origin_mm", _vec3("grid.origin_mm", list(self.origin_mm)))
        object.__setattr__(self, "voxel_mm", _positive("grid.voxel_mm", self.voxel_mm))
        object.__setattr__(self, "frame", _enum(Frame, "grid.frame", self.frame))
        if self.sign_convention != "negative_inside":
            raise ContractError("grid.sign_convention: only 'negative_inside' is supported")

    @property
    def extent_mm(self) -> Tuple[float, float, float]:
        return tuple(n * self.voxel_mm for n in self.shape)  # type: ignore[return-value]

    @property
    def bounds_mm(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        lo = self.origin_mm
        return lo, tuple(lo[i] + self.extent_mm[i] for i in range(3))  # type: ignore[return-value]

    def cell_centre_mm(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        return tuple(self.origin_mm[a] + (idx + 0.5) * self.voxel_mm for a, idx in enumerate((i, j, k)))  # type: ignore[return-value]


@dataclass(frozen=True)
class GeometrySource:
    """A solid. ``kind='analytic'``: ``params`` define a primitive (e.g. box, box_minus_pocket);
    ``kind='sampled'``: signed field file ``path`` with ``sha256`` on ``grid``;
    ``kind='mesh'``/``'step'``: external file with ``sha256`` (and the converter in params)."""

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None
    sha256: Optional[str] = None
    grid: Optional[GridSpec] = None
    frame: Frame = Frame.WORK

    def __post_init__(self):
        if self.kind not in ("analytic", "sampled", "mesh", "step"):
            raise ContractError(f"geometry.kind: {self.kind!r} not in analytic/sampled/mesh/step")
        object.__setattr__(self, "frame", _enum(Frame, "geometry.frame", self.frame))
        if self.kind == "analytic":
            if not self.params or "primitive" not in self.params:
                raise ContractError("geometry.params: analytic geometry requires params.primitive")
        else:
            if not self.path:
                raise ContractError(f"geometry.path: required for kind={self.kind!r}")
            _check_hash("geometry.sha256", self.sha256)
        if self.kind == "sampled" and self.grid is None:
            raise ContractError("geometry.grid: sampled fields must declare bounds/spacing (GridSpec)")
        if isinstance(self.grid, Mapping):
            object.__setattr__(self, "grid", GridSpec(**self.grid))


@dataclass(frozen=True)
class ToolAssembly:
    """Tool referenced at the cutter bottom-centre, extending +Z.

    Cutting body: radius ``cutter_radius_mm`` up to ``flute_length_mm``. Non-cutting
    shank from flute_length_mm to ``holder_bottom_mm`` (radius ``shank_radius_mm``).
    Holder: cylinder radius ``holder_radius_mm`` from ``holder_bottom_mm`` for
    ``holder_height_mm`` (all measured from the tip). Ball cutters are declared but a
    consumer that only models flat cutters must report coverage ``unsupported``.
    """

    cutter_kind: CutterKind
    cutter_radius_mm: float
    flute_length_mm: float
    shank_radius_mm: float
    holder_bottom_mm: float
    holder_radius_mm: float
    holder_height_mm: float
    tool_number: int = 1
    reference_point: str = "cutter_bottom_centre"

    def __post_init__(self):
        object.__setattr__(self, "cutter_kind", _enum(CutterKind, "tool.cutter_kind", self.cutter_kind))
        for n in ("cutter_radius_mm", "flute_length_mm", "shank_radius_mm", "holder_bottom_mm",
                  "holder_radius_mm", "holder_height_mm"):
            object.__setattr__(self, n, _positive(f"tool.{n}", getattr(self, n)))
        if self.reference_point != "cutter_bottom_centre":
            raise ContractError("tool.reference_point: only 'cutter_bottom_centre' is supported")
        if self.holder_bottom_mm < self.flute_length_mm:
            raise ContractError("tool.holder_bottom_mm: holder cannot start below the end of the flutes")
        if self.cutter_kind == CutterKind.BALL and self.flute_length_mm < self.cutter_radius_mm:
            raise ContractError("tool.flute_length_mm: ball cutter flute length must cover its radius")

    @property
    def stickout_mm(self) -> float:
        return self.holder_bottom_mm


@dataclass(frozen=True)
class MachineState:
    """Controller state before the first motion segment. ``known=False`` means the
    start pose is not established (e.g. after a tool change without a defined
    position); evaluators must then report coverage != complete for the approach."""

    position_mm: Optional[Tuple[float, float, float]]
    frame: Frame = Frame.MACHINE
    known: bool = True
    length_offset_applied: bool = True   # G43 active: position refers to the tool tip
    work_offset: str = "G54"

    def __post_init__(self):
        object.__setattr__(self, "frame", _enum(Frame, "initial_state.frame", self.frame))
        if self.known:
            if self.position_mm is None:
                raise ContractError("initial_state.position_mm: required when known=True")
            object.__setattr__(self, "position_mm", _vec3("initial_state.position_mm", list(self.position_mm)))
        elif self.position_mm is not None:
            raise ContractError("initial_state.position_mm: must be null when known=False (do not guess)")


@dataclass(frozen=True)
class MachineLimits:
    max_feed_mm_s: float
    rapid_mm_s: float
    max_accel_mm_s2: float
    envelope_min_mm: Tuple[float, float, float]
    envelope_max_mm: Tuple[float, float, float]   # machine frame

    def __post_init__(self):
        for n in ("max_feed_mm_s", "rapid_mm_s", "max_accel_mm_s2"):
            object.__setattr__(self, n, _positive(f"limits.{n}", getattr(self, n)))
        lo = _vec3("limits.envelope_min_mm", list(self.envelope_min_mm))
        hi = _vec3("limits.envelope_max_mm", list(self.envelope_max_mm))
        if any(a >= b for a, b in zip(lo, hi)):
            raise ContractError("limits.envelope: min must be < max on every axis")
        object.__setattr__(self, "envelope_min_mm", lo)
        object.__setattr__(self, "envelope_max_mm", hi)


@dataclass(frozen=True)
class Tolerances:
    """Absolute, scale-aware tolerances (no universal percentages).

    ``completion_residual_mm3``: max residual (unremoved) operation-target volume.
    ``gouge_volume_mm3``: max removed volume inside the part. ``local_surface_error_mm``:
    max local deviation on the operation-target surface. ``clearance_mm``: min
    clearance of the NON-CUTTING assembly (shank + holder) to remaining stock and fixtures
    during rapids and feeds. ``rapid_penetration_mm`` (v1.1): during rapids the cutter may
    graze remaining stock (e.g. retracting out of its own slot); a rapid is a collision only
    if the cutter penetrates deeper than this, and safety is certified only if the evaluator's
    upper bound on possible rapid penetration is <= this value. 0 means any possible
    interior overlap is unsafe/unknown.
    """

    completion_residual_mm3: float
    gouge_volume_mm3: float
    local_surface_error_mm: float
    clearance_mm: float
    rapid_penetration_mm: float = 0.0

    def __post_init__(self):
        for f_ in dataclasses.fields(self):
            object.__setattr__(self, f_.name, _positive(f"tolerances.{f_.name}", getattr(self, f_.name), allow_zero=True))


@dataclass(frozen=True)
class SourceInfo:
    origin: str                          # "procedural:<generator>@<version>", "titan:<id>", ...
    license: str                         # SPDX id, "proprietary", or "unknown"
    checksums: Dict[str, str] = field(default_factory=dict)  # file -> sha256:...
    notes: str = ""

    def __post_init__(self):
        if not self.origin:
            raise ContractError("source.origin: required")
        for k, v in self.checksums.items():
            _check_hash(f"source.checksums[{k}]", v)


@dataclass(frozen=True)
class OperationTarget:
    """Reachable target for this operation, separate from the original part.
    ``excluded_volume_mm3`` and ``exclusion_reason`` are fixed before optimization."""

    geometry: GeometrySource
    excluded_volume_mm3: float = 0.0
    exclusion_reason: str = ""

    def __post_init__(self):
        if isinstance(self.geometry, Mapping):
            object.__setattr__(self, "geometry", GeometrySource(**self.geometry))
        v = _positive("operation_target.excluded_volume_mm3", self.excluded_volume_mm3, allow_zero=True)
        object.__setattr__(self, "excluded_volume_mm3", v)
        if v > 0 and not self.exclusion_reason:
            raise ContractError("operation_target.exclusion_reason: required when excluded_volume_mm3 > 0")


@dataclass(frozen=True)
class MotionSemantics:
    dialects: Tuple[Dialect, ...] = (Dialect.RS274, Dialect.HAAS)
    timing_rule: TimingRule = TimingRule.EXACT_STOP_CONSTANT_SPEED
    arcs_supported: bool = True
    supported_gcodes: Tuple[str, ...] = ("G0", "G1", "G2", "G3", "G17", "G18", "G19", "G20", "G21",
                                         "G43", "G49", "G53", "G54", "G90", "G91", "G94")

    def __post_init__(self):
        object.__setattr__(self, "dialects", tuple(_enum(Dialect, "motion_semantics.dialects", d) for d in self.dialects))
        object.__setattr__(self, "timing_rule", _enum(TimingRule, "motion_semantics.timing_rule", self.timing_rule))
        object.__setattr__(self, "supported_gcodes", tuple(self.supported_gcodes))


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    split: Split
    source: SourceInfo
    length_unit: str
    stock: GeometrySource
    original_part: GeometrySource
    operation_target: OperationTarget
    grid: GridSpec
    work_to_machine: Transform
    initial_state: MachineState
    tool: ToolAssembly
    limits: MachineLimits
    tolerances: Tolerances
    motion_semantics: MotionSemantics = field(default_factory=MotionSemantics)
    fixtures: Tuple[GeometrySource, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}, got {self.schema_version!r}")
        if not self.task_id or not self.family:
            raise ContractError("task_id/family: required non-empty strings")
        _require_mm("task.length_unit", self.length_unit)
        object.__setattr__(self, "split", _enum(Split, "task.split", self.split))
        coerce = {"source": SourceInfo, "stock": GeometrySource, "original_part": GeometrySource,
                  "operation_target": OperationTarget, "grid": GridSpec, "work_to_machine": Transform,
                  "initial_state": MachineState, "tool": ToolAssembly, "limits": MachineLimits,
                  "tolerances": Tolerances, "motion_semantics": MotionSemantics}
        for name, cls in coerce.items():
            v = getattr(self, name)
            if isinstance(v, Mapping):
                object.__setattr__(self, name, cls(**v))
        object.__setattr__(self, "fixtures", tuple(GeometrySource(**f_) if isinstance(f_, Mapping) else f_
                                                   for f_ in self.fixtures))
        if self.grid.frame != Frame.WORK:
            raise ContractError("task.grid.frame: the task grid is defined in the work frame")

    # -- hashes --------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TaskSpec":
        return cls(**_strip_unknown(cls, d))

    @property
    def task_hash(self) -> str:
        return content_hash(self)

    @property
    def tool_setup_hash(self) -> str:
        return content_hash({"tool": self.tool, "work_to_machine": self.work_to_machine,
                             "fixtures": self.fixtures, "initial_state": self.initial_state})


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArcData:
    centre_mm: Tuple[float, float, float]
    plane: ArcPlane
    direction: ArcDirection
    turns: int = 0   # additional full turns beyond the first sweep (G2/G3 P word - 1)

    def __post_init__(self):
        object.__setattr__(self, "centre_mm", _vec3("arc.centre_mm", list(self.centre_mm)))
        object.__setattr__(self, "plane", _enum(ArcPlane, "arc.plane", self.plane))
        object.__setattr__(self, "direction", _enum(ArcDirection, "arc.direction", self.direction))
        if int(self.turns) != self.turns or self.turns < 0:
            raise ContractError("arc.turns: non-negative integer required")


@dataclass(frozen=True)
class Segment:
    """One motion element. ``feed_mm_s`` is required for feed/arc and must be null for
    rapids (rapids move at ``TaskSpec.limits.rapid_mm_s``). ``duration_s`` is the
    executed duration under the plan's timing rule; zero-length feeds have
    duration 0 and still occupy one stationary tool volume (they can cut)."""

    id: str
    kind: SegmentKind
    start_mm: Tuple[float, float, float]
    end_mm: Tuple[float, float, float]
    duration_s: float
    feed_mm_s: Optional[float] = None
    arc: Optional[ArcData] = None
    tag: str = ""          # free-form provenance: "plunge", "ramp", "retract", "approach", ...
    source_line: Optional[int] = None

    def __post_init__(self):
        if not self.id:
            raise ContractError("segment.id: required (stable ids are referenced by evaluation events)")
        object.__setattr__(self, "kind", _enum(SegmentKind, f"segment[{self.id}].kind", self.kind))
        object.__setattr__(self, "start_mm", _vec3(f"segment[{self.id}].start_mm", list(self.start_mm)))
        object.__setattr__(self, "end_mm", _vec3(f"segment[{self.id}].end_mm", list(self.end_mm)))
        object.__setattr__(self, "duration_s", _positive(f"segment[{self.id}].duration_s", self.duration_s, allow_zero=True))
        if isinstance(self.arc, Mapping):
            object.__setattr__(self, "arc", ArcData(**self.arc))
        if self.kind == SegmentKind.RAPID:
            if self.feed_mm_s is not None:
                raise ContractError(f"segment[{self.id}].feed_mm_s: rapids use limits.rapid_mm_s; set null")
        else:
            object.__setattr__(self, "feed_mm_s", _positive(f"segment[{self.id}].feed_mm_s", self.feed_mm_s))
        if (self.kind == SegmentKind.ARC) != (self.arc is not None):
            raise ContractError(f"segment[{self.id}].arc: arc data required iff kind == 'arc'")

    @property
    def chord_length_mm(self) -> float:
        return math.dist(self.start_mm, self.end_mm)

    @property
    def path_length_mm(self) -> float:
        """Exact path length: chord for rapid/feed, helical arc length for arcs."""
        if self.arc is None:
            return self.chord_length_mm
        return arc_geometry(self.start_mm, self.end_mm, self.arc)["length_mm"]


_PLANE_AXES = {ArcPlane.XY: (0, 1, 2), ArcPlane.XZ: (2, 0, 1), ArcPlane.YZ: (1, 2, 0)}
ARC_RADIUS_TOL_MM = 1e-3


def arc_geometry(start, end, arc: ArcData, segment_id: str = "?") -> Dict[str, float]:
    """RS274 arc semantics in the motion frame. Plane axes (a, b, normal): XY=(X,Y,Z) G17,
    XZ=(Z,X,Y) G18, YZ=(Y,Z,X) G19; CCW is positive rotation about the plane normal.
    Start == end in-plane means a full circle. ``turns`` adds full revolutions. Linear motion
    along the normal makes a helix. Returns radius, signed sweep (rad) and length (mm)."""
    a, b, n = _PLANE_AXES[arc.plane]
    c = arc.centre_mm
    r0 = math.hypot(start[a] - c[a], start[b] - c[b])
    r1 = math.hypot(end[a] - c[a], end[b] - c[b])
    if r0 <= ARC_RADIUS_TOL_MM:
        raise ContractError(f"segment[{segment_id}].arc: zero radius")
    if abs(r0 - r1) > ARC_RADIUS_TOL_MM:
        raise ContractError(f"segment[{segment_id}].arc: start/end radii differ ({r0} vs {r1} mm, tol {ARC_RADIUS_TOL_MM})")
    th0 = math.atan2(start[b] - c[b], start[a] - c[a])
    th1 = math.atan2(end[b] - c[b], end[a] - c[a])
    ccw = (th1 - th0) % (2 * math.pi)
    if arc.direction == ArcDirection.CCW:
        sweep = ccw if ccw > 1e-12 else 2 * math.pi
        sweep += 2 * math.pi * arc.turns
    else:
        cw = (2 * math.pi - ccw) % (2 * math.pi)
        sweep = -((cw if cw > 1e-12 else 2 * math.pi) + 2 * math.pi * arc.turns)
    r = 0.5 * (r0 + r1)
    dn = end[n] - start[n]
    return {"radius_mm": r, "sweep_rad": sweep, "length_mm": math.hypot(r * sweep, dn)}


def _check_duration(s: "Segment", expect: float, what: str) -> None:
    if abs(expect - s.duration_s) > 1e-6 * max(1.0, expect):
        raise ContractError(f"segment[{s.id}].duration_s: {s.duration_s} != {what} = {expect}")


@dataclass(frozen=True)
class MotionPlan:
    """Typed executed motion. A numerical trajectory array is an export of this, not a substitute.

    Contiguity: segment i+1 starts where segment i ends (within ``CONTIGUITY_TOL_MM``), and
    the first segment starts at ``initial_state.position_mm`` when that state is known.
    """

    frame: Frame
    initial_state: MachineState
    segments: Tuple[Segment, ...]
    timing_rule: TimingRule
    dt_s: Optional[float] = None
    length_unit: str = LENGTH_UNIT
    provenance: str = ""     # "cam_seed:<generator>", "optimizer:<run_id>", "parsed:<nc sha256>"
    schema_version: str = SCHEMA_VERSION

    CONTIGUITY_TOL_MM = 1e-6

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}, got {self.schema_version!r}")
        _require_mm("motion.length_unit", self.length_unit)
        object.__setattr__(self, "frame", _enum(Frame, "motion.frame", self.frame))
        object.__setattr__(self, "timing_rule", _enum(TimingRule, "motion.timing_rule", self.timing_rule))
        if isinstance(self.initial_state, Mapping):
            object.__setattr__(self, "initial_state", MachineState(**self.initial_state))
        segs = tuple(Segment(**s) if isinstance(s, Mapping) else s for s in self.segments)
        object.__setattr__(self, "segments", segs)
        if self.initial_state.known and self.initial_state.frame != self.frame:
            raise ContractError("motion.initial_state.frame: must match motion.frame")
        ids = [s.id for s in segs]
        if len(set(ids)) != len(ids):
            raise ContractError("motion.segments: duplicate segment ids")
        if self.timing_rule == TimingRule.FIXED_DT:
            dt = _positive("motion.dt_s", self.dt_s)
            for s in segs:
                n = s.duration_s / dt
                if abs(n - round(n)) > 1e-6:
                    raise ContractError(f"segment[{s.id}].duration_s: must be a multiple of dt_s={dt} under fixed_dt")
        elif self.dt_s is not None:
            raise ContractError("motion.dt_s: only meaningful for timing_rule='fixed_dt'")
        prev = self.initial_state.position_mm if self.initial_state.known else None
        for s in segs:
            if prev is not None and math.dist(prev, s.start_mm) > self.CONTIGUITY_TOL_MM:
                raise ContractError(f"segment[{s.id}].start_mm: discontinuous with previous end {prev} "
                                    "(insert an explicit rapid/feed segment)")
            prev = s.end_mm
        for s in segs:
            if s.arc is not None:
                arc_geometry(s.start_mm, s.end_mm, s.arc, segment_id=s.id)   # radius consistency
        if self.timing_rule == TimingRule.EXACT_STOP_CONSTANT_SPEED:
            for s in segs:
                if s.kind != SegmentKind.RAPID:   # rapids need the task's rapid rate: validate_for_task
                    _check_duration(s, s.path_length_mm / s.feed_mm_s, "path_length/feed")  # type: ignore[operator]

    def validate_for_task(self, task: "TaskSpec") -> None:
        """Task-aware checks every producer must pass before evaluation: rapid durations
        (exact-stop: length / limits.rapid_mm_s), feed limits, and initial-state agreement."""
        if self.initial_state != task.initial_state:
            raise ContractError("motion.initial_state: differs from task.initial_state")
        for s in self.segments:
            if s.feed_mm_s is not None and s.feed_mm_s > task.limits.max_feed_mm_s * (1 + 1e-9):
                raise ContractError(f"segment[{s.id}].feed_mm_s: {s.feed_mm_s} exceeds limits.max_feed_mm_s "
                                    f"{task.limits.max_feed_mm_s}")
            if self.timing_rule == TimingRule.EXACT_STOP_CONSTANT_SPEED and s.kind == SegmentKind.RAPID:
                _check_duration(s, s.path_length_mm / task.limits.rapid_mm_s, "length/limits.rapid_mm_s")

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MotionPlan":
        return cls(**_strip_unknown(cls, d))

    @property
    def motion_hash(self) -> str:
        return content_hash(self)

    def waypoints_mm(self) -> List[Tuple[float, float, float]]:
        """Export: start of the first segment followed by each segment end (arcs are NOT
        resolved; use the typed segments for geometry)."""
        if not self.segments:
            return [self.initial_state.position_mm] if self.initial_state.known else []  # type: ignore[list-item]
        return [self.segments[0].start_mm] + [s.end_mm for s in self.segments]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# Known metric names -> required unit. Unknown names are allowed but must carry a unit.
METRIC_UNITS: Dict[str, str] = {
    "iou": "1", "dice": "1",
    "removed_volume_mm3": "mm^3",
    "residual_volume_mm3": "mm^3",            # unremoved operation-target volume
    "gouge_volume_mm3": "mm^3",               # removed volume inside the part
    "max_surface_error_mm": "mm", "p99_surface_error_mm": "mm",
    "asd_mm": "mm", "hd95_mm": "mm",
    "min_clearance_mm": "mm",                 # non-cutting assembly (shank + holder) only
    "min_rapid_cutter_clearance_mm": "mm",    # cutter during rapids; negative = penetration depth bound
    "unique_contacted_volume_mm3": "mm^3",    # union of collided material
    "collision_sample_count": "count",        # depends on sampling; never compare across settings
    "time_integrated_overlap_mm3_s": "mm^3*s",
    "cycle_time_s": "s", "air_fraction": "1",
    "spatial_uncertainty_mm3": "mm^3", "temporal_uncertainty_mm3": "mm^3",
    "force_proxy": "proxy",                   # uncalibrated; never ranks or qualifies a program
}


@dataclass(frozen=True)
class Metric:
    """``value=None`` means unknown/not computed and requires ``null_reason``; 0 is a real zero."""

    value: Optional[float]
    unit: str
    denominator: Optional[str] = None   # e.g. "operation_target_volume_mm3" for fractions
    null_reason: Optional[str] = None

    def __post_init__(self):
        if self.value is None:
            if not self.null_reason:
                raise ContractError("metric.null_reason: required when value is null (zero must be stated as 0)")
        else:
            object.__setattr__(self, "value", _finite("metric.value", self.value))
            if self.null_reason:
                raise ContractError("metric.null_reason: only allowed when value is null")
        if not self.unit:
            raise ContractError("metric.unit: required")


@dataclass(frozen=True)
class SegmentEvent:
    segment_id: str
    kind: str              # "holder_collision", "rapid_collision", "shank_collision", "fixture_collision", ...
    s: Optional[float]     # normalized position along the segment in [0, 1], null if not localised
    detail: str = ""

    def __post_init__(self):
        if self.s is not None:
            v = _finite("event.s", self.s)
            if not 0.0 <= v <= 1.0:
                raise ContractError("event.s: must be in [0, 1]")


@dataclass(frozen=True)
class EvaluationResult:
    evaluator_id: str            # e.g. "posted_hard/production", "reference_carve", "camotics"
    evaluator_revision: str      # code commit (+dirty hash) of the evaluator
    task_hash: str
    tool_setup_hash: str
    discretization: Dict[str, Any]   # spatial/temporal settings incl. reference settings
    coverage: Coverage
    coverage_reasons: Tuple[str, ...]
    metrics: Dict[str, Metric]
    safety_pass: Optional[bool]      # None = unknown
    geometry_pass: Optional[bool]
    nc_hash: Optional[str] = None    # set for posted-program evaluation
    dialect: Optional[Dialect] = None
    motion_hash: Optional[str] = None
    events: Tuple[SegmentEvent, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}, got {self.schema_version!r}")
        if not self.evaluator_id or not self.evaluator_revision:
            raise ContractError("evaluator_id/evaluator_revision: required")
        _check_hash("task_hash", self.task_hash)
        _check_hash("tool_setup_hash", self.tool_setup_hash)
        _check_hash("nc_hash", self.nc_hash, allow_none=True)
        _check_hash("motion_hash", self.motion_hash, allow_none=True)
        object.__setattr__(self, "coverage", _enum(Coverage, "coverage", self.coverage))
        object.__setattr__(self, "coverage_reasons", tuple(self.coverage_reasons))
        if self.dialect is not None:
            object.__setattr__(self, "dialect", _enum(Dialect, "dialect", self.dialect))
        if (self.nc_hash is None) != (self.dialect is None):
            raise ContractError("nc_hash/dialect: a posted evaluation needs both; a motion evaluation neither")
        if self.nc_hash is None and self.motion_hash is None:
            raise ContractError("motion_hash: required when no NC program is evaluated")
        metrics = {k: Metric(**v) if isinstance(v, Mapping) else v for k, v in self.metrics.items()}
        for k, m in metrics.items():
            want = METRIC_UNITS.get(k)
            if want is not None and m.unit != want:
                raise ContractError(f"metrics[{k}].unit: expected {want!r}, got {m.unit!r}")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "events", tuple(SegmentEvent(**e) if isinstance(e, Mapping) else e for e in self.events))
        if self.coverage != Coverage.COMPLETE:
            if not self.coverage_reasons:
                raise ContractError("coverage_reasons: required when coverage is not complete")
            if self.safety_pass is True:
                raise ContractError("safety_pass: cannot be true without complete coverage (use null = unknown)")
        if self.safety_pass is not None and not isinstance(self.safety_pass, bool):
            raise ContractError("safety_pass: true/false/null")
        if self.geometry_pass is not None and not isinstance(self.geometry_pass, bool):
            raise ContractError("geometry_pass: true/false/null")

    @property
    def efficiency_eligible(self) -> bool:
        return self.coverage == Coverage.COMPLETE and self.safety_pass is True and self.geometry_pass is True

    def to_dict(self) -> Dict[str, Any]:
        d = _to_jsonable(self)
        d["efficiency_eligible"] = self.efficiency_eligible   # derived; ignored on load
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EvaluationResult":
        d = dict(d)
        stored = d.pop("efficiency_eligible", None)
        r = cls(**_strip_unknown(cls, d))
        if stored is not None and stored != r.efficiency_eligible:
            raise ContractError("efficiency_eligible: stored value disagrees with coverage/safety/geometry")
        return r


def geometry_pass_from_metrics(metrics: Mapping[str, Metric], tol: Tolerances) -> Optional[bool]:
    """Reference rule: completion, global gouge and local surface tolerance must all pass.
    Any required metric unknown -> None. A no-cut program fails completion on a removal task."""
    need = ("residual_volume_mm3", "gouge_volume_mm3", "max_surface_error_mm")
    vals = [metrics.get(n) for n in need]
    if any(m is None or m.value is None for m in vals):
        return None
    residual, gouge, surf = (m.value for m in vals)  # type: ignore[union-attr]
    return (residual <= tol.completion_residual_mm3 and gouge <= tol.gouge_volume_mm3
            and surf <= tol.local_surface_error_mm)


# ---------------------------------------------------------------------------
# Candidates and runs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateResult:
    """One stateless candidate evaluation (T3.0). ``objective=None`` iff invalid."""

    candidate_id: str
    theta_hash: str
    objective_kind: str          # "hard" | "smooth"
    objective: Optional[float]
    valid: bool
    invalid_reason: Optional[str] = None
    components: Dict[str, float] = field(default_factory=dict)
    gradient_available: bool = False
    wall_time_s: Optional[float] = None
    coverage: Coverage = Coverage.UNKNOWN

    def __post_init__(self):
        _check_hash("theta_hash", self.theta_hash)
        if self.objective_kind not in ("hard", "smooth"):
            raise ContractError("objective_kind: 'hard' or 'smooth'")
        object.__setattr__(self, "coverage", _enum(Coverage, "coverage", self.coverage))
        if self.valid:
            object.__setattr__(self, "objective", _finite("objective", self.objective))
            if self.invalid_reason:
                raise ContractError("invalid_reason: only for invalid candidates")
        else:
            if self.objective is not None or not self.invalid_reason:
                raise ContractError("invalid candidate: objective must be null and invalid_reason set")
        object.__setattr__(self, "components", {k: _finite(f"components[{k}]", v) for k, v in self.components.items()})


REQUIRED_RUN_ARTIFACTS = ("config.json", "history.json", "metrics.json", "initial_trajectory.npy",
                          "trajectory.npy", "trajectory_deltas.npy", "motion_plan.json", "eval_posted.json")
DIALECT_ARTIFACTS = {Dialect.RS274: ("rs274.nc", "eval_posted_rs274.json"),
                     Dialect.HAAS: ("haas.nc", "eval_posted_haas.json")}


@dataclass(frozen=True)
class PostedProgram:
    dialect: Dialect
    status: PostStatus
    nc_hash: Optional[str] = None
    eval_file: Optional[str] = None
    reason: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "dialect", _enum(Dialect, "posted.dialect", self.dialect))
        object.__setattr__(self, "status", _enum(PostStatus, "posted.status", self.status))
        if self.status == PostStatus.SUPPORTED:
            _check_hash("posted.nc_hash", self.nc_hash)
            if not self.eval_file:
                raise ContractError("posted.eval_file: required for a supported dialect")
        else:
            if self.nc_hash is not None:
                raise ContractError(f"posted[{self.dialect.value}].nc_hash: no NC may be recorded for status {self.status.value}")
            if not self.reason:
                raise ContractError(f"posted[{self.dialect.value}].reason: required for status {self.status.value}")


@dataclass(frozen=True)
class RunRecord:
    """Index of one solver run (written as ``run.json``)."""

    run_id: str
    task_hash: str
    status: RunStatus
    artifacts: Tuple[str, ...]
    posted: Tuple[PostedProgram, ...] = ()
    failure_reason: Optional[str] = None
    source_commit: Optional[str] = None
    dirty_patch_hash: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}, got {self.schema_version!r}")
        _check_hash("task_hash", self.task_hash)
        _check_hash("dirty_patch_hash", self.dirty_patch_hash, allow_none=True)
        object.__setattr__(self, "status", _enum(RunStatus, "status", self.status))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        posted = tuple(PostedProgram(**p) if isinstance(p, Mapping) else p for p in self.posted)
        object.__setattr__(self, "posted", posted)
        dialects = [p.dialect for p in posted]
        if len(set(dialects)) != len(dialects):
            raise ContractError("posted: one entry per dialect")
        if self.status == RunStatus.COMPLETE:
            missing = [a for a in REQUIRED_RUN_ARTIFACTS if a not in self.artifacts]
            if missing:
                raise ContractError(f"artifacts: complete run is missing {missing}")
            if set(dialects) != set(DIALECT_ARTIFACTS):
                raise ContractError("posted: a complete run declares every dialect (supported or unsupported)")
            for p in posted:
                if p.status == PostStatus.SUPPORTED:
                    nc, ev = DIALECT_ARTIFACTS[p.dialect]
                    if nc not in self.artifacts or ev not in self.artifacts:
                        raise ContractError(f"artifacts: supported dialect {p.dialect.value} needs {nc} and {ev}")
            if self.failure_reason:
                raise ContractError("failure_reason: only for non-complete runs")
        elif not self.failure_reason:
            raise ContractError(f"failure_reason: required for status {self.status.value}")
        for p in posted:
            if p.status != PostStatus.SUPPORTED and DIALECT_ARTIFACTS[p.dialect][0] in self.artifacts:
                raise ContractError(f"artifacts: {DIALECT_ARTIFACTS[p.dialect][0]} present but dialect is {p.status.value}")

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunRecord":
        return cls(**_strip_unknown(cls, d))


# ---------------------------------------------------------------------------
# Interfaces (implemented downstream; typed here so consumers agree)
# ---------------------------------------------------------------------------
class Evaluator(Protocol):
    """Hard evaluators (T1.3 production, T1.0 reference, CAMotics adapter).

    ``settings`` holds discretization/reference settings; they are NOT part of the task
    or tool hashes but are copied verbatim into ``EvaluationResult.discretization``.
    Evaluators must be stateless across calls (same inputs -> same result)."""

    def evaluate_candidate(self, task: TaskSpec, motion: MotionPlan, settings: Mapping[str, Any]) -> EvaluationResult: ...

    def evaluate_posted_gcode(self, task: TaskSpec, nc_bytes: bytes, dialect: Dialect,
                              settings: Mapping[str, Any]) -> EvaluationResult: ...


class Parameterization(Protocol):
    """T2.1 map from parameters ``theta`` (float64 array, normalized coordinates) to
    per-step deltas (T, 3) in mm. ``pullback`` is the exact transpose Jacobian-vector product."""

    num_params: int

    def forward(self, theta): ...            # -> ndarray (T, 3)

    def pullback(self, grad_deltas): ...     # -> ndarray (num_params,)


class ArtifactWriter(Protocol):
    """T3.0 run-artifact writer: writes files atomically, then ``run.json`` last."""

    def write(self, run_dir: str, record: RunRecord, files: Mapping[str, bytes]) -> None: ...


# ---------------------------------------------------------------------------
# Serialization entry points
# ---------------------------------------------------------------------------
def _strip_unknown(cls, d: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, Mapping):
        raise ContractError(f"{cls.__name__}: expected an object, got {type(d).__name__}")
    names = {f_.name for f_ in dataclasses.fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ContractError(f"{cls.__name__}: unknown fields {unknown} (schema {SCHEMA_VERSION})")
    return dict(d)


_RECORDS = {"TaskSpec": TaskSpec, "MotionPlan": MotionPlan, "EvaluationResult": EvaluationResult, "RunRecord": RunRecord}


def dumps(record: Any) -> str:
    """Human-readable JSON with a ``record_type`` tag."""
    name = type(record).__name__
    if name not in _RECORDS:
        raise ContractError(f"dumps: unsupported record type {name}")
    return json.dumps({"record_type": name, "record": record.to_dict()}, indent=1, sort_keys=True, allow_nan=False)


def loads(text: str) -> Any:
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise ContractError(f"invalid JSON: {e}") from None
    name = d.get("record_type") if isinstance(d, dict) else None
    if name not in _RECORDS:
        raise ContractError(f"record_type: expected one of {sorted(_RECORDS)}, got {name!r}")
    return _RECORDS[name].from_dict(d["record"])


# ---------------------------------------------------------------------------
# Tiny canonical examples (API examples for downstream owners; also test fixtures)
# ---------------------------------------------------------------------------
def example_pocket_task() -> TaskSpec:
    """25.4 mm cube stock, 12 x 12 x 5 mm pocket around a 4 x 4 mm island, flat 3.175 mm
    radius cutter. Work frame: G54 at stock top-centre; machine G54 offset (100, 80, -150)."""
    grid = GridSpec(shape=(16, 16, 16), origin_mm=(-12.7, -12.7, -25.4), voxel_mm=25.4 / 16)
    stock = GeometrySource(kind="analytic", params={"primitive": "box", "min_mm": [-12.7, -12.7, -25.4],
                                                     "max_mm": [12.7, 12.7, 0.0]})
    part = GeometrySource(kind="analytic", params={
        "primitive": "box_minus_pocket", "box_min_mm": [-12.7, -12.7, -25.4], "box_max_mm": [12.7, 12.7, 0.0],
        "pocket_min_mm": [-6.0, -6.0, -5.0], "pocket_max_mm": [6.0, 6.0, 0.0],
        "island_min_mm": [-2.0, -2.0, -5.0], "island_max_mm": [2.0, 2.0, 0.0]})
    w2m = Transform.translation((100.0, 80.0, -150.0))
    return TaskSpec(
        task_id="dev-pocket-000", family="pocket_island", split=Split.DEV,
        source=SourceInfo(origin="contracts:example_pocket_task@1", license="MIT"),
        length_unit="mm", stock=stock, original_part=part,
        operation_target=OperationTarget(geometry=part), grid=grid, work_to_machine=w2m,
        initial_state=MachineState(position_mm=w2m.apply((0.0, 0.0, 10.0)), frame=Frame.MACHINE),
        tool=ToolAssembly(cutter_kind=CutterKind.FLAT, cutter_radius_mm=3.175, flute_length_mm=12.0,
                          shank_radius_mm=3.175, holder_bottom_mm=25.0, holder_radius_mm=10.0,
                          holder_height_mm=40.0),
        limits=MachineLimits(max_feed_mm_s=50.0, rapid_mm_s=200.0, max_accel_mm_s2=500.0,
                             envelope_min_mm=(0.0, 0.0, -254.0), envelope_max_mm=(406.4, 304.8, 0.0)),
        tolerances=Tolerances(completion_residual_mm3=5.0, gouge_volume_mm3=1.0,
                              local_surface_error_mm=0.2, clearance_mm=1.0, rapid_penetration_mm=0.25),
    )


def example_motion_plan(task: Optional[TaskSpec] = None) -> MotionPlan:
    """Approach, plunge, one feed pass, one arc, retract -- machine frame."""
    task = task or example_pocket_task()
    m = task.work_to_machine.apply
    start = task.initial_state.position_mm
    pts = [m((-4.0, -4.0, 10.0)), m((-4.0, -4.0, -2.0)), m((4.0, -4.0, -2.0)), m((4.0, 4.0, -2.0)), m((4.0, 4.0, 10.0))]
    f = 10.0
    segs = [
        Segment(id="s0", kind=SegmentKind.RAPID, start_mm=start, end_mm=pts[0], duration_s=math.dist(start, pts[0]) / task.limits.rapid_mm_s, tag="approach"),
        Segment(id="s1", kind=SegmentKind.FEED, start_mm=pts[0], end_mm=pts[1], duration_s=math.dist(pts[0], pts[1]) / 5.0, feed_mm_s=5.0, tag="plunge"),
        Segment(id="s2", kind=SegmentKind.FEED, start_mm=pts[1], end_mm=pts[2], duration_s=8.0 / f, feed_mm_s=f),
        Segment(id="s3", kind=SegmentKind.ARC, start_mm=pts[2], end_mm=pts[3], duration_s=(math.pi * 4.0) / f, feed_mm_s=f,
                arc=ArcData(centre_mm=m((4.0, 0.0, -2.0)), plane=ArcPlane.XY, direction=ArcDirection.CCW)),
        Segment(id="s4", kind=SegmentKind.RAPID, start_mm=pts[3], end_mm=pts[4], duration_s=12.0 / task.limits.rapid_mm_s, tag="retract"),
    ]
    return MotionPlan(frame=Frame.MACHINE, initial_state=task.initial_state, segments=tuple(segs),
                      timing_rule=TimingRule.EXACT_STOP_CONSTANT_SPEED, provenance="contracts:example_motion_plan@1")
