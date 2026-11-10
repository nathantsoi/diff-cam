"""Direct RustCNC Face + Contour2D port for bounded rectangular DiffCAM tasks.

PROVENANCE(reference-copied): RustCNC commit
5fc4717a9342313c03461f80a19a79ad8996e757, clean read-only snapshot
/home/ntsoi/src/cnc/rustcnc-diffcam-port-5fc4717.
face.rs SHA-256 8c0ff95313ffb141eedd9f674caeca82cc8533b089b11fd481a1fe7a7891f99e;
contour2d.rs SHA-256 333fa825ce2cdb4570f387d935b344b17be016f630fb0204fd38a8f92a2ccb75.
Source docs: docs/cleanroom/spec-cam/{03-strategy-contour2d,
05-strategy-face,07-2d-engine}.md. DT-315/DT-316 constants feed the tests.

This is a direct port of the bounded paths, NOT a clean-room implementation,
an MIT-licensed file, or a wrapper around the legacy raster. No Voronoi or
arbitrary-shape fallback is claimed: loops must be axis-aligned rectangles.
Every requested loop and every sampled corner is emitted; the cap can only
reject, never truncate. Defaults are deterministic and recorded per plan.
"""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Mapping, Sequence

from benchmarks import contracts as C
from cam.seeds import seed_spec_for_task

RUSTCNC_SHA = "5fc4717a9342313c03461f80a19a79ad8996e757"
FACE_SHA256 = "8c0ff95313ffb141eedd9f674caeca82cc8533b089b11fd481a1fe7a7891f99e"
CONTOUR_SHA256 = "333fa825ce2cdb4570f387d935b344b17be016f630fb0204fd38a8f92a2ccb75"
Z_FEED = 2.0
RAMP_Z_CLEARANCE = 1.0
STOCK_SAFETY_FRACTION = 0.1
REDUCE_MACHINING_AREA_FACTOR = 0.005
LEAD_CORNER_RADIUS_FACTOR = 1.099_361_4
STEPOVER_GUARD_FACTOR = 1e-6
DEFAULT_TOLERANCE_MM = 0.01
FIT_ENCROACHMENT_MM = 0.01
DEFAULT_CHORD_ANGLE_DEGREES = 45.0
COARSE_FACE_STEPOVER_RADII = 3.0
DEFAULT_MAX_SEGMENTS = 72

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def _number(value: Any, name: str, *, positive=False, nonnegative=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    value = float(value)
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _custom_spec(spec: Any, task: C.TaskSpec, allowed: set[str]) -> tuple[dict[str, Any], Any]:
    """Default exactly to the card's named derivation; only mappings tune it."""
    derived = seed_spec_for_task(task)  # read-only geometry adapter; no legacy raster wrapper.
    if spec is None:
        return {}, derived
    if isinstance(spec, Mapping):
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError(f"unknown spec option(s): {unknown}")
        return dict(spec), derived
    if isinstance(spec, type(derived)):
        if spec != derived:
            raise ValueError("immutable seed_spec may not replace geometry; use a mapping for strategy tuning")
        return {}, spec
    raise ValueError("spec must be None, the default seed_spec, or a mapping of supported strategy options")


def _validate_context(task: C.TaskSpec) -> None:
    if task.original_part.frame is not C.Frame.WORK or task.stock.frame is not C.Frame.WORK:
        raise ValueError("task geometry must be in work coordinates")
    if task.fixtures:
        raise ValueError("fixtures are outside this bounded Face/Contour port")
    if not {"G0", "G1"}.issubset(task.motion_semantics.supported_gcodes):
        raise ValueError("coarse chords require G0/G1 motion support")
    if task.tool.cutter_kind is not C.CutterKind.FLAT:
        raise ValueError("this bounded port requires a flat cutter")
    if task.motion_semantics.timing_rule is not C.TimingRule.EXACT_STOP_CONSTANT_SPEED:
        raise ValueError("this port returns exact-stop constant-speed motions")
    if task.initial_state.frame is not C.Frame.MACHINE or not task.initial_state.known:
        raise ValueError("a known machine-frame initial state is required")
    _number(task.tool.cutter_radius_mm, "tool.cutter_radius_mm", positive=True)
    _number(task.limits.max_feed_mm_s, "max_feed_mm_s", positive=True)
    _number(task.limits.rapid_mm_s, "rapid_mm_s", positive=True)


def _bound_pair(params: Mapping[str, Any], lo: str, hi: str) -> tuple[Point3, Point3]:
    try:
        low, high = params[lo], params[hi]
        if len(low) != 3 or len(high) != 3:
            raise ValueError
        return (tuple(_number(v, lo, ) for v in low), tuple(_number(v, hi) for v in high))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"geometry requires finite three-dimensional {lo}/{hi} bounds") from error


@dataclasses.dataclass(frozen=True)
class _TaskGeometry:
    stock_lo: Point3
    stock_hi: Point3
    pocket_lo: Point3
    pocket_hi: Point3
    pad_lo: Point3 | None
    pad_hi: Point3 | None

    @property
    def retained(self) -> bool:
        return self.pad_lo is not None

    @property
    def top(self) -> float:
        return self.stock_hi[2]


def _task_geometry(task: C.TaskSpec, derived: Any) -> _TaskGeometry:
    if task.original_part.kind != "analytic" or task.original_part.params.get("primitive") != "box_minus_pocket":
        raise ValueError("only analytic box_minus_pocket task geometry is supported")
    if task.stock.kind != "analytic" or task.stock.params.get("primitive") != "box":
        raise ValueError("only analytic box stock is supported")
    params = task.original_part.params
    stock = _bound_pair(params, "box_min_mm", "box_max_mm")
    pocket = _bound_pair(params, "pocket_min_mm", "pocket_max_mm")
    if _bound_pair(task.stock.params, "min_mm", "max_mm") != stock:
        raise ValueError("stock source and original-part stock bounds differ")
    if derived.has_retained_island:
        pad_lo, pad_hi = _bound_pair(params, "island_min_mm", "island_max_mm")
        return _TaskGeometry(*stock, *pocket, pad_lo, pad_hi)
    return _TaskGeometry(*stock, *pocket, None, None)


def _rectangle(points: Sequence[Sequence[float]]) -> tuple[Point2, ...]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence) or len(points) not in (4, 5):
        raise ValueError("rectangular loop sequence must have len 4, or len 5 with a repeated closure")
    try:
        result = [(_number(point[0], "loop.x"), _number(point[1], "loop.y")) for point in points]
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError("rectangular loop points must be finite XY pairs") from error
    if result[0] == result[-1]:
        result.pop()
    if len(result) != 4 or len(set(result)) != 4:
        raise ValueError("a rectangular loop needs four distinct vertices")
    xs = {point[0] for point in result}
    ys = {point[1] for point in result}
    expected = {(x, y) for x in xs for y in ys}
    if len(xs) != 2 or len(ys) != 2 or set(result) != expected:
        raise ValueError("unsupported shape: only an axis-aligned rectangular loop is supported")
    if any((left[0] != right[0] and left[1] != right[1])
           for left, right in zip(result, result[1:] + result[:1])):
        raise ValueError("axis-aligned rectangle vertices must be consecutive in travel order")
    return tuple(result)


def _rect_bounds(lo: Point3, hi: Point3, inset=0.0, *, clockwise=False) -> tuple[Point2, ...]:
    radii = _number(inset, "inset")
    xmin, ymin = lo[0] + radii, lo[1] + radii
    xmax, ymax = hi[0] - radii, hi[1] - radii
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("offset exceeds maximum: rectangle collapses under the cutter inset")
    points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    if clockwise:
        points = [points[3], points[2], points[1], points[0]]
    return tuple(points)


def _cap(value: int, strategy: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_segments must be a positive integer")
    return value


def _require_count(segments: Sequence[Any], max_segments: int, strategy: str) -> None:
    if len(segments) > max_segments:
        raise ValueError(f"{strategy} exceeds max_segments={max_segments}; required={len(segments)} "
                         "complete segments; no loop, row, or arc chord was truncated")


class _Emitter:
    """Contiguous machine-frame lines with physical exact-stop durations."""

    def __init__(self, task: C.TaskSpec, derived: Any, prefix: str,
                 start_machine_mm: Sequence[float] | None):
        self.task, self.derived = task, derived
        self.prefix = prefix
        position = task.initial_state.position_mm if start_machine_mm is None else start_machine_mm
        if not isinstance(position, Sequence) or len(position) != 3:
            raise ValueError("start_machine_mm must be a three-dimensional machine position")
        self.machine = tuple(_number(value, "start_machine_mm") for value in position)
        self.current = tuple(float(v) for v in task.work_to_machine.inverse().apply(self.machine))
        self.segments: list[C.Segment] = []
        self.stock_top = task.stock.params["max_mm"][2]
        self.ids: set[str] = set()

    def push_linear(self, kind: C.SegmentKind, end: Point3, tag: str,
                    feed_mm_s: float | None = None) -> None:
        start = self.current
        if math.dist(start, end) <= 1e-12:
            return
        start_machine = tuple(self.task.work_to_machine.apply(start))
        end_machine = tuple(self.task.work_to_machine.apply(end))
        length = math.dist(start_machine, end_machine)
        if kind is C.SegmentKind.RAPID:
            if start[2] <= self.stock_top or end[2] <= self.stock_top:
                raise ValueError("rapid would penetrate stock: port requires both rapid endpoints above stock top")
            speed, feed = self.task.limits.rapid_mm_s, None
        else:
            speed = feed_mm_s if feed_mm_s is not None else self.derived.feed_mm_s
            speed = _number(speed, "feed_mm_s", positive=True)
            if speed > self.task.limits.max_feed_mm_s:
                raise ValueError(f"feed_mm_s {speed} exceeds task limit {self.task.limits.max_feed_mm_s}")
            feed = speed
        segment_id = f"{self.prefix}-{len(self.segments):03d}"
        if segment_id in self.ids:
            raise ValueError(f"duplicate segment id {segment_id}")
        self.ids.add(segment_id)
        self.segments.append(C.Segment(segment_id, kind, start_machine, end_machine,
                                       length / speed, feed_mm_s=feed, tag=tag))
        self.current, self.machine = tuple(end), end_machine


def _provenance(strategy: str, task: C.TaskSpec, options: Mapping[str, Any],
                derived: Any, max_segments: int, notes: Sequence[str]) -> str:
    record = {
        "strategy": strategy, "source": "RustCNC direct port", "rustcnc_commit": RUSTCNC_SHA,
        "task_id": task.task_id, "task_hash": task.task_hash,
        "face_sha256": FACE_SHA256, "contour_sha256": CONTOUR_SHA256,
        "tuning": dict(sorted(options.items())),
        "tuned_face_stepover_mm": COARSE_FACE_STEPOVER_RADII * task.tool.cutter_radius_mm,
        "original_face_maximum_stepover_mm": task.tool.cutter_radius_mm / 2.0,
        "default_spec": "cam.seeds.seed_spec_for_task(task)",
        "chord_rule": f"finite straight-line chords; max {DEFAULT_CHORD_ANGLE_DEGREES} degrees per arc chord",
        "max_segments": max_segments,
        "named_simplifications": list(notes),
        "geometry_centres": {"pocket": list(derived.pocket_centre_xy), "pad": list(derived.island_centre_xy)},
    }
    return "CAM-RUST-1 " + json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _plan(task: C.TaskSpec, segments: Sequence[C.Segment], strategy: str,
          options: Mapping[str, Any], derived: Any, max_segments: int, notes: Sequence[str]) -> C.MotionPlan:
    _require_count(segments, max_segments, strategy)
    identifier = {s.id for s in segments}
    if len(identifier) != len(segments):
        raise ValueError("segment IDs are not unique across emitted loops/stages")
    plan = C.MotionPlan(frame=C.Frame.MACHINE, initial_state=task.initial_state,
                        segments=tuple(segments), timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
                        provenance=_provenance(strategy, task, options, derived, max_segments, notes))
    plan.validate_for_task(task)
    return plan


# port: strategies/face.rs:677-739 Level + FacePlan (adapted scalar plan inputs)
@dataclasses.dataclass(frozen=True)
class Level:
    z: float
    final_pass: bool


@dataclasses.dataclass(frozen=True)
class FacePlan:
    cutter_radius: float
    corner_radius: float = 0.0
    maximum_stepover: float = 0.0
    tolerance: float = DEFAULT_TOLERANCE_MM
    retract: float = 0.0
    feed_height: float = 0.0
    feed: float = 0.0
    z_high: float = 0.0
    depth: float = 0.0
    maximum_stepdown: float = 0.0
    even_z_steps: bool = False
    finishing_stepdown: float = -1.0
    axial_stock_to_leave: float = 0.0
    from_other_side: bool = False
    both_sides: bool = False


# port: strategies/face.rs:943-973 depth_ladder; absolute z_low adapter made explicit
def depth_ladder(z_high: float, z_low: float, maximum_stepdown: float, *,
                 even_z_steps=False, finishing_stepdown=-1.0,
                 axial_stock_to_leave=0.0) -> list[Level]:
    z_high = _number(z_high, "z_high")
    z_low = _number(z_low, "z_low")
    maximum_stepdown = _number(maximum_stepdown, "maximum_stepdown", nonnegative=True)
    finishing_stepdown = _number(finishing_stepdown, "finishing_stepdown")
    axial_stock_to_leave = _number(axial_stock_to_leave, "axial_stock_to_leave")
    if not isinstance(even_z_steps, bool):
        raise ValueError("even_z_steps must be boolean")
    finishing = finishing_stepdown >= 0.0
    if finishing:
        z_low += finishing_stepdown
    if z_low > z_high:
        return []
    n = 1
    if z_high > z_low and maximum_stepdown > 0.0:
        raw = (z_high - z_low - axial_stock_to_leave) / maximum_stepdown
        n = max(math.ceil(raw), 1)
    stepdown = ((z_high - z_low - axial_stock_to_leave) / n
                if even_z_steps else maximum_stepdown)
    count = n + 1 if finishing else n
    levels = []
    for i in range(count):
        final_pass = i + 1 == count
        z = (z_low - finishing_stepdown + axial_stock_to_leave
             if final_pass and finishing else
             z_low + (n - 1.0 - i) * stepdown + axial_stock_to_leave)
        levels.append(Level(z=z, final_pass=final_pass))
    return levels


@dataclasses.dataclass(frozen=True)
class PolyLoop:
    pts: tuple[Point2, ...] | Sequence[Point2]
    depth: int

    def __post_init__(self):
        if not isinstance(self.depth, int) or isinstance(self.depth, bool) or self.depth not in (0, 1):
            raise ValueError("this bounded port supports outer/one retained loop depths only")

    def v_range(self) -> tuple[float, float]:
        return min(p[1] for p in self.pts), max(p[1] for p in self.pts)

    def u_range(self) -> tuple[float, float]:
        return min(p[0] for p in self.pts), max(p[0] for p in self.pts)


# port: strategies/face.rs:1316-1327 point_in_polygon
def point_in_polygon(p: Point2, poly: Sequence[Point2]) -> bool:
    crossings = 0
    for p0, p1 in zip(poly, list(poly[1:]) + [poly[0]]):
        if (p0[1] > p[1]) != (p1[1] > p[1]):
            t = (p[1] - p0[1]) / (p1[1] - p0[1])
            if p0[0] + t * (p1[0] - p0[0]) > p[0]:
                crossings += 1
    return crossings % 2 == 1


@dataclasses.dataclass
class Region:
    loops: list[PolyLoop]
    v_min: float
    v_max: float
    u_min: float
    u_max: float

    # port: strategies/face.rs:1026-1045 Region::new
    @classmethod
    def new(cls, loops: Sequence[PolyLoop]):
        if not loops:
            raise ValueError("the face region has no boundary")
        v_lo = min(loop.v_range()[0] for loop in loops)
        v_hi = max(loop.v_range()[1] for loop in loops)
        u_lo = min(loop.u_range()[0] for loop in loops)
        u_hi = max(loop.u_range()[1] for loop in loops)
        if not all(math.isfinite(v) for v in (v_lo, v_hi, u_lo, u_hi)) or not v_hi > v_lo or not u_hi > u_lo:
            raise ValueError("the face boundaries carry no area")
        return cls(list(loops), v_lo, v_hi, u_lo, u_hi)

    # port: strategies/face.rs:1049-1062 Region::loop_crossings
    @staticmethod
    def loop_crossings(loop: PolyLoop, v: float) -> list[float]:
        xs = []
        for p0, p1 in zip(loop.pts, list(loop.pts[1:]) + [loop.pts[0]]):
            if (p0[1] > v) != (p1[1] > v):
                t = (v - p0[1]) / (p1[1] - p0[1])
                xs.append(p0[0] + t * (p1[0] - p0[0]))
        return sorted(xs)

    # port: strategies/face.rs:1076-1115 Region::row_intervals
    def row_intervals(self, v: float, cutter_radius: float, minimum_length: float) -> list[tuple[float, float]]:
        if v < self.v_min or v > self.v_max:
            return [(self.u_min, self.u_max)]
        events = []
        for loop in self.loops:
            if loop.depth % 2 == 1:
                hlo, hhi = loop.v_range()
                if not (v - cutter_radius >= hlo and v + cutter_radius <= hhi):
                    continue
            xs = self.loop_crossings(loop, v)
            for i in range(0, len(xs), 2):
                if i + 1 < len(xs):
                    events.extend(((xs[i], 1), (xs[i + 1], -1)))
        events.sort(key=lambda event: (event[0], event[1]))
        out, parity, open_x = [], 0, 0.0
        for x, delta in events:
            was_odd = parity % 2 == 1
            parity += delta
            is_odd = parity % 2 == 1
            if not was_odd and is_odd:
                open_x = x
            if was_odd and not is_odd and x - open_x >= minimum_length:
                out.append((open_x, x))
        return out

    # port: strategies/face.rs:1119-1153 Region::band
    def band(self, plan: FacePlan, other_side: bool):
        r0 = REDUCE_MACHINING_AREA_FACTOR * plan.cutter_radius
        flat_radius = plan.cutter_radius - plan.corner_radius
        last_cut_overlap = max(0.02 * plan.maximum_stepover, 0.05 * flat_radius)
        last_cut_distance = plan.cutter_radius - plan.corner_radius - last_cut_overlap
        minimum_length = 10.0 * plan.tolerance
        if other_side:
            umin_flat = self.v_min + r0 + last_cut_distance
            umax = self.v_max - r0 + plan.cutter_radius
            row_start, step_sign, distance = umax, -1.0, umax - umin_flat
        else:
            umin = self.v_min + r0 - plan.cutter_radius
            umax_flat = self.v_max - r0 - last_cut_distance
            row_start, step_sign, distance = umin, 1.0, umax_flat - umin
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"face band distance={distance} leaves no machinable span")
        n, stepover = stepover_data(distance, plan.maximum_stepover)
        rows = [Row(u=row_start + step_sign * i * stepover,
                    slices=self.row_intervals(row_start + step_sign * i * stepover,
                                              plan.cutter_radius, minimum_length))
                for i in range(1, n + 1)]
        return Band(rows, other_side, self.v_min, self.v_max, plan.cutter_radius)


@dataclasses.dataclass(frozen=True)
class Row:
    u: float
    slices: list[tuple[float, float]]


@dataclasses.dataclass
class Band:
    rows: list[Row]
    other_side: bool
    v_min: float
    v_max: float
    cutter_radius: float

    # port: strategies/face.rs:1171-1216 Band::{travel_x,emission_order,pass_ends}
    def travel_x(self) -> float:
        return 1.0 if self.other_side else -1.0

    def emission_order(self) -> list[tuple[int, int]]:
        order, i = [], 0
        while i < len(self.rows):
            if len(self.rows[i].slices) <= 1:
                if len(self.rows[i].slices) == 1:
                    order.append((i, 0))
                i += 1
                continue
            j = i
            while j < len(self.rows) and len(self.rows[j].slices) > 1:
                j += 1
            max_slices = max(len(row.slices) for row in self.rows[i:j])
            for s in range(max_slices):
                for k, row in enumerate(self.rows[i:j]):
                    if len(row.slices) > s:
                        order.append((i + k, s))
            i = j
        return order

    def pass_ends(self, row: int, slice_: int) -> tuple[Point2, Point2]:
        row_value = self.rows[row]
        a, b = row_value.slices[slice_]
        return ((a, row_value.u), (b, row_value.u)) if self.other_side else ((b, row_value.u), (a, row_value.u))

    # port: strategies/face.rs:1222-1231 Band::lead_run
    def lead_run(self, row: int) -> float:
        nominal = self.cutter_radius * (1.0 + STOCK_SAFETY_FRACTION)
        row_u = self.rows[row].u
        dy = max(row_u - self.v_max, 0.0) if self.other_side else max(self.v_min - row_u, 0.0)
        if dy <= 0.0:
            return nominal
        radius = self.cutter_radius * LEAD_CORNER_RADIUS_FACTOR
        return math.sqrt(max(radius * radius - dy * dy, 0.0))


# port: strategies/face.rs:1240-1248 stepover_data
def stepover_data(distance: float, maximum_stepover: float) -> tuple[int, float]:
    distance = _number(distance, "distance", positive=True)
    maximum_stepover = _number(maximum_stepover, "maximum_stepover", positive=True)
    steps = math.floor(distance / maximum_stepover)
    if distance / maximum_stepover - steps > STEPOVER_GUARD_FACTOR * maximum_stepover:
        steps = math.ceil(distance / maximum_stepover)
    n = max(int(steps), 1)
    return n, distance / n


# port: strategies/face.rs:308-410 baseline per-pass stream and final retract
def _face_emit(task: C.TaskSpec, plan: FacePlan, loops: Sequence[PolyLoop], *,
               z_low: float, derived: Any, emitter: _Emitter, prefix: str) -> None:
    region = Region.new(loops)
    levels = depth_ladder(plan.z_high, z_low, plan.maximum_stepdown,
                          even_z_steps=plan.even_z_steps,
                          finishing_stepdown=plan.finishing_stepdown,
                          axial_stock_to_leave=plan.axial_stock_to_leave)
    if not levels:
        raise ValueError("face depth lies above stock top")
    position = None if not emitter.segments else emitter.current
    for level_index, level in enumerate(levels):
        other_side = plan.from_other_side != (plan.both_sides and level_index % 2 == 1)
        band = region.band(plan, other_side)
        order = band.emission_order()
        if not order:
            raise ValueError("Face emitted no passes; refusing an incomplete strategy")
        z_cut = level.z
        z_plunge_to = z_cut + RAMP_Z_CLEARANCE
        z_feed = plan.z_high + Z_FEED
        for row, slice_ in order:
            start, end = band.pass_ends(row, slice_)
            travel = band.travel_x()
            col_x = start[0] - travel * (RAMP_Z_CLEARANCE + band.lead_run(row))
            if position is None:
                # Typed initial position is not falsely relabelled as the Rust
                # probe's already-positioned column top. Initial point → feed
                # height is all above stock top, then down to zHigh + Z_FEED.
                emitter.push_linear(C.SegmentKind.RAPID, (col_x, start[1], z_feed),
                                    "approach-rapid", None)
            else:
                up = (position[0], position[1], plan.feed_height)
                if position != up:
                    emitter.push_linear(C.SegmentKind.RAPID, up, "approach-rapid", None)
                across = (col_x, start[1], plan.feed_height)
                if up != across:
                    emitter.push_linear(C.SegmentKind.RAPID, across, "approach-rapid", None)
                emitter.push_linear(C.SegmentKind.RAPID, (col_x, start[1], z_feed),
                                    "approach-rapid", None)
            emitter.push_linear(C.SegmentKind.FEED, (col_x, start[1], z_plunge_to),
                                "plunge", derived.plunge_mm_s)
            diag_end = (col_x + travel * RAMP_Z_CLEARANCE, start[1], z_cut)
            emitter.push_linear(C.SegmentKind.FEED, diag_end, "lead-in-diagonal", plan.feed)
            emitter.push_linear(C.SegmentKind.FEED, (start[0], start[1], z_cut),
                                "lead-in-run", plan.feed)
            emitter.push_linear(C.SegmentKind.FEED, (end[0], end[1], z_cut), "face-cut", plan.feed)
            out = (end[0] + travel * RAMP_Z_CLEARANCE, end[1], z_plunge_to)
            emitter.push_linear(C.SegmentKind.FEED, out, "lead-out", plan.feed)
            position = out
            # Card-required rapid adaptation: Rust's next vertical rapid can
            # start at cut+1 below stock. Lift that column at FEED to top+1
            # first; no sub-stock endpoint is hidden behind a G0 declaration.
            if z_plunge_to <= plan.z_high:
                position = (out[0], out[1], plan.z_high + RAMP_Z_CLEARANCE)
                emitter.push_linear(C.SegmentKind.FEED, position, "lead-out-safe-lift", plan.feed)
    if position is not None:
        emitter.push_linear(C.SegmentKind.RAPID, (position[0], position[1], plan.retract),
                            "retract", None)


_FACE_OPTIONS = {"maximum_stepover", "maximum_stepdown", "depth", "even_z_steps",
                 "finishing_stepdown", "axial_stock_to_leave", "from_other_side", "both_sides",
                 "tolerance", "feed_mm_s", "plunge_mm_s", "lower_step_cap"}


def face_seed(task: C.TaskSpec, spec: Any = None, *, max_segments=DEFAULT_MAX_SEGMENTS) -> C.MotionPlan:
    """Emit complete Rust Face bands, plus an explicit lower-pad cap stage.

    The outer rectangle is the task pocket inset by cutter radius (tool-centre
    boundary), not an erasure of retained stock rims. A retained full-height pad
    is an XOR hole. A lower pad is first cap-faced at its declared top with the
    hole absent, then its below-cap foot is protected by the ordinary floor Face.
    This bounded stage is explicit, never a substituted floor-level pad cut.
    Coarse default stepover is 3R, explicitly tuned from Rust's R/2 for the 72-line budget.
    """
    max_segments = _cap(max_segments, "cam.face")
    _validate_context(task)
    options, derived = _custom_spec(spec, task, _FACE_OPTIONS)
    geom = _task_geometry(task, derived)
    radius = float(task.tool.cutter_radius_mm)
    stepover = _number(options.get("maximum_stepover", COARSE_FACE_STEPOVER_RADII * radius),
                       "maximum_stepover", positive=True)  # explicit coarse tuning.
    maximum_stepdown = _number(options.get("maximum_stepdown", 0.0), "maximum_stepdown", nonnegative=True)
    depth = _number(options.get("depth", geom.pocket_lo[2]), "depth")
    even = _boolean(options.get("even_z_steps", False), "even_z_steps")
    finishing = _number(options.get("finishing_stepdown", -1.0), "finishing_stepdown")
    axial = _number(options.get("axial_stock_to_leave", 0.0), "axial_stock_to_leave")
    other = _boolean(options.get("from_other_side", False), "from_other_side")
    both = _boolean(options.get("both_sides", False), "both_sides")
    tolerance = _number(options.get("tolerance", DEFAULT_TOLERANCE_MM), "tolerance", positive=True)
    feed = _number(options.get("feed_mm_s", derived.feed_mm_s), "feed_mm_s", positive=True)
    plunge = _number(options.get("plunge_mm_s", derived.plunge_mm_s), "plunge_mm_s", positive=True)
    derived = dataclasses.replace(derived, feed_mm_s=feed, plunge_mm_s=plunge)
    if max(feed, plunge) > task.limits.max_feed_mm_s:
        raise ValueError("requested face feed exceeds the task limit")
    lower_policy = options.get("lower_step_cap", "face_then_floor")
    if lower_policy not in ("face_then_floor", "refuse"):
        raise ValueError("lower_step_cap must be face_then_floor or refuse")
    outer = _rect_bounds(geom.pocket_lo, geom.pocket_hi, radius)
    pad = (_rect_bounds(geom.pad_lo, geom.pad_hi, -radius) if geom.retained else None)
    outer_loop = PolyLoop(outer, 0)
    pad_loop = None if pad is None else PolyLoop(pad, 1)
    if pad_loop is not None and point_in_polygon(pad_loop.pts[0], outer) is False:
        raise ValueError("retained-pad tool-centre hole escapes the task pocket boundary")
    notes = ["tool-centre pocket inset + retained-pad outset by cutter radius",
             "Face coarse maximumStepover=3R, an explicit 72-budget tuning, not Rust R/2 default",
             "below-stock lead-out columns receive a FEED lift to stock top + 1 before rapid"]
    config = {"maximum_stepover": stepover, "depth": depth, "maximum_stepdown": maximum_stepdown,
              "even_z_steps": even, "finishing_stepdown": finishing,
              "axial_stock_to_leave": axial, "from_other_side": other, "both_sides": both,
              "feed_mm_s": feed, "plunge_mm_s": plunge, "tolerance": tolerance,
              "lower_step_cap": lower_policy}
    segments, start = [], None
    if geom.retained and geom.pad_hi[2] < geom.top:
        if lower_policy == "refuse":
            raise ValueError("lower-step pad cap requires Face at pad top followed by floor; policy=refuse")
        if depth > geom.pad_hi[2] or axial != 0.0:
            raise ValueError("lower-step cap requires final depth at/below pad top and zero axial allowance")
        cap_plan = FacePlan(radius, maximum_stepover=stepover, tolerance=tolerance,
                            retract=derived.home_z, feed_height=derived.clearance_z, feed=feed,
                            z_high=geom.top, maximum_stepdown=maximum_stepdown,
                            even_z_steps=even, finishing_stepdown=finishing,
                            axial_stock_to_leave=axial,
                            from_other_side=other, both_sides=both)
        emitter = _Emitter(task, derived, "face-pad-cap", None)
        _face_emit(task, cap_plan, [outer_loop], z_low=geom.pad_hi[2], derived=derived, emitter=emitter,
                   prefix="face-pad-cap")
        segments.extend(emitter.segments)
        start = emitter.machine
        notes.append("lower-step pad: outer-only Face at pad top, then retained-pad floor Face")
    plan = FacePlan(radius, maximum_stepover=stepover, tolerance=tolerance,
                    retract=derived.home_z, feed_height=derived.clearance_z, feed=feed,
                    z_high=geom.top, maximum_stepdown=maximum_stepdown,
                    even_z_steps=even, finishing_stepdown=finishing,
                    from_other_side=other, both_sides=both)
    loops = [outer_loop] if pad_loop is None else [outer_loop, pad_loop]
    emitter = _Emitter(task, derived, "face-floor", start)
    _face_emit(task, plan, loops, z_low=depth, derived=derived, emitter=emitter, prefix="face-floor")
    segments.extend(emitter.segments)
    return _plan(task, segments, "cam.face", config, derived, max_segments, notes)


# port: strategies/contour2d.rs:169-233 Piece and its interpolation
@dataclasses.dataclass(frozen=True)
class Piece:
    from_: Point2
    to_: Point2
    centre: Point2 | None = None
    sweep: float = 0.0

    def point_at(self, fraction: float) -> Point2:
        if self.centre is None:
            return (self.from_[0] + fraction * (self.to_[0] - self.from_[0]),
                    self.from_[1] + fraction * (self.to_[1] - self.from_[1]))
        radius = math.dist(self.from_, self.centre)
        a0 = math.atan2(self.from_[1] - self.centre[1], self.from_[0] - self.centre[0])
        a = a0 + self.sweep * fraction
        return (self.centre[0] + radius * math.cos(a), self.centre[1] + radius * math.sin(a))

    def tangent_at(self, fraction: float) -> Point2:
        if self.centre is None:
            return direction(self.from_, self.to_)
        radius = math.dist(self.from_, self.centre)
        a0 = math.atan2(self.from_[1] - self.centre[1], self.from_[0] - self.centre[0])
        a = a0 + self.sweep * fraction
        return (math.sin(a), -math.cos(a)) if self.sweep < 0 else (-math.sin(a), math.cos(a))


@dataclasses.dataclass(frozen=True)
class PassGeometry:
    root: Point2
    pieces: tuple[Piece, ...]

    # port: strategies/contour2d.rs:270-299 PassGeometry::reseated
    def reseated(self, piece: int, fraction: float) -> list[Piece]:
        if piece == 0 and fraction <= 0.0:
            return list(self.pieces)
        split = self.pieces[piece]
        start = split.point_at(fraction)
        out = [dataclasses.replace(split, from_=start)]
        out.extend(self.pieces[piece + 1:])
        out.extend(self.pieces[:piece])
        if fraction > 0.0:
            out.append(dataclasses.replace(split, to_=start))
        return out


# port: strategies/contour2d.rs:921-939 direction
def direction(from_: Point2, to_: Point2) -> Point2:
    length = math.dist(from_, to_)
    if length <= 0.0:
        return 1.0, 0.0
    return ((to_[0] - from_[0]) / length, (to_[1] - from_[1]) / length)


def _corner_join(left: Point2, right: Point2, n_left: Point2, n_right: Point2, radius: float) -> Point2:
    """Raw offset-line intersection (overlap joins, geometry/offset.rs:229-325)."""
    dx1, dy1 = right[0] - left[0], right[1] - left[1]
    dx2, dy2 = n_right[0] - n_left[0], n_right[1] - n_left[1]
    det = dx1 * dy2 - dy1 * dx2
    if abs(det) <= 1e-12:
        raise ValueError("parallel rectangle offset segments have no overlap join")
    t = ((n_left[0] - left[0]) * dy2 - (n_left[1] - left[1]) * dx2) / det
    return (left[0] + t * dx1, left[1] + t * dy1)


# port: strategies/contour2d.rs:907-919 seam_is_tangential + offset.rs:135-325 rect specialization
def _offset_contour(points: Sequence[Point2], radius: float, right_compensation: bool):
    vertices = _rectangle(points)
    distance = -radius if right_compensation else radius
    edges = []
    for i, (a, b) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
        tangent = direction(a, b)
        normal = (-tangent[1], tangent[0])
        raw_from = (a[0] + distance * normal[0], a[1] + distance * normal[1])
        raw_to = (b[0] + distance * normal[0], b[1] + distance * normal[1])
        edges.append([a, b, raw_from, raw_to])
    pieces: list[Piece] = []
    for i, (a, b, raw_from, raw_to) in enumerate(edges):
        here_tangent = direction(a, b)
        prev_a, prev_b = vertices[i - 1], a
        previous_tangent = direction(prev_a, prev_b)
        cross = previous_tangent[0] * here_tangent[1] - previous_tangent[1] * here_tangent[0]
        dot = previous_tangent[0] * here_tangent[0] + previous_tangent[1] * here_tangent[1]
        turn = math.atan2(cross, dot)
        gap = (distance > 0.0 and turn < 0.0) or (distance < 0.0 and turn > 0.0)
        if pieces:
            if gap:
                previous_end = pieces[-1].to_
                pieces.append(Piece(previous_end, raw_from, a,
                                    _arc_sweep(previous_end, a, raw_from, turn < 0.0)))
            else:
                previous_a = pieces[-1].from_
                joined = _corner_join(previous_a, pieces[-1].to_, raw_from, raw_to, radius)
                pieces[-1] = dataclasses.replace(pieces[-1], to_=joined)
                raw_from = joined
        pieces.append(Piece(raw_from, raw_to))
    # Close the final corner with the same true arc/intersection as every other.
    last_a, last_b = vertices[-2], vertices[-1]
    incoming, outgoing = direction(last_a, last_b), direction(vertices[-1], vertices[0])
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    turn = math.atan2(cross, dot)
    root_gap = (distance > 0.0 and turn < 0.0) or (distance < 0.0 and turn > 0.0)
    if root_gap:
        centre = vertices[0]
        sweep = _arc_sweep(pieces[-1].to_, centre, pieces[0].from_, turn < 0.0)
        pieces.append(Piece(pieces[-1].to_, pieces[0].from_, centre, sweep))
    else:
        joined = _corner_join(pieces[-1].from_, pieces[-1].to_, pieces[0].from_, pieces[0].to_, radius)
        pieces[-1] = dataclasses.replace(pieces[-1], to_=joined)
        pieces[0] = dataclasses.replace(pieces[0], from_=joined)
    first, last = pieces[0], pieces[-1]
    if math.dist(last.to_, first.from_) > 1e-8:
        raise ValueError("offset fails to close the complete input loop")
    root = first.from_
    return PassGeometry(root, tuple(pieces)), root_gap


# port: strategies/contour2d.rs:191-201 Piece::sweep
def _arc_sweep(start: Point2, centre: Point2, end: Point2, clockwise: bool) -> float:
    r0 = (start[0] - centre[0], start[1] - centre[1])
    r1 = (end[0] - centre[0], end[1] - centre[1])
    a0 = math.atan2(r0[1], r0[0])
    a1 = math.atan2(r1[1], r1[0])
    if math.dist(start, end) <= 1e-9:
        return -2 * math.pi if clockwise else 2 * math.pi
    return -((a0 - a1) % (2 * math.pi)) if clockwise else (a1 - a0) % (2 * math.pi)


@dataclasses.dataclass(frozen=True)
class LeadGeom:
    radius: float
    distance: float
    vertical_radius: float
    sweep: float


@dataclasses.dataclass(frozen=True)
class StartSelection:
    point: Point2
    tangent: Point2
    normal: Point2
    piece: int
    fraction: float


# port: strategies/contour2d.rs:482-506 lerp + dist_point_segment
def _lerp(a: Point2, b: Point2, fraction: float) -> Point2:
    return (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))


def _dist_point_segment(p: Point2, a: Point2, b: Point2) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    if l2 <= 0.0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _dist_to_walls(point: Point2, walls: Sequence[Point2]) -> float:
    # A boundary is closed independently: no fabricated connector can join
    # one loop's repeated start to the next loop's first vertex.
    chains = (walls,) if walls and isinstance(walls[0][0], (int, float)) else walls
    return min(_dist_point_segment(point, a, b)
               for chain in chains for a, b in zip(chain, chain[1:]))


def _clear(points: Sequence[Point2], walls: Sequence[Point2], radius: float) -> bool:
    # Check the emitted chord itself, not just sample points on the parent
    # arc. Crossing chords are conservatively refused even at tiny clearance.
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    walls = (walls,) if walls and isinstance(walls[0][0], (int, float)) else walls
    for start, end in zip(points, points[1:]):
        if math.dist(start, end) <= 0.0:
            continue
        for chain in walls:
            for left, right in zip(chain, chain[1:]):
                direction = (end[0] - start[0], end[1] - start[1])
                wall_direction = (right[0] - left[0], right[1] - left[1])
                if abs(cross(direction, wall_direction)) > 1e-15:
                    first = cross(direction, (start[0] - left[0], start[1] - left[1]))
                    second = cross(direction, (end[0] - left[0], end[1] - left[1]))
                    third = cross(wall_direction, (left[0] - start[0], left[1] - start[1]))
                    fourth = cross(wall_direction, (right[0] - start[0], right[1] - start[1]))
                    if first * second <= 0.0 and third * fourth <= 0.0:
                        return False
                distances = (_dist_point_segment(start, left, right),
                             _dist_point_segment(end, left, right),
                             _dist_point_segment(left, start, end),
                             _dist_point_segment(right, start, end))
                if min(distances) < radius - FIT_ENCROACHMENT_MM:
                    return False
    return all(_dist_to_walls(point, walls) >= radius - FIT_ENCROACHMENT_MM for point in points)


# port: strategies/contour2d.rs:474-480 quarter_point
def quarter_point(centre: Point3, start: Point3, end: Point3, angle: float) -> Point3:
    ca, sa = math.cos(angle), math.sin(angle)
    return tuple(centre[i] + ca * (start[i] - centre[i]) + sa * (end[i] - centre[i])
                 for i in range(3))


@dataclasses.dataclass(frozen=True)
class Lead:
    geom: LeadGeom
    t: Point2
    m: Point2
    centre: Point2
    approach: Point2
    column_top: Point2
    descent_end: Point2
    departure: Point2
    departure_top: Point2
    departure_ascent_start: Point2
    start: Point2

    # port: strategies/contour2d.rs:367-395 Lead::build
    @classmethod
    def build(cls, s: Point2, t: Point2, m: Point2, geom: LeadGeom):
        def at(scale, base, direction_):
            return (base[0] + scale * direction_[0], base[1] + scale * direction_[1])
        centre = at(geom.radius, s, m)
        if geom.sweep == math.pi / 2:
            back, forward = at(-geom.radius, s, t), at(geom.radius, s, t)
            approach, departure = at(geom.radius, back, m), at(geom.radius, forward, m)
        else:
            st, ct = math.sin(geom.sweep), math.cos(geom.sweep)
            approach = (centre[0] + geom.radius * (-st * t[0] - ct * m[0]),
                        centre[1] + geom.radius * (-st * t[1] - ct * m[1]))
            departure = (centre[0] + geom.radius * (st * t[0] - ct * m[0]),
                         centre[1] + geom.radius * (st * t[1] - ct * m[1]))
        descent_end = at(geom.distance, approach, m)
        column_top = at(geom.distance + geom.vertical_radius, approach, m)
        return cls(geom, t, m, centre, approach, column_top, descent_end, departure,
                   at(geom.distance + geom.vertical_radius, departure, m),
                   at(geom.distance, departure, m), s)

    def _horizontal(self, start: Point2, end: Point2, angle: float) -> Point2:
        if self.geom.sweep == math.pi / 2:
            return quarter_point((*self.centre, 0.0), (*start, 0.0), (*end, 0.0), angle)[:2]
        u = (start[0] - self.centre[0], start[1] - self.centre[1])
        ut, um = u[0] * self.t[0] + u[1] * self.t[1], u[0] * self.m[0] + u[1] * self.m[1]
        sa, ca = math.sin(angle), math.cos(angle)
        vt, vm = ut * ca - um * sa, ut * sa + um * ca
        return (self.centre[0] + vt * self.t[0] + vm * self.m[0],
                self.centre[1] + vt * self.t[1] + vm * self.m[1])

    # port: strategies/contour2d.rs:404-425 Lead::entry_fits
    def entry_fits(self, walls: Sequence[Point2], radius: float, z: float, top: float) -> bool:
        points = []
        v = self.geom.vertical_radius
        chord_count = max(1, math.ceil(self.geom.sweep / math.radians(DEFAULT_CHORD_ANGLE_DEGREES)))
        for k in range(chord_count + 1):
            av, ah = math.pi / 2 * k / chord_count, self.geom.sweep * k / chord_count
            if z + v * (1.0 - math.sin(av)) < top:
                points.append(_lerp(self.descent_end, self.column_top, math.cos(av)))
            if k > 0:
                points.append(_lerp(self.descent_end, self.approach, k / chord_count))
                points.append(self._horizontal(self.approach, self.start, ah))
        return _clear(points, walls, radius)

    # port: strategies/contour2d.rs:430-449 Lead::departure_fits
    def departure_fits(self, walls: Sequence[Point2], radius: float, z: float, top: float) -> bool:
        points = []
        v = self.geom.vertical_radius
        chord_count = max(1, math.ceil(self.geom.sweep / math.radians(DEFAULT_CHORD_ANGLE_DEGREES)))
        for k in range(chord_count + 1):
            av, ah = math.pi / 2 * k / chord_count, self.geom.sweep * k / chord_count
            if k > 0:
                points.append(self._horizontal(self.start, self.departure, ah))
                points.append(_lerp(self.departure, self.departure_ascent_start, k / chord_count))
            if k < chord_count and z + v * (1.0 - math.cos(av)) < top:
                points.append(_lerp(self.departure_ascent_start, self.departure_top, math.sin(av)))
        return _clear(points, walls, radius)


# port: strategies/contour2d.rs:545-604 select_pass_ends
def select_pass_ends(pass_: PassGeometry, root_tangential: bool, walls: Sequence[Point2],
                     radius: float, z: float, top: float, right_compensation: bool,
                     lead_in: LeadGeom, lead_out: LeadGeom) -> StartSelection | None:
    def side(t):
        return (t[1], -t[0]) if right_compensation else (-t[1], t[0])
    if root_tangential:
        first = pass_.pieces[0]
        tangent = first.tangent_at(0.0)
        normal = side(tangent)
        lead = Lead.build(pass_.root, tangent, normal, lead_in)
        if (lead.entry_fits(walls, radius, z, top)
                and Lead.build(pass_.root, tangent, normal, lead_out).departure_fits(walls, radius, z, top)):
            return StartSelection(pass_.root, tangent, normal, 0, 0.0)
    for index, piece in enumerate(pass_.pieces):
        point, tangent = piece.point_at(0.5), piece.tangent_at(0.5)
        normal = side(tangent)
        if (Lead.build(point, tangent, normal, lead_in).entry_fits(walls, radius, z, top)
                and Lead.build(point, tangent, normal, lead_out).departure_fits(walls, radius, z, top)):
            return StartSelection(point, tangent, normal, index, 0.5)
    return None


# port: strategies/contour2d.rs:1146-1217 lead_geom; scalar mm adapter explicit
def lead_geom(options: Mapping[str, Any], radius: float, prefix: str) -> LeadGeom:
    r = _number(options.get(prefix + "Radius", radius), prefix + "Radius", positive=True)
    v = _number(options.get(prefix + "VerticalRadius", r), prefix + "VerticalRadius", positive=True)
    distance = _number(options.get(prefix + "Distance", r / 2.0), prefix + "Distance", positive=True)
    degrees = _number(options.get(prefix + "SweepDeg", 90.0), prefix + "SweepDeg", positive=True)
    if degrees >= 360.0:
        raise ValueError(f"{prefix}SweepDeg must be in (0,360)")
    return LeadGeom(r, distance, v, math.radians(degrees))


def _push_arc(emitter: _Emitter, start: Point3, centre: Point3, end: Point3, sweep: float,
              tag: str, feed: float, chord_angle_degrees: float) -> None:
    """push_lead_arc/push_piece adapted to complete finite chords (contour2d:998-1081)."""
    r0, r1 = math.dist(start, centre), math.dist(end, centre)
    if r0 <= 0.0 or abs(r0 - r1) > 1e-8 * max(1.0, r0):
        raise ValueError("arc endpoints must share a positive centre radius")
    count = max(1, math.ceil(abs(sweep) / math.radians(chord_angle_degrees)))
    for k in range(1, count + 1):
        angle = abs(sweep) * k / count
        if k == count:
            next_point = end
        elif abs(abs(sweep) - math.pi / 2) <= 1e-12:
            next_point = quarter_point(centre, start, end, angle)
        else:
            # Horizontal non-default sweeps; vertical lead remains a quarter.
            a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
            angle_signed = sweep * k / count
            next_point = (centre[0] + r0 * math.cos(a0 + angle_signed),
                          centre[1] + r0 * math.sin(a0 + angle_signed), start[2])
        emitter.push_linear(C.SegmentKind.FEED, next_point, tag, feed)


_CONTOUR_OPTIONS = {"leadInRadius", "leadInVerticalRadius", "leadInDistance", "leadInSweepDeg",
                    "leadOutRadius", "leadOutVerticalRadius", "leadOutDistance", "leadOutSweepDeg",
                    "feed_mm_s", "plunge_mm_s"}


def emit_contour_loop(task: C.TaskSpec, points: Sequence[Sequence[float]], *, spec: Any = None,
                      contour_z: float, right_compensation: bool = False,
                      start_machine_mm: Sequence[float] | None = None,
                      chord_angle_degrees: float = DEFAULT_CHORD_ANGLE_DEGREES,
                      max_segments: int = DEFAULT_MAX_SEGMENTS,
                      additional_walls: Sequence[Sequence[Point2]] = (),
                      segment_id_prefix="contour2d-loop") -> tuple[C.Segment, ...]:
    """Ported single closed rectangular offset contour + full entry/exit lead.

    Returns the emitted C.Segment tuple, not a MotionPlan. For append/Pocket2D
    finish use, pass ``start_machine_mm=<prefix's final machine end>``; the
    destination position remains the responsibility of that prefix. Returned
    IDs start at ``contour2d-loop-000``; choose ``segment_id_prefix`` to avoid
    collisions on concatenation. ``additional_walls`` lists other independent
    closed rectangles to include in lead-clearance checks. The final retract
    is at derived.home_z. No native arcs are used.
    This emitter rejects lead collision rather than skipping a requested loop.
    It does not clear a lower-step pad's cap: use a separate suitable cap path.
    """
    _validate_context(task)
    options, derived = _custom_spec(spec, task, _CONTOUR_OPTIONS)
    geometry = _task_geometry(task, derived)
    if not isinstance(right_compensation, bool):
        raise ValueError("right_compensation must be boolean")
    vertices = _rectangle(points)
    angle = _number(chord_angle_degrees, "chord_angle_degrees", positive=True)
    if angle > 90.0:
        raise ValueError("coarse chord angle must be at most 90 degrees")
    max_segments = _cap(max_segments, "cam.contour2d")
    radius = float(task.tool.cutter_radius_mm)
    z = _number(contour_z, "contour_z")
    if not geometry.stock_lo[2] <= z <= geometry.top:
        raise ValueError("contour_z outside the declared stock depth")
    _number(options.get("feed_mm_s", derived.feed_mm_s), "feed_mm_s", positive=True)
    derived = dataclasses.replace(derived,
                                  feed_mm_s=_number(options.get("feed_mm_s", derived.feed_mm_s),
                                                     "feed_mm_s", positive=True),
                                  plunge_mm_s=_number(options.get("plunge_mm_s", derived.plunge_mm_s),
                                                       "plunge_mm_s", positive=True))
    wall_points = [vertices]
    for extra in additional_walls:
        wall_points.append(_rectangle(extra))
    walls = [(*loop, loop[0]) for loop in wall_points]
    geometry_pass, root_gap = _offset_contour(vertices, radius, right_compensation)
    in_lead, out_lead = lead_geom(options, radius, "leadIn"), lead_geom(options, radius, "leadOut")
    selected = select_pass_ends(geometry_pass, root_gap, walls, radius, z, geometry.top,
                                right_compensation, in_lead, out_lead)
    if selected is None:
        raise ValueError("contour lead collision: complete loop refused, not skipped or clipped")
    lead = Lead.build(selected.point, selected.tangent, selected.normal, in_lead)
    out = Lead.build(selected.point, selected.tangent, selected.normal, out_lead)
    emitter = _Emitter(task, derived, segment_id_prefix, start_machine_mm)
    safe_top = geometry.top + 1.0
    rapid_bottom = max(z + in_lead.vertical_radius + 1.0, safe_top)
    emitter.push_linear(C.SegmentKind.RAPID, (*lead.column_top, rapid_bottom),
                        "entry-rapid", None)
    emitter.push_linear(C.SegmentKind.FEED, (*lead.column_top, z + in_lead.vertical_radius),
                        "plunge", derived.plunge_mm_s)
    _push_arc(emitter, (*lead.column_top, z + in_lead.vertical_radius),
              (*lead.descent_end, z + in_lead.vertical_radius), (*lead.descent_end, z),
              math.pi / 2, "lead-in-vertical", derived.feed_mm_s, angle)
    emitter.push_linear(C.SegmentKind.FEED, (*lead.approach, z), "lead-in-flat", derived.feed_mm_s)
    _push_arc(emitter, (*lead.approach, z), (*lead.centre, z), (*selected.point, z),
              in_lead.sweep, "lead-in-arc", derived.feed_mm_s, angle)
    for piece in geometry_pass.reseated(selected.piece, selected.fraction):
        if piece.centre is None:
            emitter.push_linear(C.SegmentKind.FEED, (*piece.to_, z), "finish-cutting", derived.feed_mm_s)
        else:
            _push_arc(emitter, (*piece.from_, z), (*piece.centre, z), (*piece.to_, z),
                      piece.sweep, "finish-cutting", derived.feed_mm_s, angle)
    _push_arc(emitter, (*selected.point, z), (*out.centre, z), (*out.departure, z),
              out_lead.sweep, "lead-out-arc", derived.feed_mm_s, angle)
    emitter.push_linear(C.SegmentKind.FEED, (*out.departure_ascent_start, z),
                        "lead-out-flat", derived.feed_mm_s)
    _push_arc(emitter, (*out.departure_ascent_start, z),
              (*out.departure_ascent_start, z + out_lead.vertical_radius),
              (*out.departure_top, z + out_lead.vertical_radius), math.pi / 2,
              "lead-out-vertical", derived.feed_mm_s, angle)
    if z + out_lead.vertical_radius <= geometry.top:
        emitter.push_linear(C.SegmentKind.FEED, (*out.departure_top, safe_top),
                            "lead-out-safe-lift", derived.feed_mm_s)
    emitter.push_linear(C.SegmentKind.RAPID, (*out.departure_top, derived.home_z),
                        "exit-rapid", None)
    _require_count(emitter.segments, max_segments, "cam.contour2d emitter")
    return tuple(emitter.segments)


def contour2d_seed(task: C.TaskSpec, spec: Any = None, *, max_segments=DEFAULT_MAX_SEGMENTS) -> C.MotionPlan:
    """Outer pocket-wall finish plus optional square retained-pad finish.

    Outer: CCW wall with left compensation => sharp inward offset. Retained pad:
    clockwise boundary with left compensation => exact rolled outside offset.
    Those are input-semantic choices, not a winding canonicalization. Lower-pad
    contours finish its below-cap wall; the cap is explicitly NOT covered here.
    Coordinator can use emit_contour_loop as a Pocket2D finishing append stage.
    """
    _validate_context(task)
    _, derived = _custom_spec(spec, task, _CONTOUR_OPTIONS)
    geom = _task_geometry(task, derived)
    radius = float(task.tool.cutter_radius_mm)
    # In the bounded task adapter, this EXPLICIT tuning fits the default
    # lead inside thin rim/pad gaps. The low-level emitter retains Rust's
    # radius=radius, vertical radius=radius engine defaults for its vectors.
    profile = {"leadInRadius": radius / 2.0, "leadOutRadius": radius / 2.0,
               "leadInVerticalRadius": radius / 4.0, "leadOutVerticalRadius": radius / 4.0}
    if isinstance(spec, Mapping):
        profile.update(spec)
    outer = _rect_bounds(geom.pocket_lo, geom.pocket_hi)
    segments = list(emit_contour_loop(task, outer, spec=profile, contour_z=geom.pocket_lo[2],
                                      right_compensation=False, max_segments=max_segments,
                                      segment_id_prefix="contour2d-outer",
                                      additional_walls=((_rect_bounds(geom.pad_lo, geom.pad_hi),)
                                                         if geom.retained else ())))
    if geom.retained:
        pad = _rect_bounds(geom.pad_lo, geom.pad_hi, clockwise=True)
        more = emit_contour_loop(task, pad, spec=profile, contour_z=geom.pocket_lo[2],
                                 right_compensation=False, start_machine_mm=segments[-1].end_mm,
                                 max_segments=max_segments,
                                 segment_id_prefix="contour2d-pad",
                                 additional_walls=(outer,))
        segments.extend(more)
    notes = ["outer task pocket CCW + left compensation -> inward offset",
             "retained pad CW + left compensation -> rolled outside offset",
             f"coarse chord sampling max {DEFAULT_CHORD_ANGLE_DEGREES} degrees per arc chord",
             "safe feed lift before rapid when the vertical lead ends below stock top",
             "coarse task profile lead radius R/2, vertical radius R/4; lower-level defaults remain Rust R",
             "each closed boundary is independently closed for lead-clearance checks; no inter-loop wall",
             "lead clearance checks the actual emitted chord segments, not only their parent arc samples"]
    if geom.retained and geom.pad_hi[2] < geom.top:
        notes.append("lower-step pad cap is NOT covered by this finishing contour; a separate face/cap is required")
    if geom.retained:
        notes.append("contour2d-pad")
    return _plan(task, segments, "cam.contour2d", profile,
                 derived, max_segments, notes)


__all__ = ["face_seed", "contour2d_seed", "emit_contour_loop", "depth_ladder", "stepover_data",
           "PolyLoop", "Region", "Band", "Row", "FacePlan", "Piece", "PassGeometry",
           "LeadGeom", "Lead", "StartSelection", "RUSTCNC_SHA", "FACE_SHA256", "CONTOUR_SHA256"]
