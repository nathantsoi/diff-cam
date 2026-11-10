"""Production, chronological hard evaluator for strict posted RS274/Haas motion.

This is production **geometry** evaluation, not a machining safety certificate.
It supports analytic box stock and flat, axis-vertical tool assemblies. Operation/original
geometry may be analytic or a hash-checked float64 field on the exact task grid;
there is no interpolation or resampling for sampled fields in this module. The
initial pose does not cut. Feed/arc cutter poses remove their closed cylinder;
shank and holder are then checked against stock remaining at that point.
Rapids never cut. Conservative whole-cell interval bounds include stock-bearing
boundary cells and exclude only wholly covered previous cutter cells.

Arcs are evaluated on their typed circular/helical trajectory, using
``contracts.arc_geometry`` for radius, signed sweep and length. Every motion
interval has an explicit conservative nearest-sampled-station displacement
bound, including multi-turn and full-circle arcs. A sample is not an exact continuous collision
certificate. Grid metrics use the shared ``simulator.csg_metrics`` library.
"""

from __future__ import annotations

import dataclasses
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from benchmarks import contracts as C
from cam.gcode_parser import GCodeError, parse_motion_plan
from simulator import csg_metrics
from simulator.csg_metrics import compute_grid_metrics

EVALUATOR_ID = "posted_hard/production"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PLANE_AXES = {C.ArcPlane.XY: (0, 1, 2), C.ArcPlane.XZ: (2, 0, 1), C.ArcPlane.YZ: (1, 2, 0)}


@dataclass(frozen=True)
class RefinementSettings:
    """Sampling controls; these do not alter task/tool/source identity."""

    max_step_mm: Optional[float] = None
    subdivisions: Optional[int] = None
    compute_bounds: bool = True
    safety_screen: bool = True
    expected_nc_hash: Optional[str] = None
    max_samples_per_segment: int = 4096
    exclusion_tolerance_mm3: float = 0.0
    record_stations: bool = False   # full per-station list in discretization (tests/debug; large)

    @classmethod
    def from_mapping(cls, settings: Optional[Mapping[str, Any]]) -> "RefinementSettings":
        raw = {} if settings is None else dict(settings)
        unknown = set(raw) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown posted-evaluator settings: {sorted(unknown)}")
        result = cls(**raw)
        for name in ("compute_bounds", "safety_screen", "record_stations"):
            if not isinstance(getattr(result, name), bool):
                raise ValueError(f"{name} must be boolean")
        if result.max_step_mm is not None and (
            not isinstance(result.max_step_mm, (int, float))
            or not math.isfinite(result.max_step_mm) or result.max_step_mm <= 0
        ):
            raise ValueError("max_step_mm must be finite and positive")
        if result.subdivisions is not None and (
            not isinstance(result.subdivisions, int) or result.subdivisions < 1
        ):
            raise ValueError("subdivisions must be a positive integer")
        if not isinstance(result.max_samples_per_segment, int) or result.max_samples_per_segment < 2:
            raise ValueError("max_samples_per_segment must be an integer >= 2")
        if (not isinstance(result.exclusion_tolerance_mm3, (int, float))
                or not math.isfinite(result.exclusion_tolerance_mm3) or result.exclusion_tolerance_mm3 < 0):
            raise ValueError("exclusion_tolerance_mm3 must be finite and nonnegative")
        if result.expected_nc_hash is not None:
            C._check_hash("settings.expected_nc_hash", result.expected_nc_hash)
        return result


@dataclass
class _Execution:
    remaining: np.ndarray
    initial: np.ndarray
    metrics: Dict[str, C.Metric]
    discretization: Dict[str, Any]
    safety_pass: Optional[bool]
    coverage_reasons: Tuple[str, ...]
    events: Tuple[C.SegmentEvent, ...]


def _centres(grid: C.GridSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [grid.origin_mm[a] + (np.arange(grid.shape[a], dtype=np.float64) + 0.5) * grid.voxel_mm
            for a in range(3)]
    return tuple(np.meshgrid(*axes, indexing="ij"))


def _box_distance(points, lo, hi):
    try:
        lower, upper = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("analytic box requires three finite numeric min/max coordinates") from error
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("analytic box requires three finite numeric min/max coordinates")
    if np.any(upper <= lower):
        raise ValueError("analytic box min_mm must be strictly less than max_mm in every axis")
    q = [np.maximum(lower[a] - points[a], points[a] - upper[a]) for a in range(3)]
    positive = [np.maximum(value, 0.0) for value in q]
    return np.sqrt(positive[0] ** 2 + positive[1] ** 2 + positive[2] ** 2) + np.minimum(
        np.maximum.reduce(q), 0.0)


def _cylinder_distance(points, base, radius: float, height: float):
    """Exact static capped-cylinder SDF in mm, not a sweep surrogate."""
    radial = np.hypot(points[0] - base[0], points[1] - base[1]) - radius
    vertical = np.maximum(base[2] - points[2], points[2] - (base[2] + height))
    return np.hypot(np.maximum(radial, 0.0), np.maximum(vertical, 0.0)) + np.minimum(
        np.maximum(radial, vertical), 0.0)


def _body_distances(points, base, tool) -> Dict[str, Optional[np.ndarray]]:
    shank_height = tool.holder_bottom_mm - tool.flute_length_mm
    return {
        "cutter": _cylinder_distance(points, base, tool.cutter_radius_mm, tool.flute_length_mm),
        "shank": None if shank_height <= 0 else _cylinder_distance(
            points, (base[0], base[1], base[2] + tool.flute_length_mm), tool.shank_radius_mm, shank_height),
        "holder": _cylinder_distance(
            points, (base[0], base[1], base[2] + tool.holder_bottom_mm),
            tool.holder_radius_mm, tool.holder_height_mm),
    }


def _work_point(task, point, frame):
    return tuple(point) if frame == C.Frame.WORK else task.work_to_machine.inverse().apply(point)


def _arc_data(segment, task, frame) -> Tuple[C.ArcData, Dict[str, float], float, float]:
    arc = segment.arc
    if frame == C.Frame.MACHINE:
        _, _, normal = _PLANE_AXES[arc.plane]
        rotation = task.work_to_machine.inverse().matrix
        mapped = [rotation[i][normal] for i in range(3)]
        for plane in C.ArcPlane:
            _, _, index = _PLANE_AXES[plane]
            if abs(abs(mapped[index]) - 1) <= 1e-9 and all(
                abs(mapped[i]) <= 1e-9 for i in range(3) if i != index
            ):
                direction = arc.direction
                if mapped[index] < 0:
                    direction = C.ArcDirection.CW if direction == C.ArcDirection.CCW else C.ArcDirection.CCW
                arc = dataclasses.replace(arc, centre_mm=_work_point(task, arc.centre_mm, frame),
                                          plane=plane, direction=direction)
                break
        else:
            raise ValueError("rigid transform rotates arc normal outside a principal work-frame plane")
    start = _work_point(task, segment.start_mm, frame)
    end = _work_point(task, segment.end_mm, frame)
    geometry = C.arc_geometry(start, end, arc, segment.id)
    a, b, _ = _PLANE_AXES[arc.plane]
    r0 = math.hypot(start[a] - arc.centre_mm[a], start[b] - arc.centre_mm[b])
    r1 = math.hypot(end[a] - arc.centre_mm[a], end[b] - arc.centre_mm[b])
    return arc, geometry, r0, r1


def _arc_point(segment, s: float, task, frame):
    """Evaluate the actual arc, not a chord between its typed endpoints."""
    arc, geometry, r0, r1 = _arc_data(segment, task, frame)
    start = _work_point(task, segment.start_mm, frame)
    end = _work_point(task, segment.end_mm, frame)
    a, b, normal = _PLANE_AXES[arc.plane]
    theta0 = math.atan2(start[b] - arc.centre_mm[b], start[a] - arc.centre_mm[a])
    theta = theta0 + s * geometry["sweep_rad"]
    radius = r0 + s * (r1 - r0)
    result = list(start)
    result[a] = arc.centre_mm[a] + radius * math.cos(theta)
    result[b] = arc.centre_mm[b] + radius * math.sin(theta)
    result[normal] = start[normal] + s * (end[normal] - start[normal])
    return tuple(result)


def _base_at(segment, s: float, task, frame):
    if segment.kind == C.SegmentKind.ARC:
        return _arc_point(segment, s, task, frame)
    point = tuple(segment.start_mm[a] + s * (segment.end_mm[a] - segment.start_mm[a]) for a in range(3))
    return _work_point(task, point, frame)


def _arc_interval_bound(segment, subdivisions: int, task, frame) -> float:
    """Bound distance from any pose to its nearest sampled station.

    This is NOT arc-to-chord deviation. If angular/radial/normal translation
    speed is bounded by V, every point in an h=1/m subinterval is within
    V*h/2 of at least one endpoint. For the interpolated arc, that yields the
    following conservative bound, valid for minor, major, helical and full arcs.
    """
    arc, geometry, r0, r1 = _arc_data(segment, task, frame)
    start = _work_point(task, segment.start_mm, frame)
    end = _work_point(task, segment.end_mm, frame)
    _, _, normal = _PLANE_AXES[arc.plane]
    angular_and_normal = math.hypot(
        max(r0, r1) * abs(geometry["sweep_rad"]), end[normal] - start[normal])
    return (angular_and_normal + abs(r1 - r0)) / (2 * subdivisions)


def _station_summary(segment, m, zero_length, task, frame, record_all=False):
    stations = [_base_at(segment, 0.0 if zero_length else j / m, task, frame)
                for j in range(1 if zero_length else m + 1)]
    rounded = [[round(float(c), 9) for c in p] for p in stations]
    summary = {"station_count": len(rounded), "first_station_work_mm": rounded[0],
               "last_station_work_mm": rounded[-1], "stations_sha256": C.content_hash(rounded)}
    if record_all:
        summary["stations_work_mm"] = [list(p) for p in stations]
    return summary


def _subdivisions(segment, task, cfg):
    if cfg.subdivisions is not None:
        count = cfg.subdivisions
    else:
        step = cfg.max_step_mm or min(task.grid.voxel_mm / 2,
                                     task.tool.cutter_radius_mm / 4,
                                     task.tool.flute_length_mm / 4)
        count = max(1, math.ceil(segment.path_length_mm / step))
        count = 2 ** max(0, count - 1).bit_length()
        if segment.kind == C.SegmentKind.ARC:
            # Arclength refinement alone does not make a curved interval as
            # tight as a line of the same length. Add one binary subdivision.
            count *= 2
    stations = 1 if segment.path_length_mm == 0 else count + 1
    if stations > cfg.max_samples_per_segment:
        raise ValueError(f"segment {segment.id} requires {stations} stations; limit is {cfg.max_samples_per_segment}")
    return count


def _station_time_weight(segment, subdivisions, index, zero_length):
    """Assign segment duration to this station by trapezoidal integration.

    The first/last stations of each nonzero-length segment have half weight;
    all other stations have full weight. These partition the exact same
    segment-duration denominator reported as cycle_time_s.
    """
    if zero_length:
        return float(segment.duration_s)
    return float(segment.duration_s * (0.5 if index in (0, subdivisions) else 1.0)
                 / subdivisions)


def _can_inherit_engagement(previous, current):
    """Whether two adjoining cutting motions may share endpoint engagement."""
    return (previous is not None
            and previous.kind != C.SegmentKind.RAPID
            and current.kind != C.SegmentKind.RAPID
            and math.dist(previous.end_mm, current.start_mm)
                <= C.MotionPlan.CONTIGUITY_TOL_MM)


def _analytic_mask(geometry, points):
    if geometry.kind != "analytic" or geometry.frame != C.Frame.WORK:
        raise ValueError("only analytic operation/part geometry in the work frame is supported")
    p = geometry.params
    if p.get("primitive") == "box":
        return _box_distance(points, p["min_mm"], p["max_mm"]) <= 0
    if p.get("primitive") == "box_minus_pocket":
        outside = _box_distance(points, p["box_min_mm"], p["box_max_mm"])
        pocket = _box_distance(points, p["pocket_min_mm"], p["pocket_max_mm"])
        island = _box_distance(points, p["island_min_mm"], p["island_max_mm"])
        sharp = np.maximum(outside, -np.maximum(pocket, -island)) <= 0
        try:
            radius = float(p.get("pocket_unreached_corner_radius_mm", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError("pocket_unreached_corner_radius_mm must be finite and nonnegative") from error
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("pocket_unreached_corner_radius_mm must be finite and nonnegative")
        if radius == 0.0:
            return sharp
        pocket_min, pocket_max = p["pocket_min_mm"], p["pocket_max_mm"]
        if any(pocket_max[axis] - pocket_min[axis] < 2 * radius for axis in (0, 1)):
            raise ValueError("unreached-corner radius is too large for this pocket; model the complete uncut pocket explicitly")
        unreachable = np.zeros(sharp.shape, dtype=bool)
        for x_side in (0, 1):
            for y_side in (0, 1):
                cx = pocket_min[0] + radius if x_side == 0 else pocket_max[0] - radius
                cy = pocket_min[1] + radius if y_side == 0 else pocket_max[1] - radius
                xlo = pocket_min[0] if x_side == 0 else pocket_max[0] - radius
                xhi = pocket_min[0] + radius if x_side == 0 else pocket_max[0]
                ylo = pocket_min[1] if y_side == 0 else pocket_max[1] - radius
                yhi = pocket_min[1] + radius if y_side == 0 else pocket_max[1]
                in_corner = ((outside <= 0)
                             & (points[0] >= xlo) & (points[0] <= xhi)
                             & (points[1] >= ylo) & (points[1] <= yhi)
                             & (points[2] >= pocket_min[2]) & (points[2] <= pocket_max[2]))
                unreachable |= in_corner & (
                    (points[0] - cx) ** 2 + (points[1] - cy) ** 2 >= radius ** 2)
        # Only concave pocket corners change. The original outer stock and
        # convex island are retained exactly as they are in the sharp source.
        return sharp | unreachable
    raise ValueError(f"unsupported analytic geometry primitive {p.get('primitive')!r}")


def _sampled_mask(geometry, task_grid):
    """Load a checked, exact-grid sampled field without interpolation.

    The SHA-256 covers the exact bytes loaded, not a separate later read. Values
    stay float64 for the hard geometry metrics; the simulator's float32 runtime
    representation is not used as a substitute here. Cell-centre membership is
    ``phi <= 0`` in the declared millimetre ``negative_inside`` convention.
    """
    if (
        geometry.frame != C.Frame.WORK
        or task_grid.frame != C.Frame.WORK
        or geometry.grid.frame != C.Frame.WORK
    ):
        raise ValueError("sampled operation/part source and grid must use the work frame")
    if geometry.grid != task_grid:
        raise ValueError(
            "sampled geometry GridSpec must exactly match task grid "
            "(shape, origin, voxel, frame); resampling is unsupported"
        )
    if geometry.grid.sign_convention != "negative_inside":
        raise ValueError("sampled geometry must use the negative_inside sign convention")
    path = Path(geometry.path)
    if not path.is_absolute():
        # Runner paths are repository-relative; never depend on invocation CWD.
        path = (PROJECT_ROOT / path).resolve()
    try:
        size = path.stat().st_size
        # A simple numeric NPY header is tiny. The allowance also covers a
        # wider float128-like scalar kind, while rejecting unrelated gigabyte
        # files before loading an unchecked/bogus array into memory.
        byte_limit = math.prod(task_grid.shape) * 16 + 65_536
        if size > byte_limit:
            raise ValueError(
                f"sampled payload is {size} bytes, exceeds exact-grid byte limit {byte_limit}"
            )
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read sampled geometry payload {path}: {error}") from error
    actual_hash = C.sha256_bytes(data)
    if actual_hash != geometry.sha256:
        raise ValueError(
            f"sampled geometry SHA-256 mismatch: declared {geometry.sha256}, actual {actual_hash}"
        )
    try:
        values = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"cannot decode sampled geometry as a non-pickle NumPy array: {error}") from error
    if not isinstance(values, np.ndarray):
        close = getattr(values, "close", None)
        if callable(close):
            close()
        raise ValueError("sampled geometry must be one numeric .npy array, not an NPZ/archive")
    if values.dtype.kind not in "fiu":
        raise ValueError(f"sampled geometry dtype {values.dtype} is not a real numeric scalar field")
    if values.shape != task_grid.shape:
        raise ValueError(
            f"sampled payload shape {values.shape} != task grid shape {task_grid.shape}"
        )
    values = np.ascontiguousarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("sampled geometry contains non-finite values (NaN or infinity)")
    return values <= 0.0


def _geometry_mask(geometry, points, task_grid):
    if geometry.kind == "analytic":
        return _analytic_mask(geometry, points)
    if geometry.kind == "sampled":
        return _sampled_mask(geometry, task_grid)
    raise ValueError(
        f"unsupported operation/part geometry kind {geometry.kind!r}; "
        "expected analytic primitives or an exact-grid sampled field"
    )


def _geometry_masks(task, points):
    """Load independent original-P and operation-T masks and enforce P ⊆ T."""
    try:
        target = _geometry_mask(task.operation_target.geometry, points, task.grid)
    except ValueError as error:
        raise ValueError(f"operation_target: {error}") from error
    try:
        original = _geometry_mask(task.original_part, points, task.grid)
    except ValueError as error:
        raise ValueError(f"original_part: {error}") from error
    if np.any(original & ~target):
        raise ValueError(
            "original_part must be a subset of operation_target at task-grid cell centres (P ⊆ T)"
        )
    return target, original


def _unsupported_reasons(task, motion, points):
    reasons = []
    if not task.initial_state.known or not motion.initial_state.known:
        reasons.append("unknown initial tool state")
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        reasons.append("ball/non-flat tool geometry is unsupported")
    if task.fixtures:
        reasons.append("separate workholding fixtures are unsupported")
    if task.stock.kind != "analytic" or task.stock.params.get("primitive") != "box":
        reasons.append("only analytic box stock is supported")
    if task.stock.frame != C.Frame.WORK:
        reasons.append("stock must use the work frame")
    if task.motion_semantics.timing_rule != motion.timing_rule:
        reasons.append("motion and task timing rules differ")
    matrix = task.work_to_machine.matrix
    if abs(matrix[2][0]) > 1e-9 or abs(matrix[2][1]) > 1e-9 or abs(matrix[2][2] - 1.0) > 1e-9:
        reasons.append("work-to-machine rotation must keep work/tool +Z aligned with machine +Z")
    try:
        _geometry_masks(task, points)
        for segment in motion.segments:
            if segment.kind == C.SegmentKind.ARC:
                _arc_data(segment, task, motion.frame)
    except ValueError as error:
        reasons.append(str(error))
    return reasons


def _unknown_metrics(reason):
    units = dict(C.METRIC_UNITS)
    units.update({
        "min_clearance_lower_bound_mm": "mm",
        "min_rapid_cutter_clearance_lower_bound_mm": "mm",
        "rapid_cutter_penetration_upper_bound_mm": "mm",
        "air_cut_time_s": "s",
        "rapid_time_s": "s",
    })
    return {name: C.Metric(None, unit, null_reason=reason) for name, unit in units.items()}


def _error_plan(task):
    frame = task.initial_state.frame if task.initial_state.known else C.Frame.MACHINE
    return C.MotionPlan(frame, task.initial_state, (), task.motion_semantics.timing_rule)


def _unsupported(task, motion, settings, reasons, *, nc_bytes=None, dialect=None):
    raw_settings = dataclasses.asdict(settings) if isinstance(settings, RefinementSettings) else dict(settings)
    return C.EvaluationResult(
        evaluator_id=EVALUATOR_ID, evaluator_revision=_revision(),
        task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
        discretization={"settings": raw_settings}, coverage=C.Coverage.UNSUPPORTED,
        coverage_reasons=tuple(reasons), metrics=_unknown_metrics("; ".join(reasons)),
        safety_pass=None, geometry_pass=None,
        motion_hash=None if nc_bytes is not None else motion.motion_hash,
        nc_hash=None if nc_bytes is None else C.nc_hash(nc_bytes), dialect=dialect,
    )


def _execute(task, motion, cfg):
    lo, hi = task.stock.params["min_mm"], task.stock.params["max_mm"]
    grid_lo, grid_hi = task.grid.bounds_mm
    if len(lo) != 3 or len(hi) != 3 or any(a >= b for a, b in zip(lo, hi)):
        raise ValueError("stock min_mm must precede stock max_mm")
    if any(lo[a] < grid_lo[a] - 1e-9 or hi[a] > grid_hi[a] + 1e-9 for a in range(3)):
        raise ValueError("stock extends beyond represented grid; whole-cell safety is unavailable")

    points = _centres(task.grid)
    stock_d = _box_distance(points, lo, hi)
    initial = stock_d <= 0
    stock_interior = stock_d < 0
    remaining = initial.copy()
    rho = math.sqrt(3) * task.grid.voxel_mm / 2
    voxel_volume = task.grid.voxel_mm ** 3
    guaranteed = np.zeros(task.grid.shape, dtype=bool)
    definite = np.zeros(task.grid.shape, dtype=bool)
    station_upper = np.zeros(task.grid.shape, dtype=bool)
    continuous_upper = np.zeros(task.grid.shape, dtype=bool)
    unique_by_component = {name: np.zeros(task.grid.shape, dtype=bool)
                           for name in ("rapid_cutter", "cutter", "shank", "holder")}
    seen_events = set()
    events = []
    segment_records = []
    uncertain_segments = []
    collision_count = 0
    integrated = 0.0
    min_clearance = math.inf
    min_clearance_lower = math.inf
    min_rapid_clearance = math.inf
    min_rapid_lower = math.inf
    rapid_upper = 0.0
    rapid_seen = False
    safety_uncertain = False
    tolerance = task.tolerances.rapid_penetration_mm
    air_cut_time_s = 0.0
    rapid_time_s = sum(segment.duration_s for segment in motion.segments
                       if segment.kind == C.SegmentKind.RAPID)
    previous_segment = None
    last_station_engaged = None

    for segment in motion.segments:
        contiguous_cut_join = _can_inherit_engagement(previous_segment, segment)
        m = _subdivisions(segment, task, cfg)
        zero_length = segment.path_length_mm == 0
        delta_t = (_arc_interval_bound(segment, m, task, motion.frame)
                   if segment.kind == C.SegmentKind.ARC
                   else 0.0 if zero_length else segment.path_length_mm / (2 * m))
        segment_records.append({
            "id": segment.id, "kind": segment.kind.value, "length_mm": segment.path_length_mm,
            "subdivisions": m, "stations": 1 if zero_length else m + 1,
            "endpoints_included": True, "duration_s": segment.duration_s,
            # Review E2: record a compact, verifiable summary instead of every station (up to 4096/segment).
            **_station_summary(segment, m, zero_length, task, motion.frame, cfg.record_stations),
            "temporal_station_cover_bound_mm": delta_t,
            "motion_model": "contracts.arc_geometry" if segment.kind == C.SegmentKind.ARC else "linear",
        })
        current = None
        for j in range(1 if zero_length else m + 1):
            s = 0.0 if zero_length else j / m
            if current is None:
                current = _body_distances(points, _base_at(segment, s, task, motion.frame), task.tool)
            if segment.kind != C.SegmentKind.RAPID:
                cutter = current["cutter"]
                # The shared endpoint of a contiguous cutting join inherits the
                # preceding station's engagement instead of being re-tested against
                # stock already removed at that pose. Other stations retain the
                # exact-grid removal test before subtraction. Rapids reset the state.
                engaged = (last_station_engaged if j == 0 and contiguous_cut_join
                           else bool(np.any(remaining & (cutter <= 0))))
                last_station_engaged = engaged
                if not engaged:
                    air_cut_time_s += _station_time_weight(segment=segment,
                        subdivisions=m, index=j, zero_length=zero_length)
                wholly_covered = cutter < -rho
                guaranteed |= wholly_covered
                if cfg.compute_bounds:
                    definite |= (stock_d < -rho) & wholly_covered
                    station_upper |= (stock_d <= rho) & (cutter <= rho)
                    continuous_upper |= (stock_d <= rho) & (cutter <= rho + delta_t)
                remaining[cutter <= 0] = False
                checked = ("shank", "holder")
            else:
                last_station_engaged = None
                checked = ("cutter", "shank", "holder")

            material = remaining & stock_interior
            contact_union = np.zeros(task.grid.shape, dtype=bool)
            min_noncutting = None
            for name in checked:
                distance = current[name]
                if distance is None:
                    continue
                rapid_cutter = segment.kind == C.SegmentKind.RAPID and name == "cutter"
                if rapid_cutter:
                    rapid_seen = True
                    if material.any():
                        min_rapid_clearance = min(min_rapid_clearance, float(distance[material].min()))
                else:
                    min_noncutting = distance.copy() if min_noncutting is None else np.minimum(min_noncutting, distance)
                contact = material & (distance < (-tolerance if rapid_cutter else 0.0))
                if not contact.any():
                    continue
                contact_union |= contact
                component = "rapid_cutter" if rapid_cutter else name
                unique_by_component[component] |= contact
                kind = "rapid_collision" if rapid_cutter else f"{name}_collision"
                if kind not in seen_events:
                    seen_events.add(kind)
                    events.append(C.SegmentEvent(segment.id, kind, s,
                        "first sampled contact beyond the applicable tolerance; not an exact continuous first-contact time"))
            if contact_union.any():
                collision_count += 1
                weight = segment.duration_s if zero_length else segment.duration_s * (0.5 if j in (0, m) else 1.0) / m
                integrated += weight * float(contact_union.sum()) * voxel_volume
            if min_noncutting is not None and material.any():
                min_clearance = min(min_clearance, float(min_noncutting[material].min()))

            possible_here = False
            following = None
            if cfg.safety_screen and (zero_length or j < m):
                if not zero_length:
                    following = _body_distances(points, _base_at(segment, (j + 1) / m, task, motion.frame), task.tool)
                safety_material = (stock_d <= rho) & ~guaranteed
                for name in checked:
                    now = current[name]
                    if now is None or not safety_material.any():
                        continue
                    near = now if zero_length else np.minimum(now, following[name])
                    interval = 0.0 if zero_length else delta_t
                    lower = float(near[safety_material].min()) - rho - interval
                    if segment.kind == C.SegmentKind.RAPID and name == "cutter":
                        min_rapid_lower = min(min_rapid_lower, lower)
                        upper = max(0.0, -lower)
                        rapid_upper = max(rapid_upper, upper)
                        if upper > tolerance:
                            possible_here = True
                    else:
                        min_clearance_lower = min(min_clearance_lower, lower)
                        if lower <= 0:
                            possible_here = True
                if possible_here:
                    safety_uncertain = True
                    if segment.id not in uncertain_segments:
                        uncertain_segments.append(segment.id)
            current = following
        previous_segment = segment

    cycle_time_s = sum(segment.duration_s for segment in motion.segments)
    # At configured exact-grid sampling, trapezoidal station weights partition
    # each cutting segment's duration. No continuous-engagement claim is made.
    if cycle_time_s > 0:
        raw_fraction = air_cut_time_s / cycle_time_s
        # Accumulating endpoint/interior weights can differ from the duration
        # sum by a few ulps at the fully-air endpoint. Reject actual accounting
        # errors; bound only that floating-point endpoint to the physical range.
        if not 0.0 <= raw_fraction <= 1.0 + 1e-12:
            raise ValueError("air_cut_time_s is outside cycle_time_s beyond floating-point tolerance")
        air_fraction = min(1.0, raw_fraction)
    else:
        air_fraction = None
    air_sampling_note = (
        "FEED/ARC cutter stations that remove zero remaining occupied stock cells; "
        "resolution-dependent exact-grid/trapezoidal station time at this evaluation; "
        "a validated contiguous FEED/ARC join inherits its preceding engagement; rapids/discontinuities reset; "
        "rapids excluded from the numerator; the 0.125 mm verdict is authoritative under D-04")

    metrics = {
        "unique_contacted_volume_mm3": C.Metric(
            float(sum((mask.sum() for mask in unique_by_component.values()))) * voxel_volume, "mm^3"),
        "min_clearance_mm": C.Metric(min_clearance, "mm", denominator="centre-sampled only; non-cutting shank+holder versus evolving remaining-stock centres, not a continuous clearance") if math.isfinite(min_clearance) else C.Metric(None, "mm", null_reason="no non-cutting station with remaining stock"),
        "min_rapid_cutter_clearance_mm": C.Metric(min_rapid_clearance, "mm", denominator="centre-sampled only; rapid cutter versus remaining-stock centres") if math.isfinite(min_rapid_clearance) else C.Metric(None, "mm", null_reason="no rapid station with remaining stock"),
        "min_clearance_lower_bound_mm": C.Metric(min_clearance_lower, "mm", denominator="conservative non-cutting bound over stock-bearing whole cells and motion intervals") if math.isfinite(min_clearance_lower) else C.Metric(None, "mm", null_reason="no whole-cell non-cutting clearance sample"),
        "min_rapid_cutter_clearance_lower_bound_mm": C.Metric(min_rapid_lower, "mm", denominator="conservative rapid-cutter distance bound over stock-bearing whole cells and rapids") if math.isfinite(min_rapid_lower) else C.Metric(None, "mm", null_reason="no whole-cell rapid clearance sample"),
        "rapid_cutter_penetration_upper_bound_mm": C.Metric(rapid_upper, "mm", denominator="maximum possible rapid penetration over stock-bearing whole cells and rapid intervals") if rapid_seen and cfg.safety_screen else C.Metric(None, "mm", null_reason="no rapid segments" if not rapid_seen else "safety_screen=false"),
        "collision_sample_count": C.Metric(collision_count, "count", denominator="stations with one or more contacted occupied cells"),
        "time_integrated_overlap_mm3_s": C.Metric(integrated, "mm^3*s", denominator="stationwise trapezoidal integration of union-of-components contacted cells"),
        "cycle_time_s": C.Metric(cycle_time_s, "s", denominator="sum of typed motion segment durations"),
        "air_cut_time_s": C.Metric(air_cut_time_s, "s", denominator=air_sampling_note),
        "rapid_time_s": C.Metric(rapid_time_s, "s", denominator="sum of typed RAPID segment durations within cycle_time_s; never included in the air-cut numerator"),
        "spatial_uncertainty_mm3": C.Metric(float(station_upper.sum() - definite.sum()) * voxel_volume, "mm^3", denominator="station_upper minus guaranteed removed volume") if cfg.compute_bounds else C.Metric(None, "mm^3", null_reason="compute_bounds=false"),
        "temporal_uncertainty_mm3": C.Metric(float(continuous_upper.sum() - station_upper.sum()) * voxel_volume, "mm^3", denominator="continuous interval upper minus sampled-station upper") if cfg.compute_bounds else C.Metric(None, "mm^3", null_reason="compute_bounds=false"),
        "air_fraction": C.Metric(
            air_fraction, "1", denominator="air_cut_time_s / cycle_time_s; " + air_sampling_note,
            null_reason=None if air_fraction is not None else "cycle_time_s is not positive; no finite air-fraction denominator"),
        "force_proxy": C.Metric(None, "proxy", null_reason="geometry-only evaluator: no calibrated force model"),
    }
    unique = np.zeros(task.grid.shape, dtype=bool)
    for mask in unique_by_component.values():
        unique |= mask
    metrics["unique_contacted_volume_mm3"] = C.Metric(float(unique.sum()) * voxel_volume, "mm^3", denominator="spatial union of collided cells, not component sum")
    safety = False if collision_count else (None if safety_uncertain or not cfg.safety_screen else True)
    reasons = ()
    if safety_uncertain and not collision_count:
        reasons = ("conservative stock-bearing-cell clearance/tolerance certificate remains uncertain on segments " + ",".join(uncertain_segments),)
    elif not cfg.safety_screen and not collision_count:
        reasons = ("safety_screen=false; no whole-cell safety certificate",)
    bounds = None if not cfg.compute_bounds else {
        "lower_mm3": float(definite.sum()) * voxel_volume,
        "station_upper_mm3": float(station_upper.sum()) * voxel_volume,
        "upper_mm3": float(continuous_upper.sum()) * voxel_volume,
    }
    discretization = {
        "voxel_mm": task.grid.voxel_mm, "grid_shape": list(task.grid.shape),
        "grid_origin_mm": list(task.grid.origin_mm), "cell_half_diagonal_mm": rho,
        "segments": segment_records, "volume_bound": bounds,
        "unique_by_component_mm3": {name: float(mask.sum()) * voxel_volume for name, mask in unique_by_component.items()},
        "collision_samples_are_per_station": True,
        "clearance_metrics": "centre diagnostics are distinct from conservative whole-cell lower bounds",
        "tolerances": {"rapid_penetration_mm": tolerance,
                       "noncutting_clearance_qualification_mm": task.tolerances.clearance_mm,
                       "clearance_mm_role": "reported qualification only; safety uses strict non-cutting overlap"},
        "execution_order": "feed/arc closed-cutter removal before shank+holder checks against remaining stock; rapid checks all bodies without subtraction",
        "air_cut_timing": {
            "engagement": "at least one remaining occupied cell removed by the cutter at a sampled FEED/ARC pose, checked before subtraction",
            "station_time_weighting": "whole duration for a zero-length pose; otherwise trapezoidal endpoint-half/interior-whole weights summing segment durations",
            "rapid_handling": "rapid_time_s is included in cycle_time_s but excluded from air_cut_time_s; rapids also reset carried engagement",
            "join_engagement": "at j=0 of a validated contiguous FEED/ARC join, inherit the previous segment's final-station engagement instead of re-testing the shared endpoint",
            "contiguity_tolerance_mm": C.MotionPlan.CONTIGUITY_TOL_MM,
            "discontinuity_handling": "without a validated contiguous FEED/ARC join, engagement is recomputed from current remaining stock; no prior engagement is inherited",
            "resolution_dependent": True,
            "authoritative_verdict_max_step_mm": 0.125,
        },
    }
    return _Execution(remaining, initial, metrics, discretization, safety, reasons, tuple(events))


def _result(task, motion, cfg, execution):
    points = _centres(task.grid)
    target, original = _geometry_masks(task, points)
    metrics = compute_grid_metrics(
        execution.remaining, target, execution.initial, task.grid.voxel_mm,
        original_part_mask=original, surface_origin_mm=task.grid.origin_mm,
        declared_excluded_volume_mm3=task.operation_target.excluded_volume_mm3,
        exclusion_tolerance_mm3=cfg.exclusion_tolerance_mm3)
    # Shared library (core 67b7f37): gouge is removed & operation_target;
    # original-part gouge remains a diagnostic with its own separate denominator.
    metrics.update(execution.metrics)
    removed = execution.initial & ~execution.remaining
    removal_target = execution.initial & ~target
    geometry = C.geometry_pass_from_metrics(metrics, task.tolerances)
    if removal_target.any() and not removed.any():
        geometry = False
    return C.EvaluationResult(
        evaluator_id=EVALUATOR_ID, evaluator_revision=_revision(), task_hash=task.task_hash,
        tool_setup_hash=task.tool_setup_hash,
        discretization={"settings": dataclasses.asdict(cfg),
                        "metric_library_sha256": C.sha256_bytes(Path(csg_metrics.__file__).read_bytes()),
                        **execution.discretization},
        coverage=C.Coverage.COMPLETE, coverage_reasons=execution.coverage_reasons,
        metrics=metrics, safety_pass=execution.safety_pass, geometry_pass=geometry,
        motion_hash=motion.motion_hash, events=execution.events)


def _revision():
    return f"posted_evaluator/1+{C.sha256_bytes(Path(__file__).read_bytes())}"


class PostedEvaluator:
    """Stateless implementation of the ``contracts.Evaluator`` protocol."""

    def evaluate_candidate(self, task: C.TaskSpec, motion: C.MotionPlan,
                           settings: Mapping[str, Any]) -> C.EvaluationResult:
        try:
            cfg = RefinementSettings.from_mapping(settings)
        except ValueError as error:
            return _unsupported(task, motion, settings, [str(error)])
        if cfg.expected_nc_hash is not None:
            raise C.ContractError("expected_nc_hash is only valid for posted NC evaluation")
        try:
            motion.validate_for_task(task)
            reasons = _unsupported_reasons(task, motion, _centres(task.grid))
            if reasons:
                return _unsupported(task, motion, cfg, reasons)
            return _result(task, motion, cfg, _execute(task, motion, cfg))
        except ValueError as error:
            return _unsupported(task, motion, cfg, [str(error)])

    def evaluate_posted_gcode(self, task: C.TaskSpec, nc_bytes: bytes, dialect: C.Dialect,
                              settings: Mapping[str, Any]) -> C.EvaluationResult:
        if not isinstance(nc_bytes, bytes):
            raise C.ContractError("nc_bytes must be immutable bytes for exact source hashing")
        try:
            dialect = C.Dialect(dialect)
        except ValueError as error:
            raise C.ContractError(f"unknown NC dialect {dialect!r}") from error
        try:
            cfg = RefinementSettings.from_mapping(settings)
        except ValueError as error:
            return _unsupported(task, _error_plan(task), settings,
                                [str(error)], nc_bytes=nc_bytes, dialect=dialect)
        if cfg.expected_nc_hash is not None and cfg.expected_nc_hash != C.nc_hash(nc_bytes):
            raise C.ContractError("raw NC hash differs from settings.expected_nc_hash")
        try:
            motion = parse_motion_plan(nc_bytes.decode("utf-8"), task, dialect)
            motion.validate_for_task(task)
            reasons = _unsupported_reasons(task, motion, _centres(task.grid))
            if reasons:
                return _unsupported(task, motion, cfg, reasons, nc_bytes=nc_bytes, dialect=dialect)
            base = _result(task, motion, cfg, _execute(task, motion, cfg))
            discretization = dict(base.discretization)
            discretization["parsed_motion_hash"] = motion.motion_hash
            return dataclasses.replace(base, discretization=discretization, nc_hash=C.nc_hash(nc_bytes),
                                       motion_hash=None, dialect=dialect)
        except (GCodeError, ValueError) as error:
            plan = _error_plan(task)
            return _unsupported(task, plan, cfg, [str(error)], nc_bytes=nc_bytes, dialect=dialect)


def evaluate_candidate(task, motion, settings=None):
    return PostedEvaluator().evaluate_candidate(task, motion, settings or {})


def evaluate_posted_gcode(task, nc_bytes, dialect, settings=None):
    return PostedEvaluator().evaluate_posted_gcode(task, nc_bytes, dialect, settings or {})
