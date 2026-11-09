"""Slow, independent flat-cutter carve/chronology reference.

Only shared contract records and NumPy are used here.  In particular, this module
must never import production sweep/SDF helpers.  All geometric calculations are
float64 physical millimetres in the task's work frame.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmarks import contracts as C

EVALUATOR_ID = "reference_carve"
EVALUATOR_REVISION = f"reference_carve/1+{C.sha256_bytes(Path(__file__).read_bytes())}"


@dataclass(frozen=True)
class ReferenceSettings:
    """Discretization controls; these do not alter task or tool hashes."""

    max_step_mm: Optional[float] = None
    subdivisions: Optional[int] = None
    compute_bounds: bool = True
    safety_screen: bool = True

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "ReferenceSettings":
        value = {} if value is None else dict(value)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown reference settings: {unknown}")
        result = cls(**value)
        if result.max_step_mm is not None:
            if not math.isfinite(result.max_step_mm) or result.max_step_mm <= 0:
                raise ValueError("max_step_mm must be finite and > 0")
        if result.subdivisions is not None:
            if not isinstance(result.subdivisions, int) or result.subdivisions < 1:
                raise ValueError("subdivisions must be a positive integer")
        for name in ("compute_bounds", "safety_screen"):
            if not isinstance(getattr(result, name), bool):
                raise ValueError(f"{name} must be a boolean")
        return result


@dataclass(frozen=True)
class VolumeBounds:
    """Certified continuous-removed-volume bracket plus sampled-set upper bound."""

    lower_mm3: float
    station_upper_mm3: float
    upper_mm3: float

    @property
    def spatial_uncertainty_mm3(self) -> float:
        return max(0.0, self.station_upper_mm3 - self.lower_mm3)

    @property
    def temporal_uncertainty_mm3(self) -> float:
        return max(0.0, self.upper_mm3 - self.station_upper_mm3)

    def to_dict(self) -> Dict[str, float]:
        return {
            "lower_mm3": self.lower_mm3,
            "station_upper_mm3": self.station_upper_mm3,
            "upper_mm3": self.upper_mm3,
            "width_mm3": self.upper_mm3 - self.lower_mm3,
            "spatial_uncertainty_mm3": self.spatial_uncertainty_mm3,
            "temporal_uncertainty_mm3": self.temporal_uncertainty_mm3,
        }


@dataclass
class ReferenceTrace:
    """EvaluationResult plus inspectable arrays for independent geometry tests."""

    result: C.EvaluationResult
    remaining: Optional[np.ndarray] = None
    removed: Optional[np.ndarray] = None
    bounds: Optional[VolumeBounds] = None


def _next_power_of_two(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


def _work_point(task: C.TaskSpec, point: Sequence[float], frame: C.Frame) -> Tuple[float, float, float]:
    p = tuple(float(x) for x in point)
    if frame == C.Frame.WORK:
        return p
    return task.work_to_machine.inverse().apply(p)


def _unsupported(task: C.TaskSpec, motion: C.MotionPlan,
                 settings: Mapping[str, Any], reasons: Sequence[str]) -> ReferenceTrace:
    reason = "; ".join(reasons)

    def unknown(unit: str) -> C.Metric:
        return C.Metric(None, unit, null_reason=reason)

    metrics = {
        "removed_volume_mm3": unknown("mm^3"),
        "residual_volume_mm3": unknown("mm^3"),
        "gouge_volume_mm3": unknown("mm^3"),
        "max_surface_error_mm": unknown("mm"),
        "min_clearance_mm": unknown("mm"),
        "min_rapid_cutter_clearance_mm": unknown("mm"),
        "rapid_cutter_penetration_upper_bound_mm": unknown("mm"),
        "unique_contacted_volume_mm3": unknown("mm^3"),
        "collision_sample_count": unknown("count"),
        "time_integrated_overlap_mm3_s": unknown("mm^3*s"),
        "spatial_uncertainty_mm3": unknown("mm^3"),
        "temporal_uncertainty_mm3": unknown("mm^3"),
        "cycle_time_s": C.Metric(sum(s.duration_s for s in motion.segments), "s"),
    }
    result = C.EvaluationResult(
        evaluator_id=EVALUATOR_ID, evaluator_revision=EVALUATOR_REVISION,
        task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
        discretization={"settings": dict(settings), "reference_revision": EVALUATOR_REVISION},
        coverage=C.Coverage.UNSUPPORTED, coverage_reasons=tuple(reasons),
        metrics=metrics, safety_pass=None, geometry_pass=None, motion_hash=motion.motion_hash,
    )
    return ReferenceTrace(result=result)


def _precheck(task: C.TaskSpec, motion: C.MotionPlan) -> Tuple[str, ...]:
    reasons = []
    if not task.initial_state.known or not motion.initial_state.known:
        reasons.append("unknown initial tool state")
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        reasons.append("ball/non-flat cutters are outside this reference's supported geometry")
    if any(s.kind == C.SegmentKind.ARC for s in motion.segments):
        reasons.append("arc sampling is not implemented; arcs are never replaced by chords")
    if task.fixtures:
        reasons.append("separate workholding solids are not implemented by this reference")
    if task.stock.kind != "analytic" or task.stock.params.get("primitive") != "box":
        reasons.append("only analytic box stock is supported")
    if task.stock.frame != C.Frame.WORK:
        reasons.append("stock must be declared in the work frame")
    if motion.timing_rule != task.motion_semantics.timing_rule:
        reasons.append("motion and task timing rules differ")
    m = task.work_to_machine.matrix
    if abs(m[2][0]) > 1e-9 or abs(m[2][1]) > 1e-9 or abs(m[2][2] - 1.0) > 1e-9:
        reasons.append("tilted work transforms are unsupported (machine/work +Z must coincide)")
    return tuple(reasons)


def _centres(grid: C.GridSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [grid.origin_mm[a] + (np.arange(grid.shape[a], dtype=np.float64) + 0.5) * grid.voxel_mm
            for a in range(3)]
    return tuple(np.meshgrid(*axes, indexing="ij", sparse=False))  # type: ignore[return-value]


def _box_distance(points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                  lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    q = [np.maximum(lo[a] - points[a], points[a] - hi[a]) for a in range(3)]
    positive = [np.maximum(v, 0.0) for v in q]
    outside = np.sqrt(positive[0] ** 2 + positive[1] ** 2 + positive[2] ** 2)
    inside = np.minimum(np.maximum(np.maximum(q[0], q[1]), q[2]), 0.0)
    return outside + inside


def _cylinder_distance(points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       base: Sequence[float], radius: float, height: float) -> np.ndarray:
    """Exact signed distance to one static capped vertical cylinder."""
    radial = np.hypot(points[0] - base[0], points[1] - base[1]) - radius
    vertical = np.maximum(base[2] - points[2], points[2] - (base[2] + height))
    outside = np.hypot(np.maximum(radial, 0.0), np.maximum(vertical, 0.0))
    inside = np.minimum(np.maximum(radial, vertical), 0.0)
    return outside + inside


def _body_distances(points: Tuple[np.ndarray, np.ndarray, np.ndarray], base: Sequence[float],
                    tool: C.ToolAssembly) -> Dict[str, Optional[np.ndarray]]:
    cutter = _cylinder_distance(points, base, tool.cutter_radius_mm, tool.flute_length_mm)
    shank = None
    shank_height = tool.holder_bottom_mm - tool.flute_length_mm
    if shank_height > 0:
        shank_base = (base[0], base[1], base[2] + tool.flute_length_mm)
        shank = _cylinder_distance(points, shank_base, tool.shank_radius_mm, shank_height)
    holder_base = (base[0], base[1], base[2] + tool.holder_bottom_mm)
    holder = _cylinder_distance(points, holder_base, tool.holder_radius_mm, tool.holder_height_mm)
    return {"cutter": cutter, "shank": shank, "holder": holder}


def _interpolate(segment: C.Segment, s: float) -> Tuple[float, float, float]:
    return tuple(segment.start_mm[a] + s * (segment.end_mm[a] - segment.start_mm[a]) for a in range(3))


def _null_target_metrics() -> Dict[str, C.Metric]:
    reason = "T1.0 executes stock/motion; operation-target scoring belongs to the downstream evaluator"
    return {
        "residual_volume_mm3": C.Metric(None, "mm^3", null_reason=reason),
        "gouge_volume_mm3": C.Metric(None, "mm^3", null_reason=reason),
        "max_surface_error_mm": C.Metric(None, "mm", null_reason=reason),
    }


def run_reference(task: C.TaskSpec, motion: C.MotionPlan,
                  settings: Optional[Mapping[str, Any]] = None) -> ReferenceTrace:
    """Execute supported typed motion and return result, occupancy, and volume bounds.

    ``subdivisions`` is an explicit testing/convergence override. Normally the number
    of stations is derived from three-dimensional tool displacement, voxel size,
    cutter radius, and flute length. Unsupported inputs return null metrics rather
    than partial geometry presented as complete.
    """
    settings_dict = {} if settings is None else dict(settings)
    config = ReferenceSettings.from_mapping(settings)
    motion.validate_for_task(task)
    unsupported = _precheck(task, motion)
    if unsupported:
        return _unsupported(task, motion, settings_dict, unsupported)

    lo = tuple(float(v) for v in task.stock.params["min_mm"])
    hi = tuple(float(v) for v in task.stock.params["max_mm"])
    if len(lo) != 3 or len(hi) != 3 or any(not math.isfinite(v) for v in lo + hi) or any(a >= b for a, b in zip(lo, hi)):
        raise ValueError("analytic box stock requires finite min_mm < max_mm vectors")

    points = _centres(task.grid)
    stock_d = _box_distance(points, lo, hi)
    initial = stock_d <= 0.0                    # closed removal stock convention
    stock_interior = stock_d < 0.0              # strict collision convention
    remaining = initial.copy()
    shape = remaining.shape
    voxel_volume = float(task.grid.voxel_mm ** 3)
    rho = math.sqrt(3.0) * task.grid.voxel_mm / 2.0

    definite = np.zeros(shape, dtype=bool)
    station_upper = np.zeros(shape, dtype=bool)
    continuous_upper = np.zeros(shape, dtype=bool)
    unique_by_component = {name: np.zeros(shape, dtype=bool)
                           for name in ("rapid_cutter", "cutter", "shank", "holder")}

    events = []
    first_event_kinds = set()
    collision_sample_count = 0
    integrated_overlap = 0.0
    min_clearance = math.inf
    min_rapid_cutter_clearance = math.inf
    rapid_penetration_upper = 0.0
    rapid_seen = False
    rapid_tolerance = task.tolerances.rapid_penetration_mm
    safety_uncertain = False
    uncertainty_segments = []
    segment_records = []
    base_step = config.max_step_mm
    if base_step is None:
        base_step = min(task.grid.voxel_mm / 2.0,
                        task.tool.cutter_radius_mm / 4.0,
                        task.tool.flute_length_mm / 4.0)

    for segment in motion.segments:
        length = segment.chord_length_mm
        if config.subdivisions is not None:
            m = config.subdivisions
        else:
            m = _next_power_of_two(max(1, math.ceil(length / base_step)))
        zero_length = length == 0.0
        delta_t = 0.0 if zero_length else length / (2.0 * m)
        segment_records.append({
            "id": segment.id, "kind": segment.kind.value, "length_mm": length,
            "subdivisions": m, "temporal_hausdorff_mm": delta_t,
            "duration_s": segment.duration_s,
            "endpoints_included": True,
        })
        current_fields = None
        station_count = 1 if zero_length else m + 1
        for j in range(station_count):
            s = 0.0 if zero_length else j / m
            base = _work_point(task, _interpolate(segment, s), motion.frame)
            if current_fields is None:
                current_fields = _body_distances(points, base, task.tool)

            if segment.kind == C.SegmentKind.FEED:
                cutter_d = current_fields["cutter"]
                assert cutter_d is not None
                if config.compute_bounds:
                    definite |= (stock_d < -rho) & (cutter_d < -rho)
                    station_upper |= (stock_d <= rho) & (cutter_d <= rho)
                    continuous_upper |= (stock_d <= rho) & (cutter_d <= rho + delta_t)
                remaining[cutter_d <= 0.0] = False
                checked = ("shank", "holder")
            else:
                checked = ("cutter", "shank", "holder")

            material = remaining & stock_interior
            any_contact = np.zeros(shape, dtype=bool)
            noncutting_min = None
            for name in checked:
                distance = current_fields[name]
                if distance is None:
                    continue
                rapid_cutter = segment.kind == C.SegmentKind.RAPID and name == "cutter"
                if rapid_cutter:
                    rapid_seen = True
                    if material.any():
                        min_rapid_cutter_clearance = min(
                            min_rapid_cutter_clearance, float(distance[material].min()))
                else:
                    noncutting_min = (distance.copy() if noncutting_min is None
                                      else np.minimum(noncutting_min, distance))
                # Contracts 1.1: tolerated rapid-cutter grazing is not a collision.
                threshold = -rapid_tolerance if rapid_cutter else 0.0
                contact = material & (distance < threshold)
                if rapid_cutter and config.safety_screen and zero_length and material.any():
                    upper = max(0.0, float(rho - distance[material].min()))
                    rapid_penetration_upper = max(rapid_penetration_upper, upper)
                    if upper > rapid_tolerance:
                        safety_uncertain = True
                        if segment.id not in uncertainty_segments:
                            uncertainty_segments.append(segment.id)
                if not contact.any():
                    continue
                any_contact |= contact
                if rapid_cutter:
                    component = "rapid_cutter"
                    event_kind = "rapid_collision"
                else:
                    component = name
                    event_kind = f"{name}_collision"
                unique_by_component[component] |= contact
                if event_kind not in first_event_kinds:
                    first_event_kinds.add(event_kind)
                    events.append(C.SegmentEvent(
                        segment_id=segment.id, kind=event_kind, s=s,
                        detail="first sampled penetration beyond the applicable tolerance at voxel centres; see temporal uncertainty",
                    ))

            if any_contact.any():
                collision_sample_count += 1
                weight = 0.0 if zero_length else segment.duration_s * (0.5 if j in (0, m) else 1.0) / m
                integrated_overlap += weight * float(any_contact.sum()) * voxel_volume

            if noncutting_min is not None and material.any():
                min_clearance = min(min_clearance, float(noncutting_min[material].min()))

            if config.safety_screen and not zero_length and j < m:
                next_s = (j + 1) / m
                next_base = _work_point(task, _interpolate(segment, next_s), motion.frame)
                next_fields = _body_distances(points, next_base, task.tool)
                possible_here = False
                for name in checked:
                    now_d, next_d = current_fields[name], next_fields[name]
                    if now_d is None:
                        continue
                    assert next_d is not None
                    # Every pose in [s_j,s_{j+1}] lies within delta_t of one endpoint.
                    near = np.minimum(now_d, next_d)
                    if segment.kind == C.SegmentKind.RAPID and name == "cutter":
                        upper = 0.0
                        if material.any():
                            upper = max(0.0, float((rho + delta_t - near[material]).max()))
                        rapid_penetration_upper = max(rapid_penetration_upper, upper)
                        if upper > rapid_tolerance:
                            possible_here = True
                    elif (material & (near <= rho + delta_t)).any():
                        # Non-cutting assembly still uses strict-interior safety.
                        possible_here = True
                if possible_here:
                    safety_uncertain = True
                    if segment.id not in uncertainty_segments:
                        uncertainty_segments.append(segment.id)
            else:
                next_fields = None
            current_fields = next_fields

    removed = initial & ~remaining
    if config.compute_bounds:
        bounds = VolumeBounds(
            lower_mm3=float(definite.sum()) * voxel_volume,
            station_upper_mm3=float(station_upper.sum()) * voxel_volume,
            upper_mm3=float(continuous_upper.sum()) * voxel_volume,
        )
    else:
        bounds = None

    unique = np.zeros(shape, dtype=bool)
    for contact in unique_by_component.values():
        unique |= contact
    by_component = {name: float(mask.sum()) * voxel_volume for name, mask in unique_by_component.items()}
    safety_pass = False if collision_sample_count else (None if safety_uncertain else True)
    coverage_reasons = ()
    if safety_uncertain and not collision_sample_count:
        coverage_reasons = (
            "assembly clearance remains uncertain under the pre-interval spatial/temporal bound on segments "
            + ",".join(uncertainty_segments),
        )

    metrics = _null_target_metrics()
    metrics.update({
        "removed_volume_mm3": C.Metric(float(removed.sum()) * voxel_volume, "mm^3"),
        "min_clearance_mm": (C.Metric(min_clearance, "mm", denominator="sampled non-cutting assembly vs remaining-stock voxel centres")
                              if math.isfinite(min_clearance) else
                              C.Metric(None, "mm", null_reason="no non-cutting assembly station with remaining stock")),
        "min_rapid_cutter_clearance_mm": (C.Metric(
            min_rapid_cutter_clearance, "mm", denominator="sampled rapid cutter vs remaining-stock voxel centres")
            if math.isfinite(min_rapid_cutter_clearance) else
            C.Metric(None, "mm", null_reason="no rapid station with remaining stock")),
        "rapid_cutter_penetration_upper_bound_mm": (C.Metric(
            rapid_penetration_upper, "mm", denominator="maximum over remaining strict-stock centres and rapid intervals")
            if rapid_seen and config.safety_screen else
            C.Metric(None, "mm", null_reason="no rapid segments" if not rapid_seen else "safety_screen=false")),
        "unique_contacted_volume_mm3": C.Metric(float(unique.sum()) * voxel_volume, "mm^3"),
        "collision_sample_count": C.Metric(float(collision_sample_count), "count"),
        "time_integrated_overlap_mm3_s": C.Metric(integrated_overlap, "mm^3*s"),
        "spatial_uncertainty_mm3": (C.Metric(bounds.spatial_uncertainty_mm3, "mm^3") if bounds else
                                     C.Metric(None, "mm^3", null_reason="compute_bounds=false")),
        "temporal_uncertainty_mm3": (C.Metric(bounds.temporal_uncertainty_mm3, "mm^3") if bounds else
                                      C.Metric(None, "mm^3", null_reason="compute_bounds=false")),
        "cycle_time_s": C.Metric(sum(s.duration_s for s in motion.segments), "s"),
    })
    discretization = {
        "reference_revision": EVALUATOR_REVISION,
        "settings": settings_dict,
        "voxel_mm": task.grid.voxel_mm,
        "grid_shape": list(task.grid.shape),
        "grid_origin_mm": list(task.grid.origin_mm),
        "cell_half_diagonal_mm": rho,
        "membership": {"removal": "closed", "noncutting_collision": "strict_interior",
                       "rapid_cutter_collision": "signed_distance < -rapid_penetration_mm"},
        "tolerances": {
            "rapid_penetration_mm": rapid_tolerance,
            "noncutting_clearance_qualification_mm": task.tolerances.clearance_mm,
            "clearance_mm_role": "reported qualification threshold only; safety_pass uses strict noncutting intersections",
        },
        "segments": segment_records,
        "unique_by_component_mm3": by_component,
        "collision_samples_are_per_station": True,
        "collision_continuation": "complete_motion_diagnostics",
        "safety_bound": "post-cut pre-interval stock; endpoint assembly union inflated by rho+ell/(2m)",
        "volume_bound": bounds.to_dict() if bounds else None,
    }
    result = C.EvaluationResult(
        evaluator_id=EVALUATOR_ID, evaluator_revision=EVALUATOR_REVISION,
        task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
        discretization=discretization, coverage=C.Coverage.COMPLETE,
        coverage_reasons=coverage_reasons, metrics=metrics, safety_pass=safety_pass,
        geometry_pass=None, motion_hash=motion.motion_hash, events=tuple(events),
    )
    return ReferenceTrace(result=result, remaining=remaining, removed=removed, bounds=bounds)


def evaluate_reference(task: C.TaskSpec, motion: C.MotionPlan,
                       settings: Optional[Mapping[str, Any]] = None) -> C.EvaluationResult:
    return run_reference(task, motion, settings).result


class ReferenceCarve:
    """T0.0 candidate-evaluator interface (posted NC is deliberately not parsed here)."""

    def evaluate_candidate(self, task: C.TaskSpec, motion: C.MotionPlan,
                           settings: Mapping[str, Any]) -> C.EvaluationResult:
        return evaluate_reference(task, motion, settings)
