"""Independent T1.0 fixture inputs and hard-coded analytic expectations.

The factories below do not call the reference carve to obtain expected values.
Run ``python -m verification.fixtures --output ...`` for the separately executed
refinement report; ordinary imports only construct inexpensive specifications.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from benchmarks import contracts as C

B_MIN = (0.0, 0.0, 0.0)
B_MAX = (60.0, 40.0, 12.0)
FEED_MM_S = 10.0
RAPID_MM_S = 10.0

Point = Tuple[float, float, float]


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    task: C.TaskSpec
    motion: C.MotionPlan
    removed_volume_mm3: float
    derivation: str
    safety_pass: Optional[bool] = None
    unique_contacted_volume_mm3: Optional[float] = None
    time_integrated_overlap_mm3_s: Optional[float] = None
    component_unique_mm3: Mapping[str, float] = field(default_factory=dict)
    first_events: Mapping[str, Tuple[str, float]] = field(default_factory=dict)
    collision_count_by_subdivisions: Mapping[int, int] = field(default_factory=dict)


def feed(segment_id: str, start: Point, end: Point, speed: float = FEED_MM_S) -> C.Segment:
    return C.Segment(id=segment_id, kind=C.SegmentKind.FEED, start_mm=start, end_mm=end,
                     duration_s=math.dist(start, end) / speed, feed_mm_s=speed)


def rapid(segment_id: str, start: Point, end: Point) -> C.Segment:
    return C.Segment(id=segment_id, kind=C.SegmentKind.RAPID, start_mm=start, end_mm=end,
                     duration_s=math.dist(start, end) / RAPID_MM_S)


def flat_tool(radius: float, flute: float, holder_radius: Optional[float] = None) -> C.ToolAssembly:
    """A flush non-cutting extension for ordinary cases, or a wider holder when requested."""
    return C.ToolAssembly(
        cutter_kind=C.CutterKind.FLAT, cutter_radius_mm=radius, flute_length_mm=flute,
        shank_radius_mm=radius, holder_bottom_mm=flute,
        holder_radius_mm=radius if holder_radius is None else holder_radius,
        holder_height_mm=20.0,
    )


def _geometry(lo: Point, hi: Point) -> C.GeometrySource:
    return C.GeometrySource(kind="analytic", params={"primitive": "box", "min_mm": list(lo), "max_mm": list(hi)},
                            frame=C.Frame.WORK)


def _grid(lo: Point, hi: Point, voxel: float) -> C.GridSpec:
    shape = tuple(int(round((hi[a] - lo[a]) / voxel)) for a in range(3))
    if any(not math.isclose(shape[a] * voxel, hi[a] - lo[a], abs_tol=1e-9) for a in range(3)):
        raise ValueError(f"voxel {voxel} does not evenly span stock {lo}..{hi}")
    return C.GridSpec(shape=shape, origin_mm=lo, voxel_mm=voxel, frame=C.Frame.WORK)


def make_fixture(fixture_id: str, segments: Sequence[C.Segment], tool: C.ToolAssembly,
                 removed: float, derivation: str, *, stock_lo: Point = B_MIN,
                 stock_hi: Point = B_MAX, voxel: float = 2.0,
                 initial: Optional[Point] = None, **expected) -> Fixture:
    geometry = _geometry(stock_lo, stock_hi)
    if initial is None:
        if not segments:
            raise ValueError("empty fixture requires an initial position")
        initial = segments[0].start_mm
    state = C.MachineState(position_mm=initial, frame=C.Frame.WORK)
    task = C.TaskSpec(
        task_id=f"reference-{fixture_id}", family="reference_geometry", split=C.Split.DEV,
        source=C.SourceInfo(origin="verification:T1.0-fixtures@1", license="MIT"),
        length_unit="mm", stock=geometry, original_part=geometry,
        operation_target=C.OperationTarget(geometry=geometry),
        grid=_grid(stock_lo, stock_hi, voxel), work_to_machine=C.Transform.translation((0, 0, 0)),
        initial_state=state, tool=tool,
        limits=C.MachineLimits(max_feed_mm_s=100.0, rapid_mm_s=RAPID_MM_S, max_accel_mm_s2=1000.0,
                               envelope_min_mm=(-100, -100, -100), envelope_max_mm=(200, 200, 200)),
        tolerances=C.Tolerances(completion_residual_mm3=0.0, gouge_volume_mm3=0.0,
                                local_surface_error_mm=0.0, clearance_mm=0.0),
    )
    motion = C.MotionPlan(frame=C.Frame.WORK, initial_state=state, segments=tuple(segments),
                          timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
                          provenance="verification:independent_fixture")
    motion.validate_for_task(task)
    return Fixture(fixture_id=fixture_id, task=task, motion=motion,
                   removed_volume_mm3=removed, derivation=derivation, **expected)


def empty_preengaged() -> Fixture:
    return make_fixture("empty_preengaged", (), flat_tool(3, 6), 0.0,
                        "initial pose is state, not a cut", initial=(20, 20, 4), safety_pass=True)


def plunge_above(radius: float = 3.0, phase_xy: Tuple[float, float] = (0.0, 0.0)) -> Fixture:
    dx, dy = phase_xy
    p0, p1 = (20 + dx, 20 + dy, 18), (20 + dx, 20 + dy, 8)
    v = math.pi * radius ** 2 * 4
    return make_fixture("plunge_above", [feed("plunge", p0, p1)], flat_tool(radius, 20), v,
                        "pi*r^2*4: start above stock and include stock intersection only", safety_pass=True)


def plunge_preengaged() -> Fixture:
    return make_fixture("plunge_preengaged", [feed("plunge", (20, 20, 10), (20, 20, 6))],
                        flat_tool(3, 20), 54 * math.pi,
                        "pi*3^2*(2 initial overlap + 4 descent)", safety_pass=True)


def horizontal_slot(radius: float = 3.0, phase_xy: Tuple[float, float] = (0.0, 0.0)) -> Fixture:
    dx, dy = phase_xy
    p0, p1 = (14 + dx, 20 + dy, 0), (38 + dx, 20 + dy, 0)
    v = 12 * (2 * radius * 24 + math.pi * radius ** 2)
    return make_fixture("horizontal_slot", [feed("slot", p0, p1)], flat_tool(radius, 20), v,
                        "12*(2*r*24 + pi*r^2): full-depth swept capsule incl. both end disks", safety_pass=True)


def sloped_ramp(radius: float = 2.0, phase_xy: Tuple[float, float] = (0.0, 0.0)) -> Fixture:
    dx, dy = phase_xy
    p0, p1 = (8 + dx, 20 + dy, 8), (32 + dx, 20 + dy, -4)
    # q=(z+4)/12 on [0,8], (20-z)/12 on [8,12], hence Q=26/3.
    v = 12 * math.pi * radius ** 2 + 2 * radius * 24 * (26 / 3)
    return make_fixture("sloped_ramp", [feed("ramp", p0, p1)], flat_tool(radius, 12), v,
                        "pi*r^2*12 + 2*r*24*(26/3): integrate the temporal cross-section interval", safety_pass=True)


def near_vertical(radius: float = 3.0, phase_xy: Tuple[float, float] = (0.0, 0.0)) -> Fixture:
    dx, dy = phase_xy
    p0, p1 = (20 + dx, 20 + dy, 30), (20.8 + dx, 20.3 + dy, -10)
    ell = math.sqrt(0.73)
    v = 12 * math.pi * radius ** 2 + 2 * radius * ell * 3
    return make_fixture("near_vertical", [feed("near_vertical", p0, p1)], flat_tool(radius, 10), v,
                        "12*pi*r^2 + 2*r*sqrt(.73)*3: q=1/4 throughout stock Z", safety_pass=True)


def reversal() -> Fixture:
    # Half-length return makes its dyadic stations a subset of the outward sweep.
    points = [(12, 20, 0), (40, 20, 0), (26, 20, 0)]
    segments = [feed("outward", points[0], points[1]), feed("reverse", points[1], points[2])]
    return make_fixture("reversal", segments, flat_tool(3, 20), 12 * (168 + 9 * math.pi),
                        "12*(2*3*28 + pi*3^2): reverse lies in first capsule", safety_pass=True)


def repeated_plunge() -> Fixture:
    p0, p1 = (20, 20, 18), (20, 20, 8)
    segments = [feed("down_1", p0, p1), feed("up", p1, p0), feed("down_2", p0, p1)]
    return make_fixture("repeated_plunge", segments, flat_tool(3, 20), 36 * math.pi,
                        "union of three identical plunge intersections", safety_pass=True)


def through_hole(radius: float = 4.0, phase_xy: Tuple[float, float] = (0.0, 0.0)) -> Fixture:
    dx, dy = phase_xy
    p0, p1 = (20 + dx, 20 + dy, 16), (20 + dx, 20 + dy, -2)
    return make_fixture("through_hole", [feed("through", p0, p1)], flat_tool(radius, 6),
                        12 * math.pi * radius ** 2,
                        "pi*r^2*12: through motion clips to stock, leaving no geometric floor", safety_pass=True)


def zero_length_feed(radius: float = 3.0, phase_xy: Tuple[float, float] = (0.0, 0.0)) -> Fixture:
    dx, dy = phase_xy
    p = (20 + dx, 20 + dy, 4)
    return make_fixture("zero_length_feed", [feed("zero", p, p)], flat_tool(radius, 6),
                        6 * math.pi * radius ** 2,
                        "one stationary cylinder: pi*r^2*6, duration zero", safety_pass=True)


def cutter_leading_holder_safe() -> Fixture:
    return make_fixture(
        "cutter_leading_holder_safe", [feed("column_plunge", (20, 20, 14), (20, 20, 5))],
        flat_tool(6, 4, holder_radius=10), 175.0,
        "25 mm^2 stock section * 7 mm depth; cutter clears the column 4 mm before holder arrives",
        stock_lo=(17.5, 17.5, 0), stock_hi=(22.5, 22.5, 12), voxel=1.0,
        safety_pass=True, unique_contacted_volume_mm3=0.0, time_integrated_overlap_mm3_s=0.0,
    )


def genuine_holder_collision() -> Fixture:
    return make_fixture(
        "genuine_holder_collision", [feed("wide_plunge", (20, 20, 14), (20, 20, 5), speed=9.0)],
        flat_tool(6, 4, holder_radius=10), 252 * math.pi,
        "36*pi*7 removed; annulus pi*(10^2-6^2)*3 remains for chronological holder contact",
        safety_pass=False, unique_contacted_volume_mm3=192 * math.pi,
        time_integrated_overlap_mm3_s=32 * math.pi,
        first_events={"holder_collision": ("wide_plunge", 2 / 3)},
        collision_count_by_subdivisions={3: 1, 6: 2},
    )


def rapid_through_solid() -> Fixture:
    # The contract has a mandatory assembly. Flush r=3 holder extends the contacted
    # vertical interval from cutter 6..10 to assembly 6..12, giving six mm total height.
    return make_fixture(
        "rapid_through_solid", [rapid("rapid", (-10, 20, 6), (70, 20, 6))], flat_tool(3, 4), 0.0,
        "rapid removes zero; unique assembly contact = 60*6*6, cutter-only contact = 60*6*4",
        safety_pass=False, unique_contacted_volume_mm3=2160.0,
        time_integrated_overlap_mm3_s=324 * math.pi,
        component_unique_mm3={"rapid_cutter": 1440.0},
        first_events={"rapid_collision": ("rapid", 7 / 80)},
    )


def zero_length_rapid() -> Fixture:
    return make_fixture("zero_length_rapid", [rapid("zero_rapid", (20, 20, 4), (20, 20, 4))],
                        flat_tool(3, 6), 0.0, "stationary rapid inside stock collides but never carves",
                        safety_pass=False, unique_contacted_volume_mm3=None,
                        first_events={"rapid_collision": ("zero_rapid", 0.0)})


_FACTORIES = (
    empty_preengaged, plunge_above, plunge_preengaged, horizontal_slot, sloped_ramp,
    near_vertical, reversal, repeated_plunge, through_hole, zero_length_feed,
    cutter_leading_holder_safe, genuine_holder_collision, rapid_through_solid, zero_length_rapid,
)


def all_fixtures() -> Tuple[Fixture, ...]:
    return tuple(factory() for factory in _FACTORIES)


def fixture_by_id(fixture_id: str) -> Fixture:
    for fixture in all_fixtures():
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(fixture_id)


def with_voxel(fixture: Fixture, voxel: float) -> Fixture:
    grid = _grid(fixture.task.stock.params["min_mm"], fixture.task.stock.params["max_mm"], voxel)
    return dataclasses.replace(fixture, task=dataclasses.replace(fixture.task, grid=grid))


def with_joint_phase(fixture: Fixture, phase: Point) -> Fixture:
    """Translate stock and complete motion relative to a padded fixed sampling lattice."""
    if any(abs(x) >= fixture.task.grid.voxel_mm for x in phase):
        raise ValueError("phase offsets should be fractions of one voxel")
    v = fixture.task.grid.voxel_mm
    old_grid = fixture.task.grid
    grid = C.GridSpec(shape=tuple(n + 4 for n in old_grid.shape),
                      origin_mm=tuple(old_grid.origin_mm[a] - 2 * v for a in range(3)),
                      voxel_mm=v, frame=C.Frame.WORK)

    def shift(p: Sequence[float]) -> Point:
        return tuple(p[a] + phase[a] for a in range(3))

    lo = shift(fixture.task.stock.params["min_mm"])
    hi = shift(fixture.task.stock.params["max_mm"])
    geometry = _geometry(lo, hi)
    state = dataclasses.replace(fixture.task.initial_state, position_mm=shift(fixture.task.initial_state.position_mm))
    segments = tuple(dataclasses.replace(s, start_mm=shift(s.start_mm), end_mm=shift(s.end_mm))
                     for s in fixture.motion.segments)
    motion = dataclasses.replace(fixture.motion, initial_state=state, segments=segments)
    task = dataclasses.replace(fixture.task, stock=geometry, original_part=geometry,
                               operation_target=C.OperationTarget(geometry=geometry),
                               grid=grid, initial_state=state)
    motion.validate_for_task(task)
    return dataclasses.replace(fixture, task=task, motion=motion)


def generate_convergence_report() -> Dict[str, object]:
    """Run explicit, non-import-time refinement cases and return JSON-ready evidence."""
    from verification.reference_carve import EVALUATOR_REVISION, run_reference

    records = []
    for fixture_id in ("plunge_above", "horizontal_slot", "sloped_ramp", "near_vertical", "rapid_through_solid"):
        fixture = fixture_by_id(fixture_id)
        rows = []
        for voxel in (4.0, 2.0, 1.0):
            refined = with_voxel(fixture, voxel)
            trace = run_reference(refined.task, refined.motion, {"max_step_mm": voxel / 2.0})
            assert trace.bounds is not None
            rows.append({
                "voxel_mm": voxel,
                "task_hash": refined.task.task_hash, "motion_hash": refined.motion.motion_hash,
                "settings": {"max_step_mm": voxel / 2.0},
                "sampled_removed_mm3": trace.result.metrics["removed_volume_mm3"].value,
                "bounds": trace.bounds.to_dict(),
                "analytic_removed_mm3": refined.removed_volume_mm3,
                "analytic_inside_bounds": trace.bounds.lower_mm3 - 1e-8 <= refined.removed_volume_mm3 <= trace.bounds.upper_mm3 + 1e-8,
                "safety_pass": trace.result.safety_pass,
                "unique_contacted_mm3": trace.result.metrics["unique_contacted_volume_mm3"].value,
                "time_integrated_overlap_mm3_s": trace.result.metrics["time_integrated_overlap_mm3_s"].value,
            })
        records.append({"fixture_id": fixture_id, "derivation": fixture.derivation,
                        "base_task_hash": fixture.task.task_hash, "base_motion_hash": fixture.motion.motion_hash,
                        "rows": rows})

    chronology = []
    for fixture_id in ("cutter_leading_holder_safe", "genuine_holder_collision"):
        fixture = fixture_by_id(fixture_id)
        rows = []
        for subdivisions in (1, 2, 3, 4, 6, 8, 16, 32):
            trace = run_reference(fixture.task, fixture.motion, {"subdivisions": subdivisions})
            rows.append({
                "voxel_mm": fixture.task.grid.voxel_mm, "subdivisions": subdivisions,
                "sampled_removed_mm3": trace.result.metrics["removed_volume_mm3"].value,
                "safety_pass": trace.result.safety_pass,
                "collision_sample_count": trace.result.metrics["collision_sample_count"].value,
                "unique_contacted_mm3": trace.result.metrics["unique_contacted_volume_mm3"].value,
                "time_integrated_overlap_mm3_s": trace.result.metrics["time_integrated_overlap_mm3_s"].value,
            })
        chronology.append({
            "fixture_id": fixture_id, "expected_safety": fixture.safety_pass,
            "task_hash": fixture.task.task_hash, "motion_hash": fixture.motion.motion_hash,
            "analytic_removed_mm3": fixture.removed_volume_mm3,
            "analytic_unique_contacted_mm3": fixture.unique_contacted_volume_mm3,
            "analytic_time_integrated_overlap_mm3_s": fixture.time_integrated_overlap_mm3_s,
            "first_events": dict(fixture.first_events),
            "rows": rows,
        })
    return {
        "schema": "diffcam-reference-convergence/1", "evaluator_revision": EVALUATOR_REVISION,
        "continuous_volume_cases": records, "chronology_cases": chronology,
        "notes": [
            "Analytic continuous volumes are independent of sampled voxel volumes.",
            "lower/station_upper/upper brackets concern removed stock, not collision contact volumes.",
            "collision_sample_count changes with stations; unique contact and time integral are distinct diagnostics.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="verification/reference_convergence.json")
    args = parser.parse_args()
    path = Path(args.output)
    path.write_text(json.dumps(generate_convergence_report(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
