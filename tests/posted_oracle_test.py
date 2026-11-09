"""Independent production posted-path evaluator checks.

T1.0 supplies the separate NumPy reference; tests compare its independently
stationed results to the production evaluator rather than calling reference
helpers to produce production geometry. Additional NC expectations are computed
from task coordinates, independent swept volumes, and hand-enumerated rings.
"""

from dataclasses import replace
import math

import numpy as np
import pytest

from benchmarks import contracts as C
from cam.gcode_parser import parse_motion_plan
from cam.posted_evaluator import PostedEvaluator, _analytic_mask, _centres
from cam.posts import post_motion_plan
from verification import fixtures as F
from verification.reference_carve import run_reference


def metric(result, name):
    value = result.metrics[name].value
    assert value is not None, (name, result.metrics[name])
    return value


def machine_plan(task, motion, transform):
    """Rigidly map a work fixture to machine NC coordinates for posting."""
    def move(value):
        return transform.apply(value)
    initial = replace(task.initial_state, frame=C.Frame.MACHINE, position_mm=move(task.initial_state.position_mm))
    mapped = []
    for segment in motion.segments:
        arc = None if segment.arc is None else replace(segment.arc, centre_mm=move(segment.arc.centre_mm))
        mapped.append(replace(segment, start_mm=move(segment.start_mm), end_mm=move(segment.end_mm), arc=arc))
    new_task = replace(task, work_to_machine=transform, initial_state=initial)
    new_motion = replace(motion, frame=C.Frame.MACHINE, initial_state=initial, segments=tuple(mapped))
    new_motion.validate_for_task(new_task)
    return new_task, new_motion


def machine_fixture(fixture, *, rapid_penetration_mm=None):
    initial = replace(fixture.task.initial_state, frame=C.Frame.MACHINE)
    task = replace(fixture.task, initial_state=initial)
    if rapid_penetration_mm is not None:
        task = replace(task, tolerances=replace(task.tolerances, rapid_penetration_mm=rapid_penetration_mm))
    motion = replace(fixture.motion, frame=C.Frame.MACHINE, initial_state=initial)
    motion.validate_for_task(task)
    return task, motion


def simple_removal_task(fixture=None):
    """A small remaining part, with all other stock requiring removal."""
    fixture = fixture or F.plunge_above()
    geometry = C.GeometrySource(kind="analytic", params={
        "primitive": "box", "min_mm": [26, 26, 0], "max_mm": [30, 30, 12]})
    return replace(fixture.task, original_part=geometry,
                   operation_target=C.OperationTarget(geometry=geometry))


@pytest.mark.parametrize("fixture", F.all_fixtures(), ids=lambda f: f.fixture_id)
def test_t10_fixtures_match_independent_reference_geometry_and_chronology(fixture):
    reference = run_reference(fixture.task, fixture.motion)
    produced = PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {})
    assert produced.coverage == C.Coverage.COMPLETE
    assert produced.safety_pass == reference.result.safety_pass
    assert produced.motion_hash == fixture.motion.motion_hash
    assert all(produced.discretization["volume_bound"][name] == pytest.approx(getattr(reference.bounds, name))
               for name in ("lower_mm3", "station_upper_mm3", "upper_mm3"))
    assert reference.bounds.lower_mm3 - 1e-8 <= fixture.removed_volume_mm3 <= reference.bounds.upper_mm3 + 1e-8
    for name in (
        "removed_volume_mm3", "unique_contacted_volume_mm3", "collision_sample_count",
        "time_integrated_overlap_mm3_s", "cycle_time_s", "min_clearance_mm",
        "min_rapid_cutter_clearance_mm", "min_clearance_lower_bound_mm",
        "min_rapid_cutter_clearance_lower_bound_mm", "rapid_cutter_penetration_upper_bound_mm",
        "spatial_uncertainty_mm3", "temporal_uncertainty_mm3",
    ):
        assert produced.metrics[name].unit == C.METRIC_UNITS.get(name, produced.metrics[name].unit)
        if reference.result.metrics[name].value is None:
            assert produced.metrics[name].value is None
            assert produced.metrics[name].null_reason
        else:
            assert metric(produced, name) == pytest.approx(reference.result.metrics[name].value, abs=1e-9), name
    assert [(e.segment_id, e.kind, e.s) for e in produced.events] == [
        (e.segment_id, e.kind, e.s) for e in reference.result.events]
    assert metrics_whole_cell_semantics(produced)


def metrics_whole_cell_semantics(result):
    return "whole-cell" in result.discretization["clearance_metrics"]


def test_safe_holder_is_carved_before_cutter_arrives():
    fixture = F.cutter_leading_holder_safe()
    reference = run_reference(fixture.task, fixture.motion)
    result = PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {})
    assert result.safety_pass is True
    assert reference.result.safety_pass is True
    assert metric(result, "unique_contacted_volume_mm3") == 0.0
    assert metric(result, "time_integrated_overlap_mm3_s") == 0.0
    assert result.discretization["unique_by_component_mm3"]["holder"] == 0.0
    assert not result.events


def test_genuine_holder_collision_is_causal_and_repeated_fixtures_agree():
    fixture = F.genuine_holder_collision()
    for subdivisions in (3, 6):
        expected = run_reference(fixture.task, fixture.motion, {"subdivisions": subdivisions})
        result = PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {"subdivisions": subdivisions})
        assert result.safety_pass is False
        # Hand-enumerated occupied-centre ring: 48 XY cells, each 8 mm^3.
        assert metric(result, "unique_contacted_volume_mm3") == 48 * 8
        assert metric(result, "unique_contacted_volume_mm3") == metric(expected.result, "unique_contacted_volume_mm3")
        assert metric(result, "collision_sample_count") == {3: 1, 6: 2}[subdivisions]
        assert metric(result, "time_integrated_overlap_mm3_s") == pytest.approx(48 * 8 / (6 if subdivisions == 3 else 4))
        assert next(e for e in result.events if e.kind == "holder_collision").segment_id == "wide_plunge"
        assert result.discretization["volume_bound"]["lower_mm3"] <= 252 * math.pi <= result.discretization["volume_bound"]["upper_mm3"]
    # A third call uses no cached mask, positions, events or candidate history.
    repeated = PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {"subdivisions": 3})
    assert repeated.to_dict() == PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {"subdivisions": 3}).to_dict()


def test_shank_is_distinct_from_cutter_and_holder():
    tool = C.ToolAssembly(cutter_kind=C.CutterKind.FLAT, cutter_radius_mm=2.0, flute_length_mm=4.0,
                          shank_radius_mm=4.0, holder_bottom_mm=8.0,
                          holder_radius_mm=4.0, holder_height_mm=20.0)
    fixture = F.make_fixture("shank", [F.feed("shank_plunge", (20, 20, 14), (20, 20, 6))],
                             tool, 24 * math.pi, "independent annulus below the holder")
    reference = run_reference(fixture.task, fixture.motion)
    result = PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {})
    assert result.safety_pass is False
    assert metric(result, "unique_contacted_volume_mm3") == metric(reference.result, "unique_contacted_volume_mm3")
    assert any(e.kind == "shank_collision" for e in result.events)
    assert not any(e.kind == "holder_collision" for e in result.events)
    assert result.discretization["unique_by_component_mm3"]["shank"] > 0
    assert result.discretization["volume_bound"]["lower_mm3"] <= 24 * math.pi <= result.discretization["volume_bound"]["upper_mm3"]


def test_operation_target_gouge_and_original_part_diagnostic_are_separate():
    fixture = F.zero_length_feed()
    original = C.GeometrySource(kind="analytic", params={
        "primitive": "box", "min_mm": [26, 26, 0], "max_mm": [30, 30, 12]})
    target = C.GeometrySource(kind="analytic", params={
        "primitive": "box", "min_mm": [16, 16, 0], "max_mm": [30, 30, 12]})
    # At this 2mm grid, T has 7*7*6 cells and P has 2*2*6; the hand-counted
    # excluded volume is (294-24)*8 mm^3.
    declared_excluded = (7 * 7 * 6 - 2 * 2 * 6) * 8.0
    task = replace(fixture.task, original_part=original,
                   operation_target=C.OperationTarget(geometry=target,
                                                      excluded_volume_mm3=declared_excluded,
                                                      exclusion_reason="fixture T includes P plus explicit excluded stock"))
    result = PostedEvaluator().evaluate_candidate(task, fixture.motion, {})
    assert result.coverage == C.Coverage.COMPLETE
    assert metric(result, "gouge_volume_mm3") > 0
    assert metric(result, "original_part_gouge_volume_mm3") == 0.0
    assert metric(result, "excluded_volume_mm3") == declared_excluded
    assert "initial & operation_target" in result.metrics["gouge_fraction"].denominator
    assert "diagnostic only" in result.metrics["original_part_gouge_volume_mm3"].denominator
    mismatched = replace(task, operation_target=replace(
        task.operation_target, excluded_volume_mm3=declared_excluded + 8))
    rejected = PostedEvaluator().evaluate_candidate(mismatched, fixture.motion, {})
    assert rejected.coverage == C.Coverage.UNSUPPORTED
    assert rejected.safety_pass is None
    assert "excluded volume mismatch" in rejected.coverage_reasons[0]


def test_nonfinite_stock_box_fails_instead_of_becoming_empty_safe_stock():
    fixture = F.plunge_above()
    source = replace(fixture.task.stock, params=dict(fixture.task.stock.params, min_mm=[float("nan"), 0, 0]))
    task = replace(fixture.task, stock=source)
    # Contract identity cannot silently hash NaN. There is no valid
    # EvaluationResult record or unsafe "no material is present" conclusion.
    with pytest.raises(C.ContractError, match="non-finite"):
        PostedEvaluator().evaluate_candidate(task, fixture.motion, {})


@pytest.mark.parametrize("fixture", [F.rapid_through_solid(), F.zero_length_rapid()])
def test_rapids_disable_stock_subtraction_at_moving_and_stationary_poses(fixture):
    reference = run_reference(fixture.task, fixture.motion)
    result = PostedEvaluator().evaluate_candidate(simple_removal_task(fixture), fixture.motion, {})
    assert result.safety_pass is False
    assert metric(result, "removed_volume_mm3") == 0.0
    assert result.geometry_pass is False
    assert any(e.kind == "rapid_collision" for e in result.events)
    assert metric(result, "unique_contacted_volume_mm3") == metric(reference.result, "unique_contacted_volume_mm3")
    assert metric(result, "collision_sample_count") == metric(reference.result, "collision_sample_count")


def test_no_cut_rejection_and_explicit_unknown_surface_metrics():
    fixture = F.plunge_above()
    task = simple_removal_task(fixture)
    task = replace(task, tolerances=replace(
        task.tolerances, completion_residual_mm3=1_000_000, gouge_volume_mm3=1_000_000,
        local_surface_error_mm=1_000_000))
    machine_task, _ = machine_plan(task, fixture.motion, C.Transform.translation((100, 80, -150)))
    nc = b"G21 G90 G54\nM2\n"
    evaluated = PostedEvaluator().evaluate_posted_gcode(machine_task, nc, C.Dialect.RS274, {})
    assert evaluated.nc_hash == C.nc_hash(nc)
    assert evaluated.metrics["removed_volume_mm3"].value == 0.0
    assert metric(evaluated, "residual_volume_mm3") > 0
    assert evaluated.geometry_pass is False
    assert evaluated.metrics["force_proxy"].value is None
    assert evaluated.metrics["air_fraction"].value is None
    empty_geometry = C.GeometrySource(kind="analytic", params={
        "primitive": "box", "min_mm": [0, 0, 0], "max_mm": [0.5, 0.5, 0.5]})
    # Both masks have no occupied voxel centres in this 2mm grid; the original
    # target is not an out-of-stock geometry that the metric library must reject.
    empty_task = replace(task, original_part=empty_geometry,
                         operation_target=C.OperationTarget(geometry=empty_geometry))
    empty_motion = C.MotionPlan(C.Frame.WORK, empty_task.initial_state, (),
                                C.TimingRule.EXACT_STOP_CONSTANT_SPEED)
    empty = PostedEvaluator().evaluate_candidate(empty_task, empty_motion, {})
    for name in ("max_surface_error_mm", "p99_surface_error_mm", "asd_mm", "hd95_mm"):
        assert empty.metrics[name].value is None
        assert "surface" in empty.metrics[name].null_reason


def test_real_posted_nc_uses_offset_and_retract_without_unplanned_cut():
    # Remove tolerance ambiguity while executing both cut and retract portions.
    extended = F.retract_from_own_slot()
    original_task, original_motion = machine_fixture(extended, rapid_penetration_mm=10.0)
    task, motion = machine_plan(original_task, original_motion, C.Transform.translation((100, 80, -150)))
    expected = PostedEvaluator().evaluate_candidate(task, motion, {})
    for dialect in C.Dialect:
        raw = post_motion_plan(motion, task, dialect).encode()
        result = PostedEvaluator().evaluate_posted_gcode(task, raw, dialect, {})
        assert result.nc_hash == C.nc_hash(raw)
        assert result.safety_pass == expected.safety_pass
        assert metric(result, "removed_volume_mm3") == metric(expected, "removed_volume_mm3")
        assert metric(result, "cycle_time_s") == 1.6
        parsed = parse_motion_plan(raw.decode(), task, dialect)
        assert parsed.segments[0].start_mm == (120, 100, -134)
        assert parsed.segments[0].end_mm == (120, 100, -142)
        assert parsed.segments[1].kind == C.SegmentKind.RAPID
        assert result.discretization["parsed_motion_hash"] == parsed.motion_hash


def test_hand_parsed_g54_and_machine_home_semantics_have_physical_volume():
    task = C.example_pocket_task()
    nc = (b"G21 G90 G54 G61 G43 H1\n"
          b"G0 X-8 Y-8 Z10\nG1 Z-4 F240\nG1 X8 F600\nG0 Z10\nG53 G0 Z0\nM2\n")
    result = PostedEvaluator().evaluate_posted_gcode(task, nc, C.Dialect.RS274, {})
    plan = parse_motion_plan(nc.decode(), task)
    assert plan.segments[0].end_mm == (92, 72, -140)
    assert plan.segments[-1].end_mm[2] == 0.0  # no extra work-Z=-150 carve
    # 16mm straight horizontal traversal, radius 3.175, 4mm stock depth.
    analytic = 4 * (16 * 2 * task.tool.cutter_radius_mm + math.pi * task.tool.cutter_radius_mm ** 2)
    assert result.discretization["volume_bound"]["lower_mm3"] <= analytic <= result.discretization["volume_bound"]["upper_mm3"]
    assert metric(result, "removed_volume_mm3") <= result.discretization["volume_bound"]["upper_mm3"]
    assert metric(result, "cycle_time_s") == pytest.approx(sum(s.duration_s for s in plan.segments))


def test_hash_mismatch_is_an_error_and_hash_is_over_exact_bytes():
    task, _ = machine_fixture(F.plunge_above())
    raw = b"G21 G90 G54\nG0 X20 Y20 Z18\nM2\n"
    evaluator = PostedEvaluator()
    with pytest.raises(C.ContractError, match="hash"):
        evaluator.evaluate_posted_gcode(task, raw + b"\n", C.Dialect.RS274, {"expected_nc_hash": C.nc_hash(raw)})
    result = evaluator.evaluate_posted_gcode(task, raw, C.Dialect.RS274, {"expected_nc_hash": C.nc_hash(raw)})
    assert result.nc_hash == C.nc_hash(raw)


@pytest.mark.parametrize("nc,match", [
    (b"G21 G90 G54\nG0 X20 Y20 Z18\nM7\nM2\n", "unsupported M code"),
    (b"G21 G90 G54\nG0 X20 Y20 Z18\nQ12\nM2\n", "setup word Q12"),
    (b"G21 G90 G54\nG0 X20 Y20 Z18\nM8\nM2\n", None),
    (b"G21 G90 G54\nG123 X20\nM2\n", "unsupported G123"),
    (b"G21 G90 G54\nG1 X20 Y20 Z8 F600\nM2\n", None),
])
def test_strict_parser_coverage_preserves_supplied_program_identity(nc, match):
    task, _ = machine_fixture(F.plunge_above())
    result = PostedEvaluator().evaluate_posted_gcode(task, nc, C.Dialect.RS274, {})
    if match:
        assert result.coverage == C.Coverage.UNSUPPORTED
        assert result.safety_pass is None
        assert match in result.coverage_reasons[0]
        assert result.metrics["removed_volume_mm3"].value is None
        assert result.nc_hash == C.nc_hash(nc)
    else:
        assert result.coverage == C.Coverage.COMPLETE
        if b"M8" in nc:  # legal coolant control is not material removal
            assert result.metrics["removed_volume_mm3"].value == 0.0
        else:
            assert result.metrics["removed_volume_mm3"].value > 0


def test_unknown_state_and_ball_tools_cannot_claim_safe_coverage():
    fixture = F.plunge_above()
    unknown_motion = replace(fixture.motion, initial_state=replace(fixture.motion.initial_state, position_mm=None, known=False), segments=())
    unknown_task = replace(fixture.task, initial_state=unknown_motion.initial_state)
    result = PostedEvaluator().evaluate_candidate(unknown_task, unknown_motion, {})
    assert result.coverage == C.Coverage.UNSUPPORTED
    assert result.safety_pass is None
    assert "unknown initial tool state" in result.coverage_reasons[0]
    ball_task = replace(fixture.task, tool=replace(fixture.task.tool, cutter_kind=C.CutterKind.BALL))
    ball = PostedEvaluator().evaluate_candidate(ball_task, fixture.motion, {})
    assert ball.coverage == C.Coverage.UNSUPPORTED
    assert ball.safety_pass is None
    assert "ball" in ball.coverage_reasons[0]


def test_coarse_quarter_arc_bracket_includes_displaced_midpoint_removal():
    """Reviewer repro: arc-to-chord padding misses the half-arc cutter."""
    arc = C.ArcData((20, 20, 0), C.ArcPlane.XY, C.ArcDirection.CCW)
    segment = C.Segment("a", C.SegmentKind.ARC, (30, 20, 0), (20, 30, 0),
                        math.pi * 10 / 20, 10, arc)
    fixture = F.make_fixture("arc-gap", [segment], F.flat_tool(1, 20), 0.0, "reviewer midpoint probe",
                             stock_lo=(26, 26, 0), stock_hi=(30, 30, 12), voxel=2.0)
    part = C.GeometrySource(kind="analytic", params={
        "primitive": "box", "min_mm": [26.6, 26.6, 0.1], "max_mm": [27.4, 27.4, 11.9]})
    task = replace(fixture.task, original_part=part, operation_target=C.OperationTarget(geometry=part))
    coarse = PostedEvaluator().evaluate_candidate(task, fixture.motion, {"subdivisions": 1})
    midpoint_volume = 12 * math.pi
    assert metric(coarse, "removed_volume_mm3") == 0.0  # two far endpoint poses only
    assert coarse.discretization["volume_bound"]["upper_mm3"] >= midpoint_volume
    record = coarse.discretization["segments"][0]
    midpoint = (20 + 10 / math.sqrt(2), 20 + 10 / math.sqrt(2), 0)
    nearest = min(math.dist(midpoint, segment.start_mm), math.dist(midpoint, segment.end_mm))
    assert record["temporal_station_cover_bound_mm"] >= nearest
    refined = PostedEvaluator().evaluate_candidate(task, fixture.motion, {"subdivisions": 2})
    assert metric(refined, "removed_volume_mm3") > 0
    assert refined.discretization["volume_bound"]["lower_mm3"] <= midpoint_volume <= refined.discretization["volume_bound"]["upper_mm3"]


def _corner_geometry(pitch):
    # Pocket corners are each a 2x2 quadrant. In particular x=2 is never a
    # cell centre for dyadic pitch, so the tiny convex island is unresolved,
    # not accidentally rounded or erased by the new unreachable-corner code.
    side = 4.0
    small_island = [1.99, 1.99, 0.0]
    params = {
        "primitive": "box_minus_pocket",
        "box_min_mm": [0, 0, 0], "box_max_mm": [side, side, pitch],
        "pocket_min_mm": [0, 0, 0], "pocket_max_mm": [side, side, pitch],
        "island_min_mm": small_island, "island_max_mm": [2.01, 2.01, pitch],
        "pocket_unreached_corner_radius_mm": 2.0,
    }
    shape = (int(side / pitch), int(side / pitch), 1)
    grid = C.GridSpec(shape, (0, 0, 0), pitch, C.Frame.WORK)
    return C.GeometrySource(kind="analytic", params=params, frame=C.Frame.WORK), grid


def test_pocket_reachable_corner_volume_converges_to_quarter_disc_complement():
    expected_volume = 4 * (1 - math.pi / 4) * 2 ** 2 * 6
    # The coordinator's reported 20.601522612 differs by 0.00025370 mm3 from
    # this explicit closed form. Never fabricate agreement by changing pi or
    # using a percentage criterion across unrelated radii/grid sizes.
    assert expected_volume == pytest.approx(20.601522612, abs=0.001)
    history = []
    for pitch in (0.125, 0.0625, 0.03125):
        source, grid = _corner_geometry(pitch)
        points = _centres(grid)
        actual = _analytic_mask(source, points)
        assert actual.any()
        without_corner = replace(source, params=dict(source.params,
                                                       pocket_unreached_corner_radius_mm=0.0))
        assert not _analytic_mask(without_corner, points).any()  # hidden tiny island only
        # Every sampled Z point is within the full vertical corner window. Each
        # cell occupies height pitch; normalized to six mm retained corner depth.
        volume = float(actual.sum()) * pitch ** 3 * (6 / pitch)
        history.append((pitch, volume, abs(volume - expected_volume)))
    print("Fillets by pitch:", history)
    assert history[0][2] > history[1][2]
    assert history[-1][2] < 0.15
    assert history[-1][2] < history[0][2]


def test_pocket_corner_radius_zero_is_byte_identical_to_old_mask():
    fixture = C.example_pocket_task()
    source, grid = fixture.operation_target.geometry, fixture.grid
    points = _centres(grid)
    zero = replace(source, params=dict(source.params, pocket_unreached_corner_radius_mm=0.0))
    expected = _analytic_mask(source, points)
    np.testing.assert_array_equal(_analytic_mask(zero, points), expected)
    # Absence is the same code path, not a newly defaulted physical fillet.
    assert "pocket_unreached_corner_radius_mm" not in source.params
    bad = replace(source, params=dict(source.params, pocket_unreached_corner_radius_mm=-1.0))
    with pytest.raises(ValueError, match="nonnegative"):
        _analytic_mask(bad, points)
    substantial_island = np.logical_and.reduce([
        (points[axis] >= source.params["island_min_mm"][axis])
        & (points[axis] <= source.params["island_max_mm"][axis]) for axis in range(3)])
    assert substantial_island.any()
    rounded = replace(source, params=dict(source.params, pocket_unreached_corner_radius_mm=0.75))
    np.testing.assert_array_equal(_analytic_mask(rounded, points)[substantial_island],
                                  expected[substantial_island])


def test_pocket_fillet_retention_stays_inside_declared_outer_stock():
    source, grid = _corner_geometry(0.125)
    points = _centres(grid)
    exterior = replace(source, params=dict(
        source.params,
        box_min_mm=[1, 1, 0], box_max_mm=[3, 3, grid.voxel_mm],
        island_min_mm=[1.5, 1.5, 0], island_max_mm=[2.5, 2.5, grid.voxel_mm],
        pocket_unreached_corner_radius_mm=1.0))
    actual = _analytic_mask(exterior, points)
    outside_outer = (points[0] < 1) | (points[0] > 3) | (points[1] < 1) | (points[1] > 3)
    assert outside_outer.any()
    assert not actual[outside_outer].any()  # corner square outside stock is air, not material
    # The convex island is retained without any corner-fillet subtraction.
    island = ((points[0] >= 1.5) & (points[0] <= 2.5)
              & (points[1] >= 1.5) & (points[1] <= 2.5))
    assert island.any()
    assert actual[island].all()


def test_pocket_fillets_are_excluded_from_original_part_and_checked_at_quarter_mm():
    pitch = 0.25
    # A real six-mm-deep pocket at the requester's pitch; the tiny central island
    # is unchanged but unresolved at this lattice. Outer stock equals pocket bounds.
    fixture = F.make_fixture("corner-consistency", [], F.flat_tool(1, 20), 0.0, "corner exclusion",
                             stock_lo=(0, 0, 0), stock_hi=(4, 4, 6), voxel=pitch, initial=(2, 2, 8))
    rounded, _ = _corner_geometry(pitch)
    depth6 = replace(rounded, params=dict(rounded.params, box_max_mm=[4, 4, 6],
                                         pocket_max_mm=[4, 4, 6], island_max_mm=[2.01, 2.01, 6]))
    sharp = replace(depth6, params=dict(depth6.params, pocket_unreached_corner_radius_mm=0.0))
    task = replace(fixture.task, original_part=sharp,
                   operation_target=C.OperationTarget(
                       geometry=depth6,
                       excluded_volume_mm3=20.601522612,
                       exclusion_reason="continuous pocket-corner reachability exclusion; explicit pitch tolerance supplied"))
    # Independent single-corner count in physical mm: 12 cells in one 2x2 corner.
    coordinates = [0.125 + 0.25 * i for i in range(8)]
    count = sum((2 - x) ** 2 + (2 - y) ** 2 >= 2 ** 2 for x in coordinates for y in coordinates)
    assert count == 12
    expected_excluded = 4 * count * 6 * pitch ** 2
    assert expected_excluded == 18.0
    motion = C.MotionPlan(C.Frame.WORK, task.initial_state, (), C.TimingRule.EXACT_STOP_CONSTANT_SPEED)
    result = PostedEvaluator().evaluate_candidate(task, motion, {"exclusion_tolerance_mm3": 2.65})
    assert result.coverage == C.Coverage.COMPLETE
    # No cuts were executed; this is an explicit metric/exclusion test, not pass physics.
    assert result.geometry_pass is False
    assert metric(result, "excluded_volume_mm3") == expected_excluded
    assert result.discretization["settings"]["exclusion_tolerance_mm3"] == 2.65
    assert abs(task.operation_target.excluded_volume_mm3 - expected_excluded) <= 2.65


def test_major_arc_nearest_station_bound_does_not_vanish_near_full_circle():
    angle = math.radians(359)
    start, end = (30, 20, 0), (20 + 10 * math.cos(angle), 20 + 10 * math.sin(angle), 0)
    arc = C.Segment("major", C.SegmentKind.ARC, start, end, 10 * angle / 10, 10,
                    C.ArcData((20, 20, 0), C.ArcPlane.XY, C.ArcDirection.CCW))
    fixture = F.make_fixture("major", [arc], F.flat_tool(1, 20), 0.0, "near full-circle endpoint distance")
    result = PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {"subdivisions": 1})
    midpoint = (20 + 10 * math.cos(angle / 2), 20 + 10 * math.sin(angle / 2), 0)
    nearest = min(math.dist(midpoint, start), math.dist(midpoint, end))
    assert nearest > 19.9
    bound = result.discretization["segments"][0]["temporal_station_cover_bound_mm"]
    assert bound >= nearest
    assert bound >= 2 * 10


def test_uncertain_rapid_does_not_taint_later_distinct_safe_rapid_segment():
    fixture = F.retract_from_own_slot()
    high_start = C.MachineState((20, 20, 100), C.Frame.WORK)
    high_task = replace(fixture.task, initial_state=high_start)
    high_only = F.rapid("high", (20, 20, 100), (30, 20, 100))
    high_motion = C.MotionPlan(C.Frame.WORK, high_start, (high_only,), C.TimingRule.EXACT_STOP_CONSTANT_SPEED)
    assert PostedEvaluator().evaluate_candidate(high_task, high_motion, {}).safety_pass is True
    segments = fixture.motion.segments + (
        F.rapid("rise", (20, 20, 16), (20, 20, 100)),
        F.rapid("high", (20, 20, 100), (30, 20, 100)),
    )
    motion = replace(fixture.motion, segments=segments)
    result = PostedEvaluator().evaluate_candidate(fixture.task, motion, {})
    assert result.safety_pass is None  # caused by the pre-existing own-slot retract
    assert any("retract" in reason for reason in result.coverage_reasons)
    assert all("high" not in reason for reason in result.coverage_reasons)


def test_full_circle_arc_removes_annulus_not_nothing_and_has_nonzero_interval_bound():
    # Cutter radius 3 follows a radius-9 ring, with flute 20 and bottom at 0.
    # Its engagement is therefore the full 12mm stock height. The holder begins
    # above the stock, so this geometry is safe and the target is a small center block.
    fixture = F.horizontal_slot()
    geometry = C.GeometrySource(kind="analytic", params={"primitive": "box", "min_mm": [24, 24, 0], "max_mm": [26, 26, 12]})
    task = simple_removal_task(fixture)
    task = replace(task, operation_target=C.OperationTarget(geometry=geometry), original_part=geometry,
                   tool=F.flat_tool(3, 20), initial_state=replace(task.initial_state, position_mm=(29, 20, 0)))
    segment = C.Segment("ring", C.SegmentKind.ARC, (29, 20, 0), (29, 20, 0),
                        duration_s=2 * math.pi * 9 / 10, feed_mm_s=10,
                        arc=C.ArcData((20, 20, 0), C.ArcPlane.XY, C.ArcDirection.CW))
    motion = C.MotionPlan(C.Frame.WORK, task.initial_state, (segment,), C.TimingRule.EXACT_STOP_CONSTANT_SPEED)
    motion.validate_for_task(task)
    result = PostedEvaluator().evaluate_candidate(task, motion, {})
    expected_volume = math.pi * (12 ** 2 - 6 ** 2) * 12
    assert metric(result, "removed_volume_mm3") > 500
    upper = result.discretization["volume_bound"]
    assert upper["lower_mm3"] <= expected_volume <= upper["upper_mm3"]
    assert result.discretization["segments"][0]["motion_model"] == "contracts.arc_geometry"
    assert result.discretization["segments"][0]["temporal_station_cover_bound_mm"] == pytest.approx(math.pi * 9 / 256, rel=1e-12)
    coarse = PostedEvaluator().evaluate_candidate(task, motion, {"subdivisions": 1})
    assert coarse.discretization["segments"][0]["temporal_station_cover_bound_mm"] == pytest.approx(math.pi * 9)
    assert metric(result, "unique_contacted_volume_mm3") == 0.0


@pytest.mark.parametrize("plane,center,start,end,expected_mid", [
    (C.ArcPlane.XY, (20, 20, 4), (26, 20, 4), (14, 20, 4), (20, 26, 4)),
    (C.ArcPlane.XZ, (20, 20, 4), (20, 20, -2), (20, 20, 10), (14, 20, 4)),
])
def test_arc_planes_sample_real_midpoints(plane, center, start, end, expected_mid):
    fixture = F.zero_length_feed()
    task = replace(fixture.task, initial_state=replace(fixture.task.initial_state, position_mm=start))
    segment = C.Segment("curve", C.SegmentKind.ARC, start, end, math.pi * 6 / 10, feed_mm_s=10,
                        arc=C.ArcData(center, plane, C.ArcDirection.CCW))
    motion = C.MotionPlan(C.Frame.WORK, task.initial_state, (segment,), C.TimingRule.EXACT_STOP_CONSTANT_SPEED)
    # Direct candidate is nonempty only on the curved station, never on a
    # straight chord. The cyclic XZ basis is (Z,X,Y), so this CCW half-circle
    # passes through x=14,z=4 rather than its chord (x=20,z=4) or opposite arc.
    result = PostedEvaluator().evaluate_candidate(task, motion, {})
    assert metric(result, "removed_volume_mm3") > 0
    station_record = result.discretization["segments"][0]
    midpoint = station_record["stations_work_mm"][len(station_record["stations_work_mm"]) // 2]
    np.testing.assert_allclose(midpoint, expected_mid, atol=1e-9, rtol=0)
    nc_task, nc_motion = machine_plan(task, motion, C.Transform.translation((100, 80, -150)))
    expected = PostedEvaluator().evaluate_candidate(nc_task, nc_motion, {})
    raw = post_motion_plan(nc_motion, nc_task).encode()
    posted = PostedEvaluator().evaluate_posted_gcode(nc_task, raw, C.Dialect.RS274, {})
    assert metric(posted, "removed_volume_mm3") == metric(expected, "removed_volume_mm3")
    assert posted.discretization["segments"][0]["motion_model"] == "contracts.arc_geometry"


def test_rotated_workframe_arc_is_mapped_back_before_geometry():
    transform = C.Transform(((0, -1, 0, 100), (1, 0, 0, 80), (0, 0, 1, -150), (0, 0, 0, 1)))
    start, center, end = (20, 20, -2), (20, 20, 4), (20, 20, 10)
    task = replace(F.zero_length_feed().task, work_to_machine=transform,
                   initial_state=C.MachineState(transform.apply(start), C.Frame.MACHINE))
    raw = b"G21 G90 G18 G54 F600\nG3 X20 Z10 K6\nM2\n"
    parsed = parse_motion_plan(raw.decode(), task)
    assert parsed.segments[0].arc.plane == C.ArcPlane.YZ
    assert parsed.segments[0].arc.direction == C.ArcDirection.CW
    result = PostedEvaluator().evaluate_posted_gcode(task, raw, C.Dialect.RS274, {})
    assert result.coverage == C.Coverage.COMPLETE
    assert metric(result, "removed_volume_mm3") > 0
    midpoint = result.discretization["segments"][0]["stations_work_mm"][32]
    np.testing.assert_allclose(midpoint, (14, 20, 4), atol=1e-9, rtol=0)

    task = F.plunge_above().task
    result = PostedEvaluator().evaluate_posted_gcode(task, b"G21 G90 G54\nM2\n", C.Dialect.RS274, {})
    assert result.coverage == C.Coverage.UNSUPPORTED
    assert result.safety_pass is None
    assert "machine frame" in result.coverage_reasons[0]


def test_rapid_whole_cell_penetration_bound_controls_certification_not_collision():
    fixture = F.retract_from_own_slot()
    unknown = PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {})
    assert unknown.safety_pass is None
    assert metric(unknown, "collision_sample_count") == 0.0
    upper = metric(unknown, "rapid_cutter_penetration_upper_bound_mm")
    certified = F.with_rapid_tolerance(fixture, upper)
    result = PostedEvaluator().evaluate_candidate(certified.task, certified.motion, {})
    assert result.safety_pass is True
    assert metric(result, "unique_contacted_volume_mm3") == 0.0
    assert metric(result, "min_clearance_mm") > 10
    assert result.metrics["min_rapid_cutter_clearance_lower_bound_mm"].value == pytest.approx(-upper)


def test_phase_misaligned_stock_cells_still_bound_penetration():
    for phase in (0.0, 0.5, 0.99):
        base = F.make_fixture("graze", [F.rapid("r", (-10, 20, 11.01), (70, 20, 11.01))],
                              F.flat_tool(3, 20), 0.0, "0.99mm true penetration")
        phased = F.with_joint_phase(base, (0, 0, phase))
        phased = F.with_rapid_tolerance(phased, 0.8)
        result = PostedEvaluator().evaluate_candidate(phased.task, phased.motion,
                                                      {"max_step_mm": 0.01, "max_samples_per_segment": 16384})
        upper = metric(result, "rapid_cutter_penetration_upper_bound_mm")
        assert upper >= 0.99
        assert metric(result, "min_rapid_cutter_clearance_lower_bound_mm") == pytest.approx(-upper)
        assert result.safety_pass is not True


def test_sampling_budget_and_disabled_safety_screen_are_explicit():
    fixture = F.horizontal_slot()
    capped = PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {"max_samples_per_segment": 2})
    assert capped.coverage == C.Coverage.UNSUPPORTED
    assert capped.safety_pass is None
    assert "stations" in capped.coverage_reasons[0]
    disabled = PostedEvaluator().evaluate_candidate(fixture.task, fixture.motion, {"safety_screen": False})
    assert disabled.safety_pass is None
    assert disabled.metrics["min_clearance_lower_bound_mm"].value is None
