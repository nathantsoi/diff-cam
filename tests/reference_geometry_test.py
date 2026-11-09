"""T1.0 independent flat-cutter geometry and chronology checks."""

import ast
import dataclasses
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks import contracts as C
from verification import fixtures as F
from verification.reference_carve import run_reference


def metric(trace, name):
    value = trace.result.metrics[name].value
    assert value is not None
    return value


def test_reference_has_no_production_geometry_imports():
    roots = {"simulator", "cam"}
    for path in (Path("verification/reference_carve.py"), Path("verification/fixtures.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in roots for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in roots


def test_fresh_process_loads_no_production_geometry_modules():
    code = (
        "import sys; import verification.reference_carve, verification.fixtures; "
        "bad=[m for m in sys.modules if m == 'simulator' or m.startswith('simulator.') "
        "or m == 'cam' or m.startswith('cam.')]; "
        "assert not bad, bad"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("fixture", F.all_fixtures(), ids=lambda f: f.fixture_id)
def test_independent_removed_volume_is_inside_certified_bracket(fixture):
    trace = run_reference(fixture.task, fixture.motion)
    assert trace.result.evaluator_id == "reference_carve"
    assert trace.result.coverage == C.Coverage.COMPLETE
    assert trace.result.motion_hash == fixture.motion.motion_hash
    assert trace.bounds is not None
    eps = 1e-8
    assert trace.bounds.lower_mm3 - eps <= fixture.removed_volume_mm3 <= trace.bounds.upper_mm3 + eps
    sampled = metric(trace, "removed_volume_mm3")
    assert trace.bounds.lower_mm3 <= sampled <= trace.bounds.upper_mm3
    assert sampled == pytest.approx(float(trace.removed.sum()) * fixture.task.grid.voxel_mm ** 3)
    assert trace.result.geometry_pass is None
    assert trace.result.metrics["gouge_volume_mm3"].value is None


@pytest.mark.parametrize(
    "factory,radii",
    [
        (F.plunge_above, (2.35, 3.0, 3.65)),
        (F.horizontal_slot, (2.35, 3.0, 3.65)),
        (F.sloped_ramp, (1.35, 2.0, 2.65)),
        (F.near_vertical, (2.35, 3.0, 3.65)),
        (F.through_hole, (3.35, 4.0, 4.65)),
        (F.zero_length_feed, (2.35, 3.0, 3.65)),
    ],
)
def test_radius_and_fractional_xy_phase_variants(factory, radii):
    for radius in radii:
        fixture = factory(radius=radius, phase_xy=(0.26, 0.58))
        trace = run_reference(fixture.task, fixture.motion)
        assert trace.bounds.lower_mm3 <= fixture.removed_volume_mm3 <= trace.bounds.upper_mm3


@pytest.mark.parametrize("factory", [F.plunge_above, F.sloped_ramp, F.cutter_leading_holder_safe])
def test_joint_stock_tool_translation_changes_grid_phase_not_continuous_volume(factory):
    original = factory()
    phased = F.with_joint_phase(original, (0.13 * original.task.grid.voxel_mm,
                                           0.29 * original.task.grid.voxel_mm,
                                           0.37 * original.task.grid.voxel_mm))
    trace = run_reference(phased.task, phased.motion)
    assert trace.bounds.lower_mm3 <= original.removed_volume_mm3 <= trace.bounds.upper_mm3
    assert phased.motion.motion_hash != original.motion.motion_hash


def test_initial_pose_does_not_cut_and_zero_length_feed_does():
    empty = F.empty_preengaged()
    zero = F.zero_length_feed()
    empty_trace = run_reference(empty.task, empty.motion)
    zero_trace = run_reference(zero.task, zero.motion, {"subdivisions": 7})
    assert metric(empty_trace, "removed_volume_mm3") == 0
    assert empty_trace.result.safety_pass is True
    assert metric(zero_trace, "removed_volume_mm3") > 0
    assert metric(zero_trace, "time_integrated_overlap_mm3_s") == 0
    assert zero_trace.result.discretization["segments"][0]["endpoints_included"]


def test_repeated_and_reversed_paths_are_geometrically_idempotent():
    single = F.plunge_above()
    repeated = F.repeated_plunge()
    a = run_reference(single.task, single.motion)
    b = run_reference(repeated.task, repeated.motion)
    assert np.array_equal(a.remaining, b.remaining)

    rev = F.reversal()
    one_leg = F.make_fixture("one_leg", [F.feed("outward", (12, 20, 0), (40, 20, 0))],
                             F.flat_tool(3, 20), rev.removed_volume_mm3, rev.derivation)
    c = run_reference(one_leg.task, one_leg.motion)
    d = run_reference(rev.task, rev.motion)
    assert np.array_equal(c.remaining, d.remaining)


def test_fixed_grid_temporal_refinement_nests_station_removal_and_bounds():
    fixture = F.sloped_ramp()
    previous = None
    previous_bounds = None
    volumes = []
    for subdivisions in (4, 8, 16, 32):
        trace = run_reference(fixture.task, fixture.motion, {"subdivisions": subdivisions})
        if previous is not None:
            assert not np.any(previous & ~trace.removed)
        if previous_bounds is not None:
            assert trace.bounds.lower_mm3 >= previous_bounds.lower_mm3 - 1e-8
            assert trace.bounds.station_upper_mm3 >= previous_bounds.station_upper_mm3 - 1e-8
            assert trace.bounds.upper_mm3 <= previous_bounds.upper_mm3 + 1e-8
        previous = trace.removed.copy()
        previous_bounds = trace.bounds
        volumes.append(metric(trace, "removed_volume_mm3"))
    assert all(b >= a for a, b in zip(volumes, volumes[1:]))
    assert volumes[0] < volumes[-1]


def test_safe_cutter_leading_holder_converges_from_uncertain_to_safe():
    fixture = F.cutter_leading_holder_safe()
    coarse = run_reference(fixture.task, fixture.motion, {"subdivisions": 1, "compute_bounds": False})
    fine = run_reference(fixture.task, fixture.motion, {"subdivisions": 16, "compute_bounds": False})
    # With only two poses, intermediate cutter poses are absent and can leave
    # artificial remaining stock for the endpoint holder. This may be a coarse
    # false collision (false) and/or an uncertain conservative bound (None).
    assert coarse.result.safety_pass in (None, False)
    assert fine.result.safety_pass is True
    assert metric(fine, "removed_volume_mm3") > metric(coarse, "removed_volume_mm3")
    assert metric(fine, "unique_contacted_volume_mm3") == 0
    assert metric(fine, "time_integrated_overlap_mm3_s") == 0


def test_genuine_holder_count_unique_volume_and_integral_are_distinct():
    fixture = F.genuine_holder_collision()
    traces = {m: run_reference(fixture.task, fixture.motion, {"subdivisions": m}) for m in (3, 6)}
    # Independent v=2 lattice derivation. Cell centres have odd x/y offsets from (20,20).
    # At final Z=11, strict annulus 36 < dx^2+dy^2 < 100 has row-pair counts
    # |dy|=1,3,5,7,9 -> 4,4,4,8,4 per row, hence 48 cells total: 48*8=384 mm^3.
    expected_unique = 48 * 8.0
    assert metric(traces[3], "collision_sample_count") == 1
    assert metric(traces[6], "collision_sample_count") == 2
    for trace in traces.values():
        assert trace.result.safety_pass is False
        assert metric(trace, "unique_contacted_volume_mm3") == expected_unique
    # m=3 integrates only the endpoint with half weight 1/6. At m=6, lambda=5/6
    # and 1 both occupy this same single-Z-layer lattice mask; weights are 1/6,1/12.
    assert metric(traces[3], "time_integrated_overlap_mm3_s") == pytest.approx(expected_unique / 6)
    assert metric(traces[6], "time_integrated_overlap_mm3_s") == pytest.approx(expected_unique / 4)
    assert fixture.unique_contacted_volume_mm3 == pytest.approx(192 * math.pi)
    assert fixture.time_integrated_overlap_mm3_s == pytest.approx(32 * math.pi)

    expected_contact_s = fixture.first_events["holder_collision"][1]
    for m, trace in traces.items():
        event = next(e for e in trace.result.events if e.kind == "holder_collision")
        assert event.segment_id == "wide_plunge"
        assert event.s - 1 / m - 1e-12 <= expected_contact_s <= event.s + 1e-12


def test_zero_length_noncutting_screen_is_continuous_in_length():
    tool = F.flat_tool(3, 6, holder_radius=10)
    zero = F.make_fixture("zero", [F.feed("zero", (20, 20, 5.5), (20, 20, 5.5))], tool, 0.0, "zero")
    tiny = F.make_fixture("tiny", [F.feed("tiny", (20, 20, 5.5), (20, 20, 5.5 + 1e-9))], tool, 0.0, "tiny")
    zero_trace = run_reference(zero.task, zero.motion)
    tiny_trace = run_reference(tiny.task, tiny.motion)
    assert zero_trace.result.safety_pass is None
    assert tiny_trace.result.safety_pass is None
    assert metric(zero_trace, "min_clearance_lower_bound_mm") < 0
    assert metric(zero_trace, "min_clearance_lower_bound_mm") == pytest.approx(
        metric(tiny_trace, "min_clearance_lower_bound_mm"), abs=1e-7)
    assert metric(zero_trace, "min_clearance_mm") == metric(tiny_trace, "min_clearance_mm") == 0.5
    assert "centre-sampled only" in zero_trace.result.metrics["min_clearance_mm"].denominator


def test_phase_misaligned_rapid_penetration_bound_covers_analytic_value():
    base = F.make_fixture("graze", [F.rapid("r", (-10, 20, 11.01), (70, 20, 11.01))],
                          F.flat_tool(3, 20), 0.0, "0.99 mm true penetration")
    for z_phase in (0.0, 0.13, 0.5, 0.99):
        phased = F.with_joint_phase(base, (0.0, 0.0, z_phase))
        phased = F.with_rapid_tolerance(phased, 0.8)
        trace = run_reference(phased.task, phased.motion, {"max_step_mm": 0.01})
        upper = metric(trace, "rapid_cutter_penetration_upper_bound_mm")
        lower = metric(trace, "min_rapid_cutter_clearance_lower_bound_mm")
        assert upper >= 0.99
        assert lower == pytest.approx(-upper, abs=1e-9)
        assert trace.result.safety_pass is not True
        assert "whole-cell" in trace.result.discretization["clearance_metrics"]


def test_phase_boundary_noncutting_clearance_uses_lower_bound():
    base = F.make_fixture("hg", [F.feed("f", (-15, 20, 7.01), (75, 20, 7.01))],
                          F.flat_tool(3, 4, holder_radius=10), 0.0, "0.99 mm holder intrusion")
    phased = F.with_joint_phase(base, (0.0, 0.0, 0.99))
    trace = run_reference(phased.task, phased.motion, {"max_step_mm": 0.01, "compute_bounds": False})
    assert metric(trace, "min_clearance_lower_bound_mm") < 0
    assert trace.result.safety_pass is not True
    # Preserve the centre-only diagnostic separately from the solved whole-cell bound.
    assert metric(trace, "min_clearance_mm") == 1.0
    assert "centre-sampled only" in trace.result.metrics["min_clearance_mm"].denominator


def test_stock_beyond_grid_and_disabled_screen_cannot_certify_safety():
    fixture = F.plunge_above()
    task = dataclasses.replace(fixture.task, grid=dataclasses.replace(fixture.task.grid, shape=(5, 5, 5)))
    result = run_reference(task, fixture.motion).result
    assert result.coverage == C.Coverage.UNSUPPORTED
    assert result.safety_pass is None
    assert "beyond represented grid" in result.coverage_reasons[0]

    fixture = F.retract_from_own_slot()
    unknown = run_reference(fixture.task, fixture.motion)
    upper = metric(unknown, "rapid_cutter_penetration_upper_bound_mm")
    adjusted = F.with_rapid_tolerance(fixture, upper)
    disabled = run_reference(adjusted.task, adjusted.motion, {"safety_screen": False}).result
    assert disabled.safety_pass is None
    assert "safety_screen disabled" in disabled.coverage_reasons[0]


def test_rapid_crosses_solid_without_carving():
    fixture = F.rapid_through_solid()
    trace = run_reference(fixture.task, fixture.motion, {"subdivisions": 128})
    assert trace.result.safety_pass is False
    assert metric(trace, "removed_volume_mm3") == 0
    assert np.all(trace.remaining)
    # Independent v=2 occupied-centre expectation: 30 x columns, two y rows.
    # Cutter occupies Z=7,9 (30*2*2=120 cells), holder adds Z=11 (60 more).
    assert metric(trace, "unique_contacted_volume_mm3") == 180 * 8.0
    by_component = trace.result.discretization["unique_by_component_mm3"]
    assert by_component["rapid_cutter"] == 120 * 8.0
    event = next(e for e in trace.result.events if e.kind == "rapid_collision")
    assert event.segment_id == "rapid"
    # First occupied strict-collision centre is x=1,y=19: entry requires
    # sqrt((1-centre_x)^2+1) < 3. The m=128 station sequence first satisfies
    # this at j=14 (centre_x=-1.25), later than continuous stock-edge contact.
    assert event.s == pytest.approx(14 / 128)
    assert event.s >= fixture.first_events["rapid_collision"][1]


def test_vertical_retract_is_uncertain_at_zero_and_certified_at_penetration_bound():
    fixture = F.retract_from_own_slot()
    unknown = run_reference(fixture.task, fixture.motion)
    assert unknown.result.safety_pass is None
    assert metric(unknown, "collision_sample_count") == 0
    upper = metric(unknown, "rapid_cutter_penetration_upper_bound_mm")
    assert upper > 0
    # Split metrics: the holder is far above the stock, while the cutter grazes its slot.
    assert metric(unknown, "min_clearance_mm") > 10
    assert metric(unknown, "min_rapid_cutter_clearance_mm") >= -1e-12

    adjusted = F.with_rapid_tolerance(fixture, upper)
    allowed = run_reference(adjusted.task, adjusted.motion)
    assert allowed.result.safety_pass is True
    assert allowed.result.discretization["tolerances"]["clearance_mm_role"].startswith("reported qualification")
    assert metric(allowed, "rapid_cutter_penetration_upper_bound_mm") == pytest.approx(upper)
    assert metric(allowed, "collision_sample_count") == 0
    assert metric(allowed, "unique_contacted_volume_mm3") == 0
    assert metric(allowed, "min_clearance_mm") > 10
    feed_only = run_reference(F.plunge_above().task, F.plunge_above().motion)
    assert feed_only.result.metrics["min_rapid_cutter_clearance_mm"].value is None
    assert feed_only.result.metrics["rapid_cutter_penetration_upper_bound_mm"].value is None


def test_rapid_penetration_tolerance_is_phase_stable_and_deep_penetration_still_fails():
    fixture = F.retract_from_own_slot()
    phased = F.with_joint_phase(fixture, (0.26, 0.58, 0.74))
    cases = (fixture, phased)
    uppers = []
    for case in cases:
        uncertain = run_reference(case.task, case.motion)
        assert uncertain.result.safety_pass is None
        uppers.append(metric(uncertain, "rapid_cutter_penetration_upper_bound_mm"))
    # Partially carved boundary cells can hold stock beyond their removed centre.
    # The complete whole-cell bound therefore supersedes rho+delta_t alone.
    band = max(uppers)
    for case in cases:
        adjusted = F.with_rapid_tolerance(case, band)
        certified = run_reference(adjusted.task, adjusted.motion)
        assert certified.result.safety_pass is True
        assert metric(certified, "rapid_cutter_penetration_upper_bound_mm") <= band + 1e-12

    solid = F.rapid_through_solid()
    solid = dataclasses.replace(solid, task=dataclasses.replace(solid.task, tool=F.flat_tool(3, 20)))
    for tolerance in (0.0, 0.25, 1.5):
        adjusted = F.with_rapid_tolerance(solid, tolerance)
        result = run_reference(adjusted.task, adjusted.motion)
        assert result.result.safety_pass is False
        assert any(event.kind == "rapid_collision" for event in result.result.events)
        assert metric(result, "min_rapid_cutter_clearance_mm") < -1.5
        assert metric(result, "min_clearance_mm") > 10


def test_zero_length_rapid_checks_assembly_without_cutting():
    fixture = F.zero_length_rapid()
    trace = run_reference(fixture.task, fixture.motion)
    assert trace.result.safety_pass is False
    assert metric(trace, "removed_volume_mm3") == 0
    assert metric(trace, "collision_sample_count") == 1
    assert metric(trace, "time_integrated_overlap_mm3_s") == 0
    # Four strict-radial cells at each Z=5,7,9,11, each 8 mm^3.
    assert metric(trace, "unique_contacted_volume_mm3") == 128.0
    assert trace.result.events[0].s == 0.0


def test_noncutting_shank_is_checked_separately_from_holder():
    tool = C.ToolAssembly(cutter_kind=C.CutterKind.FLAT, cutter_radius_mm=2.0, flute_length_mm=4.0,
                          shank_radius_mm=4.0, holder_bottom_mm=8.0,
                          holder_radius_mm=4.0, holder_height_mm=20.0)
    fixture = F.make_fixture(
        "shank_collision", [F.feed("shank_plunge", (20, 20, 14), (20, 20, 6))], tool,
        24 * math.pi, "pi*2^2*6 removed; wider shank annulus has depth 2",
        safety_pass=False, unique_contacted_volume_mm3=24 * math.pi,
    )
    trace = run_reference(fixture.task, fixture.motion)
    assert trace.result.safety_pass is False
    assert trace.bounds.lower_mm3 <= fixture.removed_volume_mm3 <= trace.bounds.upper_mm3
    assert any(e.kind == "shank_collision" for e in trace.result.events)
    assert not any(e.kind == "holder_collision" for e in trace.result.events)
    assert trace.result.discretization["unique_by_component_mm3"]["shank"] > 0
    assert trace.result.discretization["unique_by_component_mm3"]["holder"] == 0


def test_machine_frame_motion_is_transformed_to_work_independently():
    fixture = F.plunge_above()
    transform = C.Transform.translation((100, 50, -20))
    state = C.MachineState(position_mm=transform.apply(fixture.task.initial_state.position_mm), frame=C.Frame.MACHINE)
    segments = tuple(dataclasses.replace(s, start_mm=transform.apply(s.start_mm), end_mm=transform.apply(s.end_mm))
                     for s in fixture.motion.segments)
    task = dataclasses.replace(fixture.task, work_to_machine=transform, initial_state=state)
    motion = dataclasses.replace(fixture.motion, frame=C.Frame.MACHINE, initial_state=state, segments=segments)
    motion.validate_for_task(task)
    work = run_reference(fixture.task, fixture.motion)
    machine = run_reference(task, motion)
    assert np.array_equal(work.remaining, machine.remaining)
    assert metric(work, "removed_volume_mm3") == metric(machine, "removed_volume_mm3")


def test_ball_arc_fixture_and_unknown_state_are_explicitly_unsupported():
    fixture = F.plunge_above()
    ball_task = dataclasses.replace(fixture.task, tool=dataclasses.replace(fixture.task.tool, cutter_kind=C.CutterKind.BALL))
    ball = run_reference(ball_task, fixture.motion)
    assert ball.result.coverage == C.Coverage.UNSUPPORTED
    assert ball.result.safety_pass is None and ball.result.metrics["removed_volume_mm3"].value is None

    slot = F.horizontal_slot()
    old = slot.motion.segments[0]
    arc = C.ArcData(centre_mm=(26, 20, 0), plane=C.ArcPlane.XY, direction=C.ArcDirection.CCW)
    segment = dataclasses.replace(old, kind=C.SegmentKind.ARC, arc=arc,
                                  duration_s=math.pi * 12 / old.feed_mm_s)
    motion = dataclasses.replace(slot.motion, segments=(segment,))
    arc_result = run_reference(slot.task, motion).result
    assert arc_result.coverage == C.Coverage.UNSUPPORTED and not arc_result.events

    fixtures_task = dataclasses.replace(fixture.task, fixtures=(fixture.task.stock,))
    assert run_reference(fixtures_task, fixture.motion).result.coverage == C.Coverage.UNSUPPORTED

    unknown = C.MachineState(position_mm=None, known=False)
    unknown_task = dataclasses.replace(fixture.task, initial_state=unknown)
    unknown_motion = dataclasses.replace(fixture.motion, initial_state=unknown)
    assert run_reference(unknown_task, unknown_motion).result.coverage == C.Coverage.UNSUPPORTED


@pytest.mark.skipif(os.environ.get("DIFFCNC_REFERENCE_CONVERGENCE") != "1",
                    reason="set DIFFCNC_REFERENCE_CONVERGENCE=1 for explicit refinement report validation")
def test_saved_convergence_sweep_tightens_bounds_and_settles_chronology():
    report = F.generate_convergence_report()
    for case in report["continuous_volume_cases"]:
        rows = case["rows"]
        assert all(row["analytic_inside_bounds"] for row in rows)
        widths = [row["bounds"]["width_mm3"] for row in rows]
        lowers = [row["bounds"]["lower_mm3"] for row in rows]
        uppers = [row["bounds"]["upper_mm3"] for row in rows]
        assert all(b <= a + 1e-8 for a, b in zip(widths, widths[1:]))
        assert all(b >= a - 1e-8 for a, b in zip(lowers, lowers[1:]))
        assert all(b <= a + 1e-8 for a, b in zip(uppers, uppers[1:]))
    safe = next(c for c in report["chronology_cases"] if c["fixture_id"] == "cutter_leading_holder_safe")
    assert safe["rows"][0]["safety_pass"] in (None, False)
    assert safe["rows"][-1]["safety_pass"] is True
    collision = next(c for c in report["chronology_cases"] if c["fixture_id"] == "genuine_holder_collision")
    assert all(row["safety_pass"] is False for row in collision["rows"])
    assert report["contracts_schema"] == C.SCHEMA_VERSION
    rapid_rows = {row["case"]: row for row in report["rapid_tolerance_cases"]}
    assert rapid_rows["retract_zero"]["safety_pass"] is None
    assert rapid_rows["retract_at_own_upper_bound"]["safety_pass"] is True
    assert rapid_rows["retract_band_phase_0"]["safety_pass"] is True
    assert rapid_rows["retract_zero_phase_fractional"]["safety_pass"] is None
    assert rapid_rows["retract_band_phase_fractional"]["safety_pass"] is True
    assert rapid_rows["rapid_through_solid_at_retract_upper_bound"]["safety_pass"] is False
