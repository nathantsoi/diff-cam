"""Independent machine-frame semantics for typed RS274/Haas motion.

Expected coordinates in these tests are hand-derived from TaskSpec transforms,
not from output of the post under test. Post/parse comparisons are secondary.
"""

from dataclasses import replace
import math

import numpy as np
import pytest

from benchmarks import contracts as C
from cam.config import MachineConfig
from cam.gcode_export import motion_plan_to_gcode
from cam.gcode_parser import (
    GCodeError,
    UnsupportedGCodeError,
    parse_motion_plan,
)
from cam.posts import UnsupportedPostError, post_motion_plan
from cam.trajectory_planner import plan_motion_samples


TASK = C.example_pocket_task()
CANONICAL = """G21 G90 G17 G54 G61 G43 H1
G0 X-4 Y-4 Z10
G1 Z-2 F300
G1 X4 F600
G3 X4 Y4 I0 J4
G0 Z10
G53 G0 Z0
M2
"""


def _arc_case(*, radius=3.333, centre=(0.123, 0.765, -2.0), a0=0.37, a1=2.1, plane=C.ArcPlane.XY):
    """Build a geometrically consistent arc without using the post's formatter."""
    transform = TASK.work_to_machine
    centre_mm = transform.apply(centre)
    in_plane = {C.ArcPlane.XY: (0, 1), C.ArcPlane.XZ: (2, 0), C.ArcPlane.YZ: (1, 2)}[plane]
    p0 = list(centre)
    p1 = list(centre)
    p0[in_plane[0]] += radius * math.cos(a0)
    p0[in_plane[1]] += radius * math.sin(a0)
    p1[in_plane[0]] += radius * math.cos(a1)
    p1[in_plane[1]] += radius * math.sin(a1)
    start = TASK.initial_state.position_mm
    lead = C.Segment(id="lead", kind=C.SegmentKind.RAPID,
                     start_mm=start, end_mm=transform.apply(p0),
                     duration_s=math.dist(start, transform.apply(p0)) / 200.0)
    arc = C.ArcData(centre_mm=centre_mm, plane=plane, direction=C.ArcDirection.CCW)
    segment = C.Segment(id="curve", kind=C.SegmentKind.ARC,
                        start_mm=transform.apply(p0), end_mm=transform.apply(p1),
                        duration_s=radius * (a1 - a0) / 10.0,
                        feed_mm_s=10.0, arc=arc)
    return C.MotionPlan(frame=C.Frame.MACHINE, initial_state=TASK.initial_state,
                        segments=(lead, segment), timing_rule=TASK.motion_semantics.timing_rule)


def _assert_arc_roundtrip(plan, units, precision):
    text = post_motion_plan(plan, TASK, config=MachineConfig(units=units, precision=precision))
    reparsed = parse_motion_plan(text, TASK)
    assert len(reparsed.segments) == 2
    expected_arc = plan.segments[1]
    actual_arc = reparsed.segments[1]
    assert actual_arc.arc is not None
    assert actual_arc.arc.plane == expected_arc.arc.plane
    assert actual_arc.arc.direction == expected_arc.arc.direction
    planes = {C.ArcPlane.XY: (0, 1), C.ArcPlane.XZ: (2, 0), C.ArcPlane.YZ: (1, 2)}
    a, b = planes[actual_arc.arc.plane]
    r_start = math.hypot(actual_arc.start_mm[a] - actual_arc.arc.centre_mm[a],
                         actual_arc.start_mm[b] - actual_arc.arc.centre_mm[b])
    r_end = math.hypot(actual_arc.end_mm[a] - actual_arc.arc.centre_mm[a],
                       actual_arc.end_mm[b] - actual_arc.arc.centre_mm[b])
    assert abs(r_start - r_end) <= C.ARC_RADIUS_TOL_MM + 1e-12
    quantum = (25.4 if units == "inch" else 1.0) * 10.0 ** (-precision)
    # Rounded offset repair is bounded to a documented neighbouring grid step.
    component_tolerance = 2 * math.sqrt(3) * quantum + 1e-9
    np.testing.assert_allclose(actual_arc.end_mm, expected_arc.end_mm,
                               atol=component_tolerance, rtol=0)
    np.testing.assert_allclose(actual_arc.arc.centre_mm, expected_arc.arc.centre_mm,
                               atol=component_tolerance, rtol=0)
    return actual_arc


@pytest.mark.parametrize("units", ["mm", "inch"])
@pytest.mark.parametrize("precision", [6, 4, 3])
def test_reviewed_arc_quantization_and_roundtrip(units, precision):
    plan = _arc_case()
    if units == "inch" and precision == 3:
        # The measured nearest-grid error is 0.00610844 mm > the frozen
        # 1e-3mm contract tolerance. It must be rejected, not silently posted.
        with pytest.raises(UnsupportedPostError, match="no nearest-grid arc offset preserves radii"):
            post_motion_plan(plan, TASK, config=MachineConfig(units=units, precision=precision))
        return
    _assert_arc_roundtrip(plan, units, precision)


def test_arc_offset_nearest_grid_property_across_planes():
    accepted = []
    declined = []
    for plane in (C.ArcPlane.XY, C.ArcPlane.XZ, C.ArcPlane.YZ):
        rng = np.random.default_rng(1447)
        cases = []
        for _ in range(3):
            start, end = sorted(rng.uniform(-math.pi, math.pi, 2))
            cases.append(dict(radius=float(rng.uniform(2.0, 7.0)), centre=tuple(rng.uniform(-1.0, 1.0, 3)),
                              a0=float(start), a1=float(end), plane=plane))
        for units in ("mm", "inch"):
            for precision in (6, 4, 3):
                for index, kwargs in enumerate(cases):
                    plan = _arc_case(**kwargs)
                    try:
                        _assert_arc_roundtrip(plan, units, precision)
                    except UnsupportedPostError as error:
                        assert "not emitted" in str(error)
                        declined.append((plane, units, precision, index))
                    else:
                        accepted.append((plane, units, precision, index))
    # Rejection is genuine coverage, not a skip. Seen outcomes are printed for
    # provenance when running this property with -s; exact counts are seed-dependent.
    print("ARC_PROPERTY_ACCEPTED", len(accepted), "ARC_PROPERTY_DECLINED", len(declined))
    assert len(accepted) + len(declined) == 3 * 2 * 3 * 3
    assert accepted  # arbitrary-path arcs must not be indiscriminately rejected


def test_extremely_coarse_arc_is_rejected_without_returning_nc():
    with pytest.raises(UnsupportedPostError, match="partial arc that would become a full turn"):
        post_motion_plan(_arc_case(), TASK, config=MachineConfig(units="inch", precision=0))


def task_at(start_work, *, transform=None):
    transform = transform or TASK.work_to_machine
    initial = replace(
        TASK.initial_state,
        position_mm=transform.apply(start_work),
        frame=C.Frame.MACHINE,
        known=True,
    )
    return replace(TASK, work_to_machine=transform, initial_state=initial)


def assert_plan_equivalent(actual, expected, atol=2e-5):
    assert actual.frame == expected.frame == C.Frame.MACHINE
    assert actual.initial_state == expected.initial_state
    assert actual.timing_rule == expected.timing_rule
    assert len(actual.segments) == len(expected.segments)
    for got, want in zip(actual.segments, expected.segments):
        assert got.kind == want.kind
        assert got.tag == want.tag
        assert np.allclose(got.start_mm, want.start_mm, atol=atol)
        assert np.allclose(got.end_mm, want.end_mm, atol=atol)
        assert got.duration_s == pytest.approx(want.duration_s, abs=atol)
        assert got.feed_mm_s == pytest.approx(want.feed_mm_s, abs=atol) if want.feed_mm_s is not None else got.feed_mm_s is None
        assert (got.arc is None) == (want.arc is None)
        if want.arc is not None:
            assert np.allclose(got.arc.centre_mm, want.arc.centre_mm, atol=atol)
            assert got.arc.plane == want.arc.plane
            assert got.arc.direction == want.arc.direction
            assert got.arc.turns == want.arc.turns


def test_canonical_machine_coordinates_tags_arcs_and_timing():
    plan = parse_motion_plan(CANONICAL, TASK)
    assert len(plan.segments) == 6
    # Independent of any post: G54=(100,80,-150), initial W=(0,0,10).
    expected_end = [
        (96.0, 76.0, -140.0),
        (96.0, 76.0, -152.0),
        (104.0, 76.0, -152.0),
        (104.0, 84.0, -152.0),
        (104.0, 84.0, -140.0),
        (104.0, 84.0, 0.0),  # G53 machine Z0, not work Z0=-150
    ]
    assert [segment.kind for segment in plan.segments] == [
        C.SegmentKind.RAPID,
        C.SegmentKind.FEED,
        C.SegmentKind.FEED,
        C.SegmentKind.ARC,
        C.SegmentKind.RAPID,
        C.SegmentKind.RAPID,
    ]
    assert [segment.tag for segment in plan.segments] == [
        "approach", "plunge", "", "", "retract", "home"
    ]
    for segment, end in zip(plan.segments, expected_end):
        assert segment.end_mm == end
    assert plan.segments[0].start_mm == (100.0, 80.0, -140.0)
    assert plan.segments[0].duration_s == pytest.approx(math.sqrt(32.0) / 200.0)
    assert plan.segments[1].duration_s == pytest.approx(12.0 / 5.0)
    assert plan.segments[1].feed_mm_s == pytest.approx(5.0)
    assert plan.segments[2].duration_s == pytest.approx(8.0 / 10.0)
    assert plan.segments[3].arc.centre_mm == (104.0, 80.0, -152.0)
    assert plan.segments[3].arc.plane == C.ArcPlane.XY
    assert plan.segments[3].arc.direction == C.ArcDirection.CCW
    assert plan.segments[3].duration_s == pytest.approx(4.0 * math.pi / 10.0)
    assert plan.segments[4].duration_s == pytest.approx(12.0 / 200.0)
    assert plan.segments[5].duration_s == pytest.approx(140.0 / 200.0)
    assert [segment.source_line for segment in plan.segments] == [2, 3, 4, 5, 6, 7]
    assert [segment.id for segment in plan.segments] == [
        f"nc-{line:06d}" for line in (2, 3, 4, 5, 6, 7)
    ]
    plan.validate_for_task(TASK)


def test_canonical_contract_json_roundtrip():
    plan = parse_motion_plan(CANONICAL, TASK)
    assert C.loads(C.dumps(plan)) == plan


@pytest.mark.parametrize("dialect", ["rs274", "haas"])
@pytest.mark.parametrize("units", ["mm", "inch"])
def test_typed_post_roundtrip_preserves_motion(dialect, units):
    original = parse_motion_plan(CANONICAL, TASK, dialect=dialect)
    text = post_motion_plan(
        original,
        TASK,
        dialect=dialect,
        config=MachineConfig(units=units),
    )
    assert "G3 " in text
    assert "G10" not in text
    assert "G20" in text if units == "inch" else "G21" in text
    reparsed = parse_motion_plan(text, TASK, dialect=dialect)
    assert_plan_equivalent(reparsed, original)
    assert reparsed.provenance.startswith(f"parsed:{dialect}:sha256:")


def test_bare_g43_has_a_specific_missing_register_error():
    with pytest.raises(GCodeError, match=r"G43 requires an explicit H register"):
        parse_motion_plan("G21 G90 G43\nG0 X0 Y0 Z10\nM2\n", TASK)


@pytest.mark.parametrize("statement,initial_offset", [("G43 H1", True), ("G49", False)])
def test_tool_length_mode_must_agree_with_initial_state(statement, initial_offset):
    state = replace(TASK.initial_state, length_offset_applied=initial_offset)
    matching_task = replace(TASK, initial_state=state)
    program = f"G21 G90 {statement}\nG1 X1 F600\nM2\n"
    plan = parse_motion_plan(program, matching_task)
    assert plan.initial_state.length_offset_applied is initial_offset
    mismatch_task = replace(TASK, initial_state=replace(state, length_offset_applied=not initial_offset))
    with pytest.raises(UnsupportedGCodeError, match="intermediate register changes"):
        parse_motion_plan(program, mismatch_task)


def test_offset_change_in_motion_is_rejected_even_if_cancelled_in_same_block():
    with pytest.raises(GCodeError, match="conflicting modal modes"):
        parse_motion_plan("G21 G90 G43 H1 G49 G1 X1 F600\nM2\n", TASK)


def test_h_register_must_match_task_tool_and_first_register():
    with pytest.raises(UnsupportedGCodeError, match=r"H2 is incompatible.*tool_number=1"):
        parse_motion_plan("G21 G90 G54 T1 M6 G43 H2 G0 X0 Y0 Z10\nM2\n", TASK)
    with pytest.raises(UnsupportedGCodeError, match=r"H2 is incompatible.*tool_number=1"):
        parse_motion_plan("G21 G90 G43 H1\nG0 X1\nG43 H2\nG1 X2 F600\nM2\n", TASK)


def test_selected_tool_must_match_task_tool_number():
    with pytest.raises(UnsupportedGCodeError, match=r"T2 is incompatible"):
        parse_motion_plan("G21 G90 T2 M6 G43 H1 G0 X1\nM2\n", TASK)


def test_matching_h_register_and_t_value_are_accepted():
    program = "G21 G90 G54 T1 M6 G43 H1\nG0 X1\nM2\n"
    assert parse_motion_plan(program, TASK).initial_state.known is True
    task_two = replace(TASK, tool=replace(TASK.tool, tool_number=2))
    plan = parse_motion_plan("G21 G90 G54 T2 M6 G43 H2\nG0 X1\nM2\n", task_two)
    assert plan.initial_state.known and len(plan.segments) == 1
    text = post_motion_plan(C.example_motion_plan(task_two), task_two,
                            dialect="haas", config=MachineConfig(tool_number=2))
    assert "T2 M06" in text and "G43 H2" in text


def test_post_rejects_incompatible_h_and_tool_configuration():
    plan = C.example_motion_plan(TASK)
    with pytest.raises(UnsupportedPostError, match=r"H2 does not match TaskSpec.tool.tool_number=1"):
        post_motion_plan(plan, TASK, config=MachineConfig(length_offset=2, tool_number=1))
    with pytest.raises(UnsupportedPostError, match=r"T2 does not match TaskSpec.tool.tool_number=1"):
        post_motion_plan(plan, TASK, dialect="haas", config=MachineConfig(tool_number=2, length_offset=1))


def test_wrapper_matches_direct_typed_post():
    plan = parse_motion_plan(CANONICAL, TASK)
    assert motion_plan_to_gcode(plan, TASK) == post_motion_plan(plan, TASK)


def test_incremental_inches_are_hand_computed():
    # Initial W=(0,0,10). One-inch XY moves and a -0.5 inch Z move reach
    # W=(25.4,-25.4,-2.7), then -0.1 inch reaches Wz=-5.24.
    task = task_at((0.0, 0.0, 10.0))
    text = """G20 G91 G54
G0 X1 Y-1 Z-0.5
G1 Z-0.1 F2
M2
"""
    plan = parse_motion_plan(text, task)
    assert plan.segments[0].end_mm == pytest.approx((125.4, 54.6, -152.7))
    assert plan.segments[1].end_mm == pytest.approx((125.4, 54.6, -155.24))
    assert plan.segments[1].feed_mm_s == pytest.approx(2.0 * 25.4 / 60.0)
    assert plan.segments[1].duration_s == pytest.approx(3.0)


def test_rotated_work_offset_changes_linear_machine_target_not_length():
    # +90 degrees about Z: W(x,y,z) -> M(-y+100,x+80,z-150).
    transform = C.Transform(
        (
            (0.0, -1.0, 0.0, 100.0),
            (1.0, 0.0, 0.0, 80.0),
            (0.0, 0.0, 1.0, -150.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    task = task_at((0.0, 0.0, 10.0), transform=transform)
    plan = parse_motion_plan("G21 G90 G54\nG0 X4 Y5\nM2\n", task)
    segment = plan.segments[0]
    assert segment.start_mm == (100.0, 80.0, -140.0)
    assert segment.end_mm == (95.0, 84.0, -140.0)
    assert segment.duration_s == pytest.approx(math.sqrt(41.0) / 200.0)


def test_arc_direction_flips_when_rotation_reverses_plane_normal():
    transform = C.Transform(
        (
            (1.0, 0.0, 0.0, 100.0),
            (0.0, -1.0, 0.0, 80.0),
            (0.0, 0.0, -1.0, -150.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    task = task_at((10.0, 0.0, 0.0), transform=transform)
    plan = parse_motion_plan(
        "G21 G90 G17 G54 F60\nG3 X0 Y10 I-10\nM2\n", task
    )
    arc = plan.segments[0].arc
    assert arc.plane == C.ArcPlane.XY
    assert arc.direction == C.ArcDirection.CW
    assert arc.centre_mm == pytest.approx((100.0, 80.0, -150.0))
    assert plan.segments[0].end_mm == pytest.approx((100.0, 70.0, -150.0))
    assert plan.segments[0].duration_s == pytest.approx(5.0 * math.pi)


@pytest.mark.parametrize(
    "plane,start,move,expected_plane,center_work,direction,expected_duration",
    [
        ("G18", (10.0, 0.0, 0.0), "G3 X0 Z10 K10", C.ArcPlane.XZ,
         (10.0, 0.0, 10.0), C.ArcDirection.CCW, 5.0 * math.pi),
        # In the right-handed (Y,Z,X) basis, +Y to +Z is CCW. G2 therefore
        # takes the 270-degree clockwise sweep, not the other 90-degree arc.
        ("G19", (0.0, 10.0, 0.0), "G2 Y0 Z10 J-10", C.ArcPlane.YZ,
         (0.0, 0.0, 0.0), C.ArcDirection.CW, 15.0 * math.pi),
    ],
)
def test_g18_g19_keep_typed_geometry(plane, start, move, expected_plane, center_work, direction, expected_duration):
    task = task_at(start)
    plan = parse_motion_plan(
        f"G21 G90 {plane} G54 F60\n{move}\nM2\n", task
    )
    segment = plan.segments[0]
    assert segment.arc.plane == expected_plane
    assert segment.arc.direction == direction
    assert segment.arc.centre_mm == pytest.approx(TASK.work_to_machine.apply(center_work))
    assert segment.duration_s == pytest.approx(expected_duration)
    reposted = post_motion_plan(plan, task)
    assert_plan_equivalent(parse_motion_plan(reposted, task), plan)


def test_positive_r_selects_independently_computed_minor_arc():
    task = task_at((10.0, 0.0, 0.0))
    plan = parse_motion_plan(
        "G21 G90 G17 G54 F60\nG2 X0 Y10 R10\nM2\n", task
    )
    segment = plan.segments[0]
    # The two centres are (0,0) and (10,10). Positive R selects the 90-degree
    # clockwise candidate about (10,10), not the 270-degree candidate.
    assert segment.arc.centre_mm == pytest.approx((110.0, 90.0, -150.0))
    assert segment.duration_s == pytest.approx(5.0 * math.pi)


def test_full_circle_p_word_retains_extra_turn_without_chords():
    task = task_at((10.0, 0.0, 0.0))
    plan = parse_motion_plan(
        "G21 G90 G17 G54 F60\nG2 X10 Y0 I-10 P2\nM2\n", task
    )
    segment = plan.segments[0]
    assert segment.start_mm == segment.end_mm
    assert segment.arc.turns == 1
    assert segment.duration_s == pytest.approx(40.0 * math.pi)
    reposted = post_motion_plan(plan, task, dialect="haas")
    assert "P2" in reposted and "G2 " in reposted
    assert_plan_equivalent(parse_motion_plan(reposted, task, dialect="haas"), plan)


def test_zero_length_feed_is_preserved_not_dropped():
    task = task_at((0.0, 0.0, 10.0))
    plan = parse_motion_plan(
        "G21 G90 G54 G1 F60\nX0 Y0 Z10\nM2\n", task
    )
    assert len(plan.segments) == 1
    assert plan.segments[0].kind == C.SegmentKind.FEED
    assert plan.segments[0].duration_s == 0.0
    points, times = plan_motion_samples(plan, task)
    assert points.shape == (1, 3)
    assert times.shape == (1,)
    assert times[0] == 0.0
    assert points[0] == pytest.approx(task.initial_state.position_mm)


def test_unknown_initial_position_rejects_first_approach():
    task = replace(
        TASK,
        initial_state=C.MachineState(position_mm=None, known=False),
    )
    with pytest.raises(UnsupportedGCodeError, match="initial machine position is unknown"):
        parse_motion_plan("G21 G90 G54\nG0 X0 Y0 Z10\n", task)
    # An empty non-motion header can still be represented without inventing a pose.
    empty = parse_motion_plan("G21 G90 G54\nM2\n", task)
    assert empty.segments == ()
    assert not empty.initial_state.known


@pytest.mark.parametrize(
    "text,match",
    [
        ("G21 G90 G54\nG0 X1 Q2\n", "Q2"),
        ("G21 G90 G54\nG64\n", "timing/blending"),
        ("G21 G90 G54\nG28 Z0\n", "G28"),
        ("G21 G90 G54\nG30 Z0\n", "G30"),
        ("G21 G90 G54\nG10 L2 P1 X0\n", "G10"),
        ("G21 G90 G54\nG41 X1\n", "G41"),
        ("G21 G90 G54\nG4 P1\n", "G4"),
        ("G21 G90 G54\nG0 A4\n", "A4"),
        ("G21 G90 G54\nM42\n", "unsupported M"),
    ],
)
def test_unknown_or_unsupported_words_fail_explicitly(text, match):
    with pytest.raises(UnsupportedGCodeError, match=match):
        parse_motion_plan(text, TASK)


def test_parameter_syntax_is_a_syntax_error_not_ignored():
    with pytest.raises(GCodeError, match="syntax"):
        parse_motion_plan("G21 G90 G54\nG0 X1 #2\n", TASK)


@pytest.mark.parametrize(
    "text,match",
    [
        ("G0 X0\n", "G20/G21 and G90/G91"),
        ("G21 G90 X1\n", "motion mode is unknown"),
        ("G21 G90 G1 X1\n", "requires an explicit positive F"),
        ("F5\n", "feed interpretation"),
        ("G21 G90 G1 X1 F-3\n", "F must be positive"),
        ("G21 G90 G1 F4000 X1\n", "exceeds limits.max_feed"),
    ],
)
def test_modal_and_feed_requirements_are_not_guessed(text, match):
    with pytest.raises(GCodeError, match=match):
        parse_motion_plan(text, TASK)


def test_g53_does_not_accept_incremental_or_arc_motion():
    with pytest.raises(UnsupportedGCodeError, match="G53 supports only absolute"):
        parse_motion_plan("G21 G91 G53 G0 Z0\n", TASK)
    with pytest.raises(UnsupportedGCodeError, match="G53 supports only absolute"):
        parse_motion_plan("G21 G90 G53 G2 F60 X0 I1\n", TASK)


def test_tilted_arc_plane_is_explicitly_unsupported():
    # Rotate work Z into the machine XZ diagonal: ArcPlane cannot encode it.
    c = s = math.sqrt(0.5)
    transform = C.Transform(
        (
            (c, 0.0, s, 100.0),
            (0.0, 1.0, 0.0, 80.0),
            (-s, 0.0, c, -150.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    task = task_at((10.0, 0.0, 0.0), transform=transform)
    with pytest.raises(UnsupportedGCodeError, match="outside every principal"):
        parse_motion_plan("G21 G90 G17 F60\nG3 X0 Y10 I-10\n", task)


def test_tool_offset_declarations_and_g43_h_register_validate():
    task = task_at((0.0, 0.0, 10.0))
    plan = parse_motion_plan(
        "G21 G90 G54 G49\nG43 H1\nG0 X0 Y0 Z10\nM2\n", task
    )
    assert len(plan.segments) == 1
    with pytest.raises(GCodeError, match="H is only accepted"):
        parse_motion_plan("G21 G90 G54 H2\n", task)
    with pytest.raises(GCodeError, match="H requires an integer"):
        parse_motion_plan("G21 G90 G54 G43 H1.5\n", task)


def test_exact_stop_arc_sampling_has_independent_midpoint_and_duration():
    task = task_at((10.0, 0.0, 0.0))
    plan = parse_motion_plan(
        "G21 G90 G17 G54 F60\nG3 X0 Y10 I-10\nM2\n", task
    )
    duration = plan.segments[0].duration_s
    points, times = plan_motion_samples(plan, task, max_dt_s=duration / 2.0)
    assert points.shape == (3, 3)
    assert times[-1] == pytest.approx(duration)
    # CCW quarter circle from (10,0) to (0,10) about work (0,0):
    # midpoint is (sqrt(50),sqrt(50)), never a chord midpoint (5,5).
    midpoint = (100.0 + math.sqrt(50.0), 80.0 + math.sqrt(50.0), -150.0)
    assert points[1] == pytest.approx(midpoint)
    assert points[-1] == pytest.approx((100.0, 90.0, -150.0))


def test_fixed_dt_plan_is_explicitly_unpostable():
    start = TASK.initial_state.position_mm
    end = (start[0] + 1.0, start[1], start[2])
    segment = C.Segment(
        id="fixed", kind=C.SegmentKind.FEED, start_mm=start, end_mm=end,
        duration_s=1.0, feed_mm_s=1.0,
    )
    plan = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=TASK.initial_state,
        segments=(segment,),
        timing_rule=C.TimingRule.FIXED_DT,
        dt_s=0.1,
    )
    with pytest.raises(UnsupportedPostError, match="not fixed_dt"):
        post_motion_plan(plan, TASK)


def test_arc_disabled_task_and_disabled_dialect_are_unsupported():
    task = replace(TASK, motion_semantics=C.MotionSemantics(arcs_supported=False))
    plan = parse_motion_plan(CANONICAL, TASK)
    with pytest.raises(UnsupportedPostError, match="disables arc output"):
        post_motion_plan(plan, task)
    rs_only = replace(TASK, motion_semantics=C.MotionSemantics(dialects=(C.Dialect.RS274,)))
    with pytest.raises(UnsupportedGCodeError, match="disabled by this TaskSpec"):
        parse_motion_plan("G21 G90\nM2\n", rs_only, dialect="haas")


def test_repeated_tool_number_is_allowed_but_tool_change_is_not():
    header = "G21 G90 G54 G49\n"
    parse_motion_plan(header + "T1 M06\nT1 M06\nM2\n", TASK)
    with pytest.raises(UnsupportedGCodeError, match="T2 is incompatible"):
        parse_motion_plan(header + "T1 M06\nT2 M06\n", TASK)
