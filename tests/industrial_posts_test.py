# SPDX-License-Identifier: MIT
"""End-to-end integration tests for the 6-controller clean-room post-processors & parsers."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from benchmarks import contracts as C
from benchmarks.milling_suite import build_milling_suite
from cam import (
    POSTS,
    ControllerDialect,
    MachineConfig,
    get_post,
    parse_gcode,
    parse_motion_plan,
    parse_shop_motion_plan,
    post_motion_plan,
    post_shop_program,
    trajectory_to_gcode,
)
from cam.dialects import DIALECT_SPECS, verify_hardware_syntax
from cam.gcode_parser import GCodeError, UnsupportedGCodeError
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import UnsupportedPostError
from verification.camotics_adapter import _prepare_camotics_nc

ALL_DIALECTS = ("rs274", "haas", "fanuc", "robodrill", "makino", "mazak")


def _make_arc_and_linear_plan(timing_rule: C.TimingRule) -> tuple[C.TaskSpec, C.MotionPlan]:
    base = C.example_pocket_task()
    semantics = dataclasses.replace(base.motion_semantics, timing_rule=timing_rule)
    task = dataclasses.replace(base, motion_semantics=semantics)
    to_m = task.work_to_machine.apply

    p_init = task.initial_state.position_mm
    p0 = to_m((-4.0, -4.0, 6.0))
    p1 = to_m((-4.0, -4.0, -2.0))
    p2 = to_m((2.0, -4.0, -2.0))
    p3 = to_m((4.0, -2.0, -2.0))
    centre = to_m((2.0, -2.0, -2.0))
    p4 = to_m((4.0, -2.0, 6.0))

    raw = [
        C.Segment(
            id="s0", kind=C.SegmentKind.RAPID, start_mm=p_init, end_mm=p0,
            duration_s=math.dist(p_init, p0) / task.limits.rapid_mm_s, tag="approach",
        ),
        C.Segment(
            id="s1", kind=C.SegmentKind.FEED, start_mm=p0, end_mm=p1,
            duration_s=math.dist(p0, p1) / 4.0, feed_mm_s=4.0, tag="plunge",
        ),
        C.Segment(
            id="s2", kind=C.SegmentKind.FEED, start_mm=p1, end_mm=p2,
            duration_s=math.dist(p1, p2) / 8.0, feed_mm_s=8.0,
        ),
        C.Segment(
            id="s3", kind=C.SegmentKind.ARC, start_mm=p2, end_mm=p3,
            duration_s=(0.5 * math.pi * 2.0) / 8.0, feed_mm_s=8.0,
            arc=C.ArcData(centre_mm=centre, plane=C.ArcPlane.XY, direction=C.ArcDirection.CCW),
        ),
        C.Segment(
            id="s4", kind=C.SegmentKind.RAPID, start_mm=p3, end_mm=p4,
            duration_s=math.dist(p3, p4) / task.limits.rapid_mm_s, tag="retract",
        ),
    ]
    if timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        durs = C.blended_segment_durations(raw, task.limits.rapid_mm_s, task.limits.max_accel_mm_s2)
        segs = tuple(dataclasses.replace(s, duration_s=d) for s, d in zip(raw, durs))
    else:
        segs = tuple(raw)
    plan = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=task.initial_state,
        segments=segs,
        timing_rule=timing_rule,
    )
    plan.validate_for_task(task)
    return task, plan


def test_posts_registry_contains_all_six_controllers():
    assert set(POSTS) == set(ALL_DIALECTS)
    pts = np.array([[0.2, 0.2, 0.8], [0.8, 0.2, 0.8], [0.8, 0.8, 0.6]], dtype=np.float64)
    cfg = MachineConfig(units="mm", stock_size_mm=50.0, program_number=101)
    for name in ALL_DIALECTS:
        post = get_post(name)
        assert post.name == name
        nc = trajectory_to_gcode(pts, config=cfg, post=name)
        verify_hardware_syntax(nc, name, units="mm", shop_mode=False)
        segs = parse_gcode(nc, config=cfg)
        assert len(segs) >= 2


@pytest.mark.parametrize("dialect", ALL_DIALECTS)
@pytest.mark.parametrize(
    "timing_rule",
    [C.TimingRule.EXACT_STOP_CONSTANT_SPEED, C.TimingRule.BLENDED_ACCEL_BOUNDED],
)
@pytest.mark.parametrize("units", ["mm", "inch"])
def test_typed_post_and_parse_roundtrip_all_six_dialects(dialect, timing_rule, units):
    task, plan = _make_arc_and_linear_plan(timing_rule)
    cfg = MachineConfig(
        units=units,
        precision=4,
        tool_number=task.tool.tool_number,
        length_offset=task.tool.tool_number,
        program_number=123,
    )
    nc_text = post_motion_plan(plan, task, dialect=dialect, config=cfg)
    verify_hardware_syntax(nc_text, dialect, units=units, shop_mode=False)

    if dialect == "haas":
        assert "G64" not in nc_text
        if timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
            assert "G187 P2 E" in nc_text
            assert "\nG187\n" in nc_text
    elif dialect in ("fanuc", "robodrill") and timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        assert "G05.1 Q1 R5" in nc_text
        assert "G05.1 Q0" in nc_text
    elif dialect == "makino" and timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        assert "\nM252\n" in nc_text
        assert "\nM250\n" in nc_text
    elif dialect == "mazak":
        assert not nc_text.lstrip().startswith("%")
        assert nc_text.rstrip().endswith("%")
        if timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
            assert "G61.1 P2" in nc_text
            assert "\nG64\n" in nc_text

    reparsed = parse_motion_plan(nc_text, task, dialect=dialect)
    assert reparsed.timing_rule == timing_rule
    assert reparsed.provenance.startswith(f"parsed:{dialect}:")
    assert len(reparsed.segments) == len(plan.segments)
    tol_mm = (25.4 if units == "inch" else 1.0) * 1e-3 + 1e-6
    for orig, got in zip(plan.segments, reparsed.segments):
        assert got.kind == orig.kind
        np.testing.assert_allclose(got.end_mm, orig.end_mm, atol=tol_mm, rtol=0.0)
        assert got.duration_s == pytest.approx(orig.duration_s, rel=2e-3, abs=2e-3)

    # Also verify CAMotics adapter prep neutralizes vendor smoothing cleanly
    from cam.dialects.base import strip_nc_comments

    sim_bytes, cam_plan = _prepare_camotics_nc(task, nc_text.encode("utf-8"), dialect)
    assert len(cam_plan.segments) == len(plan.segments)
    exec_lines = [strip_nc_comments(ln) for ln in sim_bytes.decode("utf-8").splitlines()]
    assert not any(ln.startswith("G187") for ln in exec_lines)
    assert not any(ln.startswith("G05.1") or ln.startswith("G5.1") for ln in exec_lines)
    assert not any(ln in ("M250", "M251", "M252", "M253") for ln in exec_lines)


@pytest.mark.parametrize("dialect", ALL_DIALECTS)
def test_posted_evaluator_and_shop_program_across_all_six_dialects(dialect):
    bundle = build_milling_suite(voxel_mm=0.5, families=("facing_open_boss",), split="test")[0]
    task = bundle.task
    plan = bundle.nominal_motion
    cfg = MachineConfig(
        units="mm",
        tool_number=task.tool.tool_number,
        length_offset=task.tool.tool_number,
        program_number=1001,
        spindle_rpm=8000,
    )

    # 1. Typed post -> PostedEvaluator
    nc_typed = post_motion_plan(plan, task, dialect=dialect, config=cfg)
    ev = evaluate_posted_gcode(
        task,
        nc_typed.encode("utf-8"),
        dialect=dialect,
        settings={"max_step_mm": 0.25, "compute_bounds": False, "safety_screen": True},
    )
    assert ev.geometry_pass is True
    assert ev.safety_pass is True
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert ev.discretization["controller_dialect"] == dialect

    # 2. Standalone physical shop packet -> hardware syntax + parse_shop_motion_plan
    nc_shop = post_shop_program(
        plan, task, dialect=dialect, config=cfg, case_label=bundle.case.case_id
    )
    spec = DIALECT_SPECS[ControllerDialect(dialect)]
    spec.verify_hardware_syntax(nc_shop, units="mm", shop_mode=True)
    if dialect in ("fanuc", "makino"):
        assert "G28 G91 Z0.0" in nc_shop
    else:
        assert "G90 G53 G0 Z0.0" in nc_shop

    lo, hi = task.limits.envelope_min_mm, task.limits.envelope_max_mm
    shop_task = dataclasses.replace(
        task,
        work_to_machine=C.Transform.translation((0.0, 0.0, -50.0)),
        initial_state=C.MachineState(
            position_mm=(0.0, 0.0, 0.0),
            frame=C.Frame.MACHINE,
            length_offset_applied=task.initial_state.length_offset_applied,
        ),
        limits=dataclasses.replace(
            task.limits,
            envelope_min_mm=(lo[0], lo[1], lo[2] - 50.0),
            envelope_max_mm=(hi[0], hi[1], max(0.0, hi[2] - 50.0) + 10.0),
        ),
    )
    shop_plan = parse_shop_motion_plan(nc_shop, shop_task, dialect=dialect, g28_home_z_mm=0.0)
    assert len(shop_plan.segments) >= len(plan.segments) + 4


def test_cross_dialect_smoothing_rejection_and_frozen_task_hash_invariance():
    task, plan = _make_arc_and_linear_plan(C.TimingRule.BLENDED_ACCEL_BOUNDED)
    cfg = MachineConfig(tool_number=task.tool.tool_number, length_offset=task.tool.tool_number, program_number=12)

    rs_nc = post_motion_plan(plan, task, dialect="rs274", config=cfg)
    haas_nc = post_motion_plan(plan, task, dialect="haas", config=cfg)
    fanuc_nc = post_motion_plan(plan, task, dialect="fanuc", config=cfg)
    makino_nc = post_motion_plan(plan, task, dialect="makino", config=cfg)
    mazak_nc = post_motion_plan(plan, task, dialect="mazak", config=cfg)

    # Haas must reject G64 (RS274 / Mazak), G05.1 (Fanuc), M252 (Makino)
    with pytest.raises(GCodeError, match="G64|G91.1|unsupported"):
        parse_motion_plan(rs_nc, task, dialect="haas")
    with pytest.raises(GCodeError, match="G64 is unsupported on Haas NGC"):
        parse_motion_plan("G21 G90 G54 G64\nM30\n", task, dialect="haas")
    with pytest.raises(GCodeError, match="G5.1|Q|unsupported"):
        parse_motion_plan(fanuc_nc, task, dialect="haas")
    with pytest.raises(GCodeError, match="252|unsupported"):
        parse_motion_plan(makino_nc, task, dialect="haas")

    # Fanuc / Robodrill must reject G187 (Haas) and G64 (RS274)
    with pytest.raises(GCodeError, match="187|E|unsupported"):
        parse_motion_plan(haas_nc, task, dialect="fanuc")
    with pytest.raises(GCodeError, match="G64|G91.1|unsupported"):
        parse_motion_plan(rs_nc, task, dialect="robodrill")

    # Makino must reject G05.1 and G187; Mazak must reject G187 and M252
    with pytest.raises(GCodeError, match="5.1|Q|unsupported"):
        parse_motion_plan(fanuc_nc, task, dialect="makino")
    with pytest.raises(GCodeError, match="252|unsupported"):
        parse_motion_plan(makino_nc, task, dialect="mazak")
    with pytest.raises(GCodeError, match="61.1|64|unsupported"):
        parse_motion_plan(mazak_nc, task, dialect="fanuc")

    # Tasks restricted to RS274 reject all 5 industrial EIA/ISO dialects
    rs_only = dataclasses.replace(
        task, motion_semantics=dataclasses.replace(task.motion_semantics, dialects=(C.Dialect.RS274,))
    )
    for iso_dialect in ("haas", "fanuc", "robodrill", "makino", "mazak"):
        with pytest.raises(UnsupportedPostError, match="disabled by this TaskSpec"):
            post_motion_plan(plan, rs_only, dialect=iso_dialect, config=cfg)
        with pytest.raises(UnsupportedGCodeError, match="disabled by this TaskSpec"):
            parse_motion_plan("G21 G90\nM30\n", rs_only, dialect=iso_dialect)

    # Frozen C.Dialect stays exact (RS274, HAAS) so campaign schemas remain invariant
    assert tuple(C.Dialect) == (C.Dialect.RS274, C.Dialect.HAAS)
