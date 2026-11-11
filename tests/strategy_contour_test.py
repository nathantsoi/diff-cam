# SPDX-License-Identifier: MIT
"""Qualification test suite for 2.5D Multi-Boss Exterior & Island Contour Profiling."""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Tuple

import pytest

from benchmarks import contracts as C
from benchmarks.milling_suite import (
    SCHEMA_MULTI,
    build_milling_suite,
    derive_multi_pocket_target,
)
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from cam.strategies.contour import ContourConfig, generate_contour_toolpath

EVAL_SETTINGS = {
    "max_step_mm": 0.25,
    "compute_bounds": False,
    "safety_screen": True,
}


def _make_custom_contour_task(
    task_id: str,
    stock_min_mm: Tuple[float, float, float],
    stock_max_mm: Tuple[float, float, float],
    pocket_z_min: float,
    islands_mm: Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], ...],
    *,
    voxel_mm: float = 0.25,
    timing_rule: C.TimingRule = C.TimingRule.BLENDED_ACCEL_BOUNDED,
    envelope_margin_mm: float = 10.0,
) -> C.TaskSpec:
    shape = tuple(
        int(round((hi - lo) / voxel_mm))
        for lo, hi in zip(stock_min_mm, stock_max_mm)
    )
    grid = C.GridSpec(
        shape=shape,
        origin_mm=stock_min_mm,
        voxel_mm=voxel_mm,
        frame=C.Frame.WORK,
    )
    stock = C.GeometrySource(
        kind="analytic",
        params={
            "primitive": "box",
            "min_mm": list(stock_min_mm),
            "max_mm": list(stock_max_mm),
        },
        frame=C.Frame.WORK,
    )
    pocket_min_mm = (stock_min_mm[0], stock_min_mm[1], pocket_z_min)
    pocket_max_mm = (stock_max_mm[0], stock_max_mm[1], stock_max_mm[2])
    original_part = C.GeometrySource(
        kind="analytic",
        params={
            "primitive": SCHEMA_MULTI,
            "box_min_mm": list(stock_min_mm),
            "box_max_mm": list(stock_max_mm),
            "pocket_min_mm": list(pocket_min_mm),
            "pocket_max_mm": list(pocket_max_mm),
            "islands_mm": [[list(ilo), list(ihi)] for ilo, ihi in islands_mm],
        },
        frame=C.Frame.WORK,
    )
    tool = C.ToolAssembly(
        cutter_kind=C.CutterKind.FLAT,
        cutter_radius_mm=2.5,
        flute_length_mm=12.0,
        shank_radius_mm=2.5,
        holder_bottom_mm=14.0,
        holder_radius_mm=8.0,
        holder_height_mm=30.0,
        tool_number=1,
    )
    op_geom = derive_multi_pocket_target(stock, original_part, tool, grid)
    excl_mm3 = float(op_geom.params["grid_excluded_volume_mm3"])
    limits = C.MachineLimits(
        max_feed_mm_s=20.0,
        rapid_mm_s=100.0,
        max_accel_mm_s2=150.0,
        envelope_min_mm=(
            stock_min_mm[0] - envelope_margin_mm,
            stock_min_mm[1] - envelope_margin_mm,
            stock_min_mm[2] - 4.0,
        ),
        envelope_max_mm=(
            stock_max_mm[0] + envelope_margin_mm,
            stock_max_mm[1] + envelope_margin_mm,
            60.0,
        ),
    )
    tolerances = C.Tolerances(
        completion_residual_mm3=2.0,
        gouge_volume_mm3=0.0,
        local_surface_error_mm=max(0.5, 2.0 * voxel_mm),
        clearance_mm=0.25,
        rapid_penetration_mm=0.0,
    )
    return C.TaskSpec(
        task_id=task_id,
        family="contour_profile",
        split=C.Split.DEV,
        source=C.SourceInfo(origin=f"contour_test:{task_id}", license="MIT"),
        length_unit="mm",
        stock=stock,
        original_part=original_part,
        operation_target=C.OperationTarget(
            geometry=op_geom,
            excluded_volume_mm3=excl_mm3,
            exclusion_reason="walled corner exclusion" if excl_mm3 > 0 else "",
        ),
        grid=grid,
        work_to_machine=C.Transform.translation((0.0, 0.0, 0.0)),
        initial_state=C.MachineState(position_mm=(0.0, 0.0, 6.0), frame=C.Frame.MACHINE),
        tool=tool,
        limits=limits,
        tolerances=tolerances,
        motion_semantics=C.MotionSemantics(timing_rule=timing_rule),
    )


def _assert_rapids_strictly_at_z_safe(task: C.TaskSpec, plan: C.MotionPlan) -> None:
    inv = task.work_to_machine.inverse().apply
    z_safe = inv(task.initial_state.position_mm)[2]
    rapids = [s for s in plan.segments if s.kind == C.SegmentKind.RAPID]
    assert len(rapids) >= 1
    for seg in rapids:
        s_w = inv(seg.start_mm)
        e_w = inv(seg.end_mm)
        assert abs(s_w[2] - z_safe) <= 1e-9, f"{seg.id} rapid start z={s_w[2]} != z_safe={z_safe}"
        assert abs(e_w[2] - z_safe) <= 1e-9, f"{seg.id} rapid end z={e_w[2]} != z_safe={z_safe}"


@pytest.mark.parametrize(
    "case_id",
    ("contour-dev-00", "contour-dev-01", "contour-test-00", "contour-test-01"),
)
def test_milling_suite_contour_tasks_haas_qualification(case_id: str) -> None:
    bundles = {
        b.case.case_id: b
        for b in build_milling_suite(families=("contour_profile",))
    }
    assert set(bundles) == {
        "contour-dev-00",
        "contour-dev-01",
        "contour-test-00",
        "contour-test-01",
    }
    task = bundles[case_id].task

    plan = generate_contour_toolpath(task, ContourConfig())
    assert plan.provenance == "mit_cleanroom:contour_strategy@2"
    plan.validate_for_task(task)
    _assert_rapids_strictly_at_z_safe(task, plan)

    nc_text = post_motion_plan(plan, task, dialect=C.Dialect.HAAS)
    ev = evaluate_posted_gcode(
        task,
        nc_text.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=EVAL_SETTINGS,
    )
    assert ev.coverage == C.Coverage.COMPLETE
    assert ev.geometry_pass is True, (
        f"{case_id} failed geometry_pass: "
        f"residual={ev.metrics['residual_volume_mm3'].value}, "
        f"gouge={ev.metrics['gouge_volume_mm3'].value}, "
        f"surf={ev.metrics['max_surface_error_mm'].value}"
    )
    assert ev.safety_pass is True, f"{case_id} failed safety_pass: {ev.coverage_reasons}"
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert ev.metrics["gouge_volume_mm3"].value == 0.0


def test_synthetic_two_boss_exterior_contour_haas_qualification() -> None:
    """(a) Synthetic 2-boss exterior contour TaskSpec with shared shoulder and central channel."""
    task = _make_custom_contour_task(
        task_id="contour-synth-two-boss",
        stock_min_mm=(-15.0, -10.0, -6.0),
        stock_max_mm=(15.0, 10.0, 0.0),
        pocket_z_min=-2.5,
        islands_mm=(
            ((-10.5, -4.0, -2.5), (-3.5, 3.5, 0.0)),
            ((3.0, -3.5, -2.5), (10.5, 4.0, 0.0)),
        ),
    )
    plan = generate_contour_toolpath(task, ContourConfig())
    assert plan.provenance == "mit_cleanroom:contour_strategy@2"
    plan.validate_for_task(task)
    _assert_rapids_strictly_at_z_safe(task, plan)

    # Verify both bosses receive dedicated finishing passes with tangential lead-in/lead-out
    tags = [s.tag for s in plan.segments]
    assert any("finish_b0_lead_in" in t for t in tags)
    assert any("finish_b1_lead_in" in t for t in tags)
    assert any("finish_b0_lead_out" in t for t in tags)
    assert any("finish_b1_lead_out" in t for t in tags)

    nc_text = post_motion_plan(plan, task, dialect=C.Dialect.HAAS)
    ev = evaluate_posted_gcode(
        task,
        nc_text.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=EVAL_SETTINGS,
    )
    assert ev.geometry_pass is True
    assert ev.safety_pass is True
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0


def test_climb_vs_conventional_direction_and_timing_rules() -> None:
    """(b) Verify CW ('climb') vs CCW ('conventional') orientation and both timing rules."""
    base_bundle = next(
        b for b in build_milling_suite(families=("contour_profile",))
        if b.case.case_id == "contour-dev-00"
    )
    task_blended = base_bundle.task
    task_exact = dataclasses.replace(
        task_blended,
        motion_semantics=C.MotionSemantics(timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED),
    )

    plan_climb = generate_contour_toolpath(
        task_blended, ContourConfig(milling_direction="climb")
    )
    plan_conv = generate_contour_toolpath(
        task_exact, ContourConfig(milling_direction="conventional")
    )
    plan_climb.validate_for_task(task_blended)
    plan_conv.validate_for_task(task_exact)

    def _finish_loop_signed_area(plan: C.MotionPlan) -> float:
        loop_segs = [
            s for s in plan.segments
            if ("_finish_b0_wall_" in s.tag or "_finish_b0_corner_" in s.tag)
        ]
        assert len(loop_segs) > 4
        pts = [s.start_mm for s in loop_segs] + [loop_segs[-1].end_mm]
        assert math.dist(pts[0], pts[-1]) <= 1e-6
        area2 = sum(
            pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
            for i in range(len(pts) - 1)
        )
        return 0.5 * area2

    area_climb = _finish_loop_signed_area(plan_climb)
    area_conv = _finish_loop_signed_area(plan_conv)
    assert area_climb < 0.0, f"expected CW negative signed area for climb, got {area_climb}"
    assert area_conv > 0.0, f"expected CCW positive signed area for conventional, got {area_conv}"
    assert math.isclose(abs(area_climb), abs(area_conv), rel_tol=1e-9)

    nc_conv = post_motion_plan(plan_conv, task_exact, dialect=C.Dialect.HAAS)
    ev_conv = evaluate_posted_gcode(
        task_exact,
        nc_conv.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=EVAL_SETTINGS,
    )
    assert ev_conv.geometry_pass is True
    assert ev_conv.safety_pass is True
    assert ev_conv.metrics["original_part_gouge_volume_mm3"].value == 0.0


def test_multi_depth_axial_stepdown_qualification() -> None:
    """(c) Verify multi-depth axial_stepdown_mm=2.0 produces stepped Z layers and passes Haas eval."""
    bundle = next(
        b for b in build_milling_suite(families=("contour_profile",))
        if b.case.case_id == "contour-test-00"  # 3.5 mm pocket depth -> 2 axial layers (-1.75, -3.5)
    )
    task = bundle.task
    cfg = ContourConfig(axial_stepdown_mm=2.0)
    plan = generate_contour_toolpath(task, cfg)
    plan.validate_for_task(task)
    _assert_rapids_strictly_at_z_safe(task, plan)

    inv = task.work_to_machine.inverse().apply
    cut_zs = sorted(
        {
            round(inv(s.end_mm)[2], 6)
            for s in plan.segments
            if s.kind == C.SegmentKind.FEED
            and abs(inv(s.start_mm)[2] - inv(s.end_mm)[2]) <= 1e-9
            and inv(s.end_mm)[2] < 0.0
        },
        reverse=True,
    )
    assert len(cut_zs) == 2
    assert math.isclose(cut_zs[0], -1.75, abs_tol=1e-6)
    assert math.isclose(cut_zs[1], -3.50, abs_tol=1e-6)
    assert (0.0 - cut_zs[0]) <= 2.0 + 1e-9
    assert (cut_zs[0] - cut_zs[1]) <= 2.0 + 1e-9

    nc_text = post_motion_plan(plan, task, dialect=C.Dialect.HAAS)
    ev = evaluate_posted_gcode(
        task,
        nc_text.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=EVAL_SETTINGS,
    )
    assert ev.geometry_pass is True
    assert ev.safety_pass is True
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0


def test_convex_corner_roll_around_exact_radius_and_tangential_leads() -> None:
    """(d) Verify convex corner roll-around segments maintain exact distance R (±1e-6 mm) to boss vertices."""
    bundle = next(
        b for b in build_milling_suite(families=("contour_profile",))
        if b.case.case_id == "contour-dev-01"
    )
    task = bundle.task
    r_tool = float(task.tool.cutter_radius_mm)
    ilo, ihi = task.original_part.params["islands_mm"][0]
    ix0, iy0 = float(ilo[0]), float(ilo[1])
    ix1, iy1 = float(ihi[0]), float(ihi[1])
    corner_vertices: Dict[str, Tuple[float, float]] = {
        "bl": (ix0, iy0),
        "tl": (ix0, iy1),
        "tr": (ix1, iy1),
        "br": (ix1, iy0),
    }

    cfg = ContourConfig(corner_roll_arc_segments=7, lead_in_radius_mm=1.2)
    plan = generate_contour_toolpath(task, cfg)
    inv = task.work_to_machine.inverse().apply

    for c_name, (vx, vy) in corner_vertices.items():
        c_segs: List[C.Segment] = [
            s for s in plan.segments if f"_finish_b0_corner_{c_name}_" in s.tag
        ]
        assert len(c_segs) == cfg.corner_roll_arc_segments
        for seg in c_segs:
            assert math.isclose(seg.feed_mm_s or 0.0, cfg.finish_feed_mm_s, abs_tol=1e-9)
            sx, sy, _ = inv(seg.start_mm)
            ex, ey, _ = inv(seg.end_mm)
            d_start = math.hypot(sx - vx, sy - vy)
            d_end = math.hypot(ex - vx, ey - vy)
            assert abs(d_start - r_tool) <= 1e-6, (
                f"corner {c_name} {seg.tag} start distance {d_start} != R={r_tool}"
            )
            assert abs(d_end - r_tool) <= 1e-6, (
                f"corner {c_name} {seg.tag} end distance {d_end} != R={r_tool}"
            )

    # Also verify plunge and retract occur off-wall in cleared shoulder space (distance == R + lead_in_radius_mm)
    plunge_seg = next(s for s in plan.segments if s.tag == "L0_finish_b0_plunge")
    retract_seg = next(s for s in plan.segments if s.tag == "L0_finish_b0_retract")
    for label, pt_mach in (("plunge", plunge_seg.end_mm), ("retract", retract_seg.start_mm)):
        px, py, _ = inv(pt_mach)
        d_boss = math.hypot(max(0.0, ix0 - px, px - ix1), max(0.0, iy0 - py, py - iy1))
        assert d_boss >= r_tool + cfg.lead_in_radius_mm - 1e-6, (
            f"finish {label} at ({px}, {py}) too close to boss wall: d_boss={d_boss}"
        )


def test_tight_envelope_clamps_lead_in_safely() -> None:
    """Verify lead-in/lead-out clamping when lead_in_radius_mm exceeds tight machine envelope clearance."""
    task = _make_custom_contour_task(
        task_id="contour-tight-envelope",
        stock_min_mm=(-12.0, -10.0, -8.0),
        stock_max_mm=(12.0, 10.0, 0.0),
        pocket_z_min=-2.5,
        islands_mm=(((-8.5, -6.5, -2.5), (8.5, 6.5, 0.0)),),
        envelope_margin_mm=0.2,  # tight envelope: stock +- 0.2 mm
    )
    cfg = ContourConfig(lead_in_radius_mm=3.5)
    plan = generate_contour_toolpath(task, cfg)
    plan.validate_for_task(task)
    for seg in plan.segments:
        for pt in (seg.start_mm, seg.end_mm):
            for axis in range(3):
                assert task.limits.envelope_min_mm[axis] <= pt[axis] <= task.limits.envelope_max_mm[axis]
