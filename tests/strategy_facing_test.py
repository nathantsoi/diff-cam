# SPDX-License-Identifier: MIT
"""Qualification test suite for the MIT clean-room 2.5D Facing strategy."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import pytest

from benchmarks import contracts as C
from benchmarks.milling_suite import (
    DEFAULT_VOXEL_MM,
    MillingCaseSpec,
    build_milling_suite,
    derive_multi_pocket_target,
)
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from cam.strategies.facing import FacingConfig, PROVENANCE, generate_facing_toolpath

EVAL_SETTINGS = {
    "max_step_mm": 0.25,
    "compute_bounds": False,
    "safety_screen": True,
}


def _make_open_facing_task(
    task_id: str,
    stock_min_mm: Tuple[float, float, float],
    stock_max_mm: Tuple[float, float, float],
    face_z_mm: float,
    islands_mm: Sequence[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    *,
    voxel_mm: float = DEFAULT_VOXEL_MM,
    timing_rule: C.TimingRule = C.TimingRule.BLENDED_ACCEL_BOUNDED,
    work_offset_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> C.TaskSpec:
    """Build a synthetic open-perimeter facing TaskSpec with 0..N rectangular bosses."""
    shape = tuple(
        int(round((hi - lo) / voxel_mm))
        for lo, hi in zip(stock_min_mm, stock_max_mm)
    )
    grid = C.GridSpec(shape=shape, origin_mm=stock_min_mm, voxel_mm=voxel_mm, frame=C.Frame.WORK)
    stock = C.GeometrySource(
        kind="analytic",
        params={"primitive": "box", "min_mm": list(stock_min_mm), "max_mm": list(stock_max_mm)},
        frame=C.Frame.WORK,
    )
    pocket_min = (stock_min_mm[0], stock_min_mm[1], face_z_mm)
    pocket_max = (stock_max_mm[0], stock_max_mm[1], stock_max_mm[2])
    original_part = C.GeometrySource(
        kind="analytic",
        params={
            "primitive": "box_minus_multi_pocket",
            "box_min_mm": list(stock_min_mm),
            "box_max_mm": list(stock_max_mm),
            "pocket_min_mm": list(pocket_min),
            "pocket_max_mm": list(pocket_max),
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
    excl_reason = (
        f"Finite 2.5 mm flat cutter walled-corner fillet exclusion ({excl_mm3} mm^3)."
        if excl_mm3 > 0.0
        else ""
    )

    w2m = C.Transform.translation(work_offset_mm)
    env_min = w2m.apply((stock_min_mm[0] - 10.0, stock_min_mm[1] - 10.0, stock_min_mm[2] - 4.0))
    env_max = w2m.apply((stock_max_mm[0] + 10.0, stock_max_mm[1] + 10.0, 60.0))
    limits = C.MachineLimits(
        max_feed_mm_s=20.0,
        rapid_mm_s=100.0,
        max_accel_mm_s2=150.0,
        envelope_min_mm=env_min,
        envelope_max_mm=env_max,
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
        family="facing_open_boss",
        split=C.Split.DEV,
        source=C.SourceInfo(origin=f"strategy_facing_test:{task_id}", license="MIT"),
        length_unit="mm",
        stock=stock,
        original_part=original_part,
        operation_target=C.OperationTarget(
            geometry=op_geom,
            excluded_volume_mm3=excl_mm3,
            exclusion_reason=excl_reason,
        ),
        grid=grid,
        work_to_machine=w2m,
        initial_state=C.MachineState(position_mm=w2m.apply((0.0, 0.0, 6.0)), frame=C.Frame.MACHINE),
        tool=tool,
        limits=limits,
        tolerances=tolerances,
        motion_semantics=C.MotionSemantics(timing_rule=timing_rule),
    )


def _qualify_haas_facing(task: C.TaskSpec, config: FacingConfig = FacingConfig()) -> Tuple[C.MotionPlan, C.EvaluationResult]:
    """Generate facing plan, validate contracts, post Haas NGC, and run hard evaluator."""
    plan = generate_facing_toolpath(task, config=config)
    assert plan.frame == C.Frame.MACHINE
    assert plan.provenance == PROVENANCE
    plan.validate_for_task(task)

    inv_mach = task.work_to_machine.inverse().apply
    z_safe = max(float(inv_mach(task.initial_state.position_mm)[2]), float(task.stock.params["max_mm"][2]) + 2.0)
    for seg in plan.segments:
        if seg.kind == C.SegmentKind.RAPID:
            ws = inv_mach(seg.start_mm)
            we = inv_mach(seg.end_mm)
            assert math.isclose(ws[2], z_safe, abs_tol=1e-6)
            assert math.isclose(we[2], z_safe, abs_tol=1e-6)
        for pt in (seg.start_mm, seg.end_mm):
            for axis in range(3):
                assert task.limits.envelope_min_mm[axis] - 1e-6 <= pt[axis] <= task.limits.envelope_max_mm[axis] + 1e-6

    nc_text = post_motion_plan(plan, task, dialect=C.Dialect.HAAS)
    ev = evaluate_posted_gcode(
        task,
        nc_text.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=EVAL_SETTINGS,
    )
    assert ev.coverage == C.Coverage.COMPLETE, f"incomplete coverage: {ev.coverage_reasons}"
    assert ev.geometry_pass is True, f"geometry failed: {ev.metrics}"
    assert ev.safety_pass is True, f"safety failed: {ev.coverage_reasons} / {ev.events}"
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert ev.metrics["gouge_volume_mm3"].value == 0.0
    return plan, ev


@pytest.mark.parametrize(
    "case_id",
    ["face-dev-00", "face-dev-01", "face-test-00", "face-test-01"],
)
def test_milling_suite_facing_open_boss_tasks(case_id: str) -> None:
    bundles = {
        b.case.case_id: b
        for b in build_milling_suite(families=("facing_open_boss",))
    }
    assert set(bundles.keys()) == {"face-dev-00", "face-dev-01", "face-test-00", "face-test-01"}
    bundle = bundles[case_id]
    _, ev = _qualify_haas_facing(bundle.task)
    assert ev.metrics["residual_volume_mm3"].value == 0.0


def test_synthetic_two_boss_open_facing_task() -> None:
    task = _make_open_facing_task(
        task_id="synth-facing-two-boss",
        stock_min_mm=(-18.0, -13.0, -8.0),
        stock_max_mm=(18.0, 13.0, 0.0),
        face_z_mm=-2.5,
        islands_mm=(
            ((-12.5, -8.0, -2.5), (-4.5, -2.0, 0.0)),
            ((3.5, 1.5, -2.5), (12.0, 8.0, 0.0)),
        ),
        work_offset_mm=(50.0, -30.0, -100.0),
    )
    _, ev = _qualify_haas_facing(task)
    assert ev.metrics["residual_volume_mm3"].value == 0.0


def test_synthetic_zero_boss_and_exact_stop_facing() -> None:
    task = _make_open_facing_task(
        task_id="synth-facing-zero-boss-exact-stop",
        stock_min_mm=(-14.0, -10.0, -8.0),
        stock_max_mm=(14.0, 10.0, 0.0),
        face_z_mm=-2.0,
        islands_mm=(),
        timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
    )
    plan, ev = _qualify_haas_facing(task)
    assert plan.timing_rule == C.TimingRule.EXACT_STOP_CONSTANT_SPEED
    assert ev.metrics["residual_volume_mm3"].value == 0.0


def test_climb_one_way_direction_mode() -> None:
    bundle = next(
        b for b in build_milling_suite(families=("facing_open_boss",)) if b.case.case_id == "face-dev-00"
    )
    cfg = FacingConfig(direction_mode="climb_one_way", stepover_fraction=0.68)
    plan, ev = _qualify_haas_facing(bundle.task, config=cfg)

    climb_cuts = [s for s in plan.segments if "climb" in s.tag and s.tag.endswith("_cut")]
    assert len(climb_cuts) >= 6
    for seg in climb_cuts:
        dx = seg.end_mm[0] - seg.start_mm[0]
        dy = seg.end_mm[1] - seg.start_mm[1]
        assert dx > 0.0
        assert abs(dy) <= 1e-9
    assert ev.metrics["residual_volume_mm3"].value == 0.0


@pytest.mark.parametrize("angle_deg", [45.0, 90.0])
def test_rotated_raster_angles(angle_deg: float) -> None:
    bundle = next(
        b for b in build_milling_suite(families=("facing_open_boss",)) if b.case.case_id == "face-dev-00"
    )
    cfg = FacingConfig(raster_angle_deg=angle_deg, stepover_fraction=0.65)
    _, ev = _qualify_haas_facing(bundle.task, config=cfg)
    assert ev.metrics["residual_volume_mm3"].value == 0.0


def test_multi_layer_axial_stepdown() -> None:
    bundle = next(
        b for b in build_milling_suite(families=("facing_open_boss",)) if b.case.case_id == "face-test-00"
    )
    cfg = FacingConfig(axial_stepdown_mm=1.5)
    plan, ev = _qualify_haas_facing(bundle.task, config=cfg)

    inv_mach = bundle.task.work_to_machine.inverse().apply
    cut_zs = {
        round(inv_mach(s.end_mm)[2], 4)
        for s in plan.segments
        if s.kind == C.SegmentKind.FEED and math.isclose(inv_mach(s.start_mm)[2], inv_mach(s.end_mm)[2], abs_tol=1e-6)
    }
    assert -1.5 in cut_zs
    assert -3.0 in cut_zs
    assert ev.metrics["residual_volume_mm3"].value == 0.0


def test_facing_config_validation() -> None:
    with pytest.raises(ValueError, match="stepover_fraction"):
        FacingConfig(stepover_fraction=0.1)
    with pytest.raises(ValueError, match="stepover_fraction"):
        FacingConfig(stepover_fraction=0.9)
    with pytest.raises(ValueError, match="raster_angle_deg"):
        FacingConfig(raster_angle_deg=-5.0)
    with pytest.raises(ValueError, match="raster_angle_deg"):
        FacingConfig(raster_angle_deg=180.0)
    with pytest.raises(ValueError, match="direction_mode"):
        FacingConfig(direction_mode="conventional")
    with pytest.raises(ValueError, match="axial_stepdown_mm"):
        FacingConfig(axial_stepdown_mm=0.0)
