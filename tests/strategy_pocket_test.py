# SPDX-License-Identifier: MIT
"""Qualification tests for the MIT clean-room 2.5D Distance-Field Pocket strategy."""

from __future__ import annotations

import dataclasses
import math

import pytest

from benchmarks import contracts as C
from benchmarks.milling_suite import (
    MillingCaseSpec,
    build_milling_suite,
    build_milling_task_bundle,
    derive_multi_pocket_target,
)
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from cam.strategies.pocket import PocketConfig, generate_pocket_toolpath

EVAL_SETTINGS = {
    "max_step_mm": 0.25,
    "compute_bounds": False,
    "safety_screen": True,
}

POCKET_BUNDLES = build_milling_suite(families=("pocket_island", "adaptive_channel"))


def _assert_all_rapids_at_safe_z(task: C.TaskSpec, plan: C.MotionPlan) -> None:
    inv = task.work_to_machine.inverse().apply
    stock_max_z = float(task.stock.params["max_mm"][2])
    for seg in plan.segments:
        if seg.kind == C.SegmentKind.RAPID:
            s_w = inv(seg.start_mm)
            e_w = inv(seg.end_mm)
            assert s_w[2] > stock_max_z + 1.0, f"RAPID {seg.id} starts too low: z={s_w[2]}"
            assert e_w[2] > stock_max_z + 1.0, f"RAPID {seg.id} ends too low: z={e_w[2]}"
            assert math.isclose(s_w[2], e_w[2], abs_tol=1e-6), f"RAPID {seg.id} changes Z: {s_w[2]} -> {e_w[2]}"


def _post_and_evaluate_haas(task: C.TaskSpec, plan: C.MotionPlan) -> C.EvaluationResult:
    plan.validate_for_task(task)
    _assert_all_rapids_at_safe_z(task, plan)
    nc_text = post_motion_plan(plan, task, dialect=C.Dialect.HAAS)
    return evaluate_posted_gcode(
        task,
        nc_text.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=EVAL_SETTINGS,
    )


def test_suite_bundle_count() -> None:
    assert len(POCKET_BUNDLES) == 8
    families = [b.case.family for b in POCKET_BUNDLES]
    assert families.count("pocket_island") == 4
    assert families.count("adaptive_channel") == 4


@pytest.mark.parametrize(
    "bundle",
    POCKET_BUNDLES,
    ids=[b.case.case_id for b in POCKET_BUNDLES],
)
def test_milling_suite_pocket_and_adaptive_channel_haas_qualification(bundle) -> None:
    task = bundle.task
    plan = generate_pocket_toolpath(task, PocketConfig())
    assert plan.provenance == "mit_cleanroom:pocket_strategy@2"
    assert plan.timing_rule == task.motion_semantics.timing_rule

    ev = _post_and_evaluate_haas(task, plan)
    assert ev.coverage == C.Coverage.COMPLETE, f"{bundle.case.case_id}: {ev.coverage_reasons}"
    assert ev.safety_pass is True, f"{bundle.case.case_id}: events={ev.events}, reasons={ev.coverage_reasons}"
    assert ev.geometry_pass is True, (
        f"{bundle.case.case_id}: residual={ev.metrics['residual_volume_mm3'].value}, "
        f"gouge={ev.metrics['gouge_volume_mm3'].value}, "
        f"surf_err={ev.metrics['max_surface_error_mm'].value}"
    )
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert ev.metrics["gouge_volume_mm3"].value == 0.0
    assert ev.metrics["residual_volume_mm3"].value == 0.0


def test_zero_island_open_rectangular_pocket() -> None:
    case = MillingCaseSpec(
        family="pocket_island",
        split="dev",
        case_id="open-rect-00",
        stock_min_mm=(-15.0, -12.0, -8.0),
        stock_max_mm=(15.0, 12.0, 0.0),
        pocket_min_mm=(-11.5, -8.5, -3.5),
        pocket_max_mm=(11.5, 8.5, 0.0),
        islands_mm=(),
        notes="0-island walled rectangular pocket.",
    )
    task = build_milling_task_bundle(case).task
    plan = generate_pocket_toolpath(task, PocketConfig())
    assert plan.provenance == "mit_cleanroom:pocket_strategy@2"

    ev = _post_and_evaluate_haas(task, plan)
    assert ev.coverage == C.Coverage.COMPLETE
    assert ev.geometry_pass is True
    assert ev.safety_pass is True
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert ev.metrics["gouge_volume_mm3"].value == 0.0
    assert ev.metrics["residual_volume_mm3"].value == 0.0


def test_entry_mode_ramp_vs_plunge() -> None:
    task = POCKET_BUNDLES[0].task  # pocket-dev-00
    inv = task.work_to_machine.inverse().apply

    ramp_cfg = PocketConfig(entry_mode="ramp", ramp_angle_deg=6.0)
    ramp_plan = generate_pocket_toolpath(task, ramp_cfg)
    ramp_segs = [s for s in ramp_plan.segments if s.tag == "ramp"]
    assert len(ramp_segs) >= 1
    expected_slope = math.tan(math.radians(ramp_cfg.ramp_angle_deg))
    for seg in ramp_segs:
        sw = inv(seg.start_mm)
        ew = inv(seg.end_mm)
        dxy = math.hypot(ew[0] - sw[0], ew[1] - sw[1])
        dz = sw[2] - ew[2]
        assert dxy > 0.0 and dz > 0.0
        assert math.isclose(dz / dxy, expected_slope, rel_tol=1e-4, abs_tol=1e-4)

    ev_ramp = _post_and_evaluate_haas(task, ramp_plan)
    assert ev_ramp.geometry_pass is True
    assert ev_ramp.safety_pass is True
    assert ev_ramp.metrics["original_part_gouge_volume_mm3"].value == 0.0

    plunge_cfg = PocketConfig(entry_mode="plunge")
    plunge_plan = generate_pocket_toolpath(task, plunge_cfg)
    assert not any(s.tag == "ramp" for s in plunge_plan.segments)
    plunge_segs = [s for s in plunge_plan.segments if s.tag == "plunge"]
    assert len(plunge_segs) >= 1
    for seg in plunge_segs:
        sw = inv(seg.start_mm)
        ew = inv(seg.end_mm)
        assert math.hypot(ew[0] - sw[0], ew[1] - sw[1]) <= 1e-6
        assert ew[2] < sw[2]

    ev_plunge = _post_and_evaluate_haas(task, plunge_plan)
    assert ev_plunge.geometry_pass is True
    assert ev_plunge.safety_pass is True
    assert ev_plunge.metrics["original_part_gouge_volume_mm3"].value == 0.0


def test_multi_depth_axial_stepdown() -> None:
    # pocket-test-00 has pocket depth 4.0 mm (-4.0 .. 0.0); stepdown=2.0 -> Z layers at -2.0 and -4.0
    deep_bundle = next(b for b in POCKET_BUNDLES if b.case.case_id == "pocket-test-00")
    task = deep_bundle.task
    inv = task.work_to_machine.inverse().apply

    cfg = PocketConfig(axial_stepdown_mm=2.0, entry_mode="ramp")
    plan = generate_pocket_toolpath(task, cfg)

    cut_zs = sorted(
        {
            round(inv(s.end_mm)[2], 3)
            for s in plan.segments
            if s.tag == "cut" and math.isclose(inv(s.start_mm)[2], inv(s.end_mm)[2], abs_tol=1e-6)
        },
        reverse=True,
    )
    assert cut_zs == [-2.0, -4.0]

    ev = _post_and_evaluate_haas(task, plan)
    assert ev.coverage == C.Coverage.COMPLETE
    assert ev.geometry_pass is True
    assert ev.safety_pass is True
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert ev.metrics["gouge_volume_mm3"].value == 0.0
    assert ev.metrics["residual_volume_mm3"].value == 0.0


def test_exact_stop_timing_and_work_offset_and_stepped_islands() -> None:
    # Construct a custom TaskSpec with non-zero work_to_machine offset, EXACT_STOP_CONSTANT_SPEED,
    # and two islands (one full-height to z=0.0, one submerged stepped island to z=-2.0) in a -4.0 mm pocket.
    voxel_mm = 0.25
    smin, smax = (-16.0, -13.0, -8.0), (16.0, 13.0, 0.0)
    shape = tuple(int(round((hi - lo) / voxel_mm)) for lo, hi in zip(smin, smax))
    grid = C.GridSpec(shape=shape, origin_mm=smin, voxel_mm=voxel_mm, frame=C.Frame.WORK)
    stock = C.GeometrySource(
        kind="analytic",
        params={"primitive": "box", "min_mm": list(smin), "max_mm": list(smax)},
        frame=C.Frame.WORK,
    )
    original_part = C.GeometrySource(
        kind="analytic",
        params={
            "primitive": "box_minus_multi_pocket",
            "box_min_mm": list(smin),
            "box_max_mm": list(smax),
            "pocket_min_mm": [-13.0, -10.0, -4.0],
            "pocket_max_mm": [13.0, 10.0, 0.0],
            "islands_mm": [
                [[-8.0, -3.0, -4.0], [-3.0, 3.0, 0.0]],
                [[3.0, -3.0, -4.0], [8.0, 3.0, -2.0]],
            ],
        },
        frame=C.Frame.WORK,
    )
    tool = POCKET_BUNDLES[0].task.tool
    op_geom = derive_multi_pocket_target(stock, original_part, tool, grid)
    excl_mm3 = float(op_geom.params["grid_excluded_volume_mm3"])
    w2m = C.Transform.translation((120.0, -75.0, -140.0))
    task = dataclasses.replace(
        POCKET_BUNDLES[0].task,
        task_id="custom-stepped-two-island",
        stock=stock,
        original_part=original_part,
        operation_target=C.OperationTarget(
            geometry=op_geom,
            excluded_volume_mm3=excl_mm3,
            exclusion_reason="Walled-corner fillet exclusion." if excl_mm3 > 0 else "",
        ),
        grid=grid,
        work_to_machine=w2m,
        initial_state=C.MachineState(position_mm=w2m.apply((0.0, 0.0, 6.0)), frame=C.Frame.MACHINE),
        limits=C.MachineLimits(
            max_feed_mm_s=20.0,
            rapid_mm_s=100.0,
            max_accel_mm_s2=150.0,
            envelope_min_mm=(80.0, -120.0, -160.0),
            envelope_max_mm=(160.0, -30.0, -80.0),
        ),
        motion_semantics=C.MotionSemantics(timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED),
    )

    plan = generate_pocket_toolpath(task, PocketConfig(axial_stepdown_mm=2.0, entry_mode="ramp"))
    assert plan.timing_rule == C.TimingRule.EXACT_STOP_CONSTANT_SPEED
    ev = _post_and_evaluate_haas(task, plan)
    assert ev.coverage == C.Coverage.COMPLETE
    assert ev.geometry_pass is True
    assert ev.safety_pass is True
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert ev.metrics["gouge_volume_mm3"].value == 0.0
    assert ev.metrics["residual_volume_mm3"].value == 0.0


def test_pocket_config_validation() -> None:
    with pytest.raises(ValueError, match="stepover_fraction"):
        PocketConfig(stepover_fraction=1.2)
    with pytest.raises(ValueError, match="entry_mode"):
        PocketConfig(entry_mode="helix_unknown")
    with pytest.raises(ValueError, match="axial_stepdown_mm"):
        PocketConfig(axial_stepdown_mm=-1.0)
