# SPDX-License-Identifier: MIT
"""Qualification tests for the MIT clean-room 2.5D adaptive/trochoidal clearing strategy."""

from __future__ import annotations

import math
from typing import Tuple

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
from cam.strategies.adaptive import (
    PROVENANCE_ID,
    AdaptiveConfig,
    estimate_radial_engagement_profile,
    generate_adaptive_toolpath,
)

POSTED_SETTINGS = {
    "max_step_mm": 0.25,
    "compute_bounds": False,
    "safety_screen": True,
}


def _build_synthetic_slot_task(
    *,
    slot_length_mm: float = 26.0,
    slot_width_mm: float = 8.0,
    slot_depth_mm: float = 3.0,
    cutter_radius_mm: float = 2.5,
    voxel_mm: float = 0.25,
    islands_mm: Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], ...] = (),
    timing_rule: C.TimingRule = C.TimingRule.BLENDED_ACCEL_BOUNDED,
) -> C.TaskSpec:
    """Build a synthetic 2.5D narrow-slot (or multi-island) TaskSpec with exact corner exclusions."""
    hx = 0.5 * slot_length_mm
    hy = 0.5 * slot_width_mm
    stock_min = (-hx - 3.0, -hy - 3.0, -8.0)
    stock_max = (hx + 3.0, hy + 3.0, 0.0)
    pocket_min = (-hx, -hy, -slot_depth_mm)
    pocket_max = (hx, hy, 0.0)

    shape = tuple(int(round((hi - lo) / voxel_mm)) for lo, hi in zip(stock_min, stock_max))
    grid = C.GridSpec(shape=shape, origin_mm=stock_min, voxel_mm=voxel_mm, frame=C.Frame.WORK)
    stock = C.GeometrySource(
        kind="analytic",
        params={"primitive": "box", "min_mm": list(stock_min), "max_mm": list(stock_max)},
        frame=C.Frame.WORK,
    )
    original_part = C.GeometrySource(
        kind="analytic",
        params={
            "primitive": "box_minus_multi_pocket",
            "box_min_mm": list(stock_min),
            "box_max_mm": list(stock_max),
            "pocket_min_mm": list(pocket_min),
            "pocket_max_mm": list(pocket_max),
            "islands_mm": [[list(ilo), list(ihi)] for ilo, ihi in islands_mm],
        },
        frame=C.Frame.WORK,
    )
    tool = C.ToolAssembly(
        cutter_kind=C.CutterKind.FLAT,
        cutter_radius_mm=cutter_radius_mm,
        flute_length_mm=12.0,
        shank_radius_mm=cutter_radius_mm,
        holder_bottom_mm=14.0,
        holder_radius_mm=8.0,
        holder_height_mm=30.0,
        tool_number=1,
    )
    op_geom = derive_multi_pocket_target(stock, original_part, tool, grid)
    excl_mm3 = float(op_geom.params["grid_excluded_volume_mm3"])
    excl_reason = (
        f"Finite {cutter_radius_mm} mm cutter corner fillet exclusion ({excl_mm3} mm^3)."
        if excl_mm3 > 0.0
        else ""
    )
    return C.TaskSpec(
        task_id="synthetic-narrow-slot-trochoid",
        family="adaptive_channel",
        split=C.Split.TEST,
        source=C.SourceInfo(origin="test:synthetic_slot@1", license="MIT"),
        length_unit="mm",
        stock=stock,
        original_part=original_part,
        operation_target=C.OperationTarget(
            geometry=op_geom,
            excluded_volume_mm3=excl_mm3,
            exclusion_reason=excl_reason,
        ),
        grid=grid,
        work_to_machine=C.Transform.translation((0.0, 0.0, 0.0)),
        initial_state=C.MachineState(position_mm=(0.0, 0.0, 6.0), frame=C.Frame.MACHINE),
        tool=tool,
        limits=C.MachineLimits(
            max_feed_mm_s=20.0,
            rapid_mm_s=100.0,
            max_accel_mm_s2=150.0,
            envelope_min_mm=(stock_min[0] - 10.0, stock_min[1] - 10.0, stock_min[2] - 4.0),
            envelope_max_mm=(stock_max[0] + 10.0, stock_max[1] + 10.0, 60.0),
        ),
        tolerances=C.Tolerances(
            completion_residual_mm3=2.0,
            gouge_volume_mm3=0.0,
            local_surface_error_mm=0.5,
            clearance_mm=0.25,
            rapid_penetration_mm=0.0,
        ),
        motion_semantics=C.MotionSemantics(timing_rule=timing_rule),
    )


_SUITE_BUNDLES = tuple(
    b
    for b in build_milling_suite()
    if b.case.family == "adaptive_channel"
    or (b.case.family == "pocket_island" and b.case.case_id in ("pocket-dev-00", "pocket-test-00"))
)


@pytest.mark.parametrize("bundle", _SUITE_BUNDLES, ids=lambda b: b.case.case_id)
def test_milling_suite_adaptive_haas_qualification(bundle) -> None:
    """Verify all 4 adaptive_channel and 2 pocket_island tasks pass posted Haas NGC evaluation."""
    task = bundle.task
    plan = generate_adaptive_toolpath(task)
    plan.validate_for_task(task)
    assert plan.provenance == PROVENANCE_ID

    inv = task.work_to_machine.inverse().apply
    z_safe = inv(task.initial_state.position_mm)[2]
    for seg in plan.segments:
        if seg.kind == C.SegmentKind.RAPID:
            assert math.isclose(inv(seg.start_mm)[2], z_safe, abs_tol=1e-6)
            assert math.isclose(inv(seg.end_mm)[2], z_safe, abs_tol=1e-6)

    # Confirm genuine helical entry and elevated-feed return strokes exist
    tags = {seg.tag for seg in plan.segments}
    assert "helix_bore" in tags
    assert "wall_finish" in tags
    assert ("trochoid_cut" in tags) or ("peel_cut" in tags)

    nc_text = post_motion_plan(plan, task, dialect=C.Dialect.HAAS)
    ev = evaluate_posted_gcode(
        task,
        nc_text.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=POSTED_SETTINGS,
    )
    assert ev.coverage == C.Coverage.COMPLETE
    assert ev.geometry_pass is True, f"{bundle.case.case_id} failed geometry: {ev.metrics}"
    assert ev.safety_pass is True, f"{bundle.case.case_id} failed safety: {ev.coverage_reasons}"
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert ev.metrics["gouge_volume_mm3"].value == 0.0


def test_narrow_slot_trochoidal_engagement_bound_and_haas_qualification() -> None:
    """Verify 8.0 mm narrow slot with 5.0 mm cutter bounds radial engagement via trochoids."""
    task = _build_synthetic_slot_task(
        slot_length_mm=26.0,
        slot_width_mm=8.0,
        slot_depth_mm=3.0,
        cutter_radius_mm=2.5,
    )
    cfg = AdaptiveConfig(
        optimal_load_fraction=0.35,
        trochoid_pitch_fraction=0.32,
        return_feed_multiplier=1.8,
        nominal_feed_mm_s=5.5,
    )
    plan = generate_adaptive_toolpath(task, config=cfg)
    plan.validate_for_task(task)

    ae_mm = estimate_radial_engagement_profile(plan, task, pitch_mm=0.25)
    assert ae_mm.shape == (len(plan.segments),)

    trochoid_cut_indices = [
        idx for idx, seg in enumerate(plan.segments) if seg.tag == "trochoid_cut"
    ]
    trochoid_ret_indices = [
        idx for idx, seg in enumerate(plan.segments) if seg.tag == "trochoid_ret"
    ]
    assert len(trochoid_cut_indices) >= 20, "expected genuine discretized trochoidal cutting arcs"
    assert len(trochoid_ret_indices) >= 5, "expected disengaged trochoidal return strokes"

    cut_ae = ae_mm[trochoid_cut_indices]
    engaged_cut_ae = cut_ae[cut_ae > 0.05]
    assert engaged_cut_ae.size >= 15

    cutter_diam_mm = 2.0 * task.tool.cutter_radius_mm  # 5.0 mm
    mean_ae = float(engaged_cut_ae.mean())
    median_ae = float(sorted(engaged_cut_ae)[len(engaged_cut_ae) // 2])
    ret_mean_ae = float(ae_mm[trochoid_ret_indices].mean())

    # Full-width slotting would be 5.0 mm; trochoidal strokes must bound mean/median well below 5.0 mm
    assert mean_ae < 0.55 * cutter_diam_mm, f"trochoidal mean ae {mean_ae:.3f} mm too high vs D={cutter_diam_mm}"
    assert median_ae < 0.55 * cutter_diam_mm, f"trochoidal median ae {median_ae:.3f} mm too high vs D={cutter_diam_mm}"
    assert ret_mean_ae < 0.15, f"return strokes should traverse cleared IPW, got mean ae={ret_mean_ae:.3f} mm"

    # Verify return strokes run at return_feed_multiplier * nominal_feed_mm_s
    for idx in trochoid_ret_indices:
        assert math.isclose(
            plan.segments[idx].feed_mm_s or 0.0,
            min(cfg.return_feed_multiplier * cfg.nominal_feed_mm_s, task.limits.max_feed_mm_s),
            rel_tol=1e-6,
        )

    nc_text = post_motion_plan(plan, task, dialect=C.Dialect.HAAS)
    ev = evaluate_posted_gcode(
        task,
        nc_text.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=POSTED_SETTINGS,
    )
    assert ev.geometry_pass is True, f"narrow slot failed geometry: {ev.metrics}"
    assert ev.safety_pass is True, f"narrow slot failed safety: {ev.coverage_reasons}"
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert ev.metrics["gouge_volume_mm3"].value == 0.0


def test_three_island_pocket_and_exact_stop_timing() -> None:
    """Verify N=3 standing islands and EXACT_STOP_CONSTANT_SPEED timing rule."""
    task = _build_synthetic_slot_task(
        slot_length_mm=30.0,
        slot_width_mm=24.0,
        slot_depth_mm=2.5,
        cutter_radius_mm=2.5,
        islands_mm=(
            ((-9.0, -6.0, -2.5), (-4.0, -1.5, 0.0)),
            ((-1.0, 2.0, -2.5), (4.5, 6.5, 0.0)),
            ((5.5, -6.5, -2.5), (10.0, -2.0, 0.0)),
        ),
        timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
    )
    cfg = AdaptiveConfig(return_feed_multiplier=4.0, nominal_feed_mm_s=6.0)
    plan = generate_adaptive_toolpath(task, config=cfg)
    plan.validate_for_task(task)
    assert plan.timing_rule == C.TimingRule.EXACT_STOP_CONSTANT_SPEED
    assert all((seg.feed_mm_s or 0.0) <= task.limits.max_feed_mm_s for seg in plan.segments)

    nc_text = post_motion_plan(plan, task, dialect=C.Dialect.HAAS)
    ev = evaluate_posted_gcode(
        task,
        nc_text.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=POSTED_SETTINGS,
    )
    assert ev.geometry_pass is True
    assert ev.safety_pass is True
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0


def test_adaptive_stay_down_linking_and_multidepth_stepdown() -> None:
    """Verify stay-down 2D IPW routing minimizes Z-retracts and steps down axially without lifting."""
    for bundle in _SUITE_BUNDLES:
        plan = generate_adaptive_toolpath(bundle.task)
        retracts = sum(1 for seg in plan.segments if seg.tag == "retract")
        assert retracts <= 8, f"{bundle.case.case_id} had excessive Z-retracts: {retracts}"

    # Deep 3-layer pocket with standing island: must step down axially per component stay-down
    deep_task = _build_synthetic_slot_task(
        slot_length_mm=28.0,
        slot_width_mm=20.0,
        slot_depth_mm=3.6,
        cutter_radius_mm=2.5,
        islands_mm=(((-3.5, -3.0, -3.6), (3.5, 3.0, 0.0)),),
    )
    cfg = AdaptiveConfig(max_doc_mm=1.25, nominal_feed_mm_s=6.0, return_feed_multiplier=2.2)
    plan = generate_adaptive_toolpath(deep_task, config=cfg)
    plan.validate_for_task(deep_task)

    # Single connected component across 3 axial passes should only retract once at completion!
    retracts = sum(1 for seg in plan.segments if seg.tag == "retract")
    assert retracts <= 2, f"expected stay-down axial stepdowns inside single component, got {retracts} retracts"

    nc_text = post_motion_plan(plan, deep_task, dialect=C.Dialect.HAAS)
    ev = evaluate_posted_gcode(
        deep_task,
        nc_text.encode("utf-8"),
        dialect=C.Dialect.HAAS,
        settings=POSTED_SETTINGS,
    )
    assert ev.geometry_pass is True
    assert ev.safety_pass is True
    assert ev.metrics["original_part_gouge_volume_mm3"].value == 0.0


def test_adaptive_config_validation() -> None:
    with pytest.raises(ValueError, match="optimal_load_fraction"):
        AdaptiveConfig(optimal_load_fraction=1.2)
    with pytest.raises(ValueError, match="return_feed_multiplier"):
        AdaptiveConfig(return_feed_multiplier=0.5)
    with pytest.raises(ValueError, match="ipw_pitch_mm"):
        AdaptiveConfig(ipw_pitch_mm=-0.1)
    with pytest.raises(ValueError, match="max_doc_mm"):
        AdaptiveConfig(max_doc_mm=0.0)
