"""Qualification tests for the 16-task 4-family MIT canonical milling suite."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks import contracts as C
from benchmarks.milling_suite import (
    MILLING_FAMILIES,
    build_milling_suite,
    make_joint_parameterization,
    materialize_sampled_sources,
    multi_pocket_fields,
)
from cam import ControllerDialect, MachineConfig
from cam.gcode_parser import parse_motion_plan
from cam.posted_evaluator import _analytic_mask, _centres, evaluate_posted_gcode
from cam.posts import post_motion_plan, post_shop_program
from verification.camotics_adapter import _geometry_mask, unsupported_probe


@pytest.mark.parametrize("seed_engine,max_segs", [("v2", 450), ("v1", 60)])
def test_all_16_milling_suite_tasks_qualify_in_posted_evaluator_and_masks_agree(
    tmp_path, seed_engine: str, max_segs: int
):
    bundles = build_milling_suite(voxel_mm=0.5, seed_engine=seed_engine)
    assert len(bundles) == 16
    assert {b.task.family for b in bundles} == set(MILLING_FAMILIES)
    assert sum(b.task.split == C.Split.DEV for b in bundles) == 8
    assert sum(b.task.split == C.Split.TEST for b in bundles) == 8

    for bundle in bundles:
        task = bundle.task
        motion = bundle.nominal_motion
        assert 8 <= len(motion.segments) <= max_segs, (
            f"{task.task_id} ({seed_engine}): unexpected segment count {len(motion.segments)}"
        )

        # Verify exact 3-way mask parity: posted_evaluator == camotics_adapter == sampled SDF < 0
        centres_tuple = _centres(task.grid)
        centres_stack = np.stack(centres_tuple, axis=-1)
        mask_posted_op = _analytic_mask(task.operation_target.geometry, centres_tuple)
        mask_camotics_op = _geometry_mask(task.operation_target.geometry, centres_stack)
        _, sdf_op = multi_pocket_fields(task.stock, task.operation_target.geometry, task.grid)
        assert np.array_equal(mask_posted_op, mask_camotics_op), f"{task.task_id}: op mask mismatch"
        assert np.array_equal(mask_posted_op, sdf_op < 0.0), f"{task.task_id}: sampled SDF<0 mismatch"

        sources = materialize_sampled_sources(task, cache_root=tmp_path)
        assert sources["stock"].kind == "sampled" and sources["target"].kind == "sampled"
        assert unsupported_probe(task, motion) == ()

        # Verify JointPathFeedSpline construction and zero-theta identity
        mapper = make_joint_parameterization(task, motion)
        z_theta = np.zeros(mapper.num_params, dtype=np.float64)
        assert np.allclose(mapper.forward_feed_multipliers(z_theta), 1.0)

        # Verify Haas NGC and RS274 posting + posted_evaluator geometry & safety pass!
        cfg = MachineConfig(tool_number=task.tool.tool_number)
        haas_nc = post_motion_plan(motion, task, dialect=C.Dialect.HAAS, config=cfg)
        rs_nc = post_motion_plan(motion, task, dialect=C.Dialect.RS274, config=cfg)
        assert "G187 P2 E0.05" in haas_nc
        assert "G64 P0.05" in rs_nc
        assert parse_motion_plan(haas_nc, task, dialect=C.Dialect.HAAS).segments

        eval_res = evaluate_posted_gcode(
            task,
            haas_nc.encode("utf-8"),
            dialect=C.Dialect.HAAS,
            settings={"max_step_mm": 0.25, "compute_bounds": False, "safety_screen": True},
        )
        assert eval_res.safety_pass is True, f"{task.task_id} failed safety: {eval_res.coverage_reasons}"
        assert eval_res.metrics["original_part_gouge_volume_mm3"].value == 0.0, (
            f"{task.task_id} gouged original part: {eval_res.metrics['original_part_gouge_volume_mm3'].value}"
        )
        assert eval_res.metrics["gouge_volume_mm3"].value == 0.0, (
            f"{task.task_id} gouged op target: {eval_res.metrics['gouge_volume_mm3'].value}"
        )
        assert eval_res.geometry_pass is True, (
            f"{task.task_id} failed geometry_pass: residual="
            f"{eval_res.metrics['residual_volume_mm3'].value}, "
            f"surf_err={eval_res.metrics['max_surface_error_mm'].value}, "
            f"reasons={eval_res.coverage_reasons}"
        )
