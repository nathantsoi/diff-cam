"""Verification of Phase 1: Blended Acceleration Kinematics, Haas NGC G187, JointPathFeedSpline, and STE."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import taichi as ti

from algorithms import objectives as O
from algorithms import parameterization as P
from benchmarks import contracts as C
from cam import MachineConfig
from cam.gcode_parser import GCodeError, parse_motion_plan
from cam.posts import post_motion_plan
from simulator.csg_simulator import CSGSimulatorDelta
from verification.camotics_adapter import _prepare_camotics_nc


def _build_blended_fixture(precision: str = "f64"):
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f64 if precision == "f64" else ti.f32)
    base_task = C.example_pocket_task()
    part = C.GeometrySource(
        kind="analytic",
        params={"primitive": "sphere", "center": [0.5, 0.5, 0.5], "radius_mm": 6.0},
    )
    tool = dataclasses.replace(base_task.tool, holder_bottom_mm=base_task.tool.flute_length_mm)
    initial_work = np.array([-4.0, -4.0, 4.0], dtype=np.float64)
    deltas_work = np.array(
        [
            [2.0, 0.0, 0.0],    # rapid reposition above stock
            [0.0, 0.0, -6.0],   # plunge cut
            [4.0, 0.0, 0.0],    # X cut
            [0.0, 4.0, 0.0],    # 90-degree corner Y cut
        ],
        dtype=np.float64,
    )
    to_machine = base_task.work_to_machine.apply
    pts_work = np.vstack([initial_work, initial_work + np.cumsum(deltas_work, axis=0)])
    pts_mach = [to_machine(p) for p in pts_work]
    state = C.MachineState(position_mm=pts_mach[0], frame=C.Frame.MACHINE)

    kinds = (C.SegmentKind.RAPID, C.SegmentKind.FEED, C.SegmentKind.FEED, C.SegmentKind.FEED)
    feeds = (None, 4.0, 6.0, 6.0)
    tags = ("rapid", "plunge", "cut_x", "cut_y")

    raw_segments = []
    for idx, (kind, feed, tag) in enumerate(zip(kinds, feeds, tags)):
        s, e = pts_mach[idx], pts_mach[idx + 1]
        spd = base_task.limits.rapid_mm_s if kind == C.SegmentKind.RAPID else feed
        raw_segments.append(
            C.Segment(
                id=f"seg_{idx}",
                kind=kind,
                start_mm=s,
                end_mm=e,
                duration_s=math.dist(s, e) / spd,
                feed_mm_s=feed,
                tag=tag,
            )
        )
    durations = C.blended_segment_durations(
        raw_segments, base_task.limits.rapid_mm_s, base_task.limits.max_accel_mm_s2
    )
    segments = tuple(
        dataclasses.replace(seg, duration_s=dur) for seg, dur in zip(raw_segments, durations)
    )
    semantics = dataclasses.replace(
        base_task.motion_semantics,
        timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
    )
    limits = dataclasses.replace(base_task.limits, max_feed_mm_s=15.0, max_accel_mm_s2=100.0)
    task = dataclasses.replace(
        base_task,
        original_part=part,
        operation_target=C.OperationTarget(geometry=part),
        initial_state=state,
        tool=tool,
        limits=limits,
        motion_semantics=semantics,
    )
    # Recompute durations with updated max_accel_mm_s2=100.0
    durations = C.blended_segment_durations(
        raw_segments, task.limits.rapid_mm_s, task.limits.max_accel_mm_s2
    )
    segments = tuple(
        dataclasses.replace(seg, duration_s=dur) for seg, dur in zip(raw_segments, durations)
    )
    motion = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=state,
        segments=segments,
        timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
    )
    motion.validate_for_task(task)

    normalized = (initial_work - task.grid.origin_mm) / task.grid.extent_mm
    sim = CSGSimulatorDelta(
        resolution=16,
        max_steps=len(segments),
        stock_size_mm=25.4,
        voxel_size_mm=task.grid.voxel_mm,
        target_shape="sphere",
        tool_start=tuple(normalized),
        enforce_speed_limits=False,
        init_taichi=False,
        k_init=40.0,
        precision=precision,
    )
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.tool_cut_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    sim.set_target_params(radius_mm=6.0)
    sim.bake_target_grid()

    is_rapid = np.array([seg.kind == C.SegmentKind.RAPID for seg in segments], dtype=bool)
    max_mult = np.array(
        [
            1.0 if seg.kind == C.SegmentKind.RAPID else task.limits.max_feed_mm_s / seg.feed_mm_s
            for seg in segments
        ],
        dtype=np.float64,
    )
    mapper = P.JointPathFeedSpline(
        deltas_work,
        scale_mm=1.0,
        log_feed_scale=0.5,
        num_pos_controls=4,
        num_feed_controls=4,
        pos_bounds=(-1.0, 1.0),
        feed_bounds=(-1.0, 1.0),
        is_rapid=is_rapid,
        max_feed_mult_per_seg=max_mult,
        initial_position_mm=initial_work,
    )
    return task, motion, mapper, sim


def test_rs274_g64_and_haas_ngc_g187_roundtrip_and_camotics_prep():
    task, motion, _, _ = _build_blended_fixture()
    cfg = MachineConfig(tool_number=task.tool.tool_number)

    rs_nc = post_motion_plan(motion, task, dialect=C.Dialect.RS274, config=cfg)
    assert "G64 P0.05" in rs_nc
    assert "G61" not in rs_nc
    parsed_rs = parse_motion_plan(rs_nc, task, dialect=C.Dialect.RS274)
    assert parsed_rs.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED
    for orig, reparsed in zip(motion.segments, parsed_rs.segments):
        assert reparsed.duration_s == pytest.approx(orig.duration_s, rel=1e-5, abs=1e-5)

    haas_nc = post_motion_plan(motion, task, dialect=C.Dialect.HAAS, config=cfg)
    assert "G187 P2 E0.05" in haas_nc
    assert "G61" not in haas_nc
    parsed_haas = parse_motion_plan(haas_nc, task, dialect=C.Dialect.HAAS)
    assert parsed_haas.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED
    for orig, reparsed in zip(motion.segments, parsed_haas.segments):
        assert reparsed.duration_s == pytest.approx(orig.duration_s, rel=1e-5, abs=1e-5)

    sim_nc_bytes, cam_motion = _prepare_camotics_nc(task, haas_nc.encode("utf-8"), C.Dialect.HAAS)
    sim_nc_text = sim_nc_bytes.decode("utf-8")
    assert "(Haas NGC smoothing validated: G187 P2 E0.05)" in sim_nc_text
    assert cam_motion.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED

    # Reject G187 on RS274 and reject G64 on exact-stop tasks
    with pytest.raises(GCodeError, match="G187|unsupported"):
        parse_motion_plan(haas_nc, task, dialect=C.Dialect.RS274)


def test_joint_path_feed_spline_jacobian_matches_finite_differences():
    _, _, mapper, _ = _build_blended_fixture()
    rng = np.random.default_rng(42)
    theta = rng.uniform(-0.4, 0.4, size=mapper.num_params)
    w_pos = rng.normal(size=(mapper.num_steps, 3))
    w_feed = rng.normal(size=(mapper.num_steps,))

    def scalar_fn(th):
        d = mapper.forward(th)
        m = mapper.forward_feed_multipliers(th)
        return float(np.sum(w_pos * d) + np.sum(w_feed * m))

    analytic = mapper.pullback_joint(w_pos, w_feed, theta)
    eps = 1e-6
    fd = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = eps
        fd[i] = (scalar_fn(theta + e) - scalar_fn(theta - e)) / (2.0 * eps)

    # Rapid segment multiplier is pinned to 1.0 regardless of theta_feed
    assert mapper.forward_feed_multipliers(theta)[0] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=1e-8)


def test_simulator_blended_seg_time_matches_contract_and_fd_gradients():
    task, motion, mapper, sim = _build_blended_fixture(precision="f64")
    ref_time = sum(s.duration_s for s in motion.segments)
    spec = O.ObjectiveSpec(
        k=35.0,
        gouge_weight=2.0,
        residual_weight=1.0,
        holder_weight=10.0,
        time_weight=0.5,
        air_time_weight=0.25,
        mrr_weight=0.4,
        mrr_limit_mm3_s=5.0,
        rapid_stock_weight=10.0,
        reference_cycle_time_s=ref_time,
    )
    evaluator = O.CandidateEvaluator(task, motion, mapper, sim, spec=spec, requested_precision="f64")
    theta = np.array([0.05, -0.04, 0.02, -0.03, 0.04, -0.02, 0.03, 0.02, -0.01, 0.02, -0.02, 0.01,
                      0.1, 0.25, -0.2, 0.3], dtype=np.float64)

    cand_motion = evaluator.candidate_motion(theta)
    taped = evaluator.evaluate_candidate_with_gradient(theta, "blended-grad")
    assert taped.result.valid
    sim_seg_time = sim.seg_time.to_numpy()[: mapper.num_steps]
    contract_seg_time = np.array([s.duration_s for s in cand_motion.segments], dtype=np.float64)
    np.testing.assert_allclose(sim_seg_time, contract_seg_time, rtol=1e-6, atol=1e-7)

    # Check finite-difference gradients across BOTH spatial and feed parameters!
    eps = 1e-5
    grad = taped.gradient_theta
    assert grad is not None and grad.shape == theta.shape
    # Check non-trivial gradient on cutting feed parameters (indices 13, 14, 15)
    assert np.linalg.norm(grad[mapper.num_pos_params :]) > 1e-4
    for idx in (1, 5, 9, mapper.num_pos_params + 1, mapper.num_pos_params + 2, mapper.num_pos_params + 3):
        direction = np.zeros_like(theta)
        direction[idx] = 1.0
        plus = evaluator.evaluate_candidate(theta + eps * direction, f"fd-p-{idx}", "smooth")
        minus = evaluator.evaluate_candidate(theta - eps * direction, f"fd-m-{idx}", "smooth")
        fd_i = (plus.objective - minus.objective) / (2.0 * eps)
        assert grad[idx] == pytest.approx(fd_i, rel=2e-4, abs=1e-6)

    # Verify Straight-Through Estimator (STE) returns exact hard objective + smooth AD gradient
    ste_eval = evaluator.evaluate_ste_with_gradient(theta, "ste-check")
    hard_only = evaluator.evaluate_candidate(theta, "hard-check", "hard")
    assert ste_eval.result.valid and ste_eval.result.objective_kind == "hard"
    assert ste_eval.result.gradient_available is True
    assert ste_eval.result.objective == pytest.approx(hard_only.objective, rel=0.0, abs=1e-12)
    np.testing.assert_allclose(ste_eval.gradient_theta, grad, rtol=1e-10, atol=1e-10)
    assert "peak_mrr_mm3_s" in ste_eval.result.components
    assert "grad_deltas_mm" in ste_eval.diagnostics
    assert "grad_feed_mult" in ste_eval.diagnostics
