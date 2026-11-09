"""Prototype T3.0 tests: real analytic simulator + synthetic contract artifacts."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

from algorithms import objectives as O
from algorithms import parameterization as P
from algorithms import run_artifacts as R
from benchmarks import audit as A
from benchmarks import contracts as C
from simulator.csg_simulator import CSGSimulatorDelta


def _prototype(seed=7):
    ti.init(arch=ti.cpu, offline_cache=False)
    initial = np.array([-4.0, -4.0, 2.0])
    task = C.example_pocket_task()
    part = C.GeometrySource(kind="analytic", params={"primitive": "sphere", "center": [0.5, 0.5, 0.5],
                                                   "radius_mm": 6.0})
    tool = dataclasses.replace(task.tool, holder_bottom_mm=task.tool.flute_length_mm)
    to_machine = task.work_to_machine.apply
    start = to_machine(initial)
    p1 = to_machine(initial + [0, 0, -6])
    p2 = to_machine(initial + [3, 3, -12])
    state = C.MachineState(position_mm=start, frame=C.Frame.MACHINE)
    segments = (
        C.Segment(id="plunge", kind=C.SegmentKind.FEED, start_mm=start, end_mm=p1,
                  duration_s=0.6, feed_mm_s=10.0, tag="plunge"),
        C.Segment(id="slot", kind=C.SegmentKind.FEED, start_mm=p1, end_mm=p2,
                  duration_s=math.dist(p1, p2) / 10, feed_mm_s=10.0),
    )
    motion = C.MotionPlan(frame=C.Frame.MACHINE, initial_state=state, segments=segments,
                          timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED)
    task = dataclasses.replace(task, original_part=part,
                               operation_target=C.OperationTarget(geometry=part),
                               initial_state=state, tool=tool)
    motion.validate_for_task(task)
    normalized = (initial - task.grid.origin_mm) / task.grid.extent_mm
    sim = CSGSimulatorDelta(
        resolution=16, max_steps=2, stock_size_mm=25.4, voxel_size_mm=task.grid.voxel_mm,
        target_shape="sphere", tool_start=tuple(normalized), enforce_speed_limits=False,
        init_taichi=False, k_init=40,
    )
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.tool_cut_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    sim.set_target_params(radius_mm=6.0)
    sim.bake_target_grid()
    mapper = P.DeltaResidual([[0, 0, -6], [3, 3, -6]], scale_mm=1.0,
                             bounds=(-0.05, 0.05), initial_position_mm=initial)
    evaluator = O.CandidateEvaluator(task, motion, mapper, sim,
                                     spec=O.ObjectiveSpec(), seed=seed)
    return evaluator


def test_aba_invariance_after_explicit_state_pollution_and_seed_determinism():
    evaluator = _prototype(seed=123)
    theta_a = np.array([0.001, -0.002, 0.003, -0.004, 0.002, 0.001])
    theta_b = np.array([-0.003, 0.001, -0.002, 0.001, -0.001, 0.002])
    evaluator.sim.loss[None] = 99.0
    evaluator.sim.tool_pos[0] = [0.777, 0.888, 0.999]
    evaluator.sim.diag_holder[None] = 42.0
    a1 = evaluator.evaluate_candidate(theta_a, "a")
    value_1 = a1.objective
    stock_1 = evaluator.sim.stock.to_numpy()[2].copy()
    b = evaluator.evaluate_candidate(theta_b, "b")
    a2 = evaluator.evaluate_candidate(theta_a, "a2")
    stock_2 = evaluator.sim.stock.to_numpy()[2].copy()
    a3 = evaluator.evaluate_candidate(theta_a, "a3")
    assert a1.valid and b.valid and a2.valid
    assert a1.coverage == C.Coverage.INCOMPLETE
    assert a2.objective == pytest.approx(value_1, abs=2e-6)
    assert a3.objective == pytest.approx(value_1, abs=2e-6)
    assert np.array_equal(stock_1, stock_2)
    assert a1.components["evaluation_seed"] == 123.0
    assert a2.theta_hash == a1.theta_hash
    assert b.theta_hash != a1.theta_hash
    assert a1.components["speed_clipping_applied"] == 0.0


def test_tape_no_tape_and_parameter_pullback_finite_difference_float32():
    evaluator = _prototype()
    theta = np.array([0.002, -0.001, 0.001, -0.001, 0.002, -0.002])
    plain = evaluator.evaluate_candidate(theta, "no-tape")
    with_gradient = evaluator.evaluate_candidate_with_gradient(theta, "tape")
    assert with_gradient.result.valid and with_gradient.result.gradient_available
    assert with_gradient.result.objective == pytest.approx(plain.objective, rel=1e-6, abs=2e-6)
    gradient = with_gradient.gradient_theta
    assert gradient.shape == (6,) and np.linalg.norm(gradient) > 1e-9

    rng = np.random.default_rng(4)
    direction = rng.normal(size=theta.size)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-3  # normalized units = 1 micron physical at scale_mm=1.
    plus = evaluator.evaluate_candidate(theta + epsilon * direction, "fd+")
    minus = evaluator.evaluate_candidate(theta - epsilon * direction, "fd-")
    finite_difference = (plus.objective - minus.objective) / (2 * epsilon)
    # Exact pullbacks use float64 maps; the simulator remains float32 and its
    # smoothed boundary/JIT arithmetic justifies the absolute+relative tolerance.
    np.testing.assert_allclose(np.dot(gradient, direction), finite_difference,
                               rtol=0.08, atol=5e-3)


def test_invalid_parameters_use_consistent_null_outcomes_and_ledger_attempts():
    evaluator = _prototype()
    nan = evaluator.evaluate_candidate([np.nan] * 6, "nan")
    assert not nan.valid and nan.objective is None
    assert nan.coverage == C.Coverage.UNKNOWN and nan.invalid_reason
    outside = evaluator.evaluate_candidate([0.2] * 6, "out-of-bounds")
    assert not outside.valid and "bounds" in outside.invalid_reason
    assert outside.gradient_available is False

    def fail_forward(_steps):
        raise FloatingPointError("synthetic non-finite backend")

    evaluator.sim.forward = fail_forward
    failure = evaluator.evaluate_candidate(np.zeros(6), "bad-backend", objective_kind="smooth")
    assert not failure.valid and failure.objective is None
    assert "non-finite" in failure.invalid_reason
    ledger = evaluator.ledger.snapshot()
    assert ledger["candidate_evaluations"] == 3
    # The two validation failures never reach the simulator. The third attempt
    # issues one forward call even though the injected backend raises.
    assert ledger["forward_calls"] == 1
    assert ledger["backward_calls"] == 0
    assert ledger["entries"][-1]["valid"] is False


def test_hard_counterpart_components_are_explicit_and_reproducible():
    evaluator = _prototype()
    theta = np.array([0.001, 0.001, -0.001, 0.002, -0.002, 0.001])
    first = evaluator.evaluate_candidate(theta, "hard", objective_kind="hard")
    second = evaluator.evaluate_candidate(theta, "hard-again", objective_kind="hard")
    expected = (2.0 * first.components["gouge_fraction"] ** 2
                + first.components["residual_fraction"] ** 2
                + 50.0 * first.components["holder_overlap_fraction"] ** 2)
    assert first.objective == pytest.approx(expected, abs=1e-8)
    assert second.objective == pytest.approx(first.objective, abs=2e-6)
    assert first.components["holder_overlap_fraction"] >= 0
    assert first.components["holder_proxy"] >= 0


def test_compute_ledger_counts_forward_backward_validation_and_work():
    evaluator = _prototype()
    theta = np.zeros(6)
    evaluator.evaluate_candidate(theta, "first")
    evaluator.evaluate_candidate_with_gradient(theta, "second")
    evaluator.evaluate_candidate(np.array([0.2] * 6), "invalid")
    evaluator.ledger.count_validation(2)
    ledger = evaluator.ledger.snapshot()
    n_voxels = 2 * 16 ** 3
    assert ledger["candidate_evaluations"] == 3
    assert ledger["forward_calls"] == 2 and ledger["backward_calls"] == 1
    assert ledger["validation_calls"] == 2
    assert ledger["segment_steps"] == 4  # two segments in each executed call
    assert ledger["voxel_steps"] == 2 * n_voxels
    assert len(ledger["entries"]) == 3
    assert ledger["warmup_wall_time_s"] == ledger["entries"][0]["wall_time_s"]
    total = sum(entry["wall_time_s"] for entry in ledger["entries"])
    assert ledger["end_to_end_candidate_wall_time_s"] == pytest.approx(total)
    assert ledger["steady_state_wall_time_s"] == pytest.approx(total - ledger["entries"][0]["wall_time_s"])
    assert ledger["process_peak_rss_bytes"] > 0


def test_candidate_motion_preserves_feed_kinds_and_exact_timing():
    evaluator = _prototype()
    theta = np.array([0.001, 0.0005, -0.001, 0.002, 0.003, 0.001])
    motion = evaluator.candidate_motion(theta)
    assert [segment.kind for segment in motion.segments] == [C.SegmentKind.FEED] * 2
    for segment in motion.segments:
        assert segment.duration_s == pytest.approx(segment.chord_length_mm / segment.feed_mm_s)
        assert segment.feed_mm_s == 10.0
    zero = evaluator.candidate_motion(np.zeros(6))
    for original, regenerated in zip(evaluator.motion.segments, zero.segments):
        assert regenerated.start_mm == pytest.approx(original.start_mm)
        assert regenerated.end_mm == pytest.approx(original.end_mm)


def _synthetic_evaluation(task, posted):
    return C.EvaluationResult(
        evaluator_id="synthetic-posted-unit", evaluator_revision="unit-only",
        task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
        discretization={"synthetic": True}, coverage=C.Coverage.COMPLETE,
        coverage_reasons=(), metrics={"removed_volume_mm3": C.Metric(20, "mm^3"),
                                      "residual_volume_mm3": C.Metric(1, "mm^3")},
        safety_pass=True, geometry_pass=False, nc_hash=posted.nc_hash,
        dialect=posted.dialect,
    )


def _write_all_artifacts(tmp_path):
    task, motion = C.example_pocket_task(), C.example_motion_plan()
    posting = R.post_motion(task, motion)
    run_id = "artifact-run"
    config = R.build_run_config(run_id, task, motion, seed=7, parameterization=None,
                                objective={"kind": "synthetic_unit", "normalization": "unit"},
                                budgets={"candidate_evaluations": 1}, repository=Path.cwd())
    assert config["source_commit"] and config["uv_lock_hash"].startswith("sha256:")
    files = dict(posting.files)
    files.update({
        "config.json": R.json_bytes(config), "history.json": R.json_bytes({"entries": []}),
        "metrics.json": R.json_bytes({"synthetic": True}),
        "motion_plan.json": R.record_bytes(motion),
        "initial_trajectory.npy": R.numpy_bytes(np.zeros((2, 3))),
        "trajectory.npy": R.numpy_bytes(np.ones((2, 3))),
        "trajectory_deltas.npy": R.numpy_bytes(np.zeros((2, 3))),
    })
    for posted in posting.posted:
        assert posted.eval_file in ("eval_posted_rs274.json", "eval_posted_haas.json")
        files[posted.eval_file] = R.record_bytes(_synthetic_evaluation(task, posted))
    record = R.assemble_record(run_id, task, C.RunStatus.COMPLETE, posting=posting,
                               artifact_names=tuple(sorted(files)),
                               source_commit=config["source_commit"],
                               dirty_patch_hash=config["dirty_patch_hash"])
    return task, posting, record, files


def test_atomic_writer_last_record_and_t40_full_audit(tmp_path, monkeypatch):
    task, posting, record, files = _write_all_artifacts(tmp_path)
    writer = R.ArtifactWriter()
    destinations = []
    original_replace = R.os.replace
    monkeypatch.setattr(R.os, "replace", lambda source, destination: (
        destinations.append(Path(destination).name), original_replace(source, destination)))
    writer.write(str(tmp_path / record.run_id), record, files)
    assert destinations[-1] == "run.json"
    assert set(destinations[:-1]) == set(files)

    manifest = A.load_manifest({
        "schema_version": 1,
        "tasks": [{"task_id": task.task_id, "family": task.family, "task_hash": task.task_hash}],
        "expected_runs": [{"run_id": record.run_id, "task_id": task.task_id,
                           "method": "candidate", "seed": 7}],
    })
    reports, audit = A.audit_runs(tmp_path, manifest)
    assert audit["summary"]["auditor_passed"] is True
    assert reports[0]["usable_for_statistics"] is True
    assert set(reports[0]["dialects"]) == {"rs274", "haas"}


def test_atomic_failure_never_commits_run_json(tmp_path, monkeypatch):
    _, _, record, files = _write_all_artifacts(tmp_path)
    original_replace = R.os.replace
    def fail_last(source, destination):
        if Path(destination).name == "run.json":
            raise OSError("injected final-record failure")
        original_replace(source, destination)
    monkeypatch.setattr(R.os, "replace", fail_last)
    with pytest.raises(OSError, match="injected"):
        R.ArtifactWriter().write(str(tmp_path / record.run_id), record, files)
    assert not (tmp_path / record.run_id / "run.json").exists()
    assert (tmp_path / record.run_id / "config.json").is_file()


def test_unsupported_dialect_is_explicit_and_emits_no_nc():
    task = C.example_pocket_task()
    task = dataclasses.replace(task, motion_semantics=dataclasses.replace(
        task.motion_semantics, dialects=(C.Dialect.RS274,)))
    posting = R.post_motion(task, C.example_motion_plan(task))
    haas = next(post for post in posting.posted if post.dialect == C.Dialect.HAAS)
    assert haas.status == C.PostStatus.UNSUPPORTED and haas.reason
    assert haas.nc_hash is None
    assert "haas.nc" not in posting.files
    assert "rs274.nc" in posting.files


def test_writer_rejects_existing_directory_and_missing_payloads(tmp_path):
    _, _, record, files = _write_all_artifacts(tmp_path)
    with pytest.raises(ValueError, match="exactly match"):
        R.ArtifactWriter().write(str(tmp_path / record.run_id), record, {})
    with pytest.raises(ValueError, match="run directory name"):
        R.ArtifactWriter().write(str(tmp_path / "different-name"), record, files)


def test_sampled_target_real_reset_grid_gradient_and_repeated_source_identity(tmp_path):
    # This is real T2.0 sampled geometry, not a claim about analytic box/pocket scope.
    coordinates = (np.arange(20, dtype=np.float64) + 0.5)
    xyz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    values = np.sqrt(sum((axis - 10.0) ** 2 for axis in xyz)) - 6.0
    path = tmp_path / "target_sphere.npy"
    np.save(path, values)
    source = C.GeometrySource(kind="sampled", path=str(path),
                               sha256=C.sha256_bytes(path.read_bytes()),
                               grid=C.GridSpec(shape=values.shape, origin_mm=(0, 0, 0), voxel_mm=1.0))
    task = C.example_pocket_task()
    stock = C.GeometrySource(kind="analytic", params={"primitive": "box",
                                                       "min_mm": [0, 0, 0], "max_mm": [20, 20, 20]})
    grid = C.GridSpec(shape=values.shape, origin_mm=(0, 0, 0), voxel_mm=1.0)
    initial = np.array([5.0, 5.0, 20.0])
    state = C.MachineState(position_mm=task.work_to_machine.apply(initial), frame=C.Frame.MACHINE)
    p1 = task.work_to_machine.apply(initial + [0, 0, -6])
    p2 = task.work_to_machine.apply(initial + [3, 3, -12])
    segments = (
        C.Segment(id="plunge", kind=C.SegmentKind.FEED, start_mm=state.position_mm,
                  end_mm=p1, duration_s=0.6, feed_mm_s=10.0),
        C.Segment(id="slot", kind=C.SegmentKind.FEED, start_mm=p1, end_mm=p2,
                  duration_s=np.linalg.norm(np.asarray(p1) - p2) / 10.0, feed_mm_s=10.0),
    )
    task = dataclasses.replace(task, stock=stock, original_part=source, grid=grid,
                               operation_target=C.OperationTarget(geometry=source),
                               initial_state=state,
                               tool=dataclasses.replace(task.tool,
                                                        holder_bottom_mm=task.tool.flute_length_mm))
    motion = C.MotionPlan(frame=C.Frame.MACHINE, initial_state=state, segments=segments,
                          timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
                          provenance="T2.0 sampled sphere objective test")
    normalized = (initial - grid.origin_mm) / grid.extent_mm
    mapper = P.DeltaResidual([[0, 0, -6], [3, 3, -6]], scale_mm=1.0,
                             bounds=(-0.05, 0.05), initial_position_mm=initial)

    ti.init(arch=ti.cpu, offline_cache=False)
    sim = CSGSimulatorDelta(resolution=20, max_steps=2, stock_size_mm=20.0, voxel_size_mm=1.0,
                            target_geometry=source, tool_start=tuple(normalized),
                            enforce_speed_limits=False, init_taichi=False, k_init=40)
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.tool_cut_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    sim.bake_target_grid()
    assert sim.target_shape == "sampled"
    assert sim.target_source_metadata["sha256"] == source.sha256
    assert sim.use_sampled_target[None] == 1
    np.testing.assert_allclose(sim.target.to_numpy(), values.astype(np.float32), atol=2e-6, rtol=0)
    reset_calls = []
    official = sim.reset_candidate_state

    def counted_reset():
        reset_calls.append(1)
        return official()

    sim.reset_candidate_state = counted_reset
    evaluator = O.CandidateEvaluator(task, motion, mapper, sim, seed=42)
    theta = np.zeros(mapper.num_params)
    first = evaluator.evaluate_candidate(theta, "sampled-first")
    taped = evaluator.evaluate_candidate_with_gradient(theta, "sampled-grad")
    again = evaluator.evaluate_candidate_with_gradient(theta, "sampled-repeat")
    assert first.valid and first.objective > 0
    assert taped.result.valid and taped.result.gradient_available
    assert taped.result.coverage == C.Coverage.INCOMPLETE
    assert again.result.objective == pytest.approx(taped.result.objective, abs=2e-6)
    assert np.isfinite(taped.gradient_theta).all() and np.linalg.norm(taped.gradient_theta) > 1e-9
    assert len(reset_calls) == 3
    expected_start = evaluator.candidate_motion(theta).initial_state.position_mm
    np.testing.assert_allclose(sim.tool_pos[0].to_numpy(), normalized, atol=2e-6)
    assert list(expected_start) == list(state.position_mm)
