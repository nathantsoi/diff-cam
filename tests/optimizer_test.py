"""T3.2 Adam reference optimizer tests on shared objective and artifact contracts."""

import dataclasses
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from algorithms import objectives as O
from algorithms import optimize as opt
from algorithms import parameterization as P
from algorithms import run_artifacts as R
from benchmarks import contracts as C


class DifferentiableObjective:
    """Independent differentiable LSE quadratic used for cheap exact gradient tests."""

    def __init__(self, components=None):
        self.parameterization = P.DeltaResidual(np.zeros((1, 3)), scale_mm=1.0,
                                                 bounds=(-3.0, 3.0))
        self.target = np.array([0.3, -0.2, 0.1])
        self.spec = {"objective": "sum_of_squared_errors", "normalization": "raw"
        }
        self.sim = type("ScheduleProbe", (), {"k": {}})()
        self.ledger = R.ComputeLedger()
        self.task = SimpleNamespace(task_hash=C.content_hash("differentiable-test-task"),
                                    tool_setup_hash=C.content_hash("differentiable-test-tool"),
                                    tolerances=C.Tolerances(
            completion_residual_mm3=1000.0, gouge_volume_mm3=0.0,
            local_surface_error_mm=1.0, clearance_mm=0.0))
        self.hard_fraction_denominators_mm3 = {"gouge": 1.0, "residual": 1000.0}
        self.components = components or {"gouge_fraction": 0.0,
                                         "holder_overlap_fraction": 0.0}

    def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
        value = self.parameterization.forward(theta)[0]
        objective = self.hard_objective(theta) if objective_kind == "hard" else float(
            np.sum((value - self.target) ** 2))
        self.ledger.start_candidate(candidate_id, objective_kind)
        result = C.CandidateResult(candidate_id, C.content_hash(np.asarray(theta).tolist()), objective_kind,
                                   float(objective), True,
                                   components=dict(self.components, log_sum_exp=objective,
                                                   residual_fraction=float(np.linalg.norm(value - self.target))))
        self.ledger.finish_candidate(valid=True, forward_calls=1, backward_calls=0,
                                     segment_steps=1, voxel_steps=1)
        return result

    def hard_objective(self, theta):
        # The gradient targets a different point: every sufficiently small Adam
        # update therefore worsens the declared hard seed's quadratic objective.
        return float(np.sum(np.asarray(theta, dtype=float) ** 2))

    def evaluate_candidate_with_gradient(self, theta, candidate_id="pure-gradient"):
        value = self.parameterization.forward(theta)[0]
        gradient_delta = (2 * (value - self.target))[None, :]
        gradient = self.parameterization.pullback(gradient_delta)
        result = self.evaluate_candidate(theta, "pure-gradient", "smooth")
        # A valid gradient invocation has separate forward/backward counters.
        self.ledger.entries[-1]["backward_calls"] = 1
        self.ledger.backward_calls += 1
        out = O.CandidateEvaluation(result, gradient_theta=gradient)
        out.result = dataclasses.replace(out.result, gradient_available=True)
        return out


def _budget(count, interval=0, max_validation=0, **kwargs):
    return opt.B.OptimizationBudget(candidate_evaluations=count,
                                    population_size=None,
                                    validation_interval=interval,
                                    max_validation_calls=max_validation, **kwargs)


def test_t21_pullback_is_exact_on_known_quadratic():
    evaluator = DifferentiableObjective()
    theta = np.array([0.8, -0.4, 0.9])
    measured = evaluator.evaluate_candidate_with_gradient(theta).gradient_theta
    expected = 2 * (theta - evaluator.target)
    np.testing.assert_allclose(measured, expected, rtol=0, atol=1e-12)
    evaluator.evaluate_candidate_with_gradient(theta)
    epsilon = 1e-6
    direction = np.array([0.3, 0.5, -0.7])
    delta = (evaluator.evaluate_candidate(theta + epsilon * direction, "plus").objective
             - evaluator.evaluate_candidate(theta - epsilon * direction, "minus").objective) / (2 * epsilon)
    np.testing.assert_allclose(measured @ direction, delta, rtol=1e-5, atol=1e-10)


def test_hard_selection_prioritizes_modeled_safety_over_dice_or_raw_loss():
    def result(gouge, holder, residual, loss):
        return C.CandidateResult("x", C.content_hash("x"), "hard", loss, True,
                                 components={"gouge_fraction": gouge,
                                             "holder_overlap_fraction": holder,
                                             "residual_fraction": residual})
    policy = opt.B._resolve_hard_selection(DifferentiableObjective(), opt.HardSelectionPolicy())
    safe_with_residual = result(0.0, 0.0, 0.8, 100.0)
    tiny_limit_violation = result(0.01, 0.0, 0.0, 0.0)
    assert opt._hard_rank(safe_with_residual, policy) < opt._hard_rank(tiny_limit_violation, policy)
    complete_but_colliding = result(0.0, 0.2, 0.0, 0.0)
    assert opt._hard_rank(safe_with_residual, policy) < opt._hard_rank(complete_but_colliding, policy)
    missing_marker = result(0.0, 0.0, 0.1, 0.0)
    del missing_marker.components["holder_overlap_fraction"]
    assert opt._hard_rank(missing_marker, policy) is None


def test_fixed_adam_known_gradient_step_clip_schedule_and_budget():
    evaluator = DifferentiableObjective()
    config = opt.AdamConfig(mode="fixed", learning_rate=0.1, k_start=40, k_end=40,
                            max_gradient_norm=0.01)
    theta = np.zeros(3)
    result = opt.solve_adam(evaluator, _budget(5, interval=2, max_validation=1), config,
                            seed=7, x0=theta, run_id="fixed-check")
    assert result.history[0]["k"] == result.history[-1]["k"] == 40
    assert result.history[0]["clip_scale"] < 1.0
    assert result.evaluations == 5 and result.validation_calls == 1
    assert result.forward_calls == 6 and result.backward_calls == 4
    assert result.gradient_updates == 4
    assert result.best.valid and result.best_theta.shape == (3,)
    assert result.history[-1]["soft_objective"] < result.history[0]["soft_objective"]
    ledger = evaluator.ledger.snapshot()
    # One hard init + four gradients + one independent hard validation call.
    assert ledger["candidate_evaluations"] == 5 and ledger["validation_calls"] == 1


def test_fixed_and_annealed_schedule_is_predeclared_and_generation_limited():
    evaluator = DifferentiableObjective()
    config = opt.AdamConfig(mode="annealed", learning_rate=0.001,
                            k_start=20, k_end=120, max_gradient_norm=0.5)
    result = opt.solve_adam(evaluator, _budget(61, max_generations=2), config,
                            seed=1, x0=np.zeros(3), run_id="annealed")
    assert [row["k"] for row in result.history] == [20.0, 120.0]
    assert result.termination_reason == "max_generations"
    assert result.evaluations == 3
    with pytest.raises(ValueError, match="k_start == k_end"):
        opt.AdamConfig(mode="fixed", k_start=10, k_end=20)


def _checkpoint_pair(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    initial = opt.solve_adam(
        DifferentiableObjective(), _budget(3, interval=2, max_validation=1),
        opt.AdamConfig("annealed", learning_rate=0.02, k_start=20, k_end=120,
                       max_gradient_norm=0.5, n_steps=5), seed=5,
        x0=np.array([1.0, 0.5, -0.5]), run_id="replay",
        checkpoint_dir=checkpoint_dir,
    )
    assert initial.best is not None
    return initial, checkpoint_dir


def test_checkpoint_resume_matches_full_run_and_schedule_state(tmp_path):
    initial, directory = _checkpoint_pair(tmp_path)
    checkpoint = directory / "ckpt-000003.json"
    assert checkpoint.is_file()
    restored_evaluator = DifferentiableObjective()
    resumed = opt.resume_adam(
        checkpoint, restored_evaluator, budget=_budget(6, interval=2, max_validation=2),
        run_id="resumed",
    )
    fresh = opt.solve_adam(
        DifferentiableObjective(), _budget(6, interval=2, max_validation=2),
        initial.config, seed=5, x0=np.array([1.0, 0.5, -0.5]), run_id="fresh",
    )
    np.testing.assert_array_equal(resumed.current_theta, fresh.current_theta)
    np.testing.assert_array_equal(resumed.moments, fresh.moments)
    np.testing.assert_array_equal(resumed.velocities, fresh.velocities)
    assert resumed.evaluations == fresh.evaluations == 6
    assert resumed.previous_evaluations == 3
    assert resumed.gradient_updates == fresh.gradient_updates
    assert resumed.forward_calls == fresh.forward_calls == 8
    assert resumed.backward_calls == fresh.backward_calls == 5
    assert resumed.validation_calls == fresh.validation_calls == 2
    assert resumed.best.objective == fresh.best.objective
    restored_snapshot = restored_evaluator.ledger.snapshot()
    assert restored_snapshot["candidate_evaluations"] == 6
    assert restored_snapshot["validation_calls"] == 2
    assert restored_snapshot["forward_calls"] == 8
    assert restored_snapshot["backward_calls"] == 5
    assert [entry["candidate_id"] for entry in restored_snapshot["entries"]].count("replay:init") == 1
    assert any(entry.get("validation", False) for entry in restored_snapshot["entries"])
    replayed = opt.replay_adam_checkpoint(checkpoint, DifferentiableObjective())
    assert replayed["objective_matches_within_tolerance"]
    assert replayed["objective_abs_difference"] is not None
    assert replayed["replayed"].objective == pytest.approx(initial.best.objective, rel=1e-5, abs=1e-9)


def test_adam_checkpoint_rejects_checksum_and_task_changes(tmp_path):
    _, directory = _checkpoint_pair(tmp_path)
    checkpoint = directory / "ckpt-000003.json"
    payload = json.loads(checkpoint.read_text())
    payload["task_setup_hash"] = "sha256:" + "0" * 64
    payload.pop("payload_hash")
    payload["payload_hash"] = C.content_hash(payload)
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(opt.B.BlackBoxError, match="identity differs"):
        opt.resume_adam(checkpoint, DifferentiableObjective())


def _prototype():
    namespace = runpy.run_path(Path(__file__).with_name("objectives_test.py"))
    return namespace["_prototype"](seed=17)


def test_analytic_gradient_path_and_incomplete_posted_artifacts(tmp_path):
    evaluator = _prototype()
    config = opt.AdamConfig(mode="fixed", learning_rate=0.01, k_start=40, k_end=40,
                            max_gradient_norm=0.1)
    result = opt.solve_adam(evaluator, _budget(5, interval=2, max_validation=2), config,
                            seed=9, x0=np.zeros(6), run_id="adam-smooth")
    assert result.best.valid and result.forward_calls > 0
    assert result.backward_calls > 0
    assert result.validation_calls == 2
    assert result.gradient_updates == 4
    assert result.best_theta.shape == (6,)
    record = opt.emit_adam_artifacts(result, evaluator, tmp_path / "adam-smooth", Path.cwd())
    assert record.status == C.RunStatus.INCOMPLETE
    assert C.loads((tmp_path / "adam-smooth/run.json").read_text()) == record
    assert (tmp_path / "adam-smooth/rs274.nc").is_file()
    assert not (tmp_path / "adam-smooth/eval_posted_rs274.json").exists()
    payload = json.loads((tmp_path / "adam-smooth/config.json").read_text())
    assert payload["budgets"]["limits"]["candidate_evaluations"] == 5
    assert payload["budgets"]["spent"]["backward_calls"] == 4


def test_invalid_selection_never_becomes_best_or_crash_checkpoint(tmp_path):
    class InvalidHard(DifferentiableObjective):
        def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
            if objective_kind == "hard":
                self.ledger.start_candidate(candidate_id, objective_kind)
                self.ledger.finish_candidate(valid=False, forward_calls=0, backward_calls=0,
                                             segment_steps=0, voxel_steps=0)
                return C.CandidateResult(candidate_id, C.content_hash("invalid"), "hard",
                                         None, False, "injected invalid validation")
            return super().evaluate_candidate(theta, candidate_id, objective_kind)

    objective = InvalidHard()
    config = opt.AdamConfig("fixed", k_start=40, k_end=40, learning_rate=0.001)
    result = opt.solve_adam(objective, _budget(5, interval=2, max_validation=2), config,
                            x0=np.zeros(3), run_id="invalid-validation",
                            checkpoint_dir=tmp_path / "checkpoints")
    assert result.best is None and result.best_theta is None
    document = json.loads((tmp_path / "checkpoints/final.json").read_text())
    assert document["best"] is None
    with pytest.raises(opt.B.BlackBoxError):
        opt.emit_adam_artifacts(result, objective, tmp_path / "invalid-run", Path.cwd())


def test_invalid_gradient_stops_after_one_attempt():
    class InvalidGradient(DifferentiableObjective):
        def evaluate_candidate_with_gradient(self, theta, candidate_id="bad-gradient"):
            return O.CandidateEvaluation(C.CandidateResult(
                candidate_id, C.content_hash("bad-gradient"), "smooth", None, False,
                "injected invalid gradient"), gradient_theta=None)

    objective = InvalidGradient()
    result = opt.solve_adam(objective, _budget(10), opt.AdamConfig("fixed", k_start=40, k_end=40),
                            x0=np.zeros(3), run_id="invalid-gradient")
    assert result.termination_reason == "invalid_evaluation"
    assert result.evaluations == 2 and result.invalid_evaluations == 1
    assert result.gradient_updates == 0
    assert result.best == result.initial_candidate
    assert result.no_improvement_over_initialization
    assert result.history[0]["theta"] == [0.0, 0.0, 0.0]


def test_seed_tag_does_not_create_fake_adam_replicates():
    config = opt.AdamConfig(mode="fixed", learning_rate=0.01, k_start=40, k_end=40)
    first = opt.solve_adam(DifferentiableObjective(), _budget(3), config, seed=0)
    second = opt.solve_adam(DifferentiableObjective(), _budget(3), config, seed=1)
    assert first.run_id == second.run_id == "adam-fixed-deterministic"
    assert first.deterministic_replicates == second.deterministic_replicates == 1
    np.testing.assert_array_equal(first.current_theta, second.current_theta)
    assert first.best == first.initial_candidate
    assert second.best == second.initial_candidate
    assert first.validation_calls == second.validation_calls == 0


def test_resume_preserves_declared_annealing_length(tmp_path):
    config = opt.AdamConfig(mode="annealed", k_start=10, k_end=100,
                            learning_rate=0.001, max_gradient_norm=0.5)
    initial = opt.solve_adam(
        DifferentiableObjective(), _budget(4, interval=3, max_validation=3), config,
        x0=np.zeros(3), run_id="fixed-schedule", checkpoint_dir=tmp_path / "schedule")
    assert initial.config.n_steps == 3
    document = json.loads((tmp_path / "schedule/final.json").read_text())
    assert document["optimizer_config"]["n_steps"] == 3
    assert document["optimizer_config_hash"] == C.content_hash(document["optimizer_config"])
    resumed = opt.resume_adam(
        tmp_path / "schedule/final.json", DifferentiableObjective(),
        budget=_budget(7, interval=3, max_validation=3), run_id="resumed-schedule")
    assert resumed.config.n_steps == 3
    assert all(row["k"] == 100.0 for row in resumed.history[3:])
    with pytest.raises(opt.B.BlackBoxError, match="no declared annealing"):
        invalid = dict(document)
        invalid["optimizer_config"].pop("n_steps")
        payload = dict(invalid)
        payload.pop("payload_hash")
        invalid["payload_hash"] = C.content_hash(payload)
        (tmp_path / "schedule/invalid.json").write_text(json.dumps(invalid))
        opt.resume_adam(tmp_path / "schedule/invalid.json", DifferentiableObjective())


class ControlledHardObjective(DifferentiableObjective):
    """Choose subsequent hard outcomes without changing the seed or gradient."""

    def __init__(self, *, objective=0.0, gouge=0.0, residual=0.0,
                 invalid_init=False):
        super().__init__()
        self.replacement = {"objective_value": objective, "gouge_fraction": gouge,
                            "residual_fraction": residual, "holder_overlap_fraction": 0.0}
        self.replacement_objective = objective
        self.invalid_init = invalid_init

    def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
        if self.invalid_init and candidate_id.endswith(":init"):
            self.ledger.start_candidate(candidate_id, "hard")
            self.ledger.finish_candidate(valid=False, forward_calls=0, backward_calls=0,
                                         segment_steps=0, voxel_steps=0)
            return C.CandidateResult(candidate_id, C.content_hash(np.asarray(theta).tolist()),
                                     "hard", None, False, "injected infeasible no-init")
        result = super().evaluate_candidate(theta, candidate_id, objective_kind)
        if objective_kind == "hard" and not candidate_id.endswith(":init"):
            components = dict(result.components)
            components.update(self.replacement)
            result = dataclasses.replace(result, objective=self.replacement_objective,
                                         components=components)
        return result


@pytest.mark.parametrize("mode", ["fixed", "annealed"])
def test_adam_perfect_seed_survives_worse_updates_with_exact_init_cost(mode):
    config = opt.AdamConfig(mode, learning_rate=0.02, k_start=20,
                            k_end=20 if mode == "fixed" else 120, n_steps=5)
    evaluator = DifferentiableObjective()
    result = opt.solve_adam(evaluator, _budget(5, interval=1, max_validation=4),
                            config, x0=np.zeros(3), run_id=f"seed-{mode}")
    assert result.gradient_updates == 4 and result.evaluations == 5
    assert result.best.candidate_id == ("seed-fixed:init" if mode == "fixed" else "seed-annealed:init")
    assert result.best_theta.tolist() == [0.0, 0.0, 0.0]
    assert result.initial_candidate_status == "feasible"
    assert result.no_improvement_over_initialization
    assert result.selection_reason == "no improvement over initialization"
    entries = evaluator.ledger.snapshot()["entries"]
    assert [entry["candidate_id"] for entry in entries].count(f"seed-{mode}:init") == 1
    assert entries[0]["objective_kind"] == "hard"
    assert entries[0]["forward_calls"] == 1
    assert result.initial_candidate_rank is not None


def test_adam_init_only_and_thresholds_are_traceable(tmp_path):
    evaluator = DifferentiableObjective()
    result = opt.solve_adam(evaluator, _budget(1), opt.AdamConfig("fixed", k_start=40, k_end=40),
                            x0=np.zeros(3), run_id="init-only", checkpoint_dir=tmp_path / "checkpoints")
    assert result.evaluations == result.forward_calls == 1
    assert result.gradient_updates == result.validation_calls == 0
    assert result.best == result.initial_candidate
    assert result.selection_thresholds["thresholds_used"]["max_gouge_fraction"] == 0.0
    assert result.selection_thresholds["thresholds_used"]["max_residual_fraction"] == 1.0
    assert result.selection_thresholds["infeasible_ranking"] == "lexicographic"
    document = json.loads((tmp_path / "checkpoints/final.json").read_text())
    assert document["initial_candidate"]["evaluation"]["candidate_id"] == "init-only:init"
    assert document["selection_thresholds"]["config_hash"]
    legacy = dict(document, schema="diffcam-adam-checkpoint/1")
    legacy.pop("initial_candidate")
    legacy.pop("payload_hash")
    legacy["payload_hash"] = C.content_hash(legacy)
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    untouched = DifferentiableObjective()
    with pytest.raises(opt.B.BlackBoxError, match="explicit coordinator migration"):
        opt.resume_adam(tmp_path / "legacy.json", untouched)
    assert untouched.ledger.snapshot()["candidate_evaluations"] == 0

@pytest.mark.parametrize("replacement", [
    {"objective": -0.25},  # strictly better feasible candidate replaces init
    {"objective": 0.0},    # exact hard tie retains init
    {"objective": -100.0, "gouge": 0.2},  # scalar-only selection would incorrectly prefer gouge
])
def test_adam_better_tie_and_gouging_adversary(replacement):
    objective = ControlledHardObjective(**replacement)
    result = opt.solve_adam(
        objective, _budget(2, interval=1, max_validation=1),
        opt.AdamConfig("fixed", learning_rate=0.02, k_start=40, k_end=40),
        x0=np.zeros(3), run_id="controlled-adam",
    )
    assert result.best.candidate_id == ("controlled-adam:init" if replacement.get("gouge", 0.0) > 0
                                         or replacement["objective"] == 0.0 else "controlled-adam:validate:1")
    assert result.no_improvement_over_initialization == (result.best.candidate_id.endswith(":init"))


def test_adam_invalid_init_is_recorded_and_later_valid_candidate_wins():
    evaluator = ControlledHardObjective(invalid_init=True, objective=-0.25)
    result = opt.solve_adam(
        evaluator, _budget(2, interval=1, max_validation=1),
        opt.AdamConfig("fixed", learning_rate=0.02, k_start=40, k_end=40),
        x0=np.zeros(3), run_id="invalid-adam-init",
    )
    assert result.initial_candidate.valid is False
    assert result.initial_candidate_status == "invalid"
    assert result.initial_candidate.invalid_reason == "injected infeasible no-init"
    assert result.initial_candidate_rank is None
    assert result.best.candidate_id == "invalid-adam-init:validate:1"
    assert result.best.objective == -0.25
    assert not result.no_improvement_over_initialization
    assert result.evaluations == 2
    assert result.invalid_evaluations == 1


def test_selection_thresholds_and_both_infeasible_rank_modes():
    def result(candidate_id, holder, gouge, residual, objective):
        return C.CandidateResult(candidate_id, C.content_hash(candidate_id), "hard", objective, True,
                                 components={"holder_overlap_fraction": holder,
                                             "gouge_fraction": gouge, "residual_fraction": residual})

    evaluator = DifferentiableObjective()
    safe = result("safe", 0.0, 0.0, 0.0, 100.0)
    gouging = result("gouging", 0.0, 0.2, 0.0, -100.0)
    assert gouging.objective < safe.objective
    for mode in ("lexicographic", "tolerance_normalised_sum"):
        policy = opt.B._resolve_hard_selection(evaluator, opt.HardSelectionPolicy(infeasible_ranking=mode))
        assert opt._hard_rank(safe, policy) < opt._hard_rank(gouging, policy)
        assert opt._hard_rank(safe, policy)[0] == 0
        assert opt._hard_rank(gouging, policy)[0] == 1
    holder = result("holder", 0.0001, 0.0, 0.0, 0.0)
    residual = result("residual", 0.0, 0.0, 1.1, 0.0)
    policy = opt.B._resolve_hard_selection(evaluator, opt.HardSelectionPolicy())
    assert opt._hard_rank(residual, policy) < opt._hard_rank(holder, policy)
    normalized = opt.B._resolve_hard_selection(
        evaluator, opt.HardSelectionPolicy(infeasible_ranking="tolerance_normalised_sum"))
    assert opt._hard_rank(holder, normalized) < opt._hard_rank(residual, normalized)
    assert opt._hard_rank(residual, normalized)[1] == pytest.approx(0.1)
    assert opt._hard_rank(holder, normalized)[1] == pytest.approx(0.0001)
    assert normalized.zero_tolerance_scale == 1.0
    with pytest.raises(opt.B.BlackBoxError, match="exceeds its task verdict tolerance"):
        opt.solve_adam(evaluator, _budget(1), opt.AdamConfig(
            "fixed", k_start=40, k_end=40,
            selection=opt.HardSelectionPolicy(max_gouge_fraction=0.01)), x0=np.zeros(3))
    with pytest.raises(ValueError, match="holder overlap must be exactly zero"):
        opt.HardSelectionPolicy(max_holder_overlap_fraction=0.001)
    thresholds = opt.B._resolve_hard_selection(
        evaluator, opt.HardSelectionPolicy(max_residual_fraction=0.75, selection_margin=0.1))
    assert thresholds.max_residual_fraction == pytest.approx(0.65)
    assert thresholds.verdict_residual_fraction == pytest.approx(1.0)


def test_fresh_production_adam_resume_and_replay_restore_selection_from_checkpoint(tmp_path):
    initial = opt.solve_adam(
        _prototype(), _budget(2), opt.AdamConfig("fixed", learning_rate=0.001, k_start=40, k_end=40),
        seed=17, x0=np.zeros(6), run_id="fresh-production-adam", checkpoint_dir=tmp_path / "checkpoints",
    )
    checkpoint = tmp_path / "checkpoints/final.json"
    restored_evaluator = _prototype()
    assert np.count_nonzero(restored_evaluator.sim.stock.to_numpy()) == 0
    resumed = opt.resume_adam(
        checkpoint, restored_evaluator, budget=_budget(4), run_id="restored-production-adam",
    )
    assert resumed.initial_candidate == initial.initial_candidate
    assert resumed.initial_candidate_id == "fresh-production-adam:init"
    assert resumed.selection_thresholds == initial.selection_thresholds
    assert resumed.evaluations == 4 and resumed.gradient_updates == 3
    entries = restored_evaluator.ledger.snapshot()["entries"]
    assert len([entry for entry in entries if entry["candidate_id"].endswith(":init")]) == 1
    assert entries[0]["candidate_id"] == "fresh-production-adam:init"
    assert not any(entry["candidate_id"] == "restored-production-adam:init" for entry in entries)
    replay_evaluator = _prototype()
    assert np.count_nonzero(replay_evaluator.sim.stock.to_numpy()) == 0
    replayed = opt.replay_adam_checkpoint(checkpoint, replay_evaluator)
    assert replayed["objective_matches_within_tolerance"]
    assert replay_evaluator.ledger.snapshot()["candidate_evaluations"] == 0
    assert replay_evaluator.ledger.snapshot()["validation_calls"] == 1


def test_legacy_smoke_cli_remains_usable(tmp_path):
    output = tmp_path / "legacy-smoke"
    completed = subprocess.run([
        sys.executable, "-m", "algorithms.optimize", "--cpu", "--steps", "2",
        "--iters", "1", "--resolution", "8", "--seed", "1", "--output", str(output),
    ], cwd=Path.cwd(), capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert (output / "trajectory.npy").is_file()
    assert (output / "rs274.nc").is_file()
    assert (output / "haas.nc").is_file()


class _FixtureArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to_numpy(self):
        return self.value


class ContinuationObjective:
    """Small fully declared evaluator fixture for continuation control flow."""

    def __init__(self, *, theta_dims=3, field_offset=0.0, gradient_size=0.01,
                 hard_mode="flat", rows=1, invalid_init=False):
        self.parameterization = P.DeltaResidual(np.zeros((rows, 3)), scale_mm=1.0,
                                                 bounds=(-3.0, 3.0))
        self.grid = C.GridSpec(shape=(4, 4, 4), origin_mm=(-2.0, -2.0, -2.0), voxel_mm=1.0)
        self.task = SimpleNamespace(
            task_hash=C.content_hash("continuation-test-task"),
            tool_setup_hash=C.content_hash("continuation-test-tool"), grid=self.grid,
            tolerances=C.Tolerances(completion_residual_mm3=1000.0, gouge_volume_mm3=0.0,
                                    local_surface_error_mm=1.0, clearance_mm=0.0),
        )
        assert field_offset >= 0.0
        assert gradient_size >= 0.0
        assert hard_mode in ("flat", "quadratic", "recovery", "invalid_init")
        self.field_offset, self.gradient_size, self.hard_mode = field_offset, gradient_size, hard_mode
        self.invalid_init = invalid_init
        self.spec = {"objective": "continuation-fixture", "field_offset": field_offset,
                     "gradient_size": gradient_size, "hard_mode": hard_mode}
        self.target = np.indices(self.grid.shape, dtype=np.float64)[0] - 1.5
        assert np.any(self.target < 0) and np.any(self.target > 0)
        self.hard_fraction_denominators_mm3 = {"gouge": 1.0, "residual": 1000.0}
        self.ledger = R.ComputeLedger()
        self.seed = 37
        self._evaluation_k_override = None
        self._loss_shift = None
        self.corrupt_soft_k = False
        self.sim = SimpleNamespace(k={None: 20.0}, loss_shift={None: 0.0},
                                   stock=_FixtureArray(self.target), target=_FixtureArray(self.target),
                                   tool_pos=_FixtureArray(np.zeros((rows + 1, 3))))
        self.shifts, self.gradients = [], []

    @property
    def evaluation_loss_shift(self):
        return self._loss_shift

    def set_evaluation_k(self, value):
        self._evaluation_k_override = value
        if value is not None:
            self.sim.k[None] = value

    def clear_evaluation_k(self):
        self._evaluation_k_override = None

    def set_evaluation_loss_shift(self, value):
        self._loss_shift = value
        self.sim.loss_shift[None] = value
        self.shifts.append(value)

    def clear_evaluation_loss_shift(self):
        self._loss_shift = None
        self.sim.loss_shift[None] = 0.0

    def _raw_fields(self, theta, kind):
        # Raw differentiable stock before the occupancy-only shift. The actual
        # stock fields, not scalar losses or an EDT, are supplied to calibration.
        steps = self.parameterization.num_steps
        raw = np.broadcast_to(
            (self.target if kind == "hard" else self.target + self.field_offset),
            (steps + 1, *self.grid.shape)).copy()
        self.sim.stock = _FixtureArray(raw)
        self.sim.tool_pos = _FixtureArray(self.parameterization.positions(theta))

    def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
        if self.corrupt_soft_k and objective_kind == "smooth":
            self._evaluation_k_override = 21.0
        self.ledger.start_candidate(candidate_id, objective_kind)
        self._raw_fields(theta, objective_kind)
        if self.invalid_init and candidate_id.endswith(":init"):
            self.ledger.finish_candidate(valid=False, forward_calls=0, backward_calls=0,
                                         segment_steps=0, voxel_steps=0)
            return C.CandidateResult(candidate_id, C.content_hash(np.asarray(theta).tolist()),
                                     "hard", None, False, "synthetic invalid initialization")
        flat = float(np.sum(np.asarray(theta, dtype=np.float64) ** 2))
        if self.hard_mode == "quadratic":
            hard = flat
        elif self.hard_mode == "recovery":
            hard = 1.0 if flat == 0.0 else 0.0
        else:
            hard = 0.0
        objective = hard if objective_kind == "hard" else float(np.sum(theta) + 3.0)
        gouge = 0.2 if self.hard_mode == "recovery" and objective_kind == "hard" and flat == 0.0 else 0.0
        result = C.CandidateResult(
            candidate_id, C.content_hash(np.asarray(theta).tolist()), objective_kind, objective, True,
            coverage=C.Coverage.INCOMPLETE,
            components={"holder_overlap_fraction": 0.0, "gouge_fraction": gouge,
                        "residual_fraction": 0.0, "loss_shift_voxels": float(self._loss_shift or 0.0)},
            gradient_available=False,
        )
        self.ledger.finish_candidate(valid=True, forward_calls=1, backward_calls=0,
                                     segment_steps=self.parameterization.num_steps, voxel_steps=64)
        return result

    def evaluate_candidate_with_gradient(self, theta, candidate_id):
        result = self.evaluate_candidate(theta, candidate_id, "smooth")
        self.ledger.entries[-1]["backward_calls"] = 1
        self.ledger.backward_calls += 1
        gradient = np.full(theta.size, self.gradient_size + float(self._loss_shift or 0.0))
        self.gradients.append({"candidate_id": candidate_id, "gradient": gradient.copy(),
                               "loss_shift": self._loss_shift, "k": self._evaluation_k_override})
        return O.CandidateEvaluation(dataclasses.replace(result, gradient_available=True), gradient)


def _continuation_config(*, shift="off", backtracking="on", eta=0.25, dmax=0.25,
                         band=None, retries=2, gradient_norm=0.5, fixed=True):
    adam = opt.AdamConfig(
        "fixed" if fixed else "annealed", learning_rate=0.01, k_start=20.0,
        k_end=20.0 if fixed else 120.0, n_steps=3, max_gradient_norm=gradient_norm,
    )
    return opt.ContinuationConfig(
        adam=adam, shift=shift, backtracking=backtracking,
        calibration_eta=eta, calibration_delta_max=dmax,
        calibration_band=band or opt.M.SurfaceBand(
            "stock", 0.0, 2.0, "small real-field unit band in physical millimetres"),
        backtracking_retries=retries,
    )


def test_continuation_off_off_is_bit_identical_to_annealed_adam():
    cfg = _continuation_config(shift="off", backtracking="off", fixed=False)
    theta = np.array([0.1, -0.2, 0.05])
    budget = _budget(4, interval=2, max_validation=1)
    baseline = opt.solve_adam(ContinuationObjective(), budget, cfg.adam,
                              seed=5, x0=theta, run_id="off-off")
    ablation = opt.solve_continuation_adam(
        ContinuationObjective(), theta, _continuation_config(shift="off", backtracking="off", fixed=False),
        budget=budget, seed=5, run_id="off-off")
    np.testing.assert_array_equal(ablation.current_theta, baseline.current_theta)
    np.testing.assert_array_equal(ablation.moments, baseline.moments)
    np.testing.assert_array_equal(ablation.velocities, baseline.velocities)
    np.testing.assert_array_equal(ablation.best_theta, baseline.best_theta)
    assert ablation.history == baseline.history
    assert ablation.best == baseline.best
    assert ablation.evaluations == baseline.evaluations == 4
    assert ablation.validation_calls == baseline.validation_calls == 1
    assert ablation.backward_calls == baseline.backward_calls == 3
    assert isinstance(ablation, opt.AdamResult) and not isinstance(ablation, opt.ContinuationResult)
    assert (ablation.evaluations, ablation.validation_calls) == (4, 1)


def test_continuation_sign_and_zero_gap_are_sourced_from_actual_stock_fields():
    evaluator = ContinuationObjective(field_offset=0.25)
    result = opt.solve_continuation_adam(
        evaluator, np.zeros(3), _continuation_config(shift="on", backtracking="off", eta=1.0),
        budget=_budget(3, interval=1, max_validation=2), run_id="signed-shift")
    assert result.calibration_events[0]["result"]["median_delta"] == pytest.approx(-0.25)
    assert result.loss_shift == pytest.approx(-0.25)
    assert evaluator.evaluation_loss_shift is None
    assert evaluator.sim.loss_shift[None] == 0.0
    refreshed = next(row for row in evaluator.gradients if "after_shift" in row["candidate_id"])
    assert refreshed["loss_shift"] == -0.25
    assert refreshed["gradient"][0] == pytest.approx(-0.24)
    recorded = evaluator.sim.stock.to_numpy()[evaluator.parameterization.num_steps]
    spec = result.calibration_events[0]
    assert spec["result"]["hard_distance_definition"].startswith("forward_hard.stock_d")
    assert spec["result"]["soft_field_definition"].startswith("differentiable forward stock_d")
    assert result.purpose_spent["fresh_incumbent_candidate_calls"] == 1
    metadata = opt.continuation_result_metadata(result)
    assert metadata["condition"] == "continuation-adam"
    assert metadata["continuation_configuration_sha256"] == result.continuation_config_hash
    assert metadata["budget_spent"]["candidate_evaluations"] == result.evaluations
    json.dumps(metadata, allow_nan=False)

    neutral = ContinuationObjective(field_offset=0.0, gradient_size=0.0)
    no_drift = opt.solve_continuation_adam(
        neutral, np.zeros(3), _continuation_config(shift="on", backtracking="on", eta=0.5),
        budget=_budget(3, interval=1, max_validation=2), run_id="zero-gap")
    assert no_drift.calibration_events[0]["result"]["disagreement"]["bounded_gap"] == 0.0
    assert no_drift.loss_shift == 0.0
    np.testing.assert_array_equal(no_drift.current_theta, np.zeros(3))
    assert no_drift.purpose_spent.get("fresh_incumbent_candidate_calls", 0) == 0


def test_continuation_invalid_initialization_can_still_select_a_later_valid_hard_result():
    evaluator = ContinuationObjective(invalid_init=True)
    result = opt.solve_continuation_adam(
        evaluator, np.zeros(3), _continuation_config(),
        budget=_budget(3, interval=1, max_validation=2), run_id="invalid-continuation")
    assert result.initial_candidate.valid is False
    assert result.initial_candidate_status == "invalid"
    assert result.initial_candidate.invalid_reason == "synthetic invalid initialization"
    assert result.initial_candidate.candidate_id == "invalid-continuation:init"
    assert result.best is not None and result.best.valid
    assert not result.best.candidate_id.endswith(":init")
    assert result.purpose_spent["initialization_candidate_calls"] == 1
    assert result.invalid_evaluations == 1


def test_continuation_rollback_restores_theta_moments_and_bills_retries():
    evaluator = ContinuationObjective(hard_mode="quadratic")
    result = opt.solve_continuation_adam(
        evaluator, np.zeros(3), _continuation_config(retries=1),
        budget=_budget(4, interval=1, max_validation=3), run_id="rollback")
    np.testing.assert_array_equal(result.current_theta, np.zeros(3))
    np.testing.assert_array_equal(result.moments, np.zeros(3))
    np.testing.assert_array_equal(result.velocities, np.zeros(3))
    assert result.retry_state == {"learning_rate": 0.01, "proposals": 2, "retries": 1, "accepted": 0}
    assert result.evaluations == 4 and result.validation_calls == 3
    assert result.gradient_updates == 0
    assert evaluator.ledger.snapshot()["candidate_evaluations"] == 4
    assert evaluator.ledger.snapshot()["validation_calls"] == 3
    assert result.purpose_spent["retry_candidate_calls"] == 1
    assert result.purpose_spent["retry_validation_calls"] == 1
    assert result.best.candidate_id == "rollback:init"


def test_continuation_infeasible_start_uses_shared_feasibility_rank():
    evaluator = ContinuationObjective(hard_mode="recovery")
    result = opt.solve_continuation_adam(
        evaluator, np.zeros(3), _continuation_config(retries=0),
        budget=_budget(3, interval=1, max_validation=2), run_id="infeasible")
    assert result.initial_candidate_status == "infeasible"
    assert result.gradient_updates == 1
    assert result.retry_state["accepted"] == 1
    assert result.best.candidate_id != "infeasible:init"
    assert result.best.components["gouge_fraction"] == 0.0


def test_continuation_changed_k_rolls_back_and_cleans_overrides():
    evaluator = ContinuationObjective()
    evaluator.corrupt_soft_k = True
    result = opt.solve_continuation_adam(
        evaluator, np.zeros(3), _continuation_config(), budget=_budget(4, interval=1, max_validation=3),
        run_id="changed-k")
    np.testing.assert_array_equal(result.current_theta, np.zeros(3))
    assert result.gradient_updates == 0
    assert "different k/shift/grid/validator" in result.termination_reason
    assert result.evaluations == 2 and result.validation_calls == 1
    assert evaluator.evaluation_loss_shift is None
    assert evaluator._evaluation_k_override is None


@pytest.mark.parametrize("budget", [
    _budget(2, interval=1, max_validation=2),
    _budget(2, interval=1, max_validation=0),
])
def test_continuation_budget_boundaries_are_atomic(budget):
    evaluator = ContinuationObjective()
    result = opt.solve_continuation_adam(
        evaluator, np.zeros(3), _continuation_config(), budget=budget, run_id="bounded")
    assert result.evaluations <= budget.candidate_evaluations
    assert result.validation_calls <= budget.max_validation_calls
    if budget.max_validation_calls == 0:
        assert result.termination_reason == "hard_validation_budget_exhausted"
        assert result.evaluations == 1
    else:
        assert result.termination_reason == "candidate_budget_exhausted"
        assert result.evaluations == 2 and result.validation_calls == 1
        assert result.gradient_updates == 0
    np.testing.assert_array_equal(result.current_theta, np.zeros(3))
    assert result.best.candidate_id == "bounded:init"


def test_continuation_checkpoint_resume_preserves_ledger_and_never_replays_init(tmp_path):
    evaluator = ContinuationObjective()
    checkpoint = tmp_path / "checkpoints"
    first = opt.solve_continuation_adam(
        evaluator, np.zeros(3), _continuation_config(retries=0),
        budget=_budget(4, interval=1, max_validation=3), run_id="resume-original",
        checkpoint_dir=checkpoint)
    assert first.evaluations == 4 and first.validation_calls == 3
    assert first.termination_reason == "candidate_budget_exhausted"
    document = json.loads((checkpoint / "final.json").read_text())
    assert document["schema"] == "diffcam-adam-checkpoint/2"
    assert document["continuation_state"]["schema"] == "diffcam-continuation-state/1"
    assert document["continuation_state"]["config_sha256"] == first.continuation_config_hash
    assert document["continuation_state"]["observation"]["soft"]["gradient"] is not None
    restored = ContinuationObjective()
    resumed = opt.resume_continuation_adam(
        checkpoint / "final.json", restored,
        budget=_budget(6, interval=1, max_validation=5), run_id="resumed")
    entries = restored.ledger.snapshot()["entries"]
    assert len([row for row in entries if row["candidate_id"] == "resume-original:init"]) == 1
    assert not any(row["candidate_id"] == "resumed:init" for row in entries)
    assert resumed.previous_evaluations == first.evaluations == 4
    assert resumed.evaluations == 6 and resumed.validation_calls == 5
    assert resumed.gradient_updates == 2
    assert resumed.retry_state["accepted"] == 2
    assert resumed.retry_state["proposals"] == 4
    uninterrupted = opt.solve_continuation_adam(
        ContinuationObjective(), np.zeros(3), _continuation_config(retries=0),
        budget=_budget(6, interval=1, max_validation=5), run_id="full-reference")
    np.testing.assert_array_equal(resumed.current_theta, uninterrupted.current_theta)
    np.testing.assert_array_equal(resumed.moments, uninterrupted.moments)
    np.testing.assert_array_equal(resumed.velocities, uninterrupted.velocities)
    np.testing.assert_allclose(resumed.moments, np.full(3, 0.0019), rtol=0, atol=1e-15)
    np.testing.assert_allclose(resumed.velocities, np.full(3, 1.999e-7), rtol=0, atol=1e-18)
    assert resumed.initial_candidate == first.initial_candidate
    assert resumed.selection_thresholds == first.selection_thresholds


def test_continuation_resume_does_not_apply_shift_twice(tmp_path):
    checkpoint = tmp_path / "shift-checkpoint"
    first = opt.solve_continuation_adam(
        ContinuationObjective(field_offset=0.25), np.zeros(3),
        _continuation_config(shift="on", backtracking="off", eta=1.0),
        budget=_budget(2, interval=1, max_validation=1), run_id="shift-resume",
        checkpoint_dir=checkpoint)
    assert first.loss_shift == -0.25
    assert first.termination_reason == "candidate_budget_exhausted"
    assert first.evaluations == 2 and first.validation_calls == 1
    assert len(first.calibration_events) == 1
    document = json.loads((checkpoint / "final.json").read_text())
    assert document["continuation_state"]["loss_shift"] == -0.25
    restored = ContinuationObjective(field_offset=0.25)
    resumed = opt.resume_continuation_adam(
        checkpoint / "final.json", restored,
        budget=_budget(3, interval=1, max_validation=2), run_id="shift-restored")
    assert resumed.loss_shift == -0.25
    assert len(resumed.calibration_events) == 1
    assert resumed.previous_evaluations == 2
    assert resumed.evaluations == 3 and resumed.validation_calls == 2
    assert [row["candidate_id"] for row in restored.ledger.snapshot()["entries"]].count("shift-resume:init") == 1
    assert not any(row["candidate_id"] == "shift-restored:init"
                   for row in restored.ledger.snapshot()["entries"])


def test_continuation_checkpoint_tamper_is_rejected(tmp_path):
    path = tmp_path / "checkpoints"
    opt.solve_continuation_adam(
        ContinuationObjective(), np.zeros(3), _continuation_config(), budget=_budget(2, interval=1, max_validation=1),
        run_id="tamper", checkpoint_dir=path)
    document = json.loads((path / "final.json").read_text())
    document["continuation_state"]["loss_shift"] = 0.25
    (path / "tampered.json").write_text(json.dumps(document))
    with pytest.raises(opt.B.BlackBoxError, match="checksum mismatch"):
        opt.resume_continuation_adam(path / "tampered.json", ContinuationObjective())

    # Even a recomputed outer checksum must not permit an inconsistent paired
    # hard result or restore a misleading historical ledger/ambient field shift.
    mismatch = json.loads((path / "final.json").read_text())
    mismatch["continuation_state"]["hard_result"]["objective"] = 1.0
    mismatch.pop("payload_hash")
    mismatch["payload_hash"] = C.content_hash(mismatch)
    (path / "inconsistent-pair.json").write_text(json.dumps(mismatch))
    untouched = ContinuationObjective()
    with pytest.raises(opt.B.BlackBoxError, match="cached hard objective"):
        opt.resume_continuation_adam(path / "inconsistent-pair.json", untouched)
    assert untouched.ledger.snapshot()["candidate_evaluations"] == 0
    assert untouched.evaluation_loss_shift is None
    assert untouched._evaluation_k_override is None


def test_continuation_real_stock_field_path_and_empty_band_are_safe():
    evaluator = _prototype()
    result = opt.solve_continuation_adam(
        evaluator, np.array([0.001, -0.002, 0.003, -0.004, 0.002, 0.001]),
        _continuation_config(shift="on", backtracking="off"),
        budget=_budget(4, interval=1, max_validation=3), run_id="real-field",
        checkpoint_dir=None)
    assert result.calibration_events
    assert result.loss_shift == result.calibration_events[-1]["result"]["shift"]
    assert abs(result.loss_shift) <= 0.25
    assert any(row["result"]["updated"] for row in result.calibration_events)
    assert evaluator.evaluation_loss_shift is None

    empty = _prototype()
    band = opt.M.SurfaceBand("stock", 1e6, 1e6, "physically empty at every finite target cell")
    narrow = opt.solve_continuation_adam(
        empty, np.zeros(6), _continuation_config(shift="on", backtracking="off", band=band),
        budget=_budget(2, interval=1, max_validation=1, max_generations=1), run_id="inconclusive-band")
    assert narrow.calibration_events[0]["result"]["updated"] is False
    assert narrow.calibration_events[0]["result"]["reason"].startswith("inconclusive:")
    assert narrow.loss_shift == 0.0
    assert narrow.gradient_updates == 1
    assert narrow.purpose_spent["calibration_candidate_calls"] == 1
    assert len(narrow.calibration_events) == 1


def test_continuation_off_off_checkpoint_payloads_match_apart_from_measured_times(tmp_path):
    cfg = _continuation_config(shift="off", backtracking="off", fixed=False)
    theta = np.array([0.1, -0.2, 0.05])
    budget = _budget(4, interval=2, max_validation=1)
    classical_dir = tmp_path / "adam"
    direct = opt.solve_adam(ContinuationObjective(), budget, cfg.adam, seed=5,
                            x0=theta, run_id="off-off-checkpoint", checkpoint_dir=classical_dir)
    ablated_dir = tmp_path / "continuation-adam"
    ablated = opt.solve_continuation_adam(
        ContinuationObjective(), theta, cfg, budget=budget, seed=5,
        run_id="off-off-checkpoint", checkpoint_dir=ablated_dir)
    left = json.loads((classical_dir / "final.json").read_text())
    right = json.loads((ablated_dir / "final.json").read_text())
    for payload in (left, right):
        assert payload["schema"] == "diffcam-adam-checkpoint/2"
        assert "continuation_state" not in payload or payload["continuation_state"] is None
        payload.pop("payload_hash")
        # Actual per-call execution timings are evidence, not optimizer state
        # that should be made byte-identical by inventing deterministic clocks.
        payload.pop("ledger_snapshot")
    assert right == left
    assert classical_dir != ablated_dir
    resumed = opt.resume_continuation_adam(
        ablated_dir / "final.json", ContinuationObjective(), budget=budget,
        run_id="off-off-restored")
    assert resumed.previous_evaluations == 4
    assert resumed.evaluations == budget.candidate_evaluations
    assert resumed.initial_candidate == direct.initial_candidate
    np.testing.assert_array_equal(resumed.current_theta, ablated.current_theta)


def test_continuation_production_checkpoint_restores_fresh_context_and_selection(tmp_path):
    evaluator = _prototype()
    original = opt.solve_continuation_adam(
        evaluator, np.zeros(6), _continuation_config(shift="on", backtracking="off"),
        budget=_budget(3, interval=1, max_validation=2), run_id="production-resume",
        checkpoint_dir=tmp_path / "production-checkpoint")
    assert original.evaluations == 3 and original.validation_calls == 2
    saved = json.loads((tmp_path / "production-checkpoint" / "final.json").read_text())
    assert (saved["continuation_state"]["hard_result"] is None) == (saved["continuation_state"]["observation"] is None)
    restored = _prototype()
    resumed = opt.resume_continuation_adam(
        tmp_path / "production-checkpoint" / "final.json", restored,
        budget=_budget(4, interval=1, max_validation=3), run_id="production-restored")
    assert resumed.previous_evaluations == 3
    assert resumed.evaluations == 4 and resumed.validation_calls <= 3
    assert [row["candidate_id"] for row in restored.ledger.snapshot()["entries"]].count("production-resume:init") == 1
    assert not any(row["candidate_id"] == "production-restored:init"
                   for row in restored.ledger.snapshot()["entries"])
    assert resumed.initial_candidate == original.initial_candidate
    assert resumed.selection_thresholds == original.selection_thresholds
    assert restored.evaluation_loss_shift is None
