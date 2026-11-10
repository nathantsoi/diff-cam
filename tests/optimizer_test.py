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
