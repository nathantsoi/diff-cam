"""T3.2 Adam reference optimizer tests on shared objective and artifact contracts."""

import dataclasses
import json
import runpy
import subprocess
import sys
from pathlib import Path

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
        self.components = components or {"gouge_fraction": 0.0,
                                         "holder_overlap_fraction": 0.0}

    def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
        value = self.parameterization.forward(theta)[0]
        objective = float(np.sum((value - self.target) ** 2))
        self.ledger.start_candidate(candidate_id, objective_kind)
        result = C.CandidateResult(candidate_id, C.content_hash(list(value)), objective_kind,
                                   objective, True,
                                   components=dict(self.components, log_sum_exp=objective,
                                                   residual_fraction=float(np.linalg.norm(value - self.target))))
        self.ledger.finish_candidate(valid=True, forward_calls=1, backward_calls=0,
                                     segment_steps=1, voxel_steps=1)
        return result

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
    policy = opt.HardSelectionPolicy()
    safe_with_residual = result(0.0, 0.0, 0.8, 100.0)
    tiny_limit_violation = result(0.01, 0.0, 0.0, 0.0)
    assert opt._hard_rank(safe_with_residual, policy) < opt._hard_rank(tiny_limit_violation, policy)
    complete_but_colliding = result(0.0, 0.2, 0.0, 0.0)
    assert opt._hard_rank(safe_with_residual, policy) < opt._hard_rank(complete_but_colliding, policy)
    missing_marker = result(0.0, 0.0, 0.1, 0.0)
    del missing_marker.components["holder_overlap_fraction"]
    assert opt._hard_rank(missing_marker, policy)[0] == 1


def test_fixed_adam_known_gradient_step_clip_schedule_and_budget():
    evaluator = DifferentiableObjective()
    config = opt.AdamConfig(mode="fixed", learning_rate=0.1, k_start=40, k_end=40,
                            max_gradient_norm=0.01)
    theta = np.zeros(3)
    result = opt.solve_adam(evaluator, _budget(4, interval=2, max_validation=1), config,
                            seed=7, x0=theta, run_id="fixed-check")
    assert result.history[0]["k"] == result.history[-1]["k"] == 40
    assert result.history[0]["clip_scale"] < 1.0
    assert result.evaluations == 4 and result.validation_calls == 1
    assert result.forward_calls == 5 and result.backward_calls == 4
    assert result.gradient_updates == 4
    assert result.best.valid and result.best_theta.shape == (3,)
    assert result.history[-1]["soft_objective"] < result.history[0]["soft_objective"]
    ledger = evaluator.ledger.snapshot()
    # Four gradient evaluations plus one independent hard validation call.
    assert ledger["candidate_evaluations"] == 5 and ledger["validation_calls"] == 1


def test_fixed_and_annealed_schedule_is_predeclared_and_generation_limited():
    evaluator = DifferentiableObjective()
    config = opt.AdamConfig(mode="annealed", learning_rate=0.001,
                            k_start=20, k_end=120, max_gradient_norm=0.5)
    result = opt.solve_adam(evaluator, _budget(60, max_generations=2), config,
                            seed=1, x0=np.zeros(3), run_id="annealed")
    assert [row["k"] for row in result.history] == [20.0, 120.0]
    assert result.termination_reason == "max_generations"
    assert result.evaluations == 2
    with pytest.raises(ValueError, match="k_start == k_end"):
        opt.AdamConfig(mode="fixed", k_start=10, k_end=20)


def _checkpoint_pair(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    initial = opt.solve_adam(
        DifferentiableObjective(), _budget(2, interval=2, max_validation=1),
        opt.AdamConfig("annealed", learning_rate=0.02, k_start=20, k_end=120,
                       max_gradient_norm=0.5), seed=5,
        x0=np.array([1.0, 0.5, -0.5]), run_id="replay",
        checkpoint_dir=checkpoint_dir,
    )
    assert initial.best is not None
    return initial, checkpoint_dir


def test_checkpoint_resume_matches_full_run_and_schedule_state(tmp_path):
    initial, directory = _checkpoint_pair(tmp_path)
    checkpoint = directory / "ckpt-000002.json"
    assert checkpoint.is_file()
    resumed = opt.resume_adam(
        checkpoint, DifferentiableObjective(), budget=_budget(5, interval=2, max_validation=2),
        run_id="resumed",
    )
    fresh = opt.solve_adam(
        DifferentiableObjective(), _budget(5, interval=2, max_validation=2),
        initial.config, seed=5, x0=np.array([1.0, 0.5, -0.5]), run_id="fresh",
    )
    np.testing.assert_array_equal(resumed.current_theta, fresh.current_theta)
    np.testing.assert_array_equal(resumed.moments, fresh.moments)
    np.testing.assert_array_equal(resumed.velocities, fresh.velocities)
    assert resumed.evaluations == fresh.evaluations == 5
    assert resumed.previous_evaluations == 2
    assert resumed.gradient_updates == fresh.gradient_updates
    assert resumed.forward_calls == fresh.forward_calls == 7
    assert resumed.backward_calls == fresh.backward_calls == 5
    assert resumed.validation_calls == fresh.validation_calls == 2
    assert resumed.best.objective == fresh.best.objective
    replayed = opt.replay_adam_checkpoint(checkpoint, DifferentiableObjective())
    assert replayed["objective_matches_backend_equality"]
    assert replayed["replayed"].objective == initial.best.objective


def test_adam_checkpoint_rejects_checksum_and_task_changes(tmp_path):
    _, directory = _checkpoint_pair(tmp_path)
    checkpoint = directory / "ckpt-000002.json"
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
    result = opt.solve_adam(evaluator, _budget(4, interval=2, max_validation=2), config,
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
    assert payload["budgets"]["limits"]["candidate_evaluations"] == 4
    assert payload["budgets"]["spent"]["backward_calls"] == 4


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
