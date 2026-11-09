"""CMA-ES optimizer tests: cheap math first, analytic simulator integration second."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

cma = pytest.importorskip("cma", reason="run with uv run --extra optim for the pinned CMA dependency")
assert cma.__version__ == "4.5.0"

from algorithms import blackbox as B
from algorithms import run_artifacts as R
from benchmarks import audit as A
from benchmarks import contracts as C


class QuadraticMap:
    """Tiny adapter with the evaluator-needed map fields, without a simulator."""

    num_params = 2
    num_steps = 1
    scale_mm = 1.0
    mode = "quadratic_test_map"
    bounds = (np.full(2, -3.0), np.full(2, 3.0))
    initial_position_mm = np.zeros(3)

    def in_bounds(self, theta):
        theta = np.asarray(theta, dtype=float)
        return theta.shape == (2,) and np.isfinite(theta).all() and np.all(np.abs(theta) <= 3.0)

    def forward(self, theta):
        if not self.in_bounds(theta):
            raise ValueError("test theta out of bounds or non-finite")
        return np.asarray(theta, dtype=float).copy()


class QuadraticObjective:
    """Pure-Python two-dimensional test objective with an independently known minimum."""

    def __init__(self, target=(0.3, -0.2)):
        self.parameterization = QuadraticMap()
        self.target = np.asarray(target, dtype=float)
        self.spec = {"normalization": "squared_distance", "holder_proxy": False}
        self.ledger = R.ComputeLedger()

    def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
        self.ledger.start_candidate(candidate_id, objective_kind)
        deltas = self.parameterization.forward(theta)
        objective = float(np.sum((deltas - self.target) ** 2))
        self.ledger.finish_candidate(valid=True, forward_calls=1, backward_calls=0,
                                     segment_steps=1, voxel_steps=1)
        return C.CandidateResult(
            candidate_id=candidate_id, theta_hash=C.content_hash(list(deltas)),
            objective_kind=objective_kind, objective=objective, valid=True,
            components={"test_quadratic": objective},
        )


def _budget(count, population=4, **changes):
    return B.OptimizationBudget(candidate_evaluations=count, population_size=population,
                                validation_interval=changes.pop("validation_interval", 0),
                                max_validation_calls=changes.pop("max_validation_calls", 0), **changes)


def test_signed_scale_is_finite_and_invalid_policy_is_explicit():
    assert B.optimizer_invalid_penalty() > 0
    objective = QuadraticObjective()
    theta = np.zeros(2)
    result = objective.evaluate_candidate(theta, "known-point")
    assert result.objective == pytest.approx(0.3 ** 2 + 0.2 ** 2, abs=1e-12)


def test_quadratic_known_optimum_and_seed_repeatability():
    kwargs = dict(seed=7, sigma0=0.5, x0=[2.0, -2.0], run_id="quadratic")
    first = B.solve_cma_es(QuadraticObjective(), _budget(120, 6), **kwargs)
    repeat = B.solve_cma_es(QuadraticObjective(), _budget(120, 6), **kwargs)
    assert first.best.valid
    assert first.best.objective < 0.01  # no solver comparison, just convergence here.
    assert np.linalg.norm(first.best_theta - np.array([0.3, -0.2])) < 0.1
    np.testing.assert_array_equal(first.best_theta, repeat.best_theta)
    assert first.best.objective == repeat.best.objective
    assert first.evaluations == repeat.evaluations == 120
    assert first.history[-1]["best_theta"] == repeat.history[-1]["best_theta"]


def test_small_zero_seed_is_deterministic():
    one = B.solve_cma_es(QuadraticObjective(), _budget(40, 5), seed=0, x0=[2.0, -1.0])
    two = B.solve_cma_es(QuadraticObjective(), _budget(40, 5), seed=0, x0=[2.0, -1.0])
    np.testing.assert_array_equal(one.best_theta, two.best_theta)
    assert one.seed == 0

    different = B.solve_cma_es(QuadraticObjective(), _budget(40, 5), seed=1, x0=[1.0, -2.0])
    # Independent scenario, not a demand for superiority of a different seed.
    assert different.seed == 1 and different.best.valid


def test_budget_exhaustion_full_and_fractional_population_and_generation_cap():
    full = B.solve_cma_es(QuadraticObjective(), _budget(8, 4), seed=1, x0=[2.0, -1.0])
    assert full.evaluations == 8 and full.generations == 2
    assert full.termination_reason == "budget_exhausted"

    under_one_population = B.solve_cma_es(QuadraticObjective(), _budget(3, 4), seed=1)
    assert under_one_population.evaluations == 0
    assert under_one_population.generations == 0
    assert under_one_population.termination_reason == "budget_exhausted"

    capped = B.solve_cma_es(QuadraticObjective(), _budget(100, 4, max_generations=2), seed=1, x0=[2, -2])
    assert capped.generations == 2 and capped.evaluations == 8
    assert capped.termination_reason == "max_generations"
    with pytest.raises(ValueError, match="non-negative"):
        B.OptimizationBudget(candidate_evaluations=-1)


def test_invalid_candidates_are_counted_and_penalized_not_resampled():
    class Invalid(QuadraticObjective):
        def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
            return C.CandidateResult(candidate_id, C.content_hash("bad"), objective_kind,
                                     None, False, "invalid candidate by construction")
    count = 4
    result = B.solve_cma_es(Invalid(), _budget(count, 4), seed=2)
    assert result.best is None
    assert result.invalid_candidates == count
    assert result.history[0]["invalid_members"] == count
    assert result.evaluations == count
    assert result.best_theta is None


def test_checkpoint_resume_is_exact_and_counts_prior_budget(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    initial = B.solve_cma_es(QuadraticObjective(), _budget(4, 4), seed=4,
                             sigma0=0.7, x0=[2.0, -1.5], run_id="resumable",
                             checkpoint_dir=checkpoint_dir)
    assert initial.evaluations == 4
    checkpoint = checkpoint_dir / "ckpt-000004.json"
    assert checkpoint.is_file()
    document = json.loads(checkpoint.read_text())
    assert document["schema"] == "diffcam-cmaes-checkpoint/1"

    resumed = B.resume_cma_es(checkpoint, QuadraticObjective(), budget=_budget(12, 4), run_id="resumed")
    fresh = B.solve_cma_es(QuadraticObjective(), _budget(12, 4), seed=4,
                           sigma0=0.7, x0=[2.0, -1.5], run_id="full")
    np.testing.assert_array_equal(resumed.best_theta, fresh.best_theta)
    assert resumed.best.objective == fresh.best.objective
    assert resumed.evaluations == 12
    assert resumed.previous_evaluations == 4
    assert resumed.resumed_from.startswith("sha256:")
    assert resumed.history[-1]["best_theta"] == fresh.history[-1]["best_theta"]

    # Resuming with no additional budget does not issue another population.
    replay = B.resume_cma_es(checkpoint, QuadraticObjective(), budget=_budget(4, 4))
    assert replay.evaluations == 4
    assert replay.termination_reason == "budget_exhausted"


def test_checkpoint_rejects_checksum_and_setup_changes(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    B.solve_cma_es(QuadraticObjective(), _budget(4, 4), seed=2, run_id="identified",
                   checkpoint_dir=checkpoint_dir)
    checkpoint = checkpoint_dir / "ckpt-000004.json"
    raw = json.loads(checkpoint.read_text())
    raw["payload_hash"] = "sha256:" + "0" * 64
    checkpoint.write_text(json.dumps(raw))
    with pytest.raises(B.BlackBoxError, match="checksum"):
        B.resume_cma_es(checkpoint, QuadraticObjective())

    raw = json.loads(checkpoint.read_text())
    raw["setup_hash"] = "sha256:" + "0" * 64
    payload = dict(raw)
    payload.pop("payload_hash", None)
    raw["payload_hash"] = C.content_hash(payload)
    checkpoint.write_text(json.dumps(raw))
    with pytest.raises(B.BlackBoxError, match="identity differs"):
        B.resume_cma_es(checkpoint, QuadraticObjective())


def test_validation_cadence_and_ledger_are_distinct(tmp_path):
    evaluator = QuadraticObjective()
    budget = B.OptimizationBudget(8, population_size=4, validation_interval=2, max_validation_calls=1)
    result = B.solve_cma_es(evaluator, budget, seed=2, x0=[2, -1], run_id="validated")
    assert result.evaluations == 8 and result.validation_calls == 1
    ledger = evaluator.ledger.snapshot()
    assert ledger["candidate_evaluations"] == 9
    assert ledger["validation_calls"] == 1
    assert result.best.valid


def _prototype():
    helpers = runpy.run_path(Path(__file__).with_name("objectives_test.py"))
    return helpers["_prototype"](seed=11)


def test_analytic_hard_smooth_shared_objective_and_incomplete_posted_artifacts(tmp_path):
    smooth_evaluator = _prototype()
    budget = _budget(4, 2, validation_interval=2, max_validation_calls=1)
    smooth = B.solve_cma_es(smooth_evaluator, budget, seed=5, objective_kind="smooth", run_id="smooth")
    assert smooth.best.valid and smooth.best_theta.shape == (6,)
    assert smooth.validation_calls == 1
    assert smooth_evaluator.ledger.snapshot()["validation_calls"] == 1
    assert smooth_evaluator.ledger.snapshot()["candidate_evaluations"] == 5

    hard_evaluator = _prototype()
    hard = B.solve_cma_es(hard_evaluator, budget, seed=5, objective_kind="hard", run_id="hard")
    assert hard.best.valid and hard.best.objective >= 0
    assert hard.best.components["holder_overlap_fraction"] >= 0
    # Shared parameterization shape/bounds; no assertion of superiority, convergence,
    # Adam comparison, or physical solver ranking is present in these tests.
    assert hard_evaluator.task.tool == smooth_evaluator.task.tool

    record = B.emit_optimization_artifacts(smooth, smooth_evaluator, tmp_path / "smooth", Path.cwd())
    assert record.status == C.RunStatus.INCOMPLETE
    assert "posted evaluation" in record.failure_reason.lower()
    run_dir = tmp_path / "smooth"
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "rs274.nc").is_file()
    assert not (run_dir / "eval_posted_rs274.json").exists()
    assert C.loads((run_dir / "run.json").read_text()) == record
    np.testing.assert_allclose(np.load(run_dir / "trajectory.npy").shape, (3, 3))

    manifest = A.load_manifest({
        "schema_version": 1,
        "tasks": [{"task_id": smooth_evaluator.task.task_id, "family": smooth_evaluator.task.family,
                   "task_hash": smooth_evaluator.task.task_hash}],
        "expected_runs": [{"run_id": "smooth", "task_id": smooth_evaluator.task.task_id,
                           "method": "candidate", "seed": smooth.seed}],
    })
    reports, audit = A.audit_runs(tmp_path / "smooth", manifest)
    # Valid artifacts honestly show why this is not scientific end-to-end evidence.
    assert reports[0]["status"] == "incomplete"
    assert audit["summary"]["evidence_complete"] is False
