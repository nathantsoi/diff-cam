"""CMA-ES optimizer tests: cheap math first, analytic simulator integration second."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

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
        self.task = SimpleNamespace(task_hash=C.content_hash("quadratic-test-task"),
                                    tool_setup_hash=C.content_hash("quadratic-test-tool"),
                                    tolerances=C.Tolerances(
            completion_residual_mm3=1000.0, gouge_volume_mm3=0.0,
            local_surface_error_mm=1.0, clearance_mm=0.0))
        self.hard_fraction_denominators_mm3 = {"gouge": 1.0, "residual": 1000.0}

    def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
        self.ledger.start_candidate(candidate_id, objective_kind)
        deltas = self.parameterization.forward(theta)
        objective = self.hard_objective(theta) if objective_kind == "hard" else float(
            np.sum((deltas - self.target) ** 2))
        self.ledger.finish_candidate(valid=True, forward_calls=1, backward_calls=0,
                                     segment_steps=1, voxel_steps=1)
        components = {"test_quadratic": objective}
        if objective_kind == "hard":
            components.update(gouge_fraction=0.0, residual_fraction=0.0,
                              holder_overlap_fraction=0.0)
        return C.CandidateResult(
            candidate_id=candidate_id, theta_hash=C.content_hash(np.asarray(theta).tolist()),
            objective_kind=objective_kind, objective=float(objective), valid=True,
            components=components,
        )

    def hard_objective(self, theta):
        # Smooth CMA's fitness points away from this perfect hard seed; hard
        # CMA cannot improve the initial mean by drawing nonzero perturbations.
        return float(np.sum(np.asarray(theta, dtype=float) ** 2))


def _budget(count, population=4, **changes):
    return B.OptimizationBudget(candidate_evaluations=count, population_size=population,
                                validation_interval=changes.pop("validation_interval", 0),
                                max_validation_calls=changes.pop("max_validation_calls", 0), **changes)


def test_hard_cma_tell_uses_feasible_rank_not_scalar_and_smooth_is_unchanged(monkeypatch):
    class StrategyProbe:
        opts = {}

        def __init__(self):
            self.told = []

        def stop(self):
            return False

        def ask(self):
            return [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])]

        def tell(self, candidates, fitness):
            self.told.append(([candidate.tolist() for candidate in candidates], list(fitness)))

    class Adversary(QuadraticObjective):
        def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
            if objective_kind != "hard" or candidate_id.endswith(":init"):
                return super().evaluate_candidate(theta, candidate_id, objective_kind)
            theta = np.asarray(theta, dtype=float)
            self.ledger.start_candidate(candidate_id, objective_kind)
            if theta[0] >= 1.5:
                self.ledger.finish_candidate(valid=False, forward_calls=0, backward_calls=0,
                                             segment_steps=0, voxel_steps=0)
                return C.CandidateResult(candidate_id, C.content_hash(theta.tolist()), "hard",
                                         None, False, "injected invalid member")
            feasible, objective = theta[0] < 0.5, 100.0 if theta[0] < 0.5 else -100.0
            self.ledger.finish_candidate(valid=True, forward_calls=1, backward_calls=0,
                                         segment_steps=1, voxel_steps=1)
            return C.CandidateResult(
                candidate_id, C.content_hash(theta.tolist()), "hard", objective, True,
                components={"holder_overlap_fraction": 0.0,
                            "gouge_fraction": 0.0 if feasible else 0.2,
                            "residual_fraction": 0.0})

    probe = StrategyProbe()
    monkeypatch.setattr(B, "_new_es", lambda *args: probe)
    result = B.solve_cma_es(Adversary(), _budget(4, 3), objective_kind="hard",
                            x0=np.zeros(2), run_id="rank-steering")
    assert probe.told == [([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                           [0.0, 1.0, B.optimizer_invalid_penalty()])]
    assert result.history[0]["fitness_semantics"] == "hard_rank_ordinal_within_population"
    assert result.history[0]["best_fitness"] == -100.0  # Raw objective is retained diagnostically.
    assert result.history[0]["invalid_members"] == 1

    smooth_probe = StrategyProbe()
    monkeypatch.setattr(B, "_new_es", lambda *args: smooth_probe)
    B.solve_cma_es(QuadraticObjective(), _budget(4, 3), objective_kind="smooth",
                   x0=np.zeros(2), run_id="smooth-tell-unchanged")
    assert smooth_probe.told[0][1] == pytest.approx([0.13, 1.93, 7.73])


def test_signed_scale_is_finite_and_invalid_policy_is_explicit():
    assert B.optimizer_invalid_penalty() > 0
    objective = QuadraticObjective()
    theta = np.zeros(2)
    result = objective.evaluate_candidate(theta, "known-point")
    assert result.objective == pytest.approx(0.3 ** 2 + 0.2 ** 2, abs=1e-12)


def test_quadratic_known_optimum_and_seed_repeatability():
    kwargs = dict(seed=7, sigma0=0.5, x0=[2.0, -2.0], run_id="quadratic")
    first = B.solve_cma_es(QuadraticObjective(), _budget(121, 6), **kwargs)
    repeat = B.solve_cma_es(QuadraticObjective(), _budget(121, 6), **kwargs)
    assert first.best.valid
    assert first.history[-1]["best_fitness"] < 0.01  # smooth search converges
    assert first.best.candidate_id == "quadratic:init"  # hard selection preserves perfect seed
    assert first.best.objective == 8.0
    np.testing.assert_array_equal(first.best_theta, np.array([2.0, -2.0]))
    np.testing.assert_array_equal(first.best_theta, repeat.best_theta)
    assert first.best.objective == repeat.best.objective
    assert first.evaluations == repeat.evaluations == 121
    assert first.history[-1]["best_theta"] == repeat.history[-1]["best_theta"]


def test_small_zero_seed_is_deterministic():
    one = B.solve_cma_es(QuadraticObjective(), _budget(41, 5), seed=0, x0=[2.0, -1.0])
    two = B.solve_cma_es(QuadraticObjective(), _budget(41, 5), seed=0, x0=[2.0, -1.0])
    np.testing.assert_array_equal(one.best_theta, two.best_theta)
    assert one.seed == 0

    different = B.solve_cma_es(QuadraticObjective(), _budget(41, 5), seed=1, x0=[1.0, -2.0])
    # Independent scenario, not a demand for superiority of a different seed.
    assert different.seed == 1 and different.best.valid


def test_budget_exhaustion_full_and_fractional_population_and_generation_cap():
    full = B.solve_cma_es(QuadraticObjective(), _budget(9, 4), seed=1, x0=[2.0, -1.0])
    assert full.evaluations == 9 and full.generations == 2
    assert full.termination_reason == "budget_exhausted"

    # Review Q4: a budget smaller than one population is rejected up front instead of ending silently.
    with pytest.raises(ValueError, match="population_size \\+ 1"):
        _budget(3, 4)

    capped = B.solve_cma_es(QuadraticObjective(), _budget(101, 4, max_generations=2), seed=1, x0=[2, -2])
    assert capped.generations == 2 and capped.evaluations == 9
    assert capped.termination_reason == "max_generations"
    with pytest.raises(ValueError, match="non-negative"):
        B.OptimizationBudget(candidate_evaluations=-1)


def test_invalid_candidates_are_counted_and_penalized_not_resampled():
    class Invalid(QuadraticObjective):
        def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
            self.ledger.start_candidate(candidate_id, objective_kind)
            self.ledger.finish_candidate(valid=False, forward_calls=0, backward_calls=0,
                                         segment_steps=0, voxel_steps=0)
            return C.CandidateResult(candidate_id, C.content_hash("bad"), objective_kind,
                                     None, False, "invalid candidate by construction")
    count = 4
    result = B.solve_cma_es(Invalid(), _budget(count + 1, 4), seed=2)
    assert result.best is None
    assert result.invalid_candidates == count + 1
    assert result.history[0]["invalid_members"] == count
    assert result.evaluations == count + 1
    assert result.best_theta is None
    assert result.initial_candidate_status == "invalid"
    assert result.initial_candidate.candidate_id.endswith(":init")


def test_checkpoint_resume_is_exact_and_counts_prior_budget(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    initial = B.solve_cma_es(QuadraticObjective(), _budget(5, 4), seed=4,
                             sigma0=0.7, x0=[2.0, -1.5], run_id="resumable",
                             checkpoint_dir=checkpoint_dir)
    assert initial.evaluations == 5
    checkpoint = checkpoint_dir / "ckpt-000005.json"
    assert checkpoint.is_file()
    document = json.loads(checkpoint.read_text())
    assert document["schema"] == "diffcam-cmaes-checkpoint/2"

    resumed = B.resume_cma_es(checkpoint, QuadraticObjective(), budget=_budget(13, 4), run_id="resumed")
    fresh = B.solve_cma_es(QuadraticObjective(), _budget(13, 4), seed=4,
                           sigma0=0.7, x0=[2.0, -1.5], run_id="full")
    np.testing.assert_array_equal(resumed.best_theta, fresh.best_theta)
    assert resumed.best.objective == fresh.best.objective
    assert resumed.evaluations == 13
    assert resumed.previous_evaluations == 5
    assert resumed.resumed_from.startswith("sha256:")
    assert resumed.history[-1]["best_theta"] == fresh.history[-1]["best_theta"]

    # Resuming with no additional budget does not issue another population.
    replay = B.resume_cma_es(checkpoint, QuadraticObjective(), budget=_budget(5, 4))
    assert replay.evaluations == 5
    assert replay.termination_reason == "budget_exhausted"


def test_checkpoint_rejects_checksum_and_setup_changes(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    B.solve_cma_es(QuadraticObjective(), _budget(5, 4), seed=2, run_id="identified",
                   checkpoint_dir=checkpoint_dir)
    checkpoint = checkpoint_dir / "ckpt-000005.json"
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
    budget = B.OptimizationBudget(9, population_size=4, validation_interval=2, max_validation_calls=1)
    result = B.solve_cma_es(evaluator, budget, seed=2, x0=[2, -1], run_id="validated")
    assert result.evaluations == 9 and result.validation_calls == 1
    ledger = evaluator.ledger.snapshot()
    assert ledger["candidate_evaluations"] == 9
    assert ledger["validation_calls"] == 1
    assert result.best.valid


def _prototype():
    helpers = runpy.run_path(Path(__file__).with_name("objectives_test.py"))
    return helpers["_prototype"](seed=11)


def test_analytic_hard_smooth_shared_objective_and_incomplete_posted_artifacts(tmp_path):
    smooth_evaluator = _prototype()
    budget = _budget(5, 2, validation_interval=2, max_validation_calls=1)
    smooth = B.solve_cma_es(smooth_evaluator, budget, seed=5, objective_kind="smooth", run_id="smooth")
    assert smooth.best.valid and smooth.best_theta.shape == (6,)
    assert smooth.validation_calls == 1
    assert smooth_evaluator.ledger.snapshot()["validation_calls"] == 1
    assert smooth_evaluator.ledger.snapshot()["candidate_evaluations"] == 5
    assert smooth.forward_calls == 6 and smooth.backward_calls == 0

    hard_evaluator = _prototype()
    hard = B.solve_cma_es(hard_evaluator, budget, seed=5, objective_kind="hard", run_id="hard")
    assert hard.best.valid and hard.best.objective >= 0
    assert hard.best.components["holder_overlap_fraction"] >= 0
    # Shared parameterization shape/bounds; no assertion of superiority, convergence,
    # Adam comparison, or physical solver ranking is present in these tests.
    assert hard_evaluator.task.tool == smooth_evaluator.task.tool

    record = B.emit_optimization_artifacts(smooth, smooth_evaluator, tmp_path / "smooth", Path.cwd())
    assert record.status == C.RunStatus.INCOMPLETE
    config = json.loads((tmp_path / "smooth" / "config.json").read_text())
    assert config["budgets"]["limits"]["candidate_evaluations"] == 5
    assert config["budgets"]["limits"]["max_validation_calls"] == 1
    assert config["budgets"]["spent"] == {"candidate_evaluations": 5,
                                          "validation_calls": 1, "generations": 2}
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


def test_annealed_adam_does_not_leak_smoothing_k_into_cma():
    from algorithms import optimize as adam
    shared = _prototype()
    adam.solve_adam(
        shared, B.OptimizationBudget(3, population_size=None),
        adam.AdamConfig(mode="annealed", learning_rate=0.001,
                        k_start=10, k_end=100, n_steps=2),
        seed=4, x0=np.zeros(6), run_id="anneal-before-cma",
    )
    budget = _budget(3, 2)
    following = B.solve_cma_es(shared, budget, seed=4, objective_kind="smooth", x0=np.zeros(6))
    fresh = B.solve_cma_es(_prototype(), budget, seed=4, objective_kind="smooth", x0=np.zeros(6))
    # Taichi CPU atomic reductions are not bitwise reproducible across runs (observed 1 float32 ulp, 6e-8,
    # in another worker's environment). A k leak (10 -> 100 vs 40) changes the smooth objective by orders of
    # magnitude more than this tolerance, so the regression is still detected.
    for got, want in zip(following.history[0]["fitness"], fresh.history[0]["fitness"]):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-9)
    assert shared.spec.k == 40.0


def test_final_checkpoint_does_not_advance_dropped_population(tmp_path):
    full = B.solve_cma_es(QuadraticObjective(), _budget(41, 4), seed=3, sigma0=0.5, x0=[2, -2])
    B.solve_cma_es(QuadraticObjective(), _budget(23, 4), seed=3, sigma0=0.5, x0=[2, -2],
                   checkpoint_dir=tmp_path / "checkpoints")
    final = tmp_path / "checkpoints/final.json"
    document = json.loads(final.read_text())
    assert document["state"]["evaluations"] == 21
    import pickle, base64
    es = pickle.loads(base64.b64decode(document["state"]["es_pickle_b64"]))
    assert es.countevals == 20
    resumed = B.resume_cma_es(final, QuadraticObjective(), budget=_budget(41, 4))
    np.testing.assert_array_equal(resumed.best_theta, full.best_theta)
    assert resumed.best.objective == full.best.objective


def test_cma_validation_is_separate_and_ledger_restores(tmp_path):
    evaluator = QuadraticObjective()
    checkpoint_dir = tmp_path / "cma-ledger"
    result = B.solve_cma_es(evaluator, _budget(9, 4, validation_interval=2, max_validation_calls=2),
                            seed=9, run_id="cma-tags", checkpoint_dir=checkpoint_dir)
    ledger = evaluator.ledger.snapshot()
    assert ledger["candidate_evaluations"] == 9
    assert ledger["validation_calls"] == 2
    assert ledger["forward_calls"] == 11
    assert ledger["backward_calls"] == 0
    assert result.forward_calls == 11 and result.backward_calls == 0
    assert sum(bool(entry.get("validation", False)) for entry in ledger["entries"]) == 2
    restored_evaluator = QuadraticObjective()
    B.resume_cma_es(checkpoint_dir / "final.json", restored_evaluator,
                    budget=_budget(9, 4, validation_interval=2, max_validation_calls=2))

    restored = restored_evaluator.ledger.snapshot()
    for field in ("candidate_evaluations", "validation_calls", "forward_calls", "backward_calls",
                  "segment_steps", "voxel_steps"):
        assert restored[field] == ledger[field]


class ControlledCmaObjective(QuadraticObjective):
    def __init__(self, *, objective=0.0, gouge=0.0, invalid_init=False):
        super().__init__()
        self.replacement_objective = objective
        self.replacement_gouge = gouge
        self.invalid_init = invalid_init
        self.hard_thetas = {}

    def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
        if objective_kind == "hard":
            self.hard_thetas[candidate_id] = np.asarray(theta, dtype=float).copy()
            self.ledger.start_candidate(candidate_id, objective_kind)
            valid = not (self.invalid_init and candidate_id.endswith(":init"))
            self.ledger.finish_candidate(valid=valid, forward_calls=1, backward_calls=0,
                                         segment_steps=1, voxel_steps=1)
            if not valid:
                return C.CandidateResult(candidate_id, C.content_hash(np.asarray(theta).tolist()),
                                         "hard", None, False, "injected invalid CMA initialization")
            components = {"gouge_fraction": self.replacement_gouge if not candidate_id.endswith(":init") else 0.0,
                          "residual_fraction": 0.0, "holder_overlap_fraction": 0.0}
            objective = 0.0 if candidate_id.endswith(":init") else self.replacement_objective
            return C.CandidateResult(candidate_id, C.content_hash(np.asarray(theta).tolist()),
                                     "hard", objective, True, components=components)
        return super().evaluate_candidate(theta, candidate_id, objective_kind)


@pytest.mark.parametrize("objective_kind", ["hard", "smooth"])
@pytest.mark.parametrize("replacement", [
    {"objective": -0.5},                      # improve
    {"objective": 0.0},                       # tie to init
    {"objective": -100.0, "gouge": 0.2},      # scalar-only selection would gouge
])
def test_cma_hard_init_better_tie_and_gouging_selection(objective_kind, replacement):
    evaluator = ControlledCmaObjective(**replacement)
    result = B.solve_cma_es(evaluator, _budget(5, 4, validation_interval=2, max_validation_calls=1),
                            seed=3, objective_kind=objective_kind, run_id=f"controlled-{objective_kind}")
    improve = replacement["objective"] < 0 and not replacement.get("gouge")
    if improve:
        assert result.best.candidate_id != f"controlled-{objective_kind}:init"
        assert result.best.objective == -0.5
        np.testing.assert_array_equal(result.best_theta, evaluator.hard_thetas[result.best.candidate_id])
        if objective_kind == "smooth":
            assert not np.array_equal(result.best_theta, np.zeros(2))
            assert result.best.candidate_id.endswith(":validation:1")
    else:
        assert result.best.candidate_id == f"controlled-{objective_kind}:init"
        assert result.no_improvement_over_initialization
        assert result.selection_reason == "no improvement over initialization"
    assert result.initial_candidate_status == "feasible"
    assert evaluator.ledger.snapshot()["candidate_evaluations"] == 5
    assert len([entry for entry in evaluator.ledger.snapshot()["entries"]
                if entry["candidate_id"].endswith(":init")]) == 1


@pytest.mark.parametrize("objective_kind", ["hard", "smooth"])
def test_cma_invalid_init_is_explicit_and_does_not_block_valid_members(objective_kind):
    evaluator = ControlledCmaObjective(invalid_init=True, objective=0.25)
    result = B.solve_cma_es(evaluator, _budget(5, 4, validation_interval=2, max_validation_calls=1),
                            seed=2, objective_kind=objective_kind, x0=[0.0, 0.0], run_id="invalid-cma")
    assert result.initial_candidate_status == "invalid"
    assert result.initial_candidate.valid is False
    assert result.best is not None and result.best.valid
    assert result.best.candidate_id != "invalid-cma:init"
    assert result.best.objective == 0.25
    assert not result.no_improvement_over_initialization


def test_cma_init_only_resume_preserves_theta_and_ledger(tmp_path):
    seed = [1.0, -1.0]
    first_evaluator = QuadraticObjective()
    first = B.solve_cma_es(first_evaluator, _budget(1, 4), objective_kind="smooth",
                           seed=7, x0=seed, run_id="initial-origin", checkpoint_dir=tmp_path / "checkpoints")
    assert first.evaluations == first.forward_calls == 1
    assert first.generations == first.validation_calls == 0
    assert first.best is not None and first.best.valid
    assert first.best.candidate_id == "initial-origin:init"
    assert first.no_improvement_over_initialization
    restored_evaluator = QuadraticObjective()
    resumed = B.resume_cma_es(tmp_path / "checkpoints/final.json", restored_evaluator,
                             budget=_budget(1, 4), run_id="initial-resumed")
    assert resumed.evaluations == resumed.forward_calls == 1
    assert resumed.generations == resumed.validation_calls == 0
    assert resumed.best == resumed.initial_candidate == first.initial_candidate
    assert resumed.initial_candidate_id == "initial-origin:init"
    assert resumed.no_improvement_over_initialization
    entries = restored_evaluator.ledger.snapshot()["entries"]
    assert len(entries) == 1 and entries[0]["candidate_id"] == "initial-origin:init"
    assert entries[0]["objective_kind"] == "hard"


def test_cma_fresh_production_resume_restores_checkpoint_selection_without_rescoring_init(tmp_path):
    initial = B.solve_cma_es(
        _prototype(), _budget(3, 2), objective_kind="smooth", seed=11,
        x0=np.zeros(6), run_id="fresh-production-cma", checkpoint_dir=tmp_path / "checkpoints",
    )
    restored_evaluator = _prototype()
    assert np.count_nonzero(restored_evaluator.sim.stock.to_numpy()) == 0
    resumed = B.resume_cma_es(
        tmp_path / "checkpoints/final.json", restored_evaluator,
        budget=_budget(5, 2), run_id="restored-production-cma",
    )
    assert resumed.initial_candidate == initial.initial_candidate
    assert resumed.initial_candidate_id == "fresh-production-cma:init"
    assert resumed.selection_thresholds == initial.selection_thresholds
    assert resumed.evaluations == 5 and resumed.generations == 2
    entries = restored_evaluator.ledger.snapshot()["entries"]
    assert len([entry for entry in entries if entry["candidate_id"].endswith(":init")]) == 1
    assert entries[0]["candidate_id"] == "fresh-production-cma:init"
    assert not any(entry["candidate_id"] == "restored-production-cma:init" for entry in entries)
    assert resumed.initial_candidate_status == initial.initial_candidate_status
    legacy = json.loads((tmp_path / "checkpoints/final.json").read_text())
    legacy["schema"] = "diffcam-cmaes-checkpoint/1"
    legacy.pop("initial_candidate")
    legacy.pop("payload_hash")
    legacy["payload_hash"] = C.content_hash(legacy)
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    untouched = _prototype()
    with pytest.raises(B.BlackBoxError, match="explicit coordinator migration"):
        B.resume_cma_es(tmp_path / "legacy.json", untouched, budget=_budget(1, 4))
    assert untouched.ledger.snapshot()["candidate_evaluations"] == 0
