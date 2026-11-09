"""Hard/smooth CMA-ES on the shared T3.0 candidate objective API.

Invalid objectives use a finite search penalty (never a resampled hidden success).
The full CMA object, not merely the incumbent, is checkpointed. Resume files are
JSON metadata with a checksum plus a base64 CMA pickle; only trusted checkpoints
written by this module should be resumed -- pickle is not a portable safe format.
Evaluation is always sequential because T3.0 uses one Taichi runtime.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from benchmarks import contracts as C
from . import run_artifacts as R

CHECKPOINT_SCHEMA = "diffcam-cmaes-checkpoint/1"
CHECKPOINT_VERSION = 1


class BlackBoxError(ValueError):
    pass


def _cma():
    try:
        import cma
    except ImportError as error:
        raise BlackBoxError("CMA-ES requires the pinned optional dependency: uv run --extra optim ...") from error
    return cma


@dataclass(frozen=True)
class OptimizationBudget:
    """Budgets are independent: CMA members versus validation evalid calls."""

    candidate_evaluations: int
    population_size: Optional[int] = None
    validation_interval: int = 0
    max_validation_calls: int = 0
    max_generations: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("candidate_evaluations", "population_size", "validation_interval",
                     "max_validation_calls", "max_generations"):
            value = getattr(self, name)
            if value is None and name in ("population_size", "max_generations"):
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.population_size is not None and self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.validation_interval == 0 and self.max_validation_calls > 0:
            raise ValueError("positive validation budget requires a positive validation_interval")
        if self.max_generations == 0:
            raise ValueError("max_generations may be None or >= 1")


@dataclass
class CmaRunResult:
    run_id: str
    objective_kind: str
    seed: int
    best: Optional[C.CandidateResult]
    best_theta: Optional[np.ndarray]
    initial_theta: np.ndarray
    evaluations: int
    validation_calls: int
    generations: int
    invalid_candidates: int
    termination_reason: str
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    resumed_from: Optional[str] = None
    previous_evaluations: int = 0

    @property
    def consumed_budget(self) -> int:
        return self.evaluations


@dataclass
class _State:
    es: Any
    budget: OptimizationBudget
    objective_kind: str
    seed: int
    run_id: str
    initial_theta: np.ndarray
    best: Optional[C.CandidateResult]
    best_theta: Optional[np.ndarray]
    evaluations: int
    validation_calls: int
    next_validation: Optional[int]
    generations: int
    invalid_candidates: int
    history: list[dict[str, Any]]
    resumed_from: Optional[str] = None
    previous_evaluations: int = 0


def _candidate_dict(result: C.CandidateResult) -> Dict[str, Any]:
    return dataclasses.replace(result, coverage=result.coverage).__dict__ | {
        "coverage": result.coverage.value,
    }


def _setup_signature(evaluator: Any, objective_kind: str) -> str:
    parameterization = evaluator.parameterization
    if hasattr(parameterization, "reference_deltas_mm"):
        reference = np.asarray(parameterization.reference_deltas_mm).tolist()
        positions = np.asarray(parameterization.reference_positions_mm).tolist()
    else:
        reference, positions = None, None
    spec = getattr(evaluator, "spec", None)
    if dataclasses.is_dataclass(spec) and not isinstance(spec, type):
        spec = dataclasses.asdict(spec)
    task = getattr(evaluator, "task", None)
    motion = getattr(evaluator, "motion", None)
    data = {
        "evaluator_type": f"{type(evaluator).__module__}.{type(evaluator).__name__}",
        "objective_kind": objective_kind,
        "objective_spec": spec,
        "task_hash": task.task_hash if task is not None else None,
        "tool_setup_hash": task.tool_setup_hash if task is not None else None,
        "motion_hash": motion.motion_hash if motion is not None else None,
        "parameterization": R.parameterization_metadata(parameterization),
        "reference_deltas_mm": reference,
        "reference_positions_mm": positions,
        "initial_position_mm": np.asarray(parameterization.initial_position_mm).tolist()
        if hasattr(parameterization, "initial_position_mm") else None,
    }
    for name in ("target", "quadratic_target_mm"):
        if hasattr(evaluator, name):
            data[name] = np.asarray(getattr(evaluator, name)).tolist()
    return C.content_hash(data)


def _options(evaluator: Any, seed: int, sigma0: float, x0: Optional[Sequence[float]],
             population_size: Optional[int]) -> tuple[list[float], float, dict[str, Any]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2 ** 32 - 2:
        raise ValueError("CMA seed must be an integer in [0, 2**32-2]")
    if not math.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("sigma0 must be finite and positive")
    dims = int(evaluator.parameterization.num_params)
    if dims < 1:
        raise ValueError("CMA requires at least one parameter")
    initial = np.zeros(dims, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1)
    if initial.shape != (dims,) or not np.isfinite(initial).all():
        raise ValueError(f"x0 must have finite shape ({dims},)")
    if not evaluator.parameterization.in_bounds(initial):
        raise ValueError("x0 violates normalized parameter bounds")
    # CMA's zero seed requests nondeterminism; preserve a bijective deterministic
    # mapping from the contract seed to a positive one for reproducibility.
    options: Dict[str, Any] = {"seed": seed + 1, "verbose": -9}
    if population_size is not None:
        options["popsize"] = population_size
    bounds = getattr(evaluator.parameterization, "bounds", None)
    if bounds is not None:
        options["bounds"] = [np.asarray(bounds[0], dtype=float).tolist(),
                             np.asarray(bounds[1], dtype=float).tolist()]
    return initial.tolist(), sigma0, options


def _new_es(evaluator: Any, budget: OptimizationBudget, seed: int, sigma0: float, x0: Any) -> Any:
    initial, sigma, options = _options(evaluator, seed, sigma0, x0, budget.population_size)
    options["maxfevals"] = budget.candidate_evaluations
    options["maxiter"] = budget.max_generations if budget.max_generations is not None else math.inf
    return _cma().CMAEvolutionStrategy(initial, sigma, options)


def _best_candidate_dict(best: Optional[C.CandidateResult], theta: Optional[np.ndarray]) -> Any:
    if best is None or theta is None:
        return None
    return {"theta": np.asarray(theta, dtype=float).tolist(), "evaluation": _candidate_dict(best)}


def _write_checkpoint(path: Path, state: _State, evaluator: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "budget": dataclasses.asdict(state.budget), "evaluations": state.evaluations,
        "validation_calls": state.validation_calls, "next_validation": state.next_validation,
        "generations": state.generations, "invalid_candidates": state.invalid_candidates,
        "es_pickle_b64": base64.b64encode(state.es.pickle_dumps()).decode("ascii"),
    }
    payload = {
        "schema": CHECKPOINT_SCHEMA, "contract_schema": C.SCHEMA_VERSION,
        "cma_version": _cma().__version__, "run_id": state.run_id, "seed": state.seed,
        "objective_kind": state.objective_kind, "setup_hash": _setup_signature(evaluator, state.objective_kind),
        "state": state_dict, "best": _best_candidate_dict(state.best, state.best_theta),
        "initial_theta": state.initial_theta.tolist(), "history": state.history,
        "ledger_snapshot": evaluator.ledger.snapshot(),
    }
    digest = C.content_hash(payload)
    document = dict(payload, payload_hash=digest)
    # Atomic replacement but no completed run evidence is implied by this state file.
    R.ArtifactWriter._replace_bytes(path, R.json_bytes(document))  # noqa: SLF001 (shared atomic utility)
    return digest


def _read_checkpoint(path: Path, evaluator: Any, objective_kind: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlackBoxError(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(document, Mapping) or document.get("schema") != CHECKPOINT_SCHEMA:
        raise BlackBoxError("unrecognized checkpoint schema")
    expected_hash = document.get("payload_hash")
    payload = dict(document)
    payload.pop("payload_hash", None)
    if expected_hash != C.content_hash(payload):
        raise BlackBoxError("checkpoint checksum mismatch")
    if document.get("objective_kind") != objective_kind:
        raise BlackBoxError("checkpoint objective kind differs")
    if document.get("setup_hash") != _setup_signature(evaluator, objective_kind):
        raise BlackBoxError("checkpoint task/tool/motion/parameterization/objective identity differs")
    return document


def _initial_state(evaluator: Any, budget: OptimizationBudget, seed: int,
                   objective_kind: str, x0: Any, state_es: Any, run_id: str) -> _State:
    flat = np.asarray(x0, dtype=np.float64).reshape(-1)
    return _State(state_es, budget, objective_kind, seed, run_id, flat,
                  None, None, 0, 0,
                  budget.validation_interval if budget.max_validation_calls > 0 else None,
                  0, 0, [])


def _run(state: _State, evaluator: Any, checkpoint_dir: Optional[Path]) -> CmaRunResult:
    reason = "budget_exhausted" if state.evaluations >= state.budget.candidate_evaluations else None
    checkpoint_path = None
    while state.evaluations < state.budget.candidate_evaluations:
        if state.budget.max_generations is not None and state.generations >= state.budget.max_generations:
            reason = "max_generations"
            break
        stopped = state.es.stop()
        if stopped:
            reason = "converged"
            break
        candidates = state.es.ask()
        if state.evaluations + len(candidates) > state.budget.candidate_evaluations:
            # Never truncate a population silently to change the update dynamics.
            reason = "budget_exhausted"
            break
        fitness = []
        best_in_population: Optional[C.CandidateResult] = None
        best_in_population_theta = None
        invalid = 0
        for candidate_index, theta in enumerate(candidates):
            physical = np.asarray(theta, dtype=np.float64)
            candidate_id = f"{state.run_id}:g{state.generations}:m{candidate_index}"
            result = evaluator.evaluate_candidate(physical, candidate_id, state.objective_kind)
            if result.valid and result.objective is not None:
                value = float(result.objective)
                if state.best is None or value < state.best.objective:
                    state.best = dataclasses.replace(result)
                    state.best_theta = physical.copy()
                if best_in_population is None or value < best_in_population.objective:
                    best_in_population = result
                    best_in_population_theta = physical.copy()
            else:
                value = float(optimizer_invalid_penalty())
                invalid += 1
            fitness.append(value)
        state.es.tell(candidates, fitness)
        state.evaluations += len(candidates)
        state.invalid_candidates += invalid
        state.generations += 1
        state.history.append({
            "generation": state.generations, "evaluations": state.evaluations,
            "population_size": len(candidates), "invalid_members": invalid,
            "best_fitness": None if best_in_population is None else best_in_population.objective,
            "best_overall": None if state.best is None else state.best.objective,
            "best_theta": None if state.best_theta is None else state.best_theta.tolist(),
        })
        if (state.next_validation is not None and state.best is not None
                and state.validation_calls < state.budget.max_validation_calls
                and state.evaluations >= state.next_validation):
            state.validation_calls += 1
            evaluator.ledger.count_validation(1)
            evaluator.evaluate_candidate(state.best_theta,
                                         f"{state.run_id}:validation:{state.validation_calls}",
                                         "hard")
            state.next_validation += state.budget.validation_interval
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"ckpt-{state.evaluations:06d}.json"
            _write_checkpoint(checkpoint_path, state, evaluator)
    if reason is None:
        reason = "budget_exhausted"
    if checkpoint_dir is not None and state.evaluations > 0:
        final_path = checkpoint_dir / "final.json"
        _write_checkpoint(final_path, state, evaluator)
        checkpoint_path = final_path
    return CmaRunResult(
        run_id=state.run_id, objective_kind=state.objective_kind, seed=state.seed,
        best=state.best, best_theta=state.best_theta, initial_theta=state.initial_theta,
        evaluations=state.evaluations, validation_calls=state.validation_calls,
        generations=state.generations, invalid_candidates=state.invalid_candidates,
        termination_reason=reason, history=state.history,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        resumed_from=state.resumed_from,
        previous_evaluations=state.previous_evaluations,
    )


def solve_cma_es(evaluator: Any, budget: OptimizationBudget, *, seed: int = 0,
                 objective_kind: str = "smooth", sigma0: float = 0.1, x0: Any = None,
                 run_id: Optional[str] = None, checkpoint_dir: Optional[str | Path] = None) -> CmaRunResult:
    if objective_kind not in ("hard", "smooth"):
        raise ValueError("objective_kind must be hard or smooth")
    initial, _, _ = _options(evaluator, seed, sigma0, x0, budget.population_size)
    es = _new_es(evaluator, budget, seed, sigma0, x0)
    state = _initial_state(evaluator, budget, seed, objective_kind, initial, es,
                           run_id or f"cma-{objective_kind}-seed-{seed}")
    return _run(state, evaluator, Path(checkpoint_dir) if checkpoint_dir is not None else None)


def resume_cma_es(path: str | Path, evaluator: Any, *, budget: Optional[OptimizationBudget] = None,
                  objective_kind: Optional[str] = None, run_id: Optional[str] = None,
                  checkpoint_dir: Optional[str | Path] = None) -> CmaRunResult:
    path = Path(path)
    with path.open() as stream:
        header = json.load(stream)
    kind = objective_kind or header.get("objective_kind", "smooth")
    document = _read_checkpoint(path, evaluator, kind)
    import pickle
    try:
        es = pickle.loads(base64.b64decode(document["state"]["es_pickle_b64"]))
    except Exception as error:
        raise BlackBoxError(f"unsafe/unsupported CMA checkpoint payload: {error}") from error
    if not isinstance(es, _cma().CMAEvolutionStrategy):
        raise BlackBoxError("checkpoint is not a CMA strategy")
    old_budget = OptimizationBudget(**document["state"]["budget"])
    new_budget = budget or old_budget
    origin = int(document["state"]["evaluations"])
    if new_budget.candidate_evaluations < origin:
        raise BlackBoxError("resume budget may not reduce already-consumed evaluation count")
    if new_budget.population_size is not None and new_budget.population_size != old_budget.population_size:
        raise BlackBoxError("resume may not change population size or validation cadence")
    if (new_budget.validation_interval != old_budget.validation_interval
            or new_budget.max_validation_calls < document["state"]["validation_calls"]):
        raise BlackBoxError("resume may not change cadence or reduce already-used validation budget")
    es.opts["maxfevals"] = new_budget.candidate_evaluations
    state_dict = document["state"]
    best = None
    if document["best"] is not None:
        best = C.CandidateResult(**document["best"]["evaluation"])
    state = _State(
        es, new_budget, document["objective_kind"], int(document["seed"]), run_id or document["run_id"],
        np.asarray(document["initial_theta"], dtype=np.float64), best,
        None if best is None else np.asarray(document["best"]["theta"], dtype=np.float64),
        origin, int(state_dict["validation_calls"]), state_dict["next_validation"],
        int(state_dict["generations"]), int(state_dict["invalid_candidates"]), list(document["history"]),
        resumed_from=document["payload_hash"], previous_evaluations=origin,
    )
    return _run(state, evaluator, Path(checkpoint_dir) if checkpoint_dir is not None else None)


def optimizer_invalid_penalty() -> float:
    """Search-only penalty. All invalid members are counted; no resampling."""
    return 1e12


def emit_optimization_artifacts(result: CmaRunResult, evaluator: Any, run_dir: str | Path,
                                repository: str | Path) -> C.RunRecord:
    """Write actual optimization outputs; posted evalid remains explicitly incomplete."""
    if result.best is None or result.best_theta is None:
        raise BlackBoxError("no feasible candidate best; cannot fabricate completed run artifacts")
    task, motion = evaluator.task, evaluator.candidate_motion(result.best_theta)
    posting = R.post_motion(task, motion)
    config = R.build_run_config(result.run_id, task, motion, seed=result.seed,
                                parameterization=evaluator.parameterization,
                                objective=evaluator.spec,
                                budgets={"candidate_evaluations": result.evaluations,
                                         "validation_calls": result.validation_calls,
                                         "generations": result.generations},
                                repository=repository)
    initial_positions = evaluator.parameterization.positions(result.initial_theta)
    best_positions = evaluator.parameterization.positions(result.best_theta)
    files = dict(posting.files)
    files.update({
        "config.json": R.json_bytes(config),
        "history.json": R.json_bytes({"generations": result.history,
                                      "ledger": evaluator.ledger.snapshot()}),
        "metrics.json": R.json_bytes({"best_candidate": _candidate_dict(result.best),
                                      "termination_reason": result.termination_reason,
                                      "invalid_candidates": result.invalid_candidates}),
        "motion_plan.json": R.record_bytes(motion),
        "initial_trajectory.npy": R.numpy_bytes(initial_positions),
        "trajectory.npy": R.numpy_bytes(best_positions),
        "trajectory_deltas.npy": R.numpy_bytes(np.diff(best_positions, axis=0)),
    })
    record = R.assemble_record(
        result.run_id, task, C.RunStatus.INCOMPLETE, posting=posting,
        artifact_names=tuple(sorted(files)), source_commit=config["source_commit"],
        dirty_patch_hash=config["dirty_patch_hash"],
        failure_reason="Posted evaluation and T1.3 integration are not available yet",
    )
    R.ArtifactWriter().write(str(run_dir), record, files)
    return record
