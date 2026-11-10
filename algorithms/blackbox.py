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

CHECKPOINT_SCHEMA = "diffcam-cmaes-checkpoint/2"
CHECKPOINT_VERSION = 2


class BlackBoxError(ValueError):
    pass


@dataclass(frozen=True)
class HardSelectionPolicy:
    """Task-relative hard selection, shared by Adam and CMA-ES.

    ``None`` for a volume-derived component means use the corresponding
    TaskSpec verdict tolerance.  Margins are in the same normalized fraction
    units as the declared gouge/residual components and are subtracted from
    their selection thresholds.  Holder overlap always must be exactly zero.
    """

    max_gouge_fraction: Optional[float] = None
    max_residual_fraction: Optional[float] = None
    max_holder_overlap_fraction: float = 0.0
    selection_margin: float = 0.0
    infeasible_ranking: str = "lexicographic"
    zero_tolerance_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_gouge_fraction", "max_residual_fraction",
                     "max_holder_overlap_fraction"):
            value = getattr(self, name)
            if value is None and name != "max_holder_overlap_fraction":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite fraction in [0,1] or None")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite fraction in [0,1] or None")
        if (isinstance(self.selection_margin, bool)
                or not isinstance(self.selection_margin, (int, float))
                or not math.isfinite(self.selection_margin) or not 0 <= self.selection_margin <= 1):
            raise ValueError("selection_margin must be a finite fraction in [0,1]")
        if self.max_holder_overlap_fraction != 0.0:
            raise ValueError("holder overlap must be exactly zero; a nonzero holder threshold is not permitted")
        if self.infeasible_ranking not in ("lexicographic", "tolerance_normalised_sum"):
            raise ValueError("infeasible_ranking must be lexicographic or tolerance_normalised_sum")
        if (isinstance(self.zero_tolerance_scale, bool)
                or not isinstance(self.zero_tolerance_scale, (int, float))
                or not math.isfinite(self.zero_tolerance_scale) or self.zero_tolerance_scale <= 0):
            raise ValueError("zero_tolerance_scale must be finite and strictly positive")


@dataclass(frozen=True)
class ResolvedHardSelection:
    """Validated policy plus the exact fraction thresholds used for this task."""

    max_gouge_fraction: float
    max_residual_fraction: float
    max_holder_overlap_fraction: float
    selection_margin: float
    infeasible_ranking: str
    verdict_gouge_fraction: float
    verdict_residual_fraction: float
    gouge_denominator_mm3: float
    residual_denominator_mm3: float
    zero_tolerance_scale: float = 1.0

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "diffcam-hard-selection/1",
            "infeasible_ranking": self.infeasible_ranking,
            "zero_tolerance_scale": self.zero_tolerance_scale,
            "selection_margin_fraction": self.selection_margin,
            "thresholds_used": {
                "max_gouge_fraction": self.max_gouge_fraction,
                "max_residual_fraction": self.max_residual_fraction,
                "max_holder_overlap_fraction": self.max_holder_overlap_fraction,
            },
            "verdict_fraction_thresholds": {
                "gouge": self.verdict_gouge_fraction,
                "residual": self.verdict_residual_fraction,
            },
            "fraction_denominators_mm3": {
                "initial_in_target": self.gouge_denominator_mm3,
                "initial_removal_target": self.residual_denominator_mm3,
            },
            "holder_safety_override": "exactly zero irrespective of residual/gouge or objective",
            "normalisation": ("positive selection threshold divides each violation; an exact-zero "
                              f"threshold uses the declared full-fraction scale {self.zero_tolerance_scale}"),
            "config_hash": C.content_hash(dataclasses.asdict(self)),
        }


def _hard_fraction_denominators_mm3(evaluator: Any) -> tuple[float, float]:
    """Use the first valid hard evaluation's unchanged initial-stock history.

    The initializer explicitly may be invalid, so a missing/empty initial grid
    cannot be converted to a positive denominator.  Cheap test evaluators can
    supply ``hard_fraction_denominators_mm3`` directly, in their declared task
    units.  Production evaluators use the same masks/denominators as
    CandidateEvaluator and csg_metrics; no second carve is issued here.
    """
    provided = getattr(evaluator, "hard_fraction_denominators_mm3", None)
    if provided is not None:
        if not isinstance(provided, Mapping):
            raise BlackBoxError("hard_fraction_denominators_mm3 must be a mapping")
        try:
            denominators = float(provided["gouge"]), float(provided["residual"])
        except (KeyError, TypeError, ValueError) as error:
            raise BlackBoxError("invalid hard fraction denominator declaration") from error
    else:
        task = getattr(evaluator, "task", None)
        sim = getattr(evaluator, "sim", None)
        if task is None or sim is None:
            raise BlackBoxError("task/simulator required to derive hard selection fraction thresholds")
        try:
            # This is the initial slot of the just-scored forward history, not
            # the final carved stock.  Its source was reset by the evaluator.
            initial = np.asarray(sim.stock.to_numpy())[0] < 0.0
            target = np.asarray(sim.target.to_numpy()) < 0.0
            if initial.shape != target.shape:
                raise BlackBoxError("initial stock and target shapes differ")
            counts = np.count_nonzero(initial & target), np.count_nonzero(initial & ~target)
            voxel_mm3 = float(task.grid.voxel_mm) ** 3
            denominators = float(counts[0] * voxel_mm3), float(counts[1] * voxel_mm3)
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise BlackBoxError("cannot derive selection fractions without initialized stock/target") from error
    if any(not math.isfinite(value) or value <= 0 for value in denominators):
        raise BlackBoxError("degenerate initial stock/target fractions cannot define selection thresholds")
    return denominators


def _resolve_hard_selection(evaluator: Any, policy: HardSelectionPolicy) -> ResolvedHardSelection:
    """Resolve and check normalized selection thresholds against task verdicts."""
    task = getattr(evaluator, "task", None)
    tolerances = getattr(task, "tolerances", None)
    if tolerances is None:
        raise BlackBoxError("hard selection requires TaskSpec verdict tolerances")
    gouge_denominator, residual_denominator = _hard_fraction_denominators_mm3(evaluator)
    verdict = {
        "gouge": float(tolerances.gouge_volume_mm3) / gouge_denominator,
        "residual": float(tolerances.completion_residual_mm3) / residual_denominator,
        "holder": 0.0,
    }
    configured = {
        "gouge": policy.max_gouge_fraction,
        "residual": policy.max_residual_fraction,
        "holder": policy.max_holder_overlap_fraction,
    }
    thresholds = {}
    for name in ("gouge", "residual", "holder"):
        raw = verdict[name] if configured[name] is None else configured[name]
        if raw > verdict[name]:
            raise BlackBoxError(f"selection {name} threshold exceeds its task verdict tolerance")
        threshold = max(0.0, raw - (policy.selection_margin if name != "holder" else 0.0))
        thresholds[name] = threshold
    return ResolvedHardSelection(
        thresholds["gouge"], thresholds["residual"], 0.0, policy.selection_margin,
        policy.infeasible_ranking, verdict["gouge"], verdict["residual"],
        gouge_denominator, residual_denominator, policy.zero_tolerance_scale,
    )


def _hard_rank(result: C.CandidateResult, policy: ResolvedHardSelection) -> Optional[tuple[float, ...]]:
    """Return a totally ordered hard-valid rank; feasibility always dominates.

    The default infeasible order implements the coordinator's D-10 addendum 2:
    holder, gouge, residual, then objective.  The alternative sums the three
    threshold-normalized violations.  All states are also ordered by the
    explicit feasibility bit, so no scalar term can buy a collision or gouge.
    Missing uniform-denominated hard components, invalid objective_kind, and a
    positive ``feasible`` override remain non-rankable, never zero violations.
    """
    if not isinstance(policy, ResolvedHardSelection):
        raise BlackBoxError("hard ranking requires task-resolved selection thresholds")
    if not result.valid or result.objective is None or result.objective_kind != "hard":
        return None
    names = ("holder_overlap_fraction", "gouge_fraction", "residual_fraction")
    if any(name not in result.components for name in names):
        return None
    holder, gouge, residual = (float(result.components[name]) for name in names)
    observed = (holder, gouge, residual)
    if any(not math.isfinite(value) or value < 0 for value in observed):
        return None
    thresholds = (policy.max_holder_overlap_fraction, policy.max_gouge_fraction,
                  policy.max_residual_fraction)
    violations = tuple(max(0.0, value - threshold) for value, threshold in zip(observed, thresholds))
    feasible = all(violation == 0 for violation in violations)
    declared = result.components.get("feasible")
    if declared is not None:
        if declared not in (0.0, 1.0):
            raise BlackBoxError("hard feasible component must be exactly 0 or 1")
        if feasible and declared != 1.0:
            # A positive verdict can never override a modeled constraint failure
            # or the stricter selection margin.
            return None
        if declared == 0.0:
            # The objective may report additional stricter feasibility evidence;
            # such a candidate stays in the infeasible set under either ranking.
            feasible = False
    infeasible = int(not feasible)
    if policy.infeasible_ranking == "lexicographic":
        return (infeasible, *violations, float(result.objective))
    normalized = []
    for violation, threshold in zip(violations, thresholds):
        # Exact-zero tolerances are dimensionless here (e.g. holder/gouge
        # fraction). Dividing by zero cannot be made safe or serializable;
        # their declared full-fraction reference is explicitly 1.0.
        normalized.append(violation / (threshold if threshold > 0 else policy.zero_tolerance_scale))
    return (infeasible, float(sum(normalized)), *violations, float(result.objective))


def _initial_candidate_dict(result: Optional[C.CandidateResult], theta: Optional[np.ndarray],
                            candidate_id: Optional[str]) -> Any:
    if result is None or theta is None or candidate_id is None:
        return None
    return {"candidate_id": candidate_id, "theta": np.asarray(theta, dtype=float).tolist(),
            "evaluation": _candidate_dict(result)}


def _restore_initial_candidate(document: Mapping[str, Any], theta: np.ndarray) -> tuple[str, C.CandidateResult]:
    record = document.get("initial_candidate")
    if not isinstance(record, Mapping):
        raise BlackBoxError("checkpoint has no preserved hard initialization; explicit migration is required")
    candidate_id = record.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.endswith(":init"):
        raise BlackBoxError("invalid initialization candidate identity")
    try:
        result = C.CandidateResult(**record["evaluation"])
    except (KeyError, TypeError, ValueError) as error:
        raise BlackBoxError("checkpoint initialization result is invalid") from error
    saved_theta = np.asarray(record.get("theta"), dtype=float)
    if saved_theta.shape != theta.shape or not np.array_equal(saved_theta, theta):
        raise BlackBoxError("checkpoint initialization theta differs from its result identity")
    if result.candidate_id != candidate_id or result.objective_kind != "hard":
        raise BlackBoxError("checkpoint initialization result identity is not the hard init evaluation")
    # Same canonical scalar/float hash used by CandidateEvaluator._theta_hash,
    # independently re-derived here; resume never re-scores the candidate.
    if result.theta_hash != C.content_hash(saved_theta.tolist()):
        raise BlackBoxError("checkpoint initialization result theta hash differs")
    return candidate_id, result


def _restore_hard_selection(document: Mapping[str, Any],
                            policy: HardSelectionPolicy) -> Optional[ResolvedHardSelection]:
    """Restore immutable thresholds without reading a fresh simulator's zero stock.

    The checkpoint itself carries the posterior task denominators/thresholds.  A
    checksum and exact setup identity were verified by its reader.  Re-deriving
    from uninitialized current stock would reject every real resume/replay.
    """
    record = document.get("selection_thresholds")
    if record is None:
        return None
    try:
        if not isinstance(record, Mapping) or record.get("schema") != "diffcam-hard-selection/1":
            raise BlackBoxError("unrecognized hard selection checkpoint schema")
        thresholds = record["thresholds_used"]
        verdict = record["verdict_fraction_thresholds"]
        denominators = record["fraction_denominators_mm3"]
        limits = ResolvedHardSelection(
            thresholds["max_gouge_fraction"], thresholds["max_residual_fraction"],
            thresholds["max_holder_overlap_fraction"], record["selection_margin_fraction"],
            record["infeasible_ranking"], verdict["gouge"], verdict["residual"],
            denominators["initial_in_target"], denominators["initial_removal_target"],
            record["zero_tolerance_scale"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BlackBoxError("invalid checkpoint hard selection metadata") from error
    if limits.metadata() != dict(record):
        raise BlackBoxError("checkpoint hard selection identity hash differs")
    if (limits.infeasible_ranking != policy.infeasible_ranking
            or limits.selection_margin != policy.selection_margin
            or limits.zero_tolerance_scale != policy.zero_tolerance_scale):
        raise BlackBoxError("checkpoint hard selection policy differs")
    if limits.max_holder_overlap_fraction != 0.0 or policy.max_holder_overlap_fraction != 0.0:
        raise BlackBoxError("checkpoint holder threshold differs")
    for name in ("gouge", "residual"):
        value = limits.verdict_gouge_fraction if name == "gouge" else limits.verdict_residual_fraction
        configured = getattr(policy, f"max_{name}_fraction")
        raw = value if configured is None else configured
        actual = limits.max_gouge_fraction if name == "gouge" else limits.max_residual_fraction
        if (any(not math.isfinite(float(item)) or float(item) < 0 for item in (
                value, raw, actual))
                or raw > value or actual != max(0.0, raw - policy.selection_margin)):
            raise BlackBoxError(f"checkpoint {name} threshold differs from its policy or verdict")
    if any(not math.isfinite(value) or value <= 0
           for value in (limits.gouge_denominator_mm3, limits.residual_denominator_mm3)):
        raise BlackBoxError("checkpoint hard fraction denominators are invalid")
    return limits


def _selection_metadata(result: Any) -> dict[str, Any]:
    return {
        "initial_candidate_id": result.initial_candidate_id,
        "initial_candidate": None if result.initial_candidate is None else _candidate_dict(result.initial_candidate),
        "initial_candidate_status": result.initial_candidate_status,
        "no_improvement_over_initialization": result.no_improvement_over_initialization,
        "selection_reason": result.selection_reason,
        "hard_selection": result.selection_thresholds,
    }


def _initial_status(initial: Optional[C.CandidateResult], rank: Optional[tuple]) -> str:
    if initial is None:
        return "unavailable"
    if not initial.valid:
        return "invalid"
    if rank is None:
        return "unknown"
    return "feasible" if rank[0] == 0 else "infeasible"


def _selection_reason(best: Optional[C.CandidateResult], initial_id: str,
                      initial_status: str) -> str:
    if best is not None:
        return ("no improvement over initialization" if best.candidate_id == initial_id
                else "improved hard-valid candidate selected")
    return f"no hard-selected candidate; initialization status: {initial_status}"


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
        if (self.population_size is not None and self.candidate_evaluations > 1
                and self.candidate_evaluations - 1 < self.population_size):
            raise ValueError("candidate_evaluations is smaller than one CMA population after the hard init; "
                             "declare 1 for init-only or at least population_size + 1")
        if self.validation_interval == 0 and self.max_validation_calls > 0:
            raise ValueError("positive validation budget requires a positive validation_interval")
        if self.max_generations == 0:
            raise ValueError("max_generations may be None or >= 1")


@dataclass
class CmaRunResult:
    run_id: str
    objective_kind: str
    seed: int
    budget: OptimizationBudget
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
    forward_calls: int = 0
    backward_calls: int = 0
    selection: HardSelectionPolicy = field(default_factory=HardSelectionPolicy)
    initial_candidate_id: Optional[str] = None
    initial_candidate: Optional[C.CandidateResult] = None
    initial_candidate_rank: Optional[tuple] = None
    initial_candidate_status: str = "unavailable"
    selection_thresholds: Optional[dict[str, Any]] = None
    no_improvement_over_initialization: bool = False
    selection_reason: str = "no hard initialization evaluation"

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
    forward_calls: int = 0
    backward_calls: int = 0
    selection: HardSelectionPolicy = field(default_factory=HardSelectionPolicy)
    selection_limits: Optional[ResolvedHardSelection] = None
    initial_candidate_id: Optional[str] = None
    initial_candidate: Optional[C.CandidateResult] = None
    initial_candidate_rank: Optional[tuple] = None
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
    simulator = getattr(evaluator, "sim", None)
    data = {
        "simulator_precision": R.simulator_precision_metadata(simulator) if simulator is not None else None,
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
        # Active per-run field shift: None and an explicit zero are both neutral.
        "evaluation_loss_shift": float(getattr(evaluator, "evaluation_loss_shift", 0.0) or 0.0),
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
        "forward_calls": state.forward_calls, "backward_calls": state.backward_calls,
        "es_pickle_b64": base64.b64encode(state.es.pickle_dumps()).decode("ascii"),
    }
    payload = {
        "schema": CHECKPOINT_SCHEMA, "contract_schema": C.SCHEMA_VERSION,
        "cma_version": _cma().__version__, "run_id": state.run_id, "seed": state.seed,
        "objective_kind": state.objective_kind, "setup_hash": _setup_signature(evaluator, state.objective_kind),
        "state": state_dict, "best": _best_candidate_dict(state.best, state.best_theta),
        "initial_theta": state.initial_theta.tolist(), "history": state.history,
        "selection": dataclasses.asdict(state.selection),
        "initial_candidate": _initial_candidate_dict(state.initial_candidate, state.initial_theta,
                                                       state.initial_candidate_id),
        "initial_candidate_rank": state.initial_candidate_rank,
        "selection_thresholds": (None if state.selection_limits is None
                                 else state.selection_limits.metadata()),
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
    if (not isinstance(document, Mapping)
            or document.get("schema") not in (CHECKPOINT_SCHEMA, "diffcam-cmaes-checkpoint/1")):
        raise BlackBoxError("unrecognized checkpoint schema")
    expected_hash = document.get("payload_hash")
    payload = dict(document)
    payload.pop("payload_hash", None)
    if expected_hash != C.content_hash(payload):
        raise BlackBoxError("checkpoint checksum mismatch")
    if document.get("objective_kind") != objective_kind:
        raise BlackBoxError("checkpoint objective kind differs")
    if document.get("setup_hash") != _setup_signature(evaluator, objective_kind):
        raise BlackBoxError("checkpoint task/tool/motion/parameterization/objective/simulator-precision identity differs")
    return document


def _restore_ledger_payload(document: Mapping[str, Any], evaluator: Any) -> None:
    snapshot = document.get("ledger_snapshot")
    if snapshot is None:
        raise BlackBoxError("missing ledger_snapshot; cannot restore compute accounting")
    restore = getattr(evaluator.ledger, "restore", None)
    if restore is None:
        raise BlackBoxError("evaluator ledger cannot restore a matching cumulative history")
    restore(snapshot)


def _initial_state(evaluator: Any, budget: OptimizationBudget, seed: int,
                   objective_kind: str, x0: Any, state_es: Any, run_id: str,
                   selection: HardSelectionPolicy) -> _State:
    flat = np.asarray(x0, dtype=np.float64).reshape(-1)
    return _State(state_es, budget, objective_kind, seed, run_id, flat,
                  None, None, 1, 0,
                  budget.validation_interval if budget.max_validation_calls > 0 else None,
                  0, 0, [], selection=selection)


def _evaluate_initial_candidate(state: _State, evaluator: Any) -> None:
    """One ledgered hard candidate. Resume calls restore, never this helper."""
    if state.initial_candidate is not None or state.evaluations != 1:
        raise BlackBoxError("hard initialization has already been evaluated")
    state.initial_candidate_id = f"{state.run_id}:init"
    before = evaluator.ledger.snapshot()
    result = evaluator.evaluate_candidate(state.initial_theta, state.initial_candidate_id, "hard")
    after = evaluator.ledger.snapshot()
    state.forward_calls += after["forward_calls"] - before["forward_calls"]
    state.backward_calls += after["backward_calls"] - before["backward_calls"]
    if after["candidate_evaluations"] != before["candidate_evaluations"] + 1:
        raise BlackBoxError("initialization must consume exactly one ordinary candidate evaluation")
    state.initial_candidate = result
    _refine_hard_selection(state, evaluator, result)
    _select_hard_candidate(state, result, state.initial_theta)


def _refine_hard_selection(state: _State, evaluator: Any, result: C.CandidateResult) -> None:
    """Resolve task fractions after initial stock has materialized, without resimulation."""
    if state.selection_limits is None:
        try:
            state.selection_limits = _resolve_hard_selection(evaluator, state.selection)
        except BlackBoxError:
            if result.valid:
                raise
            # An invalid init cannot become an incumbent. A later valid member
            # may provide usable initial geometry and define those thresholds.
            return
    rank = _hard_rank(result, state.selection_limits)
    if state.initial_candidate is result:
        state.initial_candidate_rank = rank


def _select_hard_candidate(state: _State, result: C.CandidateResult, theta: np.ndarray) -> None:
    if state.selection_limits is None:
        return
    result_rank = _hard_rank(result, state.selection_limits)
    if result is state.initial_candidate:
        result_rank = state.initial_candidate_rank
    if result_rank is None:
        return
    current_rank = None if state.best is None else _hard_rank(state.best, state.selection_limits)
    if current_rank is None or result_rank < current_rank:
        state.best = dataclasses.replace(result)
        state.best_theta = np.asarray(theta, dtype=float).copy()


def _hard_population_fitness(ranks: Sequence[Optional[tuple[float, ...]]]) -> list[float]:
    """Return CMA fitness in original member order, ranked by hard feasibility.

    CMA uses order, not the gap between scalar objective values. Assigning the
    ordinal in this population lets its distribution follow the same declared
    safety/completion rank as final selection. Invalid/non-rankable members
    always receive the finite invalid penalty, with no hidden resampling.
    """
    order = sorted(range(len(ranks)),
                   key=lambda index: (ranks[index] is None,
                                      ranks[index] if ranks[index] is not None else ()))
    fitness = [float(optimizer_invalid_penalty())] * len(ranks)
    position = 0
    for index in order:
        if ranks[index] is not None:
            fitness[index] = float(position)
            position += 1
    return fitness


def _run(state: _State, evaluator: Any, checkpoint_dir: Optional[Path]) -> CmaRunResult:
    getter = getattr(evaluator, "clear_evaluation_k", None)
    if getter is not None:
        getter()
    # Baselines start from neutral signed-field calibration. A continuation
    # must re-establish its declared shift, never inherit ambient state.
    clear_loss_shift = getattr(evaluator, "clear_evaluation_loss_shift", None)
    if clear_loss_shift is not None:
        clear_loss_shift()
    if state.initial_candidate is None:
        _evaluate_initial_candidate(state, evaluator)
    if state.previous_evaluations == 0 and not state.initial_candidate.valid:
        state.invalid_candidates += 1
    reason = "budget_exhausted" if state.evaluations >= state.budget.candidate_evaluations else None
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = checkpoint_dir / f"ckpt-{state.evaluations:06d}.json"
        _write_checkpoint(checkpoint_path, state, evaluator)
    while state.evaluations < state.budget.candidate_evaluations:
        if state.budget.max_generations is not None and state.generations >= state.budget.max_generations:
            reason = "max_generations"
            break
        stopped = state.es.stop()
        if stopped:
            reason = "converged"
            break
        expected_size = (state.budget.population_size if state.budget.population_size is not None
                         else int(state.es.sp.popsize))
        if state.budget.candidate_evaluations - state.evaluations < expected_size:
            # Check remaining evaluations BEFORE ask(): even discarded draws mutate RNG,
            # which would make final.json diverge from its last completed generation.
            reason = "budget_exhausted"
            break
        candidates = state.es.ask()
        if len(candidates) != expected_size:
            raise BlackBoxError("CMA returned an unexpected population size")
        # Unreachable after the pre-ask() check above (review O5/Q4); never truncate a population.
        assert state.evaluations + len(candidates) <= state.budget.candidate_evaluations
        fitness = []
        best_in_population: Optional[C.CandidateResult] = None
        best_in_population_theta = None
        invalid = 0
        population_ranks: list[Optional[tuple[float, ...]]] = []
        for candidate_index, theta in enumerate(candidates):
            physical = np.asarray(theta, dtype=np.float64)
            candidate_id = f"{state.run_id}:g{state.generations}:m{candidate_index}"
            before = evaluator.ledger.snapshot()
            result = evaluator.evaluate_candidate(physical, candidate_id, state.objective_kind)
            after = evaluator.ledger.snapshot()
            state.forward_calls += after["forward_calls"] - before["forward_calls"]
            state.backward_calls += after["backward_calls"] - before["backward_calls"]
            rank = None
            if result.valid and result.objective is not None:
                value = float(result.objective)
                if best_in_population is None or value < best_in_population.objective:
                    best_in_population = result
                    best_in_population_theta = physical.copy()
                if state.objective_kind == "hard":
                    _refine_hard_selection(state, evaluator, result)
                    rank = _hard_rank(result, state.selection_limits)
                    if rank is not None:
                        _select_hard_candidate(state, result, physical)
                    else:
                        # A finite result with unavailable modeled safety is
                        # not a free objective win. It cannot guide hard CMA.
                        invalid += 1
            else:
                value = float(optimizer_invalid_penalty())
                invalid += 1
            population_ranks.append(rank)
            fitness.append(value)
        if state.objective_kind == "hard":
            fitness = _hard_population_fitness(population_ranks)
        state.es.tell(candidates, fitness)
        state.evaluations += len(candidates)
        state.invalid_candidates += invalid
        state.generations += 1
        state.history.append({
            "generation": state.generations, "evaluations": state.evaluations,
            "population_size": len(candidates), "invalid_members": invalid,
            "fitness": list(fitness),
            "fitness_semantics": ("hard_rank_ordinal_within_population"
                                  if state.objective_kind == "hard" else "raw_objective"),
            "best_fitness": None if best_in_population is None else best_in_population.objective,
            "best_overall": None if state.best is None else state.best.objective,
            "best_theta": None if state.best_theta is None else state.best_theta.tolist(),
        })
        validation_theta = best_in_population_theta
        if state.objective_kind == "hard" and state.best is not None:
            validation_theta = state.best_theta
        if (state.next_validation is not None and validation_theta is not None
                and state.validation_calls < state.budget.max_validation_calls
                and state.evaluations >= state.next_validation):
            state.validation_calls += 1
            evaluator.ledger.start_validation()
            before = evaluator.ledger.snapshot()
            validated = evaluator.evaluate_candidate(validation_theta,
                                                     f"{state.run_id}:validation:{state.validation_calls}",
                                                     "hard")
            after = evaluator.ledger.snapshot()
            state.forward_calls += after["forward_calls"] - before["forward_calls"]
            state.backward_calls += after["backward_calls"] - before["backward_calls"]
            _refine_hard_selection(state, evaluator, validated)
            _select_hard_candidate(state, validated, validation_theta.copy())
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
    initial_status = _initial_status(state.initial_candidate, state.initial_candidate_rank)
    no_improvement = (state.best is not None and state.initial_candidate_id is not None
                      and state.best.candidate_id == state.initial_candidate_id)
    return CmaRunResult(
        run_id=state.run_id, objective_kind=state.objective_kind, seed=state.seed,
        budget=state.budget,
        best=state.best, best_theta=state.best_theta, initial_theta=state.initial_theta,
        evaluations=state.evaluations, validation_calls=state.validation_calls,
        generations=state.generations, invalid_candidates=state.invalid_candidates,
        forward_calls=state.forward_calls, backward_calls=state.backward_calls,
        termination_reason=reason, history=state.history,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        resumed_from=state.resumed_from,
        previous_evaluations=state.previous_evaluations,
        selection=state.selection,
        initial_candidate_id=state.initial_candidate_id,
        initial_candidate=state.initial_candidate,
        initial_candidate_rank=state.initial_candidate_rank,
        initial_candidate_status=initial_status,
        selection_thresholds=(None if state.selection_limits is None else state.selection_limits.metadata()),
        no_improvement_over_initialization=no_improvement,
        selection_reason=_selection_reason(state.best, state.initial_candidate_id or "", initial_status),
    )


def solve_cma_es(evaluator: Any, budget: OptimizationBudget, *, seed: int = 0,
                 objective_kind: str = "smooth", sigma0: float = 0.1, x0: Any = None,
                 run_id: Optional[str] = None, checkpoint_dir: Optional[str | Path] = None,
                 selection: Optional[HardSelectionPolicy] = None) -> CmaRunResult:
    if objective_kind not in ("hard", "smooth"):
        raise ValueError("objective_kind must be hard or smooth")
    if budget.candidate_evaluations < 1:
        raise ValueError("CMA hard initialization requires at least one candidate evaluation")
    selection = HardSelectionPolicy() if selection is None else selection
    if not isinstance(selection, HardSelectionPolicy):
        raise TypeError("selection must be a HardSelectionPolicy")
    initial, _, _ = _options(evaluator, seed, sigma0, x0, budget.population_size)
    es = _new_es(evaluator, budget, seed, sigma0, x0)
    state = _initial_state(evaluator, budget, seed, objective_kind, initial, es,
                           run_id or f"cma-{objective_kind}-seed-{seed}", selection)
    return _run(state, evaluator, Path(checkpoint_dir) if checkpoint_dir is not None else None)


def resume_cma_es(path: str | Path, evaluator: Any, *, budget: Optional[OptimizationBudget] = None,
                  objective_kind: Optional[str] = None, run_id: Optional[str] = None,
                  checkpoint_dir: Optional[str | Path] = None) -> CmaRunResult:
    path = Path(path)
    with path.open() as stream:
        header = json.load(stream)
    kind = objective_kind or header.get("objective_kind", "smooth")
    document = _read_checkpoint(path, evaluator, kind)
    if document["schema"] != CHECKPOINT_SCHEMA:
        raise BlackBoxError("legacy CMA checkpoint requires explicit coordinator migration; no init was re-evaluated")
    initial_theta = np.asarray(document["initial_theta"], dtype=np.float64)
    initial_id, initial = _restore_initial_candidate(document, initial_theta)
    selection = HardSelectionPolicy(**document["selection"])
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
    if origin < 1 or new_budget.candidate_evaluations < origin:
        raise BlackBoxError("resume budget may not reduce already-consumed evaluation count")
    if new_budget.population_size is not None and new_budget.population_size != old_budget.population_size:
        raise BlackBoxError("resume may not change population size or validation cadence")
    if (new_budget.validation_interval != old_budget.validation_interval
            or new_budget.max_validation_calls < document["state"]["validation_calls"]):
        raise BlackBoxError("resume may not change cadence or reduce already-used validation budget")
    es.opts["maxfevals"] = new_budget.candidate_evaluations
    _restore_ledger_payload(document, evaluator)
    state_dict = document["state"]
    best = None
    if document["best"] is not None:
        best = C.CandidateResult(**document["best"]["evaluation"])
    state = _State(
        es, new_budget, document["objective_kind"], int(document["seed"]), run_id or document["run_id"],
        initial_theta, best,
        None if best is None else np.asarray(document["best"]["theta"], dtype=np.float64),
        origin, int(state_dict["validation_calls"]), state_dict["next_validation"],
        int(state_dict["generations"]), int(state_dict["invalid_candidates"]), list(document["history"]),
        selection=selection, initial_candidate_id=initial_id, initial_candidate=initial,
        resumed_from=document["payload_hash"], previous_evaluations=origin,
        forward_calls=int(state_dict.get("forward_calls", 0)),
        backward_calls=int(state_dict.get("backward_calls", 0)),
    )
    if document["selection_thresholds"] is not None:
        state.selection_limits = _restore_hard_selection(document, selection)
        if state.selection_limits.metadata() != document["selection_thresholds"]:
            raise BlackBoxError("checkpoint selection identity differs")
    saved_rank = document.get("initial_candidate_rank")
    state.initial_candidate_rank = (None if saved_rank is None or state.selection_limits is None
                                    else _hard_rank(initial, state.selection_limits))
    if state.initial_candidate_rank != (None if saved_rank is None else tuple(saved_rank)):
        raise BlackBoxError("checkpoint initialization rank differs")
    if document["ledger_snapshot"]["candidate_evaluations"] != origin:
        raise BlackBoxError("checkpoint ledger count does not match its optimization state")
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
                                objective=evaluator.spec, simulator=evaluator.sim,
                                budgets={
                                    "limits": dataclasses.asdict(result.budget),
                                    "spent": {"candidate_evaluations": result.evaluations,
                                              "validation_calls": result.validation_calls,
                                              "generations": result.generations},
                                },
                                repository=repository)
    config["hard_selection"] = {"policy": dataclasses.asdict(result.selection),
                                "resolved": result.selection_thresholds}
    config["hard_selection_config_hash"] = (None if result.selection_thresholds is None
                                            else result.selection_thresholds["config_hash"])
    initial_positions = evaluator.parameterization.positions(result.initial_theta)
    best_positions = evaluator.parameterization.positions(result.best_theta)
    files = dict(posting.files)
    files.update({
        "config.json": R.json_bytes(config),
        "history.json": R.json_bytes({"generations": result.history,
                                      "ledger": evaluator.ledger.snapshot()}),
        "metrics.json": R.json_bytes({"best_candidate": _candidate_dict(result.best),
                                      "termination_reason": result.termination_reason,
                                      "invalid_candidates": result.invalid_candidates,
                                      "selection": _selection_metadata(result)}),
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
