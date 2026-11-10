"""Minimal Adam optimization of delta actions; hard-carve checkpoint selection.

Run with ``uv run python -m algorithms.optimize --output runs/core-smoke``.
Input trajectories and output positions use normalized stock coordinates.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import taichi as ti

from benchmarks import contracts as C
from . import blackbox as B
from algorithms import run_artifacts as R
from algorithms import continuation as M

from cam import MachineConfig, trajectory_to_gcode
from simulator.csg_metrics import dice_score, sdf_to_mask
from simulator.csg_simulator import CSGSimulatorDelta


def optimize(args):
    if args.steps < 2 or args.iters < 1 or min(args.k_start, args.k_end, args.lr, args.dt) <= 0:
        raise ValueError("steps >= 2, iters >= 1 and positive k/lr/dt required")
    if args.resolution < 4:
        raise ValueError("resolution must be at least 4")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    rng = np.random.default_rng(args.seed)
    if args.initial is not None:
        path = R.simulator_array(args, np.load(args.initial))
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 3 or not np.isfinite(path).all():
            raise ValueError("initial must contain finite (N >= 3, 3) positions")
        start, deltas = path[0], np.diff(path, axis=0)
    else:
        # Small random moves are a wiring demo, not a CAM initialization.
        start = R.simulator_array(args, [0.15, 0.15, 0.85])
        deltas = R.simulator_array(args, rng.normal(0, 0.025, (args.steps, 3)))
    ti.init(arch=ti.cpu if args.cpu else ti.gpu, offline_cache=False,
            default_fp=ti.f64 if args.precision == "f64" else ti.f32)
    sim = CSGSimulatorDelta(resolution=args.resolution, max_steps=len(deltas),
                            stock_size_in=(1, 1, 1), target_shape=args.target,
                            tool_start=tuple(start), dt=args.dt, init_taichi=False,
                            precision=args.precision)
    sim.tool_radius[None] = 3.175
    sim.tool_height[None] = 25.0
    sim.set_target_params(radius_mm=9.0, height_mm=18.0, half_size_mm=9.0)
    sim.bake_target_grid()
    target = sdf_to_mask(sim.target.to_numpy())
    initial = np.vstack([start, start + np.cumsum(deltas, axis=0)])
    m, v = np.zeros_like(deltas), np.zeros_like(deltas)
    best, history = None, []
    for iteration in range(args.iters + 1):
        k = args.k_start + (args.k_end - args.k_start) * iteration / args.iters
        sim.k[None] = k
        sim.tool_delta.from_numpy(R.simulator_array(sim, deltas))
        with ti.ad.Tape(loss=sim.loss):
            sim.forward(len(deltas) + 1)
        loss = float(sim.loss[None])
        grad = sim.tool_delta.grad.to_numpy()
        if not np.isfinite(loss) or not np.isfinite(grad).all():
            raise FloatingPointError(f"non-finite loss/gradient at iteration {iteration}")
        soft_dice = float(dice_score(sdf_to_mask(sim.stock.to_numpy()[-1]), target))
        sim.forward_hard(len(deltas) + 1)
        mask = sdf_to_mask(sim.stock.to_numpy()[-1])
        hard_dice = float(dice_score(mask, target))
        union = np.logical_or(mask, target).sum()
        metrics = dict(iteration=iteration, k=k, loss=loss, soft_dice=soft_dice,
                       hard_dice=hard_dice, iou=float(np.logical_and(mask, target).sum()/max(1, union)),
                       gouge_stock_fraction=float(np.mean(target & ~mask)),
                       residual_stock_fraction=float(np.mean(~target & mask)),
                       gradient_norm=float(np.linalg.norm(grad)))
        history.append(metrics)
        if best is None or hard_dice > best[0]['hard_dice']:
            best = (metrics.copy(), deltas.copy(), sim.tool_pos.to_numpy().copy())
        if iteration == args.iters:
            break
        grad *= min(1.0, 0.5 / max(float(np.linalg.norm(grad)), 1e-12))
        m = 0.9*m + 0.1*grad
        v = 0.999*v + 0.001*grad*grad
        t = iteration + 1
        deltas -= args.lr * (m/(1-0.9**t)) / (np.sqrt(v/(1-0.999**t)) + 1e-8)
    args.output.mkdir(parents=True)
    np.save(args.output/'initial_trajectory.npy', initial)
    np.save(args.output/'trajectory.npy', best[2])
    np.save(args.output/'trajectory_deltas.npy', best[1])
    (args.output/'metrics.json').write_text(json.dumps(best[0], indent=2)+'\n')
    (args.output/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(stock_size_in=[1, 1, 1], tool_radius_mm=3.175, tool_height_mm=25.0,
                  target_radius_mm=9.0, target_height_mm=18.0, target_half_size_mm=9.0,
                  simulator=R.simulator_precision_metadata(sim))
    (args.output/'args.json').write_text(json.dumps(config, indent=2)+'\n')
    for post in ('rs274', 'haas'):
        (args.output/f'{post}.nc').write_text(trajectory_to_gcode(best[2], MachineConfig(stock_size_in=(1,1,1)), post=post))
    print(json.dumps(best[0], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial', type=Path, help='CAM seed trajectory (.npy); otherwise random smoke initialization')
    p.add_argument('--target', choices=['sphere', 'cylinder', 'box', 'pyramid', 'sphere_hole', 'sphere_bowl'], default='sphere')
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--precision', choices=['f32', 'f64'], default='f64')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--resolution', type=int, default=16)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--dt', type=float, default=0.45)
    p.add_argument('--k-start', type=float, default=20)
    p.add_argument('--k-end', type=float, default=120)
    p.add_argument('--cpu', action='store_true')
    optimize(p.parse_args())


# ---------------------------------------------------------------------------
# T3.2: shared-objective Adam with explicit budgets and checkpoint resumption.
#
# The legacy optimize(args) above remains the small smoke CLI. New integrations
# should use solve_adam/resume_adam/replay_adam_checkpoint below. Candidate starts
# should be configured with sim.set_tool_start(...), never direct writes followed
# by reset_candidate_state(), because the current official reset restores the
# configured baseline. Fixed-k and annealed schedules are declared up front;
# hard selection is a finite-marker violation/residual ordering, not a Dice race.


# Public Adam name retained; implementation is shared with CMA-ES so the
# feasibility gates, alternative infeasible orderings and thresholds cannot drift.
HardSelectionPolicy = B.HardSelectionPolicy
ResolvedHardSelection = B.ResolvedHardSelection


@dataclasses.dataclass(frozen=True)
class AdamConfig:
    mode: str
    learning_rate: float = 0.005
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_gradient_norm: Optional[float] = 0.5
    k_start: float = 20.0
    k_end: float = 20.0
    selection: HardSelectionPolicy = dataclasses.field(default_factory=HardSelectionPolicy)
    n_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "annealed"):
            raise ValueError("mode must be fixed or annealed")
        for name in ("learning_rate", "beta1", "beta2", "epsilon", "k_start", "k_end"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.epsilon <= 0 or self.k_start <= 0 or self.k_end <= 0:
            raise ValueError("learning rate, epsilon and k values must be positive")
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("Adam beta1/beta2 must be in (0,1)")
        if self.max_gradient_norm is not None and self.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive or None")
        if self.mode == "fixed" and self.k_start != self.k_end:
            raise ValueError("fixed mode requires k_start == k_end")
        if self.n_steps is not None and (not isinstance(self.n_steps, int) or self.n_steps < 1):
            raise ValueError("n_steps must be null or a positive integer")


def _required_budget(budget: B.OptimizationBudget) -> None:
    if budget.population_size is not None:
        raise ValueError("Adam uses one gradient candidate per iteration; population_size must be None")


@dataclasses.dataclass
class AdamResult:
    run_id: str
    seed: int
    budget: B.OptimizationBudget
    config: AdamConfig
    best: Optional[C.CandidateResult]
    best_theta: Optional[np.ndarray]
    current_theta: np.ndarray
    initial_theta: np.ndarray
    moments: np.ndarray
    velocities: np.ndarray
    evaluations: int
    forward_calls: int
    backward_calls: int
    validation_calls: int
    gradient_updates: int
    invalid_evaluations: int
    termination_reason: str
    history: list[dict[str, Any]]
    checkpoint_path: Optional[str] = None
    previous_evaluations: int = 0
    selection: HardSelectionPolicy = dataclasses.field(default_factory=HardSelectionPolicy)
    initial_candidate_id: Optional[str] = None
    initial_candidate: Optional[C.CandidateResult] = None
    initial_candidate_rank: Optional[tuple] = None
    initial_candidate_status: str = "unavailable"
    selection_thresholds: Optional[dict[str, Any]] = None
    no_improvement_over_initialization: bool = False
    selection_reason: str = "no hard initialization evaluation"

    @property
    def deterministic_replicates(self) -> int:
        """One deterministic run, not multiple independent per-tasks seeds."""
        return 1


@dataclasses.dataclass
class _AdamState:
    config: AdamConfig
    budget: B.OptimizationBudget
    seed: int
    run_id: str
    config_hash: str
    theta: np.ndarray
    initial_theta: np.ndarray
    moments: np.ndarray
    velocities: np.ndarray
    best: Optional[C.CandidateResult]
    best_theta: Optional[np.ndarray]
    selection: HardSelectionPolicy
    selection_limits: Optional[ResolvedHardSelection]
    initial_candidate_id: Optional[str]
    initial_candidate: Optional[C.CandidateResult]
    initial_candidate_rank: Optional[tuple]
    evaluations: int
    gradient_updates: int
    forward_calls: int
    backward_calls: int
    validation_calls: int
    next_validation: Optional[int]
    invalid_evaluations: int
    history: list[dict[str, Any]]
    previous_evaluations: int = 0


def _config_hash(config: AdamConfig) -> str:
    return C.content_hash(dataclasses.asdict(config))


def _initial_state(evaluator: Any, budget: B.OptimizationBudget, config: AdamConfig,
                   seed: int, x0: Any, run_id: str) -> _AdamState:
    if config.mode == "annealed" and config.n_steps is None:
        steps = (budget.max_generations if budget.max_generations is not None
                 else budget.candidate_evaluations - 1)
        if steps < 1:
            raise ValueError("annealed mode needs a positive declared n_steps/budget")
        # Resolve once at initialization, freeze into config and checkpoint hash.
        config = dataclasses.replace(config, n_steps=steps)
    theta, _, _ = B._options(evaluator, seed, 0.1, x0, None)
    theta = np.asarray(theta, dtype=np.float64)
    return _AdamState(
        config, budget, seed, run_id, _config_hash(config), theta, theta.copy(),
        np.zeros(theta.size), np.zeros(theta.size), None, None,
        config.selection, None, None, None, None,
        1, 0, 0, 0, 0,
        budget.validation_interval if budget.max_validation_calls > 0 else None, 0, [],
    )


def _k_for_step(config: AdamConfig, step: int, budget: B.OptimizationBudget) -> float:
    if config.mode == "fixed":
        return config.k_start
    total = config.n_steps
    if total is None:
        raise ValueError("annealing requires n_steps in the resolved configuration")
    fraction = 1.0 if total <= 1 else min(1.0, step / (total - 1))
    return config.k_start + fraction * (config.k_end - config.k_start)


_hard_rank = B._hard_rank


def _pack_continuation_observation(observation: Optional[M.PairedObservation]) -> Optional[dict[str, Any]]:
    if observation is None:
        return None
    def vector(value):
        return None if value is None else np.asarray(value, dtype=np.float64).tolist()
    return {
        "context": observation.context.as_dict(),
        "soft": {
            "raw_phi_soft": observation.soft.raw_phi_soft,
            "path": vector(observation.soft.path), "gradient": vector(observation.soft.gradient),
            "theta_hash": observation.soft.theta_hash, "notes": list(observation.soft.notes),
        },
        "hard": {
            "coverage_complete": observation.hard.coverage_complete,
            "gouge_fraction": observation.hard.gouge_fraction,
            "residual_fraction": observation.hard.residual_fraction,
            "local_surface_error_mm": observation.hard.local_surface_error_mm,
            "holder_overlap_fraction": observation.hard.holder_overlap_fraction,
            "safety_pass": observation.hard.safety_pass,
            "raw_phi_hard": observation.hard.raw_phi_hard,
            "path": vector(observation.hard.path), "theta_hash": observation.hard.theta_hash,
            "source_valid": observation.hard.source_valid, "notes": list(observation.hard.notes),
        },
    }


def _unpack_continuation_observation(document: Mapping[str, Any], grid: C.GridSpec,
                                     ) -> Optional[M.PairedObservation]:
    packed = document["continuation_state"].get("observation")
    if packed is None:
        return None
    context = _resume_context(document, grid)
    soft_record, hard_record = packed["soft"], packed["hard"]
    soft = M.SoftObservation(
        context, soft_record["raw_phi_soft"], path=soft_record["path"],
        gradient=soft_record["gradient"], theta_hash=soft_record["theta_hash"],
        notes=tuple(soft_record["notes"]),
    )
    hard = M.HardObservation(
        context, hard_record["coverage_complete"], hard_record["gouge_fraction"],
        hard_record["residual_fraction"], hard_record["local_surface_error_mm"],
        hard_record["holder_overlap_fraction"], hard_record["safety_pass"],
        hard_record["raw_phi_hard"], path=hard_record["path"],
        theta_hash=hard_record["theta_hash"], source_valid=hard_record["source_valid"],
        notes=tuple(hard_record["notes"]),
    )
    return M.PairedObservation(soft, hard)


def _continuation_checkpoint_payload(state: _ContinuationState, evaluator: Any) -> dict[str, Any]:
    context = (state.observation.context if state.observation is not None else
               _continuation_context(state, evaluator,
                                     _k_for_step(state.config, state.gradient_updates, state.budget)))
    return {
        "schema": "diffcam-continuation-state/1",
        "config": state.continuation_config.metadata()["config"],
        "config_sha256": state.continuation_config.config_hash,
        "loss_shift": state.loss_shift, "learning_rate": state.learning_rate,
        "last_calibration_key": state.last_calibration_key,
        "calibration_events": state.calibration_events, "retry_events": state.retry_events,
        "role_records": state.role_records, "purpose_spent": state.purpose_spent,
        "mechanism_counts": state.mechanism_counts,
        "observation": _pack_continuation_observation(state.observation),
        "hard_result": None if state.hard_result is None else B._candidate_dict(state.hard_result),
        "observation_context": context.as_dict(),
    }


def _write_adam_checkpoint(path: Path, state: _AdamState, evaluator: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "diffcam-adam-checkpoint/2", "contract_schema": C.SCHEMA_VERSION,
        "run_id": state.run_id, "seed": state.seed, "optimizer_config": dataclasses.asdict(state.config),
        "optimizer_config_hash": state.config_hash, "task_setup_hash": B._setup_signature(evaluator, "smooth"),
        "simulator_precision": R.simulator_precision_metadata(getattr(evaluator, "sim", None)),
        "budget": dataclasses.asdict(state.budget),
        "theta": state.theta.tolist(), "moments": state.moments.tolist(), "velocities": state.velocities.tolist(),
        "initial_theta": state.initial_theta.tolist(),
        "best": B._best_candidate_dict(state.best, state.best_theta),
        "selection_thresholds": (None if state.selection_limits is None
                                 else state.selection_limits.metadata()),
        "initial_candidate": B._initial_candidate_dict(state.initial_candidate, state.initial_theta,
                                                        state.initial_candidate_id),
        "initial_candidate_rank": state.initial_candidate_rank,
        "evaluations": state.evaluations, "gradient_updates": state.gradient_updates,
        "forward_calls": state.forward_calls, "backward_calls": state.backward_calls,
        "validation_calls": state.validation_calls, "next_validation": state.next_validation,
        "invalid_evaluations": state.invalid_evaluations,
        "budget_at_checkpoint": _k_for_step(state.config, max(0, state.gradient_updates), state.budget),
        "next_k": _k_for_step(state.config, state.gradient_updates, state.budget),
        "history": state.history, "previous_evaluations": state.previous_evaluations,
        "ledger_snapshot": evaluator.ledger.snapshot(),
    }
    if isinstance(state, _ContinuationState):
        payload["continuation_state"] = _continuation_checkpoint_payload(state, evaluator)
    digest = C.content_hash(payload)
    R.ArtifactWriter._replace_bytes(path, R.json_bytes(dict(payload, payload_hash=digest)))
    return digest


def _read_adam_checkpoint(path: Path, evaluator: Any) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B.BlackBoxError(f"cannot read Adam checkpoint: {error}") from error
    if document.get("schema") not in ("diffcam-adam-checkpoint/2", "diffcam-adam-checkpoint/1"):
        raise B.BlackBoxError("unrecognized Adam checkpoint schema")
    payload = dict(document)
    expected_hash = payload.pop("payload_hash")
    if expected_hash != C.content_hash(payload):
        raise B.BlackBoxError("Adam checkpoint checksum mismatch")
    if document.get("simulator_precision") != R.simulator_precision_metadata(getattr(evaluator, "sim", None)):
        raise B.BlackBoxError("Adam checkpoint simulator-precision identity differs")
    if document["task_setup_hash"] != B._setup_signature(evaluator, "smooth"):
        raise B.BlackBoxError("Adam checkpoint task/setup identity differs")
    if document.get("seed") is None or not isinstance(document["seed"], int):
        raise B.BlackBoxError("missing/invalid Adam seed record")
    return document


def _restore_state(document: Mapping[str, Any], evaluator: Any, *, budget=None,
                   run_id=None) -> _AdamState:
    if document.get("schema") != "diffcam-adam-checkpoint/2":
        raise B.BlackBoxError("legacy Adam checkpoint requires explicit coordinator migration; no init was re-evaluated")
    old_budget = B.OptimizationBudget(**document["budget"])
    budget = budget or old_budget
    origin = int(document["evaluations"])
    if origin < 1 or budget.candidate_evaluations < origin:
        raise B.BlackBoxError("resume budget cannot reduce prior evaluation count")
    if (budget.validation_interval != old_budget.validation_interval
            or budget.max_validation_calls < document["validation_calls"]):
        raise B.BlackBoxError("resume cadence must be preserved and used validation budget not reduced")
    raw_config = dict(document["optimizer_config"])
    if raw_config["mode"] == "annealed" and raw_config.get("n_steps") is None:
        raise B.BlackBoxError("checkpoint has no declared annealing n_steps; refusing to change it on resume")
    if isinstance(raw_config.get("selection"), Mapping):
        raw_config["selection"] = HardSelectionPolicy(**raw_config["selection"])
    config = AdamConfig(**raw_config)
    if C.content_hash(dataclasses.asdict(config)) != document["optimizer_config_hash"]:
        raise B.BlackBoxError("optimizer moment/config schedule hash mismatch")
    initial_theta = np.asarray(document["initial_theta"], dtype=float)
    initial_id, initial = B._restore_initial_candidate(document, initial_theta)
    state = _AdamState(config, budget, int(document["seed"]), run_id or document["run_id"],
                       _config_hash(config), np.asarray(document["theta"], dtype=float).copy(),
                       initial_theta,
                       np.asarray(document["moments"], dtype=float).copy(),
                       np.asarray(document["velocities"], dtype=float).copy(), None, None,
                       config.selection, None, initial_id, initial, None,
                       origin, int(document["gradient_updates"]), int(document["forward_calls"]),
                       int(document["backward_calls"]), int(document["validation_calls"]),
                       document["next_validation"], int(document["invalid_evaluations"]),
                       list(document["history"]), previous_evaluations=origin)
    if state.theta.shape != state.moments.shape or state.theta.shape != state.velocities.shape:
        raise B.BlackBoxError("moment dimensions do not match theta")
    if state.theta.size != evaluator.parameterization.num_params:
        raise B.BlackBoxError("checkpoint theta dimension differs from parameterization")
    if not np.all(np.isfinite(state.theta)) or not np.all(np.isfinite(state.moments)) or not np.all(np.isfinite(state.velocities)):
        raise B.BlackBoxError("non-finite optimizer state rejected")
    if document["best"] is not None:
        state.best = C.CandidateResult(**document["best"]["evaluation"])
        state.best_theta = np.asarray(document["best"]["theta"], dtype=float)
    if document["selection_thresholds"] is not None:
        state.selection_limits = B._restore_hard_selection(document, config.selection)
        if state.selection_limits.metadata() != document["selection_thresholds"]:
            raise B.BlackBoxError("Adam checkpoint selection identity differs")
    saved_rank = document.get("initial_candidate_rank")
    state.initial_candidate_rank = (None if saved_rank is None or state.selection_limits is None
                                    else _hard_rank(initial, state.selection_limits))
    if state.initial_candidate_rank != (None if saved_rank is None else tuple(saved_rank)):
        raise B.BlackBoxError("Adam checkpoint initialization rank differs")
    if document["ledger_snapshot"]["candidate_evaluations"] != origin:
        raise B.BlackBoxError("Adam checkpoint ledger count does not match its optimizer state")
    return state


def _validation_and_hard_select(state: _AdamState, evaluator: Any) -> None:
    state.validation_calls += 1
    evaluator.ledger.start_validation()
    before = evaluator.ledger.snapshot()
    result = evaluator.evaluate_candidate(state.theta, f"{state.run_id}:validate:{state.validation_calls}", "hard")
    after = evaluator.ledger.snapshot()
    state.forward_calls += after["forward_calls"] - before["forward_calls"]
    state.backward_calls += after["backward_calls"] - before["backward_calls"]
    B._refine_hard_selection(state, evaluator, result)
    B._select_hard_candidate(state, result, state.theta.copy())
    return result


def _run_adam_body(state: _AdamState, evaluator: Any, checkpoint_dir: Optional[Path]) -> AdamResult:
    if state.initial_candidate is None:
        B._evaluate_initial_candidate(state, evaluator)
    if state.previous_evaluations == 0 and not state.initial_candidate.valid:
        state.invalid_evaluations += 1
    reason = "budget_exhausted" if state.evaluations >= state.budget.candidate_evaluations else None
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = checkpoint_dir / f"ckpt-{state.evaluations:06d}.json"
        _write_adam_checkpoint(checkpoint_path, state, evaluator)
    while state.evaluations < state.budget.candidate_evaluations:
        if (state.budget.max_generations is not None
                and state.gradient_updates >= state.budget.max_generations):
            reason = "max_generations"
            break
        k = _k_for_step(state.config, state.gradient_updates, state.budget)
        if hasattr(evaluator, "set_evaluation_k"):
            evaluator.set_evaluation_k(k)
        else:
            evaluator.sim.k[None] = k
        candidate_id = f"{state.run_id}:iter:{state.evaluations}"
        evaluated_theta = state.theta.copy()
        before = evaluator.ledger.snapshot()
        evaluation = evaluator.evaluate_candidate_with_gradient(state.theta, candidate_id)
        after = evaluator.ledger.snapshot()
        forward_delta = after["forward_calls"] - before["forward_calls"]
        backward_delta = after["backward_calls"] - before["backward_calls"]
        state.forward_calls += forward_delta
        state.backward_calls += backward_delta
        state.evaluations += 1
        gradient = evaluation.gradient_theta
        clip_scale = None
        if evaluation.result.valid and gradient is not None and np.all(np.isfinite(gradient)):
            grad_norm = float(np.linalg.norm(gradient))
            clip_scale = min(1.0, state.config.max_gradient_norm / max(grad_norm, 1e-30)) if state.config.max_gradient_norm is not None else 1.0
            gradient = gradient * clip_scale
            step = state.gradient_updates + 1
            state.moments = state.config.beta1 * state.moments + (1.0 - state.config.beta1) * gradient
            state.velocities = state.config.beta2 * state.velocities + (1.0 - state.config.beta2) * gradient ** 2
            unbiased_m = state.moments / (1.0 - state.config.beta1 ** step)
            unbiased_v = state.velocities / (1.0 - state.config.beta2 ** step)
            theta = state.theta - state.config.learning_rate * unbiased_m / (np.sqrt(unbiased_v) + state.config.epsilon)
            state.theta = np.asarray(evaluator.parameterization.clip(theta), dtype=float)
            state.gradient_updates = step
        else:
            # Do not spend the full budget re-evaluating an identical bad point.
            # Record the failed attempt, then terminate with the best prior valid
            # selection (if any) rather than inventing a new incumbent.
            state.invalid_evaluations += 1
            state.history.append({
                "iteration": state.evaluations, "k": k,
                "theta": evaluated_theta.tolist(), "theta_after_step": state.theta.tolist(),
                "soft_objective": evaluation.result.objective, "clip_scale": None,
                "valid": False, "forward_calls": forward_delta,
                "backward_calls": backward_delta,
            })
            reason = "invalid_evaluation"
            break
        state.history.append({
            "iteration": state.evaluations, "k": k, "theta": evaluated_theta.tolist(),
            "theta_after_step": state.theta.tolist(),
            "soft_objective": evaluation.result.objective, "clip_scale": clip_scale,
            "valid": evaluation.result.valid,
            "forward_calls": forward_delta, "backward_calls": backward_delta,
        })
        if (state.next_validation is not None
                and state.validation_calls < state.budget.max_validation_calls
                and state.evaluations >= state.next_validation):
            selected = _validation_and_hard_select(state, evaluator)
            state.next_validation += state.budget.validation_interval
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"ckpt-{state.evaluations:06d}.json"
            _write_adam_checkpoint(checkpoint_path, state, evaluator)
    if reason is None:
        reason = "budget_exhausted"
    if checkpoint_dir is not None and state.evaluations > 0:
        checkpoint_path = checkpoint_dir / "final.json"
        _write_adam_checkpoint(checkpoint_path, state, evaluator)
    initial_status = B._initial_status(state.initial_candidate, state.initial_candidate_rank)
    no_improvement = (state.best is not None and state.initial_candidate_id is not None
                      and state.best.candidate_id == state.initial_candidate_id)
    return AdamResult(state.run_id, state.seed, state.budget, state.config,
                      state.best, state.best_theta, state.theta.copy(),
                      state.initial_theta.copy(), state.moments.copy(), state.velocities.copy(),
                      state.evaluations, state.forward_calls, state.backward_calls,
                      state.validation_calls, state.gradient_updates, state.invalid_evaluations,
                      reason, state.history, str(checkpoint_path) if checkpoint_path else None,
                      state.previous_evaluations, selection=state.selection,
                      initial_candidate_id=state.initial_candidate_id,
                      initial_candidate=state.initial_candidate,
                      initial_candidate_rank=state.initial_candidate_rank,
                      initial_candidate_status=initial_status,
                      selection_thresholds=(None if state.selection_limits is None
                                            else state.selection_limits.metadata()),
                      no_improvement_over_initialization=no_improvement,
                      selection_reason=B._selection_reason(state.best, state.initial_candidate_id or "", initial_status))


def _run_adam(state: _AdamState, evaluator: Any, checkpoint_dir: Optional[Path]) -> AdamResult:
    try:
        return _run_adam_body(state, evaluator, checkpoint_dir)
    finally:
        if hasattr(evaluator, "clear_evaluation_k"):
            evaluator.clear_evaluation_k()
        else:
            # Keep the shared evaluator's specification for the next method.
            if hasattr(evaluator.spec, "k"):
                evaluator.sim.k[None] = evaluator.spec.k
        # Mirror CMA's neutral per-run override cleanup. A continuation may set
        # this explicitly before its run, but baselines never inherit it.
        clear_loss_shift = getattr(evaluator, "clear_evaluation_loss_shift", None)
        if clear_loss_shift is not None:
            clear_loss_shift()


def solve_adam(evaluator: Any, budget: B.OptimizationBudget, config: AdamConfig, *, seed: int = 0,
               x0: Any = None, run_id: Optional[str] = None, checkpoint_dir: Optional[str | Path] = None) -> AdamResult:
    _required_budget(budget)
    if budget.candidate_evaluations < 1:
        raise ValueError("Adam hard initialization requires at least one candidate evaluation")
    if not isinstance(config.selection, HardSelectionPolicy):
        raise TypeError("config.selection must be a HardSelectionPolicy")
    # Plain Adam is a neutral field-shift baseline. An intentionally continued
    # optimizer must re-establish its explicit state rather than rely on leaked
    # ambient shift; checkpoint hashes include the active evaluation shift.
    clear_loss_shift = getattr(evaluator, "clear_evaluation_loss_shift", None)
    if clear_loss_shift is not None:
        clear_loss_shift()
    run_id = run_id or f"adam-{config.mode}-deterministic"
    state = _initial_state(evaluator, budget, config, seed, x0, run_id)
    return _run_adam(state, evaluator, Path(checkpoint_dir) if checkpoint_dir is not None else None)


def resume_adam(path: str | Path, evaluator: Any, *, budget: Optional[B.OptimizationBudget] = None,
                run_id: Optional[str] = None, checkpoint_dir: Optional[str | Path] = None) -> AdamResult:
    _required_budget(budget or B.OptimizationBudget(**json.loads(Path(path).read_text())["budget"]))
    document = _read_adam_checkpoint(Path(path), evaluator)
    state = _restore_state(document, evaluator, budget=budget, run_id=run_id)
    B._restore_ledger_payload(document, evaluator)
    return _run_adam(state, evaluator, Path(checkpoint_dir) if checkpoint_dir is not None else None)


def replay_adam_checkpoint(path: str | Path, evaluator: Any) -> dict[str, Any]:
    """Re-score the saved hard-selected theta, without changing optimizer moments."""
    document = _read_adam_checkpoint(Path(path), evaluator)
    if document["best"] is None:
        raise B.BlackBoxError("no hard-selected theta to replay")
    if _restore_state(document, evaluator).config_hash != document["optimizer_config_hash"]:
        raise B.BlackBoxError("replay optimizer configuration mismatch")
    theta = np.asarray(document["best"]["theta"], dtype=float)
    evaluator.ledger.start_validation()
    replayed = evaluator.evaluate_candidate(theta, f"{document['run_id']}:replay", "hard")
    expected = document["best"]["evaluation"]["objective"]
    # Taichi CPU atomic reductions (e.g. holder_overlap_total) are reproducible only to ~1 float32 ulp
    # (review R1), so replay agreement is checked against a declared tolerance and the difference recorded.
    rel_tol, abs_tol = 1e-5, 1e-9
    difference = (None if replayed.objective is None
                  else abs(float(replayed.objective) - float(expected)))
    matches = bool(replayed.valid and difference is not None
                   and difference <= max(abs_tol, rel_tol * abs(float(expected))))
    return {"replayed": replayed, "expected_objective": expected,
            "objective_matches_within_tolerance": matches,
            "objective_abs_difference": difference,
            "replay_tolerance": {"rel": rel_tol, "abs": abs_tol},
            "checkpoint_theta": theta, "checkpoint_hash": document["payload_hash"]}


def emit_adam_artifacts(result: AdamResult, evaluator: Any, run_dir: str | Path,
                        repository: str | Path) -> C.RunRecord:
    if result.best_theta is None or result.best is None or not result.best.valid:
        raise B.BlackBoxError("no hard-selected feasible candidate; not emitting a fake completed run")
    task, motion = evaluator.task, evaluator.candidate_motion(result.best_theta)
    posting = R.post_motion(task, motion)
    config = R.build_run_config(result.run_id, task, motion, seed=result.seed,
                                parameterization=evaluator.parameterization, objective=evaluator.spec,
                                simulator=evaluator.sim,
                                budgets={
                                    "limits": dataclasses.asdict(result.budget),
                                    "spent": {"candidate_evaluations": result.evaluations,
                                              "forward_calls": result.forward_calls,
                                              "backward_calls": result.backward_calls,
                                              "validation_calls": result.validation_calls,
                                              "gradient_updates": result.gradient_updates},
                                }, repository=repository)
    config["hard_selection"] = {"policy": dataclasses.asdict(result.selection),
                                "resolved": result.selection_thresholds}
    config["hard_selection_config_hash"] = (None if result.selection_thresholds is None
                                            else result.selection_thresholds["config_hash"])
    initial = evaluator.parameterization.positions(result.initial_theta)
    best_positions = evaluator.parameterization.positions(result.best_theta)
    files = dict(posting.files)
    files.update({
        "config.json": R.json_bytes(config),
        "history.json": R.json_bytes({"history": result.history}),
        "metrics.json": R.json_bytes({"best_candidate": B._candidate_dict(result.best),
                                      "best_rank": (None if result.best is None or result.selection_thresholds is None
                                                    else _hard_rank(result.best, B._restore_hard_selection({
                                                        "selection_thresholds": result.selection_thresholds
                                                    }, result.selection))),
                                      "termination_reason": result.termination_reason,
                                      "invalid_evaluations": result.invalid_evaluations,
                                      "selection": B._selection_metadata(result)}),
        "motion_plan.json": R.record_bytes(motion),
        "initial_trajectory.npy": R.numpy_bytes(initial),
        "trajectory.npy": R.numpy_bytes(best_positions),
        "trajectory_deltas.npy": R.numpy_bytes(np.diff(best_positions, axis=0)),
    })
    record = R.assemble_record(result.run_id, task, C.RunStatus.INCOMPLETE, posting=posting,
                               artifact_names=tuple(sorted(files)), source_commit=config["source_commit"],
                               dirty_patch_hash=config["dirty_patch_hash"],
                               failure_reason="Adam outputs pending posted evaluation (T1.3)")
    R.ArtifactWriter().write(str(run_dir), record, files)
    return record


# ---------------------------------------------------------------------------
# T3.4c: production continuation-Adam with explicit calibration/backtracking
# costs. (off, off) deliberately delegates to solve_adam; there is no alternate
# arithmetic path to make an ablation approximately match its baseline.


@dataclasses.dataclass(frozen=True)
class ContinuationConfig:
    adam: AdamConfig = dataclasses.field(default_factory=lambda: AdamConfig("annealed", k_start=20.0, k_end=120.0))
    shift: str = "on"
    backtracking: str = "on"
    calibration_eta: float = 0.25
    calibration_delta_max: float = 0.25
    calibration_disagreement_floor: float = 1e-6
    calibration_band: M.SurfaceBand = dataclasses.field(default_factory=lambda: M.SurfaceBand(
        "stock", 0.0, 2.0, "explicit near-surface band in physical millimetres"))
    backtracking_retries: int = 2
    backtracking_shrink: float = 0.5
    backtracking_learning_rate_min: float = 0.0000625
    backtracking_learning_rate_max: float = 0.01
    initial_loss_shift: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.adam, AdamConfig):
            raise TypeError("continuation adam must be an AdamConfig")
        if self.shift not in ("on", "off") or self.backtracking not in ("on", "off"):
            raise ValueError("shift and backtracking switches must be 'on' or 'off'")
        if not isinstance(self.calibration_band, M.SurfaceBand):
            raise TypeError("calibration_band must be a physical SurfaceBand")
        if isinstance(self.initial_loss_shift, (bool, np.bool_)) or not math.isfinite(self.initial_loss_shift):
            raise ValueError("initial_loss_shift must be finite")
        if abs(self.initial_loss_shift) > 0.5:
            raise ValueError("initial_loss_shift must be at most 0.5 voxel")
        # Reuse, rather than duplicate, bounded retry/EMA/k validation.
        cfg = self.retry_config
        if not cfg.learning_rate_min <= self.adam.learning_rate <= cfg.learning_rate_max:
            raise ValueError("baseline Adam learning rate must be within retry bounds")

    @property
    def retry_config(self) -> M.BacktrackingConfig:
        return M.BacktrackingConfig(
            learning_rate_min=self.backtracking_learning_rate_min,
            learning_rate_max=self.backtracking_learning_rate_max,
            learning_rate_shrink=self.backtracking_shrink,
            max_retries_per_step=self.backtracking_retries,
            k_min=min(self.adam.k_start, self.adam.k_end),
            k_max=max(self.adam.k_start, self.adam.k_end),
            eta=self.calibration_eta, delta_max=self.calibration_delta_max,
            disagreement_floor=self.calibration_disagreement_floor,
        )

    @property
    def config_hash(self) -> str:
        return C.content_hash(dataclasses.asdict(self))

    def metadata(self) -> dict[str, Any]:
        return {"config": dataclasses.asdict(self), "config_sha256": self.config_hash}


@dataclasses.dataclass
class ContinuationResult(AdamResult):
    continuation_config: Optional[dict[str, Any]] = None
    continuation_config_hash: Optional[str] = None
    loss_shift: float = 0.0
    calibration_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    retry_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    purpose_spent: dict[str, int] = dataclasses.field(default_factory=dict)
    retry_state: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def condition(self) -> str:
        return "continuation-adam"


@dataclasses.dataclass
class _ContinuationState(_AdamState):
    continuation_config: Optional[ContinuationConfig] = None
    loss_shift: float = 0.0
    learning_rate: Optional[float] = None
    observation: Optional[M.PairedObservation] = None
    last_calibration_key: Optional[dict[str, Any]] = None
    calibration_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    retry_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    role_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    purpose_spent: dict[str, int] = dataclasses.field(default_factory=dict)
    hard_result: Optional[C.CandidateResult] = None
    mechanism_counts: dict[str, int] = dataclasses.field(default_factory=dict)


def _continuation_context(state: _ContinuationState, evaluator: Any, k: float,
                          shift: Optional[float] = None) -> M.EvaluationContext:
    shift = state.loss_shift if shift is None else float(shift)
    evaluator_id = f"{type(evaluator).__module__}.{type(evaluator).__name__}"
    version = getattr(evaluator, "evaluator_revision", None)
    if version is None:
        # Source identity guards against mistaking a class-name match for the
        # same hard evaluator implementation across worktrees/revisions.
        source = Path(inspect.getfile(type(evaluator))).resolve().read_bytes()
        version = C.sha256_bytes(source)
    rank_source = Path(inspect.getsourcefile(B._hard_rank)).resolve().read_bytes()
    return M.EvaluationContext(
        evaluator.task.grid, float(k), shift, evaluator_id,
        {"hard_setup_sha256": B._setup_signature(evaluator, "hard"),
         "evaluator_revision": version,
         "shared_hard_rank_sha256": C.sha256_bytes(rank_source),
         "validator": "shared CandidateResult + algorithms.blackbox._hard_rank"},
    )


def _resume_context(document: Mapping[str, Any], grid: C.GridSpec) -> M.EvaluationContext:
    record = document["continuation_state"]["observation_context"]
    if record["grid_sha256"] != M.grid_signature(grid):
        raise B.BlackBoxError("continuation checkpoint grid differs")
    return M.EvaluationContext(
        grid, record["k"], record["loss_shift"], record["validator_id"],
        record["validator_settings"],
    )


def _actual_context(state: _ContinuationState, evaluator: Any,
                    received: M.EvaluationContext) -> M.EvaluationContext:
    """See the callback's actual overrides, rather than assume it obeyed them."""
    k = getattr(evaluator, "_evaluation_k_override", received.k)
    if k is None:
        k = received.k
    shift = float(getattr(evaluator, "evaluation_loss_shift", received.loss_shift) or 0.0)
    return dataclasses.replace(received, k=float(k), loss_shift=shift)


def _apply_continuation_context(state: _ContinuationState, evaluator: Any,
                                context: M.EvaluationContext) -> None:
    if hasattr(evaluator, "set_evaluation_k"):
        evaluator.set_evaluation_k(context.k)
    else:
        evaluator.sim.k[None] = context.k
    if state.continuation_config.shift == "on":
        setter = getattr(evaluator, "set_evaluation_loss_shift", None)
        if setter is None:
            raise AttributeError("signed continuation requires evaluator.set_evaluation_loss_shift")
        setter(context.loss_shift)
    else:
        clear = getattr(evaluator, "clear_evaluation_loss_shift", None)
        if clear is not None:
            clear()
        elif hasattr(evaluator, "set_evaluation_loss_shift"):
            evaluator.set_evaluation_loss_shift(0.0)


def _last_simulator_field(evaluator: Any, steps: int) -> Any:
    """Copy the actual last raw stock field; never synthesize an occupancy EDT."""
    sim = getattr(evaluator, "sim", None)
    if sim is None:
        return None
    field = getattr(sim, "stock", getattr(sim, "stock_d", None))
    target = getattr(sim, "target", None)
    path = getattr(sim, "tool_pos", None)
    try:
        stock = field.to_numpy() if field is not None else None
        target = target.to_numpy() if target is not None else None
        path = path.to_numpy() if path is not None else None
    except (AttributeError, TypeError, ValueError):
        return None
    if stock is None or target is None or path is None:
        return None
    stock = np.asarray(stock)
    if stock.ndim != 4 or not 0 <= steps < stock.shape[0]:
        return None
    return {"stock": stock[steps].copy(), "target": np.asarray(target).copy(),
            "path": np.asarray(path).copy()}


def _observation_key(context: M.EvaluationContext, theta_hash: Optional[str]) -> str:
    return C.content_hash({"context": context.as_dict(), "theta_hash": theta_hash})


class _ProductionBacktracking(M.BacktrackingMechanism):
    """Connect the bounded mechanism to real calls and shared production rank.

    Ordinary candidate cost is the soft callback. Each hard callback is tagged
    as one validation, independently of the native validation cadence. Caches
    are bounded to the latest raw fields; full candidate rank records are small.
    Unknown posted coverage/local-surface/safety fields remain None, but they
    must not replace the D-10 modeled hard-rank constraints.
    """
    def __init__(self, state: _ContinuationState, evaluator: Any):
        self.state, self.evaluator = state, evaluator
        self._role = "incumbent"
        self._rank_cache: dict[str, C.CandidateResult] = {}
        self._field_cache: Optional[tuple[str, Any]] = None
        self._production_hard: Optional[C.CandidateResult] = None
        super().__init__(state.continuation_config.retry_config, M.ConstraintPolicy(),
                         state.budget, self._soft, self._hard)
        self._spent.update(candidate_evaluations=state.evaluations,
                           hard_validation_calls=state.validation_calls,
                           **{name: state.mechanism_counts.get(name, 0)
                              for name in ("proposals", "retries", "accepted")})
        if state.observation is not None and state.hard_result is not None:
            self.retain_observation(state.observation, state.hard_result)

    def retain_observation(self, pair: M.PairedObservation, result: C.CandidateResult) -> None:
        """Restore a rank cache only after the checkpoint/result link is checked."""
        if not isinstance(result, C.CandidateResult) or result.theta_hash != pair.hard.theta_hash:
            raise B.BlackBoxError("retained hard result and observation identities differ")
        self._rank_cache[_observation_key(pair.context, pair.hard.theta_hash)] = result
        self._production_hard = result


    def _record_role(self, *, candidates: int = 0, validations: int = 0,
                     theta: Any, candidate_id: str) -> None:
        if self._role.startswith("calibration"):
            group = "calibration"
        elif self._role.startswith("fresh_incumbent"):
            group = "fresh_incumbent"
        elif self._role == "incumbent_refresh":
            group = "fresh_incumbent"
        elif self._role.startswith("proposal"):
            group = "proposal"
        elif self._role.startswith("retry"):
            group = "retry"
        else:
            group = "incumbent"
        self.state.purpose_spent[f"{group}_candidate_calls"] = (
            self.state.purpose_spent.get(f"{group}_candidate_calls", 0) + candidates)
        self.state.purpose_spent[f"{group}_validation_calls"] = (
            self.state.purpose_spent.get(f"{group}_validation_calls", 0) + validations)
        self.state.role_records.append({
            "role": self._role, "candidate_id": candidate_id,
            "candidate_call_spent": candidates, "hard_validation_call_spent": validations,
            "theta_hash": M.parameter_hash(theta),
        })

    def _soft(self, theta: np.ndarray, context: M.EvaluationContext,
              require_gradient: bool) -> M.SoftObservation:
        _apply_continuation_context(self.state, self.evaluator, context)
        candidate_id = f"{self.state.run_id}:{self._role}:{self.state.gradient_updates}:{self._spent['candidate_evaluations']}"
        before = self.evaluator.ledger.snapshot()
        evaluated = (self.evaluator.evaluate_candidate_with_gradient(theta, candidate_id)
                     if require_gradient else self.evaluator.evaluate_candidate(theta, candidate_id, "smooth"))
        after = self.evaluator.ledger.snapshot()
        if after["candidate_evaluations"] != before["candidate_evaluations"] + 1:
            raise B.BlackBoxError("continuation soft callback must consume one ordinary candidate call")
        self.state.evaluations += 1
        self.state.forward_calls += after["forward_calls"] - before["forward_calls"]
        self.state.backward_calls += after["backward_calls"] - before["backward_calls"]
        self._record_role(candidates=1, theta=theta, candidate_id=candidate_id)
        raw_soft = evaluated.result if hasattr(evaluated, "result") else evaluated
        gradient = None
        if require_gradient:
            gradient = getattr(evaluated, "gradient_theta", None)
            if gradient is not None:
                gradient = np.asarray(gradient, dtype=np.float64).copy()
        actual_context = _actual_context(self.state, self.evaluator, context)
        if raw_soft.objective_kind != "smooth":
            raise M._ContextMismatch("soft callback returned a non-smooth result")
        self._last_soft_fields = _last_simulator_field(self.evaluator, self.evaluator.parameterization.num_steps)
        phi = raw_soft.objective if raw_soft.valid else None
        return M.SoftObservation(actual_context, phi,
                                 path=None if self._last_soft_fields is None else self._last_soft_fields["path"],
                                 gradient=gradient, theta_hash=raw_soft.theta_hash)

    def _hard(self, theta: np.ndarray, context: M.EvaluationContext) -> M.HardObservation:
        candidate_id = f"{self.state.run_id}:{self._role}:hard:{self.state.gradient_updates}:{self._spent['hard_validation_calls']}"
        before = self.evaluator.ledger.snapshot()
        self.evaluator.ledger.start_validation()
        result = self.evaluator.evaluate_candidate(theta, candidate_id, "hard")
        after = self.evaluator.ledger.snapshot()
        if (after["validation_calls"] != before["validation_calls"] + 1
                or after["candidate_evaluations"] != before["candidate_evaluations"]):
            raise B.BlackBoxError("continuation hard callback must consume exactly one validation call")
        self.state.validation_calls += 1
        self.state.forward_calls += after["forward_calls"] - before["forward_calls"]
        self.state.backward_calls += after["backward_calls"] - before["backward_calls"]
        self._record_role(validations=1, theta=theta, candidate_id=candidate_id)
        if self.state.next_validation is not None and self.state.evaluations >= self.state.next_validation:
            interval = self.state.budget.validation_interval
            self.state.next_validation = ((self.state.evaluations // interval) + 1) * interval
        hard_fields = _last_simulator_field(self.evaluator, self.evaluator.parameterization.num_steps)
        observation = M.from_merged_hard(result, context, path=None if hard_fields is None else hard_fields["path"])
        self._production_hard = result
        self._field_cache = None
        soft_fields = getattr(self, "_last_soft_fields", None)
        if soft_fields is not None and hard_fields is not None:
            try:
                pair = M.StockFieldPair(
                    context, hard_fields["stock"], soft_fields["stock"], hard_fields["target"],
                    hard_fields["path"], soft_fields["path"])
                key = _observation_key(context, result.theta_hash)
                self._field_cache = (key, pair)
            except (TypeError, ValueError):
                # Missing/incompatible raw fields remain inconclusive; no EDT or
                # Boolean remaining-stock substitute is allowed.
                self._field_cache = None
        return observation

    def _observe(self, theta, context, with_gradient):
        pair = super()._observe(theta, context, with_gradient)
        result = self._production_hard
        if not isinstance(result, C.CandidateResult):
            raise _ContinuationRankUnknown("hard callback did not produce a typed CandidateResult")
        key = _observation_key(pair.context, pair.hard.theta_hash)
        self._rank_cache[key] = result
        try:
            B._refine_hard_selection(self.state, self.evaluator, result)
            B._select_hard_candidate(self.state, result, theta)
        except B.BlackBoxError as error:
            # A syntactically typed but non-rankable result can never become the
            # incumbent merely because it exposed a favorable scalar loss.
            if not getattr(result, "valid", False):
                raise _ContinuationRankUnknown(str(error))
            raise
        return pair

    def _constraint_assessment(self, state):
        key = _observation_key(state.context, state.theta_hash)
        result = self._rank_cache.get(key)
        rank = None if result is None or self.state.selection_limits is None else _hard_rank(
            result, self.state.selection_limits)
        if rank is None:
            return M.ConstraintAssessment(M.Feasibility.UNKNOWN, 0.0, ("shared_hard_rank_unavailable",), None)
        if any(not math.isfinite(value) for value in rank):
            return M.ConstraintAssessment(M.Feasibility.UNKNOWN, 0.0, ("nonfinite_shared_hard_rank",), None)
        status = M.Feasibility.FEASIBLE if rank[0] == 0 else M.Feasibility.INFEASIBLE
        return M.ConstraintAssessment(status, 0.0, (), tuple(rank))

    def _strictly_better(self, left, right):
        # Production tuples already encode D-10 ordering, including exact
        # task tolerances. Equality is not worsening and can move along a hard
        # plateau; the shared selector still keeps θ0 on its exact tie.
        if right is None:
            return left is not None
        if left is None:
            return False
        return left <= right

    def _incumbent(self, snapshot, context):
        previous = snapshot.observation
        if (previous is not None and previous.context == context
                and previous.soft.gradient is not None):
            return previous
        self._role = "incumbent_refresh"
        # No retained observation also has an explicitly budgeted first pair.
        return self._observe(snapshot.theta, context, True)

    def field_pair(self, theta, context):
        key = _observation_key(context, M.parameter_hash(theta))
        if self._field_cache is not None and self._field_cache[0] == key:
            return self._field_cache[1]
        return None


class _ContinuationRankUnknown(RuntimeError):
    pass


def _adam_proposal(state: _ContinuationState, snapshot: M.OptimizerSnapshot,
                   learning_rate: float) -> M.Proposal:
    gradient = snapshot.observation.soft.gradient
    if gradient is None or not np.all(np.isfinite(gradient)):
        raise _ContinuationRankUnknown("gradient_unavailable")
    config = state.continuation_config.adam
    norm = float(np.linalg.norm(gradient))
    scale = 1.0 if config.max_gradient_norm is None else min(
        1.0, config.max_gradient_norm / max(norm, 1e-30))
    gradient = np.asarray(gradient, dtype=np.float64) * scale
    step = snapshot.optimizer_step + 1
    moments = config.beta1 * snapshot.first_moment + (1.0 - config.beta1) * gradient
    velocities = config.beta2 * snapshot.second_moment + (1.0 - config.beta2) * gradient ** 2
    unbiased_m = moments / (1.0 - config.beta1 ** step)
    unbiased_v = velocities / (1.0 - config.beta2 ** step)
    theta = snapshot.theta - learning_rate * unbiased_m / (np.sqrt(unbiased_v) + config.epsilon)
    theta = np.asarray(state._propose_evaluator.parameterization.clip(theta), dtype=np.float64)
    optimizer_config = dict(snapshot.config, learning_rate=learning_rate,
                            mode=config.mode, loss_shift=snapshot.context.loss_shift,
                            k=snapshot.context.k)
    return M.Proposal(theta, moments, velocities, optimizer_config, snapshot.context, step)


def _calibration_key(state: _ContinuationState, context: M.EvaluationContext) -> dict[str, Any]:
    return {"theta_hash": M.parameter_hash(state.theta), "gradient_updates": state.gradient_updates,
            "k": context.k, "loss_shift": context.loss_shift}


def _continuation_header(state: _ContinuationState, evaluator: Any, checkpoint_dir: Optional[Path]):
    if state.initial_candidate is None:
        B._evaluate_initial_candidate(state, evaluator)
        state.purpose_spent["initialization_candidate_calls"] = 1
        state.invalid_evaluations += int(not state.initial_candidate.valid)
    reason = "budget_exhausted" if state.evaluations >= state.budget.candidate_evaluations else None
    if checkpoint_dir is not None:
        checkpoint = checkpoint_dir / f"ckpt-{state.evaluations:06d}.json"
        _write_adam_checkpoint(checkpoint, state, evaluator)
    else:
        checkpoint = None
    return reason, checkpoint


def _continuation_final(state: _ContinuationState, evaluator: Any, reason: str,
                        checkpoint_dir: Optional[Path], checkpoint_path: Optional[Path]) -> ContinuationResult:
    if checkpoint_dir is not None:
        checkpoint_path = checkpoint_dir / "final.json"
        _write_adam_checkpoint(checkpoint_path, state, evaluator)
    initial_status = B._initial_status(state.initial_candidate, state.initial_candidate_rank)
    no_improvement = (state.best is not None and state.initial_candidate_id is not None
                      and state.best.candidate_id == state.initial_candidate_id)
    metadata = state.continuation_config.metadata()
    return ContinuationResult(
        state.run_id, state.seed, state.budget, state.config, state.best,
        None if state.best_theta is None else state.best_theta.copy(), state.theta.copy(),
        state.initial_theta.copy(), state.moments.copy(), state.velocities.copy(),
        state.evaluations, state.forward_calls, state.backward_calls, state.validation_calls,
        state.gradient_updates, state.invalid_evaluations, reason, state.history,
        None if checkpoint_path is None else str(checkpoint_path), state.previous_evaluations,
        selection=state.selection, initial_candidate_id=state.initial_candidate_id,
        initial_candidate=state.initial_candidate, initial_candidate_rank=state.initial_candidate_rank,
        initial_candidate_status=initial_status, selection_thresholds=(
            None if state.selection_limits is None else state.selection_limits.metadata()),
        no_improvement_over_initialization=no_improvement,
        selection_reason=B._selection_reason(state.best, state.initial_candidate_id or "", initial_status),
        continuation_config=metadata["config"], continuation_config_hash=metadata["config_sha256"],
        loss_shift=state.loss_shift, calibration_events=list(state.calibration_events),
        retry_events=list(state.retry_events), purpose_spent=dict(state.purpose_spent),
        retry_state={"learning_rate": state.learning_rate,
                     **_mechanism_state_counts(state)},
    )


def _mechanism_state_counts(state: _ContinuationState) -> dict[str, int]:
    return {name: int(state.mechanism_counts.get(name, 0))
            for name in ("proposals", "retries", "accepted")}


def _continuation_from_adam(state: _AdamState, config: ContinuationConfig,
                            continuation_state: Optional[Mapping[str, Any]] = None) -> _ContinuationState:
    extra = {
        "continuation_config": config,
        "loss_shift": config.initial_loss_shift,
        "learning_rate": config.adam.learning_rate,
        "mechanism_counts": {"proposals": 0, "retries": 0, "accepted": 0},
        "purpose_spent": {},
    }
    if continuation_state is not None:
        extra.update(
            loss_shift=float(continuation_state["loss_shift"]),
            learning_rate=float(continuation_state["learning_rate"]),
            last_calibration_key=continuation_state.get("last_calibration_key"),
            calibration_events=list(continuation_state.get("calibration_events", [])),
            retry_events=list(continuation_state.get("retry_events", [])),
            role_records=list(continuation_state.get("role_records", [])),
            purpose_spent=dict(continuation_state.get("purpose_spent", {})),
            mechanism_counts=dict(continuation_state.get("mechanism_counts", {})),
        )
    return _ContinuationState(**vars(state), **extra)


def _run_continuation(state: _ContinuationState, evaluator: Any,
                      checkpoint_dir: Optional[Path]) -> ContinuationResult:
    _apply_continuation_context(
        state, evaluator,
        _continuation_context(state, evaluator,
                              _k_for_step(state.config, state.gradient_updates, state.budget)))
    state._propose_evaluator = evaluator  # nonserialized executor, not untrusted resume data
    # Retained scalar observations are restored only through the checkpoint
    # reader, never a pickle or a second θ0 hard score.
    reason, checkpoint_path = _continuation_header(state, evaluator, checkpoint_dir)
    mechanism = _ProductionBacktracking(state, evaluator)
    calibrator = M.SignedFieldCalibrator(
        state.continuation_config.calibration_eta,
        state.continuation_config.calibration_delta_max,
        state.continuation_config.calibration_disagreement_floor,
    )
    try:
        while state.evaluations < state.budget.candidate_evaluations:
            if state.budget.max_generations is not None and state.gradient_updates >= state.budget.max_generations:
                reason = "max_generations"
                break
            iteration = state.gradient_updates
            k = _k_for_step(state.config, iteration, state.budget)
            context = _continuation_context(state, evaluator, k)
            if state.observation is None:
                base = M.OptimizerSnapshot.initial(
                    state.theta, context,
                    {"learning_rate": state.learning_rate, "mode": state.config.mode,
                     "beta1": state.config.beta1, "beta2": state.config.beta2,
                     "epsilon": state.config.epsilon})
            else:
                saved_observation = state.observation
                base = M.OptimizerSnapshot(
                    state.theta, state.moments, state.velocities,
                    {"learning_rate": state.learning_rate, "mode": state.config.mode,
                     "beta1": state.config.beta1, "beta2": state.config.beta2,
                     "epsilon": state.config.epsilon},
                    context, iteration, saved_observation)
            state.observation = base.observation
            try:
                sample = None
                if state.continuation_config.shift == "on" and state.last_calibration_key != _calibration_key(state, context):
                    # A calibration measurement is independent of θ0's one
                    # initial hard score. Its gradient is usable only until a
                    # newly applied shift changes that epoch.
                    mechanism._role = "calibration_measurement"
                    sample = mechanism._observe(state.theta, context, True)
                    field = mechanism.field_pair(state.theta, context)
                    old_shift = context.loss_shift
                    try:
                        if field is None:
                            raise ValueError("raw soft/hard stock fields or identical paths unavailable")
                        update = calibrator.update([field], state.continuation_config.calibration_band,
                                                   old_shift)
                    except (TypeError, ValueError, FloatingPointError) as error:
                        # In particular, an empty physical band is inconclusive,
                        # not a reason to abort an otherwise usable Adam step.
                        update = M.CalibrationResult(old_shift, False, "inconclusive: " + str(error),
                                                     source_count=0)
                    state.calibration_events.append({
                        "iteration": iteration, "before_shift": old_shift,
                        "result": dataclasses.asdict(update),
                        "theta_hash": M.parameter_hash(state.theta),
                    })
                    if update.shift != old_shift:
                        state.loss_shift = float(update.shift)
                        context = dataclasses.replace(context, loss_shift=state.loss_shift)
                        mechanism._role = "fresh_incumbent_after_shift"
                        state.last_calibration_key = _calibration_key(state, context)
                        _apply_continuation_context(state, evaluator, context)
                        sample = mechanism._observe(state.theta, context, True)
                    state.last_calibration_key = _calibration_key(state, context)
                if sample is None:
                    mechanism._role = "incumbent"
                    sample = mechanism._incumbent(base, context)
                base = base.with_observation(sample, context)
                # Pair the cached observation with the actual last hard result
                # only after context/theta/link validation succeeds.
                state.hard_result = mechanism._production_hard
                if base.observation.soft.gradient is None:
                    state.invalid_evaluations += 1
                    reason = "invalid_evaluation"
                    state.observation = base.observation
                    break
            except (M._BudgetExhausted, M._ContextMismatch, _ContinuationRankUnknown) as error:
                reason = str(error) or "continuation_observation_blocked"
                break
            state.observation = base.observation
            if state.continuation_config.backtracking == "off":
                try:
                    proposal = _adam_proposal(state, base, state.learning_rate)
                except _ContinuationRankUnknown:
                    state.invalid_evaluations += 1
                    reason = "invalid_evaluation"
                    break
                state.theta = proposal.theta.copy()
                state.moments = proposal.first_moment.copy()
                state.velocities = proposal.second_moment.copy()
                state.gradient_updates = proposal.optimizer_step
                # This gradient/hard observation belongs to the PRE-proposal
                # theta. Keep the hard-selected incumbent, but never restore
                # that pair as if it described the new current theta.
                state.observation = None
                state.hard_result = None
                state.history.append({
                    "iteration": state.evaluations, "k": context.k,
                    "loss_shift": context.loss_shift,
                    "theta_after_step": state.theta.tolist(),
                    "forward_calls": state.forward_calls, "backward_calls": state.backward_calls,
                    "backtracking": False,
                })
                if (state.next_validation is not None
                        and state.validation_calls < state.budget.max_validation_calls
                        and state.evaluations >= state.next_validation):
                    _validation_and_hard_select(state, evaluator)
                    # Scheduled hard validation shares the same hard-call cap;
                    # it is not a free side channel outside this mechanism.
                    mechanism._spent["hard_validation_calls"] = state.validation_calls
                    state.purpose_spent["scheduled_validation_calls"] = (
                        state.purpose_spent.get("scheduled_validation_calls", 0) + 1)
                    state.next_validation += state.budget.validation_interval
            else:
                mechanism._role = "proposal"
                attempt_before = mechanism._spent["proposals"]
                def propose(snapshot, learning_rate):
                    mechanism._role = "proposal" if learning_rate == state.learning_rate else "retry"
                    return _adam_proposal(state, snapshot, learning_rate)
                outcome = mechanism.step(base, propose, context)
                state.observation = outcome.snapshot.observation
                state.retry_events.extend(outcome.history)
                if outcome.accepted:
                    state.hard_result = mechanism._production_hard
                    state.theta = outcome.snapshot.theta.copy()
                    state.moments = outcome.snapshot.first_moment.copy()
                    state.velocities = outcome.snapshot.second_moment.copy()
                    state.gradient_updates = outcome.snapshot.optimizer_step
                    state.learning_rate = outcome.snapshot.learning_rate
                    state.history.append({
                        "iteration": state.evaluations, "k": context.k,
                        "loss_shift": context.loss_shift, "accepted": True,
                        "learning_rate": state.learning_rate,
                        "theta": base.theta.tolist(), "theta_after_step": state.theta.tolist(),
                        "moments": state.moments.tolist(), "velocities": state.velocities.tolist(),
                        "forward_calls": state.forward_calls, "backward_calls": state.backward_calls,
                    })
                    reason = "budget_exhausted"
                else:
                    # Parent returns a restored snapshot, including any rejected
                    # branch's candidate learning rate, moments and theta.
                    state.theta = outcome.snapshot.theta.copy()
                    state.moments = outcome.snapshot.first_moment.copy()
                    state.velocities = outcome.snapshot.second_moment.copy()
                    state.learning_rate = outcome.snapshot.learning_rate
                    state.history.append({"iteration": state.evaluations, "k": context.k,
                                          "loss_shift": context.loss_shift, "accepted": False,
                                          "theta": outcome.snapshot.theta.tolist(),
                                          "moments": outcome.snapshot.first_moment.tolist(),
                                          "velocities": outcome.snapshot.second_moment.tolist(),
                                          "learning_rate": outcome.snapshot.learning_rate,
                                          "spent": dict(outcome.spent), "reason": outcome.reason})
                    reason = outcome.reason
            state.mechanism_counts = {name: mechanism._spent[name]
                                      for name in ("proposals", "retries", "accepted")}
            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / f"ckpt-{state.evaluations:06d}.json"
                _write_adam_checkpoint(checkpoint_path, state, evaluator)
            if reason not in ("budget_exhausted", None):
                break
        else:
            reason = "budget_exhausted"
    except (M._BudgetExhausted, M._ContextMismatch, _ContinuationRankUnknown) as error:
        reason = str(error) or "continuation_observation_blocked"
    finally:
        state.mechanism_counts = {name: mechanism._spent[name]
                                  for name in ("proposals", "retries", "accepted")}
    return _continuation_final(state, evaluator, reason, checkpoint_dir, checkpoint_path)


def solve_continuation_adam(evaluator: Any, theta0: Any, config: ContinuationConfig, *,
                            budget: B.OptimizationBudget, seed: int = 0,
                            run_id: Optional[str] = None,
                            checkpoint_dir: Optional[str | Path] = None) -> AdamResult:
    """Annealed Adam + signed field EMA + optional shared-rank backtracking.

    ``shift=off, backtracking=off`` directly calls the unchanged solve_adam,
    including exact seed, θ0, moment, schedule and checkpoint behavior.
    See ``resume_continuation_adam`` for trusted checkpoint restoration.
    """
    if not isinstance(config, ContinuationConfig):
        raise TypeError("continuation-adam requires ContinuationConfig")
    _required_budget(budget)
    if budget.candidate_evaluations < 1:
        raise ValueError("continuation-adam requires θ0's hard initialization")
    if config.shift == "off" and config.backtracking == "off":
        return solve_adam(evaluator, budget, config.adam, seed=seed, x0=theta0,
                          run_id=run_id or f"adam-{config.adam.mode}-deterministic",
                          checkpoint_dir=checkpoint_dir)
    clear = getattr(evaluator, "clear_evaluation_loss_shift", None)
    if clear is not None:
        clear()
    run_id = run_id or f"continuation-adam-{config.adam.mode}-deterministic"
    base = _initial_state(evaluator, budget, config.adam, seed, theta0, run_id)
    cfg = dataclasses.replace(config, adam=base.config)
    state = _continuation_from_adam(base, cfg)
    # The resolved schedule (including n_steps) is fixed before the first call.
    return _run_continuation_with_cleanup(
        state, evaluator, Path(checkpoint_dir) if checkpoint_dir is not None else None)


def _run_continuation_with_cleanup(state: _ContinuationState, evaluator: Any,
                                   checkpoint_dir: Optional[Path]) -> ContinuationResult:
    try:
        return _run_continuation(state, evaluator, checkpoint_dir)
    finally:
        if hasattr(evaluator, "clear_evaluation_k"):
            evaluator.clear_evaluation_k()
        elif hasattr(getattr(evaluator, "spec", None), "k"):
            evaluator.sim.k[None] = evaluator.spec.k
        clear = getattr(evaluator, "clear_evaluation_loss_shift", None)
        if clear is not None:
            clear()
        elif hasattr(evaluator, "set_evaluation_loss_shift"):
            evaluator.set_evaluation_loss_shift(0.0)


def _peek_continuation_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B.BlackBoxError(f"cannot read continuation checkpoint: {error}") from error
    if document.get("schema") != "diffcam-adam-checkpoint/2":
        raise B.BlackBoxError("continuation resume requires Adam checkpoint/2")
    payload = dict(document)
    expected = payload.pop("payload_hash", None)
    if expected != C.content_hash(payload):
        raise B.BlackBoxError("continuation checkpoint checksum mismatch")
    return document


def _restore_continuation_config(document: Mapping[str, Any]) -> ContinuationConfig:
    record = document.get("continuation_state")
    if not isinstance(record, Mapping) or record.get("schema") != "diffcam-continuation-state/1":
        raise B.BlackBoxError("unrecognized continuation checkpoint state")
    for name in ("config", "config_sha256", "loss_shift", "learning_rate",
                 "purpose_spent", "mechanism_counts", "observation_context"):
        if name not in record:
            raise B.BlackBoxError(f"incomplete continuation checkpoint: {name}")
    try:
        adam_record = dict(record["config"]["adam"])
        if isinstance(adam_record.get("selection"), Mapping):
            adam_record["selection"] = HardSelectionPolicy(**adam_record["selection"])
        adam = AdamConfig(**adam_record)
        band = M.SurfaceBand(**record["config"]["calibration_band"])
        config = ContinuationConfig(
            **dict(record["config"], adam=adam, calibration_band=band))
    except (KeyError, TypeError, ValueError) as error:
        raise B.BlackBoxError(f"invalid continuation configuration: {error}") from error
    if (config.metadata()["config"] != record["config"]
            or config.config_hash != record["config_sha256"]):
        raise B.BlackBoxError("continuation config hash differs from checkpoint declaration")
    if isinstance(record["learning_rate"], bool) or not math.isfinite(record["learning_rate"]):
        raise B.BlackBoxError("nonfinite continuation retry learning rate")
    rate = float(record["learning_rate"])
    if not config.retry_config.learning_rate_min <= rate <= config.retry_config.learning_rate_max:
        raise B.BlackBoxError("continuation retry learning rate exceeds declared bounds")
    if isinstance(record["loss_shift"], bool) or not math.isfinite(record["loss_shift"]):
        raise B.BlackBoxError("nonfinite continuation loss shift")
    if abs(float(record["loss_shift"])) > 0.5:
        raise B.BlackBoxError("continuation loss shift exceeds the simulator bound")
    for name in ("purpose_spent", "mechanism_counts"):
        for role, count in record[name].items():
            if isinstance(role, str) is False or isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise B.BlackBoxError(f"invalid {name} ledger entry: {role}")
    return config


def _validate_retained_continuation(document: Mapping[str, Any], base: _AdamState,
                                    state: _ContinuationState) -> None:
    record = document["continuation_state"]
    if state.continuation_config.adam != state.config:
        raise B.BlackBoxError("continuation and Adam schedules differ in checkpoint")
    observation = state.observation
    result = state.hard_result
    if (observation is None) != (result is None):
        raise B.BlackBoxError("partial continuation cached observation is not resumable evidence")
    if result is not None:
        if result.theta_hash != observation.hard.theta_hash:
            raise B.BlackBoxError("cached hard-result hash differs from continuation observation")
        if result.objective != observation.hard.raw_phi_hard:
            raise B.BlackBoxError("cached hard objective differs from its paired observation")
        if observation.hard.theta_hash != M.parameter_hash(base.theta):
            raise B.BlackBoxError("retained incumbent does not belong to saved theta")
        if observation.hard.coverage_complete != (result.coverage is C.Coverage.COMPLETE):
            raise B.BlackBoxError("retained hard coverage state differs")
    ledgers = document.get("ledger_snapshot", {})
    for name, actual in (("candidate_evaluations", state.evaluations),
                         ("validation_calls", state.validation_calls),
                         ("forward_calls", state.forward_calls),
                         ("backward_calls", state.backward_calls)):
        if ledgers.get(name) != actual:
            raise B.BlackBoxError(f"continuation ledger/state mismatch: {name}")
    def total(suffix):
        return sum(count for key, count in state.purpose_spent.items() if key.endswith(suffix))
    if total("_candidate_calls") != state.evaluations:
        raise B.BlackBoxError("continuation purpose ledger does not reconcile candidate calls")
    if total("_validation_calls") != state.validation_calls:
        raise B.BlackBoxError("continuation purpose ledger does not reconcile validation calls")


def resume_continuation_adam(path: str | Path, evaluator: Any, *,
                             budget: Optional[B.OptimizationBudget] = None,
                             run_id: Optional[str] = None,
                             checkpoint_dir: Optional[str | Path] = None) -> AdamResult:
    """Restore trusted Adam/2 state without re-running prior θ0 or moments."""
    source = Path(path)
    document = _peek_continuation_checkpoint(source)
    if "continuation_state" not in document:
        # (off, off) takes exactly the unchanged Adam checkpoint/resume path.
        return resume_adam(source, evaluator, budget=budget, run_id=run_id,
                           checkpoint_dir=checkpoint_dir)
    config = _restore_continuation_config(document)
    setter = getattr(evaluator, "set_evaluation_loss_shift", None)
    if setter is None:
        raise AttributeError("continuation resume requires explicit loss-shift support")
    setter(float(document["continuation_state"]["loss_shift"]))
    document = _read_adam_checkpoint(source, evaluator)
    try:
        base = _restore_state(document, evaluator, budget=budget, run_id=run_id)
        state = _continuation_from_adam(base, config, document["continuation_state"])
        state.observation = _unpack_continuation_observation(document, evaluator.task.grid)
        packed_hard = document["continuation_state"].get("hard_result")
        state.hard_result = None if packed_hard is None else C.CandidateResult(**packed_hard)
        _validate_retained_continuation(document, base, state)
        # Only validated retained evidence is eligible to affect the new ledger.
        B._restore_ledger_payload(document, evaluator)
    except BaseException:
        if hasattr(evaluator, "clear_evaluation_k"):
            evaluator.clear_evaluation_k()
        elif hasattr(getattr(evaluator, "spec", None), "k"):
            evaluator.sim.k[None] = evaluator.spec.k
        clear = getattr(evaluator, "clear_evaluation_loss_shift", None)
        if clear is not None:
            clear()
        elif hasattr(evaluator, "set_evaluation_loss_shift"):
            evaluator.set_evaluation_loss_shift(0.0)
        raise
    return _run_continuation_with_cleanup(
        state, evaluator, Path(checkpoint_dir) if checkpoint_dir is not None else None)


def continuation_result_metadata(result: AdamResult) -> dict[str, Any]:
    """JSON-ready runner metadata; no evaluator state or mock is serialized."""
    if not isinstance(result, AdamResult):
        raise TypeError("continuation metadata requires an AdamResult")
    if isinstance(result, ContinuationResult):
        extra = {
            "continuation_configuration": result.continuation_config,
            "continuation_configuration_sha256": result.continuation_config_hash,
            "loss_shift": result.loss_shift, "calibration_events": result.calibration_events,
            "retry_events": result.retry_events, "retry_state": result.retry_state,
            "purpose_spent": result.purpose_spent,
        }
    else:
        extra = {"continuation_configuration": None, "continuation_configuration_sha256": None,
                 "loss_shift": 0.0, "calibration_events": [], "retry_events": [],
                 "retry_state": {}, "purpose_spent": {}, "baseline_delegate": "solve_adam"}
    return {
        "condition": "continuation-adam",
        "optimizer_configuration": dataclasses.asdict(result.config),
        "budget_spent": {
            "candidate_evaluations": result.evaluations,
            "validation_calls": result.validation_calls,
            "forward_calls": result.forward_calls, "backward_calls": result.backward_calls,
            "gradient_updates": result.gradient_updates,
        },
        "initialization": B._initial_candidate_dict(
            result.initial_candidate, result.initial_theta, result.initial_candidate_id),
        "selection_thresholds": result.selection_thresholds,
        **extra,
    }


if __name__ == '__main__':
    main()
