"""Minimal Adam optimization of delta actions; hard-carve checkpoint selection.

Run with ``uv run python -m algorithms.optimize --output runs/core-smoke``.
Input trajectories and output positions use normalized stock coordinates.
"""
import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import taichi as ti

from benchmarks import contracts as C
from . import blackbox as B
from algorithms import run_artifacts as R

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
        path = np.load(args.initial).astype(np.float32)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 3 or not np.isfinite(path).all():
            raise ValueError("initial must contain finite (N >= 3, 3) positions")
        start, deltas = path[0], np.diff(path, axis=0)
    else:
        # Small random moves are a wiring demo, not a CAM initialization.
        start = np.array([0.15, 0.15, 0.85], dtype=np.float32)
        deltas = rng.normal(0, 0.025, (args.steps, 3)).astype(np.float32)
    ti.init(arch=ti.cpu if args.cpu else ti.gpu, offline_cache=False)
    sim = CSGSimulatorDelta(resolution=args.resolution, max_steps=len(deltas),
                            stock_size_in=(1, 1, 1), target_shape=args.target,
                            tool_start=tuple(start), dt=args.dt, init_taichi=False)
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
        sim.tool_delta.from_numpy(deltas)
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
                  target_radius_mm=9.0, target_height_mm=18.0, target_half_size_mm=9.0)
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


@dataclasses.dataclass(frozen=True)
class HardSelectionPolicy:
    max_gouge_fraction: float = 0.0
    max_holder_overlap_fraction: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.max_gouge_fraction, self.max_holder_overlap_fraction):
            if not math.isfinite(value) or value < 0:
                raise ValueError("hard selection thresholds must be finite and non-negative")


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
    best_rank: Any
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
        steps = budget.max_generations if budget.max_generations is not None else budget.candidate_evaluations
        if steps < 1:
            raise ValueError("annealed mode needs a positive declared n_steps/budget")
        # Resolve once at initialization, freeze into config and checkpoint hash.
        config = dataclasses.replace(config, n_steps=steps)
    theta, _, _ = B._options(evaluator, seed, 0.1, x0, None)
    theta = np.asarray(theta, dtype=np.float64)
    return _AdamState(
        config, budget, seed, run_id, _config_hash(config), theta, theta.copy(),
        np.zeros(theta.size), np.zeros(theta.size), None, None, None, 0, 0, 0, 0, 0,
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


def _hard_rank(result: C.CandidateResult, policy: HardSelectionPolicy) -> Optional[tuple]:
    if not result.valid or result.objective is None:
        return None
    required = {name: result.components.get(name)
                for name in ("gouge_fraction", "residual_fraction", "holder_overlap_fraction")}
    if any(value is None for value in required.values()):
        return None
    # Do not convert unavailable or invalid modeled safety markers into a best
    # candidate. Callers must compare finite ranks only, never serialize inf.
    gouge = max(0.0, required["gouge_fraction"] - policy.max_gouge_fraction)
    holder = max(0.0, required["holder_overlap_fraction"] - policy.max_holder_overlap_fraction)
    violations = max(gouge, holder)
    return (int(violations > 0), violations, required["residual_fraction"], result.objective)


def _write_adam_checkpoint(path: Path, state: _AdamState, evaluator: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "diffcam-adam-checkpoint/1", "contract_schema": C.SCHEMA_VERSION,
        "run_id": state.run_id, "seed": state.seed, "optimizer_config": dataclasses.asdict(state.config),
        "optimizer_config_hash": state.config_hash, "task_setup_hash": B._setup_signature(evaluator, "smooth"),
        "budget": dataclasses.asdict(state.budget),
        "theta": state.theta.tolist(), "moments": state.moments.tolist(), "velocities": state.velocities.tolist(),
        "initial_theta": state.initial_theta.tolist(),
        "best": B._best_candidate_dict(state.best, state.best_theta),
        "best_rank": state.best_rank,
        "evaluations": state.evaluations, "gradient_updates": state.gradient_updates,
        "forward_calls": state.forward_calls, "backward_calls": state.backward_calls,
        "validation_calls": state.validation_calls, "next_validation": state.next_validation,
        "invalid_evaluations": state.invalid_evaluations,
        "budget_at_checkpoint": _k_for_step(state.config, max(0, state.evaluations - 1), state.budget),
        "next_k": _k_for_step(state.config, state.evaluations, state.budget),
        "history": state.history, "previous_evaluations": state.previous_evaluations,
        "ledger_snapshot": evaluator.ledger.snapshot(),
    }
    digest = C.content_hash(payload)
    R.ArtifactWriter._replace_bytes(path, R.json_bytes(dict(payload, payload_hash=digest)))
    return digest


def _read_adam_checkpoint(path: Path, evaluator: Any) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B.BlackBoxError(f"cannot read Adam checkpoint: {error}") from error
    if document.get("schema") != "diffcam-adam-checkpoint/1":
        raise B.BlackBoxError("unrecognized Adam checkpoint schema")
    payload = dict(document)
    expected_hash = payload.pop("payload_hash")
    if expected_hash != C.content_hash(payload):
        raise B.BlackBoxError("Adam checkpoint checksum mismatch")
    if document["task_setup_hash"] != B._setup_signature(evaluator, "smooth"):
        raise B.BlackBoxError("Adam checkpoint task/setup identity differs")
    if document.get("seed") is None or not isinstance(document["seed"], int):
        raise B.BlackBoxError("missing/invalid Adam seed record")
    return document


def _restore_state(document: Mapping[str, Any], evaluator: Any, *, budget=None,
                   run_id=None) -> _AdamState:
    old_budget = B.OptimizationBudget(**document["budget"])
    budget = budget or old_budget
    origin = int(document["evaluations"])
    if budget.candidate_evaluations < origin:
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
    state = _AdamState(config, budget, int(document["seed"]), run_id or document["run_id"],
                       _config_hash(config), np.asarray(document["theta"], dtype=float).copy(),
                       np.asarray(document["initial_theta"], dtype=float),
                       np.asarray(document["moments"], dtype=float).copy(),
                       np.asarray(document["velocities"], dtype=float).copy(), None, None,
                       tuple(document["best_rank"]) if document["best_rank"] is not None else None,
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
    return state


def _validation_and_hard_select(state: _AdamState, evaluator: Any) -> None:
    state.validation_calls += 1
    evaluator.ledger.start_validation()
    before = evaluator.ledger.snapshot()
    result = evaluator.evaluate_candidate(state.theta, f"{state.run_id}:validate:{state.validation_calls}", "hard")
    after = evaluator.ledger.snapshot()
    state.forward_calls += after["forward_calls"] - before["forward_calls"]
    state.backward_calls += after["backward_calls"] - before["backward_calls"]
    rank = _hard_rank(result, state.config.selection)
    if rank is not None and (state.best_rank is None or rank < state.best_rank):
        state.best, state.best_theta, state.best_rank = result, state.theta.copy(), tuple(rank)
    return result


def _run_adam_body(state: _AdamState, evaluator: Any, checkpoint_dir: Optional[Path]) -> AdamResult:
    reason = "budget_exhausted" if state.evaluations >= state.budget.candidate_evaluations else None
    checkpoint_path = None
    while state.evaluations < state.budget.candidate_evaluations:
        if state.budget.max_generations is not None and state.evaluations >= state.budget.max_generations:
            reason = "max_generations"
            break
        k = _k_for_step(state.config, state.evaluations, state.budget)
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
    return AdamResult(state.run_id, state.seed, state.budget, state.config,
                      state.best, state.best_theta, state.theta.copy(),
                      state.initial_theta.copy(), state.moments.copy(), state.velocities.copy(),
                      state.evaluations, state.forward_calls, state.backward_calls,
                      state.validation_calls, state.gradient_updates, state.invalid_evaluations,
                      reason, state.history, str(checkpoint_path) if checkpoint_path else None,
                      state.previous_evaluations)


def _run_adam(state: _AdamState, evaluator: Any, checkpoint_dir: Optional[Path]) -> AdamResult:
    try:
        return _run_adam_body(state, evaluator, checkpoint_dir)
    finally:
        if hasattr(evaluator, "clear_evaluation_k"):
            evaluator.clear_evaluation_k()
        else:
            # Keep the shared evaluator's specification for the next method.
            evaluator.sim.k[None] = evaluator.spec.k if hasattr(evaluator.spec, "k") else evaluator.sim.k[None]


def solve_adam(evaluator: Any, budget: B.OptimizationBudget, config: AdamConfig, *, seed: int = 0,
               x0: Any = None, run_id: Optional[str] = None, checkpoint_dir: Optional[str | Path] = None) -> AdamResult:
    _required_budget(budget)
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
    evaluator.ledger.count_validation(1)
    replayed = evaluator.evaluate_candidate(theta, f"{document['run_id']}:replay", "hard")
    expected = document["best"]["evaluation"]["objective"]
    matches = bool(replayed.valid and replayed.objective is not None
                   and replayed.objective == expected.__float__())
    return {"replayed": replayed, "expected_objective": expected,
            "objective_matches_backend_equality": matches,
            "checkpoint_theta": theta, "checkpoint_hash": document["payload_hash"]}


def emit_adam_artifacts(result: AdamResult, evaluator: Any, run_dir: str | Path,
                        repository: str | Path) -> C.RunRecord:
    if result.best_theta is None or result.best is None or not result.best.valid:
        raise B.BlackBoxError("no hard-selected feasible candidate; not emitting a fake completed run")
    task, motion = evaluator.task, evaluator.candidate_motion(result.best_theta)
    posting = R.post_motion(task, motion)
    config = R.build_run_config(result.run_id, task, motion, seed=result.seed,
                                parameterization=evaluator.parameterization, objective=evaluator.spec,
                                budgets={
                                    "limits": dataclasses.asdict(result.budget),
                                    "spent": {"candidate_evaluations": result.evaluations,
                                              "forward_calls": result.forward_calls,
                                              "backward_calls": result.backward_calls,
                                              "validation_calls": result.validation_calls,
                                              "gradient_updates": result.gradient_updates},
                                }, repository=repository)
    initial = evaluator.parameterization.positions(result.initial_theta)
    best_positions = evaluator.parameterization.positions(result.best_theta)
    files = dict(posting.files)
    files.update({
        "config.json": R.json_bytes(config),
        "history.json": R.json_bytes({"history": result.history}),
        "metrics.json": R.json_bytes({"best_candidate": B._candidate_dict(result.best),
                                      "best_rank": None if result.best is None else _hard_rank(result.best, result.config.selection),
                                      "termination_reason": result.termination_reason,
                                      "invalid_evaluations": result.invalid_evaluations}),
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


if __name__ == '__main__':
    main()
