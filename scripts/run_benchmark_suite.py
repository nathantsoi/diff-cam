"""T4.1a auditable benchmark runner, not a claim of a completed real pilot.

Only ``pilot`` / ``dev`` is implemented in this card. Other enum values fail clearly.
One child process per condition/seed owns at most one simulator. Jobs launch serially
(also for GPU) to satisfy runtime/device limits. A later T2.2 task and T1.3 posted
Evaluator plug into the same frozen-run protocol.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from algorithms import blackbox as B
from algorithms import objectives as O
from algorithms import optimize as adam
from algorithms import parameterization as P
from algorithms import run_artifacts as R
from benchmarks import audit as audit
from benchmarks import contracts as C
from simulator.csg_simulator import CSGSimulatorDelta

EXPERIMENTS = ("pilot", "initialization", "solvers", "mechanics", "all")
SPLITS = ("dev", "test", "external")
CONDITIONS = ("nominal-cam", "fixed-adam", "annealed-adam", "hard-cmaes", "smooth-cmaes")
RUNNER_SCHEMA = "diffcam-benchmark-runner/1"
PENDING_REASON = "Posted evaluation is unavailable until T1.3"
REPOSITORY = Path(__file__).resolve().parent.parent


class RunnerError(ValueError):
    pass


@dataclass
class Outcome:
    seed: int
    best: C.CandidateResult
    best_theta: np.ndarray
    initial_theta: np.ndarray
    budget: B.OptimizationBudget
    evaluations: int
    validation_calls: int
    generations: int
    forward_calls: int
    backward_calls: int
    invalid_candidates: int
    termination_reason: str
    history: list[dict[str, Any]]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"cannot read {path}: {error}") from error


def _atomic(path: Path, payload: Any) -> None:
    R.ArtifactWriter._replace_bytes(path, R.json_bytes(payload))


def _positive_int(value: Any, name: str, *, zero=False) -> int:
    minimum = 0 if zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RunnerError(f"{name} must be an integer >= {minimum}")
    return value


def _normalized_config(path: str | Path) -> dict[str, Any]:
    raw = _json(Path(path))
    if not isinstance(raw, dict) or raw.get("schema") != RUNNER_SCHEMA:
        raise RunnerError(f"config schema must be {RUNNER_SCHEMA}")
    for name in ("candidate_evaluations", "population_size", "max_validation_calls"):
        _positive_int(raw.get(name), name, zero=name == "max_validation_calls")
    interval = _positive_int(raw.get("validation_interval"), "validation_interval")
    for name in ("k_fixed", "k_smooth", "k_anneal_start", "k_anneal_end", "learning_rate"):
        if name not in raw:
            raise RunnerError(f"config must declare {name} explicitly")
        if not isinstance(raw[name], (int, float)) or not math.isfinite(raw[name]) or raw[name] <= 0:
            raise RunnerError(f"{name} must be finite and positive")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise RunnerError("seeds must be a nonempty, unique list")
    for seed in seeds:
        _positive_int(seed, "seed", zero=True)
    return raw


def _prototype() -> tuple[C.TaskSpec, C.MotionPlan, P.DeltaResidual]:
    """The pure task description of the accepted T3.0 analytic two-segment fixture."""
    initial = np.array([-4.0, -4.0, 2.0])
    task = C.example_pocket_task()
    part = C.GeometrySource(kind="analytic", params={"primitive": "sphere", "center": [0.5, 0.5, 0.5],
                                                   "radius_mm": 6.0})
    state = C.MachineState(position_mm=task.work_to_machine.apply(initial), frame=C.Frame.MACHINE)
    p1 = task.work_to_machine.apply(initial + [0, 0, -6])
    p2 = task.work_to_machine.apply(initial + [3, 3, -12])
    segments = (
        C.Segment(id="plunge", kind=C.SegmentKind.FEED, start_mm=state.position_mm,
                  end_mm=p1, duration_s=0.6, feed_mm_s=10.0, tag="plunge"),
        C.Segment(id="slot", kind=C.SegmentKind.FEED, start_mm=p1, end_mm=p2,
                  duration_s=np.linalg.norm(np.asarray(p1) - p2) / 10.0, feed_mm_s=10.0),
    )
    task = dataclasses.replace(task, original_part=part,
                               operation_target=C.OperationTarget(geometry=part),
                               initial_state=state,
                               tool=dataclasses.replace(task.tool, holder_bottom_mm=task.tool.flute_length_mm))
    plan = C.MotionPlan(frame=C.Frame.MACHINE, initial_state=state, segments=segments,
                        timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
                        provenance="analytic prototype task until T2.2")
    parameterization = P.DeltaResidual([[0, 0, -6], [3, 3, -6]], scale_mm=1,
                                         bounds=(-0.05, 0.05), initial_position_mm=initial)
    return task, plan, parameterization


def _context(config: Mapping[str, Any], repository: Path):
    kind = config.get("task_source", "prototype_analytic")
    if kind == "prototype_analytic":
        return (*_prototype(), _prototype_evaluator, None)
    if not config.get("allow_python_plugins", False):
        raise RunnerError("unregistered task_source and no allow_python_plugins=true opt-in")
    if not isinstance(kind, str) or ":" not in kind:
        raise RunnerError("plugin must be in the form trusted.module:factory")
    module_name, factory_name = kind.rsplit(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise RunnerError(f"task_factory {kind} is not callable")
    values = factory(config, repository)
    if len(values) not in (4, 5):
        raise RunnerError("task plugin must return (task, motion, parameterization, evaluator_factory[, posted_factory])")
    return values


def _prototype_evaluator(task, motion, parameterization, config, seed):
    """One runtime and exactly one simulator, created inside one child process."""
    guard = os.environ.get("DIFFCAM_RUNNER_SIMULATOR", "0")
    if guard not in ("0", "1"):
        raise RunnerError("child already owns a simulator; strict one-per-process rule")
    os.environ["DIFFCAM_RUNNER_SIMULATOR"] = "1"
    arch = config.get("arch", "cpu")
    if arch not in ("cpu", "gpu"):
        raise RunnerError("arch must be cpu or gpu")
    import taichi as ti
    ti.init(arch=ti.cpu if arch == "cpu" else ti.gpu, offline_cache=False, default_fp=ti.f32)
    initial = np.asarray(parameterization.initial_position_mm)
    normalized = (initial - task.grid.origin_mm) / task.grid.extent_mm
    sim = CSGSimulatorDelta(resolution=max(task.grid.shape), max_steps=parameterization.num_steps,
                            stock_size_mm=task.grid.extent_mm, voxel_size_mm=task.grid.voxel_mm,
                            target_shape="sphere", tool_start=tuple(normalized),
                            enforce_speed_limits=False, init_taichi=False, k_init=config["k_fixed"])
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.tool_cut_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    sim.set_target_params(radius_mm=6.0)
    sim.bake_target_grid()
    return O.CandidateEvaluator(task, motion, parameterization, sim,
                                spec=O.ObjectiveSpec(k=float(config["k_smooth"])), seed=seed)


def _budget(config: Mapping[str, Any], method: str) -> B.OptimizationBudget:
    return B.OptimizationBudget(
        candidate_evaluations=config["candidate_evaluations"],
        population_size=config["population_size"] if method.endswith("cmaes") else None,
        validation_interval=config["validation_interval"],
        max_validation_calls=config["max_validation_calls"],
        max_generations=config.get("max_generations"),
    )


def _identity(task: C.TaskSpec, config: Mapping[str, Any], split: str) -> str:
    payload = {
        "task_hash": task.task_hash, "task_split": task.split.value, "split": split,
        "experiment": config["experiment"], "task_source": config.get("task_source", "prototype_analytic"),
        "arch": config.get("arch", "cpu"),
        "seeds": config["seeds"], "candidate_evaluations": config["candidate_evaluations"],
        "population_size": config["population_size"], "validation_interval": config["validation_interval"],
        "max_validation_calls": config["max_validation_calls"],
        "learning_rate": config["learning_rate"], "k_fixed": config["k_fixed"], "k_smooth": config["k_smooth"],
        "k_anneal_start": config["k_anneal_start"], "k_anneal_end": config["k_anneal_end"],
        "max_generations": config.get("max_generations"),
    }
    return C.content_hash(payload)


def _plan_entries(config: Mapping[str, Any], task: C.TaskSpec, split: str, only: Optional[str]) -> list[dict[str, Any]]:
    identity = _identity(task, config, split)
    entries = []
    for condition in CONDITIONS:
        seeds = (0,) if condition in ("nominal-cam", "fixed-adam", "annealed-adam") else config["seeds"]
        for seed in seeds:
            run_id = f"{condition}-s{seed}-{task.task_hash[7:19]}-{identity[7:19]}"
            if only is None or fnmatch.fnmatchcase(run_id, only):
                entries.append({"run_id": run_id, "method": condition, "seed": seed,
                                "task_id": task.task_id, "identity": identity})
    if not entries:
        raise RunnerError(f"--only {only!r} selects none of the requested runs")
    if not all(character.isalnum() or character in "-_."
               for entry in entries for character in entry["run_id"]):
        raise RunnerError("counterexample run identifier")
    return entries


def _pecent(config: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predicted = []
    supplied = config.get("estimated_seconds_per_candidate", {})
    for entry in entries:
        optimizer_budget = 1 if entry["method"] == "nominal-cam" else config["candidate_evaluations"]
        interval = config["validation_interval"]
        if interval == 0 and config["max_validation_calls"] > 0:
            raise RunnerError("validation_interval must be positive when validation is enabled")
        validations = min(config["max_validation_calls"], optimizer_budget // interval if interval else 0)
        raw = supplied.get(entry["method"])
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
            raw = None
        predicted.append({"run_id": entry["run_id"], "candidate_evaluations": optimizer_budget,
                          "validation_calls": validations,
                          "estimated_run_wall_time_s": None if raw is None else raw * (optimizer_budget + validations),
                          "jit_warmup": "unknown without a real child measurement"})
    return predicted


def _manifest(config: Mapping[str, Any], task: C.TaskSpec, entries: Sequence[Mapping[str, Any]],
              root: Path, *, planned: bool) -> dict[str, Any]:
    return {"schema_version": 1,
            "tasks": [{"task_id": task.task_id, "family": task.family, "task_hash": task.task_hash,
                       "task_split": task.split.value}],
            "expected_runs": [{"run_id": entry["run_id"], "task_id": entry["task_id"],
                               "method": entry["method"], "seed": entry["seed"]} for entry in entries],
            "comparisons": [],
            "runner": {"schema": RUNNER_SCHEMA, "planned_not_executed": planned,
                       "experiment": config["experiment"], "split": "dev",
                       "output_dir": str(root),
                       "prototype_analytic_task": True,
                       "adam_conditions": "one_deterministic_replicate_each",
                       "cma_conditions": "one_run_per_declared_seed",
                       "nominal_cam": "one_deterministic_run"}}


def _evaluate_outcome(condition: str, evaluator: Any, config: Mapping[str, Any], seed: int) -> Outcome:
    dims = evaluator.parameterization.num_params
    zeros = np.zeros(dims)
    if condition == "nominal-cam":
        best = evaluator.evaluate_candidate(zeros, "nominal-once", "hard")
        return Outcome(0, best, zeros.copy(), zeros.copy(), B.OptimizationBudget(1, population_size=None), 1,
                       0, 0, 0, 0, 0, "deterministic_single_evaluation_of_nominal_cam", [])
    if condition.endswith("cmaes"):
        kind = "hard" if condition == "hard-cmaes" else "smooth"
        result = B.solve_cma_es(evaluator, _budget(config, condition), seed=seed,
                                objective_kind=kind, x0=zeros, sigma0=config.get("sigma0", 0.01),
                                run_id=condition)
        if result.best is None:
            raise RunnerError(f"{condition}: no valid candidate from whole population")
        return Outcome(seed, result.best, result.best_theta, result.initial_theta,
                       result.budget, result.evaluations, result.validation_calls, result.generations,
                       getattr(result, "forward_calls", 0), getattr(result, "backward_calls", 0),
                       result.invalid_candidates, result.termination_reason, result.history)
    if condition == "fixed-adam":
        mode, k_start, k_end = "fixed", config["k_fixed"], config["k_fixed"]
    else:
        mode, k_start, k_end = "annealed", config["k_anneal_start"], config["k_anneal_end"]
    cfg = adam.AdamConfig(mode, learning_rate=config["learning_rate"],
                          max_gradient_norm=config.get("max_gradient_norm", 0.5),
                          k_start=k_start, k_end=k_end)
    result = adam.solve_adam(evaluator, _budget(config, condition), cfg, seed=seed,
                             x0=zeros, run_id=condition)
    if result.best is None:
        raise RunnerError(f"{condition}: no valid hard-selected candidate "
                          f"(termination_reason={result.termination_reason})")
    return Outcome(seed, result.best, result.best_theta, result.initial_theta, result.budget,
                   result.evaluations, result.validation_calls, result.gradient_updates,
                   result.forward_calls, result.backward_calls, result.invalid_evaluations,
                   result.termination_reason, result.history)


def _write_failure(entry: Mapping[str, Any], task: C.TaskSpec, motion: C.MotionPlan,
                   config: Mapping[str, Any], status: C.RunStatus, reason: str,
                   root: Path) -> C.RunRecord:
    """Keep a typed failed/interrupted artifact even if no valid candidate exists."""
    run_dir = root / entry["run_id"]
    if run_dir.exists():
        raise RunnerError(f"cannot commit failed state over existing partial data: {run_dir}")
    posting = R.PostingResult(files={}, posted=tuple(
        C.PostedProgram(dialect, C.PostStatus.UNSUPPORTED, reason=f"{status.value}: {reason}")
        for dialect in C.Dialect))
    files = {
        "config.json": R.json_bytes({"task_hash": task.task_hash, "identity": entry["identity"],
                                     "condition": entry["method"], "seed": entry["seed"],
                                     "run_status": status.value, "message": reason}),
        "history.json": R.json_bytes({"entries": []}),
        "metrics.json": R.json_bytes({"status": status.value, "reason": reason}),
        "motion_plan.json": R.record_bytes(motion),
        "initial_trajectory.npy": R.numpy_bytes(np.zeros((1, 3))),
        "trajectory.npy": R.numpy_bytes(np.zeros((1, 3))),
        "trajectory_deltas.npy": R.numpy_bytes(np.zeros((0, 3))),
        "eval_posted.json": R.json_bytes({"schema_version": 1, "programs": {
            dialect.value: {"dialect": dialect.value, "status": "unsupported", "reason": reason}
            for dialect in C.Dialect}}),
    }
    record = R.assemble_record(entry["run_id"], task, status, posting=posting,
                               artifact_names=tuple(sorted(files)), failure_reason=reason)
    R.ArtifactWriter().write(str(run_dir), record, files)
    return record


def _write_outcome(entry: Mapping[str, Any], evaluator: Any, outcome: Outcome, posted_factory: Any,
                   config: Mapping[str, Any], root: Path, repository: Path) -> C.RunRecord:
    task, motion = evaluator.task, evaluator.candidate_motion(outcome.best_theta)
    posting = R.post_motion(task, motion)
    config_out = R.build_run_config(entry["run_id"], task, motion, seed=outcome.seed,
                                    parameterization=evaluator.parameterization, objective=evaluator.spec,
                                    budgets={"limits": dataclasses.asdict(outcome.budget),
                                             "spent": {"candidate_evaluations": outcome.evaluations,
                                                       "validation_calls": outcome.validation_calls,
                                                       "forward_calls": outcome.forward_calls,
                                                       "backward_calls": outcome.backward_calls,
                                                       "generations": outcome.generations}},
                                    repository=repository)
    initial = evaluator.parameterization.positions(outcome.initial_theta)
    best = evaluator.parameterization.positions(outcome.best_theta)
    files = dict(posting.files)
    files.update({
        "config.json": R.json_bytes(config_out),
        "history.json": R.json_bytes({"history": outcome.history, "ledger": evaluator.ledger.snapshot()}),
        "metrics.json": R.json_bytes({"best_candidate": B._candidate_dict(outcome.best),
                                      "termination_reason": outcome.termination_reason,
                                      "invalid_candidates": outcome.invalid_candidates,
                                      "execution": {"subprocess": os.getpid(), "simulators_in_worker": 1}}),
        "motion_plan.json": R.record_bytes(motion),
        "initial_trajectory.npy": R.numpy_bytes(initial),
        "trajectory.npy": R.numpy_bytes(best),
        "trajectory_deltas.npy": R.numpy_bytes(np.diff(best, axis=0)),
    })
    missing_posted = posted_factory is None
    if posted_factory is not None:
        evaluator_for_posts = posted_factory(task, config)
        for post in posting.posted:
            if post.status != C.PostStatus.SUPPORTED:
                continue
            nc_name, eval_name = C.DIALECT_ARTIFACTS[post.dialect]
            actual = evaluator_for_posts.evaluate_posted_gcode(task, files[nc_name], post.dialect, {})
            if actual.dialect != post.dialect or actual.nc_hash != post.nc_hash or actual.task_hash != task.task_hash:
                raise RunnerError("posted evaluator returned a mismatched dialect/task/NC hash")
            files[eval_name] = R.record_bytes(actual)
    status = C.RunStatus.COMPLETE if not missing_posted else C.RunStatus.INCOMPLETE
    reason = None if not missing_posted else PENDING_REASON
    record = R.assemble_record(entry["run_id"], task, status, posting=posting,
                               artifact_names=tuple(sorted(files)), source_commit=config_out["source_commit"],
                               dirty_patch_hash=config_out["dirty_patch_hash"],
                               failure_reason=reason)
    R.ArtifactWriter().write(str(root / entry["run_id"]), record, files)
    return record



def _child(config: Mapping[str, Any], entry: Mapping[str, Any], root: Path) -> None:
    task, motion, params, evaluator_factory, posted_factory = _context(config, root)
    if entry["identity"] != _identity(task, config, "dev"):
        raise RunnerError("child task identity differs from frozen parent manifest")
    if config.get("__test_interrupt__") == entry["run_id"]:
        raise KeyboardInterrupt("injected child interruption; parent must retain the status")
    evaluator = evaluator_factory(task, motion, params, config, entry["seed"])
    try:
        outcome = _evaluate_outcome(entry["method"], evaluator, config, entry["seed"])
        _write_outcome(entry, evaluator, outcome, posted_factory, config, root, REPOSITORY)
    except KeyboardInterrupt:
        raise
    except ImportError as error:
        _write_failure(entry, task, motion, config, C.RunStatus.BLOCKED, str(error), root)
        raise
    except Exception as error:
        _write_failure(entry, task, motion, config, C.RunStatus.FAILED, str(error), root)
        raise


def _resume_existing(root: Path, entry: Mapping[str, Any], task: C.TaskSpec) -> bool:
    """Skip a matching completed run, otherwise preserve its partial artifacts."""
    directory = root / entry["run_id"]
    if not directory.exists():
        return False
    record_path = directory / "run.json"
    if not record_path.is_file():
        raise RunnerError(f"partial run without run.json needs an identity review: {directory}")
    stored = C.loads(record_path.read_text(encoding="utf-8"))
    if stored.run_id != entry["run_id"] or stored.task_hash != task.task_hash:
        raise RunnerError("existing run ID/task identity differs; refusing to resume it")
    audited = audit._audit_run(record_path)
    if audited.complete_and_audited and not any(
            issue for item in (audited.dialects or {}).values() for issue in item.issues):
        return True
    backups = root / "resume-backups"
    backups.mkdir(exist_ok=True)
    destination = backups / f"{entry['run_id']}-{time.time_ns()}"
    directory.rename(destination)  # preserve, never delete or overwrite an incomplete record.
    return False


def _child_record_status(directory: Path) -> Optional[str]:
    record_path = directory / "run.json"
    if not record_path.is_file():
        return None
    return C.loads(record_path.read_text(encoding="utf-8")).status.value


def _record_interrupted(entry: Mapping[str, Any], task: C.TaskSpec, motion: C.MotionPlan,
                        config: Mapping[str, Any], root: Path, reason: str) -> None:
    if (root / entry["run_id"] / "run.json").is_file():
        return  # a completed child artifact remains what it is; do not relabel it.
    if (root / entry["run_id"]).exists():
        _atomic(root / "interruption-sidecars" / f"{entry['run_id']}.json",
                {"run_id": entry["run_id"], "reason": reason,
                 "note": "partial child artifacts have no final run.json"})
    else:
        _write_failure(entry, task, motion, config, C.RunStatus.INTERRUPTED, reason, root)


def _run_parent(config: Mapping[str, Any], args: argparse.Namespace, root: Path) -> int:
    task, motion, parameterization, _, _ = _context(config, root)
    if task.split.value != args.split:
        raise RunnerError(f"prototype source is dev-only; task split is {task.split.value}")
    if args.split != "dev":
        raise RunnerError("held-out test/external sources are not shipped in this card")
    if config["experiment"] != "pilot":
        raise RunnerError(f"experiment {config['experiment']!r} is not implemented in T4.1a")
    entries = _plan_entries(config, task, args.split, args.only)
    frozen_path = root / "frozen_config.json"
    if args.resume:
        if not frozen_path.is_file():
            raise RunnerError("resume requires the prior frozen_config.json")
        old = _json(frozen_path)
        if old.get("raw_config") != dict(config) or old.get("identity") != entries[0]["identity"]:
            raise RunnerError("resume task/config identity differs from frozen plan")
        manifest_existing = _json(root / "manifest.json")
        allowed = {run["run_id"] for run in manifest_existing["expected_runs"]}
        if any(entry["run_id"] not in allowed for entry in entries):
            raise RunnerError("--only would introduce a different run identity")
    else:
        root.mkdir(parents=True)
        _atomic(frozen_path, {"raw_config": dict(config), "identity": entries[0]["identity"]})
    _atomic(root / "manifest.json", _manifest(config, task, entries, root, planned=args.dry_run))
    if args.dry_run:
        cost = {"planned_run_count": len(entries), "costs": _pecent(config, entries),
                "serialization": {"subprocess_per_run": True, "max_simulators_per_process": 1,
                                  "global_simulator_cap": 12, "gpu_jobs": "serial"}}
        _atomic(root / "dry_run.json", cost)
        print(json.dumps(cost, indent=2, sort_keys=True))
        return 0
    records = []
    for entry in entries:
        already = root / entry["run_id"]
        if already.exists() and not args.resume:
            raise RunnerError(f"refusing to overwrite existing run {already}")
        if args.resume and already.exists() and _resume_existing(already.parent, entry, task):
            records.append({"run_id": entry["run_id"], "reported_status": "skipped_complete",
                            "returncode": 0})
            continue
        before = time.perf_counter()
        env = dict(os.environ)
        env["DIFFCAM_RUNNER_SIMULATOR"] = "0"
        command = [sys.executable, str(Path(__file__).resolve()), "--config", args.config,
                   "--output-dir", str(root), "--split", args.split, "--experiment", args.experiment,
                   "--only", entry["run_id"], "--worker-run"]
        (root / "logs").mkdir(exist_ok=True)
        with (root / "logs" / f"{entry['run_id']}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                code = process.wait(timeout=config.get("timeout_s"))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()  # exact child PID started by this runner, never a pattern kill.
                    code = process.wait()
                _record_interrupted(entry, task, motion, config, root,
                                    f"child timed out and was terminated, returncode={code}")
        artifact_status = _child_record_status(already)
        status = artifact_status or ("completed" if code == 0 else "interrupted" if code < 0 or code == 130 else "failed")
        if code < 0 or code == 130:
            _record_interrupted(entry, task, motion, config, root,
                                f"child terminated by signal/KeyboardInterrupt, returncode={code}")
        elif code != 0 and artifact_status is None:
            _write_failure(entry, task, motion, config, C.RunStatus.FAILED,
                           f"child exited with returncode {code}", root)
        records.append({"run_id": entry["run_id"], "returncode": code, "reported_status": status,
                        "wall_time_s": time.perf_counter() - before, "log": str(log.name)})
    _atomic(root / "execution_summary.json", {"runs": records,
                                                "status_counts": dict(Counter(row["reported_status"] for row in records)),
                                                "pending_evaluation": PENDING_REASON})
    print(json.dumps({"execution_count": len(records),
                      "status_counts": dict(Counter(row["reported_status"] for row in records))}, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only")
    parser.add_argument("--worker-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(args.output_dir).resolve()
    if not args.resume and not args.worker_run and root.exists():
        parser.error("output-dir must be new unless --resume is set")
    try:
        config = _normalized_config(args.config)
        config["experiment"] = args.experiment
        if config["experiment"] not in EXPERIMENTS:
            raise RunnerError("invalid advertised experiment")
        if args.resume and not root.is_dir():
            raise RunnerError("--resume needs an existing output directory")
        if args.worker_run:
            snapshot = _json(root / "frozen_config.json")
            config["experiment"] = args.experiment
            entries = _plan_entries(config, _context(config, root)[0], args.split, args.only)
            if len(entries) != 1 or snapshot["identity"] != entries[0]["identity"]:
                raise RunnerError("worker snapshot/task identity mismatch")
            _child(config, entries[0], root)
            return 0
        return _run_parent(config, args, root)
    except KeyboardInterrupt:
        raise
    except (RunnerError, B.BlackBoxError) as error:
        print(f"benchmark runner failed: {error}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired as error:
        raise RunnerError(f"subprocess cleanup failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
