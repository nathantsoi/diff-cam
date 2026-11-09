"""T4.1a auditable benchmark runner, not a claim of a completed real pilot.

Only ``pilot`` / ``dev`` is implemented in this card. Other enum values fail clearly.
One child process per condition/seed owns at most one simulator. Jobs launch serially
(also for GPU) to satisfy runtime/device limits. A later T2.2 task and T1.3 posted
Evaluator plug into the same frozen-run protocol.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import fnmatch
import importlib
from functools import partial
import json
import math
import os
import subprocess
import sys
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from algorithms import blackbox as B
from algorithms import objectives as O
from algorithms import optimize as adam
from algorithms import parameterization as P
from algorithms import run_artifacts as R
from benchmarks import audit as audit
from benchmarks import contracts as C
from benchmarks import pilot_task as pilot_io
from cam.posted_evaluator import PostedEvaluator, RefinementSettings, _analytic_mask
from cam.posts import post_motion_plan
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
    kind = config.get("task_source", "pilot_pocket")
    if kind == "prototype_analytic":
        return (*_prototype(), _prototype_evaluator, None)
    if kind == "pilot_pocket":
        # Builtin pilot source lives in the application repository, not the
        # caller's artifact output directory. Preserve the legacy plugin argument.
        return _pilot_context(config, REPOSITORY)
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


class PilotCandidateEvaluator(O.CandidateEvaluator):
    """Same physical task, with the correct sampled/analytic representations.

    The differentiable model takes the sampled photocopy of T. The production
    posted task deliberately retains its analytic T and exclusion declaration;
    these are not interchangeable sources of geometry or identity hashes.
    """

    def __init__(self, task, sampled_task, motion, parameterization, sim, config, seed, sources):
        super().__init__(task=sampled_task, motion=motion, parameterization=parameterization,
                         simulator=sim, spec=O.ObjectiveSpec(k=float(config["k_smooth"])), seed=seed)
        if sampled_task.operation_target.geometry.kind != "sampled" or sim.target_shape != "sampled":
            raise RunnerError("pilot requires task and simulator to use the same sampled operation target")
        self.task = task                 # production analytic operation target
        self.sampled_task = sampled_task # differentiable sampled operation target
        self.geometry_sources = sources


def _pilot_exclusion_tolerance(task, config):
    params = task.operation_target.geometry.params
    declared = params.get("grid_exclusion_tolerance_mm3")
    configured = config.get("exclusion_tolerance_mm3")
    if configured is not None and (
            isinstance(configured, bool)
            or not isinstance(configured, (int, float))
            or not math.isfinite(configured) or configured < 0):
        raise RunnerError("pilot exclusion tolerance in config must be finite and nonnegative mm^3")
    if declared is None:
        if task.operation_target.excluded_volume_mm3 == 0:
            declared = 0.0
        else:
            # Older builder: require config, never invent the physical volume tolerance.
            declared = configured
    if config.get("require_grid_exclusion_tolerance") and configured is None:
        raise RunnerError("real pilot requires config.exclusion_tolerance_mm3 with a physical-mm^3 value")
    if configured is not None and declared is not None and not math.isclose(
            float(configured), float(declared), rel_tol=0.0, abs_tol=1e-12):
        raise RunnerError("pilot exclusion tolerance in config disagrees with immutable target declaration")
    if isinstance(declared, bool) or not isinstance(declared, (int, float)) or not math.isfinite(declared) or declared < 0:
        raise RunnerError("operation-target grid exclusion tolerance must be finite and nonnegative mm^3")
    return float(declared)


def _sampled_pilot_sources(task, repository=None):
    """Generate and verify immutable sampled geometry in the explicit repository."""
    from cam.posted_evaluator import _centres

    stock_sdf, target_sdf, required, retained = pilot_io.source_fields(task)
    if np.any(np.asarray(target_sdf, dtype=np.float32) == 0.0):
        raise RunnerError("sampled operation target has zero-valued SDF cells: closed <=0 "
                          "membership and simulator strict <0 interior would disagree")
    centres = _centres(task.grid)
    if not np.array_equal((stock_sdf <= 0), _analytic_mask(task.stock, centres)):
        raise RunnerError("simulator stock is not equivalent to analytic task stock")
    if not np.array_equal((target_sdf <= 0),
                          _analytic_mask(task.operation_target.geometry, centres)):
        raise RunnerError("sampled target is not the declared analytic operation target T")
    if np.any(required & retained):
        raise RunnerError("pilot required-retained cell sets are not disjoint")
    if not np.isfinite(stock_sdf).all() or not np.isfinite(target_sdf).all():
        raise RunnerError("sampled pilot geometry has non-finite SDF values")
    source_root = (REPOSITORY if repository is None else Path(repository)).resolve()
    project_root = REPOSITORY.resolve()
    directory = source_root / "external" / "benchmark-inputs" / task.task_hash[7:]
    directory.mkdir(parents=True, exist_ok=True)
    sources = {}
    for label, array in (("stock", stock_sdf), ("target", target_sdf)):
        payload = R.numpy_bytes(np.asarray(array, dtype=np.float32))
        expected = C.sha256_bytes(payload)
        path = directory / f"{label}.npy"
        if path.exists():
            if C.sha256_bytes(path.read_bytes()) != expected:
                raise RunnerError(f"cached source changed at same immutable identity: {path}")
            with path.open("rb") as stream:
                loaded = np.load(stream, allow_pickle=False)
            if loaded.shape != task.grid.shape or C.sha256_bytes(path.read_bytes()) != expected:
                raise RunnerError(f"cached sampled payload is inconsistent: {path}")
        else:
            R.ArtifactWriter._replace_bytes(path, payload)
        relative = path.relative_to(project_root).as_posix() if source_root == project_root else str(path)
        sources[label] = C.GeometrySource(kind="sampled", path=relative, sha256=expected,
                                          grid=task.grid, frame=C.Frame.WORK)
    return sources


def _pilot_context(config, repository):
    if config.get("arch", "cpu") != "cpu":
        raise RunnerError("pilot_pocket: only arch='cpu' is validated here; no unvalidated GPU budget claim")
    pitch = config.get("grid_pitch_mm", pilot_io.GRID_PITCH)
    if isinstance(pitch, bool) or not isinstance(pitch, (int, float)) or not math.isfinite(pitch) or pitch <= 0:
        raise RunnerError("grid_pitch_mm must be finite and positive")
    task = pilot_io.build_pilot_task(voxel_mm=float(pitch))
    from cam.seeds import pocket_with_island_seed
    motion = pocket_with_island_seed(task)
    motion.validate_for_task(task)
    parameterization = pilot_io.fit_parameterization(motion, controls=12)
    _sampled_pilot_sources(task, repository)
    _pilot_exclusion_tolerance(task, config)
    return task, motion, parameterization, partial(_pilot_evaluator, repository=repository), _pilot_post_factory


def _pilot_evaluator(task, motion, parameterization, config, seed, *, repository=None):
    guard = os.environ.get("DIFFCAM_RUNNER_SIMULATOR", "0")
    if guard != "0":
        raise RunnerError("each pilot child must own exactly one simulator")
    os.environ["DIFFCAM_RUNNER_SIMULATOR"] = "1"
    import taichi as ti
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f32)
    if task.operation_target.geometry.kind != "analytic":
        raise RunnerError("pilot factory requires the original analytic operation target")
    sources = _sampled_pilot_sources(task, repository)
    sampled_task = dataclasses.replace(
        task, operation_target=dataclasses.replace(task.operation_target, geometry=sources["target"]))
    target_work = sampled_task.operation_target.geometry
    if target_work.kind != "sampled" or target_work.sha256 != sources["target"].sha256:
        raise RunnerError("pilot requires a sampled operation target for CandidateEvaluator")
    initial = np.asarray(parameterization.initial_position_mm)
    normalized = (initial - task.grid.origin_mm) / task.grid.extent_mm
    sim = CSGSimulatorDelta(
        resolution=max(task.grid.shape), max_steps=parameterization.num_steps,
        stock_size_mm=task.grid.extent_mm,
        stock_box_origin_mm=task.stock.params["min_mm"],
        voxel_size_mm=task.grid.voxel_mm, target_shape=None,
        target_geometry=target_work, stock_geometry=sources["stock"],
        tool_start=tuple(normalized), enforce_speed_limits=False,
        init_taichi=False, k_init=float(config["k_smooth"]),
    )
    sim.set_tool_start(tuple(normalized))
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.tool_cut_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    # Sampled fields are installed by __init__, but the reusable target grid is
    # explicitly baked as in the T2.0 examples. Hard CandidateEvaluator reads
    # this grid, not the raw source array used by source_fields/pointwise loss.
    sim.bake_target_grid()
    sim.set_target_volume()
    with open(target_work.path, "rb") as stream:
        raw_target = np.load(stream, allow_pickle=False)
    if not np.array_equal(sim.target.to_numpy() < 0.0, raw_target < 0.0):
        raise RunnerError("simulator's baked target material mask differs from operation target T")
    return PilotCandidateEvaluator(task, sampled_task, motion, parameterization, sim, config, seed, sources)


def _pilot_post_factory(task, config):
    if not callable(getattr(pilot_io, "production_settings", None)):
        raise RunnerError("T2.2b production_settings API is required; merge it before real pilot execution")
    return PilotPostedEvaluator(task, config)


def _pilot_refinement_settings(task, config):
    """Materialize the explicit frozen refinement, checked against task policy."""
    names = ("max_step_mm", "subdivisions", "compute_bounds", "safety_screen",
             "max_samples_per_segment", "record_stations")
    configured = config.get("posted_refinement")
    if not isinstance(configured, Mapping) or set(configured) != set(names):
        raise RunnerError(f"config.posted_refinement must declare exactly {list(names)}")
    producer = pilot_io.production_settings(task)
    if not isinstance(producer, Mapping):
        raise RunnerError("pilot production_settings must return a settings mapping")
    expected = {name: getattr(RefinementSettings(), name) for name in names}
    expected.update(producer)
    if dict(configured) != expected:
        raise RunnerError(f"posted refinement disagrees with immutable task policy: {expected}")
    step = configured["max_step_mm"]
    if step is not None and (
            isinstance(step, bool) or not isinstance(step, (int, float))
            or not math.isfinite(step) or step <= 0):
        raise RunnerError("posted_refinement.max_step_mm must be null or finite and positive")
    if configured["subdivisions"] is not None:
        _positive_int(configured["subdivisions"], "posted_refinement.subdivisions")
    _positive_int(configured["max_samples_per_segment"], "posted_refinement.max_samples_per_segment")
    # Effective defaults are not silently coarser/finer than this declaration.
    # The expected NC hash is bound to each actual posted byte stream below.
    settings = dict(expected, expected_nc_hash=None,
                    exclusion_tolerance_mm3=_pilot_exclusion_tolerance(task, config))
    return dataclasses.asdict(RefinementSettings.from_mapping(settings))


def _refinement_policy(config):
    """Secondary RS274 refinement-check policy; never hidden overrides."""
    if config.get("task_source", "pilot_pocket") != "pilot_pocket":
        return {"enabled": False}
    fields = {"enabled", "max_step_mm", "parallel_jobs", "timeout_s"}
    policy = config.get("refinement_check")
    if not isinstance(policy, Mapping) or set(policy) != fields:
        raise RunnerError(f"refinement_check must declare exactly {sorted(fields)}")
    if not isinstance(policy["enabled"], bool):
        raise RunnerError("refinement_check.enabled must be boolean")
    step = policy["max_step_mm"]
    if isinstance(step, bool) or not isinstance(step, (int, float)) or step != 0.125:
        raise RunnerError("refinement_check.max_step_mm must be the coordinator-declared 0.125")
    jobs = _positive_int(policy["parallel_jobs"], "refinement_check.parallel_jobs")
    if jobs > 2:
        raise RunnerError("refinement_check parallel_jobs must be bounded at 2")
    timeout = policy["timeout_s"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise RunnerError("refinement_check.timeout_s must be finite and positive")
    return dict(policy)


class PilotPostedEvaluator(PostedEvaluator):
    """Real production posted evaluator with unchanged, bounded refinement."""

    def __init__(self, task, config):
        super().__init__()
        self.task_hash = task.task_hash
        self.settings = MappingProxyType(_pilot_refinement_settings(task, config))
        self.timings = []

    def evaluate_posted_gcode(self, task, nc_bytes, dialect, settings):
        if task.task_hash != self.task_hash:
            raise RunnerError("pilot posted evaluator received a different task")
        merged = dict(self.settings)
        raw = {} if settings is None else dict(settings)
        if raw:
            raise RunnerError("pilot posted settings are fixed by the declared task; per-call overrides are unsupported")
        merged["expected_nc_hash"] = C.nc_hash(nc_bytes)
        before = time.perf_counter()
        event = {"dialect": C.Dialect(dialect).value, "nc_hash": merged["expected_nc_hash"]}
        try:
            return super().evaluate_posted_gcode(task, nc_bytes, dialect, merged)
        finally:
            event["wall_seconds"] = round(time.perf_counter() - before, 9)
            self.timings.append(event)



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


def _finite_seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RunnerError(f"invalid measured duration {name}: expected finite nonnegative seconds")
    return float(value)


def _budget_sizes(config, method):
    """Match the optimizer's actual per-parameterization/per-generation work.

    This is only workload estimation, not computation. CMA generations contain
    a whole evaluated population; unrequested or excess simulations are never
    This is only workload estimation, not computation. CMA generations contain
    a whole evaluated population; unrequested or excess simulations are never
    started here to find sizes experimentally.
    """
    if method == "nominal-cam":
        return 1, 0
    budget = _budget(config, method)
    iterations = budget.candidate_evaluations
    if budget.max_generations is not None:
        iterations = min(iterations, budget.max_generations)
    if method.endswith("cmaes") and budget.population_size is not None:
        iterations = min(iterations, budget.candidate_evaluations // budget.population_size)
    if iterations <= 0:
        raise RunnerError("the requested generation budget selects no work")
    candidates = 1 if method == "nominal-cam" else iterations
    if method.endswith("cmaes"):
        candidates *= (budget.population_size or 2)
        # A generation evaluates members, but can trigger at most one validation.
    # For Adam, candidates and iterations are the same count.
    validations = min(budget.max_validation_calls, iterations,
                      candidates // budget.validation_interval if budget.validation_interval else 0)
    if budget.max_validation_calls and not budget.validation_interval:
        raise RunnerError("validation_interval must be positive when validation is enabled")
    return candidates, validations


def _pecent(config: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predicted = []
    supplied = config.get("estimated_seconds_per_candidate", {})
    if not isinstance(supplied, Mapping):
        raise RunnerError("estimated_seconds_per_candidate must be a mapping")
    evidence = config.get("timing_evidence")
    if evidence is not None:
        required = {"schema", "task_hash", "candidate_seconds", "posted_seconds", "constructor_seconds",
                    "worker_wall_seconds", "posted_seconds_dialects"}
        if not isinstance(evidence, Mapping) or any(key not in evidence for key in required):
            raise RunnerError("timing_evidence must include task identity, all measurement rates and wall time")
        if evidence["schema"] != "diffcam-pilot-timing/1":
            raise RunnerError("unrecognized timing_evidence schema")
        if not isinstance(evidence["task_hash"], str) or not evidence["task_hash"].startswith("sha256:"):
            raise RunnerError("timing evidence task hash must have explicit SHA-256 form")
        candidates_measured = evidence["candidate_seconds"]
        if not isinstance(candidates_measured, Mapping):
            raise RunnerError("timing evidence candidate_seconds must be a structured mapping")
        for kind in ("smooth_forward", "smooth_gradient", "hard_forward"):
            _finite_seconds(candidates_measured.get(kind), kind)
        events = evidence["posted_seconds_dialects"]
        if not isinstance(events, (list, tuple)) or len(events) != 2:
            raise RunnerError("timing evidence requires two labelled posted-evaluation durations")
        try:
            labelled_post = {C.Dialect(event["dialect"]).value: _finite_seconds(event["wall_seconds"],
                                                                                "posted evaluation")
                             for event in events}
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RunnerError(f"invalid posted-evidence event: {error}") from error
        if set(labelled_post) != {dialect.value for dialect in C.Dialect}:
            raise RunnerError("posted timings must contain exactly one RS274 and one Haas evaluation")
        raw_post = evidence["posted_seconds"]
        if not isinstance(raw_post, (list, tuple)) or len(raw_post) != 2:
            raise RunnerError("posted timing must contain exactly RS274 and Haas durations")
        if sorted(_finite_seconds(value, "posted evaluation") for value in raw_post) != sorted(labelled_post.values()):
            raise RunnerError("labelled and anonymous posted timings disagree")
        measured_post = sum(labelled_post.values())
    refinement_measurement = evidence.get("refinement_check_seconds") if evidence is not None else None
    if _refinement_policy(config)["enabled"]:
        if not isinstance(refinement_measurement, Mapping):
            raise RunnerError("refinement_check is enabled but no actual refinement timing was measured")
        refined_seconds = _finite_seconds(refinement_measurement.get("wall_seconds"), "refinement evaluation")
        if refinement_measurement.get("settings", {}).get("max_step_mm") != 0.125:
            raise RunnerError("timing evidence refinement max_step_mm is not 0.125")
        if not isinstance(refinement_measurement.get("comparison"), Mapping):
            raise RunnerError("timing evidence lacks the primary/refined verdict comparison")
    else:
        refined_seconds = 0.0
    for entry in entries:
        candidates, validations = _budget_sizes(config, entry["method"])
        raw = supplied.get(entry["method"])
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
            raw = None
        extra = {}
        if evidence is not None:
            kind = "smooth_forward" if entry["method"] == "smooth-cmaes" else (
                "smooth_gradient" if entry["method"] in ("fixed-adam", "annealed-adam") else "hard_forward")
            unit = _finite_seconds(evidence["candidate_seconds"].get(kind), kind)
            setup = _finite_seconds(evidence["constructor_seconds"], "constructor_seconds")
            other = (_finite_seconds(evidence["worker_wall_seconds"], "worker_wall_seconds")
                     - setup - sum(_finite_seconds(evidence["candidate_seconds"].get(key), key)
                                   for key in ("smooth_forward", "smooth_gradient", "hard_forward"))
                     - measured_post - refined_seconds)
            extra = {
                "constructor_seconds": setup,
                "candidate_seconds": unit * candidates,
                "validation_seconds": _finite_seconds(evidence["candidate_seconds"]["hard_forward"], "hard_forward") * validations,
                "posted_seconds": measured_post,
                "other_worker_seconds": other,
                "estimation": "one measured point per evaluation type, not a timing confidence interval",
            }
            estimate = setup + extra["candidate_seconds"] + extra["validation_seconds"] + measured_post + other
            if other < -1e-9:
                raise RunnerError("timing evidence has a negative process-overhead residual")
            # Candidate durations are recorded to nanoseconds; do not make a
            # valid worker fail solely because a sum rounds a fraction above its wall clock.
            other = max(other, 0.0)
            extra["other_worker_seconds"] = other
            estimate = setup + extra["candidate_seconds"] + extra["validation_seconds"] + measured_post + other
            note = "one measured point per evaluation type in one child; workload estimate, not a timing interval or launch permission"
        elif raw is not None:
            estimate = raw * (candidates + validations)
            note = "caller-supplied per-candidate estimate"
        else:
            estimate = None
            note = "unknown without a real child measurement"
        predicted.append({"run_id": entry["run_id"], "candidate_evaluations": candidates,
                          "validation_calls": validations,
                          "estimated_run_wall_time_s": estimate, "jit_warmup": note, **extra})
    return predicted


def _refinement_projection(config, entries, evidence, root):
    """Serialize separate refinement-phase cost without charging primary runs."""
    if evidence is None or not _refinement_policy(config)["enabled"]:
        return {"enabled": False, "estimated_phase_wall_time_s": 0.0}
    if not isinstance(evidence, Mapping):
        raise RunnerError("refinement cost requires measured typed evaluation evidence")
    event = evidence.get("refinement_check_seconds")
    if not isinstance(event, Mapping):
        raise RunnerError("enabled refinement_check lacks real nominal RS274 refinement time")
    per_program = _finite_seconds(event.get("wall_seconds"), "refinement evaluation")
    artifacts = evidence.get("evaluation_artifacts", {}).get("rs274_refined")
    if not isinstance(artifacts, Mapping):
        raise RunnerError("refinement timing lacks its evaluation artifact/hash link")
    required = {"nc_hash", "evaluation_path", "evaluation_sha256"}
    if any(not isinstance(artifacts.get(key), str) or not artifacts[key] for key in required):
        raise RunnerError(f"refinement artifact requires {sorted(required)}")
    nc_path = audit._artifact_path(root, artifacts["nc_path"])
    nc_bytes = nc_path.read_bytes()
    if C.nc_hash(nc_bytes) != event.get("nc_hash"):
        raise RunnerError("refinement timing evidence does not match the primary RS274 bytes")
    evaluation_path = audit._artifact_path(root, artifacts["evaluation_path"])
    payload = evaluation_path.read_bytes()
    if C.sha256_bytes(payload) != artifacts["evaluation_sha256"]:
        raise RunnerError("refinement evaluation file no longer matches its recorded SHA-256")
    result = C.loads(payload)
    if result.dialect != C.Dialect.RS274 or result.task_hash != evidence.get("task_hash"):
        raise RunnerError("refinement artifact task/dialect mismatch")
    jobs = _refinement_policy(config)["parallel_jobs"]
    count = len(entries)
    return {
        "enabled": True, "dialect": "rs274", "max_step_mm": 0.125,
        "expected_program_count": count, "parallel_jobs": jobs,
        "per_program_seconds": per_program,
        "serial_evaluation_work_seconds": per_program * count,
        "estimated_phase_wall_time_s": per_program * ((count + jobs - 1) // jobs),
        "comparison": event.get("comparison"),
        "evidence": artifacts,
        "estimation_basis": "one real measurement; independent subprocess slots after primary runs, not a confidence interval or guaranteed parallel speedup",
    }


def _measure_budget(config, root, *, evaluator_override=None, capture_dir=None,
                    refinement_evaluator_override=None):
    """Measure one of each execution kind, with no population/Adam optimization.

    ``evaluator_override`` is a pure in-process test seam (no untrusted module
    path/name loading). Production callers always construct one real registered
    CandidateEvaluator and one real production PostedEvaluator factory.
    """
    from datetime import datetime, timezone
    started_at = datetime.now(timezone.utc).isoformat()
    if capture_dir is not None:
        if capture_dir.exists():
            raise RunnerError(f"timing evidence directory already exists: {capture_dir}")
        capture_dir.mkdir(parents=True)
    task, motion, parameterization, evaluator_factory, posted_factory = _context(config, root)
    if task.split != C.Split.DEV:
        raise RunnerError("timing collection is restricted to the declared dev task")
    created = time.perf_counter()
    evaluator = evaluator_override or evaluator_factory(task, motion, parameterization, config, 0)
    constructor_seconds = time.perf_counter() - created
    zero = np.zeros(parameterization.num_params)
    if not parameterization.in_bounds(zero):
        raise RunnerError("timing sample is outside the declared parameterization bounds")
    records = {}
    for name, gradient in (
        ("smooth_forward", False),
        ("smooth_gradient", True),
        ("hard_forward", False),
    ):
        started = time.perf_counter()
        if gradient:
            evaluated = evaluator.evaluate_candidate_with_gradient(zero, f"timing-{name}")
            result, gradient_theta = evaluated.result, evaluated.gradient_theta
            if gradient_theta is None or not np.isfinite(gradient_theta).all():
                raise RunnerError("timing gradient is missing or non-finite")
        else:
            result = evaluator.evaluate_candidate(zero, f"timing-{name}", "smooth" if name.startswith("smooth") else "hard")
        seconds = time.perf_counter() - started
        if not result.valid or not math.isfinite(result.objective):
            raise RunnerError(f"timing evaluation {name} is not a finite valid candidate: {result.invalid_reason}")
        records[name] = round(seconds, 9)
    plan = evaluator.candidate_motion(zero)
    before_post = time.perf_counter()
    posting = R.post_motion(task, plan)
    if any(post.status != C.PostStatus.SUPPORTED for post in posting.posted):
        raise RunnerError("timing measurement requires support for both posted dialects")
    post_builder = posted_factory(task, config)
    primary_rs274 = None
    evaluation_artifacts = {}
    for post in posting.posted:
        nc, eval_name = C.DIALECT_ARTIFACTS[post.dialect]
        actual = post_builder.evaluate_posted_gcode(task, posting.files[nc], post.dialect, {})
        if actual.dialect != post.dialect or actual.nc_hash != post.nc_hash or actual.task_hash != task.task_hash:
            raise RunnerError("measurement posted-evaluation identity mismatches source bytes")
        if post.dialect == C.Dialect.RS274:
            primary_rs274 = actual
        if not actual.efficiency_eligible:
            raise RunnerError(
                f"timing nominal candidate is not geometry/safety eligible in {post.dialect.value}: "
                f"coverage={actual.coverage}, safety={actual.safety_pass}, geometry={actual.geometry_pass}")
        if not actual.metrics["removed_volume_mm3"].value > 0:
            raise RunnerError("timing measurement may not treat a no-cut path as completed material removal")
        if capture_dir is not None:
            nc_path = capture_dir / f"{post.dialect.value}.nc"
            result_path = capture_dir / eval_name
            nc_payload = posting.files[nc]
            result_payload = R.record_bytes(actual)
            R.ArtifactWriter._replace_bytes(nc_path, nc_payload)
            R.ArtifactWriter._replace_bytes(result_path, result_payload)
            evaluation_artifacts[post.dialect.value] = {
                "nc_path": str(nc_path.relative_to(root)),
                "nc_hash": post.nc_hash,
                "evaluation_path": str(result_path.relative_to(root)),
                "evaluation_sha256": C.sha256_bytes(result_payload),
            }
    refinement = None
    if config.get("task_source", "pilot_pocket") == "pilot_pocket" and _refinement_policy(config)["enabled"]:
        if primary_rs274 is None:
            raise RunnerError("refinement measurement requires the primary RS274 result")
        refinement_evaluator = refinement_evaluator_override or PostedEvaluator()
        evaluate_refinement = getattr(refinement_evaluator, "evaluate_posted_gcode", refinement_evaluator)
        if not callable(evaluate_refinement):
            raise RunnerError("refinement evaluator does not implement the typed protocol")
        nc_bytes = posting.files["rs274.nc"]
        refinement_policy = _refinement_policy(config)
        settings = _refinement_settings_from_primary(primary_rs274, refinement_policy)
        started = time.perf_counter()
        refined = evaluate_refinement(task, nc_bytes, C.Dialect.RS274, settings)
        seconds = _finite_seconds(time.perf_counter() - started, "refinement evaluation")
        refined = _verified_refinement_result(primary_rs274, refined, refinement_policy)
        payload = R.record_bytes(refined)
        if capture_dir is not None:
            refined_path = capture_dir / "eval_posted_rs274_refined.json"
            R.ArtifactWriter._replace_bytes(refined_path, payload)
            evaluation_artifacts["rs274_refined"] = {
                "nc_path": evaluation_artifacts["rs274"]["nc_path"],
                "nc_hash": primary_rs274.nc_hash,
                "evaluation_path": str(refined_path.relative_to(root)),
                "evaluation_sha256": C.sha256_bytes(payload),
            }
        else:
            evaluation_artifacts["rs274_refined"] = {
                "nc_hash": primary_rs274.nc_hash,
                "evaluation_path": None, "evaluation_sha256": C.sha256_bytes(payload),
            }
        refinement_comparison = _refinement_comparison(primary_rs274, refined)
        refinement_outcome, refinement_comparison = _conservative_outcome(
            primary_rs274, refined, refinement_comparison, task)
        refinement = {
            "dialect": "rs274", "wall_seconds": seconds,
            "nc_hash": primary_rs274.nc_hash,
            "settings": refined.discretization["settings"],
            "comparison": refinement_comparison,
            "conservative_outcome": refinement_outcome,
        }
    posted_times = list(post_builder.timings)
    expected_events = [(post.dialect.value, post.nc_hash) for post in posting.posted]
    actual_events = [(event["dialect"], event["nc_hash"]) for event in posted_times]
    if len(posted_times) != 2 or actual_events != expected_events:
        raise RunnerError("exactly two real, identity-matched posted evaluations must be timed")
    posted_seconds = [float(record["wall_seconds"]) for record in posted_times]
    return {
        "schema": "diffcam-pilot-timing/1",
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TAICHI_NUM_THREADS")
        },
        "task_hash": task.task_hash,
        "tool_setup_hash": task.tool_setup_hash,
        "task_id": task.task_id,
        "grid_shape": list(task.grid.shape),
        "voxel_mm": task.grid.voxel_mm,
        "parameter_steps": parameterization.num_steps,
        "constructor_seconds": constructor_seconds,
        "config_sha256": C.sha256_bytes(R.json_bytes(config)),
        "code_sha256": {
            "runner": C.sha256_bytes(Path(__file__).read_bytes()),
            "posted_evaluator": C.sha256_bytes(Path(sys.modules[PostedEvaluator.__module__].__file__).read_bytes()),
            "strict_parser": C.sha256_bytes(Path(sys.modules["cam.gcode_parser"].__file__).read_bytes()),
            "post_writer": C.sha256_bytes(Path(sys.modules["cam.posts"].__file__).read_bytes()),
            "task_builder": C.sha256_bytes(Path(sys.modules[pilot_io.__name__].__file__).read_bytes()),
        },
        "candidate_seconds": records,
        "posted_seconds": posted_seconds,
        "posted_seconds_dialects": posted_times,
        "evaluation_artifacts": evaluation_artifacts,
        "refinement_check_seconds": refinement,
        "source_stage": {
            "target": "analytic operation target for posted execution; exact sampled copy for simulator",
            "sources": {name: {"path": source.path, "sha256": source.sha256}
                        for name, source in evaluator.geometry_sources.items()},
        },
        "started_at": started_at,
        "worker_wall_seconds": time.perf_counter() - created,
        "post_file_preparation_seconds": time.perf_counter() - before_post,
        "scope": "one real candidate of each kind, both primary dialects, and one real RS274 refinement per enabled policy; not the 7-run pilot",
    }


def _measure_timing_subprocess(config, root):
    """Spawn one exemplar worker at the original task and collect timing."""
    opt_in = config.get("measure_budget", True)
    if opt_in is not True and type(opt_in) is not bool:
        raise RunnerError("measure_budget must be an explicit boolean")
    task, motion, parameterization, evaluator_factory, posted_factory = _context(config, root)
    if opt_in is False:
        if not isinstance(config.get("timing_evidence"), Mapping):
            raise RunnerError("measure_budget=false requires caller-supplied timing_evidence")
        return config["timing_evidence"]
    if not isinstance(config.get("timing_evidence"), (Mapping, type(None))):
        raise RunnerError("timing_evidence must be a structured mapping when supplied")
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    _atomic(root / "timing-config.json", config)
    started = time.perf_counter()
    command = [sys.executable, str(Path(__file__).resolve()),
               "--config", str(root / "timing-config.json"),
               "--output-dir", str(root), "--split", "dev", "--experiment", "pilot",
               "--dry-run", "--timing-worker-run"]
    timeout = config.get("timing_timeout_s", 1800.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise RunnerError("timing_timeout_s must be finite and positive")
    log = root / "logs" / "budget_measurement.log"
    env = dict(os.environ)
    env["DIFFCAM_RUNNER_SIMULATOR"] = "0"
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=REPOSITORY, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            raise RunnerError(f"timing worker terminated: {code}")
    if code:
        raise RunnerError(f"timing worker failed: {code}; see {log}")
    elapsed = time.perf_counter() - started
    evidence_path = root / "budget-timings.json"
    if not evidence_path.is_file():
        raise RunnerError("timing worker did not produce a budget-timings.json report")
    data = _json(evidence_path)
    if not isinstance(data, dict) or data.get("task_hash") != task.task_hash:
        raise RunnerError("timing evidence source/task identity mismatch")
    data["worker_wall_seconds"] = elapsed
    data["schema"] = "diffcam-pilot-timing/1"
    return data


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
                       "prototype_analytic_task": config.get("task_source", "pilot_pocket") == "prototype_analytic",
                       "task_source": config.get("task_source", "pilot_pocket"),
                       "adam_conditions": "one_deterministic_replicate_each",
                       "cma_conditions": "one_run_per_declared_seed",
                       "nominal_cam": "one_deterministic_run"}}


def _evaluate_outcome(condition: str, evaluator: Any, config: Mapping[str, Any], seed: int) -> Outcome:
    dims = evaluator.parameterization.num_params
    zeros = np.zeros(dims)
    if condition == "nominal-cam":
        best = evaluator.evaluate_candidate(zeros, "nominal-once", "hard")
        ledger = evaluator.ledger.snapshot()
        # One direct CAM evaluation, not an optimizer generation. Preserve the
        # real call counters instead of recording a zero-forward phantom result.
        if (ledger["candidate_evaluations"], ledger["forward_calls"],
                ledger["backward_calls"]) != (1, 1, 0):
            raise RunnerError("nominal-cam ledger does not match one hard candidate evaluation")
        return Outcome(
            seed=0, best=best, best_theta=zeros.copy(), initial_theta=zeros.copy(),
            budget=B.OptimizationBudget(candidate_evaluations=1, population_size=None),
            evaluations=1, validation_calls=0, generations=0,
            forward_calls=ledger["forward_calls"], backward_calls=ledger["backward_calls"],
            invalid_candidates=0 if best.valid else 1,
            termination_reason="deterministic_single_evaluation_of_nominal_cam", history=[],
        )
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
                                      "execution": {"subprocess": os.getpid(), "simulators_in_worker": 1,
                                                    "simulator_tool_start_normalized": (
                                                        evaluator.sim.tool_start[None].to_list()),
                                                    "target_grid_material_mask_verified": isinstance(evaluator, PilotCandidateEvaluator),
                                                    "simulator_geometry_sources": {
                                                        name: json.loads(C.canonical_json(source))
                                                        for name, source in getattr(evaluator, "geometry_sources", {}).items()
                                                    }}}),
        "motion_plan.json": R.record_bytes(motion),
        "initial_trajectory.npy": R.numpy_bytes(initial),
        "trajectory.npy": R.numpy_bytes(best),
        "trajectory_deltas.npy": R.numpy_bytes(np.diff(best, axis=0)),
    })
    evaluated = {}
    if posted_factory is not None:
        evaluator_for_posts = posted_factory(task, config)
        for post in posting.posted:
            if post.status != C.PostStatus.SUPPORTED:
                raise RunnerError(f"cannot complete run: no supported {post.dialect.value} post: {post.reason}")
            nc_name, eval_name = C.DIALECT_ARTIFACTS[post.dialect]
            actual = evaluator_for_posts.evaluate_posted_gcode(task, files[nc_name], post.dialect, {})
            if actual.dialect != post.dialect or actual.nc_hash != post.nc_hash or actual.task_hash != task.task_hash:
                raise RunnerError("posted evaluator returned a mismatched dialect/task/NC hash")
            evaluated[post.dialect] = actual
            files[eval_name] = R.record_bytes(actual)
    complete = posted_factory is not None and set(evaluated) == set(C.Dialect)
    legacy_incomplete = posted_factory is None
    if not complete and not legacy_incomplete:
        raise RunnerError("complete requires real eval_posted_rs274.json and eval_posted_haas.json")
    status = C.RunStatus.COMPLETE if complete else C.RunStatus.INCOMPLETE
    reason = None if complete else PENDING_REASON
    if complete and getattr(evaluator_for_posts, "timings", None):
        details = json.loads(files["metrics.json"])
        details["execution"]["posted_evaluation_timings"] = evaluator_for_posts.timings
        details["execution"]["posted_sha256_pairs"] = {
            key: {"nc_hash": value.nc_hash, "evaluation_sha256": C.sha256_bytes(C.dumps(value).encode("utf-8"))}
            for key, value in evaluated.items()
        }
        files["metrics.json"] = R.json_bytes(details)
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


def _refinement_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _refinement_jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {key: _refinement_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_refinement_jsonable(item) for item in value]
    return value


def _refinement_comparison(primary, refined):
    """Report actual outcome/quantity changes, never a preferred verdict."""
    result = {}
    for name in ("safety_pass", "geometry_pass"):
        result[name] = {"primary": getattr(primary, name), "refined": getattr(refined, name),
                        "changed": getattr(primary, name) != getattr(refined, name)}
    result["coverage"] = {"primary": primary.coverage.value, "refined": refined.coverage.value,
                          "changed": primary.coverage != refined.coverage}
    result["efficiency_eligible"] = {
        "primary": primary.efficiency_eligible, "refined": refined.efficiency_eligible,
        "changed": primary.efficiency_eligible != refined.efficiency_eligible,
    }
    for name in ("residual_volume_mm3", "gouge_volume_mm3", "max_surface_error_mm"):
        expected_unit = "mm" if name.endswith("_mm") else "mm^3"
        values = []
        for result_doc in (primary, refined):
            metric = result_doc.metrics.get(name)
            if metric is None or metric.unit != expected_unit:
                raise RunnerError(f"refinement metric {name} lacks expected {expected_unit}")
            if metric.value is not None and not math.isfinite(metric.value):
                raise RunnerError(f"refinement metric {name} is non-finite")
            values.append(metric.value)
        delta = values[1] - values[0] if all(value is not None for value in values) else None
        result[name] = {"primary": values[0], "refined": values[1], "delta": delta,
                        "unit": expected_unit}
    result["any_outcome_verdict_change"] = any(
        result[name]["changed"] for name in ("safety_pass", "geometry_pass", "coverage", "efficiency_eligible"))
    return result


def _conservative_outcome(primary, refined, comparison, task):
    comparison = dict(comparison)
    if not isinstance(task, C.TaskSpec):
        raise RunnerError("conservative completion requires the actual pilot TaskSpec")
    if primary.task_hash != task.task_hash or refined.task_hash != task.task_hash:
        raise RunnerError("conservative outcomes require one actual task identity")
    task_info = audit.TaskInfo(
        task.task_id, task.task_hash, task.family,
        completion_residual_mm3=task.tolerances.completion_residual_mm3)
    primary_completion = audit._completion_value(primary, task_info)
    refined_completion = audit._completion_value(refined, task_info)
    comparison["completion_pass"] = {
        "primary": primary_completion, "refined": refined_completion,
        "changed": primary_completion != refined_completion,
    }
    comparison["any_outcome_verdict_change"] = any(
        comparison[name]["changed"]
        for name in ("safety_pass", "geometry_pass", "completion_pass", "coverage", "efficiency_eligible"))

    def combined(left, right):
        if left is False or right is False:
            return False
        if left is None or right is None:
            return None
        if left is True and right is True:
            return True
        raise RunnerError("conservative evaluation outcomes must be bool or None")

    disagreement_names = [
        name for name in ("safety_pass", "geometry_pass", "completion_pass", "coverage")
        if comparison[name]["changed"]]
    completion = combined(primary_completion, refined_completion)
    eligible = bool(primary.efficiency_eligible and refined.efficiency_eligible
                    and completion is True and not disagreement_names)
    return {
        "safety_pass": combined(primary.safety_pass, refined.safety_pass),
        "geometry_pass": combined(primary.geometry_pass, refined.geometry_pass),
        "completion_pass": completion,
        "efficiency_eligible": eligible,
        "refinement_disagreement": bool(disagreement_names),
        "disagreement_fields": disagreement_names,
        "excluded_from_efficiency": not eligible,
        "rule": "primary AND refined for safety/geometry/completion at the task threshold; unknown completion => efficiency_eligible false (conservative); None is never upgraded; disagreements exclude efficiency",
    }, comparison


def _refinement_settings_from_primary(primary, policy):
    """Copy every primary sampling control and change only max_step_mm."""
    primary_settings = primary.discretization.get("settings")
    if not isinstance(primary_settings, Mapping):
        raise RunnerError("refinement requires the primary discretization settings")
    settings = dict(primary_settings)
    if settings.get("compute_bounds") is not True or settings.get("safety_screen") is not True:
        raise RunnerError("refinement may not disable primary bounds or safety screening")
    if settings.get("expected_nc_hash") != primary.nc_hash:
        raise RunnerError("refinement requires the primary exact-NC hash policy")
    try:
        copied = RefinementSettings.from_mapping(dict(settings, max_step_mm=policy["max_step_mm"]))
    except ValueError as error:
        raise RunnerError(f"invalid copied primary refinement settings: {error}") from error
    return dataclasses.asdict(copied)


def _verified_refinement_result(primary, refined, policy):
    if not isinstance(refined, C.EvaluationResult) or refined.dialect != C.Dialect.RS274:
        raise RunnerError("refinement did not return a typed RS274 EvaluationResult")
    if refined.nc_hash != primary.nc_hash or refined.task_hash != primary.task_hash:
        raise RunnerError("refinement result has a different task or NC hash")
    expected = _refinement_settings_from_primary(primary, policy)
    if refined.discretization.get("settings") != expected:
        raise RunnerError("refinement result did not preserve every primary setting except max_step_mm")
    return refined


def _safe_refinement_directory(primary_root: Path, auxiliary_root: Path) -> tuple[Path, Path]:
    primary_root, auxiliary_root = primary_root.resolve(), auxiliary_root.resolve()
    if primary_root == auxiliary_root or auxiliary_root.is_relative_to(primary_root):
        raise RunnerError("auxiliary refinement root must be a sibling, not inside a primary run root")
    return primary_root, auxiliary_root


def _load_refinement_case(root: Path, run_id: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,200}", run_id):
        raise RunnerError(f"unsafe refinement run id: {run_id!r}")
    directory = root.resolve() / run_id
    parent = audit._audit_run(directory / "run.json")
    if not parent.complete_and_audited or parent.record.status != C.RunStatus.COMPLETE:
        raise RunnerError(f"refinement requires a completed, cleanly audited primary run: {run_id}")
    record = parent.record
    doc = _json(directory / "config.json")
    task = C.TaskSpec.from_dict(doc["task"])
    primary = C.loads((directory / "eval_posted_rs274.json").read_text(encoding="utf-8"))
    raw_nc = (directory / "rs274.nc").read_bytes()
    nc_hash = C.nc_hash(raw_nc)
    if (record.task_hash != task.task_hash or primary.task_hash != task.task_hash
            or primary.nc_hash != nc_hash or primary.dialect != C.Dialect.RS274):
        raise RunnerError("refinement primary task/dialect/NC identity mismatch")
    return directory, record, doc, task, primary, raw_nc


def _compare_existing_refinement(directory: Path, task, primary_directory: Path, raw_nc):
    """Accept a fully audited, identical copy; never overwrite another result."""
    audit_doc = audit._audit_run(directory / "run.json")
    if not audit_doc.complete_and_audited or audit_doc.record.task_hash != task.task_hash:
        raise RunnerError(f"existing refinement directory is not a clean audited result: {directory}")
    refined = C.loads((directory / "eval_posted_rs274.json").read_text(encoding="utf-8"))
    if refined.nc_hash != C.nc_hash(raw_nc) or refined.task_hash != task.task_hash:
        raise RunnerError("existing refinement has a different task/NC identity")
    if refined.discretization["settings"]["max_step_mm"] != 0.125:
        raise RunnerError("existing refinement has the wrong max_step_mm; refusing to overwrite it")
    if (directory / "haas.nc").read_bytes() != (primary_directory / "haas.nc").read_bytes():
        raise RunnerError("existing refinement changed primary Haas evidence")
    return refined


def _refinement_worker(config, root, run_id, auxiliary_root, *, evaluator_override=None):
    """Evaluate the original RS274 bytes and publish immutable primary/auxiliary evidence."""
    policy = _refinement_policy(config)
    if not policy["enabled"]:
        raise RunnerError("refinement checking was explicitly disabled by the frozen policy")
    root, auxiliary_root = _safe_refinement_directory(root, Path(auxiliary_root))
    directory, record, parent_config, task, primary, raw_nc = _load_refinement_case(root, run_id)
    auxiliary_id = f"{run_id}-refined"
    auxiliary_dir = auxiliary_root / auxiliary_id
    created = time.perf_counter()
    computed = evaluator_override
    if auxiliary_dir.exists():
        refined = _compare_existing_refinement(auxiliary_dir, task, directory, raw_nc)
        elapsed = 0.0
        status = "skipped_complete"
    else:
        if computed is None:
            computed = PostedEvaluator()
        settings = _refinement_settings_from_primary(primary, policy)
        evaluate = getattr(computed, "evaluate_posted_gcode", computed)
        if not callable(evaluate):
            raise RunnerError("refinement evaluator must implement the typed protocol or a callable")
        started = time.perf_counter()
        refined = evaluate(task, raw_nc, C.Dialect.RS274, settings)
        elapsed = time.perf_counter() - started
        status = "complete"
    refined = _verified_refinement_result(primary, refined, policy)
    comparison = _refinement_comparison(primary, refined)
    conservative, comparison = _conservative_outcome(primary, refined, comparison, task)
    payload = R.record_bytes(refined)

    auxiliary_config = dict(parent_config)
    aux_spent = {
        "candidate_evaluations": 0, "forward_calls": 0, "backward_calls": 0,
        "generations": 0, "validation_calls": 0, "posted_refinement_evaluations": 1,
    }
    auxiliary_config.update({
        "run_id": auxiliary_id, "refinement_of": record.run_id,
        "dialect": C.Dialect.RS274.value, "max_step_mm": policy["max_step_mm"],
        "budgets": {"limits": parent_config["budgets"]["limits"], "spent": aux_spent},
    })
    aux_files = {
        "config.json": R.json_bytes(auxiliary_config),
        "history.json": R.json_bytes({
            "stage": "post-pilot-refinement-check", "parent_run_id": record.run_id,
            "new_evaluation": "RS274 with max_step_mm=0.125 on the exact original NC bytes",
            "primary_haas_evidence_copied": True, "wall_seconds": elapsed,
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TAICHI_NUM_THREADS")},
        }),
        "motion_plan.json": (directory / "motion_plan.json").read_bytes(),
        "metrics.json": R.json_bytes({"evaluation": refined.to_dict(),
                                      "comparison": comparison,
                                      "conservative_outcome": conservative,
                                      "refinement_of": record.run_id}),
        "rs274.nc": raw_nc,
        "haas.nc": (directory / "haas.nc").read_bytes(),
        "eval_posted_haas.json": (directory / "eval_posted_haas.json").read_bytes(),
        "eval_posted.json": (directory / "eval_posted.json").read_bytes(),
        "eval_posted_rs274.json": payload,
        "initial_trajectory.npy": (directory / "initial_trajectory.npy").read_bytes(),
        "trajectory.npy": (directory / "trajectory.npy").read_bytes(),
        "trajectory_deltas.npy": (directory / "trajectory_deltas.npy").read_bytes(),
    }
    source_meta = R.source_metadata(REPOSITORY)
    auxiliary_record = R.assemble_record(
        auxiliary_id, task, C.RunStatus.COMPLETE, posting=R.PostingResult(files={}, posted=record.posted),
        artifact_names=tuple(sorted(aux_files)), source_commit=source_meta["source_commit"],
        dirty_patch_hash=source_meta["dirty_patch_hash"])
    if auxiliary_dir.exists():
        # _compare_existing_refinement validated the immutable bundle and sidecar hashes.
        auxiliary_payload = (auxiliary_dir / "run.json").read_bytes()
    else:
        R.ArtifactWriter().write(str(auxiliary_dir), auxiliary_record, aux_files)
        auxiliary_payload = (auxiliary_dir / "run.json").read_bytes()

    # All new artifacts stay in the auxiliary namespace; never write primary files.
    refined_path = auxiliary_dir / "eval_posted_rs274_refined.json"
    if refined_path.exists() and refined_path.read_bytes() != payload:
        raise RunnerError("existing refined artifact differs; refusing to overwrite it")
    if not refined_path.exists():
        R.ArtifactWriter._replace_bytes(refined_path, payload)
    comparison_document = {
        "primary_run_id": record.run_id, "refined_run_id": auxiliary_id,
        "comparison": comparison, "conservative_outcome": conservative,
    }
    comparison_path = auxiliary_dir / "comparison.json"
    comparison_payload = R.json_bytes(comparison_document)
    if comparison_path.exists() and comparison_path.read_bytes() != comparison_payload:
        raise RunnerError("existing refinement comparison differs; refusing to overwrite it")
    if not comparison_path.exists():
        R.ArtifactWriter._replace_bytes(comparison_path, comparison_payload)
    link = {
        "schema": "diffcam-refinement-link/3",
        "primary_run_id": record.run_id,
        "primary_run_path": str(directory.relative_to(root)),
        "primary_metrics_sha256": C.sha256_bytes((directory / "metrics.json").read_bytes()),
        "primary_rs274_evaluation_sha256": C.sha256_bytes(
            (directory / "eval_posted_rs274.json").read_bytes()),
        "primary_haas_evaluation_sha256": C.sha256_bytes(
            (directory / "eval_posted_haas.json").read_bytes()),
        "primary_run_json_sha256": C.sha256_bytes((directory / "run.json").read_bytes()),
        "refined_run_id": auxiliary_id,
        "refined_artifact_path": str(refined_path.relative_to(auxiliary_root)),
        "refined_evaluation_sha256": C.sha256_bytes(payload),
        "auxiliary_record": {"path": str((auxiliary_dir / "run.json").relative_to(auxiliary_root)),
                             "sha256": C.sha256_bytes(auxiliary_payload)},
        "comparison_path": str(comparison_path.relative_to(auxiliary_root)),
        "comparison_sha256": C.sha256_bytes(comparison_payload),
        "conservative_outcome": conservative,
        "comparison": comparison,
    }
    link_path = auxiliary_root / f"{record.run_id}-refinement-link.json"
    if link_path.exists() and _json(link_path) != link:
        raise RunnerError("existing refinement link differs; refusing to overwrite it")
    _atomic(link_path, link)
    return {
        "root": root, "run_id": run_id, "refined_run_id": auxiliary_id,
        "primary_config": parent_config, "source_config": config,
        "source_commit": record.source_commit, "auxiliary_id": auxiliary_id,
        "auxiliary_root": auxiliary_root, "auxiliary_directory": auxiliary_dir,
        "status": status, "comparison": comparison,
        "completion_residual_mm3": task.tolerances.completion_residual_mm3,
        "refined_sha256": C.sha256_bytes(payload),
        "link": link, "conservative_outcome": conservative,
        "auxiliary_link_path": link_path,
        "worker_seconds": time.perf_counter() - created,
    }


def _copy_primary_for_refinement(directory: Path, auxiliary_root: Path):
    """Clone immutable baseline evidence for a distinct, local audit root."""
    target = auxiliary_root / directory.name
    record = C.loads((directory / "run.json").read_text(encoding="utf-8"))
    sources = {name: (directory / name).read_bytes()
               for name in (*record.artifacts, "run.json")}
    if target.exists():
        for name, content in sources.items():
            path = target / name
            if not path.is_file() or path.read_bytes() != content:
                raise RunnerError(f"refinement baseline differs at {path}; refusing to overwrite")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(directory, target)
    for name, content in sources.items():
        if (target / name).read_bytes() != content:
            raise RunnerError(f"baseline copy failed byte verification: {target / name}")
    return target


def _audit_refinement_root(outcomes, auxiliary_root):
    """Expose baseline and refined artifacts through the real benchmark auditor."""
    tasks, runs, comparisons = {}, {}, []
    seen_methods = set()
    for case in outcomes:
        expected = case["manifest"].runs[case["run_id"]]
        task_info = case["manifest"].tasks[expected.task_id]
        if case.get("completion_residual_mm3") is not None:
            task_info = dataclasses.replace(
                task_info, completion_residual_mm3=case["completion_residual_mm3"])
        previous = tasks.get(expected.task_id)
        if previous is not None and previous.task_hash != task_info.task_hash:
            raise RunnerError("one task id has different source hashes in a refinement bundle")
        tasks[expected.task_id] = task_info
        disagreement = bool(case.get("conservative_outcome", {}).get("refinement_disagreement"))
        method = expected.method if not disagreement else f"{expected.method}-refinement-disagreement"
        runs[case["run_id"]] = dataclasses.replace(expected, method=method)
        _copy_primary_for_refinement(case["root"] / case["run_id"], auxiliary_root)
        aux_id = case["refined_run_id"]
        if aux_id in runs:
            raise RunnerError(f"duplicate baseline/refinement run identifier: {aux_id}")
        refined_method = method + "-refinement"
        runs[aux_id] = audit.ExpectedRun(aux_id, expected.task_id, refined_method, expected.seed)
        if disagreement:
            continue
        if expected.method not in seen_methods:
            seen_methods.add(expected.method)
            comparisons.append(audit.ComparisonSpec(
                name=f"{expected.method} RS274 refinement", method=refined_method,
                baseline=expected.method, dialect=C.Dialect.RS274,
                metrics=("residual_volume_mm3", "gouge_volume_mm3", "max_surface_error_mm"),
            ))
    report_flags = {}
    disagreement_counts = {}
    for case in outcomes:
        expected = case["manifest"].runs[case["run_id"]]
        conservative = case.get("conservative_outcome")
        if not isinstance(conservative, Mapping):
            raise RunnerError("refinement result lacks conservative eligibility outcomes")
        report_flags[case["run_id"]] = {
            "condition": expected.method, "seed": expected.seed,
            "safety_pass": conservative["safety_pass"],
            "geometry_pass": conservative["geometry_pass"],
            "completion_pass": conservative["completion_pass"],
            "efficiency_eligible": conservative["efficiency_eligible"],
            "efficiency_excluded": conservative["excluded_from_efficiency"],
            "refinement_disagreement": conservative["refinement_disagreement"],
            "disagreement_fields": conservative["disagreement_fields"],
            "refined_max_step_mm": 0.125,
            "refined_evaluation_path": f"{case['refined_run_id']}/eval_posted_rs274_refined.json",
            "auxiliary_record_path": f"{case['refined_run_id']}/run.json",
            "auxiliary_link_path": os.path.relpath(case["auxiliary_link_path"], auxiliary_root),
        }
        disagreement_counts[expected.method] = (
            disagreement_counts.get(expected.method, 0)
            + int(conservative["refinement_disagreement"]))
    manifest = audit.Manifest(tasks=tasks, runs=runs, comparisons=tuple(comparisons))
    manifest_path = auxiliary_root / "manifest.json"
    if manifest_path.exists():
        if audit.load_manifest(manifest_path) != manifest:
            raise RunnerError("refinement auxiliary manifest differs; refusing mixed audit roots")
    else:
        _atomic(manifest_path, {
            "schema_version": 1,
            "tasks": [{"task_id": task.task_id, "task_hash": task.task_hash, "family": task.family}
                      for task in tasks.values()],
            "expected_runs": [{"run_id": item.run_id, "task_id": item.task_id,
                                "method": item.method, "seed": item.seed} for item in runs.values()],
            "comparisons": [
                {"name": comparison.name, "method": comparison.method,
                 "baseline": comparison.baseline, "dialect": comparison.dialect.value,
                 "metrics": list(comparison.metrics)} for comparison in comparisons],
            "runner": {"schema": "diffcam-refinement-check/1",
                       "policy": "baseline copied unchanged; refined RS274 is a named auxiliary method",
                       "source_commit": R.source_metadata(REPOSITORY)["source_commit"],
                       "physical_machining_certificate": False},
        })
    reports, audit_result = audit.audit_runs(auxiliary_root, manifest)
    if not audit_result["summary"]["auditor_passed"]:
        raise RunnerError(f"auxiliary refinement audit failed: {audit_result['summary']}")
    aggregate_rows, statistics = audit.aggregate_evaluations(
        manifest, audit_result["_audits_by_id"], bootstrap_replicates=100)
    document = {
        "manifest": audit_result["summary"], "records": reports,
        "reporting_policy": {
            "primary_files": "immutable; auxiliary root holds refinement artifacts and hash links",
            "efficiency_comparison": "disagreement cases are excluded; conservative outcomes remain visible",
            "disagreement_audit_namespace": "<condition>-refinement-disagreement, isolated from comparison aggregates",
            "quantity_comparison": "residual_volume_mm3, gouge_volume_mm3, max_surface_error_mm; cycle time is not silently changed",
        },
        "reported_outcomes": report_flags,
        "refinement_disagreement_by_condition": {
            condition: {"count": count, "refinement_disagreement": count > 0}
            for condition, count in disagreement_counts.items()
        },
        "comparisons": aggregate_rows, "statistics": statistics,
        "case_count": len(outcomes), "baseline_and_refined_records": len(reports),
        "audited_result_paths": [str(case["auxiliary_directory"]) for case in outcomes],
    }
    _atomic(auxiliary_root / "refinement-audit.json", _refinement_jsonable(document))
    return document


def _verified_refinement_for_report(root, run_id, auxiliary_root, policy):
    """Recompute reported outcomes from actual typed evidence and verify every link."""
    directory, _, _, task, primary, _ = _load_refinement_case(root, run_id)
    refined_run_id = f"{run_id}-refined"
    refined_directory = auxiliary_root / refined_run_id
    refined = _compare_existing_refinement(refined_directory, task,
                                           directory, (directory / "rs274.nc").read_bytes())
    refined = _verified_refinement_result(primary, refined, policy)
    payload = R.record_bytes(refined)
    refined_path = refined_directory / "eval_posted_rs274_refined.json"
    if refined_path.read_bytes() != payload:
        raise RunnerError("named refinement evaluation differs from its typed auxiliary record")
    comparison = _refinement_comparison(primary, refined)
    conservative, comparison = _conservative_outcome(primary, refined, comparison, task)
    comparison_document = {
        "primary_run_id": run_id, "refined_run_id": refined_run_id,
        "comparison": comparison, "conservative_outcome": conservative,
    }
    comparison_payload = R.json_bytes(comparison_document)
    if (refined_directory / "comparison.json").read_bytes() != comparison_payload:
        raise RunnerError("refinement comparison bytes differ from recomputed typed evidence")
    link_path = auxiliary_root / f"{run_id}-refinement-link.json"
    link = _json(link_path)
    if not isinstance(link, Mapping) or link.get("schema") != "diffcam-refinement-link/3":
        raise RunnerError(f"refinement link must use schema /3 for conservative completion: {run_id}")
    expected_link = {
        "schema": "diffcam-refinement-link/3",
        "primary_run_id": run_id,
        "primary_run_path": str(directory.relative_to(root)),
        "primary_metrics_sha256": C.sha256_bytes((directory / "metrics.json").read_bytes()),
        "primary_rs274_evaluation_sha256": C.sha256_bytes(
            (directory / "eval_posted_rs274.json").read_bytes()),
        "primary_haas_evaluation_sha256": C.sha256_bytes(
            (directory / "eval_posted_haas.json").read_bytes()),
        "primary_run_json_sha256": C.sha256_bytes((directory / "run.json").read_bytes()),
        "refined_run_id": refined_run_id,
        "refined_artifact_path": str(refined_path.relative_to(auxiliary_root)),
        "refined_evaluation_sha256": C.sha256_bytes(payload),
        "auxiliary_record": {
            "path": f"{refined_run_id}/run.json",
            "sha256": C.sha256_bytes((refined_directory / "run.json").read_bytes()),
        },
        "comparison_path": f"{refined_run_id}/comparison.json",
        "comparison_sha256": C.sha256_bytes(comparison_payload),
        "conservative_outcome": conservative,
        "comparison": comparison,
    }
    if link != expected_link:
        raise RunnerError(f"refinement link does not match actual evidence for {run_id}")
    return conservative, {
        "path": os.path.relpath(link_path, root),
        "sha256": C.sha256_bytes(link_path.read_bytes()),
    }, task


def _pilot_report(config, root, auxiliary_root, *, bootstrap_replicates=2000):
    """Produce the pilot table with conservative refinement outcomes, not primary-only ranks."""
    root = Path(root).resolve()
    try:
        manifest = audit.load_manifest(root / "manifest.json")
    except (OSError, ValueError) as error:
        raise RunnerError(f"cannot load primary pilot manifest: {error}") from error
    if not _refinement_policy(config)["enabled"]:
        report, _ = audit.generate_report(
            root, manifest, bootstrap_replicates=bootstrap_replicates)
        return report
    if auxiliary_root is None:
        raise RunnerError("enabled refinement requires an auxiliary report root")
    _, auxiliary_root = _safe_refinement_directory(root, Path(auxiliary_root))
    policy = _refinement_policy(config)
    report_path = auxiliary_root / "refinement-audit.json"
    auxiliary_report = _json(report_path)
    try:
        auxiliary_manifest = audit.load_manifest(auxiliary_root / "manifest.json")
    except (OSError, ValueError) as error:
        raise RunnerError(f"cannot load auxiliary refinement manifest: {error}") from error
    _, auxiliary_audit = audit.audit_runs(auxiliary_root, auxiliary_manifest)
    if not (auxiliary_audit["summary"]["auditor_passed"]
            and auxiliary_audit["summary"]["all_runs_completed"]):
        raise RunnerError("refinement auxiliary audit is incomplete or invalid")
    reported = auxiliary_report.get("reported_outcomes") if isinstance(auxiliary_report, Mapping) else None
    if not isinstance(reported, Mapping) or not reported:
        raise RunnerError("enabled refinement has no reported conservative outcomes")
    primary_ids = set(reported)
    required_aux_ids = primary_ids | {f"{run_id}-refined" for run_id in primary_ids}
    if (set(auxiliary_manifest.runs) != required_aux_ids
            or auxiliary_report.get("case_count") != len(primary_ids)
            or auxiliary_report.get("baseline_and_refined_records") != 2 * len(primary_ids)
            or auxiliary_report.get("manifest") != auxiliary_audit["summary"]):
        raise RunnerError("auxiliary refinement report does not reconcile with its actual audit")
    missing = sorted(set(manifest.runs) - primary_ids)
    if missing:
        raise RunnerError(f"enabled refinement lacks links/reported outcomes for primary runs: {missing}")
    outcomes = {}
    links = {}
    report_tasks = dict(manifest.tasks)
    completion_sources = {}
    for run_id in sorted(primary_ids):
        # Additional records in a shared auxiliary root are checked against the
        # immutable copied baselines, never used to substitute another primary root.
        source_root = root if run_id in manifest.runs else auxiliary_root
        expected = manifest.runs.get(run_id, auxiliary_manifest.runs[run_id])
        conservative, link, task = _verified_refinement_for_report(
            source_root, run_id, auxiliary_root, policy)
        condition = expected.method.removesuffix("-refinement-disagreement")
        expected_reported = {
            "condition": condition,
            "seed": expected.seed,
            "safety_pass": conservative["safety_pass"],
            "geometry_pass": conservative["geometry_pass"],
            "completion_pass": conservative["completion_pass"],
            "efficiency_eligible": conservative["efficiency_eligible"],
            "efficiency_excluded": conservative["excluded_from_efficiency"],
            "refinement_disagreement": conservative["refinement_disagreement"],
            "disagreement_fields": conservative["disagreement_fields"],
            "refined_max_step_mm": policy["max_step_mm"],
            "refined_evaluation_path": f"{run_id}-refined/eval_posted_rs274_refined.json",
            "auxiliary_record_path": f"{run_id}-refined/run.json",
            "auxiliary_link_path": f"{run_id}-refinement-link.json",
        }
        if reported[run_id] != expected_reported:
            raise RunnerError(f"auxiliary reported outcome is not backed by typed evidence: {run_id}")
        if run_id not in manifest.runs:
            continue
        task_info = report_tasks[expected.task_id]
        if task_info.task_hash != task.task_hash:
            raise RunnerError(f"primary/refinement task identity mismatch: {run_id}")
        if task_info.completion_residual_mm3 is None:
            task_info = dataclasses.replace(
                task_info,
                completion_residual_mm3=task.tolerances.completion_residual_mm3)
            completion_sources[expected.task_id] = "TaskSpec.tolerances.completion_residual_mm3"
            report_tasks[expected.task_id] = task_info
        else:
            completion_sources[expected.task_id] = "primary manifest"
        outcomes[run_id] = audit.ConservativeOutcome(
            safety_pass=conservative["safety_pass"],
            geometry_pass=conservative["geometry_pass"],
            completion_pass=conservative["completion_pass"],
            efficiency_eligible=conservative["efficiency_eligible"],
            refinement_disagreement=conservative["refinement_disagreement"],
            disagreement_fields=tuple(conservative["disagreement_fields"]),
        )
        links[run_id] = link
    if set(outcomes) != set(manifest.runs):
        raise RunnerError("refinement outcomes did not cover every expected primary run")
    report_manifest = dataclasses.replace(manifest, tasks=report_tasks)
    report, _ = audit.generate_report(
        root, report_manifest, bootstrap_replicates=bootstrap_replicates,
        reported_outcomes=outcomes)
    report["refinement"]["auxiliary_audit_sha256"] = C.sha256_bytes(report_path.read_bytes())
    report["refinement"]["links_by_run_id"] = links
    report["refinement"]["completion_thresholds_mm3"] = {
        task_id: task_info.completion_residual_mm3
        for task_id, task_info in report_tasks.items()
    }
    report["refinement"]["completion_threshold_sources"] = completion_sources
    return report


def _refine_process(config_path, source_root, run_id, auxiliary_root, timeout):
    """Own exactly one exact-PID refinement worker; no process-pattern kills."""
    auxiliary_log = Path(auxiliary_root) / "logs"
    auxiliary_log.mkdir(parents=True, exist_ok=True)
    log_path = auxiliary_log / f"{run_id}.{time.time_ns()}.log"
    env = dict(os.environ)
    env["DIFFCAM_RUNNER_SIMULATOR"] = "0"
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
               "--output-dir", str(source_root), "--split", "dev", "--experiment", "pilot",
               "--only", run_id, "--aux-root", str(auxiliary_root), "--refinement-worker-run"]
    started = time.perf_counter()
    with log_path.open("x") as stream:
        process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
    if code:
        return {"run_id": run_id, "source_root": source_root, "status": "failed",
                "returncode": code, "log": str(log_path),
                "elapsed_seconds": time.perf_counter() - started}
    text = log_path.read_text(encoding="utf-8")
    for line in reversed(text.splitlines()):
        if line.startswith("REFINEMENT_RESULT="):
            result = json.loads(line[len("REFINEMENT_RESULT="):])
            result.update({"elapsed_seconds": time.perf_counter() - started, "log": str(log_path)})
            return result
    return {"run_id": run_id, "source_root": source_root, "status": "failed",
            "reason": "refinement worker returned no typed result", "returncode": code,
            "log": str(log_path), "elapsed_seconds": time.perf_counter() - started}


def _refinement_driver(config, config_path, primary_roots, only, auxiliary_root):
    """Run independent post-evaluation subprocesses after selected primary work."""
    policy = _refinement_policy(config)
    if not policy["enabled"]:
        raise RunnerError("refinement checking was explicitly disabled by the frozen policy")
    work = []
    ready_ids = set()
    for raw_root in primary_roots:
        root = Path(raw_root).resolve()
        if not root.is_dir():
            raise RunnerError(f"primary refinement source directory does not exist: {root}")
        manifest = audit.load_manifest(root / "manifest.json")
        for entry in manifest.runs.values():
            if only and not fnmatch.fnmatch(entry.run_id, only):
                continue
            if entry.run_id in ready_ids:
                raise RunnerError(f"duplicate primary run ID across refinement roots: {entry.run_id}")
            ready_ids.add(entry.run_id)
            child = root / entry.run_id
            if not child.is_dir():
                work.append({"root": root, "run_id": entry.run_id, "status": "missing_primary",
                             "reason": "no run directory for final selected program"})
                continue
            record = audit._audit_run(child / "run.json")
            if not record.complete_and_audited:
                work.append({"root": root, "run_id": entry.run_id, "status": "skipped_incomplete",
                             "reason": "primary record is not complete and cleanly audited"})
                continue
            work.append({"root": root, "run_id": entry.run_id, "manifest": manifest,
                         "auxiliary_root": Path(auxiliary_root).resolve(), "status": "scheduled"})
    if not work:
        raise RunnerError("refinement selection matched no expected primary runs")
    executable = [item for item in work if item["status"] == "scheduled"]
    if not executable:
        raise RunnerError("no completed primary programs are available for refinement")
    for item in executable:
        _safe_refinement_directory(item["root"], item["auxiliary_root"])
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=policy["parallel_jobs"]) as pool:
        pending = {
            pool.submit(_refine_process, config_path, item["root"], item["run_id"],
                        item["auxiliary_root"], policy["timeout_s"]): item
            for item in executable
        }
        for future in concurrent.futures.as_completed(pending):
            completed.append(future.result())
    scheduled = [item for item in completed if item["status"] in ("complete", "skipped_complete")]
    aux_root = Path(auxiliary_root).resolve()
    if scheduled:
        for item in scheduled:
            planned = next(w for w in executable if w["run_id"] == item["run_id"])
            if str(item["root"]) != str(planned["root"]):
                raise RunnerError("refinement child returned a different primary source root")
            if str(item["auxiliary_root"]) != str(planned["auxiliary_root"]):
                raise RunnerError("refinement children must use one declared auxiliary root")
            item["manifest"] = planned["manifest"]
        for item in scheduled:
            if Path(item["auxiliary_directory"]).resolve() != aux_root / item["refined_run_id"]:
                raise RunnerError("refinement result points outside the declared audit root")
            for key in ("root", "auxiliary_root", "auxiliary_directory", "auxiliary_link_path"):
                item[key] = Path(item[key])
        audited = _audit_refinement_root(scheduled, aux_root)
    else:
        audited = None
    remaining = [item for item in work if item["status"] != "scheduled"]
    reporting_items = []
    for item in [*completed, *remaining]:
        row = {key: value for key, value in item.items() if key != "manifest"}
        for key in ("root", "auxiliary_root", "auxiliary_directory"):
            if key in row:
                row[key] = str(row[key])
        reporting_items.append(_refinement_jsonable(row))
    output = {
        "schema": "diffcam-refinement-check/1", "policy": policy,
        "required_primary_programs": len(work), "scheduled_worker_count": len(executable),
        "status_counts": dict(Counter(item["status"] for item in [*completed, *remaining])),
        "results": reporting_items, "audit": audited,
        "auxiliary_root": str(aux_root),
        "process_policy": "independent subprocesses, at most two; exact child PID timeout handling",
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TAICHI_NUM_THREADS")
        },
    }
    _atomic(aux_root / "refinement_report.json", output)
    failed = [item for item in completed if item["status"] == "failed"]
    if failed:
        raise RunnerError(f"refinement workers failed: {[item['run_id'] for item in failed]}")
    return output


def _default_auxiliary_root(root: Path) -> Path:
    return root.resolve().parent / f"{root.resolve().name}-refinement"


def _refinement_check_main(config, args, root: Path) -> int:
    """Read-only primary selection with real subprocess evaluations, never a solver launch."""
    roots = [root]
    if args.split != "dev" or config["experiment"] != "pilot":
        raise RunnerError("refinement is implemented only for the declared dev pilot")
    if args.primary_root:
        roots = [Path(item).resolve() for item in args.primary_root]
    auxiliary_root = Path(args.aux_root).resolve() if args.aux_root else _default_auxiliary_root(roots[0])
    result = _refinement_driver(config, args.config, roots, args.only, auxiliary_root)
    report = _pilot_report(config, roots[0], auxiliary_root)
    report_path = roots[0] / "pilot_report.json"
    _atomic(report_path, report)
    result["pilot_report"] = {"path": report_path.name,
                              "sha256": C.sha256_bytes(R.json_bytes(report))}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _refinement_worker_main(config, root: Path, args) -> int:
    if args.split != "dev" or config["experiment"] != "pilot":
        raise RunnerError("refinement worker is implemented only for the declared dev pilot")
    if not args.only or args.only != args.only.strip() or "*" in args.only:
        raise RunnerError("refinement worker requires one exact primary run id")
    if not args.aux_root:
        raise RunnerError("refinement worker requires an explicit auxiliary root")
    result = _refinement_worker(config, root, args.only, args.aux_root)
    serialized = _refinement_jsonable({key: value for key, value in result.items()
                                       if key not in ("primary_config", "source_config")})
    print("REFINEMENT_RESULT=" + json.dumps(serialized, sort_keys=True))
    return 0


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


def _timing_worker_main(config, root):
    if config.get("task_source", "pilot_pocket") != "pilot_pocket":
        raise RunnerError("timing-worker-run is only available for the registered pilot_pocket task")
    path = root / "budget-timings.json"
    if path.exists():
        raise RunnerError(f"existing timing report will not be overwritten: {path}")
    root.mkdir(parents=True, exist_ok=True)
    report = _measure_budget(config, root, capture_dir=root / "timing-example")
    _atomic(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_parent(config: Mapping[str, Any], args: argparse.Namespace, root: Path) -> int:
    _refinement_policy(config)
    task, motion, parameterization, _, posted_factory = _context(config, root)
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
        timing = config.get("timing_evidence")
        if config.get("task_source", "pilot_pocket") == "pilot_pocket":
            timing_path = root / "budget-timings.json"
            cached = {}
            if args.resume and timing_path.is_file() and config.get("measure_budget", True) is not False:
                cached = _json(timing_path)
                if not isinstance(cached, dict) or cached.get("task_hash") != task.task_hash:
                    raise RunnerError("resumed timing report does not describe the frozen pilot task")
            if cached:
                timing = dict(cached)
            else:
                timing = _measure_timing_subprocess(config, root)
            if not isinstance(timing, Mapping):
                raise RunnerError("real pilot dry-run needs actual timing evidence")
            if timing.get("task_hash") != task.task_hash:
                raise RunnerError("timing evidence task identity differs from the actual pilot task")
        costs = _pecent(dict(config, timing_evidence=timing) if timing is not None else config, entries)
        refinement = _refinement_projection(config, entries, timing, root)
        costs_with_total = {"planned_run_count": len(entries), "costs": costs,
                            "estimated_total_wall_time_s": (
                                sum(row["estimated_run_wall_time_s"] for row in costs)
                                if all(row["estimated_run_wall_time_s"] is not None for row in costs) else None),
                            "timing_evidence": timing,
                            "serialization": {"subprocess_per_run": True, "max_simulators_per_process": 1,
                                              "global_simulator_cap": 12, "gpu_jobs": "serial",
                                              "refinement_subprocesses_per_program": 1,
                                              "refinement_parallel_jobs": refinement["parallel_jobs"] if refinement["enabled"] else 0,
                                              "pilot_launched": False}}
        if refinement["enabled"]:
            costs_with_total["refinement_check"] = refinement
        if costs_with_total["estimated_total_wall_time_s"] is not None:
            costs_with_total["estimated_total_wall_time_including_refinement_s"] = (
                costs_with_total["estimated_total_wall_time_s"]
                + refinement["estimated_phase_wall_time_s"])
        else:
            costs_with_total["estimated_total_wall_time_including_refinement_s"] = None
        if config.get("task_source", "pilot_pocket") == "pilot_pocket":
            elapsed = costs_with_total["estimated_total_wall_time_including_refinement_s"]
            if elapsed is not None and elapsed > 7200.0:
                costs_with_total["budget_review"] = {
                    "recommendation": "do not launch before a coordinator review of budget/grid choices and their fidelity costs",
                    "current_execution_policy": [
                        "one production evaluation per dialect only on each run's final selected motion, not per optimizer validation",
                        "strict 0.25mm production geometry, sampling and posted-evaluation fidelity unchanged",
                        "primary runs remain serial with one simulator per child; at most two independent refinement subprocesses after primary work, with GPU unselected",
                    ],
                    "options_to_discuss_not_applied": [
                        "reduce or stage sampling budgets with independently shown fidelity changes",
                        "profile a coarse analytic grid first, without relaxing the declared 0.25mm production target",
                        "profile/vectorise production-evaluation hot spots through the file owner, preserving semantics and independent oracles",
                        "coordinator-approved bounded CPU subprocess parallelism with measured memory/contention effects and unchanged per-program evaluation settings",
                    ]}
        _atomic(root / "dry_run.json", costs_with_total)
        print(json.dumps(costs_with_total, indent=2, sort_keys=True))
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
    summary = {"runs": records,
               "status_counts": dict(Counter(row["reported_status"] for row in records)),
               "pending_evaluation": PENDING_REASON if posted_factory is None else None}
    _atomic(root / "execution_summary.json", summary)
    if _refinement_policy(config)["enabled"]:
        # Only after the selected primary records exist. These are subprocess
        # evaluations of the exact selected RS274 bytes, never solver launches.
        refinement = _refinement_driver(
            config, args.config, [root], args.only,
            Path(args.aux_root).resolve() if args.aux_root else _default_auxiliary_root(root))
        summary["refinement_check"] = refinement
        report = _pilot_report(
            config, root,
            Path(args.aux_root).resolve() if args.aux_root else _default_auxiliary_root(root))
        _atomic(root / "pilot_report.json", report)
        summary["pilot_report"] = {
            "path": "pilot_report.json", "sha256": C.sha256_bytes(R.json_bytes(report))}
        _atomic(root / "execution_summary.json", summary)
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
    parser.add_argument("--timing-worker-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--refinement-check", action="store_true",
                        help="evaluate final primary RS274 programs at 0.125 mm")
    parser.add_argument("--refinement-worker-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--aux-root", help="sibling audit root for refinement evidence")
    parser.add_argument("--primary-root", action="append", help="additional completed primary root")
    args = parser.parse_args(argv)
    root = Path(args.output_dir).resolve()
    if (not args.resume and not args.worker_run and not args.timing_worker_run
            and not args.refinement_check and not args.refinement_worker_run and root.exists()):
        parser.error("output-dir must be new unless --resume is set")
    try:
        config = _normalized_config(args.config)
        config["experiment"] = args.experiment
        if config["experiment"] not in EXPERIMENTS:
            raise RunnerError("invalid advertised experiment")
        if args.refinement_check and (args.worker_run or args.timing_worker_run):
            raise RunnerError("refinement checking cannot be combined with solver/timing worker modes")
        if args.refinement_worker_run and (args.worker_run or args.timing_worker_run):
            raise RunnerError("refinement worker cannot be combined with solver/timing worker modes")
        if args.dry_run and args.refinement_check:
            raise RunnerError("--dry-run estimates refinement cost without dispatching that phase")
        if args.primary_root and not args.refinement_check:
            raise RunnerError("--primary-root requires --refinement-check")
        if args.refinement_worker_run and not root.is_dir():
            raise RunnerError("refinement worker requires an existing primary root")
        if args.refinement_worker_run:
            return _refinement_worker_main(config, root, args)
        if args.refinement_check:
            return _refinement_check_main(config, args, root)
        if args.timing_worker_run:
            if not args.dry_run or args.worker_run:
                raise RunnerError("timing-worker-run requires --dry-run and cannot combine with --worker-run")
            return _timing_worker_main(config, root)
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
