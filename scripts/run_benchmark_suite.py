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
from types import MappingProxyType, SimpleNamespace
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
from cam.posted_evaluator import PostedEvaluator, RefinementSettings, _analytic_mask, _revision, EVALUATOR_ID
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
    # The actual solver result is needed for θ0/selection evidence in artifacts.
    # Nominal and older synthetic Outcome construction have no such solver state.
    optimizer_result: Optional[Any] = None


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


def _requested_precision(config, *, pilot=False):
    precision = config.get("precision")
    if precision not in ("f32", "f64"):
        raise RunnerError("config.precision must explicitly declare 'f32' or 'f64'")
    if pilot and precision != "f64":
        raise RunnerError("the comparison pilot requires precision='f64'; f32 is exploration-only")
    return precision


def _normalized_config(path: str | Path) -> dict[str, Any]:
    raw = _json(Path(path))
    if not isinstance(raw, dict) or raw.get("schema") != RUNNER_SCHEMA:
        raise RunnerError(f"config schema must be {RUNNER_SCHEMA}")
    _requested_precision(raw, pilot=raw.get("task_source", "pilot_pocket") == "pilot_pocket")
    _configured_typed_sources(raw)
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
                         simulator=sim, spec=O.ObjectiveSpec(k=float(config["k_smooth"])), seed=seed,
        requested_precision=config["precision"])
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
    precision = _requested_precision(config, pilot=True)
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f32 if precision == "f32" else ti.f64)
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
        init_taichi=False, k_init=float(config["k_smooth"]), precision=precision,
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
    """Fidelity roles and D-04/D-07 posted-worker resource limits."""
    if config.get("task_source", "pilot_pocket") != "pilot_pocket":
        return {"enabled": False}
    fields = {
        "enabled", "verdict_step_mm", "screen_step_mm", "convergence_step_mm",
        "parallel_jobs", "max_rss_bytes_per_job", "poll_interval_s", "timeout_s",
    }
    policy = config.get("refinement_check")
    if not isinstance(policy, Mapping) or set(policy) != fields:
        raise RunnerError(f"refinement_check must declare exactly {sorted(fields)}")
    if not isinstance(policy["enabled"], bool):
        raise RunnerError("refinement_check.enabled must be boolean")
    for name, required in (("verdict_step_mm", 0.125), ("screen_step_mm", 1.0),
                           ("convergence_step_mm", 0.25)):
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != required:
            raise RunnerError(f"refinement_check.{name} must be the declared {required} mm")
    jobs = _positive_int(policy["parallel_jobs"], "refinement_check.parallel_jobs")
    if jobs > 2:
        raise RunnerError("posted evaluation parallel_jobs must be bounded at 2")
    limit = _positive_int(policy["max_rss_bytes_per_job"], "refinement_check.max_rss_bytes_per_job")
    if limit > 8 * 1024 ** 3:
        raise RunnerError("posted evaluation memory ceiling may not exceed the D-07 approved 8 GiB per job")
    for name in ("poll_interval_s", "timeout_s"):
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RunnerError(f"refinement_check.{name} must be finite and positive")
    if policy["poll_interval_s"] > 30.0:
        raise RunnerError("refinement_check.poll_interval_s must not exceed 30 seconds")
    return dict(policy, max_step_mm=policy["verdict_step_mm"])


def _phase_settings(task, config, dialect, nc_bytes, role, *, primary=None, controls=None):
    """Freeze production controls, changing only the declared role's step."""
    policy = _refinement_policy(config)
    if not policy["enabled"]:
        raise RunnerError("posted job phases require the declared real-pilot fidelity policy")
    steps = {"screen": policy["screen_step_mm"], "verdict": policy["verdict_step_mm"],
             "convergence": policy["convergence_step_mm"]}
    if role not in steps:
        raise RunnerError(f"unknown posted-evaluation role: {role}")
    if controls is not None and primary is not None:
        raise RunnerError("posted job controls and primary evaluation are mutually exclusive")
    if controls is not None:
        settings = dict(controls)
    else:
        settings = dict(primary.discretization["settings"] if primary is not None
                        else _pilot_refinement_settings(task, config))
    if settings.get("compute_bounds") is not True or settings.get("safety_screen") is not True:
        raise RunnerError("posted jobs may not disable production bounds or safety screening")
    nc_hash = C.nc_hash(nc_bytes)
    if settings.get("expected_nc_hash") not in (None, nc_hash):
        raise RunnerError("posted job source has a different exact-NC hash")
    settings.update(max_step_mm=steps[role], expected_nc_hash=nc_hash)
    try:
        return dataclasses.asdict(RefinementSettings.from_mapping(settings))
    except ValueError as error:
        raise RunnerError(f"invalid {role} posted settings: {error}") from error


def _validate_posted_result(task, nc_bytes, dialect, settings, result,
                            *, require_current_evaluator=False):
    if not isinstance(result, C.EvaluationResult):
        raise RunnerError("posted worker did not return a typed EvaluationResult")
    if require_current_evaluator:
        expected = f"{EVALUATOR_ID}/{_revision()}"
        actual = f"{result.evaluator_id}/{result.evaluator_revision}"
        if actual != expected:
            raise RunnerError(
                f"reused result evaluator identity/revision mismatch: current {expected}, reused {actual}")
    if result.dialect is not dialect or result.nc_hash != C.nc_hash(nc_bytes):
        raise RunnerError("posted result has the wrong dialect or exact NC hash")
    if result.task_hash != task.task_hash or result.tool_setup_hash != task.tool_setup_hash:
        raise RunnerError("posted result has a different task/tool identity")
    if result.discretization.get("settings") != dict(settings):
        raise RunnerError("posted result settings differ from the frozen immutable job")
    return result


@dataclass(frozen=True)
class PostedJob:
    role: str
    dialect: C.Dialect
    nc_bytes: bytes
    settings: Mapping[str, Any]
    workspace: Path
    reused_source: Optional[Path] = None
    # Infer only at construction. Later supplied unit-test override files are
    # not historical cache and keep the initial False setting. Declared cached
    # sources, including direct PostedJob(path) uses, require current identity.
    require_current_evaluator: Optional[bool] = None

    def __post_init__(self):
        if self.require_current_evaluator is None:
            object.__setattr__(self, "require_current_evaluator", self.reused_source is not None)


def _prepare_posted_job(task, job):
    nc_name, _ = C.DIALECT_ARTIFACTS[job.dialect]
    settings_bytes = R.json_bytes({
        "schema": "diffcam-post-job/1", "role": job.role,
        "dialect": job.dialect.value, "task": task.to_dict(), "task_hash": task.task_hash,
        "nc_hash": C.nc_hash(job.nc_bytes), "settings": dict(job.settings),
    })
    settings_path = job.workspace / "job.json"
    if settings_path.exists() and settings_path.read_bytes() != settings_bytes:
        raise RunnerError(f"posted job identity differs; refusing to mix files in {job.workspace}")
    nc_path = job.workspace / nc_name
    if nc_path.exists() and nc_path.read_bytes() != job.nc_bytes:
        raise RunnerError(f"posted job NC bytes differ; refusing to reuse {nc_path}")
    R.ArtifactWriter._replace_bytes(settings_path, settings_bytes)
    R.ArtifactWriter._replace_bytes(nc_path, job.nc_bytes)


def _posted_job_main(workspace: Path):
    """A single posted evaluation in an independent, memory-monitored process."""
    import resource

    spec = json.loads((workspace / "job.json").read_text(encoding="utf-8"))
    task = C.TaskSpec.from_dict(spec["task"])
    dialect = C.Dialect(spec["dialect"])
    nc_path = workspace / C.DIALECT_ARTIFACTS[dialect][0]
    nc_bytes = nc_path.read_bytes()
    if task.task_hash != spec["task_hash"] or C.nc_hash(nc_bytes) != spec["nc_hash"]:
        raise RunnerError("posted worker task/NC identity differs from its job record")
    settings = spec["settings"]
    if settings.get("expected_nc_hash") != spec["nc_hash"]:
        raise RunnerError("posted worker's expected NC hash is not bound to exact bytes")
    started = time.perf_counter()
    result = PostedEvaluator().evaluate_posted_gcode(task, nc_bytes, dialect, settings)
    elapsed = time.perf_counter() - started
    job = PostedJob(spec["role"], dialect, nc_bytes, MappingProxyType(settings), workspace)
    result = _validate_posted_result(task, nc_bytes, dialect, settings, result)
    evaluation_path = workspace / "evaluation.json"
    payload = R.record_bytes(result)
    if evaluation_path.exists() and evaluation_path.read_bytes() != payload:
        raise RunnerError("posted evaluation already exists with different bytes")
    if not evaluation_path.exists():
        R.ArtifactWriter._replace_bytes(evaluation_path, payload)
    record = {
        "schema": "diffcam-post-job-result/1", "status": "complete",
        "role": job.role, "dialect": dialect.value, "task_hash": task.task_hash,
        "nc_hash": C.nc_hash(nc_bytes), "settings": settings,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": C.sha256_bytes(payload),
        "wall_seconds": elapsed, "worker_peak_rss_kib": _self_peak_rss_kib(),
        "worker_peak_rss_source": "VmHWM of /proc/self/status (own address space; resets at exec)",
        # Diagnostic only: Linux ru_maxrss is inherited across fork and kept across
        # execve, so it reports max(parent peak at spawn, own peak). Never a cap input.
        "diagnostic_inherited_ru_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "worker_pid": os.getpid(),
        "thread_environment": {key: os.environ.get(key)
                               for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TAICHI_NUM_THREADS")},
    }
    R.ArtifactWriter._replace_bytes(workspace / "worker-result.json", R.json_bytes(record))
    print("POST_JOB_RESULT=" + json.dumps(record, sort_keys=True))


def _self_peak_rss_kib():
    """This process's own peak RSS (VmHWM), excluding any peak inherited from a parent.

    getrusage(RUSAGE_SELF).ru_maxrss is unsuitable for a spawned worker: Linux carries it
    across fork and execve, so a child of a large optimizer reports the optimizer's peak.
    """
    if not sys.platform.startswith("linux"):
        raise RunnerError("posted-job peak-RSS measurement is currently Linux-only")
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    raise RunnerError("/proc/self/status has no VmHWM line")


def _process_rss_bytes(pid):
    if not sys.platform.startswith("linux"):
        raise RunnerError("posted-job peak-RSS monitoring is currently Linux-only")
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return None
    return None


def _posted_job_process(task, job, config):
    if job.reused_source is not None:
        source = job.reused_source.read_bytes()
        result = C.loads(source.decode("utf-8"))
        _validate_posted_result(task, job.nc_bytes, job.dialect, job.settings, result,
                                require_current_evaluator=job.require_current_evaluator)
        if R.record_bytes(result) != source:
            raise RunnerError("cached typed refinement is not canonical production evidence")
        return result, {
            "status": "reused_typed_evaluation", "role": job.role,
            "dialect": job.dialect.value, "evaluation_sha256": C.sha256_bytes(source),
            "source": str(job.reused_source), "worker_peak_rss_kib": None,
            "wall_seconds": None, "worker_pid": None,
            "timings_note": "reused existing typed result; no new evaluation or timing claim",
        }
    if job.workspace.exists():
        raise RunnerError(f"posted job directory must be new: {job.workspace}")
    job.workspace.mkdir(parents=True)
    env = dict(os.environ, DIFFCAM_RUNNER_SIMULATOR="0")
    _prepare_posted_job(task, job)
    evaluation_path = job.workspace / "evaluation.json"
    record_path = job.workspace / "worker-result.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(job.workspace / "job.json"),
               "--output-dir", str(job.workspace), "--split", "dev", "--experiment", "pilot",
               "--posted-job-dir", str(job.workspace)]
    policy = _refinement_policy(config)
    started = time.perf_counter()
    log_path = job.workspace / f"worker.{time.time_ns()}.log"
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdout=stream, stderr=subprocess.STDOUT)
        timed_out_or_memory = None
        observed_child_hwm = None
        try:
            deadline = started + policy["timeout_s"]
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    timed_out_or_memory = "timeout"
                    break
                try:
                    code = process.wait(timeout=min(policy["poll_interval_s"], remaining))
                    break
                except subprocess.TimeoutExpired:
                    observed = _process_rss_bytes(process.pid)
                    if observed is not None:
                        observed_child_hwm = max(observed_child_hwm or 0, observed)
                    if observed is not None and observed > policy["max_rss_bytes_per_job"]:
                        timed_out_or_memory = "memory_limit"
                        break
        except BaseException:
            timed_out_or_memory = timed_out_or_memory or "interrupted"
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        code = process.wait()
        if code:
            raise RunnerError(f"posted worker failed ({timed_out_or_memory or 'exit'}): {log_path}")
        if timed_out_or_memory is not None:
            raise RunnerError(f"posted worker was stopped for {timed_out_or_memory}: {log_path}")
    text = log_path.read_text(encoding="utf-8")
    results = [line.removeprefix("POST_JOB_RESULT=") for line in text.splitlines()
               if line.startswith("POST_JOB_RESULT=")]
    if len(results) != 1 or not record_path.is_file() or not evaluation_path.is_file():
        raise RunnerError(f"posted worker produced no unique typed result: {log_path}")
    record = json.loads(results[-1])
    if record != json.loads(record_path.read_text()):
        raise RunnerError("posted worker result and file metadata disagree")
    if record.get("worker_pid") != process.pid:
        raise RunnerError("posted worker returned a different child PID")
    if record["role"] != job.role or record["dialect"] != job.dialect.value:
        raise RunnerError("posted worker returned a different role/dialect")
    if record["nc_hash"] != C.nc_hash(job.nc_bytes) or record["task_hash"] != task.task_hash:
        raise RunnerError("posted worker returned mismatched identities")
    if record["settings"] != dict(job.settings):
        raise RunnerError("posted worker returned different sampling settings")
    self_reported_kib = _positive_int(record.get("worker_peak_rss_kib"), "posted peak RSS KiB")
    polled_kib = None if observed_child_hwm is None else observed_child_hwm // 1024
    # The worker samples its own VmHWM when it writes its record; serialisation and
    # interpreter teardown can raise it slightly afterwards, which the parent's poll
    # may see. The recorded peak is the larger of the two, and the cap applies to it.
    record["worker_self_reported_peak_rss_kib"] = self_reported_kib
    record["parent_polled_child_vmhwm_kib"] = polled_kib
    record["worker_peak_rss_kib"] = max(self_reported_kib, polled_kib or 0)
    peak = record["worker_peak_rss_kib"] * 1024
    if peak > policy["max_rss_bytes_per_job"]:
        raise RunnerError(f"posted worker exceeded its memory ceiling: {peak} bytes")
    record["memory_policy_result"] = "within_declared_peak_rss_ceiling"
    record["elapsed_seconds"] = time.perf_counter() - started
    record["process_wall_seconds"] = record["elapsed_seconds"]
    record["evaluation_path"] = str(evaluation_path)
    if record["evaluation_sha256"] != C.sha256_bytes(evaluation_path.read_bytes()):
        raise RunnerError("posted worker evaluation hash differs from the retained typed file")
    record["evaluation_sha256"] = C.sha256_bytes(evaluation_path.read_bytes())
    record["log"] = str(log_path)
    return C.loads(evaluation_path.read_text(encoding="utf-8")), record


def _post_jobs(task, jobs, config):
    """Execute an ordered set of independent posted child processes."""
    policy = _refinement_policy(config)
    workspace_roots = [job.workspace.resolve() for job in jobs]
    if len(set(workspace_roots)) != len(jobs):
        raise RunnerError("posted job directories must be unique per role and dialect")
    started = time.perf_counter()
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=policy["parallel_jobs"]) as pool:
        pending = {pool.submit(_posted_job_process, task, job, config): index
                   for index, job in enumerate(jobs)}
        for future in concurrent.futures.as_completed(pending):
            index = pending[future]
            results[index] = future.result()
    ordered = [results[index] for index in range(len(jobs))]
    return {
        "job_policy": dict(policy,
                         total_posted_rss_ceiling_bytes=policy["parallel_jobs"] * policy["max_rss_bytes_per_job"]),
        "phase_wall_seconds": time.perf_counter() - started,
        "processes": [
            {
                "role": job.role,
                "dialect": job.dialect.value,
                "evaluator_wall_seconds": metadata.get("wall_seconds"),
                "process_wall_seconds": metadata.get("elapsed_seconds"),
                "worker_pid": metadata.get("worker_pid"),
                "worker_peak_rss_kib": metadata.get("worker_peak_rss_kib"),
                "worker_peak_rss_source": metadata.get("worker_peak_rss_source"),
                "parent_polled_child_vmhwm_kib": metadata.get("parent_polled_child_vmhwm_kib"),
                "worker_self_reported_peak_rss_kib": metadata.get("worker_self_reported_peak_rss_kib"),
                "diagnostic_inherited_ru_maxrss_kib": metadata.get("diagnostic_inherited_ru_maxrss_kib"),
                "status": metadata["status"],
                "log": metadata.get("log"),
                "thread_environment": metadata.get("thread_environment"),
                "max_step_mm": job.settings["max_step_mm"],
                "expected_nc_hash": job.settings["expected_nc_hash"],
                "source": metadata.get("source"),
                "evaluation_path": metadata.get("evaluation_path"),
                "evaluation_sha256": metadata["evaluation_sha256"],
                "memory_note": "per-child process peak RSS; shared parent and all other system work excluded",
            }
            for (_, metadata), job in zip(ordered, jobs)
        ],
        "results": ordered,
    }


class PilotPostedEvaluator(PostedEvaluator):
    """Real production posted evaluator with unchanged, bounded refinement."""

    def __init__(self, task, config):
        super().__init__()
        self.task_hash = task.task_hash
        self._config = dict(config)
        self.settings = MappingProxyType(_pilot_refinement_settings(task, config))
        self.phase_records = []
        self.timings = []

    def _jobs_for_posting(self, task, posting, workspace, role, *, single_dialect=False):
        dialects = [post.dialect for post in posting.posted]
        if len(dialects) != len(set(dialects)):
            raise RunnerError("posted batch has duplicate dialects")
        if role == "convergence" and dialects != [C.Dialect.RS274]:
            raise RunnerError("the 0.25 mm convergence point requires exactly one RS274 program")
        if role in ("screen", "verdict") and not single_dialect and set(dialects) != set(C.Dialect):
            raise RunnerError("screen and verdict phases require both RS274 and Haas")
        jobs = []
        for post in posting.posted:
            if post.status != C.PostStatus.SUPPORTED:
                raise RunnerError(f"posted evaluation is unsupported for {post.dialect.value}: {post.reason}")
            nc_name = C.DIALECT_ARTIFACTS[post.dialect][0]
            nc_bytes = posting.files[nc_name]
            if post.nc_hash != C.nc_hash(nc_bytes):
                raise RunnerError("posting's declared hash differs from its actual NC bytes")
            settings = _phase_settings(task, self._config, post.dialect, nc_bytes, role,
                                       controls=self.settings)
            jobs.append(PostedJob(role, post.dialect, nc_bytes, MappingProxyType(settings),
                                  workspace / post.dialect.value))
        return jobs

    def evaluate_posting(self, task, posting, workspace, *, role="screen", single_dialect=False):
        workspace = Path(workspace).resolve() / f"{role}-step-{time.time_ns()}"
        batch = _post_jobs(task, self._jobs_for_posting(
            task, posting, workspace, role, single_dialect=single_dialect), self._config)
        self.phase_records.append({key: value for key, value in batch.items() if key != "results"})
        self.timings = []
        results = {}
        for dialect, (evaluation, metadata) in zip((post.dialect for post in posting.posted), batch["results"]):
            results[dialect] = evaluation
            self.timings.append({"dialect": dialect.value, "nc_hash": evaluation.nc_hash,
                                 "wall_seconds": metadata.get("wall_seconds"),
                                 "worker_peak_rss_kib": metadata.get("worker_peak_rss_kib"),
                                 "worker_pid": metadata.get("worker_pid")})
        return results

    def evaluate_posted_gcode(self, task, nc_bytes, dialect, settings):
        if task.task_hash != self.task_hash:
            raise RunnerError("pilot posted evaluator received a different task")
        raw = {} if settings is None else dict(settings)
        if raw:
            raise RunnerError("pilot posted settings are fixed by the declared task; per-call overrides are unsupported")
        dialect = C.Dialect(dialect)
        posted = C.PostedProgram(dialect, C.PostStatus.SUPPORTED, C.nc_hash(nc_bytes))
        nc_name = C.DIALECT_ARTIFACTS[dialect][0]
        namespace = (REPOSITORY / "external" / "pilot-posted-jobs" / self.task_hash[7:19]
                     / posted.nc_hash[7:19] / "screen")
        return self.evaluate_posting(
            task, R.PostingResult(files={nc_name: nc_bytes}, posted=(posted,)),
            namespace, single_dialect=True)[dialect]



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
    precision = _requested_precision(config)
    ti.init(arch=ti.cpu if arch == "cpu" else ti.gpu, offline_cache=False,
            default_fp=ti.f32 if precision == "f32" else ti.f64)
    initial = np.asarray(parameterization.initial_position_mm)
    normalized = (initial - task.grid.origin_mm) / task.grid.extent_mm
    sim = CSGSimulatorDelta(resolution=max(task.grid.shape), max_steps=parameterization.num_steps,
                            stock_size_mm=task.grid.extent_mm, voxel_size_mm=task.grid.voxel_mm,
                            target_shape="sphere", tool_start=tuple(normalized),
                            enforce_speed_limits=False, init_taichi=False, k_init=config["k_fixed"],
                            precision=precision)
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.tool_cut_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    sim.set_target_params(radius_mm=6.0)
    sim.bake_target_grid()
    return O.CandidateEvaluator(task, motion, parameterization, sim,
                                spec=O.ObjectiveSpec(k=float(config["k_smooth"])), seed=seed,
                                requested_precision=precision)


def _budget(config: Mapping[str, Any], method: str) -> B.OptimizationBudget:
    assert method != "nominal-cam"
    # Raw config retains the previously declared search evaluations. Exactly
    # one ordinary hard θ0 evaluation precedes every optimizer's updates.
    return B.OptimizationBudget(
        candidate_evaluations=config["candidate_evaluations"] + 1,
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
        "precision": _requested_precision(config, pilot=config.get("task_source", "pilot_pocket") == "pilot_pocket"),
        "seeds": config["seeds"], "candidate_evaluations": config["candidate_evaluations"],
        "population_size": config["population_size"], "validation_interval": config["validation_interval"],
        "max_validation_calls": config["max_validation_calls"],
        "learning_rate": config["learning_rate"], "k_fixed": config["k_fixed"], "k_smooth": config["k_smooth"],
        "k_anneal_start": config["k_anneal_start"], "k_anneal_end": config["k_anneal_end"],
        "max_generations": config.get("max_generations"),
        "hard_init_accounting": "one ordinary hard candidate before every optimizer update",
        "hard_selection_policy": dataclasses.asdict(B.HardSelectionPolicy()),
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
    remaining = budget.candidate_evaluations - 1
    iterations = remaining
    if budget.max_generations is not None:
        iterations = min(iterations, budget.max_generations)
    if method.endswith("cmaes") and budget.population_size is not None:
        iterations = min(iterations, remaining // budget.population_size)
    if iterations <= 0:
        raise RunnerError("the requested generation budget selects no work")
    # Budget total includes θ0; generations/gradient updates remain search work.
    candidates = 1 + iterations
    if method.endswith("cmaes"):
        candidates = 1 + iterations * budget.population_size
        # A generation evaluates members, but can trigger at most one validation.
    # For Adam, candidates are init plus gradient evaluations.
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
        if evidence["schema"] != "diffcam-pilot-timing/2":
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
        measured_post = _finite_seconds(
            evidence["posted_evaluation_levels"]["screen"]["phase_wall_seconds"], "screen phase")
    if _refinement_policy(config)["enabled"]:
        levels = evidence.get("posted_evaluation_levels")
        if not isinstance(levels, Mapping) or any(
                key not in levels or not isinstance(levels[key], Mapping)
                for key in ("screen", "verdict", "convergence")):
            raise RunnerError("timing evidence requires real screen, fine-verdict and convergence phases")
        refined_seconds = (
            _finite_seconds(levels["verdict"].get("phase_wall_seconds"), "fine verdict phase")
            + _finite_seconds(levels["convergence"].get("phase_wall_seconds"), "convergence phase"))
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
            init_seconds = 0.0 if entry["method"] == "nominal-cam" else _finite_seconds(
                evidence["candidate_seconds"]["hard_forward"], "hard initialization")
            setup = _finite_seconds(evidence["constructor_seconds"], "constructor_seconds")
            other = (_finite_seconds(evidence["worker_wall_seconds"], "worker_wall_seconds")
                     - setup - sum(_finite_seconds(evidence["candidate_seconds"].get(key), key)
                                   for key in ("smooth_forward", "smooth_gradient", "hard_forward"))
                     - measured_post - refined_seconds)
            extra = {
                "constructor_seconds": setup,
                "candidate_seconds": unit * (candidates - int(entry["method"] != "nominal-cam")),
                "initialization_seconds": init_seconds,
                "validation_seconds": _finite_seconds(evidence["candidate_seconds"]["hard_forward"], "hard_forward") * validations,
                "posted_seconds": measured_post,
                "other_worker_seconds": other,
                "estimation": "one measured point per evaluation type, not a timing confidence interval",
            }
            estimate = (setup + extra["candidate_seconds"] + extra["initialization_seconds"]
                        + extra["validation_seconds"] + measured_post + other)
            if other < -1e-9:
                raise RunnerError("timing evidence has a negative process-overhead residual")
            # Candidate durations are recorded to nanoseconds; do not make a
            # valid worker fail solely because a sum rounds a fraction above its wall clock.
            other = max(other, 0.0)
            extra["other_worker_seconds"] = other
            estimate = (setup + extra["candidate_seconds"] + extra["initialization_seconds"]
                        + extra["validation_seconds"] + measured_post + other)
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


class _OverridePhaseEvaluator:
    """Unit seam for direct typed responses, never a claimed independent worker."""

    def __init__(self, evaluator, config):
        self.evaluate = getattr(evaluator, "evaluate_posted_gcode", evaluator)
        if not callable(self.evaluate):
            raise RunnerError("posted override must implement the typed evaluator protocol")
        self._config = dict(config)
        self.phase_records = []
        self.timings = []

    def evaluate_posting(self, task, posting, workspace, *, role, single_dialect=False):
        outputs = {}
        processes = []
        phase_started = time.perf_counter()
        for post in posting.posted:
            nc_name, eval_name = C.DIALECT_ARTIFACTS[post.dialect]
            nc_bytes = posting.files[nc_name]
            settings = _phase_settings(task, self._config, post.dialect, nc_bytes, role)
            started = time.perf_counter()
            result = self.evaluate(task, nc_bytes, post.dialect, settings)
            elapsed = time.perf_counter() - started
            result = _validate_posted_result(task, nc_bytes, post.dialect, settings, result)
            outputs[post.dialect] = result
            path = workspace / eval_name
            payload = R.record_bytes(result)
            processes.append({
                "role": role, "dialect": post.dialect.value, "status": "test_override",
                "worker_pid": None, "worker_peak_rss_kib": None,
                "evaluator_wall_seconds": elapsed, "process_wall_seconds": elapsed,
                "evaluation_sha256": C.sha256_bytes(payload), "evaluation_path": str(path),
                "source": str(path), "max_step_mm": settings["max_step_mm"],
                "expected_nc_hash": result.nc_hash, "thread_environment": None,
                "memory_note": "direct in-process test override; no independent worker or RSS measurement",
            })
            self.timings.append({"dialect": post.dialect.value, "nc_hash": result.nc_hash,
                                 "wall_seconds": elapsed})
        self.phase_records = [{"phase_wall_seconds": time.perf_counter() - phase_started,
                               "processes": processes}]
        return outputs


def _phase_evaluator_adapter(task, config, override=None):
    if override is None:
        return PilotPostedEvaluator(task, config)
    if callable(getattr(override, "evaluate_posting", None)):
        return override
    return _OverridePhaseEvaluator(override, config)


def _collect_phase_sample(task, posting, evaluator, config, workspace, role, *, evaluation_root, in_process_override=False):
    """Measure one fidelity phase and retain actual typed evaluation artifacts."""
    evaluate = getattr(evaluator, "evaluate_posting", None)
    if not callable(evaluate):
        raise RunnerError("production timing requires the independent posted-batch protocol")
    expected_dialects = [C.Dialect.RS274] if role == "convergence" else list(C.Dialect)
    if [post.dialect for post in posting.posted] != expected_dialects:
        raise RunnerError(f"{role} timing requires dialects {expected_dialects}")
    for post in posting.posted:
        if post.status != C.PostStatus.SUPPORTED:
            raise RunnerError(f"{role} requires supported posting: {post.reason}")
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True)
    started = time.perf_counter()
    if isinstance(getattr(evaluator, "timings", None), list):
        evaluator.timings.clear()  # Events are for this role only, never an accumulated cross-phase estimate.
    evaluations = evaluate(task, posting, workspace, role=role, single_dialect=(role == "convergence"))
    artifacts = {}
    for post in posting.posted:
        nc_name, eval_name = C.DIALECT_ARTIFACTS[post.dialect]
        nc_bytes = posting.files[nc_name]
        if post.dialect not in evaluations:
            raise RunnerError(f"{role} omitted typed evaluation for {post.dialect.value}")
        actual = _validate_posted_result(
            task, nc_bytes, post.dialect,
            _phase_settings(task, config, post.dialect, nc_bytes, role), evaluations[post.dialect])
        if role == "screen":
            if not actual.efficiency_eligible:
                raise RunnerError(
                    f"timing nominal candidate is not eligible in {post.dialect.value}: "
                    f"coverage={actual.coverage}, safety={actual.safety_pass}, geometry={actual.geometry_pass}")
            if not actual.metrics["removed_volume_mm3"].value > 0:
                raise RunnerError("timing measurement may not treat a no-cut path as completed material removal")
        payload = R.record_bytes(actual)
        nc_path = workspace / nc_name
        eval_path = workspace / eval_name
        R.ArtifactWriter._replace_bytes(nc_path, nc_bytes)
        R.ArtifactWriter._replace_bytes(eval_path, payload)
        artifacts[post.dialect.value] = {
            "nc_path": str(nc_path.relative_to(evaluation_root)),
            "evaluation_path": str(eval_path.relative_to(evaluation_root)),
            "nc_hash": actual.nc_hash,
            "evaluation_sha256": C.sha256_bytes(payload),
        }
    if getattr(evaluator, "phase_records", None):
        phase = evaluator.phase_records[-1]
        processes = phase["processes"]
        for record in processes:
            path = record.get("evaluation_path")
            if path is not None and C.sha256_bytes(Path(path).read_bytes()) != record["evaluation_sha256"]:
                raise RunnerError("native phase result does not match retained evaluation bytes")
    else:
        timings = list(getattr(evaluator, "timings", []))
        processes = []
        for post in posting.posted:
            event = next((item for item in timings if item.get("dialect") == post.dialect.value), None)
            seconds = event["wall_seconds"] if event is not None else 0.0
            actual = evaluations[post.dialect]
            processes.append({
                "role": role, "dialect": post.dialect.value,
                "status": "test_override",
                "worker_pid": None, "worker_peak_rss_kib": None,
                "evaluator_wall_seconds": seconds, "process_wall_seconds": seconds,
                "evaluation_sha256": artifacts[post.dialect.value]["evaluation_sha256"],
                "evaluation_path": artifacts[post.dialect.value]["evaluation_path"],
                "source": str(workspace / (post.eval_file or C.DIALECT_ARTIFACTS[post.dialect][1])),
                "max_step_mm": actual.discretization["settings"]["max_step_mm"],
                "expected_nc_hash": actual.nc_hash,
                "thread_environment": None,
                "memory_note": "in-process evaluator override; no independent posted worker or measured peak RSS",
            })
    measured_time = time.perf_counter() - started
    policy = _refinement_policy(config)
    for record in processes:
        if record.get("role") != role or record.get("dialect") not in artifacts:
            raise RunnerError("phase resource record has a different role or dialect")
        _validate_resource_record(
            {"job_policy": dict(policy, total_posted_rss_ceiling_bytes=(
                policy["parallel_jobs"] * policy["max_rss_bytes_per_job"])),
             "phase_wall_seconds": measured_time, "processes": processes}, config,
            {dialect: artifacts[dialect]["nc_hash"] for dialect in artifacts}, workspace,
            phase=role, allow_test_override=in_process_override)
    return {"job_policy": dict(policy, total_posted_rss_ceiling_bytes=(
        policy["parallel_jobs"] * policy["max_rss_bytes_per_job"])),
        "phase_wall_seconds": measured_time, "processes": processes, "artifacts": artifacts}


def _refinement_projection(config, entries, evidence, root):
    """Project each posted fidelity phase from real typed, memory-limited jobs."""
    if evidence is None or not _refinement_policy(config)["enabled"]:
        return {"enabled": False, "estimated_phase_wall_time_s": 0.0,
                "levels": {}, "serial_evaluation_work_seconds": 0.0}
    if not isinstance(evidence, Mapping):
        raise RunnerError("posted timing evidence must be a structured mapping")
    samples = evidence.get("posted_evaluation_levels")
    if not isinstance(samples, Mapping) or set(samples) != {"screen", "verdict", "convergence"}:
        raise RunnerError("timing evidence must contain screen, verdict and convergence phase samples")
    policy = _refinement_policy(config)
    projected = {}
    for role in ("screen", "verdict", "convergence"):
        sample = samples[role]
        if not isinstance(sample, Mapping):
            raise RunnerError(f"missing typed phase measurement: {role}")
        dialects = [C.Dialect.RS274.value] if role == "convergence" else [d.value for d in C.Dialect]
        artifacts = sample.get("artifacts", {})
        job_policy = dict(policy, total_posted_rss_ceiling_bytes=(
            policy["parallel_jobs"] * policy["max_rss_bytes_per_job"]))
        sample = dict(sample, job_policy=job_policy)
        hashes = {}
        for dialect in artifacts:
            if C.Dialect(dialect).value not in dialects:
                raise RunnerError(f"unexpected {role} phase dialect {dialect}")
        for dialect in dialects:
            artifact = artifacts.get(dialect, {})
            nc_path = audit._artifact_path(root, artifact.get("nc_path"))
            nc_bytes = nc_path.read_bytes()
            nc_hash = C.nc_hash(nc_bytes)
            if nc_hash != artifact.get("nc_hash"):
                raise RunnerError(f"{role}/{dialect}: NC hash no longer matches the measured bytes")
            eval_path = audit._artifact_path(root, artifact.get("evaluation_path"))
            payload = eval_path.read_bytes()
            if C.sha256_bytes(payload) != artifact.get("evaluation_sha256"):
                raise RunnerError(f"{role}/{dialect}: evaluation file hash changed")
            evaluation = C.loads(payload.decode("utf-8"))
            if evaluation.dialect.value != dialect or evaluation.task_hash != evidence.get("task_hash"):
                raise RunnerError(f"{role}/{dialect}: evaluation task/dialect mismatch")
            if evaluation.tool_setup_hash != evidence.get("tool_setup_hash"):
                raise RunnerError(f"{role}/{dialect}: tool identity mismatch")
            if evaluation.discretization["settings"]["max_step_mm"] != policy[f"{role}_step_mm"]:
                raise RunnerError(f"{role}/{dialect}: required role sampling step was not used")
            hashes[dialect] = nc_hash
        _validate_resource_record(sample, config, hashes, root, phase=role,
                                  allow_test_override=bool(evidence.get("in_process_unit_override", False)))
        processes = sample["processes"]
        rows = {}
        for dialect in dialects:
            row = next(row for row in processes if row["dialect"] == dialect)
            rows[dialect] = _finite_seconds(row["evaluator_wall_seconds"], f"{role}/{dialect}")
        serial_program_seconds = sum(rows.values())
        phase_wall = _finite_seconds(sample["phase_wall_seconds"], f"{role} phase")
        projected[role] = {
            "max_step_mm": policy[f"{role}_step_mm"], "dialects": dialects,
            "per_program_seconds": rows, "serial_evaluation_work_per_program_seconds": serial_program_seconds,
            "measured_phase_wall_seconds": phase_wall,
            "serial_evaluation_work_seconds": serial_program_seconds * len(entries),
            "estimated_phase_wall_time_s": phase_wall * len(entries),
            "actual_independent_job_count": sum(row["status"] == "complete" for row in processes),
            "memory_ceiling_per_job_bytes": policy["max_rss_bytes_per_job"],
            "concurrent_memory_ceiling_bytes": len(processes) * policy["max_rss_bytes_per_job"],
            "peak_rss_kib_by_dialect": {row["dialect"]: row.get("worker_peak_rss_kib") for row in processes},
            "resources": processes, "artifacts": artifacts,
        }
    refined_wall = sum(projected[role]["estimated_phase_wall_time_s"]
                       for role in ("verdict", "convergence"))
    return {
        "enabled": True, "expected_program_count": len(entries),
        "parallel_jobs": policy["parallel_jobs"], "program_level_parallel_jobs": 1,
        "screen_in_primary_estimates": True,
        "levels": projected,
        "serial_evaluation_work_seconds": sum(
            projected[role]["serial_evaluation_work_seconds"] for role in projected),
        "estimated_phase_wall_time_s": refined_wall,
        "memory_per_posted_job_bytes": policy["max_rss_bytes_per_job"],
        "optimizer_parent_peak_rss_kib": evidence.get("optimizer_parent_peak_rss_kib"),
        "optimizer_reference_peak_rss_bytes_estimated": evidence.get(
            "optimizer_reference_peak_rss_bytes_estimated"),
        "optimizer_reference_cost_note": evidence.get("optimizer_reference_cost_note"),
        "optimizer_parallelism": "serialized by default",
        "estimation_basis": (
            "one real typed process batch per role; serial program groups; no parallel-speedup or confidence-interval claim"
            if not evidence.get("in_process_unit_override", False)
            else "explicit in-process typed unit fixture; not independent subprocess evidence or a production f64 cost estimate"),
    }


def _measure_budget(config, root, *, evaluator_override=None, capture_dir=None,
                    refinement_evaluator_override=None, posted_evaluator_override=None):
    """Measure one of each execution kind, with no population/Adam optimization.

    ``evaluator_override`` is a pure in-process test seam (no untrusted module
    path/name loading). Production callers always construct one real registered
    CandidateEvaluator and one real production PostedEvaluator factory.
    """
    from datetime import datetime, timezone
    started_at = datetime.now(timezone.utc).isoformat()
    if capture_dir is None:
        capture_dir = Path(root).resolve() / "posted-phase-artifacts"
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
    if posted_evaluator_override is not None and refinement_evaluator_override is not None:
        raise RunnerError("choose at most one posted-evaluation override")
    in_process_override = bool(evaluator_override or posted_evaluator_override or refinement_evaluator_override)
    posted_override = posted_evaluator_override or refinement_evaluator_override
    override = posted_override if posted_override is not None else posted_factory(task, config)
    post_builder = _phase_evaluator_adapter(task, config, override)
    phases = {}
    for role in ("screen", "verdict", "convergence"):
        workspace = capture_dir / role
        if role == "convergence":
            sample_posting = R.PostingResult(
                files={"rs274.nc": posting.files["rs274.nc"]},
                posted=tuple(post for post in posting.posted if post.dialect == C.Dialect.RS274),
            )
        else:
            sample_posting = posting
        phases[role] = _collect_phase_sample(
            task, sample_posting, post_builder, config, workspace, role,
            evaluation_root=root, in_process_override=in_process_override)
        if role == "screen":
            posted_times = list(post_builder.timings)
    expected_events = [(post.dialect.value, post.nc_hash) for post in posting.posted]
    actual_events = [(event["dialect"], event["nc_hash"]) for event in posted_times]
    if len(posted_times) != 2 or actual_events != expected_events:
        raise RunnerError("exactly two real, identity-matched posted evaluations must be timed")
    posted_seconds = [float(record["wall_seconds"]) for record in posted_times]
    if sys.platform.startswith("linux"):
        optimizer_peak_rss_kib = _self_peak_rss_kib()  # own VmHWM; ru_maxrss would include a launcher's peak
    else:
        optimizer_peak_rss_kib = None
    return {
        "schema": "diffcam-pilot-timing/2",
        "in_process_unit_override": in_process_override,
        "simulator_precision": R.simulator_precision_metadata(getattr(evaluator, "sim", None)),
        "optimizer_parent_peak_rss_kib": optimizer_peak_rss_kib,
        "optimizer_reference_peak_rss_bytes_estimated": 3_670_000_000,
        "optimizer_reference_cost_note": "D-05 f64 coordinator estimate, not a measured claim about this sample",
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
        "code_sha256": _timing_module_hashes(),
        "candidate_seconds": records,
        "posted_seconds": posted_seconds,
        "posted_seconds_dialects": posted_times,
        "posted_evaluation_levels": phases,
        "source_stage": {
            "target": "analytic operation target for posted execution; exact sampled copy for simulator",
            "sources": {name: {"path": source.path, "sha256": source.sha256}
                        for name, source in evaluator.geometry_sources.items()},
        },
        "started_at": started_at,
        "worker_wall_seconds": time.perf_counter() - created,
        "posted_phase_wall_seconds": time.perf_counter() - before_post,
        "scope": "one real candidate of each kind, 1.0 mm screen in both dialects, 0.125 mm verdict in both dialects, and one 0.25 mm RS274 convergence evaluation; not the 7-run pilot",
    }


def _timing_module_hashes():
    modules = {
        "runner": Path(__file__).resolve(),
        "posted_evaluator": Path(sys.modules[PostedEvaluator.__module__].__file__).resolve(),
        "strict_parser": Path(sys.modules["cam.gcode_parser"].__file__).resolve(),
        "post_writer": Path(sys.modules["cam.posts"].__file__).resolve(),
        "task_builder": Path(pilot_io.__file__).resolve(),
        "objectives": Path(O.__file__).resolve(),
        "simulator": Path(sys.modules[CSGSimulatorDelta.__module__].__file__).resolve(),
        "run_artifacts": Path(R.__file__).resolve(),
    }
    return {name: C.sha256_bytes(path.read_bytes()) for name, path in modules.items()}


def _validate_production_timing_evidence(config, evidence, task):
    if not isinstance(evidence, Mapping) or evidence.get("schema") != "diffcam-pilot-timing/2":
        raise RunnerError("production timing requires the current typed schema /2")
    precision = _requested_precision(config, pilot=True)
    if evidence.get("in_process_unit_override") is not False:
        raise RunnerError("production timing requires real f64 simulation and independent posted workers, not in-process overrides")
    if (evidence.get("task_hash") != task.task_hash
            or evidence.get("tool_setup_hash") != task.tool_setup_hash):
        raise RunnerError("timing evidence task/tool identity differs from the frozen task")
    metadata = evidence.get("simulator_precision")
    record = SimpleNamespace(
        precision=metadata.get("requested_precision") if isinstance(metadata, Mapping) else None,
        precision_metadata=lambda: metadata,
    )
    try:
        R.simulator_precision_metadata(record, requested_precision=precision)
    except ValueError as error:
        raise RunnerError(f"timing evidence is not a valid {precision} simulator measurement: {error}") from error
    expected_config = {key: value for key, value in config.items() if key != "timing_evidence"}
    if evidence.get("config_sha256") != C.sha256_bytes(R.json_bytes(expected_config)):
        raise RunnerError("timing evidence differs from the exact frozen pilot config")
    if evidence.get("code_sha256") != _timing_module_hashes():
        raise RunnerError("timing evidence module hashes differ from current evaluator/simulator code")
    _positive_int(evidence.get("optimizer_parent_peak_rss_kib"), "optimizer parent peak RSS KiB")
    _finite_seconds(evidence.get("worker_wall_seconds"), "timing worker whole process")
    phases = evidence.get("posted_evaluation_levels")
    if not isinstance(phases, Mapping):
        raise RunnerError("production timing lacks role-specific independent process records")
    for role, expected_jobs in (("screen", 2), ("verdict", 2), ("convergence", 1)):
        phase = phases.get(role)
        processes = phase.get("processes") if isinstance(phase, Mapping) else None
        if (not isinstance(processes, list) or len(processes) != expected_jobs
                or any(record.get("status") != "complete" for record in processes)):
            raise RunnerError(f"timing phase {role} must contain independent posted-worker records, not reused or in-process data")
    return evidence


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
    data["schema"] = "diffcam-pilot-timing/2"
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


def _evaluate_outcome(condition: str, evaluator: Any, config: Mapping[str, Any], seed: int,
                      run_id: Optional[str] = None) -> Outcome:
    run_id = run_id or condition
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
                                run_id=run_id)
        if result.best is None:
            raise RunnerError(f"{condition}: no valid candidate from whole population")
        return Outcome(seed, result.best, result.best_theta, result.initial_theta,
                       result.budget, result.evaluations, result.validation_calls, result.generations,
                       getattr(result, "forward_calls", 0), getattr(result, "backward_calls", 0),
                       result.invalid_candidates, result.termination_reason, result.history,
                       optimizer_result=result)
    if condition == "fixed-adam":
        mode, k_start, k_end = "fixed", config["k_fixed"], config["k_fixed"]
    else:
        mode, k_start, k_end = "annealed", config["k_anneal_start"], config["k_anneal_end"]
    cfg = adam.AdamConfig(mode, learning_rate=config["learning_rate"],
                          max_gradient_norm=config.get("max_gradient_norm", 0.5),
                          k_start=k_start, k_end=k_end)
    result = adam.solve_adam(evaluator, _budget(config, condition), cfg, seed=seed,
                             x0=zeros, run_id=run_id)
    if result.best is None:
        raise RunnerError(f"{condition}: no valid hard-selected candidate "
                          f"(termination_reason={result.termination_reason})")
    return Outcome(seed, result.best, result.best_theta, result.initial_theta, result.budget,
                   result.evaluations, result.validation_calls, result.gradient_updates,
                   result.forward_calls, result.backward_calls, result.invalid_evaluations,
                   result.termination_reason, result.history, optimizer_result=result)


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
                                    simulator=getattr(evaluator, "sim", None),
                                    budgets={"limits": dataclasses.asdict(outcome.budget),
                                             "spent": {"candidate_evaluations": outcome.evaluations,
                                                       "validation_calls": outcome.validation_calls,
                                                       "forward_calls": outcome.forward_calls,
                                                       "backward_calls": outcome.backward_calls,
                                                       "generations": outcome.generations}},
                                    repository=repository)
    optimized = outcome.optimizer_result
    config_out["hard_selection"] = (
        None if optimized is None else {
            "policy": dataclasses.asdict(optimized.selection),
            "resolved": optimized.selection_thresholds,
        })
    config_out["hard_selection_config_hash"] = (
        None if optimized is None or optimized.selection_thresholds is None
        else optimized.selection_thresholds["config_hash"])
    initial = evaluator.parameterization.positions(outcome.initial_theta)
    best = evaluator.parameterization.positions(outcome.best_theta)
    files = dict(posting.files)
    files.update({
        "config.json": R.json_bytes(config_out),
        "history.json": R.json_bytes({"history": outcome.history, "ledger": evaluator.ledger.snapshot()}),
        "metrics.json": R.json_bytes({"best_candidate": B._candidate_dict(outcome.best),
                                      "termination_reason": outcome.termination_reason,
                                      "invalid_candidates": outcome.invalid_candidates,
                                      "selection": (None if optimized is None
                                                    else B._selection_metadata(optimized)),
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
        if isinstance(evaluator_for_posts, PilotPostedEvaluator):
            evaluations = evaluator_for_posts.evaluate_posting(
                task, posting, root / "posted-jobs" / entry["run_id"] / "screen")
            for post in posting.posted:
                nc_name, eval_name = C.DIALECT_ARTIFACTS[post.dialect]
                actual = evaluations[post.dialect]
                if actual.dialect != post.dialect or actual.nc_hash != post.nc_hash or actual.task_hash != task.task_hash:
                    raise RunnerError("posted evaluator returned a mismatched dialect/task/NC hash")
                evaluated[post.dialect] = actual
                files[eval_name] = R.record_bytes(actual)
        else:
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
        if getattr(evaluator_for_posts, "phase_records", None):
            details["execution"]["posted_evaluation_phases"] = evaluator_for_posts.phase_records
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
        outcome = _evaluate_outcome(entry["method"], evaluator, config, entry["seed"],
                                    run_id=entry["run_id"])
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
        result[name]["changed"] for name in (
            "safety_pass", "geometry_pass", "coverage", "efficiency_eligible"))
    return result


def _safe_refinement_directory(primary_root: Path, auxiliary_root: Path) -> tuple[Path, Path]:
    primary_root, auxiliary_root = primary_root.resolve(), auxiliary_root.resolve()
    if auxiliary_root.is_relative_to(primary_root) or primary_root.is_relative_to(auxiliary_root):
        raise RunnerError("primary and auxiliary refinement roots must be disjoint sibling trees")
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


def _read_phase(task, directory, config, role, *, evaluated_dir=None, screen_copy=False, primary=None):
    """Read and validate actual posted-evaluator JSON for one fidelity role."""
    evaluated_dir = directory if evaluated_dir is None else evaluated_dir
    dialects = (C.Dialect.RS274,) if role == "convergence" else tuple(C.Dialect)
    evaluated = {}
    for dialect in dialects:
        nc_name, eval_name = C.DIALECT_ARTIFACTS[dialect]
        if role == "convergence":
            eval_name = "eval_convergence_rs274.json"
        elif role == "screen" and screen_copy:
            eval_name = "screen_" + eval_name
        nc_bytes = (directory / nc_name).read_bytes()
        source = primary.get(dialect) if isinstance(primary, Mapping) else primary if dialect == C.Dialect.RS274 else None
        settings = _phase_settings(task, config, dialect, nc_bytes, role, primary=source)
        try:
            payload = (evaluated_dir / eval_name).read_bytes()
        except FileNotFoundError as error:
            raise RunnerError(f"missing {role} posted evaluation for {dialect.value}: {evaluated_dir / eval_name}") from error
        try:
            decoded = C.loads(payload.decode("utf-8"))
        except (C.ContractError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RunnerError(f"invalid {role} evaluation JSON for {dialect.value}: {evaluated_dir / eval_name}") from error
        result = _validate_posted_result(task, nc_bytes, dialect, settings, decoded)
        if R.record_bytes(result) != payload:
            raise RunnerError(f"noncanonical {role} evaluation bytes: {evaluated_dir / eval_name}")
        evaluated[dialect] = result
    return evaluated


def _evaluation_outcome(evaluation, task):
    """The shared task-qualified outcome for one actual typed evaluation."""
    info = audit.TaskInfo(task.task_id, task.task_hash, task.family,
                          completion_residual_mm3=task.tolerances.completion_residual_mm3)
    return {
        "coverage": evaluation.coverage.value,
        "safety_pass": evaluation.safety_pass,
        "geometry_pass": evaluation.geometry_pass,
        "completion_pass": audit._completion_value(evaluation, info),
        "efficiency_eligible": audit._outcome_value(evaluation, info, "efficiency_eligible"),
    }


def _fidelity_deltas(left, right, task):
    comparison = _refinement_comparison(left, right)
    left_outcome = _evaluation_outcome(left, task)
    right_outcome = _evaluation_outcome(right, task)
    comparison["completion_pass"] = {
        "primary": left_outcome["completion_pass"], "refined": right_outcome["completion_pass"],
        "changed": left_outcome["completion_pass"] != right_outcome["completion_pass"],
    }
    comparison["any_outcome_verdict_change"] = any(
        comparison[name]["changed"] for name in (
            "safety_pass", "geometry_pass", "completion_pass", "coverage", "efficiency_eligible"))
    comparison["changed_outcome_fields"] = [
        name for name in ("safety_pass", "geometry_pass", "completion_pass", "coverage", "efficiency_eligible")
        if comparison[name]["changed"]]
    return comparison


def _screen_fidelity_diagnostics(screen, verdict, task):
    comparisons = {
        dialect.value: _fidelity_deltas(screen[dialect], verdict[dialect], task)
        for dialect in C.Dialect
    }
    fields = sorted({name for comparison in comparisons.values()
                     for name in comparison["changed_outcome_fields"]})
    return comparisons, fields


def _provided_typed_source(task, job, override):
    if override is None:
        return None
    if isinstance(override, C.EvaluationResult):
        supplied = override
    elif isinstance(override, Mapping):
        supplied = override.get((job.role, job.dialect.value))
    else:
        evaluate = getattr(override, "evaluate_posted_gcode", override)
        if not callable(evaluate):
            raise RunnerError("posted-evaluation override must implement the typed protocol")
        supplied = evaluate(task, job.nc_bytes, job.dialect, dict(job.settings))
    if supplied is None:
        return None
    if isinstance(supplied, Path):
        source = supplied.resolve()
        cached = C.loads(source.read_text(encoding="utf-8"))
        _validate_posted_result(task, job.nc_bytes, job.dialect, job.settings, cached,
                                require_current_evaluator=job.require_current_evaluator)
        R.record_bytes(cached)  # Preserve the existing canonical typed boundary.
        return source
    source_bytes = R.record_bytes(supplied)
    _validate_posted_result(task, job.nc_bytes, job.dialect, job.settings, supplied,
                            require_current_evaluator=job.require_current_evaluator)
    job.workspace.mkdir(parents=True, exist_ok=False)
    source = job.workspace / "provided-typed-result.json"
    R.ArtifactWriter._replace_bytes(source, source_bytes)
    return source


def _configured_typed_sources(config):
    """Explicit immutable, canonical typed sources; never automatic cache discovery."""
    declared = config.get("reuse_typed_evaluations")
    if declared is None:
        return {}
    if not isinstance(declared, Mapping):
        raise RunnerError("reuse_typed_evaluations must be a mapping of role/dialect to path+sha256")
    sources = {}
    for key, entry in declared.items():
        try:
            role, dialect_name = key.split("/")
            dialect = C.Dialect(dialect_name)
            if role not in ("verdict", "convergence") or (role == "convergence" and dialect != C.Dialect.RS274):
                raise ValueError("roles are verdict and RS274-only convergence")
            if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"} or not isinstance(entry["path"], str):
                raise ValueError("declared source requires exactly path and sha256")
            source = Path(entry["path"])
            if not source.is_absolute():
                raise ValueError("reused source must be an explicitly named absolute path")
            payload = source.resolve(strict=True).read_bytes()
            if entry["sha256"] != C.sha256_bytes(payload):
                raise ValueError("declared source hash differs from the actual bytes")
            result = C.loads(payload.decode("utf-8"))
            if R.record_bytes(result) != payload:
                raise ValueError("reused result is not canonical typed production JSON")
            source_key = (role, dialect.value)
            if source_key in sources:
                raise ValueError("duplicate role/dialect reuse declaration")
            sources[source_key] = source.resolve()
        except (ValueError, TypeError, OSError, AttributeError) as error:
            raise RunnerError(f"invalid reuse_typed_evaluations[{key!r}]: {error}") from error
    return sources


def _refinement_worker(config, root, run_id, auxiliary_root, *, evaluator_override=None, reuse_sources=None):
    """Create fine verdict evidence for both dialects and an RS274 convergence point."""
    policy = _refinement_policy(config)
    if not policy["enabled"]:
        raise RunnerError("posted-fidelity evaluation was disabled by the frozen policy")
    root, auxiliary_root = _safe_refinement_directory(root, Path(auxiliary_root))
    directory, record, parent_config, task, primary, _ = _load_refinement_case(root, run_id)
    auxiliary_id = f"{run_id}-verdict"
    auxiliary_dir = auxiliary_root / auxiliary_id
    started = time.perf_counter()
    if reuse_sources is None:
        reuse_sources = _configured_typed_sources(config)
    if not isinstance(reuse_sources, Mapping):
        raise RunnerError("reuse_sources must be keyed by (role, dialect)")
    if auxiliary_dir.exists():
        audited = audit._audit_run(auxiliary_dir / "run.json")
        if not audited.complete_and_audited or audited.record.run_id != auxiliary_id:
            raise RunnerError("existing verdict bundle is not a complete matching auxiliary run")
        history = _json(auxiliary_dir / "history.json")
        resources = history.get("resources")
        if not isinstance(resources, Mapping) or not isinstance(resources.get("processes"), list):
            raise RunnerError("existing verdict bundle lacks per-job process/resource records")
        status = "skipped_complete"
    else:
        batch_workspace = auxiliary_root / "posted-evaluation-work" / f"{run_id}-{time.time_ns()}"
        jobs = []
        for role, dialect in (("verdict", C.Dialect.RS274), ("verdict", C.Dialect.HAAS),
                              ("convergence", C.Dialect.RS274)):
            nc_bytes = (directory / C.DIALECT_ARTIFACTS[dialect][0]).read_bytes()
            source = primary if dialect == C.Dialect.RS274 else None
            settings = _phase_settings(task, config, dialect, nc_bytes, role, primary=source)
            workspace = batch_workspace / role / dialect.value
            reused = reuse_sources.get((role, dialect.value)) if reuse_sources is not None else None
            conflicting_override = evaluator_override
            if isinstance(evaluator_override, Mapping):
                conflicting_override = evaluator_override.get((role, dialect.value))
            if reused is not None and conflicting_override is not None:
                raise RunnerError("declare either a reused typed result or an evaluator override per job, not both")
            if reused is not None and not isinstance(reused, (C.EvaluationResult, Path)):
                raise RunnerError("reused evidence must be a typed EvaluationResult or canonical JSON path")
            job = PostedJob(role, dialect, nc_bytes, MappingProxyType(settings), workspace, reused)
            if isinstance(reused, C.EvaluationResult):
                job = dataclasses.replace(
                    job, reused_source=_provided_typed_source(task, job, reused))
            supplied = _provided_typed_source(task, job, evaluator_override)
            if supplied is not None:
                job = dataclasses.replace(job, reused_source=supplied)
            jobs.append(job)
        phases = []
        for phase_name in ("verdict", "convergence"):
            phase_jobs = [job for job in jobs if job.role == phase_name]
            phase = _post_jobs(task, phase_jobs, config)
            phases.append((phase_name, phase_jobs, phase))
        resources = {
            "job_policy": phases[0][2]["job_policy"],
            "phase_wall_seconds": sum(phase[2]["phase_wall_seconds"] for phase in phases),
            "processes": [record for _, _, phase in phases for record in phase["processes"]],
            "phase_records": [
                {"phase": name, "phase_wall_seconds": phase["phase_wall_seconds"],
                 "processes": phase["processes"]}
                for name, _, phase in phases
            ],
        }
        status = "complete"

    screen = _read_phase(task, directory, config, "screen",
                         evaluated_dir=auxiliary_dir if auxiliary_dir.exists() else directory,
                         screen_copy=auxiliary_dir.exists(), primary=primary)
    if auxiliary_dir.exists():
        verdict = _read_phase(task, directory, config, "verdict", evaluated_dir=auxiliary_dir, primary=primary)
        convergence = _read_phase(task, directory, config, "convergence", evaluated_dir=auxiliary_dir)[C.Dialect.RS274]
    else:
        all_results = [result for _, _, phase in phases for result in phase["results"]]
        verdict = {evaluation.dialect: evaluation for evaluation, metadata in all_results
                   if metadata["role"] == "verdict"}
        convergence = next(evaluation for evaluation, metadata in all_results
                           if metadata["role"] == "convergence")
    screen_deltas, disagreement_fields = _screen_fidelity_diagnostics(screen, verdict, task)
    screen_delta = screen_deltas[C.Dialect.RS274.value]
    haas_delta = screen_deltas[C.Dialect.HAAS.value]
    convergence_delta = _fidelity_deltas(convergence, verdict[C.Dialect.RS274], task)
    if not auxiliary_dir.exists():
        aux_config = dict(parent_config, run_id=auxiliary_id, refinement_of=run_id,
                          protocol="diffcam-posted-fidelity/4",
                          verdict_step_mm=policy["verdict_step_mm"],
                          convergence_step_mm=policy["convergence_step_mm"])
        spent = {"candidate_evaluations": 0, "forward_calls": 0, "backward_calls": 0,
                 "generations": 0, "validation_calls": 0,
                 "posted_worker_jobs": sum(item["status"] == "complete" for item in resources["processes"])}
        aux_config["budgets"] = {"limits": parent_config["budgets"]["limits"], "spent": spent}
        hist = {"stage": "fine-verdict-and-convergence-posted-evaluations", "parent_run_id": run_id,
                "resources": resources, "screen_is_diagnostic_only": True}
        metrics = {
            "screen_outcomes": {d.value: _evaluation_outcome(value, task) for d, value in screen.items()},
            "verdict_outcomes": {d.value: _evaluation_outcome(value, task) for d, value in verdict.items()},
            "screen_to_verdict_rs274": screen_delta,
            "screen_to_verdict_haas": haas_delta,
            "convergence_to_verdict_rs274": convergence_delta,
            "resource_record": resources, "refinement_of": run_id,
        }
        files = {}
        for dialect, nc_eval in screen.items():
            nc_name, eval_name = C.DIALECT_ARTIFACTS[dialect]
            files[nc_name] = (directory / nc_name).read_bytes()
            files["screen_" + eval_name] = (directory / eval_name).read_bytes()
        for dialect, evaluation in verdict.items():
            files[C.DIALECT_ARTIFACTS[dialect][1]] = R.record_bytes(evaluation)
        files["eval_convergence_rs274.json"] = R.record_bytes(convergence)
        files.update({
            "config.json": R.json_bytes(aux_config), "history.json": R.json_bytes(hist),
            "metrics.json": R.json_bytes(metrics),
            "motion_plan.json": (directory / "motion_plan.json").read_bytes(),
            "eval_posted.json": (directory / "eval_posted.json").read_bytes(),
            "initial_trajectory.npy": (directory / "initial_trajectory.npy").read_bytes(),
            "trajectory.npy": (directory / "trajectory.npy").read_bytes(),
            "trajectory_deltas.npy": (directory / "trajectory_deltas.npy").read_bytes(),
        })
        source_meta = R.source_metadata(REPOSITORY)
        aux_record = R.assemble_record(
            auxiliary_id, task, C.RunStatus.COMPLETE,
            posting=R.PostingResult(files={}, posted=record.posted),
            artifact_names=tuple(sorted(files)), source_commit=source_meta["source_commit"],
            dirty_patch_hash=source_meta["dirty_patch_hash"])
        R.ArtifactWriter().write(str(auxiliary_dir), aux_record, files)

    screen = _read_phase(task, directory, config, "screen", evaluated_dir=auxiliary_dir,
                         screen_copy=True, primary=primary)
    verdict = _read_phase(task, directory, config, "verdict", evaluated_dir=auxiliary_dir, primary=primary)
    convergence = _read_phase(task, directory, config, "convergence", evaluated_dir=auxiliary_dir)[C.Dialect.RS274]
    screen_deltas, disagreement_fields = _screen_fidelity_diagnostics(screen, verdict, task)
    screen_delta = screen_deltas[C.Dialect.RS274.value]
    haas_delta = screen_deltas[C.Dialect.HAAS.value]
    convergence_delta = _fidelity_deltas(convergence, verdict[C.Dialect.RS274], task)
    comparison_document = {
        "schema": "diffcam-fidelity-comparison/1", "primary_run_id": run_id,
        "verdict_run_id": auxiliary_id, "screen_to_verdict_rs274": screen_delta,
        "screen_to_verdict_haas": haas_delta,
        "screen_disagreement_fields": disagreement_fields,
        "screen_disagreement_fields_by_dialect": {
            dialect: comparison["changed_outcome_fields"] for dialect, comparison in screen_deltas.items()},
        "convergence_to_verdict_rs274": convergence_delta,
        "role": "diagnostics only; 0.125 mm posted evaluations in both dialects are the sole verdict source",
    }
    comparison_path = auxiliary_dir / "comparison.json"
    comparison_payload = R.json_bytes(comparison_document)
    if comparison_path.exists() and comparison_path.read_bytes() != comparison_payload:
        raise RunnerError("existing fidelity comparison differs; refusing to overwrite it")
    if not comparison_path.exists():
        R.ArtifactWriter._replace_bytes(comparison_path, comparison_payload)
    links = {}
    for role, values in (("screen", screen), ("verdict", verdict), ("convergence", {C.Dialect.RS274: convergence})):
        links[role] = {
            "max_step_mm": policy[f"{role}_step_mm"],
            "evaluations": {
                dialect.value: {
                    "evaluator_revision": evaluation.evaluator_revision,
                    "path": f"{auxiliary_id}/{'screen_' if role == 'screen' else ''}{C.DIALECT_ARTIFACTS[dialect][1] if role != 'convergence' else 'eval_convergence_rs274.json'}",
                    "sha256": C.sha256_bytes(R.record_bytes(evaluation)),
                } for dialect, evaluation in values.items()
            },
        }
    link = {
        "schema": "diffcam-refinement-link/4", "primary_run_id": run_id,
        "primary_run_path": str(directory.relative_to(root)),
        "primary_metrics_sha256": C.sha256_bytes((directory / "metrics.json").read_bytes()),
        "primary_rs274_evaluation_sha256": C.sha256_bytes((directory / "eval_posted_rs274.json").read_bytes()),
        "primary_haas_evaluation_sha256": C.sha256_bytes((directory / "eval_posted_haas.json").read_bytes()),
        "primary_run_json_sha256": C.sha256_bytes((directory / "run.json").read_bytes()),
        "verdict_run_id": auxiliary_id,
        "auxiliary_record": {"path": f"{auxiliary_id}/run.json",
                             "sha256": C.sha256_bytes((auxiliary_dir / "run.json").read_bytes())},
        "comparison_path": str(comparison_path.relative_to(auxiliary_root)),
        "comparison_sha256": C.sha256_bytes(comparison_payload),
        "screen_disagreement_fields": disagreement_fields, "roles": links,
        "rule": "only 0.125 mm evaluations in both dialects determine pilot verdicts; screen is diagnostic",
        "resources": resources,
    }
    link_path = auxiliary_root / f"{run_id}-refinement-link.json"
    if link_path.exists() and _json(link_path) != link:
        raise RunnerError("existing fidelity link differs; refusing to overwrite it")
    _atomic(link_path, link)
    return {
        "root": root, "run_id": run_id, "refined_run_id": auxiliary_id,
        "primary_config": parent_config, "source_config": config,
        "auxiliary_root": auxiliary_root, "auxiliary_directory": auxiliary_dir,
        "auxiliary_link_path": link_path, "status": status,
        "screen_disagreement_fields": disagreement_fields,
        "comparison": screen_delta, "haas_comparison": haas_delta,
        "convergence_comparison": convergence_delta,
        "verdict_evaluations": {d.value: {"path": str(auxiliary_dir / C.DIALECT_ARTIFACTS[d][1]),
                                           "sha256": C.sha256_bytes(R.record_bytes(e))}
                               for d, e in verdict.items()},
        "completion_residual_mm3": task.tolerances.completion_residual_mm3,
        "resources": resources, "link": link,
        "worker_seconds": time.perf_counter() - started,
        "new_spawned_workers": sum(item["status"] == "complete" for item in resources["processes"]),
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


def _case_fine_evaluations(case, auxiliary_root):
    task = C.TaskSpec.from_dict(case["primary_config"]["task"])
    directory = auxiliary_root / case["refined_run_id"]
    primary = C.loads((case["root"] / case["run_id"] / "eval_posted_rs274.json").read_text())
    verdict = _read_phase(task, case["root"] / case["run_id"], case["source_config"], "verdict",
                          evaluated_dir=directory, primary=primary)
    convergence = _read_phase(task, case["root"] / case["run_id"], case["source_config"],
                              "convergence", evaluated_dir=directory)[C.Dialect.RS274]
    return task, verdict, convergence


def _audit_refinement_root(outcomes, auxiliary_root):
    """Audit immutable screen baselines and their two-dialect fine verdicts."""
    tasks, runs, comparisons, reported = {}, {}, [], {}
    diagnostic_counts, resources = {}, {}
    seen_methods = set()
    for case in outcomes:
        expected = case["manifest"].runs[case["run_id"]]
        task_info = case["manifest"].tasks[expected.task_id]
        if case.get("completion_residual_mm3") is not None:
            task_info = dataclasses.replace(
                task_info, completion_residual_mm3=case["completion_residual_mm3"])
        previous = tasks.get(expected.task_id)
        if previous is not None and previous != task_info:
            raise RunnerError("one task id has inconsistent typed task/audit information")
        tasks[expected.task_id] = task_info
        runs[case["run_id"]] = expected
        _copy_primary_for_refinement(case["root"] / case["run_id"], auxiliary_root)
        aux_id = case["refined_run_id"]
        if aux_id in runs:
            raise RunnerError(f"duplicate baseline/verdict run identifier: {aux_id}")
        if case["auxiliary_directory"] != auxiliary_root / aux_id:
            raise RunnerError("verdict record points outside its declared auxiliary root")
        fine_method = expected.method + "-fine-verdict"
        runs[aux_id] = audit.ExpectedRun(aux_id, expected.task_id, fine_method, expected.seed)
        if expected.method not in seen_methods:
            seen_methods.add(expected.method)
            comparisons.append(audit.ComparisonSpec(
                name=f"{expected.method} 1.0-to-0.125 mm fidelity", method=fine_method,
                baseline=expected.method, dialect=C.Dialect.RS274,
                metrics=("residual_volume_mm3", "gouge_volume_mm3", "max_surface_error_mm"),
            ))
    for case in outcomes:
        task, verdict, convergence = _case_fine_evaluations(case, auxiliary_root)
        expected = case["manifest"].runs[case["run_id"]]
        fields = case["screen_disagreement_fields"]
        diagnostic_counts[expected.method] = diagnostic_counts.get(expected.method, 0) + int(bool(fields))
        paths = {dialect.value: f"{case['refined_run_id']}/{C.DIALECT_ARTIFACTS[dialect][1]}" for dialect in verdict}
        for dialect, evaluation in verdict.items():
            path = auxiliary_root / paths[dialect.value]
            if C.sha256_bytes(path.read_bytes()) != case["verdict_evaluations"][dialect.value]["sha256"]:
                raise RunnerError(f"fine evaluation hash mismatches the worker link: {case['run_id']}/{dialect.value}")
        convergence_path = auxiliary_root / case["refined_run_id"] / "eval_convergence_rs274.json"
        if convergence.discretization["settings"]["max_step_mm"] != 0.25:
            raise RunnerError("convergence evidence must be RS274 at 0.25 mm")
        reported[case["run_id"]] = {
            "condition": expected.method, "seed": expected.seed,
            "outcomes": {dialect.value: _evaluation_outcome(evaluation, task)
                         for dialect, evaluation in verdict.items()},
            "screen_disagreement_fields": fields,
            "verdict_step_mm": 0.125, "screen_step_mm": 1.0,
            "convergence_step_mm": 0.25,
            "verdict_evaluation_paths": paths,
            "convergence_evaluation_path": f"{case['refined_run_id']}/eval_convergence_rs274.json",
            "auxiliary_record_path": f"{case['refined_run_id']}/run.json",
            "auxiliary_link_path": os.path.relpath(case["auxiliary_link_path"], auxiliary_root),
        }
        resources[case["run_id"]] = case["resources"]
    manifest = audit.Manifest(tasks=tasks, runs=runs, comparisons=tuple(comparisons))
    manifest_path = auxiliary_root / "manifest.json"
    if manifest_path.exists():
        if audit.load_manifest(manifest_path) != manifest:
            raise RunnerError("fidelity auxiliary manifest differs; refusing mixed audit roots")
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
            "runner": {"schema": "diffcam-posted-fidelity/4",
                       "verdict_source": "both posted dialects at 0.125 mm; screen is diagnostic only",
                       "source_commit": R.source_metadata(REPOSITORY)["source_commit"],
                       "physical_machining_certificate": False},
        })
    reports, audit_result = audit.audit_runs(auxiliary_root, manifest)
    if not audit_result["summary"]["auditor_passed"]:
        raise RunnerError(f"auxiliary fidelity audit failed: {audit_result['summary']}")
    aggregate_rows, statistics = audit.aggregate_evaluations(
        manifest, audit_result["_audits_by_id"], bootstrap_replicates=100)
    document = {
        "schema": "diffcam-fidelity-audit/4",
        "manifest": audit_result["summary"], "records": reports,
        "reporting_policy": {
            "primary_files": "immutable; auxiliary root holds fine verdict, screen and convergence evidence",
            "verdicts": "only both 0.125 mm posted evaluations; never combined with 1.0 mm screen",
            "efficiency_eligibility": "computed only from the fine typed evaluation of each dialect",
            "quantity_comparison": "residual_volume_mm3, gouge_volume_mm3, max_surface_error_mm",
            "convergence": "0.25 mm RS274 point; not a monotonicity/error-bound claim",
        },
        "reported_outcomes": reported,
        "screen_disagreement_by_condition": {
            condition: {"count": count, "screen_disagreement": count > 0}
            for condition, count in diagnostic_counts.items()
        },
        "resources_by_run_id": resources,
        "comparisons": aggregate_rows, "statistics": statistics,
        "case_count": len(outcomes), "baseline_and_refined_records": len(reports),
        "audited_result_paths": [str(case["auxiliary_directory"]) for case in outcomes],
    }
    _atomic(auxiliary_root / "refinement-audit.json", _refinement_jsonable(document))
    return document


def _validate_resource_record(resources, config, expected_nc_hashes, auxiliary_root, *, phase=None,
                              allow_test_override=False):
    policy = _refinement_policy(config)
    if not isinstance(resources, Mapping) or not isinstance(resources.get("processes"), list):
        raise RunnerError("fidelity record lacks independent process/resource observations")
    expected_policy = dict(policy, total_posted_rss_ceiling_bytes=(
        policy["parallel_jobs"] * policy["max_rss_bytes_per_job"]))
    if resources.get("job_policy") != expected_policy:
        raise RunnerError("fidelity process policy differs from frozen memory/parallelism limits")
    _finite_seconds(resources.get("phase_wall_seconds"), "fidelity phase")
    if phase is None:
        required = {("verdict", "rs274"), ("verdict", "haas"), ("convergence", "rs274")}
    else:
        if phase not in {"screen", "verdict", "convergence"}:
            raise RunnerError(f"unknown posted resource phase: {phase}")
        required = {(phase, dialect.value) for dialect in
                    ((C.Dialect.RS274,) if phase == "convergence" else tuple(C.Dialect))}
    jobs = resources["processes"]
    observed = {(item.get("role"), item.get("dialect")) for item in jobs}
    if len(jobs) != len(required) or observed != required:
        raise RunnerError("fidelity resources do not cover every required role/dialect job")
    for item in jobs:
        role, dialect = item["role"], item["dialect"]
        if item.get("max_step_mm") != policy[f"{role}_step_mm"]:
            raise RunnerError("resource record's sampling step differs from the declared role")
        if item.get("expected_nc_hash") != expected_nc_hashes[dialect]:
            raise RunnerError("resource record does not bind the actual dialect NC bytes")
        state = item.get("status")
        if state == "test_override":
            if not allow_test_override:
                raise RunnerError("unmarked in-process override is not subprocess evidence")
            source = Path(item.get("source", ""))
            if not source.is_file() or C.sha256_bytes(source.read_bytes()) != item.get("evaluation_sha256"):
                raise RunnerError("test override result linkage does not match its source hash")
        elif state == "complete":
            _positive_int(item.get("worker_pid"), "posted worker PID")
            peak = _positive_int(item.get("worker_peak_rss_kib"), "posted worker peak RSS")
            if peak * 1024 > policy["max_rss_bytes_per_job"]:
                raise RunnerError("recorded posted-worker RSS exceeds its declared cap")
            evaluator_seconds = _finite_seconds(item.get("evaluator_wall_seconds"), "posted evaluation")
            process_seconds = _finite_seconds(item.get("process_wall_seconds"), "posted subprocess")
            if process_seconds + 1e-9 < evaluator_seconds:
                raise RunnerError("whole posted-process time is shorter than its evaluator call")
            path = Path(item.get("evaluation_path", "")).resolve()
            if not path.is_relative_to(auxiliary_root) or not path.is_file():
                raise RunnerError("posted evaluation path escapes its auxiliary root")
            if C.sha256_bytes(path.read_bytes()) != item.get("evaluation_sha256"):
                raise RunnerError("resource result hash differs from the actual posted evaluation")
        elif state == "reused_typed_evaluation":
            if item.get("worker_peak_rss_kib") is not None or item.get("worker_pid") is not None:
                raise RunnerError("reused typed evidence cannot claim a new measured worker/PID")
            if item.get("evaluator_wall_seconds") is not None or item.get("process_wall_seconds") is not None:
                raise RunnerError("reused typed evidence cannot supply fresh evaluator or process durations")
            source = Path(item.get("source", ""))
            if not source.is_file() or C.sha256_bytes(source.read_bytes()) != item.get("evaluation_sha256"):
                raise RunnerError("reused typed result linkage does not match its source hash")
        else:
            raise RunnerError(f"unknown posted worker status: {state}")


def _verified_refinement_for_report(root, run_id, auxiliary_root, config):
    """Verify schema-4 typed files, both fine verdicts, and convergence evidence."""
    directory, _, _, task, primary, _ = _load_refinement_case(root, run_id)
    auxiliary_id = f"{run_id}-verdict"
    auxiliary_dir = auxiliary_root / auxiliary_id
    screen = _read_phase(task, directory, config, "screen", evaluated_dir=auxiliary_dir,
                         screen_copy=True, primary=primary)
    verdict = _read_phase(task, directory, config, "verdict", evaluated_dir=auxiliary_dir, primary=primary)
    convergence = _read_phase(task, directory, config, "convergence", evaluated_dir=auxiliary_dir)[C.Dialect.RS274]
    screen_deltas, disagreement_fields = _screen_fidelity_diagnostics(screen, verdict, task)
    screen_delta = screen_deltas[C.Dialect.RS274.value]
    haas_delta = screen_deltas[C.Dialect.HAAS.value]
    convergence_delta = _fidelity_deltas(convergence, verdict[C.Dialect.RS274], task)
    history = _json(auxiliary_dir / "history.json")
    if not isinstance(history, Mapping) or history.get("stage") != "fine-verdict-and-convergence-posted-evaluations":
        raise RunnerError("verdict history does not describe the independent fine/convergence phase")
    resources = history.get("resources")
    nc_hashes = {dialect.value: C.nc_hash((directory / C.DIALECT_ARTIFACTS[dialect][0]).read_bytes())
                 for dialect in C.Dialect}
    _validate_resource_record(resources, config, nc_hashes, auxiliary_root)
    roles = {}
    for role, values in (("screen", screen), ("verdict", verdict), ("convergence", {C.Dialect.RS274: convergence})):
        roles[role] = {
            "max_step_mm": config["refinement_check"][f"{role}_step_mm"],
            "evaluations": {
                dialect.value: {
                    "evaluator_revision": evaluation.evaluator_revision,
                    "path": f"{auxiliary_id}/{'screen_' if role == 'screen' else ''}{C.DIALECT_ARTIFACTS[dialect][1] if role != 'convergence' else 'eval_convergence_rs274.json'}",
                    "sha256": C.sha256_bytes(R.record_bytes(evaluation)),
                } for dialect, evaluation in values.items()
            },
        }
    comparison_document = {
        "schema": "diffcam-fidelity-comparison/1", "primary_run_id": run_id,
        "verdict_run_id": auxiliary_id, "screen_to_verdict_rs274": screen_delta,
        "screen_to_verdict_haas": haas_delta,
        "screen_disagreement_fields": disagreement_fields,
        "screen_disagreement_fields_by_dialect": {
            dialect: comparison["changed_outcome_fields"] for dialect, comparison in screen_deltas.items()},
        "convergence_to_verdict_rs274": convergence_delta,
        "role": "diagnostics only; 0.125 mm posted evaluations in both dialects are the sole verdict source",
    }
    comparison_payload = R.json_bytes(comparison_document)
    if (auxiliary_dir / "comparison.json").read_bytes() != comparison_payload:
        raise RunnerError("fidelity comparison differs from the actual typed evaluations")
    expected_link = {
        "schema": "diffcam-refinement-link/4", "primary_run_id": run_id,
        "primary_run_path": str(directory.relative_to(root)),
        "primary_metrics_sha256": C.sha256_bytes((directory / "metrics.json").read_bytes()),
        "primary_rs274_evaluation_sha256": C.sha256_bytes((directory / "eval_posted_rs274.json").read_bytes()),
        "primary_haas_evaluation_sha256": C.sha256_bytes((directory / "eval_posted_haas.json").read_bytes()),
        "primary_run_json_sha256": C.sha256_bytes((directory / "run.json").read_bytes()),
        "verdict_run_id": auxiliary_id,
        "auxiliary_record": {"path": f"{auxiliary_id}/run.json",
                             "sha256": C.sha256_bytes((auxiliary_dir / "run.json").read_bytes())},
        "comparison_path": f"{auxiliary_id}/comparison.json",
        "comparison_sha256": C.sha256_bytes(comparison_payload),
        "screen_disagreement_fields": disagreement_fields, "roles": roles,
        "rule": "only 0.125 mm evaluations in both dialects determine pilot verdicts; screen is diagnostic",
        "resources": resources,
    }
    link_path = auxiliary_root / f"{run_id}-refinement-link.json"
    link = _json(link_path)
    if not isinstance(link, Mapping) or link.get("schema") != "diffcam-refinement-link/4":
        raise RunnerError(f"verdict links must use schema /4: {run_id}")
    if link != expected_link:
        raise RunnerError(f"fine verdict link does not match its actual typed evidence: {run_id}")
    return verdict, disagreement_fields, task, {
        "path": os.path.relpath(link_path, root),
        "sha256": C.sha256_bytes(link_path.read_bytes()),
    }, resources


def _selection_vs_verdict_record(root: Path, run_id: str, evaluated: Mapping[str, C.EvaluationResult],
                                 method: str) -> dict[str, Any]:
    """Compare the *same selected program's* stored 0.25 and 0.125 mm evidence.

    Nominal has no optimizer selection. Older fixture/legacy runs may also lack
    one; both remain explicit null, never a guessed feasibility pass. A present
    optimizer record is re-ranked from its immutable result/thresholds without
    calling the simulator; an internally inconsistent record fails closed.
    """
    metrics_path = root / run_id / "metrics.json"
    config_path = root / run_id / "config.json"
    metrics = _json(metrics_path)
    selection = metrics.get("selection") if isinstance(metrics, Mapping) else None
    reason = None
    candidate_id = theta_hash = None
    selection_config_hash = None
    feasible = None
    if not isinstance(selection, Mapping):
        reason = ("nominal-direct-hard-has-no-optimizer-selection-record" if method == "nominal-cam"
                  else "legacy-run-no-optimizer-selection-record")
    else:
        config_doc = _json(config_path)
        selection_config = config_doc.get("hard_selection")
        try:
            if not isinstance(selection_config, Mapping):
                raise ValueError("run config has no hard selection policy")
            thresholds = selection_config.get("resolved")
            if thresholds != selection.get("hard_selection") or not isinstance(thresholds, Mapping):
                raise ValueError("config/metrics selection thresholds differ")
            policy = B.HardSelectionPolicy(**selection_config["policy"])
            limits = B._restore_hard_selection({"selection_thresholds": thresholds}, policy)
            candidate = C.CandidateResult(**metrics["best_candidate"])
            rank = B._hard_rank(candidate, limits)
            if rank is None:
                raise ValueError("the selected candidate is not hard-rankable")
            if not candidate.candidate_id.startswith(run_id + ":"):
                raise ValueError("selected candidate belongs to another run identity")
            feasible = rank[0] == 0
            selection_config_hash = thresholds["config_hash"]
            if selection_config_hash != config_doc.get("hard_selection_config_hash"):
                raise ValueError("selection config hash differs")
            candidate_id, theta_hash = candidate.candidate_id, candidate.theta_hash
            if not isinstance(selection.get("no_improvement_over_initialization"), bool):
                raise ValueError("missing boolean no-improvement record")
        except (KeyError, TypeError, ValueError) as error:
            raise RunnerError(f"invalid optimizer selection evidence for {run_id}: {error}") from error
    by_dialect = {}
    for dialect in C.Dialect:
        evaluation = evaluated[dialect]
        if evaluation.discretization["settings"]["max_step_mm"] != 0.125:
            raise RunnerError("selection comparison requires 0.125 mm typed verdicts")
        agree = None if feasible is None or evaluation.geometry_pass is None else (
            feasible == evaluation.geometry_pass)
        by_dialect[dialect.value] = {
            "geometry_pass_at_0125": evaluation.geometry_pass,
            "agree": agree,
            "nc_hash": evaluation.nc_hash,
            "verdict_step_mm": 0.125,
        }
    return {
        "schema": "diffcam-selection-vs-verdict/1",
        "run_id": run_id,
        "feasible_at_025": feasible,
        "reason": reason,
        "selected_candidate_id": candidate_id,
        "selected_candidate_theta_hash": theta_hash,
        "selection_config_hash": selection_config_hash,
        "no_improvement_over_initialization": (
            None if not isinstance(selection, Mapping)
            else selection["no_improvement_over_initialization"]),
        "selection_source": {
            "step_mm": 0.25,
            "meaning": "stored optimizer hard-sim selection evidence, not a posted verdict recomputation",
            "metrics_path": f"{run_id}/metrics.json",
            "metrics_sha256": C.sha256_bytes(metrics_path.read_bytes()),
            "config_path": f"{run_id}/config.json",
            "config_sha256": C.sha256_bytes(config_path.read_bytes()) if reason is None else None,
        },
        "by_dialect": by_dialect,
    }


def _selection_verdict_summary(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    null_rows = sorted(run_id for run_id, record in records.items()
                       if record["feasible_at_025"] is None)
    summary = {
        "total_runs": len(records),
        "null_selection_runs_excluded_from_denominator": null_rows,
        "null_selection_count": len(null_rows),
        "known_selection_count": len(records) - len(null_rows),
        "by_dialect": {},
    }
    for dialect in C.Dialect:
        comparable = [run_id for run_id, record in records.items()
                      if record["feasible_at_025"] is not None
                      and record["by_dialect"][dialect.value]["geometry_pass_at_0125"] is not None]
        agreement = [run_id for run_id in comparable
                     if records[run_id]["by_dialect"][dialect.value]["agree"]]
        unknown_verdict = sorted(run_id for run_id, record in records.items()
                                 if record["feasible_at_025"] is not None
                                 and record["by_dialect"][dialect.value]["geometry_pass_at_0125"] is None)
        summary["by_dialect"][dialect.value] = {
            "comparable_run_count": len(comparable),
            "agreement_count": len(agreement),
            "disagreement_count": len(comparable) - len(agreement),
            "disagreement_run_ids": sorted(set(comparable) - set(agreement)),
            "unknown_verdict_run_ids": unknown_verdict,
            "unknown_verdict_count": len(unknown_verdict),
        }
    return summary


def _pilot_report(config, root, auxiliary_root, *, bootstrap_replicates=2000):
    """Publish pilot verdicts only from fine posted evaluations in both dialects."""
    root = Path(root).resolve()
    precision = _requested_precision(config, pilot=config.get("task_source", "pilot_pocket") == "pilot_pocket")
    try:
        manifest = audit.load_manifest(root / "manifest.json")
    except (OSError, ValueError) as error:
        raise RunnerError(f"cannot load primary pilot manifest: {error}") from error
    if not _refinement_policy(config)["enabled"]:
        if config.get("task_source", "pilot_pocket") == "pilot_pocket":
            raise RunnerError("pilot verdicts require enabled fine refinement; never substitute the primary screen")
        report, _ = audit.generate_report(
            root, manifest, bootstrap_replicates=bootstrap_replicates,
            required_precision=precision)
        return report
    if auxiliary_root is None:
        raise RunnerError("fine verdict evaluation requires an auxiliary root")
    _, auxiliary_root = _safe_refinement_directory(root, Path(auxiliary_root))
    report_path = auxiliary_root / "refinement-audit.json"
    auxiliary_report = _json(report_path)
    try:
        auxiliary_manifest = audit.load_manifest(auxiliary_root / "manifest.json")
    except (OSError, ValueError) as error:
        raise RunnerError(f"cannot load auxiliary verdict manifest: {error}") from error
    _, auxiliary_audit = audit.audit_runs(auxiliary_root, auxiliary_manifest)
    if not (auxiliary_audit["summary"]["auditor_passed"]
            and auxiliary_audit["summary"]["all_runs_completed"]):
        raise RunnerError("fine/screen/convergence auxiliary audit is incomplete or invalid")
    reported = auxiliary_report.get("reported_outcomes") if isinstance(auxiliary_report, Mapping) else None
    if auxiliary_report.get("schema") != "diffcam-fidelity-audit/4" or not isinstance(reported, Mapping):
        raise RunnerError("auxiliary report must contain schema-4 verdicts for both dialects")
    primary_ids = set(reported)
    required_aux_ids = primary_ids | {f"{run_id}-verdict" for run_id in primary_ids}
    if (set(auxiliary_manifest.runs) != required_aux_ids
            or auxiliary_report.get("case_count") != len(primary_ids)
            or auxiliary_report.get("baseline_and_refined_records") != 2 * len(primary_ids)
            or auxiliary_report.get("manifest") != auxiliary_audit["summary"]):
        raise RunnerError("auxiliary verdict report does not reconcile with its actual audit")
    missing = sorted(set(manifest.runs) - primary_ids)
    if missing:
        raise RunnerError(f"primary runs have no fine verdict link/report: {missing}")
    verdicts, diagnostics, links = {}, {}, {}
    resources = {}
    selection_records = {}
    report_tasks = dict(manifest.tasks)
    completion_sources = {}
    for run_id in sorted(primary_ids):
        source_root = root if run_id in manifest.runs else auxiliary_root
        expected = manifest.runs.get(run_id, auxiliary_manifest.runs[run_id])
        evaluated, fields, task, link, resource = _verified_refinement_for_report(
            source_root, run_id, auxiliary_root, config)
        expected_reported = {
            "condition": expected.method, "seed": expected.seed,
            "outcomes": {dialect: _evaluation_outcome(evaluation, task)
                         for dialect, evaluation in evaluated.items()},
            "screen_disagreement_fields": fields,
            "verdict_step_mm": 0.125, "screen_step_mm": 1.0,
            "convergence_step_mm": 0.25,
            "verdict_evaluation_paths": {
                dialect: f"{run_id}-verdict/{C.DIALECT_ARTIFACTS[C.Dialect(dialect)][1]}"
                for dialect in evaluated},
            "convergence_evaluation_path": f"{run_id}-verdict/eval_convergence_rs274.json",
            "auxiliary_record_path": f"{run_id}-verdict/run.json",
            "auxiliary_link_path": f"{run_id}-refinement-link.json",
        }
        if reported[run_id] != expected_reported:
            raise RunnerError(f"auxiliary reported verdict is not backed by typed evidence: {run_id}")
        if run_id not in manifest.runs:
            continue
        task_info = report_tasks[expected.task_id]
        if task_info.task_hash != task.task_hash:
            raise RunnerError(f"primary/fine-verdict task identity mismatch: {run_id}")
        if task_info.completion_residual_mm3 is not None and task_info.completion_residual_mm3 != task.tolerances.completion_residual_mm3:
            raise RunnerError(f"primary manifest completion threshold differs from the typed task: {run_id}")
        report_tasks[expected.task_id] = dataclasses.replace(
            task_info, completion_residual_mm3=task.tolerances.completion_residual_mm3)
        completion_sources[expected.task_id] = "TaskSpec.tolerances.completion_residual_mm3"
        verdicts[run_id] = evaluated
        diagnostics[run_id] = fields
        links[run_id] = link
        resources[run_id] = resource
        selection_records[run_id] = _selection_vs_verdict_record(
            root, run_id, evaluated, expected.method)
    if set(verdicts) != set(manifest.runs):
        raise RunnerError("fine typed verdicts did not cover every expected primary run")
    report_manifest = dataclasses.replace(manifest, tasks=report_tasks)
    report, _ = audit.generate_report(
        root, report_manifest, bootstrap_replicates=bootstrap_replicates,
        verdict_evaluations=verdicts, screen_disagreements=diagnostics,
        required_precision=precision)
    report["refinement"]["auxiliary_audit_sha256"] = C.sha256_bytes(report_path.read_bytes())
    report["refinement"]["links_by_run_id"] = links
    report["refinement"]["resources_by_run_id"] = resources
    report["refinement"]["completion_thresholds_mm3"] = {
        task_id: task_info.completion_residual_mm3 for task_id, task_info in report_tasks.items()}
    report["refinement"]["completion_threshold_sources"] = completion_sources
    for row in report["runs"]:
        if row["run_id"] not in selection_records:
            raise RunnerError(f"missing selection comparison for primary run: {row['run_id']}")
        row["selection_vs_verdict"] = selection_records[row["run_id"]]
    report["refinement"]["selection_vs_verdict_by_run_id"] = selection_records
    report["refinement"]["selection_vs_verdict_summary"] = _selection_verdict_summary(selection_records)
    return report


def _refine_process(config_path, source_root, run_id, auxiliary_root, timeout):
    """Coordinate one case while posted children each own their PID and memory."""
    config = _normalized_config(config_path)
    config["experiment"] = "pilot"
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or timeout != _refinement_policy(config)["timeout_s"]):
        raise RunnerError("program-level timeout may only be the frozen posted-job timeout")
    started = time.perf_counter()
    result = _refinement_worker(config, source_root, run_id, auxiliary_root)
    result.update(elapsed_seconds=time.perf_counter() - started,
                  log=str(result["auxiliary_directory"] / "history.json"))
    return result


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
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
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
            item["source_config"] = config
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
        "schema": "diffcam-posted-fidelity/4", "policy": policy,
        "required_primary_programs": len(work), "scheduled_worker_count": len(executable),
        "program_level_parallel_jobs": 1,
        "posted_job_parallel_jobs": policy["parallel_jobs"],
        "status_counts": dict(Counter(item["status"] for item in [*completed, *remaining])),
        "results": reporting_items, "audit": audited,
        "auxiliary_root": str(aux_root),
        "process_policy": "one program coordinator; at most two independent posted children, exact IDs, timeout and peak-RSS limits",
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
            _validate_production_timing_evidence(config, timing, task)
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
    parser.add_argument("--posted-job-dir", help=argparse.SUPPRESS)
    parser.add_argument("--aux-root", help="sibling audit root for refinement evidence")
    parser.add_argument("--primary-root", action="append", help="additional completed primary root")
    args = parser.parse_args(argv)
    root = Path(args.output_dir).resolve()
    if (not args.resume and not args.worker_run and not args.timing_worker_run
            and not args.refinement_check and not args.refinement_worker_run
            and not args.posted_job_dir and root.exists()):
        parser.error("output-dir must be new unless --resume is set")
    try:
        if args.posted_job_dir:
            if (args.split != "dev" or args.experiment != "pilot" or args.dry_run
                    or args.worker_run or args.timing_worker_run
                    or args.refinement_check or args.refinement_worker_run or args.primary_root):
                raise RunnerError("posted-job-dir accepts only its own single posted-job protocol")
            return _posted_job_main(root)
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
    except (RunnerError, B.BlackBoxError, audit.AuditInputError) as error:
        print(f"benchmark runner failed: {error}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired as error:
        raise RunnerError(f"subprocess cleanup failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
