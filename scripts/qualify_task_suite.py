"""Stand-alone, resumable 0.125 mm dual-dialect nominal qualification.

Each RS274/Haas pair uses the production runner's bounded child-job protocol.
A second, pair-level executor bounds live children: ``ceil(parallelism / 2)``
pairs times at most two children per pair, hence no more than eight processes.
No solver, optimizer, physical CNC motion or seven-run pilot is launched.
"""
from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import MappingProxyType

from benchmarks import contracts as C
from benchmarks.pilot_task import production_settings
from cam.seeds import seed_for_task
from cam.posted_evaluator import _revision as evaluator_revision
from cam.posts import post_motion_plan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_VERSION = "t2.3c-post-qualification/1"
MEMORY_LIMIT_BYTES = 8 * 1024**3
ROLE = "verdict"
FAMILIES = ("facing_asymmetric", "pocket_island", "variable_width_depth")
FULL_COUNTS = {"dev": 3, "test": 10}


class QualificationError(RuntimeError):
    pass


_RUNNER_MODULE = None


def _runner():
    """Load the runner without executing its top-level campaign entry point."""
    global _RUNNER_MODULE
    name = "t23c_production_runner"
    path = PROJECT_ROOT / "scripts" / "run_benchmark_suite.py"
    if _RUNNER_MODULE is not None and Path(_RUNNER_MODULE.__file__).resolve() == path.resolve():
        return _RUNNER_MODULE
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise QualificationError("cannot load the production runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _RUNNER_MODULE = module
    return module


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _config(*, parallel, timeout_s, memory_limit, poll_interval, task):
    if isinstance(parallel, bool) or not isinstance(parallel, int) or not 1 <= parallel <= 8:
        raise QualificationError("parallelism must be an integer in [1,8]")
    for name, value in (("timeout_s", timeout_s), ("poll_interval_s", poll_interval)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise QualificationError(f"{name} must be finite and positive")
    if (isinstance(memory_limit, bool) or not isinstance(memory_limit, int)
            or memory_limit <= 0 or memory_limit > MEMORY_LIMIT_BYTES):
        raise QualificationError("per-job memory limit must be in (0, 8 GiB]")
    base = dict(production_settings(task))
    base.update(subdivisions=None, compute_bounds=True, safety_screen=True,
                max_samples_per_segment=4096, record_stations=False)
    return {
        "task_source": "pilot_pocket",
        "exclusion_tolerance_mm3": 0.0,
        "require_grid_exclusion_tolerance": True,
        "posted_refinement": base,
        "refinement_check": {
            "enabled": True,
            "verdict_step_mm": 0.125,
            "screen_step_mm": 1.0,
            "convergence_step_mm": 0.25,
            # This is only the existing helper's allowed group size. The suite
            # separately caps parallel groups at ceil(requested_parallelism/2).
            "parallel_jobs": min(parallel, 2),
            "max_rss_bytes_per_job": memory_limit,
            "poll_interval_s": poll_interval,
            "timeout_s": timeout_s,
        },
    }


def _task_from_entry(entry):
    record = entry.get("task_record")
    if not isinstance(record, dict):
        raise QualificationError("task entry lacks typed task_record")
    task = C.loads(json.dumps(record))
    if not isinstance(task, C.TaskSpec) or task.task_id != entry["task_id"]:
        raise QualificationError("manifest task identity is inconsistent")
    if task.task_hash != entry["task_hash"]:
        raise QualificationError("manifest task hash does not match TaskSpec")
    motion = seed_for_task(task)
    if motion.motion_hash != entry["motion_hash"]:
        raise QualificationError("nominal seed does not reproduce the manifest motion hash")
    motion.validate_for_task(task)
    return task, motion


def load_cohort(manifest, *, require_full=True):
    path = Path(manifest).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "t2.3-procedural/2":
        raise QualificationError("expected t2.3-procedural/2 manifest")
    if document.get("generation_version") != "t2.3/2":
        raise QualificationError("task cohort generator version is not t2.3/2")
    entries = document.get("tasks")
    if not isinstance(entries, list) or not entries:
        raise QualificationError("task manifest contains no tasks")
    ids = []
    cohort = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise QualificationError("task entry is not an object")
        task, motion = _task_from_entry(entry)
        if task.task_id in ids:
            raise QualificationError(f"duplicate task_id {task.task_id}")
        if task.task_id not in document.get("task_details", {}):
            raise QualificationError(f"missing task_details for {task.task_id}")
        detail = document["task_details"][task.task_id]
        if detail.get("task_hash") != task.task_hash or detail.get("split") != task.split.value:
            raise QualificationError(f"stale task detail identity for {task.task_id}")
        if detail.get("design_seed") is None:
            raise QualificationError(f"missing design seed for {task.task_id}")
        ids.append(task.task_id)
        cohort.append((task, motion, entry, detail))
    if require_full:
        expected = {(family, split): count for family in FAMILIES
                    for split, count in FULL_COUNTS.items()}
        actual = {}
        for task, _motion, _entry, _detail in cohort:
            actual[(task.family, task.split.value)] = actual.get((task.family, task.split.value), 0) + 1
        if actual != expected or len(cohort) != 39:
            raise QualificationError("full cohort must contain 3 dev and 10 test per family (39 tasks)")
    return path, document, cohort


def _nc_artifacts(task, motion, dialect, workspace, config):
    from cam.gcode_parser import parse_motion_plan
    from benchmarks.pilot_task import compare_motion

    text = post_motion_plan(motion, task, dialect=dialect).encode("utf-8")
    comparison = compare_motion(motion, parse_motion_plan(text.decode("utf-8"), task, dialect=dialect))
    if comparison["semantic_tag_mismatches"] != 0:
        raise QualificationError(f"posted motion semantics differ for {task.task_id}/{dialect.value}")
    settings = MappingProxyType(_runner()._phase_settings(task, config, dialect, text, ROLE))
    return _runner().PostedJob(ROLE, dialect, text, settings, workspace), comparison


def _attempt_paths(workspace):
    root = Path(workspace)
    if not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        if path.is_symlink() and path.name.startswith("attempt-"):
            raise QualificationError(f"attempt directories may not be symlinks: {path}")
        if path.is_dir() and path.name.startswith("attempt-"):
            try:
                number = int(path.name.removeprefix("attempt-"))
            except ValueError:
                continue
            if number > 0:
                found.append((number, path))
    return sorted(found)


def _require_finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 or not math.isfinite(value):
        raise QualificationError(f"invalid prior {name}: {value!r}")


def _write_process_resources(task, job, metadata):
    """Keep controller-only augmentation out of primary child/evaluator files."""
    record_path = job.workspace / "worker-result.json"
    _write_new(job.workspace / "suite-process.json", {
        "schema_version": "t2.3c-process-resource/1",
        "worker_pid": metadata["worker_pid"],
        "role": job.role,
        "dialect": job.dialect.value,
        "task_hash": task.task_hash,
        "nc_hash": C.nc_hash(job.nc_bytes),
        "settings": dict(job.settings),
        "settings": dict(job.settings),
        "wall_seconds": metadata["wall_seconds"],
        "elapsed_seconds": metadata.get("elapsed_seconds", metadata.get("process_wall_seconds")),
        "worker_self_reported_peak_rss_kib": metadata["worker_self_reported_peak_rss_kib"],
        "parent_observed_hwm_kib": metadata["parent_polled_child_vmhwm_kib"],
        "combined_peak_rss_kib": metadata["worker_peak_rss_kib"],
        "evaluation_sha256": metadata["evaluation_sha256"],
        "worker_result_sha256": C.sha256_bytes(record_path.read_bytes()),
    })


def _controller_metadata(workspace, child):
    path = workspace / "suite-process.json"
    if not path.is_file():
        # A raw child record is valid historical evidence, but it has no fresh
        # parent-observed elapsed time or peak to attribute to today's reuse.
        return dict(child, worker_self_reported_peak_rss_kib=child["worker_peak_rss_kib"])
    controller = json.loads(path.read_text())
    expected = {
        "schema_version": "t2.3c-process-resource/1",
        "worker_pid": child["worker_pid"],
        "role": child["role"],
        "dialect": child["dialect"],
        "task_hash": child["task_hash"],
        "nc_hash": child["nc_hash"],
        "settings": child["settings"],
        "wall_seconds": child["wall_seconds"],
        "evaluation_sha256": child["evaluation_sha256"],
        "worker_result_sha256": C.sha256_bytes((workspace / "worker-result.json").read_bytes()),
        "worker_self_reported_peak_rss_kib": child["worker_peak_rss_kib"],
    }
    if any(controller.get(key) != value for key, value in expected.items()):
        raise QualificationError("controller sidecar does not match immutable child evidence")
    _require_finite_positive(controller.get("elapsed_seconds"), "controller elapsed seconds")
    parent = controller.get("parent_observed_hwm_kib")
    if parent is not None and (isinstance(parent, bool) or not isinstance(parent, int) or parent < 0):
        raise QualificationError("invalid controller-observed child VmHWM")
    expected_peak = max(child["worker_peak_rss_kib"], parent or 0)
    if controller.get("combined_peak_rss_kib") != expected_peak:
        raise QualificationError("controller combined peak is not the maximum child VmHWM sample")
    return dict(child, elapsed_seconds=controller["elapsed_seconds"],
                process_wall_seconds=controller["elapsed_seconds"],
                parent_polled_child_vmhwm_kib=parent,
                worker_self_reported_peak_rss_kib=child["worker_peak_rss_kib"],
                worker_peak_rss_kib=expected_peak)


def _check_worker_log(workspace, metadata):
    """The retained child stdout is the independent PID/result declaration."""
    declarations = []
    paths = []
    for path in sorted(workspace.glob("worker.*.log")):
        if path.is_symlink() or not path.is_file():
            raise QualificationError(f"invalid worker log path: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("POST_JOB_RESULT="):
                declarations.append(json.loads(line.removeprefix("POST_JOB_RESULT=")))
                paths.append(path)
    if declarations != [metadata]:
        raise QualificationError("prior worker file and child stdout declaration differ")
    return paths[0]



def _read_old_result(workspace, task, motion, job, *, memory_limit=MEMORY_LIMIT_BYTES):
    """Return complete reuse evidence, preserve failure, or None for partial work."""
    packet_path = workspace / "suite-failure.json"
    if packet_path.is_file():
        evidence = json.loads(packet_path.read_text())
        expected = {
            "schema_version": "t2.3c-failure/1", "status": "failed",
            "task_hash": task.task_hash, "motion_hash": motion.motion_hash,
            "dialect": job.dialect.value, "role": job.role,
            "nc_hash": C.nc_hash(job.nc_bytes), "settings": dict(job.settings),
        }
        if any(evidence.get(key) != value for key, value in expected.items()):
            raise QualificationError(f"prior failure identity differs at {packet_path}")
        if not isinstance(evidence.get("reason"), str) or not evidence["reason"]:
            raise QualificationError(f"prior failure has no recorded reason: {packet_path}")
        return {"kind": "previous_failure", "evidence": evidence}
    has_packet = (workspace / "job.json").is_file()
    has_eval = (workspace / "evaluation.json").is_file()
    has_record = (workspace / "worker-result.json").is_file()
    if not has_packet and not has_eval and not has_record:
        return None
    if not has_packet and (has_eval or has_record):
        raise QualificationError(f"orphaned prior output in {workspace}")
    if not (has_eval and has_record):
        packet = json.loads((workspace / "job.json").read_text())
        expected = {
            "schema": "diffcam-post-job/1", "role": job.role,
            "dialect": job.dialect.value, "task": task.to_dict(),
            "task_hash": task.task_hash, "nc_hash": C.nc_hash(job.nc_bytes),
            "settings": dict(job.settings),
        }
        if packet != expected:
            raise QualificationError(f"prior posted-job declaration differs in {workspace}")
        nc_name = C.DIALECT_ARTIFACTS[job.dialect][0]
        if (workspace / nc_name).read_bytes() != job.nc_bytes:
            raise QualificationError(f"prior NC bytes differ in {workspace}")
        return None
    packet = json.loads((workspace / "job.json").read_text())
    expected = {
        "schema": "diffcam-post-job/1",
        "role": job.role,
        "dialect": job.dialect.value,
        "task": task.to_dict(),
        "task_hash": task.task_hash,
        "nc_hash": C.nc_hash(job.nc_bytes),
        "settings": dict(job.settings),
    }
    if packet != expected:
        raise QualificationError(f"prior posted-job declaration differs in {workspace}")
    nc_name = C.DIALECT_ARTIFACTS[job.dialect][0]
    if (workspace / nc_name).read_bytes() != job.nc_bytes:
        raise QualificationError(f"prior NC bytes differ in {workspace}")
    source = workspace / "evaluation.json"
    source_bytes = source.read_bytes()
    result = C.loads(source_bytes.decode("utf-8"))
    _runner()._validate_posted_result(task, job.nc_bytes, job.dialect, job.settings, result)
    from algorithms import run_artifacts as R
    if R.record_bytes(result) != source_bytes:
        raise QualificationError(f"prior evaluation is not canonical production evidence: {source}")
    metadata = json.loads((workspace / "worker-result.json").read_text())
    if metadata.get("schema") != "diffcam-post-job-result/1":
        raise QualificationError("unrecognized prior worker-result schema")
    if (metadata.get("status") != "complete"
            or metadata.get("role") != job.role
            or metadata.get("dialect") != job.dialect.value
            or metadata.get("task_hash") != task.task_hash
            or metadata.get("nc_hash") != C.nc_hash(job.nc_bytes)
            or metadata.get("settings") != dict(job.settings)):
        raise QualificationError("prior worker metadata differs from requested identity")
    if metadata.get("evaluation_sha256") != C.sha256_bytes(source_bytes):
        raise QualificationError("prior worker/evaluation SHA-256 differs")
    _require_finite_positive(metadata.get("worker_peak_rss_kib"), "worker peak KiB")
    _require_finite_positive(metadata.get("wall_seconds"), "worker wall time")
    retained_log = _check_worker_log(workspace, metadata)
    metadata = _controller_metadata(workspace, metadata)
    metadata["source"] = str(source)
    metadata["log"] = str(retained_log)
    if metadata["worker_peak_rss_kib"] * 1024 > memory_limit:
        raise QualificationError("prior worker exceeded the declared per-job memory limit")
    if (isinstance(metadata.get("worker_pid"), bool)
            or not isinstance(metadata.get("worker_pid"), int) or metadata["worker_pid"] <= 0):
        raise QualificationError("prior worker PID is missing")
    if not str(metadata.get("worker_peak_rss_source", "")).startswith("VmHWM of /proc/self/status"):
        raise QualificationError("prior worker memory source is not own-process VmHWM")
    if result.evaluator_revision != evaluator_revision():
        raise QualificationError("prior evaluator revision differs from current production")
    if result.coverage != C.Coverage.COMPLETE:
        return {"kind": "previous_not_qualified",
                "reason": f"coverage={result.coverage.value}: {list(result.coverage_reasons)}",
                "evaluation": result, "worker": metadata}
    return {"kind": "reuse", "evaluation": result, "worker": metadata}


def make_job(task, motion, dialect, workspace, config):
    """Reserve a new immutable attempt dir, or return verified historical output."""
    workspace = Path(workspace)
    job, comparison = _nc_artifacts(task, motion, dialect, workspace, config)
    attempts = _attempt_paths(workspace.parent)
    for _number, candidate in reversed(attempts):
        try:
            previous = _read_old_result(candidate, task, motion, job,
                                        memory_limit=config["refinement_check"]["max_rss_bytes_per_job"])
        except (ValueError, UnicodeError, KeyError, TypeError, QualificationError) as error:
            return {
                "kind": "evidence_failure",
                "reason": f"cannot validate prior result at {candidate}: {error}",
                "prior_artifact": str(candidate),
            }
        if previous is not None:
            return previous
    # No prior attempt is needed for a fresh workspace; partial work gets a new
    # immutable attempt subdirectory rather than deletion or in-place repair.
    last = attempts[-1][0] if attempts else 0
    job = _runner().PostedJob(job.role, job.dialect, job.nc_bytes, job.settings,
                              workspace.parent / f"attempt-{last + 1:04d}")
    return {"kind": "new_job", "job": job, "motion_comparison": comparison}


def _failure_summary(task, motion, job, error):
    return {
        "schema_version": "t2.3c-failure/1",
        "status": "failed",
        "task_hash": task.task_hash,
        "motion_hash": motion.motion_hash,
        "dialect": job.dialect.value,
        "role": job.role,
        "task_id": task.task_id,
        "error_type": type(error).__name__,
        "reason": str(error),
        "traceback": traceback.format_exc(),
        "created_utc": _utc_now(),
        "nc_hash": C.nc_hash(job.nc_bytes),
        "settings": dict(job.settings),
        "worker_pid": None,
    }


def _result_summary(result, metadata, *, reused=False):
    metric = result.metrics.get("air_fraction")
    return {
        "schema_version": "t2.3c-dialect-result/1",
        "status": "evaluated",
        "qualified": bool(result.efficiency_eligible),
        "coverage": result.coverage.value,
        "coverage_reasons": list(result.coverage_reasons),
        "safety_pass": result.safety_pass,
        "geometry_pass": result.geometry_pass,
        "efficiency_eligible": bool(result.efficiency_eligible),
        "air_fraction": None if metric is None else metric.value,
        "worker_pid": None if reused else metadata.get("worker_pid"),
        "worker_own_hwm_bytes": None if reused else metadata["worker_peak_rss_kib"] * 1024,
        "original_worker_pid": metadata.get("worker_pid"),
        "original_worker_own_hwm_bytes": metadata["worker_peak_rss_kib"] * 1024,
        "original_worker_self_reported_hwm_bytes": metadata["worker_self_reported_peak_rss_kib"] * 1024,
        "original_parent_observed_hwm_kib": metadata.get("parent_polled_child_vmhwm_kib"),
        "worker_wall_seconds": metadata.get("wall_seconds"),
        "process_wall_seconds": metadata.get("process_wall_seconds") or metadata.get("elapsed_seconds"),
        "reused": reused,
        "evaluation_path": str(metadata.get("source") or metadata.get("evaluation_path")),
        "evaluation_sha256": metadata["evaluation_sha256"],
        "worker_log": metadata.get("log"),
        "parsed_motion_hash": result.discretization.get("parsed_motion_hash"),
        "settings": result.discretization.get("settings"),
    }


def _execute_new_jobs(task, motion, prepared, config):
    """Execute pairs with unique process identities and preserve outcomes."""
    jobs = [item["job"] for item in prepared if item["kind"] == "new_job"]
    contexts = {item["job"].dialect.value: item for item in prepared if item["kind"] == "new_job"}
    if len(contexts) != len(jobs):
        raise QualificationError("duplicate dialect in a new posted pair")
    if not jobs:
        return {}, {"job_policy": {"parallel_jobs": 0}, "phase_wall_seconds": 0.0}
    try:
        phase = _runner()._post_jobs(task, jobs, config)
        for job, (result, metadata), process_metadata in zip(
                jobs, phase["results"], phase["processes"], strict=True):
            if not isinstance(result, C.EvaluationResult) or result.dialect != job.dialect:
                raise QualificationError(
                    f"worker result/dialect mismatch for {job.dialect.value}: "
                    f"result={getattr(result, 'dialect', None)!r}; job={job.dialect!r}; "
                    f"typed_result={isinstance(result, C.EvaluationResult)}")
            if (metadata["dialect"] != job.dialect.value
                    or process_metadata["dialect"] != metadata["dialect"]
                    or metadata["worker_pid"] != process_metadata["worker_pid"]):
                raise QualificationError(f"worker/process identities differ for {job.dialect.value}")
            _runner()._validate_posted_result(task, job.nc_bytes, job.dialect, job.settings, result)
            _write_process_resources(task, job, metadata)
            contexts[job.dialect.value]["result"] = result
            contexts[job.dialect.value]["metadata"] = metadata
    except Exception as error:
        # One dialect can finish after another has failed. Preserve a valid
        # completed evaluation and never delete it or relabel it as qualified.
        raise QualificationError(f"posted pair failure: {error}") from error
    # C.EvaluationResult is not JSON serializable. The immutable typed results
    # remain in each attempt; the report retains only child-process projections.
    return contexts, {key: value for key, value in phase.items() if key != "results"}


def _fallback_failing_jobs(task, motion, jobs, error, config):
    """Recover per-worker evidence without rewriting an earlier failure."""
    grouped = {}
    for job in jobs:
        dialect = job.dialect.value
        try:
            old = _read_old_result(job.workspace, task, motion, job,
                                   memory_limit=config["refinement_check"]["max_rss_bytes_per_job"])
        except Exception as old_error:
            if isinstance(old_error, FileNotFoundError):
                old = None
            else:
                grouped[dialect] = {"status": "failed", "failure_class": "evidence_failure",
                                    "reason": str(old_error), "error_type": type(old_error).__name__,
                                    "prior_artifact": str(job.workspace), "reused": False}
                _write_new(job.workspace / "suite-failure.json",
                           _failure_summary(task, motion, job, old_error))
                continue
        if old is None:
            failure = _failure_summary(task, motion, job, error)
            try:
                _write_new(job.workspace / "suite-failure.json", failure)
            except FileExistsError:
                pass  # Preserve an earlier recorded failure rather than replacing it.
            grouped[dialect] = {
                "status": "failed", "failure_class": "worker_failure",
                "reason": str(error), "error_type": type(error).__name__,
                "prior_artifact": str(job.workspace / "suite-failure.json"), "reused": False,
            }
            continue
        if old["kind"] == "previous_failure":
            grouped[dialect] = {"status": "failed", "failure_class": "previous_failure",
                                "reason": old["evidence"]["reason"],
                                "prior_artifact": str(job.workspace / "suite-failure.json"), "reused": True}
        else:
            # These were new attempts, not historical reuse. One failed sibling
            # must not erase a completed sibling's fresh PID/timing/RSS evidence.
            grouped[dialect] = _result_summary(old["evaluation"], old["worker"], reused=False)
    return grouped


def execute_dialect_job(task, motion, job, config):
    """Execute one prepared PostedJob through the production bounded helper."""
    try:
        result, metadata = _runner()._posted_job_process(task, job, config)
        if not isinstance(result, C.EvaluationResult):
            raise QualificationError("child did not return a typed EvaluationResult")
        _write_process_resources(task, job, metadata)
    except Exception as error:
        failure = _failure_summary(task, motion, job, error)
        _write_new(job.workspace / "suite-failure.json", failure)
        return {"status": "failed", "reason": str(error),
                "failure_artifact": str(job.workspace / "suite-failure.json"),
                "error_type": failure["error_type"],
                "worker_pid": None, "reused": False}
    return _result_summary(result, metadata)


def execute_task(task, motion, workspace, config):
    """Evaluate both dialects, preserving already recorded failure evidence."""
    workspace = Path(workspace)
    grouped = {}
    pending = []
    try:
        for dialect in C.Dialect:
            outcome = make_job(task, motion, dialect, workspace / dialect.value / "attempt-0001", config)
            if outcome["kind"] == "new_job":
                pending.append(outcome)
            elif outcome["kind"] in ("reuse", "previous_not_qualified"):
                grouped[dialect.value] = _result_summary(
                    outcome["evaluation"], outcome["worker"], reused=True)
            else:
                grouped[dialect.value] = {
                    "status": "failed", "failure_class": outcome["kind"],
                    "reason": outcome.get("reason") or outcome.get("evidence", {}).get("reason", "recorded failure"),
                    "prior_artifact": outcome.get("prior_artifact"), "reused": True,
                }
    except Exception as error:
        for dialect in C.Dialect:
            if dialect.value in grouped:
                continue
            grouped[dialect.value] = {"status": "failed", "reason": str(error),
                                      "error_type": type(error).__name__, "reused": False}
        return {"task_id": task.task_id, "task_hash": task.task_hash,
                "motion_hash": motion.motion_hash, "dialects": grouped,
                "efficiency_eligible": False}
    try:
        new_results, phase = _execute_new_jobs(task, motion, pending, config)
    except QualificationError as error:
        phase = {"failure_class": "worker_failure", "reason": str(error)}
        new_results = _fallback_failing_jobs(
            task, motion, [item["job"] for item in pending], error, config)
    for dialect, outcome in new_results.items():
        if "result" in outcome:
            summary = _result_summary(outcome["result"], outcome["metadata"])
            if "motion_comparison" in outcome:
                summary["motion_comparison"] = outcome["motion_comparison"]
            grouped[dialect] = summary
        else:
            grouped[dialect] = outcome
    return {
        "task_id": task.task_id,
        "task_hash": task.task_hash,
        "motion_hash": motion.motion_hash,
        "dialects": grouped,
        "efficiency_eligible": all(item.get("status") == "evaluated" and item["efficiency_eligible"]
                                 for item in grouped.values()),
        "phase": phase,
    }


@contextmanager
def _batch_lock(root):
    """One controller owns one attempt hierarchy; child limits remain separate."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "controller.lock"
    if path.is_symlink():
        raise QualificationError(f"batch lock may not be a symlink: {path}")
    stream = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QualificationError(f"another controller owns the batch lock: {path}") from error
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _cohort_identity(document):
    """Posted fields/provenance do not change task/design/cohort identity."""
    stripped = deepcopy(document)
    stripped.pop("pending_posted_qualification", None)
    stripped.pop("posted_qualification_batch", None)
    for detail in stripped.get("task_details", {}).values():
        detail.pop("posted_qualification", None)
    return C.content_hash(stripped)


CODE_SOURCES = (
    "benchmarks/contracts.py", "benchmarks/pilot_task.py",
    "cam/gcode_parser.py", "cam/posts.py", "cam/posted_evaluator.py", "cam/seeds.py",
    "scripts/qualify_task_suite.py", "scripts/run_benchmark_suite.py",
    "simulator/csg_simulator.py", "verification/reference_carve.py",
)


def _source_snapshot(repository=PROJECT_ROOT, *, require_clean=True):
    repository = Path(repository).resolve()
    if require_clean:
        def git(*args):
            return subprocess.run(
                ["git", "-C", str(repository), *args],
                check=True, capture_output=True, text=True,
            ).stdout
        state = git("status", "--porcelain=v1", "--untracked-files=all")
        if state.strip():
            raise QualificationError(f"batch requires a clean committed tree:\n{state}")
        commit = git("rev-parse", "HEAD").strip()
    else:
        # Internal test harness only; the production CLI requires a clean tree.
        commit = "test-only-uncommitted-fixture"
    return {
        "source_commit": commit,
        "source_tree_clean": require_clean,
        # No dirty patch is admitted into a clean production launch.
        "dirty_patch_hash": None if require_clean else "not-applicable-test-fixture",
        "evaluator_revision": evaluator_revision(),
        "code_source_sha256": {
            relative: C.sha256_bytes((repository / relative).read_bytes())
            for relative in CODE_SOURCES
        },
    }


def _declaration(document, source, config, *, concurrency, timeout_s, memory_limit, poll_interval):
    return {
        "schema_version": "t2.3c-batch-declaration/1",
        "workflow_version": WORKFLOW_VERSION,
        "cohort_identity_sha256": _cohort_identity(document),
        "source": source,
        "dialects": [dialect.value for dialect in C.Dialect],
        "role": ROLE,
        "verdict_step_mm": 0.125,
        "concurrency": concurrency,
        "pair_concurrency": (concurrency + 1) // 2,
        "children_per_pair": 2,
        "timeout_seconds_per_job": timeout_s,
        "memory_limit_bytes_per_job": memory_limit,
        "poll_interval_seconds": poll_interval,
        "evaluator_settings_policy": dict(config["refinement_check"]),
        "qualification_rule": (
            "coverage == complete AND safety_pass is True AND geometry_pass is True "
            "in BOTH rs274 AND haas; no optimization-performance exclusion"
        ),
        "physical_cnc_motion": False,
        "timing_varying_fields": [
            "started_utc", "ended_utc", "controller_wall_seconds",
            "phase_wall_seconds", "worker_wall_seconds", "process_wall_seconds",
        ],
    }


def _qualified_of(outcome):
    """Derive the predeclared verdict; never trust a stored qualified boolean."""
    if not isinstance(outcome, dict) or outcome.get("status") != "evaluated":
        return False
    required = ("coverage", "safety_pass", "geometry_pass", "efficiency_eligible")
    if any(name not in outcome for name in required):
        raise QualificationError("evaluated dialect result lacks a verdict field")
    value = (outcome["coverage"] == C.Coverage.COMPLETE.value
             and outcome["safety_pass"] is True and outcome["geometry_pass"] is True)
    if outcome["efficiency_eligible"] is not value:
        raise QualificationError("recorded efficiency eligibility differs from typed verdict")
    return value


def _manifest_dialect(outcome):
    if outcome.get("status") != "evaluated":
        return {
            "schema_version": "t2.3c-dialect-result/1",
            "status": "failed",
            "qualified": False,
            "failure_class": outcome.get("failure_class", "worker_failure"),
            "reason": str(outcome.get("reason", "unspecified worker failure")),
            "prior_artifact": outcome.get("prior_artifact"),
        }
    qualified = _qualified_of(outcome)
    return {
        "schema_version": "t2.3c-dialect-result/1",
        "status": "qualified" if qualified else "not_qualified",
        "qualified": qualified,
        "coverage": outcome["coverage"],
        "coverage_reasons": list(outcome.get("coverage_reasons", [])),
        "safety_pass": outcome["safety_pass"],
        "geometry_pass": outcome["geometry_pass"],
        "efficiency_eligible": qualified,
        "air_fraction": outcome.get("air_fraction"),
        "evaluation_sha256": outcome["evaluation_sha256"],
        "evaluation_path": outcome["evaluation_path"],
        "worker_log": outcome.get("worker_log"),
        "parsed_motion_hash": outcome.get("parsed_motion_hash"),
        "settings": outcome["settings"],
        # These belong to the original immutable evaluation. A resume does not
        # replace them with null fresh-worker fields or a fresh process claim.
        "original_worker_pid": outcome["original_worker_pid"],
        "original_worker_own_hwm_bytes": outcome["original_worker_own_hwm_bytes"],
        "worker_wall_seconds": outcome["worker_wall_seconds"],
        "process_wall_seconds": outcome.get("process_wall_seconds"),
    }


def apply_manifest(document, task_results, provenance, *, expected_path=None):
    """Change only allowed qualification fields and convert the pending block."""
    actual_sources = {relative: C.sha256_bytes((PROJECT_ROOT / relative).read_bytes())
                      for relative in CODE_SOURCES}
    if (provenance.get("code_source_sha256") != actual_sources
            or provenance.get("evaluator_revision") != evaluator_revision()
            or provenance.get("memory_limit_bytes_per_job", MEMORY_LIMIT_BYTES) > MEMORY_LIMIT_BYTES):
        raise QualificationError("batch source/evaluator provenance differs from this production tree")
    updated = deepcopy(document)
    before = deepcopy(document)
    old_block = updated.pop("pending_posted_qualification", None)
    old_existing = updated.pop("posted_qualification_batch", None)
    if old_block is not None:
        if (old_block.get("status") != "pending"
                or old_block.get("dialects") != [dialect.value for dialect in C.Dialect]
                or old_block.get("grid_pitch_mm") != 0.125):
            raise QualificationError("unexpected pending posted-qualification declaration")
    elif old_existing is None:
        raise QualificationError("manifest lacks a pending declaration or prior batch provenance")
    if set(task_results) != set(document["task_details"]):
        raise QualificationError("batch results do not cover every declared task exactly")
    bundles = {}
    resources = {}
    counts = {}
    failures = []
    memory_limit = provenance.get("memory_limit_bytes_per_job", MEMORY_LIMIT_BYTES)
    timeout_s = provenance.get("timeout_seconds_per_job", 7200.0)
    poll_interval = provenance.get("poll_interval_seconds", 1.0)
    for task_id in sorted(task_results):
        result = task_results[task_id]
        if result.get("task_id") != task_id or set(result.get("dialects", {})) != {d.value for d in C.Dialect}:
            raise QualificationError(f"incomplete or mismatched task outcome: {task_id}")
        dialect_records = {name: _manifest_dialect(result["dialects"][name])
                           for name in sorted(result["dialects"])}
        # Family/split and every referenced evaluation are validated against
        # immutable source evidence, not inferred from IDs or report flags.
        entry = next(item for item in document["tasks"] if item["task_id"] == task_id)
        task, motion = _task_from_entry(entry)
        if result.get("motion_hash") != entry["motion_hash"]:
            raise QualificationError(f"task result identity differs from the manifest: {task_id}")
        config = _config(parallel=1, timeout_s=timeout_s, memory_limit=memory_limit,
                         poll_interval=poll_interval, task=task)
        for dialect_name, outcome in result["dialects"].items():
            if outcome.get("status") != "evaluated":
                continue
            source = Path(outcome["evaluation_path"])
            dialect = C.Dialect(dialect_name)
            job, _comparison = _nc_artifacts(task, motion, dialect, source.parent, config)
            old = _read_old_result(source.parent, task, motion, job, memory_limit=memory_limit)
            if old is None or "evaluation" not in old:
                raise QualificationError(f"source evaluation is incomplete for {task_id}/{dialect_name}")
            expected = _result_summary(old["evaluation"], old["worker"], reused=False)
            stable_fields = (
                "coverage", "coverage_reasons", "safety_pass", "geometry_pass",
                "efficiency_eligible", "air_fraction", "original_worker_pid",
                "original_worker_own_hwm_bytes", "original_worker_self_reported_hwm_bytes",
                "original_parent_observed_hwm_kib", "worker_wall_seconds", "process_wall_seconds",
                "evaluation_path", "evaluation_sha256", "worker_log", "parsed_motion_hash", "settings",
            )
            if any(outcome.get(name) != expected.get(name) for name in stable_fields):
                raise QualificationError(f"report differs from immutable evidence for {task_id}/{dialect_name}")
        valid = all(item.get("status") == "evaluated" for item in result["dialects"].values())
        qualified = valid and all(_qualified_of(item) for item in result["dialects"].values())
        details = [f"{dialect}: {record['reason']}" for dialect, record in dialect_records.items()
                   if record["status"] == "failed"]
        for name, record in dialect_records.items():
            if record["status"] == "not_qualified":
                details.append(f"{name}: coverage={record['coverage']}, "
                               f"coverage_reasons={record['coverage_reasons']}, "
                               f"safety_pass={record['safety_pass']}, geometry_pass={record['geometry_pass']}")
        status = "failed" if any(item["status"] == "failed" for item in dialect_records.values()) else (
            "qualified" if qualified else "not_qualified")
        bundles[task_id] = {
            "schema_version": "t2.3c-posted-qualification/1",
            "status": status,
            "qualified": qualified,
            "efficiency_eligible": qualified,
            "reason": None if qualified else "; ".join(details),
            "task_hash": result.get("task_hash"),
            "motion_hash": result.get("motion_hash"),
            "dialects": dialect_records,
        }
        resources[task_id] = {
            name: {
                "worker_pid": record.get("original_worker_pid"),
                "peak_rss_bytes": record.get("original_worker_own_hwm_bytes"),
                "worker_wall_seconds": record.get("worker_wall_seconds"),
                "process_wall_seconds": record.get("process_wall_seconds"),
            } for name, record in dialect_records.items()
        }
        if not qualified:
            failures.append({"task_id": task_id, "status": status, "reason": bundles[task_id]["reason"]})
        # Family/split are validated against the task record, not inferred from IDs.
        key = f"{task.family}/{task.split.value}"
        bucket = counts.setdefault(key, {"qualified": 0, "not_qualified": 0, "failed": 0, "total": 0})
        bucket[status] += 1
        bucket["total"] += 1
        updated["task_details"][task_id]["posted_qualification"] = bundles[task_id]
    provenance = deepcopy(provenance)
    provenance.update(counts_by_family_split=counts, per_job_process=resources)
    updated["posted_qualification_batch"] = provenance
    # Structurally verify the lease rather than relying on textual diff shape.
    before.pop("pending_posted_qualification", None)
    before.pop("posted_qualification_batch", None)
    comparison = deepcopy(updated)
    comparison.pop("posted_qualification_batch")
    for task_id in before["task_details"]:
        before["task_details"][task_id].pop("posted_qualification", None)
        comparison["task_details"][task_id].pop("posted_qualification")
    if before != comparison:
        raise QualificationError("manifest update would modify fields outside its lease")
    text = _json_bytes(updated).decode("utf-8")
    if expected_path is not None:
        Path(expected_path).write_text(text, encoding="utf-8")
    return text


def _run_one(task, motion, task_root, config):
    result = execute_task(task, motion, task_root, config)
    # Outcome indexes are append-only. They are auxiliary summaries; immutable
    # child/evaluator files and old failures are never rewritten by this index.
    path = task_root / f"outcome-{time.time_ns()}.json"
    _write_new(path, result)
    return result


def run_batch(manifest, output_root, *, concurrency=8, timeout_s=7200.0,
              memory_limit=MEMORY_LIMIT_BYTES, poll_interval=1.0,
              require_full=True, require_clean=True):
    """Run every declared task, retaining per-task failures rather than dropping them."""
    if concurrency < 1 or concurrency > 8:
        raise QualificationError("production child concurrency is limited to eight")
    manifest_path, document, cohort = load_cohort(manifest, require_full=require_full)
    output_root = Path(output_root).resolve()
    source = _source_snapshot(require_clean=require_clean)
    first_config = _config(parallel=concurrency, timeout_s=timeout_s, memory_limit=memory_limit,
                           poll_interval=poll_interval, task=cohort[0][0])
    declaration = _declaration(document, source, first_config, concurrency=concurrency,
                               timeout_s=timeout_s, memory_limit=memory_limit,
                               poll_interval=poll_interval)
    started = time.perf_counter()
    results = {}
    with _batch_lock(output_root):
        _write_new(output_root / "batch-declaration.json", declaration)
        # Four pairs have at most two live children each: a hard ceiling of eight.
        pools = (concurrency + 1) // 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=pools) as pool:
            pending = {}
            for task, motion, _entry, _detail in cohort:
                config = _config(parallel=concurrency, timeout_s=timeout_s, memory_limit=memory_limit,
                                 poll_interval=poll_interval, task=task)
                task_root = output_root / "tasks" / task.task_id
                pending[pool.submit(_run_one, task, motion, task_root, config)] = task
            for future in concurrent.futures.as_completed(pending):
                task = pending[future]
                try:
                    value = future.result()
                except Exception as error:
                    # Even an outer-controller failure leaves a task-level record.
                    value = {
                        "task_id": task.task_id, "task_hash": task.task_hash,
                        "motion_hash": None,
                        "dialects": {dialect.value: {
                            "status": "failed", "failure_class": "controller_failure",
                            "reason": f"{type(error).__name__}: {error}", "reused": False,
                        } for dialect in C.Dialect},
                        "efficiency_eligible": False,
                        "controller_traceback": traceback.format_exc(),
                    }
                results[task.task_id] = value
        ended = time.perf_counter()
        provenance = dict(source, workflow_version=WORKFLOW_VERSION,
                          concurrency=concurrency, timeout_seconds_per_job=timeout_s,
                          memory_limit_bytes_per_job=memory_limit,
                          poll_interval_seconds=poll_interval,
                          verdict_step_mm=0.125,
                          source_manifest_sha256=C.sha256_bytes(manifest_path.read_bytes()),
                          cohort_identity_sha256=_cohort_identity(document),
                          settings_sha256=C.content_hash({
                              "declaration": declaration["evaluator_settings_policy"],
                              "role": ROLE, "verdict_step_mm": 0.125,
                          }),
                          started_utc=datetime.fromtimestamp(
                              time.time() - (ended - started), timezone.utc).isoformat(),
                          ended_utc=datetime.now(timezone.utc).isoformat(),
                          controller_wall_seconds=ended - started,
                          timing_varying_fields=declaration["timing_varying_fields"])
        # apply_manifest validates task/dialect verdicts and allowed fields. Raw
        # failed records are retained separately when their two dialects differ.
        for result in results.values():
            if result.get("motion_hash") is None:
                # bind the known seed even for a controller-level exception
                task = next(task for task, *_ in cohort if task.task_id == result["task_id"])
                result["motion_hash"] = seed_for_task(task).motion_hash
        manifest_text = apply_manifest(document, results, provenance)
        report = {
            "schema_version": "t2.3c-batch-report/1",
            "status": "complete_with_failures" if any(
                value["posted_qualification"]["status"] == "failed"
                for value in json.loads(manifest_text)["task_details"].values()
            ) else "complete",
            "provenance": json.loads(manifest_text)["posted_qualification_batch"],
            "tasks": results,
            "manifest_sha256": C.sha256_bytes(manifest_text.encode("utf-8")),
        }
        _write_new(output_root / "batch-report.json", report)
        return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs" / "task_manifest.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--memory-bytes", type=int, default=MEMORY_LIMIT_BYTES)
    parser.add_argument("--report-only", action="store_true",
                        help="render a verified completed report after the final manifest commit; no jobs")
    parser.add_argument("--write-manifest", action="store_true",
                        help="write only allowed per-task qualification fields plus batch provenance")
    args = parser.parse_args(argv)
    if args.report_only:
        report = json.loads((args.output_root / "batch-report.json").read_text(encoding="utf-8"))
        _path, document, _cohort = load_cohort(args.manifest)
        if _cohort_identity(document) != report["provenance"]["cohort_identity_sha256"]:
            raise QualificationError("report cohort identity differs from the current manifest")
        text = apply_manifest(document, report["tasks"], report["provenance"])
        if C.sha256_bytes(text.encode("utf-8")) != report["manifest_sha256"]:
            raise QualificationError("reported manifest is not reproducible")
    else:
        report = run_batch(args.manifest, args.output_root, concurrency=args.parallel,
                           timeout_s=args.timeout_s, memory_limit=args.memory_bytes,
                           poll_interval=args.poll_interval_s)
        text = apply_manifest(json.loads(Path(args.manifest).read_text()), report["tasks"],
                              report["provenance"])
    if args.write_manifest:
        Path(args.manifest).write_text(text, encoding="utf-8")
    print(json.dumps({"status": report["status"], "manifest_sha256": C.sha256_bytes(text.encode("utf-8")),
                      "batch_report": str(args.output_root / "batch-report.json")}, indent=2))
    return 2 if report["status"] == "complete_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
