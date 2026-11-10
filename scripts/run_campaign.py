"""Resumable, bounded campaign orchestration over typed benchmark runner workers.

This module schedules experiments; it does not decide machining eligibility.
Only the coordinator may launch a real campaign from a reviewed configuration.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import heapq
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Callable, Mapping, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import run_artifacts as R
from benchmarks import audit
from benchmarks import contracts as C
from scripts import run_benchmark_suite as runner


CAMPAIGN_SCHEMA = "diffcam-campaign/1"
REPOSITORY = runner.REPOSITORY


class CampaignError(RuntimeError):
    pass


def _atomic(path: Path, payload: Any):
    R.ArtifactWriter._replace_bytes(path, R.json_bytes(payload))


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise CampaignError(f"{name} must be an object")
    return value


def _positive_int(value, name, *, maximum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        raise CampaignError(f"{name} must be an integer in [1,{maximum}]")
    return value


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise CampaignError(f"{name} must be finite and nonnegative")
    return float(value)


def _conditions(document):
    if not isinstance(document, list):
        raise CampaignError("conditions must be a list")
    raw = document
    result = []
    seen = set()
    for case in raw:
        case = _mapping(case, "conditions[]")
        if set(case) != {"condition", "seeds"}:
            raise CampaignError("each campaign condition has exactly condition and seeds")
        method = case["condition"]
        if method not in runner.CONDITIONS:
            raise CampaignError(f"unsupported campaign condition {method!r}")
        seeds = case["seeds"]
        if (not isinstance(seeds, list) or not seeds
                or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
            raise CampaignError("condition seeds must be a nonempty list of nonnegative integers")
        if method in ("nominal-cam", "fixed-adam", "annealed-adam") and seeds != [0]:
            raise CampaignError(f"{method} has exactly one deterministic seed slot 0")
        if len(set(seeds)) != len(seeds):
            raise CampaignError(f"duplicate campaign seed declaration for {method}")
        for seed in seeds:
            key = (method, seed)
            if key in seen:
                raise CampaignError(f"duplicate expected campaign case {key}")
            seen.add(key)
        result.append((method, tuple(seeds)))
    if not result:
        raise CampaignError("campaign requires at least one condition")
    if not dict(result).get("nominal-cam") and len(result) > 1:
        raise CampaignError("paired campaign comparisons require nominal-cam")
    return tuple(result)


def _load(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read campaign config {path}: {error}") from error
    document = dict(_mapping(document, "campaign config"))
    if document.get("schema") != CAMPAIGN_SCHEMA:
        raise CampaignError(f"config.schema must be {CAMPAIGN_SCHEMA}")
    source = document.get("task_source", "suite_manifest")
    if source not in ("suite_manifest", "prototype_analytic"):
        raise CampaignError("campaign source is suite_manifest, or explicitly opted-in prototype_analytic")
    if source == "prototype_analytic" and document.get("prototype_unit_test") is not True:
        raise CampaignError("prototype_analytic is permitted only for explicit campaign unit tests")
    if source == "suite_manifest" and document.get("task_manifest") != runner._SUITE_MANIFEST_RELATIVE:
        raise CampaignError("campaign must identify the registered configs/task_manifest.json")
    concurrency = _positive_int(document.get("concurrency", 6), "concurrency", maximum=12)
    template = copy.deepcopy(_mapping(document.get("runner_config"), "runner_config"))
    if template.get("task_source", source) != source or "task_id" in template:
        raise CampaignError("task_source must be selected once by the campaign config, not a task_id override")
    if template.get("initialization"):
        raise CampaignError("this campaign driver supports the existing CAM init axis only")
    template["task_source"] = source
    template["experiment"] = "pilot"
    if source == "suite_manifest":
        template.setdefault("campaign_verdict_only", True)
    filters = _mapping(document.get("task_filter"), "task_filter")
    if set(filters) - {"splits", "families", "task_ids"}:
        raise CampaignError("task_filter accepts only splits, families, task_ids")
    splits = filters.get("splits", ["dev"])
    if (not isinstance(splits, list) or not splits
            or any(split not in runner.SPLITS for split in splits) or len(set(splits)) != len(splits)):
        raise CampaignError("task_filter.splits must contain unique dev/test/external values")
    for name in ("families", "task_ids"):
        values = filters.get(name, [])
        if not isinstance(values, list) or any(
                not isinstance(value, str) or not value or "/" in value or "\\" in value for value in values):
            raise CampaignError(f"task_filter.{name} must contain safe string identifiers")
    conditions = _conditions(document.get("conditions"))
    if source == "prototype_analytic":
        if splits != ["dev"] or filters.get("task_ids"):
            raise CampaignError("prototype unit campaign selects the one supplied dev task")
    return {"document": document, "source": source, "template": template,
            "conditions": conditions, "filters": filters, "concurrency": concurrency,
            "config_sha256": C.sha256_bytes(path.read_bytes())}


def _task_rows(loaded):
    if loaded["source"] == "prototype_analytic":
        task, _motion, _params, _factory, _posted = runner._context(
            dict(loaded["template"], task_source="prototype_analytic"), REPOSITORY)
        return [task]
    path = REPOSITORY / runner._SUITE_MANIFEST_RELATIVE
    manifest = runner._json(path)
    if C.sha256_bytes(path.read_bytes()) != runner._SUITE_MANIFEST_SHA256:
        raise CampaignError("the campaign's suite manifest SHA differs from the registered input")
    splits = set(loaded["filters"]["splits"])
    families = set(loaded["filters"].get("families", []))
    ids = set(loaded["filters"].get("task_ids", []))
    selected = []
    known_ids = set()
    for item in manifest.get("tasks", []):
        if not isinstance(item, Mapping):
            raise CampaignError("suite manifest contains a non-object task entry")
        task_id = item.get("task_id")
        if item.get("split") not in splits or task_id in known_ids:
            continue
        known_ids.add(task_id)
        if families and item.get("family") not in families:
            continue
        if ids and task_id not in ids:
            continue
        selected.append(task_id)
    if ids and ids != set(selected):
        raise CampaignError(f"task filter selects unknown/out-of-scope task IDs: {sorted(ids - set(selected))}")
    if not selected:
        raise CampaignError("task filter selects no campaign tasks")
    return sorted(selected)


def _selected_entries(task, config, conditions):
    entries = runner._plan_entries(config, task, task.split.value, None)
    wanted = {(method, seed) for method, seeds in conditions for seed in seeds}
    selected = [entry for entry in entries
                if entry.get("init", "cam") == "cam" and (entry["method"], entry["seed"]) in wanted]
    if len(selected) != len(wanted):
        raise CampaignError(f"task {task.task_id}: runner did not plan every requested condition/seed")
    order = {method: index for index, method in enumerate(runner.CONDITIONS)}
    selected.sort(key=lambda entry: (order[entry["method"]], entry["seed"], entry["run_id"]))
    return selected


def _comparisons(conditions, dialects=C.Dialect):
    methods = list(dict.fromkeys(method for method, _seeds in conditions if method != "nominal-cam"))
    return [
        {"name": f"{method}-versus-nominal-{dialect.value}", "method": method,
         "baseline": "nominal-cam", "dialect": dialect.value,
         "metrics": ["cycle_time_s", "air_fraction", "force_proxy", "mrr_mm3_s",
                     "material_removal_rate_mm3_s"]}
        for method in methods for dialect in dialects
    ]


def _prepare(loaded, root: Path, *, resume: bool, dry_run: bool):
    if root.exists() and not resume and any(root.iterdir()):
        raise CampaignError(f"campaign output root must be empty/new, or explicitly resumed: {root}")
    if resume and not (root / "campaign_plan.json").is_file():
        raise CampaignError("--resume requires an existing campaign_plan.json")
    root.mkdir(parents=True, exist_ok=True)
    tasks = _task_rows(loaded)
    cma_seeds = sorted({seed for method, seeds in loaded["conditions"]
                        if method.endswith("cmaes") for seed in seeds})
    jobs = []
    task_documents = []
    for item in tasks:
        config = copy.deepcopy(loaded["template"])
        if isinstance(item, str):
            config["task_id"] = item
        task = runner._context(config, root)[0]
        if task.split.value not in loaded["filters"]["splits"]:
            raise CampaignError(f"task {task.task_id} has an unselected split")
        if (loaded["filters"].get("families")
                and task.family not in loaded["filters"]["families"]):
            raise CampaignError(f"task {task.task_id} has an unselected family")
        config["experiment"] = "pilot"
        config["seeds"] = cma_seeds or [0]
        runner._refinement_policy(config)
        runner._campaign_verdict_only(config)
        entries = _selected_entries(task, config, loaded["conditions"])
        task_root = root / "tasks" / task.task_id
        config_path = task_root / "campaign-runner-config.json"
        document = {"task": task, "entries": entries, "config": config,
                    "root": task_root, "config_path": config_path}
        jobs.extend(dict(document, entry=entry) for entry in entries)
        task_documents.append(document)
    if len({job["entry"]["run_id"] for job in jobs}) != len(jobs):
        raise CampaignError("campaign run identities are not unique")
    source = R.source_metadata(REPOSITORY)
    for job in jobs:
        job["source_metadata"] = source
    expected = [{"run_id": job["entry"]["run_id"], "task_id": job["task"].task_id,
                 "method": job["entry"]["method"], "seed": job["entry"]["seed"],
                 "status": "pending", "source_commit": source["source_commit"],
                 "dirty_patch_hash": source["dirty_patch_hash"]}
                for job in jobs]
    plan_core = {
        "schema": CAMPAIGN_SCHEMA, "campaign_id": loaded["document"].get("campaign_id", "unnamed-campaign"),
        "source": loaded["source"], "suite_manifest_sha256": runner._SUITE_MANIFEST_SHA256,
        "config_sha256": loaded["config_sha256"],
        "conditions": [[method, list(seeds)] for method, seeds in loaded["conditions"]],
        "source_metadata": source,
        "workspaces": [{"task_hash": job["task"].task_hash, "task_id": job["task"].task_id,
                        "config_sha256": C.sha256_bytes(R.json_bytes(job["config"])),
                        "run_id": job["entry"]["run_id"], "identity": job["entry"]["identity"]}
                       for job in jobs],
        "comparisons": _comparisons(loaded["conditions"]),
    }
    fingerprint = C.content_hash(plan_core)
    if resume:
        previous = runner._json(root / "campaign_plan.json")
        if previous.get("plan_hash") != fingerprint or previous.get("core") != plan_core:
            raise CampaignError("resume config, plan, or source revision differs")
        if dry_run and (root / "execution.json").is_file():
            raise CampaignError("an executed campaign cannot be reset to a dry-run manifest")
    else:
        _atomic(root / "campaign_plan.json", {"core": plan_core, "plan_hash": fingerprint})
    tasks_manifest = []
    for document in task_documents:
        task, config = document["task"], document["config"]
        task_root = document["root"]
        task_root.mkdir(parents=True, exist_ok=True)
        frozen = {"raw_config": config, "identity": document["entries"][0]["identity"]}
        manifest = runner._manifest(config, task, document["entries"], task_root, planned=dry_run)
        if resume:
            if runner._json(document["config_path"]) != config:
                raise CampaignError(f"resume config differs for {task.task_id}; refusing to overwrite its snapshot")
            before = runner._json(task_root / "frozen_config.json")
            if before != frozen:
                raise CampaignError(f"resume task/config snapshot differs for {task.task_id}")
        else:
            _atomic(document["config_path"], config)
            _atomic(task_root / "frozen_config.json", frozen)
        _atomic(task_root / "manifest.json", manifest)
        tasks_manifest.append({"task_id": task.task_id, "family": task.family,
                               "task_hash": task.task_hash, "task_split": task.split.value,
                               "completion_residual_mm3": task.tolerances.completion_residual_mm3})
    return {"jobs": jobs, "tasks": tasks_manifest, "plan_core": plan_core,
            "plan_hash": fingerprint, "source": source}


def _worker_command(job):
    return [sys.executable, str(REPOSITORY / "scripts" / "run_benchmark_suite.py"),
            "--config", str(job["config_path"]), "--output-dir", str(job["root"]),
            "--split", job["task"].split.value, "--experiment", "pilot",
            "--only", job["entry"]["run_id"], "--worker-run"]


def _read_outcome(job):
    path = job["root"] / job["entry"]["run_id"] / "run.json"
    if not path.is_file():
        raise CampaignError(f"worker produced no run.json: {job['entry']['run_id']}")
    record = C.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, C.RunRecord) or record.run_id != job["entry"]["run_id"]
            or record.task_hash != job["task"].task_hash):
        raise CampaignError(f"worker run/task identity differs: {path}")
    return record


def _outcome(job, record, **fields):
    """The artifact is canonical; failing legacy files lack source fields, so use their frozen parent."""
    return {"run_id": record.run_id, "status": record.status.value,
            "source_commit": record.source_commit or job["source_metadata"]["source_commit"],
            "dirty_patch_hash": (record.dirty_patch_hash
                                 if record.dirty_patch_hash is not None
                                 else job["source_metadata"]["dirty_patch_hash"]),
            "run_json_sha256": C.sha256_bytes(
                (job["root"] / record.run_id / "run.json").read_bytes()), **fields}


def _execute_one(job, *, resume: bool, spawn: Optional[Callable] = None):
    entry, task, root = job["entry"], job["task"], job["root"]
    started = time.perf_counter()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{entry['run_id']}-{time.time_ns()}.log"
    code = None
    try:
        if resume and runner._resume_existing(root, entry, task):
            record = _read_outcome(job)
            return _outcome(job, record, resumed_skipped=True,
                            elapsed_seconds=time.perf_counter() - started)
        command = _worker_command(job)
        environment = dict(os.environ, DIFFCAM_RUNNER_SIMULATOR="0")
        make_process = subprocess.Popen if spawn is None else spawn
        with log_path.open("wb") as stream:
            process = make_process(command, cwd=REPOSITORY, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=job["config"].get("timeout_s"))
            except subprocess.TimeoutExpired:
                # Only this worker's exact direct child PID; no pattern kill.
                process.terminate()
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
                raise CampaignError(f"worker timeout; exact child pid {process.pid} stopped: {entry['run_id']}")
        if code:
            raise CampaignError(f"runner worker returned {code}")
        record = _read_outcome(job)
    except Exception as error:
        try:
            motion = runner._context(job["config"], root)[1]
            runner._record_interrupted(entry, task, motion, job["config"], root, str(error))
        except Exception as preservation_error:
            return {"run_id": entry["run_id"], "status": "invalid",
                    "log": str(log_path), "error": f"{error}; preservation failed: {preservation_error}"}
        record = _read_outcome(job)
        return _outcome(job, record, log=str(log_path), returncode=code, error=str(error))
    return _outcome(job, record, log=str(log_path), returncode=code,
                    elapsed_seconds=time.perf_counter() - started)


def _expected_manifest(prepared, expected, root):
    return {"schema_version": 1, "campaign_schema": CAMPAIGN_SCHEMA,
            "tasks": prepared["tasks"], "expected_runs": expected,
            "comparisons": prepared["plan_core"]["comparisons"],
            "campaign": {"plan_hash": prepared["plan_hash"],
                         "config_sha256": prepared["plan_core"]["config_sha256"],
                         "campaign_id": prepared["plan_core"]["campaign_id"],
                         "source_metadata": prepared["source"]},
            "audit_runs_root": "audit-runs", "audit_manifest_root": str(root)}


def _cost_estimate(loaded, prepared, concurrency):
    model = loaded["document"].get("cost_model")
    if model is None:
        return {"estimated_wall_seconds": None, "reason": "no declared arithmetic cost model"}
    model = _mapping(model, "cost_model")
    rates = _mapping(model.get("seconds_per_candidate_by_condition"), "candidate cost model")
    overhead = _mapping(model.get("fixed_overhead_seconds_by_condition", {}), "fixed overhead cost model")
    rows = []
    for job in prepared["jobs"]:
        method = job["entry"]["method"]
        if method not in rates:
            return {"estimated_wall_seconds": None, "reason": f"no cost unit for {method}"}
        config = dict(job["config"], estimated_seconds_per_candidate={method: _seconds(rates[method], method)})
        cost = runner._pecent(config, [job["entry"]])[0]
        seconds = cost["estimated_run_wall_time_s"] + _seconds(overhead.get(method, 0.0), method + " overhead")
        rows.append({"run_id": job["entry"]["run_id"], "candidate_evaluations": cost["candidate_evaluations"],
                     "validation_calls": cost["validation_calls"], "workload_seconds": seconds})
    free = [0.0] * concurrency
    heapq.heapify(free)
    for row in rows:
        available = heapq.heappop(free)
        heapq.heappush(free, available + row["workload_seconds"])
    return {"scope": model.get("scope", "caller-supplied arithmetic cost model, not measured timing"),
            "serial_work_seconds": sum(row["workload_seconds"] for row in rows),
            "estimated_wall_seconds": max(free),
            "schedule_lower_bound_seconds": sum(row["workload_seconds"] for row in rows) / concurrency,
            "concurrency": concurrency, "costs": rows}


def _project_one(record, source_dir, audit_root):
    destination_dir = audit_root / record.run_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in (*record.artifacts, "run.json"):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise CampaignError(f"unsafe projection artifact {name!r}")
        source = source_dir / path
        target = destination_dir / path
        if not source.is_file():
            continue
        temporary = target.with_name(target.name + f".projection-{time.time_ns()}")
        try:
            os.link(source, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()


def _audit_projection(prepared, root):
    projection = root / "audit-runs"
    projection.mkdir(exist_ok=True)
    expected_by_root = {}
    for job in prepared["jobs"]:
        expected_by_root.setdefault(job["root"], set()).add(job["entry"]["run_id"])
    for task_root, expected_ids in expected_by_root.items():
        for child in task_root.iterdir():
            run_json = child / "run.json"
            if not run_json.is_file():
                continue
            if child.name not in expected_ids:
                raise CampaignError(f"unexpected run artifact outside expected identities: {child}")
            record = C.loads(run_json.read_text(encoding="utf-8"))
            if not isinstance(record, C.RunRecord) or record.run_id != child.name:
                raise CampaignError(f"cannot audit a malformed campaign record: {run_json}")
            _project_one(record, child, projection)


def _write_audit(prepared, root, configured):
    manifest_path = root / "manifest.json"
    manifest = audit.load_manifest(manifest_path)
    reports, checked = audit.audit_runs(root / "audit-runs", manifest)
    model = configured["document"].get("cost_model", {})
    replicates = model.get("aggregate_bootstrap_replicates", 2000)
    seed = model.get("aggregate_bootstrap_seed", 20261108)
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise CampaignError("aggregate_bootstrap_replicates must be >= 100")
    comparisons, policy = audit.aggregate_evaluations(
        manifest, checked["_audits_by_id"], bootstrap_seed=seed,
        bootstrap_replicates=replicates)
    report = {"schema": "diffcam-campaign-audit/1", "reports": reports,
              "reconciliation": checked["reconciliation"],
              "summary": checked["summary"], "comparisons": comparisons, "policy": policy}
    _atomic(root / "audit.json", report)
    return report


def run_campaign(config_path: Path, root: Path, *, concurrency=None, resume=False,
                 dry_run=False, spawn=None):
    config_path = Path(config_path).resolve()
    root = Path(root).resolve()
    loaded = _load(config_path)
    if root.is_relative_to(REPOSITORY):
        raise CampaignError("campaign output root must be outside the source checkout")
    concurrency = loaded["concurrency"] if concurrency is None else _positive_int(
        concurrency, "--concurrency", maximum=12)
    prepared = _prepare(loaded, root, resume=resume, dry_run=dry_run)
    expected = [{"run_id": job["entry"]["run_id"], "task_id": job["task"].task_id,
                 "method": job["entry"]["method"], "seed": job["entry"]["seed"],
                 "status": "pending", "source_commit": prepared["source"]["source_commit"],
                 "dirty_patch_hash": prepared["source"]["dirty_patch_hash"]}
                for job in prepared["jobs"]]
    by_id = {row["run_id"]: row for row in expected}
    _atomic(root / "manifest.json", _expected_manifest(prepared, expected, root))
    estimate = _cost_estimate(loaded, prepared, concurrency)
    if dry_run:
        for row in expected:
            row["status"] = "planned"
        document = _expected_manifest(prepared, expected, root)
        document["cost_estimate"] = estimate
        _atomic(root / "manifest.json", document)
        _atomic(root / "dry_run.json", {"run_count": len(expected), **estimate})
        return {"run_count": len(expected), "cost_estimate": estimate, "launched": False}
    _atomic(root / "execution.json", {
        "schema": "diffcam-campaign-execution/1", "plan_hash": prepared["plan_hash"],
        "concurrency": concurrency, "source_metadata": prepared["source"],
        "scope": "benchmark workers, not equipment execution"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}
        for job in prepared["jobs"]:
            by_id[job["entry"]["run_id"]]["status"] = "running"
            _atomic(root / "manifest.json", _expected_manifest(prepared, expected, root))
            pending[pool.submit(_execute_one, job, resume=resume, spawn=spawn)] = job
        for future in concurrent.futures.as_completed(pending):
            job = pending[future]
            try:
                outcome = future.result()
            except Exception as error:
                outcome = {"run_id": job["entry"]["run_id"], "status": "invalid", "error": str(error)}
            by_id[outcome["run_id"]].update(outcome)
            _atomic(root / "manifest.json", _expected_manifest(prepared, expected, root))
    _audit_projection(prepared, root)
    report = _write_audit(prepared, root, loaded)
    document = _expected_manifest(prepared, expected, root)
    document["audit_summary"] = report["summary"]
    document["audit_file"] = "audit.json"
    _atomic(root / "manifest.json", document)
    counts = dict(sorted(Counter(row["status"] for row in expected).items()))
    return {"run_count": len(expected), "status_counts": counts,
            "audit_summary": report["summary"], "audit_file": str(root / "audit.json"),
            "campaign_launched": not dry_run}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_campaign(args.config.resolve(), args.output_dir.resolve(),
                              concurrency=args.concurrency, resume=args.resume, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CampaignError, runner.RunnerError, audit.AuditInputError) as error:
        print(f"campaign failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
