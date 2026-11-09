"""Run-artifact auditing and task-clustered evaluation statistics (T4.0).

No simulator is imported. Input identities are checked before any result reaches
statistics. Seeds are averaged inside a task/method first; every CI and p-value
then treats one task, not one solver seed, as an independent unit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import beta

from benchmarks import contracts as C


REPORT_SCHEMA = "diffcam-eval-audit/1"
EXACT_SIGN_FLIP_MAX_TASKS = 16
EFFICIENCY_METRICS = frozenset(
    {"cycle_time_s", "air_fraction", "force_proxy", "mrr_mm3_s", "material_removal_rate_mm3_s"}
)
OUTCOMES = ("safety_pass", "geometry_pass", "completion_pass", "efficiency_eligible")


class AuditInputError(ValueError):
    """A malformed manifest, run directory, or aggregation request."""


def _validate_verdict_evaluations(values, manifest):
    """Validate the fine typed evaluations that alone supply pilot verdicts."""
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise AuditInputError("verdict evaluations must be a mapping by run ID")
    unknown = set(values) - set(manifest.runs)
    if unknown:
        raise AuditInputError(f"verdict evaluations for unknown run IDs: {sorted(map(str, unknown))}")
    if set(values) != set(manifest.runs):
        raise AuditInputError("every expected primary run needs both fine dialect evaluations")
    result = {}
    for run_id, dialects in values.items():
        if not isinstance(dialects, Mapping) or set(dialects) != {item.value for item in C.Dialect}:
            raise AuditInputError(f"run {run_id}: verdict evaluations require both rs274 and haas")
        task_id = manifest.runs[run_id].task_id
        task_hash = manifest.tasks[task_id].task_hash
        result[run_id] = {}
        for dialect, evaluation in dialects.items():
            if not isinstance(evaluation, C.EvaluationResult) or evaluation.dialect is not C.Dialect(dialect):
                raise AuditInputError(f"run {run_id}, {dialect}: verdict evaluation is not the same typed dialect")
            if evaluation.task_hash != task_hash:
                raise AuditInputError(f"run {run_id}, {dialect}: verdict task hash differs from the primary task")
            settings = evaluation.discretization.get("settings")
            if not isinstance(settings, Mapping) or settings.get("max_step_mm") != 0.125:
                raise AuditInputError(f"run {run_id}, {dialect}: verdict source must explicitly use max_step_mm=0.125")
            result[run_id][dialect] = evaluation
    return result


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    task_hash: str
    family: str
    completion_residual_mm3: Optional[float] = None
    removal_required: bool = True


@dataclass(frozen=True)
class ExpectedRun:
    run_id: str
    task_id: str
    method: str
    seed: int


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    method: str
    baseline: str
    dialect: C.Dialect
    metrics: tuple[str, ...]
    families: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class Manifest:
    tasks: dict[str, TaskInfo]
    runs: dict[str, ExpectedRun]
    comparisons: tuple[ComparisonSpec, ...]


@dataclass
class DialectAudit:
    dialect: C.Dialect
    status: str
    issues: list[dict[str, str]]
    evaluation: Optional[C.EvaluationResult] = None
    nc_bytes: Optional[int] = None
    declared_nc_hash: Optional[str] = None
    actual_nc_hash: Optional[str] = None


@dataclass
class RunAudit:
    run_id: str
    run_dir: str
    status: str
    issues: list[dict[str, str]]
    task_hash: Optional[str] = None
    record: Optional[C.RunRecord] = None
    dialects: dict[str, DialectAudit] | None = None
    duplicate_id: bool = False
    expected: bool = True

    def __post_init__(self) -> None:
        if self.dialects is None:
            self.dialects = {}

    @property
    def complete_and_audited(self) -> bool:
        return (
            self.record is not None
            and self.record.status == C.RunStatus.COMPLETE
            and not self.issues
            and not self.duplicate_id
            and self.expected
        )


def _issue(code: str, message: str, path: str = "") -> dict[str, str]:
    return {"code": code, "message": message, "path": path}


def _json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    _finite_json(value, str(path))
    return value


def _finite_json(value: Any, where: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {where}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_json(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_json(child, f"{where}[{index}]")


def _valid_hash(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value[7:], 16)
        return True
    except ValueError:
        return False


def _nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditInputError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise AuditInputError(f"{field} must be finite and non-negative")
    return number


def load_manifest(value: Mapping[str, Any] | str | Path) -> Manifest:
    if isinstance(value, (str, Path)):
        raw = _json_load(Path(value))
    else:
        raw = value
        _finite_json(raw, "manifest")
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise AuditInputError("manifest schema_version must be 1")

    raw_tasks = raw.get("tasks")
    task_items: list[tuple[Optional[str], Any]]
    if isinstance(raw_tasks, Mapping):
        task_items = [(str(key), item) for key, item in raw_tasks.items()]
    elif isinstance(raw_tasks, list):
        task_items = [(None, item) for item in raw_tasks]
    else:
        raise AuditInputError("tasks must be a mapping or list")
    tasks: dict[str, TaskInfo] = {}
    for key, item in task_items:
        if not isinstance(item, Mapping):
            raise AuditInputError(f"task {key!r} must be an object")
        task_id = item.get("task_id", key)
        family = item.get("family")
        task_hash = item.get("task_hash")
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise AuditInputError(f"duplicate or missing task_id: {task_id!r}")
        if not isinstance(family, str) or not family:
            raise AuditInputError(f"task {task_id}: non-empty family required")
        if not _valid_hash(task_hash):
            raise AuditInputError(f"task {task_id}: invalid task_hash")
        tolerance = item.get("completion_residual_mm3")
        if tolerance is not None:
            tolerance = _nonnegative(tolerance, f"task {task_id}.completion_residual_mm3")
        removal_required = item.get("removal_required", True)
        if not isinstance(removal_required, bool):
            raise AuditInputError(f"task {task_id}.removal_required must be boolean")
        tasks[task_id] = TaskInfo(
            task_id, task_hash, family, tolerance, removal_required
        )

    raw_runs = raw.get("expected_runs", raw.get("runs"))
    if not isinstance(raw_runs, list):
        raise AuditInputError("expected_runs must be a list of run objects")
    runs: dict[str, ExpectedRun] = {}
    for item in raw_runs:
        if not isinstance(item, Mapping):
            raise AuditInputError("every expected run must be an object")
        run_id, task_id, method, seed = (
            item.get("run_id"), item.get("task_id"), item.get("method"), item.get("seed")
        )
        if not isinstance(run_id, str) or not run_id or run_id in runs:
            raise AuditInputError(f"duplicate or missing run_id: {run_id!r}")
        if task_id not in tasks:
            raise AuditInputError(f"run {run_id}: unknown task_id {task_id!r}")
        if not isinstance(method, str) or not method:
            raise AuditInputError(f"run {run_id}: non-empty method required")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AuditInputError(f"run {run_id}: seed must be a non-negative integer")
        runs[run_id] = ExpectedRun(run_id, task_id, method, seed)

    comparisons: list[ComparisonSpec] = []
    names: set[str] = set()
    raw_comparisons = raw.get("comparisons", [])
    if not isinstance(raw_comparisons, list):
        raise AuditInputError("comparisons must be a list")
    known_families = {task.family for task in tasks.values()}
    for item in raw_comparisons:
        if not isinstance(item, Mapping):
            raise AuditInputError("comparison must be an object")
        name, method, baseline = item.get("name"), item.get("method"), item.get("baseline")
        if not all(isinstance(part, str) and part for part in (name, method, baseline)):
            raise AuditInputError("comparison name/method/baseline are required strings")
        if name in names or method == baseline:
            raise AuditInputError(f"comparison {name}: duplicate name or same method/baseline")
        names.add(name)
        try:
            dialect = C.Dialect(item["dialect"])
        except (KeyError, ValueError) as error:
            raise AuditInputError(f"comparison {name}: rs274 or haas dialect required") from error
        metrics = item.get("metrics")
        if (
            not isinstance(metrics, list)
            or not metrics
            or any(not isinstance(metric, str) or not metric for metric in metrics)
            or len(set(metrics)) != len(metrics)
        ):
            raise AuditInputError(f"comparison {name}: metrics must be a unique non-empty list")
        families = item.get("families")
        if families is not None:
            if not isinstance(families, list) or not families or any(
                family not in known_families for family in families
            ):
                raise AuditInputError(f"comparison {name}: unknown/empty families")
            families = tuple(dict.fromkeys(families))
        methods = {run.method for run in runs.values()}
        missing_methods = {method, baseline} - methods
        if missing_methods:
            raise AuditInputError(
                f"comparison {name}: methods absent from expected runs: {sorted(missing_methods)}"
            )
        comparisons.append(
            ComparisonSpec(
                name, method, baseline, dialect, tuple(metrics), families
            )
        )
    return Manifest(tasks, runs, tuple(comparisons))


def _artifact_path(run_dir: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise AuditInputError(f"unsafe artifact path {value!r}")
    candidate = run_dir.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=False).relative_to(run_dir.resolve())
    except ValueError as error:
        raise AuditInputError(f"artifact escapes run directory: {value!r}") from error
    return candidate


def _index_programs(index: Any) -> Mapping[str, Any]:
    if not isinstance(index, Mapping):
        raise AuditInputError("eval_posted.json must be an object")
    programs = index.get("programs", index.get("dialects", index))
    if not isinstance(programs, Mapping):
        raise AuditInputError("eval_posted.json programs/dialects must be an object")
    unknown = set(programs) - {dialect.value for dialect in C.DIALECT_ARTIFACTS}
    if unknown:
        raise AuditInputError(f"eval_posted.json unknown dialect keys: {sorted(unknown)}")
    schema_keys = {"schema_version", "programs", "dialects"}
    if not programs and set(index) - schema_keys:
        raise AuditInputError("eval_posted.json contains no dialect index")
    return programs


def _evaluation_semantics(evaluation: C.EvaluationResult) -> list[str]:
    errors: list[str] = []
    for name in ("removed_volume_mm3", "residual_volume_mm3", "gouge_volume_mm3", "cycle_time_s"):
        metric = evaluation.metrics.get(name)
        if metric is not None and metric.value is not None and metric.value < 0.0:
            errors.append(f"metric {name} cannot be negative")
    for name in ("iou", "dice", "air_fraction"):
        metric = evaluation.metrics.get(name)
        if metric is not None and metric.value is not None and not 0.0 <= metric.value <= 1.0:
            errors.append(f"metric {name} must be in [0,1]")
    return errors


def _audit_run(run_json: Path) -> RunAudit:
    run_dir = run_json.parent
    try:
        raw = _json_load(run_json)
        fallback_id = raw.get("record", {}).get("run_id", run_dir.name) if isinstance(raw, Mapping) else run_dir.name
        if not isinstance(fallback_id, str) or not fallback_id:
            fallback_id = run_dir.name
    except Exception as error:
        return RunAudit(
            run_dir.name, str(run_dir), "invalid",
            [_issue("malformed_run_json", str(error), str(run_json))],
        )
    try:
        record = C.loads(json.dumps(raw, allow_nan=False))
        if not isinstance(record, C.RunRecord):
            raise TypeError(f"expected RunRecord, got {type(record).__name__}")
    except Exception as error:
        return RunAudit(
            fallback_id, str(run_dir), "invalid",
            [_issue("invalid_run_record", str(error), str(run_json))],
        )

    run = RunAudit(
        record.run_id, str(run_dir), record.status.value, [], record.task_hash, record
    )
    artifact_paths: dict[str, Path] = {}
    for artifact in record.artifacts:
        try:
            path = _artifact_path(run_dir, artifact)
            artifact_paths[artifact] = path
            if not path.is_file():
                run.issues.append(_issue("missing_artifact", artifact, str(path)))
        except (AuditInputError, OSError) as error:
            run.issues.append(_issue("invalid_artifact_path", str(error), str(run_dir)))

    for name in ("config.json", "history.json", "metrics.json"):
        path = artifact_paths.get(name)
        if path is not None and path.is_file():
            try:
                _json_load(path)
            except Exception as error:
                run.issues.append(_issue("malformed_artifact", str(error), str(path)))
    for name in ("initial_trajectory.npy", "trajectory.npy", "trajectory_deltas.npy"):
        path = artifact_paths.get(name)
        if path is not None and path.is_file():
            try:
                array = np.load(path, allow_pickle=False)
                if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must contain finite real numeric values")
            except Exception as error:
                run.issues.append(_issue("malformed_array", str(error), str(path)))
    motion_path = artifact_paths.get("motion_plan.json")
    if motion_path is not None and motion_path.is_file():
        try:
            motion = C.loads(motion_path.read_text(encoding="utf-8"))
            if not isinstance(motion, C.MotionPlan):
                raise TypeError("motion_plan.json is not a MotionPlan")
        except Exception as error:
            run.issues.append(_issue("invalid_motion_plan", str(error), str(motion_path)))

    index_path = artifact_paths.get("eval_posted.json")
    indexed: Mapping[str, Any] = {}
    if index_path is not None and index_path.is_file():
        try:
            indexed = _index_programs(_json_load(index_path))
            for posted in record.posted:
                key = posted.dialect.value
                entry = indexed.get(key)
                if entry is None:
                    run.issues.append(
                        _issue("missing_dialect_index", key, str(index_path))
                    )
                    continue
                if not isinstance(entry, Mapping):
                    raise AuditInputError(f"index {key} must be an object")
                if entry.get("dialect", key) != key:
                    run.issues.append(_issue("index_dialect_mismatch", key, str(index_path)))
                if entry.get("status", posted.status.value) != posted.status.value:
                    run.issues.append(_issue("index_status_mismatch", key, str(index_path)))
                if entry.get("nc_hash") != posted.nc_hash:
                    run.issues.append(_issue("index_nc_hash_mismatch", key, str(index_path)))
                if entry.get("eval_file") != posted.eval_file:
                    run.issues.append(_issue("index_eval_file_mismatch", key, str(index_path)))
        except Exception as error:
            run.issues.append(_issue("invalid_dialect_index", str(error), str(index_path)))

    for posted in record.posted:
        key = posted.dialect.value
        audit = DialectAudit(
            posted.dialect, posted.status.value, [], declared_nc_hash=posted.nc_hash
        )
        run.dialects[key] = audit
        if posted.status != C.PostStatus.SUPPORTED:
            continue
        expected_eval_name = C.DIALECT_ARTIFACTS[posted.dialect][1]
        if posted.eval_file != expected_eval_name:
            audit.issues.append(
                _issue("unexpected_eval_filename", f"expected {expected_eval_name}", key)
            )
        nc_name = C.DIALECT_ARTIFACTS[posted.dialect][0]
        nc_path = artifact_paths.get(nc_name)
        eval_path = artifact_paths.get(posted.eval_file or "")
        if nc_path is not None and nc_path.is_file():
            data = nc_path.read_bytes()
            audit.nc_bytes = len(data)
            audit.actual_nc_hash = C.nc_hash(data)
            if audit.actual_nc_hash != posted.nc_hash:
                audit.issues.append(
                    _issue(
                        "nc_hash_mismatch",
                        f"declared {posted.nc_hash}, actual {audit.actual_nc_hash}",
                        str(nc_path),
                    )
                )
        else:
            audit.issues.append(_issue("missing_nc", nc_name, str(nc_path or run_dir)))
        if eval_path is None or not eval_path.is_file():
            audit.issues.append(
                _issue("missing_evaluation", str(posted.eval_file), str(eval_path or run_dir))
            )
            continue
        try:
            evaluation = C.loads(eval_path.read_text(encoding="utf-8"))
            if not isinstance(evaluation, C.EvaluationResult):
                raise TypeError("not an EvaluationResult")
            audit.evaluation = evaluation
            if evaluation.task_hash != record.task_hash:
                audit.issues.append(_issue("eval_task_hash_mismatch", evaluation.task_hash, key))
            if evaluation.dialect != posted.dialect:
                audit.issues.append(_issue("eval_dialect_mismatch", str(evaluation.dialect), key))
            if evaluation.nc_hash != posted.nc_hash:
                audit.issues.append(_issue("eval_nc_hash_mismatch", str(evaluation.nc_hash), key))
            if audit.actual_nc_hash is not None and evaluation.nc_hash != audit.actual_nc_hash:
                audit.issues.append(_issue("eval_actual_nc_mismatch", str(evaluation.nc_hash), key))
            for message in _evaluation_semantics(evaluation):
                audit.issues.append(_issue("invalid_metric", message, key))
            audit.status = evaluation.coverage.value
        except Exception as error:
            audit.issues.append(_issue("invalid_evaluation", str(error), str(eval_path)))
    return run


def audit_runs(
    runs_root: str | Path,
    manifest: Manifest,
) -> tuple[list[RunAudit], dict[str, Any]]:
    root = Path(runs_root)
    if not root.is_dir():
        raise AuditInputError(f"runs root is not a directory: {root}")
    observed = [_audit_run(path) for path in sorted(root.rglob("run.json"))]
    ids = Counter(run.run_id for run in observed)
    for run in observed:
        run.duplicate_id = ids[run.run_id] > 1
        if run.duplicate_id:
            run.issues.append(_issue("duplicate_run_id", run.run_id, run.run_dir))
        run.expected = run.run_id in manifest.runs
        if not run.expected:
            run.issues.append(_issue("unexpected_run_id", run.run_id, run.run_dir))
            continue
        expected = manifest.runs[run.run_id]
        task = manifest.tasks[expected.task_id]
        if run.record is not None and run.task_hash != task.task_hash:
            run.issues.append(
                _issue(
                    "task_hash_mismatch",
                    f"manifest {task.task_hash}, run {run.task_hash}",
                    run.run_dir,
                )
            )

    expected_ids = set(manifest.runs)
    observed_ids = set(ids)
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    duplicates = sorted(run_id for run_id, count in ids.items() if count > 1)

    reports: list[dict[str, Any]] = []
    all_issue_codes: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    coverage_counts: Counter[str] = Counter()
    for run in observed:
        status_counts[run.status] += 1
        for issue in run.issues:
            all_issue_codes[issue["code"]] += 1
        dialect_reports: dict[str, Any] = {}
        for key, dialect in sorted((run.dialects or {}).items()):
            for issue in dialect.issues:
                all_issue_codes[issue["code"]] += 1
            coverage_counts[dialect.status] += 1
            evaluation = dialect.evaluation
            dialect_reports[key] = {
                "status": dialect.status,
                "usable": run.complete_and_audited and not dialect.issues and evaluation is not None,
                "issues": dialect.issues,
                "declared_nc_hash": dialect.declared_nc_hash,
                "actual_nc_hash": dialect.actual_nc_hash,
                "nc_bytes": dialect.nc_bytes,
                "evaluation": evaluation.to_dict() if evaluation is not None else None,
            }
        reports.append(
            {
                "run_id": run.run_id,
                "run_dir": run.run_dir,
                "status": run.status,
                "expected": run.expected,
                "duplicate_id": run.duplicate_id,
                "task_hash": run.task_hash,
                "issues": run.issues,
                "usable_for_statistics": run.complete_and_audited,
                "dialects": dialect_reports,
            }
        )
    for run_id in missing:
        expected = manifest.runs[run_id]
        reports.append(
            {
                "run_id": run_id,
                "run_dir": None,
                "status": "missing",
                "expected": True,
                "expected_task_id": expected.task_id,
                "expected_method": expected.method,
                "expected_seed": expected.seed,
                "duplicate_id": False,
                "task_hash": None,
                "issues": [_issue("missing_expected_run", run_id)],
                "usable_for_statistics": False,
                "dialects": {},
            }
        )
        status_counts["missing"] += 1
        all_issue_codes["missing_expected_run"] += 1
    reports.sort(key=lambda item: (item["run_id"], item["run_dir"] or ""))
    reconciliation = {
        "expected_run_ids": sorted(expected_ids),
        "observed_run_ids": sorted(observed_ids),
        "missing_run_ids": missing,
        "unexpected_run_ids": unexpected,
        "duplicate_run_ids": duplicates,
        "expected_count": len(expected_ids),
        "observed_record_count": len(observed),
        "exact_id_reconciliation": not missing and not unexpected and not duplicates,
    }
    summary = {
        "run_status_counts": dict(sorted(status_counts.items())),
        "dialect_status_counts": dict(sorted(coverage_counts.items())),
        "issue_counts": dict(sorted(all_issue_codes.items())),
        "auditor_passed": (
            reconciliation["exact_id_reconciliation"]
            and not all_issue_codes
        ),
        "evidence_complete": bool(
            observed and reconciliation["exact_id_reconciliation"]
            and all(run.record is not None and not run.issues and not run.duplicate_id
                    and all(not dialect.issues for dialect in (run.dialects or {}).values())
                    for run in observed if run.expected)
        ),
        "all_runs_completed": bool(
            observed and reconciliation["exact_id_reconciliation"]
            and all(run.status == C.RunStatus.COMPLETE.value for run in observed if run.expected)
        ),
    }
    by_id = {run.run_id: run for run in observed if not run.duplicate_id}
    return reports, {
        "reconciliation": reconciliation,
        "summary": summary,
        "_audits_by_id": by_id,
    }


def _observation(
    expected: ExpectedRun,
    audits: Mapping[str, RunAudit],
    dialect: C.Dialect,
    verdict_evaluations=None,
) -> Optional[C.EvaluationResult]:
    if verdict_evaluations is not None:
        if expected.run_id not in verdict_evaluations or dialect.value not in verdict_evaluations[expected.run_id]:
            raise AuditInputError(f"missing fine verdict evaluation: {expected.run_id}/{dialect.value}")
        return verdict_evaluations[expected.run_id][dialect.value]
    run = audits.get(expected.run_id)
    if run is None or not run.complete_and_audited:
        return None
    result = (run.dialects or {}).get(dialect.value)
    if result is None or result.issues:
        return None
    return result.evaluation


def _completion_value(evaluation: C.EvaluationResult, task: TaskInfo) -> Optional[bool]:
    if evaluation.coverage != C.Coverage.COMPLETE:
        return None
    removed_metric = evaluation.metrics.get("removed_volume_mm3")
    if removed_metric is None or removed_metric.value is None:
        return None
    if task.removal_required and removed_metric.value <= 0.0:
        return False
    if task.completion_residual_mm3 is None:
        return None
    residual_metric = evaluation.metrics.get("residual_volume_mm3")
    if residual_metric is None or residual_metric.value is None:
        return None
    return residual_metric.value <= task.completion_residual_mm3


def _outcome_value(
    evaluation: C.EvaluationResult,
    task: TaskInfo,
    outcome: str,
) -> Optional[bool]:
    if outcome == "efficiency_eligible":
        if not evaluation.efficiency_eligible:
            return False
        completion = _completion_value(evaluation, task)
        # A no-cut removal task cannot supply efficiency evidence even if a malformed
        # producer marked geometry_pass=true. Unknown completion stays unknown.
        return True if completion is True else completion
    if evaluation.coverage != C.Coverage.COMPLETE:
        return None
    if outcome == "safety_pass":
        return evaluation.safety_pass
    if outcome == "geometry_pass":
        return evaluation.geometry_pass
    if outcome == "completion_pass":
        return _completion_value(evaluation, task)
    raise AuditInputError(f"unknown outcome {outcome}")


def _recorded_noncomplete(expected: ExpectedRun, audits: Mapping[str, RunAudit]) -> bool:
    """A real, unambiguous RunRecord explicitly says the run did not complete.

    Malformed/missing/duplicate records stay unknown. Missing evaluation artifacts do
    not erase a valid terminal status recorded before an interrupted or blocked run.
    """
    run = audits.get(expected.run_id)
    return bool(
        run is not None and run.expected and not run.duplicate_id
        and run.record is not None and run.record.status != C.RunStatus.COMPLETE
    )


def _outcome_by_task(
    expected_runs: Sequence[ExpectedRun],
    audits: Mapping[str, RunAudit],
    dialect: C.Dialect,
    outcome: str,
    tasks: Mapping[str, TaskInfo],
    verdict_evaluations=None,
) -> dict[str, dict[str, Any]]:
    collapsed: dict[str, dict[str, Any]] = {}
    for expected in expected_runs:
        evaluation = _observation(expected, audits, dialect, verdict_evaluations)
        entry = collapsed.setdefault(
            expected.task_id,
            {"known": [], "unknown_or_missing_observations": 0},
        )
        if evaluation is None:
            recorded_failure = _recorded_noncomplete(expected, audits)
            value = (False if recorded_failure and outcome in
                     ("safety_pass", "completion_pass", "efficiency_eligible") else None)
        else:
            value = _outcome_value(evaluation, tasks[expected.task_id], outcome)
        if value is None:
            entry["unknown_or_missing_observations"] += 1
        else:
            entry["known"].append(float(bool(value)))
    for entry in collapsed.values():
        known = entry.pop("known")
        entry["value"] = float(np.mean(known)) if known else None
        entry["known_observations"] = len(known)
    return collapsed


def _metric_by_task(
    expected_runs: Sequence[ExpectedRun],
    audits: Mapping[str, RunAudit],
    dialect: C.Dialect,
    metric_name: str,
    tasks: Mapping[str, TaskInfo],
    verdict_evaluations=None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ExpectedRun]] = defaultdict(list)
    for expected in expected_runs:
        grouped[expected.task_id].append(expected)
    collapsed: dict[str, dict[str, Any]] = {}
    for task_id, runs in grouped.items():
        values: list[float] = []
        reasons: Counter[str] = Counter()
        for expected in runs:
            evaluation = _observation(expected, audits, dialect, verdict_evaluations)
            if evaluation is None:
                reason = "recorded_noncomplete" if _recorded_noncomplete(expected, audits) else "missing_or_invalid_evidence"
                reasons[reason] += 1
                continue
            if evaluation.coverage != C.Coverage.COMPLETE:
                reasons["incomplete_or_unsupported_coverage"] += 1
                continue
            metric = evaluation.metrics.get(metric_name)
            if metric is None or metric.value is None:
                reasons["missing_or_null_metric"] += 1
                continue
            if metric_name in EFFICIENCY_METRICS:
                eligible = _outcome_value(evaluation, tasks[expected.task_id], "efficiency_eligible")
                if eligible is not True:
                    reasons["not_efficiency_eligible" if eligible is False else "efficiency_eligibility_unknown"] += 1
                    continue
            values.append(metric.value)
        collapsed[task_id] = {
            "value": float(np.mean(values)) if values else None,
            "observations_used": len(values),
            "exclusions": dict(sorted(reasons.items())),
        }
    return collapsed


def _rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _bootstrap_design(values: np.ndarray, families: Sequence[str]) -> tuple[list[np.ndarray], str]:
    groups = [values[np.asarray(families) == family] for family in sorted(set(families))]
    if any(len(group) < 2 for group in groups):
        # A singleton stratum contributes no resampling variation. Pooling preserves
        # independent task-level resampling while making that limitation explicit.
        return [values], "pooled_tasks_due_to_singleton_stratum"
    return groups, "family_stratified"


def _resampled_mean_distribution(
    values: Sequence[float],
    families: Sequence[str],
    rng: np.random.Generator,
    replicates: int,
) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    groups, _design = _bootstrap_design(values_array, families)
    distribution = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        parts = [group[rng.integers(0, len(group), len(group))] for group in groups]
        distribution[index] = float(np.mean(np.concatenate(parts)))
    return distribution


def _interval(distribution: np.ndarray, confidence: float) -> dict[str, float]:
    alpha = 1.0 - confidence
    return {
        "confidence": confidence,
        "lower": float(np.quantile(distribution, alpha / 2.0)),
        "upper": float(np.quantile(distribution, 1.0 - alpha / 2.0)),
    }


def _sign_flip_p_value(differences: np.ndarray, key: str, seed: int,
                       replicates: int) -> dict[str, Any]:
    """Two-sided paired sign-flip test over independent task differences."""
    task_count = len(differences)
    if task_count <= EXACT_SIGN_FLIP_MAX_TASKS:
        total = 1 << task_count
        assignments = np.arange(total, dtype=np.uint64)[:, None]
        bits = (assignments >> np.arange(task_count, dtype=np.uint64)) & 1
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        means = np.mean(signs * differences[None, :], axis=1)
        method = "exact_paired_sign_flip"
    else:
        rng = _rng(seed, f"{key}|sign-flip")
        signs = rng.choice((-1.0, 1.0), size=(replicates, task_count))
        signs[0, :] = 1.0  # identity is included; all remaining draws are independent.
        total = replicates
        means = np.mean(signs * differences[None, :], axis=1)
        method = "seeded_monte_carlo_paired_sign_flip"
    observed = abs(float(np.mean(differences)))
    tie_tolerance = 1e-9 * max(1.0, float(np.sum(np.abs(differences))))
    extreme = int(np.count_nonzero(np.abs(means) >= observed - tie_tolerance))
    return {
        "bootstrap_p": extreme / total,
        "sign_flip_method": method,
        "sign_flip_extreme": extreme,
        "sign_flip_total": total,
        "tie_tolerance_mm_or_metric_units": tie_tolerance,
    }


def _paired_stats(
    method: Sequence[float],
    baseline: Sequence[float],
    families: Sequence[str],
    key: str,
    seed: int,
    replicates: int,
    confidence: float,
) -> dict[str, Any]:
    if len(method) != len(baseline) or len(method) != len(families):
        raise AuditInputError("paired data length mismatch")
    if not method:
        return {
            "paired_tasks": 0,
            "method_mean": None,
            "baseline_mean": None,
            "mean_difference": None,
            "interval": None,
            "interval_null_reason": "no paired independent tasks",
            "ci_resampling_method": None,
            "bootstrap_p": None,
            "sign_flip_method": None,
            "sign_flip_extreme": None,
            "sign_flip_total": None,
            "small_n_warning": "no paired independent tasks",
            "holm_p": None,
        }
    method_values = np.asarray(method, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    differences = method_values - baseline_values
    observed = float(np.mean(differences))
    sufficient_tasks = len(method) >= 2
    interval = None
    ci_resampling_method = None
    if sufficient_tasks:
        distribution = _resampled_mean_distribution(
            differences, families, _rng(seed, f"{key}|task-bootstrap-ci"), replicates)
        interval = _interval(distribution, confidence)
        ci_resampling_method = _bootstrap_design(differences, families)[1]
    randomization = _sign_flip_p_value(differences, key, seed, replicates)
    return {
        "paired_tasks": len(method),
        "method_mean": float(np.mean(method_values)),
        "baseline_mean": float(np.mean(baseline_values)),
        "mean_difference": observed,
        "interval": interval,
        "interval_null_reason": (
            None if sufficient_tasks else "insufficient independent tasks (need at least 2)"),
        "ci_resampling_method": ci_resampling_method,
        **randomization,
        "small_n_warning": (
            "fewer than 5 independent tasks; treat inference as descriptive" if len(method) < 5 else None),
        "holm_p": None,
    }


def _degenerate(differences: Sequence[float]) -> Optional[dict[str, Any]]:
    """Use exact statistics outside hypothesis tests (currently a reserved helper)."""
    return None


def clopper_pearson(successes: int, total: int, confidence: float = 0.95) -> Optional[dict[str, float]]:
    """Exact binomial interval; callers must establish independent binary units."""
    if not 0.0 < confidence < 1.0:
        raise AuditInputError("confidence must be between 0 and 1")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise AuditInputError("total must be a non-negative integer")
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise AuditInputError("successes must be an integer")
    if total <= 0:
        return None
    if successes < 0 or successes > total:
        raise AuditInputError("Clopper-Pearson requires 0 <= successes <= total")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return {"confidence": confidence, "lower": lower, "upper": upper}


def _rate_summary(
    collapsed: Mapping[str, Mapping[str, Any]],
    task_ids: Sequence[str],
    tasks: Mapping[str, TaskInfo],
    key: str,
    seed: int,
    replicates: int,
    confidence: float,
) -> dict[str, Any]:
    known_ids = [task_id for task_id in task_ids if collapsed.get(task_id, {}).get("value") is not None]
    values = [float(collapsed[task_id]["value"]) for task_id in known_ids]
    families = [tasks[task_id].family for task_id in known_ids]
    point = float(np.mean(values)) if values else None
    single_binary = bool(values) and all(
        collapsed[task_id]["known_observations"] == 1
        and collapsed[task_id]["unknown_or_missing_observations"] == 0
        and collapsed[task_id]["value"] in (0.0, 1.0)
        for task_id in known_ids
    )
    if single_binary:
        successes = int(round(sum(values)))
        interval = clopper_pearson(successes, len(values), confidence)
        method = "clopper_pearson_independent_tasks"
        resampling_method = "clopper_pearson_exact_binomial"
    elif values:
        interval = _interval(
            _resampled_mean_distribution(values, families, _rng(seed, key), replicates),
            confidence,
        )
        method = "cluster_bootstrap_over_tasks"
        resampling_method = _bootstrap_design(np.asarray(values), families)[1]
    else:
        interval = None
        method = "unavailable"
        resampling_method = None
    return {
        "estimate_method": method,
        "resampling_method": resampling_method,
        "expected_task_units": len(task_ids),
        "known_task_units": len(values),
        "unknown_or_missing_task_units": len(task_ids) - len(values),
        "unknown_or_missing_observations": sum(
            collapsed.get(task_id, {}).get("unknown_or_missing_observations", 0)
            for task_id in task_ids
        ),
        "task_mean_rate": point,
        "successes": int(round(sum(values))) if single_binary else None,
        "interval": interval,
    }


def _apply_holm(rows: Iterable[Mapping[str, Any]]) -> None:
    valid = [(index, row) for index, row in enumerate(rows) if row.get("bootstrap_p") is not None]
    ordered = sorted(valid, key=lambda item: (item[1]["bootstrap_p"], item[0]))
    running = 0.0
    count = len(ordered)
    for rank, (_, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * row["bootstrap_p"])
        running = max(running, adjusted)
        row["holm_p"] = float(running)


def _metric_drop_counts(task_ids: Sequence[str], method_values, baseline_values) -> dict[str, Any]:
    failed_or_ineligible_reasons = {
        "recorded_noncomplete", "incomplete_or_unsupported_coverage",
        "not_efficiency_eligible", "efficiency_eligibility_unknown",
    }
    exclusions_by_side = {}
    failed_total = 0
    missing_total = 0
    for side, values in (("method", method_values), ("baseline", baseline_values)):
        counts: Counter[str] = Counter()
        for task_id in task_ids:
            counts.update(values[task_id]["exclusions"])
        exclusions_by_side[side] = dict(sorted(counts.items()))
        failed_total += sum(counts[reason] for reason in failed_or_ineligible_reasons)
        missing_total += sum(count for reason, count in counts.items()
                             if reason not in failed_or_ineligible_reasons)
    return {
        "dropped_because_side_failed_or_ineligible": failed_total,
        "dropped_missing_or_null_evidence": missing_total,
        "exclusions_by_side": exclusions_by_side,
    }


def _comparison_rows(
    spec: ComparisonSpec,
    family: Optional[str],
    manifest: Manifest,
    audits: Mapping[str, RunAudit],
    seed: int,
    replicates: int,
    confidence: float,
    verdict_evaluations=None,
) -> dict[str, Any]:
    allowed_runs = [
        run for run in manifest.runs.values()
        if family is None or manifest.tasks[run.task_id].family == family
    ]
    if spec.families is not None:
        allowed_runs = [
            run for run in allowed_runs
            if manifest.tasks[run.task_id].family in spec.families
        ]
    method_runs = [run for run in allowed_runs if run.method == spec.method]
    baseline_runs = [run for run in allowed_runs if run.method == spec.baseline]
    task_ids = sorted(
        {run.task_id for run in method_runs} & {run.task_id for run in baseline_runs}
    )
    row: dict[str, Any] = {
        "comparison": spec.name,
        "method": spec.method,
        "baseline": spec.baseline,
        "dialect": spec.dialect.value,
        "family": family if family is not None else "overall",
        "expected_paired_tasks": len(task_ids),
        "metrics": {},
        "outcomes": {},
        "_p_value_rows": [],
    }
    key_prefix = f"{spec.name}|{family or 'overall'}|{spec.dialect.value}"
    for metric_name in spec.metrics:
        method_values = _metric_by_task(
            method_runs, audits, spec.dialect, metric_name, manifest.tasks, verdict_evaluations)
        baseline_values = _metric_by_task(
            baseline_runs, audits, spec.dialect, metric_name, manifest.tasks, verdict_evaluations)
        methods: list[float] = []
        baselines: list[float] = []
        families: list[str] = []
        for task_id in task_ids:
            method_value = method_values.get(task_id, {}).get("value")
            baseline_value = baseline_values.get(task_id, {}).get("value")
            if method_value is not None and baseline_value is not None:
                methods.append(method_value)
                baselines.append(baseline_value)
                families.append(manifest.tasks[task_id].family)
        summary = _paired_stats(
            methods, baselines, families, f"{key_prefix}|metric|{metric_name}",
            seed, replicates, confidence,
        )
        summary["efficiency_eligibility_required"] = metric_name in EFFICIENCY_METRICS
        summary.update(_metric_drop_counts(task_ids, method_values, baseline_values))
        summary["method_task_availability"] = {
            task_id: method_values[task_id] for task_id in task_ids
        }
        summary["baseline_task_availability"] = {
            task_id: baseline_values[task_id] for task_id in task_ids
        }
        row["metrics"][metric_name] = summary
        row["_p_value_rows"].append(summary)

    for outcome in OUTCOMES:
        method_outcomes = _outcome_by_task(
            method_runs, audits, spec.dialect, outcome, manifest.tasks, verdict_evaluations
        )
        baseline_outcomes = _outcome_by_task(
            baseline_runs, audits, spec.dialect, outcome, manifest.tasks, verdict_evaluations
        )
        methods: list[float] = []
        baselines: list[float] = []
        families: list[str] = []
        for task_id in task_ids:
            method_value = method_outcomes.get(task_id, {}).get("value")
            baseline_value = baseline_outcomes.get(task_id, {}).get("value")
            if method_value is not None and baseline_value is not None:
                methods.append(method_value)
                baselines.append(baseline_value)
                families.append(manifest.tasks[task_id].family)
        difference = _paired_stats(
            methods, baselines, families, f"{key_prefix}|difference|{outcome}",
            seed, replicates, confidence,
        )
        row["_p_value_rows"].append(difference)
        row["outcomes"][outcome] = {
            "method": _rate_summary(
                method_outcomes, task_ids, manifest.tasks,
                f"{key_prefix}|method-rate|{outcome}", seed, replicates, confidence,
            ),
            "baseline": _rate_summary(
                baseline_outcomes, task_ids, manifest.tasks,
                f"{key_prefix}|baseline-rate|{outcome}", seed, replicates, confidence,
            ),
            "paired_difference": difference,
        }
    return row


def aggregate_evaluations(
    manifest: Manifest,
    audits: Mapping[str, RunAudit],
    *,
    bootstrap_seed: int = 20261108,
    bootstrap_replicates: int = 2000,
    confidence: float = 0.95,
    verdict_evaluations=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise AuditInputError("bootstrap_seed must be an integer")
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 100:
        raise AuditInputError("bootstrap_replicates must be an integer >= 100")
    if not 0.0 < confidence < 1.0:
        raise AuditInputError("confidence must be between 0 and 1")
    validated_outcomes = _validate_verdict_evaluations(verdict_evaluations, manifest)
    rows: list[dict[str, Any]] = []
    p_value_rows: list[dict[str, Any]] = []
    for spec in manifest.comparisons:
        families = sorted(
            {task.family for task in manifest.tasks.values()}
            if spec.families is None else set(spec.families)
        )
        for family in ([None] if len(families) == 1 else [*families, None]):
            row = _comparison_rows(
                spec, family, manifest, audits, bootstrap_seed,
                bootstrap_replicates, confidence, validated_outcomes,
            )
            p_value_rows.extend(row.pop("_p_value_rows"))
            rows.append(row)
    _apply_holm(p_value_rows)
    policy = {
        "independent_unit": "task",
        "seed_aggregation": "mean within each task/method before pairing",
        "pairing": "common task and dialect; method minus baseline",
        "bootstrap": "percentile task-cluster bootstrap; family-stratified only when every stratum has >=2 tasks, otherwise pooled over tasks (singleton strata cannot vary independently)",
        "recorded_noncomplete": "a valid non-complete run is a known zero for safety, completion and efficiency eligibility; continuous metrics exclude it with explicit per-comparison drop counts",
        "auditor_flags": "auditor_passed is mechanical validity only; evidence_complete means every expected run has a unique cleanly audited record including valid failed/blocked statuses; all_runs_completed additionally requires RunStatus.COMPLETE for every expected run",
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence": confidence,
        "p_value": "two-sided paired sign-flip randomization (exact enumeration through 16 independent tasks, seeded Monte Carlo above including identity); Holm across every comparison/metric/outcome and family row",
        "interval_rule": "task-cluster bootstrap CI with at least two independent tasks; one paired task yields null CI with insufficient-independent-tasks reason",
        "small_sample_rule": "n < 5 paired tasks is labelled descriptive; no tight interval or minimum attainable randomization p is presented as strong evidence",
        "clopper_pearson": "only for observed binary outcomes with exactly one known observation per independent task; otherwise task-cluster bootstrap",
        "efficiency_metrics": sorted(EFFICIENCY_METRICS),
        "efficiency_rule": "efficiency metrics require declared eligibility plus established completion (and positive removal for removal tasks); both sides need a finite metric",
        "completion_rule": "residual <= task completion_residual_mm3 and positive removed volume when removal_required; no-cut fails completion even if residual is zero",
        "missing_data": "never treated as success; missing/invalid/failed/blocked and null statuses are counted explicitly",
    }
    return rows, policy


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _interval_text(interval: Optional[Mapping[str, Any]]) -> str:
    if interval is None:
        return "NA"
    return f"[{_fmt(interval['lower'])}, {_fmt(interval['upper'])}]"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Evaluation pilot table",
        "",
        "Seeds are averaged within each task before pairing. Bootstrap intervals "
        "resample tasks within family strata only when every stratum has at least two tasks; "
        "otherwise they pool tasks. No-cut runs are not efficiency eligible. "
        "Recorded failed/interrupted/blocked runs are known zero outcomes, never invisible successes. "
        "Missing/invalid/blocked/null evidence is not converted into success. "
        "Δ = method − baseline. Negative differences are retained as reported.",
        "",
        f"**Holm policy:** {report['policy']['p_value']}",
        "",
        "## Metric comparisons",
        "",
        "| Comparison | Family | Dialect | Metric | Paired tasks | Method | Baseline | Δ | 95% CI | Failed/ineligible drops | Holm p |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in report["comparisons"]:
        for metric, summary in row["metrics"].items():
            lines.append(
                "| {comparison} | {family} | {dialect} | {metric} | {n} | {method} | {baseline} | {delta:.6g} | {interval} | {dropped} | {p} |".format(
                    comparison=str(row["comparison"]).replace("|", "\\|"),
                    family=row["family"], dialect=row["dialect"], metric=metric,
                    n=summary["paired_tasks"], method=_fmt(summary["method_mean"]),
                    baseline=_fmt(summary["baseline_mean"]),
                    delta=summary["mean_difference"] if summary["mean_difference"] is not None else float("nan"),
                    interval=_interval_text(summary["interval"]),
                    dropped=summary["dropped_because_side_failed_or_ineligible"],
                    p=_fmt(summary["holm_p"]),
                ).replace("nan", "NA") if summary["mean_difference"] is None else
                "| {comparison} | {family} | {dialect} | {metric} | {n} | {method} | {baseline} | {delta} | {interval} | {dropped} | {p} |".format(
                    comparison=str(row["comparison"]).replace("|", "\\|"),
                    family=row["family"], dialect=row["dialect"], metric=metric,
                    n=summary["paired_tasks"], method=_fmt(summary["method_mean"]),
                    baseline=_fmt(summary["baseline_mean"]),
                    delta=_fmt(summary["mean_difference"]),
                    interval=_interval_text(summary["interval"]),
                    dropped=summary["dropped_because_side_failed_or_ineligible"],
                    p=_fmt(summary["holm_p"]),
                )
            )
    lines.extend(
        [
            "",
            "## Task-level outcomes",
            "",
            "Clopper–Pearson is used only for one observed binary result per independent task; "
            "replicated seeds or fractional task outcomes use task-cluster bootstrap.",
            "",
            "| Comparison | Family | Outcome | Side | Known/expected tasks | Task mean rate | Rate CI | Paired Δ | Δ CI | Holm p |",
            "|---|---|---|---|---:|---:|---|---:|---|---:|",
        ]
    )
    for row in report["comparisons"]:
        for outcome, summary in row["outcomes"].items():
            difference = summary["paired_difference"]
            for side in ("method", "baseline"):
                rate = summary[side]
                lines.append(
                    f"| {str(row['comparison']).replace('|', chr(92) + '|')} | {row['family']} | {outcome} | {side} | "
                    f"{rate['known_task_units']}/{rate['expected_task_units']} | "
                    f"{_fmt(rate['task_mean_rate'])} | {_interval_text(rate['interval'])} | "
                    f"{_fmt(difference['mean_difference']) if side == 'method' else '—'} | "
                    f"{_interval_text(difference['interval']) if side == 'method' else '—'} | "
                    f"{_fmt(difference['holm_p']) if side == 'method' else '—'} |"
                )
    reconciliation = report["expected_vs_observed"]
    lines.extend(
        [
            "",
            "## Evidence audit",
            "",
            f"- auditor_passed (mechanical artifact validity, not a success requirement): {report['summary']['auditor_passed']}",
            f"- evidence_complete (valid expected records present, including recorded failures): {report['evidence_complete']}",
            f"- all_runs_completed (every expected RunStatus is COMPLETE): {report['all_runs_completed']}",
            f"- Expected IDs reconciled: {reconciliation['exact_id_reconciliation']}",
            f"- Missing expected run IDs: {', '.join(reconciliation['missing_run_ids']) or 'none'}",
            f"- Unexpected run IDs: {', '.join(reconciliation['unexpected_run_ids']) or 'none'}",
            f"- Duplicate run IDs: {', '.join(reconciliation['duplicate_run_ids']) or 'none'}",
            f"- Run status counts: `{json.dumps(report['summary']['run_status_counts'], sort_keys=True)}`",
            f"- Dialect status counts: `{json.dumps(report['summary']['dialect_status_counts'], sort_keys=True)}`",
            f"- Issue counts: `{json.dumps(report['summary']['issue_counts'], sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(
    runs_root: str | Path,
    manifest_value: Mapping[str, Any] | str | Path | Manifest,
    *,
    bootstrap_seed: int = 20261108,
    bootstrap_replicates: int = 2000,
    confidence: float = 0.95,
    verdict_evaluations=None,
    screen_disagreements=None,
) -> tuple[dict[str, Any], Manifest]:
    manifest = manifest_value if isinstance(manifest_value, Manifest) else load_manifest(manifest_value)
    fine = _validate_verdict_evaluations(verdict_evaluations, manifest)
    diagnostics = {}
    if fine is not None:
        allowed = {"safety_pass", "geometry_pass", "completion_pass", "efficiency_eligible", "coverage"}
        if not isinstance(screen_disagreements, Mapping) or set(screen_disagreements) != set(fine):
            raise AuditInputError("fine verdict evidence requires screen diagnostics for every run")
        for run_id, fields in screen_disagreements.items():
            if (not isinstance(fields, (list, tuple, set)) or isinstance(fields, (str, bytes))
                    or len(set(fields)) != len(fields) or not set(fields) <= allowed):
                raise AuditInputError(f"invalid screen diagnostic fields for {run_id}")
            diagnostics[run_id] = sorted(fields)
    run_reports, audit = audit_runs(runs_root, manifest)
    if fine and not audit["summary"]["auditor_passed"]:
        raise AuditInputError("fine verdict reporting requires a clean primary screen audit")
    for row in run_reports:
        if fine is not None:
            expected = manifest.runs[row["run_id"]]
            row["verdict_evaluations"] = {}
            for dialect, evaluation in fine[row["run_id"]].items():
                task_info = manifest.tasks[expected.task_id]
                row["verdict_evaluations"][dialect] = {
                    "evaluator_revision": evaluation.evaluator_revision,
                    "nc_hash": evaluation.nc_hash,
                    "coverage": evaluation.coverage.value,
                    "safety_pass": evaluation.safety_pass,
                    "geometry_pass": evaluation.geometry_pass,
                    "completion_pass": _completion_value(evaluation, task_info),
                    "efficiency_eligible": _outcome_value(evaluation, task_info, "efficiency_eligible"),
                    "max_step_mm": evaluation.discretization["settings"]["max_step_mm"],
                }
    comparisons, policy = aggregate_evaluations(
        manifest, audit.pop("_audits_by_id"),
        bootstrap_seed=bootstrap_seed, bootstrap_replicates=bootstrap_replicates,
        confidence=confidence, verdict_evaluations=fine,
    )
    report = {
        "schema_version": REPORT_SCHEMA, "runs_root": str(runs_root),
        "task_count": len(manifest.tasks), "expected_run_count": len(manifest.runs),
        "expected_vs_observed": audit["reconciliation"], "summary": audit["summary"],
        "evidence_complete": audit["summary"]["evidence_complete"],
        "all_runs_completed": audit["summary"]["all_runs_completed"],
        "policy": policy, "runs": run_reports, "comparisons": comparisons,
    }
    if fine is not None:
        counts = Counter({manifest.runs[run_id].method: 0 for run_id in fine})
        for run_id, fields in diagnostics.items():
            if fields:
                counts[manifest.runs[run_id].method] += 1
        report["refinement"] = {
            "producer": "both 0.125 mm posted evaluations joined to the primary screen audit by run ID",
            "verdict_source": {"max_step_mm": 0.125, "dialects": [item.value for item in C.Dialect]},
            "screen_role": "1.0 mm fidelity diagnostic only; never ANDed into pilot verdicts",
            "convergence_point": {"max_step_mm": 0.25, "dialect": C.Dialect.RS274.value,
                                  "role": "evidence only; no monotonicity/error-bound or verdict claim"},
            "screen_disagreements_by_run_id": diagnostics,
            "screen_disagreement_by_condition": {
                method: {"count": count} for method, count in sorted(counts.items())},
            "applied_run_ids": sorted(fine),
        }
    json.dumps(report, allow_nan=False)
    return report, manifest
