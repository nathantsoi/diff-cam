#!/usr/bin/env python3
"""Read-only progress from canonical campaign manifests and typed run records.

No optimizer, geometry, or physical execution is performed here. Running data
are not a transactional snapshot. Missing or invalid selection metadata never
becomes "improved", zero failure, a passed simulation, or a completed campaign.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import audit as A
from benchmarks import contracts as C


class StatusError(RuntimeError):
    """A manifest, record identity, or JSON object is unusable."""


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes(), parse_constant=lambda text: (
            _ for _ in ()).throw(ValueError(f"non-finite JSON value {text}")))
    except (OSError, UnicodeError, ValueError) as error:
        raise StatusError(f"cannot parse {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise StatusError(f"{path.name} must contain a JSON object")
    return value


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _init_for(run_id: str, root: Path, expected: Mapping[str, Any], issues: list[dict[str, str]]) -> str:
    """Prefer the root manifest, then the exact per-run config, not the filename."""
    configured = expected.get("init", "cam")
    path = root / run_id / "config.json"
    if not path.is_file():
        return str(configured)
    try:
        document = _json(path)
        initialization = document.get("initialization")
        if isinstance(initialization, Mapping) and isinstance(initialization.get("init"), str):
            value = initialization["init"]
            if value in ("cam", "straight_line", "random"):
                return value
            return "unknown"
        return "cam"  # A legacy config without that field is CAM-only by contract.
    except StatusError:
        issues.append({"kind": "invalid.config.json", "run_id": run_id, "message": path.name})
        return "unknown"


def _record_for(run_id: str, directory: Path, expected, task_hash: str) -> tuple[str, C.RunRecord]:
    try:
        record = C.loads((directory / "run.json").read_bytes().decode("utf-8"))
    except Exception as error:
        raise StatusError(f"{run_id}: invalid run.json: {error}") from error
    if not isinstance(record, C.RunRecord) or record.run_id != run_id:
        raise StatusError(f"{run_id}: incorrect run ID or record type")
    if record.task_hash != task_hash:
        raise StatusError(f"{run_id}: task hash differs from the manifest")
    if record.status is C.RunStatus.COMPLETE:
        return "complete", record
    if record.status in (C.RunStatus.FAILED, C.RunStatus.BLOCKED, C.RunStatus.INTERRUPTED):
        return "failed", record
    if record.status is C.RunStatus.INCOMPLETE:
        return "pending", record
    raise StatusError(f"{run_id}: unsupported record status {record.status!r}")


def _selection_for(run_id: str, directory: Path, method: str) -> tuple[Any, str | None]:
    """Consume only completion's explicit selection flag; nominal is not optimizer evidence."""
    if method == "nominal-cam":
        return None, "nominal CAM has no optimizer-selection flag"
    path = directory / "metrics.json"
    if not path.is_file():
        return None, "missing.metrics.json"
    try:
        selection = _json(path).get("selection")
    except StatusError:
        raise StatusError(f"{run_id}: invalid selection metadata in metrics.json")
    if not isinstance(selection, Mapping):
        return None, "no optimizer-selection record"
    initial = selection.get("initial_candidate")
    if initial is not None and (not isinstance(initial, Mapping) or initial.get("valid") is not True):
        return None, "initial candidate unavailable or not a valid candidate"
    value = selection.get("no_improvement_over_initialization")
    if not isinstance(value, bool):
        return None, "no explicit boolean no-improvement flag"
    return value, "selection.no_improvement_over_initialization"


def _observe(root: Path) -> dict[str, list[Path]]:
    parents = [root]
    tasks = root / "tasks"
    if tasks.is_dir():
        parents.extend(sorted(path for path in tasks.iterdir() if path.is_dir()))
    found: dict[str, list[Path]] = collections.defaultdict(list)
    for parent in parents:
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "run.json").is_file():
                found[child.name].append(child)
    return found


def _counts() -> dict[str, int]:
    return {"expected": 0, "complete": 0, "failed": 0, "pending": 0, "invalid": 0}


def _rate(bucket: Mapping[str, Any]) -> float | None:
    known = bucket["no_improvement"]["known"]
    return bucket["no_improvement"]["true"] / known if known else None


def _add_flag(bucket: dict[str, Any], flag: Any) -> None:
    if isinstance(flag, bool):
        bucket["no_improvement"]["known"] += 1
        key = "true" if flag else "false"
        bucket["no_improvement"][key] += 1
    else:
        bucket["no_improvement"]["unknown"] += 1


def campaign_status(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise StatusError(f"campaign root is not a directory: {root}")
    if (root / "audit-runs").is_dir() and not (root / "manifest.json").is_file():
        root = root / "audit-runs"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise StatusError(f"campaign manifest is missing: {manifest_path}")
    try:
        manifest = A.load_manifest(manifest_path)
    except (A.AuditInputError, ValueError, TypeError, OSError) as error:
        raise StatusError(f"invalid campaign manifest: {error}") from error
    if not manifest.runs:
        raise StatusError("campaign manifest contains no expected runs")
    raw_manifest = _json(manifest_path)
    source_metadata = raw_manifest.get("campaign", {}).get("source_metadata")
    if not isinstance(source_metadata, dict):
        source_metadata = None

    by_run_init = {
        row["run_id"]: row.get("init", "cam")
        for row in raw_manifest.get("initialization_cases", [])
        if isinstance(row, dict) and row.get("run_id") in manifest.runs
    }
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    group_by_run: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    for expected in manifest.runs.values():
        init = _init_for(expected.run_id, root, {
            "method": expected.method, "init": by_run_init.get(expected.run_id, "cam")
        }, issues)
        key = (expected.method, init)
        if key not in groups:
            groups[key] = _counts()
            groups[key]["no_improvement"] = {"true": 0, "false": 0, "unknown": 0, "known": 0}
            groups[key]["no_improvement_cases"] = []
        group_by_run[expected.run_id] = groups[key]
        groups[key]["expected"] += 1

    totals = _counts()
    totals["flags"] = {"true": 0, "false": 0, "unknown": 0, "known": 0}
    observations = _observe(root)
    for run_id in manifest.runs.keys() - observations.keys():
        group = group_by_run[run_id]
        group["pending"] += 1
    for run_id, directories in sorted(observations.items()):
        if run_id not in manifest.runs:
            for directory in directories:
                issues.append({"kind": "unexpected_run", "run_id": run_id,
                               "message": str(directory.relative_to(root))})
            continue
        expected = manifest.runs[run_id]
        init = _init_for(run_id, root, {
            "method": expected.method, "init": by_run_init.get(run_id, "cam")}, issues)
        key = (expected.method, init)
        if key not in groups:
            raise StatusError(f"{run_id}: init declaration differs from frozen manifest/config")
        group = groups[key]
        if len(directories) != 1:
            for directory in directories:
                issues.append({"kind": "duplicate_run_id", "run_id": run_id,
                               "message": str(directory.relative_to(root))})
            group["invalid"] += 1
            continue
        directory = directories[0]
        try:
            status, record = _record_for(run_id, directory, expected,
                                         manifest.tasks[expected.task_id].task_hash)
        except StatusError as error:
            issues.append({"kind": "invalid.run.json", "run_id": run_id,
                           "message": str(error)})
            group["invalid"] += 1
            continue
        if status == "complete" and expected.method != "nominal-cam":
            try:
                flag, source = _selection_for(run_id, directory, expected.method)
                if source == "missing.metrics.json":
                    issues.append({"kind": source, "run_id": run_id,
                                   "message": "completed optimizer run lacks metrics.json"})
                _add_flag(group, flag)
                if flag is True:
                    group["no_improvement_cases"].append({
                        "run_id": run_id, "source_commit": record.source_commit,
                        "dirty_patch_hash": record.dirty_patch_hash,
                        "selection_source": source})
            except StatusError as error:
                issues.append({"kind": "invalid.metrics.json", "run_id": run_id,
                               "message": str(error)})
                group["no_improvement"]["unknown"] += 1
        elif status != "complete":
            # Pending/failed/interrupted cases cannot update the selection ratio.
            group["no_improvement"]["unknown"] += 1
        if status != "complete":
            issues.append({"kind": status, "run_id": run_id,
                           "message": record.failure_reason or status.value})
        group[status] += 1

    for group in groups.values():
        for status in ("expected", "complete", "failed", "pending", "invalid"):
            totals[status] += group[status]
        for flag_state in ("true", "false", "unknown", "known"):
            totals["flags"][flag_state] += group["no_improvement"][flag_state]

    conditions = []
    for (method, init), group in sorted(groups.items()):
        condition = {key: group[key] for key in ("expected", "complete", "failed", "pending", "invalid")}
        condition.update({
            "method": method, "init": init,
            "no_improvement_count": group["no_improvement"]["true"],
            "no_improvement_known": group["no_improvement"]["known"],
            "no_improvement_unknown": group["no_improvement"]["unknown"],
            "no_improvement_rate": _rate(group),
            "no_improvement_cases": group.get("no_improvement_cases", []),
        })
        conditions.append(condition)
    total_counts = {"complete": totals["complete"], "failed": totals["failed"],
                    "pending": totals["pending"], "invalid": totals["invalid"]}
    missing = max(0, len(manifest.runs) - sum(total_counts.values()))
    issues = [item for _key, item in sorted(
        {(entry["kind"], entry["run_id"], entry["message"]): entry
         for entry in issues}.items())]
    return {
        "scope": "early provisional progress, not a campaign outcome",
        "source_metadata": source_metadata,
        "schema": "diffcam-campaign-status/1",
        "input_root": str(root),
        "manifest_sha256": _sha(manifest_path),
        "expected_runs": len(manifest.runs),
        "counts": {"expected": totals["expected"], "complete": totals["complete"],
                   "failed": totals["failed"], "pending": totals["pending"],
                   "invalid": totals["invalid"], "missing": missing},
        "conditions": conditions,
        "no_improvement": {
            "numerator": totals["flags"]["true"],
            "denominator": totals["flags"]["known"],
            "value": _rate({"no_improvement": {"true": totals["flags"]["true"],
                                               "known": totals["flags"]["known"]}}),
            "unknown": totals["flags"]["unknown"],
        },
        "provisional": True,
        "notes": [
            "Progress counts are read from typed run.json files, not from a performance aggregate.",
            "No-improvement rates only use complete runs with explicit selection flags.",
            "Unexpected, duplicate, or malformed records are reported, not silently accepted as valid.",
            "This directory is a read-only input; it is never modified by this utility.",
        ],
        "issues": issues,
    }


def _format_human(document: Mapping[str, Any]) -> str:
    counts = document["counts"]
    lines = ["EARLY provisional progress — not a campaign, geometry, or effectiveness result.",
             f"Expected {document['expected_runs']}: complete {counts['complete']}; "
             f"failed {counts['failed']}; pending {counts['pending']}; "
             f"invalid {counts['invalid']}; missing {counts['missing']}.",
             "condition/init | complete | failed | pending | invalid | no-improvement / known"]
    for row in document["conditions"]:
        rate = row["no_improvement_rate"]
        rate_text = f"{rate:.6g}" if rate is not None else "n/a"
        lines.append(f"{row['method']}/{row['init']} | {row['complete']} | {row['failed']} | "
                     f"{row['pending']} | {row['invalid']} | {row['no_improvement_count']}/"
                     f"{row['no_improvement_known']} ({rate_text})")
    if document["issues"]:
        lines.append(f"Issues: {len(document['issues'])}; shown explicitly, not converted to pass/zero improvement.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="campaign output root (read-only; never modified)")
    parser.add_argument("--json", action="store_true", help="emit canonical JSON")
    parser.add_argument("--strict", action="store_true",
                        help="return 2 for malformed, duplicate or unexpected records")
    args = parser.parse_args(argv)
    try:
        result = campaign_status(args.root)
    except (StatusError, A.AuditInputError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(_format_human(result), file=sys.stdout)
    if args.strict and result["issues"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
