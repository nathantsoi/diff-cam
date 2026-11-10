#!/usr/bin/env python3
"""Independent-reference qualification of selected RustCNC nominal motions.

This runs the existing NumPy reference and target/stock-mask checks against an
explicit generated candidate. It does NOT post G-code, run an optimizer, or
operate equipment. A subset report is not cohort qualification.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from benchmarks import procedural_suite as suite
from cam import rustcnc_seed_adapter as adapter


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", action="append", default=[],
                        help="select an exact registered task id (repeatable)")
    parser.add_argument("--family", action="append", default=[],
                        help="select all members of a declared family (repeatable)")
    parser.add_argument("--output", required=True, type=Path,
                        help="exclusive JSON report path; never overwritten")
    args = parser.parse_args(argv)
    if not args.task_id and not args.family:
        parser.error("select at least one --task-id or --family")
    if args.output.exists():
        parser.error(f"output already exists; refusing to overwrite: {args.output}")
    selected = [bundle for bundle in suite.generate_tasks()
                if bundle.task.task_id in args.task_id or bundle.task.family in args.family]
    requested = set(args.task_id)
    found = {bundle.task.task_id for bundle in selected}
    if requested - found:
        raise ValueError(f"unregistered task ids: {sorted(requested - found)}")
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for original in selected:
        task = original.task
        print(f"GENERATING {task.task_id}", flush=True)
        started_one = datetime.now(timezone.utc).isoformat()
        try:
            motion = adapter.cam_seed_for_task(task)
            replaced = dataclasses.replace(original, motion=motion)
            assessment = suite._reference_qualify_seed(
                replaced, seed_factory=adapter.cam_seed_for_task)
            row = {
                "task_id": task.task_id, "task_hash": task.task_hash,
                "motion_hash": motion.motion_hash,
                "strategy_provenance": json.loads(motion.provenance),
                "assessment": assessment,
            }
            print(f"QUALIFIED REFERENCE {task.task_id}", flush=True)
        except Exception as error:
            row = {
                "task_id": task.task_id, "task_hash": task.task_hash,
                "status": "not_qualified_or_incomplete",
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(),
            }
            print(f"NOT QUALIFIED {task.task_id}: {type(error).__name__}: {error}",
                  flush=True)
        row["started_utc"] = started_one
        row["finished_utc"] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
    report = {
        "schema": "diffcam-rustcnc-reference-qualification/1",
        "status": "qualified_reference_subset" if all("assessment" in row for row in rows)
                  else "not_qualified_or_incomplete",
        "scope": "explicit selected tasks only; independent reference; no posted G-code, optimization, or equipment run",
        "task_count_requested": len(selected),
        "task_count_qualified": sum("assessment" in row for row in rows),
        "seed_generation_version": adapter.SEED_GENERATION_VERSION,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(suite.PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty_state": subprocess.check_output(
            ["git", "-C", str(suite.PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"],
            text=True).strip(),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"REPORT {args.output} status={report['status']} "
          f"qualified={report['task_count_qualified']}/{len(rows)}", flush=True)
    return 0 if report["task_count_qualified"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
