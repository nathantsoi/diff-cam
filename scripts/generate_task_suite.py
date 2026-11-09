#!/usr/bin/env python3
"""Generate the bounded T2.3 families and qualify their nominal seeds.

``--verify-only`` checks deterministic manifest reconstruction only. Normal
execution also writes the identical manifest and performs the actual two-post
production evaluation plus independent reference check for each task. Artifact
case directories and the report are never overwritten. There is no option that
marks an unverified or failing seed as qualified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

from benchmarks import contracts as C
from benchmarks import procedural_suite as suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=suite.PROJECT_ROOT / "configs" / "task_manifest.json",
        help="deterministic task-definition manifest (default: tracked T2.3 manifest)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=suite.PROJECT_ROOT / "runs" / "procedural-bootstrap",
        help="parent for new per-task nominal artifact directories",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="only compare an existing manifest with deterministic reconstruction; no evaluation or writes",
    )
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="construct tasks and emit/check only the manifest; do not qualify seeds",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_only and args.manifest_only:
        _parser().error("--verify-only and --manifest-only are mutually exclusive")
    if args.verify_only:
        suite.verify_manifest(args.manifest)
        print(f"manifest reproduced byte-for-byte: {args.manifest}")
        return 0

    # Do not use an unchecked old manifest as the input to generation. A change
    # in bounded task definitions must be reviewed, not silently overwritten.
    generated = suite.manifest_text()
    suite.write_manifest(args.manifest, expected_text=generated)
    if args.manifest_only:
        print(f"manifest constructed; seed qualification NOT run: {args.manifest}")
        return 0

    bundles = suite.generate_tasks()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "generation_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite generation report: {report_path}")
    report = {
        "schema_version": "t2.3-generate-report/1",
        "generation_version": suite.GENERATION_VERSION,
        "generator_seed": suite.GENERATOR_SEED,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "requested_tasks": [bundle.task.task_id for bundle in bundles],
        "qualification": {
            "production_dialects": [dialect.value for dialect in C.Dialect],
            "independent_reference": "typed-motion stock/chronology; custom target geometry is not independently scored",
            "max_step": "cutter_radius_mm / 2",
            "exclusion_tolerance_mm3": 0.0,
        },
        "tasks": [],
    }
    failed = False
    for bundle in bundles:
        task_id = bundle.task.task_id
        started = time.perf_counter()
        try:
            records = suite.verify_all((bundle,), output_root=root)
        except Exception as error:
            # Continue independent cases instead of hiding the rest behind one
            # bad seed. This diagnostic is not a qualified RunRecord.
            failed = True
            record = {
                "task_id": task_id,
                "task_hash": bundle.task.task_hash,
                "status": "failed_qualification",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            print(f"UNQUALIFIED {task_id}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        else:
            record = {
                "task_id": task_id,
                "task_hash": bundle.task.task_hash,
                "status": "qualified_and_retained",
                "run_id": records[0].run_id,
                "run_json": str(root / task_id / "run.json"),
                "motion_hash": bundle.motion.motion_hash,
                "nc_hashes": {
                    dialect.value: C.nc_hash((root / task_id / C.DIALECT_ARTIFACTS[dialect][0]).read_bytes())
                    for dialect in C.Dialect
                },
            }
            print(f"QUALIFIED AND RETAINED {task_id} ({time.perf_counter() - started:.3f} s)", flush=True)
        record["elapsed_seconds"] = time.perf_counter() - started
        report["tasks"].append(record)
    report["status"] = "failed" if failed else "complete"
    # Exclusive creation preserves any earlier interrupted/failed investigation.
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"generation report: {report_path} (status={report['status']})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
