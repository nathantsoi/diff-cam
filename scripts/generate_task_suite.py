#!/usr/bin/env python3
"""Generate declared T2.3 task families and run their cheap nominal gates.

The v2 path does NOT post G-code or perform the pending 0.125 mm dual-dialect
qualification. It runs the independent reference on each task grid and scores
actual stock masks against the reachable target. Definition manifests are
byte-reproducible; wall times and any exclusion diagnostics belong in the
fresh ``generation_report.json``, never silently back-propagated into task
hashes. Existing manifests and output reports are not overwritten.
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


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=suite.PROJECT_ROOT / "configs" / "task_manifest.json",
        help="deterministic task-definition manifest (default: tracked T2.3 manifest)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=suite.PROJECT_ROOT / "runs" / "procedural-t23v2",
        help="parent for new, unposted task/seed/cheap-gate artifacts",
    )
    parser.add_argument("--seed", type=int, default=suite.GENERATOR_SEED,
                        help="declared deterministic design seed")
    parser.add_argument("--dev-count", type=int, default=suite.DEFAULT_DEV_PER_FAMILY,
                        help="cases per family in dev (default: 3)")
    parser.add_argument("--test-count", type=int, default=suite.DEFAULT_TEST_PER_FAMILY,
                        help="held-out cases per family (default: 10)")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="compare a manifest with reconstruction for these counts/seed; no evaluation or writes",
    )
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="construct and emit/check the manifest without cheap gates or artifact directories",
    )
    return parser


def _task_files(bundle, qualification, provenance):
    """Typed, unposted seed evidence. No fabricated RunRecord or NC data."""
    commit, dirty = provenance
    return {
        "task.json": C.dumps(bundle.task).encode("utf-8"),
        "motion_plan.json": C.dumps(bundle.motion).encode("utf-8"),
        "seed_spec.json": suite._json(bundle.seed_spec.as_dict()).encode("utf-8"),
        "cheap_qualification.json": suite._json(qualification).encode("utf-8"),
        "generation.json": suite._json({
            "generation_version": bundle.generation_version,
            "generator_seed": bundle.generator_seed,
            "design_seed": bundle.design_seed,
            "source_commit": commit,
            "dirty_patch_hash": dirty,
            "posted_qualification": "pending",
            "cheap_executor": suite.CHEAP_GATE_EXECUTOR,
            "settings": dict(suite.CHEAP_GATE_SETTINGS),
        }).encode("utf-8"),
    }


def _write_case(bundle, qualification, root, provenance):
    if root.exists():
        raise FileExistsError(f"existing cheap task directory will not be overwritten: {root}")
    root.mkdir(parents=False, exist_ok=False)
    files = _task_files(bundle, qualification, provenance)
    for name, content in files.items():
        (root / name).write_bytes(content)
    hashes = {name: C.sha256_bytes(content) for name, content in sorted(files.items())}
    (root / "hashes.json").write_text(suite._json(hashes), encoding="utf-8")


def _error_details(error, qualification=None):
    child_trace = getattr(error, "worker_traceback", None)
    return {
        "failure_class": getattr(error, "failure_kind", "controller_or_artifact"),
        "error_type": getattr(error, "worker_error_type", None) or type(error).__name__,
        "error": str(error),
        "traceback": child_trace or traceback.format_exc(),
        "traceback_origin": "child_worker" if child_trace else "controller",
        "worker_pid": (qualification["worker_pid"] if qualification is not None
                        else getattr(error, "worker_pid", None)),
        "worker_own_hwm_bytes": (
            qualification["worker_own_hwm_bytes"] if qualification is not None
            else getattr(error, "worker_own_hwm_bytes", None)
        ),
    }


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.verify_only and args.manifest_only:
        _parser().error("--verify-only and --manifest-only are mutually exclusive")
    command_started = time.perf_counter()
    bundles = suite.generate_tasks(args.dev_count, args.test_count, seed=args.seed)
    generated = suite.manifest_text(bundles)
    definition_seconds = time.perf_counter() - command_started
    if args.verify_only:
        suite.verify_manifest(args.manifest, expected_text=generated)
        print(f"manifest reproduced byte-for-byte: {args.manifest}")
        return 0

    # Refuse to silently replace an old generator definition. Intentional v2
    # upgrades are reviewed and copied explicitly to the tracked configuration.
    suite.write_manifest(args.manifest, expected_text=generated)
    if args.manifest_only:
        print(f"manifest constructed; cheap seed qualification NOT run: {args.manifest}")
        return 0

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "generation_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite generation report: {report_path}")
    provenance = suite._provenance()
    report = {
        "schema_version": "t2.3-generate-report/2",
        "generation_version": suite.GENERATION_VERSION,
        "generator_seed": args.seed,
        "requested_counts": {
            family: {"dev": args.dev_count, "test": args.test_count}
            for family in sorted({bundle.task.family for bundle in bundles})
        },
        "per_family_counts": json.loads(generated)["per_family_counts"],
        "source_commit": provenance[0],
        "dirty_patch_hash": provenance[1],
        "manifest_sha256": C.sha256_bytes(generated.encode("utf-8")),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "definition_seconds": definition_seconds,
        "qualification": {
            "executor": suite.CHEAP_GATE_EXECUTOR,
            "settings": dict(suite.CHEAP_GATE_SETTINGS),
            "task_grid_pitch_mm": suite.PITCH_MM,
            "posted_qualification": "pending",
            "posted_dialects": [dialect.value for dialect in C.Dialect],
            "posted_grid_pitch_mm": 0.125,
            "method": "fresh bounded per-task f64 hard-sim workers; actual stock masks versus exact operation-target masks",
            "fresh_subprocess_per_task": True,
            "task_process_parallelism": 1,
            "predeclared_exclusion_reasons": list(suite.PREDECLARED_EXCLUSION_REASONS),
        },
        "requested_tasks": [bundle.task.task_id for bundle in bundles],
        "tasks": [],
    }
    failed = False
    cheap_gate_sum = 0.0
    loop_started = time.perf_counter()
    for bundle in bundles:
        task_id = bundle.task.task_id
        case_started = time.perf_counter()
        qualification = None
        try:
            qualification = suite.cheap_qualify_seed(bundle, log_directory=root / "worker-logs")
            gate_elapsed = time.perf_counter() - case_started
        except Exception as error:
            failed = True
            elapsed = time.perf_counter() - case_started
            cheap_gate_sum += elapsed
            seed_failure = getattr(error, "failure_kind", None) == "seed_qualification"
            record = {
                **_error_details(error),
                "task_id": task_id,
                "task_hash": bundle.task.task_hash,
                "motion_hash": bundle.motion.motion_hash,
                "status": "excluded_unqualified_seed" if seed_failure else "infrastructure_failure",
                "predeclared_exclusion_reason": str(error) if seed_failure else None,
                "seed_qualification_completed": False,
                "posted_qualification": "pending",
                "cheap_gate_seconds": elapsed,
                "elapsed_seconds": elapsed,
            }
            label = "UNQUALIFIED" if seed_failure else "WORKER/EVIDENCE FAILED"
            print(f"{label} {task_id}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        else:
            cheap_gate_sum += gate_elapsed
            try:
                _write_case(bundle, qualification, root / task_id, provenance)
            except Exception as error:
                # A successful gate does not make a failed artifact write an
                # unqualified seed. Preserve the evidence failure distinctly.
                failed = True
                record = {
                    **_error_details(error, qualification),
                    "failure_class": "artifact_generation",
                    "task_id": task_id,
                    "task_hash": bundle.task.task_hash,
                    "motion_hash": bundle.motion.motion_hash,
                    "status": "artifact_generation_failure",
                    "predeclared_exclusion_reason": None,
                    "seed_qualification_completed": True,
                    "seed_qualification_status": qualification["status"],
                    "posted_qualification": "pending",
                    "cheap_gate_seconds": gate_elapsed,
                    "elapsed_seconds": time.perf_counter() - case_started,
                }
                print(f"ARTIFACT FAILED {task_id}: {type(error).__name__}: {error}",
                      file=sys.stderr, flush=True)
            else:
                record = {
                    **qualification,
                    "task_id": task_id,
                    "task_json": str(root / task_id / "task.json"),
                    "motion_json": str(root / task_id / "motion_plan.json"),
                    "cheap_qualification_seconds": gate_elapsed,
                    "elapsed_seconds": time.perf_counter() - case_started,
                    "posted_qualification": "pending",
                }
                print(f"QUALIFIED CHEAP {task_id} ({record['elapsed_seconds']:.3f} s)", flush=True)
        report["tasks"].append(record)
    report["qualified_task_ids"] = [row["task_id"] for row in report["tasks"]
                                    if row["status"] == "qualified_cheap_hard_sim"]
    report["excluded_task_ids"] = [
        row["task_id"] for row in report["tasks"]
        if row.get("failure_class") == "seed_qualification"
    ]
    report["infrastructure_failure_task_ids"] = [
        row["task_id"] for row in report["tasks"]
        if row["status"] in {"infrastructure_failure", "artifact_generation_failure"}
    ]
    report["cheap_gate_seconds_sum"] = cheap_gate_sum
    report["task_loop_wall_seconds"] = time.perf_counter() - loop_started
    report["status"] = "failed" if failed else "qualified_cheap"
    command_elapsed = time.perf_counter() - command_started
    report["generation_wall_seconds"] = command_elapsed
    report["wall_time_scope"] = "generate_task_suite.main after imports, through report assembly (before generation_report.json write)"
    # Exclusive creation preserves any earlier interrupted/failed investigation.
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"generation report: {report_path} (status={report['status']}; "
          f"elapsed through assembly={command_elapsed:.3f} s; "
          f"through report write={time.perf_counter() - command_started:.3f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
