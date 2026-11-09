"""Audit DiffCAM run artifacts and write task-clustered JSON/Markdown reports.

Manifest JSON has ``schema_version: 1``, ``tasks`` (mapping or list), an
``expected_runs`` list, and optional predeclared ``comparisons``. Task entries
provide task_id/family/task_hash and optionally completion_residual_mm3 and
removal_required. Expected runs provide run_id/task_id/method/seed. Comparisons
provide name/method/baseline/dialect/metrics and optional families.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from benchmarks.audit import AuditInputError, generate_report, render_markdown
from benchmarks.contracts import ContractError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, help="directory recursively containing run.json files")
    parser.add_argument("--manifest", required=True, help="predeclared task/run/comparison manifest JSON")
    parser.add_argument("--output-json", required=True, help="destination for the machine-readable report")
    parser.add_argument("--output-md", required=True, help="destination for the Markdown pilot table")
    parser.add_argument("--bootstrap-seed", type=int, default=20261108)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, _ = generate_report(
            args.runs_root, args.manifest, bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates, confidence=args.confidence,
        )
        markdown = render_markdown(report)
        json_path, md_path = Path(args.output_json), Path(args.output_md)
        for path in (json_path, md_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
    except (AuditInputError, ContractError, OSError, ValueError, TypeError) as error:
        print(f"evaluation aggregation failed: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {json_path} and {md_path}; "
        f"auditor_passed={report['summary']['auditor_passed']}; "
        f"exact_id_reconciliation={report['expected_vs_observed']['exact_id_reconciliation']}"
    )
    # A syntactically valid report can document a failed audit or missing runs.
    return 0 if report["summary"]["auditor_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
