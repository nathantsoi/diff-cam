#!/usr/bin/env python3
"""Build deterministic, traceable result artifacts from audited records.

The published inputs remain immutable. The legacy scaffold preserves its
explicit audit/config interface. ``--results`` additionally reads published
pilot reports, a ``run_campaign.py`` audit bundle, and declared T3.3 geometry
studies, then emits the main/family/init/selection/resource/alignment CSVs and
``reports/results/summary.md`` into a fresh destination. Partial and absent
inputs are labelled pending/absent, never replaced with synthetic results.

This program does not evaluate machining validity, physical safety, or optimizer
superiority. Statistical tables use the existing task-cluster audit utilities.
The optional resource plotting source requires separately provisioned matplotlib;
core tests cover data generation rather than optional image rendering.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import run_artifacts as R
from benchmarks import audit as A
from benchmarks import contracts as C

SCHEMA = "diffcam-paper-artifacts/1"
AUDIT_SCHEMA = "diffcam-eval-audit/1"
DECISION_PATTERN = re.compile(r"^## (D-\d{4}-\d{2}-\d{2}-\d{2})\s+—\s*(.+)$")
COUNTERS = (
    "candidate_evaluations", "forward_calls", "backward_calls", "validation_calls",
    "segment_steps", "voxel_steps", "process_peak_rss_bytes",
)
COST_TIMES = ("warmup_wall_time_s", "steady_state_wall_time_s",
              "end_to_end_candidate_wall_time_s")
RUN_METRICS = (
    "removed_volume_mm3", "residual_volume_mm3", "gouge_volume_mm3",
    "max_surface_error_mm", "cycle_time_s", "air_fraction", "force_proxy",
)
OUTCOME_FIELDS = ("safety_pass", "geometry_pass", "completion_pass", "efficiency_eligible")


class PaperArtifactError(ValueError):
    """An input lacks a complete, unambiguous evidence chain."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperArtifactError(message)


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False) + "\n").encode("utf-8")


def finite_tree(value: Any, where: str = "input") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite number at {where}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            finite_tree(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{where}[{index}]")


def mapping(value: Any, where: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{where} must be an object")
    return value


def hash_field(value: Any, where: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    require(isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None,
            f"{where} must be a canonical sha256 hash")
    return value


def no_manuscript(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    require("manuscript" not in resolved.parts,
            f"{role} must not read or write manuscript/: {resolved}")
    return resolved


def source(path: Path, role: str, *, json_input: bool = True) -> tuple[Any, dict[str, Any]]:
    """Read one declared input, returning finite content plus its immutable identity."""
    actual = no_manuscript(path, role)
    try:
        payload = actual.read_bytes()
    except OSError as error:
        raise PaperArtifactError(f"cannot read {role} {actual}: {error}") from error
    identity = {"role": role, "sha256": digest(payload)}
    if not json_input:
        try:
            return payload.decode("utf-8"), identity
        except UnicodeDecodeError as error:
            raise PaperArtifactError(f"{role} is not UTF-8 text") from error
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=(
            lambda text: (_ for _ in ()).throw(PaperArtifactError(
                f"non-finite JSON constant {text} in {role}"))))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperArtifactError(f"invalid JSON for {role} {actual}: {error}") from error
    finite_tree(value, role)
    return value, identity


def typed_evaluation(value: Any, where: str) -> C.EvaluationResult:
    if value is None:
        raise PaperArtifactError(f"{where} lacks a linked typed evaluation")
    try:
        raw = dict(mapping(value, where))
        require(raw.get("record_type", "EvaluationResult") == "EvaluationResult",
                f"{where} has the wrong record_type")
        # Audit dialects embed evaluation.to_dict(), which intentionally has no
        # record_type discriminator. Complete eval_*.json wraps that payload in
        # {"record_type": ..., "record": ...}; handle each without weakening it.
        if "record_type" in raw:
            raw = dict(mapping(raw.get("record"), where + ".record"))
        return C.EvaluationResult.from_dict(raw)
    except (TypeError, ValueError) as error:
        raise PaperArtifactError(f"{where} is not a valid typed evaluation: {error}") from error


def display(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        require(math.isfinite(value), "cannot serialize non-finite table value")
        return format(value, ".12g")
    if isinstance(value, str):
        return value
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def latex(value: Any) -> str:
    text = display(value)
    # LaTeX tables carry a visibly abbreviated hash; claim_evidence.json retains
    # the entire hash in its corresponding cell and source_file_identity.
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        text = text[:19] + r"\ldots{}"
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise",
                           lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: display(row.get(name)) for name in columns})
    return stream.getvalue().encode("utf-8")


def table_tex(columns: Sequence[str], rows: Sequence[Mapping[str, Any]], caption: str) -> bytes:
    lines = [
        "% Generated scaffold only; no campaign-completeness or machining-safety claim.",
        "% Requires booktabs. Column-width/layout review is still required before publication.",
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\caption{" + latex(caption) + "}",
        r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
        " & ".join(latex(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex(row.get(column)) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines).encode("utf-8")


class Emit:
    """One code path for CSV/LaTeX rows and per-cell claim provenance."""
    def __init__(self, common_sources: Mapping[str, dict[str, Any]]):
        self.common_sources = dict(common_sources)
        self.files: dict[str, bytes] = {}
        self.claims: list[dict[str, Any]] = []
        self.columns: dict[str, list[str]] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.row_numbers: dict[str, int] = {}
        self.claim_ids: set[str] = set()

    def row(self, table: str, columns: Sequence[str], values: Mapping[str, Any], *,
            evidence: Mapping[str, Any], pointers: Mapping[str, str | None],
            claim_kind: str, statistical_unit: str = "not_an_estimate",
            extra_sources: Mapping[str, dict[str, Any]] | None = None,
            caption: str = "") -> None:
        target = "tables/" + table
        if table not in self.columns:
            self.columns[table] = list(columns)
            self.row_numbers[table] = 0
        require(self.columns[table] == list(columns), f"inconsistent columns for {table}")
        require(set(values) == set(columns), f"cell schema mismatch for {table}")
        row_number = self.row_numbers[table]
        self.row_numbers[table] += 1
        row_id = str(evidence["row_id"])
        claim_id = target + "#" + row_id
        require(claim_id not in self.claim_ids, f"duplicate claim id {claim_id}")
        self.claim_ids.add(claim_id)
        self.rows.setdefault(table, []).append(dict(values))
        required_evidence = dict(evidence)
        required_evidence.pop("row_id")
        cells = {}
        for name in columns:
            pointer = pointers.get(name)
            cells[name] = {
                "value": values[name], "display": display(values[name]),
                "source_pointer": pointer,
                "derivation": ("direct audited/source value" if pointer else
                                "deterministic grouping or explicit unavailable/derived label"),
            }
        sources = dict(self.common_sources)
        sources.update(dict(extra_sources or {}))
        self.claims.append({
            "claim_id": claim_id, "artifact": target + ".csv", "latex": target + ".tex",
            "row_id": row_id, "row_number_zero_based": row_number,
            "claim_kind": claim_kind, "statistical_unit": statistical_unit,
            "column_order": list(columns), "table_caption": caption,
            "cells": cells, "evidence": required_evidence,
            "source_file_identity": {key: value["sha256"] for key, value in sorted(sources.items())},
        })

    def table(self, name: str, caption: str) -> None:
        columns = self.columns.get(name)
        if columns is None:
            raise PaperArtifactError(f"no declared columns for requested table {name}")
        rows = self.rows.get(name, [])
        self.files[f"tables/{name}.csv"] = csv_bytes(columns, rows)
        self.files[f"tables/{name}.tex"] = table_tex(columns, rows, caption)

    def every_table(self, captions: Mapping[str, str]) -> None:
        for name in sorted(self.columns):
            self.table(name, captions[name])


def outcome(evaluation: C.EvaluationResult, task: A.TaskInfo, name: str) -> Any:
    """Use the same task-qualified outcome reader as the existing audit."""
    try:
        return A._outcome_value(evaluation, task, name)
    except (A.AuditInputError, ValueError) as error:
        raise PaperArtifactError(f"cannot read {name}: {error}") from error


def metric_value(evaluation: C.EvaluationResult, key: str) -> Any:
    metric = evaluation.metrics.get(key)
    return None if metric is None else metric.value


def precision_values(value: Any, where: str) -> list[tuple[str, Any]]:
    """Find explicitly named precision metadata without guessing an ambient default."""
    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "simulator_precision":
                found.append((f"{where}/{key}", child))
            else:
                found.extend(precision_values(child, f"{where}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(precision_values(child, f"{where}[{index}]"))
    return found


def validate_precision_and_decisions(config: Mapping[str, Any], *, run_id: str,
                                     precision: str, decisions: set[str],
                                     external_decision_binding: bool = False,
                                     allow_unavailable_precision: bool = False) -> tuple[Any, str]:
    values = precision_values(config, "")
    legacy = config.get("simulator")
    if isinstance(legacy, Mapping):
        requested = legacy.get("requested_precision")
        record = SimpleNamespace(precision=requested,
                                 precision_metadata=lambda: dict(legacy))
        try:
            R.simulator_precision_metadata(record, requested_precision=requested)
        except (TypeError, ValueError) as error:
            raise PaperArtifactError(f"{run_id}: invalid typed simulator identity: {error}") from error
        values.append(("/simulator", requested))
    if not values:
        if allow_unavailable_precision:
            values.append(("/unavailable", "unavailable"))
        else:
            require(values, f"{run_id}: config has no explicit simulator_precision metadata")
    require(all(value in ("f32", "f64", "unavailable") for _, value in values),
            f"{run_id}: simulator_precision must be explicit f32 or f64")
    distinct = {value for _, value in values}
    require(len(distinct) == 1, f"{run_id}: conflicting precision declarations: {values}")
    actual = values[0][1]
    require(actual == precision,
            f"{run_id}: campaign precision {precision} conflicts with run precision {actual}")
    require(isinstance(config.get("decision_ids"), list) or external_decision_binding,
            f"{run_id}: decision_ids must be embedded or explicitly bound at dataset scope")
    ids = config.get("decision_ids", list(decisions))
    require(all(isinstance(item, str) and item.startswith("D-") for item in ids),
            f"{run_id}: decision_ids entries must be explicit D identifiers")
    require(len(ids) == len(set(ids)), f"{run_id}: duplicate decision ID")
    require(set(ids) == decisions,
            f"{run_id}: decision IDs {sorted(set(ids))} != campaign {sorted(decisions)}")
    return actual, values[0][0]


def nominal_method(method: str) -> bool:
    return method.casefold() in {"cam", "nominal", "nominal-cam", "nominal_cam"}


def validate_runs(report: Mapping[str, Any], manifest: A.Manifest) -> dict[str, Mapping[str, Any]]:
    require(report.get("schema_version") == AUDIT_SCHEMA,
            f"audit schema must be {AUDIT_SCHEMA}")
    require(report.get("evidence_complete") is True,
            "refusing to build: audit evidence_complete=false")
    require(isinstance(report.get("summary"), Mapping), "audit summary is missing")
    require(report["summary"].get("auditor_passed") is True,
            "refusing to build an audit that failed mechanical validation")
    reconciliation = mapping(report.get("expected_vs_observed"), "expected_vs_observed")
    require(reconciliation.get("exact_id_reconciliation") is True,
            "refusing to build without exact expected-vs-observed run reconciliation")
    expected_ids = sorted(manifest.runs)
    require(reconciliation.get("missing_run_ids") == [] and
            reconciliation.get("unexpected_run_ids") == [] and
            reconciliation.get("duplicate_run_ids") == [] and
            reconciliation.get("expected_run_ids") == expected_ids and
            reconciliation.get("observed_run_ids") == expected_ids,
            "audit run links are incomplete, unexpected, or duplicated")
    policy = mapping(report.get("policy"), "policy")
    require(policy.get("independent_unit") == "task",
            "statistical independent unit must be task, not solver seed")
    require("within each task" in str(policy.get("seed_aggregation")),
            "audit must average seeds within task before inference")
    rows = report.get("runs")
    require(isinstance(rows, list), "audit.runs must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(rows):
        row = mapping(item, f"runs[{index}]")
        run_id = row.get("run_id")
        require(isinstance(run_id, str) and run_id in manifest.runs,
                f"run[{index}] has an unknown/missing run_id link")
        require(run_id not in by_id, f"duplicate run row {run_id}")
        expected = manifest.runs[run_id]
        task = manifest.tasks[expected.task_id]
        require(row.get("task_hash") == task.task_hash,
                f"{run_id}: report task_hash does not match manifest")
        require(row.get("expected", True) is True and row.get("duplicate_id", False) is False,
                f"{run_id}: unexpected or duplicate audited record")
        require(isinstance(row.get("issues"), list), f"{run_id}: missing issue list")
        require(isinstance(row.get("status"), str) and row["status"], f"{run_id}: missing status")
        dialects = mapping(row.get("dialects"), f"{run_id}.dialects")
        for key, raw in dialects.items():
            require(isinstance(key, str) and key in ("rs274", "haas"),
                    f"{run_id}: unknown dialect {key}")
            record = mapping(raw, f"{run_id}.{key}")
            declared = record.get("declared_nc_hash")
            hash_field(declared, f"{run_id}.{key}.declared_nc_hash", optional=True)
            actual = record.get("actual_nc_hash")
            hash_field(actual, f"{run_id}.{key}.actual_nc_hash", optional=True)
            if record.get("evaluation") is not None:
                typed = typed_evaluation(record["evaluation"], f"{run_id}.{key}.evaluation")
                require(typed.task_hash == task.task_hash and typed.dialect.value == key,
                        f"{run_id}.{key}: evaluation task/dialect identity mismatch")
                require(typed.nc_hash == declared and actual == declared,
                        f"{run_id}.{key}: NC identity does not match typed evaluation/audit linkage")
            if record.get("status") == "unsupported":
                require(record.get("evaluation") is None and declared is None,
                        f"{run_id}.{key}: unsupported row cannot contain a phantom NC result")
        by_id[run_id] = row
    require(set(by_id) == set(expected_ids), "run rows do not cover every expected run")
    if any("reported_outcomes" in row for row in by_id.values()) and "refinement" not in report:
        raise PaperArtifactError("reported refinement outcomes lack a refinement/link evidence section")
    comparisons = report.get("comparisons")
    require(isinstance(comparisons, list), "audit.comparisons must be a list")
    mapping(report.get("summary"), "summary")
    return by_id


def validate_config_links(configs: Mapping[str, tuple[Mapping[str, Any], dict[str, Any]]],
                          manifest: A.Manifest) -> None:
    require(set(configs) == set(manifest.runs),
            "every expected run needs exactly one bound config (use --run-root or --run-config)")
    for run_id, (config, _) in sorted(configs.items()):
        expected = manifest.runs[run_id]
        if "task_hash" in config:
            require(config["task_hash"] == manifest.tasks[expected.task_id].task_hash,
                    f"{run_id}: config.task_hash differs from manifest")
        if "task_id" in config:
            require(config["task_id"] == expected.task_id, f"{run_id}: config task_id mismatch")
        if "method" in config:
            require(config["method"] == expected.method, f"{run_id}: config method mismatch")
        if "seed" in config:
            require(not isinstance(config["seed"], bool) and config["seed"] == expected.seed,
                    f"{run_id}: config seed mismatch")


def relative_payload(link_file: Path, value: Any, where: str) -> Path:
    require(isinstance(value, str) and value, f"{where} must be a nonempty relative path")
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and ".." not in relative.parts and "." not in relative.parts,
            f"{where} contains an unsafe path: {value}")
    payload = (link_file.parent / Path(relative)).resolve()
    require(payload.is_relative_to(link_file.parent.resolve()),
            f"{where} escapes its evidence bundle")
    require(payload.is_file(), f"{where} is missing: {payload}")
    return payload


def conservative_consistency(value: Any, where: str) -> Mapping[str, Any]:
    outcome_value = mapping(value, where)
    for name in OUTCOME_FIELDS[:3]:
        require(outcome_value.get(name) is None or isinstance(outcome_value.get(name), bool),
                f"{where}.{name} must be true/false/null")
    for name in ("efficiency_eligible", "excluded_from_efficiency", "refinement_disagreement"):
        require(isinstance(outcome_value.get(name), bool), f"{where}.{name} must be boolean")
    require(outcome_value["excluded_from_efficiency"] is (not outcome_value["efficiency_eligible"]),
            f"{where}: efficiency exclusion disagrees with eligibility")
    fields = outcome_value.get("disagreement_fields")
    require(isinstance(fields, list) and len(fields) == len(set(fields)) and
            all(item in ("safety_pass", "geometry_pass", "completion_pass", "coverage")
                for item in fields), f"{where}: invalid disagreement_fields")
    require(outcome_value["refinement_disagreement"] == bool(fields),
            f"{where}: disagreement flag and fields differ")
    require(outcome_value.get("efficiency_eligible") is False or not fields,
            f"{where}: disagreement cannot retain efficiency eligibility")
    require(isinstance(outcome_value.get("rule"), str) and outcome_value["rule"],
            f"{where}: explicit conservative rule is missing")
    return outcome_value


def refinement_from_report(report: Mapping[str, Any], run_ids: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    """Read inline upstream refinement evidence without reinterpreting its verdict policy."""
    if "refinement" not in report:
        return {}
    refinement = mapping(report["refinement"], "refinement")
    for field in ("links_by_run_id", "resources_by_run_id"):
        if field in refinement:
            require(isinstance(refinement[field], Mapping), f"refinement.{field} must be an object")
    links = refinement.get("links_by_run_id")
    if links is None:
        return {}
    require(set(links) == set(run_ids),
            "inline refinement field must supply a validated link for every expected run")
    for run_id, value in links.items():
        link = mapping(value, f"refinement.links_by_run_id.{run_id}")
        require(link.get("primary_run_id") == run_id,
                f"{run_id}: inline refinement primary identity differs")
        require(link.get("refined_run_id") == run_id + "-refined",
                f"{run_id}: inline refined run must have its own distinct ID")
        hash_field(link.get("refined_evaluation_sha256"),
                   f"{run_id}.refined_evaluation_sha256")
        if "sha256" in link:
            hash_field(link["sha256"], f"{run_id}.inline_link_sha256")
        conservative_consistency(link.get("conservative_outcome"),
                                 f"{run_id}.inline conservative_outcome")
        require(isinstance(link.get("refined_artifact_path"), str) and link["refined_artifact_path"],
                f"{run_id}: missing inline refined-evaluation link")
    return dict(links)


def load_refinement_file(path: Path, run_ids: Sequence[str], runs: Mapping[str, Any],
                         manifest: A.Manifest, primary_configs: Mapping[str, Path]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Verify raw schema-3 schema bytes and typed links for an explicitly supplied bundle."""
    value, identity = source(path, "refinement-file")
    link = mapping(value, "refinement_link")
    require(link.get("schema") == "diffcam-refinement-link/3",
            f"{path}: expected diffcam-refinement-link/3")
    run_id = link.get("primary_run_id")
    require(run_id in run_ids, "refinement link has unknown primary_run_id")
    require(link.get("refined_run_id") == run_id + "-refined",
            f"{run_id}: refined evaluation must have its own distinct run ID")
    expected = manifest.runs[run_id]
    for name in ("primary_metrics_sha256", "primary_rs274_evaluation_sha256",
                 "primary_run_json_sha256", "refined_evaluation_sha256", "comparison_sha256"):
        hash_field(link.get(name), f"{run_id}.{name}")
    hash_field(link.get("primary_haas_evaluation_sha256"),
               f"{run_id}.primary_haas_evaluation_sha256", optional=True)
    pure_path = PurePosixPath(str(link.get("primary_run_path", "")))
    require(pure_path.name == run_id, f"{run_id}: primary_run_path mismatch")
    primary_row = runs[run_id]
    dialect = primary_row["dialects"].get("rs274")
    require(dialect is not None and dialect.get("evaluation") is not None,
            f"{run_id}: refinement requires an RS274 primary evaluation")
    primary = typed_evaluation(dialect["evaluation"], f"{run_id}.primary.rs274")
    task = manifest.tasks[expected.task_id]
    require(run_id in primary_configs, f"{run_id}: no bound primary config for refinement verification")
    primary_root = no_manuscript(primary_configs[run_id].parent, f"{run_id}.primary_root")
    for filename, field in (
        ("metrics.json", "primary_metrics_sha256"),
        ("eval_posted_rs274.json", "primary_rs274_evaluation_sha256"),
        ("run.json", "primary_run_json_sha256"),
    ):
        candidate = primary_root / filename
        require(candidate.is_file() and digest(candidate.read_bytes()) == link[field],
                f"{run_id}: immutable primary {filename} differs from refinement link")
    haas_path = primary_root / "eval_posted_haas.json"
    if haas_path.is_file():
        require(digest(haas_path.read_bytes()) == link["primary_haas_evaluation_sha256"],
                f"{run_id}: immutable primary Haas evaluation differs from refinement link")
    else:
        require(link.get("primary_haas_evaluation_sha256") is None and
                runs[run_id]["dialects"].get("haas", {}).get("status") != "supported",
                f"{run_id}: supported posted Haas evaluation is missing or its hash is not null")
    for dialect_name in ("rs274", "haas"):
        declared_hash = runs[run_id]["dialects"].get(dialect_name, {}).get("declared_nc_hash")
        if declared_hash is None:
            continue
        nc_file = primary_root / f"{dialect_name}.nc"
        require(nc_file.is_file() and C.nc_hash(nc_file.read_bytes()) == declared_hash,
                f"{run_id}: primary {dialect_name} NC bytes differ from the audited identity")
    refined_relative = PurePosixPath(str(link.get("refined_artifact_path", "")))
    require(refined_relative.parts == (run_id + "-refined", "eval_posted_rs274_refined.json"),
            f"{run_id}: refined artifact must be in its distinct auxiliary directory")
    refined_file = relative_payload(path, link.get("refined_artifact_path"),
                                    f"{run_id}.refined_artifact")
    refined_bytes = refined_file.read_bytes()
    require(digest(refined_bytes) == link["refined_evaluation_sha256"],
            f"{run_id}: refined evaluation hash mismatch")
    refined = typed_evaluation(json.loads(refined_bytes.decode("utf-8")),
                               f"{run_id}.refined evaluation")
    require(refined.dialect is C.Dialect.RS274 and refined.task_hash == task.task_hash and
            refined.nc_hash == primary.nc_hash and refined.tool_setup_hash == primary.tool_setup_hash and
            refined.evaluator_id == "posted_hard/production",
            f"{run_id}: refined evaluation identity differs from primary")
    require(refined.coverage is C.Coverage.COMPLETE,
            f"{run_id}: production refinement is not complete")
    refined_parsed = refined.discretization.get("parsed_motion_hash")
    primary_parsed = primary.discretization.get("parsed_motion_hash")
    hash_field(primary_parsed, f"{run_id}.primary.parsed_motion_hash")
    require(refined_parsed == primary_parsed and refined.motion_hash in (None, primary_parsed),
            f"{run_id}: typed parsed motion differs across fidelity levels")
    settings = mapping(refined.discretization.get("settings"), f"{run_id}.refined.settings")
    primary_settings = mapping(primary.discretization.get("settings"), f"{run_id}.primary.settings")
    require(set(settings) == set(primary_settings),
            f"{run_id}: refinement silently changed sampling controls")
    for name, value_settings in primary_settings.items():
        if name != "max_step_mm":
            require(settings[name] == value_settings,
                    f"{run_id}: refinement changed sampling control {name}")
    refined_step = settings["max_step_mm"]
    primary_step = primary_settings["max_step_mm"]
    require(isinstance(refined_step, (int, float)) and isinstance(primary_step, (int, float)) and
            not isinstance(refined_step, bool) and not isinstance(primary_step, bool) and
            0 < refined_step < primary_step,
            f"{run_id}: refined max_step_mm must be positive and strictly smaller")
    require(settings.get("expected_nc_hash") == refined.nc_hash and
            settings.get("compute_bounds") is True and settings.get("safety_screen") is True,
            f"{run_id}: missing exact NC/bounds/safety controls")
    # Recompute only the declared source-level rule, not a second geometry evaluator.
    expected_comparison = {
        "safety_pass": {"primary": primary.safety_pass, "refined": refined.safety_pass,
                        "changed": primary.safety_pass != refined.safety_pass},
        "geometry_pass": {"primary": primary.geometry_pass, "refined": refined.geometry_pass,
                          "changed": primary.geometry_pass != refined.geometry_pass},
        "coverage": {"primary": primary.coverage.value, "refined": refined.coverage.value,
                     "changed": primary.coverage is not refined.coverage},
        "efficiency_eligible": {
            "primary": primary.efficiency_eligible, "refined": refined.efficiency_eligible,
            "changed": primary.efficiency_eligible != refined.efficiency_eligible},
    }
    for name in ("residual_volume_mm3", "gouge_volume_mm3", "max_surface_error_mm"):
        left, right = metric_value(primary, name), metric_value(refined, name)
        expected_comparison[name] = {
            "primary": left, "refined": right,
            "delta": right - left if left is not None and right is not None else None,
            "unit": "mm" if name.endswith("_mm") else "mm^3",
        }
    primary_completion = outcome(primary, task, "completion_pass")
    refined_completion = outcome(refined, task, "completion_pass")
    expected_comparison["completion_pass"] = {
        "primary": primary_completion, "refined": refined_completion,
        "changed": primary_completion != refined_completion,
    }
    expected_comparison["any_outcome_verdict_change"] = any(
        expected_comparison[name]["changed"]
        for name in ("safety_pass", "geometry_pass", "coverage", "efficiency_eligible"))
    actual_comparison = mapping(link.get("comparison"), f"{run_id}.comparison")
    required_rows = set(expected_comparison)
    require(required_rows.issubset(actual_comparison),
            f"{run_id}: schema-3 comparison lacks completion/fidelity rows")
    for name, expected_row in expected_comparison.items():
        require(actual_comparison[name] == expected_row,
                f"{run_id}: comparison row {name} differs from typed primary/refined evidence")
    disagreement_fields = [name for name in
                           ("safety_pass", "geometry_pass", "completion_pass", "coverage")
                           if expected_comparison[name]["changed"]]
    require(actual_comparison.get("changed_outcome_fields", [
        name for name in ("safety_pass", "geometry_pass", "completion_pass",
                          "coverage", "efficiency_eligible")
        if expected_comparison[name]["changed"]]) == [
            name for name in ("safety_pass", "geometry_pass", "completion_pass",
                              "coverage", "efficiency_eligible")
            if expected_comparison[name]["changed"]],
        f"{run_id}: changed_outcome_fields differs from typed evidence")

    def combined(left, right):
        if left is False or right is False:
            return False
        if left is None or right is None:
            return None
        return True

    completion = combined(primary_completion, refined_completion)
    expected_conservative = {
        "safety_pass": combined(primary.safety_pass, refined.safety_pass),
        "geometry_pass": combined(primary.geometry_pass, refined.geometry_pass),
        "completion_pass": completion,
        "efficiency_eligible": bool(primary.efficiency_eligible and refined.efficiency_eligible
                                    and completion is True and not disagreement_fields),
        "refinement_disagreement": bool(disagreement_fields),
        "disagreement_fields": disagreement_fields,
        "excluded_from_efficiency": False,
    }
    expected_conservative["excluded_from_efficiency"] = not expected_conservative["efficiency_eligible"]
    stored = mapping(link.get("conservative_outcome"), f"{run_id}.conservative_outcome")
    for name, expected_flag in expected_conservative.items():
        require(stored.get(name) == expected_flag,
                f"{run_id}: conservative outcome {name} differs from typed values")
    conservative_consistency(stored, f"{run_id}.conservative_outcome")
    comparison_file = relative_payload(path, link.get("comparison_path"), f"{run_id}.comparison")
    require(PurePosixPath(link["comparison_path"]).parts == (run_id + "-refined", "comparison.json"),
            f"{run_id}: comparison must be in the distinct auxiliary directory")
    comparison_bytes = comparison_file.read_bytes()
    require(digest(comparison_bytes) == link["comparison_sha256"],
            f"{run_id}: comparison byte hash mismatch")
    comparison = mapping(json.loads(comparison_bytes.decode("utf-8")),
                         f"{run_id}.comparison_file")
    require(comparison.get("comparison") == link.get("comparison") and
            comparison.get("conservative_outcome") == stored and
            comparison.get("primary_run_id") == run_id,
            f"{run_id}: comparison file disagrees with link")
    auxiliary = mapping(link.get("auxiliary_record"), f"{run_id}.auxiliary_record")
    require(PurePosixPath(str(auxiliary.get("path", ""))).parts == (run_id + "-refined", "run.json"),
            f"{run_id}: auxiliary record must be the distinct refined run's run.json")
    aux_file = relative_payload(path, auxiliary.get("path"), f"{run_id}.auxiliary_record")
    aux_bytes = aux_file.read_bytes()
    require(digest(aux_bytes) == hash_field(auxiliary.get("sha256"),
                                            f"{run_id}.auxiliary.sha256"),
            f"{run_id}: auxiliary record hash mismatch")
    try:
        auxiliary_record = C.loads(aux_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise PaperArtifactError(f"{run_id}: invalid auxiliary record: {error}") from error
    require(isinstance(auxiliary_record, C.RunRecord) and
            auxiliary_record.run_id == run_id + "-refined" and
            auxiliary_record.task_hash == task.task_hash and
            auxiliary_record.status is C.RunStatus.COMPLETE,
            f"{run_id}: auxiliary record identity/status mismatch")
    return {"primary_run_id": run_id, "link": link,
            "conservative_outcome": dict(stored), "comparison": dict(actual_comparison),
            "refined_evaluation": refined.to_dict(), "validation": "typed_metrics_and_schema3_links_recomputed"}, identity


# This source is deterministic data + one optional single-series resource-cost
# diagnostic, not an auto-installed dependency or a core image-render gate.
OPTIONAL_MPL_SCRIPT = '''#!/usr/bin/env python3
"""Optional resource-cost plot. Requires separately provisioned matplotlib.
Missing ledger values are excluded from plotting, never turned into bars at zero.
The CSV beside this script remains the authoritative table and trace of every row.
"""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="explicit PNG destination outside manuscript/")
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).with_name("resource_cost.csv"))
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if "manuscript" in destination.parts:
        parser.error("refusing to write into manuscript/")
    if destination.exists():
        parser.error("choose a fresh image path; do not overwrite unrelated output")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        parser.error("optional matplotlib is unavailable; use a separately provisioned plotting environment")
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    available = [row for row in rows if row["end_to_end_candidate_wall_time_s"] not in ("", "NA")
                 and row["process_peak_rss_bytes"] not in ("", "NA")]
    if not available:
        parser.error("no linked resource measurements to plot; inspect the explicit NA rows")
    names = [row["run_id"] for row in available]
    times = [float(row["end_to_end_candidate_wall_time_s"]) for row in available]
    rss_mib = [float(row["process_peak_rss_bytes"]) / (1024 * 1024) for row in available]
    if not all(value >= 0 for value in times + rss_mib):
        parser.error("resource values must be nonnegative")
    # Default validated categorical slot 1, #2a78d6, one series per panel.
    # Two separately titled panels, NOT a dual-axis graph. No precision or
    # performance ranking is inferred from the image.
    width = max(6.0, min(18.0, len(available) * 0.55 + 2))
    fig, axes = plt.subplots(2, 1, figsize=(width, 6), sharex=True)
    for axis in axes:
        axis.grid(axis="y", color="#c3c2b7", linewidth=0.5, alpha=0.5, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    axes[0].bar(names, times, color="#2a78d6", width=0.6, zorder=2)
    axes[1].bar(names, rss_mib, color="#2a78d6", width=0.6, zorder=2)
    axes[0].set_title("Candidate-call wall time by run (not machining efficiency)")
    axes[1].set_title("Process peak RSS (not device-only memory)")
    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("MiB")
    axes[1].set_xlabel("Run ID; fixed source order")
    axes[1].tick_params(axis="x", rotation=45)
    figure_note = ("Unavailable ledgers remain NA in resource_cost.csv; "
                   "available values are not statistical task-level estimates.")
    fig.suptitle(figure_note, fontsize=8)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, format="png", dpi=140)
    parser.exit(0)


if __name__ == "__main__":
    main()
'''


def task_scope_rows(manifest_raw: Mapping[str, Any], manifest: A.Manifest) -> list[dict[str, Any]]:
    """Preserve declared target exclusions without inventing grid geometry."""
    raw_tasks = manifest_raw["tasks"]
    if isinstance(raw_tasks, Mapping):
        entries = [(str(key), index, value) for index, (key, value) in enumerate(raw_tasks.items())]
    else:
        entries = [(None, index, value) for index, value in enumerate(raw_tasks)]

    def number(owner, key, where):
        value = owner.get(key)
        if value is None:
            return None
        require(not isinstance(value, bool) and isinstance(value, (int, float))
                and value >= 0, f"{where}.{key} must be nonnegative and numeric")
        return value

    def same_declared_values(left, left_path, right, right_path):
        if left is not None and right is not None:
            require(left == right, f"conflicting exclusion values at {left_path} and {right_path}")
        return left if left is not None else right

    rows = []
    for fallback_id, index, source_task in entries:
        raw = mapping(source_task, f"manifest.tasks[{index}]")
        task_id = raw.get("task_id", fallback_id)
        info = manifest.tasks[task_id]
        pointer = f"/tasks/{index}" if not isinstance(raw_tasks, Mapping) else f"/tasks/{fallback_id}"
        record = mapping(raw.get("task_record", {}), pointer + ".task_record")
        record_prefix = pointer + ".task_record"
        if "record" in record:
            record = mapping(record["record"], pointer + ".task_record.record")
            record_prefix += ".record"
        target = mapping(record.get("operation_target", {}), pointer + ".operation_target")
        geometry = mapping(target.get("geometry", {}), pointer + ".geometry")
        params = mapping(geometry.get("params", {}), pointer + ".geometry.params")
        excluded = number(target, "excluded_volume_mm3", pointer)
        grid_excluded = same_declared_values(
            number(raw, "grid_excluded_volume_mm3", pointer),
            pointer + ".grid_excluded_volume_mm3",
            number(params, "grid_excluded_volume_mm3", pointer + ".params"),
            pointer + ".params.grid_excluded_volume_mm3")
        tolerance = same_declared_values(
            number(raw, "evaluator_exclusion_tolerance_mm3", pointer),
            pointer + ".evaluator_exclusion_tolerance_mm3",
            number(params, "grid_exclusion_tolerance_mm3", pointer + ".params"),
            pointer + ".params.grid_exclusion_tolerance_mm3")
        continuous = same_declared_values(
            number(raw, "continuous_exclusion_diagnostic_mm3", pointer),
            pointer + ".continuous_exclusion_diagnostic_mm3",
            number(params, "continuous_excluded_volume_mm3", pointer + ".params"),
            pointer + ".params.continuous_excluded_volume_mm3")
        if tolerance == 0.0 and grid_excluded is not None and excluded is not None:
            require(grid_excluded == excluded,
                    f"{task_id}: zero-tolerance grid/primary exclusion volumes disagree")
        reason = target.get("exclusion_reason")
        require(reason is None or isinstance(reason, str), f"{task_id}: exclusion_reason must be text")
        kind = geometry.get("kind")
        require(kind is None or isinstance(kind, str), f"{task_id}: geometry.kind must be text or null")
        pointers = {
            "task_hash": pointer + ".task_hash", "family": pointer + ".family",
            "excluded_volume_mm3": (record_prefix + ".operation_target.excluded_volume_mm3"
                                      if excluded is not None else None),
            "geometry_source_kind": (record_prefix + ".operation_target.geometry.kind"
                                     if kind is not None else None),
            "exclusion_reason": (record_prefix + ".operation_target.exclusion_reason"
                                 if reason is not None else None),
        }
        for column, nested in (
            ("grid_excluded_volume_mm3", "grid_excluded_volume_mm3"),
            ("grid_exclusion_tolerance_mm3", "grid_exclusion_tolerance_mm3"),
            ("continuous_exclusion_diagnostic_mm3", "continuous_excluded_volume_mm3"),
        ):
            value = raw.get(column) if column != "continuous_exclusion_diagnostic_mm3" else continuous
            pointers[column] = None
            if value is not None:
                pointers[column] = pointer + "." + column
            elif params.get(nested) is not None:
                pointers[column] = record_prefix + ".operation_target.geometry.params." + nested
        row = {
            "task_id": task_id,
            "values": {
                "task_hash": info.task_hash, "family": info.family,
                "excluded_volume_mm3": excluded,
                "grid_excluded_volume_mm3": grid_excluded,
                "grid_exclusion_tolerance_mm3": tolerance,
                "continuous_exclusion_diagnostic_mm3": continuous,
                "geometry_source_kind": kind, "exclusion_reason": reason,
            },
            "pointers": pointers,
        }
        rows.append(row)
    return rows


def build_artifacts(*, audit_path: Path, manifest_path: Path, decisions_path: Path,
                    run_configs: Mapping[str, Path], precision: str,
                    decision_ids: Sequence[str], input_kind: str = "synthetic",
                    ledger_paths: Mapping[str, Path] | None = None,
                    baseline_coverage_path: Path | None = None,
                    refinement_paths: Sequence[Path] = (),
                    run_root: Path | None = None,
                    report_override: Mapping[str, Any] | None = None,
                    report_ref_override: Mapping[str, Any] | None = None,
                    external_decision_binding: bool = False,
                    allow_unavailable_precision_for_noncomplete: bool = False) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Read every input before publishing any bytes; the call itself is side-effect free."""
    require(input_kind in ("synthetic", "smoke", "pilot", "campaign"),
            "input kind must be synthetic/smoke/pilot/campaign")
    require(precision in ("f32", "f64"), "campaign precision must be explicit f32 or f64")
    require(decision_ids and all(isinstance(item, str) and item.startswith("D-")
                                 for item in decision_ids), "declare at least one campaign decision ID")
    decisions = set(decision_ids)
    require(len(decisions) == len(decision_ids), "campaign decision IDs must be unique")
    ledger_paths = dict(ledger_paths or {})
    if report_override is None:
        report, audit_ref = source(audit_path, "audit")
    else:
        report = dict(report_override)
        audit_bytes = json_bytes(report)
        audit_ref = dict(report_ref_override or {"role": "derived_audit",
                                                 "sha256": digest(audit_bytes)})
    report = mapping(report, "audit")
    manifest_raw, manifest_ref = source(manifest_path, "manifest")
    try:
        manifest = A.load_manifest(mapping(manifest_raw, "manifest"))
    except (A.AuditInputError, ValueError) as error:
        raise PaperArtifactError(f"invalid task manifest: {error}") from error
    decision_text, decision_ref = source(decisions_path, "decisions", json_input=False)
    common_sources = {item["role"]: item for item in (audit_ref, manifest_ref)}
    runs = validate_runs(report, manifest)
    if run_root is not None:
        require(not run_configs, "--run-root and --run-config bindings are alternative input modes")
        root = no_manuscript(run_root, "run_root")
        for run_id in sorted(runs):
            require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,200}", run_id) is not None,
                    f"unsafe run ID for --run-root: {run_id!r}")
            run_configs = dict(run_configs)
            run_configs[run_id] = root / run_id / "config.json"
    run_index = {row["run_id"]: index for index, row in enumerate(report["runs"])}
    expected_entries = {item.get("run_id"): index
                        for index, item in enumerate(manifest_raw.get(
                            "expected_runs", manifest_raw.get("runs", [])))}
    config_docs: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
    for run_id, config_path in sorted(run_configs.items()):
        if run_id not in runs:
            raise PaperArtifactError(f"config is bound to unknown run_id {run_id}")
        raw, ref = source(config_path, f"config:{run_id}")
        config_docs[run_id] = (mapping(raw, f"config:{run_id}"), ref)
    validate_config_links(config_docs, manifest)
    for run_id, (_, config_ref) in sorted(config_docs.items()):
        actual, _pointer = validate_precision_and_decisions(
            config_docs[run_id][0], run_id=run_id, precision=precision, decisions=decisions,
            external_decision_binding=external_decision_binding,
            allow_unavailable_precision=(allow_unavailable_precision_for_noncomplete
                                         and runs[run_id]["status"] != "complete"))
        config_docs[run_id][0]["_validated_precision"] = actual
    require(all(name in manifest.runs for name in ledger_paths), "ledger bound to an unknown run ID")
    ledgers = {}
    for run_id, ledger_path in sorted(ledger_paths.items()):
        body, ref = source(ledger_path, f"ledger:{run_id}")
        wrapper = mapping(body, f"ledger:{run_id}")
        ledger = wrapper.get("ledger", wrapper)
        mapping(ledger, f"ledger:{run_id}.ledger")
        for key in COUNTERS:
            value = ledger.get(key)
            require(not isinstance(value, bool) and isinstance(value, int) and value >= 0,
                    f"{run_id}: ledger {key} must be a nonnegative integer")
        for key in COST_TIMES:
            value = ledger.get(key)
            require(not isinstance(value, bool) and isinstance(value, (int, float))
                    and value >= 0, f"{run_id}: ledger {key} must be finite and nonnegative")
        require(isinstance(ledger.get("memory_scope"), str) and ledger["memory_scope"],
                f"{run_id}: ledger memory_scope missing")
        declared = {item[1] for item in precision_values(wrapper, "")}
        if declared:
            require(declared == {precision},
                    f"{run_id}: ledger/config precision differs from campaign {precision}: {declared}")
        ledgers[run_id] = (ledger, ref, "history.ledger" if "ledger" in wrapper else "snapshot")
    baseline = None
    baseline_ref = None
    if baseline_coverage_path is not None:
        baseline, baseline_ref = source(baseline_coverage_path, "baseline_coverage")
        baseline = mapping(baseline, "baseline_coverage")
        require(baseline.get("schema") == "diffcam-paper-coverage/1",
                "baseline coverage schema must be diffcam-paper-coverage/1")
        require(isinstance(baseline.get("items"), list), "baseline coverage items must be a list")
    inline_refinements = refinement_from_report(report, sorted(runs))
    refinements: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
    for run_id, link in sorted(inline_refinements.items()):
        refinements[run_id] = (link, audit_ref)
    for path in sorted(refinement_paths, key=lambda item: str(item)):
        record, ref = load_refinement_file(path, sorted(runs), runs, manifest, run_configs)
        run_id = record["primary_run_id"]
        require(run_id not in refinements,
                f"multiple refinement links for {run_id}; choose one audited file explicitly")
        refinements[run_id] = (record, ref)

    texts = list(decision_text.splitlines())
    decisions_found = {}
    for index, line in enumerate(texts):
        match = DECISION_PATTERN.match(line)
        if match:
            end = next((i for i in range(index + 1, len(texts)) if texts[i].startswith("## D-")), len(texts))
            decisions_found[match[1]] = {"id": match[1], "title": match[2],
                                         "text": "\n".join(texts[index + 1:end]).strip(),
                                         "line": index + 1}
    for required_id in ("D-2026-11-09-03", "D-2026-11-09-05"):
        require(required_id in decisions_found, f"decision record lacks {required_id}")
    require("provisional" in decisions_found["D-2026-11-09-03"]["text"].casefold(),
            "D-03 gradient provisional note is missing")
    require("full float64" in decisions_found["D-2026-11-09-05"]["title"].casefold(),
            "D-05 full-float64 precision note is missing")
    require(decisions.issubset(decisions_found),
            f"campaign decision IDs missing from record: {sorted(decisions - set(decisions_found))}")

    emit = Emit(common_sources)
    config_ref_for = {run_id: {"decisions": decision_ref,
                              f"config:{run_id}": config_docs[run_id][1]} for run_id in runs}

    def nc_hashes_for(run_id):
        return sorted({raw["declared_nc_hash"] for raw in runs[run_id]["dialects"].values()
                       if raw.get("declared_nc_hash") is not None})

    def row_evidence(row_id, ids, tasks, *, nc_hashes=None):
        return {"row_id": row_id, "run_ids": sorted(ids), "task_ids": sorted(tasks),
                "task_hashes": sorted({manifest.tasks[item].task_hash for item in tasks}),
                "nc_hashes": sorted(set(nc_hashes or [])),
                "decision_ids": sorted(decisions), "input_kind": input_kind}

    run_columns = [
        "run_id", "task_id", "task_hash", "method", "seed", "nominal_anchor", "dialect",
        "run_status", "coverage", "safety_pass", "geometry_pass", "completion_pass",
        "efficiency_eligible", "no_cut",
        "reported_safety_pass", "reported_geometry_pass", "reported_completion_pass",
        "reported_efficiency_eligible", "disagreement_fields",
        *RUN_METRICS, "declared_nc_hash", "actual_nc_hash", "evaluator_id",
        "precision", "precision_source_reason", "ledger_status",
    ]
    for run_id in sorted(runs):
        expected = manifest.runs[run_id]
        cfg, cfg_ref = config_docs[run_id]
        for dialect in ("rs274", "haas"):
            raw = runs[run_id]["dialects"].get(dialect)
            row_status = "not_provided" if raw is None else raw.get("status")
            values = {key: None for key in RUN_METRICS}
            safety = geometry = completion = eligibility = no_cut = None
            evaluator_id = None
            if raw is not None and raw.get("evaluation") is not None:
                evaluated = typed_evaluation(raw["evaluation"], f"{run_id}.{dialect}")
                safety = evaluated.safety_pass
                geometry = evaluated.geometry_pass
                task = manifest.tasks[expected.task_id]
                completion = outcome(evaluated, task, "completion_pass")
                eligibility = outcome(evaluated, task, "efficiency_eligible")
                removed = metric_value(evaluated, "removed_volume_mm3")
                if removed is None:
                    no_cut = None
                elif not task.removal_required:
                    no_cut = False
                else:
                    no_cut = removed <= 0
                values = {name: metric_value(evaluated, name) for name in RUN_METRICS}
                evaluator_id = evaluated.evaluator_id
            reported_row = runs[run_id].get("reported_outcomes")
            reported_prefix = f"/runs/{run_index[run_id]}/reported_outcomes"
            reported = {"safety_pass": None, "geometry_pass": None,
                        "completion_pass": None, "efficiency_eligible": None}
            disagreement = None
            if reported_row is not None:
                reported_map = mapping(reported_row, f"{run_id}.reported_outcomes")
                if "outcomes" in reported_map:
                    nested = mapping(reported_map["outcomes"], f"{run_id}.reported_outcomes.outcomes")
                    selected = nested.get(dialect)
                    require(isinstance(selected, Mapping),
                            f"{run_id}.{dialect}: reported dialect outcome link missing")
                    reported_row = selected
                    reported_prefix += f"/outcomes/{dialect}"
                else:
                    reported_row = reported_map
                for flag in OUTCOME_FIELDS:
                    if flag in reported_row:
                        require(reported_row[flag] is None or isinstance(reported_row[flag], bool),
                                f"{run_id}: reported {flag} must be true/false/null")
                        reported[flag] = reported_row[flag]
                disagreement = runs[run_id]["reported_outcomes"].get(
                    "screen_disagreement_fields", disagreement)
            cells = {
                "run_id": run_id, "task_id": expected.task_id,
                "task_hash": runs[run_id]["task_hash"], "method": expected.method,
                "seed": expected.seed, "nominal_anchor": nominal_method(expected.method),
                "dialect": dialect, "run_status": runs[run_id]["status"],
                "coverage": row_status if raw is not None else "unavailable",
                "safety_pass": safety, "geometry_pass": geometry,
                "completion_pass": completion, "efficiency_eligible": eligibility,
                "no_cut": no_cut, "disagreement_fields": disagreement,
                **{f"reported_{key}": value for key, value in reported.items()},
                **values,
                "declared_nc_hash": raw.get("declared_nc_hash") if raw else None,
                "actual_nc_hash": raw.get("actual_nc_hash") if raw else None,
                "evaluator_id": evaluator_id, "precision": cfg["_validated_precision"],
                "precision_source_reason": "declared metadata matched campaign; not a re-execution of numerical precision",
                "ledger_status": "linked" if run_id in ledgers else "unavailable",
            }
            base = f"/runs/{run_index[run_id]}"
            pointers = {"run_id": base + "/run_id", "task_hash": base + "/task_hash",
                        "run_status": base + "/status", "coverage": (
                            base + f"/dialects/{dialect}/status" if raw else None),
                        "safety_pass": base + f"/dialects/{dialect}/evaluation/safety_pass",
                        "geometry_pass": base + f"/dialects/{dialect}/evaluation/geometry_pass",
                        "declared_nc_hash": base + f"/dialects/{dialect}/declared_nc_hash",
                        "actual_nc_hash": base + f"/dialects/{dialect}/actual_nc_hash",
                        "evaluator_id": base + f"/dialects/{dialect}/evaluation/evaluator_id",
                        **{f"reported_{name}": reported_prefix + f"/{name}" if reported_row else None
                           for name in OUTCOME_FIELDS}}
            for key in ("task_id", "method", "seed"):
                pointers[key] = f"manifest#/expected_runs/{expected_entries[run_id]}/{key}"
            for name in RUN_METRICS:
                pointers[name] = base + f"/dialects/{dialect}/evaluation/metrics/{name}/value"
            emit.row("run_status", run_columns, cells,
                     evidence=row_evidence(f"{run_id}/{dialect}", [run_id], [expected.task_id],
                                           nc_hashes=nc_hashes_for(run_id)),
                     pointers=pointers, claim_kind="run_and_program_outcome",
                     extra_sources=config_ref_for[run_id])

    resource_columns = ["run_id", "method", "task_hash", "precision", "status", "ledger_availability",
                        *COUNTERS, *COST_TIMES, "memory_scope", "ledger_source_shape"]
    for run_id in sorted(runs):
        expected = manifest.runs[run_id]
        source_ledger = ledgers.get(run_id)
        if source_ledger is None:
            cells = {column: None for column in resource_columns}
            cells.update({"run_id": run_id, "method": expected.method,
                          "task_hash": manifest.tasks[expected.task_id].task_hash,
                          "precision": precision, "status": runs[run_id]["status"],
                          "ledger_availability": "unavailable",
                          "memory_scope": "no ledger supplied; missing values are not zero",
                          "ledger_source_shape": None})
            ledger_ref = {}
            pointers = {"ledger_availability": None, "memory_scope": None}
        else:
            ledger, ref, shape = source_ledger
            cells = {key: ledger[key] for key in (*COUNTERS, *COST_TIMES)}
            cells.update({"run_id": run_id, "method": expected.method,
                          "task_hash": manifest.tasks[expected.task_id].task_hash,
                          "precision": precision, "status": runs[run_id]["status"],
                          "ledger_availability": "linked", "memory_scope": ledger["memory_scope"],
                          "ledger_source_shape": shape})
            ledger_ref = {f"ledger:{run_id}": ref}
            prefix = "/ledger/" if shape == "history.ledger" else "/"
            pointers = {key: prefix + key for key in (*COUNTERS, *COST_TIMES, "memory_scope")}
        emit.row("resource_cost", resource_columns, cells,
                 evidence=row_evidence(f"resources/{run_id}", [run_id], [expected.task_id],
                                       nc_hashes=nc_hashes_for(run_id)),
                 pointers=pointers, claim_kind="declared_compute_resource_measurement",
                 extra_sources=ledger_ref)

    outcome_columns = ["comparison", "family", "dialect", "outcome", "side",
                       "known_task_units", "expected_task_units", "task_mean_rate",
                       "ci_lower", "ci_upper", "estimate_method",
                       "paired_mean_difference", "paired_ci_lower", "paired_ci_upper",
                       "small_n_warning", "holm_p"]
    compare_columns = [
        "comparison", "family", "dialect", "metric", "paired_tasks", "method_mean",
        "baseline_mean", "mean_difference", "ci_lower", "ci_upper", "interval_null_reason",
        "raw_sign_flip_p", "holm_p", "small_n_warning", "ci_resampling_method",
        "sign_flip_method", "sign_flip_extreme", "sign_flip_total",
        "dropped_failed_or_ineligible", "dropped_missing_or_null", "independent_task_ids",
    ]
    for comparison_index, comparison in enumerate(report["comparisons"]):
        row = mapping(comparison, f"comparisons[{comparison_index}]")
        family = row.get("family")
        method_runs = [item for item in manifest.runs.values()
                       if item.method == row["method"] and (family == "overall" or
                       manifest.tasks[item.task_id].family == family)]
        baseline_runs = [item for item in manifest.runs.values()
                         if item.method == row["baseline"] and (family == "overall" or
                         manifest.tasks[item.task_id].family == family)]
        all_ids = sorted({item.run_id for item in method_runs + baseline_runs})
        for metric, summary_raw in row["metrics"].items():
            summary = mapping(summary_raw, f"comparison metric {metric}")
            available_a = mapping(summary["method_task_availability"], f"{metric}.method.tasks")
            available_b = mapping(summary["baseline_task_availability"], f"{metric}.baseline.tasks")
            require(set(available_a) == set(available_b) and set(available_a).issubset(manifest.tasks),
                    f"{metric}: method/baseline task links differ or are outside the manifest")
            task_ids = sorted(task_id for task_id in available_a
                              if available_a[task_id].get("value") is not None
                              and available_b[task_id].get("value") is not None)
            require(len(task_ids) == summary["paired_tasks"],
                    f"{metric}: paired task count disagrees with task links; possible seed-level pseudoreplication")
            require("small_n_warning" in summary and "holm_p" in summary,
                    f"{metric}: missing uncertainty/warning metadata")
            if "bootstrap_p" not in summary:
                require(summary["paired_tasks"] == 0,
                        f"{metric}: nonempty paired comparison is missing its sign-flip p-value")
            interval = summary.get("interval")
            require(interval is None or isinstance(interval, Mapping), f"{metric}: malformed CI")
            nc_hashes = [value for item in all_ids for value in nc_hashes_for(item)]
            source_configs = {"config:" + run_id: config_docs[run_id][1] for run_id in all_ids}
            source_configs["decisions"] = decision_ref
            cells = {
                "comparison": row["comparison"], "family": family, "dialect": row["dialect"],
                "metric": metric, "paired_tasks": summary["paired_tasks"],
                "method_mean": summary["method_mean"], "baseline_mean": summary["baseline_mean"],
                "mean_difference": summary["mean_difference"],
                "ci_lower": interval.get("lower") if interval else None,
                "ci_upper": interval.get("upper") if interval else None,
                "interval_null_reason": summary.get("interval_null_reason"),
                "raw_sign_flip_p": summary.get("bootstrap_p"), "holm_p": summary.get("holm_p"),
                "small_n_warning": summary.get("small_n_warning"),
                "ci_resampling_method": summary.get("ci_resampling_method"),
                "sign_flip_method": summary.get("sign_flip_method"),
                "sign_flip_extreme": summary.get("sign_flip_extreme"),
                "sign_flip_total": summary.get("sign_flip_total"),
                "dropped_failed_or_ineligible": summary.get(
                    "dropped_because_side_failed_or_ineligible"),
                "dropped_missing_or_null": summary.get("dropped_missing_or_null_evidence"),
                "independent_task_ids": task_ids,
            }
            pointers = {key: f"/comparisons/{comparison_index}/metrics/{metric}/{field}"
                        for key, field in (
                            ("paired_tasks", "paired_tasks"), ("method_mean", "method_mean"),
                            ("baseline_mean", "baseline_mean"), ("mean_difference", "mean_difference"),
                            ("interval_null_reason", "interval_null_reason"),
                            ("ci_resampling_method", "ci_resampling_method"),
                            ("raw_sign_flip_p", "bootstrap_p"), ("holm_p", "holm_p"),
                            ("small_n_warning", "small_n_warning"),
                            ("sign_flip_method", "sign_flip_method"),
                            ("sign_flip_extreme", "sign_flip_extreme"),
                            ("sign_flip_total", "sign_flip_total"),
                            ("dropped_failed_or_ineligible", "dropped_because_side_failed_or_ineligible"),
                            ("dropped_missing_or_null", "dropped_missing_or_null_evidence"))}
            if interval:
                pointers.update({"ci_lower": f"/comparisons/{comparison_index}/metrics/{metric}/interval/lower",
                                 "ci_upper": f"/comparisons/{comparison_index}/metrics/{metric}/interval/upper"})
            if summary.get("bootstrap_p") is None:
                pointers["raw_sign_flip_p"] = None
            emit.row("metric_comparisons", compare_columns, cells,
                     evidence=row_evidence(f"comparison-{comparison_index}/{family}/{metric}", all_ids, task_ids,
                                           nc_hashes=nc_hashes), pointers=pointers,
                     claim_kind="paired_task_metric_comparison", statistical_unit="task",
                     extra_sources=source_configs)
        outcome_summary = mapping(row["outcomes"], f"comparisons[{comparison_index}].outcomes")
        for outcome_name, container in outcome_summary.items():
            difference = container["paired_difference"]
            for side in ("method", "baseline"):
                rate = container[side]
                ids = method_runs if side == "method" else baseline_runs
                task_ids = sorted({item.task_id for item in ids})
                cells = {
                    "comparison": row["comparison"], "family": family, "dialect": row["dialect"],
                    "outcome": outcome_name, "side": side,
                    "known_task_units": rate["known_task_units"],
                    "expected_task_units": rate["expected_task_units"],
                    "task_mean_rate": rate["task_mean_rate"],
                    "ci_lower": rate["interval"].get("lower") if rate["interval"] else None,
                    "ci_upper": rate["interval"].get("upper") if rate["interval"] else None,
                    "estimate_method": rate["estimate_method"],
                    "paired_mean_difference": difference["mean_difference"],
                    "paired_ci_lower": difference["interval"].get("lower") if difference["interval"] else None,
                    "paired_ci_upper": difference["interval"].get("upper") if difference["interval"] else None,
                    "small_n_warning": difference.get("small_n_warning"),
                    "holm_p": difference.get("holm_p"),
                }
                pointers = {key: f"/comparisons/{comparison_index}/outcomes/{outcome_name}/{side}/{name}"
                            for key, name in (
                                ("known_task_units", "known_task_units"),
                                ("expected_task_units", "expected_task_units"),
                                ("task_mean_rate", "task_mean_rate"),
                                ("estimate_method", "estimate_method"))}
                difference_prefix = f"/comparisons/{comparison_index}/outcomes/{outcome_name}/paired_difference"
                pointers.update({"paired_mean_difference": difference_prefix + "/mean_difference",
                                 "small_n_warning": difference_prefix + "/small_n_warning",
                                 "holm_p": difference_prefix + "/holm_p"})
                emit.row("outcome_comparisons", outcome_columns, cells,
                         evidence=row_evidence(f"comparison-{comparison_index}/{outcome_name}/{side}",
                                              [item.run_id for item in ids], task_ids,
                                              nc_hashes=[h for item in ids for h in nc_hashes_for(item.run_id)]),
                         pointers=pointers, claim_kind="paired_task_outcome_rate",
                         statistical_unit="task",
                         extra_sources={"decisions": decision_ref})

    scope_columns = ["task_id", "task_hash", "family", "excluded_volume_mm3",
                     "grid_excluded_volume_mm3", "grid_exclusion_tolerance_mm3",
                     "continuous_exclusion_diagnostic_mm3", "geometry_source_kind", "exclusion_reason"]
    for index, scope in enumerate(task_scope_rows(manifest_raw, manifest)):
        run_ids = sorted(run_id for run_id, expected in manifest.runs.items()
                         if expected.task_id == scope["task_id"])
        emit.row("task_scope", scope_columns,
                 {"task_id": scope["task_id"], **scope["values"]},
                 evidence=row_evidence(f"scope/{scope['task_id']}", run_ids, [scope["task_id"]],
                                       nc_hashes=[h for run_id in run_ids for h in nc_hashes_for(run_id)]),
                 pointers=scope["pointers"], claim_kind="declared_task_grid_and_exclusion_scope",
                 extra_sources={"decisions": decision_ref})
    coverage_columns = ["category", "scope", "status", "reason", "run_id", "task_hash",
                        "declared_nc_hash", "disagreement_fields", "source_declaration"]
    def coverage_row(category, scope, status, reason, run_id=None, nc_hash=None,
                     disagreement=None, source_ref=None, key=None):
        task_hash = manifest.tasks[manifest.runs[run_id].task_id].task_hash if run_id else None
        emit.row("coverage", coverage_columns,
                 {"category": category, "scope": scope, "status": status, "reason": reason,
                  "run_id": run_id, "task_hash": task_hash, "declared_nc_hash": nc_hash,
                  "disagreement_fields": disagreement, "source_declaration": status},
                 evidence=row_evidence(f"coverage/{key or category}/{run_id or scope}",
                                      [run_id] if run_id else [],
                                      [manifest.runs[run_id].task_id] if run_id else [],
                                      nc_hashes=[nc_hash] if nc_hash else []),
                 pointers={"status": None, "reason": None, "source_declaration": None},
                 claim_kind="explicit_failure_unavailable_or_disagreement",
                 extra_sources=source_ref or {"decisions": decision_ref})

    for run_id in sorted(runs):
        status = runs[run_id]["status"]
        if status != "complete":
            coverage_row("recorded_run_failure", status, "recorded_noncomplete",
                         "valid terminal run record retained; not treated as success or zero compute-cost",
                         run_id, key="run-status")
        for dialect, raw in sorted(runs[run_id]["dialects"].items()):
            if raw.get("evaluation") is not None:
                evaluated = typed_evaluation(raw["evaluation"], f"{run_id}.{dialect}")
                if metric_value(evaluated, "removed_volume_mm3") == 0 and manifest.tasks[
                        manifest.runs[run_id].task_id].removal_required:
                    coverage_row("no_cut", dialect, "completion_pass=false_and_efficiency_ineligible",
                                 "zero removal on a removal-required task; no efficiency bar is inferred",
                                 run_id, raw["declared_nc_hash"], key="no-cut")
            if raw.get("evaluation") is None:
                coverage_row("program_route", dialect, raw.get("status", "unavailable"),
                             "unsupported/unavailable posted result retained; no NC value invented",
                             run_id, raw.get("declared_nc_hash"), key="program-route")
        if run_id not in refinements:
            coverage_row("refinement", "conservative_or_refined_outcomes", "not_provided",
                         "no validated refinement link supplied; this is not zero disagreements or proof of no disagreement",
                         run_id, key="refinement-unavailable")
    # A selected subset of explicit refinement links is allowed. An inline
    # upstream refinement declaration has already required all expected IDs;
    # every absent optional link remains an explicit unavailable coverage row.
    for run_id, (refined, ref) in sorted(refinements.items()):
        conservative = conservative_consistency(
            refined.get("conservative_outcome"), f"{run_id}.conservative_outcome")
        extra = {"refinement": ref}
        if refined.get("validation"):
            cells_extra = {"rule": refined["validation"]}
        else:
            cells_extra = {"rule": "upstream inline refinement link; supplied raw bundle not re-read"}
        facts = {"conservative_outcome": conservative, **cells_extra}
        if isinstance(refined.get("comparison"), Mapping):
            facts["comparison"] = dict(refined["comparison"])
        coverage_row("refinement_disagreement" if conservative["refinement_disagreement"] else "refinement",
                     run_id, "disagreement" if conservative["refinement_disagreement"] else "no_recorded_disagreement",
                     json.dumps(facts, sort_keys=True),
                     run_id,
                     nc_hash=runs[run_id]["dialects"].get("rs274", {}).get("declared_nc_hash"),
                     disagreement=conservative["disagreement_fields"],
                     source_ref=extra, key="refinement")
    resources_by_run = mapping(report.get("refinement", {}), "refinement").get("resources_by_run_id", {})
    for run_id, raw in sorted(mapping(resources_by_run, "refinement.resources_by_run_id").items()):
        require(run_id in runs, "refinement resources reference unknown primary run")
        coverage_row("refinement_resources", run_id, "linked_auxiliary_resource_record",
                     json.dumps(raw, allow_nan=False, sort_keys=True), run_id,
                     key="refinement-resource")

    for run_id in sorted(runs):
        if not nominal_method(manifest.runs[run_id].method):
            coverage_row("optimization_precision", run_id, precision,
                         "configuration metadata matches declared campaign precision; no numerical engine verification is inferred",
                         run_id, key="precision")
    decisions_columns = ["id", "title", "text", "role"]
    for decision_id in sorted(decisions_found):
        record = decisions_found[decision_id]
        role = "historical_gradient_provisional_note" if decision_id == "D-2026-11-09-03" else (
            "matched_comparison_full_f64_precision_policy" if decision_id == "D-2026-11-09-05"
            else "protocol_context")
        emit.row("decisions", decisions_columns,
                 {"id": decision_id, "title": record["title"], "text": record["text"], "role": role},
                 evidence={"row_id": "decision/" + decision_id, "run_ids": [], "task_ids": [],
                           "task_hashes": [], "nc_hashes": [], "decision_ids": [decision_id],
                           "input_kind": input_kind},
                 pointers={"id": f"decisions.md#line-{record['line']}",
                           "title": f"decisions.md#line-{record['line']}",
                           "text": f"decisions.md#heading-{decision_id}", "role": None},
                 claim_kind="declared_protocol_decision",
                 extra_sources={"decisions": decision_ref})

    if baseline is not None:
        for index, item in enumerate(baseline["items"]):
            entry = mapping(item, f"baseline_coverage[{index}]")
            for key in ("kind", "scope", "status", "reason"):
                require(isinstance(entry.get(key), str) and entry[key],
                        f"baseline coverage[{index}].{key} must be nonempty")
            run_id = entry.get("run_id")
            require(run_id is None or run_id in runs,
                    f"baseline coverage[{index}] references unknown run_id")
            nc_hash = hash_field(entry.get("nc_hash"),
                                 f"baseline_coverage[{index}].nc_hash", optional=True)
            if nc_hash is not None:
                require(run_id is not None and nc_hash in nc_hashes_for(run_id),
                        "baseline coverage NC hash is not tied to its audited run")
            coverage_row(entry["kind"], entry["scope"], entry["status"], entry["reason"],
                         run_id, nc_hash, entry.get("disagreement_fields"),
                         source_ref={"baseline_coverage": baseline_ref}, key=f"declared-{index}")
    coverage_row("external_baseline", "CNC-Net executable-NC route",
                 "not_provided_unless_explicitly_declared" if baseline is None else "see_declared_coverage_rows",
                 "no default availability/performance success is inferred for CNC-Net; executable route needs an explicit evidence declaration")
    coverage_row("external_baseline", "CAMotics", "supplementary_not_primary_oracle",
                 "D-01 keeps CAMotics supplementary; null/non-watertight/unsupported outcomes are not silently treated as passing cross-checks")
    coverage_row("soft_hard_posted", "evaluator_roles", "see_per_run_and_refinement_rows",
                 "raw primary/report/refinement roles remain separate; an unavailable fidelity comparison is not reclassified as agreement")
    if not report["comparisons"]:
        # An evidence-complete but comparison-free input must still regenerate,
        # without inventing a missing baseline pair or dropping its run rows.
        emit.columns["metric_comparisons"] = compare_columns
        emit.columns["outcome_comparisons"] = outcome_columns
    emit.every_table({
        "run_status": "Run/program outcomes with full row provenance in claim_evidence.json",
        "task_scope": "Declared task-grid target/exclusion scope; continuous values remain diagnostic",
        "resource_cost": "Declared compute resources (not machining efficiency or device-only memory)",
        "metric_comparisons": "Task-paired metric comparisons supplied by the audit; no seed pseudoreplication",
        "outcome_comparisons": "Task-level outcome comparisons supplied by the audit",
        "coverage": "Explicit failures, no-cut outcomes, unsupported routes, refinement unavailability and disagreements",
        "decisions": "Immutable protocol decision-record excerpts",
    })
    # Figure data is identical to source tables, never a second numeric pipeline.
    emit.files["figure_data/metric_comparisons.csv"] = emit.files["tables/metric_comparisons.csv"]
    emit.files["figure_data/resource_cost.csv"] = emit.files["tables/resource_cost.csv"]
    emit.files["figure_data/outcome_comparisons.csv"] = emit.files["tables/outcome_comparisons.csv"]
    emit.files["figure_data/plot_resource_cost.py"] = OPTIONAL_MPL_SCRIPT.encode("utf-8")
    figure_plan = {
        "schema": "diffcam-paper-figure-data/1",
        "rendering_status": "not_rendered_by_core_scaffold",
        "optional_renderer": "figure_data/plot_resource_cost.py",
        "optional_dependency": "matplotlib in a separately declared/provisioned environment; absent from the core lock",
        "roles": ["candidate-compute wall time and process peak RSS; not CNC efficiency or device-only memory"],
        "source_tables": {
            "metric_comparisons": {"file": "figure_data/metric_comparisons.csv",
                                   "claims_artifact": "tables/metric_comparisons.csv",
                                   "statistical_unit": "task"},
            "outcome_comparisons": {"file": "figure_data/outcome_comparisons.csv",
                                    "claims_artifact": "tables/outcome_comparisons.csv",
                                    "statistical_unit": "task"},
            "resource_cost": {"file": "figure_data/resource_cost.csv",
                              "claims_artifact": "tables/resource_cost.csv",
                              "statistical_unit": "not_an_estimate"},
        },
        "missing_values": "NA is unavailable; never plotted as zero or success",
    }
    emit.files["figure_data/figure_plan.json"] = json_bytes(figure_plan)
    emit.files["README.txt"] = (
        "T7.2 SCAFFOLD ONLY; no final campaign or machining-safety claim.\n"
        "Start with artifact_manifest.json for immutable input/output hashes. claim_evidence.json "
        "maps every generated table row and cell to its source pointer(s), run/task/NC hashes.\n"
        "CSV values NA mean unknown/unavailable, not zero or success. LaTeX shortens hash display; "
        "the claim map and CSV retain full hashes. Figures are data CSVs plus an optional plotting "
        "source that requires separately provisioned matplotlib. Core output does not claim a "
        "rendered or visually validated figure. Per-run precision/decision metadata must match the "
        "explicit campaign declaration; no alternative default or mixed-precision comparison is hidden.\n"
        "Regenerate into a fresh or empty explicit destination outside manuscript/. Absolute host "
        "paths, timestamps and git dirty-state metadata are intentionally excluded from generated "
        "bytes, so equal source bytes and campaign declaration produce byte-identical outputs.\n"
    ).encode("utf-8")
    emit.files["claim_evidence.json"] = json_bytes({
        "schema": "diffcam-paper-claim-evidence/1", "claims": emit.claims,
        "link_rule": "claim_id -> artifact + row_number_zero_based -> ordered cells -> source pointers and immutable identities",
        "short_latex_hash_rule": "LaTeX abbreviates hashes only for display; full hashes remain here and in CSV",
    })
    source_inventory = list(common_sources.values())
    source_inventory.extend(config_docs[key][1] for key in sorted(config_docs))
    source_inventory.extend(value[1] for _, value in sorted(ledgers.items()))
    source_inventory.append(decision_ref)
    if baseline_ref:
        source_inventory.append(baseline_ref)
    for run_id, refined_source in sorted(refinements.items()):
        ref = refined_source[1]
        if ref not in source_inventory:
            source_inventory.append(ref)
    sorted_sources = sorted(source_inventory, key=lambda item: (item["role"], item["sha256"]))
    manifest_document = {
        "schema": SCHEMA, "input_kind": input_kind,
        "phase": "scaffold_only_not_final_campaign_evidence",
        "final_frozen_campaign_validated": False,
        "audit_schema_version": report["schema_version"],
        "audit_evidence_complete": report["evidence_complete"],
        "all_runs_completed": report.get("all_runs_completed"),
        "campaign_precision": precision,
        "campaign_decision_ids": sorted(decisions),
        "precision_validation": {
            "rule": "each bound run config must explicitly match campaign simulator_precision and decision_ids",
            "scope": "metadata/protocol consistency; this builder does not independently execute the numerical precision implementation",
        },
        "statistical_policy": {name: report["policy"].get(name) for name in (
            "independent_unit", "seed_aggregation", "pairing", "p_value", "small_sample_rule",
            "confidence", "clopper_pearson", "efficiency_rule", "completion_rule")},
        "input_file_identities": sorted_sources,
        "builder_source_sha256": digest(Path(__file__).read_bytes()),
        "output_policy": "fresh/empty destination only; reject all manuscript/ paths; relative/absolute build paths are not embedded",
        "determinism": "UTF-8 canonical JSON/CSV/LaTeX; sorted source order; no wall-clock timestamps, random draws, host paths or git dirty-state macros",
        "optional_plotting_status": "blocked in core unless separately provisioned matplotlib; not a core rendering pass",
        "file_count_excluding_this_manifest": len(emit.files) + 1,
    }
    files_with_digests = sorted((path, digest(payload), len(payload))
                                for path, payload in emit.files.items())
    manifest_document["files"] = [
        {"path": path, "sha256": sha, "bytes": size} for path, sha, size in files_with_digests
    ]
    manifest_document["files"].append({
        "path": "artifact_manifest.json", "sha256": None,
        "bytes": None, "note": "self-hash omitted intentionally; all other output bytes are hashed",
    })
    manifest_document["files"].sort(key=lambda item: item["path"])
    emit.files["artifact_manifest.json"] = json_bytes(manifest_document)
    finite_tree(manifest_document, "artifact_manifest")
    return emit.files, manifest_document


def _comparison_specs(manifest: A.Manifest):
    if "nominal-cam" not in {run.method for run in manifest.runs.values()}:
        return []
    methods = sorted({run.method for run in manifest.runs.values() if run.method != "nominal-cam"})
    return [{"name": f"{method}-versus-nominal-{dialect.value}", "method": method,
             "baseline": "nominal-cam", "dialect": dialect.value,
             "metrics": ["cycle_time_s", "air_fraction"]}
            for method in methods for dialect in C.Dialect]


def _filtered_manifest(document, manifest: A.Manifest, *, test_split_only=False):
    """A selection of immutable full-manifest records, never a new task config."""
    raw_tasks = document["tasks"]
    if isinstance(raw_tasks, Mapping):
        entries = [(key, value) for key, value in raw_tasks.items()]
    else:
        entries = [(None, value) for value in raw_tasks]
    keep = set()
    for key, value in entries:
        task_id = value.get("task_id", key)
        if test_split_only:
            if value.get("task_split", value.get("split")) != "test":
                continue
        keep.add(task_id)
    if isinstance(raw_tasks, Mapping):
        tasks = {key: value for key, value in raw_tasks.items()
                 if value.get("task_id", key) in keep}
    else:
        tasks = [dict(value) for value in raw_tasks
                 if value.get("task_id") in keep]
    runs = [raw for raw in document.get("expected_runs", document.get("runs", []))
            if raw["task_id"] in keep]
    require(bool(keep and runs), "no test-split/evaluable campaign tasks selected")
    filtered = dict(document, tasks=tasks, expected_runs=runs,
                    comparisons=_comparison_specs(manifest))
    return A.load_manifest(filtered), filtered


def _pilot_verdicts(root: Path, published: Mapping[str, Any], manifest: A.Manifest):
    refinement = mapping(published.get("refinement"), "pilot.refinement")
    links_ref = refinement.get("links_by_run_id")
    if not isinstance(links_ref, Mapping):
        raise PaperArtifactError("pilot refinement links are unavailable")
    evaluations, references, source_hashes = {}, [], {}
    for run_id in sorted(manifest.runs):
        pointer = mapping(links_ref.get(run_id), f"{run_id}.reported refinement link")
        relative = PurePosixPath(pointer.get("path", ""))
        require(not relative.is_absolute() and "." not in relative.parts,
                f"{run_id}: unsafe refinement link path")
        path = (root / relative).resolve()
        require(path.is_relative_to(root.parent.resolve()), f"{run_id}: refinement link escapes pilot bundle")
        if digest(path.read_bytes()) != pointer.get("sha256"):
            raise PaperArtifactError(f"{run_id}: refinement link hash differs from published report")
        raw, ref = source(path, f"refinement_link:{run_id}")
        link = mapping(raw, f"{run_id}.refinement_link")
        require(link.get("schema") == "diffcam-refinement-link/4"
                and link.get("verdict_run_id") == run_id + "-verdict",
                f"{run_id}: invalid schema-4 verdict identity")
        evaluations[run_id], source_hashes[run_id] = {}, {}
        references.append(ref)
        for dialect in C.Dialect:
            item = mapping(link["roles"]["verdict"]["evaluations"].get(dialect.value),
                           f"{run_id}.{dialect.value}.verdict.role")
            require(item.get("sha256") and link["roles"]["verdict"].get("max_step_mm") == 0.125,
                    f"{run_id}.{dialect.value}: no linked 0.125 mm verdict")
            filename = relative_payload(path, item["path"], f"{run_id}.{dialect.value}.verdict.path")
            require(digest(filename.read_bytes()) == item["sha256"],
                    f"{run_id}.{dialect.value}: verdict bytes differ from link")
            doc, evidence = source(filename, f"verdict:{run_id}:{dialect.value}")
            typed = typed_evaluation(doc, f"{run_id}.{dialect.value}")
            require(typed.discretization["settings"]["max_step_mm"] == 0.125,
                    f"{run_id}.{dialect.value}: wrong posted verdict step")
            evaluation = evaluations[run_id][dialect.value] = typed
            references.append(evidence)
            source_hashes[run_id][dialect.value] = evidence["sha256"]
    diagnostics = refinement.get("screen_disagreements_by_run_id")
    require(isinstance(diagnostics, Mapping) and set(diagnostics) == set(evaluations),
            "pilot must expose exact screen-disagreement links for every verdict")
    return evaluations, tuple(diagnostics.items()), references, source_hashes


def _campaign_verdicts(root: Path, manifest: A.Manifest):
    artifact_root = root / "audit-runs" if (root / "audit-runs").is_dir() else root
    evaluations, references = {}, []
    for run_id in sorted(manifest.runs):
        if not (artifact_root / run_id / "run.json").is_file():
            return None, references
        evaluations[run_id] = {}
        for dialect in C.Dialect:
            filename = artifact_root / run_id / C.DIALECT_ARTIFACTS[dialect][1]
            if not filename.is_file():
                return None, references
            doc, ref = source(filename, f"verdict:{run_id}:{dialect.value}")
            typed = typed_evaluation(doc, f"{run_id}/{dialect.value}")
            if typed.discretization["settings"]["max_step_mm"] != 0.125:
                return None, references
            evaluations[run_id][dialect.value] = typed
            references.append(ref)
    return evaluations, references


def _auxiliary_records(root: Path, manifest: A.Manifest):
    configs, metrics, histories, refs, observations = {}, {}, {}, {}, []
    for run_id in sorted(manifest.runs):
        directory = root / run_id
        if not directory.is_dir():
            continue
        record = C.loads((directory / "run.json").read_text(encoding="utf-8"))
        require(isinstance(record, C.RunRecord) and record.run_id == run_id
                and record.task_hash == manifest.tasks[manifest.runs[run_id].task_id].task_hash,
                f"{run_id}: auxiliary result identity mismatch")
        observations.append(record)
        for filename, target in (("config.json", configs), ("metrics.json", metrics),
                                 ("history.json", histories)):
            if (directory / filename).is_file():
                raw, identity = source(directory / filename, f"{filename[:-5]}:{run_id}")
                target[run_id] = mapping(raw, f"{run_id}.{filename}")
                refs[f"{filename[:-5]}:{run_id}"] = identity
    return {"configs": configs, "metrics": metrics, "histories": histories,
            "refs": refs, "observations": observations}


def _task_tolerances(document, configs, manifest):
    """Recover absent task metadata only from a matching, hash-checked TaskSpec."""
    by_task = {}
    for run_id, config in configs.items():
        expected = manifest.runs[run_id]
        source_task = config.get("task")
        if not isinstance(source_task, Mapping):
            continue
        task = C.TaskSpec.from_dict(source_task)
        require(task.task_hash == manifest.tasks[expected.task_id].task_hash,
                f"{run_id}: config's typed task hash differs from the audit manifest")
        value = task.tolerances.completion_residual_mm3
        if expected.task_id in by_task and by_task[expected.task_id] != value:
            raise PaperArtifactError(f"{expected.task_id}: conflicting typed residual tolerances")
        by_task[expected.task_id] = value
    tasks = document["tasks"]
    if isinstance(tasks, Mapping):
        items = list(tasks.items())
        new_tasks = dict(tasks)
    else:
        items = [(None, item) for item in tasks]
        new_tasks = list(tasks)
    for index, (key, task_info) in enumerate(items):
        task_id = task_info.get("task_id", key)
        if task_id in by_task:
            value = dict(task_info)
            declared = value.get("completion_residual_mm3")
            require(declared is None or declared == by_task[task_id],
                    f"{task_id}: manifest and typed TaskSpec residual tolerances disagree")
            value["completion_residual_mm3"] = by_task[task_id]
            new_tasks[index] = value
    return dict(document, tasks=new_tasks)


def _derived_audit_bytes(report):
    """Identity for a generated audit, excluding its host filesystem location.

    The complete host-bound report is still validated and used for all values.
    Only the derived metadata identity replaces run-root/run-directory paths
    with logical task/run identifiers. Immutable input byte hashes are retained.
    """
    neutral = dict(report, runs_root="task-root-relative")
    neutral["runs"] = [dict(row, run_dir=row["run_id"]) for row in report["runs"]]
    return json_bytes(neutral)


def _dataset(label: str, root: Path | None, kind: str):
    if root is None or not root.is_dir():
        return {"label": label, "kind": kind, "state": "absent",
                "reason": "declared input directory is absent", "sources": [],
                "auxiliary": {"configs": {}, "metrics": {}, "histories": {}, "refs": {}, "observations": []}}
    root = no_manuscript(root, f"{label}.root")
    if not (root / "manifest.json").is_file():
        return {"label": label, "kind": kind, "state": "pending",
                "reason": "manifest/report not yet published", "sources": [], "manifest": None,
                "auxiliary": {"configs": {}, "metrics": {}, "histories": {}, "refs": {}, "observations": []}}
    manifest_raw, manifest_ref = source(root / "manifest.json", f"manifest:{label}")
    manifest = A.load_manifest(manifest_raw)
    artifact_root = root / "audit-runs" if (root / "audit-runs").is_dir() else root
    observed = list(artifact_root.glob("*/run.json"))
    auxiliary = _auxiliary_records(artifact_root, manifest)
    manifest_raw = _task_tolerances(manifest_raw, auxiliary["configs"], manifest)
    manifest = A.load_manifest(manifest_raw)
    dataset = {"label": label, "kind": kind, "root": root, "manifest_raw": manifest_raw,
               "manifest_ref": manifest_ref, "manifest": manifest, "artifact_root": artifact_root,
               "auxiliary": auxiliary, "sources": [manifest_ref, *auxiliary["refs"].values()],
               "observed_run_count": len(observed), "expected_run_count": len(manifest.runs)}
    published_path = root / ("audit.json" if (root / "audit.json").is_file() else "pilot_report.json")
    if not published_path.is_file():
        dataset.update(state="pending",
                       reason=f"no finalized audit/report; {len(observed)}/{len(manifest.runs)} run records visible")
        return dataset
    published, reference = source(published_path, f"published_audit:{label}")
    dataset["published"] = published
    dataset["sources"].append(reference)
    summary = mapping(published.get("summary"), f"{label}.summary") if kind != "campaign" else {}
    if kind == "campaign":
        summary = mapping(published.get("summary"), f"{label}.audit.summary")
    reconciliation = mapping(published.get("expected_vs_observed", published.get("reconciliation")),
                             f"{label}.reconciliation")
    if not (published.get("evidence_complete", summary.get("evidence_complete")) is True
            and summary.get("auditor_passed") is True
            and reconciliation.get("exact_id_reconciliation") is True):
        dataset.update(state="pending", reason="published audit is not mechanical/evidence complete")
        return dataset
    complete_records = all(record.status is C.RunStatus.COMPLETE for record in auxiliary["observations"])
    if len(observed) != len(manifest.runs) or not complete_records:
        dataset.update(state="pending",
                       reason="D-04 table waits for terminal completion of every expected run; existing failed/partial rows remain visible")
        return dataset
    try:
        filtered_manifest, _filtered_raw = _filtered_manifest(
            manifest_raw, manifest, test_split_only=kind == "campaign")
        if root / "audit-runs" is not None and (root / "audit-runs").is_dir():
            run_root = root / "audit-runs"
        else:
            run_root = root
        verdicts, sources = _campaign_verdicts(run_root, manifest)
        # A verdict-only campaign does not produce a screen comparison. Its
        # required empty sets record no additional evidence, not proven agreement.
        diagnostics = ({run_id: [] for run_id in manifest.runs}
                       if kind == "campaign" else {})
        if kind != "campaign" and published_path.name == "pilot_report.json":
            verdicts, raw_diagnostics, sources, _hashes = _pilot_verdicts(root, published, manifest)
            diagnostics = dict(raw_diagnostics)
        if verdicts is None:
            dataset.update(state="pending", reason="one or more 0.125 mm typed verdicts are absent")
            return dataset
        report, selected_manifest = A.generate_report(
            run_root, filtered_manifest, bootstrap_seed=20261108,
            verdict_evaluations=verdicts, screen_disagreements=diagnostics)
    except (A.AuditInputError, C.ContractError) as error:
        raise PaperArtifactError(f"{label}: cannot derive its validated audit: {error}") from error
    dataset["sources"].extend(sources)
    report_ref = {"role": f"derived_audit:{label}", "sha256": digest(_derived_audit_bytes(report))}
    dataset.update(state="available", reason="validated immutable audit; all expected runs have D-04 verdicts",
                   report=report, report_ref=report_ref, selected_manifest=selected_manifest,
                   verdicts=verdicts,
                   selected_task_count=len(selected_manifest.tasks),
                   selected_task_ids=sorted(selected_manifest.tasks),
                   expected_task_count=len(filtered_manifest.tasks))
    return dataset


def _csv_rows(files, name):
    import csv as _csv
    import io as _io
    return list(_csv.DictReader(_io.StringIO(files[name].decode("utf-8"))))


def _meta(ref):
    return {"role": ref["role"], "sha256": ref["sha256"]}


def _built_dataset(dataset, decision_ids, decisions_path, precision):
    if dataset["state"] != "available":
        return {"files": {}, "rows": {}, "sources": dataset["sources"]}
    root = dataset["root"]
    artifact_root = dataset["artifact_root"]
    configs = {run_id: artifact_root / run_id / "config.json"
               for run_id in dataset["manifest"].runs}
    histories = {run_id: artifact_root / run_id / "history.json"
                 for run_id in dataset["manifest"].runs
                 if (artifact_root / run_id / "history.json").is_file()}
    files, metadata = build_artifacts(
        audit_path=root / "manifest.json", manifest_path=root / "manifest.json",
        decisions_path=decisions_path, run_configs=configs, precision=precision,
        decision_ids=decision_ids, input_kind=dataset["kind"], ledger_paths=histories,
        report_override=dataset["report"], report_ref_override=dataset["report_ref"],
        external_decision_binding=True, allow_unavailable_precision_for_noncomplete=True)
    return {"files": files, "manifest": metadata,
            "rows": {name: _csv_rows(files, f"tables/{name}.csv")
                     for name in ("run_status", "metric_comparisons", "outcome_comparisons",
                                  "resource_cost", "coverage")},
            "sources": [_meta(item) for item in metadata["input_file_identities"]]}


def _number(value):
    if value is None or value == "NA":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = float(value)
    require(isinstance(value, (int, float)) and math.isfinite(value), "result number must be finite")
    return value


def _init_dataset(dataset):
    if dataset.get("manifest") is None:
        return {}
    auxiliary = dataset["auxiliary"]
    manifest = dataset["manifest"]
    configs, metrics = auxiliary["configs"], auxiliary["metrics"]
    histories = auxiliary["histories"]
    by_id = {}
    for record in auxiliary["observations"]:
        cfg = dict(configs[record.run_id])
        init = mapping(cfg.get("initialization"), "initialization") if isinstance(
            cfg.get("initialization"), Mapping) else {}
        by_id[record.run_id] = {"record": record, "config": cfg, "metrics": metrics.get(record.run_id, {}),
                                "method": manifest.runs[record.run_id].method,
                                "seed": manifest.runs[record.run_id].seed,
                                "history": histories.get(record.run_id, {}),
                                "init": init.get("init", "cam"),
                                "scope": init.get("scope", "primary")}
    return by_id


def _improvement(dataset, method, family):
    manifest = dataset.get("selected_manifest", dataset.get("manifest"))
    if manifest is None:
        return None, None, 0
    ids = [run.run_id for run in manifest.runs.values()
           if run.method == method and not nominal_method(method)
           and (family in (None, "overall") or manifest.tasks[run.task_id].family == family)]
    observations = _init_dataset(dataset)
    flags = [run["metrics"].get("selection", {}).get("no_improvement_over_initialization")
             for run in (observations.get(run_id) for run_id in ids) if run is not None]
    known = [value for value in flags if isinstance(value, bool) and value is not None]
    return ((sum(value is True for value in known) / len(known) if known else None),
            len(known), len(flags))


def _main_rows(dataset, built):
    columns = ["dataset", "state", "reason", "method", "family", "dialect", "expected_tasks",
               "geometry_pass_rate", "geometry_known_n", "safety_pass_rate", "safety_known_n",
               "completion_pass_rate", "completion_known_n", "delta_cycle_s", "cycle_ci_lower",
               "cycle_ci_upper", "cycle_holm_p", "cycle_paired_n", "delta_air_fraction",
               "air_ci_lower", "air_ci_upper", "air_holm_p", "air_paired_n",
               "no_improvement_rate", "no_improvement_known_n", "optimizer_runs",
               "small_sample_warning", "statistical_unit"]
    if dataset["state"] != "available":
        if dataset.get("manifest") is None:
            methods = ["nominal-cam"]
        else:
            methods = sorted({run.method for run in dataset["manifest"].runs.values()})
        rows = []
        for method in methods:
            values = {column: None for column in columns}
            values.update(dataset=dataset["label"], state=dataset["state"], reason=dataset["reason"],
                          method=method, family="overall", statistical_unit="task",
                          expected_tasks=(dataset.get("selected_task_count")))
            rows.append({"table": "main_results", "columns": columns, "values": values,
                         "source_rows": [], "sources": dataset.get("sources", [])})
        return rows, columns, []
    comparison_names = {row["comparison"]: row for row in dataset["report"]["comparisons"]}
    metrics = {(comparison_names[row["comparison"]]["method"], row["family"], row["dialect"], row["metric"]): row
               for row in built["rows"]["metric_comparisons"]}
    outcomes = {(comparison_names[row["comparison"]]["baseline" if row["side"] == "baseline" else "method"],
                 row["family"], row["dialect"], row["outcome"], row["side"]): row
                for row in built["rows"]["outcome_comparisons"]}
    manifest = dataset["selected_manifest"]
    families = ["overall", *sorted({task.family for task in manifest.tasks.values()})]
    methods = sorted({run.method for run in manifest.runs.values()})
    rows, source_rows = [], []
    for method in methods:
        for family in families:
            for dialect in C.Dialect:
                # benchmarks.audit reports one family as `overall` only.
                # Never discard the known statistic in its named family row.
                stat_family = "overall" if len(families) == 2 else family
                outcome_row = outcomes.get((method, stat_family, dialect.value, "geometry_pass", "method")) or (
                    outcomes.get(("nominal-cam", stat_family, dialect.value, "geometry_pass", "baseline"))
                    if nominal_method(method) else None)
                values = {"dataset": dataset["label"], "state": dataset["state"],
                          "reason": dataset["reason"], "method": method, "family": family,
                          "dialect": dialect.value,
                          "expected_tasks": sum(info.family == family or family == "overall"
                                               for info in manifest.tasks.values())}
                evidence_rows = []
                for flag, col in (("geometry_pass", "geometry"), ("safety_pass", "safety"),
                                  ("completion_pass", "completion")):
                    row = outcomes.get((method, stat_family, dialect.value, flag, "method"))
                    if row is None and nominal_method(method):
                        row = outcomes.get(("nominal-cam", stat_family, dialect.value, flag, "baseline"))
                    values[col + "_pass_rate"] = _number(row["task_mean_rate"]) if row else None
                    values[col + "_known_n"] = row["known_task_units"] if row else None
                    if row:
                        side = "baseline" if nominal_method(method) else "method"
                        evidence_rows.append(f"outcome_comparisons#{row['comparison']}/{flag}/{side}")
                for metric, prefix in (("cycle_time_s", "cycle"), ("air_fraction", "air")):
                    row = metrics.get((method, stat_family, dialect.value, metric))
                    values["delta_" + ("cycle_s" if prefix == "cycle" else "air_fraction")] = (
                        _number(row["mean_difference"]) if row else None)
                    values[prefix + "_ci_lower"] = _number(row["ci_lower"]) if row else None
                    values[prefix + "_ci_upper"] = _number(row["ci_upper"]) if row else None
                    values[prefix + "_holm_p"] = _number(row["holm_p"]) if row else None
                    values[prefix + "_paired_n"] = row["paired_tasks"] if row else None
                    if row:
                        evidence_rows.append(f"metric_comparisons#{row['comparison']}/{family}/{metric}")
                rate, known, total = _improvement(dataset, method, family)
                values.update(no_improvement_rate=rate, no_improvement_known_n=known,
                              optimizer_runs=total,
                              small_sample_warning=(outcome_row.get("small_n_warning") if outcome_row else None),
                              statistical_unit="task")
                rows.append({"table": "main_results", "columns": columns, "values": values,
                             "source_rows": evidence_rows, "sources": built["sources"]})
    return rows, columns, source_rows


def _collapse_rules(dataset):
    frozen = dataset["root"] / "frozen_config.json"
    raw = dataset.get("manifest_raw", {})
    candidate = raw.get("collapse_criterion")
    if candidate is None and frozen.is_file():
        document, ref = source(frozen, f"frozen_protocol:{dataset['label']}")
        dataset["sources"].append(ref)
        raw = document.get("raw_config", document)
        candidate = raw.get("collapse_criterion")
    candidate = candidate or {"geometry_or_completion_failure": True,
                              "residual_multiplier": 1.5, "residual_operator": "gt",
                              "require_no_hard_loss_improvement_over_initial": True,
                              "predicate": "provisional D-08 example; not frozen protocol"}
    return mapping(candidate, "collapse_criterion")


def _collapse_case(case, parent, evaluation, parent_evaluation, rules):
    if case is None or parent is None or evaluation is None or parent_evaluation is None:
        return None, "case/parent/typed fine verdict unavailable"
    metric = mapping(case["metrics"], "case metrics")
    selection = mapping(metric.get("selection"), "case selection")
    initial = selection.get("initial_candidate")
    best = metric.get("best_candidate")
    if not isinstance(initial, Mapping) or not isinstance(best, Mapping):
        return None, "hard initial/selected loss unavailable"
    h0, h1 = initial.get("objective"), best.get("objective")
    if not isinstance(h0, (int, float)) or not isinstance(h1, (int, float)):
        return None, "hard initial/selected objective unavailable"
    improved = h1 < h0
    # Use the raw typed task tolerance carried by this run's config/task record.
    task = case["config"].get("task", {})
    tolerance = mapping(task.get("tolerances"), "case task tolerances")
    residual_limit = tolerance.get("completion_residual_mm3")
    removed = metric_value(evaluation, "removed_volume_mm3")
    residual = metric_value(evaluation, "residual_volume_mm3")
    completion = None
    if isinstance(residual_limit, (int, float)):
        completion = residual is not None and removed is not None and removed > 0 and residual <= residual_limit
    if completion is None or evaluation.geometry_pass is None:
        return None, "typed geometry/completion verdict is unavailable"
    geom_bad = evaluation.geometry_pass is False or completion is False
    opposite = parent_evaluation
    parent_residual = metric_value(opposite, "residual_volume_mm3")
    excess = None
    if residual is not None and parent_residual is not None:
        multiplier = _number(rules["residual_multiplier"])
        compare = residual > multiplier * parent_residual if rules["residual_operator"] == "gt" else residual >= multiplier * parent_residual
        excess = compare
    vote = bool(geom_bad if rules.get("geometry_or_completion_failure", True) else False) or bool(excess)
    if rules.get("require_no_hard_loss_improvement_over_initial", True):
        vote = vote and not improved
    return bool(vote), {"hard_loss_initial": h0, "hard_loss_selected": h1,
                        "hard_loss_improved": improved,
                        "geometry_or_completion_failure": geom_bad,
                        "residual_excess_vs_cam": excess,
                        "completion_pass": completion}


def _init_rows(dataset):
    columns = ["dataset", "state", "method", "init", "scope", "included_in_init_axis",
               "expected_runs", "observed_runs", "failed_runs", "collapse_count",
               "collapse_rate", "collapse_criterion", "failure_details", "reason"]
    records = _init_dataset(dataset)
    grouped = {}
    if dataset.get("manifest") is not None:
        for expected in dataset["manifest"].runs.values():
            case = records.get(expected.run_id)
            init = case["init"] if case else "pending"
            scope = case["scope"] if case else "pending"
            key = (expected.method, init, scope)
            grouped.setdefault(key, []).append((expected.run_id, case))
    rules = _collapse_rules(dataset) if dataset["state"] == "available" else {}
    rows = []
    for (method, init, scope), cases in sorted(grouped.items()):
        exploratory = scope == "exploratory_cam_perturbation"
        if init not in ("cam", "straight_line", "random") and not (
                init == "pending" and any(case is None for _, case in cases)):
            raise PaperArtifactError(f"unsupported initialization value: {init}")
        if init not in ("cam", "pending") and method not in ("annealed-adam", "hard-cmaes"):
            raise PaperArtifactError(f"unsupported non-CAM solver for initialization: {method}")
        known = [case for _, case in cases if case is not None]
        values = {"dataset": dataset["label"], "state": dataset["state"], "method": method,
                  "init": init, "scope": scope, "included_in_init_axis": not exploratory,
                  "expected_runs": len(cases), "observed_runs": len(known),
                  "failed_runs": sum(case["record"].status is not C.RunStatus.COMPLETE for case in known),
                  "collapse_count": None, "collapse_rate": None,
                  "collapse_criterion": rules or None, "reason": dataset["reason"]}
        sources = list(dataset.get("sources", []))
        failure_details = []
        for case in known:
            record = case["record"]
            if record.status is C.RunStatus.COMPLETE:
                continue
            detail = {"run_id": record.run_id, "seed": case["seed"],
                      "status": record.status.value, "failure_reason": record.failure_reason}
            filename = case["config"].get("initialization_seed_result_file")
            if filename is not None:
                relative = PurePosixPath(filename)
                require(not relative.is_absolute() and ".." not in relative.parts and "." not in relative.parts,
                        f"{record.run_id}: unsafe initializer seed-result path")
                source_path = dataset["artifact_root"] / record.run_id / relative
                seed_document, seed_ref = source(source_path, f"initialization_seed_result:{record.run_id}")
                detail["seed_result"] = seed_document
                sources.append(seed_ref)
            failure_details.append(detail)
        values["failure_details"] = failure_details
        if exploratory:
            values["reason"] = "exploratory CAM perturbation; not a plain-CAM parent or init-axis comparison"
        elif dataset["state"] == "available":
            cam_parent = next((case for _, case in sorted(records.items()) if (
                case["method"] == method and case["init"] == "cam"
                and case["scope"] != "exploratory_cam_perturbation"
                and case["seed"] == 0)), None)
            collapse, descriptions = [], []
            for case in known:
                run_id = case["record"].run_id
                if case["init"] == "cam":
                    continue
                parent_run = case["config"].get("initialization", {}).get("parent_run_id")
                parent = records.get(parent_run) if parent_run is not None else cam_parent
                if parent is not None and (parent["method"] != method or parent["init"] != "cam"
                                           or parent["scope"] == "exploratory_cam_perturbation"):
                    parent = None  # no substitution for a failed declared-parent lookup
                eval_pair = dataset["verdicts"].get(run_id)
                parent_pair = dataset["verdicts"].get(parent_run or (
                    cam_parent["record"].run_id if cam_parent else ""))
                details = []
                case_flags = []
                for dialect in C.Dialect:
                    evaluation = eval_pair.get(dialect.value) if isinstance(eval_pair, dict) else None
                    parent_eval = parent_pair.get(dialect.value) if isinstance(parent_pair, dict) else None
                    if not isinstance(evaluation, C.EvaluationResult):
                        verdict, why = None, "fine verdict unavailable"
                    else:
                        verdict, why = _collapse_case(case, parent, evaluation, parent_eval, rules)
                    if verdict is None:
                        descriptions.append({"run_id": run_id, "dialect": dialect.value,
                                             "status": "pending", "reason": why})
                    else:
                        descriptions.append({"run_id": run_id, "dialect": dialect.value,
                                             "status": verdict, "criteria": why,
                                             "parent_run_id": parent["record"].run_id if parent else None})
                    case_flags.append(verdict)
                # A collapse count is in cases, never in dialect observations.
                # A mixed/missing D-04 verdict cannot be silently normalized away.
                consistent = len(case_flags) == len(C.Dialect) and len(set(case_flags)) == 1
                collapse.append(bool(consistent and all(case_flags)))
                if not consistent:
                    collapse = None
                    break
            values["collapse_criterion"] = {"rules": rules,
                                            "default_parent_rule": "same solver, init=cam, primary scope, seed 0; valid declared parent takes precedence",
                                            "per_case": descriptions}
            if collapse is not None:
                denominator = sum(case["init"] != "cam" for case in known)
                if denominator:
                    values["collapse_count"] = sum(collapse)
                    values["collapse_rate"] = sum(collapse) / denominator
                else:
                    values["reason"] = "plain-CAM reference; no non-CAM denominator"
            else:
                values["reason"] = "collapse unknown for at least one case; no filtered success rate"
        rows.append({"table": "init_axis", "columns": columns, "values": values,
                     "source_rows": [f"config:{run_id}" for run_id, _ in cases], "sources": sources})
    if not rows:
        return [{"table": "init_axis", "columns": columns,
                 "values": dict.fromkeys(columns) | {"dataset": dataset["label"],
                                                      "state": dataset["state"],
                                                      "reason": dataset["reason"]},
                 "source_rows": [], "sources": dataset.get("sources", [])}], columns
    return rows, columns


def _disagreement_rows(dataset):
    columns = ["dataset", "run_id", "method", "init", "dialect", "feasible_at_025",
               "geometry_pass_at_0125", "agree", "selection_config_hash", "seed_soft_objective",
               "seed_hard_objective", "soft_minus_hard", "no_improvement_over_init", "reason"]
    records = _init_dataset(dataset)
    rows = []
    if dataset["state"] != "available":
        values = dict.fromkeys(columns)
        values.update(dataset=dataset["label"], reason=dataset["reason"])
        return [{"table": "selection_disagreement", "columns": columns, "values": values,
                 "source_rows": [], "sources": dataset.get("sources", [])}], columns
    for run_id, case in sorted(records.items()):
        metric = case["metrics"]
        selection = mapping(metric.get("selection", {}), "selection")
        best = metric.get("best_candidate")
        selection = mapping(selection, "selection")
        for dialect in C.Dialect:
            evaluation = dataset["verdicts"].get(run_id, {}).get(dialect.value)
            if not isinstance(evaluation, C.EvaluationResult):
                continue
            feasible, reason = None, "missing recorded hard-selection components"
            if isinstance(best, Mapping):
                components = best.get("components", {})
                feasible = components.get("feasible_at_025")
                if feasible is None and components.get("feasible") in (0.0, 1.0):
                    feasible = bool(components["feasible"])
                if isinstance(feasible, bool):
                    reason = "recorded selected hard-evaluation flag; not a posted re-evaluation"
            fine_geometry = evaluation.geometry_pass
            initial = mapping(metric.get("selection", {}), "selection").get("initial_candidate", {})
            hard = initial.get("objective") if isinstance(initial, Mapping) else None
            soft, gap = None, None
            history = case["history"].get("history", [])
            if history:
                first = history[0]
                theta = first.get("theta")
                if (isinstance(theta, list) and isinstance(initial, Mapping)
                        and initial.get("theta_hash") is not None):
                    require(C.content_hash(theta) == initial.get("theta_hash"),
                            f"{run_id}: first-history θ does not match its initial candidate")
                    soft = first.get("soft_objective", first.get("objective"))
                    if isinstance(soft, (int, float)) and isinstance(hard, (int, float)):
                        gap = soft - hard
            values = {"dataset": dataset["label"], "run_id": run_id, "method": case["method"],
                      "init": case["init"], "dialect": dialect.value, "feasible_at_025": feasible,
                      "geometry_pass_at_0125": fine_geometry,
                      "agree": feasible == fine_geometry if isinstance(feasible, bool)
                      and isinstance(fine_geometry, bool) else None,
                      "selection_config_hash": case["config"].get("hard_selection_config_hash"),
                      "seed_soft_objective": soft, "seed_hard_objective": hard,
                      "soft_minus_hard": gap,
                      "no_improvement_over_init": metric.get("selection", {}).get(
                          "no_improvement_over_initialization"), "reason": reason}
            rows.append({"table": "selection_disagreement", "columns": columns, "values": values,
                         "source_rows": [f"metrics:{run_id}", f"verdict:{run_id}:{dialect.value}"],
                         "sources": dataset["sources"]})
    return rows, columns


def _resource_rows(dataset):
    columns = ["dataset", "state", "run_id", "role", "candidate_evaluations", "forward_calls",
               "backward_calls", "validation_calls", "entry_count", "wall_time_s",
               "process_peak_rss_bytes", "phase_wall_time_s", "memory_scope", "note"]
    rows = []
    for run_id, case in sorted(_init_dataset(dataset).items()):
        history = case["history"].get("ledger", {})
        entries = history.get("entries", [])
        walls = [item.get("wall_time_s") for item in entries]
        wall = math.fsum(walls) if all(isinstance(value, (int, float)) and math.isfinite(value) for value in walls) else None
        values = {"dataset": dataset["label"], "state": dataset["state"],
                  "run_id": run_id, "role": "optimizer_total",
                  "candidate_evaluations": history.get("candidate_evaluations"),
                  "forward_calls": history.get("forward_calls"),
                  "backward_calls": history.get("backward_calls"),
                  "validation_calls": history.get("validation_calls"),
                  "entry_count": len(entries), "wall_time_s": wall,
                  "process_peak_rss_bytes": history.get("process_peak_rss_bytes"),
                  "phase_wall_time_s": None, "memory_scope": history.get("memory_scope"),
                  "note": "whole optimizer process peak; not per-candidate or device-only memory"}
        rows.append({"table": "resource_costs", "columns": columns, "values": values,
                     "source_rows": [f"ledger:{run_id}"], "sources": dataset["sources"]})
        phases = case["metrics"].get("execution", {}).get("posted_evaluation_phases", [])
        values = dict.fromkeys(columns)
        values.update(dataset=dataset["label"], state=dataset["state"], run_id=run_id)
        peaks = []
        for phase in phases:
            for process in phase.get("processes", []):
                peaks.append((process.get("role"), process))
        for role, process in peaks:
            cells = dict(values)
            peak_kib = process.get("worker_peak_rss_kib")
            cells.update(role=role, entry_count=None, phase_wall_time_s=process.get("process_wall_seconds"),
                         process_peak_rss_bytes=(peak_kib * 1024 if isinstance(peak_kib, (int, float)) else None),
                         memory_scope=process.get("memory_note"),
                         note="individual posted worker; phase overlap is not folded into candidate wall time")
            rows.append({"table": "resource_costs", "columns": columns, "values": cells,
                         "source_rows": [f"metrics:{run_id}"], "sources": dataset["sources"]})
    return rows, columns


def _gradient_rows(root: Path | None):
    columns = ["case_id", "path_id", "objective_mode", "k", "precision", "status",
               "ad_fd_cosine", "ad_fd_relative_difference_norm", "epsilon_normalized_theta",
               "fd_indices", "ad_primary_cosine", "ad_reference_cosine",
               "ad_primary_relative_difference_norm", "ad_reference_relative_difference_norm",
               "primary_pairs", "reference_pairs", "primary_standard_error_rms",
               "reference_standard_error_rms", "hard_reused_across_k", "wall_time_s",
               "actual_candidate_evaluations", "own_process_vmhwm_bytes", "reason"]
    if root is None or not root.is_dir():
        return [{"table": "gradient_alignment", "columns": columns,
                 "values": dict.fromkeys(columns, None) | {"status": "absent"},
                 "source_rows": [], "sources": []}], columns
    studies = sorted((path for path in root.glob("full-reuse-*/matrix-plan.json")),
                     key=lambda path: path.parent.name)
    if not studies:
        return [{"table": "gradient_alignment", "columns": columns,
                 "values": dict.fromkeys(columns, None) | {"status": "pending"},
                 "source_rows": [], "sources": []}], columns
    plan_file = studies[-1]
    plan, reference = source(plan_file, "gradient_matrix")
    rows = []
    for case in plan.get("cases", []):
        path_id, k = case["path_id"], case["k"]
        filename = plan_file.parent / f"case-{case['case_index']:03d}-{path_id}-k{k:g}-geometry.json"
        sources = [reference]
        if not filename.is_file():
            values = dict.fromkeys(columns) | {"case_id": case["case_index"], "path_id": path_id,
                                               "objective_mode": case.get("objective_mode", "geometry"),
                                               "k": k, "status": "pending",
                                               "reason": "case is declared but its result is not present"}
            rows.append({"table": "gradient_alignment", "columns": columns, "values": values,
                         "source_rows": [], "sources": sources})
            continue
        document, ref = source(filename, f"gradient_case:{case['case_index']}")
        found_case = document.get("case", {})
        require(found_case.get("case_index") == case["case_index"]
                and found_case.get("path_id") == path_id and found_case.get("k") == k,
                f"gradient case identity differs from the declared matrix: {filename}")
        sources.append(ref)
        alignment = document["ad_vs_primary_hard"]
        reference_alignment = document["ad_vs_reference_hard"]
        primary = document["randomized_hard"]["primary"]
        high = document["randomized_hard"]["reference"]
        fd_rows = document["smooth_ad"].get("fd_alignment_and_coordinate_errors", [])
        first_fd = fd_rows[0].get("alignment_on_selected_coordinates", {}) if fd_rows else {}
        epsilon = fd_rows[0].get("epsilon_normalized_theta") if fd_rows else None
        values = {"case_id": case["case_index"], "path_id": path_id,
                  "objective_mode": case.get("objective_mode", document.get("objective_mode")),
                  "k": k, "precision": document["setup"].get("precision"),
                  "status": document.get("status"),
                  "ad_fd_cosine": first_fd.get("cosine"),
                  "ad_fd_relative_difference_norm": first_fd.get("relative_difference_norm"),
                  "epsilon_normalized_theta": epsilon,
                  "fd_indices": (fd_rows[0].get("evaluated_indices") if fd_rows else None),
                  "ad_primary_cosine": alignment.get("cosine"),
                  "ad_reference_cosine": reference_alignment.get("cosine"),
                  "ad_primary_relative_difference_norm": alignment.get("relative_difference_norm"),
                  "ad_reference_relative_difference_norm": reference_alignment.get("relative_difference_norm"),
                  "primary_pairs": primary.get("pairs"), "reference_pairs": high.get("pairs"),
                  "primary_standard_error_rms": primary.get("standard_error_rms"),
                  "reference_standard_error_rms": high.get("standard_error_rms"),
                  "hard_reused_across_k": document["randomized_hard"].get("reused_across_k"),
                  "wall_time_s": document.get("wall_time_s"),
                  "actual_candidate_evaluations": document.get("actual_candidate_evaluations"),
                  "own_process_vmhwm_bytes": document["memory"].get("own_process_vmhwm_bytes"),
                  "reason": document.get("diagnostic_status")}
        rows.append({"table": "gradient_alignment", "columns": columns, "values": values,
                     "source_rows": [f"matrix-plan#/cases/{case['case_index']}"], "sources": sources})
    return rows, columns


def _markdown_table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(display(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _results_summary(datasets, main, init, disagreements, gradients, resources):
    lines = ["# Results package", "",
             "Generated only from locally available typed records and the existing task-cluster audit machinery. "
             "This is not a finished 39-task campaign, a physical machining result, or a claim that a program is safe to run.", "",
             "## Input availability", ""]
    availability = []
    for dataset in datasets:
        availability.append({"dataset": dataset["label"], "state": dataset["state"],
                             "reason": dataset["reason"],
                             "expected_run_records": dataset.get("expected_run_count"),
                             "observed_run_records": dataset.get("observed_run_count")})
    lines.append(_markdown_table(["dataset", "state", "expected_run_records",
                                   "observed_run_records", "reason"], availability))
    lines.extend(["", "## Main method-versus-CAM outcomes", "",
                  "Overall rows only below; `family_breakdown.csv` retains every family separately. "
                  "Pilot results are bounded one-task/dev evidence. Campaign main rows are restricted to the test split. "
                  "A delta is method minus CAM, not an automatically beneficial direction. NA is unavailable, not zero.", ""])
    columns = ["dataset", "method", "dialect", "expected_tasks", "geometry_pass_rate",
               "safety_pass_rate", "completion_pass_rate", "delta_cycle_s", "cycle_ci_lower",
               "cycle_ci_upper", "cycle_holm_p", "cycle_paired_n", "delta_air_fraction",
               "air_holm_p", "air_paired_n", "no_improvement_rate"]
    overall = [item["values"] for item in main
               if item["values"].get("family") in (None, "overall")]
    lines.append(_markdown_table(columns, overall))
    lines.extend(["", "### Unfavourable and limiting findings", ""])
    unfavourable = []
    for row in overall:
        for flag in ("geometry_pass_rate", "safety_pass_rate", "completion_pass_rate"):
            value = row.get(flag)
            if isinstance(value, (int, float)) and value < 1:
                unfavourable.append(f"- **{row['dataset']} / {row['method']} / {row['dialect']}**: "
                                    f"{flag} = {display(value)} over {display(row['expected_tasks'])} declared task units.")
        for name in ("no_improvement_rate",):
            if row.get(name) == 1:
                unfavourable.append(f"- **{row['dataset']} / {row['method']}**: every recorded non-nominal optimizer arm has no hard improvement over its initialization.")
        for name in ("delta_cycle_s", "delta_air_fraction"):
            if isinstance(row.get(name), (int, float)) and row[name] > 0:
                unfavourable.append(f"- **{row['dataset']} / {row['method']} / {row['dialect']}**: {name} is positive ({display(row[name])}); no efficiency win is inferred.")
    for item in disagreements:
        if item["values"].get("agree") is False:
            values = item["values"]
            unfavourable.append(
                f"- **{values['run_id']} / {values['dialect']}**: recorded 0.25 mm "
                "selection flag and 0.125 mm geometry verdict disagree.")
    if unfavourable:
        lines.extend(unfavourable)
    else:
        lines.append("- No numerical improvement claim or favourable inference is made for unavailable inputs.")
    lines.extend(["", "## Initialization axis", "", init and _markdown_table(
        ["dataset", "method", "init", "scope", "expected_runs", "observed_runs", "failed_runs",
         "collapse_count", "collapse_rate", "failure_details", "reason"], [item["values"] for item in init]) or
        "Pending: no initialized arms with a typed record.", "",
        "CAM perturbation is exploratory and excluded from the init-axis comparison. A CAM semantic anchor is never replaced by a perturbation "
        "or inferred parent. Collapse is pending when the declared hard-loss, parent or 0.125 mm verdict evidence is missing.", "",
        "## Selection vs 0.125 mm verdict and seed soft/hard gap", ""])
    lines.append(_markdown_table(["dataset", "run_id", "init", "dialect", "feasible_at_025",
                                   "geometry_pass_at_0125", "agree", "seed_soft_objective",
                                   "seed_hard_objective", "soft_minus_hard", "no_improvement_over_init"],
                                  [item["values"] for item in disagreements]))
    lines.extend(["", "The 0.25 mm hard-selection flag and the independently posted 0.125 mm geometry verdict are separate roles; "
                  "missing selection evidence is excluded rather than counted as agreement. The signed soft−hard gap at the matched seed θ "
                  "is a model discrepancy, not a safety verdict.", ""])
    lines.extend(["## Resource costs", "", _markdown_table(
        ["dataset", "run_id", "role", "candidate_evaluations", "forward_calls", "backward_calls",
         "validation_calls", "wall_time_s", "phase_wall_time_s", "process_peak_rss_bytes", "memory_scope"],
        [item["values"] for item in resources]), "",
        "Optimizer process peaks cover the whole optimizer process, not one candidate. Posted-worker rows retain their process-local role. "
        "Elapsed process durations, candidate ledger sums and phase walls are different quantities; concurrency and other system work are not "
        "converted into a universal speedup or aggregate-memory certificate.", ""])
    lines.extend(["", "## T3.3 AD/FD/randomized-hard alignment", "", _markdown_table(
        ["case_id", "path_id", "objective_mode", "k", "status", "ad_fd_cosine",
         "ad_fd_relative_difference_norm", "ad_primary_cosine", "ad_reference_cosine",
         "ad_primary_relative_difference_norm", "ad_reference_relative_difference_norm",
         "primary_standard_error_rms", "reference_standard_error_rms", "actual_candidate_evaluations"],
        [item["values"] for item in gradients]), "",
        "These cases measure alignment, not gradient equality or unbiasedness. Randomized hard smoothing is a different surrogate; "
        "k-independent hard probes are reused here (not fresh samples at every k). Standard errors are reported, not hidden. "
        "FD alignment is on the declared coordinate subset and epsilon values. Declared-but-absent cases remain pending. "
        "Positive alignment/cosine does not establish machining correctness.", ""])
    for row in gradients:
        values = row["values"]
        if values.get("status") == "complete":
            primary = values.get("ad_primary_cosine")
            if isinstance(primary, (int, float)) and primary < 0.999:
                lines.append(f"- **{values['path_id']} / k={display(values['k'])} / {values['objective_mode']}**: "
                             f"AD/primary-randomized-hard cosine={display(primary)}; "
                             f"relative difference={display(values['ad_primary_relative_difference_norm'])}. "
                             f"AD/reference cosine={display(values['ad_reference_cosine'])}. "
                             "The differing surrogates remain materially distinct.")
    lines.extend(["", "## Limitations and scope", "",
                  "- **CEM is unavailable**: hard CMA-ES is only a zero-order representative and must not be relabelled CEM.",
                  "- **Titan evidence is partial**; it is not re-executed or validated by this package.",
                  "- **CNC-Net is native-route only here**: no executable-NC performance evidence is inferred. CAMotics remains supplementary, not a silent correctness oracle.",
                  "- All optimizer/pilot/gradient values are simulation/CPU studies, not physical CNC runs, tool-monitoring calibrations, collision-free certificates or trial-ready directions.",
                  "- Efficiency weights are provisional where the source records declare them provisional (including 0.01/0.01 in D-11 pilot work); study-specific weights are likewise not globally frozen protocol values.",
                  "- The smooth time/air channels use the documented native surrogate; the n/reference factor restores total-time scale but does not assert exact native-to-hard timing equality.",
                  "- Pilot 1 is retained pre-D-09/D-10 evidence with its own geometry-only setup and failure modes. Later pilot inputs are not silently substituted for it, and pilot/campaign defaults must not be conflated.",
                  "- Selection at 0.25 mm and fine 0.125 mm posted verdicts remain separate. Unsupported/failed/missing outcomes are never silently admitted as success or automatic geometry passes.",
                  "- Verdict-only campaign inputs have no additional screen comparison. Empty diagnostic sets do not establish zero disagreement or manufacture a screen-agreement statistic.",
                  "- Test tasks are independent units; solver seeds are averaged within a task. Sparse paired counts and null intervals remain explicit. A pass on an isolated dev task does not establish suite-wide performance.",
                  "- Unmeasured resources and missing comparisons stay NA. Process-local memory does not prove global peak-memory sufficiency at higher concurrency.",
                  "- Pending pilot/campaign cases are not completed experiments. No claim of final protocol freezing, published-study completeness or manuscript acceptance is made.", ""])
    return "\n".join(lines).encode("utf-8")


def build_results_package(*, pilot_1_root: Path, pilot_2_root: Path | None,
                          pilot_init_root: Path | None, campaign_root: Path | None,
                          gradient_root: Path | None, decisions_path: Path,
                          decision_ids: Sequence[str], precision: str = "f64"):
    require(precision in ("f32", "f64"), "results precision must be explicit f32 or f64")
    datasets = [_dataset("pilot_1", Path(pilot_1_root), "pilot"),
                _dataset("pilot_2", pilot_2_root, "pilot"),
                _dataset("pilot_init", pilot_init_root, "pilot"),
                _dataset("campaign", campaign_root, "campaign")]
    files, rows = {}, {"main_results": [], "family_breakdown": [], "init_axis": [],
                       "selection_disagreement": [], "resource_costs": [], "input_status": []}
    source_inventory = []
    for dataset in datasets:
        files[f"input_status/{dataset['label']}.json"] = json_bytes({
            "state": dataset["state"], "reason": dataset["reason"],
            "expected_run_count": dataset.get("expected_run_count"),
            "observed_run_count": dataset.get("observed_run_count")})
        rows["input_status"].append({"table": "input_status", "columns": [
            "dataset", "state", "reason", "expected_run_count", "observed_run_count"],
            "values": {"dataset": dataset["label"], "state": dataset["state"], "reason": dataset["reason"],
                       "expected_run_count": dataset.get("expected_run_count"),
                       "observed_run_count": dataset.get("observed_run_count")},
            "source_rows": [], "sources": dataset.get("sources", [])})
        built = _built_dataset(dataset, decision_ids, decisions_path, precision)
        main, columns, _sources = _main_rows(dataset, built)
        rows["main_results"].extend(main)
        rows["family_breakdown"].extend([dict(item, table="family_breakdown")
                                         for item in main if item["values"].get("family") != "overall"])
        init, _columns = _init_rows(dataset)
        rows["init_axis"].extend(init)
        disagreement, _columns = _disagreement_rows(dataset)
        rows["selection_disagreement"].extend(disagreement)
        resource, _columns = _resource_rows(dataset)
        rows["resource_costs"].extend(resource)
        for name, payload in built["files"].items():
            files[f"raw/{dataset['label']}/{name}"] = payload
        source_inventory.extend(built["sources"])
        if dataset["state"] != "available":
            source_inventory.extend(dataset.get("sources", []))
    gradients, gradient_columns = _gradient_rows(gradient_root)
    rows["gradient_alignment"] = gradients
    for item in gradients:
        source_inventory.extend(item.pop("sources", []))
    for name, group in rows.items():
        if not group and name == "resource_costs":
            files[f"{name}.csv"] = csv_bytes(_resource_rows(datasets[0])[1], [])
        if not group:
            continue
        columns = group[0]["columns"]
        require(all(item["columns"] == columns for item in group),
                f"inconsistent results table columns: {name}")
        files[f"{name}.csv"] = csv_bytes(columns, [item["values"] for item in group])
        for index, item in enumerate(group):
            item["claim_id"] = f"{name}.csv#row-{index}"
            item["source_file_identity"] = {_meta(ref)["role"]: _meta(ref)["sha256"]
                                            for ref in item.get("sources", [])}
            item.pop("sources", None)
    files["summary.md"] = _results_summary(
        datasets, rows["main_results"], rows["init_axis"], rows["selection_disagreement"],
        gradients, rows["resource_costs"])
    files["claim_evidence_results.json"] = json_bytes({
        "schema": "diffcam-results-claim-evidence/1",
        "rows": [{key: value for key, value in item.items() if key != "sources"} for group in rows.values() for item in group]})
    unique_sources = {(item["role"], item["sha256"]): item for item in source_inventory}
    manifest = {
        "schema": "diffcam-results-package/1", "scope": "audited pilot/gradient data with explicit missing/pending campaign inputs",
        "derived_audit_identity": "content hash with only runs_root/run_dir made task-local; immutable input bytes and statistics remain unchanged",
        "input_states": [{"label": dataset["label"], "state": dataset["state"],
                          "reason": dataset["reason"]} for dataset in datasets],
        "campaign_main_split": "test", "pilot_main_scope": "retained bounded pilot data, not held-out generalization",
        "precision": precision, "decision_binding": "explicit dataset-scoped protocol declaration unless embedded per-run decision IDs exist",
        "decision_ids": list(decision_ids),
        "statistics": {"independent_unit": "task", "bootstrap_seed": 20261108,
                       "bootstrap_replicates": 2000, "confidence": 0.95,
                       "pairing": "method minus nominal-cam on common efficiency-eligible tasks", "holm": "existing audit adjustments"},
        "limitations": ["CEM unavailable", "Titan evidence partial", "CNC-Net native-route only",
                        "simulation rather than physical execution", "provisional efficiency weights unless frozen",
                        "pending/missing inputs are explicit, not fabricated"],
        "input_file_identities": sorted(unique_sources.values(), key=lambda item: (item["role"], item["sha256"])),
        "builder_source_sha256": digest(Path(__file__).read_bytes()),
        "files": [{"path": path, "sha256": digest(payload), "bytes": len(payload)}
                  for path, payload in sorted(files.items())],
        "physical_execution_status": "not_executed_not_machining_qualified",
    }
    manifest["file_count_excluding_manifest"] = len(files)
    files["results_manifest.json"] = json_bytes(manifest)
    return files, manifest


def parse_bindings(items: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        run_id, separator, filename = item.partition("=")
        require(bool(separator) and bool(run_id) and bool(filename) and run_id not in result,
                f"invalid or repeated {label} binding {item!r}; use RUN_ID=PATH exactly once per run")
        result[run_id] = Path(filename)
    return result


def publish_outputs(files: Mapping[str, bytes], destination: Path) -> Path:
    """Publish only known new files into a fresh/empty non-manuscript directory."""
    destination = no_manuscript(destination, "artifact destination")
    if destination.exists():
        require(destination.is_dir(), f"output is not a directory: {destination}")
        require(not any(destination.iterdir()),
                f"output must be a fresh/empty directory (nothing is overwritten): {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    for relative, payload in sorted(files.items()):
        relative_path = PurePosixPath(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts,
                f"unsafe generated relative path {relative}")
        target = destination / Path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
        require(target.read_bytes() == payload, f"output byte verification failed: {target}")
    return destination


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, help="complete pilot_report.json from benchmarks.audit")
    parser.add_argument("--manifest", type=Path, help="matching task/run manifest")
    parser.add_argument("--decisions", type=Path, default=ROOT / "reports/decisions.md",
                        help="immutable reports/decisions.md")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/results",
                        help="fresh/empty destination, e.g. reports/paper_artifacts/pilot-smoke; never manuscript/")
    parser.add_argument("--campaign-precision", default="f64", choices=("f32", "f64"),
                        help="required precision of every bound run's simulator_precision metadata")
    parser.add_argument("--decision", action="append", default=[], dest="decision_ids", metavar="D-ID",
                        help="dataset decision binding; required for legacy mode (results mode defaults to D-05 only)")
    parser.add_argument("--results", action="store_true",
                        help="build the real results package, preserving pending campaign/pilot input states")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs",
                        help="root containing the named pilot/retained-study directories in results mode")
    parser.add_argument("--pilot-1-root", type=Path)
    parser.add_argument("--pilot-2-root", type=Path)
    parser.add_argument("--pilot-init-root", type=Path)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--gradient-study-root", type=Path)
    parser.add_argument("--input-kind", choices=("synthetic", "smoke", "pilot"), default="synthetic",
                        help="explicit scope label; final frozen campaign inputs are outside this scaffold")
    bindings = parser.add_mutually_exclusive_group(required=False)
    bindings.add_argument("--run-root", type=Path,
                          help="root containing each expected RUN_ID/config.json")
    bindings.add_argument("--run-config", action="append", default=[], metavar="RUN_ID=PATH",
                          help="one immutable config per expected run; repeat for every run")
    parser.add_argument("--ledger", action="append", default=[], metavar="RUN_ID=PATH",
                        help="ComputeLedger snapshot or history.json with .ledger; missing ledgers are explicit NA rows")
    parser.add_argument("--baseline-coverage", type=Path,
                        help="optional diffcam-paper-coverage/1 declarations (external baselines/disagreements)")
    parser.add_argument("--refinement", action="append", default=[], type=Path,
                        help="optional schema-3 refinement-link file plus its linked immutable evidence files; repeat")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.results:
        runs_root = args.runs_root
        try:
            files, metadata = build_results_package(
                pilot_1_root=args.pilot_1_root or runs_root / "pilot-pocket-20261109-t41b-pilot",
                pilot_2_root=args.pilot_2_root or runs_root / "pilot-pocket-20261110-pilot2",
                pilot_init_root=args.pilot_init_root or runs_root / "pilot-pocket-20261110-pilot2_init",
                campaign_root=args.campaign_root,
                gradient_root=args.gradient_study_root or runs_root / "retained/t33-real-study-20261110-w3-r1",
                decisions_path=args.decisions,
                decision_ids=args.decision_ids or ["D-2026-11-09-05"],
                precision=args.campaign_precision)
            output = publish_outputs(files, args.output)
        except (PaperArtifactError, A.AuditInputError, C.ContractError, OSError, KeyError, TypeError,
                json.JSONDecodeError) as error:
            parser.error(str(error))
        print(json.dumps({"status": "results_package_complete", "output": str(output),
                          "input_states": metadata["input_states"], "file_count": len(files)}, sort_keys=True))
        return 0
    legacy_required = {"--audit": args.audit, "--manifest": args.manifest,
                       "--decision": args.decision_ids,
                       "--run-root or --run-config": args.run_root or args.run_config}
    for flag, value in legacy_required.items():
        if not value:
            parser.error(f"{flag} is required for the legacy scaffold interface")
    try:
        configs = ({} if args.run_root is not None
                   else parse_bindings(args.run_config, "--run-config"))
        ledgers = parse_bindings(args.ledger, "--ledger")
        files, _ = build_artifacts(
            audit_path=args.audit, manifest_path=args.manifest, decisions_path=args.decisions,
            run_configs=configs, precision=args.campaign_precision,
            decision_ids=args.decision_ids, input_kind=args.input_kind,
            ledger_paths=ledgers, baseline_coverage_path=args.baseline_coverage,
            refinement_paths=args.refinement, run_root=args.run_root,
        )
        output = publish_outputs(files, args.output)
    except (PaperArtifactError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "scaffold_complete", "input_kind": args.input_kind,
                      "output": str(output), "file_count": len(files)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
