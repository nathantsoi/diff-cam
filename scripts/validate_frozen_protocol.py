#!/usr/bin/env python3
"""Validate a T4.2 protocol declaration and run its permitted dry-run checks.

Tooling only: this file does not choose frozen values or execute optimization.
It validates declarations and task/file provenance, then converts supported
prototype dry-run cases into the existing benchmark-runner configuration.

The current runner gates executable checks to prototype_analytic, dev, pilot.
Other advertised splits/experiments are declared for later integration but are
rejected explicitly by this adapter rather than silently run as a pilot.
f64/D-04/D-07 are protocol constraints, not proofs of numerical execution,
machine safety, measured resource use, or satisfactory benchmark outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from benchmarks import contracts as C
from scripts import run_benchmark_suite as runner

PROTOCOL_SCHEMA = "diffcam-frozen-benchmark/1"
HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
DECISION_PATTERN = re.compile(r"D-\d{4}-\d{2}-\d{2}-\d{2}")
REQUIRED_DECISIONS = {
    "D-2026-11-09-01", "D-2026-11-09-02", "D-2026-11-09-03",
    "D-2026-11-09-04", "D-2026-11-09-05", "D-2026-11-09-06", "D-2026-11-09-07",
    "D-2026-11-10-08",
}
INIT_VALUES = ("cam", "straight_line", "random")
NON_CAM_SOLVERS = frozenset(("annealed-adam", "hard-cmaes"))
THRESHOLD_FIELDS = (
    "completion_residual_mm3", "gouge_volume_mm3", "local_surface_error_mm",
    "clearance_mm", "rapid_penetration_mm",
)
SINGLE_SEED_METHODS = ("nominal-cam", "fixed-adam", "annealed-adam")
FORBIDDEN_GATE_KEYS = {
    "strict_improvement", "strict_improvement_required",
    "require_strict_improvement", "must_improve",
}


class ProtocolError(ValueError):
    """An actionable declaration, evidence-link, or runner-compatibility error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def finite_json(value: Any, where: str = "protocol") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{where}: non-finite number is not allowed")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            finite_json(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_json(child, f"{where}[{index}]")


def load_json(path: Path) -> tuple[Any, str, bytes]:
    actual = no_manuscript(path, "declared input")
    try:
        payload = actual.read_bytes()
        value = json.loads(payload.decode("utf-8"), parse_constant=(
            lambda text: require(False, f"{actual}: non-finite JSON constant {text}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read JSON input {actual}: {error}") from error
    finite_json(value, str(actual))
    return value, "sha256:" + hashlib.sha256(payload).hexdigest(), payload


def file_sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return "sha256:" + hasher.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def split_hash(entries: Sequence[Mapping[str, str]]) -> str:
    """Hash only (task_id, task_hash) pairs, sorted by task_id, in canonical JSON."""
    return "sha256:" + hashlib.sha256(canonical_json(entries)).hexdigest()


def no_manuscript(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    require("manuscript" not in resolved.parts,
            f"{role}: manuscript/ is not an allowed input or output location: {resolved}")
    return resolved


def relative_file(root: Path, value: Any, where: str) -> Path:
    require(isinstance(value, str) and value and not value.isspace(),
            f"{where}: expected a nonempty repository-relative path")
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and ".." not in relative.parts and
            "." not in relative.parts, f"{where}: absolute/traversal path is forbidden: {value}")
    return no_manuscript(root.joinpath(*relative.parts), where)


def object_fields(value: Any, where: str, required: Sequence[str],
                  optional: Sequence[str] = ()) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{where}: expected an object")
    unknown = sorted(set(value) - set(required) - set(optional))
    require(not unknown, f"{where}: unknown fields {unknown}; use a versioned protocol amendment")
    for key in required:
        require(key in value, f"{where}: missing required field {key}")
    return value


def reject_placeholders(value: Any, where: str = "protocol") -> None:
    """A coordinator's blank template can never become a frozen configuration."""
    if isinstance(value, str):
        stripped = value.strip()
        require(not re.search(r"\b(?:TO[-_ ]BE[-_ ]FROZEN|TODO|TBD)\b", value, flags=re.IGNORECASE)
                and not (stripped.startswith("<") and stripped.endswith(">")),
                f"{where}: frozen configuration contains an unresolved placeholder")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            reject_placeholders(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_placeholders(child, f"{where}[{index}]")


def text(value: Any, where: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{where}: expected nonempty text")
    reject_placeholders(value, where)
    return value


def integer(value: Any, where: str, minimum: int = 0) -> int:
    require(not isinstance(value, bool) and isinstance(value, int) and value >= minimum,
            f"{where}: expected an integer >= {minimum}, got {value!r}")
    return value


def number(value: Any, where: str, minimum: float = 0.0, *, strict: bool = False) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value), f"{where}: expected a finite number")
    require(value > minimum if strict else value >= minimum,
            f"{where}: expected a number {'>' if strict else '>='} {minimum}")
    return float(value)


def boolean(value: Any, where: str, expected: bool | None = None) -> bool:
    require(isinstance(value, bool), f"{where}: expected true or false, not {value!r}")
    if expected is not None:
        require(value is expected, f"{where}: must be {str(expected).lower()}")
    return value


def sha(value: Any, where: str) -> str:
    require(isinstance(value, str) and HASH_PATTERN.fullmatch(value),
            f"{where}: expected sha256:<64 lowercase hex characters>")
    return value


def seed_list(value: Any, where: str) -> list[int]:
    require(isinstance(value, list) and value, f"{where}: expected a nonempty seed list")
    cleaned = [integer(item, f"{where} seed") for item in value]
    require(len(cleaned) == len(set(cleaned)), f"{where}: duplicate seeds are not allowed")
    return cleaned


def nonempty_names(value: Any, where: str) -> list[str]:
    require(isinstance(value, list) and value, f"{where}: expected a nonempty list")
    return [text(item, f"{where}[{index}]") for index, item in enumerate(value)]


def require_if_file(path: Path, where: str) -> str:
    require(path.is_file(), f"{where}: required evidence file does not exist: {path}")
    try:
        return file_sha(path)
    except OSError as error:
        raise ProtocolError(f"{where}: cannot hash evidence file {path}: {error}") from error


def verify_asset_hash(path: Path, expected: str, where: str) -> str:
    actual = require_if_file(path, where)
    require(actual == expected, f"{where}: declared hash {expected} != actual {actual}")
    return actual


def task_registry(manifest: Mapping[str, Any], repo: Path) -> tuple[
        dict[str, dict[str, Any]], dict[str, str | None]]:
    """Resolve TaskSpec identity rather than trusting a decorative task_hash."""
    require(isinstance(manifest, Mapping), "task_manifest must be a JSON object")
    raw_items = manifest.get("tasks")
    require(isinstance(raw_items, list) and raw_items, "task_manifest.tasks must be nonempty")
    registry: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, str]]] = {"dev": [], "test": [], "external": []}
    for index, raw_item in enumerate(raw_items):
        where = f"task_manifest.tasks[{index}]"
        item = object_fields(raw_item, where, [], [
            "task_id", "task_hash", "task_record", "task_split", "split", "family",
            "continuous_exclusion_diagnostic_mm3", "evaluator_exclusion_tolerance_mm3",
            "grid_excluded_volume_mm3", "seed_spec", "seed_provenance", "motion_hash",
        ])
        require(isinstance(item.get("task_id"), str) and item["task_id"],
                f"{where}.task_id must be nonempty")
        declared_hash = sha(item.get("task_hash"), where + ".task_hash")
        source = item.get("task_record", item)
        if isinstance(source, Mapping) and "record" in source:
            source = source["record"]
        require(isinstance(source, Mapping), f"{where}.task_record must contain a task object")
        try:
            task = C.TaskSpec.from_dict(source)
        except (TypeError, ValueError) as error:
            raise ProtocolError(f"{where}: invalid TaskSpec: {error}") from error
        task_id = task.task_id
        require(task_id == item["task_id"], f"{where}: embedded task_id and declared ID differ")
        require(task_id not in registry, f"{where}: duplicate task ID {task_id}")
        require(task.task_hash == declared_hash,
                f"{where}: embedded TaskSpec.hash != declared task_hash")
        for name in ("family", "seed_provenance", "motion_hash", "seed_spec"):
            if name not in item:
                continue
            if name == "family":
                require(item[name] == task.family, f"{where}.family differs from the embedded TaskSpec")
            elif name == "motion_hash":
                # Nominal MotionPlan provenance is separate from TaskSpec's identity.
                sha(item[name], where + ".motion_hash")
        for geometry in (task.stock, task.original_part, task.operation_target.geometry):
            if geometry is not None and geometry.path is not None and geometry.sha256 is not None:
                path_value = Path(geometry.path)
                if path_value.is_absolute():
                    resolved = no_manuscript(path_value, where + ".geometry")
                    require(resolved.is_relative_to(repo), f"{where}: geometry payload escapes repository")
                else:
                    resolved = relative_file(repo, geometry.path, where + ".geometry")
                verify_asset_hash(resolved, geometry.sha256, where + ".geometry")
        split = item.get("task_split", item.get("split", task.split.value))
        require(split == task.split.value and split in runner.SPLITS,
                f"{where}: split metadata and embedded TaskSpec differ")
        registry[task_id] = {"task": task, "split": split, "task_hash": declared_hash}
        groups[split].append({"task_id": task_id, "task_hash": declared_hash})
    digests = {
        name: split_hash(sorted(entries, key=lambda entry: entry["task_id"]))
        if entries else None
        for name, entries in groups.items()
    }
    return registry, digests


def validate_decisions(decision_ids: Sequence[str], decisions_path: Path) -> tuple[list[str], str]:
    require(isinstance(decision_ids, (list, tuple)) and decision_ids,
            "decision_ids must be a nonempty list")
    require(all(isinstance(item, str) and DECISION_PATTERN.fullmatch(item)
                for item in decision_ids),
            "decision_ids must contain explicit D-YYYY-MM-DD-NN identifiers")
    require(len(decision_ids) == len(set(decision_ids)), "decision_ids contain duplicates")
    actual_path = no_manuscript(decisions_path, "decision_record")
    try:
        document = actual_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ProtocolError(f"cannot read decision record {actual_path}: {error}") from error
    found = re.findall(r"^##\s+(D-\d{4}-\d{2}-\d{2}-\d{2})\s+—", document, re.MULTILINE)
    missing = sorted((REQUIRED_DECISIONS | set(decision_ids)) - set(found))
    require(not missing, f"decision record {actual_path}: missing decision IDs {missing}")
    return sorted(set(decision_ids)), file_sha(actual_path)


def matching_thresholds(protocol: Mapping[str, Any], task: C.TaskSpec) -> None:
    values = object_fields(protocol["thresholds"], "thresholds", THRESHOLD_FIELDS)
    actual = task.to_dict()["tolerances"]
    for name in THRESHOLD_FIELDS:
        declared = number(values[name], f"thresholds.{name}")
        require(name in actual, f"task tolerances does not define {name}")
        require(declared == actual[name],
                f"thresholds.{name}={declared} does not match task tolerance {actual[name]}")


def forbid_improvement_gate(value: Any, where: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "strict_improvement_required" and child is False:
                continue
            require(key not in FORBIDDEN_GATE_KEYS,
                f"{where}/{key}: strict-improvement gates are forbidden; accept reject/narrow findings")
            forbid_improvement_gate(child, f"{where}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            forbid_improvement_gate(child, f"{where}[{index}]")


def init_mode(protocol: Mapping[str, Any], entry: Mapping[str, Any],
              condition: str, where: str) -> str:
    """Resolve CAM's default independently from the optimizer condition."""
    value = entry.get("init", protocol.get("init", "cam"))
    require(isinstance(value, str) and value in INIT_VALUES,
            f"{where}.init must be one of {list(INIT_VALUES)}, got {value!r}")
    require(condition != "nominal-cam" or value == "cam",
            "nominal-cam requires init=cam; non-CAM init is an optimizer condition")
    require(value == "cam" or condition in NON_CAM_SOLVERS,
            f"{where}: init={value} is allowed only with annealed-adam or hard-cmaes, not {condition}")
    return value


def validate_initializations(protocol: Mapping[str, Any]) -> list[str]:
    """Validate the three mandatory, immutable D-08 no-init declarations.

    These declarations define a future MotionPlan recipe; this tool neither
    constructs a MotionPlan nor asserts geometric safety/efficiency eligibility.
    """
    explicit = [protocol.get("init", "cam")]
    for key in ("methods", "dimensionality_subset"):
        rows = protocol.get(key, [])
        if isinstance(rows, list):
            explicit.extend(row.get("init", protocol.get("init", "cam"))
                            for row in rows if isinstance(row, Mapping))
    if all(value == "cam" for value in explicit):
        return []
    for value in explicit:
        require(value in INIT_VALUES, f"init must be one of {list(INIT_VALUES)}, got {value!r}")
    return initializations(protocol)


def initializations(protocol: Mapping[str, Any]) -> list[str]:
    """Strong typing and content checks for all no-init declaration objects."""
    for key in ("random_init", "straight_line_init", "collapse_criterion"):
        require(isinstance(protocol.get(key), Mapping) and bool(protocol[key]),
                f"{key}: a nonempty predeclared object is required when any init != cam")
    random = object_fields(protocol["random_init"], "random_init",
                           ("parameter_space", "distribution", "scale", "seed_derivation_rule"))
    require(random["parameter_space"] in ("theta", "delta"),
            "random_init.parameter_space must be theta or delta")
    distribution = object_fields(random["distribution"], "random_init.distribution", ("kind", "detail"))
    require(distribution["kind"] in ("gaussian", "uniform", "other_declared"),
            "random_init.distribution.kind must be gaussian, uniform, or other_declared")
    text(distribution["detail"], "random_init.distribution.detail")
    scale = object_fields(random["scale"], "random_init.scale", ("fraction", "reference"))
    number(scale["fraction"], "random_init.scale.fraction", 0, strict=True)
    reference = object_fields(scale["reference"], "random_init.scale.reference",
                              ("kind", "description", "units"))
    require(reference["kind"] in ("stock_xy_extent", "cam_delta_norm", "other_declared"),
            "random_init.scale.reference.kind must name a scale reference explicitly")
    text(reference["description"], "random_init.scale.reference.description")
    text(reference["units"], "random_init.scale.reference.units")
    derivation = object_fields(random["seed_derivation_rule"], "random_init.seed_derivation_rule",
                               ("algorithm", "root_seed", "input", "per_seed_rule"))
    text(derivation["algorithm"], "random_init.seed_derivation_rule.algorithm")
    integer(derivation["root_seed"], "random_init.seed_derivation_rule.root_seed")
    text(derivation["input"], "random_init.seed_derivation_rule.input")
    text(derivation["per_seed_rule"], "random_init.seed_derivation_rule.per_seed_rule")
    straight = object_fields(protocol["straight_line_init"], "straight_line_init",
                             ("start", "end", "segment_placement", "feed_and_rapid_entry",
                              "same_segment_count_and_k", "safe_rapid_entry",
                              "respect_task_feed_limits", "efficiency_not_required"))
    for key in ("start", "end", "segment_placement", "feed_and_rapid_entry"):
        text(straight[key], "straight_line_init." + key)
    boolean(straight["same_segment_count_and_k"], "straight_line_init.same_segment_count_and_k", True)
    boolean(straight["safe_rapid_entry"], "straight_line_init.safe_rapid_entry", True)
    boolean(straight["respect_task_feed_limits"], "straight_line_init.respect_task_feed_limits", True)
    boolean(straight["efficiency_not_required"], "straight_line_init.efficiency_not_required", True)
    collapse = object_fields(protocol["collapse_criterion"], "collapse_criterion",
                             ("geometry_or_completion_failure", "residual_multiplier", "residual_operator",
                              "require_no_hard_loss_improvement_over_initial", "predicate",
                              "collapse_and_infeasibility_are_valid_recorded_outcomes"))
    boolean(collapse["geometry_or_completion_failure"],
            "collapse_criterion.geometry_or_completion_failure")
    number(collapse["residual_multiplier"], "collapse_criterion.residual_multiplier", 0, strict=True)
    require(collapse["residual_operator"] in ("gt", "ge"),
            "collapse_criterion.residual_operator must be gt or ge")
    boolean(collapse["require_no_hard_loss_improvement_over_initial"],
            "collapse_criterion.require_no_hard_loss_improvement_over_initial")
    text(collapse["predicate"], "collapse_criterion.predicate")
    boolean(collapse["collapse_and_infeasibility_are_valid_recorded_outcomes"],
            "collapse_criterion.collapse_and_infeasibility_are_valid_recorded_outcomes", True)
    return ["D-08 initialization is declared but has no runner/seed implementation yet; pending rows are not executable or measured outcomes."]


def validate_tuning(tuning: Mapping[str, Any], repo: Path,
                    init_selections: set[tuple[str, str]]) -> tuple[int, list[dict[str, Any]]]:
    object_fields(tuning, "tuning", ("total_candidate_evaluations", "rounds", "only_dev"))
    require(tuning["only_dev"] is True, "tuning must be restricted to dev")
    require(tuning["only_dev"] is True, "tuning must be restricted to dev")
    require(isinstance(tuning["rounds"], list), "tuning.rounds must be a list")
    total = 0
    rows = []
    forbidden_parts = {"test", "external", "heldout", "held-out"}
    for index, row_value in enumerate(tuning["rounds"]):
        where = f"tuning.rounds[{index}]"
        row = object_fields(row_value, where, ("split", "candidate_evaluations", "conditions", "changes", "evidence"), ("init",))
        mode = row.get("init", "cam")
        require(mode in INIT_VALUES, f"{where}.init must be cam, straight_line, or random")
        require(row["split"] == "dev", f"{where}.split must be dev, never test/external")
        spend = integer(row["candidate_evaluations"], where + ".candidate_evaluations", 1)
        total += spend
        conditions = nonempty_names(row["conditions"], where + ".conditions")
        require(set(conditions) <= set(runner.CONDITIONS) and len(set(conditions)) == len(conditions),
                f"{where}.conditions contains an unknown or duplicate condition")
        require(all((condition, mode) in init_selections for condition in conditions),
                f"{where}: conditions must match retained primary (condition, init) selections")
        require(isinstance(row["changes"], list), f"{where}.changes must be a list")
        for change in row["changes"]:
            text(change, where + ".changes")
        require(isinstance(row["evidence"], list) and row["evidence"],
                f"{where}: tuning rounds require linked evidence")
        verified = []
        for evidence_index, supplied in enumerate(row["evidence"]):
            ewhere = f"{where}.evidence[{evidence_index}]"
            entry = object_fields(supplied, ewhere, ("path", "sha256", "split"))
            require(entry["split"] == "dev", f"{ewhere}.split must be dev")
            raw_path = text(entry["path"], ewhere + ".path")
            segments = {part.casefold() for part in PurePosixPath(raw_path).parts}
            require(not segments.intersection(forbidden_parts),
                    f"{ewhere}.path references prohibited test/external tuning data: {raw_path}")
            expected_hash = sha(entry["sha256"], ewhere + ".sha256")
            path = relative_file(repo, raw_path, ewhere + ".path")
            verify_asset_hash(path, expected_hash, ewhere)
            verified.append({"path": raw_path, "sha256": expected_hash, "split": "dev"})
        rows.append({"split": "dev", "conditions": conditions, "init": mode, "candidate_evaluations": spend,
                     "changes": list(row["changes"]), "evidence": verified})
    require(integer(tuning["total_candidate_evaluations"], "tuning.total_candidate_evaluations")
            == total, "tuning.total_candidate_evaluations must equal the sum of declared rounds")
    return total, rows


def check_budget(protocol: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    budget = protocol["budget"]
    expected = protocol["estimates"]
    candidate_total = 0
    validation_total = 0
    total_seconds = 0.0
    largest_job_seconds = 0.0
    uncertain = []
    costed = []
    rates = expected["seconds_per_candidate_by_method"]
    require(isinstance(rates, Mapping), "estimates.seconds_per_candidate_by_method must be an object")
    require(set(rates).issubset({job["condition"] for job in jobs}),
            "time estimates may name only conditions present in this protocol")
    for condition, value in rates.items():
        number(value, f"estimates.seconds_per_candidate_by_method.{condition}", 0, strict=True)
    for job in jobs:
        condition = job["condition"]
        candidate_count = 1 if condition == "nominal-cam" else budget["candidate_evaluations"]
        validation_count = min(budget["max_validation_calls"], candidate_count // budget["validation_interval"])
        candidate_total += candidate_count
        validation_total += validation_count
        if condition not in rates:
            uncertainty = f"no seconds-per-candidate estimate for {condition}"
            uncertain.append(job)
            estimate = None
        else:
            seconds_per_candidate = rates[condition]
            estimate = seconds_per_candidate * (candidate_count + validation_count)
            total_seconds += estimate
            largest_job_seconds = max(largest_job_seconds, estimate)
        costed.append((job, candidate_count, validation_count, estimate))
    require(expected["run_count"] == len(jobs),
            f"estimates.run_count {expected['run_count']} != computed {len(jobs)}")
    require(expected["candidate_evaluations"] == candidate_total,
            f"estimates.candidate_evaluations {expected['candidate_evaluations']} != computed {candidate_total}")
    require(expected["validation_calls"] == validation_total,
            f"estimates.validation_calls {expected['validation_calls']} != computed {validation_total}")
    if uncertain:
        require(expected["worker_seconds"] is None,
                "worker_seconds must be null when any condition lacks a declared time estimate")
    else:
        require(expected["worker_seconds"] is not None,
                "provide worker_seconds or explicitly mark an unknown estimate")
        require(expected["worker_seconds"] >= 0 and
                abs(expected["worker_seconds"] - total_seconds) <= 1e-9 * max(1.0, total_seconds),
                f"estimates.worker_seconds {expected['worker_seconds']} != computed {total_seconds}")
        require(budget["total_wall_time_s"] >= total_seconds,
                "total_wall_time_s is smaller than the declared serial worker-second estimate")
    require(budget["timeout_s"] >= largest_job_seconds,
            "timeout_s is smaller than the largest declared per-job estimate")
    if expected["cpu_rss_bytes_per_job"] > protocol["memory"]["max_rss_bytes_per_job"]:
        raise ProtocolError("cpu_rss_bytes_per_job exceeds D-07 max_rss_bytes_per_job; do not proceed without an evaluator memory optimization")
    require_if_unknown_memory = text(expected["uncertainty"], "estimates.uncertainty")
    return {"jobs": len(jobs), "candidate_evaluations": candidate_total,
            "validation_calls": validation_total,
            "worker_seconds": total_seconds if not uncertain else None,
            "largest_job_seconds": largest_job_seconds,
            "timing_unknown_jobs": [{"condition": job["condition"], "init": job["init"], "seed": job["seed"],
                                     "scope": job["scope"]} for job in uncertain],
            "per_job": costed, "uncertainty_note": require_if_unknown_memory}


def validate_declaration(config: Mapping[str, Any], repo: Path,
                         decisions_path: Path) -> dict[str, Any]:
    """Validate only declaration/domain consistency; do not create output yet."""
    fields = object_fields(
        config, "protocol",
        ("schema", "decision_ids", "task_manifest", "primary_task_id", "dimensionality_subset",
         "methods", "objective", "thresholds", "posted_evaluation", "memory", "budget",
         "estimates", "tuning", "comparisons", "multiplicity", "external_baselines",
         "acceptance", "execution", "precision"),
        ("notes", "init", "random_init", "straight_line_init", "collapse_criterion"))
    repo = no_manuscript(repo, "repository_root").resolve()
    protocol = config
    reject_placeholders(config)
    require(config["schema"] == PROTOCOL_SCHEMA,
            f"protocol.schema must be {PROTOCOL_SCHEMA}")
    forbid_improvement_gate(config)
    if "notes" in config:
        require(isinstance(config["notes"], list) and
                all(isinstance(note, str) for note in config["notes"]),
                "protocol.notes must be a list of strings when supplied")
    decision_ids, decision_hash = validate_decisions(decision_ids=config["decision_ids"],
                                                     decisions_path=decisions_path)
    proto_task, _, _ = runner._prototype()
    registry_source = object_fields(protocol["task_manifest"], "task_manifest",
                                    ("path", "sha256", "split_hashes", "acquisition"))
    requested_split_hashes = object_fields(
        registry_source["split_hashes"], "task_manifest.split_hashes", runner.SPLITS)
    text(registry_source["acquisition"], "task_manifest.acquisition")
    manifest_path = relative_file(repo, registry_source["path"], "task_manifest.path")
    manifest_document, manifest_sha, _ = load_json(manifest_path)
    verify_asset_hash(manifest_path, sha(registry_source["sha256"], "task_manifest.sha256"),
                      "task_manifest.sha256")
    registry, split_digests = task_registry(manifest_document, repo)
    for split in runner.SPLITS:
        require((split_digests[split] is None) == (requested_split_hashes[split] is None),
                f"task_manifest.split_hashes.{split}: a populated split requires a hash, an empty one requires null")
        if split_digests[split] is not None:
            require(split_digests[split] == sha(requested_split_hashes[split],
                                                f"task_manifest.split_hashes.{split}"),
                    f"task_manifest.split_hashes.{split} does not match the typed task selection")
    primary_id = text(config["primary_task_id"], "primary_task_id")
    require(primary_id in registry, f"primary_task_id {primary_id!r} is not in task_manifest.tasks")
    primary = registry[primary_id]
    task = primary["task"]
    matching_thresholds(config, task)
    execution = object_fields(
        config["execution"], "execution",
        ("task_source", "experiment", "split", "architecture", "output_root",
         "prototype_task_id", "prototype_task_hash"))
    require(execution["task_source"] == "prototype_analytic",
            "current adapter supports only the real prototype_analytic factory; no unregistered plugin fallback")
    require(execution["experiment"] in runner.EXPERIMENTS,
            f"unsupported experiment {execution['experiment']!r}")
    require(execution["split"] in runner.SPLITS, f"unsupported split {execution['split']!r}")
    require(execution["architecture"] in ("cpu", "gpu"), "execution.architecture must be cpu or gpu")
    probe_match = (task.task_id == proto_task.task_id and task.task_hash == proto_task.task_hash
                   and execution["experiment"] == "pilot")
    require(execution["prototype_task_id"] == proto_task.task_id,
            f"prototype_task_id {execution['prototype_task_id']!r} does not match actual prototype {proto_task.task_id!r}")
    require(execution["prototype_task_hash"] == proto_task.task_hash,
            "prototype_task_hash differs from the actual constructible task")
    require(execution["split"] == task.split.value,
            "execution.split does not match the selected TaskSpec's split")
    relative_file(repo, execution["output_root"], "execution.output_root")

    objective = object_fields(
        config["objective"], "objective",
        ("primary_k", "anneal_start", "anneal_end", "gouge_weight", "residual_weight",
         "holder_weight", "holder_margin_mm", "loss_shift_voxels"))
    primary_k = number(objective["primary_k"], "objective.primary_k", 0, strict=True)
    anneal_start = number(objective["anneal_start"], "objective.anneal_start", 0, strict=True)
    anneal_end = number(objective["anneal_end"], "objective.anneal_end", 0, strict=True)
    require(anneal_end > anneal_start, "objective.anneal_end must exceed anneal_start")
    for name in ("gouge_weight", "residual_weight", "holder_weight", "holder_margin_mm"):
        number(objective[name], f"objective.{name}")
    require(not isinstance(objective["loss_shift_voxels"], bool)
            and isinstance(objective["loss_shift_voxels"], (int, float))
            and math.isfinite(objective["loss_shift_voxels"]),
            "objective.loss_shift_voxels must be a finite signed number")
    posted = config["posted_evaluation"]
    object_fields(posted, "posted_evaluation",
                  ("verdict_max_step_mm", "screen_max_step_mm", "convergence_max_step_rs274_mm",
                   "screen_is_diagnostic_only", "convergence_is_rs274_only",
                   "record_memory_by_phase_and_dialect"))
    require(number(posted["verdict_max_step_mm"], "posted_evaluation.verdict_max_step_mm") == 0.125,
            "D-04 requires verdicts at 0.125 mm for both dialects")
    require(number(posted["screen_max_step_mm"], "posted_evaluation.screen_max_step_mm") == 1.0,
            "D-04 requires a separate 1.0 mm screen")
    require(number(posted["convergence_max_step_rs274_mm"],
                   "posted_evaluation.convergence_max_step_rs274_mm") == 0.25,
            "D-04 requires a 0.25 mm RS274 convergence point")
    boolean(posted["screen_is_diagnostic_only"], "posted_evaluation.screen_is_diagnostic_only", True)
    boolean(posted["convergence_is_rs274_only"], "posted_evaluation.convergence_is_rs274_only", True)
    boolean(posted["record_memory_by_phase_and_dialect"],
            "posted_evaluation.record_memory_by_phase_and_dialect", True)
    memory = config["memory"]
    object_fields(memory, "memory",
                  ("max_rss_bytes_per_job", "parallel_jobs", "stop_and_optimize_when_exceeded",
                   "vram_estimate_bytes_per_job"))
    require(integer(memory["max_rss_bytes_per_job"], "memory.max_rss_bytes_per_job") == 8589934592,
            "D-07 requires exactly 8 GiB maximum RSS per job; do not silently raise the cap")
    parallel_jobs = integer(memory["parallel_jobs"], "memory.parallel_jobs", 1)
    require(parallel_jobs <= 2, "D-07 limits parallel jobs to at most two")
    boolean(memory["stop_and_optimize_when_exceeded"],
            "memory.stop_and_optimize_when_exceeded", True)
    if execution["architecture"] == "gpu":
        require(parallel_jobs == 1, "GPU jobs must be serialized on the single shared device")
        require(integer(memory["vram_estimate_bytes_per_job"], "memory.vram_estimate_bytes_per_job", 1)
                >= 1, "GPU execution requires a positive VRAM estimate")
    else:
        require(memory["vram_estimate_bytes_per_job"] == 0,
                "CPU execution must declare zero VRAM usage explicitly")

    budget = object_fields(
        config["budget"], "budget",
        ("candidate_evaluations", "population_size", "validation_interval",
         "max_validation_calls", "max_generations", "learning_rate", "max_gradient_norm",
         "sigma0", "timeout_s", "total_wall_time_s"))
    integer(budget["candidate_evaluations"], "budget.candidate_evaluations", 1)
    integer(budget["population_size"], "budget.population_size", 2)
    integer(budget["validation_interval"], "budget.validation_interval", 1)
    integer(budget["max_validation_calls"], "budget.max_validation_calls")
    if budget["max_generations"] is not None:
        integer(budget["max_generations"], "budget.max_generations", 1)
    for name in ("learning_rate", "max_gradient_norm", "sigma0", "timeout_s", "total_wall_time_s"):
        number(budget[name], f"budget.{name}", 0, strict=True)
    require(config["precision"] == "f64",
            "D-05 requires full f64 precision for matched optimizer comparisons")
    require('precision' in inspect.signature(runner.CSGSimulatorDelta.__init__).parameters,
            "the installed public constructor must expose the f64 precision option")

    initialization_warnings = validate_initializations(config)

    methods = config["methods"]
    require(isinstance(methods, list) and methods, "methods must be a nonempty list")
    method_names: set[str] = set()
    primary_init_selections: set[tuple[str, str]] = set()
    run_selections: set[tuple[str, str, int]] = set()
    jobs = []
    for index, entry in enumerate(methods):
        where = f"methods[{index}]"
        method = object_fields(entry, where, ("condition", "seeds"), ("init",))
        condition = method["condition"]
        require(condition in runner.CONDITIONS, f"{where}.condition is not supported by the runner: {condition!r}")
        selected_init = init_mode(config, method, condition, where)
        selection = (condition, selected_init)
        require(selection not in primary_init_selections,
                f"duplicate primary (condition, init) selection {selection!r}; separate solver/init rows are allowed but not solver seed-splitting")
        primary_init_selections.add(selection)
        method_names.add(condition)
        seeds = seed_list(method["seeds"], where + ".seeds")
        if condition in SINGLE_SEED_METHODS and selected_init == "cam":
            require(seeds == [0], f"{condition} requires deterministic seed 0, not independent CAM/Adam replicas")
        if selected_init == "straight_line":
            require(len(seeds) == 1,
                    f"{where}: straight_line is deterministic and requires exactly one declared run")
        for seed in seeds:
            run_selection = (condition, selected_init, seed)
            require(run_selection not in run_selections,
                    f"{where}: duplicate primary run (condition, init, seed) {run_selection!r}")
            run_selections.add(run_selection)
            jobs.append({"scope": "primary", "condition": condition, "init": selected_init, "seed": seed,
                         "task_id": primary_id, "task_hash": primary["task_hash"]})

    variants = config["dimensionality_subset"]
    require(isinstance(variants, list) and 1 <= len(variants) <= 3,
            "dimensionality_subset must contain 1..3 explicit bounded variants, not an automatic Cartesian sweep")
    variant_names: set[str] = set()
    for index, entry in enumerate(variants):
        where = f"dimensionality_subset[{index}]"
        variant = object_fields(entry, where, ("name", "task_id", "condition", "seed", "k", "rationale"), ("init",))
        name = text(variant["name"], where + ".name")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None,
                f"{where}.name must be a filename-safe variant label")
        require(name not in variant_names, f"duplicate variant label {name!r}")
        variant_names.add(name)
        require(variant["task_id"] in registry, f"{where}: unknown variant task_id {variant['task_id']!r}")
        require(variant["task_id"] == primary_id,
                f"{where}: the current prototype adapter can run only its declared primary task; register alternate task sources before launch")
        condition = variant["condition"]
        require(condition in runner.CONDITIONS, f"{where}.condition is not supported by the runner: {condition!r}")
        require(condition in method_names, f"{where}: condition {condition!r} must be retained in methods")
        selected_init = init_mode(config, variant, condition, where)
        require((condition, selected_init) in primary_init_selections,
                f"{where}: (condition, init) {(condition, selected_init)!r} must be retained in primary methods")
        seed = integer(variant["seed"], where + ".seed")
        if condition in SINGLE_SEED_METHODS and selected_init == "cam":
            require(seed == 0, f"{where}: {condition} requires seed 0")
        k = number(variant["k"], where + ".k", 0, strict=True)
        require(k != primary_k, f"{where}: variant repeats primary_k; it is not a distinct bounded ablation")
        text(variant["rationale"], where + ".rationale")
        jobs.append({"scope": "dimensionality_subset", "variant": name,
                     "condition": condition, "init": selected_init, "seed": seed,
                     "task_id": primary_id, "task_hash": primary["task_hash"], "k": k})

    multiplicity = config["multiplicity"]
    object_fields(multiplicity, "multiplicity",
                  ("independent_unit", "seed_aggregation", "sign_flip_two_sided", "holm_scope",
                   "confidence", "bootstrap_replicates", "minimum_ci_tasks", "descriptive_below_tasks"))
    require(multiplicity["independent_unit"] == "task", "independent unit must be task, not independent seeds")
    require(multiplicity["seed_aggregation"] == "within_task_before_pairing",
            "seeds must be averaged within each task before pairing")
    boolean(multiplicity["sign_flip_two_sided"], "multiplicity.sign_flip_two_sided", True)
    require(multiplicity["holm_scope"] == "all_comparisons_metrics_outcomes_and_families",
            "Holm scope must cover every declared comparison/metric/outcome/family row")
    require(0 < number(multiplicity["confidence"], "multiplicity.confidence") < 1,
            "multiplicity.confidence must be between zero and one")
    integer(multiplicity["bootstrap_replicates"], "multiplicity.bootstrap_replicates", 100)
    require(multiplicity["minimum_ci_tasks"] == 2, "confidence intervals need at least two independent tasks")
    integer(multiplicity["descriptive_below_tasks"], "multiplicity.descriptive_below_tasks", 2)
    comparison_names: set[str] = set()
    metric_names = {"safety_pass", "geometry_pass", "completion_pass", "efficiency_eligible",
                    "cycle_time_s", "air_fraction", "force_proxy", "mrr_mm3_s",
                    "material_removal_rate_mm3_s", "residual_volume_mm3", "gouge_volume_mm3",
                    "max_surface_error_mm", "removed_volume_mm3"}
    for index, raw_comparison in enumerate(config["comparisons"]):
        where = f"comparisons[{index}]"
        comparison = object_fields(raw_comparison, where, ("name", "method", "baseline", "metrics", "dialects"),
                                   ("method_init", "baseline_init"))
        name = text(comparison["name"], where + ".name")
        require(name not in comparison_names, f"duplicate comparison name {name!r}")
        comparison_names.add(name)
        method, baseline = comparison["method"], comparison["baseline"]
        method_init = comparison.get("method_init", "cam")
        baseline_init = comparison.get("baseline_init", "cam")
        for condition, mode, role in ((method, method_init, "method"), (baseline, baseline_init, "baseline")):
            require(mode in INIT_VALUES, f"{where}.{role}_init must be cam, straight_line, or random")
            require((condition, mode) in primary_init_selections,
                    f"{where}: {role} selection {(condition, mode)!r} is not a retained primary (condition, init)")
        require((method, method_init) != (baseline, baseline_init),
                f"{where}: method/init and baseline/init selections must differ")
        metrics = nonempty_names(comparison["metrics"], where + ".metrics")
        require(len(set(metrics)) == len(metrics), f"{where}: duplicate metric")
        require(set(metrics) <= metric_names, f"{where}: unknown metric(s) {sorted(set(metrics)-metric_names)}")
        dialects = comparison["dialects"]
        require(isinstance(dialects, list) and set(dialects) == {"rs274", "haas"},
                f"{where}: both RS274 and Haas evaluations are required by D-04")

    external = config["external_baselines"]
    object_fields(external, "external_baselines",
                  ("cnc_net_executable", "camotics", "cem", "physical_claims", "policy_claims"))
    cem = object_fields(external["cem"], "external_baselines.cem",
                        ("status", "allow_optimizer_condition", "claim_scope", "reason"))
    require(cem["status"] == "unavailable",
            "D-08 records CEM as an unavailable baseline, not an implemented optimizer")
    boolean(cem["allow_optimizer_condition"], "external_baselines.cem.allow_optimizer_condition", False)
    require(cem["claim_scope"] == "zero-order (CMA-ES), not CEM",
            "D-08 requires zero-order (CMA-ES), not CEM claim scope")
    text(cem["reason"], "external_baselines.cem.reason")
    for key, status, flag_name, expected_flag in (
        ("cnc_net_executable", "unsupported", "allow_success_claim", False),
        ("camotics", "supplementary", "primary_oracle", False),
    ):
        row = object_fields(external[key], f"external_baselines.{key}", ("status", flag_name))
        require(row["status"] == status, f"external_baselines.{key}.status must be {status}")
        boolean(row[flag_name], f"external_baselines.{key}.{flag_name}", expected_flag)
    text(external["physical_claims"], "external_baselines.physical_claims")
    text(external["policy_claims"], "external_baselines.policy_claims")
    acceptance = config["acceptance"]
    object_fields(acceptance, "acceptance", ("strict_improvement_required", "allowed_conclusions"))
    require(acceptance["strict_improvement_required"] is False,
            "strict-improvement gating is forbidden")
    require(set(acceptance["allowed_conclusions"]) == {"accept", "reject", "narrow"},
            "acceptance must explicitly permit accept, reject, and narrow outcomes")
    tuning_spend, tuning_rows = validate_tuning(config["tuning"], repo, primary_init_selections)
    costs = check_budget(config, jobs)
    return {"task": task, "task_hash": task.task_hash, "primary_task_id": primary_id,
            "protocol": config,
            "prototype_task_matches": probe_match,
            "primary_split": registry[primary_id]["split"], "split_digests": split_digests,
            "decision_ids": decision_ids, "decision_record_sha256": decision_hash,
            "jobs": jobs, "costs": costs,
            "prototype_only_warnings": [
                "The real native runner currently supports only prototype_analytic + dev + pilot. These are plan records, not a full frozen-campaign integration result.",
                "D-05 f64 declaration is checked against the public precision option; this dry-run does not initialize f64 fields or prove an adjoint calculation.",
                "D-04 fidelity and D-07 memory declarations are validated, but no posted evaluator or phase memory allocation is executed here.",
                "No tuned or test-split results are inferred from incomplete optional timing or baseline coverage.",
                *initialization_warnings,
            ], "tuning_spend": tuning_spend, "tuning_rounds": tuning_rows}


def native_config(job: Mapping[str, Any], declaration: Mapping[str, Any],
                  protocol_sha256: str) -> dict[str, Any]:
    """Translate one approved case into the existing runner's native schema."""
    budget = declaration["budget"]
    objective = declaration["objective"]
    execution = declaration["execution"]
    k = job.get("k", objective["primary_k"])
    seconds = declaration["estimates"]["seconds_per_candidate_by_method"]
    return {
        "schema": runner.RUNNER_SCHEMA,
        "experiment": execution["experiment"],
        "task_source": execution["task_source"],
        "precision": declaration["precision"],
        "init": job["init"],
        "init_implementation_available": job["init"] == "cam",
        "arch": execution["architecture"],
        "seeds": [job["seed"]],
        "candidate_evaluations": budget["candidate_evaluations"],
        "population_size": budget["population_size"],
        "validation_interval": budget["validation_interval"],
        "max_validation_calls": budget["max_validation_calls"],
        "max_generations": budget["max_generations"],
        "learning_rate": budget["learning_rate"],
        "max_gradient_norm": budget["max_gradient_norm"],
        "k_fixed": k,
        "k_smooth": k,
        "k_anneal_start": objective["anneal_start"],
        "k_anneal_end": objective["anneal_end"],
        "sigma0": budget["sigma0"],
        "timeout_s": budget["timeout_s"],
        # Declared metadata for downstream adapters, not exercised by prototype.
        "simulator_precision": declaration["precision"],
        "protocol_objective": dict(objective),
        "posted_evaluation": dict(declaration["posted_evaluation"]),
        "memory": dict(declaration["memory"]),
        "protocol_reference_sha256": protocol_sha256,
        "notes": "prototype dry-run only; not an optimizer run, f64 field proof, or physical pilot result",
        "estimated_seconds_per_candidate": (
            {job["condition"]: seconds[job["condition"]]} if job["condition"] in seconds else {}),
    }


def execute_dry_runs(declaration: Mapping[str, Any], output: Path,
                     protocol_sha256: str) -> list[dict[str, Any]]:
    """Run only the actual CLI dry-run path and reconcile its immutable plans."""
    destination = no_manuscript(output, "dry-run output")
    if destination.exists():
        raise ProtocolError(f"dry-run output must be a fresh directory: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    reports = []
    proto_task, _, _ = runner._prototype()
    for index, job in enumerate(declaration["jobs"]):
        label = f"{index:03d}-{job['scope']}-{job['condition']}-i{job['init']}-s{job['seed']}"
        root = destination / label
        root.mkdir(parents=True, exist_ok=False)
        if job["init"] != "cam":
            # D-08 runner/seeds belong to a separate downstream implementation.
            # Do not call the CAM runner or imply that this declaration generated
            # a valid MotionPlan, an executable configuration, or an outcome.
            preview = {
                "schema": "diffcam-initialization-declaration/1",
                "declared_not_yet_runnable": True,
                "scope": job["scope"], "condition": job["condition"],
                "init": job["init"], "seed": job["seed"],
                "variant": job.get("variant"),
                "task_id": job["task_id"], "task_hash": job["task_hash"],
                "k": job.get("k", declaration["protocol"]["objective"]["primary_k"]),
                "objective": dict(declaration["protocol"]["objective"]),
                "budget": dict(declaration["protocol"]["budget"]),
                "precision": declaration["protocol"]["precision"],
                "random_init": dict(declaration["protocol"]["random_init"]),
                "straight_line_init": dict(declaration["protocol"]["straight_line_init"]),
                "collapse_criterion": dict(declaration["protocol"]["collapse_criterion"]),
                "protocol_reference_sha256": protocol_sha256,
                "reason": "D-08 initializer/runner seed construction is not implemented yet; this is neither a native configuration nor an execution plan",
            }
            preview_path = root / "initialization-declaration.json"
            preview_path.write_bytes(canonical_json(preview) + b"\n")
            reports.append({
                "scope": job["scope"], "condition": job["condition"], "init": job["init"],
                "seed": job["seed"], "variant": job.get("variant"), "run_id": None,
                "task_hash": job["task_hash"],
                "initialization_declaration_sha256": file_sha(preview_path),
                "declared_not_yet_runnable": True, "native_dry_run_performed": False,
                "planned_run_count": 0, "candidate_evaluations": None, "validation_calls": None,
                "estimated_wall_time_s": None, "planned_not_executed": True,
                "reason": preview["reason"], "output_subdirectory": label,
            })
            continue
        native = native_config(job, declaration["protocol"], protocol_sha256)
        config_path = root / "runner-config.json"
        config_path.write_bytes(canonical_json(native) + b"\n")
        selector = f"{job['condition']}-s{job['seed']}-*"
        expected_entries = runner._plan_entries(
            native, proto_task, declaration["protocol"]["execution"]["split"], selector)
        require(len(expected_entries) == 1,
                f"{label}: selector does not identify exactly one run: {expected_entries}")
        expected_id = expected_entries[0]["run_id"]
        command = [sys.executable, str(REPOSITORY / "scripts/run_benchmark_suite.py"),
                   "--config", str(config_path), "--output-dir", str(root / "plan"),
                   "--split", declaration["protocol"]["execution"]["split"],
                   "--experiment", declaration["protocol"]["execution"]["experiment"],
                   "--dry-run", "--only", selector]
        environment = dict(os.environ)
        try:
            completed = subprocess.run(
                command, cwd=str(REPOSITORY), env=environment,
                capture_output=True, text=True, timeout=60, check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ProtocolError(f"{label}: prototype dry-run timed out before evidence was produced") from error
        require(completed.returncode == 0,
                f"{label}: real runner dry-run failed ({completed.returncode}): "
                f"{completed.stdout}\n{completed.stderr}")
        manifest, manifest_sha, _ = load_json(root / "plan/manifest.json")
        cost, cost_sha, _ = load_json(root / "plan/dry_run.json")
        budget, budget_sha, _ = load_json(root / "plan/frozen_config.json")
        require(manifest.get("schema_version") == 1, f"{label}: unexpected runner manifest schema")
        require(manifest.get("runner", {}).get("planned_not_executed") is True,
                f"{label}: runner did not declare this output a plan")
        require(manifest["runner"]["split"] == "dev" and
                manifest["runner"]["experiment"] == declaration["protocol"]["execution"]["experiment"],
                f"{label}: runner experiment/split drift")
        observed = manifest.get("expected_runs")
        require(isinstance(observed, list) and len(observed) == 1,
                f"{label}: expected one dry-run, got {len(observed) if isinstance(observed,list) else 'unknown'}")
        require(observed[0].get("run_id") == expected_id and
                observed[0].get("method") == job["condition"] and observed[0].get("seed") == job["seed"],
                f"{label}: runner run identity was not preserved: {observed!r} versus expected {expected_id}")
        expected_budget = 1 if job["condition"] == "nominal-cam" else declaration["protocol"]["budget"]["candidate_evaluations"]
        expected_validations = min(declaration["protocol"]["budget"]["max_validation_calls"],
                                   expected_budget // declaration["protocol"]["budget"]["validation_interval"])
        require(cost.get("planned_run_count") == 1 and len(cost.get("costs", [])) == 1,
                f"{label}: native dry-run count does not reconcile")
        native_cost = cost["costs"][0]
        require(native_cost["candidate_evaluations"] == expected_budget and
                native_cost["validation_calls"] == expected_validations,
                f"{label}: native budget/validation count differs from the frozen declaration")
        seconds = declaration["protocol"]["estimates"]["seconds_per_candidate_by_method"].get(job["condition"])
        expected_seconds = seconds * (expected_budget + expected_validations) if seconds is not None else None
        require(native_cost["estimated_run_wall_time_s"] == expected_seconds,
                f"{label}: native per-run time estimate differs: {native_cost['estimated_run_wall_time_s']} != {expected_seconds}")
        require(not (root / "plan" / "nominal-cam").exists(),
                f"{label}: a real nominal evaluation must not be launched during dry-run")
        require(budget.get("raw_config", {}).get("simulator_precision") == "f64" and
                budget.get("raw_config", {}).get("precision") == "f64",
                f"{label}: frozen config lost f64 declaration")
        reports.append({
            "scope": job["scope"], "condition": job["condition"], "init": job["init"], "seed": job["seed"],
            "variant": job.get("variant"), "run_id": expected_id,
            "task_hash": job["task_hash"],
            "raw_runner_config_sha256": file_sha(config_path),
            "runner_manifest_sha256": manifest_sha,
            "runner_dry_run_sha256": cost_sha,
            "runner_frozen_config_sha256": budget_sha,
            "planned_run_count": 1, "candidate_evaluations": expected_budget,
            "validation_calls": expected_validations,
            "estimated_wall_time_s": expected_seconds,
            "planned_not_executed": True, "output_subdirectory": label,
            "declared_not_yet_runnable": False, "native_dry_run_performed": True,
        })
    return reports


def validate(config: Mapping[str, Any], *, repo_root: Path | None = None,
             decisions_path: Path | None = None, output_dir: Path | None = None,
             dry_run: bool = True, protocol_sha256: str | None = None) -> dict[str, Any]:
    """Full declaration gate, followed by optional explicit prototype planning."""
    repo = REPOSITORY if repo_root is None else repo_root
    decisions = REPOSITORY / "reports/decisions.md" if decisions_path is None else decisions_path
    declaration = validate_declaration(config, repo, decisions)
    source_sha = protocol_sha256 or "sha256:" + hashlib.sha256(canonical_json(config)).hexdigest()
    report = {
        "schema": "diffcam-protocol-validation/1",
        "mode": "tooling_only_not_a_campaign_launch",
        "campaign_precision": config["precision"],
        "decision_ids": declaration["decision_ids"],
        "decision_record_sha256": declaration["decision_record_sha256"],
        "protocol_sha256": source_sha,
        "primary_task_id": declaration["primary_task_id"],
        "primary_task_hash": declaration["task_hash"],
        "prototype_adapter_compatible": declaration["prototype_task_matches"],
        "split_hashes": declaration["split_digests"],
        "cost_accounting": {key: value for key, value in declaration["costs"].items()
                             if key not in ("per_job", "uncertainty_note")},
        "tuning_spend_candidate_evaluations": declaration["tuning_spend"],
        "tuning_evidence": declaration["tuning_rounds"],
        "warnings": declaration["prototype_only_warnings"],
        "dry_run_performed": False,
        "dry_runs": [],
        "declared_not_yet_runnable_count": sum(job["init"] != "cam" for job in declaration["jobs"]),
        "initialization_declarations": [
            {"scope": job["scope"], "condition": job["condition"], "init": job["init"],
             "seed": job["seed"], "variant": job.get("variant"), "task_hash": job["task_hash"]}
            for job in declaration["jobs"] if job["init"] != "cam"
        ],
    }
    if dry_run and declaration["protocol"]["execution"]["experiment"] != "pilot":
        raise ProtocolError(
            f"current runner implements only pilot; {declaration['protocol']['execution']['experiment']!r} requires a reviewed runner upgrade")
    if dry_run and not declaration["prototype_task_matches"]:
        raise ProtocolError(
            "current dry-run adapter requires the actual prototype task identity; "
            "use --validate-only for registered non-prototype task manifests or supply a matched task-source adapter")
    if dry_run:
        report["dry_runs"] = execute_dry_runs(declaration, output_dir, source_sha)
        report["dry_run_performed"] = any(row["native_dry_run_performed"] for row in report["dry_runs"])
        require(len(report["dry_runs"]) == declaration["costs"]["jobs"],
                "not every declared run/variant was either natively checked or marked pending")
        report["validated_run_count"] = sum(row["native_dry_run_performed"] for row in report["dry_runs"])
        if output_dir is not None:
            output = no_manuscript(output_dir, "dry-run report")
            (output / "validation-report.json").write_bytes(canonical_json(report) + b"\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="protocol JSON. The coordinator supplies the frozen values; no defaults are invented.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="fresh dry-run output directory outside manuscript/")
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY,
                        help="root under which manifest and tuning evidence paths resolve")
    parser.add_argument("--decisions", type=Path,
                        default=REPOSITORY / "reports/decisions.md",
                        help="immutable decision markdown containing all declared D-IDs")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate declaration and hashes without invoking the current prototype adapter")
    args = parser.parse_args(argv)
    try:
        config, protocol_sha, _ = load_json(args.config)
        report = validate(config, repo_root=args.repo_root, decisions_path=args.decisions,
                          output_dir=args.output_dir, dry_run=not args.validate_only,
                          protocol_sha256=protocol_sha)
    except (ProtocolError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "status": report["mode"], "dry_run_performed": report["dry_run_performed"],
        "validated_run_count": report.get("validated_run_count", 0),
        "declared_not_yet_runnable_count": report["declared_not_yet_runnable_count"],
        "protocol_sha256": protocol_sha,
        "output_dir": str(args.output_dir.resolve()) if not args.validate_only else None,
    }, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
