"""Export reviewable physical-trial packets, without operating equipment.

This is a packaging tool, not a machine controller or a physical-safety proof.
A development packet always has physical status ``pending setup review``. Its
simulation scope may be ``simulation-qualified`` only after source hashes,
complete production/reference coverage, local geometry, conservative assembly
clearance, rapid penetration and independent stock-motion checks are present.

Final protocol packets additionally require an explicitly supplied, hash-checked
T4.2 frozen-protocol record. In either case, a qualified human operator must
perform the setup review, including an air cut, Z touch-off, holder-stickout
measurement and fixture-clearance review. No exported status authorizes cutting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence

from benchmarks import contracts as C
from cam.gcode_parser import GCodeError, parse_motion_plan


SCHEMA_VERSION = "physical-trial-packet/1"
SETUP_SCHEMA = "physical-trial-setup/1"
PROTOCOL_SCHEMA = "physical-trial-protocol/1"
DIALECTS = tuple(dialect.value for dialect in C.Dialect)
ROLES = ("baseline", "optimized")
MANUAL_SETUP = ("first-article air cut", "Z touch-off", "holder-stickout measurement", "fixture clearance")
_PAYLOAD = (
    "task.json", "motion_plan.json", "run.json", "config.json", "metrics.json",
    "sources.json", "hashes.json", "nominal_result.json", "eval_posted.json",
    "rs274.nc", "haas.nc", "eval_posted_rs274.json", "eval_posted_haas.json",
    "reference_rs274.json", "reference_haas.json",
)


class PacketError(ValueError):
    """An incomplete, inconsistent or unsafe-to-classify packet input."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{name} must be numeric, not boolean")
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    if positive:
        _require(result > 0, f"{name} must be positive")
    return result


def _text(mapping: Mapping[str, Any], key: str, prefix: str) -> str:
    value = mapping.get(key)
    _require(isinstance(value, str) and value.strip(), f"{prefix}.{key} is required")
    return value


def _mapping(value: Any, prefix: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{prefix} must be a JSON object")
    return copy.deepcopy(dict(value))


def _plain(value: Any) -> Any:
    return json.loads(C.canonical_json(value))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    _require(not relative.is_absolute() and relative.parts and
             all(part not in ("", ".", "..") for part in relative.parts),
             f"unsafe relative artifact name: {name!r}")
    return root.joinpath(*relative.parts)


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise PacketError(f"cannot read packet input {path}: {error}") from error


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PacketError(f"cannot read immutable input {path}: {error}") from error
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON number {token}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise PacketError(f"cannot load packet input {path}: {error}") from error


def _typed(path: Path, expected: type) -> Any:
    try:
        result = C.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, C.ContractError) as error:
        raise PacketError(f"typed record {path} is invalid: {error}") from error
    _require(isinstance(result, expected), f"{path} must contain {expected.__name__}")
    return result


def _hash_field(value: Any, name: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value),
             f"{name} must be an explicit sha256 digest")
    return value


def _metric(result: C.EvaluationResult, name: str, unit: str, label: str) -> float:
    metric = result.metrics.get(name)
    _require(metric is not None, f"{label} lacks required metric {name}")
    _require(metric.unit == unit and metric.value is not None,
             f"{label}.{name} is unavailable or has wrong unit {metric.unit!r}")
    return _number(metric.value, f"{label}.{name}")


def _checked_clearance(result: C.EvaluationResult, task: C.TaskSpec, label: str,
                       *, geometry: bool) -> dict[str, float]:
    _require(result.coverage is C.Coverage.COMPLETE,
             f"{label} simulation coverage is {result.coverage.value}, not complete")
    _require(result.safety_pass is True, f"{label} safety is not true")
    if geometry:
        _require(result.geometry_pass is True, f"{label} local geometry did not pass")
    else:
        _require(result.geometry_pass is not False,
                 f"{label} reports an explicit independent geometry failure")
    clearance = _metric(result, "min_clearance_lower_bound_mm", "mm", label)
    penetration = _metric(result, "rapid_cutter_penetration_upper_bound_mm", "mm", label)
    local = _metric(result, "max_surface_error_mm", "mm", label) if geometry else None
    residual = _metric(result, "residual_volume_mm3", "mm^3", label) if geometry else None
    gouge = _metric(result, "gouge_volume_mm3", "mm^3", label) if geometry else None
    _require(clearance >= task.tolerances.clearance_mm,
             f"{label} noncutting clearance lower bound {clearance} mm is below "
             f"declared {task.tolerances.clearance_mm} mm")
    _require(penetration <= task.tolerances.rapid_penetration_mm,
             f"{label} rapid penetration upper bound exceeds declared tolerance")
    if local is not None:
        _require(local <= task.tolerances.local_surface_error_mm,
                 f"{label} max local surface error exceeds declared tolerance")
        _require(residual <= task.tolerances.completion_residual_mm3,
                 f"{label} residual exceeds declared tolerance")
        _require(gouge <= task.tolerances.gouge_volume_mm3,
                 f"{label} gouge exceeds declared tolerance")
        original = _metric(result, "original_part_gouge_volume_mm3", "mm^3", label)
        _require(original <= task.tolerances.gouge_volume_mm3,
                 f"{label} original-part gouge exceeds declared tolerance")
        _require(_metric(result, "removed_volume_mm3", "mm^3", label) > 0,
                 f"{label} has no actual material removal in this operation")
    return {
        "min_noncutting_clearance_lower_bound_mm": clearance,
        "declared_min_noncutting_clearance_mm": task.tolerances.clearance_mm,
        "max_rapid_cutter_penetration_upper_bound_mm": penetration,
        "declared_max_rapid_penetration_mm": task.tolerances.rapid_penetration_mm,
        **({} if local is None else {
            "max_local_surface_error_mm": local,
            "declared_max_local_surface_error_mm": task.tolerances.local_surface_error_mm,
            "residual_volume_mm3": residual,
            "gouge_volume_mm3": gouge,
        }),
    }


def _source_fields(run_dir: Path, sources: Mapping[str, Any]) -> dict[str, Any]:
    verification = {}
    for kind, filename in (("stock", "source_stock.npy"), ("target", "source_target.npy")):
        record = _mapping(sources.get(kind), f"sources.{kind}")
        expected = _hash_field(record.get("sha256"), f"sources.{kind}.sha256")
        # Retained runs carry local immutable snapshots; source JSON may name a
        # since-cleaned authoring worktree. Never silently use that stale path.
        path = _safe_path(run_dir, filename)
        _require(path.is_file(), f"missing retained immutable {kind} field snapshot {filename}")
        actual = _file_digest(path)
        _require(actual == expected,
                 f"immutable {kind} field hash mismatch: {actual} != {expected}")
        _require(record.get("grid") == _plain(run_dir_task_grid(run_dir)),
                 f"source {kind} grid does not match typed task grid")
        verification[kind] = {
            "retained_snapshot": filename, "sha256": actual,
            "declared_source_path_preserved_in_sources_json": record.get("path"),
            "grid": record["grid"],
        }
    return verification


def run_dir_task_grid(run_dir: Path) -> C.GridSpec:
    """Read the grid only for source-field verification; no geometry is rebuilt."""
    return _typed(run_dir / "task.json", C.TaskSpec).grid


def _validate_setup(raw: Mapping[str, Any], task: C.TaskSpec, motion: C.MotionPlan,
                    dialect: str) -> dict[str, Any]:
    setup = _mapping(raw, "setup")
    _require(setup.get("schema_version") == SETUP_SCHEMA,
             f"setup.schema_version must be {SETUP_SCHEMA}")
    machine = _mapping(setup.get("machine"), "machine")
    for key in ("identifier", "manufacturer", "model", "record_source"):
        _text(machine, key, "machine")
    controllers = _mapping(machine.get("controllers"), "machine.controllers")
    _text(controllers, dialect, f"machine.controllers[{dialect!r}]")

    tool = _mapping(setup.get("tool"), "tool")
    _text(tool, "setup_record_id", "tool")
    _require(tool.get("assembly") == _plain(task.tool),
             "physical tool/holder assembly does not exactly match the evaluated task tool")
    declared = _number(tool.get("declared_stickout_mm"), "tool.declared_stickout_mm", positive=True)
    _require(declared == task.tool.stickout_mm,
             "declared holder stickout differs from evaluated task.tool.stickout_mm")
    measurement = _mapping(tool.get("measurement"), "tool.measurement")
    state = _text(measurement, "state", "tool.measurement")
    _require(state in ("declared_not_measured", "measured_with_record"),
             "tool measurement state must be declared_not_measured or measured_with_record")
    if state == "measured_with_record":
        _text(measurement, "operator_record_id", "tool.measurement")

    offset = _mapping(setup.get("work_offset"), "work_offset")
    _require(offset.get("identifier") == task.initial_state.work_offset,
             "work-offset identifier differs from evaluated initial state")
    _require(offset.get("transform") == _plain(task.work_to_machine),
             "work offset transform differs from task.work_to_machine")
    _require(task.initial_state.known and task.initial_state.position_mm is not None,
             "initial machine state must be known; cannot infer an approach pose")
    _require(offset.get("initial_machine_position_mm") == list(task.initial_state.position_mm),
             "setup initial machine position differs from evaluated motion state")

    fixtures = _mapping(setup.get("fixtures"), "fixtures")
    _require(fixtures.get("geometry") == _plain(task.fixtures),
             "fixture geometry/ordering differs from evaluated task fixtures")
    identities = fixtures.get("identifiers")
    _require(isinstance(identities, list) and
             all(isinstance(item, str) and item.strip() for item in identities) and
             len(identities) == len(task.fixtures),
             "physical fixture identity records must match each evaluated fixture")
    if not task.fixtures:
        _text(fixtures, "unrepresented_geometry_note", "fixtures")
        _require(fixtures.get("complete_fixture_representation") is False,
                 "empty task fixtures require explicit incomplete physical fixture representation")

    floor = _mapping(setup.get("floor"), "floor")
    floor_kind = _text(floor, "kind", "floor")
    _text(floor, "basis", "floor")
    if floor_kind == "retained_stock_below_cut":
        params = task.original_part.params
        _require(params.get("primitive") == "box_minus_pocket",
                 "retained-floor automatic cross-check requires the declared box-minus-pocket task")
        stock_bottom = params["box_min_mm"][2]
        cut_floor = params["pocket_min_mm"][2]
        _require(_number(floor.get("stock_bottom_mm"), "floor.stock_bottom_mm") == stock_bottom and
                 _number(floor.get("cut_floor_mm"), "floor.cut_floor_mm") == cut_floor,
                 "floor coordinates differ from the evaluated part")
        remaining = cut_floor - stock_bottom
        _require(remaining > 0 and
                 _number(floor.get("remaining_thickness_mm"), "floor.remaining_thickness_mm", positive=True) == remaining,
                 "remaining floor thickness differs from evaluated part geometry")
    elif floor_kind == "through_hole_with_setup_specific_clearance":
        _number(floor.get("minimum_bottom_clearance_mm"), "floor.minimum_bottom_clearance_mm", positive=True)
        _text(floor, "operator_reference", "floor")
    else:
        raise PacketError("unsupported floor policy; through-hole rules must be setup/part-specific")

    material = _mapping(setup.get("material"), "material")
    for key in ("name", "alloy_or_grade", "stock_record"):
        _text(material, key, "material")
    process = _mapping(setup.get("process"), "process")
    _number(process.get("spindle_rpm"), "process.spindle_rpm", positive=True)
    _require(process.get("spindle_direction") in ("clockwise", "counterclockwise"),
             "process.spindle_direction must be declared")
    _text(process, "coolant", "process")
    _require(process.get("motion_mode") == motion.timing_rule.value,
             "process motion mode differs from typed motion timing")
    _require(process.get("speed_clipping") is False,
             "speed clipping cannot silently change an evaluated physical program")
    _require(process.get("length_unit") == "mm", "physical packet requires millimetres")
    return setup


def _validate_protocol(raw: Mapping[str, Any], final_for_protocol: bool) -> dict[str, Any]:
    protocol = _mapping(raw, "protocol")
    _require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
             f"protocol.schema_version must be {PROTOCOL_SCHEMA}")
    _text(protocol, "protocol_id", "protocol")
    freeze = protocol.get("freeze_state")
    _require(freeze in ("pending", "frozen"), "protocol.freeze_state must be pending or frozen")
    _require(not final_for_protocol or freeze == "frozen",
             "final packets require the T4.2 frozen protocol")
    if freeze == "frozen":
        record = _mapping(protocol.get("frozen_record"), "protocol.frozen_record")
        raw_path = _text(record, "path", "protocol.frozen_record")
        actual = _file_digest(Path(raw_path).expanduser())
        _require(actual == _hash_field(record.get("sha256"), "protocol.frozen_record.sha256"),
                 "frozen protocol record hash mismatch")
    return protocol


def _selection(root: Path, role: str, selected_dialect: str,
               setup: Mapping[str, Any], final_for_protocol: bool,
               refinement_link: str | Path | None) -> tuple[
                   C.TaskSpec, dict[str, str], dict[str, Any], dict[str, bytes]
               ]:
    _require(role in ROLES, f"selection role must be one of {ROLES}")
    _require(root.is_dir(), f"missing selected {role} run directory: {root}")
    hashes = _mapping(_load(_safe_path(root, "hashes.json")), f"{role}.hashes")
    result = _mapping(_load(_safe_path(root, "nominal_result.json")), f"{role}.nominal_result")
    _require(result.get("hashes") == hashes,
             f"{role} nominal-result hashes differ from the separate artifact map")
    for name, expected in sorted(hashes.items()):
        _safe_path(root, name)
        _hash_field(expected, f"{role}.hashes[{name!r}]")
        _require(_file_digest(_safe_path(root, name)) == expected,
                 f"immutable {role} artifact hash mismatch: {name}")
    for name in _PAYLOAD:
        _require(name in hashes or name in ("hashes.json", "nominal_result.json"),
                 f"selected {role} record lacks required artifact {name}")
        _require(_safe_path(root, name).is_file(), f"missing selected {role} artifact {name}")

    run = _typed(root / "run.json", C.RunRecord)
    task = _typed(root / "task.json", C.TaskSpec)
    motion = _typed(root / "motion_plan.json", C.MotionPlan)
    try:
        motion.validate_for_task(task)
    except (ValueError, C.ContractError) as error:
        raise PacketError(f"typed motion does not match selected task: {error}") from error
    _require(run.status is C.RunStatus.COMPLETE and run.failure_reason is None,
             f"selected {role} run is not a complete immutable source run")
    _require(result.get("run_id") == run.run_id and result.get("status") == "complete",
             f"{role} nominal result identity/status differs from typed run")
    _require(run.task_hash == task.task_hash,
             f"selected {role} run task hash differs from typed task")
    _require(set(run.artifacts) | {"run.json"} == set(hashes),
             f"{role} run artifact index differs from immutable hash map")
    config = _mapping(_load(root / "config.json"), "config")
    _require(config.get("source_commit") == run.source_commit,
             f"{role} source commit differs between run and config")
    _require(config.get("dirty_patch_hash") == run.dirty_patch_hash,
             f"{role} dirty-patch provenance differs between run and config")
    _require(config.get("independent_evaluation"),
             f"{role} lacks declared independent evaluation methods")
    sources = _mapping(_load(root / "sources.json"), "sources")
    source_verification = _source_fields(root, sources)
    index = _mapping(_load(root / "eval_posted.json"), "eval_posted")
    programs = _mapping(index.get("programs"), "eval_posted.programs")
    metrics = _mapping(_load(root / "metrics.json"), "metrics")
    completion = _mapping(metrics.get("completion"), "metrics.completion")
    _require(completion.get("sampled_geometry_pass") is True,
             f"{role} sampled simulator geometry is unavailable/failing")
    _require(completion.get("reference_geometry_pass") is None and
             metrics.get("reference_geometry_pass") is None,
             f"{role} stock-motion reference geometry must remain unknown, not be converted to a pass")
    _require(result.get("completion") == completion,
             f"{role} completion records differ between nominal result and metrics")
    sampled_exclusion = _mapping(completion.get("sampled_exclusion"), "sampled_exclusion")
    _require(sampled_exclusion.get("declared_grid_volume_mm3") == task.operation_target.excluded_volume_mm3 and
             sampled_exclusion.get("grid_volume_mm3") == task.operation_target.excluded_volume_mm3,
             f"{role} sampled operation-target exclusion differs from predeclared target")
    _validate_setup(setup, task, motion, selected_dialect)
    training = _mapping(metrics.get("training_execution"), "training_execution")
    _require(training.get("speed_clipping") is False,
             f"{role} silently speed-clipped its execution")
    _require(_number(training.get("max_position_error_mm"), "training.max_position_error_mm") <=
             _number(training.get("position_tolerance_mm"), "training.position_tolerance_mm", positive=True),
             f"{role} training execution path differs from typed motion")
    parse_comparisons = _mapping(metrics.get("post_parse_equivalence"), "post_parse_equivalence")
    disagreements = _mapping(metrics.get("raw_stock_cell_disagreement"), "raw_stock_cell_disagreement")
    production_summaries = _mapping(metrics.get("production_posted_evaluator"), "production_posted_evaluator")

    evaluations = {}
    cross_checks = {}
    posted_results = {}
    for name in DIALECTS:
        dialect = C.Dialect(name)
        program = _mapping(programs.get(name), f"posted index {name}")
        nc_path = root / f"{name}.nc"
        nc_bytes = nc_path.read_bytes()
        nc_hash = C.nc_hash(nc_bytes)
        _require(program.get("status") == "supported" and program.get("nc_hash") == nc_hash,
                 f"{role}/{name} posted NC index/hash mismatch")
        _require(program.get("eval_file") == f"eval_posted_{name}.json",
                 f"{role}/{name} points to an unexpected posted evaluation")
        posted = _typed(root / f"eval_posted_{name}.json", C.EvaluationResult)
        posted_results[name] = posted
        reference = _typed(root / f"reference_{name}.json", C.EvaluationResult)
        for kind, record in (("production", posted), ("reference", reference)):
            _require(record.dialect is dialect and record.nc_hash == nc_hash,
                     f"{role}/{name} {kind} evaluation does not match raw NC bytes")
            _require(record.task_hash == task.task_hash and record.tool_setup_hash == task.tool_setup_hash,
                     f"{role}/{name} {kind} task/setup hash mismatch")
        _require(posted.evaluator_id == "posted_hard/production" and posted.efficiency_eligible is True,
                 f"{role}/{name} is not a complete production hard evaluation")
        _require(reference.evaluator_id == "reference_carve" and
                 reference.evaluator_id != posted.evaluator_id and reference.efficiency_eligible is False,
                 f"{role}/{name} lacks a distinct independent reference cross-check")
        try:
            parsed_motion = parse_motion_plan(nc_bytes.decode("utf-8"), task, dialect=dialect)
        except (UnicodeDecodeError, GCodeError, ValueError, C.ContractError) as error:
            raise PacketError(f"{role}/{name} raw NC does not parse as its declared task/dialect: {error}") from error
        parsed_hash = parsed_motion.motion_hash
        _require(posted.discretization.get("parsed_motion_hash") == parsed_hash,
                 f"{role}/{name} actual parsed-NC motion hash differs from production record")
        _require(posted.motion_hash in (None, parsed_hash),
                 f"{role}/{name} posted motion hash contradicts actual parsed NC")
        settings = _mapping(posted.discretization.get("settings"), f"{name}.posted.settings")
        _require(settings.get("expected_nc_hash") == nc_hash and settings.get("compute_bounds") is True and
                 settings.get("safety_screen") is True,
                 f"{role}/{name} production evaluator did not check complete bounds/expected NC")
        tolerance = task.operation_target.geometry.params.get("grid_exclusion_tolerance_mm3")
        _require(settings.get("exclusion_tolerance_mm3") == tolerance,
                 f"{role}/{name} silently changed grid exclusion tolerance")
        reference_settings = _mapping(reference.discretization.get("settings"), f"{name}.reference.settings")
        subdivisions = reference_settings.get("subdivisions")
        _require(isinstance(subdivisions, int) and not isinstance(subdivisions, bool) and subdivisions >= 2,
                 f"{role}/{name} lacks explicit independently refined replay settings")
        primary = _checked_clearance(posted, task, f"{role}/{name}/production", geometry=True)
        independent = _checked_clearance(reference, task, f"{role}/{name}/reference", geometry=False)
        comparison = _mapping(disagreements.get(name), f"disagreement.{name}")
        _require(comparison.get("scope_supported") is True and
                 comparison.get("status") in ("identical", "boundary_contacts_only"),
                 f"{role}/{name} independent stock-motion disagreement is unresolved")
        outside = comparison.get("outside_local_tolerance_cells")
        _require(isinstance(outside, int) and not isinstance(outside, bool) and outside == 0,
                 f"{role}/{name} independent disagreement exceeds local cell tolerance")
        _require(comparison.get("local_boundary_tolerance_mm") == task.tolerances.local_surface_error_mm and
                 _number(comparison.get("max_abs_contact_distance_mm"), "contact.max_abs_contact_distance_mm") <=
                 task.tolerances.local_surface_error_mm,
                 f"{role}/{name} contact disagreement exceeds declared local tolerance")
        parsed = _mapping(parse_comparisons.get(name), f"post_parse_equivalence.{name}")
        _require(parsed.get("semantic_tag_mismatches") == 0,
                 f"{role}/{name} parse equivalence changed physical segment classification")
        _require(_number(parsed.get("max_position_error_mm"), f"{name}.parse.max_position_error_mm") <= 0.002 and
                 _number(parsed.get("max_time_error_s"), f"{name}.parse.max_time_error_s") <= 1e-5,
                 f"{role}/{name} parse equivalence exceeds declared pilot replay tolerances")
        summary = _mapping(production_summaries.get(name), f"production summary.{name}")
        _require(summary.get("status") == "evaluated" and
                 summary.get("coverage") == posted.coverage.value and
                 summary.get("safety_pass") is True and summary.get("geometry_pass") is True and
                 summary.get("residual_volume_mm3") == posted.metrics["residual_volume_mm3"].value and
                 summary.get("gouge_volume_mm3") == posted.metrics["gouge_volume_mm3"].value and
                 summary.get("max_surface_error_mm") == posted.metrics["max_surface_error_mm"].value and
                 summary.get("min_clearance_lower_bound_mm") == primary["min_noncutting_clearance_lower_bound_mm"] and
                 summary.get("rapid_penetration_upper_bound_mm") == primary["max_rapid_cutter_penetration_upper_bound_mm"],
                 f"{role}/{name} production summary differs from typed evaluation")
        evaluations[name] = {
            "selected": name == selected_dialect,
            "nc": f"programs/{role}/{name}.nc", "nc_sha256": nc_hash,
            "production_record": f"programs/{role}/eval_posted_{name}.json",
            "evaluator_id": posted.evaluator_id,
            "evaluator_revision": posted.evaluator_revision,
            "coverage": posted.coverage.value, "safety_pass": posted.safety_pass,
            "geometry_pass": posted.geometry_pass,
            "qualification": primary,
            "discretization": _plain(posted.discretization),
            "force_proxy": _plain(posted.metrics.get("force_proxy")),
            "refined_recheck": {
                "record": f"programs/{role}/reference_{name}.json",
                "evaluator_id": reference.evaluator_id,
                "evaluator_revision": reference.evaluator_revision,
                "coverage": reference.coverage.value, "safety_pass": reference.safety_pass,
                "geometry_pass": reference.geometry_pass,
                "scope": "independent stock-motion replay; not a production geometry score",
                "settings": _plain(reference_settings),
                "temporal_refinement_is_nested_vs_production": None,
                "qualification": independent,
                "discretization": _plain(reference.discretization),
            },
        }
        cross_checks[name] = {
            "parse_equivalence": parsed,
            "raw_stock_cell_disagreement": comparison,
            "production_independent_evaluators_are_distinct": True,
            "reference_geometry_pass": reference.geometry_pass,
            "reference_coverage": reference.coverage.value,
            "reference_safety_pass": reference.safety_pass,
        }

    payload_names = {name: f"programs/{role}/{name}" for name in _PAYLOAD}

    refinement, external_files = _production_refinement(
        role=role, link_path=refinement_link, final_for_protocol=final_for_protocol,
        run=run, task=task, motion=motion, primary_hashes=hashes, posted=posted_results,
    )
    for name in DIALECTS:
        evaluations[name]["production_refinement"] = dict(
            refinement,
            applies_to_dialect=name,
            refinement_dialect="rs274" if refinement["present"] else None,
            motion_relationship=(
                "RS274-only production geometry/safety recheck of the same typed motion/task/tool setup; Haas and RS274 raw NC bytes are not conflated and Haas has its own posted/reference evaluations"
                if refinement["present"] else None
            ),
        )
    details = {
        "role": role, "run_id": run.run_id, "task_id": task.task_id,
        "task_hash": task.task_hash, "tool_setup_hash": task.tool_setup_hash,
        "motion_hash": motion.motion_hash, "selected_dialect": selected_dialect,
        "source_commit": run.source_commit, "dirty_patch_hash": run.dirty_patch_hash,
        "source_record": f"programs/{role}/run.json",
        "task_record": f"programs/{role}/task.json",
        "motion_record": f"programs/{role}/motion_plan.json",
        "tooling": {"evaluated_assembly": _plain(task.tool),
                    "declared_stickout_mm": task.tool.stickout_mm},
        "work_offset": {"identifier": task.initial_state.work_offset,
                        "transform": _plain(task.work_to_machine),
                        "initial_machine_position_mm": list(task.initial_state.position_mm)},
        "fixture_geometry": _plain(task.fixtures),
        "stock": _plain(task.stock), "original_part": _plain(task.original_part),
        "operation_target": _plain(task.operation_target), "tolerances": _plain(task.tolerances),
        "motion_mode": motion.timing_rule.value,
        "process_record": f"programs/{role}/config.json",
        "material_process_setup_record": "setup.json",
        "source_fields": source_verification,
        "evaluations": evaluations,
        "independent_cross_checks": cross_checks,
        "simulation_qualification": "simulation-qualified",
        "spatial_temporal_discretization_uncertainty_retained": True,
        "continuous_collision_free_certificate": False,
    }
    return task, payload_names, details, external_files


def _verify_refinement_comparison(*, role: str, task: C.TaskSpec,
                                  primary: C.EvaluationResult, refined: C.EvaluationResult,
                                  actual: Mapping[str, Any], outcome: Mapping[str, Any]) -> None:
    """Recompute schema-3 rows; hashes alone cannot prove a verdict's meaning."""
    expected: dict[str, Any] = {}
    for name in ("safety_pass", "geometry_pass"):
        left, right = getattr(primary, name), getattr(refined, name)
        expected[name] = {"primary": left, "refined": right, "changed": left != right}
    expected["coverage"] = {
        "primary": primary.coverage.value, "refined": refined.coverage.value,
        "changed": primary.coverage is not refined.coverage,
    }
    expected["efficiency_eligible"] = {
        "primary": primary.efficiency_eligible, "refined": refined.efficiency_eligible,
        "changed": primary.efficiency_eligible != refined.efficiency_eligible,
    }
    for name in ("residual_volume_mm3", "gouge_volume_mm3", "max_surface_error_mm"):
        left, right = primary.metrics[name].value, refined.metrics[name].value
        unit = "mm" if name.endswith("_mm") else "mm^3"
        _require(left is not None and right is not None,
                 f"{role} refinement metric {name} must remain explicit")
        left = _number(left, f"{role}.primary.{name}")
        right = _number(right, f"{role}.refined.{name}")
        expected[name] = {
            "primary": left, "refined": right, "delta": right - left, "unit": unit,
        }

    def completion(result: C.EvaluationResult) -> bool | None:
        if result.coverage is not C.Coverage.COMPLETE:
            return None
        removed = result.metrics.get("removed_volume_mm3")
        residual = result.metrics.get("residual_volume_mm3")
        if removed is None or residual is None or removed.value is None or residual.value is None:
            return None
        if removed.value <= 0.0:
            return False
        return residual.value <= task.tolerances.completion_residual_mm3

    left, right = completion(primary), completion(refined)
    expected["completion_pass"] = {"primary": left, "refined": right, "changed": left != right}
    verdict_names = ("safety_pass", "geometry_pass", "completion_pass", "coverage", "efficiency_eligible")
    expected["any_outcome_verdict_change"] = any(expected[name]["changed"] for name in verdict_names)
    _require(set(actual) == set(expected), f"{role} refinement comparison rows do not match schema 3")
    for name, row in expected.items():
        _require(actual.get(name) == row,
                 f"{role} refinement comparison row {name} differs from typed primary/refined evidence")

    def combined(left, right):
        if left is False or right is False:
            return False
        if left is None or right is None:
            return None
        return True

    disagree = [name for name in ("safety_pass", "geometry_pass", "completion_pass", "coverage")
                if expected[name]["changed"]]
    complete = combined(left, right)
    eligible = bool(primary.efficiency_eligible and refined.efficiency_eligible
                    and complete is True and not disagree)
    expected_outcome = {
        "safety_pass": combined(primary.safety_pass, refined.safety_pass),
        "geometry_pass": combined(primary.geometry_pass, refined.geometry_pass),
        "completion_pass": complete,
        "efficiency_eligible": eligible,
        "refinement_disagreement": bool(disagree),
        "disagreement_fields": disagree,
        "excluded_from_efficiency": not eligible,
    }
    for name, value in expected_outcome.items():
        _require(outcome.get(name) == value,
                 f"{role} conservative outcome {name} differs from recomputed typed evidence")


def _production_refinement(*, role: str, link_path: str | Path | None,
                           final_for_protocol: bool, run: C.RunRecord, task: C.TaskSpec,
                           motion: C.MotionPlan, primary_hashes: Mapping[str, str],
                           posted: Mapping[str, C.EvaluationResult]) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Validate the pilot runner's auxiliary ``diffcam-refinement-link/3``.

    The refinement is RS274-only. Haas packets cite its identical typed motion,
    task/tool identity and their own Haas evaluations; NC bytes are not compared
    across dialects. The optional dev fixture has genuinely no such link.
    """
    if link_path is None:
        _require(not final_for_protocol,
                 f"final packet requires qualified diffcam-refinement-link/3 production refinement for {role}")
        return {
            "present": False, "link": None, "link_sha256": None,
            "link_schema_version": None, "record": None, "qualification": None,
            "reason": "no stored production refinement link in this retained development nominal; independent reference rechecks are separate and final frozen-protocol packets require this slot",
        }, {}
    link_file = Path(link_path).expanduser()
    link_bytes = _read_bytes(link_file)
    link = _mapping(_load(link_file), f"{role}.refinement_link")
    _require(link.get("schema") == "diffcam-refinement-link/3",
             f"{role} refinement link must use diffcam-refinement-link/3")
    _require(link.get("primary_run_id") == run.run_id,
             f"{role} refinement link primary_run_id differs from selected run")
    primary_path = _text(link, "primary_run_path", f"{role}.refinement_link")
    _safe_path(link_file.parent, primary_path)  # relative metadata, never trusted absolute/traversal input.
    _require(PurePosixPath(primary_path).name == run.run_id,
             f"{role} refinement link primary path does not name the selected primary run")
    _require(link.get("refined_run_id") == f"{run.run_id}-refined",
             f"{role} refined run ID does not name the selected primary's distinct auxiliary refinement")
    for field, filename in (
        ("primary_metrics_sha256", "metrics.json"),
        ("primary_rs274_evaluation_sha256", "eval_posted_rs274.json"),
        ("primary_haas_evaluation_sha256", "eval_posted_haas.json"),
        ("primary_run_json_sha256", "run.json"),
    ):
        _require(primary_hashes.get(filename) == link.get(field),
                 f"{role} refinement link {field} differs from immutable primary artifact")
    root = link_file.parent
    refined_relative = _text(link, "refined_artifact_path", f"{role}.refinement_link")
    _require(PurePosixPath(refined_relative).name == "eval_posted_rs274_refined.json",
             f"{role} refinement must point to the RS274 refined production record")
    _require(PurePosixPath(refined_relative).parts[0] == link.get("refined_run_id"),
             f"{role} refined evaluation must be in its distinct auxiliary run directory")
    refined_path = _safe_path(root, refined_relative)
    refined_bytes = _read_bytes(refined_path)
    _require(_digest(refined_bytes) == _hash_field(link.get("refined_evaluation_sha256"),
                                                    f"{role}.refined_evaluation_sha256"),
             f"{role} refinement evaluation hash differs from link")
    refined = _typed(refined_path, C.EvaluationResult)
    _require(refined.dialect is C.Dialect.RS274 and
             refined.nc_hash == posted["rs274"].nc_hash and
             refined.task_hash == task.task_hash and
             refined.tool_setup_hash == task.tool_setup_hash,
             f"{role} refined RS274 motion/task/setup/NC identity differs from primary")
    parsed_hash = posted["rs274"].discretization.get("parsed_motion_hash")
    _hash_field(parsed_hash, f"{role}.primary_rs274.parsed_motion_hash")
    _require(refined.discretization.get("parsed_motion_hash") == parsed_hash and
             refined.motion_hash in (None, parsed_hash),
             f"{role} refined and primary RS274 parsed-motion hashes differ")
    _require(refined.evaluator_id == "posted_hard/production" and
             refined.coverage is C.Coverage.COMPLETE,
             f"{role} production refinement is not complete production evaluation")
    settings = _mapping(refined.discretization.get("settings"), f"{role}.refined.settings")
    primary_settings = _mapping(posted["rs274"].discretization.get("settings"),
                                f"{role}.primary.settings")
    _require(set(settings) == set(primary_settings),
             f"{role} refinement must preserve every primary sampling-control name")
    for name, primary_value in primary_settings.items():
        if name != "max_step_mm":
            _require(settings.get(name) == primary_value,
                     f"{role} refinement changed sampling control {name} beyond max_step_mm")
    refined_step = _number(settings.get("max_step_mm"), f"{role}.refined.max_step_mm", positive=True)
    primary_step = _number(primary_settings.get("max_step_mm"), f"{role}.primary.max_step_mm", positive=True)
    _require(refined_step < primary_step,
             f"{role} refinement max_step_mm must be strictly smaller than primary")
    _require(settings.get("expected_nc_hash") == refined.nc_hash and
             settings.get("compute_bounds") is True and settings.get("safety_screen") is True,
             f"{role} refined production bounds/NC identity were not evaluated")
    _require(settings.get("exclusion_tolerance_mm3") ==
             task.operation_target.geometry.params.get("grid_exclusion_tolerance_mm3"),
             f"{role} refined production silently changed excluded-volume tolerance")
    qualification = _checked_clearance(
        refined, task, f"{role}/production_refinement", geometry=True,
    )
    outcome = _mapping(link.get("conservative_outcome"), f"{role}.conservative_outcome")
    for key in ("safety_pass", "geometry_pass", "completion_pass"):
        _require(outcome.get(key) is True, f"{role} refinement conservative outcome {key} is not true")
    disagreement = outcome.get("refinement_disagreement")
    _require(isinstance(disagreement, bool),
             f"{role} refinement_disagreement must be an explicit boolean")
    fields = outcome.get("disagreement_fields")
    _require(isinstance(fields, list) and
             all(name in ("safety_pass", "geometry_pass", "completion_pass", "coverage") for name in fields) and
             len(fields) == len(set(fields)) and disagreement == bool(fields),
             f"{role} refinement disagreement fields/verdict must agree")
    eligible = outcome.get("efficiency_eligible")
    _require(isinstance(eligible, bool) and
             outcome.get("excluded_from_efficiency") is (not eligible),
             f"{role} refinement eligibility/exclusion must agree")
    if final_for_protocol:
        _require(not disagreement and eligible is True,
                 f"{role} final packet cannot use a production refinement with conservative disagreement or efficiency exclusion")
    _text(outcome, "rule", f"{role}.conservative_outcome")

    auxiliary = _mapping(link.get("auxiliary_record"), f"{role}.auxiliary_record")
    auxiliary_relative = _text(auxiliary, "path", f"{role}.auxiliary_record")
    auxiliary_relative_path = PurePosixPath(auxiliary_relative)
    _require(auxiliary_relative_path.parent == PurePosixPath(link["refined_run_id"]) and
             auxiliary_relative_path.name == "run.json",
             f"{role} auxiliary_record must be the refined run's run.json")
    auxiliary_path = _safe_path(root, auxiliary_relative)
    auxiliary_bytes = _read_bytes(auxiliary_path)
    _require(_digest(auxiliary_bytes) == _hash_field(auxiliary.get("sha256"),
                                                      f"{role}.auxiliary_record.sha256"),
             f"{role} auxiliary record hash differs from refinement link")
    comparison_relative = _text(link, "comparison_path", f"{role}.refinement_link")
    comparison_relative_path = PurePosixPath(comparison_relative)
    _require(comparison_relative_path.parent == PurePosixPath(link["refined_run_id"]) and
             comparison_relative_path.name == "comparison.json",
             f"{role} comparison must be in the distinct refined run directory")
    comparison_bytes = _read_bytes(_safe_path(root, comparison_relative))
    _require(_digest(comparison_bytes) == _hash_field(link.get("comparison_sha256"),
                                                       f"{role}.comparison_sha256"),
             f"{role} comparison artifact hash differs from link")
    comparison_path = _safe_path(root, comparison_relative)
    comparison = _mapping(_load(comparison_path), f"{role}.comparison")
    _require(comparison.get("comparison") == link.get("comparison") and
             comparison.get("conservative_outcome") == outcome and
             comparison.get("primary_run_id") == run.run_id and
             comparison.get("refined_run_id") == link.get("refined_run_id"),
             f"{role} comparison wrapper differs from refinement-link primary/refined/comparison identities")
    _verify_refinement_comparison(
        role=role, task=task, primary=posted["rs274"], refined=refined,
        actual=_mapping(link.get("comparison"), f"{role}.comparison.payload"), outcome=outcome,
    )

    prefix = f"programs/{role}/production_refinement"
    external = {
        f"{prefix}/refinement-link.json": link_bytes,
        f"{prefix}/eval_posted_rs274_refined.json": refined_bytes,
        f"{prefix}/auxiliary-record.json": auxiliary_bytes,
        f"{prefix}/comparison.json": comparison_bytes,
    }
    slot = {
        "present": True,
        "link": f"{prefix}/refinement-link.json", "link_sha256": _digest(link_bytes),
        "link_schema_version": "diffcam-refinement-link/3",
        "primary_run_id": run.run_id, "refined_run_id": _text(link, "refined_run_id", f"{role}.refinement_link"),
        "primary_run_path": primary_path,
        "record": f"{prefix}/eval_posted_rs274_refined.json",
        "record_sha256": _digest(refined_bytes),
        "auxiliary_record": f"{prefix}/auxiliary-record.json",
        "comparison": f"{prefix}/comparison.json",
        "evaluator_id": refined.evaluator_id, "evaluator_revision": refined.evaluator_revision,
        "coverage": refined.coverage.value, "safety_pass": refined.safety_pass,
        "geometry_pass": refined.geometry_pass, "qualification": qualification,
        "conservative_outcome": _plain(outcome),
        "settings": _plain(settings), "discretization": _plain(refined.discretization),
        "force_proxy": _plain(refined.metrics.get("force_proxy")),
    }
    return slot, external


def _write_new(root: Path, name: str, data: bytes) -> None:
    path = _safe_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def verify_packet(root: Path) -> dict[str, Any]:
    """Re-read a packet and verify its manifest/payload byte hashes."""
    manifest = _mapping(_load(_safe_path(root, "packet.json")), "packet")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "packet schema mismatch")
    _require(manifest.get("status") == "pending setup review",
             "unreviewed physical packets must remain pending setup review")
    files = _mapping(manifest.get("payload_sha256"), "packet.payload_sha256")
    _require(files, "packet payload hash map is empty")
    _require(_load(root / "hashes.json") == files,
             "packet hashes.json differs from packet.json payload hashes")
    for name, expected in files.items():
        path = _safe_path(root, name)
        _require(_file_digest(path) == _hash_field(expected, f"payload[{name}]"),
                 f"packet payload hash mismatch: {name}")
    return manifest


def export_packet(*, selections: Mapping[str, str | Path], setup: Mapping[str, Any],
                  protocol: Mapping[str, Any], selected_dialect: str,
                  output: str | Path, final_for_protocol: bool = False,
                  refinement_links: Mapping[str, str | Path] | None = None) -> dict[str, Any]:
    """Build all packet bytes before creating a new output directory."""
    _require(selected_dialect in DIALECTS, f"selected dialect must be one of {DIALECTS}")
    _require(isinstance(selections, Mapping) and selections, "at least one selected program is required")
    _require(set(selections).issubset(ROLES) and len(selections) == len(set(selections)),
             "selection roles must be unique baseline/optimized entries")
    refinement_links = {} if refinement_links is None else dict(refinement_links)
    _require(set(refinement_links).issubset(selections),
             "a refinement link must name a selected baseline/optimized role")
    protocol_record = _validate_protocol(protocol, final_for_protocol)
    if final_for_protocol:
        _require(all(role in refinement_links for role in selections),
                 "every final selected program requires a qualified schema-3 refinement link")
    root = Path(output)
    _require(not root.exists(), f"packet output already exists; use a fresh directory: {root}")
    files: dict[str, bytes] = {"setup.json": _json_bytes(setup),
                               "protocol.json": _json_bytes(protocol_record)}
    programs = []
    identity = None
    for role in ROLES:
        if role not in selections:
            continue
        source = Path(selections[role]).expanduser()
        task, paths, details, external_files = _selection(
            source, role, selected_dialect, setup, final_for_protocol,
            refinement_links.get(role),
        )
        current_identity = (task.task_hash, task.tool_setup_hash)
        _require(identity is None or current_identity == identity,
                 "baseline/optimized selections must belong to the same task and physical setup")
        identity = current_identity
        for original, destination in paths.items():
            files[destination] = _safe_path(source, original).read_bytes()
        files.update(external_files)
        programs.append(details)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pending setup review",
        "simulation_scope": "qualified only for the retained task/evaluator field and motion bounds",
        "final_protocol_packet": final_for_protocol,
        "protocol": _plain(protocol_record),
        "machine": _mapping(setup, "setup").get("machine"),
        "material_process": {"material": _mapping(setup, "setup").get("material"),
                             "process": _mapping(setup, "setup").get("process")},
        "manual_setup_verification": [{
            "item": item, "required": True, "operator_record": None,
            "state": "pending setup review",
        } for item in MANUAL_SETUP],
        "equipment_execution_authorized": False,
        "physical_trial_performed": False,
        "clearance_semantics": "lower bounds must meet declared task clearance; rapid penetration upper bound must not exceed declared tolerance",
        "force_quantities": "proxies only unless a separate independently calibrated physical measurement record is supplied",
        "continuous_certificate": False,
        "programs": programs,
        "payload_sha256": {name: _digest(data) for name, data in sorted(files.items())},
    }
    files["hashes.json"] = _json_bytes(manifest["payload_sha256"])
    root.mkdir(parents=True, exist_ok=False)
    for name, data in sorted(files.items()):
        _write_new(root, name, data)
    _write_new(root, "packet.json", _json_bytes(manifest))
    return verify_packet(root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", action="append", required=True, metavar="ROLE=RUN_DIR",
                        help="baseline or optimized completed immutable run directory; may repeat")
    parser.add_argument("--dialect", choices=DIALECTS, required=True,
                        help="selected machine-program dialect; both evaluated records are retained")
    parser.add_argument("--refinement-link", action="append", default=[], metavar="ROLE=LINK_JSON",
                        help="auxiliary diffcam-refinement-link/3 for the selected role; required for final packets")
    parser.add_argument("--setup", type=Path, required=True, help="explicit machine/tool/work/setup JSON")
    parser.add_argument("--protocol", type=Path, required=True, help="pending or hash-pinned frozen protocol JSON")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing files are never mixed")
    parser.add_argument("--final-for-protocol", action="store_true",
                        help="only valid with an explicitly hash-pinned frozen T4.2 record")
    args = parser.parse_args(argv)
    selected = {}
    refinements = {}
    try:
        for entry in args.refinement_link:
            role, separator, path = entry.partition("=")
            if not separator or role not in ROLES or not path or role in refinements:
                raise PacketError(f"invalid/repeated --refinement-link {entry!r}")
            refinements[role] = path
        for entry in args.selection:
            role, separator, path = entry.partition("=")
            if not separator or role not in ROLES or not path or role in selected:
                raise PacketError(f"invalid/repeated --selection {entry!r}; expected baseline=PATH or optimized=PATH")
            selected[role] = path
        result = export_packet(
            selections=selected, setup=_load(args.setup), protocol=_load(args.protocol),
            selected_dialect=args.dialect, output=args.output,
            final_for_protocol=args.final_for_protocol, refinement_links=refinements,
        )
    except PacketError as error:
        parser.exit(2, f"packet rejected: {error}\n")
    print(json.dumps({
        "status": result["status"], "packet": str(args.output / "packet.json"),
        "programs": [item["run_id"] for item in result["programs"]],
        "equipment_execution_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
