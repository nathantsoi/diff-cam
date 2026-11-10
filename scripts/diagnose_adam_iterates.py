#!/usr/bin/env python3
"""Read-only D-15 diagnostic for six persisted annealed-Adam hard validations.

Run ONE task per process: the runner factory owns this process's sole Taichi
runtime/simulator. THIS DOES NOT RE-RUN OR MODIFY AN OPTIMIZER OR A CAMPAIGN.
Only the CLI --output path is written. Refuse output beneath the campaign root.
Continuations' rejected proposal theta is not persisted in their history; do
not mislabel their restored base theta as a proposed iterate (logging gap).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import run_artifacts as R
from benchmarks import contracts as C
from scripts import run_benchmark_suite as runner

METHOD = "annealed-adam"


class DiagnosticError(RuntimeError):
    """The persisted record does not identify a supported exact replay."""


def _mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes(), parse_constant=lambda text: (
            _ for _ in ()).throw(ValueError(f"nonfinite {text}")))
    except (OSError, UnicodeError, ValueError) as error:
        raise DiagnosticError(f"cannot read {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise DiagnosticError(f"{path} must contain a JSON object")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DiagnosticError(f"{name} must be a finite number")
    return float(value)


def _validation_points(history: Mapping[str, Any], run_id: str,
                       raw_config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """The post-step state is what the ordinary Adam loop validates.

    No checkpoint is needed for these ordinary hard validations. Validate1 is
    theta_after_step at history iteration4, not that row's pre-step theta.
    """
    interval = raw_config.get("validation_interval")
    maximum = raw_config.get("max_validation_calls")
    if (isinstance(interval, bool) or not isinstance(interval, int) or interval < 1
            or isinstance(maximum, bool) or not isinstance(maximum, int) or maximum != 6):
        raise DiagnosticError("requires integer validation_interval and exactly six validation calls")
    rows = history.get("history")
    if not isinstance(rows, list) or not rows:
        raise DiagnosticError("history has no persisted Adam iterates")
    points = []
    try:
        sequence = [row["iteration"] for row in rows]
    except (KeyError, TypeError) as error:
        raise DiagnosticError("history does not contain an explicit iteration sequence") from error
    if (not sequence or isinstance(sequence[0], bool) or not isinstance(sequence[0], int)
            or sequence[0] < 1 or sequence != list(range(sequence[0], sequence[0] + len(sequence)))):
        raise DiagnosticError("history.iteration is not consecutive")
    for number in range(1, maximum + 1):
        iteration = number * interval
        matches = [row for row in rows if isinstance(row, Mapping)
                   and row.get("iteration") == iteration]
        if len(matches) != 1:
            raise DiagnosticError(f"validation {number}: expected one history iteration {iteration}")
        row = matches[0]
        if row.get("valid") is not True or iteration > raw_config["candidate_evaluations"] + 1:
            raise DiagnosticError(f"validation {number}: invalid or beyond-budget history iteration")
        try:
            after = np.asarray(row["theta_after_step"], dtype=np.float64)
            before = np.asarray(row["theta"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise DiagnosticError(f"validation {number}: invalid persisted theta: {error}") from error
        if (after.ndim != 1 or before.shape != after.shape or after.size == 0
                or not np.isfinite(after).all() or not np.isfinite(before).all()):
            raise DiagnosticError(f"validation {number}: nonfinite or inconsistent theta")
        points.append({"validation": number, "iteration": iteration,
                       "original_candidate_id": f"{run_id}:validate:{number}",
                       "theta": after, "pre_step_theta": before,
                       "k": _finite(row.get("k"), "k"),
                       "step_l2": float(np.linalg.norm(after - before))})
    origin = np.asarray(rows[0].get("theta"), dtype=np.float64)
    if origin.shape != points[0]["theta"].shape or not np.array_equal(origin, np.zeros_like(origin)):
        raise DiagnosticError("CAM Adam initialization is not the persisted zero theta")
    return tuple(points)


def _context(run_directory: Path) -> dict[str, Any]:
    directory = run_directory.expanduser().resolve()
    task_dir = directory.parent
    campaign = task_dir.parent.parent
    if task_dir.parent.name != "tasks" or task_dir.name != _mapping(directory / "config.json").get("task_id"):
        raise DiagnosticError("run must be the canonical campaign tasks/<task_id>/<annealed-run-id> directory")
    config = _mapping(directory / "config.json")
    if config.get("run_id") != directory.name or not directory.name.startswith(METHOD + "-s0-"):
        raise DiagnosticError("run identity is not the deterministic annealed-Adam seed-0 case")
    record = C.loads((directory / "run.json").read_bytes().decode("utf-8"))
    if not isinstance(record, C.RunRecord) or record.status is not C.RunStatus.COMPLETE:
        raise DiagnosticError("only a completed typed run record can be replayed")
    if record.run_id != directory.name or record.task_hash != config.get("task_hash"):
        raise DiagnosticError("typed run identity/task hash disagrees with config")
    frozen = _mapping(task_dir / "frozen_config.json")
    raw = frozen.get("raw_config")
    if not isinstance(raw, Mapping) or raw.get("task_id") != config.get("task_id"):
        raise DiagnosticError("task's frozen raw_config does not match this task")
    if (raw.get("precision") != "f64" or raw.get("arch") != "cpu"
            or raw.get("task_source") != "suite_manifest"
            or config.get("simulator", {}).get("runtime_default_fp") != "f64"
            or config.get("initialization", {}).get("init") != "cam"):
        raise DiagnosticError("requires registered-task CAM f64 CPU campaign evidence")
    campaign_manifest = _mapping(campaign / "manifest.json")
    source = campaign_manifest.get("campaign", {}).get("source_metadata", {})
    expected_config_hash = C.sha256_bytes((ROOT / "configs" / "campaign_suite.json").read_bytes())
    if (campaign_manifest.get("campaign", {}).get("config_sha256") != expected_config_hash
            or source.get("source_commit") != config.get("source_commit")
            or source.get("dirty_patch_hash") is not None or config.get("dirty_patch_hash") is not None):
        raise DiagnosticError("campaign source/config identity differs from the reviewed inputs")
    return {"directory": directory, "task_dir": task_dir, "campaign": campaign,
            "config": config, "raw_config": raw, "source": source,
            "history_path": directory / "history.json", "metrics_path": directory / "metrics.json"}


def _hard_components(evaluator: Any, theta: np.ndarray, candidate_id: str) -> dict[str, Any]:
    result = evaluator.evaluate_candidate(theta, candidate_id, "hard")
    row: dict[str, Any] = {"valid": result.valid, "theta_hash": result.theta_hash,
                           "objective": result.objective, "invalid_reason": result.invalid_reason}
    if not result.valid:
        row.update({name: None for name in ("gouge_fraction", "gouge_voxels", "residual_fraction",
                                             "holder_overlap_fraction", "cycle_time_s", "air_fraction",
                                             "feasible")})
        return row
    components = result.components
    stock = np.asarray(evaluator.sim.stock.to_numpy())
    steps = evaluator.parameterization.num_steps
    initial = stock[0] < 0
    final = stock[steps] < 0
    target = np.asarray(evaluator.sim.target.to_numpy()) < 0
    in_target = int(np.count_nonzero(initial & target))
    gouge_voxels = int(np.count_nonzero(initial & target & ~final))
    if in_target <= 0 or not math.isclose(gouge_voxels / in_target,
                                          components["gouge_fraction"], rel_tol=1e-12, abs_tol=1e-15):
        raise DiagnosticError("raw hard gouge count disagrees with CandidateEvaluator fraction")
    row.update({
        "gouge_fraction": components["gouge_fraction"], "gouge_voxels": gouge_voxels,
        "in_target_voxels": in_target,
        "residual_fraction": components["residual_fraction"],
        "holder_overlap_fraction": components["holder_overlap_fraction"],
        "cycle_time_s": components["cycle_time_s"], "air_fraction": components["air_fraction"],
        "feasible": bool(components["feasible"]), "objective": result.objective,
    })
    return row


def diagnose(run_directory: Path) -> dict[str, Any]:
    context = _context(run_directory)
    history = _mapping(context["history_path"])
    points = _validation_points(history, context["config"]["run_id"], context["raw_config"])
    task, motion, parameterization, reference = runner._suite_task_binding(context["raw_config"])
    if (task.task_hash != context["config"]["task_hash"]
            or reference["reference_cycle_time_s"] != context["config"]["reference_cycle_time_s"]
            or parameterization.num_params != points[0]["theta"].size):
        raise DiagnosticError("registered suite task/motion/reference/parameterization differs from the run")
    # A fresh subprocess per CLI invocation gives this factory its sole runtime.
    evaluator = runner._pilot_evaluator(
        task, motion, parameterization, context["raw_config"], context["config"]["seed"],
        repository=ROOT, reference_metadata=reference)
    initial_theta = np.zeros_like(points[0]["theta"])
    # The selected θ0 plan carries stateless-candidate provenance; the nominal
    # CAM reference is a distinct, separately checked identity, not this hash.
    if evaluator.candidate_motion(initial_theta).motion_hash != context["config"]["motion_hash"]:
        raise DiagnosticError("reconstructed theta0 candidate-motion hash differs from the run")
    initial = _hard_components(evaluator, initial_theta, context["config"]["run_id"] + ":diagnostic:init")
    stored_initial = _mapping(context["metrics_path"])["selection"]["initial_candidate"]
    for name, stored_name in (
            ("gouge_fraction", "gouge_fraction"), ("residual_fraction", "residual_fraction"),
            ("holder_overlap_fraction", "holder_overlap_fraction"),
            ("cycle_time_s", "cycle_time_s"), ("air_fraction", "air_fraction"),
            ("objective", "objective_value")):
        if not math.isclose(initial[name], stored_initial["components"][stored_name], rel_tol=1e-9, abs_tol=1e-12):
            raise DiagnosticError(f"initial theta re-evaluation differs from stored {name}")
    if initial["theta_hash"] != stored_initial["theta_hash"]:
        raise DiagnosticError("initial theta hash differs from the retained hard initialization")
    learning_rate = _finite(context["raw_config"].get("learning_rate"), "learning_rate")
    first_row = history["history"][0]
    first_theta = np.asarray(first_row["theta_after_step"], dtype=np.float64)
    first_delta = first_theta - initial_theta
    first_update = _hard_components(evaluator, first_theta, context["config"]["run_id"] + ":diagnostic:first_update")
    first_update.update({
        "iteration": first_row["iteration"], "learning_rate": learning_rate,
        "delta_theta_l2": float(np.linalg.norm(first_delta)),
        "delta_theta_l2_over_lr": float(np.linalg.norm(first_delta)) / learning_rate,
        "delta_theta_max_abs": float(np.max(np.abs(first_delta))),
        "theta": first_theta.tolist(),
    })
    rows = []
    for point in points:
        delta = point["theta"] - initial_theta
        row = _hard_components(evaluator, point["theta"], point["original_candidate_id"] + ":diagnostic")
        row.update({"validation": point["validation"], "iteration": point["iteration"],
                    "original_candidate_id": point["original_candidate_id"],
                    "delta_theta_l2": float(np.linalg.norm(delta)),
                    "delta_theta_max_abs": float(np.max(np.abs(delta))),
                    "step_l2": point["step_l2"], "k": point["k"],
                    "theta": point["theta"].tolist()})
        rows.append(row)
    scale = float(parameterization.scale_mm)
    return {
        "schema": "diffcam-adam-iterate-diagnostic/1", "method": METHOD,
        "run_id": context["config"]["run_id"], "task_id": task.task_id, "task_hash": task.task_hash,
        "source_commit": context["config"]["source_commit"],
        "input_sha256": {name: C.sha256_bytes(path.read_bytes()) for name, path in (
            ("config", context["directory"] / "config.json"),
            ("history", context["history_path"]), ("metrics", context["metrics_path"]),
            ("run_record", context["directory"] / "run.json"))},
        "evaluator_source_sha256": {
            "objectives": C.sha256_bytes((ROOT / "algorithms" / "objectives.py").read_bytes()),
            "runner": C.sha256_bytes((ROOT / "scripts" / "run_benchmark_suite.py").read_bytes()),
            "parameterization": C.sha256_bytes((ROOT / "benchmarks" / "pilot_task.py").read_bytes())},
        "precision": "f64", "arch": "cpu", "objective_kind": "hard",
        "theta_units": f"DeltaSpline normalized controls; scale_mm={scale}; L2 is normalized control Euclidean norm",
        "learning_rate": learning_rate,
        "initial_theta_reference": initial, "iterates": rows,
        "first_history_update": first_update,
        "continuation_logging_gap": (
            "Not evaluated here: rejected continuation proposals are not persisted as theta_after_step. "
            "Their restored base theta is not a proposed iterate; raw validation ledgers have no hard components."),
        "notes": [
            "Raw gouge count uses the same hard evaluator masks as gouge_fraction: initial & target & ~final stock.",
            "Hard timing is exact candidate-motion duration, not the native smooth timing surrogate.",
            "This is a read-only, two-task D-15 diagnostic, not a feasible-direction proof or campaign/geometry verdict.",
            "No posted/physical execution was performed. Output is allowed only outside the campaign root.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="canonical completed annealed-Adam run directory")
    parser.add_argument("--output", type=Path, required=True, help="new JSON report outside campaign root")
    args = parser.parse_args(argv)
    try:
        context = _context(args.run_directory)
        destination = args.output.expanduser().resolve()
        if destination.is_relative_to(context["campaign"]):
            raise DiagnosticError("refusing to write inside the campaign output")
        if destination.exists():
            raise DiagnosticError(f"refusing to overwrite an existing report: {destination}")
        result = diagnose(args.run_directory)
        destination.parent.mkdir(parents=True, exist_ok=True)
        R.ArtifactWriter._replace_bytes(destination, R.json_bytes(result))
    except (DiagnosticError, KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(2, f"diagnostic BLOCKED: {error}\n")
    print(f"wrote {destination}: six validation points, first Adam step and theta0; provisional diagnostic only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
