"""D-13/BB1 Exp1 initialization reach analysis; construction and arithmetic only.

This does not run an optimizer, simulator, posted evaluator or physical controller.
R is the prescribed coordinate-distance ratio, not a proof of trajectory
reachability or an initializer-quality ranking.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import run_artifacts as R
from benchmarks import contracts as C
from scripts import run_benchmark_suite as runner
from scripts import run_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
INITIALIZERS = (("straight_line", 0), ("random", 0), ("random", 1))
METHODS = ("annealed-adam", "continuation-adam")
RESULT_JSON = "exp1_reach.json"
RESULT_TABLE = "exp1_reach_table.md"


@dataclass(frozen=True)
class SolverReach:
    method: str
    updates: int
    lr_max: float


def _solver_settings(config) -> tuple[SolverReach, ...]:
    """Resolve the declared planning horizon, never from execution timing."""
    raw_candidates = config["candidate_evaluations"]
    if isinstance(raw_candidates, bool) or not isinstance(raw_candidates, int) or raw_candidates < 1:
        raise ValueError("candidate_evaluations must be a positive integer")
    settings = []
    for method in METHODS:
        budget = runner._budget(config, method)
        # Identical to the runner's non-CMA update-count estimate. The budget's
        # sole ordinary hard initial evaluation is not an Adam update.
        remaining = budget.candidate_evaluations - 1
        if remaining != raw_candidates:
            raise ValueError("unexpected initialization accounting in the Exp1 budget")
        updates = remaining if budget.max_generations is None else min(
            remaining, budget.max_generations)
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
            raise ValueError("Exp1 reach requires a positive Adam update count")
        if method == "continuation-adam":
            base = runner.adam.AdamConfig(
                "annealed", learning_rate=config["learning_rate"],
                max_gradient_norm=config.get("max_gradient_norm", 0.5),
                k_start=config["k_anneal_start"], k_end=config["k_anneal_end"],
            )
            continued = runner._make_continuation_config(config, base)
            if continued.adam.n_steps != updates:
                raise ValueError("continuation update count differs from its typed configuration")
            lr_max = config["continuation_adam"]["backtracking_learning_rate_max"]
        else:
            lr_max = config["learning_rate"]
        if isinstance(lr_max, bool) or not isinstance(lr_max, (int, float)):
            raise ValueError("lr_max must be numeric")
        lr_max = float(lr_max)
        if not math.isfinite(lr_max) or lr_max <= 0:
            raise ValueError("lr_max must be finite and positive")
        settings.append(SolverReach(method, updates, lr_max))
    # Guard against silently applying BB1's table to a different recipe.
    if [(setting.method, setting.updates, setting.lr_max) for setting in settings] != [
            ("annealed-adam", 24, 0.005), ("continuation-adam", 24, 0.01)]:
        raise ValueError("this BB1 table requires 24 updates and lr_max 0.005/0.01")
    return tuple(settings)


def _theta(metadata) -> np.ndarray:
    """Copy the exact float64 initial vector emitted by the shared runner path."""
    if metadata.get("scale_mm") != 1.0 or metadata.get("bounds") is not None:
        raise ValueError("Exp1 reach requires scale 1 mm and null bounds")
    theta = np.array(metadata["initial_theta"], dtype=np.float64, copy=True)
    if theta.ndim != 1 or not theta.size or theta.size % 3 or not np.all(np.isfinite(theta)):
        raise ValueError("initial theta must be a nonempty finite 3T vector")
    if metadata.get("initial_theta_sha256") != C.content_hash(theta.tolist()):
        raise ValueError("initializer metadata initial-theta hash mismatch")
    return theta


def _seed_theta(config, task, motion, cam_params, method, init, seed):
    motion, params, record = runner._initializer_motion(
        config, task, motion, cam_params,
        {"method": method, "init": init, "seed": seed, "scope": "optimizer"},
    )
    if motion is None or params is None:
        raise ValueError(
            f"{task.task_id}/{method}/{init}/seed={seed}: initializer failed at its "
            f"original slot: {record.get('invalid_reason', record.get('require_plan_failure'))}")
    metadata = params.experiment1_metadata
    steps = len(motion.segments)
    if (params.num_steps != steps or params.num_control_points != steps
            or params.num_params != steps * 3 or params.num_control_points != params.num_steps
            or params.scale_mm != 1.0 or params.bounds is not None):
        raise ValueError(f"{task.task_id}/{init}: initializer is not zero-base DeltaResidual K=T")
    expected_hash = C.content_hash(R.parameterization_metadata(params))
    if metadata["parameterization_sha256"] != expected_hash:
        raise ValueError(f"{task.task_id}/{init}: shared parameterization hash mismatch")
    return {
        "theta": _theta(metadata),
        "source_motion_hash": motion.motion_hash,
        "theta_sha256": metadata["initial_theta_sha256"],
        "parameterization_sha256": expected_hash,
    }


def _distance(cam, initial) -> dict[str, float | str]:
    """Absolute statistics over θ coordinates (millimetres at scale 1)."""
    difference = cam - initial
    if difference.shape != cam.shape or difference.ndim != 1:
        raise ValueError("CAM and initializer theta shapes must agree")
    magnitudes = np.abs(difference)
    return {
        "median_abs_delta_theta_mm": float(np.median(magnitudes)),
        "max_abs_delta_theta_mm": float(np.max(magnitudes)),
        "cam_minus_initializer_difference_sha256": C.content_hash(difference.tolist()),
    }


def _ratio(distance_mm: float, updates: int, lr_max: float) -> float:
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer")
    if not math.isfinite(distance_mm) or distance_mm < 0:
        raise ValueError("distance must be finite and nonnegative")
    denominator = updates * lr_max
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("reach denominator must be finite and positive")
    ratio = distance_mm / denominator
    if not math.isfinite(ratio):
        raise ValueError("reach ratio must be finite")
    return ratio


def _task_label(task_id: str) -> str:
    match = re.search(r"^procedural-(?:facing_asymmetric|pocket_island|variable_width_depth)-dev-case-(\d+)", task_id)
    if match is None:
        raise ValueError(f"unrecognized Exp1 dev task identifier: {task_id}")
    for prefix, family in (("face", "facing_asymmetric"), ("pocket", "pocket_island"),
                           ("var", "variable_width_depth")):
        if task_id.startswith("procedural-" + family + "-dev-case-"):
            return f"{prefix}-{match.group(1)}"
    raise ValueError(f"unrecognized Exp1 dev family: {task_id}")


def build_report(config_path: Path) -> dict:
    config_path = Path(config_path).resolve()
    loaded = campaign._load(config_path)
    if loaded["document"].get("experiment") != "experiment1" or loaded["source"] != "suite_manifest":
        raise ValueError("reach analysis requires the suite-manifest experiment1 campaign")
    if runner.REPOSITORY.resolve() != ROOT:
        raise RuntimeError("runner source is not from this analysis worktree")
    config = loaded["template"]
    settings = _solver_settings(config)
    ids = campaign._task_rows(loaded)
    if len(ids) != 9 or len(set(ids)) != 9:
        raise ValueError("Exp1 reach requires exactly nine distinct dev tasks")
    task_rows = []
    for task_id in ids:
        task, motion, cam_params, *_ = runner._context(dict(config, task_id=task_id), ROOT)
        if task.split != C.Split.DEV:
            raise ValueError(f"{task_id}: Exp1 reach requires the dev split")
        shared_hashes = set()
        camera = None
        per_method = {}
        for setting in settings:
            cam = _seed_theta(config, task, motion, cam_params,
                              setting.method, "experiment1_cam", 0)
            if camera is None:
                camera = cam
            elif (not np.array_equal(cam["theta"], camera["theta"])
                    or cam["source_motion_hash"] != camera["source_motion_hash"]):
                raise ValueError(f"{task_id}: CAM initializer differs between solvers")
            shared_hashes.add(cam["parameterization_sha256"])
            for init, seed in INITIALIZERS:
                initial = _seed_theta(config, task, motion, cam_params,
                                      setting.method, init, seed)
                shared_hashes.add(initial["parameterization_sha256"])
                stats = _distance(camera["theta"], initial["theta"])
                # With a 1 mm parameter scale, this is a componentwise MM distance.
                per_method[(setting.method, init, seed)] = {
                    "max_abs_delta_theta_mm": stats["max_abs_delta_theta_mm"],
                    "R": _ratio(stats["max_abs_delta_theta_mm"], setting.updates, setting.lr_max),
                }
                method_key = (init, seed)
                entry = per_method.setdefault(method_key, {
                    "init": init, "seed": seed,
                    "theta0_cam_sha256": camera["theta_sha256"],
                    "theta0_initializer_sha256": initial["theta_sha256"],
                    "cam_source_motion_hash": camera["source_motion_hash"],
                    "initializer_source_motion_hash": initial["source_motion_hash"],
                })
                if setting.method != "annealed-adam":
                    # Random construction is solver-independent and must not be
                    # re-drawn or retuned when the scalar R denominator changes.
                    if (entry["theta0_cam_sha256"] != camera["theta_sha256"]
                            or entry["theta0_initializer_sha256"] != initial["theta_sha256"]
                            or entry["initializer_source_motion_hash"] != initial["source_motion_hash"]):
                        raise ValueError(f"{task_id}/{init}: initializer differs between solvers")
                else:
                    entry.update(stats)
        if len(shared_hashes) != 1:
            raise ValueError(f"{task_id}: initializers do not share one parameterization")
        methods_by_initializer = []
        for init, seed in INITIALIZERS:
            entry = dict(per_method[(init, seed)])
            entry["reach_ratio_by_method"] = {
                method: per_method[(method, init, seed)]["R"] for method in METHODS
            }
            methods_by_initializer.append(entry)
        task_rows.append({
            "task_id": task_id, "task_hash": task.task_hash, "family": task.family,
            "label": _task_label(task_id), "n_segments": len(motion.segments),
            "parameterization_sha256": shared_hashes.pop(),
            "arms": methods_by_initializer,
        })
    result = {
        "schema": "diffcam-exp1-reach/1",
        "experiment": "experiment1",
        "review": "BB1",
        "task_count": len(task_rows), "arm_count": sum(len(row["arms"]) for row in task_rows),
        "source": {
            **R.source_metadata(ROOT),
            "config_path": str(config_path),
            "config_sha256": loaded["config_sha256"],
            "task_manifest_sha256": runner._SUITE_MANIFEST_SHA256,
            "analysis_script_path": str(Path(__file__).resolve()),
            "analysis_script_sha256": C.sha256_bytes(Path(__file__).read_bytes()),
            "initializer_constructor": "runner._initializer_motion for each solver/arm",
        },
        "solvers": {
            setting.method: {
                "adam_update_count": setting.updates,
                "lr_max_mm_per_update": setting.lr_max,
                "raw_candidate_evaluations": config["candidate_evaluations"],
                "ordinary_hard_initial_evaluations_before_updates": 1,
                "max_generations": config.get("max_generations"),
            } for setting in settings
        },
        "definitions": {
            "parameterization": "shared zero-base DeltaResidual K=T, scale_mm=1.0, bounds=null",
            "theta_units": "physical millimetres at the declared 1 mm scale",
            "statistics": "median and maximum of abs(theta0_experiment1_cam - theta0_init) over all 3T scalar θ coordinates; not per-vertex L2 distances",
            "reach_ratio": "max_abs_delta_theta_mm / (adam_update_count * lr_max_mm_per_update)",
            "r_dimensionless": True,
            "construction_only": True,
        },
        "interpretation": (
            "Arithmetic required by D-13 addendum/BB1. Where R >> 1, report "
            "'unreachable within budget/step size', not 'init quality matters'. "
            "The nominal update count is not an observed run count or a trust-region, "
            "reachability, feasibility, optimizer-performance or physical-safety certificate."
        ),
        "tasks": task_rows,
    }
    if result["arm_count"] != 27:
        raise ValueError("expected nine tasks times three non-CAM initializer slots")
    return result


def _table(document) -> str:
    lines = [
        "# Exp1 reach ratios (D-13 addendum / review BB1)", "",
        "`R = max abs(θ0_cam − θ0_init) / (n_steps · lr_max)`; θ coordinates are millimetres at scale 1, and R is dimensionless.",
        "Both solvers use the declared 24 updates; lr_max is 0.005 (annealed) and 0.01 (continuation).",
        "Medians/maxima are over scalar θ coordinates, not per-vertex Euclidean norms.",
        "R ≫ 1 is reported as unreachable within budget/step size, not as evidence that initializer quality matters.",
        "These are initializer-construction calculations, not executed optimizer, simulation, physical-safety or machining results.", "",
        "Task labels: `face`=facing_asymmetric, `pocket`=pocket_island, `var`=variable_width_depth; full IDs and hashes are in the JSON.", "",
        "| Task | Init | Seed | Median abs Δθ0 (mm) | Max abs Δθ0 (mm) | R annealed | R continuation |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in document["tasks"]:
        for arm in task["arms"]:
            lines.append(
                f"| {task['label']} | {arm['init']} | {arm['seed']} "
                f"| {arm['median_abs_delta_theta_mm']:.6f} "
                f"| {arm['max_abs_delta_theta_mm']:.6f} "
                f"| {arm['reach_ratio_by_method']['annealed-adam']:.6f} "
                f"| {arm['reach_ratio_by_method']['continuation-adam']:.6f} |"
            )
    lines.extend([
        "", f"Config SHA-256: `{document['source']['config_sha256']}`.",
        f"Analysis script SHA-256: `{document['source']['analysis_script_sha256']}`.",
        "No optimizer, simulator, posted evaluation or physical CNC operation was run.", "",
    ])
    return "\n".join(lines)


def write_outputs(document, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / RESULT_JSON
    table_path = output_dir / RESULT_TABLE
    for path in (json_path, table_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing reach artifact: {path}")
    # Exclusive creation avoids replacing previously retained evidence.
    with json_path.open("xb") as stream:
        stream.write(R.json_bytes(document))
    with table_path.open("x", encoding="utf-8") as stream:
        stream.write(_table(document))
    return json_path, table_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/campaign_exp1_dev.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_report(args.config)
    json_path, table_path = write_outputs(document, args.output_dir)
    print(json.dumps({
        "json_path": str(json_path), "table_path": str(table_path),
        "json_sha256": C.sha256_bytes(json_path.read_bytes()),
        "table_sha256": C.sha256_bytes(table_path.read_bytes()),
        "task_count": document["task_count"], "arm_count": document["arm_count"],
        "optimizer_or_simulator_launched": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
