#!/usr/bin/env python3
"""Evaluate completed direct-feedback children and prepare the next A/B pair.

This is the Phase 5/6 boundary: explanations use measured artifacts, and a
pair is declared eligible for Phase 7 only when every predeclared gate passes.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
TASK = REPO / "autoresearch" / "tasks" / "train_csg"
ITERATIONS = TASK / "direct_feedback"
EVENT_STORE = TASK / "direct_feedback_events.jsonl"
PAIR_STORE = TASK / "pairwise.json"
METRIC_KEYS = (
    "hard_dice", "gouge", "residual", "air_time", "total_time",
    "break_prob_any", "air_time_frac", "air_time_frac_late",
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _append_event(record: dict) -> None:
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with EVENT_STORE.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def _set_source_pair_status(pair_id: str, status: str) -> None:
    lock_path = PAIR_STORE.with_suffix(".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pairs = json.loads(PAIR_STORE.read_text())
        pair = next((item for item in pairs if item.get("id") == pair_id), None)
        if pair is None:
            raise ValueError(f"source pair not found: {pair_id}")
        pair["direct_feedback_status"] = status
        tmp = PAIR_STORE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(pairs, indent=2, sort_keys=True))
        os.replace(tmp, PAIR_STORE)
        fcntl.flock(lock, fcntl.LOCK_UN)


def _resolve_run(run_rel: str) -> Path:
    path = REPO / run_rel
    if not path.is_dir() or not str(path.resolve()).startswith(str((REPO / "runs").resolve())):
        raise ValueError(f"invalid run directory: {run_rel}")
    return path


def _relative_difference(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else 0.0


def _relative_increase(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if value == 0 else float("inf")
    return (value - baseline) / abs(baseline)


def _trajectory_metrics(path_a: Path, path_b: Path, stock_size_in) -> dict:
    a = np.load(path_a / "trajectory.npy").astype(np.float64)
    b = np.load(path_b / "trajectory.npy").astype(np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"incompatible trajectories: {a.shape} versus {b.shape}")
    scale_mm = np.asarray(stock_size_in, dtype=np.float64) * 25.4
    a_mm, b_mm = a * scale_mm, b * scale_mm
    pointwise = np.linalg.norm(a_mm - b_mm, axis=1)
    length_a = np.linalg.norm(np.diff(a_mm, axis=0), axis=1).sum()
    length_b = np.linalg.norm(np.diff(b_mm, axis=0), axis=1).sum()
    return {
        "points": int(len(a)),
        "mean_pointwise_displacement_mm": float(pointwise.mean()),
        "max_pointwise_displacement_mm": float(pointwise.max()),
        "path_length_a_mm": float(length_a),
        "path_length_b_mm": float(length_b),
        "path_length_difference_mm": float(abs(length_a - length_b)),
    }


def _metric_snapshot(metrics: dict) -> dict:
    return {key: metrics.get(key) for key in METRIC_KEYS}


def write_overlay(iteration_id: str, output: Path) -> Path:
    """Render side-by-side and overlaid paths in physical millimetres."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = _read_json(ITERATIONS / iteration_id / "launch_plan.json")
    event = _read_json(ITERATIONS / iteration_id / "event.json")
    scale = np.asarray(event["fixed_controls"]["stock_size_in"], dtype=float) * 25.4
    paths = []
    for side in ("a", "b"):
        paths.append(np.load(_resolve_run(plan[f"run_{side}"]) / "trajectory.npy") * scale)
    fig = plt.figure(figsize=(15, 5))
    views = ((paths[0], None, "Option A"), (paths[1], None, "Option B"),
             (paths[0], paths[1], "A/B overlay"))
    for index, (first, second, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.plot(*first.T, color="#2563eb", linewidth=1.4, label="A")
        if second is not None:
            ax.plot(*second.T, color="#dc2626", linewidth=1.4, alpha=.85, label="B")
            ax.legend()
        elif index == 2:
            ax.lines[0].set_color("#dc2626")
        ax.set(title=title, xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
        ax.set_box_aspect((1, 1, 1))
    fig.suptitle(f"Direct-feedback iteration {iteration_id} — measured trajectories")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _explanation(side: str, before: float, after: float, metrics: dict) -> str:
    strength = "conservative" if side == "a" else "stronger"
    return (
        f"Option {side.upper()} — {strength} break-loss reduction. "
        f"Change: w_break changed from {before:g} to {after:g}; other loss weights and "
        f"run controls were held fixed. Intent: respond to the preference for a quicker, "
        f"similar trajectory by testing whether less break-loss pressure reduces time while "
        f"keeping measured break risk low. Observed: total time {metrics['total_time']:.3f} s, "
        f"air time {metrics['air_time']:.3f} s, hard Dice {metrics['hard_dice']:.6f}, "
        f"gouge {metrics['gouge']:.3f}, residual {metrics['residual']:.3f}, and break "
        f"probability {metrics['break_prob_any']:.6f}."
    )


def evaluate(iteration_id: str, visual_assessment: str) -> tuple[dict, dict]:
    work = ITERATIONS / iteration_id
    event = _read_json(work / "event.json")
    plan = _read_json(work / "launch_plan.json")
    if plan.get("execution") != "completed" or not plan.get("run_a") or not plan.get("run_b"):
        raise ValueError("both Phase 4 runs must complete before evaluation")
    run_a, run_b = _resolve_run(plan["run_a"]), _resolve_run(plan["run_b"])
    metrics_a = _read_json(run_a / "metrics.json")
    metrics_b = _read_json(run_b / "metrics.json")
    parent = _resolve_run(str(Path(event["parent_args_path"]).parent))
    parent_metrics = _read_json(parent / "metrics.json")
    thresholds = plan["meaningful_difference_threshold"]
    trajectory = _trajectory_metrics(
        run_a, run_b, event["fixed_controls"]["stock_size_in"]
    )

    # This mapping was fixed by the raw critique ("quicker") and Qwen's saved
    # target behavior ("save time") before child results existed.
    target_metric = "total_time"
    target_rel = _relative_difference(metrics_a[target_metric], metrics_b[target_metric])
    target_pass = target_rel >= thresholds["targeted_metric_relative_change_min"]
    trajectory_pass = trajectory["mean_pointwise_displacement_mm"] >= thresholds[
        "trajectory_mean_displacement_mm_min"
    ]
    quality = {}
    for side, metrics in (("a", metrics_a), ("b", metrics_b)):
        quality[side] = {
            "hard_dice_drop": parent_metrics["hard_dice"] - metrics["hard_dice"],
            "hard_dice_pass": parent_metrics["hard_dice"] - metrics["hard_dice"]
            <= thresholds["hard_dice_drop_max"],
            "gouge_relative_increase": _relative_increase(
                metrics["gouge"], parent_metrics["gouge"]
            ),
            "gouge_pass": _relative_increase(metrics["gouge"], parent_metrics["gouge"])
            <= thresholds["gouge_relative_increase_max"],
        }
    visual_pass = visual_assessment == "distinguishable"
    passed = target_pass and trajectory_pass and visual_pass and all(
        item[gate] for item in quality.values() for gate in ("hard_dice_pass", "gouge_pass")
    )
    overlay = work / "trajectory_overlay.png"
    visual_artifact = None
    if overlay.is_file():
        visual_artifact = {
            "path": str(overlay.relative_to(REPO)),
            "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
        }
    result = {
        "target_metric": target_metric,
        "target_metric_a": metrics_a[target_metric],
        "target_metric_b": metrics_b[target_metric],
        "target_metric_relative_difference": target_rel,
        "target_metric_pass": target_pass,
        "trajectory": trajectory,
        "trajectory_pass": trajectory_pass,
        "quality_vs_selected_parent": quality,
        "visual_assessment": visual_assessment,
        "visual_artifact": visual_artifact,
        "visual_pass": visual_pass,
        "thresholds": thresholds,
        "passed": passed,
        "classification": "meaningful" if passed else "weak_intervention",
        "retry_recommendation": None if passed else {
            "action": "redesign_contrast_once",
            "reason": (
                "The total-time contrast missed its predeclared 20% gate, and the "
                "stronger child already set w_break to its lower bound of zero."
            ),
            "proposed_target": "increase w_time directly in conservative and stronger variants",
            "requires_new_gpu_authorization": True,
        },
    }
    explanations = {
        "a": _explanation("a", event["variant_a"]["before"]["w_break"],
                          event["variant_a"]["changes"]["w_break"], metrics_a),
        "b": _explanation("b", event["variant_b"]["before"]["w_break"],
                          event["variant_b"]["changes"]["w_break"], metrics_b),
    }
    record = {
        "schema_version": 1,
        "event_type": "phase5_6_evaluation",
        "iteration_id": iteration_id,
        "ts": time.time(),
        "runs": {"a": plan["run_a"], "b": plan["run_b"]},
        "measured_metrics": {"a": _metric_snapshot(metrics_a), "b": _metric_snapshot(metrics_b)},
        "generated_explanations": explanations,
        "meaningful_difference": result,
        "next_pair_id": None,
    }
    return record, event


def record_evaluation(iteration_id: str, visual_assessment: str) -> dict:
    work = ITERATIONS / iteration_id
    event = _read_json(work / "event.json")
    existing_path = work / "phase5_6_evaluation.json"
    if event.get("status") in ("meaningful", "weak_intervention") and existing_path.is_file():
        existing = _read_json(existing_path)
        _set_source_pair_status(event["pair_snapshot"]["id"], event["status"])
        return existing
    record, event = evaluate(iteration_id, visual_assessment)
    event.update({
        "status": record["meaningful_difference"]["classification"],
        "runs": record["runs"],
        "measured_metrics": record["measured_metrics"],
        "generated_explanations": record["generated_explanations"],
        "meaningful_difference": record["meaningful_difference"],
        "next_pair_id": None,
    })
    tmp = work / "event.json.tmp"
    tmp.write_text(json.dumps(event, indent=2, sort_keys=True))
    os.replace(tmp, work / "event.json")
    existing_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    _append_event(record)
    _set_source_pair_status(event["pair_snapshot"]["id"], event["status"])
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iteration-id", required=True)
    ap.add_argument(
        "--visual-assessment", required=True,
        choices=("distinguishable", "not-distinguishable", "not-inspected"),
        help="assessment of an A/B overlay or equivalent rendered paths",
    )
    ap.add_argument("--record", action="store_true", help="persist the immutable evaluation event")
    ap.add_argument("--write-overlay", action="store_true", help="write a physical-space A/B path image")
    args = ap.parse_args()
    if args.write_overlay:
        write_overlay(args.iteration_id, ITERATIONS / args.iteration_id / "trajectory_overlay.png")
    if args.record:
        result = record_evaluation(args.iteration_id, args.visual_assessment)
    else:
        result, _ = evaluate(args.iteration_id, args.visual_assessment)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
