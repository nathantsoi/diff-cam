#!/usr/bin/env python3
"""Prepare the single controlled redesign allowed after a weak intervention."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
TASK = REPO / "autoresearch" / "tasks" / "train_csg"
ITERATIONS = TASK / "direct_feedback"
EVENT_STORE = TASK / "direct_feedback_events.jsonl"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_retry(source: dict, source_eval: dict, git_head: str,
                value_a: float, value_b: float) -> dict:
    if source.get("status") != "weak_intervention":
        raise ValueError("retry requires a recorded weak intervention")
    recommendation = source_eval["meaningful_difference"].get("retry_recommendation") or {}
    if recommendation.get("action") != "redesign_contrast_once":
        raise ValueError("source evaluation did not authorize one redesign")
    if not (0.0 < value_a < value_b <= 10.0):
        raise ValueError("require 0 < conservative w_time < stronger w_time <= 10")
    effective = dict(source["variant_a"].get("effective_loss", {}))
    before = float(effective.get("w_time", 0.001))
    # The retry clones the originally selected parent, not either failed child.
    effective["w_break"] = float(source["variant_a"]["before"].get("w_break", 0.001))
    iteration_id = source["iteration_id"] + "_retry1"
    event = {
        "schema_version": 1,
        "iteration_id": iteration_id,
        "status": "objective_decided",
        "recorded_ts": time.time(),
        "git_before": git_head,
        "parent_run": source["parent_run"],
        "parent_side": source["parent_side"],
        "parent_args_path": source["parent_args_path"],
        "fixed_controls": source["fixed_controls"],
        "raw_choice": source["raw_choice"],
        "raw_critique": source["raw_critique"],
        "pair_snapshot": source["pair_snapshot"],
        "alteration_mode": "one_time_phase6_redesign_of_loss_weight_configuration",
        "source_iteration_id": source["iteration_id"],
        "interpretation": {
            "target_behavior": "Directly penalize total trajectory time after w_break reached zero without meeting the declared time-separation gate.",
            "rationale": "The critique explicitly preferred quicker trajectories; w_time is the differentiable loss term for that measured behavior.",
            "uncertainty": "A stronger time penalty may trade carving quality for speed; the existing Dice and gouge gates remain binding.",
        },
        "variant_a": {
            "before": {"w_time": before},
            "changes": {"w_time": value_a},
            "effective_loss": dict(effective, w_time=value_a),
            "hypothesis": "A conservative direct time-loss increase may shorten the path while preserving carving quality.",
        },
        "variant_b": {
            "before": {"w_time": before},
            "changes": {"w_time": value_b},
            "effective_loss": dict(effective, w_time=value_b),
            "hypothesis": "A stronger direct time-loss increase should create a measurable speed/path contrast, with quality checked rather than assumed.",
        },
        "generated_explanations": {"a": None, "b": None},
        "measured_metrics": {"a": None, "b": None},
        "meaningful_difference": None,
        "runs": {"a": None, "b": None},
        "next_pair_id": None,
        "source_diff": None,
    }
    return event


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-iteration", required=True)
    ap.add_argument("--w-time-a", required=True, type=float)
    ap.add_argument("--w-time-b", required=True, type=float)
    args = ap.parse_args()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, text=True
    ).strip()
    if dirty:
        raise SystemExit("retry preparation requires a clean, versioned worktree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    source_work = ITERATIONS / args.source_iteration
    event = build_retry(
        _read(source_work / "event.json"),
        _read(source_work / "phase5_6_evaluation.json"),
        head, args.w_time_a, args.w_time_b,
    )
    work = ITERATIONS / event["iteration_id"]
    if work.exists():
        raise SystemExit(f"retry already exists: {work}")
    work.mkdir(parents=True)
    (work / "event.json").write_text(json.dumps(event, indent=2, sort_keys=True))
    line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    with EVENT_STORE.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)
    print(json.dumps({"ok": True, "iteration_id": event["iteration_id"]}, indent=2))


if __name__ == "__main__":
    main()
