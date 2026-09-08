#!/usr/bin/env python3
"""Build and optionally execute a controlled A/B direct-feedback experiment.

The parent run's saved reproduction command is the source of truth. Both child
commands are cloned from it, with only (1) the validated loss weights and (2)
the output subdirectory changed. Dry-run planning is the default; ``--execute``
must be explicit before this script can consume GPU time.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
TASK = REPO / "autoresearch" / "tasks" / "train_csg"
ITERATIONS = TASK / "direct_feedback"
EVENT_STORE = TASK / "direct_feedback_events.jsonl"
RUN_LINE = re.compile(r"^\[run\] writing outputs to (runs/\S+)\s*$")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parent_command(event: dict) -> list[str]:
    args_path = REPO / event["parent_args_path"]
    repro = args_path.parent / "reproduce_command.sh"
    try:
        lines = [line.strip() for line in repro.read_text().splitlines()]
    except OSError as exc:
        raise ValueError(f"missing parent reproduction command: {repro}") from exc
    command = next((line for line in lines if line and not line.startswith("#")), "")
    tokens = shlex.split(command)
    if len(tokens) < 4 or tokens[1:3] != ["-m", "algorithms.train_csg"]:
        raise ValueError("parent reproduction command is not python -m algorithms.train_csg")
    tokens[0] = sys.executable
    return tokens


def _remove_flag(tokens: list[str], flag: str) -> list[str]:
    return [token for token in tokens if token != flag]


def _set_scalar(tokens: list[str], flag: str, value) -> list[str]:
    """Replace every scalar flag occurrence, or append it when absent."""
    out, found, i = [], False, 0
    while i < len(tokens):
        if tokens[i] == flag:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ValueError(f"malformed scalar option in parent command: {flag}")
            if not found:
                out.extend([flag, str(value)])
                found = True
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    if not found:
        out.extend([flag, str(value)])
    return out


def _variant_command(base: list[str], iteration_id: str, side: str, variant: dict) -> list[str]:
    cmd = list(base)
    # Never transmit data externally or activate the legacy feedback warm start.
    cmd = _remove_flag(cmd, "--track")
    cmd = _remove_flag(cmd, "--use_feedback")
    cmd = _remove_flag(cmd, "--use-feedback")
    if "--no-track" not in cmd:
        cmd.append("--no-track")
    cmd = _set_scalar(cmd, "--runs_subdir", f"direct-feedback/{iteration_id}/{side}")
    for key, value in sorted(variant["changes"].items()):
        cmd = _set_scalar(cmd, "--" + key, value)
    return cmd


def _option_map(tokens: list[str]) -> dict[str, list[str | bool]]:
    """Parse this known argparse-style command for exact A/B diff checking."""
    out: dict[str, list[str | bool]] = {}
    i = 3  # python -m algorithms.train_csg
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional token: {token}")
        values = []
        i += 1
        while i < len(tokens) and not tokens[i].startswith("--"):
            values.append(tokens[i])
            i += 1
        out[token] = values or [True]
    return out


def _validate_isolated(cmd_a: list[str], cmd_b: list[str], event: dict) -> list[str]:
    a, b = _option_map(cmd_a), _option_map(cmd_b)
    changed = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
    expected = sorted(
        ["--runs_subdir"] + ["--" + key for key in event["variant_a"]["changes"]]
    )
    if changed != expected:
        raise ValueError(f"uncontrolled A/B command difference: {changed}; expected {expected}")
    return changed


def _append_event(record: dict) -> None:
    EVENT_STORE.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with EVENT_STORE.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def build_plan(iteration_id: str) -> tuple[dict, Path]:
    work = ITERATIONS / iteration_id
    event = _read_json(work / "event.json")
    if event.get("iteration_id") != iteration_id or event.get("status") != "objective_decided":
        raise ValueError("iteration is not in objective_decided state")
    if set(event["variant_a"]["changes"]) != set(event["variant_b"]["changes"]):
        raise ValueError("A and B do not alter the same loss terms")
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO, text=True,
    ).strip()
    if dirty:
        raise ValueError("launch planning requires a clean, versioned working tree")
    if current_head != event.get("git_before"):
        raise ValueError(
            f"code drift since objective decision: event={event.get('git_before')} current={current_head}"
        )

    base = _parent_command(event)
    cmd_a = _variant_command(base, iteration_id, "a", event["variant_a"])
    cmd_b = _variant_command(base, iteration_id, "b", event["variant_b"])
    changed = _validate_isolated(cmd_a, cmd_b, event)
    plan = {
        "schema_version": 1,
        "event_type": "phase4_launch_plan",
        "iteration_id": iteration_id,
        "created_ts": time.time(),
        "git_commit": current_head,
        "parent_run": event["parent_run"],
        "fixed_controls": event["fixed_controls"],
        "targeted_loss_terms": sorted(event["variant_a"]["changes"]),
        "variant_a": {
            "hypothesis": event["variant_a"]["hypothesis"],
            "before": event["variant_a"]["before"],
            "after": event["variant_a"]["changes"],
            "command": shlex.join(cmd_a),
        },
        "variant_b": {
            "hypothesis": event["variant_b"]["hypothesis"],
            "before": event["variant_b"]["before"],
            "after": event["variant_b"]["changes"],
            "command": shlex.join(cmd_b),
        },
        "allowed_command_differences": changed,
        "execution": "planned",
        "gpu": None,
        "run_a": None,
        "run_b": None,
        "meaningful_difference_threshold": {
            "targeted_metric_relative_change_min": 0.20,
            "trajectory_mean_displacement_mm_min": 0.5,
            "hard_dice_drop_max": 0.03,
            "gouge_relative_increase_max": 0.10,
        },
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    plan["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    (work / "launch_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan, work


def _run_one(cmd: list[str], gpu: str, log_path: Path) -> tuple[str, dict]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    run_rel = None
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            match = RUN_LINE.match(line.rstrip())
            if match:
                run_rel = match.group(1)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"training failed with exit {rc}; see {log_path}")
    if not run_rel:
        raise RuntimeError(f"training produced no parseable run path; see {log_path}")
    metrics = _read_json(REPO / run_rel / "metrics.json")
    return run_rel, metrics


def execute(plan: dict, work: Path, gpu: str) -> dict:
    if plan.get("execution") != "planned":
        raise ValueError("launch plan is not executable")
    (REPO / "runs/direct-feedback" / plan["iteration_id"] / "a").mkdir(parents=True, exist_ok=True)
    (REPO / "runs/direct-feedback" / plan["iteration_id"] / "b").mkdir(parents=True, exist_ok=True)
    started = {
        "schema_version": 1, "event_type": "phase4_runs_started",
        "iteration_id": plan["iteration_id"], "ts": time.time(), "gpu": gpu,
    }
    _append_event(started)
    run_a, metrics_a = _run_one(shlex.split(plan["variant_a"]["command"]), gpu, work / "run_a.log")
    run_b, metrics_b = _run_one(shlex.split(plan["variant_b"]["command"]), gpu, work / "run_b.log")
    completed = {
        "schema_version": 1, "event_type": "phase4_runs_completed",
        "iteration_id": plan["iteration_id"], "ts": time.time(), "gpu": gpu,
        "run_a": run_a, "run_b": run_b,
        "metrics_a": metrics_a, "metrics_b": metrics_b,
    }
    _append_event(completed)
    plan.update({
        "execution": "completed", "gpu": gpu,
        "run_a": run_a, "run_b": run_b,
        "metrics_a": metrics_a, "metrics_b": metrics_b,
    })
    (work / "launch_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iteration-id", required=True)
    ap.add_argument("--execute", action="store_true", help="run both variants sequentially")
    ap.add_argument("--gpu", help="physical CUDA device index; required with --execute")
    args = ap.parse_args()
    if args.execute and args.gpu is None:
        raise SystemExit("--gpu is required with --execute")
    plan, work = build_plan(args.iteration_id)
    _append_event(plan)
    if args.execute:
        plan = execute(plan, work, args.gpu)
    print(json.dumps({
        "ok": True, "iteration_id": args.iteration_id,
        "execution": plan["execution"], "plan_sha256": plan["plan_sha256"],
        "run_a": plan.get("run_a"), "run_b": plan.get("run_b"),
    }, indent=2))


if __name__ == "__main__":
    main()
