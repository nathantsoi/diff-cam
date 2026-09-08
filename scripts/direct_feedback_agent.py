#!/usr/bin/env python3
"""Turn one raw pairwise critique into two bounded loss configurations.

This is the direct-feedback decision boundary.  It reads the answered pair
itself (not the aggregate preference digest), sends that exact choice and raw
critique to the configured local OpenAI-compatible model, validates the model's
structured decision against an allowlist, and writes an append-only audit event
plus per-variant JSON files.  Training is deliberately a later orchestration
step so a malformed model response can never launch a GPU job.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
TASK = REPO / "autoresearch" / "tasks" / "train_csg"
PAIR_STORE = TASK / "pairwise.json"
EVENT_STORE = TASK / "direct_feedback_events.jsonl"
ITERATIONS = TASK / "direct_feedback"

DEFAULT_BASE_URL = "http://10.1.0.116:11434/v1"
DEFAULT_MODEL = "qwen3.8:latest"

# Bounds are intentionally narrower than the CLI's numerical domain.  The MVP
# permits loss-weight engineering while preventing arbitrary commands/source
# edits and pathological values.  At most two terms may change per variant.
LOSS_BOUNDS = {
    "w_gouge": (0.0, 50.0),
    "w_residual": (0.0, 20.0),
    "w_air": (0.0, 10.0),
    "w_jerk": (0.0, 1.0),
    "w_step": (0.0, 10.0),
    "w_prox": (0.0, 10.0),
    "w_traj_prox": (0.0, 10.0),
    "w_len": (0.0, 1.0),
    "w_tool_gouge": (0.0, 50.0),
    "w_time": (0.0, 0.1),
    "w_air_time": (0.0, 0.1),
    "w_air_late": (0.0, 0.1),
    "w_break": (0.0, 1.0),
}

CONTROL_KEYS = (
    "seed", "iters", "max_steps", "target_shape", "target_radius_mm",
    "target_height_mm", "stock_size_in", "voxel_size_mm", "init_mode",
)
METRIC_KEYS = (
    "hard_dice", "dice", "residual", "gouge", "holder_overlap",
    "air_cut_fraction", "air_time", "total_time", "break_prob_any",
    "best_score",
)


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _find_pair(pair_id: str) -> dict:
    for pair in _load_json(PAIR_STORE, []):
        if pair.get("id") == pair_id:
            return pair
    raise ValueError(f"unknown pair id: {pair_id}")


def _find_run(name: str) -> Path:
    hits = [p for p in (REPO / "runs").rglob(name) if p.is_dir() and p.name == name]
    if not hits:
        raise ValueError(f"run not found: {name}")
    return max(hits, key=lambda p: p.stat().st_mtime)


def _run_snapshot(name: str) -> dict:
    run = _find_run(name)
    args = _load_json(run / "args.json", {})
    metrics = _load_json(run / "metrics.json", {})
    return {
        "id": name,
        "path": str(run.relative_to(REPO)),
        "controls": {key: args.get(key) for key in CONTROL_KEYS},
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
        "loss": {k: float(args.get(k, 0.0)) for k in LOSS_BOUNDS},
    }


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response contained no JSON object")
        return json.loads(text[start:end + 1])


def _call_model(payload: dict, base_url: str, model: str) -> tuple[dict, str]:
    schema = {
        "interpretation": {
            "target_behavior": "short description",
            "rationale": "why these existing loss terms address the raw critique",
            "uncertainty": "what may not be captured by these metrics",
        },
        "variant_a": {
            "hypothesis": "conservative adaptation",
            "changes": {"one_allowlisted_weight": 0.0},
        },
        "variant_b": {
            "hypothesis": "stronger adaptation of the same interpretation",
            "changes": {"same_allowlisted_weight": 0.0},
        },
    }
    system = (
        "/no_think\nYou are the bounded objective-engineering agent for DiffCAM. Respond with JSON only. "
        "Use the raw human critique directly. Both variants must respond to the same critique: "
        "A is conservative and B is stronger. Change one or at most two existing loss weights, "
        "prefer the same terms in A and B, and do not propose source edits, commands, metric edits, "
        "initialization changes, or unrelated bundles. Be factual and state uncertainty. "
        "Keep every prose field under 25 words."
    )
    user = json.dumps({
        "event": payload,
        "allowed_loss_weights_and_bounds": LOSS_BOUNDS,
        "required_response_shape": schema,
    }, sort_keys=True)
    request_body = json.dumps({
        "model": model,
        "temperature": 0.1,
        "max_tokens": 700,
        "stream": False,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            raw = response.read().decode()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"local model request failed: {exc}") from exc
    envelope = json.loads(raw)
    content = envelope["choices"][0]["message"]["content"]
    return _extract_json(content), content


def _validate_variant(raw: dict, base_loss: dict, label: str) -> dict:
    if not isinstance(raw, dict) or not isinstance(raw.get("changes"), dict):
        raise ValueError(f"{label} must contain a changes object")
    changes = raw["changes"]
    if not 1 <= len(changes) <= 2:
        raise ValueError(f"{label} must change one or two loss weights")
    normalized = {}
    for key, value in changes.items():
        if key not in LOSS_BOUNDS:
            raise ValueError(f"{label} attempted non-allowlisted change: {key}")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.{key} is not numeric") from exc
        lo, hi = LOSS_BOUNDS[key]
        if not lo <= value <= hi:
            raise ValueError(f"{label}.{key}={value} outside [{lo}, {hi}]")
        if value == float(base_loss[key]):
            raise ValueError(f"{label}.{key} does not alter the parent objective")
        normalized[key] = value
    return {
        "hypothesis": str(raw.get("hypothesis", "")).strip(),
        "changes": normalized,
        "before": {key: float(base_loss[key]) for key in normalized},
        "effective_loss": {**base_loss, **normalized},
    }


def _append_event(event: dict) -> None:
    EVENT_STORE.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    with EVENT_STORE.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def process(pair_id: str, base_url: str, model: str) -> dict:
    pair = _find_pair(pair_id)
    if pair.get("answer") not in ("a", "b", "tie"):
        raise ValueError(f"pair {pair_id} has not been answered")
    if not str(pair.get("note", "")).strip():
        raise ValueError("a non-empty raw critique is required for direct feedback")

    run_a, run_b = _run_snapshot(pair["run_a"]), _run_snapshot(pair["run_b"])
    parent_side = pair["answer"] if pair["answer"] in ("a", "b") else "a"
    parent = run_a if parent_side == "a" else run_b
    iteration_id = f"df_{pair_id}_{int(float(pair['answer_ts']) * 1000)}"
    work = ITERATIONS / iteration_id
    work.mkdir(parents=True, exist_ok=False)

    agent_input = {
        "iteration_id": iteration_id,
        "pair": pair,
        "raw_choice": pair["answer"],
        "raw_critique": pair["note"],
        "parent_rule": "chosen option; A on tie",
        "parent_side": parent_side,
        "parent_run": parent["id"],
        "run_a": run_a,
        "run_b": run_b,
    }
    (work / "agent_input.json").write_text(json.dumps(agent_input, indent=2, sort_keys=True))
    decision, raw_response = _call_model(agent_input, base_url, model)
    variant_a = _validate_variant(decision.get("variant_a"), parent["loss"], "variant_a")
    variant_b = _validate_variant(decision.get("variant_b"), parent["loss"], "variant_b")
    if set(variant_a["changes"]) != set(variant_b["changes"]):
        raise ValueError("A and B must vary the same loss terms in the initial MVP")
    for key in variant_a["changes"]:
        base = float(parent["loss"][key])
        da = variant_a["changes"][key] - base
        db = variant_b["changes"][key] - base
        if da * db <= 0 or abs(db) <= abs(da):
            raise ValueError(
                f"variant B must move {key} farther than A in the same direction"
            )

    interpretation = decision.get("interpretation")
    if not isinstance(interpretation, dict):
        raise ValueError("model response lacks structured interpretation")
    event = {
        "schema_version": 1,
        "iteration_id": iteration_id,
        "status": "objective_decided",
        "recorded_ts": time.time(),
        "pair_snapshot": pair,
        "raw_choice": pair["answer"],
        "raw_critique": pair["note"],
        "agent": {"base_url": base_url, "model": model, "temperature": 0.1},
        "interpretation": {
            "target_behavior": str(interpretation.get("target_behavior", "")).strip(),
            "rationale": str(interpretation.get("rationale", "")).strip(),
            "uncertainty": str(interpretation.get("uncertainty", "")).strip(),
        },
        "parent_side": parent_side,
        "parent_run": parent["id"],
        "parent_args_path": parent["path"] + "/args.json",
        "alteration_mode": "validated_loss_weight_configuration",
        "source_diff": None,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "fixed_controls": {
            key: parent["controls"].get(key) for key in CONTROL_KEYS
        },
        "git_before": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "next_pair_id": None,
        "runs": {"a": None, "b": None},
        "measured_metrics": {"a": None, "b": None},
        "meaningful_difference": None,
        "generated_explanations": {"a": None, "b": None},
    }
    canonical = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    event["decision_sha256"] = hashlib.sha256(canonical).hexdigest()
    (work / "agent_response.txt").write_text(raw_response)
    (work / "variant_a.json").write_text(json.dumps(variant_a, indent=2, sort_keys=True))
    (work / "variant_b.json").write_text(json.dumps(variant_b, indent=2, sort_keys=True))
    (work / "event.json").write_text(json.dumps(event, indent=2, sort_keys=True))
    _append_event(event)
    return event


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-id", required=True)
    ap.add_argument("--base-url", default=os.environ.get("DIFFCAM_LLM_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--model", default=os.environ.get("DIFFCAM_LLM_MODEL", DEFAULT_MODEL))
    args = ap.parse_args()
    event = process(args.pair_id, args.base_url, args.model)
    print(json.dumps({
        "ok": True,
        "iteration_id": event["iteration_id"],
        "changes_a": event["variant_a"]["changes"],
        "changes_b": event["variant_b"]["changes"],
    }, indent=2))


if __name__ == "__main__":
    main()
