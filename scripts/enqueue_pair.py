#!/usr/bin/env python3
"""Enqueue a pairwise A/B preference pair (agent-side CLI).

After running two experiments that vary a single objective knob at two
magnitudes, the autoresearch agent calls this to queue the pair for the human
to judge in compare.html. It writes directly to pairwise.json (atomic
temp+replace under a lock) so it works headless — the web server need not be
running, since serve_web_https.py reads the same file.

The schema mirrors `add_pair` in scripts/serve_web_https.py exactly; run names
are normalized to basenames (the unique key into the stores).

Usage:
    uv run python scripts/enqueue_pair.py \
        --run-a CamEnvDiff-v0__train_csg__1__1783745927078 \
        --run-b CamEnvDiff-v0__train_csg__1__1783746004455 \
        --dimension w_air_time --mag-a 1e-3 --mag-b 1e-2 \
        --scenario "sphere s1 iters5000" \
        --prompt "Which trajectory air-cuts less at the end?"
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    # scripts/ is one level under the repo root.
    return here.parent.parent


def _pairwise_path() -> Path:
    return _repo_root() / "autoresearch" / "tasks" / "train_csg" / "pairwise.json"


def _run_key(run: str) -> str:
    """Normalize a runs/<batch>/<name> path or bare <name> to its basename."""
    return run.rstrip("/").rsplit("/", 1)[-1]


def _load() -> list:
    p = _pairwise_path()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text() or "[]")
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save(data: list) -> None:
    p = _pairwise_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, p)


def _new_pair_id(pairs: list) -> str:
    used = {p.get("id") for p in pairs}
    n = 1
    while f"p_{n:04d}" in used:
        n += 1
    return f"p_{n:04d}"


def enqueue(run_a: str, run_b: str, prompt: str = "", dimension: str = "",
            magnitude_a: str = "", magnitude_b: str = "", scenario: str = "") -> dict:
    """Append a new unanswered pair to the store; returns the stored pair."""
    with _LOCK:
        data = _load()
        pair = {
            "id": _new_pair_id(data),
            "run_a": _run_key(run_a),
            "run_b": _run_key(run_b),
            "prompt": (prompt or "").strip(),
            "dimension": (dimension or "").strip(),
            "magnitude_a": str(magnitude_a or "").strip(),
            "magnitude_b": str(magnitude_b or "").strip(),
            "scenario": (scenario or "").strip(),
            "ts": time.time(),
            "answer": None,
            "answer_ts": None,
            "note": "",
        }
        data.append(pair)
        _save(data)
        return pair


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-a", required=True, help="run name or runs/<batch>/<name> for side A")
    ap.add_argument("--run-b", required=True, help="run name or runs/<batch>/<name> for side B")
    ap.add_argument("--dimension", default="", help="objective knob varied, e.g. w_air_time")
    ap.add_argument("--mag-a", default="", help="magnitude of the knob for side A")
    ap.add_argument("--mag-b", default="", help="magnitude of the knob for side B")
    ap.add_argument("--scenario", default="", help="fixed config label, e.g. 'sphere s1 iters5000'")
    ap.add_argument("--prompt", default="", help="short prompt shown to the human in compare.html")
    args = ap.parse_args()

    if _run_key(args.run_a) == _run_key(args.run_b):
        raise SystemExit("error: --run-a and --run-b resolve to the same run")

    pair = enqueue(
        args.run_a, args.run_b, args.prompt,
        dimension=args.dimension, magnitude_a=args.mag_a,
        magnitude_b=args.mag_b, scenario=args.scenario,
    )
    print(json.dumps(pair, indent=2))
    print(f"[enqueue] {pair['id']} queued: {pair['run_a']} vs {pair['run_b']} "
          f"(dimension={pair['dimension'] or '(none)'})", flush=True)


if __name__ == "__main__":
    main()
