#!/usr/bin/env python3
"""Print a human-readable digest of answered pairwise preferences.

The autoresearch agent runs this at the top of each loop iteration to see what
the human has answered: per dimension (the single objective knob a pair
varies), the win direction (A / B / tie), the count, and the free-text reasons.
It then uses this to steer the next dimension/magnitude sweep and to
reformulate the objective in pref_objective_plan.md.

Reads the same pairwise.json the web server reads; the web UI's digest panel
fetches the same view via GET /__api/pref-digest (both use pref_lib.digest).

Usage:
    uv run python scripts/pref_digest.py [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import sibling pref_lib (scripts/ dir is on sys.path when run as a script).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pref_lib import digest, pending, summary_counts  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pairwise_path() -> Path:
    return _repo_root() / "autoresearch" / "tasks" / "train_csg" / "pairwise.json"


def _load() -> list:
    p = _pairwise_path()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text() or "[]")
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _render(dims: dict, counts: dict, pend: list) -> str:
    lines = []
    lines.append(f"pairwise preferences: {counts['answered']}/{counts['total']} answered "
                 f"({counts['pending']} pending -> {', '.join(pend) or 'none'})")
    if not dims:
        lines.append("  (no answered pairs yet — enqueue pairs via scripts/enqueue_pair.py)")
        return "\n".join(lines)
    # Stable, informative order: structured dimensions alpha, then unstructured last.
    keys = sorted(k for k in dims if k != "(unstructured)")
    if "(unstructured)" in dims:
        keys.append("(unstructured)")
    for d in keys:
        rec = dims[d]
        pref = rec["preferred"] or "no clear winner"
        lines.append(f"  [{d}] n={rec['n']} preferred={pref} "
                     f"(A={rec['a_wins']} B={rec['b_wins']} tie={rec['ties']})")
        if rec["notes"]:
            for note in rec["notes"][:8]:
                snip = note if len(note) <= 240 else note[:237] + "..."
                lines.append(f"      - {snip}")
            if len(rec["notes"]) > 8:
                lines.append(f"      ... +{len(rec['notes']) - 8} more notes")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit the raw digest JSON instead of text")
    args = ap.parse_args()

    pairs = _load()
    dims = digest(pairs)
    pend = pending(pairs)
    counts = summary_counts(pairs)
    if args.json:
        print(json.dumps({"by_dimension": dims, "pending": pend, "counts": counts}, indent=2))
        return
    print(_render(dims, counts, pend))


if __name__ == "__main__":
    main()
