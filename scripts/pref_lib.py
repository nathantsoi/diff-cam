"""Shared aggregation for pairwise preference learning.

Both the headless digest CLI (`pref_digest.py`) and the web server's
`/__api/pref-digest` endpoint call `digest(pairs)` so the agent and the UI see
the exact same view of what the human has answered.

A "dimension" is the single objective knob a pair varies (e.g. `w_air_time`,
`init_mode`); the two magnitudes are stored on the pair as `magnitude_a` /
`magnitude_b`. Pairs enqueued before the structured fields existed (or ad-hoc
pairs with no dimension) bucket under the "(unstructured)" dimension so they
are not lost.
"""
from __future__ import annotations

from collections import defaultdict


def _dim(p: dict) -> str:
    d = (p.get("dimension") or "").strip()
    return d if d else "(unstructured)"


def digest(pairs: list) -> dict:
    """Aggregate answered pairs by dimension.

    Returns `{dimension: {n, a_wins, b_wins, ties, preferred, notes, pairs}}`
    where `preferred` is "a" | "b" | "tie" | None (None when there is no clear
    winner), `notes` is the list of non-empty answer notes (newest-first), and
    `pairs` is the list of answered pair ids in that dimension. Unanswered pairs
    are ignored.
    """
    by_dim: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "a_wins": 0, "b_wins": 0, "ties": 0, "notes": [], "pairs": []}
    )
    for p in pairs:
        ans = p.get("answer")
        if ans not in ("a", "b", "tie"):
            continue
        d = _dim(p)
        rec = by_dim[d]
        rec["n"] += 1
        rec["pairs"].append(p.get("id", ""))
        if ans == "a":
            rec["a_wins"] += 1
        elif ans == "b":
            rec["b_wins"] += 1
        else:
            rec["ties"] += 1
        note = (p.get("note") or "").strip()
        if note:
            rec["notes"].append(note)

    for rec in by_dim.values():
        a, b, t = rec["a_wins"], rec["b_wins"], rec["ties"]
        if a > b and a > t:
            rec["preferred"] = "a"
        elif b > a and b > t:
            rec["preferred"] = "b"
        elif t > a and t > b:
            rec["preferred"] = "tie"
        else:
            rec["preferred"] = None
        rec["notes"] = list(reversed(rec["notes"]))  # newest-first
    return dict(by_dim)


def pending(pairs: list) -> list:
    """Ids of unanswered pairs, in store order."""
    return [p.get("id", "") for p in pairs if not p.get("answer")]


def summary_counts(pairs: list) -> dict:
    """Headline counts used by the UI stats bar and the training-run log."""
    total = len(pairs)
    answered = sum(1 for p in pairs if p.get("answer") in ("a", "b", "tie"))
    return {"total": total, "answered": answered, "pending": total - answered}
