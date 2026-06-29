"""Summary plot for the train_csg autoresearch.

Reads ``results.tsv`` (commit, dice, memory_gb, status, description, command)
and writes ``results_plot.png`` with:

  1. Dice over experiments (chronological run order) with a running-best line,
     points coloured by keep/discard/crash.
  2. Best dice per target-shape scenario (parsed from the command), so the
     per-scenario ceiling is visible at a glance.

Run from the repo root (or anywhere with ``results.tsv`` on the path):

    python autoresearch/tasks/train_csg/plot_results.py
"""
import csv
import os
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# results.tsv lives at the repo root; this script is under autoresearch/tasks/train_csg.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.path.join(REPO, "results.tsv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_plot.png")

STATUS_MARKER = {"keep": "o", "discard": "x", "crash": "v"}
STATUS_LABEL = {"keep": "keep", "discard": "discard", "crash": "crash"}


def parse_shape(desc, cmd):
    """Recover the target shape from the description or command string."""
    for s in ("sphere", "cylinder", "box", "pyramid"):
        if s in cmd or s in desc:
            return s
    return "sphere"  # default scenario


def main():
    rows = []
    with open(RESULTS, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                dice = float(r["dice"])
            except (ValueError, KeyError):
                continue
            cmd = r.get("command", "")
            desc = r.get("description", "")
            rows.append({
                "commit": r.get("commit", "?"),
                "dice": dice,
                "status": r.get("status", "discard"),
                "desc": desc,
                "cmd": cmd,
                "shape": parse_shape(desc, cmd),
            })

    if not rows:
        raise SystemExit(f"no rows parsed from {RESULTS}")

    x = list(range(len(rows)))
    dice = [r["dice"] for r in rows]
    statuses = [r["status"] for r in rows]
    shapes = [r["shape"] for r in rows]

    # Running best across all experiments (the headline number).
    running_best, best = [], -1.0
    for d in dice:
        best = max(best, d)
        running_best.append(best)

    # Best dice per scenario.
    per_shape = {}
    for r in rows:
        per_shape[r["shape"]] = max(per_shape.get(r["shape"], -1.0), r["dice"])
    shape_order = sorted(per_shape, key=lambda k: -per_shape[k])

    cmap = plt.get_cmap("tab10")
    status_color = {"keep": "tab:green", "discard": "tab:gray", "crash": "tab:red"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [2.2, 1]})

    # --- Panel 1: dice over time ---
    for status, marker in STATUS_MARKER.items():
        xs = [i for i, s in enumerate(statuses) if s == status]
        if not xs:
            continue
        ax1.scatter([x[i] for i in xs], [dice[i] for i in xs],
                    c=[status_color[status] for _ in xs], marker=marker,
                    s=70 if status == "keep" else 45, edgecolors="black",
                    linewidths=0.6 if status == "keep" else 0.0,
                    label=STATUS_LABEL[status], zorder=3)
    ax1.plot(x, running_best, color="crimson", lw=1.8, ls="--", label="running best", zorder=2)
    # Annotate the overall best.
    bi = int(max(range(len(dice)), key=lambda i: dice[i]))
    ax1.annotate(f"best={dice[bi]:.4f}\n({rows[bi]['shape']})", xy=(bi, dice[bi]),
                 xytext=(bi, dice[bi] + 0.05), fontsize=8, color="crimson",
                 ha="center", arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
    ax1.set_xlabel("experiment # (chronological run order)")
    ax1.set_ylabel("dice")
    ax1.set_title("GradMill (train_csg) autoresearch — dice over time")
    ax1.set_xlim(-1, len(rows))
    ax1.set_ylim(-0.05, max(dice) * 1.18)
    ax1.grid(True, alpha=0.25, zorder=0)
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # --- Panel 2: best dice per scenario ---
    bars = ax2.bar(shape_order, [per_shape[s] for s in shape_order],
                   color=[cmap(i % 10) for i in range(len(shape_order))], edgecolor="black")
    for b, s in zip(bars, shape_order):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{per_shape[s]:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("best dice")
    ax2.set_title("Best dice per target shape")
    ax2.set_ylim(0, max(per_shape.values()) * 1.18)
    ax2.grid(True, alpha=0.25, axis="y", zorder=0)

    fig.suptitle(f"train_csg autoresearch — {len(rows)} experiments, overall best {max(dice):.4f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT} ({len(rows)} experiments; per-shape: "
          + ", ".join(f"{s}={per_shape[s]:.4f}" for s in shape_order) + ")")


if __name__ == "__main__":
    main()
