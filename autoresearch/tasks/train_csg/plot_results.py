"""Plot autoresearch train_csg results over time, grouped by experiment branch/commit."""
import csv
import os
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = "results.tsv"
OUT = "results_over_time.png"


def main():
    rows = []
    with open(RESULTS, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                rows.append({
                    "commit": r["commit"],
                    "dice": float(r["dice"]),
                    "status": r["status"],
                    "desc": r["description"],
                })
            except (ValueError, KeyError):
                continue

    # Order = experiment order (file order = chronological run order).
    x = list(range(len(rows)))
    dice = [r["dice"] for r in rows]
    commits = [r["commit"] for r in rows]
    statuses = [r["status"] for r in rows]

    # Distinct commit -> branch label + color. The commit hash is the branch state.
    unique_commits = []
    for c in commits:
        if c not in unique_commits:
            unique_commits.append(c)

    # Short, human-readable branch labels mapping each commit to its theme.
    branch_labels = {
        "6e0d2f4": "baseline (sphere, HARD dice)",
        "2b2f30a": "k=5 sharper soft union (crash)",
        "6278dd0": "raster_fine init",
        "116078a": "w_gouge / w_tool_gouge barriers",
        "625ec5d": "w_gouge=16 seed2",
        "edfd235": "zlayer init discovery",
        "d2f5f02": "zlayer + feed_ipm (speed cap)",
        "3e09bf7": "zlayer param sweep (revs/osc/margin)",
        "38392cb": "shape-aware pyramid/box",
        "2bf863b": "pyramid 4-phase (below-disk recovery)",
        "08d6f14": "sphere osc resonance + cyl osc",
        "7e31dfa": "sphere T-scaling + determinism",
    }

    cmap = plt.get_cmap("tab10")
    commit_color = {c: cmap(i % 10) for i, c in enumerate(unique_commits)}

    status_marker = {
        "keep": "o",
        "discard": "x",
        "crash": "v",
    }
    status_label = {
        "keep": "keep",
        "discard": "discard",
        "crash": "crash",
    }

    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot per status so the legend stays clean.
    for status, marker in status_marker.items():
        xs = [i for i, s in enumerate(statuses) if s == status]
        if not xs:
            continue
        ax.scatter(
            [x[i] for i in xs],
            [dice[i] for i in xs],
            c=[commit_color[commits[i]] for i in xs],
            marker=marker,
            s=70 if status == "keep" else 45,
            edgecolors="black",
            linewidths=0.6 if status == "keep" else 0.0,
            label=status_label[status],
            zorder=3,
        )

    # Running best (max so far) line.
    running_best = []
    best = -1.0
    for d in dice:
        if d > best:
            best = d
        running_best.append(best)
    ax.plot(x, running_best, color="crimson", lw=1.8, ls="--",
            label="running best", zorder=2)

    # Vertical separators + labels per commit block.
    boundaries = []
    for i in range(1, len(commits)):
        if commits[i] != commits[i - 1]:
            boundaries.append(i)
    for b in boundaries:
        ax.axvline(b - 0.5, color="gray", lw=0.7, ls=":", alpha=0.7, zorder=1)

    # Commit labels centered over each block, placed near the top.
    blocks = []
    start = 0
    for b in boundaries + [len(commits)]:
        blocks.append((start, b - 1))
        start = b
    ymax = max(dice) if max(dice) > 0 else 1.0
    for (s, e), c in zip(blocks, unique_commits):
        mid = (s + e) / 2.0
        label = branch_labels.get(c, c)
        ax.text(mid, ymax * 1.03, label, ha="center", va="bottom",
                fontsize=8, color=commit_color[c], fontweight="bold", rotation=0)

    ax.set_xlabel("experiment # (chronological run order)")
    ax.set_ylabel("dice")
    ax.set_title("GradMill (train_csg) autoresearch — dice over time, by branch/commit")
    ax.set_xlim(-1, len(rows))
    ax.set_ylim(-0.05, ymax * 1.18)
    ax.grid(True, alpha=0.25, zorder=0)

    # Legend: status markers + commit colors + running best.
    handles = []
    for status, marker in status_marker.items():
        handles.append(Line2D([0], [0], marker=marker, color="gray",
                              linestyle="none", markersize=8, label=status))
    handles.append(Line2D([0], [0], color="crimson", ls="--", lw=1.8, label="running best"))
    for c in unique_commits:
        handles.append(Line2D([0], [0], marker="s", color=commit_color[c],
                              linestyle="none", markersize=9, markeredgecolor="black",
                              markeredgewidth=0.5,
                              label=branch_labels.get(c, c)))
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, ncol=2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT} ({len(rows)} experiments, {len(unique_commits)} commits)")


if __name__ == "__main__":
    main()
