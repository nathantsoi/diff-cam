"""Plot the jul5-anneal-gap research results from results.tsv.

Two panels:
  1. Progress over experiments: dice vs experiment order, kept/discard/crash
     distinguished, with a running-best line.
  2. Generality across scenarios: best dice per (shape, method-family) for the
     1in stock — the payoff of varying the scenario and the method.

Robust: skips crash/0.0 rows; takes a commit's best when it appears >1x.
"""
import csv
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "results.tsv")
OUT = os.path.join(HERE, "results_plot.png")


def family(cmd):
    """Map a command to a method family label + (shape, stock)."""
    shape = "?"
    stock = "?"
    m = re.search(r"--target-shape (\S+)", cmd)
    if m:
        shape = m.group(1)
    m = re.search(r"--stock-size-in (\S+ \S+ \S+)", cmd)
    if m:
        stock = m.group(1)
    kf = re.search(r"--k-final (\S+)", cmd)
    ls = re.search(r"--loss-shift (\S+)", cmd)
    th = re.search(r"--tool-height-mm (\S+)", cmd)
    parts = []
    if "--k-anneal" in cmd and kf:
        parts.append(f"k{int(float(kf.group(1)))}")
    if ls and float(ls.group(1)) != 0.0:
        parts.append(f"ls{ls.group(1)}")
    if th and float(th.group(1)) > 25.0:
        parts.append(f"t{int(float(th.group(1)))}")
    label = "+".join(parts) if parts else "k10-base"
    return label, shape, stock


def main():
    rows = []
    with open(TSV, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                d = float(r["dice"])
            except (ValueError, KeyError):
                continue
            rows.append({"dice": d, "status": r["status"], "cmd": r.get("command", "")})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11))

    # Panel 1: progress over experiments
    kept = [(i, r["dice"]) for i, r in enumerate(rows) if r["status"] == "keep"]
    disc = [(i, r["dice"]) for i, r in enumerate(rows) if r["status"] == "discard"]
    crash = [(i, r["dice"]) for i, r in enumerate(rows) if r["status"] == "crash"]
    if kept:
        ax1.scatter([p[0] for p in kept], [p[1] for p in kept], c="green", s=20, label="keep", zorder=3)
    if disc:
        ax1.scatter([p[0] for p in disc], [p[1] for p in disc], c="red", s=16, marker="x", label="discard", zorder=3)
    if crash:
        ax1.scatter([p[0] for p in crash], [p[1] for p in crash], c="gray", s=16, marker="v", label="crash", zorder=3)
    best = 0.0
    rb_x, rb_y = [], []
    for i, r in enumerate(rows):
        if r["status"] == "keep" and r["dice"] > best:
            best = r["dice"]
        rb_x.append(i)
        rb_y.append(best)
    ax1.plot(rb_x, rb_y, "k-", lw=1.6, label="running best", zorder=2)
    ax1.set_xlabel("experiment order")
    ax1.set_ylabel("dice (deployable viz hard-carve)")
    ax1.set_title("jul5-anneal-gap: progress over experiments")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Panel 2: generality — best dice per (shape, family) for 1in stock
    best_per = defaultdict(lambda: 0.0)
    for r in rows:
        if r["dice"] <= 0.0:
            continue
        fam, shape, stock = family(r["cmd"])
        if stock.startswith("1 1 1"):
            key = (shape, fam)
            if r["dice"] > best_per[key]:
                best_per[key] = r["dice"]
    shapes = sorted({k[0] for k in best_per})
    fams = sorted({k[1] for k in best_per})
    x = np.arange(len(shapes))
    w = 0.8 / max(1, len(fams))
    for fi, fam in enumerate(fams):
        vals = [best_per.get((s, fam), 0.0) for s in shapes]
        ax2.bar(x + fi * w, vals, w, label=fam)
    ax2.set_xticks(x + w * (len(fams) - 1) / 2)
    ax2.set_xticklabels(shapes, rotation=15)
    ax2.set_ylabel("best deployable dice")
    ax2.set_title("Generality: best dice per shape × method-family (1in stock)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT} ({len(rows)} experiments, {len(shapes)} shapes, {len(fams)} families)")


if __name__ == "__main__":
    main()
