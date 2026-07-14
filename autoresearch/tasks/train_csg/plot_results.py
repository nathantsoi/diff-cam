"""Plot jul13-phys-plausible results: dice progress + physical-violation metrics.

Left: hard dice per experiment (chronological, keep/discard/crash markers,
running best) from results.tsv. Right: the campaign's real story — physical
plausibility metrics per experiment (plunge fraction, peak force vs cap,
fragile margin raw/scheduled) parsed from runs/jul13-phys-plausible/*/
metrics.json, ordered by run timestamp.

Usage: uv run python autoresearch/tasks/train_csg/plot_results.py
Writes autoresearch/tasks/train_csg/results_plot.png
"""

import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "results.tsv")
RUNS = os.path.join(HERE, "..", "..", "..", "runs", "jul13-phys-plausible")
OUT = os.path.join(HERE, "results_plot.png")


def load_tsv():
    rows = []
    with open(TSV) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                r["dice"] = float(r["dice"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def load_run_metrics():
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "metrics.json"))):
        try:
            with open(p) as f:
                m = json.load(f)
        except Exception:
            continue
        ts = os.path.basename(os.path.dirname(p)).rsplit("__", 1)[-1]
        m["_ts"] = int(ts) if ts.isdigit() else 0
        out.append(m)
    out.sort(key=lambda m: m["_ts"])
    return out


def main():
    rows = load_tsv()
    metrics = load_run_metrics()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: dice progress over experiments ---
    style = {"keep": ("o", "tab:green"), "discard": ("s", "tab:orange"),
             "crash": ("x", "tab:red")}
    seen = set()
    best = 0.0
    best_line = []
    for i, r in enumerate(rows):
        mk, c = style.get(r["status"], ("d", "gray"))
        lbl = r["status"] if r["status"] not in seen else None
        seen.add(r["status"])
        ax1.scatter(i, r["dice"], marker=mk, color=c, zorder=3, label=lbl)
        if r["status"] == "keep":
            best = max(best, r["dice"])
        best_line.append(best)
    ax1.plot(range(len(rows)), best_line, "-", color="tab:blue", alpha=0.6,
             label="running best (keep)")
    ax1.set_xlabel("experiment #")
    ax1.set_ylabel("hard dice")
    ax1.set_title("jul13-phys-plausible: dice per experiment (rrph, 15-min budget)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    # --- Right: physical plausibility per run ---
    if metrics:
        labels, plunge, fmax, marg, marg_s, dice = [], [], [], [], [], []
        for k, m in enumerate(metrics):
            labels.append(f"run{k + 1}")
            dice.append(m.get("hard_dice", 0.0))
            plunge.append(m.get("plunge_frac", np.nan))
            fmax.append(m.get("fcut_seq_max", np.nan))
            marg.append(min(m.get("fragile_margin_min", np.nan), 3.0))
            marg_s.append(min(m.get("fragile_margin_sched",
                                    m.get("fragile_margin_min", np.nan)), 3.0))
        x = np.arange(len(labels))
        w = 0.2
        ax2.bar(x - 1.5 * w, np.asarray(plunge) * 10, w,
                label="plunge frac x10", color="tab:red", alpha=0.8)
        ax2.bar(x - 0.5 * w, np.asarray(fmax) / 100.0, w,
                label="F max / cap(100 N)", color="tab:purple", alpha=0.8)
        ax2.bar(x + 0.5 * w, marg, w, label="fragile margin (raw)",
                color="tab:orange", alpha=0.8)
        ax2.bar(x + 1.5 * w, marg_s, w,
                label="fragile margin (feed-sched)", color="tab:green", alpha=0.8)
        ax2.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
        for k in x:
            ax2.text(k, -0.3, f"dice\n{dice[k]:.3f}", ha="center", fontsize=7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_title("physical plausibility per run (dashed = cap/safe line)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
