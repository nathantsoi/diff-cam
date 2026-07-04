"""Plot the autoresearch results.tsv: progress over experiments + per-shape k-sweep.

Reads autoresearch/tasks/train_csg/results.tsv (tab-separated; header + rows):
    commit  dice  memory_gb  status  description  command

Produces autoresearch/tasks/train_csg/results_plot.png with two panels:
  A. dice vs experiment order, with keep/discard/crash distinguished + running-best line.
  B. HARD dice per shape across the k sweep (the headline lever of this run):
     grouped bars, one cluster per target-shape, one bar per k value.
"""
import os
import re
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "results.tsv")
OUT = os.path.join(HERE, "results_plot.png")

STATUS_STYLE = {
    "keep":    {"color": "#2e7d32", "marker": "o", "label": "keep"},
    "discard": {"color": "#c62828", "marker": "x", "label": "discard"},
    "crash":   {"color": "#9e9e9e", "marker": "s", "label": "crash"},
}


def parse_cmd(cmd):
    """Extract scenario flags from a run command string."""
    def flag(name, cast=str, default=None):
        m = re.search(rf"--{re.escape(name)}\s+(\S+)", cmd)
        return cast(m.group(1)) if m else default
    shape = flag("target-shape", default="?")
    init = flag("init-mode", default="random")
    k = flag("k-init", cast=float, default=10.0)
    m = re.search(r"--stock-size-in\s+(\S+)\s+(\S+)\s+(\S+)", cmd)
    stock = f"{m.group(1)}x{m.group(2)}x{m.group(3)}" if m else "?"
    return shape, init, stock, k


def main():
    rows = []
    with open(TSV, newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            if not r.get("command"):
                continue
            try:
                dice = float(r["dice"])
            except (ValueError, KeyError):
                continue
            shape, init, stock, k = parse_cmd(r["command"])
            rows.append({
                "dice": dice,
                "status": r.get("status", ""),
                "desc": r.get("description", ""),
                "command": r["command"],
                "shape": shape,
                "init": init,
                "stock": stock,
                "k": k,
            })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel A: dice vs experiment order + running best ---
    order = list(range(len(rows)))
    for st, style in STATUS_STYLE.items():
        xs = [i for i, r in enumerate(rows) if r["status"] == st]
        ys = [rows[i]["dice"] for i in xs]
        if xs:
            ax1.scatter(xs, ys, c=style["color"], marker=style["marker"],
                        s=55, label=style["label"], zorder=3)
    best = -1.0
    best_line = []
    for r in rows:
        if r["status"] != "crash" and r["dice"] > 0:
            best = max(best, r["dice"])
        best_line.append(best if best > 0 else None)
    ax1.plot(order, best_line, color="#1565c0", lw=1.8, ls="--",
             label="running best", zorder=2)
    ax1.set_xlabel("experiment order")
    ax1.set_ylabel("HARD dice")
    ax1.set_title("Progress over experiments")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.35, 0.92)

    # --- Panel B: HARD dice per shape across the k sweep (grouped bars) ---
    # Collect, per shape, the best dice at each k value tested.
    shape_order = ["sphere", "cylinder", "box", "pyramid"]
    shapes_present = [s for s in shape_order if any(r["shape"] == s for r in rows)]
    # k values in ascending order, bucketed to integers for labeling.
    k_vals = sorted({int(round(r["k"])) for r in rows if r["status"] != "crash"})
    # One stable color per k value (low k = blue, high k = red).
    cmap = plt.cm.viridis
    k_color = {kv: cmap(i / max(1, len(k_vals) - 1)) for i, kv in enumerate(k_vals)}
    import numpy as np
    n_k = len(k_vals)
    n_sh = len(shapes_present)
    bar_w = 0.8 / max(1, n_k)
    xpos = np.arange(n_sh)
    for ki, kv in enumerate(k_vals):
        ys = []
        for sh in shapes_present:
            dices = [r["dice"] for r in rows
                     if r["shape"] == sh and r["status"] != "crash"
                     and int(round(r["k"])) == kv]
            ys.append(max(dices) if dices else 0.0)
        offs = (ki - (n_k - 1) / 2.0) * bar_w
        bars = ax2.bar(xpos + offs, ys, bar_w, color=k_color[kv],
                       edgecolor="black", lw=0.4, label=f"k={kv}")
        for b, v in zip(bars, ys):
            if v > 0:
                ax2.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(list(xpos))
    ax2.set_xticklabels(shapes_present, fontsize=9)
    ax2.set_ylabel("best HARD dice")
    ax2.set_title("HARD dice per shape across the k sweep")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0.35, 0.92)
    ax2.legend(loc="lower right", fontsize=8, ncol=2)

    fig.suptitle("ar-agd/jul3-hard-carve-gap: the k lever (stable smooth_max unlocks high-k)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130)
    print(f"[plot] saved {OUT} ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
