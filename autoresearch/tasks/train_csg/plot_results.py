"""Plot the autoresearch results.tsv: progress over experiments + per-scenario dice.

Reads autoresearch/tasks/train_csg/results.tsv (tab-separated; header + rows):
    commit  dice  memory_gb  status  description  command

Produces autoresearch/tasks/train_csg/results_plot.png with two panels:
  A. dice vs experiment order, with keep/discard/crash distinguished + running-best line.
  B. best dice per (target-shape, init-mode) machining scenario, parsed from the
     command column. Shows where the uniform raster_fine init wins vs random.
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
    m = re.search(r"--stock-size-in\s+(\S+)\s+(\S+)\s+(\S+)", cmd)
    stock = f"{m.group(1)}x{m.group(2)}x{m.group(3)}" if m else "?"
    return shape, init, stock


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
            shape, init, stock = parse_cmd(r["command"])
            rows.append({
                "dice": dice,
                "status": r.get("status", ""),
                "desc": r.get("description", ""),
                "command": r["command"],
                "shape": shape,
                "init": init,
                "stock": stock,
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
    ax1.set_ylabel("dice")
    ax1.set_title("Progress over experiments")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.55, 0.92)

    # --- Panel B: best dice per (shape, init) ---
    groups = {}
    for r in rows:
        if r["status"] == "crash" or r["dice"] <= 0:
            continue
        key = (r["shape"], r["init"])
        groups.setdefault(key, []).append(r["dice"])
    labels = []
    bests = []
    colors = []
    color_map = {"raster_fine": "#2e7d32", "random": "#1976d2"}
    for (shape, init), dices in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        labels.append(f"{shape}\n{init}")
        bests.append(max(dices))
        colors.append(color_map.get(init, "#9e9e9e"))
    x = range(len(labels))
    bars = ax2.bar(x, bests, color=colors, edgecolor="black", lw=0.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("best dice")
    ax2.set_title("Best dice per machining scenario (shape x init)")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0.55, 0.92)
    for b, v in zip(bars, bests):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color="#2e7d32", label="raster_fine (uniform)"),
                        Patch(color="#1976d2", label="random")],
               loc="lower right", fontsize=9)

    fig.suptitle("ar-agd/jul1-uniform-toolpath: uniform CNC raster init (raster_fine)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130)
    print(f"[plot] saved {OUT} ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
