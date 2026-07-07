"""Plot jul6-spline-sweep autoresearch results.

Left panel: hard dice per experiment in chronological order (marker = keep/discard
status, color = target scenario parsed from the logged command).
Right panel: best dice per scenario, sweep method vs the delta baseline where one
was run, with the sphere 3-axis structural ceiling marked.

Reads results.tsv in this directory; writes results_plot.png.
"""
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = "results.tsv"
OUT = "results_plot.png"

# Exact 3-axis reachability ceilings (tool-disc max-filter; see idea.md)
CEILINGS = {"sphere": 0.848, "titan": 0.965, "rrph": 0.970,
            "extrusion": 0.648, "bowl": 0.342}


def scenario_of(cmd: str) -> str:
    m = re.search(r"--target-sdf-path\s+\S*?([\w-]+)\.npz", cmd)
    if m:
        return m.group(1).removesuffix("_hi")
    m = re.search(r"--target-shape\s+(\w+)", cmd)
    return m.group(1) if m else "sphere"


def method_of(cmd: str) -> str:
    return "sweep" if "--method sweep" in cmd else "delta"


def main():
    rows = []
    with open(RESULTS, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                rows.append({
                    "dice": float(r["dice"]),
                    "status": r["status"],
                    "desc": r["description"],
                    "scenario": scenario_of(r["command"]),
                    "method": method_of(r["command"]),
                })
            except (ValueError, KeyError):
                continue

    scenarios = []
    for r in rows:
        if r["scenario"] not in scenarios:
            scenarios.append(r["scenario"])
    cmap = plt.get_cmap("tab10")
    scen_color = {s: cmap(i % 10) for i, s in enumerate(scenarios)}
    status_marker = {"keep": "o", "discard": "x", "crash": "v", "regress": "D"}

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.7, 1]})

    # --- Left: chronological progress ---
    for i, r in enumerate(rows):
        filled = r["status"] != "discard"
        ax.scatter(i, r["dice"], color=scen_color[r["scenario"]],
                   marker=status_marker.get(r["status"], "o"),
                   s=80 if r["status"] == "keep" else 50,
                   edgecolors="black" if filled else None,
                   linewidths=0.8 if filled else 1.5, zorder=3)
        short = r["desc"][:34] + ("…" if len(r["desc"]) > 34 else "")
        above = i % 2 == 0
        ax.annotate(short, (i, r["dice"]), textcoords="offset points",
                    xytext=(2, 10 if above else -12), ha="left",
                    va="bottom" if above else "top", fontsize=6.5,
                    rotation=18 if above else -18)

    # running best per scenario (sweep runs only)
    for s in scenarios:
        xs, ys, best = [], [], -1.0
        for i, r in enumerate(rows):
            if r["scenario"] != s or r["method"] != "sweep":
                continue
            best = max(best, r["dice"])
            xs.append(i)
            ys.append(best)
        if len(xs) > 1:
            ax.plot(xs, ys, color=scen_color[s], lw=1.4, ls="--", alpha=0.7,
                    zorder=2)

    delta_rows = [r for r in rows if r["method"] == "delta"]
    if delta_rows:
        ax.axhline(delta_rows[0]["dice"], color="gray", lw=1.2, ls=":",
                   label=f"delta baseline (sphere) {delta_rows[0]['dice']:.3f}")
    ax.set_xlabel("experiment # (chronological)")
    ax.set_ylabel("hard-carve dice")
    ax.set_title("Spline-sweep campaign: dice per experiment")
    lo = min(r["dice"] for r in rows)
    hi = max(r["dice"] for r in rows)
    ax.set_ylim(lo - 0.04, hi + 0.05)
    ax.set_xlim(-0.6, len(rows) + 1.2)
    ax.grid(True, alpha=0.25, zorder=0)
    handles = [Line2D([0], [0], marker=m, color="gray", ls="none", ms=8, label=st)
               for st, m in status_marker.items()
               if any(r["status"] == st for r in rows)]
    handles += [Line2D([0], [0], marker="s", color=scen_color[s], ls="none",
                       ms=9, label=s) for s in scenarios]
    if delta_rows:
        handles.append(Line2D([0], [0], color="gray", ls=":", lw=1.2,
                              label="delta baseline"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    # --- Right: best per scenario, sweep vs delta ---
    width = 0.38
    for i, s in enumerate(scenarios):
        best_sweep = max((r["dice"] for r in rows
                          if r["scenario"] == s and r["method"] == "sweep"),
                         default=None)
        best_delta = max((r["dice"] for r in rows
                          if r["scenario"] == s and r["method"] == "delta"),
                         default=None)
        if best_delta is not None:
            ax2.bar(i - width / 2, best_delta, width, color="lightgray",
                    edgecolor="black", label="delta" if i == 0 else None)
            ax2.text(i - width / 2, best_delta + 0.008, f"{best_delta:.3f}",
                     ha="center", fontsize=8)
        if best_sweep is not None:
            xoff = i + width / 2 if best_delta is not None else i
            ax2.bar(xoff, best_sweep, width, color=scen_color[s],
                    edgecolor="black", label="sweep" if i == 0 else None)
            ax2.text(xoff, best_sweep + 0.008, f"{best_sweep:.3f}",
                     ha="center", fontsize=8)
    for s, c in CEILINGS.items():
        if s not in scenarios:
            continue
        i = scenarios.index(s)
        ax2.hlines(c, i - 0.55, i + 0.55, color="crimson", lw=1.4, ls="--")
        ax2.text(i - 0.55, c - 0.03, "3-axis\nceiling",
                 ha="left", va="top", fontsize=7, color="crimson")
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels(scenarios)
    ax2.set_ylabel("best hard-carve dice")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Best per scenario: sweep vs delta")
    ax2.grid(True, axis="y", alpha=0.25, zorder=0)
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT} ({len(rows)} experiments, {len(scenarios)} scenarios)")


if __name__ == "__main__":
    main()
