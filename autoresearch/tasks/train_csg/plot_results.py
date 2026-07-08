"""Plot the jul6-traj-quality research results from results.tsv + metrics.json.

Three panels:
  1. Progress over experiments: dice vs experiment order, kept/discard/crash
     distinguished, with a running-best line.
  2. Generality across scenarios: best dice per (shape, family) for the
     1in stock — the payoff of varying the scenario and the method.
  3. Trajectory-quality panel: air_time / total_time / break_prob_any
     alongside dice for kept runs (deployable cost visibility).

Robust: skips crash/0.0 rows; takes a commit's best when it appears >1x.
"""
import csv
import json
import os
import re
import glob
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "results.tsv")
OUT = os.path.join(HERE, "results_plot.png")
RUNS = os.path.join(HERE, "..", "..", "..", "runs")


def family(cmd):
    """Map a command to a method family label + (shape, stock)."""
    vals = re.findall(r"--target[-_]shape\s+(\S+)", cmd)
    shape = vals[-1] if vals else "?"
    vals = re.findall(r"--stock[-_]size[-_]in (\S+ \S+ \S+)", cmd)
    stock = vals[-1] if vals else "?"
    # method family: baseline / +w_air_time / +fcalib / etc.
    # use LAST occurrence (argparse takes the last of duplicate flags)
    parts = []
    def lastval(kre):
        vs = re.findall(rf"{kre}\s+(\S+)", cmd)
        return vs[-1] if vs else None
    wat = lastval(r"--w[-_]air[-_]time")
    if wat is not None and float(wat) != 0.0:
        parts.append("w_air")
    wt = lastval(r"--w[-_]time")
    if wt is not None and float(wt) != 0.0:
        parts.append("w_time")
    wb = lastval(r"--w[-_]break")
    if wb is not None and float(wb) != 0.0:
        parts.append("w_break")
    bwa = lastval(r"--best[-_]w[-_]airtime")
    if bwa is not None and float(bwa) != 0.0:
        parts.append("bwa")
    fref = lastval(r"--f[-_]ref")
    if fref and float(fref) < 50:
        parts.append(f"fref{fref}")
    label = "+".join(parts) if parts else "baseline"
    return label, shape, stock


def load_metrics_for_cmd(target_cmd):
    """Find a runs/<dir>/metrics.json whose reproduce_command.sh matches target_cmd."""
    # distinctive flags as signature — use LAST occurrence (argparse semantics)
    sig_keys = ["--target[-_]shape", "--w[-_]air[-_]time", "--f[-_]ref", "--f[-_]max",
                "--w[-_]time", "--w[-_]break", "--best[-_]w[-_]airtime", "--init[-_]mode", "--seed",
                "--max[-_]steps", "--w[-_]len"]
    sig_vals = []
    for k in sig_keys:
        vs = re.findall(rf"{k}\s+(\S+)", target_cmd)
        if vs:
            sig_vals.append((k, vs[-1]))
    # search both runs/ and runs/jul6-traj-quality/
    cand = sorted(glob.glob(os.path.join(RUNS, "*")), reverse=True) + \
           sorted(glob.glob(os.path.join(RUNS, "jul6-traj-quality", "*")), reverse=True)
    best = None
    best_score = -1
    for rd in cand:
        mj = os.path.join(rd, "metrics.json")
        rc = os.path.join(rd, "reproduce_command.sh")
        if not (os.path.exists(mj) and os.path.exists(rc)):
            continue
        try:
            rcmd = open(rc).read()
        except Exception:
            continue
        score = 0
        for k, v in sig_vals:
            if re.search(rf"{k}\s+{re.escape(v)}\b", rcmd):
                score += 1
        if score > best_score:
            best_score = score
            try:
                best = json.load(open(mj))
            except Exception:
                pass
    if best_score >= max(3, len(sig_vals) // 2):
        return best
    return None


def main():
    rows = []
    with open(TSV, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                d = float(r["dice"])
            except (ValueError, KeyError):
                continue
            rows.append({"dice": d, "status": r["status"], "cmd": r.get("command", ""),
                         "desc": r.get("description", ""), "commit": r.get("commit", "")})

    # NOTE: this run varies only CLI params (no code change), so every row
    # shares commit 1a6fc67. Do NOT dedup by commit — treat each row as its own
    # experiment. (Dedup only if a commit truly repeats.)
    seen = set()
    dedup = []
    for r in rows:
        key = (r["commit"], r["desc"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)

    fig = plt.figure(figsize=(17, 13))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    # Panel 1: progress over experiments
    kept = [(i, r["dice"]) for i, r in enumerate(dedup) if r["status"] == "keep"]
    disc = [(i, r["dice"]) for i, r in enumerate(dedup) if r["status"] == "discard"]
    crash = [(i, r["dice"]) for i, r in enumerate(dedup) if r["status"] == "crash"]
    if kept:
        ax1.scatter([p[0] for p in kept], [p[1] for p in kept], c="green", s=30, label="keep", zorder=3)
    if disc:
        ax1.scatter([p[0] for p in disc], [p[1] for p in disc], c="red", s=20, marker="x", label="discard", zorder=3)
    if crash:
        ax1.scatter([p[0] for p in crash], [p[1] for p in crash], c="gray", s=20, marker="v", label="crash", zorder=3)
    best = 0.0
    rb_x, rb_y = [], []
    for i, r in enumerate(dedup):
        if r["status"] == "keep" and r["dice"] > best:
            best = r["dice"]
        rb_x.append(i)
        rb_y.append(best)
    ax1.plot(rb_x, rb_y, "k-", lw=1.6, label="running best", zorder=2)
    ax1.set_xlabel("experiment order")
    ax1.set_ylabel("dice (soft, proven metric)")
    ax1.set_title("jul6-traj-quality: progress over experiments")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Panel 2: generality — best dice per (shape, family) for 1in stock
    best_per = defaultdict(lambda: 0.0)
    for r in dedup:
        if r["dice"] <= 0.0 or r["status"] != "keep":
            continue
        fam, shape, stock = family(r["cmd"])
        if stock.startswith("1.0 1.0 1.0") or stock.startswith("1 1 1"):
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
    ax2.set_ylabel("best soft dice")
    ax2.set_title("Generality: best dice per shape × method-family (1in stock)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: trajectory-quality — air_time / total_time / break_prob_any vs dice
    tq = []
    for r in dedup:
        if r["status"] != "keep":
            continue
        m = load_metrics_for_cmd(r["cmd"])
        if m is None:
            continue
        fam, shape, _ = family(r["cmd"])
        tq.append({
            "label": f"{shape}|{fam}",
            "dice": float(m.get("dice", 0)),
            "air_time": float(m.get("air_time", 0)),
            "total_time": float(m.get("total_time", 0)),
            "break_prob_any": float(m.get("break_prob_any", 0)),
            "broken": float(m.get("broken", 0)),
        })
    if not tq:
        ax3.text(0.5, 0.5,
                 "No metrics.json matched — traj-quality panel empty\n"
                 "(measures may be all-zero / uncalibrated this run)",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=11)
        ax3.set_title("Trajectory quality (air_time / total_time / break vs dice)")
    else:
        labels = [t["label"] for t in tq]
        xs = np.arange(len(tq))
        w = 0.2
        ax3.bar(xs - 1.5*w, [t["dice"] for t in tq], w, label="dice", color="steelblue")
        ax3.bar(xs - 0.5*w, [t["air_time"] for t in tq], w, label="air_time(s)", color="orange")
        ax3.bar(xs + 0.5*w, [t["total_time"] for t in tq], w, label="total_time(s)", color="green")
        ax3.bar(xs + 1.5*w, [t["break_prob_any"] for t in tq], w, label="break_prob", color="red")
        ax3.set_xticks(xs)
        ax3.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax3.set_title("Trajectory quality: deployable cost (air/total/break) alongside dice")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")
        # caption
        all_zero = all(t["air_time"] == 0 and t["break_prob_any"] == 0 for t in tq)
        if all_zero:
            ax3.text(0.01, 0.95, "all air_time/break = 0 (uncalibrated)",
                     transform=ax3.transAxes, fontsize=8, color="gray", va="top")

    fig.suptitle("autoresearch jul6-traj-quality: dice + trajectory-quality results", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT} ({len(dedup)} experiments, {len(shapes)} shapes, {len(fams)} families, {len(tq)} tq-rows)")


if __name__ == "__main__":
    main()
