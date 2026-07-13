#!/usr/bin/env python3
"""Collect metrics from runs/jul10-gouge/*/ into a table + append to results.tsv.

Reads each run dir's metrics.json (final/best-checkpoint metrics) and args.json
(config). Prints a sorted table and appends new rows to results.tsv.
results.tsv is untracked (gitignored). Tab-separated, no tabs in the command
field, no log redirect in the command field.
"""
import json, os, glob, sys

BATCH = "runs/jul10-gouge"
TSV = "autoresearch/tasks/train_csg/results.tsv"
COMMIT = os.popen("git rev-parse --short HEAD").read().strip()

rows = []
for d in sorted(glob.glob(os.path.join(BATCH, "*/"))):
    mp = os.path.join(d, "metrics.json")
    ap = os.path.join(d, "args.json")
    if not (os.path.exists(mp) and os.path.exists(ap)):
        continue
    with open(mp) as f:
        m = json.load(f)
    with open(ap) as f:
        a = json.load(f)
    rows.append((d, a, m))

# print table
hdr = ["run", "margin", "w_tg", "seed", "hard_dice", "soft_dice", "gouge",
       "loss_tg", "air_frac", "brk", "dice_imp", "secs", "vram_mb"]
print("  ".join(f"{h:>9}" for h in hdr))
for d, a, m in rows:
    name = os.path.basename(d.rstrip("/"))
    vals = [name[:18],
            f"{a.get('tool_gouge_margin_mm',0)}",
            f"{a.get('w_tool_gouge',0)}",
            f"{a.get('seed',1)}",
            f"{m.get('hard_dice',0):.4f}",
            f"{m.get('dice',0):.4f}",
            f"{m.get('gouge',0):.4f}",
            f"{m.get('loss_tool_gouge',0):.3f}",
            f"{m.get('air_time_frac',0):.3f}",
            f"{m.get('break_prob_any',0):.2f}",
            f"{m.get('dice_improvement',0):.3f}",
            f"{m.get('training_seconds',0):.0f}",
            f"{m.get('peak_vram_mb',0):.0f}"]
    print("  ".join(f"{v:>9}" for v in vals))

# append new rows to results.tsv
existing = set()
if os.path.exists(TSV):
    with open(TSV) as f:
        for line in f:
            if line.strip():
                existing.add(line.split("\t")[5])  # command col

with open(TSV, "a") as f:
    for d, a, m in rows:
        name = os.path.basename(d.rstrip("/"))
        # rebuild a clean command (no redirect, no tabs)
        parts = ["uv run python scripts/run_pipeline.py --stages train",
                 f"--iters {a.get('iters',5000)}",
                 f"--max-steps {a.get('max_steps',128)}",
                 "--stock-size-in 1 1 1 --voxel-size-mm 0.5",
                 f"--target-shape {a.get('target_shape','sphere')}",
                 f"--target-radius-mm {a.get('target_radius_mm',11.43)}",
                 "--post haas --eval-freq 10",
                 f"--runs-subdir jul10-gouge",
                 f"--seed {a.get('seed',1)}"]
        if a.get("init_mode") and a["init_mode"] != "random":
            parts.append(f"--init-mode {a['init_mode']}")
        if a.get("k_anneal"):
            parts.append(f"--k-anneal --k-init {a.get('k_init',10)} --k-final {a.get('k_final',10)}")
        if a.get("loss_shift", 0) != 0:
            parts.append(f"--loss-shift {a.get('loss_shift')}")
        if a.get("best_on_hard"):
            parts.append("--best-on-hard")
        if a.get("w_tool_gouge", 0) != 0:
            parts.append(f"--w-tool-gouge {a.get('w_tool_gouge')}")
        if a.get("tool_gouge_margin_mm", 0) != 0:
            parts.append(f"--tool-gouge-margin-mm {a.get('tool_gouge_margin_mm')}")
        if a.get("w_tool_gouge_warmup_frac", 0) != 0:
            parts.append(f"--w-tool-gouge-warmup-frac {a.get('w_tool_gouge_warmup_frac')}")
        cmd = " ".join(parts)
        if cmd in existing:
            continue
        desc = (f"{name} hd={m.get('hard_dice',0):.4f} sd={m.get('dice',0):.4f} "
                f"gouge={m.get('gouge',0):.4f} ltg={m.get('loss_tool_gouge',0):.3f} "
                f"air={m.get('air_time_frac',0):.3f} brk={m.get('break_prob_any',0):.2f} "
                f"imp={m.get('dice_improvement',0):.3f} "
                f"m={a.get('tool_gouge_margin_mm',0)} tg={a.get('w_tool_gouge',0)} "
                f"s={a.get('seed',1)} {a.get('init_mode','random')}")
        mem = f"{m.get('peak_vram_mb',0)/1024:.1f}"
        f.write(f"{COMMIT}\t{m.get('hard_dice',0):.4f}\t{mem}\tOK\t{desc}\t{cmd}\n")
print(f"\nappended {len(rows)} runs to {TSV}")
