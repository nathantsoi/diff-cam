import json, glob, os, pickle
rows = pickle.load(open("/tmp/w4_rows.pkl","rb"))  # [(name, dir, metrics)]
COMMIT = os.popen("git rev-parse --short HEAD").read().strip()
out = []
for name, d, m in rows:
    cmd = open(os.path.join("/home/ntsoi/papers/icra26-diffcam/diff-cam", d, "reproduce_command.sh")).read().strip()
    # strip leading "python scripts/run_pipeline.py ..." or "uv run ..." to get args
    # keep the full args portion after the script path
    import re
    # reproduce_command typically: python algorithms/train_csg.py --iters ...
    cmd = cmd.replace("\n"," ").strip()
    hd = m["hard_dice"]; sd = m["dice"]
    air = m["air_time"]; brk = m["break_prob_any"]; impr = m["dice_improvement"]
    resid = m["residual"]; gouge = m["gouge"]
    desc = (f"hard_dice={hd:.6f} soft_dice={sd:.6f} WAVE4 {name} "
            f"(air={air} t={m['total_time']} brk={brk:.6f} impr={impr:.6f} "
            f"resid={resid} gouge={gouge})")
    out.append((name, COMMIT, sd, 0.0, "keep", desc, cmd))
# order: sphere first then cylinder, by hdice desc within group
def key(r):
    n=r[0]
    grp = 0 if n.startswith("w4_sph") else 1
    return (grp, -r[1] and 0)  # keep order stable
order = ["w4_sph_wr5","w4_sph_wr3_rev24","w4_sph_wr10","w4_sph_wr3_lr1e2",
         "w4_sph_wr5_wg8","w4_sph_wr5_shift","w4_cyl_md_wr3","w4_cyl_rand"]
by = {r[0]:r for r in out}
with open("results.tsv","a") as f:
    for n in order:
        r = by[n]
        f.write("\t".join([r[1], f"{r[2]:.6f}", f"{r[3]}", r[4], r[5], r[6]])+"\n")
print("appended", len(order), "rows; commit", COMMIT)
