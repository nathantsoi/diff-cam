import json, os, subprocess, re
TSV="autoresearch/tasks/train_csg/results.tsv"
TAG="jul8-multidepth"
os.makedirs(f"runs/{TAG}",exist_ok=True)
COMMIT=subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
LD="autoresearch/tasks/train_csg"
# (name, log, command-desc)
runs=[
 ("w2_md_default","run_w2_md_default.log","multidepth feed10 revs~3 (default)"),
 ("w2_md_feed60","run_w2_md_feed60.log","multidepth feed60 revs12"),
 ("w2_md_feed60_tight","run_w2_md_feed60_tight.log","multidepth feed60 revs12 margin0.01"),
 ("w2_md_feed60_rev24","run_w2_md_feed60_rev24.log","multidepth feed60 revs24"),
 ("w2_md_feed60_lvl10","run_w2_md_feed60_lvl10.log","multidepth feed60 revs12 levels10"),
 ("w2_md_feed60_tg","run_w2_md_feed60_tg.log","multidepth feed60 revs12 w_tool_gouge1.0"),
 ("w2_md_feed60_shift","run_w2_md_feed60_shift.log","multidepth feed60 revs12 loss_shift3.0"),
 ("w2_raster_tg","run_w2_raster_tg.log","raster + w_tool_gouge1.0 (de-gouge leader)"),
]
rows=[]
for name,log,desc in runs:
    logp=os.path.join(LD,log)
    rd=None
    with open(logp) as f:
        for line in f:
            if "writing outputs to runs/" in line:
                rd=line.split("writing outputs to runs/")[1].strip().split()[0]
                rd="runs/"+rd if not rd.startswith("runs/") else rd
                break
    cmd=re.search(r"python3 -m algorithms\.train_csg (.+?) ===", open(logp).read())
    cmdstr=cmd.group(1).strip() if cmd else ""
    # find metrics (may be moved already)
    cand=[rd, f"runs/{TAG}/{os.path.basename(rd)}"] if rd else []
    mp=None
    for c in cand:
        if c and os.path.exists(os.path.join(c,"metrics.json")): mp=os.path.join(c,"metrics.json"); rd=c; break
    if not mp:
        print(f"[{name}] NO metrics"); continue
    m=json.load(open(mp))
    dice=m["dice"]; hdice=m["hard_dice"]; mem=m.get("peak_vram_mb",0.0)/1024.0
    air=m.get("air_time","NA"); ttime=m.get("total_time","NA"); brk=m.get("break_prob_any","NA"); impr=m.get("dice_improvement","NA")
    resid=m.get("residual","NA"); gouge=m.get("gouge","NA")
    full=f"hard_dice={hdice} soft_dice={dice} WAVE2 {desc} (air={air} t={ttime} brk={brk} impr={impr} resid={resid} gouge={gouge})".replace("\t"," ")
    rows.append((COMMIT,dice,round(mem,1),"keep",full,cmdstr.replace("\t"," ")))
    print(f"[collect] {name} -> hdice={hdice} dice={dice} gouge={gouge} resid={resid}")
with open(TSV,"a") as f:
    for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
print("appended",len(rows),"rows")
