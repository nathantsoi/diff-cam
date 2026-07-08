import json, os, subprocess
TSV = "autoresearch/tasks/train_csg/results.tsv"
TAG = "jul8-multidepth"
os.makedirs(f"runs/{TAG}", exist_ok=True)
COMMIT = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()

# (name, log, rundir, init_mode, command)
runs = [
 ("baseline","run_baseline.log","runs/CamEnvDiff-v0__train_csg__1__1783544918478","random",
  "uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --grad-clip 0.5 --eval-freq 10 --init-mode random"),
 ("zlayer","run_zlayer.log","runs/CamEnvDiff-v0__train_csg__1__1783545047470","zlayer",
  "uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --grad-clip 0.5 --eval-freq 10 --init-mode zlayer --zlayer-revs 16 --zlayer-margin 0.01"),
 ("shell","run_shell.log","runs/CamEnvDiff-v0__train_csg__1__1783545124190","shell",
  "uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --grad-clip 0.5 --eval-freq 10 --init-mode shell"),
 ("raster_fine","run_rf.log","runs/CamEnvDiff-v0__train_csg__1__1783545124123","raster_fine",
  "uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --grad-clip 0.5 --eval-freq 10 --init-mode raster_fine"),
 ("raster_fine_wide","run_rfw.log","runs/CamEnvDiff-v0__train_csg__1__1783545047503","raster_fine_wide",
  "uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --grad-clip 0.5 --eval-freq 10 --init-mode raster_fine_wide"),
 ("raster","run_raster.log","runs/CamEnvDiff-v0__train_csg__1__1783545124152","raster",
  "uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --grad-clip 0.5 --eval-freq 10 --init-mode raster"),
 ("spiral","run_spiral.log","runs/CamEnvDiff-v0__train_csg__1__1783545124139","spiral",
  "uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --grad-clip 0.5 --eval-freq 10 --init-mode spiral"),
 ("zlayer_dense","run_zlayer_dense.log","runs/CamEnvDiff-v0__train_csg__1__1783545124233","zlayer",
  "uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --dt 0.45 --grad-clip 0.5 --eval-freq 10 --init-mode zlayer --zlayer-revs 24 --zlayer-osc 9 --zlayer-margin 0.01"),
]
LD = "autoresearch/tasks/train_csg"
rows=[]
for name,log,rd,im,cmd in runs:
    logp=os.path.join(LD,log)
    m=json.load(open(os.path.join(rd,"metrics.json")))
    dice=m.get("dice",0.0); hdice=m.get("hard_dice",0.0)
    vram=m.get("peak_vram_mb",0.0); mem=vram/1024.0
    air=m.get("air_time","NA"); ttime=m.get("total_time","NA"); brk=m.get("break_prob_any","NA"); impr=m.get("dice_improvement","NA")
    resid=m.get("residual","NA"); gouge=m.get("gouge","NA")
    desc=f"hard_dice={hdice} soft_dice={dice} WAVE1 init={im} (air={air} t={ttime} brk={brk} impr={impr} resid={resid} gouge={gouge})"
    desc=desc.replace("\t"," ")
    cmd=cmd.replace("\t"," ")
    rows.append((COMMIT,dice,round(mem,1),"keep",desc,cmd))
    # move run dir into runs/<TAG>/
    dest=f"runs/{TAG}/{os.path.basename(rd)}"
    if os.path.isdir(rd) and not os.path.isdir(dest):
        subprocess.run(["mv",rd,dest],check=False)
    print(f"[collect] {name} -> hdice={hdice} dice={dice} mem={mem:.1f}GB (moved->{dest})")

with open(TSV,"a") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r)+"\n")
print("appended",len(rows),"rows to",TSV)
