#!/bin/bash
# Wave 19: probe the tool-gouge MARGIN (new shape-agnostic lever) on the sphere,
# where w_tool_gouge was INACTIVE (loss_tool_gouge=0 at margin=0 -- the tool is
# tangent at sampled midpoints yet the boolean union still over-erodes, gouge
# ~620). Hypothesis: sph gouge is overlapping-tangent-capsules biting into the
# convex part at pass seams. A positive margin lifts the tool center `margin` mm
# beyond mere tangency (barrier fires when target_sdf(center) < r_tool + margin),
# so the union of capsules stays tangent-only -> no seam gouge, at the cost of a
# little uncut residual. Shape-agnostic (target_sdf only). Confirmed in smoke:
# sph tg8 margin1mm -> loss_tool_gouge 0.0 -> 1.71 (barrier now active).
#
# 6 sph doses {0,0.5,1,2,4,8}mm define the dose-response (margin0 reproduces the
# inactive barrier baseline; margin8mm is the upper-bound collapse point). + 2
# cross-shape sanity: pyr tg8 margin2 (does margin break pyr's clean w18 gouge
# fix?) and bowl tg4 margin2 (bowl gouge 962 is the WORST of all shapes).
# 8 runs, seed=1, 5000 iters, canonical K70. Single reps (probe); confirm the
# winning margin x3 in wave 20 per the >=3-reps variance rule.
# Usage: bash launch_wave19.sh [first_gpu]
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10 --runs-subdir jul8-multidepth"
K70="--k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7"
MD5="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
SPH="--target-shape sphere --target-radius-mm 11.43"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
BOWL="--target-shape sphere_bowl --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <iters> <args...>
  local gpu="$1"; local name="$2"; local iters="$3"; shift 3
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters "$iters" --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave19] GPU$gpu -> $name (iters=$iters)"
}

# sph dose-response: tg8, margin {0,0.5,1,2,4,8}mm. margin0 = inactive-barrier
# baseline (should reproduce ~0.79 hard, ~620 gouge, loss_tool_gouge~0).
launch 0 w19_sph_tg8_m0    5000 $SPH $MD5 $K70 --best-on-hard --w-tool-gouge 8.0 --tool-gouge-margin-mm 0.0
launch 1 w19_sph_tg8_m0p5  5000 $SPH $MD5 $K70 --best-on-hard --w-tool-gouge 8.0 --tool-gouge-margin-mm 0.5
launch 2 w19_sph_tg8_m1    5000 $SPH $MD5 $K70 --best-on-hard --w-tool-gouge 8.0 --tool-gouge-margin-mm 1.0
launch 3 w19_sph_tg8_m2    5000 $SPH $MD5 $K70 --best-on-hard --w-tool-gouge 8.0 --tool-gouge-margin-mm 2.0
launch 4 w19_sph_tg8_m4    5000 $SPH $MD5 $K70 --best-on-hard --w-tool-gouge 8.0 --tool-gouge-margin-mm 4.0
launch 5 w19_sph_tg8_m8    5000 $SPH $MD5 $K70 --best-on-hard --w-tool-gouge 8.0 --tool-gouge-margin-mm 8.0
# pyr sanity: w18 pyr tg8 margin0 = clean gouge fix (509->~75, dice 0.700). Does
# margin2 break it (over-lift -> residual -> dice drop)?
launch 6 w19_pyr_tg8_m2    5000 $PYR $MD5 $K70 --best-on-hard --w-tool-gouge 8.0 --tool-gouge-margin-mm 2.0
# bowl: worst gouge of all shapes (962). w17 tg8 HURT bowl (gouge 1942). Try the
# milder tg4 + margin2 -- does the margin relieve bowl's gouge without the dice
# collapse tg8 caused?
launch 7 w19_bowl_tg4_m2   5000 $BOWL $MD5 $K70 --best-on-hard --w-gouge 6.0 --w-tool-gouge 4.0 --tool-gouge-margin-mm 2.0

echo "[wave19] all 8 launched. Logs in $D/run_w19_*.log"
