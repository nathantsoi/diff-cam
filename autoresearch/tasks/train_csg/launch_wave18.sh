#!/bin/bash
# Wave 18: confirm pyr w_tool_gouge win + probe sph gouge via sharper k + revert bowl/hole.
# Wave-17 finding (memory best-on-hard-gouge-tradeoff.md): w_tool_gouge is
# SHAPE-DEPENDENT.
#   pyr: CLEAN WIN -- hard 0.674->0.720, gouge 509->53 (tg fixed tool-penetration).
#   sph: hard 0.777->0.818 but gouge 572->585 UNRESOLVED. loss_tool_gouge=0 yet
#        gouge=585 => sph gouge is soft-union OVER-EROSION, not tool penetration;
#        w_tool_gouge cannot reach it. Lever: sharper k_final (less soft over-
#        erosion -> smaller soft/hard gouge gap).
#   bowl: HURT (0.568->0.489, gouge 921->2011). Revert to w16.
#   hole: COLLAPSED (0.247->0.168). Revert to w16 soft selector.
# 8 runs, seed=1, 5000 iters. 3 reps for pyr+sph (variance rule, delta<2σ=0.034
# pyr is noise); bowl/hole single-run revert (just to re-land w16 numbers).
# Usage: bash launch_wave18.sh [first_gpu]
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10 --runs-subdir jul8-multidepth"
K70="--k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7"
K150="--k-anneal --k-init 2.0 --k-final 150.0 --loss-shift 0.7"
MD5="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
HOLEBASE="--init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 4.0"
SPH="--target-shape sphere --target-radius-mm 11.43"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
BOWL="--target-shape sphere_bowl --target-radius-mm 11.43 --target-sub-radius-mm 9.525"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <iters> <args...>
  local gpu="$1"; local name="$2"; local iters="$3"; shift 3
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters "$iters" --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave18] GPU$gpu -> $name (iters=$iters)"
}

# pyr: confirm w_tool_gouge=8 win (w17: hard 0.714, gouge 57) x3 reps.
launch 0 w18_pyr_tg8_r1 5000 $PYR $MD5 $K70 --best-on-hard --w-tool-gouge 8.0
launch 1 w18_pyr_tg8_r2 5000 $PYR $MD5 $K70 --best-on-hard --w-tool-gouge 8.0
launch 2 w18_pyr_tg8_r3 5000 $PYR $MD5 $K70 --best-on-hard --w-tool-gouge 8.0
# sph: sharper k_final=150 + best_on_hard, NO tool_gouge (tg inactive for sph
# anyway). Tests whether less soft over-erosion drops gouge while keeping high
# hard dice. x3 reps.
launch 3 w18_sph_k150_r1 5000 $SPH $MD5 $K150 --best-on-hard
launch 4 w18_sph_k150_r2 5000 $SPH $MD5 $K150 --best-on-hard
launch 5 w18_sph_k150_r3 5000 $SPH $MD5 $K150 --best-on-hard
# bowl: revert to w16 (best_on_hard, w_gouge 6, no tool_gouge -> 0.568).
launch 6 w18_bowl_rev 5000 $BOWL $MD5 $K70 --best-on-hard --w-gouge 6.0
# hole: revert to w16 soft selector (no best_on_hard -> 0.247).
launch 7 w18_hole_rev 5000 $HOLE $HOLEBASE $K70

echo "[wave18] all 8 launched. Logs in $D/run_w18_*.log"
