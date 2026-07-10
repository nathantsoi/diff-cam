#!/bin/bash
# Wave 12: 8-wide. THE UNIFIED CANONICAL METHOD test.
# Locked recipe (wave 9-11): k-anneal k 2->70 + loss_shift 0.7. Essential
# companions: loss_shift 0.7 (k-anneal collapses without it on concave); the
# low->high ANNEAL schedule (fixed high-k gouges catastrophically, wave11
# bowl kfix50 gouge 2476).
# (A) Locked recipe on all 5 shapes (best init each): sphere, cylinder,
#     pyramid, bowl(+w_gouge8 -- bowl needs it), sphere_hole. The deliverable.
# (B) SHAPE-AGNOSTIC INIT probe: sphere_hole with MULTIDEPTH + k70 (not random).
#     If it works, ONE init + ONE loss serves all 5 -- the generalizable method.
#     (multidepth failed on concave pre-k-anneal, wave8 0.078.)
# (C) w_gouge6 compromise: hole+wg6, bowl+wg6 -- middle weight that may serve
#     both convex bowl and concave hole without per-shape tuning.
# Usage: bash launch_wave12.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
K70="--k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7"
# Convex native base: multidepth wr5, default lr 5e-3.
MD5="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
# Concave proven base: random wr3 lr1e-3.
HOLEBASE="--init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 4.0"
SPH="--target-shape sphere --target-radius-mm 11.43"
CYL="--target-shape cylinder --target-radius-mm 11.43"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
BOWL="--target-shape sphere_bowl --target-radius-mm 11.43 --target-sub-radius-mm 9.525"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <args...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave12] GPU$gpu -> $name"
}

# (A) Locked canonical recipe on all 5 shapes.
launch 0 w12_sph_canon    $SPH  $MD5 $K70
launch 1 w12_cyl_canon    $CYL  $MD5 $K70
launch 2 w12_pyr_canon    $PYR  $MD5 $K70
launch 3 w12_bowl_canon   $BOWL $MD5 $K70 --w-gouge 8.0
launch 4 w12_hole_canon   $HOLE $HOLEBASE $K70

# (B) Shape-agnostic init: multidepth on concave + k70.
launch 5 w12_hole_md_k70  $HOLE $MD5 $K70

# (C) w_gouge6 compromise.
launch 6 w12_hole_wg6     $HOLE $HOLEBASE $K70 --w-gouge 6.0
launch 7 w12_bowl_wg6     $BOWL $MD5 $K70 --w-gouge 6.0

echo "[wave12] all 8 launched. Logs in $D/run_w12_*.log"
