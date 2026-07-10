#!/bin/bash
# Wave 16: BEST-ON-HARD checkpoint selection test. 8-wide.
# Wave-14 finding: pyr_8k_anlr final-iter hard_dice=0.815 (stable across iters
# 7995-7999) but the SOFT-dice best-checkpoint selector deployed a checkpoint
# with hard_dice=0.713 -- throwing away 0.10 of deployable dice. Root cause:
# composite_score substitutes soft_dice (lines ~1256/1393 of train_csg.py).
# Fix shipped: --best-on-hard selects the deployable checkpoint by HARD dice.
# This wave tests whether it reliably raises the DEPLOYED hard_dice vs soft.
# 4 shapes x {soft control, --best-on-hard} = 8 runs, seed=1, 5000 iters.
# All into runs/jul8-multidepth/ via --runs-subdir.
# Usage: bash launch_wave16.sh [first_gpu]
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10 --runs-subdir jul8-multidepth"
K70="--k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7"
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
  echo "[wave16] GPU$gpu -> $name (iters=$iters)"
}

# 4 shapes x {soft (control), hard (--best-on-hard)}. 5000 iters canonical.
launch 0 w16_pyr_soft   5000 $PYR  $MD5 $K70
launch 1 w16_pyr_hard   5000 $PYR  $MD5 $K70 --best-on-hard
launch 2 w16_hole_soft  5000 $HOLE $HOLEBASE $K70
launch 3 w16_hole_hard  5000 $HOLE $HOLEBASE $K70 --best-on-hard
launch 4 w16_sph_soft   5000 $SPH  $MD5 $K70
launch 5 w16_sph_hard   5000 $SPH  $MD5 $K70 --best-on-hard
launch 6 w16_bowl_soft  5000 $BOWL $MD5 $K70 --w-gouge 6.0
launch 7 w16_bowl_hard  5000 $BOWL $MD5 $K70 --w-gouge 6.0 --best-on-hard

echo "[wave16] all 8 launched. Logs in $D/run_w16_*.log"
