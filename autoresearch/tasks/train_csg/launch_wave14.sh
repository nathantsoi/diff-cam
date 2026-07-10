#!/bin/bash
# Wave 14: 8-wide. STABILITY wave -- the win is fragile to longer training.
# Wave 13: 8k iters COLLAPSED hole (0.253->0.123) and DROPPED pyr (0.733->0.607).
# Diagnosis: late-training HIGH-lr + HIGH-k sharpened-union tail destabilizes
# the carve. Fix: --anneal-lr (linear LR->0) to hold the peak. Just exposed in
# run_pipeline.
# (A) anneal-lr rescue: hole+pyr at 8k WITH --anneal-lr (hold the peak to 8k?).
# (B) Shorter: hole+pyr at 3000 (is the peak EARLY, <5000?).
# (C) Variance: hole+pyr K70 5k repeat (run-to-run noise of the peak).
# (D) anneal-lr at 5k canonical (helps even at 5k?).
# Usage: bash launch_wave14.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
K70="--k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7"
MD5="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
HOLEBASE="--init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 4.0"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <iters> <args...>
  local gpu="$1"; local name="$2"; local iters="$3"; shift 3
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters "$iters" --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave14] GPU$gpu -> $name (iters=$iters)"
}

# (A) anneal-lr rescue at 8k.
launch 0 w14_hole_8k_anlr    8000 $HOLE $HOLEBASE $K70 --anneal-lr
launch 1 w14_pyr_8k_anlr     8000 $PYR  $MD5 $K70 --anneal-lr

# (B) Shorter 3000.
launch 2 w14_hole_3k         3000 $HOLE $HOLEBASE $K70
launch 3 w14_pyr_3k          3000 $PYR  $MD5 $K70

# (C) Variance: 5k repeat.
launch 4 w14_hole_5k_b       5000 $HOLE $HOLEBASE $K70
launch 5 w14_pyr_5k_b        5000 $PYR  $MD5 $K70

# (D) anneal-lr at 5k canonical.
launch 6 w14_hole_5k_anlr    5000 $HOLE $HOLEBASE $K70 --anneal-lr
launch 7 w14_pyr_5k_anlr     5000 $PYR  $MD5 $K70 --anneal-lr

echo "[wave14] all 8 launched. Logs in $D/run_w14_*.log"
