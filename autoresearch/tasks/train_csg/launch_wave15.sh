#!/bin/bash
# Wave 15: VARIANCE characterization. 6-wide (GPUs 2-7; 0-1 still on wave-14 anlr).
# Wave 14 shock: pyr_5k_b=0.541 vs canonical pyr_5k=0.733 -- a 0.19 gap at FIXED
# seed=1 (GPU nondeterminism). The 0.733 "win" may be a lucky draw. Before chasing
# any more frontier we MUST know the run-to-run spread of the canonical method.
# Also: 3k≈5k (peak is early), anneal-lr HURTS at 5k (pyr 0.608<0.733).
# (A) 4x pyramid 5k K70 MD5 seed=1 -- variance of the headline result.
# (B) 2x hole 5k K70 HOLEBASE seed=1 -- hole looked stable (0.250/0.249/0.253); confirm.
# All runs now write into runs/jul8-multidepth/ via --runs-subdir (webapp batch fix).
# Usage: bash launch_wave15.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
SUB=runs/jul8-multidepth
mkdir -p "$SUB"

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10 --runs-subdir jul8-multidepth"
K70="--k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7"
MD5="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
HOLEBASE="--init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 4.0"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <iters> <args...>
  local gpu="$1"; local name="$2"; local iters="$3"; shift 3
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters "$iters" --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave15] GPU$gpu -> $name (iters=$iters)"
}

# (A) pyramid variance: 4 repeats of the canonical 5k K70 MD5.
launch 2 w15_pyr_5k_r1  5000 $PYR  $MD5 $K70
launch 3 w15_pyr_5k_r2  5000 $PYR  $MD5 $K70
launch 4 w15_pyr_5k_r3  5000 $PYR  $MD5 $K70
launch 5 w15_pyr_5k_r4  5000 $PYR  $MD5 $K70

# (B) hole stability: 2 repeats of canonical 5k K70 HOLEBASE.
launch 6 w15_hole_5k_r1 5000 $HOLE $HOLEBASE $K70
launch 7 w15_hole_5k_r2 5000 $HOLE $HOLEBASE $K70

echo "[wave15] all 6 launched. Logs in $D/run_w15_*.log"
