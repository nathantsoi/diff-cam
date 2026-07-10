#!/bin/bash
# Wave 11: 8-wide. Two threads.
# HEADLINE (wave 10): k-anneal is LARGELY SHAPE-AGNOSTIC. Concave k->70 hit
# hard_dice 0.2497 (2x cap). Convex: sphere/cylinder tie, pyramid +0.20
# (0.494->0.697), bowl FAILED (0.531->0.394, gouge 988 -- sharpening exposes a
# wall gouge). Gouge-taming by w_gouge HURTS concave (0.231->0.132); loss_shift
# >0.7 hurts once k-anneal active.
# (A) Lock concave sweet spot: k->60, k->70 confirm, k->70+lshift0.
# (B) Fix the BOWL (the one failure): k->30 (gentler), k->50+w_gouge8 (barrier
#     on gentler convex bowl), k_fixed50 (isolate anneal-vs-sharpness).
# (C) Confirm pyramid +0.20: k->50 repeat, k->70 (climb further?).
# Usage: bash launch_wave11.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"
HOLEBASE="--init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 4.0"
MD5="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
BOWL="--target-shape sphere_bowl --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <args...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave11] GPU$gpu -> $name"
}

# (A) Concave sweet spot.
launch 0 w11_hole_k60        $HOLE $HOLEBASE --k-anneal --k-init 2.0 --k-final 60.0 --loss-shift 0.7
launch 1 w11_hole_k70b       $HOLE $HOLEBASE --k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7
launch 2 w11_hole_k70_ls0    $HOLE $HOLEBASE --k-anneal --k-init 2.0 --k-final 70.0

# (B) Bowl fix.
launch 3 w11_bowl_k30        $BOWL $MD5 --k-anneal --k-init 2.0 --k-final 30.0
launch 4 w11_bowl_k50_g8     $BOWL $MD5 --k-anneal --k-init 2.0 --k-final 50.0 --w-gouge 8.0
launch 5 w11_bowl_kfix50     $BOWL $MD5 --k-init 50.0 --k-final 50.0

# (C) Pyramid confirm.
launch 6 w11_pyr_k50b        $PYR $MD5 --k-anneal --k-init 2.0 --k-final 50.0
launch 7 w11_pyr_k70         $PYR $MD5 --k-anneal --k-init 2.0 --k-final 70.0

echo "[wave11] all 8 launched. Logs in $D/run_w11_*.log"
