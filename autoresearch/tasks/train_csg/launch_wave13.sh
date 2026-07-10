#!/bin/bash
# Wave 13: 8-wide. Lock the canonical method + push the frontier.
# Canonical (wave 12): k-anneal k 2->70 + loss_shift 0.7. Convex init=multidepth
# wr5 lr5e-3; concave init=random wr3 lr1e-3. w_gouge 4 (6 for bowl).
# (A) Lock bowl at wg6 (wave12 0.551, beats baseline). + wg4 bowl control.
# (B) Push convex k higher: sph, pyr at k->90 (climb past 0.63/0.73?).
# (C) Robustness: 8000 iters on hole + pyr (the two big-win shapes) -- does the
#     gain keep climbing or saturate at 5000?
# (D) Bowl k70+wg6 already done; add bowl k90+wg6 to pair with (B).
# Usage: bash launch_wave13.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
K70="--k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7"
K90="--k-anneal --k-init 2.0 --k-final 90.0 --loss-shift 0.7"
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
  echo "[wave13] GPU$gpu -> $name (iters=$iters)"
}

# (A) Lock bowl wg6 + control.
launch 0 w13_bowl_k70_wg6b  5000 $BOWL $MD5 $K70 --w-gouge 6.0
launch 1 w13_bowl_k70_wg4   5000 $BOWL $MD5 $K70 --w-gouge 4.0

# (B) Push convex k->90.
launch 2 w13_sph_k90        5000 $SPH  $MD5 $K90
launch 3 w13_pyr_k90        5000 $PYR  $MD5 $K90
launch 4 w13_bowl_k90_wg6   5000 $BOWL $MD5 $K90 --w-gouge 6.0

# (C) Robustness: 8000 iters on the two big-win shapes.
launch 5 w13_hole_k70_8k    8000 $HOLE $HOLEBASE $K70
launch 6 w13_pyr_k70_8k     8000 $PYR  $MD5 $K70

# (D) hole k90 probe (does concave climb past 0.25 with more sharpening?).
launch 7 w13_hole_k90       5000 $HOLE $HOLEBASE $K90

echo "[wave13] all 8 launched. Logs in $D/run_w13_*.log"
