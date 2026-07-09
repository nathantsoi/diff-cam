#!/bin/bash
# Wave 5: generality sweep on 4 NEW shapes (box, pyramid, sphere_hole,
# sphere_bowl), each with the winning method (multidepth + w_res5) AND a matched
# random+wr5 baseline (same loss config -- isolates the init method). Combined
# with sphere (wave4) + cylinder (wave4) this completes the 6-shape generality
# table; average dice_improvement across shapes = the generality score.
# Uses default shape sizes: radius 11.43, height 22.86, sub_radius 9.525.
# Usage: bash launch_wave5.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
MD="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
RAN="--init-mode random --w-residual 5.0"
BOX="--target-shape box --target-radius-mm 11.43"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"
BOWL="--target-shape sphere_bowl --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <shape-args...> <method-or-ran-flags...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave5] GPU$gpu -> $name"
}

# 4 new shapes x {multidepth+wr5, random+wr5} = 8 matched runs.
launch 0 w5_box_md    $BOX  $MD
launch 1 w5_box_rand  $BOX  $RAN
launch 2 w5_pyr_md    $PYR  $MD
launch 3 w5_pyr_rand  $PYR  $RAN
launch 4 w5_hole_md   $HOLE $MD
launch 5 w5_hole_rand $HOLE $RAN
launch 6 w5_bowl_md   $BOWL $MD
launch 7 w5_bowl_rand $BOWL $RAN

echo "[wave5] all 8 launched. Logs in $D/run_w5_*.log"
