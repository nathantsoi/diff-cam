#!/bin/bash
# Wave 7: 8-wide. Two threads.
# (A) AIR-LEVER RE-PROBE with the FIXED differentiable seg_air gradient
# (commit 6d4602a: pre-cut, so af = true air fraction, not the buggy af~=1
# total-time penalty). The 25mm tool body re-traverses carved space during
# helical descent -> sharp air 0.80-0.91; can honest air-time cut it without
# killing dice? Sweep w_air_time on sphere multidepth (the run-1972 config,
# wr5) + a cylinder generality check. w6_sph_md_air10 (buggy) already did
# 0.91->0.80 at dice 0.633; the fixed gradient should do equal-or-better air
# at a TRUE air-only penalty (engaged cutting no longer penalized).
# (B) SPHERE_HOLE: gradients DESTROY the concave carve (wave-6: all 4 lever
# variants -> iter-0 best, soft_dice 0.122->0.000). Attack via INIT + lr, not
# hyperparameters: the `spiral` init starts at r=0 (center) spiraling OUT
# while descending, so its early steps pass through the hole column; pair
# with slow lr (1e-3) to refine instead of over-erode the exterior.
# Confirmed losers (excluded): loss_shift, w_gouge>4, lr>5e-3, w_traj_prox
# on concave (gouge 816), k-anneal/k60/wr10 on sphere_hole (zero effect).
# Usage: bash launch_wave7.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
MD="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
SPHERE="--target-shape sphere --target-radius-mm 11.43"
CYL="--target-shape cylinder --target-radius-mm 11.43 --target-height-mm 22.86"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <args...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave7] GPU$gpu -> $name"
}

# (A) Air-lever re-probe, FIXED gradient.
launch 0 w7_sph_md_air1m3  $SPHERE $MD --w-air-time 1e-3    # control (re-baseline w/ fix)
launch 1 w7_sph_md_air1m2  $SPHERE $MD --w-air-time 1e-2
launch 2 w7_sph_md_air1m1  $SPHERE $MD --w-air-time 1e-1
launch 3 w7_cyl_md_air1m2  $CYL    $MD --w-air-time 1e-2    # generality: cylinder air 0.852

# (B) sphere_hole: spiral init (center-out, passes hole column) + slow lr.
launch 4 w7_hole_spiral_wr5     $HOLE --init-mode spiral --w-residual 5.0
launch 5 w7_hole_spiral_wr5_lr3 $HOLE --init-mode spiral --w-residual 5.0 --learning-rate 1e-3
launch 6 w7_hole_rand_wr5_lr3   $HOLE --init-mode random  --w-residual 5.0 --learning-rate 1e-3
launch 7 w7_hole_rand_wr3_lr3   $HOLE --init-mode random  --w-residual 3.0 --learning-rate 1e-3

echo "[wave7] all 8 launched. Logs in $D/run_w7_*.log"
