#!/bin/bash
# Wave 10: 8-wide. Two threads.
# HEADLINE (wave 9): k-anneal k 2->50 broke the sphere_hole soft/hard gap
# (hard_dice 0.124 -> 0.231, ~2x). Only k_final=50 works; k=30 and loss_shift
# alone cannot. The gap is a union-sharpness problem.
# OPEN QUESTION (primary goal): is k-anneal SHAPE-AGNOSTIC? It broke concave;
# does it help/hurt convex? If k-anneal50 holds/improves convex AND broke
# concave, it is THE generalizable method.
# (A) Push sphere_hole k-anneal harder (4 runs, wr3+lr1e-3 base):
#   - k_final 70, 100 (sharper -- does hard_dice climb further?)
#   - k50 + w_gouge 8 (tame the 217 gouge, keep the 0.231 dice)
#   - k50 + loss_shift 1.2 (more shift than 0.7)
# (B) Generalize k-anneal50 to convex shapes on native md-wr5 base (4 runs):
#   sphere, cylinder, pyramid, bowl. k_init 2 -> k_final 50.
# Usage: bash launch_wave10.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"
HOLEBASE="--init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 4.0"
# Native convex base: multidepth wr5, default lr 5e-3, default w_gouge 4.
MD5="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0"
KAN50="--k-anneal --k-init 2.0 --k-final 50.0"
SPH="--target-shape sphere --target-radius-mm 11.43"
CYL="--target-shape cylinder --target-radius-mm 11.43"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
BOWL="--target-shape sphere_bowl --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <args...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave10] GPU$gpu -> $name"
}

# (A) Push sphere_hole k-anneal.
launch 0 w10_hole_k70        $HOLE $HOLEBASE --k-anneal --k-init 2.0 --k-final 70.0 --loss-shift 0.7
launch 1 w10_hole_k100       $HOLE $HOLEBASE --k-anneal --k-init 2.0 --k-final 100.0 --loss-shift 0.7
launch 2 w10_hole_k50_g8     $HOLE $HOLEBASE --k-anneal --k-init 2.0 --k-final 50.0 --w-gouge 8.0 --loss-shift 0.7
launch 3 w10_hole_k50_ls12   $HOLE $HOLEBASE --k-anneal --k-init 2.0 --k-final 50.0 --loss-shift 1.2

# (B) Generalize k-anneal50 to convex (native md-wr5 base).
launch 4 w10_sph_k50     $SPH  $MD5 $KAN50
launch 5 w10_cyl_k50     $CYL  $MD5 $KAN50
launch 6 w10_pyr_k50     $PYR  $MD5 $KAN50
launch 7 w10_bowl_k50    $BOWL $MD5 $KAN50

echo "[wave10] all 8 launched. Logs in $D/run_w10_*.log"
