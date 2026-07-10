#!/bin/bash
# Wave 9: 8-wide. Two threads.
# (A) ATTACK THE SOFT/HARD GAP on sphere_hole. Wave 8 confirmed wr3+lr1e-3 CURES
# the iter-0 collapse (all 4 hole variants train stably, final soft_dice 0.23-
# 0.35) but hard_dice is stuck ~0.124 regardless. Diagnosis: compute_loss uses
# SIGMOID-BLURRED stock_occ/target_occ (line 1083-1089). On the thin hole wall
# the blur rewards carving NEAR the wall (residual satisfied) while the SHARP
# cut gouges through -> hard_dice capped. w_gouge 8/16 could NOT fix this
# (wave 8) because the barrier is ALSO blurred. Two principled, shape-agnostic
# fixes (both already coded, previously losers ONLY because collapse dominated):
#   - loss_shift: add ~log(2)*k_ref/k_final to stock_d before the loss sigmoid
#     so the loss targets the (less-eroded) HARD carve. k_ref~51, so shift~3.5
#     at k=10, ~1.2 at k=30.
#   - k-anneal: ramp k low->high so late-training soft-union SHARPENS and soft
#     coverage tracks HARD coverage on the concave wall.
# Run each on the PROVEN wr3+lr1e-3 hole recipe.
# (B) RE-PROBE THE AIR LEVER at gentler 3e-3. Wave 8 found w_air_time=1e-2 is
# NOT a clean generalization win: pyramid dice DROPPED 0.494->0.448, bowl
# gained dice but GOUGED 299. Test 3e-3 (between default 1e-3 and 1e-2) on
# pyramid (undo regression?), bowl (keep gain, drop gouge?), sphere (reference
# interpolant vs 1e-3/1e-2), + a pyramid 1e-3 control for a clean same-seed A/B.
# Usage: bash launch_wave9.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"
# Proven collapse-curing hole recipe (wave 7/8): low residual lets optimizer
# explore the interior, slow lr refines.
HOLEBASE="--init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 4.0"
# Improved multidepth method (air-lever probe).
MDAIR3="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0 --w-air-time 3e-3"
MDAIR0="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0 --w-air-time 1e-3"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
BOWL="--target-shape sphere_bowl --target-radius-mm 11.43 --target-sub-radius-mm 9.525"
SPH="--target-shape sphere --target-radius-mm 11.43"

launch() {  # <gpu> <name> <args...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave9] GPU$gpu -> $name"
}

# (A) Soft/hard gap attack on sphere_hole.
launch 0 w9_hole_lshift35   $HOLE $HOLEBASE --loss-shift 3.5
launch 1 w9_hole_lshift12   $HOLE $HOLEBASE --loss-shift 1.2
launch 2 w9_hole_kanneal30  $HOLE $HOLEBASE --k-anneal --k-init 2.0 --k-final 30.0 --loss-shift 1.2
launch 3 w9_hole_kanneal50  $HOLE $HOLEBASE --k-anneal --k-init 2.0 --k-final 50.0 --loss-shift 0.7

# (B) Air lever re-probe at 3e-3 + pyramid control.
launch 4 w9_pyr_air3m3   $PYR  $MDAIR3
launch 5 w9_bowl_air3m3  $BOWL $MDAIR3
launch 6 w9_sph_air3m3   $SPH  $MDAIR3
launch 7 w9_pyr_air1m3   $PYR  $MDAIR0   # control: same-seed 1e-3 vs wave8 1e-2

echo "[wave9] all 8 launched. Logs in $D/run_w9_*.log"
