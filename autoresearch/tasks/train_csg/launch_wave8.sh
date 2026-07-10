#!/bin/bash
# Wave 8: 8-wide. Two threads.
# (A) LOCK IN THE AIR LEVER across shapes. Wave-7 showed w_air_time=1e-2 (fixed
# gradient) cuts sharp air 0.886->0.831 on sphere and 0.852->0.825 on cylinder
# with dice preserved-or-better. Run the improved method (multidepth wr5 +
# w_air_time 1e-2) on the 4 shapes not yet tested with it: pyramid, box,
# sphere_bowl, and sphere_hole (control -- expected to stay near 0). Confirms
# the air win generalizes (the primary goal: a shape-agnostic method).
# (B) TAME THE sphere_hole GOUGE. Wave-7 found wr3+lr1e-3 is the FIRST recipe
# to train past iter0 (best @ iter2930, soft_dice 0.211) but it gouges 395
# (tool rams the hole column through the sphere top). Escalate w_gouge on that
# recipe (8, 16) to suppress the plunge while keeping the low residual that
# reaches the interior. w_gouge>4 was a loser on CONVEX; concave may differ.
# Confirmed losers (excluded): spiral init (NaN iter0), raster_fine (gouge
# 215), wr5/lr1e-3 (still iter0 collapse), loss_shift, lr>5e-3, w_traj_prox
# on concave, k-anneal/k60/wr10 on sphere_hole.
# Usage: bash launch_wave8.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

COMMON="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --post haas --eval-freq 10"
# Improved method: multidepth wr5 + the tuned honest air lever (1e-2).
MDAIR="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 5.0 --w-air-time 1e-2"
BOX="--target-shape box --target-radius-mm 11.43"
PYR="--target-shape pyramid --target-radius-mm 11.43 --target-height-mm 22.86"
HOLE="--target-shape sphere_hole --target-radius-mm 11.43 --target-sub-radius-mm 9.525"
BOWL="--target-shape sphere_bowl --target-radius-mm 11.43 --target-sub-radius-mm 9.525"

launch() {  # <gpu> <name> <args...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 $COMMON "$@" > "$log" 2>&1 &
  echo "[wave8] GPU$gpu -> $name"
}

# (A) Air-lever generality: improved method on 4 new shapes.
launch 0 w8_pyr_md_air   $PYR  $MDAIR
launch 1 w8_box_md_air   $BOX  $MDAIR
launch 2 w8_bowl_md_air  $BOWL $MDAIR
launch 3 w8_hole_md_air  $HOLE $MDAIR   # control: concave, expected near 0

# (B) Hole-gouge taming: wr3+lr1e-3 (the iter2930 recipe) + escalating w_gouge.
launch 4 w8_hole_wr3_lr3_g8   $HOLE --init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 8.0
launch 5 w8_hole_wr3_lr3_g16  $HOLE --init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 16.0
launch 6 w8_hole_wr3_lr3_g8a  $HOLE --init-mode random --w-residual 3.0 --learning-rate 1e-3 --w-gouge 8.0 --w-air-time 1e-2
launch 7 w8_hole_wr2_lr3_g8   $HOLE --init-mode random --w-residual 2.0 --learning-rate 1e-3 --w-gouge 8.0

echo "[wave8] all 8 launched. Logs in $D/run_w8_*.log"
