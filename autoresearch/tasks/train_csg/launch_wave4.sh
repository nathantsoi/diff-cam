#!/bin/bash
# Wave 4: push the winning lever (w_residual) harder + complementary knobs, and
# START the generality probe. Wave-3 showed: margin sweep is flat (init geometry
# is inert -- the optimizer re-positions regardless); w_residual=3.0 is the real
# lever (0.599->0.621, ZERO gouge). Best clean config: multidepth feed60 revs12
# margin0 --w-residual 3.0. Now: raise w_residual further, add lr/iters, and test
# the best config on a SECOND shape (cylinder) to begin the generality sweep.
# Usage: bash launch_wave4.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

SPHERE="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --eval-freq 10"
# cylinder: radius 11.43mm, height 22.86mm (matches the args default sub-radius not needed)
CYL="--stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape cylinder --target-radius-mm 11.43 --target-height-mm 22.86 --post haas --eval-freq 10"
MD="--init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0"

launch() {  # <gpu> <name> <base-args...> -- <extra-flags...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"; : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 "$@" > "$log" 2>&1 &
  echo "[wave4] GPU$gpu -> $name"
}

# --- sphere: push w_residual + complementary knobs (keep best geometry) ---
# GPU0: w_residual 5.0 -- push the lever harder.
launch 0 w4_sph_wr5 $SPHERE $MD --w-residual 5.0
# GPU1: w_residual 10.0 -- aggressive residual-clearing.
launch 1 w4_sph_wr10 $SPHERE $MD --w-residual 10.0
# GPU2: w_residual 3.0 + higher lr 1e-2 -- faster convergence on residual.
launch 2 w4_sph_wr3_lr1e2 $SPHERE $MD --w-residual 3.0 --learning-rate 1e-2
# GPU3: w_residual 5.0 + w_gouge 8.0 -- more residual AND stronger gouge barrier.
launch 3 w4_sph_wr5_wg8 $SPHERE $MD --w-residual 5.0 --w-gouge 8.0
# GPU4: w_residual 3.0 + revs24 -- best residual lever + denser coverage.
launch 4 w4_sph_wr3_rev24 $SPHERE $MD --w-residual 3.0 --multidepth-revs 24
# GPU5: w_residual 5.0 + loss-shift 2.0 -- residual lever + mild hard de-bias.
launch 5 w4_sph_wr5_shift $SPHERE $MD --w-residual 5.0 --loss-shift 2.0

# --- generality probe: best config on a SECOND shape (cylinder) ---
# GPU6: cylinder, best clean config (multidepth + w_residual3) -- does the
#       shape-agnostic method transfer WITHOUT retuning?
launch 6 w4_cyl_md_wr3 $CYL $MD --w-residual 3.0
# GPU7: cylinder baseline (random init, default w_residual) -- generality
#       reference for the cylinder scenario.
launch 7 w4_cyl_rand $CYL --init-mode random

echo "[wave4] all 8 launched. Logs in $D/run_w4_*.log"
