#!/bin/bash
# Wave 3: find the engagement sweet spot. Wave-2 showed dense multidepth
# (feed60, zero gouge) leaves high RESIDUAL (~8300 vox, hard_dice 0.60) while
# sparse/gouging multidepth leaves low residual (6573, hdice 0.64) -- hard_dice
# tracks RESIDUAL, and the tool tolerates mild gouge. Goal: let multidepth bite
# closer to the surface to drop residual WITHOUT relying on heavy gouge. Sweep
# multidepth_margin from +0.02 (tangent-outside) through 0 (on surface) to
# negative (intentional light engagement), plus raise w_residual to drive
# residual-clearing. Default sphere scenario, seed=1.
# Usage: bash launch_wave3.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

BASE="--stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 \
  --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 \
  --post haas --eval-freq 10"

launch() {  # <gpu> <name> <extra-flags...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"
  : > "$log"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py $BASE "$@" > "$log" 2>&1 &
  echo "[wave3] GPU$gpu -> $name : $*"
}

# Use the proven dense geometry (feed60 revs12) as the base, vary engagement.
# GPU0: margin +0.01 (already tested as w2_md_feed60_tight=0.600; reproducibility check)
launch 0 w3_md_m001 --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.01
# GPU1: margin 0.0 -- tool tangent to target surface (no gouge by init, optimizer hugs).
launch 1 w3_md_m0 --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0
# GPU2: margin -0.01 -- light intentional engagement (~0.5mm into part) to clear residual.
launch 2 w3_md_mn01 --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin -0.01
# GPU3: margin -0.02 -- ~1mm engagement, more aggressive residual removal.
launch 3 w3_md_mn02 --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin -0.02
# GPU4: margin -0.03 -- pushes toward raster's gouge-to-win regime.
launch 4 w3_md_mn03 --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin -0.03
# GPU5: margin 0.0 + w_residual 3.0 -- drive residual-clearing via loss not gouge.
launch 5 w3_md_m0_wr3 --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.0 --w-residual 3.0
# GPU6: margin -0.01 + w_residual 3.0 -- light engagement + strong residual drive.
launch 6 w3_md_mn01_wr3 --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin -0.01 --w-residual 3.0
# GPU7: margin -0.01 + revs24 -- light engagement + denser angular coverage.
launch 7 w3_md_mn01_rev24 --init-mode multidepth --feed-ipm 60 --multidepth-revs 24 --multidepth-margin -0.01

echo "[wave3] all 8 launched. Logs in $D/run_w3_*.log"
