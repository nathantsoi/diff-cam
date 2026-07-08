#!/bin/bash
# Wave 2: does multidepth beat the wave-1 leaders (raster 0.654 hdice, rfw 0.643
# zero-gouge) by matching coverage WITHOUT gouging? Sweep multidepth density
# (feed/revs) + hard-dice loss levers, and test de-gouging the raster leader.
# Default sphere scenario; seed=1 fixed (deterministic), so raster/rfw are NOT
# re-run -- wave-2 multidepth results compare directly to their recorded numbers.
# Each run pinned to one GPU via CUDA_VISIBLE_DEVICES, nohup'd to its own log.
# Usage: bash launch_wave2.sh
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
mkdir -p runs/jul8-multidepth

# Common template (matches wave-1: lr 5e-3, k_init 10, dt 0.45, feed 10 default).
BASE="--stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 \
  --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 \
  --post haas --eval-freq 10"

launch() {  # <gpu> <name> <extra-flags...>
  local gpu="$1"; local name="$2"; shift 2
  local log="$D/run_${name}.log"
  : > "$log"  # truncate any stale log
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python scripts/run_pipeline.py $BASE "$@" > "$log" 2>&1 &
  echo "[wave2] GPU$gpu -> $name : $*"
}

# --- multidepth density sweep (core test of the feed/revs hypothesis) ---
# GPU0: multidepth at default feed (10) -- revs auto-shrunk to ~3 (sparse channel).
launch 0 w2_md_default --init-mode multidepth
# GPU1: multidepth feed60 revs12 -- ~6x budget -> dense angular bulk removal.
launch 1 w2_md_feed60 --init-mode multidepth --feed-ipm 60 --multidepth-revs 12
# GPU2: dense + tighter margin (less residual waste, still no gouge).
launch 2 w2_md_feed60_tight --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-margin 0.01
# GPU3: even denser (revs24) -- push the budget limit.
launch 3 w2_md_feed60_rev24 --init-mode multidepth --feed-ipm 60 --multidepth-revs 24
# GPU4: more radial sweep cycles (levels10) -- denser multi-depth passes.
launch 4 w2_md_feed60_lvl10 --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --multidepth-levels 10

# --- multidepth + hard-dice loss levers ---
# GPU5: dense multidepth + tool-gouge barrier (direct hard-gouge penalty).
launch 5 w2_md_feed60_tg --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --w-tool-gouge 1.0
# GPU6: dense multidepth + loss-shift (de-bias soft loss toward hard carve).
launch 6 w2_md_feed60_shift --init-mode multidepth --feed-ipm 60 --multidepth-revs 12 --loss-shift 3.0

# --- de-gouge the wave-1 leader ---
# GPU7: raster (wave-1 winner 0.654) + tool-gouge barrier -- can it keep its dice
#       while removing its 12.875-voxel gouge? (raster gouges to win; barrier may
#       either clean it up or kill the aggressive coverage that made it win.)
launch 7 w2_raster_tg --init-mode raster --w-tool-gouge 1.0

echo "[wave2] all 8 launched. Logs in $D/run_w2_*.log"
