#!/bin/bash
# Combined-CSG task runner: sphere_hole + sphere_bowl.
#
# These tasks exercise the GRADIENT-BASED optimizer (compute_loss + Taichi
# autodiff through tool_delta -> tool_pos -> apply_cut) on COMBINED CSG targets,
# not just single primitives. Each target is a 0.9in stock sphere with a 0.75in
# sub-primitive subtracted:
#   sphere_hole  : concentric through-hole cylinder (Z-axis) punched clean through.
#   sphere_bowl  : lower hemisphere of a concentric 0.75in sphere removed ->
#                  a bowl whose cavity opens upward at the equator.
# The sub-primitive radius defaults to 0.375in (9.525mm) in run_pipeline.py /
# set_target_params, so the orchestrator spec needs no extra flag.
#
# Config = the PROVEN operating point (see autoresearch.md "Proven operating
# point"): dt0.45, lr1e-3, raster_fine init, w_len 0.03, max-steps 256, gc0.5,
# eval10, iters5000. 3 paired seeds per shape -- the methodology's minimum for
# distinguishing a real lever from seed reshuffling (single-seed wins overstate
# ~2-3x).
#
# Usage: run_combined_shapes.sh <gpu> [tag]
#   <gpu> : CUDA_VISIBLE_DEVICES index
#   <tag> : run-folder tag (runs/<tag>/); default "combined_shapes"
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
GPU="${1:?usage: $0 <gpu> [tag]}"
TAG="${2:-combined_shapes}"
COMMIT=$(git rev-parse --short HEAD)
mkdir -p "runs/$TAG"

# shape|tag|maxsteps|wlen|seed
SPECS=(
  "sphere_hole|hole_s1|256|0.03|1"
  "sphere_hole|hole_s2|256|0.03|2"
  "sphere_hole|hole_s3|256|0.03|3"
  "sphere_bowl|bowl_s1|256|0.03|1"
  "sphere_bowl|bowl_s2|256|0.03|2"
  "sphere_bowl|bowl_s3|256|0.03|3"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r shape stag msteps wlen seed <<<"$spec"
  log="autoresearch/tasks/train_csg/run_${stag}.log"
  echo "[combined:GPU$GPU] === $stag (shape=$shape m=$msteps w_len=$wlen seed=$seed) ==="
  CUDA_VISIBLE_DEVICES=$GPU nohup uv run python scripts/run_pipeline.py --stages train \
    --iters 5000 --max-steps "$msteps" --stock-size-in 1 1 1 --voxel-size-mm 0.5 \
    --target-shape "$shape" --target-radius-mm 11.43 --target-height-mm 22.86 \
    --target-sub-radius-mm 9.525 --post haas --dt 0.45 --learning-rate 1e-3 \
    --grad-clip 0.5 --eval-freq 10 --init-mode raster_fine --w-len "$wlen" \
    --seed "$seed" > "$log" 2>&1 &
  PID=$!
  # Wait for THIS job to finish (or crash).
  while ! grep -qE "^dice:|Traceback|NaN detected" "$log" 2>/dev/null; do sleep 20; done
  wait $PID 2>/dev/null
  rundir=$(grep -oE "runs/CamEnvDiff-v0__train_csg__[0-9]+__1783[0-9]+" "$log" | head -1)
  dice=$(grep -E "^dice:" "$log" | awk '{print $2}')
  [ -z "$dice" ] && dice="0.000000"
  [ -n "$rundir" ] && mv "$rundir" "runs/$TAG/" 2>/dev/null
  if grep -qE "NaN detected|Traceback" "$log"; then status="crash"; else status="keep"; fi
  cmd="uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps $msteps --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape $shape --target-radius-mm 11.43 --target-height-mm 22.86 --target-sub-radius-mm 9.525 --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 --init-mode raster_fine --w-len $wlen --seed $seed"
  printf '%s\t%s\t0.1\t%s\t%s combined-shape %s (GPU%s)\t%s\n' \
    "$COMMIT" "$dice" "$status" "$stag" "$shape" "$GPU" "$cmd" \
    >> autoresearch/tasks/train_csg/results.tsv
  echo "[combined:GPU$GPU] $stag -> dice=$dice status=$status"
done
echo "[combined:GPU$GPU] queue complete."
