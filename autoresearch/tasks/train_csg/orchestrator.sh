#!/bin/bash
# Per-GPU experiment orchestrator for the autoresearch loop.
# Usage: orchestrator.sh <gpu> <wait_log> <tag> <specs...>
#   <gpu>       : CUDA_VISIBLE_DEVICES index
#   <wait_log>  : log file of an ALREADY-RUNNING job to wait for before starting the queue
#                 (empty string "" = start the queue immediately)
#   <tag>       : run-folder tag (runs/<tag>/)
#   <specs...>  : each spec is "tag|shape|maxsteps|k|wlen|seed|radius_mm|height_mm"
#                 e.g. "box_k10|box|128|10|0.0|1|11.43|22.86"
# For each spec: run train (5000 iters, bare config: random init, lr1e-3, dt0.45,
# gc0.5, eval10), capture dice+vrampath+commit, move run dir into runs/<tag>/,
# append a row to results.tsv. Stops on first crash (NaN/Traceback) after recording
# it as a crash row, then CONTINUES to the next spec (a crash is logged, not fatal).
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
GPU="$1"; WAIT_LOG="$2"; TAG="$3"; shift 3
COMMIT=$(git rev-parse --short HEAD)
mkdir -p "runs/$TAG"

# Wait for an already-running job to finish (so we don't double-launch on the GPU).
if [ -n "$WAIT_LOG" ]; then
  echo "[orch:GPU$GPU] waiting for $WAIT_LOG to finish..."
  while ! grep -qE "^dice:|Traceback|NaN detected" "$WAIT_LOG" 2>/dev/null; do sleep 20; done
  echo "[orch:GPU$GPU] $WAIT_LOG finished."
fi

for spec in "$@"; do
  IFS='|' read -r stag shape msteps k wlen seed rad hgt <<<"$spec"
  log="autoresearch/tasks/train_csg/run_${stag}.log"
  echo "[orch:GPU$GPU] === $stag (shape=$shape m=$msteps k=$k w_len=$wlen seed=$seed) ==="
  CUDA_VISIBLE_DEVICES=$GPU nohup uv run python scripts/run_pipeline.py --stages train \
    --iters 5000 --max-steps "$msteps" --stock-size-in 1 1 1 --voxel-size-mm 0.5 \
    --target-shape "$shape" --target-radius-mm "$rad" --target-height-mm "$hgt" \
    --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 \
    --k-init "$k" --w-len "$wlen" --seed "$seed" > "$log" 2>&1 &
  PID=$!
  # Wait for THIS job to finish (or crash).
  while ! grep -qE "^dice:|Traceback|NaN detected" "$log" 2>/dev/null; do sleep 20; done
  wait $PID 2>/dev/null
  # Collect results.
  rundir=$(grep -oE "runs/CamEnvDiff-v0__train_csg__[0-9]+__1783[0-9]+" "$log" | head -1)
  dice=$(grep -E "^dice:" "$log" | awk '{print $2}')
  if [ -z "$dice" ]; then dice="0.000000"; fi
  # Move run dir into the tag folder.
  if [ -n "$rundir" ]; then mv "$rundir" "runs/$TAG/" 2>/dev/null; fi
  # Status: crash if NaN/Traceback, else keep.
  if grep -qE "NaN detected|Traceback" "$log"; then status="crash"; else status="keep"; fi
  cmd="uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps $msteps --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape $shape --target-radius-mm $rad --target-height-mm $hgt --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 --k-init $k --w-len $wlen --seed $seed"
  printf '%s\t%s\t0.1\t%s\t%s (GPU%s)\t%s\n' "$COMMIT" "$dice" "$status" "$stag HARD dice" "$GPU" "$cmd" >> autoresearch/tasks/train_csg/results.tsv
  echo "[orch:GPU$GPU] $stag -> dice=$dice status=$status"
done
echo "[orch:GPU$GPU] queue complete."
