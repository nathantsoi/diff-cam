#!/usr/bin/env bash
# PREFERENCE-LEARNING PAIR DRIVER. Runs TWO experiments that vary a single
# objective knob at two magnitudes (everything else -- shape, radius, seed,
# iters, all other flags -- held fixed), lands both in runs/<tag>/ with
# --save-model (so trajectories are viewable in compare.html), then enqueues an
# A/B pair for the human to judge via scripts/enqueue_pair.py.
#
# This is the agent-side counterpart to compare.html: the agent picks a
# dimension + two magnitudes, this script produces the two runs and the queued
# pair, the human answers in the web UI, and pref_digest.py feeds the answer
# back into the next loop. Preferences can be used to STEER the agent's next
# sweeps / objective formulation AND, where it helps, to ENCODE a preference
# directly as a training-loss term (the loss is fully editable; per-shape
# branching is also allowed -- see autoresearch.md "What you CAN/CANNOT do").
#
# USAGE (parametric -- the agent clones this per dimension/knob):
#   DIM=w_air_time  \            # objective knob / dimension label
#   FLAG_A=--w-air-time MAG_A=1e-3 \   # side A: flag + value
#   FLAG_B=--w-air-time MAG_B=1e-2 \   # side B: flag + value (same flag, diff val)
#   SHAPE=sphere RADIUS=9.0 SEED=1 ITERS=5000 TAG=pref_wairtime_sph \
#   PROMPT="Which trajectory air-cuts less at the end?" \
#   SCENARIO="sphere s1 iters5000" \
#   bash autoresearch/tasks/train_csg/sweep_pref_pair.sh
#
# A concrete filled-in example is invoked at the bottom of this file when no
# DIM env var is set (w_air_time 1e-3 vs 1e-2, sphere). Override any of the
# env vars above to explore a different dimension/magnitude pair.
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam

# ---- defaults (the w_air_time sphere example) ----
DIM="${DIM:-w_air_time}"
FLAG_A="${FLAG_A:---w-air-time}"
MAG_A="${MAG_A:-1e-3}"
FLAG_B="${FLAG_B:---w-air-time}"
MAG_B="${MAG_B:-1e-2}"
SHAPE="${SHAPE:-sphere}"
RADIUS="${RADIUS:-9.0}"
SEED="${SEED:-1}"
ITERS="${ITERS:-5000}"
MAX_STEPS="${MAX_STEPS:-128}"
# Optional extra flags applied to BOTH sides so the pair runs on a deployable
# config (e.g. the SOTA contour+dual-adaptive method) rather than the bare
# baseline. The dimension knob (FLAG_A/MAG_A vs FLAG_B/MAG_B) is still the only
# thing that differs between A and B. Quote as a single string of args.
BASE_FLAGS="${BASE_FLAGS:-}"
TAG="${TAG:-pref_wairtime_sph}"
PROMPT="${PROMPT:-Which trajectory air-cuts less at the end?}"
SCENARIO="${SCENARIO:-${SHAPE} s${SEED} iters${ITERS}}"

LOGDIR=autoresearch/tasks/train_csg
SUB="$TAG"
mkdir -p "runs/$SUB"

GPUS=(0 1 2 3 4 5 6 7)
declare -A BUSY

gpu_free() {
  local g=$1 used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
  [ -z "$used" ] && used=9999
  [ "$used" -lt 100 ]
}

# launch_one <gpu> <side a|b> <flag> <mag>
launch_one() {
  local gpu=$1 side=$2 flag=$3 mag=$4
  local log="$LOGDIR/run_${TAG}_${side}.log"
  echo "[gpu $gpu] launch $TAG side=$side $flag $mag $SHAPE r$RADIUS s$SEED iters=$ITERS -> $log"
  CUDA_VISIBLE_DEVICES=$gpu uv run python -m algorithms.train_csg \
    --target_shape "$SHAPE" --target_radius_mm "$RADIUS" \
    --seed "$SEED" --runs_subdir "$SUB" --no-track --best_on_hard \
    --iters "$ITERS" --max_steps "$MAX_STEPS" \
    $BASE_FLAGS \
    $flag "$mag" \
    --save-model \
    > "$log" 2>&1 &
  BUSY[$gpu]=$!
}

# Launch side A then side B on the first two free GPUs (sequential scan so the
# pair reliably co-locates when GPUs are free; falls back to the wait loop).
idx=0
SIDES=( "a:$FLAG_A:$MAG_A" "b:$FLAG_B:$MAG_B" )
for g in "${GPUS[@]}"; do
  if [ $idx -lt 2 ] && gpu_free "$g"; then
    IFS=':' read -r sside sflag smag <<< "${SIDES[$idx]}"
    launch_one "$g" "$sside" "$sflag" "$smag"
    idx=$((idx+1))
  fi
done
while [ $idx -lt 2 ]; do
  sleep 30
  for g in "${GPUS[@]}"; do
    pid="${BUSY[$g]:-}"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      if [ $idx -lt 2 ] && gpu_free "$g"; then
        IFS=':' read -r sside sflag smag <<< "${SIDES[$idx]}"
        launch_one "$g" "$sside" "$sflag" "$smag"
        idx=$((idx+1))
      fi
    fi
  done
done
for g in "${GPUS[@]}"; do [ -n "${BUSY[$g]:-}" ] && wait "${BUSY[$g]}" 2>/dev/null; done

# ---- locate the two finished run dirs (newest in runs/<tag>/) ----
mapfile -t DONE < <(ls -dt "runs/$SUB"/*/ 2>/dev/null | head -2)
if [ ${#DONE[@]} -lt 2 ]; then
  echo "[pref-pair] ERROR: expected 2 runs in runs/$SUB/, found ${#DONE[@]} -- not enqueuing" >&2
  exit 1
fi
RUN_A="$(basename "${DONE[0]}")"
RUN_B="$(basename "${DONE[1]}")"
# Sanity: side A should be the one whose args.json holds MAG_A. If the newest
# two are swapped relative to launch order, fix up by inspecting args.json.
A_ARGS="$(grep -o "\"$FLAG_A\"[^,}]*\|\"$(echo "$FLAG_A" | sed 's/^--//')\"[^,}]*" "runs/$SUB/$RUN_A/args.json" 2>/dev/null | head -1)"
if echo "$A_ARGS" | grep -q "$MAG_B" && ! echo "$A_ARGS" | grep -q "$MAG_A"; then
  # newest run is actually side B; swap.
  tmp="$RUN_A"; RUN_A="$RUN_B"; RUN_B="$tmp"
fi

echo "[pref-pair] enqueue: A=$RUN_A ($MAG_A) vs B=$RUN_B ($MAG_B) dim=$DIM scenario=$SCENARIO"
uv run python scripts/enqueue_pair.py \
  --run-a "$RUN_A" --run-b "$RUN_B" \
  --dimension "$DIM" --mag-a "$MAG_A" --mag-b "$MAG_B" \
  --scenario "$SCENARIO" --prompt "$PROMPT"
echo "ALL DONE — open compare.html to judge the pair."
