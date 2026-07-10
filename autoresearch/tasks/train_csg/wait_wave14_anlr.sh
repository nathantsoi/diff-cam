#!/bin/bash
# Wait for the 2 still-running wave-14 8k_anlr runs, then move them into the
# batch dir and report final hard_dice (the anneal-lr-rescues-8k-collapse test).
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
D=autoresearch/tasks/train_csg
logs=(w14_hole_8k_anlr w14_pyr_8k_anlr)
declare -A done
while true; do
  alldone=1
  for f in "${logs[@]}"; do
    [ "${done[$f]:-0}" = "1" ] && continue
    if grep -qE "^hard_dice:" "$D/run_${f}.log" 2>/dev/null; then
      done[$f]=1
    elif ! pgrep -f "run_pipeline" >/dev/null 2>&1; then
      for g in "${logs[@]}"; do done[$g]=1; done
      alldone=1; break
    else
      alldone=0
    fi
  done
  [ "$alldone" = "1" ] && break
  sleep 30
done
echo "=== WAVE 14 ANLR COMPLETE ==="
# Move the two top-level anlr run dirs into the batch dir.
for ts in 1783632046922 1783632046946; do
  d="runs/CamEnvDiff-v0__train_csg__1__$ts"
  if [ -d "$d" ] && [ -f "$d/trajectory.npy" ]; then
    mv "$d" runs/jul8-multidepth/ && echo "moved $ts -> jul8-multidepth"
  elif [ -d "$d" ]; then
    echo "WARN $tST still no trajectory.npy; leaving in place"
  fi
done
for f in "${logs[@]}"; do
  hd=$(grep -E "^hard_dice:" "$D/run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  dc=$(grep -E "^dice:" "$D/run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  gg=$(grep -E "^gouge:" "$D/run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  echo "[$f] hard_dice=${hd:-NA} soft_dice=${dc:-NA} gouge=${gg:-NA}"
done
