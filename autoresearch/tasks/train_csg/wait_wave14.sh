#!/bin/bash
# Block until all 8 wave-14 runs print a final summary (^hard_dice:) or die.
cd "$(dirname "$0")"
logs=(w14_hole_8k_anlr w14_pyr_8k_anlr w14_hole_3k w14_pyr_3k w14_hole_5k_b w14_pyr_5k_b w14_hole_5k_anlr w14_pyr_5k_anlr)
declare -A done
while true; do
  alldone=1
  for f in "${logs[@]}"; do
    [ "${done[$f]:-0}" = "1" ] && continue
    if grep -qE "^hard_dice:" "run_${f}.log" 2>/dev/null; then
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
echo "=== WAVE 14 COMPLETE ==="
for f in "${logs[@]}"; do
  hd=$(grep -E "^hard_dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  dc=$(grep -E "^dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  gg=$(grep -E "^gouge:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  echo "[$f] hard_dice=${hd:-NA} soft_dice=${dc:-NA} gouge=${gg:-NA}"
done
