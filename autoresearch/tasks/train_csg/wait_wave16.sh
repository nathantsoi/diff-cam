#!/bin/bash
# Block until all 8 wave-16 runs print a final summary (^hard_dice:) or die.
cd "$(dirname "$0")"
logs=(w16_pyr_soft w16_pyr_hard w16_hole_soft w16_hole_hard w16_sph_soft w16_sph_hard w16_bowl_soft w16_bowl_hard)
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
echo "=== WAVE 16 COMPLETE: best-on-hard vs soft ==="
for f in "${logs[@]}"; do
  hd=$(grep -E "^hard_dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  dc=$(grep -E "^dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  # also capture final_iter_hard_dice for the deployed-vs-final gap
  echo "[$f] hard_dice=${hd:-NA} soft_dice=${dc:-NA}"
done
