#!/bin/bash
# Block until all 6 wave-15 runs print a final summary (^hard_dice:) or die.
cd "$(dirname "$0")"
logs=(w15_pyr_5k_r1 w15_pyr_5k_r2 w15_pyr_5k_r3 w15_pyr_5k_r4 w15_hole_5k_r1 w15_hole_5k_r2)
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
echo "=== WAVE 15 COMPLETE ==="
echo "[pyramid variance]"
for f in w15_pyr_5k_r1 w15_pyr_5k_r2 w15_pyr_5k_r3 w15_pyr_5k_r4; do
  hd=$(grep -E "^hard_dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  echo "  [$f] hard_dice=${hd:-NA}"
done
echo "[hole stability]"
for f in w15_hole_5k_r1 w15_hole_5k_r2; do
  hd=$(grep -E "^hard_dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  echo "  [$f] hard_dice=${hd:-NA}"
done
