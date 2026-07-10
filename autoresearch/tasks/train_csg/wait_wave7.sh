#!/bin/bash
# Block until all 8 wave-7 runs print a final summary (^hard_dice:) or die.
cd "$(dirname "$0")"
logs=(w7_sph_md_air1m3 w7_sph_md_air1m2 w7_sph_md_air1m1 w7_cyl_md_air1m2 w7_hole_rasterf_wr5 w7_hole_rasterf_wr5_lr3 w7_hole_rand_wr5_lr3 w7_hole_rand_wr3_lr3)
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
echo "=== WAVE 7 COMPLETE ==="
for f in "${logs[@]}"; do
  hd=$(grep -E "^hard_dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  dc=$(grep -E "^dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  gg=$(grep -E "^gouge:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  echo "[$f] hard_dice=${hd:-NA} soft_dice=${dc:-NA} gouge=${gg:-NA}"
done
