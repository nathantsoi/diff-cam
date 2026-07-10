#!/bin/bash
# Block until all 8 wave-1 runs print a final summary (^hard_dice:) or die.
logs=(run_baseline run_zlayer run_shell run_rf run_rfw run_raster run_spiral run_zlayer_dense)
declare -A done
while true; do
  alldone=1
  for f in "${logs[@]}"; do
    [ "${done[$f]:-0}" = "1" ] && continue
    if grep -qE "^hard_dice:" "$f.log" 2>/dev/null; then
      done[$f]=1
    elif ! pgrep -f "init_mode ${f#run_}" >/dev/null 2>&1 && ! pgrep -f "run_pipeline" >/dev/null 2>&1; then
      # no train procs left at all -> everything finished/crashed; mark all done
      for g in "${logs[@]}"; do done[$g]=1; done
      alldone=1; break
    else
      alldone=0
    fi
  done
  [ "$alldone" = "1" ] && break
  sleep 30
done
echo "=== WAVE 1 COMPLETE ==="
for f in "${logs[@]}"; do
  hd=$(grep -E "^hard_dice:" "$f.log" 2>/dev/null | tail -1 | awk '{print $2}')
  dc=$(grep -E "^dice:" "$f.log" 2>/dev/null | tail -1 | awk '{print $2}')
  li=$(grep -oE "iter +[0-9]+/5000" "$f.log" | tail -1)
  echo "[$f] hard_dice=${hd:-NA} soft_dice=${dc:-NA} last=$li"
done
