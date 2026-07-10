#!/bin/bash
# Block until all 8 wave-17 runs print a final summary (^hard_dice:) or die.
# Reports deployed hard_dice, final_iter_hard_dice, gouge, air_time_frac so the
# best_on_hard + w_tool_gouge tradeoff is directly visible.
cd "$(dirname "$0")"
logs=(w17_pyr_tg2 w17_pyr_tg8 w17_pyr_tg32 w17_sph_tg8 w17_sph_tg32 w17_bowl_tg8 w17_bowl_tg32 w17_hole_tg8)
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
echo "=== WAVE 17 COMPLETE: best_on_hard + w_tool_gouge sweep ==="
for f in "${logs[@]}"; do
  hd=$(grep -E "^hard_dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  # final_iter_hard_dice + gouge + air_time_frac live in metrics.json; map via the
  # run dir printed in the log.
  d=$(grep -oE "writing outputs to runs/\S+" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $4}')
  extra=""
  if [ -n "$d" ] && [ -f "../../$d/metrics.json" ]; then
    extra=$(python3 -c "import json;m=json.load(open('../../$d/metrics.json'));print(f'fihd={m[\"final_iter_hard_dice\"]} gouge={m[\"gouge\"]} airf={m[\"air_time_frac\"]} brk={m[\"break_prob_any\"]}')" 2>/dev/null)
  fi
  echo "[$f] hard_dice=${hd:-NA} ${extra}"
done
