#!/bin/bash
# Block until all 8 wave-18 runs print ^hard_dice: or die. Reports deployed
# hard_dice, final_iter_hard_dice, gouge, air_time_frac (absolute-path metrics).
cd "$(dirname "$0")"
REPO=/home/ntsoi/papers/icra26-diffcam/diff-cam
logs=(w18_pyr_tg8_r1 w18_pyr_tg8_r2 w18_pyr_tg8_r3 w18_sph_k150_r1 w18_sph_k150_r2 w18_sph_k150_r3 w18_bowl_rev w18_hole_rev)
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
echo "=== WAVE 18 COMPLETE: pyr tg8 confirm + sph k150 gouge-probe + bowl/hole revert ==="
for f in "${logs[@]}"; do
  hd=$(grep -E "^hard_dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  d=$(grep -oE "writing outputs to runs/\S+" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $4}')
  extra=""
  if [ -n "$d" ] && [ -f "$REPO/$d/metrics.json" ]; then
    extra=$(python3 -c "import json;m=json.load(open('$REPO/$d/metrics.json'));print(f'fihd={m[\"final_iter_hard_dice\"]} gouge={m[\"gouge\"]} airf={m[\"air_time_frac\"]} brk={m[\"break_prob_any\"]}')" 2>/dev/null)
  fi
  echo "[$f] hard_dice=${hd:-NA} ${extra}"
done
echo "--- pyr tg8 mean (vs w16 0.674, w17 0.714) ---"
echo "--- sph k150 (vs w16 0.777 gouge572, w17 0.818 gouge585) ---"
