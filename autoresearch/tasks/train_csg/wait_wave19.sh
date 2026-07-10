#!/bin/bash
# Block until all 8 wave-19 runs print ^hard_dice: or die. Reports deployed
# hard_dice, final_iter_hard_dice, gouge, air_time_frac, loss_tool_gouge
# (absolute-path metrics) -- loss_tool_gouge is the key diagnostic for whether
# the margin activated the barrier (was 0.0 on sph at margin=0).
cd "$(dirname "$0")"
REPO=/home/ntsoi/papers/icra26-diffcam/diff-cam
logs=(w19_sph_tg8_m0 w19_sph_tg8_m0p5 w19_sph_tg8_m1 w19_sph_tg8_m2 w19_sph_tg8_m4 w19_sph_tg8_m8 w19_pyr_tg8_m2 w19_bowl_tg4_m2)
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
echo "=== WAVE 19 COMPLETE: tool-gouge MARGIN probe (sph dose-response + pyr/bowl sanity) ==="
for f in "${logs[@]}"; do
  hd=$(grep -E "^hard_dice:" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $2}')
  d=$(grep -oE "writing outputs to runs/\S+" "run_${f}.log" 2>/dev/null | tail -1 | awk '{print $4}')
  extra=""
  if [ -n "$d" ] && [ -f "$REPO/$d/metrics.json" ]; then
    extra=$(python3 -c "import json;m=json.load(open('$REPO/$d/metrics.json'));print(f'fihd={m[\"final_iter_hard_dice\"]:.4f} gouge={m[\"gouge\"]:.1f} airf={m[\"air_time_frac\"]:.3f} brk={m[\"break_prob_any\"]:.4f} tg_loss={m.get(\"loss_tool_gouge\",0):.4f}')" 2>/dev/null)
  fi
  echo "[$f] hard_dice=${hd:-NA} ${extra}"
done
echo "--- sph margin dose-response (vs w18 k70-equivalent ~0.79 hard, ~620 gouge, tg_loss~0) ---"
echo "--- pyr tg8 m2 (vs w18 tg8 m0: dice 0.700, gouge ~75) -- does margin break it? ---"
echo "--- bowl tg4 m2 (vs w18: dice 0.564, gouge 962) -- does margin relieve bowl gouge? ---"
