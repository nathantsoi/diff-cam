#!/bin/bash
# Collect one finished run's results into results.tsv and move its run dir into
# runs/<TAG>/. Usage: collect.sh <tag> <name> <log> <status> <description> <command>
# Reads hard_dice / dice / peak_vram_mb from the log summary lines, and the
# trajectory-quality measures from runs/latest_metrics.json.
set -u
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
TAG="$1"; NAME="$2"; LOG="$3"; STATUS="$4"; DESC="$5"; CMD="$6"
TSV=autoresearch/tasks/train_csg/results.tsv

dice=$(grep -E "^dice:" "$LOG" 2>/dev/null | awk '{print $2}' | tail -1)
hdice=$(grep -E "^hard_dice:" "$LOG" 2>/dev/null | awk '{print $2}' | tail -1)
vram=$(grep -E "^peak_vram_mb:" "$LOG" 2>/dev/null | awk '{print $2}' | tail -1)
[ -z "$dice" ] && dice="0.000000"
[ -z "$vram" ] && vram="0.0"
mem=$(awk -v v="$vram" 'BEGIN{printf "%.1f", v/1024}')

# trajectory-quality from the metrics json (the ^metric: lines lack them)
air="NA"; ttime="NA"; brk="NA"; impr="NA"
if [ -f runs/latest_metrics.json ]; then
  readjson=$(python3 -c "
import json
try:
    m=json.load(open('runs/latest_metrics.json'))
    print(m.get('air_time','NA'), m.get('total_time','NA'), m.get('break_prob_any','NA'), m.get('dice_improvement','NA'))
except Exception as e:
    print('NA NA NA NA')
" 2>/dev/null)
  read air ttime brk impr <<<"$readjson"
fi

# Move the run dir into runs/<TAG>/
rundir=$(grep -oE "writing outputs to runs/[^ ]+" "$LOG" 2>/dev/null | awk '{print $4}' | head -1)
if [ -n "$rundir" ] && [ -d "$rundir" ]; then
  mv "$rundir" "runs/$TAG/" 2>/dev/null
fi

# description: lead with hard_dice, then soft dice + quality measures
full="hard_dice=${hdice:-NA} soft_dice=${dice} ${DESC} (air=${air} t=${ttime} brk=${brk} impr=${impr})"
# TSV row: commit \t dice \t mem \t status \t desc \t cmd  (NO literal tabs in desc/cmd)
full=$(echo "$full" | tr '\t' ' ')
CMD=$(echo "$CMD" | tr '\t' ' ')
COMMIT=$(git rev-parse --short HEAD)
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$COMMIT" "$dice" "$mem" "$STATUS" "$full" "$CMD" >> "$TSV"
echo "[collect] $NAME -> hdice=${hdice} dice=${dice} mem=${mem}GB status=${STATUS}  (moved $rundir)"
