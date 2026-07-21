#!/bin/bash
# TACC Lonestar6 A100 VRAM scaling matrix.
#
# First submit the A100 confirmation only:
#   sbatch --array=0 scripts/vram_scaling.slurm gate
# After checking its peak_vram_delta_mb, submit three workers which divide the
# two full sweeps among themselves:
#   sbatch scripts/vram_scaling.slurm full

#SBATCH -J diffcam-vram
#SBATCH -o diffcam-vram.o%A_%a
#SBATCH -e diffcam-vram.e%A_%a
#SBATCH -p gpu-a100-small
#SBATCH -A IRI25030
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --array=0-2%3

set -uo pipefail
umask 0027

module load gcc/13.2.0 cuda/12.8

repo_dir="${DIFF_CAM_DIR:-${SCRATCH:?SCRATCH must be set}/diff-cam}"
matrix_path="$repo_dir/scripts/vram_scaling_matrix.tsv"
result_root="${VRAM_RESULTS_DIR:-${SCRATCH:?SCRATCH must be set}/diffcam-vram/$SLURM_ARRAY_JOB_ID}"

cd "$repo_dir"
source .venv/bin/activate
export DIFF_CAM_DIR="$repo_dir"
export VRAM_RESULTS_DIR="$result_root"

mode="${1:-full}"
if [[ "$mode" == "gate" ]]; then
    if [[ "$SLURM_ARRAY_TASK_ID" -ne 0 ]]; then
        echo "gate mode only uses array task 0" >&2
        exit 2
    fi
    probe_ids=(0)
elif [[ "$mode" == "full" ]]; then
    probe_ids=()
    for ((probe_id = 1 + SLURM_ARRAY_TASK_ID; probe_id <= 30; probe_id += 3)); do
        probe_ids+=("$probe_id")
    done
else
    echo "unknown mode '$mode' (expected gate or full)" >&2
    exit 2
fi

echo "=== diff-cam VRAM allocation probe ==="
echo "Date UTC : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Node     : $(hostname)"
echo "Job/task : $SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID"
echo "Repo     : $repo_dir"
echo "Results  : $result_root"
echo "Mode/IDs : $mode / ${probe_ids[*]}"
git rev-parse HEAD
nvidia-smi -L || true

worker_status=0
for probe_id in "${probe_ids[@]}"; do
    row="$(awk -F '\t' -v wanted="$probe_id" 'NR > 1 && $1 == wanted { print; exit }' "$matrix_path")"
    if [[ -z "$row" ]]; then
        echo "No VRAM matrix row for probe $probe_id" >&2
        worker_status=2
        continue
    fi
    IFS=$'\t' read -r _ sweep_axis method requested_n max_steps iters numerically_usable note <<< "$row"
    echo "--- probe $probe_id: $row ---"
    "$repo_dir/scripts/run_vram_probe.sh" \
        "$probe_id" "$sweep_axis" "$method" "$requested_n" "$max_steps" \
        "$iters" "$numerically_usable" "$note"
    probe_status=$?
    result_path="$(printf '%s/task_%03d/result.tsv' "$result_root" "$probe_id")"
    if [[ ! -f "$result_path" ]]; then
        echo "probe $probe_id left no result row (rc=$probe_status)" >&2
        worker_status=2
    elif [[ "$probe_status" -ne 0 ]]; then
        # OOM / illegal address / timeout are expected wall-defining outcomes.
        echo "probe $probe_id returned $probe_status; outcome preserved in $result_path"
    fi
done

exit "$worker_status"
