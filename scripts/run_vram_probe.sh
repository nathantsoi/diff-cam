#!/bin/bash
# Run one short VRAM allocation probe and always leave a one-row TSV behind.
#
# This is normally called by vram_scaling.slurm. Keeping the recorder separate
# makes a single matrix row easy to reproduce on an interactive A100 node.

set -uo pipefail

if (( $# != 8 )); then
    echo "usage: $0 PROBE_ID SWEEP_AXIS METHOD N T ITERS NUMERICALLY_USABLE NOTE" >&2
    exit 2
fi

probe_id="$1"
sweep_axis="$2"
method="$3"
requested_n="$4"
max_steps="$5"
iters="$6"
numerically_usable="$7"
note="$8"

repo_dir="${DIFF_CAM_DIR:-${SCRATCH:?SCRATCH must be set when DIFF_CAM_DIR is unset}/diff-cam}"
required_commit="${VRAM_REQUIRED_COMMIT:-fb2a91e}"
array_job_id="${SLURM_ARRAY_JOB_ID:-local}"
array_task_id="${SLURM_ARRAY_TASK_ID:-$probe_id}"
slurm_job_id="${SLURM_JOB_ID:-local}"
result_root="${VRAM_RESULTS_DIR:-${SCRATCH:?SCRATCH must be set when VRAM_RESULTS_DIR is unset}/diffcam-vram/$array_job_id}"
task_name="$(printf 'task_%03d' "$probe_id")"
task_dir="$result_root/$task_name"
result_path="$task_dir/result.tsv"
log_path="$task_dir/train.log"

mkdir -p "$task_dir"

voxel_size_mm="$(awk -v n="$requested_n" 'BEGIN { printf "%.12g", 25.4 / n }')"
voxel_count=$((requested_n * requested_n * requested_n))
# Current dense CSG simulator model: value+grad history plus six N^3 fields.
analytic_dense_bytes=$((8 * (max_steps + 1) * voxel_count + 24 * voxel_count))
analytic_dense_mib="$(awk -v b="$analytic_dense_bytes" 'BEGIN { printf "%.6f", b / 1048576.0 }')"

git_sha=""
git_branch=""
hostname_value="$(hostname 2>/dev/null || printf unknown)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p')"
gpu="${gpu:-unavailable}"
start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
end_utc=""
elapsed_seconds=""
exit_status=""
outcome="running"
status_detail="probe started; an unchanged running row after job completion means the harness was killed"
run_dir=""
metrics_path=""
peak_vram_mb=""
peak_vram_delta_mb=""
vram_baseline_mb=""
vram_total_mb=""
cuda_device=""
vram_measurement=""
finalized=0
start_epoch="$(date +%s)"

write_result() {
    local tmp_path="$result_path.tmp.$$"
    local header=(
        probe_id sweep_axis method requested_n max_steps actual_nx actual_ny actual_nz
        voxel_size_mm iters numerically_usable note required_commit git_sha git_branch
        slurm_job_id slurm_array_job_id slurm_array_task_id hostname gpu start_utc end_utc
        elapsed_seconds exit_status outcome status_detail analytic_dense_bytes analytic_dense_mib
        run_dir metrics_path peak_vram_mb peak_vram_delta_mb vram_baseline_mb vram_total_mb
        cuda_device vram_measurement log_path
    )
    local row=(
        "$probe_id" "$sweep_axis" "$method" "$requested_n" "$max_steps"
        "$requested_n" "$requested_n" "$requested_n" "$voxel_size_mm" "$iters"
        "$numerically_usable" "$note" "$required_commit" "$git_sha" "$git_branch"
        "$slurm_job_id" "$array_job_id" "$array_task_id" "$hostname_value" "$gpu"
        "$start_utc" "$end_utc" "$elapsed_seconds" "$exit_status" "$outcome"
        "$status_detail" "$analytic_dense_bytes" "$analytic_dense_mib" "$run_dir"
        "$metrics_path" "$peak_vram_mb" "$peak_vram_delta_mb" "$vram_baseline_mb"
        "$vram_total_mb" "$cuda_device" "$vram_measurement" "$log_path"
    )
    {
        printf '%s' "${header[0]}"
        printf '\t%s' "${header[@]:1}"
        printf '\n%s' "${row[0]}"
        printf '\t%s' "${row[@]:1}"
        printf '\n'
    } > "$tmp_path"
    mv "$tmp_path" "$result_path"
}

finish_unexpected_exit() {
    local trap_status=$?
    if (( finalized == 0 )); then
        end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        elapsed_seconds=$(( $(date +%s) - start_epoch ))
        exit_status="${exit_status:-$trap_status}"
        if [[ "$outcome" == "running" ]]; then
            outcome="interrupted"
            status_detail="recorder exited before classifying the training process"
        fi
        write_result
    fi
}
trap finish_unexpected_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

# Pre-create the row. A SIGKILL or whole-job cgroup OOM cannot run a trap, so
# this preserves the configuration instead of silently dropping it.
write_result

preflight_fail() {
    status_detail="$1"
    outcome="preflight_failed"
    exit_status="2"
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    elapsed_seconds=$(( $(date +%s) - start_epoch ))
    write_result
    finalized=1
    echo "[vram-probe] $status_detail" >&2
    exit 2
}

git -C "$repo_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || preflight_fail "DIFF_CAM_DIR is not a git checkout: $repo_dir"
git_sha="$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null)" || preflight_fail "cannot resolve git SHA"
git_branch="$(git -C "$repo_dir" branch --show-current 2>/dev/null)"
if ! git -C "$repo_dir" merge-base --is-ancestor "$required_commit" HEAD 2>/dev/null; then
    preflight_fail "HEAD $git_sha does not contain required VRAM commit $required_commit"
fi
[[ "$method" == "delta" || "$method" == "sweep" ]] || preflight_fail "unsupported method: $method"
[[ "$requested_n" =~ ^[0-9]+$ && "$requested_n" -gt 0 ]] || preflight_fail "N must be a positive integer"
[[ "$max_steps" =~ ^[0-9]+$ && "$max_steps" -gt 1 ]] || preflight_fail "T must be an integer greater than one"
command -v python >/dev/null 2>&1 || preflight_fail "python is unavailable; activate the project venv"

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$task_dir" || preflight_fail "cannot enter task directory: $task_dir"

train_cmd=(
    python -u -m algorithms.train_csg
    --exp-name "vram_${probe_id}_${method}_n${requested_n}_t${max_steps}"
    --method "$method"
    --iters "$iters"
    --resolution "$requested_n"
    --max-steps "$max_steps"
    --stock-size-in 1 1 1
    --voxel-size-mm "$voxel_size_mm"
    --eval-freq 0
    --record-video-freq 0
    --log-freq 1
    --headless
    --no-track
)

# A separate srun step lets the batch shell survive and record the exit status
# when Slurm kills the training step for memory use. It also preserves the GPU
# binding assigned by Lonestar6.
launch_cmd=("${train_cmd[@]}")
if [[ -n "${SLURM_JOB_ID:-}" && "${VRAM_USE_SRUN:-1}" == "1" ]] && command -v srun >/dev/null 2>&1; then
    launch_cmd=(srun --nodes=1 --ntasks=1 "${train_cmd[@]}")
fi

echo "[vram-probe] id=$probe_id axis=$sweep_axis method=$method N=$requested_n T=$max_steps iters=$iters"
echo "[vram-probe] git=$git_sha branch=${git_branch:-detached} gpu=$gpu"
echo "[vram-probe] voxel_size_mm=$voxel_size_mm analytic_dense_mib=$analytic_dense_mib"
printf '[vram-probe] command:'
printf ' %q' "${launch_cmd[@]}"
printf '\n'

probe_timeout="${VRAM_PROBE_TIMEOUT:-50m}"
timeout --signal=TERM --kill-after=30s "$probe_timeout" "${launch_cmd[@]}" 2>&1 | tee "$log_path"
exit_status=${PIPESTATUS[0]}

run_dir="$(sed -n 's/.*writing outputs to \(runs\/[^[:space:]]*\).*/\1/p' "$log_path" | tail -n 1)"
if [[ -n "$run_dir" ]]; then
    metrics_path="$task_dir/$run_dir/metrics.json"
fi

if [[ -n "$metrics_path" && -f "$metrics_path" ]]; then
    metrics_values="$(python -c '
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
keys = ("peak_vram_mb", "peak_vram_delta_mb", "vram_baseline_mb", "vram_total_mb", "cuda_device", "vram_measurement")
print("\t".join(str(m.get(k, "")) for k in keys))
' "$metrics_path")"
    IFS=$'\t' read -r peak_vram_mb peak_vram_delta_mb vram_baseline_mb vram_total_mb cuda_device vram_measurement <<< "$metrics_values"
fi

if (( exit_status == 0 )) && [[ -f "$metrics_path" ]]; then
    outcome="ok"
    status_detail="metrics captured"
elif (( exit_status == 0 )); then
    outcome="missing_metrics"
    status_detail="training exited zero but metrics.json was not found"
elif grep -Eiq 'out[ _]of[ _]memory|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|cuMemAlloc|failed to allocate|oom-kill' "$log_path"; then
    outcome="oom"
    status_detail="non-zero exit with an explicit OOM signature in train.log"
elif grep -Eiq 'illegal (memory )?access|CUDA_ERROR_ILLEGAL_ADDRESS' "$log_path"; then
    outcome="cuda_illegal_address"
    status_detail="non-zero exit with an illegal-address signature in train.log"
elif (( exit_status == 124 )); then
    outcome="timeout"
    status_detail="probe exceeded $probe_timeout; partial log retained"
elif (( exit_status == 137 )); then
    outcome="oom_or_killed"
    status_detail="exit 137 (SIGKILL); inspect the Slurm .out file for an oom-kill event"
else
    outcome="failed"
    status_detail="training returned non-zero exit $exit_status; inspect train.log"
fi

end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_seconds=$(( $(date +%s) - start_epoch ))
write_result
finalized=1

echo "[vram-probe] outcome=$outcome exit_status=$exit_status result=$result_path"
exit "$exit_status"
