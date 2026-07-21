# VRAM scaling on TACC Lonestar6

This harness measures the current dense-history baseline on
`ar-agd/jul6-step-detail`. It covers both methods along two axes:

- fixed `N=128`, varying `T={64,128,256,1024,2048,2560,5120}`;
- fixed `T=128`, varying `N={48,96,128,192,256,320,352,448}`.

The last points intentionally bracket the expected allocation walls on both
40 and 80 GiB A100s. Every probe runs only two Adam iterations (three for the
confirmation), enough to materialize Taichi's lazy gradient fields. The
current sweep path still constructs `CSGSimulatorDelta`, so this is the
**before** curve: both methods retain the dense `(T+1) x N^3` stock history.

The harness uses three sequential workers rather than a 31-element job array.
This respects the current `gpu-a100-small` limit of three running jobs and
avoids unsupported GPU flags: Lonestar6 rejects `--gres` and
`--gpus-per-task`. Check live limits with `qlimits`; TACC warns that queue limits
can change. See the official [Lonestar6 running guide](https://docs.tacc.utexas.edu/hpc/lonestar6/).

## 1. Put the instrumented branch on TACC

Push the local branch before logging in. On Lonestar6:

```bash
cd "$SCRATCH/diff-cam"
git fetch origin
git checkout ar-agd/jul6-step-detail
git pull --ff-only
git merge-base --is-ancestor fb2a91e HEAD
```

The ancestry check must succeed: `fb2a91e` is the Taichi-inclusive VRAM
instrumentation. If dependencies are stale, request a dev GPU node and sync
there, not on the login node:

```bash
idev -p gpu-a100-dev -N 1 -n 1 -t 01:00:00
cd "$SCRATCH/diff-cam"
uv sync --frozen
exit
```

## 2. Submit the A100 gate

First submit only matrix row 0:

```bash
cd "$SCRATCH/diff-cam"
qlimits
sbatch --array=0 scripts/vram_scaling.slurm gate
```

When it finishes, collect the numeric job ID from `squeue`/`sacct` and inspect:

```bash
scripts/collect_vram_results.sh \
  "$SCRATCH/diffcam-vram/GATE_JOB_ID"
column -ts $'\t' "$SCRATCH/diffcam-vram/GATE_JOB_ID/results.tsv"
```

Proceed only if the row says `ok`, identifies an A100, and contains non-zero
`peak_vram_mb` and `peak_vram_delta_mb`. Compare the delta with
`analytic_dense_mib`; a small fixed CUDA-context/intercept difference is fine.

## 3. Submit both full sweeps

```bash
sbatch scripts/vram_scaling.slurm full
```

The default array has three tasks. Each task processes every third matrix row
sequentially, so all configs 1..30 run without exceeding the small-queue
three-job limit. Each probe has a 50-minute timeout and the worker continues
after OOM, illegal address, or timeout. The batch reservation is 12 hours, but
Slurm releases it as soon as the worker finishes.

Monitor and collect:

```bash
squeue -u "$USER"
sacct -j FULL_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
scripts/collect_vram_results.sh \
  "$SCRATCH/diffcam-vram/FULL_JOB_ID"
```

Override the checkout or output root without editing the scripts by exporting
the variables before `sbatch`:

```bash
export DIFF_CAM_DIR="$SCRATCH/diff-cam"
export VRAM_RESULTS_DIR="$SCRATCH/my-vram-run"
sbatch scripts/vram_scaling.slurm full
```

## Failure capture and task isolation

Each probe runs under `srun` inside its own directory, for example
`task_011/`. That isolates `runs/latest_metrics.json`, `args.json`, and the
repo-root trajectory copies that would otherwise race between workers.

Before Python starts, `run_vram_probe.sh` writes a one-row `result.tsv` with
outcome `running`. Its exit trap atomically replaces the row with one of:

- `ok`, `oom`, `cuda_illegal_address`, `timeout`;
- `oom_or_killed`, `missing_metrics`, `failed`, or `preflight_failed`.

This preserves wall-defining configurations even when the trainer exits before
writing `metrics.json`. If a completed Slurm task still says `running`, the
entire recorder received SIGKILL before its trap could run; retain the row and
inspect the Slurm output for a cgroup OOM message.

Successful rows include the git SHA, requested/actual dimensions, analytical
dense prediction, absolute and baseline-adjusted VRAM, CUDA device, and exact
log/metrics paths. `collect_vram_results.sh` retains failure rows when building
the combined `results.tsv`.

## Interpretation guardrails

- Plot `peak_vram_delta_mb` against the analytical model. `peak_vram_mb` is
  whole-device usage including the baseline.
- The scheduler assigns the GPU resource exclusively on Lonestar6, avoiding
  unrelated-process contamination of the baseline-adjusted measurement.
- Delta at `T>~192` is tagged `allocation_only`: its numerical path fails
  before its memory limit, so those points do not demonstrate usable training.
- Large sweep points are numerically meaningful, but the current code still
  allocates delta's unused stock history. Rerun the unchanged matrix after the
  two-slot/sweep-specific port for the flat-in-`T` comparison.
