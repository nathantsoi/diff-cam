# diff-cam / GradMill

A differentiable CNC machining simulator (Taichi) and the experiment harness for
the **GradMill** paper — gradient-directed milling for continuous volumetric
process planning. The repo supports two process-planning **modes** and a CAM
layer that turns optimized toolpaths into machine-ready G-code.

| Mode | State representation | Action space | Methods | Env id |
|------|----------------------|--------------|---------|--------|
| **Continuous (CSG / GradMill)** | Parametric SDFs, smooth boolean cuts | continuous `Box(3)` delta moves | analytic gradient descent **and** PPO | `CamEnvDiff-v0` |
| **Discrete (voxel)** | voxel SDF grid | `Discrete(27)` (±1 per axis) | PPO | `CamEnvDisc-v0` |

> Run everything as a **module** from the repo root (`python -m algorithms.…`,
> `python -m eval.…`). Running a script by path (`python algorithms/ppo.py`)
> breaks the `cam_env` / `simulator` imports.

## Setup

Install [uv](https://docs.astral.sh/uv/) and sync the environment:

```bash
uv sync
```

GPU work runs on Taichi's CUDA backend. Live GUIs (the continuous gradient-descent
viewer and the discrete env's GGUI window) need a display; on headless machines
install TurboVNC and start a server, then run inside that session:

```bash
./scripts/setup.sh
```

All training/eval entry points also run **headless** (see the flags below), so a
display is only needed for interactive visualization.

## Repository layout

```
simulator/
  csg_simulator.py    # continuous, differentiable CSG simulator (PSDFs); configurable (Mini Mill) envelope, cubic voxels, feed/rapid speed clipping
  voxel_simulator.py  # discrete voxel simulator (CNCSimulator)
  csg_metrics.py      # Dice / ASD / HD95 + gouge/residual on SDF grids
cam_env/
  cam_env.py          # continuous Gymnasium env  -> CamEnvDiff-v0
  cam_env_voxel.py    # discrete  Gymnasium env  -> CamEnvDisc-v0
algorithms/
  train_csg.py        # continuous: analytic gradient descent (GradMill, Method 1)
  csg_ppo.py          # continuous: PPO baseline (Method 2)
  ppo.py              # discrete: PPO (pufferlib)
eval/
  eval_csg.py         # evaluate continuous trajectories / checkpoints
  eval.py             # evaluate discrete checkpoints
cam/                  # trajectory -> G-code -> executed-trajectory (CAM layer)
  units.py            # inch<->mm / feed-rate conversions (I/O boundary; internal is mm)
scripts/              # launchers, SLURM jobs, round-trip + export demos
```

## Training

### Continuous — analytic gradient descent (GradMill, Method 1)

Directly optimizes the per-step delta toolpath through the differentiable CSG
simulator (no policy/RL). Logging, metrics, video encoding and STL export reuse
the **same code paths** as the PPO trainers, so runs are directly comparable.
Outputs land under `runs/CamEnvDiff-v0__train_csg__<seed>__<ts>/` (same
env/simulator as `csg_ppo`; `train_csg` in the name marks the method): the learned
`trajectory.npy` / `trajectory_deltas.npy` (read by the evaluator and CAM layer)
are written there with `--save_model` (and always copied to the repo root for the
CAM round-trip demo), alongside `videos/`, `meshes/` and a `metrics.png`. It
shares the `--eval_freq` / `--record_video_freq` / `--video_fps` / `--track`
flags with the PPO trainers — see
[Recording policy videos](#recording-policy-videos-during-training).

`--resolution 203` gives **2 mm cubic voxels** on the Mini Mill envelope, and
`--dt 0.4` makes each feed step ≈ 1 voxel (see [Machine envelope & resolution](#machine-envelope--resolution)).

```bash
# Headless (HPC / no display) — comparable to the csg_ppo baseline below:
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 --resolution 203 --dt 0.4 \
    --save_model --eval_freq 1 --record_video_freq 100 --video_fps 30

# Interactive live GUI (needs a display):
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 --resolution 203 --dt 0.4
```

#### Machine envelope & resolution

The simulator runs on a configurable, possibly **non-cubic** machine envelope.
The default is the **Haas Mini Mill** cutting volume — **16 × 12 × 10 in**
(x, y; z up) = 406.4 × 304.8 × 254.0 mm. Geometry lives in the normalized box
`[0,1]³` (so trajectories stay scale-free), but the voxel grid uses **per-axis
dimensions** `(Nx, Ny, Nz)` chosen so every voxel is a physical **cube** of side
`v` mm — keeping the cutter and cuts undistorted. `--resolution` is the voxel
count along the **longest** axis; the others get proportionally fewer voxels.
Set the envelope with `--workspace_in X Y Z` (inches); tool/target sizes are in
millimetres (`--tool_radius_mm`, `--tool_height_mm`, `--target_radius_mm`).

**Resolution for non-sub-voxel cuts.** A cut only registers if the cutter spans
several voxels. The `stock[max_steps+1, Nx,Ny,Nz]` field dominates memory, and
under autodiff it is allocated **twice** (value + gradient), so peak VRAM is
≈ `2 × (max_steps+1) × Nx·Ny·Nz × 4 B`. Below: peak at the default
`--max_steps 64` (65 slices, f32), and whether it fits an **80 GB** GPU:

| voxel `v` | `--resolution` | grid `Nx×Ny×Nz` | peak VRAM (value+grad) | fits 80 GB | 1/4″ tool Ø |
|-----------|----------------|------------------|------------------------|------------|-------------|
| 4 mm   | 102 | 102×76×64   | 0.3 GB  | ✅ | 1.6 vox |
| 3 mm   | 135 | 135×102×85  | 0.6 GB  | ✅ | 2.1 vox |
| **2 mm** | **203** | **203×152×127** | **2.0 GB** | ✅ | **3.2 vox** |
| 1.5 mm | 271 | 271×203×169 | 4.8 GB  | ✅ | 4.2 vox |
| 1 mm   | 406 | 406×305×254 | 16.4 GB | ✅ | 6.3 vox |
| 0.8 mm | 508 | 508×381×318 | 32.0 GB | ✅ | 7.9 vox |
| 0.6 mm | 677 | 677×508×423 | 75.6 GB | ✅ (tight) | 10.6 vox |
| 0.5 mm | 813 | 813×610×508 | 131 GB  | ❌ | 12.7 vox |

→ **~2 mm voxels** (`--resolution 203`, the default in the examples) is a strong
sweet spot — a 1/4″ cutter spans ~3 voxels at only ~2 GB peak. With **80 GB** of
VRAM there is large headroom: go to **1 mm** (`--resolution 406`, ~16 GB, ~6
voxels across the cutter) for finer cuts, or down to ~**0.6 mm** (~76 GB) if you
also drop `--max_steps` (peak scales linearly with it). 0.5 mm needs ~131 GB —
halve `--max_steps` to fit.

#### Units & speed limits

The differentiable simulator enforces two machine-style **max speeds** by
clipping each step, mirroring a real controller's feed/rapid override (cf. the
LinuxCNC trajectory planner and CAMotics). A per-step displacement `delta`
(normalized) spans `delta · L` mm on the per-axis envelope `L`, over `dt`
seconds, so its speed is `|delta · L| / dt` (mm/s). All scale-related math is in
**millimetres** internally — inches appear only at the I/O boundary (inch inputs
converted up front; G-code requested in inches converted just before emission),
via the `cam.units` helper library.

Two regimes are clipped per step:

- **rapid** — when the cutter has clearance from the remaining stock (a
  *traverse*);
- **feed** — when the cutter is within `safe_distance` of the remaining stock (it
  is *cutting*).

The regime is decided by probing the **commanded destination** against the
remaining stock at the start of the step ("am I moving into material?"). When a
commanded move exceeds the regime's max speed, the step is scaled down so the
*actual* move runs at the cap (direction preserved); gradients still flow through
the clipped magnitude into `tool_delta`.

| flag | default | meaning |
|------|---------|---------|
| `--workspace_in X Y Z` | `16 12 10` | machine envelope (in), default Mini Mill |
| `--dt F` | `0.01` | seconds per simulator step; sets the speed scale |
| `--rapid_ipm F` | `500.0` | max traverse speed (inches/min) when clear of the stock |
| `--feed_ipm F` | `10.0` | max cutting speed (inches/min) when near the stock |
| `--safe_distance_in F` | `0.1` | clearance (inches) below which a move is limited to feed speed |
| `--enforce_speed_limits` / `--no-enforce_speed_limits` | enabled | clip each step to its feed/rapid cap (disable to run unconstrained) |

The saved `trajectory.npy` is the **speed-clipped** path that was actually carved
(not the raw cumulative sum of commanded deltas), so the exported G-code matches
the optimized result.

> **Note on `dt`.** The speed caps become per-step displacement caps
> (`speed · dt`). At the default `dt=0.01` the feed cap is ~0.042 mm/step — far
> below any feasible voxel — so each feed step is sub-voxel *in time*. To advance
> ~1 voxel per feed step pair `~2 mm` voxels with `--dt ≈ 0.3–0.5` (≈ voxel/feed),
> or pass `--no-enforce_speed_limits` to disable the constraint.

### Continuous — PPO baseline (Method 2)

> **Resolution note.** The PPO examples stay at `--resolution 32` (not the 2 mm /
> `--resolution 203` used for gradient descent). The PPO observation embeds the
> **full `stock` and `target` voxel grids** (~7.8M floats, ~31 MB each at 2 mm),
> so the `num_steps × num_envs` rollout buffer is the bottleneck: at 2 mm it is
> ~16 GB for `--num_envs 1` (fits an 80 GB GPU, but heavy/slow) and ~64 GB for
> `--num_envs 4`; at 1 mm it is ~129 GB (infeasible). The env also uses the
> simulator's default `dt` (no `--dt` flag). The gradient-descent method
> (`train_csg`) keeps no rollout buffer and runs at 2 mm in ~2 GB. So for PPO,
> push `--resolution` up only with `--num_envs 1` and watch VRAM; the examples
> stay at 32 for practicality.

```bash
uv run python -m algorithms.csg_ppo \
    --total_timesteps 10000000 --num_envs 1 --resolution 32 --max_steps 64 \
    --num_steps 512 --num_minibatches 8 --update_epochs 4 \
    --learning_rate 3e-4 --ent_coef 0.02 --save_model
```

`--save_model` writes a self-describing checkpoint
`runs/<run>/csg_ppo.cleanrl_model` (`{"agent", "args"}`) that the evaluator
reads. Convenience wrapper: `bash train_local.bash` (it calls `uv run` internally).
`--num_envs` runs that many parallel environments (each its own simulator).

Add `--record_video_freq N` to also record + upload a greedy policy-rollout video
(and Dice/ASD/HD95 metrics) to wandb every `N` iterations — see
[Recording policy videos](#recording-policy-videos-during-training):

```bash
uv run python -m algorithms.csg_ppo \
    --total_timesteps 10000000 --num_envs 4 --resolution 32 --max_steps 64 \
    --num_steps 512 --num_minibatches 8 --update_epochs 4 --save_model \
    --record_video_freq 100 --video_fps 30
```

### Discrete — PPO (voxel)

```bash
uv run python -m algorithms.ppo \
    --total-timesteps 2000000 --num-envs 4 --resolution 32 --max-steps 256 \
    --render_mode rgb_array
```

Checkpoints are written to `runs/<run>/checkpoint_{iter}.pt` and
`checkpoint_final.pt`.

### Recording policy videos during training

All three trainers can render the policy/toolpath **headlessly** during training
and upload short videos — plus the same Dice/ASD/HD95 metrics the evaluator
reports — to Weights & Biases. No display is needed, so this works on HPC nodes
and inside the Apptainer image (which ships `ffmpeg`). It is fully optional and
off by default; if encoding ever fails it warns and training continues.

Both PPO trainers (`csg_ppo.py` and `ppo.py`) share the same flags — a dedicated
off-screen env rolls the current policy out *greedily* (like the evaluator). Eval
metrics and video recording are on independent cadences: a cheap metrics-only
eval can run frequently, while the (more expensive) video is encoded only at
`--record_video_freq`:

| flag | default | meaning |
|------|---------|---------|
| `--eval` | `False` | compute evaluation metrics (Dice/ASD/HD95/reward) during training and at the end |
| `--eval_freq N` | `0` | run a greedy eval rollout + log Dice/ASD/HD95/reward every `N` iterations (`0` disables). No video is encoded unless the video cadence also lands on that iteration. |
| `--record_video_freq N` | `0` | additionally record + upload a greedy rollout **video** every `N` iterations (`0` disables) |
| `--progress_bar` | `False` | use interactive `tqdm` progress bar instead of clean scrolling log lines (set `False` for clean log files and LLM harness compatibility) |
| `--log_freq N` | `1` | print scrolling log output every `N` iterations when `--progress_bar` is disabled |
| `--video_fps F` | `30` | frame rate of the encoded mp4 |
| `--video_seed S` | `0` | seed for the rollout env, so the scenario is fixed across iterations and runs are comparable |
| `--track` / `--no-track` | `--track` | upload to wandb; with `--no-track` the mp4s are still written locally |

The recorder is built if **either** cadence is enabled or `--eval` is passed. Metrics land in wandb
under `eval/reward`, `eval/dice`, `eval/asd`, `eval/hd95` (logged on every eval
*and* every video); videos appear under `media/policy_rollout`, with local copies
at `runs/<run>/videos/policy_step_<global_step>.mp4`.

After training finishes (including on early stopping), the **final** model's
geometry is also exported as STL meshes — the initial uncarved stock, the carved
stock, and the target part — to `runs/<run>/meshes/` (and uploaded to wandb when
`--track`). Meshes are the zero-level surface of each SDF (marching cubes) in the
unit-cube frame. Furthermore, all training scripts output a structured summary table to console and save machine-readable evaluation results to `runs/<run>/metrics.json` and `runs/latest_metrics.json` (containing `dice`, `asd`, `hd95`, `reward`, `training_seconds`, `peak_vram_mb`, `num_steps`), making automated monitoring and LLM research harnesses (such as autoresearch agents) seamless.

```bash
# Continuous PPO, no video (default):
uv run python -m algorithms.csg_ppo --total_timesteps 10000000 --num_envs 4 \
    --resolution 32 --max_steps 64 --save_model

# Continuous PPO for autoresearch / LLM harnesses (clean scrolling logs, no W&B sync):
uv run python -m algorithms.csg_ppo --total_timesteps 10000000 --num_envs 4 \
    --resolution 32 --max_steps 64 --eval --no-track
```

The discrete voxel PPO (`ppo.py`) renders its 3D GGUI scene off-screen for the
video:

```bash
# Discrete PPO, no video (default):
uv run python -m algorithms.ppo --total-timesteps 2000000 --num-envs 4 \
    --resolution 32 --max-steps 256

# Discrete PPO for autoresearch / LLM harnesses:
uv run python -m algorithms.ppo --total-timesteps 2000000 --num-envs 4 \
    --resolution 32 --max-steps 256 --eval --no-track
```

**Gradient descent (`train_csg.py`).** Uses the **same** flags, encoder and
metric/STL code paths as the PPO trainers above — `--eval`, `--eval_freq`, `--progress_bar`, `--log_freq`,
`--record_video_freq`, `--video_fps`, `--track` / `--no-track` — measured in Adam
iterations. Each video raymarches the optimized toolpath and is encoded to mp4
(ffmpeg) at `runs/<run>/videos/policy_step_<iter>.mp4`, uploaded to
`media/policy_rollout`. Geometry metrics land under `eval/dice`, `eval/asd`,
`eval/hd95` (the differentiable objective is logged as `losses/loss`; there is
**no `eval/reward`**, since this method has no RL reward), with `gouge`/`residual`
under `metrics/*`. The final STL meshes and `metrics.json` files are exported just like the PPO trainers.
Pass `--headless` to skip the live GUI (auto-disabled when no display is present).

```bash
# Headless gradient descent for autoresearch / LLM harnesses:
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 --resolution 203 --dt 0.4 \
    --save_model --eval --no-track
```

> The continuous env renders by GPU raymarching; the discrete voxel env renders
> its 3D particle scene through an off-screen Taichi GGUI window — both work
> without a display.

### HPC (TACC Lonestar6)

`cd $SCRATCH/diff-cam`, set your allocation/email, then:

```bash
sbatch train_hpc.bash      # continuous PPO inside the Apptainer image
sbatch scripts/train.slurm # discrete PPO in a venv
```

## Evaluation

All modes report the same geometric metrics of the final carved stock vs. the
target: **Dice** (DSC), **Average Surface Distance** (ASD), and **95% Hausdorff
Distance** (HD95).

### Continuous (`eval/eval_csg.py`)

Evaluate a gradient-descent trajectory:

```bash
uv run python -m eval.eval_csg --trajectory trajectory.npy --resolution 203
```

(`--resolution 203` ≈ 2 mm voxels on the Mini Mill envelope, matching training;
carving doesn't use `--dt`.)

Evaluate continuous-PPO checkpoint(s) over paired random episodes:

```bash
uv run python -m eval.eval_csg --checkpoints runs/*/csg_ppo.cleanrl_model --num-runs 10
```

### Discrete (`eval/eval.py`)

```bash
uv run python -m eval.eval \
    --checkpoints runs/CamEnvDisc-v0__ppo__*/checkpoint_final.pt \
    --num-runs 10 --no-render
```

Add `--render` for the interactive GGUI playback (needs a display).

## G-code based evaluation and the Haas Mini Mill

The `cam/` package bridges optimized trajectories and real CNC G-code, following
the LinuxCNC pipeline (interpreter → canonical moves → trajectory planner). The
G-code dialect is chosen by a **post-processor** (`cam/posts.py`):

- `rs274` — generic RS274/NGC (`G21`/`G90`/`G61`, `G0`/`G1`, `M2`); the faithful,
  approach-free output used for round-trip fidelity checks.
- `haas` — Fanuc-style program for a **Haas Mini Mill**: program number, safety
  block, tool change with length offset (`Txx M06` + `G43 Hxx`), spindle + flood
  coolant, an explicit clearance/plunge approach, retract, and `M30`.

Add new machines by subclassing `PostProcessor` and registering them in
`cam.posts.POSTS`.

Internally the CAM layer (and the simulator) work in **millimetres**; pass
`--units inch` to emit an inch program (`G20`) instead of mm (`G21`). Coordinates
and feeds are converted mm→inch just before emission via `cam.units`, so the
program round-trips exactly through the inch-aware parser.

**Generate G-code for the Haas:**

The exporter defaults to the same Mini Mill envelope (`--workspace-in 16 12 10`,
inches); pass `--workspace-mm <edge>` to fall back to an isotropic cube instead.

```bash
# Using the default root trajectory (Mini Mill envelope)
uv run python scripts/export_gcode.py --post haas --tool 3 --rpm 6000 \
    --program-number 1234 -o part.nc

# Using a trajectory from a specific run folder
uv run python scripts/export_gcode.py \
    --trajectory runs/CamEnvDiff-v0__train_csg__1__1782599757/trajectory.npy \
    --post haas \
    -o runs/CamEnvDiff-v0__train_csg__1__1782599757/gcode_haas.nc
```

**Evaluate based on the G-code program** (round-trip the trajectory through the
CAM layer and score both path fidelity and the carved stock of the *executed*
program):

```bash
# Using the default root trajectory
uv run python -m eval.eval_csg --trajectory trajectory.npy --resolution 203 --gcode --post rs274

# Using a trajectory from a specific run folder
uv run python -m eval.eval_csg \
    --trajectory runs/CamEnvDiff-v0__train_csg__1__1782599757/trajectory.npy \
    --resolution 203 \
    --gcode \
    --post rs274
```

**End-to-end CAM round-trip demo** (`trajectory → G-code → executed trajectory`,
reporting Fréchet/DTW/RMSE and carved-stock Dice):

```bash
uv run python scripts/roundtrip_demo.py                 # rs274 (default)
uv run python scripts/roundtrip_demo.py --post haas     # parse/plan the Haas program
```

For the saved trajectory the `rs274` round trip is approach-free, so fidelity is
≈ machine precision and carved-stock Dice ≈ 0.99. The `haas` post intentionally
adds clearance/plunge/retract moves for the real machine, so its waypoint count
differs from the source path (the strict waypoint metric is reported as `n/a`);
use it to sanity-check that the Haas program parses and plans, not for exact
round-trip fidelity.

### CAM API

- `cam.trajectory_to_gcode(positions, config, post="rs274")` — export an
  `(T, 3)` unit-cube trajectory to G-code with the chosen post.
- `cam.parse_gcode(text)` — parse G-code back to motion segments (G0/G1 plus
  G2/G3 arcs, units, distance modes).
- `cam.plan_trajectory(segments)` / `cam.gcode_to_trajectory(text)` — re-plan
  ("execute") the G-code into a time-sampled trajectory using an
  acceleration-limited **trapezoidal** velocity profile in **exact-stop (G61)**
  mode, so the tool passes through every waypoint.
- `cam.trajectory_metrics` — path-similarity metrics (discrete Fréchet, DTW,
  arc-length-resampled RMSE, waypoint round-trip error).
- `cam.sim_exec.carve_stock(positions)` — execute a trajectory in the simulator
  with a hard (step-count-invariant) CSG carve, for carved-stock validation.
- `cam.units` — unit conversions used at the I/O boundary (`inch_to_mm`,
  `mm_to_inch`, `ipm_to_mm_per_s`, `mm_per_min_to_ipm`, …); everything internal
  stays in millimetres.

## Testing

```bash
uv run python -m pytest
```

(`test_perf.py` and `test_mpi.py` at the repo root are standalone benchmark /
MPI scripts, not part of the pytest suite.)

## AMD GPU Support

Taichi support for AMD GPUs may be possible via ROCm: see
<https://rocm.docs.amd.com/projects/taichi/en/latest/install/taichi-install.html>
