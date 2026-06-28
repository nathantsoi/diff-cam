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
  csg_simulator.py    # continuous, differentiable CSG simulator (PSDFs)
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

```bash
# Headless (HPC / no display) — comparable to the csg_ppo baseline below:
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 --resolution 32 \
    --save_model --eval_freq 1 --record_video_freq 100 --video_fps 30

# Interactive live GUI (needs a display):
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 --resolution 32
```

### Continuous — PPO baseline (Method 2)

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
| `--eval_freq N` | `0` | run a greedy eval rollout + log Dice/ASD/HD95/reward every `N` iterations (`0` disables). No video is encoded unless the video cadence also lands on that iteration. |
| `--record_video_freq N` | `0` | additionally record + upload a greedy rollout **video** every `N` iterations (`0` disables) |
| `--video_fps F` | `30` | frame rate of the encoded mp4 |
| `--video_seed S` | `0` | seed for the rollout env, so the scenario is fixed across iterations and runs are comparable |
| `--track` / `--no-track` | `--track` | upload to wandb; with `--no-track` the mp4s are still written locally |

The recorder is built if **either** cadence is enabled. Metrics land in wandb
under `eval/reward`, `eval/dice`, `eval/asd`, `eval/hd95` (logged on every eval
*and* every video); videos appear under `media/policy_rollout`, with local copies
at `runs/<run>/videos/policy_step_<global_step>.mp4`.

After training finishes (including on early stopping), the **final** model's
geometry is also exported as STL meshes — the initial uncarved stock, the carved
stock, and the target part — to `runs/<run>/meshes/` (and uploaded to wandb when
`--track`). Meshes are the zero-level surface of each SDF (marching cubes) in the
unit-cube frame.

```bash
# Continuous PPO, no video (default):
uv run python -m algorithms.csg_ppo --total_timesteps 10000000 --num_envs 4 \
    --resolution 32 --max_steps 64 --save_model

# Continuous PPO, eval metrics every iteration, video only every 100 iterations:
uv run python -m algorithms.csg_ppo --total_timesteps 10000000 --num_envs 4 \
    --resolution 32 --max_steps 64 --save_model \
    --eval_freq 1 --record_video_freq 100 --video_fps 30
```

The discrete voxel PPO (`ppo.py`) renders its 3D GGUI scene off-screen for the
video:

```bash
# Discrete PPO, no video (default):
uv run python -m algorithms.ppo --total-timesteps 2000000 --num-envs 4 \
    --resolution 32 --max-steps 256

# Discrete PPO, video + metrics to wandb every 25 iterations:
uv run python -m algorithms.ppo --total-timesteps 2000000 --num-envs 4 \
    --resolution 32 --max-steps 256 --record_video_freq 100 --video_fps 30
```

**Gradient descent (`train_csg.py`).** Uses the **same** flags, encoder and
metric/STL code paths as the PPO trainers above — `--eval_freq`,
`--record_video_freq`, `--video_fps`, `--track` / `--no-track` — measured in Adam
iterations. Each video raymarches the optimized toolpath and is encoded to mp4
(ffmpeg) at `runs/<run>/videos/policy_step_<iter>.mp4`, uploaded to
`media/policy_rollout`. Geometry metrics land under `eval/dice`, `eval/asd`,
`eval/hd95` (the differentiable objective is logged as `losses/loss`; there is
**no `eval/reward`**, since this method has no RL reward), with `gouge`/`residual`
under `metrics/*`. The final STL meshes are exported just like the PPO trainers.
Pass `--headless` to skip the live GUI (auto-disabled when no display is present).

```bash
# Headless, eval every iteration + video every 100 iterations:
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 --resolution 32 \
    --save_model --eval_freq 1 --record_video_freq 25 --video_fps 30
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
uv run python -m eval.eval_csg --trajectory trajectory.npy --resolution 32
```

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

**Generate G-code for the Haas:**

```bash
# Using the default root trajectory
uv run python scripts/export_gcode.py --post haas --tool 3 --rpm 6000 \
    --workspace-mm 100 --program-number 1234 -o part.nc

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
uv run python -m eval.eval_csg --trajectory trajectory.npy --gcode --post rs274

# Using a trajectory from a specific run folder
uv run python -m eval.eval_csg \
    --trajectory runs/CamEnvDiff-v0__train_csg__1__1782599757/trajectory.npy \
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

## Testing

```bash
uv run python -m pytest
```

(`test_perf.py` and `test_mpi.py` at the repo root are standalone benchmark /
MPI scripts, not part of the pytest suite.)

## AMD GPU Support

Taichi support for AMD GPUs may be possible via ROCm: see
<https://rocm.docs.amd.com/projects/taichi/en/latest/install/taichi-install.html>
