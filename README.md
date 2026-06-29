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
  csg_simulator.py    # continuous, differentiable CSG simulator (PSDFs); voxelizes only the STOCK box (placed at a G54 origin inside the Mini Mill work volume), cubic voxels, feed/rapid speed clipping
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
  run_pipeline.py     # one-step train -> eval -> G-code -> visualize
  export_gcode.py     # trajectory -> machine G-code (Haas / RS274)
  visualize_trajectory.py  # 6-panel trajectory + G-code/sim diagnostic figure
  roundtrip_demo.py   # trajectory -> G-code -> executed trajectory fidelity demo
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

The normalized cube is the **stock box** (default a **1 in cube**), so
`--voxel_size_mm 0.5` gives **sub-mm cubic voxels** on a tiny grid (51³, ~0.14 GB),
and `--dt 0.12` makes each feed step ≈ 1 voxel (see
[Stock box, work volume & precision](#stock-box-work-volume--precision)).

```bash
# Headless (HPC / no display) — comparable to the csg_ppo baseline below:
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 \
    --stock_size_in 1 1 1 --voxel_size_mm 0.5 --target_radius_mm 11.43 --dt 0.12 \
    --save_model --eval_freq 1 --record_video_freq 100 --video_fps 30 --progress_bar

# Interactive live GUI (needs a display):
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 \
    --stock_size_in 1 1 1 --voxel_size_mm 0.5 --dt 0.12
```

#### Stock box, work volume & precision

There are **two** boxes, and keeping them separate is what makes sub-mm precision
cheap:

- **Stock box** — the raw block you actually machine. The normalized geometry box
  `[0,1]³` *is* the stock box (so trajectories stay scale-free), and **only the
  stock is voxelized**. Set it with `--stock_size_in X Y Z` (inches); the default
  is a **1 in cube**. RAM scales with the *part*, not the machine.
- **Machine work volume** — the toolhead's reachable envelope (toolhead limits),
  default the **Haas Mini Mill** **16 × 12 × 10 in** = 406.4 × 304.8 × 254.0 mm.
  Set it with `--workspace_in X Y Z` (inches). It is *not* voxelized; it is used
  for G-code export, the holder-collision barrier, and a reachability check.

The stock is placed in the machine at a **work origin (G54 offset)** — its
**top-centre** — set with `--stock_origin_in X Y Z` (inches). Exported G-code is
relative to that origin (`Z = 0` at the stock top, plunges go negative).

The voxel grid uses **per-axis** dimensions `(Nx, Ny, Nz)` chosen so every voxel
is a physical **cube** of side `v` mm — keeping the cutter and cuts undistorted.
Set the precision directly with **`--voxel_size_mm`** (the sub-mm knob); if you
omit it, `--resolution` is used as a fallback (voxel count along the stock's
**longest** axis). Tool/target sizes are in millimetres (`--tool_radius_mm`,
`--tool_height_mm`, `--target_radius_mm`).

**Memory.** The `stock[max_steps+1, Nx,Ny,Nz]` field dominates memory, and under
autodiff it is allocated **twice** (value + gradient), so peak VRAM is
≈ `2 × (max_steps+1) × Nx·Ny·Nz × 4 B`. Because only the small stock is
voxelized, sub-mm is trivially affordable. Below: peak for the default **1 in
cube** stock at `--max_steps 64` (65 slices, f32):

| `--voxel_size_mm` | grid `Nx×Ny×Nz` (1″ cube) | peak VRAM (value+grad) | 1/4″ tool Ø |
|-------------------|----------------------------|------------------------|-------------|
| 1.0 mm   | 25×25×25    | 16 MB    | 6.4 vox |
| **0.5 mm** (default) | **51×51×51** | **0.14 GB** | **12.7 vox** |
| 0.25 mm  | 102×102×102 | 1.1 GB   | 25 vox |
| 0.1 mm   | 254×254×254 | 17 GB    | 64 vox |

→ **0.5 mm voxels** is a strong default — sub-mm fidelity on a 1″ cube at ~0.14 GB.
Drop to **0.25 mm** (~1.1 GB) for fine detail, or **0.1 mm** (~17 GB) for very
fine work; peak scales linearly with `--max_steps`. A *larger* stock at the same
`--voxel_size_mm` costs more (VRAM ∝ stock volume): e.g. a **2 in cube** at
0.5 mm is 102³ ≈ 1.1 GB. Pick `--voxel_size_mm` for the precision you need and
size the grid from the stock — don't voxelize the whole machine.

#### Units & speed limits

The differentiable simulator enforces two machine-style **max speeds** by
clipping each step, mirroring a real controller's feed/rapid override (cf. the
LinuxCNC trajectory planner and CAMotics). A per-step displacement `delta`
(normalized) spans `delta · S` mm on the per-axis **stock box** `S`, over `dt`
seconds, so its speed is `|delta · S| / dt` (mm/s). All scale-related math is in
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
| `--stock_size_in X Y Z` | `1 1 1` | stock box (in) — the normalized cube, the only thing voxelized |
| `--voxel_size_mm F` | `0.5` | physical voxel edge (mm) — the sub-mm precision knob |
| `--stock_origin_in X Y Z` | none | work origin (G54) = stock top-centre in machine inches |
| `--workspace_in X Y Z` | `16 12 10` | machine work volume (in), default Mini Mill (toolhead limits) |
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
> below a 0.5 mm voxel — so each feed step is sub-voxel *in time*. To advance ~1
> voxel per feed step set `--dt ≈ voxel_size_mm / feed_mm_per_s` (e.g. ≈ `0.12`
> for 0.5 mm voxels at the default 10 ipm feed), or pass
> `--no-enforce_speed_limits` to disable the constraint.

### Continuous — PPO baseline (Method 2)

> **Precision note.** The PPO observation embeds the **full `stock` and `target`
> voxel grids**, so the `num_steps × num_envs` rollout buffer — not the carve — is
> the bottleneck. Buffer ≈ `num_steps × num_envs × 2·Nx·Ny·Nz × 4 B`. On the
> default 1″ cube stock at `--voxel_size_mm 0.8` (≈ 32³) it is only ~0.5 GB at
> `--num_envs 4, --num_steps 512`; at `--voxel_size_mm 0.5` (51³, the env default)
> ~2.2 GB; at `--voxel_size_mm 0.25` (102³) ~17 GB. The examples use **0.8 mm**
> (~32³) for fast iteration — drop it for finer parts and watch VRAM. The
> gradient-descent method (`train_csg`) keeps no rollout buffer, so it carves at
> 0.25–0.5 mm cheaply; prefer it when you need sub-mm fidelity.

```bash
uv run python -m algorithms.csg_ppo \
    --total_timesteps 10000000 --num_envs 1 --stock_size_in 1 1 1 --voxel_size_mm 0.8 \
    --max_steps 64 --num_steps 512 --num_minibatches 8 --update_epochs 4 \
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
    --total_timesteps 10000000 --num_envs 4 --stock_size_in 1 1 1 --voxel_size_mm 0.8 \
    --max_steps 64 --num_steps 512 --num_minibatches 8 --update_epochs 4 --save_model \
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
    --stock_size_in 1 1 1 --voxel_size_mm 0.8 --max_steps 64 --save_model

# Continuous PPO for autoresearch / LLM harnesses (clean scrolling logs, no W&B sync):
uv run python -m algorithms.csg_ppo --total_timesteps 10000000 --num_envs 4 \
    --stock_size_in 1 1 1 --voxel_size_mm 0.8 --max_steps 64 --eval --no-track
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
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 \
    --stock_size_in 1 1 1 --voxel_size_mm 0.5 --dt 0.12 \
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

### Examples — stock sizes & part shapes

The normalized cube is the **stock box**; choose its size with `--stock_size_in`
(inches, X Y Z) and the precision with `--voxel_size_mm`. The target part is set
by `--target_shape` (`sphere`, `cylinder`, `box`, `pyramid`) with
`--target_radius_mm` (sphere/cylinder radius, or box/pyramid half-size) and
`--target_height_mm` (cylinder/pyramid height). Tool/target sizes are millimetres
(1 in = 25.4 mm), so size the part to fit *inside* the stock. Tune `--dt` to
≈ `voxel_size_mm / feed_mm_per_s` so feed steps advance ≈ 1 voxel.

```bash
# 1" cube stock, 0.9" diameter SPHERE, 0.5 mm voxels (51³, ~0.14 GB)
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 \
    --stock_size_in 1 1 1 --voxel_size_mm 0.5 --dt 0.12 \
    --target_shape sphere --target_radius_mm 11.43 \
    --headless --save_model --eval --no-track

# 1" cube stock, 0.9" diameter x 0.9" tall CYLINDER, 0.5 mm voxels
uv run python -m algorithms.train_csg --iters 128 --max_steps 64 \
    --stock_size_in 1 1 1 --voxel_size_mm 0.5 --dt 0.12 \
    --target_shape cylinder --target_radius_mm 11.43 --target_height_mm 22.86 \
    --headless --save_model --eval --no-track

# Larger 2" cube stock at the SAME 0.5 mm voxels (102³, ~1.1 GB) — a 1.6" sphere
uv run python -m algorithms.train_csg --iters 128 --max_steps 96 \
    --stock_size_in 2 2 2 --voxel_size_mm 0.5 --dt 0.12 \
    --target_shape sphere --target_radius_mm 20.32 \
    --headless --save_model --eval --no-track

# Non-cubic 2 x 1 x 1" bar, fine 0.25 mm voxels (203×102×102, ~2.2 GB) — a box part
uv run python -m algorithms.train_csg --iters 128 --max_steps 96 \
    --stock_size_in 2 1 1 --voxel_size_mm 0.25 --dt 0.06 \
    --target_shape box --target_radius_mm 9.0 \
    --headless --save_model --eval --no-track
```

Each run writes `trajectory.npy` (with `--save_model`) and copies it to the repo
root. Score and export it with the **matching** `--stock-size-in` /
`--voxel-size-mm` / `--target-shape` (see [Evaluation](#evaluation) and
[G-code based evaluation](#g-code-based-evaluation-and-the-haas-mini-mill)):

```bash
# Score the cylinder run
uv run python -m eval.eval_csg --trajectory trajectory.npy \
    --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape cylinder

# Export the 2" cube run to a Haas program, fixturing the stock top-centre at machine (8,6,5)"
uv run python scripts/export_gcode.py --post haas \
    --stock-size-in 2 2 2 --stock-origin-in 8 6 5 -o part.nc
```

### One-step pipeline (train → eval → G-code → visualize)

`scripts/run_pipeline.py` chains all four stages into a single command — from
geometry to a machine-ready G-code program plus a diagnostic figure, no manual
script-chaining required. It runs each stage as a subprocess (a fresh process
per stage so Taichi re-initialises cleanly), threads the discovered run
directory between them, and forwards the geometry flags consistently. The
exporter and visualizer additionally auto-read the run's `args.json`.

```bash
# Fast end-to-end smoke test (small part, headless, no W&B):
uv run python scripts/run_pipeline.py

# A real 1" sphere at 0.5 mm voxels, Haas G-code + figure:
uv run python scripts/run_pipeline.py --iters 128 --max-steps 64 \
    --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere \
    --target-radius-mm 11.43 --post haas

# Fixture the stock top-centre at machine (8,6,5)" (emits G10 L2 in the Haas program):
uv run python scripts/run_pipeline.py --stock-origin-in 8 6 5 --post haas
```

Outputs land in the training `runs/<run>/` dir: `trajectory.npy`,
`metrics.json`, `gcode_<post>.nc`, and `trajectory_viz_<post>.png` (see
[Visualizing a trained trajectory](#visualizing-a-trained-trajectory)). Run a
subset with `--stages` (e.g. `--stages viz`) and `--run-dir` to re-run one stage
on an existing run. The console ends with a pipeline summary listing every
artifact path.

## Evaluation

All modes report the same geometric metrics of the final carved stock vs. the
target: **Dice** (DSC), **Average Surface Distance** (ASD), and **95% Hausdorff
Distance** (HD95).

### Continuous (`eval/eval_csg.py`)

Evaluate a gradient-descent trajectory (defaults to the 1″ cube stock; match the
`--stock_size_in` / `--voxel_size_mm` you trained with):

```bash
uv run python -m eval.eval_csg --trajectory trajectory.npy \
    --stock-size-in 1 1 1 --voxel-size-mm 0.5
```

(`--voxel-size-mm 0.5` ≈ sub-mm carving on the 1″ cube, matching training;
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

**Work coordinate system (G54).** Trajectories are normalized over the **stock
box**, and the work origin (G54) is the stock's **top-centre**: exported X/Y are
relative to the stock's XY centre and Z is relative to the stock's top face
(`Z = 0` at the top, plunges negative). Set the stock with `--stock-size-in X Y Z`
(inches). If you give `--stock-origin-in X Y Z` (the stock top-centre in machine
coords), the Haas post also emits `G10 L2 P1 X.. Y.. Z..` so the program
self-programs its G54 offset; otherwise it just declares `G54` and assumes the
operator has set it.

Internally the CAM layer (and the simulator) work in **millimetres**; pass
`--units inch` to emit an inch program (`G20`) instead of mm (`G21`). Coordinates
and feeds are converted mm→inch just before emission via `cam.units`, so the
program round-trips exactly through the inch-aware parser.

**Generate G-code for the Haas:**

**The exporter auto-matches the training run.** `train_csg.py` writes the full run
config to `runs/<run>/args.json` next to `trajectory.npy`; when you point
`--trajectory` at a run's trajectory, the exporter reads that file and uses the
run's `stock_size_in`, `stock_origin_in`, and `workspace_in` so the program lines
up with the part it was optimized for (it prints the resolved config and its
source). Any flag you pass explicitly overrides the run config; `--no-run-config`
ignores it. For the repo-root `trajectory.npy` copy (no adjacent `args.json`) it
falls back to CLI flags / defaults (1 in cube stock, Mini Mill work volume).

```bash
# Auto-match the run's stock/part/location from runs/<run>/args.json
uv run python scripts/export_gcode.py --post haas --tool 3 --rpm 6000 \
    --trajectory runs/CamEnvDiff-v0__train_csg__1__1782599757/trajectory.npy \
    -o runs/CamEnvDiff-v0__train_csg__1__1782599757/gcode_haas.nc

# Override the stock placement explicitly (top-centre at machine (8, 6, 5) in;
# emits G10 L2 P1 to program the G54 offset)
uv run python scripts/export_gcode.py --post haas \
    --stock-size-in 1.5 1.5 1.0 --stock-origin-in 8 6 5 \
    --trajectory runs/CamEnvDiff-v0__train_csg__1__1782599757/trajectory.npy -o part.nc
```

**Evaluate based on the G-code program** (round-trip the trajectory through the
CAM layer and score both path fidelity and the carved stock of the *executed*
program):

```bash
# Using the default root trajectory
uv run python -m eval.eval_csg --trajectory trajectory.npy \
    --stock-size-in 1 1 1 --voxel-size-mm 0.5 --gcode --post rs274

# Using a trajectory from a specific run folder
uv run python -m eval.eval_csg \
    --trajectory runs/CamEnvDiff-v0__train_csg__1__1782599757/trajectory.npy \
    --stock-size-in 1 1 1 --voxel-size-mm 0.5 \
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

## Visualizing a trained trajectory

`scripts/visualize_trajectory.py` renders a **6-panel diagnostic figure** that
makes the trajectory, its G-code, and the simulation carve legible at a glance.
It auto-matches the training run's `args.json` (stock/part/location), so pointing
it at a run dir is usually enough:

  - **A. Normalized frame** — the path in the simulator's `[0,1]³` stock box, with
    the speed-clipped path, the pre-clip commanded deltas, and start/end markers.
  - **B. WCS / G-code frame** — the same path mapped to the machine's work
    coordinate system (top-centre G54, `Z=0` at the stock top), with the stock
    box, G54 origin, and safe-Z plane.
  - **C. G-code round-trip** — the original path vs the executed G-code path, with
    rapid approach/retract moves drawn faded so cutting moves are separable.
  - **D. Sim carve vs target** — the hard-CSG result of the trajectory next to the
    target part ("did it machine the part?").
  - **E. G-code vs sim carve** — the carve of the exported G-code overlaid on the
    carve of the original trajectory; the console reports their voxel **Dice**
    (~1.0 proves the G-code reproduces the simulation).
  - **F. Metrics** — a text summary of the round-trip, Z ranges, cut depth, and
    the G-code-vs-sim and carved-vs-target Dice.

The export→parse round trip is geometrically exact, so a perceived G-code/sim
mismatch is **not** the export math — panel E's Dice is the definitive check (it
has been `1.00000` in every run tested). Real causes of a visual mismatch are the
Haas approach/retract moves (panel C) or an under-trained trajectory (panel F's
cut-depth flag).

```bash
# Auto-match the repo-root trajectory.npy + args.json
uv run python scripts/visualize_trajectory.py

# Visualize a specific run (reads runs/<run>/trajectory.npy + args.json)
uv run python scripts/visualize_trajectory.py \
    --run runs/CamEnvDiff-v0__train_csg__1__1782599757

# Haas post + skip the carved-stock panels (no Taichi, faster/headless)
uv run python scripts/visualize_trajectory.py --post haas --no-carve
```

The figure is written next to the trajectory as
`trajectory_viz_<post>.png` (or `--save`); add `--show` for the interactive
matplotlib window.

### CAM API

- `cam.trajectory_to_gcode(positions, config, post="rs274")` — export an
  `(T, 3)` stock-normalized trajectory to G-code (work coordinate system,
  top-centre G54) with the chosen post. `config` must carry a `stock_size_in`.
- `cam.parse_gcode(text, config)` — parse G-code back to motion segments (G0/G1
  plus G2/G3 arcs, units, distance modes), inverting the stock/G54 mapping.
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
