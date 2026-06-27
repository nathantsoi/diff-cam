# diff-cam

A differentiable CNC simulator based on Taichi.

## Setup

- Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync
```

- Install TurboVNC server:

```bash
./scripts/setup.sh
```

## Documentation

See [docs/design.md](docs/design.md) for design details.

## Simulators

Currently, there are two simulators: voxel_simulator.py and csg_simulator.py

- voxel_simulator.py = Uses voxelization to initialize stock, target, and tool geometries. Applies cuts sequentially using a max operation. Rewards are meant to be differentiable, although the entire voxel space is not. 
- csg_simulator.py = Uses a constructive solid geometry approach to define geometries (fully differentiable). The compute_loss function mirrors the reward function in the RL approach.

## Environment and PPO
- cam_env.py = Defines a Gymnasium environment for the voxelization simulator with each individual action moving the tool in one direction (and applying cutting). 
- ppo.py is the CleanRL implementation of the PPO algorithm. It trains an agent using CamEnv-v0

## Differentiable Simulation
- UNDER CONSTRUCTION. Gradient descent will be employed 

## Training
To run locally, run the command:
```bash
uv run ./train_local.bash
```

To run on lonestar6, run
```cd $SCRATCH$```, ```cd diff-cam``` and ensure that your allocation and training hyperparameters are correct - these are currently hard-coded. 
```bash
sbatch train_hpc.bash
```


## Measures
We employ a few evaluation metrics.
- Dice Similarity Coefficient (DSC)
- Average Surface-to-Surface Distance (ASD)
- 95% Hausdorff Distance (HD95)


## Evaluation
- 10 randomly-generated episodes. Each model is tested against each episode
- Measures are calculated for each episode for each policy
- Summary statistics (mean, std) are computed
- Note that this is only a fair comparison if the hyperparameters are the same
Example:
``` ```

## CAM / G-code (`cam/`)

The `cam/` package bridges optimized trajectories and real CNC G-code, following
the LinuxCNC pipeline (interpreter → canonical moves → trajectory planner):

- `cam.trajectory_to_gcode(positions)` — export an `(T, 3)` unit-cube trajectory
  to RS274/NGC G-code (`G21`/`G90`/`G61`, `G0` to start, `G1` feeds, `M2`).
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

End-to-end round-trip demo (`trajectory → G-code → executed trajectory`):

```bash
uv run python scripts/roundtrip_demo.py
```

This reports trajectory similarity (≈ machine precision for the saved
trajectory) and carved-stock Dice (≈ 0.99).

## Testing

```bash
uv run pytest
```

## AMD GPU Support

Taichi support for AMD GPUs may be possible via the ROCm: see <https://rocm.docs.amd.com/projects/taichi/en/latest/install/taichi-install.html>


