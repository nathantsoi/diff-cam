"""Evaluation for the continuous (CSG / GradMill) mode.

Two artifacts can be scored, both with the same internal geometric metrics
(Dice / ASD / HD95 of the final carved stock vs. the target):

* ``--trajectory traj.npy`` -- a saved ``(T, 3)`` tool path. This is what
  ``algorithms/train_csg.py`` (the analytic gradient-descent method) writes. The
  path is carved into stock with a hard, step-count-invariant CSG subtraction and
  compared to the baked target.

* ``--checkpoints a.cleanrl_model b...`` -- one or more continuous-PPO
  checkpoints from ``algorithms/csg_ppo.py``. Each policy is rolled out
  deterministically in ``CamEnv-v0`` and the resulting stock is scored. Several
  checkpoints are evaluated on the same per-seed scenarios so the numbers are
  paired/comparable.

With ``--gcode`` a trajectory is additionally round-tripped through the CAM
layer (``trajectory -> G-code -> executed trajectory``) using the chosen
post-processor, reporting both path-fidelity metrics and the carved-stock Dice
of the *executed* program -- i.e. eval based on the actual G-code that would run
on the machine.

Examples
--------
    uv run python -m eval.eval_csg --trajectory trajectory.npy
    uv run python -m eval.eval_csg --trajectory trajectory.npy --gcode --post haas
    uv run python -m eval.eval_csg --checkpoints runs/*/csg_ppo.cleanrl_model --num-runs 5
"""

import argparse

import numpy as np
import torch

from simulator.csg_simulator import CSGSimulatorDelta
from simulator.csg_metrics import (
    sdf_to_mask,
    dice_score,
    average_surface_distance,
    hd95,
)

# Default target/tool geometry -- must match CamEnv.reset() and train_csg.py.
TARGET_SHAPE = "sphere"
TARGET_RADIUS = 0.4
TARGET_CENTER = [0.5, 0.5, 0.5]
TOOL_RADIUS = 0.05
TOOL_HEIGHT = 0.15


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _metrics(stock_grid, target_grid, dx):
    """Dice / ASD / HD95 between a carved stock grid and the target grid."""
    pred = sdf_to_mask(stock_grid)
    targ = sdf_to_mask(target_grid)
    out = {"dice": float(dice_score(pred, targ))}
    try:
        out["asd"] = float(average_surface_distance(pred, targ)) * dx
    except ValueError:
        out["asd"] = float("inf")
    try:
        out["hd95"] = float(hd95(pred, targ)) * dx
    except ValueError:
        out["hd95"] = float("inf")
    return out


def carve_trajectory_metrics(positions, resolution=32, target_shape=TARGET_SHAPE):
    """Hard-carve a trajectory and score the result against the baked target.

    Returns (metrics_dict, stock_grid, target_grid).
    """
    from cam.sim_exec import _HardCarveSimulator

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError(f"positions must be (T>=2, 3); got {positions.shape}")

    deltas = np.diff(positions, axis=0)
    sim = _HardCarveSimulator(
        resolution=resolution,
        max_steps=len(positions) - 1,
        target_shape=target_shape,
        tool_start=tuple(float(v) for v in positions[0]),
    )
    sim.tool_radius[None] = TOOL_RADIUS
    sim.tool_height[None] = TOOL_HEIGHT
    sim.target_params["radius"][None] = TARGET_RADIUS
    sim.target_params["center"][None] = TARGET_CENTER
    sim.bake_target_grid()
    sim.set_target_volume()

    padded = np.zeros((sim.max_steps, 3), dtype=np.float32)
    padded[: len(deltas)] = deltas
    sim.tool_delta.from_numpy(padded)
    sim.forward_hard(len(positions))

    stock = sim.stock.to_numpy()[len(positions) - 1]
    target = sim.target.to_numpy()
    return _metrics(stock, target, 1.0 / resolution), stock, target


# ---------------------------------------------------------------------------
# Trajectory evaluation (gradient descent output, or any saved path)
# ---------------------------------------------------------------------------
def eval_trajectory(path, resolution, do_gcode, post, workspace_mm):
    positions = np.load(path).astype(np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{path} must hold an (T, 3) array; got {positions.shape}")
    print(f"Loaded trajectory: {positions.shape} from {path}")

    m, _, _ = carve_trajectory_metrics(positions, resolution=resolution)
    print("\n=== Internal metrics (carved stock vs target) ===")
    print(f"  Dice : {m['dice']:.4f}")
    print(f"  ASD  : {m['asd']:.4f}")
    print(f"  HD95 : {m['hd95']:.4f}")

    if not do_gcode:
        return

    from cam import (
        MachineConfig, trajectory_to_gcode, parse_gcode, segment_waypoints,
        gcode_to_trajectory, discrete_frechet, dtw_distance, resampled_rmse,
        waypoint_roundtrip_error,
    )

    cfg = MachineConfig(workspace_mm=workspace_mm)
    scale = cfg.workspace_mm
    gcode = trajectory_to_gcode(positions, cfg, post=post)
    segments = parse_gcode(gcode, cfg)
    recovered = segment_waypoints(segments)
    executed, times = gcode_to_trajectory(gcode, cfg)

    print(f"\n=== G-code round-trip (post='{post}') ===")
    print(f"  program blocks            : {len([l for l in gcode.splitlines() if l and not l.startswith('(')])}")
    print(f"  executed samples          : {executed.shape[0]} ({times[-1]:.2f}s of motion)")
    print(f"  waypoint round-trip error : {waypoint_roundtrip_error(positions, recovered, scale):.3e} mm")
    print(f"  discrete Frechet          : {discrete_frechet(positions, executed, scale):.3e} mm")
    print(f"  DTW (mean matched dist)   : {dtw_distance(positions, executed, scale):.3e} mm")
    print(f"  arc-length resampled RMSE : {resampled_rmse(positions, executed, scale=scale):.3e} mm")

    me, _, _ = carve_trajectory_metrics(executed, resolution=resolution)
    print("  --- carved stock of executed program vs target ---")
    print(f"  Dice : {me['dice']:.4f}   ASD : {me['asd']:.4f}   HD95 : {me['hd95']:.4f}")


# ---------------------------------------------------------------------------
# Checkpoint (continuous PPO) evaluation
# ---------------------------------------------------------------------------
def _load_ppo_agent(checkpoint_path):
    """Load a continuous-PPO checkpoint and rebuild its Agent + env params."""
    import gymnasium as gym
    from types import SimpleNamespace
    from algorithms.csg_ppo import Agent
    import cam_env  # noqa: F401  registers CamEnv-v0

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "args" in ckpt and "agent" in ckpt):
        raise ValueError(
            f"{checkpoint_path}: expected a continuous-PPO checkpoint "
            f"{{'agent', 'args'}} produced by csg_ppo.py --save_model."
        )
    a = ckpt["args"]
    resolution = int(a["resolution"])
    max_steps = int(a["max_steps"])
    target_shape = a.get("target_shape", TARGET_SHAPE)

    env = gym.make("CamEnv-v0", resolution=resolution, max_steps=max_steps,
                   target_shape=target_shape)
    shim = SimpleNamespace(single_action_space=env.action_space)
    agent = Agent(shim, resolution=resolution)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent, env, resolution, max_steps


def _rollout(agent, env, seed):
    """Deterministic rollout; returns (metrics, reward, positions)."""
    obs, _ = env.reset(seed=seed)
    sim = env.unwrapped.simulator
    resolution = env.unwrapped.resolution
    total_reward, done = 0.0, False
    while not done:
        with torch.no_grad():
            feats = agent.features(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = agent.actor_mean(feats).squeeze(0).numpy()  # deterministic
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    t = env.unwrapped.current_step
    stock = sim.stock.to_numpy()[t]
    target = sim.target.to_numpy()
    return _metrics(stock, target, 1.0 / resolution), float(total_reward)


def eval_checkpoints(paths, num_runs, base_seed):
    results = {}
    for path in paths:
        print(f"\nEvaluating {path}")
        agent, env, res, ms = _load_ppo_agent(path)
        print(f"  -> resolution={res}, max_steps={ms}")
        per_run = []
        try:
            for i in range(num_runs):
                seed = base_seed + i
                m, r = _rollout(agent, env, seed)
                per_run.append({"seed": seed, "reward": r, **m})
                print(f"  run {i} (seed {seed}): reward={r:+.3f} "
                      f"dice={m['dice']:.3f} asd={m['asd']:.4f} hd95={m['hd95']:.4f}")
        finally:
            env.close()
        results[path] = per_run

    print("\n=== Summary (mean +/- std over runs) ===")
    for path, runs in results.items():
        print(f"\n{path}")
        for key in ("reward", "dice", "asd", "hd95"):
            vals = np.array([r[key] for r in runs], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if len(finite):
                print(f"  {key:5s}: {finite.mean():+.4f} +/- {finite.std():.4f}")
            else:
                print(f"  {key:5s}: n/a")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trajectory", type=str,
                     help="path to a saved (T,3) trajectory .npy (e.g. from train_csg.py)")
    src.add_argument("--checkpoints", nargs="+",
                     help="one or more continuous-PPO checkpoints from csg_ppo.py")
    ap.add_argument("--resolution", type=int, default=32,
                    help="grid resolution for trajectory carving / scoring")
    ap.add_argument("--num-runs", type=int, default=10,
                    help="episodes per checkpoint (checkpoint mode)")
    ap.add_argument("--seed", type=int, default=42, help="base seed")
    ap.add_argument("--gcode", action="store_true",
                    help="also round-trip the trajectory through the CAM/G-code layer")
    ap.add_argument("--post", type=str, default="rs274",
                    help="post-processor for --gcode (rs274 | haas)")
    ap.add_argument("--workspace-mm", type=float, default=100.0,
                    help="physical edge length of the unit cube for G-code")
    args = ap.parse_args()

    if args.trajectory:
        eval_trajectory(args.trajectory, args.resolution, args.gcode,
                        args.post, args.workspace_mm)
    else:
        eval_checkpoints(args.checkpoints, args.num_runs, args.seed)


if __name__ == "__main__":
    main()
