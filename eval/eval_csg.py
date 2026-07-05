"""Evaluation for the continuous (CSG / GradMill) mode.

Two artifacts can be scored, both with the same internal geometric metrics
(Dice / ASD / HD95 of the final carved stock vs. the target):

* ``--trajectory traj.npy`` -- a saved ``(T, 3)`` tool path. This is what
  ``algorithms/train_csg.py`` (the analytic gradient-descent method) writes. The
  path is carved into stock with a hard, step-count-invariant CSG subtraction and
  compared to the baked target.

* ``--checkpoints a.cleanrl_model b...`` -- one or more continuous-PPO
  checkpoints from ``algorithms/csg_ppo.py``. Each policy is rolled out
  deterministically in ``CamEnvDiff-v0`` and the resulting stock is scored. Several
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

# Default target/tool/stock geometry -- must match CamEnvDiff.reset() and
# train_csg.py. Sizes are in MILLIMETRES. The normalized cube [0,1]^3 is the
# STOCK box (only it is voxelized); the machine work volume is separate metadata.
# Center is a normalized [0,1] position inside the stock.
STOCK_SIZE_IN = (1.0, 1.0, 1.0)   # stock box, inches
VOXEL_SIZE_MM = 0.5               # sub-mm precision knob
WORK_VOLUME_IN = (16.0, 12.0, 10.0)  # Haas Mini Mill toolhead limits
TARGET_SHAPE = "sphere"
TARGET_RADIUS = 11.43   # mm (0.9 in diameter sphere/cylinder inside the stock)
TARGET_HEIGHT = 22.86   # mm (0.9 in tall cylinder/pyramid; ignored for sphere/box)
TARGET_CENTER = [0.5, 0.5, 0.5]
TOOL_RADIUS = 3.175     # mm (1/4" end mill)
TOOL_HEIGHT = 25.0      # mm


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


def carve_trajectory_metrics(positions, resolution=32, target_shape=TARGET_SHAPE,
                             voxel_size_mm=VOXEL_SIZE_MM):
    """Hard-carve a trajectory and score the result against the baked target.

    The normalized cube is the STOCK box (``STOCK_SIZE_IN``); ``voxel_size_mm``
    sets the sub-mm carving precision (``resolution`` is only a fallback when it
    is None). Returns (metrics_dict, stock_grid, target_grid).
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
        stock_size_in=STOCK_SIZE_IN,
        voxel_size_mm=voxel_size_mm,
        work_volume_in=WORK_VOLUME_IN,
    )
    sim.tool_radius[None] = TOOL_RADIUS
    sim.tool_height[None] = TOOL_HEIGHT
    sim.set_target_params(radius_mm=TARGET_RADIUS, height_mm=TARGET_HEIGHT,
                          half_size_mm=TARGET_RADIUS, center=TARGET_CENTER)
    sim.bake_target_grid()
    sim.set_target_volume()

    padded = np.zeros((sim.max_steps, 3), dtype=np.float32)
    padded[: len(deltas)] = deltas
    sim.tool_delta.from_numpy(padded)
    sim.forward_hard(len(positions))

    stock = sim.stock.to_numpy()[len(positions) - 1]
    target = sim.target.to_numpy()
    # Surface distances reported in mm: one voxel == sim.v mm (cubic voxels).
    return _metrics(stock, target, sim.v), stock, target


# ---------------------------------------------------------------------------
# Trajectory evaluation (gradient descent output, or any saved path)
# ---------------------------------------------------------------------------
def eval_trajectory(path, resolution, do_gcode, post, config, voxel_size_mm=VOXEL_SIZE_MM,
                    target_shape=TARGET_SHAPE):
    positions = np.load(path).astype(np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{path} must hold an (T, 3) array; got {positions.shape}")
    print(f"Loaded trajectory: {positions.shape} from {path}")

    import os
    import time

    import wandb
    run = wandb.init(
        project="diffcam",
        entity="diffcam",
        job_type="evaluation",
        name=f"eval_trajectory_{os.path.basename(path)}_{int(time.time() * 1000)}",
        config={
            "trajectory_path": path,
            "resolution": resolution,
            "voxel_size_mm": voxel_size_mm,
            "stock_size_in": list(STOCK_SIZE_IN),
            "do_gcode": do_gcode,
            "post": post,
            "workspace_vec_mm": config.workspace_vec.tolist(),
        }
    )

    try:
        m, _, _ = carve_trajectory_metrics(positions, resolution=resolution,
                                           voxel_size_mm=voxel_size_mm,
                                           target_shape=target_shape)
        print("\n=== Internal metrics (carved stock vs target) ===")
        print(f"  Dice : {m['dice']:.4f}")
        print(f"  ASD  : {m['asd']:.4f}")
        print(f"  HD95 : {m['hd95']:.4f}")

        wandb.log({
            "trajectory/dice": m["dice"],
            "trajectory/asd": m["asd"],
            "trajectory/hd95": m["hd95"],
        })

        if not do_gcode:
            return

        from cam import (
            trajectory_to_gcode, parse_gcode, segment_waypoints,
            gcode_to_trajectory, discrete_frechet, dtw_distance, resampled_rmse,
            waypoint_roundtrip_error,
        )

        cfg = config
        ws = cfg.stock_size_vec   # (3,) mm per-axis stock box (the normalized cube)
        gcode = trajectory_to_gcode(positions, cfg, post=post)
        segments = parse_gcode(gcode, cfg)
        recovered = segment_waypoints(segments)
        executed, times = gcode_to_trajectory(gcode, cfg)

        # Pre-scale normalized coords to physical mm (per-axis) so the
        # path-similarity metrics report true millimetres on the anisotropic
        # envelope; pass scale=1.0 since the arrays are already in mm.
        pos_mm = positions * ws
        rec_mm = recovered * ws
        exec_mm = executed * ws

        print(f"\n=== G-code round-trip (post='{post}') ===")
        print(f"  program blocks            : {len([l for l in gcode.splitlines() if l and not l.startswith('(')])}")
        print(f"  executed samples          : {executed.shape[0]} ({times[-1]:.2f}s of motion)")
        # The Haas post adds approach/plunge/retract moves, so the recovered
        # waypoint count differs from the source path; the strict waypoint
        # metric needs matching counts, so report n/a there (the path-similarity
        # metrics below use the executed trajectory and remain valid).
        if pos_mm.shape == rec_mm.shape:
            wp_err = waypoint_roundtrip_error(pos_mm, rec_mm, 1.0)
            print(f"  waypoint round-trip error : {wp_err:.3e} mm")
        else:
            print(f"  waypoint round-trip error : n/a ({rec_mm.shape[0]} waypoints "
                  f"vs {pos_mm.shape[0]} source points; post adds approach moves)")
        print(f"  discrete Frechet          : {discrete_frechet(pos_mm, exec_mm, 1.0):.3e} mm")
        print(f"  DTW (mean matched dist)   : {dtw_distance(pos_mm, exec_mm, 1.0):.3e} mm")
        print(f"  arc-length resampled RMSE : {resampled_rmse(pos_mm, exec_mm, scale=1.0):.3e} mm")

        me, _, _ = carve_trajectory_metrics(executed, resolution=resolution,
                                            voxel_size_mm=voxel_size_mm,
                                            target_shape=target_shape)
        print("  --- carved stock of executed program vs target ---")
        print(f"  Dice : {me['dice']:.4f}   ASD : {me['asd']:.4f}   HD95 : {me['hd95']:.4f}")

        wp_err = (waypoint_roundtrip_error(pos_mm, rec_mm, 1.0)
                  if pos_mm.shape == rec_mm.shape else float("nan"))
        wandb.log({
            "gcode/blocks": len([l for l in gcode.splitlines() if l and not l.startswith('(')]),
            "gcode/executed_samples": executed.shape[0],
            "gcode/motion_time": times[-1],
            "gcode/waypoint_roundtrip_error": wp_err,
            "gcode/discrete_frechet": discrete_frechet(pos_mm, exec_mm, 1.0),
            "gcode/dtw": dtw_distance(pos_mm, exec_mm, 1.0),
            "gcode/rmse": resampled_rmse(pos_mm, exec_mm, scale=1.0),
            "gcode/executed_dice": me["dice"],
            "gcode/executed_asd": me["asd"],
            "gcode/executed_hd95": me["hd95"],
        })
    finally:
        run.finish()


# ---------------------------------------------------------------------------
# Checkpoint (continuous PPO) evaluation
# ---------------------------------------------------------------------------
def _load_ppo_agent(checkpoint_path):
    """Load a continuous-PPO checkpoint and rebuild its Agent + env params."""
    import gymnasium as gym
    from types import SimpleNamespace
    from algorithms.csg_ppo import Agent
    import cam_env  # noqa: F401  registers CamEnvDiff-v0

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

    env = gym.make("CamEnvDiff-v0", resolution=resolution, max_steps=max_steps,
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
    import os
    import time

    import wandb

    for path in paths:
        print(f"\nEvaluating {path}")
        agent, env, res, ms = _load_ppo_agent(path)
        print(f"  -> resolution={res}, max_steps={ms}")

        run = wandb.init(
            project="diffcam",
            entity="diffcam",
            job_type="evaluation",
            name=f"eval_checkpoint_{os.path.basename(path)}_{int(time.time() * 1000)}",
            config={
                "checkpoint_path": path,
                "resolution": res,
                "max_steps": ms,
                "num_runs": num_runs,
                "base_seed": base_seed,
            }
        )

        per_run = []
        table = wandb.Table(columns=["run_index", "seed", "reward", "dice", "asd", "hd95"])
        try:
            for i in range(num_runs):
                seed = base_seed + i
                m, r = _rollout(agent, env, seed)
                per_run.append({"seed": seed, "reward": r, **m})
                print(f"  run {i} (seed {seed}): reward={r:+.3f} "
                      f"dice={m['dice']:.3f} asd={m['asd']:.4f} hd95={m['hd95']:.4f}")

                # Log episodic metrics to wandb
                wandb.log({
                    "episode/run_index": i,
                    "episode/seed": seed,
                    "episode/reward": r,
                    "episode/dice": m["dice"],
                    "episode/asd": m["asd"],
                    "episode/hd95": m["hd95"],
                })
                table.add_data(i, seed, r, m["dice"], m["asd"], m["hd95"])

            wandb.log({"eval_results_table": table})

            # Calculate and log summary metrics
            for key in ("reward", "dice", "asd", "hd95"):
                vals = np.array([r[key] for r in per_run], dtype=np.float64)
                finite = vals[np.isfinite(vals)]
                if len(finite):
                    wandb.run.summary[f"mean_{key}"] = float(finite.mean())
                    wandb.run.summary[f"std_{key}"] = float(finite.std())
                    wandb.log({
                        f"summary/mean_{key}": float(finite.mean()),
                        f"summary/std_{key}": float(finite.std()),
                    })
        finally:
            env.close()
            run.finish()
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
    from cam_env.utils import load_env_or_abort
    load_env_or_abort()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trajectory", type=str,
                     help="path to a saved (T,3) trajectory .npy (e.g. from train_csg.py)")
    src.add_argument("--checkpoints", nargs="+",
                     help="one or more continuous-PPO checkpoints from csg_ppo.py")
    ap.add_argument("--resolution", type=int, default=32,
                    help="fallback voxel count along the stock's longest axis (used only if --voxel-size-mm is 0)")
    ap.add_argument("--voxel-size-mm", type=float, default=VOXEL_SIZE_MM,
                    help="physical voxel edge (mm) for carving/scoring -- the sub-mm precision knob")
    ap.add_argument("--target-shape", type=str, default=TARGET_SHAPE,
                    choices=["sphere", "cylinder", "box", "pyramid",
                             "sphere_hole", "sphere_bowl"],
                    help="target part shape (must match what the trajectory was optimized for)")
    ap.add_argument("--num-runs", type=int, default=10,
                    help="episodes per checkpoint (checkpoint mode)")
    ap.add_argument("--seed", type=int, default=42, help="base seed")
    ap.add_argument("--gcode", action="store_true",
                    help="also round-trip the trajectory through the CAM/G-code layer")
    ap.add_argument("--post", type=str, default="rs274",
                    help="post-processor for --gcode (rs274 | haas)")
    ap.add_argument("--workspace-in", type=float, nargs=3, default=list(WORK_VOLUME_IN),
                    metavar=("X", "Y", "Z"),
                    help="machine work volume (x y z) in inches (default Mini Mill)")
    ap.add_argument("--workspace-mm", type=float, default=None,
                    help="work-volume cube edge length (mm); overrides --workspace-in")
    ap.add_argument("--stock-size-in", type=float, nargs=3, default=list(STOCK_SIZE_IN),
                    metavar=("X", "Y", "Z"),
                    help="stock box (x y z) in inches -- the normalized cube (default 1 in cube)")
    ap.add_argument("--stock-origin-in", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="work origin (G54) = stock top-centre in machine inches")
    args = ap.parse_args()

    from cam import MachineConfig
    cfg = MachineConfig(
        workspace_mm=args.workspace_mm if args.workspace_mm else 100.0,
        workspace_in=None if args.workspace_mm else tuple(args.workspace_in),
        stock_size_in=tuple(args.stock_size_in),
        stock_origin_in=tuple(args.stock_origin_in) if args.stock_origin_in else None,
    )

    if args.trajectory:
        vox = args.voxel_size_mm if args.voxel_size_mm else None
        eval_trajectory(args.trajectory, args.resolution, args.gcode,
                        args.post, cfg, voxel_size_mm=vox,
                        target_shape=args.target_shape)
    else:
        eval_checkpoints(args.checkpoints, args.num_runs, args.seed)


if __name__ == "__main__":
    main()
