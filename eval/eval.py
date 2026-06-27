import gymnasium as gym
import torch
import argparse
import time
import numpy as np

from algorithms.ppo import Agent, layer_init
import pufferlib
import pufferlib.vector
import pufferlib.emulation

from cam_env.cam_env_voxel import CamEnvDisc

from algorithms.ppo import *

SEED = 42


import numpy as np
from scipy.ndimage import distance_transform_edt


def _surface_voxels(mask):
    """
    Return boolean mask of surface voxels: voxels inside the shape
    that have at least one neighbor outside (6-connectivity).
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    # A voxel is on the surface if it's inside but any 6-neighbor is outside.
    # Equivalent: inside AND NOT(eroded). We do erosion via shifts to avoid
    # importing more scipy ops.
    surface = np.zeros_like(mask, dtype=bool)
    for axis in range(mask.ndim):
        for shift in (-1, 1):
            shifted = np.roll(mask, shift, axis=axis)
            # Edge voxels: rolling wraps, so blank out the wrap row
            slicer = [slice(None)] * mask.ndim
            slicer[axis] = 0 if shift == 1 else -1
            shifted[tuple(slicer)] = False
            surface |= mask & ~shifted
    return surface


def _surface_distances(sdf_a, sdf_b, resolution):
    """
    For each surface voxel of A, distance to nearest surface voxel of B,
    and vice versa. Returns (d_a_to_b, d_b_to_a) as 1D arrays in physical units.
    """
    dx = 1/resolution

    mask_a = sdf_a < 0
    mask_b = sdf_b < 0

    surf_a = _surface_voxels(mask_a)
    surf_b = _surface_voxels(mask_b)

    if not surf_a.any() or not surf_b.any():
        return np.array([]), np.array([])

    # Distance transform: at every voxel, distance to the nearest TRUE voxel
    # in the input. We invert surf_b so that distance is computed FROM surf_b.
    dt_from_b = distance_transform_edt(~surf_b) * dx
    dt_from_a = distance_transform_edt(~surf_a) * dx

    d_a_to_b = dt_from_b[surf_a]   # at each A surface voxel, distance to nearest B surface
    d_b_to_a = dt_from_a[surf_b]

    return d_a_to_b, d_b_to_a


# ---------- Public metrics ----------

def dice(sdf_stock, sdf_target):
    """Dice Similarity Coefficient. 1.0 = perfect overlap."""
    a = sdf_stock < 0
    b = sdf_target < 0
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def asd(sdf_stock, sdf_target, resolution):
    """Average Symmetric Surface Distance, in physical units."""
    d1, d2 = _surface_distances(sdf_stock, sdf_target, resolution)
    if len(d1) == 0 or len(d2) == 0:
        return float("inf")
    return float((d1.sum() + d2.sum()) / (len(d1) + len(d2)))


def hd95(sdf_stock, sdf_target, resolution):
    """95th-percentile symmetric Hausdorff distance, in physical units."""
    d1, d2 = _surface_distances(sdf_stock, sdf_target, resolution)
    if len(d1) == 0 or len(d2) == 0:
        return float("inf")
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


# -------- Evaluation code --------

def _make_agent(state_dict, dummy_envs):
    """Build the discrete 3D-CNN agent and load weights."""
    agent = Agent(dummy_envs)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def run_episode(agent, env, seed=None, render=True):
    """Run one episode on the given env. Returns metrics dict."""
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    sim = env.unwrapped.simulator
    resolution = env.unwrapped.resolution

    total_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)

        obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        done = terminated or truncated

        if render:
            env.render()
            # bail out if the user closed the window
            if env.unwrapped.window is not None and not env.unwrapped.window.running:
                break

    sdf_stock = sim.sdf_stock.to_numpy()
    sdf_target = sim.sdf_target.to_numpy()

    return {
        "reward": total_reward,
        "steps": info["step"],
        "dice": dice(sdf_stock, sdf_target),
        "asd": asd(sdf_stock, sdf_target, resolution),
        "hd95": hd95(sdf_stock, sdf_target, resolution),
    }


def evaluate_n_runs(models, num_runs=10, base_seed=42, render=False):
    """
    Evaluate each model on num_runs episodes.

    Order: model-outer, run-inner. One env per model, reused across all runs
    of that model, then torn down before moving to the next model. This minimizes
    Taichi window lifecycle churn.

    Pairing is preserved via seed: (model_A, seed=k) and (model_B, seed=k) see
    the same scenario, so downstream paired statistics are still valid.

    Args:
        models: dict[str, dict] with keys 'agent', 'resolution', 'max_steps'.
        num_runs: number of episodes per model.
        base_seed: seed for run i is base_seed + i.
        render: whether to render each episode.

    Returns:
        paired_results: list of per-run dicts in seed-major order, matching
            the original API: [{"seed": s, model_name: metrics, ...}, ...].
        summary: aggregated stats per model per metric.
    """
    # Collect per-model results keyed by seed, so we can reassemble paired
    # order at the end regardless of evaluation order.
    per_model = {name: {} for name in models}

    import os

    import wandb

    for name, model_info in models.items():
        print(f"\nEvaluating {name}")
        env = gym.make(
            "CamEnvDisc-v0",
            resolution=model_info["resolution"],
            max_steps=model_info["max_steps"],
            render_mode="human" if render else None,
        )

        run = wandb.init(
            project="diffcam",
            entity="diffcam",
            job_type="evaluation",
            name=f"eval_checkpoint_{os.path.basename(name)}_{int(time.time())}",
            config={
                "checkpoint_path": name,
                "resolution": model_info["resolution"],
                "max_steps": model_info["max_steps"],
                "num_runs": num_runs,
                "base_seed": base_seed,
            }
        )

        table = wandb.Table(columns=["run_index", "seed", "reward", "steps", "dice", "asd", "hd95"])
        try:
            for i in range(num_runs):
                seed = base_seed + i
                metrics = run_episode(
                    model_info["agent"],
                    env,
                    seed=seed,
                    render=render,
                )
                per_model[name][seed] = metrics
                print(
                    f"run {i} (seed {seed}): "
                    f"reward={metrics['reward']:+.3f} "
                    f"dice={metrics['dice']:.3f} "
                    f"asd={metrics['asd']:.4f} "
                    f"hd95={metrics['hd95']:.4f}"
                )

                # Log episodic metrics to wandb
                wandb.log({
                    "episode/run_index": i,
                    "episode/seed": seed,
                    "episode/reward": metrics["reward"],
                    "episode/steps": metrics["steps"],
                    "episode/dice": metrics["dice"],
                    "episode/asd": metrics["asd"],
                    "episode/hd95": metrics["hd95"],
                })
                table.add_data(i, seed, metrics["reward"], metrics["steps"], metrics["dice"], metrics["asd"], metrics["hd95"])

            wandb.log({"eval_results_table": table})

            # Calculate and log summary metrics
            for key in ("reward", "steps", "dice", "asd", "hd95"):
                vals = np.array([per_model[name][s][key] for s in per_model[name]], dtype=np.float64)
                finite = vals[np.isfinite(vals)]
                if len(finite):
                    wandb.run.summary[f"mean_{key}"] = float(finite.mean())
                    wandb.run.summary[f"std_{key}"] = float(finite.std())
                    wandb.run.summary[f"min_{key}"] = float(finite.min())
                    wandb.run.summary[f"max_{key}"] = float(finite.max())
                    wandb.log({
                        f"summary/mean_{key}": float(finite.mean()),
                        f"summary/std_{key}": float(finite.std()),
                        f"summary/min_{key}": float(finite.min()),
                        f"summary/max_{key}": float(finite.max()),
                    })
        finally:
            env.close()
            run.finish()

    # Reassemble in seed-major order to preserve the original paired_results shape.
    paired_results = []
    for i in range(num_runs):
        seed = base_seed + i
        run_data = {"seed": seed}
        for name in models:
            run_data[name] = per_model[name][seed]
        paired_results.append(run_data)

    # Aggregate
    summary = {}
    for name in models:
        metric_keys = paired_results[0][name].keys()
        summary[name] = {}
        for m in metric_keys:
            vals = np.array([run[name][m] for run in paired_results])
            # Filter inf (asd/hd95 return inf when a surface is empty)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                summary[name][m] = {"mean": float("inf"), "std": 0.0,
                                    "min": float("inf"), "max": float("inf")}
            else:
                summary[name][m] = {
                    "mean": float(np.mean(finite)),
                    "std":  float(np.std(finite)),
                    "min":  float(np.min(finite)),
                    "max":  float(np.max(finite)),
                }

    return paired_results, summary


def load_agent(checkpoint_path):
    """Load an agent and its environment hyperparameters from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    resolution = None
    max_steps = None

    if isinstance(checkpoint, dict) and "args" in checkpoint:
        # Our custom checkpoint format
        saved_args = checkpoint["args"]
        resolution = saved_args.get("resolution")
        max_steps = saved_args.get("max_steps")
        state_dict = checkpoint.get("agent", checkpoint)
    else:
        # Native pufferlib checkpoint (or other raw state dict)
        state_dict = checkpoint.get("agent", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Check for parameters injected as buffers
    if resolution is None and "env_resolution" in state_dict:
        resolution = state_dict["env_resolution"].item()
    if max_steps is None and "env_max_steps" in state_dict:
        max_steps = state_dict["env_max_steps"].item()

    if resolution is None or max_steps is None:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' does not contain environment parameters "
            f"(resolution, max_steps) in 'args' dict or as injected buffers. "
            f"Cannot evaluate without these."
        )

    # Strip "agent." prefix if present (from PuffeRLWrapperPolicy) and remove injected buffers
    stripped = {}
    for k, v in state_dict.items():
        if k in ("env_resolution", "env_max_steps"):
            continue
        new_key = k.replace("agent.", "", 1) if k.startswith("agent.") else k
        stripped[new_key] = v
    state_dict = stripped

    print(f"  -> resolution={resolution}, max_steps={max_steps}")

    dummy_envs = pufferlib.vector.make(
        lambda buf=None, **kwargs: pufferlib.emulation.GymnasiumPufferEnv(
            env_creator=lambda: gym.make(
                "CamEnvDisc-v0",
                resolution=resolution,
                max_steps=max_steps,
            ),
            buf=buf,
        ),
        num_envs=1,
        backend=pufferlib.vector.Serial,
    )

    agent = _make_agent(state_dict, dummy_envs)
    dummy_envs.close()

    return {
        "agent": agent,
        "resolution": resolution,
        "max_steps": max_steps,
    }

# KNOWN ERROR: eval script exits after 5 runs if rendering on
if __name__ == "__main__":
    from cam_env.utils import load_env_or_abort
    load_env_or_abort()

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", dest="render", action="store_true",
                        help="render each episode in a Taichi window (needs a display)")
    parser.add_argument("--no-render", dest="render", action="store_false",
                        help="run headless (default)")
    parser.set_defaults(render=False)

    args = parser.parse_args()

    # ---- load all models ----
    models = {}

    for ckpt in args.checkpoints:
        print(f"Loading {ckpt}")
        models[ckpt] = load_agent(ckpt)

    # Warn if checkpoints disagree on env hyperparameters
    resolutions = {m["resolution"] for m in models.values()}
    max_steps_set = {m["max_steps"] for m in models.values()}
    if len(resolutions) > 1 or len(max_steps_set) > 1:
        print(
            "\nWARNING: checkpoints have differing env hyperparameters. "
            f"resolutions={resolutions}, max_steps={max_steps_set}. "
            "Each model will run in its own env; results are not strictly paired.\n"
        )

    # ---- evaluate ----
    paired_results, summary = evaluate_n_runs(
        models,
        num_runs=args.num_runs,
        base_seed=args.seed,
        render=args.render,
    )

    # ---- print summary ----
    print("\n" + "="*60)
    print(f"Evaluation over {args.num_runs} runs")
    print("="*60)

    for model_name, metrics in summary.items():
        print(f"\nModel: {model_name}")
        for m, stats in metrics.items():
            print(f"{m:15s} | mean={stats['mean']:.3f} std={stats['std']:.3f}")