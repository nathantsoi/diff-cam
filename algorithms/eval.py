from flax.linen import checkpoint
import gymnasium as gym
import torch
import argparse
import time
import numpy as np

from algorithms.ppo import Agent, layer_init
import pufferlib
import pufferlib.vector
import pufferlib.emulation

from cam_env.cam_env import CamEnv

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


def _make_agent(state_dict, dummy_envs):
    """Auto-detect architecture from checkpoint keys and create the right agent."""
    if "critic.0.weight" in state_dict:
        # Old flat MLP checkpoint
        print("Detected legacy MLP checkpoint — using LegacyAgent")
        agent = LegacyAgent(dummy_envs)
    else:
        # New 3D CNN checkpoint
        print("Detected 3D CNN checkpoint — using Agent")
        agent = Agent(dummy_envs)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def run_episode(agent, resolution, max_steps, seed=None, render=False):
    """ returns metric for one specific run """
    env = gym.make(
        "CamEnv-v0",
        resolution=resolution,
        max_steps=max_steps,
        render_mode="human" if render else None,
        use_buffer=False,
    )

    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    sim = env.unwrapped.simulator
    initial_stock = float(np.sum(sim.sdf_stock.to_numpy() < 0))
    target_vol = float(np.sum(sim.sdf_target.to_numpy() < 0))

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
            time.sleep(0.05)

    # Final metrics
    sdf_stock = sim.sdf_stock.to_numpy()
    sdf_target = sim.sdf_target.to_numpy()

    # current_stock = float(np.sum(sdf_stock < 0))

    # overlap = float(np.sum((sdf_stock < 0) & (sdf_target < 0)))
    # overlap_pct = 100.0 * overlap / max(target_vol, 1)

    # excess_initial = initial_stock - target_vol
    # excess_now = current_stock - overlap
    # removed_pct = 100.0 * (1.0 - excess_now / max(excess_initial, 1))
    
    env.close()

    return {
        "reward": total_reward,
        "steps": info["step"],
        "dice": dice(sdf_stock, sdf_target),
        "asd": asd(sdf_stock, sdf_target, resolution),
        "hd95": hd95(sdf_stock, sdf_target, resolution)
    }


def evaluate_n_runs(models, resolution, max_steps, num_runs=10, base_seed=42):
    """
    policies: dict[str, agent]
    returns:
        paired_results: list of per-run dicts
        summary: aggregated stats
    """

    paired_results = []

    for i in range(num_runs):
        seed = base_seed + i
        run_data = {"seed": seed}

        for name, agent in models.items():
            metrics = run_episode(agent, resolution, max_steps, seed)
            run_data[name] = metrics

        paired_results.append(run_data)
        print(f"Run {i} done")

    # -------- aggregate --------
    summary = {}

    for name in models.keys():
        metrics = paired_results[0][name].keys()

        summary[name] = {}
        for m in metrics:
            vals = np.array([run[name][m] for run in paired_results])
            summary[name][m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    return paired_results, summary


def load_agent(checkpoint_path, resolution, max_steps):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "args" in checkpoint:
        state_dict = checkpoint.get("agent", checkpoint)
    else:
        state_dict = checkpoint.get("agent", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        
    # Strip "agent." prefix if present (from PuffeRLWrapperPolicy)
    stripped = {}
    for k, v in state_dict.items():
        new_key = k.replace("agent.", "", 1) if k.startswith("agent.") else k
        stripped[new_key] = v
    state_dict = stripped


    dummy_envs = pufferlib.vector.make(
        lambda buf=None, **kwargs: pufferlib.emulation.GymnasiumPufferEnv(
            env_creator=lambda: gym.make(
                "CamEnv-v0",
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
    return agent


# def eval(checkpoint_path, fallback_resolution=None, fallback_max_steps=None):
#     # Load checkpoint once
#     checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

#     resolution = None
#     max_steps = None

#     if isinstance(checkpoint, dict) and "args" in checkpoint:
#         # Our custom checkpoint format
#         saved_args = checkpoint["args"]
#         resolution = saved_args.get("resolution")
#         max_steps = saved_args.get("max_steps")
#         state_dict = checkpoint.get("agent", checkpoint)
#     else:
#         # Native pufferlib checkpoint (or other raw state dict)
#         state_dict = checkpoint.get("agent", checkpoint) if isinstance(checkpoint, dict) else checkpoint

#     # Check for parameters injected as buffers
#     if resolution is None and "env_resolution" in state_dict:
#         resolution = state_dict["env_resolution"].item()
#     if max_steps is None and "env_max_steps" in state_dict:
#         max_steps = state_dict["env_max_steps"].item()

#     if resolution is None or max_steps is None:
#         if fallback_resolution is not None and fallback_max_steps is not None:
#              print("Warning: Checkpoint missing internal parameters, using provided CLI fallbacks.")
#              resolution = fallback_resolution
#              max_steps = fallback_max_steps
#         else:
#              raise ValueError(
#                 "Checkpoint does not contain 'args' dictionary or injected buffer parameters.\n"
#                 "This checkpoint appears to be missing its environment parameters (resolution, max_steps).\n"
#                 "Evaluation now strictly requires checkpoints saved with these parameters OR for you to pass "
#                 "--resolution and --max-steps explicitly via the CLI."
#              )

#     # Strip "agent." prefix if present (from PuffeRLWrapperPolicy) and remove injected buffers
#     stripped = {}
#     for k, v in state_dict.items():
#         if k in ["env_resolution", "env_max_steps"]:
#             continue
#         new_key = k.replace("agent.", "", 1) if k.startswith("agent.") else k
#         stripped[new_key] = v
#     state_dict = stripped

#     print(f"Using resolution={resolution}, max_steps={max_steps}")

#     # Dummy vectorized env to get shapes for Agent init
#     dummy_envs = pufferlib.vector.make(
#         lambda buf=None, **kwargs: pufferlib.emulation.GymnasiumPufferEnv(
#             env_creator=lambda: gym.make("CamEnv-v0", resolution=resolution, max_steps=max_steps),
#             buf=buf,
#         ),
#         num_envs=1,
#         backend=pufferlib.vector.Serial,
#     )

#     agent = _make_agent(state_dict, dummy_envs)
#     dummy_envs.close()

#     # Real env with rendering
#     env = gym.make("CamEnv-v0", resolution=resolution, max_steps=max_steps, render_mode="human")
#     obs, info = env.reset()

#     # Get initial stock/target volumes for comparison
#     sim = env.unwrapped.simulator
#     initial_stock = float(np.sum(sim.sdf_stock.to_numpy() < 0))
#     target_vol = float(np.sum(sim.sdf_target.to_numpy() < 0))

#     print(f"\n{'='*60}")
#     print(f"Evaluating checkpoint: {checkpoint_path}")
#     print(f"Resolution: {resolution}, Max steps: {max_steps}")
#     print(f"Initial stock voxels: {initial_stock:.0f}")
#     print(f"Target voxels:        {target_vol:.0f}")
#     print(f"Voxels to remove:     {initial_stock - target_vol:.0f}")
#     print(f"{'='*60}")
#     print(f"{'Step':>5} {'Action':>12} {'Reward':>8} {'Value':>8} {'Stock':>8} {'Overlap%':>9} {'Removed%':>9}")
#     print(f"{'-'*5} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

#     total_reward = 0
#     done = False

#     while not done:
#         with torch.no_grad():
#             obs_tensor = torch.Tensor(obs).unsqueeze(0)
#             action, _, _, value = agent.get_action_and_value(obs_tensor)

#         obs, reward, terminated, truncated, info = env.step(action.item())
#         total_reward += reward
#         done = terminated or truncated

#         env.render()
#         time.sleep(.15)

#         # Compute progress metrics
#         step = info.get("step", 0)
#         vol = info.get("vol", None)
#         good_cuts = info.get("good_cuts", None)
#         bad_cuts = info.get("bad_cuts", None)
#         boundary_bonus = info.get("boundary_bonus", None)

#         sdf_stock = sim.sdf_stock.to_numpy()
#         sdf_target = sim.sdf_target.to_numpy()
#         current_stock = float(np.sum(sdf_stock < 0))
#         # Overlap: voxels that are in stock AND in target (good — should keep these)
#         overlap = float(np.sum((sdf_stock < 0) & (sdf_target < 0)))
#         overlap_pct = 100.0 * overlap / max(target_vol, 1)
#         # How much excess stock has been removed
#         excess_initial = initial_stock - target_vol
#         excess_now = current_stock - overlap
#         removed_pct = 100.0 * (1.0 - excess_now / max(excess_initial, 1))

#         a = action.item()
#         x = (a // 9) - 1
#         y = ((a // 3) % 3) - 1
#         z = (a % 3) - 1
#         print(f"{info['step']:>5} {str([x,y,z]):>12} {reward:>8.4f} {value.item():>8.4f} {current_stock:>8.0f} {overlap_pct:>8.1f}% {removed_pct:>8.1f}%")

#     print(f"\n{'='*60}")
#     print(f"Episode finished in {info['step']} steps")
#     print(f"Total reward:        {total_reward:.4f}")
#     print(f"Target preserved:    {overlap_pct:.1f}% (want 100%)")
#     print(f"Excess removed:      {removed_pct:.1f}% (want 100%)")
#     print(f"{'='*60}\n")
#     env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)

    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ---- load all models ----
    models = {}

    for ckpt in args.checkpoints:
        print(f"Loading {ckpt}")
        models[ckpt] = load_agent(
            ckpt,
            args.resolution,
            args.max_steps,
        )

    # ---- evaluate ----
    paired_results, summary = evaluate_n_runs(
        models,
        args.resolution,
        args.max_steps,
        num_runs=args.n_runs,
        base_seed=args.seed,
    )

    # ---- print summary ----
    print("\n" + "="*60)
    print(f"Evaluation over {args.n_runs} runs")
    print("="*60)

    for model_name, metrics in summary.items():
        print(f"\nModel: {model_name}")
        for m, stats in metrics.items():
            print(f"{m:15s} | mean={stats['mean']:.3f} std={stats['std']:.3f}")