import gymnasium as gym
import torch
import argparse
import time
import numpy as np

from ppo import Agent
import pufferlib
import pufferlib.vector
import pufferlib.emulation

from cam_env.cam_env import CamEnv


def eval(checkpoint_path):
    # Load checkpoint once
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_args = checkpoint["args"]
    resolution = saved_args["resolution"]
    max_steps = saved_args["max_steps"]

    # Dummy vectorized env to get shapes for Agent init
    dummy_envs = pufferlib.vector.make(
        lambda buf=None, **kwargs: pufferlib.emulation.GymnasiumPufferEnv(
            env_creator=lambda: gym.make("CamEnv-v0", resolution=resolution, max_steps=max_steps),
            buf=buf,
        ),
        num_envs=1,
        backend=pufferlib.vector.Serial,
    )

    agent = Agent(dummy_envs)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    dummy_envs.close()

    # Real env with rendering
    env = gym.make("CamEnv-v0", resolution=resolution, max_steps=max_steps, render_mode="human")
    obs, info = env.reset()

    # Get initial stock/target volumes for comparison
    sim = env.unwrapped.simulator
    initial_stock = float(np.sum(sim.sdf_stock.to_numpy() < 0))
    target_vol = float(np.sum(sim.sdf_target.to_numpy() < 0))

    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Resolution: {resolution}, Max steps: {max_steps}")
    print(f"Initial stock voxels: {initial_stock:.0f}")
    print(f"Target voxels:        {target_vol:.0f}")
    print(f"Voxels to remove:     {initial_stock - target_vol:.0f}")
    print(f"{'='*60}")
    print(f"{'Step':>5} {'Action':>12} {'Reward':>8} {'Value':>8} {'Stock':>8} {'Overlap%':>9} {'Removed%':>9}")
    print(f"{'-'*5} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    total_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            action, _, _, value = agent.get_action_and_value(obs_tensor)

        obs, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        done = terminated or truncated

        env.render()
        time.sleep(.15)

        # Compute progress metrics
        sdf_stock = sim.sdf_stock.to_numpy()
        sdf_target = sim.sdf_target.to_numpy()
        current_stock = float(np.sum(sdf_stock < 0))
        # Overlap: voxels that are in stock AND in target (good — should keep these)
        overlap = float(np.sum((sdf_stock < 0) & (sdf_target < 0)))
        overlap_pct = 100.0 * overlap / max(target_vol, 1)
        # How much excess stock has been removed
        excess_initial = initial_stock - target_vol
        excess_now = current_stock - overlap
        removed_pct = 100.0 * (1.0 - excess_now / max(excess_initial, 1))

        a = action.item()
        x = (a // 9) - 1
        y = ((a // 3) % 3) - 1
        z = (a % 3) - 1
        print(f"{info['step']:>5} {str([x,y,z]):>12} {reward:>8.4f} {value.item():>8.4f} {current_stock:>8.0f} {overlap_pct:>8.1f}% {removed_pct:>8.1f}%")

    print(f"\n{'='*60}")
    print(f"Episode finished in {info['step']} steps")
    print(f"Total reward:        {total_reward:.4f}")
    print(f"Target preserved:    {overlap_pct:.1f}% (want 100%)")
    print(f"Excess removed:      {removed_pct:.1f}% (want 100%)")
    print(f"{'='*60}\n")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    eval(args.checkpoint)