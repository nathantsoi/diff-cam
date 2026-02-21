import gymnasium as gym
import torch
import argparse
import time

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
        time.sleep(.3)

        print(f"Step: {info['step']}, Action: {action.item()}, Reward: {reward:.4f}, Value: {value.item():.4f}")

    print(f"\nEpisode finished. Total reward: {total_reward:.4f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    eval(args.checkpoint)