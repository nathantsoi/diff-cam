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


class LegacyAgent(torch.nn.Module):
    """Original flat MLP agent for loading old checkpoints."""
    def __init__(self, envs):
        super().__init__()
        obs_size = np.array(envs.single_observation_space.shape).prod()
        self.critic = torch.nn.Sequential(
            layer_init(torch.nn.Linear(obs_size, 256)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(256, 128)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(128, 1), std=1.0),
        )
        self.actor = torch.nn.Sequential(
            layer_init(torch.nn.Linear(obs_size, 256)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(256, 128)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(128, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        from torch.distributions.categorical import Categorical
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


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


def eval(checkpoint_path, fallback_resolution=None, fallback_max_steps=None):
    # Load checkpoint once
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
        if fallback_resolution is not None and fallback_max_steps is not None:
             print("Warning: Checkpoint missing internal parameters, using provided CLI fallbacks.")
             resolution = fallback_resolution
             max_steps = fallback_max_steps
        else:
             raise ValueError(
                "Checkpoint does not contain 'args' dictionary or injected buffer parameters.\n"
                "This checkpoint appears to be missing its environment parameters (resolution, max_steps).\n"
                "Evaluation now strictly requires checkpoints saved with these parameters OR for you to pass "
                "--resolution and --max-steps explicitly via the CLI."
             )

    # Strip "agent." prefix if present (from PuffeRLWrapperPolicy) and remove injected buffers
    stripped = {}
    for k, v in state_dict.items():
        if k in ["env_resolution", "env_max_steps"]:
            continue
        new_key = k.replace("agent.", "", 1) if k.startswith("agent.") else k
        stripped[new_key] = v
    state_dict = stripped

    print(f"Using resolution={resolution}, max_steps={max_steps}")

    # Dummy vectorized env to get shapes for Agent init
    dummy_envs = pufferlib.vector.make(
        lambda buf=None, **kwargs: pufferlib.emulation.GymnasiumPufferEnv(
            env_creator=lambda: gym.make("CamEnv-v0", resolution=resolution, max_steps=max_steps),
            buf=buf,
        ),
        num_envs=1,
        backend=pufferlib.vector.Serial,
    )

    agent = _make_agent(state_dict, dummy_envs)
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
    parser.add_argument("--resolution", type=int, default=None,
                        help="Fallback resolution for older checkpoints missing internal parameters.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Fallback max_steps for older checkpoints missing internal parameters.")
    args = parser.parse_args()

    eval(args.checkpoint, fallback_resolution=args.resolution, fallback_max_steps=args.max_steps)