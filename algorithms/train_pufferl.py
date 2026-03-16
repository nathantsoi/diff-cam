import argparse
import logging
import os
import sys
from pathlib import Path

# Setup standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("train_pufferl")

import torch
import torch.nn as nn

import pufferlib
import pufferlib.vector
from pufferlib import pufferl

from cam_env.vectorized_env import PufferBatchedCamEnv
from algorithms.ppo import Agent


class PuffeRLWrapperPolicy(nn.Module):
    """Wraps the existing PPO Agent to comply with PuffeRL's interface."""
    def __init__(self, agent, resolution=8, max_steps=1024):
        super().__init__()
        self.agent = agent
        self.is_continuous = False
        self.register_buffer("env_resolution", torch.tensor(resolution, dtype=torch.int32))
        self.register_buffer("env_max_steps", torch.tensor(max_steps, dtype=torch.int32))

    def forward(self, observations, state=None):
        features = self.agent._encode(observations)
        logits = self.agent.actor_head(features)
        values = self.agent.critic_head(features)
        return logits, values

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


def train(
    exp_name="diff-cam-batched",
    num_envs=4,
    resolution=8,
    step_size=0.05,
    max_steps=1024,
    total_timesteps=1_000_000,
    learning_rate=2.5e-4,
    device="cuda",
):
    save_dir = Path("experiments") / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Add a file handler to log outputs to the experiment directory
    fh = logging.FileHandler(save_dir / "training.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Starting training run: {exp_name}")
    logger.info(f"Training parameters: num_envs={num_envs}, resolution={resolution}, "
                f"step_size={step_size}, max_steps={max_steps}, "
                f"total_timesteps={total_timesteps}, learning_rate={learning_rate}, device={device}")

    # Set up PufferLib vectorized environment
    logger.info("Initializing PufferBatchedCamEnv...")
    env_creator = lambda **kwargs: PufferBatchedCamEnv(
        num_envs=num_envs,
        resolution=resolution,
        step_size=step_size,
        max_steps=max_steps,
        total_timesteps=total_timesteps,
        **kwargs
    )
    vec_env = pufferlib.vector.make(
        env_creator,
        num_envs=1, # The creator already returns an env with `num_envs` batched inside!
        backend=pufferlib.PufferEnv # Use Native PufferEnv backend because batching is handled internally
    )

    # Initialize the neural network agent
    logger.info("Initializing neural network agent...")
    base_agent = Agent(vec_env).to(device)
    policy = PuffeRLWrapperPolicy(base_agent, resolution=resolution, max_steps=max_steps).to(device)

    # Build PuffeRL train config manually.
    # We cannot use `pufferl.load_config("default")` here because it internally calls 
    # `argparse.parse_args()`, which strictly parses CLI args and throws an error when 
    # it sees our custom args (like --num-envs and --resolution).
    train_cfg = {
        "env": "diff-cam",
        "seed": 42,
        "torch_deterministic": True,
        "cpu_offload": False,
        "device": device,
        "optimizer": "adam",
        "anneal_lr": True,
        "precision": "float32",
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "update_epochs": 4,
        "clip_coef": 0.2,
        "vf_coef": 0.5,
        "vf_clip_coef": 0.2,
        "max_grad_norm": 0.5,
        "ent_coef": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-5,
        "data_dir": "experiments",
        "checkpoint_interval": 999999,  # Disable PuffeRL's native checkpoints
        "batch_size": num_envs * 512,
        "minibatch_size": (num_envs * 512) // 4,
        "max_minibatch_size": 32768,
        "bptt_horizon": 16,
        "compile": False,
        "compile_mode": "max-autotune-no-cudagraphs",
        "compile_fullgraph": True,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "prio_alpha": 0.8,
        "prio_beta0": 0.2,
        "use_rnn": False,
    }

    logger.info(f"Starting batched training with {num_envs} environments on a single GPU (Device: {device})")
    logger.info(f"PuffeRL Training Config: {train_cfg}")

    trainer = pufferl.PuffeRL(train_cfg, vec_env, policy)

    logger.info("Starting training loop...")
    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        trainer.train()
        
        if trainer.epoch % 5 == 0:
            logger.info(f"Epoch {trainer.epoch}/{trainer.total_epochs} completed.")
            trainer.print_dashboard()
            ckpt_path = save_dir / f"model_{trainer.epoch}.pt"
            logger.info(f"Saving checkpoint to {ckpt_path}")
            torch.save({
                "agent": base_agent.state_dict(),
                "args": {
                    "resolution": resolution,
                    "max_steps": max_steps,
                }
            }, ckpt_path)

    logger.info("Training complete.")
    trainer.print_dashboard()
    trainer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    logger.info(f"Parsed CLI arguments: {args}")
    
    train(num_envs=args.num_envs, 
    resolution=args.resolution, 
    step_size=args.step_size,
    max_steps=args.max_steps, 
    total_timesteps=args.total_timesteps, 
    learning_rate=args.learning_rate, 
    device=args.device)