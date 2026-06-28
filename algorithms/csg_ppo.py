# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cam_env.cam_env import CamEnvDiff # needed for env registration


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "diffcam"
    """the wandb's project name"""
    wandb_entity: str = "diffcam"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    record_video_freq: int = 0
    """record + upload a greedy policy rollout video every N training iterations (0 = disabled)"""
    eval_freq: int = 0
    """run a greedy eval rollout + log Dice/ASD/HD95/reward every N iterations (0 = disabled); no video unless the video cadence also lands on this iteration"""
    eval: bool = False
    """if True, compute evaluation metrics (Dice/ASD/HD95/reward) during training and at the end"""
    progress_bar: bool = False
    """use tqdm progress bar instead of scrolling log lines (set False for clean log files and LLM harness compatibility)"""
    log_freq: int = 1
    """print scrolling log output every N iterations when progress_bar is disabled"""
    video_fps: int = 30
    """frames per second for recorded policy videos"""
    video_seed: int = 0
    """seed for the video rollout env (reproducible scenarios across iterations)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CamEnvDiff-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # --- CamEnvDiff specific ---
    resolution: int = 32
    """voxel grid resolution per axis"""
    max_steps: int = 64
    """max episode steps (== max_cuts + 1 in CamEnvDiff)"""
    target_shape: str = "sphere"
    """target shape: 'box', 'cylinder', 'sphere', 'pyramid', or None for random"""
    k_init: float = 10.0
    """initial smoothness parameter for the smooth-min/max SDF ops"""


def make_env(env_id, idx, capture_video, run_name, gamma,
             resolution=32, max_steps=64, target_shape="sphere"):
    def thunk():
        # Only the first env initializes Taichi. ti.init() resets the whole
        # runtime and would invalidate every other env's simulator fields, so
        # envs 1.. allocate on the runtime env 0 set up (init_taichi=False).
        env_kwargs = dict(resolution=resolution, max_steps=max_steps,
                          target_shape=target_shape, init_taichi=(idx == 0))
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class VoxelEncoder(nn.Module):
    """Small 3D CNN over a (2, Nx, Ny, Nz) stack of [stock, target] SDFs.
    """
 
    def __init__(self, grid_shape, out_dim: int = 256):
        super().__init__()
        if isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape, grid_shape)
        self.grid_shape = grid_shape
        # stride-2 downsampling, ReLU. Orthogonal init with sqrt(2) gain.
        self.conv = nn.Sequential(
            layer_init(nn.Conv3d(2, 16, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
        )
        # Determine flat feature size with a dry run.
        with torch.no_grad():
            dummy = torch.zeros(1, 2, *grid_shape)
            feat = self.conv(dummy)
            self.flat_dim = int(np.prod(feat.shape[1:]))
        self.proj = nn.Sequential(
            layer_init(nn.Linear(self.flat_dim, out_dim)),
            nn.ReLU(),
        )
        self.out_dim = out_dim
 
    def forward(self, voxels):
        # voxels: (B, 2, Nx, Ny, Nz)
        h = self.conv(voxels)
        h = h.view(h.size(0), -1)
        return self.proj(h)
 
 
class Agent(nn.Module):
    """Actor-critic over CamEnvDiff's split observation.
 
    Observation layout (must match CamEnvDiff._get_obs):
        [ tool_pos (3) | radius (1) | height (1) | stock_grid (n_vox) | target_grid (n_vox) ]
    """
 
    def __init__(self, envs, resolution: int, initial_logstd: float = -1.0):
        super().__init__()
        self.resolution = resolution
        base_env = envs.envs[0].unwrapped
        if hasattr(base_env, "Nx"):
            self.grid_shape = (base_env.Nx, base_env.Ny, base_env.Nz)
            self.n_vox = base_env.n_vox
        else:
            self.grid_shape = (resolution, resolution, resolution)
            self.n_vox = resolution ** 3
        # Layout offsets for splitting the flat obs.
        self.scalar_dim = 5  # 3 tool_pos + radius + height
        action_dim = int(np.prod(envs.single_action_space.shape))
 
        self.encoder = VoxelEncoder(self.grid_shape, out_dim=256)
 
        feature_dim = self.encoder.out_dim + self.scalar_dim
 
        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, action_dim) * initial_logstd
        )
 
    def _split_obs(self, x):
        """Split flat obs into (scalar, voxels[B, 2, Nx, Ny, Nz])."""
        B = x.size(0)
        scalar = x[:, : self.scalar_dim]                                    # (B, 5)
        stock = x[:, self.scalar_dim : self.scalar_dim + self.n_vox]
        target = x[:, self.scalar_dim + self.n_vox : self.scalar_dim + 2 * self.n_vox]
        stock = stock.view(B, 1, *self.grid_shape)
        target = target.view(B, 1, *self.grid_shape)
        voxels = torch.cat([stock, target], dim=1)                          # (B, 2, Nx, Ny, Nz)
        return scalar, voxels
 
    def features(self, x):
        scalar, voxels = self._split_obs(x)
        h = self.encoder(voxels)
        return torch.cat([h, scalar], dim=1)
 
    def get_value(self, x):
        return self.critic(self.features(x))
 
    def get_action_and_value(self, x, action=None):
        feats = self.features(x)
        action_mean = self.actor_mean(feats)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(feats),
        )


if __name__ == "__main__":
    from cam_env.utils import load_env_or_abort
    load_env_or_abort()

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma,
                  resolution=args.resolution, max_steps=args.max_steps,
                  target_shape=args.target_shape) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, resolution=args.resolution, initial_logstd=-1).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Optional: greedy eval (metrics) and/or video recording during training.
    recorder = None
    if args.record_video_freq > 0 or args.eval_freq > 0 or args.eval:
        from algorithms.policy_video import make_continuous_recorder

        recorder = make_continuous_recorder(
            run_name=run_name, resolution=args.resolution, max_steps=args.max_steps,
            target_shape=args.target_shape, env_id=args.env_id,
            fps=args.video_fps, track=args.track, device=device,
        )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    from tqdm import tqdm
    iteration = 0           # bound even if the loop never runs
    last_video_iter = -1    # iteration whose model was last recorded as video
    last_eval_iter = -1     # iteration whose model was last evaluated (metrics)
    last_m = None
    eval_interval = args.eval_freq if args.eval_freq > 0 else (max(1, args.num_iterations // 10) if args.eval else 0)
    pbar = tqdm(range(1, args.num_iterations + 1), desc=run_name) if args.progress_bar else range(1, args.num_iterations + 1)
    for iteration in pbar:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            writer.add_scalar("charts/algo_reward", np.mean(reward), global_step)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        #tqdm.write(f"[{run_name}] global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        early_stopped = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            break_loop = False
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    early_stopped = True
                    break_loop = True
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if break_loop:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/early_stop", 1.0 if early_stopped else 0.0, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if recorder is not None:
            do_video = args.record_video_freq > 0 and iteration % args.record_video_freq == 0
            do_eval = eval_interval > 0 and (iteration % eval_interval == 0 or iteration == args.num_iterations)
            if do_video or do_eval:
                m = recorder.evaluate(agent, global_step, seed=args.video_seed, record_video=do_video)
                if m is not None:
                    last_m = m
                last_eval_iter = iteration
                if do_video:
                    last_video_iter = iteration

        if args.progress_bar:
            if last_m is not None:
                pbar.set_postfix(SPS=sps, reward=f"{last_m.get('reward', 0.0):+.2f}", dice=f"{last_m.get('dice', 0.0):.3f}")
            else:
                pbar.set_postfix(SPS=sps)
        elif iteration % args.log_freq == 0 or iteration == args.num_iterations:
            lr_val = optimizer.param_groups[0]["lr"]
            if last_m is not None:
                line = (f"[iter {iteration:4d}/{args.num_iterations}] step: {global_step} | vloss: {v_loss.item():.4f} | "
                        f"ploss: {pg_loss.item():.4f} | lr: {lr_val:.2e} | SPS: {sps} | "
                        f"reward: {last_m.get('reward', 0.0):+.2f} | dice: {last_m.get('dice', 0.0):.4f} | "
                        f"asd: {last_m.get('asd', 0.0):.2f} | hd95: {last_m.get('hd95', 0.0):.2f}")
            else:
                line = (f"[iter {iteration:4d}/{args.num_iterations}] step: {global_step} | vloss: {v_loss.item():.4f} | "
                        f"ploss: {pg_loss.item():.4f} | lr: {lr_val:.2e} | SPS: {sps}")
            print(line, flush=True)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        # Self-describing checkpoint: the eval scripts read `args` to rebuild
        # the env (resolution / max_steps / target_shape) and the Agent.
        torch.save({"agent": agent.state_dict(), "args": vars(args)}, model_path)
        tqdm.write(f"[{run_name}] model saved to {model_path}")

    if recorder is not None:
        # Capture the *final* model once: a video if recording is enabled, else a
        # metrics-only eval -- unless the last iteration already did it.
        if args.record_video_freq > 0 and iteration != last_video_iter:
            m = recorder.evaluate(agent, global_step, seed=args.video_seed, record_video=True)
            if m is not None:
                last_m = m
        elif (args.eval or args.eval_freq > 0) and iteration != last_eval_iter:
            m = recorder.evaluate(agent, global_step, seed=args.video_seed, record_video=False)
            if m is not None:
                last_m = m
        if last_m is None and (args.eval or args.eval_freq > 0):
            m = recorder.evaluate(agent, global_step, seed=args.video_seed, record_video=False)
            if m is not None:
                last_m = m
        # Export the final model's geometry (initial stock, carved stock, target).
        recorder.export_stls(agent, global_step, seed=args.video_seed)
        recorder.close()

    # Save summary metrics for automated agents and LLM harnesses.
    import json
    total_seconds = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    final_dice = float(last_m.get("dice", 0.0)) if last_m is not None else 0.0
    final_asd = float(last_m.get("asd", 0.0)) if last_m is not None else 0.0
    final_hd95 = float(last_m.get("hd95", 0.0)) if last_m is not None else 0.0
    final_reward = float(last_m.get("reward", 0.0)) if last_m is not None else 0.0

    summary_data = {
        "dice": round(final_dice, 6),
        "asd": round(final_asd, 6),
        "hd95": round(final_hd95, 6),
        "reward": round(final_reward, 6),
        "training_seconds": round(total_seconds, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "num_steps": global_step,
    }
    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    latest_metrics_path = os.path.join("runs", "latest_metrics.json")
    try:
        with open(latest_metrics_path, "w") as f:
            json.dump(summary_data, f, indent=2)
    except Exception as e:
        print(f"[metrics] failed to write {latest_metrics_path}: {e}")

    print("\n---")
    for k, v in summary_data.items():
        if isinstance(v, float):
            print(f"{k + ':':18s} {v:.6f}")
        else:
            print(f"{k + ':':18s} {v}")
    print("---\n", flush=True)

    envs.close()
    writer.close()