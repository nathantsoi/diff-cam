"""Headless policy-video recording for training runs.

During training we optionally want to *see* what the current policy does, every N
iterations, without a display window -- training runs headless on HPC / inside
Apptainer. ``PolicyVideoRecorder`` rolls the policy out greedily in a dedicated
``render_mode="rgb_array"`` env, encodes the frames to an mp4, and logs both the
video and the same geometric metrics the evaluator reports (Dice / ASD / HD95 of
the carved stock vs target, plus reward) to Weights & Biases.

It is policy/env-agnostic: the greedy action selection and the metric computation
are passed in as callables, so the same recorder serves both the continuous
(``CamEnvDiff-v0``) and discrete (``CamEnvDisc-v0``) trainers. Use the
``make_continuous_recorder`` / ``make_discrete_recorder`` factories.

Frames are encoded by piping raw RGB straight to the system ``ffmpeg``, which
keeps the standard top-left ``(H, W, 3)`` orientation that ``env.render()``
returns and stays env-agnostic.

Recording never crashes training: any failure (missing ffmpeg, render error, ...)
is caught and reported as a warning.
"""

import os
import subprocess

import numpy as np
import torch


class PolicyVideoRecorder:
    """Records greedy policy rollouts to mp4 and uploads them + metrics to wandb.

    Args:
        env: a pre-built env with ``render_mode="rgb_array"``.
        run_name: used for the local ``runs/<run_name>/videos`` path.
        select_action: ``fn(agent, obs_tensor) -> action`` (deterministic). The
            returned action is passed straight to ``env.step``.
        compute_metrics: optional ``fn(env) -> dict`` of geometry metrics logged
            under ``eval/<key>``.
        fps / track / device: encode rate, whether to upload to wandb, torch device.
    """

    def __init__(self, env, run_name, select_action, compute_metrics=None,
                 fps=30, track=False, device="cpu"):
        self._env = env
        self.run_name = run_name
        self.select_action = select_action
        self.compute_metrics = compute_metrics
        self.fps = fps
        self.track = track
        self.device = device

    @torch.no_grad()
    def record(self, agent, global_step, seed):
        """Roll the policy out greedily, encode an mp4, and log it to wandb."""
        try:
            env = self._env

            # --- greedy rollout ---
            obs, _ = env.reset(seed=seed)
            frames = [env.render()]
            total_reward, done = 0.0, False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.select_action(agent, obs_t)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                frames.append(env.render())
                done = terminated or truncated

            # --- geometric metrics of the carved stock vs target (same as eval) ---
            metrics = self.compute_metrics(env) if self.compute_metrics else {}
            metric_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[video] step {global_step}: reward={total_reward:+.4f} {metric_str}")

            # --- encode mp4 (raw RGB -> system ffmpeg) ---
            video_dir = os.path.join("runs", self.run_name, "videos")
            os.makedirs(video_dir, exist_ok=True)
            out_path = os.path.join(video_dir, f"policy_step_{global_step:09d}.mp4")
            _encode_mp4(frames, out_path, self.fps)

            # --- upload video + metrics to wandb ---
            if self.track:
                import wandb

                log = {"eval/reward": total_reward}
                log.update({f"eval/{k}": v for k, v in metrics.items()})
                if os.path.exists(out_path):
                    log["media/policy_rollout"] = wandb.Video(out_path, fps=self.fps, format="mp4")
                wandb.log(log, step=global_step)
            return out_path
        except Exception as e:  # never kill training over a video
            print(f"[video] failed to record policy rollout at step {global_step}: {e}")
            return None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def _encode_mp4(frames, out_path, fps):
    """Pipe raw RGB frames to ffmpeg -> h264 mp4 (standard orientation)."""
    arr = np.ascontiguousarray(np.stack(frames), dtype=np.uint8)  # (T, H, W, 3)
    h, w = arr.shape[1], arr.shape[2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", out_path,
    ]
    try:
        proc = subprocess.run(cmd, input=arr.tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    except FileNotFoundError:
        print(
            "[video] mp4 encoding needs ffmpeg on your PATH "
            "(e.g. `conda install ffmpeg`, `brew install ffmpeg`, "
            "or `apt install ffmpeg`)."
        )
        raise


# ---------------------------------------------------------------------------
# Continuous (CamEnvDiff-v0 / csg_ppo.py)
# ---------------------------------------------------------------------------
def _continuous_action(agent, obs_t):
    # deterministic: mean action (mirrors eval_csg._rollout)
    return agent.actor_mean(agent.features(obs_t)).squeeze(0).cpu().numpy()


def _continuous_metrics(env, resolution):
    from eval.eval_csg import _metrics

    sim = env.unwrapped.simulator
    t = env.unwrapped.current_step
    return _metrics(sim.stock.to_numpy()[t], sim.target.to_numpy(), 1.0 / resolution)


def make_continuous_recorder(run_name, resolution, max_steps, target_shape,
                             env_id="CamEnvDiff-v0", fps=30, track=False, device="cpu"):
    """Recorder for the continuous CSG PPO policy.

    init_taichi=False is critical: the dedicated render env's simulator must
    co-exist with the training env(s) in this process, so it allocates on the
    already-running Taichi runtime instead of resetting it. Build this AFTER the
    training envs (so Taichi is initialized) and before they reset.
    """
    import gymnasium as gym
    import cam_env  # noqa: F401  registers CamEnvDiff-v0

    env = gym.make(env_id, render_mode="rgb_array", resolution=resolution,
                   max_steps=max_steps, target_shape=target_shape, init_taichi=False)
    return PolicyVideoRecorder(
        env, run_name, select_action=_continuous_action,
        compute_metrics=lambda e: _continuous_metrics(e, resolution),
        fps=fps, track=track, device=device,
    )


# ---------------------------------------------------------------------------
# Discrete (CamEnvDisc-v0 / ppo.py)
# ---------------------------------------------------------------------------
def _discrete_action(agent, obs_t):
    # deterministic: argmax over the categorical logits
    logits = agent.actor_head(agent._forward_features(obs_t))
    return int(logits.argmax(dim=-1).item())


def _discrete_metrics(env, resolution):
    from eval.eval import dice, asd, hd95

    sim = env.unwrapped.simulator
    sdf_stock = sim.sdf_stock.to_numpy()
    sdf_target = sim.sdf_target.to_numpy()
    return {
        "dice": dice(sdf_stock, sdf_target),
        "asd": asd(sdf_stock, sdf_target, resolution),
        "hd95": hd95(sdf_stock, sdf_target, resolution),
    }


def make_discrete_recorder(run_name, resolution, max_steps,
                           env_id="CamEnvDisc-v0", fps=30, track=False, device="cpu"):
    """Recorder for the discrete voxel PPO policy.

    The voxel simulator initializes Taichi once at module import, so (unlike the
    continuous side) there is no per-env ti.init reset to work around.
    """
    import gymnasium as gym
    import cam_env  # noqa: F401  registers CamEnvDisc-v0

    env = gym.make(env_id, render_mode="rgb_array", resolution=resolution,
                   max_steps=max_steps)
    return PolicyVideoRecorder(
        env, run_name, select_action=_discrete_action,
        compute_metrics=lambda e: _discrete_metrics(e, resolution),
        fps=fps, track=track, device=device,
    )
