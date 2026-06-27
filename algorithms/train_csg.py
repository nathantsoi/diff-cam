"""GradMill: differentiable-simulation trajectory optimization (the paper's novel method).

This optimizes a tool trajectory (``T-1`` per-step displacements) directly via
Adam over the differentiable Taichi CSG simulator (``CSGSimulatorDelta``) -- no
RL, no policy. Gradients of the terminal geometry loss flow back through every
cut via ``ti.ad.Tape`` into ``sim.tool_delta.grad``.

Logging, metric calculation, video encoding and STL export reuse the *same code
paths* as the PPO baselines (``algorithms/csg_ppo.py``) so the runs are directly
comparable:

* metrics come from ``eval.eval_csg._metrics`` (Dice / ASD / HD95),
* videos are encoded by ``policy_video._encode_mp4`` (ffmpeg) from raymarched
  frames -- identical to ``CamEnvDiff``'s rgb_array path,
* meshes are exported by ``policy_video._sdf_to_stl``,
* WandB / TensorBoard are wired up exactly like ``csg_ppo.py``.

Run outputs are written under ``runs/CamEnvDiff-v0__train_csg__<seed>__<ts>/``
-- the same env/simulator as ``csg_ppo`` (``exp_name`` distinguishes the method).

Example (mirrors the csg_ppo baseline command):
    uv run python -m algorithms.train_csg --iters 128 --resolution 32 \
        --max_steps 64 --save_model --eval_freq 1 --record_video_freq 100 \
        --video_fps 30
"""

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import taichi as ti
import tyro
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from simulator.csg_metrics import _gouge, _residual, sdf_to_mask
from simulator.csg_simulator import CSGSimulatorDelta
from eval.eval_csg import _metrics
from algorithms.policy_video import _encode_mp4, _sdf_to_stl, raymarch_buffer_to_rgb

# Fixed render camera (matches the look of the live GUI / paper figures).
CAM_POS = (2.0, 2.0, 1.6)
CAM_TARGET = (0.5, 0.5, 0.5)
CAM_UP = (0.0, 0.0, 1.0)


@dataclass
class Args:
    exp_name: str = "train_csg"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "diffcam"
    """the wandb's project name"""
    wandb_entity: str = "diffcam"
    """the entity (team) of wandb's project"""
    env_id: str = "CamEnvDiff-v0"
    """run-name prefix; same env/simulator as csg_ppo (exp_name distinguishes the method)"""
    save_model: bool = False
    """whether to save the learned trajectory into the `runs/{run_name}` folder"""

    # eval / video cadence -- measured in Adam iterations (same flags as csg_ppo)
    eval_freq: int = 0
    """compute + log Dice/ASD/HD95 every N iterations (0 = disabled)"""
    record_video_freq: int = 0
    """render + upload a trajectory rollout video every N iterations (0 = disabled)"""
    video_fps: int = 30
    """frames per second for recorded videos"""

    # Optimization
    iters: int = 128
    """number of Adam iterations"""
    learning_rate: float = 5e-3
    """Adam learning rate"""
    anneal_lr: bool = False
    """linearly anneal the learning rate to 0 over training"""

    # CamEnvDiff / CSG specific (mirrors csg_ppo)
    resolution: int = 32
    """voxel grid resolution per axis"""
    max_steps: int = 64
    """trajectory length T (number of tool motions)"""
    target_shape: str = "sphere"
    """target shape: 'box', 'cylinder', 'sphere', 'pyramid'"""
    k_init: float = 10.0
    """initial smoothness parameter for the smooth-min/max SDF ops"""

    # Local interactive view
    headless: bool = False
    """disable the live GUI (auto-disabled if no display is available)"""


def render_trajectory_live(sim, gui, T, label=""):
    """Replay stock[0..T] in the live GUI (interactive runs only)."""
    if gui is None:
        return
    for t in range(T):
        if not gui.running:
            return
        sim.set_current_step(t)
        sim.render(cam_pos=CAM_POS, cam_target=CAM_TARGET, cam_up=CAM_UP,
                   show_stock=True, show_target=True, show_tool=(t < T))
        gui.set_image(sim.raymarch_buffer)
        gui.text(f"{label}  step {t}/{T}", pos=(0.02, 0.97),
                 color=0xFFFFFF, font_size=18)
        gui.show()


def record_video(sim, gui, T, out_path, fps):
    """Render stock[0..T] as raymarched frames and encode one mp4 via ffmpeg.

    Uses the simulator's ``raymarch_buffer`` -- the same renderer ``CamEnvDiff``
    uses -- and ``policy_video._encode_mp4`` -- the same encoder ``csg_ppo``
    uses. Frames are also pushed to the live GUI when one is available. Never
    raises into training. Returns the written path or None.
    """
    frames = []
    for t in range(T):
        if gui is not None and not gui.running:
            break
        sim.set_current_step(t)
        sim.render(cam_pos=CAM_POS, cam_target=CAM_TARGET, cam_up=CAM_UP,
                   show_stock=True, show_target=True, show_tool=(t < T))
        ti.sync()
        frames.append(raymarch_buffer_to_rgb(sim.raymarch_buffer))
        if gui is not None:
            gui.set_image(sim.raymarch_buffer)
            gui.text(f"step {t}/{T}", pos=(0.02, 0.97),
                     color=0xFFFFFF, font_size=18)
            gui.show()
    if not frames:
        return None
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _encode_mp4(frames, out_path, fps)
        return out_path
    except Exception as e:  # never kill training over a video
        print(f"[video] failed to build {out_path}: {e}")
        return None


def eval_metrics(sim, T, dx):
    """Dice/ASD/HD95 (shared `_metrics` path) + gouge/residual of the carved stock."""
    stock = sim.stock.to_numpy()[T - 1]
    target = sim.target.to_numpy()
    m = _metrics(stock, target, dx)  # {"dice", "asd", "hd95"} -- same as csg_ppo
    pred_mask = sdf_to_mask(stock)
    target_mask = sdf_to_mask(target)
    m["gouge"] = float(_gouge(pred_mask, target_mask) * (dx ** 3))
    m["residual"] = float(_residual(pred_mask, target_mask) * (dx ** 3))
    return m


def export_stls(sim, T, dx, run_name, step, track):
    """Export initial stock / carved stock / target meshes (shared `_sdf_to_stl`)."""
    initial_stock = sim.stock.to_numpy()[0].copy()      # before the first cut
    carved_stock = sim.stock.to_numpy()[T - 1].copy()
    target = sim.target.to_numpy().copy()

    mesh_dir = os.path.join("runs", run_name, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    written = []
    for name, sdf in (("stock_initial", initial_stock),
                      ("stock_carved", carved_stock),
                      ("target", target)):
        path = os.path.join(mesh_dir, f"{name}_step_{step:09d}.stl")
        try:
            if _sdf_to_stl(sdf, dx, path):
                written.append(path)
        except Exception as e:
            print(f"[stl] failed to export {name}: {e}")
    print(f"[stl] exported {len(written)} STL(s) to {mesh_dir}")
    if track and written:
        import wandb
        for path in written:
            wandb.save(path, base_path=os.path.dirname(path), policy="now")
    return written


def plot(run_name, iter_X, losses, eval_X, dices, asds, hs95s, gouges, residuals,
         target_volume, gui):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))

    axs[1][1].plot(iter_X, losses);  axs[1][1].set_title("Loss")
    if eval_X:  # metric curves are only populated on eval iterations
        axs[0][0].plot(eval_X, dices);  axs[0][0].set_title("Dice Score")
        axs[0][1].plot(eval_X, asds);   axs[0][1].set_title("ASD");  axs[0][1].set_ylim(0, 1)
        axs[1][0].plot(eval_X, hs95s);  axs[1][0].set_title("HD95"); axs[1][0].set_ylim(0, 1)
        axs[2][0].plot(eval_X, gouges, label="Gouge Volume (-> 0)")
        axs[2][0].axhline(target_volume, color="r", linestyle="--", label="Target Volume")
        axs[2][0].legend(); axs[2][0].set_title("Gouge Volume"); axs[2][0].set_ylim(0, 1)
        axs[2][1].plot(eval_X, residuals, label="Residual Volume (-> 0)")
        axs[2][1].legend(); axs[2][1].set_title("Residual Volume"); axs[2][1].set_ylim(0, 1)
    for ax in axs.ravel():
        ax.set_xlabel("Iteration")

    plt.tight_layout()
    out_path = os.path.join("runs", run_name, "metrics.png")
    plt.savefig(out_path, dpi=120)
    print(f"[plot] saved metrics figure to {out_path}")
    if gui is not None:
        plt.show()


def main():
    args = tyro.cli(Args)

    T = args.max_steps
    dx = 1.0 / args.resolution
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = os.path.join("runs", run_name)
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    print(f"[run] writing outputs to {run_dir}")

    if args.track:
        from cam_env.utils import load_env_or_abort
        load_env_or_abort()
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
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    # Seeding (mirrors csg_ppo)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Live GUI (interactive only) ---
    gui = None
    if not args.headless:
        try:
            gui = ti.GUI("GradMill Training", res=(1024, 768))
        except Exception as e:
            print(f"[gui] no display available, running headless ({e})")
            gui = None

    # --- Simulator setup (must match CamEnvDiff.reset / eval_csg defaults) ---
    sim = CSGSimulatorDelta(resolution=args.resolution, max_steps=T, k_init=args.k_init,
                            target_shape=args.target_shape, tool_start=(0.5, 0.5, 1.0))
    sim.target_params["radius"][None] = 0.4
    sim.target_params["center"][None] = [0.5, 0.5, 0.5]
    sim.tool_radius[None] = 0.05
    sim.tool_height[None] = 0.15
    sim.bake_target_grid()
    sim.set_target_volume()

    # --- Init parameters (T-1 per-step displacements) ---
    init = np.random.uniform(-0.05, 0.05, size=(T - 1, 3)).astype(np.float32)
    params = torch.tensor(init, requires_grad=True)
    opt = torch.optim.Adam([params], lr=args.learning_rate)

    # Metric accumulators (for the final plot). Losses are per-iteration; the
    # geometry metrics are only computed on eval iterations, so they carry their
    # own x-axis (eval_X).
    iter_X, losses = [], []
    eval_X, dices, asds, hs95s, gouges, residuals = [], [], [], [], [], []

    from tqdm import tqdm
    last_video_iter, last_eval_iter = -1, -1
    start_time = time.time()
    it = 0
    pbar = tqdm(range(args.iters), desc=run_name)
    try:
        for it in pbar:
            if gui is not None and not gui.running:
                break

            if args.anneal_lr:
                lrnow = (1.0 - it / max(1, args.iters)) * args.learning_rate
                opt.param_groups[0]["lr"] = lrnow

            # Push current displacements into Taichi, then forward+backward.
            sim.tool_delta.from_torch(params.detach())
            with ti.ad.Tape(loss=sim.loss):
                sim.forward(T)

            grad = sim.tool_delta.grad.to_torch()[:T - 1]  # (T-1, 3)
            params.grad = grad
            opt.step()
            opt.zero_grad()

            loss = float(sim.loss[None])
            grad_norm = float(grad.norm().item())
            losses.append(loss)
            iter_X.append(it)

            # --- per-iter scalars (TensorBoard; synced to wandb via sync_tensorboard) ---
            writer.add_scalar("losses/loss", loss, it)
            writer.add_scalar("charts/grad_norm", grad_norm, it)
            writer.add_scalar("charts/learning_rate", opt.param_groups[0]["lr"], it)
            sps = it / max(1e-9, time.time() - start_time)
            writer.add_scalar("charts/SPS", sps, it)

            do_eval = args.eval_freq > 0 and it % args.eval_freq == 0
            do_video = args.record_video_freq > 0 and it % args.record_video_freq == 0

            # --- eval metrics (shared `_metrics` path; same keys as csg_ppo) ---
            if do_eval:
                m = eval_metrics(sim, T, dx)
                writer.add_scalar("eval/dice", m["dice"], it)
                writer.add_scalar("eval/asd", m["asd"], it)
                writer.add_scalar("eval/hd95", m["hd95"], it)
                writer.add_scalar("metrics/gouge", m["gouge"], it)
                writer.add_scalar("metrics/residual", m["residual"], it)
                eval_X.append(it)
                dices.append(m["dice"]); asds.append(m["asd"]); hs95s.append(m["hd95"])
                gouges.append(m["gouge"]); residuals.append(m["residual"])
                last_eval_iter = it
                pbar.set_postfix(loss=f"{loss:.4f}", dice=f"{m['dice']:.3f}")
            else:
                pbar.set_postfix(loss=f"{loss:.4f}", grad=f"{grad_norm:.2e}")

            # --- video (raymarch -> ffmpeg; logged under media/policy_rollout) ---
            if do_video:
                out_path = os.path.join(video_dir, f"policy_step_{it:09d}.mp4")
                if record_video(sim, gui, T, out_path, args.video_fps) and args.track:
                    import wandb
                    writer.flush()
                    wandb.log(
                        {"media/policy_rollout": wandb.Video(out_path, fps=args.video_fps, format="mp4")},
                        step=it,
                    )
                last_video_iter = it
            elif gui is not None:
                render_trajectory_live(sim, gui, T, label=f"iter {it}")

        # --- Final capture: ensure the last model is recorded (mirrors csg_ppo) ---
        if args.record_video_freq > 0 and it != last_video_iter:
            out_path = os.path.join(video_dir, f"policy_step_{it:09d}.mp4")
            if record_video(sim, gui, T, out_path, args.video_fps) and args.track:
                import wandb
                writer.flush()
                wandb.log(
                    {"media/policy_rollout": wandb.Video(out_path, fps=args.video_fps, format="mp4")},
                    step=it,
                )
        if args.eval_freq > 0 and it != last_eval_iter:
            m = eval_metrics(sim, T, dx)
            writer.add_scalar("eval/dice", m["dice"], it)
            writer.add_scalar("eval/asd", m["asd"], it)
            writer.add_scalar("eval/hd95", m["hd95"], it)
            writer.add_scalar("metrics/gouge", m["gouge"], it)
            writer.add_scalar("metrics/residual", m["residual"], it)
            eval_X.append(it)
            dices.append(m["dice"]); asds.append(m["asd"]); hs95s.append(m["hd95"])
            gouges.append(m["gouge"]); residuals.append(m["residual"])

        # Export the final geometry (initial stock, carved stock, target).
        export_stls(sim, T, dx, run_name, it, args.track)

        # --- Save the learned trajectory (this is GradMill's "model") ---
        deltas = params.detach().numpy()
        sim.tool_delta.from_torch(params.detach())
        sim.reconstruct_positions(T - 1)
        positions = sim.tool_pos.to_torch()[:T].numpy()
        if args.save_model:
            np.save(os.path.join(run_dir, "trajectory_deltas.npy"), deltas)
            np.save(os.path.join(run_dir, "trajectory.npy"), positions)
            print(f"[{run_name}] trajectory saved to {run_dir}/trajectory.npy")
        # Repo-root copy for the CAM round-trip demo / G-code export defaults.
        np.save("trajectory_deltas.npy", deltas)
        np.save("trajectory.npy", positions)

        # Final interactive replay.
        if gui is not None and gui.running:
            sim.tool_delta.from_torch(params.detach())
            sim.forward(T)
            render_trajectory_live(sim, gui, T, label="final")
    finally:
        if iter_X:
            plot(run_name, iter_X, losses, eval_X, dices, asds, hs95s, gouges,
                 residuals, float(sim.target_volume[None]), gui)
        writer.close()
        if args.track:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
