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

import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import taichi as ti
import tyro
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
    autoresearch: bool = False
    """if True, prefix the run name with 'AR-' for tracking experiments"""

    # eval / video cadence -- measured in Adam iterations (same flags as csg_ppo)
    eval: bool = False
    """if True, compute evaluation metrics (Dice/ASD/HD95) during training and at the end"""
    eval_freq: int = 0
    """compute + log Dice/ASD/HD95 every N iterations (0 = disabled)"""
    progress_bar: bool = False
    """use tqdm progress bar instead of scrolling log lines (set False for clean log files and LLM harness compatibility)"""
    log_freq: int = 1
    """print scrolling log output every N iterations when progress_bar is disabled"""
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
    lr_decay_frac: float = 0.0
    """fraction of iters (at the end) over which LR linearly decays to 0; 0 = constant LR (preserves exploration, then settles)"""
    init_scale: float = 0.05
    """half-range of the uniform random init for per-step displacements"""
    init_mode: str = "random"
    """trajectory init: 'random', 'raster', 'spiral', 'shell', or 'zlayer' (z-level descent that pre-clears the sphere exterior layer by layer, using the tall tool's vertical extent)"""
    grad_clip: float = 0.0
    """clip per-iteration gradient L2 norm to this (0 = disabled); stabilizes long trajectories (large max_steps) that otherwise NaN"""

    # CamEnvDiff / CSG specific (mirrors csg_ppo)
    resolution: int = 32
    """voxel grid resolution per axis"""
    max_steps: int = 64
    """trajectory length T (number of tool motions)"""
    target_shape: str = "sphere"
    """target shape: 'box', 'cylinder', 'sphere', 'pyramid'"""
    k_init: float = 10.0
    """initial smoothness parameter for the smooth-min/max SDF ops"""

    # Loss balancing (objective vs. safety barriers; see CSGSimulatorDelta)
    w_residual: float = 1.0
    """weight on leftover material outside the part -- the objective that REWARDS cutting"""
    w_gouge: float = 4.0
    """weight on cutting INTO the part -- barrier; > w_residual keeps the cutter just outside the surface"""
    holder_penalty_weight: float = 50.0
    """weight on the holder/stock penetration barrier (one-sided; inactive until the holder contacts stock)"""
    holder_margin: float = 0.0
    """required holder standoff in unit-cube length (>0 keeps a clearance gap before contact)"""

    # Stock box (the normalized cube, voxelized) & machine work volume
    stock_size_in: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """stock box (x, y, z up) in inches -- the normalized cube [0,1]^3 (only this is voxelized)"""
    voxel_size_mm: float = 0.5
    """physical voxel edge in mm -- the sub-mm precision knob (overrides --resolution)"""
    workspace_in: tuple[float, float, float] = (16.0, 12.0, 10.0)
    """machine work volume (x, y, z up) in inches -- default Haas Mini Mill (toolhead limits)"""
    stock_origin_in: tuple[float, float, float] | None = None
    """work origin (G54) = stock top-centre in machine inches (export/validation only)"""

    # Units & speed limits (enforced by per-step clipping in the simulator)
    target_radius_mm: float = 11.43
    """target sphere/cylinder radius (or box/pyramid half-size) in mm (default 0.9 in diameter)"""
    target_height_mm: float = 22.86
    """target cylinder/pyramid height in mm (default 0.9 in); ignored for sphere/box"""
    tool_radius_mm: float = 3.175
    """cutter radius in mm (default 1/4" end mill)"""
    tool_height_mm: float = 25.0
    """cutter flute length in mm"""
    dt: float = 0.01
    """seconds per simulator step; speed = |delta (.) envelope_mm| / dt"""
    rapid_ipm: float = 500.0
    """max traverse speed (inches/min) when clear of the stock"""
    feed_ipm: float = 10.0
    """max cutting speed (inches/min) when within safe distance of the stock"""
    safe_distance_in: float = 0.1
    """clearance (inches) below which a move is limited to feed speed"""
    enforce_speed_limits: bool = True
    """clip each step to its feed/rapid speed cap (disable to run unconstrained)"""

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
    # Holder/stock collision volume across the trajectory (0 = holder stays
    # clear of the remaining stock, which is what we want for safe deployment).
    m["holder_overlap"] = float(sim.holder_overlap_total(T - 1))
    # Weighted loss components, so the objective/barrier balance is observable:
    # loss_residual is what cutting drives down; loss_gouge / loss_holder are the
    # safety barriers that should sit near zero.
    sim.compute_diagnostics(T - 1)
    m["loss_residual"] = float(sim.diag_residual[None])
    m["loss_gouge"] = float(sim.diag_gouge[None])
    m["loss_holder"] = float(sim.diag_holder[None])
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

def main():
    args = tyro.cli(Args)

    T = args.max_steps
    dx = 1.0 / args.resolution
    prefix = "AR-" if args.autoresearch else ""
    run_name = f"{prefix}{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}_{os.getpid()}"
    run_dir = os.path.join("runs", run_name)
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    print(f"[run] writing outputs to {run_dir}")

    # Save reproduction command and arguments
    try:
        import sys
        import shlex
        import json

        # Save reproduction command
        reproduce_cmd_path = os.path.join(run_dir, "reproduce_command.sh")
        cmd_args = ["python", "-m", "algorithms.train_csg"] + sys.argv[1:]
        cmd_str = "#!/bin/bash\n" + " ".join(shlex.quote(arg) for arg in cmd_args) + "\n"
        with open(reproduce_cmd_path, "w") as f:
            f.write(cmd_str)
        try:
            os.chmod(reproduce_cmd_path, 0o755)
        except Exception as e:
            print(f"[run] failed to make reproduce_command.sh executable: {e}")

        # Save parsed parameters
        args_path = os.path.join(run_dir, "args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2)
    except Exception as e:
        print(f"[run] failed to save reproduction files: {e}")

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
    # The normalized cube is the STOCK box (default 1 in cube at 0.5 mm voxels);
    # the work volume (Mini Mill 16x12x10 in) is separate metadata. Sizes are mm.
    sim = CSGSimulatorDelta(resolution=args.resolution, max_steps=T, k_init=args.k_init,
                            target_shape=args.target_shape, tool_start=(0.5, 0.5, 1.0),
                            stock_size_in=args.stock_size_in,
                            voxel_size_mm=args.voxel_size_mm,
                            work_volume_in=args.workspace_in,
                            stock_origin_in=args.stock_origin_in, dt=args.dt,
                            rapid_ipm=args.rapid_ipm, feed_ipm=args.feed_ipm,
                            safe_distance_in=args.safe_distance_in,
                            enforce_speed_limits=args.enforce_speed_limits)
    sim.set_target_params(radius_mm=args.target_radius_mm,
                          height_mm=args.target_height_mm,
                          half_size_mm=args.target_radius_mm,
                          center=(0.5, 0.5, 0.5))
    sim.tool_radius[None] = args.tool_radius_mm
    sim.tool_height[None] = args.tool_height_mm
    # Tool holder: 2.5 inch diameter cylinder above the cutter (mm; default).
    from cam.units import inch_to_mm
    sim.holder_radius[None] = inch_to_mm(2.5 / 2.0)
    # Loss balancing: objective (residual) vs. safety barriers (gouge, holder).
    sim.w_residual[None] = args.w_residual
    sim.w_gouge[None] = args.w_gouge
    sim.holder_penalty_weight[None] = args.holder_penalty_weight
    sim.holder_margin[None] = args.holder_margin
    sim.bake_target_grid()
    sim.set_target_volume()

    # Voxels are physical cubes of side sim.v mm: use that as the grid spacing
    # for metric surface distances (mm) and STL mesh export.
    dx = sim.v

    # --- Init parameters (T-1 per-step displacements) ---
    # tool_pos[0] = tool_start (fixed); delta[t] = tool_pos[t+1] - tool_pos[t].
    # For structured inits we generate the desired tool_pos[1..T-1] (T-1 points)
    # then difference (with the first delta measured from tool_start).
    tool_start = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    if args.init_mode == "raster":
        # Boustrophedon (zigzag) sweep over the cube cross-section at descending
        # z-levels. The tool carves a swept capsule along each segment, so this
        # pre-clears the stock exterior (the region the random init never reaches)
        # and gives the optimizer a trajectory that already covers the whole part.
        n = T - 1
        positions = []
        z_levels = np.linspace(0.90, 0.10, 9)
        rows = np.linspace(0.15, 0.85, 5)
        for z in z_levels:
            for i, y in enumerate(rows):
                xs = np.linspace(0.15, 0.85, 4) if i % 2 == 0 else np.linspace(0.85, 0.15, 4)
                for x in xs:
                    positions.append([x, y, z])
                    if len(positions) >= n:
                        break
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        positions = np.array(positions[:n], dtype=np.float32)
        if len(positions) < n:  # pad with the last point (zero deltas)
            positions = np.vstack([positions, np.tile(positions[-1:], (n - len(positions), 1))])
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    elif args.init_mode == "spiral":
        # Descending spiral with radius growing 0 -> ~0.5 so the tool sweeps the
        # cross-section while descending through the full stock height.
        n = T - 1
        r_max, revs = 0.5, 5.0
        z_top, z_bot = 1.0, 0.05
        positions = np.zeros((n, 3), dtype=np.float32)
        for t in range(n):
            frac = t / max(1, n - 1)
            r = r_max * frac
            phase = 2.0 * np.pi * revs * frac
            positions[t, 0] = 0.5 + r * np.cos(phase)
            positions[t, 1] = 0.5 + r * np.sin(phase)
            positions[t, 2] = z_top + (z_bot - z_top) * frac
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    elif args.init_mode == "shell":
        # Helix that orbits JUST OUTSIDE the target sphere surface while
        # descending through the stock. The tool center rides at
        # r_sphere(z) + tool_radius + margin, so its inner edge clears the
        # exterior annulus without gouging the part (a full-cross-section raster
        # passes through the sphere and gouges it). Sphere-specific.
        n = T - 1
        stock_mm = args.stock_size_in[0] * 25.4
        r_sp = args.target_radius_mm / stock_mm          # normalized sphere radius
        r_tool = args.tool_radius_mm / stock_mm          # normalized tool radius
        margin = 0.02
        revs = 8.0
        z_top, z_bot = 0.95, 0.05
        positions = np.zeros((n, 3), dtype=np.float32)
        for t in range(n):
            frac = t / max(1, n - 1)
            z = z_top + (z_bot - z_top) * frac
            rs = math.sqrt(max(0.0, r_sp * r_sp - (z - 0.5) * (z - 0.5)))
            r_orbit = rs + r_tool + margin
            phase = 2.0 * np.pi * revs * frac
            positions[t, 0] = 0.5 + r_orbit * math.cos(phase)
            positions[t, 1] = 0.5 + r_orbit * math.sin(phase)
            positions[t, 2] = z
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    elif args.init_mode == "zlayer":
        # Z-level descent: the tool is a tall vertical cylinder (height ~= stock)
        # whose tool_pos.z is its BASE, extending upward by h. By descending the
        # base from above the stock down past the bottom, each layer's tool only
        # reaches DOWN to its base, so a high base never touches the equator and
        # can safely carve the top interior exterior at small radius. The safe
        # orbit radius at each base is set by the sphere radius at the
        # equator-closest z the tool reaches, plus the tool radius. A radius
        # oscillation sweeps the annulus out to the cube wall. Sphere-specific.
        n = T - 1
        stock_mm = args.stock_size_in[0] * 25.4
        r_sp = args.target_radius_mm / stock_mm
        r_tool = args.tool_radius_mm / stock_mm
        margin = 0.03
        revs = 12.0
        z_top, z_bot = 0.95, -0.95
        r_outer = 0.5 + r_tool
        positions = np.zeros((n, 3), dtype=np.float32)
        for t in range(n):
            frac = t / max(1, n - 1)
            zb = z_top + (z_bot - z_top) * frac          # base descends through stock
            # equator-closest in-stock z the tool reaches (in [zb, zb+h])
            zhi = zb + 1.0
            if zb > 0.5:
                z_eq = zb
            elif zhi < 0.5:
                z_eq = zhi
            else:
                z_eq = 0.5
            rs = math.sqrt(max(0.0, r_sp * r_sp - (z_eq - 0.5) * (z_eq - 0.5)))
            r_safe = rs + r_tool + margin
            # oscillate orbit radius to cover the annulus out to the cube wall
            r_orbit = r_safe + (r_outer - r_safe) * (0.5 + 0.5 * math.sin(2.0 * math.pi * 3.0 * frac))
            phase = 2.0 * np.pi * revs * frac
            positions[t, 0] = 0.5 + r_orbit * math.cos(phase)
            positions[t, 1] = 0.5 + r_orbit * math.sin(phase)
            positions[t, 2] = zb
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    else:
        init = np.random.uniform(-args.init_scale, args.init_scale, size=(T - 1, 3)).astype(np.float32)
    params = torch.tensor(init, requires_grad=True)
    opt = torch.optim.Adam([params], lr=args.learning_rate)

    from tqdm import tqdm
    last_video_iter, last_eval_iter = -1, -1
    last_m = None
    # Best-checkpoint tracking: dice can peak transiently mid-training (the
    # optimizer over-carves past the optimum, so loss keeps dropping while dice
    # falls). Capture the best-iter trajectory + the dice measured AT that iter
    # and save those instead of the final -- standard "save best validation, not
    # final" practice. We snapshot positions/deltas/metrics directly (not params)
    # and do NOT re-evaluate at the end: the carve is nondeterministic under GPU
    # atomic-adds, so re-running forward on restored params would give a different
    # dice than the one we measured.
    best_dice = -1.0
    best_positions = None
    best_deltas = None
    best_m = None
    best_it = -1
    start_time = time.time()
    it = 0

    eval_interval = args.eval_freq if args.eval_freq > 0 else (max(1, args.iters // 10) if args.eval else 0)
    pbar = tqdm(range(args.iters), desc=run_name) if args.progress_bar else range(args.iters)
    prior_params = params.detach().clone()
    try:
        for it in pbar:
            if gui is not None and not gui.running:
                break

            if args.anneal_lr:
                lrnow = (1.0 - it / max(1, args.iters)) * args.learning_rate
                opt.param_groups[0]["lr"] = lrnow
            elif args.lr_decay_frac > 0.0:
                # Constant LR for the first (1 - lr_decay_frac) of iters, then
                # linearly decay to 0 over the last lr_decay_frac. Keeps early
                # exploration full-strength, then settles the trajectory so the
                # final stock converges instead of oscillating under atomics.
                decay_start = int(args.iters * (1.0 - args.lr_decay_frac))
                if it >= decay_start:
                    span = max(1, args.iters - decay_start)
                    lrnow = args.learning_rate * (1.0 - (it - decay_start) / span)
                    opt.param_groups[0]["lr"] = lrnow

            # Push current displacements into Taichi, then forward+backward.
            sim.tool_delta.from_torch(params.detach())
            with ti.ad.Tape(loss=sim.loss):
                sim.forward(T)

            grad = sim.tool_delta.grad.to_torch()[:T - 1]  # (T-1, 3)
            loss = float(sim.loss[None])
            grad_norm = float(grad.norm().item())

            if math.isnan(loss) or math.isnan(grad_norm) or torch.isnan(grad).any():
                prior_it = max(0, it - 1)
                print(f"\n[WARNING] NaN detected at iteration {it} (loss={loss}, grad_norm={grad_norm}). Stopping training and ending on prior epoch {prior_it}.", flush=True)
                with torch.no_grad():
                    params.copy_(prior_params)
                it = prior_it
                break

            params.grad = grad
            if args.grad_clip > 0.0:
                gn = params.grad.norm()
                if gn > args.grad_clip:
                    params.grad.mul_(args.grad_clip / (gn + 1e-12))
            opt.step()
            opt.zero_grad()
            prior_params = params.detach().clone()

            # --- per-iter scalars (TensorBoard; synced to wandb via sync_tensorboard) ---
            writer.add_scalar("losses/loss", loss, it)
            writer.add_scalar("charts/grad_norm", grad_norm, it)
            writer.add_scalar("charts/learning_rate", opt.param_groups[0]["lr"], it)
            sps = it / max(1e-9, time.time() - start_time)
            writer.add_scalar("charts/SPS", sps, it)

            do_eval = eval_interval > 0 and (it % eval_interval == 0 or it == args.iters - 1)
            do_video = args.record_video_freq > 0 and it % args.record_video_freq == 0

            # --- eval metrics (shared `_metrics` path; same keys as csg_ppo) ---
            if do_eval:
                m = eval_metrics(sim, T, dx)
                last_m = m
                if m["dice"] > best_dice:
                    # sim.tool_delta / sim.tool_pos here reflect the PRE-step
                    # params (set before the forward at the top of the loop),
                    # which are exactly what produced this dice. Snapshot them
                    # directly so the saved best trajectory is consistent with
                    # the measured best dice (no re-eval needed later).
                    best_dice = float(m["dice"])
                    best_m = dict(m)
                    best_positions = sim.tool_pos.to_torch()[:T].numpy().copy()
                    best_deltas = sim.tool_delta.to_torch()[:T].numpy().copy()
                    best_it = it
                writer.add_scalar("eval/dice", m["dice"], it)
                writer.add_scalar("eval/asd", m["asd"], it)
                writer.add_scalar("eval/hd95", m["hd95"], it)
                writer.add_scalar("metrics/gouge", m["gouge"], it)
                writer.add_scalar("metrics/residual", m["residual"], it)
                writer.add_scalar("metrics/holder_overlap", m["holder_overlap"], it)
                writer.add_scalar("loss/residual", m["loss_residual"], it)
                writer.add_scalar("loss/gouge", m["loss_gouge"], it)
                writer.add_scalar("loss/holder", m["loss_holder"], it)
                last_eval_iter = it
                if args.progress_bar:
                    pbar.set_postfix(loss=f"{loss:.4f}", dice=f"{m['dice']:.3f}",
                                     resid=f"{m['loss_residual']:.3f}",
                                     gouge=f"{m['loss_gouge']:.3f}",
                                     hold=f"{m['loss_holder']:.2e}")
            elif args.progress_bar:
                pbar.set_postfix(loss=f"{loss:.4f}", grad=f"{grad_norm:.2e}")

            if not args.progress_bar and (it % args.log_freq == 0 or it == args.iters - 1):
                lr_val = opt.param_groups[0]["lr"]
                time_str = time.strftime("[%Y-%m-%d %H:%M:%S] ")
                if last_m is not None:
                    line = (f"{time_str}[iter {it:4d}/{args.iters}] loss: {loss:.4f} | grad: {grad_norm:.2e} | lr: {lr_val:.2e} | "
                            f"dice: {last_m['dice']:.4f} | asd: {last_m['asd']:.2f} | hd95: {last_m['hd95']:.2f} | "
                            f"resid: {last_m['loss_residual']:.4f} | gouge: {last_m['loss_gouge']:.4f} | hold: {last_m['loss_holder']:.2e}")
                else:
                    line = f"{time_str}[iter {it:4d}/{args.iters}] loss: {loss:.4f} | grad: {grad_norm:.2e} | lr: {lr_val:.2e}"
                print(line, flush=True)

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
        # --- Update simulator state to the final optimized model ---
        deltas = params.detach().numpy()
        sim.tool_delta.from_torch(params.detach())
        # forward() (not reconstruct_positions) so the saved positions are the
        # speed-clipped trajectory that was actually carved/optimized, not the
        # raw cumulative sum of the commanded deltas.
        sim.forward(T)
        positions = sim.tool_pos.to_torch()[:T].numpy()

        if (args.eval or args.eval_freq > 0) and it != last_eval_iter:
            m = eval_metrics(sim, T, dx)
            last_m = m
            writer.add_scalar("eval/dice", m["dice"], it)
            writer.add_scalar("eval/asd", m["asd"], it)
            writer.add_scalar("eval/hd95", m["hd95"], it)
            writer.add_scalar("metrics/gouge", m["gouge"], it)
            writer.add_scalar("metrics/residual", m["residual"], it)
            writer.add_scalar("metrics/holder_overlap", m["holder_overlap"], it)
            writer.add_scalar("loss/residual", m["loss_residual"], it)
            writer.add_scalar("loss/gouge", m["loss_gouge"], it)
            writer.add_scalar("loss/holder", m["loss_holder"], it)

        if last_m is None and (args.eval or args.eval_freq > 0):
            last_m = eval_metrics(sim, T, dx)

        # If a mid-training checkpoint had a better dice than the final iter
        # (transient peak from over-carving), save THAT trajectory and report
        # THAT dice directly. We do NOT re-evaluate: the carve is nondeterministic
        # under GPU atomic-adds, so re-running forward on restored params would
        # give a different dice than the one measured at the best iter. Saving
        # the exact best-iter positions + the measured dice is the honest
        # representation of the best model training found.
        used_best = False
        if best_positions is not None and best_dice > 0.0:
            final_iter_dice = float(last_m["dice"]) if last_m is not None else 0.0
            if best_dice > final_iter_dice:
                positions = best_positions
                deltas = best_deltas
                last_m = best_m
                # Re-load the best trajectory into the sim so STL export and
                # holder-overlap reflect the best model (a fresh carve -- visually
                # equivalent; the reported dice is the already-measured best_m).
                sim.tool_delta.from_torch(torch.as_tensor(best_deltas))
                sim.forward(T)
                used_best = True
                print(f"[best] using best-dice checkpoint: dice={best_dice:.6f} @ iter {best_it} "
                      f"(final-iter dice was {final_iter_dice:.6f})", flush=True)

        final_overlap = float(sim.holder_overlap_total(T - 1))
        if final_overlap > 0.0:
            print(f"[holder] WARNING: final trajectory still collides the holder "
                  f"with the stock (overlap volume {final_overlap:.3e}).")
        else:
            print("[holder] final trajectory keeps the holder clear of the stock.")

        # Save summary metrics for automated agents and LLM harnesses.
        import json
        total_seconds = time.time() - start_time
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        final_dice = float(last_m["dice"]) if last_m is not None else 0.0
        final_asd = float(last_m["asd"]) if last_m is not None else 0.0
        final_hd95 = float(last_m["hd95"]) if last_m is not None else 0.0
        final_gouge = float(last_m["gouge"]) if last_m is not None else 0.0
        final_resid = float(last_m["residual"]) if last_m is not None else 0.0
        final_hold = float(last_m["holder_overlap"]) if last_m is not None else final_overlap

        summary_data = {
            "dice": round(final_dice, 6),
            "asd": round(final_asd, 6),
            "hd95": round(final_hd95, 6),
            "loss": round(float(sim.loss[None]), 6),
            "residual": round(final_resid, 6),
            "gouge": round(final_gouge, 6),
            "holder_overlap": round(final_hold, 6),
            "training_seconds": round(total_seconds, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "num_steps": args.iters,
        }
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

        # Export the final geometry (initial stock, carved stock, target).
        export_stls(sim, T, dx, run_name, it, args.track)

        # --- Save the learned trajectory (this is GradMill's "model") ---
        if args.save_model:
            np.save(os.path.join(run_dir, "trajectory_deltas.npy"), deltas)
            np.save(os.path.join(run_dir, "trajectory.npy"), positions)
            print(f"[{run_name}] trajectory saved to {run_dir}/trajectory.npy")
        # Repo-root copy for the CAM round-trip demo / G-code export defaults.
        # Also copy args.json next to it so the exporter can auto-match the
        # stock/part/location config of this (most recent) run.
        np.save("trajectory_deltas.npy", deltas)
        np.save("trajectory.npy", positions)
        try:
            import json as _json
            with open("args.json", "w") as _f:
                _json.dump(vars(args), _f, indent=2)
        except Exception as e:
            print(f"[run] failed to write repo-root args.json: {e}")

        # Final interactive replay.
        if gui is not None and gui.running:
            sim.tool_delta.from_torch(params.detach())
            sim.forward(T)
            render_trajectory_live(sim, gui, T, label="final")
    finally:
        writer.close()
        if args.track:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
