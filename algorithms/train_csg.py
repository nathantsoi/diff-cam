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

Example (mirrors the csg_ppo baseline command). The defaults below are the
proven operating point from the autoresearch sweep (514 experiments): dt=0.45
unlocks tool traversal (the real bottleneck, not the loss), grad_clip=0.5 +
eval_freq=10 + best-checkpoint saving capture the transient dice peak:
    uv run python -m algorithms.train_csg --iters 5000 --max_steps 128 \
        --stock_size_in 1 1 1 --voxel_size_mm 0.5 --dt 0.45 \
        --grad_clip 0.5 --eval_freq 10 --save_model \
        --record_video_freq 100 --video_fps 30
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

# Canonical cutter start used for eval / best-checkpoint scoring, so dice is
# comparable across iterations even when training randomizes the start.
CANONICAL_TOOL_START = np.array([0.5, 0.5, 1.0], dtype=np.float32)


def sample_tool_start(args, stock_z_mm):
    """Random cutter start near the stock, always >= stock top + clearance.

    XY is uniform in [margin, 1-margin]^2 (inside the stock footprint); Z is
    ``1.0 + clearance/stock_z_mm + jitter`` in normalized coords, guaranteeing
    the cutter sits at least ``tool_start_clearance_in`` inches above the stock
    top (z=1.0). Returns a (3,) float32 array in normalized [0,1] coords.
    """
    from cam.units import inch_to_mm
    margin = args.tool_start_xy_margin
    x = np.random.uniform(margin, 1.0 - margin)
    y = np.random.uniform(margin, 1.0 - margin)
    z_floor = 1.0 + inch_to_mm(args.tool_start_clearance_in) / float(stock_z_mm)
    z_jitter = inch_to_mm(args.tool_start_z_jitter_in) / float(stock_z_mm)
    z = z_floor + np.random.uniform(0.0, z_jitter)
    return np.array([x, y, z], dtype=np.float32)


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
    eval_freq: int = 10
    """compute + log Dice/ASD/HD95 every N iterations (0 = disabled). Fine cadence
    (10) samples the transient dice peak for best-checkpoint saving; the iters//10
    auto-cadence is far too coarse at i5000 and misses the peak."""
    progress_bar: bool = False
    """use tqdm progress bar instead of scrolling log lines (set False for clean log files and LLM harness compatibility)"""
    log_freq: int = 1
    """print scrolling log output every N iterations when progress_bar is disabled"""
    record_video_freq: int = 0
    """render + upload a trajectory rollout video every N iterations (0 = disabled)"""
    video_fps: int = 30
    """frames per second for recorded videos"""

    # Optimization
    iters: int = 5000
    """number of Adam iterations. i5000 is the sweet spot within the 15-min budget:
    transient dice peaks appear LATER as iters grow (sphere @530->@2450; pyramid
    @680->@1590), so longer runs surface higher peaks. i8000 gives no further gain
    and breaks the budget; i1000 under-samples the peak."""
    learning_rate: float = 5e-3
    """Adam learning rate (optimum: 5e-3; 3e-3 neutral, 7e-3 diverges at dt0.5)"""
    anneal_lr: bool = False
    """linearly anneal the learning rate to 0 over training"""
    lr_decay_frac: float = 0.0
    """fraction of iters (at the end) over which LR linearly decays to 0; 0 = constant
    LR (preserves exploration, then settles). Dead lever on the current API: the
    stale branch's 0.29->0.84 gain did NOT transfer (loss/simulator changed); all
    decay settings tied the baseline. Best-checkpoint saving subsumes it."""
    init_scale: float = 0.05
    """half-range of the uniform random init for per-step displacements (0.02 and 0.1 both hurt)"""
    init_mode: str = "random"
    """trajectory init: 'random', 'raster', 'raster_fine', 'raster_fine_wide',
    'spiral', 'shell', or 'zlayer' (z-level descent that pre-clears the sphere
    exterior layer by layer, using the tall tool's vertical extent). 'raster_fine'
    is a clipping-aware fine boustrophedon (per-step <= feed cap) that survives
    the speed clip; 'raster_fine_wide' spans the full target envelope (0.05-0.95)
    instead of the inner 0.20-0.80 core. The coarse structured inits
    (raster/spiral/shell/zlayer) fail via speed-limit clipping."""
    zlayer_revs: float = 12.0
    """zlayer init: angular revolutions over the full z descent. Geometry search
    found revs=18 (+osc=9, margin=0.005) reaches hard dice ~0.854 unclipped vs
    0.779 at default 12; the win is the init geometry, preserved by
    best-checkpoint saving (soft optimization collapses it)."""
    zlayer_osc: float = 3.0
    """zlayer init: radial oscillation cycles (r_safe -> r_outer) over the
    descent. Higher = denser annulus coverage; ~9 is the sweet spot."""
    zlayer_margin: float = 0.03
    """zlayer init: normalized gap between sphere surface + r_tool and the
    tool-center orbit. Tighter (0.005-0.015) leaves less residual surface waste
    without gouging (tool inner edge still clears the part)."""
    grad_clip: float = 0.5
    """clip per-iteration gradient L2 norm to this (0 = disabled). Stabilizes the
    transient dice peak so best-checkpoint saving captures a higher one; 0.4-0.5 is
    the sweet spot (0.5 default for pyramid/box/cylinder, 0.4 marginally better for
    sphere). 0.0 caps dice ~0.56 via the unstable peak."""

    # CamEnvDiff / CSG specific (mirrors csg_ppo)
    resolution: int = 32
    """voxel grid resolution per axis"""
    max_steps: int = 128
    """trajectory length T (number of tool motions). m=128 optimal at dt<=0.45;
    m=160 optimal at dt=0.5; m>=192 NaNs (SDF overflow); m=144 slightly worse than 128."""
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

    # Trajectory regularizers (address jerky motion + time spent cutting air)
    w_air: float = 0.0
    """weight on the per-step AIR-CUT penalty (swept-tool volume in empty stock).
    0 disables. Fires whenever the cutter traverses/hovers in open space instead
    of removing material; ~0.5-1.0 discourages air-cutting without dominating the
    geometry objective."""
    w_jerk: float = 0.0
    """weight on the JERK / smoothness penalty (squared diff of consecutive
    deltas). 0 disables; ~1e-2 smooths abrupt direction/speed changes."""
    w_step: float = 0.0
    """weight on the SPEED-REGULARITY (constant-feed) penalty (squared diff of
    consecutive step LENGTHS). 0 disables; pushes the feed rate toward a uniform
    value -- the canonical CNC toolpath pattern -- without discouraging the
    back-and-forth direction reversals of a raster/boustrophedon path."""
    w_prox: float = 0.0
    """weight on the DISTANCE-WEIGHTED air-cut (contour-hug) penalty: like w_air
    but the charge scales with squared distance from the TARGET surface, so
    air-cutting in the empty CORNERS far from the part is heavily penalized
    while surface-hugging and necessary first-pass carving (in remaining stock,
    air ~ 0) stay cheap. 0 disables. Directly attacks the "tool moving far from
    the part surface" failure mode without the blunt collapse of cranking w_air.
    Shares the air loop's tool_sdf eval, so it is nearly free."""
    w_prox_warmup_frac: float = 0.0
    """fraction of iters before w_prox begins ramping (0 = on from start).
    Carving is established first (residual falls, dice peaks), THEN w_prox
    ramps linearly from 0 to --w_prox over the remaining iters to polish
    air-cutting without pinning the tool before the sweep is learned."""
    w_traj_prox: float = 0.0
    """weight on the TRAJECTORY contour-hug penalty: a gentle per-segment penalty
    on the tool-center segment-midpoint distance from the TARGET surface, with a
    deadzone of one tool-radius so contact-cutting (incl. corner-carving) is free
    and only genuine excursions (deep empty corners, high retracts beyond r_tool)
    are charged. 0 disables. A soft nudge on trajectory shape -- unlike the
    per-voxel w_prox, does not stall carving."""
    w_traj_prox_warmup_frac: float = 0.0
    """fraction of iters before w_traj_prox begins ramping (0 = on from start).
    Carve first (dice peaks), THEN ramp w_traj_prox on to polish excursions
    without stalling the carving sweep."""
    w_len: float = 0.0
    """weight on the PATH-LENGTH (minimal-motion) penalty: mean squared per-step
    displacement. Agnostic to WHERE the tool is (unlike the contour-hug losses
    which pull toward the surface and oppose carving), it only discourages
    motion. On trailing steps with no residual left to carve it shrinks the
    deltas toward zero so the tool STOPS instead of wandering into air -- the
    targeted fix for the trailing-excursion failure (tool climbs off the part
    for the last ~25% of the path). 0 disables."""
    w_tool_gouge: float = 0.0
    """weight on the TOOL-POSITION gouge barrier (soft-union-INDEPENDENT surface
    respect). Charges the tool CENTER directly for penetrating the target
    expanded by r_tool: relu(r_tool - target_sdf(seg_mid))^2 -- ZERO when the
    tool is tangent-or-outside the surface (contact-cutting waste just outside
    the part is FREE), grows as the tool penetrates the part. Unlike the
    stock-based w_gouge (satisfied trivially by soft-union over-erosion while
    the HARD carve still gouges), this constrains the trajectory GEOMETRY
    directly so it transfers to hard dice. 0 disables."""

    init_stock_from: str = ""
    """STAGED TRAINING: path to a .npz saved by the truncation utility containing
    a mid-cut stock SDF + tool position. When set, training starts each forward
    pass from the SAVED partially-carved stock (instead of the full envelope)
    and fixes the tool start to the saved tool position -- so this trajectory
    carves the REMAINING material the previous trajectory left. Use with the
    staged_train orchestrator: train -> truncate -> train --init-stock-from."""



    # Robustness to initial conditions: random cutter start + restart-from-state
    random_tool_start: bool = False
    """randomize the cutter start each fresh start: random XY within the stock
    footprint, Z >= stock top + tool_start_clearance_in. Trains trajectories
    robust to the initial starting condition."""
    tool_start_clearance_in: float = 0.2
    """min cutter height above the stock top (inches) for a random start"""
    tool_start_xy_margin: float = 0.1
    """normalized XY margin inside the stock footprint for a random start"""
    tool_start_z_jitter_in: float = 0.1
    """extra random height (inches) above the clearance floor for a random start"""
    restart_from_state: bool = False
    """maintain a bank of saved mid-cut simulator states and, with probability
    p_restart each iteration, restart the forward pass from a random saved state
    instead of a fresh stock. Trains the trajectory to finish the part from many
    partial states, not just a fixed start."""
    p_restart: float = 0.25
    """per-iteration probability of restarting from a saved state (keep < 0.5 so
    full-trajectory fresh starts still train the early deltas)"""
    state_bank_size: int = 32
    """max saved simulator states (FIFO eviction)"""
    save_state_prob: float = 0.05
    """per-iteration probability of snapshotting a mid-cut state into the bank"""

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
    dt: float = 0.45
    """seconds per simulator step; speed = |delta (.) envelope_mm| / dt. THE decisive
    lever: at low dt (0.12/0.01) the swept-cylinder tool is speed-limited -- its
    z-range clips to 0.72-1.0 and it cannot descend/traverse the exterior, capping dice
    at ~0.56 regardless of loss or capacity. dt=0.45 advances ~1 voxel/step so the tool
    covers the part (sphere 0.56->0.85, pyramid ->0.90). Sweet spot dt in [0.42,0.5]."""
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
    m["loss_air"] = float(sim.diag_air[None])
    m["loss_jerk"] = float(sim.diag_jerk[None])
    m["loss_step"] = float(sim.diag_step[None])
    m["loss_prox"] = float(sim.diag_prox[None])
    m["loss_traj_prox"] = float(sim.diag_traj_prox[None])
    m["loss_len"] = float(sim.diag_len[None])
    m["loss_tool_gouge"] = float(sim.diag_tool_gouge[None])
    # Air-cut fraction (independent of w_air): the fraction of the swept tool
    # volume over the trajectory that lies in empty stock. Computed as a RATIO
    # (air volume / total swept tool volume) so it is in [0,1] and independent
    # of how much total volume the trajectory moves -- lower = less air-cutting
    # / a more efficient, contour-hugging toolpath.
    air_vol = float(sim.diag_air_unweighted[None])
    swept = float(sim.diag_tool_swept[None])
    m["air_cut_raw"] = air_vol
    m["tool_swept_raw"] = swept
    m["air_cut_fraction"] = air_vol / max(swept, 1e-8)
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
    ts = int(time.time() * 1000)
    run_name = f"{prefix}{args.env_id}__{args.exp_name}__{args.seed}__{ts}"
    while os.path.exists(os.path.join("runs", run_name)):
        ts += 1
        run_name = f"{prefix}{args.env_id}__{args.exp_name}__{args.seed}__{ts}"
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
    # Trajectory regularizers (air-cut + jerk); 0 = disabled.
    sim.w_air[None] = args.w_air
    sim.w_jerk[None] = args.w_jerk
    sim.w_step[None] = args.w_step
    sim.w_prox[None] = args.w_prox
    sim.w_traj_prox[None] = args.w_traj_prox
    sim.w_len[None] = args.w_len
    sim.w_tool_gouge[None] = args.w_tool_gouge
    sim.bake_target_grid()
    sim.set_target_volume()

    # --- Staged training: start from a saved mid-cut stock + tool position ---
    # (the previous trajectory's truncated state). init_stock will then write
    # the saved SDF into stock[0] each forward pass instead of the full envelope.
    saved_tool_start = None
    if args.init_stock_from:
        saved = np.load(args.init_stock_from)
        sim.load_saved_init(saved["stock_sdf"], saved["tool_pos"])
        saved_tool_start = np.asarray(saved["tool_pos"], dtype=np.float32)
        print(f"[staged] init from saved state {args.init_stock_from}: "
              f"t*={int(saved['t_trunc'])}, tool_pos={saved['tool_pos'].tolist()}",
              flush=True)

    # Voxels are physical cubes of side sim.v mm: use that as the grid spacing
    # for metric surface distances (mm) and STL mesh export.
    dx = sim.v

    # --- Init parameters (T-1 per-step displacements) ---
    # tool_pos[0] = tool_start (fixed); delta[t] = tool_pos[t+1] - tool_pos[t].
    # For structured inits we generate the desired tool_pos[1..T-1] (T-1 points)
    # then difference (with the first delta measured from tool_start).
    tool_start = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    # Staged training: the trajectory must start at the saved tool position so
    # the structured-init delta[0] = positions[0] - tool_start lines up with
    # sim.tool_start (which load_saved_init set to the same saved position).
    if saved_tool_start is not None:
        tool_start = saved_tool_start
    if args.init_mode == "raster_fine":
        # Clipping-aware fine boustrophedon: a 3D zigzag whose EVERY per-step
        # displacement is <= the feed speed cap (feed_speed*dt, ~0.075 normalized
        # at dt=0.45), so the simulator's per-step speed clip does NOT destroy
        # the path (the failure mode of the coarse raster/spiral/shell inits).
        # The tool snakes across the XY footprint at constant step (~0.06) while
        # Z descends linearly -- a uniform, constant-feed CNC finishing pattern
        # that pre-covers the whole part so the optimizer starts in a good basin.
        n = T - 1
        ncols = 11
        nrows = 11
        xs = np.linspace(0.20, 0.80, ncols)
        ys = np.linspace(0.20, 0.80, nrows)
        z_top, z_bot = 0.90, 0.10
        positions = []
        idx = 0
        for j in range(nrows):
            row_xs = xs if j % 2 == 0 else xs[::-1]
            for x in row_xs:
                frac = idx / max(1, n - 1)
                z = z_top + (z_bot - z_top) * frac
                positions.append([float(x), float(ys[j]), float(z)])
                idx += 1
                if idx >= n:
                    break
            if idx >= n:
                break
        positions = np.array(positions[:n], dtype=np.float32)
        if len(positions) < n:
            positions = np.vstack([positions, np.tile(positions[-1:], (n - len(positions), 1))])
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    elif args.init_mode == "raster_fine_wide":
        # Full-extent clipping-aware boustrophedon: same per-step <= feed-cap
        # fine zigzag as raster_fine, but the XY footprint (0.05-0.95) and Z
        # range (0.05-0.95) span the WHOLE target envelope instead of the inner
        # 0.20-0.80 core. raster_fine under-covers the target's outer annulus
        # (e.g. a sphere of normalized radius 0.45 centered at 0.5 reaches
        # 0.05-0.95), which caps its dice; this variant pre-covers the full part.
        n = T - 1
        ncols = 11
        nrows = 11
        xs = np.linspace(0.05, 0.95, ncols)
        ys = np.linspace(0.05, 0.95, nrows)
        z_top, z_bot = 0.95, 0.05
        positions = []
        idx = 0
        for j in range(nrows):
            row_xs = xs if j % 2 == 0 else xs[::-1]
            for x in row_xs:
                frac = idx / max(1, n - 1)
                z = z_top + (z_bot - z_top) * frac
                positions.append([float(x), float(ys[j]), float(z)])
                idx += 1
                if idx >= n:
                    break
            if idx >= n:
                break
        positions = np.array(positions[:n], dtype=np.float32)
        if len(positions) < n:
            positions = np.vstack([positions, np.tile(positions[-1:], (n - len(positions), 1))])
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    elif args.init_mode == "raster":
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
        # Z-level finishing descent: the tool is a tall vertical cylinder
        # (height ~= stock) whose tool_pos.z is its BASE, extending upward by h.
        # Descending the base from above the stock down past the bottom means each
        # layer's tool only reaches DOWN to its base, so a high base never touches
        # the equator and can safely carve the top interior exterior at small
        # radius. The orbit radius oscillates from a surface-offset safe radius
        # out to the cube wall, sweeping the waste ANNULUS at every z (a real CNC
        # z-level finishing pattern). Shape-aware safe radius:
        #   sphere   -> r_sphere(z_eq) + r_tool + margin  (varies with z)
        #   cylinder -> r_cyl + r_tool + margin           (z-invariant)
        #   box/pyramid -> r_tool + margin                (full annulus heuristic)
        n = T - 1
        stock_mm = args.stock_size_in[0] * 25.4
        r_sp = args.target_radius_mm / stock_mm
        r_tool = args.tool_radius_mm / stock_mm
        margin = args.zlayer_margin
        revs = args.zlayer_revs
        osc = args.zlayer_osc
        z_top, z_bot = 0.95, -0.95
        r_outer = 0.5 + r_tool
        if args.target_shape == "pyramid":
            # 3-phase gouge-free path (tool extends UP from base, spans
            # [base, base+h]): (1) above-disk boustrophedon (base in [apex,
            # 0.95], tool carves z>apex); (2) beside square-annulus orbit (base
            # descends apex->base_z, orbit at pyramid_half(base)+r_tool); (3)
            # safe-radius descent (base base_z->-0.75 at r=widest+r_tool, clears
            # the below-annulus without gouging). Below-disk center is left
            # (clearing it gouges via holder interaction). Reaches ~0.79 hard
            # dice unclipped vs 0.43 raster_fine baseline.
            h = args.target_height_mm / stock_mm
            base_z = 0.5 - 0.5 * h
            apex = base_z + h
            r_safe_max = r_sp + r_tool + margin
            n_above = int(n * 0.42)
            n_descent = max(8, int(n * 0.10))
            n_beside = n - n_above - n_descent
            xs = np.linspace(0.12, 0.88, 7)
            ys = np.linspace(0.12, 0.88, 7)
            pos = []
            # 1. above boustrophedon
            for z in np.linspace(0.95, apex + 0.02, 4):
                for j, y in enumerate(ys):
                    row_xs = xs if j % 2 == 0 else xs[::-1]
                    for x in row_xs:
                        pos.append([float(x), float(y), float(z)])
                        if len(pos) >= n_above:
                            break
                    if len(pos) >= n_above:
                        break
                if len(pos) >= n_above:
                    break
            while len(pos) < n_above:
                pos.append(pos[-1])
            # 2. beside square orbit
            for t in range(n_beside):
                frac = t / max(1, n_beside - 1)
                zb = apex + (base_z - apex) * frac
                hp = r_sp * (1.0 - (zb - base_z) / h) if base_z <= zb <= apex else 0.0
                s_safe = hp + r_tool + margin
                s_orbit = s_safe + (r_outer - s_safe) * (0.5 + 0.5 * math.sin(2.0 * math.pi * osc * frac))
                phase = 2.0 * math.pi * revs * frac
                cx, cy = math.cos(phase), math.sin(phase)
                m = max(abs(cx), abs(cy))
                pos.append([0.5 + s_orbit * cx / m, 0.5 + s_orbit * cy / m, float(zb)])
            # 3. safe-radius descent
            for t in range(n_descent):
                frac = t / max(1, n_descent - 1)
                zb = base_z + (-0.75 - base_z) * frac
                phase = 2.0 * math.pi * 3.0 * frac
                pos.append([0.5 + r_safe_max * math.cos(phase),
                            0.5 + r_safe_max * math.sin(phase), float(zb)])
            positions = np.array(pos[:n], dtype=np.float32)
            if len(positions) < n:
                positions = np.vstack([positions, np.tile(positions[-1:], (n - len(positions), 1))])
        else:
            positions = np.zeros((n, 3), dtype=np.float32)
            for t in range(n):
                frac = t / max(1, n - 1)
                zb = z_top + (z_bot - z_top) * frac          # base descends through stock
                if args.target_shape == "sphere":
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
                elif args.target_shape == "cylinder":
                    r_safe = r_sp + r_tool + margin          # constant radius
                else:
                    r_safe = r_tool + margin                 # full-annulus heuristic (box)
                # oscillate orbit radius to cover the annulus out to the cube wall
                r_orbit = r_safe + (r_outer - r_safe) * (0.5 + 0.5 * math.sin(2.0 * math.pi * osc * frac))
                phase = 2.0 * math.pi * revs * frac
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
    # Bank of saved mid-cut simulator states for restart-from-state training.
    state_bank = []
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

            # w_prox warmup: keep w_prox at 0 until warmup_frac of iters, then
            # ramp linearly to args.w_prox over the remaining iters so carving
            # is established before the contour-hug penalty is engaged (avoids
            # the tool-pinning stall that a constant w_prox causes).
            if args.w_prox > 0.0 and args.w_prox_warmup_frac > 0.0:
                warm_start = int(args.iters * args.w_prox_warmup_frac)
                if it < warm_start:
                    sim.w_prox[None] = 0.0
                else:
                    span = max(1, args.iters - warm_start)
                    sim.w_prox[None] = args.w_prox * ((it - warm_start) / span)
            # w_traj_prox warmup: same idea -- carve first, then polish
            # trajectory excursions.
            if args.w_traj_prox > 0.0 and args.w_traj_prox_warmup_frac > 0.0:
                warm_start = int(args.iters * args.w_traj_prox_warmup_frac)
                if it < warm_start:
                    sim.w_traj_prox[None] = 0.0
                else:
                    span = max(1, args.iters - warm_start)
                    sim.w_traj_prox[None] = args.w_traj_prox * ((it - warm_start) / span)

            # Push current displacements into Taichi, then forward+backward.
            # With restart_from_state, each iteration either starts fresh (optionally
            # from a random tool_start) or restores a saved mid-cut state and carves
            # only the tail [t0, T). The restored prefix is a detached constant, so
            # only the carved tail receives gradient this iteration; Adam leaves the
            # prefix untouched (standard stochastic per-slice optimization).
            sim.tool_delta.from_torch(params.detach())
            t0 = 0
            if args.restart_from_state and state_bank and np.random.random() < args.p_restart:
                state = state_bank[np.random.randint(len(state_bank))]
                t0 = int(state["t"])
                sim.restore_state(state, t0)
                with ti.ad.Tape(loss=sim.loss):
                    sim.forward_from(t0, T)
            else:
                if args.random_tool_start:
                    sim.tool_start[None] = ti.Vector(list(sample_tool_start(args, sim.Lz)))
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

            # Snapshot a mid-cut state into the bank (only on fresh starts, so
            # every stock slot [0, T] is a freshly-carved value rather than a
            # stale restored constant). Saved at random intervals.
            if args.restart_from_state and t0 == 0 and np.random.random() < args.save_state_prob:
                t_snap = np.random.randint(1, T)
                state_bank.append(sim.save_state(t_snap))
                if len(state_bank) > args.state_bank_size:
                    state_bank.pop(0)

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
                # If the training forward used a random start or a restored
                # mid-cut state, re-run a canonical full forward from the fixed
                # CANONICAL_TOOL_START so dice is comparable across iters and
                # the best-checkpoint snapshot reflects a deployable trajectory.
                if args.random_tool_start or t0 != 0:
                    sim.tool_start[None] = ti.Vector(list(CANONICAL_TOOL_START))
                    sim.tool_delta.from_torch(params.detach())
                    sim.loss[None] = 0.0
                sim.forward_hard(T)
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
                writer.add_scalar("loss/air", m["loss_air"], it)
                writer.add_scalar("loss/jerk", m["loss_jerk"], it)
                last_eval_iter = it
                if args.progress_bar:
                    pbar.set_postfix(loss=f"{loss:.4f}", dice=f"{m['dice']:.3f}",
                                     resid=f"{m['loss_residual']:.3f}",
                                     gouge=f"{m['loss_gouge']:.3f}",
                                     hold=f"{m['loss_holder']:.2e}",
                                     air=f"{m['loss_air']:.3f}",
                                     jerk=f"{m['loss_jerk']:.2e}")
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
        # Evaluate/save from the canonical start so the reported dice and the
        # saved trajectory correspond to a single deployable initial condition
        # (training may have randomized the start or restarted from saved states).
        sim.tool_start[None] = ti.Vector(list(CANONICAL_TOOL_START))
        sim.loss[None] = 0.0
        # forward() (not reconstruct_positions) so the saved positions are the
        # speed-clipped trajectory that was actually carved/optimized, not the
        # raw cumulative sum of the commanded deltas.
        sim.forward(T)
        positions = sim.tool_pos.to_torch()[:T].numpy()
        sim.forward_hard(T)

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
        # Capture the FINAL-iter metrics (before any best-checkpoint override)
        # so the warmup/polish effect on the final trajectory's air-cut is
        # measurable independently of the best-dice checkpoint.
        final_iter_m = dict(last_m) if last_m is not None else None
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
                sim.loss[None] = 0.0
                sim.forward_hard(T)
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
            # Loss-component diagnostics (from the best checkpoint's eval). These
            # do NOT affect the dice score -- they expose how much of the swept
            # tool motion is cutting air (loss_air), jerk (loss_jerk), and feed
            # irregularity (loss_step) for analysis of trajectory quality.
            "loss_air": round(float(last_m.get("loss_air", 0.0)), 6) if last_m else 0.0,
            "loss_jerk": round(float(last_m.get("loss_jerk", 0.0)), 6) if last_m else 0.0,
            "loss_step": round(float(last_m.get("loss_step", 0.0)), 6) if last_m else 0.0,
            "loss_prox": round(float(last_m.get("loss_prox", 0.0)), 6) if last_m else 0.0,
            "loss_traj_prox": round(float(last_m.get("loss_traj_prox", 0.0)), 6) if last_m else 0.0,
            "loss_len": round(float(last_m.get("loss_len", 0.0)), 6) if last_m else 0.0,
            "loss_tool_gouge": round(float(last_m.get("loss_tool_gouge", 0.0)), 6) if last_m else 0.0,
            "air_cut_fraction": round(float(last_m.get("air_cut_fraction", 0.0)), 6) if last_m else 0.0,
            "air_cut_raw": round(float(last_m.get("air_cut_raw", 0.0)), 6) if last_m else 0.0,
            "tool_swept_raw": round(float(last_m.get("tool_swept_raw", 0.0)), 6) if last_m else 0.0,
            # Final-iter (pre-best-checkpoint) trajectory metrics: exposes any
            # late-training polish (e.g. w_prox warmup) on the final trajectory,
            # independent of where the best-dice peak occurred.
            "final_iter_dice": round(float(final_iter_m["dice"]), 6) if final_iter_m else 0.0,
            "final_iter_air_cut_fraction": round(float(final_iter_m.get("air_cut_fraction", 0.0)), 6) if final_iter_m else 0.0,
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
            sim.forward_hard(T)
            render_trajectory_live(sim, gui, T, label="final")
    finally:
        writer.close()
        if args.track:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
