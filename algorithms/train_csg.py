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
import sys
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
CAM_TARGET = None  # None -> sim.render targets the stock box center
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


def target_cross_section_radii(sim):
    """Shape-agnostic per-z target cross-section radius profile (normalized).

    Reads the baked target SDF grid (``sim.target``) -- the actual carved
    geometry, NOT any shape name or shape parameter -- so it generalizes to any
    target including unseen combined-CSG shapes. Returns a ``(Nz,)`` array where
    entry k is the max distance from the stock center (0.5, 0.5) among voxels
    INSIDE the target (SDF < 0) at z-slice k, or 0 if that slice has no target
    voxels. The shell/zlayer inits use this to orbit just outside the target
    surface without gouging it, with no task-specific metadata.
    """
    tgt = sim.target.to_numpy()                          # (Nx, Ny, Nz)
    Nx, Ny, Nz = tgt.shape
    xs = (np.arange(Nx) + 0.5) / Nx - 0.5                # normalized x offset from center
    ys = (np.arange(Ny) + 0.5) / Ny - 0.5
    R = np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2)     # (Nx, Ny) dist from center
    inside = tgt <= 0.0                                  # (Nx, Ny, Nz) target occupancy
    # max R among inside voxels per z-slice; -1 sentinel where none are inside.
    r_cross = np.where(inside, R[:, :, None], -1.0).max(axis=(0, 1))
    return np.clip(r_cross, 0.0, None)                   # (Nz,)


def _r_cross_at(r_cross, z):
    """Sample the cross-section radius profile at normalized height z."""
    k = int(np.clip(z * len(r_cross), 0, len(r_cross) - 1))
    return float(r_cross[k])


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
    use_feedback: bool = False
    """if True, warm-start the trajectory from the highest-rated prior run (by
    human star rating) that matches this target_shape + max_steps, then continue
    optimizing — a lightweight RLHF-style policy improvement. The feedback store
    (autoresearch/tasks/train_csg/run_feedback.json, written by the web
    dashboard's star/note UI) is ALWAYS read and logged + recorded in
    metrics.json; this flag only gates whether the warm-start deltas are applied.
    Shape-agnostic in the method: matching is in this selection layer (reading
    prior args.json metadata), not in the optimizer/init/loss code."""
    autoresearch: bool = False
    """if True, prefix the run name with 'AR-' for tracking experiments"""
    runs_subdir: str = ""
    """optional subdirectory under runs/ to write the run into (e.g. 'jul8-multidepth'); empty = top-level runs/. Lets the web dashboard group runs into a batch without moving them after the fact."""

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
    transient dice peaks appear LATER as iters grow, so longer runs surface
    higher peaks. i8000 gives no further gain and breaks the budget; i1000
    under-samples the peak."""
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
    'spiral', 'shell', 'zlayer', or 'multidepth'. 'raster_fine' is a
    clipping-aware fine boustrophedon (per-step <= feed cap) that survives the
    speed clip; 'raster_fine_wide' spans the full target envelope (0.05-0.95)
    instead of the inner 0.20-0.80 core. 'shell'/'zlayer'/'multidepth' derive
    the target surface from the baked SDF grid (shape-agnostic -- no task
    metadata). 'multidepth' is a continuous multi-depth helical roughing: it
    descends through the target's full z-extent while a triangle-wave radius
    sweeps the waste annulus from the cube wall inward to the target surface
    + r_tool (bulk removal without gouging), arc-length-resampled + auto-revs
    so the whole path fits the speed-clip budget and survives it. The coarse
    structured inits (raster/spiral/shell/zlayer) fail via speed-limit
    clipping."""
    zlayer_revs: float = 12.0
    """zlayer init: angular revolutions over the full z descent. Higher revs =
    denser annulus coverage; the win is the init geometry, preserved by
    best-checkpoint saving (soft optimization collapses it)."""
    zlayer_osc: float = 3.0
    """zlayer init: radial oscillation cycles (r_safe -> r_outer) over the
    descent. Higher = denser annulus coverage."""
    zlayer_margin: float = 0.03
    """zlayer init: normalized gap between the target surface + r_tool and the
    tool-center orbit. Tighter (0.005-0.015) leaves less residual surface waste
    without gouging (tool inner edge still clears the part)."""

    multidepth_levels: float = 5.0
    """multidepth init: number of RADIAL sweep cycles across the waste annulus
    (r_outer -> r_safe -> r_outer) over the full z descent. More cycles = denser
    radial coverage of the annulus (more multi-depth passes through the bulk
    waste). The helix descends continuously through the target's full z-extent
    (read shape-agnostically from the baked SDF grid) -- this is the
    multi-depth aspect, not discrete z-levels."""
    multidepth_revs: float = 3.0
    """multidepth init: angular revolutions of the helix over the full z
    descent. More revs = denser angular coverage but longer arc; revs is
    auto-shrunk so the total path fits the speed-clip budget (the tool can
    only traverse (T-1)*feed_cap arc-units), so the full z-extent stays
    reachable with every step <= the feed cap."""
    multidepth_margin: float = 0.02
    """multidepth init: normalized gap between target surface + r_tool and the
    innermost spiral radius (keeps the tool tangent-or-outside the part -> no
    gouge). Same role as zlayer_margin."""
    grad_clip: float = 0.5
    """clip per-iteration gradient L2 norm to this (0 = disabled). Stabilizes the
    transient dice peak so best-checkpoint saving captures a higher one; 0.4-0.5
    is the sweet spot. 0.0 caps dice via the unstable peak."""

    # --- Spline-swept-volume method (method="sweep"; see simulator/sweep.py) ---
    method: str = "delta"
    """optimization method: 'delta' (per-step displacements through the
    sequential soft carve — the original GradMill) or 'sweep' (cubic B-spline
    control points through the one-shot swept-volume carve: the final geometry
    is max(stock0, -min_s seg_sdf), order-independent and hard, so training
    optimizes a near-unbiased surrogate of the hard-carve dice)."""
    n_ctrl: int = 40
    """sweep: number of B-spline control points (path capacity knob)"""
    sweep_init: str = "raster"
    """sweep: init reference path the control points are least-squares fitted
    to: 'raster' (boustrophedon over the target bbox), 'raster_arc' (tool-sized
    serpentine z-layer raster, arc-length-uniform samples), 'helix', or 'random'.
    Shape-agnostic (reads only the baked target SDF grid's bounding box)."""
    amin_refresh: int = 1
    """sweep: recompute the per-voxel argmin (winning segment) every this many
    iterations. 1 = exact (every iteration). The argmin pass is O(T * N^3) and
    dominates per-iteration cost at large T; the winner index is stable under
    clipped sub-voxel path motion, so 4-8 trades exactness for ~2-4x throughput
    with a periodic re-tightening."""
    reach_gate: bool = False
    """sweep: gate the residual + attraction loss terms by the exact vertical
    3-axis reachability mask (utils/reachability.py) so waste that NO legal
    tool position can remove stops pulling the path into part walls.
    Shape-agnostic (derived from the target SDF grid + tool radius)."""
    w_feed: float = 5.0
    """sweep: weight of the feed-cap penalty relu(speed/cap - 1)^2 (dimensionless
    excess) on the sampled path -- keeps every step within the feed clip so the
    evaluator's speed clipping is a no-op and the swept model matches the hard
    carve. Only needs to hold at convergence; moderate weights beat huge ones
    (a huge weight drowns the geometry gradient early)."""
    w_broad: float = 0.0
    """sweep: weight of the non-saturating residual attraction term
    relu(d_swept)^2 on uncut waste voxels (SDF-valued, so material far from the
    swept tube still pulls the nearest segment; 0 disables)"""
    sigma_broad: float = 4.0
    """sweep: distance scale (voxels) normalizing the attraction term"""

    # --- Physical plausibility (sweep method; see idea.md jul13-phys-plausible) ---
    w_force: float = 0.0
    """sweep: weight of the tool cutting-force penalty relu(F/f_max - 1)^2,
    F[s] = kc * chip_mm3[s] / len_mm[s] (mechanistic chip-area force) with
    sequential first-cover chip attribution. Bounds material removal per pass
    so the optimized path cannot demand cuts that snap the end mill. 0 = off."""
    w_fragile: float = 0.0
    """sweep: weight of the part-fragility penalty relu(F*finv - 1)^2: segments
    cutting within contact range of a slender target feature (thin pin, raised
    letter) get a tighter force cap = that feature's cantilever breakage force
    (utils/fragility.py, sigma_y * t^3/(6h)). 'Light passes near thin walls'.
    0 = off."""
    w_ramp: float = 0.0
    """sweep: weight of the plunge penalty relu(-dz - tan(ramp_deg)*|dxy|)^2 on
    ENGAGED steps (chip > 0): an end mill cannot feed axially like a drill, so
    engaged descent steeper than ramp_deg is penalized (CAM ramp-entry rule).
    0 = off."""
    ramp_deg: float = 3.0
    """max engaged descent angle (degrees) for w_ramp and --sweep-init-ramp;
    CAM practice is 2-5 degrees for ramp entry"""
    sigma_y: float = 276.0
    """part material bending strength (MPa) for the fragility field. 276 = Al
    6061 yield; ~10 = machining wax; ~70 = acrylic."""
    spindle_rpm: float = 5000.0
    """spindle speed for the force scale: cutting force ~ kc*MRR/v_c with
    v_c = pi*D*rpm/60 (energy is delivered via spindle rotation, not feed
    travel — without this the chip-area force overestimates ~400x)"""
    frag_thin_mm: float = 0.0
    """fragility thinness threshold radius (mm); features with local
    half-thickness below this are fragile candidates. 0 = auto (tool radius)."""
    frag_contact_mm: float = 1.0
    """force-transmission distance (mm) beyond a fragile feature's surface:
    cutting within this band pushes on the feature"""
    sweep_init_ramp: bool = False
    """sweep: enter each raster_arc/raster_terrain z-layer by ramping along the
    first scan row at ramp_deg instead of plunging the stepdown vertically at
    the corner (constructive plunge-free init)"""

    # CamEnvDiff / CSG specific (mirrors csg_ppo)
    resolution: int = 32
    """voxel grid resolution per axis"""
    max_steps: int = 128
    """trajectory length T (number of tool motions). m=128 optimal at dt<=0.45;
    m=160 optimal at dt=0.5; m>=192 NaNs (SDF overflow); m=144 slightly worse than 128."""
    target_shape: str = "sphere"
    """target shape name (selects the CSG primitive the simulator carves toward).
    The optimizer itself is shape-agnostic: it reads only the baked target SDF
    grid, so any supported shape -- including combined-CSG shapes and unseen
    additions -- trains without per-shape code or metadata."""
    target_sdf_path: str | None = None
    """path to a .npz target grid from utils/step_to_sdf.py (required when
    --target_shape grid). The npz shape overrides --resolution/--voxel_size_mm."""
    k_init: float = 10.0
    """initial smoothness parameter for the smooth-min/max SDF ops"""
    k_anneal: bool = False
    """linearly ramp the smoothness k from k_init to k_final over training.
    Continuation method: low k early = smooth landscape, broad gradients (good
    exploration of the carve basin); high k late = sharp union, soft loss
    tracks HARD coverage (polish of real carving). May beat any fixed k by
    combining exploration with hard-tracking. Only meaningful with the stable
    smooth_max (high k no longer NaNs)."""
    k_final: float = 10.0
    """final smoothness k when --k-anneal is on (ramped linearly from k_init)."""

    loss_shift: float = 0.0
    """de-biased (hard-carve-aware) loss: add this to stock_d before the loss
    sigmoid to compensate soft-union over-erosion (~log(2)*k_ref/k_final at the
    final sharpness) so the loss targets the deployable HARD carve. 0 = off."""

    # Loss balancing (objective vs. safety barriers; see CSGSimulatorDelta)
    w_residual: float = 1.0
    """weight on leftover material outside the part -- the objective that REWARDS cutting"""
    w_gouge: float = 4.0
    """weight on cutting INTO the part -- barrier; > w_residual keeps the cutter just outside the surface"""
    holder_penalty_weight: float = 50.0
    """weight on the holder/stock penetration barrier (one-sided; inactive until the holder contacts stock)"""
    holder_margin: float = 0.0
    """required holder standoff in unit-cube length (>0 keeps a clearance gap before contact)"""

    # Z-floor: hard limit on the tool BASE's negative-z travel. The zlayer init
    # descends the tool base below the part to carve the below-part slab, but the
    # tool extends UPWARD from its base by tool_height and the wide holder rides
    # above that -- so a deep base plunge drops the holder into the remaining
    # stock (a machine crash: the spindle slams the workpiece). The floor clamps
    # the EXECUTED move's base z (the init deltas may still command a deeper z),
    # exactly like the feed/rapid speed clip. Set --no-enforce-z-floor to disable.
    z_floor_epsilon_mm: float = 1.0
    """allowed tool-base travel below the part bottom, in mm. The floor is
    part_bottom_z - epsilon/stock_z_mm; the executed base z is clamped to it.
    part_bottom_z is read shape-agnostically from the baked target SDF grid
    (lowest solid voxel), so it is correct for any target including combined-CSG
    shapes. The crash-free floor (holder bottom >= stock top) is roughly
    1 - tool_height/stock_z ~= 0.016 for the default tool/stock, i.e. epsilon
    must keep the floor >= ~0; epsilon=1.0 carves a hair below the part and
    relies on truncate_collision as the backstop."""
    enforce_z_floor: bool = True
    """clamp the executed tool base z to the z-floor (disable to recover the
    unbounded deep-plunge behaviour; truncate_collision then catches the crash)"""

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
    tool_gouge_margin_mm: float = 0.0
    """margin (mm) added to the tool radius in the TOOL-POSITION gouge barrier.
    The barrier fires when target_sdf(tool_center) < r_tool + margin, i.e. the
    tool center must stay `margin` mm FURTHER off the surface than mere
    tangency. Convex parts (sphere) gouge at pass seams where overlapping
    TANGENT capsules bite into the part -- loss_tool_gouge=0 at midpoints yet
    the boolean union still over-erodes; a positive margin lifts the tool so the
    union of capsules stays tangent-only, trading a little uncut residual for
    no gouge. Shape-agnostic (target_sdf only). 0 = tangent-only (default)."""
    w_tool_gouge_warmup_frac: float = 0.0
    """fraction of iters before w_tool_gouge begins ramping (0 = on from start).
    The tool-position gouge barrier is spuriously active at correct TANGENT
    passes on convex parts (sphere), so a constant w_tool_gouge can pin the tool
    off the surface during the low-k exploration phase and stall carving before
    it establishes. With a warmup, carving is established first (residual falls,
    dice peaks), THEN w_tool_gouge ramps linearly from 0 to --w-tool-gouge over
    the remaining iters so the barrier suppresses gouge without pre-empting the
    carve. Pairs naturally with --k-anneal (barrier ramps in as k sharpens)."""

    # ---- Trajectory-quality measures (time / air-cut time / breakage) ----
    # Three deployable measures reported alongside dice and (when their weight
    # is > 0) added to the soft loss as differentiable surrogates. Hard,
    # non-differentiable final-metric forms are always reported in metrics.json
    # regardless of these weights. See docs/algorithms.md + docs/design.md.
    w_time: float = 1e-3
    """weight on the TOTAL TOOLPATH TIME soft term (sum of per-segment motion
    time at the feed/rapid regime speed, seconds). Shorter is better for equal
    dice. 0 disables. NOTE: time (~10s) and breakage (~[0,1]) live on very
    different scales, so the three w_* here are equal-by-default starting
    points -- tune per-run."""
    w_air_time: float = 1e-3
    """weight on the AIR-CUTTING TIME soft term (seconds spent cutting air,
    weighted by the per-segment air fraction). High retracts clear of the
    surface are free; surface-hugging air in empty corners is charged. 0
    disables."""
    w_break: float = 1e-3
    """weight on the TOOL-BREAKAGE PROBABILITY soft term (docs/algorithms.md
    §4.1/§4.2 stress-strength interference, simplified to a single threshold
    f_ref and log-variance sigma_risk). 0 disables. A too-risky toolpath should
    be rejected even if shorter; raising this (and best_w_break) enforces that."""
    kc: float = 700.0
    """specific cutting force (N/mm^2) for the breakage force model
    (Al ~600-800). mu_F = kc * V_chip_mm3 / (dt * D)."""
    f_ref: float = 50.0
    """nominal cutting force (N) at which per-step P_break = 0.5 (effective
    S_bar / alpha_mean). Calibrate from known-good/bad cuts; raw fcut_max is
    reported in metrics.json to aid this."""
    sigma_risk: float = 0.5
    """combined log-std of the stress-strength interference,
    sqrt(sigma_alpha^2 + pi^2/(6 m^2)). Sets the transition band of the
    breakage sigmoid."""
    f_max: float = 100.0
    """hard threshold force (N) for the docs/design.md broken flag: broken=1
    iff F_cut_max > f_max. Calibrate; raw fcut_max is reported."""
    best_w_airtime: float = 0.05
    """composite best-checkpoint weight on normalized air-cutting time. The
    best trajectory is selected by score = dice - best_w_airtime*air_time_norm
    - best_w_time*total_time_norm - best_w_break*break_prob_any, where the time
    metrics are normalized by T*dt (max possible time) into [0,1]. Small
    weights keep dice dominant; raising best_w_break enforces the reject-too-
    risky behavior."""
    best_w_time: float = 0.05
    """composite best-checkpoint weight on normalized total toolpath time."""
    best_w_break: float = 0.05
    """composite best-checkpoint weight on breakage probability (already
    [0,1]). Raising this rejects high-engagement checkpoints even at higher
    dice."""
    best_on_hard: bool = False
    """select the best/deployable checkpoint by HARD dice (the deployable sharp
    carve metric) instead of the default SOFT dice. The soft selector is less
    noisy but can deploy a checkpoint whose soft dice is high yet hard dice is
    much lower (the soft/hard gap) -- throwing away a higher-hard-dice final
    iter. Hard selection aligns deployment with the metric we advance on, at the
    cost of selecting on a nondeterministic carve (mitigated by the air/break
    penalties and by hard dice being stable at convergence). The reported
    `hard_dice` then reflects the hard-dice-best checkpoint."""

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
    stock_size_in: tuple[float, float, float] | None = None
    """stock box (x, y, z up) in inches -- the normalized cube [0,1]^3 (only
    this is voxelized). Default: 1 in cube for analytic targets; grid targets
    take the stock box from the target NPZ (the part's bounding box), keeping
    the STEP model's physical dimensions."""
    voxel_size_mm: float = 0.5
    """physical voxel edge in mm -- the sub-mm precision knob (overrides --resolution)"""
    workspace_in: tuple[float, float, float] = (16.0, 12.0, 10.0)
    """machine work volume (x, y, z up) in inches -- default Haas Mini Mill (toolhead limits)"""
    stock_origin_in: tuple[float, float, float] | None = None
    """work origin (G54) = stock top-centre in machine inches (export/validation only)"""

    # Units & speed limits (enforced by per-step clipping in the simulator)
    target_radius_mm: float = 11.43
    """target feature radius / half-size in mm (default 0.9 in diameter). Passed
    to the simulator to DEFINE the target; not read by the optimizer."""
    target_height_mm: float = 22.86
    """target feature height in mm (default 0.9 in). Passed to the simulator to
    DEFINE the target; not read by the optimizer."""
    target_sub_radius_mm: float = 9.525
    """sub-primitive radius in mm for combined-CSG targets (default 0.375 in =
    0.75 in diameter). Passed to the simulator to DEFINE the target; not read
    by the optimizer."""
    tool_radius_mm: float = 3.175
    """cutter radius in mm (default 1/4" end mill)"""
    tool_height_mm: float = 25.0
    """cutter flute length in mm"""
    tool_cut_height_mm: float = 0.0
    """height (mm) of the CUTTING TIP band over which the air-time metric/loss
    integrates swept/air/engage volumes (see airfrac-shank-volume-bias). 0 =
    use tool_radius_mm (one-radius tip band); the full tool_height shank is no
    longer counted as air when it sits in already-carved empty space. The carve
    itself is unaffected (still the full cylinder)."""
    dt: float = 0.45
    """seconds per simulator step; speed = |delta (.) envelope_mm| / dt. THE decisive
    lever: at low dt (0.12/0.01) the swept-cylinder tool is speed-limited -- its
    z-range clips to 0.72-1.0 and it cannot descend/traverse the exterior, capping dice
    at ~0.56 regardless of loss or capacity. dt=0.45 advances ~1 voxel/step so the tool
    covers the part. Sweet spot dt in [0.42,0.5]."""
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


def _safe_round(v, ndigits=6):
    """Round a metric to ndigits, coercing non-finite (nan/inf) to None so the
    summary JSON stays strictly parseable (autoresearch harnesses reject NaN)."""
    if v is None:
        return None
    f = float(v)
    return None if not np.isfinite(f) else round(f, ndigits)


def eval_metrics(sim, T, dx):
    """Dice/ASD/HD95 (shared `_metrics` path) + gouge/residual of the carved stock.

    Also reports dice_baseline (do-nothing: uncut stock vs target) and
    dice_improvement = (dice - baseline)/(1 - baseline), a difficulty-normalized
    accuracy score (0 = idle, 1 = perfect). stock[0] is the pristine uncut stock
    (forward writes stock[t+1], never stock[0]), so it is the right baseline for
    both soft and hard carves."""
    stock = sim.stock.to_numpy()[T - 1]
    target = sim.target.to_numpy()
    uncut = sim.stock.to_numpy()[0]
    m = _metrics(stock, target, dx, uncut)  # +dice_baseline/dice_improvement
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
    # Trajectory-quality measures (hard, final-metric form on the hard carve):
    # total toolpath time, air-cutting time, breakage probability + force, and
    # the docs/design.md broken flag. Computed by compute_traj_diagnostics_hard.
    sim.compute_traj_diagnostics_hard(T - 1)
    m["air_time"] = float(sim.diag_air_time[None])
    m["total_time"] = float(sim.diag_time[None])
    m["break_prob_any"] = float(sim.diag_break_prob_any[None])
    m["break_prob_max"] = float(sim.diag_break_prob_max[None])
    m["fcut_max"] = float(sim.diag_fcut_max[None])
    m["broken"] = float(sim.diag_broken[None])
    m["engage_max"] = float(sim.diag_engage_max[None])
    m["engage_mean"] = float(sim.diag_engage_mean[None])
    return m


def composite_score(m, args, T, dt):
    """Best-checkpoint composite: dice minus normalized trajectory penalties.

    score = dice - best_w_airtime*air_time_norm - best_w_time*total_time_norm
            - best_w_break*break_prob_any

    air_time and total_time are normalized by T*dt (the max possible toolpath
    time = every segment at the dt cap) into ~[0,1] so the weights are
    comparable; break_prob_any is already [0,1]. With small weights dice still
    dominates; the new metrics break ties toward shorter/safer paths, and a
    large best_w_break enforces the reject-too-risky behavior. m may be None
    (returns -inf).
    """
    if m is None:
        return -1e9
    t_cap = max(T * dt, 1e-8)
    air_norm = float(m.get("air_time", 0.0)) / t_cap
    time_norm = float(m.get("total_time", 0.0)) / t_cap
    brk = float(m.get("break_prob_any", 0.0))
    return (
        float(m["dice"])
        - args.best_w_airtime * air_norm
        - args.best_w_time * time_norm
        - args.best_w_break * brk
    )


def export_stls(sim, T, dx, run_dir, step, track):
    """Export initial stock / carved stock / target meshes (shared `_sdf_to_stl`)."""
    initial_stock = sim.stock.to_numpy()[0].copy()      # before the first cut
    carved_stock = sim.stock.to_numpy()[T - 1].copy()
    target = sim.target.to_numpy().copy()

    mesh_dir = os.path.join(run_dir, "meshes")
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


# ---------------------------------------------------------------------------
# Human-feedback warm-start (RLHF-style policy improvement).
#
# The web dashboard lets a user rate each run 1-7 stars + a free-text note;
# those ratings persist to autoresearch/tasks/train_csg/run_feedback.json (keyed
# by run basename). Every training run reads that store here so the human's
# qualitative judgments are (a) logged to stderr and (b) recorded in the new
# run's metrics.json under "feedback". With --use-feedback, the run additionally
# warm-starts its trajectory from the highest-rated prior run that matches this
# target_shape + max_steps (matching is in THIS selection layer — reading prior
# args.json metadata — so the optimizer/init/loss code stays shape-agnostic).
# ---------------------------------------------------------------------------
def _feedback_store_path():
    return os.path.join("autoresearch", "tasks", "train_csg", "run_feedback.json")


def _load_feedback_store():
    """Read the feedback store; returns {} if missing/corrupt (never raises)."""
    try:
        with open(_feedback_store_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def _find_run_dir_by_name(name):
    """Locate runs/<batch>/<name> by basename (newest on ties), or None."""
    import glob
    hits = [d for d in glob.glob(f"runs/**/{name}", recursive=True) if os.path.isdir(d)]
    if not hits:
        return None
    hits.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return hits[0]


def _read_run_args(run_dir):
    """Read a prior run's args.json, or None if unreadable."""
    p = os.path.join(run_dir, "args.json")
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def load_human_feedback(target_shape, max_steps):
    """Read the feedback store and find a warm-start trajectory.

    Returns (warmstart_deltas_or_None, summary). `summary` always carries the
    top-rated prior runs (for logging + metrics.json). `warmstart_deltas` is the
    (T-1, 3) float32 delta array from the highest-starred prior run whose
    target_shape + max_steps match, or None when no match / no saved trajectory.
    Deltas that don't exactly fit T-1 are truncated or last-padded (never
    rescaled — per-step displacements don't interpolate cleanly).
    """
    store = _load_feedback_store()
    # Rank rated entries by stars desc, then recency desc.
    ranked = sorted(
        [(k, v) for k, v in store.items() if v.get("stars")],
        key=lambda kv: (-int(kv[1]["stars"]), -float(kv[1].get("ts", 0.0))),
    )
    summary = {
        "top_rated": [
            {"run": k, "stars": int(v["stars"]), "feedback": str(v.get("feedback", ""))}
            for k, v in ranked[:8]
        ],
        "warmstart": None,
    }
    for k, v in ranked:
        # Warm-start only from above-average runs (5-7 stars on the 1-7 scale).
        if int(v["stars"]) < 5:
            break
        run_dir = _find_run_dir_by_name(k)
        if not run_dir:
            continue
        pa = _read_run_args(run_dir) or {}
        if pa.get("target_shape") != target_shape:
            continue
        try:
            if int(pa.get("max_steps", -1)) != int(max_steps):
                continue
        except (TypeError, ValueError):
            continue
        dp = os.path.join(run_dir, "trajectory_deltas.npy")
        if not os.path.exists(dp):
            continue
        try:
            arr = np.load(dp)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] != 3:
            continue
        arr = arr.astype(np.float32)
        need = int(max_steps) - 1
        if arr.shape[0] >= need:
            init = arr[:need].copy()
        else:
            pad = np.tile(arr[-1:], (need - arr.shape[0], 1))
            init = np.vstack([arr, pad]).astype(np.float32)
        summary["warmstart"] = {
            "run": k, "stars": int(v["stars"]),
            "feedback": str(v.get("feedback", "")),
            "shape": target_shape, "max_steps": int(max_steps),
        }
        return init, summary
    return None, summary


def main():
    args = tyro.cli(Args)

    T = args.max_steps
    dx = 1.0 / args.resolution
    prefix = "AR-" if args.autoresearch else ""
    ts = int(time.time() * 1000)
    run_name = f"{prefix}{args.env_id}__{args.exp_name}__{args.seed}__{ts}"
    run_dir = os.path.join("runs", args.runs_subdir, run_name) if args.runs_subdir else os.path.join("runs", run_name)
    # Atomically claim a unique run dir. The old check-then-create loop
    # (`while os.path.exists`) raced when two launches landed in the same
    # millisecond: both saw no dir, both proceeded, and os.makedirs(exist_ok=True)
    # silently let the loser overwrite the winner's artifacts. makedirs with
    # exist_ok=False is atomic on POSIX -- exactly one process wins; the other
    # gets FileExistsError and retries with an incremented timestamp.
    while True:
        try:
            os.makedirs(run_dir, exist_ok=False)
            break
        except FileExistsError:
            ts += 1
            run_name = f"{prefix}{args.env_id}__{args.exp_name}__{args.seed}__{ts}"
            run_dir = os.path.join("runs", args.runs_subdir, run_name) if args.runs_subdir else os.path.join("runs", run_name)
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    print(f"[run] writing outputs to {run_dir}")

    # Human feedback: always read the rating store, log top-rated prior runs,
    # and (with --use-feedback) warm-start from the best matching trajectory.
    fb_warmstart, fb_summary = load_human_feedback(args.target_shape, args.max_steps)
    if fb_summary["top_rated"]:
        print("[feedback] top-rated prior runs (human stars):", file=sys.stderr, flush=True)
        for t in fb_summary["top_rated"][:5]:
            print(f"  {t['stars']}★ {t['run']}: {t['feedback'] or '(no note)'}",
                  file=sys.stderr, flush=True)
    if fb_summary["warmstart"]:
        ws = fb_summary["warmstart"]
        if args.use_feedback:
            print(f"[feedback] --use-feedback: warm-starting from {ws['stars']}★ "
                  f"run {ws['run']} (shape={ws['shape']}, max_steps={ws['max_steps']})",
                  file=sys.stderr, flush=True)
        else:
            print(f"[feedback] {ws['stars']}★ warm-start available ({ws['run']}); "
                  f"pass --use-feedback to apply", file=sys.stderr, flush=True)

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
    # Grid targets: leave stock_size_in unset so the simulator takes the stock
    # box (and voxel size) from the target NPZ's physical dimensions.
    if args.stock_size_in is None and args.target_shape != "grid":
        args.stock_size_in = (1.0, 1.0, 1.0)
    sim = CSGSimulatorDelta(resolution=args.resolution, max_steps=T, k_init=args.k_init,
                            target_shape=args.target_shape, tool_start=(0.5, 0.5, 1.0),
                            stock_size_in=args.stock_size_in,
                            voxel_size_mm=args.voxel_size_mm,
                            work_volume_in=args.workspace_in,
                            stock_origin_in=args.stock_origin_in, dt=args.dt,
                            rapid_ipm=args.rapid_ipm, feed_ipm=args.feed_ipm,
                            safe_distance_in=args.safe_distance_in,
                            enforce_speed_limits=args.enforce_speed_limits,
                            target_sdf_path=args.target_sdf_path)
    # Grid targets define their own physical stock box (from the NPZ); reflect
    # the sim's actual box back into args so downstream consumers (z-floor,
    # structured inits, G-code export) use the real dimensions.
    if args.stock_size_in is None:
        args.stock_size_in = (sim.Lx / 25.4, sim.Ly / 25.4, sim.Lz / 25.4)
        print(f"[stock] from target NPZ: {sim.Lx:.2f} x {sim.Ly:.2f} x {sim.Lz:.2f} mm "
              f"(grid {sim.Nx} x {sim.Ny} x {sim.Nz} @ {sim.v:.4f} mm/voxel)", flush=True)
    sim.set_target_params(radius_mm=args.target_radius_mm,
                          height_mm=args.target_height_mm,
                          half_size_mm=args.target_radius_mm,
                          center=(0.5, 0.5, 0.5),
                          sub_radius_mm=args.target_sub_radius_mm)
    sim.tool_radius[None] = args.tool_radius_mm
    sim.tool_height[None] = args.tool_height_mm
    # Cutting-tip band for the air-time metric/loss (0 -> one tool radius).
    sim.tool_cut_height[None] = (
        args.tool_cut_height_mm if args.tool_cut_height_mm > 0.0 else args.tool_radius_mm
    )
    # Tool holder: 2.5 inch diameter cylinder above the cutter (mm; default).
    from cam.units import inch_to_mm
    sim.holder_radius[None] = inch_to_mm(2.5 / 2.0)
    # Loss balancing: objective (residual) vs. safety barriers (gouge, holder).
    sim.w_residual[None] = args.w_residual
    sim.w_gouge[None] = args.w_gouge
    sim.loss_shift[None] = args.loss_shift
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
    # Margin is in mm; the barrier works in voxels (target_sdf + r_vox are both
    # voxel units), so convert mm -> voxels via the voxel size v (mm/voxel).
    sim.tool_gouge_margin[None] = args.tool_gouge_margin_mm / sim.v
    # Trajectory-quality measures (time / air-cut time / breakage) + breakage
    # model constants.
    sim.w_time[None] = args.w_time
    sim.w_air_time[None] = args.w_air_time
    sim.w_break[None] = args.w_break
    sim.kc[None] = args.kc
    sim.f_ref[None] = args.f_ref
    sim.sigma_risk[None] = args.sigma_risk
    sim.f_max[None] = args.f_max
    sim.bake_target_grid()
    sim.set_target_volume()

    # --- Z-floor: clamp the executed tool BASE z so the holder (which rides
    # above the base by tool_height) cannot plunge into the remaining stock.
    # Floor = part_bottom_z - epsilon_mm/stock_z_mm, in normalized [0,1].
    # Shape-agnostic part bottom z: query the baked target SDF grid directly
    stock_z_mm = float(args.stock_size_in[2]) * inch_to_mm(1.0)
    sdf_grid = sim.target.to_numpy()
    solid_voxels = np.where(sdf_grid <= 0.0)[2]
    if len(solid_voxels) > 0:
        part_bottom_z = float(solid_voxels.min()) / float(sdf_grid.shape[2])
    else:
        part_bottom_z = 0.0
    z_floor = part_bottom_z - args.z_floor_epsilon_mm / stock_z_mm
    sim.z_floor[None] = float(z_floor)
    sim.enforce_z_floor[None] = 1 if args.enforce_z_floor else 0
    print(f"[z-floor] part_bottom_z={part_bottom_z:.4f} epsilon={args.z_floor_epsilon_mm}mm "
          f"-> floor={z_floor:.4f} (enforced={bool(sim.enforce_z_floor[None])})", flush=True)

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

    # --- Sweep method: B-spline control points over the one-shot swept carve ---
    # The optimizable parameters are the control points P[1:] (P[0] pinned to
    # the tool start so the sampled path X = B @ P begins at the canonical
    # start). The swept-volume loss lives in simulator/sweep.py; the torch side
    # holds only the basis matmul and the path regularizers, so grad_P is one
    # matmul away from the Taichi path gradient.
    sweep = None
    if args.method == "sweep":
        # Best-checkpoint selection MUST compare on hard dice for the sweep
        # method: its soft (sequential k=10) dice is identically ~0, so the
        # composite score would otherwise reduce to "smallest trajectory
        # penalties" and select an early useless path (observed: iter-50
        # hard-dice 0.28 chosen over final 0.97).
        if not args.best_on_hard:
            args.best_on_hard = True
            print("[sweep] best-checkpoint selection forced to hard dice "
                  "(--best-on-hard): sweep soft dice is a dead signal", flush=True)
        from simulator.sweep import (SweepCarve, bspline_basis,
                                     init_reference_path, fit_control_points)
        from cam.units import ipm_to_mm_per_s
        sweep = SweepCarve(sim, n_points=T)
        sweep.w_broad[None] = args.w_broad
        sweep.sigma_broad[None] = args.sigma_broad
        # Physical-plausibility terms (idea.md jul13-phys-plausible). The
        # fragility field is computed from the target SDF ALONE (thin-feature
        # detection + cantilever strength — shape-agnostic geometry) and is
        # always built: the hard diagnostics report fragile margins even when
        # the soft penalties are off.
        sweep.w_force[None] = args.w_force
        sweep.w_fragile[None] = args.w_fragile
        # Effective feed-force coefficient (N per mm^2 of engagement cross-
        # section): F = kc*MRR/v_c = [kc*feed/v_c] * (chip_vol/len). The
        # specific cutting energy kc (N/mm^2 = J/mm^3) is spent at the CUTTING
        # speed v_c (spindle surface speed), so the force felt along the feed
        # is scaled down by feed/v_c. Sanity: full slot (a_p 6 x a_e 6.35 mm)
        # at kc 700, feed 4.23 mm/s, 5000 rpm -> ~68 N (realistic; the raw
        # chip-area form reads 27 kN).
        _feed_mm_s_setup = float(ipm_to_mm_per_s(args.feed_ipm))
        _v_cut = np.pi * 2.0 * float(sim.tool_radius[None]) * args.spindle_rpm / 60.0
        _kc_eff = args.kc * _feed_mm_s_setup / max(_v_cut, 1e-6)
        sweep.kc[None] = _kc_eff
        sweep.f_cap[None] = args.f_max
        from utils.fragility import compute_fragility, F_ALLOW_SAFE
        frag = compute_fragility(
            sim.target.to_numpy(), sim.v, sigma_y_mpa=args.sigma_y,
            r_thin_mm=(args.frag_thin_mm if args.frag_thin_mm > 0 else None),
            contact_mm=args.frag_contact_mm,
            tool_radius_mm=float(sim.tool_radius[None]))
        sweep.set_fragility(frag["f_allow_vox"])
        if frag["features"]:
            worst = min(frag["features"], key=lambda f: f["f_allow_n"])
            print(f"[fragility] {len(frag['features'])} fragile feature(s) "
                  f"(sigma_y {args.sigma_y:.0f} MPa); weakest: t={worst['t_mm']:.2f} mm "
                  f"h={worst['h_mm']:.2f} mm F_allow={worst['f_allow_n']:.1f} N",
                  flush=True)
        else:
            print("[fragility] no fragile features detected in the target",
                  flush=True)
        if args.reach_gate:
            from utils.reachability import compute_reachable_mask
            _tgt = sim.target.to_numpy()
            _reach = compute_reachable_mask(_tgt, sim.tool_radius[None] / sim.v)
            _part = _tgt <= 0.0
            _unreach = int(((~_part) & (~_reach)).sum())
            _ceiling = 2.0 * _part.sum() / (2.0 * _part.sum() + _unreach)
            sweep.reach.from_numpy(_reach.astype(np.float32))
            print(f"[sweep] reach gate ON: {_unreach} unreachable waste voxels "
                  f"masked from residual/attraction; dice ceiling {_ceiling:.4f}",
                  flush=True)
        B_np = bspline_basis(args.n_ctrl, T)
        _cap_budget_mm = (T - 1) * float(ipm_to_mm_per_s(args.feed_ipm)) * args.dt
        X_ref = init_reference_path(
            sim, tool_start, T, mode=args.sweep_init, seed=args.seed,
            max_len_mm=_cap_budget_mm,
            ramp_deg=(args.ramp_deg if args.sweep_init_ramp else 0.0))
        P_init = fit_control_points(B_np, X_ref)
        P_init[0] = tool_start
        B_t = torch.from_numpy(B_np)                       # (T, K)
        P0_const = torch.tensor(tool_start, dtype=torch.float32).unsqueeze(0)
        L_mm_t = torch.tensor([sim.Lx, sim.Ly, sim.Lz], dtype=torch.float32)
        feed_mm_s = float(ipm_to_mm_per_s(args.feed_ipm))
        z_floor_t = torch.tensor(float(z_floor), dtype=torch.float32)
        params = torch.tensor(P_init[1:], requires_grad=True)  # (K-1, 3)
        opt = torch.optim.Adam([params], lr=args.learning_rate)
        print(f"[sweep] K={args.n_ctrl} control points, T={T} samples, "
              f"init={args.sweep_init}, feed cap {feed_mm_s * args.dt:.2f} mm/step",
              flush=True)
        # Feed-feasibility of the init: the evaluator clips every step to the
        # cap, so an init whose length exceeds (T-1)*cap can only be executed
        # truncated — coverage (and dice) die silently. Surface it up front.
        _cap_mm = feed_mm_s * args.dt
        _ramp_tan = float(np.tan(np.radians(args.ramp_deg)))
        _step_mm = np.linalg.norm(np.diff(X_ref.astype(np.float64), axis=0)
                                  * np.array([sim.Lx, sim.Ly, sim.Lz]), axis=1)
        _len = float(_step_mm.sum())
        print(f"[sweep] init path {_len:.0f} mm vs executable budget "
              f"{(T - 1) * _cap_mm:.0f} mm (ratio {_len / ((T - 1) * _cap_mm):.2f}); "
              f"max step {_step_mm.max() / _cap_mm:.1f}x cap; "
              f"feasible at T >= {int(np.ceil(_len / _cap_mm)) + 1}", flush=True)

        def sweep_path():
            """Sampled path X (T,3) as a torch tensor differentiable in params."""
            return B_t @ torch.cat([P0_const, params], dim=0)

        def sweep_load_sim(X_det):
            """Push the sampled path into the sim (deltas + canonical start)."""
            full = torch.zeros((T, 3), dtype=torch.float32)
            full[:T - 1] = X_det[1:] - X_det[:-1]
            sim.tool_start[None] = ti.Vector([float(X_det[0, 0]),
                                              float(X_det[0, 1]),
                                              float(X_det[0, 2])])
            sim.tool_delta.from_torch(full)

    if args.method == "sweep":
        pass  # control points built above; the delta init chain is skipped
    elif args.init_mode == "raster_fine":
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
        # Descending helix that orbits JUST OUTSIDE the target surface. The
        # orbit radius at each z is the target's cross-section radius at that z
        # (read shape-agnostically from the baked SDF grid) + tool radius +
        # margin, so the tool's inner edge clears the exterior annulus without
        # gouging the part. Deriving the surface from the grid (not from a shape
        # name / shape-specific surface formula) generalizes to any target,
        # including combined-CSG and unseen shapes.
        n = T - 1
        stock_mm = args.stock_size_in[0] * 25.4
        r_tool = args.tool_radius_mm / stock_mm
        margin = 0.02
        revs = 8.0
        z_top, z_bot = 0.95, 0.05
        r_cross = target_cross_section_radii(sim)
        positions = np.zeros((n, 3), dtype=np.float32)
        for t in range(n):
            frac = t / max(1, n - 1)
            z = z_top + (z_bot - z_top) * frac
            r_orbit = _r_cross_at(r_cross, z) + r_tool + margin
            phase = 2.0 * np.pi * revs * frac
            positions[t, 0] = 0.5 + r_orbit * math.cos(phase)
            positions[t, 1] = 0.5 + r_orbit * math.sin(phase)
            positions[t, 2] = z
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    elif args.init_mode == "zlayer":
        # Z-level finishing descent: the tool is a tall vertical cylinder whose
        # tool_pos.z is its BASE, extending upward by h. Descending the base
        # from above the stock down past the bottom means each layer's tool only
        # reaches DOWN to its base, so a high base never touches the equator and
        # can safely carve the top interior exterior at small radius. The orbit
        # radius oscillates from a surface-offset safe radius (target
        # cross-section at the equator-closest z the tool reaches, read shape-
        # agnostically from the baked SDF grid + r_tool + margin) out to the
        # cube wall, sweeping the waste annulus at every z (a real CNC z-level
        # finishing pattern). Generalizes to any target -- no shape names.
        n = T - 1
        stock_mm = args.stock_size_in[0] * 25.4
        r_tool = args.tool_radius_mm / stock_mm
        margin = args.zlayer_margin
        revs = args.zlayer_revs
        osc = args.zlayer_osc
        z_top, z_bot = 0.95, -0.95
        # Crash-safe floor: the tool BASE must stay above z_floor so the holder
        # (riding above the base by tool_height ~= stock height) cannot plunge
        # into the remaining stock -- a real machine crash the new collision
        # handling enforces. The deep plunge to z_bot=-0.95 was the old zlayer's
        # mechanism for carving the lower exterior (it put the tool's z-range
        # below the equator); it is now forbidden, so stop the descent at the
        # floor. (Mirrors the sim z_floor computed at line ~567.) The radius
        # scheduling below already uses the equator radius for base in
        # [z_floor, 0.5] (z_eq clamps to 0.5), so the lower region carves the
        # equator annulus [r_sphere_equator, 0.5] -- crash-free but it cannot
        # reach the lower-interior waste (below equator, inside r_equator),
        # which is the new crash-safe ceiling.
        stock_z_mm = args.stock_size_in[2] * 25.4
        # Shape-agnostic part-bottom z: lowest solid voxel in the baked target
        # grid (correct for any target, including combined-CSG shapes -- mirrors
        # the sim z_floor computed in main()).
        _sdf_grid = sim.target.to_numpy()
        _solid = np.where(_sdf_grid <= 0.0)[2]
        _part_bottom_z = (float(_solid.min()) / float(_sdf_grid.shape[2])
                          if len(_solid) else 0.0)
        _z_floor = _part_bottom_z - args.z_floor_epsilon_mm / stock_z_mm
        # Also keep the holder (bottom = base + tool_height) >= 1mm above the
        # stock top so the trunc stage never trims trailing low-base steps (the
        # holder is wide -- 2.5in -- so within 1mm of the stock it is flagged).
        # base + tool_h_norm >= 1 + clearance_norm  =>  base >= 1 + clr - tool_h.
        _tool_h_norm = args.tool_height_mm / stock_z_mm
        _clr_norm = 1.0 / stock_z_mm  # matches --collision-clearance-mm default
        _z_holder_clear = 1.0 + _clr_norm - _tool_h_norm + 0.01
        z_bot = max(z_bot, _z_floor + 0.005, _z_holder_clear)
        r_outer = 0.5 + r_tool
        r_cross = target_cross_section_radii(sim)
        positions = np.zeros((n, 3), dtype=np.float32)
        for t in range(n):
            frac = t / max(1, n - 1)
            zb = z_top + (z_bot - z_top) * frac          # base descends through stock
            # equator-closest in-stock z the tool reaches (tool spans [zb, zb+1])
            zhi = zb + 1.0
            if zb > 0.5:
                z_eq = zb
            elif zhi < 0.5:
                z_eq = zhi
            else:
                z_eq = 0.5
            r_safe = _r_cross_at(r_cross, z_eq) + r_tool + margin
            r_orbit = r_safe + (r_outer - r_safe) * (0.5 + 0.5 * math.sin(2.0 * np.pi * osc * frac))
            phase = 2.0 * np.pi * revs * frac
            positions[t, 0] = 0.5 + r_orbit * math.cos(phase)
            positions[t, 1] = 0.5 + r_orbit * math.sin(phase)
            positions[t, 2] = zb
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    elif args.init_mode == "multidepth":
        # Shape-agnostic MULTI-DEPTH surface-following helical roughing. Builds
        # a continuous 3D toolpath from the baked target SDF grid ONLY (no shape
        # names / params). A single helix simultaneously (a) descends through
        # the target's full z-extent (multi-depth), (b) rotates to cover the
        # annulus angularly, and (c) sweeps the radius back and forth across the
        # waste annulus -- from the cube wall (r_outer = 0.5 + r_tool, tangent so
        # corner waste is removed) inward to the target surface offset by the
        # tool radius (r_safe(z) = target cross-section radius at z + r_tool +
        # margin, read shape-agnostically from target_cross_section_radii). The
        # tool center stays >= r_tool outside the target surface at every point,
        # so it removes the bulk EXTERIOR waste (the residual a surface-hugging
        # shell leaves behind) WITHOUT gouging -- a regular, generalizable CNC
        # helical-z-level roughing pattern. No retracts (continuous descent) ->
        # low air time.
        #
        # The simulator's per-step speed clip is a HARD wall: in T-1 steps the
        # tool can traverse at most (T-1)*feed_cap arc-units, so any commanded
        # path longer than that has its tail never reached (the discrete
        # per-level-spiral version of this init hit exactly that -- 42 units of
        # arc for a 9.5-unit budget, so only the top of the part was carved).
        # We therefore (1) build a CONTINUOUS helix (interleave z/angle/radius
        # so coverage is simultaneous, not sequential per level) and (2) resample
        # the path at constant arc length to EXACTLY n points spanning the whole
        # path, so every step is <= feed_cap and the entire z-extent is reached.
        n = T - 1
        stock_mm_x = args.stock_size_in[0] * 25.4
        r_tool = args.tool_radius_mm / stock_mm_x
        margin = args.multidepth_margin
        # Feed speed cap in normalized units per step (the clip-survival budget).
        # feed_ipm [in/min] * dt [s] / 60 [s/min] / stock_size_in[0] [in/unit].
        feed_cap = args.feed_ipm * args.dt / 60.0 / args.stock_size_in[0]
        # Target z-extent from the baked grid (shape-agnostic): lowest/highest
        # solid voxel, padded by one voxel and clamped inside the stock.
        _sdf_grid = sim.target.to_numpy()
        Nz_grid = float(_sdf_grid.shape[2])
        _solid = np.where(_sdf_grid <= 0.0)[2]
        if len(_solid):
            z_top = float(_solid.max()) / Nz_grid + 1.0 / Nz_grid
            z_bot = float(_solid.min()) / Nz_grid
        else:
            z_top, z_bot = 0.95, 0.05
        z_top = min(z_top + 2.0 * r_tool, 0.98)
        z_bot = max(z_bot, 0.02)
        r_cross = target_cross_section_radii(sim)
        r_outer = 0.5 + r_tool  # tool tangent to cube wall removes corner waste

        def r_safe_at(z):
            return _r_cross_at(r_cross, z) + r_tool + margin

        # Continuous helix over u in [0,1]: z descends, angle rotates (revs
        # turns), and radius sweeps the annulus via a triangle wave (radial_cycles
        # in->out passes) clamped to [r_safe(z), r_outer].
        radial_cycles = max(1.0, float(args.multidepth_levels))
        u = np.linspace(0.0, 1.0, 4000)
        z = z_top + (z_bot - z_top) * u
        r_safe_u = np.clip(np.array([r_safe_at(zz) for zz in z]), 0.0, r_outer)
        budget = max(1.0, (n - 1)) * feed_cap  # max arc the tool can traverse

        def build_helix(revs):
            theta = 2.0 * np.pi * revs * u
            ph = (u * radial_cycles) % 1.0
            tri = np.abs(1.0 - 2.0 * ph)            # 0 at walls -> 1 mid
            r = r_outer + (r_safe_u - r_outer) * tri
            pts = np.stack([0.5 + r * np.cos(theta),
                            0.5 + r * np.sin(theta), z], axis=1
                           ).astype(np.float32)
            seg = np.diff(pts, axis=0)
            seglen = np.sqrt((seg ** 2).sum(axis=1))
            cum = np.concatenate([[0.0], np.cumsum(seglen)])
            return pts, cum, float(cum[-1])

        # Adaptively shrink revs so the helix total arc fits the speed-clip
        # budget (arc ~ linear in revs, dominant term r*2*pi*revs). This keeps
        # the FULL z descent reachable in n steps (no tail dropped) while every
        # step stays <= feed_cap. One refinement suffices.
        revs = max(0.5, args.multidepth_revs)
        _, _, total = build_helix(revs)
        if total > 0.95 * budget:
            revs = max(0.5, revs * (0.95 * budget) / total)
            revs = max(0.5, revs * (0.95 * budget) / build_helix(revs)[2])
        pts, cum, total = build_helix(revs)
        # Resample the (now budget-fitting) full path at constant arc length to
        # EXACTLY n points: every step <= feed_cap, full z-extent reached.
        s_targets = np.linspace(0.0, total, n)
        idx = np.clip(np.searchsorted(cum, s_targets) - 1, 0, len(pts) - 2)
        frac = ((s_targets - cum[idx]) /
                np.maximum(cum[idx + 1] - cum[idx], 1e-9))
        positions = (pts[idx] + (pts[idx + 1] - pts[idx]) * frac[:, None]
                     ).astype(np.float32)
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    elif args.init_mode == "multidepth_cavity":
        # Shape-agnostic MULTI-DEPTH helical roughing WITH an interior-cavity
        # pass. Identical to `multidepth` for targets with NO interior cavity
        # (solid sphere/box/cyl/pyramid/bowl): the cavity detection finds none,
        # so the path is the plain exterior helix -- zero regression on the 5
        # working shapes. For targets with an interior concave cavity (e.g. a
        # through-hole), the outer-envelope r_safe(z)=r_cross(z)+r_tool+margin
        # keeps the tool OUTSIDE the cavity, so plain multidepth can never clear
        # the cavity's waste (this is the hole root cause; see idea.md Wave 5).
        # This mode detects the cavity shape-agnostically -- target SDF>0 voxels
        # INSIDE the outer envelope r_cross -- and appends a retract -> rapid ->
        # plunge -> small-radius spiral through the cavity centroid so the
        # interior waste is reached. Reads only the baked target SDF grid; no
        # shape names / shape params.
        n = T - 1
        stock_mm_x = args.stock_size_in[0] * 25.4
        r_tool = args.tool_radius_mm / stock_mm_x
        margin = args.multidepth_margin
        feed_cap = args.feed_ipm * args.dt / 60.0 / args.stock_size_in[0]
        grid = sim.target.to_numpy()
        Nx, Ny, Nz = grid.shape
        _solid = np.where(grid <= 0.0)[2]
        z_top = (float(_solid.max()) / Nz + 1.0 / Nz) if len(_solid) else 0.95
        z_bot = (float(_solid.min()) / Nz) if len(_solid) else 0.05
        z_top = min(z_top + 2.0 * r_tool, 0.98)
        z_bot = max(z_bot, 0.02)
        r_cross = target_cross_section_radii(sim)
        r_outer = 0.5 + r_tool

        def r_safe_at(z):
            return _r_cross_at(r_cross, z) + r_tool + margin

        budget = max(1.0, (n - 1)) * feed_cap
        radial_cycles = max(1.0, float(args.multidepth_levels))

        def helix(revs, z_arr, r_low, r_high, cx, cy):
            # radius sweeps [r_low, r_high] via a triangle wave; (cx,cy) center.
            m = max(8, len(z_arr))
            u = np.linspace(0.0, 1.0, m)
            theta = 2.0 * np.pi * revs * u
            tri = np.abs(1.0 - 2.0 * ((u * radial_cycles) % 1.0))
            r = r_low + (r_high - r_low) * tri
            pts = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta),
                            z_arr], axis=1).astype(np.float32)
            seg = np.diff(pts, axis=0)
            cum = np.concatenate([[0.0], np.cumsum(np.sqrt((seg ** 2).sum(1)))])
            return pts, cum, float(cum[-1])

        def resample(pts, cum, npts):
            s = np.linspace(0.0, cum[-1], npts)
            idx = np.clip(np.searchsorted(cum, s) - 1, 0, len(pts) - 2)
            frac = (s - cum[idx]) / np.maximum(cum[idx + 1] - cum[idx], 1e-9)
            return (pts[idx] + (pts[idx + 1] - pts[idx]) * frac[:, None]
                    ).astype(np.float32)

        # --- exterior helix (same construction as multidepth) ---
        u1 = np.linspace(0.0, 1.0, 4000)
        z1 = z_top + (z_bot - z_top) * u1
        rs1 = np.clip(np.array([r_safe_at(zz) for zz in z1]), 0.0, r_outer)
        c1 = np.full_like(z1, 0.5)
        revs = max(0.5, args.multidepth_revs)
        _, _, te = helix(revs, z1, rs1, r_outer, c1, c1)
        if te > 0.95 * budget:
            revs = max(0.5, revs * (0.95 * budget) / te)
            revs = max(0.5, revs * (0.95 * budget) / helix(revs, z1, rs1, r_outer, c1, c1)[2])
        pts_e, cum_e, te = helix(revs, z1, rs1, r_outer, c1, c1)

        # --- interior cavity detection (shape-agnostic, from the SDF grid) ---
        xs = (np.arange(Nx) + 0.5) / Nx - 0.5
        ys = (np.arange(Ny) + 0.5) / Ny - 0.5
        Rxy = np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2)         # (Nx,Ny) from center
        rc_z = np.clip(r_cross, 0.0, None)                          # (Nz,) outer envelope
        empty_in = (grid > 0.0) & (Rxy[:, :, None] < rc_z[None, None, :])
        cav_R = np.where(empty_in, Rxy[:, :, None], -1.0).max(axis=(0, 1))
        r_cav_z = np.clip(cav_R, 0.0, None)                          # cavity outer radius/z
        has_cav = r_cav_z > r_tool + 1e-3                            # tool fits in cavity

        if has_cav.any():
            cx_v = (np.arange(Nx) + 0.5) / Nx
            cy_v = (np.arange(Ny) + 0.5) / Ny
            sx = np.where(empty_in, cx_v[:, None, None], 0.0).sum(axis=(0, 1))
            sy = np.where(empty_in, cy_v[None, :, None], 0.0).sum(axis=(0, 1))
            cn = empty_in.sum(axis=(0, 1))
            cxz = np.where(cn > 0, sx / np.maximum(cn, 1), 0.5)
            cyz = np.where(cn > 0, sy / np.maximum(cn, 1), 0.5)
            zk = (np.arange(Nz) + 0.5) / Nz
            kc = np.where(has_cav)[0]
            z_ct = min(z_top, kc.max() / Nz + 1.0 / Nz)
            z_cb = max(z_bot, kc.min() / Nz)
            u2 = np.linspace(0.0, 1.0, 2000)
            z2 = z_ct + (z_cb - z_ct) * u2
            cx2 = np.interp(z2, zk, cxz)
            cy2 = np.interp(z2, zk, cyz)
            # interior spiral radius stays INSIDE the cavity (<= r_cav - r_tool),
            # so the tool clears cavity waste without gouging the solid ring.
            r_in2 = np.clip(np.interp(z2, zk, r_cav_z) - r_tool - margin, 0.0, None)
            z_ret = 1.0 + 2.0 * r_tool                              # just above stock top
            # INTERIOR-FIRST ordering. tool_start [0.5,0.5,1.0] is already
            # centered above the cavity mouth, so the path opens with an axial
            # PLUNGE down the channel center, then the interior spiral, then a
            # retract, then the exterior skin. The prior exterior-first ordering
            # stranded the tool: the exterior orbits at r_safe=r_cross+r_tool+
            # margin which, for a near-filling sphere (r_cross up to 0.45, tool
            # 0.125), lies OUTSIDE the stock box (half-width 0.5) -> rapid air-
            # cut, so 40+ steps vanished and the interior spiral -- the whole
            # point of this mode -- never executed (commanded into the channel
            # at steps 93-127 but the tool was still lagging at r~1.0). Interior-
            # first guarantees the channel is carved before budget expires.
            #
            # tool_start is prepended to allpts so the plunge is resampled into
            # ~feed_cap-sized steps (a single 0.25-mag plunge delta would be
            # speed-clipped to 0.075 and the z-lag would accumulate, never
            # reaching the cavity). positions[0]==tool_start -> delta[0]=0.
            # HELICAL plunge from tool_start z=1.0 down to the cavity mouth
            # z2[0], circling at radius r_in2[0] (tool edge r_in2[0]+r_tool <<
            # hole radius, so it clears the cavity mouth on the way down). This
            # MUST be helical, not a straight axial plunge: tool_sdf's capsule
            # projection uses h_param = pa.ba/(ba.ba+1e-12); an axial segment has
            # ba_xy=[0,0] so the gradient ~pa/1e-12 overflows the autodiff to NaN
            # (simulator code, not modifiable). Circling keeps every segment's XY
            # displacement non-zero so ba.ba stays finite. positions[0] differs
            # from tool_start (small XY offset) so delta[0] is non-zero too.
            # Built before the budget-fit loop so the loop accounts for the real
            # (longer, circling) plunge arc, not a straight-line underestimate.
            r_plunge = float(r_in2[0]) if float(r_in2[0]) > 1e-4 else 0.5 * r_tool
            n_pz = max(12, int(np.ceil(abs(float(z2[0]) - 1.0) / feed_cap)) * 4)
            uz = np.linspace(0.0, 1.0, n_pz)
            zp = (1.0 + (float(z2[0]) - 1.0) * uz).astype(np.float32)
            plunge, _, plunge_arc = helix(1.5, zp,
                                          np.full_like(zp, r_plunge),
                                          np.full_like(zp, r_plunge),
                                          float(cx2[0]), float(cy2[0]))
            revs_i = max(2.0, revs * 0.6)
            scale = 1.0
            for _ in range(4):
                pi, _, ti_s = helix(revs_i * scale, z2, np.zeros_like(z2), r_in2, cx2, cy2)
                pe, _, te_s = helix(revs * scale, z1, rs1, r_outer, c1, c1)
                p_end = pi[-1]
                ext0 = pe[0]
                ret = np.stack([p_end,
                                [p_end[0], p_end[1], z_ret],
                                [ext0[0], ext0[1], z_ret],
                                ext0], axis=0).astype(np.float32)
                rseg = np.diff(ret, axis=0)
                tr = float(np.sqrt((rseg ** 2).sum(1)).sum())
                total = plunge_arc + ti_s + tr + te_s
                if total <= 0.95 * budget:
                    break
                scale *= (0.95 * budget) / total
            pts_i, _, _ = helix(revs_i * scale, z2, np.zeros_like(z2), r_in2, cx2, cy2)
            pts_e, _, _ = helix(revs * scale, z1, rs1, r_outer, c1, c1)
            p_end = pts_i[-1]
            ext0 = pts_e[0]
            ret = np.stack([p_end,
                            [p_end[0], p_end[1], z_ret],
                            [ext0[0], ext0[1], z_ret],
                            ext0], axis=0).astype(np.float32)
            allpts = np.concatenate([plunge, pts_i, ret[1:], pts_e], axis=0)
            seg = np.diff(allpts, axis=0)
            cum = np.concatenate([[0.0], np.cumsum(np.sqrt((seg ** 2).sum(1)))])
            positions = resample(allpts, cum, n)
        else:
            # No interior cavity -> identical to plain multidepth.
            positions = resample(pts_e, cum_e, n)
        init = np.empty((n, 3), dtype=np.float32)
        init[0] = positions[0] - tool_start
        init[1:] = np.diff(positions, axis=0)
    else:
        init = np.random.uniform(-args.init_scale, args.init_scale, size=(T - 1, 3)).astype(np.float32)
    # Human-feedback warm-start: with --use-feedback, replace the heuristic init
    # with the highest-rated prior run's learned deltas (matched on shape +
    # max_steps). The optimizer then refines a human-approved trajectory instead
    # of starting from the heuristic — the concrete "feedback improves policy"
    # loop. init is (T-1, 3) per-step displacements; the warm-start array is
    # already sized to T-1 by load_human_feedback. Delta-method only: the sweep
    # method builds its own control-point params/optimizer on the sweep path.
    if args.use_feedback and fb_warmstart is not None:
        init = fb_warmstart.astype(np.float32)
        print(f"[feedback] warm-start deltas applied: shape={init.shape} "
              f"from {fb_summary['warmstart']['run']}", file=sys.stderr, flush=True)
    if args.method != "sweep":
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
    best_score = -1e9
    best_positions = None
    best_deltas = None
    best_ctrl = None
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

            # k-anneal: ramp smoothness k from k_init to k_final over training
            # (low k = smooth exploration early; high k = hard-tracking late).
            # Independent of the lr block above (the two can combine).
            if args.k_anneal:
                sim.k[None] = args.k_init + (args.k_final - args.k_init) * (
                    it / max(1, args.iters)
                )

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
            # w_tool_gouge warmup: keep w_tool_gouge at 0 until warmup_frac of
            # iters, then ramp linearly to args.w_tool_gouge over the remaining
            # iters. The tool-position barrier is spuriously active at tangent
            # passes on convex parts, so a constant w_tool_gouge can pin the tool
            # off-surface during low-k exploration and stall carving; ramping it
            # in after carving establishes avoids that stall.
            if args.w_tool_gouge > 0.0 and args.w_tool_gouge_warmup_frac > 0.0:
                warm_start = int(args.iters * args.w_tool_gouge_warmup_frac)
                if it < warm_start:
                    sim.w_tool_gouge[None] = 0.0
                else:
                    span = max(1, args.iters - warm_start)
                    sim.w_tool_gouge[None] = args.w_tool_gouge * ((it - warm_start) / span)

            t0 = 0
            if args.method == "sweep":
                # Swept-volume forward/backward: sample the spline, regularize
                # the path in torch (feed cap + z-floor barriers), get the
                # swept-carve loss gradient from Taichi, and pull both back to
                # the control points through the basis matrix.
                if params.grad is not None:
                    params.grad.zero_()
                X = sweep_path()                            # (T, 3), diff in params
                step_mm = (X[1:] - X[:-1]) * L_mm_t
                speed = step_mm.norm(dim=1) / args.dt        # mm/s per step
                # Dimensionless excess (fraction of the cap) so w_feed trades
                # off against the O(0.1-1) geometry loss on a sane scale.
                feed_pen = torch.relu(speed / feed_mm_s - 1.0).pow(2).mean()
                zfloor_pen = torch.relu(z_floor_t - X[:, 2]).pow(2).mean()
                reg_loss = args.w_feed * feed_pen + 100.0 * zfloor_pen
                if args.w_ramp > 0.0:
                    # Plunge penalty: engaged steps (chip attribution from the
                    # last argmin refresh, detached gate) may not descend
                    # steeper than ramp_deg. Excess is in mm, normalized by
                    # the per-step feed cap so the scale is dimensionless.
                    engaged_t = torch.from_numpy(
                        (sweep.seg_chip_np > 1e-9).astype(np.float32))
                    dz_mm = step_mm[:, 2]
                    dxy_mm = step_mm[:, :2].norm(dim=1)
                    ramp_ex = torch.relu(-dz_mm - _ramp_tan * dxy_mm) / _cap_mm
                    reg_loss = reg_loss + args.w_ramp * (engaged_t * ramp_ex).pow(2).mean()
                reg_loss.backward()
                X_det = X.detach()
                _refresh = args.amin_refresh <= 1 or it % args.amin_refresh == 0
                ti_loss, grad_X = sweep.loss_and_grad(X_det.numpy(),
                                                      refresh_argmin=_refresh,
                                                      want_chip=args.w_ramp > 0.0)
                grad = params.grad + (B_t.T @ torch.from_numpy(grad_X))[1:]
                loss = ti_loss + float(reg_loss)
                grad_norm = float(grad.norm().item())
                sim.loss[None] = loss  # keep the summary's loss field honest
                # Keep the sim loaded with the current path so the eval below
                # (forward_hard from the sampled start) scores this iterate.
                sweep_load_sim(X_det)
            else:
                # Push current displacements into Taichi, then forward+backward.
                # With restart_from_state, each iteration either starts fresh (optionally
                # from a random tool_start) or restores a saved mid-cut state and carves
                # only the tail [t0, T). The restored prefix is a detached constant, so
                # only the carved tail receives gradient this iteration; Adam leaves the
                # prefix untouched (standard stochastic per-slice optimization).
                sim.tool_delta.from_torch(params.detach())
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
                # (Not for sweep: params are control points, not deltas, and the
                # sim was already loaded with the sampled path this iteration.)
                if args.method != "sweep" and (args.random_tool_start or t0 != 0):
                    sim.tool_start[None] = ti.Vector(list(CANONICAL_TOOL_START))
                    sim.tool_delta.from_torch(params.detach())
                    sim.loss[None] = 0.0
                # SOFT carve eval: the proven 0.85/0.92/0.89/0.92 operating-point
                # metric. forward() uses smooth_max (differentiable), so soft
                # dice climbs smoothly with soft optimization. This is the
                # primary "dice" used for best_score / best-checkpoint selection
                # (matches the jul1 baseline that the task spec ceilings refer
                # to).
                sim.forward(T)
                soft_stock = sim.stock.to_numpy()[T - 1]
                soft_target = sim.target.to_numpy()
                soft_uncut = sim.stock.to_numpy()[0]
                soft_m = _metrics(soft_stock, soft_target, dx, soft_uncut)
                # HARD carve eval: deployable measures. forward_hard() uses
                # boolean ti.max (non-differentiable) with tool_sdf_sharp.
                # Hard dice is quantized to 1-voxel steps (flat for many iters
                # with small tool moves) and reports the actual carved result.
                # compute_traj_diagnostics_hard (air_time/total_time/
                # break_prob_any) also runs on this hard carve.
                sim.forward_hard(T)
                m = eval_metrics(sim, T, dx)
                m["soft_dice"] = float(soft_m["dice"])
                m["soft_asd"] = float(soft_m["asd"])
                m["soft_hd95"] = float(soft_m["hd95"])
                m["soft_dice_baseline"] = float(soft_m.get("dice_baseline", float("nan")))
                m["soft_dice_improvement"] = float(soft_m.get("dice_improvement", float("nan")))
                last_m = m
                # best_score uses SOFT dice (proven operating-point metric) with
                # the hard-carve traj-quality penalties (best_w_* default 0.05,
                # so penalties barely affect checkpoint selection; with
                # --best-w-* 0 the composite is pure soft dice). With
                # --best-on-hard, select on HARD dice (m["dice"]) instead so the
                # deployed checkpoint is the hard-dice-best, aligning with the
                # deployable metric we advance on.
                sel_m = m if args.best_on_hard else {**m, "dice": m["soft_dice"]}
                score = composite_score(sel_m, args, T, args.dt)
                if score > best_score:
                    # sim.tool_delta / sim.tool_pos here reflect the PRE-step
                    # params (set before the forward at the top of the loop),
                    # which are exactly what produced this dice. Snapshot them
                    # directly so the saved best trajectory is consistent with
                    # the measured metrics (no re-eval needed later).
                    best_score = score
                    best_dice = float(sel_m["dice"])
                    best_m = dict(m)
                    best_positions = sim.tool_pos.to_torch()[:T].numpy().copy()
                    best_deltas = sim.tool_delta.to_torch()[:T].numpy().copy()
                    if args.method == "sweep":
                        best_ctrl = torch.cat([P0_const, params.detach()],
                                              dim=0).numpy().copy()
                    best_it = it
                writer.add_scalar("eval/dice", m["soft_dice"], it)
                writer.add_scalar("eval/hard_dice", m["dice"], it)
                writer.add_scalar("eval/asd", m["soft_asd"], it)
                writer.add_scalar("eval/hd95", m["soft_hd95"], it)
                writer.add_scalar("eval/dice_baseline", m.get("dice_baseline", float("nan")), it)
                writer.add_scalar("eval/dice_improvement", m.get("dice_improvement", float("nan")), it)
                writer.add_scalar("eval/soft_dice_improvement", m["soft_dice_improvement"], it)
                writer.add_scalar("metrics/gouge", m["gouge"], it)
                writer.add_scalar("metrics/residual", m["residual"], it)
                writer.add_scalar("metrics/holder_overlap", m["holder_overlap"], it)
                writer.add_scalar("loss/residual", m["loss_residual"], it)
                writer.add_scalar("loss/gouge", m["loss_gouge"], it)
                writer.add_scalar("loss/holder", m["loss_holder"], it)
                writer.add_scalar("loss/air", m["loss_air"], it)
                writer.add_scalar("loss/jerk", m["loss_jerk"], it)
                # Trajectory-quality measures (reported alongside dice).
                writer.add_scalar("metrics/air_time", m["air_time"], it)
                writer.add_scalar("metrics/total_time", m["total_time"], it)
                writer.add_scalar("metrics/break_prob_any", m["break_prob_any"], it)
                writer.add_scalar("metrics/break_prob_max", m["break_prob_max"], it)
                writer.add_scalar("metrics/fcut_max", m["fcut_max"], it)
                writer.add_scalar("metrics/broken", m["broken"], it)
                writer.add_scalar("charts/best_score", best_score, it)
                last_eval_iter = it
                if args.progress_bar:
                    pbar.set_postfix(loss=f"{loss:.4f}", dice=f"{m['soft_dice']:.3f}",
                                     hd=f"{m['dice']:.3f}",
                                     resid=f"{m['loss_residual']:.3f}",
                                     gouge=f"{m['loss_gouge']:.3f}",
                                     hold=f"{m['loss_holder']:.2e}",
                                     air=f"{m['loss_air']:.3f}",
                                     t=f"{m['total_time']:.2f}",
                                     at=f"{m['air_time']:.2f}",
                                     brk=f"{m['break_prob_any']:.2f}",
                                     jerk=f"{m['loss_jerk']:.2e}")
            elif args.progress_bar:
                pbar.set_postfix(loss=f"{loss:.4f}", grad=f"{grad_norm:.2e}")

            if not args.progress_bar and (it % args.log_freq == 0 or it == args.iters - 1):
                lr_val = opt.param_groups[0]["lr"]
                time_str = time.strftime("[%Y-%m-%d %H:%M:%S] ")
                if last_m is not None:
                    # soft_dice is the proven operating-point metric (jul1
                    # baseline); hard_dice is the deployable sharp carve.
                    _sd = last_m.get("soft_dice", last_m["dice"])
                    _hd = last_m["dice"]
                    line = (f"{time_str}[iter {it:4d}/{args.iters}] loss: {loss:.4f} | grad: {grad_norm:.2e} | lr: {lr_val:.2e} | "
                            f"dice: {_sd:.4f} | hdice: {_hd:.4f} | asd: {last_m.get('soft_asd', last_m['asd']):.2f} | hd95: {last_m.get('soft_hd95', last_m['hd95']):.2f} | "
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
        if args.method == "sweep":
            X_final = sweep_path().detach()
            deltas = (X_final[1:] - X_final[:-1]).numpy()
            sweep_load_sim(X_final)
            ctrl_points = torch.cat([P0_const, params.detach()], dim=0).numpy()
        else:
            deltas = params.detach().numpy()
            sim.tool_delta.from_torch(params.detach())
            ctrl_points = None
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
            # composite_score reads m["dice"]; by default substitute soft_dice so
            # the best-vs-final comparison uses the proven SOFT-dice metric. With
            # --best-on-hard, compare on HARD dice (last_m["dice"]) instead so a
            # higher-hard-dice final iter isn't rejected in favor of a soft-best
            # checkpoint that deploys worse.
            final_sel_m = last_m if args.best_on_hard else {
                **last_m, "dice": last_m.get("soft_dice", last_m["dice"])}
            final_iter_score = composite_score(final_sel_m, args, T, args.dt)
            # Use the best checkpoint when its COMPOSITE score (dice + the three
            # trajectory-quality penalties) beats the final iter's composite --
            # generalizes the old dice-only comparison. The reported metrics are
            # the already-measured best_m (no re-eval: the hard carve is
            # nondeterministic under GPU atomic-adds).
            if best_score > final_iter_score:
                positions = best_positions
                deltas = best_deltas
                if best_ctrl is not None:
                    ctrl_points = best_ctrl
                last_m = best_m
                # Re-load the best trajectory into the sim so STL export and
                # holder-overlap reflect the best model (a fresh carve -- visually
                # equivalent; the reported dice is the already-measured best_m).
                sim.tool_delta.from_torch(torch.as_tensor(best_deltas))
                sim.loss[None] = 0.0
                sim.forward_hard(T)
                used_best = True
                print(f"[best] using best-composite checkpoint: dice={best_dice:.6f} "
                      f"score={best_score:.6f} @ iter {best_it} "
                      f"(final-iter soft-dice={float(final_iter_m.get('soft_dice', final_iter_m['dice'])):.6f} "
                      f"hard-dice={float(final_iter_m['dice']):.6f} "
                      f"score={final_iter_score:.6f})", flush=True)

        final_overlap = float(sim.holder_overlap_total(T - 1))
        if final_overlap > 0.0:
            print(f"[holder] WARNING: final trajectory still collides the holder "
                  f"with the stock (overlap volume {final_overlap:.3e}).")
        else:
            print("[holder] final trajectory keeps the holder clear of the stock.")

        # --- Physical-plausibility hard diagnostics (sweep; idea.md
        # jul13-phys-plausible). Computed on the REPORTED trajectory (the sim
        # holds the best-checkpoint carve if it was selected above). Chip
        # attribution is the delta sim's sequential hard engagement
        # (compute_traj_diagnostics_hard scratch), force is the mechanistic
        # chip-area model F = kc * chip_mm3 / len_mm — same definitions as the
        # soft penalties, so penalty-off runs report honest violation levels.
        phys_diag = {}
        if args.method == "sweep":
            sim.compute_traj_diagnostics_hard(T - 1)
            _grid_mm3 = sim.Lx * sim.Ly * sim.Lz
            _chip = sim.seg_engage.to_numpy()[:T - 1].astype(np.float64) * _grid_mm3
            _pos = positions.astype(np.float64)
            _smm = np.diff(_pos, axis=0) * np.array([sim.Lx, sim.Ly, sim.Lz])
            _len = np.linalg.norm(_smm, axis=1)
            _F = _kc_eff * _chip / np.maximum(_len, 0.1)
            _engaged = _chip > 0.5 * sim.v ** 3
            _dz, _dxy = _smm[:, 2], np.linalg.norm(_smm[:, :2], axis=1)
            # 5% slope + 0.02 mm headroom: ramp legs sit exactly AT ramp_deg,
            # and spline fit / f32 roundoff must not flag the boundary.
            _plunge = _engaged & (-_dz > 1.05 * _ramp_tan * _dxy + 0.02)
            _dims = np.array([sim.Nx, sim.Ny, sim.Nz])
            _mid = np.clip((0.5 * (_pos[1:] + _pos[:-1]) * _dims).astype(int),
                           0, _dims - 1)
            _cap = frag["f_allow_tool"][_mid[:, 0], _mid[:, 1], _mid[:, 2]]
            _viol = _engaged & (_F > _cap)
            _margin = float(np.min(_cap[_engaged] / np.maximum(_F[_engaged], 1e-6))
                            ) if _engaged.any() else float(F_ALLOW_SAFE)
            phys_diag = {
                # max sequential-attribution cutting force (N, spindle-
                # normalized kc*MRR/v_c) and the hard tool-break flag at the
                # f_max threshold. Nathan's fcut_max (vol/(dt*D) form, delta
                # attribution) is reported separately -- both are surrogates,
                # this one is calibrated to physical Newtons.
                "fcut_seq_max": round(float(_F.max()) if len(_F) else 0.0, 3),
                "tool_broken_seq": float(bool((_F > args.f_max).any())),
                # engaged steps descending steeper than ramp_deg (end mills
                # cannot drill): count and fraction of engaged steps.
                "plunge_count": int(_plunge.sum()),
                "plunge_frac": round(float(_plunge.sum() / max(_engaged.sum(), 1)), 6),
                # part-side: worst allowable/applied force ratio near fragile
                # features (<1 means a feature would snap) and the break flag.
                "fragile_margin_min": round(min(_margin, F_ALLOW_SAFE), 3),
                "part_broken": float(bool(_viol.any())),
                "n_fragile_features": len(frag["features"]),
            }
            print(f"[phys] F_seq max {phys_diag['fcut_seq_max']:.1f} N "
                  f"(cap {args.f_max:.0f}); plunges {phys_diag['plunge_count']} "
                  f"({100 * phys_diag['plunge_frac']:.1f}% of engaged); "
                  f"fragile margin {phys_diag['fragile_margin_min']:.2f} "
                  f"({phys_diag['n_fragile_features']} feature(s)); "
                  f"part_broken={int(phys_diag['part_broken'])}", flush=True)

        # Save summary metrics for automated agents and LLM harnesses.
        import json
        total_seconds = time.time() - start_time
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        final_dice = float(last_m.get("soft_dice", last_m["dice"])) if last_m is not None else 0.0
        final_hard_dice = float(last_m["dice"]) if last_m is not None else 0.0
        final_asd = float(last_m.get("soft_asd", last_m["asd"])) if last_m is not None else 0.0
        final_hd95 = float(last_m.get("soft_hd95", last_m["hd95"])) if last_m is not None else 0.0
        final_gouge = float(last_m["gouge"]) if last_m is not None else 0.0
        final_resid = float(last_m["residual"]) if last_m is not None else 0.0
        final_hold = float(last_m["holder_overlap"]) if last_m is not None else final_overlap

        summary_data = {
            # hard_dice is the FINAL/DEPLOYABLE metric (sharp boolean carve) and
            # is printed first so log-grepping agents read it as the headline.
            # "dice" below it is the soft (differentiable, sigmoid-blurred) dice
            # — an inflated proxy that can mask runs which don't boolean-cut.
            "hard_dice": round(final_hard_dice, 6),
            "dice": round(final_dice, 6),
            "asd": round(final_asd, 6),
            "hd95": round(final_hd95, 6),
            # Do-nothing baseline (uncut stock vs target) and difficulty-normalized
            # dice improvement = (dice - baseline)/(1 - baseline): 0 = idle, 1 =
            # perfect. Both reported for the soft (operating-point) and hard
            # (deployable) dice; nan when the part fills the stock (no headroom).
            "dice_baseline": _safe_round(last_m.get("dice_baseline")) if last_m else 0.0,
            "dice_improvement": _safe_round(last_m.get("dice_improvement")) if last_m else 0.0,
            "soft_dice_baseline": _safe_round(last_m.get("soft_dice_baseline")) if last_m else 0.0,
            "soft_dice_improvement": _safe_round(last_m.get("soft_dice_improvement")) if last_m else 0.0,
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
            # Trajectory-quality measures (hard, final-metric form on the hard
            # carve) for the selected (best-composite) checkpoint. Reported
            # alongside dice so the autoresearch harness can gate/compose on
            # them. air_time/total_time are seconds; break_prob_* are [0,1];
            # fcut_max is Newtons; broken is the docs/design.md hard flag;
            # engage_* are raw chip volumes (unit-cube^3) for calibration.
            "air_time": round(float(last_m.get("air_time", 0.0)), 6) if last_m else 0.0,
            "total_time": round(float(last_m.get("total_time", 0.0)), 6) if last_m else 0.0,
            # Sharp air-time fraction = air_time / total_time in [0,1]. This is
            # the deployable air-cut metric (fraction of toolpath TIME in air on
            # the hard carve). Distinct from air_cut_fraction, a SOFT blurred
            # volume ratio that does NOT track this and can read ~0.09 while
            # air_time==total_time (i.e. the whole path is air). Use this one.
            "air_time_frac": (round(float(last_m["air_time"]) / max(float(last_m["total_time"]), 1e-8), 6)
                              if last_m and last_m.get("total_time", 0.0) > 0 else 0.0),
            "break_prob_any": round(float(last_m.get("break_prob_any", 0.0)), 6) if last_m else 0.0,
            "break_prob_max": round(float(last_m.get("break_prob_max", 0.0)), 6) if last_m else 0.0,
            "fcut_max": round(float(last_m.get("fcut_max", 0.0)), 6) if last_m else 0.0,
            "broken": round(float(last_m.get("broken", 0.0)), 6) if last_m else 0.0,
            "engage_max": round(float(last_m.get("engage_max", 0.0)), 6) if last_m else 0.0,
            "engage_mean": round(float(last_m.get("engage_mean", 0.0)), 6) if last_m else 0.0,
            "best_score": round(best_score, 6),
            # Physical-plausibility diagnostics (sweep method; empty-safe for
            # delta runs). See the [phys] log line / idea.md jul13-phys-plausible.
            **phys_diag,
            # Human feedback that informed this run. Recorded on EVERY run
            # (even without --use-feedback) so the dashboard/metrics log which
            # prior runs a human rated and whether a warm-start was available.
            # warmstart is None when no >=5★ prior run matched this
            # target_shape+max_steps. Shape-agnostic: matching lives only in the
            # feedback selection layer (load_human_feedback), never in
            # optimizer/init/loss.
            "feedback_used": bool(args.use_feedback and fb_warmstart is not None),
            "feedback_top_rated": fb_summary.get("top_rated", []) if fb_summary else [],
            "feedback_warmstart": fb_summary.get("warmstart") if fb_summary else None,
            # Final-iter (pre-best-checkpoint) trajectory metrics: exposes any
            # late-training polish (e.g. w_prox warmup) on the final trajectory,
            # independent of where the best-dice peak occurred.
            "final_iter_dice": round(float(final_iter_m.get("soft_dice", final_iter_m["dice"])), 6) if final_iter_m else 0.0,
            "final_iter_hard_dice": round(float(final_iter_m["dice"]), 6) if final_iter_m else 0.0,
            "final_iter_air_cut_fraction": round(float(final_iter_m.get("air_cut_fraction", 0.0)), 6) if final_iter_m else 0.0,
            "final_iter_air_time": round(float(final_iter_m.get("air_time", 0.0)), 6) if final_iter_m else 0.0,
            "final_iter_total_time": round(float(final_iter_m.get("total_time", 0.0)), 6) if final_iter_m else 0.0,
            "final_iter_air_time_frac": (round(float(final_iter_m["air_time"]) / max(float(final_iter_m["total_time"]), 1e-8), 6)
                                         if final_iter_m and final_iter_m.get("total_time", 0.0) > 0 else 0.0),
            "final_iter_break_prob_any": round(float(final_iter_m.get("break_prob_any", 0.0)), 6) if final_iter_m else 0.0,
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
        export_stls(sim, T, dx, run_dir, it, args.track)

        # --- Save the learned trajectory (this is GradMill's "model") ---
        if args.save_model:
            np.save(os.path.join(run_dir, "trajectory_deltas.npy"), deltas)
            np.save(os.path.join(run_dir, "trajectory.npy"), positions)
            # Sweep method: also save the B-spline control polygon (K,3) that
            # generated the planned path (X = B @ P), consistent with whichever
            # checkpoint (best or final) the saved trajectory came from. The
            # results web viewer overlays these on the 3D trajectory plot.
            if ctrl_points is not None:
                np.save(os.path.join(run_dir, "control_points.npy"), ctrl_points)
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
            if args.method != "sweep":  # sweep: sim already holds the final path
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
