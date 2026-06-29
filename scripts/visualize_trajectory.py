"""Visualize a trained diff-cam trajectory and reconcile it with the G-code.

A trained run saves ``trajectory.npy`` (the speed-clipped tool path in the
normalized stock box ``[0,1]^3``) plus an ``args.json`` describing the stock
box, work origin, tool and target. This tool loads them and renders a 6-panel
figure that makes the G-code-vs-simulation relationship legible:

  A. Normalized frame -- what the simulator sees (path + stock box + target).
  B. WCS / G-code frame -- what the machine runs (the same path mapped to the
     work coordinate system; the affine transform of panel A).
  C. G-code round-trip -- the original path vs. the executed path recovered
     from the exported G-code, with rapid approach/retract moves drawn faded so
     the cutting moves are separable from the post-processor overhead.
  D. Sim carve vs target -- the hard-CSG result of running the trajectory,
     next to the target part, so "did it machine the part?" is visible.
  E. G-code vs sim carve -- the carve of the G-code program overlaid on the
     carve of the original trajectory (plus the target). The console reports
     their voxel Dice; ~1.0 proves the G-code reproduces the simulation.
  F. Metrics -- a text summary of the round-trip, Z ranges, cut depth, and the
     G-code-vs-sim and carved-vs-target Dice.

Why a dedicated tool: the export -> parse round trip is geometrically exact
(verified near machine precision), so a perceived G-code/sim mismatch usually
comes from the frame/Z convention, post-processor approach moves, or an
under-trained trajectory -- not from the export math. The G-code-vs-sim carve
Dice (panel E / console) is the definitive check: it has been 1.00000 in every
run tested, confirming the export is faithful. This figure surfaces each
candidate cause at a glance, and the console report flags them.

Examples
--------
    # Auto-match the repo-root trajectory.npy + args.json
    uv run python scripts/visualize_trajectory.py

    # Visualize a specific run (reads runs/<run>/trajectory.npy + args.json)
    uv run python scripts/visualize_trajectory.py --run runs/AR-CamEnvDiff-v0__train_csg__1__1782623511

    # Haas post (shows the approach/plunge/retract moves); skip the carve panel
    uv run python scripts/visualize_trajectory.py --post haas --no-carve
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

from cam import (  # noqa: E402
    MachineConfig,
    trajectory_to_gcode,
    parse_gcode,
    segment_waypoints,
    gcode_to_trajectory,
    plan_trajectory,
    waypoint_roundtrip_error,
)


# ---------------------------------------------------------------------------
# Config resolution (mirrors scripts/export_gcode.py)
# ---------------------------------------------------------------------------
def _load_run_config(trajectory_path, run_path, disabled):
    """Return (config_dict, args_path) from a training run's args.json.

    Prefers an explicit --run dir, else args.json next to the trajectory. Returns
    ({}, None) when disabled or not found.
    """
    if disabled:
        return {}, None
    if run_path is not None:
        path = os.path.join(run_path, "args.json")
    else:
        cand = os.path.join(os.path.dirname(os.path.abspath(trajectory_path)), "args.json")
        path = cand if os.path.exists(cand) else None
    if not path or not os.path.exists(path):
        return {}, None
    try:
        with open(path) as f:
            return json.load(f), path
    except (OSError, ValueError) as e:
        print(f"[config] WARNING: could not read run config {path}: {e}")
        return {}, None


def _resolve_paths(args, repo):
    """Resolve the trajectory path (and deltas) from --run or --trajectory."""
    if args.run is not None:
        run_dir = args.run if os.path.isabs(args.run) else os.path.join(repo, args.run)
        traj_path = os.path.join(run_dir, "trajectory.npy")
    else:
        traj_path = args.trajectory
        if not os.path.isabs(traj_path):
            traj_path = os.path.join(repo, args.trajectory)
    if not os.path.exists(traj_path):
        raise SystemExit(f"trajectory not found: {traj_path}")
    # deltas live next to the trajectory when --save_model was used.
    delta_path = os.path.join(os.path.dirname(traj_path), "trajectory_deltas.npy")
    if not os.path.exists(delta_path):
        delta_path = None
    return traj_path, delta_path


def _pick(cli_val, cfg_val, default):
    """CLI value if given, else the run config's value, else the default."""
    if cli_val is not None:
        return tuple(cli_val)
    if cfg_val is not None:
        return tuple(cfg_val)
    return default


# ---------------------------------------------------------------------------
# Carved-stock + target meshes (reuse cam.sim_exec._HardCarveSimulator)
# ---------------------------------------------------------------------------
def _mesh_from_sdf(grid, spacing, pad_value):
    """Marching-cubes the zero level set of an SDF grid -> (verts, faces).

    ``spacing`` is the per-axis physical step (so vertices land in [0,1]^3 for a
    normalized grid). Pads with ``pad_value`` (outside) so boundary-touching
    surfaces (e.g. the uncarved stock block) still close.
    """
    from skimage.measure import marching_cubes

    grid = np.asarray(grid, dtype=np.float32)
    padded = np.pad(grid, 1, mode="constant", constant_values=pad_value)
    verts, faces, _, _ = marching_cubes(padded, level=0.0, spacing=spacing)
    verts -= np.asarray(spacing)  # undo the 1-cell pad offset
    return verts, faces


def _carve(positions, cfg_dict, cfg: MachineConfig):
    """Hard-carve ``positions`` and return (stock_grid, target_grid, sim).

    Builds a ``_HardCarveSimulator`` with geometry from ``cfg_dict`` (the run's
    args.json) so the carved-stock scale matches the G-code scale. A fresh
    simulator is constructed per call (Taichi is re-initialised), so callers
    comparing two trajectories must carve one, copy the result to NumPy, then
    carve the other.
    """
    from cam.sim_exec import _HardCarveSimulator

    positions = np.asarray(positions, dtype=np.float32)
    if len(positions) < 2:
        raise ValueError("need >=2 positions to carve")
    deltas = np.diff(positions, axis=0)

    stock_size_in = cfg_dict.get("stock_size_in", (1.0, 1.0, 1.0))
    voxel_size_mm = cfg_dict.get("voxel_size_mm", 0.5) or 0.5
    work_volume_in = cfg_dict.get("workspace_in", (16.0, 12.0, 10.0))
    target_shape = cfg_dict.get("target_shape", "sphere")
    tool_radius = cfg_dict.get("tool_radius_mm", 3.175)
    tool_height = cfg_dict.get("tool_height_mm", 25.0)
    target_radius = cfg_dict.get("target_radius_mm", 11.43)
    target_height = cfg_dict.get("target_height_mm", 22.86)
    stock_origin_in = cfg_dict.get("stock_origin_in")

    sim = _HardCarveSimulator(
        resolution=32,
        max_steps=len(positions) - 1,
        target_shape=target_shape,
        tool_start=tuple(float(v) for v in positions[0]),
        stock_size_in=tuple(stock_size_in),
        voxel_size_mm=voxel_size_mm,
        work_volume_in=tuple(work_volume_in),
        stock_origin_in=tuple(stock_origin_in) if stock_origin_in is not None else None,
    )
    sim.tool_radius[None] = tool_radius
    sim.tool_height[None] = tool_height
    sim.set_target_params(
        radius_mm=target_radius,
        height_mm=target_height,
        half_size_mm=target_radius,
        center=(0.5, 0.5, 0.5),
    )
    sim.bake_target_grid()
    sim.set_target_volume()

    padded = np.zeros((sim.max_steps, 3), dtype=np.float32)
    padded[: len(deltas)] = deltas
    sim.tool_delta.from_numpy(padded)
    sim.forward_hard(len(positions))

    stock = sim.stock.to_numpy()[len(positions) - 1].copy()
    target = sim.target.to_numpy().copy()
    return stock, target, sim


def _dice(a_mask, b_mask):
    """Voxel Dice between two boolean masks (2*|A∩B| / (|A|+|B|); 1.0 = identical)."""
    inter = float(np.logical_and(a_mask, b_mask).sum())
    denom = float(a_mask.sum() + b_mask.sum())
    return 2.0 * inter / denom if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _box_edges(lo, hi):
    """Return the 12 edges of an axis-aligned box as a list of (start, end)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    pts = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ])
    idx = [(0, 1), (1, 2), (2, 3), (3, 0),
           (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(pts[i], pts[j]) for i, j in idx]


def _draw_box(ax, lo, hi, **kw):
    for s, e in _box_edges(lo, hi):
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], **kw)


def _style_axes(ax, title, xlabel="X", ylabel="Y", zlabel="Z"):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _set_lims(ax, lo, hi, pad=0.05):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    rng = hi - lo
    lo = lo - rng * pad
    hi = hi + rng * pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--trajectory", default=os.path.join(repo, "trajectory.npy"),
                     help="input (T,3) trajectory .npy (ignored when --run is given)")
    src.add_argument("--run", default=None,
                     help="training run dir (loads runs/<run>/trajectory.npy + args.json)")
    ap.add_argument("--post", default="rs274", choices=("rs274", "haas"),
                    help="post-processor for the round-trip panel")
    ap.add_argument("--no-carve", action="store_true",
                    help="skip the carved-stock panel (avoids Taichi)")
    ap.add_argument("--save", default=None,
                    help="write the figure PNG here (default next to the trajectory)")
    ap.add_argument("--show", action="store_true",
                    help="open the interactive matplotlib window")
    ap.add_argument("--no-run-config", action="store_true",
                    help="ignore args.json; use CLI flags / defaults only")
    # Geometry overrides (else from args.json, else defaults).
    ap.add_argument("--stock-size-in", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="stock box in inches (overrides run config)")
    ap.add_argument("--stock-origin-in", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="G54 top-centre in machine inches")
    ap.add_argument("--workspace-in", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="machine work volume in inches")
    args = ap.parse_args()

    if not args.show and args.save is None:
        # Default to saving when not interactive so headless runs produce output.
        pass

    traj_path, delta_path = _resolve_paths(args, repo)
    positions = np.load(traj_path).astype(np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise SystemExit(f"{traj_path} must hold an (T,3) array; got {positions.shape}")

    run_cfg, cfg_path = _load_run_config(traj_path, args.run, args.no_run_config)
    if cfg_path:
        print(f"[config] matching training run config from {cfg_path}")
    elif not args.no_run_config:
        print("[config] no args.json found; using CLI flags / defaults")

    stock_size_in = _pick(args.stock_size_in, run_cfg.get("stock_size_in"), (1.0, 1.0, 1.0))
    stock_origin_in = run_cfg.get("stock_origin_in") if args.stock_origin_in is None else tuple(args.stock_origin_in)
    workspace_in = _pick(args.workspace_in, run_cfg.get("workspace_in"), (16.0, 12.0, 10.0))

    mc = MachineConfig(
        stock_size_in=stock_size_in,
        stock_origin_in=tuple(stock_origin_in) if stock_origin_in is not None else None,
        workspace_in=workspace_in,
    )
    target_shape = run_cfg.get("target_shape", "sphere")
    target_radius = run_cfg.get("target_radius_mm", 11.43)
    target_height = run_cfg.get("target_height_mm", 22.86)
    tool_radius = run_cfg.get("tool_radius_mm", 3.175)
    tool_height = run_cfg.get("tool_height_mm", 25.0)
    voxel_size_mm = run_cfg.get("voxel_size_mm", 0.5) or 0.5

    print(f"[config] trajectory: {traj_path}  ({positions.shape[0]} points)")
    print(f"[config] stock_size_in={tuple(stock_size_in)}  "
          f"stock_origin_in={tuple(stock_origin_in) if stock_origin_in else None}  "
          f"workspace_in={tuple(workspace_in)}")
    print(f"[config] target_shape={target_shape} radius_mm={target_radius} "
          f"height_mm={target_height}  tool r/h={tool_radius}/{tool_height} mm  "
          f"voxel={voxel_size_mm} mm")

    # --- Trajectory geometry in normalized + WCS frames ---
    z_norm = (float(positions[:, 2].min()), float(positions[:, 2].max()))
    wcs = mc.to_wcs(positions)
    z_wcs = (float(wcs[:, 2].min()), float(wcs[:, 2].max()))
    print(f"[traj] normalized Z range: [{z_norm[0]:.4f}, {z_norm[1]:.4f}]  "
          f"(1.0 = stock top)")
    print(f"[traj] WCS Z range:        [{z_wcs[0]:.4f}, {z_wcs[1]:.4f}] mm  "
          f"(0 = stock top, negative = into stock)")

    # Flag an under-trained trajectory that never reaches the target depth.
    target_depth_mm = float(target_height) if target_shape in ("cylinder", "pyramid") else 2.0 * float(target_radius)
    cut_depth = -z_wcs[0]  # deepest point below the stock top, mm
    if cut_depth < target_depth_mm * 0.5:
        print(f"[flag] trajectory only cuts {cut_depth:.3f} mm below the stock top; "
              f"target needs ~{target_depth_mm:.2f} mm -- the path may be under-trained "
              f"and won't resemble the target.")

    # --- G-code round-trip ---
    gcode = trajectory_to_gcode(positions, mc, post=args.post)
    segments = parse_gcode(gcode, mc)
    recovered = segment_waypoints(segments)
    executed, times = gcode_to_trajectory(gcode, mc)
    n_orig = len(positions)
    n_wp = len(recovered)
    n_extra = n_wp - n_orig
    if n_orig == n_wp:
        rt_err_mm = float(waypoint_roundtrip_error(positions, recovered, float(np.mean(mc.stock_size_vec))))
    else:
        rt_err_mm = float("nan")
    print(f"[gcode] post={args.post}: {n_orig} original -> {n_wp} waypoints "
          f"({n_extra:+d} approach/retract moves)")
    if n_orig == n_wp:
        print(f"[gcode] waypoint round-trip error: {rt_err_mm:.3e} mm")
    else:
        print(f"[gcode] waypoint round-trip error: n/a (post adds approach moves)")
    print(f"[gcode] executed trajectory: {executed.shape[0]} samples, "
          f"{times[-1]:.2f}s of motion")

    # --- G-code-vs-sim carve agreement (definitive mismatch check) ---
    # The export->parse round trip is geometrically exact, so the only honest
    # test of "does the G-code match the simulation?" is to carve the G-code
    # program's waypoints and compare the resulting stock to the original
    # trajectory's carve. We carve `recovered` (the parsed waypoints, which for
    # the Haas post include the approach/plunge/retract moves at safe Z) -- those
    # overhead moves sit above the stock and carve nothing, so a Dice ~1.0 means
    # the G-code reproduces the simulation exactly.
    carve_metrics = {}
    if not args.no_carve:
        print("[carve] hard-carving original trajectory ...")
        try:
            stock_orig, target_grid, sim_orig = _carve(positions, run_cfg, mc)
            carved_orig_mask = stock_orig < 0
            target_mask = target_grid < 0
            # Carve the G-code program (its parsed waypoints).
            print(f"[carve] hard-carving G-code program ({n_wp} waypoints) ...")
            stock_gcode, _, _ = _carve(recovered, run_cfg, mc)
            carved_gcode_mask = stock_gcode < 0
            dice_gcode = _dice(carved_orig_mask, carved_gcode_mask)
            dice_target = _dice(carved_orig_mask, target_mask)
            spacing = (1.0 / sim_orig.Nx, 1.0 / sim_orig.Ny, 1.0 / sim_orig.Nz)
            carve_metrics = {
                "dice_gcode": dice_gcode,
                "dice_target": dice_target,
                "vox_carved": int(carved_orig_mask.sum()),
                "vox_target": int(target_mask.sum()),
                "vox_gcode": int(carved_gcode_mask.sum()),
                "grid": (sim_orig.Nx, sim_orig.Ny, sim_orig.Nz),
                # Reused by panels D/E so we carve exactly once per trajectory.
                "stock_orig": stock_orig,
                "stock_gcode": stock_gcode,
                "target_grid": target_grid,
                "spacing": spacing,
                "v": float(sim_orig.v),
            }
            print(f"[carve] G-code vs sim carve Dice = {dice_gcode:.5f} "
                  f"(1.0 = G-code reproduces the simulation exactly)")
            print(f"[carve] carved-vs-target Dice    = {dice_target:.5f} "
                  f"(1.0 = the trajectory machines the target part)")
            print(f"[carve] carved voxels: sim={carve_metrics['vox_carved']} "
                  f"gcode={carve_metrics['vox_gcode']} target={carve_metrics['vox_target']} "
                  f"({sim_orig.Nx}x{sim_orig.Ny}x{sim_orig.Nz})")
        except Exception as e:
            print(f"[carve] gcode-vs-sim comparison failed: {e}")

    # --- Figure ---
    matplotlib.use("Agg") if not args.show else None
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

    fig = plt.figure(figsize=(17, 12))
    fig.suptitle(f"diff-cam trajectory viz  ({os.path.basename(traj_path)})  "
                 f"stock={tuple(stock_size_in)} in  post={args.post}", fontsize=12)

    # Panel A: normalized frame.
    axA = fig.add_subplot(2, 3, 1, projection="3d")
    sc = np.linspace(0, 1, len(positions))
    axA.plot(positions[:, 0], positions[:, 1], positions[:, 2],
             color="C0", lw=2, label="saved (clipped) path")
    axA.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=sc,
                cmap="viridis", s=20, zorder=5)
    axA.scatter(*positions[0], color="green", s=60, marker="o", label="start", zorder=6)
    axA.scatter(*positions[-1], color="red", s=60, marker="s", label="end", zorder=6)
    if delta_path is not None:
        deltas = np.load(delta_path).astype(np.float64)
        commanded = np.cumsum(np.vstack([positions[0], deltas]), axis=0)
        axA.plot(commanded[:, 0], commanded[:, 1], commanded[:, 2],
                 color="C1", lw=1, ls="--", label="commanded (pre-clip)")
    _draw_box(axA, [0, 0, 0], [1, 1, 1], color="gray", lw=0.8, alpha=0.5)
    axA.text(0.5, 0.5, 1.0, " stock top", color="gray", fontsize=8)
    axA.legend(fontsize=7, loc="upper left")
    _style_axes(axA, "A. Normalized frame (simulator)")
    _set_lims(axA, [0, 0, 0], [1, 1, 1])

    # Panel B: WCS / G-code frame.
    axB = fig.add_subplot(2, 3, 2, projection="3d")
    axB.plot(wcs[:, 0], wcs[:, 1], wcs[:, 2], color="C0", lw=2, label="path (WCS)")
    axB.scatter(wcs[:, 0], wcs[:, 1], wcs[:, 2], c=sc, cmap="viridis", s=20, zorder=5)
    # Stock box in WCS: top-centre G54 -> [±sx/2, ±sy/2] x [-sz, 0].
    sx, sy, sz = mc.stock_size_vec
    _draw_box(axB, [-sx / 2, -sy / 2, -sz], [sx / 2, sy / 2, 0.0],
              color="gray", lw=0.8, alpha=0.5)
    axB.scatter(0, 0, 0, color="magenta", s=50, marker="*", label="G54 (Z=0 top)", zorder=6)
    axB.text(0, 0, float(mc.safe_z_mm), f" safe Z={mc.safe_z_mm:.1f}mm",
             color="darkgreen", fontsize=8)
    axB.legend(fontsize=7, loc="upper left")
    _style_axes(axB, "B. WCS / G-code frame (machine)", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    pad = 0.1 * max(sx, sy, sz)
    _set_lims(axB, [-sx / 2 - pad, -sy / 2 - pad, -sz - pad],
              [sx / 2 + pad, sy / 2 + pad, mc.safe_z_mm + pad])

    # Panel C: round-trip, with approach/retract (rapids) drawn faded so the
    # cutting moves are visually separable from the post-processor overhead.
    axC = fig.add_subplot(2, 3, 3, projection="3d")
    axC.plot(positions[:, 0], positions[:, 1], positions[:, 2],
             color="C0", lw=2, label="original")
    n_rapid = n_feed = 0
    for seg in segments:
        pts, _ = plan_trajectory([seg], mc)
        if len(pts) < 2:
            continue
        if seg.kind == "rapid":
            axC.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="0.70", lw=1, ls=":", alpha=0.8)
            n_rapid += 1
        else:
            axC.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="C3", lw=1, ls="--")
            n_feed += 1
    axC.plot([], [], color="0.70", lw=1, ls=":", label=f"rapid/approach ({n_rapid})")
    axC.plot([], [], color="C3", lw=1, ls="--", label=f"cutting ({n_feed})")
    if n_orig == n_wp:
        axC.scatter(recovered[:, 0], recovered[:, 1], recovered[:, 2],
                    color="C2", s=25, marker="x", label="recovered waypoints")
    axC.legend(fontsize=7, loc="upper left")
    _style_axes(axC, f"C. G-code round-trip (post={args.post}, {n_extra:+d} moves)")
    _set_lims(axC, [0, 0, 0], [1, 1, 1])

    # Panel D: carved stock (original trajectory) vs target.
    axD = fig.add_subplot(2, 3, 4, projection="3d")
    if args.no_carve:
        axD.text2D(0.5, 0.5, "D. Carve skipped (--no-carve)", transform=axD.transAxes,
                   ha="center", va="center", fontsize=11)
    elif "grid" in carve_metrics:
        try:
            spacing = carve_metrics["spacing"]
            v = carve_metrics["v"]
            sov, sof = _mesh_from_sdf(carve_metrics["stock_orig"], spacing,
                                      abs(float(carve_metrics["stock_orig"].min())) + v)
            tvv, tvf = _mesh_from_sdf(carve_metrics["target_grid"], spacing,
                                      abs(float(carve_metrics["target_grid"].min())) + v)
            if len(sof):
                axD.plot_trisurf(sov[:, 0], sov[:, 1], sof, sov[:, 2],
                                 color="C0", alpha=0.6, linewidth=0)
            if len(tvf):
                axD.plot_trisurf(tvv[:, 0], tvv[:, 1], tvf, tvv[:, 2],
                                 color="C3", alpha=0.25, linewidth=0)
            axD.text(0.5, 0.5, 1.0, " target (red) / carved (blue)",
                     color="gray", fontsize=8)
            _style_axes(axD, f"D. Sim carve vs target  ({carve_metrics['grid'][0]}x{carve_metrics['grid'][1]}x{carve_metrics['grid'][2]})")
        except Exception as e:
            axD.text2D(0.5, 0.5, f"D. Carve failed:\n{e}", transform=axD.transAxes,
                       ha="center", va="center", fontsize=10)
            _style_axes(axD, "D. Sim carve vs target")
    _set_lims(axD, [0, 0, 0], [1, 1, 1])

    # Panel E: G-code carve vs sim carve (the gcode/sim mismatch proof).
    axE = fig.add_subplot(2, 3, 5, projection="3d")
    if args.no_carve:
        axE.text2D(0.5, 0.5, "E. Carve skipped (--no-carve)", transform=axE.transAxes,
                   ha="center", va="center", fontsize=11)
    elif "grid" in carve_metrics:
        try:
            # Original carve in blue surface, G-code carve in green wireframe so
            # any divergence is visible; target shown faint. Carved once above.
            spacing = carve_metrics["spacing"]
            v = carve_metrics["v"]
            sov, sof = _mesh_from_sdf(carve_metrics["stock_orig"], spacing,
                                      abs(float(carve_metrics["stock_orig"].min())) + v)
            sgv, sgf = _mesh_from_sdf(carve_metrics["stock_gcode"], spacing,
                                      abs(float(carve_metrics["stock_gcode"].min())) + v)
            tvv, tvf = _mesh_from_sdf(carve_metrics["target_grid"], spacing,
                                      abs(float(carve_metrics["target_grid"].min())) + v)
            if len(sof):
                axE.plot_trisurf(sov[:, 0], sov[:, 1], sof, sov[:, 2],
                                 color="C0", alpha=0.35, linewidth=0)
            if len(sgf):
                axE.plot_trisurf(sgv[:, 0], sgv[:, 1], sgf, sgv[:, 2],
                                 color="C2", alpha=0.0, edgecolor="C2", linewidth=0.2)
            if len(tvf):
                axE.plot_trisurf(tvv[:, 0], tvv[:, 1], tvf, tvv[:, 2],
                                 color="C3", alpha=0.15, linewidth=0)
            axE.text(0.5, 0.5, 1.0, " sim (blue) / G-code (green) / target (red)",
                     color="gray", fontsize=8)
            _style_axes(axE, f"E. G-code vs sim carve  (Dice={carve_metrics['dice_gcode']:.4f})")
        except Exception as e:
            axE.text2D(0.5, 0.5, f"E. Carve failed:\n{e}", transform=axE.transAxes,
                       ha="center", va="center", fontsize=10)
            _style_axes(axE, "E. G-code vs sim carve")
    _set_lims(axE, [0, 0, 0], [1, 1, 1])

    # Panel F: metrics summary.
    axF = fig.add_subplot(2, 3, 6)
    axF.axis("off")
    lines = [
        f"post-processor: {args.post}",
        f"original points: {n_orig}",
        f"G-code waypoints: {n_wp} ({n_extra:+d} approach/retract)",
        f"executed samples: {executed.shape[0]}",
    ]
    if n_orig == n_wp:
        lines.append(f"waypoint round-trip err: {rt_err_mm:.3e} mm")
    else:
        lines.append("waypoint round-trip err: n/a (approach moves)")
    lines.append(f"normalized Z: [{z_norm[0]:.4f}, {z_norm[1]:.4f}]  (1.0 = top)")
    lines.append(f"WCS Z: [{z_wcs[0]:.3f}, {z_wcs[1]:.3f}] mm  (0 = top)")
    lines.append(f"cut depth below top: {cut_depth:.3f} mm "
                 f"(target needs ~{target_depth_mm:.2f} mm)")
    if carve_metrics:
        lines.append("")
        lines.append(f"G-code vs sim carve Dice: {carve_metrics['dice_gcode']:.5f}")
        lines.append(f"carved-vs-target Dice:    {carve_metrics['dice_target']:.5f}")
        lines.append(f"carved vox: sim={carve_metrics['vox_carved']} "
                     f"gcode={carve_metrics['vox_gcode']} target={carve_metrics['vox_target']}")
        if carve_metrics["dice_gcode"] >= 0.999:
            lines.append("-> G-code reproduces the simulation (mismatch is NOT export math).")
        if carve_metrics["dice_target"] < 0.5:
            lines.append("-> carve does not resemble target (under-trained / wrong path).")
    else:
        lines.append("")
        lines.append("carve metrics: skipped (--no-carve)")
    axF.text(0.02, 0.98, "F. Metrics\n" + "\n".join(lines), transform=axF.transAxes,
             ha="left", va="top", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.7"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = args.save
    if out is None:
        out = os.path.join(os.path.dirname(traj_path) or ".",
                           f"trajectory_viz_{args.post}.png")
    plt.savefig(out, dpi=130)
    print(f"[viz] wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
