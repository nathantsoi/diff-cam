#!/usr/bin/env python3
"""Render a saved training run's carve as an mp4 via the Taichi CSG simulator.

A training run saves ``trajectory.npy`` (+ ``trajectory_deltas.npy``) and an
``args.json`` describing the stock box, tool, target and simulator settings.
This script rebuilds the ``CSGSimulatorDelta`` exactly as ``algorithms.train_csg``
does, replays the saved trajectory through ``sim.forward``, and renders the
stock being carved step-by-step into an mp4 -- the same raymarch + ffmpeg path
``train_csg.record_video`` uses during training.

The web dashboard's "Generate video" button (served by ``scripts/serve_web_https.py``)
calls this script on demand so a run's carve can be watched without re-running
training or firing up the live GUI.

Examples
--------
    # Render one run (writes runs/<run>/videos/run.mp4)
    uv run python scripts/render_run_video.py --run runs/CamEnvDiff-v0__train_csg__104__1782757879_8480

    # Custom output path / fps
    uv run python scripts/render_run_video.py --run runs/<run> --out /tmp/run.mp4 --fps 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Make `algorithms` / `simulator` / `cam` importable when run as a script.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from algorithms.train_csg import record_video  # noqa: E402
from cam.units import inch_to_mm  # noqa: E402
from simulator.csg_simulator import CSGSimulatorDelta  # noqa: E402


def _load_run(run_dir: str):
    """Load (args_dict, positions, deltas) for a training run dir."""
    args_path = os.path.join(run_dir, "args.json")
    if not os.path.exists(args_path):
        raise SystemExit(f"args.json not found in {run_dir}")
    with open(args_path) as f:
        args = json.load(f)

    pos_path = os.path.join(run_dir, "trajectory.npy")
    if not os.path.exists(pos_path):
        raise SystemExit(f"trajectory.npy not found in {run_dir}")
    positions = np.load(pos_path).astype(np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise SystemExit(f"{pos_path} must be (T,3); got {positions.shape}")

    delta_path = os.path.join(run_dir, "trajectory_deltas.npy")
    deltas = np.load(delta_path).astype(np.float32) if os.path.exists(delta_path) else None
    return args, positions, deltas


def _g(cfg, key, default):
    """cfg[key] coerced to a plain float/tuple, else default."""
    v = cfg.get(key, None)
    if v is None:
        return default
    return v


def _tuple3(v, default):
    if v is None:
        return tuple(default)
    return tuple(float(x) for x in v)


def build_sim(args):
    """Reconstruct the CSGSimulatorDelta from a run's args.json (mirrors train_csg)."""
    max_steps = int(args.get("max_steps", 128))
    # Grid/STEP targets carry their stock box in the target NPZ and don't pass
    # --stock-size-in; hand the sim None so the NPZ's box is used directly
    # (passing the 1" default would only trigger the sim's conflict warning).
    stock_in = args.get("stock_size_in")
    if stock_in is None and args.get("target_shape") == "grid":
        stock_size_in = None
    else:
        stock_size_in = _tuple3(stock_in, (1.0, 1.0, 1.0))
    sim = CSGSimulatorDelta(
        resolution=int(args.get("resolution", 32)),
        max_steps=max_steps,
        k_init=float(args.get("k_init", 10.0)),
        target_shape=args.get("target_shape", "sphere"),
        target_sdf_path=args.get("target_sdf_path"),
        tool_start=(0.5, 0.5, 1.0),
        stock_size_in=stock_size_in,
        voxel_size_mm=float(args.get("voxel_size_mm", 0.5) or 0.5),
        work_volume_in=_tuple3(args.get("workspace_in"), (16.0, 12.0, 10.0)),
        stock_origin_in=_tuple3(args.get("stock_origin_in"), None) if args.get("stock_origin_in") is not None else None,
        dt=float(args.get("dt", 0.01)),
        rapid_ipm=float(args.get("rapid_ipm", 500.0)),
        feed_ipm=float(args.get("feed_ipm", 10.0)),
        safe_distance_in=float(args.get("safe_distance_in", 0.1)),
        enforce_speed_limits=bool(args.get("enforce_speed_limits", True)),
    )
    sim.set_target_params(
        radius_mm=float(args.get("target_radius_mm", 11.43)),
        height_mm=float(args.get("target_height_mm", 22.86)),
        half_size_mm=float(args.get("target_radius_mm", 11.43)),
        center=(0.5, 0.5, 0.5),
    )
    sim.tool_radius[None] = float(args.get("tool_radius_mm", 3.175))
    sim.tool_height[None] = float(args.get("tool_height_mm", 25.0))
    sim.holder_radius[None] = inch_to_mm(2.5 / 2.0)
    sim.bake_target_grid()
    sim.set_target_volume()
    return sim, max_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run dir (runs/<name> or absolute)")
    ap.add_argument("--out", default=None, help="output mp4 path (default: <run>/videos/run.mp4)")
    ap.add_argument("--fps", type=int, default=30, help="frames per second")
    ap.add_argument("--mode", choices=["hard", "soft", "both"], default="hard", help="carving mode for video rendering")
    args = ap.parse_args()

    run_dir = args.run if os.path.isabs(args.run) else os.path.join(REPO, args.run)
    run_dir = os.path.normpath(run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run dir not found: {run_dir}")

    cfg, positions, deltas = _load_run(run_dir)
    print(f"[video] run: {run_dir}")
    print(f"[video] target_shape={cfg.get('target_shape')}  "
          f"dt={cfg.get('dt')}  resolution={cfg.get('resolution')}  "
          f"voxel_size_mm={cfg.get('voxel_size_mm')}  points={len(positions)}")

    sim, max_steps = build_sim(cfg)
    T = len(positions)

    # Reload the saved displacements into the sim and carve, exactly as
    # train_csg does for the final replay (params -> tool_delta -> forward).
    if deltas is not None:
        d = deltas
    else:
        # Fallback: differences of the saved positions (positions[0] == tool_start).
        d = np.diff(positions, axis=0, prepend=np.asarray([[0.5, 0.5, 1.0]], np.float32))
    pad = np.zeros((max_steps, 3), dtype=np.float32)
    pad[: min(len(d), max_steps)] = d[: max_steps]
    sim.tool_delta.from_torch(torch.as_tensor(pad))

    import shutil
    canonical = args.out or os.path.join(run_dir, "videos", "run.mp4")
    written_last = None

    if args.mode in ("hard", "both"):
        print(f"[video] carving {T} steps with hard boolean subtraction...")
        sim.forward_hard(T)
        out_hard = args.out if (args.out and args.mode == "hard") else os.path.join(run_dir, "videos", "run_hard.mp4")
        written_last = record_video(sim, None, T, out_hard, args.fps)
        if not written_last:
            raise SystemExit("[video] rendering hard video failed (no frames produced; check ffmpeg/taichi)")
        print(f"[video] wrote {written_last}")
        if not args.out or args.mode == "both":
            shutil.copyfile(written_last, canonical)

    if args.mode in ("soft", "both"):
        print(f"[video] carving {T} steps with soft smooth_max union...")
        sim.forward(T)
        out_soft = args.out if (args.out and args.mode == "soft") else os.path.join(run_dir, "videos", "run_soft.mp4")
        written_last = record_video(sim, None, T, out_soft, args.fps)
        if not written_last:
            raise SystemExit("[video] rendering soft video failed (no frames produced; check ffmpeg/taichi)")
        print(f"[video] wrote {written_last}")

    # Emit a machine-readable marker the server can parse.
    final_marker = canonical if os.path.exists(canonical) else written_last
    print(f"VIDEO_PATH={os.path.relpath(final_marker, REPO)}")


if __name__ == "__main__":
    main()
