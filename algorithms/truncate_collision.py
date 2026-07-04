"""Truncate a trajectory where the tool holder would collide with the stock.

Given a saved ``(T, 3)`` trajectory (from ``algorithms.train_csg``), this:

  1. Hard-carves it step-by-step into the stock (the same idempotent boolean
     subtraction ``eval.eval_csg`` uses), populating the per-step stock SDF.
  2. For each segment measures the holder-to-stock CLEARANCE -- the minimum
     sharp holder SDF over remaining-material voxels (positive = gap,
     negative = penetration).
  3. Finds the first segment whose clearance drops below a safety margin
     (the imminent collision) and stops the toolpath at the last segment whose
     clearance exceeds the margin -- i.e. it cuts the trajectory a configurable
     distance *before* the holder would contact the remaining material.
  4. Overwrites ``trajectory.npy`` with the truncated path (the original is
     preserved as ``trajectory.untruncated.npy``) so the exported G-code is
     collision-free.

This is the "respect collisions in the sim and stop the toolpath a small
configurable distance prior to collision" step. It is a post-process backstop:
the trainer's z-floor clamp (``--z-floor-epsilon-mm``) prevents the deep plunge
that causes most collisions, but any trajectory that still collides is trimmed
here before export.

Examples
--------
    uv run python -m algorithms.truncate_collision \
        --run-dir runs/<run> --clearance-mm 1.0
"""

import argparse
import json
import os
import shutil

import numpy as np

from cam.sim_exec import _HardCarveSimulator
from cam.units import inch_to_mm


def _build_sim(args_path):
    """Build a _HardCarveSimulator matching a run's args.json geometry.

    Mirrors ``truncate_trajectory._build_sim`` but ALSO sets the holder params
    (radius/height) to match training -- the holder is the collision body, so it
    must be configured identically or the clearance query is meaningless.
    """
    with open(args_path) as f:
        a = json.load(f)
    stock_size_in = tuple(a.get("stock_size_in", (1.0, 1.0, 1.0)))
    voxel_size_mm = float(a.get("voxel_size_mm", 0.5))
    target_shape = a.get("target_shape", "sphere")
    radius_mm = float(a.get("target_radius_mm", 11.43))
    height_mm = float(a.get("target_height_mm", 22.86))
    workspace_in = tuple(a.get("workspace_in", (16.0, 12.0, 10.0)))
    tool_radius_mm = float(a.get("tool_radius_mm", 3.175))
    tool_height_mm = float(a.get("tool_height_mm", 25.0))

    traj = np.load(os.path.join(os.path.dirname(args_path), "trajectory.npy"))
    T = len(traj)

    sim = _HardCarveSimulator(
        resolution=32,
        max_steps=T - 1,
        target_shape=target_shape,
        tool_start=tuple(float(v) for v in traj[0]),
        stock_size_in=stock_size_in,
        voxel_size_mm=voxel_size_mm,
        work_volume_in=workspace_in,
    )
    sim.tool_radius[None] = tool_radius_mm
    sim.tool_height[None] = tool_height_mm
    # Holder: 2.5 inch diameter cylinder above the cutter (matches train_csg).
    sim.holder_radius[None] = inch_to_mm(2.5 / 2.0)
    sim.holder_height[None] = float(sim.work_volume_mm[2])
    sim.set_target_params(radius_mm=radius_mm, height_mm=height_mm,
                          half_size_mm=radius_mm, center=(0.5, 0.5, 0.5))
    sim.bake_target_grid()
    sim.set_target_volume()
    # Disable the z-floor here: we are measuring the AS-COMMANDED trajectory's
    # collisions (the saved path is already the clamped one if the floor ran
    # during training; re-clamping would only mask collisions we want to find).
    sim.enforce_z_floor[None] = 0
    return sim, traj, a


def truncate_at_collision(args_path, out_path=None, clearance_mm=1.0,
                          min_keep_frac=0.3, verbose=True):
    """Hard-carve a trajectory, find the first holder/stock collision, truncate.

    ``clearance_mm``: a segment is SAFE while the holder's minimum clearance to
    remaining material exceeds this (millimetres). The toolpath is cut at the
    last safe segment -- i.e. a configurable distance *before* the holder would
    contact the stock. ``min_keep_frac``: never truncate below this fraction of
    the trajectory (keep at least the first ``min_keep_frac * T`` positions) so a
    spurious early collision cannot discard the bulk.

    ``out_path``: if given, the truncated trajectory is written there (the
    original at ``<args_path dir>/trajectory.npy`` is first copied to
    ``trajectory.untruncated.npy`` if that backup does not yet exist).

    Returns ``(t_stop, t_bad, min_clearance_mm, n_kept)`` where ``t_stop`` is the
    last safe segment index, ``t_bad`` is the first unsafe segment (or -1 if all
    safe), ``min_clearance_mm`` is the worst clearance over the trajectory, and
    ``n_kept`` is the number of positions kept.
    """
    sim, traj, a = _build_sim(args_path)
    T = len(traj)
    deltas = np.diff(traj, axis=0)
    padded = np.zeros((sim.max_steps, 3), dtype=np.float32)
    padded[: len(deltas)] = deltas
    sim.tool_delta.from_numpy(padded)
    sim.forward_hard(T)

    n_seg = T - 1
    clearances = np.full(n_seg, np.inf, dtype=np.float64)
    for t in range(n_seg):
        clearances[t] = float(sim.holder_min_clearance_at(t)) * sim.v

    min_clearance_mm = float(clearances.min()) if n_seg > 0 else np.inf
    # First segment whose clearance is below the safety margin (the collision).
    unsafe = np.where(clearances < clearance_mm)[0]
    if len(unsafe) == 0:
        t_bad = -1
        t_stop = n_seg - 1  # keep everything
    else:
        t_bad = int(unsafe[0])
        t_stop = max(0, t_bad - 1)  # last safe segment
    # Respect the minimum-keep floor (in positions).
    min_keep_pos = int(np.ceil(min_keep_frac * T))
    t_stop = max(t_stop, min_keep_pos - 1)
    t_stop = min(t_stop, n_seg - 1)

    # Truncated trajectory: positions [0 .. t_stop+1] (t_stop+1 positions,
    # i.e. through the end of the last safe segment).
    n_kept = t_stop + 2
    truncated = traj[:n_kept].copy()

    if out_path is not None:
        # Preserve the original (only the first time) before overwriting.
        traj_path = os.path.join(os.path.dirname(args_path), "trajectory.npy")
        backup = os.path.join(os.path.dirname(args_path), "trajectory.untruncated.npy")
        if os.path.abspath(out_path) == os.path.abspath(traj_path):
            if not os.path.exists(backup):
                shutil.copy2(traj_path, backup)
        np.save(out_path, truncated.astype(np.float32))

    if verbose:
        status = "no collision" if t_bad < 0 else f"first collision @ seg {t_bad}"
        print(f"[trunc-coll] T={T} segs={n_seg} clearance_mm={clearance_mm} "
              f"-> {status}")
        print(f"[trunc-coll] min clearance over trajectory: {min_clearance_mm:.4f} mm")
        if t_bad >= 0:
            print(f"[trunc-coll] stopping at seg {t_stop} (kept {n_kept}/{T} "
                  f"positions, dropped {T - n_kept} trailing)")
            # Show the clearance profile around the collision for diagnosis.
            lo = max(0, t_bad - 2)
            hi = min(n_seg, t_bad + 3)
            window = " ".join(f"{clearances[i]:.3f}" for i in range(lo, hi))
            print(f"[trunc-coll] clearance window segs [{lo},{hi}): {window} mm")
        else:
            print(f"[trunc-coll] keeping full trajectory ({n_kept} positions)")
        if out_path is not None:
            print(f"[trunc-coll] wrote {out_path}")

    return t_stop, t_bad, min_clearance_mm, n_kept


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True,
                    help="run directory containing trajectory.npy + args.json")
    ap.add_argument("--out", default=None,
                    help="output .npy path (default: overwrite <run-dir>/trajectory.npy)")
    ap.add_argument("--clearance-mm", type=float, default=1.0,
                    help="safety margin in mm: stop at the last segment whose "
                         "holder-to-stock clearance exceeds this (the toolpath "
                         "is cut this distance before the collision)")
    ap.add_argument("--min-keep-frac", type=float, default=0.3,
                    help="never truncate below this fraction of the trajectory")
    args = ap.parse_args()

    out = args.out or os.path.join(args.run_dir, "trajectory.npy")
    truncate_at_collision(os.path.join(args.run_dir, "args.json"), out,
                          clearance_mm=args.clearance_mm,
                          min_keep_frac=args.min_keep_frac)


if __name__ == "__main__":
    main()
