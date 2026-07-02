"""Truncate a trajectory at the point the tool stops cutting the part.

Given a saved ``(T, 3)`` trajectory (from ``algorithms.train_csg``), this:

  1. Hard-carves it step-by-step into the stock (the same idempotent boolean
     subtraction ``eval.eval_csg`` uses), recording the carved stock SDF after
     every step.
  2. Measures the per-step REMOVED material volume (how much stock the tool cut
     on that step).
  3. Finds ``t*`` -- the position index after the LAST step whose removal exceeds
     a threshold -- i.e. it drops the trailing run where the tool is no longer
     cutting the part (the "moving away / cutting air" tail).
  4. Saves the mid-cut state at ``t*`` (the stock SDF + the tool position) to a
     ``.npz`` that ``train_csg --init-stock-from`` consumes to train a SECOND
     trajectory that finishes carving the remaining material.

This is the "remove the end of the trajectory where the tool is not cutting,
save that state, then train another trajectory to finish" step of staged
training. See ``scripts/staged_train.py`` for the orchestrator.

Examples
--------
    uv run python -m algorithms.truncate_trajectory \
        --trajectory runs/<run>/trajectory.npy --out state.npz
"""

import argparse
import json
import os

import numpy as np

from cam.sim_exec import _HardCarveSimulator
from cam.units import inch_to_mm


def _build_sim(args_path):
    """Build a _HardCarveSimulator matching a run's args.json geometry."""
    with open(args_path) as f:
        a = json.load(f)
    stock_size_in = tuple(a.get("stock_size_in", (1.0, 1.0, 1.0)))
    voxel_size_mm = float(a.get("voxel_size_mm", 0.5))
    target_shape = a.get("target_shape", "sphere")
    radius_mm = float(a.get("target_radius_mm", 11.43))
    height_mm = float(a.get("target_height_mm", 22.86))

    traj = np.load(os.path.join(os.path.dirname(args_path), "trajectory.npy"))
    T = len(traj)

    sim = _HardCarveSimulator(
        resolution=32,
        max_steps=T - 1,
        target_shape=target_shape,
        tool_start=tuple(float(v) for v in traj[0]),
        stock_size_in=stock_size_in,
        voxel_size_mm=voxel_size_mm,
        work_volume_in=(16.0, 12.0, 10.0),
    )
    sim.tool_radius[None] = 3.175
    sim.tool_height[None] = 25.0
    sim.set_target_params(radius_mm=radius_mm, height_mm=height_mm,
                          half_size_mm=radius_mm, center=(0.5, 0.5, 0.5))
    sim.bake_target_grid()
    sim.set_target_volume()
    return sim, traj, a


def truncate_trajectory(args_path, out_path, remove_thresh=1e-6,
                        min_keep_frac=0.3, verbose=True):
    """Hard-carve a trajectory, find the last meaningful cut, save the state.

    ``remove_thresh``: a step counts as "cutting" if its removed volume (as a
    fraction of the stock envelope volume) exceeds this. The default is a sub-
    voxel noise floor so the last TRUE cutting step is captured (the trailing
    excursion removes ~0). ``min_keep_frac``: never truncate below this fraction
    of the trajectory (keep at least the first ``min_keep_frac * T`` steps) so a
    spurious late cut cannot discard the bulk.

    Returns ``(t_star, removed_frac, n_kept)``. Writes the saved state to
    ``out_path``: ``stock_sdf`` (Nx,Ny,Nz), ``tool_pos`` (3,), ``t_trunc`` (int).
    """
    sim, traj, a = _build_sim(args_path)
    T = len(traj)
    deltas = np.diff(traj, axis=0)
    padded = np.zeros((sim.max_steps, 3), dtype=np.float32)
    padded[: len(deltas)] = deltas
    sim.tool_delta.from_numpy(padded)
    sim.forward_hard(T)

    stock = sim.stock.to_numpy()  # (T, Nx, Ny, Nz); stock[t] is after t cuts
    voxel_vol = 1.0 / (sim.Nx * sim.Ny * sim.Nz)  # unit-cube volume per voxel
    # Material present at step t = voxels where the stock SDF is NEGATIVE
    # (box_sdf < 0 inside the envelope; carving sets carved voxels POSITIVE via
    # stock[t+1] = max(stock[t], -tool_d) with -tool_d > 0 inside the tool).
    # Removed volume on cut t (t in 0..T-2) = material[t] - material[t+1] >= 0.
    material = (stock < 0.0).reshape(T, -1).sum(axis=1) * voxel_vol
    removed = np.maximum(0.0, material[:-1] - material[1:])  # len T-1

    # Last cut whose removal exceeds the threshold.
    cutting = np.where(removed > remove_thresh)[0]
    if len(cutting) == 0:
        t_last_cut = T - 2  # keep everything if nothing clearly "cut"
    else:
        t_last_cut = int(cutting[-1])
    # t* = position index AFTER the last meaningful cut (the truncation point).
    t_star = min(T - 1, t_last_cut + 1)
    # Respect the minimum-keep floor.
    t_star = max(t_star, int(np.ceil(min_keep_frac * (T - 1))) + 1)
    t_star = min(t_star, T - 1)

    saved_stock = stock[t_star].copy()
    saved_pos = traj[t_star].copy()
    np.savez(out_path, stock_sdf=saved_stock.astype(np.float32),
             tool_pos=saved_pos.astype(np.float32), t_trunc=np.int64(t_star))

    if verbose:
        peak = float(removed.max()) if len(removed) else 0.0
        n_cut = int((removed > remove_thresh).sum())
        print(f"[truncate] T={T} cuts={T-1} cutting_steps={n_cut} "
              f"peak_removed={peak:.4f} thresh={remove_thresh:.4f}")
        print(f"[truncate] last meaningful cut at idx {t_last_cut} -> "
              f"t*={t_star} (kept {t_star+1}/{T} positions, dropped "
              f"{T-1-t_star} trailing)")
        print(f"[truncate] saved state to {out_path}: tool_pos="
              f"{saved_pos.tolist()}")

    return t_star, removed, t_star + 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True,
                    help="run directory containing trajectory.npy + args.json")
    ap.add_argument("--out", required=True, help="output .npz state path")
    ap.add_argument("--remove-thresh", type=float, default=1e-6,
                    help="per-step removed-volume fraction (of stock envelope) "
                         "above which a step counts as 'cutting' (sub-voxel "
                         "noise floor by default)")
    ap.add_argument("--min-keep-frac", type=float, default=0.3,
                    help="never truncate below this fraction of the trajectory")
    args = ap.parse_args()
    truncate_trajectory(os.path.join(args.run_dir, "args.json"), args.out,
                        remove_thresh=args.remove_thresh,
                        min_keep_frac=args.min_keep_frac)


if __name__ == "__main__":
    main()
