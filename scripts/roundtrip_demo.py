"""End-to-end demo: trajectory -> G-code -> executed trajectory.

Loads the saved trajectory, exports it to RS274/NGC G-code, re-plans
("executes") that G-code with the exact-stop trapezoidal planner, and reports
how closely the executed trajectory matches the original — both geometrically
(Fréchet / DTW / arc-length RMSE / waypoint round-trip) and, end-to-end, by
carving both trajectories in the simulator and comparing the resulting stock.

Run:
    uv run python scripts/roundtrip_demo.py
    uv run python scripts/roundtrip_demo.py --resolution 32 --dt 0.05
"""

import argparse
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cam import (
    MachineConfig,
    trajectory_to_gcode,
    save_gcode,
    parse_gcode,
    segment_waypoints,
    gcode_to_trajectory,
    discrete_frechet,
    dtw_distance,
    resampled_rmse,
    waypoint_roundtrip_error,
)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default=os.path.join(repo, "trajectory.npy"))
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: runs/roundtrip_<timestamp>)")
    ap.add_argument("--workspace-mm", type=float, default=100.0,
                    help="machine work-volume cube edge (mm)")
    ap.add_argument("--stock-size-in", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    metavar=("X", "Y", "Z"),
                    help="stock box (x y z) in inches -- the normalized cube (default 1 in cube)")
    ap.add_argument("--feed", type=float, default=600.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--resolution", type=int, default=24)
    ap.add_argument("--post", default="rs274",
                    help="post-processor / G-code dialect (rs274 | haas)")
    ap.add_argument("--no-carve", action="store_true",
                    help="skip the simulator carved-stock comparison")
    args = ap.parse_args()

    # All artifacts go under runs/ (matching algorithms/train_csg.py).
    out_dir = args.out_dir or os.path.join(
        repo, "runs",
        "roundtrip_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    gcode_path = os.path.join(out_dir, "trajectory.nc" if args.post == "haas" else "trajectory.ngc")
    print(f"[run] writing outputs to {out_dir}")

    cfg = MachineConfig(workspace_mm=args.workspace_mm, feed=args.feed, dt=args.dt,
                        stock_size_in=tuple(args.stock_size_in))
    # Report metric distances in mm using a representative stock scale.
    scale = float(np.mean(cfg.stock_size_vec))

    original = np.load(args.trajectory).astype(np.float64)
    print(f"Loaded original trajectory: {original.shape} points")

    # 1. Export to G-code.
    gcode = save_gcode(original, gcode_path, cfg, post=args.post)
    n_lines = len([ln for ln in gcode.splitlines() if ln and not ln.startswith("(")])
    print(f"Exported G-code: {n_lines} blocks -> {gcode_path}")

    # 2. Parse + plan (execute) the G-code.
    segments = parse_gcode(gcode, cfg)
    recovered_wp = segment_waypoints(segments)
    executed, times = gcode_to_trajectory(gcode, cfg)
    np.save(os.path.join(out_dir, "executed_trajectory.npy"), executed)
    print(f"Executed trajectory: {executed.shape} points, "
          f"{times[-1]:.2f}s of motion")

    # 3. Geometric similarity.
    print("\n=== Trajectory similarity (original vs executed) ===")
    if len(recovered_wp) == len(original):
        print(f"  waypoint round-trip error : {waypoint_roundtrip_error(original, recovered_wp, scale):.3e} mm")
    else:
        # Posts that add approach/retract moves (e.g. haas) change the waypoint
        # count, so the strict waypoint round-trip metric does not apply.
        print(f"  waypoint round-trip error : n/a "
              f"({len(recovered_wp)} waypoints vs {len(original)}; post '{args.post}' adds approach moves)")
    print(f"  discrete Frechet          : {discrete_frechet(original, executed, scale):.3e} mm")
    print(f"  DTW (mean matched dist)   : {dtw_distance(original, executed, scale):.3e} mm")
    print(f"  arc-length resampled RMSE : {resampled_rmse(original, executed, scale=scale):.3e} mm")

    # 4. Carved-stock similarity.
    if not args.no_carve:
        from cam.sim_exec import carve_stock
        from simulator.csg_metrics import sdf_to_mask, dice_score, hd95

        print("\n=== Carved-stock similarity (hard CSG) ===")
        stock_orig = carve_stock(original, resolution=args.resolution)
        mask_orig = sdf_to_mask(stock_orig)
        stock_exec = carve_stock(executed, resolution=args.resolution)
        mask_exec = sdf_to_mask(stock_exec)
        print(f"  solid voxels  original/executed : "
              f"{int(mask_orig.sum())} / {int(mask_exec.sum())}")
        print(f"  Dice                            : {dice_score(mask_orig, mask_exec):.5f}")
        try:
            print(f"  HD95 (voxels)                   : {float(hd95(mask_orig, mask_exec)):.4f}")
        except ValueError:
            print("  HD95                            : n/a (empty surface)")

    print("\nDone.")


if __name__ == "__main__":
    main()
