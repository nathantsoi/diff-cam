"""Visualize a STEP-to-SDF target grid in the CamEnv gymnasium environment.

Usage:
    python scripts/visualize_target.py utils/extrusion.npz --resolution 128
    python scripts/visualize_target.py utils/bowl.npz --resolution 32
"""
import os
import sys
import time
import argparse
import numpy as np

# Append project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cam_env.cam_env import CamEnvDiff


def main():
    parser = argparse.ArgumentParser(description="Visualize a target SDF grid.")
    parser.add_argument("target_npz", help="Path to the .npz target file")
    parser.add_argument("--resolution", "-r", type=int, default=None,
                        help="Grid resolution (default: read from npz)")
    args = parser.parse_args()

    # Read resolution from npz if not specified
    data = np.load(args.target_npz)
    sdf = data["sdf"]
    if args.resolution is None:
        args.resolution = sdf.shape[0]
    print(f"Target: {args.target_npz} (resolution={args.resolution})")
    print(f"SDF shape: {sdf.shape}, min={sdf.min():.6f}, max={sdf.max():.6f}")

    if sdf.shape[0] != args.resolution:
        print(f"WARNING: NPZ grid shape {sdf.shape} doesn't match resolution {args.resolution}")
        print(f"Using resolution from NPZ: {sdf.shape[0]}")
        args.resolution = sdf.shape[0]

    print("Initializing environment...")
    env = CamEnvDiff(
        resolution=args.resolution,
        max_steps=64,
        target_shape="grid",
        target_sdf_path=args.target_npz,
        render_mode="human"
    )

    obs, info = env.reset()
    print("Environment reset complete. Target loaded!")

    # Hide stock to show target clearly
    env.show_part = True
    env.show_stock = False
    env.show_tool = False

    print("Starting visualizer window...")
    print("Controls:")
    print("  Drag Left Mouse Button (LMB) to rotate camera")
    print("  Press R to zoom in, F to zoom out")
    print("  Press C to toggle stock visibility")
    print("  Press V to toggle target part visibility")
    print("  Press Escape to close window")

    while env.gui is None or env.gui.running:
        env.render()
        time.sleep(0.01)

    env.close()
    print("Visualizer closed.")


if __name__ == "__main__":
    main()
