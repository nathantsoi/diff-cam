"""Visualize a STEP-to-SDF target grid in the CamEnv gymnasium environment.

The grid's physical dimensions (stock box, voxel size) come from the NPZ.

Usage:
    python scripts/visualize_target.py utils/extrusion.npz
    python scripts/visualize_target.py utils/bowl.npz
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
    args = parser.parse_args()

    data = np.load(args.target_npz)
    sdf = data["sdf"]
    print(f"Target: {args.target_npz}")
    print(f"SDF shape: {sdf.shape}, min={sdf.min():.4f} mm, max={sdf.max():.4f} mm")
    print(f"Stock: {np.asarray(data['stock_size_mm']).tolist()} mm "
          f"@ {float(data['voxel_size_mm']):.4f} mm/voxel")

    print("Initializing environment...")
    env = CamEnvDiff(
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
