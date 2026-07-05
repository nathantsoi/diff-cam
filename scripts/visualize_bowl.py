import os
import sys
import time
import numpy as np
import taichi as ti

# Append project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cam_env.cam_env import CamEnv

def main():
    print("Initializing Gym environment with bowl target...")
    env = CamEnv(
        resolution=32,
        max_steps=64,
        target_shape="grid",
        target_sdf_path="utils/bowl.npz",
        render_mode="human"
    )
    
    obs, info = env.reset()
    print("Environment reset complete. Target loaded!")
    
    # Hide the stock by default to make the target bowl perfectly visible.
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
    
    # Keep rendering while the GUI window is open
    while env.gui is None or env.gui.running:
        env.render()
        time.sleep(0.01)
        
    env.close()
    print("Visualizer closed.")

if __name__ == "__main__":
    main()
