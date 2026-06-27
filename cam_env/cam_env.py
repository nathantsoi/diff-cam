import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import taichi as ti
import time

from simulator.csg_simulator import *


class CamEnvDiff(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        resolution=32,
        max_steps=64,
        k_init=10.0,
        target_shape = "sphere",
        render_mode: Optional[str] = None,
        init_taichi: bool = True,
    ):
        """
        Args:
            resolution: int, the resolution of the voxel grid (e.g. 32 means 32x32x32)
            max_steps: int, maximum number of steps per episode
            k_init: float, initial value for reward shaping parameter K
            target_shape: str or None, if specified should be one of ["cylinder", "box", "sphere"].
            render_mode: str or None, if "human" will render with Taichi GUI. If "rgb_array", will return RGB arrays from render() instead. If None, no rendering.
        """

        super().__init__()

        self.resolution = resolution
        self.dx = 1.0 / resolution
        self.grid_norm = float(resolution ** 3)

        self.max_steps = max_steps
        self.max_cuts = max_steps - 1
        self.target_shape = target_shape
        self.tool_radius = 0.05
        self.tool_height = 0.15
        self.tool_start = [0.5, 0.5, 1.0]
        self.render_mode = render_mode

        self.simulator = None
        self.global_step = 0
        self.current_step = 0

        self.k_init = k_init
        self.init_taichi = init_taichi

        voxels_per_step = 3.0                       # design choice
        self.max_delta = voxels_per_step * self.dx
        self.action_space = spaces.Box(
            low=-self.max_delta, high=self.max_delta, shape=(3,), dtype=np.float32
        )

        self.obs_dims = 3 + 2 + (resolution**3) + (resolution**3) # tool_pos + radius + height + stock_grid + target_grid
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dims,), dtype=np.float32
        )

        self._initialize_sim()

        # Rendering state
        # --- Rendering state (raymarch only) ---
        self.gui = None  # ti.GUI handle; uses the simulator's raymarch_buffer

        # Orbit camera params (spherical coords around cam_center)
        self.cam_r = 2.5
        self.cam_theta = -1.57
        self.cam_phi = 1.0
        self.cam_center = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Mouse state for orbit
        self.last_mouse_pos = None
        self.lmb_down = False

        # Layer toggles
        self.show_tool = True
        self.show_stock = True
        self.show_part = True
        self.show_help = False

        self._last_loss = 0.0
        self.last_info = {}


    def _initialize_sim(self):
        self.simulator = CSGSimulatorDelta(
            resolution=self.resolution,
            max_steps=self.max_steps,
            k_init=self.k_init,
            target_shape=self.target_shape,
            tool_start=self.tool_start,
            init_taichi=self.init_taichi,
        )
        self.simulator.tool_radius[None] = self.tool_radius
        self.simulator.tool_height[None] = self.tool_height

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.simulator.zero_tool_deltas()

        self.simulator.init_stock()

        self.simulator.target_params["radius"][None] = 0.4
        self.simulator.target_params["center"][None] = [0.5, 0.5, 0.5]
        self.simulator.tool_radius[None] = 0.05
        self.simulator.tool_height[None] = 0.15
        self.simulator.bake_target_grid()
        self.simulator.set_target_volume()

        self.simulator.reconstruct_positions(0)
        # self.simulator.init_tool_template()

        self.current_step = 0

        obs = self._get_obs()
        info = {"step": 0}
        return obs, info

    def _get_obs(self):
        t = self.current_step
        tool_pos = self.simulator.tool_pos.to_numpy()[t].astype(np.float32)      # (3,)
        radius = float(self.simulator.tool_radius[None])
        height = float(self.simulator.tool_height[None])
        stock_grid = self.simulator.stock.to_numpy()[t].ravel().astype(np.float32)
        target_grid = self.simulator.target.to_numpy().ravel().astype(np.float32)
        return np.concatenate(
            [tool_pos, [radius, height], stock_grid, target_grid]
        ).astype(np.float32)    

    def step(self, action):
        # before
        t = self.current_step
        loss_before = self.simulator.loss_at(t)

        # act
        action = np.asarray(action, dtype=np.float32)
        t = self.current_step

        self.simulator.tool_delta[t] = ti.Vector([float(action[0]),
                                              float(action[1]),
                                              float(action[2])])
        self.simulator.reconstruct_positions(t + 1)
        self.simulator.apply_cut(t)
        self.simulator.set_current_step(t) # rendering
        loss_after = self.simulator.loss_at(t + 1)
        self.current_step += 1

        # calculate reward based on compute_loss
        reward = loss_before - loss_after
        truncated = (t + 1) >= self.max_cuts
        terminated = False

        info = {
            "step": t,
            "reward": reward,
        }
        return self._get_obs(), reward, terminated, truncated, info
    
    # ========================================================================
    # Rendering
    # ========================================================================
    def _initialize_render(self):
        if self.gui is not None:
            return
        # Match the simulator's raymarch_buffer resolution (1024x768).
        self.gui = ti.GUI(
            "CNC RL Environment (raymarch)",
            res=(1024, 768),
            fast_gui=False,
        )
        self.last_mouse_pos = self.gui.get_cursor_pos()

    def _handle_input(self):
        # Layer toggles + camera orbit/zoom via ti.GUI events.
        for e in self.gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                self.lmb_down = True
                self.last_mouse_pos = self.gui.get_cursor_pos()
            elif e.key == ti.GUI.ESCAPE:
                self.gui.running = False
            elif e.key in ("h", "H"):
                self.show_help = not self.show_help
            elif e.key in ("1", "z", "Z"):
                self.show_tool = not self.show_tool
            elif e.key in ("3", "c", "C"):
                self.show_stock = not self.show_stock
            elif e.key in ("4", "v", "V"):
                self.show_part = not self.show_part

        for e in self.gui.get_events(ti.GUI.RELEASE):
            if e.key == ti.GUI.LMB:
                self.lmb_down = False

        # Drag-to-orbit
        curr_mouse_pos = self.gui.get_cursor_pos()
        if self.lmb_down and self.last_mouse_pos is not None:
            dx = curr_mouse_pos[0] - self.last_mouse_pos[0]
            dy = curr_mouse_pos[1] - self.last_mouse_pos[1]
            self.cam_theta -= dx * 5.0
            self.cam_phi += dy * 5.0
            self.cam_phi = max(0.05, min(3.09, self.cam_phi))
        self.last_mouse_pos = curr_mouse_pos

        # Zoom: R closer, F farther
        zoom_speed = 0.05
        if self.gui.is_pressed("r"):
            self.cam_r = max(0.5, self.cam_r - zoom_speed)
        if self.gui.is_pressed("f"):
            self.cam_r = min(10.0, self.cam_r + zoom_speed)

    def _compute_camera(self):
        cx = self.cam_r * np.sin(self.cam_phi) * np.cos(self.cam_theta)
        cy = self.cam_r * np.sin(self.cam_phi) * np.sin(self.cam_theta)
        cz = self.cam_r * np.cos(self.cam_phi)
        cam_pos = np.array(
            [self.cam_center[0] + cx, self.cam_center[1] + cy, self.cam_center[2] + cz],
            dtype=np.float32,
        )
        cam_target = self.cam_center.astype(np.float32)
        cam_dir = cam_target - cam_pos
        cam_dir /= np.linalg.norm(cam_dir) + 1e-8
        return cam_pos, cam_dir

    def _render_human(self):
        self._initialize_render()
        if not self.gui.running:
            return

        self._handle_input()
        cam_pos, cam_dir = self._compute_camera()

        # Raymarch into simulator.raymarch_buffer
        self.simulator.render_raymarch(
            ti.Vector(cam_pos.tolist()),
            ti.Vector([0.0, 0.0, 1.0]),  # world up
            ti.Vector(cam_dir.tolist()),
            int(self.show_stock),
            int(self.show_part),
            int(self.show_tool),
        )
        ti.sync()
        self.gui.set_image(self.simulator.raymarch_buffer)

        # HUD: step, loss, reward; toggles
        info = self.last_info or {}
        self.gui.text(
            f"step {self.current_step}/{self.max_cuts}",
            pos=(0.02, 0.97), color=0xFFFFFF, font_size=16,
        )
        self.gui.text(
            f"loss   {info.get('loss', 0.0):+.5f}",
            pos=(0.02, 0.94), color=0xFFFFFF, font_size=16,
        )
        self.gui.text(
            f"reward {info.get('reward', 0.0):+.5f}",
            pos=(0.02, 0.91), color=0xFFFFFF, font_size=16,
        )

        if self.show_help:
            self.gui.text("Controls:", pos=(0.02, 0.20),
                          color=0xFFFF00, font_size=16)
            self.gui.text("  H        toggle help",
                          pos=(0.02, 0.17), color=0xFFFFFF, font_size=14)
            self.gui.text("  LMB drag orbit camera",
                          pos=(0.02, 0.14), color=0xFFFFFF, font_size=14)
            self.gui.text("  R / F    zoom in / out",
                          pos=(0.02, 0.11), color=0xFFFFFF, font_size=14)
            self.gui.text(f"  1/Z      tool   ({self.show_tool})",
                          pos=(0.02, 0.08), color=0xFFFFFF, font_size=14)
            self.gui.text(f"  3/C      stock  ({self.show_stock})",
                          pos=(0.02, 0.05), color=0xFFFFFF, font_size=14)
            self.gui.text(f"  4/V      target ({self.show_part})",
                          pos=(0.02, 0.02), color=0xFFFFFF, font_size=14)
        else:
            self.gui.text("press H for controls",
                          pos=(0.02, 0.02), color=0xAAAAAA, font_size=14)

        self.gui.show()

    def _render_rgb_array(self):
        """Headless raymarch: returns an (H, W, 3) uint8 image."""
        cam_pos, cam_dir = self._compute_camera()
        self.simulator.render_raymarch(
            ti.Vector(cam_pos.tolist()),
            ti.Vector([0.0, 0.0, 1.0]),
            ti.Vector(cam_dir.tolist()),
            int(self.show_stock),
            int(self.show_part),
            int(self.show_tool),
        )
        ti.sync()
        # raymarch_buffer is (W, H, 3) float32 in [0,1]; the shared helper
        # transposes/flips it to standard (H, W, 3) uint8 (top-left origin).
        from algorithms.policy_video import raymarch_buffer_to_rgb
        return raymarch_buffer_to_rgb(self.simulator.raymarch_buffer)

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    # ========================================================================
    # Close
    # ========================================================================

    def close(self):
        if self.gui is not None:
            self.gui.close()
            self.gui = None


if __name__ == "__main__":
    env = CamEnvDiff(resolution=32, max_steps=64, target_shape="sphere", render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        print(f"Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        time.sleep(0.3)
        print(f"Step {info['step']:3d} | "
              f"R={reward:+.4f} | "
              )
    env.close()