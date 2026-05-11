import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import taichi as ti
import time

from simulator.voxel_simulator import *

 
REWARD_W_GOOD     =  100.0
REWARD_W_BAD      = -200.0
REWARD_W_BOUNDARY =    1.0
REWARD_W_PROG     =   50.0
REWARD_W_IDLE     =   -1.0
REWARD_W_HOLDER   =   -5.0

IDLE_THRESHOLD   = 1e-5
K = 10.0

_EMPTY_REWARD_INFO = {
    "step": 0,
    "action": [0, 0, 0],
    "vol": 0.0,
    "reward": 0.0,
    "completed": False,
    "good_cuts": 0.0,
    "bad_cuts": 0.0,
    "boundary": 0.0,
    "progress": 0.0,
    "idle": 0.0,
    "holder": 0.0,
}

class CamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        resolution=32,
        max_steps=512,
        shape = None,
        stock_params = None,
        target_params = None,
        tool_params = None,
        render_mode: Optional[str] = None,
        debug: bool = False,
        use_buffer = True,
    ):
        """
        Args:
            resolution: grid resolution.
            max_steps: episode length cap.
            shape: the shape of the target object.
            stock_params: parameters for the stock object.
            target_params: parameters for the target object.
            tool_params: parameters for the cutting tool.
            render_mode: "human" or "rgb_array".
            debug: Taichi debug mode (slower, better errors).
            use_buffer: if True, maintain a buffer of past states for potential
        """
        super().__init__()

        self.resolution = resolution
        self.dx = 1.0 / resolution
        self.grid_norm = float(resolution ** 3)
        self.boundary_sigma = float(resolution * resolution)

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.debug = debug

        self.shape = shape
        self.stock_params = stock_params
        self.target_params = target_params
        self.tool_params = tool_params

        self.simulator = None
        self.global_step = 0
        self.current_step = 0

        self.action_dims = [3, 3, 3]
        self.action_space = spaces.Discrete(int(np.prod(self.action_dims)))

        self.obs_dims = 3 + (resolution**3) + (resolution**3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dims,), dtype=np.float32
        )

        self.use_buffer = use_buffer
        self.state_buffer = []
        
        self.last_info = dict(_EMPTY_REWARD_INFO)

        # Rendering state
        self.window = None
        self.canvas = None
        self.scene = None
        self.camera = None
        self.gui = None
        self.axes_points = None
        self.axes_colors = None
        self.cam_r = 3.0
        self.cam_theta = -1.57
        self.cam_phi = 1.0
        self.cam_center = None
        self.last_mouse_pos = None
        self.lmb_down = False
        self.show_tool = True
        self.show_holder = True
        self.show_stock = True
        self.show_part = True
        self.show_debug = False
        self.show_raymarch = False
        self.show_help = False

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v

    def _initialize_sim(self, shape: str, stock_params: dict, target_params: dict, tool_params: dict):
        """(Re)build the simulator with concrete params. Called every reset."""
        self.simulator = CNCSimulator(
            resolution=self.resolution,
            shape=shape,
        )
        self.simulator.initialize_stock(half_size=stock_params["half_size"])
        if shape == "sphere":
            self.simulator.initialize_target_sphere(radius=target_params["radius"])
        elif shape == "box":
            self.simulator.initialize_target_box(half_size=target_params["half_size"])
        elif shape == "cylinder":
            self.simulator.initialize_target_cylinder(radius=target_params["radius"], height=target_params["height"])
        elif shape == "pyramid":
            self.simulator.initialize_target_pyramid(half_base=target_params["half_base"], height=target_params["height"])
        else:
            raise ValueError(f"Unknown shape: {shape}")
        self.simulator.initialize_tool(tool_params["tool_pos"], tool_params["radius"], tool_params["height"])

    def _initialize_render(self):
        if self.window is not None:
            return

        self.window = ti.ui.Window("CNC RL Environment", (1024, 768))
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.gui = self.window.get_gui()

        self.camera.position(1.5, 1.5, 1.5)
        self.camera.lookat(0.5, 0.5, 0.5)
        self.camera.up(0, 0, 1)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)

        self.axes_points = ti.Vector.field(3, dtype=ti.f32, shape=6)
        self.axes_colors = ti.Vector.field(3, dtype=ti.f32, shape=6)
        self.axes_points[0] = [0, 0, 0]; self.axes_points[1] = [1, 0, 0]
        self.axes_colors[0] = [1, 0, 0]; self.axes_colors[1] = [1, 0, 0]
        self.axes_points[2] = [0, 0, 0]; self.axes_points[3] = [0, 1, 0]
        self.axes_colors[2] = [0, 1, 0]; self.axes_colors[3] = [0, 1, 0]
        self.axes_points[4] = [0, 0, 0]; self.axes_points[5] = [0, 0, 1]
        self.axes_colors[4] = [0, 0, 1]; self.axes_colors[5] = [0, 0, 1]

        self.grad_points = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.grad_colors = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.move_points = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.move_colors = ti.Vector.field(3, dtype=ti.f32, shape=2)

        self.cam_center = ti.Vector([0.5, 0.5, 0.5])
        self.last_mouse_pos = self.window.get_cursor_pos()

    def _get_tool_pos(self) -> np.ndarray:
        return self.simulator.tool_pos[None].to_numpy().astype(np.float32)

    def _get_sdf_stock(self) -> np.ndarray:
        return np.clip(self.simulator.sdf_stock.to_numpy(), -1.0, 1.0).astype(np.float32)

    def _get_sdf_target(self) -> np.ndarray:
        return np.clip(self.simulator.sdf_target.to_numpy(), -1.0, 1.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self._get_tool_pos(),
            self._get_sdf_stock().flatten(),
            self._get_sdf_target().flatten(),
        ])

    def _calculate_reward(
        self,
        sdf_stock_before: np.ndarray,
        sdf_stock_after:  np.ndarray,
        sdf_target:       np.ndarray,
        tool_pos:         np.ndarray,
    ):
        res = self.resolution
        k_sdf = K * (res / 32.0)         # scale sharpness with resolution      # controls boundary bonus falloff; higher means bonus is more tightly concentrated near the surface

        # ----- Soft inside/outside masks (sigmoid on SDF) -----
        # Numerically-safe sigmoid via clipping the exponent.
        def _sig(x):
            return 1.0 / (1.0 + np.exp(np.clip(x, -50.0, 50.0)))

        inside_stock_before = _sig( k_sdf * sdf_stock_before)
        inside_stock_after  = _sig( k_sdf * sdf_stock_after)
        inside_target       = _sig( k_sdf * sdf_target)
        outside_target      = _sig(-k_sdf * sdf_target)
        material_removed    = inside_stock_before - inside_stock_after

        # ----- 1. Cutting quality -----
        good_raw = float(np.sum(material_removed * outside_target) / self.grid_norm)
        bad_raw  = float(np.sum(material_removed * inside_target)  / self.grid_norm)


        # ----- 2. Progress: excess material eliminated this step -----
        excess_before = float(np.sum(inside_stock_before * outside_target) / self.grid_norm)
        excess_after  = float(np.sum(inside_stock_after  * outside_target) / self.grid_norm)
        progress_raw = excess_before - excess_after

        # ----- 3. Idle penalty -----
        # Smooth gate: ~0 when nothing was touched, ~1 once removal exceeds threshold.
        total_removed_raw = good_raw + bad_raw
        K_IDLE = 1.0 / IDLE_THRESHOLD   # ~1e5, makes the gate flip near the threshold
        idle_gate = 1.0 / (1.0 + np.exp(
            np.clip(-K_IDLE * (total_removed_raw - IDLE_THRESHOLD), -50.0, 50.0)
        ))

        idle_raw = 1.0 - idle_gate # idle_raw in [0, 1]: 1 == idle (no work done), 0 == cutting

        # ----- 4. Boundary bonus at tool tip -----
        ix = int(np.clip(tool_pos[0] * res, 0, res - 1))
        iy = int(np.clip(tool_pos[1] * res, 0, res - 1))
        iz = int(np.clip(tool_pos[2] * res, 0, res - 1))

        target_at_tool = float(sdf_target[ix, iy, iz])
        stock_at_tool_before = float(sdf_stock_before[ix, iy, iz])

        near_target_surface = float(np.exp(-self.boundary_sigma * target_at_tool ** 2))
        stock_presence      = _sig(k_sdf * stock_at_tool_before)

        boundary_raw = float(near_target_surface * stock_presence * idle_gate)

        # ----- 5. Holder penalty -----
        tool_height = float(self.simulator.tool_height[None])
        holder_z_lo = (tool_pos[2] + tool_height)
        holder_z_hi = holder_z_lo + 0.5 * tool_height

        z_lo_idx = int(np.clip(np.floor(holder_z_lo * res), 0, res - 1))
        z_hi_idx = int(np.clip(np.ceil (holder_z_hi * res), 0, res))
        z_hi_idx = max(z_hi_idx, z_lo_idx + 1)

        holder_column = sdf_stock_after[ix, iy, z_lo_idx:z_hi_idx]
        holder_buried = _sig(k_sdf * holder_column)             # in [0,1] per voxel
        holder_raw = float(np.mean(holder_buried)) if holder_column.size else 0.0

        # ----- Apply weights -----
        good_cuts = REWARD_W_GOOD     * good_raw
        bad_cuts  = REWARD_W_BAD      * bad_raw
        progress  = REWARD_W_PROG     * progress_raw
        idle      = REWARD_W_IDLE     * idle_raw
        boundary  = REWARD_W_BOUNDARY * boundary_raw
        holder    = REWARD_W_HOLDER   * holder_raw

        reward = good_cuts + bad_cuts + boundary + progress + idle + holder

        return {
            "reward":    float(reward),
            "good_cuts": float(good_cuts),
            "bad_cuts":  float(bad_cuts),
            "boundary":  float(boundary),
            "progress":  float(progress),
            "idle":      float(idle),
            "holder":    float(holder),
        }

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)

        resume_from_buffer = (
            self.use_buffer
            and len(self.state_buffer) > 0
            and self.np_random.random() < 0.3
        )

        if resume_from_buffer:
            # ---- Resume from a past state ----
            saved = self.state_buffer[self.np_random.integers(len(self.state_buffer))]

            shape = saved["shape"]
            self.simulator = CNCSimulator(
                resolution=self.resolution,
                shape=shape
            )

            self.simulator.sdf_stock.from_numpy(saved["sdf_stock"])
            self.simulator.sdf_target.from_numpy(saved["sdf_target"])
            self.simulator.tool_pos[None] = ti.Vector(saved["tool_pos"])
            self.simulator.tool_radius[None] = saved.get("tool_radius", 0.1)
            self.simulator.tool_height[None] = saved.get("tool_height", 0.2)

        else:
            # ---- Sample a fresh problem ----
            if self.shape is not None:
                shape = self.shape
            else:
                shape = str(self.np_random.choice(["sphere", "box", "cylinder", "pyramid"]))

            # Set stock params
            if self.stock_params is not None:
                stock_params = self.stock_params
            else:
                stock_params = {"half_size": round(0.4 / self.dx) * self.dx}

            # Set target params
            if self.target_params is not None:
                target_params = self.target_params
            else:
                if shape == "sphere":
                    target_params = {
                        "radius": round(float(self.np_random.uniform(0.15, 0.30)) / self.dx) * self.dx,
                    }
                elif shape == "box":
                    target_params = {
                        "half_size": round(float(self.np_random.uniform(0.15, 0.30)) / self.dx) * self.dx,
                    }
                elif shape == "cylinder":
                    target_params = {
                        "radius": round(float(self.np_random.uniform(0.10, 0.25)) / self.dx) * self.dx,
                        "height": round(float(self.np_random.uniform(0.15, 0.30)) / self.dx) * self.dx,
                    }
                else:  # pyramid
                    target_params = {
                        "half_base": round(float(self.np_random.uniform(0.15, 0.30)) / self.dx) * self.dx,
                        "height":    round(float(self.np_random.uniform(0.20, 0.40)) / self.dx) * self.dx,
                    }

            # Set tool params
            if self.tool_params is not None:
                tool_params = self.tool_params
            else:
                tool_params = {
                    "radius": round(float(self.np_random.uniform(0.05, 0.15)) / self.dx) * self.dx,
                    "height": round(float(self.np_random.uniform(0.10, 0.30)) / self.dx) * self.dx,
                    "tool_pos": [
                        round(float(self.np_random.uniform(0.1, 0.9)) / self.dx) * self.dx,
                        round(float(self.np_random.uniform(0.1, 0.9)) / self.dx) * self.dx,
                        round(0.9 / self.dx) * self.dx,
                    ]
                }

            self._initialize_sim(shape, stock_params, target_params, tool_params)

        self.simulator.init_tool_template()
        self.current_step = 0

        obs = self._get_obs()
        info = {"step": 0}
        return obs, info

    def _append_state_buffer(self, state):
        if self.use_buffer:
            self.state_buffer.append(state)
            if len(self.state_buffer) > 500:
                self.state_buffer.pop(0)

    def step(self, action):
        self.global_step += 1
        self.current_step += 1

        x = float((action // 9) - 1)
        y = float(((action // 3) % 3) - 1)
        z = float((action % 3) - 1)

        # --- Snapshot target ---
        sdf_stock_before= self._get_sdf_stock()

        # --- Act: move tool, apply cut ---
        self.simulator.move_tool_one_unit(ti.math.vec3(x, y, z))
        vol_removed = float(self.simulator.apply_cut()[0])

        # --- Snapshot stock AFTER cut---
        tool_pos_after   = self._get_tool_pos()
        sdf_stock_after  = self._get_sdf_stock()       
        sdf_target_after = self._get_sdf_target()    
        obs_after = np.concatenate([
            tool_pos_after,
            sdf_stock_after.flatten(),
            sdf_target_after.flatten(),
        ])

        reward_info = self._calculate_reward(
            sdf_stock_before, 
            sdf_stock_after, 
            sdf_target_after, # same as sdf_target_before
            tool_pos_after
        )

        reward = reward_info["reward"]
        good_cuts = reward_info["good_cuts"]
        bad_cuts = reward_info["bad_cuts"]
        boundary = reward_info["boundary"]
        progress = reward_info["progress"]
        idle = reward_info["idle"]
        holder = reward_info["holder"]


        # --- State buffer for resumable resets ---
        if vol_removed > 0 and self.np_random.random() < 0.1:
            self._append_state_buffer({
                "shape": self.simulator.shape,
                "sdf_stock": sdf_stock_after.copy(),
                "sdf_target": sdf_target_after.copy(),
                "tool_pos": tool_pos_after.copy(),
                "tool_radius": float(self.simulator.tool_radius[None]),
                "tool_height": float(self.simulator.tool_height[None]),
            })

        truncated = self.current_step >= self.max_steps
        terminated = False

        info = {
            "step": self.current_step,
            "action": [int(x), int(y), int(z)],
            "vol": float(vol_removed),
            "reward": float(reward),
            "completed": terminated,

            # Reward component breakdown
            "good_cuts":       good_cuts,
            "bad_cuts":        bad_cuts,
            "boundary":        boundary,
            "progress":        progress,
            "idle":            idle,
            "holder":          holder,
            }

        self.last_info = info
        return obs_after, float(reward), terminated, truncated, info

    # ========================================================================
    # Rendering
    # ========================================================================
    def _handle_input(self):
        for e in self.window.get_events(ti.ui.PRESS):
            key = e.key
            if key == "!": key = "1"
            if key == "@": key = "2"
            if key == "#": key = "3"
            if key == "$": key = "4"
            if key == "%": key = "5"

            if key == ti.ui.LMB:
                self.lmb_down = True
            elif key == ti.ui.ESCAPE:
                self.window.running = False
            elif key in ("h", "H"):
                self.show_help = not self.show_help
            elif key in ("1", "z", "Z"):
                self.show_tool = not self.show_tool
            elif key in ("2", "x", "X"):
                self.show_holder = not self.show_holder
            elif key in ("3", "c", "C"):
                self.show_stock = not self.show_stock
            elif key in ("4", "v", "V"):
                self.show_part = not self.show_part
            elif key in ("5", "b", "B"):
                self.show_debug = not self.show_debug
            elif key in ("m", "M"):
                self.show_raymarch = not self.show_raymarch

        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                self.lmb_down = False

        curr_mouse_pos = self.window.get_cursor_pos()
        if self.lmb_down:
            dx = curr_mouse_pos[0] - self.last_mouse_pos[0]
            dy = curr_mouse_pos[1] - self.last_mouse_pos[1]
            self.cam_theta -= dx * 5.0
            self.cam_phi += dy * 5.0
            self.cam_phi = max(0.01, min(3.14, self.cam_phi))
        self.last_mouse_pos = curr_mouse_pos

        zoom_speed = 0.05
        if self.window.is_pressed("r"):
            self.cam_r = max(0.5, self.cam_r - zoom_speed)
        if self.window.is_pressed("f"):
            self.cam_r = min(10.0, self.cam_r + zoom_speed)

    def _update_camera(self):
        cam_x = self.cam_r * np.sin(self.cam_phi) * np.cos(self.cam_theta)
        cam_y = self.cam_r * np.sin(self.cam_phi) * np.sin(self.cam_theta)
        cam_z = self.cam_r * np.cos(self.cam_phi)
        self.camera.position(self.cam_center.x + cam_x, self.cam_center.y + cam_y, self.cam_center.z + cam_z)
        self.camera.lookat(self.cam_center.x, self.cam_center.y, self.cam_center.z)
        self.camera.up(0, 0, 1)

    def _render_human(self):
        self._initialize_render()
        if not self.window.running:
            return

        self._handle_input()
        self._update_camera()

        try:
            if self.show_stock:
                self.simulator.generate_stock_visualization_mesh()
            if self.show_part:
                self.simulator.generate_target_visualization_mesh()
            self.simulator.update_tool()
            ti.sync()
        except Exception as e:
            print(f"Error during mesh gen: {e}")

        if self.show_help:
            with self.gui.sub_window("Controls", x=0.05, y=0.05, width=0.3, height=0.45):
                self.gui.text(f"H: Toggle Help")
                self.gui.text("LMB Drag: Orbit Camera")
                self.gui.text("R/F: Zoom In/Out")
                self.gui.text(f"1/Z: Toggle Tool ({self.show_tool})")
                self.gui.text(f"2/X: Toggle Holder ({self.show_holder})")
                self.gui.text(f"3/C: Toggle Stock ({self.show_stock})")
                self.gui.text(f"4/V: Toggle Part ({self.show_part})")
                self.gui.text(f"5/B: Toggle Debug ({self.show_debug})")
                self.gui.text(f"M: Toggle Raymarch ({self.show_raymarch})")
                self.gui.text(f"Step: {self.current_step}/{self.max_steps}")

        if self.show_raymarch:
            cam_x = self.cam_r * np.sin(self.cam_phi) * np.cos(self.cam_theta)
            cam_y = self.cam_r * np.sin(self.cam_phi) * np.sin(self.cam_theta)
            cam_z = self.cam_r * np.cos(self.cam_phi)
            cam_pos_vec = ti.Vector([self.cam_center.x + cam_x, self.cam_center.y + cam_y, self.cam_center.z + cam_z])
            cam_dir_vec = ti.Vector([-cam_x, -cam_y, -cam_z]).normalized()
            cam_up_vec = ti.Vector([0.0, 0.0, 1.0])
            self.simulator.render_raymarch(
                cam_pos_vec, cam_up_vec, cam_dir_vec,
                int(self.show_stock), int(self.show_part), int(self.show_tool),
            )
            self.canvas.set_image(self.simulator.raymarch_buffer)
        elif self.show_debug:
            self.simulator.generate_slices()
            self.simulator.compose_debug_view()
            self.canvas.set_image(self.simulator.debug_buffer)
        else:
            self.scene.set_camera(self.camera)
            self.scene.ambient_light((0.5, 0.5, 0.5))

            if self.show_stock:
                count = min(self.simulator.stock_count[None], self.simulator.stock_points.shape[0])
                if count > 0:
                    self.scene.particles(self.simulator.stock_points, radius=0.005,
                                         color=(0.2, 0.8, 0.2), index_count=count)
            if self.show_part:
                count = min(self.simulator.target_count[None], self.simulator.target_points.shape[0])
                if count > 0:
                    self.scene.particles(self.simulator.target_points, radius=0.005,
                                         color=(0.5, 0.5, 1.0), index_count=count)
            if self.show_tool:
                self.scene.particles(self.simulator.tool_points, radius=0.005,
                                     color=(1.0, 0.2, 0.2),
                                     index_count=self.simulator.tool_count[None])
                tool_pos_py = self.simulator.tool_pos[None]
                self.move_points[0] = tool_pos_py
                #self.move_points[1] = tool_pos_py + ti.Vector(self.last_move_dir.tolist()) * 0.15
                self.move_colors[0] = [0, 1, 1]; self.move_colors[1] = [0, 1, 1]
                self.scene.lines(self.grad_points, width=4.0, per_vertex_color=self.grad_colors)
                self.scene.lines(self.move_points, width=4.0, per_vertex_color=self.move_colors)
            if self.show_holder:
                self.scene.particles(self.simulator.holder_points, radius=0.005,
                                     color=(0.2, 0.2, 0.2),
                                     index_count=self.simulator.holder_count[None])

            self.scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
            self.scene.lines(self.axes_points, width=5.0, per_vertex_color=self.axes_colors)
            self.canvas.scene(self.scene)

            if not self.show_help:
                with self.gui.sub_window("Help", x=0.05, y=0.05, width=0.2, height=0.1):
                    self.gui.text("Press 'h' for controls")
                with self.gui.sub_window("Rewards", x=0.05, y=0.15, width=0.3, height=0.25):
                    self.gui.text(f"Total:    {self.last_info.get('reward', 0.0):+.4f}")
                    self.gui.text(f"Progress: {self.last_info.get('progress', 0.0):+.4f}")
                    self.gui.text(f"Boundary: {self.last_info.get('boundary', 0.0):+.4f}")
                    self.gui.text(f"Good:     {self.last_info.get('good_cuts', 0.0):+.4f}")
                    self.gui.text(f"Bad:      {self.last_info.get('bad_cuts', 0.0):+.4f}")
                    self.gui.text(f"Idle:     {self.last_info.get('idle', 0.0):+.4f}")
                    self.gui.text(f"Holder:   {self.last_info.get('holder', 0.0):+.4f}")

        self.window.show()

    def _render_rgb_array(self):
        raise NotImplementedError()

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            self._render_rgb_array()

    def close(self):
        if self.window is not None:
            self.window.running = False
            self.window = None


if __name__ == "__main__":
    env = CamEnv(resolution=32, max_steps=128, render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        time.sleep(0.3)
        print(f"Step {info['step']:3d} | "
              f"R={reward:+.4f} | "
              f"good={info['good_cuts']:+.4f} | "
              f"bad={info['bad_cuts']:+.4f} | "
              f"holder={info['holder']:+.4f} | "
              f"progress={info['progress']:+.4f} | "
              f"boundary={info['boundary']:+.4f} | "
              f"idle={info['idle']:+.4f}")
    env.close()