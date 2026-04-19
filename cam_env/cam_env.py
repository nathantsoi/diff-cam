import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import taichi as ti
import time

from simulator.voxel_simulator import (
    CNCSimulator,
    # Import unified reward constants so env and sim cannot drift apart.
    REWARD_K_SIGMOID,
    REWARD_K_IDLE,
    REWARD_W_GOOD,
    REWARD_W_BAD,
    REWARD_W_BOUND,
    REWARD_W_PROG,
    REWARD_W_IDLE,
    REWARD_W_HOLDER,
    BOUNDARY_SIGMA,
    BOUNDARY_STOCK_OFFSET,
    IDLE_THRESHOLD,
)


class CamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        resolution=32,
        max_steps=512,
        render_mode: Optional[str] = None,
        debug: bool = False,
        debug_gradients: bool = False,
    ):
        """
        Args:
            resolution: grid resolution.
            max_steps: episode length cap.
            render_mode: "human" or "rgb_array".
            debug: Taichi debug mode (slower, better errors).
            debug_gradients: if True, compute per-component reward gradients
                every step via autodiff (6 tape runs — expensive). If False,
                info dict will not contain 'grad_mag_*' or 'grad_x/y/z'.
        """
        super().__init__()

        self.resolution = resolution
        self.dx = 1.0 / resolution
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.debug = debug
        self.debug_gradients = debug_gradients

        self.simulator = None
        self.global_step = 0
        self.current_step = 0

        self.action_dims = [3, 3, 3]
        self.action_space = spaces.Discrete(int(np.prod(self.action_dims)))

        self.obs_dims = 3 + (resolution**3) + (resolution**3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dims,), dtype=np.float32
        )

        self.state_buffer = []

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

        self.last_grad_diff = np.zeros(3, dtype=np.float32)
        self.last_move_dir = np.zeros(3, dtype=np.float32)
        self.last_info = {}

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v

    def _initialize_sim(self):
        if self.simulator is None:
            self.simulator = CNCSimulator(resolution=self.resolution, debug=self.debug)

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

    def _get_obs(self) -> np.ndarray:
        """Tool pos + clipped flattened stock + clipped flattened target."""
        tool_pos = self.simulator.tool_pos[None].to_numpy().astype(np.float32)
        sdf_stock = (
            np.clip(self.simulator.sdf_stock.to_numpy(), -1.0, 1.0)
            .astype(np.float32).flatten()
        )
        sdf_target = (
            np.clip(self.simulator.sdf_target.to_numpy(), -1.0, 1.0)
            .astype(np.float32).flatten()
        )
        return np.concatenate([tool_pos, sdf_stock, sdf_target])

    def _calculate_reward(
        self,
        sdf_stock_before: np.ndarray,
        sdf_stock_after: np.ndarray,
        sdf_target: np.ndarray,
        tool_pos: np.ndarray,
    ):
        res = self.resolution
        k = REWARD_K_SIGMOID
        grid_norm = float(res ** 3)

        # 1. Cutting reward (voxel-wise, summed, normalized by grid)
        inside_stock_before = 1.0 / (1.0 + np.exp(k * sdf_stock_before))
        inside_stock_after  = 1.0 / (1.0 + np.exp(k * sdf_stock_after))
        inside_target       = 1.0 / (1.0 + np.exp(k * sdf_target))
        outside_target      = 1.0 / (1.0 + np.exp(-k * sdf_target))

        material_removed = inside_stock_before - inside_stock_after
        good_cuts = float(np.sum(material_removed * outside_target) / grid_norm)
        bad_cuts  = float(np.sum(material_removed * inside_target)  / grid_norm)

        # 2. Progress (same normalization)
        excess_before = float(np.sum(inside_stock_before * outside_target) / grid_norm)
        excess_after  = float(np.sum(inside_stock_after  * outside_target) / grid_norm)
        progress = excess_before - excess_after

        # 3. Idle — uses normalized total_removed, same as simulator
        total_removed = good_cuts + bad_cuts
        idle = -0.2 + 0.2 * (1.0 / (1.0 + np.exp(-REWARD_K_IDLE * (total_removed - IDLE_THRESHOLD))))

        # 4. Boundary bonus (point-sampled at tool tip)
        ix = int(np.clip(tool_pos[0] * res, 0, res - 1))
        iy = int(np.clip(tool_pos[1] * res, 0, res - 1))
        iz = int(np.clip(tool_pos[2] * res, 0, res - 1))
        target_at_tool = float(sdf_target[ix, iy, iz])
        stock_at_tool  = float(sdf_stock_after[ix, iy, iz])

        near_target_surface = np.exp(-BOUNDARY_SIGMA * target_at_tool ** 2)
        stock_presence = 1.0 / (1.0 + np.exp(k * (stock_at_tool - BOUNDARY_STOCK_OFFSET)))
        cutting_mask = 1.0 / (1.0 + np.exp(-REWARD_K_IDLE * (total_removed - IDLE_THRESHOLD)))
        boundary = float(near_target_surface * stock_presence * cutting_mask)

        # 5. Holder penalty (point-sampled at holder base)
        tool_height = float(self.simulator.tool_height[None])
        holder_z = tool_pos[2] + tool_height
        holder_z_idx = int(np.clip(holder_z * res, 0, res - 1))
        # Use sdf_stock_before here to match the simulator (post-cut is the same
        # anywhere the cut didn't reach, and the holder base is above the tool
        # so the cut doesn't touch it — so pre vs post is equivalent there).
        stock_at_holder = float(sdf_stock_before[ix, iy, holder_z_idx])
        holder_inside = 1.0 / (1.0 + np.exp(k * stock_at_holder))
        holder = float(-holder_inside)  # negative contribution

        reward = (
            REWARD_W_GOOD   * good_cuts
          - REWARD_W_BAD    * bad_cuts
          + REWARD_W_BOUND  * boundary
          + REWARD_W_PROG   * progress
          + REWARD_W_IDLE   * idle
          + REWARD_W_HOLDER * holder
        )
        return {
            "reward":   float(reward),
            "good":     good_cuts,
            "bad":      bad_cuts,
            "boundary": boundary,
            "progress": progress,
            "idle":     idle,
            "holder":   holder,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Gymnasium-compliant reset (accepts options)."""
        super().reset(seed=seed)
        self._initialize_sim()

        if len(self.state_buffer) > 0 and self.np_random.random() < 0.3:
            saved = self.state_buffer[self.np_random.integers(len(self.state_buffer))]
            self.simulator.sdf_stock.from_numpy(saved["sdf_stock"])
            self.simulator.sdf_target.from_numpy(saved["sdf_target"])
            x = float(saved["tool_pos"][0])
            y = float(saved["tool_pos"][1])
            z = float(saved["tool_pos"][2])
            self.simulator.initialize_tool(
                [x, y, z], float(saved["tool_radius"]), float(saved["tool_height"])
            )
        else:
            x = round(float(self.np_random.uniform(0.1, 0.9)) / self.dx) * self.dx
            y = round(float(self.np_random.uniform(0.1, 0.9)) / self.dx) * self.dx
            z = round(0.9 / self.dx) * self.dx
            radius = round(float(self.np_random.uniform(0.05, 0.15)) / self.dx) * self.dx
            height = round(float(self.np_random.uniform(0.2, 0.4)) / self.dx) * self.dx
            if self.debug:
                print(f"Init tool at {[x, y, z]}, r={radius:.4f}, h={height:.4f}")
            self.simulator.initialize_tool([x, y, z], radius, height)
            self.simulator.initialize_stock(half_size=(round(0.4 / self.dx) * self.dx))

            shape = self.np_random.choice(["sphere", "cube", "cylinder", "pyramid"])
            if shape == "sphere":
                r = round(float(self.np_random.uniform(0.15, 0.3)) / self.dx) * self.dx
                self.simulator.initialize_target_sphere(r)
            elif shape == "cube":
                s = round(float(self.np_random.uniform(0.15, 0.3)) / self.dx) * self.dx
                self.simulator.initialize_target_cube(s)
            elif shape == "cylinder":
                r = round(float(self.np_random.uniform(0.1, 0.25)) / self.dx) * self.dx
                h = round(float(self.np_random.uniform(0.15, 0.3)) / self.dx) * self.dx
                self.simulator.initialize_target_cylinder(r, h)
            elif shape == "pyramid":
                b = round(float(self.np_random.uniform(0.15, 0.3)) / self.dx) * self.dx
                h = round(float(self.np_random.uniform(0.2, 0.4)) / self.dx) * self.dx
                self.simulator.initialize_target_pyramid(b, h)

        self.simulator.init_tool_template()
        self.current_step = 0

        obs = self._get_obs()
        info = {"step": 0, "excess": self.simulator.compute_excess()}
        return obs, info

    def _holder_hit_stock(self):
        return self.simulator.check_holder_collision() == 1

    def _tool_cuts_into_target(self):
        return self.simulator.check_tool_intersects_target() == 1

    def step(self, action):
        self.global_step += 1
        self.current_step += 1

        x = (action // 9) - 1
        y = ((action // 3) % 3) - 1
        z = (action % 3) - 1

        # --- Snapshot stock BEFORE cut (for autodiff to see pre-cut state) ---
        self.simulator.snapshot_stock()
        sdf_target_np = self.simulator.sdf_target.to_numpy()
        sdf_stock_before_np = self.simulator.sdf_stock.to_numpy()

        # --- Act: move tool, apply cut ---
        self.simulator.move_tool_one_unit(ti.math.vec3(x, y, z))
        vol_removed = self.simulator.apply_cut()

        # --- Gradients (optional, expensive: 7 tape runs) ---
        # Must be called while sdf_stock_before still holds the pre-cut snapshot.
        # snapshot_stock() was called above, and we haven't overwritten it.
        reward_info = self.simulator.compute_reward_and_gradients()
        grad = reward_info["grad"]
        reward = reward_info["reward"]
        good_cuts = reward_info["good_cuts"]
        bad_cuts = reward_info["bad_cuts"]
        boundary = reward_info["boundary"]
        progress = reward_info["progress"]
        idle = reward_info["idle"]
        holder = reward_info["holder"]

        # --- Read post-cut state ---
        tool_pos_after = self.simulator.tool_pos[None].to_numpy()
        sdf_stock_after_np = self.simulator.sdf_stock.to_numpy()

        obs = self._get_obs()

        # --- Reward (unified formulation; matches simulator's autodiff target) ---
        # reward_components = self._calculate_reward(
        #     sdf_stock_before_np, sdf_stock_after_np, sdf_target_np, tool_pos_after
        # )
        # reward = reward_components["reward"]
        

        # # --- Update render vectors (gradient direction vs. chosen move) ---
        # if grad is not None:
        #     self.last_grad_diff = self._normalize(grad).astype(np.float32)
        # self.last_move_dir = self._normalize(
        #     np.array([x, y, z], dtype=np.float32)
        # ).astype(np.float32)

        # --- State buffer for resumable resets ---
        if vol_removed > 0 and self.np_random.random() < 0.1:
            self.state_buffer.append({
                "sdf_stock": sdf_stock_after_np.copy(),
                "sdf_target": sdf_target_np.copy(),
                "tool_pos": tool_pos_after.copy(),
                "tool_radius": float(self.simulator.tool_radius[None]),
                "tool_height": float(self.simulator.tool_height[None]),
            })
            if len(self.state_buffer) > 500:
                self.state_buffer.pop(0)

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

            "grad_x": float(grad[0]),
            "grad_y": float(grad[1]),
            "grad_z": float(grad[2]),
            "grad_magnitude": float(np.linalg.norm(grad)),
            }

        self.last_info = info
        return obs, float(reward), terminated, truncated, info

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
            self.simulator.update_tool(self.simulator.tool_pos[None])
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
                self.grad_points[0] = tool_pos_py
                self.grad_points[1] = tool_pos_py + ti.Vector(self.last_grad_diff.tolist()) * 0.15
                self.grad_colors[0] = [1, 1, 0]; self.grad_colors[1] = [1, 1, 0]
                self.move_points[0] = tool_pos_py
                self.move_points[1] = tool_pos_py + ti.Vector(self.last_move_dir.tolist()) * 0.15
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
    env = CamEnv(resolution=32, max_steps=128, render_mode="human", debug_gradients=True)
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        time.sleep(0.3)
        print(f"Step {info['step']:3d} | R={reward:+.4f} | vol={info['vol']:.1f} | "
              f"good={info['good_cuts']:+.4f} bad={info['bad_cuts']:+.4f} "
              f"holder={info['holder']:+.4f} | "
              f"|∇R|={info.get('grad_magnitude', 0.0):.4f}")
    env.close()