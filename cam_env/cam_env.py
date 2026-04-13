import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import taichi as ti
import time

from simulator.simulator import CNCSimulator
from cam_env.physics_config import (
    TIME_PENALTY, COMPLETION_BONUS, COMPLETION_THRESHOLD,
    PROXIMITY_COEF_INITIAL, PROXIMITY_ANNEAL_STEPS, PROXIMITY_CLIP,
)

class CamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, resolution=128, max_steps=100, render_mode: Optional[str] = None, debug: bool = False):
        """ Initializes the CAM environment.

        Args:
            resolution (int): The resolution of the simulation grid.
            max_steps (int): The maximum number of steps per episode.
            render_mode (Optional[str]): The mode for rendering the environment.
            debug (bool): If True, enables Taichi debug mode (slower but better errors).
        """
        super().__init__()

         # --- Simulator State ---
        self.resolution = resolution
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.debug = debug

        self.simulator = None
        self.global_step = 0
        self.current_step = 0

        self.action_dims = [3, 3, 3]
        self.action_space = spaces.Discrete(np.prod(self.action_dims))

        # self.obs_dims = 9 + (resolution ** 3) + (resolution ** 3) # tool position (3) + grad_stock (3) + grad_diff (3) + stock SDF + target SDF
        self.obs_dims = 3 + (resolution ** 3) + (resolution ** 3) # tool position + stock SDF + target SDF
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dims,), dtype=np.float32)
        
        self.state_buffer = []

        self.writer = None

        # --- Rendering State ---
        self.window = None
        self.canvas = None
        self.scene = None
        self.camera = None
        self.gui = None

        # Axes visualization
        self.axes_points = None
        self.axes_colors = None

        # Camera orbit state
        self.cam_r = 3.0
        self.cam_theta = -1.57
        self.cam_phi = 1.0
        self.cam_center = None

        # Mouse state
        self.last_mouse_pos = None
        self.lmb_down = False

        # Toggle flags
        self.show_tool = True
        self.show_holder = True
        self.show_stock = True
        self.show_part = True
        self.show_debug = False
        self.show_raymarch = False
        self.show_help = False

        # NEW PARAMS

        # Proximity shaping annealing: linearly decay from initial to 0
        self.proximity_coef_initial = PROXIMITY_COEF_INITIAL
        self.proximity_anneal_steps = PROXIMITY_ANNEAL_STEPS

        # --- Cached excess for reward progress (set in reset/step) ---
        self._prev_excess = 0.0
        
        self.last_grad_diff = np.zeros(3, dtype=np.float32)
        self.last_move_dir = np.zeros(3, dtype=np.float32)
        self.last_info = {}


    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """ Normalizes a vector to unit length; returns zero vector if norm is near zero. """
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v


    def _initialize_sim(self):
        """ Initializes the CNC simulator if not already initialized (lazy initialization). """
        if self.simulator is None:
            self.simulator = CNCSimulator(resolution=self.resolution, debug=self.debug)


    def _initialize_render(self):
        """ Initializes GGUI rendering components (lazy initialization). """
        if self.window is not None:
            return

        self.window = ti.ui.Window("CNC RL Environment", (1024, 768))
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.gui = self.window.get_gui()

        # Camera init
        self.camera.position(1.5, 1.5, 1.5)
        self.camera.lookat(0.5, 0.5, 0.5)
        self.camera.up(0, 0, 1)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)

        # Coordinate axes
        self.axes_points = ti.Vector.field(3, dtype=ti.f32, shape=6)
        self.axes_colors = ti.Vector.field(3, dtype=ti.f32, shape=6)

        # X Axis (Red)
        self.axes_points[0] = [0, 0, 0]
        self.axes_points[1] = [1, 0, 0]
        self.axes_colors[0] = [1, 0, 0]
        self.axes_colors[1] = [1, 0, 0]

        # Y Axis (Green)
        self.axes_points[2] = [0, 0, 0]
        self.axes_points[3] = [0, 1, 0]
        self.axes_colors[2] = [0, 1, 0]
        self.axes_colors[3] = [0, 1, 0]

        # Z Axis (Blue)
        self.axes_points[4] = [0, 0, 0]
        self.axes_points[5] = [0, 0, 1]
        self.axes_colors[4] = [0, 0, 1]
        self.axes_colors[5] = [0, 0, 1]

        # Reward vectors
        self.grad_points = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.grad_colors = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.move_points = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.move_colors = ti.Vector.field(3, dtype=ti.f32, shape=2)

        # Camera state
        self.cam_center = ti.Vector([0.5, 0.5, 0.5])
        self.last_mouse_pos = self.window.get_cursor_pos()


    # More methods
    def _get_obs(self) -> Dict[str, Any]:
        """ Retrieves the current state of the tool, stock, and target from the simulator."""

        fov_scale = np.tan(np.pi / 4.0)
        width, height = 1024, 768
        aspect_ratio = width / height

        tool_pos = self.simulator.tool_pos[None].to_numpy().astype(np.float32)
        sdf_stock = np.clip(self.simulator.sdf_stock.to_numpy(), -1.0, 1.0).astype(np.float32).flatten()
        sdf_target = np.clip(self.simulator.sdf_target.to_numpy(), -1.0, 1.0).astype(np.float32).flatten()
        return np.concatenate([tool_pos, sdf_stock, sdf_target])


    def _calculate_reward(self, sdf_stock_before, sdf_stock_after, sdf_target, tool_pos):
        res = self.resolution
        k = 10

        ix = int(np.clip(tool_pos[0] * res, 0, res - 1))
        iy = int(np.clip(tool_pos[1] * res, 0, res - 1))
        iz = int(np.clip(tool_pos[2] * res, 0, res - 1))

        # --- 1. Cutting Reward ---

        inside_stock_before = 1.0 / (1.0 + np.exp(k * sdf_stock_before))
        inside_stock_after = 1.0 / (1.0 + np.exp(k * sdf_stock_after))

        inside_target = 1.0 / (1.0 + np.exp(k * sdf_target))
        outside_target = 1.0 / (1.0 + np.exp(-k * sdf_target))

        material_removed = inside_stock_before - inside_stock_after 
        good_cuts = float(np.sum(material_removed * outside_target))
        bad_cuts = float(np.sum(material_removed * inside_target))

        # --- 2. Boundary Proximity Bonus ---
        target_at_tool = float(sdf_target[ix, iy, iz])
        stock_at_tool = float(sdf_stock_after[ix, iy, iz])
        
        near_target_surface = np.exp(-8.0 * target_at_tool ** 2)
        stock_presence = 1.0 / (1.0 + np.exp(k * (stock_at_tool - 0.05))) 
        
        material_was_cut = float(np.sum(np.clip(material_removed, 0, None)))
        cutting_mask = 1.0 / (1.0 + np.exp(-k * (material_was_cut - 0.1)))
        boundary_bonus = 0.3 * near_target_surface * stock_presence * cutting_mask
    
        # --- 3. Global Progress Reward ---
        excess_before = float(np.sum(inside_stock_before * outside_target))
        excess_after = float(np.sum(inside_stock_after * outside_target))
        progress_reward = excess_before - excess_after

        # --- 4. Idle Penalty ---
        total_removed = float(np.sum(np.clip(material_removed, 0, None)))
        idle_penalty = -0.2 + 0.2 * (1.0 / (1.0 + np.exp(-k * (total_removed - 0.1))))


        # --- 5. Holder Collision Penalty ---
        tool_radius = self.simulator.tool_radius[None]
        tool_height = self.simulator.tool_height[None]

        holder_z = tool_pos[2] + tool_height
        holder_z_idx = int(np.clip(holder_z * res, 0, res - 1))
        stock_at_holder = float(sdf_stock_after[ix, iy, holder_z_idx])
        holder_inside_stock = 1.0 / (1.0 + np.exp(k * stock_at_holder))

        holder_penalty = -1.0 * holder_inside_stock

        # Reward calculation
        reward = 10.0 * good_cuts - 10.0 * bad_cuts + 0.5 * boundary_bonus + progress_reward + idle_penalty + holder_penalty
        return reward


    def reset(self, seed: Optional[int] = None):
        """ Resets the environment to an initial state and returns an initial observation.
        
        Args:
            seed (Optional[int]): An optional seed for random number generation.

        Returns:
            obs (Dict[str, Any]): The initial observation of the environment.
            info (Dict[str, Any]): Additional information about the reset.
        """
        super().reset(seed=seed)

        self._initialize_sim()

        # Small probability of loading a previously saved state
        if len(self.state_buffer) > 0 and self.np_random.random() < 0.3:
            saved = self.state_buffer[self.np_random.integers(len(self.state_buffer))]
            self.simulator.sdf_stock.from_numpy(saved["sdf_stock"])
            self.simulator.sdf_target.from_numpy(saved["sdf_target"])
            x = float(saved["tool_pos"][0])
            y = float(saved["tool_pos"][1])
            z = float(saved["tool_pos"][2])
            radius = float(saved["tool_radius"])
            height = float(saved["tool_height"])
            self.simulator.initialize_tool_primitive([x, y, z], radius, height)
        # Initialize new random state
        else:
            # Initialize tool
            x = float(self.np_random.uniform(0.1, 0.9))
            y = float(self.np_random.uniform(0.1, 0.9))
            z = 0.9

            radius = float(self.np_random.uniform(0.05, 0.15))
            height = float(self.np_random.uniform(0.2, 0.4))
            self.simulator.initialize_tool_primitive([x, y, z], radius, height)

            # Initialize stock
            self.simulator.initialize_stock_primitive(half_size=0.4)
        
            # Initialize target with random shape and size
            shape = self.np_random.choice(["sphere", "cube", "cylinder", "pyramid"])
            if shape == "sphere":
                r = float(self.np_random.uniform(0.15, 0.3))
                self.simulator.initialize_target_sphere_primitive(r)
            elif shape == "cube":
                s = float(self.np_random.uniform(0.15, 0.3))
                self.simulator.initialize_target_cube(s)
            elif shape == "cylinder":
                r = float(self.np_random.uniform(0.1, 0.25))
                h = float(self.np_random.uniform(0.15, 0.3))
                self.simulator.initialize_target_cylinder(r, h)
            elif shape == "pyramid":
                b = float(self.np_random.uniform(0.15, 0.3))
                h = float(self.np_random.uniform(0.2, 0.4))
                self.simulator.initialize_target_pyramid(b, h)

        # Initialize tool visualization (must be after tool primitive init)
        self.simulator.init_tool_template()

        self.current_step = 0

        # Build initial obs on GPU (also computes initial excess)
        obs = self._get_obs()
        self._prev_excess = float(self.simulator.excess_field[None])
        info = {"step": 0}
        
        return obs, info


    def _holder_hit_stock(self):
        return self.simulator.check_holder_collision() == 1


    def _tool_cuts_into_target(self):
        return self.simulator.check_tool_intersects_target() == 1


    def step(self, action):
        """ Executes one time step within the environment. 
        Args:
            action (np.ndarray): The action to take, discrete decomposed into 3 dimensions (x, y, z) with values in {0, 1, 2} corresponding to movement of -1, 0, +1 units respectively.
            
        Returns:
            obs (Dict[str, Any]): The observation after taking the action.
            reward (float): The reward obtained from taking the action.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (Dict[str, Any]): Additional information about the step.
        """
        self.global_step += 1
        self.current_step += 1

        x = (action // 9) - 1
        y = ((action // 3) % 3) - 1
        z = (action % 3) - 1

        # State before cut
        sdf_target = self.simulator.sdf_target.to_numpy()
        sdf_stock_before = self.simulator.sdf_stock.to_numpy()

        # Cutting action
        self.simulator.move_tool_one_unit(ti.math.vec3(x, y, z))
        vol_removed = self.simulator.apply_cut()

        # State after cut
        tool_pos_after = self.simulator.tool_pos[None].to_numpy()
        sdf_stock_after = self.simulator.sdf_stock.to_numpy()

        tool_cut_stock = vol_removed > 0.0
        holder_hit = self._holder_hit_stock()
        tool_cut_target = self._tool_cuts_into_target()


        obs = self._get_obs()
        reward = self._calculate_reward(sdf_stock_before, sdf_stock_after, sdf_target, tool_pos_after)
        if vol_removed > 0 and self.np_random.random() < 0.1:
            self.state_buffer.append({
                "sdf_stock": sdf_stock_after.copy(),
                "sdf_target": sdf_target.copy(),
                "tool_pos": tool_pos_after.copy(),
                "tool_radius": float(self.simulator.tool_radius[None]),
                "tool_height": float(self.simulator.tool_height[None]),
            })
            if len(self.state_buffer) > 500:
                self.state_buffer.pop(0)

        #if self.writer:
        #    self.writer.add_scalar("charts/env_reward", reward, self.global_step)
        # print(f"Step {self.current_step}: action = [{x}, {y}, {z}], reward = {reward}")

        truncated = self.current_step >= self.max_steps
        terminated = False
        info = {"step": self.current_step, "action": [x, y, z], "vol": vol_removed, "tool_cut_target": tool_cut_target}

        return obs, reward, terminated, truncated, info


    # Rendering Methods
    def _handle_input(self):
        """ Handles keyboard and mouse input for orbit camera control and toggles. """
        # Event handling (key presses)
        for e in self.window.get_events(ti.ui.PRESS):
            key = e.key
            # Handle shifted keys
            if key == '!': key = '1'
            if key == '@': key = '2'
            if key == '#': key = '3'
            if key == '$': key = '4'
            if key == '%': key = '5'

            if key == ti.ui.LMB:
                self.lmb_down = True
            elif key == ti.ui.ESCAPE:
                self.window.running = False
            elif key == 'h' or key == 'H':
                self.show_help = not self.show_help
            elif key == '1' or key == 'z' or key == 'Z':
                self.show_tool = not self.show_tool
            elif key == '2' or key == 'x' or key == 'X':
                self.show_holder = not self.show_holder
            elif key == '3' or key == 'c' or key == 'C':
                self.show_stock = not self.show_stock
            elif key == '4' or key == 'v' or key == 'V':
                self.show_part = not self.show_part
            elif key == '5' or key == 'b' or key == 'B':
                self.show_debug = not self.show_debug
            elif key == 'm' or key == 'M':
                self.show_raymarch = not self.show_raymarch

        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                self.lmb_down = False

        # LMB drag for orbit
        curr_mouse_pos = self.window.get_cursor_pos()
        if self.lmb_down:
            dx = curr_mouse_pos[0] - self.last_mouse_pos[0]
            dy = curr_mouse_pos[1] - self.last_mouse_pos[1]
            self.cam_theta -= dx * 5.0
            self.cam_phi += dy * 5.0
            self.cam_phi = max(0.01, min(3.14, self.cam_phi))
        self.last_mouse_pos = curr_mouse_pos

        # Zoom in/out with R/F keys
        zoom_speed = 0.05
        if self.window.is_pressed('r'):
            self.cam_r = max(0.5, self.cam_r - zoom_speed)
        if self.window.is_pressed('f'):
            self.cam_r = min(10.0, self.cam_r + zoom_speed)


    def _update_camera(self):
        """ Updates camera position based on orbit state. """
        cam_x = self.cam_r * np.sin(self.cam_phi) * np.cos(self.cam_theta)
        cam_y = self.cam_r * np.sin(self.cam_phi) * np.sin(self.cam_theta)
        cam_z = self.cam_r * np.cos(self.cam_phi)

        self.camera.position(
            self.cam_center.x + cam_x,
            self.cam_center.y + cam_y,
            self.cam_center.z + cam_z
        )
        self.camera.lookat(self.cam_center.x, self.cam_center.y, self.cam_center.z)
        self.camera.up(0, 0, 1)


    def _render_human(self):
        """ Interactive rendering with GGUI window. """
        self._initialize_render()

        if not self.window.running:
            return

        # Handle input
        self._handle_input()
        self._update_camera()

        # Generate visualization meshes
        try:
            if self.show_stock:
                self.simulator.generate_stock_visualization_mesh()
            if self.show_part:
                self.simulator.generate_target_visualization_mesh()
            self.simulator.update_tool(self.simulator.tool_pos[None])
            ti.sync()
        except Exception as e:
            print(f"Error during mesh gen: {e}")

        # Draw GUI overlay
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
            self.simulator.render_raymarch(cam_pos_vec, cam_up_vec, cam_dir_vec, int(self.show_stock), int(self.show_part), int(self.show_tool))
            self.canvas.set_image(self.simulator.raymarch_buffer)
            
            with self.gui.sub_window("Raymarch Info", x=0.5, y=0.05, width=0.45, height=0.2):
                self.gui.text("Raymarching View")
                self.gui.text("High-quality SDF rendering")

        elif self.show_debug:
            # 2D slice debug view
            self.simulator.generate_slices()
            self.simulator.compose_debug_view()
            self.canvas.set_image(self.simulator.debug_buffer)

            with self.gui.sub_window("Debug Info", x=0.5, y=0.05, width=0.45, height=0.2):
                self.gui.text("TL: XY (Top) | TR: XZ (Front)")
                self.gui.text("BL: YZ (Side)")
                self.gui.text("Green: Tool Radius")
                self.gui.text("Blue: Stock (SDF < 0)")
        else:
            # 3D scene rendering
            self.scene.set_camera(self.camera)
            self.scene.ambient_light((0.5, 0.5, 0.5))

            # Draw Stock
            if self.show_stock:
                count = min(self.simulator.stock_count[None], self.simulator.stock_points.shape[0])
                if count > 0:
                    self.scene.particles(
                        self.simulator.stock_points,
                        radius=0.005,
                        color=(0.2, 0.8, 0.2),
                        index_count=count
                    )

            # Draw Target
            if self.show_part:
                count = min(self.simulator.target_count[None], self.simulator.target_points.shape[0])
                if count > 0:
                    self.scene.particles(
                        self.simulator.target_points,
                        radius=0.005,
                        color=(0.5, 0.5, 1.0),
                        index_count=count
                    )

            # Draw Tool
            if self.show_tool:
                self.scene.particles(
                    self.simulator.tool_points,
                    radius=0.005,
                    color=(1.0, 0.2, 0.2),
                    index_count=self.simulator.tool_count[None]
                )
                
                # Draw Vectors
                tool_pos_py = self.simulator.tool_pos[None]
                # Gradient Vector (Yellow)
                self.grad_points[0] = tool_pos_py
                self.grad_points[1] = tool_pos_py + ti.Vector(self.last_grad_diff.tolist()) * 0.15
                self.grad_colors[0] = [1, 1, 0]
                self.grad_colors[1] = [1, 1, 0]
                
                # Movement Vector (Cyan)
                self.move_points[0] = tool_pos_py
                self.move_points[1] = tool_pos_py + ti.Vector(self.last_move_dir.tolist()) * 0.15
                self.move_colors[0] = [0, 1, 1]
                self.move_colors[1] = [0, 1, 1]
                
                self.scene.lines(self.grad_points, width=4.0, per_vertex_color=self.grad_colors)
                self.scene.lines(self.move_points, width=4.0, per_vertex_color=self.move_colors)

            # Draw Holder
            if self.show_holder:
                self.scene.particles(
                    self.simulator.holder_points,
                    radius=0.005,
                    color=(0.2, 0.2, 0.2),
                    index_count=self.simulator.holder_count[None]
                )

            self.scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

            # Draw coordinate axes
            self.scene.lines(self.axes_points, width=5.0, per_vertex_color=self.axes_colors)

            self.canvas.scene(self.scene)

            if not self.show_help:
                with self.gui.sub_window("Help", x=0.05, y=0.05, width=0.2, height=0.1):
                    self.gui.text("Press 'h' for controls")
                    
                with self.gui.sub_window("Rewards", x=0.05, y=0.15, width=0.3, height=0.25):
                    self.gui.text(f"Total: {self.last_info.get('reward_total', 0.0):.4f}")
                    self.gui.text(f"Progress: {self.last_info.get('reward_progress', 0.0):.4f}")
                    self.gui.text(f"Prox Bonus: {self.last_info.get('reward_prox_bonus', 0.0):.4f}")
                    self.gui.text(f"Time Penalty: {self.last_info.get('reward_time_penalty', 0.0):.4f}")
                    self.gui.text(f"Completion: {self.last_info.get('reward_completion_bonus', 0.0):.4f}")
                    self.gui.text(f"Coef: {self.last_info.get('proximity_coef', 0.0):.4f}")

        self.window.show()


    def _render_rgb_array(self):
        raise NotImplementedError()


    def render(self):
        """ Renders the environment. """

        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            self._render_rgb_array()


    # Close
    def close(self):
        """ Cleans up the environment resources. """
        if self.window is not None:
            self.window.running = False
            self.window = None


if __name__ == "__main__":
    # Test the environment with random actions
    env = CamEnv(resolution=8, max_steps=512, render_mode="human")
    obs, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(type(reward))
        done = terminated or truncated

        env.render()
        time.sleep(0.3)

        print(f"Step: {info['step']}, Reward: {reward}, Info: {info}")

    env.close()