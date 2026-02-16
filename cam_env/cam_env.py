import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import taichi as ti

from simulator.simulator import CNCSimulator

class CamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, resolution=128, max_steps=100, render_mode: Optional[str] = None):
        """ Initializes the CAM environment.

        Args:
            resolution (int): The resolution of the simulation grid.
            max_steps (int): The maximum number of steps per episode.
            render_mode (Optional[str]): The mode for rendering the environment.
        """
        super().__init__()

         # --- Simulator State ---
        self.resolution = resolution
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.simulator = None
        self.current_step = 0

        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self.observation_space = spaces.Dict({
            "tool_position": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            "sdf_stock": spaces.Box(low=-1.0, high=1.0, shape=(resolution, resolution, resolution), dtype=np.float32),
            "sdf_target": spaces.Box(low=-1.0, high=1.0, shape=(resolution, resolution, resolution), dtype=np.float32),
        })

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
        self.rmb_down = False

        # Toggle flags
        self.show_tool = True
        self.show_holder = True
        self.show_stock = True
        self.show_part = True
        self.show_debug = False
        self.show_help = False



    def _initialize_sim(self):
        """ Initializes the CNC simulator if not already initialized (lazy initialization). """
        if self.simulator is None:
            self.simulator = CNCSimulator(resolution=self.resolution)


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

        # Camera state
        self.cam_center = ti.Vector([0.5, 0.5, 0.5])
        self.last_mouse_pos = self.window.get_cursor_pos()


    def _get_obs(self) -> Dict[str, Any]:
        """ Retrieves the current state of the tool, stock, and target from the simulator.

        Returns:
            Dict[str, Any]: The current observation including tool position and SDFs.
        """ 

        return {
            "tool_position": self.simulator.tool_pos[None].to_numpy().astype(np.float32),
            "sdf_stock": np.clip(self.simulator.sdf_stock.to_numpy(), -1.0, 1.0).astype(np.float32),
            "sdf_target": np.clip(self.simulator.sdf_target.to_numpy(), -1.0, 1.0).astype(np.float32),
        }
    

    def _calculate_reward(self) -> float:
        """ Calculates the reward based on the current state of the stock and target.

        Returns:
            float: The calculated reward.
        """
        sdf_stock = self.simulator.sdf_stock.to_numpy()
        sdf_target = self.simulator.sdf_target.to_numpy()

        # Reward is negative L1 distance between stock and target SDFs
        reward = -np.mean(np.abs(sdf_stock - sdf_target))
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
        self.simulator.initialize_stock_primitive()
        self.simulator.initialize_target_primitive()
        self.simulator.initialize_tool_primitive()
        self.simulator.init_tool_template()

        self.current_step = 0

        obs = self._get_obs()
        info = {"step": 0}
        
        return obs, info


    def _holder_hit_stock(self):
        return self.simulator.check_holder_collision() == 1


    def _tool_cuts_into_target(self):
        return self.simulator.check_tool_intersects_target() == 1


    def step(self, action):
        """ Executes one time step within the environment. 
        Args:
            action (np.ndarray): An array of shape (3,) with values in {0,1,2} representing tool movement directions.
            
        Returns:
            obs (Dict[str, Any]): The observation after taking the action.
            reward (float): The reward obtained from taking the action.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (Dict[str, Any]): Additional information about the step.
        """
        self.current_step += 1
        
        direction = action.astype(np.float32) - 1.0  # Map {0,1,2} to {-1,0,1}
        self.simulator.move_tool_one_unit(ti.math.vec3(direction[0], direction[1], direction[2])) # Currently does not handle collisons or out-of-bounds
        force = self.simulator.apply_cut()

        obs = self._get_obs()
        reward = self._calculate_reward()

        truncated = self.current_step >= self.max_steps
        # terminated condition holder hits stock or cut into target
        terminated = False
        info = {"step": self.current_step}

        return obs, reward, terminated, truncated, info


    def _handle_input(self):
        """ Handles keyboard and mouse input for camera control and toggles. """
        # Event handling (key presses)
        for e in self.window.get_events(ti.ui.PRESS):
            key = e.key
            # Handle shifted keys
            if key == '!': key = '1'
            if key == '@': key = '2'
            if key == '#': key = '3'
            if key == '$': key = '4'
            if key == '%': key = '5'

            if key == ti.ui.RMB:
                self.rmb_down = True
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

        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.RMB:
                self.rmb_down = False

        # Mouse drag for orbit
        curr_mouse_pos = self.window.get_cursor_pos()
        if self.rmb_down:
            dx = curr_mouse_pos[0] - self.last_mouse_pos[0]
            dy = curr_mouse_pos[1] - self.last_mouse_pos[1]
            self.cam_theta -= dx * 5.0
            self.cam_phi += dy * 5.0
            self.cam_phi = max(0.01, min(3.14, self.cam_phi))
        self.last_mouse_pos = curr_mouse_pos

        # WASD pan
        pan_speed = 0.02
        if self.window.is_pressed('w'):
            self.cam_center.x += np.cos(self.cam_theta) * pan_speed
            self.cam_center.y += np.sin(self.cam_theta) * pan_speed
        if self.window.is_pressed('s'):
            self.cam_center.x -= np.cos(self.cam_theta) * pan_speed
            self.cam_center.y -= np.sin(self.cam_theta) * pan_speed
        if self.window.is_pressed('a'):
            self.cam_center.x += np.cos(self.cam_theta - 1.57) * pan_speed
            self.cam_center.y += np.sin(self.cam_theta - 1.57) * pan_speed
        if self.window.is_pressed('d'):
            self.cam_center.x += np.cos(self.cam_theta + 1.57) * pan_speed
            self.cam_center.y += np.sin(self.cam_theta + 1.57) * pan_speed

        # Q/E tilt
        if self.window.is_pressed('q'):
            self.cam_phi -= 0.02
        if self.window.is_pressed('e'):
            self.cam_phi += 0.02
        self.cam_phi = max(0.01, min(3.14, self.cam_phi))


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
                self.gui.text("WASD: Pan Camera")
                self.gui.text("QE: Tilt Camera")
                self.gui.text("RMB Drag: Orbit Camera")
                self.gui.text(f"1/Z: Toggle Tool ({self.show_tool})")
                self.gui.text(f"2/X: Toggle Holder ({self.show_holder})")
                self.gui.text(f"3/C: Toggle Stock ({self.show_stock})")
                self.gui.text(f"4/V: Toggle Part ({self.show_part})")
                self.gui.text(f"5/B: Toggle Debug ({self.show_debug})")
                self.gui.text(f"Step: {self.current_step}/{self.max_steps}")

        if self.show_debug:
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

        self.window.show()


    def _render_rgb_array(self):
        raise NotImplementedError()


    def render(self):
        """ Renders the environment. """

        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            self._render_rgb_array()


    def close(self):
        """ Cleans up the environment resources. """
        if self.window is not None:
            self.window.running = False
            self.window = None


if __name__ == "__main__":
    env = gym.make('CamEnv-v0')
    obs, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env.render()

        print(f"Step: {info['step']}, Action: {action-1}, Reward: {reward}")

    env.close()