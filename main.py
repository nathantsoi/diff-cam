import taichi as ti
import numpy as np

# Initialize Taichi on GPU
# Initialize Taichi on GPU with debug mode enabled for better error messages
ti.init(arch=ti.gpu, debug=True)

@ti.data_oriented
class CNCSimulator:
    def __init__(self, resolution=128):
        self.res = resolution
        self.dx = 1.0 / self.res
        
        # --- 1. Sparse Data Structure ---
        # We use a sparse grid to store the Signed Distance Field (SDF).
        # This saves memory by only allocating blocks where the surface exists.
        self.sdf_stock = ti.field(dtype=ti.f32)
        self.sdf_target = ti.field(dtype=ti.f32)
        
        # Define the sparse layout (Pointer -> Dense)
        self.block_size = 8
        self.sdf_layout = ti.root.pointer(ti.ijk, self.res // self.block_size)
        self.sdf_layout.dense(ti.ijk, self.block_size).place(self.sdf_stock)
        self.sdf_layout.dense(ti.ijk, self.block_size).place(self.sdf_target)

        # Visualization fields (for GGUI)
        self.render_points = ti.Vector.field(3, dtype=ti.f32, shape=self.res**3) # Full resolution size to be safe
        self.render_count = ti.field(dtype=ti.i32, shape=())

        # Tool Visualization
        self.tool_points = ti.Vector.field(3, dtype=ti.f32, shape=4096)
        self.tool_template = ti.Vector.field(3, dtype=ti.f32, shape=4096)


    @ti.kernel
    def initialize_stock_primitive(self):
        """ Initializes stock as a solid block (SDF < 0 inside) """
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            # Simple box SDF: max(|x|-s, |y|-s, |z|-s)
            # Centered at 0.5, size 0.8
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            d = ti.abs(p - center) - 0.4
            dist = ti.max(d.x, ti.max(d.y, d.z))
            
            self.sdf_stock[i, j, k] = dist
            # In a real app, you would load self.sdf_target here from an STL

    @ti.func
    def tool_sdf(self, p, tool_pos, tool_radius, tool_height):
        """ Analytic SDF for a cylindrical cutter (aligned with Z-axis) """
        d_h = ti.Vector([p.x - tool_pos.x, p.y - tool_pos.y]).norm() - tool_radius
        
        # Finite height cylinder
        # bottom = tool_pos.z
        # top = tool_pos.z + tool_height
        d_z_bottom = tool_pos.z - p.z 
        d_z_top = p.z - (tool_pos.z + tool_height)
        d_z = ti.max(d_z_bottom, d_z_top)
        
        return ti.max(d_h, d_z)

    @ti.kernel
    def apply_cut(self, tool_pos: ti.types.vector(3, ti.f32), tool_radius: ti.f32, tool_height: ti.f32) -> ti.f32:
        """ 
        Boolean Subtraction: Stock = max(Stock, -Tool)
        Returns the approximate volume removed (useful for force calculation)
        """
        removed_vol = 0.0
        
        # Optimization: Only iterate over the bounding box of the tool
        # Convert tool position and radius to index space
        
        # Calculate bounds in integer coordinates
        # Calculate bounds in integer coordinates
        # X and Y are bounded by tool radius
        min_x = int(ti.floor((tool_pos.x - tool_radius) / self.dx - 4.0))
        max_x = int(ti.ceil((tool_pos.x + tool_radius) / self.dx + 4.0))
        
        min_y = int(ti.floor((tool_pos.y - tool_radius) / self.dx - 4.0))
        max_y = int(ti.ceil((tool_pos.y + tool_radius) / self.dx + 4.0))

        # Z is bounded at bottom by tool tip, and top by tool tip + height
        min_z = int(ti.floor((tool_pos.z) / self.dx - 4.0))
        max_z = int(ti.ceil((tool_pos.z + tool_height) / self.dx + 4.0))

        # Clamp to domain size
        min_x = ti.max(min_x, 0)
        max_x = ti.min(max_x, self.res)
        min_y = ti.max(min_y, 0)
        max_y = ti.min(max_y, self.res)
        min_z = ti.max(min_z, 0)
        max_z = ti.min(max_z, self.res)

        # Iterate over the bounding box
        for i, j, k in ti.ndrange((min_x, max_x), (min_y, max_y), (min_z, max_z)):
            p = ti.Vector([i, j, k]) * self.dx
            
            tool_dist = self.tool_sdf(p, tool_pos, tool_radius, tool_height)
            stock_dist = self.sdf_stock[i, j, k]
            
            # The Cut Logic:
            # We want the intersection of the Stock and the NOT(Tool)
            # In SDF math: max(A, -B)
            new_dist = ti.max(stock_dist, -tool_dist)
            
            if new_dist != stock_dist:
                # If values changed, we removed material
                removed_vol += 1.0 # simplistic volume proxy
                self.sdf_stock[i, j, k] = new_dist
                
        return removed_vol

    @ti.kernel
    def generate_visualization_mesh(self):
        """ 
        Extracts a point cloud of the surface (SDF approx 0) for GGUI.
        Real implementations might use Marching Cubes or Raymarching. 
        """
        self.render_count[None] = 0
        
        # Iterate over the entire domain safely
        # Sparse iterator was causing crashes, likely due to internal structural updates in apply_cut
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            val = self.sdf_stock[i, j, k]
            # If close to surface
            if ti.abs(val) < self.dx * 1.5: 
                idx = ti.atomic_add(self.render_count[None], 1)
                # Bounds check to prevent CUDA invalid address
                if idx < self.render_points.shape[0]:
                    self.render_points[idx] = ti.Vector([i, j, k]) * self.dx

    @ti.kernel
    def init_tool_template(self, radius: ti.f32, height: ti.f32):
        # Generate a point cloud for the cylinder ONCE
        # We subtract the visualization particle radius (0.005) so the visual edge matches physics
        visual_r = radius - 0.005
        
        for i in self.tool_template:
            # Random cylindrical coords
            theta = ti.random() * 2 * 3.14159
            r = ti.sqrt(ti.random()) * visual_r # uniform disk
            h = ti.random() * height
            
            # Add some points on surface for better visual
            if ti.random() < 0.3:
                r = visual_r # Surface
            
            # Centered at (0,0,0) relative to tool pos, but Z goes from 0 to height
            # Existing logic assumed tool_pos is bottom.
            self.tool_template[i] = ti.Vector([r * ti.cos(theta), r * ti.sin(theta), h])

    @ti.kernel
    def update_tool(self, tool_pos: ti.types.vector(3, ti.f32)):
        # Fast update: just add position
        for i in self.tool_points:
            self.tool_points[i] = self.tool_template[i] + tool_pos

# --- Main Execution ---

def main():
    sim = CNCSimulator(resolution=64)
    sim.initialize_stock_primitive()

    # GGUI Setup
    window = ti.ui.Window("Differentiable CNC Simulator", (1024, 768))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    # Camera Init
    camera.position(1.5, 1.5, 1.5)
    camera.lookat(0.5, 0.5, 0.5)
    camera.up(0, 0, 1)
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)
    
    # Simulation Loop
    tool_pos = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    tool_height = 0.3
    
    # Pre-calculate tool template
    sim.init_tool_template(0.1, tool_height)
    frame = 0
    
    # Coordinate Frame Data
    axes_points = ti.Vector.field(3, dtype=ti.f32, shape=6)
    axes_colors = ti.Vector.field(3, dtype=ti.f32, shape=6)
    
    # X Axis (Red)
    axes_points[0] = [0, 0, 0]
    axes_points[1] = [1, 0, 0]
    axes_colors[0] = [1, 0, 0]
    axes_colors[1] = [1, 0, 0]
    
    # Y Axis (Green)
    axes_points[2] = [0, 0, 0]
    axes_points[3] = [0, 1, 0]
    axes_colors[2] = [0, 1, 0]
    axes_colors[3] = [0, 1, 0]
    
    # Z Axis (Blue)
    axes_points[4] = [0, 0, 0]
    axes_points[5] = [0, 0, 1]
    axes_colors[4] = [0, 0, 1]
    axes_colors[5] = [0, 0, 1]


    # Orbit Camera State
    cam_r = 3.0
    cam_theta = -1.57 # Start looking from front (roughly)
    cam_phi = 1.0     # Slight elevation
    cam_center = ti.Vector([0.5, 0.5, 0.5])
    
    last_mouse_pos = window.get_cursor_pos()
    rmb_down = False

    while window.running:
        # --- Input Handling ---
        # We handle events manually for Orbit Camera
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.RMB:
                rmb_down = True
            elif e.key == ti.ui.ESCAPE:
                window.running = False
        
        for e in window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.RMB:
                rmb_down = False

        curr_mouse_pos = window.get_cursor_pos()
        if rmb_down:
            dx = curr_mouse_pos[0] - last_mouse_pos[0]
            dy = curr_mouse_pos[1] - last_mouse_pos[1]
            
            # Sensitivity
            cam_theta -= dx * 5.0
            cam_phi   += dy * 5.0 # Inverted Y for intuitive drag
            
            # Clamp phi to avoid flipping
            cam_phi = max(0.01, min(3.14, cam_phi))
            
        last_mouse_pos = curr_mouse_pos
        
        # Zoom (simple hack via key presses or just use scroll if available? 
        # Taichi UI doesn't always expose scroll nicely in all backends, 
        # but let's try assuming standard window events or just fixed distance for now if unsure.
        # Actually track_user_inputs supports scroll, but we are replacing it.
        # Let's add simple keys for zoom if scroll is tricky, or just fixed radius.
        # We'll try to stick to basic dragging first.
        
        # Update Camera
        # Spherical to Cartesian relative to center
        # x = r * sin(phi) * cos(theta)
        # y = r * sin(phi) * sin(theta) # This assumes Z up in math convention?
        # Taichi Y-up vs Z-up: We set up=(0,0,1).
        # Standard physics spherical coordinates (Z-up):
        # x = r * sin(phi) * cos(theta)
        # y = r * sin(phi) * sin(theta)
        # z = r * cos(phi)
        
        cam_x = cam_r * np.sin(cam_phi) * np.cos(cam_theta)
        cam_y = cam_r * np.sin(cam_phi) * np.sin(cam_theta)
        cam_z = cam_r * np.cos(cam_phi)
        
        camera.position(cam_center.x + cam_x, cam_center.y + cam_y, cam_center.z + cam_z)
        camera.lookat(cam_center.x, cam_center.y, cam_center.z)
        camera.up(0, 0, 1)

        # 1. Update Tool Position (Simple linear animation)
        tool_pos[0] = (np.sin(frame * 0.02) * 0.5) + 0.5 # Move back and forth
        
        # 2. Run Physics (Cut)
        print(f"Applying cut at frame {frame}...")
        tool_height = 0.3
        try:
            force = sim.apply_cut(ti.Vector(tool_pos), 0.1, tool_height)
            ti.sync() # Sync to catch kernel errors
        except Exception as e:
            print(f"Error during cut: {e}")

        # 3. Prepare Render Data
        # print("Generating mesh...")
        try:
            sim.generate_visualization_mesh()
            sim.update_tool(ti.Vector(tool_pos))
            ti.sync()
        except Exception as e:
            print(f"Error during mesh gen: {e}")
        
        # 4. Render Scene
        # Remove camera.track_user_inputs(window, ...)
        scene.set_camera(camera)
        
        # Draw the Stock (Point Cloud)
        scene.ambient_light((0.5, 0.5, 0.5))
        
        # Clamp the index count to the max buffer size to be safe
        count = min(sim.render_count[None], sim.render_points.shape[0])
        if count > 0:
            scene.particles(sim.render_points, per_vertex_color=None, radius=0.005, color=(0.2, 0.8, 0.2), index_count=count)
        
        # Draw the Tool (Point Cloud)
        scene.particles(sim.tool_points, radius=0.005, color=(1.0, 0.2, 0.2))
        
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
        
        # Draw Coordinate Frame
        scene.lines(axes_points, width=5.0, per_vertex_color=axes_colors)
        
        canvas.scene(scene)
        window.show()
        
        frame += 1

if __name__ == "__main__":
    main()