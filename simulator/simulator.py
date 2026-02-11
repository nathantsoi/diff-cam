import taichi as ti

# Initialize Taichi on GPU with debug mode enabled for better error messages
# Otherwise, initialize Taichi on CPU

print("Initializing Taichi in simulator.py")
if ti._lib.core.with_cuda():
    ti.init(arch=ti.gpu, debug=True)
else:
    ti.init(arch=ti.cpu, debug=True)

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

        # Define the tool
        self.tool_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.tool_radius = ti.field(dtype=ti.f32, shape=())
        self.tool_height = ti.field(dtype=ti.f32, shape=())
        
        # Define the sparse layout (Pointer -> Dense)
        self.block_size = 8
        self.sdf_layout = ti.root.pointer(ti.ijk, self.res // self.block_size)
        self.sdf_layout.dense(ti.ijk, self.block_size).place(self.sdf_stock)
        self.sdf_layout.dense(ti.ijk, self.block_size).place(self.sdf_target)

        # Visualization fields (for GGUI)

        # Stock visualization
        self.stock_points = ti.Vector.field(3, dtype=ti.f32, shape=self.res**3) # Full resolution size to be safe
        self.stock_count = ti.field(dtype=ti.i32, shape=())

        # Tool Visualization
        self.tool_points = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.tool_template = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.tool_count = ti.field(dtype=ti.i32, shape=())

        # Holder Visualization
        self.holder_points = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.holder_template = ti.Vector.field(3, dtype=ti.f32, shape=100000)
        self.holder_count = ti.field(dtype=ti.i32, shape=())

        # Part (Target) Visualization
        self.target_points = ti.Vector.field(3, dtype=ti.f32, shape=self.res**3)
        self.target_count = ti.field(dtype=ti.i32, shape=())

        # Debug Slices
        self.slice_xy = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
        self.slice_xz = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
        self.slice_yz = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
        
        # Combined debug view using 2x2 grid with padding to match window Aspect Ratio (4:3)
        # Window: 1024x768. AR = 1.333
        # Buffer Height: 2 * res = 256.
        # Buffer Width Target: 256 * 1.333 = 341.33 -> 341
        self.debug_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(341, 2 * self.res))



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

    @ti.kernel
    def initialize_target_primitive(self):
        """ Initializes target as a smaller sphere/box for visualization """
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            p = ti.Vector([i, j, k]) * self.dx
            center = ti.Vector([0.5, 0.5, 0.5])
            # Sphere target
            d = (p - center).norm() - 0.25
            self.sdf_target[i, j, k] = d


    def initialize_tool_primitive(self):
        """ Initializes the tool position, tool radius, and tool height """
        self.tool_pos[None] = ti.Vector([0.0, 0.5, 0.5])
        self.tool_radius[None] = 0.1
        self.tool_height[None] = 0.3

    @ti.func
    def tool_sdf(self, p):
        """ Analytic SDF for a cylindrical cutter (aligned with Z-axis) """
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        d_h = ti.Vector([p.x - tool_pos.x, p.y - tool_pos.y]).norm() - tool_radius
        
        # Finite height cylinder
        # bottom = tool_pos.z
        # top = tool_pos.z + tool_height
        d_z_bottom = tool_pos.z - p.z 
        d_z_top = p.z - (tool_pos.z + tool_height)
        d_z = ti.max(d_z_bottom, d_z_top)
        
        return ti.max(d_h, d_z)

    @ti.kernel
    def apply_cut(self) -> ti.f32:
        """ 
        Boolean Subtraction: Stock = max(Stock, -Tool)
        Returns the approximate volume removed (useful for force calculation)
        """
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]


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
            
            tool_dist = self.tool_sdf(p)
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
    def move_tool_one_unit(self, dir: ti.types.vector(3, ti.f32)):
        """ Moves the tool one unit in the direction of a unit vector """
        valid = True
        for i in ti.static(range(3)):
            if not (dir[i] == -1.0 or dir[i] == 0.0 or dir[i] == 1.0):
                valid = False

        if not valid:
            print("Error: Direction must be a unit step vector with components in {-1, 0, 1}")
            
        self.tool_pos[None] += dir * self.dx


    # Visualization Kernels
    @ti.kernel
    def generate_stock_visualization_mesh(self):
        """ 
        Extracts a point cloud of the surface (SDF approx 0) for GGUI.
        Real implementations might use Marching Cubes or Raymarching. 
        """
        self.stock_count[None] = 0
        
        # Iterate over the entire domain safely
        # Sparse iterator was causing crashes, likely due to internal structural updates in apply_cut
        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            val = self.sdf_stock[i, j, k]
            # If close to surface
            if ti.abs(val) < self.dx * 1.5: 
                idx = ti.atomic_add(self.stock_count[None], 1)
                # Bounds check to prevent CUDA invalid address
                if idx < self.stock_points.shape[0]:
                    self.stock_points[idx] = ti.Vector([i, j, k]) * self.dx

    @ti.kernel
    def generate_target_visualization_mesh(self):
        """ Extracts points for the target/part """
        self.target_count[None] = 0

        for i, j, k in ti.ndrange(self.res, self.res, self.res):
            val = self.sdf_target[i, j, k]
            if ti.abs(val) < self.dx * 1.5:
                idx = ti.atomic_add(self.target_count[None], 1)
                if idx < self.target_points.shape[0]:
                    self.target_points[idx] = ti.Vector([i, j, k]) * self.dx

    @ti.kernel
    def init_tool_template(self):
        # Grab radius and height
        radius = self.tool_radius[None]
        height = self.tool_height[None]

        # Generate a point cloud for the cylinder ONCE
        # We subtract the visualization particle radius (0.005) so the visual edge matches physics
        visual_r = radius - 0.005
        
        self.tool_count[None] = 0
        
        # Grid based generation
        r_steps = int(ti.ceil(visual_r / self.dx))
        h_steps = int(ti.ceil(height / self.dx))
        
        for i, j, k in ti.ndrange((-r_steps, r_steps+1), (-r_steps, r_steps+1), (0, h_steps+1)):
             x = i * self.dx
             y = j * self.dx
             z = k * self.dx
             
             if x*x + y*y <= visual_r*visual_r and z <= height:
                 idx = ti.atomic_add(self.tool_count[None], 1)
                 if idx < 100000:
                     self.tool_template[idx] = ti.Vector([x, y, z])

        # Generate Holder Template (Wider cylinder on top)
        self.holder_count[None] = 0
        holder_r = visual_r * 2.0
        holder_h = height * 0.5
        hr_steps = int(ti.ceil(holder_r / self.dx))
        hh_steps = int(ti.ceil(holder_h / self.dx))
        
        for i, j, k in ti.ndrange((-hr_steps, hr_steps+1), (-hr_steps, hr_steps+1), (0, hh_steps+1)):
             x = i * self.dx
             y = j * self.dx
             z = k * self.dx
             
             if x*x + y*y <= holder_r*holder_r and z <= holder_h:
                 idx = ti.atomic_add(self.holder_count[None], 1)
                 if idx < 100000:
                     # Offset slightly up so it sits on top of the tool, but we'll handle placement in update
                     self.holder_template[idx] = ti.Vector([x, y, z])

    @ti.kernel
    def update_tool(self, tool_pos: ti.types.vector(3, ti.f32)):
        # Fast update: just add position
        for i in self.tool_points:
            self.tool_points[i] = self.tool_template[i] + tool_pos
            
        # Update holder position (Attach to top of tool)
        # Holder sits at tool_pos.z + tool_height (0.3)
        # We need to know tool height here or assume it's checked elsewhere.
        # For this simple viz, let's assume a fix offset or pass it in.
        holder_offset = ti.Vector([0.0, 0.0, 0.3]) # Hardcoded 0.3 tool height for now
        for i in self.holder_points:
             self.holder_points[i] = self.holder_template[i] + tool_pos + holder_offset

    @ti.kernel
    def generate_slices(self):
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        # 1. XY Slice at tool Z
        z_idx = int(tool_pos.z / self.dx)
        z_idx = ti.max(0, ti.min(z_idx, self.res - 1))
        
        for i, j in ti.ndrange(self.res, self.res):
            # SDF Color
            val = self.sdf_stock[i, j, z_idx]
            color = ti.Vector([0.0, 0.0, 0.0])
            
            # Helper for checkerboard pattern to see grid
            grid_check = ((i // 4) + (j // 4)) % 2
            
            if val < 0: # Inside stock
                color = ti.Vector([0.3, 0.3, 0.9])
            else: # Outside
                bg_col = 0.8 if grid_check == 0 else 0.7
                color = ti.Vector([bg_col, bg_col, bg_col])
                
            # Tool Overlay
            p = ti.Vector([i, j, z_idx]) * self.dx
            dist_to_center = (ti.Vector([p.x, p.y]) - ti.Vector([tool_pos.x, tool_pos.y])).norm()
            
            # Green outline for tool radius
            if abs(dist_to_center - tool_radius) < self.dx:
                 color = ti.Vector([0.0, 1.0, 0.0])
            
            self.slice_xy[i, j] = color

        # 2. XZ Slice at tool Y
        y_idx = int(tool_pos.y / self.dx)
        y_idx = ti.max(0, ti.min(y_idx, self.res - 1))
        
        for i, k in ti.ndrange(self.res, self.res):
             val = self.sdf_stock[i, y_idx, k]
             color = ti.Vector([0.0, 0.0, 0.0])
             if val < 0: color = ti.Vector([0.3, 0.3, 0.9])
             else: color = ti.Vector([0.8, 0.8, 0.8])
             
             p = ti.Vector([i, y_idx, k]) * self.dx
             dist_x = abs(p.x - tool_pos.x)
             if abs(dist_x - tool_radius) < self.dx:
                  color = ti.Vector([0.0, 1.0, 0.0])
             
             self.slice_xz[i, k] = color

        # 3. YZ Slice at tool X
        x_idx = int(tool_pos.x / self.dx)
        x_idx = ti.max(0, ti.min(x_idx, self.res - 1))
        
        for j, k in ti.ndrange(self.res, self.res):
             val = self.sdf_stock[x_idx, j, k]
             color = ti.Vector([0.0, 0.0, 0.0])
             if val < 0: color = ti.Vector([0.3, 0.3, 0.9])
             else: color = ti.Vector([0.8, 0.8, 0.8])
             
             p = ti.Vector([x_idx, j, k]) * self.dx
             dist_y = abs(p.y - tool_pos.y)
             if abs(dist_y - tool_radius) < self.dx:
                  color = ti.Vector([0.0, 1.0, 0.0])

             self.slice_yz[j, k] = color

    @ti.kernel
    def compose_debug_view(self):
        # Clear buffer (black)
        for i, j in ti.ndrange(341, 2 * self.res):
             self.debug_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])

        # Stitch slices into 2x2 grid
        # Top-Left: XY
        # Top-Right: XZ
        # Bottom-Left: YZ
        # Botom-Right: Legend area (empty)
        
        for i, j in ti.ndrange(self.res, self.res):
            # 1. XY at (0, res) to (res, 2*res) -- Top Left?
            # Canvas coords: (0,0) is bottom-left usually in Taichi? 
            # Let's verify. standard graphical convention is usually bottom-left origin.
            
            # Place XY at Top-Left: x=[0, res], y=[res, 2*res]
            self.debug_buffer[i, j + self.res] = self.slice_xy[i, j]
            
            # Place XZ at Top-Right: x=[res, 2*res], y=[res, 2*res]
            self.debug_buffer[i + self.res, j + self.res] = self.slice_xz[i, j]
            
            # Place YZ at Bottom-Left: x=[0, res], y=[0, res]
            self.debug_buffer[i, j] = self.slice_yz[i, j]

def main():
    sim = CNCSimulator(resolution=128)
    print("Simulator initialized!")

if __name__ == "__main__":
    main()