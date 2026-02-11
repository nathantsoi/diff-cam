import taichi as ti
import numpy as np
from simulator.simulator import CNCSimulator
from simulator.paths import StairStepPath

# Initialize Taichi on GPU with debug mode enabled for better error messages
# Otherwise, initialize Taichi on CPU

print("Initializing Taichi in main.py")
if ti._lib.core.with_cuda():
    ti.init(arch=ti.gpu, debug=True)
else:
    ti.init(arch=ti.cpu, debug=True)

# --- Main Execution ---

def main():
    sim = CNCSimulator(resolution=128)
    sim.initialize_stock_primitive()
    sim.initialize_target_primitive()
    sim.initialize_tool_primitive()

    # Initialize a path for testing
    path = StairStepPath(step_length=10, num_steps=5)

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
    
    # Simulation Loop (moved to simulator)
    # tool_pos = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    # tool_height = 0.3
    
    # Pre-calculate tool template
    sim.init_tool_template()
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
    
    # GUI State
    show_help = False
    paused = False
    show_tool = True
    show_holder = True
    show_stock = True
    show_part = True
    show_debug = False
    
    gui = window.get_gui()

    while window.running:
        # --- Input Handling ---
        # We handle events manually for Orbit Camera and Hotkeys
        for e in window.get_events(ti.ui.PRESS):
            key = e.key
            # Handle shifted keys on common layouts
            if key == '!': key = '1'
            if key == '@': key = '2'
            if key == '#': key = '3'
            if key == '$': key = '4'
            if key == '%': key = '5'
            
            if key == ti.ui.RMB:
                rmb_down = True
            elif key == ti.ui.ESCAPE:
                window.running = False
            elif key == 'h' or key == 'H':
                show_help = not show_help
            elif key == ti.ui.SPACE:
                paused = not paused
            # Number keys (1-4) or Alternatives (Z-V)
            # Some environments fail to map number keys correctly
            elif key == '1' or key == 'z' or key == 'Z':
                show_tool = not show_tool
            elif key == '2' or key == 'x' or key == 'X':
                show_holder = not show_holder
            elif key == '3' or key == 'c' or key == 'C':
                show_stock = not show_stock
            elif key == '4' or key == 'v' or key == 'V':
                show_part = not show_part
            elif key == '5' or key == 'b' or key == 'B':
                show_debug = not show_debug
        
        for e in window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.RMB:
                rmb_down = False

        # Camera Control (WASD + mouse drag)
        curr_mouse_pos = window.get_cursor_pos()
        if rmb_down:
            dx = curr_mouse_pos[0] - last_mouse_pos[0]
            dy = curr_mouse_pos[1] - last_mouse_pos[1]
            
            # Rotate
            cam_theta -= dx * 5.0
            cam_phi   += dy * 5.0 
            cam_phi = max(0.01, min(3.14, cam_phi))

        last_mouse_pos = curr_mouse_pos
        
        # Pan Camera with WASD (Relative to view)
        pan_speed = 0.02
        if window.is_pressed('w'):
            # Move forward in XY plane relative to camera angle
            cam_center.x += np.cos(cam_theta) * pan_speed
            cam_center.y += np.sin(cam_theta) * pan_speed
        if window.is_pressed('s'):
             cam_center.x -= np.cos(cam_theta) * pan_speed
             cam_center.y -= np.sin(cam_theta) * pan_speed
        if window.is_pressed('a'):
             # Strafe Left
             cam_center.x += np.cos(cam_theta - 1.57) * pan_speed
             cam_center.y += np.sin(cam_theta - 1.57) * pan_speed
        if window.is_pressed('d'):
             # Strafe Right
             cam_center.x += np.cos(cam_theta + 1.57) * pan_speed
             cam_center.y += np.sin(cam_theta + 1.57) * pan_speed
             
        # Tilt/Elevate with Q/E (Absolute Z)
        if window.is_pressed('q'):
             cam_phi -= 0.02
        if window.is_pressed('e'):
             cam_phi += 0.02
        cam_phi = max(0.01, min(3.14, cam_phi)) # Clamp
        
        
        # Update Camera Vectors
        cam_x = cam_r * np.sin(cam_phi) * np.cos(cam_theta)
        cam_y = cam_r * np.sin(cam_phi) * np.sin(cam_theta)
        cam_z = cam_r * np.cos(cam_phi)
        
        camera.position(cam_center.x + cam_x, cam_center.y + cam_y, cam_center.z + cam_z)
        camera.lookat(cam_center.x, cam_center.y, cam_center.z)
        camera.up(0, 0, 1)

        # 1. Update Tool Position 
        if not paused:
            # sim.tool_pos[None].x = (np.sin(frame * 0.02) * 0.5) + 0.5 
            sim.move_tool_one_unit(path.move())
        
        # 2. Run Physics (Cut)
        if not paused:
            # print(f"Applying cut at frame {frame}...")
            try:
                force = sim.apply_cut()
                ti.sync() 
            except Exception as e:
                print(f"Error during cut: {e}")

        # 3. Generate visualization meshes
        try:
            if show_stock:
                sim.generate_stock_visualization_mesh()
            if show_part:
                sim.generate_target_visualization_mesh()
            
            sim.update_tool(sim.tool_pos[None])
            ti.sync()
        except Exception as e:
            print(f"Error during mesh gen: {e}")
    

        
        # 4. Draw GUI Overlay
        if show_help:
            with gui.sub_window("Controls", x=0.05, y=0.05, width=0.3, height=0.45):
                gui.text(f"H: Toggle Help ({show_help})")
                gui.text(f"Space: Pause/Resume ({paused})")
                gui.text("WASD: Pan Camera")
                gui.text("QE: Tilt Camera (Phi)")
                gui.text(f"Mouse Drag: Orbit Camera")
                gui.text(f"1/Z: Toggle Tool ({show_tool})")
                gui.text(f"2/X: Toggle Holder ({show_holder})")
                gui.text(f"3/C: Toggle Stock ({show_stock})")
                gui.text(f"4/V: Toggle Part ({show_part})")
                gui.text(f"5/B: Toggle Debug Slices ({show_debug})")
       
        if show_debug:
            sim.generate_slices()
            sim.compose_debug_view()
            canvas.set_image(sim.debug_buffer)
            
            # Overlay simple text logic to explain view
            with gui.sub_window("Debug Info", x=0.5, y=0.05, width=0.45, height=0.2):
                gui.text("TL: XY (Top) | TR: XZ (Front)")
                gui.text("BL: YZ (Side)")
                gui.text("Green Layer: Tool Radius")
                gui.text("Blue: Stock (SDF < 0)")
                gui.text(f"Res: {sim.res}, AR Corrected")
       
        else:
            # Render Scene
            scene.set_camera(camera)
            scene.ambient_light((0.5, 0.5, 0.5))
            
            # Draw Stock
            if show_stock:
                count = min(sim.stock_count[None], sim.stock_points.shape[0])
                if count > 0:
                    scene.particles(sim.stock_points, per_vertex_color=None, radius=0.005, color=(0.2, 0.8, 0.2), index_count=count)
            
            # Draw Part (Target)
            if show_part:
                count = min(sim.target_count[None], sim.target_points.shape[0])
                if count > 0:
                    scene.particles(sim.target_points, radius=0.005, color=(0.5, 0.5, 1.0), index_count=count)

            # Draw Tool
            if show_tool:
                scene.particles(sim.tool_points, radius=0.005, color=(1.0, 0.2, 0.2), index_count=sim.tool_count[None])

            # Draw Holder
            if show_holder:
                scene.particles(sim.holder_points, radius=0.005, color=(0.2, 0.2, 0.2), index_count=sim.holder_count[None])
            
            scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
            
            # Draw Coordinate Frame
            scene.lines(axes_points, width=5.0, per_vertex_color=axes_colors)
            
            canvas.scene(scene)

            if not show_help:
                with gui.sub_window("Help", x=0.05, y=0.05, width=0.2, height=0.1):
                    gui.text("Press 'h' for controls")

        window.show()
        
        if not paused:
            frame += 1

if __name__ == "__main__":
    main()