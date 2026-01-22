### Background: What are Signed Distance Fields (SDFs)?

An SDF represents a 3D shape not as a mesh of triangles (like an STL), but as a mathematical field. For every point $p(x,y,z)$ in space, the SDF function $\\phi(p)$ returns the shortest distance to the surface of the object.

* **$\\phi(p) \< 0$**: Point is **inside** the object.  
* **$\\phi(p) \> 0$**: Point is **outside** the object.  
* **$\\phi(p) \= 0$**: Point is **on the surface**.

From STL to SDF:  
To import an STL, you essentially "voxelize" space and compute the distance from each voxel center to the nearest triangle in the mesh. In Taichi, this is a one-time pre-computation step. You calculate the SDF for your Target Part (the goal) and initialize a solid block SDF for your Stock Part (the raw material).

### Architecture

#### A. Material Removal (The Cutting Process)

In an SDF representation, cutting is simply a Boolean subtraction operation.  
Let $\\phi\_{stock}$ be your material and $\\phi\_{tool}$ be your cutter (e.g., a cylinder or ball end mill).  
The cut operation at any timestep is:

$$\\phi\_{stock}^{new} \= \\max(\\phi\_{stock}^{old}, \-\\phi\_{tool})$$

* **Note:** $\\phi\_{tool}$ changes as the tool moves. In Taichi, you don't update the entire grid; you only update the voxels inside the tool's bounding box.

#### B. Modeling Forces & Cutter Breakage

This is where SDFs shine over simple voxel occupancy grids.

* Chip Load / Instantaneous Force: The force on the cutter is proportional to the volume of material being removed per time step.

  $$F\_{cut} \\propto \\sum\_{p \\in Tool} \\max(0, \-\\phi\_{stock}(p))$$

  Because you have the distance values, you can approximate the "depth of cut" and "engagement angle" much more accurately than with binary voxels.  
* Breakage Reward: You can define a threshold force $F\_{max}$. If the calculated force exceeds this, the simulation returns a "broken" state (negative reward) and terminates the episode.

  $$R\_{breakage} \= \\begin{cases} \-100 & \\text{if } F\_{cut} \> F\_{max} \\\\ 0 & \\text{otherwise} \\end{cases}$$

#### C. Calculating Similarity (The Reward)

For RL, you need a fast way to measure how close the current stock is to the target.

* **IoU (Intersection over Union):** Good for rough shaping.  
* Chamfer Distance / Surface Error: Better for precision.

  $$Loss \= \\sum\_{p} |\\phi\_{stock}(p) \- \\phi\_{target}(p)|^2$$

  Since you have both SDFs in the same grid system, this is a simple element-wise kernel subtraction in Taichi, which is incredibly fast and fully differentiable.

### Infrastructure

* **[Taichi](https://github.com/taichi-dev/taichi):** For the physics simulation and differentiable SDF operations.  
* **[PufferLib](https://github.com/vwxyzjn/pufferlib):** For vectorizable  Gymnasium-compatible environment.  
* **[CleanRL](https://github.com/vwxyzjn/cleanrl):** PPO for training the high-level policy.  
* **CAM Library:** derived from [LinuxCNC](https://linuxcnc.org/)'s HAL and G-code interpreter.
