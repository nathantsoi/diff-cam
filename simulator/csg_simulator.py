import random
import taichi as ti

from simulator.simulator_utils import *


@ti.data_oriented
class CSGSimulator:
    def __init__(self, resolution=32, max_steps=512, k=50, shape=None):
        # initialize Taichi
        try:
            if ti._lib.core.with_cuda():
                ti.init(arch=ti.gpu, debug=False)
            else:
                ti.init(arch=ti.cpu, debug=False)
        except:
            pass  # taichi alrady initialized, ignore

        # define simulation parameters
        self.resolution = resolution
        self.max_steps = max_steps
        self.k = k
        self.num_steps = ti.field(dtype=ti.i32, shape=())

        self.dx = 1.0 / resolution
        self.inv_dx = float(resolution)

        # Tool
        self.tool_pos = ti.Vector.field(3, dtype=ti.f32, shape=max_steps)
        self.tool_radius = ti.field(dtype=ti.f32, shape=())
        self.tool_height = ti.field(dtype=ti.f32, shape=())

        # Target
        shape_options = ["box", "cylinder", "sphere", "pyramid"]
        self.shape_params = {}

        if shape is None:
            shape = random.choice(shape_options)
        self.shape = shape
        if self.shape == "box":
            self.shape_params["half_size"] = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.shape_params["center"] = ti.Vector.field(3, dtype=ti.f32, shape=())
        elif self.shape == "cylinder":
            self.shape_params["radius"] = ti.field(dtype=ti.f32, shape=())
            self.shape_params["height"] = ti.field(dtype=ti.f32, shape=())
        elif self.shape == "sphere":
            self.shape_params["radius"] = ti.field(dtype=ti.f32, shape=())
            self.shape_params["center"] = ti.Vector.field(3, dtype=ti.f32, shape=())
        elif self.shape == "pyramid":
            self.shape_params["base_half_size"] = ti.Vector.field(
                3, dtype=ti.f32, shape=()
            )
            self.shape_params["height"] = ti.field(dtype=ti.f32, shape=())
            self.shape_params["center"] = ti.Vector.field(3, dtype=ti.f32, shape=())
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")

        # Loss
        self.loss = ti.field(dtype=ti.f32, shape=())
        ti.root.lazy_grad()
        self._initialize_samples()


    @ti.func
    def target_sdf(self, p):
        # use simulator_utils SDFs
        if self.shape == "sphere":
            return sphere_sdf(
                p,
                center=ti.Vector([0.5, 0.5, 0.5]),
                radius=self.shape_params["radius"][None],
            )
        elif self.shape == "box":
            return box_sdf(
                p,
                center=self.shape_params["center"][None],
                half_size=self.shape_params["half_size"][None],
            )
        elif self.shape == "cylinder":
            return cylinder_sdf(
                p,
                center=ti.Vector([0.5, 0.5, 0.5]),
                radius=self.shape_params["radius"][None],
                height=self.shape_params["height"][None],
            )
        elif self.shape == "pyramid":
            return pyramid_sdf(
                p,
                center=self.shape_params["center"][None],
                base_half_size=self.shape_params["base_half_size"][None],
                height=self.shape_params["height"][None],
            )
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")

    @ti.func
    def stock_sdf(self, p):
        center = ti.Vector([0.5, 0.5, 0.5])
        half_size = ti.Vector([0.5, 0.5, 0.5])
        d = box_sdf(p, center, half_size)

        for t in range(self.max_steps):
            tool_p = self.tool_pos[t]
            # tool_d = dist_from_tool(p, tool_p)

            # subtract tool
            d = smooth_max(d, -tool_d, self.k)

        return d

    @ti.func
    def tool_sdf(self, p):
        r = self.tool_radius[None]
        h = self.tool_height[None]

        d_xy = ti.Vector([p.x - self.tool_pos.x, p.y - self.tool_pos.y]).norm() - r

        d_z_bottom = self.tool_pos.z - p.z
        d_z_top = p.z - (self.tool_pos.z + h)
        d_z = ti.max(d_z_bottom, d_z_top)

        return ti.max(d_xy, d_z)

    @ti.func
    def holder_sdf(self, p):
        tool_pos = self.tool_pos[None]
        tool_radius = self.tool_radius[None]
        tool_height = self.tool_height[None]

        holder_radius = tool_radius * 2.0
        holder_height = tool_height * 0.5
        holder_z_start = tool_pos.z + tool_height

        dx = p.x - tool_pos.x
        dy = p.y - tool_pos.y
        d_h = ti.sqrt(dx * dx + dy * dy + 1e-12) - holder_radius

        d_z_bottom = holder_z_start - p.z
        d_z_top = p.z - (holder_z_start + holder_height)
        d_z = ti.max(d_z_bottom, d_z_top)
        return ti.max(d_h, d_z)


    @ti.kernel
    def compute_loss(self):
        self.loss[None] = 0.0

        for i, j, k in ti.ndrange(self.resolution, self.resolution, self.resolution):

            p = ti.Vector(
                [(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx]
            )

            stock_val = self.stock_sdf(p)
            target_val = self.target_sdf(p)

            # alignment loss
            diff = stock_val - target_val
            geom_loss = diff * diff

            # undercut penalty
            undercut = ti.max(0.0, target_val - stock_val)

            # overcut penalty
            overcut = ti.max(0.0, stock_val - target_val)

            loss = geom_loss + 2.0 * undercut + 2.0 * overcut

            ti.atomic_add(self.loss[None], loss)

        # normalize
        self.loss[None] /= self.resolution**3

