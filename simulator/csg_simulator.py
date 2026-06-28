import numpy as np
import random
import taichi as ti
from simulator.simulator_utils import *


@ti.data_oriented
class CSGSimulatorDelta:
    """
    Differentiable CSG simulator for CNC trajectory optimization.

    Parameter: tool_delta, a sequence of T per-step displacements (T × 3 floats).
    Tool positions are reconstructed by a cumulative sum from a fixed start:

        tool_pos[0]   = tool_start          (fixed, not learnable)
        tool_pos[t+1] = tool_pos[t] + tool_delta[t]

    The simulator carves the stock by subtracting a swept cylinder for each
    segment, then computes a terminal loss against a target shape. Adam
    (external) optimizes tool_delta to minimize the loss.

    """

    def __init__(
        self,
        resolution=32,
        max_steps=512,
        k_init=10.0,
        target_shape=None,
        tool_start=(0.5, 0.5, 1.0),
        init_taichi=True,
    ):
        # ti.init() RESETS the whole Taichi runtime, invalidating every field
        # allocated by any previously-created simulator. That is fine when each
        # process uses one simulator at a time (training with a single env, or
        # eval scoring checkpoints sequentially), but it corrupts simulators
        # that must co-exist in one process. Pass init_taichi=False to allocate
        # this simulator's fields on the *already-running* runtime instead of
        # resetting it -- Taichi supports adding fields after materialization.
        if init_taichi:
            try:
                if ti._lib.core.with_cuda():
                    ti.init(arch=ti.gpu, debug=False, default_fp=ti.f32)
                else:
                    ti.init(arch=ti.cpu, debug=False, default_fp=ti.f32)
            except:
                pass  # taichi already initialized
            ti.set_logging_level(ti.WARN)

        self.resolution = resolution
        self.max_steps = max_steps
        self.dx = 1.0 / resolution
        self.inv_dx = float(resolution)

        self.k = ti.field(dtype=ti.f32, shape=())  # anneal during training
        self.k[None] = k_init

        # ---- Tool (learnable) ----
        self.tool_delta = ti.Vector.field(
            3, dtype=ti.f32, shape=max_steps, needs_grad=True
        )
        self.tool_pos = ti.Vector.field(
            3, dtype=ti.f32, shape=max_steps + 1, needs_grad=True
        )

        self.tool_start = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.tool_start[None] = ti.Vector(list(tool_start))

        self.tool_radius = ti.field(dtype=ti.f32, shape=())
        self.tool_height = ti.field(dtype=ti.f32, shape=())

        # ---- Tool holder (collision body, not learnable) ----
        # The holder is the wide spindle/collet shaft that sits coaxially ABOVE
        # the slender cutting flutes. It never removes material, but if it ever
        # touches the remaining stock that is a crash: the spindle would slam
        # into the workpiece. We model it as a cylinder of radius holder_radius
        # whose bottom face is at the top of the tool (tool tip + tool_height)
        # and which extends upward by holder_height.
        #
        # Default radius is a 2.5 inch diameter holder expressed in unit-cube
        # coordinates. The unit cube is workspace_mm on a side (MachineConfig,
        # default 100 mm), so r = (2.5 in * 25.4 mm/in / 2) / 100 mm = 0.3175.
        # Call sites that know their own workspace scale should overwrite this.
        self.holder_radius = ti.field(dtype=ti.f32, shape=())
        self.holder_height = ti.field(dtype=ti.f32, shape=())
        self.holder_radius[None] = 0.3175
        self.holder_height[None] = 1.0  # tall enough to clear the whole domain

        # ---- Loss balancing (constrained-optimization framing) ----
        # The terminal loss is a SOFT OBJECTIVE plus two ONE-SIDED BARRIERS:
        #
        #   objective  : w_residual * residual   (material left outside the part)
        #                -> minimizing this is what *rewards* cutting material away
        #   barrier A  : w_gouge   * gouge       (material removed from inside part)
        #                -> one-sided: only fires when the cutter eats into the part
        #   barrier B  : holder penetration into the stock (see below)
        #
        # Keeping w_gouge >= w_residual puts the stock surface just OUTSIDE the
        # part surface: as close as possible without cutting in. The barriers are
        # heavy because they encode "do not violate", while residual is the light
        # objective optimized up against them.
        self.w_gouge = ti.field(dtype=ti.f32, shape=())
        self.w_residual = ti.field(dtype=ti.f32, shape=())
        self.w_gouge[None] = 2.0
        self.w_residual[None] = 1.0

        # Holder collision is a PENETRATION BARRIER, not a proximity field: it is
        # exactly zero (zero gradient) while the holder has clearance, and grows
        # with the squared depth the holder pushes into remaining stock. This is
        # what lets the optimizer bring the holder right up to the surface (so the
        # cutter removes the most material) and only resists actual contact.
        # holder_margin (in unit-cube length) requests a standoff: the barrier
        # starts engaging when stock comes within holder_margin of the holder.
        self.holder_penalty_weight = ti.field(dtype=ti.f32, shape=())
        self.holder_penalty_weight[None] = 50.0
        self.holder_margin = ti.field(dtype=ti.f32, shape=())
        self.holder_margin[None] = 0.0

        # Diagnostics (non-differentiable read-outs of each loss component so the
        # objective/barrier balance is observable during training).
        self.diag_gouge = ti.field(dtype=ti.f32, shape=())
        self.diag_residual = ti.field(dtype=ti.f32, shape=())
        self.diag_holder = ti.field(dtype=ti.f32, shape=())

        # ---- Stock ----
        self.stock = ti.field(
            dtype=ti.f32,
            shape=(max_steps + 1, resolution, resolution, resolution),
            needs_grad=True,
        )
        self.stock_volume = ti.field(dtype=ti.f32, shape=())

        # ---- Target ----
        target_options = ["box", "cylinder", "sphere", "pyramid"]
        if target_shape is None:
            target_shape = random.choice(target_options)
        self.target_shape = target_shape
        self.target_params = {}
        self.target_volume = ti.field(dtype=ti.f32, shape=())
        self._init_target_fields()

        self.target = ti.field(
            dtype=ti.f32, shape=(resolution, resolution, resolution)
        )  # used for evaluation

        # ---- Loss ----
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        # ---- Rendering state ----
        self.current_step = ti.field(dtype=ti.i32, shape=())
        self.current_step[None] = 0
        self.raymarch_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(1024, 768))

        ti.root.lazy_grad()

    def _init_target_fields(self):
        if self.target_shape == "box":
            self.target_params["half_size"] = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.target_params["center"] = ti.Vector.field(3, dtype=ti.f32, shape=())
        elif self.target_shape == "cylinder":
            self.target_params["radius"] = ti.field(dtype=ti.f32, shape=())
            self.target_params["height"] = ti.field(dtype=ti.f32, shape=())
        elif self.target_shape == "sphere":
            self.target_params["radius"] = ti.field(dtype=ti.f32, shape=())
            self.target_params["center"] = ti.Vector.field(3, dtype=ti.f32, shape=())
        elif self.target_shape == "pyramid":
            self.target_params["base_half_size"] = ti.Vector.field(
                3, dtype=ti.f32, shape=()
            )
            self.target_params["height"] = ti.field(dtype=ti.f32, shape=())
            self.target_params["center"] = ti.Vector.field(3, dtype=ti.f32, shape=())
        else:
            raise ValueError(f"Unsupported target shape: {self.target_shape}")

    # ========================================================================
    # SDFs
    # ========================================================================

    @ti.func
    def tool_sdf(self, p, t):
        """Smoothed swept-cylinder SDF: tool sweeps from tool_pos[t] to tool_pos[t+1].

        With the delta parametrization, tool_pos[t+1] == tool_pos[t] + tool_delta[t],
        so the swept segment for cut t is exactly the displacement tool_delta[t].
        """
        r = self.tool_radius[None]
        h = self.tool_height[None]
        kv = self.k[None]

        a = self.tool_pos[t]
        b = self.tool_pos[t + 1]

        # --- Distance in XY to the swept segment (a capsule axis) ---
        pa_xy = ti.Vector([p.x - a.x, p.y - a.y])
        ba_xy = ti.Vector([b.x - a.x, b.y - a.y])
        ba_len2 = ba_xy.dot(ba_xy) + 1e-12
        h_param = ti.max(0.0, ti.min(1.0, pa_xy.dot(ba_xy) / ba_len2))
        closest_xy = ti.Vector([a.x, a.y]) + ba_xy * h_param
        d_xy = ti.sqrt((p.x - closest_xy.x) ** 2 + (p.y - closest_xy.y) ** 2 + 1e-8) - r

        # --- Z extent: tool z at the closest point along the segment ---
        # Linearly interpolate the tool's base z between a and b.
        z_base = a.z + (b.z - a.z) * h_param
        z_center = z_base + 0.5 * h
        d_z = ti.sqrt((p.z - z_center) ** 2 + 1e-8) - 0.5 * h

        # --- Combine (same smooth-CSG combination as before) ---
        d_xy_pos = smooth_max(d_xy, 0.0, kv)
        d_z_pos = smooth_max(d_z, 0.0, kv)
        outside = ti.sqrt(d_xy_pos * d_xy_pos + d_z_pos * d_z_pos + 1e-8)
        inside = -smooth_max(-smooth_max(d_xy, d_z, kv), 0.0, kv)
        return outside + inside

    @ti.func
    def holder_sdf(self, p, t):
        """Smoothed swept-cylinder SDF for the tool holder over segment t.

        Geometry mirrors ``tool_sdf`` (a capsule axis in XY swept from
        tool_pos[t] to tool_pos[t+1]) but with the holder radius and a z-range
        that begins at the TOP of the tool. The holder therefore tracks the
        tool laterally while riding above the cutting flutes. Differentiable in
        tool_pos (hence tool_delta), so the collision penalty has gradients.
        """
        r = self.holder_radius[None]
        h = self.holder_height[None]
        tool_h = self.tool_height[None]
        kv = self.k[None]

        a = self.tool_pos[t]
        b = self.tool_pos[t + 1]

        pa_xy = ti.Vector([p.x - a.x, p.y - a.y])
        ba_xy = ti.Vector([b.x - a.x, b.y - a.y])
        ba_len2 = ba_xy.dot(ba_xy) + 1e-12
        h_param = ti.max(0.0, ti.min(1.0, pa_xy.dot(ba_xy) / ba_len2))
        closest_xy = ti.Vector([a.x, a.y]) + ba_xy * h_param
        d_xy = ti.sqrt((p.x - closest_xy.x) ** 2 + (p.y - closest_xy.y) ** 2 + 1e-8) - r

        # Holder bottom sits at (interpolated tool tip z) + tool_height; the
        # holder body spans [z_bottom, z_bottom + h].
        z_base = a.z + (b.z - a.z) * h_param
        z_bottom = z_base + tool_h
        z_center = z_bottom + 0.5 * h
        d_z = ti.sqrt((p.z - z_center) ** 2 + 1e-8) - 0.5 * h

        d_xy_pos = smooth_max(d_xy, 0.0, kv)
        d_z_pos = smooth_max(d_z, 0.0, kv)
        outside = ti.sqrt(d_xy_pos * d_xy_pos + d_z_pos * d_z_pos + 1e-8)
        inside = -smooth_max(-smooth_max(d_xy, d_z, kv), 0.0, kv)
        return outside + inside

    @ti.func
    def target_sdf(self, p):
        """Target shape SDF — branches resolved at compile time via ti.static."""
        d = 0.0

        if ti.static(self.target_shape == "sphere"):
            d = sphere_sdf(
                p,
                self.target_params["center"][None],
                self.target_params["radius"][None],
            )
        elif ti.static(self.target_shape == "box"):
            d = box_sdf(
                p,
                self.target_params["center"][None],
                self.target_params["half_size"][None],
            )
        elif ti.static(self.target_shape == "cylinder"):
            d = cylinder_sdf(
                p,
                ti.Vector([0.5, 0.5, 0.5]),
                self.target_params["radius"][None],
                self.target_params["height"][None],
            )
        elif ti.static(self.target_shape == "pyramid"):
            d = pyramid_sdf(
                p,
                self.target_params["center"][None],
                self.target_params["base_half_size"][None],
                self.target_params["height"][None],
            )
        return d

    @ti.kernel
    def set_target_volume(self):
        """have one function to compute the target volume based on parameters"""
        count = 0
        for i, j, k in ti.ndrange(self.resolution, self.resolution, self.resolution):
            p = ti.Vector(
                [(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx]
            )
            if self.target_sdf(p) < 0:
                count += 1
        self.target_volume[None] = count * (self.dx**3)

    @ti.kernel
    def init_stock(self):
        """Initial stock: unit cube SDF in stock[0]."""
        for i, j, k in ti.ndrange(self.resolution, self.resolution, self.resolution):
            p = ti.Vector(
                [(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx]
            )
            self.stock[0, i, j, k] = box_sdf(
                p, ti.Vector([0.5, 0.5, 0.5]), ti.Vector([0.5, 0.5, 0.5])
            )

    @ti.kernel
    def bake_target_grid(self):
        for i, j, k in ti.ndrange(self.resolution, self.resolution, self.resolution):
            p = ti.Vector(
                [
                    (i + 0.5) * self.dx,
                    (j + 0.5) * self.dx,
                    (k + 0.5) * self.dx,
                ]
            )

            self.target[i, j, k] = self.target_sdf(p)

    # ========================================================================
    # Forward pass: reconstruct positions → init → carving → loss
    # ========================================================================

    @ti.kernel
    def reconstruct_positions(self, T: ti.i32):
        """Cumulative-sum scan: tool_pos[t+1] = tool_pos[t] + tool_delta[t].

        TIt MUST run serially — each iteration depends on the previous one's result — 
        so ti.loop_config(serialize=True) is mandatory. 
        Without it, Taichi would parallelize the top-level loop
        and the scan would be wrong (and so would its gradient).

        It is gradient-tracked: tool_pos.grad accumulates the indirect-path
        contributions, and Taichi's autodiff propagates those back into
        tool_delta.grad. Run this inside the same ti.ad.Tape as the carving.
        """
        ti.loop_config(serialize=True)
        for _ in range(1):
            self.tool_pos[0] = self.tool_start[None]
            for t in range(T):
                self.tool_pos[t + 1] = self.tool_pos[t] + self.tool_delta[t]

    @ti.kernel
    def zero_tool_deltas(self):
        for t in range(self.max_steps):
            self.tool_delta[t] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_cut(self, t: ti.i32):
        """stock[t+1] = smooth_max(stock[t], -tool_sdf at segment t)."""
        for i, j, k in ti.ndrange(self.resolution, self.resolution, self.resolution):
            kv = self.k[None]  # moved inside loop for autodiff
            p = ti.Vector(
                [(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx]
            )
            tool_d = self.tool_sdf(p, t)
            self.stock[t + 1, i, j, k] = smooth_max(self.stock[t, i, j, k], -tool_d, kv)

    @ti.kernel
    def loss_at(self, t: ti.i32) -> ti.f32:
        """Exact replica of compute_loss's objective, evaluated on stock[t].

        Same soft-occupancy formulation, same weights, same sigmoid scale.
        Returns the scalar instead of writing to self.loss, and reads no grad —
        safe to call outside a Tape (for RL reward / eval).
        """
        total = 0.0
        for i, j, k in ti.ndrange(self.resolution, self.resolution, self.resolution):
            scale = self.inv_dx
            inv_n = 1.0 / (self.resolution ** 3)
            p = ti.Vector([(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx])
            stock_d = self.stock[t, i, j, k]
            target_d = self.target_sdf(p)

            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            ta = ti.max(-50.0, ti.min(50.0, target_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))
            target_occ = 1.0 / (1.0 + ti.exp(ta))

            gouge = target_occ * (1.0 - stock_occ)
            residual = (1.0 - target_occ) * stock_occ

            w_gouge = self.w_gouge[None]
            w_residual = self.w_residual[None]
            total += inv_n * (w_gouge * gouge * gouge + w_residual * residual * residual)
        return total

    @ti.kernel
    def compute_loss(self, T: ti.i32):
        """Terminal occupancy loss: weighted gouge² + residual²."""
        for i, j, k in ti.ndrange(self.resolution, self.resolution, self.resolution):
            # All scalar reads moved inside for autodiff compatibility.
            scale = self.inv_dx
            inv_n = 1.0 / (self.resolution**3)
            p = ti.Vector(
                [(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx]
            )
            stock_d = self.stock[T, i, j, k]
            target_d = self.target_sdf(p)

            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            ta = ti.max(-50.0, ti.min(50.0, target_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))
            target_occ = 1.0 / (1.0 + ti.exp(ta))

            gouge = target_occ * (1.0 - stock_occ)
            residual = (1.0 - target_occ) * stock_occ

            w_gouge = self.w_gouge[None]
            w_residual = self.w_residual[None]
            ti.atomic_add(
                self.loss[None],
                inv_n * (w_gouge * gouge * gouge + w_residual * residual * residual),
            )

    @ti.kernel
    def compute_holder_penalty(self, T: ti.i32):
        """Differentiable holder-collision PENETRATION BARRIER added to ``self.loss``.

        For every cut t the holder rides above the tool along segment t. For each
        remaining-material voxel we measure how far the holder pushes INTO it:

            penetration = relu( (holder_margin - holder_sdf) * scale )

        This is exactly zero -- with zero gradient -- whenever the holder has
        clearance (holder_sdf >= holder_margin), so the optimizer is free to
        bring the holder right down to the surface and remove the most material.
        It only grows (quadratically) once the holder actually penetrates the
        stock, pushing the holder back out. Gated by stock occupancy so only
        contact with REMAINING material is penalized (empty space is free).

        This is a one-sided barrier, not a proximity field: a high weight makes
        it a near-hard constraint that stays inactive until violated, instead of
        a force that shoves the tool away from the workpiece pre-emptively.
        """
        for t, i, j, k in ti.ndrange(T, self.resolution, self.resolution, self.resolution):
            scale = self.inv_dx
            inv_n = 1.0 / (self.resolution ** 3)
            w = self.holder_penalty_weight[None]
            margin = self.holder_margin[None]
            p = ti.Vector(
                [(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx]
            )
            stock_d = self.stock[t + 1, i, j, k]
            holder_d = self.holder_sdf(p, t)

            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))    # ~1 inside remaining stock
            penetration = ti.max(0.0, (margin - holder_d) * scale)

            ti.atomic_add(self.loss[None], inv_n * w * stock_occ * penetration * penetration)

    @ti.kernel
    def compute_diagnostics(self, T: ti.i32):
        """Non-differentiable breakdown of the loss into its three components.

        Fills diag_gouge / diag_residual / diag_holder with the SAME weighted
        terms that compute_loss + compute_holder_penalty add to ``self.loss``,
        so training can log how the objective (residual) trades off against the
        two barriers (gouge, holder). Reads no grad; call outside a Tape.
        """
        g = 0.0
        r = 0.0
        h = 0.0
        scale = self.inv_dx
        inv_n = 1.0 / (self.resolution ** 3)
        w_g = self.w_gouge[None]
        w_r = self.w_residual[None]
        w_h = self.holder_penalty_weight[None]
        margin = self.holder_margin[None]
        # Geometry terms on the final stock (stock[T]).
        for i, j, k in ti.ndrange(self.resolution, self.resolution, self.resolution):
            p = ti.Vector([(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx])
            stock_d = self.stock[T, i, j, k]
            target_d = self.target_sdf(p)
            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            ta = ti.max(-50.0, ti.min(50.0, target_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))
            target_occ = 1.0 / (1.0 + ti.exp(ta))
            gouge = target_occ * (1.0 - stock_occ)
            residual = (1.0 - target_occ) * stock_occ
            g += inv_n * w_g * gouge * gouge
            r += inv_n * w_r * residual * residual
        # Holder barrier summed over every segment.
        for t, i, j, k in ti.ndrange(T, self.resolution, self.resolution, self.resolution):
            p = ti.Vector([(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx])
            stock_d = self.stock[t + 1, i, j, k]
            holder_d = self.holder_sdf(p, t)
            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))
            penetration = ti.max(0.0, (margin - holder_d) * scale)
            h += inv_n * w_h * stock_occ * penetration * penetration
        self.diag_gouge[None] = g
        self.diag_residual[None] = r
        self.diag_holder[None] = h

    @ti.kernel
    def holder_overlap_at(self, t: ti.i32) -> ti.f32:
        """Hard (non-diff) holder/stock overlap VOLUME for a single segment t.

        Counts remaining-material voxels (stock[t+1] < 0) that lie inside the
        holder (sharp SDF < 0) and returns their volume in unit-cube^3. A
        positive value means the holder is contacting the stock -- used by the
        RL env to terminate the episode. Safe to call outside a Tape.
        """
        vol = 0.0
        for i, j, k in ti.ndrange(self.resolution, self.resolution, self.resolution):
            p = ti.Vector(
                [(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx]
            )
            if self.stock[t + 1, i, j, k] < 0.0 and self.holder_sdf_sharp(p, t) < 0.0:
                vol += self.dx ** 3
        return vol

    @ti.kernel
    def holder_overlap_total(self, T: ti.i32) -> ti.f32:
        """Hard holder/stock overlap summed over all segments (diagnostics).

        Sum of per-segment overlap volume; > 0 means the trajectory collides
        the holder with the stock somewhere. Non-differentiable.
        """
        vol = 0.0
        for t, i, j, k in ti.ndrange(T, self.resolution, self.resolution, self.resolution):
            p = ti.Vector(
                [(i + 0.5) * self.dx, (j + 0.5) * self.dx, (k + 0.5) * self.dx]
            )
            if self.stock[t + 1, i, j, k] < 0.0 and self.holder_sdf_sharp(p, t) < 0.0:
                vol += self.dx ** 3
        return vol

    def forward(self, num_active_steps):
        """Pure forward pass. Wrap in ti.ad.Tape externally if you need gradients.

        num_active_steps is the number of tool positions in use; with
        num_active_steps positions there are num_active_steps-1 segments/cuts.
        reconstruct_positions is part of the forward pass and must be inside
        the same Tape as the carving for tool_delta.grad to be correct.
        """
        self.reconstruct_positions(num_active_steps - 1)
        self.init_stock()
        for t in range(num_active_steps - 1):
            self.apply_cut(t)
        self.compute_loss(num_active_steps - 1)
        self.compute_holder_penalty(num_active_steps - 1)

    # ========================================================================
    # Rendering
    # ========================================================================

    @ti.func
    def interpolate_stock(self, p):
        """Trilinear lookup into stock[current_step, ...]."""
        t_idx = self.current_step[None]
        p_grid = p * self.resolution

        x0 = ti.cast(ti.floor(p_grid.x), ti.i32)
        y0 = ti.cast(ti.floor(p_grid.y), ti.i32)
        z0 = ti.cast(ti.floor(p_grid.z), ti.i32)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        tx, ty, tz = p_grid.x - x0, p_grid.y - y0, p_grid.z - z0

        R = self.resolution
        x0 = ti.max(0, ti.min(R - 1, x0))
        x1 = ti.max(0, ti.min(R - 1, x1))
        y0 = ti.max(0, ti.min(R - 1, y0))
        y1 = ti.max(0, ti.min(R - 1, y1))
        z0 = ti.max(0, ti.min(R - 1, z0))
        z1 = ti.max(0, ti.min(R - 1, z1))

        c000 = self.stock[t_idx, x0, y0, z0]
        c100 = self.stock[t_idx, x1, y0, z0]
        c010 = self.stock[t_idx, x0, y1, z0]
        c110 = self.stock[t_idx, x1, y1, z0]
        c001 = self.stock[t_idx, x0, y0, z1]
        c101 = self.stock[t_idx, x1, y0, z1]
        c011 = self.stock[t_idx, x0, y1, z1]
        c111 = self.stock[t_idx, x1, y1, z1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    @ti.func
    def stock_normal(self, p):
        """Approximate surface normal of stock via central differences."""
        eps = self.dx * 1.5
        dx = ti.Vector([eps, 0.0, 0.0])
        dy = ti.Vector([0.0, eps, 0.0])
        dz = ti.Vector([0.0, 0.0, eps])
        nx = self.interpolate_stock(p + dx) - self.interpolate_stock(p - dx)
        ny = self.interpolate_stock(p + dy) - self.interpolate_stock(p - dy)
        nz = self.interpolate_stock(p + dz) - self.interpolate_stock(p - dz)
        return ti.math.normalize(ti.Vector([nx, ny, nz]) + 1e-8)

    @ti.func
    def target_normal(self, p):
        """Analytic-ish normal of target via central differences on target_sdf."""
        eps = self.dx * 1.5
        dx = ti.Vector([eps, 0.0, 0.0])
        dy = ti.Vector([0.0, eps, 0.0])
        dz = ti.Vector([0.0, 0.0, eps])
        nx = self.target_sdf(p + dx) - self.target_sdf(p - dx)
        ny = self.target_sdf(p + dy) - self.target_sdf(p - dy)
        nz = self.target_sdf(p + dz) - self.target_sdf(p - dz)
        return ti.math.normalize(ti.Vector([nx, ny, nz]) + 1e-8)

    @ti.func
    def tool_normal(self, p, t):
        """Central-diff normal of the SHARP tool SDF (rendering only)."""
        eps = 1e-3
        dx = ti.Vector([eps, 0.0, 0.0])
        dy = ti.Vector([0.0, eps, 0.0])
        dz = ti.Vector([0.0, 0.0, eps])
        nx = self.tool_sdf_sharp(p + dx, t) - self.tool_sdf_sharp(p - dx, t)
        ny = self.tool_sdf_sharp(p + dy, t) - self.tool_sdf_sharp(p - dy, t)
        nz = self.tool_sdf_sharp(p + dz, t) - self.tool_sdf_sharp(p - dz, t)
        return ti.math.normalize(ti.Vector([nx, ny, nz]) + 1e-8)

    @ti.kernel
    def render_raymarch(
        self,
        cam_pos: ti.types.vector(3, ti.f32),
        cam_up: ti.types.vector(3, ti.f32),
        cam_dir: ti.types.vector(3, ti.f32),
        show_stock: ti.i32,
        show_target: ti.i32,
        show_tool: ti.i32,
    ):
        """Raymarch the scene and fill raymarch_buffer with RGB."""
        cam_right = cam_dir.cross(cam_up).normalized()
        cam_up_actual = cam_right.cross(cam_dir).normalized()
        fov_scale = ti.tan(3.14159 / 4.0)
        width = self.raymarch_buffer.shape[0]
        height = self.raymarch_buffer.shape[1]
        aspect_ratio = float(width) / float(height)

        t_idx = self.current_step[None]

        for i, j in self.raymarch_buffer:
            u = (2.0 * (i + 0.5) / float(width) - 1.0) * aspect_ratio * fov_scale
            v = (2.0 * (j + 0.5) / float(height) - 1.0) * fov_scale
            ray_dir = (cam_dir + cam_right * u + cam_up_actual * v).normalized()

            t = 0.0
            max_t = 10.0
            max_steps = 150
            color = ti.Vector([0.1, 0.1, 0.1])  # background

            for _step in range(max_steps):
                p = cam_pos + ray_dir * t
                d_stock = 1e6
                d_target = 1e6
                d_tool = 1e6
                d_holder = 1e6

                # Inside the unit cube — use interpolated voxel SDFs.
                # Outside — fall back to an AABB so rays from the camera
                # can still hit something on their way in.
                inside_cube = (
                    0.0 <= p.x
                    and p.x <= 1.0
                    and 0.0 <= p.y
                    and p.y <= 1.0
                    and 0.0 <= p.z
                    and p.z <= 1.0
                )

                if inside_cube:
                    if show_stock == 1:
                        d_stock = self.interpolate_stock(p)
                    if show_target == 1:
                        d_target = self.target_sdf(p)
                else:
                    d_box = p - ti.Vector([0.5, 0.5, 0.5])
                    d_aabb = (
                        ti.max(
                            ti.abs(d_box.x), ti.max(ti.abs(d_box.y), ti.abs(d_box.z))
                        )
                        - 0.5
                    )
                    if show_stock == 1:
                        d_stock = ti.max(d_aabb, 2e-3)
                    if show_target == 1:
                        d_target = ti.max(d_aabb, 2e-3)

                if show_tool == 1:
                    d_tool = self.tool_sdf_sharp(p, t_idx)
                    d_holder = self.holder_sdf_sharp(p, t_idx)

                d = ti.min(d_stock, ti.min(d_target, ti.min(d_tool, d_holder)))

                if d < 1e-3:
                    # Hit — shade with the material whose distance dominated.
                    mat_color = ti.Vector([0.8, 0.8, 0.8])
                    norm = ti.Vector([0.0, 0.0, 1.0])

                    if d == d_tool:
                        mat_color = ti.Vector([1.0, 0.2, 0.2])  # red tool
                        norm = self.tool_normal(p, t_idx)
                    elif d == d_holder:
                        mat_color = ti.Vector([0.55, 0.55, 0.6])  # gray holder
                        norm = self.holder_normal(p, t_idx)
                    elif d == d_stock:
                        mat_color = ti.Vector([0.2, 0.8, 0.2])  # green stock
                        norm = self.stock_normal(p)
                        # Voxel-checker shading: darken every other voxel so
                        # the discrete grid is visible. This is what gives the
                        # "voxelized" look and makes individual cuts pop.
                        grid_p = p * float(self.resolution)
                        cx = int(grid_p.x) % 2
                        cy = int(grid_p.y) % 2
                        cz = int(grid_p.z) % 2
                        if (cx + cy + cz) % 2 == 0:
                            mat_color = mat_color * 0.8
                    elif d == d_target:
                        mat_color = ti.Vector([0.5, 0.5, 1.0])  # light blue target
                        norm = self.target_normal(p)

                    light_dir = ti.Vector([1.0, 1.0, 1.0]).normalized()
                    diffuse = ti.max(0.0, norm.dot(light_dir))
                    ambient = 0.2
                    color = mat_color * (diffuse * 0.8 + ambient)
                    break

                t += d
                if t > max_t:
                    break

            self.raymarch_buffer[i, j] = color

    # ----- Python-side helpers around rendering -----

    def set_current_step(self, t):
        """Tell the renderer which stock slot and tool position to draw.

        After apply_cut(t), the freshly-cut stock lives at slot t+1, so a
        typical call is set_current_step(t+1).
        """
        self.current_step[None] = int(t)

    def render(
        self,
        cam_pos=(2.0, 2.0, 2.0),
        cam_target=(0.5, 0.5, 0.5),
        cam_up=(0.0, 0.0, 1.0),
        show_stock=True,
        show_target=False,
        show_tool=True,
    ):
        """Render the current scene into raymarch_buffer.

        cam_pos: camera position in world coords.
        cam_target: point the camera looks at.
        cam_up: world up vector.
        show_*: toggles for each layer.
        """
        cp = ti.Vector(list(cam_pos))
        ct = ti.Vector(list(cam_target))
        cu = ti.Vector(list(cam_up))
        cd = (ct - cp).normalized()
        self.render_raymarch(
            cp,
            cu,
            cd,
            int(show_stock),
            int(show_target),
            int(show_tool),
        )

    @ti.func
    def tool_sdf_sharp(self, p, t):
        """Exact (non-smoothed) capped-cylinder SDF for rendering only.

        Same geometry as tool_sdf but uses hard ti.max instead of smooth_max,
        so the rendered cylinder has crisp edges. Not differentiable — never
        call this from a kernel that runs under ti.ad.Tape.
        """

        r = self.tool_radius[None]
        h = self.tool_height[None]
        kv = self.k[None]

        a = self.tool_pos[t]
        b = self.tool_pos[t + 1]

        # --- Distance in XY to the swept segment (a capsule axis) ---
        pa_xy = ti.Vector([p.x - a.x, p.y - a.y])
        ba_xy = ti.Vector([b.x - a.x, b.y - a.y])
        ba_len2 = ba_xy.dot(ba_xy) + 1e-12
        h_param = ti.max(0.0, ti.min(1.0, pa_xy.dot(ba_xy) / ba_len2))
        closest_xy = ti.Vector([a.x, a.y]) + ba_xy * h_param
        d_xy = ti.sqrt((p.x - closest_xy.x) ** 2 + (p.y - closest_xy.y) ** 2 + 1e-8) - r

        # --- Z extent: tool z at the closest point along the segment ---
        # Linearly interpolate the tool's base z between a and b.
        z_base = a.z + (b.z - a.z) * h_param
        z_center = z_base + 0.5 * h
        d_z = ti.sqrt((p.z - z_center) ** 2 + 1e-8) - 0.5 * h

        # --- Combine (hard max for crisp edges) ---
        d_xy_pos = ti.max(d_xy, 0.0)
        d_z_pos = ti.max(d_z, 0.0)
        outside = ti.sqrt(d_xy_pos * d_xy_pos + d_z_pos * d_z_pos + 1e-8)
        inside = -ti.max(-ti.max(d_xy, d_z), 0.0)
        return outside + inside

    @ti.func
    def holder_sdf_sharp(self, p, t):
        """Exact (non-smoothed) holder SDF for collision detection and rendering.

        Same geometry as ``holder_sdf`` but with hard ti.max. Not
        differentiable -- never call under ti.ad.Tape.
        """
        r = self.holder_radius[None]
        h = self.holder_height[None]
        tool_h = self.tool_height[None]

        a = self.tool_pos[t]
        b = self.tool_pos[t + 1]

        pa_xy = ti.Vector([p.x - a.x, p.y - a.y])
        ba_xy = ti.Vector([b.x - a.x, b.y - a.y])
        ba_len2 = ba_xy.dot(ba_xy) + 1e-12
        h_param = ti.max(0.0, ti.min(1.0, pa_xy.dot(ba_xy) / ba_len2))
        closest_xy = ti.Vector([a.x, a.y]) + ba_xy * h_param
        d_xy = ti.sqrt((p.x - closest_xy.x) ** 2 + (p.y - closest_xy.y) ** 2 + 1e-8) - r

        z_base = a.z + (b.z - a.z) * h_param
        z_bottom = z_base + tool_h
        z_center = z_bottom + 0.5 * h
        d_z = ti.sqrt((p.z - z_center) ** 2 + 1e-8) - 0.5 * h

        d_xy_pos = ti.max(d_xy, 0.0)
        d_z_pos = ti.max(d_z, 0.0)
        outside = ti.sqrt(d_xy_pos * d_xy_pos + d_z_pos * d_z_pos + 1e-8)
        inside = -ti.max(-ti.max(d_xy, d_z), 0.0)
        return outside + inside

    @ti.func
    def holder_normal(self, p, t):
        """Central-diff normal of the SHARP holder SDF (rendering only)."""
        eps = 1e-3
        dx = ti.Vector([eps, 0.0, 0.0])
        dy = ti.Vector([0.0, eps, 0.0])
        dz = ti.Vector([0.0, 0.0, eps])
        nx = self.holder_sdf_sharp(p + dx, t) - self.holder_sdf_sharp(p - dx, t)
        ny = self.holder_sdf_sharp(p + dy, t) - self.holder_sdf_sharp(p - dy, t)
        nz = self.holder_sdf_sharp(p + dz, t) - self.holder_sdf_sharp(p - dz, t)
        return ti.math.normalize(ti.Vector([nx, ny, nz]) + 1e-8)
