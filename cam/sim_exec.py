"""Execute a position trajectory in the differentiable simulator.

This is the bridge used for the *carved-stock* success measure: given any
``(M, 3)`` trajectory of unit-cube tool positions, drive ``CSGSimulatorDelta`` by
setting its fixed start to ``positions[0]`` and its per-step deltas to the
successive displacements, then run the forward carve and return the final stock
SDF grid.

Importing this module pulls in Taichi; the top-level ``cam`` package does not, so
pure G-code work stays lightweight.
"""

import numpy as np
import taichi as ti

from simulator.csg_simulator import CSGSimulatorDelta


@ti.data_oriented
class _HardCarveSimulator(CSGSimulatorDelta):
    """Simulator variant that carves with a *hard* boolean union.

    The base class carves with a soft ``smooth_max`` union for differentiability.
    That soft union adds a small ``log(2)/k`` bias per step, so the carved result
    depends on how many steps the path is split into — a dense executed
    trajectory (hundreds of samples) erodes far more than the same path described
    by a few waypoints. Real machining is a *hard* boolean subtraction, which is
    idempotent and therefore step-count invariant. We reuse the parent's existing
    sharp capped-cylinder SDF (``tool_sdf_sharp``) to get exactly that.
    """

    def forward_hard(self, num_active_steps, clip_speeds=False):
        super().forward_hard(num_active_steps, clip_speeds=clip_speeds)


def carve_stock(
    positions,
    resolution=24,
    target_shape="sphere",
    tool_radius=3.175,   # mm (1/4" end mill)
    tool_height=25.0,    # mm
    stock_size_in=(1.0, 1.0, 1.0),   # stock box (the normalized cube), inches
    voxel_size_mm=0.5,               # sub-mm precision knob
    work_volume_in=(16.0, 12.0, 10.0),
    stock_origin_in=None,
):
    """Run ``positions`` through the simulator and return the final stock SDF grid.

    Uses a hard (idempotent) CSG carve so the result depends only on the
    geometric path, not on how finely it is sampled. A fresh simulator is
    constructed (which re-initialises Taichi), so callers comparing two
    trajectories should carve one, copy the result to NumPy, then carve the next.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError(f"positions must be (M>=2, 3); got {positions.shape}")

    n_pos = len(positions)
    deltas = np.diff(positions, axis=0)            # (n_pos-1, 3)

    sim = _HardCarveSimulator(
        resolution=resolution,
        max_steps=n_pos - 1,
        target_shape=target_shape,
        tool_start=tuple(float(v) for v in positions[0]),
        stock_size_in=stock_size_in,
        voxel_size_mm=voxel_size_mm,
        work_volume_in=work_volume_in,
        stock_origin_in=stock_origin_in,
    )
    sim.tool_radius[None] = tool_radius
    sim.tool_height[None] = tool_height

    padded = np.zeros((sim.max_steps, 3), dtype=np.float32)
    padded[: len(deltas)] = deltas
    sim.tool_delta.from_numpy(padded)

    sim.forward_hard(n_pos)                         # carves n_pos-1 segments
    stock = sim.stock.to_numpy()[n_pos - 1]
    return stock
