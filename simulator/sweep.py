"""One-shot differentiable swept-volume carve for spline toolpath optimization.

The delta method carves sequentially (``stock[t+1] = smooth_max(stock[t],
-tool_sdf)``), which costs T serial N^3 kernels per forward, a ``(T+1) x N^3``
autodiff stock history, and accumulates ~log(2)/k of soft-union erosion per
step (the documented soft/hard dice gap). But material removal by a rigid tool
is ORDER-INDEPENDENT: the final geometry is

    carved(x) = max(stock0(x), -d_swept(x)),   d_swept(x) = min_s seg_sdf(x, s)

— a single union with the min over all path segments of the same sharp
swept-cylinder SDF the hard evaluator uses (``tool_sdf_sharp``). This module
computes that swept carve and its gradient w.r.t. the sampled path points in
two N^3 passes, with no stock history:

  1. ``find_argmin`` (non-diff): per voxel, brute-force the winning segment
     index into ``amin`` (and the winning clamp state of the capsule
     projection). Exact hard min.
  2. ``compute_loss`` (under ``ti.ad.Tape``): re-evaluate ONLY the winning
     segment per voxel — straight-line code, so Taichi autodiff yields the
     exact hard-min subgradient (the maxpool trick) — and accumulate the same
     soft-occupancy ``w_g*gouge^2 + w_r*residual^2`` loss as
     ``CSGSimulatorDelta.compute_loss`` (1-voxel sigmoid), plus an optional
     non-saturating attraction term (``w_broad``): relu(d_tool)^2 on uncut
     waste voxels, whose gradient does not vanish with distance from the
     swept tube (the sigmoid residual's does).

Distances are in VOXELS (isotropic cubic voxels), mirroring the simulator.
"""

import numpy as np
import taichi as ti


@ti.data_oriented
class SweepCarve:
    """Swept-volume loss over a sampled toolpath, differentiable in the path.

    Reads geometry (grid dims, tool size, target SDF grid, loss weights) from a
    constructed ``CSGSimulatorDelta`` so the objective matches the delta method
    exactly. Owns only the path field, the argmin field, and the loss.
    """

    def __init__(self, sim, n_points):
        self.sim = sim
        self.n_points = n_points  # path samples; n_points-1 segments
        self.Nx, self.Ny, self.Nz = sim.Nx, sim.Ny, sim.Nz

        self.path = ti.Vector.field(3, dtype=ti.f32, shape=n_points, needs_grad=True)
        self.amin = ti.field(dtype=ti.i32, shape=(self.Nx, self.Ny, self.Nz))
        self.carved = ti.field(dtype=ti.f32, shape=(self.Nx, self.Ny, self.Nz))
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        # Loss shaping (defaults mirror the delta method's compute_loss).
        self.w_broad = ti.field(dtype=ti.f32, shape=())      # 0 = disabled
        self.sigma_broad = ti.field(dtype=ti.f32, shape=())  # voxels
        self.w_broad[None] = 0.0
        self.sigma_broad[None] = 4.0

        # Non-diff diagnostics (hard counts on the swept carve).
        self.diag_residual_vox = ti.field(dtype=ti.i32, shape=())
        self.diag_gouge_vox = ti.field(dtype=ti.i32, shape=())

    # ------------------------------------------------------------------
    # Geometry (identical to CSGSimulatorDelta.tool_sdf_sharp, but taking
    # explicit endpoints so it can index any segment of the sampled path).
    # ------------------------------------------------------------------
    @ti.func
    def _seg_sdf(self, pv, a, b):
        """Sharp swept-cylinder SDF (voxels): tool sweeps a -> b.

        pv/a/b are voxel-space coords. XY distance to the segment (capsule
        axis); Z slab tracks the tool base lerped at the XY-closest parameter
        and extends up by tool_height — the exact geometry of tool_sdf_sharp.
        """
        r = self.sim.tool_radius[None] / self.sim.v
        h = self.sim.tool_height[None] / self.sim.v

        pa_xy = ti.Vector([pv.x - a.x, pv.y - a.y])
        ba_xy = ti.Vector([b.x - a.x, b.y - a.y])
        ba_len2 = ba_xy.dot(ba_xy) + 1e-12
        h_param = ti.max(0.0, ti.min(1.0, pa_xy.dot(ba_xy) / ba_len2))
        closest_xy = ti.Vector([a.x, a.y]) + ba_xy * h_param
        d_xy = ti.sqrt((pv.x - closest_xy.x) ** 2 + (pv.y - closest_xy.y) ** 2 + 1e-8) - r

        z_base = a.z + (b.z - a.z) * h_param
        z_center = z_base + 0.5 * h
        d_z = ti.sqrt((pv.z - z_center) ** 2 + 1e-8) - 0.5 * h

        d_xy_pos = ti.max(d_xy, 0.0)
        d_z_pos = ti.max(d_z, 0.0)
        outside = ti.sqrt(d_xy_pos * d_xy_pos + d_z_pos * d_z_pos + 1e-8)
        inside = -ti.max(-ti.max(d_xy, d_z), 0.0)
        return outside + inside

    @ti.func
    def _vox_of_path(self, s):
        p = self.path[s]
        return ti.Vector([p.x * self.Nx, p.y * self.Ny, p.z * self.Nz])

    @ti.func
    def _stock0(self, pv):
        """Initial stock (full envelope box) SDF in voxel space."""
        half = ti.Vector([0.5 * self.Nx, 0.5 * self.Ny, 0.5 * self.Nz])
        d = ti.abs(pv - half) - half
        return ti.max(d.x, ti.max(d.y, d.z))

    # ------------------------------------------------------------------
    # Pass 1: hard argmin over segments (non-differentiable).
    # ------------------------------------------------------------------
    @ti.kernel
    def find_argmin(self, S: ti.i32):
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            pv = ti.Vector([i + 0.5, j + 0.5, k + 0.5])
            best_d = 1e9
            best_s = 0
            for s in range(S):
                d = self._seg_sdf(pv, self._vox_of_path(s), self._vox_of_path(s + 1))
                if d < best_d:
                    best_d = d
                    best_s = s
            self.amin[i, j, k] = best_s

    # ------------------------------------------------------------------
    # Pass 2: loss on the winning segment only (autodiff-safe).
    # ------------------------------------------------------------------
    @ti.kernel
    def compute_loss(self):
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            inv_n = 1.0 / (self.Nx * self.Ny * self.Nz)
            pv = ti.Vector([i + 0.5, j + 0.5, k + 0.5])
            s = self.amin[i, j, k]
            d_tool = self._seg_sdf(pv, self._vox_of_path(s), self._vox_of_path(s + 1))
            carved = ti.max(self._stock0(pv), -d_tool)
            target_d = self.sim.target[i, j, k]

            sa = ti.max(-50.0, ti.min(50.0, carved))
            ta = ti.max(-50.0, ti.min(50.0, target_d))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))
            target_occ = 1.0 / (1.0 + ti.exp(ta))

            gouge = target_occ * (1.0 - stock_occ)
            residual = (1.0 - target_occ) * stock_occ

            w_gouge = self.sim.w_gouge[None]
            w_residual = self.sim.w_residual[None]
            contrib = inv_n * (w_gouge * gouge * gouge + w_residual * residual * residual)

            # Non-saturating residual attraction (SDF-valued): every still-uncut
            # waste voxel pulls its argmin segment with force ~ distance to the
            # swept tube (envelope gradient of relu(d_tool)^2), unlike the
            # sigmoid residual whose gradient dies a few voxels from the tube.
            # Gated by waste (outside the part) and by remaining material
            # (stock_occ ~ 1 until the voxel is actually cut, then the pull
            # dies). One-sided, so it never fights the gouge barrier.
            w_b = self.w_broad[None]
            if w_b > 0.0:
                pull = ti.max(0.0, d_tool) / self.sigma_broad[None]
                contrib += inv_n * w_b * (1.0 - target_occ) * stock_occ * pull * pull

            ti.atomic_add(self.loss[None], contrib)

    # ------------------------------------------------------------------
    # Diagnostics: hard voxel counts on the swept carve (no grad, no sigmoid).
    # ------------------------------------------------------------------
    @ti.kernel
    def count_hard(self):
        self.diag_residual_vox[None] = 0
        self.diag_gouge_vox[None] = 0
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            pv = ti.Vector([i + 0.5, j + 0.5, k + 0.5])
            s = self.amin[i, j, k]
            d_tool = self._seg_sdf(pv, self._vox_of_path(s), self._vox_of_path(s + 1))
            carved = ti.max(self._stock0(pv), -d_tool)
            target_d = self.sim.target[i, j, k]
            if carved < 0.0 and target_d >= 0.0:
                ti.atomic_add(self.diag_residual_vox[None], 1)
            if carved >= 0.0 and target_d < 0.0:
                ti.atomic_add(self.diag_gouge_vox[None], 1)

    @ti.kernel
    def write_carved(self):
        """Fill ``carved`` with the swept-carve SDF (voxels) for diagnostics."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            pv = ti.Vector([i + 0.5, j + 0.5, k + 0.5])
            s = self.amin[i, j, k]
            d_tool = self._seg_sdf(pv, self._vox_of_path(s), self._vox_of_path(s + 1))
            self.carved[i, j, k] = ti.max(self._stock0(pv), -d_tool)

    def hard_carve_mask(self, X):
        """Boolean material mask of the swept carve for path samples ``X``."""
        self.path.from_numpy(np.asarray(X, dtype=np.float32))
        self.find_argmin(self.n_points - 1)
        self.write_carved()
        return self.carved.to_numpy() < 0.0

    # ------------------------------------------------------------------
    # Host-side driver: one forward+backward, returns (loss, grad_X).
    # ------------------------------------------------------------------
    def loss_and_grad(self, X):
        """X: (n_points, 3) float32 normalized path samples.

        Returns (loss_value, grad wrt X as (n_points, 3) float32 array).
        """
        S = self.n_points - 1
        self.path.from_numpy(X.astype(np.float32))
        self.find_argmin(S)
        self.loss[None] = 0.0
        with ti.ad.Tape(loss=self.loss):
            self.compute_loss()
        return float(self.loss[None]), self.path.grad.to_numpy()


# ----------------------------------------------------------------------
# Spline basis + shape-agnostic init (host side, numpy).
# ----------------------------------------------------------------------

def bspline_basis(n_ctrl, n_samples, degree=3):
    """Clamped uniform B-spline design matrix B (n_samples x n_ctrl).

    Endpoint-interpolating: X = B @ P passes through P[0] and P[-1]. Cox-de Boor
    recursion; no scipy dependency.
    """
    k = degree
    n_knots = n_ctrl + k + 1
    knots = np.concatenate([
        np.zeros(k),
        np.linspace(0.0, 1.0, n_knots - 2 * k),
        np.ones(k),
    ])
    u = np.linspace(0.0, 1.0, n_samples)
    # Degree-0 basis.
    Bk = np.zeros((n_samples, n_knots - 1))
    for i in range(n_knots - 1):
        Bk[:, i] = (u >= knots[i]) & (u < knots[i + 1])
    Bk[-1, np.nonzero(knots[:-1] < 1.0)[0][-1]] = 1.0  # include u=1 in last span
    # Elevate degree.
    for d in range(1, k + 1):
        Bn = np.zeros((n_samples, Bk.shape[1] - 1))
        for i in range(Bn.shape[1]):
            den1 = knots[i + d] - knots[i]
            den2 = knots[i + d + 1] - knots[i + 1]
            t1 = (u - knots[i]) / den1 * Bk[:, i] if den1 > 0 else 0.0
            t2 = (knots[i + d + 1] - u) / den2 * Bk[:, i + 1] if den2 > 0 else 0.0
            Bn[:, i] = t1 + t2
        Bk = Bn
    assert Bk.shape == (n_samples, n_ctrl)
    return Bk.astype(np.float32)


def target_bbox(sim, margin_vox=2.0):
    """Normalized bounding box (lo, hi) of the target (SDF<=0) + margin.

    Shape-agnostic: reads only the baked SDF grid. Returns ((3,), (3,)) arrays
    in normalized [0,1] coords, clipped to the unit cube.
    """
    tgt = sim.target.to_numpy()
    idx = np.argwhere(tgt <= 0.0)
    dims = np.array(tgt.shape, dtype=np.float64)
    if len(idx) == 0:
        return np.zeros(3), np.ones(3)
    lo = (idx.min(axis=0) - margin_vox) / dims
    hi = (idx.max(axis=0) + 1 + margin_vox) / dims
    return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)


def init_reference_path(sim, tool_start, n_samples, mode="raster", seed=0):
    """Shape-agnostic reference polyline (n_samples, 3) for the init fit.

    Modes:
      raster — boustrophedon over the target bbox footprint with linearly
               descending z (fine zigzag, the proven coverage pattern);
      helix  — descending spiral shrinking from the bbox wall to its center;
      random — small random walk below the start (sanity baseline).
    The first point is ``tool_start``; z descends from the stock top toward the
    target bbox bottom (the z-floor clamp in the trainer keeps it legal).
    """
    rng = np.random.default_rng(seed)
    lo, hi = target_bbox(sim)
    n = n_samples
    pts = np.zeros((n, 3), dtype=np.float64)
    if mode == "helix":
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        rx, ry = 0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1])
        revs = 10.0
        for t in range(n):
            f = t / max(1, n - 1)
            ph = 2.0 * np.pi * revs * f
            shrink = 1.0 - 0.85 * f
            pts[t, 0] = cx + rx * shrink * np.cos(ph)
            pts[t, 1] = cy + ry * shrink * np.sin(ph)
            pts[t, 2] = 1.0 + (lo[2] - 1.0) * f
    elif mode == "random":
        steps = rng.uniform(-0.03, 0.03, size=(n, 3))
        steps[:, 2] -= 0.01
        pts = np.cumsum(steps, axis=0) + np.asarray(tool_start)
        pts = np.clip(pts, 0.02, 0.98)
    else:  # raster
        ncols = nrows = max(3, int(round(np.sqrt(n / 2.0))))
        xs = np.linspace(lo[0], hi[0], ncols)
        ys = np.linspace(lo[1], hi[1], nrows)
        z_top, z_bot = 0.95, lo[2]
        seq = []
        for j in range(nrows):
            row = xs if j % 2 == 0 else xs[::-1]
            for x in row:
                seq.append((x, ys[j]))
        reps = int(np.ceil(n / len(seq)))
        seq = (seq * reps)[:n]
        for t, (x, y) in enumerate(seq):
            f = t / max(1, n - 1)
            pts[t] = (x, y, z_top + (z_bot - z_top) * f)
    pts[0] = tool_start
    return pts.astype(np.float32)


def fit_control_points(B, X_ref):
    """Least-squares fit P so that B @ P ~= X_ref. Returns (n_ctrl, 3)."""
    P, *_ = np.linalg.lstsq(B, X_ref, rcond=None)
    return P.astype(np.float32)
