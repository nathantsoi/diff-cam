"""Standalone measurement of geometry scalars (ang_cv, z_cv, annularity) for all
target shapes. Validates that the proposed ANNULARITY scalar separates the hole
(sphere with a central column removed -> annular cross-sections) from the solid
shapes (sphere/cyl/box/pyr/bowl), BEFORE wiring it into train_csg.py.

annul = 1 - mean_z( z_area[z] / (pi * r_bound_mean[z]^2) )
  -- ~0 for SOLID cross-sections (z_area == pi*r^2 of the outer boundary)
  -- >0 for ANNULAR cross-sections (a central void removes area the outer
     boundary implies); r_bound is the MAX radius per theta = outer boundary only,
     so a through-hole reads as area-deficient relative to its outer circle.
"""
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, debug=False, default_fp=ti.f32)
from simulator.csg_simulator import CSGSimulatorDelta
from algorithms.train_csg import _contour_geometry_scalars

# (shape, radius_mm, label)
SHAPES = [
    ("cylinder", 11.43),
    ("sphere", 11.43),
    ("box", 9.0),
    ("pyramid", 9.0),
    ("sphere_bowl", 11.43),
    ("sphere_hole", 11.43),
]


def build_grid(shape, radius_mm, resolution=32):
    sim = CSGSimulatorDelta(resolution=resolution, max_steps=128, k_init=20.0,
                            init_taichi=False,
                            target_shape=shape, tool_start=(0.5, 0.5, 1.0),
                            stock_size_in=(1.0, 1.0, 1.0),
                            voxel_size_mm=0.5,
                            work_volume_in=(16.0, 12.0, 10.0),
                            stock_origin_in=(0.0, 0.0, 0.0), dt=0.45,
                            rapid_ipm=200.0, feed_ipm=40.0,
                            safe_distance_in=0.05,
                            enforce_speed_limits=True)
    sim.set_target_params(radius_mm=radius_mm, height_mm=radius_mm,
                          half_size_mm=radius_mm, center=(0.5, 0.5, 0.5),
                          sub_radius_mm=9.525)
    sim.bake_target_grid()
    return sim.target.to_numpy()


def annularity(grid):
    """annul = 1 - mean_z( z_area[z] / (pi * r_vox[z]^2) ) over slices with target,
    where r_vox = r_bound_mean * Nx puts the normalized outer-boundary radius into
    voxel units so it is comparable to z_area (an inside-voxel COUNT = voxel^2 area).
    ~0 for SOLID cross-sections (z_area == pi*r^2); >0 for ANNULAR ones (a central
    void removes area the outer boundary implies)."""
    ang_cv, z_cv, r_bound, z_area = _contour_geometry_scalars(grid)
    Nx = grid.shape[0]
    Nz, Nth = r_bound.shape
    ratios = []
    for iz in range(Nz):
        if z_area[iz] > 0.0:
            b = r_bound[iz]
            b = b[b > 0.0]
            if len(b) > 0:
                r_vox = float(b.mean()) * Nx
                implied = np.pi * r_vox * r_vox
                if implied > 1e-9:
                    ratios.append(float(z_area[iz]) / implied)
    mean_ratio = float(np.mean(ratios)) if ratios else 1.0
    return 1.0 - mean_ratio


print(f"{'shape':14s} {'ang_cv':>7s} {'z_cv':>7s} {'annul':>7s}   init-winner")
print("-" * 60)
for shape, rad in SHAPES:
    for res in (32, 64):
        g = build_grid(shape, rad, resolution=res)
        ang_cv, z_cv, _, _ = _contour_geometry_scalars(g)
        ann = annularity(g)
        concavity = (ang_cv < 0.06) and (z_cv > 0.50)
        init = "cavity" if concavity else "contour"
        print(f"{shape:14s} {ang_cv:7.3f} {z_cv:7.3f} {ann:7.3f}   {init}   (res={res})")
    print()
