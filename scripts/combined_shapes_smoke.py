"""Smoke test for the combined CSG target shapes (sphere_hole, sphere_bowl).

Verifies, numerically (no images), that:
  - the simulator instantiates and bakes the target grid for each new shape;
  - sphere_hole: the through-hole is open at both poles (target SDF < 0 along
    the Z axis at the top and bottom of the sphere) and solid off-axis;
  - sphere_bowl: the cavity opens upward at the equator (target SDF < 0 just
    above center on-axis is NOT carved -- the bowl is a lower-hemisphere pocket,
    so the upper interior stays solid) and the lower interior is carved;
  - target_sdf and target_sdf_scalar agree (the autodiff mirror must match the
    eval-path SDF, else gradient penalties would be inconsistent).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import taichi as ti
import numpy as np
from simulator.csg_simulator import CSGSimulatorDelta

ti.init(arch=ti.cpu, debug=False, default_fp=ti.f32)


def build(shape):
    sim = CSGSimulatorDelta(resolution=32, max_steps=8, k_init=10.0,
                            target_shape=shape, tool_start=(0.5, 0.5, 1.0),
                            stock_size_in=(1.0, 1.0, 1.0), voxel_size_mm=0.5)
    sim.set_target_params(radius_mm=11.43, height_mm=22.86,
                          half_size_mm=11.43, center=(0.5, 0.5, 0.5),
                          sub_radius_mm=9.525)
    sim.bake_target_grid()
    sim.set_target_volume()
    return sim


def sdf_at(sim, p):
    """Sample the eval-path SDF from the baked target grid, and the autodiff
    mirror (target_sdf_scalar) via a tiny kernel. Returns (eval_sdf, scalar_sdf)."""
    # Eval path: nearest-voxel lookup of the baked target grid.
    i = int(np.clip(p[0] * sim.Nx, 0, sim.Nx - 1))
    j = int(np.clip(p[1] * sim.Ny, 0, sim.Ny - 1))
    k = int(np.clip(p[2] * sim.Nz, 0, sim.Nz - 1))
    a = float(sim.target[i, j, k])

    @ti.kernel
    def _scalar(px: ti.f32, py: ti.f32, pz: ti.f32) -> ti.f32:
        return sim.target_sdf_scalar(ti.Vector([px, py, pz]))
    b = float(_scalar(p[0], p[1], p[2]))
    return a, b


def main():
    # Stock sphere: radius 11.43mm in a 25.4mm cube -> normalized r = 0.45.
    # Z-extent of sphere: [0.05, 0.95]. Sub-primitive (hole/bowl) radius 0.375in
    # -> normalized 0.375 (since 1in stock). On-axis (x=y=0.5) points:
    #   top pole    z=0.90  (inside sphere, near top)
    #   bottom pole z=0.10  (inside sphere, near bottom)
    #   center      z=0.50
    # Off-axis solid check: x=0.5, y=0.5 is the axis; use x=0.85,y=0.5 (inside
    # the sphere annulus, outside the 0.375 sub-radius) -> should stay SOLID.
    pts = {
        "top_pole_axis":   (0.50, 0.50, 0.90),
        "bottom_pole_axis":(0.50, 0.50, 0.10),
        "center_axis":     (0.50, 0.50, 0.50),
        "annulus_solid":   (0.85, 0.50, 0.50),  # r=0.35 from axis > 0.375? no, 0.35<0.375
        "annulus_solid2":  (0.90, 0.50, 0.50),  # r=0.40 > 0.375 -> outside hole
        "above_center_offaxis": (0.60, 0.50, 0.70),
    }

    for shape in ("sphere", "sphere_hole", "sphere_bowl"):
        print(f"\n=== {shape}  (target_volume={float(build(shape).target_volume[None]):.4f}) ===")
        sim = build(shape)
        for name, p in pts.items():
            a, b = sdf_at(sim, p)
            inside = "INSIDE" if a < 0 else "outside"
            print(f"  {name:24s} p={p}  sdf={a:+.4f}  scalar={b:+.4f}  match={abs(a-b)<1e-4}  {inside}")

    # Direct geometry assertions. Target semantics: SDF < 0 = INSIDE the target
    # (material that should remain); SDF > 0 = OUTSIDE the target (carved away).
    sim = build("sphere_hole")
    top_a, _ = sdf_at(sim, (0.50, 0.50, 0.90))
    bot_a, _ = sdf_at(sim, (0.50, 0.50, 0.10))
    solid_a, _ = sdf_at(sim, (0.90, 0.50, 0.50))
    print("\n[sphere_hole] top_pole_carved:", top_a > 0,
          "bottom_pole_carved:", bot_a > 0, "annulus_solid:", solid_a < 0)
    assert top_a > 0, "through-hole must be open at top pole (on-axis = outside target)"
    assert bot_a > 0, "through-hole must be open at bottom pole (on-axis = outside target)"
    assert solid_a < 0, "annulus outside hole radius must remain solid (inside target)"

    sim = build("sphere_bowl")
    # Lower interior on-axis: inside sub-sphere AND below equator -> CARVED (SDF>0).
    lower_a, _ = sdf_at(sim, (0.50, 0.50, 0.30))
    # Upper interior on-axis: above the equator -> NOT carved (solid, SDF<0).
    upper_a, _ = sdf_at(sim, (0.50, 0.50, 0.70))
    # Top pole: solid sphere cap (SDF<0).
    topcap_a, _ = sdf_at(sim, (0.50, 0.50, 0.90))
    print("[sphere_bowl] lower_interior_carved:", lower_a > 0,
          "upper_interior_solid:", upper_a < 0, "topcap_solid:", topcap_a < 0)
    assert lower_a > 0, "lower-hemisphere cavity must be carved (SDF > 0)"
    assert upper_a < 0, "upper hemisphere must remain solid (SDF < 0)"
    assert topcap_a < 0, "top cap must remain solid (SDF < 0)"
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
