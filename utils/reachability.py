"""Exact 3-axis vertical-tool reachability over a voxel target SDF.

A flat-end cylindrical tool (radius r, height >= stock top) with base at
(x0, y0, z) occupies the disc of radius r for all z' >= z. It is LEGAL there
iff no part voxel lies inside that cylinder; a waste voxel (i, j, k) is
REMOVABLE iff some legal tool base position at z = k covers it. Legality is
monotone in z (raising the base only shrinks the cylinder), so removability
at the voxel's own z is the binding test.

Per z-slice, with U(x, y, z) = "part occupies (x, y) at some z' >= z":

    legal(x0, y0, z)   = disc_max_r[U(., ., z)](x0, y0) == 0
    removable(i, j, z) = disc_max_r[legal(., ., z)](i, j) == 1

i.e. two disc dilations per slice. Shape-agnostic: reads only the SDF grid
and the tool radius in voxels. This is exact for the simulator's sharp
cylindrical tool with vertical (3-axis) access from above.
"""

import numpy as np
from scipy.ndimage import binary_dilation


def _disc(r_vox):
    """Boolean disc footprint of radius r_vox (in voxels, center included)."""
    n = int(np.floor(r_vox))
    xx, yy = np.mgrid[-n:n + 1, -n:n + 1]
    return (xx * xx + yy * yy) <= r_vox * r_vox


def compute_reachable_mask(target_sdf, r_tool_vox):
    """Boolean (Nx, Ny, Nz) mask: True where a waste voxel is removable by a
    vertical tool from above without intersecting the part.

    ``target_sdf``: voxel-space SDF grid (<= 0 inside the part), z index up.
    ``r_tool_vox``: tool radius in voxels.
    Part voxels are False (they are not waste; removing them is a gouge).
    """
    part = target_sdf <= 0.0
    nx, ny, nz = part.shape
    disc = _disc(r_tool_vox)
    reach = np.zeros_like(part)

    # U(., ., z): part occupancy anywhere at z' >= z (suffix-OR down the z axis).
    part_above = np.zeros_like(part)
    acc = np.zeros((nx, ny), dtype=bool)
    for k in range(nz - 1, -1, -1):
        acc |= part[:, :, k]
        part_above[:, :, k] = acc

    for k in range(nz):
        blocked = binary_dilation(part_above[:, :, k], structure=disc)
        legal = ~blocked
        reach[:, :, k] = binary_dilation(legal, structure=disc)

    reach &= ~part
    return reach


def dice_ceiling(target_sdf, r_tool_vox):
    """Best achievable dice for a vertical 3-axis tool: every reachable waste
    voxel removed, every unreachable one left. Returns (ceiling, n_unreachable,
    n_waste, n_part)."""
    part = target_sdf <= 0.0
    reach = compute_reachable_mask(target_sdf, r_tool_vox)
    waste = ~part
    unreachable = int((waste & ~reach).sum())
    n_part = int(part.sum())
    ceiling = 2.0 * n_part / (2.0 * n_part + unreachable)
    return ceiling, unreachable, int(waste.sum()), n_part


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Reachability ceiling for an NPZ target")
    ap.add_argument("npz", help="target grid from utils/step_to_sdf.py")
    ap.add_argument("--tool-radius-mm", type=float, default=3.175)
    args = ap.parse_args()

    d = np.load(args.npz)
    v = float(d["voxel_size_mm"])
    sdf = d["sdf"] / v  # mm -> voxels
    ceiling, unreach, waste, part = dice_ceiling(sdf, args.tool_radius_mm / v)
    print(f"{args.npz}: dice ceiling {ceiling:.4f} "
          f"(part {part}, waste {waste}, unreachable {unreach})")
