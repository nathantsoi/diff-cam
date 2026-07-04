"""Box face-offset sweep test.

The box target fills [0.05, 0.95]^3 (half-size 0.45). Its waste is the 6 face
slivers (x/y/z in [0,0.05] or [0.95,1]) + 12 edge slivers + 8 corners. The face
slivers are the bulk (0.243 vol). A cylindrical tool (r=0.125) is too FAT to
fit in the 0.05 sliver from inside, but placed JUST OUTSIDE the stock face
(center at x=-0.08 etc.) it removes the sliver [0, 0.045] without touching the
box (starts at 0.05). The tall tool (h~=1) spanning the stock height clears the
full sliver in one pass. Tests whether this pushes box hard dice above the
do-nothing floor (0.844) toward ~0.95.
"""
import math
import numpy as np
import tyro
from dataclasses import dataclass


def box_face_traj(n, r_sp, r_tool, target_height_mm, stock_mm):
    """6 face-offset sweeps. Tool center just outside each face."""
    tool_start = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    # box face at 0.5 - r_sp (=0.05) and 0.5 + r_sp (=0.95)
    face_in = 0.5 - r_sp           # 0.05
    face_out = 0.5 + r_sp          # 0.95
    margin = 0.005
    # tool center just outside the face so its inner edge = face_in - margin
    x_off = face_in - r_tool - margin      # -0.075 (outside stock, removes [0, 0.045])
    # sweep extent (cover the face fully; tool radius covers the 0.05 sliver)
    s_lo, s_hi = 0.0 + r_tool, 1.0 - r_tool
    z_top_face = face_out + margin         # 0.955
    z_bot_face = face_in - margin          # 0.045
    positions = []
    pts_per_face = max(4, n // 6)

    def add(p):
        positions.append(p)

    # x-faces: tool at x=x_off (or 1-x_off), base=0 (spans full z), sweep y
    for k in range(pts_per_face):
        frac = k / max(1, pts_per_face - 1)
        y = s_lo + (s_hi - s_lo) * frac
        add([x_off, float(y), 0.0])
    for k in range(pts_per_face):
        frac = k / max(1, pts_per_face - 1)
        y = s_hi - (s_hi - s_lo) * frac
        add([1.0 - x_off, float(y), 0.0])
    # y-faces: tool at y=x_off, sweep x, base=0
    for k in range(pts_per_face):
        frac = k / max(1, pts_per_face - 1)
        x = s_lo + (s_hi - s_lo) * frac
        add([float(x), x_off, 0.0])
    for k in range(pts_per_face):
        frac = k / max(1, pts_per_face - 1)
        x = s_hi - (s_hi - s_lo) * frac
        add([float(x), 1.0 - x_off, 0.0])
    # z-faces: base=z_bot_face-1+? -> tool top at z_bot_face. base = z_bot_face - h
    h = 1.0
    base_bot = z_bot_face - h            # tool spans [base_bot, z_bot_face], removes z<z_bot_face
    base_top = z_top_face                # tool spans [z_top_face, ...], removes z>z_top_face
    nz = max(2, pts_per_face // 4)
    xs = np.linspace(s_lo, s_hi, max(3, nz))
    ys = np.linspace(s_lo, s_hi, max(3, nz))
    # bottom z-face boustrophedon
    for j, y in enumerate(ys):
        row_xs = xs if j % 2 == 0 else xs[::-1]
        for x in row_xs:
            add([float(x), float(y), base_bot])
    # top z-face boustrophedon
    for j, y in enumerate(ys):
        row_xs = xs if j % 2 == 0 else xs[::-1]
        for x in row_xs:
            add([float(x), float(y), base_top])

    positions = np.array(positions[:n], dtype=np.float32)
    if len(positions) < n:
        positions = np.vstack([positions, np.tile(positions[-1:], (n - len(positions), 1))])
    return np.vstack([tool_start[None, :], positions])


@dataclass
class Args:
    target_radius_mm: float = 11.43
    max_steps: int = 256


def main():
    args = tyro.cli(Args)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval.eval_csg import carve_trajectory_metrics
    stock_mm = 25.4
    r_sp = args.target_radius_mm / stock_mm
    r_tool = 3.175 / stock_mm
    n = args.max_steps - 1
    pos = box_face_traj(n, r_sp, r_tool, 22.86, stock_mm)
    m, stock, target = carve_trajectory_metrics(pos, resolution=32, target_shape="box", voxel_size_mm=0.5)
    so = int((stock < 0).sum()); to = int((target < 0).sum())
    floor = 2.0 * to / max(1, so + to)
    print(f"=== box face-offset (T={args.max_steps}) ===")
    print(f"dice: {m['dice']:.4f}  asd: {m['asd']:.2f}  hd95: {m['hd95']:.2f}")
    print(f"stock_occ={so} target_occ={to} stationary_floor={floor:.4f}")


if __name__ == "__main__":
    main()
