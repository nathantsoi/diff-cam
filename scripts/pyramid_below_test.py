"""Pyramid 4-phase test: recover the below-disk via a FIXED-base boustrophedon.

The pyramid: base z=0.275, apex z=0.725 (h=0.45, half-base 0.45). Waste slab
below the base (z in [0, 0.275]) is the bulk of the unreachable volume. The tool
spans [z_base, z_base+h] (h~=1.0 = full stock). To clear the below-slab WITHOUT
gouging the pyramid, set z_base = -0.73 so the tool top = 0.27 < pyramid base
0.275. A boustrophedon over the full XY at this FIXED base carves the whole
below-slab in one pass. forward_hard uses tool_sdf_sharp only (no holder), so the
wide holder above the tool does NOT carve in the hard eval.

The earlier "below" mode gouged because it swept z_base in [0.05, 0.255] (tool
top reached 1.05+, carving the pyramid). Fixed-low-base is the fix.

Usage:
    uv run python scripts/pyramid_below_test.py --mode full4
    uv run python scripts/pyramid_below_test.py --mode below_only
"""
import math
import numpy as np
import tyro
from dataclasses import dataclass


def pyramid_half(z, base_z, h, r_sp):
    if z < base_z or z > base_z + h:
        return 0.0
    return r_sp * (1.0 - (z - base_z) / h)


def make_traj(mode, n, r_sp, r_tool, target_height_mm, stock_mm,
              margin=0.005, split=(0.34, 0.30, 0.08), osc_above=8.0, revs_beside=20.0,
              descent_r=None):
    tool_start = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    h = target_height_mm / stock_mm
    base_z = 0.5 - 0.5 * h
    apex = base_z + h
    r_outer = 0.5 + r_tool
    positions = []

    if mode == "below_only":
        # Fixed-base boustrophedon: tool top = z_base + 1.0 = 0.27 < 0.275.
        z_base_below = base_z - 1.0 - 0.005   # -0.73: tool spans [-0.73, 0.27]
        xs = np.linspace(0.0 + r_tool, 1.0 - r_tool, 11)
        ys = np.linspace(0.0 + r_tool, 1.0 - r_tool, 11)
        for j, y in enumerate(ys):
            row_xs = xs if j % 2 == 0 else xs[::-1]
            for x in row_xs:
                positions.append([float(x), float(y), float(z_base_below)])
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
    elif mode == "full4":
        # 4-phase: above + beside + safe descent + below-fixed boustrophedon.
        f_above, f_below, f_descent = split
        n_above = int(n * f_above)
        n_below = int(n * f_below)
        n_descent = max(8, int(n * f_descent))
        n_beside = n - n_above - n_below - n_descent
        xs = np.linspace(0.12, 0.88, 7)
        ys = np.linspace(0.12, 0.88, 7)
        r_safe_max = r_sp + r_tool + margin

        def boustrophedon(z_levels, cap, xs_arr=None, ys_arr=None):
            xa = xs if xs_arr is None else xs_arr
            ya = ys if ys_arr is None else ys_arr
            out = []
            for z in z_levels:
                for j, y in enumerate(ya):
                    row_xs = xa if j % 2 == 0 else xa[::-1]
                    for x in row_xs:
                        out.append([float(x), float(y), float(z)])
                        if len(out) >= cap:
                            return out
            return out

        # 1. above
        above_z = np.linspace(0.95, apex + 0.02, 4)
        positions += boustrophedon(above_z, n_above)
        while len(positions) < n_above:
            positions.append(positions[-1] if positions else [0.5, 0.5, 0.9])
        # 2. beside: square orbit, base descends apex -> base_z
        for t in range(n_beside):
            frac = t / max(1, n_beside - 1)
            zb = apex + (base_z - apex) * frac
            hp = pyramid_half(zb, base_z, h, r_sp)
            s_safe = hp + r_tool + margin
            s_orbit = s_safe + (r_outer - s_safe) * (0.5 + 0.5 * math.sin(2.0 * math.pi * 8.0 * frac))
            phase = 2.0 * math.pi * 20.0 * frac
            cx, cy = math.cos(phase), math.sin(phase)
            m = max(abs(cx), abs(cy))
            positions.append([0.5 + s_orbit * cx / m, 0.5 + s_orbit * cy / m, float(zb)])
        # 3. safe-radius descent: base_z -> below-phase base. Circular orbit at
        # r_safe_max clears the lower annulus (net positive despite corner
        # proximity; the square alternative leaves too much waste).
        z_base_below = base_z - 1.0 - 0.005
        r_desc = r_safe_max if descent_r is None else descent_r
        for t in range(n_descent):
            frac = t / max(1, n_descent - 1)
            zb = base_z + (z_base_below - base_z) * frac
            phase = 2.0 * math.pi * 3.0 * frac
            positions.append([0.5 + r_desc * math.cos(phase),
                              0.5 + r_desc * math.sin(phase), float(zb)])
        # 4. below: FIXED-base boustrophedon (tool top = 0.27 < pyramid base 0.275)
        bx = np.linspace(0.0 + r_tool, 1.0 - r_tool, 9)
        by = np.linspace(0.0 + r_tool, 1.0 - r_tool, 9)
        positions += boustrophedon(np.array([z_base_below]), n_below, xs_arr=bx, ys_arr=by)
        while len(positions) < n_above + n_beside + n_descent + n_below:
            positions.append(positions[-1] if positions else [0.5, 0.5, z_base_below])

    positions = np.array(positions[:n], dtype=np.float32)
    if len(positions) < n:
        positions = np.vstack([positions, np.tile(positions[-1:], (n - len(positions), 1))])
    return np.vstack([tool_start[None, :], positions])


@dataclass
class Args:
    mode: str = "full4"
    target_radius_mm: float = 11.43
    target_height_mm: float = 22.86
    max_steps: int = 384


def main():
    args = tyro.cli(Args)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval.eval_csg import carve_trajectory_metrics
    stock_mm = 25.4
    r_sp = args.target_radius_mm / stock_mm
    r_tool = 3.175 / stock_mm
    n = args.max_steps - 1
    pos = make_traj(args.mode, n, r_sp, r_tool, args.target_height_mm, stock_mm)
    m, stock, target = carve_trajectory_metrics(pos, resolution=32, target_shape="pyramid", voxel_size_mm=0.5)
    so = int((stock < 0).sum()); to = int((target < 0).sum())
    floor = 2.0 * to / max(1, so + to)
    print(f"=== pyramid {args.mode} (T={args.max_steps}) ===")
    print(f"dice: {m['dice']:.4f}  asd: {m['asd']:.2f}  hd95: {m['hd95']:.2f}")
    print(f"stock_occ={so} target_occ={to} stationary_floor={floor:.4f}")


if __name__ == "__main__":
    main()
