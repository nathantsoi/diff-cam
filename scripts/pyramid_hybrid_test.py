"""Pyramid hybrid init test: measure the structural ceiling.

The pyramid is centered (base z=0.275, apex z=0.725). The tall tool (h~=stock)
extends UPWARD from its base, so waste BELOW the pyramid (z<0.275) is
structurally unreachable -- carving it would gouge through the pyramid. This
tests how high the hard dice can climb with a clean init that clears the
REACHABLE waste (above-disk z>0.725 via boustrophedon + beside-annulus via
square orbit), leaving the below-block.

Usage:
    uv run python scripts/pyramid_hybrid_test.py --mode above
    uv run python scripts/pyramid_hybrid_test.py --mode hybrid
    uv run python scripts/pyramid_hybrid_test.py --mode below
"""
import math
import numpy as np
import tyro
from dataclasses import dataclass


def pyramid_half(z, base_z, h, r_sp):
    if z < base_z or z > base_z + h:
        return 0.0
    return r_sp * (1.0 - (z - base_z) / h)


def make_traj(mode, n, r_sp, r_tool, target_height_mm, stock_mm, max_steps):
    tool_start = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    h = target_height_mm / stock_mm
    base_z = 0.5 - 0.5 * h
    apex = base_z + h
    margin = 0.005
    r_outer = 0.5 + r_tool
    positions = []

    if mode == "above":
        # Boustrophedon over full cross-section at z in [apex, 0.95] (clear the
        # above-disk). Tool base >= apex so it never reaches the pyramid.
        nrows = 9
        xs = np.linspace(0.12, 0.88, 9)
        ys = np.linspace(0.12, 0.88, nrows)
        z_levels = np.linspace(0.95, apex, 8)
        for z in z_levels:
            for j, y in enumerate(ys):
                row_xs = xs if j % 2 == 0 else xs[::-1]
                for x in row_xs:
                    positions.append([float(x), float(y), float(z)])
                    if len(positions) >= n:
                        break
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
    elif mode == "below":
        # Try to clear below-disk (z<base_z) -- should GOUGE the pyramid
        # (demonstrates the structural limit). Boustrophedon at z<base_z.
        xs = np.linspace(0.12, 0.88, 9)
        ys = np.linspace(0.12, 0.88, 9)
        z_levels = np.linspace(base_z - 0.02, 0.05, 6)
        for z in z_levels:
            for j, y in enumerate(ys):
                row_xs = xs if j % 2 == 0 else xs[::-1]
                for x in row_xs:
                    positions.append([float(x), float(y), float(z)])
                    if len(positions) >= n:
                        break
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
    elif mode == "hybrid":
        # 4-phase gouge-free path. Tool extends UP from base (spans [base,base+h]).
        # To carve a z-slice without gouging what is above it, base is set so the
        # tool's TOP sits at the slice's upper bound. Phases ordered by descending
        # base; transitions that would transit the pyramid at base~0 are routed
        # at a SAFE radius (outside the pyramid's widest half-size + r_tool).
        #   1. above-disk (z>apex): base in [apex, 0.95], full boustrophedon.
        #   2. beside-annulus (z in [base_z, apex]): base descends apex->base_z,
        #      square orbit at pyramid_half(base)+r_tool.
        #   3. safe-radius descent: base descends base_z -> -0.75 at r=r_safe_max
        #      (outside pyramid), carving the lower annulus without gouging.
        #   4. below-disk (z<base_z): base in [-0.95, -0.75], full boustrophedon
        #      (tool top below pyramid base -> no gouge).
        n_above = int(n * 0.42)
        n_below = 0  # below-phase gouges (holder/height interaction); descent clears the below-annulus
        n_descent = max(8, int(n * 0.10))
        n_beside = n - n_above - n_below - n_descent
        xs = np.linspace(0.12, 0.88, 7)
        ys = np.linspace(0.12, 0.88, 7)
        r_safe_max = r_sp + r_tool + margin  # widest pyramid half-size + r_tool

        def boustrophedon(z_levels, cap):
            out = []
            for z in z_levels:
                for j, y in enumerate(ys):
                    row_xs = xs if j % 2 == 0 else xs[::-1]
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
        # 3. safe-radius descent: base_z -> -0.75 at r=r_safe_max (spiral, safe)
        for t in range(n_descent):
            frac = t / max(1, n_descent - 1)
            zb = base_z + (-0.75 - base_z) * frac
            phase = 2.0 * math.pi * 3.0 * frac
            positions.append([0.5 + r_safe_max * math.cos(phase),
                              0.5 + r_safe_max * math.sin(phase), float(zb)])
        # 4. below: fixed-base spiral at base=-0.80 (tool top at 0.18 < pyramid
        #   base 0.275 -> carves only the below-disk, never the pyramid). Spiral
        #   from center out covers the disk densely with safe transits (all at
        #   base=-0.80, carving z in [0, 0.18]).
        zb_below = -0.80
        for t in range(n_below):
            frac = t / max(1, n_below - 1)
            r = r_outer * frac
            phase = 2.0 * math.pi * 12.0 * frac
            positions.append([0.5 + r * math.cos(phase), 0.5 + r * math.sin(phase), float(zb_below)])
        while len(positions) < n_above + n_beside + n_descent + n_below:
            positions.append(positions[-1] if positions else [0.5, 0.5, -0.8])

    positions = np.array(positions[:n], dtype=np.float32)
    if len(positions) < n:
        positions = np.vstack([positions, np.tile(positions[-1:], (n - len(positions), 1))])
    return np.vstack([tool_start[None, :], positions])


@dataclass
class Args:
    mode: str = "hybrid"
    target_radius_mm: float = 11.43
    target_height_mm: float = 22.86
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
    pos = make_traj(args.mode, n, r_sp, r_tool, args.target_height_mm, stock_mm, args.max_steps)
    m, stock, target = carve_trajectory_metrics(pos, resolution=32, target_shape="pyramid", voxel_size_mm=0.5)
    so = int((stock < 0).sum()); to = int((target < 0).sum())
    floor = 2.0 * to / max(1, so + to)
    print(f"=== pyramid {args.mode} (T={args.max_steps}) ===")
    print(f"dice: {m['dice']:.4f}  asd: {m['asd']:.2f}  hd95: {m['hd95']:.2f}")
    print(f"stock_occ={so} target_occ={to} stationary_floor={floor:.4f}")


if __name__ == "__main__":
    main()
