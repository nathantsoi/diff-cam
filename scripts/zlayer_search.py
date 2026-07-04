"""Fast zlayer geometry search: score hard dice of parameterized zlayer inits.

Mirrors train_csg.py's `zlayer` init but exposes revs / osc_cycles / margin /
z-range as parameters, and scores via eval.eval_csg.carve_trajectory_metrics
(UNCLIPPED hard carve -- matches the trainer's clipped eval at feed_ipm>=60,
where the zlayer per-step is no longer speed-capped). Lets us search zlayer
geometry in seconds per config instead of 25-min training runs.

Answers: is 0.779 the zlayer-geometry ceiling for sphere, or can a denser
sweep (more revs / more oscillation cycles / tighter margin) beat it?

Usage:
    uv run python scripts/zlayer_search.py --target-shape sphere
    uv run python scripts/zlayer_search.py --target-shape sphere --revs 24 --osc 6
"""
import math
import numpy as np
import tyro
from dataclasses import dataclass, field


def zlayer_positions(n, target_shape, target_radius_mm, target_height_mm,
                     revs=12.0, osc=3.0, margin=0.03, z_top=0.95, z_bot=-0.95,
                     tool_radius_mm=3.175, stock_size_in=1.0):
    """Replicate train_csg.py's zlayer init (T-1 points; prepend tool_start).

    Returns (n+1, 3) positions including the canonical tool_start (0.5,0.5,1.0).
    """
    tool_start = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    stock_mm = stock_size_in * 25.4
    r_tool = tool_radius_mm / stock_mm
    r_outer = 0.5 + r_tool
    positions = np.zeros((n, 3), dtype=np.float32)

    if target_shape == "sphere":
        r_sp = target_radius_mm / stock_mm
        for t in range(n):
            frac = t / max(1, n - 1)
            zb = z_top + (z_bot - z_top) * frac
            zhi = zb + 1.0
            if zb > 0.5:
                z_eq = zb
            elif zhi < 0.5:
                z_eq = zhi
            else:
                z_eq = 0.5
            rs = math.sqrt(max(0.0, r_sp * r_sp - (z_eq - 0.5) * (z_eq - 0.5)))
            r_safe = rs + r_tool + margin
            r_orbit = r_safe + (r_outer - r_safe) * (0.5 + 0.5 * math.sin(2.0 * math.pi * osc * frac))
            phase = 2.0 * math.pi * revs * frac
            positions[t, 0] = 0.5 + r_orbit * math.cos(phase)
            positions[t, 1] = 0.5 + r_orbit * math.sin(phase)
            positions[t, 2] = zb
    elif target_shape == "cylinder":
        # Cylinder axis along z, radius z-invariant. Safe radius is constant:
        # r_cyl + r_tool + margin. Sweep the annulus out to the cube wall at
        # every z (a real z-level finishing pattern for a vertical cylinder).
        r_cyl = target_radius_mm / stock_mm
        r_safe = r_cyl + r_tool + margin
        for t in range(n):
            frac = t / max(1, n - 1)
            zb = z_top + (z_bot - z_top) * frac
            r_orbit = r_safe + (r_outer - r_safe) * (0.5 + 0.5 * math.sin(2.0 * math.pi * osc * frac))
            phase = 2.0 * math.pi * revs * frac
            positions[t, 0] = 0.5 + r_orbit * math.cos(phase)
            positions[t, 1] = 0.5 + r_orbit * math.sin(phase)
            positions[t, 2] = zb
    else:
        # box / pyramid: fall back to a full-annulus sweep (no shape-specific
        # inner radius -- r_safe = r_tool + margin, sweeps to the wall). Coarse
        # but tests whether the annulus pattern generalizes.
        r_safe = r_tool + margin
        for t in range(n):
            frac = t / max(1, n - 1)
            zb = z_top + (z_bot - z_top) * frac
            r_orbit = r_safe + (r_outer - r_safe) * (0.5 + 0.5 * math.sin(2.0 * math.pi * osc * frac))
            phase = 2.0 * math.pi * revs * frac
            positions[t, 0] = 0.5 + r_orbit * math.cos(phase)
            positions[t, 1] = 0.5 + r_orbit * math.sin(phase)
            positions[t, 2] = zb

    full = np.vstack([tool_start[None, :], positions])
    return full


@dataclass
class Args:
    target_shape: str = "sphere"
    target_radius_mm: float = 11.43
    target_height_mm: float = 22.86
    max_steps: int = 128
    revs: float = 12.0
    osc: float = 3.0
    margin: float = 0.03
    sweep: bool = False


def score(pos, args):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval.eval_csg import carve_trajectory_metrics
    m, stock, target = carve_trajectory_metrics(
        pos, resolution=32, target_shape=args.target_shape, voxel_size_mm=0.5)
    stock_occ = int((stock < 0).sum())
    target_occ = int((target < 0).sum())
    floor = 2.0 * target_occ / max(1, stock_occ + target_occ)
    return m, floor, stock_occ, target_occ


def main():
    args = tyro.cli(Args)
    n = args.max_steps - 1
    print(f"=== zlayer search ({args.target_shape}, T={args.max_steps}) ===")

    if args.sweep:
        configs = []
        for revs in [15.0, 18.0, 21.0]:
            for osc in [8.0, 9.0, 10.0, 12.0]:
                for margin in [0.005, 0.01, 0.015]:
                    configs.append((revs, osc, margin))
        best = (None, -1.0)
        for revs, osc, margin in configs:
            pos = zlayer_positions(n, args.target_shape, args.target_radius_mm,
                                   args.target_height_mm, revs=revs, osc=osc, margin=margin)
            m, floor, so, to = score(pos, args)
            d = m["dice"]
            tag = "  *BEST*" if d > best[1] else ""
            print(f"revs={revs:5.1f} osc={osc:3.1f} margin={margin:.3f}  dice={d:.4f}  (floor {floor:.4f}){tag}")
            if d > best[1]:
                best = ((revs, osc, margin), d)
        print(f"\nBEST: revs={best[0][0]} osc={best[0][1]} margin={best[0][2]}  dice={best[1]:.4f}")
    else:
        pos = zlayer_positions(n, args.target_shape, args.target_radius_mm,
                               args.target_height_mm, revs=args.revs, osc=args.osc, margin=args.margin)
        m, floor, so, to = score(pos, args)
        print(f"revs={args.revs} osc={args.osc} margin={args.margin}")
        for k in ("dice", "asd", "hd95"):
            print(f"{k + ':':10s} {m[k]:.6f}")
        print(f"floor:     {floor:.6f}  (stock={so}, target={to})")


if __name__ == "__main__":
    main()
