"""Coverage diagnostic: score a trajectory's HARD dice with zero optimization.

Answers: does a SYSTEMATIC coverage path (raster_fine init) carve the part in
hard space, where the soft-optimized trajectory scores the stationary floor
(~0.553 sphere)? If yes, coverage is confirmed as the real lever for hard dice
and soft optimization is what collapses it.

Reuses eval.eval_csg.carve_trajectory_metrics (the exact hard-carve eval the
trainer reports) so numbers are directly comparable.

Usage:
    uv run python scripts/coverage_diagnostic.py --target-shape sphere
    uv run python scripts/coverage_diagnostic.py --target-shape cylinder \
        --target-height-mm 22.86
"""
import numpy as np
import tyro
from dataclasses import dataclass


def raster_fine_positions(n, target_shape="sphere"):
    """Replicate train_csg.py's raster_fine init (T-1 points; prepend tool_start).

    Clipping-aware fine boustrophedon: per-step <= feed cap, snakes across XY
    at constant step while Z descends. Returns (n+1, 3) positions including
    the canonical tool_start (0.5, 0.5, 1.0) prepended.
    """
    tool_start = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    ncols, nrows = 11, 11
    xs = np.linspace(0.20, 0.80, ncols)
    ys = np.linspace(0.20, 0.80, nrows)
    z_top, z_bot = 0.90, 0.10
    positions = []
    idx = 0
    for j in range(nrows):
        row_xs = xs if j % 2 == 0 else xs[::-1]
        for x in row_xs:
            frac = idx / max(1, n - 1)
            z = z_top + (z_bot - z_top) * frac
            positions.append([float(x), float(ys[j]), float(z)])
            idx += 1
            if idx >= n:
                break
        if idx >= n:
            break
    positions = np.array(positions[:n], dtype=np.float32)
    if len(positions) < n:
        positions = np.vstack([positions, np.tile(positions[-1:], (n - len(positions), 1))])
    full = np.vstack([tool_start[None, :], positions])
    return full


@dataclass
class Args:
    target_shape: str = "sphere"
    target_radius_mm: float = 11.43
    target_height_mm: float = 22.86
    max_steps: int = 128


def main():
    args = tyro.cli(Args)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval.eval_csg import carve_trajectory_metrics
    n = args.max_steps - 1
    pos = raster_fine_positions(n, args.target_shape)
    m, stock, target = carve_trajectory_metrics(
        pos, resolution=32, target_shape=args.target_shape, voxel_size_mm=0.5)
    print(f"--- coverage diagnostic ({args.target_shape}, raster_fine init, 0 optimization) ---")
    for k in ("dice", "asd", "hd95"):
        print(f"{k + ':':10s} {m[k]:.6f}")
    # stationary-tool floor for reference: 2|target|/(|stock|+|target|)
    stock_occ = (stock < 0).sum()
    target_occ = (target < 0).sum()
    floor = 2.0 * target_occ / max(1, stock_occ + target_occ)
    print(f"stationary floor (2|tgt|/(|stk|+|tgt|)): {floor:.6f}")
    print(f"(stock voxels={stock_occ}, target voxels={target_occ})")


if __name__ == "__main__":
    main()
