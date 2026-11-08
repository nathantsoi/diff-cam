"""Minimal Adam optimization of delta actions; hard-carve checkpoint selection.

Run with ``uv run python -m algorithms.optimize --output runs/core-smoke``.
Input trajectories and output positions use normalized stock coordinates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import taichi as ti

from cam import MachineConfig, trajectory_to_gcode
from simulator.csg_metrics import dice_score, sdf_to_mask
from simulator.csg_simulator import CSGSimulatorDelta


def optimize(args):
    if args.steps < 2 or args.iters < 1 or min(args.k_start, args.k_end, args.lr, args.dt) <= 0:
        raise ValueError("steps >= 2, iters >= 1 and positive k/lr/dt required")
    if args.resolution < 4:
        raise ValueError("resolution must be at least 4")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    rng = np.random.default_rng(args.seed)
    if args.initial is not None:
        path = np.load(args.initial).astype(np.float32)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 3 or not np.isfinite(path).all():
            raise ValueError("initial must contain finite (N >= 3, 3) positions")
        start, deltas = path[0], np.diff(path, axis=0)
    else:
        # Small random moves are a wiring demo, not a CAM initialization.
        start = np.array([0.15, 0.15, 0.85], dtype=np.float32)
        deltas = rng.normal(0, 0.025, (args.steps, 3)).astype(np.float32)
    ti.init(arch=ti.cpu if args.cpu else ti.gpu, offline_cache=False)
    sim = CSGSimulatorDelta(resolution=args.resolution, max_steps=len(deltas),
                            stock_size_in=(1, 1, 1), target_shape=args.target,
                            tool_start=tuple(start), dt=args.dt, init_taichi=False)
    sim.tool_radius[None] = 3.175
    sim.tool_height[None] = 25.0
    sim.set_target_params(radius_mm=9.0, height_mm=18.0, half_size_mm=9.0)
    sim.bake_target_grid()
    target = sdf_to_mask(sim.target.to_numpy())
    initial = np.vstack([start, start + np.cumsum(deltas, axis=0)])
    m, v = np.zeros_like(deltas), np.zeros_like(deltas)
    best, history = None, []
    for iteration in range(args.iters + 1):
        k = args.k_start + (args.k_end - args.k_start) * iteration / args.iters
        sim.k[None] = k
        sim.tool_delta.from_numpy(deltas)
        with ti.ad.Tape(loss=sim.loss):
            sim.forward(len(deltas) + 1)
        loss = float(sim.loss[None])
        grad = sim.tool_delta.grad.to_numpy()
        if not np.isfinite(loss) or not np.isfinite(grad).all():
            raise FloatingPointError(f"non-finite loss/gradient at iteration {iteration}")
        soft_dice = float(dice_score(sdf_to_mask(sim.stock.to_numpy()[-1]), target))
        sim.forward_hard(len(deltas) + 1)
        mask = sdf_to_mask(sim.stock.to_numpy()[-1])
        hard_dice = float(dice_score(mask, target))
        union = np.logical_or(mask, target).sum()
        metrics = dict(iteration=iteration, k=k, loss=loss, soft_dice=soft_dice,
                       hard_dice=hard_dice, iou=float(np.logical_and(mask, target).sum()/max(1, union)),
                       gouge_stock_fraction=float(np.mean(target & ~mask)),
                       residual_stock_fraction=float(np.mean(~target & mask)),
                       gradient_norm=float(np.linalg.norm(grad)))
        history.append(metrics)
        if best is None or hard_dice > best[0]['hard_dice']:
            best = (metrics.copy(), deltas.copy(), sim.tool_pos.to_numpy().copy())
        if iteration == args.iters:
            break
        grad *= min(1.0, 0.5 / max(float(np.linalg.norm(grad)), 1e-12))
        m = 0.9*m + 0.1*grad
        v = 0.999*v + 0.001*grad*grad
        t = iteration + 1
        deltas -= args.lr * (m/(1-0.9**t)) / (np.sqrt(v/(1-0.999**t)) + 1e-8)
    args.output.mkdir(parents=True)
    np.save(args.output/'initial_trajectory.npy', initial)
    np.save(args.output/'trajectory.npy', best[2])
    np.save(args.output/'trajectory_deltas.npy', best[1])
    (args.output/'metrics.json').write_text(json.dumps(best[0], indent=2)+'\n')
    (args.output/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(stock_size_in=[1, 1, 1], tool_radius_mm=3.175, tool_height_mm=25.0,
                  target_radius_mm=9.0, target_height_mm=18.0, target_half_size_mm=9.0)
    (args.output/'args.json').write_text(json.dumps(config, indent=2)+'\n')
    for post in ('rs274', 'haas'):
        (args.output/f'{post}.nc').write_text(trajectory_to_gcode(best[2], MachineConfig(stock_size_in=(1,1,1)), post=post))
    print(json.dumps(best[0], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial', type=Path, help='CAM seed trajectory (.npy); otherwise random smoke initialization')
    p.add_argument('--target', choices=['sphere', 'cylinder', 'box', 'pyramid', 'sphere_hole', 'sphere_bowl'], default='sphere')
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--resolution', type=int, default=16)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--dt', type=float, default=0.45)
    p.add_argument('--k-start', type=float, default=20)
    p.add_argument('--k-end', type=float, default=120)
    p.add_argument('--cpu', action='store_true')
    optimize(p.parse_args())


if __name__ == '__main__':
    main()
