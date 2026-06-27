"""Profile each component of the CamEnvDisc step loop — v2 (fused build_obs).

Compares OLD flow (to_numpy + cpu obs) vs NEW flow (fused GPU kernel).
"""
import sys, time
sys.path.insert(0, '/workspaces/puffertank/diff-cam')

import numpy as np
import taichi as ti
from simulator.voxel_simulator import CNCSimulator

# ── Config ──────────────────────────────────────────────
RES = 64        # match PPO default
NUM_STEPS = 200
WARMUP = 10

# ── Setup ───────────────────────────────────────────────
sim = CNCSimulator(resolution=RES)
sim.initialize_stock(0.4)
sim.initialize_target_sphere(0.25)
sim.initialize_tool([0.0, 0.5, 0.5], 0.1, 0.3)
sim.init_tool_template()

rng = np.random.default_rng(42)

# ── Profile NEW flow (fused) ────────────────────────────
t_move, t_collision, t_cut = 0.0, 0.0, 0.0
t_build_obs_kernel, t_obs_transfer = 0.0, 0.0
t_total_new = 0.0
prev_excess = 0.0

# Warmup
for _ in range(WARMUP):
    action = rng.integers(0, 27)
    x, y, z = (action // 9) - 1, ((action // 3) % 3) - 1, (action % 3) - 1
    sim.move_tool_one_unit(ti.math.vec3(x, y, z))
    sim.apply_cut()
    sim.compute_excess()

rng2 = np.random.default_rng(42)  # same seed for fair comparison

for step_i in range(NUM_STEPS):
    action = rng2.integers(0, 27)
    x, y, z = (action // 9) - 1, ((action // 3) % 3) - 1, (action % 3) - 1

    step_start = time.perf_counter()

    # excess_before is cached (free)
    excess_before = prev_excess

    # Move tool
    t0 = time.perf_counter()
    _tp = sim.tool_pos[None]
    old_pos = (float(_tp[0]), float(_tp[1]), float(_tp[2]))
    sim.move_tool_one_unit(ti.math.vec3(x, y, z))
    ti.sync()
    t_move += time.perf_counter() - t0

    # Collision check
    t0 = time.perf_counter()
    cuts_target = sim.check_tool_intersects_target()
    ti.sync()
    t_collision += time.perf_counter() - t0

    # Apply cut
    t0 = time.perf_counter()
    if cuts_target == 1:
        sim.tool_pos[None] = ti.math.vec3(*old_pos)
    else:
        sim.apply_cut()
    ti.sync()
    t_cut += time.perf_counter() - t0

    # Fused compute_excess kernel
    t0 = time.perf_counter()
    sim.compute_excess()
    ti.sync()
    t_build_obs_kernel += time.perf_counter() - t0

    # excess field read
    t0 = time.perf_counter()
    excess_after = float(sim.excess_field[None])
    t_obs_transfer += time.perf_counter() - t0
    prev_excess = excess_after

    t_total_new += time.perf_counter() - step_start

# ── Profile OLD flow for comparison ─────────────────────
sim2 = CNCSimulator(resolution=RES)
sim2.initialize_stock(0.4)
sim2.initialize_target_sphere(0.25)
sim2.initialize_tool([0.0, 0.5, 0.5], 0.1, 0.3)
sim2.init_tool_template()

cached_target = sim2.sdf_target.to_numpy()

# Warmup old flow
rng3 = np.random.default_rng(42)
for _ in range(WARMUP):
    action = rng3.integers(0, 27)
    x, y, z = (action // 9) - 1, ((action // 3) % 3) - 1, (action % 3) - 1
    sim2.move_tool_one_unit(ti.math.vec3(x, y, z))
    sim2.apply_cut()
    cached_stock = sim2.sdf_stock.to_numpy()

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

t_old_excess, t_old_tonumpy, t_old_obs, t_total_old = 0.0, 0.0, 0.0, 0.0
rng4 = np.random.default_rng(42)

for step_i in range(NUM_STEPS):
    action = rng4.integers(0, 27)
    x, y, z = (action // 9) - 1, ((action // 3) % 3) - 1, (action % 3) - 1

    step_start = time.perf_counter()

    # Old excess before
    t0 = time.perf_counter()
    cached_stock = sim2.sdf_stock.to_numpy()
    _ = float(np.sum(np.maximum(np.minimum(-cached_stock, cached_target), 0.0)))
    t_old_excess += time.perf_counter() - t0

    sim2.move_tool_one_unit(ti.math.vec3(x, y, z))
    ti.sync()
    sim2.apply_cut()
    ti.sync()

    # Old to_numpy
    t0 = time.perf_counter()
    cached_stock = sim2.sdf_stock.to_numpy()
    t_old_tonumpy += time.perf_counter() - t0

    # Old excess after
    t0 = time.perf_counter()
    _ = float(np.sum(np.maximum(np.minimum(-cached_stock, cached_target), 0.0)))
    t_old_excess += time.perf_counter() - t0

    # Old obs build
    t0 = time.perf_counter()
    tool_pos = sim2.tool_pos[None].to_numpy().astype(np.float32)
    sdf_stock_3d = np.clip(cached_stock, -1.0, 1.0).astype(np.float32)
    sdf_target_3d = np.clip(cached_target, -1.0, 1.0).astype(np.float32)
    res = RES; dx = 1.0/res
    ix = int(np.clip(tool_pos[0]*res, 1, res-2))
    iy = int(np.clip(tool_pos[1]*res, 1, res-2))
    iz = int(np.clip(tool_pos[2]*res, 1, res-2))
    grad_stock = np.array([
        sdf_stock_3d[ix+1,iy,iz] - sdf_stock_3d[ix-1,iy,iz],
        sdf_stock_3d[ix,iy+1,iz] - sdf_stock_3d[ix,iy-1,iz],
        sdf_stock_3d[ix,iy,iz+1] - sdf_stock_3d[ix,iy,iz-1],
    ], dtype=np.float32) / (2.0 * dx)
    diff_3d = sdf_target_3d - sdf_stock_3d
    grad_diff = np.array([
        diff_3d[ix+1,iy,iz] - diff_3d[ix-1,iy,iz],
        diff_3d[ix,iy+1,iz] - diff_3d[ix,iy-1,iz],
        diff_3d[ix,iy,iz+1] - diff_3d[ix,iy,iz-1],
    ], dtype=np.float32) / (2.0 * dx)
    grad_stock = normalize(grad_stock)
    obs_old = np.concatenate([tool_pos, grad_stock, grad_diff,
                              sdf_stock_3d.flatten(), sdf_target_3d.flatten()])
    t_old_obs += time.perf_counter() - t0

    t_total_old += time.perf_counter() - step_start

# ── Report ──────────────────────────────────────────────
n = NUM_STEPS
print(f"\nResolution: {RES}  |  Obs size: {9 + 2*RES**3}  |  Steps: {n}")
print(f"{'='*70}")

print(f"\n--- NEW FLOW (fused build_obs kernel) ---")
print(f"{'Component':<30} {'Per step (ms)':>14} {'% of step':>10}")
print(f"{'-'*55}")
for label, val in [
    ("move_tool_one_unit",       t_move),
    ("check_tool_intersects",    t_collision),
    ("apply_cut",                t_cut),
    ("build_obs kernel (fused)", t_build_obs_kernel),
    ("obs_buffer.to_numpy()",    t_obs_transfer),
]:
    pct = val / t_total_new * 100 if t_total_new > 0 else 0
    print(f"  {label:<28} {val/n*1000:>13.3f}  {pct:>9.1f}%")
print(f"{'-'*55}")
print(f"  {'TOTAL':<28} {t_total_new/n*1000:>13.3f}")
print(f"  SPS: {n/t_total_new:.0f} steps/sec")

print(f"\n--- OLD FLOW (to_numpy + CPU obs) ---")
print(f"{'Component':<30} {'Per step (ms)':>14} {'% of step':>10}")
print(f"{'-'*55}")
for label, val in [
    ("excess (CPU, 2×)",         t_old_excess),
    ("sdf_stock.to_numpy()",     t_old_tonumpy),
    ("build obs (numpy)",        t_old_obs),
]:
    pct = val / t_total_old * 100 if t_total_old > 0 else 0
    print(f"  {label:<28} {val/n*1000:>13.3f}  {pct:>9.1f}%")
print(f"{'-'*55}")
print(f"  {'TOTAL (obs+excess only)':<28} {t_total_old/n*1000:>13.3f}")
print(f"  SPS: {n/t_total_old:.0f} steps/sec")

speedup = t_total_old / t_total_new if t_total_new > 0 else 0
old_obs_cost = (t_old_excess + t_old_tonumpy + t_old_obs) / n * 1000
new_obs_cost = (t_build_obs_kernel + t_obs_transfer) / n * 1000
print(f"\n--- COMPARISON ---")
print(f"  Old obs+excess cost: {old_obs_cost:.3f} ms/step")
print(f"  New obs+excess cost: {new_obs_cost:.3f} ms/step")
print(f"  Obs pipeline savings: {old_obs_cost - new_obs_cost:.3f} ms/step")
