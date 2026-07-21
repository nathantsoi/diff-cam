"""Measure argmin-cache staleness along a REAL training trajectory.

Why this exists
---------------
``scripts/grad_check_sweep.py`` validates the sweep gradient against finite
differences and reports ``cos(stale, true)`` under *synthetic* drift on a small
grid (32^3, T=32, random-direction perturbation of the control points). Two
things about that setup are not the operating point:

  * campaign runs are ~1M voxels with T in the hundreds, so the same normalized
    drift crosses proportionally more argmin switching surfaces, and
  * real drift is Adam's descent direction, not isotropic noise.

This script closes that gap by running the actual sweep optimizer on a real
STEP target and, at the point of MAXIMUM staleness in each refresh cycle,
computing both gradients and recording their agreement.

Probe placement
---------------
With ``--amin-refresh R`` the trainer refreshes when ``it % R == 0``. Staleness
is therefore worst at ``it % R == R-1``, just before the next refresh. At that
iteration this script:

  1. computes the STALE gradient (cache untouched) -- this is the gradient the
     optimizer actually steps with, so the trajectory is unaffected;
  2. computes the FRESH gradient, which also re-runs ``find_argmin``;
  3. records cosine, loss gap, argmin churn and path drift.

Step 2 leaves the cache fresh entering ``it+1``, which is exactly the state the
schedule prescribes there anyway -- so the measured trajectory is identical to
an unprobed run. Only the refresh happens one iteration early.

The optimizer step below mirrors ``algorithms/train_csg.py`` (feed-cap and
z-floor barriers in torch, Taichi swept-carve loss, ``B.T`` pullback, gradient
clipping, Adam). It is a separate file so this lane cannot collide with edits
to the trainer.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def percentile_report(name, vals, fmt="{:.6f}"):
    if not vals:
        print(f"{name}: (no samples)")
        return
    a = np.asarray(vals, dtype=np.float64)
    print(f"{name}: n={a.size}  min {fmt.format(a.min())}  "
          f"p10 {fmt.format(np.percentile(a, 10))}  "
          f"median {fmt.format(np.median(a))}  "
          f"mean {fmt.format(a.mean())}  max {fmt.format(a.max())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-sdf-path", default="utils/NPZs/rrph_hi.npz")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=832, help="path samples T")
    ap.add_argument("--n-ctrl", type=int, default=138, help="control points K")
    ap.add_argument("--amin-refresh", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.45)
    ap.add_argument("--feed-ipm", type=float, default=10.0)
    ap.add_argument("--w-feed", type=float, default=5.0)
    ap.add_argument("--w-broad", type=float, default=0.1)
    ap.add_argument("--sigma-broad", type=float, default=4.0)
    ap.add_argument("--w-gouge", type=float, default=4.0)
    ap.add_argument("--w-residual", type=float, default=1.0)
    ap.add_argument("--tool-radius-mm", type=float, default=3.175)
    ap.add_argument("--tool-height-mm", type=float, default=25.0)
    ap.add_argument("--z-floor-epsilon-mm", type=float, default=1.0)
    ap.add_argument("--sweep-init", default="raster_arc")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="", help="optional .npz dump of per-probe series")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from simulator.csg_simulator import CSGSimulatorDelta
    from simulator.sweep import (SweepCarve, bspline_basis, init_reference_path,
                                 fit_control_points)
    from cam.units import ipm_to_mm_per_s, inch_to_mm

    T, K, R = args.max_steps, args.n_ctrl, max(1, args.amin_refresh)
    tool_start = (0.5, 0.5, 1.0)

    # Grid/STEP target: stock box comes from the NPZ (stock_size_in stays None).
    sim = CSGSimulatorDelta(resolution=32, max_steps=T, target_shape="grid",
                            tool_start=tool_start, stock_size_in=None,
                            dt=args.dt, feed_ipm=args.feed_ipm,
                            target_sdf_path=args.target_sdf_path)
    sim.tool_radius[None] = args.tool_radius_mm
    sim.tool_height[None] = args.tool_height_mm
    sim.w_gouge[None] = args.w_gouge
    sim.w_residual[None] = args.w_residual
    sim.bake_target_grid()
    sim.set_target_volume()

    V = sim.Nx * sim.Ny * sim.Nz
    print(f"[target] {os.path.basename(args.target_sdf_path)}  "
          f"grid {sim.Nx}x{sim.Ny}x{sim.Nz} = {V/1e6:.3f}M voxels  "
          f"stock {sim.Lx:.1f}x{sim.Ly:.1f}x{sim.Lz:.1f} mm @ {sim.v:.3f} mm/vox")

    sweep = SweepCarve(sim, n_points=T)
    sweep.w_broad[None] = args.w_broad
    sweep.sigma_broad[None] = args.sigma_broad

    # z-floor, mirroring train_csg (shape-agnostic part bottom from the SDF).
    stock_z_mm = float(sim.Lz)
    sdf_grid = sim.target.to_numpy()
    solid = np.where(sdf_grid <= 0.0)[2]
    part_bottom_z = float(solid.min()) / float(sdf_grid.shape[2]) if len(solid) else 0.0
    z_floor = part_bottom_z - args.z_floor_epsilon_mm / stock_z_mm

    B_np = bspline_basis(K, T)
    feed_mm_s = float(ipm_to_mm_per_s(args.feed_ipm))
    cap_budget_mm = (T - 1) * feed_mm_s * args.dt
    X_ref = init_reference_path(sim, tool_start, T, mode=args.sweep_init,
                                seed=args.seed, max_len_mm=cap_budget_mm)
    P_init = fit_control_points(B_np, X_ref)
    P_init[0] = tool_start

    B_t = torch.from_numpy(B_np)
    P0_const = torch.tensor(tool_start, dtype=torch.float32).unsqueeze(0)
    L_mm_t = torch.tensor([sim.Lx, sim.Ly, sim.Lz], dtype=torch.float32)
    z_floor_t = torch.tensor(float(z_floor), dtype=torch.float32)
    params = torch.tensor(P_init[1:], requires_grad=True)
    opt = torch.optim.Adam([params], lr=args.learning_rate)

    axis_n = np.array([sim.Nx, sim.Ny, sim.Nz], dtype=np.float64)
    print(f"[config] T={T} K={K} amin_refresh={R} lr={args.learning_rate} "
          f"grad_clip={args.grad_clip} w_broad={args.w_broad}\n")

    cos_ti, cos_tot, loss_rel, churn, drift = [], [], [], [], []
    X_at_refresh = None
    t_start = time.time()

    for it in range(args.iters):
        if params.grad is not None:
            params.grad.zero_()
        X = B_t @ torch.cat([P0_const, params], dim=0)
        step_mm = (X[1:] - X[:-1]) * L_mm_t
        speed = step_mm.norm(dim=1) / args.dt
        feed_pen = torch.relu(speed / feed_mm_s - 1.0).pow(2).mean()
        zfloor_pen = torch.relu(z_floor_t - X[:, 2]).pow(2).mean()
        reg_loss = args.w_feed * feed_pen + 100.0 * zfloor_pen
        reg_loss.backward()
        reg_val = float(reg_loss.detach())
        X_det = X.detach()
        X_np = X_det.numpy()

        refresh = (R <= 1) or (it % R == 0)
        ti_loss, grad_X = sweep.loss_and_grad(X_np, refresh_argmin=refresh)
        if refresh:
            X_at_refresh = X_np.copy()

        # --- probe at maximum staleness, just before the next refresh ---
        probe = (R > 1) and (it % R == R - 1)
        if probe:
            amin_before = sweep.amin.to_numpy()
            ti_loss_true, grad_X_true = sweep.loss_and_grad(X_np, refresh_argmin=True)
            amin_after = sweep.amin.to_numpy()

            g_ti_s = (B_t.T @ torch.from_numpy(grad_X))[1:].numpy().ravel()
            g_ti_t = (B_t.T @ torch.from_numpy(grad_X_true))[1:].numpy().ravel()
            reg_g = params.grad.detach().numpy().ravel()
            g_tot_s, g_tot_t = reg_g + g_ti_s, reg_g + g_ti_t

            def cosine(a, b):
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")

            cos_ti.append(cosine(g_ti_s, g_ti_t))
            cos_tot.append(cosine(g_tot_s, g_tot_t))
            # Signed relative gap: >0 means the stale loss OVERSTATES (upper
            # bound, as sweep.py claims); <0 means it understates.
            loss_rel.append((ti_loss - ti_loss_true) / max(abs(ti_loss_true), 1e-12))
            churn.append(float((amin_before != amin_after).mean()))
            if X_at_refresh is not None:
                drift.append(float(np.abs((X_np - X_at_refresh) * axis_n).max()))
            # The fresh call left the cache current, which is the state the
            # schedule prescribes at it+1 -- trajectory is unperturbed.
            X_at_refresh = X_np.copy()

        # --- optimizer step uses the STALE gradient, exactly as training does ---
        grad = params.grad + (B_t.T @ torch.from_numpy(grad_X))[1:]
        params.grad = grad
        if args.grad_clip > 0.0:
            gn = params.grad.norm()
            if gn > args.grad_clip:
                params.grad.mul_(args.grad_clip / (gn + 1e-12))
        opt.step()
        opt.zero_grad()

        if it % 50 == 0:
            tail = f" cos={cos_tot[-1]:.5f} churn={churn[-1]*100:.2f}%" if cos_tot else ""
            print(f"[iter {it:4d}/{args.iters}] loss={ti_loss + reg_val:.5f}"
                  f" |g|={float(grad.norm()):.3e}{tail}", flush=True)

    dur = time.time() - t_start

    # End-to-end quality at the FINAL iterate. Scoring the last iterate (rather
    # than a best-checkpoint) keeps the R arms directly comparable: composite
    # best-checkpoint selection is blind for sweep runs (soft dice is ~0), and
    # the fix for that lives only on jul13-phys-plausible.
    #
    # hard_carve_mask reproduces forward_hard's geometry voxel-for-voxel
    # (tests/sweep_test.py::test_sweep_matches_sequential_carve), so this is the
    # deployable-carve dice, not a soft surrogate. It does NOT apply the
    # evaluator's per-step speed clip -- valid here because both arms share the
    # same feed penalty and the comparison is like-for-like.
    with torch.no_grad():
        X_final = (B_t @ torch.cat([P0_const, params], dim=0)).numpy()
    from simulator.csg_metrics import dice_score, sdf_to_mask
    pred_mask = sweep.hard_carve_mask(X_final.astype(np.float32))
    targ_mask = sdf_to_mask(sim.target.to_numpy())
    final_dice = float(dice_score(pred_mask, targ_mask))

    print(f"\n=== {len(cos_tot)} probes over {args.iters} iters "
          f"({dur:.0f}s, {args.iters/max(dur,1e-9):.2f} it/s) ===")
    print(f"FINAL swept-carve dice (last iterate): {final_dice:.6f}")
    print(f"target {os.path.basename(args.target_sdf_path)}  "
          f"{V/1e6:.3f}M voxels  T={T}  amin_refresh={R}\n")
    percentile_report("cos(stale, true)  Taichi only ", cos_ti)
    percentile_report("cos(stale, true)  full gradient", cos_tot)
    percentile_report("argmin churn (fraction)       ", churn)
    percentile_report("path drift since refresh (vox)", drift, fmt="{:.4f}")
    percentile_report("signed loss gap (stale-true)/true", loss_rel, fmt="{:+.3e}")

    if loss_rel:
        n_over = int((np.asarray(loss_rel) > 0).sum())
        print(f"\nstale loss overstates in {n_over}/{len(loss_rel)} probes "
              f"({100.0*n_over/len(loss_rel):.1f}%) -- sweep.py:216 asserts the "
              f"cached distance makes this an UPPER bound.")

    if args.out:
        np.savez(args.out, cos_taichi=cos_ti, cos_total=cos_tot, churn=churn,
                 drift_vox=drift, loss_rel=loss_rel, amin_refresh=R,
                 iters=args.iters, max_steps=T, n_ctrl=K, voxels=V,
                 final_dice=final_dice, wall_seconds=dur)
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
