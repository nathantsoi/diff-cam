"""Finite-difference verification of the sweep method's gradient w.r.t. the
B-spline CONTROL POINTS.

Why this exists
---------------
``tests/sweep_test.py`` probes 6 random components of dL/dX (the *sampled path*)
and asserts a pass/fail. Two things are missing:

  1. Nothing perturbs ``P``. The trainer optimizes control points and obtains
     their gradient by the pullback ``grad_P = B.T @ grad_X`` (see
     ``algorithms/train_csg.py``), and that matmul is never exercised against
     finite differences. This script perturbs ``P`` directly, so the pullback
     is inside the measurement.
  2. A 6-of-93 spot check with a pass threshold hides the error DISTRIBUTION.
     For a paper you want median / p90 / max relative error and the cosine
     between the analytic and FD gradient vectors, not a boolean.

The measured quantity is the Taichi swept-carve loss only. The trainer adds two
torch-side barriers (feed cap, z-floor) that are ordinary autograd on an
elementwise expression; the novel machinery -- the two-pass argmin subgradient
and the Taichi->torch handoff -- is what is checked here.

Correctness note: every probe refreshes the per-voxel argmin. Reusing a cached
winner (``--amin-refresh > 1`` in training) measures a DIFFERENT operator, and
comparing it to finite differences of the true min would conflate the envelope
subgradient with staleness. Use ``--stale-cos`` to quantify that separately.

Portability: the three sweep implementations in this repo's history expose
different ``loss_and_grad`` signatures (spline-sweep has no argmin caching, so
no ``refresh_argmin`` kwarg; phys-plausible adds ``want_chip``). The call is
built by introspection so one script runs unmodified on all of them.
"""

import argparse
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_raster(T, z_hi, z_lo, n_rows):
    """Simple descending boustrophedon in normalized [0,1]^3.

    Built here rather than via ``init_reference_path`` so the script does not
    depend on that function's signature, which drifted across branches.
    """
    pts = []
    for r in range(n_rows):
        y = 0.2 + 0.6 * (r / max(1, n_rows - 1))
        xs = np.linspace(0.2, 0.8, 8)
        if r % 2:
            xs = xs[::-1]
        for x in xs:
            pts.append((x, y, 0.0))
    pts = np.asarray(pts, dtype=np.float64)
    # Resample to exactly T samples, then ramp z from z_hi down to z_lo.
    idx = np.linspace(0, len(pts) - 1, T)
    out = np.empty((T, 3))
    for c in range(2):
        out[:, c] = np.interp(idx, np.arange(len(pts)), pts[:, c])
    out[:, 2] = np.linspace(z_hi, z_lo, T)
    return out


def make_caller(sweep):
    """Return ``f(X, refresh) -> (loss, grad_X)`` for any branch's signature."""
    params = inspect.signature(sweep.loss_and_grad).parameters
    has_refresh = "refresh_argmin" in params

    def call(X, refresh=True):
        if has_refresh:
            return sweep.loss_and_grad(X, refresh_argmin=refresh)
        # No caching in this implementation: the argmin is always recomputed,
        # which is exactly what refresh=True asks for.
        if not refresh:
            raise RuntimeError(
                "this sweep implementation has no argmin caching; "
                "--stale-cos is not applicable"
            )
        return sweep.loss_and_grad(X)

    return call, has_refresh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel-size-mm", type=float, default=0.8, help="~32^3 on a 1in cube")
    ap.add_argument("--n-points", type=int, default=32, help="path samples T")
    ap.add_argument("--n-ctrl", type=int, default=16, help="control points K")
    ap.add_argument("--target-radius-mm", type=float, default=11.43)
    ap.add_argument("--w-broad", type=float, default=0.1,
                    help="attraction weight; 0 exposes the bare sigmoid loss")
    ap.add_argument("--sigma-broad", type=float, default=4.0)
    ap.add_argument("--w-gouge", type=float, default=4.0)
    ap.add_argument("--w-residual", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="central-difference step in normalized units")
    ap.add_argument("--eps-sweep", action="store_true",
                    help="repeat over a decade of eps to separate FD noise from bias")
    ap.add_argument("--sig-thresh", type=float, default=1e-3,
                    help="|g| below this is reported separately, not scored")
    ap.add_argument("--stale-cos", type=int, default=0, metavar="N",
                    help="also report cos(grad_stale, grad_true) after N iters of "
                         "drift without an argmin refresh (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    from simulator.csg_simulator import CSGSimulatorDelta
    from simulator.sweep import SweepCarve, bspline_basis, fit_control_points

    T, K = args.n_points, args.n_ctrl
    tool_start = (0.5, 0.5, 1.0)

    sim = CSGSimulatorDelta(resolution=32, max_steps=T, target_shape="sphere",
                            tool_start=tool_start, stock_size_in=(1.0, 1.0, 1.0),
                            voxel_size_mm=args.voxel_size_mm)
    sim.set_target_params(radius_mm=args.target_radius_mm,
                          height_mm=args.target_radius_mm,
                          half_size_mm=args.target_radius_mm,
                          center=(0.5, 0.5, 0.5))
    sim.w_gouge[None] = args.w_gouge
    sim.w_residual[None] = args.w_residual

    sweep = SweepCarve(sim, n_points=T)
    sweep.w_broad[None] = args.w_broad
    sweep.sigma_broad[None] = args.sigma_broad

    call, has_refresh = make_caller(sweep)

    B = bspline_basis(K, T)                       # (T, K)
    X_ref = build_raster(T, z_hi=0.75, z_lo=0.32, n_rows=6)
    P = np.asarray(fit_control_points(B, X_ref), dtype=np.float64)
    P[0] = tool_start                             # pinned, matches the trainer

    print(f"grid {sim.Nx}x{sim.Ny}x{sim.Nz}  T={T}  K={K}  "
          f"w_broad={args.w_broad}  free params={(K - 1) * 3}")
    print(f"implementation: argmin caching {'present' if has_refresh else 'ABSENT'}")

    def loss_of_P(Pm):
        X = (B @ Pm).astype(np.float32)
        return call(X, refresh=True)[0]

    def analytic_grad_P(Pm):
        """Mirror the trainer exactly: grad_P = (B.T @ grad_X)[1:]."""
        X = (B @ Pm).astype(np.float32)
        loss, gX = call(X, refresh=True)
        gP = B.T @ np.asarray(gX, dtype=np.float64)   # (K,3)
        return loss, gP[1:]                            # drop the pinned P[0]

    base_loss, g_an = analytic_grad_P(P)
    print(f"loss at init: {base_loss:.6f}   ||grad_P||: {np.linalg.norm(g_an):.6e}")
    if np.linalg.norm(g_an) == 0.0:
        print("\n*** analytic gradient is IDENTICALLY ZERO ***")
        print("The path receives no gradient signal at all in this configuration.")
        if args.w_broad == 0.0:
            print("This is the saturated-sigmoid regime: with --w-broad 0 the "
                  "occupancy loss has no gradient more than ~2 voxels off-surface.")
        return 1

    eps_list = ([args.eps] if not args.eps_sweep
                else [args.eps * m for m in (0.1, 0.3, 1.0, 3.0, 10.0)])

    for eps in eps_list:
        g_fd = np.zeros_like(g_an)
        for i in range(g_an.shape[0]):
            for c in range(3):
                Pp = P.copy(); Pp[i + 1, c] += eps
                Pm = P.copy(); Pm[i + 1, c] -= eps
                g_fd[i, c] = (loss_of_P(Pp) - loss_of_P(Pm)) / (2.0 * eps)

        a, f = g_an.ravel(), g_fd.ravel()
        denom = np.maximum(np.maximum(np.abs(a), np.abs(f)), 1e-12)
        rel = np.abs(a - f) / denom
        cos = float(a @ f / (np.linalg.norm(a) * np.linalg.norm(f) + 1e-30))

        sig = np.abs(f) > args.sig_thresh
        n_zero_an = int((np.abs(a) == 0.0).sum())
        signflip = int(((a * f) < 0).sum())

        print(f"\n--- eps = {eps:.2e} ---")
        print(f"cosine(analytic, FD)          : {cos:.6f}")
        print(f"components                    : {a.size} total, {int(sig.sum())} "
              f"significant (|FD| > {args.sig_thresh:g})")
        print(f"analytic components exactly 0 : {n_zero_an}")
        print(f"sign disagreements            : {signflip}")
        if sig.any():
            r = rel[sig]
            print(f"rel err (significant)         : median {np.median(r):.3e}  "
                  f"p90 {np.percentile(r, 90):.3e}  max {r.max():.3e}")
            w = int(np.argmax(np.where(sig, rel, -1)))
            print(f"worst significant component   : idx {w}  "
                  f"analytic {a[w]:+.4e}  FD {f[w]:+.4e}")
        print(f"rel err (all)                 : median {np.median(rel):.3e}  "
              f"p90 {np.percentile(rel, 90):.3e}")

    if args.stale_cos > 0:
        if not has_refresh:
            print("\n[stale] skipped: no argmin caching in this implementation.")
        else:
            print(f"\n--- staleness: {args.stale_cos} drift steps, no refresh ---")
            X0 = (B @ P).astype(np.float32)
            call(X0, refresh=True)               # seed the cache
            step = 1e-3
            Pd = P.copy()
            for _ in range(args.stale_cos):
                Pd[1:] += step * np.random.randn(*Pd[1:].shape)
            Xd = (B @ Pd).astype(np.float32)
            _, g_stale = call(Xd, refresh=False)  # cached winners
            _, g_true = call(Xd, refresh=True)    # recomputed winners
            s, t = g_stale.ravel(), g_true.ravel()
            cos_s = float(s @ t / (np.linalg.norm(s) * np.linalg.norm(t) + 1e-30))
            drift_vox = np.abs((B @ (Pd - P))).max() * max(sim.Nx, sim.Ny, sim.Nz)
            print(f"max path drift   : {drift_vox:.3f} voxels")
            print(f"cos(stale, true) : {cos_s:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
