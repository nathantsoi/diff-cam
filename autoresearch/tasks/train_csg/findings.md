# Findings — jul6-spline-sweep: one-shot spline-swept-volume optimization

Branch `ar-agd/jul6-spline-sweep`. All numbers are **hard-carve dice** from the
untouched evaluator (`forward_hard` from the canonical start + `eval_metrics`),
i.e. deployable-carve quality — not the old soft-dice scale. 15-min budget,
RTX 5070 Ti (WSL2; every run needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib` or
Taichi silently falls back to Vulkan).

## Headline

Replacing the per-step delta optimization with a **B-spline toolpath whose
swept volume is computed and differentiated in one shot** raises sphere hard
dice from **0.643 → 0.841** (+0.198, reproducible to ±0.0002 across 3 seeds)
and trains ~15× faster per iteration than the delta forward. The same
untouched, shape-agnostic config carves **box 0.911, cylinder 0.937,
pyramid 0.829**. The sphere result is within 0.007 of that scenario's
**structural 3-axis ceiling (~0.848)** — the remaining error is geometry
(unreachable below-equator shadow), not optimization.

| scenario | delta baseline | sweep | note |
|---|---|---|---|
| sphere   | 0.643 | **0.841** | at ~0.848 3-axis ceiling; seeds 1/2/3: .8411/.8408/.8410 |
| box      | —     | **0.911** | sustained, final = best |
| cylinder | —     | **0.937** | best shape for a vertical tool |
| pyramid  | —     | **0.829** | tapered walls stair-step — the open frontier |

## The method (commit f8babcc, tuned in bd6465d)

- Path = clamped cubic B-spline, K=40 control points, sampled at T=256:
  X = B·P (precomputed basis; torch). X[0] pinned to the canonical start.
- Swept SDF per voxel = min over segments of the *exact* evaluator tool
  geometry (`tool_sdf_sharp`); carve = one hard union `max(stock0, −d_swept)`.
  Material removal is order-independent, so this equals the evaluator's
  sequential carve exactly — verified **0/15625 voxels differ** in tests —
  provided steps stay under the feed cap (dimensionless penalty
  `relu(speed/cap−1)²` makes eval speed-clipping a no-op) and above z_floor.
- Gradient: two-pass argmin ("maxpool trick") — non-diff pass records the
  winning segment per voxel; diff pass under `ti.ad.Tape` evaluates only that
  segment → exact hard-min subgradient. Pullback `grad_P = Bᵀ grad_X`, Adam.
- Loss: same sigmoid-occupancy `w_g·gouge² + w_r·residual²` as the delta
  method, plus an SDF-valued attraction term (`w_broad=0.1`:
  `relu(d_tool)²/σ²` on uncut waste, gated by stock occupancy) that fixes
  sigmoid gradient starvation — grads had collapsed to ~6e-4 at plateaus.
- Init: shape-agnostic boustrophedon raster over the **target SDF bbox**,
  descending z, least-squares fitted to the spline.

## Why it wins

1. **Kills the soft/hard gap.** The delta forward chains T smooth_max unions,
   each adding ~log(2)/k erosion (the documented ~0.21 soft→hard drop). Here
   there is ONE hard union; the only softness is the 1-voxel loss sigmoid.
   Training optimizes (a smooth surrogate of) the true deployed carve.
2. **~15× throughput.** Old forward: T sequential N³ kernels + (T+1)·N³
   autodiff stock history. New: ~2 N³ kernels, no history. 12k iters ≈ 4 min;
   `max_steps` is no longer a memory/NaN bound (T=512 ran fine).
3. **Right prior.** A low-DOF smooth spline is the natural CNC path space;
   feed-rate feasibility is a differentiable penalty, not an emergent hope.

## What did NOT help (all logged in results.tsv)

- **K=64** control points: 0.840 (tie) — capacity isn't the bottleneck.
- **T=512** samples: 0.836 — chord discretization isn't either (this also
  falsifies continuous-time argmin à la SVSDF as a useful upgrade here).
- **lr 3e-3**: 0.842 (tie) — optimization is converged, not starved.

These ties plus the ±0.0002 seed spread all point the same way: each shape
plateaus at its **geometric** limit for a vertical 3-axis tool, not at an
optimization failure. For the sphere that limit is ≈0.848 (waste below the
equator can't be reached without sweeping up through the part); the pyramid's
limit is set by stair-stepping on 45° faces.

## Where to go next

`future_work.md` holds a ranked, cited experiment list from two literature
scans. Top of the list:
1. **Multi-spline union with safe-z pinned retracts** (retract moves are free
   in this formulation — end control points pinned above stock cut nothing);
2. **Multi-start over pattern-diverse inits** (gradient descent cannot change
   path topology — LaValle §14.7);
3. **Slope-adaptive z-stepping + |N_z|-weighted residual** targeting the
   pyramid's terraces (Dolenc scallop formula, shape-agnostic via ∇SDF);
4. Curriculum loss (annealed sigmoid band, hard-voxel reweighting);
5. CNC-Net-style path-side attraction to residual clusters.
5-axis tool tilt is the only lever that breaks the sphere ceiling itself —
out of scope until the simulator/evaluator grow orientation DOF.

## Artifacts

- `results.tsv` — 11 runs (1 baseline + 10 sweep), exact commands.
- `results_plot.png` — per-experiment progress + per-scenario sweep-vs-delta.
- `runs/jul6-spline-sweep/` — all run dirs (metrics.json, trajectory.npy, STLs).
- `tests/sweep_test.py` — order-independence (0-voxel mismatch), FD gradient
  check, basis properties; full suite 39 passed / 1 skipped.
- `future_work.md` — ranked improvement backlog with sources.
