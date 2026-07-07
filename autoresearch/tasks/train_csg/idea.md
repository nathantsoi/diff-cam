# jul6-spline-sweep — working log

Branch: `ar-agd/jul6-spline-sweep` (from `aditya-autoresearch` @ 1a383b7, superset of `autoresearch`).
Worktree: `.claude/worktrees/spline-sweep`. GPU: RTX 5070 Ti 12GB (single).
**Every run needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib`** or Taichi silently falls back to Vulkan (verified CUDA works with it).

## The idea (user-proposed, new method — not a tweak)

Replace the per-step delta optimization with a **spline-parameterized toolpath whose
swept volume is computed in one shot**:

- Path = cubic B-spline with K control points; sampled at T = max_steps points
  `X = B @ P` (precomputed basis matrix, torch). `X[0]` pinned to the canonical
  tool start (0.5, 0.5, 1.0).
- Swept SDF: `d(x) = min_s seg_sdf(x, X[s], X[s+1])` with the *exact* sharp
  swept-cylinder geometry used by the evaluator (`tool_sdf_sharp`).
- Carve: `carved(x) = max(stock0(x), -d(x))` — **single union**. Loss = same
  soft-occupancy `w_g*gouge² + w_r*residual²` as `compute_loss`, plus torch
  regularizers on the sampled path (feed-cap penalty so eval speed-clipping is
  inert; z-floor).
- Gradient: two-pass argmin (non-diff argmin pass → diff pass evaluates only the
  winning segment per voxel = exact hard-min subgradient, maxpool-style), Taichi
  autodiff → `path.grad` → `grad_P = Bᵀ grad_X` → Adam on P.

### Why this attacks the known frontier
1. **Soft/hard gap** (~0.21 dice per prior findings): the old forward chains T
   `smooth_max` unions, each adding ~log(2)/k erosion. Here there is ONE hard
   union; the only softness is the 1-voxel sigmoid in the loss. Training
   optimizes (a smooth surrogate of) the true hard-carve dice directly.
2. **Order-independence**: material removal is a union of per-segment sweeps, so
   the final geometry equals the evaluator's sequential hard carve EXACTLY as
   long as steps stay ≤ feed cap (clipping no-op) and z ≥ z_floor.
3. **Speed**: old forward = T sequential N³ kernels + (T+1)×N³ autodiff field.
   New forward = ~2 N³ kernels (argmin + loss), no stock history. Expect ~10×+
   more Adam iters in the fixed 15-min budget; max_steps no longer memory/NaN
   bound (no smooth-union chain).
4. **Parametric low-air toolpath** was explicitly the "productive frontier" in
   prior findings: a low-DOF smooth spline is the natural CNC-like path prior.

### Method rules compliance
- Shape-agnostic: init fits control points to a boustrophedon/helix spanning the
  **target SDF grid bounding box** (allowed: geometric representation only), no
  shape names anywhere.
- Eval untouched: dice comes from the existing `forward_hard` + `eval_metrics`
  path (push sampled deltas into sim). metrics.json/trajectory.npy identical.

## Plan
1. Baseline delta run (protocol) — in progress.
2. Implement `simulator/sweep.py` + `--method sweep` in `train_csg.py` +
   run_pipeline flag forwarding. Commit.
3. Smoke test (iters 50), calibrate iters to the 15-min budget.
4. Experiment loop: K, T, lr, w_feed, sigmoid-scale annealing (if saturation
   plateaus), init variants; then box/cyl/pyramid; ≥3 seeds on the best config.
5. Plot + findings.md.

## Log

- (setup) Worktree + branch created; results.tsv truncated; CUDA-on-WSL fix
  found (`LD_LIBRARY_PATH=/usr/lib/wsl/lib`, libcuda.so not in ldconfig cache).
- Baseline delta sphere run started (killed once by session restart; relaunched).
- Web-research agent on VCPP/differentiable swept volumes relaunched (first one
  was killed by the session restart). Report received — validates the design:
  swept SDF as min-over-time + hard-argmin envelope gradients (Sellán 2021,
  SVSDF TOG 2024), one-sided ReLU² hinge losses with gouge weight annealed UP
  (implicit neural process planning, arXiv 2511.17578), B-spline control-point
  difference bounds for feed constraints (EGO-planner lineage), and the key
  warning that saturated-sigmoid coverage losses have vanishing gradients →
  adopted an SDF-valued attraction term (`w_broad`: relu(d_tool)² on uncut
  waste voxels, gated by stock_occ so it dies once cut).
- **Implemented** `--method sweep` (commit f8babcc): `simulator/sweep.py`
  (SweepCarve: argmin pass + autodiff loss pass; B-spline basis; shape-agnostic
  raster/helix inits fitted by least squares), train_csg integration (torch
  Adam on control points, feed-cap + z-floor penalties in torch, eval through
  the untouched forward_hard/eval_metrics path), run_pipeline forwarding.
- **Tests** (tests/sweep_test.py, all pass + full suite 39 passed):
  - Swept carve == sequential hard carve: **0/15625 voxels differ** — the
    one-shot union reproduces the evaluator's geometry exactly.
  - Autodiff vs finite differences: meaningful-magnitude grads match within
    0.1–3%; sub-1e-4 entries are f32 FD noise (documented in test).
  - Basis partition-of-unity/endpoint pinning; init fit starts at tool start.
- **Baseline (delta, canonical command)**: hard dice **0.6426** (best @ i1840
  ≈ 13 min; final-iter 0.551 — the usual transient-peak-then-degrade). 5000
  iters took 34.7 min on this laptop GPU (protocol command was tuned for a
  bigger card), but the peak fell inside the 15-min window, so 0.6426 stands as
  the fair reference. NOTE: this branch evals HARD dice (forward_hard), so all
  numbers here are deployable-carve dice, not the old soft-dice scale.
- **Smoke (sweep, 200i)**: wiring OK, ~47 ms/iter. Feed penalty dominated the
  loss (74 vs 0.5 geometry) → made it dimensionless (relu(speed/cap - 1)²,
  w_feed 5, commit bd6465d).
- **Exp2 (sweep v1: K40 T256 raster, lr1e-3, w_feed5)**: hard dice **0.8358**
  (best @ i2600, final 0.8346 — SUSTAINED, no over-carve degrade), 12000 iters
  in 4.3 min (~47 it/s incl eval@50 ≈ 15× delta throughput). +0.193 over the
  delta baseline. Gradient collapsed to 6e-4 by i2600 → plateau is gradient
  starvation (sigmoid saturation beyond ~1 voxel from the tube), NOT capacity
  or budget. Next lever: w_broad SDF-valued attraction.
- **Exp3 (+ w_broad 0.1)**: **0.8411** (best @ i11850, still creeping up).
  +0.005; keep. Attraction term works as designed.
- **Exp4 (K=64)**: 0.8401 — capacity is NOT the bottleneck. Discard (keep K40).
- **Exp5 (T=512)**: 0.8355 — executable path length NOT the bottleneck either.
  Discard (keep T256).
- **Ceiling analysis (why ~0.84 plateaus)**: a vertical 3-axis tool cannot
  remove the waste in the shadow below the sphere's equator (any placement
  covering it sweeps up through the part): unreachable volume ≈ πR²·(z_eq) −
  hemisphere + equator fillet ring ≈ 2.2 cm³ vs sphere 6.26 cm³ → structural
  hard-dice ceiling ≈ 2V/(2V + V_shadow) ≈ **0.848** for this scenario. At
  0.8411 we are within ~0.007 of the geometric optimum. The frontier is now
  generality (other shapes have different ceilings), not more sphere tuning.
- (Exp6 = lr 3e-3 tie, logged above the ceiling note in results.tsv.)
- **Exp7 (GENERALITY: box, same config)**: **0.9111** (sustained, final=best;
  508 s ≈ 8.5 min on battery-throttled GPU). Same optimizer/init/losses, zero
  shape-specific code — the bbox raster init + attraction carry over. Box has
  no below-equator shadow → higher ceiling, and the method finds it. Keep.
  Next: cylinder, pyramid, then ≥3 seeds on the sphere config.
- **Exp8 (GENERALITY: cylinder, same config)**: **0.9368** (final iter 0.9244
  — mild late drift, best-checkpoint deployed; 5.5 min). Keep. Vertical walls
  + flat top suit a vertical tool: highest dice yet.
- **Exp9 (GENERALITY: pyramid, same config)**: **0.8285** (sustained, final
  0.8278; 3.6 min). Keep. Tapered walls stair-step under a flat vertical tool
  — the main open gap; slope-aware ideas queued in future_work.md.
- **Exp10/11 (seed variance, sphere best config)**: seed2 **0.8408**, seed3
  **0.8410** vs seed1 0.8411 — spread ±0.0002 (protocol threshold ±0.04).
  The +0.198 over the delta baseline is real and reproducible, and the plateau
  location is deterministic → consistent with the structural-ceiling story.
- Two literature agents (diff. swept volumes / VCPP+CAM) launched — findings
  to be written to future_work.md for later testing.
