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
  was killed by the session restart).
