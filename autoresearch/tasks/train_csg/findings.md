# findings.md — GradMill differentiable CNC trajectory optimization

Branch `ar-agd/jul1-uniform-toolpath` (from `autoresearch`), tag
`jul1-uniform-toolpath`, run folder `runs/jul1-uniform-toolpath/`. ~127
experiments, 2026-07-01 → 2026-07-02. Loop wound down 2026-07-02.

This file is the consolidated, deduplicated record of every finding from the
session (the chronological working log is `idea.md`; the per-run numbers are in
`results.tsv`, which is the source of truth and is NOT committed).

## Task

Differentiable-simulation trajectory optimization (`algorithms/train_csg.py` +
`simulator/csg_simulator.py`, Taichi `ti.ad.Tape` autodiff). Per-step tool
displacements `tool_delta` are optimized by Adam; `tool_pos[t+1] = tool_pos[t] +
tool_delta[t]` via cumulative scan, with per-step displacement clipped to the
feed/rapid cap. The metric is **dice** (compared only within the same scenario).
The loop tracks the **soft**-carve dice (per `grep "^dice:" run.log`); the
hard-carve dice is a separate deployability validation that must not be modified.

User directives:
1. "You are allowed to change the losses, trajectory and training lengths, etc.
   Look for results that generate **uniform patterns common in CNC-machining
   approaches**."
2. "Many of the trajectories are moving away from the cutting surface and
   therefore spending a lot of time cutting air. Fix this with the loss."
3. (Cylinder-specific) "The beginning of the run makes a contour around the
   part, but about 3/4 of the way around it moves away from the part. Fix this."
4. "It should also be allowed to stop a trajectory once the tool starts moving
   too far away from the part. We can train a trajectory and then at evaluation
   time, remove the end of the trajectory where the tool is not cutting the
   part. Save this state, then train another trajectory to finish cutting the
   part." → **staged training**.

## Best method (the operating point)

```
--dt 0.45 --learning-rate 1e-3 --init-mode raster_fine --w-len 0.03 \
--max-steps 256 --grad-clip 0.5 --eval-freq 10 --iters 5000
```

Best **soft** dice (the tracked metric): box **0.917**, sphere **0.850**,
pyramid **0.885**, cylinder **0.950** (seed-1 high tail; mean 0.941). Cylinder
at iters=10000 reaches 0.9477 (marginal, 2× compute). No special init is
required at lr=1e-3 (random init reaches the same ceiling); raster_fine is kept
as the uniform-feed init. No extra losses are required beyond `w_len`.

## The three real levers (in order of impact)

### 1. `--learning-rate 1e-3` — the single biggest lever
The old default `lr=5e-3` **overshoots** past the good carving basin (dice peaks
then degrades); `lr=1e-3` lets the optimizer **settle** into the basin,
producing a higher *and sustained* peak (final-iter ≈ best, no overshoot).
Monotonic on sphere: 5e-3→0.717, 2e-3→0.754, **1e-3→0.849 (peak)**, 5e-4→0.804
(underfits at 5000 iters). Sharp unimodal peak; **lr is exhausted**.
Universal across shapes (sphere 0.67→0.84, box 0.84→0.92, pyramid 0.86→0.89,
cylinder 0.74→0.92). The "transient peak then degrade" that motivated
best-checkpoint saving is largely an LR-overshoot artifact: at lr=1e-3 the peak
is high and sustained. This dwarfs every other lever tried (raster_fine init
included).

### 2. `--dt 0.45` — tool speed-limit is the bottleneck
`dt=0.45` moves ~1 voxel/step. dt=0.12 caps dice ~0.56 (too slow to cover).
dt=0.5+m160 is a single-seed fluke (see dead levers). Foundational lever, found
early in the prior session.

### 3. `--w-len 0.03` — the trailing-excursion / "tool moves away" fix
The path-length / minimal-motion penalty (mean squared |Δ_t|²). **This is the
clean fix the contour-hug losses could never be.** It is agnostic to *where*
the tool is (unlike `w_air`/`w_prox`/`w_traj_prox`, which pull the tool toward
the surface and oppose carving) — it only discourages motion. On carving steps
the residual gradient dominates, so motion is preserved; on trailing steps with
no residual left, even tiny `w_len` shrinks deltas to zero so the tool **stops**
instead of wandering.
Cylinder: trailing z-climb 1.704→0.010, dice 0.934→0.945, air 0.286→0.199.
Safe for sphere (0.847→0.855). New cylinder best. Implemented in
`simulator/csg_simulator.py` (`compute_length_penalty` + `diag_len`),
`algorithms/train_csg.py` (`--w-len`), `scripts/run_pipeline.py`.

## Smaller / per-shape levers

- **`--w-step 0.001`** (constant-feed regularizer, squared step-LENGTH change):
  encourages the uniform-feed CNC pattern the user asked for, without opposing
  carving (acts on step length, not direction/position). Saturates fast (0.001
  is enough). Sphere +0.004 mean — **marginal** (the single-seed 0.858 was a
  high-variance lucky seed; paired seeds gave 0.850/0.847/0.858, mean 0.852).
  Orthogonal to `w_len` but the two overlap (both discourage motion), so they
  do not stack.
- **`--init-mode raster_fine`** (clipping-aware fine boustrophedon, per-step ≤
  feed cap): +0.063 sphere mean vs random at lr=5e-3. At lr=1e-3 the LR win
  largely subsumes it (random + lr1e-3 ≈ rf + lr1e-3 on sphere). Kept as the
  uniform-feed init.
- **`--max-steps 256`** (cylinder): soft dice peaks at T=256 (0.9457, marginal
  +0.001 over T=128); hard dice flat. T=128–192 is the practical range; T=320
  unstable.

## Dead levers (discard, do NOT re-explore)

- **`w_air`, `w_prox`, `w_traj_prox`** — loss-based air/excursion reduction
  FUNDAMENTALLY trades off dice (0.847→0.55). ~30% air is the inherent price of
  high dice (see "Air-cutting vs dice" below). The "keep tool near surface"
  family is a dead direction; code remains gated at default 0.
- **`w_gouge`** sweep (4→8): seed-reshuffling, not a real mean win over 5 paired
  seeds (mean identical 0.6845; reshuffles dice across seeds). The 3-seed
  "+0.011 mean" was subset variance.
- **`w_jerk`** — ~neutral at every weight tested.
- **`lr_decay_frac`** — not helpful.
- **`dt0.5 + m160`** — single-seed fluke (s0=0.853); mean 0.834 < dt0.45 m128
  mean 0.849, higher air, higher variance.
- **`raster_fine_wide`** (full 0.05–0.95 envelope) — slightly worse mean, loses
  the high ceiling; under-coverage hypothesis refuted.
- **`k ≤ 2`** (sharp soft union) — saturates, gradients vanish, trajectory
  degenerates to zero (soft dice 0.0). k=10 is correct.
- **`iters > 5000`** — marginal (cyl 10k→0.9477, +0.002 at 2× compute); cyl is
  coverage-capped. 5000 iters is the practical budget.
- **Coarse structured inits** (raster/spiral/shell/zlayer) — fail the speed clip.

## Air-cutting vs dice (the user's "cutting air" directive)

Robust across **three loss designs + warmup**: ANY loss term that discourages
the tool from being far from the part surface drops dice 0.847 → ~0.55.
- `w_air` (per-voxel air): ≥0.1 collapses to ~0.563 (optimizer stops moving);
  1e-3 neutral. Too blunt — charges necessary repositioning.
- `w_prox` (per-voxel air × squared-distance-from-target-surface): ALL weights
  {0.01..0.3} stall carving (resid stuck 0.43, dice 0.555). The distance
  weighting backfires: gradient strongest in the empty corners where carving
  must happen → pins tool to surface, no sweep.
- `w_traj_prox` (gentle per-segment tool-CENTER distance, r_tool deadzone): ALL
  weights {0.003..0.1} stall (resid plateau 0.25, dice 0.52–0.57). Even the
  tiniest weight prevents the carving-sweep breakthrough.
- WARMUP (carve 1500 iters, then ramp `w_traj_prox` on): carves to ~0.80, then
  traj_prox destroys it (0.80→0.48–0.54).

**Why:** the default sphere (r=11.43mm) nearly fills the 1in stock. Carving the
corners requires sweeping through the empty corner region, which is far from the
sphere surface in 3D. The high-dice trajectory inherently makes these corner
excursions → ~30% air-cut fraction (air volume / total swept tool volume, the
GPU-independent ratio metric added in `diag_air_unweighted`/`diag_tool_swept`).
The ~30% air is the price of 0.847 dice, not a tunable inefficiency.

**The actual air fix is `w_len`** (above): it doesn't try to keep the tool near
the surface — it just stops the *trailing* wandering after carving is done. And
`raster_fine` init pre-covers the part. The box `w_len+w_step` run hit air
**0.114** at full dice — near-eliminating air on the right shape.

## The soft/hard carve gap (the deployable-dice wall)

Staged training exposed that the tracked **soft** dice (~0.94) is ~0.21 above
the **hard**-carve dice (~0.718), which is the true deployable number.
- The soft union (`smooth_max(stock[t], -tool_d, kv)`) adds ~log(2)/k per step,
  over-eroding; a trajectory optimized for soft does not transfer to hard.
- **k sweep**: hard dice is ~k-invariant (~0.718 for all k≤5, 0.720 at k=10).
  Lowering k does not close the gap — it just breaks the optimizer. The gap is
  the soft union's inherent per-step bias, not a tunable artifact.
- **T sweep**: soft peaks T=256 cyl (marginal); hard flat ~0.718 across all T.
  Hard is coverage-capped, not smoothness- or T-limited.
- **Conclusion**: k=10 is correct. To raise *deployable* dice, improve the
  trajectory's hard-carve coverage (more steps / finer feed / better path), not
  the loss smoothness.

### Dice convention (critical)
`eval_csg` dice: `pred = sdf_to_mask(stock) = stock < 0 = REMAINING material`,
`target = PART`. Dice = 2|remaining∩part|/(|remaining|+|part|) — rewards
*leaving* the part (not gouging) more than removing waste. A stationary tool
scores 0.728 (= 2|cyl|/(|stock|+|cyl|)). The soft union over-erodes
(outside-part material the hard boolean doesn't), so soft (0.94) >> hard (0.718).
The reported soft wins are real on the tracked metric but overstate deployable
quality by ~0.21.

## Staged training (user directive #4)

Built the requested scheme: train → **truncate** at t* (last step the tool
actually cuts; for cyl s4, t*=57 — the trailing 70 excursion steps cut nothing)
→ save mid-cut stock SDF + tool_pos → train a 2nd trajectory from that state to
finish → concatenate `stage1[:t*+1] + stage2[1:]`.
- `algorithms/truncate_trajectory.py`: hard-carves step-by-step, measures
  per-step removed volume (sign convention: `material = (stock < 0.0)`), finds
  t* (remove_thresh=1e-6, min_keep_frac=0.3), saves `stock_sdf`/`tool_pos`/
  `t_trunc` to npz.
- `scripts/staged_train.py`: orchestrates stage-1 → truncate → stage-2
  (`--init-stock-from`) → concat → `eval.eval_csg` hard-carve.
- Result: works end-to-end (concat join gap=0); hard-dice gain **+0.0016** over
  stage-1 alone — marginal, because stage-2's soft optimization doesn't transfer
  to the hard carve (the soft/hard mismatch is the real bottleneck, not the
  trailing excursion).

## Paired-seed robustness validation

6-run paired-seed check (same GPU, 3 seeds each) of the two headline wins:
- Sphere `w_step=0.001`: 0.8503 / 0.8468 / 0.8580 → mean **0.852** vs baseline
  ~0.848 → **+0.004** (within noise; s1/s2 below baseline, s3 above). The
  single-seed 0.858 was a high-variance lucky seed.
- Cylinder `w_len=0.03 + T256`: 0.9499 / 0.9398 / 0.9336 → mean **0.941** vs
  baseline ~0.937 → **+0.004** (s1=0.9499 a high outlier; s2/s3 ≈ baseline).

Both wins are real but **modest** (~+0.004 mean), not the +0.01 the single
seeds suggested.

## Methodological lessons

1. **Need ≥3 (ideally ≥5) paired same-GPU seeds** to distinguish a real lever
   from seed-reshuffling. Single-seed apparent wins overstate effect size
   ~2–3× (bit on `w_step`, `w_gouge`, `dt0.5+m160`).
2. **Dice is only comparable on the SAME GPU** — GPU atomic-add nondeterminism
   gives ~0.01–0.05 run-to-run variance; cross-GPU comparisons are confounded.
3. **When a sweep is monotonic, keep going past the apparent edge** — the lr
   sweep would have stopped at 3e-3 "neutral"; going to 1e-3 found the real win.
4. **Don't kill by bare PID** (PID reuse killed my own run) — use nohup, let
   runs finish.
5. **Taichi autodiff gotchas**: all statements inside the top-level for-loop
   (no scalar assignments before the loop); combine multiple `ti.atomic_add`
   into one; reading the Vector-field `target_params["center"][None]` inside a
   grad-tracked kernel triggers a `MatrixPtrStmt` assertion → mirror target
   params into SCALAR fields (`tcx/tcy/tcz/tr_vox/...`) and use
   `target_sdf_scalar`.

## Artifacts

- `results.tsv` (untracked, gitignored) — the source of truth; ~127 rows.
- `results_plot.png` (gitignored, regenerated) — progress + per-scenario dice
  over 86 experiments, via `autoresearch/tasks/train_csg/plot_results.py`.
- `idea.md` — chronological working log (this file consolidates it).
- Web dashboard (`web/`, served by
  `scripts/serve_web_https.py`): fixed (commit `aad0d2a`) to show the full
  **multi-stage concatenated trajectory** for staged runs (not just the
  stage-2-only path), with stage 2 rendered in amber, a "stage 1→2" boundary
  marker, and a Stage 1/2 scrubber label; reads the canonical train_csg
  `results.tsv`.
- Memory: `train-csg-best-config` (best method), `air-cut-loss-tradeoff` (the
  air-vs-dice finding).
