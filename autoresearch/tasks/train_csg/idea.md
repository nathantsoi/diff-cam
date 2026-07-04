# idea.md — ar-agd/jul4-hard-carve-gap

Branch: `ar-agd/jul4-hard-carve-gap` (created fresh from `autoresearch` on 2026-07-04).
This run continues the **soft/hard carve-gap** frontier opened by `ar-agd/jul3-hard-carve-gap`
(preserved on its own branch in git history) — fresh branch because the prior one already
existed; numbers below are re-established from a clean baseline, not inherited.

## Starting point (the baked-in operating point)

The proven operating point from `ar-agd/jul1-uniform-toolpath` (~127 experiments;
see `autoresearch.md` "Proven operating point & dead levers") is the baseline:

```
--dt 0.45 --learning-rate 1e-3 --init-mode raster_fine --w-len 0.03 \
--max-steps 256 --grad-clip 0.5 --eval-freq 10 --iters 5000
```

Fresh baseline (soft dice, the tracked metric): sphere ~0.85, box ~0.92,
pyramid ~0.89, cylinder ~0.94. **Remember**: code default `--learning-rate` is
still `5e-3` — always pass `--learning-rate 1e-3` explicitly.

## The open frontier: the soft/hard carve gap (~0.21)

The jul1 run's headline *fundamental finding*: the tracked **soft** dice (~0.94 cyl)
is a BIASED proxy. The true deployable **hard**-carve dice is ~0.718 and is
k-invariant, T-invariant (coverage-capped). The soft union over-erodes (adds
~log(2)/k per step), so a trajectory optimized for soft does NOT transfer to hard.
Staged training works end-to-end but gave only +0.0016 hard dice because stage-2's
soft objective doesn't transfer.

**This is the highest-value open lever for deployable dice.** To raise it, improve
the trajectory's hard-carve coverage (more steps / finer feed / better path), or
find a less-biased soft objective whose optimum transfers to hard — NOT loss
smoothness (k is settled at 10) and NOT more soft-dice levers (lr/iters/w_len are
exhausted).

## Plan

1. **Baseline** sphere + cylinder (lr1e-3, default scenario) — re-establish
   reference soft AND hard dice. Measure hard dice (`scripts/staged_train.py` /
   `algorithms/truncate_trajectory.py` hard-carve eval) alongside soft, so every
   idea is judged on the deployable number, not just the biased soft proxy.
2. **Hard-carve coverage levers**: finer feed (smaller per-step cap relative to
   voxel), more max-steps with a motion budget, parametric low-air toolpath
   (raster/spiral — inherently uniform + covers systematically). Goal: lift HARD
   dice, accept soft-dice neutrality.
3. **Less-biased soft objective**: experiment with union forms whose per-step bias
   is smaller than log(2)/k WITHOUT breaking gradients (k<=2 is dead — saturates;
   look for alternatives, e.g. a corrected/smoothed union, or anneal k during
   training). Judge by soft-vs-hard transfer, not soft alone.
4. **Parametric toolpath** (major architectural direction if the above stalls):
   low-dim raster/spiral parameters optimized end-to-end — directly serves the
   "uniform CNC patterns" + "less air" directives and may cover more hard material
   per step than free-form tool_delta.
5. Validate any real win across sphere/cylinder/box/pyramid with ≥3 paired same-GPU
   seeds before claiming it (single-seed wins overstate ~2–3× — bit the prior run).

## Dead levers (do NOT re-explore — see autoresearch.md)

w_air / w_prox / w_traj_prox (contour-hug losses trade dice 0.847→0.55; ~30% air
is the price of high dice), w_gouge (seed-reshuffling), w_jerk, lr_decay_frac,
dt0.5+m160 (single-seed fluke), raster_fine_wide, k≤2 (saturates), iters>5000
(marginal, 2× compute), finer voxel_size_mm, coarse structured inits, lr sweep
(exhausted, peak 1e-3).

## Notes / findings

### Pivotal: the tracked metric is now HARD dice (autoresearch.md "0.85" baseline is STALE)

Git-blame on `algorithms/train_csg.py:844` (`sim.forward_hard(T)` in the eval
block) → commit `7dc8008` (the jul1 PR #5 merge, 2026-07-03 15:26). The jul1
*loop* ran Jul 1–2 with the **soft** `forward()` eval → its findings.md soft-dice
numbers (sphere 0.85, cyl 0.94) are pre-port. The merge ported eval to
`forward_hard` (`apply_cut_hard` = exact `ti.max` union with `tool_sdf_sharp`,
binary-mask `dice_score`) AFTER the loop wound down. So:

- **This run's tracked `dice:` is HARD-carve dice**, not soft. autoresearch.md's
  "fresh baseline scores ~0.85 (sphere)" / "tracked soft dice ~0.94" text is
  stale (predates the port). The real baseline is the hard-dice number this run
  establishes (~0.72 cyl per jul1's separate hard measurement; sphere TBD).
- The "open frontier — close the soft/hard gap" is now DIRECTLY the tracked
  metric: every experiment already selects/saves checkpoints by hard dice, and
  the soft loss is only the differentiable proxy. Good — plan aligns.
- **Dice convention** (jul1 findings): `pred = stock<0 = REMAINING material`,
  `target = PART`. A STATIONARY tool scores 2|target|/(|stock|+|target|) =
  **0.553 (sphere r=11.43)** / 0.728 (cyl). Baseline iter 130 = 0.5486 ≈ the
  stationary floor (trajectory hasn't carved waste yet). Soft-optimized hard
  carve actively GOUGES (cyl hard 0.718 < stationary 0.728) — the gap is real.

**Implication**: the "proven operating point" (lr=1e-3 → 0.85) was tuned for
SOFT dice. For HARD dice it may be suboptimal — lr is worth re-examining on the
hard metric. The productive levers for HARD dice are coverage (more steps /
finer feed / better path) and a less-biased soft objective, NOT the exhausted
soft-dice knobs. Note: jul1's "T≥192 NaNs / iters>5000 marginal" were SOFT-dice
findings — for HARD (coverage-capped) dice they may differ; re-test on hard.

### jul1 already ruled out k-annealing AND more-steps for hard dice
jul1 findings: **hard dice is k-INVARIANT** (~0.718 for all k≤5, 0.720 at k=10)
and **T-INVARIANT** (hard flat ~0.718 across all T). `smooth_max(a,b,k) =
(1/k)log(exp(ka)+exp(kb))`; larger k = sharper = less bias, but k≤2 saturates
(gradients vanish). So: do NOT try k-annealing (k-invariant) or more steps of
the SOFT-optimized path (T-invariant). **T-invariance was tested on
soft-optimized paths only** — a SYSTEMATIC coverage path (parametric raster) may
use T more effectively, so "better path" is NOT ruled out. jul1 cyl hard 0.718 ≈
stationary cyl 0.728; sphere hard ~0.553 = stationary sphere 0.553. ⇒ the
soft-optimized trajectory does not carve the part in hard space AT ALL (soft
over-erosion did 100% of the apparent soft-dice work). Huge room: 0.553 →
potentially 0.8+ if real coverage is forced.

### Speed problem: forward_hard eval doubled per-iter cost (port side-effect)
jul1's 5.5 iter/s (soft eval) → now 2.66 iter/s (soft forward + forward_hard eval
every 10 iters) = 5000 iters in ~31 min, OVER the 20-min kill threshold. For
experiments use `--eval-freq 25` (≈22 min) or `--eval-freq 50` (≈18 min) +
`--iters 3000` if hard dice is flat (no transient peak to capture on hard). The
baseline (already running at freq=10) is allowed to run long — it's the
reference. Hard dice appears flat (no sharp peak), so coarser eval is safe.

### Experiment priority (after baseline confirms the stationary floor)
1. **Coverage diagnostic**: eval the `raster_fine` init's HARD dice with no/1
   iter optimization. If >> 0.553, coverage is confirmed as the lever and soft
   optimization is what collapses it. (cheap, fast)
2. **Parametric toolpath** (the real lever): low-dim raster/spiral params
   optimized end-to-end so the optimizer can ONLY produce systematic sweeps —
   cannot cheat with small motions. Directly serves "uniform CNC patterns" +
   coverage. (big, architectural)
3. Re-examine lr / w_len on the HARD metric (they were tuned for soft). (cheap)

## Methodological reminders

- ≥3 (ideally ≥5) paired same-GPU seeds to call a lever real.
- Dice only comparable on the SAME GPU (atomic-add nondeterminism ±0.01–0.05).
- When a sweep is monotonic, keep going past the apparent edge.
- Don't kill by bare PID (PID reuse) — use nohup, least-loaded GPU, let runs finish.
- Taichi autodiff: all statements inside the top-level for-loop; combine
  `ti.atomic_add`s; mirror Vector-field target params into SCALAR fields
  (`tcx/tcy/tcz/tr_vox/...`) and use `target_sdf_scalar` to avoid the
  `MatrixPtrStmt` load-forwarding assertion when SDF input is grad-tracked.
