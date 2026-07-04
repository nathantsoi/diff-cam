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

_(populated as experiments run)_

## Methodological reminders

- ≥3 (ideally ≥5) paired same-GPU seeds to call a lever real.
- Dice only comparable on the SAME GPU (atomic-add nondeterminism ±0.01–0.05).
- When a sweep is monotonic, keep going past the apparent edge.
- Don't kill by bare PID (PID reuse) — use nohup, least-loaded GPU, let runs finish.
- Taichi autodiff: all statements inside the top-level for-loop; combine
  `ti.atomic_add`s; mirror Vector-field target params into SCALAR fields
  (`tcx/tcy/tcz/tr_vox/...`) and use `target_sdf_scalar` to avoid the
  `MatrixPtrStmt` load-forwarding assertion when SDF input is grad-tracked.
