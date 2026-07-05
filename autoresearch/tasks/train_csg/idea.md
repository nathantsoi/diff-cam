# idea.md — ar-agd/jul5-anneal-gap

Branch: `ar-agd/jul5-anneal-gap` (from `autoresearch` @ `16b6e44`).
Tag: `jul5-anneal-gap`. Run folder: `runs/jul5-anneal-gap/`.

## Starting point

The proven operating point (soft train-dice, the tracked metric):
`--dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 --iters 5000`
→ sphere ~0.85, box ~0.92, pyramid ~0.89, cylinder ~0.92 (soft dice ceilings).

Soft-dice hyperparameter levers (lr, iters, w_len, w_step, dt, grad-clip) are
EXHAUSTED (sharp unimodal peaks; dead-lever list in autoresearch.md). The
documented open frontier is a **new method lever**, top candidate: **close the
soft/hard carve gap** — the soft `smooth_max` union (k=10) over-erodes by
~log(2)/k per step, so a trajectory optimized for soft dice does not transfer to
the deployed hard carve (`apply_cut_hard`, sharp max).

## Core idea

**Anneal the union sharpness k during training** (high→low): keep k=10 early so
gradients flow (avoids the k≤2 gradient-death failure), then lower k late so the
final-iteration soft stock is sharper / less over-eroded — closer to the hard
carve, lifting the *deployable* dice. The `--k-anneal` / `--k-init` / `--k-final`
flags already exist in run_pipeline.py but are NOT in the dead-lever list and
have NOT been swept at the lr=1e-3 operating point. Shape-blind by construction
(k is a global SDF smoothness param, no shape metadata).

This is a *less-biased soft objective* — exactly frontier candidate #1.

## Plan

1. **Baseline** (this run's reference): exact protocol command, sphere, lr=1e-3,
   soft train dice. Expect ~0.85.
2. `--k-anneal --k-init 10 --k-final 3` (gentle sharpen) vs baseline.
3. Sweep `--k-final` ∈ {5, 3, 2, 1.5} (k_final=2 is the floor before gradient
   death; k-anneal lets us approach it late without the fixed-low death).
4. If k-anneal helps soft dice AND hard viz dice, also test on box/pyramid/cyl
   for generality.
5. Pivot if needed: if k-anneal is dead, try a **hard-carve-aware loss**
   (compute_loss on a sharpened stock while apply_cut stays soft for gradient
   flow) — decouples gradient path from objective, the deeper version of the
   same idea.

## Notes

- Metric: primary = train-summary soft `dice:` (protocol-compliant, comparable
  to documented ceilings). For promising runs also run `--stages train,trunc,viz`
  to read the deployable hard-carve dice.
- Seed variance is ±0.04–0.05; a single run that moves <0.02 is noise. Re-seed
  wins ≥3× before believing.
- GPUs 0,3,4,5,6,7,9 free (A6000 48GB). Use CUDA_VISIBLE_DEVICES per run.
