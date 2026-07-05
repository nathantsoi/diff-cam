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

**Anneal the union sharpness k during training** — LOW early (smooth, gradient
flow, exploration) → HIGH late (sharp, less over-erosion, closer to hard carve).

Sign convention (verified from `smooth_max` in simulator_utils.py + SDF-negative-
inside): `smooth_max = max + (1/k)·log(1+exp(-k|a-b|))`, so the soft-union excess
over the hard carve is ~log(2)/k. **Lower k ⇒ MORE over-erosion; higher k ⇒
sharper ⇒ LESS over-erosion** (closer to `apply_cut_hard`). The default k=10 is
constant. To close the soft/hard gap we want k HIGH at the final iteration so the
optimized trajectory is less biased toward over-carving — lifting deployable
hard dice AND likely soft dice (less gouge + less residual-region over-erosion).

The dead-lever note "k≤2 saturates, gradients vanish" is the LOW-k failure
(very smooth). High-k has its own failure (softmax→one-hot → gradient
concentrates/vanishes), so there is a sweet spot. `--k-anneal`/`--k-init`/
`--k-final` exist but are NOT swept at lr=1e-3 and NOT in the dead list.
Shape-blind (k is a global SDF smoothness param). This is a *less-biased soft
objective* — frontier candidate #1.

## Plan

1. **Baseline** (this run's reference): exact protocol command, sphere, lr=1e-3,
   k=10 constant, soft train dice. Expect ~0.85. (RUNNING on GPU 0.)
2. `--k-anneal --k-init 10 --k-final 30` (sharpen late) vs baseline.
3. Sweep `--k-final` ∈ {20, 30, 50, 80} with k_init=10; also try k_init=5 (more
   early exploration) → k_final=30.
4. If a k_final helps soft dice, re-seed ≥3× and also read hard viz dice
   (`--stages train,trunc,viz`) to confirm the gap actually closes.
5. Generality: run the winning k-anneal config on box/pyramid/cylinder.
6. Pivot if dead: **hard-carve-aware loss** — keep `apply_cut` soft (gradient
   path) but evaluate `compute_loss` on a separately-sharpened stock replica, so
   the objective is less biased without killing gradients.

## Notes

- Metric: primary = train-summary soft `dice:` (protocol-compliant, comparable
  to documented ceilings). For promising runs also run `--stages train,trunc,viz`
  to read the deployable hard-carve dice.
- Seed variance is ±0.04–0.05; a single run that moves <0.02 is noise. Re-seed
  wins ≥3× before believing.
- GPUs 0,3,4,5,6,7,9 free (A6000 48GB). Use CUDA_VISIBLE_DEVICES per run.
