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

## Time budget (measured)

5000 iters @ max-steps 128, voxel 0.5mm, eval-freq 10 takes ~30 min on an A6000
(not 15). The eval `forward_hard(T)` every 10 iters is ~half the per-cycle cost
(train forward+backward 0.22s/iter + ~2s/eval). I keep the documented config
(iters 5000 / eval-freq 10) for comparability to the 0.85 ceiling and run 5-7
concurrent on the free GPU farm to maintain throughput. Runs are NOT killed at
20 min as long as they are making progress — the kill threshold is for
runaway/stuck runs, and the protocol's own baseline command takes this long
here.

## k-sweep RESULT (sphere, seed 1, bare random init, k_init=10, 5000 iters)

| k_final | dice (best ckpt) | grad regime | notes |
|---------|------------------|-------------|-------|
| 10 (baseline) | 0.642 | huge/oscillating (8-10) | STUCK ~0.55, best@4250 transient spike; soft-loss grad too biased to improve hard dice |
| 20 | 0.639 | — | no help (k barely above 10) |
| 30 | 0.712 | healthy then small | +0.070 |
| 50 | 0.762 | small (0.02) | +0.120 |
| **80** | **0.784** | small but recovering | **+0.142** — monotonic climb 0.549→0.784, best@4960 (sustained) |

**Monotonic in k_final.** Higher k sharpens the training `apply_cut` toward the
hard `forward_hard` eval → the soft-loss gradient aligns with the hard-dice
metric → optimizer actually improves hard dice. The "gradient death" at high k
(small grad norm) is HARMLESS: the sharp loss is very sensitive to tool position,
so tiny param steps yield large dice gains. k=10's large gradients are BIG but
BIASED (over-erosion), so they don't help hard dice.

**Caveat**: baseline ran on GPU 0, k80 on GPU 3 — cross-GPU (but all identical
A6000s; confound small vs the ±0.05 run variance and the +0.142 gap). Re-seeds
seed2/seed3 of k80 running on GPU 7/9 to confirm.

**BUT**: the bare-random baseline (0.642) is far below the documented 0.85
(which uses the operating point: `--init-mode raster_fine --w-len 0.03
--w-step 0.001`). Batch 2 tests k-anneal ON TOP of the operating point — does it
beat 0.85? Also testing k_final=150/250 (push higher, since k80 was still
climbing monotonically at iter 3500).

## METRIC RESOLUTION (critical)

The documented 0.85 sphere ceiling is **SOFT dice** (the jul1 run's eval used
the soft `forward`). The CURRENT code (autoresearch HEAD `16b6e44`) evals with
`sim.forward_hard(T)` (line 990) — i.e. the reported `dice:` is now **HARD
carve dice** (the honest, deployable metric). At k=10 the soft/hard gap is ~0.30:
soft ≈0.85 but HARD ≈0.55. So:

- op_base (raster_fine + w_len + w_step, k=10): dice @ iter 0 = **0.525**, stays
  ~0.55 — the raster_fine init does NOT give 0.85 on the current (hard-dice)
  code; the old 0.85 was soft dice. k=10 soft optimization cannot lift hard dice
  (gradient biased by over-erosion).
- k-anneal to high k is **closing the soft/hard gap**: it makes the training
  `apply_cut` sharpen toward the hard eval, so the optimizer targets HARD
  coverage. rand_k150 trajectory: 0.549→0.671→0.756→0.798→**0.812** @ iter 2000
  (still climbing) — HARD dice approaching the old soft ceiling, a real
  deployable-dice win. This is exactly the documented #1 frontier.
- k150 (0.812 @ 2000) > k250 (0.769 @ 2000): sweet spot ~100-200, inverts higher.
- op_k80 (0.734 @ 1962) ≈ rand_k80 s1 (0.734 @ 1962): k-anneal is INIT-ROBUST
  (OP vs random barely matters; k dominates). Simplifies: can drop the fragile
  raster_fine/w_len/w_step machinery and use bare random + k-anneal.
- k80 re-seeds: s1 0.734, s2 0.753, s3 0.730 @ ~iter 1800-1960 — reproducible
  across seeds (not a seed-1 fluke).

## k80 multi-seed (confirmed reproducible)

| seed | dice |
|------|------|
| 1 | 0.784 |
| 2 | 0.821 |
| 3 | 0.795 |
| **mean** | **0.800 ± 0.016** |

vs baseline (k=10) hard dice 0.635 best-ckpt (sustained ~0.55). k80 is
**+0.165 over baseline, reproducibly** (3 seeds, all > baseline's best). The
win is real, not seed-1 luck.

## Batch-2 finals (sphere, hard dice)

| config | dice |
|--------|------|
| op_base (rf+w_len+w_step, k=10) | 0.635 |
| op_k50 | 0.775 |
| op_k80 | 0.788 (≈ rand_k80 — OP vs random ~same, k dominates) |
| rand_k250 | 0.773 (too high, inverts) |
| **rand_k150** | **0.830** (best, single seed) |

Sweet spot k_final ∈ [100, 150]. Batch 3 sweeping 100/120/180/200 + k150
re-seeds s2/s3 + k_init=40→150 (faster early climb, was 0.681 @ iter 109).
Also running k150 + viz stages (GPU 8) to confirm hard dice survives trunc
(deployable post-trunc metric).

## Next: refine k_final around 150 (100/120/180/200), re-seed k150, then
generality on box/pyramid/cylinder.

## DEPLOYABLE (viz) REALITY CHECK — important

Ran `--stages train,trunc,eval,export,viz` on k150 and re-ran trunc+viz on the
baseline trajectory. Both trajectories **collide the holder at seg ~16-17** and
get truncated to 40/128 waypoints (min clearance 0.246mm < 1mm threshold):

| config | train dice (pre-trunc) | **viz dice (post-trunc, deployable)** |
|--------|------------------------|---------------------------------------|
| baseline k=10 | 0.642 | **0.579** |
| k150 | 0.830 | **0.648** |

So k-anneal's **deployable** win is **+0.069** (0.579→0.648), NOT the +0.172
pre-trunc illusion. The late-step plunges that boost the pre-trunc train dice
carve the lower interior but collide the holder (25mm tool ≈ 1in stock → holder
clears only when tool base is above stock top → lower interior unreachable
crash-free). This is the **reachability ceiling** the prior toolholder run found
(shape × tool-to-stock-ratio limited). k-anneal carves the deployable prefix
(40 steps) more effectively (+0.069) but can't break the structural ceiling.

**Metric decision**: primary = train-summary hard `dice:` (protocol-compliant,
what results.tsv/baseline use — k150 = 0.830 vs baseline 0.642, +0.172). Note
the deployable viz dice in descriptions for honesty. The viz gap is a
reachability-ceiling issue, not a k-anneal limitation. To raise DEPLOYABLE dice,
need crash-safe trajectories (z-floor + holder clearance so all 128 steps
deploy) — combine k-anneal with the prior run's crash-safe zlayer init, or
enforce a stricter holder penalty. Prior run's crash-safe zlayer sphere viz =
0.819 (50mm tool) / 0.819 (25mm, max-steps 1536) — the bar to beat on viz.

## Next: generality (box/pyramid/cylinder × {k=10, k150} with viz) + crash-safety
test (k150 + higher holder-penalty-weight / z-floor to keep all 128 steps
deployable).

## GENERALITY RESULT (25mm tool, 1in stock, train / viz deployable)

| shape | k=10 train/viz | k150 train/viz | train Δ | viz Δ |
|-------|----------------|----------------|---------|-------|
| sphere | 0.642 / 0.579 | 0.830 / 0.648 | +0.188 | +0.069 |
| box | 0.816 / 0.816 | 0.843 / 0.843 | +0.027 | +0.027 (crash-safe, no gap) |
| pyramid | 0.425 / 0.425 | 0.799 / 0.451 | +0.374 | +0.026 |
| cylinder | 0.802 / 0.717 | 0.895 / 0.793 | +0.093 | +0.076 |
| **mean** | | | **+0.170** | **+0.050** |

k-anneal is shape-blind and helps ALL shapes on train dice (large, esp. pyramid
+0.374). Viz (deployable) wins are modest because the aggressive k-anneal
trajectories collide the holder at seg ~16-17 → truncated to 40/128 waypoints
(reachability ceiling: 25mm tool ≈ 1in stock). Box is the exception (z-invariant,
crash-safe, viz = train). Higher holder-penalty-weight (500) did NOT enforce
crash-safety (viz 0.649 = k150's 0.648) — the soft penalty can't prevent the
structural collision.

## CURRENT EXPERIMENTS (running)

- **50mm tool** (6 runs): longer tool reaches lower interior from above → holder
  clears → full 128 steps deploy → viz should approach train. Prior run got
  sphere viz 0.840 with 50mm + zlayer; testing 50mm + k-anneal (random init) on
  sphere/box/pyramid/cyl. If viz ≈ train, this is the deployable win.
- **lr sweep** (k150 + lr 3e-3 / 5e-3): at high k, grad norm is tiny (0.005-0.2),
  so effective step ~100x smaller than k=10. Higher lr may compensate. lr=1e-3
  was tuned for k=10; k-anneal may want higher.

## 🏆 50mm-TOOL BREAKTHROUGH (deployable viz = train, NO truncation)

The 50mm tool removes the reachability ceiling (longer tool reaches the lower
interior from above, holder clears the stock) → trajectories deploy ALL 128
steps (NO truncation) → **viz dice = train dice**. The k-anneal train-dice wins
become FULLY deployable:

| shape | 25mm k150 viz | 50mm k=10 viz | **50mm k150 viz** | deployable Δ (50mm) |
|-------|---------------|---------------|--------------------|---------------------|
| sphere | 0.648 | 0.638 | **0.830** | **+0.192** |
| pyramid | 0.451 | 0.427 | **0.796** | **+0.369** |
| cylinder | 0.793 | ~0.717* | **0.895** | **+0.178** |
| box | 0.843 | ~0.816* | (re-running) | TBD |

(*cyl/box 50mm k=10 baseline not yet run; 25mm k=10 used as approximation —
tool length barely changes the k=10 stuck-baseline.)

**This is the headline result.** 50mm tool + k-anneal (random init, k_init=10→
k_final=150) gives large, DEPLOYABLE dice wins across all shapes:
- sphere 0.638→0.830, pyramid 0.427→0.796, cylinder 0.717→0.895.
- Pyramid 0.796 CRUSHES the prior zlayer crash-safe result (0.421 with 50mm,
  0.526 with 25mm) — k-anneal is a fundamentally better method for the pyramid.
- Method is shape-blind (k is global; tool length is a physical scenario param,
  not shape metadata) and init-robust (random init, no fragile zlayer tuning).

**lr sweep result**: higher lr HURTS k-anneal (lr3e-3→0.823, lr5e-3→0.776 vs
lr1e-3→0.830). The tiny high-k gradients are well-sized; lr=1e-3 remains optimal.
lr EXHAUSTED for k-anneal too.

## 50mm k150 MULTI-SEED CONFIRMED (viz=train, NO truncation, deployable)

| shape | s1 | s2 | s3 | mean ± std | 50mm k=10 base | viz Δ |
|-------|----|----|----|------------|----------------|-------|
| sphere | 0.830 | 0.832 | 0.798 | **0.820 ± 0.019** | 0.638 | +0.182 |
| pyramid | 0.796 | 0.801 | 0.793 | **0.797 ± 0.004** | 0.427 | +0.370 |
| cylinder | 0.895 | 0.840 | (s3 running) | ~0.868 | 0.774 | +0.094 |
| box | 0.843 | — | — | 0.843 (single) | 0.816 | +0.027 |
| **mean** | | | | **~0.832** | ~0.664 | **+0.168** |

**Reproducible across seeds.** Pyramid is the standout (+0.370, crushes prior
zlayer crash-safe 0.421). The 50mm-tool + k-anneal deployable win is real and
shape-blind. G-code-vs-sim carve Dice = 0.99998 (the deployed G-code reproduces
the simulation exactly — full deployment, no truncation).

**Cylinder high variance** (0.895 vs 0.840 across seeds) — s3 running to nail
down. Sphere s3 (0.798) is the low outlier; mean 0.820 is solid.

## CURRENT: 75mm tool + 2in-stock ratio generality (running)

- **75mm tool** (sphere/pyramid/cyl, k150): does a longer tool beat 50mm? Prior
  zlayer saturated at 50mm for sphere, but pyramid (steep faces) may differ.
- **2in stock + 100mm tool** (sphere, k150 + k=10 baseline, r=22.86mm proportional):
  tool-to-stock RATIO generality test — does the method scale to a larger
  absolute scenario? Same ratio (tool ~2× stock height) as 1in/50mm.
- **cyl k150 s3**: nail down cylinder seed variance.

## Next: read 75mm/2in results; if 75mm helps, re-seed; consider larger stock
sweep (1.5in/2in/3in) to map where the method holds.

## Mid-run k-sweep signal (sphere, seed 1, ~iter 1500)

| run | dice @ ~1500 | grad | read |
|-----|--------------|------|------|
| baseline k=10 | 0.573 @ 2015 | ~2.5 (clipping) | slow random-init ramp |
| k_final=30 | 0.622 @ 1486 | ~3.0 (healthy) | ahead of baseline |
| k_final=50 | 0.654 @ 1469 | 0.02 (near-dead) | fast early, may plateau |
| k_final=80 | 0.715 @ 1518 | 0.09 (recovering) | BEST so far, still climbing |

Higher k_final climbs faster early (training forward sharpens toward the hard
eval → better-aligned gradient) and k80's gradient is recovering, not dead.
Open question: does k80 sustain the climb to ≥0.85, or plateau below? Finals
will decide.

## Notes

- Metric clarification (verified in code): the in-loop eval calls
  `sim.forward_hard(T)` (sharp max, k-INDEPENDENT) then `eval_metrics` — so the
  reported `dice:` is already **hard-carve dice** (best checkpoint, pre-trunc),
  comparable to the documented ~0.85 sphere ceiling. k only affects the TRAINING
  forward (`apply_cut`, soft union) and thus the loss GRADIENT. So k-anneal is
  purely a **gradient-bias reduction** lever: sharpening k late makes the soft
  gradient less over-erosion-biased, better matching the hard-dice metric we
  score. The viz stage (`--stages ...,viz`) gives the post-trunc deployable dice
  for crash-safe runs; use it to confirm a win transfers.
- Seed variance is ±0.04–0.05; a single run that moves <0.02 is noise. Re-seed
  wins ≥3× before believing.
- GPUs 0,3,4,5,6,7,9 free (A6000 48GB). Use CUDA_VISIBLE_DEVICES per run.
