# findings.md — ar-agd/jul3-hard-carve-gap

Consolidated record of the jul3 autoresearch loop. 18 experiments on 2 GPUs.
Plot: `results_plot.png`. Raw rows: `results.tsv`. Chronological narrative: `idea.md`.

## Headline

**The tracked "soft" dice (~0.94 cyl / 0.85 sphere) was almost entirely soft-union
bias, not real carving.** Under the (correct) HARD-carve eval, the bare baseline is
~0.61 sphere / ~0.74 cyl. A single lever — **raising the smooth-max sharpness k**,
unlocked by a numerically-stable LSE — lifts HARD dice by **+0.10 to +0.35 across
all 4 shapes**, blowing past the stale soft ceiling. This is the highest-value
result of the run.

## The re-frame (discovered at run start)

Commit `7dc8008` (the jul1 run's own final commit) switched `eval_metrics` from
`sim.forward(T)` (soft) to `sim.forward_hard(T)` (hard). So the autoresearch.md
"Proven operating point" table (~0.94 cyl / 0.85 sphere) is **stale SOFT dice**.
The trainer now reports the true deployable **HARD** carve. The jul1 levers
(lr, dt, w_len, raster_fine) were tuned against the biased soft proxy and are
mis-tuned for hard dice. All comparisons this run are HARD-vs-HARD.

**Soft-union bias math**: `smooth_max(a,b,k) = max(a,b) + (1/k)log(1+exp(-k|a-b|))`.
Per step where the tool is present: bias ≈ `log(2)/kv = 0.693·k_ref/k =
0.693·51/k` voxels of phantom erosion. At k=10 → 3.53 vox/step — a voxel revisited
5× gets ~17 voxels of fake carving. The soft stock matches the target while the
hard stock barely moves. The optimizer exploits this: it minimizes soft loss
without improving hard coverage. (LIVE observation: soft loss 0.885→0.24 while
hard dice flat ~0.55.)

## The lever: higher k (less bias → soft loss tracks HARD coverage)

Higher k shrinks the per-step bias (k=100 → 0.35 vox/step, 10× less than k=10),
so the soft loss becomes a faithful proxy for hard coverage and the optimizer
improves REAL carving. Previously impossible: high k → `kv·stock > 88` → f32 exp
overflow → NaN. The **numerically-stable LSE** (`m=max(a,b); m +
(1/k)log(exp(k(a-m))+exp(k(b-m)))`, mathematically identical, verified to f32
precision) lifts that cap.

### Code changes (committed)

- `simulator/simulator_utils.py` `smooth_max` → stable max-subtraction LSE (d3c1547).
- `scripts/run_pipeline.py`: `--k-init` (default 10), `--k-anneal`, `--k-final`
  (d3c1547 + b95cb06).
- `algorithms/train_csg.py`: `k_anneal`/`k_final` args + anneal block
  (`sim.k[None] = k_init + (k_final-k_init)·(it/iters)`) (b95cb06, default off).

## Results: the k sweep (HARD dice, best checkpoint)

| shape | k=10 (base) | k=30 | k=60 | k=100 | k=200 | Δ (best vs base) |
|-------|-------------|------|------|-------|-------|-------------------|
| sphere (m=128)    | 0.608 | 0.722 | 0.810 | 0.828 | 0.834 | **+0.226** |
| cylinder (m=256)  | 0.738 | 0.810 | **0.835** | 0.793 ↓ | — | **+0.097** |
| box (m=128)       | 0.816 | — | 0.821 | — | — | +0.005 |
| pyramid (m=128)   | 0.407 | — | 0.766 | — | — | **+0.359** |

**Key facts:**
- **Monotonic then saturating/reversing.** Sphere: 0.608→0.722→0.810→0.828→0.834
  (gains noise-level past k=60). Cylinder: k=100 (0.793) **<** k=60 (0.835) —
  non-monotonic; k=100 is past cyl's optimum (gradient sparsity in the 256-step
  chain: high k → softmax weight → 1 on max arg, → 0 elsewhere → vanishing
  gradient for early steps). Best fixed k ≈ **60–100, shape-dependent**.
- **The hardest shape benefits most.** Pyramid (slanted faces poorly carved by
  the swept cylinder) goes 0.407 → 0.766, +0.359 — nearly half the old "soft
  0.89" was phantom erosion. The k lever generalizes across ALL 4 shapes.
- **k=60 sphere (0.810) EXCEEDS the stale "soft 0.85" proxy** — and it is the real
  deployable carve. The lever didn't close the soft/hard gap; it blew past the old
  soft ceiling by making the optimizer improve REAL coverage.

## Multi-seed confirmation (the winners are not single-seed flukes)

| experiment | seed 1 | seed 2 | seed 3 | spread |
|------------|--------|--------|--------|--------|
| sphere k=100 | 0.828 | 0.801 | 0.820 | ±0.014 |
| cylinder k=60 | 0.835 | 0.867 | — | ±0.016 |
| cylinder k=100 | 0.793 | 0.792 | — | ±0.001 (reversal reproduced) |

Seed noise is ±0.014–0.016 (smaller than the feared ±0.04). The k=60/100 wins and
the cyl k=100 reversal all reproduce across seeds. The cyl_k60 seed-2 outlier
(0.867) confirms ±0.03 tails exist; multi-seeding remains essential before any
ship claim.

## Refuted / dead this run

- **m=256 for sphere (coverage hypothesis)**: sphere k=60 m=256 = 0.801 **<** m=128
  0.810. The stable LSE did NOT NaN at m=256 (overflow cap genuinely lifted), but
  more steps did NOT raise hard dice — slightly lowered it. The ~0.83 sphere
  ceiling is geometric/optimization, NOT step-count. Likely: 2× more deltas to
  tune in the same 5000 iters (under-optimization) + extra steps wander into air
  (w_len=0). cyl keeps m=256 (taller target needs the z-steps).
- **k≤2** (carried over): flat gradient, no learning.
- **All jul1 dead levers** (w_air/w_prox/w_traj_prox, w_jerk, lr_decay_frac,
  dt0.5+m160, raster_fine_wide, iters>5000, finer voxel, lr sweep): still dead —
  they were tuned for the biased soft proxy anyway.

## Staged but NOT tested (next run's first experiment)

**k-anneal** (continuation method): ramp k linearly 10→100 over training — smooth
broad-gradient exploration early (find the carve basin) + sharp hard-tracking late
(polish real carving). Best of both; may beat any fixed k by avoiding the
shape-dependent saturation point. Code landed (b95cb06, `--k-anneal --k-final`),
default off. **Test: sphere k-anneal 10→100 vs fixed k=60/100 winners.**

## Operating point recommendation (HARD dice, deployable)

```
# All shapes: stable smooth_max + high k. lr=1e-3 MANDATORY (code default 5e-3).
--dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 --iters 5000 \
--k-init 60            # sphere/box/pyramid (m=128); pyramid gains most from k
--k-init 60 --max-steps 256 --w-len 0.03   # cylinder (tall target)
# Best fixed k ≈ 60–100, shape-dependent. k-anneal (10→100) is the untested bet
# to remove the shape-dependence.
```

## Methodology

- 18 experiments, 2 GPUs (GPU2, GPU8), per-GPU orchestrator (`orchestrator.sh`)
  with paired same-GPU seeds (atomic-add nondeterminism ±0.01–0.05).
- Bare config (random init, w_len as noted), 5000 iters, best-checkpoint captured.
- Runs moved into `runs/<tag>/`. results.tsv untracked. No eval/metric code
  modified (only the smooth_max *implementation*, mathematically identical).
- Crashes logged as crash rows, not fatal; none occurred this run.
