# idea.md — ar-agd/jul3-hard-carve-gap

Branch: `ar-agd/jul3-hard-carve-gap` (from `autoresearch`), created 2026-07-03.

## CRITICAL RE-FRAME (discovered at run start)

The jul1 run's OWN final commit (`7dc8008`) switched the eval path in
`train_csg.py` from `sim.forward(T)` (soft) to `sim.forward_hard(T)` (hard):
`git log -L 844,845` shows `sim.forward(T)` → `sim.forward_hard(T)` in 7dc8008.
`eval_metrics` reads `stock[T-1]` right after that call, so **the dice the
trainer prints/logs/saves-as-best is now the HARD carve**, not the soft proxy.

Implication: the autoresearch.md "Proven operating point" table (~0.94 cyl /
0.85 sphere) is **stale SOFT dice**. A fresh baseline on this branch should
print the **hard** dice (~0.718 sphere per jul1 findings). So the "soft/hard
gap to close" is mis-framed — the gap is already baked into the tracked number;
this run simply maximizes HARD dice directly. The jul1 levers (lr, dt, w_len,
raster_fine) were tuned against SOFT dice and may be mis-tuned for HARD dice.
**The baseline run will confirm which number actually prints.** All comparisons
this run are HARD dice vs HARD dice.

### LIVE observation during baseline (sphere, bare baseline: random init, w_len=0)

Watched the per-iter dice during the baseline run. **Soft loss drops hard
(0.885 → 0.24) but the HARD dice is FLAT at ~0.55** (spikes to ~0.57) from iter
~0 through ~2760. The eval is `forward_hard`, so that ~0.55 is the true hard
carve of the current trajectory — confirmed: the soft optimizer is chasing
soft-union bias, NOT real hard coverage. The celebrated ~0.85 "soft dice" was
almost entirely soft-union bias (`log(2)/kv ≈ 3.53 voxels/step` of fake erosion
at k=10, kv=10/51), not carving.

**Headline re-frame**: under HARD eval, the bare baseline sphere hard-dice is
~0.55, and the soft gradient is a BAD proxy for hard coverage. The productive
frontier is to make the *training objective* track hard coverage, or to use a
trajectory that inherently covers — NOT to push soft-dice levers (lr/dt/w_len
were all tuned for the biased soft proxy).

### Why k=10 over-erodes so badly (the bias math)

`smooth_max(a,b,k) = (1/k)log(exp(ka)+exp(kb))` = `max(a,b) + (1/k)log(1+exp(-k|a-b|))`.
Per step where the tool is present: bias ≈ `log(2)/kv` = `log(2)·k_ref/k` =
`0.693·51/10 = 3.53 voxels` of fake erosion. A voxel revisited 5× gets ~17
voxels of phantom carving in the soft stock — so the soft stock matches the
target while the hard stock (no bias) barely changes. The optimizer exploits
this gap: it minimizes soft loss without improving hard coverage.

### Two levers this unlocks (both gated by the naive-form exp overflow before)

1. **Numerically-stable `smooth_max`** (max-subtraction LSE; mathematically
   identical, verified to f32 precision) — staged in `simulator_utils.py`.
   Lifts the `kv·stock > 88` overflow that capped k and max_steps.
2. **Higher k** (k=30/60/100): `kv=k/51` ↑, per-step bias `log(2)/kv` ↓
   (k=100 → 0.35 vox/step, 10× less bias). Soft loss tracks hard coverage
   better → optimizer improves REAL carving. Previously impossible (high k →
   `kv·stock` overflow → NaN); stable LSE makes it safe. Risk: high k sharpens
   the softmax gradient (→ 1 on max arg, → 0 elsewhere) — could be too
   sparse/unstable. k≤2 is dead (flat gradient, no learning); the high-k regime
   is UNEXPLORED and the highest-value lever this run.
3. **Larger max_steps** (256/384): more path length → more hard coverage. Was
   capped by the same overflow; now potentially safe.

### EXP 1 — k=30 sphere (commit d3c1547, GPU2, paired with baseline) — WIN

Early trend (hard dice, k=30 vs k=10 baseline flat ~0.55):

| iter | k=10 baseline | k=30 |
|------|---------------|------|
| 0    | 0.549         | 0.549 (identical — hard carve k-invariant at init ✓) |
| 100  | ~0.55         | 0.616 |
| 179  | ~0.55         | 0.640 |

k=30 EXCEEDS the k=10 peak (0.607 @ iter 4100) by iter 179 (4% of training).
Loss at iter 0: 0.49 (k=30) vs 0.89 (k=10) — less soft-union bias → less
phantom gouge. Occasional grad spikes (8e3 at iter 39) caught by grad_clip=0.5;
dice keeps climbing. **High-k lever WORKS**: soft loss tracks hard coverage →
optimizer improves real carving. Final number pending (run ~40min on contended
GPU2). Next: k=60 / k=100 sweep on GPU2 (paired), multi-seed the winner.

### EXP 1 update — k=30 sphere @ iter ~2300: HARD dice ~0.71 (peak 0.7125)

| k | peak hard dice | soft-loss floor |
|---|----------------|-----------------|
| 10 (baseline) | 0.607 @ iter 4100 (flat ~0.55 throughout) | 0.24 |
| 30 | 0.7125 @ ~iter 2300 (still climbing/oscillating ~0.70) | 0.12 |

**+0.10 hard dice from the k lever alone.** Soft loss floor dropped 0.24→0.12 —
the soft objective is now MEANINGFUL (tracks hard coverage) instead of chasing
bias. The optimizer is genuinely carving more material. Dice plateauing/oscillating
~0.70-0.71 past iter 2100 (grad spikes still caught by clip). Best-checkpoint
will capture the peak. Strong evidence to push k higher (k=60/100) — risk:
sharper gradients may stall/NaN at very high k; watch early iters.

### EXP 2 — k=60 sphere (GPU2) + EXP 3 — cyl k=30 (GPU8) — running

k=60: does higher k beat k=30's 0.722, or stall from gradient sparsity?
(As k→∞, smooth_max→max, gradient→indicator: 1 on max arg, 0 elsewhere —
sparse through the T-step chain, possible vanishing for early steps.)
cyl k=30 vs cyl k=10 baseline (0.738): does the k lever help cyl too, or is
cyl already bias-limited differently (tall target, less corner-air exploitation)?

### Midway results (iter ~1500 / ~1090) — k lever keeps giving

| shape | k=10 | k=30 | k=60 | Δ vs k=10 |
|-------|------|------|------|-----------|
| sphere | 0.607 (peak@4100) | 0.722 (peak@3460) | **0.791** peak so far, still rising @ iter1500 | +0.184 |
| cyl | 0.738 (peak@940) | **0.781** peak so far, rising @ iter1090 | — | +0.043 |

k=60 sphere already +0.184 over baseline and still climbing. Monotonic in k so
far (10→30→60 all up). Push k=100 next. cyl also benefits (less dramatically —
cyl's tall target already had less corner-air bias to exploit).

### EXP 2 FINAL — k=60 sphere: HARD dice = 0.810296 (best @ iter 4950, final 0.802)

| k | sphere hard dice | Δ vs k=10 |
|---|------------------|-----------|
| 10 | 0.607552 | — |
| 30 | 0.721839 | +0.114 |
| 60 | **0.810296** | **+0.203** |
| 100 | (running) | ? |

**Monotonic in k (10→30→60 all up).** k=60 hard dice (0.810) now EXCEEDS the
stale "soft 0.85" proxy — and it's the real deployable carve. The k lever
didn't just close the soft/hard gap; it blew past the old soft ceiling by
making the optimizer improve REAL coverage instead of chasing bias. This is the
headline result of the run so far. k=100 launched (GPU2); watch for stall/NaN
from over-sharp gradients.

### EXP 3 FINAL — cyl k=30: HARD dice = 0.810147 (best @ iter 2070, final 0.805)

| shape | k=10 | k=30 | k=60 | k=100 | k=200 |
|-------|------|------|------|-------|-------|
| sphere (m=128) | 0.608 | 0.722 | 0.810 | 0.828 | **0.834** (gains noise-level past 60) |
| cyl (m=256)    | 0.738 | 0.810 | **0.835** | 0.793 ↓ | — |

**k SATURATES and can REVERSE**: cyl k=100 (0.793) < cyl k=60 (0.835) —
non-monotonic. k=100 is past cyl's optimum (likely gradient sparsity in the
256-step chain: high k → softmax weight → 1 on max arg, → 0 elsewhere →
vanishing gradient for early steps in the longer trajectory). Sphere k=200
still slowly rising (0.834) but +0.006 over k=100 is within seed noise. **Best
fixed k ≈ 60–100, shape-dependent.** Strongly motivates **k-anneal** (10→100:
smooth exploration early without over-sharp fixed-k commitment) — lever staged
in code (b95cb06, default off), test after the generality queue. Box k=10
starts at dice 0.79 even at iter 0 (box is the easiest shape — axis-aligned,
swept cylinder covers it well).

### EXP 4 — sphere k=60 m=256: HARD dice = 0.801348 (best@4660, final 0.783)

**m=256 is WORSE than m=128 for sphere at k=60** (0.801 < 0.810). The stable
smooth_max did NOT NaN at m=256 (the old overflow cap is genuinely lifted), but
more steps did NOT raise hard dice — slightly lowered it. **Refutes the
"coverage-capped → add steps" hypothesis**: the ~0.83 sphere ceiling is
geometric/optimization, NOT step-count. Likely causes: (a) 2× more deltas to
tune in the same 5000 iters → under-optimization; (b) extra steps wander into
air (w_len=0) → wasted motion. m=128 remains optimal for sphere. (cyl keeps
m=256 — it's taller, needs the z-steps; cyl k=60 m=256 = 0.835.)

**Implication**: to break past ~0.83 sphere, the lever is NOT more steps. It's
either a better toolpath GEOMETRY (parametric raster/spiral that covers the
sphere exterior systematically), k-anneal (exploration→hard-tracking), or
accepting ~0.83 as near the geometric optimum for a 128-step 1/4" tool path.
Pushing k higher (k=200, running) and multi-seeding the winner are the
near-term priorities; toolpath geometry is the longer frontier.

### Autonomous queue (orchestrator.sh, 2026-07-04 ~02:00)

Two per-GPU orchestrators running 4 experiments each (~2.7h per GPU):
- **GPU2** (after k=200 finishes): box k=10 → box k=60 (generality + k lever on
  box), then sphere k=100 seed 2 & 3 (multi-seed the 0.828 winner).
- **GPU8** (after cyl_k100 finishes): pyramid k=10 → pyramid k=60 (generality +
  k lever on pyramid), then cyl k=60 & k=100 seed 2 (multi-seed cyl).

All bare config (random init, w_len as noted, m=128 sphere/box/pyr, m=256 cyl),
paired same-GPU. Goal: confirm the k lever generalizes across all 4 shapes
(box/pyramid hard baselines unknown — stale soft table says box 0.92 / pyr 0.89,
almost certainly biased high like sphere was), and validate the sphere/cyl
winners aren't single-seed flukes.

### Candidate lever if fixed-high-k stalls: k-ANNEAL (continuation method)

Start k=10 (smooth, broad gradients → good early exploration of the carve
basin), ramp k linearly to ~100 over training (sharp, tracks hard coverage →
late polish of real carving). Best of both: exploration at low k, hard-tracking
at high k. Needs a code change (anneal `sim.k[None]` in the training loop, like
the existing lr anneal). Try ONLY if fixed k=60/100 underperforms k=30.

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
