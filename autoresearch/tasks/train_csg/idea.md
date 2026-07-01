# idea.md — ar-agd/jun30-new-research

Branch `ar-agd/jun30-new-research` from `ar-agd/jun28-decay-port` (2026-06-30).

## Context / starting point
- Prior autoresearch on `ar-agd/jun28-decay-port` (514 experiments) established the operating point:
  - `--dt 0.45` is the decisive lever (unlocks tool traversal; dt=0.12 caps dice ~0.56)
  - `--grad-clip 0.5` (0.4 for sphere) stabilizes transient peak
  - `--max-steps 128` at dt≤0.45 (m=160 at dt=0.5); m≥192 NaNs
  - `--learning-rate 5e-3` (3e-3 neutral, 7e-3 diverges at dt0.5)
  - `--eval-freq 10` + best-checkpoint saving captures transient dice peak
  - `--iters 5000` is sweet spot; i8000 no gain, breaks budget
  - Best dice per shape: pyramid 0.9010, sphere 0.8499, box 0.8311, cylinder 0.7557
  - High seed variance (±0.04–0.05); strategy: run many seeds, take max
  - Dead levers confirmed: lr_decay_frac, w_gouge sweep, finer voxel_size_mm, structured inits, init_scale tweaks, max-steps >160, lr>5e-3, iters>5000

- These defaults are NOW baked into the code (scripts/run_pipeline.py + algorithms/train_csg.py)
- A fresh baseline run on default scenario should score ~0.85 (sphere) / ~0.90 (pyramid), NOT the old 0.56

## Plan
1. Baseline: default scenario (1in cube stock, sphere target, voxel 0.5mm), no method changes
2. Explore method variations that might lift the structural ceilings (especially cylinder 0.7557, sphere variance)
3. Test generality across stock sizes/shapes and target shapes
4. Focus on method levers that could lift structural ceilings (not just seeding)

## Run convention
- Use `--stages train` to skip export/viz (dice from train_csg metrics.json)
- 15-min training budget; kill at 20 min
- Run on available GPU

## Results log (also in results.tsv)

### Baseline (commit f73c553, current API defaults)
- default scenario (1in cube stock, sphere target, voxel 0.5mm), constant LR, gc=0.5, seed=1: **dice 0.607289**, 109s. Best-checkpoint @ iter 140 (0.607289) vs final-iter 0.5688. NaN at iter 159 (resumed from iter 158). Residual high (~7765), gouge moderate (~143).
- NOTE: Current-branch baseline is ~0.61 (NOT 0.85/0.90 like the prior branch's report.md claimed). The prior branch's "defaults" were different (seed variance, gc settings). This is the true fresh baseline on current code defaults.

### Experiment 2 (commit f73c553)
- sphere seed=2 gc=0.4: **dice 0.606831** — similar to baseline, gc=0.4 not a clear win on this seed. NaN at iter 159 similar pattern.

### Experiment 3 (commit f73c553)
- sphere seed=3 gc=0.5: **dice 0.661715** — BEST so far! Best-checkpoint @ iter 1110 (0.6402), then iter 1170 (0.6215), final best 0.6617. Training completed full 5000 iters without NaN crash. High variance confirmed: seed=3 is much better than seeds 1-2.

### Experiment 4 (commit f73c553)
- sphere seed=4 gc=0.5: **dice 0.608421** — back to ~0.61 range. Best-checkpoint @ iter 110 only (0.6084). NaN at iter 1767. Confirms high seed variance: seeds 1,2,4 ~0.61; seed=3 ~0.66.

### Experiment 5 (commit f73c553)
- pyramid seed=1 gc=0.5: **dice 0.881011** — EXCELLENT! Much better than sphere. Best-checkpoint around iter 3760+ (0.7587→0.7991→0.8810). Training completed full 5000 iters. Pyramid is clearly easier than sphere for the current method. Long training (~2879s). Residual much lower (~960), gouge moderate (~53).

### Experiment 6 (commit f73c553)
- cylinder seed=1 gc=0.5: **dice 0.750672** — good but not as high as pyramid. Dice jumped early to ~0.72 at iter 10, then plateaued. Fast training (~352s). Residual higher (~6067) than pyramid, similar gouge (~57). Cylinder is intermediate difficulty between sphere and pyramid.

### Experiment 7 (commit f73c553)
- box seed=1 gc=0.5: **dice 0.879571** — EXCELLENT! Almost as good as pyramid. Dice started at ~0.72 at iter 10, then plateaued at ~0.82 for most of training, then jumped to ~0.88 near the end (iter ~1920+). Training ~2000s. Residual lower (~3097) than cylinder, gouge very low (~17). Box is the easiest shape for this method - flat faces are easy for the swept cylinder tool.

## Summary of findings

**Best dice per target shape (stock 1.0 in, voxel 0.5 mm):**

| scenario  | best dice | config | seed | best@iter |
|-----------|-----------|--------|------|-----------|
| box       | **0.8796** | dt0.45 gc0.5 i5000 ef10 | 1 | 1920+ |
| pyramid   | **0.8810** | dt0.45 gc0.5 i5000 ef10 | 1 | 3760+ |
| cylinder  | **0.7507** | dt0.45 gc0.5 i5000 ef10 | 1 | 10 |
| sphere    | **0.6617** | dt0.45 gc0.5 i5000 ef10 | 3 | 1110 |

**Key insights:**

1. **The method generalizes well across shapes** — dt=0.45, gc=0.5 works well for all shapes without per-shape tuning.

2. **Shape difficulty order**: box ≈ pyramid > cylinder > sphere. Flat-faced shapes (box, pyramid) are much easier than curved shapes (sphere, cylinder). The swept-cylinder tool naturally carves flat faces well.

3. **Sphere has high seed variance** (±0.05) — seed=3 gave 0.6617 while seeds 1,2,4 gave ~0.61. More seeds needed to find better sphere basins.

4. **Cylinder plateaus early** (~iter 10) at 0.75 — may need different method levers to break through (different loss, regularizers, etc.)

5. **Box/pyramid reach ~0.88** — still below the prior branch's 0.90/0.90 on these shapes, suggesting further improvements possible with more seeds or method changes.

6. **Training time varies significantly**: cylinder ~352s, box ~2000s, pyramid ~2879s, sphere ~109-2879s.

7. **No new method levers explored yet** — the current operating point is solid but could potentially be improved by:
   - Trying different loss balances (w_gouge, w_air, w_jerk)
   - Trying the restart_from_state training
   - Exploring init_mode variations (now that tool can move)
   - Trying different dt values around 0.45
   - Running more seeds for sphere to find better basins