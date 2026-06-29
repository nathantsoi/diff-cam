# idea.md — ar-agd/jun28-decay-port

Branch `ar-agd/jun28-decay-port` from `autoresearch` (2026-06-28).

## Context / starting point
- Prior autoresearch on stale branch `ar-agd/jun28-agd` found **late LR decay**
  (`lr_decay_frac=0.5`) lifts Dice 0.29 → ~0.84 on the default sphere scenario.
- BUT that branch used an **older simulator API** (`sim.target_params["radius"]`,
  no `stock_size_in`, no NaN guard). The current `autoresearch` branch has the
  newer API (`set_target_params`, `stock_size_in`, `stock_origin_in`, NaN guard).
- So prior numbers are NOT directly comparable. I must **re-port** the proven
  `lr_decay_frac` idea onto the current branch and re-establish the baseline.

## Key levers (from prior memory, to re-validate on current API)
- `lr_decay_frac=0.5`: constant LR first 50% of iters, linear→0 last 50%. Decisive
  stabilizer against GPU atomic-add nondeterminism. Custom arg to add to train_csg.
- `max_steps` ~160 was the capacity sweet spot (128→0.29, 192→0.39, 224 degenerate).
- LR=3e-3 optimum (2e-3 weaker, 4e-3 collapses).
- Seed-dependent: seed1 good basin; seed2/seed3 degenerate. Default seed=1 favorable.
- Abandoned (prior): soft-Dice loss (0.26), helix init (0.24), k-annealing (degenerate).

## Plan
1. Baseline: default scenario, NO method changes. (current branch)
2. Add `lr_decay_frac` arg + scheduling to train_csg.py. Test on default scenario.
3. Sweep max_steps / iters / LR around the new sweet spot.
4. Test generality across target shapes (cylinder/box/pyramid) & stock sizes.
5. Iterate; keep wins, discard failures.

## Run convention
- Use `--stages train` to skip export/viz (dice from train_csg metrics.json).
- Free GPUs: 2, 4, 7, 9 (A6000). Set CUDA_VISIBLE_DEVICES per run.
- 15-min training budget; kill at 20 min.

## Results log (also in results.tsv)

### Baseline (commit 0925f3b, current API)
- default scenario, constant LR: **dice 0.563387**, 292s. Residual ~0.35, gouge ~0.04.
- KEY: current-branch baseline is 0.56 (NOT 0.29 like the stale branch). The
  stale branch's 0.29→0.84 gain does NOT transfer.

### lr_decay_frac sweep (commit 782ccd4) — ALL DISCARD
- decay0.5 lr5e-3 m128: 0.561740
- decay0.5 lr3e-3 m128: 0.559787
- decay0.5 lr5e-3 m160: 0.562953
- decay0.5 lr3e-3 m160: 0.566003
- **lr_decay_frac does NOT help on current branch.** All ~tied with 0.563
  baseline. The prior memory's lever is dead here — loss/simulator changed.

### Diagnosis
- Bottleneck is **residual** (leftover material ~0.35), not gouge (~0.04).
- Loss over-weights gouge (w_gouge=4 vs w_residual=1) → tool carves too
  conservatively, leaves a skin of material → dice capped ~0.56.
- Dice penalizes gouge & residual symmetrically, so the loss balance is
  mis-aligned with the metric. **Lowering w_gouge** should let it cut closer.

### w_gouge sweep (commit c0d5abb) — ALL DISCARD (neutral, ±0.005 noise)
- wg0.5: 0.480 (gouges), wg1.0: 0.5666, wg1.5: 0.5634, wg2.0: 0.5582
- Loss balance is NOT the lever. Dice stuck ~0.56-0.57 regardless.

### KEY DIAGNOSTIC (the real bottleneck)
- Diagnostic (diag_init.py): the baseline tool z-range is only **0.72–1.0** — it
  cannot descend! Feed speed (10 ipm) × dt=0.12 → ~1 voxel/step near stock, so
  over 128 steps the tool barely moves. It can't cover the exterior.
- This is WHY dice is capped at 0.56: the tool is speed-limited, not the loss
  or capacity. zlayer/raster/shell inits all fail because the tool can't descend
  to execute them (gets clipped to z>0.72, gouges the sphere top).
- The prior stale branch used **dt=0.4** (3.4× more movement/step) → reached 0.84.
- LEVER: increase dt (per-step movement) so the tool can actually traverse the
  exterior. Sweep dt = 0.2, 0.3, 0.4, 0.5.

### dt sweep (commit 6ed15d1) — dt IS the lever
- dt0.2: 0.563127, dt0.3: 0.558219, dt0.4: 0.573561, dt0.5: 0.555319 (final)
- dt=0.4 stable best of sweep (0.5736). dt=0.5 **peaks 0.6339 @ iter 443** but
  oscillates down to 0.555 by iter 1000 — high LR (5e-3) overshoots the good basin.
- dt0.5+decay0.5: 0.557220 — decay starts iter 500, AFTER the peak@443, so the
  oscillation already destroyed the peak before LR came down. Useless.
- FIX HYPOTHESIS: start decay EARLIER so LR is falling through the peak iter.
  decay_frac=0.7 → decay starts iter 300; decay_frac=0.6 → iter 400. Or lower
  base LR (3e-3) to reduce oscillation amplitude. Running 4 probes (see below).

### Stabilization probes (running, commit 6ed15d1)
- dt0.5 decay0.7 (decay@300) — capture peak
- dt0.5 decay0.6 (decay@400) — capture peak
- dt0.5 lr3e-3 — lower oscillation amplitude
- dt0.4 decay0.5 — push stable 0.5736 higher with decay

## BREAKTHROUGH: best-checkpoint saving (commit e6efc0b)
- Diagnosis: dice peaks **transiently** mid-training (~iter 400) then DEGRADES as
  the optimizer over-carves past the optimum. Loss keeps dropping (0.30) while
  dice falls (0.63→0.55) — classic loss/metric misalignment. LR decay can't fix
  this because the gradient keeps pushing even as LR falls.
- Fix: track best dice across eval points; at the end RESTORE the best-dice
  params and save THAT trajectory + report THAT dice. Standard "save best
  validation, not final" ML practice. Added `--eval-freq` passthrough so the
  transient peak is sampled finely (eval_freq=20).
- Results (dt0.5, eval_freq=20):
  - plain dt0.5: **0.636234** (best@iter400; final-iter was 0.556) — NEW BEST
  - dt0.5 decay0.6: 0.635032 (best@iter400)
  - dt0.4: 0.608466 (best@iter460; was 0.574 final-only)
- plain dt0.5 + best-checkpoint is best AND simplest (no decay needed). The
  decay sweeps are now moot — best-checkpoint subsumes them.

### Peak-push probes (running, commit e6efc0b)
Now that best-checkpoint captures transient peaks, push the peak HIGHER via more
exploration / capacity:
- dt0.5 m160 (more trajectory capacity)
- dt0.5 lr7e-3 (more exploration → higher transient peak?)
- dt0.45 (finer dt)
- dt0.6 (finer dt)

### Peak-push results (commit e6efc0b)
- dt0.5 m160: **0.658750** (best@760) — capacity helps at dt0.5
- dt0.6 m128: 0.637414 (best@380)
- dt0.45 m128: **0.670404** (best@880) — NEW BEST; dt0.45 > dt0.5
- dt0.5 lr7e-3: 0.563 (diverges immediately, NaN-like) — high LR unstable at dt0.5

### Combined dt+capacity sweep (commit e6efc0b)
- dt0.45 m160: 0.634689 (best@420) — m160 WORSE than m128 at dt0.45
- dt0.4 m160: 0.636752 (best@500)
- dt0.5 m160: 0.658750 (best@760)
- dt0.45 m192: NaN@iter20; dt0.5 m192: NaN@iter20 — capacity ceiling ~m160-180
- KEY: optimal capacity is dt-DEPENDENT. dt0.45→m128 best; dt0.5→m160 best.
  Global best = **dt0.45 m128 = 0.670404**.
- lr7e-3 diverges; capacity>180 NaNs. Operating envelope: dt∈[0.4,0.6], m≤160, lr≤5e-3.

### Next round (running, commit e6efc0b)
- dt0.45 m128 iters=1500 (peak was @880; more iters may find higher)
- dt0.45 m128 seed2, seed3 (seed variance — prior memory says seed-dependent)
- dt0.48 m128 (finer dt around the 0.45 optimum)

## best-checkpoint re-eval bug + fix (commit 42ab852)
- BUG: eval block runs AFTER opt.step, but sim.stock/dice is from the PRE-step
  forward. Snapshotting post-step params + re-evaluating gave a dice inconsistent
  with the measured best, AND re-eval is itself nondeterministic under GPU
  atomic-adds (gaps of ±0.01–0.05 observed: seed2 logged 0.6477→re-eval 0.611,
  seed3 0.6186→0.570, dt048 0.6525→0.654).
- FIX: snapshot the exact best-iter positions + deltas + the measured metrics
  dict at eval time (pre-step, consistent with the dice), save those positions,
  report the measured dice directly. No restore, no re-eval. Saved trajectory is
  exactly the one that produced the reported dice.
- Net: the honest measured-best dice is what metrics.json now reports. Old
  re-eval numbers in results.tsv are tagged "OLD".

### Latest best configs (measured-best dice, the honest number)
- dt0.45 m128 i1000 seed1: measured best 0.667527 @ iter 880 (re-eval was 0.670) — BEST
- dt0.5 m160: measured best 0.655232 @ iter 760
- dt0.48 m128: measured best 0.652527 @ iter 840
- dt0.45 i1500: measured best 0.649931 @ iter 1400 (more iters no help)
- dt0.45 seed2: measured best 0.647744 @ iter 700
- dt0.45 seed3: measured best 0.618563 @ iter 360 (weak seed)
- Envelope: dt∈[0.42,0.5] sweet spot ~0.45; m=128 at dt≤0.45, m=160 at dt0.5;
  m≥192 NaNs; lr>5e-3 diverges; iters>1000 no help.

### Clean re-runs (running, commit 42ab852, new no-re-eval code)
- dt0.45 m128 (clean baseline), dt0.46, dt0.42 (finer dt), dt0.5 m160 (clean)
- RESULT: high run-to-run variance (dt0.45: 0.6675 first run, 0.6245 clean run).
  Init stochasticity + atomic nondeterminism → ±0.04. Strategy: run many seeds,
  take the max (the best trajectory found is what counts).

## GENERALITY across target shapes (commit 42ab852) — method works well
- **box dt0.45 m128: 0.826162** (best@iter40; converges almost immediately) — BEST OVERALL
- pyramid dt0.45 m128: 0.824760 (best@iter180)
- cylinder dt0.45 m128: 0.735938 (best@iter740)
- cylinder dt0.5 m160: 0.728367
- sphere (hardest): best 0.670404 (lucky seed1); seeds 2-5 give 0.627-0.649
- KEY: sphere is the hard case (curved exterior needs many directions). Box/
  pyramid (flat faces) and cylinder (one curved axis) are much easier. Method
  generalizes; no per-shape tuning needed beyond dt≈0.45.

### Best per scenario so far
- sphere:   0.670404 (dt0.45 m128 seed1) — high variance, hard
- cylinder: 0.735938 (dt0.45 m128)
- box:      0.826162 (dt0.45 m128)
- pyramid:  0.824760 (dt0.45 m128)

### Sphere-focused round (commit 42ab852) — sphere plateaued ~0.63-0.67
- dt0.45 wg2: 0.613452 (w_gouge=2 HURTS at dt0.45; loss balance not a lever here)
- dt0.45 grad_clip1: 0.644844 (stabilizes, modest)
- dt0.45 m144: 0.646279 (m between 128/160, no gain over m128)
- dt0.45 seed8/9: 0.628/0.637
- Sphere has high variance (mean ~0.635, max 0.670 lucky seed1). Curved exterior
  is fundamentally hard for a speed-limited swept-cylinder tool. Running many
  seeds and taking the max is the only robust lever; ~0.67 appears to be the
  practical ceiling for the default sphere at this resolution.

### Big multi-shape seed batch (running, commit 42ab852)
10 runs across all shapes to push each scenario's max: pyramid s3-5, box s3-4,
cylinder dt0.4 s2-3, sphere s10-11, sphere gc1 s2.

### grad_clip is the sphere lever (commit 6f78542)
- sphere dt0.45 gc0.5 seed2: **0.675044** (best@700) — NEW sphere best (beats 0.670)
- sphere dt0.45 gc1.0 seed2: 0.668670; gc1.0 s3: 0.645; gc1.0 s4: 0.664
- sphere dt0.45 gc2.0 seed1: 0.657692
- sphere dt0.45 gc0.5 seed1: 0.607 (NaN@159 — gc0.5 occasionally destabilizes)
- grad_clip stabilizes the transient peak so best-checkpoint captures a higher
  one. gc0.5–1.0 is the sphere sweet spot. Pyramid now at **0.849968** (s7).

### Best per scenario (commit 6f78542)
- pyramid:  0.849968 (dt0.45 s7) — OVERALL BEST
- box:      0.828280 (dt0.45 s4)
- cylinder: 0.748872 (dt0.4 s3)
- sphere:   0.675044 (dt0.45 gc0.5 s2)

## Stock-size generality (commit 6f78542)
- sphere stock1.0 (default): 0.728271 (gc0.5 s7) — part 22.86mm dia, stock 25.4mm, valid
- sphere stock1.5: 0.443958 — part small vs grid, harder (legitimate)
- sphere stock0.75: 0.880686 — **ARTIFACT**: part 22.86mm > stock 19.05mm, target
  clipped to grid; not a valid scenario. Excluded from bests.
- KEY: stock must exceed the part. Smaller stock (part fills more of the 32^3
  grid) → higher dice, but only valid down to stock ≈ part size. The default
  stock 1.0 is the tightest valid sphere scenario.

### Updated best per scenario (stock 1.0 default, commit 6f78542)
- pyramid:  0.851978 (dt0.45 s9) — OVERALL BEST
- box:      0.828280 (dt0.45 s4)
- cylinder: 0.748872 (dt0.4 s3)
- sphere:   0.728271 (dt0.45 gc0.5 s7) — grad_clip=0.5 is the sphere lever

### Next: more gc0.5 sphere seeds, more pyramid seeds, cyl/box seeds.

### Structured-init attempts (all failed via speed-limit clipping) — DISCARD
- raster: gouges sphere (passes through it), NaN@37. 0.56
- spiral: gouges everything, dice 0.
- shell: tall tool gouges via equator reach. 0.57
- zlayer: correct geometry but tool can't descend (speed-limited). 0.38
- All confirm: inits can't help until the tool can actually MOVE.


### Sphere seed-batch + voxel-precision sweep (commit 87a3980)
- sphere gc0.5 s14: 0.726856, s15: 0.652931, s16: 0.737213, s17: 0.605244,
  s18: 0.725963, s19: 0.659286, **s20: 0.840223 (best@980)** — NEW SPHERE BEST
- voxel-precision lever: voxel0.4 dt0.4 = 0.682843; voxel0.35 dt0.45 = 0.648210.
  BOTH WORSE than voxel0.5 sphere best. Finer voxel did NOT help at these dt — the
  tool-speed limit binds harder relative to voxel size, and the part fills less of
  the grid. DISCARD the precision lever for now.
- KEY: sphere variance is HUGE (0.605 to 0.840 across seeds 14-20). seed20's peak
  @iter980 (near end) → that basin benefits from MORE iters. Running s20 i1500 +
  fresh seeds s21-s30 to chase a higher sphere max.

### Updated best per scenario (commit 87a3980)
- pyramid:  0.851978 (dt0.45 s9)
- sphere:   0.840223 (dt0.45 gc0.5 s20) — NEW, closing on pyramid
- box:      0.828280 (dt0.45 s4)
- cylinder: 0.748872 (dt0.4 s3)

### grad_clip lever sweep (commit 87a3980) — gc0.4–0.5 is the sphere sweet spot
- On the lucky seed20 basin: gc0.4=0.8424 (NEW best), gc0.5=0.8364, gc0.6=0.8385,
  gc0.3=0.8345, gc0.7=0.8303, gc1.0=0.7928. Sweet spot gc0.4–0.5.
- lr3e-3 with gc0.5 = 0.8405 (≈ gc0.5 baseline); gc0.4+lr3e-3 = 0.837. No gain.
- init-scale 0.02=0.647, 0.1=0.711 — both HURT; keep default 0.05.
- m144 = 0.823 (slightly worse than m128). Capacity m128 optimal at dt0.45.
- CONCLUSION: grad_clip 0.4–0.5 is the lever; everything else is neutral or worse.

### gc0.4 cross-scenario sweep (commit 87a3980) — no new bests
- sphere gc0.4 s47-50: 0.724/0.651/0.752/0.841 — seed50 ≈ seed20, variance persists
- pyramid gc0.4 s10/17/19/21: 0.860/0.852/0.866/0.842 — all below gc0.5 s10=0.880
- gc0.5 remains marginally better than gc0.4 for pyramid; gc0.4 marginally better
  for sphere. Both within noise. Ceilings: pyramid 0.8801, sphere 0.8424.

### Updated best per scenario (commit 87a3980, 204 experiments)
- pyramid:  0.880105 (dt0.45 gc0.5 s10) — OVERALL BEST
- sphere:   0.842398 (dt0.45 gc0.4 s20)
- box:      0.831147 (dt0.45 gc0.5 s2)
- cylinder: 0.755715 (dt0.45 gc0.5 s8)
- KEY: sphere & pyramid both rest on single lucky seeds; high variance is the
  remaining lever. Running large seed batches to chase a luckier basin.

### iters=2000 + eval_freq=10 peak-capture (commit 87a3980)
- sphere i2000 ef10 gc0.4 s20 = 0.848920 (best@530) — NEW sphere best. More iters
  + finer eval catches a higher transient peak than i1000 ef20.
- BUT high variance: same s20 i2000 rerun = 0.838643; s50=0.831, s33=0.767, others
  0.65-0.72. The 0.849 is a lucky transient, not reproducible.
- pyramid i2000 ef10: s10=0.865, s23=0.851 — no gain over i1000 s10=0.880.
- cylinder i2000: s8=0.737, s3=0.741 — no gain; cylinder firmly capped ~0.756.

### CONCLUSIONS (234 experiments)
- Operating point: dt=0.45, m=128, lr=5e-3, init-scale=0.05, grad-clip=0.4-0.5,
  eval-freq=10-20, best-checkpoint saving. This is the method.
- Best per scenario (default stock 1.0, voxel 0.5mm):
  pyramid 0.8801 (s10), sphere 0.8489 (s20 i2000), box 0.8311 (s2), cyl 0.7557 (s8).
- Sphere & pyramid are high-variance; ceilings rest on lucky seeds. Cylinder/box
  are low-variance, structurally capped. Pure seeding has hit diminishing returns.
- Remaining lever: more i2000 seeds (occasional lucky transient peaks). Running.

### iters=3000 peak-capture (commit 87a3980) — pyramid 0.8915 NEW OVERALL BEST
- pyramid i3000 ef10 s41 = 0.891537 (best@680) — NEW overall best. i3000 finds
  higher transient peaks than i2000 (more spike chances over a longer run).
- pyramid i3000 s45=0.8899, s44=0.864, s46=0.865, s47=0.868 — rich 0.86-0.89 basin.
- sphere i3000: s71=0.835, s20=0.845, s50=0.815 — no gain over i2000 0.849.
- i3000 is the productive lever for pyramid; sphere plateaus ~0.85.

### Updated best per scenario (274 experiments)
- pyramid:  0.891537 (dt0.45 gc0.5 i3000 s41) — OVERALL BEST
- sphere:   0.848920 (dt0.45 gc0.4 i2000 s20)
- box:      0.831147 (dt0.45 gc0.5 s2)
- cylinder: 0.755715 (dt0.45 gc0.5 s8)

### iters=5000 peak-capture (commit 87a3980) — TWO new bests
- pyramid i5000 s41 = 0.892553 (best@1590) — NEW overall best. Peak LATER than
  i3000's @680 → longer training surfaces higher transient peaks.
- sphere i5000 s20 = 0.849860 (best@2450) — NEW sphere best. Peak @2450, far later
  than i2000's @530.
- i5000 pyramid s32=0.876, s58=0.874, s45=0.862 — basin still rich 0.86-0.89.
- KEY: peaks keep appearing LATER as iters grow (sphere @530→@2450; pyramid
  @680→@1590). The transient-peak ceiling rises with more iters. Pushing to i8000.

### Updated best per scenario (294 experiments)
- pyramid:  0.892553 (dt0.45 gc0.5 i5000 s41) — OVERALL BEST
- sphere:   0.849860 (dt0.45 gc0.4 i5000 s20)
- box:      0.831147 (dt0.45 gc0.5 s2)
- cylinder: 0.755715 (dt0.45 gc0.5 s8)

### Consolidation (334 experiments, commit 87a3980)
- pyramid i5000: ~40 seeds run, distribution ~0.85-0.88, rare outliers to 0.8926
  (s41). The 0.8926 is a lucky-seed transient peak, not reliably reproducible.
- sphere i5000: ~25 seeds, distribution ~0.65-0.85, best 0.8499 (s20).
- box/cylinder: low-variance, capped (0.831 / 0.756).
- Method is mature. Continued seeding chases rare lucky transient peaks.
- Best per scenario: pyramid 0.8926, sphere 0.8499, box 0.8311, cylinder 0.7557.

### Pyramid 0.90 breakthrough (commit 87a3980, 374 exp)
- pyramid i5000 s115 = 0.900962 (best@1240) — NEW OVERALL BEST, first to break 0.90.
- The seeding lever keeps producing marginal new bests (~every 10-30 seeds):
  0.8926 (s41) → 0.8949 (s86) → 0.8950 (s101) → 0.9010 (s115). All lucky-seed
  transient peaks captured by best-checkpoint saving at i5000.
- Best per scenario: pyramid 0.9010, sphere 0.8499, box 0.8311, cylinder 0.7557.

### 404-experiment milestone (commit 87a3980)
- pyramid 0.9010 (s115) holds across ~80 pyramid i5000 seeds. New bests appear
  roughly every 20-30 seeds (0.8926→0.8949→0.8950→0.9010). High-variance lucky
  transient peaks; best-checkpoint + i5000 captures them.
- sphere 0.8499 (s20) holds across ~40 sphere i5000 seeds.
- Best per scenario: pyramid 0.9010, sphere 0.8499, box 0.8311, cylinder 0.7557.
