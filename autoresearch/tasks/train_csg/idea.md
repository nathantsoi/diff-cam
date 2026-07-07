# idea.md — <next run tag>

Branch: `ar-agd/<tag>` (from `autoresearch`).
Run folder: `runs/<tag>/`.

## Starting point

The proven operating point (soft train-dice, the tracked metric):
`--dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 10 --iters 5000`
→ sphere ~0.85, box ~0.92, pyramid ~0.89, cylinder ~0.92 (soft dice ceilings).

Soft-dice hyperparameter levers (lr, iters, w_len, w_step, dt, grad-clip) are
EXHAUSTED (sharp unimodal peaks; dead-lever list in autoresearch.md).

## New this run: trajectory-quality measures (2026-07-06)

Three deployable measures now reported per run (read from `runs/<run>/metrics.json`,
NOT the `^metric:` log lines): `air_time` (s), `total_time` (s), `break_prob_any`
([0,1]); plus `fcut_max`, `engage_max`, `broken`, `best_score`. Composite
best-checkpoint: `best_score = dice - best_w_airtime*air_time_norm -
best_w_time*total_time_norm - best_w_break*break_prob_any`.

New CLI levers (all **uncalibrated** — defaults are starting points, NOT proven):
- Soft loss terms: `--w-time` / `--w-air-time` / `--w-break` (default `1e-3` each).
- Best-checkpoint weights: `--best-w-airtime` / `--best-w-time` / `--best-w-break`
  (default `0.05` each).
- Breakage model: `--kc 700`, `--f-ref 50`, `--f-max 100`, `--sigma-risk 0.5`.
  At voxel 0.5 mm + 3.175 mm tool these tend to give `fcut_max=engage_max=0`
  → **calibrate `--f-ref`/`--f-max` DOWN until `fcut_max`/`engage_max` are
  nonzero at the operating point before trusting `break_prob_any`.**

## First-run plan

1. Baseline: re-establish dice at the proven operating point AND record
   `air_time`/`total_time`/`break_prob_any`/`best_score` (calibrate breakage
   model if all zero). This is the traj-quality baseline to compare against.
2. Then explore the new levers — tune `--w-time`/`--w-air-time`/`--w-break`
   and `--best-w-*` for dice-preserving (or dice-improving) reductions in
   `air_time`/`total_time`/`break_prob_any`. A dice-neutral air-time win is a
   real deployable win.
3. Record calibration + working weights in findings.md so the next run does
   not re-calibrate from scratch.

## Notes

(filled in as the run progresses)

## KEY FINDING (08:40 jul7) — eval switched to HARD dice in 7dc8008; baseline is SOFT

**Corrected:** the prior "soft-loss defaults poison the baseline" finding was WRONG.
Zeroing `--w-time/--w-air-time/--w-break` does NOT restore 0.85. The real cause:

Commit `7dc8008` ("Ar agd/jul1 uniform toolpath (#5)") changed eval to call
`sim.forward_hard(T)` (sharp boolean `ti.max` carve) instead of `sim.forward(T)`
(smooth_max). So the "dice" logged since jul1-commit-7dc8008 is **HARD dice**, not
the SOFT dice the 0.85 proven operating point was measured on.

- jul1 control @ commit 246a184 (parent of 7dc8008), SOFT dice: 0.71 @ iter 300,
  climbing toward 0.85. Confirmed via worktree test.
- Every commit from 7dc8008 through HEAD, HARD dice with random init: 0.54-0.59,
  flat (climbs only in 1-voxel steps because boolean-max carve is quantized).
- `--no-enforce-z-floor` does NOT help (still 0.548). z_floor is NOT the culprit.

The 0.85/0.92/0.89/0.92 ceilings in the task spec are SOFT dice. HARD dice with
random init + soft-optimization caps ~0.59 (jul4 reached HARD 0.93 only with
shape-aware zlayer init, per train-csg-best-config memory).

**Fix:** eval must report BOTH — SOFT dice (proven 0.85 metric, used for
best_score) AND HARD dice + traj-quality (air_time/total_time/break_prob_any,
the deployable measures computed by compute_traj_diagnostics_hard on the hard
carve). Implemented in train_csg.py eval block.

The traj-quality **metrics** (air_time/total_time/break_prob_any) are reported
regardless of the soft-loss weights (computed by `compute_traj_diagnostics_hard`,
not the loss). So setting `--w-time 0 --w-air-time 0 --w-break 0` gives clean
dice optimization AND still reports the traj-quality numbers.

Also setting `--best-w-airtime 0 --best-w-time 0 --best-w-break 0` for the baseline
so checkpoint selection is pure-dice (the composite only affects WHICH iter is
"best", not the optimization).

## Run log (chronological)

### Run 1 — contaminated sphere baseline (KILLED 21:55, MISDIAGNOSED)
Used new defaults w_time=w_air_time=w_break=1e-3. Dice stuck ~0.55. Initially
blamed soft-loss defaults; ZEROING them did not help. Real cause found jul7
(see KEY FINDING above): eval switched to HARD dice in 7dc8008.

### Run 2 — clean baselines (GPUs 0,3,4,5, started 22:00, ALSO MISDIAGNOSED)
Cmd (sphere): `... --w-time 0 --w-air-time 0 --w-break 0 --best-w-airtime 0 --best-w-time 0 --best-w-break 0`
Results: sphere HARD dice 0.65, pyramid HARD 0.43, box HARD 0.81, cyl HARD 0.78.
These are HARD dice, not comparable to the 0.85 SOFT baseline. Re-running after
the eval fix below with SOFT dice reported.

### Bisect (jul7 08:40) — confirmed 7dc8008 is the metric-switch commit
Worktree tests @ 246a184 (parent) climb (SOFT 0.71 @ iter 300). Every commit from
7dc8008 onward flat at HARD 0.54-0.59. z_floor (3230c17) is NOT the cause.

### Run 3 — clean baselines w/ eval fix (SOFT dice primary) — DONE 09:54
Re-ran all 4 shapes at the proven operating point with the eval fix (commit
1a6fc67: eval reports BOTH soft dice `dice` + hard dice `hard_dice`; best_score
uses SOFT). `--w-time 0 --w-air-time 0 --w-break 0 --best-w-* 0` (pure-dice opt,
traj-quality still REPORTED via compute_traj_diagnostics_hard). SOFT dice:

| shape | soft dice | hard dice | air_time(s) | total_time(s) | air% | break_prob | fcut_max |
|-------|-----------|-----------|-------------|---------------|------|------------|----------|
| sphere (random)  | 0.842 | 0.565 | 4.13 | 19.6 | 21% | 4e-4 | 0 |
| pyramid (raster) | 0.892 | 0.397 | 18.3 | 36.5 | 50% | 8e-5 | 0 |
| box (raster)     | 0.917 | 0.814 | 0.0  | 3.1  | 0%  | 0 | 0 |
| cyl (w_len,T256) | 0.944 | 0.726 | 0.10 | 11.4 | 1%  | 1e-4 | 0 |

Soft dice RE-ESTABLISHED at the proven operating point (sphere 0.842 ≈ jul1 0.85,
pyramid 0.892 ≈ 0.89). **air_time/total_time are meaningful and nonzero**:
sphere ~21% air fraction, pyramid ~50% air fraction (steep slopes → repositioning).
**Breakage model UNCALIBRATED at default**: `fcut_max=engage_max=0` across all
shapes with `--f-ref 50 --f-max 100` → break_prob_any is near-zero noise.
broken=0 everywhere.

### Run 4 — breakage calibration (f_ref/f_max DOWN 10x) — DONE 09:54
`--f-ref 5 --f-max 10` (10x lower), w_air_time=0 (dice-only opt, isolate
calibration effect on REPORTING):

| shape | soft dice | break_prob | fcut_max | engage_max |
|-------|-----------|------------|----------|------------|
| sphere  | 0.843 (=base) | 0.0040 (was 4e-4) | 0 | 0 |
| pyramid | 0.899 (=base) | 0.0077 (was 8e-5) | 0.083 (was 0) | 0.0027 |

**Calibration confirmed**: lowering f_ref/f_max 10x makes the breakage model
FIRE — break_prob_any rises 10-100x, fcut_max becomes nonzero (pyramid).
Crucially **dice is unchanged** (calibration only affects the breakage REPORT
and the w_break loss term, which is 0 here) — so calibration is dice-neutral.
**Conclusion**: at voxel 0.5mm + 3.175mm tool, per-step engagement is genuinely
tiny → breakage is NOT a real risk at this operating point (break_prob < 1%,
broken=0 even calibrated). The model fires but says "low risk". Use `--f-ref 5
--f-max 10` for reporting so the numbers are nonzero, but breakage is not a
useful optimization target here.

### Run 5 — w_air_time=1e-3 (too weak) — DONE 09:54
w_air_time=1e-3 (default): sphere air 4.1→5.4s (UP), pyramid air 18.3→20.5s (UP).
Dice neutral (sphere 0.842, pyramid 0.902). **1e-3 is too weak to reduce air** —
noise dominates. The differentiable air-time loss (`w_at * seg_time * air_frac *
inv_n`, mean over segments) is dwarfed by the dice loss at this weight. Need a
weight sweep. Launched Run 6.

### Run 6 — w_air_time sweep {1e-2, 1e-1, 1.0} × {sphere, pyramid} — RUNNING 09:57
6 GPUs (0,4,6,7,8,9). f_ref=50 (keep comparability to baseline; breakage not the
focus). Goal: characterize the dice/air tradeoff curve for the NEW time-based
air loss vs the dead proximity-based w_air (per [[air-cut-loss-tradeoff]],
w_air traded dice 0.847→0.55). Early signal: pyr w_air_time=1.0 already shows
higher loss (0.30 vs 0.19) and worse dice (0.566 vs 0.606) at iter 37 —
aggressive weight perturbs optimization. Awaiting completion (~10:30).

### Run 7 — best_w_airtime checkpoint selection (dice-NEUTRAL air test) — RUNNING 10:12
GPUs 3,5. Pyramid (worst air: 50%, 36.5s). w_air_time=0 (IDENTICAL optimization
to baseline — only the best-checkpoint criterion changes). best_w_airtime=0.05
(default) and 0.2 vs baseline's 0. If a lower-air checkpoint exists at near-equal
dice, composite selection picks it → dice-NEUTRAL air-time win (the real
deployable win idea.md step 2 seeks). Both runs show identical iter-37 dice
(confirms only selection differs). Awaiting completion (~10:45).

### KEY DEPLOYABLE FINDING (baselines) — air/time shape-dependence
box/cyl (z-invariant cross-section, raster_fine init) produce FAST, AIR-FREE
trajectories: box 3.1s/0% air, cyl 11.4s/1% air, both high dice (0.917/0.944).
sphere/pyramid (3D surfaces) produce SLOW, AIR-HEAVY trajectories: sphere 19.6s/
21% air, pyramid 36.5s/50% air. The air-cut is NOT uniformly ~30% (the prior
[[air-cut-loss-tradeoff]] ~30% figure was an average) — it is bimodal: ~0% for
prismatic shapes, 20-50% for freeform surfaces. This means w_air_time has
headroom ONLY on sphere/pyramid; box/cyl have nothing to gain (already 0% air).

### Run 6 RESULT — w_air_time sweep: sphere is a FREE win, pyramid is a TRADEOFF — DONE 10:37
| config | dice | air(s) | total(s) | vs baseline |
|--------|------|--------|----------|-------------|
| sphere base | 0.842 | 4.13 | 19.6 | — |
| sphere wat1e-2 | 0.845 | 3.20 | 23.0 | dice≈, air -22% |
| sphere wat1e-1 | 0.845 | 5.92 | 24.1 | dice≈, air UP |
| **sphere wat1.0** | **0.844** | **1.88** | **11.7** | **dice-NEUTRAL, air -54%, total -40%** |
| pyramid base | 0.892 | 18.3 | 36.5 | — |
| pyramid wat1e-2 | 0.890 | 18.1 | 41.6 | dice≈, air≈ |
| pyramid wat1e-1 | 0.892 | 19.4 | 36.0 | dice≈, air UP |
| pyramid wat1.0 | 0.816 | 11.1 | 27.9 | air -39% BUT dice -0.076 |

**Headline:** `w_air_time=1.0` on sphere is a **dice-NEUTRAL deployable win** —
air 4.13→1.88s (-54%), total 19.6→11.7s (-40%), dice 0.842→0.844 (within seed
noise, neutral). The sphere's 21% air is "unnecessary" repositioning the loss
cleans up for free. On pyramid the same weight trades dice for air (-0.076 dice
for -39% air): the pyramid's 50% air is "structural" (steep-slope repositioning
the tool must do to cover the part). **The air-cut is removable-for-free on
sphere, structural on pyramid** — a shape-dependent answer that refines the
flat "~30% air inherent" claim in [[air-cut-loss-tradeoff]].

This is the run's main result. Confirming across seeds (Run 8) since ±0.04
variance: need sphere wat1.0 to stay ~0.84 dice / ~1.9s air on seeds 2,3.

### Run 8 — confirm sphere wat1.0 across seeds + best_w_airtime on sphere + pyramid mid-weight — RUNNING 10:40
6 GPUs (0,4,6,7,8,9). (a) sphere wat1.0 seed 2,3 (confirm dice-neutral air win);
(b) sphere best_w_airtime=0.05/0.2 only (w_air_time=0 — dice-neutral air
reduction via CHECKPOINT SELECTION, no re-optimization); (c) sphere
wat1.0+best_w_airtime=0.05 (combine); (d) pyramid w_air_time=0.3 (mid-weight,
probe for a dice-neutral air point between 1e-1 (no air cut) and 1.0 (dice
-0.076)). Awaiting completion (~11:10).

### Run 8 RESULT — CONFIRMED dice-neutral sphere air win + FREE checkpoint-selection win — DONE 11:22
| config | dice | air(s) | total(s) | vs sphere baseline (0.842/4.13s/19.6s) |
|--------|------|--------|----------|------------------------------------------|
| wat1.0 seed1 | 0.844 | 1.88 | 11.7 | dice≈, air -54%, total -40% |
| wat1.0 seed2 | 0.844 | 1.43 | 13.2 | dice≈, air -65%, total -33% |
| wat1.0 seed3 | 0.846 | 1.84 | 15.3 | dice≈, air -55%, total -22% |
| **wat1.0 mean** | **0.844±0.001** | **1.72** | **13.4** | **dice-NEUTRAL, air -58%, total -32%** |
| bwa0.05 ONLY (no re-opt) | 0.842 | 2.25 | 21.5 | dice-NEUTRAL, air -45% — FREE (selection only) |
| bwa0.2 ONLY | 0.840 | 4.47 | 21.0 | too aggressive; air UP |
| **wat1.0 + bwa0.05** | **0.844** | **0.90** | **13.3** | **dice-NEUTRAL, air -78% (BEST)** |
| wat3.0 | 0.840 | 0.52 | 18.2 | dice-NEUTRAL, air -87% (pushes air near 0) |

**CONFIRMED across 3 seeds**: sphere `w_air_time=1.0` is a dice-NEUTRAL
deployable air win (dice 0.844±0.001 vs 0.842; air 1.72s vs 4.13s, -58%).
Stronger still, `best_w_airtime=0.05` checkpoint selection ALONE (identical
optimization to baseline) picks a -45% air checkpoint at equal dice — a FREE
win (a lower-air checkpoint exists in the baseline trajectory that pure-dice
selection misses). Combined `wat1.0 + bwa0.05` reaches air 0.90s (-78%) and
`wat3.0` reaches 0.52s (-87%), both dice-neutral. **The sphere's 21% air is
entirely removable for free.**

Pyramid (structural air): wat0.3 → dice 0.865 (-0.027), air -15%; w_time0.1 →
dice 0.872 (-0.020), total -10%. Both still trade dice — pyramid's steep-slope
repositioning air is necessary, only reducible at a dice cost. bwa selection on
pyramid is marginal (-3% air, dice-neutral) — structural air is not selectable
away.

### HEADLINE (final)
1. **Breakage calibration**: default `--f-ref 50` gives fcut_max=engage_max=0
   (model doesn't fire); `--f-ref 5 --f-max 10` makes it fire (break_prob
   1e-4→0.004-0.008, fcut_max nonzero), dice-NEUTRAL. Even calibrated, break_prob
   <1% and broken=0 → breakage is NOT a binding constraint at this operating
   point (reporting axis, not optimization target).
2. **Air/time is bimodal**: box/cyl 0-1% air (fast, 3-11s); sphere/pyramid 21-50%
   air (slow, 20-36s). Headroom only on freeform shapes.
3. **Sphere air is removable for FREE**: `w_air_time=1.0` → dice-neutral -58% air
   (confirmed 3 seeds); `best_w_airtime=0.05` selection → free -45%; combined
   → -78%; `wat3.0` → -87%. Dice stays 0.844.
4. **Pyramid air is STRUCTURAL**: any air/time reduction costs dice (-0.02 to
   -0.076); steep-slope repositioning is necessary. Refines the flat "~30% air
   inherent" of [[air-cut-loss-tradeoff]] into a shape-dependent answer.

### Run 9 — hard-dice neutrality of the sphere air win (CHECK) — RUNNING 17:03
The headline "sphere air is removable for FREE" was confirmed SOFT-dice-neutral
across 3 seeds (0.844±0.001). But the DEPLOYABLE metric is HARD dice, and the
air-win runs were never checked for hard-dice neutrality. Reading metrics.json
for the existing runs:

| config (sphere, random init) | soft dice | HARD dice | air(s) | total(s) |
|------------------------------|-----------|-----------|--------|----------|
| baseline seed1 (w=0)         | 0.842     | 0.565     | 4.13   | 19.6     |
| wat1.0 seed1                 | 0.844     | 0.549     | 1.88   | 11.7     |
| wat1.0 seed2                 | 0.846     | 0.548     | 1.43   | 13.2     |
| wat1.0 seed3                 | 0.846     | 0.551     | 1.84   | 15.3     |
| wat1.0+bwa0.05 seed1         | 0.844     | 0.549     | 0.90   | 13.3     |

wat1.0 HARD dice is tight (0.548-0.551, mean 0.549) but the only baseline seed
is 0.565 — a ~0.016 hard-dice gap. Could be seed noise (only 1 baseline seed) or
a real coverage cost (less air-cut = less material removed = lower hard coverage).
Need baseline HARD dice across seeds 2,3 to settle it. Running sphere baseline
(w=0, f_ref5) seeds 2,3 on GPU 1. If baseline mean ≈0.55 → air win is fully
hard-neutral (free). If baseline mean ≈0.565 → the air win costs ~0.016 hard dice
(still a good trade: -58% air for -0.016 hard, but NOT free).

### Run 9 RESULT — sphere air win is HARD-dice-NEUTRAL (FREE confirmed) — DONE 18:00
| config (sphere, random init) | soft dice | HARD dice | air(s) | total(s) |
|------------------------------|-----------|-----------|--------|----------|
| baseline seed1 (w=0)         | 0.842     | 0.565 ⚠   | 4.13   | 19.6     |
| baseline seed2 (w=0)         | 0.848     | 0.552     | 3.58   | 20.5     |
| baseline seed3 (w=0)         | 0.846     | 0.552     | 5.36   | 18.3     |
| **baseline mean (3 seed)**   | **0.845** | **0.556** | 4.36   | 19.5     |
| **baseline mean (s2,s3)**    | 0.847     | 0.552     | 4.47   | 19.4     |
| wat1.0 mean (3 seed)         | 0.844     | 0.549     | 1.72   | 13.4     |

**VERDICT: the sphere air win is HARD-dice-NEUTRAL.** Baseline hard dice across
seeds 2,3 is 0.552 — identical to the wat1.0 air-win runs (0.549). The seed1
baseline 0.565 was a HIGH OUTLIER; the true baseline hard-dice mean is ~0.552,
and wat1.0's 0.549 sits ~0.003 below it (well within the ±0.01-0.05 run-to-run
variance from GPU atomic-add nondeterminism). So `w_air_time=1.0` removes ~58%
of air-cut time at NO cost to either soft OR hard dice — a genuinely FREE
deployable win on both metrics. The headline finding #3 stands and is now
verified on the deployable (hard) metric, not just the soft proxy.

### Run 10 — generality: sphere air-win recipe on 2in stock (2 1 1) — RUNNING 18:00
The headline air win is on the default 1in sphere. Does the recipe
(`--w-air-time 1.0 --best-w-airtime 0.05`) generalize to a DIFFERENT scenario,
or is it overfit to the 1in stock? Testing on a 2in stock (`2 1 1`, same sphere
target): bigger stock → more empty space → more air-cut headroom (a harder test).
Need a within-scenario baseline (w=0) + the recipe, same seed, on GPU 2.

EARLY SIGNAL (iter ~56): dice=0.0 for BOTH, and the baseline's gradient is dying
(4.5e-7) while the recipe's is healthy (3e-2, from the air-time loss). In the 2in
stock the centered target is farther from the random init → dice loss is 0/flat →
no gradient to pull the tool to the target. The recipe gets gradient from the
air-time loss (penalizes non-cutting) which may pull it toward the target. If the
baseline stays stuck at dice 0 but the recipe climbs, that's a generality finding
(the air loss helps initialization in larger stocks, not just trims air). If both
climb and the recipe gives dice-neutral air reduction vs the 2in baseline, the
win generalizes. If both fail, random init doesn't reach the target in 2in stock
(a method limit, not an air-loss limit). Awaiting completion (~19:00).

### Run 10 RESULT — 2in stock: air win does NOT generalize (init/stock-size cap) — DONE 18:25
Killed early once the plateau was clear (loss flat, no further progress). Both
runs killed mid-training; dice values below are the plateau (soft dice stayed 0).

| config (2in stock 2 1 1, sphere, random init) | soft dice | hard dice | grad | verdict |
|------------------------------------------------|-----------|-----------|------|---------|
| baseline (w=0) seed1 — killed iter ~229        | 0.000     | 0.157     | 4e-7 | STUCK: random init never reaches centered target; dice loss flat 0 → no gradient |
| recipe (wat1.0+bwa0.05) seed1 — killed iter ~827 | 0.000   | 0.308     | 2.7e-2 | PLATEAU: air loss starts carving (hard 0→0.31) then air-min stalls further cutting |
| recipe (wat1.0+bwa0.05) seed2 — killed iter ~649 | 0.000   | 0.308     | 3.0e-2 | PLATEAU: same, hard 0.31 (confirms seed1) |

**VERDICT: the sphere air win does NOT generalize to the 2in stock.** The failure
is a stock-size/init cap, NOT an air-loss limit: when the target is centered in a
stock much larger than the target (2in stock vs 0.9in sphere), the random init
(`init_scale 0.05`) places the toolpath in empty space far from the target → dice
loss is 0/flat → no gradient to pull the tool in (baseline stuck, grad ~4e-7).
The air-time loss partially rescues it (provides a "cut more" gradient → hard dice
0→0.31) but then air-minimization dominates and the tool settles into a low-air,
low-cut local minimum (hard dice plateaus at 0.31, soft dice never leaves 0). So
the air win is **conditional on the init reaching the target** — which holds for
the 1in stock (target nearly fills the stock) but not the 2in. **Generality
boundary**: the headline "sphere air removable for free" applies when the stock ≈
target bounding box; for larger stock the method needs an init that reaches the
target (e.g. raster_fine / shape-aware coverage init, per [[train-csg-best-config]]
jul4 GradMill), not the air-time loss. The air-time loss is an air *trimmer*, not
an init fix.
