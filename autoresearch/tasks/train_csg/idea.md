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
