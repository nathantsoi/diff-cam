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
