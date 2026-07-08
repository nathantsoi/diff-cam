# idea.md — jul7-balanced

Branch: `ar-agd/jul7-balanced` (from `autoresearch`).
Run folder: `runs/jul7-balanced/`.

## Objective (this run)

Maximize the **deployable composite**: highest `hard_dice` (the sharp boolean
carve — the metric we advance on) **without breaking the tool** and in the
**shortest trajectory / execution time** possible. Secondary guides:
`air_time`, `total_time`, `break_prob_any`. The method must generalize across
stock/target shapes/sizes — the optimizer/init/loss must NOT branch on shape
name, only on the baked target SDF grid.

## Starting point

Establish a baseline on the default scenario (1 in cube stock, sphere target,
`--voxel-size-mm 0.5`, `--target-radius-mm 11.43`, defaults otherwise) before
varying anything. All later scenario/method variations are compared against
this baseline.

## Key code facts (from train_csg.py read)

- **Best-checkpoint selection uses SOFT dice**, not hard dice:
  `composite_score({**m, "dice": m["soft_dice"]}, ...)` (line ~1139, ~1276).
  The reported headline is `hard_dice`. So the checkpoint we deploy is chosen
  by soft dice + small traj-quality penalties — a potential mismatch where the
  best-soft-dice iter is not the best-hard-dice iter. Worth probing: select on
  HARD dice and see if `hard_dice` improves (it's the deployable metric).
- New traj-quality soft terms default ON tiny: `w_time=w_air_time=w_break=1e-3`.
- `best_w_airtime=best_w_time=best_w_break=0.05` (composite checkpoint weights).
- `dice_improvement = (dice - baseline)/(1 - baseline)` is the cross-scenario
  ranking axis; `hard_dice` is the absolute gate.

## Plan

1. Baseline (defaults, default scenario) — establish hard_dice / time / break.
2. Probe: best-checkpoint selection on HARD dice instead of soft.
3. Tune the composite objective weights (best_w_*, w_time/w_air_time/w_break)
   to shorten time / cut breakage at equal-or-better hard_dice.
4. Test generalization across shapes (cylinder/box/pyramid/sphere_hole/
   sphere_bowl) and stock sizes; use dice_improvement to compare fairly.
5. Keep wins, discard regressions, advance the branch on hard_dice.

## Notes

(filled in as the run progresses)

## Findings

(filled in as the run progresses)
