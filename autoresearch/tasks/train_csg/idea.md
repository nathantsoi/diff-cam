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

- **Best-checkpoint selection used SOFT dice**, not hard dice (now a flag):
  `composite_score({**m, "dice": m["soft_dice"]}, ...)` (line ~1139, ~1276).
  The reported headline is `hard_dice`. Added `--best-metric` (soft|hard) so
  the deployed checkpoint can be selected on the deployable hard dice.
- New traj-quality soft terms default ON tiny: `w_time=w_air_time=w_break=1e-3`.
- `best_w_airtime=best_w_time=best_w_break=0.05` (composite checkpoint weights).
- `dice_improvement = (dice - baseline)/(1 - baseline)` in the summary is
  computed on HARD dice; `hard_dice` is the absolute gate.
- `loss_shift` de-biases the soft loss toward the hard carve (over-erosion
  ~log(2)/k ≈ 0.069 at k=10).

## Plan

1. Baseline (defaults, default scenario) — DONE: hard_dice=0.580.
2. Probe: best-checkpoint selection on HARD dice — DONE: hard_dice=0.634 WIN.
3. loss-shift=0.1 (isolate, soft-select) — running (exp3).
4. best-metric=hard + loss-shift=0.1 (combined) — running (exp4).
5. Sweep loss-shift (0.05/0.15/0.2); try k-anneal; raise w_break / best_w_break
   to reject risky paths if break_prob climbs.
6. Test generalization across shapes (cylinder/box/pyramid/sphere_hole/
   sphere_bowl) and stock sizes; use dice_improvement to compare fairly.
7. Keep wins, discard regressions, advance the branch on hard_dice.

## Notes

- Baseline (1ef9f58, GPU1, 5011s): hard_dice=0.580, soft=0.826, impr=0.072,
  total_time=air_time=11.29s, break=0.0006, fcut_max=0, broken=0.
- **Soft-dice checkpoint masks failure**: deployed (best-soft-dice) trajectory
  has air_time==total_time and fcut_max≈0 on the HARD carve — almost entirely
  off-grid rapid moves, ~no boolean material removed. hard_dice 0.580 barely
  above do-nothing 0.548. 0.826 soft dice is sigmoid-blur inflation. Motivates
  best-metric=hard.
- Exp#2 (6d09c1c, --best-metric hard): hard_dice=0.634, soft=0.593, impr=0.189,
  gouge≈0, total_time=10.55s, break=0.0086. WIN +0.054. Kept. (clean part)
- Exp#3 (loss-shift=0.1, soft): hard_dice=0.619 but gouge=0.375 → discard.
- Exp#4 (hard+loss-shift=0.1): CRASH (my dir-move mid-run); mid-run 0.622, not
  promising. loss-shift doesn't help with hard-select.
- Exp#5 (hard+k-anneal k2->10): KILLED — k_init=2 over-erodes, soft dice stuck
  0, dead gradients. k-anneal too aggressive at low k. discard.
- Exp#6 (raster_fine_wide+hard): hard_dice=0.632, gouge=181 → discard.
- Exp#7 (zlayer+hard): hard_dice=0.665 (>0.634) BUT gouge=897 (~14% of target
  destroyed). hard-dice peak at a gouged state — composite ignored gouge.
  Kept on the number but NON-DEPLOYABLE (part is scrap). Reveals: hard-dice
  selection alone can deploy a part-damaging checkpoint.
- **Fix (commit 114565c): gouge-aware composite** --best-w-gouge penalizes
  normalized gouge (gouge/target_volume) so the deployed checkpoint rejects
  part-damaging states on equal dice. Testing (exp#8 zlayer, exp#9 random).
- dice_improvement in the summary is computed on HARD dice.
- peak_vram_mb reports 0.0 (Taichi allocates outside torch) — record 0.0.
- fcut_max rounding to 0.000000 is normal (sub-Newton forces), not a bug.
- Shared GPUs: ~1.5 it/s → 5000 iters ≈ 50 min/run. 2 concurrent (GPU1+GPU2).
- **DO NOT move a run dir while its process is alive** (exp#4 crash cause);
  rely on each launch command's own post-exit mv.

## Findings

1. **Soft-dice checkpoint selection masks failure** (baseline 0.580 barely
   beats do-nothing 0.548; 0.826 soft dice is inflation). Selecting on HARD
   dice is necessary. See [[soft-dice-checkpoint-masks-failure]].
2. **Hard-dice-only selection picks GOUGED checkpoints.** exp#2 (random+hard)
   0.634 had gouge=738 (~12% of target destroyed); exp#7 (zlayer+hard) 0.665
   had gouge=897 (~14%). The high hard_dice was gouge-gaming — a gouged part is
   unrecoverable scrap. hard_dice alone is NOT deployable.
3. **Gouge-aware composite is the real win** (commit 114565c, `--best-w-gouge`
   penalizes gouge/target_volume in checkpoint selection). exp#9
   (random+hard+`--best-w-gouge 1.0`): hard_dice **0.660**, gouge **90 (1.4%)**
   — dominates exp#2 on BOTH dice (0.660>0.634) and gouge (90<<738). NEW BEST.
   exp#8 (zlayer+gougeaware) 0.611 gouge 0.0 — clean but lower dice + slower.
4. loss-shift=0.1 over-carves (gouge), no gain over hard-select.
5. k-anneal k_init=2 is broken (over-erosion, dead gradients).
6. Structured inits (zlayer, raster_fine_wide) without gouge-awareness gouge
   heavily; zlayer clean (0.611) still loses to random clean-ish (0.660).
7. **Run-to-run variance is ~±0.035, NOT ±0.02.** expE (seed 2, EXACT best
   config) reproduced at hard_dice **0.626** vs seed 1's 0.660 — a 0.034 drop.
   The 0.660 "best" is the high tail of a ~0.64-mean distribution, NOT a robust
   advance. **A single run cannot detect a ±0.02 dice difference.** Need an
   effect >~0.04 on one run, or multi-seed averaging, to claim a real advance.
   See [[run-variance-larger-than-expected]].
8. **Single-flag levers all land in the same ~0.63-0.66 noise band** (none beat
   0.66 robustly): lr-decay 0.5 → 0.654 (BUT total_time 7.0s = half of 0.660's
   15.1s, gouge 28.5 cleaner — a real secondary-axis win at equal dice);
   w-gouge 2.0 → 0.643 (gouge 7.6 — aggressive optimizer barely gouges; the
   gouge-gaming was from checkpoint SELECTION, not the optimizer); max_steps
   160 → 0.635 (longer trajectory harder to optimize, no coverage gain);
   init-scale 0.1 → 0.636 (broader init carves a lot — soft 0.81 — but
   imprecisely); lr 1e-2 + decay → 0.620 (higher LR HURTS vs 5e-3+decay).
9. lr-decay-frac 0.5 with default lr=5e-3 is the best explore-then-settle
   variant (expB 0.654, clean, 2x faster). Higher LR (1e-2) + decay is worse.

## Plan (updated)

- DONE: baseline; best-metric=hard; gouge-aware composite; best_w_gouge=3.0;
  w_gouge=8.0; max_steps 160; lr-decay 0.5; w_gouge 2.0; lr 1e-2+decay;
  init-scale 0.1; seed-2 reproducibility; iters 8000; restart-from-state;
  k-anneal k_init=5 (broken, like k=2).
- KEY: 0.660 best is noise-tail; real level ~0.64. Stop chasing 0.660; lift the
  MEAN or take secondary-axis wins (lr-decay halves time at equal dice).
- **CANDIDATE ADVANCE — k-anneal sharpen-late (k_init=10→k_final=14): expJ
  hard_dice 0.674, dice_improvement 0.278 (HIGHEST yet vs ~0.21 base), gouge 102,
  time 12.3s.** Blurry→sharp curriculum done RIGHT (start at safe k=10, sharpen
  late — NOT k_init=2/5 which collapses soft dice). First run to clear the 0.66
  noise band. NEEDS seed confirmation (expL seed2) — single run could be tail.
- RUNNING: expL sharpen-late seed2 (confirm not noise-tail); expM sharpen
  k_init=10→k_final=16 (does lift scale with more sharpening?); expK cylinder
  (cross-shape generality of winning config — already ~0.66 mid-run).
- Next: if expL confirms ~0.67, advance the branch on k-anneal sharpen-late;
  pick best k_final from expM. Then sweep cross-shape (box/pyramid/sphere_hole/
  sphere_bowl) with the winning config; use dice_improvement to compare fairly.
  Multi-seed a promising config (3 seeds) to confirm before declaring real.
