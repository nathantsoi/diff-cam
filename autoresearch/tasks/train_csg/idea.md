# idea.md — jul10-gouge

Branch: `ar-agd/jul10-gouge` (from `autoresearch`).
Run folder: `runs/jul10-gouge/`.
Hardware: 8× Quadro RTX 6000 (24 GB each), all idle at start.

## Starting point

Establish a baseline on the default scenario (1 in cube stock, sphere target
r=11.43 mm, `--voxel-size-mm 0.5`) with NO method changes (defaults: init_mode
random, dt=0.45, grad_clip=0.5, eval_freq=10). All later method variations are
compared against this baseline's `hard_dice`. The method MUST stay shape-agnostic
(no branching on shape name in optimizer/init/loss).

## Prior context (archived memory, used only as guardrails not as claims)

The jul8-multidepth run established `multidepth` init + k-anneal (k 2→70) +
loss_shift 0.7 as the method. Deployed hard_dice (best config, seed=1, 1 rep):
sphere ~0.78, pyramid ~0.70, bowl ~0.56, hole ~0.245. Key lessons carried over
as METHODOLOGY (not bias):
- **Variance floor**: hard_dice is nondeterministic at seed=1 (GPU atomics).
  pyramid ±0.017, hole ±0.006. A delta is real only if > ~2σ. Use ≥3 repeats
  for any keep decision on a single shape.
- **Gouge is the deployability blocker**: `--best-on-hard` chases hard dice blind
  to gouge. Sphere gouge (~572-620) is union-over-erosion at TANGENT-capsule
  pass seams — `w_tool_gouge` is INACTIVE there (tool tangent at sampled
  midpoints) so it cannot reach sphere gouge. Pyramid gouge IS tool-penetration
  and `w_tool_gouge=8` fixed it (509→75). Bowl gouge (962) is the worst.
- **tool_gouge_margin** (wave 19, never recorded): inflate barrier radius
  r_tool → r_tool+margin so the union of capsules stays tangent-only at seams.
  Smoke showed it activates the barrier (sph tg8 m1mm: loss_tool_gouge 0→1.7).
  This is the prime untested lever for sphere/bowl gouge.

## Goal / hypothesis

The deployable metric is **hard_dice**; the user goal is highest hard_dice WITHOUT
breaking the tool, in the shortest trajectory. The wall the prior run hit is the
**gouge/dice tradeoff**: pushing hard_dice up (via best_on_hard + sharp k) deploys
gouging checkpoints. The untested `tool_gouge_margin` is designed to break that
tradeoff by lifting the tool off the surface so overlapping capsules don't bite.

**Hypothesis**: a positive `tool_gouge_margin` (shape-agnostic, target_sdf only)
lets us keep `--best-on-hard` (capture the high hard dice) while suppressing the
union-over-erosion gouge on sphere/bowl — trading a little uncut residual for a
gouge-free, actually-deployable part. If margin alone under-cuts (residual up,
dice down), combine with a sharper k_final or a residual re-weight.

## Plan

1. **Baseline** (GPU 0): default scenario, no changes. Record hard_dice ref + the
   default soft/hard gap + gouge. (REQUIRED first run.)
2. **SOTA reference** (GPU 1, parallel): re-establish the prior method
   (`multidepth` + k-anneal k2→70 + loss_shift 0.7 + best_on_hard) on sphere as
   my reference point for hard_dice/gouge, since I will not trust archived
   numbers as-is.
3. **Margin dose-response** (GPUs 2-7, parallel): sphere, SOTA config + best_on_hard
   + w_tool_gouge 8 + tool_gouge_margin {0,0.5,1,2,4,8} mm. Find the knee where
   gouge collapses without killing dice. ≥3 reps on the winner.
4. Generalize the winner to pyramid/bowl/hole/cylinder/box; verify shape-agnostic.
5. Keep wins (higher hard_dice at fixed-or-lower gouge), discard regressions,
   advance branch. Report dice_improvement for cross-scenario comparison.

## Notes (updated as results come in)

(none yet)
