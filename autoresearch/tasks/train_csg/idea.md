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

### Wave 1 (sphere, 8-GPU batch) — IN PROGRESS

**Confirmed references (final metrics.json):**
- **Baseline** (random init, k=10 const, defaults, best_on_hard=OFF): hard_dice
  **0.599**, soft 0.832 → soft/hard gap **0.23** (soft massively inflated;
  soft-dice selector deployed a near-air checkpoint). Deployed gouge **0**,
  air_frac **0.884** (88% air — barely cuts). dice_improvement 0.113. This is
  the floor and the proof that soft dice masks failure.
- **m1 (SOTA+tg8+margin1, seed1)**: hard_dice **0.725**, soft 0.743 (gap 0.018 —
  best_on_hard working), deployed gouge **94.6**, air_frac 0.340, break 0.0001,
  dice_imp 0.391. vs prior SOTA sphere gouge ~572-620 → **~6× less gouge at
  0.725 dice**. The tradeoff-breaker works.

**KEY DISTINCTION**: the log's `gouge:` column is the SOFT loss_gouge (training
diagnostic, ~0.01). The DEPLOYED gouge is `metrics.json["gouge"]` (hard carve,
dx³-scaled volume). Use the latter. b5 log showed gouge 0.014 but deployed 94.6.

**Dose-response (current-iter, mid-training):**
- margin {0, 0.5, 2, 4, 8} with tg8: ALL PINNED at hdice 0.5480 / resid 0.4462 /
  grad 1e-4 (dead, k sharpened 11→36 with zero movement — killed). Only m1
  escaped.
- m1 repeats (seed 2,3,4): c2/c3 carving strong (0.64-0.66), c1 stalled (~0.56).
  So m1 escape is partly reproducible but high-variance (bimodal succeed/stall).
- tg4 m1, tg2 m1 (softer barriers): BOTH pinned. Softer tg does NOT fix pinning.

**Insight**: w_tool_gouge is shape-INappropriate for sphere tangent-seam gouge —
at correct tangent passes the barrier is spuriously active and pushes the tool
off-surface, stalling the carve. The stall is also partly inherent to
multidepth+k-anneal (b2 no-tg stalls too late). Pinning is stochastic, not a
clean function of tg weight or margin.

**Open**: b2 (SOTA no-tg) final + m1×3 finals needed to quantify variance and
confirm the gouge reduction. Considering a w_tool_gouge WARMUP (0→full over
early iters, mirroring w_prox warmup) to let carving establish before the
barrier engages — principled, shape-agnostic fix for the pinning.

### m1 (SOTA+tg8+margin1, k_init2, no-warmup) — CONFIRMED WIN over SOTA

| seed | hard_dice | soft | gouge | air_frac | loss_tg | dice_imp |
|------|-----------|------|-------|----------|---------|----------|
| 1 | 0.725 | 0.743 | 94.6 | 0.340 | 0.916 | 0.391 |
| 2 | 0.622 | 0.625 | 104.5 | 0.927 | 0.000 | 0.163 |
| 3 | 0.663 | 0.681 | 115.8 | 0.827 | 0.000 | 0.254 |
| 4 | 0.707 | 0.754 | 26.5 | 0.267 | 0.916 | 0.351 |

Mean hard_dice **0.679** (min 0.622, max 0.725), gouge 26-116. vs SOTA 0.624/135
→ **higher dice AND lower gouge**; worst m1 seed (0.622) ≈ SOTA dice at lower
gouge. m1 is the new SOTA on sphere.

**KEY**: best_on_hard captures the hard-dice PEAK during training even if the
current-iter later stalls — c3 (s4) log showed a 0.548 stall at iter 4339 yet
finished 0.707. Do NOT read current-iter hdice as the result; use metrics.json.
The "stalls" are mostly current-iter transients, not final failures.

Two checkpoint flavors: barrier-shaped (loss_tg=0.9, low air 0.27-0.34, s1/s4)
vs pre-barrier high-air (loss_tg=0, air 0.83-0.93, s2/s3). Both ≥0.62 dice.

### In-flight: warmup (e) + k_init sweep (d)
- e2 (warmup0.3, s2) at iter 2056: hdice 0.711 — vs c1 (no-warmup s2) 0.622.
  Warmup may lift the medium seeds. Watching finals.
- d3 (k_init10) STALLED — higher k_init does NOT prevent stalls (stall is
  stochastic, not k-driven). k_init sweep likely not the lever.
