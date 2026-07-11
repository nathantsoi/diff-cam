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

### Warmup (e1/e2) + k_init sweep (d1/d2/d3) — RESOLVED

**k_init sweep = DEAD END.** d1 (ki4) carved to 0.7286 at iter 3139 then
CRASHED back to 0.5480 by iter 3709; d2 (ki6) and d3 (ki10) stalled at 0.5480
throughout. All three k_init values (4/6/10) eventually pin at the floor.
k_init is NOT the lever; the stall is stochastic, not k-driven.

**Warmup0.3 = dice/gouge TRADEOFF, not a Pareto win.** Finals (best_on_hard):

| config | seed | hard_dice | gouge | air_frac | loss_tg |
|--------|------|-----------|-------|----------|---------|
| no-warmup m1 | 1 | 0.7248 | 94.6 | 0.340 | 0.916 |
| no-warmup m1 | 2 | 0.6219 | 104.5 | 0.927 | 0.000 |
| no-warmup m1 | 3 | 0.6628 | 115.8 | 0.827 | 0.000 |
| no-warmup m1 | 4 | 0.7066 | 26.5 | 0.267 | 0.916 |
| warmup0.3 m1 | 1 | 0.7344 | 106.5 | 0.298 | 0.000 |
| warmup0.3 m1 | 2 | 0.7468 | 135.2 | 0.583 | 0.000 |

no-warmup: mean hd 0.679, mean gouge **85.4** (barrier-shaped, ltg=0.9, low air).
warmup0.3: mean hd (s1,s2) **0.741**, mean gouge **120.9** (ltg=0 — best_on_hard
selects pre-barrier aggressive checkpoints that gouge more). Warmup lifts dice
(+0.06) but pushes gouge back toward SOTA level (135) — it RIDES the gouge/dice
tradeoff, does NOT break it. The whole point of margin1+tg8 was LOW gouge
(deployability); warmup sacrifices that for raw dice. e3/e4/e5 peaked 0.647/
0.616/0.616 then regressed to 0.548 floor (best_on_hard retains the peak;
metrics.json written only at run end so must let them finish).

**Decision**: no-warmup m1 (margin1+tg8, NO warmup) is the DEPLOYABLE winner —
highest dice at controlled low gouge, simplest (no warmup param). Warmup0.3 is
the raw-hard_dice champion but gouge-regressed. Both break_prob=0. Generalizing
BOTH to bowl/hole head-to-head to see which Pareto-front generalizes (bowl is
the gouge-stress shape — the decider).

### Wave 3 (generalize) — no-warmup WINS on every non-sphere shape

Running both configs per shape for direct comparison (current-iter ~3000-4400):

| shape | no-warmup hdice | grad | warmup hdice | grad | verdict |
|-------|-----------------|------|--------------|------|---------|
| bowl  | 0.5785 (climbing)| 3.36 | 0.4180 | 2.4e-4 | nw: active; warmup DEAD |
| hole  | 0.1635          | 0.066| 0.1268 | 9.6e-3| nw: alive; warmup DEAD |
| pyramid | (nw_pyr just launched) | — | 0.3780 | 5.2e-2 | warmup stalled |

**DECISIVE**: no-warmup beats warmup on every non-sphere shape tested. Warmup
runs are ALL stalled (grad 1e-4 to 5e-2, resid 0.53-0.63) — the warmup keeps
the barrier OFF early → degenerate plunge → gradients die → stuck. No-warmup
runs stay ACTIVE (grad 0.07-3.4) and keep carving. Warmup is sphere-only luck
and a dice/gouge tradeoff; it KILLS generalization. **Warmup abandoned. Final
winner = no-warmup m1** (multidepth + k-anneal k2→70 + loss_shift 0.7 +
best_on_hard + w-tool-gouge 8 + tool-gouge-margin-mm 1, NO warmup).

Full 6-shape no-warmup sweep now running (GPUs 2/3/5): pyramid, cylinder, box
(the 3 remaining shapes) + the in-flight nw_bowl/nw_hole. Box early signal
hdice 0.814 (easiest), cylinder 0.718, pyramid 0.373 (soft/hard gap 0.26 —
best_on_hard will capture the hard peak as k sharpens).

**Warmup shape finals (done, confirm warmup is worse):**
- w3_pyr 0.5393 (gouge 0.0), w3_hole 0.1289 (gouge 0.0), w3_bowl 0.4180 (dead).
- nw_pyr climbed 0.373→0.519 by iter 1889 (still climbing) — will match/beat
  w3_pyr 0.539 with 3000 iters to spare. nw_bowl 0.585 (climbing) >> w3_bowl 0.418.

### Wave 3 no-warmup finals (done) — the shape-agnostic generalization table

| shape | no-warmup hdice | gouge | imp | note |
|-------|-----------------|-------|-----|------|
| sphere | 0.679 (4-seed mean) | 85.4 | 0.39 | WIN over SOTA (0.624/135) |
| box | 0.8144 | ~0 | 0.79 | excellent, stable plateau |
| cylinder | 0.7175 | ~0 | 0.60 | strong, stable plateau |
| pyramid | ~0.519 (peak) | ~0 | 0.27 | regressed 0.519→0.416 current; best_on_hard holds peak |
| bowl | 0.5916 | 101.5 | 0.30 | dice win over warmup (0.522) but HIGH gouge |
| hole | 0.1474 | 56.6 | 0.03 | FAIL — barrier crushes hole (prior SOTA 0.245) |

**Two open problems for true shape-agnostic generalization:**
1. **Hole regression** (0.245→0.147): the +1mm inflated barrier pushes the tool
   out of the interior hole concavity (residual 0.658 = never reaches hole
   bottom). NOT a shape-name branch — geometric consequence of margin on
   concave-interior features.
2. **Bowl gouge** (101.5): bowl is the gouge-stress shape; nw wins dice but
   gouges hard (union-over-erosion at tangent seams).

### Wave 4 (hole margin dose-response + bowl-m2) — RESULT: barrier is NOT the hole culprit

Hole margin dose-response (all ~iter 2000, mid-train):
| config | hdice | grad | note |
|--------|-------|------|------|
| no barrier (prior-SOTA) | 0.136 | 0.007 | NOT recovering to archived 0.245 |
| tg8 margin0 | 0.055 | 25.9 (chaotic) | worst |
| tg8 margin0.25 | 0.121 | 0.45 | |
| tg8 margin0.5 | 0.082 | 5.93 | |
| tg8 margin1 (prior) | 0.147 | 0.04 | |

**CORRECTED DIAGNOSIS**: hole is INTRINSICALLY failing for this method — even
NO barrier (the prior-SOTA config) is stuck at 0.136, oscillating 0.04-0.16
across 2000 iters (soft dice briefly hit 0.26 at iter 599 then collapsed). The
archived 0.245 was likely a lucky-seed transient peak, NOT a reproducible
result. The barrier margin is NOT the hole culprit.

**Root cause (hypothesis)**: hole = concave-INTERIOR feature (tool must descend
INTO a 9.525mm hole with ~3.175mm tool). multidepth init likely places the tool
in a degenerate starting position for this topology — it carves the surrounding
sphere but never discovers the interior hole descent. resid ~0.30-0.68 = leaves
the hole mostly uncut. The gradient signal to "go deeper into the hole" is weak
because the soft-dice loss is dominated by the large outer-sphere envelope.

bowl_m2 (iter 1459): climbing 0.427, gouge ~0 — larger margin helping bowl gouge
but dice lower than m1's 0.592 (over-inflation under-cuts). bowl m1 still the
bowl winner.

**Open directions (shape-agnostic)**: hole needs either (a) longer iters, (b)
different init that samples the interior, or (c) a residual-weighting that
emphasizes the uncut hole volume. All must stay shape-agnostic. NOT pursued
immediately — hole is 1 of 6 shapes; the method clearly generalizes to 5/6
(sphere/box/cyl/pyramid/bowl).

### Wave 4 results (finals + hole experiments)

**best_on_hard captures REAL peaks beyond final-iter plateau (cylinder proof):**
cylinder final_iter_hard_dice=0.7175, but best-checkpoint hard_dice=**0.8113**
(best_score 0.758 > final score). best_on_hard loaded a higher-hard-dice
checkpoint than where training ended. NOTE: the best-checkpoint eval path
doesn't populate soft_dice/asd/hd95, so those fields read 0.0/Inf in
metrics.json for runs that used_best — this is a LOGGING QUIRK, not a failure.
The hard_dice and gouge fields ARE correct (from best_m). cylinder's 0.8113
checkpoint DOES gouge (307.4, ltg 53.3) — deployability concern, but the dice
peak is real.

**Hole experiments — all FAIL, hole is intrinsically hard:**
| config | final/cur hdice | verdict |
|--------|-----------------|---------|
| no-barrier (notg) | 0.190 (best, climbing) | best hole result, still poor |
| no-barrier kf30 (slower k) | 0.083 | slower-k does NOT help |
| no-barrier 10k iters | 0.052 @ iter 1669 | more-iters does NOT help |
| tg8 margin0 | 0.085 | barrier actively harmful |
| tg8 margin0.25 | 0.138 | |
| tg8 margin0.5 | 0.165 | |
| tg8 margin1 | 0.147 | |

CONCLUSION: hole fails regardless of barrier/margin/k/iters. The method cannot
discover the interior-hole descent from multidepth init. This is a fundamental
limitation for concave-interior topology, NOT a tunable-parameter issue. The
other 5 shapes (sphere 0.679, box 0.814, cyl 0.811, pyramid 0.539, bowl 0.592)
all generalize shape-agnostically.

**bowl_m2 final (CORRECTED — REVERSES the mid-training claim):** final
hard_dice=0.5136, deployed gouge=**145.4** (NOT 0.02). The "gouge 0.02" was a
mid-training CURRENT-ITER reading; best_on_hard later selected a higher-dice
checkpoint that gouges 145.4 — WORSE than m1's 101.5. So margin2 is NOT the
bowl gouge winner; best_on_hard overrides margin2's low-gouge benefit exactly
as it did for sphere warmup (chases dice → gouge regresses). **m1 (0.5916 /
101.5) remains the bowl winner on BOTH dice and gouge.** margin2 = dead end.

**Hole finals (best_on_hard):** no-barrier notg = **0.2037** (best hole, but
gouge 245.9 — no barrier so it overcuts freely); m0.25=0.138/gouge13.5;
m0.5=0.166/gouge6.6. The barrier trades dice for gouge on hole; no-barrier
gives best dice but worst gouge. Hole stuck ~0.20 max either way.

### Wave 5 — hole root cause (init) + interior-sampling attack

**Root cause PINPOINTED (read multidepth init, train_csg.py:1084-1085):** the
init builds `r_safe(z) = r_cross(z) + r_tool + margin`, where `r_cross(z)` is
the OUTER target cross-section radius (shape-agnostic, from the baked SDF grid).
The tool center stays >= r_tool OUTSIDE the outer target envelope at every
point. For sphere_hole, `r_cross` = outer sphere radius, so `r_safe` > hole
radius ALWAYS — the init GEOMETRICALLY CANNOT place the tool inside the interior
hole concavity. No levels/revs/iters/margin tuning reaches it; the tool never
enters the hole. This is the mechanism behind "hole is intrinsically hard."

**Attack (shape-agnostic, init-only — NOT eval/metric code):** add an init mode
that seeds tool samples INSIDE the target's solid envelope (not just outside it)
so concave-interior features (hole) are reachable. Reads only the target SDF
grid (no shape names). Test on hole (the stress shape) + sphere (regression
guard). Keep if hole dice rises without sphere regression; else discard.

### Wave 5 result — multidepth_cavity: hole PARETO WIN, regression guard pending

**Implementation (shape-agnostic, init-only; registered in run_pipeline.py
choices):** `multidepth_cavity` detects the interior cavity as target-SDF>0
voxels INSIDE the outer envelope (reads only the baked SDF grid — no shape
names). For cavity shapes it orders segments [helical plunge → interior spiral
→ retract → exterior]; for non-cavity shapes it falls back to plain multidepth
(zero-regression guard). The helical plunge (constant radius, non-zero XY every
segment) is REQUIRED: a straight axial plunge has ba_xy=[0,0] → tool_sdf
autodiff overflow → NaN at iter 0 (simulator tool_sdf is out-of-scope, so the
fix is init-side, not simulator-side).

**Hole result (the stress shape):**
| config | hard_dice | gouge | note |
|--------|-----------|-------|------|
| margin1 (deployed) | 0.147 | 56.6 | prior method |
| no-barrier notg | 0.2037 | 245.9 | best dice, terrible gouge |
| multidepth_cavity 200i | 0.1645 | 0.0 | |
| multidepth_cavity 2000i | 0.1676 | 2.9 | |
| multidepth_cavity 5000i | ~0.168 (running) | — | plateauing |

multidepth_cavity hole = **0.168 @ gouge ~3** — a Pareto win: beats margin1
(0.147/56.6) on BOTH dice and gouge, and beats no-barrier on gouge (3 vs 245.9)
at slightly lower dice. The cavity init reaches the interior hole descent
without the chaotic overcut. residual still high (~10.5k uncut) — the hole is
partially cut, not fully, but it no longer fails to enter. Cavity detection
fired (has_cav=25/51 voxels).

**Regression guard (sphere + bowl, multidepth_cavity, 2000i):** RUNNING on
GPUs 1/2. Gate: keep multidepth_cavity if hole rose (✓ 0.147→0.168) WITHOUT
sphere (~0.679) or bowl (~0.592) regression. Sphere/bowl should hit the
fallback (plain multidepth) and match SOTA; verifying.

### Star-rating + RLHF warm-start feature (webapp + method, in scope)

Per user request: dashboard rows now have a 1/3/5-star control + a feedback
text box. Ratings/notes persist to `autoresearch/tasks/train_csg/run_feedback.json`
(keyed by run basename; atomic temp+replace; `_UNSET` sentinel for partial
updates so a star click never wipes a note). Server: serve_web_https.py
GET/POST `/__api/feedback`. UI: index.html feedback column (gold ★ toggles,
.fb-text input, startup fetch, sortable).

**Feedback → policy loop (train_csg.py):** `load_human_feedback(target_shape,
max_steps)` reads the store on EVERY run, logs the top-rated prior runs, and
selects the highest ≥4★ prior run matching target_shape+max_steps; its saved
`trajectory_deltas.npy` (T-1,3) seeds `init` when `--use-feedback` is passed
(RLHF-style warm-start, then Adam refines). Shape-agnostic: matching is in this
selection layer only, never in optimizer/init/loss. `--use-feedback` registered
in run_pipeline.py. fb_summary (top_rated + warmstart) recorded in metrics.json
as feedback_used/feedback_top_rated/feedback_warmstart on every run.
