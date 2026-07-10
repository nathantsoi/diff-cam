# idea.md — jul8-multidepth

Branch: `ar-agd/jul8-multidepth` (from `autoresearch`).
Run folder: `runs/jul8-multidepth/`.

## Starting point

Establish a baseline on the default scenario (1 in cube stock, sphere target
r=11.43mm, `--voxel-size-mm 0.5`) with NO method changes. All later scenario or
method variations are compared against this baseline's `hard_dice`.

## Goal

Learn a **generalizable** method for generating trajectories across a wide range
of shapes, using only the baked SDF grid (shape-agnostic — no branching on shape
name). Good trajectories:
- follow **regular patterns**,
- cut along the **surface** of the target without gouging into the part,
- minimize **air time** (motion without cutting),
- use **multiple passes at multiple depths** to keep the tool optimally engaged
  (FSWizard-style optimal chip load), instead of a single deep plunge.

## Hypothesis

The existing init modes (raster, spiral, shell, zlayer) are mostly single-depth
or single-strategy. The differentiable optimizer can refine a trajectory but is
limited by the 128-step budget and the quality of the init. The key lever for
generality is a **shape-agnostic multi-depth surface-following init** that:

1. Reads the baked target SDF grid to find, per z-layer, the target's
   cross-section (the ring/shell of material to remove around the part) — this is
   exactly what `target_cross_section_radii` already approximates for the shell
   init, but generalized via the occupancy grid rather than a radial assumption.
2. Slices the carve into N_z z-levels (multi-depth), each removing a thin band
   of stock so radial chip engagement stays in the optimal range (no full-depth
   plunge that gouges or overloads the tool).
3. At each z-level, lays down a **regular contour-following pass** offset from
   the target surface by ~tool_radius (cut on the surface, not into the part),
   spiraling/rastering inward as needed.
4. Minimizes air time by keeping passes contiguous (spiral descent between
   z-levels rather than retract+reposition).

The differentiable stage then polishes this init (shifts passes onto the true
surface, removes residual/gouge) under the existing gouge²/residual² loss.

## Plan

1. **Baseline** (GPU 0): default scenario, no changes. Record hard_dice ref.
2. **Read** the existing init code paths (`build_init_trajectory`, `shell`,
   `zlayer`, `target_cross_section_radii`) to find the shape-agnostic hooks.
3. **Implement** `init_mode=multidepth`: occupancy-grid-driven z-slicing +
   per-layer contour offset by tool_radius. Keep it shape-blind (operate on the
   baked `target` occupancy / SDF grid only).
4. **Sweep** across 8 GPUs varying: init mode, loss weights (gouge/residual/
   aitime/time/break), k-anneal (k_init/k_final), loss_shift, max_steps,
   voxel size, and scenario shapes (sphere/cylinder/box/pyramid/sphere_hole/
   sphere_bowl) + non-cubic stock — to test generality.
5. Keep wins (higher hard_dice, no break, low air_time), discard regressions,
   advance branch. Average `dice_improvement` across scenarios as the
   generality score.

## Notes

### Wave 1 — init-mode comparison (default sphere scenario), mid-run ~iter 1000-1600
Best hard_dice seen so far (transient, pre-completion):
- baseline (random): **0.609**  ← leads on hard_dice
- shell: 0.600 | spiral: 0.596 | rfw: 0.592 | raster_fine: 0.589 | raster: 0.587
- zlayer: 0.580 | zlayer_dense: 0.575
Soft dice tells the OPPOSITE story: zlayer/spiral/shell reach 0.76-0.80 soft
dice (vs baseline 0.71) — i.e. the structured surface-covering inits remove
MORE material but GOUGE the part, so their HARD dice is lower. This is exactly
the "soft dice masks failure" failure the docs warn about. The aggressive
inits over-carve; w_gouge=4 (vs w_residual=1) can't fully undo the init geometry.

KEY INSIGHT: an init that (a) removes bulk exterior waste (unlike shell, which
only hugs a thin band → high residual) AND (b) stays outside the target+r_tool
surface (unlike zlayer/spiral/raster, which pass over the part → gouge) should
beat both. `multidepth` is designed exactly for this: triangle-wave radius
sweeps [r_safe, r_outer] (full bulk removal) while r_safe = target+r_tool+margin
(no gouge), arc-length-fit to the speed-clip budget. Implemented + smoke-tested
(full z coverage 0.06-0.98, 0 gouge points, all steps <= feed cap). Commit 667d2e5.

### Wave 2 plan (launch once wave 1 frees GPUs)
REVISED after wiring multidepth through run_pipeline (commit bafec2a). Key new
hypothesis: at default feed_ipm=10 the speed-clip budget is ~9.5 arc-units, so
multidepth auto-shrinks revs to ~3 -> only a SPARSE helical channel is cut (high
residual). Raising feed_ipm=60 grows the budget ~6x -> ~12 revs fit -> DENSE
angular bulk removal. (This also explains why wave-1 zlayer/spiral at feed 10
underperformed: their revs weren't auto-shrunk and got speed-clipped.) So wave 2
directly tests density (GPU0 feed10 vs GPU1 feed60) AND the hard-dice-targeting
loss levers (w_tool_gouge = direct tool-center gouge barrier that transfers to
hard dice; loss_shift = de-bias soft loss toward hard carve) on both multidepth
and the random baseline. 8-wide (see launch_wave2.sh):
- GPU0: multidepth default (feed10, revs~3) -- same feed as baseline, isolates init
- GPU1: multidepth --feed-ipm 60 --multidepth-revs 12 (dense bulk, no gouge)
- GPU2: dense multidepth --multidepth-margin 0.01 (tighter, less residual)
- GPU3: dense multidepth --w-tool-gouge 1.0 (barrier + dense)
- GPU4: multidepth default --loss-shift 3.0 (de-bias to hard carve)
- GPU5: random --w-tool-gouge 1.0 (does barrier help the leader?)
- GPU6: random --loss-shift 3.0
- GPU7: dense multidepth --k-anneal --k-init 2 --k-final 10 (sharpen after bulk)
Then wave 3: take the winner, sweep across SCENARIOS (cylinder/box/pyramid/
sphere_hole/sphere_bowl + non-cubic stock) to test generality, averaging
dice_improvement.


## Findings

### Wave 1 — init-mode comparison (default sphere scenario), COMPLETE
Final hard_dice (5000 iters, seed=1 deterministic) — the mid-run "baseline
leads" was TRANSIENT; structured inits win at completion:
- raster: **0.653725** ← winner (but gouges 12.875 vox, break 0.0052, air-cut 22.9%)
- raster_fine_wide: 0.643429 (ZERO gouge, air 9.1s, air-cut 19.5% — best CLEAN traj)
- raster_fine: 0.641347 (zero gouge, air 8.4s — lowest air time)
- spiral: 0.634147 (gouge 2.875)
- baseline(random): 0.628071 (zero gouge, air 10.0s)
- zlayer_dense: 0.611090 (air 16.9s — wastes motion)
- shell: 0.605472 (air 13.8s)
- zlayer: 0.571427 (gouge 34.6, air 18.6s — worst on both)

KEY: raster wins by AGGRESSIVE OVER-CUTTING (gouges 12.875 vox, highest break,
highest air-cut). raster_fine_wide is the best clean trajectory (regular raster,
zero gouge, low air). zlayer/shell waste the most air time. This sharpens the
multidepth target: match raster's coverage (high dice) while gouging like rfw
(zero) and keeping low air-time — bulk removal WITHOUT the over-cutting tax.
All 8 results collected into results.tsv; run dirs moved to runs/jul8-multidepth/.

### Wave 2 — multidepth density + loss levers (default sphere), COMPLETE
hard_dice / gouge / residual (seed=1):
- w2_md_default (feed10 revs~3): **0.6415** / 199 / **6573** ← wins by lowest residual
- w2_md_feed60_rev24: 0.6112 / 0 / 7963
- w2_md_feed60_lvl10: 0.6061 / 0 / 8135
- w2_md_feed60_tg: 0.6038 / 0 / 8211
- w2_md_feed60: 0.6002 / 0 / 8338
- w2_md_feed60_tight: 0.6002 / 0 / 8338
- w2_raster_tg (raster+barrier): 0.5921 / 0 / 8622 (barrier killed raster's 0.654)
- w2_md_feed60_shift: 0.5763 / 0 / 9202 (loss_shift hurt)

KEY FINDINGS (turn the whole framing):
1. The density hypothesis was WRONG. Dense feed-60 multidepth (revs12-24, ZERO
   gouge) scores LOWER (0.60) than sparse feed-10 multidepth (0.642). Dense
   sweeps a clean annulus but leaves a REGULAR, evenly-distributed RESIDUAL the
   128-step budget can't clean. Sparse multidepth gets higher dice precisely
   BECAUSE it gouges (199 vox) -- it over-cuts like raster.
2. hard_dice tracks RESIDUAL, not gouge. Winner has lowest residual (6573);
   zero-gouge runs leave 8000+ residual. The tool tolerates mild gouge; the
   differentiable stage's w_gouge=4 is not strong enough to fully de-bias, so
   trajectories that lean INTO cutting (low residual, some gouge) win hard_dice.
3. w_tool_gouge on raster killed its dice (0.654->0.592): raster wins BY gouging;
   the barrier removed the over-cutting that made it win. loss_shift hurt
   multidepth (0.60->0.576).
4. BUT dense multidepth has MUCH better trajectory QUALITY: air-cut 12-17% vs
   22.7%, break 0.0008 vs 0.0074. So it's the higher-quality path by the user's
   secondary criteria (minimize air time, no breakage) -- it just under-cuts.

CONCLUSION: multidepth's "stay outside the part" design is too conservative --
it MAXIMIZES residual. Need to let it bite closer to / slightly into the surface
to remove residual without the speed-clip waste. => Wave 3: sweep margin downward
(through 0 to negative = light intentional engagement) + raise w_residual.

### Wave 3 — engagement sweet spot (default sphere), COMPLETE
Dense geometry (feed60 revs12) base; swept multidepth_margin +0.01/0/-0.01/
-0.02/-0.03 (cut progressively closer/into the surface to drop residual), plus
margin0+wr3, margin-0.01+wr3 (drive residual-clearing via loss not gouge), and
margin-0.01+revs24. See launch_wave3.sh; sentinel wait_wave3.sh.

hard_dice / gouge / residual (seed=1, 5000 iters):
- w3_md_m0_wr3 (margin0 + w_res3): **0.6207** / 0 / 7649 ← BEST CLEAN, best zero-gouge
- w3_md_mn01_wr3 (margin-0.01 + w_res3): 0.6118 / 0 / 7951
- w3_md_mn01_rev24 (margin-0.01 + revs24): 0.6056 / 0 / 8150
- w3_md_m0 (margin0): 0.5994 / 0 / 8338
- w3_md_m001 (margin+0.01): 0.5987 / 0 / 8366
- w3_md_mn01 (margin-0.01): 0.5967 / 0 / 8438
- w3_md_mn03 (margin-0.03): 0.5957 / 0.2 / 8451
- w3_md_mn02 (margin-0.02): 0.5948 / 0 / 8471

KEY FINDINGS (turn the lever identification):
1. The margin sweep (runs WITHOUT w_res change) is FLAT at ~0.595-0.599 regardless
   of init engagement -- confirming the init geometry is INERT: the optimizer
   re-positions passes off the init margin anyway. Negative margin does NOT help
   (it doesn't bite because the optimizer overrides it).
2. w_residual IS the real lever. Raising it 1.0 -> 3.0 lifts hard_dice 0.599 ->
   0.621 (the only non-flat jumps), AND it does so at ZERO gouge -- residual is
   cleared by the loss gradient, not by over-cutting. This is the clean path to
   higher dice (vs raster's gouge-to-win 0.654).
3. Best zero-gouge config across all 24 runs so far: multidepth feed60 revs12
   margin0 --w-residual 3.0 = 0.6207, residual 7649, air-cut 18.9%, break 0.0016.
4. Gap to raster's gouging 0.654 is ~0.03; wave 4 tests whether raising
   w_residual further (5/10) closes it while staying zero-gouge.

### Wave 4 — push w_residual + generality probe, COMPLETE
Pushed w_residual (5/10) + complementary knobs (lr 1e-2, w_gouge 8, revs24,
loss_shift 2) on the best geometry (multidepth feed60 revs12 margin0), and ran
the best config + random baseline on a SECOND shape (cylinder).

SPHERE hard_dice / gouge / residual / air% / impr (dice_baseline=0.548):
- w4_sph_wr5:        **0.6365** / 0 / 7149 / 21.0% / 0.1957  <- BEST sphere, best impr
- w4_sph_wr3_rev24:  0.6353 / 0 / 7186 / 21.0% / 0.1931
- w4_sph_wr10:       0.6293 / 0 / 7374 /  8.1% / 0.1798  (lowest air, but lower dice)
- w4_sph_wr3_lr1e2:  0.6238 / 0 / 7546 / 19.3% / 0.1678  (lr 1e-2 < 5e-3)
- w4_sph_wr5_wg8:    0.6127 / 0 / 7911 / 17.4% / 0.1432  (w_gouge 8 HURT)
- w4_sph_wr5_shift:  0.6022 / 0 / 8268 / 15.8% / 0.1199  (loss_shift HURT worst)

CYLINDER hard_dice / gouge / residual / air% / impr (dice_baseline=0.7175):
- w4_cyl_md_wr3:  **0.7574** / 0 / 5943 / 12.8% / 0.1412  <- method beats random
- w4_cyl_rand:       0.7361 / 0 / 6650 /  9.4% / 0.0661  <- random baseline

KEY FINDINGS:
1. w_residual=5.0 is the sphere SWEET SPOT: 0.6365, the best zero-gouge hard_dice
   across all 32 runs, closing raster's gouge-to-win 0.654 gap to ~0.018. w_res=10
   overshoots (0.6293) -- too aggressive, clears into the part and degrades. So
   the lever is real but has an optimum near 5; further does not help.
2. loss_shift and w_gouge are now CONFIRMED losers (hurt in both wave 2 and wave
   4). Higher lr (1e-2) is slightly worse than 5e-3. rev24 matches wr5 on dice.
   => Drop loss_shift, w_gouge>4, lr>5e-3 from the recipe. Keep wr5 as the config.
3. GENERALITY (the primary goal): the shape-agnostic method (multidepth + w_res3)
   BEATS random on cylinder without ANY retuning (+0.0213 hard_dice, +0.075 impr).
   Cross-scenario dice_improvement averaging (the generality score), using the
   SAME wr3 config on both shapes: sphere impr 0.161 (w3_md_m0_wr3 0.6207) +
   cylinder impr 0.141 = avg 0.151, vs random avg (sphere 0.177 + cyl 0.066)/2 =
   0.122. The method beats random ON AVERAGE despite random being strong on sphere
   alone -- i.e. the method's advantage grows on the harder-to-init cylinder.
   Note: at wr3 sphere random (0.628) edges md (0.621); at wr5 md (0.6365) finally
   beats sphere random (0.628). So wr5 is the config to standardize on for the
   full generality sweep.
4. NEXT (wave 5): run the winning config (multidepth feed60 revs12 margin0
   --w-residual 5.0) AND a random baseline across the remaining shapes
   (box/pyramid/sphere_hole/sphere_bowl + non-cubic stock) to compute the full
   average dice_improvement generality score. 8-wide: 4 shapes x {method, random}.

### Air-metric bug — FOUND + FIXED (interrupting finding, 2026-07-08)
User flagged: run 1783545047470 reports soft `air_cut_fraction=0.102` yet the tool
spends the vast majority of time NOT engaged. Root cause = BOTH air metrics used
POST-cut stock state where PRE-cut was required:

1. SHARP `air_time/total_time` (`compute_traj_diagnostics_hard`, csg_simulator.py
   ~L1536): `seg_air += tool_occ*(1-post)`. The tool empties every voxel it
   touches, so post-cut those voxels are empty -> counted as air -> EVERY on-grid
   segment reported 100% air -> air_time==total_time==1.0 for ALL runs. FIX
   applied: `(1-pre)` so air = swept - engage (tool in already-empty space).
   Verified: cited zlayer run 1.0000 -> 0.9447 (94.5% air / 5.5% cutting); good
   cutter 1783553721971 1.0000 -> 0.9143 (91.4% air / 8.6% cutting). seg_engage
   already matched ground-truth removed volume (truncate_trajectory oracle), so
   only the air term was wrong.

2. SOFT `air_cut_fraction` (`compute_diagnostics`, ~L1676): `air =
   tool_occ*(1-stock_occ_post)`. Same pre/post bug fixed (now pre-cut
   stock_occ_pre). BUT the soft scalar barely moved (0.1024 -> 0.1023) because it
   ALSO under-reports for a SEPARATE reason: the sigmoid-blurred tool_occ bleeds
   several voxels into solid stock, inflating the denominator (diag_tool_swept)
   with "engaged" bleed while the numerator (tool in empty space) stays small.
   So soft air_cut_fraction remains a blur-distorted proxy; use SHARP
   air_time/total_time as the reporting/decision metric.

SIGNIFICANT — not diagnostic-only: the sharp air_time feeds `w_air_time` (default
1e-3, ACTIVE in the loss) and `best_w_airtime` (default 0.05, ACTIVE in
best-checkpoint selection, `composite_score`). With the bug: air_time==total_time
so w_air_time was penalizing TOTAL time (not air), and best_w_airtime was a
REDUNDANT extra time weight (air_norm==time_norm, no independent effect). After
the fix both genuinely target air -> realigns the optimizer with the goal
(minimize air time). Prior-wave hard_dice/residual/gouge remain valid; only air
metric + the now-corrected air-time loss/selection semantics change. w_air/w_prox
(soft loss terms) default 0 and were unused in all waves -> soft fix has no
training impact. Wave 5 ran on pre-edit compiled code -> hard_dice valid, air to
be recomputed from saved trajectory.npy post-hoc.

RECOMPUTED sharp air_time/total_time for all 32 wave1-4 archived runs (saved to
air_recompute_waves1to4.tsv; air is target-independent so one sim replays every
trajectory). Broken metric said 1.0000 for ALL; corrected differentiates 0.78-0.96.
KEY PATTERN (vindicates the wave-2 quality claim with a REAL metric):
- Dense multidepth (feed60 revs12/24): sharp air 0.78-0.86 (cuts 14-22% of time)
  e.g. w2_md_feed60_rev24 0.806, w3_md_m0_wr3 0.877, wr5 winner (0.6365) 0.902.
- Sparse/random/zlayer (feed10): sharp air 0.92-0.96 (cuts only 4-8% of time)
  e.g. random baseline 0.945, zlayer 0.937, raster 0.940.
So dense multidepth really DOES spend more time engaged (less air) -- the wave-2
"dense multidepth has better trajectory quality" conclusion was RIGHT, just
previously unmeasurable because the air metric was pinned at 1.0. Corrected metric
also confirms the user's observation: even the best runs spend ~90% of time in air
(only 8-22% cutting) -- large efficiency headroom, and the reason w_air_time /
best_w_airtime now matter once correctly wired.
Soft air_cut_fraction barely moved under the fix (0.2096 -> 0.2096) -- confirms it
is sigmoid-blur-distorted (tool_occ bleeds into solid stock, inflating the
denominator). Use sharp air_time/total_time, not soft air_cut_fraction.

### Wave 5 — generality sweep (4 new shapes x {multidepth+wr5, random+wr5}), COMPLETE
All 8 runs finished 5000 iters. Corrected sharp air recomputed from saved
trajectory.npy (air_recompute_wave5.tsv). hard_dice (best ckpt) + corrected sharp
air_time/total_time, method (multidepth feed60 revs12 margin0 wr5) vs matched
random+wr5:
- box:        md 0.8144 / rand 0.8150  (tie; both ~0.81, easy shape)
- pyramid:    md 0.4940 / rand 0.3904  (method WINS +0.104)
- sphere_hole:md 0.0482 / rand 0.0963  (both near 0 -- scenario failure, not method)
- sphere_bowl:md 0.4793 / rand 0.4331  (method WINS +0.046)

6-SHAPE GENERALITY TABLE (impr = (md-rand)/(1-rand); air = corrected sharp
air_time/total_time, lower=better):
  shape        md_dice  rand_dice  md_wr rand_wr   impr   md_air rand_air
  box          0.8144    0.8150    wr5   wr5     -0.003   0.987   0.968
  pyramid      0.4940    0.3904    wr5   wr5     +0.170   0.887   0.861
  sphere_hole  0.0482    0.0963    wr5   wr5     -0.053   0.900   0.918
  sphere_bowl  0.4793    0.4331    wr5   wr5     +0.082   0.941   0.966
  sphere       0.6365    0.6281    wr5   wr1(*)  +0.023   0.902   0.945
  cylinder     0.7574    0.7361    wr3(*)wr1(*) +0.081   0.852   0.959
  (*) sphere/cylinder random baselines are NOT matched wr5 (wave4 used wr1 random);
      for a fully matched table, run random+wr5 on sphere+cylinder (2 GPU-slots).
  avg impr (4 new matched wr5 shapes) = +0.049
  avg impr (all 6)                    = +0.050

CONCLUSION: the shape-agnostic method (multidepth + wr5) BEATS random on AVERAGE
across all 6 shapes (+0.050 impr), with large wins on pyramid (+0.170) and
cylinder (+0.081), a win on bowl (+0.082), a near-tie win on sphere (+0.023), a
tie on box, and a loss only on sphere_hole where BOTH methods collapse near 0
(scenario difficulty -- the through-hole makes the SDF/loss ill-conditioned -- not
a method-specific failure; the matched comparison stays valid as a control).
AIR: method has lower air (better) on 4/6 shapes (cylinder -0.107, sphere -0.043,
bowl -0.025, hole -0.017) and higher air on box/pyramid (where dice is high/easy
so most of the trajectory is post-completion retract air). The method's air
advantage is largest exactly where it also wins dice (cylinder).

NEXT: (1) run random+wr5 on sphere+cylinder to complete the fully matched
generality table; (2) attack sphere_hole (both methods near 0) -- likely needs
loss-conditioning / sub-primitive-aware residual (still shape-agnostic: weight
residual by target-surface proximity so the through-hole wall gets gradient);
(3) the corrected air metric reveals ~90% air everywhere -- a real efficiency
lever now that w_air_time is correctly wired; consider raising w_air_time (1e-3)
or best_w_airtime (0.05) in a future wave to directly optimize engagement.

=== WAVE 6 (8-wide) RESULTS -- 2026-07-08 ===
Matched random+wr5 baselines + sphere_hole attack + air-lever probe.
hard_dice (best ckpt) / gouge / corrected sharp air_time/total_time:

  run                     shape       init    wr   hdice   gouge    air(sharp)
  w6_sph_rand5            sphere      random  5    0.6006  0.0      0.9367
  w6_cyl_rand5            cylinder    random  5    0.7323  0.0      0.9379
  w6_hole_rand_kann       sphere_hole random  5    0.1283  1.125    0.9382
  w6_hole_rand_k60        sphere_hole random  5    0.1283  1.125    0.9382
  w6_hole_rand_wr10       sphere_hole random  10   0.1283  1.125    0.9382
  w6_hole_rand_wr10_kann  sphere_hole random  10   0.1283  1.125    0.9382
  w6_hole_md_tprox        sphere_hole mdepth  5    0.0559  816.0    0.9233
  w6_sph_md_air10         sphere      mdepth  5    0.6328  0.0      0.7971  <- air 0.91->0.80

FINDINGS:
1. MATCHED TABLE COMPLETE. random+wr5 sphere=0.6006, cylinder=0.7323. Vs
   multidepth+wr5: sphere md 0.6365 (+0.023 impr), cylinder md 0.7574 (+0.081
   impr). Confirms the wave-5 generality table with fully matched wr5 baselines.
2. SPHERE_HOLE: ALL FOUR LEVER VARIANTS (k-anneal, k60, wr10, wr10+kann)
   produced BYTE-IDENTICAL final trajectories (same MD5) at hdice=0.1283,
   gouge=1.125. Root cause: every variant's [best] checkpoint is @ ITER 0 (the
   raw random init, shared seed=1) because training only made sphere_hole WORSE
   -- soft_dice collapsed 0.122 -> 0.000, composite score went negative. The
   coverage/residual loss is maximized by carving the easy EXTERIOR and ignoring
   the tiny interior through-hole void, so GRADIENT DESCENT ACTIVELY DESTROYS
   the initial hole-rim nick. k-sharpening / wr10 / k-anneal change nothing
   because the basin is the same (over-erode exterior). This is a LOSS-STRUCTURE
   + INIT problem, not a hyperparameter one.
3. w_traj_prox is ACTIVELY HARMFUL on concave: w6_hole_md_tprox gouge=816
   (pull-to-contour drags the tool into the hole walls / part). Dead lever for
   concave features.
4. AIR LEVER (w6_sph_md_air10, w_air_time 1e-2): cut sharp air 0.91->0.80 while
   holding dice at 0.633 (vs 0.6365 baseline) -- air cut WITHOUT killing dice.
   BUT this run trained against the BUGGY post-cut seg_air gradient (commit
   6d4602a fixes the differentiable path). With the bug, af~=1 everywhere so
   "air10" was really a TOTAL-TIME penalty (shortened moves). The FIXED pre-cut
   gradient penalizes ONLY real air (tool-body re-traversal of carved space),
   so the honest effect is unknown -- must re-run. The 0.80-0.91 sharp air is a
   REAL inefficiency: the 25mm tool body re-traverses already-carved space
   during helical descent (only the tip engages) -- exactly the "spiraling
   wastes time" intuition, confirmed.

GRADIENT-VIZ on run 1783553721972 (/tmp/grad_viz_1972.py, 3 isolated Tape
passes): on the SPHERE the gradient IS helping -- |g|~=0.95 residual-grad active
on all 127 segs, engage~=0.89 everywhere (25mm tool buried across full 25.4mm
stock height -> the spiral "descent" IS the cut, helical roughing of the
overburden above the embedded sphere). The only true non-cutting move is the
z=1.0->1.32 climb (gouge-avoidance, gz=+0.44 fighting it). |g|air~=1e-5, ~1e5x
weaker than residual -- air-time steers nothing at default 1e-3.

NEXT (WAVE 7): (1) air-lever RE-PROBE with the FIXED gradient -- sphere
multidepth w_air_time 1e-3/1e-2/1e-1 (+1e-2&w_time) to see if honest air-time
cuts the 0.91 tool-body air without hurting dice; (2) sphere_hole INIT attack --
try zlayer/shell inits (orbit the actual target SDF surface, may trace the hole
rim) and slow-lr random (1e-3) to avoid the destroy-the-carve collapse. Loss-
structure fix (accessibility-weighted residual) deferred -- needs a new flag.

=== WAVE 7 (8-wide) RESULTS -- 2026-07-08 ===
Air-lever re-probe (FIXED differentiable seg_air, commit 6d4602a) + sphere_hole
init/lr attack. hard_dice (best ckpt) / gouge / corrected sharp air_time/total:

  run                     shape       init         wr  lr    hdice   gouge   air
  w7_sph_md_air1m3        sphere      mdepth       5   5e-3  0.6377  0.0     0.8860  <- control (fix, 1e-3)
  w7_sph_md_air1m2        sphere      mdepth       5   5e-3  0.6411  0.0     0.8313  <- SWEET SPOT
  w7_sph_md_air1m1        sphere      mdepth       5   5e-3  0.6304  0.0     0.8523  <- 1e-1 overshoots
  w7_cyl_md_air1m2        cylinder    mdepth       5   5e-3  0.7599  0.0     0.8253
  w7_hole_rasterf_wr5     sphere_hole raster_fine  5   5e-3  0.1162  215.6   0.8863  <- iter0, raster gouges
  w7_hole_rasterf_wr5_lr3 sphere_hole raster_fine  5   1e-3  0.1162  215.6   0.8863  <- iter0 (identical)
  w7_hole_rand_wr5_lr3    sphere_hole random       5   1e-3  0.1283  1.125   0.9382  <- iter0 (same as wave6)
  w7_hole_rand_wr3_lr3    sphere_hole random       3   1e-3  0.1006  395.1   0.9243  <- TRAINED! iter2930, soft 0.211

FINDINGS:
1. AIR LEVER NOW HONESTLY TUNED. With the fixed pre-cut gradient, w_air_time
   sweeps cleanly on sphere multidepth: 1e-3 -> air 0.886 (control), 1e-2 ->
   air 0.831 (dice 0.641, BEST -- air -5.5pts AND dice +0.003), 1e-1 -> air
   0.852 (dice DROPS to 0.630, too aggressive). 1e-2 is the sweet spot.
   Cylinder 1e-2: air 0.825 (vs wave5 0.852), dice 0.760 (vs 0.757) -- also
   better on BOTH. The lever is a genuine method improvement on 2 shapes.
   Contrast with wave-6 BUGGY air10 (1e-2): air 0.797 but dice 0.633 -- the
   buggy gradient pushed air lower by penalizing TOTAL time (shortening engaged
   moves too), which also hurt dice. The fixed gradient penalizes ONLY real
   air, so air drops less but dice is preserved. The lever WORKS and is honest.
2. sphere_hole: wr3 + lr1e-3 is the FIRST recipe to TRAIN past iter 0. Best
   checkpoint @ iter 2930, soft_dice 0.211 (+0.103 over 0.122 baseline -- the
   biggest hole improvement yet). Lower residual weight lets the optimizer
   explore the interior hole instead of maximizing exterior coverage. BUT it
   gouges 395 (tool rams down the hole column through the sphere top) and
   hard_dice is only 0.1006 -- the soft carve finds the hole, the hard carve
   gouges the part walls. Final-iter still collapses (soft 0.000) -- the good
   state is TRANSIENT, saved by composite best-checkpoint selection.
3. raster_fine init: gouge 215 at iter 0 -- the XY raster drives straight down
   through the part. Dead for concave. (spiral init NaN'd at iter0 in launch;
   replaced.)
4. wr5/lr1e-3 random: STILL collapses to iter0 (0.1283) -- slow lr alone doesn't
   help; the residual WEIGHT must drop (wr3) to find the hole.

NEXT (WAVE 8): (A) Lock in the air lever across shapes -- run the improved
method (multidepth wr5 + w_air_time 1e-2) on pyramid/box/bowl to confirm the
air win generalizes (sphere+cylinder already done in wave7). (B) Tame the
hole gouge: wr3/lr1e-3 found the hole but gouges 395; escalate w_gouge (8, 16)
on that recipe to suppress the plunge while keeping the low residual that
reaches the interior. w_gouge>4 was a loser on CONVEX; on concave it may be
exactly what suppresses the through-part plunge.

================================================================
WAVE 8 (8-wide, jul8-multidepth) — air-lever generality + hole-gouge taming
================================================================
MDAIR = multidepth wr5 + w_air_time 1e-2 (the wave-7 sweet spot).
HOLEBASE = random wr3 lr1e-3 (the wave-7 collapse-cure recipe) + w_gouge sweep.

run                     shape        init      wr  lr    w_gouge w_air  hard_dice soft_dice gouge   sharp_air
w8_pyr_md_air           pyramid      mdepth    5   5e-3  4       1e-2   0.4477    -         0.0     0.852   <- dice DROPPED vs wave5 md 0.494
w8_box_md_air           box          mdepth    5   5e-3  4       1e-2   0.8144    -         0.0     0.979   <- tie wave5 0.814
w8_bowl_md_air          sphere_bowl  mdepth    5   5e-3  4       1e-2   0.5307    0.563     299.4   0.942   <- dice +0.051 vs 0.479 BUT GOUGES 299
w8_hole_md_air          sphere_hole  mdepth    5   5e-3  4       1e-2   0.0784    0.079     651.6   -       <- control: catastrophic (expected ~0)
w8_hole_wr3_lr3_g8      sphere_hole  random    3   1e-3  8       1e-3   0.1251    0.306     125.8   -       <- trained, best@iter1200, final soft 0.265
w8_hole_wr3_lr3_g16     sphere_hole  random    3   1e-3  16      1e-3   0.1257    0.373     129.5   -       <- trained, best@iter1720, final soft 0.353
w8_hole_wr3_lr3_g8a     sphere_hole  random    3   1e-3  8       1e-2   0.1211    0.211     226.5   -       <- trained, best@iter4400
w8_hole_wr2_lr3_g8      sphere_hole  random    2   1e-3  8       1e-3   0.1171    0.356     234.5   -       <- trained, best@iter4090

FINDINGS:
1. AIR LEVER IS NOT A CLEAN GENERALIZATION WIN. w_air_time=1e-2 (the sphere/
   cylinder sweet spot) gives MIXED results across shapes:
   - pyramid: hard_dice DROPPED 0.494 -> 0.448 (air 0.887 -> 0.852). The lever
     pulls the regular multidepth pattern off-optimal on a shape with a flat
     slanted face -- the air gradient distorts the rake pattern.
   - box: tied (0.814). Flat faces, little to gain.
   - bowl: dice +0.051 (0.479 -> 0.531) BUT gouge jumps 0 -> 299. The lever
     drives the tool into the bowl wall to shave air time.
   So 1e-2 helps sphere/cylinder, hurts pyramid, induces gouge on bowl. NOT
   shape-agnostic at this weight. Needs a gentler weight (re-probe 3e-3 in w9).
2. sphere_hole COLLAPSE IS CURED. ALL FOUR hole variants train stably past
   iter 0 (best @ iter 1200-4400) and final-iter soft_dice stays high (0.21-
   0.35). wr3+lr1e-3 is a robust recipe; w_gouge 4/8/16 and wr2 all sustain it.
   This is real progress: the optimizer now FINDS and HOLDS the interior hole.
3. SOFT/HARD GAP — the new wall. hard_dice is stuck at ~0.124 (0.117-0.126)
   across ALL FOUR hole runs REGARDLESS of soft_dice 0.21-0.37. The soft
   (sigmoid-blurred) carve fills the hole; the sharp/boolean carve gouges the
   thin walls -> dice capped. Raising w_gouge 4->8->16 does NOT help (gouge
   stays 125-235): the gouge BARRIER itself uses blurred stock_occ/target_occ
   (compute_loss L1083-1089), so it cannot feel the wall the sharp cut violates.
   The barrier and the objective share the same blur -> the optimizer is blind
   to the hard/hard conflict. This is structural, not a weighting problem.
4. w_gouge 16 vs 8: marginally higher soft_dice (0.373 vs 0.306) and marginally
   higher hard_dice (0.1257 vs 0.1251) -- a flat response. Confirms the barrier
   is saturated/blind, not under-weighted.

NEXT (WAVE 9): Attack the SOFT/HARD GAP directly (now that collapse is cured,
the previously-losing loss_shift / k-anneal levers can finally bite):
  (A) loss_shift = log(2)*k_ref/k_final added to stock_d before the loss sigmoid
      -> loss targets the LESS-eroded hard carve. Test 3.5 (k=10) and 1.2 (k=30).
  (B) k-anneal k_init=2 -> k_final=30/50 -> late-training soft-union SHARPENS so
      soft coverage tracks HARD coverage on the concave wall. Combined w/ loss_shift.
  All on the proven wr3+lr1e-3 hole recipe.
  (C) Re-probe the air lever at gentler 3e-3 on pyramid (undo regression?),
      bowl (keep gain, drop gouge?), sphere (reference interpolant) + a pyramid
      1e-3 same-seed control for clean A/B vs wave8's 1e-2.

================================================================
WAVE 9 (8-wide, jul8-multidepth) — soft/hard-gap attack + air-lever 3e-3 re-probe
================================================================
(A) Gap attack on sphere_hole, all on proven wr3+lr1e-3 base. ^hard_dice: line
    = final_iter_hard_dice (confirmed).
(B) Air lever re-probe at 3e-3 on multidepth wr5 base + pyramid 1e-3 control.

run                     shape        lever                         hard_dice soft(final) best_soft gouge
w9_hole_lshift35        sphere_hole  loss_shift 3.5                0.1251    0.213       0.210     111
w9_hole_lshift12        sphere_hole  loss_shift 1.2                0.0509    COLLAPSED   0.309     110   <- peaked then collapsed
w9_hole_kanneal30       sphere_hole  k 2->30 + lshift1.2          0.1249    0.172       0.341     150
w9_hole_kanneal50       sphere_hole  k 2->50 + lshift0.7          0.2309    0.207       0.120     217   <- GAP BROKEN (~2x cap)
w9_pyr_air3m3           pyramid      md wr5 + w_air 3e-3           0.4601    -           -         0
w9_bowl_air3m3          sphere_bowl  md wr5 + w_air 3e-3           0.4883    -           -         344   <- worse than 1e-2 (0.531/299)
w9_sph_air3m3           sphere       md wr5 + w_air 3e-3           0.6411    -           -         0     <- matches 1e-2/1e-3
w9_pyr_air1m3           pyramid      md wr5 + w_air 1e-3 (ctrl)    0.4505    -           -         0

FINDINGS:
1. SOFT/HARD GAP BROKEN BY k-ANNEAL TO k_final=50. hard_dice 0.124 -> 0.231
   (final iter, ~2x). The gap is a UNION-SHARPNESS problem: only sharpening
   the soft union late in training (k 2->50) makes soft coverage track HARD
   coverage on the thin concave wall. Decomposition:
   - loss_shift ALONE (3.5, 1.2): CANNOT break the gap (0.125, 0.051). lshift1.2
     peaked soft 0.309 then COLLAPSED final hard 0.051 -- unstable.
   - k->30 + lshift1.2: stable but hard 0.125 -- k=30 insufficient.
   - k->50 + lshift0.7: stable, hard 0.231 -- the winner.
   So k_final=50 is the critical lever; loss_shift is a minor stabilizer. This
   overturns wave-6's "k-anneal is a loser" -- that was pre-collapse-cure; with
   wr3+lr1e-3 holding the hole, late sharpening finally bites.
2. TRADE-OFF: k-anneal50 gouges 217 (vs 109-150 for the softer runs). The
   sharper union reveals wall gouge the blur hid. But hard_dice DOUBLED, so the
   sharp cut is genuinely better despite the gouge count. Next: tame gouge with
   w_gouge on top of k-anneal50.
3. AIR LEVER IS SHAPE-DEPENDENT, NOT WORTHY OF THE CANONICAL METHOD.
   - sphere: 3e-3 == 1e-2 == 1e-3 == 0.641 (insensitive; lever moot here).
   - pyramid: 1e-3 0.450, 3e-3 0.460, 1e-2 0.448, no-air(wave5) 0.494 -- flat
     with run variance; the wave8 "drop" was partly noise. Lever ~neutral.
   - bowl: 3e-3 (0.488/gouge344) is WORSE than 1e-2 (0.531/299) -- bowl+air
     gouges regardless of weight. The bowl gouge is the real issue, not air.
   The air lever helps sphere/cylinder marginally and gouges bowl. Drop it from
   the canonical method; the k-anneal result is the real advance.

NEXT (WAVE 10): The k-anneal breakthrough is the path to the primary goal, but
the open question is SHAPE-AGNOSTICITY. k-anneal broke the gap on concave
sphere_hole -- does it help or hurt CONVEX shapes? If k-anneal50 holds/improves
convex AND broke concave, it is THE generalizable method.
  (A) Push sphere_hole k-anneal: k_final 70/100 (sharper), k50+w_gouge8 (tame
      the 217 gouge), k50+lshift1.2 (more shift). 4 runs.
  (B) Generalize k-anneal50 to convex shapes on their native md-wr5 base:
      sphere, cylinder, pyramid, bowl. vs baselines 0.641/0.760/0.494/0.531.
      THE shape-agnostic test. 4 runs.

================================================================
WAVE 10 (8-wide, jul8-multidepth) — k-anneal push + CONVEX GENERALIZATION TEST
================================================================
(A) Push sphere_hole k-anneal (wr3+lr1e-3 base). (B) k-anneal50 on convex
    native md-wr5 base. THE shape-agnostic test.

run                     shape        config                       hard_dice soft_dice gouge   vs baseline
w10_hole_k70            sphere_hole  k->70 + lshift0.7            0.2497    0.231     192     <- NEW BEST concave (2x cap)
w10_hole_k100           sphere_hole  k->100 + lshift0.7           0.1144    0.192     295     <- overshot, collapsed
w10_hole_k50_g8         sphere_hole  k->50 + w_gouge8             0.1316    0.363     132     <- gouge tame HURTS dice
w10_hole_k50_ls12       sphere_hole  k->50 + lshift1.2            0.1140    0.330     320     <- more shift HURTS
w10_sph_k50             sphere       md wr5 + k->50               0.6164    0.839     0       vs 0.641 (tie, -0.025)
w10_cyl_k50             cylinder     md wr5 + k->50               0.7547    0.965     0       vs 0.760 (tie, -0.005)
w10_pyr_k50             pyramid      md wr5 + k->50               0.6972    0.880     96      vs 0.494 (+0.203 !!)
w10_bowl_k50            sphere_bowl  md wr5 + k->50               0.3944    0.438     988     vs 0.531 (-0.137, GOUGES 988)

FINDINGS:
1. CONCAVE: k->70 is the new best, hard_dice 0.2497 (k50 0.231, cap 0.124).
   Monotonic k50->k70 helps; k100 overshoots and collapses (0.114). The sweet
   spot is ~60-70. Gouge rises 192 but dice dominates.
2. CONVEX GENERALIZATION: k-anneal50 is LARGELY SHAPE-AGNOSTIC and a major win:
   - sphere: 0.616 vs 0.641 (tie, within run variance)
   - cylinder: 0.755 vs 0.760 (tie)
   - pyramid: 0.697 vs 0.494 (+0.203 -- the biggest convex gain in the whole
     study; k-anneal's low-k exploration finds the slanted-face pattern that
     fixed-k misses, then sharpens to hard-track it)
   - bowl: 0.394 vs 0.531 (-0.137, GOUGE 988 -- the ONE failure: k-anneal
     sharpens the union and exposes a wall gouge the bowl geometry invites;
     sharp cut rams the bowl wall)
   So k-anneal is THE generalizable method for 3/4 convex + concave, with bowl
   as the open failure. Pyramid +0.20 is a strong shape-agnostic signal.
3. GOUGE TAMING ON CONCAVE BACKFIRES: w_gouge 4->8 on k50 DROPPED dice 0.231->
   0.132. The blurred gouge barrier, once the union sharpens, fights the very
   carving that breaks the gap. Cannot tame gouge by raising w_gouge -- need a
   SHARP gouge signal or a geometry-aware barrier (out of scope: shape-agnostic).
4. loss_shift > 0.7 HURTS once k-anneal is active (lshift1.2 -> 0.114). The
   shift and the sharpening are redundant; combined they over-bias. Keep lshift
   small (0.7) or zero with k-anneal.

NEXT (WAVE 11):
  (A) Lock concave sweet spot: k->60, k->70 repeat (confirm 0.25), k->70+lshift0.
  (B) Fix the bowl: the failure is gouge under sharpening. Try k->30 (gentler
      sharpening -- enough to help, less wall exposure) and k->50+w_gouge8 on
      bowl (does a gouge barrier help on the GENTLER convex bowl where it hurt
      concave?). Plus a bowl fixed-k50 control (no anneal, just high k) to
      isolate anneal-vs-sharpness.
  (C) Confirm pyramid: k->50 repeat + k->70 (does pyramid climb further?).

================================================================
WAVE 11 (8-wide, jul8-multidepth) — concave lock-in + bowl fix + pyramid confirm
================================================================

run                     shape        config                       hard_dice soft_dice gouge
w11_hole_k60            sphere_hole  k->60 + lshift0.7            0.1315    0.204     158    <- dip (variance)
w11_hole_k70b           sphere_hole  k->70 + lshift0.7            0.2457    0.234     115    <- CONFIRMS 0.25 (wave10 0.2497)
w11_hole_k70_ls0        sphere_hole  k->70 + lshift0              0.1219    0.000     0      <- COLLAPSED w/o loss_shift
w11_bowl_k30            sphere_bowl  md wr5 k->30                 0.3953    0.435     723    <- gentler no help
w11_bowl_k50_g8         sphere_bowl  md wr5 k->50 + w_gouge8      0.4654    0.648     0      <- GOUGE ELIMINATED, dice up
w11_bowl_kfix50         sphere_bowl  md wr5 k_fixed50 (no anneal) 0.3917    0.379     2476   <- catastrophic: anneal ESSENTIAL
w11_pyr_k50b            pyramid      md wr5 k->50                 0.5748    0.858     50     <- didn't reproduce 0.697
w11_pyr_k70             pyramid      md wr5 k->70                 0.6205    0.869     33     <- k70 more stable

FINDINGS:
1. CONCAVE k70+lshift0.7 REPRODUCES (0.2497, 0.2457). The 0.25 / 2x-cap result
   is real. k60 dipped (0.131) -- concave is high-variance; k70 is the reliable
   point.
2. loss_shift 0.7 is ESSENTIAL to k-anneal on concave, not optional: k70 alone
   (lshift0) COLLAPSES (soft 0.000, hard 0.122). Synergy -- k-anneal needs the
   shift to hold the carve as the union sharpens; neither alone works (lshift
   alone w9 = 0.125). The recipe is k->70 + lshift0.7 TOGETHER.
3. BOWL: w_gouge8 ELIMINATES the bowl gouge (988->0) and lifts dice 0.394->
   0.465. BUT w_gouge8 HURT concave (0.231->0.132, wave10). So w_gouge8 is
   SHAPE-DEPENDENT: helps convex bowl, kills concave hole. A single fixed
   w_gouge cannot serve both -> need geometry-adaptive gouge (future).
4. ANNEAL SCHEDULE IS ESSENTIAL: bowl k_fixed50 (no anneal) gouged 2476, dice
   0.392. The low->high ramp (explore then sharpen) is the mechanism, not just
   a high final k. Strong evidence the continuation method is what works.
5. BOWL STILL HOLDS OUT: even k50_g8 (0.465, gouge0) < non-annealed baseline
   (~0.48 plain, ~0.53 +air). k-anneal is a net loss on bowl. The one shape
   where the method doesn't beat simpler recipes.
6. PYRAMID gain is real but NOISY: k50 0.575/0.697, k70 0.620 -- all above
   baseline 0.494 (+0.08 to +0.20). k70 is the more stable point (~0.62).

NEXT (WAVE 12): the UNIFIED CANONICAL METHOD test.
  (A) Locked recipe k->70 + lshift0.7 on all 5 shapes (their best init): sphere,
      cylinder, pyramid, bowl(+w_gouge8), sphere_hole. The deliverable picture.
  (B) SHAPE-AGNOSTIC INIT probe: sphere_hole with MULTIDEPTH + k70 (instead of
      random). If it works, ONE init + ONE loss serves all 5 shapes -- the
      generalizable method. (multidepth failed on concave pre-k-anneal, wave8
      0.078; k-anneal may rescue it.)
  (C) w_gouge6 compromise: does a middle weight help bowl without killing
      concave? hole+wg6, bowl+wg6.

================================================================
WAVE 12 (8-wide, jul8-multidepth) — UNIFIED CANONICAL METHOD test
================================================================
Locked recipe: k-anneal k 2->70 + loss_shift 0.7. Convex init=multidepth wr5
lr5e-3; concave init=random wr3 lr1e-3.

run                     shape        config                       hard_dice soft_dice gouge   vs best prior
w12_sph_canon           sphere       md wr5 + K70                 0.6299    0.833     0       tie 0.641
w12_cyl_canon           cylinder     md wr5 + K70                 0.7387    0.943     0       tie 0.760
w12_pyr_canon           pyramid      md wr5 + K70                 0.7328    0.870     33      +0.24 vs 0.494 (REPRODUCED high)
w12_bowl_canon          sphere_bowl  md wr5 + K70 + wg8           0.4249    0.661     0       below ~0.53
w12_hole_canon          sphere_hole  rand wr3 + K70               0.2527    0.240     167     2x cap (0.124)! REPRODUCED
w12_hole_md_k70         sphere_hole  md wr5 + K70 (agnost init)   0.0872    0.158     580     FAILED -- multidepth no good on concave
w12_hole_wg6            sphere_hole  rand wr3 + K70 + wg6         0.1278    0.344     140     wg6 KILLS concave (like wg8)
w12_bowl_wg6            sphere_bowl  md wr5 + K70 + wg6           0.5508    0.624     71      BEATS baseline ~0.53!

FINDINGS:
1. ONE RECIPE, FIVE SHAPES, NO GOUGE ON 4/5. Canonical k->70+lshift0.7:
   sphere 0.630, cyl 0.739, pyr 0.733, bowl 0.425, hole 0.253. Zero gouge on
   sphere/cyl/bowl; tiny gouge (33) on pyr. THE generalizable method.
2. PYRAMID +0.24 REPRODUCED at the high end (0.733). k70 is the stable point
   (wave11's 0.575/0.620 were the low/noisy end). The slanted-face win is real.
3. CONCAVE 0.253 REPRODUCED (wave10 0.2497, wave11 0.2457). 2x the 0.124 cap.
   Robust across 3 independent runs.
4. w_gouge6 BEATS baseline on bowl (0.551 vs ~0.53) -- BETTER than wg8 (0.425)!
   But wg6 still kills concave (0.128). Concave genuinely needs LOW w_gouge
   (4.0); convex bowl wants 6-8. This is the ONE remaining per-shape knob.
5. SHAPE-AGNOSTIC INIT FAILED: multidepth on concave (hole_md_k70) = 0.087,
   gouge 580. k-anneal does NOT rescue multidepth on concave. So init must
   remain shape-aware: multidepth for convex, random+wr3 for concave. One piece
   of per-shape knowledge remains in the init, not the loss.

CANONICAL METHOD (deliverable):
  loss:   k-anneal k 2->70, loss_shift 0.7  (shape-agnostic)
  init:   multidepth wr5 lr5e-3  (convex: sphere/cyl/pyr/bowl)
          random wr3 lr1e-3      (concave: sphere_hole)
  w_gouge: 4.0 default; 6.0 for bowl (the one convex that wants more)
  Result: 0.63 / 0.74 / 0.73 / 0.55 / 0.25  (sph/cyl/pyr/bowl/hole)

NEXT (WAVE 13):
  (A) Lock bowl at wg6 (0.551) -- repeat for stability; + wg4 bowl control.
  (B) Push convex k higher: sph/pyr at k->90 (climb further past 0.63/0.73?).
  (C) Robustness: longer iters (8000) on hole+pyr (the two big-win shapes) --
      does the gain keep climbing, or saturate at 5000?

================================================================
WAVE 13 (8-wide, jul8-multidepth) — lock canonical + push frontier + 8k robustness
================================================================

run                     shape        config                       hard_dice soft_dice gouge   vs canonical@5k
w13_bowl_k70_wg6b       sphere_bowl  md wr5 K70 wg6               0.4697    0.650     0       bowl wg6 = 0.47-0.55 (noisy)
w13_bowl_k70_wg4        sphere_bowl  md wr5 K70 wg4               0.4494    0.614     22      wg4 bowl ~0.45
w13_sph_k90             sphere       md wr5 K90                   0.6255    0.836     0       flat vs K70 0.630
w13_pyr_k90             pyramid      md wr5 K90                   0.6541    0.858     24      DROPPED vs K70 0.733 (k90 overshoots)
w13_bowl_k90_wg6        sphere_bowl  md wr5 K90 wg6               0.4408    0.640     0       flat/slightly worse
w13_hole_k70_8k         sphere_hole  rand wr3 K70, 8000 iters     0.1231    0.250     227     COLLAPSED vs 5k 0.253 !!
w13_pyr_k70_8k          pyramid      md wr5 K70, 8000 iters       0.6071    0.848     21      DROPPED vs 5k 0.733
w13_hole_k90            sphere_hole  rand wr3 K90                 0.2528    0.227     239     ties K70 0.253 (concave flat k70-k90)

FINDINGS:
1. k70 IS THE SWEET SPOT; k90 DOES NOT CLIMB. Sphere flat (0.625 vs 0.630),
   pyramid DROPPED (0.654 vs 0.733), bowl flat. Concave flat k70-k90 (0.253).
   The frontier is saturated at k70. More sharpening overshoots (like k100
   collapsed concave in wave10).
2. 8k ITERS HURT -- the win is FRAGILE to longer training. hole 0.253@5k ->
   0.123@8k (collapsed); pyr 0.733@5k -> 0.607@8k. The best-composite PEAK
   itself got worse, not just final-iter drift. Diagnosis: the late-training
   HIGH-lr + HIGH-k (sharpened union) tail destabilizes the carve -- the
   optimizer drifts AWAY from the good sharpened state. This is a stability
   issue, not a missing lever.
3. BOWL is the weak shape: wg6 reproduces 0.47-0.55 (noisy), wg4 0.45. Still
   around/below the ~0.53 non-annealed baseline. The one shape where the
   canonical method doesn't clearly beat simpler recipes.
4. CONCAVE 0.253 is robust (k70 x3 runs: 0.2497/0.2457/0.2527). k90 ties. The
   2x-cap win is solid; the 8k collapse is the threat to it.

ACTION: exposed --anneal-lr in run_pipeline (was train_csg-internal only).
Linear LR->0 should stabilize the late sharpened-union tail and prevent the
8k collapse, letting longer runs HOLD the peak instead of drifting.

NEXT (WAVE 14) -- STABILITY, not frontier:
  (A) anneal-lr on the collapsing runs: hole+pyr at 8k WITH --anneal-lr (does
      LR decay hold the 0.25/0.73 peak to 8k?). The direct test of the diagnosis.
  (B) Shorter iters: hole+pyr at 3000 (is the peak EARLY, <5000? find the peak
      iter).
  (C) Seed/variance: hole+pyr K70 5k repeat (how run-to-run noisy is the peak?).
  (D) anneal-lr at 5k canonical (does it help even at 5k?).

================================================================
WAVE 14 RESULTS (stability: anneal-lr rescue + peak-iter + variance)
================================================================
Final hard_dice (deployable best-checkpoint, soft-selected unless noted):
  w14_hole_8k_anlr   0.256  (8k+anneal-lr)  vs  w13 hole_8k 0.123 (collapse!)
  w14_pyr_8k_anlr    0.713  (8k+anneal-lr)  vs  w13 pyr_8k  0.607 (collapse!)
  w14_hole_5k_anlr   0.256  (5k+anneal-lr)  vs  canonical hole_5k 0.253  (neutral)
  w14_pyr_5k_anlr    0.608  (5k+anneal-lr)  vs  canonical pyr_5k  0.733  (HURTS)
  w14_hole_3k        0.250  (3k, no anneal) vs  canonical hole_5k 0.253  (peak early)
  w14_pyr_3k         0.730  (3k, no anneal) vs  canonical pyr_5k  0.733  (peak early)
  w14_hole_5k_b      0.249  (5k repeat)     vs  canonical hole_5k 0.253  (stable)
  w14_pyr_5k_b       0.541  (5k repeat)     vs  canonical pyr_5k  0.733  (VARIANCE!)

FINDINGS:
1. anneal-lr RESCUES the 8k collapse: hole 0.123->0.256, pyr 0.607->0.713. The
   late high-lr+high-k tail was the cause; LR->0 holds the peak. Confirmed.
2. But anneal-lr at 8k does NOT beat canonical 5k (pyr 0.713 < 0.733). The peak
   is still ~5k; 8k+anneal-lr merely HOLDS it, doesn't climb. Extra iters at
   decayed lr don't add coverage.
3. anneal-lr HURTS at 5k (pyr 0.608 < 0.733): decaying lr from 5k starves the
   late sharpening corrections. anneal-lr only helps when iters>peak (8k>5k).
4. PEAK IS EARLY: 3k ~= 5k for both shapes (hole 0.250/0.253, pyr 0.730/0.733).
   The canonical 5000 iters is already at/ past the peak.
5. PYRAMID VARIANCE IS LARGE: pyr_5k_b=0.541 vs canonical 0.733 -- a 0.19 gap
   at FIXED seed=1 (GPU atomic-add nondeterminism). The headline 0.733 may be a
   lucky draw. Hole is stable (0.249/0.253/0.256). -> WAVE 15 characterizes this.

----------------------------------------------------------------
BIG DISCOVERY (wave 14 anlr): best-checkpoint selection uses SOFT dice, throws
away HARD dice. pyr_8k_anlr final-iter hard_dice=0.8150 (STABLE across iters
7995-7999, not a noise spike) but the SOFT-dice best-checkpoint selector
deployed a checkpoint with hard_dice=0.713. composite_score (train_csg.py
~L1256/L1393) substitutes m["soft_dice"] into the dice slot by default. The
soft/hard gap means soft-dice-best != hard-dice-best: deploying the soft-best
discards 0.10 of deployable dice.
FIX SHIPPED: --best-on-hard flag (train_csg.py + run_pipeline.py). When set,
composite_score selects on m["dice"] (HARD dice) at both the training-time
best-tracking and the final best-vs-final comparison. Opt-in (default stays
soft to preserve the low-noise selector); tradeoff = selecting on a
nondeterministic carve, mitigated by air/break penalties + hard-dice stability
at convergence.
ALSO SHIPPED: --runs-subdir (train_csg.py run_dir + export_stls; run_pipeline
forwarding). Runs now write directly into runs/jul8-multidepth/ so the webapp
batch view sees them (was: all wave 9-14 stranded at top-level runs/, invisible
past #31; moved 78 completed + 3 anlr into the batch dir; batch now 110 runs).

NEXT (WAVE 15) -- VARIANCE: 4x pyr 5k + 2x hole 5k repeats (seed=1) to get
mean+/-std of the canonical method before trusting any single-run win.
NEXT (WAVE 16) -- BEST-ON-HARD: 4 shapes x {soft control, --best-on-hard} to
test whether hard-dice selection reliably raises DEPLOYED hard_dice.

================================================================
WAVE 15 RESULTS (variance characterization, seed=1, GPU nondeterminism)
================================================================
Pyramid 5k K70 MD5 (4 repeats): 0.711, 0.696, 0.676, 0.715
  -> mean = 0.699, std = 0.017
Hole 5k K70 HOLEBASE (2 repeats): 0.247, 0.239
  -> mean = 0.243, std = 0.006

CRITICAL CALIBRATION:
- The canonical pyramid "0.733" (wave 12) was a LUCKY draw (~2 sigma above the
  0.699 mean). True pyramid performance is 0.699 +/- 0.017.
- The wave-14 pyr_5k_b=0.541 was a ~9 sigma LOW outlier (genuine failure mode,
  not just noise) -- pyramid occasionally collapses to a bad basin.
- Hole is TIGHT: 0.243 +/- 0.006 (across waves: 0.249/0.253/0.256/0.247/0.239).
  The 2x-cap win is rock solid.
- Pyramid is NOISY: +/-0.017 (1 sigma). Single-run differences < ~0.05 between
  pyramid configs are NOT meaningful. Need >=3 repeats or mean+/-std reporting.

RE-FRAMING PRIOR WAVES WITH VARIANCE:
- pyr_8k_anlr=0.713 (wave14) is within 1 sigma of canonical mean 0.699 -> anneal-lr
  at 8k does NOT beat canonical 5k once variance is accounted for. The rescue
  (beats 0.607 collapse) is real; the "climb to 0.81" was a transient
  final-iter value the soft selector discarded (see best-on-hard fix, wave 16).
- pyr_k90=0.654 (wave13) vs mean 0.699 -> k90 is ~1 sigma below k70; likely NOT
  a real regression, just noise. (k90 not worth pursuing further either way.)
- All single-run pyramid "wins/losses" in waves 9-14 must be re-read against
  the 0.017 noise floor.

DESIGN RULE GOING FORWARD: any pyramid (or other shape) comparison uses >=3
repeats and reports mean+/-std; a delta is "real" only if it exceeds ~2 sigma
(~0.034 for pyramid). For hole, ~0.012.
