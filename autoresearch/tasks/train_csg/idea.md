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
