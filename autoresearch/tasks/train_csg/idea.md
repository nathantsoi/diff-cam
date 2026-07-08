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

### Wave 3 — engagement sweet spot (default sphere), LAUNCHED 17:10
Dense geometry (feed60 revs12) base; sweep multidepth_margin +0.01/0/-0.01/
-0.02/-0.03 (cut progressively closer/into the surface to drop residual), plus
margin0+wr3, margin-0.01+wr3 (drive residual-clearing via loss not gouge), and
margin-0.01+revs24. See launch_wave3.sh; sentinel wait_wave3.sh.
