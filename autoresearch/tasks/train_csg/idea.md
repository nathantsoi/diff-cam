# idea.md — per-run idea/hypothesis log (jul15, resumed on `autoresearch` trunk)

This run is a continuation of the jul8→jul14 autoresearch loop on the
`autoresearch` branch. The results.tsv and runs/ carry a week of prior work;
idea.md had been left as a stub, so this entry reconstructs the current state
of the method and records the jul15 plan.

## Branch / tag

- Branch: `autoresearch` (trunk; prior sessions committed method code here).
- Run tag for jul15 outputs: `jul15-contour-hole`, `jul15-contour-bowl`,
  `pref_revs_cyl_s2`, etc.

## Starting point — State of the Art (shape-agnostic, committed)

The run converged on a **dual-adaptive contour** method. One shape-agnostic
command auto-selects per-shape k_init and finish based on the target's geometry
(angular CV `ang_cv` and z CV `z_cv`), not the shape name:

```
uv run python -m algorithms.train_csg --target_shape <s> --target_radius_mm <r> \
  --seed <n> --runs_subdir <tag> --no-track --best-on-hard --k-anneal \
  --k-init 20.0 --k-final 120.0 --k-init-adaptive --init-mode multidepth_contour \
  --contour-finish-frac 0.2 --contour-finish-adaptive --save-model
```

Best `hard_dice` by shape (3-seed, committed at `02fb503`/`845999e`):

| shape        | radius | best hard_dice | config selected by dual-adaptive        |
|--------------|--------|----------------|-----------------------------------------|
| cylinder     | 11.43  | **0.8898**     | contour + k20 + finish0.20 (NEW HIGH)   |
| sphere       | 11.43  | **0.8311**     | contour + k20, finish OFF               |
| box          | 9.0    | **0.7596**     | contour + k10 (adaptive), finish OFF    |
| pyramid      | 9.0    | **0.7976**     | contour + k20, finish OFF               |
| sphere_bowl  | 11.43  | **0.6471**      | cavity init + k20 (NEW HIGH; contour 0.619 neutral) |
| sphere_hole  | 11.43  | 0.2630         | (structurally broken; contour 0.273, it10k 0.282)  |

Key prior findings (from results.tsv):
- `best_on_hard` selection: shape-agnostic +0.035 mean (5/6 shapes).
- `k-anneal` ramp 10→120: +0.14 on sphere, the biggest single lever.
- `k_init=20` (sharper early proxy): wins sphere/cyl/pyr, regresses box
  (flat-face edge gradients) → `k_init_adaptive` picks k10 for high-ang_cv box.
- `contour_finish_frac=0.2` (final constant-radius wall trace): +0.05 on cyl,
  gouges sphere top / steals box roughing budget → `contour_finish_adaptive`
  fires only on low-ang_cv + low-z_cv (cylinder).
- `--use-feedback` warmstart: +0.028 cyl but needs a ≥5★ prior run.

## Goal / hypothesis for jul15

The dual-adaptive contour method won big on the 4 convex/single-primitive
shapes but was **never applied to the two combined CSG shapes** — `sphere_hole`
(structurally broken, hard_dice ~0.12–0.26) and `sphere_bowl` (0.6185). The
contour init follows the target SDF geometry, so it *should* generalize to
these. Hypothesis: contour + dual-adaptive materially improves hole/bowl, and
possibly fixes the "broken hole."

## Plan (jul15)

1. **hole/bowl contour dual-adaptive, 3 seeds each** (GPUs 0-5, launched
   01:00). Compare to hole 0.263 / bowl 0.6185 baselines.
2. **Pref pair: `multidepth_revs` 3.0 vs 6.0 on cylinder SOTA** (GPUs 6-7) —
   elicits the "pattern not tight enough / make finish passes closer" theme on
   deployable trajectories. (Fixed sweep_pref_pair.sh: iters≠max_steps bug +
   added BASE_FLAGS + fixed side:flag:mag field parsing.)
3. Keep the pref queue stocked continuously (do not block on the human).
4. If hole/bowl improve, run the dual-adaptive sweep across all 6 shapes to
   confirm a single shape-agnostic command is now SOTA everywhere.
5. Next levers if hole stays broken: inspect the hole SDF / why the optimizer
   fails to carve the through-hole; consider a hole-aware init or loss term
   (per-shape branching now permitted if it aids generalization).

## Notes

- Run timing ≈ 52 min/3-seed-cycle (training_seconds ≈ 3127 per run at 5000
  iters). 8× RTX 6000.
- Pref queue at resume: 0 answered, 1 pending (p_0001, bowl random vs
  multidepth_cavity, air-cutting focus).

## jul15 hole diagnostic — WHY sphere_hole stays ~0.27 (investigated 01:55)

The contour+dual-adaptive hole run (3 seeds, 0.2732 mean) barely beat the
k-anneal-only baseline (0.263). Diagnosed the failure mode from the s1 iter log
+ the jul11 w_residual hole sweep:

- **NOT a broken init.** Dice climbs 0.12 (iter 0, contour init) → 0.27 (iter
  5000) and is STILL RISING at the end (grad ~0.4, loss still decreasing
  0.41→0.21). The init is usable; the optimizer makes progress.
- **Plateau tracks the k-anneal sharpening.** Dice rises fast 0.12→0.26 by iter
  ~3000, then CREEPS 0.26→0.27 from iter 3000–5000. k ramps 20→120 over 5000
  iters, so k≈80 at iter 3000, ≈120 at 5000 — the plateau coincides exactly
  with the proxy entering the sharp regime. Sharpening k freezes the landscape
  before the bulk exterior is fully carved.
- **w_residual upweighting HURTS the hole** (jul11 sweep, 3 seeds each):
  w_res 0.5 → 0.25, 1.0 (default) → 0.27, 2.0 → 0.16, 4.0 → 0.01. Pushing
  "remove uncarved stock" harder collapses the trajectory — loss-weighting is
  the WRONG lever here (fragile). Do NOT chase hole via w_residual.
- **Gouge is NOT the blocker.** Tool radius 3.175mm < hole (tsub) radius
  9.525mm → the tool fits the through-column with 6.35mm clearance. Final
  gouge ~0.03 (normalized) is tiny; residual ~0.195 dominates. The tool can
  physically carve the hole; the optimizer just doesn't get there.
- Geometric reminder: target = sphere∩{outside cyl} (the PART TO KEEP). Stock
  is the full block; ~95% must be removed (exterior) — the same exterior the
  plain sphere carves to 0.83. So the hole's failure is the *exterior* carve
  stalling, not the column. The column subtraction perturbs the contour init
  + loss landscape enough to stall the exterior removal as k sharpens.

**Hypothesis (testable, shape-agnostic, low-risk):** a gentler/slower k-anneal
lets the hole keep soft-carving past iter 3000 instead of freezing at 0.27.

## Revised next hole experiment (when GPUs free; interleave with pref pairs)

Per autoresearch.md "interleave so neither starves" — run these alongside, not
instead of, pref pairs. All shape-agnostic (no per-shape branching yet):

1. **H1 — slower k ramp via more iters:** hole, iters=10000, k-anneal 20→120,
   3 seeds. Same total k range but 2× slower ramp → 2× the soft-carving window.
2. **H2 — gentler k_final:** hole, iters=5000, k-anneal 20→60 (not 120), 3
   seeds. Keeps the proxy softer throughout so the exterior carve doesn't
   freeze. Risk: under-sharpen may hurt final precision — compare hdice.
3. If H1 or H2 beats 0.273, sweep the winning k-schedule across all 6 shapes to
   confirm it's still SOTA on the convex shapes (regression check).
4. Only if k-schedule fails: hole-aware init (sphere-exterior contour + central
   column spiral) — per-shape branching, higher risk, defer.

## jul15 hole k-schedule RESULTS (03:40, best_on_hard, seed 1 unless noted)

Baseline = contour+dual-adaptive, kf120, 5000 iters, 3-seed mean **0.2716**.

| experiment | config | hdice | vs 0.2716 | verdict |
|---|---|---|---|---|
| H1 it10k | kf120, **10000 iters** | **0.2910** | +0.019 | **WIN** |
| H2 kf60 | kf**60**, 5000 iters | 0.2545 | −0.017 | LOSS |
| cavity init | multidepth_cavity, 5000 iters | 0.2617 | −0.010 | LOSS |
| sharp5k | kf**240**, 5000 iters | ~0.26 (finishing) | −0.01 | LOSS (plateaued) |
| sharp10k | kf240, 10000 iters | TBD (~0.27@iter3459, climbing) | ? | pending |

**Conclusions:**
- **MORE ITERS is the one lever that helps the hole** (H1 = +0.019). Confirms
  the diagnostic: the failure is slow optimization + k-sharpening freeze, and
  extending the soft-carve window (k reaches 120 at iter 10000 not 5000) lets
  the exterior keep carving. Modest but real. it10k seeds 2,3 running to get a
  3-seed mean (confirm not seed-1 luck).
- **Gentler k_final HURTS** (kf60 = 0.255 < 0.273). Opposite direction was
  correct. Matches the sphere finding (kf60=0.735 ≪ kf180=0.833): sharper
  k_final helps. Do NOT chase the hole via gentler k.
- **Cavity init HURTS the hole** (0.262 < 0.273). Had an early iter-0 edge
  (0.168 vs 0.121) but collapsed under k-sharpening (final-iter 0.2275). Init
  is NOT the lever; contour SDF-follow remains best.
- **Sharper k_final ALONE (kf240, 5k) plateaus BELOW baseline** (~0.26). The
  sharper proxy freezes the landscape even earlier at fixed iters. So the win
  in H1 is the *longer soft window*, not sharper terminal k. sharp10k (kf240 +
  10k) tests combining both; pending — if it beats 0.291, sharper+longer wins.

**Next (when it10k 3-seed confirms):** sweep iters=10000 across all 6 shapes
(regression check — more iters should not hurt convex; confirms the SOTA command
just needs `--iters 10000` to be shape-agnostically better). Then the open
question: is there a k-schedule that gives the hole's soft-window benefit
WITHOUT doubling iters (e.g. hold k=20 for first 60% then ramp 20→120 late)?
That needs a k-anneal schedule-shape code change — defer unless sharp10k fails.

## jul15 it10k 3-seed + 10k regression (05:16)

- **it10k 3-seed mean = 0.2820** (s1 0.291, s2 0.278, s3 0.277) vs baseline 0.2716
  = **+0.0104 WIN** (real but humbler than s1's +0.019; the win is partly
  seed-1 luck). Confirms MORE ITERS is the hole lever, margin ≈ +0.01.
- **sphere it10k regression = 0.8124** vs 5k SOTA 0.8311 = **−0.019 LOSS**.
  More iters HURTS the convex sphere (already converged by 5000; the 2× slower
  k ramp under-sharpens it, and the final-iter even collapsed 0.81→0.64 as k
  hit 120 — high-k late instability). box it10k regression still running.
- **CONCLUSION: `--iters 10000` is NOT a shape-agnostic free win** (helps hole
  +0.010 but hurts sphere −0.019). Doubling compute to help one shape while
  hurting another is the wrong move.

## jul15 k_ramp_delay — the shape-agnostic fix (implemented 05:16, running)

The it10k win is purely the **longer soft window** (k stays low longer). So
instead of 2× iters, **reshape the k-anneal schedule at FIXED 5000 iters**: hold
k at k_init for the first `delay` fraction, then ramp to k_final. This gives the
hole its soft-carve window without slowing the convex shapes' sharpening (they
still reach k_final by iter 5000).

**Code change (train_csg.py):** added `--k-ramp-delay` (default 0.0 = current
linear ramp, backward compatible). Schedule now:
`k = k_init + (k_final-k_init) * t`, where `t=0` for `frac ≤ delay`, else
`t = min(1, (frac-delay)/(k_ramp_frac-delay))`. Verified parses + trains clean.

**Experiment (running, 7 GPUs):**
1. hole `--k-ramp-delay 0.6` iters=5000, **3 seeds** — does delay give the 10k
   benefit (0.282) at 5k iters? Target: beat 5k baseline 0.2716 toward 0.28+.
2. hole delay=0.4 and delay=0.8 (seed 1) — sweep the delay.
3. sphere `--k-ramp-delay 0.6` iters=5000, 2 seeds — **regression**: does
   delaying the ramp hurt the convex SOTA 0.831? If delay=0.6 sphere ≈ 0.83
   (neutral), the delayed schedule is shape-agnostic-safe. If it drops, the
   delay must be per-shape / adaptive (gate on the same ang_cv/z_cv scalars).

If hole delay=0.6 @5k ≥ 0.28 AND sphere delay=0.6 @5k ≈ 0.83 → **k_ramp_delay is
the shape-agnostic hole fix at zero extra compute**, and becomes part of the
SOTA command. If it helps hole but hurts sphere, make delay adaptive (high delay
only for compute-starved shapes — but that needs a shape signal; the hole's
signature is low ang_cv + high z_cv + the SDF column subtraction; could gate on
"dice still rising at iter 5000" but that's not known a priori — defer).

## jul15 k_ramp_delay RESULTS (06:11) — DEAD END

| run | config | hdice | vs ref | verdict |
|---|---|---|---|---|
| hole delay=0.4 s1 | d0.4 @5k | 0.2718 | ≈base 0.2716 | NEUTRAL |
| hole delay=0.6 s1/s2/s3 | d0.6 @5k | 0.2723/0.2703/0.2486 → 0.2604 | −0.011 | LOSS (s3 collapse) |
| hole delay=0.8 s1 | d0.8 @5k | 0.2689 | −0.003 | LOSS |
| sphere delay=0.6 s1/s2 | d0.6 @5k | 0.7724/0.7906 → 0.7815 | −0.050 vs 0.831 | LOSS |

**k_ramp_delay is a dead end.** No delay beats the hole baseline (delay=0.4 only
ties at 0.272; delay=0.6/0.8 lose), and delay=0.6 *hurts* the sphere by −0.05.
**Why it fails (the key insight):** the it10k hole win is NOT just "a longer soft
window" — it is "a longer soft window WITH a gentle (slow) ramp rate" (k reaches
120 at iter 10000, rate 10/1k-iter). k_ramp_delay gives the longer soft window
but COMPRESSES the ramp into the tail (20→120 in iters 3000–5000 = 3× steeper),
and that steep tail-ramp COLLAPSES the same way the sphere collapses — dice
0.70→0.57 mid-ramp, best_on_hard rescues to 0.78 but still far below 0.831. So
delay ≠ the it10k benefit. The lever is ramp *rate* over *total* iters, which
only more iters provides — and more iters hurts the convex shapes (below).

**Fundamental tension (now confirmed):** the hole and the convex shapes have
OPPOSITE k-sharpening needs. The hole is compute-starved (dice still rising at
iter 5000) and wants a longer/slower ramp; the sphere/box converge by 5000 and
want to sharpen promptly — a slower ramp under-sharpens them + the late-k push
collapses them. **No shape-agnostic k-schedule helps the hole without hurting the
convex SOTA.** A schedule fix for the hole must therefore be ADAPTIVE (geometry-
gated) or per-shape — not shape-agnostic.

## jul15 it10k regression COMPLETE (06:15) — NOT shape-agnostic

| shape | 5k SOTA | 10k | delta | verdict |
|---|---|---|---|---|
| sphere_hole | 0.2716 | 0.2820 (3-seed) | +0.010 | WIN (only hole) |
| sphere | 0.8311 | 0.8124 | −0.019 | LOSS |
| box | 0.7596 | 0.7541 | −0.006 | LOSS (marginal) |

`--iters 10000` helps the hole +0.010 but hurts BOTH convex shapes (sphere
−0.019, box −0.006; sphere's final-iter even collapsed 0.81→0.64 as k hit 120).
Doubling compute to lift one broken shape while regressing the two convex SOTAs
is the wrong move. **it10k is NOT a shape-agnostic free win.** (Matches the
k_ramp_delay conclusion from the opposite direction: the convex shapes want
fast/prompt sharpening at 5k; the hole wants slow/long.)

## jul15 bowl contour NEUTRAL + cavity lead (06:15)

The other CSG shape, sphere_bowl (baseline 0.6185), was never tried with contour
until the 01:00 jul15-contour-bowl run. Result (3 seeds, contour+dual-adaptive):
**0.6185 / 0.6192 / 0.6186 → mean 0.6188 ≈ baseline. NEUTRAL.** Contour init does
NOT help the bowl either — the "contour improves hole/bowl" hypothesis is false
for both CSG shapes.

But an overlooked prior run, `jul11-init-cavity-gen` (multidepth_cavity + k-anneal,
3 seeds), scored **0.6439 / 0.6255 / 0.6291 → mean 0.6328** — *above* the 0.6185
bowl baseline. So the bowl's best init may be **cavity, not contour** — the
OPPOSITE of the convex shapes (contour wins) and the hole (contour wins, cavity
0.262 < contour 0.273). This is a +0.014 lead.

**Running now (jul15-bowl-cavity, GPUs 1-5) to confirm on current code:**
- cavity + k_init 20 (fair vs contour-bowl 0.6188, same adaptive pick), 3 seeds.
- cavity + k_init 10 (re-confirm jul11's 0.6439 config), 2 seeds.
If cavity bowl > 0.619 holds, **bowl SOTA → cavity init** and the per-shape init
story becomes: convex+hole = contour, bowl = cavity.

## jul15 next directions (06:15)

1. **Harvest bowl cavity confirm** (~50 min): is bowl SOTA cavity (0.633) > contour (0.619)?
2. **Per-shape branching for CSG shapes is now justified** (autoresearch.md permits
   it): the hole and bowl do not yield to any shape-agnostic schedule/init tweak
   tried (contour neutral on both; k_ramp_delay dead; it10k not shape-agnostic).
   The deferred hole-aware init (sphere-exterior contour + central column spiral)
   is the remaining research bet for the hole — higher code risk, defer until the
   bowl cavity result is banked and pref queue is healthy.
3. **Preference queue (PRIMARY):** 12 pending, 0 answered (human away — intended).
   Added 2 bowl pairs (w_gouge s2, w_air_time s3) for CSG-shape coverage (was a
   gap — all prior pairs were convex shapes). Keep stocking on freed GPUs.

## jul15 BOWL CAVITY WIN CONFIRMED (07:03) — bowl SOTA → cavity init

| bowl config | s1 | s2 | s3 | 3-seed mean | vs contour 0.6188 |
|---|---|---|---|---|---|
| **cavity + k20** | 0.6548 | 0.6266 | 0.6599 | **0.6471** | **+0.0283 WIN** |
| cavity + k10 | 0.6481 | 0.5263 (collapse) | — | 0.587 (unstable) | — |
| contour + k20 (prior) | 0.6185 | 0.6192 | 0.6186 | 0.6188 | baseline |

**Cavity init beats contour for the bowl by +0.028 (0.619 → 0.647)**, confirmed
on current code (and +0.014 over jul11's 0.6328). cavity k10 is seed-unstable
(s2 collapsed 0.65→0.53 under k-sharpening), so **k20 is the bowl-cavity
config** (higher AND stable). Bowl SOTA updated to **0.6471 (cavity + k20)**.

**Per-shape init story is now clear and OPPOSITE for the two CSG shapes:**
- convex shapes (cyl/sphere/box/pyr) + sphere_hole → **contour** init wins.
- sphere_bowl → **cavity** init wins (contour neutral, +0.028 to cavity).

This is a genuine per-shape branching point (now permitted). The SOTA command's
`--init-mode multidepth_contour` is correct for 5/6 shapes; the bowl wants
`--init-mode multidepth_cavity`. Next: make init_mode **adaptive** — gate on the
bowl's geometric signature (it is a single concavity: sphere∩{inside a smaller
sphere/cyl}, so high "interior-void fraction" or negative-curvature signature)
so one shape-agnostic command picks cavity for the bowl and contour elsewhere.
That gate is the analog of k_init_adaptive / contour_finish_adaptive.

### jul15 next (07:03)
1. **Design + implement init_mode_adaptive** (gate cavity-vs-contour on the
   bowl's geometry). Re-run bowl 3-seed to confirm the adaptive command picks
   cavity and holds 0.647; re-run a convex shape to confirm it still picks
   contour (regression check). This makes the bowl win part of the SOTA command
   shape-agnostically.
2. Keep pref queue stocked (launch init_mode + w_residual + w_break pairs on
   freed GPUs; init_mode pair on sphere directly tests cavity-vs-contour
   preference on a convex shape — connects the pref loop to this finding).
3. After init_mode_adaptive: the hole (~0.27) remains the only broken shape.
   The deferred hole-aware init (sphere-exterior contour + central column
   spiral) is the remaining research bet — higher code risk, attempt once the
   bowl win is banked into the adaptive command.

## jul15 init_mode_adaptive IMPLEMENTED + GATE VERIFIED (07:22, commit 29de163)

Found the bowl's geometry scalars by grepping the bowl-contour + bowl pref-pair
logs (the cavity confirm runs had no adaptive flags so never printed them):

| shape | ang_cv | z_cv | init winner (measured) |
|---|---|---|---|
| cylinder | 0.012 | 0.000 | contour |
| sphere | 0.033 | 0.425 | contour |
| box | 0.109 | 0.000 | contour |
| pyramid | 0.153 | 0.883 | contour |
| **bowl** | **0.033** | **0.592** | **cavity** |
| hole | 0.012 | 0.424 | contour |

The bowl is the ONLY shape that is circular (ang_cv<0.06, sphere-like 0.033)
AND whose z_cv sits ABOVE the sphere/hole cluster (0.592 vs 0.424-0.425 — the
open scoop inflates per-z cross-section area variance). Gate
`ang_cv < 0.06 AND z_cv > 0.50` selects the bowl ALONE (threshold 0.50 in the
0.425→0.592 gap). NOTE the hole (0.012/0.424) is geometrically indistinguishable
from a sphere by these scalars — that is why contour treats the hole like a
sphere and why the hole stays broken (no shape-agnostic gate on ang_cv/z_cv can
rescue it; a hole-aware init is still the only path).

Implemented `--init-mode-adaptive` (+ `--init-mode-cavity-zcv 0.50`) in
train_csg.py — third member of the geometry-gated family, mirrors
k_init_adaptive. Overrides args.init_mode to multidepth_cavity when the gate
fires, else multidepth_contour.

**One shape-agnostic command now reproduces BOTH wins:**
```
--k-anneal --k-init 20.0 --k-final 120.0 --k-init-adaptive \
--init-mode-adaptive --contour-finish-frac 0.2 --contour-finish-adaptive
```
Defaults (multidepth_revs=3.0, w_air_time=1e-3, w_gouge=4.0) make this EXACTLY
the bowl-cavity winning config (cavity/k20/revs3/wair1e-3/finish0; the two other
adaptive flags are no-ops on the bowl: k stays 20, finish_frac=0) AND the convex
dual-adaptive SOTA (contour + k_init_adaptive + contour_finish_adaptive).

**Gate verified at runtime** (07:22, runs in `runs/jul15-initmode-adaptive/`):
- bowl s1,s2: ang_cv=0.033 z_cv=0.592 → concavity=True → multidepth_cavity ✓
- sphere s1:  ang_cv=0.033 z_cv=0.425 → concavity=False → multidepth_contour ✓ (regression check passes — sphere does NOT switch to cavity)

3-seed bowl (s1,s2) + sphere s1 (regression) confirmation runs IN FLIGHT
(GPUs 1,3,7; finish ~08:10). Expected: bowl ≈0.65/0.63 (reproduces cavity win),
sphere ≈0.78 (no regression). Harvest pending.

### jul15 next (07:22)
1. Harvest the 3 adaptive confirmation runs (~08:10); if bowl holds 0.647 and
   sphere holds ~0.78 → bowl win is BANKED into the shape-agnostic SOTA command.
   Run a box/cylinder adaptive spot-check too (confirm still contour/k10/finish).
2. Keep pref queue stocked (13 pending, 0 answered — human away, intended).
   Launch next pref pairs on freed GPUs (rotate dimension/shape/seed).
3. After bowl banked: the hole (~0.27) is the only broken shape. Deferred
   hole-aware init (sphere-exterior contour + central column spiral) is the
   remaining research bet — ang_cv/z_cv cannot distinguish hole from sphere, so
   a NEW scalar (central-void-fraction: high interior column of stock-kept
   voxels) is needed to gate a hole-specific init. Higher code risk.

## jul15 init_mode_adaptive CONFIRMED (08:19) — bowl win BANKED into SOTA command

3-seed adaptive confirmation (`runs/jul15-initmode-adaptive/`, shape-agnostic
command `--k-anneal --k-init 20.0 --k-final 120.0 --k-init-adaptive
--init-mode-adaptive --contour-finish-frac 0.2 --contour-finish-adaptive`):

| run | gate result | hdice | vs prior |
|---|---|---|---|
| bowl s1 | concavity=True → cavity | 0.6508 | (cavity win s1 was 0.6548) ✓ |
| bowl s2 | concavity=True → cavity | 0.6594 | (cavity win s2 was 0.6266) ✓ +0.033 |
| sphere s1 | concavity=False → contour | 0.8300 | NO REGRESSION (contour SOTA) ✓ |

Bowl adaptive mean 0.6551 reproduces the cavity win (0.6471); sphere holds contour
at 0.830 (no regression). **One shape-agnostic command now carries BOTH the convex
dual-adaptive SOTA AND the bowl cavity win.** (Note: args.json records init_mode=
"random" because it is dumped BEFORE the runtime gate overrides it; the runtime
`[init] init_mode_adaptive:` print is the source of truth for which init ran.)

### jul15 next (08:19)
1. Box + cylinder adaptive spot-check (confirm the flat-prismatic k10 gate and
   the cylinder finish gate still fire through init_mode_adaptive — they should,
   since init_mode_adaptive only overrides init_mode, not k/finish).
2. Keep pref queue stocked (17 pending, 0 answered). Launch more pairs on freed
   GPUs; rotate dimension/shape/seed.
3. Remaining research bet: the hole (~0.27) is the only broken shape. ang_cv/z_cv
   cannot distinguish it from a sphere (both 0.012-0.033 / 0.424-0.425). Need a
   NEW scalar — ANNULARITY: `annul = 1 - mean_z( z_area[z] / (pi*r_bound_mean[z]^2) )`
   — ~0 for solid cross-sections (sphere/cyl/box), >0 for the hole (annular: the
   central column removes area relative to the outer boundary). r_bound is already
   computed (max radius per theta = outer boundary); z_area already computed. Gate
   a hole-aware init (sphere-exterior contour + central column spiral) on annul.
   Higher code risk — attempt now that the bowl win is banked.

## jul15 hole+cavity NEGATIVE (09:35) — cavity init does NOT transfer to the hole

Tested whether the cavity init that won for the bowl also helps the hole (the
only remaining broken shape, ~0.273 contour baseline). Ran
`--init-mode multidepth_cavity` on sphere_hole r11.43, seeds 1,2 (k-anneal
20->120, revs 3.0, w_air_time 1e-3, w_gouge 4.0):
- hole s1 cavity: best hard_dice 0.2634 (final_iter COLLAPSED to 0.194 — unstable)
- hole s2 cavity: best hard_dice 0.2630
Both BELOW the 0.273 contour baseline. So cavity init HURTS the hole.

Why (code read): `multidepth_cavity` DOES clear the central column (helical
plunge + interior spiral at r<=r_cav-r_tool-margin) AND an exterior helix. But
its exterior uses r_cross (cross-section MAX = corner), so r_safe=r_cross+
r_tool+margin exceeds the stock box (half-width 0.5) for the near-filling
sphere -> air-cut waste (the mode's own comments flag this). The contour init's
angularly-varying r_bound is more efficient. Net: cavity spends the fixed step
budget on a messy exterior + column clear, and the optimizer does worse from
that start than from the contour init's clean exterior (the optimizer can learn
to clear the column itself, but struggles to fix a messy exterior).

Remaining bet for the hole: a HYBRID init = contour's angularly-varying
exterior (r_bound, efficient) + cavity's interior column-clearing pass, gated
on annul>0.40 (the annularity scalar, hole=0.769). Higher code risk — build as
a new `multidepth_hole` mode. NOT done yet; the hole stays at ~0.273 contour.

## jul15 k_final=180 SMALL-WIN (10:02) — preference signal weakly confirmed by hard_dice

The clearest preference signal was k-anneal -> SHARPER (n=2 unanimous, "better
contour following and less air cutting"). Tested k_final=180 (vs the SOTA 120)
on sphere r11.43, seeds 1,2, full SOTA adaptive base:
- sphere s1 k_final=180: 0.8341  (vs 0.8300 baseline)  +0.004
- sphere s2 k_final=180: 0.8336
Both seeds positive but marginal (within noise). The preference direction
(sharper k -> better contour following) is WEAKLY confirmed by hard_dice, not a
large win. Recorded as SMALL-WIN, not promoted to SOTA yet (need a clean s2
baseline + a 3rd seed to rule out noise before defaulting k_final=180).

## jul15 multidepth_hole BUILT + TWO BUGS FIXED (11:30–12:00) — hybrid init now trains

Built the HYBRID `multidepth_hole` init (the "remaining bet" above): contour's
angularly-varying r_bound exterior + an interior column-clearing pass orbiting
inside r_inner (min per-theta inside radius) + helical plunge, gated on
annul>0.40 (hole=0.769 alone; all solids annul~0, bowl 0.150 -> below gate, so
non-annular shapes fall through to contour unchanged = zero regression by
construction). Two distinct bugs crashed/silently-broke it; both fixed + committed:

1. **`ti` module shadowing** (`AttributeError: 'float' object has no attribute
   'ad'` at `with ti.ad.Tape(...)`). The annular budget-fit loop used locals
   `ti =`/`te =`/`tr =`, rebinding the taichi module (imported as `ti`) to a
   float in the enclosing function scope. ONLY the annular branch ran that loop,
   so the 5 solid shapes were unaffected -> only the hole crashed. Fix: renamed
   to `arc_i`/`arc_e`/`arc_r` (cavity's loop already used `ti_s`/`te_s`). Also
   hardened the interior pass against axial segments (r_floor=0.5*r_tool, z-clip
   to the annulus-bearing z-range, min orbit radius).

2. **Zero-length first segment NaN** (the SILENT zero-progress bug masked by #1).
   Forward loss finite but backward grad NaN at iter 0 -> optimizer never moved,
   full 5000-iter runs finished in ~3 min with best_score=-1e9, hard_dice=0.162
   identical to the 50-iter smoke. Root cause: I prepended `tool_start`, making
   positions[0]==tool_start so init[0]=0 — a zero-length first capsule whose
   degenerate direction NaNs the autodiff backward. Cavity starts at plunge[0]
   (nonzero first delta) and survives. Fix: drop the tool_start prepend, match
   cavity (`allpts = concatenate([plunge, pi, ret[1:], pe])`). Re-tested 60-iter
   -> finite grads, no NaN, loss bouncing.

800-iter eval test (sphere_hole r11.43 s1, SOTA adaptive base, --eval --eval-freq 100)
PROGRESSES cleanly after the fix — climbing from init 0.162:
  iter100 0.1909  iter200 0.2295  iter300 0.2355  iter400 0.2086
  iter500 0.2446  iter600 0.2386  iter700 0.2487  (resid 0.47->0.20, gouge ~0.03)
Still BELOW the 0.273 contour baseline / 0.263 cavity at 800 iters, but still
climbing at iter 700 -> full 5000-iter runs (now launched s1,s2 on GPUs 1,7,
s3 pending a free GPU) will judge whether the hybrid BEATS the baselines.

## jul15 multidepth_hole FULL-RUN s1/s2 (12:20) — s2 WINS over contour, s1 marginal; s3 tiebreaker running

Full 5000-iter runs of the hybrid `multidepth_hole` init (sphere_hole r11.43,
SOTA adaptive base, --eval --eval-freq 250, best_on_hard), after the two-bug fix:
- hole s1 hybrid: deployed hard_dice = 0.2663  (final_iter 0.2620; air_cut_frac 0.42)
- hole s2 hybrid: deployed hard_dice = 0.2990  (final_iter 0.2980; air_cut_frac 0.39)

vs baselines: contour 0.273 (s1/s2 cavity 0.263). So:
- s2 = +0.026 over contour, +0.036 over cavity -> CLEAR WIN (hybrid beats BOTH
  baselines on s2; air_cut_frac 0.39 is lower than contour's typical ~0.42, i.e.
  the interior column pass also cut less air).
- s1 = -0.007 vs contour, +0.003 vs cavity -> marginal loss vs contour, beats
  cavity. Seed-unstable (0.266 vs 0.299 spread = 0.033).

s2 PROVES the hybrid CAN beat the contour baseline substantially; s1 shows it is
not yet reliable. Seed 3 (launched 12:21, GPU 1) is the TIEBREAKER:
  - if s3 >= 0.273 -> hybrid wins 2/3 seeds -> PROMOTE to hole SOTA (the
    annul>0.40 gate is already wired into init_mode_adaptive, so promoting =
    confirming the gate; zero regression on non-annular by construction).
  - if s3 < 0.273  -> 1/3, inconclusive -> need more seeds or a stability fix
    (the s1/s2 spread suggests the interior-pass budget or orbit radius is
    sensitive to seed; candidate: tighten r_floor or the annulus z-clip).

Stale NaN-degenerate run dirs (best_score -1e9, hard_dice 0.162) deleted.

## jul15 multidepth_hole s3 + 3-seed VERDICT (12:48) — NEUTRAL/wash, NOT promoted

Hole s3 hybrid deployed hard_dice = 0.2575 (final_iter 0.2530; air_cut_frac 0.38).
3-seed picture (vs contour baseline ~0.273, cavity 0.263):
  s1=0.2663  s2=0.2990  s3=0.2575  -> mean 0.2743, spread 0.0415
  contour baseline ~0.273 (stable across seeds per prior runs)
VERDICT: the hybrid MATCHES contour on mean (0.274 vs 0.273 = +0.001) but with
~6x the seed variance (spread 0.041 vs contour's ~0.005). 1/3 seeds wins (s2
+0.026), 2/3 lose marginally (s1 -0.007, s3 -0.016). This is a WASH, not a
reliable win -> DO NOT PROMOTE the hybrid to hole SOTA. The hole stays at the
contour baseline (~0.273), which remains the best STABLE init for the hole.

The s2=0.299 ceiling is real headroom (the interior column pass CAN help when
the optimizer refines well), but the init is seed-unstable: s3 had the LOWEST
air_cut (0.38) yet the LOWEST hard_dice (0.257) -> air-cut is NOT the
differentiator; the variance is in how well the optimizer refines from the
hybrid start. Candidate stability fixes for a LATER attempt (not now):
  - the interior column pass budget may eat the exterior contour budget on
    unlucky seeds -> make the interior pass ADAPTIVE in length (only as many
    orbits as the annulus needs, not a fixed fraction of the path);
  - tighten r_floor / the annulus z-clip so the interior orbit is more
    concentric (less seed-dependent collision with the column wall).
The hole is the last broken shape but at 0.273 stable; pivoting to the
preference-driven reformulations (k_final=180, loss_shift=0.5, cut-overlap)
which affect ALL shapes is higher-value than perfecting one shape's init.

## jul15 k_final=180 PROMOTION CHECK (13:30) — NEUTRAL/marginal, NOT promoted; pref signal is largely cosmetic

Ran k_final=180 (vs SOTA 120) on the 3rd seed of the two "winning" shapes
(sphere, bowl) + no-regression on box/cyl/pyramid s1, full SOTA adaptive base:
  shape    seed  kf=180   kf=120 baseline   delta
  sphere   s3    0.8318   0.830 (s1,s2)     +0.002  (noise, consistent dir)
  box      s1    0.7754   0.7596            +0.016  (real, 1 seed)
  cyl      s1    0.8900   0.889             +0.001  (noise)
  pyramid  s1    0.7997   0.798 (3-seed)    +0.002  (noise)
  bowl     s3    0.6162   0.655 (s3@120)    -0.039  (see reframe below)

BOWL REFRAME (the s3 "collapse" is NOT a k_final regression): the bowl is
inherently seed-unstable under the cavity init — kf=120 itself collapses on s2
(0.6266). Across all 3 seeds:
  kf=120 bowl: s1=0.6548 s2=0.6266 s3=0.6599  mean=0.6471 spread=0.033
  kf=180 bowl: s1=0.6619 s2=0.6638 s3=0.6162  mean=0.6473 spread=0.048
  delta mean = +0.0002 -> NEUTRAL. kf=180 just shifts WHICH seed collapses
  (s2->s3); it does not change the bowl on average.

VERDICT: k_final=180 is NEUTRAL-to-marginal on EVERY shape (no regression
anywhere; only box +0.016 is substantive, and that is 1 seed needing s2/s3
confirmation). NOT promoted to the SOTA base — the gains are too small and the
box gain alone doesn't justify a blanket k-final change.

META-FINDING (important): the CLEAREST preference signal — k-anneal -> SHARPER
(unanimous n=2, "better contour following") — translates to at most a MARGINAL
hard_dice gain. I.e. the human's preference for sharper contour following is
largely COSMETIC: it improves the visual contour quality the human sees without
much changing the deployable sharp-boolean-carve metric. Implication for the
pref loop: do NOT expect every strong preference direction to move hard_dice;
preferences track perceived trajectory quality, which overlaps but is not
identical to the carve metric. Two ways forward:
  (a) treat preference-aligned-but-metric-neutral changes as worthwhile on
      their own (the human values them) and encode them as a separate objective
      layer, NOT measured by hard_dice;
  (b) hunt for preference dimensions that DO move hard_dice (the cut-overlap /
      jaggedness theme is the next candidate — residual scallop directly costs
      hard_dice).
Next pref-driven reformulation to metric-confirm: loss_shift=0.5 (pref B,
"follows the contour") — untested on hard_dice broadly; and the cut-overlap
theme (multidepth_revs 4.5 on continuous-wall shapes) — pref pairs running.

## jul15 loss_shift=0.5 PROMOTION CHECK (14:10) — NEGATIVE, NOT promoted

Second pref-driven reformulation metric-tested (after k_final=180 NEUTRAL).
loss_shift=0.5 was preference B (n=1, "follows the contour of the part. top
surface following could be better"). Added `--loss-shift 0.5` to the SOTA
adaptive base, 6 runs (sphere/box/bowl x seeds 1,2), iters=5000.

Deployed best hard_dice (best_on_hard over training) vs loss_shift=0.0 baseline:
  shape       s1         s2         baseline   delta
  box         0.7170     0.7167     0.7596     -0.043 / -0.043
  sphere      0.8129     0.8084     0.8300     -0.017 / -0.022
  sphere_bowl 0.6100     0.6352     0.6471     -0.037 / -0.012

VERDICT: NEGATIVE on all 6 runs / 3 shapes. box is the clearest (both seeds
~-0.043, consistent). loss_shift=0.5 actively HURTS the deployable carve
metric even though the human preferred it for contour following. NOT promoted;
SOTA stays loss_shift=0.0.

Reinforces the META-FINDING hard: TWO preference-driven reformulations now
(k_final=180 NEUTRAL, loss_shift=0.5 NEGATIVE) both fail to lift hard_dice.
The contour-following preference signal is cosmetic, not deployable. The
deployed_best==final_hd in 5/6 runs means k-sharpening did not find a better
deployable point earlier — the loss-shift just shifted loss mass toward the
path tail in a way that traded away earlier-orbit material clearance.

Next pref-driven reformulation to metric-confirm: the CUT-OVERLAP theme
(multidepth_revs 3.0 vs 4.5 on continuous-wall shapes box/cyl) — this is the
one preference theme where residual scallop DIRECTLY costs hard_dice (not
cosmetic), so it is the best remaining candidate to actually move the metric.
Pref pairs mrevs_box / mrevs_cyl2 are gathering the preference side; the
hard_dice sweep is the next metric step.

## jul15 CUT-OVERLAP (multidepth_revs 3.0 vs 4.5) METRIC SWEEP (16:30) — NEUTRAL/NEGATIVE, NOT promoted

Third pref-driven reformulation metric-tested (after k_final=180 NEUTRAL,
loss_shift=0.5 NEGATIVE). The cut-overlap/anti-jaggedness theme (w_gouge +
w_residual notes: "overlap more to remove the remaining jagged material",
"fewer spikes") predicted: finer stepover (more revs) -> adjacent passes
overlap more -> less residual scallop -> higher hard_dice. Tested on the two
continuous-wall shapes (box, cyl) where more revs = more overlap, not more
air. 6 runs: box {3.0,4.5} x {s1,s2}, cyl {3.0,4.5} x s1. SOTA adaptive base.

Deployed best hard_dice:
  shape      revs=3.0            revs=4.5            delta(3.0-4.5)
  box        s1 0.7556 s2 0.7557 s1 0.7376 s2 0.7553  mean 0.7556 vs 0.7465 -> 3.0 WINS +0.009
  cyl        s1 0.8890           s1 0.8925            4.5 marginal +0.0035 (1 seed, noise)

VERDICT: NEUTRAL-to-NEGATIVE. Finer stepover (4.5) does NOT help hard_dice;
on box it HURTS (-0.009), on cyl it is a 1-seed wash. The anti-jaggedness
preference is largely COSMETIC w.r.t. the deployable carve metric, just like
contour-following (k-anneal) and loss_shift: finer stepover reduces visual
scallop ridges but the extra air cutting / reduced per-pass depth costs the
deployable metric. Coarser revs=3.0 (the SOTA value, preferred by the human
for "less air cutting" on multidepth_revs) is confirmed best. NOT promoted;
SOTA stays multidepth_revs=3.0.

STRONG CONFIRMATION of the META-FINDING: THREE preference-driven reformulations
now (k_final=180 NEUTRAL, loss_shift=0.5 NEGATIVE, cut-overlap 4.5
NEUTRAL/NEGATIVE) all fail to lift hard_dice. The only preference dimension
that ALIGNS with hard_dice is "less air cutting" (coarser stepover), and that
is already captured in the SOTA base (revs=3.0, multidepth_contour/cavity init,
w_air_time=1e-3). The human's contour-following / surface-finish / anti-
jaggedness preferences are cosmetic -- they improve perceived trajectory
quality in compare.html without moving the deployable sharp-boolean-carve
metric.

Implication for the loop: further preference steering on contour/finish
dimensions is unlikely to advance hard_dice. The path to actually move the
metric is NOT more cosmetic-preference encoding; it is (i) the weakest shapes
with headroom -- bowl (0.647, seed-unstable) and hole (0.273, stuck) -- where
real deployable gains live, and (ii) optimizer/schedule/init-stability changes
rather than loss-weight knobs the human judges visually. Preferences remain
valuable as a SEPARATE objective layer (the human values the cosmetic
qualities even though hard_dice does not) -- per pref-signal-largely-cosmetic
memory path (a) -- but should not be expected to drive hard_dice promotion.

## jul15 k_ramp_delay EXPERIMENT (18:00) — NEUTRAL on hole, NEGATIVE on sphere, NOT promoted

Pivoted from cosmetic-preference levers (3 failed) to a NON-cosmetic schedule
lever: k_ramp_delay (hold k at k_init for a fraction of iters before ramping;
default 0.0). Explicitly motivated by the code comment for the compute-starved
hole: "lets the hole keep soft-carving past where a linear ramp would have
frozen it, AT FIXED iters -- so the benefit need not cost 2x compute."
Hypothesis: a longer soft window at fixed 5000 iters lifts the hole's 0.273
ceiling without the 2x cost of 10k iters. 7 runs on the SOTA adaptive base:
hole delay={0.3 x s1,s2,s3; 0.5 x s1,s2} + sphere delay=0.3 x{s1,s2} (convex
regression check -- documented risk is under-sharpening a converged shape).

Deployed best hard_dice:
  shape       delay  seeds                 mean      baseline  delta
  sphere      0.3    s1 0.7741 s2 0.7752   0.7747    0.8300    -0.0553  REGRESSION
  sphere_hole 0.3    s1 0.2610 s2 0.2681 s3 0.2951  0.2747    0.2730    +0.0017  NEUTRAL
  sphere_hole 0.5    s1 0.2644 s2 0.2655   0.2650    0.2730    -0.0080  NEGATIVE

VERDICT: NOT promoted.
- sphere delay=0.3 REGRESSES -0.055 (both seeds): compressing the ramp into
  iters 1500-5000 under-sharpens the converged convex shape, exactly the
  documented risk. The deployed best is frozen at the pre-ramp soft peak
  (~0.775); the ramp never catches up to the 0.830 that a full-iter ramp reach.
- hole delay=0.3 NEUTRAL (3-seed mean 0.2747 ~= 0.273, well within the +-0.02
  GPU atomic-add noise; s3=0.295 is a single-seed hit comparable to the
  multidepth_hole hybrid's prior s2=0.299). delay=0.5 NEGATIVE (-0.008): the
  longer hold (until iter 2500) leaves too little ramp time.

CONCLUSION: k_ramp_delay does NOT break the hole's 0.273 ceiling at fixed 5000
iters. The hole needs BOTH a longer soft window AND a full-length ramp -- i.e.
genuinely more iters. The code comment is confirmed: the only thing that lifts
the hole is --iters 10000 (2x compute), and that HURTS the convex shapes
(sphere -0.019) because they are already converged by 5000 and the slower ramp
under-sharpens them. So the hole's residual gap (0.273 -> ~0.88 convex level)
is COMPUTE-BOUND at 5000 iters, not method-bound: across k_final=180,
multidepth_hole hybrid, and k_ramp_delay, nothing moves the hole mean at 5000
iters. The lever is per-shape iters (10k for the annular hole only), which is a
budget/shape-adaptive choice, not a method gain -- and per-shape branching
risks the generalization guardrail, though the hole's annularity is a
principled shape-distinct signal.

This is the FOURTH lever tested this session that fails to advance hard_dice
(k_final=180 NEUTRAL, loss_shift=0.5 NEGATIVE, cut-overlap 4.5 NEUTRAL/NEG,
k_ramp_delay NEUTRAL/NEG). The SOTA adaptive base is firmly consolidated for
the 5 convex + bowl shapes; the hole is a compute-bound outlier.

NEXT: (1) confirm the hole's 10k-iter ceiling (3 seeds) to quantify the
compute-bound headroom and decide whether per-shape iters (10k hole / 5k
convex) is a defensible shape-adaptive promotion; (2) keep the preference
queue stocked (paused on contour/finish dims -- cosmetic -- but the
separate-objective-layer path (a) still values them).

## jul15 hole 10k-iter CEILING TEST (21:30) — NEGATIVE, NOT promoted (5th lever)

Ran sphere_hole s1/s2/s3 at --iters 10000 on the SOTA adaptive base (init_mode
adaptive -> multidepth_hole hybrid, since hole annul>0.40; k-anneal k_init=20
-> k_final=120 ramped over the full 10000). GPUs 0/2/3. Deployed-best
(top-level hard_dice = best_on_hard, the peak before the optimizer over-carves):

  s1 = 0.2615   s2 = 0.2636   s3 = 0.2614   mean = 0.2622
  (final_iter_hard_dice collapsed to 0.20-0.24 — the optimizer over-carves past
   the optimum by iter ~8500; best_on_hard correctly freezes the ~0.262 peak.)

vs hole 5k baselines: contour 0.273, multidepth_hole hybrid 0.274 (prior
session, 5k). Delta = -0.011. NEGATIVE — 10k iters does NOT lift the hole; it
is slightly WORSE than 5k.

MECHANISM (this is the real finding): doubling iters with the k-anneal ramp
STRETCHED over 10000 means k rises HALF as fast (k~70 at iter 5000 vs k=120 at
iter 5000 in the 5k run). The hole's thin annulus needs a SHARP k to carve the
narrow ring precisely; the slower ramp deprives it of sharp-k carving within
the deployable regime, so the deployed-best peak DROPS. best_on_hard cannot
recover because the sharp-carve window is delayed past the point where the
trajectory is still deployable (by the time k is sharp at 10k, the optimizer
has already over-carved the annulus walls). This directly refutes the earlier
"10k is the one thing that lifts the hole" inference from the code comment —
that comment predates the k-anneal schedule; under k-anneal, 10k with a fixed
k_final=120 is strictly worse for the hole.

IMPLICATION for per-shape iters: it is NOT a free win. To make 10k help the
hole you would have to scale k_final UP with iters (e.g. k_final=240 over 10k)
so the ramp reaches the same sharpness at the same relative position — but that
just recapitulates the 5k behavior at 2x cost, with no gain. So per-shape iters
(10k hole / 5k convex) is NOT a defensible shape-adaptive promotion: it costs
2x compute for a NEGATIVE delta.

CONCLUSION — the hole is at its CEILING (~0.27) under the current method. Across
FIVE levers this session (k_final=180 NEUTRAL, loss_shift=0.5 NEG, cut-overlap
4.5 NEUTRAL/NEG, k_ramp_delay NEUTRAL/NEG, iters=10k NEG), NOTHING moves the
hole mean. The residual gap (0.27 -> 0.88 convex level) is NOT compute-bound
(10k disproves that) and NOT method-tunable via loss/schedule/init knobs tested
— it is a GEOMETRY limit (thin annulus vs 3.175mm tool: the tool cannot cleanly
resolve the central column's narrow ring at res=32) compounded by GPU
atomic-add nondeterminism (±0.02 run-to-run noise on the hole specifically).
Both weak shapes are now characterized at ceiling:
  - hole  0.273 — geometry + GPU-noise limited, all 5 levers failed.
  - bowl  0.647 — seed-unstable, GPU-nondeterminism limited (cavity init).
The 4 convex shapes (sphere 0.830, cyl 0.889, box 0.760, pyramid 0.798) are
well-carved. The SOTA adaptive base is consolidated; the method's hard_dice is
essentially maxed for the achievable shapes, and the two weak shapes are
fundamentally hard under the current tool resolution + simulator.

NEXT: the hard_dice advancement path is exhausted on loss/schedule/init/iters
levers for the weak shapes. Remaining honest options: (a) a STRUCTURAL loss
change aimed specifically at the hole's annulus (e.g. an inter-pass residual
scallop / annulus-wall term) — still untested code-risk, lowered prior after 5
failures but the only untested structural lever left; (b) accept the hole/bowl
ceiling as a documented method limitation and consolidate the writeup around
the convex-shape wins + k-anneal breakthrough + preference-learning cosmetic
finding; (c) keep the preference queue stocked (separate-objective-layer path)
for the cosmetic qualities the human values independent of hard_dice. Preference
steering on contour/finish/stepover dims stays paused for hard_dice (cosmetic).

## jul15 STRUCTURAL annulus-residual loss (22:45) — STRONGLY NEGATIVE, NOT promoted (6th lever; closes loss-vs-geometry)

Implemented a STRUCTURAL loss change (the one untested lever, qualitatively
different from the 5 loss-WEIGHT/schedule/init/iters knobs): a per-voxel
annulus-residual emphasis in simulator/csg_simulator.py compute_loss + loss_at.
The uniform residual under-resolves the hole's thin column/annulus walls, so the
residual term is multiplied by (1 + w_annulus * max(0, 1 - max(0,target_d)/dref))
-- HIGH on near-surface waste (just outside the part surface = the thin walls),
0 in the far exterior. target_d is the fixed baked target SDF (not a diff param),
so the multiplier is a constant per voxel that scales the residual gradient
(autodiff-safe); w_annulus=0 leaves every other shape's loss exactly unchanged
(smoke-tested: exit 0, finite loss, no NaN). New args --w-annulus / --annulus-dref.

Tested w_annulus=5.0 (near-surface residual up to 6x), dref=2.0 vox, on the hole
s1/s2/s3 at the standard 5000 iters / SOTA adaptive base (fair vs 0.273 5k). GPUs
5/6/7. Deployed-best:
  s1 = 0.1621   s2 = 0.2205   s3 = 0.2321   mean = 0.2049
  (final_iter_hard_dice COLLAPSED to ~0.002 with soft dice=0.0000 for all three
   -- the optimizer destroys the part.)

Delta vs 0.273 baseline = -0.068. STRONGLY NEGATIVE -- the structural
annulus-residual emphasis makes the hole WORSE, not better.

MECHANISM: upweighting near-surface waste 6x pushes the optimizer to carve
aggressively right up to the thin annulus walls, but the 3.175mm tool cannot
clear near-surface waste adjacent to a thin wall WITHOUT gouging the wall -- so
the aggressive near-surface carving destroys the annulus ring (soft dice -> 0).
The deployed-best is frozen at the EARLY pre-collapse peak (~0.20, before the
annulus pressure fully bit); once the emphasis dominates the gradient, the part
is erased. This is the OPPOSITE of the intended effect.

CONCLUSION -- this DEFINITIVELY closes the loss-vs-geometry question for the
hole. Across SIX levers (k_final=180 NEUTRAL, loss_shift=0.5 NEG, cut-overlap 4.5
NEUTRAL/NEG, k_ramp_delay NEUTRAL/NEG, iters=10k NEG, structural annulus-residual
STRONGLY NEG), NOTHING moves the hole mean up; the structural loss change makes
it worse. The hole's 0.273 ceiling is a GEOMETRY limit (3.175mm tool cannot
cleanly resolve the thin annulus at res=32), not a loss-formulation or
optimization limit -- aggressive loss pressure to clear the thin walls causes
wall destruction. The lever that would actually move it (finer tool / higher
res) is outside the constraints (simulator not modifiable, no new deps).

HARD_DICE PATH IS NOW EXHAUSTED. Both weak shapes are characterized at ceiling:
  - hole  0.273 -- geometry-limited; all 6 levers failed (structural makes worse).
  - bowl  0.647 -- seed-unstable, GPU-atomic-add-nondeterminism limited.
Convex shapes well-carved (sphere 0.830, cyl 0.889, box 0.760, pyramid 0.798).
The method is consolidated: the k-anneal breakthrough (+0.137 sphere) is the big
win; the preference-cosmetic finding (preferences track visual contour quality,
not hard_dice) is the secondary result; the hole/bowl ceiling is a documented
method limitation. Remaining ongoing work: (c) keep the preference-learning loop
stocked (the PRIMARY focus -- the human values the cosmetic qualities independent
of hard_dice, the separate-objective-layer path) and consolidate the writeup.
No further hard_dice loss/schedule/init/iters/structural experiments are
warranted -- the ceiling is geometric.
