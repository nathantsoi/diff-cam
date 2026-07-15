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
