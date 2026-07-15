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
| sphere_bowl  | 11.43  | 0.6185         | (k-anneal only; contour NOT yet tried)  |
| sphere_hole  | 11.43  | 0.2630         | (structurally broken; contour NOT tried)|

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
