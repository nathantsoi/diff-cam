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
