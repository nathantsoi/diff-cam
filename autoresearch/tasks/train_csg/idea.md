# idea.md — arq-agd/jul10-prefair run

**Branch:** `arq-agd/jul10-prefair` (from `autoresearch` @ e3b2f73)
**Tag:** `jul10-prefair` — preference learning + end-of-trajectory air (the two
leading ideas from `idea_list.md`).
**Run-output folder:** `runs/jul10-prefair/`
**Date:** 2026-07-10

## Starting point
- Standing instructions: `autoresearch/tasks/train_csg/autoresearch.md`.
- Idea backlog: `autoresearch/tasks/train_csg/idea_list.md` (9 ideas; synthesized
  from human star ratings + free-text notes in `run_feedback.json`).
- Pairwise preference infra already shipped (compare.html + `/__api/pairs` +
  `load_pairwise_preferences` in train_csg.py) — ready for Idea 9 to start
  enqueuing A/B queries.
- In-scope files read: `README.md`, `algorithms/train_csg.py` (Args dataclass).
- Key confirmed defaults: `dt=0.45` (line 471), `init_mode="random"` (168),
  `grad_clip=0.5` (211), `learning_rate=5e-3` (157), `iters=5000`, `max_steps=128`,
  `w_residual=1.0`, `w_gouge=4.0`, `w_time=w_air_time=w_break=1e-3`.

## Plan (from idea_list.md suggested ordering)
1. **Baseline** (mandatory first run) — default 1in cube / sphere / 0.5mm voxels,
   NO method changes (init_mode=random, all defaults). This is the reference for
   every later variation.
2. **Idea 1** — promote `multidepth_cavity` interior-first init to default
   (cheapest, highest signal; only 5★ on hole + beat SOTA on sphere 0.741 vs
   0.679). Finish sphere/bowl regression guards.
3. **Idea 8** — late-air diagnostic (`air_time_frac_early/mid/late`) → makes
   Idea 2/5 measurable. Ship alongside Idea 2.
4. **Idea 9** — start enqueuing pairwise A/B pairs (near-ties, same-init/diff-
   weight, champion-vs-challenger) in parallel with structural fixes so the
   Bradley-Terry preference model has data by the time Ideas 2-5 land. BT layer +
   selection-layer tiebreaker once ≥15-20 pairs answered.
5. **Idea 2** — position-weighted (ramp-up-late) air-time loss → kills the #1
   complaint (end-of-trajectory air-cutting).
6. **Idea 3** — CAM best-practice heuristics from SDF geometry (helix-in entry,
   flat-top facemilling, perimeter finish pass). Explicit user request.
7. **Idea 4** — tighter stepover (`multidepth_revs` up) + dedicated finish pass.
8. **Idea 5** — coverage-completion loss (reward covering uncut target voxels).
9. **Idea 6** — break-metric recalibration (human-safe anchor 1783724205256).
10. **Idea 7** — RLHF warm-start from 5★ runs (`--use-feedback`).

## Constraints (binding)
- Shape-agnostic: NO branching on shape name in optimizer/init/loss.
- Advance ONLY on `hard_dice` (soft dice is inflated proxy; read hard_dice first).
- results.tsv: untracked, tab-separated, no literal tabs, no `> run.log 2>&1` in
  command column.
- Cannot modify eval/metric/scoring-carve code or harness; cannot install
  packages. Init/trajectory-structure/loss-weight/selection-layer code = in scope.
- Simplicity preferred. First run on fresh branch MUST be baseline.
- NEVER STOP the loop.

## Working log
(appended below as experiments run)
