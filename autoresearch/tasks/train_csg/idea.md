# idea.md — ar-agd/jul4-hard-carve-gap

Branch: `ar-agd/jul4-hard-carve-gap` (created fresh from `autoresearch` on 2026-07-04).
This run continues the **soft/hard carve-gap** frontier opened by `ar-agd/jul3-hard-carve-gap`
(preserved on its own branch in git history) — fresh branch because the prior one already
existed; numbers below are re-established from a clean baseline, not inherited.

## Starting point (the baked-in operating point)

The proven operating point from `ar-agd/jul1-uniform-toolpath` (~127 experiments;
see `autoresearch.md` "Proven operating point & dead levers") is the baseline:

```
--dt 0.45 --learning-rate 1e-3 --init-mode raster_fine --w-len 0.03 \
--max-steps 256 --grad-clip 0.5 --eval-freq 10 --iters 5000
```

Fresh baseline (soft dice, the tracked metric): sphere ~0.85, box ~0.92,
pyramid ~0.89, cylinder ~0.94. **Remember**: code default `--learning-rate` is
still `5e-3` — always pass `--learning-rate 1e-3` explicitly.

## The open frontier: the soft/hard carve gap (~0.21)

The jul1 run's headline *fundamental finding*: the tracked **soft** dice (~0.94 cyl)
is a BIASED proxy. The true deployable **hard**-carve dice is ~0.718 and is
k-invariant, T-invariant (coverage-capped). The soft union over-erodes (adds
~log(2)/k per step), so a trajectory optimized for soft does NOT transfer to hard.
Staged training works end-to-end but gave only +0.0016 hard dice because stage-2's
soft objective doesn't transfer.

**This is the highest-value open lever for deployable dice.** To raise it, improve
the trajectory's hard-carve coverage (more steps / finer feed / better path), or
find a less-biased soft objective whose optimum transfers to hard — NOT loss
smoothness (k is settled at 10) and NOT more soft-dice levers (lr/iters/w_len are
exhausted).

## Plan

1. **Baseline** sphere + cylinder (lr1e-3, default scenario) — re-establish
   reference soft AND hard dice. Measure hard dice (`scripts/staged_train.py` /
   `algorithms/truncate_trajectory.py` hard-carve eval) alongside soft, so every
   idea is judged on the deployable number, not just the biased soft proxy.
2. **Hard-carve coverage levers**: finer feed (smaller per-step cap relative to
   voxel), more max-steps with a motion budget, parametric low-air toolpath
   (raster/spiral — inherently uniform + covers systematically). Goal: lift HARD
   dice, accept soft-dice neutrality.
3. **Less-biased soft objective**: experiment with union forms whose per-step bias
   is smaller than log(2)/k WITHOUT breaking gradients (k<=2 is dead — saturates;
   look for alternatives, e.g. a corrected/smoothed union, or anneal k during
   training). Judge by soft-vs-hard transfer, not soft alone.
4. **Parametric toolpath** (major architectural direction if the above stalls):
   low-dim raster/spiral parameters optimized end-to-end — directly serves the
   "uniform CNC patterns" + "less air" directives and may cover more hard material
   per step than free-form tool_delta.
5. Validate any real win across sphere/cylinder/box/pyramid with ≥3 paired same-GPU
   seeds before claiming it (single-seed wins overstate ~2–3× — bit the prior run).

## Dead levers (do NOT re-explore — see autoresearch.md)

w_air / w_prox / w_traj_prox (contour-hug losses trade dice 0.847→0.55; ~30% air
is the price of high dice), w_gouge (seed-reshuffling), w_jerk, lr_decay_frac,
dt0.5+m160 (single-seed fluke), raster_fine_wide, k≤2 (saturates), iters>5000
(marginal, 2× compute), finer voxel_size_mm, coarse structured inits, lr sweep
(exhausted, peak 1e-3).

## Notes / findings

### Pivotal: the tracked metric is now HARD dice (autoresearch.md "0.85" baseline is STALE)

Git-blame on `algorithms/train_csg.py:844` (`sim.forward_hard(T)` in the eval
block) → commit `7dc8008` (the jul1 PR #5 merge, 2026-07-03 15:26). The jul1
*loop* ran Jul 1–2 with the **soft** `forward()` eval → its findings.md soft-dice
numbers (sphere 0.85, cyl 0.94) are pre-port. The merge ported eval to
`forward_hard` (`apply_cut_hard` = exact `ti.max` union with `tool_sdf_sharp`,
binary-mask `dice_score`) AFTER the loop wound down. So:

- **This run's tracked `dice:` is HARD-carve dice**, not soft. autoresearch.md's
  "fresh baseline scores ~0.85 (sphere)" / "tracked soft dice ~0.94" text is
  stale (predates the port). The real baseline is the hard-dice number this run
  establishes (~0.72 cyl per jul1's separate hard measurement; sphere TBD).
- The "open frontier — close the soft/hard gap" is now DIRECTLY the tracked
  metric: every experiment already selects/saves checkpoints by hard dice, and
  the soft loss is only the differentiable proxy. Good — plan aligns.
- **Dice convention** (jul1 findings): `pred = stock<0 = REMAINING material`,
  `target = PART`. A STATIONARY tool scores 2|target|/(|stock|+|target|) =
  **0.553 (sphere r=11.43)** / 0.728 (cyl). Baseline iter 130 = 0.5486 ≈ the
  stationary floor (trajectory hasn't carved waste yet). Soft-optimized hard
  carve actively GOUGES (cyl hard 0.718 < stationary 0.728) — the gap is real.

**Implication**: the "proven operating point" (lr=1e-3 → 0.85) was tuned for
SOFT dice. For HARD dice it may be suboptimal — lr is worth re-examining on the
hard metric. The productive levers for HARD dice are coverage (more steps /
finer feed / better path) and a less-biased soft objective, NOT the exhausted
soft-dice knobs. Note: jul1's "T≥192 NaNs / iters>5000 marginal" were SOFT-dice
findings — for HARD (coverage-capped) dice they may differ; re-test on hard.

### jul1 already ruled out k-annealing AND more-steps for hard dice
jul1 findings: **hard dice is k-INVARIANT** (~0.718 for all k≤5, 0.720 at k=10)
and **T-INVARIANT** (hard flat ~0.718 across all T). `smooth_max(a,b,k) =
(1/k)log(exp(ka)+exp(kb))`; larger k = sharper = less bias, but k≤2 saturates
(gradients vanish). So: do NOT try k-annealing (k-invariant) or more steps of
the SOFT-optimized path (T-invariant). **T-invariance was tested on
soft-optimized paths only** — a SYSTEMATIC coverage path (parametric raster) may
use T more effectively, so "better path" is NOT ruled out. jul1 cyl hard 0.718 ≈
stationary cyl 0.728; sphere hard ~0.553 = stationary sphere 0.553. ⇒ the
soft-optimized trajectory does not carve the part in hard space AT ALL (soft
over-erosion did 100% of the apparent soft-dice work). Huge room: 0.553 →
potentially 0.8+ if real coverage is forced.

### Speed problem: forward_hard eval doubled per-iter cost (port side-effect)
jul1's 5.5 iter/s (soft eval) → now 2.66 iter/s (soft forward + forward_hard eval
every 10 iters) = 5000 iters in ~31 min, OVER the 20-min kill threshold. For
experiments use `--eval-freq 25` (≈22 min) or `--eval-freq 50` (≈18 min) +
`--iters 3000` if hard dice is flat (no transient peak to capture on hard). The
baseline (already running at freq=10) is allowed to run long — it's the
reference. Hard dice appears flat (no sharp peak), so coarser eval is safe.

### Coverage diagnostic RESULT (sphere, raster_fine init, 0 optimization)
- `scripts/coverage_diagnostic.py` hard-carves a trajectory with the exact eval
  path (`eval.eval_csg.carve_trajectory_metrics`) and scores it.
- **Do-nothing floor = 0.548** (tool never enters material; stock_occ=132651,
  target_occ=50061, floor=2·50061/(132651+50061)=0.548). Matches the baseline's
  flat 0.5482 ⇒ the soft-optimized trajectory carves NOTHING in hard space; the
  soft 0.85 was 100% over-erosion artifact.
- **raster_fine init (systematic boustrophedon) = 0.311** — WORSE than doing
  nothing. The blind raster plunges through the sphere and GOUGES it (over-
  removes part material). Coverage alone is not the answer; the path must
  OFFSET from the target surface by tool radius (a real CNC finishing /
  shell-offset path), not blindly sweep the bounding box.

⇒ **Refined direction**: the lever is a parametric toolpath that (a) covers the
waste region systematically AND (b) stays offset ≥ r_tool from the target
surface so it removes waste without gouging. The existing `shell` init orbits
just outside the sphere surface — re-examine its hard dice (it was discarded on
SOFT dice / speed-clip grounds; on HARD dice it may be the right starting
point). Then optimize its params end-to-end.

### Soft loss diverges late, hard dice stays flat — they are DECOUPLED
Baseline sphere iter 4859: soft loss jumped 0.2 → **274**, grad 5e4 (the soft
over-erosion cheat blowing up), yet hard dice **0.561** (still the floor). The
soft objective is now utterly disconnected from the deployable hard metric —
optimizing it harder cannot help. This is the strongest evidence that the
productive lever is NOT loss-tuning but trajectory-coverage structure.

### Skip-inside-sphere z-raster = 0.582 (first path to beat the 0.548 floor)
A z-layer boustrophedon over the full cube footprint that SKIPS points inside
the sphere (so it removes corner waste without gouging) scores 0.582 — the only
hand-designed path above the floor. Still far from good (coarse, tool can't
reach all corner waste), but it confirms: **systematic waste removal that
respects the surface is the direction.** The optimizer must be steered toward
this, not free to cheat via soft over-erosion.

### Experiment results

**Baseline sphere (lr1e-3, HARD dice) = 0.6170** @ iter 2540 best (final-iter
0.5554). Above the 0.548 do-nothing floor but FAR below the stale soft 0.85.
The soft-optimized trajectory carves a little in hard space, peaks mid-training,
then degrades as soft loss diverges. → results.tsv row 1.

**k=5 (sharper soft union) = CRASH (grad vanished)**. grad=7e-10, loss frozen at
1.3381, dice stuck 0.43 < floor. Confirms jul1: lower k saturates the soft union
and gradients vanish. **Soft-union sharpness (k) is a DEAD lever for hard dice**
— k=10 is the only viable value, and at k=10 the soft loss is decoupled from
hard dice. → results.tsv row 2 (crash). Do NOT re-explore k.

**raster_fine init**: running (hypothesis: coverage init → hard-dice-friendlier
basin). [result pending]

**raster_fine init = 0.6007** @ iter 4175 — WORSE than random-init baseline
0.6170. The coverage init does NOT help hard dice: soft optimization collapses
it back via over-erosion (the init pre-covers, but the optimizer's soft
objective rewards small cheating motions over maintaining coverage). → results.tsv
row 3 (discard). **Init is not the lever** (consistent with jul1: at lr=1e-3 the
LR win subsumed raster_fine on soft dice too).

### Why the soft loss can't fix hard dice (the structural reason)
The terminal `compute_loss` operates on SOFT occupancy `sigmoid(stock_d)` where
stock is the smooth_max-unioned soft carve. The optimizer satisfies residual+
gouge in SOFT space by over-erosion (small motions that read as full coverage
softly). Lowering k to sharpen hits the vanishing-gradient wall (k=5 dead). So
NO soft-loss tuning can transfer optimization to hard dice — confirmed by k=5
AND by the baseline's loss diverging to 274 while hard dice stayed flat. The
lever MUST be trajectory structure (coverage path that respects the surface) or
a non-soft objective. Next: high w_gouge (force surface respect) + raster_fine
init; then parametric surface-offset toolpath if those stall.

### NEW LEVER: w_tool_gouge — soft-union-INDEPENDENT surface respect
Implemented `compute_tool_gouge_penalty`: charges the TOOL CENTER directly for
penetrating target+r_tool — `relu(r_tool - target_sdf(seg_mid))^2`, ZERO when
tangent-or-outside (contact-cutting waste is FREE). Unlike stock-based w_gouge
(satisfied by soft over-erosion while hard carve gouges), this constrains
trajectory GEOMETRY directly → should transfer to hard dice. Differentiable via
midpoint→target_sdf_scalar. Gated by `--w-tool-gouge` (forwarded through
run_pipeline). Smoke-tested (40 iters, exit 0, grad ~10 flows).

**Running**: w_gouge=16 (stock barrier, GPU8) AND w_tool_gouge=1.0 (geometric
barrier, GPU2) in parallel. w_gouge=16 hit 0.6458 @ iter 847 (above baseline
0.6170 — possibly seed variance or the stronger barrier helping even softly).
[result pending for both]

**w_gouge=16 = 0.6458** @ iter 825 (final-iter 0.5484) — **+0.029 vs baseline
0.6170**. The stock-based soft gouge barrier at 4x weight HELPS hard dice
(single seed — needs paired-seed verification; jul1 found w_gouge 4→8 was
seed-reshuffling on SOFT dice, but this is HARD dice at a much higher weight).
→ results.tsv row 4 (keep, provisional). Best peaks EARLY (iter 825) then
degrades — best-checkpoint saving is essential.

**w_tool_gouge=1.0 = 0.5508 (floor)** — TOO STRONG. The geometric barrier pins
the tool off the part entirely (gouge diag 0.0001) so it can't carve the waste
adjacent to the surface either. → results.tsv row 5 (discard). Running
w_tool_gouge=0.1 (10x gentler) to test whether a nudge helps without pinning.

### Open question: is w_gouge=16's +0.029 real or seed variance?
jul1's lesson: single-seed wins overstate ~2–3×. Need ≥3 paired same-GPU seeds
of baseline vs w_gouge=16 on GPU 8 to confirm. QUEUED after the w_tool_gouge
sweep. If real, sweep w_gouge {8, 16, 32, 64} to find the peak.

## Methodological reminders

- ≥3 (ideally ≥5) paired same-GPU seeds to call a lever real.
- Dice only comparable on the SAME GPU (atomic-add nondeterminism ±0.01–0.05).
- When a sweep is monotonic, keep going past the apparent edge.
- Don't kill by bare PID (PID reuse) — use nohup, least-loaded GPU, let runs finish.
- Taichi autodiff: all statements inside the top-level for-loop; combine
  `ti.atomic_add`s; mirror Vector-field target params into SCALAR fields
  (`tcx/tcy/tcz/tr_vox/...`) and use `target_sdf_scalar` to avoid the
  `MatrixPtrStmt` load-forwarding assertion when SDF input is grad-tracked.
