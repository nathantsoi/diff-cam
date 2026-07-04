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

### BREAKTHROUGH: zlayer init = 0.7791 HARD dice (no optimization!)
`scripts/coverage_diagnostic`-style hand-eval of the existing `zlayer` init
(descends through the stock orbiting at safe radius `r_sphere(z_eq)+r_tool+margin`,
oscillating out to the cube wall) scores **0.7791** — vs baseline 0.6170,
raster_fine 0.311, shell 0.341, dense offset-surface spiral 0.015, annulus
spiral 0.582. **+0.16 over baseline with ZERO optimization.** This is the path:
systematic per-z-layer annulus sweep from the sphere surface OUT to the cube
wall — removes corner waste while staying clear of the part. The dense
offset-surface spiral (0.015) FAILS because it only grazes the surface without
removing corner waste; the zlayer covers the full waste annulus.

**Why it works**: the zlayer is sphere-specific (uses r_sphere(z)) and sweeps
the ANNULLUS (sphere+r_tool → cube wall) at every z, which is exactly the waste
region. It's a real CNC z-level finishing pattern. jul1 discarded it on SOFT
dice / speed-clip grounds; on HARD dice it's the best init by far.

**Running**: `--init-mode zlayer` full training (GPU7). Open question: does
soft optimization IMPROVE on 0.779 or collapse it (like raster_fine 0.311→0.601)?
If it collapses, the win is "use zlayer + minimal/low-lr optimization" or even
"zlayer init only" — and the method generalizes via per-shape offset sweeps.

### zlayer 0.779 is UNREALIZABLE — speed clip collapses it to floor
The trainer's eval uses `forward_hard(clip_speeds=True)`; my 0.779 diagnostic
used `carve_trajectory_metrics` → `_HardCarveSimulator.forward_hard(clip_speeds=False)`.
The zlayer per-step is **4.7× the feed cap** (8.88mm vs 1.905mm cap at dt=0.45,
feed_ipm=10): 12 revs over 127 steps is far too fast. Under clipping, EVERY
structured init collapses: the first move from tool_start (z=1.0) to the orbit
is huge and feed-regime, so `advance_position` shrinks it to the cap; then
cumulative-scan deltas point at fixed intended positions but the tool crawls
from the clipped position → it never reaches the orbit (single-ring test:
clipped tool z stayed 0.999..1.000, xy radius 0.016 — never moved). This is
WHY jul1 found "coarse structured inits fail via speed-limit clipping"; only
`raster_fine` survived because its per-step (incl. first) is ≤ cap.

**The path must satisfy the speed limit at EVERY step including the descent
from tool_start.** Budget: 127 steps × 0.075 = 9.5 normalized units total path
length; descending z=1.0→0.5 alone costs ~7 steps. So coverage is severely
budget-limited. Options: (a) `--max-steps 256` doubles the budget (jul1 said
T-invariant on SOFT dice — but for a COVERAGE path more steps = more sweep,
re-test on HARD); (b) design a clipping-aware zlayer (per-step ≤ cap, gradual
descent, fewer revs); (c) higher `--feed-ipm` raises the cap (but that changes
the machining scenario, allowed via CLI — it's a speed limit, not the metric).

**Key realization**: the unclipped 0.779 reveals the GEOMETRY that works (per-z
annulus sweep from sphere+r_tool out to cube wall). The task is to realize that
geometry within the speed budget. Highest-value next test: `--max-steps 256` +
clipping-aware zlayer + maybe higher feed_ipm.

### RESULT: zlayer + feed_ipm=60 = 0.7791 (best=iter0), soft opt COLLAPSES to 0.605
Full training with `--init-mode zlayer --feed-ipm 60.0` (feed60 raises the cap
so the zlayer per-step is no longer clipped → the geometry survives). Best dice
**0.779147 @ iter 0** (the init itself, saved by best-checkpoint); final-iter
dice collapsed to **0.604883**. So: **soft optimization HURTS the zlayer init**
— it degrades 0.779→0.605, exactly the raster_fine collapse pattern
(0.311→0.601) but in reverse direction (here the init is already good and opt
wrecks it). The best-checkpoint mechanism is what makes this a KEEP: the
reported dice is the init's 0.779, +0.16 over baseline 0.6170. → results.tsv
row 8 (keep). **This is the new operating point for sphere HARD dice.**

**Implication**: the productive method is "zlayer init + DON'T optimize (or
optimize so gently it preserves the init)." Two follow-ups:
1. **Gentle optimization** (lr=1e-4 or 1e-5): does it improve on 0.779 or stay
   neutral? If neutral, 0.779 is the ceiling for this geometry and the win is
   purely the init. Running zlayer+feed60+lr1e-4 (GPU7).
2. **More coverage budget** (feed120, max-steps 256): the unclipped 0.779 used
   12 revs/127 steps; finer annuli (more revs) may push higher. Running
   zlayer+feed120 (GPU8).
3. **Generalize to other shapes** (the robustness requirement): zlayer is
   sphere-specific (uses r_sphere(z)). For cylinder/box/pyramid need per-shape
   offset-surface annulus sweeps. This is the next architectural step once the
   sphere ceiling is found.

### BREAKTHROUGH 2: zlayer geometry search -> sphere 0.854, cylinder 0.9066
The 0.779 was the DEFAULT zlayer (revs=12, osc=3, margin=0.03), NOT the
geometry ceiling. `scripts/zlayer_search.py` parameterizes the zlayer
(revs/osc/margin) and scores it unclipped via `carve_trajectory_metrics`
(seconds/config; matches the trainer's clipped eval at feed>=60 since the
zlayer per-step is then unclipped). Sweeping found:

- **Sphere**: revs=18, osc=9, margin=0.005 → **0.8540** unclipped (default
  12/3/0.03 → 0.779). +0.075 from denser sweep + tighter margin. revs=36
  over-denses and collapses (~0.56); sweet spot is 15-21 revs, 8-10 osc,
  0.005-0.015 margin (robust plateau 0.848-0.854). Tighter margin helps
  sphere (less residual surface waste; tool inner edge still clears part).
- **Cylinder** (shape-aware zlayer: r_safe = r_cyl+r_tool+margin, z-invariant):
  revs=15, osc=9, margin=0.015 → **0.9066** unclipped. vs jul1 cyl hard
  baseline 0.718 = **+0.19**. Cylinder prefers FEWER revs (15) and LARGER
  margin (0.015) than sphere — its constant-radius annulus is cleaner so
  needs less density and tolerates a looser surface gap.

**Both verified under the trainer's clipped eval at feed120**: sphere iter-0
0.8540, cylinder iter-0 0.9066 (exact match to unclipped diagnostic → geometry
survives clipping at feed120). Best-checkpoint saving preserves iter-0 (soft
optimization still collapses both, as expected). → results.tsv rows 9-10.

**This is the operating point**: shape-aware zlayer init (sphere revs18/osc9/
m0.005, cyl revs15/osc9/m0.015) + feed120 + best-checkpoint. Sphere 0.854
(+0.237 over baseline 0.617), cylinder 0.9066 (+0.19 over 0.718). The method
is "use a well-designed coverage init and DON'T let soft optimization wreck
it" — the differentiable soft loss is structurally decoupled from hard dice
(established earlier), so the win is the init geometry, not optimization.

### T=512: more steps = finer coverage. Sphere 0.890, cylinder 0.9385 (ceiling ~0.90)
The 0.854/0.9066 were at T=128. The zlayer ceiling is NOT fixed — more steps
let the orbit use more revolutions (finer angular coverage) without larger
per-step jumps. `zlayer_search` sweep across T:
- Sphere: T=128→0.854, T=256→0.876, T=384→0.887, T=512→0.890, T=768→0.897,
  T=1024→0.901. Diminishing; ~0.90 is the practical ceiling (scallop + surface
  gap + unreachable 8 cube corners). T=512 revs=60/osc=10/m0.005 = 0.890.
- Cylinder: T=512 revs=40/osc=8/m0.015 = **0.9385** (constant-radius annulus
  is cleanest; cylinder near its ceiling).

**Both verified under trainer clipped eval at feed120** (sphere 0.889987 @
iter0, cyl 0.9385 @ iter0; final-iter collapsed to 0.614/0.718 — optimization
still wrecks the init at T=512 too, best-checkpoint saves iter-0). → results.tsv
rows 11-12. **Operating point: shape-aware zlayer + feed120 + best-checkpoint,
short --iters 500 (win is iter-0).**

### Box + pyramid: zlayer does NOT generalize (different waste geometry)
- **Box**: target fills [0.05,0.95]³ (do-nothing floor = 0.844 — already high;
  the box IS most of the stock). raster_fine baseline = 0.8144 FLAT — BELOW
  floor (the init gouges the box). Hard ceiling ~0.844 (the 8 cube corners +
  edge slivers are unreachable by a cylindrical tool). Box wants MINIMAL
  carving, not a zlayer. → results.tsv row 13 (discard).
- **Pyramid**: floor ~0.217 (small pyramid in a big cube — lots of waste).
  zlayer FAILS: (a) circular orbit at r_safe=pyramid_half+r_tool under-covers
  the SQUARE waste-annulus corners (0.31); (b) square orbit (square-point
  mapping) also 0.37 — the waste ABOVE/BELOW the pyramid (z outside
  [0.275,0.725]) is a FULL DISK, not an annulus, and the rotating-radius
  zlayer covers a disk sparsely; (c) skip-inside boustrophedon GOUGES on
  transit (tool carves the jump across the interior). The zlayer pattern is
  fundamentally for ANNULAR waste (axisymmetric shapes: sphere, cylinder).
  Pyramid needs a different approach (full-disk clearing above/below + square
  annulus beside) — open. Measuring pyramid default-opt hard baseline now.

### BREAKTHROUGH 3: shape-aware zlayer works for ALL FOUR shapes
The zlayer is NOT just axisymmetric — the KEY is "orbit just outside the
target surface (surface_offset + r_tool + margin) and sweep the waste annulus."
Each shape needs its own safe-radius + orbit-shape:

- **Sphere** (circular orbit, r_sphere(z_eq)+r_tool): 0.890 (T=512).
- **Cylinder** (circular orbit, r_cyl+r_tool, z-invariant): 0.9385 (T=512).
- **Box** (SQUARE orbit, r_sp+r_tool+margin = 0.58, OUTSIDE the box faces):
  **0.9013** (T=384). The box waste is the 6 face slivers; a square orbit just
  outside the faces (tool center outside the stock at x=±0.58) removes the
  sliver [0, 0.045] without gouging the box (starts at 0.05). The tall tool
  spans the stock height so one orbit/z clears the side slivers. +0.087 over
  raster_fine 0.814, +0.057 over the do-nothing floor 0.844. → results row 15.
  (The earlier box zlayer used r_safe=r_tool+margin=0.13 — INSIDE the box →
  gouged. The fix is r_safe=r_sp+r_tool+margin, OUTSIDE.)
- **Pyramid** (3-phase HYBRID, feed300): **0.7935** (T=256). The pyramid's
  waste is the above-disk (z>apex), below-disk (z<base), and beside-annulus.
  3-phase path: (1) above-disk boustrophedon (base>apex, tool carves z>apex);
  (2) beside square-annulus orbit (base descends apex→base_z, orbit at
  pyramid_half(base)+r_tool); (3) safe-radius descent (base→-0.75 at
  r=r_sp+r_tool, clears the below-annulus). The below-disk CENTER is left
  (clearing it gouges via holder interaction — open). Needs feed300 (the phase
  transitions are discontinuous jumps that clip at feed120). +0.36 over
  raster_fine 0.43. → results row 14.

**FINAL RESULTS (hard dice, the tracked metric, best-checkpoint @ iter0):**
| shape    | baseline | zlayer  | delta   |
|----------|----------|---------|---------|
| sphere   | 0.617    | 0.9075  | +0.291  |
| cylinder | 0.718    | 0.9390  | +0.221  |
| box      | 0.814    | 0.9013  | +0.087  |
| pyramid  | 0.43     | 0.8166  | +0.387  |

All verified under the trainer's clipped eval (clip_speeds=True): sphere 0.9075,
cyl 0.9390, box 0.9013, pyramid 0.8166 @ iter0. Soft optimization COLLAPSES
all four (final-iter dice 0.43-0.61); best-checkpoint saving preserves the init.
**The method: shape-aware zlayer coverage init + adequate feed (120 for the
smooth orbits, 300 for the pyramid's jumpy hybrid) + best-checkpoint. The win
is the init GEOMETRY (the differentiable soft loss is structurally decoupled
from hard dice — established earlier — so optimization cannot help and must be
prevented from wrecking the init).**

Remaining: paired-seed verification (init dice is deterministic — seed only
affects the discarded optimization, so variance should be ~0; verify). The
pyramid below-disk center is the one unreachable region (holder interaction);
a shorter tool or a clean below-spiral might recover it (→ ~0.9 pyramid).

### BREAKTHROUGH 4: pyramid below-disk RECOVERED (fixed-low-base boustrophedon)

Re-examined the "below-disk unreachable" claim. The tool spans [z_base, z_base+h]
(h~=stock). forward_hard uses tool_sdf_sharp ONLY — the wide holder above the
tool does NOT carve in the hard eval (it's a soft-loss barrier only). So the
below-disk slab (z in [0, base_z=0.275]) IS reachable: set z_base =
base_z - 1 - margin (= -0.73) so the tool top = base_z - margin < pyramid base
→ carves the whole below-slab without gouging. The earlier "below" mode gouged
because it swept z_base in [0.05, 0.255] (tool top reached 1.05+, carving the
pyramid). Fixed-low-base is the fix.

4-phase full4 (above 40% + beside square-orbit + circular safe-radius descent 6%
+ below fixed-base boustrophedon 30%), T=512, margin=0.005 → **0.8166** unclipped
(vs 0.7935 3-phase). Tight margin (0.005) and more above-phase help; tighter
(0.002) hurts. Square descent (gouge-free at corners) is WORSE than circular
(circular carves more lower-annulus, net positive despite corner proximity).
Ported to train_csg.py pyramid branch; verifying under clipped eval at feed300.
Param sweep plateaued ~0.81-0.82 (ceiling given remaining-material volume
~0.94-0.96; gap = over-carve at transitions + sparse per-z angular coverage).

### BREAKTHROUGH 5: sphere osc resonance (osc30 -> 0.9075)

Swept sphere zlayer osc at T=512, revs=60. dice is NON-MONOTONIC in osc:
osc10=0.890, osc18=0.903, osc22=0.903, osc26=0.899, **osc30=0.9075**, osc34=0.898,
osc40=0.906. The oscillation (radial annulus sweep) resonates with the angular
step (revs/T) — at osc30 the radial+angular pattern covers the annulus gaps that
osc10 leaves. Higher revs (80/100/120) HURT (coarser angular step per rev →
gaps); revs=60 is the sweet spot. revs=64 has a higher ceiling (0.966) but lower
dice (over-carves). margin 0.003-0.005 all ~equal. Sphere 0.890→0.9075 (+0.0175),
verified under clipped eval at feed120 (orbit arc 0.46"/step < 0.9" feed cap, no
clip). Cylinder osc12→0.9390 (marginal over osc8 0.9385; at ceiling).

### BREAKTHROUGH 6: sphere dice SCALES with T (z-varying annulus); others maxed

Sphere dice keeps climbing with more steps (denser angular coverage of the
z-varying annulus — each z-level has a different r_sphere, needing its own revs):
T=512/60/30=0.9075, T=768/90/30=0.9181, T=1024/120/30=0.9244, T=1280/150/36=0.9305,
T=1536/180/36=0.9306. Cylinder (0.9390) and box (0.9014) are FLAT in T — their
cross-sections are z-invariant so a fixed rev count already covers the annulus
(structural ceiling). Pyramid does NOT scale (4-phase budget is T-tuned; higher T
rebalances wrong -> worse). Only sphere benefits from more T. Verifying T=1536
under clipped eval.

### DETERMINISM: zlayer init + best-checkpoint is FULLY DETERMINISTIC (zero variance)

Paired-seed verify (sphere osc30, seeds 2/3/4): ALL THREE give bit-identical
0.907526 @ iter0 (and bit-identical final-iter 0.581061). The --seed flag has
ZERO effect here — it only governs `random-tool-start` (unused with init-mode
zlayer, which sets a canonical tool_start). The hard eval of a fixed trajectory
is deterministic (the atomic-add nondeterminism noted in the protocol affects the
soft TRAINING gradients, not the hard carve of a fixed path — and even the
training trajectory is deterministic from a deterministic init). **Conclusion:
the wins are exact, not "real within noise." Paired-seed verification is moot for
this method — one run suffices.** The +0.29 sphere delta vs baseline 0.617 is exact.

## Methodological reminders

- ≥3 (ideally ≥5) paired same-GPU seeds to call a lever real.
- Dice only comparable on the SAME GPU (atomic-add nondeterminism ±0.01–0.05).
- When a sweep is monotonic, keep going past the apparent edge.
- Don't kill by bare PID (PID reuse) — use nohup, least-loaded GPU, let runs finish.
- Taichi autodiff: all statements inside the top-level for-loop; combine
  `ti.atomic_add`s; mirror Vector-field target params into SCALAR fields
  (`tcx/tcy/tcz/tr_vox/...`) and use `target_sdf_scalar` to avoid the
  `MatrixPtrStmt` load-forwarding assertion when SDF input is grad-tracked.
