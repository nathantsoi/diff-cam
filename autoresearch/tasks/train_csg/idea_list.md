# idea_list.md — research ideas from human feedback (jul10-gouge batch)

Generated from the human star ratings + free-text notes in
`run_feedback.json`, cross-referenced with each run's `args.json`/`metrics.json`.
Every idea below is **shape-agnostic** (no branching on shape name in
optimizer/init/loss) and **in scope** (init / trajectory-structure / loss-weight /
selection-layer code only — never the eval/metric/scoring-carve code or the
simulator autodiff core). Gate metric is `hard_dice` (deployable); trajectory
qualia (air-cutting, coverage, tightness) are tracked via the existing
`air_time_frac`/`residual` and the new diagnostics proposed here.

## What the feedback says (synthesis)

The dominant, recurring complaints across ~30 rated runs:

1. **Air cutting at the END of the trajectory.** "ends with tons of air cutting
   instead of cutting the outside"; "very long air cutting trajectory at the
   end"; "the end of the trajectory just cuts air"; "moves away, then cuts,
   then moves away again." Appears in 15+ runs. The trajectory does useful work
   early then wastes the tail budget in air. **Notably ABSENT from the 5★ hole
   cavity run** — "there is not air cutting at the end of the trajectory."
2. **Incomplete coverage — doesn't finish the outside / bottom.** "doesn't cut
   the outside of the part"; "makes it 1/2 way around the outside, but then
   wanders off"; "continue the path all the way to the bottom of the part";
   "doesn't cut enough inside the bowl or all the way around the outside."
3. **Pattern not tight enough / material left.** "the pattern is not tight
   enough"; "lots of material left"; "make finish passes closer to the surface
   of the part." Coarse stepover leaves scallops/uncut stock between passes.
4. **Axial plunging into the part** (bad practice + break risk). "the endmill
   plunges into the part"; "plunges way too fast and would probably break the
   tool." → user explicitly requests **helix-in**.
5. **Top of stock not faced off.** "the top of the stock is not cut to match
   the sphere." No top-surface facing strategy.
6. **Tool-break metric miscalibrated.** "i dont think the endmill will break,
   so calibrate the tool break metric based on this" — a human-judged-safe
   trajectory reads as break-risk.

The **highest-praise runs**:
- **5★ hole `multidepth_cavity`** (1783733370940): "great start... moves down
  into the hole in a regular pattern, then cuts the outside... **learn more from
  this one!** notably, there is not air cutting at the end." → interior-first
  ordering kills end-air-cutting.
- **5★ cylinder `multidepth`** (1783725757990): "great! the cutter follows the
  contour." → multidepth works on convex shapes; only needs tighter surface
  following + bottom completion.
- **3★ sphere `multidepth_cavity`** (1783735965963): **hard_dice 0.741**, beats
  the prior SOTA sphere 0.679. → cavity init also helps sphere.

**Headline signal:** the `multidepth_cavity` interior-first init (plunge →
interior → retract → exterior) is the winning direction — it got the only 5★
on the hardest shape AND beat SOTA on sphere. The remaining problems are
budget-allocation (end-air), coverage (finish the perimeter/bottom), stepover
tightness, and missing CAM heuristics. **A second signal cuts across all of
these:** the feedback is fundamentally *comparative* ("learn more from *this*
one," "which finishes the outside"), so a pairwise A/B preference loop
(Idea 9) — with the agent actively querying near-tie and champion/challenger
pairs and learning a Bradley-Terry ranking from the answers — is the right
vehicle to capture the qualitative judgments (air-cutting, tightness, break
safety) that `hard_dice` cannot measure.

---

## Idea 1 (PRIORITY): Make `multidepth_cavity` the default init for all shapes

**Feedback basis:** the 5★ hole ("learn more from this one", no end-air-cutting)
and the 3★ sphere (0.741 dice > SOTA 0.679). The interior-first segment ordering
is what eliminated end-air-cutting on hole.

**What:** `multidepth_cavity` already falls back to plain multidepth when no
interior cavity is detected (zero-regression guard). Promote it to the default
`--init-mode` and finish the stopped sphere/bowl regression guards (they were
at ~iter 280/2000 when paused). If sphere holds ≥0.68 and bowl ≥0.55, make it
the new baseline init for the whole sweep.

**Gate:** sphere hard_dice ≥ 0.68 (no regression vs SOTA 0.679), bowl ≥ 0.55,
box/cyl/pyramid within 2σ of their Wave-3 finals. **Lowest-risk, highest-signal
idea — do first.**

## Idea 2 (PRIORITY): Kill end-of-trajectory air via a late-weighted air penalty

**Feedback basis:** the single most common complaint. Currently `w_air_time`
penalizes TOTAL air time uniformly, so the optimizer is free to dump the tail
budget into air after finishing the "easy" cuts. The 5★ hole (low end-air) is
the exception because interior-first ordering structurally consumes the budget.

**What:** add a position-weighted air-time loss — penalize air time with a
weight that **ramps up over the trajectory** (low early, high late), mirroring
the existing `w_prox_warmup_frac` mechanism but inverted (early entry/reposition
air is cheap; late air is expensive). Implement as a per-step weight in the
air-time loss term (shape-agnostic: depends only on step index). Also add a
diagnostic `air_time_frac_late` (air time in the last 1/3 of steps / total late
time) to `metrics.json` so the complaint becomes measurable.

**Gate:** `air_time_frac_late` drops vs the same-config baseline, with hard_dice
held or improved. User feedback ("no air cutting at the end") is the real check.

## Idea 3 (PRIORITY): CAM best-practice heuristics from SDF geometry (explicit user request)

**Feedback basis:** verbatim — "the endmill should always helix in, if it fits.
applying heuristics like this can be done without knowing the part metadata,
only from the part shape in the sim. other heuristics include finding flat top
surfaces to facemill. look up best practices for these heuristics. find a way to
apply them."

**What:** implement shape-agnostic feature detection on the baked target SDF
grid (reads voxels only, no shape names) and inject corresponding structured
passes into the init trajectory:
- **Helix-in entry (generalize the cavity fix):** never axial-plunge. Any time
  the tool must descend into material, ramp/helix at a constant radius so every
  segment has non-zero XY displacement (also dodges the tool_sdf NaN). Already
  done for `multidepth_cavity`; lift it into a shared entry routine used by all
  init modes.
- **Flat-top facemilling:** detect z-slices where the target cross-section is
  large and near-constant (a flat top surface) and insert a zigzag/raster facing
  pass over it. Attacks "top of stock not cut."
- **Perimeter finish pass:** trace the outer contour of each z-slice (boundary
  of target SDF < 0) as a final finish pass so the outside is fully cut —
  attacks "all the way around the outside" / "1/2 way around then wanders off."

**Gate:** hard_dice up on the shapes where the relevant feature exists
(sphere/box/cyl top-facing; hole/bowl helix entry), no regression elsewhere.
User "follows the contour / cuts all the way around" is the qualitative check.

## Idea 4: Tighter stepover + explicit finish passes (kill "material left")

**Feedback basis:** "the pattern is not tight enough"; "lots of material left";
"make finish passes closer to the surface of the part"; "could follow the
surface of the part more closely."

**What:** the multidepth spiral stepover is set by `multidepth_revs` (default
3.0) and the budget. Sweep `multidepth_revs` upward (denser angular coverage →
smaller scallop) and add a dedicated finish-pass segment that traces the target
surface at full depth after bulk removal. Init-structure + parameter change
(in scope). Watch the budget: tighter stepover costs arc length, so pair with
Idea 2 (no wasted tail) to keep within `max_steps`.

**Gate:** `residual` (uncut target volume) drops at fixed-or-lower `air_time_frac`,
hard_dice up. ≥3 reps (variance floor).

## Idea 5: Coverage-completion loss (finish the outside/bottom)

**Feedback basis:** "doesn't cut the outside"; "1/2 way around the outside,
then wanders off"; "continue the path all the way to the bottom"; "doesn't cut
enough inside the bowl or all the way around the outside."

**What:** a loss term that rewards the trajectory for **covering currently-uncut
target voxels**, with diminishing returns so the optimizer is pushed to spend
remaining budget completing the perimeter/bottom rather than re-cutting already-
cut regions or wandering. Shape-agnostic: operates on the swept-vs-target voxel
set, no shape names. Complements Idea 2 (which removes bad tail; this directs
good tail).

**Gate:** residual down on the uncut-exterior portion specifically; hard_dice up.

## Idea 6: Tool-break metric recalibration (explicit user request)

**Feedback basis:** "i dont think the endmill will break, so calibrate the tool
break metric based on this" (run 1783724205256, judged safe by the human).

**What:** use the human-labeled SAFE trajectory (1783724205256) as a calibration
anchor — read its `break_prob_*`/`fcut_max`/`engage_*` values and recalibrate
the **thresholds** used to interpret them (the `broken` flag cutoff and the
`best_w_break` weight in the composite best-score), NOT the metric computation
itself (which stays in scope as a selection-layer/parameter change only). Then
re-examine runs the user flagged "would break" (1783728116708, axial plunge) to
confirm they still read as break-risk after recalibration.

**Gate:** the human-safe anchor reads `broken≈0` / low break_prob; the
human-flagged-unsafe plunge run still reads high. Consistency with human labels
on ≥3 labeled runs.

## Idea 7: RLHF warm-start from the 5★ runs (use the feature we built)

**Feedback basis:** the 5★ hole cavity run — "learn more from this one!" — is
exactly the kind of human-approved trajectory the `--use-feedback` warm-start
mechanism is designed to propagate.

**What:** with the 5★ runs now in the store, run `--use-feedback` on hole (seed
from the 5★ hole cavity `trajectory_deltas.npy`) and cylinder (seed from the 5★
cylinder), then refine with Adam. The warm-start threshold is ≥5★ (above-avg on
the 1-7 scale). Directly tests whether human-approved trajectories are a better
starting point than heuristic inits.

**Gate:** hard_dice ≥ the warm-start donor's, at fewer iters (warm-start should
converge faster). Pair with Ideas 2-5 so the refinement also fixes the tail-air
and coverage the donor still had.

## Idea 8: Late-air diagnostic + "wander" detection (measurement infra)

**Feedback basis:** the user repeatedly distinguishes *late* air from useful
repositioning air. Current `air_time_frac` is a single total — it can't tell
"good entry air" from "bad tail wander."

**What:** split `air_time` into early/mid/late thirds in `metrics.json`
(`air_time_frac_early/mid/late`), plus a "wander" metric counting the number of
late-trajectory excursions away from the target envelope. This is a diagnostic
only (no scoring change) — it makes Ideas 2 and 5 measurable and gives the user
a column that directly reflects their complaint.

**Gate:** no performance gate — this is instrumentation. Ship alongside Idea 2.

## Idea 9 (PRIORITY): Active pairwise preference learning (RLHF via A/B queries)

**Feedback basis:** star ratings are coarse and force an absolute judgment per
run ("is this a 4 or a 5?"), which is hard and noisy — humans are far better at
*relative* judgments ("which of these two is better?"). The single clearest
human signal in the feedback is *comparative* (the 5★ hole beats everything;
"learn more from this one"). The pairwise-comparison webapp (`compare.html` +
`/__api/pairs` + `pairwise.json` + `load_pairwise_preferences`) was built to
elicit exactly this signal; this idea makes the autoresearch agent *actively
generate* the pairs and *learn* from the answers rather than waiting for ad-hoc
queries.

**What (three layers, all in-scope selection/infra code — no optimizer/init/loss
changes):**

1. **Active pair selection (the agent asks good questions).** Instead of random
   pairs, the agent enqueues pairs that maximize expected information:
   - **Near-ties on `hard_dice` within the same shape+max_steps.** Two runs with
     ~equal dice but visibly different trajectories (e.g. one air-cuts at the
     end, one doesn't) are exactly where a human preference breaks the metric
     tie — the highest-value query. Rank candidate pairs by `|dice_a − dice_b|`
     ascending and by trajectory-structure distance (e.g. end-air-frac /
     stepover / coverage diagnostics from Ideas 4/8) descending, so the user is
     only asked about pairs they can actually tell apart and that the metric
     can't.
   - **Same-init, different-loss-weight runs** (e.g. two sphere runs that differ
     only in `w_air_time`): a preference here directly tells the optimizer
     which knob direction the human wants, decoupled from dice.
   - **Champion vs. challenger:** the current best-on-hard_dice run for a shape
     vs. a new candidate. A human "challenger wins" is a deployability signal
     that dice alone misses (Idea 6's break-risk case is the canonical example:
     a higher-dice run that the human flags as break-unsafe should NOT be the
     champion).
   Cap pending queue at a small N (e.g. 8) so the user isn't buried; the agent
   POSTs `{run_a, run_b, prompt}` to `/__api/pairs` and the user answers in
   `compare.html`.

2. **Preference model (learn from the answers).** Maintain a lightweight
   Bradley-Terry / Elo ranking over runs from the recorded A/B/tie outcomes
   (`pairwise.json`, already read by `load_pairwise_preferences` — extend it from
   a win-counter into a proper relative-skill score with confidence intervals).
   Tie = 0.5 credit (already handled). This gives every run a
   `human_preference_score` that is *comparative* and *calibrated against the
   field*, unlike the absolute 1-7 stars. Shape-agnostic: the model operates on
   run identities + outcomes, never on shape name.

3. **Close the loop (preferences steer selection, not the loss).** Use the
   preference score in the **selection layer only** — as a tiebreaker among
   hard_dice-equal runs, as a gate on the "champion" (a run the human
   consistently ranks below a break-safe alternative does not become the
   deployed trajectory even if its dice is marginally higher), and as a richer
   seed source for Idea 7's warm-start (prefer trajectories with high
   preference score AND high dice, not dice alone). Never feed it into the
   optimizer/loss — the deployable metric stays `hard_dice`; preferences are a
   human-alignment overlay on top of it.

**Gate:** the preference model's ranking agrees with the existing star ratings
on the rated runs (≥80% concordance on pairs where both runs have stars) — i.e.
the comparative signal recovers the absolute signal where one exists, and
extends it where it doesn't. Deployability check: when the human preference
champion differs from the hard_dice champion, the preference champion must read
as break-safe (Idea 6) and not air-cut at the end (Idea 2/8) — the human is
picking up qualitative faults dice misses.

---

## Suggested ordering

1. **Idea 1** (promote `multidepth_cavity` to default) — cheapest, highest
   signal, unblocks everything by setting the new baseline init.
2. **Idea 8** (late-air diagnostic) — cheap, makes 2/5 measurable.
3. **Idea 9** (pairwise preference learning) — the infra is already built and
   the query cost is near-zero; start enqueues on near-tie + champion/challenger
   pairs in parallel with the structural fixes so the preference model has data
   by the time Ideas 2-5 land. The Bradley-Terry layer + selection-layer
   tiebreaker come once ≥15-20 pairs are answered.
4. **Idea 2** (late-weighted air penalty) — attacks the #1 complaint.
5. **Idea 3** (CAM heuristics: helix-in, facemill, perimeter finish) — explicit
   user request; structurally fixes entry-safety, top-facing, perimeter coverage.
6. **Idea 4** (tighter stepover + finish passes) + **Idea 5** (coverage loss) —
   attack "material left / not tight / not finished."
7. **Idea 6** (break recalibration) + **Idea 7** (RLHF warm-start) — polish,
   once the structural fixes are in; Idea 7's warm-start seed is improved by
   Idea 9's preference-aware selection.

Each step keeps the method shape-agnostic and advances only on `hard_dice`
(with trajectory-qualia checks from user feedback — and, once Idea 9 has data,
the Bradley-Terry preference ranking — as the deployability gate).
