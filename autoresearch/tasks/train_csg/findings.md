# findings.md — ar-agd/jul4-toolholder-collision

Branch: `ar-agd/jul4-toolholder-collision` (HEAD `3230c17` = "handle toolholder
collision on depth"). Objective: re-establish the proven zlayer-init operating
point under the simulator's NEW toolholder-collision handling (z-floor clamp +
holder swept-Z union + `trunc` pipeline stage), per shape, crash-safe.

Metric: viz-stage G-code **carved-vs-target Dice** (post-trunc) — the honest
deployed metric. Tracked alongside the train-summary HARD dice (pre-trunc).

## Headline

The prior zlayer wins **do not survive** collision safety unchanged. Three
shapes (sphere/cyl/box) re-baseline close to their old numbers once the init is
made crash-aware (orbits stay above the part, so the z-floor barely bites).
**Pyramid regresses hard** (0.817 → ~0.46 at 1in): its old win relied on a deep
tool-base plunge that is now (correctly) forbidden — that plunge was a real
machine crash. The crash-safe ceilings (sphere lower-interior wedge, pyramid
sloped-face inner band) are **reachability ceilings set by the fixed 25mm tool
vs the stock height** — not purely geometric. They are **monotonic in stock
size**: both sphere (0.843/0.819/0.705) and pyramid (0.556/0.457/0.431) rise at
0.75in (tool taller than stock → more interior reachable) and fall at 1.5in.
Crash-safe pyramid improved 0.416 → 0.457 (25-iter) → 0.492 (50-iter) → 0.517
(100-iter) → **0.526 (200-iter, best@iter199, PLATEAUING)** via a 2D
descending-annulus boustrophedon + extended optimization. **The 25-iter "0.457
ceiling" was severely UNDER-OPTIMIZED** — the pyramid opt does NOT plateau early:
it climbs iter15 0.453 → iter45 0.492 → iter80 0.517 → iter199 0.526. The huge
gains are in the first 80 iters; beyond iter100 diminishing returns (+0.009 over
the last 120 iters, loss 0.082 still decreasing). **The true 1in crash-safe
pyramid ceiling is ~0.526** (vs infeasible 0.817 — the inner band remains
partially unreachable, but the reachable annulus is now fully carved). It hurts
the 0.75in pyramid at 25 iters (best@iter0, like the sphere) — but the 0.75in
opt-headroom at 200 iters is under test. **Max-steps matters**: the pyramid
opt-helps at max-steps=512 (shorter trajectory, init marginal) but opt-COLLAPSES
at max-steps=1536 (best@iter0, 0.417 — the init saturates the larger budget, like
the sphere at 0.75in). The opt-helps pattern is a symptom of a marginal init
(shape × max-steps × stock size), not a pyramid invariant.

## Crash-safe per-shape results (viz carve Dice, post-trunc)

| shape    | infeasible (old) | crash-safe | max-steps | notes |
|----------|------------------|------------|-----------|-------|
| sphere   | 0.9306           | 0.819      | 1536      | lower-interior wedge + below-part slab un-carvable without the deep plunge; best@iter0 |
| cylinder | 0.9390           | 0.916      | 512       | z-invariant; floor barely hurts; best@iter0 |
| box      | 0.9014           | 0.892      | 384       | square orbit at equator; floor barely hurts; best@iter0 |
| pyramid  | 0.8166           | **0.526**  | 512       | sloped-face inner band unreachable crash-free (see below); 0.416→0.457 (25-iter) → 0.492 (50-iter) → 0.517 (100-iter) → 0.526 (200-iter, plateauing); opt-HELPS, max-steps-sensitive |
| **mean** | **0.897**        | **~0.788** | | (pyramid revised 0.457→0.526; mean was 0.771) |

**Each shape's champion uses a different max-steps** (sphere 1536, cyl 512, box
384, pyramid 512) — the optimal trajectory length scales with surface complexity
/ reachability. Comparisons MUST hold max-steps fixed per shape.

All crash-safe runs: `holder_overlap=0`, trunc trims only genuine near-collisions
(cyl 19 steps @ 0.15mm; sphere/box holder-clear runs byte-identical pre/post
trunc — verified `trajectory.untruncated.npy == trajectory.npy`).

## Size robustness (0.75in / 1in / 1.5in stock, proportional target)

The method **generalizes** across stock sizes (crash-safe, best@iter0 init-peak
pattern holds for all shapes at 0.75in; opt-helps for the 1in pyramid only at
adequate iters), and dice is **monotonically decreasing in stock size for ALL 4
shapes** — the fixed 25mm tool spans more of a smaller stock (1.31 stock-heights
at 0.75in → 0.66 at 1.5in), so more interior is reachable:

| shape    | 0.75in | 1in    | 1.5in        | crash-safe? | notes |
|----------|--------|--------|--------------|-------------|-------|
| box      | 0.937  | 0.892  | 0.884        | yes | z-invariant square orbit; most size-robust (Δ-0.05 full range) |
| cyl      | 0.933  | 0.916  | 0.842        | yes | curved side; moderate |
| sphere   | 0.843  | 0.819  | 0.705        | yes | 3D interior; largest swing (Δ-0.14) |
| pyramid  | 0.546  | 0.526  | ~0.34 ±0.05  | yes | ceiling-limited; opt-HELPS at 1in (0.40→0.526 @200-iter); 1.5in NON-deterministic (see methodology); 0.75in best@iter0 |
| **mean** | **0.816** | **0.788** | **~0.693** | | (pyramid revised: 0.556→0.546, 0.457→0.526, 0.431→~0.34) |

Takeaway: the crash-safe ceilings are **reachability ceilings set by the
tool-to-stock ratio**. Flat/z-invariant surfaces (box) are most robust; curved/3D
surfaces (sphere, cyl, pyramid) swing more. All runs crash-safe by construction
(trunc no-trim, holder_overlap=0). Eval stage OOMs/crashes on the larger grids
(known) but viz produces the canonical carved-vs-target Dice.

At 0.75in the 25mm tool is *taller* than the 19mm stock (z-floor dips below the
part bottom, -0.0025), so the lower-interior wedge / sloped-face inner band
become partially reachable → every shape's dice rises above its 1in ceiling.

The pyramid trend **revises the earlier "structural/geometric ceiling" framing**:
the pyramid ceiling is *not* purely geometric — it is tool-to-stock-ratio
limited, same driver as the sphere. At 0.75in the sloped-face inner band becomes
partially reachable and best@iter0 returns (init=win, like the sphere), whereas
at 1in the init is marginal so optimization helps (best@iter15). The opt-helps
pattern is a symptom of a marginal init, not a pyramid invariant.

**Confirmed directly via tool length**: a 1in sphere with a 50mm tool (2× stock
height) raises viz dice 0.819 → 0.840, crash-safe (min clr 25mm, no trim). The
gain is a **crash-safety gain, not a reachability gain**: the 50mm train dice
(0.8396) is *identical* to the 25mm train dice (cead72f, 0.8396) — the longer
tool reaches no more stock. But the 25mm aggressive init (train 0.840) suffered
a 55-step trunc trim → viz 0.821; the 50mm holder rides higher so the same init
survives trunc untrimmed → viz 0.840. Retuning the init for the 50mm tool does
NOT help (doubled revs → 0.652, gouge): revs180 is optimal and 0.840 is the
1in+50mm ceiling. **The init, not the tool length, is the bottleneck.**

**Deep z-floor + long tool REGRESSES (re-confirms the ceiling)**: 50mm tool
with `--z-floor-epsilon-mm 25` (floor -0.934) → viz 0.394, gouge 0.891, optimizer
stuck. Letting the base plunge deep makes the 50mm tool span the *whole* stock,
so its orbit must clear the equator's max radius (0.578 > stock half 0.5 →
outside the stock) → it gouges the equator. The winning 50mm config keeps the
base **high** (default z-floor, floor 0.011): the tool reaches the lower
interior *from above* (spanning [base, base+1.97], mostly above the stock)
without the orbit clearing the equator. **A tool spanning the equator must orbit
outside it** — the fundamental reachability ceiling, independent of tool length.

**Tool length is shape-dependent AND saturates**: a 50mm tool raises the 1in
sphere (0.819→0.840, lower-interior wedge reachable from above) but does NOT
help the 1in pyramid (0.457→0.421, slight regress). The pyramid's sloped-face
inner band is *alongside* the slope, not below it — a longer tool reaching down
from above gains nothing; only a side approach reaches it, and that gouges.
And the 50mm tool at 0.75in is **byte-identical** to the 25mm tool (viz 0.84264,
same carved-voxel count): once the tool is taller than the stock (25mm > 19mm),
the lower-interior is already reachable and extra length adds nothing. So:
smaller stock helps every shape (geometry shrinks into the tool's reach); a
longer tool helps only shapes whose ceiling is *below* (sphere) not *beside*
(pyramid), and only when the stock exceeds the default tool.

**Tool length SATURATES at ~2× stock height (confirmed):** 1in sphere @25mm
→0.819, @50mm→0.840 (+0.021), @75mm→0.8396 (+0.000). A 75mm tool (ratio 2.95)
rides higher (min clr 50mm) but reaches **no more** stock than 50mm — the
equator ceiling is **tool-length-independent once the base is high enough** to
reach the lower-interior from above. **0.840 is the genuine 1in sphere ceiling
for any tool ≥50mm.** The practical takeaway holds: use a tool ~2× the stock
height (not more) to recover the lower-interior; beyond that, diminishing
returns are exactly zero.

**The tool-to-stock RATIO is the primary driver (confirmed):** 1.5in@50mm and
0.75in@25mm have the same ratio (1.31) and give ~the same sphere dice (0.828 vs
0.843) — a small absolute-size residual remains (init params tuned for 1in don't
perfectly transfer). The 50mm tool recovers most of the 1.5in dice drop
(0.705→0.828). **Practical takeaway: at larger stock, use a proportionally
longer tool to maintain the ratio and recover dice.**

## What changed in the sim (recap)

- **Z-floor clamp** (`--z-floor-epsilon-mm`, default 1.0mm): executed tool BASE
  z clamped to `part_bottom_z - eps/stock_z` so the holder cannot plunge into
  remaining stock. Subgradient (grad flows above floor, zero below).
- **Holder swept-Z union**: holder Z extent unioned over the swept segment
  (fixes near-vertical-segment false-clear readings).
- **`trunc` stage**: post-train hard-carve, measures per-segment holder-to-stock
  clearance, stops at the last segment >1mm clear before the first collision.

## Crash-safe zlayer init (the fix that recovered sphere/cyl/box)

The zlayer `z_bot` clamp (and the pyramid descent floor) now respect THREE
constraints (in `algorithms/train_csg.py`):
1. `z_floor = part_bottom_z - eps/stock_z` (sim-enforced; below it the full-height
   tool gouges the body).
2. `z_holder_clear = 1 + clearance/stock_z - tool_height/stock_z` (trunc-enforced;
   the 2.5in holder is wider than the stock so it only clears when its bottom is
   above the stock top).
3. A small slack.

For sphere/cyl/box the `else`-branch `z_bot = max(z_bot, z_floor+0.005,
z_holder_clear)` suffices — the equator orbit stays above the floor.

## Pyramid: why ~0.46 is the 1in crash-safe ceiling (reachability, not coverage)

The pyramid's old 0.8166 used a **below-disk boustrophedon at fixed
`z_base_below = base_z - 1 - margin ≈ -0.955`** — a deep plunge placing the
tool's z-range below the pyramid base to carve the bottom slab. The z-floor
forbids this; clamped, the full-height tool gouges the body (0.8166 → 0.426).

Redesign attempts (all crash-safe):
- Drop deep-plunge, beside-orbit descending to `z_descent_bot`: 0.4159.
- Higher-T coverage (T=1024, revs=80, osc=24): 0.4102 (worse).
- 2D descending annulus boustrophedon (replace sparse 1D orbit): 0.4169.
- 2D boustrophedon LOW→HIGH zb + per-level cap (fix budget): 0.4517.
- Coarser grid (7pts, spacing ~2·r_tool) + higher per-level cap: **0.4570**.
- Pure-init (`--iters 1`): 0.4019 → optimization HELPS pyramid
  (0.40→0.457, best@iter15), unlike sphere/cyl/box.
- Direct ring-annulus on `s_safe` (face-hug at every zb): **0.402 — regression**.
  Better face-hugging geometry, but placing tools at low zb lets the optimizer
  push the base below holder-clear → the wide 2.5in holder breaches the stock-top
  clearance (min clr 0.021mm) → trunc trims 357/512 trailing steps. The
  coarse-grid init (0.457) stays holder-clear (min clr 18mm, no trim). Reverted.

**Diagnostic** (`_diag_pyramid.py`, per-z target retention on the carved stock):
the trajectory **under-carves** — only ~17% of waste removed (85321 waste voxels
standing), with **~zero gouge** (21 voxels). So it is not gouging the target;
the tool simply cannot reach the inner band of the annulus.

The structural reason: at height z the pyramid half-size is `hp(z)`. To carve
the annulus right against the face (radius `hp(z)+r_tool`), the tool must be at
base `zb ≤ z` (to reach z) with radius `hp(z)+r_tool`. But at that base `zb ≤ z`
the pyramid half-size `hp(zb) ≥ hp(z)`, so a tool at radius `hp(z)+r_tool` would
**gouge the pyramid at its own base** (`hp(zb) > hp(z)+r_tool` for the slope).
The only crash-free radius at base `zb` is `≥ hp(zb)+r_tool+margin`, which at
height z leaves the inner band `[hp(z), hp(zb)+r_tool]` un-carvable. A
full-height tool cannot hug a sloped face without the deep plunge. This mirrors
the sphere's lower-interior wedge. **However, this ceiling is tool-to-stock-
ratio dependent, not absolute**: at 0.75in (tool 25mm taller than the 19mm
stock) the inner band becomes partially reachable and the pyramid rises to
0.556 (see Size robustness). At 1.5in it falls to 0.431. The "fundamental
ceiling" holds for a FIXED tool/stock ratio; a relatively taller tool (or
articulated holder) lifts it.

## Dead levers confirmed / added

- **Voxel resolution** (0.4mm vs 0.5mm, 1in sphere): viz 0.818 vs 0.819 —
  identical within noise. The crash-safe ceiling is **geometric (reachability),
  not quantization-limited** — finer voxels reach no more stock. Dead lever.
- **Tool radius** (1/8" r=1.5875mm vs 1/4" r=3.175mm, 1in pyramid): viz 0.406
  vs 0.457 — **regression**. The uncarvable band is `[hp(z), hp(zb)+r_tool]`,
  whose width `(hp(zb)-hp(z)) + r_tool` is dominated by the **slope term**
  `(hp(zb)-hp(z))`, not `r_tool` — halving the cutter barely shrinks the band,
  while the smaller tool carves less per pass at fixed max-steps (coverage loss
  outweighs the marginal band-narrowing). Also flips the 1in pyramid from
  opt-helps (best@iter15) to opt-hurts (best@iter0): the smaller tool's loss
  landscape differs. Crash-safe (min clr 11.3mm, no trim). Dead lever — confirms
  the ceiling is set by the **slope/holder geometry**, not the cutter width.
- **Tool radius on sphere** (1/8", 1in): viz 0.794 vs 0.819 — also regression,
  but smaller (-0.025 vs pyramid's -0.051). The sphere's CURVED geometry makes
  `r_tool` a larger term in its uncarvable band (vs the pyramid's linear-slope-
  dominated band), so band-narrowing helps more — but coverage loss (smaller
  tool, fixed 1536 steps) still dominates. **Tool radius is a dead lever across
  both shapes**: the ceiling is set by slope/holder geometry + the fixed-step
  coverage budget, not the cutter width.
- **Tool radius + doubled steps** (1/8", 1in sphere, 3072 steps): viz 0.797 vs
  0.794 (1× steps) vs 0.819 (1/4"). Doubling the step budget recovered only
  +0.003 — **coverage was NOT the issue**; the smaller tool genuinely carves no
  more reachable area even with 2× the passes. Definitive: **the 1/4" end mill
  is optimal**, and tool radius is dead neither at fixed nor doubled steps.
- **Tool radius LARGER** (1/2" r=6.35mm, 1in box): viz 0.850 vs 0.892 — also
  regression. A bigger cutter **rounds the box's sharp corners** (corner-radius
  effect) → loses dice. **Tool radius is dead from BOTH directions**: smaller
  hurts sphere/pyramid (coverage + band-narrowing insufficient), larger hurts box
  (corner radius). The 1/4" end mill is the sweet spot across all shapes.
- Higher-T coverage for pyramid (revs/osc/T): does NOT help (ceiling is
  reachability, not density). 0.4159 → 0.4102.
- 1D orbit vs 2D boustrophedon for the pyramid annulus: same ~0.42 (the
  reachable outer annulus is captured either way; the inner band is unreachable).
- Direct face-hugging ring (radius `s_safe` per zb): REGRESSES to 0.402 — low-zb
  placement triggers holder collision → trunc trims 357 steps. Crash-safety, not
  geometry, is the binding constraint at low zb.
- Dense ring (308pts, full budget) + `--holder-margin 0.045`: REGRESSES to 0.389.
  The ring places tools at a SINGLE radius per zb, leaving the inter-radius bands
  uncarved; the coarse grid's MULTI-radius (rc 0.25 + 0.375) covers a wider band
  per pass and wins. `holder_margin` did NOT prevent the trunc trim — the soft
  barrier (weight 50) is overpowered by the residual loss (still min clr 0.021mm).
  The crash-safe pyramid ceiling is **0.457** (coarse 7pt grid, confirmed across 3
  ring variants all worse).
- `--no-z-floor --no-trunc` recovers 0.9306 sphere / 0.8166 pyramid but with
  `holder_overlap` in the millions — physically infeasible, a real crash.
- Higher holder penalty weight (500, 10× default) + margin 0.04 on the pyramid:
  REGRESSES to 0.417 (over-constrains optimizer, pushes holder away). Default
  weight 50 is optimal — the champion's moves don't violate clearance.

## Methodology notes

- The viz "carved voxels: sim=N" line is the **remaining solid** count
  (`stock<0`), not removed — read accordingly.
- Pyramid best-checkpoint is at **iter 199** at 1in @200-iter (optimization HELPS:
  0.40 → 0.526, plateauing — 25-iter 0.457@iter15, 50-iter 0.492@iter45, 100-iter
  0.517@iter80, 200-iter 0.526@iter199; huge gains in first 80 iters, diminishing
  after), but at **iter 0** at 0.75in @25-iter (init strong → opt collapses, like
  the sphere) AND at 1in @max-steps=1536 (0.417, init saturates the larger budget
  → opt collapses). The opt-helps pattern is a symptom of a marginal init (shape
  × max-steps × stock size), not a pyramid invariant.
- Crash-safe pyramid `holder_overlap=0`, trunc no-trim (min clearance 21mm) —
  the path is genuinely collision-free; the loss is purely the unreachable waste.
- **NON-DETERMINISM at 1.5in pyramid (important caveat):** the 1.5in pyramid
  optimization varies ~0.1 across GPU-state regimes — same config/seed/code, the
  init is deterministic (iter0=0.319 reproducibly) but the optimizer trajectory
  diverges (original 25-iter gave best@iter20 0.431; two re-runs gave 0.328 @iter10
  IDENTICALLY; a 200-iter gave 0.344). Consecutive runs agree exactly (regime-
  stable), but cross-regime varies ~0.1 (CUDA atomic scheduling). The true 1.5in
  ceiling is ~0.33–0.43; single-run 1.5in comparisons are unreliable. **The 1in
  pyramid IS regime-robust** — independent 100-iter and 200-iter runs (different
  times) agree to 4 decimals at iter80 (0.5171), so the 1in 0.526 ceiling is
  solid. The 0.75in is init-determined (best@iter0). Likely cause: the rough 1.5in
  loss landscape makes the optimizer path-sensitive to atomic-add noise.
- **Sphere 0.819 ceiling CONFIRMED solid (not under-optimized):** a 200-iter
  sphere run gives best@iter0 0.819 (opt collapses: 0.819→0.55 @iter20→slow
  recovery 0.607 @iter199, never approaching 0.819). Unlike the pyramid (which
  had hidden opt headroom: 0.457→0.526), the sphere's init IS its ceiling. The
  opt-helps pattern is genuinely **pyramid-specific (1in only)**; sphere/cyl/box
  are init-determined (best@iter0, opt hurts even at 200 iters).
- **Training-barrier vs trunc-threshold mismatch**: the holder collision barrier
  (`holder_margin`, default 0.0) only prevents holder *penetration* during
  training (`holder_overlap=0`), but `truncate-collision` requires *1.0mm
  clearance*. With margin=0 the optimizer legally brings the holder to ~0.02mm
  of the stock, then trunc trims trailing steps. Set `--holder-margin ~0.04`
  (1mm/stock) so the barrier engages at the trunc threshold; the optimizer then
  keeps the holder ≥1mm clear. This is what lets low-zb (face-hugging) inits
  survive trunc instead of being decimated.

## Still open

- The crash-safe ceilings are **tool-to-stock-ratio limited** (confirmed): the
  pyramid rises 0.431→0.457→0.556 and the sphere 0.705→0.819→0.843 as the stock
  shrinks 1.5→1→0.75in (25mm tool). The ratio theory is verified — 1.5in@50mm
  (ratio 1.31) ≈ 0.75in@25mm (ratio 1.31) → 0.828 vs 0.843. Recovering ≥0.9
  crash-free at 1in would require a relatively taller tool (a 50mm tool only
  reaches 0.840; the init doesn't fully exploit it) or an articulated holder —
  out of scope for this sim.
- A **tool-aware zlayer init** for a longer tool was tested and does NOT help:
  the 50mm train dice is identical to 25mm (the gain is crash-safety, not
  reachability), and doubled revs regress to 0.652 (gouge). 0.840 is the
  1in+50mm sphere ceiling. **An extreme 75mm tool (ratio 2.95) also gives 0.8396
  — saturates at 50mm; 0.840 is the ceiling for any tool ≥50mm.**
- The `holder_margin` soft barrier (weight 50) is overpowered by the residual
  loss — but raising it does NOT help (see dead lever above): a 10× barrier
  (weight 500 + margin 0.04) REGRESSES the pyramid to 0.417 (over-constrains the
  optimizer, pushes the holder away). Default weight 50 / margin 0 is optimal.
  The face-hugging-low-zb path stays closed because of **coverage geometry**
  (single-radius bands), not the barrier weight. A hard holder clamp (analogous
  to the z-floor) remains untested but would only recover the dead ring variants
  to ~0.42, still below the 0.457 coarse-grid champion — not a path to beating
  the ceiling.
