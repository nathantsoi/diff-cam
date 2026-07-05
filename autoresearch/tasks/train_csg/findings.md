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
Crash-safe pyramid improved 0.416 → 0.457 via a 2D descending-annulus
boustrophedon. Optimization HELPS the 1in pyramid (0.402 → 0.457, best@iter15)
but hurts the 0.75in pyramid (best@iter0, like the sphere) — the init is strong
enough at 0.75in that opt collapses it.

## Crash-safe per-shape results (viz carve Dice, post-trunc)

| shape    | infeasible (old) | crash-safe | notes |
|----------|------------------|------------|-------|
| sphere   | 0.9306           | 0.819      | lower-interior wedge + below-part slab un-carvable without the deep plunge |
| cylinder | 0.9390           | 0.916      | z-invariant; floor barely hurts |
| box      | 0.9014           | 0.892      | square orbit at equator; floor barely hurts |
| pyramid  | 0.8166           | 0.457      | sloped-face inner band unreachable crash-free (see below); improved 0.416→0.457 via 2D annulus boustrophedon |
| **mean** | **0.897**        | **~0.771** | |

All crash-safe runs: `holder_overlap=0`, trunc trims only genuine near-collisions
(cyl 19 steps @ 0.15mm; sphere/box holder-clear runs byte-identical pre/post
trunc — verified `trajectory.untruncated.npy == trajectory.npy`).

## Size robustness (1in → 1.5in stock, proportional target)

The method **generalizes** across stock sizes (crash-safe, best@iter0 init-peak
pattern holds for sphere/cyl/box; opt-helps for pyramid), but absolute dice
**drops with larger stock** because the fixed 25mm tool spans less of it
(≈1.0 stock-heights at 1in → 0.66 at 1.5in). The drop is **shape-dependent**:

| shape    | 1in    | 1.5in  | Δ      | crash-safe? | notes |
|----------|--------|--------|--------|-------------|-------|
| box      | 0.892  | 0.884  | -0.008 | yes (1.69mm) | z-invariant square orbit carves full-height sides → size-robust |
| cyl      | 0.916  | 0.842  | -0.074 | yes (1.69mm) | curved side; wider annulus at 1.5in less fully covered |
| sphere   | 0.819  | 0.705  | -0.114 | yes (1.69mm) | 3D interior; tool spans less → more lower-interior unreachable |
| pyramid  | 0.457  | 0.431  | -0.026 | yes (10.6mm) | already ceiling-limited; barely changes |
| **mean** | **0.771** | **0.716** | -0.055 | | |

Takeaway: flat/z-invariant surfaces (box) are size-robust; curved/3D surfaces
(sphere, cyl) lose dice as the tool-to-stock ratio falls. The 1.5in runs are
crash-safe by construction (trunc no-trim, holder_overlap=0). Eval stage OOMs
on the 76³ grid (known) but viz produces the canonical carved-vs-target Dice.

The trend is **monotonic in stock size** (sphere tested at 3 sizes): 0.843 /
0.819 / 0.705 at 0.75in / 1in / 1.5in. At 0.75in the 25mm tool is *taller* than
the 19mm stock (z-floor dips below the part bottom, -0.0025), so the
lower-interior wedge becomes partially reachable → dice rises above the 1in
ceiling. Larger stock → tool spans less → more interior unreachable.

The **pyramid follows the same monotonic trend**: 0.556 / 0.457 / 0.431 at
0.75in / 1in / 1.5in. This **revises the earlier "structural/geometric ceiling"
framing** — the pyramid ceiling is *not* purely geometric; it is tool-to-stock-
ratio limited, same driver as the sphere. At 0.75in (tool taller than stock) the
sloped-face inner band becomes partially reachable and best@iter0 returns
(init=win, like the sphere), whereas at 1in the init is marginal so optimization
helps (best@iter15). The crash-safe ceilings are fundamentally reachability
ceilings set by the fixed 25mm tool vs the stock height.

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

## Methodology notes

- The viz "carved voxels: sim=N" line is the **remaining solid** count
  (`stock<0`), not removed — read accordingly.
- Pyramid best-checkpoint is at **iter 15** at 1in (optimization HELPS: 0.33 →
  0.46), but at **iter 0** at 0.75in (init strong enough that opt collapses it,
  like the sphere). The opt-helps pattern is a symptom of a marginal init, not a
  pyramid invariant — when reachability improves (smaller stock) the init
  dominates and opt hurts.
- Crash-safe pyramid `holder_overlap=0`, trunc no-trim (min clearance 21mm) —
  the path is genuinely collision-free; the loss is purely the unreachable waste.
- **Training-barrier vs trunc-threshold mismatch**: the holder collision barrier
  (`holder_margin`, default 0.0) only prevents holder *penetration* during
  training (`holder_overlap=0`), but `truncate-collision` requires *1.0mm
  clearance*. With margin=0 the optimizer legally brings the holder to ~0.02mm
  of the stock, then trunc trims trailing steps. Set `--holder-margin ~0.04`
  (1mm/stock) so the barrier engages at the trunc threshold; the optimizer then
  keeps the holder ≥1mm clear. This is what lets low-zb (face-hugging) inits
  survive trunc instead of being decimated.

## Still open

- A **shorter tool** is worse (less reach). A **longer tool** reaches deeper but
  gouges (full-height carve through the body). No tool_height recovers the slab
  crash-free at a fixed stock size.
- The crash-safe ceilings are **tool-to-stock-ratio limited**, not absolute: the
  pyramid rises 0.431→0.457→0.556 and the sphere 0.705→0.819→0.843 as the stock
  shrinks 1.5→1→0.75in (tool 25mm fixed). Recovering ≥0.8 crash-free at 1in would
  require a relatively taller tool (scenario change) or an articulated holder —
  out of scope for this sim.
- The `holder_margin` soft barrier (weight 50) is overpowered by the residual
  loss; a higher weight or a hard holder clamp (analogous to the z-floor) would
  let face-hugging low-zb inits survive trunc — untested.
