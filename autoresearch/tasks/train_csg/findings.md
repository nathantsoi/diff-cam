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
**Pyramid regresses hard** (0.817 → ~0.46): its old win relied on a deep
tool-base plunge that is now (correctly) forbidden — that plunge was a real
machine crash. The crash-safe pyramid ceiling is **structural**, not a coverage
bug (confirmed below). Crash-safe pyramid improved 0.416 → 0.457 via a 2D
descending-annulus boustrophedon (replacing the measure-zero 1D orbit), and
optimization HELPS the pyramid (pure-init 0.402 → 25-iter 0.457, best@iter15),
unlike sphere/cyl/box where init=iter0 is best.

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

## Pyramid: why ~0.42 is the crash-safe ceiling (NOT a coverage bug)

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
the sphere's lower-interior wedge — a fundamental crash-safe ceiling, not a
tuning problem.

## Dead levers confirmed / added

- Higher-T coverage for pyramid (revs/osc/T): does NOT help (ceiling is
  reachability, not density). 0.4159 → 0.4102.
- 1D orbit vs 2D boustrophedon for the pyramid annulus: same ~0.42 (the
  reachable outer annulus is captured either way; the inner band is unreachable).
- `--no-z-floor --no-trunc` recovers 0.9306 sphere / 0.8166 pyramid but with
  `holder_overlap` in the millions — physically infeasible, a real crash.

## Methodology notes

- The viz "carved voxels: sim=N" line is the **remaining solid** count
  (`stock<0`), not removed — read accordingly.
- Pyramid best-checkpoint is at **iter 15** (NOT iter-0): optimization HELPS the
  pyramid (0.33 → 0.42), unlike sphere/cyl/box where init=iter0 is best and soft
  opt collapses it. The pyramid soft loss is partially correlated with hard dice.
- Crash-safe pyramid `holder_overlap=0`, trunc no-trim (min clearance 21mm) —
  the path is genuinely collision-free; the loss is purely the unreachable waste.

## Still open

- A **shorter tool** is worse (less reach). A **longer tool** reaches deeper but
  gouges (full-height carve through the body). No tool_height recovers the slab
  crash-free.
- The pyramid crash-safe ceiling (~0.42) is accepted as structural. Recovering
  ≥0.8 would require either a non-full-height tool model (scenario change) or
  an articulated holder — out of scope for this sim.
