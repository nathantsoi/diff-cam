# idea.md — ar-agd/jul4-toolholder-collision

Branch: `ar-agd/jul4-toolholder-collision` (created fresh from `autoresearch` on
2026-07-04, HEAD = `3230c17` "handle toolholder collision on depth").

## Why this run

The simulator just gained **toolholder-collision handling** (commit `3230c17`).
This run's job is to re-establish the proven operating point *under* the new
collision safety, and find out whether the prior zlayer wins survive it.

What changed in the sim:
- **Z-floor clamp** (`--z-floor-epsilon-mm`, default 1.0mm; `--no-z-floor` to
  disable). The executed tool BASE z is clamped to `part_bottom_z - eps/stock_z`
  so the holder (which rides above the base by `tool_height`) cannot plunge into
  the remaining stock — a machine crash. Like the feed/rapid speed clip, it is a
  subgradient: gradient flows above the floor, zero below.
- **Holder swept-Z union** (`holder_sdf_sharp` + soft). The holder Z extent is
  now unioned over the swept segment `[min(bottom_a, bottom_b),
  max(bottom_a, bottom_b) + h]` instead of evaluated at a single `h_param`. This
  fixes a real bug: on near-vertical segments (deep plunge) the old single-point
  SDF read the holder as clear while it actually swept through material.
- **`trunc` pipeline stage** (`algorithms/truncate_collision.py`; default
  `--stages train,trunc,eval,export,viz`). After training, hard-carves the
  trajectory step-by-step, measures holder-to-stock clearance per segment, and
  stops the toolpath a `--collision-clearance-mm` (default 1.0) margin *before*
  the first collision. The truncated path replaces `trajectory.npy` (original
  saved as `trajectory.untruncated.npy`).

## Starting point (carried from jul4-hard-carve-gap)

The proven operating point is **shape-aware zlayer coverage init + best-checkpoint**.
The win is the init GEOMETRY — the differentiable soft loss is structurally
decoupled from hard dice, so soft optimization collapses the init and must be
prevented from wrecking it (best-checkpoint saves iter-0). Fully deterministic
(bit-identical across seeds; `--seed` only governs the unused random tool-start).

Prior HARD-dice results (best-checkpoint @ iter0, the tracked metric):

| shape    | baseline | zlayer  | delta   | T    |
|----------|----------|---------|---------|------|
| sphere   | 0.617    | 0.9349  | +0.318  | 2048 |
| cylinder | 0.718    | 0.9390  | +0.221  | 512  |
| box      | 0.814    | 0.9014  | +0.087  | 384  |
| pyramid  | 0.43     | 0.8166  | +0.387  | 512  |
| **mean** | **0.645**| **0.898**| **+0.253**|  |

Method: shape-aware zlayer coverage init (per-shape safe-radius + orbit shape) +
adequate feed (120 for smooth orbits, 300 for the pyramid's jumpy 4-phase
hybrid) + best-checkpoint. Sphere scales with T (z-varying annulus); cyl/box/
pyramid at structural ceilings (T-flat).

**⚠️ Caveat carried forward**: the pyramid 0.8166 relied on a **deep tool-base
plunge** (z_base ≈ -0.73) to carve the below-part slab — exactly the crash the
new z-floor prevents. The pyramid number is the one most likely to change under
the new collision handling. Re-baseline it first.

## Plan

1. **Baseline under new defaults** — run the default scenario (sphere, lr1e-3)
   with the new `--stages train,trunc,eval,export,viz` defaults (z-floor on,
   trunc on). Confirm the z-floor print (`[z-floor] ...`) and that the `trunc`
   stage runs without trimming a non-colliding trajectory. This is the
   reference for everything after.
2. **Re-establish the zlayer wins under collision safety** — for each shape,
   run the zlayer init config from the table above and check whether the
   iter-0 hard dice survives the z-floor clamp + trunc stage. Expect sphere/
   cyl/box ~unchanged (orbits stay above the part); pyramid is the open
   question (its below-disk phase commands a deep plunge → may be floored or
   truncated). This is the highest-value test of the run.
3. **If pyramid regresses**: the below-disk carve is now crash-bounded.
   Investigate (a) `--z-floor-epsilon-mm` sweep (how much below-part travel is
   actually safe given the holder geometry?), (b) a shorter tool / different
   tool_height (changes the scenario — note it), (c) a below-spiral that stays
   above the floor. Goal: recover pyramid ≥ 0.8 crash-free.
4. **Verify the trunc stage never silently trims a good trajectory** — for each
   kept run, confirm `trajectory.untruncated.npy` == `trajectory.npy` (or that
   any trim is a genuine collision, not a false positive from the unioned
   holder SDF being conservative). A false-positive trim would quietly cost
   dice and must be caught.
5. Validate any real change across sphere/cylinder/box/pyramid. Wins are
   deterministic (init dice is seed-independent), so single runs suffice for
   the zlayer-init numbers; paired seeds only matter if optimization is
   actually kept on.

## Dead levers (do NOT re-explore — see autoresearch.md + prior findings)

w_air / w_prox / w_traj_prox (contour-hug losses trade dice 0.847→0.55; ~30%
air is the price of high dice), w_gouge (seed-reshuffling), w_jerk,
lr_decay_frac, dt0.5+m160 (single-seed fluke), raster_fine_wide, k≤2
(saturates; k=10 only viable value and at k=10 soft loss is decoupled from
hard dice), iters>5000 (marginal), finer voxel_size_mm, coarse structured
inits that violate the speed clip (only raster_fine survives), lr sweep
(exhausted, peak 1e-3), more steps of a SOFT-optimized path (T-invariant on
hard dice — but a COVERAGE path using more T is NOT ruled out).

## Notes / findings

_(chronological working log — fill in as experiments run)_
