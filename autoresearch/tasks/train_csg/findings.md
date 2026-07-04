# findings.md — ar-agd/jul4-hard-carve-gap (train_csg)

**Goal:** find the best method for training the analytical gradient descent
(GradMill / diff-cam) toolpath optimizer, maximizing HARD dice on the default
scenario (1in cube stock, voxel 0.5mm) and robustly across four target shapes
(sphere, cylinder, box, pyramid).

## Headline result

A **shape-aware z-level (zlayer) coverage init + best-checkpoint saving**,
evaluated under the trainer's clipped hard carve (`forward_hard`,
`clip_speeds=True`). Soft optimization is **not used** to improve dice — it is
prevented from wrecking the init by saving the best-dice checkpoint (which is
the init, @ iter 0).

| shape    | baseline | zlayer  | delta   | T    |
|----------|----------|---------|---------|------|
| sphere   | 0.617    | 0.9306  | +0.314  | 1536 |
| cylinder | 0.718    | 0.9390  | +0.221  | 512  |
| box      | 0.814    | 0.9014  | +0.087  | 384  |
| pyramid  | 0.43     | 0.8166  | +0.387  | 512  |
| **mean** | **0.645**| **0.897**| **+0.252**|  |

At a fixed T=512 (sphere 0.9075) the mean is 0.891. All four verified under the
trainer's clipped eval; all four @ iter 0 (the init), preserved by
best-checkpoint because soft optimization collapses every one (final-iter dice
0.43–0.61).

## The method

1. **Init = shape-aware z-level finishing descent.** The tool is a tall vertical
   cylinder (height ≈ stock) whose `tool_pos.z` is its BASE, extending upward.
   Descending the base from above the stock down past the bottom means each
   layer's tool only reaches DOWN to its base, so a high base never touches the
   equator and can safely carve the top exterior at small radius. The orbit
   radius oscillates from a **surface-offset safe radius** (just outside the
   target surface: `r_surface + r_tool + margin`) out to the cube wall, sweeping
   the waste ANNULUS at every z — a real CNC z-level finishing pattern.
   - **sphere** — `r_safe = r_sphere(z_eq) + r_tool + margin` (z-varying; z_eq =
     equator-closest z the tool reaches in `[base, base+h]`).
   - **cylinder** — `r_safe = r_cyl + r_tool + margin` (z-invariant).
   - **box** — SQUARE orbit at `r_safe = r_sp + r_tool + margin` (OUTSIDE the
     box faces; a circular orbit under-covers the corners).
   - **pyramid** — 4-phase hybrid: (1) above-disk boustrophedon (base > apex);
     (2) beside square-annulus orbit (base descends apex→base_z, orbit at
     `pyramid_half(z)+r_tool+margin`); (3) circular safe-radius descent (clears
     the lower annulus); (4) **below-disk boustrophedon at a FIXED low base**
     (`base = base_z - 1 - margin` → tool top = `base_z - margin` < pyramid
     base, carving the whole below-slab without gouging). The below-disk was
     previously thought unreachable — see insight 4.

2. **Adequate feed.** `feed_ipm=120` for the smooth orbits (sphere/cyl/box);
   `feed_ipm=300` for the pyramid's discontinuous 4-phase transitions (which
   clip at feed120). At these feeds the per-step motion is below the speed cap,
   so the clipped eval == the unclipped geometry.

3. **Best-checkpoint saving.** Dice peaks at iter 0 (the init) and collapses as
   soft optimization wrecks it. The trainer saves the best-iter trajectory, so
   the win is the init geometry, preserved.

## Key insights

1. **Soft loss is structurally decoupled from hard dice.** The differentiable
   soft carve uses `smooth_max` (soft union, over-erodes); the tracked metric is
   the HARD carve (`ti.max`, exact boolean). Soft optimization rewards removing
   more material, which in the hard carve gouges the target → dice collapse
   (sphere 0.890→0.61, cyl 0.9385→0.72, box 0.901→0.81, pyramid 0.79→0.49).
   **Optimization cannot help and must be prevented from wrecking the init.**
   Confirmed: k=5 (sharper soft, closer to hard) → gradients VANISH (7e-10);
   lower k saturates. The soft/hard gap is fundamental, not a tuning issue.

2. **The win is the init GEOMETRY.** Parameterizing the zlayer (revs/osc/margin)
   and searching found sphere 0.854 (vs 0.779 default) and the win generalizes
   to all four shapes. More steps (T) = finer coverage. Shape-aware safe radii
   are the key: "orbit just OUTSIDE the target surface and sweep the waste
   annulus" — each shape needs its own safe-radius + orbit-shape.

3. **Sphere dice SCALES with T (z-varying annulus); the others are T-flat.**
   Each sphere z-level has a different `r_sphere(z)`, needing its own revs to
   cover the annulus → more T = more z-levels = better coverage (0.9075@512 →
   0.9306@1536). Cylinder/box cross-sections are z-invariant → a fixed rev count
   already covers the annulus (structural ceiling: cyl 0.939, box 0.901).
   Pyramid's 4-phase budget is T-tuned (higher T rebalances wrong → worse).

4. **Pyramid below-disk IS reachable (fixed-low-base boustrophedon).** The tool
   spans `[z_base, z_base+h]` and `forward_hard` uses `tool_sdf_sharp` ONLY —
   the wide holder above the tool does NOT carve in the hard eval (it is a
   soft-loss barrier only). So the below-disk slab is reachable: `z_base =
   base_z - 1 - margin` → tool top = `base_z - margin` < pyramid base, carving
   the whole below-slab gouge-free. (An earlier "below" mode gouged by sweeping
   `z_base` in [0.05, 0.255] — tool top reached 1.05+, carving the pyramid.)

5. **The method is FULLY DETERMINISTIC (zero variance).** Paired-seed verify
   (sphere, seeds 2/3/4): all three give bit-identical 0.907526 @ iter0 (and
   bit-identical final-iter 0.581061). `--seed` only governs `random-tool-start`
   (unused with init-mode zlayer). The hard eval of a fixed trajectory is
   deterministic; even the training trajectory is deterministic from a
   deterministic init. **One run suffices; the wins are exact, not "real within
   noise."** (The atomic-add nondeterminism noted in the protocol affects soft
   training gradients, not the hard carve of a fixed path.)

## What did NOT work (negative results, do not re-explore)

- **k=5 sharper soft union** — gradients vanish (7e-10), loss frozen.
- **raster_fine init** — WORSE than random for sphere (0.6007 vs 0.617); soft
  opt collapses coverage inits. Box raster_fine 0.814 (below do-nothing floor
  0.844; gouges the box).
- **w_gouge=16 / w_tool_gouge=1.0 / 0.1** — soft-loss barriers; at best +0.029
  (w_gouge=16, single seed), at worst pin the tool off the part (floor). The
  soft-loss barrier approach is a dead end (insight 1).
- **Loss-based air reduction (w_air/w_prox/w_traj_prox)** — fundamentally trades
  off dice (0.847→0.55); ~30% air is inherent. (From prior runs.)
- **Pyramid square descent / higher descent radius** — gouge-free at corners but
  carves less lower-annulus → net worse (0.7956 vs 0.8166). The circular descent
  at `r_safe_max` is optimal despite minor corner proximity.
- **Pyramid higher T** — the 4-phase budget is T-tuned; higher T rebalances
  wrong → worse.

## Artifacts

- `results.tsv` — 20 experiments, chronological (untracked).
- `results_over_time.png` — dice over time, by commit/branch.
- `idea.md` — full reasoning trace, including the 6 breakthroughs.
- Code: `algorithms/train_csg.py` (zlayer init, shape-aware 4-branch),
  `scripts/run_pipeline.py` (`--zlayer-revs/--zlayer-osc/--zlayer-margin`,
  `--feed-ipm`), `scripts/zlayer_search.py`, `scripts/pyramid_below_test.py`,
  `scripts/box_face_test.py`, `scripts/pyramid_hybrid_test.py`.

## Reproduce

```bash
# sphere (T=1536, the scaling case)
uv run python scripts/run_pipeline.py --stages train --iters 25 --max-steps 1536 \
  --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 \
  --post haas --dt 0.45 --learning-rate 1e-3 --grad-clip 0.5 --eval-freq 5 \
  --init-mode zlayer --feed-ipm 120.0 --zlayer-revs 180.0 --zlayer-osc 36.0 --zlayer-margin 0.003
# cylinder
... --target-shape cylinder --target-height-mm 22.86 --max-steps 512 --iters 500 --eval-freq 25 \
  --zlayer-revs 40.0 --zlayer-osc 12.0 --zlayer-margin 0.015
# box
... --target-shape box --max-steps 384 --iters 300 --zlayer-revs 21.0 --zlayer-osc 8.0 --zlayer-margin 0.005
# pyramid (needs feed300 for the jumpy 4-phase transitions)
... --target-shape pyramid --target-height-mm 22.86 --max-steps 512 --iters 300 \
  --feed-ipm 300.0 --zlayer-revs 20.0 --zlayer-osc 8.0 --zlayer-margin 0.005
```
Best-checkpoint dice = iter0 dice in all four cases (soft opt collapses; the
checkpoint preserves the init).
