# Findings — jul6-step-detail: spline sweep on real STEP-file targets

Branch `ar-agd/jul6-step-detail` (built on the spline-sweep method from
`ar-agd/jul6-spline-sweep`). All numbers are hard-carve dice from the
untouched evaluator. Targets are real CAD parts converted by
`utils/step_to_sdf.py` (`--target-shape grid --target-sdf-path <npz>`).

## Headline

The one-shot spline-swept-volume method transfers to real CAD geometry, but
**detailed parts required four adaptations** (none of which change the
evaluator): a feed-feasible arc-length init sized to the part, a gouge-free
terrain-following init, argmin caching for long paths, and an eval
memory fix. With them:

| target | part | ceiling* | baseline sweep | best | best/ceiling |
|---|---|---|---|---|---|
| rrph (pin+hole, 0.3 mm vox) | 25×51×13 mm | 0.970 | 0.9681 | **0.9753** (raster_arc T832) | ~100% |
| titan (nameplate, 0.5 mm vox) | 138×49×19 mm | 0.965 | 0.7298 (T256, feed-starved) | **0.8189** (raster_terrain T1536) | 85% |
| extrusion (0.5 mm vox) | 20×100×20 mm | 0.648 | — | crashed (2 open bugs) | — |
| bowl (1.5 mm vox) | 260×260×76 mm | 0.342 | not run (ceiling too low to rank) | — | — |

*Ceiling = exact 3-axis reachability bound: part height field max-filtered by
the tool disc; a waste voxel is removable iff some tool placement covers it
without the cylinder clipping part above. Shape-agnostic, computed per target.

**rrph is solved** — 0.9753 exceeds the voxel-quantized ceiling estimate
(sub-voxel effects make the bound slightly conservative). The pin and the
through-hole are both carved: detail at 0.3 mm voxels with a 6.35 mm tool is
not the hard part; **path-length budget is**.

## What detailed parts actually changed (the user's question)

The core method (B-spline path, min-over-segments swept SDF, two-pass argmin
gradient, Adam on control points) transferred **unchanged**. What broke was
everything around it, all for one reason: **real parts need ~10× more toolpath
than CSG primitives**, and several O(T·N³) or O(N³·surface²) assumptions hid
in the pipeline.

1. **Eval OOM (infra, fixed)**: `surface_distances` built a dense cdist
   matrix — ~16 GB at 50k×40k surface points on hi-res grids; every grid eval
   died in swap. Replaced with exact cKDTree queries (identical values,
   tested). This alone made STEP targets runnable at all.
2. **Feed-feasibility (the big one)**: executable path length is capped at
   (T−1)·feed·dt (1.905 mm/step at dt 0.45). The CSG-era T=256 budget is
   488 mm; titan's raster init measured **7.4× over budget** (bowl 15×), and
   uniform-in-index spline sampling put single steps at 26–192× the cap at
   row wraps — the evaluator's speed clip then truncates them and the
   executed path diverges from the optimized one. Fix (`raster_arc` init):
   tool-sized serpentine z-layers **resampled uniformly in physical arc
   length** (every step ≤ cap by construction), T sized from length/cap,
   auto-coarsened pitches when the part demands more path than VRAM allows.
   rrph: T=832 → +0.007 over baseline (0.9753).
3. **Gouging inits**: a plain descending raster plows through raised features
   (titan's lettering) and the optimizer must climb out of a deep gouge basin.
   Fix (`raster_terrain` init): scan lines ride the *legal tool-base height*
   z = max(layer_z, height field max-filtered by the tool disc + ½ voxel) —
   gouge-free by construction, shape-agnostic (pure geometry). On titan it
   starts at dice 0.716 ≈ the old baseline's *final* plateau and reaches
   0.8189 (best at iter 440 of 620; later evals oscillate 0.806–0.815 —
   converged, not budget-cut).
4. **Throughput at large T**: brute-force argmin is O(T·N³) per iter.
   `--amin-refresh N` re-runs the full argmin every N iters and reuses the
   cached winning segment between (the envelope gradient stays exact at the
   cached argmin; it lags path movement by ≤N iters). N=4 ≈ 3× throughput.
5. **VRAM wall (identified, not yet fixed)**: the delta simulator allocates a
   (T+1)×N³ f32 stock **history** that sweep training never reads — 10.7 GB
   at T=2560 on titan's 1.04M-voxel grid → CUDA OOM. This caps T≈1536 (12 GB
   card). The fix is a 2-slot stock field when method=sweep; highest-value
   next change.

## Honest per-target verdicts

- **rrph 0.9753 / ceiling 0.970**: solved. Detail features (pin, hole) fine.
- **titan 0.8189 / ceiling 0.965 (85%)**: feed-starvation diagnosis was
  correct — T256→T1536 with a feasible, gouge-free init moved dice +0.089 in
  a single 620-iter run, where the T256 baseline had fully plateaued at
  0.7298 after 8k iters. The run itself converged (grad ~1e-2, evals
  oscillating ±0.01 around 0.81), so the remaining 0.15 gap is structural,
  not more-iters: the path budget still can't cover the full lettering, and
  24% of the executed path cuts air between pockets (air_cut_fraction 0.24).
  Needs the VRAM fix → T≥2560, and/or multi-spline retracts so air moves
  stop consuming feed budget. (Run took 27 min wall on the battery-throttled
  GPU — sized for 15 at 1.5 s/iter, ran at ~2.6.)
- **extrusion**: not a carving benchmark — 98% of its waste opens sideways
  (lies flat), ceiling 0.648, and uncarved stock already scores 0.644. It is
  a *gouge-avoidance* test. Two open bugs block it: CUDA illegal address at
  T1952 (large-T argmin suspect) and a lazy Taichi field creation in the
  reach-gate path ("create field after materialization").
- **bowl 0.342 ceiling**: curved underside is one big 3-axis overhang shadow;
  real machining flips the part (two setups) — out of scope for the current
  simulator. Any dice number would measure the ceiling, not the method.

## Method lessons (transferable)

- **Compute the reachability ceiling before optimizing.** It is cheap (two
  scipy filters), shape-agnostic, and completely reframes targets: it showed
  rrph/titan are winnable (~0.97), extrusion is ceiling-limited at 0.648, and
  bowl at 0.342 — preventing wasted tuning on structural gaps (the previous
  campaign burned experiments learning the sphere's 0.848 ceiling the hard
  way).
- **Feed-feasibility of the init is a hard prerequisite, not a regularizer.**
  The w_feed penalty can *maintain* feasibility but cannot repair an init
  that is 7× over budget — the optimizer trades feed violations against
  coverage and stalls (titan baseline froze at 0.7298 with grad ~1e-3).
  Sizing T from the init's arc length is the adaptation rule:
  T ≈ init_length / (feed·dt) (+ headroom), K ≈ T/6.
- **dt is a legitimate path-budget knob for the sweep method** (titan used
  dt 1.5 → 6.35 mm step cap): the carve is exact per segment regardless of
  sampling density; only spline-sampling resolution coarsens. This was a
  dead-end lever for the delta method but is safe here.
- Single-seed caveat: all grid-target numbers are 1 seed on 1 GPU
  (finish-up constraint); the prior campaign's ±0.0002 sweep-method seed
  spread suggests low risk, but the rrph +0.007 margin should be
  seed-replicated before being cited as a ceiling-beating result.

## Artifacts

- `results.tsv` — all runs incl. crashes, exact commands.
- `results_plot.png` — progress + best-per-scenario vs ceilings.
- `runs/jul6-step-detail/` — run dirs (metrics.json, trajectory.npy, STLs).
- `utils/NPZs/*_hi.npz` — regenerated targets (correct solids, padding 0,
  bowl z-up); conversion commands in idea.md.
- `utils/reachability.py` + ceiling numbers in idea.md.
- Key commits: 363675f (grid pipeline + NPZs), 72383b3 (KD-tree metrics,
  raster_arc, amin refresh, reach gate), d6fd796 (raster_terrain).

## Next steps (ranked)

1. **2-slot stock field for sweep** — removes the (T+1)×N³ history, unlocks
   T≥2560 → titan full lettering coverage; the main lever on the remaining
   0.15 gap to titan's ceiling.
2. Multi-spline union with safe-z retracts (future_work.md #1) — titan's
   separated letter pockets are exactly its use case (air-skip), and it
   reduces required T.
3. Fix reach-gate lazy field bug (move mask field creation to __init__) and
   the T1952 illegal address (suspect i32 overflow or block-dim in the
   argmin kernel at T·N³ ≈ 6.6e8·segments scale).
4. Seed-replicate rrph/titan bests (≥3 seeds).
5. Bowl: only meaningful after a part-flip (second setup) exists in the sim.
