# Delta vs sweep on a real STEP part (rrph_hi)

The comparison cell that was empty: **the delta method has never been run on a
real CAD part at its own champion configuration.** Nathan's branches
(`origin/autoresearch` and every `ar-agd/*` campaign branch) carry no
`target_sdf_path` support at all -- grid/STEP targets exist only on this line of
work. The only prior delta-on-STEP runs (8 of them, local, July 6) predate both
the corrected `*_hi.npz` targets and the `multidepth_cavity` champion init, so
they cannot answer "would the existing method have solved this part anyway?"

This run answers it.

## Protocol

Delta method at its documented operating point (T=128, dt=0.45; T>=192 NaNs per
`train_csg.py`), with `--init-mode multidepth_cavity` -- the init Nathan's
human-feedback loop identified as the winner (only 5-star run on the hardest
shape, and SOTA on sphere). 5000 iters, `--eval-freq 25`, 3 seeds, on
`rrph_hi.npz` (0.603M voxels, 25.4 x 50.8 x 12.7 mm stock at 0.3 mm/voxel) --
the same target the sweep method solves and the candidate for physical milling.

Run from `ar-agd/jul13-phys-plausible`, the only branch carrying both grid
support and `multidepth_cavity`. Artifacts in `runs/delta-step-baseline/`
(trajectories retained for canonical replay), full log at `campaign.log`.

## Result

| seed | soft dice | hard dice | uncarved baseline | ASD | HD95 | air / total | wall |
|---|---|---|---|---|---|---|---|
| 1 | 0.7648 | **0.6870** | 0.6870 | 1.137 | 3.00 | **100.0 %** | 85 min |
| 2 | 0.7598 | **0.6870** | 0.6870 | 1.187 | 3.43 | **100.0 %** | 87 min |
| 3 | 0.7644 | **0.6870** | 0.6870 | 1.147 | 3.00 | **100.0 %** | 84 min |

Across the whole campaign -- 15000 iterations, 600 evaluations -- hard dice took
exactly **two** distinct values: 0.7103 (the initialization, at iteration 0) and
0.6870 (the uncarved stock, at every evaluation thereafter).

**The delta method's best deployable result on this part is its own
initialization.** After iteration 25 the executed trajectory removes zero net
material: `hard_dice == dice_baseline` exactly, and `air_time == total_time`, so
the tool spends 100 % of the program cutting air. Meanwhile soft dice climbs
monotonically to ~0.765 and the training loss falls 1.83 -> 0.26. The optimizer
is making confident progress on a surrogate that has detached from the
deployable carve.

This is not an evaluator artifact: the same eval pipeline scored the init at
0.7103, above baseline, at iteration 0 -- it can see real carving on this
target. The path genuinely left the part and never returned.

## Against the sweep method, same target

| method | hard dice | budget | note |
|---|---|---|---|
| delta (champion init, T=128) | **0.6870** (best-ever 0.7103) | 85 min x 3 seeds | never carves; 100 % air |
| sweep (raster_arc, T=832) | **0.9716 / 0.9721 / 0.9503** | ~15 min x 3 seeds | 0.970 reachability ceiling |
| sweep (raster_arc, T=832, tuned) | **0.9753** | ~15 min | at ceiling |

The sweep method reaches ~100 % of the exact 3-axis reachability ceiling in a
sixth of the wall-clock, on the same part, same evaluator.

## Why (the mechanism, not the scoreboard)

Two independent causes, both structural rather than tuning:

1. **Soft/hard detachment.** The delta forward chains T `smooth_max` unions,
   each adding ~log(2)/k erosion. The training signal is a carve that the
   deployed hard carve does not reproduce; here the gap is not a discount but a
   disconnection -- soft 0.765 against hard 0.687. The sweep method's one hard
   union is exactly the fix, and this run is the clearest evidence for it.

2. **Path budget.** At T=128 and dt=0.45 the executable path is
   (T-1) * feed * dt ~ 244 mm. Covering rrph's surface needs ~1600 mm. The delta
   method is ~6.5x short before optimization begins, and it cannot lengthen its
   path -- T is bounded above by SDF-overflow NaNs at T>=192. The sweep method
   sizes T from the init's arc length (T=832 here) because its memory does not
   scale with T.

Cause 2 is the deeper one: it means no amount of delta-side tuning fixes this.
The path-length ceiling is set by the method's numerics, and the memory model
that would allow a longer path is the thing the sweep formulation removes.

## Honest caveats

- Delta was run at its documented operating point, not exhaustively tuned on
  this target. A different `--dt`, `--init-scale`, or loss weighting might carve
  *something*. The claim is not "delta cannot exceed 0.687 under any settings";
  it is "at the configuration its own research loop selected as champion, on a
  real part, it does not carve at all across 3 seeds."
- `multidepth_cavity` was designed and validated on analytic CSG shapes
  (sphere / cylinder / hole / bowl). Transferring it unchanged to a STEP part is
  the fair test of generalization, and it is the test that fails.
- `peak_vram_mb` reads 0.0 in these run dirs: this branch predates the
  Taichi-inclusive VRAM instrumentation. Memory numbers come from the TACC
  scaling sweep, not from here.
