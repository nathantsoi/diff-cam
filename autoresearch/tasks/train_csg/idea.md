# idea.md — jul13-phys-plausible: physically plausible paths (don't snap the end mill, don't snap the part)

Branch `ar-agd/jul13-phys-plausible`, worktree `.claude/worktrees/phys-plausible`.
Starting point: `ar-agd/jul6-step-detail` head (58b6c10 — sweep method, STEP-grid
targets, raster_arc/raster_terrain inits, reachability ceilings) **merged with**
`origin/autoresearch` (88b8289 — Nathan's trajectory-quality measures
w_time/w_air_time/w_break + kc/f_ref/sigma_risk/f_max force model, jul8 RLHF
feedback loop, jul10 w_tool_gouge warmup, difficulty-normalized dice).
**Every run needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib`** (WSL2 CUDA).
**GPU is shared with a second Claude instance this session — check load before
launching, alternate runs.**

## The problem this campaign attacks

The step-detail campaign showed the sweep planner reaches 0.98 dice on rrph and
0.82 on titan — geometrically. But nothing in the objective knows machining
physics: an optimized segment may take a full-depth full-width slot cut, plunge
straight down into material, or side-load the rrph pin / titan lettering with a
heavy cut. On a real Haas these paths snap the end mill, or snap the part's
delicate features ("end bits"). Two physics gaps:

1. **Tool-side**: nothing bounds per-segment material removal (chip load /
   cutting force). Nathan's w_break model exists for the DELTA method (it reads
   the per-step stock history `stock[t]`/`stock[t+1]`), but the sweep method
   never materializes per-step stock — carve = min over ALL segments of the
   swept SDF, order-free. The breakage surrogate must be re-derived for the
   sweep architecture (sequential chip attribution).
2. **Part-side (new, no model exists)**: slender target features (rrph pin,
   titan letter strokes) have finite bending strength. A path that engages
   heavily next to an already-freed thin feature snaps it. Physics: cantilever
   root stress σ = M·c/I; breakage force scales ~t²·(strength)/h for a wall of
   thickness t, height h. The plan must (a) know where fragile features are
   (computable from the target SDF alone — shape-agnostic), and (b) keep
   engagement light near them and/or order cuts so features stay supported by
   surrounding stock as long as possible.

## Plan (draft — research synthesis in progress)

- Port a per-segment engagement/force surrogate to the sweep method:
  sequential chip attribution (a voxel is cut by the EARLIEST segment whose
  swept tool covers it, not the argmin-deepest), giving seg_engage[t] without a
  (T+1)×N³ history; then reuse Nathan's F = kc·chip_vol/(dt·D) → lognormal
  P_break aggregation as a differentiable penalty.
- Fragility field from the target SDF: local thickness (inscribed-sphere via
  SDF magnitude on the medial band) + height-above-root → allowable side force
  per voxel; penalize per-segment force weighted by adjacency to fragile
  target voxels.
- Init/pathing constraints (constructive, not just penalties): bound stepdown
  per z-layer (a_p ≤ f(D)), forbid vertical plunges (ramp/helix entry), keep
  raster_terrain's gouge-free property.
- Report hard (non-differentiable) violation diagnostics alongside dice so
  physical plausibility is a tracked metric, not a vibe.

## Log

- [setup] Created branch/worktree from 58b6c10, merged origin/autoresearch
  (conflicts: train_csg.py optimizer guard × feedback warm-start; run_pipeline
  runs_subdir forwarding; web dashboard ctrl-points × stock/target overlays —
  all unioned). findings.md removed, results.tsv truncated per protocol.
- [research] Two literature scans launched (force-bounded toolpath generation;
  workpiece-side fragility / thin-feature machining). Synthesis pending.
