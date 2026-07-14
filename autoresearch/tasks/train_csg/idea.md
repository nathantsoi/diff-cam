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
**Throughput on this box: rrph T832 runs ~2.5 it/s (NOT the step-detail
campaign's implied ~13 it/s — likely battery-throttled then too) → size
--iters ≈ 2200 for the 15-min budget. rrph converges by ~2000 anyway
(hdice 0.9751 at iter 3500 in the killed 12k run).**

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

## Research synthesis (two literature scans, 2026-07-13)

### Part-side physics (thin features breaking)

- **Failure model**: cantilever bending at the feature root. Wall (thickness t,
  height h, length b): σ = 6·F·h/(b·t²) → F_allow = σ_y·b·t²/(6h). Pin
  (diameter d): σ = 32·F·h/(π·d³) → F_allow = σ_y·π·d³/(32h). Cross-check
  from the scan: a 2 mm × 10 mm machining-wax pin fails at ~0.6 N — below
  ANY routine milling force (5–50 N); the same pin in Al 6061 (σ_y 276 MPa)
  fails at ~22 N — survives finishing, dies in roughing. So fragility is
  material- and geometry-scaled, exactly what a per-voxel F_allow field captures.
- **Practice tiers** (Sandvik/Forbes/Materials-2020 review, all agree): thin =
  H:T > ~8:1; ≤15:1 → alternate-side non-overlapping passes; 15–30:1 → step
  support / waterline (finish each z-level while lower stock is intact);
  >30:1 → "christmas tree" (thicker below supports thinner above). Axial step
  ≤ 8× wall thickness (steel) / 4× (non-ferrous, per Forbes). Numerical
  validation (Sci. Reports 2024): waterline vs side-by-side = 247 μm vs 670 μm
  max wall error — a 2.7× penalty for cutting full-height beside unsupported
  walls. Doctrine: **the uncut stock IS the fixture** — heavy cuts while
  supported, light cuts (0.08–0.15 mm) once thin.
- **Sequencing literature**: Wang/Ibaraki/Matsubara (Precision Eng. 2017)
  formalize exactly our ordering problem — remove blocks such that
  δ = F/k(removal-state) ≤ δ_max at every step. Stiffness scales with wall
  thickness CUBED (Del Sol review, Materials 2019). Smith (CIRP 2012):
  sacrificial support structures machined away last. Blisk practice: finish
  tip-down, lower sections left rough for stiffness.
- **Fragility from a voxel/SDF grid** (Telea & Jalba ISMM 2011 — the recipe):
  the target SDF is already the EDT. (a) thin mask = part \ opening(part, ball
  r) = two distance-transform thresholds, O(N); (b) per-voxel local radius from
  the EDT; (c) geodesic height above the attachment interface (BFS from the
  rump interface through the thin component) = the cantilever lever arm; their
  breakage-likelihood metrics are ratios of exactly these. All cheap
  (scipy.ndimage.distance_transform_edt, sub-second at our grids), pure
  geometry — **shape-agnostic, task-rule-clean**.

### Tool-side physics (snapping the end mill)

- **Mechanistic force**: average cutting force F ≈ kc · A, where A = engaged
  chip cross-section (mm²) = removed volume per mm of travel, kc = specific
  cutting force (Al ~700–800 N/mm², already the repo default kc=700). This is
  dimensionally exact Newtons and reduces to the textbook F = kc·a_p·a_e for a
  slot/side cut. Nathan's delta-method model uses the same kc with a
  vol/(dt·D) form; per-mm-of-travel normalization is cleaner for variable
  segment lengths and gives the same calibration point.
- **What CAM bounds**: radial engagement (adaptive clearing holds engagement
  angle ~ constant, a_e often ≤10–20% D for finishing, slotting = worst case),
  axial depth (a_p ≤ 0.5–1.5 D), and **plunging**: an end mill cannot feed
  axially like a drill — practice is ramp entry at ~2–5° or helical entry.
  A vertical plunge into stock is the single most implausible move our
  optimizer (and our raster_arc init, at every layer corner!) produces.
- Nathan's stress-strength interference (lognormal, P_break = σ((ln F −
  ln f_ref)/σ_risk), trajectory 1−exp(−ΣP)) is a good aggregation; keep his
  constants/CLI and re-derive only the per-segment F for the sweep method.

## Design: physics for the sweep method

The sweep carve is order-free (min over segments), so per-segment physics
needs **sequential chip attribution**: physically, segment s removes the
voxels it covers that no earlier segment already removed:

    cut_seg(x) = min { s : d_s(x) < 0 }        (first-covering segment)

computed in the same O(T·N³) brute-force pass as `find_argmin` (early-exit at
the first covering s), cached and refreshed on the same `--amin-refresh`
cadence. NOT the argmin (deepest cover) — argmin drives geometry gradients,
first-cover drives physics attribution. From it:

1. **seg_chip[s]** (differentiable, Taichi): Σ over voxels with cut_seg==s of
   soft-occupancy(−d_s) · v³ — envelope trick at fixed attribution, same
   pattern as the cached argmin. Per-segment force surrogate
   **F[s] = kc · seg_chip[s] / len_mm[s]** (chip area × specific force).
2. **w_force** (tool breakage): penalty Σ_s softplus(F[s]/f_cap − 1)² with
   f_cap = f_max (Nathan's threshold, default 100 N). Gradient pushes control
   points to spread removal across passes (shallower stepdown, narrower
   engagement) exactly where force spikes.
3. **w_fragile** (part breakage): host-side static field from the target SDF —
   F_allow(x) per part voxel via the cantilever formulas above (thin mask ×
   geodesic-height lever arm, σ_y CLI-tunable, default Al 276 MPa); then
   f_allow_near(x) for waste voxels = min F_allow over part voxels within
   contact distance (grey erosion, host-side, once). Penalty
   Σ_s softplus(F[s]·max_{x: cut_seg=s} finv_near(x) − 1)², i.e. a segment
   cutting adjacent to a fragile feature gets a *tighter* force cap — "light
   passes near thin walls". finv per segment comes from the non-diff
   attribution pass (constant between refreshes); only F[s] carries grad.
4. **w_ramp** (plunge feasibility, torch-side): per step, penalize descent
   steeper than ramp_deg (default 3°) while engaged:
   relu(−Δz_mm − tanθ·|Δxy_mm|)² gated by seg_chip[s] > 0 (detached gate).
   Plus a constructive fix: raster_arc/raster_terrain layer entries currently
   PLUNGE vertically at the serpentine corner — add ramped layer entry
   (`--sweep-ramp-entry`) that descends the stepdown along the first scan row.
5. **Hard diagnostics** (always on, deployable metrics): per-segment hard chip
   volume on the binary carve → F_hard[s]; report diag_fcut_max,
   diag_tool_broken (any F_hard > f_max), diag_plunge_count (engaged steps
   steeper than ramp_deg), diag_fragile_margin (min over segments of
   f_allow_near/F_hard), diag_part_broken. These make physical plausibility a
   tracked metric of every run, penalties on or off.

Order-of-operations (waterline/support-aware sequencing as a differentiable
objective over cut_seg TIMING) is the stretch lever — noted for later; the
force-cap-near-fragile penalty plus top-down raster_terrain layers already
encode most of the practice doctrine for our 3-axis one-setup regime.

### Experiment sequence

1. **Measure first**: re-run step-detail best configs (rrph raster_arc T832;
   titan raster_terrain T1536 if budget allows) with diagnostics only —
   quantify how implausible the current SOTA paths are. This is the baseline.
2. w_force on (sweep force penalty) — dice cost vs F_max reduction curve.
3. w_fragile on — rrph pin / titan lettering exposure margin.
4. w_ramp + ramped-entry init — plunge count → 0 without dice loss (expected
   nearly free: entry moves are a tiny fraction of path length).
5. Combined operating point; seed-replicate the keeper.

## Log

- [setup] Created branch/worktree from 58b6c10, merged origin/autoresearch
  (conflicts: train_csg.py optimizer guard × feedback warm-start; run_pipeline
  runs_subdir forwarding; web dashboard ctrl-points × stock/target overlays —
  all unioned). findings.md removed, results.tsv truncated per protocol.
- [research] Two literature scans launched (force-bounded toolpath generation;
  workpiece-side fragility / thin-feature machining). Part-side scan landed in
  full (synthesis above); the tool-side scan died on an API session limit —
  its gap is covered by Nathan's in-repo force model + standard mechanistic
  forms.
- [impl] Physics levers implemented (commit follows): utils/fragility.py
  (per-feature cantilever F_allow from the target SDF, contact splat,
  tool-center lookup field); sweep.py first-cover cut_seg attribution in the
  argmin pass + seg_chip/force-penalty kernels under the Tape (w_force,
  w_fragile) + ramped layer entry in raster_arc/raster_terrain (ramp_deg);
  train_csg.py flags + torch-side w_ramp plunge penalty + [phys] hard
  diagnostics (fcut_area_max, plunge_count/frac, fragile_margin_min,
  part_broken) on the reported trajectory; run_pipeline forwarding. 5 new
  tests + 11 existing all pass on CPU.
- [finding: the ramp floor] Plunge-free entry has an IRREDUCIBLE path-length
  cost: descending the part's z-span at ramp angle θ takes z_span/tan(θ) of
  horizontal travel regardless of layer count (sphere: 23 mm span at 3° =
  437 mm — 90% of the T256 budget of 486 mm; at 10° = 130 mm). This is real
  machining economics (why CAM sells helical entry and why plunging tempts),
  and it means physically plausible paths NEED either bigger T, faster dt, or
  a steeper ramp on soft materials. First smoke run hung on exactly this: the
  init auto-coarsening only shrinks serpentine pitches, which cannot buy back
  the ramp floor → guarded (bd commit), floor reported with actionable knobs.
- [bugfix] Fragility phantom features: matched-radius EDT opening left
  sub-voxel sliver shells on smooth thick surfaces (sphere read 144 phantom
  features, weakest 92 N). +1 voxel dilation tolerance kills them (real thin
  features clear it); thick-ball regression test added.
- [impl note] First-cover chip attribution is DISCONTINUOUS at attribution
  flips (a boundary voxel hands occ~0.5 between segments when a cover
  appears/disappears) — inherent to sequential semantics with overlapping
  passes. The gradient is exact at fixed attribution (same convention as the
  cached argmin); refresh jumps are bounded and absorbed by the amin-refresh
  cadence. Deepest-cover (argmin) attribution would be continuous but
  misattributes re-swept channels as engagement (air reads as chip) — worse.
  Soft chip volume is biased ~0.6-0.75x of hard at coarse grids (surface
  sigmoid, one-sided band) — a calibratable scale; ranking across segments is
  what the penalty uses (corr > 0.9 in tests).
