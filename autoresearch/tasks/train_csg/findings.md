# Findings — jul13-phys-plausible: physically plausible sweep paths

Branch `ar-agd/jul13-phys-plausible` = `ar-agd/jul6-step-detail` (sweep method,
STEP targets) ⋈ `origin/autoresearch` (trajectory-quality measures, RLHF loop,
gouge warmup). Task: make the step-detail planner's paths obey machining
physics — bound cutting force (don't snap the end mill), respect slender part
features (don't snap the part's "end bits"), and forbid moves an end mill
cannot make (vertical plunges) — without giving up dice.

## Headline (rrph, 0.5 mm vox, 15-min budget, n=3 seeds)

| | baseline (SOTA config) | full physics package (e6) |
|---|---|---|
| hard dice | 0.9716 (1 seed) | **0.9723 ± 0.0015** (0.9704/0.9724/0.9740) |
| engaged plunges > 3° | **21.9 %** of cutting steps | **3.8–6.2 %** |
| peak cutting force | 65 N unconstrained (139 N in one ablation) | 73–91 N raw → **59 N scheduled** (cap 100) |
| wax fragile margin | 0.29 → part snaps | 0.58 raw → **1.11 scheduled → survives (3/3)** |
| cycle-time cost | — | **× 1.009** (2.2–2.4 % of steps slowed) |

**Physical plausibility is free on rrph.** e6 config:
`--iters 1900 --max-steps 960 --n-ctrl 160 --sweep-init raster_arc
--sweep-init-ramp --ramp-deg 3 --w-ramp 15 --w-force 5 --sigma-y 10
--w-fragile 60 --amin-refresh 4 [rrph_hi.npz grid config]` + automatic offline
feed scheduling (b16bd04, always on in the diagnostics).

## The method levers (all shape-agnostic: target SDF + tool geometry only)

1. **Sequential chip attribution** (`cut_seg`): a voxel is removed by the
   FIRST segment whose swept tool covers it — computed in the existing
   argmin pass, cached on the same `--amin-refresh` cadence, no (T+1)×N³
   history. Differentiable per-segment chip volume via the envelope trick at
   fixed attribution (exact at fixed attribution; bounded O(1) jumps at
   refresh — same convention as the cached argmin). Deepest-cover (argmin)
   attribution is continuous but reads re-swept channels as engagement — worse.
2. **Force surrogate** `F[s] = kc_eff · chip_mm³[s] / len_mm[s]` with
   `kc_eff = kc · feed / v_c`, `v_c = π·D·rpm/60` (`--spindle-rpm`, 5000).
   The spindle normalization matters: the naive chip-area force overestimates
   ~400× (27 kN for a full slot vs ~68 N correct) because cutting energy is
   spent at the spindle surface speed, not the feed speed.
3. **Penalties** (Taichi, inside the Tape, gated by weight like the delta
   trajectory-quality terms): `--w-force` vs the tool cap `--f-max`;
   `--w-fragile` vs per-feature caps splatted onto contact bands.
4. **Fragility field** (utils/fragility.py, pure numpy/scipy at startup):
   thin mask = part \ opening(part, ball r_tool) via two EDTs (+1 voxel
   dilation tolerance — without it, grid quantization reads phantom sliver
   features: a sphere reported 144); per-feature cantilever
   `F_allow = σ_y · t² · b_eff / (6h)` with lever arm h from geodesic-
   height-above-attachment and **b_eff = A_interface/t** (supported-wire
   rule: pins keep the t³ law, long-attached rim strips get credit for
   their attachment — with b=t the wax rims read 3.2 N, an unmachinable
   ask that lost dice AND still broke; with b_eff they read 17–122 N).
5. **Plunge lever**: `--sweep-init-ramp` (zigzag ramp entries in
   raster_arc/raster_terrain at `--ramp-deg`, replacing corner plunges) +
   `--w-ramp` (torch-side penalty on engaged descent steeper than ramp_deg,
   engagement gate from the chip attribution).
6. **Offline feed scheduling** (deployable close-out): F ∝ feed at fixed
   geometry → deterministically slow only the violating segments to 90 % of
   their cap (floor 0.2). Breakage removed by construction, dice untouched,
   cost = cycle time (×1.009 here). Saved as `feed_mult.npy`.
7. **Always-on hard diagnostics** on the reported trajectory:
   `fcut_seq_max, tool_broken_seq, plunge_count/frac, fragile_margin_min,
   part_broken, n_fragile_features` + scheduled variants + `[phys]` log lines.

## The real levers, in order of impact

1. **Path budget is the physics currency.** Ramped descent has an
   irreducible floor `z_span/tan(θ)` (rrph at 3°: ~250 mm; sphere: 437 mm ≈
   90 % of its T256 budget). At fixed T the levers COMPETE: e2 (ramp only,
   T832) cut plunges 4× but pushed force 65→139 N (coarser pitches → heavier
   chips). Sizing T to the init's own feasibility estimate (T≥907 → 960)
   dissolved the conflict. Corollary: the unguarded init auto-coarsening
   loop hung on the floor (fixed with a stop-when-not-shrinking guard).
2. **Feed scheduling closes what geometry cannot.** With feed fixed, force
   is pure engagement geometry, and weight-dialing w_fragile 5→20→60 moved
   the wax margin only 0.60→0.67→0.58 (dilution by /S mean + no path budget
   for extra light passes). The linear feed lever fixed it outright for 1 %
   cycle time. (A differentiable per-segment feed DOF is the natural
   upgrade; see next steps.)
3. **w_ramp + ramped init**: 21.9 % → ~4–6 % plunges at zero dice cost.
   Residual plunges: spline corner-rounding at ramp turnarounds + terrain
   drop-offs.
4. **Honest physics beats strict physics.** Two model corrections were the
   difference between "lever fails" and "lever free": the spindle force
   normalization (else every cut reads as tool-breaking) and the
   supported-wire b_eff (else supported rims read as unmachinable and the
   optimizer sacrifices dice to a fake constraint).

## Bugs found by the diagnostics (worth keeping)

- **Composite best-checkpoint selection is blind for sweep runs**: soft dice
  is identically ~0, so the score reduced to "smallest penalties" and picked
  an iter-50 path (hard 0.28) over the converged 0.97. Sweep now forces
  `--best-on-hard` (0b09ee0).
- Phantom sliver fragile features from matched-radius EDT opening (fixed,
  +1 vox tolerance, regression test).
- This GPU runs rrph T832 at ~2.5 it/s (not the prior campaign's implied
  ~13): size `--iters ≈ 2200` (T832) / 1900 (T960) for the 15-min budget.
  rrph converges by ~2000 iters anyway.

## Explicitly rejected / open

- **b=t cantilever caps** (rejected: treats attached strips as posts).
- **Attribution-band dilation (+0.5 vox)** for chip conservation (rejected:
  lets grazing segments steal deep-covered voxels; worse gradients).
- **Weight escalation for w_fragile** beyond ~20 (dead: budget-bound).
- rrph's pin is ~12 mm diameter — NOT fragile even in wax; rrph's fragile
  set is its rounded top rims. **Titan lettering (1–2 mm strokes) is the
  real part-side stress test — not yet run** (needs T≈2560 → the 2-slot
  stock-field VRAM fix from step-detail findings, plus ~40 min at this
  GPU's throughput).

## Next steps (ranked)

1. **Differentiable per-segment feed DOF** (feed multiplier vector, sigmoid
   bounds, chained through the Tape like path grads): lets the OPTIMIZER
   trade time for force during training instead of post-hoc, and unlocks
   fragile-safe finishing without path-budget cost. The offline scheduler
   is its lower bound and already ships.
2. **Titan** with the physics package (after the 2-slot stock VRAM fix).
3. Plunge tail (4–6 %): penalize in the INIT's spline fit too (corner
   overshoot), or raise w_ramp with a warmup.
4. Port the fragility caps into Nathan's delta-method breakage loss (the
   two models share kc/f_max; the delta side has per-step stock so its
   attribution is exact).
5. Support-aware ORDERING (waterline/support doctrine as a soft objective
   over cut_seg timing) — deferred; contact-band force caps + top-down
   layers already encode most of the practice.

## Artifacts

- `results.tsv` (11 rows, exact commands), `results_plot.png` (dice +
  violation metrics per run), `idea.md` (chronological log incl. the
  research synthesis with citations).
- `runs/jul13-phys-plausible/` — 9 run dirs with metrics.json incl. the
  phys fields and feed_mult.npy.
- Key commits: 1312f52 (physics levers), 1ebe653/ca6f769 (fragility fixes),
  4e90679 (spindle force scale), 0b09ee0 (best-on-hard), b16bd04 (feed
  scheduling), be50842 (log).
