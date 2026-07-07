# future_work.md — ranked, testable improvements for the sweep method

Synthesized from two literature scans (2026-07-06): differentiable swept-volume
geometry, and volumetric coverage path planning / CAM practice. Every proposal
respects the task rules (shape-agnostic — target SDF grid/bbox only, no shape
names; no new packages; eval untouched; 15-min budget) and is assessed against
where the headroom actually is: **pyramid 0.829 (tapered walls) and the
single-spline no-retract limitation** — sphere is at its ~0.848 3-axis ceiling.

## Combined top picks (payoff / effort)

### 1. Multi-spline union with safe-z pinned retracts  [effort M, payoff HIGH]
The single spline cannot lift the tool and rapid elsewhere; the field treats
retraction count + straightness-in-cut as the two primary roughing objectives
(layer-based roughing, CAD 2025; VDAC SIGGRAPH Asia 2020 decomposes into a
minimum number of continuously carvable volumes, one continuous path each).
- Replace 1 spline (K=40) with M sub-splines (e.g. M=5 × K=10, equal DOF).
  The swept union is already min-over-all-segments — kernel unchanged; just
  skip the connector segment between sub-splines in the argmin.
- Pin each sub-spline's end control points at z ≥ stock_top + clearance:
  retracts are FREE (cut nothing, no transition modeling); the executed
  program joins sub-splines with safe-z rapids, which the evaluator treats
  identically (rapids above stock don't cut).
- DOGE-style (arXiv 2511.19850) discrete outer loop: every ~500 iters, delete
  a sub-spline whose samples are >90% air and respawn it seeded over the
  largest connected residual blob (gradients cannot change topology; the
  literature alternates continuous refinement with discrete split/merge).
- Init: cut the existing boustrophedon LSQ fit at descending-z layer
  boundaries into M chunks + safe-z lead-in/out per chunk.
- Expect: pyramid +0.02–0.05, box +0.01–0.02, air-cut fraction (now ~13%) down.
- Sources: VDAC https://dl.acm.org/doi/10.1145/3414685.3417772 · layer-based
  roughing https://www.sciencedirect.com/science/article/abs/pii/S0010448525001861
  · DOGE https://arxiv.org/abs/2511.19850 · DiffVG (stroke-set gradient descent
  works) https://people.csail.mit.edu/tzumao/diffvg/

### 2. Multi-start with pattern-diverse inits + successive halving  [effort LOW, payoff HIGH]
Gradient descent cannot leave the init's path topology (LaValle §14.7; ICRA
2020 topological replanning runs one local optimization per distinct guide
path and keeps the best).
- N=6 shape-agnostic inits (zigzag-X, zigzag-Y, inward/outward spiral,
  different z-layer orderings — all bbox-derived), successive halving:
  6 × 90 s → top 3 × 2 min → 1 for the remainder. Orchestration only; a
  125³ f32 grid is ~8 MB so batching is trivial.
- Optional upgrade only if starts collapse to one basin: SVGD repulsion
  between the parallel starts (CSVTO, T-RO 2024) — skeptical it pays within
  15 min otherwise.
- Sources: https://arxiv.org/abs/1912.12644 · https://lavalle.pl/planning/node795.html
  · https://arxiv.org/abs/2308.12110

### 3. Slope-adaptive z-step init + |N_z|-weighted residual loss  [effort LOW, payoff MED-HIGH on pyramid]
Staircase deviation on an inclined face: C = t·|cos θ| (Dolenc & Mäkelä), θ =
angle(surface normal, vertical), t = stepdown. The pyramid's 45° faces terrace
at C = 0.707·t — this is its dice deficit; box/cylinder walls have |N_z| ≈ 0
and are untouched (clean shape-agnostic differentiation).
- (a) Init: adaptive stepdown t(z) ∝ C_max / max_shell|∇SDF·ẑ| (clamped),
  computed from the target SDF gradient in the near-surface band per z.
  Rebalance total path length against the feed cap by coarsening
  vertical-wall bands (the steep/shallow trade from PowerMill/Fusion).
- (b) Loss: weight residual² by 1 + λ·|N_z| for shell-band voxels so terrace
  ridges on inclined faces are expensive. Ablate (a) and (b) separately.
- Sources: Dolenc formula restated in
  https://tams.informatik.uni-hamburg.de/publications/2017/Adaptive%20Slicing%20for%20the%20FDM%20Process%20Revisited.pdf
  · steep/shallow https://www.autodesk.com/products/fusion-360/blog/why-steep-and-shallow/
  · constant-scallop 3-axis 2024 https://www.sciencedirect.com/science/article/pii/S266684592400028X

### 4. Curriculum loss: annealed sigmoid band + hard-voxel reweighting  [effort LOW, payoff MED]
Curriculum DeepSDF (ECCV 2020) reports 32% Chamfer reduction from schedule
alone: tolerance ε annealed to 0 + hard/wrong-sign samples upweighted (1+λ).
- (a) Anneal our sigmoid band width ~3 voxels → 1 over the run (coarse walls
  first, precision later).
- (b) Hard-example memory: EMA of per-voxel loss; long-uncut residual voxels
  get weight (1+λ). Pure torch loss code.
- Related two-phase variant from CAM practice (roughing → finishing): compute
  residual/gouge against a δ-dilated target early (δ ≈ 1–2 voxels via SDF
  threshold), anneal δ → 0 while upweighting gouge; optionally add a
  finishing-band attraction pulling samples toward |SDF − r_tool| ≈ 0 (the
  flank-kissing offset surface).
- Sources: https://arxiv.org/abs/2003.08593 · roughing/finishing split per
  https://www.sciencedirect.com/science/article/abs/pii/S0010448525001861

### 5. Path-side Chamfer attraction to residual clusters (CNC-Net "center loss")  [effort LOW, payoff MED]
Our voxel-side attraction only pulls the argmin-winning segment; path samples
that are nobody's argmin get zero gradient. CNC-Net (CVPR 2024) adds the dual:
pull the nearest path sample toward remaining-material centroids.
- For each residual cluster centroid c: add min_i ||X_i − c||² (torch-side,
  negligible cost). The feed-cap penalty holds the yanked segment executable.
- Sources: https://arxiv.org/abs/2312.09925 (their §loss; system also
  validates the whole approach — 3-axis cylindrical tools, max(stock,−tool)
  carve at 64³, ~30 min/shape vs our ~4 min at 125³)

### 6. Residual-attributed knot insertion (Boehm, exact)  [effort MED, payoff MED]
Standard adaptive-spline practice (IGA/THB refinement, ELSPIA): optimize
coarse → insert knots where an error indicator is large → continue. Insertion
is EXACT (curve unchanged) — a free warm restart with local DOF added.
- The per-voxel argmin already attributes every residual voxel to a segment →
  knot span. 2–3 times per run, insert the midpoint knot of the worst span
  (K 40 → ~52), map P through the insertion matrix, re-project/reset Adam
  moments locally.
- Note: plain K=64-from-scratch was already tried and tied (capacity isn't
  the bottleneck globally) — this differs by adding DOF *where residual is
  stuck*, late, without re-optimizing everything. Targets pyramid stair-steps.
- Sources: https://academic.oup.com/jcde/article/2/4/218/5715267 ·
  https://arxiv.org/pdf/2512.17666

### 7. Contour-parallel / Fermat-spiral init from the SDF removal region  [effort MED-HIGH, payoff MED-HIGH]
Replaces the shape-blind bbox raster: per z-layer, removal region = stock ∧
(2D distance transform ≥ r_tool offset); order iso-distance rings outside-in,
connect Fermat-style (entry ≈ exit), LSQ-fit. Path exists only where material
must be removed → air-cut ~13% → <5%; the wall-hugging final ring per layer is
a built-in semi-finishing pass. scipy.ndimage.distance_transform_edt — no new
packages. Best combined with #1 (one spiral sub-spline per layer).
- Sources: Connected Fermat Spirals https://dl.acm.org/doi/10.1145/2897824.2925958
  · VDAC used CFS for physical 3-axis carving.

### 8. Top-2 annealed soft-min in the gradient pass  [effort MED, payoff LOW-MED]
The hard argmin subgradient is already exact; gains are landscape-smoothing
only (argmin flips between competing passes = gradient chatter, mostly in the
pyramid wall regime). Record top-2 segments in pass 1; differentiate
softmin_α(d₁, d₂) with dimensionless α·l₀ annealed ~8 → 256 (TreeTOp-style
scaling; CAPRI-Net-style stage switch also fine). Soft-min UNDERESTIMATES the
true min by ≤ log(N)/α (Madan & Levin) → dilates the sweep → over-carve bias;
top-2 keeps the bias at log(2)/α and the anneal kills it. Never ship
un-annealed.
- Sources: https://arxiv.org/abs/2108.10480 · https://arxiv.org/pdf/2104.05652
  · https://arxiv.org/html/2409.02300v1

### 9. Sample-budget reclamation: arc-length re-fit + air-run trimming  [effort LOW-MED, payoff LOW-MED]
Every ~200 iters: re-fit K control points by LSQ under chord/arc-length
parameterization (uniform-in-space samples → feed penalty uniformly
informative), and contract maximal parameter runs whose samples have zero
engagement (Fusion "Reduce Air Cutting" analog). Trust-region guard: reject a
re-fit if loss jumps. Interacts with Adam moments — reset locally.
- Sources: survey https://arxiv.org/abs/2303.01368 ·
  https://help.autodesk.com/cloudhelp/ENU/Fusion-CAM/files/REDUCE-AIR-CUTTING-ADAPTIVE.htm

### 10. Parameter-space plateau smoothing (perturbed optimizers)  [effort LOW-MED, payoff LOW]
Replace ∇L(P) with ∇E_z[L(P+σz)] via 2–4 antithetic perturbed forwards, σ ≈ 1
voxel annealed to 0 — directly targets the diagnosed gradient-starvation
plateaus, but w_broad already fixed the worst of it; expect incremental.
- Sources: https://arxiv.org/abs/2211.17263 ·
  https://papers.nips.cc/paper/2020/file/6bb56208f672af0dd65451f869fedfd9-Paper.pdf

## Cheap diagnostics to run first
- **T=512 falsification for continuous-time sweeps**: already run — 0.8355 vs
  0.8411 (no gain). Confirms segment-chord discretization is NOT a bottleneck
  at 125³, so continuous-time argmin (SVSDF-style Newton on t*, GSIP) is a
  dead end here. Independently, TVCG 2025 (arXiv 2509.09325) shows
  min-over-segment-fields is the detail-preserving formulation — our
  architecture — and notes no one has treated its differentiability (novelty
  claim for a write-up).
- **Exact vertical-accessibility mask** (one column-max pass over the voxel
  grid): don't attract toward waste whose vertical clearance column intersects
  the part. Free, exact, shape-agnostic; no-op for convex-from-above targets
  but prevents wasted gradient on concave ones (and cleanly formalizes the
  sphere's below-equator ceiling).

## Explicitly rejected / deprioritized (and why)
- **Continuous-time t\* argmin (Sellán 2021 / SVSDF TOG 2024 machinery)**:
  sub-voxel chordal bias at T=256/125³ (sagitta ≈ c²κ/8 ≲ 0.25 voxel); T=512
  falsification showed no headroom. Sellán's method is also patent-pending
  (Adobe). Keep as citations, not code.
- **Sphere tuning of any kind**: 0.007 from the structural ceiling.
- **Full SVGD / homotopy-class enumeration**: over-engineered vs plain
  multi-start under a 15-min budget.
- **RL/neural CAM (DeepMill, meta-RL flank milling)**: their wins are
  accessibility-query speed and cycle time, not geometric accuracy; 3-axis
  vertical accessibility is exact and trivial for us (column max).
- **Un-annealed global soft-min**: systematic sweep dilation → gouge bias.
- **5-axis / tool tilt** (would break the sphere ceiling, e.g. reach the
  below-equator shadow): requires simulator + evaluator changes — out of
  scope for this task's rules, first candidate if the harness ever grows
  orientation DOF. Implicit neural process planning (arXiv 2511.17578) is the
  reference design: SIREN layer-field + path-field, differentiable collision
  loss sampled on the tool body.
