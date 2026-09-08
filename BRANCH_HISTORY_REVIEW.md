# Branch History Review

Phase 1 review for the direct free-form-feedback experiment. This review is
read-only with respect to prior research: no historical training job was
rerun. It was prepared on branch `ethan-direct-freeform-feedback` at
`6b7aa285643e8205d2da7f79afd3f6403b21e76a`, whose merge base with
`origin/autoresearch` is `41b7355f05a6315bf2af47ee478501e519ca9232`.

## Evidence policy

The conclusions below use commit graphs, merge bases, diffs, source code,
branch-local research notes, and committed `results.tsv` files. A number in a
finding/idea document or commit message is classified as a **recorded claim**
unless a committed row supports it. Even a committed result row is not a fresh
reproduction: raw run directories, `metrics.json`, logs, and trajectories are
not committed on these branches. The repository intentionally ignores
`runs/*`, task `results.tsv`, and local feedback stores. Therefore this audit
can verify code and recorded history, but cannot independently validate most
old numerical results.

The apparent “Ar agd/... (#N)” integrations on `autoresearch` are single-parent
squash-style commits, not Git merge commits. Retention is therefore determined
from the current source, not ancestry alone.

## Branch topology

Counts are `branch-only / autoresearch-only` commits from each merge base.

| Branch tip | Relationship to `origin/autoresearch` |
|---|---|
| `master` `baaebb0` | ancestor; autoresearch is 164 commits ahead |
| `sim-features` `fd4c601` | ancestor; autoresearch is 57 commits ahead |
| `sim-logic` `667565a` | ancestor; autoresearch is 195 commits ahead |
| `nt-cleanup` `71636df` | ancestor; autoresearch is 62 commits ahead |
| `cleanup` `3a3ba53` | ancestor; autoresearch is 79 commits ahead |
| `aditya` `56f763c` | ancestor; autoresearch is 82 commits ahead |
| `aditya-autoresearch` `fc1a109` | diverged at `16b6e44`; 3 / 36 |
| `jun28-decay-port` `9d75efc` | diverged at `8d5057b`; 27 / 46; squashed as `3b33dde` |
| `jun30-new-research` `b3447cf` | diverged at `f73c553`; 3 / 44; squashed as `ac065c7` |
| `jul1-uniform-toolpath` `43ec112` | diverged at `ac065c7`; 33 / 43; squashed as `7dc8008` |
| `jul3-hard-carve-gap` `7771763` | diverged at `2a11475`; 6 / 41; squashed as `6557ce1` |
| `jul4-hard-carve-gap` `efeca28` | diverged at `6557ce1`; 31 / 40; squashed as `cd5ce1d` |
| `jul4-toolholder-collision` `52defd8` | diverged at `3230c17`; 52 / 38; squashed as `e440ab4` |
| `jul5-anneal-gap` `bdfe03c` | diverged at `16b6e44`; 39 / 36; squashed as `c177189` |
| `jul6-step-detail` `f7f5957` | diverged at `16b6e44`; 22 / 36; not directly squashed into autoresearch |
| `jul6-traj-quality` `7bb0e70` | diverged at `35a8292`; 9 / 32; squashed as `f6945eb` |
| `jul7-balanced` `98d4f7a` | diverged at `e5cadc8`; 5 / 30; no named squash integration |
| `jul8-multidepth` `63a6b1d` | diverged at `e5cadc8`; 16 / 30; squashed as `2bb9849` |
| `jul10-gouge` `72f2aaa` | diverged at `2bb9849`; 7 / 29; squashed as `88b8289` |
| `jul13-phys-plausible` `39efff9` | diverged at `88b8289`; 24 / 28; not integrated into autoresearch |

Commit counts on the STEP/physical branches include merged side histories and
should not be read as a measure of implementation size.

## Foundational simulator and training lines

### `sim-logic` → `master`

- **Question/failure mode:** establish a differentiable voxel-removal simulator,
  environment, and PPO training path; then improve collision handling, rewards,
  randomization, and cluster execution.
- **Changes:** discrete voxel simulation and RL environment/trainer, collision
  logic, later PPO/reward/domain-randomization and TACC setup changes.
- **Evidence:** source history and diffs are verified. Some early run/cache
  artifacts were committed on `sim-logic` and later removed, but they do not
  form a controlled research record.
- **Disposition:** retained as historical foundation; the current direct CSG
  optimizer is a different training path.
- **Support:** branch tips `667565a`, `baaebb0`; simulator/environment and early
  algorithm diffs.

### `aditya` → `cleanup` → `nt-cleanup` → `sim-features`

- **Question/failure mode:** move from discrete RL prototypes toward continuous
  differentiable CSG optimization and a standardized CAM pipeline.
- **Changes:** `aditya` introduced continuous CSG simulation, direct trajectory
  trainers, delta-position optimization, evaluation, and vectorized/voxel
  variants (`4eebab8`, `55e853e`). `cleanup` removed prototypes, renamed the
  delta trainer to `train_csg.py`, and added CAM/G-code parsing, export, planning,
  and round-trip tools. `nt-cleanup` standardized trainers/evaluation/video/W&B
  and split continuous/discrete environments. `sim-features` added holder and
  machine dimensions, real units, workspace/stock handling, and speed clipping.
- **Evidence:** verified directly in diffs and current descendants.
- **Disposition:** retained. Current `algorithms/train_csg.py` directly optimizes
  trajectory deltas with `torch.optim.Adam`, copies them into Taichi, and obtains
  gradients through a Taichi tape; it is not a policy-network trainer.
- **Support:** tips `56f763c`, `3a3ba53`, `71636df`, `fd4c601` and their diffs.

### `aditya-autoresearch`

- **Question/failure mode:** support imported STEP/SDF targets and non-default
  grids, including the Titan target.
- **Changes:** STEP-to-SDF conversion and grid-target plumbing; large generated
  files were subsequently removed.
- **Evidence:** code changes are verified; no branch-local committed raw run
  artifacts were found.
- **Disposition:** retained only through the related STEP/sweep line, not as a
  direct integration into `autoresearch`; current grid/target support overlaps
  with later work.
- **Support:** tip `fc1a109`, three commits after merge base `16b6e44`.

## Autoresearch experiment lines

### `jun28-decay-port`

- **Question/failure mode:** improve optimization stability and automate
  experiment bookkeeping.
- **Changes:** learning-rate decay, initialization scale, exposed objective
  weights, gradient clipping, unique run directories, structured initializers,
  best-checkpoint handling, and dashboard support.
- **Evidence:** report text claims 514 experiments and highlights `dt=0.45`,
  pyramid 0.9010 and sphere 0.8499. No task `results.tsv` or raw run artifacts
  are committed at this tip, and later history identifies these as soft-Dice or
  mixed-metric-era results. These numbers are documentary claims only.
- **Disposition:** infrastructure retained; numerical conclusions superseded by
  hard-carve and holder-safe evaluation.
- **Support:** branch tip `9d75efc`, `report.md`, `idea.md`; squash `3b33dde`.

### `jun30-new-research`

- **Question/failure mode:** improve experiment inspection rather than introduce
  a new optimizer hypothesis.
- **Changes:** visualization, instructions, and UI changes.
- **Evidence:** seven lines exist in the committed task `results.tsv`, but the
  branch has no distinct substantive optimization result or raw run artifacts.
- **Disposition:** tooling retained; no independent research claim to promote.
- **Support:** tip `b3447cf`; squash `ac065c7`.

### `jul1-uniform-toolpath`

- **Question/failure mode:** reduce uneven coverage and air travel and align
  optimization with hard carving.
- **Changes:** step, air, proximity, trajectory-proximity, and length terms;
  `raster_fine`; staged training/truncation; hard-carve alignment (`43ec112`).
- **Evidence:** 129 committed `results.tsv` lines. Notes report `lr=1e-3` and
  length regularization as useful, air-loss tradeoffs, and only marginal staged
  gains. Raw run artifacts are absent.
- **Disposition:** objective plumbing and hard evaluation retained; individual
  hyperparameter claims remain historical/not freshly verified.
- **Support:** tip `43ec112`, branch findings, task `results.tsv`; squash `7dc8008`.

### `jul3-hard-carve-gap`

- **Question/failure mode:** close the gap between smooth training loss and the
  discrete hard-carve evaluation.
- **Changes:** numerically stable log-sum-exp smooth maximum, configurable
  sharpness `k`, and annealing controls.
- **Evidence:** 19 committed result-table lines; notes report fixed `k=60–100`
  hard-Dice gains across 18 experiments. Annealing was staged but not established
  here. No raw artifacts are committed.
- **Disposition:** stable smoothing and `k` controls retained; exact performance
  claims are recorded, not reproduced.
- **Support:** tip `7771763`, findings/results; squash `6557ce1`.

### `jul4-hard-carve-gap`

- **Question/failure mode:** improve iteration-zero coverage with a shape-aware
  depth-layer initialization.
- **Changes:** target-derived `zlayer` coverage initialization.
- **Evidence:** 26 committed result-table lines and notes claiming very large
  initial hard-Dice gains. The next holder-safe branch showed that much of the
  gain came from physically unsafe deep plunges.
- **Disposition:** rejected as evidence of a valid physical improvement; the
  general structured-init idea continued in safer forms.
- **Support:** tip `efeca28`, findings/results; squash `cd5ce1d`; correction in
  `jul4-toolholder-collision` findings.

### `jul4-toolholder-collision`

- **Question/failure mode:** prevent the holder and finite tool reach from
  invalidating apparent carving gains.
- **Changes:** collision-safe floor/holder constraints and truncation correction.
- **Evidence:** 47 committed result-table lines. Notes explicitly revise prior
  baselines, report pyramid falling to about 0.526, and identify underoptimization
  and nondeterminism. No raw artifacts are committed.
- **Disposition:** retained and treated as a necessary validity correction;
  pre-holder results should not be compared directly to later results.
- **Support:** tip `52defd8`, branch findings/results, precursor `3230c17`, squash
  `e440ab4`.

### `jul5-anneal-gap`

- **Question/failure mode:** optimize softly early while ending close to the hard
  carve, under finite tool reach.
- **Changes:** shape-blind `k` annealing (10→150), 50 mm reachability, and
  `loss_shift=-0.5` experiments.
- **Evidence:** 107 committed result-table lines. Notes claim mean hard Dice
  0.877 over four primitives while acknowledging composite-hole difficulty.
  Later preference-driven tests found `loss_shift=0.5` negative and higher
  `k_final` largely neutral; these are different settings, not a clean refutation
  of all annealing.
- **Disposition:** annealing retained and is currently enabled by default;
  specific headline score remains unreplicated in this audit.
- **Support:** tip `bdfe03c`, findings/results; squash `c177189`; current
  `algorithms/train_csg.py` (`k_anneal=True`).

### `jul6-step-detail`

- **Question/failure mode:** handle detailed STEP parts where point-delta carving
  and coarse primitives are inadequate.
- **Changes:** STEP/NPZ targets, spline swept-volume simulation, grid support,
  KD-tree metric correction, feed-feasible raster/terrain initializers,
  reachability, argmin caching, and VRAM diagnostics.
- **Evidence:** notes claim rrph near 0.9753 and Titan near 0.8189, with poorer
  extrusion/bowl behavior. A later branch note reports a three-seed rrph delta
  baseline collapsing to uncut 0.687/100% air while sweep remains about 0.97.
  No task result table or raw run artifacts are committed at this tip, so these
  are documented claims with explicit single-seed/bug caveats.
- **Disposition:** uncertain and not integrated into the main autoresearch line;
  technically important support reappears in `jul13-phys-plausible`.
- **Support:** tip `f7f5957`, findings and `delta_vs_sweep_step.md`.

### `jul6-traj-quality`

- **Question/failure mode:** account for cycle time, air cutting, and breakage in
  addition to geometric overlap.
- **Changes:** differentiable and hard time/air/break diagnostics and losses;
  off-grid air-accounting fix; soft and hard metric reporting.
- **Evidence:** 38 committed result-table lines. Notes report a sphere air-time
  improvement with neutral Dice, but contradict themselves on whether the named
  `best_w_airtime` configuration was a no-op or a free win. Treat that conclusion
  as stale/internally inconsistent.
- **Disposition:** diagnostics and objective hooks retained; the branch-level
  hyperparameter recommendation remains uncertain. Current defaults include
  small nonzero air-time and break weights.
- **Support:** tip `7bb0e70`, findings/results; precursor commits `57b03a2` and
  `35a8292`; squash `f6945eb`.

### `jul7-balanced`

- **Question/failure mode:** stop hard-Dice-only checkpointing from selecting
  gouged or otherwise poor trajectories.
- **Changes:** hard-metric and gouge-aware best-checkpoint scoring.
- **Evidence:** only a one-line result-table header is committed. Notes report
  improved tradeoffs, high variance, and a one-run `k`-anneal candidate around
  0.674 that was not confirmed.
- **Disposition:** composite checkpoint concepts were retained and expanded;
  this branch's empirical claim is uncertain.
- **Support:** tip `98d4f7a`, `idea.md`, task result header; current checkpoint
  score and `best_on_hard` code in `algorithms/train_csg.py`.

### `jul8-multidepth`

- **Question/failure mode:** obtain better shape-agnostic waste coverage while
  fixing misleading air and checkpoint metrics.
- **Changes:** multidepth initialization, residual-weight sweeps, `k` annealing,
  hard-checkpoint selection, run subdirectories, and corrected air metrics based
  on pre-cut stock in both hard and differentiable paths.
- **Evidence:** the long `idea.md` records successive corrections: earlier air
  claims were invalidated by a metric bug; residual weight 5 was a provisional
  sweet spot; generalization was mixed; final hard checkpoints sometimes beat
  soft-selected ones; pyramid was characterized around 0.699±0.017 and hole
  around 0.243±0.006. No result table or raw artifacts are committed at the tip.
- **Disposition:** initializer family, corrected metrics, and hard selection
  retained; many early wave conclusions were explicitly superseded, and
  single-run gains remain uncertain.
- **Support:** tip `63a6b1d`, `idea.md`, branch code; squash `2bb9849`; current
  multidepth variants and `best_on_hard=True`.

### `jul10-gouge`

- **Question/failure mode:** reduce tool-envelope gouging and begin incorporating
  human trajectory preferences.
- **Changes:** tool-gouge objective with warmup, star-rating/free-text feedback,
  `--use-feedback` warm start, and `multidepth_cavity`.
- **Evidence:** notes claim sphere margin-1/tool-gouge-8 mean hard Dice about
  0.679 with less gouge, mixed generalization, and a cavity-hole Pareto result
  around 0.168 while regression guards were unfinished. No result table or raw
  artifacts are committed at the tip.
- **Disposition:** objective/UI plumbing and cavity initializer retained; broad
  empirical benefit remains uncertain.
- **Support:** tip `72f2aaa`, `idea.md`; squash `88b8289`; current
  `w_tool_gouge`, warmup, feedback, and cavity code.

### `jul13-phys-plausible`

- **Question/failure mode:** make STEP/sweep trajectories physically plausible,
  especially plunge force, fragility, feed rate, and executable output.
- **Changes:** combined STEP/swept-volume support with force/plunge/fragility
  diagnostics, ramped initialization, offline feed scheduling, and G-code export.
- **Evidence:** findings claim rrph 0.9723±0.0015 with fewer plunges and a roughly
  0.9% time cost. Titan was not run. The stated 11 result rows and raw runs are
  not committed at the branch tip.
- **Disposition:** promising but uncertain and not integrated into current
  `autoresearch`; it must not be presented as validated on Titan.
- **Support:** tip `39efff9`, branch findings and code.

## Current preference-learning line on `autoresearch`

After squash `88b8289`, commits `e3b2f73` onward add pairwise comparison
infrastructure, preference digests, experiment enqueueing, and preference-driven
manual research notes. This is not yet the direct FPL loop requested by the
handoff:

- `algorithms/train_csg.py` explicitly labels the pairwise digest a “steering
  signal only, not in loss.” An agent/human must manually translate the digest
  into a hypothesis and code/config change.
- The pair schema stores run IDs, prompt, dimension/magnitudes/scenario,
  timestamps, answer, and note, but not randomized display order, first-view
  time, exact displayed explanations/metrics, interpretation, objective diff,
  code revision, or child-run lineage.
- The UI keeps fixed A/B order. The digest counts A/B/tie by dimension but does
  not aggregate the selected magnitude, so an “A” count has no stable semantic
  direction across differently ordered pairs.
- The web server and enqueuer each use process-local locks around separate
  read-modify-write operations. Atomic replacement protects file integrity but
  does not prevent a cross-process lost update.
- `pref_objective_plan.md` records 20/29 answered comparisons as of its stated
  2026-07-15 snapshot and three preference-motivated checks rejected by hard
  Dice: `k_final=180` neutral/marginal, `loss_shift=.5` negative, and
  `multidepth_revs=4.5` neutral/negative. Calling finish/contour preferences
  “cosmetic” overstates this evidence: hard Dice does not directly measure
  surface finish, scallop, or overlap quality, and no direct finish objective is
  present in the current loss.

Current pairwise and run-feedback stores are intentionally Git-ignored. Their
contents are experimental data and were not needed for this history audit.

## Corrections to inherited architectural claims

1. The current primary CSG trainer is direct trajectory-parameter optimization,
   not PPO and not a learned policy.
2. Training uses smooth carving while evaluation/checkpoint metrics can use
   `forward_hard`; “soft Dice” and “hard Dice” are not interchangeable across
   old branches.
3. Holder/reach constraints materially changed the benchmark. Pre-holder
   z-layer scores are not physically comparable to holder-safe scores.
4. `k_anneal` and `best_on_hard` are currently defaults, despite older notes
   describing them as optional experiments.
5. Air/time/break and gouge terms exist, but no direct surface-finish,
   scallop-height, or pass-overlap loss was found.
6. Existing pairwise feedback is read and reported, but it does not directly
   alter the loss. The requested critique→agent interpretation→objective
   change→child runs transition therefore requires new traceable orchestration.
7. Branch prose is not raw evidence. No reviewed research tip commits complete
   run directories with metrics/logs/trajectories; several tips omit even the
   result table, and some findings explicitly correct earlier findings.

## Phase 1 conclusion

The strongest retained technical line is: direct delta optimization → stable
smooth/hard evaluation → holder-safe constraints → annealed sharpness →
trajectory-quality diagnostics → multidepth/cavity/hole initializers →
gouge-aware selection and pairwise UI. The empirical record is useful for
hypothesis generation but is uneven and often prose-only. Phase 2 should use
the unchanged `41b7355` baseline first, record fresh raw artifacts and exact
provenance, and avoid using historical headline scores as acceptance thresholds.
