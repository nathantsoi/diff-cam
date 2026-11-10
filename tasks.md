# DiffCAM implementation tasks

Status: proposed work; no implementation task is marked complete by this document.  
Authoritative task registry: this file.  
Agent assignments and scheduling: [detailed_plan.md](detailed_plan.md).  
Scientific rationale and sources: [experimental plan](reports/experimental_plan.md).  
Target: `core`, integrating isolated `task/<id>-<description>` branches.

## Objective and scope

Establish whether differentiable stock removal improves a competent CAM seed faster than strong derivative-free search, with gains preserved under independent hard evaluation of the posted program. First deliver one complete pocket experiment; expand only after correctness and integration gates pass. Policy learning is a separate extension.

A research hypothesis may fail. Completion means correct implementation, auditable experiments, and a recorded conclusion, not a preferred ranking, universal improvement, or a predetermined curve shape. `DisagreementContinuation` is an experimental candidate, not an established contribution.

## Execution rules

1. Run Python through `uv run` in the assigned repository/worktree. For restricted caches use `TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-<agent>` and `UV_CACHE_DIR=/tmp/diffcam-uv-<agent>`. Cache variables do not install dependencies or solve offline package availability. Record an unavailable backend/dependency explicitly.
2. Keep Taichi lifecycle explicit. `ti.init()` invalidates existing fields; initialize once per process, pass `init_taichi=False` to additional simulators, or use independent processes. Use distinct outputs and cache directories for concurrent agents.
3. Follow task dependencies rather than a global stage barrier. Independent work may proceed while an optional tool is unavailable. Integration requires real dependencies to pass; mocks are allowed for unit development, never as evidence of end-to-end correctness.
4. Tune on development tasks only. Freeze configurations, evaluator revision, task hashes, metrics and budgets before test runs. Hard feedback on an individual test part is allowed only as a predeclared solver operation, equally budgeted; it must not retune the method or task definitions across test cases.
5. Record pass, fail, blocked, not-applicable, and incomplete coverage distinctly. Missing tools or unmodeled motion never become a successful check. Tests must check semantics, not merely file existence.
6. Each task delivers changed-file list, dependency revisions, exact validation commands/results, known limitations, and a runnable reproduction. Only the coordinator updates task status or shared dependencies. See [detailed plan](detailed_plan.md) for ownership.
7. Dataset generation, downloads and benchmark execution are separate commands. No large download or sweep as a unit-test side effect. Physical packets are artifacts for a qualified machining setup; producing them does not authorize operating equipment.
8. The gate commands below are implementation targets. Their modules/tests generally do not exist yet. Do not claim a gate passed before implementing and running it.

## Shared contracts (owned by T0.0)

### Task and motion

`TaskSpec` must specify schema version, task ID/family/split, source/license/checksums, stock/target geometry and units, grid bounds/spacing/sign convention, work-to-machine transforms, initial machine state, tool cutting/non-cutting/holder geometry, fixtures, supported motion semantics, machine limits, and evaluation tolerances. Separate the original part from any reachable operation target, with excluded regions/volume and a reason established before optimization.

`MotionPlan` retains segment types (rapid/feed/arc), physical positions, feed rates, timing assumptions, coordinate frame, and initial state. A numerical trajectory array is an export of this contract, not its complete substitute. Define a single training execution contract: fixed segment timing or documented time parameterization with constraints. All compared solvers use it. Zero residual must reproduce the nominal path through execution and posting within declared tolerances.

`EvaluationResult` includes coverage status, evaluator revision, task hash, NC hash, tool/setup hash, discretization settings, metrics with units/denominators, and reasons for null results. Define:

- `safety_pass`: modeled safety constraints pass with sufficient declared coverage; otherwise false or unknown.
- `geometry_pass`: completion, global gouge and local dimensional tolerances pass.
- `efficiency_eligible`: complete coverage and both passes are true.

A no-cut program must fail completion on a removal task. Collision counts, unique contacted volume, and time-integrated overlap are different quantities. Name them separately and test their sampling behavior. Force-like quantities remain labeled proxies unless calibrated; do not rank or physically qualify a program using an unvalidated force estimate.

### Run artifacts

Each completed solver run writes `config.json`, `history.json`, `metrics.json`, `initial_trajectory.npy`, `trajectory.npy`, `trajectory_deltas.npy`, and `motion_plan.json`, plus `rs274.nc`, `haas.nc`, and separate `eval_posted_rs274.json` / `eval_posted_haas.json`. `eval_posted.json` indexes the two files by hash; it must not silently mix their metrics. A dialect that cannot represent the motion is explicitly unsupported, with no fabricated NC/result. Failed/incomplete runs write a status and whatever artifacts were produced.

Config includes source commit/dirty patch hash, dependencies, seed, task/split, parameterization, objective/normalization, smoothing/checkpoint policy, machine/tool setup, and resource budgets. History counts candidate evaluations, forward/backward calls, validation calls, segment/voxel work, JIT/warm-up, end-to-end and steady-state wall time, and peak memory. Do not assume a backward pass universally costs one forward pass.

## Foundation

### T0.0 — Freeze interfaces and artifact schemas

- **Depends on:** none.
- **Owns:** `benchmarks/contracts.py`, `docs/contracts.md`, `tests/contracts_test.py`, `tests/data/contracts/`.
- **Work:** define the contracts above, checked serialization, validation and hashes; include a tiny valid pocket specification, invalid units/frame cases, incomplete coverage and failed-run examples. Specify interfaces for evaluators, parameterizations and candidate results before downstream implementations.
- **Gate:** `uv run python -m pytest tests/contracts_test.py -q`; round trips preserve meaning/hashes, invalid/inconsistent inputs fail with actionable errors.

### T0.1 — Selectively acquire and pin references

- **Depends on:** none.
- **Owns:** `scripts/pull_references.sh`, `configs/references.json`; coordinator owns `.gitignore` changes.
- **Work:** support explicit subsets and offline verification. Ignore clones under `external/` and bulk assets under `data/external/`, but track the manifest under `configs/`. Record repository URL, pinned SHA/revision, license evidence (including missing/unclear), artifact checksums, retrieval date, and reproduction status. Do not fetch moving HEAD on repeated runs or overwrite dirty clones.
- **Initial subset:** pycma, CNC-Net, and CAMotics. Later acquire DeepMS-DataGen/DeepMS, AAGNet dataset resources, OpenCAMLib, FusionCut, CNCGEN preview, and SHAC only for the tasks that use them. Source links are in the experimental plan. Verify actual dataset payloads rather than LFS pointer files; CNCGEN preview is not the full benchmark.
- **Gate:** subset acquisition/verification command reports every requested item and hash; missing network/license/payload is explicit. No arbitrary `len(manifest)>=8` gate. An unavailable optional reference does not block local geometry work.

### T0.2 — Reproducible core environment

- **Depends on:** none; independent of T0.1.
- **Owns:** `pyproject.toml`, `uv.lock`, `.gitignore`, `docs/environment.md` (coordinator/integrator).
- **Work:** lock core dependencies; isolate optimizer, CAD, independent simulator, and policy extras/environments. Do not force CNC-Net's dependency stack into the Taichi environment. Record CPU/GPU availability; test actual backends separately. Use unique smoke-output directories.
- **Gate:** locked core sync and `uv run python -m pytest -q`, plus the existing optimizer smoke on CPU. GPU validation is separately required before GPU claims; unavailable hardware is not a pass. Record a supported offline route when possible without presenting it as a fresh locked install.

## Geometry, motion semantics, and evaluation

### T1.0 — Independent geometry/chronology reference and fixtures

- **Depends on:** T0.0, T0.2.
- **Owns:** `verification/reference_carve.py`, `verification/fixtures.py`, `tests/reference_geometry_test.py`.
- **Work:** implement a slow reference independent of production sweep helpers. Use analytic vertical/horizontal occupancy and controlled dense temporal sampling for general movement; establish convergence and uncertainty. Specify stock intersection when computing expected removed volume (a plunge formula is only valid for its chosen initial geometry). Add sloped ramps, near-vertical moves, reversal, repeated cuts, through-holes, legitimate cutter-leading-holder motion and genuine holder/rapid collisions.
- **Gate:** `uv run python -m pytest tests/reference_geometry_test.py -q`; known cases have independently derived expectations; refinement bounds error. No shared production geometry imports. No universal percentage tolerance independent of tool/grid scale.

### T1.1 — Correct production cutter sweeps

- **Depends on:** T1.0.
- **Owns:** cutter methods in `simulator/csg_simulator.py`, `tests/sweep_geometry_test.py`.
- **Work:** define hard swept membership and a separate differentiable surrogate. Validate pure Z, horizontal and simultaneous XYZ motion. A Z-admissible interval projection may support membership but must not be labeled an exact Euclidean SDF without proof. Handle empty intervals, degenerate segments and endpoint cases. Compare any sampling/minimization approximation with the independent reference.
- **Gate:** `uv run python -m pytest tests/sweep_geometry_test.py tests/core_test.py -q`; hard occupancy/subdivision error converges with resolution; finite differences of the same smooth objective agree across appropriate epsilon values away from gates. Document nondifferentiable boundaries and absolute/relative tolerances for float32.

### T1.2 — Holder chronology and conservative training penalty

- **Depends on:** T1.1.
- **Owns:** holder penalty/diagnostic methods in `simulator/csg_simulator.py`, `tests/holder_chronology_test.py`.
- **Work:** use pre-segment stock for conservative training checks and consistently update diagnostics. Label this a conservative whole-segment screen, not exact chronology. Validate time-resolved evaluation against both real collisions and valid cutter-leading-holder paths; quantify conservative false positives versus temporal refinement. Confirm a small negative-gradient update reduces penetration on a controlled differentiable fixture; do not assert an upward-positive derivative.
- **Gate:** `uv run python -m pytest tests/holder_chronology_test.py -q`; independent fixture verdicts and descent-direction test pass. Stock unions are unnecessary when subtraction is monotone. Physical clearance units and modeled assembly extent are explicit.

### T1.4 — NC interpretation and motion-preserving posting

- **Depends on:** T0.0, T0.2.
- **Owns:** `cam/config.py`, `cam/gcode_parser.py`, `cam/posts.py`, `cam/gcode_export.py`, `cam/trajectory_planner.py`, `tests/nc_semantics_test.py`.
- **Work:** support the declared RS274/Haas subset, initial machine state, units, work offsets, `G53`, tool-length assumptions, modal motion and arcs. Retain rapid/feed/plunge semantics during export. Explicitly reject or report incomplete coverage for unsupported geometry-affecting commands. Unknown initial state must not silently omit an approach while asserting complete coverage. Declare timing model support, including exact stop versus blending.
- **Gate:** `uv run python -m pytest tests/nc_semantics_test.py tests/cam_test.py -q`; independently specified expected machine positions cover offsets, initial approach, home retract and units; parser/post agreement alone is insufficient evidence.

### T1.3 — Hard posted-path evaluator and external cross-check

- **Depends on:** T0.0, T1.0, T1.1, T1.2, T1.4.
- **Owns:** `cam/posted_evaluator.py`, `verification/camotics_adapter.py`, `simulator/csg_metrics.py`, `tests/posted_oracle_test.py`.
- **Work:** execute parsed typed motion chronologically using bounded spatial/time refinement. Rapid moves check cutter and assembly clearance and never carve. Feed moves remove material with ordered holder checking. Quantify temporal/spatial uncertainty. Production hard evaluation can reuse production geometry but must be named accordingly; independent reference fixture checks are mandatory. Cross-check at least one supported posted fixture through a separately implemented system such as CAMotics and retain provenance/coverage.
- **Metrics:** IoU/Dice, gouge/residual volumes and denominators, local/max/p99 surface errors, explicitly defined ASD/HD95, clearance, unique collision volume, timing and air fraction. Define empty-surface behavior. Keep proxy forces separate and nullable. Use streaming stock state for evaluation where possible rather than allocating full differentiable history.
- **Gate:** `uv run python -m pytest tests/posted_oracle_test.py -q` plus recorded external-fixture cross-check. Tests include no-cut rejection, rapid crossing solid, offset/retract semantics, both collision fixtures, hash mismatch and unsupported command coverage. Unavailable external software blocks this task's full integration gate, not independent foundation work.

## One complete task before scale-up

### T2.0 — Custom stock/target integration

- **Depends on:** T0.0, T1.2.
- **Owns:** target/stock initialization and queries in `simulator/csg_simulator.py`, `tests/custom_geometry_test.py`.
- **Work:** import sampled signed fields with explicit bounds/spacing. Route all target consumers through the chosen target representation, including `target_sdf`, `target_sdf_scalar`, `compute_loss`, metrics, volume and position penalties. Define interpolation and out-of-grid behavior, and correct mm/voxel scaling. Preserve analytic-target regressions.
- **Gate:** `uv run python -m pytest tests/custom_geometry_test.py tests/core_test.py -q`; analytic versus sampled fixture agreement converges; changing a custom grid changes both terminal loss and position penalties; gradients verified within interpolation cells.

### T2.1 — Shared trajectory parameterization

- **Depends on:** T0.0, T0.2.
- **Owns:** `algorithms/parameterization.py`, `tests/parameterization_test.py`.
- **Work:** implement delta residuals, spline basis and optional absolute-position residuals with a fixed initial position and explicit endpoint rules. For delta splines, `Delta = Delta_CAM + B theta` and pullback `B.T @ grad_Delta`; for absolute mode, include the exact differencing operator and its transpose. Restrict valid K/T combinations; use the same scale/bounds for every solver. Keep numerical maps distinct from physical execution.
- **Gate:** `uv run python -m pytest tests/parameterization_test.py -q`; zero residual, shapes, bounds and analytical pullbacks tested in float64 independently of simulator noise. Integration gradient checks use documented float32 tolerances later.

### T2.2 — One pocket task and nominal CAM execution

- **Depends on:** T2.0, T2.1, T1.3.
- **Owns:** `cam/seeds.py`, `benchmarks/pilot_task.py`, `tests/task_suite_test.py`.
- **Work:** produce one reachable pocket task with an island and competent nominal CAM seed, preserving motion kinds/feed/timing. Record whether seed is FreeCAD-generated or an in-repo heuristic. Export both posts and establish zero-residual equivalence through parameterization, training execution, posting and evaluation. Fix resampling/timing mismatch rather than accepting silent speed clipping. Do not call a circular-loop heuristic constant-engagement adaptive CAM without validation.
- **Gate:** `uv run python -m pytest tests/task_suite_test.py -q`; nominal task passes declared completion/safety/tolerance checks and zero-residual equivalence for supported dialects. Retain artifacts and independent cross-check.

### T2.3 — Expand and freeze task families

- **Depends on:** T2.2, T4.1.
- **Owns:** `benchmarks/procedural_suite.py`, `scripts/generate_task_suite.py`, `configs/task_manifest.json`, `tests/procedural_suite_test.py`.
- **Work:** facing on asymmetric stock; pocket/island contouring; variable-width/depth clearing. Start with three dev and ten held-out test geometries per family if pilot compute permits. Split by topology/design family, track generated hashes and accessibility definitions. Preserve original and reachable operation targets; report excluded volume. Validate finite-radius cutter, shank and holder accessibility, not only vertical visibility. Every task has a qualified nominal seed or a predeclared reason for exclusion.
- **Gate:** `uv run python -m pytest tests/procedural_suite_test.py -q`; deterministic regeneration, disjoint splits/provenance, task validity and seed verification. No performance-based exclusion and no tuning from test outcomes.

## Solvers and first comparison

### T3.0 — Stateless candidate evaluation and compute accounting

- **Depends on:** T2.0, T2.1, T1.2.
- **Owns:** `algorithms/objectives.py`, `algorithms/run_artifacts.py`, `tests/objectives_test.py`.
- **Work:** expose shared hard/smooth candidate evaluation per T0.0. Reset stock, loss and all accumulators before each candidate; save diagnostics before subsequent passes overwrite fields. Define matched hard counterparts and weighting/normalization. Distinguish fixed-path surrogate disagreement from differing executions under stock-dependent speed gates. Initially use geometry plus modeled safety; add time/force-related terms only with validated semantics.
- **Gate:** `uv run python -m pytest tests/objectives_test.py -q`; A/B/A evaluation-order invariance, tape/no-tape objective equality, gradients, seed determinism and explicit compute ledger. No solver-specific objective advantage.

### T3.1 — Hard/smooth derivative-free solvers

- **Depends on:** T3.0 for development; T2.2 and T1.3 for real-task/full-artifact acceptance; pycma subset from T0.1 and declared environment extra.
- **Owns:** `algorithms/blackbox.py`, `tests/blackbox_test.py`.
- **Work:** first implement CMA-ES on identical parameters for hard and smooth objectives; add CEM and separable CMA only when dimensionality/pilot warrants them. Handle invalid candidates consistently. Count every population member and validation call; retain allowed batching and actual timing. Match checkpoint opportunities across solvers.
- **Gate:** `uv run python -m pytest tests/blackbox_test.py -q`; known synthetic optima, budget/termination behavior, repeatability and small task integration. No requirement to outperform Adam.

### T3.2 — Fixed and annealed Adam reference methods

- **Depends on:** T3.0 for development; T2.2 and T1.3 for real-task/full-artifact acceptance.
- **Owns:** `algorithms/optimize.py`, `tests/optimizer_test.py`.
- **Work:** adapt the minimal runner to shared contracts/objectives/parameterization. Implement fixed-k and predefined linear annealing, hard selection, reusable checkpoints and complete failure records. Preserve parameter/moment/config consistency. Export through typed motion rather than raw positions alone.
- **Gate:** `uv run python -m pytest tests/optimizer_test.py -q`; gradient pullback, budgets, checkpoint replay, artifacts and small task run pass. Selection follows explicit safety/completion priorities, not hard Dice alone.

### T4.0 — Early run auditor and statistical aggregator

- **Depends on:** T0.0, T0.2.
- **Owns:** `scripts/aggregate_eval_results.py`, `tests/eval_aggregator_test.py`.
- **Work:** validate schema, task/NC hashes, dialect identity, coverage and failures from synthetic records first. Aggregate paired task-level differences with seeds nested within geometry; use task-level or hierarchical uncertainty, family stratification and predeclared comparisons. Do not treat seeds on one part as independent parts. Apply Clopper–Pearson only to appropriately independent binary units; otherwise use cluster-aware reporting. Keep safety, completion and efficiency eligibility distinct.
- **Gate:** `uv run python -m pytest tests/eval_aggregator_test.py -q`; known calculations, missing/corrupt artifacts, clustered seeds, no-cut outcomes, and incomplete coverage tested. Produce a pilot table before the first sweep.

### T4.1 — One-task end-to-end pilot

- **Depends on:** T2.2, T3.1, T3.2, T4.0.
- **Owns:** `scripts/run_benchmark_suite.py`, `configs/pilot.json`, `tests/benchmark_runner_test.py`.
- **Work:** nominal CAM, fixed Adam, annealed Adam, hard CMA-ES and smooth CMA-ES on one pocket at one K, with a small declared seed/budget set. Support `--split {dev,test,external}`, `--experiment {pilot,initialization,solvers,mechanics,all}`, `--dry-run`, frozen config and explicit output directory. Serialize GPU jobs unless measured capacity supports concurrency. Implement resumability and record failed/interrupted runs.
- **Gate:** `uv run python -m pytest tests/benchmark_runner_test.py -q` and a real `--split dev --experiment pilot` run audited by T4.0. Gate requires complete data and a bottleneck/decision report, not a preferred method winning.

## Experimental candidate and mechanisms (after the pilot)

### T3.4 — Disagreement-aware continuation hypothesis

- **Depends on:** T4.1.
- **Owns:** `algorithms/continuation.py`, `tests/continuation_test.py`; optimizer integration is serialized with T3.2's owner.
- **Work:** first isolate signed loss calibration from backtracking/annealing. On an identical executed path and grid, let `delta = median(phi_hard - phi_soft)` over a physically specified target band. Because the current occupancy loss uses `phi_soft + loss_shift`, use bounded signed EMA `shift <- (1-eta)*shift + eta*clip(delta,-dmax,dmax)`. Over-erosion should yield a negative correction. This is an empirical hypothesis: a scalar cannot necessarily correct spatially varying bias.
- **Acceptance:** compare incumbent/candidate under the same k/shift and hard validation state. Enforce gouge/local-error/safety constraints explicitly. Distinguish feasible from infeasible initialization; define restoration/least-violation behavior. Bound retry counts, sharpness and learning rate. Retain accepted theta, moments and config consistently. Use bounded disagreement measures; never divide by a raw zero/signed Dice gap. Call the mechanism backtracking unless a genuine trust-region radius/model is implemented.
- **Gate:** `uv run python -m pytest tests/continuation_test.py -q`; sign, no-drift fixture, rollback, infeasible start, zero gap, changed-k comparison and budget behavior pass. Dev ablations may conclude that the method should be dropped.

### T3.3 — Gradient mechanics and segmentation study

- **Depends on:** T4.1; add T3.4 only for continuation-specific comparisons.
- **Owns:** `scripts/run_gradient_mechanics.py`, `tests/gradient_mechanics_test.py`.
- **Work:** verify AD using the same smooth objective. Define randomized hard objective `J_sigma(theta)=E[L_hard(theta+sigma*z)]`, with `z ~ N(0,I)` in declared normalized parameter coordinates and sigma in those coordinates, and antithetic estimator `[L(theta+sigma*z)-L(theta-sigma*z)]*z/(2*sigma)` averaged over pairs. Report actual 2N function evaluations and uncertainty of the high-sample reference. Compare gradient alignment/discrepancy rather than claiming unbiased equivalence to a different surrogate. Deterministic smooth gradients do not acquire estimator variance without a specified randomness source.
- **Experiments:** pilot selected k, grid and segmentation values at fixed geometric paths. Include zero-gradient cases and finite-difference epsilon sweeps. Expand only if informative; no prescribed U-curve.
- **Gate:** `uv run python -m pytest tests/gradient_mechanics_test.py -q`; known differentiable estimator fixture, accounting, uncertainty and fixed-path identity pass; retain all diagnostic outcomes.

### T3.5 — Optional horizon/memory study

- **Depends on:** T4.1.
- **Owns:** `algorithms/horizon.py`, `tests/horizon_test.py`; simulator/optimizer hooks require exclusive handoff.
- **Work:** choose and document a precise objective: detached intermediate losses, terminal surrogate/value, or checkpointed full terminal differentiation. Do not call detached windows equivalent to full-terminal optimization. Verify H=T equivalence for the chosen method, prefix sensitivity and recomputation after updates. If claiming memory savings, implement window-local fields or checkpoint/recomputation and measure peak allocation; saved NumPy states alone retain full stock storage and are not memory savings.
- **Gate:** `uv run python -m pytest tests/horizon_test.py -q` plus measured memory/time/objective diagnostics. Optional scope; it does not block the core comparison.

## Early external compatibility, then frozen evaluation

### T5.0 — External compatibility reconnaissance

- **Depends on:** T0.0; required subsets of T0.1.
- **Owns:** `reports/external_compatibility.md`, `configs/external_compatibility.json`.
- **Work:** inspect actual CNC-Net inputs/actions/checkpoints/tools/orientations and reproduction commands; do not assume 64-cubed inputs or directly exportable fixed-tool waypoints. Identify one compatible case or explicitly document incompatibility. Inspect one Titan setup and CAD dependency path. Verify CNCGEN preview payload and reference-path provenance. This task runs before broad sweeps; no test-performance tuning.
- **Gate:** evidence-backed compatibility matrix with revisions, licensing status, commands tried, native versus adapted outputs, required transformations and blockers. A documented incompatibility is a valid discovery outcome, not a completed performance comparison.

### T5.1 — CNC-Net adapter pilot

- **Depends on:** T5.0, T1.3, T2.0.
- **Owns:** `benchmarks/cncnet_adapter.py`, `tests/cncnet_adapter_test.py`.
- **Work:** preserve native reconstruction evaluation; add executable NC only for a justified compatible subset. Disclose all operation restrictions, added connecting moves and tool assumptions. Separate pretraining/checkpoint costs from per-instance refinement. If no faithful motion mapping exists, record an explicit scope decision before benchmark freeze rather than fabricate paths or silently omit the baseline.
- **Gate:** `uv run python -m pytest tests/cncnet_adapter_test.py -q` plus one real compatible reproduction with both metric routes, or an audited unavailable/incompatible disposition. Never mark absent executable results as passing.

### T5.2 — External/Titan task adapters and one practical pilot

- **Depends on:** T5.0, T2.2.
- **Owns:** `benchmarks/external_tasks.py`, `benchmarks/titan_setups.py`, `tests/external_tasks_test.py`.
- **Work:** first create one Titan single-tool operation and one feature task from a selected source. CAD geometry must pass through a verified STEP-capable backend; mesh tools alone are not a STEP importer. Preserve original targets, predeclared operation subtargets and excluded volume. Check radius/length/holder accessibility. Keep CNCGEN's 20-item preview descriptive; generated DeepMS sequences are not optimal NC ground truth. Publish source and split provenance without assuming redistribution rights.
- **Gate:** `uv run python -m pytest tests/external_tasks_test.py -q`, nominal CAM + posted independent pilot checks. Expand to additional Titan/feature instances only after this succeeds and before freezing external evaluation membership.

### T4.2 — Freeze the bounded protocol

- **Depends on:** T2.3, T4.1, T5.0, T5.1, T5.2; T3.4/T3.3/T3.5 only if those methods/claims are retained.
- **Owns:** `configs/frozen_benchmark.json`, `reports/protocol_decision.md`.
- **Work:** predeclare tasks, inclusion/exclusion, primary comparisons, tolerances, resource budgets, validation opportunities, split hashes, tuning budget and retained methods. Separate geometric and efficiency outcomes. Choose one primary K; add a small dimensionality subset and selected ablations rather than an automatic Cartesian product. Record runtime/VRAM estimates and run count before launch. Decide how unavailable baselines limit claims. Keep physical measurements separate from simulated/proxy metrics.
- **Gate:** runner `--dry-run` and schema/auditor validation of frozen config; reproducible run manifest and signed-off scientific decision record. Negative dev findings are allowed. No strict-improvement gate.

### T4.3 — Held-out execution and audit

- **Depends on:** T4.2.
- **Owns:** immutable run manifests/results and execution ledger; code changes only through reviewed fixes.
- **Work:** execute frozen procedural/external configurations, retaining failures and interrupted runs. Deterministic CAM is not rerun as fictitious independent seeds. Record hardware/JIT/end-to-end costs and evaluator coverage. A correctness fix after inspecting results requires a versioned protocol amendment and affected reruns, not selective replacement.
- **Gate:** `uv run python -m scripts.run_benchmark_suite --split test --experiment all --config configs/frozen_benchmark.json --output-dir <new-dir>` and corresponding declared external run; audit expected-versus-observed run IDs, hashes, failures and aggregate coverage. No success based solely on output files existing.

## Physical and policy extensions; publication

### T6.1 — Physical trial packets

- **Depends on:** T5.2, T4.2.
- **Owns:** `scripts/export_physical_packets.py`, `tests/physical_packet_test.py`.
- **Work:** package selected baseline/optimized programs with actual machine model, tooling/stickout, work offsets, fixture geometry, motion mode, material/process settings and source hashes. Require complete simulation coverage, local dimensional tolerances, documented clearance margins and independent checks. A global gouge fraction or zero occupied-voxel intersections alone is insufficient. Through-hole floor rules depend on the setup/part. Mark packet status as simulation-qualified or pending setup review, never universally collision-free.
- **Gate:** `uv run python -m pytest tests/physical_packet_test.py -q`; malformed setup/missing coverage/hash mismatch rejected, known valid packet reproduced. Equipment execution is a separate qualified-operator activity, not an automated agent gate.

### T6.2 — Physical measurement ingestion

- **Depends on:** T6.1 and actual trial records for empirical conclusions.
- **Owns:** `scripts/evaluate_physical_trials.py`, `tests/physical_measurement_test.py`.
- **Work:** ingest controller timing, tool/setup identity, spindle load and scans/CMM data with calibration/registration/uncertainty. Distinguish measured force from spindle-load and simulation proxies. Report matched physical comparisons and failures. Software can be tested before trials; physical validation remains pending until measured records exist.
- **Gate:** `uv run python -m pytest tests/physical_measurement_test.py -q`; empirical report additionally requires actual traceable data, not synthetic ingestion fixtures.

### T7.1 — Optional policy-learning track

- **Depends on:** T2.3, T3.0, T1.3 and a separate training split/gradient interface contract.
- **Owns:** `algorithms/residual_env.py`, `algorithms/train_policy.py`, `tests/residual_env_test.py`.
- **Work:** residual PPO/SAC and SHAC with matched observations/actions/CAM priors and separate train/dev/test design families. Specify local observation partial observability and any recurrent/global state consistently. Implement and verify the Taichi-to-policy gradient bridge and reset semantics. Add hard/smooth environment controls where needed to separate gradients from smoothing. Report training, inference and adaptation costs separately; AHAC only if its mechanism is central. This track does not gate the first trajectory paper.
- **Gate:** `uv run python -m pytest tests/residual_env_test.py -q`; real small training/evaluation with held-out task identity checks and gradient bridge verification before policy claims. Offline simulator work does not authorize physical exploration.

### T7.2 — Publication artifacts and final audit

- **Depends on:** T4.3, T4.0; T6.2/T7.1 only for physical/policy claims included.
- **Owns:** `scripts/build_paper_artifacts.py`, `manuscript/` integration, `reports/final_audit.md`.
- **Work:** build tables/figures from audited records, including no-cut and nominal CAM anchors, failures, uncertainty, soft/hard/posted disagreement and resource costs. State evaluator/model limitations and actual external baseline coverage. Preserve unfavorable findings. If physical data or policy evidence is absent, omit those empirical claims explicitly. Reproduction includes tracked manifests, configs and dataset acquisition instructions.
- **Gate:** clean artifact regeneration, trace every claimed result to run/task/NC hashes, validate family-level statistical units and exclusions. Confirm literature/novelty against primary sources before submission; no predetermined curve or algorithm ranking.

## User-directed RustCNC CAM starting-program replacement

This section was added on 2026-11-10 at the user's direction. It pauses the old-start campaigns; it does not erase their completed historical records. Detailed implementation state and immutable inputs live in `docs/rustcnc_strategy_port.md`; the source ports retain reference provenance rather than becoming clean-room/MIT code by copying.

**Handoff override:** later on 2026-11-10 the user requested winding down all implementation workers for a next-agent takeover. The cards below are **deferred, not complete**. No new work, campaign, or REAL_* validation is assigned by this section now. Begin with `docs/CAM-RUST-HANDOFF-20261110.md`, especially its directory map and the preserved worker reports named there.

### CAM-RUST-1 — Face and Contour 2D direct port

- **Depends on:** RustCNC source pinned in `docs/rustcnc_strategy_port.md`; existing typed task/motion contracts.
- **Owns:** `cam/rustcnc_face_contour.py`, `tests/rustcnc_face_contour_test.py` (worker-1), exclusively for this card.
- **Work:** port the relevant Face band/depth ladder/approach and Contour 2D offset/lead core for the current analytic rectangular family. Expose a reusable Contour finishing emitter. Preserve source constants/order and label all coarse/novel parameter choices explicitly. Closed unsupported shapes are preferable to invented output.
- **Gate:** source-derived tests, deterministic typed plans, safe approach/retract, valid feeds/frames, coarse chords and complete ≤72-segment output; explicit over-cap refusal with no trimmed machining, focused and full default tests, and an exact retained-pad/topology coverage report.

### CAM-RUST-2 — Pocket 2D direct port

- **Depends on:** pinned RustCNC Pocket 2D source and typed task/motion contracts; no need to await an unrelated worker file.
- **Owns:** `cam/rustcnc_pocket2d.py`, `tests/rustcnc_pocket2d_test.py` (worker-2), exclusively for this card.
- **Work:** directly port pocket offsets, pass/depth entry and links for rectangular open/retained-pad tasks. Separate the roughing operation from the coordinator's Contour finish. A lower retained-pad cap requires its explicit clearing phase, never bottom-level gouging.
- **Gate:** source-derived and retained-pad tests, all attempted task/topology outcomes, safe valid complete plans under the hard cap, and focused plus full default tests. No optimizer-performance filtering or legacy-raster fallback.

### CAM-RUST-3 — Adaptive 2D direct port

- **Depends on:** pinned RustCNC Adaptive 2D source, including its explicitly unpinned stand-ins and limitations.
- **Owns:** `cam/rustcnc_adaptive2d.py`, `tests/rustcnc_adaptive2d_test.py` (worker-3), exclusively for this card.
- **Work:** port the tool-centre erosion, top-down levels, entry helix/circle, clearing frontier and clear-link-or-retract logic. Coarsely sample finite chords under a declared entry recipe. Explicitly clear a lower retained-pad cap before any lower-floor clearing. Do not claim constant-engagement or Fusion parity beyond the source evidence.
- **Gate:** source-level algorithm tests plus over-cap/entry/link/retained-pad cases; report how every current family task fits or explicitly refuses the total 72-segment cap. Deterministic safe typed plans, focused and full default tests, no path truncation and no physical execution.

### CAM-RUST-4 — Seed replacement, integration qualification and replacement experiments

- **Depends on:** CAM-RUST-1..3 and a reviewed integration decision for any unsupported current task; CAM-RUST-4 alone does not resurrect stopped runs.
- **Owns:** coordinator's existing CAM seed/suite/qualification leases, fresh generated suite manifests/protocol/configs/output roots, and the integration handoff.
- **Work:** route typed families to the sourced strategies, compose Pocket roughing with Contour finish without redundant/gouging cuts, include explicit pad-cap phases, and record the port/chord/cap identity. Replace default nominal starts while preserving old outputs as old protocol. Regenerate qualified evidence and a new config/manifest identity, run a new dev pilot, amend T4.2, and only then execute a fresh bounded campaign; regenerate final results and physical-review packets from the new evidence.
- **Gate:** full suite plus actual nominal stock/target and both-dialect checks on the claimed integration cohort, complete cap/coverage accounting, no skipped current task based on performance, a frozen amended protocol/dry-run, new dev integration pilot, and completed replacement runs with signed absolute and relative cycle deltas. Never reinterpret stopped non-complete entries as failures or successes. Finish with the user's requested Slack MCP verifier handoff; no physical trials are implied.
