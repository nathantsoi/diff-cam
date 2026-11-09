# DiffCAM detailed agent execution plan

Authority: [tasks.md](tasks.md). Scientific context: [experimental plan](reports/experimental_plan.md).  
Scope: planning implementation and experiments; this document does not report any task as implemented.  
Concurrency limit: **one coordinator plus at most three workers**.

## 1. Coordination and ownership

The coordinator maintains the integration checkout on `core`, approves interface revisions, owns `tasks.md`, this file, dependency files, and the campaign registry, and verifies integration gates. Agents receive one bounded card at a time, not “implement the paper.” A worker may take a new role after finishing a card.

Prefer separate worktrees/branches rooted at the same integrated commit. An agent must not switch branches in another agent's checkout. If isolation is unavailable, use explicit disjoint file ownership and serialize all shared-file edits. Do not run concurrent environment synchronization against the same virtual environment or timing experiments on an oversubscribed device.

### File leases

| Area | Exclusive writer | Rule |
|---|---|---|
| `benchmarks/contracts.py`, `docs/contracts.md` | Coordinator/contracts owner | Downstream interface changes require a versioned handoff |
| `pyproject.toml`, `uv.lock`, `.gitignore` | Coordinator | Workers request dependency changes; isolate incompatible baseline environments |
| `simulator/csg_simulator.py` | Geometry owner | T1.1 → T1.2 → T2.0; horizon hooks later by explicit lease |
| Existing `cam/` parser/posts/config/planner | Motion owner | T1.4; later seed/optimizer integration changes by explicit lease |
| `verification/reference_carve.py`, reference fixtures | Reference owner | Independent expected results must not call production sweep helpers |
| `cam/posted_evaluator.py`, `simulator/csg_metrics.py` | Evaluator owner | T1.3 after reference/motion/geometry integration |
| `algorithms/objectives.py`, `algorithms/run_artifacts.py` | Objective owner | Merge before parallel solver implementation |
| `algorithms/optimize.py` | Gradient owner | T3.2 first; continuation/horizon hooks serialized |
| Aggregation and run manifests | Evaluation owner/coordinator | Distinct campaign shard outputs; only coordinator edits frozen protocol |
| `manuscript/` | Publication owner | Preserve existing user edits; generated artifacts have explicit destinations |

A lease covers a file even when two agents intend to change different functions. Independent agents can draft tests or proposals, but only the assigned owner merges edits to that file.

### Required handoff for every card

Provide:

1. Task ID, input commit and upstream schema/API revisions.
2. Commit or reviewable diff and exact owned files changed.
3. Validation commands, exit codes, backend/environment, and results.
4. Artifact paths and task/input/NC hashes where applicable.
5. Known failures, coverage limits, unavailable dependencies, and unresolved questions.
6. Requested downstream API changes and reproduction instructions.

Keep a coordinator-maintained ledger, preferably `reports/task_status.json`, with `planned`, `ready`, `in_progress`, `blocked`, `complete`, or `deferred`; blockers name their dependency and next action. All cards below initially have status **planned**. A research result can be negative and still complete its card. An external compatibility investigation may complete with “unavailable,” but that does not mark a corresponding performance comparison complete.

## 2. Parallel scheduling waves

These are recommended allocations, not global barriers. Start a ready task when its inputs and file leases are available. Each cell lists at most one active worker; sequential tasks inside a cell run in order.

| Wave | Coordinator | Worker A | Worker B | Worker C | Integration checkpoint |
|---|---|---|---|---|---|
| W0 | T0.0 contracts, then T0.2 environment | T0.1 selective references | Read-only fixture design, then T1.0 once inputs merge | Read-only external inspection, then T5.0 once inputs merge | Contracts and core environment accepted |
| W1 | Integrate foundational handoffs | T1.4 NC semantics | T1.0 reference implementation | T2.1 parameterization | Reference fixtures, typed motion and parameter maps accepted |
| W2 | Integrate and triage references | T1.1 sweeps → T1.2 holder | T4.0 auditor/statistics | T5.0 finish compatibility | Correctness gates; no competing simulator writers |
| W3 | Integration tests | T2.0 custom geometry | T1.3 posted evaluator | Finish outstanding T2.1/T4.0/T5.0 | Evaluator and custom targets accepted |
| W4 | Pilot coordination | T3.0 objective/artifacts | T2.2 one pocket after T2.0/T1.3 | T5.1 CNC-Net adapter after its dependencies | Shared objective and complete nominal pocket |
| W5 | Integrate solver branches | T3.1 black-box solver | T3.2 Adam variants | T5.2 external/Titan pilot | Two solvers share one objective/representation |
| W6 | T4.1 integrated pilot | Assist only with assigned integration fixes | Assist only with assigned integration fixes | Finish external pilots | Real end-to-end records and runtime estimates |
| W7 | Scope decision; prepare protocol | T2.3 procedural expansion | T3.4 continuation candidate | T3.3 gradient mechanics | Bounded dev evidence; keep/drop decisions |
| W8 | T4.2 freeze; schedule optional T3.5 separately | Selected ablations or T3.5 under lease | Finish external setup qualification | Audit manifest and predicted compute | No held-out launch until frozen contract passes |
| W9 | T4.3 campaign supervision | Procedural shard | External shard | Audit completed records | Resource-isolated runs, complete coverage ledger |
| W10 | T7.2 publication integration | T6.1 packets → T6.2 ingestion | Final statistical/artifact audit | Optional T7.1 policy track | Claims match available evidence |

W0/W1 repeat labels mean a task may carry over, not that two agents own it at once. Reassign the same owner or explicitly transfer its lease. W9's shards may execute sequentially on one GPU; parallelize CPU preparation/audit while the GPU is occupied. Physical experiments themselves require a separately scheduled qualified operator and actual records.

Critical path for the first result:

`T0.0/T0.2 → T1.0 → T1.1 → T1.2 → T2.0`, joined with `T1.4 → T1.3`, then `T2.2 + T2.1 → T3.0 → T3.1/T3.2 → T4.1`.

T4.0 and early external compatibility run alongside this path. The full publication path then adds suite construction, optional mechanism experiments, frozen evaluation and claim-specific validation.

## 3. Agent task cards

Acceptance criteria in [tasks.md](tasks.md) are authoritative. The commands below and those in the registry are targets to implement, not statements that these modules already exist.

### T0.0 — Contracts owner

- **Inputs:** current simulator/CAM interfaces and the shared contracts section of `tasks.md`.
- **Deliver:** `benchmarks/contracts.py`, `docs/contracts.md`, `tests/contracts_test.py`, tiny serialized fixtures.
- **Steps:** define typed task/motion/evaluation/run records; fix units, coordinate frames and initial state; specify hashing and unknown/failed states; document `evaluate_candidate`, `evaluate_posted_gcode`, parameter pullback and artifact writer interfaces. Keep the contract small enough for the first pocket.
- **Acceptance:** `uv run python -m pytest tests/contracts_test.py -q`; round trips, invalid units, unsupported coverage, failed-run records, and hash stability. Obtain consumer acknowledgement before freezing v1.
- **Handoff:** API examples for independent reference, motion, parameterization, evaluator and aggregator owners. **Parallel:** external acquisition and read-only reconnaissance only until the contract merges.

### T0.1 — Reference acquisition owner

- **Inputs:** selected URLs in the experimental plan; no simulator dependency.
- **Deliver:** selective idempotent acquisition script and tracked `configs/references.json`.
- **Steps:** pin pycma, CNC-Net and CAMotics first; check license evidence and actual downloaded files; preserve dirty clones; support verify-only/offline mode. Later subsets remain optional until consumed. Ask coordinator for ignore rules rather than editing shared files concurrently.
- **Acceptance:** acquire/verify a requested subset twice with unchanged pins; test missing checkout/network/payload behavior. Report exact commands and reproducibility status.
- **Handoff:** revisions, local paths and environment obstacles to T3.1/T5.0/T1.3. **Parallel:** all foundation work.

### T0.2 — Coordinator/environment owner

- **Inputs:** current project and existing test suite; independent of reference cloning.
- **Deliver:** reproducible core lock, extras/environment policy and `docs/environment.md`.
- **Steps:** validate CPU installation and smoke in a unique temporary directory; isolate CAD and CNC-Net environments; record actual hardware. Coordinate dependency requests into one patch.
- **Acceptance:** `uv sync --locked`, `uv run python -m pytest -q`, and existing optimizer smoke once lock generation succeeds. A blocked fresh install is recorded; do not silently substitute cached tests for locked reproducibility.
- **Handoff:** exact command prefix and environment fingerprint for every worker. **Parallel:** T0.1 and contract design; runtime work starts after a usable environment is established.

### T1.0 — Independent reference owner

- **Inputs:** merged T0.0/T0.2; analytic fixture specifications.
- **Deliver:** independent reference carve/chronology implementation and expected geometry fixtures.
- **Steps:** implement exact simple occupancy cases; dense reference sampling for general motion; quantify refinement error; include must-pass and must-fail safety cases. Vary radius, grid phase and segment degeneracy. Avoid importing production sweep helpers even indirectly.
- **Acceptance:** reference tests plus saved convergence report. Expected volumes account for initial stock intersection and initial tool placement.
- **Handoff:** fixture inputs/expected outputs/uncertainty to T1.1/T1.2/T1.3. **Parallel:** T1.4 and T2.1.

### T1.1 — Production geometry owner

- **Inputs:** T1.0 fixtures and current Taichi kernel.
- **Deliver:** corrected cutter/tip hard and smooth sweep paths; `tests/sweep_geometry_test.py`.
- **Steps:** select a justified membership algorithm; define surrogate separately; document approximation and degeneracies; compare production outputs to reference; verify gradients with epsilon sweeps. Do not conflate implicit sign fields with exact distance.
- **Acceptance:** registry gate plus convergence for ramps and subdivision; existing core tests retained. Absolute and relative gradient tolerances reflect float32 and nondifferentiable boundaries.
- **Handoff:** approved sweep semantics, runtime cost and limitations to chronology/evaluator owners. **Parallel:** statistics and external reconnaissance; exclusive simulator lease.

### T1.2 — Production geometry owner, next serial card

- **Inputs:** merged T1.1 and T1.0 chronology fixtures.
- **Deliver:** consistent conservative pre-cut holder penalties/diagnostics and chronology tests.
- **Steps:** change all holder consumers consistently; distinguish screened overlap from chronological collision; demonstrate cutter-leading-holder false-positive behavior and refinement; test loss-reducing descent rather than an asserted derivative sign.
- **Acceptance:** both independently specified true collision and valid clearing cases; complete units/assembly definitions. Safe case must remain safe in converged time-resolved evaluation even if coarse screening flags it.
- **Handoff:** holder API and chronology assumptions to T1.3, simulator lease to T2.0. **Parallel:** T4.0/T5.0.

### T1.4 — Motion semantics owner

- **Inputs:** T0.0 motion/task contract and T0.2 environment.
- **Deliver:** parser/post/planner support and tests for the declared dialect subset.
- **Steps:** implement initial-state and frame handling; retain segment types/feeds; check `G53`, offsets, units and arcs; define tool-length assumptions; reject/report unsupported commands. Document supported timing semantics and explicit coverage.
- **Acceptance:** expected machine-frame coordinates specified independently of the post; first approach/home retract verified; existing CAM regressions and new semantics tests pass.
- **Handoff:** canonical typed-motion examples to T1.3/T2.2/T3.2. **Parallel:** T1.0/T2.1; exclusive existing CAM-file lease.

### T1.3 — Evaluator owner

- **Inputs:** T1.0/T1.1/T1.2/T1.4 and T0.0; selected external verifier installation.
- **Deliver:** posted evaluator, metrics, independent adapter and `tests/posted_oracle_test.py`.
- **Steps:** stream stock chronologically; no rapid carving; resolve arcs and frame transforms; refine spatial/time steps; track coverage and distinct collision metrics. Use reference fixtures for mandatory checks; preserve separate dialect results and input hashes.
- **Acceptance:** registry tests and an actual external-system cross-check on a supported fixture. Missing external binary is a blocked integration item. Empty surfaces and unknown commands produce defined outcomes.
- **Handoff:** callable evaluator, tiny golden records with numerical tolerances, and evaluated NC examples. **Parallel:** T2.0 because its simulator file and evaluator files have different leases.

### T2.0 — Production geometry owner, third serial card

- **Inputs:** T1.2 and T0.0 grid contract.
- **Deliver:** custom stock/target integration across all target consumers.
- **Steps:** enumerate target consumers before editing; implement sampled query/interpolation and out-of-bounds behavior; update initialization, scalar/dense query, terminal loss, volume and position penalties. Test noncubic physical bounds.
- **Acceptance:** custom-grid tests and analytic regressions; a changed target affects both terminal and trajectory penalties; interpolation gradients agree away from cell boundaries.
- **Handoff:** arbitrary-target fixture and API to T2.2/T3.0/T5.1. **Parallel:** T1.3; exclusive simulator lease.

### T2.1 — Parameterization owner

- **Inputs:** T0.0/T0.2; no simulator correction dependency.
- **Deliver:** delta, spline and optional absolute residual maps plus exact pullbacks.
- **Steps:** specify T as segments versus positions; anchor start/end appropriately; define physical parameter scaling, K limits, and segment boundary behavior. Absolute mode uses a differencing matrix and its transpose, not numerical differentiation of the mapping.
- **Acceptance:** float64 algebra tests, zero residual, and pullback finite differences on a synthetic objective. Taichi integration error checked later at suitable tolerance.
- **Handoff:** interface and small arrays to T2.2/T3.0. **Parallel:** NC/reference work.

### T2.2 — Pilot-task/seed owner

- **Inputs:** T2.0/T2.1/T1.3; merged motion contract.
- **Deliver:** one dev pocket, typed seed generator and full nominal artifacts.
- **Steps:** define stock/target/tool/setup; create a competent seed with retained rapid/plunge/feed timing; compare zero-residual executed and posted results. Resolve fixed-dt/speed-clipping differences with the motion owner. Describe seed provenance and any simplifications.
- **Acceptance:** completion, coverage and safety for the nominal program; zero-residual geometry/motion agreement with documented discretization tolerances.
- **Handoff:** frozen tiny task and nominal records to solver/pilot owners. **Parallel:** T3.0 and CNC-Net pilot.

### T3.0 — Objective/record owner

- **Inputs:** T2.0/T2.1/T1.2 and shared contracts.
- **Deliver:** stateless candidate API, hard/smooth objective definitions, resource ledger and artifact writer.
- **Steps:** reset all mutable state including accumulated loss; evaluate A/B/A; implement consistent invalid-candidate outcomes and hard/soft weights; save data before another pass overwrites it. Include candidate IDs and timing/coverage fields even before final posted evaluation.
- **Acceptance:** objective test gate, tape/no-tape equality, order invariance and synthetic parameter-gradient pullbacks. Baselines must receive identical objective settings for each matched comparison.
- **Handoff:** stable versioned API to two solver agents; late changes require coordinated updates. **Parallel:** T2.2/T5.1.

### T3.1 — Derivative-free solver owner

- **Inputs:** T3.0 plus pycma pin/environment extra from T0.1/T0.2 for development; T2.2/T1.3 for real-task and posted-artifact acceptance.
- **Deliver:** hard/smooth CMA-ES adapter first; CEM/separable variants after pilot justification.
- **Steps:** pass shared theta mapping; implement budgets/population accounting, invalid states, reproducible seeds, checkpoint policy and resume records; expose batching without changing evaluation semantics.
- **Acceptance:** known synthetic optimization test, budget exhaustion and repeatability, plus tiny task integration. Improvement over another method is not a test assertion.
- **Handoff:** CLI/config condition definitions and sample artifacts to T4.1. **Parallel:** T3.2 after common API merges.

### T3.2 — Gradient solver owner

- **Inputs:** T3.0 for development; T2.2/T1.3 for real-task and posted-artifact acceptance.
- **Deliver:** shared-contract fixed/annealed Adam and coherent checkpoints.
- **Steps:** connect pullback, gradient clipping, schedules, budgets and hard selection; retain accepted parameter/config/moment snapshots; emit typed motion and per-dialect results. Preserve small legacy smoke functionality where practical.
- **Acceptance:** optimizer tests and tiny task integration; replay selected checkpoint reproduces scored path/metrics within backend tolerance.
- **Handoff:** baseline configurations and extension interface to T3.4; full run to T4.1. **Parallel:** T3.1; exclusive optimizer lease.

### T4.0 — Evaluation/statistics owner

- **Inputs:** T0.0/T0.2; synthetic records sufficient for initial development.
- **Deliver:** artifact auditor and paired aggregation before campaign code.
- **Steps:** reject incorrect hashes, missing dialect identity and malformed finite values; retain explicit nulls and failures; implement task-level paired summaries with nested seeds and family coverage. Declare independent units for binary intervals; include multiplicity policy and missing-data rules.
- **Acceptance:** known calculation tests, replicated-seed pseudoreplication test, missing/partial run audit, no-cut efficiency exclusion. Negative comparisons must render normally.
- **Handoff:** example pilot table and audit API to T4.1/T4.2. **Parallel:** geometry/NC work.

### T4.1 — Coordinator/integration pilot owner

- **Inputs:** T2.2/T3.1/T3.2/T4.0.
- **Deliver:** benchmark runner, pilot config, actual one-task comparison and bottleneck report.
- **Steps:** validate advertised CLI enums; construct run IDs and resume rules; dry-run predicted counts/cost; run CAM, fixed Adam, annealed Adam, hard CMA-ES and smooth CMA-ES at one K; audit both dialects. Keep JIT and end-to-end timing.
- **Acceptance:** runner tests plus complete real records; explain observed failures and resource costs. Fix correctness before expanding; do not tune until a favorite method wins.
- **Handoff:** common execution path and measured budget to T2.3/T3.3/T3.4/T4.2. **Parallel:** external compatibility pilots; integration fixes require explicit file leases.

### T5.0 — External compatibility owner

- **Inputs:** T0.0 and selected available T0.1 pins.
- **Deliver:** factual compatibility matrix and baseline-adaptation plan early in the project.
- **Steps:** inspect actual CNC-Net code/data/checkpoints; attempt minimal native invocation in isolation; inspect one Titan CAD/setup path; check CNCGEN preview fields and provenance. Record unavailable assets/licensing rather than guessing APIs.
- **Acceptance:** evidence and commands support each supported/unsupported claim; no success claimed solely from a repository clone.
- **Handoff:** bounded adapters for T5.1/T5.2; early scope warnings for coordinator. **Parallel:** nearly all foundation work after contracts.

### T5.1 — CNC-Net adapter owner

- **Inputs:** T5.0/T1.3/T2.0.
- **Deliver:** native geometry scoring and optional faithful executable adapter on one compatible dev instance.
- **Steps:** isolate environment/process; transform units and targets; map operations only when justified; record additional transitions and restrictions. Track pretraining and refinement separately. Native reconstruction remains available even if executable comparison is unsupported.
- **Acceptance:** adapter tests and real reproduction artifacts, or documented incompatible/unavailable disposition. Freeze comparison scope before held-out performance inspection.
- **Handoff:** external baseline condition or explicit exclusion rationale to T4.2. **Parallel:** solver development; no edits to shared evaluator.

### T5.2 — External/Titan task owner

- **Inputs:** T5.0/T2.2.
- **Deliver:** one Titan operation and one feature-task pilot, then a bounded external manifest.
- **Steps:** use a real STEP backend; define original target and reachable operation target separately; validate finite-radius/assembly accessibility; produce nominal CAM and independent posted checks. Preserve dataset license/provenance and avoid interpreting CNCGEN reference paths as model output.
- **Acceptance:** external task tests and actual nominal artifacts. Exclusions are geometry/setup decisions made before solver scores; inaccessible volume is reported.
- **Handoff:** external task manifest to T4.2 and physical setup candidate to T6.1. **Parallel:** optimizer/pilot development in disjoint files.

### T2.3 — Procedural expansion owner

- **Inputs:** T2.2/T4.1.
- **Deliver:** facing, contour/island and variable-width/depth families with manifests and nominal seeds.
- **Steps:** use pilot costs to set suite size; generate dev/test separation by topology; verify every seed and units/setup contract; retain source hashes and generation version. If modifying `cam/seeds.py`, obtain its lease from T2.2's owner.
- **Acceptance:** deterministic suite generation, split/provenance and geometric validity tests. Invalid tasks cannot disappear based on optimizer performance.
- **Handoff:** candidate frozen task list to T4.2. **Parallel:** optional method/mechanics development.

### T3.4 — Continuation research owner

- **Inputs:** T4.1 baseline data and approved extension API.
- **Deliver:** separate signed-shift and disagreement/backtracking mechanisms with ablations.
- **Steps:** follow T3.4's signed correction and same-path requirement; bound updates/retries; compare both candidates at identical settings; define infeasible-start recovery and rollback. Integrator handles shared optimizer edits under lease.
- **Acceptance:** sign, no-drift, zero-gap, constraint, rollback and budget tests; dev report compares against fixed/annealed references. Valid result includes “no useful advantage; omit from paper.”
- **Handoff:** frozen candidate configuration or rejection decision to T4.2. **Parallel:** T2.3/T3.3; baseline API unchanged unless coordinated.

### T3.3 — Gradient-mechanics owner

- **Inputs:** T4.1; T3.4 only for candidate-specific curves.
- **Deliver:** fixed-path smoothing/segmentation diagnostics and gradient-estimator uncertainty.
- **Steps:** verify AD on the same smooth objective; implement explicit randomized hard objective using standard Gaussian perturbations in declared normalized parameter coordinates and antithetic evaluation accounting; measure high-sample reference uncertainty; handle near-zero gradients and coordinate scaling. Identify the randomness underlying any variance claim.
- **Acceptance:** estimator test on known function, finite-difference epsilon study, fixed-path identity and correct 2N counts. Plot observed behavior without enforcing U-shape.
- **Handoff:** bounded mechanism-study configuration and results to T4.2/T7.2. **Parallel:** procedural expansion and continuation.

### T3.5 — Optional horizon owner

- **Inputs:** T4.1 plus exclusive simulator/optimizer lease for required hooks.
- **Deliver:** clearly defined horizon objective and measured compute/memory study.
- **Steps:** choose truncation versus full-gradient checkpointing explicitly; test H=T equivalence where appropriate; refresh detached states after changed prefixes; implement window allocation/recomputation if claiming reduced memory.
- **Acceptance:** horizon tests, actual peak memory and wall time, and documented objective bias. No memory claim from merely using `save_state`/`restore_state`.
- **Handoff:** optional protocol addition or deferral. **Parallel:** disjoint task/analysis work only; never overlap simulator writers.

### T4.2 — Coordinator/protocol owner

- **Inputs:** T2.3/T4.1/T5.0/T5.1/T5.2 dispositions and any retained candidate/mechanism tasks.
- **Deliver:** frozen benchmark config, complete run manifest and scientific decision record.
- **Steps:** choose one primary dimension and bounded ablations; establish all metric/feasibility thresholds and primary comparisons; record tuning expenditures; estimate run count/compute/VRAM; predeclare failures and unavailable baseline handling. Version test task hashes and evaluator.
- **Acceptance:** dry-run validates every combination, required dependency and output destination; no test-tuned parameters or strict-improvement condition. Freeze before scheduling T4.3.
- **Handoff:** immutable manifest to campaign workers and auditor. **Parallel:** documentation/physical packet preparation only once requisite scope is fixed.

### T4.3 — Campaign shard agents

- **Inputs:** same frozen T4.2 protocol and executable revision for every shard.
- **Deliver:** disjoint assigned run IDs, artifact directories and execution ledger.
- **Steps:** coordinator allocates explicit IDs/devices/output roots; workers run only their shard, preserve failures, and resume only identity-matching runs. Record hardware contention and timing conditions. Independently audit completed results while later runs execute.
- **Acceptance:** expected/observed ID reconciliation, artifact/hash/coverage validation and recorded failures. Never fill missing measurements with successes. Numerical fixes require a versioned amendment and affected reruns.
- **Handoff:** immutable results to T7.2. **Parallel:** only within measured compute capacity; one GPU usually means serialized timed runs.

### T6.1 — Physical packet owner

- **Inputs:** T5.2 qualified setup definition and T4.2 frozen protocol.
- **Deliver:** machine-specific packets and setup sheets with verification evidence and review state.
- **Steps:** pin machine/tool/holder/fixture/work-offset details; require local error and clearance margins; qualify both approach and retract; retain NC hashes; label force proxies. Document manual setup verification requirements rather than encoding a generic 0.5% gouge threshold as physical qualification.
- **Acceptance:** packet tests reject unknown coverage and inconsistent setups. A valid packet completes software packaging, not physical experimentation.
- **Handoff:** reproducible reviewable packet to the qualified operator and T6.2. **Parallel:** campaign execution where no shared resources conflict.

### T6.2 — Physical measurement owner

- **Inputs:** T6.1 schema/packets; actual trial data for empirical results.
- **Deliver:** ingestion/registration/uncertainty pipeline and traceable measured comparisons.
- **Steps:** verify setup/NC identity; align scans using a declared procedure; retain calibration and uncertainty; distinguish spindle load from calibrated force; record rejected/interrupted trials with reasons.
- **Acceptance:** synthetic fixture tests validate the software; physical evidence requires real logs and measurements. Status remains pending for empirical conclusions until those exist.
- **Handoff:** measured result records to T7.2. **Parallel:** publication code and separate policy research.

### T7.1 — Optional policy owner

- **Inputs:** T2.3/T3.0/T1.3 and a separate train/dev/test protocol.
- **Deliver:** residual environment, Taichi-policy gradient bridge, PPO/SAC/SHAC adapters and small training pilot.
- **Steps:** align observations, action bounds, reward and nominal priors; handle partial observability consistently; isolate incompatible dependencies; verify reset/detach/backprop semantics; separate training, inference and per-instance adaptation budgets.
- **Acceptance:** environment tests plus actual held-out pilot and gradient-bridge verification. No policy-generalization claim from direct trajectory optimization; no physical exploration implied.
- **Handoff:** separate policy experiment or deferral, not a blocker for the first trajectory paper. **Parallel:** final artifact work if compute is isolated.

### T7.2 — Publication/reproduction owner

- **Inputs:** T4.3 and T4.0; physical/policy evidence only if included in claims.
- **Deliver:** generated manuscript tables/figures and `reports/final_audit.md`.
- **Steps:** audit every figure/table against hashes and source runs; use task-level uncertainty; show failures and external coverage; distinguish simulated, measured and proxy values; retain original target exclusions. Update novelty against primary sources before submission.
- **Acceptance:** clean regeneration with declared dependencies and complete claim-to-evidence mapping. A missing physical trial cannot be replaced by the physical packet test. Preserve existing manuscript content unless a reviewed edit requires changing it.
- **Handoff:** publication-ready artifact manifest and remaining limitations to coordinator.

## 4. Integration and gate policy

Before assigning a card, the coordinator checks upstream artifacts, file leases, environment availability and output paths. An agent can prototype against frozen schema fixtures while dependencies develop, but the final task gate must use merged real implementations.

At handoff, review the diff, run focused checks, and integrate in dependency order. Rebase/retest dependent branches after interface changes. Do not rerun the complete suite for every documentation-only edit; run affected tests plus the full core suite at meaningful integration checkpoints.

Three distinct gates apply:

1. **Implementation gate:** behavioral tests and review pass.
2. **Integration gate:** real task/program/artifact paths work with merged dependencies and required independent verification.
3. **Scientific gate:** evidence is complete enough to accept, reject or narrow a hypothesis. A null or unfavorable result is valid evidence.

External-tool unavailability must be visible in the ledger. Continue unrelated work, then decide explicitly whether to obtain the tool, use another independent implementation, or narrow the experimental claim. Never silently mark the blocked cross-check complete.

## 5. First dispatch and stop points

The first dispatch is limited to contracts/environment, selective reference acquisition, and read-only fixture/external reconnaissance. Once contracts and a usable CPU environment are merged, dispatch independent reference, NC semantics and parameterization. Do not launch the full old task list concurrently.

Stop and reassess scope after:

- Reference/production disagreement fails to converge.
- A nominal CAM seed cannot pass the declared operation target.
- Zero-residual training/posting equivalence fails.
- Candidate evaluation depends on previous candidates.
- External adapters require changing the baseline's task beyond a defensible comparison.
- Pilot cost makes the proposed Cartesian sweep impractical.
- The continuation candidate does not add value under bounded dev testing.

These trigger a recorded technical/scientific decision, not an instruction to manufacture a passing result. The coordinator should select the smallest complete experiment that answers the research question and expand from demonstrated evidence.
