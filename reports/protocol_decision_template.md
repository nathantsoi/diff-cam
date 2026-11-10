# T4.2 protocol decision TEMPLATE — coordinator fills after pilot; not a frozen protocol

This file is a checklist, not a scientific decision, measured result, or launch authorization. The coordinator supplies every frozen value after reviewing actual pilot evidence. Empty declarations are not defaults, permission to run, or approval to machine. No text here replaces the coordinator's sign-off, actual pilot records, or real numerical validation.

## 1. Protocol identity and review scope

- Decision ID and amendment chain: `<D-YYYY-MM-DD-NN + any superseding amendments>`
- Status: `<draft / frozen-after-signoff>`
- Authorizer and actual review date: `<coordinator and date>`
- Frozen config path and exact byte SHA-256: `<path, sha256>`
- Schema version: `diffcam-frozen-benchmark/1`
- Benchmark-runner schema/version: `<native version, currently diffcam-benchmark-runner/1>`
- Source commit and optional dirty-patch hash: `<actual executables>`
- Data/CAD/external source provenance, licenses and asset hashes: `<records; local-only or withheld assets remain undeclared/redistribution-limited>`
- Why this bounded evaluation is being performed, including negative/null hypotheses: `<written aims, not an expected ranking>`
- Allowed conclusions: `accept`, `reject`, `narrow`; strict-improvement gate required: `false`, always.

Maintain historical decision context, including D-2026-11-09-03 (gradient provisional history), D-2026-11-09-04 (fidelity), D-2026-11-09-05 (full-f64 comparisons), D-2026-11-09-06 (where precision is recorded), and D-2026-11-09-07 (posted worker memory ceilings), plus newer amendments. An amendment after inspecting results requires an explicit reason, new IDs/hashes, and affected-run reruns; never silently replace an inconvenient run.

## 2. Task source, splits and manifest identities

| Selection | Task ID | `TaskSpec.task_hash` | Source/plugin | Split | Split digest SHA-256 |
|---|---|---|---|---|---|
| Primary task | `<task_id>` | `<task_hash>` | `<registered factory>` | `<dev/test/external>` | `<digest>` |
| Dimensionality/variant subset | `<up to three explicit entries>` | `<hashes>` | `<registered factory/factories>` | `<split(s)>` | `<digests>` |

- Manifest path and exact byte SHA-256: `<path, sha256>`
- Split hashes: dev `<digest>`, test `<digest>`, external `<digest>`.
- Split-hash rule used by this tooling: canonical UTF-8 JSON of sorted `{"task_id", "task_hash"}` pairs, compact JSON with sorted keys, no newline added. Empty split selection is explicit null, not an invented SHA.
- Actual acquisition/reproduction steps and immutable source locations: `<instructions; no network/download is implied by this template>`.
- Prototype probe task ID/hash, if using the current limited adapter: `<prototype ID and hash>`.

The current native adapter executes **only `prototype_analytic`, `dev`, and `pilot`**. Other experiment/split declarations need their matching runner/task-source adapter before any actual execution. `--validate-only` supports declaration/hash review but is not evidence that a campaign ran. Do not substitute the prototype for a different selected production task.

Record original part, reachable operation target, excluded volume, finite-cutter/holder exclusions and grid exclusions separately. Native excluded-volume values are diagnostic; grid-resolved exclusions are normative at exact zero sampling tolerance. Never replace original part P with sampled T, resample one target to generate a more favorable result, or turn a no-cut into a passing result.

| Decision | Exact value(s) or reference |
|---|---|
| Task completion/residual tolerance | `<mm³, matches selected TaskSpec>` |
| Gouge volume threshold | `<mm³, matches TaskSpec and is not a physical clearance gate>` |
| Local surface-error threshold | `<mm, matches TaskSpec>` |
| Clearance/rapid-penetration limits | `<mm, matches TaskSpec>` |
| Removal required | `<true/false; if true, zero removal fails completion and cannot support efficiency>` |
| Evaluator coverage and missing outcomes | `<explicit complete/incomplete/unsupported/unknown policy; never reclassify unknown as pass>` |

## 3. Conditions, primary K and bounded subset

| Retained condition | Runner experiment/split | Task ID/hash | Allowed seeds | Precision config/hash | Fixed budget/variant roles |
|---|---|---|---|---|---|
| `nominal-cam` | `<...>` | `<...>` | exactly `[0]`, one deterministic anchor | `<...>` | `<primary only unless explicitly selected>` |
| `fixed-adam` | `<...>` | `<...>` | exactly `[0]` under the current runner | `<...>` | `<primary and/or selected variant>` |
| `annealed-adam` | `<...>` | `<...>` | exactly `[0]` under current runner | `<...>` | `<...>` |
| `hard-cmaes` | `<...>` | `<...>` | `<explicit nonnegative integer seeds>` | `<...>` | `<...>` |
| `smooth-cmaes` | `<...>` | `<...>` | `<explicit nonnegative integer seeds>` | `<...>` | `<...>` |

No duplicate condition entries or seed repetitions masquerading as extra independent CAM/Adam results. CMA runs are once per declared seed. Name all retained experiment condition IDs; unknown runner condition names are forbidden. A concept/experiment not present in the runner is either registered with evidence or unavailable, not masked by a similar known condition.

| Objective/control | Frozen declaration |
|---|---|
| Primary K | `<positive float, one primary K>` |
| Anneal start/end | `<positive start, larger positive end>` |
| Gouge/residual/holder weights | `<nonnegative floats>` |
| Holder margin (mm) | `<nonnegative value>` |
| Signed loss shift (voxel units) | `<finite signed value, explicitly not an unlabelled millimetre offset>` |
| Learning rate/max-gradient norm | `<positive floats>` |
| CMA `sigma0` and population size | `<positive float, integer >= 2>` |

### Small K/dimensionality/ablation subset (1–3 explicit variants, never automatic Cartesian expansion)

| Variant label | Task ID/hash | Retained condition + seed | K (distinct from primary K) | Exact rationale/budget |
|---|---|---|---|---|
| `<filename-safe-label>` | `<...>` | `<...>` | `<positive value>` | `<...>` |
| `<optional second>` | `<...>` | `<...>` | `<positive value>` | `<...>` |
| `<optional third>` | `<...>` | `<...>` | `<positive value>` | `<...>` |

Document which physical/design dimension is being varied, why its count is bounded, and how each variant contributes to hypothesis acceptance/rejection. Do not silently generate a task × K × method grid, stepwise-explore new knobs after seeing external/test results, or call repeated CAM runs independent evidence. All selected conditions and variant task identities must exist in the registered manifest; current prototype execution can't silently handle another task source.

## 4. D-05 precision and D-06 metadata

| Field | Frozen declaration | Actual evidence to record |
|---|---|---|
| Campaign optimizer precision | `f64` | `<required for every matched optimizer condition>` |
| Native runner `precision` | `f64` | `<matches protocol and normalized native config>` |
| `simulator_precision` in run/optimizer setup | `f64` | `<explicit metadata and signatures from all runs>` |
| Taichi `default_fp` | `<matching f64 in each process>` | `<actual initialized runtime; one precision per process>` |
| Primal and adjoint dtypes | `<f64 fields/stock-history>` | `<actual observed buffers, not an inference from the label>` |
| External normalized-delta input assignment | `<existing f32 assignment remains distinct>` | `<do not claim every external input array is f64>` |

Matched optimizer comparisons do not mix f32/f64 conditions. F32 may remain an exploration option, not part of this matched campaign. Widening only the scalar loss is not equivalent to f64 stock/history adjoints and does not fix the observed gradient bias. Check the actual longest-path numerical validation for the retained production revision and precision, rather than assuming an argument name proves numerical correctness.

## 5. D-04 posted-evaluation levels and evidence roles

| Role | Frozen maximum step | Dialects | Required meaning |
|---|---|---|---|
| Screen | `1.0 mm` | RS274 and Haas | diagnostic/report only, never ANDed into a supposedly conservative verdict |
| Verdict | `0.125 mm` | RS274 and Haas | determines safety, geometry, completion and efficiency eligibility using the declared rules |
| Convergence | `0.25 mm` | RS274 only | retained as reported convergence evidence, not a comparison of NC bytes across dialects |

Record exact NC/task/motion hashes, copied settings, conservative/refined or explicitly newer verdict rules, disagreements, raw quantities and coverage. Finer resolution cannot hide geometric/safety/completion disagreement or efficiency exclusion. Missing posted or refinement evidence is marked unavailable/blocked, not agreement or success. No current prototype dry-run is allowed to masquerade as posted-fidelity evidence.

Required native production controls and exact-NC settings are immutable: sampling, bounds, safety screen, compute bounds, max samples/station policy, expected NC hashes, original grid and operation-target exclusions. Only the declared role-specific step changes. Preserve immutable primary bytes and distinct auxiliary/refinement records; no overwriting a failing primary with a favorable later verdict. `max_step_mm` may be null only under the actual producer's documented null semantics, not as an accident.

## 6. D-07 budgets and actual resource estimates

| Budget/counter | Frozen value | Evidence/relationship |
|---|---|---|
| Candidate evaluations per non-nominal run | `<positive integer>` | `<integer exactly, not an optional hidden second engine budget>` |
| Nominal CAM candidate evaluation | exactly 1 | `<deterministic single recorded hard call; not optimizer generations>` |
| Population size | `<integer >= 2>` | `<CMA-specific positive population>` |
| Validation interval | `<positive integer>` | `<period in candidate evaluations>` |
| Maximum validation calls | `<nonnegative integer>` | `<max/min across the declared hard cap>` |
| Maximum generations | `<positive integer or explicit null>` | `<candidate budget still forces a bounded run>` |
| Per-child `timeout_s` | `<positive seconds>` | `<matched per-child timeout, not a guaranteed global wall deadline>` |
| Total wall-time budget | `<positive seconds>` | `<must cover declared serial worker-second/largest-job estimates>` |
| Run count | `<explicit integer>` | `<must equal sum of explicit primary cases and small-variant cases>` |
| Worker seconds | `<finite total or explicit null>` | `<sum of per-method (candidate+validation) estimates; unknown timing stays null>` |
| CPU RSS estimate per job | `<positive bytes, not above 8 GiB>` | `<estimate basis and uncertainty, not a measured or guaranteed peak>` |
| Parallel jobs | `<1 or 2, exactly 1 for GPU>` | `<D-07 bounded concurrency; current prototype primary scheduling/caching must be described>` |
| VRAM estimate per job | `<0 for CPU; positive, explicitly supported estimate for GPU>` | `<scope and hardware; no implied physical capacity>` |

Required cap: `max_rss_bytes_per_job = 8589934592` bytes (8 GiB), with fail-closed stop+optimize if exceeded. Do not silently raise this cap because an allocation peaks. Run-count, candidate calls and validation calls must reconcile with `planned_run_count`, native costs and actual execution records.

For actual posted/evaluator measurements, record own-process VmHWM and parent-polled child VmHWM per phase/dialect, including maximum over both readings and architecture/cache/parallelism/load. Do not use the withdrawn parent-process inherited-RSS artifact as the actual child's phase allocation. Estimates/hypotheses about shared CPU memory, device VRAM and parallel contention are not interchangeable. Report peak behavior, robust failures and blocked phases rather than deriving a passing allocation from an in-flight process.

| Cost distinction | Rows to include |
|---|---|
| Warm-up/JIT/candidate time versus steady-state | `<explicit accounting; warm-up is the first candidate call, not guaranteed pure compilation>` |
| Candidate compute time versus CNC cycle time | `<separate units and metrics>` |
| Primary evaluator versus refinement/secondary phases | `<separate per-role timings and measured memory>` |
| Per-run peak process RSS versus device-only or global pool | `<exact memory_scope; never claim peak process RSS is device-only memory>` |
| Unknown/unavailable resources | `<null and reason; never zero>` |

## 7. Comparisons and multiplicity

| Named comparison | Method | Baseline | Metrics/outcomes | Dialects | Expected task clusters |
|---|---|---|---|---|---|
| `<name>` | `<retained condition>` | `<different retained condition>` | `<declared, finite metric list>` | RS274 + Haas | `<selected tasks>` |
| `<optional selected variant comparison>` | `<condition>` | `<nominal or declared other baseline>` | `<...>` | RS274 + Haas | `<...>` |

- Include nominal CAM and its actual controls as anchors, not fabricated comparison winners.
- Keep negative/zero/significant differences and invalid candidates. No post-hoc best-of-seed ranking, curation of failures, or rule silently excluding an inconvenient baseline.
- Independent unit: **task**. Seeds are averaged within each task and method before pairing. Expected paired task IDs, not run counts, establish the effective independent sample.
- Name uncertainty method: two-sided paired sign-flip (exact through 16 independent tasks; seeded Monte Carlo above that, identity included), with raw p and Holm-adjusted p shown separately.
- Holm applies across every declared comparison, metric, outcome and family row. Declare confidence `<number strictly between 0 and 1>` and bootstrap replicates `<integer >= 100>`.
- Retain at-least-two-task CI semantics: a singleton task yields null CI with an explicit reason. More than one task is still descriptive when n < `<small-n threshold, normally 5>`.
- Binary observed outcomes only: record whether exact Clopper–Pearson is valid for one observed result per independent task; otherwise task-cluster bootstrap is required. Unknown/missing is not counted as true.
- For efficiency metrics, declare eligibility and positive removal/completion requirements separately. Unsupported fidelities and ineligible/no-cut evaluations are explicit exclusions, not zeros or eligible successful samples.

## 8. Tuning history and forbidden tuning

| Dev round | Conditions | Candidate evaluations | Changes | Dev evidence path + SHA-256 | Notes |
|---|---|---|---|---|---|
| `<round>` | `<registered conditions>` | `<positive integer>` | `<explicit changes before observing held-out outcomes>` | `<dev-scoped artifact and hash>` | `<purpose>` |
| `<additional round>` | `<...>` | `<...>` | `<...>` | `<...>` | `<...>` |

Tuning spend total equals the sum of rounds: `<exact integer or 0 only when rounds are empty>`. Every nonempty round needs linked dev-scoped artifact hashes and explicit changes; otherwise list full limitations. Repeat only identity-matching runs, preserve every failed/interrupted/blocked attempt, and avoid deriving a smaller total by concealing exploratory attempts.

Declare **no test/external tuning artifacts**. Path and declared-split checks must both pass; an `external/` path or declared test/external split cannot be normalized into a dev-fitting round. These mechanical checks are not a substitute for the coordinator's signed confirmation that held-out results were never used for tuning. No strict-improvement threshold determines inclusion or freezing.

## 9. Unavailable and nonphysical evidence

- CNC-Net executable-NC route: **unsupported**. Native reconstruction-only evidence does not create an executable NC comparison, faithful identical task mapping or method win. Describe all limitations and unsupported baseline coverage explicitly.
- CAMotics: **supplementary**, never the primary oracle or an automatically passing cross-check. Record missing/null/watertight/sweep-cap limitations. The declared independent NumPy reference has its own supported scope, not unrestricted CAD authority.
- Physical trials: `<present/declared independent evidence or unavailable>`; if unavailable, omit physical execution, safety qualification and machined-part claims. Simulation flags and downloadable simulation artifacts cannot replace real trials or a qualified operator. Preserve the separate pending-setup-review workflow.
- Policy-learning: `<present evidence or unavailable>`; if absent, omit policy-training, generalization and learned-controller claims. Physical exploration by the software is not implied.
- External CAD datasets: record source, actual STEP/B-rep/import verification, local-only assets and any unrecorded provenance/license. No undocumented STEP is redistributed or replaced by a mesh/box/analytic surrogate while claiming native evidence.

## 10. Scientific decision record and sign-off

| Check | Required coordinator evidence / decision |
|---|---|
| Scope and D-IDs reviewed | `<versioned amendment chain and links>` |
| Actual frozen manifest/config hashes verified | `<file hashes and validation output>` |
| Every selected condition has a runner-supported path | `<registered source, CLI, no prototype substitution>` |
| D-04/D-05/D-07 declarations honored | `<native precision/fidelity/resource records>` |
| No test/external tuning references | `<declared tuning record plus explicit attestations>` |
| Count/cost/Memory estimates reconcile or explicitly unknown | `<structured job/cost rows and limitations>` |
| Nothing revised in favor of a preferred algorithm | `<review notes and preserved contradictory cases>` |
| Optimization/evaluator execution evidence complete enough | `<not inferred from this tooling or previous pilot>` |
| Output directory fresh/authorized and default heavy jobs scheduled | `<paths, process/runtime and shared-load controls>` |
| Reproducibility records and any campaign rerun requirements | `<immutable configs, run IDs, task/NC hashes, outputs>` |
| Final scientific conclusion | `<accept/reject/narrow; uncertainty and excluded claims>` |

No checked box here means the work happened. Each checkbox above remains unchecked until the actual records exist and the coordinator signs.

### Exact commands actually executed (fill after execution)
```text
<exact validation/dry-run command, config/schema hashes, stdout/stderr and exit code>
<exact focused test command and exit code/results>
<exact ordinary full-suite command and exit code/results/skips/warnings>
<report paths and hash-list validation results>
```

Known current mechanism dependency: this project deliberately has no `jsonschema` dependency. Structural and semantic checking uses `scripts/validate_frozen_protocol.py` and the actual runner; it is not a hand-waived schema assertion. A future generic standards-validator is optional and needs an explicit dependency review. The optional prototype plan is still not an optimizer launch, validated numerical slope, certified physical result, or frozen scientific finding.
