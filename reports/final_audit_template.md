# Final audit TEMPLATE — not a completed final audit

Use this only with the frozen campaign and immutable records. Replace every bracketed field with an audited source fact or an explicit unavailable/not-applicable state. Do not create `reports/final_audit.md` from this scaffold until the evidence review is real. Never overwrite user work under `manuscript/`; generated artifacts belong under `reports/paper_artifacts/<campaign-id>/`.

## 1. Campaign identity and immutable inputs

- Campaign ID: `<identifier>`
- Scope: `<frozen campaign / actual pilot / explicit smoke / synthetic>`
- Evidence cutoff and frozen configuration locations: `<source records>`
- Builder commit and builder source SHA: `<commit and sha256>`
- Builder output directory: `<reports/paper_artifacts/<campaign-id>>`
- `artifact_manifest.json` SHA-256: `<hash>`
- `claim_evidence.json` SHA-256: `<hash>`

| Input role | Logical record ID | Schema/revision | SHA-256 | Verification performed |
|---|---|---|---|---|
| Primary audit (`pilot_report.json`) | `<id>` | `<schema>` | `<sha256>` | `<result>` |
| Task/run manifest | `<id>` | `<schema>` | `<sha256>` | `<result>` |
| Primary per-run configs | `<list/count>` | `<schema>` | `<hashes>` | `<result>` |
| Per-run compute ledgers | `<list/count>` | `<source shape>` | `<hashes>` | `<linked or explicitly unavailable>` |
| Refinement links/comparisons/evaluations | `<list/count>` | `<schema>` | `<hashes>` | `<result>` |
| Refinement/auxiliary audit | `<id or unavailable>` | `<schema>` | `<hash>` | `<result>` |
| Decision record | `reports/decisions.md` | `<declared D-ID set>` | `<sha256>` | `<result>` |
| External baseline coverage declarations | `<id or unavailable>` | `<schema>` | `<sha256>` | `<result>` |
| Optional figure environment | `<if applicable>` | `<actual dependency versions>` | `<record>` | `<rendered or blocked>` |

Do not claim that a file hash alone establishes physical correctness, numerical precision or semantic agreement. Hashes bind the displayed claim to these exact records; each semantic check must be named independently.

## 2. Reconciliation and evidence completeness

- [ ] `evidence_complete` is true. Every expected run has a unique valid record, including valid failed/interrupted/blocked terminal records.
- [ ] `expected_vs_observed` has no missing, unexpected or duplicate IDs.
- [ ] Every run has a task hash matching its manifest, and every supported posted result has matching declared, actual and typed-evaluation NC hashes.
- [ ] Every run has an immutable config binding and a matching task/method/seed link.
- [ ] All required refinement links, refined typed evaluations, comparisons and auxiliary records resolve to existing byte-verified payloads. Missing optional bundles are explicitly unavailable, not silently passed.
- [ ] The report does not equate `auditor_passed` with `all_runs_completed`. Failures remain visible in tables and outcomes.
- [ ] No evidence from an unpublished/in-progress campaign is relabeled as a frozen campaign input.

## 3. Precision and decision policy matching

- Campaign `simulator_precision`: `<f32 or f64; final optimizer comparisons require f64 under D-2026-11-09-05>`
- Campaign decision IDs, in canonical sorted order: `<D identifiers>`

For each run/config binding:

| Run ID | Config SHA-256 | Declared precision | Decision IDs | Matches campaign? | Runtime/metadata evidence limits |
|---|---|---|---|---|---|
| `<run>` | `<hash>` | `<explicit value>` | `<list>` | `<yes/no>` | `<declaration versus independent numerical verification>` |

- [ ] Reject f32 rows in an f64 campaign, and any other mixed-precision condition unless a separately approved protocol explicitly defines it.
- [ ] Reject absent/ambiguous precision metadata rather than inferring it from a runtime default.
- [ ] Reject configs whose decision IDs differ from the campaign declaration. Verify every declared D-ID exists in the immutable decision record.
- [ ] Do not mistake metadata matching for proof of numerical accumulation correctness; cite the actual numerical regression evidence separately.
- [ ] Preserve the D-2026-11-09-03 historical provisional-gradient context and D-2026-11-09-05 full-float64 decision, including its qualification about the relevant implementation/regression state.
- [ ] Check any newer/refined evidence that supersedes earlier decisions. Do not automatically repeat a superseded provisional verdict.
- [ ] Pilot gradient conditions still marked provisional for an unresolved reason must not be described as unqualified completed evidence.

## 4. Claim-to-evidence audit

For every generated number and every narrative claim:

| Claim | Artifact/cell or text location | Run IDs | Task hashes | NC hashes (where applicable) | Source pointer(s) | Input SHA-256 |
|---|---|---|---|---|---|---|
| `<claim>` | `<artifact + row/cell>` | `<IDs>` | `<hashes>` | `<hashes>` | `<JSON pointer/record heading>` | `<hashes>` |

- [ ] `claim_evidence.json` covers every generated table cell, not only every row.
- [ ] Values printed with `NA` mean unknown/unavailable, not zero, false or success.
- [ ] Abbreviated LaTeX hash displays resolve to full hashes in the claim map and CSV.
- [ ] Original-part/operation-target distinctions, declared excluded volume and grid-resolved exclusion rules are retained wherever geometry is described.
- [ ] Evaluator roles remain distinct: soft, hard, primary posted, refined/conservative or explicit fine-verdict evidence. Screen and refined disagreements are not hidden by a composite flag.

## 5. Statistical units and uncertainty

- [ ] Independent unit is **task**, not seed, candidate call or solver generation.
- [ ] Seeds are averaged within each task/method before pairing.
- [ ] Paired task IDs and `paired_tasks` agree; replication of seeds does not shrink the task-cluster uncertainty estimate by inflating the independent sample size.
- [ ] Intervals are named by the audit method (task-cluster bootstrap or eligible exact binomial interval), including assumptions about binary observations.
- [ ] Report the two-sided paired sign-flip method and raw p separately from the Holm-adjusted p. Preserve extreme/total counts and randomization identities when present.
- [ ] Retain `small_n_warning` and `interval_null_reason`. A singleton task has a null paired CI; n < 5 is descriptive, not strong inference.
- [ ] Record confidence level and bootstrap/randomization seed/replicate counts from the audit policy.
- [ ] No confidence interval, error bar or success rate was invented from per-seed repetitions.

## 6. Results, included failures and scope-specific rows

| Run/task/dialect | Run status | Nominal CAM anchor? | Safety | Geometry | Completion | Efficiency eligibility | Removed volume | Residual | Surface error | NC hash |
|---|---|---|---|---|---|---|---|---|---|---|
| `<row>` | `<status>` | `<yes/no>` | `<state>` | `<state>` | `<state>` | `<state>` | `<value>` | `<value>` | `<value>` | `<hash>` |

- [ ] Include every failed/interrupted/blocked result, with valid recorded terminal status rather than fabricated geometry metrics.
- [ ] Include no-cut cases: on removal-required tasks, zero removal fails completion and cannot supply efficiency evidence.
- [ ] Include nominal CAM anchors as anchors, not automatic method wins or proof of a perfect zero-residual path.
- [ ] Keep negative differences, failed geometric comparisons and worse outcomes rather than plotting only favorable examples.
- [ ] Distinguish measured, simulated, reconstructed, calibrated and proxy values. Force-like metrics remain proxies unless their validation supports the statement.
- [ ] No simulator flag alone establishes physical machining safety or qualification.

### Conservative/refined outcomes

| Primary run | Primary values | Refined/reported values | Changed fields | Conservative or declared verdict rule | Efficiency implication | Link/evaluation/comparison hashes |
|---|---|---|---|---|---|---|
| `<run>` | `<values>` | `<values>` | `<fields>` | `<rule>` | `<implication>` | `<hashes>` |

- [ ] False-worst/unknown semantics and completion semantics are checked using the actual source rule; do not reinterpret a newer fine-verdict policy as an older conservative AND without naming the change.
- [ ] Refinement disagreement is explicit and excluded from efficiency where required. A finer completion improvement cannot override an actual conservative disagreement silently.
- [ ] Optional missing refinement evidence is recorded as `not_provided`, never as “no disagreement.”
- [ ] Report refinement metrics separately from cycle time, and record their evidence path and auxiliary timing rather than merging status with primary optimizer cost.

### External and unavailable coverage

| Route | Status | Supported subset | Unavailable/unsupported portion | Reason and source hash |
|---|---|---|---|---|
| CNC-Net executable NC | `<status>` | `<subset>` | `<unsupported>` | `<record>` |
| CAMotics | `<supplementary/status>` | `<subset>` | `<unsupported/null>` | `<record>` |
| Independent NumPy reference | `<status>` | `<subset>` | `<limits>` | `<record>` |
| Physical trials | `<status>` | `<subset>` | `<unavailable>` | `<record>` |
| Policy/training route | `<status>` | `<subset>` | `<unavailable>` | `<record>` |

Do not infer executable CNC-Net parity from native reconstruction-only evidence. Keep CAMotics supplementary under D-2026-11-09-01. Missing physical or policy evidence requires omission of the corresponding empirical claims, not a simulated replacement.

## 7. Resource accounting

| Run/stage | Candidate/forward/backward/validation calls | Candidate wall time | Other stage time | Peak process RSS | Hardware/runtime scope | Source hash |
|---|---|---|---|---|---|---|
| `<run/stage>` | `<counts>` | `<value>` | `<value>` | `<value>` | `<scope>` | `<hash>` |

- [ ] Warm-up is identified as the first candidate call, not a measured JIT-only duration.
- [ ] Candidate wall time, CNC cycle time, posted/refinement evaluation time and physical trial time are separate rows.
- [ ] Process peak RSS is not described as device-only memory. Partial failed-run cost is not fabricated as a complete budget.
- [ ] Missing ledgers are explicit `NA`; they never become zero time or zero cost in a figure.
- [ ] Host load, parallel workers and cache context are stated for any timing comparison. A one-run measurement is not a universal performance factor.
- [ ] Optional plots are separate single-axis figures or panels, never dual-axis charts. Their source tables remain available, and missing data stays `NA`.

## 8. Regeneration and validation

Record commands actually run from this repository/worktree, exit status, backend/runtime and results. Do not paste an intended command as a passing gate.

```
<exact focused command and result>
<exact full-suite command and result>
<exact clean builder regeneration command and result>
<exact byte-comparison result across all files>
<optional plotting environment, version and render/accessibility result, or explicit blocked status>
```

- [ ] Two separate fresh output directories from identical source bytes and the same campaign declaration have identical relative file sets and SHA-256 values for every generated file, including both manifests.
- [ ] No absolute host paths, wall-clock generation timestamps or unrelated git dirty-state values cause output drift.
- [ ] Empty/nonempty destination and `manuscript/` rejection behavior is tested without overwriting unrelated work.
- [ ] The default core environment has no hidden optional CAD, simulator or plotting dependency. Missing optional backends are blocked, not called successful.
- [ ] LaTeX requirements and layout review are recorded; source table bytes and source claims retain full precision/provenance independently of display formatting.
- [ ] The final conclusion identifies accepted, rejected and explicitly narrowed hypotheses. A negative or unfavorable result remains part of the audit.

## 9. Final claim disposition

`<pending actual evidence review: accepted / rejected / narrowed, with exact claim text and source cell mapping>`

This template supplies no conclusion, ranking, physical qualification or final-campaign approval of its own.
