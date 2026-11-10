# DiffCAM coordinator decision record

Decisions taken under detailed_plan.md §4/§5 (scope, protocol, stop/reassess). Newest last.

## D-2026-11-09-01 — CAMotics is supplementary; T1.0 reference is T1.3's independent cross-check
CAMotics 1.2.0 exports were non-watertight and its horizontal ConicSweep drops the start disc (T1.3a/T1.3c). The independent NumPy reference
(T1.0) serves as the required independent implementation for supported line fixtures; CAMotics geometry_pass is null for constant-Z cuts.

## D-2026-11-09-02 — Excluded-volume convention
operation_target.excluded_volume_mm3 is declared at the task grid resolution and checked with tolerance 0; continuous/native values are
diagnostics only (docs/contracts.md).

## D-2026-11-09-03 — Gradient rows provisional pending T3.3b
Systematic AD/FD mismatch (index 2: 4.1% ε-independent; index 35 sign flip). Pilot Adam rows are provisional; T3.4 ablations and any T4.2
optimizer freeze wait for T3.3b; Adam rows rerun if a backward bug is found.

## D-2026-11-09-04 — Posted-evaluation fidelity for the pilot (supersedes the 1.0 mm-primary + AND rule)
Evidence (hard-CMA smoke, first optimized path): residual 9.375 mm³ @1.0 mm vs 0.75 mm³ @0.125 mm (completion threshold 2.0 mm³);
gouge 346.86 vs 347.52 mm³. The 1.0 mm step's error is ~4× the completion threshold and of opposite signs for residual and gouge, so
AND-ing it into verdicts biases rather than bounds them. My earlier straight-wall scallop argument (0.064 mm) does not hold for optimized paths.
DECISION: verdicts (completion, safety, geometry, efficiency eligibility) come from the 0.125 mm posted evaluation for BOTH dialects.
The 1.0 mm evaluation is retained as a screen/fidelity diagnostic (reported per run, never combined into verdicts). Record a 0.25 mm RS274
point on every pilot program as convergence evidence for T4.2's cost/fidelity choice. Cost ≈ 18 min/program/dialect; parallel subprocesses
bounded by memory (~2 GB each). Reviewer: diffcam-discriminator E1.

## D-2026-11-09-05 — Simulator precision for comparisons: full float64
Evidence (T3.3b, worker-3; reviewer G2): the float32 ADJOINT is biased and the bias grows ~linearly with path length (0.24% at 8 steps →
4.1% at 75 on index 2); the float32 forward is accurate (f32 FD ≈ f64 FD). Full f64 fields+adjoints: AD −0.49917075 vs FD −0.49915996
(index 2), 112.5 s and 3.67 GB peak on the pilot probe vs 93 s / 1.94 GB in f32. Scalar-loss-only f64 (83 s / 2.0 GB) does NOT fix it.
Index-35 sign flip: unreproduced on the exact c955b60 replay (AD −0.00296 ≈ FD) — closed.
DECISION: all optimizer conditions in a matched comparison (CMA hard/smooth, fixed/annealed Adam, continuation) use full f64 so every
method sees the same numerical objective; f32 remains an option for exploration only. Implemented by T3.3c (declared `precision` option,
validated with a per-coordinate relative-error gate on the LONGEST suite path). The pilot declares precision f64 and launches after T3.3c
merges (avoids rerunning gradient rows); its Adam rows are then no longer provisional for this reason.

## D-2026-11-09-06 — Where run precision is recorded (no contract bump)
A RunRecord field would require a SCHEMA_VERSION bump, which changes every TaskSpec hash and invalidates retained bundles. DECISION:
config.json (required run artifact) metadata['simulator'] holds sim.precision_metadata(); benchmarks.audit requires it and fails closed on
missing or mixed precision within a campaign; the objective setup signature (`simulator_precision`) and Adam/CMA checkpoints carry it too;
T7.2 refuses reports whose run precision differs from the campaign's.

## D-2026-11-09-07 — Posted-evaluation memory cap for the pilot
The real-grid dry-run failed closed in the 1.0 mm screen phase: production NumPy posted evaluation peaked at 3,689,144,320 B (≈3.44 GiB)
per job, over the 2 GiB cap extrapolated from a small grid. Host: ~124 GiB total, ~45 GiB available. DECISION: max_rss_bytes_per_job = 8 GiB,
parallel_jobs = 2, geometry/sampling controls frozen; cap still fails closed; peaks measured per phase/dialect and reported. If a phase
exceeds 8 GiB, stop and open an evaluator memory-optimization card rather than raise the cap further.

### D-07 amendment (2026-11-09, T4.1b-L1) — premise corrected; cap retained
The 3,689,144,320 B "screen peak" behind D-07 was not the evaluator's memory. Posted workers recorded
`getrusage(RUSAGE_SELF).ru_maxrss`, which Linux carries across fork/execve, so every child reported the f64 optimizer parent's
peak (3,600,904 KiB for every role in the 553a7d2 dry-run). Fixed on task/T4.1b-L1-rss (e7c5b3b): workers record their own
VmHWM, the parent cross-checks its polled child VmHWM, and ru_maxrss is kept only as a labelled diagnostic.
Re-measured by re-running the 553a7d2 dry-run's five posted jobs as standalone workers (2 concurrent), on unchanged job specs:
| role / dialect | own VmHWM | wall | evaluation bytes vs 553a7d2 |
|---|---|---|---|
| screen 1.0 mm rs274 / haas | 0.418 / 0.418 GiB | 153 / 152 s | identical / identical |
| verdict 0.125 mm rs274 / haas | 0.416 / 0.416 GiB | 1235 / 1213 s | identical / identical |
| convergence 0.25 mm rs274 | 0.417 GiB | 505 s | identical |
(`/usr/bin/time -v` max RSS agrees with VmHWM for every job.) Evidence: /tmp/diffcam-l1-remeasure.0yLJ72.
DECISION: keep max_rss_bytes_per_job = 8 GiB and parallel_jobs = 2 unchanged. The cap is a fail-closed ceiling, not a tuned budget.
With real peaks at about 0.42 GiB it has roughly 19× headroom; changing it would mean re-gating the runner for no pilot benefit.
The "6–8 GiB review" flag cannot trigger at these peaks. The optimizer parent's own peak (~3.4–3.7 GiB, f64; D-05) is a separate,
host-level budget and is not compared against the per-job cap. A tighter cap (for example 2 GiB) can be revisited at the T4.2 freeze
if a leak-detection margin is wanted.

## D-2026-11-10-08 — Experiment 1 (initialization) stays in protocol scope
Review P1: the T4.2a schema (4daf0c9, configs/frozen_benchmark.schema.json:260) allows only the 5 CAM-initialised conditions, so it
could not declare Experiment 1, even though reports/experimental_plan.md §3 marks `No-Init + Smooth-Adam` / `No-Init + Hard-CEM`
**mandatory (Exp. 1)**, and the manuscript's insight (1) ("uninitialised optimisation collapses; CAM init required") rests on it.
The plan already requires it, so this is not a scope change. DECISION:
- The protocol gains an **initialization axis** that is independent of the solver: `init ∈ {cam, straight_line, random}`, with
  `cam` the default for every existing condition. Experiment 1 adds `straight_line|random × {smooth-adam, hard-cmaes}` on the same
  task, K, budgets and hard-validation rules as the CAM-initialised conditions. Smooth-Adam uses the annealed recipe (the plan's
  internal reference). `straight_line` is deterministic and gets one run; `random` gets one run per declared seed.
- **CEM is unavailable** (nothing in the codebase implements it). Hard CMA-ES stands as the zero-order representative, and claims
  are limited accordingly: "zero-order (CMA-ES)", not "CEM". Implementing CEM later would require a protocol amendment.
- No-init seeds must be valid MotionPlans for the task: the same segment count / K parameterisation, a safe rapid entry, and
  feed ≤ max. They are not required to be efficiency-eligible. Collapse or infeasibility is a valid recorded outcome, not a failure.
- Routing: worker-3 adds the init axis to the T4.2a schema/validator now (in T4.2a). After the pilot audit, worker-2 implements
  the runner conditions and the cam/seeds `straight_line`/`random` seeds (new card). These conditions are not part of the
  running 7-run pilot.
Override: the user may instead scope Experiment 1 and insight (1) out of the first paper; that would be recorded as a superseding decision.
