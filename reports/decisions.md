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
D-08 addendum (review P2): three items must be pinned in the frozen protocol before any no-init run: (a) the `random` init
distribution (θ or delta space, scale and its reference, seed derivation); (b) the `straight_line` endpoints and segment placement;
(c) an operational `collapse` criterion (for example, the 0.125 mm verdict fails geometry or completion, or the residual exceeds
X × the CAM-init residual, with no hard-loss improvement over the run's own initial path). The T4.2a validator requires all three
whenever init ≠ cam and rejects placeholders in a frozen config. The values themselves are chosen at the T4.2 freeze.

## D-2026-11-10-09 — The initial program is a hard-scored selection candidate (pilot finding)
Pilot (runs/pilot-pocket-20261109-t41b-pilot): nominal CAM passes the 0.125 mm verdict with residual 0 and gouge 0, while all 5
optimizer arms fail geometry (residual 0.008–11%, gouge 0.5–1.75%). Their selected candidates' 0.25 mm hard-sim metrics match the
posted verdicts (for example fixed-adam: sim residual 6.614% / gouge 0.525% vs posted 6.614% / 0.525%), so this is NOT
sim-to-post disagreement.
Cause: no arm ever hard-scores its own initialization. Adam's first hard validation (`validate:1`) comes after a step, and CMA-ES
never evaluates its mean (θ=0 = the CAM seed). Selection therefore can only return a program that has already moved away from the
seed, here a perfect one. A second, separate observation: at the seed, the smooth objective (k=40) is 0.645 while the hard objective
is 0, a large soft-to-hard bias (T3.4's subject).
DECISION:
- Every optimizer arm hard-scores θ0 (its initialization) once, before any update. It is recorded as candidate `…:init`, counts as
  one candidate evaluation in the budget ledger, and is the initial incumbent for hard selection. The final program is the best
  hard-valid candidate including θ0; ties go to θ0. "No improvement over initialization" is a valid recorded outcome, not a failure.
- This applies identically to fixed and annealed Adam and to hard and smooth CMA-ES, and later to the D-08 no-init arms (where θ0 is
  the straight-line/random path).
- The current pilot stays as retained evidence of the defect. The pilot is re-run after the fix (same config apart from what the fix
  changes), before the T4.2 freeze. The questions about objective weighting, budgets and k (review Q1(c)) are deferred until the
  re-run shows whether optimizers can improve on CAM at all.

## D-2026-11-10-10 — The objective must carry efficiency; selection is feasibility-first (review R1/R2)
Finding (R1): algorithms/objectives.py:208-210 forces every non-geometry simulator weight to 0 (w_time, w_air_time, w_air, w_jerk,
…), and has since the prototype (a14a7d4). The hard objective is therefore 2·gouge² + residual² + 50·holder². A geometrically perfect
CAM seed sits at its global minimum, so no optimizer can improve on it. The pilot as configured cannot test the plan's stated goal
(experimental_plan §1.3/§5: simultaneous improvement of geometry, posted cycle time and air-cut fraction under zero-gouge and
holder-safety gates). This was a prototype leftover, not a design choice.
Finding (R2): selection compares the summed scalar objective, so a gouging candidate can beat a residual-only one.
DECISION:
- Selection is **feasibility-first (lexicographic)** for every arm, using hard scores: (1) safety (holder overlap = 0), (2) gouge
  within the task tolerance, (3) residual within the completion tolerance, then (4) the efficiency objective. An infeasible candidate
  never beats a feasible one; among infeasible candidates, lower total constraint violation wins. Combined with D-09, a feasible seed
  can only be replaced by a feasible candidate that is more efficient.
- The objective gains declared efficiency terms: cycle time (w_time, in seconds from the candidate MotionPlan's durations, which
  objectives.candidate_motion already recomputes) and air-cut time (w_air_time), in both the smooth objective (gradient-bearing,
  through the existing sim fields) and the hard objective (exact). Geometry and safety terms stay in the smooth objective as penalties.
- Weight values and normalisation (for example time relative to the seed's cycle time) are protocol values. They are chosen on the dev
  split within the declared tuning budget and frozen at T4.2, not hard-coded now; the defaults in code must be explicit config, not 0
  by fiat.
- Routing: worker-1's T3.xc gains the feasibility-first ranking (optimize.py/blackbox.py). Worker-2's new card T3.xd adds the efficiency
  terms (objectives.py), with configurable weights and hard/smooth parity tests. After both merge, the pilot is re-run with a declared
  efficiency weight before T4.2.
D-10 addendum (review S1/S2):
- S1: infeasible-candidate ranking uses a **declared normalisation**. Default: lexicographic within the infeasible set (holder violation
  first, then gouge, then residual), with an optional sum of each violation as a fraction of its tolerance selectable by config. The
  choice is a protocol value frozen at T4.2, not implicit in code.
- S2: selection feasibility is judged on the 0.25 mm hard sim, while the verdict is the 0.125 mm posted evaluation (D-04). Selection
  thresholds are declared as ≤ the verdict thresholds, with an optional declared margin. The margin value is frozen at T4.2.
  The pilot re-run reports selection-vs-verdict feasibility agreement per arm.
D-10 clarification (2026-11-10): the default `lexicographic` infeasible-set ranking (holder → gouge → residual → objective) supersedes
the earlier phrase "lower total constraint violation wins", which now applies only to the `tolerance_normalised_sum` mode. In that
mode a zero tolerance uses a recorded `zero_tolerance_scale` (default 1.0, i.e. the raw fraction), which is part of the config hash.
D-10 note (T1.xs, 2026-11-10): the legacy w_air penalty (compute_air_penalty, csg_simulator.py ~:1771) reads post-cut stock[t+1],
so it charges productive cutting as air. The efficiency objective therefore uses **w_air_time**, which is based on the pre-cut
seg_air/seg_engage volumes after the T1.xs sign fix. **w_air stays explicitly 0** until a separate fix (a stock[t] contract change)
lands.

## D-2026-11-10-11 — Efficiency time reference is a task constant; provisional pilot weights (review U1/U2)
- **Reference:** `reference_cycle_time_s` is a per-task constant, the posted cycle time of that task's nominal CAM MotionPlan
  (`seed_for_task(task)`). The runner computes it once per task and passes it explicitly to every arm (CAM-init and D-08 no-init
  alike), so the effective weights are identical across arms. The objective rejects `reference_cycle_time_s=None` whenever
  time_weight or air_time_weight > 0, except under an explicit test-only flag. The T4.2a validator requires the reference rule
  `task_nominal_cam_cycle_time` whenever an efficiency weight is > 0.
- **Provisional pilot weights (dev only, NOT protocol values):** time_weight = 0.01, air_time_weight = 0.01 (normalised by the task
  reference). Rationale: a 1% change in cycle time then moves the objective by about 1e-4, the same order as the geometry penalty at
  about 1% residual/gouge. Efficiency can therefore drive the search without overwhelming geometry, while selection stays
  feasibility-first (D-10). The final weights are tuned on dev within the declared budget and frozen at T4.2.
- **Routing:** T3.xd-wiring card (worker-1, right after T3.xc-runner merges). It covers runner/config wiring, the objectives.py
  None-rejection and the validator rule. The pilot re-run is blocked on T3.xc-runner and T3.xd-wiring.
D-11 addendum (review V1): the native smooth time/air terms are means over n = T − t_start segments (csg_simulator ~2075), while the
hard terms are totals over the reference. At equal declared weights, smooth arms would get about 1/T of the time pressure
(≈1/75 on the pilot path). objectives `_configure` sets sim.w_time = time_weight · n / reference and sim.w_air_time likewise, so the
smooth term is on the same total-time scale as the hard term. The factor is recorded in the normalisation metadata. Regression: on a
fixed path at high k, the smooth/hard time-term ratio is O(1) within a declared band for the documented native-vs-exact gap.
