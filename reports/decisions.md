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
