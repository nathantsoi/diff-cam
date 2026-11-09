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
