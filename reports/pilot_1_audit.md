# Pilot 1 audit (T4.1) — core 9656e29, 2026-11-09/10

**Status: mechanically valid. Scientifically, it is a NEGATIVE result explained by two defects (D-09, D-10). It is retained as defect evidence, and the pilot is re-run after T3.xc and T3.xd.**

## Provenance and audit
- Launched from a clean detached worktree at core 9656e29 (pid 832300). Output: runs/pilot-pocket-20261109-t41b-pilot (+ `-refinement`, `.log`).
- Every run.json/config.json: `source_commit` 9656e29…, `dirty_patch_hash` null; simulator requested and runtime f64, x64.
- pilot_report.json: `auditor_passed` true, `evidence_complete` true, `all_runs_completed` true, 7/7 runs complete, 14/14 dialect results
  complete, no issues, `precision_evidence_complete` true.
- refinement-audit.json: 7 cases, 14 baseline+refined records, 28 complete dialect records, `auditor_passed` true.
- Verdicts are the 0.125 mm posted evaluation (D-04). Each optimizer had 4 candidate evaluations (pilot plumbing budget, not a scientific budget).
- Posted jobs: own-process peak 0.42–0.44 GiB (D-07 amended). About 28.6 min of posted work per program; total wall time about 4 h.

## Verdicts (0.125 mm, rs274 = haas in every row)
| run | geometry | safety | eff. eligible | residual | gouge | cycle time |
|---|---|---|---|---|---|---|
| nominal-cam s0 | **pass** | pass | yes | 0 | 0 | 231.525 s |
| fixed-adam s0 | fail | pass | no | 6.614% | 0.525% | 231.520 s |
| annealed-adam s0 | fail | pass | no | 11.041% | 0.496% | 231.515 s |
| hard-cmaes s0 | fail | pass | no | 0.008% | 1.747% | 231.528 s |
| hard-cmaes s1 | fail | pass | no | 1.170% | 0.804% | 231.522 s |
| smooth-cmaes s0 | fail | pass | no | 7.894% | 0.594% | 231.520 s |
| smooth-cmaes s1 | fail | pass | no | 1.477% | 0.616% | 231.526 s |
Optimizer geometry pass rate: **0/6**. No efficiency comparison is possible: no optimizer row is eligible, and cycle times differ by < 0.01 s.
The 1.0 mm screen agrees with the verdict on geometry in every row. Review E1 noted a completion flip on hard-cmaes s0, which does not
change the geometry verdict.

## Diagnosis
1. **The initialization is never a selection candidate (D-09).** Adam's first hard validation comes after a step, and CMA-ES never
   evaluates its mean (θ0 = the perfect CAM seed). Every arm returned a program that had moved away from the seed.
2. **The objective is geometry-only (D-10).** objectives.py forces every efficiency weight to 0 (a prototype leftover), so a perfect
   seed is the global minimum and nothing can improve on it. The unchanged cycle times confirm there is no efficiency signal.
3. **Not sim-to-post disagreement.** The selected candidates' 0.25 mm hard-sim metrics match the 0.125 mm verdict (for example,
   fixed-adam 6.614% / 0.525% in both).
4. Separate observation: at the seed, the smooth objective (k=40) = 0.645 while hard = 0. That soft-to-hard bias is T3.4's subject.
5. `air_fraction` is null in every posted evaluation, so the air-cut metric is not yet produced. T3.xd/T4.2 must close this before
   efficiency claims.

## Next
T3.xc (worker-1: init incumbent + feasibility-first selection), T3.xd (worker-2: efficiency terms) → merge → pilot re-run with a
declared efficiency weight, reporting selection-vs-verdict agreement per arm → T4.2 freeze (collapse defined relative to same-solver
CAM-init, typed any/all predicate).
