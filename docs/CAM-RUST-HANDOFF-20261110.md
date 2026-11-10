# RustCNC strategy port — handoff to next agent (2026-11-10)

## Why this stops here

The user explicitly requested **wind down the workers, document their work, and let another agent take over**. The `/loop` fallback wakeup was cancelled. This is an incomplete implementation handoff, **not a successful strategy rollout, experiment completion, or physical machining qualification**. Do not infer endorsement from code being merged or from the later handoff commits.

**Wind-down verified:** workers 1, 2 and 3 each sent explicit idle confirmation and have directory-level handoffs. Their tracked worktrees are clean at the final tips documented below. Worker 1's failed raw-boundary experiment was preserved outside git rather than being committed as a candidate fix. No CAM-RUST worker test processes were found in the final path scan, and no new worker task, validation or campaign was started to reach this state. The coordinator's handoff monitor was stopped and the dynamic wakeup remains cancelled. Other sessions/projects running on this shared machine were not touched.

All three implementation workers were told to stop at safe boundaries, preserve exact checkpoints/diffs and untracked files, retain raw logs outside worktrees, report actual test outcomes (including interrupted/red runs), and then remain idle with no new cards or merges. Their reports must be consulted before any resumption:

- `~/src/cnc/dispatch/diffcam/CAM-RUST-1-diffcam-worker-1.reply.md`
- `~/src/cnc/dispatch/diffcam/CAM-RUST-1b-diffcam-worker-1.reply.md` (if created)
- `~/src/cnc/dispatch/diffcam/CAM-RUST-2-diffcam-worker-2.reply.md`
- `~/src/cnc/dispatch/diffcam/CAM-RUST-3-diffcam-worker-3.reply.md`
- `~/src/cnc/dispatch/diffcam/CAM-RUST-{1,2,3}-handoff-20261110.md`

Worktrees and branches are preserved, **not deleted by the wind-down**. Inspect each worker's final documentation to get its exact updated tip and any remaining dirty files; the commit ids below are earlier committed integration milestones, not a substitute for that final inventory.

## Directory map — where the next agent should look

### Current implementation and integration work

| Directory | What to find there |
|---|---|
| `/home/ntsoi/src/cnc/diffcam` | Coordinator-owned `core`: merged groundwork, current `cam/seeds.py` (still legacy), contracts, nominal evaluator, manifests/configs, ledger and this handoff. Do not treat this as a finished four-strategy system. |
| `/home/ntsoi/src/cnc/diffcam-wt/CAM-RUST-1-face-contour` | Worker-1 Face/Contour: restored clean at `ee560b6`, **still with the independent residual defect**. Do not mistake the clean status for a fix; the failed WIP patch and raw masks are under `dispatch/diffcam/CAM-RUST-1b-validation/`. |
| `/home/ntsoi/src/cnc/diffcam-wt/CAM-RUST-2-pocket` | Worker-2 Pocket 2D: clean WIP `c5f612abfbba2108e12250d3c515ebcaa0698679`. Source, tests and `tests/fixtures/rustcnc_pocket2d/suite_report.json` are committed. Only 6/13 construct; full suite/geometric qualification not run. |
| `/home/ntsoi/src/cnc/diffcam-wt/CAM-RUST-3-adaptive` | Worker-3 Adaptive 2D: clean WIP `9525b6364075875f58a24d7e126d0e85a6b9db90`. Source/tests committed, final test edits were never collected/run; cap and retained-pad blockers remain. |
| `/home/ntsoi/src/cnc/diffcam-wt/CAM-RUST-4-integration` | Coordinator's merged integration preparation: composition, mock seams, explicit-reference probe, with the Face branch merged for focused checks. `core` has those milestones now; do not accidentally treat unmerged future WIP here as authoritative. |
| `/home/ntsoi/src/cnc/dispatch/diffcam` | Per-worker cards/replies/handoffs, validation logs and ownership details. Check `CAM-RUST-1-validation/`, `CAM-RUST-4-validation/`, and any validation directories named by the final reports, not just the prose summaries. The lower-Face JSON records residual 61.078125 mm³. |

### Pinned inputs and other source/study trees

| Directory | How to use it |
|---|---|
| `/home/ntsoi/src/cnc/rustcnc-diffcam-port-5fc4717` | Coordinator-created **clean detached read-only** RustCNC snapshot at `5fc4717a`. Its `crates/cam-gen/src/{strategies,geometry}` and `docs/cleanroom/spec-cam` are the permitted pinned implementation inputs. |
| `/home/ntsoi/src/cnc/rustcnc` | Shared live RustCNC checkout with unrelated uncommitted Face/UI/other work. Never switch/commit/edit it or substitute its changing files for the pinned port input. |
| `/home/ntsoi/src/cnc/rustcnc-t863-adaptive2d` | Optional prior Adaptive-2D worktree, **not** the source used for these new Python ports. If consulted, treat it as a separate earlier investigation and verify its card/tip/evidence; do not import or overwrite it as ready code. |
| `/home/ntsoi/src/cnc/fusion_academic_understanding` / `fusion_exploration` / `fusion_cam` | Read-only study/reference material. Use only the scoped corpus/spec/posted evidence actually cited by the pinned Rust sources; these large trees are not live DiffCAM worktrees and do not need a wholesale audit for the current bounded task. |

### Preserved historical launch trees — not current implementation inputs

`~/src/cnc/diffcam-wt/campaign-launch` (old Exp2), `~/src/cnc/diffcam-wt/exp1-launch` (old Exp1), and `~/src/cnc/diffcam-wt/pilot2-launch` are detached versions used by stopped runs. `~/src/cnc/diffcam-wt/T4.3d-exp1b-config` is the suspended/later-cancelled Exp1b staging. Keep them for provenance/replay; do not resume or use them as the next task's code base. Their retained evidence lives under the corresponding `~/src/cnc/diffcam/runs/` roots named below, with `USER_STOPPED_20261110.json` sidecars.

## Immutable source and permitted port decision

The user chose a direct RustCNC port, these four strategies first: Face, Contour 2D, Pocket 2D, Adaptive 2D. They also chose replacement of the suite's single-raster starts, coarse chords, and a hard cap. Coordinator chose a complete-program default cap of **72 segments**.

Read `docs/rustcnc_strategy_port.md`, `tasks.md` CAM-RUST-1..4 and decision D-2026-11-10-16. Port input is the **read-only detached snapshot** `/home/ntsoi/src/cnc/rustcnc-diffcam-port-5fc4717`, commit `5fc4717a9342313c03461f80a19a79ad8996e757`. Source file hashes are in the port plan. The live `rustcnc` checkout contains unrelated uncommitted work, including Face, and must not be switched, committed, edited, or silently used as the port source.

Keep the `PROVENANCE(reference)`/`reference-copied` annotations. A second direct port does not create clean-room/MIT/publication status. The user may later supply an external CAM-init project (ledger `EXT-CAM-INIT`); it is not available here.

## Integration milestones on core

Relevant published core commits before this handoff's final status commits:

- `685efda`: stop decision, source snapshot pin, four cards, new protocol requirements.
- `f6c76af`: coordinator Pocket+Contour composition helper and synthetic seam tests.
- `08a7db3`: bounded Face/Contour port, composition and explicit-reference plumbing.
- `b4a51d1`: honest status including the real Face geometry failure and incomplete Pocket/Adaptive work.

`cam/rustcnc_face_contour.py`, `cam/rustcnc_seed_adapter.py`, and `scripts/qualify_rustcnc_seed.py` are present on core. **They do not yet replace the active `cam.seeds.seed_for_task`.**

### Worker 1 — Face + Contour 2D

Worktree: `~/src/cnc/diffcam-wt/CAM-RUST-1-face-contour`; branch `task/CAM-RUST-1-face-contour`.

- Earlier structural handoff: `ee560b631b79f4f61e281cb3515ff57d6c864fe6`, with 14 focused tests passing. The initial under-clearing 3R raster pitch was corrected to ≤2R and sourced `linkDirection=2` stay-down links; all 13 Facing programs constructed at 23–72 segments after that correction. This is not geometric qualification.
- Actual independent NumPy-reference evaluation of `procedural-facing_asymmetric-dev-case-00-offset_square_lower_step_pad` **fails: residual 61.078125 mm³ >2.0 mm³** (two dominant Y strips: 33.625 and 25.09375 mm³). Safety-pass is true, but original target gouge was not separately summarized and must be checked by takeover.
- Worker 1 restored its worktree cleanly to that original ee560b6 after preserving its experimental patch, inventories and raw mask evidence externally under `~/src/cnc/dispatch/diffcam/CAM-RUST-1b-validation/`. The clean worktree **still has the residual defect**. No source fix was committed.
- The archived `raw-boundary-WIP.patch` is **NOT a fix**: its 70-segment alternative had residual 0 but **129.578125 mm³ target gouge** in the retained rim. Do not cherry-pick/reapply that direction. Use `pre-fix-mask-diagnostics.json`, `pre-fix-residual-components.json`, `remaining-full.npz`, and the explicitly named valid raw-alternative JSON/NPZ files; some earlier failed/aborted remnants are explicitly marked non-evidence.
- Next physical diagnosis is Y-band reach/reserve/overhang and `lastCutDistance` against actual tool-centre sweeps, preserving rim/pad clearance, ≤2R and cap 72. Typed cap success and safety_pass alone cannot clear residual/gouge.
- Contour retains its lower-level append emitter and actual wall/lead checks; legitimate lead-fit refusals stay explicit.
- The worker's full default suite had 856 passing, 11 skipped, and **one shared f32 precision failure**. The coordinator reproduced the exact `-0.3401482105255127` baseline failure at `TI_CPU_MAX_NUM_THREADS=4` in 70.41s. Do not claim a green default suite, and do not silently change this unrelated precision fixture.

The full raw handoff is `~/src/cnc/dispatch/diffcam/CAM-RUST-1-handoff-20261110.md`; worker 1 explicitly confirmed it was idle.

### Worker 2 — Pocket 2D

Worktree: `~/src/cnc/diffcam-wt/CAM-RUST-2-pocket`; branch `task/CAM-RUST-2-pocket`.

The wind-down report is present at `~/src/cnc/dispatch/diffcam/CAM-RUST-2-diffcam-worker-2.reply.md`, with companion `CAM-RUST-2-handoff-20261110.md` and evidence directory `CAM-RUST-2-handoff-evidence-20261110/`.

- Exact clean WIP tip: **`c5f612abfbba2108e12250d3c515ebcaa0698679`**. Owned implementation, test and `tests/fixtures/rustcnc_pocket2d/suite_report.json` are committed; tracked worktree was clean at the report. Earlier checkpoint `0c5b6e1` is history.
- Focused suite: **8 passed, 1 warning in 1.24s**. One targeted Rust reference test passed with a separately configured `/tmp` build target. **The full default CPU pytest suite was not run before wind-down**, and no new run is initiated by this handoff.
- Bounded constructor outcome: **6 constructed / 7 explicit cap refusals across 13** pocket-island tasks. Constructed counts: centred full-height 2, lower-step 3, open 1. Refusal segment counts are 84, 112, 116, 124×2, 140 and 156, all retained in the fixture/report. This is neither qualified geometry nor permission to delete unfinished passes.
- Retained-pad/cap support, named roughing/wall-finish ranges, coarse arc-to-line conversion, explicit safe lift/reposition routes and cap errors are implemented. Retained-pad operation is explicitly a local extension beyond Rust source's island rejection. The next agent should validate these against actual masks/posts and combine with Contour finishing only after review.

Start from `c5f612ab...` for Pocket work; do not assume the checkpoint is merge- or qualification-ready.

### Worker 3 — Adaptive 2D

Worktree: `~/src/cnc/diffcam-wt/CAM-RUST-3-adaptive`; branch `task/CAM-RUST-3-adaptive`.

- Exact clean WIP tip: **`9525b6364075875f58a24d7e126d0e85a6b9db90`**, after first checkpoint `360657f`. Its owned `cam/rustcnc_adaptive2d.py` and `tests/rustcnc_adaptive2d_test.py` are committed and the tracked worktree was clean at handoff. Earlier interim/uncommitted states are history, not a completed port.
- Implemented bounded erosion/regions, entry helix/circle, body levels, direct-or-retract links, distinct lower-pad cap/body phases, coarse chords and explicit cap/refusal/provenance. Local typed-stream connectors and retained-pad chord screening are consequential seams that still need independent validation.
- **No final default suite ran; no final focused-green claim is allowed.** The last focused run before later seam/test edits was 11 passed / 3 failed. The final saved test file declares 18 test functions and the final edits **were never collected or run**. Focused output: `/tmp/diffcam-CAM-RUST-3-worker3-focused.log` (overwritten across iterations).
- One final targeted open-case probe returned 63 structurally valid segments, not geometric qualification. Cap limits and physical retained-pad intersections still block most examples: a last targeted table preserved requires 109/181 in two cases and retained-pad chord intersections in others. An earlier full 13 sweep is explicitly historical, before the final connector change, not a final-tip result. Default source-oriented open cases previously required 809/1091 segments in samples; source/coarse recipe is not silently replaced default.
- Source raster/frontier stand-ins are not recovered producer-CAM wave/A* behavior or proof of safe complete stock removal. Retained-pad entry and full-circle/lead-in failures, current residual/frontier diagnostics, source-path/chunk semantics and a fresh all-13 table at final HEAD remain next work.
- Full evidence and exact recipe/probe distinctions: `~/src/cnc/dispatch/diffcam/CAM-RUST-3-handoff-20261110.md`. Worker 3 explicitly confirmed idle; no owned test/process was running and no old suite was started merely for wind-down.

## What has actually been checked

- Coordinator integration of the Face port plus mock-only composition and explicit source-factory checks: **19 focused tests passed**.
- A test-double cache issue in `tests/rustcnc_seed_adapter_test.py` was corrected by patching both `sys.modules` and the parent `cam` package attributes. The tests are intentionally mocks, not CAM algorithms or independent geometry qualification.
- Default contour composition matches the ported wrapper's explicit thin-gap profile (R/2 horizontal, R/4 vertical), not the lower-level emitter's different R/R defaults. The helper identifies which deepest native wall passes are replaced; no region/depth is silently skipped.
- `scripts/qualify_rustcnc_seed.py --task-id <exact-id> --family <name> --output <new-json>` is an independent-reference probe. It calls the existing reference mask/safety checks with an explicit seed factory rather than global monkey-patching. No G-code posting, optimizer call or equipment execution is implied.

## Old stopped experiments — do not resume as completion

Experiment 2, Experiment 1 and Pilot 2 refinement were stopped at the user's request; the isolated process groups were verified gone afterwards. Exp1b was never launched. Non-complete entries must remain stopped/incomplete and must not vanish from an aggregate or be relabelled as failures/successes.

| Output | Stop-shutdown state |
|---|---|
| `runs/campaign-exp2-20261110-run` | 233/312 manifest-complete; 79 non-complete |
| `runs/campaign-exp1-20261110-run` | 41/72 manifest-complete; 31 non-complete |
| `runs/pilot-pocket-20261110-pilot2_init` | all 14 primary runs already complete; refinement stopped partway |

The original manifests remain unchanged. Each root has a `USER_STOPPED_20261110.json` sidecar with its source-manifest SHA-256, run ids and semantics. Old records are old-single-raster evidence, not final replacement results. The pending T6.1b final packet export and any final T7.2 table regeneration were explicitly cancelled/deferred for that old incomplete cohort. There was no physical equipment execution.

## Resume procedure

1. Read every final worker handoff and inspect `git status --short --branch`, `git log`, source paths, and raw evidence. Preserve dirty work and untracked fixtures. Verify which files were actually committed before checkout/rebase. Use fresh worktrees off the actual required tips; do not damage the owners' preserved checkpoints or the read-only RustCNC source snapshot.
2. Fix and **independently qualify** the lower-Face adapter/cap/floor motion. Inspect residual masks to distinguish physical-boundary versus tool-centre mistakes; do not simply relax thresholds, exclude real material or reuse the legacy raster.
3. Finish Pocket 2D and Adaptive 2D bounded rectangular support with complete retained-pad/cap phases, safe entry/link/retract motions, and total ≤72 segments. A cap/refusal is valid; dropping loops, levels or material to fit is not. Preserve Adaptive's unpinned source stand-ins explicitly rather than claiming full Fusion parity.
4. Only then wire nominal dispatch, compose Pocket roughing with Contour finishing without discontinuity or double finish, and check every claimed task's identity/cap/contiguity. Use real stock/target geometric checks, not only `MotionPlan.validate_for_task`, and both G-code dialects for the claimed integration cohort. Generation count alone is not qualification.
5. Preserve the historical old manifest and mixes of bootstrap records. Add a new manifest/config identity carrying the RustCNC source, coarse-chord, segment cap and seed-generation provenance. Runner `_SUITE_MANIFEST_RELATIVE` and `_SUITE_MANIFEST_SHA256` currently pin the old 39-task manifest; updating `cam.seeds` alone will create identity failures. Do not mutate the old frozen D-14 config to pretend these programs are the same experiment.
6. Run a new dev integration pilot first, record a new protocol amendment and **fresh output roots**, then a bounded replacement campaign. Report signed absolute and relative cycle changes prominently, retain no-improvement/failed/blocked cases, and rebuild final packets/results only from complete new evidence. Physical setup/machine data still needs a real qualified operator; software packets never authorize cutting.
7. The earlier requested Slack MCP verifier handoff applies only after complete implementation **and actual replacement experiments**, not to this paused state. It still needs a human/next-agent decision about continuing the work.

## Repository / process hygiene

DiffCAM integration branch is `core`; never commit into the shared RustCNC checkout. Use explicit scoped `git add <paths>`, keep task worktrees, and push `core` after accepted coordinator merges under the existing user preference. New worktrees need `.env` and per-agent Taichi/uv caches. CPU is the default for tests; serialize timing/GPU runs.

No process/group-pattern kills. Preserve evidence before stopping any owned test, use exact owned task ids or verified pids, and record interrupted tests. Do not start old campaigns, new campaigns or REAL_* tests implicitly as part of wind-down. Another agent now owns the next implementation/testing decision.
