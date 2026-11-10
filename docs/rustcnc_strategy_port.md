# RustCNC strategy port — active implementation plan

Status: **implementation and qualification in progress; the ported strategies are not activated** (2026-11-10).
Authority: `tasks.md` CAM-RUST-1..4, the per-worker cards in `~/src/cnc/dispatch/diffcam/`, and decision D-2026-11-10-16 in `reports/decisions.md`.

## Current integration state

- Worker-1's bounded Face/Contour implementation (`ee560b6`) is merged. Its 14 focused tests pass and all 13 Facing programs construct within 23–72 segments after replacing the under-clearing 3R pitch with bounded 2R pitch and sourced stay-down links.
- A real independent-reference check of `procedural-facing_asymmetric-dev-case-00-offset_square_lower_step_pad` is **NOT qualified: residual 61.078125 mm³ >2.0 mm³**. CAM-RUST-1b is assigned to correct the physical adapter. Cap/contract checks are not qualification.
- Worker-1's full default run was 856 passed, 11 skipped, 1 failed. The standalone f32 precision failure (`-0.3401482105255127`) was reproduced on the integration base at `TI_CPU_MAX_NUM_THREADS=4` (70.41s), so it is shared/base redness. The full suite is not green.
- Coordinator composition/qualification seams contribute four mock-only tests plus one explicit seed-factory test; integrated focused checks across them and Face/Contour are 19 passing. Independent-reference coverage remains incomplete.
- Pocket 2D has a checkpoint but no final handoff. Adaptive 2D is interim only: it is fixing large cap overruns, retained-region clearing and its serialized motion continuity. Neither is activated or merged into `core`.
- No nominal dispatch, registered manifest, frozen protocol or old experiment is being silently replaced by this partial code. The old stopped cohorts remain historical and all replacement experiments await the remaining qualified ports.

## User decisions and scope

The user redirected the coordinator and workers to implement trajectory-generation strategies from RustCNC, itself based on `fusion_academic_understanding`, `fusion_exploration`, and `fusion_cam`. The explicit choices were:

1. Directly port RustCNC code, not a parallel clean-room design.
2. Face, Contour 2D, Pocket 2D, and Adaptive 2D first.
3. Replace the benchmark suite's existing single-raster starting programs.
4. Use coarse chord sampling and a hard segment cap.

The coordinator's default cap is **72 segments per complete program**, matching the old suite's measured 22–62 segments and remaining below the optimizer's established roughly-75-segment regime. A cap is not a coverage licence: truncate no contour, level, region, or retained-pad cap to fit. A path that cannot fit is an explicit construction failure. There is no fallback to the old raster under a ported strategy's name.

This first adapter targets the current analytic rectangular outer pockets with zero/one rectangular retained pad. Unsupported geometry and unported options fail closed. These are CAM starting programs for numerical optimization, not physical machining authorization, not complete Fusion 360 parity, and not proof of constant engagement. RustCNC records unpinned stand-ins especially in Adaptive 2D; the port must retain those limitations.

## Immutable RustCNC inputs

The port reads only the coordinator-created detached checkout:

`/home/ntsoi/src/cnc/rustcnc-diffcam-port-5fc4717`

Commit: `5fc4717a9342313c03461f80a19a79ad8996e757`.

The live shared RustCNC checkout has unrelated uncommitted changes, including `face.rs`. Those changes are not silently included, and that checkout must not be switched, committed, or edited by this work.

| Module | RustCNC source | SHA-256 of source bytes |
|---|---|---|
| Face | `crates/cam-gen/src/strategies/face.rs` | `8c0ff95313ffb141eedd9f674caeca82cc8533b089b11fd481a1fe7a7891f99e` |
| Contour 2D | `crates/cam-gen/src/strategies/contour2d.rs` | `333fa825ce2cdb4570f387d935b344b17be016f630fb0204fd38a8f92a2ccb75` |
| Pocket 2D | `crates/cam-gen/src/strategies/pocket2d.rs` | `1ae9e7aac8bda99aba15c587ce4e1318ed52cfc2bad84e3e334537fef4f7f038` |
| Adaptive 2D | `crates/cam-gen/src/strategies/adaptive2d.rs` | `a34b58700c8f8ee4ae665831988a1385ee62ea8a79e36969309710946384e535` |

Preserve the source algorithms, constants, intermediate order and function decomposition (Python `snake_case`). Cite Rust file/line ranges above ported functions and identify deliberate parameter/coarse-chord deviations. Every ported file starts with a `PROVENANCE(reference):` header naming the above commit and relevant spec-cam sections. A second port does not make the code clean-room, MIT, or free of the source's publication restrictions.

Every returned `MotionPlan` additionally records the strategy id, source identity, parameter recipe, phase/segment counts, chord rule, hard cap and named approximations in its provenance.

## Work allocation and interfaces

| Task | Worker | Exclusive implementation files | Role |
|---|---|---|---|
| CAM-RUST-1 | diffcam-worker-1 | `cam/rustcnc_face_contour.py`, own test | One-way facing band/depth logic, Contour 2D offsets/leads, reusable finish emitter |
| CAM-RUST-2 | diffcam-worker-2 | `cam/rustcnc_pocket2d.py`, own test | Pocket offset ladder/entry/link roughing and explicit retained-pad handling |
| CAM-RUST-3 | diffcam-worker-3 | `cam/rustcnc_adaptive2d.py`, own test | Adaptive erosion/levels/entry/frontier/linking and coarse-helix recipe |
| CAM-RUST-4 | coordinator | seed dispatch, suite/manifest/config and integration tests | Compose contour finish, replace nominal starts, regenerate evidence, freeze and rerun |

Worker APIs are `face_seed`, `contour2d_seed`, `pocket2d_seed`, and `adaptive2d_seed`, accepting a typed `TaskSpec`, optional `IslandSeedSpec`, and `max_segments=72`. Lower-level loop emitters are available where the coordinator needs to compose a finishing pass without re-running a complete approach. Workers do not edit the seed dispatcher, suite, campaign configuration, shared contracts or ledger.

The intended dispatch is: facing geometry → sourced Face path plus required retained-pad phases; pocket/island geometry → sourced Pocket 2D roughing plus explicit Contour 2D wall finishing; variable width/depth → sourced Adaptive 2D clearing. Actual per-topology support and composition are integration findings, not promises made ahead of the implementations. A strategy may not be routed on optimizer performance, and unsupported existing tasks may not silently disappear from the declared cohort.

## Acceptance and scientific identity

Each card requires source-derived tests first, focused tests and one full default CPU suite on the worker's final tip. No REAL_* opt-in or campaign run is needed per card. Before replacing the suite, the coordinator must:

- review diffs and run the focused ports and full merged suite;
- construct deterministic, contiguous machine-frame plans with task-limit feeds and safe entry/retract on every claimed supported task;
- verify segment caps and prove that over-cap cases raise rather than skip machining;
- test both full-height and lower retained pads and prove that any lower cap is explicitly cleared before lower floor cutting;
- run nominal geometric qualification with actual stock/target evaluation and both G-code dialects on the integration cohort, not treat `validate_for_task` as qualification;
- record rejected/incomplete tasks and coverage separately from optimizer outcomes;
- write a new seed-generation/protocol amendment and fresh campaign configs/output roots, then run a new dev integration pilot before declaring a campaign.

The old task body's frozen `t2.3/2` geometry is not rewritten to rescue a path. A seed amendment changes nominal motion and its hashes, run ids, configuration identity and protocol identity. If it is deliberately kept identical at task level, the seed-generation and coarse-chord/cap identities are separately explicit and bound in the new manifest and configs.

## Stopped evidence and top-level tasks

At the user's direction, Experiment 2, Experiment 1, and Pilot 2 refinement were stopped on 2026-11-10. `USER_STOPPED_20261110.json` sidecars preserve the source-manifest SHA-256, counts and unfinished run ids without altering the original manifests:

- Exp2: 233/312 entries manifest-complete at shutdown; 79 non-complete.
- Exp1: 41/72 manifest-complete; 31 non-complete. Exploratory Exp1b does not launch, despite the already-recorded D-13 reach trigger.
- Pilot 2: 14/14 primary runs already complete; refinement checking stopped partway through.

The completed records remain old-start, old-protocol evidence. Non-complete entries are not negative results and cannot vanish from reconciliation. Old partial campaigns cannot supply a final green results table, physical trial rerun, or publication claim after the new starts replace their nominal programs.

Top-level status follows this sequence: **ported modules → integrated nominal seeds → new dev pilot → amended protocol → fresh experiments → regenerated results and physical-review packets**. T4.2 therefore needs a new amendment; T4.3 awaits that amendment; T6.1 and T7.2 rebuild only from the resulting complete dataset. Physical setup review remains a separate operator activity. After the implementation and replacement experiments finish, use the requested Slack MCP handoff to a verification agent, including the scientific and provenance limitations.
