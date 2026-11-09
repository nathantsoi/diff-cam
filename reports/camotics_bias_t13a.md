# CAMotics v1.2.0 independent-route and bias study (T1.3a)

## Status and scope

This is a **prototype external geometry cross-check, not a machine-safety certificate or a calibrated production oracle**. It supports the declared analytic box stock/target forms, a fixed flat cylindrical tool, known initial state, and a pure work-to-machine translation. Rotations and explicit G53 absolute-machine motion fail closed. Arbitrary meshes, STEP, fixtures, ball cutters, and non-G54 work-offset setups are unsupported. `safety_pass` is null; shank/holder widening, whole-cell safety bounds, assembly clearance and physical acceleration behavior are not inferred from STL membership. They belong to the production evaluator's separate guards.

The real route is the already qualified upstream binary Docker image `diffcam-camotics:1.2.0`. The adapter additionally refuses any image whose content ID differs from `sha256:e680bff023d9190ed16fe1bee21d9c3c72f0e35bfbceb0a19e704c62f6ddca80` or whose reported version differs from `1.2`; changing the tag alone is not accepted as compatible. The documented upstream release package has SHA-256 `5dc5ba0c53a8eb8ce6b3bd2c3bd903465fde860742a7764f2511b188a507c2c0`. Matching public source tag `v1.2.0` is revision `a26af50604877e3c863091103b6667b207fa79ad`; this is inspection/reference evidence, not a claim that the local unrelated checkout or a source build reproduced the binary. The current invocation records the Docker image ID, both `camsim --version` streams, command, source and derived NC SHA-256 values, project SHA-256, output topology and elapsed time.

## Root cause of the original zero-removal result

The initial canonical run supplied CAMotics' project with stock in physical machine coordinates `(100,80,-150)` plus the stock offset. `post_motion_plan()` correctly posted its machine motion in **G54 work coordinates**, e.g. `X-4 Y-4 Z-2`. No G54 offset was initialized in the external interpreter. Consequently it cut near origin, outside that physically translated stock. The `Tool Length Offset not supported` warning did not explain the zero removal; both state and geometry needed explicit treatment.

The typed parser also establishes that G43 Hn/G49 declares a calibrated cutter-tip convention, not a license to infer a physical H-table from `ToolAssembly.flute_length_mm`. The packaged simulator loads per-tool registered offsets, whose X/Y/Z offsets here are zero, and separately warns that general tool-length offsets are unsupported. Evaluations preserve this warning and document the restricted zero-register-vector interpretation; real H-table provisioning remains external and unverified.

## Calibrated coordinate convention

Let `T` be the pure work-to-machine translation, `C` the known machine-frame initial position, and `P` a caller-programmed G54 coordinate. The whole CAMotics computation uses

```
Q = machine_position_mm - C = P + T - C.
```

CAMotics starts at `Q=0`, exactly the already declared initial state, so no synthetic origin-to-stocking move or proxy cutter is needed. The project stock box and every mesh query point use this same Q frame. The task grid is still sampled in its original work frame, transformed to machine coordinates, then shifted by `C`.

The packaged v1.2.0 executable was probed directly, rather than guessing the convention from another controller. A G10 example with `X100 Y80 Z-150` produced machine `(-104,-84,160)` from original `(-4,-4,10)`, while `X-100 Y-80 Z150` produced `(96,76,-140)` and the next `G1 Z-2` produced `Z-152`. These are the intended machine positions for the original canonical task. The adapter therefore writes

```
G21 G90
G10 L2 P1 (C.x-T.x) (C.y-T.y) (C.z-T.z)
T(task.tool.tool_number) M06
```

after a setup comment and before the **unchanged caller source NC**. These four setup lines only establish the simulator frame/tool; they do not rewrite any caller move. `source_program.nc` is exact input; `program.nc` contains prefix + exact input and is the actual executed program. Their distinct hashes are retained. This G10 behavior is release-specific measured evidence, not portable G-code advice. Raw absolute G53 would not undergo the initial-position conversion, so this prototype rejects it rather than simulate it incorrectly.

A real canonical pocket smoke after this correction returned a sampled removed volume of `244.045630859375 mm^3`, rather than zero. That run is only an execution/nonzero-removal smoke: that run had `7210` exact-vertex edge-incidence defects, `coverage=incomplete`, `geometry_pass=null`, and `safety_pass=null`. These are not passing machining-quality results. The raw historical result is `/tmp/diffcam-t13-calibrated-result.json`, SHA-256 `76b338c8819050fd9ab0792d68c96e0bc43bd572a50d5bca9e6ac5b06b4b87c4`.

That historical canonical smoke predates the rapid guard. The parsed canonical retract's cutter AABB can overlap stock, so the current adapter returns `Coverage.UNSUPPORTED` rather than presenting its removal as valid feed-only reference geometry. Routine escape/retraction moves therefore remain a substantive scope limitation, not a solved compatibility issue.

## Independent fixtures and comparison definition

Five line fixtures are used; no arc is silently converted into a reference-feed segment. CAMotics itself linearizes arcs, and the canonical arc smoke is not promoted to a continuous-analytic arc study. The independent reference implementations' continuous expectations are imported as constants from `verification.fixtures`, never fitted to camsim:

- `plunge_above`: `pi*3^2*4`;
- `horizontal_slot`: `12*(2*3*24 + pi*3^2)`;
- `sloped_ramp`: `12*pi*2^2 + 2*2*24*(26/3)`;
- `near_vertical`: `12*pi*3^2 + 2*3*sqrt(0.73)*3`;
- `repeated_plunge`: `36*pi`, the union of repeated up/down plunges, not a sum of overlapping removals.

All five use 60 x 40 x 12 mm stock, nominal 28,800 mm^3, and a cell-centered task grid with shape 30 x 20 x 6, voxel size 2 mm. The external CAMotics resolutions are **0.25 and 0.1 mm**. The task grid is deliberately unchanged across these two external resolutions. Reported `removed_volume_mm3` is independent ray-parity occupancy on that grid, not the signed triangle sum of the full STL. Signed STL sum, exact-vertex edge-incidence count, and triangle count are diagnostics. A non-watertight raw output does not prove a closed solid and cannot receive a passing geometry result, even if sampled occupancy is usable diagnostically.

The task fixture originally describes a work-frame state; the study converts that state to the machine frame with its existing transform before the strict typed-NC path. Its transform is the identity, so physical coordinates and analytic removals are unchanged. Haas source NC is generated by the already reviewed typed post and uses fixed tool/register 1. Output directories separately record the task JSON, input motion, source/derived NC files, project, STL, command/stdout/stderr, image identity and full `EvaluationResult`.

## Measured volume comparison

Rows marked `2.0/0.5 mm grid` refer to the independent task-sample grid, not the external CAMotics resolution. A single execution was recorded per cell; no uncertainty or repeatability distribution is claimed. `error = sampled grid removal - independent continuous removal`. `STL edges` is the exact-vertex boundary/nonmanifold-edge incidence count, not a repair estimate.

| Fixture | grid [mm] | CAMotics [mm] | expected removal [mm³] | sampled removal [mm³] | error [mm³] | STL watertight / edge defects | geometry pass |
|---|---:|---:|---:|---:|---:|---|---|
| plunge_above | 2.0 | 0.25 | 113.097336 | 64.000000 | -49.097336 | False / 23878 | None |
| plunge_above | 2.0 | 0.1 | 113.097336 | 64.000000 | -49.097336 | False / 62942 | None |
| horizontal_slot | 2.0 | 0.25 | 2067.292007 | 2304.000000 | 236.707993 | False / 24582 | None |
| horizontal_slot | 2.0 | 0.1 | 2067.292007 | 2016.000000 | -51.292007 | False / 65327 | None |
| sloped_ramp | 2.0 | 0.25 | 982.796447 | 960.000000 | -22.796447 | False / 24446 | None |
| sloped_ramp | 2.0 | 0.1 | 982.796447 | 960.000000 | -22.796447 | False / 65011 | None |
| near_vertical | 2.0 | 0.25 | 354.671213 | 336.000000 | -18.671213 | False / 24126 | None |
| near_vertical | 2.0 | 0.1 | 354.671213 | 336.000000 | -18.671213 | False / 63586 | None |
| repeated_plunge | 2.0 | 0.25 | 113.097336 | 64.000000 | -49.097336 | False / 23878 | None |
| repeated_plunge | 2.0 | 0.1 | 113.097336 | 64.000000 | -49.097336 | False / 62942 | None |
| plunge_above | 0.5 | 0.25 | 113.097336 | 112.000000 | -1.097336 | False / 23878 | None |
| plunge_above | 0.5 | 0.1 | 113.097336 | 112.000000 | -1.097336 | False / 62942 | None |
| horizontal_slot | 0.5 | 0.25 | 2067.292007 | 1728.000000 | -339.292007 | False / 24582 | None |
| horizontal_slot | 0.5 | 0.1 | 2067.292007 | 1728.000000 | -339.292007 | False / 65327 | None |

### Resolution interpretation

At the fixed 2 mm task grid, **4 of 5 fixtures retained exactly the same cell-count removal** across external 0.25 to 0.1 mm: plunge_above, sloped_ramp, near_vertical, repeated_plunge.
One had a smaller scalar volume error: horizontal_slot (absolute error 51.292007 vs 236.707993 mm³). This error reduction also changes the sign, so it is not evidence for a universal additive correction.
Horizontal-slot removal changed by -288 mm³ even though its absolute scalar error shrank, demonstrating that changing the external surface can change sampled cells significantly while the coarse grid masks other changes. The other unchanged pairs must not be reported as a demonstrated external-resolution convergence result.

### Refinement control at 0.5 mm task grid

Four extra real runs used the same fixtures but a disclosed grid of shape 120 x 80 x 24 (230,400 points). Each case records its own task grid; the continuous expectations were not re-fitted to the results.

- `plunge_above`, CAMotics `0.25` mm, grid `[120, 80, 24]` at `0.5` mm: expected `113.097336`, sampled `112.000000`, error `-1.097336` mm³; topology `{'watertight': False, 'unmatched_or_multiply_used_edges': 23878, 'geometry_pass_rule': 'null unless watertight topology and all tolerances pass'}`.
- `plunge_above`, CAMotics `0.1` mm, grid `[120, 80, 24]` at `0.5` mm: expected `113.097336`, sampled `112.000000`, error `-1.097336` mm³; topology `{'watertight': False, 'unmatched_or_multiply_used_edges': 62942, 'geometry_pass_rule': 'null unless watertight topology and all tolerances pass'}`.
- `horizontal_slot`, CAMotics `0.25` mm, grid `[120, 80, 24]` at `0.5` mm: expected `2067.292007`, sampled `1728.000000`, error `-339.292007` mm³; topology `{'watertight': False, 'unmatched_or_multiply_used_edges': 24582, 'geometry_pass_rule': 'null unless watertight topology and all tolerances pass'}`.
- `horizontal_slot`, CAMotics `0.1` mm, grid `[120, 80, 24]` at `0.5` mm: expected `2067.292007`, sampled `1728.000000`, error `-339.292007` mm³; topology `{'watertight': False, 'unmatched_or_multiply_used_edges': 65327, 'geometry_pass_rule': 'null unless watertight topology and all tolerances pass'}`.

### Endpoint-cap model limitation exposed by refinement

The two horizontal-slot runs at 0.5 mm task grid both measure exactly **1728 mm³** at their recorded precision. That equals `12 * (2*3) * 24`, the rectangular prism, whereas the independent full capsule requires another `12 * pi * 3^2 = 339.292007 mm³` at the two end positions. The missing independent contribution is preserved as the observed -339.292007 mm³ error; no fitted correction is applied. External 0.25 -> 0.1 mm does not recover it in these runs.

The inspected matching v1.2.0 tag's `ConicSweep::depth` (`src/camotics/sim/ConicSweep.cpp`, especially lines 81-106 and the axial bounds at 50-117) tests its bottom/top discs only in the branch outside the computed point's axial interval, with beta based on division by `Bz-Az`. For a horizontal constant-Z move, that is not a substitute for a flat cylindrical cutter disk at the path endpoints. The measured prism-vs-capsule result is consistent with a **model/endpoint limitation**, not something to explain away as grid thickness alone. This is matching-source inspection plus binary behavior, not a rebuilt-binary proof. Other trajectory families need their own validation; the horizontal capsule cannot be promoted to an accurate external machining oracle by this study.

The plunge, by contrast, goes from 64 mm³ at a 2 mm grid to 112 mm³ at a 0.5 mm grid, against 113.097336 mm³ continuous. That demonstrates a large cell-center sampling component for this case. It does not license extrapolating its near-1 mm³ error to other shapes.

### Earlier refinement blocker, preserved

The first four 0.5 mm-grid attempts (`/tmp/diffcam-t13a-bias.3b64_vk_/results.json`, SHA-256 `sha256:b797db82f9300e465be61147063c36520d48075c0b007d605ba42a6e379b14fb`) all stopped at the old 20M-entry distance-matrix guard. They produced no validated numeric volumes and were not used as coverage or calibration evidence. The bounded nearest-neighbor implementation in the metrics handoff (`4d9e024`, integrated by the coordinator as `cdc9071`) replaces that dense-allocation failure for 64^3 and these surfaces. The successful four-case rerun is reported separately; the blocked original records were not deleted or retroactively relabeled.

### Rapid-motion negative control

`rapid_through_solid` has independent reference removal **0 mm³**, and independent fixture safety `False` due assembly contact. Before the guard was added, a real historical CAMotics run returned sampled removal **1792.000000 mm³** at 0.5 mm external / 2 mm task-grid resolution. It returned `safety_pass=null`, not a discovered holder collision or a guarantee of a safe rapid. Its coverage was `incomplete`, geometry_pass `None`, and the external metrics do not satisfy the independent reference-carve rapid-removal semantics. Summary `/tmp/diffcam-t13a-rapid.se29aklv/summary.json`, SHA-256 `sha256:bd3edaaf16908ce3a8ba825c9fe57f4ae08639cea11242e23bbb4e5b6bf594d8`.

**Current behavior:** after shared NC interpretation, a rapid whose modeled cutter AABB can overlap initial stock returns `Coverage.UNSUPPORTED`, null geometry metrics and null safety/geometry verdicts without running camsim. Safe stationary initial-state markers are still allowed. The AABB check is only a conservative compatibility guard, not an exact collision test or holder-clearance proof. The positive pre-guard control above is retained to explain the guard.

CAMotics' own ToolSweep traverses the recorded moves with the selected cutter. Consequently an end-user must not treat a CAMotics geometric comparison as proof that rapid motion is removed correctly from the machine or that its non-cutting assembly clears stock. Explicit rapid collision and whole-cell guards remain separate obligations of T1.3b/T1.0c.


## Correction and tolerance policy

There is **no blanket subtraction of a positive 'padding bias'** and no fitted gain applied to CAMotics volume. Earlier one-project +594 -> +78 mm^3 findings are not a universal calibration constant. The fixed 2 mm evaluation grid and the external surface discretization have separate effects, so one external-resolution pair cannot establish a universal cell error law or a machining-safety margin.

The policy is:

1. Preserve the original coarse task-grid comparison as a combined mesh + grid diagnostic. Compare it against independent continuous expectations and report the sign and absolute error, not just correlation/pass fractions.
2. `geometry_pass` is null for a non-watertight output or unexpected interpreter warning. A successful Docker exit and nonempty STL are not semantic validation. This never sets `safety_pass` from cutter geometry.
3. Signed whole-STL sums are not used as exact remaining-stock volume unless valid closure is established; stock grids protect against querying the padded outer slab as extra initial material. Mean/max/p99/HD95 distances are nearest voxel-centre distances, not continuous Hausdorff guarantees or whole-cell safety intervals.
4. A production comparison needs a stated fixture-family tolerance in the task contract, a checked closed geometry result, and independent subtraction of grid quantization from reference expectations. The data below do not establish a universal resolution-to-tolerance curve. More grid refinement and consistent analytic-grid reference masks are needed before proposing one.
5. Preserve original part `P` and operation target `T`, with `P ⊆ T` enforced. `excluded_volume_mm3 = initial ∧ T ∧ ¬P`, `removal_target_volume_mm3 = initial ∧ ¬T`; the old erroneous `excluded_from_operation_volume_mm3` and `operation_target_removed_volume_mm3` names are not aliases. Gouge is removed stock inside `T` (including unreachable excluded stock), with `gouge_fraction` divided by `|initial ∧ T|`; original-part gouge `removed ∧ P` is a separate diagnostic, not the score. Residual is unremoved removal target. The adapter compares computed exclusion to the task declaration using explicit `exclusion_tolerance_mm3`; default zero means floating roundoff only, not a cell-size waiver. Mismatches are blocked with null metrics. This correction is `7638839` in the worker history / `67b7f37` on core, with shared contract names and the reported objective/target parity updates.
6. Rapid motion, holder/fixture contact, register calibration, spindle dynamics and path timing are outside this geometry-only certificate. Never turn the absence of an STL defect into holder or whole-cell clearance.

## Reproduction and validation records

The experiment driver was authored in `/tmp/diffcam-t13-bias/script/run_bias.py`; its source hash and the worktree/core revisions are in the study JSON. Each case contains exact task/motion JSON and both source/executed NC. The quantified report is generated from the case JSON, not from memory or fitting. Unavailable tools or blocked images are reported distinctly, not as skipped evidence or passing comparisons.

The calibration/sign probes, full task/motion JSON and source+generated NC hashes are attached to the cases rather than copied into new passing claims. Every companion case verifies its retained source, generated program and project hashes and checks the unchanged source suffix.

The image ID across the consolidated cases is `sha256:e680bff023d9190ed16fe1bee21d9c3c72f0e35bfbceb0a19e704c62f6ddca80`. A direct hardened probe with the source-corrected adapter recorded `camsim --version` exit 0, empty stdout and `stderr='1.2\n'`. Earlier batches have an empty `camsim_version` field because their adapter read stdout only; that metadata defect is explicitly retained in their raw records and fixed in the current implementation. It does not change the measured cell removal or image ID. Do not claim the old batch's empty revision field was already correct.

Focused gate: `TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-diffcam-worker-3 UV_CACHE_DIR=/tmp/diffcam-uv-diffcam-worker-3 uv run python -m pytest tests/csg_metrics_test.py tests/camotics_adapter_test.py -q` -> **exit 0, 35 passed** (14 metric + 21 adapter tests). Log `/tmp/diffcam-t13a-final-formal-focused.log`. NC/CAM compatibility was additionally exercised by the 117-test focused run, logged as `/tmp/diffcam-t13a-final-focused.log`. Real Docker checks ran and distinguish execution from proof of quality.

Full suite: same environment, `uv run python -m pytest -q` -> **exit 0, 422 passed, 1 skipped, 1 Taichi deprecation warning in 317.23s**, log `/tmp/diffcam-t13a-final-gate-fixed.log`. The sole skip is the existing opt-in `DIFFCNC_REFERENCE_CONVERGENCE=1` report test. This gate covers the merged target semantics, the adapter, and the existing production posted-path tests; it is not a statement that every semantic mismatch exposed by the external CAMotics study is resolved.

Two earlier integration failures were retained rather than hidden. The first was the runner's missing declared optional CMA dependency; activating `uv run --extra optim` made its isolated test pass in 11.34s (`/tmp/diffcam-t13a-runner-with-declared-extra.log`). The second was an exact-value optimizer comparison, observed as a 5.96e-8 difference both in the merged suite and alone. The previous merged head `7945d6f` passed once in a temporary detached comparison worktree, removed after that check (**1 passed**, isolated command log `/tmp/diffcam-t13a-blackbox-parent.log`). The coordinator diagnosed Taichi CPU atomic reduction variation and changed the test on core to relative `1e-6` / absolute `1e-9` bounds, retaining a scalar K=40 restoration assertion. The K-leak test passed after that fix (`/tmp/diffcam-t13-blackbox-current-fixed.log`), and the final full suite above is the actual complete gate result, not a claim that the earlier failures did not occur.
