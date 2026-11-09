# DiffCAM contracts — `diffcam-contracts/1`

Executable source: `benchmarks/contracts.py` (T0.0). Tests: `tests/contracts_test.py`.
Fixtures: `tests/data/contracts/` (`hashes.json` pins the serialized form — a hash change is a schema change).
Owner: coordinator. Changing a field's meaning needs a version bump and a handoff to every consumer.

## Units, frames, conventions

| Item | Contract |
|---|---|
| Units | mm, s, mm/s, mm³. Inches are converted at I/O (`cam.units`); `length_unit` other than `"mm"` is rejected. |
| Work frame | G54, origin at stock top-centre, Z up. Task grid and analytic geometry live here. |
| Machine frame | `TaskSpec.work_to_machine`: rigid 4×4 (orthonormal, det +1, no scaling). Motion plans normally use the machine frame. |
| Tool reference | Cutter bottom-centre (tip); tool extends +Z. G43 active ⇒ positions are tip positions. |
| Grid | `GridSpec(shape, origin_mm, voxel_mm)`: cubic voxels, `origin_mm` = min corner of cell (0,0,0), cell centre = origin + (idx+½)·voxel. Grid phase / radius variants move `origin_mm` or stock bounds — never normalized coordinates. |
| Sign | Signed fields negative inside. |
| Membership | Removal: closed set (φ ≤ 0). Collision: strict interior (φ < 0); touching is not a collision. |
| Hashes | `sha256:<hex>` over canonical JSON (sorted keys, compact, shortest round-trip floats). NC hashes over raw bytes. |

## TaskSpec

`task_id, family, split ∈ {dev,test,external}, source{origin, license, checksums}, length_unit, stock,
original_part, operation_target{geometry, excluded_volume_mm3, exclusion_reason}, grid, work_to_machine,
initial_state, tool, limits, tolerances, motion_semantics, fixtures`.

- `GeometrySource.kind`: `analytic` (`params.primitive` required), `sampled` (path + sha256 + GridSpec),
  `mesh`/`step` (path + sha256; STEP must go through a real STEP backend).
- Original part vs operation target are separate; any excluded volume needs a reason, fixed before optimization.
- `ToolAssembly`: `cutter_kind ∈ {flat, ball}`, cutter radius, flute length, shank radius, holder bottom/radius/height
  measured from the tip. Consumers that model only flat cutters report coverage `unsupported` for ball tools.
- `MachineState.known=False` ⇒ `position_mm` must be null. An evaluator cannot claim complete coverage of the approach.
- `Tolerances` are absolute (mm³ / mm), not percentages: `completion_residual_mm3, gouge_volume_mm3, local_surface_error_mm, clearance_mm`.
- `task_hash` covers the whole spec. `tool_setup_hash` covers tool + transform + fixtures + initial state.

## MotionPlan

`frame, initial_state, segments[], timing_rule, dt_s, provenance`. Segment: `id` (stable, unique), `kind ∈ {rapid, feed, arc}`,
`start_mm, end_mm, duration_s, feed_mm_s` (null for rapids, which use `limits.rapid_mm_s`), `arc{centre_mm, plane, direction, turns}`,
`tag` (plunge/ramp/approach/retract…), `source_line`.

- Contiguous: each segment starts where the previous ended (≤ 1e-6 mm), the first at a known initial position.
- Timing rules: `exact_stop_constant_speed` (feed duration = length / feed, checked) or `fixed_dt` (durations are multiples of `dt_s`).
  All compared solvers use the same rule for a given task.
- Exact-stop durations: feed/arc = `path_length_mm / feed_mm_s` (checked on construction; arcs use helical arc length from
  `arc_geometry`, plane axes XY=(X,Y), XZ=(Z,X), YZ=(Y,Z), CCW positive about the plane normal, start==end ⇒ full circle, radius
  mismatch > 1e-3 mm rejected). Rapids = `length / limits.rapid_mm_s`, checked by `MotionPlan.validate_for_task(task)`, which also
  checks feed ≤ `limits.max_feed_mm_s` and initial-state agreement. Producers call it; evaluators may rely on it.
- Arcs carry centre/plane/direction. Consumers that can't sample arcs report `unsupported`/`incomplete` coverage — never silently chord them.
- `waypoints_mm()` is an export only.

### Execution semantics (all evaluators)

- The initial pose does not carve.
- A feed/arc segment removes the closed swept tool volume including both endpoint poses; a zero-length feed removes one
  stationary tool volume and contributes zero to time-integrated quantities.
- Repeating a feed is geometrically idempotent; temporal accumulators are not.
- Rapids never remove stock; they check cutter **and** assembly clearance against current stock and fixtures.
- Feed holder/shank checks use the evolving (chronological) remaining stock.

## EvaluationResult

`evaluator_id, evaluator_revision, task_hash, tool_setup_hash, discretization, coverage, coverage_reasons, metrics,
safety_pass, geometry_pass, nc_hash+dialect` (posted) **or** `motion_hash` (typed motion), `events[]`.

- `coverage ∈ {complete, incomplete, unsupported, unknown}`; non-complete requires reasons and forbids `safety_pass=true`.
- `safety_pass`/`geometry_pass` are true/false/null(unknown). `efficiency_eligible` is derived =
  complete ∧ safety ∧ geometry; a stored value that disagrees is rejected.
- `geometry_pass_from_metrics`: residual ≤ completion tol ∧ gouge ≤ gouge tol ∧ max surface error ≤ local tol;
  any missing metric ⇒ null. A no-cut program fails completion.
- `Metric(value, unit, denominator, null_reason)`: null requires a reason; 0 is a measured zero. Known metric names have fixed units
  (`METRIC_UNITS`), including the distinct collision quantities `unique_contacted_volume_mm3`, `collision_sample_count`,
  `time_integrated_overlap_mm3_s`, and separate `spatial_uncertainty_mm3` / `temporal_uncertainty_mm3`. `force_proxy` is unit `proxy`
  and never ranks or qualifies a program.
- `events`: `SegmentEvent(segment_id, kind, s∈[0,1] or null, detail)` — e.g. first holder/rapid collision.
- `discretization` holds evaluator/reference settings (voxel size, max step, endpoint inclusion, refinement limits, seed).
  These are **not** part of task/tool hashes.

## Candidates and runs

- `CandidateResult(candidate_id, theta_hash, objective_kind ∈ {hard, smooth}, objective, valid, invalid_reason, components,
  gradient_available, wall_time_s, coverage)`: invalid ⇔ objective null + reason.
- `RunRecord` (`run.json`): `run_id, task_hash, status ∈ {complete, failed, incomplete, interrupted, blocked}, artifacts, posted[], failure_reason,
  source_commit, dirty_patch_hash`. Complete runs must list `REQUIRED_RUN_ARTIFACTS` and one `PostedProgram` per dialect
  (`supported` with nc_hash + `eval_posted_<dialect>.json`, or `unsupported`/`failed` with a reason and no NC file). Non-complete runs need a
  reason and keep whatever artifacts exist.

## Interfaces

- `Evaluator.evaluate_candidate(task, motion, settings)` and `.evaluate_posted_gcode(task, nc_bytes, dialect, settings)` → `EvaluationResult`;
  stateless.
- `Parameterization.forward(theta) → (T,3) deltas mm`; `.pullback(grad_deltas) → (num_params,)` exact transpose.
- `ArtifactWriter.write(run_dir, record, files)`: files first, `run.json` last.

## API examples

```python
from benchmarks import contracts as C
task = C.example_pocket_task()            # 25.4 mm cube, pocket + island, flat 3.175 mm cutter
motion = C.example_motion_plan(task)      # approach, plunge, feed, arc, retract (machine frame)
text = C.dumps(task); assert C.loads(text) == task
task.task_hash, task.tool_setup_hash, motion.motion_hash
```
