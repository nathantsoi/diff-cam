# Physical-trial packet export (software only)

`python -m scripts.export_physical_packets` packages selected programs for review
by a separately authorized, qualified operator. It does **not** execute NC,
contact a machine, approve a setup, or establish that a physical trial is safe.
The physical status in an exported packet is always `pending setup review` until
those independent human checks exist outside this software.

## Scope and evidence boundary

The current real reproduction case is the retained T2.2b nominal at
`runs/retained/pilot-pocket-20261109-t22b-exact-exclusion/` in the integration
checkout. Its real simulator and evaluation files are used without re-running
or modifying them. `DIFFCAM_RETAINED_NOMINAL` can point to another copy of that
same immutable evidence when working in a linked worktree.

The retained nominal supplies:

- typed task, tool/holder, stock, original-part and operation-target records;
- typed motion and raw RS274 and Haas NC bytes with their actual SHA-256 values;
- complete production hard evaluations for **both** dialects, with local
  residual/gouge/surface-error values and actual removal;
- conservative non-cutting-assembly clearance lower bounds and rapid-cutter
  penetration upper bounds, including approach and retract motion;
- an independent `reference_carve` replay for each dialect at its actual explicit
  eight-subdivision setting, plus scoped stock-cell disagreement/contact checks;
- immutable source-field snapshots and their actual hashes and grid identity.

It has no stored production refinement. The packet therefore records
`production_refinement.present = false`, a null record/qualification and an
explicit reason for each dialect. The independent reference replay is **not**
relabeled as a production refinement or as a production geometry score. Its
`geometry_pass = null` remains unavailable, not true or false. The packet does
not claim globally nested temporal refinement or an exact continuous-collision
proof.

The exporter currently validates this retained nominal record schema. General
pilot/campaign packet generation remains dependent on the merged compatible
record/evaluation evidence and the T4.2 protocol freeze. Do not treat synthetic
schema examples or software packaging as physical-trial or optimizer evidence.

## Required setup sheet

The caller must provide an explicit `--setup` JSON file; the exporter does not
invent a machine, stock certificate, fixture or tool measurement. Its schema is
`physical-trial-setup/1` with:

- `machine`: identifier, manufacturer, model, record source and per-dialect
  controller declarations;
- `tool`: a setup-record identifier, the **exact** evaluated tool/holder assembly,
  declared stickout, and a measurement state. `declared_not_measured` is retained
  explicitly, not upgraded to an operator measurement. A measured state requires
  a separate operator-record identifier;
- `work_offset`: the evaluated offset identifier, full work-to-machine transform
  and known initial machine position;
- `fixtures`: exactly the evaluated fixture geometry and identity records. If the
  evaluated task has no fixture geometry, the setup must say so and must mark the
  physical fixture representation incomplete. A zero evaluated stock collision
  does not prove clearance from an unrepresented clamp;
- `floor`: part/setup-specific retained-stock or through-hole policy. The
  retained nominal's declared analytic stock bottom and pocket floor determine
  its remaining thickness; a generic global floor percentage is not used;
- `material`: name, alloy/grade and stock-record provenance;
- `process`: spindle direction/RPM, coolant, millimetres, the typed motion mode
  and no silent speed clipping.

Setup fields cannot silently differ from the evaluated task. Holder geometry,
stickout, fixture geometry, work transform, initial state and floor checks are
cross-checked. These schema checks do not certify physical dimensions on a real
machine.

## Manual operator review remains mandatory

Every packet contains four required items, with no fabricated completion record:

1. **first-article air cut**;
2. **Z touch-off**;
3. **holder-stickout measurement**;
4. **fixture clearance** review, including any geometry not represented in the
   evaluated task.

The selected NC dialect, controller declaration, holder/tool identity, work
frame, stock and fixture geometry still require operator review. Exporting a
packet authorizes none of these actions and no equipment execution.
The physical status is `pending setup review`. `simulation-qualified` refers only
to the preserved task/evaluator bounds and is not a universal or physical status.

## Qualification checks

Before creating the output directory, the exporter requires all of the
following for every selected run and dialect:

- complete immutable run/NC/task hashes and matching raw NC identity, including
  actual parsing of the NC and its recorded parsed-motion hash;
- complete production coverage and true production geometry/safety;
- complete distinct reference coverage and true reference safety. Reference
  stock-motion geometry may remain null, never a fabricated completion pass;
- residual, gouge and original-part gouge within declared task limits, nonzero
  actual removal, and maximum local surface deviation no greater than the
  declared millimetre tolerance;
- `min_clearance_lower_bound_mm` **at least** the task's declared non-cutting
  assembly clearance threshold in both production and reference checks;
- rapid-cutter penetration **upper bound no greater than** its declared
  tolerance, in both checks. Clearance lower bounds and penetration upper bounds
  are deliberately compared in opposite directions;
- independently scoped stock-cell disagreement, with no cells outside the
  declared local tolerance and physical parse-kind/position/time equivalence;
- matching operation-target exclusions and non-clipped typed execution;
- retained source-field hashes/grids for the nominal bundle.

A global gouge value or zero occupied-voxel collision by itself is insufficient.
Spatial and temporal bounds remain in `discretization`; their approximations and
uncertainty are retained rather than converted to a continuous-collision claim.
A qualifying packet does not suppress unknown coverage, failing fixtures or
missing dimensions.

Force-related evaluation values remain **proxies** unless separate calibrated
physical measurement evidence is supplied. The nominal's `force_proxy` is null
with its geometry-only reason; this is not calibrated force, spindle-load
calibration, or measured cutting performance.

## Development command

```sh
uv run python -m scripts.export_physical_packets \
  --selection baseline=/absolute/path/to/immutable-primary-run \
  --dialect haas \
  --setup /absolute/path/to/setup.json \
  --protocol /absolute/path/to/protocol.json \
  --output /absolute/path/to/a-new-packet-directory
```

`baseline` and `optimized` may each be selected once. Both selections must have
the same task/tool/setup identity. Selecting an optimized run does not create an
optimizer benefit; only independently retained completed records are packaged.
`--dialect` chooses the proposed controller record; both raw NC dialects and both
posted/reference evaluations are still packaged for independent review.

The protocol schema is `physical-trial-protocol/1`. For development it contains
an identifier and `freeze_state: pending`, and no `final_for_protocol` flag:

```json
{
  "schema_version": "physical-trial-protocol/1",
  "protocol_id": "T4.2-pending",
  "freeze_state": "pending",
  "frozen_record": null
}
```

The output root must not already exist. Every payload is prepared and checked
before it is created. `packet.json` is written last. `payload_sha256` and
`hashes.json` list every exported payload's raw-byte SHA-256. There are no export
timestamps or runtime-output paths in the payload, so identical immutable inputs
reproduce packet bytes identically. `verify_packet()` rehashes the packet and its
`packet.json` manifest.

## Final protocol refinement requirement

Final packets are not generated from the development nominal. When `--final-for-protocol`
is used, T4.2 must be `frozen` and a separate `frozen_record.path` plus actual
SHA-256 must identify its immutable record. Every selected role also needs
`--refinement-link ROLE=LINK_JSON`, naming the actual auxiliary link produced by
the pilot/final runner. The retained nominal's missing refinement is never used
to waive this requirement.

The runner link's key is `schema: diffcam-refinement-link/3`. The exporter checks:

- primary run ID and relative primary path;
- primary `run.json`, `metrics.json`, RS274 evaluation and Haas evaluation hashes;
- refined run ID and RS274-only `eval_posted_rs274_refined.json` relative path
  and hash;
- the auxiliary `run.json` path/hash and comparison path/hash, in the distinct
  `<primary-run-id>-refined` auxiliary directory;
- the comparison wrapper's primary/refined identities, comparison object and
  conservative outcome against the link;
- recomputed boolean and metric comparison rows from both typed evaluations;
  self-consistent hashes alone cannot establish a conservative outcome;
- preservation of every primary sampling control except a strictly smaller
  numeric `max_step_mm` on the refined RS274 replay;
- matching refined RS274 raw-NC/task/tool/setup/parsed-motion identities,
  complete coverage, passing local geometry/safety and conservative bounds;
- explicit `refinement_disagreement` as a boolean, unique declared disagreement
  fields, and a consistent `efficiency_eligible`/`excluded_from_efficiency` pair.

Both Haas and RS274 packet slots may cite this same **RS274-only** production
geometry/safety recheck because it shares the typed motion/task/tool setup. Haas
raw NC is not claimed to equal RS274 raw NC: the Haas program retains its own
primary production and independent-reference evaluations.

A final packet requires **no** conservative `safety_pass`, `geometry_pass`,
`completion_pass` or coverage disagreement, true conservative eligibility and
no efficiency exclusion. Unknown or non-conservative outcomes fail closed. A
flagged disagreement and its efficiency exclusion remain recorded in optional
development packets; they do not pass the final path.

Even after the protocol/refinement checks, packet status remains
`pending setup review`. Physical operation requires separate operator judgment,
calibration, first-article review and authorization. `equipment_execution_authorized`
and `physical_trial_performed` are always false in this software artifact.

## Focused validation

```sh
TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-worker1 \
UV_CACHE_DIR=/tmp/diffcam-uv-worker1 \
uv run python -m pytest tests/physical_packet_test.py -q
```

The tests use the real retained nominal for end-to-end packaging/reproduction.
Machine/material/setup and auxiliary-link fixtures are explicitly synthetic
metadata for rejection and schema-mechanism tests, not replacement simulation
results. No final protocol packet, actual refinement performance, physical
safety result or trial is asserted. The test reproduces the valid development
packet byte-for-byte and rejects malformed setups, inconsistent holders/work
offsets/fixtures/floors, hash corruption, unknown coverage, missing or failing
local/clearance/rapid bounds, unresolved independent disagreement, invalid final
protocol state and missing/incorrect refinement links.
