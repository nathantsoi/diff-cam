# Physical measurement ingestion (software only)

`python -m scripts.evaluate_physical_trials` reads measurement records against a
verified T6.1 packet. It does not execute an NC program, operate a scanner,
contact a controller or authorize a physical trial. A successful ingestion is not
a manufacturing result or an optimized-trajectory conclusion. Physical validation
remains pending until genuine, traceable measurement records exist.

The included tests use **synthetic** timing, identity/operator, registration,
calibration, surface and force records. They test software behavior only. The
retained qualified T2.2b packet provides real simulation hashes for those
fixtures. Repeating it under the synthetic `optimized` role tests pairing rules;
it is not an optimized trajectory or an improvement/no-improvement result.

## Packet and executed-program identity

Identity is checked **before** controller logs, fiducials, scans or force samples
are read. `ingest_trial(packet_root, trial_json)` requires:

- a valid T6.1 `packet.json` and every packet payload hash;
- actual SHA-256 of `packet.json` recorded by the trial source;
- the selected role and dialect, present in that same packet;
- task, tool/setup, motion, run and work-offset identities;
- actual raw executed-NC bytes whose digest equals the packet's selected NC hash;
- explicit operator setup-review and trial-log identifiers;
- a predeclared pairing key for baseline/optimized comparisons.

A stale packet, changed tool/holder setup, wrong run/NC identity, inconsistent
work offset or changed byte of executed NC is rejected before data parsing.
There is no fallback to whatever physical setup a measurement file happens to
declare. The two paired programs need the same task/setup/packet/dialect and
reference samples, but their **motion and NC hashes may and normally will differ**.

If T6.1's refinement-link schema changes, that interface must be re-reviewed.
This ingestion checks actual selected packet records and NC hashes rather than
inventing a refined-program relationship.

## Input schema and immutable resources

The trial JSON contains `identity`, `provenance`, `termination`, `controller`,
`position_calibration`, `registration`, `surfaces`, `spindle_instrument`, an
optional `calibrated_force` object and an optional explicit `pairing_policy`.
File references are relative to the trial JSON's directory and have this form:

```json
{"path": "controller.csv", "sha256": "sha256:<64 hex digits>"}
```

Absolute paths and parent-directory traversal are rejected. File digests are
verified before parsing. CSV headers must include all required fields and may
not contain duplicate names. Surface and fiducial identifiers must be unique;
missing IDs, additions and duplicates are not silently discarded.

`provenance.synthetic` is an explicit Boolean, not an inferred default. Real
empirical input must declare `synthetic: false` and
`origin: measured_physical_trial`, with traceable calibration/operator records.
Those declarations and source identities remain provenance records, not a
cryptographic proof of what happened in the physical world.

## Controller time and interrupted/rejected trials

`controller.log` is a CSV keyed by typed packet-motion segment ID, with observed
start/end times and start/end timestamp standard uncertainties. It may also
include spindle-load percentages and their standard uncertainties.

Time calibration declares the clock, record source, seconds-per-logged-second,
scale standard uncertainty and `independent_timestamp_errors`. Durations use the
typed commanded durations, never semantic labels such as "raster" as a hidden
physical kind. Reversed/overlapping intervals, unknown segment IDs and duplicate
segments fail ingestion. Completed trials require every typed segment, in order.

An interrupted trial keeps a contiguous measured prefix and an explicit list of
missing segments, with its termination reason. No hidden missing rows are
replaced by nominal timing. A final `rejected` operator state likewise retains
any available measurements but cannot enter an accepted paired comparison.
Available partial surface diagnostics can coexist with `accepted_for_matched_comparison
= false`; they do not become a completed machining result.

A datum or final registration failure retains controller/calibration information,
fit stage/criteria, available residuals and the rejection reason. Degenerate
fiducial geometry and non-finite/ill-conditioned fits remain rejected records
with unknown transform fields, not zero-error successful alignment.

## Registration: declared datums and fiducials, never target best-fit

The supported registration procedure is `datum_then_weighted_fiducial_svd`.
A future ICP procedure needs its own declared correspondence, outlier and
convergence contracts. The exporter does not silently select an alternative.

The input supplies:

1. a hash-identified fixture/fiducial reference table;
2. measured fiducial IDs, positions and positive per-point standard uncertainties;
3. an initial proper rotation and translation estimate;
4. a datum-RMS acceptance criterion and final RMS/maximum-point criteria.

The declared initial datum transformation is checked first. The weighted SVD
fit then minimizes residual between measured fiducials and **fixture reference
points on declared IDs**, with at least three non-collinear markers. Reflection,
non-orthogonality, underdetermination and failed residual thresholds are rejected.
The transform uncertainty uses an inverse-variance-weighted linearized six-
parameter rigid-transform information matrix with explicit parameter order:
world-frame angular increments in radians and translation in millimetres.

`fits_target_geometry` must be false. The fiducials are not the cut-surface
points used to score residual/gouge. There is no free best fit of a measured cut
surface to the target that could hide real gouges or local error.

The known instrument correction is applied **before** fitting. Otherwise the
known offset/scale can masquerade as a scanner translation error. The fitted
record includes the actual applied calibration record and scale/offset.

## Position calibration and surface correspondence

The supported position-calibration model is `scale_plus_translation` with a
traceable record, instrument/certificate IDs, scalar `scale`, 3-D `offset_mm`
and a symmetric positive-semidefinite 4×4 covariance. Parameter order is
`[unitless_scale, offset_x_mm, offset_y_mm, offset_z_mm]`. Nontrivial calibration
uncertainty is required; zero covariance cannot erase it.

For raw scanner point `p`, fitted rotation `R` and translation `t`, the corrected
point is:

```text
q = scale * R * (p + offset_mm) + t
```

Covariances include mixed offset/scale terms as declared by the certificate.
This is a numerical unit-conversion model, not a calibration performed by the
agent. The caller must supply real calibration evidence for empirical reports.

`surfaces.reference` supplies declared target/sample IDs, reference points,
unit normals and positive point uncertainties. `surfaces.measured` supplies raw
scan/probe points and standard uncertainties. `surfaces.simulated` supplies the
corresponding simulated points and their positional standard uncertainties.
The three sample-ID sets and order must agree. No points are cropped because they
would produce unfavorable gouges or surface errors.

The simulated positional file is linked to the selected packet production
evaluation by its actual SHA-256. Hash-identification is not alone proof that a
surface sample is a production simulator result: its declared derivation and
coverage must also be retained by the caller. The fixture surface files are
plainly marked synthetic.

Per-point signed comparisons use the declared unit normal `n`:

- measured minus target: `n · (q - target_point)`;
- simulated minus target: `n · (q_simulated - target_point)`;
- measured minus simulated: `n · (q - q_simulated)`.

Every point retains physical and simulation variance, reference variance, its
standard uncertainty and registered/reference/simulated coordinates. Aggregate
mean/RMS/max observed errors and arithmetic-mean uncertainty are reported. A
maximum is an observed sparse-sample maximum, not a full continuous-surface
bound or universal completion certificate.

## Uncertainty propagation

The per-point physical variance has independent raw-point, shared calibration
and shared registration terms. Calibration covariance includes scale/offset
cross-terms; registration covariance uses angular/translation Jacobians. With
`u_i` and `v_i` denoting those linearized gradients, it is represented as:

```text
independent = (scale * raw_sigma_i)^2
physical = independent + u_i * Sigma_calibration * u_i^T
                         + v_i * Sigma_registration * v_i^T
```

Target comparisons add the declared reference-point variance. Simulation
comparisons add the declared simulation positional variance and reference
variance, **not** scanner calibration/registration covariance. Measured-minus-
simulated differences use the same declared reference sample/normal, whose
independent reference uncertainty cancels in that particular paired difference.

For the arithmetic mean:

- independent raw-point terms are accumulated with the `1/N^2` rule;
- shared calibration/registration terms use the **mean gradient**, so their
  variance does not incorrectly shrink as `1/N`;
- simulation terms use only their declared positional uncertainties;
- matching reference terms appear in target errors and cancel in paired
  measured-vs-simulated differences.

Expanded intervals show observed mean ± `k=2` propagated standard uncertainty.
They are explicitly linearized/model-dependent intervals, not exact 95% coverage
guarantees, autocorrelation-aware spatial-field confidence regions or physical
safety certificates. Sparse/independent spatial-sample and timestamp/force
assumptions must match the actual measurement procedure.

## Baseline versus optimized pairing

Baseline and optimized ingestions can be compared only when they retain matching
packet, task/setup, role, dialect, pairing key, hashed reference surface and
sample order. Selected programs themselves may be different. Paired differences
are recorded as **optimized minus baseline** without assigning success to a sign.

By default, their mean-difference uncertainty is `null` with an explicit reason:
shared calibration/registration covariance is not invented as zero. A standard
uncertainty is computed only when both trials declare independent calibration
and acquisition, and their calibration certificate IDs differ. The physical
mean variances preserve correlated calibration/registration effects within each
trial; common reference-point uncertainty is not double counted.

Interrupted, rejected or failed-registration trials return a blocked paired
comparison carrying their reasons/statuses. They are not removed to manufacture
an accepted pair. No preferred optimizer result or strict-improvement condition
is built into this software.

## Force, spindle load and simulation proxies

These are separate fields with no inferred conversion:

- `calibrated_force_n`: optional dynamometer samples in newtons, with SI
  calibration-certificate ID, time series, per-sample 3×3 covariance and explicit
  independent-sample assumption for the mean. The mean covariance is accumulated
  using `1/N^2`; unknown time-series independence is not silently accepted.
- `spindle_load`: the controller's raw percentage indicator, with its declared
  instrumentation/calibration state and per-segment standard uncertainty.
  Percent load is not newtons and is not converted to physical force.
- `simulation_proxy`: the packet evaluator's proxy value with its original unit
  and null/reason metadata. It is never substituted for calibrated force.

A missing calibrated dynamometer remains `null`, not zero or a value inferred
from spindle load. When registration is rejected, unprocessed later evidence is
left unavailable rather than filled with a passing measurement.

## Empirical report refusal and CLI

Development ingestion, including paired processing and a blocked rejection, is
permitted for synthetic fixtures. It is not a physical-trial report. An empirical
report is validated before any output directory is created:

```sh
uv run python -m scripts.evaluate_physical_trials \
  --packet /path/to/packet \
  --trial /path/to/baseline/trial.json \
  --trial /path/to/optimized/trial.json \
  --output-dir /path/to/new-output \
  --empirical-report
```

`empirical_report()` and `--empirical-report` refuse synthetic-only or missing
physical provenance, missing operator/calibration traceability and duplicate
roles. A requested empirical-report failure creates no output directory or
claimed empirical artifact. Ordinary explicitly marked synthetic ingestion still
writes only its development measurements/blocked comparisons, as described above.
Each role may appear at most once in a CLI batch; duplicates are rejected rather
than overwriting an already ingested trial record. Genuine rejected/interrupted trials are retained
in an incomplete report with accepted/blocked counts rather than filtered out.
Accepted matched observations still carry their measurement, covariance,
calibration and limited-scope limitations.

Focused software gate:

```sh
TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-worker1 \
UV_CACHE_DIR=/tmp/diffcam-uv-worker1 \
uv run python -m pytest tests/physical_measurement_test.py -q
```

The tests exercise packet/NC/setup mismatch before payload reads, known noisy
rigid-transform recovery, datum/fit rejection retention, target-fit prevention,
shared calibration/registration and reference variance, interruption/rejection,
separate force/load/proxy channels, matched-program distinctions, unavailable
cross-trial covariance, explicitly independent pairs and empirical-report
refusal. They do not establish physical force calibration, machine operation,
CMM coverage or empirical optimizer performance.
