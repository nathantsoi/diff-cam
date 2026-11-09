"""T6.2 software-only measurement-ingestion tests.

All trial/measurement/operator/calibration records below are SYNTHETIC fixtures.
The actual retained-qualified T2.2b nominal packets identity/hash the fixture,
but there was no physical trial, optimized result, dynamometer calibration or
physical surface measurement. Repeating the source run in the optional
``optimized`` packet role is an explicit paired-ingestion fixture, not evidence
of an optimized trajectory or any improvement/no-improvement conclusion.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import contracts as C
from scripts import evaluate_physical_trials as M
from scripts.export_physical_packets import export_packet
from tests import physical_packet_test as packets


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-measurement") / "packet"
    export_packet(
        selections={"baseline": packets.RETAINED, "optimized": packets.RETAINED},
        setup=packets.valid_setup(), protocol=packets.pending_protocol(),
        selected_dialect="haas", output=root, final_for_protocol=False,
    )
    return root


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def csv_rows(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resource(root: Path, name: str) -> dict:
    return {"path": name, "sha256": digest(root / name)}


def refresh_resources(descriptor: dict, root: Path) -> None:
    for record, name in (
        (descriptor["controller"]["log"], "controller.csv"),
        (descriptor["registration"]["fixture_reference"], "fixture_fiducials.csv"),
        (descriptor["registration"]["measured_fiducials"], "measured_fiducials.csv"),
        (descriptor["surfaces"]["reference"], "reference_surface.csv"),
        (descriptor["surfaces"]["measured"], "measured_surface.csv"),
        (descriptor["surfaces"]["simulated"], "simulated_surface.csv"),
    ):
        record["sha256"] = digest(root / name)
    if descriptor.get("calibrated_force") is not None:
        descriptor["calibrated_force"]["samples"]["sha256"] = digest(root / "force.npz")
    write(root / "trial.json", descriptor)


def trial_fixture(root: Path, packet, *, role="baseline", physical_bias_mm=0.0,
                  interrupted=False, operator_rejected=False) -> dict:
    """Create one SYNTHETIC measured/replay fixture, never real physical evidence."""
    root.mkdir()
    manifest = load(packet / "packet.json")
    program = next(item for item in manifest["programs"] if item["role"] == role)
    produced = program["evaluations"]["haas"]
    nc = packet / produced["nc"]
    nc_bytes = nc.read_bytes()
    (root / "observed.nc").write_bytes(nc_bytes)
    motion = C.loads((packet / program["motion_record"]).read_text())
    angle = 0.20
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([0.3, -0.2, 0.1])
    offset = np.array([0.02, -0.01, 0.03])
    scale = 1.001
    rng = np.random.default_rng(5)

    def scanner(world):
        return (rotation.T @ (np.asarray(world) - translation)) / scale - offset

    refs = np.array([[10.0, 10.0, 0.0], [12.0, 10.0, 0.0],
                     [10.0, 13.0, 0.0], [11.0, 11.0, 1.5]])
    csv_rows(root / "fixture_fiducials.csv", ["fiducial_id", "x_mm", "y_mm", "z_mm"], [
        {"fiducial_id": f"F{i}", "x_mm": p[0], "y_mm": p[1], "z_mm": p[2]}
        for i, p in enumerate(refs)
    ])
    noisy_refs = refs + rng.normal(0.0, 0.0004, refs.shape)
    csv_rows(root / "measured_fiducials.csv", ["fiducial_id", "x_mm", "y_mm", "z_mm", "sigma_mm"], [
        {"fiducial_id": f"F{i}", "x_mm": p[0], "y_mm": p[1], "z_mm": p[2], "sigma_mm": 0.001}
        for i, p in enumerate(map(scanner, noisy_refs))
    ])
    samples = np.array([[-0.5, -0.5, -6.0], [0.5, -0.5, -6.0],
                        [-0.5, 0.5, -6.0], [0.5, 0.5, -6.0]])
    csv_rows(root / "reference_surface.csv", ["sample_id", "x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "sigma_mm"], [
        {"sample_id": f"S{i}", "x_mm": p[0], "y_mm": p[1], "z_mm": p[2],
         "nx": 0, "ny": 0, "nz": 1, "sigma_mm": 0.01}
        for i, p in enumerate(samples)
    ])
    physical = samples + np.array([0.0, 0.0, physical_bias_mm]) + rng.normal(0, 0.0002, samples.shape)
    csv_rows(root / "measured_surface.csv", ["sample_id", "x_mm", "y_mm", "z_mm", "sigma_mm"], [
        {"sample_id": f"S{i}", "x_mm": p[0], "y_mm": p[1], "z_mm": p[2], "sigma_mm": 0.002}
        for i, p in enumerate(map(scanner, physical))
    ])
    simulated = samples + np.array([0.0, 0.0, 0.001])
    csv_rows(root / "simulated_surface.csv", ["sample_id", "x_mm", "y_mm", "z_mm", "sigma_mm"], [
        {"sample_id": f"S{i}", "x_mm": p[0], "y_mm": p[1], "z_mm": p[2], "sigma_mm": 0.005}
        for i, p in enumerate(simulated)
    ])
    timestamp = 0.0
    rows = []
    segments = motion.segments[:3] if interrupted else motion.segments
    for index, segment in enumerate(segments):
        duration = segment.duration_s + (0.001 if int(segment.duration_s * 1e6) % 2 else 0.002)
        rows.append({
            "segment_id": segment.id, "observed_start_s": timestamp,
            "observed_end_s": timestamp + duration,
            "start_standard_uncertainty_s": 0.0004, "end_standard_uncertainty_s": 0.0005,
            "spindle_load_percent": 20.0 + index % 7,
            "spindle_load_standard_uncertainty_percent": 0.2,
        })
        timestamp += duration + 0.01
    csv_rows(root / "controller.csv", [
        "segment_id", "observed_start_s", "observed_end_s",
        "start_standard_uncertainty_s", "end_standard_uncertainty_s",
        "spindle_load_percent", "spindle_load_standard_uncertainty_percent",
    ], rows)
    np.savez(root / "force.npz", time_s=np.array([0.0, 0.1]),
             force_n=np.array([[12.0, 8.0, -40.0], [13.0, 7.0, -39.0]]),
             force_covariance_n2=np.repeat(np.diag([0.04, 0.03, 0.02])[None, ...], 2, axis=0))
    identity = {
        "role": role, "dialect": "haas", "run_id": program["run_id"],
        "packet_manifest_sha256": digest(packet / "packet.json"),
        "task_hash": program["task_hash"], "tool_setup_hash": program["tool_setup_hash"],
        "motion_hash": program["motion_hash"], "work_offset": program["work_offset"]["identifier"],
        "nc_sha256": produced["nc_sha256"],
        "executed_nc": {"path": "observed.nc", "sha256": digest(root / "observed.nc")},
        "operator_setup_review_id": "synthetic-setup-review-not-a-human-signoff",
        "operator_trial_log_id": "synthetic-trial-log-not-a-physical-run",
        "pairing_key": "synthetic-paired-surface-v1",
    }
    state = "interrupted" if interrupted else "rejected" if operator_rejected else "complete"
    reason = ("synthetic e-stop interruption for partial-record testing" if interrupted else
              "synthetic operator rejection for retention testing" if operator_rejected else
              "synthetic complete terminal state; not a physical execution")
    descriptor = {
        "schema": "synthetic-measurement-fixture/1",
        "identity": identity,
        "provenance": {"synthetic": True, "origin": "synthetic_unit_fixture", "sweep": "known rigid transform plus injected noise"},
        "termination": {"state": state, "reason": reason},
        "controller": {
            "log": resource(root, "controller.csv"),
            "time_calibration": {
                "record_id": "SYNTHETIC-CONTROLLER-TIME", "source": "synthetic log", "clock_id": "synthetic-controller-clock",
                "noise_model": "independent_timestamp_errors", "seconds_per_logged_second": 1.00001,
                "scale_standard_uncertainty": 1e-7,
            },
        },
        "position_calibration": {
            "kind": "scale_plus_translation", "record_id": "SYNTHETIC-SCAN-CAL",
            "instrument_id": "synthetic-scanner", "certificate_id": f"SYNTHETIC-SCAN-CAL-{role}",
            "scale": scale, "offset_mm": offset.tolist(),
            "covariance": np.diag([1e-10, 2e-8, 2e-8, 2e-8]).tolist(),
            "uncertainty": "one_standard_deviation_synthetic_fixture",
        },
        "registration": {
            "procedure": "datum_then_weighted_fiducial_svd", "fits_target_geometry": False,
            "acceptance": {"datum_rms_max_mm": 0.01, "rms_max_mm": 0.008, "max_point_mm": 0.015},
            "fixture_reference": resource(root, "fixture_fiducials.csv"),
            "measured_fiducials": resource(root, "measured_fiducials.csv"),
            "datum_rotation": rotation.tolist(),
            "datum_translation_mm": translation.tolist(),
        },
        "surfaces": {
            "reference": resource(root, "reference_surface.csv"),
            "measured": resource(root, "measured_surface.csv"),
            "simulated": resource(root, "simulated_surface.csv"),
            "simulation_provenance": {
                "description": "synthetic positional fixtures for software mechanics, linked to packet evaluator identity, not actual trial evidence",
                "evaluation_sha256": digest(packet / produced["production_record"]),
            },
        },
        "spindle_instrument": {
            "instrument_id": "synthetic-controller-load", "signal_kind": "spindle_load_percent_indicator",
            "calibration_state": "uncalibrated_indicator", "certificate_id": None,
        },
        "calibrated_force": {
            "instrument_id": "synthetic-dynamometer", "certificate_id": "SYNTHETIC-FORCE-CAL",
            "unit": "N", "samples": resource(root, "force.npz"),
            "sample_independence": "declared_independent",
        },
        "pairing_policy": "unknown_cross_trial_covariance",
    }
    write(root / "trial.json", descriptor)
    return descriptor


def input_pair(tmp_path, packet, **kwargs):
    left = trial_fixture(tmp_path / "baseline", packet, role="baseline", **kwargs)
    right = trial_fixture(tmp_path / "optimized", packet, role="optimized", **kwargs)
    return M.ingest_trial(packet, tmp_path / "baseline/trial.json"), M.ingest_trial(packet, tmp_path / "optimized/trial.json")


def test_registration_recovers_known_transform_with_noise_and_keeps_target_out_of_fit(tmp_path, packet):
    descriptor = trial_fixture(tmp_path / "trial", packet, physical_bias_mm=0.025)
    result = M.ingest_trial(packet, tmp_path / "trial/trial.json")
    assert result["synthetic"] is True
    assert result["accepted_for_matched_comparison"] is True
    assert result["empirical_eligibility"]["eligible"] is False
    assert result["physical_validation_pending"] is True
    registration = result["registration"]
    assert registration["accepted"] is True and registration["target_fit_used"] is False
    expected = np.asarray(descriptor["registration"]["datum_rotation"])
    np.testing.assert_allclose(registration["rotation"], expected, atol=5e-4, rtol=0.0)
    # A small fiducial basis amplifies angular noise into translation. The 5 um
    # bound admits the injected 0.4 um noise while rejecting the ~30 um offset-absorption error.
    np.testing.assert_allclose(registration["translation_mm"],
                               descriptor["registration"]["datum_translation_mm"], atol=5e-3, rtol=0.0)
    assert registration["fit_residual"]["rms_mm"] <= 0.008
    assert registration["fit_residual"]["max_mm"] <= 0.015
    assert len(result["controller"]["segments"]) == 75
    assert result["controller"]["coverage"] == "complete"
    for row in result["surfaces"]["samples"]:
        # The 25 um injected deviation is recovered within a 5 um small-basis
        # registration tolerance, not a fictitious 1.2 um exact-point guarantee.
        assert row["measured_minus_target_mm"] == pytest.approx(0.025, abs=0.005)
        assert row["simulated_minus_target_mm"] == pytest.approx(0.001, abs=1e-12)
        assert row["measured_minus_simulated_mm"] == pytest.approx(
            row["measured_minus_target_mm"] - row["simulated_minus_target_mm"]
        )
        assert row["measured_target_standard_uncertainty_mm"] > 0.0
        assert row["measured_simulated_standard_uncertainty_mm"] > 0.0
    assert result["surfaces"]["target_fit_used"] is False
    assert result["surfaces"]["sample_ids_discarded"] == []
    assert result["surfaces"]["aggregate"]["measured_vs_target"]["mean_signed_error_mm"] == pytest.approx(
        0.025, abs=0.005,
    )


@pytest.mark.parametrize("field", ["tool_setup_hash", "nc_sha256", "packet_manifest_sha256", "run_id"])
def test_identity_mismatch_rejected_before_measurement_payload_is_read(tmp_path, packet, field, monkeypatch):
    descriptor = trial_fixture(tmp_path / "trial", packet)
    descriptor["identity"][field] = "sha256:" + "0" * 64 if field != "run_id" else "other-run"
    write(tmp_path / "trial/trial.json", descriptor)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("measurement table was read before identity rejection")

    monkeypatch.setattr(M, "_csv", forbidden)
    with pytest.raises(M.MeasurementError, match="differs from( packet| verified packet)"):
        M.ingest_trial(packet, tmp_path / "trial/trial.json")


def test_executed_nc_bytes_must_match_packet_before_any_timing_parse(tmp_path, packet, monkeypatch):
    descriptor = trial_fixture(tmp_path / "trial", packet)
    with (tmp_path / "trial/observed.nc").open("ab") as stream:
        stream.write(b"( changed bytes )\n")
    # Intentionally do not 'repair' the linked hash; this is the immutable-identity test.
    monkeypatch.setattr(M, "_csv", lambda *_args, **_kwargs: pytest.fail("unexpected payload parse"))
    with pytest.raises(M.MeasurementError, match="executed NC bytes do not match"):
        M.ingest_trial(packet, tmp_path / "trial/trial.json")


def test_registration_failure_is_retained_with_residuals_not_dropped(tmp_path, packet):
    descriptor = trial_fixture(tmp_path / "trial", packet)
    path = tmp_path / "trial/measured_fiducials.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["x_mm"] = float(rows[0]["x_mm"]) + 1.0
    # Exercise FIT rejection with measured residuals, not the earlier datum gate;
    # the production fit rms/max acceptance thresholds remain unchanged.
    descriptor["registration"]["acceptance"]["datum_rms_max_mm"] = 1.1
    csv_rows(path, list(rows[0]), rows)
    refresh_resources(descriptor, tmp_path / "trial")
    result = M.ingest_trial(packet, tmp_path / "trial/trial.json")
    assert result["processing_status"] == "registration_rejected"
    assert result["accepted_for_matched_comparison"] is False
    assert result["registration"]["accepted"] is False
    assert result["registration"]["fit_residual"]["max_mm"] > 0.015
    assert "exceeds declared criteria" in result["rejection_reason"]
    assert result["registration"]["fit_residual"]["point_residual_vectors_mm"]
    assert result["surfaces"] is None


def test_initial_datum_and_noncollinear_fit_failures_remain_records(tmp_path, packet):
    descriptor = trial_fixture(tmp_path / "datum", packet)
    descriptor["registration"]["datum_translation_mm"] = [100.0, -0.2, 0.1]
    write(tmp_path / "datum/trial.json", descriptor)
    datum = M.ingest_trial(packet, tmp_path / "datum/trial.json")
    assert datum["processing_status"] == "registration_rejected"
    assert datum["registration"]["stage"] == "datum"
    assert datum["registration"]["datum_residual"]["rms_mm"] > 0.01
    assert datum["accepted_for_matched_comparison"] is False

    descriptor = trial_fixture(tmp_path / "collinear", packet)
    path = tmp_path / "collinear/measured_fiducials.csv"
    csv_rows(path, ["fiducial_id", "x_mm", "y_mm", "z_mm", "sigma_mm"], [
        {"fiducial_id": f"F{i}", "x_mm": 4, "y_mm": 4, "z_mm": 4, "sigma_mm": 0.001}
        for i in range(4)
    ])
    refresh_resources(descriptor, tmp_path / "collinear")
    # Datum failure was checked above. Deliberately permissive ONLY for this
    # negative test so execution reaches the non-collinear-fit estimator gate.
    descriptor["registration"]["acceptance"]["datum_rms_max_mm"] = 100.0
    write(tmp_path / "collinear/trial.json", descriptor)
    collinear = M.ingest_trial(packet, tmp_path / "collinear/trial.json")
    assert collinear["processing_status"] == "registration_rejected"
    assert collinear["registration"]["stage"] == "fit_estimation"
    assert collinear["registration"]["fit_residual"] is None
    assert collinear["registration"]["rotation"] is None
    assert collinear["registration"]["parameter_covariance"] is None
    assert "non-collinear" in collinear["registration"]["reason"]


def test_target_best_fit_registration_is_rejected_not_used_to_hide_error(tmp_path, packet):
    descriptor = trial_fixture(tmp_path / "trial", packet)
    descriptor["registration"]["procedure"] = "best_fit_to_operation_target"
    write(tmp_path / "trial/trial.json", descriptor)
    with pytest.raises(M.MeasurementError, match="only the explicit"):
        M.ingest_trial(packet, tmp_path / "trial/trial.json")
    descriptor["registration"]["procedure"] = "datum_then_weighted_fiducial_svd"
    descriptor["registration"]["fits_target_geometry"] = True
    write(tmp_path / "trial/trial.json", descriptor)
    with pytest.raises(M.MeasurementError, match="forbidden"):
        M.ingest_trial(packet, tmp_path / "trial/trial.json")


def test_uncertainty_propagates_shared_calibration_and_reference_cancellation(tmp_path, packet):
    descriptor = trial_fixture(tmp_path / "trial", packet)
    descriptor["position_calibration"]["covariance"] = np.diag([1e-10, 0.0, 0.0, 0.04]).tolist()
    write(tmp_path / "trial/trial.json", descriptor)
    result = M.ingest_trial(packet, tmp_path / "trial/trial.json")
    summary = result["surfaces"]["aggregate"]
    # Shared Z-offset sigma=0.2 mm does not divide by four independent points.
    assert summary["measured_vs_target"]["standard_uncertainty_mm"] > 0.19
    assert summary["measured_vs_target"]["expanded_uncertainty_k2_mm"] > 0.19 * 2
    assert summary["measured_vs_target"]["interval_mm"]["lower"] < 0.0
    assert summary["simulated_vs_target"]["standard_uncertainty_mm"] == pytest.approx(
        np.sqrt((0.005 ** 2 + 0.01 ** 2) / 4.0), abs=1e-12,
    )
    for sample in result["surfaces"]["samples"]:
        assert sample["measured_target_standard_uncertainty_mm"] > 0.19
        physically_paired_var = sample["physical_variance_without_reference_mm2"]
        simulated_var = sample["simulation_variance_without_reference_mm2"]
        expected = np.sqrt(physically_paired_var + simulated_var)
        assert sample["measured_simulated_standard_uncertainty_mm"] == pytest.approx(expected)
    covariance = np.asarray(result["registration"]["parameter_covariance"])
    assert covariance.shape == (6, 6)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_unknown_covariance_needs_a_real_failure_instead_of_shrinking_variance(tmp_path, packet):
    descriptor = trial_fixture(tmp_path / "trial", packet)
    descriptor["position_calibration"]["covariance"] = np.diag([0.0, 0.0, 0.0, 0.0]).tolist()
    write(tmp_path / "trial/trial.json", descriptor)
    with pytest.raises(M.MeasurementError, match="cannot be silently zero"):
        M.ingest_trial(packet, tmp_path / "trial/trial.json")


def test_interrupted_trial_retains_partial_log_and_blocks_paired_result(tmp_path, packet):
    left, right = input_pair(tmp_path, packet, interrupted=True, physical_bias_mm=0.0)
    assert left["processing_status"] == "retained_interrupted_partial_diagnostics"
    assert left["controller"]["coverage"] == "partial"
    assert len(left["controller"]["segments"]) == 3
    assert len(left["controller"]["missing_segment_ids"]) == 72
    assert left["termination"]["reason"]
    assert left["surfaces"] is not None  # retained partial measurements, not a completed machining result
    assert left["accepted_for_matched_comparison"] is False
    assert right["accepted_for_matched_comparison"] is False
    pair = M.compare_trials(left, right)
    assert pair["status"] == "blocked"
    assert pair["baseline_status"] == left["processing_status"]
    assert pair["optimized_status"] == right["processing_status"]


def test_operator_rejected_trial_remains_visible_even_if_sensor_arrays_exist(tmp_path, packet):
    descriptor = trial_fixture(tmp_path / "trial", packet, operator_rejected=True)
    result = M.ingest_trial(packet, tmp_path / "trial/trial.json")
    assert result["termination"]["state"] == "rejected"
    assert result["processing_status"] == "retained_operator_rejected_trial"
    assert result["accepted_for_matched_comparison"] is False
    assert result["surfaces"] is not None
    assert "operator rejection" in result["rejection_reason"]


def test_force_spindle_and_simulation_proxy_are_separate_channels(tmp_path, packet):
    descriptor = trial_fixture(tmp_path / "trial", packet)
    result = M.ingest_trial(packet, tmp_path / "trial/trial.json")
    channels = result["force_channels"]
    calibrated = channels["calibrated_force_n"]
    assert calibrated["unit"] == "N"
    assert calibrated["mean_vector_n"] == pytest.approx([12.5, 7.5, -39.5])
    assert calibrated["source_is_calibrated_dynamometer"] is True
    assert calibrated["certificate_id"] == "SYNTHETIC-FORCE-CAL"
    assert calibrated["not_inferred_from_spindle_load"] is True
    assert channels["spindle_load"]["unit"] == "percent_of_controller_rating"
    assert channels["spindle_load"]["calibrated_force_conversion"] is None
    assert channels["simulation_proxy"]["record"]["unit"] == "proxy"
    assert channels["simulation_proxy"]["record"]["value"] is None
    assert channels["simulation_proxy"]["calibrated_force_conversion"] is None
    first = result["controller"]["segments"][0]
    assert first["spindle_load_percent"] == 20.0
    assert calibrated["mean_vector_n"] != [20.0, 20.0, 20.0]
    descriptor.pop("calibrated_force")
    write(tmp_path / "trial/trial.json", descriptor)
    absent = M.ingest_trial(packet, tmp_path / "trial/trial.json")
    assert absent["force_channels"]["calibrated_force_n"] is None


def test_matched_baseline_optimized_allows_different_programs_with_unknown_covariance(tmp_path, packet):
    left, right = input_pair(tmp_path, packet, physical_bias_mm=0.01)
    # This checks the adapter's allowed baseline/optimized program distinction;
    # the underlying generated fixture itself is not a new optimizer output.
    right["identity"]["motion_hash"] = "sha256:" + "1" * 64
    right["identity"]["nc_sha256"] = "sha256:" + "2" * 64
    result = M.compare_trials(left, right)
    assert result["status"] == "paired"
    assert result["orientation"].startswith("optimized minus baseline")
    assert np.isfinite(result["optimized_minus_baseline_mean_mm"])
    assert abs(result["optimized_minus_baseline_mean_mm"]) < 0.002
    assert result["optimized_minus_baseline_standard_uncertainty_mm"] is None
    assert result["optimized_minus_baseline_expanded_uncertainty_k2_mm"] is None
    assert "not promoted to an uncertainty-qualified win" in result["uncertainty_status"]
    assert len(result["samples"]) == 4


def test_explicit_independent_pair_has_propagated_difference_uncertainty(tmp_path, packet):
    left = trial_fixture(tmp_path / "left", packet, role="baseline")
    right = trial_fixture(tmp_path / "right", packet, role="optimized", physical_bias_mm=0.02)
    baseline = M.ingest_trial(packet, tmp_path / "left/trial.json")
    optimized = M.ingest_trial(packet, tmp_path / "right/trial.json")
    for record in (baseline, optimized):
        record["pairing_policy"] = "declared_independent_calibration_and_acquisition"
    result = M.compare_trials(baseline, optimized)
    assert result["status"] == "paired"
    assert result["optimized_minus_baseline_mean_mm"] == pytest.approx(0.02, abs=0.002)
    assert result["optimized_minus_baseline_standard_uncertainty_mm"] > 0.0
    expected_variance = sum(record["surfaces"]["aggregate"]["variance_components_mm2"][
        "physical_mean_without_reference"
    ] for record in (baseline, optimized))
    assert result["optimized_minus_baseline_standard_uncertainty_mm"] == pytest.approx(np.sqrt(expected_variance))
    assert result["optimized_minus_baseline_expanded_uncertainty_k2_mm"] > 0.0


def test_empirical_report_refuses_synthetic_and_missing_provenance(tmp_path, packet):
    descriptor = trial_fixture(tmp_path / "trial", packet)
    record = M.ingest_trial(packet, tmp_path / "trial/trial.json")
    with pytest.raises(M.MeasurementError, match="refuse synthetic"):
        M.empirical_report([record], tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()
    stripped = dict(record, provenance={"origin": "measured_physical_trial", "synthetic": False}, synthetic=False)
    stripped.pop("position_calibration")
    with pytest.raises(M.MeasurementError, match="certificate_id"):
        M.empirical_report([stripped], tmp_path / "missing-certificate.json")


def test_cli_ingests_synthetic_record_but_refuses_empirical_report(tmp_path, packet, capsys):
    trial_fixture(tmp_path / "trial", packet)
    args = ["--packet", str(packet), "--trial", str(tmp_path / "trial/trial.json"),
            "--output-dir", str(tmp_path / "out")]
    assert M.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted"] == 1 and summary["synthetic"] is True
    assert summary["physical_validation_pending"] is True
    assert load(tmp_path / "out/baseline-measurement.json")["surfaces"]["aggregate"]
    args[-1] = str(tmp_path / "empirical")
    with pytest.raises(SystemExit) as failure:
        M.main(args + ["--empirical-report"])
    assert failure.value.code == 2
    assert "refuse synthetic" in capsys.readouterr().err
    assert not (tmp_path / "empirical").exists()


def test_cli_never_overwrites_an_existing_trial_role(tmp_path, packet, capsys):
    trial_fixture(tmp_path / "trial", packet)
    args = [
        "--packet", str(packet), "--trial", str(tmp_path / "trial/trial.json"),
        "--trial", str(tmp_path / "trial/trial.json"),
        "--output-dir", str(tmp_path / "duplicate-role-output"),
    ]
    with pytest.raises(SystemExit) as failure:
        M.main(args)
    assert failure.value.code == 2
    assert "duplicate trial role" in capsys.readouterr().err
    assert not (tmp_path / "duplicate-role-output").exists()