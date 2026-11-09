"""Ingest physical-measurement records against immutable T6.1 packets.

Synthetic records are supported for software testing only. They never become an
empirical physical verdict. No controller or measurement hardware is operated.
NC identity is checked before instrument, timing or surface payloads are read.
Registration is against declared fixture datums/fiducials, never a free best fit
to the measured or target surface. Calibrated force, spindle load and simulation
proxies stay separate. See ``docs/physical_measurements.md`` for the schema.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks import contracts as C
from scripts.export_physical_packets import DIALECTS, PacketError, verify_packet


SCHEMA_VERSION = "physical-measurement-ingestion/1"
REPORT_SCHEMA = "physical-measurement-report/1"


class MeasurementError(ValueError):
    """Malformed identity, calibration or measurement input."""


class DataQualityError(MeasurementError):
    """Measured data that must be retained but cannot qualify for comparisons."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise MeasurementError(message)


def _text(record: Mapping[str, Any], key: str, where: str) -> str:
    value = record.get(key)
    _need(isinstance(value, str) and value.strip(), f"{where}.{key} is required")
    return value


def _map(value: Any, where: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping), f"{where} must be a JSON object")
    return copy.deepcopy(dict(value))


def _finite(value: Any, where: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    _need(isinstance(value, (int, float)) and not isinstance(value, bool),
          f"{where} must be numeric, not boolean")
    result = float(value)
    _need(math.isfinite(result), f"{where} must be finite")
    if positive:
        _need(result > 0, f"{where} must be positive")
    if nonnegative:
        _need(result >= 0, f"{where} must be non-negative")
    return result


def _cell(row: Mapping[str, Any], key: str, where: str, *, positive: bool = False) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise MeasurementError(f"{where} must be a finite numeric value") from error
    return _finite(value, where, positive=positive)


def _vec(value: Any, shape: tuple[int, ...], where: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise MeasurementError(f"{where} must be a numeric array") from error
    _need(result.shape == shape, f"{where} shape {result.shape} must be {shape}")
    _need(np.isfinite(result).all(), f"{where} must be finite")
    return result.astype(float, copy=True)


def _cov(value: Any, n: int, where: str) -> np.ndarray:
    matrix = _vec(value, (n, n), where)
    _need(np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10),
          f"{where} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    _need(eigenvalues.min() >= -1e-9, f"{where} must be positive semidefinite")
    result = 0.5 * (matrix + matrix.T)
    result.flags.writeable = False
    return result


def _json(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON value {token}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise MeasurementError(f"cannot load {path}: {error}") from error


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise MeasurementError(f"cannot hash {path}: {error}") from error
    return "sha256:" + digest.hexdigest()


def _safe(root: Path, name: Any, where: str) -> Path:
    _need(isinstance(name, str), f"{where} must be a relative path string")
    path = PurePosixPath(name)
    _need(not path.is_absolute() and path.parts and all(part not in ("", ".", "..") for part in path.parts),
          f"{where} must not be absolute, empty or traverse parent directories")
    return root.joinpath(*path.parts)


def _resource(root: Path, record: Mapping[str, Any], where: str) -> tuple[Path, dict[str, Any]]:
    value = _map(record, where)
    path = _safe(root, _text(value, "path", where), where + ".path")
    expected = _text(value, "sha256", where)
    actual = _file_digest(path)
    _need(actual == expected, f"{where} hash mismatch: {actual} != {expected}")
    return path, {"path": value["path"], "sha256": actual}


def _csv(path: Path, required: set[str], where: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            _need(reader.fieldnames is not None and required <= set(reader.fieldnames) and
                  len(reader.fieldnames) == len(set(reader.fieldnames)),
                  f"{where} lacks required columns {sorted(required)} or has duplicates")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise MeasurementError(f"cannot read {where}: {error}") from error
    _need(rows, f"{where} must contain at least one data row")
    for line, row in enumerate(rows, 2):
        for name in required:
            _need(row.get(name) not in (None, ""), f"{where} row {line} missing {name}")
    return rows


def _identity(packet_root: Path, trial_root: Path,
              raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], C.TaskSpec, C.MotionPlan]:
    """Do not unpack measurement payloads before this whole identity check."""
    try:
        packet = verify_packet(packet_root)
    except (PacketError, OSError, TypeError, ValueError) as error:
        raise MeasurementError(f"T6.1 packet failed verification: {error}") from error
    data = _map(raw.get("identity"), "identity")
    role = _text(data, "role", "identity")
    dialect = _text(data, "dialect", "identity")
    _need(dialect in DIALECTS, "identity.dialect must be rs274 or haas")
    program = next((item for item in packet["programs"] if item["role"] == role), None)
    _need(program is not None, f"packet has no selected {role!r} program")
    _need(program["selected_dialect"] == dialect, "selected packet program dialect differs from trial log")
    _need(_file_digest(packet_root / "packet.json") == _text(data, "packet_manifest_sha256", "identity"),
          "measured packet manifest hash differs from verified packet")
    _need(program["run_id"] == _text(data, "run_id", "identity"), "trial run identity differs from packet")
    for field in ("task_hash", "tool_setup_hash", "motion_hash"):
        _need(program[field] == _text(data, field, "identity"),
              f"trial {field} differs from packet")
    _need(program["work_offset"]["identifier"] == _text(data, "work_offset", "identity"),
          "trial work offset differs from packet")
    measurement = program["evaluations"][dialect]
    nc_hash = measurement["nc_sha256"]
    _need(nc_hash == _text(data, "nc_sha256", "identity"), "trial NC hash differs from packet")
    external_nc = _map(data.get("executed_nc"), "identity.executed_nc")
    nc_path = _safe(trial_root, _text(external_nc, "path", "identity.executed_nc"),
                    "identity.executed_nc.path")
    actual = _file_digest(nc_path)
    _need(actual == nc_hash, "executed NC bytes do not match selected packet program")
    task = _typed(packet_root / program["task_record"], C.TaskSpec)
    _need(task.task_hash == program["task_hash"] and task.tool_setup_hash == program["tool_setup_hash"],
          "packet typed task identity is inconsistent")
    motion = _typed(packet_root / program["motion_record"], C.MotionPlan)
    _need(motion.motion_hash == program["motion_hash"], "packet typed motion hash is inconsistent")
    verified = dict(data)
    verified["executed_nc"] = {
        "path": external_nc["path"], "sha256": actual,
        "matches_packet_bytes": True,
    }
    verified["operator_setup_review_id"] = _text(data, "operator_setup_review_id", "identity")
    verified["operator_trial_log_id"] = _text(data, "operator_trial_log_id", "identity")
    _text(data, "pairing_key", "identity")
    return verified, program, task, motion


def _typed(path: Path, expected: type) -> Any:
    try:
        result = C.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, C.ContractError) as error:
        raise MeasurementError(f"invalid typed record {path}: {error}") from error
    _need(isinstance(result, expected), f"{path} is not {expected.__name__}")
    return result


def _timing(root: Path, raw: Mapping[str, Any], motion: C.MotionPlan,
            state: Mapping[str, Any]) -> dict[str, Any]:
    spec = _map(raw.get("controller"), "controller")
    path, source = _resource(root, spec.get("log"), "controller.log")
    rows = _csv(path, {"segment_id", "observed_start_s", "observed_end_s",
                       "start_standard_uncertainty_s", "end_standard_uncertainty_s"}, "controller log")
    calibration = _map(spec.get("time_calibration"), "controller.time_calibration")
    _text(calibration, "record_id", "controller.time_calibration")
    _text(calibration, "source", "controller.time_calibration")
    _text(calibration, "clock_id", "controller.time_calibration")
    _need(calibration.get("noise_model") == "independent_timestamp_errors",
          "controller timing requires independent timestamp-error declaration")
    scale = _finite(calibration.get("seconds_per_logged_second"), "controller.time calibration scale", positive=True)
    scale_std = _finite(calibration.get("scale_standard_uncertainty"),
                        "controller.time calibration scale uncertainty", nonnegative=True)
    seen = []
    segments = {segment.id: segment for segment in motion.segments}
    spindle_rows = []
    previous_end = 0.0
    for row in rows:
        segment = segments.get(row["segment_id"])
        _need(segment is not None, f"controller log names unknown segment {row['segment_id']}")
        _need(segment.id not in seen, f"controller log repeats segment {segment.id}")
        start = _cell(row, "observed_start_s", f"{segment.id}.start")
        end = _cell(row, "observed_end_s", f"{segment.id}.end")
        _need(end > start and start >= previous_end - 1e-12,
              f"controller timestamps are reversed/overlapping at {segment.id}")
        su = _cell(row, "start_standard_uncertainty_s", f"{segment.id}.start uncertainty", positive=True)
        eu = _cell(row, "end_standard_uncertainty_s", f"{segment.id}.end uncertainty", positive=True)
        duration = (end - start) * scale
        variance = (scale * scale * (su * su + eu * eu)
                    + ((end - start) * scale_std) ** 2)
        seen.append(segment.id)
        segments_record = {
            "segment_id": segment.id, "kind": segment.kind.value,
            "commanded_duration_s": segment.duration_s,
            "observed_duration_s": duration,
            "observed_minus_commanded_s": duration - segment.duration_s,
            "duration_standard_uncertainty_s": math.sqrt(variance),
        }
        spindle_rows.append(segments_record)
        if row.get("spindle_load_percent") not in (None, ""):
            value = _cell(row, "spindle_load_percent", f"{segment.id}.spindle_load")
            sigma = _cell(row, "spindle_load_standard_uncertainty_percent",
                          f"{segment.id}.spindle_load uncertainty", positive=True)
            segments_record["spindle_load_percent"] = value
            segments_record["spindle_load_standard_uncertainty_percent"] = sigma
        previous_end = end
    ordered = [segment.id for segment in motion.segments]
    if state["state"] == "complete":
        _need(seen == ordered, "complete trial controller log does not cover every typed segment in order")
        coverage, absent = "complete", []
    else:
        _need(seen == ordered[: len(seen)],
              "interrupted log must be an ordered, undropped prefix of the typed motion")
        absent = ordered[len(seen):]
        coverage = "partial" if absent else "complete"
    return {
        "source": source, "coverage": coverage,
        "missing_segment_ids": absent, "clock_calibration": calibration,
        "segments": spindle_rows,
        "commanded_duration_s": sum(segment.duration_s for segment in motion.segments),
        "observed_duration_covered_s": sum(row["observed_duration_s"] for row in spindle_rows),
        "interpretation": "controller timing, not an optimized-efficiency or physical-safety verdict",
    }


def _fit_transform(measured: np.ndarray, reference: np.ndarray,
                   sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted non-collinear fiducial SVD; no target-surface input exists here."""
    _need(measured.shape == reference.shape and measured.ndim == 2 and measured.shape[0] >= 3 and measured.shape[1] == 3,
          "fiducial fit needs matching Nx3 arrays with N >= 3")
    _need(sigmas.shape == (measured.shape[0],) and (sigmas > 0).all(), "fiducial uncertainties must be positive")
    _need(np.linalg.matrix_rank(np.column_stack((measured, np.ones(len(measured))))) >= 3 and
          (np.linalg.norm(reference - reference.mean(axis=0), axis=1) > 0).any() and
          (np.linalg.norm(measured - measured.mean(axis=0), axis=1) > 0).any(),
          "fiducials must be non-collinear in three dimensions")
    weights = 1.0 / (sigmas * sigmas)
    weights /= weights.sum()
    left = np.sum(measured * weights[:, None], axis=0)
    right = np.sum(reference * weights[:, None], axis=0)
    u, _, vt = np.linalg.svd((measured - left).T @ ((reference - right) * weights[:, None]))
    sign = np.eye(3)
    sign[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ sign @ u.T
    translation = right - rotation @ left
    _need(np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0) and
          np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0),
          "fiducial fit did not produce a proper rotation")
    residual = (rotation @ measured.T).T + translation - reference
    _need(np.isfinite(residual).all(), "non-finite fiducial registration residuals")
    transformed = (rotation @ measured.T).T + translation
    information = np.zeros((6, 6), dtype=float)
    inverse_variances = 1.0 / (sigmas * sigmas)
    for point, weight in zip(transformed, inverse_variances):
        x, y, z = point
        block = np.array([[0.0, z, -y], [-z, 0.0, x], [y, -x, 0.0]])
        # angular increments are world-frame radians; translation is mm.
        jacobian = np.concatenate((block, np.eye(3)), axis=1)
        information += weight * jacobian.T @ jacobian
    _need(np.linalg.cond(information) < 1e12, "fiducial registration covariance is ill-conditioned")
    covariance = np.linalg.inv(information)
    _need(np.isfinite(covariance).all(), "non-finite registration covariance")
    return rotation, translation, covariance


def _residual_report(residual: np.ndarray) -> dict[str, Any]:
    lengths = np.linalg.norm(residual, axis=1)
    return {
        "rms_mm": float(np.sqrt(np.mean(residual ** 2))),
        "max_mm": float(lengths.max()),
        "point_residual_vectors_mm": residual.tolist(),
    }


def _registration(root: Path, raw: Mapping[str, Any],
                  calibration: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    spec = _map(raw.get("registration"), "registration")
    _need(spec.get("procedure") == "datum_then_weighted_fiducial_svd",
          "only the explicit datum_then_weighted_fiducial_svd registration procedure is supported")
    _need(spec.get("fits_target_geometry") is False,
          "fitting to operation-target geometry is forbidden and could hide machining gouges")
    limit = _map(spec.get("acceptance"), "registration.acceptance")
    datum_limit = _finite(limit.get("datum_rms_max_mm"), "registration.datum_rms_max_mm", nonnegative=True)
    rms_limit = _finite(limit.get("rms_max_mm"), "registration.rms_max_mm", nonnegative=True)
    max_limit = _finite(limit.get("max_point_mm"), "registration.max_point_mm", nonnegative=True)
    reference_path, reference_source = _resource(root, spec.get("fixture_reference"), "registration.fixture_reference")
    reference_rows = _csv(reference_path, {"fiducial_id", "x_mm", "y_mm", "z_mm"}, "fixture fiducial reference")
    reference = {row["fiducial_id"]: _vec([row["x_mm"], row["y_mm"], row["z_mm"]], (3,), "fiducial reference")
                 for row in reference_rows}
    measured_path, measured_source = _resource(root, spec.get("measured_fiducials"), "registration.measured_fiducials")
    measured_rows = _csv(measured_path, {"fiducial_id", "x_mm", "y_mm", "z_mm", "sigma_mm"}, "measured fiducials")
    ids = [row["fiducial_id"] for row in measured_rows]
    _need(len(ids) == len(set(ids)) and len(reference) == len(reference_rows) and
          set(ids) == set(reference),
          "measured and declared fixture fiducial IDs differ")
    observed = np.asarray([_vec([row["x_mm"], row["y_mm"], row["z_mm"]], (3,), "measured fiducial")
                           for row in measured_rows])
    # Apply the SAME known instrument correction as surface ingestion BEFORE
    # fitting the rigid registration. Otherwise scale/offset can masquerade as a
    # translation in the recovered scanner pose (T6.2 fixture regression).
    scale = _finite(calibration.get("scale"), "registration calibration scale", positive=True)
    offset = _vec(calibration.get("offset_mm"), (3,), "registration calibration offset")
    observed = scale * (observed + offset)
    expected = np.asarray([reference[index] for index in ids])
    sigmas = np.asarray([_cell(row, "sigma_mm", "fiducial sigma", positive=True) for row in measured_rows])
    sigmas = sigmas * scale
    initial_r = _vec(spec.get("datum_rotation"), (3, 3), "registration.datum_rotation")
    initial_t = _vec(spec.get("datum_translation_mm"), (3,), "registration.datum_translation_mm")
    _need(np.isclose(np.linalg.det(initial_r), 1.0, atol=1e-8, rtol=0.0) and
          np.allclose(initial_r.T @ initial_r, np.eye(3), atol=1e-8, rtol=0.0),
          "declared datum rotation must be proper and orthonormal")
    datum_residual = (initial_r @ observed.T).T + initial_t - expected
    datum_report = _residual_report(datum_residual)
    if datum_report["rms_mm"] > datum_limit:
        raise DataQualityError(
            f"declared datum residual {datum_report['rms_mm']} mm exceeds {datum_limit} mm",
            {
                "accepted": False, "stage": "datum", "procedure": spec["procedure"],
                "target_fit_used": False, "fixture_reference": reference_source,
                "measured_fiducials": measured_source, "fiducial_ids": ids,
                "criteria": {"datum_rms_criterion_mm": datum_limit},
                "datum_residual": datum_report,
                "datum_rotation": initial_r.tolist(), "datum_translation_mm": initial_t.tolist(),
            },
        )
    try:
        rotation, translation, covariance = _fit_transform(observed, expected, sigmas)
    except MeasurementError as error:
        raise DataQualityError(
            f"fiducial fit could not be estimated: {error}",
            {
                "accepted": False, "stage": "fit_estimation", "procedure": spec["procedure"],
                "target_fit_used": False, "fixture_reference": reference_source,
                "measured_fiducials": measured_source, "fiducial_ids": ids,
                "datum_residual": datum_report, "reason": str(error),
                "fit_residual": None, "rotation": None, "translation_mm": None,
                "parameter_covariance": None,
            },
        ) from error
    residual = (rotation @ observed.T).T + translation - expected
    fit_report = _residual_report(residual)
    factors = {
        "rms_criterion_mm": rms_limit, "max_point_criterion_mm": max_limit,
        "datum_rms_criterion_mm": datum_limit,
    }
    diagnostics = {
        "procedure": spec["procedure"], "target_fit_used": False,
        "instrument_correction_applied_before_fit": {
            "record_id": calibration["record_id"], "scale": scale, "offset_mm": offset.tolist(),
        },
        "fixture_reference": reference_source, "measured_fiducials": measured_source,
        "fiducial_ids": ids, "criteria": factors,
        "datum_residual": datum_report, "fit_residual": fit_report,
        "rotation": rotation.tolist(), "translation_mm": translation.tolist(),
        "parameter_covariance": covariance.tolist(),
        "parameter_order": ["rotation_x_rad", "rotation_y_rad", "rotation_z_rad", "x_mm", "y_mm", "z_mm"],
        "uncertainty_model": "inverse-variance weighted linearized rigid-transform Fisher information; not an exact nonlinear distribution certificate",
    }
    if fit_report["rms_mm"] > rms_limit or fit_report["max_mm"] > max_limit:
        raise DataQualityError(
            f"fiducial registration residual exceeds declared criteria: rms={fit_report['rms_mm']:.9g} mm, "
            f"max={fit_report['max_mm']:.9g} mm",
            diagnostics | {"accepted": False},
        )
    return diagnostics | {"accepted": True}, rotation, translation, covariance


def _position_calibration(raw: Mapping[str, Any]) -> dict[str, Any]:
    record = _map(raw.get("position_calibration"), "position_calibration")
    _need(record.get("kind") == "scale_plus_translation", "position calibration must be scale_plus_translation")
    _text(record, "record_id", "position_calibration")
    _text(record, "instrument_id", "position_calibration")
    _text(record, "certificate_id", "position_calibration")
    scale = _finite(record.get("scale"), "position calibration.scale", positive=True)
    offset = _vec(record.get("offset_mm"), (3,), "position calibration.offset_mm")
    covariance = _cov(record.get("covariance"), 4, "calibration covariance")
    _need(np.trace(covariance) > 0, "position calibration uncertainty cannot be silently zero")
    return {
        "record_id": record["record_id"], "instrument_id": record["instrument_id"],
        "certificate_id": record["certificate_id"], "kind": record["kind"],
        "scale": scale, "offset_mm": offset.tolist(), "covariance": covariance.tolist(),
        "parameter_order": ["unitless_scale", "offset_x_mm", "offset_y_mm", "offset_z_mm"],
        "uncertainty": record.get("uncertainty", "one_standard_deviation"),
        "covariance_units": "unitless scale with mm offsets; mixed covariances as declared by certificate",
    }


def _surface(root: Path, raw: Mapping[str, Any], calibration: Mapping[str, Any],
             registration: Mapping[str, Any]) -> dict[str, Any]:
    declaration = _map(raw.get("surfaces"), "surfaces")
    reference_path, reference_source = _resource(root, declaration.get("reference"), "surfaces.reference")
    measured_path, measured_source = _resource(root, declaration.get("measured"), "surfaces.measured")
    simulation_path, simulation_source = _resource(root, declaration.get("simulated"), "surfaces.simulated")
    reference_rows = _csv(reference_path, {"sample_id", "x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "sigma_mm"}, "target surface")
    measured_rows = _csv(measured_path, {"sample_id", "x_mm", "y_mm", "z_mm", "sigma_mm"}, "measured surface")
    simulation_rows = _csv(simulation_path, {"sample_id", "x_mm", "y_mm", "z_mm", "sigma_mm"}, "simulation surface")
    refs = {row["sample_id"]: row for row in reference_rows}
    _need(len(refs) == len(reference_rows), "reference sample IDs must be unique")
    measured = {row["sample_id"]: row for row in measured_rows}
    simulated = {row["sample_id"]: row for row in simulation_rows}
    _need(len(measured) == len(measured_rows) and len(simulated) == len(simulation_rows),
          "measured/simulated sample IDs must be unique")
    _need(set(measured) == set(simulated) == set(refs) and measured,
          "measured, simulated and reference samples must match exactly; no silently discarded points")
    rotation = _vec(registration["rotation"], (3, 3), "registration.rotation")
    translation = _vec(registration["translation_mm"], (3,), "registration.translation_mm")
    registration_covariance = _cov(registration["parameter_covariance"], 6, "registration covariance")
    calibration_covariance = _cov(calibration["covariance"], 4, "calibration covariance")
    scale = calibration["scale"]
    offset = np.asarray(calibration["offset_mm"])
    percent, point_records = [], []
    for sample_id, required in refs.items():
        normal = _vec([required["nx"], required["ny"], required["nz"]], (3,), "surface normal")
        _need(abs(np.linalg.norm(normal) - 1.0) < 1e-8, f"{sample_id} normal is not unit length")
        target = _vec([required["x_mm"], required["y_mm"], required["z_mm"]], (3,), "reference point")
        reference_sigma = _cell(required, "sigma_mm", "reference point sigma", positive=True)
        raw_point = _vec([measured[sample_id][name] for name in ("x_mm", "y_mm", "z_mm")], (3,), "measured point")
        raw_sigma = _cell(measured[sample_id], "sigma_mm", "measured point sigma", positive=True)
        scaled_point = scale * rotation @ (raw_point + offset) + translation
        simulated_point = _vec([simulated[sample_id][name] for name in ("x_mm", "y_mm", "z_mm")], (3,), "simulated point")
        simulated_sigma = _cell(simulated[sample_id], "sigma_mm", "simulation point sigma", positive=True)
        # Linearized shared calibration and registration terms; the reference
        # property is kept separate because it cancels in paired P-S errors.
        u = np.concatenate(([float(normal @ (rotation @ raw_point))], normal @ rotation))
        v = np.concatenate((normal @ np.array([[0.0, scaled_point[2], -scaled_point[1]],
                                               [-scaled_point[2], 0.0, scaled_point[0]],
                                               [scaled_point[1], -scaled_point[0], 0.0]]),
                            normal @ np.eye(3)))
        independent_variance = (scale * raw_sigma) ** 2
        calibration_variance = float(u @ calibration_covariance @ u)
        registration_variance = float(v @ registration_covariance @ v)
        physical_variance = independent_variance + calibration_variance + registration_variance
        _need(physical_variance >= 0.0, "negative propagated physical variance")
        physical_error = float(normal @ (scaled_point - target))
        simulation_error = float(normal @ (simulated_point - target))
        relative_error = float(normal @ (scaled_point - simulated_point))
        point = {
            "sample_id": sample_id, "registered_point_mm": scaled_point.tolist(),
            "reference_point_mm": target.tolist(), "simulation_point_mm": simulated_point.tolist(),
            "normal": normal.tolist(),
            "measured_minus_target_mm": physical_error,
            "simulated_minus_target_mm": simulation_error,
            "measured_minus_simulated_mm": relative_error,
            "physical_variance_without_reference_mm2": physical_variance,
            "reference_variance_mm2": reference_sigma ** 2,
            "simulation_variance_without_reference_mm2": simulated_sigma ** 2,
            "measured_target_standard_uncertainty_mm": math.sqrt(physical_variance + reference_sigma ** 2),
            "simulated_target_standard_uncertainty_mm": math.sqrt(simulated_sigma ** 2 + reference_sigma ** 2),
            "measured_simulated_standard_uncertainty_mm": math.sqrt(physical_variance + simulated_sigma ** 2),
        }
        point_records.append(point)
        gradient = (u, v, scale * raw_sigma, reference_sigma, simulated_sigma)
        percent.append(gradient)

    count = len(percent)
    mean_u = np.mean([gradient[0] for gradient in percent], axis=0)
    mean_v = np.mean([gradient[1] for gradient in percent], axis=0)
    # Physical calibration and registration are SHARED across measured points.
    # Simulation has its own declared positional uncertainty, not the scanner's.
    physical_mean_variance = float(np.sum([gradient[2] ** 2 for gradient in percent])) / count ** 2
    physical_mean_variance += float(mean_u @ calibration_covariance @ mean_u)
    physical_mean_variance += float(mean_v @ registration_covariance @ mean_v)
    simulation_mean_variance = float(np.sum([gradient[4] ** 2 for gradient in percent])) / count ** 2
    reference_mean_variance = float(np.sum([gradient[3] ** 2 for gradient in percent])) / count ** 2

    def aggregate(key: str, variance: float) -> dict[str, Any]:
        errors = np.asarray([row[key] for row in point_records])
        _need(variance >= -1e-12, "aggregate uncertainty became negative")
        sigma = math.sqrt(max(0.0, variance))
        mean = float(np.mean(errors))
        return {
            "sample_count": len(errors), "mean_signed_error_mm": mean,
            "rms_observed_error_mm": float(np.sqrt(np.mean(errors ** 2))),
            "max_abs_observed_error_mm": float(np.max(np.abs(errors))),
            "standard_uncertainty_mm": sigma, "expanded_uncertainty_k2_mm": 2.0 * sigma,
            "interval_mm": {"lower": mean - 2.0 * sigma, "upper": mean + 2.0 * sigma,
                            "model": "mean +/- k=2 propagated standard uncertainty; linearized, not an exact coverage guarantee"},
        }

    summary = {
        "measured_vs_target": aggregate(
            "measured_minus_target_mm", physical_mean_variance + reference_mean_variance,
        ),
        "simulated_vs_target": aggregate(
            "simulated_minus_target_mm", simulation_mean_variance + reference_mean_variance,
        ),
        # Shared physical terms must not be averaged away as independent point
        # noise in a measured-vs-simulated difference either.
        "measured_vs_simulated": aggregate(
            "measured_minus_simulated_mm", physical_mean_variance + simulation_mean_variance,
        ),
        "variance_components_mm2": {
            "physical_mean_without_reference": physical_mean_variance,
            "simulation_mean_without_reference": simulation_mean_variance,
            "declared_reference_mean": reference_mean_variance,
        },
    }
    summary["uncertainty_model"] = {
        "pointwise": "first-order shared calibration and registration covariance, independent raw-point errors, and reference/simulation sigmas",
        "aggregate": "linear propagation of SHARED calibration/registration terms in the arithmetic mean; independent point variances averaged as 1/N^2",
        "paired_reference": "the same declared reference point/normal is used; its independent point uncertainty cancels in measured-vs-simulated differences",
        "simulated_sigma_meaning": "declared simulation positional standard uncertainty excluding reference-point uncertainty; not calibrated force",
        "unknown_regime": "not an autocorrelation-aware spatial-field interval, exact CMM coverage proof, or nonlinear confidence certificate",
    }
    return {
        "reference": reference_source, "measured": measured_source, "simulated": simulation_source,
        "simulation_provenance": declaration.get("simulation_provenance"),
        "samples": point_records, "aggregate": summary,
        "coverage": "complete_for_declared_samples" if registration["accepted"] else "aligned_diagnostics_not_accepted",
        "target_fit_used": False, "sample_ids_discarded": [],
    }


def _force(root: Path, raw: Mapping[str, Any], packet: Mapping[str, Any], dialect: str) -> dict[str, Any]:
    spindle = _map(raw.get("spindle_instrument"), "spindle_instrument")
    _text(spindle, "instrument_id", "spindle_instrument")
    _text(spindle, "signal_kind", "spindle_instrument")
    calibration = _text(spindle, "calibration_state", "spindle_instrument")
    _need(calibration in ("uncalibrated_indicator", "calibrated_indicator"),
          "spindle-instrument calibration must be declared")
    if calibration == "calibrated_indicator":
        _text(spindle, "certificate_id", "spindle_instrument")
    force_record = raw.get("calibrated_force")
    if force_record is None:
        measured_force = None
    else:
        record = _map(force_record, "calibrated_force")
        _text(record, "instrument_id", "calibrated_force")
        _text(record, "certificate_id", "calibrated_force")
        _need(record.get("unit") == "N", "calibrated force must have SI unit N")
        path, source = _resource(root, record.get("samples"), "calibrated_force.samples")
        with np.load(path, allow_pickle=False) as payload:
            _need("force_n" in payload and "force_covariance_n2" in payload and "time_s" in payload,
                  "calibrated force NPZ needs force_n, force_covariance_n2 and time_s")
            forces = _vec(payload["force_n"], (len(payload["time_s"]), 3), "force_n")
            covariance = _vec(payload["force_covariance_n2"], (len(forces), 3, 3), "force covariance")
            times = _vec(payload["time_s"], (len(forces),), "force timestamps")
        _need(len(forces) > 0 and np.all(np.diff(times) >= 0), "force samples need increasing timestamps")
        for index in range(len(forces)):
            _cov(covariance[index], 3, f"force covariance at index {index}")
        _need(record.get("sample_independence") == "declared_independent",
              "averaged calibrated-force uncertainty requires declared independent samples")
        mean_covariance = np.sum(covariance, axis=0) / len(forces) ** 2
        measured_force = {
            "source": source, "instrument_id": record["instrument_id"],
            "certificate_id": record["certificate_id"], "unit": "N",
            "sample_count": len(forces), "mean_vector_n": forces.mean(axis=0).tolist(),
            "mean_covariance_n2": mean_covariance.tolist(),
            "source_is_calibrated_dynamometer": True,
            "not_inferred_from_spindle_load": True,
        }
    proxy = packet[dialect].get("force_proxy")
    return {
        "calibrated_force_n": measured_force,
        "spindle_load": {
            "instrument": spindle, "unit": "percent_of_controller_rating",
            "calibrated_force_conversion": None,
            "note": "spindle-load indicator is not calibrated force; raw per-segment values stay in controller log",
        },
        "simulation_proxy": {
            "record": proxy, "calibrated_force_conversion": None,
            "note": "simulation proxy is never substituted for a physical force measurement",
        },
    }


def ingest_trial(packet_dir: str | Path, trial_file: str | Path) -> dict[str, Any]:
    """Verify packet identity first, then ingest one synthetic or physical trial."""
    packet_root = Path(packet_dir)
    trial_path = Path(trial_file)
    raw = _map(_json(trial_path), "trial")
    identity, program, task, motion = _identity(packet_root, trial_path.parent, raw)
    # Identity is deliberately complete before these data/calibration payload reads.
    origin = _map(raw.get("provenance"), "provenance")
    _need(isinstance(origin.get("synthetic"), bool), "provenance.synthetic must be an explicit boolean")
    _text(origin, "origin", "provenance")
    if origin["synthetic"]:
        _need(origin.get("origin") != "measured_physical_trial",
              "synthetic input cannot declare measured physical provenance")
    else:
        _need(origin.get("origin") == "measured_physical_trial",
              "empirical inputs need actual measured_physical_trial provenance, not an implicit default")
    terminal = _map(raw.get("termination"), "termination")
    state = terminal.get("state")
    _need(state in ("complete", "interrupted", "rejected"), "unsupported trial termination state")
    _text(terminal, "reason", "termination")
    pairing_policy = raw.get("pairing_policy", "unknown_cross_trial_covariance")
    _need(pairing_policy in ("unknown_cross_trial_covariance",
                             "declared_independent_calibration_and_acquisition"),
          "pairing_policy must declare unknown covariance or explicitly independent acquisition/calibration")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "provenance": origin,
        "termination": terminal,
        "pairing_policy": pairing_policy,
        "packet_program_id": program["run_id"],
        "task_hash": task.task_hash,
        "accepted_for_matched_comparison": False,
        "physical_validation_pending": True,
    }
    try:
        result["controller"] = _timing(trial_path.parent, raw, motion, terminal)
        calibration = _position_calibration(raw)
        result["position_calibration"] = calibration
        try:
            registration, _, _, _ = _registration(trial_path.parent, raw, calibration)
            result["registration"] = registration
        except DataQualityError as error:
            result["registration"] = error.diagnostics
            raise
        if identity["dialect"] not in program["evaluations"]:
            raise MeasurementError("selected dialect lacks evaluation")
        simulation_link = _map(_map(raw.get("surfaces"), "surfaces").get("simulation_provenance"),
                               "surfaces.simulation_provenance")
        posted_hash = _file_digest(packet_root / program["evaluations"][identity["dialect"]]["production_record"])
        _need(simulation_link.get("evaluation_sha256") == posted_hash,
              "simulation surface source does not match selected packet posted-evaluator hash")
        result["simulation_provenance"] = simulation_link
        result["surfaces"] = _surface(trial_path.parent, raw, calibration, registration)
        result["force_channels"] = _force(trial_path.parent, raw, program["evaluations"], identity["dialect"])
        accepted = terminal["state"] == "complete" and result["controller"]["coverage"] == "complete"
        result["accepted_for_matched_comparison"] = accepted
        if terminal["state"] == "rejected":
            result["processing_status"] = "retained_operator_rejected_trial"
        elif accepted:
            result["processing_status"] = "ingested_complete"
        else:
            result["processing_status"] = "retained_interrupted_partial_diagnostics"
        result["rejection_reason"] = None if accepted else terminal["reason"]
    except DataQualityError as error:
        result["processing_status"] = "registration_rejected"
        result["rejection_reason"] = str(error)
        result["surfaces"] = None
        result["accepted_for_matched_comparison"] = False
    result["synthetic"] = origin["synthetic"]
    result["empirical_eligibility"] = {
        "eligible": bool(not origin["synthetic"] and origin["origin"] == "measured_physical_trial" and
                         result["accepted_for_matched_comparison"]),
        "synthetic_input": origin["synthetic"],
        "physical_origin": origin["origin"],
        "not_a_machine_or_operator_authorization": True,
    }
    return result


def compare_trials(baseline: Mapping[str, Any], optimized: Mapping[str, Any]) -> dict[str, Any]:
    """Compare paired measurements without inventing cross-trial covariance."""
    result: dict[str, Any] = {"schema_version": "physical-paired-comparison/1"}
    if not (baseline["accepted_for_matched_comparison"] and optimized["accepted_for_matched_comparison"]):
        result.update({"status": "blocked", "reason": "rejected/interrupted/partial trials remain recorded; no accepted paired result",
                       "baseline_status": baseline["processing_status"],
                       "optimized_status": optimized["processing_status"]})
        return result
    _need(baseline["identity"]["role"] == "baseline" and optimized["identity"]["role"] == "optimized",
          "paired comparison must be baseline then optimized")
    for key in ("task_hash", "tool_setup_hash", "dialect", "packet_manifest_sha256"):
        _need(baseline["identity"][key] == optimized["identity"][key], f"paired identity differs at {key}")
    _need(baseline["identity"]["pairing_key"] == optimized["identity"]["pairing_key"],
          "paired trials require a predeclared common pairing_key")
    left = baseline["surfaces"]["samples"]
    right = optimized["surfaces"]["samples"]
    _need(baseline["surfaces"]["reference"] == optimized["surfaces"]["reference"],
          "paired comparisons require exactly the same hashed reference surface")
    _need([row["sample_id"] for row in left] == [row["sample_id"] for row in right],
          "paired measured samples must have exactly the same declared order and IDs")
    deltas = []
    for a, b in zip(left, right):
        delta = b["measured_minus_target_mm"] - a["measured_minus_target_mm"]
        deltas.append({
            "sample_id": a["sample_id"], "optimized_minus_baseline_mm": delta,
            "variance_if_declared_independent_mm2": (
                a["physical_variance_without_reference_mm2"] + b["physical_variance_without_reference_mm2"]
            ),
        })
    difference = float(np.mean([row["optimized_minus_baseline_mm"] for row in deltas]))
    baseline_policy = baseline.get("pairing_policy", {})
    optimized_policy = optimized.get("pairing_policy", {})
    shared_uncertainty = None
    if (baseline_policy == optimized_policy == "declared_independent_calibration_and_acquisition" and
            baseline["position_calibration"]["certificate_id"] != optimized["position_calibration"]["certificate_id"]):
        variance = (
            baseline["surfaces"]["aggregate"]["variance_components_mm2"]["physical_mean_without_reference"]
            + optimized["surfaces"]["aggregate"]["variance_components_mm2"]["physical_mean_without_reference"]
        )
        shared_uncertainty = math.sqrt(variance)
    explanation = (
        "computed only with predeclared independent calibration/acquisition records"
        if shared_uncertainty is not None else
        "cross-trial calibration/registration covariance is unavailable; observed paired difference is not promoted to an uncertainty-qualified win"
    )
    result.update({
        "status": "paired", "pairing_key": baseline["identity"]["pairing_key"],
        "orientation": "optimized minus baseline; no preferred sign is a success criterion",
        "optimized_minus_baseline_mean_mm": difference,
        "optimized_minus_baseline_standard_uncertainty_mm": shared_uncertainty,
        "optimized_minus_baseline_expanded_uncertainty_k2_mm": None if shared_uncertainty is None else 2.0 * shared_uncertainty,
        "uncertainty_status": explanation,
        "samples": deltas, "reference_uncertainty_cancellation": "same declared reference samples/normals, not a globally aligned free fit",
    })
    return result


def _empirical_payload(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate provenance and build the report before creating any output."""
    trials = list(trials)
    _need(trials, "empirical report needs actual trial records")
    by_role = {}
    for trial in trials:
        provenance = trial.get("provenance", {})
        _need(provenance.get("synthetic") is False and provenance.get("origin") == "measured_physical_trial",
              "empirical reports refuse synthetic, synthetic-only, or unspecified-provenance input")
        _need(trial.get("synthetic") is False, "empirical reports refuse synthetic-only input")
        for key in ("operator_setup_review_id", "operator_trial_log_id"):
            _text(trial.get("identity", {}), key, f"trial.identity[{key}]")
        _text(trial.get("position_calibration", {}), "certificate_id", "calibration")
        role = trial["identity"]["role"]
        _need(role not in by_role, f"multiple trial records for role {role}")
        by_role[role] = trial
    comparisons = None
    if set(by_role) == {"baseline", "optimized"}:
        comparisons = compare_trials(by_role["baseline"], by_role["optimized"])
    complete = comparisons is not None and comparisons["status"] == "paired"
    record = {
        "schema_version": REPORT_SCHEMA,
        "physical_evidence_origin": "measured_physical_trial",
        "empirical_status": "matched_observations_ingested" if complete else "incomplete_physical_evidence_preserved",
        "measured_comparison_available": complete,
        "trial_records": trials,
        "paired_comparison": comparisons,
        "rejected_interrupted_incomplete_counts": {
            "rejected_or_interrupted": sum(trial["termination"]["state"] in ("interrupted", "rejected") for trial in trials),
            "not_accepted_for_matched_comparison": sum(not trial["accepted_for_matched_comparison"] for trial in trials),
        },
        "limitations": "operator/device identity and calibration IDs remain traceable records, not cryptographic proof of physical reality; unmeasured/unknown scopes remain unknown",
    }
    return record


def empirical_report(trials: Sequence[Mapping[str, Any]], output: str | Path) -> dict[str, Any]:
    """Refuse synthetic-only/missing provenance; preserve failed physical trials."""
    record = _empirical_payload(trials)
    path = Path(output)
    _need(not path.exists(), f"report output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json_bytes(record))
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--trial", type=Path, action="append", required=True,
                        help="trial descriptor with identity role baseline or optimized; may repeat for a pair")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--empirical-report", action="store_true",
                        help="refuses synthetic/missing-provenance inputs; failed real trials are retained")
    args = parser.parse_args(argv)
    try:
        _need(not args.output_dir.exists(), "output directory must be new")
        records = [ingest_trial(args.packet, path) for path in args.trial]
        payloads = {}
        for record in records:
            role = record["identity"]["role"]
            _need(f"{role}-measurement.json" not in payloads,
                  f"duplicate trial role {role}; refusing to replace an existing trial record")
            payloads[f"{role}-measurement.json"] = record
        pairs = [record for record in records if record["identity"]["role"] in ("baseline", "optimized")]
        by_role = {record["identity"]["role"]: record for record in pairs}
        if set(by_role) == {"baseline", "optimized"}:
            payloads["paired-comparison.json"] = compare_trials(by_role["baseline"], by_role["optimized"])
        if args.empirical_report:
            payloads["empirical-report.json"] = _empirical_payload(records)
    except MeasurementError as error:
        parser.exit(2, f"measurement rejected: {error}\n")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, record in sorted(payloads.items()):
        (args.output_dir / name).write_bytes(_json_bytes(record))
    summary = {
        "records": len(records), "accepted": sum(row["accepted_for_matched_comparison"] for row in records),
        "synthetic": any(row["synthetic"] for row in records),
        "physical_validation_pending": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
