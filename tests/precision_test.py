"""T3.3c declared simulator precision.

The default is f32 for exploration; f64 is used for the comparison protocol.
Each precision is isolated by process where required because Taichi default_fp
is runtime-wide. The pilot
prefix driver is the read/reviewed T3.3b shared reference; numeric comparisons
below are against its reported repeated values, not a manufacturing result,
complete objective validation, or a new performance study.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi as ti

from simulator.csg_simulator import CSGSimulatorDelta


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_STOCK = "sha256:dffae7dc632bfaed2f593c7910949e523fd290e28ca0dd5beb0773557a29e246"
REFERENCE_TARGET = "sha256:473e90d99f7157b21b42707ace7c6882d88256bc0dba7f124fbaa1c6a7d9d6db"


def tiny(precision="f32"):
    return CSGSimulatorDelta(
        resolution=4, max_steps=1, stock_size_mm=4.0, voxel_size_mm=1.0,
        target_shape="box", init_taichi=False, precision=precision,
    )


def test_default_f32_metadata_dtype_and_mismatch_is_rejected_before_fields():
    from simulator import csg_simulator as module
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f32)
    # A rejected precision mismatch must not allocate/register another CSG.
    # Repeated runtime initializations make the absolute count unsuitable as a
    # universal delta across this suite, so compare exactly around the rejection.
    before_rejection = dict(module._TAICHI_RUNTIME_CSG_STATE)
    with pytest.raises(RuntimeError, match="one precision per process"):
        tiny("f64")
    assert dict(module._TAICHI_RUNTIME_CSG_STATE) == before_rejection
    for invalid in (None, "f16", "float64", "mixed"):
        with pytest.raises(ValueError, match="precision must"):
            tiny(invalid)


def _write(path, document):
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _run_reference_prefix(precision: str, output_dir: Path, *, steps=48):
    """Execute the owned physical-prefix protocol in a fresh child process.

    Only the declared precision changes. Actual source fields and typed paths
    are constructed and checked by this worktree, with no external helper path.
    """
    code = """
import importlib.util, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('precision_worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._reference_prefix_child(sys.argv[2], pathlib.Path(sys.argv[3]), steps=int(sys.argv[4]))
"""
    environment = dict(os.environ)
    # Inherit the caller's isolated TI/UV caches. Never pin a peer worker's cache.
    environment["DIFFCAM_ADFD_SMOOTHMAX"] = "canonical"
    environment["DIFFCAM_ADFD_AXIS"] = "2"
    environment.pop("TI_ARCH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, __file__, precision, str(output_dir), str(steps)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=1200,
    )
    (output_dir.parent / f"{precision}-{steps}-stdout.txt").write_text(completed.stdout)
    (output_dir.parent / f"{precision}-{steps}-stderr.txt").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    return json.loads((output_dir / "prefix.json").read_text())


class DirectionalResidual:
    """The original T2.1 theta_2 direction through a physical prefix."""

    def __init__(self, base_deltas, direction_mm, initial_position_mm):
        self.base = np.array(base_deltas, dtype=np.float64, copy=True)
        self.direction_mm = np.array(direction_mm, dtype=np.float64, copy=True)
        self.initial_position_mm = np.array(initial_position_mm, dtype=np.float64, copy=True)
        self.num_steps = self.base.shape[0]
        self.num_params = 1
        self.scale_mm = 1.0
        self.bounds = None

    def scalar(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.size != 1 or not np.isfinite(theta).all():
            raise ValueError("one finite original-theta_2 scalar is required")
        return float(theta.reshape(-1)[0])

    def in_bounds(self, theta):
        self.scalar(theta)
        return True

    def forward(self, theta):
        return self.base + self.scalar(theta) * self.direction_mm

    def positions(self, theta):
        return np.vstack([self.initial_position_mm,
                          self.initial_position_mm + np.cumsum(self.forward(theta), axis=0)])

    def pullback(self, gradient):
        return np.array([np.sum(np.asarray(gradient, dtype=np.float64) * self.direction_mm)])


def _reference_prefix_child(precision: str, output_dir: Path, *, steps: int):
    """Self-contained reviewed 48-segment-prefix protocol in a fresh process."""
    from dataclasses import replace
    from algorithms.objectives import CandidateEvaluator, ObjectiveSpec
    from algorithms.run_artifacts import ComputeLedger, source_metadata
    from benchmarks import contracts as C
    from benchmarks import pilot_task
    from cam.seeds import pocket_with_island_seed
    from scripts.run_gradient_mechanics import _actual_path, _write_source

    if precision not in ("f32", "f64") or not 1 <= steps <= 75:
        raise ValueError("one declared f32/f64 runtime and a physical prefix of 1..75 segments")
    output_dir.mkdir(parents=True, exist_ok=False)
    task = pilot_task.build_pilot_task()
    full_motion = pocket_with_island_seed(task)
    full_parameterization = pilot_task.fit_parameterization(full_motion)
    base = np.zeros(full_parameterization.num_params, dtype=np.float64)
    probe = base.copy()
    probe[2] = 0.001
    deltas = full_parameterization.forward(base)
    direction = (full_parameterization.forward(probe) - full_parameterization.forward(-probe)) / 0.002
    parameterization = (
        full_parameterization if steps == full_parameterization.num_steps else
        DirectionalResidual(deltas[:steps], direction[:steps], full_parameterization.initial_position_mm)
    )

    def theta_at(value):
        if steps == full_parameterization.num_steps:
            point = np.zeros(full_parameterization.num_params, dtype=np.float64)
            point[2] = value
            return point
        return np.array([value], dtype=np.float64)

    # Exact first-N physical direction, not a different truncated task or
    # discarded endpoint. Five shared-driver scale checks include both signs and
    # both smaller and larger than the FD epsilon values.
    for value in (0.0, 0.025, -0.025, 0.5, -0.5):
        full_theta = np.zeros(full_parameterization.num_params)
        full_theta[2] = value
        np.testing.assert_allclose(
            parameterization.positions(theta_at(value)),
            full_parameterization.positions(full_theta)[:steps + 1], atol=2e-12, rtol=0.0,
        )
    stock, target, required, retained = pilot_task.source_fields(task)
    assert not np.any(required & retained)
    stock_source = _write_source(output_dir, "stock", stock, task.grid, task.stock.params)
    target_source = _write_source(output_dir, "target", target, task.grid,
                                  task.operation_target.geometry.params)
    evaluated_task = replace(
        task, stock=stock_source,
        operation_target=replace(task.operation_target, geometry=target_source),
    )
    motion = replace(full_motion, segments=full_motion.segments[:steps],
                     provenance=f"t33c-owned-physical-prefix-{steps}")
    motion.validate_for_task(evaluated_task)
    normalized_start = ((parameterization.initial_position_mm - np.asarray(task.grid.origin_mm))
                        / np.asarray(task.grid.extent_mm))
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f32 if precision == "f32" else ti.f64)
    sim = CSGSimulatorDelta(
        max_steps=steps, stock_size_mm=task.grid.extent_mm,
        stock_box_origin_mm=task.grid.origin_mm, voxel_size_mm=task.grid.voxel_mm,
        target_shape="box", target_geometry=target_source, stock_geometry=stock_source,
        tool_start=normalized_start.tolist(), enforce_speed_limits=False,
        init_taichi=False, k_init=40.0, precision=precision,
    )
    for name, value in (
        ("tool_radius", task.tool.cutter_radius_mm),
        ("tool_height", task.tool.flute_length_mm),
        ("tool_cut_height", task.tool.flute_length_mm),
        ("holder_radius", task.tool.holder_radius_mm),
        ("holder_height", task.tool.holder_height_mm),
    ):
        getattr(sim, name)[None] = value
    sim.bake_target_grid()
    evaluator = CandidateEvaluator(
        evaluated_task, motion, parameterization, sim,
        spec=ObjectiveSpec(k=40.0), ledger=ComputeLedger(),
    )
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    ad = evaluator.evaluate_candidate_with_gradient(theta_at(0.0), f"prefix-{steps}-native-ad")
    elapsed = time.perf_counter() - started
    assert ad.result.valid and ad.gradient_theta.shape == (parameterization.num_params,), ad.result.invalid_reason
    assert np.all(np.isfinite(ad.gradient_theta))
    assert _actual_path(evaluator, theta_at(0.0))["same_within_tolerance"]
    record = {
        "declaration": {
            "steps": steps, "axis": 2, "role": "physical prefix; NOT a completion, safety or optimization result",
            "parameterization_mode": "original full T2.1 DeltaSpline" if steps == 75 else "owned prefix scalar direction with first-N exact positions",
            "direction_z_mm": direction[:steps].tolist(),
            "raw_direction_mm": direction[:steps].tolist(),
            "source_stock_sha": stock_source.sha256,
            "source_target_sha": target_source.sha256,
            "source_task_hash": evaluated_task.task_hash,
            "simulator_instances_this_process": 1,
        },
        "ad": {"objective": ad.result.objective, "gradient": float(ad.gradient_theta[2] if steps == 75 else ad.gradient_theta[0])},
        "fd": [],
    }
    for epsilon in (0.5, 0.025, 0.00625):
        values = {}
        for sign in (-1, 1):
            theta = theta_at(sign * epsilon)
            result = evaluator.evaluate_candidate(theta, f"prefix-{steps}-fd-{epsilon}-{sign}", "smooth")
            assert result.valid and _actual_path(evaluator, theta)["same_within_tolerance"], result.invalid_reason
            values[sign] = float(result.objective)
        record["fd"].append({
            "epsilon": epsilon, "plus_objective": values[1], "minus_objective": values[-1],
            "central": (values[1] - values[-1]) / (2 * epsilon),
            "left": (ad.result.objective - values[-1]) / epsilon,
            "right": (values[1] - ad.result.objective) / epsilon,
        })
    expected_dtype = ti.f32 if precision == "f32" else ti.f64
    assert sim.loss.dtype == sim.stock.dtype == sim.tool_delta.grad.dtype == expected_dtype
    record["native_precision"] = sim.precision_metadata()
    record["declared_known_field_bytes"] = sim.estimated_known_field_bytes()
    record["single_gradient_measurement"] = {
        "wall_time_s": [elapsed], "peak_rss_kib_before": [before],
        "peak_rss_kib_after": [resource.getrusage(resource.RUSAGE_SELF).ru_maxrss],
        "scope": "one forward+backward candidate followed by forward-only FD calls; not optimizer runtime",
        "peak_rss_scope": "whole-process cumulative peak including one simulator/compiler and data, not field-only memory",
    }
    record["shared_driver_args_match_except_named_precision"] = {
        "steps": steps, "smoothing": "canonical", "runtime_backend": "cpu",
        "ti_init_call_count_by_this_driver": 1, "csgsimulator_instances": 1,
        "input_deltas": (
            "canonical external normalized deltas converted by "
            "algorithms.run_artifacts.simulator_array to the selected simulator dtype; "
            "f64 preserves the float64 theta/parameterization path"
        ),
        "source_geometry": "np.float32 source arrays with immutable source hashes; selected simulator fields carry declared dtype",
    }
    record["owner_protocol"] = "self-contained in tests/precision_test.py, derived from reviewed T3.3b prefix; no external helper worktree is executed"
    record["source"] = source_metadata(ROOT)
    record["ledger_counts"] = {
        key: evaluator.ledger.snapshot()[key]
        for key in ("candidate_evaluations", "forward_calls", "backward_calls", "validation_calls")
    }
    _write(output_dir / "prefix.json", record)



def test_full_f64_matches_shared_prefix_index_two_direction(tmp_path):
    output = tmp_path / "prefix-48-f64-native"
    report = _run_reference_prefix("f64", output, steps=48)
    declaration = report["declaration"]
    assert declaration["source_stock_sha"] == REFERENCE_STOCK
    assert declaration["source_target_sha"] == REFERENCE_TARGET
    assert declaration["direction_z_mm"] == declaration["raw_direction_mm"]
    assert declaration["steps"] == 48
    epsilon = next(row for row in report["fd"] if row["epsilon"] == 0.00625)
    gradient = report["ad"]["gradient"]
    assert gradient == pytest.approx(-0.34076470567421824, abs=5e-7)
    # True-f64 theta/tool-delta inputs at epsilon=0.00625. At f6569fc, the isolated
    # legacy-input control (normalized float32 cast restored) reproduced
    # -0.34076627324900244; the un-narrowed f64 input path produced this value.
    # Evidence: /tmp/diffcam-t41b-fd-cast-control.QNU6Ln/prefix-48-f64-legacy-cast/.
    # The separate AD/FD agreement threshold below is deliberately unchanged.
    assert epsilon["central"] == pytest.approx(-0.3407654171104779, abs=5e-7)
    # Reference delta 1.57e-6, with a tight 2e-6 bound from worker-3's protocol.
    assert abs(gradient - epsilon["central"]) < 2e-6
    assert report["native_precision"]["requested_precision"] == "f64"
    assert report["shared_driver_args_match_except_named_precision"]["steps"] == 48


def test_default_f32_prefix_remains_the_shared_accumulation_baseline(tmp_path):
    output = tmp_path / "prefix-48-f32-default"
    report = _run_reference_prefix("f32", output, steps=48)
    assert report["native_precision"]["requested_precision"] == "f32"
    assert report["declaration"]["source_stock_sha"] == REFERENCE_STOCK
    assert report["declaration"]["source_target_sha"] == REFERENCE_TARGET
    assert report["ad"]["gradient"] == pytest.approx(-0.33225263288349316, abs=1.6e-5)
    epsilon = next(row for row in report["fd"] if row["epsilon"] == 0.025)
    assert epsilon["central"] == pytest.approx(-0.3407251834869385, abs=2e-6)
    # Keep float32's known substantial reverse-bias; this is NOT a bitwise
    # cross-process reproducibility certificate or a claim of f64 accuracy.
    observed_relative = abs(report["ad"]["gradient"] - epsilon["central"]) / abs(epsilon["central"])
    assert observed_relative == pytest.approx(0.025, abs=0.001)
    assert report["ad"]["gradient"] != pytest.approx(epsilon["central"], abs=0.002)


@pytest.mark.skipif(os.environ.get("DIFFCAM_PRECISION_COST") != "1",
                    reason="explicit full-pilot-grid candidate cost pair; set DIFFCAM_PRECISION_COST=1 and run alone")
def test_full_pilot_grid_single_gradient_cost_pair(tmp_path):
    root = tmp_path / "precision-cost"
    root.mkdir()
    reports = []
    for precision in ("f32", "f64"):
        output = root / f"whole-75-{precision}"
        if output.exists():
            pytest.fail(f"cost output already exists: {output}")
        report = _run_pilot_precision_job(precision, output)
        assert report["gradient_gate"]["longest_declared_path_segments"] == 75
        assert report["task_and_geometry"]["stock_sha256"] == REFERENCE_STOCK
        assert report["task_and_geometry"]["target_sha256"] == REFERENCE_TARGET
        if precision == "f64":
            assert report["gradient_gate"]["all_declared_checks_pass_1_percent"]
        reports.append(report)
    factor = (reports[1]["known_field_bytes"]["stock_history_and_adjoint_bytes"]
              / reports[0]["known_field_bytes"]["stock_history_and_adjoint_bytes"])
    assert factor == pytest.approx(2.0)
    assert all(record["cost"]["one_gradient_wall_time_s"] and
               record["cost"]["whole_process_peak_rss_bytes_after_gradient"] for record in reports)
    summary = {
        "status": "complete_one_gradient_each_plus_reference_forward_finite_differences",
        "production_precision_decision": "full f64 for optimizer comparisons; f32 remains exploration-only",
        "measured_on": "CPU; shared-box contention not controlled",
        "reports": reports,
    }
    _write(root / "cost-summary.json", summary)


def _pilot_precision_child(precision: str, output_dir: Path):
    """One full 75-segment pilot, with only the declared precision changed.

    The existing gradient-mechanics driver supplies the real hard warm-up,
    one smooth AD, forward replay, selected FD and hard base. No optimization.
    The declared coordinate gate corpus is {2,35}; others are not claimed as
    FD-qualified merely because their AD values were computed by one Tape.
    """
    from algorithms import objectives as objectives_module
    from scripts import run_gradient_mechanics as driver

    capture = {"sim": None, "parameterization": None, "gradient_times": [], "rss_before": [], "rss_after": []}
    original_init = ti.init
    from simulator import csg_simulator as backend
    original_simulator_class = backend.CSGSimulatorDelta
    original_driver_simulator = getattr(driver, "CSGSimulatorDelta", None)

    def init_matching(**kwargs):
        kwargs["default_fp"] = ti.f32 if precision == "f32" else ti.f64
        return original_init(**kwargs)

    ti.init = init_matching
    driver.CSGSimulatorDelta = functools.partial(CSGSimulatorDelta, precision=precision)
    backend.CSGSimulatorDelta = driver.CSGSimulatorDelta
    actual_builder = driver.build_pilot_evaluator

    def capture_builder(*args, **kwargs):
        task, motion, parameterization, evaluator, metadata = actual_builder(*args, **kwargs)
        capture["sim"] = evaluator.sim
        capture["parameterization"] = parameterization
        return task, motion, parameterization, evaluator, metadata

    driver.build_pilot_evaluator = capture_builder
    original_gradient = objectives_module.CandidateEvaluator.evaluate_candidate_with_gradient

    def measured_gradient(self, *args, **kwargs):
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        started = time.perf_counter()
        try:
            return original_gradient(self, *args, **kwargs)
        finally:
            capture["gradient_times"].append(time.perf_counter() - started)
            capture["rss_before"].append(before)
            capture["rss_after"].append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    objectives_module.CandidateEvaluator.evaluate_candidate_with_gradient = measured_gradient
    config = driver.CostConfig(pairs=2, reference_pairs=8, seed=0, sigma_theta=0.05,
                               epsilon_theta=(0.025, 0.00625), fd_indices=(2, 35), voxel_mm=0.25)
    try:
        report = driver.run_pilot(output_dir, config=config)
    finally:
        objectives_module.CandidateEvaluator.evaluate_candidate_with_gradient = original_gradient
        ti.init = original_init
        driver.build_pilot_evaluator = actual_builder
        backend.CSGSimulatorDelta = original_simulator_class
        if original_driver_simulator is None:
            driver.__dict__.pop("CSGSimulatorDelta", None)
        else:
            driver.CSGSimulatorDelta = original_driver_simulator
    sim = capture["sim"]
    assert sim is not None
    expected_dtype = ti.f32 if precision == "f32" else ti.f64
    assert sim.loss.dtype == expected_dtype
    assert sim.stock.dtype == expected_dtype and sim.tool_delta.grad.dtype == expected_dtype
    precision_identity = sim.precision_metadata()
    known_bytes = sim.estimated_known_field_bytes()
    smooth = report["smooth_ad_fd"]
    ad_gradient = smooth["ad_gradient"]
    coordinate_rows = []
    epsilon_records = {}
    for record in smooth["fd"]:
        assert tuple(record["evaluated_indices"]) == (2, 35)
        for index in (2, 35):
            epsilon_records.setdefault(float(record["epsilon"][index]), []).append((record, index))
    for epsilon in (0.025, 0.00625):
        records = epsilon_records[epsilon]
        assert len(records) == 2
        for record, index in records:
            ad = float(ad_gradient[index])
            fd_value = record["gradient"][index]
            assert isinstance(fd_value, (float, int)) and np.isfinite(fd_value), {
                "undefined_FD_coordinate": index, "status": record["status"],
            }
            fd = float(fd_value)
            absolute = abs(ad - fd)
            floor = max(abs(fd), 1e-6)
            coordinate_rows.append({
                "index": index, "epsilon": epsilon, "ad": ad, "fd": fd,
                "absolute_difference": absolute, "relative_error": absolute / floor,
                "noise_floor": 1e-6, "relative_error_limit": 0.01,
                "passed_1_percent": absolute / floor <= 0.01,
            })
    gate = {
        "longest_declared_path_segments": len(driver.pocket_with_island_seed(
            driver.pilot_task.build_pilot_task(voxel_mm=0.25)).segments),
        "declared_gate_indices": [2, 35],
        "number_of_finite_difference_coordinate_checks": len(coordinate_rows),
        "noise_floor": 1e-6, "relative_error_limit": 0.01,
        "total_parameter_coordinates": int(capture["parameterization"].num_params),
        "ungraded_coordinate_scope": "remaining 34 coordinates are AD-only diagnostic, not a complete 36-coordinate FD validation",
        "all_declared_checks_pass_1_percent": all(row["passed_1_percent"] for row in coordinate_rows),
        "rows": coordinate_rows,
        "definition": "|AD-FD| / max(|FD|, 1e-6), from the same actual CandidateEvaluator smooth objective and typed paths; no external ground-truth bound",
    }
    non_pilot = [
        {"task_id": bundle.task.task_id, "segments": len(bundle.motion.segments),
         "grid": list(bundle.task.grid.shape)}
        for bundle in __import__("benchmarks.procedural_suite", fromlist=["generate_tasks"]).generate_tasks()
    ]
    candidate_lengths = [
        {"case": "T2.2b-pilot-pocket-nominal", "segments": 75},
        *non_pilot,
        {"case": "Titan-M1-nominal", "segments": 4, "note": "four-segment partial roughing path, not a complete finishing program"},
    ]
    assert gate["longest_declared_path_segments"] == max(row["segments"] for row in candidate_lengths) == 75
    result = {
        "schema": "native-simulator-precision-check/1",
        "precision": precision_identity,
        "known_field_bytes": known_bytes,
        "task_and_geometry": report["task_and_geometry"],
        "gradient_gate": gate,
        "candidate_lengths_in_scope": candidate_lengths,
        "titan": {
            "native_step_sha256_validated": True,
            "nominal_status": "blocked_missing_pinned_CAD_interpreter",
            "reason": "parts/Titan-M1.step was found, but external/cad-venv/bin/python is missing; no interpolated Titan substitute was run",
        },
        "source_probe_report": str(output_dir / "gradient_mechanics.json"),
        "cost": {
            "one_gradient_wall_time_s": capture["gradient_times"],
            "first_gradient_includes_ti_compile_and_backward": True,
            "whole_process_peak_rss_bytes_before_gradient": [value * 1024 for value in capture["rss_before"]],
            "whole_process_peak_rss_bytes_after_gradient": [value * 1024 for value in capture["rss_after"]],
            "scope": "whole-process cumulative Linux ru_maxrss, not field-only or incremental allocator memory; shared-host timing, not a universal performance factor",
            "report_wrapper": "outer native metadata/dtypes are authoritative; the source driver's legacy generic float32 string is not precision-aware",
            "selected_coordinate_forward_count": 2 * 2 * 2,
        },
        "reference_scope": "real full pilot with original hard warm-ups, one forward+backward, a forward replay, 8 finite-difference calls on declared coordinates, and hard base; no optimization or held-out solver selection",
    }
    _write(output_dir / "precision-gradient.json", result)
    if precision == "f64" and not gate["all_declared_checks_pass_1_percent"]:
        raise AssertionError(f"f64 longest-path coordinate gate failed: {coordinate_rows}")
    return result


def _run_pilot_precision_job(precision: str, output_dir: Path) -> dict:
    code = """
import importlib.util, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('precision_worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._pilot_precision_child(sys.argv[2], pathlib.Path(sys.argv[3]))
"""
    environment = dict(os.environ)
    # Inherit the caller's isolated TI/UV caches. Never pin a peer worker's cache.    environment.pop("TI_ARCH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, __file__, precision, str(output_dir)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=1200,
    )
    (output_dir.parent / f"full-pilot-{precision}-stdout.txt").write_text(completed.stdout)
    (output_dir.parent / f"full-pilot-{precision}-stderr.txt").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    return json.loads((output_dir / "precision-gradient.json").read_text())


@pytest.mark.skipif(os.environ.get("DIFFCAM_PRECISION_CURVE") != "1",
                    reason="explicit repeated physical-prefix bias curve; set DIFFCAM_PRECISION_CURVE=1 and run alone")
def test_bias_versus_physical_prefix_length(tmp_path):
    """Only original theta_2 is measured for the entire physical prefix curve."""
    root = tmp_path / "prefix-bias-curve"
    root.mkdir()
    rows = []
    for precision in ("f32", "f64"):
        for steps in (8, 36, 48, 75):
            output = root / f"{precision}-{steps}"
            report = _run_reference_prefix(precision, output, steps=steps)
            assert report["declaration"]["source_stock_sha"] == REFERENCE_STOCK
            assert report["declaration"]["source_target_sha"] == REFERENCE_TARGET
            assert report["shared_driver_args_match_except_named_precision"]["steps"] == steps
            ad = float(report["ad"]["gradient"])
            epsilon = next(row for row in report["fd"] if row["epsilon"] == 0.00625)
            fd = float(epsilon["central"])
            absolute = abs(ad - fd)
            relative = absolute / max(abs(fd), 1e-6)
            assert np.isfinite(ad) and np.isfinite(fd)
            row = {
                "precision": precision, "segments": steps,
                "coordinate": "theta_2_of_the_original_full_75_segment_trajectory",
                "path_normalization": "physical prefix, not a redefined or interpolated truncated task",
                "ad": ad, "fd": fd, "epsilon": 0.00625,
                "absolute_difference": absolute, "relative_error": relative,
                "relative_error_limit": 0.01, "noise_floor": 1e-6,
                "source_stock_sha256": report["declaration"]["source_stock_sha"],
                "source_target_sha256": report["declaration"]["source_target_sha"],
                "artifact": str(output / "prefix.json"),
                "passes_f64_one_percent_gate": relative <= 0.01 if precision == "f64" else None,
            }
            if precision == "f64":
                assert relative <= 0.01, row
            rows.append(row)
    document = {
        "schema": "t33c-physical-prefix-bias/1",
        "definition": "|AD(theta_2)-central_FD(theta_2)| / max(|FD|, 1e-6) at epsilon=.00625, not a geometric ground-truth or all-coordinate certificate",
        "rows": rows,
        "gate": "each reported f64 point must satisfy <=1%; f32 retained as a negative comparison, never promoted",
        "remaining_coordinates": "coordinate 35 and all untested coordinates are outside this prefix-curve scope; the separate full-pilot index gate measures 2 and 35",
        "titan": "four-segment nominal is shorter than pilot/T2.3; pinned CAD interpreter unavailable, with no fabricated longer substitute",
    }
    _write(root / "bias-curve.json", document)


def test_f64_staged_stock_roundtrip_and_candidate_reset():
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f64)
    sim = tiny("f64")
    original = np.linspace(-2.0, 3.0, 64, dtype=np.float64).reshape(
        sim.Nx, sim.Ny, sim.Nz,
    )
    assert np.any(original != original.astype(np.float32).astype(np.float64))

    sim.load_saved_init(original, [.2, .3, .4])
    restored = sim.saved_stock.to_numpy()
    assert restored.dtype == np.float64
    np.testing.assert_array_equal(restored, original)
    assert int(sim.use_saved_init[None]) == 1
    np.testing.assert_allclose(sim.tool_start.to_numpy(), [.2, .3, .4], rtol=0, atol=1e-15)

    sim.reset_candidate_state()
    assert int(sim.use_saved_init[None]) == 0
    np.testing.assert_array_equal(sim.saved_stock.to_numpy(), np.zeros_like(original))
    assert sim.saved_stock.dtype == ti.f64
    np.testing.assert_allclose(sim.tool_start.to_numpy(), [.5, .5, 1.0], rtol=0, atol=1e-15)
    assert sim.precision_metadata()["requested_precision"] == "f64"