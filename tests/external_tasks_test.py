"""Unit tests for external-task adapters; CAD evidence is separately opt-in.

The default tests use explicit synthetic arrays for contract/mask behavior.
Those arrays never establish an actual STEP import, a native swept-volume
result, or a passing machining evaluation. Only the opt-in real-backend test
uses the isolated OCP environment and the exact local Titan source hash.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import contracts as C
from benchmarks import external_tasks as E
from benchmarks import titan_setups as T
from cam.gcode_parser import parse_motion_plan
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from cam.units import inch_to_mm

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_SHAPE = (86, 44, 15)  # rectangular physical stock, not a forced cube grid


def _source(path: Path, values: np.ndarray, grid: C.GridSpec) -> C.GeometrySource:
    np.save(path, np.asarray(values, dtype=np.float64), allow_pickle=False)
    return C.GeometrySource(
        kind="sampled", path=str(path),
        sha256=C.sha256_bytes(path.read_bytes()), grid=grid,
    )


def _synthetic_task(tmp_path: Path) -> C.TaskSpec:
    """Schema fixture only, explicitly not real Titan geometry evidence.

    The profile is an independent, deliberately coarse small-bore raster. It
    proves the constructor cannot accidentally omit the checker in unit tests;
    it does not establish an actual Titan/CAD source-common-volume proof.
    """
    grid = T.pilot_grid()
    first_layer_z = grid.bounds_mm[0][2] + 1.5 * grid.voxel_mm
    top_z = grid.bounds_mm[1][2]
    native = {
        "operation_geometry": {
            "source_to_work_mm": [0.0, 0.0, 0.0],
            "hole_centre_source_xy_mm": list(grid.cell_centre_mm(0, 0, 0)[:2]),
            "exclusion": {
                "plug_radius_mm": 0.8,
                "plug_min_source_z_mm": grid.bounds_mm[0][2],
                "plug_max_source_z_mm": top_z,
                "nominal_void_profile": [
                    {"radius_mm": 0.7, "source_z_min_mm": grid.bounds_mm[0][2],
                     "source_opening_z_mm": first_layer_z,
                     "face_z_range_mm": [grid.bounds_mm[0][2], first_layer_z],
                     "nominal_cylinder_common_with_original_part_mm3": 0.0},
                    {"radius_mm": 0.6, "source_z_min_mm": first_layer_z,
                     "source_opening_z_mm": top_z, "face_z_range_mm": [first_layer_z, top_z],
                     "nominal_cylinder_common_with_original_part_mm3": 0.0},
                ],
            },
        },
    }
    original = np.full(grid.shape, -0.5, dtype=np.float64)
    original[0, 0, :] = 0.5
    operation = original.copy()
    operation[0, 0, :] = -0.5
    # A separate, non-bore air cell is a nonzero material-removal target.
    original[1, 1, 0] = 0.5
    operation[1, 1, 0] = 0.5
    bore_check = T.independent_bore_crosscheck(
        native, grid, T.box_signed_field(grid, *T.stock_bounds()), original, operation)
    assert bore_check["nominal_mask_match"] is True
    assert bore_check["expected_cells"] == grid.shape[2]
    return T.build_task(
        _source(tmp_path / "part.npy", original, grid),
        _source(tmp_path / "target.npy", operation, grid),
        native_excluded_volume_mm3=None,
        independent_bore_check=bore_check,
    )


def test_registered_titan_builder_rejects_missing_independent_bore_check(tmp_path):
    grid = T.pilot_grid()
    original = np.full(grid.shape, -0.5)
    target = original.copy()
    with pytest.raises(ValueError, match="independent bore cross-check"):
        T.build_task(_source(tmp_path / "missing-check-part.npy", original, grid),
                     _source(tmp_path / "missing-check-target.npy", target, grid),
                     independent_bore_check=None)


def test_normal_parent_import_never_imports_heavy_cad():
    assert not E._IS_WORKER
    assert not any(name == "OCP" or name.startswith("OCP.") for name in sys.modules)


def test_inches_are_converted_at_the_boundary_with_exact_multiplier():
    assert inch_to_mm(0.195) == pytest.approx(4.953)
    assert T.REFERENCE_RADIUS_MM == pytest.approx(4.953)
    assert T.HOLE_RADII_MM == pytest.approx((1.0287, 2.38125))
    assert max(T.HOLE_RADII_MM) < T.nominal_tool().cutter_radius_mm


def test_physical_grid_stock_top_centre_and_tool_reach():
    grid = T.pilot_grid()
    lo, hi = T.stock_bounds()
    assert grid.shape == EXPECTED_SHAPE
    assert grid.origin_mm == lo
    np.testing.assert_allclose(grid.bounds_mm[0], lo, atol=1e-12, rtol=0)
    np.testing.assert_allclose(grid.bounds_mm[1], hi, atol=1e-12, rtol=0)
    assert hi[2] == 0
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0)
    assert (lo[1] + hi[1]) / 2 == pytest.approx(0)
    tool = T.nominal_tool()
    assert tool.cutter_radius_mm == pytest.approx(3.175)
    assert lo[2] + tool.flute_length_mm > hi[2] + 1.0
    assert lo[2] + tool.holder_bottom_mm > hi[2] + 1.0
    transform = C.Transform.translation(T.WORK_TO_MACHINE_MM)
    assert transform.apply((0, 0, 0)) == T.WORK_TO_MACHINE_MM
    assert transform.inverse().apply(transform.apply((1, 2, 3))) == pytest.approx((1, 2, 3))
    assert T.SOURCE_TO_WORK_MM == (-50.8, -24.13, 0.0)
    work = np.array(T.HOLE_SOURCE_XY_MM) + np.array(T.SOURCE_TO_WORK_MM[:2])
    np.testing.assert_allclose(work, [-46.99, -20.32], atol=1e-12, rtol=0)


def test_incompatible_grid_pitch_is_rejected_not_rescaled():
    with pytest.raises(ValueError, match="integral physical grid extents"):
        T.pilot_grid(0.5)


def test_box_field_is_in_physical_mm_at_cell_centres():
    grid = C.GridSpec(shape=(2, 1, 1), origin_mm=(2, 10, -1), voxel_mm=2.0)
    values = T.box_signed_field(grid, (2, 10, -1), (4, 12, 1))
    np.testing.assert_allclose(values, [[[-1.0]], [[1.0]]], atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        T.box_signed_field(grid, (2, 10, -1), (2, 12, 1))


def test_native_field_payload_is_float64_hashed_and_complete(tmp_path):
    payload = {
        "part": [-1.25, 0.5],
        "operation_target": [-1.25, -0.25],
    }
    for name, values in payload.items():
        (tmp_path / f"{name}.f64le").write_bytes(struct.pack("<2d", *values))
    result = E.load_native_fields(tmp_path, (2, 1, 1))
    for name, expected in payload.items():
        record = result[name]
        assert record["values"].dtype == np.float64
        assert record["values"].shape == (2, 1, 1)
        assert float(record["values"][0, 0, 0]) == expected[0]
        assert float(record["values"][1, 0, 0]) == expected[1]
        digest = hashlib.sha256(record["path"].read_bytes()).hexdigest()
        assert record["sha256"] == "sha256:" + digest
        loaded = np.load(record["path"], allow_pickle=False)
        np.testing.assert_array_equal(loaded, record["values"])


@pytest.mark.parametrize("defect", ["length", "nan", "shape"])
def test_invalid_native_fields_fail_closed(tmp_path, defect):
    values = (float("nan"), 1.0) if defect == "nan" else (-1.0, 2.0)
    (tmp_path / "part.f64le").write_bytes(struct.pack("<2d", *values))
    (tmp_path / "operation_target.f64le").write_bytes(struct.pack("<d", -1.0))
    with pytest.raises((E.CADValidationError, ValueError)):
        if defect == "shape":
            E.load_native_fields(tmp_path, (1, True, 2))
        else:
            E.load_native_fields(tmp_path, (2, 1, 1))


def test_cell_masks_keep_original_excluded_and_removal_regions_distinct():
    stock = -np.ones((2, 2, 2), dtype=np.float64)
    original = stock.copy()
    original[0, 0, 0] = 0.5
    target = original.copy()
    target[0, 0, 0] = -2.0
    original[0, 0, 1] = 1.0
    target[0, 0, 1] = 1.0
    result = E.field_statistics(stock, original, target, 2.0)
    assert result["excluded_cells_initial_and_T_not_P"] == 1
    assert result["excluded_volume_voxel_estimate_mm3"] == pytest.approx(8.0)
    assert result["removal_target_cells_initial_not_T"] == 1
    assert result["removal_target_volume_voxel_estimate_mm3"] == pytest.approx(8.0)
    assert result["original_part_subset_of_operation_target_at_cell_centres"] is True
    target[1, 1, 1] = 1.0
    with pytest.raises(E.CADValidationError, match="subset"):
        E.field_statistics(stock, original, target, 2.0)
    stock[1, 1, 1] = 1.0
    target[1, 1, 1] = -1.0
    with pytest.raises(E.CADValidationError, match="outside declared stock"):
        E.field_statistics(stock, original, target, 2.0)


def test_task_preserves_sampled_sources_and_unrecorded_provenance(tmp_path):
    task = _synthetic_task(tmp_path)
    assert task.split == C.Split.EXTERNAL
    assert task.source.license == "unknown / provenance unrecorded"
    assert task.source.checksums["parts/Titan-M1.step"] == "sha256:" + E.TITAN_M1_SHA256
    assert task.original_part.kind == "sampled"
    assert task.operation_target.geometry.kind == "sampled"
    assert task.original_part.path != task.operation_target.geometry.path
    assert task.original_part.path and task.operation_target.geometry.path
    assert Path(task.original_part.path).is_file()
    assert Path(task.operation_target.geometry.path).is_file()
    assert task.original_part.params.get("primitive") is None
    assert task.operation_target.excluded_volume_mm3 == pytest.approx(15 * task.grid.voxel_mm ** 3)
    check = task.operation_target.geometry.params["independent_bore_verification"]
    assert check["nominal_mask_match"] is True
    assert check["mismatched_cells"] == 0
    assert check["expected_cells"] == check["actual_cells"] == 15
    assert "1.0287" in task.operation_target.exclusion_reason
    assert "3.175" in task.operation_target.exclusion_reason
    assert "not certified accessible" in task.operation_target.exclusion_reason


def test_exclusion_is_grid_resolved_with_native_volume_diagnostic_only(tmp_path):
    fixture = _synthetic_task(tmp_path)
    grid = fixture.grid
    original = np.load(fixture.original_part.path, allow_pickle=False)
    target = np.load(fixture.operation_target.geometry.path, allow_pickle=False)
    task = T.build_task(
        _source(tmp_path / "grid-original.npy", original, grid),
        _source(tmp_path / "grid-target.npy", target, grid),
        # Deliberately different synthetic diagnostic, not evidence of a native
        # volume measurement. The task declaration must come from the masks.
        native_excluded_volume_mm3=123.456,
        independent_bore_check=fixture.operation_target.geometry.params[
            "independent_bore_verification"
        ],
    )
    expected = 15 * grid.voxel_mm ** 3
    assert task.operation_target.excluded_volume_mm3 == pytest.approx(expected)
    geometry = task.operation_target.geometry
    assert geometry.params["grid_declared_excluded_volume_mm3"] == pytest.approx(expected)
    assert geometry.params["native_excluded_volume_mm3_diagnostic"] == pytest.approx(123.456)
    assert geometry.params["excluded_volume_convention"].endswith("tolerance 0")
    assert "grid-resolved" in task.operation_target.exclusion_reason


def test_nominal_line_motion_has_noncutting_rapids_and_vertical_retract(tmp_path):
    task = _synthetic_task(tmp_path)
    motion = T.nominal_motion(task)
    motion.validate_for_task(task)
    assert len(motion.segments) == 4
    assert [segment.kind for segment in motion.segments] == [
        C.SegmentKind.RAPID, C.SegmentKind.FEED, C.SegmentKind.RAPID, C.SegmentKind.RAPID,
    ]
    inverse = task.work_to_machine.inverse().apply
    for rapid in (motion.segments[0], motion.segments[3]):
        assert inverse(rapid.start_mm)[2] > 0
        assert inverse(rapid.end_mm)[2] > 0
    feed = motion.segments[1]
    assert inverse(feed.start_mm) == pytest.approx((*T.NOMINAL_XY_WORK_MM, T.APPROACH_Z_MM))
    assert inverse(feed.end_mm) == pytest.approx((*T.NOMINAL_XY_WORK_MM, T.NOMINAL_END_Z_MM))
    assert feed.duration_s == pytest.approx(
        (T.APPROACH_Z_MM - T.NOMINAL_END_Z_MM) / T.FEED_MM_S
    )
    # The upward rapid starts at the cleared feed endpoint, then precedes the
    # lateral safe-height return. It must not be mislabeled as starting above stock.
    assert motion.segments[2].start_mm == motion.segments[1].end_mm
    assert inverse(motion.segments[2].start_mm)[:2] == pytest.approx(T.NOMINAL_XY_WORK_MM)
    assert inverse(motion.segments[2].end_mm)[2] == pytest.approx(T.SAFE_Z_MM)
    assert inverse(motion.segments[3].start_mm)[2] == inverse(motion.segments[3].end_mm)[2] == pytest.approx(T.SAFE_Z_MM)


def test_line_comparison_records_tags_but_does_not_waive_kind_or_geometry(tmp_path):
    motion = T.nominal_motion(_synthetic_task(tmp_path))
    segments = list(motion.segments)
    segments[0] = dataclasses.replace(segments[0], tag="plunge")
    relabelled = dataclasses.replace(motion, segments=tuple(segments))
    comparison = T.compare_line_motion(motion, relabelled)
    assert comparison["max_position_error_mm"] == 0
    assert len(comparison["free_form_tag_differences"]) == 1
    assert comparison["free_form_tag_differences"][0]["kind"] == "rapid"
    segments[1] = dataclasses.replace(segments[1], kind=C.SegmentKind.RAPID, feed_mm_s=None)
    with pytest.raises(AssertionError, match="kind/feed"):
        T.compare_line_motion(motion, dataclasses.replace(motion, segments=tuple(segments)))


def _bore_metadata():
    """Synthetic runtime metadata only; no claim of a native CAD proof."""
    return {
        "operation_geometry": {
            "source_to_work_mm": [0.0, 0.0, 0.0],
            "hole_centre_source_xy_mm": [1.27, 1.27],
            "exclusion": {
                "plug_radius_mm": 3.175,
                "plug_min_source_z_mm": -1.27,
                "plug_max_source_z_mm": 1.905,
                "nominal_void_profile": [
                    {
                        "radius_mm": 1.0287, "source_z_min_mm": -1.27,
                        "source_opening_z_mm": 1.905,
                        "face_z_range_mm": [-1.27, -0.0414],
                        "nominal_cylinder_common_with_original_part_mm3": 0.0,
                    },
                    {
                        "radius_mm": 2.38125, "source_z_min_mm": -0.0414,
                        "source_opening_z_mm": 1.905,
                        "face_z_range_mm": [-0.0414, 1.905],
                        "nominal_cylinder_common_with_original_part_mm3": 0.0,
                    },
                ],
            },
        }
    }


def _bore_grid():
    return C.GridSpec(shape=(2, 2, 4), origin_mm=(0.0, 0.0, -1.27), voxel_mm=1.27)


def _bore_actual_fields():
    # Independently declared actual masks: P is all air, T retains the three
    # lower bore layers, and the fourth layer is outside T (removal region).
    part = np.ones((2, 2, 4), dtype=np.float64)
    target = part.copy()
    target[:, :, :3] = -1.0
    initial = np.full((2, 2, 4), -1.0, dtype=np.float64)
    return initial, part, target


def test_independent_nominal_bore_raster_matches_cell_locations():
    grid = _bore_grid()
    result = T.independent_bore_crosscheck(_bore_metadata(), grid, *_bore_actual_fields())
    assert result["expected_cells"] == result["actual_cells"] == 12
    assert result["mismatched_cells"] == 0
    assert result["nominal_mask_match"] is True
    assert result["expected_volume_mm3"] == pytest.approx(12 * grid.voxel_mm ** 3)
    assert np.all(result["expected_mask"][:, :, :3])
    assert not np.any(result["expected_mask"][:, :, 3])
    expected_inside = np.zeros(grid.shape, dtype=bool)
    expected_inside[:, :, :3] = True
    assert np.array_equal(result["expected_mask"], expected_inside)
    resolution = result["resolution"]
    assert resolution["small_bore_cells_across_diameter"] == pytest.approx(2.0574 / 1.27)
    assert resolution["small_bore_pitch_for_eight_cells_mm"] == pytest.approx(0.257175)
    assert resolution["small_bore_under_resolved"] is True


def test_nominal_bore_radius_perturbations_and_source_negative_guard():
    grid = _bore_grid()
    actual = _bore_actual_fields()
    metadata = _bore_metadata()
    result = T.independent_bore_crosscheck(metadata, grid, *actual, include_perturbations=True)
    by_name = {item["name"]: item for item in result["perturbations"]}
    assert by_name["primary_radius_0.875_mm"]["expected_cells"] == 8
    assert by_name["primary_radius_0.875_mm"]["mismatched_cells"] == 4
    assert by_name["primary_radius_0.875_mm"]["actual_mask_mismatch_rejected"] is True
    expanded = by_name["primary_radius_times_1.25"]
    assert expanded["expected_cells"] == 12
    assert expanded["mismatched_cells"] == 0
    assert expanded["detectable_at_this_pitch"] is False
    invalid = _bore_metadata()
    invalid["operation_geometry"]["exclusion"]["nominal_void_profile"][0][
        "nominal_cylinder_common_with_original_part_mm3"
    ] = 0.5
    with pytest.raises(ValueError, match="not source-negative"):
        T.independent_bore_crosscheck(invalid, grid, *actual)


def test_relative_sampled_paths_resolve_from_repository_root(tmp_path, monkeypatch):
    task = _synthetic_task(tmp_path)
    motion = T.nominal_motion(task)
    nc = post_motion_plan(motion, task, dialect=C.Dialect.RS274).encode("utf-8")
    source = task.original_part
    relative = os.path.relpath(source.path, PROJECT)
    assert not os.path.isabs(relative)
    other = tmp_path / "unrelated-working-directory"
    other.mkdir()
    monkeypatch.chdir(other)
    task = dataclasses.replace(task, original_part=dataclasses.replace(source, path=relative))
    result = evaluate_posted_gcode(task, nc, C.Dialect.RS274, {
        "expected_nc_hash": C.nc_hash(nc), "subdivisions": 4,
    })
    assert result.coverage == C.Coverage.COMPLETE
    assert result.metrics["residual_volume_mm3"].value is not None
    assert result.efficiency_eligible is False


def test_relative_path_does_not_fall_back_to_process_cwd(tmp_path, monkeypatch):
    task = _synthetic_task(tmp_path)
    motion = T.nominal_motion(task)
    nc = post_motion_plan(motion, task, dialect=C.Dialect.RS274).encode("utf-8")
    relative = "payload-that-exists-only-under-cwd.npy"
    candidate = tmp_path / relative
    np.save(candidate, np.load(task.original_part.path, allow_pickle=False), allow_pickle=False)
    assert candidate.is_file()  # the unit fixture created it
    assert not (PROJECT / relative).exists()
    source = dataclasses.replace(task.original_part, path=relative)
    task = dataclasses.replace(task, original_part=source)
    monkeypatch.chdir(tmp_path)
    result = evaluate_posted_gcode(task, nc, C.Dialect.RS274, {
        "expected_nc_hash": C.nc_hash(nc), "subdivisions": 4,
    })
    assert result.coverage == C.Coverage.UNSUPPORTED
    assert result.safety_pass is None and result.geometry_pass is None
    assert str(PROJECT / relative) in result.coverage_reasons[0]


def test_actual_production_evaluator_supports_exact_grid_sampled_geometry(tmp_path):
    task = _synthetic_task(tmp_path)
    motion = T.nominal_motion(task)
    for dialect in (C.Dialect.RS274, C.Dialect.HAAS):
        nc = post_motion_plan(motion, task, dialect=dialect)
        parsed = parse_motion_plan(nc, task, dialect=dialect)
        comparison = T.compare_line_motion(motion, parsed)
        assert comparison["line_segment_count"] == 4
        assert comparison["max_position_error_mm"] <= 0.002
        posted = C.loads(C.dumps(motion))
        assert posted == motion
        result = evaluate_posted_gcode(task, nc.encode(), dialect, {
            "expected_nc_hash": C.nc_hash(nc.encode()), "subdivisions": 4,
        })
        assert result.coverage == C.Coverage.COMPLETE
        assert result.geometry_pass is False
        assert result.metrics["residual_volume_mm3"].value is not None
        assert result.metrics["residual_volume_mm3"].value > 0
        assert result.metrics["gouge_volume_mm3"].value is not None
        assert result.metrics["excluded_volume_mm3"].value == pytest.approx(
            15 * task.grid.voxel_mm ** 3
        )
        assert result.efficiency_eligible is False
        if dialect == C.Dialect.RS274:
            assert "G0 X-20.32 Y0 Z5" in nc
            assert "G1 X-20.32 Y0 Z-9.525 F150" in nc
        else:
            assert "T1 M06" in nc
        assert "G43 H1" in nc
        assert "G21" in nc
    assert "G20" not in nc


def test_fine_bore_sampling_cost_is_explicitly_bounded():
    grid = T.pilot_grid(0.254)
    cells = math.prod(grid.shape)
    assert grid.shape == (430, 220, 75)
    assert cells == 7_095_000
    assert cells > 200_000  # current Native sampler's bounded cell cap
    assert 2 * cells * 8 > 100_000_000  # two raw f64 fields before work arrays
    source_seconds = 0.000926173501966332
    target_seconds = 0.0009712408573250286
    estimated_seconds = cells * (source_seconds + target_seconds)
    assert estimated_seconds > 2 * 3600
    assert T.GRID_PITCH_MM == 1.27
    assert 2.0574 / 8 < T.GRID_PITCH_MM  # eight-cell fine pitch was not chosen


def test_missing_cad_environment_is_explicitly_blocked(tmp_path):
    with pytest.raises(E.CADUnavailable, match="^blocked: cad env unavailable"):
        E.cad_request({"operation": "environment"}, interpreter=tmp_path / "not-a-python")


class _StubProcess:
    def __init__(self, record, timeout=False):
        self.record = record
        self.timeout = timeout
        self.returncode = 1 if record.get("status") != "ok" else 0
        self.killed = False
        self.calls = 0

    def communicate(self, payload=None, timeout=None):
        self.calls += 1
        if self.timeout and self.calls == 1:
            raise subprocess.TimeoutExpired("stub", timeout)
        import json
        return json.dumps(self.record).encode(), b""

    def kill(self):
        self.killed = True


def test_protocol_failure_and_timeout_are_not_reported_as_passing(monkeypatch):
    for record, match, timeout in (
        ({"status": "error", "error_kind": "unavailable", "error": "wrong pinned runtime"},
         "wrong pinned runtime", False),
        ({"status": "error", "error_kind": "OSError", "error": "native import failed"},
         "native import failed", False),
        ({"status": "ok"}, "timed out", True),
    ):
        process = _StubProcess(record, timeout)
        monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
        expected = E.CADUnavailable if timeout or record.get("error_kind") == "unavailable" else E.CADValidationError
        with pytest.raises(expected, match=match):
            E.cad_request({"operation": "environment"}, interpreter=Path(__file__), timeout_s=0.01)
    assert process.killed


@pytest.mark.skipif(
    os.environ.get("DIFFCNCAD_REAL_TITAN") != "1",
    reason="opt-in DIFFCNCAD_REAL_TITAN=1; isolated pinned CAD env and local-only exact Titan source",
)
def test_real_step_sampling_nominal_and_actual_evaluator_outcomes(tmp_path):
    root = tmp_path / "real-titan-pilot"
    result = T.run_pilot(root)
    inspection = result["native"]["inspection"]
    assert inspection["source_sha256"] == E.TITAN_M1_SHA256
    assert inspection["native_model_unit"] == "mm"
    assert inspection["solid_count"] == 1
    assert inspection["brep_valid"] is True
    assert inspection["face_count"] == 67
    assert inspection["reference_dimension"]["max_abs_error_mm"] < 1e-6
    assert not result["native"]["environment"]["std_path_has_repository"]
    geometry = result["native"]["operation_geometry"]
    declared = result["task"].operation_target
    faces = geometry["hole_cylinders"]
    assert len(faces) == 2
    assert all(len(face["face_z_range_mm"]) == 2 for face in faces)
    assert all(np.isfinite(face["face_z_range_mm"]).all() for face in faces)
    bore = declared.geometry.params["independent_bore_verification"]
    assert bore["expected_cells"] == bore["actual_cells"] == 68
    assert bore["mismatched_cells"] == 0 and bore["nominal_mask_match"] is True
    assert bore["resolution"]["small_bore_under_resolved"] is True
    assert bore["resolution"]["small_bore_cells_across_diameter"] == pytest.approx(2.0574 / 1.27)
    perturbations = {entry["name"]: entry for entry in bore["perturbations"]}
    assert perturbations["primary_radius_0.875_mm"]["expected_cells"] == 12
    assert perturbations["primary_radius_0.875_mm"]["mismatched_cells"] == 56
    assert perturbations["primary_radius_times_1.25"]["mismatched_cells"] == 0
    assert perturbations["primary_radius_times_1.25"]["detectable_at_this_pitch"] is False
    assert all(face["nominal_cylinder_common_with_original_part_mm3"] == pytest.approx(0.0)
               for face in geometry["exclusion"]["nominal_void_profile"])
    assert declared.excluded_volume_mm3 == pytest.approx(139.290044)
    diagnostic = geometry["exclusion"]["native_excluded_volume_mm3"]
    assert declared.geometry.params["native_excluded_volume_mm3_diagnostic"] == pytest.approx(diagnostic)
    assert declared.geometry.params["grid_declared_excluded_volume_mm3"] == pytest.approx(139.290044)
    assert geometry["target"]["brep_valid"] is True
    sweep = result["native"]["native_nominal_sweep"]
    assert sweep["start_source_mm"][:2] == pytest.approx([30.48, 24.13])
    assert sweep["stock_common_volume_mm3"] > 0
    assert sweep["original_part_common_volume_mm3"] <= 1e-6
    assert sweep["operation_target_common_volume_mm3"] <= 1e-6
    assert result["sampled"]["original_part_subset_of_operation_target_at_cell_centres"]
    assert result["sampled"]["excluded_cells_initial_and_T_not_P"] > 0
    for name in ("rs274", "haas"):
        record = result["evaluations"][name]
        assert record["production"].coverage == C.Coverage.COMPLETE
        assert record["production"].geometry_pass is False
        assert record["production"].safety_pass is None
        assert record["production"].efficiency_eligible is False
        assert record["production"].metrics["residual_volume_mm3"].value > 0
        assert record["production"].metrics["gouge_volume_mm3"].value == 0
        assert record["production"].metrics["original_part_gouge_volume_mm3"].value == 0
        assert record["production"].metrics["excluded_volume_mm3"].value == pytest.approx(139.290044)
        assert record["reference"].result.coverage == C.Coverage.COMPLETE
        # Stock-chronology safety is intentionally retained whether true, false
        # or null; it is never converted into a physical safety certificate.
        assert record["reference"].result.geometry_pass is None
        assert record["grid_material_metrics"]["removed_volume_mm3"]["value"] > 0
        assert record["grid_material_metrics"]["gouge_volume_mm3"]["value"] == 0
        assert record["operational_geometry_pass_from_grid"] is False
        assert (root / f"production_{name}.json").is_file()
        assert (root / f"independent_{name}.json").is_file()
    assert (root / "hashes.json").is_file()
    assert not list(root.rglob("*.step")) and not list(root.rglob("*.stp"))
    assert result["status"] == "incomplete"
