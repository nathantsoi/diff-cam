"""T2.3 deterministic bounded families and their real nominal qualification.

The ``qualified_root`` fixture runs the generator CLI in a fresh directory and
actually evaluates all six seeds through both production posts and the
independent reference. Fixtures never synthesize a passing result. Geometric
minus-pocket checks below independently sample the public declared grid.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import audit as audit_reader
from benchmarks import contracts as C
from benchmarks import procedural_suite as suite
from benchmarks.pilot_task import (
    build_pilot_task, compare_motion, expected_positions_from_motion, fit_parameterization,
)
from cam.gcode_parser import parse_motion_plan
from cam.seeds import (
    IslandSeedSpec, pocket_with_island_seed, seed_deltas_mm, seed_for_task, seed_spec_for_task,
)


ACCEPTED_PILOT_MOTION_HASH = "sha256:565fcf62661859ce00392f5fd4673ebbdd88809c3d1bc54e808fef7162a8bb34"
# This exact one-dev/one-test census applies only to bootstrap/1. It is not a
# cap or count decision for a later T4.1b-sized suite.
BUNDLES = suite.generate_tasks()
BY_ID = {bundle.task.task_id: bundle for bundle in BUNDLES}
SCRIPT = suite.PROJECT_ROOT / "scripts" / "generate_task_suite.py"


def _geometry_masks(bundle):
    """Independently sample P and the four finite-radius corner exclusions."""
    task = bundle.task
    params = task.original_part.params
    grid = task.grid
    axes = [
        grid.origin_mm[a] + (np.arange(grid.shape[a]) + 0.5) * grid.voxel_mm
        for a in range(3)
    ]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")

    def box(lo, hi):
        return (
            (xx >= lo[0]) & (xx <= hi[0])
            & (yy >= lo[1]) & (yy <= hi[1])
            & (zz >= lo[2]) & (zz <= hi[2])
        )

    stock = box(params["box_min_mm"], params["box_max_mm"])
    pocket = box(params["pocket_min_mm"], params["pocket_max_mm"])
    pad = box(params["island_min_mm"], params["island_max_mm"])
    original = stock & (~pocket | pad)
    radius = task.tool.cutter_radius_mm
    excluded = np.zeros(task.grid.shape, dtype=bool)
    for sx in (-1, 1):
        for sy in (-1, 1):
            pocket_lo, pocket_hi = params["pocket_min_mm"], params["pocket_max_mm"]
            cx = pocket_hi[0] - radius if sx == 1 else pocket_lo[0] + radius
            cy = pocket_hi[1] - radius if sy == 1 else pocket_lo[1] + radius
            corner = box(
                (cx if sx == 1 else pocket_lo[0],
                 cy if sy == 1 else pocket_lo[1], pocket_lo[2]),
                (pocket_hi[0] if sx == 1 else cx,
                 pocket_hi[1] if sy == 1 else cy, pocket_hi[2]),
            )
            excluded |= corner & ((xx - cx)**2 + (yy - cy)**2 >= radius**2)
    return stock, original, excluded


def _run_cli(*args, timeout=120):
    process = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        cwd=suite.PROJECT_ROOT, env=dict(os.environ),
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    assert process.returncode == 0, (
        f"CLI exit {process.returncode}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
    )
    return process


@pytest.fixture(scope="session")
def qualified_root(tmp_path_factory):
    # One fixed six-case qualification, not an optimizer sweep or experiment.
    root = tmp_path_factory.mktemp("t23-qualified") / "bootstrap"
    process = _run_cli("--output-root", root, timeout=900)
    report = json.loads((root / "generation_report.json").read_text())
    assert report["status"] == "complete", process.stdout + process.stderr
    assert [row["task_id"] for row in report["tasks"]] == list(BY_ID)
    assert all(row["status"] == "qualified_and_retained" for row in report["tasks"])
    assert all(math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
               for row in report["tasks"])
    return root


def test_family_census_stable_manifest_and_split_designs():
    # Versioned bootstrap census only: future suite expansion gets a new
    # generator/schema and explicit count policy from T4.2, not this cap.
    assert Counter((bundle.task.family, bundle.task.split) for bundle in BUNDLES) == {
        ("facing_asymmetric", C.Split.DEV): 1,
        ("facing_asymmetric", C.Split.TEST): 1,
        ("pocket_island", C.Split.DEV): 1,
        ("pocket_island", C.Split.TEST): 1,
        ("variable_width_depth", C.Split.DEV): 1,
        ("variable_width_depth", C.Split.TEST): 1,
    }
    assert len({bundle.task.task_hash for bundle in BUNDLES}) == 6
    first = suite.manifest_text()
    assert first == suite.manifest_text()
    assert suite.verify_manifest()
    manifest = json.loads(first)
    assert manifest["generation_version"] == suite.GENERATION_VERSION
    assert manifest["generator_seed"] == suite.GENERATOR_SEED
    assert "pending pilot cost" in manifest["suite_count_policy"]
    assert [entry["task_id"] for entry in manifest["tasks"]] == list(BY_ID)
    for entry, bundle in zip(manifest["tasks"], BUNDLES):
        assert entry["task_hash"] == bundle.task.task_hash
        assert entry["motion_hash"] == bundle.motion.motion_hash
        assert entry["seed_spec"] == json.loads(json.dumps(bundle.seed_spec.as_dict()))
        assert entry["evaluator_exclusion_tolerance_mm3"] == 0.0
        assert C.loads(json.dumps(entry["task_record"])) == bundle.task
        assert entry["task_record"]["record"]["source"]["origin"] == bundle.task.source.origin
        assert bundle.task.source.origin == f"{suite.GENERATION_VERSION}:{bundle.topology}"
        assert bundle.task.source.license == "MIT"

    by_family_split = {(b.task.family, b.task.split): b for b in BUNDLES}
    facing_dev = by_family_split["facing_asymmetric", C.Split.DEV]
    facing_test = by_family_split["facing_asymmetric", C.Split.TEST]
    assert facing_dev.seed_spec.pocket_centre_xy != facing_test.seed_spec.pocket_centre_xy
    assert facing_dev.task.original_part.params != facing_test.task.original_part.params
    assert facing_dev.topology != facing_test.topology
    island_dev = by_family_split["pocket_island", C.Split.DEV]
    island_test = by_family_split["pocket_island", C.Split.TEST]
    assert island_dev.seed_spec.island_centre_xy != island_test.seed_spec.island_centre_xy
    assert (island_dev.task.original_part.params["island_max_mm"][2]
            != island_test.task.original_part.params["island_max_mm"][2])
    # The lower-pad test is a structural topology variant, not merely a rename.
    assert island_test.task.original_part.params["island_max_mm"][2] == -1.0
    widths_depths = set()
    for split in C.Split.DEV, C.Split.TEST:
        bundle = by_family_split["variable_width_depth", split]
        params = bundle.task.original_part.params
        width = params["pocket_max_mm"][0] - params["pocket_min_mm"][0]
        depth = params["pocket_max_mm"][2] - params["pocket_min_mm"][2]
        widths_depths.add((width, depth))
    assert widths_depths == {(8.0, 8.0), (16.0, 3.0)}


def test_source_containment_grid_and_physically_derived_exclusion():
    for bundle in BUNDLES:
        task = bundle.task
        grid = task.grid
        params = task.original_part.params
        assert task.original_part.kind == "analytic"
        assert task.original_part.frame == C.Frame.WORK
        assert params["primitive"] == "box_minus_pocket"
        assert task.stock.frame == C.Frame.WORK
        assert task.stock.params["min_mm"] == params["box_min_mm"]
        assert task.stock.params["max_mm"] == params["box_max_mm"]
        stock = (params["box_min_mm"], params["box_max_mm"])
        pocket = (params["pocket_min_mm"], params["pocket_max_mm"])
        pad = (params["island_min_mm"], params["island_max_mm"])
        for axis in range(3):
            assert stock[0][axis] <= pocket[0][axis] < pocket[1][axis] <= stock[1][axis]
        np.testing.assert_allclose(grid.bounds_mm, stock, atol=1e-12, rtol=0.0)
        assert grid.voxel_mm == suite.PITCH_MM == 0.25
        overlap = all(stock[0][a] < pad[1][a] and pad[0][a] < stock[1][a] for a in range(3))
        assert overlap == bundle.seed_spec.has_retained_island
        if not overlap:
            assert params["empty_island"] is True
            assert pad[0][2] > stock[1][2]
        else:
            assert pad[0][2] == pocket[0][2]
            assert pad[1][2] <= stock[1][2]
            for axis in (0, 1):
                assert pocket[0][axis] <= pad[0][axis] < pad[1][axis] <= pocket[1][axis]

        original_stock, original, excluded = _geometry_masks(bundle)
        target = original | excluded
        assert np.all(original <= original_stock)
        assert np.all(original <= target)
        assert not np.array_equal(original, target), "P and operation target T must be distinct"
        exclusion = int(np.count_nonzero(excluded))
        assert exclusion > 0
        assert exclusion * grid.voxel_mm**3 == task.operation_target.excluded_volume_mm3
        operation = task.operation_target.geometry.params
        assert operation["grid_excluded_voxels"] == exclusion
        assert operation["grid_exclusion_tolerance_mm3"] == 0.0
        depth = pocket[1][2] - pocket[0][2]
        continuous = 4.0 * (1.0 - math.pi / 4.0) * task.tool.cutter_radius_mm**2 * depth
        assert operation["continuous_excluded_volume_mm3"] == pytest.approx(continuous, abs=1e-12)
        # The continuous fillet amount is diagnostic, not an allowance used by
        # the evaluator. It and its zero grid tolerance were fixed pre-motion.
        holder = operation["holder_clearance"]
        assert task.tool.cutter_kind == C.CutterKind.FLAT
        assert holder["above_stock_mm"] >= holder["qualification_and_cell_margin_mm"]
        assert task.tool.holder_radius_mm >= task.tool.cutter_radius_mm


def test_source_motion_reaches_declared_walls_and_preserves_pad():
    for bundle in BUNDLES:
        task, plan, spec = bundle.task, bundle.motion, bundle.seed_spec
        plan.validate_for_task(task)
        assert plan.segments[0].start_mm == task.initial_state.position_mm
        assert plan.segments[-1].end_mm == task.initial_state.position_mm
        assert all(segment.arc is None for segment in plan.segments)
        for segment in plan.segments:
            if segment.kind == C.SegmentKind.RAPID:
                assert min(segment.start_mm[2], segment.end_mm[2]) > task.stock.params["max_mm"][2]
        parameterization = fit_parameterization(plan)
        theta = np.zeros((parameterization.num_control_points, 3))
        assert parameterization.num_control_points <= 12
        np.testing.assert_array_equal(parameterization.forward(theta), seed_deltas_mm(plan))
        np.testing.assert_allclose(
            parameterization.positions(theta), expected_positions_from_motion(plan),
            atol=1e-12, rtol=0.0,
        )
        floor_points = {}
        for segment in plan.segments:
            start = task.work_to_machine.inverse().apply(segment.start_mm)
            end = task.work_to_machine.inverse().apply(segment.end_mm)
            for point in (start, end):
                if point[2] == spec.floor_z:
                    floor_points[(round(float(point[0]), 9), round(float(point[1]), 9))] = point
        for x in (spec.outer_xy[0], -spec.outer_xy[0]):
            for y in (spec.outer_xy[1], -spec.outer_xy[1]):
                centre = (round(x + spec.pocket_centre_xy[0], 9),
                          round(y + spec.pocket_centre_xy[1], 9))
                assert centre in floor_points
        if spec.has_retained_island:
            finishing = [segment for segment in plan.segments if segment.tag == "island-finish"]
            assert len(finishing) == 4
            # A lower pad needs a cap pass at its physical top, not a cut at
            # pocket floor through retained material (original failed topology).
            pad_top = task.original_part.params["island_max_mm"][2]
            cap = [segment for segment in plan.segments if segment.tag.startswith("pad-cap-")
                   and segment.tag != "pad-cap-plunge" and segment.tag != "pad-cap-link"]
            if pad_top < task.stock.params["max_mm"][2]:
                assert cap
                assert all(segment.start_mm[2] == segment.end_mm[2] == pad_top for segment in cap)
            else:
                assert not cap


def test_cli_verify_only_does_not_evaluate_or_create_run_directories(tmp_path):
    manifest = tmp_path / "manifest.json"
    unused_output = tmp_path / "must-not-exist"
    process = _run_cli("--manifest-only", "--manifest", manifest, "--output-root", unused_output)
    assert "seed qualification NOT run" in process.stdout
    assert manifest.read_text() == suite.manifest_text()
    assert not unused_output.exists()
    _run_cli("--verify-only", "--manifest", manifest, "--output-root", unused_output)
    assert not unused_output.exists()
    manifest.write_text(manifest.read_text() + "\n")
    before = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        bundle_text = suite.manifest_text()
        suite.write_manifest(manifest, expected_text=bundle_text)
    assert manifest.read_bytes() == before
    changed = subprocess.run(
        [sys.executable, str(SCRIPT), "--verify-only", "--manifest", str(manifest),
         "--output-root", str(unused_output)],
        cwd=suite.PROJECT_ROOT, capture_output=True, text=True, check=False, timeout=120,
    )
    assert changed.returncode != 0
    assert "not reproducible byte-for-byte" in changed.stderr
    assert not unused_output.exists()
    with pytest.raises(ValueError):
        suite.make_task(replace(suite.SCENARIOS[0], stock=((0.0, 0.0), (1.0, 1.0))))


def test_unsupported_source_and_pad_geometries_fail_closed():
    bundle = BUNDLES[0]
    mesh = C.GeometrySource(
        kind="mesh", path="unused-seed-rejects-before-reading",
        sha256=C.sha256_bytes(b"synthetic invalid-input marker; never evaluated"),
    )
    non_analytic = replace(bundle.task, original_part=mesh)
    with pytest.raises(ValueError, match="analytic box-minus-pocket"):
        seed_spec_for_task(non_analytic)
    ball = replace(bundle.task, tool=replace(bundle.task.tool, cutter_kind=C.CutterKind.BALL))
    with pytest.raises(ValueError, match="flat cutter"):
        seed_spec_for_task(ball)
    pad = BUNDLES[3]
    params = dict(pad.task.original_part.params)
    params["island_min_mm"] = [0.5, -2.5, -4.0]
    stepped_floor = replace(pad.task, original_part=replace(pad.task.original_part, params=params))
    with pytest.raises(ValueError, match="flat-floor"):
        seed_spec_for_task(stepped_floor)
    rectangle_params = dict(params)
    rectangle_params["island_min_mm"] = [0.5, -2.5, -5.0]
    rectangle_params["island_max_mm"] = [4.5, 0.5, -1.0]
    rectangular = replace(pad.task, original_part=replace(pad.task.original_part, params=rectangle_params))
    with pytest.raises(ValueError, match="square retained"):
        seed_spec_for_task(rectangular)
    unmarked_empty = replace(
        bundle.task.original_part,
        params={key: value for key, value in bundle.task.original_part.params.items() if key != "empty_island"},
    )
    with pytest.raises(ValueError, match="mark empty_island"):
        seed_spec_for_task(replace(bundle.task, original_part=unmarked_empty))
    lying_island = replace(BUNDLES[3].task.original_part, params={
        **BUNDLES[3].task.original_part.params, "empty_island": True,
    })
    with pytest.raises(ValueError, match="empty_island"):
        seed_spec_for_task(replace(BUNDLES[3].task, original_part=lying_island))
    with pytest.raises(ValueError, match="centres disagree"):
        pocket_with_island_seed(BUNDLES[3].task, IslandSeedSpec())
    wrong_empty_flag = replace(BUNDLES[3].seed_spec, has_retained_island=False)
    with pytest.raises(ValueError, match="flag disagrees"):
        pocket_with_island_seed(BUNDLES[3].task, wrong_empty_flag)


def test_accepted_pilot_seed_is_preserved_exactly():
    task = build_pilot_task()
    expected = C.MotionPlan.from_dict(json.loads(C.dumps(pocket_with_island_seed(task, IslandSeedSpec())))["record"])
    actual = seed_for_task(task)
    assert actual.motion_hash == ACCEPTED_PILOT_MOTION_HASH == expected.motion_hash
    assert actual.segments == expected.segments
    assert len(actual.segments) == 75


def test_real_six_case_production_and_reference_artifacts(qualified_root):
    for bundle in BUNDLES:
        task = bundle.task
        root = qualified_root / task.task_id
        run = C.loads((root / "run.json").read_text())
        assert run.status == C.RunStatus.COMPLETE
        assert run.task_hash == task.task_hash
        hashes = json.loads((root / "hashes.json").read_text())
        assert {path.name for path in root.iterdir()} == set(hashes) | {"hashes.json"}
        assert set(hashes) == set(run.artifacts) | {"run.json"}
        for name, digest in hashes.items():
            assert C.sha256_bytes((root / name).read_bytes()) == digest
        assert C.loads((root / "task.json").read_text()) == task
        saved_motion = C.loads((root / "motion_plan.json").read_text())
        assert saved_motion == bundle.motion
        trajectory = np.load(root / "trajectory.npy", allow_pickle=False)
        initial_trajectory = np.load(root / "initial_trajectory.npy", allow_pickle=False)
        deltas = np.load(root / "trajectory_deltas.npy", allow_pickle=False)
        machine_waypoints = np.load(root / "machine_waypoints_mm.npy", allow_pickle=False)
        machine_deltas = np.load(root / "machine_deltas_mm.npy", allow_pickle=False)
        assert trajectory.shape[1] == deltas.shape[1] == 3
        assert trajectory.shape[0] == len(bundle.motion.segments) + 1
        assert np.all(np.isfinite(trajectory))
        # This nominal stores the complete initial seed, not a single-point
        # placeholder. The two representations have separate physical units.
        np.testing.assert_array_equal(initial_trajectory, trajectory)
        expected_machine = np.asarray(
            [bundle.task.initial_state.position_mm]
            + [segment.end_mm for segment in bundle.motion.segments]
        )
        np.testing.assert_allclose(machine_waypoints, expected_machine, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(machine_deltas, seed_deltas_mm(bundle.motion), atol=1e-6, rtol=0.0)
        inverse_work = bundle.task.work_to_machine.inverse()
        expected_work = np.asarray([inverse_work.apply(point) for point in expected_machine])
        normalized = (
            (expected_work - np.asarray(bundle.task.grid.origin_mm))
            / np.asarray(bundle.task.grid.extent_mm)
        )
        np.testing.assert_allclose(trajectory, normalized, atol=1e-7, rtol=0.0)
        inverse_rotation = np.asarray(inverse_work.matrix)[:3, :3]
        work_deltas = np.asarray(
            seed_deltas_mm(bundle.motion)
        ) @ inverse_rotation.T
        np.testing.assert_allclose(deltas, work_deltas / np.asarray(bundle.task.grid.extent_mm),
                                   atol=1e-6, rtol=0.0)
        np.testing.assert_array_equal(deltas, np.diff(trajectory, axis=0))
        config = json.loads((root / "config.json").read_text())
        assert config["trajectory_representation"]["normalized_formula"].startswith("(work_mm")
        for dialect in C.Dialect:
            nc_file, eval_file = C.DIALECT_ARTIFACTS[dialect]
            nc_bytes = (root / nc_file).read_bytes()
            result = C.loads((root / eval_file).read_text())
            assert result.dialect == dialect
            assert result.task_hash == task.task_hash
            assert result.tool_setup_hash == task.tool_setup_hash
            assert result.nc_hash == C.nc_hash(nc_bytes)
            assert result.coverage == C.Coverage.COMPLETE
            assert result.safety_pass is True
            assert result.geometry_pass is True
            assert result.metrics["residual_volume_mm3"].value <= task.tolerances.completion_residual_mm3
            assert result.metrics["gouge_volume_mm3"].value <= task.tolerances.gouge_volume_mm3
            assert result.metrics["max_surface_error_mm"].value <= task.tolerances.local_surface_error_mm
            assert result.metrics["min_clearance_lower_bound_mm"].value >= task.tolerances.clearance_mm
            assert result.metrics["rapid_cutter_penetration_upper_bound_mm"].value <= task.tolerances.rapid_penetration_mm
            assert result.discretization["settings"]["exclusion_tolerance_mm3"] == 0.0
            restored = parse_motion_plan(nc_bytes.decode("utf-8"), task, dialect=dialect)
            assert compare_motion(bundle.motion, restored)["semantic_tag_mismatches"] == 0
        reference = C.loads((root / "reference.json").read_text())
        assert reference.evaluator_id == "reference_carve"
        assert reference.coverage == C.Coverage.COMPLETE
        assert reference.safety_pass is True
        # Independent stock motion does not fabricate a custom target score.
        assert reference.geometry_pass is None
        assert reference.metrics["residual_volume_mm3"].value is None
        assert reference.metrics["gouge_volume_mm3"].value is None
        assert reference.metrics["min_clearance_lower_bound_mm"].value >= task.tolerances.clearance_mm
        assert reference.metrics["rapid_cutter_penetration_upper_bound_mm"].value <= task.tolerances.rapid_penetration_mm


def test_retained_bundle_passes_existing_project_auditor(qualified_root):
    # The temporary expected-run mapping identifies records for the existing
    # T4.0 reader. It neither freezes later suite counts nor makes a comparison.
    tasks, runs = {}, {}
    for bundle in BUNDLES:
        task = bundle.task
        run_id = f"seed-{task.task_id}"
        tasks[task.task_id] = audit_reader.TaskInfo(
            task.task_id, task.task_hash, task.family,
            task.tolerances.completion_residual_mm3,
        )
        runs[run_id] = audit_reader.ExpectedRun(
            run_id, task.task_id, "procedural-nominal", suite.GENERATOR_SEED,
        )
    records, report = audit_reader.audit_runs(
        qualified_root, audit_reader.Manifest(tasks, runs, ()),
    )
    assert [record["run_id"] for record in records] == sorted(runs)
    assert all(record["usable_for_statistics"] and not record["issues"] for record in records)
    assert report["reconciliation"]["exact_id_reconciliation"] is True
    assert report["summary"] == {
        "run_status_counts": {"complete": 6},
        "dialect_status_counts": {"complete": 12},
        "issue_counts": {},
        "auditor_passed": True,
        "evidence_complete": True,
        "all_runs_completed": True,
    }
