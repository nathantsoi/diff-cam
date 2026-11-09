"""T2.2 one-pocket task, nominal motion, and artifact tests.

Only ``test_real_nominal_pipeline_and_evidence`` executes the simulator and
both production/reference evaluators as end-to-end evidence, then replays the
independent stock-motion reference. ``_synthetic_posts`` is for serialization
tests only, not a simulator or a nominal safety result.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import contracts as C
from benchmarks.pilot_task import (
    CORNER_RADIUS_PARAM,
    GRID_PITCH,
    OP_POCKET_MAX,
    OP_POCKET_MIN,
    POCKET_MAX,
    POCKET_MIN,
    PROJECT_ROOT,
    STOCK_MAX,
    STOCK_MIN,
    build_pilot_task,
    compare_motion,
    contact_boundary_diagnostics,
    evaluate_sampled_geometry,
    expected_positions_from_motion,
    fit_parameterization,
    original_part_field,
    production_settings,
    run_nominal,
    source_fields,
    write_artifacts,
)
from cam.gcode_parser import parse_motion_plan
from cam.posts import post_motion_plan
from cam.seeds import IslandSeedSpec, pocket_with_island_seed, seed_deltas_mm
from verification.reference_carve import run_reference


def _file_hash(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources(tmp_path, task):
    stock, part, _, _ = source_fields(task)
    sources = {}
    for name, field in (("stock", stock), ("target", part)):
        path = tmp_path / f"source_{name}.npy"
        np.save(path, field)
        sources[name] = C.GeometrySource(
            kind="sampled", path=str(path), sha256=_file_hash(path), grid=task.grid
        )
    return sources


def _synthetic_posts(task, motion):
    """Build NC payloads plus typed records for artifact-writer tests only."""
    records = {}
    for dialect in C.Dialect:
        nc = post_motion_plan(motion, task, dialect=dialect).encode("utf-8")
        digest = C.nc_hash(nc)
        result = C.EvaluationResult(
            evaluator_id="synthetic-artifact-writer-unit-fixture",
            evaluator_revision="not-a-simulation-or-nominal-evaluation",
            task_hash=task.task_hash,
            tool_setup_hash=task.tool_setup_hash,
            discretization={"purpose": "test serialization, not physical evidence"},
            coverage=C.Coverage.COMPLETE,
            coverage_reasons=(),
            metrics={},
            safety_pass=True,
            geometry_pass=None,
            nc_hash=digest,
            dialect=dialect,
        )
        records[dialect.value] = {
            "nc_bytes": nc,
            "nc_hash": digest,
            "eval": C.dumps(result).encode("utf-8"),
        }
    return records


def _check_hashes(root):
    manifest = json.loads((root / "hashes.json").read_text())
    for name, digest in manifest.items():
        assert Path(name).name == name
        assert _file_hash(root / name) == digest
    source_index = json.loads((root / "sources.json").read_text())
    for record in source_index.values():
        assert Path(record["path"]).is_absolute()
        assert Path(record["path"]).is_file()
        assert _file_hash(record["path"]) == record["sha256"]
    return manifest


def _index(task, point):
    values = tuple(
        int(round((point[a] - task.grid.origin_mm[a]) / task.grid.voxel_mm - 0.5))
        for a in range(3)
    )
    assert all(0 <= values[a] < task.grid.shape[a] for a in range(3))
    np.testing.assert_allclose(task.grid.cell_centre_mm(*values), point, atol=1e-10, rtol=0)
    return values


def test_declared_pocket_is_full_except_finite_cutter_corners():
    task = build_pilot_task()
    assert task.split == C.Split.DEV
    assert task.task_id == "dev-pocket-island-000-v2"
    assert task.grid.voxel_mm == GRID_PITCH == 0.25
    assert task.grid.shape == (240, 160, 48)
    assert task.grid.extent_mm == tuple(hi - lo for lo, hi in zip(STOCK_MIN, STOCK_MAX))
    original = task.original_part.params
    target = task.operation_target.geometry.params
    for key in ("box_min_mm", "box_max_mm", "pocket_min_mm", "pocket_max_mm",
                "island_min_mm", "island_max_mm"):
        assert target[key] == original[key]
    assert OP_POCKET_MIN == POCKET_MIN and OP_POCKET_MAX == POCKET_MAX
    radius = target[CORNER_RADIUS_PARAM]
    depth = POCKET_MAX[2] - POCKET_MIN[2]
    assert radius == task.tool.cutter_radius_mm == 2.0
    exact_volume = 4 * (1 - math.pi / 4) * radius**2 * depth
    assert task.operation_target.excluded_volume_mm3 == exact_volume
    assert exact_volume == pytest.approx(20.601776313844965)
    assert exact_volume < 0.01 * (
        (POCKET_MAX[0] - POCKET_MIN[0])
        * (POCKET_MAX[1] - POCKET_MIN[1]) * depth
    )
    assert "No wall corridor inset" in task.operation_target.exclusion_reason
    assert target["continuous_excluded_volume_mm3"] == exact_volume
    assert target["grid_exclusion_tolerance_mm3"] == abs(
        target["grid_excluded_volume_mm3"] - exact_volume
    )
    assert target["grid_excluded_voxels"] == int(round(
        target["grid_excluded_volume_mm3"] / task.grid.voxel_mm ** 3
    ))
    holder = target["holder_clearance"]
    assert holder["bottom_at_floor_mm"] == 4.0
    assert holder["above_stock_mm"] == 4.0
    assert holder["above_stock_mm"] > holder["qualification_and_cell_margin_mm"]
    assert task.tool.holder_bottom_mm == task.tool.flute_length_mm
    assert task.tolerances.local_surface_error_mm >= 2 * task.grid.voxel_mm
    assert task.tolerances.rapid_penetration_mm == 0.0


def test_analytic_source_fields_match_declared_target_not_original_part():
    task = build_pilot_task()
    stock, part, required, retained = source_fields(task)
    assert stock.shape == part.shape == required.shape == retained.shape == task.grid.shape
    assert np.all(np.isfinite(stock)) and np.all(np.isfinite(part))
    original = original_part_field(task)
    assert not np.any((original <= 0) & ~(part <= 0))

    reachable_corner = _index(task, (23.875, 14.125, -0.125))
    assert (23.875 - 22.0) ** 2 + (14.125 - 14.0) ** 2 < 4.0
    assert required[reachable_corner]
    assert not retained[reachable_corner]
    assert part[reachable_corner] > 0
    fillet = _index(task, (23.875, 15.875, -0.125))
    assert (23.875 - 22.0) ** 2 + (15.875 - 14.0) ** 2 > 4.0
    assert not required[fillet]
    assert retained[fillet]
    assert part[fillet] < 0 < original[fillet]
    waste = _index(task, (0.125, 5.375, -5.875))
    assert required[waste] and not retained[waste]
    assert part[waste] > 0
    island = _index(task, (0.125, 0.125, -5.875))
    assert not required[island] and retained[island] and part[island] < 0
    below_floor = _index(task, (0.125, 0.125, -6.125))
    assert not required[below_floor]
    assert retained[below_floor] and part[below_floor] < 0
    outside = (0, 0, 0)
    assert stock[outside] < 0 and part[outside] < 0 and retained[outside]
    assert not np.any(required & retained)

    params = task.operation_target.geometry.params
    excluded = (part <= 0) & (original > 0)
    measured = int(excluded.sum())
    assert measured == params["grid_excluded_voxels"]
    assert measured * task.grid.voxel_mm ** 3 == params["grid_excluded_volume_mm3"]
    assert measured < int(np.count_nonzero(required) / 100)
    # Independently count one positive corner using its physical coordinates,
    # not the target-construction helper. All four corners are symmetric.
    pitch = task.grid.voxel_mm
    xs = np.arange(22.0 + pitch / 2, 24.0, pitch)
    ys = np.arange(14.0 + pitch / 2, 16.0, pitch)
    zs = np.arange(-6.0 + pitch / 2, 0.0, pitch)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    corner_outside_disk = (xx - 22.0) ** 2 + (yy - 14.0) ** 2 > 2.0 ** 2
    assert 4 * int(corner_outside_disk.sum()) == measured

    other = build_pilot_task(voxel_mm=0.5)
    assert source_fields(other)[0].shape == other.grid.shape == (120, 80, 24)
    assert other.grid.extent_mm == task.grid.extent_mm
    with pytest.raises(ValueError):
        source_fields(replace(task, stock=C.GeometrySource(kind="analytic", params={})))
    inverted_stock = replace(
        task.stock, params={**task.stock.params, "min_mm": [31.0, -20.0, -12.0]}
    )
    with pytest.raises(ValueError, match="inverted"):
        source_fields(replace(task, stock=inverted_stock))
    with pytest.raises(ValueError, match="fillet radius"):
        source_fields(replace(task, tool=replace(task.tool, cutter_radius_mm=1.0)))


def test_seed_is_typed_line_only_and_avoids_below_stock_cross_disc_transfers():
    task = build_pilot_task()
    motion = pocket_with_island_seed(task)
    motion.validate_for_task(task)
    assert motion.provenance.startswith("cam_seeds:")
    assert all(s.kind != C.SegmentKind.ARC for s in motion.segments)
    assert any(s.tag == "outer-finish" for s in motion.segments)
    assert sum(s.tag == "outer-finish" for s in motion.segments) == 4
    assert all(s.path_length_mm > 0 for s in motion.segments)
    assert sum(s.tag == "island-finish" for s in motion.segments) == 4
    perimeter = [s for s in motion.segments if s.tag == "outer-finish"]
    assert [(s.start_mm, s.end_mm) for s in perimeter] == [
        ((22.0, -14.0, -6.0), (22.0, 14.0, -6.0)),
        ((22.0, 14.0, -6.0), (-22.0, 14.0, -6.0)),
        ((-22.0, 14.0, -6.0), (-22.0, -14.0, -6.0)),
        ((-22.0, -14.0, -6.0), (22.0, -14.0, -6.0)),
    ]
    clear_radius = IslandSeedSpec().raster_clear_radius
    for segment in motion.segments:
        if segment.tag != "raster-link":
            continue
        y = segment.end_mm[1]
        if abs(y) < clear_radius:
            expected_x = math.sqrt(clear_radius**2 - y**2)
            assert abs(segment.end_mm[0]) == pytest.approx(expected_x, abs=1e-9)
        else:
            assert segment.end_mm[0] == 0.0
        assert math.hypot(segment.end_mm[0], y) >= clear_radius - 1e-9
    for segment in motion.segments:
        if segment.kind == C.SegmentKind.RAPID:
            assert segment.start_mm[2] > 0 and segment.end_mm[2] > 0
            assert segment.feed_mm_s is None
            assert segment.duration_s == pytest.approx(
                segment.path_length_mm / task.limits.rapid_mm_s, abs=1e-12
            )
        else:
            assert segment.feed_mm_s < task.limits.max_feed_mm_s
            assert segment.duration_s == pytest.approx(
                segment.path_length_mm / segment.feed_mm_s, abs=1e-12
            )
    for index, segment in enumerate(motion.segments):
        if segment.tag != "cross-disc-retract":
            continue
        assert segment.kind == C.SegmentKind.FEED
        assert segment.start_mm[2] < 0 and segment.end_mm[2] == 10.0
        transfer = motion.segments[index + 1]
        descent = motion.segments[index + 2]
        assert transfer.kind == C.SegmentKind.RAPID
        assert transfer.start_mm[2] == transfer.end_mm[2] == 10.0
        assert descent.kind == C.SegmentKind.FEED and descent.tag == "plunge"
        assert descent.end_mm[2] < 0

    def island_xy_clearance(segment):
        a = np.asarray(segment.start_mm[:2])
        b = np.asarray(segment.end_mm[:2])
        if segment.start_mm[2] != segment.end_mm[2]:
            fraction = (-6.0 - segment.start_mm[2]) / (
                segment.end_mm[2] - segment.start_mm[2]
            )
            point = a + fraction * (b - a)
            return math.hypot(
                max(abs(point[0]) - 3.5, 0.0),
                max(abs(point[1]) - 3.5, 0.0),
            )
        # Squared XY distance to a convex box is convex along a line. This
        # bounded minimization is a test oracle, not the production validator.
        lo, hi = 0.0, 1.0
        for _ in range(80):
            left = (2 * lo + hi) / 3
            right = (lo + 2 * hi) / 3
            values = []
            for t in (left, right):
                point = a + t * (b - a)
                values.append(
                    max(abs(point[0]) - 3.5, 0.0) ** 2
                    + max(abs(point[1]) - 3.5, 0.0) ** 2
                )
            if values[0] < values[1]:
                hi = right
            else:
                lo = left
        point = a + (lo + hi) * 0.5 * (b - a)
        return math.hypot(
            max(abs(point[0]) - 3.5, 0.0),
            max(abs(point[1]) - 3.5, 0.0),
        )

    for segment in motion.segments:
        if not min(segment.start_mm[2], segment.end_mm[2]) <= -6.0 <= max(
            segment.start_mm[2], segment.end_mm[2]
        ):
            continue
        assert island_xy_clearance(segment) >= task.tool.cutter_radius_mm - 1e-9
        # At the floor, both centres are inside the operation walls with room
        # for the entire cutter, not merely centre-point containment.
        for point in (segment.start_mm, segment.end_mm):
            if point[2] != -6.0:
                continue
            assert point[0] - OP_POCKET_MIN[0] >= task.tool.cutter_radius_mm - 1e-9
            assert OP_POCKET_MAX[0] - point[0] >= task.tool.cutter_radius_mm - 1e-9
            assert point[1] - OP_POCKET_MIN[1] >= task.tool.cutter_radius_mm - 1e-9
            assert OP_POCKET_MAX[1] - point[1] >= task.tool.cutter_radius_mm - 1e-9
    np.testing.assert_array_equal(
        expected_positions_from_motion(motion)[0], np.asarray((0.0, 0.0, 10.0))
    )
    np.testing.assert_array_equal(
        np.diff(expected_positions_from_motion(motion), axis=0), seed_deltas_mm(motion)
    )


def test_seed_clearance_arguments_reject_under_sized_offsets():
    task = build_pilot_task()
    with pytest.raises(ValueError, match="outer perimeter"):
        pocket_with_island_seed(task, IslandSeedSpec(outer_xy=(23.0, 14.0)))
    with pytest.raises(ValueError, match="holder above stock"):
        pocket_with_island_seed(task, IslandSeedSpec(floor_z=-15.0))
    with pytest.raises(ValueError, match="odd"):
        pocket_with_island_seed(task, IslandSeedSpec(y_rows=14))
    unknown = replace(
        task,
        initial_state=replace(task.initial_state, known=False, position_mm=None),
    )
    with pytest.raises(ValueError, match="known"):
        pocket_with_island_seed(unknown)
    wide = replace(task, tool=replace(task.tool, holder_radius_mm=1.0))
    with pytest.raises(ValueError, match="at least as wide"):
        pocket_with_island_seed(wide)


def test_zero_residual_and_both_posts_preserve_physical_motion():
    task = build_pilot_task()
    motion = pocket_with_island_seed(task)
    parameterization = fit_parameterization(motion)
    theta = np.zeros((parameterization.num_control_points, 3))
    np.testing.assert_array_equal(parameterization.forward(theta), seed_deltas_mm(motion))
    np.testing.assert_allclose(
        parameterization.positions(theta), expected_positions_from_motion(motion),
        atol=1e-12, rtol=0.0,
    )
    assert parameterization.num_control_points <= 12
    for dialect in C.Dialect:
        text = post_motion_plan(motion, task, dialect=dialect)
        parsed = parse_motion_plan(text, task, dialect=dialect)
        comparison = compare_motion(motion, parsed)
        assert comparison["max_position_error_mm"] <= 0.002
        assert comparison["max_time_error_s"] <= 1e-5
        assert comparison["semantic_tag_mismatches"] == 0
        assert len(parsed.segments) == len(motion.segments)
        for original, restored in zip(motion.segments, parsed.segments):
            assert original.kind == restored.kind
            assert original.arc is None and restored.arc is None
            assert (original.feed_mm_s is None) == (restored.feed_mm_s is None)


def test_motion_comparison_rejects_physical_and_semantic_corruption():
    task = build_pilot_task()
    original = pocket_with_island_seed(task)
    feed = next(s for s in original.segments if s.tag == "plunge")
    changed_feed = replace(
        feed, feed_mm_s=feed.feed_mm_s + 0.1,
        duration_s=feed.path_length_mm / (feed.feed_mm_s + 0.1),
    )
    segments = tuple(changed_feed if s.id == feed.id else s for s in original.segments)
    changed = replace(original, segments=segments)
    changed.validate_for_task(task)
    with pytest.raises(AssertionError, match="feed changed"):
        compare_motion(original, changed)
    first = original.segments[0]
    assert first.tag == "approach"
    changed_tag = replace(
        original,
        segments=(replace(first, tag="plunge"),) + original.segments[1:],
    )
    with pytest.raises(AssertionError, match="semantic tag changed"):
        compare_motion(original, changed_tag)
    changed_provenance = replace(
        original,
        segments=tuple(
            replace(s, tag=s.tag + "-source-label") if s.tag == "island-finish" else s
            for s in original.segments
        ),
    )
    # Free-form island labels are provenance, not a license for physical changes.
    result = compare_motion(original, changed_provenance)
    assert len(result["provenance_tag_differences"]) == 4
    assert all(
        difference["seed_tag"] == "island-finish"
        for difference in result["provenance_tag_differences"]
    )


def test_contact_disagreement_is_scoped_and_rejects_deep_or_sloped_cases():
    task = build_pilot_task()
    motion = pocket_with_island_seed(task)
    # Shift only the diagnostic grid, so a known cell at (20,0,-5.75) is
    # exactly tangent to the outer X=22 wall, not approximately grazed.
    grazing_task = replace(
        task, grid=replace(task.grid, origin_mm=(-30.125, -20.125, -12.125))
    )
    grazing = contact_boundary_diagnostics(grazing_task, motion, [[200, 80, 25]])
    assert grazing["status"] == "boundary_contacts_only"
    assert grazing["scope_supported"] is True
    assert grazing["max_abs_contact_distance_mm"] == 0.0
    deep = contact_boundary_diagnostics(task, motion, [[0, 0, 6]])
    assert deep["contact_boundary_only"] is False
    assert deep["outside_local_tolerance_cells"] == 1
    assert deep["status"] == "unresolved_contact_disagreement"
    empty = contact_boundary_diagnostics(task, motion, np.empty((0, 3), dtype=int))
    assert empty["status"] == "identical"
    assert empty["max_abs_contact_distance_mm"] == 0.0

    feed = motion.segments[2]
    end = (feed.end_mm[0], feed.end_mm[1], 2.0)
    changed = replace(
        feed, end_mm=end,
        duration_s=math.dist(feed.start_mm, end) / feed.feed_mm_s
    )
    sloped = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=replace(task.initial_state, position_mm=feed.start_mm),
        segments=(changed,), timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
    )
    unsupported = contact_boundary_diagnostics(task, sloped, [[11, 12, 6]])
    assert unsupported["scope_supported"] is False
    assert unsupported["contact_boundary_only"] is False
    assert unsupported["status"] == "unresolved_contact_disagreement"


def test_artifact_writer_preserves_hashes_sources_and_typed_records(tmp_path):
    task = build_pilot_task()
    motion = pocket_with_island_seed(task)
    parameterization = fit_parameterization(motion)
    theta = np.zeros((parameterization.num_control_points, 3))
    expected = (expected_positions_from_motion(motion) - STOCK_MIN) / (
        np.asarray(STOCK_MAX) - STOCK_MIN
    )
    actual = np.asarray(expected, dtype=np.float32)
    removed = np.ones(task.grid.shape, dtype=bool)
    _, _, required, retained = source_fields(task)
    completion = {
        "target_residual_mm3": float(np.count_nonzero(required & ~removed)),
        "retained_part_removed_mm3": float(np.count_nonzero(retained & removed)),
        "target_metrics_status": "synthetic writer fixture",
        "reference_geometry_pass": None,
    }
    sources = _sources(tmp_path, task)
    digests = {name: _file_hash(source.path) for name, source in sources.items()}
    result = write_artifacts(
        tmp_path, task, motion, parameterization, theta, sources, actual,
        parameterization.forward(theta) / (np.asarray(STOCK_MAX) - STOCK_MIN),
        removed, _synthetic_posts(task, motion), completion,
        {dialect: {"synthetic": True} for dialect in ("rs274", "haas")},
    )
    manifest = _check_hashes(tmp_path)
    assert set(result["run"].artifacts) | {"run.json"} == set(manifest)
    assert len(set(result["run"].artifacts) & {"run.json"}) == 0
    assert {name: _file_hash(source.path) for name, source in sources.items()} == digests
    task_back = C.loads((tmp_path / "task.json").read_text())
    assert task_back.task_hash == task.task_hash
    assert task_back.original_part != task_back.operation_target.geometry
    motion_back = C.loads((tmp_path / "motion_plan.json").read_text())
    assert motion_back.motion_hash == motion.motion_hash
    run_back = C.loads((tmp_path / "run.json").read_text())
    assert run_back == result["run"]
    assert run_back.status == C.RunStatus.COMPLETE
    assert run_back.failure_reason is None
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["reference_geometry_pass"] is None
    assert metrics["production_posted_evaluator"] == {
        "rs274": {"status": "synthetic_unit_fixture"},
        "haas": {"status": "synthetic_unit_fixture"},
    }
    assert metrics["camotics_cross_check"]["status"] == "not_run"
    assert np.load(tmp_path / "sim_removed_mask.npy").shape == task.grid.shape
    assert not (tmp_path / "sim_final_stock.npy").exists()
    np.testing.assert_array_equal(np.load(tmp_path / "initial_trajectory.npy"), actual)
    np.testing.assert_allclose(np.load(tmp_path / "trajectory.npy"), actual, atol=0, rtol=0)
    for name, dialect in (("rs274", C.Dialect.RS274), ("haas", C.Dialect.HAAS)):
        assert C.nc_hash((tmp_path / f"{name}.nc").read_bytes()) == result["hashes"][f"{name}.nc"]
        posted = C.loads((tmp_path / f"eval_posted_{name}.json").read_text())
        assert posted.dialect == dialect
        assert posted.nc_hash == result["hashes"][f"{name}.nc"]
        assert posted.geometry_pass is None
        assert "synthetic-artifact-writer-unit-fixture" == posted.evaluator_id

    changed = np.zeros_like(theta)
    changed[0, 0] = 1.0
    with pytest.raises(AssertionError):
        write_artifacts(
            tmp_path, task, motion, parameterization, changed, sources, actual,
            actual, removed, _synthetic_posts(task, motion), completion, {},
        )
    corrupt = actual.copy()
    corrupt[0, 0] += 1.0
    with pytest.raises(AssertionError, match="path tolerance"):
        write_artifacts(
            tmp_path, task, motion, parameterization, theta, sources, corrupt,
            actual, removed, _synthetic_posts(task, motion), completion, {},
        )


def test_nominal_output_directory_cannot_be_reused(tmp_path):
    with pytest.raises(FileExistsError, match="NEW nominal output"):
        run_nominal(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_real_nominal_pipeline_and_evidence(tmp_path):
    output = tmp_path / "real-nominal"
    command = [sys.executable, "-m", "benchmarks.pilot_task", "--output", str(output)]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=540,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    payload = json.loads((output / "nominal_result.json").read_text())
    manifest = _check_hashes(output)
    assert manifest == payload["hashes"]
    task = C.loads((output / "task.json").read_text())
    motion = C.loads((output / "motion_plan.json").read_text())
    motion.validate_for_task(task)
    run = C.loads((output / "run.json").read_text())
    assert run.status == C.RunStatus.COMPLETE
    assert run.task_hash == task.task_hash
    assert run.source_commit == subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    metrics = json.loads((output / "metrics.json").read_text())
    completion = metrics["completion"]
    assert completion["target_residual_mm3"] == 0.0
    assert completion["retained_part_removed_mm3"] == 0.0
    assert completion["target_remaining_voxels"] == 0
    assert completion["retained_part_removed_voxels"] == 0
    assert metrics["reference_geometry_pass"] is None
    assert completion["sampled_geometry_pass"] is True
    for dialect in ("rs274", "haas"):
        production_summary = metrics["production_posted_evaluator"][dialect]
        assert production_summary["status"] == "evaluated"
        assert production_summary["geometry_pass"] is True
    assert metrics["camotics_cross_check"]["status"] == "not_run"
    assert metrics["training_execution"]["max_position_error_mm"] <= 0.002
    assert not metrics["training_execution"]["speed_clipping"]
    assert len(motion.segments) > 12  # not a one/two-segment toy or hidden no-cut pass

    _, _, required, retained = source_fields(task)
    removed = np.load(output / "sim_removed_mask.npy").astype(bool)
    assert np.count_nonzero(required & ~removed) == 0
    assert np.count_nonzero(retained & removed) == 0
    assert np.any(required & removed)  # actual material removal, not a trivial no-cut result
    expected = np.load(output / "initial_trajectory.npy")
    trained = np.load(output / "trajectory.npy")
    deltas = np.load(output / "trajectory_deltas.npy")
    assert trained.shape == expected.shape == (len(motion.segments) + 1, 3)
    assert deltas.shape == (len(motion.segments), 3)
    np.testing.assert_allclose(np.diff(trained, axis=0), deltas, atol=2e-7, rtol=0)
    assert np.isfinite(trained).all() and np.all(trained[:, 0] >= -1e-6)
    assert np.all(trained[:, 0] <= 1 + 1e-6)
    expected_start = (
        np.asarray(task.initial_state.position_mm) - np.asarray(STOCK_MIN)
    ) / (np.asarray(STOCK_MAX) - np.asarray(STOCK_MIN))
    np.testing.assert_allclose(trained[0], expected_start, atol=2e-7, rtol=0)
    assert trained[0, 2] > 1.0  # start Z=10 is above, not at, stock top Z=0
    for name, dialect in (("rs274", C.Dialect.RS274), ("haas", C.Dialect.HAAS)):
        evaluation = C.loads((output / f"eval_posted_{name}.json").read_text())
        assert evaluation.evaluator_id == "posted_hard/production"
        assert evaluation.coverage == C.Coverage.COMPLETE
        assert evaluation.safety_pass is True
        assert evaluation.geometry_pass is True
        assert evaluation.efficiency_eligible is True
        assert evaluation.nc_hash == C.nc_hash((output / f"{name}.nc").read_bytes())
        assert evaluation.dialect == dialect
        assert evaluation.discretization["voxel_mm"] == task.grid.voxel_mm == 0.25
        assert evaluation.discretization["settings"]["max_step_mm"] == task.tool.cutter_radius_mm / 2
        assert evaluation.discretization["settings"]["expected_nc_hash"] == evaluation.nc_hash
        assert evaluation.discretization["settings"]["exclusion_tolerance_mm3"] == (
            task.operation_target.geometry.params["grid_exclusion_tolerance_mm3"]
        )
        assert evaluation.metrics["removed_volume_mm3"].value > 0
        assert evaluation.metrics["residual_volume_mm3"].value <= task.tolerances.completion_residual_mm3
        assert evaluation.metrics["gouge_volume_mm3"].value == 0.0
        assert evaluation.metrics["max_surface_error_mm"].value <= task.tolerances.local_surface_error_mm
        assert evaluation.metrics["excluded_volume_mm3"].value == pytest.approx(
            task.operation_target.geometry.params["grid_excluded_volume_mm3"]
        )
        restored = parse_motion_plan(
            (output / f"{name}.nc").read_text(), task, dialect=dialect
        )
        comparison = compare_motion(motion, restored)
        reference = C.loads((output / f"reference_{name}.json").read_text())
        assert reference.evaluator_id == "reference_carve"
        assert reference.geometry_pass is None
        assert reference.efficiency_eligible is False
        assert reference.coverage == C.Coverage.COMPLETE
        assert reference.safety_pass is True
        assert reference.metrics["residual_volume_mm3"].value is None
        assert reference.metrics["gouge_volume_mm3"].value is None
        fresh_trace = run_reference(task, restored, {"subdivisions": 8})
        fresh_evaluation = replace(
            fresh_trace.result,
            nc_hash=reference.nc_hash,
            dialect=dialect,
            motion_hash=None,
        )
        assert fresh_evaluation == reference
        disagreement = np.logical_xor(fresh_trace.removed, removed)
        contact = contact_boundary_diagnostics(task, restored, np.argwhere(disagreement))
        assert contact["status"] in {"identical", "boundary_contacts_only"}
        assert contact == metrics["raw_stock_cell_disagreement"][dialect.value]
        assert contact["outside_local_tolerance_cells"] == 0
        assert contact["max_abs_contact_distance_mm"] <= task.tolerances.local_surface_error_mm
        clearance = evaluation.metrics["min_clearance_lower_bound_mm"].value
        assert clearance is not None and clearance >= task.tolerances.clearance_mm
        assert evaluation.metrics[
            "rapid_cutter_penetration_upper_bound_mm"
        ].value == 0.0
        assert comparison == metrics["post_parse_equivalence"][dialect.value]
        assert comparison["max_position_error_mm"] <= 0.002
        assert comparison["max_time_error_s"] <= 1e-5
        assert comparison["semantic_tag_mismatches"] == 0
        assert any(s.kind == C.SegmentKind.RAPID for s in restored.segments)
        assert any(s.kind == C.SegmentKind.FEED for s in restored.segments)
