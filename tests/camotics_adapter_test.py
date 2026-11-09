"""Independent mesh/adapter tests; real Docker skips name unavailable tools/images."""
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess

import benchmarks.contracts as C
import numpy as np
import pytest

from verification import fixtures as reference_fixtures
from cam.posts import post_motion_plan
from verification.camotics_adapter import (
    EVALUATOR_ID,
    AdapterError,
    CamoticsAdapter,
    _geometry_mask,
    _rapid_sweep_conflicts,
    _prepare_camotics_nc,
    DEFAULT_IMAGE_DIGEST,
    build_camotics_project,
    mesh_volume_mm3,
    points_in_mesh,
    read_stl,
    TriangleMesh,
    unsupported_probe,
)


def _write_cube(path, scale=1.0):
    p = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ]) * scale
    # Consistent outward winding, verified by a positive signed volume.
    local_faces = [
        (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
        (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2),
        (0, 4, 5), (0, 5, 1), (3, 2, 6), (3, 6, 7),
    ]
    oriented = []
    for face in local_faces:
        a, b, c = p[list(face)]
        if np.dot(np.cross(b - a, c - a), (a + b + c) / 3) < 0:
            b, c = c, b
        oriented.extend((a, b, c))
    triangles = np.asarray(oriented).reshape(-1, 3, 3)
    geometry = triangles.reshape(-1).tolist()
    payload = struct.pack("<I", len(local_faces))
    for start in range(0, len(geometry), 9):
        payload += struct.pack("<12fH", 0, 0, 0, *geometry[start:start + 9], 0)
    with path.open("wb") as stream:
        stream.write(b"solid".ljust(80) + payload)
    return triangles


def test_binary_and_ascii_stl_reading_roundtrip(tmp_path):
    path = tmp_path / "cube.stl"
    expected = _write_cube(path)
    result = read_stl(path)
    assert result.stl_format == "binary"
    np.testing.assert_array_equal(result.vertices_mm, expected.reshape(-1, 3, 3))
    ascii_path = tmp_path / "cube_ascii.stl"
    with ascii_path.open("w") as stream:
        stream.write("solid cube\n")
        for index, tri in enumerate(expected):
            stream.write("facet normal 0 0 0\nouter loop\n")
            for point_index, _ in enumerate(tri):
                if not (index == len(expected) - 1 and point_index == 2):
                    stream.write("vertex 1 2 4\n")
            stream.write("endloop\nendfacet\n")
        stream.write("endsolid cube\n")
    with pytest.raises(AdapterError, match="vertex count"):
        read_stl(ascii_path)
    # The reader accepts independent binary facets even when the ASCII file
    # coexists, rather than guessing based on an arbitrary 'solid' header.


def test_truncated_and_degenerate_mesh_are_rejected(tmp_path):
    valid = tmp_path / "valid.stl"
    _write_cube(valid)
    data = valid.read_bytes()
    truncated = tmp_path / "truncated.stl"
    truncated.write_bytes(data[:-1])
    with pytest.raises(AdapterError, match="neither valid binary nor text STL"):
        read_stl(truncated)
    degenerate = tmp_path / "degenerate.stl"
    first_facet = struct.pack("<12fH", 0, 0, 0, *([0.0] * 9), 0)
    degenerate.write_bytes(data[:84] + first_facet + data[134:])
    with pytest.raises(AdapterError, match="degenerate"):
        read_stl(degenerate)


def test_cube_volume_and_closed_surface_grid_queries(tmp_path):
    mesh = TriangleMesh(_write_cube(tmp_path / "cube.stl"), "binary")
    facts = mesh_volume_mm3(mesh)
    assert facts["volume_mm3"] == pytest.approx(1.0)
    assert facts["watertight"]
    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # interior, exterior
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],  # duplicate corner: one contact, two queries
        [0.5, 0.5, -0.5], [0.5, 0.5, 1.5],  # bottom corner; exterior above
        [0.5, 0.25, 0.0],                  # vertical face, not an XY ray root
        [0.0, 0.0, 0.0],                   # duplicate interior input retained as work
    ])
    result = points_in_mesh(mesh, points, touch_tolerance_mm=1e-8)
    np.testing.assert_array_equal(result.occupancy, [True, False, True, True, True, False, True, True])
    assert result.surface_contact_count == 3
    assert result.projected_ray_queries >= 6
    assert result.spatial_bin_count > 0
    assert result.boundary_policy == "closed_include"


def test_inconsistent_facet_winding_is_not_classified_watertight(tmp_path):
    triangles = _write_cube(tmp_path / "cube.stl")
    triangles[0] = triangles[0][::-1]
    mesh = TriangleMesh(triangles, "binary")
    facts = mesh_volume_mm3(mesh, require_watertight=False)
    assert facts["watertight"] is False
    assert facts["inconsistent_direction_edges"] > 0
    # Each shared edge counts twice, but one shared direction no longer cancels.
    with pytest.raises(AdapterError, match="consistent orientation"):
        mesh_volume_mm3(mesh)


def test_small_closed_mesh_ray_classification_uses_area_cutoff(tmp_path):
    mesh = TriangleMesh(_write_cube(tmp_path / "small.stl") * 5e-7, "binary")
    # Face projection area is 2.5e-13 mm^2. Mixing the old 1e-9 mm tolerance
    # into that threshold wrongly discarded every ray root for this valid cube.
    result = points_in_mesh(mesh, np.array([[0.0, 0.0, 0.0], [6e-7, 0.0, 0.0]]),
                            touch_tolerance_mm=1e-9)
    np.testing.assert_array_equal(result.occupancy, [True, False])
    assert result.projected_ray_queries == 1
    assert result.surface_contact_count == 0


def test_open_mesh_does_not_claim_watertight_volume(tmp_path):
    facets = _write_cube(tmp_path / "cube.stl")
    # Use a simple single-facet plane: definitely no watertight solid.
    mesh = TriangleMesh(facets[:1], "binary")
    with pytest.raises(AdapterError, match="closed two-face manifold"):
        mesh_volume_mm3(mesh)
    limited = mesh_volume_mm3(mesh, require_watertight=False)
    assert limited["watertight"] is False
    assert limited["boundary_or_nonmanifold_edges"] == 3
    with pytest.raises(AdapterError, match="shape"):
        TriangleMesh(np.zeros((6, 3), dtype=float), "binary")


def test_box_minus_pocket_preserves_island():
    task = C.example_pocket_task()
    grid = task.grid
    centres = [grid.cell_centre_mm(8, 8, k) for k in range(grid.shape[2])]
    target = _geometry_mask(task.operation_target.geometry, np.asarray(centres))
    outer = _geometry_mask(task.stock, np.asarray(centres))
    part = _geometry_mask(task.original_part, np.asarray(centres))
    assert outer.all()
    assert target[:7].all()          # material below the pocket remains
    assert target[-1]                 # preserved island at the pocket's top
    # At island XY centre, the original and reachable parts retain island stock.
    # This also checks empty/nonempty island conditions without an ad-hoc SDF.
    fixed = np.array([[0.0, 0.0, -2.5], [0.0, 0.0, -7.5], [0.0, 0.0, -1.0],
                      [3.0, 3.0, -1.0], [7.0, 7.0, -1.0]])
    island = _geometry_mask(task.original_part, fixed)
    assert list(island) == [True, True, True, False, True]
    assert list(_geometry_mask(task.operation_target.geometry, fixed)) == list(island)
    assert len(part) == len(centres)
    with pytest.raises(AdapterError, match="island"):
        bad_part = replace(task.original_part, params=dict(task.original_part.params,
                                                           island_min_mm=[-20, -20, -5], island_max_mm=[-19, -19, 0]))
        _geometry_mask(bad_part, np.zeros((1, 3)))


def test_canonical_project_schema_flat_tool_and_transformed_bounds():
    task = C.example_pocket_task()
    spec = build_camotics_project(task, 0.25)
    assert spec["units"] == "metric"
    assert spec["resolution-mode"] == "manual"
    assert spec["resolution"] == 0.25
    assert spec["tools"]["1"]["shape"] == "cylindrical"
    assert spec["tools"]["1"]["diameter"] == pytest.approx(6.35)
    assert spec["tools"]["1"]["length"] == 12.0
    assert spec["workpiece"]["automatic"] is False
    assert spec["workpiece"]["margin"] == 0
    # These are machine coordinates minus the declared initial machine pose,
    # NOT raw physical machine coordinates. They exactly recenter CAMotics'.
    np.testing.assert_allclose(spec["workpiece"]["bounds"]["min"], [-12.7, -12.7, -35.4], atol=1e-12)
    np.testing.assert_allclose(spec["workpiece"]["bounds"]["max"], [12.7, 12.7, -10.0], atol=1e-12)
    assert spec["files"] == ["program.nc"]


def test_unsupported_tool_stock_and_bad_bounds_fail_closed():
    task = C.example_pocket_task()
    ball = replace(task, tool=replace(task.tool, cutter_kind=C.CutterKind.BALL, flute_length_mm=25.0))
    assert any("flat" in reason for reason in unsupported_probe(ball))
    stepped = replace(task, stock=replace(task.stock, kind="mesh", path="x.stl", sha256="sha256:" + "0" * 64))
    assert any("analytic box" in reason for reason in unsupported_probe(stepped))
    bad = replace(task.stock, params={"primitive": "box", "min_mm": [0, 0], "max_mm": [1, 1, 1]})
    with pytest.raises(AdapterError):
        _geometry_mask(bad, np.zeros((1, 3)))


def test_valid_ascii_stl_is_parsed_not_just_rejected(tmp_path):
    cube = tmp_path / "binary.stl"
    triangles = _write_cube(cube)
    ascii_path = tmp_path / "cube.stl"
    with ascii_path.open("w", encoding="utf-8") as stream:
        stream.write("solid cube\n")
        for facet in triangles:
            stream.write("facet normal 0 0 0\nouter loop\n")
            for vertex in facet:
                stream.write("vertex " + " ".join(format(value, ".17g") for value in vertex) + "\n")
            stream.write("endloop\nendfacet\n")
        stream.write("endsolid cube\n")
    result = read_stl(ascii_path)
    assert result.stl_format == "ascii"
    assert result.triangle_count == 12
    assert mesh_volume_mm3(result)["volume_mm3"] == pytest.approx(1.0)


def test_calibration_preserves_raw_nc_and_selects_the_declared_tool():
    task = C.example_pocket_task()
    motion = C.example_motion_plan()
    source = post_motion_plan(motion, task, dialect=C.Dialect.HAAS).encode("utf-8")
    derived, parsed = _prepare_camotics_nc(task, source, C.Dialect.HAAS)
    # Camera frame: machine initial (100,80,-140), translation (100,80,-150).
    # In v1.2.0 the actual motion is raw coordinates minus the G10 words.
    assert b"G10 L2 P1 X0 Y0 Z10\n" in derived
    assert b"\nT1 M06\n" in derived
    assert derived.endswith(source)
    assert len(parsed.segments) == len(motion.segments)
    for recovered, expected in zip(parsed.segments, motion.segments):
        assert recovered.kind == expected.kind
        np.testing.assert_allclose(recovered.start_mm, expected.start_mm, atol=1e-6)
        np.testing.assert_allclose(recovered.end_mm, expected.end_mm, atol=1e-6)
    other_tool = replace(task, tool=replace(task.tool, tool_number=2))
    assert set(build_camotics_project(other_tool, 0.25)["tools"]) == {"2"}
    with pytest.raises(AdapterError, match="must use TaskSpec"):
        build_camotics_project(other_tool, 0.25, tool_number=1)


def test_calibration_rejects_g53_and_rotated_work_frames():
    task = C.example_pocket_task()
    motion = C.example_motion_plan()
    segments = (replace(motion.segments[0], tag="home"), *motion.segments[1:])
    absolute = post_motion_plan(replace(motion, segments=segments), task,
                                dialect=C.Dialect.HAAS).encode("utf-8")
    assert b"G53" in absolute
    with pytest.raises(ValueError, match="G53"):
        _prepare_camotics_nc(task, absolute, C.Dialect.HAAS)
    rotation = ((0.0, -1.0, 0.0, 100.0), (1.0, 0.0, 0.0, 80.0),
                (0.0, 0.0, 1.0, -150.0), (0.0, 0.0, 0.0, 1.0))
    rotated = replace(task, work_to_machine=C.Transform(rotation))
    assert any("rotation" in reason for reason in unsupported_probe(rotated))
    with pytest.raises(AdapterError, match="G10"):
        _prepare_camotics_nc(rotated, post_motion_plan(motion, task).encode(), C.Dialect.RS274)


def _cube_task():
    task = C.example_pocket_task()
    geometry = C.GeometrySource(kind="analytic", params={"primitive": "box",
                                                          "min_mm": [-1.0] * 3, "max_mm": [1.0] * 3})
    initial = replace(task.initial_state, position_mm=(0.0, 0.0, 2.0), frame=C.Frame.MACHINE)
    return replace(task, stock=geometry, original_part=geometry,
                   operation_target=replace(task.operation_target, geometry=geometry),
                   grid=C.GridSpec((2, 2, 2), (-1.0, -1.0, -1.0), 1.0),
                   work_to_machine=C.Transform.translation((0.0, 0.0, 0.0)),
                   initial_state=initial)


def _cube_source_nc():
    # The known machine/work initial pose is Z2 (identity task translation).
    # These stationary blocks remain there, not a fresh rapid into stock.
    return (b"G21\nG17 G40 G49 G80 G90 G94\nG54 G61\nG43 H1\n"
            b"G0 X0 Y0 Z2\nG1 X0 Y0 Z2 F600\nM2\n")


def test_invalid_nc_fails_before_docker_and_keeps_source_hash(tmp_path, monkeypatch):
    source = _cube_source_nc().replace(b"G43 H1", b"G43 H2")
    adapter = CamoticsAdapter(scratch_root=tmp_path)
    def no_docker(*arguments, **kwargs):
        raise AssertionError("Docker was invoked for invalid source NC")
    monkeypatch.setattr(adapter, "_docker_call", no_docker)
    result = adapter.evaluate_posted_gcode(_cube_task(), source, C.Dialect.RS274,
                                           {"scratch_root": str(tmp_path)})
    assert result.coverage == C.Coverage.UNSUPPORTED
    assert result.nc_hash == C.nc_hash(source)
    assert result.safety_pass is None
    assert result.geometry_pass is None
    assert "H2" in " ".join(result.coverage_reasons)
    assert "tool.tool_number=1" in " ".join(result.coverage_reasons)


def _install_mock_cube(monkeypatch, adapter):
    """Post-process-only simulator mock, never evidence of CAMotics accuracy."""
    observed = {}
    def fake_run(task, simulation_nc, project, workdir, timeout, *, source_nc_bytes):
        observed["simulation_nc"] = simulation_nc
        observed["source_nc"] = source_nc_bytes
        (workdir / "source_program.nc").write_bytes(source_nc_bytes)
        (workdir / "program.nc").write_bytes(simulation_nc)
        spec_bytes = json.dumps(project, sort_keys=True).encode()
        (workdir / "project.camotics").write_bytes(spec_bytes)
        triangles = _write_cube(workdir / "cut_workpiece.stl", scale=2.0)
        triangles[:, :, 2] -= 2.0
        payload = struct.pack("<I", len(triangles))
        for facet in triangles:
            payload += struct.pack("<12fH", 0, 0, 0, *facet.reshape(-1), 0)
        (workdir / "cut_workpiece.stl").write_bytes(b"MOCK".ljust(80) + payload)
        return {"image": "MOCK-NOT-REAL", "image_digest": "MOCK-NOT-REAL",
                "camsim_version": "MOCK-NOT-REAL", "binary_release": "1.2.0",
                "command": ["MOCK-NOT-REAL"], "exit_code": 0, "stdout": "", "stderr": "",
                "wall_time_s": 0.0, "workdir": str(workdir),
                "simulation_nc_sha256": C.sha256_bytes(simulation_nc),
                "source_nc_sha256": C.sha256_bytes(source_nc_bytes),
                "project_sha256": C.sha256_bytes(spec_bytes)}
    monkeypatch.setattr(adapter, "_run_camsim", fake_run)
    return observed


def test_mocked_mesh_completion_has_paired_artifacts_not_external_evidence(tmp_path, monkeypatch):
    """Unit-only downstream test: this specifically is NOT a real Docker smoke."""
    adapter = CamoticsAdapter(scratch_root=tmp_path)
    observed = _install_mock_cube(monkeypatch, adapter)
    task = _cube_task()
    source = _cube_source_nc()
    result = adapter.evaluate_posted_gcode(task, source, C.Dialect.RS274,
                                           {"scratch_root": str(tmp_path), "resolution_mm": 0.5})
    assert result.coverage == C.Coverage.INCOMPLETE
    assert result.safety_pass is None
    assert result.geometry_pass is True
    assert result.metrics["remaining_volume_mm3"].value == pytest.approx(8.0)
    assert result.metrics["removed_volume_mm3"].value == 0.0
    path = Path(result.discretization["workdir"])
    assert (path / "source_program.nc").read_bytes() == observed["source_nc"] == source
    derived = observed["simulation_nc"]
    assert derived.endswith(source)
    assert result.nc_hash == result.discretization["source_nc_sha256"] == C.nc_hash(source)
    assert result.discretization["simulation_nc_sha256"] == C.sha256_bytes(derived)
    assert result.discretization["simulation_nc_sha256"] != result.nc_hash
    project = json.loads((path / "project.camotics").read_text())
    assert project["workpiece"]["bounds"] == {"min": [-1.0, -1.0, -3.0],
                                             "max": [1.0, 1.0, -1.0]}
    assert result.discretization["source_program_filename"] == "source_program.nc"
    assert result.discretization["stl_topology"]["inconsistent_direction_edges"] == 0
    assert result.discretization["stl_topology"]["self_intersection_check"].startswith("not performed")
    assert result.discretization["exclusion_tolerance_mm3"] == 0.0
    assert result.metrics["excluded_volume_mm3"].value == 0.0
    assert result.metrics["original_part_gouge_volume_mm3"].value == 0.0
    assert result.metrics["removal_target_volume_mm3"].value == 0.0


def test_declared_exclusion_mismatch_and_bad_tolerance_are_blocked(tmp_path, monkeypatch):
    task = _cube_task()
    source = _cube_source_nc()
    adapter = CamoticsAdapter(scratch_root=tmp_path)
    observed = _install_mock_cube(monkeypatch, adapter)
    mismatched = replace(task, operation_target=replace(
        task.operation_target, excluded_volume_mm3=0.5,
        exclusion_reason="explicit test of a declaration inconsistent with the grid"))
    failed = adapter.evaluate_posted_gcode(mismatched, source, C.Dialect.RS274,
                                            {"scratch_root": str(tmp_path)})
    assert failed.coverage == C.Coverage.UNKNOWN
    assert "excluded volume mismatch" in " ".join(failed.coverage_reasons)
    assert failed.geometry_pass is None and failed.safety_pass is None
    assert failed.nc_hash == C.nc_hash(source)
    for name in ("excluded_volume_mm3", "original_part_gouge_volume_mm3",
                 "removal_target_volume_mm3"):
        assert failed.metrics[name].unit == "mm^3"
        assert failed.metrics[name].value is None
        assert failed.metrics[name].null_reason
    calls = len(observed)
    bad_tolerance = adapter.evaluate_posted_gcode(task, source, C.Dialect.RS274,
                                                   {"scratch_root": str(tmp_path),
                                                    "exclusion_tolerance_mm3": -0.1})
    assert len(observed) == calls  # invalid numeric policy must not re-enter the simulator
    assert bad_tolerance.coverage == C.Coverage.UNKNOWN
    assert "exclusion_tolerance_mm3" in " ".join(bad_tolerance.coverage_reasons)
    assert bad_tolerance.metrics["excluded_volume_mm3"].value is None


def test_camsim_command_enforces_serial_mcubes_and_records_reason(tmp_path, monkeypatch):
    """Command-construction unit test, not runtime evidence of corrected geometry."""
    task = _cube_task()
    adapter = CamoticsAdapter(scratch_root=tmp_path)
    monkeypatch.setattr(adapter, "_image_facts", lambda: {
        "image": "diffcam-camotics:1.2.0", "image_digest": DEFAULT_IMAGE_DIGEST,
        "camsim_version": "1.2", "binary_release": "1.2.0"})
    captured = []
    def fake_call(arguments, timeout=None):
        captured.append(arguments)
        _write_cube(tmp_path / "cut_workpiece.stl")
        return subprocess.CompletedProcess(arguments, 0, "", "")
    monkeypatch.setattr(adapter, "_docker_call", fake_call)
    facts = adapter._run_camsim(task, b"G21\n", build_camotics_project(task, 0.25),
                                tmp_path, 3.0, source_nc_bytes=b"G21\n")
    assert len(captured) == 1
    index = captured[0].index("camsim")
    assert captured[0][index + 1:] == ["--binary", "--render-mode", "0", "--threads", "1",
                                      "project.camotics", "cut_workpiece.stl"]
    assert facts["renderer_threads"] == 1
    assert facts["render_mode"] == "MCUBES_MODE"
    assert facts["render_mode_id"] == 0
    assert "no general manifold or geometry-accuracy guarantee" in facts["render_policy"]


def test_timeout_returns_blocked_and_stops_only_its_unique_named_container(tmp_path, monkeypatch):
    adapter = CamoticsAdapter(scratch_root=tmp_path)
    monkeypatch.setattr(adapter, "_image_facts", lambda: {"image": "test", "image_digest": "test-id",
                                                          "camsim_version": "test", "binary_release": "1.2.0"})
    started = []
    def timeout(command, timeout=None):
        started.append((command, timeout))
        raise subprocess.TimeoutExpired(command, timeout)
    monkeypatch.setattr(adapter, "_docker_call", timeout)
    stopped = []
    def stop(command, **kwargs):
        stopped.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(subprocess, "run", stop)
    task = _cube_task()
    nc = _cube_source_nc()
    result = adapter.evaluate_posted_gcode(task, nc, C.Dialect.RS274,
                                           {"scratch_root": str(tmp_path), "timeout_s": 0.2})
    assert result.coverage == C.Coverage.UNKNOWN
    assert result.safety_pass is None and result.geometry_pass is None
    assert result.nc_hash == C.nc_hash(nc)
    assert len(started) == len(stopped) == 1
    name_index = started[0][0].index("--name") + 1
    name = started[0][0][name_index]
    assert name.startswith("diffcam-t13-")
    assert stopped[0][:4] == ["docker", "stop", "--time", "5"]
    assert stopped[0][4:] == [name]
    assert all(value.value is None for value in result.metrics.values())


def test_conflicting_rapid_is_unsupported_not_reported_as_real_removal(tmp_path, monkeypatch):
    fixture, task = _machine_fixture(reference_fixtures.rapid_through_solid)
    adapter = CamoticsAdapter(scratch_root=tmp_path)
    forbidden = []
    def forbidden_run(*args, **kwargs):
        forbidden.append((args, kwargs))
        raise AssertionError("a conflicting rapid must never run camsim")
    monkeypatch.setattr(adapter, "_run_camsim", forbidden_run)
    result = adapter.evaluate_candidate(task, fixture.motion, {"dialect": C.Dialect.HAAS})
    assert result.coverage == C.Coverage.UNSUPPORTED
    assert result.geometry_pass is None and result.safety_pass is None
    assert result.motion_hash is not None
    assert not forbidden
    assert all(metric.value is None for metric in result.metrics.values())
    assert any("rapid" in reason and "would carve" in reason for reason in result.coverage_reasons)


def test_canonical_retract_is_flagged_while_safe_approach_is_not_blanket_rejected():
    task = C.example_pocket_task()
    reasons = _rapid_sweep_conflicts(task, C.example_motion_plan())
    assert len(reasons) == 1
    assert "rapid segment" in reasons[0]
    assert "would carve" in reasons[0]


def test_version_probe_reads_stderr_and_refuses_an_empty_version(monkeypatch):
    adapter = CamoticsAdapter()
    valid_info = f"{DEFAULT_IMAGE_DIGEST}\n"
    replies = [subprocess.CompletedProcess([], 0, valid_info, ""),
               subprocess.CompletedProcess([], 0, "", "1.2\n")]
    monkeypatch.setattr(adapter, "_docker_call", lambda *args, **kwargs: replies.pop(0))
    facts = adapter._image_facts()
    assert facts["camsim_version"] == "1.2"
    assert facts["version_probe_stdout"] == ""
    assert facts["version_probe_stderr"] == "1.2\n"
    empty = [subprocess.CompletedProcess([], 0, valid_info, ""),
             subprocess.CompletedProcess([], 0, "", "")]
    monkeypatch.setattr(adapter, "_docker_call", lambda *args, **kwargs: empty.pop(0))
    with pytest.raises(AdapterError, match="no version"):
        adapter._image_facts()


def test_incompatible_image_or_version_requires_explicit_requalification(monkeypatch):
    adapter = CamoticsAdapter()
    other = [subprocess.CompletedProcess([], 0, "sha256:not-the-calibrated-image\n", "")]
    monkeypatch.setattr(adapter, "_docker_call", lambda *args, **kwargs: other.pop(0))
    with pytest.raises(AdapterError, match="requalify"):
        adapter._image_facts()
    alternate = [subprocess.CompletedProcess([], 0, DEFAULT_IMAGE_DIGEST + "\n", ""),
                 subprocess.CompletedProcess([], 0, "", "1.3\n")]
    monkeypatch.setattr(adapter, "_docker_call", lambda *args, **kwargs: alternate.pop(0))
    with pytest.raises(AdapterError, match="G10"):
        adapter._image_facts()


def test_absent_image_is_a_probe_failure_not_a_pass(monkeypatch):
    adapter = CamoticsAdapter()
    monkeypatch.setattr(adapter, "_docker_call",
                        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "missing image"))
    with pytest.raises(AdapterError, match="Docker/image unavailable"):
        adapter._image_facts()


def _machine_fixture(factory):
    fixture = factory()
    task = fixture.task
    state = task.initial_state
    position = task.work_to_machine.apply(state.position_mm)
    machine_task = replace(task, initial_state=replace(state, position_mm=position, frame=C.Frame.MACHINE))
    return fixture, machine_task


@pytest.mark.skipif(shutil.which("docker") is None, reason="real CAMotics Docker route unavailable")
def test_real_docker_control_and_plunge_are_executed_not_quality_certificates(tmp_path):
    adapter = CamoticsAdapter(scratch_root=tmp_path, default_resolution_mm=0.5)
    results = []
    for factory in (reference_fixtures.empty_preengaged, reference_fixtures.plunge_above):
        fixture, task = _machine_fixture(factory)
        result = adapter.evaluate_candidate(task, fixture.motion,
                                            {"dialect": C.Dialect.HAAS, "resolution_mm": 0.5})
        reasons = " ".join(result.coverage_reasons)
        if result.coverage == C.Coverage.UNKNOWN and any(
                marker in reasons for marker in ("Docker/image unavailable", "binary version probe failed")):
            pytest.skip("declared CAMotics image is unavailable: " + reasons)
        assert result.coverage == C.Coverage.INCOMPLETE, reasons
        assert result.evaluator_id == EVALUATOR_ID
        assert result.safety_pass is None
        assert result.discretization["image_digest"].startswith("sha256:")
        assert result.discretization["camsim_version"] == "1.2"
        assert result.discretization["version_probe_stderr"] == "1.2\n"
        assert result.discretization["simulation_frame"] == "machine_mm - task.initial_state.position_mm"
        assert result.discretization["renderer_threads"] == 1
        assert result.discretization["render_mode"] == "MCUBES_MODE"
        if result.discretization["stl_topology"]["watertight"]:
            assert result.discretization["stl_topology"]["inconsistent_direction_edges"] == 0
        workdir = Path(result.discretization["workdir"])
        assert (workdir / "program.nc").read_bytes().endswith((workdir / "source_program.nc").read_bytes())
        assert result.nc_hash == result.discretization["source_nc_sha256"]
        results.append(result)
    assert results[0].metrics["remaining_volume_mm3"].value == pytest.approx(28800.0)
    assert results[0].metrics["removed_volume_mm3"].value == 0.0
    # Analytic 36*pi mm^3 is an independent fixture, NOT loosened to whatever
    # camsim happens to produce. This checks only nonempty removal on the E2E
    # route; measured bias belongs in reports/camotics_bias_t13a.md.
    assert math.pi * 36 == pytest.approx(reference_fixtures.plunge_above().removed_volume_mm3)
    assert 0.0 < results[1].metrics["removed_volume_mm3"].value < 28800.0
    assert results[1].metrics["remaining_volume_mm3"].value < 28800.0
