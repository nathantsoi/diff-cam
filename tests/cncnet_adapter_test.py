"""Offline CNC-Net adapter tests: no network, torch, Open3D or GPU required."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import contracts as C
from benchmarks import cncnet_adapter as A


def test_frozen_task_unit_transform_and_per_axis_extent_roundtrip():
    task = A.make_dev_box_pocket_task()
    frame = A.unit_frame_for_pilot(task)
    corners = np.array([task.stock.params["min_mm"], task.stock.params["max_mm"]])
    native = frame.to_native(corners)
    np.testing.assert_allclose(native, [[-.125, -.1, -.05], [.125, .1, .05]], atol=0, rtol=0)
    np.testing.assert_allclose(native.max(axis=0) - native.min(axis=0), [.25, .2, .1], atol=0, rtol=0)
    np.testing.assert_allclose(frame.to_work_mm(native), corners, atol=0, rtol=0)
    assert 0.025 * frame.mm_per_object_unit == 2.0
    assert frame.to_work_mm([[0, 0, 0]])[0].tolist() == [0.0, 0.0, -4.0]
    with pytest.raises(ValueError, match="positive"):
        A.ObjectUnitFrame((0, 0, 0), 0)
    with pytest.raises(ValueError, match="finite"):
        A.ObjectUnitFrame((float("inf"), 0, 0), 1)


def test_native_watertight_mesh_winding_topology_and_bounds(tmp_path):
    task, frame = A.make_dev_box_pocket_task(), A.unit_frame_for_pilot(A.make_dev_box_pocket_task())
    vertices, faces = A.prepare_native_mesh(task, frame)
    assert len(faces) >= 60  # includes T-junction splits on shared boundaries
    assert len(vertices) == 40  # boundary-grid vertices are welded exactly once.
    report = A.audit_mesh(vertices, faces)
    assert report["signed_volume_object3"] > 0
    coordinates = vertices.max(axis=0) - vertices.min(axis=0)
    np.testing.assert_allclose(coordinates, (.25, .2, .1), rtol=0, atol=0)
    expected_part_volume = (20 * 16 * 8 - (12 * 8 - 2 * 2) * 4) / 80.0 ** 3
    assert report["signed_volume_object3"] == pytest.approx(expected_part_volume, abs=1e-13)
    path = tmp_path / "pilot.off"
    A.write_off(path, vertices, faces)
    actual = path.read_text().splitlines()
    assert actual[0] == "OFF"
    assert actual[1] == f"{len(vertices)} {len(faces)} 0"
    assert len(actual) == 2 + len(vertices) + len(faces)
    assert A.sha256_bytes(path.read_bytes()).startswith("sha256:")
    with pytest.raises(ValueError, match="overwrite"):
        A.write_off(path, vertices, faces)


def test_open_or_inconsistent_meshes_are_rejected_before_object_file_generation(tmp_path):
    vertices, faces = A.prepare_native_mesh(A.make_dev_box_pocket_task(), A.unit_frame_for_pilot(A.make_dev_box_pocket_task()))
    with pytest.raises(ValueError, match="edge-manifold"):
        A.audit_mesh(vertices, faces[:-1])
    reversed_face = faces.copy()
    reversed_face[0], reversed_face[1] = reversed_face[0][::-1], reversed_face[1][::-1]
    with pytest.raises(ValueError, match="oriented|winding|edge-manifold"):
        A.audit_mesh(vertices, reversed_face)


def _cube():
    return (np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)], dtype=float),
            np.array([face for a,b,c,d in ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
                       for face in ((a,b,c),(a,c,d))], dtype=np.int32))


def test_audit_compares_unit_cube_geometry_with_inverted_winding():
    vertices, faces = _cube()
    assert A.audit_mesh(vertices, faces)["signed_volume_object3"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="oriented"):
        A.audit_mesh(vertices, faces[:, ::-1])


def test_native_low_iou_is_negative_voxel_ratio_not_material_overlap():
    target = np.array([[1.0, 1.0, 1.0, -1.0, -1.0]])
    current = np.full((1, 5), -0.8)
    assert A.native_iou_nonnegative_voxels(current, target) == pytest.approx(2 / 5)
    assert A.native_iou_nonnegative_voxels(np.zeros_like(current), target) == 0.0
    with pytest.raises(ValueError, match="different shapes"):
        A.native_iou_nonnegative_voxels(target, np.zeros((1, 3)))


def test_positive_native_occupancy_nearest_indices_and_out_of_bounds():
    frame = A.ObjectUnitFrame((0, 0, 0), 2)
    grid = C.GridSpec((2, 2, 2), (-1, -1, -1), 1.0, C.Frame.WORK)
    native = A.map_native_occupancy(np.zeros((2, 2, 2)), [1, 1, 1], frame, grid)
    assert not native.any()  # positive/zero raw field must not become whole stock
    field = np.ones((2, 2, 2))
    mask = A.map_native_occupancy(field, [1, 1, 1], frame, grid)
    assert mask[0, 0, 0]
    assert mask[1, 1, 1]  # inside the declared mesh extent: nearest-node boundary clamp
    partial_shape = C.GridSpec((1, 1, 2), (-.5, -.25, -.25), 1.0, C.Frame.WORK)
    partial = A.map_native_occupancy(np.ones((4, 4, 4)), [1, 1, 1], frame, partial_shape)
    np.testing.assert_array_equal(partial, [[[True, False]]])
    with pytest.raises(ValueError, match="isometric"):
        A.map_native_occupancy(np.ones((2, 3, 2)), [1, 1, 1], frame, grid)
    with pytest.raises(ValueError, match="positive object extents"):
        A.map_native_occupancy(field, [1, 0, 1], frame, grid)
    with pytest.raises(ValueError, match="non-finite"):
        A.map_native_occupancy(np.full((2, 2, 2), np.nan), [1, 1, 1], frame, grid)


def test_exact_index_mapping_is_not_inferred_from_a_filled_bounding_box():
    frame = A.ObjectUnitFrame((0, 0, 0), 1.0)
    grid = C.GridSpec((2, 1, 1), (-1, -.25, -.25), 1.0, C.Frame.WORK)
    values = np.zeros((4, 4, 4))
    values[0, :, :] = 1  # mark source sample index zero individually
    mask = A.map_native_occupancy(values, [1, 1, 1], frame, grid)
    assert mask[0, 0, 0]
    assert not mask[1, 0, 0]


def test_empty_and_full_reconstruction_metrics_keep_known_null_semantics():
    task, frame = A.make_dev_box_pocket_task(), A.unit_frame_for_pilot(A.make_dev_box_pocket_task())
    for values, removed in ((0, 2560), (1, 0)):
        metrics = A.score_reconstruction(task, frame, np.full((4, 4, 4), values), [.25, .2, .1])
        assert metrics["removed_volume_mm3"].value == removed
        if values == 0:
            assert metrics["max_surface_error_mm"].value is None
            assert metrics["max_surface_error_mm"].null_reason
        else:
            assert metrics["max_surface_error_mm"].value > 0
    filled = A.score_reconstruction(task, frame, np.ones((4, 4, 4)), [.25, .2, .1])
    assert filled["residual_volume_mm3"].value > 0


def test_disposition_is_audited_unsupported_without_nc_or_whole_mask_fallbacks():
    value = A.executable_disposition()
    assert value["status"] == "unsupported"
    assert value["disposition"] == "incompatible: executable NC mapping unsupported"
    assert value["native_ordered_waypoint_or_nc_export"] is False
    assert value["fixed_plus_z_and_one_cutter_declared"] is False
    assert value["nc_bytes"] is None and value["nc_hash"] is None
    assert value["pretraining_or_model_or_loss_substitution"] is False
    assert not value["fixed_plus_z_and_one_cutter_declared"]
    assert not value["holder_feeds_retract_g54_semantics_declared"]
    target = A.make_dev_box_pocket_task()
    target_metric_shape = C.TaskSpec.from_dict(target.to_dict())
    assert target_metric_shape.task_hash == target.task_hash


def test_obj_reader_and_atomic_json_evidence(tmp_path):
    path = tmp_path / "mesh.obj"
    path.write_text("(uninterpreted comment)\nv 1.0 2.0 3.0\nv 4 5 6\nf 0 1 1\n")
    points = A.parse_obj_vertices(path)
    np.testing.assert_array_equal(points, [[1, 2, 3], [4, 5, 6]])
    empty = tmp_path / "empty.obj"
    empty.write_text("# no points\n")
    with pytest.raises(ValueError, match="nonempty"):
        A.parse_obj_vertices(empty)
    destination = tmp_path / "evidence.json"
    A.write_json(destination, {"ok": True, "unit": "mm", "value": 0.0})
    assert json.loads(destination.read_text())["value"] == 0.0


def test_native_log_metric_parser_accepts_only_one_finite_native_score(tmp_path):
    from baselines.cncnet.score_t5p1_pilot import read_native_log_iou
    filepath = tmp_path / "native.log"
    filepath.write_text("Average IOU:\t0.423786\nBEST IOU:\t0.423786\n")
    assert read_native_log_iou(filepath) == pytest.approx(0.423786)
    for text in ("Average IOU:\tnan\n", "Average IOU:\t1.3\n", "Average IOU:\t0.1\nAverage IOU:\t0.2\n"):
        filepath.write_text(text)
        with pytest.raises(ValueError):
            read_native_log_iou(filepath)


def test_typed_prepared_task_record_hash_retained_for_auxiliary_readers():
    task = A.make_dev_box_pocket_task()
    raw = task.to_dict()
    assert C.TaskSpec.from_dict(raw) == task
    assert json.loads(C.dumps(task))["record_type"] == "TaskSpec"
    assert C.TaskSpec.from_dict(raw).task_hash == task.task_hash
