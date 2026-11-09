"""Hand-computed contract and regression tests for grid material metrics."""
import math

import benchmarks.contracts as C
import numpy as np
import pytest

from simulator.csg_metrics import (
    _gouge,
    _residual,
    average_surface_distance,
    compute_grid_metrics,
    dice_score,
    extract_surface,
    grid_material_metrics,
    hd95,
    iou_score,
    sdf_to_mask,
    surface_distances,
    surface_error_metrics,
)


def test_legacy_helpers_remain_compatible():
    np.testing.assert_array_equal(sdf_to_mask(np.array([-0.1, 0.0, 0.1])), [True, False, False])
    pred = np.array([True, True, False, False])
    target = np.array([True, False, True, False])
    assert _gouge(pred, target) == 1
    assert _residual(pred, target) == 1
    assert dice_score(pred, target) == pytest.approx((2 + 1e-8) / (4 + 1e-8))
    assert iou_score(pred, target) == pytest.approx((1 + 1e-8) / (3 + 1e-8))
    assert dice_score(np.zeros(2), np.zeros(2)) == 1
    assert iou_score(np.zeros(2), np.zeros(2)) == 1
    distances = surface_distances(pred.reshape(1, 1, 4), target.reshape(1, 1, 4))
    np.testing.assert_allclose(distances, [[0.0, 1.0], [0.0, 1.0]])


def test_material_counts_volumes_fractions_and_denominators():
    initial = np.ones(8, dtype=bool).reshape(2, 2, 2)
    part = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool).reshape(2, 2, 2)
    remaining = np.array([1, 0, 1, 0, 1, 1, 1, 1], dtype=bool).reshape(2, 2, 2)
    metrics = compute_grid_metrics(remaining, part, initial, 0.5)
    # Cells 1 and 3 are removed. Cell 1 is inside the desired part; cell 3 is
    # removal target (not inside desired part). Removal-target count = 5, part count = 3.
    expected = {
        "remaining_volume_mm3": 6 / 8,
        "removed_volume_mm3": 2 / 8,
        "residual_volume_mm3": 4 / 8,
        "gouge_volume_mm3": 1 / 8,
        "initial_stock_volume_mm3": 8 / 8,
        "operation_target_removed_volume_mm3": 5 / 8,
        "original_part_volume_mm3": 3 / 8,
        "residual_fraction": 4 / 5,
        "gouge_fraction": 1 / 3,
        "operation_target_removed_fraction": 1 / 5,
        "iou": 2 / 7,
        "voxel_volume_mm3": 1 / 8,
    }
    for name, expected_value in expected.items():
        assert metrics[name].value == pytest.approx(expected_value)
    assert metrics["residual_volume_mm3"].unit == "mm^3"
    assert metrics["residual_fraction"].unit == "1"
    assert metrics["residual_fraction"].denominator.endswith("0.625")
    assert metrics["gouge_fraction"].denominator.endswith("0.375")


def test_empty_operation_target_denominators_are_not_fake_zero_or_one():
    initial = np.ones((2, 2, 2), dtype=bool)
    metrics = compute_grid_metrics(np.zeros_like(initial), initial, initial, 2.0)
    for name in ("operation_target_removed_fraction", "residual_fraction"):
        assert metrics[name].value is None
        assert "operation target volume is zero" in metrics[name].null_reason
    assert metrics["gouge_fraction"].value == 1  # all of the desired part is removed
    assert all(metrics[name].value is None for name in
               ("max_surface_error_mm", "p99_surface_error_mm", "asd_mm", "hd95_mm"))
    assert metrics["dice"].value == pytest.approx(1e-8 / (8 + 1e-8))


def test_surface_errors_are_symmetric_mm_distances_with_linear_percentiles():
    remaining = np.zeros((4, 3, 1), dtype=bool)
    target = np.zeros((4, 3, 1), dtype=bool)
    remaining[0, 0, 0] = True
    target[3, 2, 0] = True
    metrics = surface_error_metrics(remaining, target, spacing_mm=0.5, origin_mm=(10, 20, 30))
    # In index coordinates: (3,2) away. Physical millimetre spacing 0.5.
    expected = math.sqrt(3**2 + 2**2) * 0.5
    for name in ("max_surface_error_mm", "p99_surface_error_mm", "asd_mm", "hd95_mm"):
        assert metrics[name].value == pytest.approx(expected, abs=2e-15)
        assert metrics[name].unit == "mm"
    assert "nearest voxel-centre" in metrics["max_surface_error_mm"].denominator
    assert "linear p99" in metrics["p99_surface_error_mm"].denominator


def test_surface_nulls_describe_which_side_is_empty():
    empty = np.zeros((2, 2, 2), dtype=bool)
    solid = np.ones((2, 2, 2), dtype=bool)
    left = surface_error_metrics(empty, solid, 1)
    right = surface_error_metrics(solid, empty, 1)
    assert "remaining_surface_voxels=0" in left["max_surface_error_mm"].null_reason
    assert f"operation_target_surface_voxels={int(solid.sum())}" in left["max_surface_error_mm"].null_reason
    assert "operation_target_surface_voxels=0" in right["max_surface_error_mm"].null_reason
    both = surface_error_metrics(empty, empty, 1)
    for metric in both.values():
        assert metric.value is None and metric.null_reason
    with pytest.raises(ValueError, match="Empty surface"):
        surface_distances(empty, solid)


def test_local_max_hd95_and_p99_are_ordered_and_legacy_asd_is_preserved():
    remaining = np.zeros((5, 1, 1), dtype=bool)
    target = np.zeros((5, 1, 1), dtype=bool)
    remaining[0, 0, 0] = True
    remaining[2, 0, 0] = True
    target[1, 0, 0] = True
    target[4, 0, 0] = True
    metrics = surface_error_metrics(remaining, target, 1)
    values = [metrics[name].value for name in ("asd_mm", "hd95_mm", "p99_surface_error_mm", "max_surface_error_mm")]
    assert values[0] <= values[1] <= values[2] <= values[3]
    assert metrics["max_surface_error_mm"].value == 2.0
    assert average_surface_distance(remaining, target) == pytest.approx(1.25)
    assert hd95(remaining, target) == pytest.approx(1.85)


def test_matching_masks_have_zero_surface_error_and_geometry_pass_inputs_work():
    initial = np.ones((3, 3, 3), dtype=bool)
    part = initial.copy()
    result = compute_grid_metrics(part, part, initial, 2)
    for name in ("max_surface_error_mm", "p99_surface_error_mm", "asd_mm", "hd95_mm"):
        assert result[name].value == 0
    assert result["iou"].value == pytest.approx(1)
    assert result["residual_volume_mm3"].value == 0
    assert result["gouge_volume_mm3"].value == 0
    tolerances = C.Tolerances(0, 0, 0, 0)
    assert C.geometry_pass_from_metrics(result, tolerances) is True
    unknown = dict(result)
    unknown["residual_volume_mm3"] = C.Metric(None, "mm^3", null_reason="unsupported target")
    assert C.geometry_pass_from_metrics(unknown, tolerances) is None


def test_empty_masks_have_defined_overlap_policy_but_unknown_surface():
    empty = np.zeros((2, 1, 1), dtype=bool)
    metrics = grid_material_metrics(empty, empty, empty, 1)
    assert metrics["iou"].value == 1
    assert metrics["dice"].value == 1
    assert metrics["removed_volume_mm3"].value == 0
    assert metrics["max_surface_error_mm"].value is None


def test_invalid_shape_spacing_origin_and_stock_containment_fail():
    initial = np.ones((2, 2, 2), dtype=bool)
    part = np.ones((2, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="three-dimensional"):
        grid_material_metrics(np.ones(4), part, initial, 1)
    with pytest.raises(ValueError, match="shapes must match"):
        grid_material_metrics(part, np.ones((3, 3, 3)), initial, 1)
    for value in (0, -1, np.inf, np.nan, "unset"):
        with pytest.raises(ValueError, match="voxel_mm"):
            grid_material_metrics(part, part, initial, value)
    with pytest.raises(ValueError, match="surface_origin_mm"):
        grid_material_metrics(part, part, initial, 1, surface_origin_mm=(0, 1))
    remaining = initial.copy()
    remaining[0, 0, 0] = False
    with pytest.raises(ValueError, match="outside the declared initial stock"):
        grid_material_metrics(remaining, part, np.zeros_like(initial), 1)


def test_nearest_surface_distance_memory_guard_is_explicit():
    # Equal shapes are required. These point grids intentionally exceed the cap.
    remaining = np.ones((100, 100, 1), dtype=bool)
    target = np.ones((100, 100, 1), dtype=bool)
    with pytest.raises(ValueError, match="20M-entry"):
        surface_error_metrics(remaining, target, 1)
