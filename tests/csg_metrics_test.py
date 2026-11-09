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
        "removal_target_volume_mm3": 5 / 8,
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


def test_operation_target_encloses_original_part_and_excluded_stock_is_distinct():
    initial = np.ones((2, 2, 2), dtype=bool)
    original = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool).reshape(2, 2, 2)
    target = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool).reshape(2, 2, 2)
    assert not np.any(original & ~target)           # P ⊆ T: T adds one excluded voxel
    remaining = np.array([1, 0, 1, 0, 1, 1, 1, 1], dtype=bool).reshape(2, 2, 2)
    metrics = compute_grid_metrics(remaining, target, initial, 0.5,
                                   original_part_mask=original)
    # Excluded = initial ∧ T ∧ ¬P is exactly voxel 2, not the other six cells.
    assert metrics["excluded_volume_mm3"].value == pytest.approx(1 / 8)
    assert metrics["removal_target_volume_mm3"].value == pytest.approx(5 / 8)
    assert metrics["residual_volume_mm3"].value == pytest.approx(4 / 8)
    assert metrics["gouge_volume_mm3"].value == pytest.approx(1 / 8)
    assert metrics["original_part_gouge_volume_mm3"].value == pytest.approx(1 / 8)
    assert metrics["original_part_volume_mm3"].value == pytest.approx(2 / 8)
    assert metrics["residual_fraction"].value == pytest.approx(4 / 5)
    assert metrics["gouge_fraction"].value == pytest.approx(1 / 3)  # denominator |initial ∧ T| = 3/8
    assert metrics["operation_target_removed_fraction"].value == pytest.approx(1 / 5)
    empty_original = compute_grid_metrics(remaining, target, initial, 1,
                                           original_part_mask=np.zeros_like(initial))
    # Empty P is allowed; the fraction is based on T, not on an empty P.
    assert empty_original["gouge_fraction"].value == pytest.approx(1 / 3)
    assert empty_original["original_part_gouge_volume_mm3"].value == 0.0
    outside_target = np.zeros_like(initial)
    outside_target[-1] = True
    with pytest.raises(ValueError, match="subset"):
        compute_grid_metrics(remaining, target, initial, 1, original_part_mask=outside_target)


def test_excluded_volume_repro_perfect_default_and_36_voxel_layer():
    # Default P=T, a perfect remaining stock: no excluded stock and no cut.
    initial = np.ones((2, 2, 2), dtype=bool)
    default = compute_grid_metrics(initial, initial.copy(), initial, 1.0,
                                  declared_excluded_volume_mm3=0.0,
                                  exclusion_tolerance_mm3=0.0)
    assert default["excluded_volume_mm3"].value == 0.0
    assert default["gouge_volume_mm3"].value == 0.0
    assert default["original_part_gouge_volume_mm3"].value == 0.0

    # T=initial, P=initial except a 6x6 middle Z layer. Removing only the
    # excluded layer must count as target gouge, not original-part gouge.
    initial = np.ones((6, 6, 3), dtype=bool)
    target = initial.copy()
    original = initial.copy()
    original[:, :, 1] = False              # 36 excluded voxels: initial ∧ T ∧ ¬P
    remaining = original.copy()
    result = compute_grid_metrics(remaining, target, initial, 1.0,
                                  original_part_mask=original,
                                  declared_excluded_volume_mm3=36.0,
                                  exclusion_tolerance_mm3=0.0)
    assert result["excluded_volume_mm3"].value == 36.0
    assert result["removed_volume_mm3"].value == 36.0
    assert result["gouge_volume_mm3"].value == 36.0
    assert result["original_part_gouge_volume_mm3"].value == 0.0
    assert result["original_part_volume_mm3"].value == 72.0
    assert result["gouge_fraction"].value == pytest.approx(36 / 108)
    assert result["residual_volume_mm3"].value == 0.0
    assert result["removal_target_volume_mm3"].value == 0.0
    assert result["original_part_gouge_volume_mm3"].denominator.startswith(
        "original_part_volume_mm3=72.0; diagnostic only")
    # An explicit reported tolerance is required and retained as caller policy.
    compute_grid_metrics(remaining, target, initial, 1.0, original_part_mask=original,
                         declared_excluded_volume_mm3=35.0, exclusion_tolerance_mm3=1.0)
    with pytest.raises(ValueError, match="excluded volume mismatch"):
        compute_grid_metrics(remaining, target, initial, 1.0, original_part_mask=original,
                             declared_excluded_volume_mm3=34.0, exclusion_tolerance_mm3=1.0)
    with pytest.raises(ValueError, match="tolerance must be finite nonnegative"):
        compute_grid_metrics(remaining, target, initial, 1.0, original_part_mask=original,
                             declared_excluded_volume_mm3=36.0, exclusion_tolerance_mm3=-1.0)
    with pytest.raises(ValueError, match="supplied together"):
        compute_grid_metrics(remaining, target, initial, 1.0, original_part_mask=original,
                             exclusion_tolerance_mm3=1.0)


def test_cut_only_excluded_layer_uses_target_gouge_for_geometry_pass():
    initial = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool).reshape(2, 2, 2)
    target = initial.copy()
    original = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool).reshape(2, 2, 2)
    remaining = original.copy()              # remove only T∧¬P
    result = compute_grid_metrics(remaining, target, initial, 0.5,
                                  original_part_mask=original)
    assert result["excluded_volume_mm3"].value == pytest.approx(1 / 8)
    assert result["gouge_volume_mm3"].value == pytest.approx(1 / 8)
    assert result["original_part_gouge_volume_mm3"].value == 0.0
    assert result["gouge_fraction"].value == pytest.approx(1 / 3)
    assert result["residual_volume_mm3"].value == 0.0
    assert result["max_surface_error_mm"].value <= 1.0
    allowed = C.Tolerances(0.0, 1.0, 1.0, 0.0)
    strict = C.Tolerances(0.0, 0.0, 1.0, 0.0)
    assert C.geometry_pass_from_metrics(result, allowed) is True
    # P's diagnostic gouge is zero; geometry_pass still fails on removed T.
    assert C.geometry_pass_from_metrics(result, strict) is False


def test_empty_operation_target_denominators_are_not_fake_zero_or_one():
    initial = np.ones((2, 2, 2), dtype=bool)
    metrics = compute_grid_metrics(np.zeros_like(initial), initial, initial, 2.0)
    for name in ("operation_target_removed_fraction", "residual_fraction"):
        assert metrics[name].value is None
        assert "removal target volume is zero" in metrics[name].null_reason
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


def test_large_full_stock_surfaces_use_bounded_nearest_neighbour():
    # 24,056 boundary points per side -> 578,691,136 pairs. This formerly
    # exceeded the 20M-entry cdist guard even though the grid itself is modest.
    remaining = np.ones((64, 64, 64), dtype=bool)
    target = remaining.copy()
    result = compute_grid_metrics(remaining, target, remaining, 1.0)
    for name in ("max_surface_error_mm", "p99_surface_error_mm", "asd_mm", "hd95_mm"):
        assert result[name].value == 0.0
    assert result["initial_stock_volume_mm3"].value == 64**3
    assert result["remaining_volume_mm3"].value == 64**3
    assert result["iou"].value == pytest.approx(1.0)


def test_surface_count_guard_precedes_argwhere(monkeypatch):
    # A shallow (20001,500,1) sheet is cheap to hold, but has more than ten
    # million boundary cells under the default zero-border erosion policy.
    remaining = np.ones((20001, 500, 1), dtype=bool)
    def forbidden_argwhere(*args, **kwargs):
        raise AssertionError("argwhere must not be called before the count guard")
    monkeypatch.setattr(np, "argwhere", forbidden_argwhere)
    with pytest.raises(ValueError, match="surface sample count exceeds explicit 10M"):
        surface_error_metrics(remaining, remaining.copy(), 1.0)
