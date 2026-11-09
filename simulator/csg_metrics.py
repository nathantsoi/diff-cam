"""Grid-based geometric metrics with explicit physical units.

Masks use ``True`` for material occupancy. The primary grid helper takes three
same-shaped masks: initial stock, desired final part (operation target), and
remaining material after motion. Residual is *unremoved removal target*
``initial & ~part & remaining``; gouge is *removed desired part*
``initial & part & ~remaining``. They are not interchangeable.

``voxel_mm`` is one cubic cell's physical edge in millimetres. Volumes use
``voxel_mm**3`` regardless of the third-power exponent in their names. Surface
errors are Euclidean nearest distances between **cell centres on voxel
surfaces**, not areas, exact mesh Hausdorff distances, or machining tolerances.
"""
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

from benchmarks.contracts import Metric


_MM_CUBE_UNIT = "mm^3"
_SURFACE_VARIABLE_UNIT = "mm"


def sdf_to_mask(sdf, threshold=0.0):
    """Convert SDF to binary occupancy. Negative SDF denotes solid material."""
    return sdf < threshold


def extract_surface(mask):
    """Occupied cells touching empty cells under scipy's six-neighbour erosion."""
    eroded = binary_erosion(mask)
    return mask ^ eroded


def surface_points(mask):
    """Legacy integer-grid surface samples as float32 ``(row, column, plane)``."""
    pts = np.argwhere(mask)
    return pts.astype(np.float32)


def dice_score(pred_mask, target_mask, eps=1e-8):
    """Legacy epsilon-smoothed Dice. Two empty masks return one for eps>0."""
    pred = pred_mask.astype(np.bool_)
    target = target_mask.astype(np.bool_)
    intersection = np.logical_and(pred, target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred_mask, target_mask, eps=1e-8):
    """Intersection over union using the legacy Dice family's empty-set policy."""
    pred = pred_mask.astype(np.bool_)
    target = target_mask.astype(np.bool_)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return (intersection + eps) / (union + eps)


def surface_distances(pred_mask, target_mask, spacing=(1.0, 1.0, 1.0)):
    """Legacy bidirectional nearest-surface distances in index spacing units.

    Empty-surface behaviour remains a ValueError in this low-level helper.
    ``surface_error_metrics`` is the contract-facing API with explicit nulls.
    """
    pred_surface = extract_surface(pred_mask)
    target_surface = extract_surface(target_mask)
    pred_pts = surface_points(pred_surface)
    target_pts = surface_points(target_surface)
    pred_pts = pred_pts * np.array(spacing)
    target_pts = target_pts * np.array(spacing)
    if len(pred_pts) == 0 or len(target_pts) == 0:
        raise ValueError("Empty surface encountered.")
    dists = cdist(pred_pts, target_pts)
    return dists.min(axis=1), dists.min(axis=0)


def average_surface_distance(pred_mask, target_mask, spacing=(1.0, 1.0, 1.0)):
    """Legacy symmetric mean of both directions' mean nearest distances."""
    d1, d2 = surface_distances(pred_mask, target_mask, spacing)
    return (d1.mean() + d2.mean()) / 2.0


def hd95(pred_mask, target_mask, spacing=(1.0, 1.0, 1.0)):
    """Legacy combined-direction 95th percentile nearest-surface distance."""
    d1, d2 = surface_distances(pred_mask, target_mask, spacing)
    return np.percentile(np.concatenate([d1, d2]), 95.0)


def _gouge(pred_mask, target_mask):
    """Legacy count of desired-part material removed from the remaining mask."""
    pred = pred_mask.astype(np.bool_)
    target = target_mask.astype(np.bool_)
    return np.logical_and(target, ~pred).sum()


def _residual(pred_mask, target_mask):
    """Legacy count of unremoved removal-target cells in the remaining mask."""
    pred = pred_mask.astype(np.bool_)
    target = target_mask.astype(np.bool_)
    return np.logical_and(~target, pred).sum()


def _spacing(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("voxel_mm/spacing: expected one finite positive length in mm") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError("voxel_mm/spacing: expected one finite positive length in mm")
    return result


def _origin(value) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float64)
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("surface_origin_mm: expected three finite mm coordinates") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("surface_origin_mm: expected three finite mm coordinates")
    return result


def _same_masks(*masks) -> Sequence[np.ndarray]:
    if len(masks) != 3:
        raise ValueError("remaining, operation_target and initial masks are all required")
    arrays = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if any(array.ndim != 3 for array in arrays):
        raise ValueError("grid metric masks must be three-dimensional (Nx,Ny,Nz)")
    if any(array.size == 0 for array in arrays):
        raise ValueError("grid metric masks must not be empty")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError(f"grid metric mask shapes must match: {[a.shape for a in arrays]}")
    return arrays


def _unknown_surface_metrics(reason: str) -> Dict[str, Metric]:
    return {name: Metric(None, _SURFACE_VARIABLE_UNIT, null_reason=reason)
            for name in ("max_surface_error_mm", "p99_surface_error_mm", "asd_mm", "hd95_mm")}


def surface_error_metrics(remaining_mask, target_mask, spacing_mm=1.0,
                          origin_mm=None) -> Dict[str, Metric]:
    """Nearest voxel-centre surface errors with defined empty-surface nulls.

    ``max_surface_error_mm``: greatest nearest-cell-centre distance in either
    direction. ``asd_mm``: unweighted mean over *all* remaining-to-target and
    target-to-remaining samples. ``hd95_mm``/``p99_surface_error_mm``: 95th/99th
    linear percentiles of that same concatenated sample distribution. This is
    an asymmetry between target surfaces; no area weighting or continuous-mesh
    Hausdorff guarantee is implied.
    """
    remaining, target, _ = _same_masks(remaining_mask, target_mask, target_mask)
    remaining_surface = np.asarray(remaining & ~np.asarray(binary_erosion(remaining)), dtype=bool)
    target_surface = np.asarray(target & ~np.asarray(binary_erosion(target)), dtype=bool)
    reason = ("nearest-cell-centre surface distances require both surfaces to be nonempty; "
              f"remaining_surface_voxels={int(remaining_surface.sum())}, "
              f"operation_target_surface_voxels={int(target_surface.sum())}")
    if not remaining_surface.any() or not target_surface.any():
        return _unknown_surface_metrics(reason)

    spacing = _spacing(spacing_mm)
    origin = _origin(origin_mm)
    remaining_points = np.argwhere(remaining_surface).astype(np.float64) * spacing + origin
    target_points = np.argwhere(target_surface).astype(np.float64) * spacing + origin
    if len(remaining_points) > 10_000_000 or len(target_points) > 10_000_000:
        raise ValueError("surface sample count exceeds explicit 10M-point memory guard")
    if len(remaining_points) * len(target_points) > 20_000_000:
        raise ValueError("surface distance matrix exceeds explicit 20M-entry memory guard; use a coarser grid or a chunked distance implementation")
    distances = cdist(remaining_points, target_points)
    remaining_to_target = distances.min(axis=1)
    target_to_remaining = distances.min(axis=0)
    both = np.concatenate((remaining_to_target, target_to_remaining))
    maximum = float(both.max())
    return {
        "max_surface_error_mm": Metric(maximum, "mm", denominator="max of bidirectional nearest voxel-centre distances"),
        "p99_surface_error_mm": Metric(float(np.percentile(both, 99, method="linear")), "mm",
                                       denominator="linear p99 of all bidirectional nearest voxel-centre distances"),
        "asd_mm": Metric(float(both.mean()), "mm",
                         denominator="unweighted mean of all bidirectional nearest voxel-centre distances"),
        "hd95_mm": Metric(float(np.percentile(both, 95, method="linear")), "mm",
                          denominator="linear p95 of all bidirectional nearest voxel-centre distances"),
    }


def grid_material_metrics(remaining_mask, target_mask, initial_mask, voxel_mm,
                          *, surface_origin_mm=None) -> Dict[str, Metric]:
    """Return volumes, fractions, overlap and sampled errors for a material grid.

    ``initial_mask`` is required: never infer stock from the final mask or from
    grid dimensions alone. A part-empty or removal-target-empty denominator is
    null with a reason, not misleadingly assigned zero or one. Empty-surface
    metrics are null even if IoU/Dice coincidentally equal one for empty masks.
    """
    remaining, part, initial = _same_masks(remaining_mask, target_mask, initial_mask)
    if np.any(remaining & ~initial):
        raise ValueError("remaining mask contains material outside the declared initial stock")
    voxel = _spacing(voxel_mm)
    cell_volume = voxel ** 3
    removed = initial & ~remaining
    removal_target = initial & ~part
    residual = removal_target & remaining
    gouge = removed & part
    intersection = remaining & part
    union = remaining | part

    removal_target_volume = float(removal_target.sum()) * cell_volume
    part_volume = float(part.sum()) * cell_volume
    numbers = {
        "remaining_volume_mm3": int(remaining.sum()),
        "removed_volume_mm3": int(removed.sum()),
        "residual_volume_mm3": int(residual.sum()),
        "gouge_volume_mm3": int(gouge.sum()),
        "initial_stock_volume_mm3": int(initial.sum()),
        "operation_target_removed_volume_mm3": int(removal_target.sum()),
        "original_part_volume_mm3": int(part.sum()),
    }
    result = {
        "iou": Metric(float((intersection.sum() + 1e-12) / (union.sum() + 1e-12)), "1",
                      denominator=f"union_voxels={int(union.sum())}"),
        "dice": Metric(dice_score(remaining, part), "1",
                       denominator=f"remaining_voxels+target_voxels={int(remaining.sum() + part.sum())}"),
    }
    for name, count in numbers.items():
        result[name] = Metric(count * cell_volume, _MM_CUBE_UNIT,
                              denominator=f"voxel_count={count}, voxel_mm={voxel}")
    result.update({
        "voxel_volume_mm3": Metric(cell_volume, _MM_CUBE_UNIT),
        "operation_target_removed_fraction": (
            Metric(float((removal_target & removed).sum() / removal_target.sum()), "1",
                   denominator=f"initial & ~part volume_mm3={removal_target_volume}")
            if removal_target.any() else
            Metric(None, "1", null_reason="operation target volume is zero (initial & ~part is empty)")),
        "residual_fraction": (
            Metric(float(residual.sum() / removal_target.sum()), "1",
                   denominator=f"initial & ~part volume_mm3={removal_target_volume}")
            if removal_target.any() else
            Metric(None, "1", null_reason="operation target volume is zero (initial & ~part is empty)")),
        "gouge_fraction": (
            Metric(float(gouge.sum() / part.sum()), "1", denominator=f"original_part_volume_mm3={part_volume}")
            if part.any() else Metric(None, "1", null_reason="original part volume is zero")),
    })
    result.update(surface_error_metrics(remaining, part, spacing_mm=voxel,
                                        origin_mm=surface_origin_mm))
    return result


# Concise public alias; the longer name documents the occupancy semantics.
compute_grid_metrics = grid_material_metrics


__all__ = [
    "sdf_to_mask", "extract_surface", "surface_points", "dice_score", "iou_score",
    "surface_distances", "average_surface_distance", "hd95", "surface_error_metrics",
    "grid_material_metrics", "compute_grid_metrics", "_gouge", "_residual",
]
