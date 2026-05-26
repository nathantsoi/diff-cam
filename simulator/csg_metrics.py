import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

# ============================================================
# Helpers
# ============================================================

def sdf_to_mask(sdf, threshold=0.0):
    """
    Convert SDF to binary occupancy mask.

    Negative SDF = solid material.
    """
    return sdf < threshold


def extract_surface(mask):
    """
    Extract surface voxels from a binary volume.

    Surface = occupied voxel touching empty space.
    """
    eroded = binary_erosion(mask)
    surface = mask ^ eroded
    return surface


def surface_points(mask):
    """
    Convert surface voxels to Nx3 point cloud.
    """
    pts = np.argwhere(mask)
    return pts.astype(np.float32)


# ============================================================
# Dice
# ============================================================

def dice_score(pred_mask, target_mask, eps=1e-8):
    """
    Dice similarity coefficient.

    Returns:
        float in [0, 1]
    """
    pred = pred_mask.astype(np.bool_)
    target = target_mask.astype(np.bool_)

    intersection = np.logical_and(pred, target).sum()

    return (
        2.0 * intersection + eps
    ) / (
        pred.sum() + target.sum() + eps
    )


# ============================================================
# Surface Distance Utilities
# ============================================================

def surface_distances(pred_mask, target_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Compute bidirectional surface distances.

    Returns:
        d_pred_to_target
        d_target_to_pred
    """

    pred_surface = extract_surface(pred_mask)
    target_surface = extract_surface(target_mask)

    pred_pts = surface_points(pred_surface)
    target_pts = surface_points(target_surface)

    # Apply voxel spacing
    pred_pts = pred_pts * np.array(spacing)
    target_pts = target_pts * np.array(spacing)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        raise ValueError("Empty surface encountered.")

    dists = cdist(pred_pts, target_pts)

    d_pred_to_target = dists.min(axis=1)
    d_target_to_pred = dists.min(axis=0)

    return d_pred_to_target, d_target_to_pred


# ============================================================
# ASD
# ============================================================

def average_surface_distance(pred_mask, target_mask,
                             spacing=(1.0, 1.0, 1.0)):
    """
    Symmetric Average Surface Distance (ASD).

    Lower is better.
    """

    d1, d2 = surface_distances(
        pred_mask,
        target_mask,
        spacing
    )

    return (d1.mean() + d2.mean()) / 2.0


# ============================================================
# HD95
# ============================================================

def hd95(pred_mask, target_mask,
         spacing=(1.0, 1.0, 1.0)):
    """
    95th percentile Hausdorff distance.

    Lower is better.
    """

    d1, d2 = surface_distances(
        pred_mask,
        target_mask,
        spacing
    )

    all_dists = np.concatenate([d1, d2])

    return np.percentile(all_dists, 95.0)


def _gouge(pred_mask, target_mask):
    """
    Total gouge volume.

    Higher is worse.
    """
    pred = pred_mask.astype(np.bool_)
    target = target_mask.astype(np.bool_)

    gouge_volume = np.logical_and(target, ~pred).sum()

    return gouge_volume


def _residual(pred_mask, target_mask):
    """
    Total residual volume.

    Higher is worse.
    """
    pred = pred_mask.astype(np.bool_)
    target = target_mask.astype(np.bool_)

    residual_volume = np.logical_and(~target, pred).sum()

    return residual_volume