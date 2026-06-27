"""Trajectory-similarity metrics.

The original trajectory (waypoints) and the executed trajectory (dense, time
sampled) have different point counts, so similarity must be measured in a
parameterisation-invariant way. We provide:

  * ``discrete_frechet`` -- primary geometric path-agreement metric
  * ``dtw_distance``     -- dynamic time warping distance
  * ``resampled_rmse``   -- RMSE after arc-length resampling both paths
  * ``waypoint_roundtrip_error`` -- max error of recovered export/parse waypoints

All operate on ``(N, 3)`` arrays in unit-cube coordinates. Pass ``scale`` (e.g.
``MachineConfig.workspace_mm``) to report distances in millimetres.
"""

import numpy as np
from scipy.spatial.distance import cdist


def _as_xyz(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected (N, 3) array, got {a.shape}")
    return a


def discrete_frechet(a, b, scale=1.0, resample=400):
    """Discrete Fréchet distance between two polylines.

    Iterative (stack-safe) version of the classic Eiter & Mannila coupling
    measure over the pairwise distance matrix.

    The discrete measure only couples *vertices*, so comparing a sparse polyline
    (e.g. corner waypoints) against a dense one would report the vertex spacing
    rather than the geometric path gap. To make this a true path-similarity
    metric we arc-length-resample both inputs to ``resample`` points first; pass
    ``resample=None`` to compare the raw vertices.
    """
    a = _as_xyz(a)
    b = _as_xyz(b)
    if resample is not None:
        a = _resample_by_arclength(a, resample)
        b = _resample_by_arclength(b, resample)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    d = cdist(a, b)
    ca = np.full((n, m), -1.0)
    ca[0, 0] = d[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d[i, j])
    return float(ca[-1, -1]) * scale


def dtw_distance(a, b, scale=1.0, resample=400):
    """Mean dynamic-time-warping distance along the optimal monotone alignment.

    Returns the accumulated matched-point distance normalised by the alignment
    length (a per-point average), so the value is comparable across trajectories
    of different sizes. Both inputs are arc-length-resampled to ``resample``
    points first (pass ``resample=None`` to use the raw points)."""
    a = _as_xyz(a)
    b = _as_xyz(b)
    if resample is not None:
        a = _resample_by_arclength(a, resample)
        b = _resample_by_arclength(b, resample)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    d = cdist(a, b)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row = d[i - 1]
        for j in range(1, m + 1):
            acc[i, j] = row[j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m]) / (n + m) * scale


def _resample_by_arclength(path, n):
    """Resample a polyline to ``n`` points evenly spaced by cumulative arc length."""
    path = _as_xyz(path)
    if len(path) == 1:
        return np.repeat(path, n, axis=0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-12:
        return np.repeat(path[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(targets, cum, path[:, axis])
    return out


def resampled_rmse(a, b, n=200, scale=1.0):
    """RMSE between two paths after arc-length resampling both to ``n`` points."""
    ra = _resample_by_arclength(a, n)
    rb = _resample_by_arclength(b, n)
    diff = np.linalg.norm(ra - rb, axis=1)
    return float(np.sqrt(np.mean(diff ** 2))) * scale


def waypoint_roundtrip_error(original, recovered, scale=1.0):
    """Max per-coordinate absolute error between original and recovered
    waypoints (both ``(T, 3)``)."""
    original = _as_xyz(original)
    recovered = _as_xyz(recovered)
    if original.shape != recovered.shape:
        raise ValueError(
            f"shape mismatch: {original.shape} vs {recovered.shape}"
        )
    return float(np.max(np.abs(original - recovered))) * scale


def all_metrics(original, executed, scale=1.0):
    """Return a dict of every similarity metric (reported at ``scale``)."""
    return {
        "frechet": discrete_frechet(original, executed, scale),
        "dtw": dtw_distance(original, executed, scale),
        "resampled_rmse": resampled_rmse(original, executed, scale=scale),
    }
