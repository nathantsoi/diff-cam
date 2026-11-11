"""Qualified CAMotics v1.2.0 binary-Docker route and independent grid sampling.

Only analytic box stock/target geometry, flat fixed cylindrical tools, known
initial state, pure work-frame translations and G54 source motion are supported.
The exact calibrated Docker image ID and the binary's reported version are
checked before simulation. G54 is calibrated in a separately retained simulation
program using ``machine - initial_state`` coordinates; raw caller NC is not
rewritten and both SHA-256 values are preserved. Explicit G53 is unsupported.

CAMotics carves rapid moves too. Conflicting rapid cutter AABB/stock overlaps
are therefore rejected rather than reported as feed-removal agreement. Mesh
queries use explicit closed-surface policies and do not require watertightness,
but raw topology stays labeled and a non-watertight result never receives a
passing geometry decision. ``safety_pass`` is always null: shank/holder
clearance, whole-cell bounds, fixtures and controller timing are not covered.

``evaluate_candidate`` posts typed motion before validation/simulation;
``evaluate_posted_gcode`` consumes actual NC bytes. Arbitrary sampled/STEP
stock, rotated setups, unsupported tools, invalid source programs and malformed
meshes fail closed. A successful camsim exit is not a machining-safety verdict.

The invocation uses MCUBES rendering (mode 0) with one thread. In this
release the alternate CMS mode has empty doSlice/doCell implementations and
produces no facets, so it is not a supported alternative. Parallel emission
also produced edge-incidence defects in the tested fixtures. Watertight means
exact-vertex two-face incidence and consistent shared-edge orientation; this
still does not prove global non-self-intersection or correct swept geometry.
The ConicSweep endpoint behavior is not a faithful arbitrary closed cylinder
union, and mandatory independent line-geometry checks must not use CAMotics as
their only oracle.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time
from typing import Any, Dict, Mapping, Optional, Tuple
from uuid import uuid4

import numpy as np

from benchmarks import contracts as C
from cam.config import MachineConfig
from cam.posts import post_motion_plan
from cam.gcode_parser import GCodeError, UnsupportedGCodeError, parse_motion_plan
from simulator.csg_metrics import grid_material_metrics


EVALUATOR_ID = "camotics"
DEFAULT_IMAGE = "diffcam-camotics:1.2.0"
DEFAULT_RELEASE = "1.2.0"
# Exact image used for the coordinate-behavior calibration and bias report.
# Changing this digest requires requalification, not merely a new image tag.
DEFAULT_IMAGE_DIGEST = "sha256:e680bff023d9190ed16fe1bee21d9c3c72f0e35bfbceb0a19e704c62f6ddca80"


class AdapterError(RuntimeError):
    """Actionable unsupported input or failed mesh/project interpretation."""


@dataclass(frozen=True)
class TriangleMesh:
    vertices_mm: np.ndarray             # (N, 3, 3), actual triangle vertices
    stl_format: str

    def __post_init__(self):
        try:
            vertices = np.asarray(self.vertices_mm, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise AdapterError("mesh triangles must be numeric vertices in (N,3,3) form") from error
        if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
            raise AdapterError(f"mesh triangles require shape (N,3,3), got {vertices.shape}")
        object.__setattr__(self, "vertices_mm", vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.vertices_mm)


@dataclass(frozen=True)
class MeshSample:
    """Grid occupancy and sampling provenance.

    ``surface_contact_count`` counts distinct input point locations (exact
    coordinates) whose returned surface root matches within the tolerance; it is
    not a physical-contact volume or safety bound. Repeated ray work remains in
    ``projected_ray_queries`` (including repeated input locations).
    """
    occupancy: np.ndarray
    surface_contact_count: int
    projected_ray_queries: int
    touch_tolerance_mm: float
    boundary_policy: str
    spatial_bin_count: int = 0


def _finite_positive(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AdapterError(f"{name}: expected a finite positive value") from error
    if not math.isfinite(result) or result <= 0:
        raise AdapterError(f"{name}: expected a finite positive value")
    return result


def _finite_nonnegative(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AdapterError(f"{name}: expected a finite nonnegative value") from error
    if not math.isfinite(result) or result < 0:
        raise AdapterError(f"{name}: expected a finite nonnegative value")
    return result


def read_stl(path: Path, *, max_triangles: int = 2_000_000) -> TriangleMesh:
    """Read a finite, uncorrupted binary or ASCII triangle STL (no geometry units inference)."""
    path = Path(path)
    max_triangles = int(max_triangles)
    if max_triangles < 1:
        raise AdapterError("max_triangles must be positive")
    data = path.read_bytes()
    triangles = None
    stl_format = ""
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if count <= max_triangles and 84 + count * 50 == len(data):
            if count:
                raw = np.frombuffer(data, dtype=np.dtype([
                    ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")
                ]), count=count, offset=84)
                triangles = raw["vertices"].astype(np.float64)
            else:
                triangles = np.empty((0, 3, 3), dtype=np.float64)
            stl_format = "binary"
    if triangles is None and data.lstrip().lower().startswith(b"solid"):
        vertices = []
        try:
            text_lines = data.decode("ascii").splitlines()
        except UnicodeDecodeError as error:
            raise AdapterError(f"non-ASCII bytes in {path}; neither valid binary nor text STL") from error
        for line in text_lines:
            words = line.strip().split()
            if len(words) == 4 and words[0].lower() == "vertex":
                try:
                    values = [float(item) for item in words[1:]]
                except ValueError as error:
                    raise AdapterError(f"non-numeric ASCII STL vertex in {path}") from error
                vertices.append(values)
        if len(vertices) % 3:
            raise AdapterError("ASCII STL vertex count is not divisible by three")
        if len(vertices) // 3 > max_triangles:
            raise AdapterError("STL exceeds declared max_triangles")
        triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
        stl_format = "ascii"
    if triangles is None:
        raise AdapterError(f"{path}: expected a complete binary or ASCII triangle STL")
    if not np.isfinite(triangles).all():
        raise AdapterError("STL contains non-finite or unusable coordinates")
    areas = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 0] - triangles[:, 2])
    degenerate = np.linalg.norm(areas, axis=1) == 0.0
    if degenerate.any():
        raise AdapterError(f"STL contains {int(degenerate.sum())} exactly degenerate facets")
    return TriangleMesh(triangles, stl_format)


def mesh_volume_mm3(mesh: TriangleMesh, *, require_watertight: bool = True) -> Dict[str, Any]:
    """Triangulated signed volume plus closure facts; no absolute-value repair.

    Coordinate origin shift only reduces cancellation. Caller labels this STL
    diagnostic volume (possibly one-cell-padded), never exact task-stock volume.
    ``watertight`` means closed two-face incidence with opposite directed
    shared edges, not a global self-intersection or machining-validity proof.
    """
    vertices = mesh.vertices_mm
    if len(vertices) == 0:
        return {"volume_mm3": 0.0, "watertight": True, "boundary_or_nonmanifold_edges": 0,
                "inconsistent_direction_edges": 0}
    flat = vertices.reshape(-1, 3)
    _, inverse = np.unique(flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    ascending = directed[:, 0] <= directed[:, 1]
    sorted_edges = np.sort(directed, axis=1)
    _, inverse, edge_counts = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
    # Two incident facets must also have opposite directed shared edges. A
    # count of two alone lets identically oriented shared edges look watertight.
    direction_balance = np.bincount(inverse.reshape(-1),
                                    weights=np.where(ascending, 1, -1))
    inconsistent = (edge_counts == 2) & (direction_balance != 0)
    inconsistent_count = int(inconsistent.sum())
    defects = int(np.count_nonzero((edge_counts != 2) | inconsistent))
    if defects and require_watertight:
        raise AdapterError(f"mesh is not a closed two-face manifold with consistent orientation: {defects} incidence/direction defects")
    origin = vertices.reshape(-1, 3).mean(axis=0)
    signed = sum(np.dot(t[0] - origin, np.cross(t[1] - origin, t[2] - origin)) for t in vertices)
    signed /= 6.0
    if not math.isfinite(signed):
        raise AdapterError("mesh signed volume is not finite")
    if signed < 0:
        raise AdapterError("mesh triangle winding produces a negative signed volume; not repaired with an absolute value")
    return {"volume_mm3": float(signed), "watertight": defects == 0,
            "boundary_or_nonmanifold_edges": defects,
            "inconsistent_direction_edges": inconsistent_count}


def points_in_mesh(mesh: TriangleMesh, points_mm: np.ndarray, *,
                   touch_tolerance_mm: float = 1e-7, query_chunk: int = 256) -> MeshSample:
    """Closed-surface-inclusive +Z ray parity without watertightness assumptions.

    AABB-binned queries avoid a full-mesh scan for every voxel centre. Shared
    projection roots are merged within tolerance. A three-dimensional
    point-in-triangle check also includes vertical and edge contacts, which a
    projected-ray-only contact test would miss. ``surface_contact_count`` is
    still a count of distinct exact input locations, not a physical-contact or
    safety bound. Duplicate input work remains in ``projected_ray_queries``.
    """
    if touch_tolerance_mm <= 0 or not math.isfinite(touch_tolerance_mm):
        raise AdapterError("touch_tolerance_mm must be finite and positive")
    query_chunk = int(query_chunk)
    if query_chunk < 1:
        raise AdapterError("query_chunk must be at least one")
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise AdapterError("point queries need an N by 3 finite coordinate array")
    if not np.isfinite(points).all():
        raise AdapterError("point queries need finite millimetre coordinates")
    if len(points) > 1_000_000:
        raise AdapterError("point query count exceeds explicit one-million-point guard")
    if len(mesh.vertices_mm) == 0:
        return MeshSample(np.zeros(len(points), dtype=bool), 0, len(points), touch_tolerance_mm, "closed_include")

    vertices = mesh.vertices_mm
    lower = vertices.min(axis=(0, 1)) - touch_tolerance_mm
    upper = vertices.max(axis=(0, 1)) + touch_tolerance_mm
    tri_lower = vertices.min(axis=1)
    tri_upper = vertices.max(axis=1)
    extent = upper - lower
    if not np.isfinite(extent).all():
        raise AdapterError("mesh coordinate range is too large for finite spatial-bin arithmetic")
    median_extent = float(np.median(np.max(tri_upper[:, :2] - tri_lower[:, :2], axis=1)))
    bin_width = max(4.0 * median_extent, float(np.max(extent[:2])) / 64, touch_tolerance_mm * 8, 1e-12)
    bin_origin = lower[:2]
    # Sparse bins handle arbitrarily shaped meshes without allocating a giant
    # dense spatial array. The bound avoids quadratic tiny-facet bin filling.
    bin_keys = {}
    for index in range(len(vertices)):
        lo = np.floor((tri_lower[index, :2] - bin_origin - touch_tolerance_mm) / bin_width).astype(np.int64)
        hi = np.floor((tri_upper[index, :2] - bin_origin + touch_tolerance_mm) / bin_width).astype(np.int64)
        if np.any(hi < lo) or int(np.prod(hi - lo + 1)) > 100_000:
            raise AdapterError("triangle spans an excessive AABB-bin range; reject oversized/degenerate coordinate data")
        for x in range(int(lo[0]), int(hi[0]) + 1):
            for y in range(int(lo[1]), int(hi[1]) + 1):
                bin_keys.setdefault((x, y), []).append(index)
    bins = {key: np.asarray(value, dtype=np.int64) for key, value in bin_keys.items()}
    results = np.zeros(len(points), dtype=bool)
    contacts = set()
    projected_queries = 0
    for start in range(0, len(points), query_chunk):
        for offset, point in enumerate(points[start:start + query_chunk]):
            if np.any(point < lower) or np.any(point > upper):
                continue
            key = np.floor((point[:2] - bin_origin) / bin_width).astype(np.int64)
            indices = bins.get((int(key[0]), int(key[1])))
            if indices is None:
                continue
            a = vertices[indices, 0]
            b = vertices[indices, 1]
            c = vertices[indices, 2]
            u, v, r = b - a, c - a, point - a

            # Actual closed-triangle proximity includes vertical faces. The
            # Gram determinant is positive for non-degenerate STL triangles.
            uu = np.einsum("ij,ij->i", u, u)
            vv = np.einsum("ij,ij->i", v, v)
            uv = np.einsum("ij,ij->i", u, v)
            ru = np.einsum("ij,ij->i", r, u)
            rv = np.einsum("ij,ij->i", r, v)
            gram = uu * vv - uv * uv
            nondegenerate = gram > 0.0
            wb3 = np.zeros(len(u), dtype=float)
            wc3 = np.zeros(len(u), dtype=float)
            wb3[nondegenerate] = ((ru * vv - rv * uv) / gram)[nondegenerate]
            wc3[nondegenerate] = ((rv * uu - ru * uv) / gram)[nondegenerate]
            wa3 = 1.0 - wb3 - wc3
            squared_distance = np.einsum("ij,ij->i", r, r)
            height2 = squared_distance - wb3 * ru - wc3 * rv
            bary_tolerance = touch_tolerance_mm / max(bin_width, touch_tolerance_mm)
            close = (nondegenerate & (wa3 >= -bary_tolerance) & (wb3 >= -bary_tolerance)
                     & (wc3 >= -bary_tolerance) & (height2 <= touch_tolerance_mm ** 2))

            denominator = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
            projected = np.abs(denominator) > 1e-14  # projected area, mm^2: not a length tolerance
            if projected.any():
                projected_queries += 1
            if close.any():
                results[start + offset] = True
                contacts.add(tuple(float(value) for value in point))
                continue
            if not projected.any():
                continue
            up, vp, rp, denom = u[projected], v[projected], r[projected], denominator[projected]
            aa = a[projected, 2]
            bb = b[projected, 2]
            cc = c[projected, 2]
            wb = (rp[:, 0] * vp[:, 1] - vp[:, 0] * rp[:, 1]) / denom
            wc = (up[:, 0] * rp[:, 1] - rp[:, 0] * up[:, 1]) / denom
            valid = (wb >= -touch_tolerance_mm) & (wc >= -touch_tolerance_mm) & (wb + wc <= 1 + touch_tolerance_mm)
            if not valid.any():
                continue
            roots = np.sort(aa[valid] + wb[valid] * (bb[valid] - aa[valid]) + wc[valid] * (cc[valid] - aa[valid]))
            if roots.size and np.min(np.abs(roots - point[2])) <= touch_tolerance_mm:
                results[start + offset] = True
                contacts.add(tuple(float(value) for value in point))
                continue
            if roots.size:
                keep = np.concatenate(([True], np.diff(roots) > touch_tolerance_mm * 2))
                roots = roots[keep]
            roots = roots[roots > point[2] + touch_tolerance_mm]
            if len(roots) % 2:
                results[start + offset] = True
    return MeshSample(results, len(contacts), projected_queries, touch_tolerance_mm,
                      "closed_include", len(bins))


def _box_mask(points_mm: np.ndarray, minimum, maximum) -> np.ndarray:
    try:
        lower, upper = np.asarray(minimum, dtype=float), np.asarray(maximum, dtype=float)
    except (TypeError, ValueError) as error:
        raise AdapterError("box bounds must be numeric finite millimetre triples") from error
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise AdapterError("analytic box stock/target requires finite min_mm < max_mm triples in work frame")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise AdapterError("non-finite stock bounds")
    points = np.asarray(points_mm, dtype=float)
    return ((points >= lower) & (points <= upper)).all(axis=-1)


def _geometry_mask(source: C.GeometrySource, points_mm: np.ndarray) -> np.ndarray:
    """Mask declared analytic geometry in its work frame, with no mesh heuristics."""
    params = source.params
    primitive = params.get("primitive")
    if primitive == "box":
        required = {"min_mm", "max_mm"}
        bounds = (params.get("min_mm"), params.get("max_mm"))
        if set(params) - {"primitive", "min_mm", "max_mm"}:
            raise AdapterError("box: only primitive/min_mm/max_mm parameters are supported")
        return _box_mask(points_mm, *bounds)
    if primitive == "box_minus_multi_pocket":
        required_mp = {"box_min_mm", "box_max_mm", "pocket_min_mm", "pocket_max_mm"}
        if not required_mp.issubset(params):
            raise AdapterError(
                f"box_minus_multi_pocket: missing parameters {sorted(required_mp - set(params))}"
            )
        box_min, box_max = params["box_min_mm"], params["box_max_mm"]
        pocket_min, pocket_max = params["pocket_min_mm"], params["pocket_max_mm"]
        outside = _box_mask(points_mm, box_min, box_max)
        pocket = _box_mask(points_mm, pocket_min, pocket_max)
        if any(pocket_min[a] < box_min[a] - 1e-9 or pocket_max[a] > box_max[a] + 1e-9 for a in range(3)):
            raise AdapterError("box_minus_multi_pocket: pocket must be contained in declared stock box")
        remaining = outside & ~pocket
        for isl in params.get("islands_mm", ()):
            if not isinstance(isl, (list, tuple)) or len(isl) != 2:
                raise AdapterError("box_minus_multi_pocket: each island must be [min_mm, max_mm]")
            ilo, ihi = isl[0], isl[1]
            if any(ilo[a] < pocket_min[a] - 1e-9 or ihi[a] > pocket_max[a] + 1e-9 for a in range(3)):
                raise AdapterError("box_minus_multi_pocket: preserved island must lie within pocket region")
            remaining |= outside & _box_mask(points_mm, ilo, ihi)
        try:
            radius = float(params.get("pocket_unreached_corner_radius_mm", 0.0))
        except (TypeError, ValueError) as error:
            raise AdapterError("pocket_unreached_corner_radius_mm must be finite and nonnegative") from error
        if not math.isfinite(radius) or radius < 0:
            raise AdapterError("pocket_unreached_corner_radius_mm must be finite and nonnegative")
        if radius == 0.0:
            return remaining
        if any(pocket_max[axis] - pocket_min[axis] < 2 * radius for axis in (0, 1)):
            raise AdapterError("unreached-corner radius is too large for this pocket")
        pts = np.asarray(points_mm, dtype=float)
        unreachable = np.zeros(remaining.shape, dtype=bool)
        for x_side in (0, 1):
            x_walled = (
                (pocket_min[0] > box_min[0] + 1e-9)
                if x_side == 0
                else (pocket_max[0] < box_max[0] - 1e-9)
            )
            if not x_walled:
                continue
            for y_side in (0, 1):
                y_walled = (
                    (pocket_min[1] > box_min[1] + 1e-9)
                    if y_side == 0
                    else (pocket_max[1] < box_max[1] - 1e-9)
                )
                if not y_walled:
                    continue
                cx = pocket_min[0] + radius if x_side == 0 else pocket_max[0] - radius
                cy = pocket_min[1] + radius if y_side == 0 else pocket_max[1] - radius
                xlo = pocket_min[0] if x_side == 0 else pocket_max[0] - radius
                xhi = pocket_min[0] + radius if x_side == 0 else pocket_max[0]
                ylo = pocket_min[1] if y_side == 0 else pocket_max[1] - radius
                yhi = pocket_min[1] + radius if y_side == 0 else pocket_max[1]
                in_corner = (
                    outside
                    & (pts[..., 0] >= xlo) & (pts[..., 0] <= xhi)
                    & (pts[..., 1] >= ylo) & (pts[..., 1] <= yhi)
                    & (pts[..., 2] >= pocket_min[2]) & (pts[..., 2] <= pocket_max[2])
                )
                unreachable |= in_corner & (
                    (pts[..., 0] - cx) ** 2 + (pts[..., 1] - cy) ** 2 >= radius ** 2
                )
        return remaining | unreachable
    if primitive != "box_minus_pocket":
        raise AdapterError(f"unsupported analytic primitive {primitive!r}")
    required = {
        "box_min_mm", "box_max_mm", "pocket_min_mm", "pocket_max_mm",
        "island_min_mm", "island_max_mm",
    }
    unknown = set(params) - {"primitive", *required}
    if unknown:
        raise AdapterError(f"box_minus_pocket: unsupported parameters {sorted(unknown)}")
    try:
        outside = _box_mask(points_mm, params["box_min_mm"], params["box_max_mm"])
        pocket = _box_mask(points_mm, params["pocket_min_mm"], params["pocket_max_mm"])
    except KeyError as error:
        raise AdapterError(f"box_minus_pocket: missing required parameter {error.args[0]}") from error
    pocket_bounds = [tuple(params[f"pocket_{edge}_mm"][axis] for edge in ("min", "max"))
                     for axis in range(3)]
    box_bounds = [tuple(params[f"box_{edge}_mm"][axis] for edge in ("min", "max"))
                  for axis in range(3)]
    if any(pocket_bounds[a][0] < box_bounds[a][0] or pocket_bounds[a][1] > box_bounds[a][1] for a in range(3)):
        raise AdapterError("pocket must be contained in the declared analytic stock box")
    remaining = outside & ~pocket
    island_min, island_max = params.get("island_min_mm"), params.get("island_max_mm")
    if (island_min is None) != (island_max is None):
        raise AdapterError("both island_min_mm and island_max_mm must be null or supplied")
    if island_min is not None:
        if any(island_min[a] < pocket_bounds[a][0] or island_max[a] > pocket_bounds[a][1] for a in range(3)):
            raise AdapterError("preserved island must be contained in the pocket region")
        remaining |= _box_mask(points_mm, island_min, island_max)
    if not required.issubset(params):
        raise AdapterError(f"box_minus_pocket: missing parameters {sorted(required - set(params))}")
    return remaining


def _stock_bounds_machine(task: C.TaskSpec) -> Dict[str, Any]:
    params = task.stock.params
    lo, hi = params["min_mm"], params["max_mm"]
    # Transform all corners before taking the axis-aligned machine bounds.
    # Projecting a rotated stock box to an AABB here is a conservative TASK grid
    # envelope and must not be described as the oriented stock geometry itself.
    if abs(task.work_to_machine.matrix[2][0]) > 1e-9 or abs(task.work_to_machine.matrix[2][1]) > 1e-9:
        raise AdapterError("tilted work transforms are outside this prototype's analytic-box CAMotic route")
    transformed = [task.work_to_machine.apply(tuple(float(axis) for axis in corner))
                   for corner in ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))]
    min_machine = tuple(min(p[a] for p in transformed) for a in range(3))
    max_machine = tuple(max(p[a] for p in transformed) for a in range(3))
    if any(a >= b for a, b in zip(min_machine, max_machine)):
        raise AdapterError("transformed stock does not have positive bounds")
    return {"min": list(min_machine), "max": list(max_machine)}


def _translation_mm(task: C.TaskSpec) -> Tuple[float, float, float]:
    """Only pure translations are representable in this CAMotics project.

    A rotated TaskSpec needs an actual transformed solid/work-offset
    implementation, not merely transformed axis-aligned bounds. G10 supports
    no work-frame rotation in this release.
    """
    matrix = task.work_to_machine.matrix
    if any(matrix[axis][column] != (1.0 if axis == column else 0.0)
           for axis in range(3) for column in range(3)):
        raise AdapterError("only a pure work_to_machine translation is supported; G10 cannot encode a rotation")
    return tuple(float(matrix[axis][3]) for axis in range(3))


def _rapid_sweep_conflicts(task: C.TaskSpec, motion: C.MotionPlan) -> Tuple[str, ...]:
    """Reject rapid paths whose modeled cutter AABB overlaps initial stock.

    CAMotics v1.2.0 sweeps rapid moves as well as feeds. That is incompatible
    with the independent zero-removal rapid semantics. This is only a coarse
    conservative compatibility guard, never a clearance or assembly safety
    certificate. Stationary typed initial-state markers do not move the tool
    and therefore are not converted into a rapid-motion rejection.
    """
    stock = _stock_bounds_machine(task)
    lower, upper = stock["min"], stock["max"]
    radius = float(task.tool.cutter_radius_mm)
    length = float(task.tool.flute_length_mm)
    reasons = []
    for segment in motion.segments:
        if segment.kind != C.SegmentKind.RAPID or segment.start_mm == segment.end_mm:
            continue
        start, end = segment.start_mm, segment.end_mm
        swept_min = (min(start[0], end[0]) - radius, min(start[1], end[1]) - radius,
                     min(start[2], end[2]))
        swept_max = (max(start[0], end[0]) + radius, max(start[1], end[1]) + radius,
                     max(start[2], end[2]) + length)
        # Inclusive contact is compatibility-unknown, not a proof of clearance.
        if all(swept_min[a] <= upper[a] and swept_max[a] >= lower[a] for a in range(3)):
            reasons.append(
                f"rapid segment {segment.id!r} (source line {segment.source_line}) "
                "may intersect stock in the modeled cutter AABB; CAMotics would carve "
                "this rapid, contrary to zero-removal rapid semantics, so grid geometry is unsupported"
            )
    return tuple(reasons)


def _simulation_stock_bounds(task: C.TaskSpec) -> Dict[str, Any]:
    """Box bounds in the disclosed machine-relative-to-initial-pose frame."""
    initial = np.asarray(task.initial_state.position_mm, dtype=float)
    bounds = _stock_bounds_machine(task)
    return {"min": (np.asarray(bounds["min"]) - initial).tolist(),
            "max": (np.asarray(bounds["max"]) - initial).tolist()}


def _prepare_camotics_nc(task: C.TaskSpec, nc_bytes: bytes,
                         dialect: C.Dialect | str) -> Tuple[bytes, C.MotionPlan]:
    """Validate source NC and establish CAMotics' own zero-origin setup.

    The entire simulation uses ``machine_mm - task.initial_state.position_mm``.
    CAMotics' default origin is then exactly the declared initial pose, with
    no synthetic cutter movement or origin-to-workpiece rapids. Stock bounds,
    STL query points and the source program's G54 transform all use this one
    frame. Explicit G53 is rejected because raw machine coordinates would need
    a corresponding separate absolute-motion re-parameterization here.

    The shared parser is the semantic boundary. The packaged v1.2.0 runtime
    was measured to evaluate X/Y/Z as ``programmed - G10``; the configured
    G10 words are therefore ``initial_machine - work_to_machine_translation``.
    Positive raw G10 values took the opposite direction in the calibration
    probes. This is release-specific adapter evidence, not portable G-code
    advice. The source NC is retained unchanged after the separate prefix.
    """
    if not isinstance(nc_bytes, (bytes, bytearray)) or not nc_bytes:
        raise AdapterError("NC input must be nonempty bytes")
    try:
        text = bytes(nc_bytes).decode("utf-8")
    except UnicodeDecodeError as error:
        raise AdapterError("NC must be valid UTF-8 text") from error
    translation = _translation_mm(task)
    state = task.initial_state
    if not state.known or state.frame != C.Frame.MACHINE:
        raise AdapterError("CAMotics anchoring requires a known machine-frame TaskSpec.initial_state")
    motion = parse_motion_plan(text, task, dialect=dialect)
    if any(segment.tag == "home" for segment in motion.segments):
        raise UnsupportedGCodeError(
            "CAMotics initial-pose-relative calibration supports G54 motion only; "
            "explicit absolute G53 is not reparameterized yet"
        )
    work_offset = np.asarray(state.position_mm, dtype=float) - np.asarray(translation)
    offset_words = " ".join(f"{axis}{float(value):.17g}"
                             for axis, value in zip("XYZ", work_offset))
    prefix = (
        "(DiffCAM adapter setup only: machine coordinates relative to the declared initial pose.)\n"
        "G21 G90\n"
        f"G10 L2 P1 {offset_words}\n"
        f"T{task.tool.tool_number} M06\n"
    ).encode("ascii")
    sim_body = bytes(nc_bytes)
    if C.Dialect(dialect) == C.Dialect.HAAS:
        # Preserve raw source_program.nc verbatim; neutralize Haas-proprietary G187
        # smoothing words for LinuxCNC-based camsim after strict Haas parsing passed.
        cleaned_lines = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped.upper().startswith("G187"):
                cleaned_lines.append(f"(Haas NGC smoothing validated: {stripped})")
            else:
                cleaned_lines.append(raw_line)
        sim_body = ("\n".join(cleaned_lines) + ("\n" if text.endswith("\n") else "")).encode("utf-8")
    return prefix + sim_body, motion


def unsupported_probe(task: C.TaskSpec, motion: Optional[C.MotionPlan] = None) -> Tuple[str, ...]:
    """Explicit precheck; a failed check must not invoke Docker or create a project."""
    reasons = []
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        reasons.append("only flat cylindrical cutters are supported, not ball/non-flat tools")
    geometry = [("stock", task.stock), ("original_part", task.original_part), ("operation_target", task.operation_target.geometry)]
    for label, source in geometry:
        primitive = source.params.get("primitive")
        if source.kind != "analytic" or (label == "stock" and primitive != "box"):
            reasons.append(f"{label}: stock must be an analytic box in the work frame")
        if source.frame != C.Frame.WORK:
            reasons.append(f"{label}: geometry must be declared in the work frame")
        try:
            _geometry_mask(source, np.zeros((1, 3), dtype=float))
        except AdapterError as error:
            reasons.append(f"{label}: {error}")
    if task.grid.frame != C.Frame.WORK:
        reasons.append("CAMotics grid evaluation requires a work-frame GridSpec")
    if not task.initial_state.known or task.initial_state.frame != C.Frame.MACHINE:
        reasons.append("a known machine-frame TaskSpec.initial_state is required for the simulator reference origin")
    else:
        try:
            _simulation_stock_bounds(task)
        except AdapterError as error:
            reasons.append(str(error))
    try:
        _translation_mm(task)
    except AdapterError as error:
        reasons.append(str(error))
    if task.fixtures:
        reasons.append("separate workholding solids are not modeled")
    if motion is not None:
        if not motion.initial_state.known:
            reasons.append("unknown motion initial tool state")
        if motion.timing_rule != task.motion_semantics.timing_rule:
            reasons.append("motion/task timing-rule mismatch")
    return tuple(reasons)


def build_camotics_project(task: C.TaskSpec, resolution_mm: float, *,
                           program_name: str = "program.nc",
                           tool_number: Optional[int] = None) -> Dict[str, Any]:
    """Build the v1.2 project in machine-relative-to-declared-initial coordinates."""
    error = unsupported_probe(task)
    if error:
        raise AdapterError("; ".join(error))
    resolution = _finite_positive("resolution_mm", resolution_mm)
    effective_tool = task.tool.tool_number if tool_number is None else tool_number
    if not isinstance(effective_tool, int) or isinstance(effective_tool, bool) or effective_tool < 1:
        raise AdapterError("tool_number must be a positive integer")
    if effective_tool != task.tool.tool_number:
        raise AdapterError("CAMotics tool table must use TaskSpec.tool.tool_number, not a second implicit tool")
    bounds = _simulation_stock_bounds(task)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.nc", program_name):
        raise AdapterError("CAMotics program must be a plain local .nc filename (no URLs/paths)")
    return {
        "units": "metric", "resolution-mode": "manual", "resolution": resolution,
        "tools": {str(effective_tool): {
            "units": "metric", "shape": "cylindrical",
            "diameter": float(2.0 * task.tool.cutter_radius_mm),
            "length": float(task.tool.flute_length_mm),
            "description": "flat cutter inferred from ToolAssembly; holder is not modeled",
        }},
        "workpiece": {"automatic": False, "margin": 0, "bounds": bounds},
        "files": [program_name],
    }


def _machine_motion(task: C.TaskSpec, motion: C.MotionPlan) -> C.MotionPlan:
    if motion.frame == C.Frame.MACHINE:
        return motion
    output_state = replace(motion.initial_state, frame=C.Frame.MACHINE)
    if motion.initial_state.known:
        position = task.work_to_machine.apply(motion.initial_state.position_mm)
        output_state = replace(output_state, position_mm=position)
    segments = []
    for segment in motion.segments:
        start = task.work_to_machine.apply(segment.start_mm)
        end = task.work_to_machine.apply(segment.end_mm)
        new_arc = segment.arc
        if new_arc is not None:
            new_arc = replace(new_arc, centre_mm=task.work_to_machine.apply(new_arc.centre_mm))
            determinant = np.linalg.det(np.asarray(task.work_to_machine.matrix, dtype=float)[:3, :3])
            if determinant < 0:
                new_direction = C.ArcDirection.CW if new_arc.direction == C.ArcDirection.CCW else C.ArcDirection.CCW
                new_arc = replace(new_arc, direction=new_direction)
        segments.append(replace(segment, start_mm=start, end_mm=end, arc=new_arc))
    return replace(motion, frame=C.Frame.MACHINE, initial_state=output_state, segments=tuple(segments))


def _unknown_metrics(reason: str) -> Dict[str, C.Metric]:
    units = {"iou": "1", "dice": "1", "removed_volume_mm3": "mm^3",
             "residual_volume_mm3": "mm^3", "gouge_volume_mm3": "mm^3",
             "excluded_volume_mm3": "mm^3", "removal_target_volume_mm3": "mm^3",
             "original_part_volume_mm3": "mm^3", "original_part_gouge_volume_mm3": "mm^3",
             "gouge_fraction": "1", "residual_fraction": "1", "operation_target_removed_fraction": "1",
             "max_surface_error_mm": "mm", "p99_surface_error_mm": "mm", "asd_mm": "mm", "hd95_mm": "mm",
             "remaining_volume_mm3": "mm^3", "stl_padded_volume_mm3": "mm^3",
             "min_clearance_mm": "mm", "min_rapid_cutter_clearance_mm": "mm",
             "unique_contacted_volume_mm3": "mm^3", "collision_sample_count": "count"}
    return {name: C.Metric(None, unit, null_reason=reason) for name, unit in units.items()}


def _empty_evaluation(task, reasons, *, coverage=C.Coverage.UNSUPPORTED, nc=None, motion=None, settings=None):
    reason = "; ".join(reasons)
    return C.EvaluationResult(
        evaluator_id=EVALUATOR_ID,
        evaluator_revision=f"{DEFAULT_RELEASE}+prototype-grid-sampler/1",
        task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
        discretization={"status": "unsupported_or_blocked", "reason": reason, "settings": dict(settings or {})},
        coverage=coverage, coverage_reasons=tuple(reasons), metrics=_unknown_metrics(reason),
        safety_pass=None, geometry_pass=None,
        nc_hash=None if nc is None else C.nc_hash(nc[0]), dialect=None if nc is None else nc[1],
        motion_hash=None if motion is None else motion.motion_hash,
    )


class CamoticsAdapter:
    """Isolated subprocess CAMotics evaluation; completed geometry is never a safety certificate."""

    def __init__(self, image: str = DEFAULT_IMAGE, default_resolution_mm: float = 0.25,
                 scratch_root: Optional[Path] = None, timeout_s: float = 300.0):
        self.image = image
        self.default_resolution_mm = _finite_positive("default_resolution_mm", default_resolution_mm)
        self.scratch_root = None if scratch_root is None else Path(scratch_root)
        self.timeout_s = _finite_positive("timeout_s", timeout_s)

    def _docker_call(self, arguments, timeout: Optional[float] = None):
        return subprocess.run(arguments, check=False, capture_output=True, text=True,
                              timeout=self.timeout_s if timeout is None else timeout)

    def _image_facts(self) -> Mapping[str, Any]:
        info = self._docker_call(["docker", "image", "inspect", self.image, "--format", "{{.Id}}"], timeout=20)
        if info.returncode:
            raise AdapterError(f"Docker/image unavailable: {info.stderr.strip() or info.stdout.strip()}")
        digest = info.stdout.strip()
        if digest != DEFAULT_IMAGE_DIGEST:
            raise AdapterError(
                f"CAMotics image {digest!r} is not the calibrated {DEFAULT_IMAGE_DIGEST}; "
                "requalify coordinate semantics before running this adapter on another image"
            )
        binary = self._docker_call(["docker", "run", "--rm", "--network", "none", self.image,
                                    "camsim", "--version"], timeout=30)
        if binary.returncode:
            raise AdapterError(f"CAMotics binary version probe failed: {binary.stderr.strip() or binary.stdout.strip()}")
        # The qualified upstream binary writes '1.2' to stderr, not stdout.
        # An empty stdout must not turn into a missing evaluator revision.
        version = binary.stdout.strip() or binary.stderr.strip()
        if not version:
            raise AdapterError("CAMotics binary version probe returned no version on stdout or stderr")
        if version != "1.2":
            raise AdapterError(
                f"CAMotics runtime version {version!r} is not the calibrated v1.2.0 binary reporting '1.2'; "
                "the release-specific G10 convention cannot be assumed"
            )
        return {"image": self.image, "image_digest": digest,
                "camsim_version": version, "binary_release": DEFAULT_RELEASE,
                "version_probe_stdout": binary.stdout, "version_probe_stderr": binary.stderr}

    def _run_camsim(self, task, nc_bytes, project, workdir: Path, timeout_s: float,
                    source_nc_bytes: bytes) -> Mapping[str, Any]:
        # program.nc has a controlled adapter prologue. Preserve the exact
        # caller input beside it so neither program nor hash is conflated.
        (workdir / "source_program.nc").write_bytes(source_nc_bytes)
        program = workdir / "program.nc"
        program.write_bytes(nc_bytes)
        spec = workdir / "project.camotics"
        spec.write_text(json.dumps(project, indent=2, sort_keys=True) + "\n")
        facts = self._image_facts()
        name = f"diffcam-t13-{uuid4().hex}"
        command = [
            "docker", "run", "--rm", "--name", name, "--network", "none", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=64m",
            "--env", "HOME=/tmp", "--user", f"{os.getuid()}:{os.getgid()}",
            "--volume", f"{workdir}:/work", self.image, "camsim", "--binary",
            "--render-mode", "0", "--threads", "1", "project.camotics", "cut_workpiece.stl",
        ]
        started = time.monotonic()
        running = None
        try:
            running = self._docker_call(command, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Only stop a unique Docker container created by this invocation.
            subprocess.run(["docker", "stop", "--time", "5", name], check=False,
                           capture_output=True, text=True, timeout=15)
            raise AdapterError(f"CAMotics exceeded {timeout_s}s timeout; attempted to stop only container {name}")
        facts = dict(facts)
        facts.update({
            "command": command, "exit_code": running.returncode,
            "stdout": running.stdout, "stderr": running.stderr, "wall_time_s": time.monotonic() - started,
            "workdir": str(workdir), "simulation_nc_sha256": C.sha256_bytes(nc_bytes),
            "source_nc_sha256": C.sha256_bytes(source_nc_bytes),
            "project_sha256": C.sha256_bytes(spec.read_bytes()),
            "renderer_threads": 1, "render_mode": "MCUBES_MODE", "render_mode_id": 0,
            "render_policy": "single-thread MCUBES: eliminates the observed parallel-emission incidence defects; no general manifold or geometry-accuracy guarantee",
        })
        if running.returncode:
            raise AdapterError(f"camsim exited {running.returncode}: {running.stderr.strip() or running.stdout.strip()}")
        mesh_path = workdir / "cut_workpiece.stl"
        if not mesh_path.is_file():
            raise AdapterError("camsim returned success but produced no STL output")
        return facts

    def _evaluate(self, task: C.TaskSpec, nc_bytes: bytes, dialect: C.Dialect, settings: Mapping[str, Any]):
        if not isinstance(nc_bytes, (bytes, bytearray)) or not nc_bytes:
            raise ValueError("nc_bytes must be nonempty bytes for NC evaluation")
        nc_bytes = bytes(nc_bytes)
        settings = dict(settings)
        reasons = unsupported_probe(task)
        if reasons:
            return _empty_evaluation(task, reasons, nc=(nc_bytes, dialect), settings=settings)
        resolution = _finite_positive("resolution_mm", settings.get("resolution_mm", self.default_resolution_mm))
        timeout = _finite_positive("timeout_s", settings.get("timeout_s", self.timeout_s))
        root = Path(settings.get("scratch_root", self.scratch_root or (Path(__file__).resolve().parents[1] / "external" / "camotics-evaluations")))
        root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="t13a-", dir=root))
        try:
            exclusion_tolerance = _finite_nonnegative(
                "exclusion_tolerance_mm3", settings.get("exclusion_tolerance_mm3", 0.0))
            try:
                simulation_nc, parsed_motion = _prepare_camotics_nc(task, nc_bytes, dialect)
            except (GCodeError, AdapterError, C.ContractError) as error:
                return _empty_evaluation(task, [str(error)], coverage=C.Coverage.UNSUPPORTED,
                                         nc=(nc_bytes, dialect), settings=settings)
            rapid_reasons = _rapid_sweep_conflicts(task, parsed_motion)
            if rapid_reasons:
                return _empty_evaluation(task, rapid_reasons, coverage=C.Coverage.UNSUPPORTED,
                                         nc=(nc_bytes, dialect), motion=parsed_motion, settings=settings)
            project = build_camotics_project(task, resolution)
            facts = self._run_camsim(task, simulation_nc, project, workdir, timeout,
                                     source_nc_bytes=nc_bytes)
            facts = {**facts, "source_motion_hash": parsed_motion.motion_hash,
                     "simulation_frame": "machine_mm - task.initial_state.position_mm",
                     "source_nc_unchanged_suffix": True,
                     "adapter_prefix_lines": 4,
                     "external_warnings": [line.strip() for line in
                                           (facts["stdout"] + "\n" + facts["stderr"]).splitlines()
                                           if "WARNING" in line or "ERROR" in line]}
            mesh = read_stl(workdir / "cut_workpiece.stl", max_triangles=int(settings.get("max_triangles", 2_000_000)))
            stl_facts = mesh_volume_mm3(mesh, require_watertight=False)
            grid = task.grid
            axes = [grid.origin_mm[a] + (np.arange(grid.shape[a]) + 0.5) * grid.voxel_mm for a in range(3)]
            coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            flat_work = coordinates.reshape(-1, 3)
            flat_machine = np.asarray([task.work_to_machine.apply(tuple(point)) for point in flat_work])
            flat_simulation = flat_machine - np.asarray(task.initial_state.position_mm, dtype=float)
            sample = points_in_mesh(mesh, flat_simulation, touch_tolerance_mm=settings.get(
                "touch_tolerance_mm", max(1e-9, min(grid.voxel_mm, resolution) * 1e-6)))
            remaining = sample.occupancy.reshape(grid.shape)
            initial = _box_mask(coordinates, task.stock.params["min_mm"], task.stock.params["max_mm"])
            part = _geometry_mask(task.operation_target.geometry, coordinates)
            original_part = _geometry_mask(task.original_part, coordinates)
            if np.any(remaining & ~initial):
                raise AdapterError("sampled grid unexpectedly contains stock outside declared initial stock")
            excluded_mm3 = float(task.operation_target.excluded_volume_mm3)
            metrics = grid_material_metrics(
                remaining, part, initial, grid.voxel_mm,
                original_part_mask=original_part, surface_origin_mm=grid.origin_mm,
                declared_excluded_volume_mm3=excluded_mm3,
                exclusion_tolerance_mm3=exclusion_tolerance)
            if excluded_mm3 and np.array_equal(original_part, part):
                raise AdapterError("excluded_volume_mm3 is declared but original_part equals operation_target; supply filtered geometry without remasking")
            metrics["remaining_volume_mm3"] = C.Metric(int(remaining.sum()) * grid.voxel_mm**3, "mm^3")
            metrics["stl_padded_volume_mm3"] = C.Metric(
                stl_facts["volume_mm3"], "mm^3",
                denominator="whole STL diagnostic surface, including CAMotics boundary padding; not exact initial-stock volume")
            if excluded_mm3:
                metrics["declared_excluded_volume_mm3"] = C.Metric(excluded_mm3, "mm^3")
            reasons = [
                "non-cutting shank/holder and workholding clearance are not modeled; safety_pass is null",
                "controller acceleration/blending timings and machine dynamics are not certified by STL comparison",
                "grid and STL metrics include discretization/slab padding effects; see camotics bias study",
                "simulation is expressed relative to TaskSpec.initial_state; source and generated-setup NC hashes are distinct",
                "G43 is supported only under this adapter's zero tool-register-vector convention; physical H-table calibration is not checked by camsim",
                "the prototype supports pure work-frame translations and rejects G53 rather than misrepresenting absolute-machine moves",
                "rapid cutter/stock AABB overlap is unsupported because CAMotics can carve rapids; this is not an assembly-clearance proof",
                "grid exclusions are checked against the declaration with explicit exclusion_tolerance_mm3 (default 0, roundoff-only)",
            ]
            if facts["external_warnings"]:
                reasons.append("external interpreter diagnostics retained: " + " | ".join(facts["external_warnings"]))
            unexpected_warnings = [warning for warning in facts["external_warnings"]
                                   if "Tool Length Offset not supported" not in warning]
            if not stl_facts["watertight"]:
                reasons.append(
                    f"CAMotics exported topology with {stl_facts['boundary_or_nonmanifold_edges']} incident/multiply-used/inconsistently-directed STL edges; mesh comparisons are approximate and geometry_pass is null"
                )
            metadata = {
                **facts, "resolution_mm": resolution, "stl_format": mesh.stl_format,
                "triangle_count": mesh.triangle_count, "stl_volume_mm3": stl_facts["volume_mm3"],
                "grid_shape": list(grid.shape), "grid_voxel_mm": grid.voxel_mm,
                "grid_origin_mm": list(grid.origin_mm), "grid_frame": "work",
                "simulation_reference_offset_mm": list(task.initial_state.position_mm),
                "simulation_frame": "machine_mm - task.initial_state.position_mm",
                "source_program_filename": "source_program.nc",
                "simulation_program_filename": "program.nc",
                "stl_frame": "same initial-pose-relative frame as cut_workpiece.stl and project.camotics",
                "source_motion_hash": parsed_motion.motion_hash,
                "external_warning_geometry_policy": "geometry_pass null for any diagnostic other than the documented G43 warning",
                "sample_ray": "+Z projected XY barycentric intersection with duplicate-root merging",
                "on_surface_policy": sample.boundary_policy, "touch_tolerance_mm": sample.touch_tolerance_mm,
                "surface_contact_count": sample.surface_contact_count,
                "padded_volume_caveat": "STL output is on stock expanded by one resolution cell; do not use its whole volume as true stock",
                "safety_pass_role": "null because holder/fixture/dynamics are not modeled by camsim mesh comparison",
                "whole_cell_safety_bounds": "not computed; centre-only mesh occupancy/bounding-box masks are not interval safety clearances",
                "stl_topology": {
                    "watertight": stl_facts["watertight"],
                    "unmatched_or_multiply_used_edges": stl_facts["boundary_or_nonmanifold_edges"],
                    "inconsistent_direction_edges": stl_facts["inconsistent_direction_edges"],
                    "self_intersection_check": "not performed; closure is not a global geometric-validity certificate",
                    "geometry_pass_rule": "null unless watertight two-face and coherently oriented topology and all tolerances pass",
                },
                "complete_stock_cell_count": int(initial.sum()),
                "complete_stock_grid_volume_mm3": float(initial.sum()) * grid.voxel_mm**3,
                "declared_excluded_volume_mm3": excluded_mm3,
                "exclusion_tolerance_mm3": exclusion_tolerance,
                "exclusion_rule": "computed initial & operation_target & ~original_part compared with declaration; default tolerance 0 permits only floating roundoff",
            }
            # Review C1: CAMotics 1.2.0 horizontal ConicSweep never cuts the starting disc when the segment
            # start and end Z are equal (beta<0 rejection + (Bz-Az)=0 fallback), so any constant-Z cutting move
            # yields known-wrong removed geometry. geometry_pass must stay null for such programs.
            constant_z_cuts = [s.id for s in parsed_motion.segments
                               if s.kind != C.SegmentKind.RAPID and s.path_length_mm > 0  # path, not chord: full circles (review C2)
                               and abs(s.end_mm[2] - s.start_mm[2]) <= 1e-9]
            if constant_z_cuts:
                reasons = list(reasons) + [
                    "geometry_pass null: CAMotics 1.2.0 horizontal ConicSweep endpoint defect (start disc not cut for "
                    f"constant-Z cutting moves; {len(constant_z_cuts)} affected segments, first {constant_z_cuts[0]})"]
            return C.EvaluationResult(
                evaluator_id=EVALUATOR_ID, evaluator_revision=f"{facts['camsim_version']}+grid-sampler/1",
                task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
                discretization=metadata, coverage=C.Coverage.INCOMPLETE,
                coverage_reasons=tuple(reasons), metrics=metrics,
                safety_pass=None, geometry_pass=(C.geometry_pass_from_metrics(metrics, task.tolerances)
                                             if stl_facts["watertight"] and not unexpected_warnings
                                             and not constant_z_cuts else None),
                nc_hash=C.nc_hash(nc_bytes), dialect=dialect,
            )
        except (AdapterError, OSError, ValueError, subprocess.TimeoutExpired) as error:
            return _empty_evaluation(task, [f"evaluation blocked: {error}"], coverage=C.Coverage.UNKNOWN,
                                     nc=(nc_bytes, dialect), settings={**settings, "workdir": str(workdir)})

    def evaluate_candidate(self, task: C.TaskSpec, motion: C.MotionPlan, settings: Mapping[str, Any]):
        reasons = unsupported_probe(task, motion)
        if reasons:
            return _empty_evaluation(task, reasons, motion=motion, settings=settings)
        settings = dict(settings)
        dialect = C.Dialect(settings.get("dialect", C.Dialect.HAAS))
        config = settings.get(
            "machine_config",
            MachineConfig(tool_number=task.tool.tool_number),
        )
        try:
            machine_motion = _machine_motion(task, motion)
            nc_bytes = post_motion_plan(machine_motion, task, dialect=dialect, config=config).encode("utf-8")
        except (ValueError, TypeError) as error:
            return _empty_evaluation(task, [str(error)], motion=motion, settings=settings)
        return self._evaluate(task, nc_bytes, dialect, settings)

    def evaluate_posted_gcode(self, task: C.TaskSpec, nc_bytes: bytes, dialect: C.Dialect,
                              settings: Mapping[str, Any]):
        return self._evaluate(task, nc_bytes, C.Dialect(dialect), settings)


__all__ = [
    "AdapterError", "CamoticsAdapter", "TriangleMesh", "MeshSample", "read_stl", "mesh_volume_mm3",
    "points_in_mesh", "build_camotics_project", "unsupported_probe",
    "_prepare_camotics_nc", "DEFAULT_IMAGE", "DEFAULT_RELEASE", "DEFAULT_IMAGE_DIGEST",
]
