"""Bounded CNC-Net geometry adapter, with explicit object/mm coordinates.

The modernized native condition is not a faithful legacy reproduction, a
trained-baseline benchmark, or an executable-NC adapter. Default tests exercise
only deterministic meshing, transforms, and audit records; no torch/Open3D is
required by this module. Native fields are captured by its isolated subprocess,
not loaded from third-party pickle checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks import contracts as C
from cam.posted_evaluator import _analytic_mask, _centres
from simulator.csg_metrics import compute_grid_metrics
from verification import fixtures as F


@dataclass(frozen=True)
class ObjectUnitFrame:
    """Native object coordinates centred on a declared work-frame point in mm."""

    centre_work_mm: tuple[float, float, float]
    mm_per_object_unit: float

    def __post_init__(self) -> None:
        centre = tuple(float(x) for x in self.centre_work_mm)
        if len(centre) != 3 or not all(math.isfinite(x) for x in centre):
            raise ValueError("centre_work_mm must contain three finite mm values")
        object.__setattr__(self, "centre_work_mm", centre)
        value = float(self.mm_per_object_unit)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("mm_per_object_unit must be finite and positive")
        object.__setattr__(self, "mm_per_object_unit", value)

    def _aligned_centre(self, points):
        if points.ndim == 1:
            if points.shape != (3,):
                raise ValueError("points must be an xyz vector, N-by-3 points, or a (3,nx,ny,nz) grid")
            return np.asarray(self.centre_work_mm)
        if points.ndim == 2 and points.shape[1] == 3:
            return np.asarray(self.centre_work_mm)
        if points.ndim == 4 and points.shape[0] == 3:
            return np.asarray(self.centre_work_mm).reshape(3, 1, 1, 1)
        raise ValueError("points must be an xyz vector, N-by-3 points, or a (3,nx,ny,nz) grid")

    def to_native(self, points_mm):
        points = np.asarray(points_mm, dtype=np.float64)
        return (points - self._aligned_centre(points)) / self.mm_per_object_unit

    def to_work_mm(self, points):
        points = np.asarray(points, dtype=np.float64)
        return points * self.mm_per_object_unit + self._aligned_centre(points)

    def to_dict(self):
        return {
            "convention": "native=(work_mm-centre_mm)/mm_per_object_unit; export is centred, not min-corner-preserving",
            "centre_work_mm": list(self.centre_work_mm),
            "mm_per_object_unit": self.mm_per_object_unit,
        }


@dataclass
class _MeshBuilder:
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    _indices: dict[tuple[float, float, float], int]

    @classmethod
    def empty(cls):
        return cls([], [], {})

    def vertex(self, point):
        key = tuple(float(x) for x in point)
        if key not in self._indices:
            self._indices[key] = len(self.vertices)
            self.vertices.append(key)
        return self._indices[key]

    def quad(self, corners, normal):
        """Add two flat triangles, welding shared vertices and orienting outward."""
        points = [np.asarray(p, dtype=np.float64) for p in corners]
        a = np.cross(points[1] - points[0], points[2] - points[0])
        if not np.isfinite(a).all() or np.linalg.norm(a) <= 0:
            raise ValueError("degenerate pocket-mesh quad")
        if np.dot(a, np.asarray(normal)) < 0:
            points = [points[0], points[3], points[2], points[1]]
        for triangle in ((0, 1, 2), (0, 2, 3)):
            self.faces.append(tuple(self.vertex(points[index]) for index in triangle))


def _rect(builder, lo, hi, z, normal):
    builder.quad(((lo[0], lo[1], z), (hi[0], lo[1], z),
                  (hi[0], hi[1], z), (lo[0], hi[1], z)), normal)


def _walls(builder, lo, hi, bottom, top, *, inward):
    for axis in (0, 1):
        other = 1 - axis
        for side in (0, 1):
            fixed = lo[axis] if side == 0 else hi[axis]
            normal = [0.0, 0.0, 0.0]
            sign = 1.0 if side == 0 else -1.0
            normal[axis] = sign if inward else -sign
            a = [0.0, 0.0, 0.0]
            b = [0.0, 0.0, 0.0]
            a[axis] = b[axis] = fixed
            a[other], b[other] = lo[other], hi[other]
            bottom_a, bottom_b = a.copy(), b.copy()
            top_a, top_b = a.copy(), b.copy()
            bottom_a[2] = bottom_b[2] = bottom
            top_a[2] = top_b[2] = top
            builder.quad((bottom_a, bottom_b, top_b, top_a), normal)


def _rim(builder, outer_lo, outer_hi, inner_lo, inner_hi, z, normal):
    """A 3x3 ring minus its centre, avoiding zero-area strips/T-junctions."""
    for xi in ((outer_lo[0], inner_lo[0]), (inner_lo[0], inner_hi[0]), (inner_hi[0], outer_hi[0])):
        for yi in ((outer_lo[1], inner_lo[1]), (inner_lo[1], inner_hi[1]), (inner_hi[1], outer_hi[1])):
            bounds = (xi, yi)
            if bounds == ((inner_lo[0], inner_hi[0]), (inner_lo[1], inner_hi[1])):
                continue
            builder.quad(((xi[0], yi[0], z), (xi[1], yi[0], z),
                          (xi[1], yi[1], z), (xi[0], yi[1], z)), normal)


def make_dev_box_pocket_task(voxel_mm: float = 0.25):
    """The one frozen T5.1 simplified dev box-pocket with retained island."""
    lo, hi = (-10.0, -8.0, -8.0), (10.0, 8.0, 0.0)
    fixture = F.make_fixture(
        "cnc-net-box-island", (), F.flat_tool(2.0, 8.0, holder_radius=5.0), 0.0,
        "simple native-geometry dev instance, not a qualified operation",
        stock_lo=lo, stock_hi=hi, voxel=voxel_mm, initial=(0.0, 0.0, 10.0),
    )
    part = C.GeometrySource(
        kind="analytic",
        params={
            "primitive": "box_minus_pocket",
            "box_min_mm": list(lo), "box_max_mm": list(hi),
            "pocket_min_mm": [-6.0, -4.0, -4.0], "pocket_max_mm": [6.0, 4.0, 0.0],
            "island_min_mm": [-1.0, -1.0, -4.0], "island_max_mm": [1.0, 1.0, 0.0],
        },
        frame=C.Frame.WORK,
    )
    task = replace(
        fixture.task,
        task_id="dev-cncnet-box-pocket-island-001",
        family="cncnet_adapter_pilot",
        source=C.SourceInfo(origin="procedural:t5.1-dev-box-pocket-island@1", license="MIT"),
        original_part=part,
        operation_target=C.OperationTarget(geometry=part),
        tolerances=C.Tolerances(completion_residual_mm3=0.0, gouge_volume_mm3=0.0,
                                local_surface_error_mm=voxel_mm, clearance_mm=0.5),
    )
    expected = (12 * 8 - 2 * 2) * 4
    if not math.isclose(expected, 368.0):
        raise AssertionError("frozen dev pocket volume changed")
    if not math.isclose((hi[0] - lo[0]) / 80.0, 0.25) or not math.isclose(
        (hi[1] - lo[1]) / 80.0, 0.2
    ) or not math.isclose((hi[2] - lo[2]) / 80.0, 0.1):
        raise AssertionError("frozen object-unit extents changed")
    return task


def unit_frame_for_pilot(task, mm_per_object_unit=80.0):
    # The selected stock is centred at (0,0,-4), hence the export's omitted
    # min-corner has an explicit, controlled zero-centred counterpart.
    return ObjectUnitFrame((0.0, 0.0, -4.0), mm_per_object_unit)


def _split_coplanar_t_junctions(vertices, faces):
    """Split this small analytical surface at shared internal edge vertices.

    Only exact in-edge vertices already present in the builder are inserted.
    No surface position, winding or area is intentionally changed. Incomplete
    holes and inverted patches are not repaired or hidden by this operation.
    """
    working = [list(int(i) for i in face) for face in faces]
    if len(vertices) > 1024 or len(working) > 4096:
        raise ValueError("analytical pilot mesh exceeds conforming-polyhedron limits")
    while True:
        changed = False
        for face_index in range(len(working)):
            if changed:
                break
            triangle = working[face_index]
            for edge_start, edge_end in ((0, 1), (1, 2), (2, 0)):
                a, b = vertices[triangle[edge_start]], vertices[triangle[edge_end]]
                direction = b - a
                limit = float(np.dot(direction, direction))
                if limit == 0:
                    continue
                outcome = None
                for point_index, point in enumerate(vertices):
                    if point_index in (triangle[edge_start], triangle[edge_end]):
                        continue
                    fraction = float(np.dot(point - a, direction)) / limit
                    if not 1e-12 < fraction < 1.0 - 1e-12:
                        continue
                    if np.linalg.norm(point - (a + fraction * direction)) > 1e-10:
                        continue
                    outcome = point_index
                    break
                if outcome is None:
                    continue
                opposite = triangle[next(i for i in range(3) if i not in (edge_start, edge_end))]
                first = [triangle[edge_start], outcome, opposite]
                second = [outcome, triangle[edge_end], opposite]
                working[face_index:face_index + 1] = [first, second]
                changed = True
                break
        if not changed:
            return working


def prepare_native_mesh(task, frame: ObjectUnitFrame):
    """Return the closed welded outer/pocket/island triangle mesh in task mm."""
    p = task.stock.params
    stock_lo, stock_hi = p["min_mm"], p["max_mm"]
    target = task.operation_target.geometry.params
    pocket_lo, pocket_hi = target["pocket_min_mm"], target["pocket_max_mm"]
    island_lo, island_hi = target["island_min_mm"], target["island_max_mm"]
    if task.operation_target.geometry.params.get("primitive") != "box_minus_pocket":
        raise ValueError("this pilot mesh builder supports box_minus_pocket with an island")
    mesh = _MeshBuilder.empty()
    _rect(mesh, stock_lo, stock_hi, stock_lo[2], (0, 0, -1))
    _walls(mesh, stock_lo, stock_hi, stock_lo[2], stock_hi[2], inward=False)
    _rim(mesh, stock_lo, stock_hi, pocket_lo, pocket_hi, stock_hi[2], (0, 0, 1))
    _walls(mesh, pocket_lo, pocket_hi, pocket_lo[2], pocket_hi[2], inward=True)
    _rim(mesh, pocket_lo, pocket_hi, island_lo, island_hi, pocket_lo[2], (0, 0, 1))
    _walls(mesh, island_lo, island_hi, pocket_lo[2], island_hi[2], inward=False)
    _rect(mesh, island_lo, island_hi, island_hi[2], (0, 0, 1))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    conforming = _split_coplanar_t_junctions(vertices, mesh.faces)
    faces = np.asarray(conforming, dtype=np.int32)
    audit_mesh(vertices, faces)
    return frame.to_native(vertices), faces


def audit_mesh(vertices, faces):
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("triangle vertices must be finite N-by-3 objects")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangle faces must be N-by-3 indices")
    if len(faces) == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("triangle face index outside vertex range")
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) == 0):
        raise ValueError("degenerate triangle in CAD input")
    edge_counts: dict[tuple[int, int], int] = {}
    oriented_edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(u), int(v))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            oriented_edges[edge] = oriented_edges.get(edge, 0) + (1 if int(u) < int(v) else -1)
    if any(count != 2 for count in edge_counts.values()):
        raise ValueError("CAD OFF boundary is not edge-manifold and watertight")
    if any(balance != 0 for balance in oriented_edges.values()):
        raise ValueError("CAD OFF triangles are not consistently outward-oriented at shared edges")
    signed_volume = np.einsum("ij,ij->i", triangles[:, 0],
                              np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0
    if not math.isfinite(signed_volume) or signed_volume <= 0:
        raise ValueError("CAD mesh must be consistently outward-oriented with positive signed volume")
    return {
        "vertices": len(vertices), "triangles": len(faces),
        "edge_count": len(edge_counts), "edges_used_exactly_twice": True,
        "signed_volume_object3": float(signed_volume),
        "minimum": vertices.min(axis=0).tolist(), "maximum": vertices.max(axis=0).tolist(),
        "finite": True,
    }


def write_off(path, vertices_native, faces):
    vertices, faces = np.asarray(vertices_native, dtype=np.float64), np.asarray(faces, dtype=np.int32)
    audit_mesh(vertices, faces)
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite existing native input: {path}")
    lines = ["OFF", f"{len(vertices)} {len(faces)} 0"]
    lines.extend(" ".join(format(float(value), ".17g") for value in point) for point in vertices)
    lines.extend(" ".join(["3", *(str(int(index)) for index in face)]) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {"path": str(path), "sha256": sha256_bytes(path.read_bytes())}


def sha256_bytes(data: bytes):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def parse_obj_vertices(path):
    """Read only OBJ vertex records; untrusted faces/comments are not executed."""
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        words = line.strip().split()
        if not words or words[0] != "v":
            continue
        if len(words) != 4:
            raise ValueError(f"unexpected OBJ vertex record: {line!r}")
        values.append(tuple(float(x) for x in words[1:]))
    vertices = np.asarray(values, dtype=np.float64)
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("native OBJ has no finite nonempty vertices")
    return vertices


def native_iou_nonnegative_voxels(current, target):
    """Reproduce native IOU on negative target labels, excluding outside IoU."""
    current, target = np.asarray(current), np.asarray(target)
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("native IOU inputs must be finite")
    if current.shape != target.shape:
        raise ValueError("current and target low-grid labels have different shapes")
    inferred = np.logical_and(current < 0, target < 0).sum()
    union = np.logical_or(current < 0, target < 0).sum()
    return float(inferred) / float(union) if union else 0.0


def map_native_occupancy(current_high, dimensions, frame: ObjectUnitFrame, grid):
    """Nearest native 256^3 samples on a physical task grid, in work-mm.

    Native coordinates are centred by ``prepare_native_mesh``. Its sample grid
    is min=-dimension/2 and i/N spacing. Fractional-index half-way positions
    round upward deterministically. The closest represented index is selected;
    at a mesh-extent boundary, that is a **declared nearest-node clamp** to the
    first/last available native sample, not a continuous-surface guarantee.
    Points outside [-dimension/2, +dimension/2] are excluded, rather than
    extending the model outside its documented bounding box. This maps
    binarized retained occupancy (current_high > 0.5), not raw labels.
    """
    field = np.asarray(current_high)
    if field.ndim != 3 or len(set(field.shape)) != 1:
        raise ValueError("native current_high must be an isometric cubic rank-three grid")
    if not np.isfinite(field).all():
        raise ValueError("native current_high contains non-finite values")
    dimension = np.asarray(dimensions, dtype=np.float64).reshape(-1)
    if dimension.shape != (3,) or not np.isfinite(dimension).all() or np.any(dimension <= 0):
        raise ValueError("native dimensions must be three finite positive object extents")
    points = frame.to_native(_centres(grid))
    n = field.shape[0]
    extent = dimension.reshape(3, 1, 1, 1)
    fractions = (points + extent / 2.0) * n / extent
    indices = np.floor(fractions + 0.5).astype(np.int64)
    valid = np.all((points >= -extent / 2.0) & (points <= extent / 2.0), axis=0)
    safe = np.clip(indices, 0, n - 1)
    inside = np.zeros(grid.shape, dtype=bool)
    if valid.any():
        inside[valid] = field[safe[0], safe[1], safe[2]][valid] > 0.5
    return inside


def score_reconstruction(task, frame, current_high, dimensions):
    grid = task.grid
    centres = _centres(grid)
    occupancy = map_native_occupancy(current_high, dimensions, frame, grid)
    initial = _analytic_mask(task.stock, centres)
    target = _analytic_mask(task.operation_target.geometry, centres)
    original = _analytic_mask(task.original_part, centres)
    if not np.any(initial & ~target):
        raise ValueError("reconstruction pilot requires a removal target")
    return compute_grid_metrics(
        occupancy, target, initial, grid.voxel_mm,
        original_part_mask=original, surface_origin_mm=grid.origin_mm,
        declared_excluded_volume_mm3=task.operation_target.excluded_volume_mm3,
        exclusion_tolerance_mm3=0.0,
    )


def executable_disposition():
    """Frozen source-evidence disposition, never an invented NC program."""
    return {
        "schema": "diffcam-cncnet-executable-disposition/1",
        "status": "unsupported",
        "disposition": "incompatible: executable NC mapping unsupported",
        "source_revision": "ba0652f9941686058c07c98617b9748e72f5f019",
        "native_ordered_waypoint_or_nc_export": False,
        "fixed_plus_z_and_one_cutter_declared": False,
        "holder_feeds_retract_g54_semantics_declared": False,
        "nc_bytes": None,
        "nc_hash": None,
        "reasons": (
            "models predict mill/drill selections with learned radii and elided physical units",
            "two learned axis-angle components rotate effective tool axes; no fixed-+Z declaration exists",
            "native entrypoint returns loss/current fields and writes reconstruction meshes; no ordered waypoint or NC export",
            "no faithful tool-holder approach/link/retract/feed contract exists that preserves learned operations",
        ),
        "evidence": (
            "external/cnc-net/networks/model.py:246-432",
            "external/cnc-net/run.py:243-320",
            "external/cnc-net/networks/primitives.py:9-31",
        ),
        "pretraining_or_model_or_loss_substitution": False,
    }


def write_json(path, value):
    from algorithms.run_artifacts import ArtifactWriter, json_bytes
    target = Path(path)
    ArtifactWriter._replace_bytes(target, json_bytes(value))


__all__ = [
    "ObjectUnitFrame", "make_dev_box_pocket_task", "unit_frame_for_pilot",
    "prepare_native_mesh", "audit_mesh", "write_off", "parse_obj_vertices",
    "native_iou_nonnegative_voxels", "map_native_occupancy", "score_reconstruction",
    "sha256_bytes", "executable_disposition", "write_json",
]
