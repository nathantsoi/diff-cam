"""CAMotics 1.2.x binary-in-Docker adapter and independent STL grid sampling.

Only analytic box stock/target geometry and flat cylindrical cutters are
supported. The adapter uses the real ``camsim`` binary; it never treats CAMotics
as a holder/assembly safety oracle. ``safety_pass`` is always null and coverage
is incomplete because non-cutting assembly, fixtures and exact controller timing
are outside CAMotics's geometry output. ``geometry_pass`` is computed from the
shared grid metrics only, when returned, and is not a machining-safety verdict.

``evaluate_candidate`` posts the current typed motion to NC before simulation,
including arc-aware rigid frame conversion. ``evaluate_posted_gcode`` consumes
actual NC bytes. Arbitrary STEP/sampled/non-analytic stock, ball tools,
workholding, unsupported tool changes and malformed meshes fail closed.
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
from simulator.csg_metrics import grid_material_metrics


EVALUATOR_ID = "camotics"
DEFAULT_IMAGE = "diffcam-camotics:1.2.0"
DEFAULT_RELEASE = "1.2.0"


class AdapterError(RuntimeError):
    """Actionable unsupported input or failed mesh/project interpretation."""


@dataclass(frozen=True)
class TriangleMesh:
    vertices_mm: np.ndarray             # (N, 3, 3), actual triangle vertices
    stl_format: str

    @property
    def triangle_count(self) -> int:
        return len(self.vertices_mm)


@dataclass(frozen=True)
class MeshSample:
    occupancy: np.ndarray
    surface_contact_count: int
    projected_ray_queries: int
    touch_tolerance_mm: float
    boundary_policy: str


def _finite_positive(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AdapterError(f"{name}: expected a finite positive value") from error
    if not math.isfinite(result) or result <= 0:
        raise AdapterError(f"{name}: expected a finite positive value")
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
        for line in data.decode("ascii").splitlines():
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
    """
    vertices = mesh.vertices_mm
    if len(vertices) == 0:
        return {"volume_mm3": 0.0, "watertight": True, "boundary_or_nonmanifold_edges": 0}
    if require_watertight:
        flat = vertices.reshape(-1, 3)
        _, inverse = np.unique(flat, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
        edges.sort(axis=1)
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        defects = int(np.count_nonzero(edge_counts != 2))
        if defects:
            raise AdapterError(f"mesh is not a closed two-face manifold: {defects} unmatched/multiply-used edges")
    origin = vertices.reshape(-1, 3).mean(axis=0)
    signed = sum(np.dot(t[0] - origin, np.cross(t[1] - origin, t[2] - origin)) for t in vertices)
    signed /= 6.0
    if not math.isfinite(signed):
        raise AdapterError("mesh signed volume is not finite")
    if signed < 0:
        raise AdapterError("mesh triangle winding produces a negative signed volume; not repaired with an absolute value")
    return {"volume_mm3": float(signed), "watertight": True, "boundary_or_nonmanifold_edges": 0}


def points_in_mesh(mesh: TriangleMesh, points_mm: np.ndarray, *,
                   touch_tolerance_mm: float = 1e-7, query_chunk: int = 256) -> MeshSample:
    """Closed-surface-inclusive +Z ray parity with duplicate-root suppression.

    Vertical projected triangles are skipped. Shared/coplanar roots are merged
    within the declared tolerance; a point within tolerance of a surface is
    deliberately counted as MATERIAL under the explicit ``closed_include``
    policy. This is a geometry-sampling choice, not a collision/safety proof.
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
    lower = mesh.vertices_mm.min(axis=(0, 1)) - touch_tolerance_mm
    upper = mesh.vertices_mm.max(axis=(0, 1)) + touch_tolerance_mm
    results = np.zeros(len(points), dtype=bool)
    contacts = 0
    projected_queries = 0
    for start in range(0, len(points), query_chunk):
        for point in points[start:start + query_chunk]:
            if np.any(point < lower) or np.any(point > upper):
                continue
            a, b, c = mesh.vertices_mm[:, 0], mesh.vertices_mm[:, 1], mesh.vertices_mm[:, 2]
            u, v, r = b - a, c - a, point - a
            denominator = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
            projected = np.abs(denominator) > max(touch_tolerance_mm * 1e-3, 1e-14)
            if not projected.any():
                continue
            projected_queries += 1
            u, v, r, denom = u[projected], v[projected], r[projected], denominator[projected]
            aa, bb, cc = a[projected, 2], b[projected, 2], c[projected, 2]
            wb = (r[:, 0] * v[:, 1] - v[:, 0] * r[:, 1]) / denom
            wc = (u[:, 0] * r[:, 1] - r[:, 0] * u[:, 1]) / denom
            valid = (wb >= -touch_tolerance_mm) & (wc >= -touch_tolerance_mm) & (wb + wc <= 1 + touch_tolerance_mm)
            if not valid.any():
                continue
            roots = np.sort(aa[valid] + wb[valid] * (bb[valid] - aa[valid]) + wc[valid] * (cc[valid] - aa[valid]))
            if roots.size and np.min(np.abs(roots - point[2])) <= touch_tolerance_mm:
                results[start + np.where(points[start:start + query_chunk] == point)[0][0]] = True
                contacts += 1
                continue
            if roots.size:
                # Combine adjacent nearly coincident roots, including shared edges.
                keep = np.concatenate(([True], np.diff(roots) > touch_tolerance_mm * 2))
                roots = roots[keep]
            roots = roots[roots > point[2] + touch_tolerance_mm]
            if len(roots) % 2:
                results[start + np.where(points[start:start + query_chunk] == point)[0][0]] = True
    return MeshSample(results, contacts, projected_queries, touch_tolerance_mm, "closed_include")


def _box_mask(points_mm: np.ndarray, minimum, maximum) -> np.ndarray:
    lower, upper = np.asarray(minimum, dtype=float), np.asarray(maximum, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise AdapterError("analytic box stock/target requires finite min_mm < max_mm triples in work frame")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise AdapterError("non-finite stock bounds")
    points = np.asarray(points_mm, dtype=float)
    return ((points >= lower) & (points <= upper)).all(axis=-1)


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


def unsupported_probe(task: C.TaskSpec, motion: Optional[C.MotionPlan] = None) -> Tuple[str, ...]:
    """Explicit precheck; a failed check must not invoke Docker or create a project."""
    reasons = []
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        reasons.append("only flat cylindrical cutters are supported, not ball/non-flat tools")
    geometry = [("stock", task.stock), ("original_part", task.original_part), ("operation_target", task.operation_target.geometry)]
    for label, source in geometry:
        if source.kind != "analytic" or source.params.get("primitive") != "box":
            reasons.append(f"{label}: only analytic box geometry is supported by this prototype")
        if source.frame != C.Frame.WORK:
            reasons.append(f"{label}: geometry must be declared in the work frame")
    if task.fixtures:
        reasons.append("separate workholding solids are not modeled")
    if motion is not None:
        if not task.initial_state.known or not motion.initial_state.known:
            reasons.append("unknown initial tool state")
        if motion.timing_rule != task.motion_semantics.timing_rule:
            reasons.append("motion/task timing-rule mismatch")
    return tuple(reasons)


def build_camotics_project(task: C.TaskSpec, resolution_mm: float, *,
                           program_name: str = "program.nc", tool_number: int = 1) -> Dict[str, Any]:
    """Build the CAMotics JSON schema used by the binary (actual source schema v1.2)."""
    error = unsupported_probe(task)
    if error:
        raise AdapterError("; ".join(error))
    resolution = _finite_positive("resolution_mm", resolution_mm)
    if not isinstance(tool_number, int) or isinstance(tool_number, bool) or tool_number < 0:
        raise AdapterError("tool_number must be a nonnegative integer")
    bounds = _stock_bounds_machine(task)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.nc", program_name):
        raise AdapterError("CAMotics program must be a plain local .nc filename (no URLs/paths)")
    return {
        "units": "metric", "resolution-mode": "manual", "resolution": resolution,
        "tools": {str(tool_number): {
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
             "max_surface_error_mm": "mm", "p99_surface_error_mm": "mm", "asd_mm": "mm", "hd95_mm": "mm",
             "remaining_volume_mm3": "mm^3", "stl_padded_volume_mm3": "mm^3"}
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
        binary = self._docker_call(["docker", "run", "--rm", "--network", "none", self.image,
                                    "camsim", "--version"], timeout=30)
        if binary.returncode:
            raise AdapterError(f"CAMotics binary version probe failed: {binary.stderr.strip() or binary.stdout.strip()}")
        return {"image": self.image, "image_digest": info.stdout.strip(),
                "camsim_version": binary.stdout.strip(), "binary_release": DEFAULT_RELEASE}

    def _run_camsim(self, task, nc_bytes, project, workdir: Path, timeout_s: float) -> Mapping[str, Any]:
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
            "--volume", f"{workdir}:/work", self.image, "camsim", "--binary", "project.camotics", "cut_workpiece.stl",
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
            "workdir": str(workdir), "nc_sha256": C.sha256_bytes(nc_bytes),
            "project_sha256": C.sha256_bytes(spec.read_bytes()),
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
            project = build_camotics_project(task, resolution, tool_number=int(settings.get("tool_number", 1)))
            facts = self._run_camsim(task, nc_bytes, project, workdir, timeout)
            mesh = read_stl(workdir / "cut_workpiece.stl", max_triangles=int(settings.get("max_triangles", 2_000_000)))
            stl_facts = mesh_volume_mm3(mesh)
            grid = task.grid
            axes = [grid.origin_mm[a] + (np.arange(grid.shape[a]) + 0.5) * grid.voxel_mm for a in range(3)]
            coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            flat_work = coordinates.reshape(-1, 3)
            flat_machine = np.asarray([task.work_to_machine.apply(tuple(point)) for point in flat_work])
            sample = points_in_mesh(mesh, flat_machine, touch_tolerance_mm=settings.get(
                "touch_tolerance_mm", max(1e-9, min(grid.voxel_mm, resolution) * 1e-6)))
            remaining = sample.occupancy.reshape(grid.shape)
            initial = _box_mask(coordinates, task.stock.params["min_mm"], task.stock.params["max_mm"])
            part = _box_mask(coordinates, task.operation_target.geometry.params["min_mm"],
                             task.operation_target.geometry.params["max_mm"])
            if np.any(remaining & ~initial):
                raise AdapterError("sampled grid unexpectedly contains stock outside declared initial stock")
            metrics = grid_material_metrics(remaining, part, initial, grid.voxel_mm,
                                            surface_origin_mm=grid.origin_mm)
            metrics["remaining_volume_mm3"] = C.Metric(int(remaining.sum()) * grid.voxel_mm**3, "mm^3")
            metrics["stl_padded_volume_mm3"] = C.Metric(
                stl_facts["volume_mm3"], "mm^3",
                denominator="whole STL diagnostic surface, including CAMotics boundary padding; not exact initial-stock volume")
            reasons = [
                "non-cutting shank/holder and workholding clearance are not modeled; safety_pass is null",
                "controller acceleration/blending timings and machine dynamics are not certified by STL comparison",
                "grid and STL metrics include discretization/slab padding effects; see camotics bias study",
            ]
            metadata = {
                **facts, "resolution_mm": resolution, "stl_format": mesh.stl_format,
                "triangle_count": mesh.triangle_count, "stl_volume_mm3": stl_facts["volume_mm3"],
                "grid_shape": list(grid.shape), "grid_voxel_mm": grid.voxel_mm,
                "grid_origin_mm": list(grid.origin_mm), "grid_frame": "work",
                "sample_ray": "+Z projected XY barycentric intersection with duplicate-root merging",
                "on_surface_policy": sample.boundary_policy, "touch_tolerance_mm": sample.touch_tolerance_mm,
                "surface_contact_count": sample.surface_contact_count,
                "padded_volume_caveat": "STL output is on stock expanded by one resolution cell; do not use its whole volume as true stock",
                "safety_pass_role": "null because holder/fixture/dynamics are not modeled by camsim mesh comparison",
                "complete_stock_grid_volume_mm3": float(initial.sum()) * grid.voxel_mm**3,
            }
            return C.EvaluationResult(
                evaluator_id=EVALUATOR_ID, evaluator_revision=f"{facts['camsim_version']}+grid-sampler/1",
                task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
                discretization=metadata, coverage=C.Coverage.INCOMPLETE,
                coverage_reasons=tuple(reasons), metrics=metrics,
                safety_pass=None, geometry_pass=C.geometry_pass_from_metrics(metrics, task.tolerances),
                nc_hash=C.nc_hash(nc_bytes), dialect=dialect,
            )
        except (AdapterError, OSError) as error:
            return _empty_evaluation(task, [str(error)], coverage=C.Coverage.UNKNOWN,
                                     nc=(nc_bytes, dialect), settings={**settings, "workdir": str(workdir)})

    def evaluate_candidate(self, task: C.TaskSpec, motion: C.MotionPlan, settings: Mapping[str, Any]):
        reasons = unsupported_probe(task, motion)
        if reasons:
            return _empty_evaluation(task, reasons, motion=motion, settings=settings)
        settings = dict(settings)
        dialect = C.Dialect(settings.get("dialect", C.Dialect.HAAS))
        config = settings.get("machine_config", MachineConfig(tool_number=1))
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
    "points_in_mesh", "build_camotics_project", "unsupported_probe", "DEFAULT_IMAGE", "DEFAULT_RELEASE",
]
