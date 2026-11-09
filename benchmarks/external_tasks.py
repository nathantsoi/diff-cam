"""Bounded, isolated OpenCascade adapters for local external STEP tasks.

There is deliberately no OCP import in the normal project environment.  The
only heavy backend is the separately pinned ``external/cad-venv`` interpreter,
invoked with ``-I`` and a JSON request.  Its ability to execute a subprocess is
not geometric evidence: the worker must validate the source hash, transfer
STEP roots, validate a single native solid, and check an actual dimension
before any sampling starts.  Mesh/STL input is not a substitute.

The Titan pilot additionally performs a native additive Boolean for a declared
small-hole exclusion and a native prismatic sweep of one vertical feed.  Those
are geometry diagnostics, not a complete machining plan or safety certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAD_VERSION = "7.7.2"
CAD_WHEEL_SHA256 = "7ac2d83b4f4b3d7c35421a1d9f8fec2adc73b6ed6cac50c1ffcede5552e38e9b"
TITAN_M1_SHA256 = "3ba16d17de50526b452f844a7faf66a624ddd9509fcc3eac2087fdbdbc2066e3"
SAMPLED_EVALUATOR_REASON = "only analytic operation/part geometry in the work frame is supported"


class CADUnavailable(RuntimeError):
    """The required isolated, qualified CAD runtime is unavailable."""


class CADValidationError(RuntimeError):
    """The actual native CAD task failed an explicit geometric invariant."""


def default_cad_interpreter() -> Path:
    configured = os.environ.get("DIFFCAM_CAD_PYTHON")
    return Path(configured) if configured else PROJECT_ROOT / "external/cad-venv/bin/python"


def _format_worker_failure(stderr: str, stdout: str) -> str:
    text = (stderr.strip() or stdout.strip()).replace("\x00", "")
    return text[-3000:] or f"exit status without diagnostic; stdout={stdout[:200]!r}"


def cad_request(request: Mapping[str, Any], *, interpreter: Optional[Path] = None,
                timeout_s: float = 600.0) -> Any:
    """Run one checked, bounded request; never auto-install CAD packages."""
    executable = Path(interpreter) if interpreter is not None else default_cad_interpreter()
    if not executable.is_file():
        raise CADUnavailable(
            f"blocked: cad env unavailable: isolated interpreter {executable} does not exist; "
            "see baselines/cad/README.md"
        )
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("CAD request timeout must be finite and positive")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "LD_PRELOAD", "PYTHONINSPECT"):
        env.pop(key, None)
    command = [str(executable), "-I", str(Path(__file__).resolve()), "--cad-worker"]
    try:
        payload = json.dumps(dict(request), sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid CAD request payload: {error}") from error
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(PROJECT_ROOT), env=env,
    )
    try:
        stdout_bytes, stderr_bytes = process.communicate(payload, timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        # Only the exact process started above is killed; no pattern kills.
        process.kill()
        process.communicate()
        raise CADUnavailable(
            f"blocked: cad env unavailable: isolated native CAD request timed out after {timeout_s:g}s"
        ) from error
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    try:
        record = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise CADValidationError(
            "native CAD did not return its required audit record: "
            + _format_worker_failure(stderr, stdout)
        ) from error
    if not isinstance(record, dict) or "status" not in record:
        raise CADValidationError("malformed native CAD audit record")
    if process.returncode or record["status"] != "ok":
        message = record.get("error") or _format_worker_failure(stderr, stdout)
        if record.get("error_kind") == "unavailable":
            raise CADUnavailable("blocked: cad env unavailable: " + str(message))
        raise CADValidationError(str(message) + ("\n" + record["traceback"] if record.get("traceback") else ""))
    return record


# ---------------------------------------------------------------------------
# Parent-only sampled-file handling.  No heavy CAD imports occur here.
# ---------------------------------------------------------------------------
def _shape_values(name: str, values: Sequence[int]) -> tuple[int, int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{name}: expected a three-element shape")
    result = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name}: positive integer dimensions required, got {values!r}")
        result.append(value)
    return tuple(result)  # type: ignore[return-value]


def _require_hash(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("expected a 64-hex source SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("source SHA-256 must be hexadecimal") from error


def load_native_fields(output_dir: Path, shape: Sequence[int]) -> dict[str, Any]:
    """Load finite float64 files and convert them to ordinary hashed ``.npy``."""
    import numpy as np

    shape = _shape_values("native sample shape", shape)
    result = {}
    for name in ("part", "operation_target"):
        source = output_dir / f"{name}.f64le"
        expected = math.prod(shape) * 8
        if source.stat().st_size != expected:
            raise CADValidationError(
                f"incomplete native sampled field {name}: {source.stat().st_size} bytes, expected {expected}"
            )
        values = np.fromfile(source, dtype="<f8", count=math.prod(shape)).reshape(shape)
        if not np.isfinite(values).all():
            raise CADValidationError(f"non-finite values in native sampled field {name}")
        destination = output_dir / f"{name}.npy"
        np.save(destination, values, allow_pickle=False)
        result[name] = {
            "values": values,
            "path": destination,
            "sha256": "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    return result


def field_statistics(stock: Any, part: Any, target: Any, voxel_mm: float) -> dict[str, Any]:
    """Cellwise checks, explicitly not native continuous-volume measurements."""
    import numpy as np

    stock = np.asarray(stock, dtype=np.float64)
    part = np.asarray(part, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if stock.shape != part.shape or stock.shape != target.shape:
        raise ValueError("stock, part and operation-target shapes must match")
    if not (math.isfinite(voxel_mm) and voxel_mm > 0):
        raise ValueError("sample voxel size must be finite and positive")
    if not (np.isfinite(stock).all() and np.isfinite(part).all() and np.isfinite(target).all()):
        raise ValueError("all sampled fields must be finite in physical millimetres")
    initial = stock <= 0
    original = part <= 0
    operation = target <= 0
    if np.any(original & ~operation):
        raise CADValidationError("sampled original part is not a subset of sampled operation target T")
    if np.any(original & ~initial):
        raise CADValidationError("sampled original part extends outside declared stock")
    excluded = initial & operation & ~original
    removal = initial & ~operation
    if not excluded.any():
        raise CADValidationError("declared operation exclusion is empty at this grid resolution")
    if not removal.any():
        raise CADValidationError("operation has empty removal target; nominal geometry may be unfinished")
    volume = voxel_mm ** 3
    return {
        "membership_rule": "phi <= 0; cell centres; float64 millimetres",
        "cell_count": int(stock.size),
        "original_part_cells": int(original.sum()),
        "operation_target_cells": int(operation.sum()),
        "excluded_cells_initial_and_T_not_P": int(excluded.sum()),
        "excluded_volume_voxel_estimate_mm3": float(excluded.sum() * volume),
        "removal_target_cells_initial_not_T": int(removal.sum()),
        "removal_target_volume_voxel_estimate_mm3": float(removal.sum() * volume),
        "original_part_subset_of_operation_target_at_cell_centres": True,
        "note": "Voxel counts are not exact B-rep volumes and do not qualify nominal CAM.",
    }


def inspect_titan_step(step_path: Path, *, interpreter: Optional[Path] = None) -> Mapping[str, Any]:
    """Real STEP inspection only; this route never writes an SDF."""
    return cad_request({
        "operation": "inspect", "step_path": str(Path(step_path).resolve()),
        "expected_sha256": TITAN_M1_SHA256,
        "reference_cylinder_in": 0.195,
    }, interpreter=interpreter)


# ---------------------------------------------------------------------------
# Isolated worker.  OCP is imported only for the exact command-line worker,
# never as a dependency of normal tests or the core project environment.
# ---------------------------------------------------------------------------
_IS_WORKER = __name__ == "__main__" and sys.argv[1:] == ["--cad-worker"]
if _IS_WORKER:
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRep import BRep_Builder
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepPrimAPI import (
            BRepPrimAPI_MakeBox,
            BRepPrimAPI_MakeCylinder,
        )
        from OCP.GProp import GProp_GProps
        from OCP.GeomAbs import GeomAbs_SurfaceType
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.Interface import Interface_Static
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import (
            TopAbs_FACE,
            TopAbs_SHELL,
            TopAbs_SOLID,
            TopAbs_State,
            TopAbs_ShapeEnum,
        )
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Compound
    except (ImportError, OSError, AttributeError) as error:
        print(json.dumps({
            "status": "error", "error_kind": "unavailable",
            "error": f"qualified isolated OCP import failed: {error}",
        }))
        raise SystemExit(1) from error


def _worker_environment() -> dict[str, Any]:
    import importlib.metadata
    import platform

    version = importlib.metadata.version("cadquery-ocp")
    if not (
        sys.platform == "linux"
        and platform.machine() == "x86_64"
        and sys.implementation.name == "cpython"
        and sys.version_info[:2] == (3, 12)
        and version == CAD_VERSION
    ):
        raise CADUnavailable(
            f"wrong pinned CAD runtime: CPython {(3, 12)}, linux x86_64, "
            f"cadquery-ocp=={CAD_VERSION} required; got {sys.version_info[:2]}, "
            f"{sys.platform} {platform.machine()}, {version}"
        )
    if str(PROJECT_ROOT) in sys.path:
        raise CADUnavailable("repository path leaked into the isolated CAD interpreter's sys.path")
    return {
        "cadquery_ocp": version,
        "wheel_sha256": CAD_WHEEL_SHA256,
        "isolated_python": sys.executable,
        "isolated_prefix": sys.prefix,
        "std_path_has_repository": str(PROJECT_ROOT) in sys.path,
        "python_version": platform.python_version(),
        "platform": f"{sys.platform}/{platform.machine()}",
    }


def _walk(shape: Any, kind: Any) -> list[Any]:
    result = []
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result


def _volume(shape: Any) -> float:
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    value = float(properties.Mass())
    if not math.isfinite(value):
        raise CADValidationError("non-finite native volume")
    return value


def _bounds(shape: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    raw = tuple(float(value) for value in box.Get())
    if len(raw) != 6 or not all(math.isfinite(value) for value in raw):
        raise CADValidationError("non-finite native bounding box")
    lower, upper = raw[:3], raw[3:]
    if any(hi <= lo for lo, hi in zip(lower, upper)):
        raise CADValidationError(f"empty or inverted native bounding box: {raw}")
    return lower, upper


def _valid_solid(shape: Any, *, label: str) -> Any:
    if shape.IsNull():
        raise CADValidationError(f"{label}: native shape is null after transfer/operation")
    solids = _walk(shape, TopAbs_SOLID)
    if len(solids) != 1:
        raise CADValidationError(f"{label}: expected exactly one solid, got {len(solids)}")
    solid = TopoDS.Solid_s(solids[0])
    if not BRepCheck_Analyzer(shape).IsValid():
        raise CADValidationError(f"{label}: OpenCascade B-rep validity check failed")
    if len(_walk(shape, TopAbs_SHELL)) != 1:
        raise CADValidationError(f"{label}: expected one closed boundary shell")
    if _volume(shape) <= 0:
        raise CADValidationError(f"{label}: nonpositive native solid volume")
    _bounds(shape)
    return solid


def _check_source_file(request: Mapping[str, Any]) -> tuple[Path, bytes]:
    expected = request.get("expected_sha256", "")
    _require_hash(expected)
    path = Path(request["step_path"]).resolve(strict=True)
    if path.stat().st_size > 100_000_000:
        raise CADValidationError("STEP file exceeds the bounded 100 MB adapter input")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise CADValidationError(f"STEP SHA-256 mismatch: expected {expected}, got {actual}")
    return path, data


def _import_step(request: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    path, data = _check_source_file(request)
    text = data.decode("ascii")
    if "CONVERSION_BASED_UNIT('inch'" not in text or "MANIFOLD_SOLID_BREP" not in text:
        raise CADValidationError("expected inch-unit manifold STEP B-rep source is not present")
    reader = STEPControl_Reader()
    if not Interface_Static.SetCVal_s("write.step.unit", "MM"):
        raise CADValidationError("could not select native STEP model-length unit MM")
    if Interface_Static.IVal_s("write.step.unit") != 2:
        raise CADValidationError("native STEP model unit is not MM")
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise CADValidationError(f"STEP ReadFile failed: {status}")
    roots = int(reader.NbRootsForTransfer())
    if roots != 1:
        raise CADValidationError(f"expected one STEP root for transfer, got {roots}")
    translated_roots = int(reader.TransferRoots())
    transferred_shapes = int(reader.NbShapes())
    if translated_roots != 1 or transferred_shapes != 1:
        raise CADValidationError(
            f"expected one translated root and shape, got {translated_roots} / {transferred_shapes}"
        )
    shape = reader.OneShape()
    if shape.ShapeType() != TopAbs_ShapeEnum.TopAbs_SOLID:
        raise CADValidationError(f"expected a STEP solid, got {shape.ShapeType()}")
    solid = _valid_solid(shape, label="Titan-M1 source")
    faces = _walk(shape, TopAbs_FACE)
    if len(faces) != 67:
        raise CADValidationError(f"Titan-M1 source topology changed: {len(faces)} faces, expected 67")
    lower, upper = _bounds(shape)
    extent = tuple(hi - lo for lo, hi in zip(lower, upper))
    nominal = (101.6, 48.26, 19.05)
    if any(abs(a - b) > 1e-5 for a, b in zip(extent, nominal)):
        raise CADValidationError(f"physical source extents are not {nominal} mm: {extent}")
    inch_radius = float(request.get("reference_cylinder_in", 0.0))
    if not math.isfinite(inch_radius) or inch_radius <= 0:
        raise CADValidationError("known cylindrical reference dimension must be finite and positive")
    expected_radius = inch_radius * 25.4
    matches = []
    for untyped_face in faces:
        face = TopoDS.Face_s(untyped_face)
        adapter = BRepAdaptor_Surface(face)
        if adapter.GetType() != GeomAbs_SurfaceType.GeomAbs_Cylinder:
            continue
        cylinder = adapter.Cylinder()
        axis = cylinder.Axis().Direction()
        if abs(axis.Z()) <= 0.999999999:
            continue
        radius = float(cylinder.Radius())
        if abs(radius - expected_radius) <= 1e-6:
            location = cylinder.Location()
            matches.append({
                "radius_mm": radius,
                "axis": [float(axis.X()), float(axis.Y()), float(axis.Z())],
                "location_mm": [float(location.X()), float(location.Y()), float(location.Z())],
            })
    if not matches:
        raise CADValidationError(f"source cylinder at {inch_radius} in / {expected_radius} mm was not found")
    return solid, {
        "schema": "native-step-inspection/1",
        "status": "qualified",
        "source_unit": "inch",
        "native_model_unit": "mm",
        "inch_to_mm_multiplier": 25.4,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "root_count": roots,
        "transferred_shape_count": transferred_shapes,
        "solid_count": 1,
        "shell_count": 1,
        "face_count": len(faces),
        "brep_valid": True,
        "volume_mm3": _volume(shape),
        "bounds_mm": [list(lower), list(upper)],
        "extent_mm": list(extent),
        "reference_dimension": {
            "artifact_statement": f"CYLINDRICAL_SURFACE literal {inch_radius} inch",
            "expected_radius_mm": expected_radius,
            "matches": matches,
            "max_abs_error_mm": max(abs(m["radius_mm"] - expected_radius) for m in matches),
            "not_vendor_certification": True,
        },
        "license": "unknown / provenance unrecorded",
        "usage": "local only; input STEP is not copied or redistributed by this adapter",
        "mesh_import_used": False,
    }


def _native_boundary(solid: Any) -> Any:
    # This OCP wheel does not expose BRepTools.OuterShell.  A compound of all
    # faces measures every boundary, including an internal closed shell; it
    # does not turn interior points of a solid into zero-distance points.
    boundary = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(boundary)
    for face in _walk(solid, TopAbs_FACE):
        builder.Add(boundary, face)
    return boundary


def _inside(solid: Any, point: Sequence[float]) -> bool:
    classifier = BRepClass3d_SolidClassifier(solid, gp_Pnt(*point), 1e-7)
    state = classifier.State()
    if state not in (TopAbs_State.TopAbs_IN, TopAbs_State.TopAbs_ON, TopAbs_State.TopAbs_OUT):
        raise CADValidationError(f"unclassified native point {tuple(point)}: {state}")
    return state in (TopAbs_State.TopAbs_IN, TopAbs_State.TopAbs_ON)


class _BoundaryDistance:
    def __init__(self, solid: Any):
        self.solid = solid
        self.query = BRepExtrema_DistShapeShape()
        self.query.LoadS2(_native_boundary(solid))

    def value(self, point: Sequence[float]) -> float:
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Shape()
        self.query.LoadS1(vertex)
        if not self.query.Perform() or not self.query.IsDone():
            raise CADValidationError(f"native point-to-boundary distance failed at {tuple(point)}")
        value = float(self.query.Value())
        if not math.isfinite(value) or value < -1e-9:
            raise CADValidationError(f"invalid native distance {value} at {tuple(point)}")
        return abs(value)


def _vertical_cylinders_at(solid: Any, xy: Sequence[float]) -> list[dict[str, Any]]:
    result = []
    for shape in _walk(solid, TopAbs_FACE):
        face = TopoDS.Face_s(shape)
        adapter = BRepAdaptor_Surface(face)
        if adapter.GetType() != GeomAbs_SurfaceType.GeomAbs_Cylinder:
            continue
        cylinder = adapter.Cylinder()
        location, axis = cylinder.Location(), cylinder.Axis().Direction()
        if abs(axis.Z()) <= 0.999999999:
            continue
        if max(abs(location.X() - xy[0]), abs(location.Y() - xy[1])) > 1e-6:
            continue
        lower, upper = _bounds(face)
        result.append({
            "centre_xy_mm": [float(location.X()), float(location.Y())],
            "radius_mm": float(cylinder.Radius()),
            "axis": [float(axis.X()), float(axis.Y()), float(axis.Z())],
            "face_z_range_mm": [float(lower[2]), float(upper[2])],
        })
    return result


def _common_volume(left: Any, right: Any) -> float:
    common = BRepAlgoAPI_Common(left, right)
    if not common.IsDone():
        raise CADValidationError("native common Boolean failed")
    return _volume(common.Shape())


def _nominal_void_profiles(source: Any, faces: Sequence[Mapping[str, Any]],
                           opening_z_mm: float) -> list[dict[str, Any]]:
    """Record conservative nominal bore void, verified against the source.

    Every cylinder starts at an observed native face's lower Z bound and ends
    at the validated solid's upper Z bound. The face's own z-range is retained
    separately; it is not asserted to bound the full nominal void. A nonzero
    common with the original source rejects the profile before any sampling.
    """
    profiles = []
    seen = set()
    previous_upper = None
    for face in sorted(faces, key=lambda record: record["face_z_range_mm"]):
        z_min, face_max = map(float, face["face_z_range_mm"])
        if not (math.isfinite(z_min) and math.isfinite(face_max)):
            raise CADValidationError("non-finite native bore face z-range")
        if face["axis"][2] <= 0:
            raise CADValidationError("this +Z nominal void needs a positive native bore axis")
        radius = float(face["radius_mm"])
        key = tuple(round(value, 8)
                    for value in (*face["centre_xy_mm"], radius, z_min, face_max))
        if key in seen:
            continue
        seen.add(key)
        if previous_upper is not None and abs(z_min - previous_upper) > 1e-6:
            raise CADValidationError("native bore face depth intervals are not contiguous")
        if opening_z_mm <= z_min:
            raise CADValidationError("nominal bore opening height must exceed its lower bound")
        centre = face["centre_xy_mm"]
        nominal = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(float(centre[0]), float(centre[1]), z_min), gp_Dir(0, 0, 1)),
            radius, opening_z_mm - z_min,
        ).Shape()
        source_common = _common_volume(source, nominal)
        if abs(source_common) > 1e-6:
            raise CADValidationError(
                f"nominal bore void has {source_common} mm^3 common with original part"
            )
        profiles.append({
            "radius_mm": radius,
            "source_z_min_mm": z_min,
            "source_opening_z_mm": opening_z_mm,
            "face_z_range_mm": [z_min, face_max],
            "nominal_cylinder_common_with_original_part_mm3": source_common,
            "construction": "face lower z to source upper z; source-negative native common zero",
        })
        previous_upper = face_max
    if len(profiles) != 2:
        raise CADValidationError(
            f"this bounded pilot requires two contiguous bore profiles, got {len(profiles)}"
        )
    return profiles


def _point_from_work(point: Sequence[float], source_to_work: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(point[axis]) - float(source_to_work[axis]) for axis in range(3))


def _sample_solid(solid: Any, path: Path, grid: Mapping[str, Any],
                  source_to_work: Sequence[float]) -> None:
    shape = _shape_values("request.grid.shape", grid["shape"])
    if math.prod(shape) > 200_000:
        raise CADValidationError("native sampler cell limit is 200000; choose a bounded pilot grid")
    origin, voxel = grid["origin_mm"], float(grid["voxel_mm"])
    if len(origin) != 3 or any(not math.isfinite(float(v)) for v in origin):
        raise CADValidationError("grid origin must be finite physical millimetres")
    if not math.isfinite(voxel) or voxel <= 0:
        raise CADValidationError("grid voxel_mm must be finite and positive")
    distance = _BoundaryDistance(solid)
    with path.open("xb") as stream:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    work = [origin[a] + (index + 0.5) * voxel for a, index in enumerate((i, j, k))]
                    point = _point_from_work(work, source_to_work)
                    sign = -1.0 if _inside(solid, point) else 1.0
                    stream.write(struct.pack("<d", sign * distance.value(point)))


def _pilot(request: Mapping[str, Any]) -> dict[str, Any]:
    source, inspection = _import_step(request)
    geometry = request["geometry"]
    source_to_work = geometry["source_to_work_mm"]
    if len(source_to_work) != 3 or any(not math.isfinite(float(v)) for v in source_to_work):
        raise CADValidationError("source-to-work translation must be finite")
    tool = geometry["tool"]
    radius = float(tool["cutter_radius_mm"])
    flute = float(tool["flute_length_mm"])
    if not (math.isfinite(radius) and math.isfinite(flute) and radius > 0 and flute > 0):
        raise CADValidationError("pilot flat-cutter radius and flute length must be finite and positive")
    stock = geometry["stock"]
    stock_native = [
        _point_from_work(stock[key], source_to_work) for key in ("min_mm", "max_mm")
    ]
    if any(hi <= lo for lo, hi in zip(*stock_native)):
        raise CADValidationError("stock bounds must be strictly increasing")
    stock_depth = stock_native[1][2] - stock_native[0][2]
    if flute - stock_depth < 1.0:
        raise CADValidationError("flute does not extend at least 1 mm above stock at its deepest pose")
    if float(tool["holder_bottom_mm"]) <= flute:
        raise CADValidationError("this flat-cutter pilot requires a distinct shank below the holder")
    hole_xy = geometry["hole_centre_source_xy_mm"]
    faces = _vertical_cylinders_at(source, hole_xy)
    radii = sorted({round(item["radius_mm"], 8) for item in faces})
    required_radii = sorted(float(v) for v in geometry["expected_hole_radii_mm"])
    if len(radii) != len(required_radii) or any(
        abs(a - b) > 1e-5 for a, b in zip(radii, required_radii)
    ):
        raise CADValidationError(f"hole dimensions differ: {radii} versus {required_radii}")
    if max(radii) >= radius:
        raise CADValidationError("declared small-hole exclusion is not narrower than the selected cutter")
    hole_probe = (float(hole_xy[0]), float(hole_xy[1]), -9.525)
    if _inside(source, hole_probe):
        raise CADValidationError("selected source bore is not an actual through-hole void at its centre")
    lower, upper = _bounds(source)
    nominal_void_profile = _nominal_void_profiles(source, faces, upper[2])
    # The plug uses native model-unit millimetres.  Tiny bounding-box tolerance
    # padding is not silently added to physical dimensions.  The selected bore
    # is verified to fit inside the declared stock before constructing the plug.
    plug_low = [hole_xy[0], hole_xy[1], stock_native[0][2]]
    if any(
        plug_low[a] - radius < stock_native[0][a]
        or plug_low[a] + radius > stock_native[1][a]
        for a in (0, 1)
    ):
        raise CADValidationError("selected exclusion plug would extend outside stock laterally")
    if stock_native[0][2] > lower[2] + 1e-5 or stock_native[1][2] < upper[2] - 1e-5:
        raise CADValidationError("selected stock does not cover the full source part vertically")
    plug = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(float(hole_xy[0]), float(hole_xy[1]), stock_native[0][2]), gp_Dir(0, 0, 1)),
        radius, stock_depth,
    ).Shape()
    target_operation = BRepAlgoAPI_Fuse(source, plug)
    if not target_operation.IsDone():
        raise CADValidationError("native original-part / exclusion union failed")
    target = _valid_solid(target_operation.Shape(), label="operation target T = P UNION exclusion")
    if not _inside(target, hole_probe):
        raise CADValidationError("selected through-hole exclusion was not retained in target T")
    excluded_volume = _volume(target) - _volume(source)
    if excluded_volume <= 0:
        raise CADValidationError("native excluded volume is empty")
    grid = geometry["grid"]
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    # Sampling occurs only AFTER source topology, units/dimension and target
    # topology plus the native additive subset construction have been checked.
    _sample_solid(source, output / "part.f64le", grid, source_to_work)
    _sample_solid(target, output / "operation_target.f64le", grid, source_to_work)
    feed = geometry["nominal_feed"]
    start_work, end_work = feed["start_work_mm"], feed["end_work_mm"]
    if any(abs(float(start_work[a]) - float(end_work[a])) > 1e-9 for a in (0, 1)):
        raise CADValidationError("this bounded native sweep supports one vertical feed only")
    start = _point_from_work(start_work, source_to_work)
    end = _point_from_work(end_work, source_to_work)
    if end[2] >= start[2] or min(start[2], end[2]) + flute < upper[2]:
        raise CADValidationError("plunge direction/flute coverage is invalid for this +Z pilot")
    cutter = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(*start), gp_Dir(0, 0, 1)), radius, flute,
    ).Shape()
    # A vertical translation of a cylinder along its own axis is exactly the
    # cylinder whose base is the deepest cutter pose and whose height includes
    # the feed length.  This is not a sampled/chord approximation or a silent
    # substitution of a generic swept-box tool.
    sweep = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(start[0], start[1], end[2]), gp_Dir(0, 0, 1)),
        radius, flute + (start[2] - end[2]),
    ).Shape()
    _valid_solid(sweep, label="one continuous vertical cylindrical feed sweep")
    stock_solid = BRepPrimAPI_MakeBox(gp_Pnt(*stock_native[0]), gp_Pnt(*stock_native[1])).Shape()
    removed_volume = _common_volume(sweep, stock_solid)
    if removed_volume <= 0:
        raise CADValidationError("nominal feed actually removes no stock")
    original_common = _common_volume(sweep, source)
    target_common = _common_volume(sweep, target)
    return {
        "status": "ok",
        "environment": _worker_environment(),
        "inspection": inspection,
        "operation_geometry": {
            "source_frame": "native STEP model coordinates",
            "work_frame": "stock top-centre G54; physical millimetres",
            "source_to_work_mm": list(source_to_work),
            "tool": dict(tool),
            "hole_cylinders": faces,
            "hole_centre_source_xy_mm": list(hole_xy),
            "exclusion": {
                "construction": "native BRepAlgoAPI_Fuse(P, cutter-radius +Z cylinder)",
                "plug_radius_mm": radius,
                "plug_min_source_z_mm": stock_native[0][2],
                "plug_max_source_z_mm": upper[2],
                "nominal_void_profile": nominal_void_profile,
                "native_excluded_volume_mm3": excluded_volume,
                "source_void_probe_source_mm": list(hole_probe),
                "source_void_probe_is_outside_P": True,
                "probe_retained_inside_T": True,
                "scope": "one measured small hole; not a certification that every other feature is accessible",
            },
            "target": {
                "volume_mm3": _volume(target),
                "bounds_mm": [list(values) for values in _bounds(target)],
                "solid_count": 1,
                "brep_valid": True,
            },
        },
        "native_nominal_sweep": {
            "start_source_mm": list(start),
            "end_source_mm": list(end),
            "continuous_prismatic_cutter_sweep": True,
            "construction": "exact native cylinder spanning deepest cutter base through the start cutter top",
            "stock_common_volume_mm3": removed_volume,
            "original_part_common_volume_mm3": original_common,
            "operation_target_common_volume_mm3": target_common,
            "zero_common_volume_tolerance_mm3": 1e-6,
            "part_no_gouge_supported": original_common <= 1e-6,
            "operation_target_no_gouge_supported": target_common <= 1e-6,
            "scope": "one feed only; no general toolpath, fixture or machining-safety certificate",
        },
        "sample_format": {
            "dtype": "little-endian IEEE float64",
            "order": "C (i,j,k)",
            "values": "exact native point-to-boundary distances with BRepClass3d IN/ON negative",
            "grid": dict(grid),
            "cell_membership": "phi <= 0",
            "unit": "mm",
        },
    }


def _worker_main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        _worker_environment()
        operation = request.get("operation")
        if operation == "environment":
            return_record = {"status": "ok", "environment": _worker_environment()}
        elif operation == "inspect":
            _, record = _import_step(request)
            return_record = {"status": "ok", "environment": _worker_environment(), "inspection": record}
        elif operation == "titan-pilot":
            return_record = _pilot(request)
        else:
            raise ValueError(f"unknown CAD operation {operation!r}")
        print(json.dumps(return_record, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except CADUnavailable as error:
        print(json.dumps({"status": "error", "error_kind": "unavailable", "error": str(error)}))
        return 1
    except Exception as error:
        import traceback
        print(json.dumps({
            "status": "error", "error_kind": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc()[-6000:],
        }))
        return 1


if __name__ == "__main__":
    if _IS_WORKER:
        raise SystemExit(_worker_main())
    raise SystemExit("import this module, or invoke it with -I --cad-worker from benchmarks.external_tasks")
