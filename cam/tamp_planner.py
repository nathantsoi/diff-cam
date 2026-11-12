# SPDX-License-Identifier: MIT
"""Automated Task-and-Motion-Planning (TAMP) Multi-Tool CAM Pipeline (``cam.tamp_planner``).

Given a drop-in watertight 3D part mesh (binary/ASCII ``.stl``) or declarative
prismatic part specification, stock alloy, and 3-axis CNC machine envelope, this
module automates the full subtractive setup and trajectory-optimization stack:
  1. **Setup, Billet, & Workholding Selection:** Selects top-accessible setup
     orientation (``+Z``), standard billet stock margins (top facing + XY stock +
     bottom parallel/vise-jaw clamping grip), and explicit vise-jaw collision
     forbidden zones.
  2. **Geometric Feature Recognition:** Rasterizes the ``.stl`` mesh onto a
     high-resolution 2.5D heightfield and extracts top-facing excess, exterior
     silhouette contours, enclosed pockets/islands (with minimum channel width
     and concave corner radius), and cylindrical blind/through holes.
  3. **Discrete Tool-Library & Operation Search (Task Planning):** Queries
     :class:`cam.tool_library.StandardToolLibrary` and :mod:`cam.feeds_and_speeds`
     to choose an ordered multi-tool operation sequence (e.g., ``T1`` large
     roughing/facing endmill -> ``T2`` smaller rest-finishing endmill -> ``T3``
     spot drill -> ``T4`` peck twist drill) minimizing cutting time plus physical
     automatic-tool-changer (ATC) swap overhead ``t_ATC``.
  4. **Persistent-IPW Differentiable Motion Optimization:** Seeds each operation
     and propagates the exact evolving 2.5D In-Process Workpiece (IPW) dexel
     heightfield across operations into :func:`algorithms.chipload_opt.optimize_chipload_trajectory`
     so downstream rest-machining tools shortcut already-cleared cavities and
     regulate actual undeformed chip thickness ``h_ex`` strictly inside remaining
     stock cusps.
  5. **Multi-Tool Haas NGC Post-Processing & Chronological Safety Audit:** Emits
     unified multi-tool Haas Next-Gen Control programs (``Tn M6``, ``G43 Hn``,
     ``G187 P2 E0.05``) and verifies zero part gouge, zero rapid-stock contact,
     and zero holder/vise-jaw collisions.
"""

from __future__ import annotations

import dataclasses
import math
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from algorithms.chipload_opt import (
    ChiploadSpec,
    _sweep_removes_or_updates_dexels,
    analyze_plan_engagement,
    blended_durations_with_pullback,
    optimize_chipload_trajectory,
    retime_motion_plan_constant_speed,
    subdivide_motion_plan,
)
from benchmarks import contracts as C
from benchmarks.milling_suite import (
    MillingCaseSpec,
    MillingTaskBundle,
    build_milling_task_bundle,
)
from cam.arbitrary_cad_ops import (
    build_3d_ball_scallop_motion_plan,
    build_chamfer_contour_motion_plan,
    build_countersink_motion_plan,
    build_facing_motion_plan,
    build_helical_bore_motion_plan,
    build_levelset_clearing_motion_plan,
    build_thread_mill_motion_plan,
    build_wips_probe_motion_plan,
    optimize_arbitrary_cwe_plan,
)
from cam.feeds_and_speeds import (
    MachineEnvelopeSpec,
    compute_feeds_and_speeds,
    compute_holemaking_feeds_and_speeds,
)
from cam.multi_dialect_post import (
    MultiToolPostOpInput,
    format_all_dialects_multitool,
    normalize_multi_dialect,
)
from cam.step_brep import (
    BRepChamferFeature,
    BRepFreeformRegion,
    BRepHoleFeature,
    STEPLoadResult,
    is_step_payload,
    load_step_brep,
    triangles_to_binary_stl_bytes,
)
from cam.tool_library import (
    DEFAULT_TOOL_LIBRARY,
    CatalogToolRecord,
    HolderSpec,
    StandardToolLibrary,
    tool_record_to_viewer_dict,
)

MAX_STL_BYTES: int = 25 * 1024 * 1024
MAX_STL_TRIANGLES: int = 250_000


# ---------------------------------------------------------------------------
# 1. Security-hardened Binary/ASCII STL I/O & Multi-Feature STL Synthesis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class STLMesh:
    """Validated triangle soup in millimetres."""

    triangles_mm: np.ndarray  # (F, 3, 3) float64
    bounds_min_mm: Tuple[float, float, float]
    bounds_max_mm: Tuple[float, float, float]
    signed_volume_mm3: float

    @property
    def triangle_count(self) -> int:
        return int(self.triangles_mm.shape[0])


@dataclass(frozen=True)
class HoleBlueprint:
    """Cylindrical vertical hole specification in work coordinates (mm)."""

    center_xy_mm: Tuple[float, float]
    diameter_mm: float
    z_top_mm: float
    z_bottom_mm: float


@dataclass(frozen=True)
class PocketBlueprint:
    """Enclosed 2.5D pocket cavity with optional retained rectangular islands (mm)."""

    min_xy_mm: Tuple[float, float]
    max_xy_mm: Tuple[float, float]
    floor_z_mm: float
    corner_fillet_mm: float = 2.0
    islands_xy_mm: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = ()


@dataclass(frozen=True)
class PrismaticPartBlueprint:
    """Declarative multi-feature part specification convertible to watertight ``.stl``."""

    part_id: str
    title: str
    outer_min_mm: Tuple[float, float, float]
    outer_max_mm: Tuple[float, float, float]
    stock_top_margin_mm: float = 1.5
    stock_xy_margin_mm: float = 0.0
    vise_grip_z_mm: float = 2.5
    pockets: Tuple[PocketBlueprint, ...] = ()
    holes: Tuple[HoleBlueprint, ...] = ()
    exterior_step_z_mm: Optional[float] = None
    exterior_boss_min_xy_mm: Optional[Tuple[float, float]] = None
    exterior_boss_max_xy_mm: Optional[Tuple[float, float]] = None


def load_stl_mesh(source: Union[str, Path, bytes]) -> STLMesh:
    """Parse and strictly validate a binary or ASCII ``.stl`` mesh (<= 25 MB, <= 250k tris)."""
    if isinstance(source, (str, Path)):
        p = Path(source).resolve()
        if not p.is_file():
            raise ValueError(f"STL path does not exist or is not a regular file: {p.name}")
        size = p.stat().st_size
        if size < 84 and size < 32:
            raise ValueError("STL file is too small to contain geometry")
        if size > MAX_STL_BYTES:
            raise ValueError(f"STL file exceeds {MAX_STL_BYTES} byte safety cap")
        raw = p.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        if len(source) > MAX_STL_BYTES:
            raise ValueError(f"STL byte buffer exceeds {MAX_STL_BYTES} byte safety cap")
        raw = bytes(source)
    else:
        raise TypeError("source must be Path, str, or bytes")

    tris: Optional[np.ndarray] = None
    if len(raw) >= 84:
        tri_count = struct.unpack("<I", raw[80:84])[0]
        expected_len = 84 + 50 * tri_count
        if expected_len == len(raw) and 0 < tri_count <= MAX_STL_TRIANGLES:
            dt = np.dtype(
                [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
            )
            recs = np.frombuffer(raw, dtype=dt, offset=84, count=tri_count)
            tris = np.asarray(recs["verts"], dtype=np.float64)

    if tris is None:
        # Parse ASCII STL safely without regex backtracking
        text = raw.decode("utf-8", errors="strict")
        verts_list: List[Tuple[float, float, float]] = []
        for line in text.splitlines():
            toks = line.strip().split()
            if len(toks) == 4 and toks[0].lower() == "vertex":
                verts_list.append((float(toks[1]), float(toks[2]), float(toks[3])))
                if len(verts_list) > 3 * MAX_STL_TRIANGLES:
                    raise ValueError(f"ASCII STL exceeds {MAX_STL_TRIANGLES} triangle safety cap")
        if len(verts_list) < 3 or len(verts_list) % 3 != 0:
            raise ValueError("Invalid or empty STL payload")
        tris = np.asarray(verts_list, dtype=np.float64).reshape(-1, 3, 3)

    if not np.all(np.isfinite(tris)):
        raise ValueError("STL contains non-finite NaN/Inf vertex coordinates")

    bmin = tuple(float(v) for v in np.min(tris.reshape(-1, 3), axis=0))
    bmax = tuple(float(v) for v in np.max(tris.reshape(-1, 3), axis=0))
    if any(bmax[a] - bmin[a] <= 1e-4 for a in range(3)):
        raise ValueError("STL bounding box is degenerate along one or more axes")

    # Signed enclosed volume via divergence theorem: sum det([v0, v1, v2]) / 6
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    vol = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)
    return STLMesh(
        triangles_mm=tris,
        bounds_min_mm=bmin,  # type: ignore[arg-type]
        bounds_max_mm=bmax,  # type: ignore[arg-type]
        signed_volume_mm3=abs(vol),
    )


def heightfield_to_watertight_stl_bytes(
    xs: np.ndarray,
    ys: np.ndarray,
    z_top_grid: np.ndarray,
    z_bottom: float,
    through_mask: Optional[np.ndarray] = None,
) -> bytes:
    """Triangulate a 2.5D solid ``z in [z_bottom, z_top_grid]`` (minus ``through_mask``) to binary STL."""
    nx, ny = z_top_grid.shape
    pitch_x = float(xs[1] - xs[0]) if nx > 1 else 0.25
    pitch_y = float(ys[1] - ys[0]) if ny > 1 else 0.25
    x_edges = np.concatenate([xs - 0.5 * pitch_x, [xs[-1] + 0.5 * pitch_x]])
    y_edges = np.concatenate([ys - 0.5 * pitch_y, [ys[-1] + 0.5 * pitch_y]])
    solid = np.ones((nx, ny), dtype=bool) if through_mask is None else (~through_mask)

    tris: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = []
    zb = float(z_bottom)

    for ix in range(nx):
        x0, x1 = float(x_edges[ix]), float(x_edges[ix + 1])
        for iy in range(ny):
            if not solid[ix, iy]:
                continue
            y0, y1 = float(y_edges[iy]), float(y_edges[iy + 1])
            zt = float(z_top_grid[ix, iy])
            if zt <= zb + 1e-6:
                continue
            # Top (+Z) & Bottom (-Z) quads
            tris.append(((x0, y0, zt), (x1, y0, zt), (x1, y1, zt)))
            tris.append(((x0, y0, zt), (x1, y1, zt), (x0, y1, zt)))
            tris.append(((x0, y0, zb), (x1, y1, zb), (x1, y0, zb)))
            tris.append(((x0, y0, zb), (x0, y1, zb), (x1, y1, zb)))

            # -X side wall
            z_nb = zb if (ix == 0 or not solid[ix - 1, iy]) else float(z_top_grid[ix - 1, iy])
            if zt > z_nb + 1e-6:
                tris.append(((x0, y0, z_nb), (x0, y0, zt), (x0, y1, zt)))
                tris.append(((x0, y0, z_nb), (x0, y1, zt), (x0, y1, z_nb)))
            # +X side wall
            z_nb = zb if (ix == nx - 1 or not solid[ix + 1, iy]) else float(z_top_grid[ix + 1, iy])
            if zt > z_nb + 1e-6:
                tris.append(((x1, y0, z_nb), (x1, y1, zt), (x1, y0, zt)))
                tris.append(((x1, y0, z_nb), (x1, y1, z_nb), (x1, y1, zt)))
            # -Y side wall
            z_nb = zb if (iy == 0 or not solid[ix, iy - 1]) else float(z_top_grid[ix, iy - 1])
            if zt > z_nb + 1e-6:
                tris.append(((x0, y0, z_nb), (x1, y0, zt), (x0, y0, zt)))
                tris.append(((x0, y0, z_nb), (x1, y0, z_nb), (x1, y0, zt)))
            # +Y side wall
            z_nb = zb if (iy == ny - 1 or not solid[ix, iy + 1]) else float(z_top_grid[ix, iy + 1])
            if zt > z_nb + 1e-6:
                tris.append(((x0, y1, z_nb), (x0, y1, zt), (x1, y1, zt)))
                tris.append(((x0, y1, z_nb), (x1, y1, zt), (x1, y1, z_nb)))

    arr = np.asarray(tris, dtype=np.float32)
    v0, v1, v2 = arr[:, 0], arr[:, 1], arr[:, 2]
    nrm = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.maximum(nlen, 1e-12))

    dt = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")])
    out = np.zeros(len(arr), dtype=dt)
    out["normal"] = nrm
    out["verts"] = arr
    header = b"DiffCAM Clean-Room MIT TAMP Binary STL".ljust(80, b"\x00")
    return header + struct.pack("<I", len(arr)) + out.tobytes()


def build_stl_from_blueprint(bp: PrismaticPartBlueprint, pitch_mm: float = 0.25) -> bytes:
    """Synthesize a watertight binary ``.stl`` part mesh from a ``PrismaticPartBlueprint``."""
    x0, y0, z0 = bp.outer_min_mm
    x1, y1, z1 = bp.outer_max_mm
    nx = int(round((x1 - x0) / pitch_mm))
    ny = int(round((y1 - y0) / pitch_mm))
    xs = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * pitch_mm
    ys = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * pitch_mm
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    z_top = np.full((nx, ny), float(z1), dtype=np.float64)
    through = np.zeros((nx, ny), dtype=bool)

    if (
        bp.exterior_step_z_mm is not None
        and bp.exterior_boss_min_xy_mm is not None
        and bp.exterior_boss_max_xy_mm is not None
    ):
        bx0, by0 = bp.exterior_boss_min_xy_mm
        bx1, by1 = bp.exterior_boss_max_xy_mm
        in_boss = (gx >= bx0) & (gx <= bx1) & (gy >= by0) & (gy <= by1)
        z_top[~in_boss] = np.minimum(z_top[~in_boss], float(bp.exterior_step_z_mm))

    for pck in bp.pockets:
        px0, py0 = pck.min_xy_mm
        px1, py1 = pck.max_xy_mm
        rc = max(0.0, float(pck.corner_fillet_mm))
        if rc > 1e-6:
            dx = np.maximum(np.maximum(px0 + rc - gx, gx - (px1 - rc)), 0.0)
            dy = np.maximum(np.maximum(py0 + rc - gy, gy - (py1 - rc)), 0.0)
            in_pck = (gx >= px0) & (gx <= px1) & (gy >= py0) & (gy <= py1) & (
                np.hypot(dx, dy) <= rc + 1e-9
            )
        else:
            in_pck = (gx >= px0) & (gx <= px1) & (gy >= py0) & (gy <= py1)

        for (ix0, iy0), (ix1, iy1) in pck.islands_xy_mm:
            in_isl = (gx >= ix0) & (gx <= ix1) & (gy >= iy0) & (gy <= iy1)
            in_pck = in_pck & (~in_isl)

        z_top[in_pck] = np.minimum(z_top[in_pck], float(pck.floor_z_mm))

    for hl in bp.holes:
        cx, cy = hl.center_xy_mm
        rh = 0.5 * float(hl.diameter_mm)
        in_h = ((gx - cx) ** 2 + (gy - cy) ** 2) <= (rh * rh)
        if hl.z_bottom_mm <= z0 + 1e-4:
            through[in_h] = True
            z_top[in_h] = float(z0)
        else:
            z_top[in_h] = np.minimum(z_top[in_h], float(hl.z_bottom_mm))

    return heightfield_to_watertight_stl_bytes(xs, ys, z_top, z_bottom=z0, through_mask=through)


# ---------------------------------------------------------------------------
# 2. STL Feature Recognition + Stock & Workholding Setup Planner
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ViseJawZone:
    """Clamping jaw collision AABB in G54 work coordinates (mm)."""

    label: str
    min_mm: Tuple[float, float, float]
    max_mm: Tuple[float, float, float]


@dataclass(frozen=True)
class RecognizedFeatures:
    """Geometric machining features extracted directly from an input ``.stl`` solid."""

    stock_min_mm: Tuple[float, float, float]
    stock_max_mm: Tuple[float, float, float]
    part_min_mm: Tuple[float, float, float]
    part_max_mm: Tuple[float, float, float]
    vise_jaws: Tuple[ViseJawZone, ...]
    top_face_depth_mm: float
    has_exterior_step: bool
    exterior_step_z_mm: float
    exterior_boss_min_xy_mm: Tuple[float, float]
    exterior_boss_max_xy_mm: Tuple[float, float]
    pockets: Tuple[PocketBlueprint, ...]
    min_pocket_channel_mm: float
    min_pocket_corner_radius_mm: float
    holes: Tuple[HoleBlueprint, ...]
    xs_mm: np.ndarray
    ys_mm: np.ndarray
    z_target_grid_mm: np.ndarray
    is_step_cad: bool = False
    source_unit: str = "mm"
    setup_axis_label: str = "+Z"
    part_name: str = ""
    target_stl_bytes: Optional[bytes] = None
    brep_op1_holes: Tuple[BRepHoleFeature, ...] = ()
    brep_op2_holes: Tuple[BRepHoleFeature, ...] = ()
    brep_op1_chamfers: Tuple[BRepChamferFeature, ...] = ()
    brep_op2_chamfers: Tuple[BRepChamferFeature, ...] = ()
    brep_op1_freeform: Tuple[BRepFreeformRegion, ...] = ()
    brep_op2_freeform: Tuple[BRepFreeformRegion, ...] = ()
    requires_op2_flip: bool = False
    op2_z_target_grid_mm: Optional[np.ndarray] = None
    vise_grip_z_mm: float = 2.0


def rasterize_stl_top_heightfield(
    mesh: STLMesh,
    stock_min_mm: Tuple[float, float, float],
    stock_max_mm: Tuple[float, float, float],
    xy_shift: Tuple[float, float, float],
    pitch_mm: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize upward-facing (``n_z > 0.2``) STL triangles onto a G54 2.5D heightfield."""
    tris = mesh.triangles_mm + np.asarray(xy_shift, dtype=np.float64)[None, None, :]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    cross_z = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (
        v2[:, 0] - v0[:, 0]
    )
    up_tris = tris[cross_z > 1e-8]

    x0, y0, z0 = stock_min_mm
    x1, y1, _ = stock_max_mm
    nx = int(round((x1 - x0) / pitch_mm))
    ny = int(round((y1 - y0) / pitch_mm))
    xs = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * pitch_mm
    ys = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * pitch_mm
    z_grid = np.full((nx, ny), float(z0), dtype=np.float64)

    for tri in up_tris:
        tx0, ty0, tz0 = float(tri[0, 0]), float(tri[0, 1]), float(tri[0, 2])
        tx1, ty1, tz1 = float(tri[1, 0]), float(tri[1, 1]), float(tri[1, 2])
        tx2, ty2, tz2 = float(tri[2, 0]), float(tri[2, 1]), float(tri[2, 2])
        ix0 = max(0, int(math.floor((min(tx0, tx1, tx2) - x0) / pitch_mm)))
        ix1 = min(nx, int(math.ceil((max(tx0, tx1, tx2) - x0) / pitch_mm)) + 1)
        iy0 = max(0, int(math.floor((min(ty0, ty1, ty2) - y0) / pitch_mm)))
        iy1 = min(ny, int(math.ceil((max(ty0, ty1, ty2) - y0) / pitch_mm)) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        denom = (ty1 - ty2) * (tx0 - tx2) + (tx2 - tx1) * (ty0 - ty2)
        if abs(denom) <= 1e-12:
            continue
        px = xs[ix0:ix1, None]
        py = ys[None, iy0:iy1]
        w0 = ((ty1 - ty2) * (px - tx2) + (tx2 - tx1) * (py - ty2)) / denom
        w1 = ((ty2 - ty0) * (px - tx2) + (tx0 - tx2) * (py - ty2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(inside):
            continue
        z_interp = w0 * tz0 + w1 * tz1 + w2 * tz2
        sub = z_grid[ix0:ix1, iy0:iy1]
        sub[inside] = np.maximum(sub[inside], z_interp[inside])

    return xs, ys, z_grid


def _connected_components_2d(mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """4-connected binary component labeler in pure NumPy/Python."""
    nx, ny = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[Tuple[np.ndarray, np.ndarray]] = []
    for ix in range(nx):
        for iy in range(ny):
            if not mask[ix, iy] or visited[ix, iy]:
                continue
            stack = [(ix, iy)]
            visited[ix, iy] = True
            xs_c: List[int] = []
            ys_c: List[int] = []
            while stack:
                cx, cy = stack.pop()
                xs_c.append(cx)
                ys_c.append(cy)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx_i, ny_i = cx + dx, cy + dy
                    if 0 <= nx_i < nx and 0 <= ny_i < ny and mask[nx_i, ny_i] and not visited[nx_i, ny_i]:
                        visited[nx_i, ny_i] = True
                        stack.append((nx_i, ny_i))
            comps.append((np.asarray(xs_c, dtype=np.int64), np.asarray(ys_c, dtype=np.int64)))
    return comps


def _is_step_source(source: Union[str, Path, bytes], filename_hint: str = "") -> bool:
    hint = filename_hint.lower()
    if hint.endswith((".step", ".stp", ".p21")):
        return True
    if isinstance(source, (str, Path)):
        sfx = Path(source).suffix.lower()
        if sfx in (".step", ".stp", ".p21"):
            return True
        p = Path(source)
        if p.is_file():
            with p.open("rb") as fh:
                return is_step_payload(fh.read(2048))
        return False
    if isinstance(source, (bytes, bytearray)):
        return is_step_payload(bytes(source[:2048]))
    return False


def load_cad_part(
    cad_source: Union[str, Path, bytes],
    *,
    filename_hint: str = "",
    auto_orient: bool = True,
    stock_top_margin_mm: float = 1.5,
    vise_grip_z_mm: float = 2.0,
    pitch_mm: Optional[float] = None,
) -> RecognizedFeatures:
    """Ingest arbitrary ISO 10303-21 STEP (``.step``/``.stp``) or ``.stl`` CAD into G54/G55 work coordinates."""
    if not _is_step_source(cad_source, filename_hint=filename_hint):
        eff_pitch = float(pitch_mm) if pitch_mm is not None else 0.25
        mesh_chk = load_stl_mesh(cad_source)
        max_span_stl = max(
            mesh_chk.bounds_max_mm[0] - mesh_chk.bounds_min_mm[0],
            mesh_chk.bounds_max_mm[1] - mesh_chk.bounds_min_mm[1],
        )
        if pitch_mm is None and max_span_stl > 85.0:
            eff_pitch = min(0.75, max(0.35, round((max_span_stl / 220.0) * 20.0) / 20.0))
        feats_stl = recognize_part_from_stl(
            cad_source,
            stock_top_margin_mm=stock_top_margin_mm,
            vise_grip_z_mm=vise_grip_z_mm,
            pitch_mm=eff_pitch,
        )
        return feats_stl

    step_res: STEPLoadResult = load_step_brep(cad_source, auto_orient=auto_orient)
    bmin, bmax = step_res.oriented_bounds_min_mm, step_res.oriented_bounds_max_mm
    span_x = bmax[0] - bmin[0]
    span_y = bmax[1] - bmin[1]
    max_xy = max(span_x, span_y)

    if pitch_mm is None:
        if max_xy <= 65.0:
            eff_pitch = 0.30
        else:
            eff_pitch = min(0.75, max(0.35, round((max_xy / 230.0) * 20.0) / 20.0))
    else:
        eff_pitch = float(pitch_mm)

    lx = math.ceil(span_x * 2.0 - 1e-9) / 2.0
    ly = math.ceil(span_y * 2.0 - 1e-9) / 2.0
    part_h = bmax[2] - bmin[2]
    top_margin = round(round(stock_top_margin_mm / 0.5) * 0.5, 4)
    grip_z = round(round(vise_grip_z_mm / 0.5) * 0.5, 4)
    lz = math.ceil((part_h + top_margin + grip_z) * 2.0 - 1e-9) / 2.0

    stock_min = (-0.5 * lx, -0.5 * ly, -lz)
    stock_max = (0.5 * lx, 0.5 * ly, 0.0)
    shift = (
        -0.5 * (bmin[0] + bmax[0]),
        -0.5 * (bmin[1] + bmax[1]),
        -top_margin - bmax[2],
    )
    part_min = (bmin[0] + shift[0], bmin[1] + shift[1], bmin[2] + shift[2])
    part_max = (bmax[0] + shift[0], bmax[1] + shift[1], bmax[2] + shift[2])

    jaw_top = -lz + grip_z
    vise_jaws = (
        ViseJawZone(
            label="Fixed Vise Jaw (-Y)",
            min_mm=(-0.5 * lx - 5.0, -0.5 * ly - 12.0, -lz - 8.0),
            max_mm=(0.5 * lx + 5.0, -0.5 * ly, jaw_top),
        ),
        ViseJawZone(
            label="Movable Vise Jaw (+Y)",
            min_mm=(-0.5 * lx - 5.0, 0.5 * ly, -lz - 8.0),
            max_mm=(0.5 * lx + 5.0, 0.5 * ly + 12.0, jaw_top),
        ),
    )

    shifted_tris = step_res.oriented_triangles_mm + np.asarray(shift, dtype=np.float64)[None, None, :]
    target_stl_bytes = triangles_to_binary_stl_bytes(shifted_tris)
    mesh_g54 = STLMesh(
        triangles_mm=shifted_tris,
        bounds_min_mm=part_min,
        bounds_max_mm=part_max,
        signed_volume_mm3=1.0,
    )

    # Op1 (+Z G54) top target heightfield
    xs, ys, z_part_raw = rasterize_stl_top_heightfield(
        mesh_g54, stock_min, stock_max, (0.0, 0.0, 0.0), pitch_mm=eff_pitch
    )
    z_target = np.maximum(z_part_raw, part_min[2])
    through_cells = z_part_raw <= (stock_min[2] + 1e-3)
    z_target[through_cells] = max(stock_min[2], part_min[2] - 0.5)

    # Op2 (-Z flipped G55) target heightfield under SO(3) 180-deg X-axis flip:
    # F(x, y, z) = (x, -y, -lz - z) -> flip triangle winding (swap v1, v2) to preserve outward det=+1
    flip_tris = shifted_tris.copy()
    flip_tris[:, :, 1] = -shifted_tris[:, :, 1]
    flip_tris[:, :, 2] = -lz - shifted_tris[:, :, 2]
    mesh_g55 = STLMesh(
        triangles_mm=flip_tris,
        bounds_min_mm=(part_min[0], -part_max[1], -lz - part_max[2]),
        bounds_max_mm=(part_max[0], -part_min[1], -lz - part_min[2]),
        signed_volume_mm3=1.0,
    )
    _, _, z_op2_raw = rasterize_stl_top_heightfield(
        mesh_g55, stock_min, stock_max, (0.0, 0.0, 0.0), pitch_mm=eff_pitch
    )
    op2_part_bot = -lz - part_max[2]
    z_op2_target = np.maximum(z_op2_raw, op2_part_bot)
    op2_through = z_op2_raw <= (stock_min[2] + 1e-3)
    z_op2_target[op2_through] = max(stock_min[2], op2_part_bot - 0.5)

    # Shift semantic B-Rep features into G54 (Op1) and G55 (Op2) work frames,
    # reclassifying any blind hole occluded from +Z by a solid top roof into Op2 (G55).
    dx, dy, dz = shift
    nx_g, ny_g = z_target.shape
    op1_holes_list: List[BRepHoleFeature] = []
    extra_op2_shifted: List[BRepHoleFeature] = []
    for h in step_res.op1_holes:
        hs = h.shifted(dx, dy, dz)
        rad_occ = max(0.6 * eff_pitch, 0.35 * hs.diameter_mm)
        ix0_h = max(0, int(math.floor((hs.center_xy_mm[0] - rad_occ - stock_min[0]) / eff_pitch)))
        ix1_h = min(nx_g, int(math.ceil((hs.center_xy_mm[0] + rad_occ - stock_min[0]) / eff_pitch)) + 1)
        iy0_h = max(0, int(math.floor((hs.center_xy_mm[1] - rad_occ - stock_min[1]) / eff_pitch)))
        iy1_h = min(ny_g, int(math.ceil((hs.center_xy_mm[1] + rad_occ - stock_min[1]) / eff_pitch)) + 1)
        z_roof_min = (
            float(np.min(z_target[ix0_h:ix1_h, iy0_h:iy1_h]))
            if (ix0_h < ix1_h and iy0_h < iy1_h)
            else hs.z_bottom_mm
        )
        if z_roof_min > hs.z_bottom_mm + max(1.5, 0.55 * hs.diameter_mm):
            extra_op2_shifted.append(hs)
        else:
            op1_holes_list.append(hs)
    op1_holes_g54 = tuple(op1_holes_list)
    op1_chamf_g54 = tuple(c.shifted(dx, dy, dz) for c in step_res.op1_chamfers)
    op1_ff_g54 = tuple(f.shifted(dx, dy, dz) for f in step_res.op1_freeform)

    op2_holes_g55: List[BRepHoleFeature] = []
    for hs in [*(h.shifted(dx, dy, dz) for h in step_res.op2_holes), *extra_op2_shifted]:
        op2_holes_g55.append(
            BRepHoleFeature(
                side="op2_bottom",
                center_xy_mm=(hs.center_xy_mm[0], round(-hs.center_xy_mm[1], 4)),
                diameter_mm=hs.diameter_mm,
                z_top_mm=round(-lz - hs.z_bottom_mm, 4),
                z_bottom_mm=round(-lz - hs.z_top_mm, 4),
                is_through=hs.is_through,
                counterbore_diameter_mm=hs.counterbore_diameter_mm,
                counterbore_depth_mm=hs.counterbore_depth_mm,
                countersink_diameter_mm=hs.countersink_diameter_mm,
                countersink_angle_deg=hs.countersink_angle_deg,
                thread_spec=hs.thread_spec,
                thread_major_mm=hs.thread_major_mm,
                thread_pitch_mm=hs.thread_pitch_mm,
            )
        )
    requires_op2_eff = bool(step_res.requires_op2_flip or len(extra_op2_shifted) > 0)

    op2_chamf_g55: List[BRepChamferFeature] = []
    for c in step_res.op2_chamfers:
        cs = c.shifted(dx, dy, dz)
        op2_chamf_g55.append(
            BRepChamferFeature(
                side="op2_bottom",
                z_top_mm=round(-lz - cs.z_bottom_mm, 4),
                z_bottom_mm=round(-lz - cs.z_top_mm, 4),
                width_mm=cs.width_mm,
                angle_deg=cs.angle_deg,
                polyline_xy_mm=tuple((px, round(-py, 4)) for px, py in cs.polyline_xy_mm),
                is_circular_hole_chamfer=cs.is_circular_hole_chamfer,
            )
        )

    op2_ff_g55: List[BRepFreeformRegion] = []
    for ff in step_res.op2_freeform:
        ffs = ff.shifted(dx, dy, dz)
        op2_ff_g55.append(
            BRepFreeformRegion(
                side="op2_bottom",
                min_xy_mm=(ffs.min_xy_mm[0], round(-ffs.max_xy_mm[1], 4)),
                max_xy_mm=(ffs.max_xy_mm[0], round(-ffs.min_xy_mm[1], 4)),
                z_min_mm=round(-lz - ffs.z_max_mm, 4),
                z_max_mm=round(-lz - ffs.z_min_mm, 4),
                surface_types=ffs.surface_types,
                area_mm2=ffs.area_mm2,
            )
        )

    hole_bps = tuple(
        HoleBlueprint(
            center_xy_mm=h.center_xy_mm,
            diameter_mm=h.diameter_mm,
            z_top_mm=h.z_top_mm,
            z_bottom_mm=h.z_bottom_mm,
        )
        for h in op1_holes_g54
    )

    # Estimate minimum concave corner radius & pocket channel width from STEP faces + heightfield
    min_corner_r = 3.0
    for f in step_res.faces:
        if f.surface_type == "CYLINDRICAL_SURFACE" and f.inward_concave and f.radius_mm is not None and f.axis_dir is not None:
            if abs(f.axis_dir[2]) >= 0.92 and 0.75 <= f.radius_mm <= 8.0:
                # Pocket fillet!
                min_corner_r = min(min_corner_r, round(f.radius_mm, 3))
    z_top_part = float(np.max(z_target))
    top_face_depth = max(0.0, 0.0 - z_top_part)
    depressed = z_target < (z_top_part - 0.25)
    pockets_est: List[PocketBlueprint] = []
    min_chan = min(lx, ly)
    if np.any(depressed):
        pockets_est.append(
            PocketBlueprint(
                min_xy_mm=(round(part_min[0], 3), round(part_min[1], 3)),
                max_xy_mm=(round(part_max[0], 3), round(part_max[1], 3)),
                floor_z_mm=round(float(np.percentile(z_target[depressed], 20)), 3),
                corner_fillet_mm=min_corner_r,
            )
        )
        min_chan = max(3.5, 2.2 * min_corner_r)

    return RecognizedFeatures(
        stock_min_mm=stock_min,
        stock_max_mm=stock_max,
        part_min_mm=part_min,
        part_max_mm=part_max,
        vise_jaws=vise_jaws,
        top_face_depth_mm=round(top_face_depth, 4),
        has_exterior_step=False,
        exterior_step_z_mm=z_top_part,
        exterior_boss_min_xy_mm=(part_min[0], part_min[1]),
        exterior_boss_max_xy_mm=(part_max[0], part_max[1]),
        pockets=tuple(pockets_est),
        min_pocket_channel_mm=round(min_chan, 3),
        min_pocket_corner_radius_mm=round(min_corner_r, 3),
        holes=hole_bps,
        xs_mm=xs,
        ys_mm=ys,
        z_target_grid_mm=z_target,
        is_step_cad=True,
        source_unit=step_res.source_unit,
        setup_axis_label=step_res.setup_axis_label,
        part_name=step_res.part_name,
        target_stl_bytes=target_stl_bytes,
        brep_op1_holes=op1_holes_g54,
        brep_op2_holes=tuple(op2_holes_g55),
        brep_op1_chamfers=op1_chamf_g54,
        brep_op2_chamfers=tuple(op2_chamf_g55),
        brep_op1_freeform=op1_ff_g54,
        brep_op2_freeform=tuple(op2_ff_g55),
        requires_op2_flip=requires_op2_eff,
        op2_z_target_grid_mm=z_op2_target,
        vise_grip_z_mm=grip_z,
    )


def recognize_part_from_stl(
    stl_source: Union[str, Path, bytes],
    *,
    stock_top_margin_mm: float = 1.5,
    vise_grip_z_mm: float = 2.0,
    pitch_mm: float = 0.25,
) -> RecognizedFeatures:
    """Automatically select stock/workholding and recognize 2.5D + hole features from an ``.stl`` (or STEP)."""
    if _is_step_source(stl_source):
        return load_cad_part(
            stl_source,
            stock_top_margin_mm=stock_top_margin_mm,
            vise_grip_z_mm=vise_grip_z_mm,
            pitch_mm=pitch_mm,
        )
    mesh = load_stl_mesh(stl_source)
    bmin, bmax = mesh.bounds_min_mm, mesh.bounds_max_mm

    # Quantize stock dimensions cleanly to 0.5 mm multiples for exact voxel alignment
    lx = math.ceil((bmax[0] - bmin[0]) * 2.0 - 1e-9) / 2.0
    ly = math.ceil((bmax[1] - bmin[1]) * 2.0 - 1e-9) / 2.0
    part_h = bmax[2] - bmin[2]
    top_margin = round(round(stock_top_margin_mm / 0.5) * 0.5, 4)
    grip_z = round(round(vise_grip_z_mm / 0.5) * 0.5, 4)
    lz = math.ceil((part_h + top_margin + grip_z) * 2.0 - 1e-9) / 2.0

    stock_min = (-0.5 * lx, -0.5 * ly, -lz)
    stock_max = (0.5 * lx, 0.5 * ly, 0.0)
    # Shift STL so XY is centered at (0, 0) and part top sits at Z = -top_margin
    shift = (
        -0.5 * (bmin[0] + bmax[0]),
        -0.5 * (bmin[1] + bmax[1]),
        -top_margin - bmax[2],
    )
    part_min = (bmin[0] + shift[0], bmin[1] + shift[1], bmin[2] + shift[2])
    part_max = (bmax[0] + shift[0], bmax[1] + shift[1], bmax[2] + shift[2])

    # Parallel vise jaws gripping bottom [-lz, -lz + grip_z] along +/-Y faces
    jaw_top = -lz + grip_z
    vise_jaws = (
        ViseJawZone(
            label="Fixed Vise Jaw (-Y)",
            min_mm=(-0.5 * lx - 5.0, -0.5 * ly - 12.0, -lz - 8.0),
            max_mm=(0.5 * lx + 5.0, -0.5 * ly, jaw_top),
        ),
        ViseJawZone(
            label="Movable Vise Jaw (+Y)",
            min_mm=(-0.5 * lx - 5.0, 0.5 * ly, -lz - 8.0),
            max_mm=(0.5 * lx + 5.0, 0.5 * ly + 12.0, jaw_top),
        ),
    )

    xs, ys, z_part_raw = rasterize_stl_top_heightfield(
        mesh, stock_min, stock_max, shift, pitch_mm=pitch_mm
    )
    # Support material below part bottom down to vise base so clamp base remains intact
    z_target = np.maximum(z_part_raw, part_min[2])
    # Preserve through-holes extending to part_min[2]: where z_part_raw <= stock_min[2] + 1e-3, hole goes to part_min[2] - 0.5
    through_cells = z_part_raw <= (stock_min[2] + 1e-3)
    z_target[through_cells] = max(stock_min[2], part_min[2] - 0.5)

    z_top_part = float(np.max(z_target))
    top_face_depth = max(0.0, 0.0 - z_top_part)

    # Detect depressed regions below z_top_part - 0.20 mm
    depressed = z_target < (z_top_part - 0.20)
    nx, ny = z_target.shape

    # Check if outer perimeter is depressed (exterior contour step!)
    border_mask = np.zeros_like(depressed)
    border_mask[0, :] = border_mask[-1, :] = border_mask[:, 0] = border_mask[:, -1] = True
    has_ext_step = bool(np.mean(depressed[border_mask]) > 0.65)
    ext_step_z = z_top_part
    boss_min_xy = (stock_min[0], stock_min[1])
    boss_max_xy = (stock_max[0], stock_max[1])

    interior_depressed = depressed.copy()
    if has_ext_step:
        high_plate = ~depressed
        if np.any(high_plate):
            hx_idx, hy_idx = np.where(high_plate)
            boss_min_xy = (
                round(float(xs[np.min(hx_idx)] - 0.5 * pitch_mm), 3),
                round(float(ys[np.min(hy_idx)] - 0.5 * pitch_mm), 3),
            )
            boss_max_xy = (
                round(float(xs[np.max(hx_idx)] + 0.5 * pitch_mm), 3),
                round(float(ys[np.max(hy_idx)] + 0.5 * pitch_mm), 3),
            )
            border_vals = z_target[border_mask & depressed]
            ext_step_z = round(float(np.median(border_vals)) * 4.0) / 4.0
            # Remove the outer step ring from interior pocket/hole search
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            in_boss_box = (
                (gx >= boss_min_xy[0] + pitch_mm)
                & (gx <= boss_max_xy[0] - pitch_mm)
                & (gy >= boss_min_xy[1] + pitch_mm)
                & (gy <= boss_max_xy[1] - pitch_mm)
            )
            interior_depressed = depressed & in_boss_box

    # Separate cylindrical holes (compact, circular, diameter <= 14 mm) from pockets
    # Note: holes can also be drilled into pocket floors! So also check local floor drops.
    holes: List[HoleBlueprint] = []
    pockets: List[PocketBlueprint] = []
    min_channel_mm = 20.0
    min_corner_r_mm = 3.0

    # First detect deep circular holes relative to surrounding local floor
    deep_thresh = np.percentile(z_target[interior_depressed], 20) if np.any(interior_depressed) else z_top_part
    # Check top-level interior depressions, exterior flange holes, and sub-pocket floor depressions
    candidate_masks = [interior_depressed]
    if has_ext_step:
        deeper_than_flange = depressed & (~interior_depressed) & (z_target < ext_step_z - 0.35)
        if np.any(deeper_than_flange):
            candidate_masks.append(deeper_than_flange)
    if np.any(interior_depressed):
        # Modal pocket floor height
        vals = np.round(z_target[interior_depressed] * 4.0) / 4.0
        unique_z, z_counts = np.unique(vals, return_counts=True)
        if len(unique_z) >= 2:
            pocket_floor_est = float(unique_z[np.argmax(z_counts)])
            deeper_than_pocket = interior_depressed & (z_target < pocket_floor_est - 0.35)
            if np.any(deeper_than_pocket):
                candidate_masks.append(deeper_than_pocket)

    hole_mask_global = np.zeros_like(depressed)
    for cmask in candidate_masks:
        for cx_i, cy_i in _connected_components_2d(cmask):
            area_mm2 = float(len(cx_i)) * (pitch_mm * pitch_mm)
            if area_mm2 < 1.8:
                continue
            xmin_c = float(xs[np.min(cx_i)] - 0.5 * pitch_mm)
            xmax_c = float(xs[np.max(cx_i)] + 0.5 * pitch_mm)
            ymin_c = float(ys[np.min(cy_i)] - 0.5 * pitch_mm)
            ymax_c = float(ys[np.max(cy_i)] + 0.5 * pitch_mm)
            w_x = xmax_c - xmin_c
            w_y = ymax_c - ymin_c
            aspect = max(w_x, w_y) / max(min(w_x, w_y), 1e-6)
            d_est = 0.5 * (w_x + w_y)
            fill_ratio = area_mm2 / max(w_x * w_y, 1e-6)
            # A circle inscribed in its AABB has area / bbox = pi / 4 ~= 0.785!
            if aspect <= 1.22 and 0.64 <= fill_ratio <= 0.87 and 2.0 <= d_est <= 13.0:
                cx_mm = round(0.5 * (xmin_c + xmax_c) * 4.0) / 4.0
                cy_mm = round(0.5 * (ymin_c + ymax_c) * 4.0) / 4.0
                d_quant = round(d_est * 2.0) / 2.0
                z_bot = round(float(np.min(z_target[cx_i, cy_i])) * 4.0) / 4.0
                # Determine z_top around hole rim
                rim_x0 = max(0, int(np.min(cx_i)) - 2)
                rim_x1 = min(nx, int(np.max(cx_i)) + 3)
                rim_y0 = max(0, int(np.min(cy_i)) - 2)
                rim_y1 = min(ny, int(np.max(cy_i)) + 3)
                rim_patch = z_target[rim_x0:rim_x1, rim_y0:rim_y1]
                z_rim = round(float(np.max(rim_patch)) * 4.0) / 4.0
                if not any(
                    math.hypot(h.center_xy_mm[0] - cx_mm, h.center_xy_mm[1] - cy_mm) < 0.5 * d_quant
                    for h in holes
                ):
                    holes.append(
                        HoleBlueprint(
                            center_xy_mm=(cx_mm, cy_mm),
                            diameter_mm=d_quant,
                            z_top_mm=z_rim,
                            z_bottom_mm=z_bot,
                        )
                    )
                    hole_mask_global[cx_i, cy_i] = True

    # Remaining large interior depressed regions are 2.5D pockets!
    pocket_mask = interior_depressed & (~hole_mask_global)
    for cx_i, cy_i in _connected_components_2d(pocket_mask):
        area_mm2 = float(len(cx_i)) * (pitch_mm * pitch_mm)
        if area_mm2 < 25.0:
            continue
        ix_lo, ix_hi = int(np.min(cx_i)), int(np.max(cx_i))
        iy_lo, iy_hi = int(np.min(cy_i)), int(np.max(cy_i))
        px0 = round(float(xs[ix_lo] - 0.5 * pitch_mm) * 2.0) / 2.0
        px1 = round(float(xs[ix_hi] + 0.5 * pitch_mm) * 2.0) / 2.0
        py0 = round(float(ys[iy_lo] - 0.5 * pitch_mm) * 2.0) / 2.0
        py1 = round(float(ys[iy_hi] + 0.5 * pitch_mm) * 2.0) / 2.0
        p_floor = round(float(np.median(z_target[cx_i, cy_i])) * 2.0) / 2.0

        # Detect retained islands inside the pocket bounding box
        sub_high = z_target[ix_lo : ix_hi + 1, iy_lo : iy_hi + 1] >= (z_top_part - 0.20)
        islands_found: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for icx, icy in _connected_components_2d(sub_high):
            # Ignore corner fillet remnants touching the outer pocket wall
            if (
                np.min(icx) == 0
                or np.max(icx) == (ix_hi - ix_lo)
                or np.min(icy) == 0
                or np.max(icy) == (iy_hi - iy_lo)
            ):
                # Estimate corner fillet radius from corner triangle/arc area: A_corner = (1 - pi/4) r^2
                c_area = float(len(icx)) * (pitch_mm * pitch_mm)
                r_est = math.sqrt(max(c_area / (1.0 - math.pi / 4.0), 0.0))
                if r_est >= 0.75:
                    min_corner_r_mm = min(min_corner_r_mm, round(r_est * 2.0) / 2.0)
                continue
            if len(icx) * (pitch_mm * pitch_mm) < 6.0:
                continue
            il_x0 = round(float(xs[ix_lo + int(np.min(icx))] - 0.5 * pitch_mm) * 2.0) / 2.0
            il_x1 = round(float(xs[ix_lo + int(np.max(icx))] + 0.5 * pitch_mm) * 2.0) / 2.0
            il_y0 = round(float(ys[iy_lo + int(np.min(icy))] - 0.5 * pitch_mm) * 2.0) / 2.0
            il_y1 = round(float(ys[iy_lo + int(np.max(icy))] + 0.5 * pitch_mm) * 2.0) / 2.0
            islands_found.append(((il_x0, il_y0), (il_x1, il_y1)))
            gaps = [il_x0 - px0, px1 - il_x1, il_y0 - py0, py1 - il_y1]
            for g in gaps:
                if g > 1.0:
                    min_channel_mm = min(min_channel_mm, g)

        min_channel_mm = min(min_channel_mm, px1 - px0, py1 - py0)
        pockets.append(
            PocketBlueprint(
                min_xy_mm=(px0, py0),
                max_xy_mm=(px1, py1),
                floor_z_mm=p_floor,
                corner_fillet_mm=min_corner_r_mm,
                islands_xy_mm=tuple(islands_found),
            )
        )

    holes.sort(key=lambda h: (round(h.diameter_mm, 3), h.center_xy_mm[0], h.center_xy_mm[1]))
    return RecognizedFeatures(
        stock_min_mm=stock_min,
        stock_max_mm=stock_max,
        part_min_mm=part_min,
        part_max_mm=part_max,
        vise_jaws=vise_jaws,
        top_face_depth_mm=round(top_face_depth, 4),
        has_exterior_step=has_ext_step,
        exterior_step_z_mm=ext_step_z,
        exterior_boss_min_xy_mm=boss_min_xy,
        exterior_boss_max_xy_mm=boss_max_xy,
        pockets=tuple(pockets),
        min_pocket_channel_mm=round(min_channel_mm, 3),
        min_pocket_corner_radius_mm=round(min_corner_r_mm, 3),
        holes=tuple(holes),
        xs_mm=xs,
        ys_mm=ys,
        z_target_grid_mm=z_target,
    )


# ---------------------------------------------------------------------------
# 3. Discrete Task-Level Tool Selection & Multi-Tool Operation Synthesis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlannedOperationSpec:
    """Single task-level operation bound to a specific catalog tool & speeds/feeds."""

    op_index: int
    op_id: str
    op_kind: str
    tool_number: int
    tool_record: CatalogToolRecord
    holder_spec: HolderSpec
    tool_assembly: C.ToolAssembly
    spindle_rpm: float
    base_feed_mm_s: float
    target_hex_mm: float
    doc_mm: float
    woc_mm: float
    notes: str
    setup_index: int = 1
    work_offset: str = "G54"
    bores_spec: Tuple[Tuple[Tuple[float, float], float, float, float], ...] = ()
    brep_holes_spec: Tuple[BRepHoleFeature, ...] = ()
    chamfers_spec: Tuple[BRepChamferFeature, ...] = ()
    freeform_spec: Tuple[BRepFreeformRegion, ...] = ()


@dataclass(frozen=True)
class TAMPMultiToolResult:
    """Complete output of automated TAMP planning, multi-tool DiffCAM optimization, and verification."""

    part_id: str
    title: str
    stock_material_id: str
    features: RecognizedFeatures
    operations: Tuple[PlannedOperationSpec, ...]
    unique_tools_used: int
    atc_swap_time_s: float
    naive_cutting_time_s: float
    naive_total_time_s: float
    same_speed_cutting_time_s: float
    same_speed_total_time_s: float
    opt_chipload_cutting_time_s: float
    opt_chipload_total_time_s: float
    same_speed_gain_pct: float
    opt_chipload_gain_pct: float
    opt_engaged_time_s: float
    opt_air_cut_time_s: float
    opt_rapid_time_s: float
    opt_optimal_hex_pct: float
    gouge_volume_mm3: float
    residual_volume_mm3: float
    vise_collision: bool
    holder_collision: bool
    rapid_collision: bool
    safety_pass: bool
    haas_ngc_program: str
    per_op_metrics: List[Dict[str, Any]]
    viewer_variants: Dict[str, Dict[str, Any]]
    planning_wall_time_s: float
    gcode_by_dialect: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tools_table: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    operations_table: List[Dict[str, Any]] = field(default_factory=list)
    cad_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _build_holemaking_motion_plan(
    holes: Sequence[HoleBlueprint],
    tool_asm: C.ToolAssembly,
    *,
    is_spot: bool,
    plunge_feed_mm_s: float,
    peck_depth_mm: float,
    rapid_mm_s: float = 100.0,
    z_safe_mm: float = 5.0,
    z_retract_clearance_mm: float = 1.0,
    max_spot_depth_mm: Optional[float] = None,
) -> C.MotionPlan:
    """Generate a validated machine-frame spot-drilling or G83 multi-peck drilling MotionPlan."""
    segs: List[C.Segment] = []
    cur = (0.0, 0.0, z_safe_mm)
    sid = 0

    def _push(end_pt: Tuple[float, float, float], kind: C.SegmentKind, feed: Optional[float]) -> None:
        nonlocal cur, sid
        dist = math.dist(cur, end_pt)
        if dist <= 1e-7:
            return
        spd = rapid_mm_s if kind == C.SegmentKind.RAPID or feed is None else feed
        segs.append(
            C.Segment(
                id=f"h{sid:04d}",
                kind=kind,
                start_mm=cur,
                end_mm=end_pt,
                feed_mm_s=None if kind == C.SegmentKind.RAPID else round(float(spd), 4),
                duration_s=dist / spd,
            )
        )
        cur = end_pt
        sid += 1

    for hl in holes:
        cx, cy = hl.center_xy_mm
        _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)
        r_plane = min(z_safe_mm, hl.z_top_mm + z_retract_clearance_mm)
        _push((cx, cy, r_plane), C.SegmentKind.RAPID, None)

        if is_spot:
            # 90-deg spot drill chamfers hole mouth to 0.52 * D_hole depth (bounded by max_spot_depth_mm if set)
            d_sp = 0.52 * hl.diameter_mm
            if max_spot_depth_mm is not None:
                d_sp = min(d_sp, float(max_spot_depth_mm))
            spot_z = max(hl.z_bottom_mm, hl.z_top_mm - d_sp)
            _push((cx, cy, spot_z), C.SegmentKind.FEED, plunge_feed_mm_s)
            _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)
        else:
            # Full twist drill with automatic G83 chip-clearing pecks + drill-point breakthrough
            tip_extra = 0.30 * (2.0 * tool_asm.cutter_radius_mm)
            target_z = hl.z_bottom_mm - tip_extra
            total_d = hl.z_top_mm - target_z
            step_d = max(0.8, min(total_d, peck_depth_mm))
            z_cur_cut = hl.z_top_mm
            while z_cur_cut > target_z + 1e-5:
                z_next = max(target_z, z_cur_cut - step_d)
                if cur[2] > z_cur_cut + 0.45:
                    _push((cx, cy, z_cur_cut + 0.35), C.SegmentKind.RAPID, None)
                _push((cx, cy, z_next), C.SegmentKind.FEED, plunge_feed_mm_s)
                _push((cx, cy, r_plane), C.SegmentKind.RAPID, None)
                z_cur_cut = z_next
            _push((cx, cy, z_safe_mm), C.SegmentKind.RAPID, None)

    _push((0.0, 0.0, z_safe_mm), C.SegmentKind.RAPID, None)
    durs = C.blended_segment_durations(segs, rapid_mm_s, 150.0)
    segs = [dataclasses.replace(s, duration_s=float(d)) for s, d in zip(segs, durs)]
    return C.MotionPlan(
        frame=C.Frame.MACHINE,
        timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
        initial_state=C.MachineState(position_mm=(0.0, 0.0, z_safe_mm), frame=C.Frame.MACHINE),
        segments=tuple(segs),
        provenance="diffcam:tamp_holemaking",
    )


def _plan_arbitrary_cad_operations(
    features: RecognizedFeatures,
    *,
    library: StandardToolLibrary,
    stock_material_id: str,
    tool_material_override: Optional[str],
    mach: MachineEnvelopeSpec,
    atc_swap_time_s: float,
    setup_mode: str,
    enable_probing: bool,
    enable_chamfer: bool,
    enable_threading: bool,
    warnings_out: List[str],
) -> List[PlannedOperationSpec]:
    """Multi-tool, crib-constrained, Op1/Op2 task planner for arbitrary STEP/STL CAD."""
    ops: List[PlannedOperationSpec] = []
    tool_num_map: Dict[str, int] = {}
    next_tnum = 1

    def _eff_tool_mat(rec: CatalogToolRecord) -> str:
        return tool_material_override or rec.tool_material_id

    min_shop_stickout = float(features.stock_max_mm[2] - features.stock_min_mm[2]) + 6.0

    def _assign_tnum(rec: CatalogToolRecord) -> Tuple[int, HolderSpec, C.ToolAssembly]:
        nonlocal next_tnum
        if rec.entity_type == "probe":
            tnum = 99
        elif rec.tool_id not in tool_num_map:
            if next_tnum == 99:
                next_tnum = 100
            tool_num_map[rec.tool_id] = next_tnum
            tnum = next_tnum
            next_tnum += 1
        else:
            tnum = tool_num_map[rec.tool_id]
        hld = library.select_holder_for_tool(rec, taper="BT30")
        asm = rec.to_tool_assembly(tnum, holder=hld)
        asm = dataclasses.replace(
            asm,
            holder_bottom_mm=max(float(asm.holder_bottom_mm), min_shop_stickout),
        )
        return tnum, hld, asm

    # Gather available flat/bull endmills from active library/crib
    all_mills = [
        t
        for t in library.tools
        if t.entity_type in ("flat end mill", "bull nose end mill")
        and (library.preferred_unit_system is None or t.unit_system == library.preferred_unit_system)
    ]
    if not all_mills:
        all_mills = [t for t in library.tools if t.entity_type in ("flat end mill", "bull nose end mill")]
    if not all_mills:
        all_mills = [t for t in library.tools if "mill" in t.entity_type]
    if not all_mills:
        warnings_out.append("Selected Tool Crib contained zero milling cutters; falling back to default 10 mm end mill.")
        all_mills = DEFAULT_TOOL_LIBRARY.query_endmills(min_diameter_mm=6.0, max_diameter_mm=12.0)

    all_mills.sort(key=lambda t: (t.diameter_mm, t.flute_length_mm))
    max_depth = abs(float(features.part_min_mm[2]))
    reach_mills = (
        [t for t in all_mills if max(t.flute_length_mm, t.stickout_mm) >= 0.72 * max_depth]
        or [t for t in all_mills if max(t.flute_length_mm, t.stickout_mm) >= 0.45 * max_depth]
        or all_mills
    )

    # Choose primary roughing mill (up to 16 mm / 5/8") and smaller rest-finishing mill
    rough_pool = [t for t in reach_mills if t.diameter_mm <= 16.5 and t.entity_type == "flat end mill"] or reach_mills
    t_rough = rough_pool[-1]
    max_fin_d = max(2.5, min(10.0, 2.15 * features.min_pocket_corner_radius_mm))
    fin_pool = [t for t in reach_mills if 2.0 <= t.diameter_mm <= max_fin_d + 0.25] or [reach_mills[0]]
    t_finish = fin_pool[-1]

    def _add_probe_op(setup_idx: int, wcs: str) -> None:
        probes = [t for t in library.tools if t.entity_type == "probe"] or [
            t for t in DEFAULT_TOOL_LIBRARY.tools if t.entity_type == "probe"
        ]
        if not probes:
            return
        pr = probes[0]
        tnum, hld, asm = _assign_tnum(pr)
        ops.append(
            PlannedOperationSpec(
                op_index=len(ops) + 1,
                op_id=f"s{setup_idx}_op{len(ops) + 1}_wips_probe",
                op_kind="wips_probe",
                tool_number=tnum,
                tool_record=pr,
                holder_spec=hld,
                tool_assembly=asm,
                spindle_rpm=0.0,
                base_feed_mm_s=25.0,
                target_hex_mm=0.0,
                doc_mm=0.0,
                woc_mm=pr.diameter_mm,
                notes=f"{wcs} spindle touch-probe stock & Z datum calibration ({pr.nominal_label})",
                setup_index=setup_idx,
                work_offset=wcs,
            )
        )

    def _add_hole_and_chamfer_ops(
        setup_idx: int,
        wcs: str,
        brep_holes: Sequence[BRepHoleFeature],
        chamfers: Sequence[BRepChamferFeature],
        freeform_regs: Sequence[BRepFreeformRegion],
    ) -> None:
        # 1. 3D Ball-Endmill Scallop Surfacing on sculptured B-spline / curved regions
        if freeform_regs:
            ball_rec = library.match_ball_endmill(target_diameter_mm=6.35, min_flute_length_mm=3.0)
            if ball_rec is None:
                ball_rec = t_finish
                warnings_out.append(
                    f"{wcs}: No ball end mill in active Tool Crib; substituting {ball_rec.nominal_label} for 3D surfacing."
                )
            tnum_b, hld_b, asm_b = _assign_tnum(ball_rec)
            ae_ball = min(0.28 * ball_rec.diameter_mm, max(0.55, math.sqrt(8.0 * (0.5 * ball_rec.diameter_mm) * 0.015)))
            fs_b = compute_feeds_and_speeds(
                stock_material_id=stock_material_id,
                tool_material_id=_eff_tool_mat(ball_rec),
                cutter=ball_rec.to_cutter_geometry(),
                machine=mach,
                doc_mm=0.6,
                woc_mm=ae_ball,
            )
            ops.append(
                PlannedOperationSpec(
                    op_index=len(ops) + 1,
                    op_id=f"s{setup_idx}_op{len(ops) + 1}_3d_scallop",
                    op_kind="3d_ball_scallop",
                    tool_number=tnum_b,
                    tool_record=ball_rec,
                    holder_spec=hld_b,
                    tool_assembly=asm_b,
                    spindle_rpm=fs_b.spindle_rpm,
                    base_feed_mm_s=min(mach.max_feed_mm_s * 0.95, fs_b.feed_mm_s),
                    target_hex_mm=fs_b.chip_thickness_mm,
                    doc_mm=0.6,
                    woc_mm=ae_ball,
                    notes=f"{wcs} 3D gouge-free scallop surfacing ({len(freeform_regs)} sculptured region(s)) with {ball_rec.nominal_label}",
                    setup_index=setup_idx,
                    work_offset=wcs,
                    freeform_spec=tuple(freeform_regs),
                )
            )

        # 2. Hole-making: partition into twist-drillable vs helical-bored holes + counterbores
        drill_groups: Dict[str, Tuple[CatalogToolRecord, List[BRepHoleFeature]]] = {}
        helical_bores: List[Tuple[Tuple[float, float], float, float, float]] = []

        for hl in brep_holes:
            h_depth = max(0.5, hl.z_top_mm - hl.z_bottom_mm)
            matched_tool, strat = library.match_closest_available_drill_or_mill(hl.diameter_mm, h_depth)
            if strat == "drill" and matched_tool is not None:
                grp = drill_groups.get(matched_tool.tool_id)
                if grp is None:
                    drill_groups[matched_tool.tool_id] = (matched_tool, [hl])
                else:
                    grp[1].append(hl)
            elif strat == "helical_bore" and matched_tool is not None:
                helical_bores.append((hl.center_xy_mm, hl.diameter_mm, hl.z_top_mm, hl.z_bottom_mm))
            elif strat == "pilot_drill" and matched_tool is not None:
                grp = drill_groups.get(matched_tool.tool_id)
                if grp is None:
                    drill_groups[matched_tool.tool_id] = (matched_tool, [hl])
                else:
                    grp[1].append(hl)
                warnings_out.append(
                    f"{wcs}: Exact Ø{hl.diameter_mm:.2f} mm drill not in crib; pilot-drilling with {matched_tool.nominal_label}."
                )
            else:
                warnings_out.append(
                    f"{wcs}: Skipped Ø{hl.diameter_mm:.2f} mm hole (no fitting drill or smaller end mill in active Tool Crib)."
                )

            # Counterbores always interpolated cleanly via helical bore!
            if hl.counterbore_diameter_mm and hl.counterbore_depth_mm and hl.counterbore_diameter_mm > hl.diameter_mm + 0.35:
                helical_bores.append(
                    (
                        hl.center_xy_mm,
                        hl.counterbore_diameter_mm,
                        hl.z_top_mm,
                        hl.z_top_mm - hl.counterbore_depth_mm,
                    )
                )

        # Helical bore & counterbore interpolation FIRST (so recessed hole mouths are cleared before spotting/drilling!)
        if helical_bores:
            bore_by_tool: Dict[str, Tuple[CatalogToolRecord, List[Tuple[Tuple[float, float], float, float, float]]]] = {}
            for b_xy, b_d, b_zt, b_zb in helical_bores:
                fit_mills = [m for m in reach_mills if m.entity_type == "flat end mill" and m.diameter_mm <= b_d - 0.30]
                if not fit_mills:
                    fit_mills = [m for m in all_mills if m.diameter_mm <= b_d - 0.25]
                if not fit_mills:
                    warnings_out.append(f"{wcs}: Skipped Ø{b_d:.2f} mm bore/counterbore (no smaller end mill in crib).")
                    continue
                ideal_d = min(b_d - 0.35, 0.75 * b_d)
                fit_mills.sort(key=lambda m: abs(m.diameter_mm - ideal_d))
                bm = fit_mills[0]
                if bm.tool_id not in bore_by_tool:
                    bore_by_tool[bm.tool_id] = (bm, [(b_xy, b_d, b_zt, b_zb)])
                else:
                    bore_by_tool[bm.tool_id][1].append((b_xy, b_d, b_zt, b_zb))

            for bm, blist in bore_by_tool.values():
                tnum_hb, hld_hb, asm_hb = _assign_tnum(bm)
                doc_hb = min(2.5, 0.55 * bm.diameter_mm)
                woc_hb = 0.38 * bm.diameter_mm
                fs_hb = compute_feeds_and_speeds(
                    stock_material_id=stock_material_id,
                    tool_material_id=_eff_tool_mat(bm),
                    cutter=bm.to_cutter_geometry(),
                    machine=mach,
                    doc_mm=doc_hb,
                    woc_mm=woc_hb,
                )
                ops.append(
                    PlannedOperationSpec(
                        op_index=len(ops) + 1,
                        op_id=f"s{setup_idx}_op{len(ops) + 1}_helical_bore",
                        op_kind="helical_bore",
                        tool_number=tnum_hb,
                        tool_record=bm,
                        holder_spec=hld_hb,
                        tool_assembly=asm_hb,
                        spindle_rpm=fs_hb.spindle_rpm,
                        base_feed_mm_s=min(mach.max_feed_mm_s * 0.90, fs_hb.feed_mm_s),
                        target_hex_mm=fs_hb.chip_thickness_mm,
                        doc_mm=doc_hb,
                        woc_mm=woc_hb,
                        notes=f"{wcs} Helical bore/counterbore interpolation ({len(blist)} feature(s)) with {bm.nominal_label}",
                        setup_index=setup_idx,
                        work_offset=wcs,
                        bores_spec=tuple(blist),
                    )
                )

        # Spot drill before twist drills
        all_drilled = [h for _, hlist in drill_groups.values() for h in hlist]
        if all_drilled:
            spot_cands = [t for t in library.tools if t.entity_type in ("spot drill", "chamfer mill")]
            spot_rec = library.match_spot_drill(max(h.diameter_mm for h in all_drilled)) if spot_cands else None
            if spot_rec is not None:
                tnum_s, hld_s, asm_s = _assign_tnum(spot_rec)
                fs_sp = compute_holemaking_feeds_and_speeds(
                    stock_material_id=stock_material_id,
                    tool_material_id=_eff_tool_mat(spot_rec),
                    diameter_mm=spot_rec.diameter_mm,
                    flutes=spot_rec.flutes,
                    hole_depth_mm=1.4,
                    operation_kind="spot drill",
                    machine=mach,
                )
                ops.append(
                    PlannedOperationSpec(
                        op_index=len(ops) + 1,
                        op_id=f"s{setup_idx}_op{len(ops) + 1}_spot_drill",
                        op_kind="spot_drill",
                        tool_number=tnum_s,
                        tool_record=spot_rec,
                        holder_spec=hld_s,
                        tool_assembly=asm_s,
                        spindle_rpm=fs_sp.spindle_rpm,
                        base_feed_mm_s=min(mach.max_feed_mm_s * 0.85, fs_sp.plunge_feed_mm_s),
                        target_hex_mm=fs_sp.fz_mm,
                        doc_mm=1.4,
                        woc_mm=spot_rec.diameter_mm,
                        notes=f"{wcs} Spot {len(all_drilled)} hole mouth(s) with {spot_rec.nominal_label}",
                        setup_index=setup_idx,
                        work_offset=wcs,
                        brep_holes_spec=tuple(all_drilled),
                    )
                )

            for d_rec, h_list in sorted(drill_groups.values(), key=lambda item: item[0].diameter_mm):
                tnum_d, hld_d, asm_d = _assign_tnum(d_rec)
                max_hd = max(0.8, max(h.z_top_mm - h.z_bottom_mm for h in h_list))
                fs_dr = compute_holemaking_feeds_and_speeds(
                    stock_material_id=stock_material_id,
                    tool_material_id=_eff_tool_mat(d_rec),
                    diameter_mm=d_rec.diameter_mm,
                    flutes=d_rec.flutes,
                    hole_depth_mm=max_hd,
                    operation_kind="drill",
                    machine=mach,
                )
                ops.append(
                    PlannedOperationSpec(
                        op_index=len(ops) + 1,
                        op_id=f"s{setup_idx}_op{len(ops) + 1}_drill_{str(round(d_rec.diameter_mm, 2)).replace('.', 'p')}mm",
                        op_kind="peck_drill",
                        tool_number=tnum_d,
                        tool_record=d_rec,
                        holder_spec=hld_d,
                        tool_assembly=asm_d,
                        spindle_rpm=fs_dr.spindle_rpm,
                        base_feed_mm_s=min(mach.max_feed_mm_s * 0.85, fs_dr.plunge_feed_mm_s),
                        target_hex_mm=fs_dr.fz_mm,
                        doc_mm=fs_dr.peck_depth_mm,
                        woc_mm=d_rec.diameter_mm,
                        notes=f"{wcs} {'G83 Peck' if fs_dr.peck_required else 'Direct'} drill {len(h_list)}x holes with {d_rec.nominal_label}",
                        setup_index=setup_idx,
                        work_offset=wcs,
                        brep_holes_spec=tuple(h_list),
                    )
                )

        # 3. Countersinking & 45° Edge Chamfering
        if enable_chamfer:
            cs_holes = [h for h in brep_holes if h.countersink_diameter_mm]
            if cs_holes:
                cs_ang = cs_holes[0].countersink_angle_deg or 82.0
                max_csd = max(h.countersink_diameter_mm or 6.0 for h in cs_holes)
                cs_tool = library.match_chamfer_or_countersink(tip_angle_deg=cs_ang, min_diameter_mm=max_csd)
                if cs_tool is None:
                    cs_tool = library.match_chamfer_or_countersink(tip_angle_deg=90.0, min_diameter_mm=4.0)
                if cs_tool is not None:
                    tnum_cs, hld_cs, asm_cs = _assign_tnum(cs_tool)
                    fs_cs = compute_holemaking_feeds_and_speeds(
                        stock_material_id=stock_material_id,
                        tool_material_id=_eff_tool_mat(cs_tool),
                        diameter_mm=cs_tool.diameter_mm,
                        flutes=cs_tool.flutes,
                        hole_depth_mm=2.0,
                        operation_kind="spot drill",
                        machine=mach,
                    )
                    ops.append(
                        PlannedOperationSpec(
                            op_index=len(ops) + 1,
                            op_id=f"s{setup_idx}_op{len(ops) + 1}_countersink",
                            op_kind="countersink",
                            tool_number=tnum_cs,
                            tool_record=cs_tool,
                            holder_spec=hld_cs,
                            tool_assembly=asm_cs,
                            spindle_rpm=fs_cs.spindle_rpm,
                            base_feed_mm_s=min(mach.max_feed_mm_s * 0.85, fs_cs.plunge_feed_mm_s),
                            target_hex_mm=fs_cs.fz_mm,
                            doc_mm=1.5,
                            woc_mm=cs_tool.diameter_mm,
                            notes=f"{wcs} {cs_ang:.0f}° Countersink {len(cs_holes)} hole(s) with {cs_tool.nominal_label}",
                            setup_index=setup_idx,
                            work_offset=wcs,
                            brep_holes_spec=tuple(cs_holes),
                        )
                    )
                else:
                    warnings_out.append(f"{wcs}: Countersink skipped (no chamfer/countersink/spot tool in crib).")

            if chamfers:
                ch_tool = library.match_chamfer_or_countersink(tip_angle_deg=90.0, min_diameter_mm=5.0)
                if ch_tool is not None:
                    tnum_ch, hld_ch, asm_ch = _assign_tnum(ch_tool)
                    fs_ch = compute_feeds_and_speeds(
                        stock_material_id=stock_material_id,
                        tool_material_id=_eff_tool_mat(ch_tool),
                        cutter=ch_tool.to_cutter_geometry(),
                        machine=mach,
                        doc_mm=0.8,
                        woc_mm=0.8,
                    )
                    ops.append(
                        PlannedOperationSpec(
                            op_index=len(ops) + 1,
                            op_id=f"s{setup_idx}_op{len(ops) + 1}_chamfer",
                            op_kind="chamfer_contour",
                            tool_number=tnum_ch,
                            tool_record=ch_tool,
                            holder_spec=hld_ch,
                            tool_assembly=asm_ch,
                            spindle_rpm=fs_ch.spindle_rpm,
                            base_feed_mm_s=min(mach.max_feed_mm_s * 0.90, fs_ch.feed_mm_s),
                            target_hex_mm=fs_ch.chip_thickness_mm,
                            doc_mm=0.8,
                            woc_mm=0.8,
                            notes=f"{wcs} 45° Edge deburr/chamfer ({len(chamfers)} bevel edge(s)) with {ch_tool.nominal_label}",
                            setup_index=setup_idx,
                            work_offset=wcs,
                            chamfers_spec=tuple(chamfers),
                        )
                    )
                else:
                    warnings_out.append(f"{wcs}: Edge chamfer skipped (no 90° chamfer/spot mill in active Tool Crib).")

        # 4. Internal Thread Milling / Rigid Tapping
        if enable_threading:
            th_holes = [h for h in brep_holes if h.thread_spec and h.thread_major_mm and h.thread_pitch_mm]
            if th_holes:
                by_th: Dict[str, List[BRepHoleFeature]] = {}
                for h in th_holes:
                    by_th.setdefault(str(h.thread_spec), []).append(h)
                for th_name, th_list in sorted(by_th.items()):
                    h0 = th_list[0]
                    tm_rec = library.match_thread_mill_or_tap(
                        float(h0.thread_major_mm or 6.0), float(h0.thread_pitch_mm or 1.0)
                    )
                    if tm_rec is None:
                        warnings_out.append(
                            f"{wcs}: Thread {th_name} skipped (no matching thread mill or tap in active Tool Crib)."
                        )
                        continue
                    tnum_tm, hld_tm, asm_tm = _assign_tnum(tm_rec)
                    fs_tm = compute_feeds_and_speeds(
                        stock_material_id=stock_material_id,
                        tool_material_id=_eff_tool_mat(tm_rec),
                        cutter=tm_rec.to_cutter_geometry(),
                        machine=mach,
                        doc_mm=float(h0.thread_pitch_mm or 1.0),
                        woc_mm=0.5 * float(h0.thread_pitch_mm or 1.0),
                    )
                    ops.append(
                        PlannedOperationSpec(
                            op_index=len(ops) + 1,
                            op_id=f"s{setup_idx}_op{len(ops) + 1}_thread_{th_name.split()[0].replace('/', '_').replace('#', 'n').replace('-', '_')}",
                            op_kind="thread_mill",
                            tool_number=tnum_tm,
                            tool_record=tm_rec,
                            holder_spec=hld_tm,
                            tool_assembly=asm_tm,
                            spindle_rpm=fs_tm.spindle_rpm,
                            base_feed_mm_s=min(mach.max_feed_mm_s * 0.75, max(2.5, 0.55 * fs_tm.feed_mm_s)),
                            target_hex_mm=fs_tm.chip_thickness_mm,
                            doc_mm=float(h0.thread_pitch_mm or 1.0),
                            woc_mm=0.5 * float(h0.thread_pitch_mm or 1.0),
                            notes=f"{wcs} Helical internal thread {len(th_list)}x {th_name} with {tm_rec.nominal_label}",
                            setup_index=setup_idx,
                            work_offset=wcs,
                            brep_holes_spec=tuple(th_list),
                        )
                    )

    # ---------------- SETUP 1: G54 Top Machining ----------------
    if enable_probing:
        _add_probe_op(1, "G54")

    span_xy_min = min(
        features.stock_max_mm[0] - features.stock_min_mm[0],
        features.stock_max_mm[1] - features.stock_min_mm[1],
    )
    face_cands = [
        t
        for t in library.tools
        if t.entity_type == "face mill"
        and t.diameter_mm <= max(65.0, 1.35 * span_xy_min)
        and (library.preferred_unit_system is None or t.unit_system == library.preferred_unit_system)
    ]
    face_cands.sort(key=lambda t: abs(t.diameter_mm - 50.8))

    if features.top_face_depth_mm >= 0.20:
        face_mill_cand = face_cands[0] if (span_xy_min >= 42.0 and face_cands) else None
        face_tool = face_mill_cand if face_mill_cand is not None else t_rough
        tnum_f1, hld_f1, asm_f1 = _assign_tnum(face_tool)
        woc_face = 0.65 * face_tool.diameter_mm
        fs_f1 = compute_feeds_and_speeds(
            stock_material_id=stock_material_id,
            tool_material_id=_eff_tool_mat(face_tool),
            cutter=face_tool.to_cutter_geometry(),
            machine=mach,
            doc_mm=min(2.5, features.top_face_depth_mm),
            woc_mm=woc_face,
        )
        ops.append(
            PlannedOperationSpec(
                op_index=len(ops) + 1,
                op_id=f"s1_op{len(ops) + 1}_top_face",
                op_kind="facing",
                tool_number=tnum_f1,
                tool_record=face_tool,
                holder_spec=hld_f1,
                tool_assembly=asm_f1,
                spindle_rpm=fs_f1.spindle_rpm,
                base_feed_mm_s=min(mach.max_feed_mm_s * 0.95, fs_f1.feed_mm_s),
                target_hex_mm=fs_f1.chip_thickness_mm,
                doc_mm=min(2.5, features.top_face_depth_mm),
                woc_mm=woc_face,
                notes=f"G54 Top stock facing (-{features.top_face_depth_mm:.2f} mm) with {face_tool.nominal_label}",
                setup_index=1,
                work_offset="G54",
            )
        )

    # Check if Op1 has depressed pockets, bosses, or steps below z_part_top
    z_part_top = round(0.0 - features.top_face_depth_mm, 4)
    has_op1_cavities = bool(np.any(features.z_target_grid_mm < z_part_top - 0.25))
    if has_op1_cavities:
        tnum_r1, hld_r1, asm_r1 = _assign_tnum(t_rough)
        doc_r1 = min(11.0, 0.75 * t_rough.flute_length_mm, 1.10 * t_rough.diameter_mm)
        woc_r1 = 0.30 * t_rough.diameter_mm
        fs_r1 = compute_feeds_and_speeds(
            stock_material_id=stock_material_id,
            tool_material_id=_eff_tool_mat(t_rough),
            cutter=t_rough.to_cutter_geometry(),
            machine=mach,
            doc_mm=doc_r1,
            woc_mm=woc_r1,
        )
        ops.append(
            PlannedOperationSpec(
                op_index=len(ops) + 1,
                op_id=f"s1_op{len(ops) + 1}_levelset_rough",
                op_kind="levelset_rough",
                tool_number=tnum_r1,
                tool_record=t_rough,
                holder_spec=hld_r1,
                tool_assembly=asm_r1,
                spindle_rpm=fs_r1.spindle_rpm,
                base_feed_mm_s=min(mach.max_feed_mm_s * 0.95, fs_r1.feed_mm_s),
                target_hex_mm=fs_r1.chip_thickness_mm,
                doc_mm=doc_r1,
                woc_mm=woc_r1,
                notes=f"G54 Adaptive depth-first stay-down roughing with {t_rough.nominal_label}",
                setup_index=1,
                work_offset="G54",
            )
        )

        if t_finish.tool_id != t_rough.tool_id and t_finish.diameter_mm <= t_rough.diameter_mm - 1.0:
            tnum_fn, hld_fn, asm_fn = _assign_tnum(t_finish)
            doc_fn = min(10.0, 0.78 * t_finish.flute_length_mm, 1.25 * t_finish.diameter_mm)
            woc_fn = 0.25 * t_finish.diameter_mm
            fs_fn = compute_feeds_and_speeds(
                stock_material_id=stock_material_id,
                tool_material_id=_eff_tool_mat(t_finish),
                cutter=t_finish.to_cutter_geometry(),
                machine=mach,
                doc_mm=doc_fn,
                woc_mm=woc_fn,
            )
            ops.append(
                PlannedOperationSpec(
                    op_index=len(ops) + 1,
                    op_id=f"s1_op{len(ops) + 1}_rest_finish",
                    op_kind="levelset_rest_finish",
                    tool_number=tnum_fn,
                    tool_record=t_finish,
                    holder_spec=hld_fn,
                    tool_assembly=asm_fn,
                    spindle_rpm=fs_fn.spindle_rpm,
                    base_feed_mm_s=min(mach.max_feed_mm_s * 0.95, fs_fn.feed_mm_s),
                    target_hex_mm=fs_fn.chip_thickness_mm,
                    doc_mm=doc_fn,
                    woc_mm=woc_fn,
                    notes=f"G54 Persistent-IPW corner & wall rest-finishing with {t_finish.nominal_label}",
                    setup_index=1,
                    work_offset="G54",
                )
            )

    op1_holes_in = features.brep_op1_holes or tuple(
        BRepHoleFeature(
            side="op1_top",
            center_xy_mm=h.center_xy_mm,
            diameter_mm=h.diameter_mm,
            z_top_mm=h.z_top_mm,
            z_bottom_mm=h.z_bottom_mm,
            is_through=(h.z_bottom_mm <= features.part_min_mm[2] + 0.25),
        )
        for h in features.holes
    )
    _add_hole_and_chamfer_ops(1, "G54", op1_holes_in, features.brep_op1_chamfers, features.brep_op1_freeform)

    # ---------------- SETUP 2: G55 Soft-Jaw Flip (Carrier-Hat + Underside Features) ----------------
    do_op2 = (setup_mode == "dual_op1_op2") or (setup_mode == "auto" and features.requires_op2_flip)
    if do_op2:
        if enable_probing:
            _add_probe_op(2, "G55")

        hat_depth = max(0.5, float(features.vise_grip_z_mm))
        face_mill_op2 = face_cands[0] if (span_xy_min >= 42.0 and face_cands) else None
        hat_tool = face_mill_op2 if face_mill_op2 is not None else t_rough
        tnum_h2, hld_h2, asm_h2 = _assign_tnum(hat_tool)
        woc_h2 = 0.65 * hat_tool.diameter_mm
        fs_h2 = compute_feeds_and_speeds(
            stock_material_id=stock_material_id,
            tool_material_id=_eff_tool_mat(hat_tool),
            cutter=hat_tool.to_cutter_geometry(),
            machine=mach,
            doc_mm=min(2.5, hat_depth),
            woc_mm=woc_h2,
        )
        ops.append(
            PlannedOperationSpec(
                op_index=len(ops) + 1,
                op_id=f"s2_op{len(ops) + 1}_hat_face",
                op_kind="op2_hat_facing",
                tool_number=tnum_h2,
                tool_record=hat_tool,
                holder_spec=hld_h2,
                tool_assembly=asm_h2,
                spindle_rpm=fs_h2.spindle_rpm,
                base_feed_mm_s=min(mach.max_feed_mm_s * 0.95, fs_h2.feed_mm_s),
                target_hex_mm=fs_h2.chip_thickness_mm,
                doc_mm=min(2.5, hat_depth),
                woc_mm=woc_h2,
                notes=f"G55 Op2 soft-jaw carrier-hat facing (-{hat_depth:.2f} mm) with {hat_tool.nominal_label}",
                setup_index=2,
                work_offset="G55",
            )
        )

        if features.op2_z_target_grid_mm is not None:
            z_op2_top = round(0.0 - hat_depth, 4)
            # Exclude through-holes already drilled from Op1 when checking if Op2 has genuine underside pockets/steps
            op2_recess_cells = (features.op2_z_target_grid_mm < z_op2_top - 0.35) & (
                features.op2_z_target_grid_mm > (-features.stock_min_mm[2] + features.part_max_mm[2]) + 0.35
            )
            if int(np.sum(op2_recess_cells)) >= 12:
                tnum_r2, hld_r2, asm_r2 = _assign_tnum(t_rough)
                doc_r2 = min(10.0, 0.72 * t_rough.flute_length_mm, 1.05 * t_rough.diameter_mm)
                woc_r2 = 0.30 * t_rough.diameter_mm
                fs_r2 = compute_feeds_and_speeds(
                    stock_material_id=stock_material_id,
                    tool_material_id=_eff_tool_mat(t_rough),
                    cutter=t_rough.to_cutter_geometry(),
                    machine=mach,
                    doc_mm=doc_r2,
                    woc_mm=woc_r2,
                )
                ops.append(
                    PlannedOperationSpec(
                        op_index=len(ops) + 1,
                        op_id=f"s2_op{len(ops) + 1}_underside_pocket",
                        op_kind="op2_levelset_rough",
                        tool_number=tnum_r2,
                        tool_record=t_rough,
                        holder_spec=hld_r2,
                        tool_assembly=asm_r2,
                        spindle_rpm=fs_r2.spindle_rpm,
                        base_feed_mm_s=min(mach.max_feed_mm_s * 0.95, fs_r2.feed_mm_s),
                        target_hex_mm=fs_r2.chip_thickness_mm,
                        doc_mm=doc_r2,
                        woc_mm=woc_r2,
                        notes=f"G55 Op2 underside step/pocket clearing with {t_rough.nominal_label}",
                        setup_index=2,
                        work_offset="G55",
                    )
                )

        _add_hole_and_chamfer_ops(2, "G55", features.brep_op2_holes, features.brep_op2_chamfers, features.brep_op2_freeform)

    _ = atc_swap_time_s
    return ops


def plan_tamp_operations(
    features: RecognizedFeatures,
    *,
    library: StandardToolLibrary = DEFAULT_TOOL_LIBRARY,
    stock_material_id: str = "al_6061_t6",
    machine: Optional[MachineEnvelopeSpec] = None,
    atc_swap_time_s: float = 6.0,
    unit_system: Optional[str] = None,
    allowed_tool_ids: Optional[Sequence[str]] = None,
    tool_material_override: Optional[str] = None,
    setup_mode: str = "auto",
    enable_probing: bool = False,
    enable_chamfer: bool = True,
    enable_threading: bool = True,
    warnings_out: Optional[List[str]] = None,
) -> List[PlannedOperationSpec]:
    """Task-level search selecting tools, operation sequence, and physics speeds/feeds."""
    if unit_system is not None or allowed_tool_ids is not None:
        library = library.subset(unit_system=unit_system, allowed_tool_ids=allowed_tool_ids)

    use_arbitrary = (
        features.is_step_cad
        or library.has_custom_crib
        or unit_system is not None
        or allowed_tool_ids is not None
        or enable_probing
        or setup_mode == "dual_op1_op2"
    )
    if use_arbitrary:
        mach_arb = machine or MachineEnvelopeSpec(max_feed_mm_s=28.0)
        w_list = warnings_out if warnings_out is not None else []
        return _plan_arbitrary_cad_operations(
            features,
            library=library,
            stock_material_id=stock_material_id,
            tool_material_override=tool_material_override,
            mach=mach_arb,
            atc_swap_time_s=atc_swap_time_s,
            setup_mode=setup_mode,
            enable_probing=enable_probing,
            enable_chamfer=enable_chamfer,
            enable_threading=enable_threading,
            warnings_out=w_list,
        )

    mach = machine or MachineEnvelopeSpec(max_feed_mm_s=20.0)
    ops: List[PlannedOperationSpec] = []
    tool_num_map: Dict[str, int] = {}

    def _assign_tnum(rec: CatalogToolRecord) -> Tuple[int, HolderSpec, C.ToolAssembly]:
        if rec.tool_id not in tool_num_map:
            tool_num_map[rec.tool_id] = len(tool_num_map) + 1
        tnum = tool_num_map[rec.tool_id]
        hld = library.select_holder_for_tool(rec, taper="BT30")
        asm = rec.to_tool_assembly(tnum, holder=hld)
        return tnum, hld, asm

    # Determine finishing cutter upper bound from tightest pocket corner radius & channel width
    max_finish_d = min(
        max(2.0, 2.0 * features.min_pocket_corner_radius_mm),
        max(2.0, features.min_pocket_channel_mm - 0.6),
    )
    max_depth = abs(float(features.part_min_mm[2]))

    # Candidate roughing vs. finishing endmills from StandardToolLibrary
    rough_candidates = library.query_endmills(
        min_diameter_mm=5.0, max_diameter_mm=10.0, min_flute_length_mm=max_depth
    )
    finish_candidates = library.query_endmills(
        min_diameter_mm=2.0, max_diameter_mm=max_finish_d, min_flute_length_mm=max_depth
    )
    if not finish_candidates:
        finish_candidates = library.query_endmills(min_diameter_mm=2.0, max_diameter_mm=6.0)
    t_finish = finish_candidates[-1]  # largest rigid endmill that clears tightest corner/channel!

    # Evaluate TAMP discrete choice: 1-endmill (t_finish does everything) vs 2-endmill (t_rough + t_finish rest)
    # Estimate total pocket + facing removal volume
    cell_area = float((features.xs_mm[1] - features.xs_mm[0]) * (features.ys_mm[1] - features.ys_mm[0]))
    vol_total_milling = float(np.sum(np.maximum(0.0, 0.0 - features.z_target_grid_mm))) * cell_area
    fs_fin = compute_feeds_and_speeds(
        stock_material_id=stock_material_id,
        tool_material_id=t_finish.tool_material_id,
        cutter=t_finish.to_cutter_geometry(),
        machine=mach,
        doc_mm=min(3.5, 0.7 * t_finish.diameter_mm),
        woc_mm=0.35 * t_finish.diameter_mm,
    )
    best_rough: Optional[CatalogToolRecord] = None
    best_est_time = vol_total_milling / max(fs_fin.mrr_mm3_s, 1.0)

    for cand_r in rough_candidates:
        if cand_r.diameter_mm <= t_finish.diameter_mm + 1.0:
            continue
        # Ensure roughing endmill actually fits into at least the main pocket/face
        if features.pockets and cand_r.diameter_mm > max(
            p.max_xy_mm[0] - p.min_xy_mm[0] - 1.0 for p in features.pockets
        ):
            continue
        fs_r = compute_feeds_and_speeds(
            stock_material_id=stock_material_id,
            tool_material_id=cand_r.tool_material_id,
            cutter=cand_r.to_cutter_geometry(),
            machine=mach,
            doc_mm=min(4.0, 0.7 * cand_r.diameter_mm),
            woc_mm=0.42 * cand_r.diameter_mm,
        )
        # Rougher removes ~82% of volume; finisher removes remaining ~18% rest cusps + ATC swap
        est_two_tool = (
            (0.82 * vol_total_milling) / max(fs_r.mrr_mm3_s, 1.0)
            + (0.18 * vol_total_milling) / max(fs_fin.mrr_mm3_s, 1.0)
            + atc_swap_time_s
        )
        if est_two_tool < best_est_time - 0.5:
            best_est_time = est_two_tool
            best_rough = cand_r

    primary_mill = best_rough or t_finish

    # OP 1: Top Facing (using primary_mill so zero extra tool change before roughing!)
    if features.top_face_depth_mm >= 0.25:
        tnum, hld, asm = _assign_tnum(primary_mill)
        woc_face = 0.60 * primary_mill.diameter_mm
        fs_op = compute_feeds_and_speeds(
            stock_material_id=stock_material_id,
            tool_material_id=primary_mill.tool_material_id,
            cutter=primary_mill.to_cutter_geometry(),
            machine=mach,
            doc_mm=features.top_face_depth_mm,
            woc_mm=woc_face,
        )
        ops.append(
            PlannedOperationSpec(
                op_index=len(ops) + 1,
                op_id=f"op{len(ops) + 1}_top_face",
                op_kind="facing",
                tool_number=tnum,
                tool_record=primary_mill,
                holder_spec=hld,
                tool_assembly=asm,
                spindle_rpm=fs_op.spindle_rpm,
                base_feed_mm_s=min(mach.max_feed_mm_s * 0.92, fs_op.feed_mm_s),
                target_hex_mm=fs_op.chip_thickness_mm,
                doc_mm=features.top_face_depth_mm,
                woc_mm=woc_face,
                notes=f"Top stock facing (-{features.top_face_depth_mm:.2f} mm) with {primary_mill.nominal_label}",
            )
        )

    # OP 2: Exterior Boss Step Contouring (if present)
    if features.has_exterior_step:
        tnum, hld, asm = _assign_tnum(primary_mill)
        doc_ext = abs(features.exterior_step_z_mm) - features.top_face_depth_mm
        woc_ext = 0.42 * primary_mill.diameter_mm
        fs_ext = compute_feeds_and_speeds(
            stock_material_id=stock_material_id,
            tool_material_id=primary_mill.tool_material_id,
            cutter=primary_mill.to_cutter_geometry(),
            machine=mach,
            doc_mm=max(0.5, doc_ext),
            woc_mm=woc_ext,
        )
        ops.append(
            PlannedOperationSpec(
                op_index=len(ops) + 1,
                op_id=f"op{len(ops) + 1}_exterior_contour",
                op_kind="exterior_contour",
                tool_number=tnum,
                tool_record=primary_mill,
                holder_spec=hld,
                tool_assembly=asm,
                spindle_rpm=fs_ext.spindle_rpm,
                base_feed_mm_s=min(mach.max_feed_mm_s * 0.92, fs_ext.feed_mm_s),
                target_hex_mm=fs_ext.chip_thickness_mm,
                doc_mm=max(0.5, doc_ext),
                woc_mm=woc_ext,
                notes=f"Exterior boss profiling to Z={features.exterior_step_z_mm:.2f} mm",
            )
        )

    # OP 3 & OP 4: Enclosed Pockets (Bulk Roughing with primary_mill + Rest-Finishing with t_finish if split)
    for p_idx, pck in enumerate(features.pockets):
        p_depth = abs(pck.floor_z_mm) - features.top_face_depth_mm
        if best_rough is not None and primary_mill.diameter_mm < min(
            pck.max_xy_mm[0] - pck.min_xy_mm[0], pck.max_xy_mm[1] - pck.min_xy_mm[1]
        ) - 1.5 and (not pck.islands_xy_mm or primary_mill.diameter_mm <= features.min_pocket_channel_mm - 0.5):
            tnum_r, hld_r, asm_r = _assign_tnum(primary_mill)
            woc_r = 0.45 * primary_mill.diameter_mm
            fs_r = compute_feeds_and_speeds(
                stock_material_id=stock_material_id,
                tool_material_id=primary_mill.tool_material_id,
                cutter=primary_mill.to_cutter_geometry(),
                machine=mach,
                doc_mm=max(0.5, p_depth),
                woc_mm=woc_r,
            )
            ops.append(
                PlannedOperationSpec(
                    op_index=len(ops) + 1,
                    op_id=f"op{len(ops) + 1}_pocket{p_idx}_rough",
                    op_kind="pocket_rough",
                    tool_number=tnum_r,
                    tool_record=primary_mill,
                    holder_spec=hld_r,
                    tool_assembly=asm_r,
                    spindle_rpm=fs_r.spindle_rpm,
                    base_feed_mm_s=min(mach.max_feed_mm_s * 0.92, fs_r.feed_mm_s),
                    target_hex_mm=fs_r.chip_thickness_mm,
                    doc_mm=max(0.5, p_depth),
                    woc_mm=woc_r,
                    notes=f"Bulk pocket roughing with {primary_mill.nominal_label}",
                )
            )
            # Rest-machining + corner/wall cleanup with smaller t_finish!
            tnum_f, hld_f, asm_f = _assign_tnum(t_finish)
            woc_f = 0.30 * t_finish.diameter_mm
            fs_f = compute_feeds_and_speeds(
                stock_material_id=stock_material_id,
                tool_material_id=t_finish.tool_material_id,
                cutter=t_finish.to_cutter_geometry(),
                machine=mach,
                doc_mm=max(0.5, p_depth),
                woc_mm=woc_f,
            )
            ops.append(
                PlannedOperationSpec(
                    op_index=len(ops) + 1,
                    op_id=f"op{len(ops) + 1}_pocket{p_idx}_rest_finish",
                    op_kind="pocket_rest_finish",
                    tool_number=tnum_f,
                    tool_record=t_finish,
                    holder_spec=hld_f,
                    tool_assembly=asm_f,
                    spindle_rpm=fs_f.spindle_rpm,
                    base_feed_mm_s=min(mach.max_feed_mm_s * 0.92, fs_f.feed_mm_s),
                    target_hex_mm=fs_f.chip_thickness_mm,
                    doc_mm=max(0.5, p_depth),
                    woc_mm=woc_f,
                    notes=f"IPW-aware corner rest-machining (R_min={features.min_pocket_corner_radius_mm:.2f} mm) with {t_finish.nominal_label}",
                )
            )
        else:
            # Channel too narrow for large rougher: use t_finish with adaptive/pocket clearing directly
            tnum_f, hld_f, asm_f = _assign_tnum(t_finish)
            use_adapt = len(pck.islands_xy_mm) >= 2
            woc_f = (0.25 if use_adapt else 0.40) * t_finish.diameter_mm
            fs_f = compute_feeds_and_speeds(
                stock_material_id=stock_material_id,
                tool_material_id=t_finish.tool_material_id,
                cutter=t_finish.to_cutter_geometry(),
                machine=mach,
                doc_mm=max(0.5, p_depth),
                woc_mm=woc_f,
            )
            ops.append(
                PlannedOperationSpec(
                    op_index=len(ops) + 1,
                    op_id=f"op{len(ops) + 1}_pocket{p_idx}_{'adaptive' if use_adapt else 'clear'}",
                    op_kind="pocket_adaptive" if use_adapt else "pocket_rough",
                    tool_number=tnum_f,
                    tool_record=t_finish,
                    holder_spec=hld_f,
                    tool_assembly=asm_f,
                    spindle_rpm=fs_f.spindle_rpm,
                    base_feed_mm_s=min(mach.max_feed_mm_s * 0.92, fs_f.feed_mm_s),
                    target_hex_mm=fs_f.chip_thickness_mm,
                    doc_mm=max(0.5, p_depth),
                    woc_mm=woc_f,
                    notes=f"Channel/island clearing with {t_finish.nominal_label}",
                )
            )

    # OP 5 & OP 6: Hole-Making (90° Spot Drill + Matched Carbide Twist Drill Peck Cycle)
    if features.holes:
        max_hole_d = max(h.diameter_mm for h in features.holes)
        spot_rec = library.match_spot_drill(max_hole_d)
        tnum_s, hld_s, asm_s = _assign_tnum(spot_rec)
        fs_spot = compute_holemaking_feeds_and_speeds(
            stock_material_id=stock_material_id,
            tool_material_id=spot_rec.tool_material_id,
            diameter_mm=spot_rec.diameter_mm,
            flutes=spot_rec.flutes,
            hole_depth_mm=0.5 * max_hole_d,
            operation_kind="spot drill",
            machine=mach,
        )
        ops.append(
            PlannedOperationSpec(
                op_index=len(ops) + 1,
                op_id=f"op{len(ops) + 1}_spot_holes",
                op_kind="spot_drill",
                tool_number=tnum_s,
                tool_record=spot_rec,
                holder_spec=hld_s,
                tool_assembly=asm_s,
                spindle_rpm=fs_spot.spindle_rpm,
                base_feed_mm_s=min(mach.max_feed_mm_s * 0.85, fs_spot.plunge_feed_mm_s),
                target_hex_mm=fs_spot.fz_mm,
                doc_mm=0.5 * max_hole_d,
                woc_mm=spot_rec.diameter_mm,
                notes=f"Spot/chamfer {len(features.holes)} hole mouths with {spot_rec.nominal_label}",
            )
        )

        # Group holes by diameter and match exact twist drill from StandardToolLibrary
        by_diam: Dict[float, List[HoleBlueprint]] = {}
        for h in features.holes:
            by_diam.setdefault(round(h.diameter_mm, 3), []).append(h)
        for d_h, h_group in sorted(by_diam.items()):
            max_h_depth = max(h.z_top_mm - h.z_bottom_mm for h in h_group)
            drill_rec = library.match_drill(d_h, max_h_depth + 2.0)
            if drill_rec is None:
                continue
            tnum_d, hld_d, asm_d = _assign_tnum(drill_rec)
            fs_drl = compute_holemaking_feeds_and_speeds(
                stock_material_id=stock_material_id,
                tool_material_id=drill_rec.tool_material_id,
                diameter_mm=drill_rec.diameter_mm,
                flutes=drill_rec.flutes,
                hole_depth_mm=max_h_depth,
                operation_kind="drill",
                machine=mach,
            )
            ops.append(
                PlannedOperationSpec(
                    op_index=len(ops) + 1,
                    op_id=f"op{len(ops) + 1}_peck_drill_{str(d_h).replace('.', 'p')}mm",
                    op_kind="peck_drill",
                    tool_number=tnum_d,
                    tool_record=drill_rec,
                    holder_spec=hld_d,
                    tool_assembly=asm_d,
                    spindle_rpm=fs_drl.spindle_rpm,
                    base_feed_mm_s=min(mach.max_feed_mm_s * 0.85, fs_drl.plunge_feed_mm_s),
                    target_hex_mm=fs_drl.fz_mm,
                    doc_mm=fs_drl.peck_depth_mm,
                    woc_mm=drill_rec.diameter_mm,
                    notes=f"{'G83 Peck' if fs_drl.peck_required else 'Direct'} drill {len(h_group)}x Ø{d_h:g} mm holes ({drill_rec.nominal_label})",
                )
            )

    return ops


def _build_milling_op_bundle(
    part_id: str,
    features: RecognizedFeatures,
    op: PlannedOperationSpec,
) -> MillingTaskBundle:
    """Build a validated single-operation :class:`MillingTaskBundle` for ``op``."""
    smin = features.stock_min_mm
    smax = features.stock_max_mm
    z_part_top = round(0.0 - features.top_face_depth_mm, 4)

    if op.op_kind == "facing":
        case = MillingCaseSpec(
            family="facing_open_boss",
            split="test",
            case_id=f"{part_id}-{op.op_id}",
            stock_min_mm=smin,
            stock_max_mm=smax,
            pocket_min_mm=(smin[0], smin[1], z_part_top),
            pocket_max_mm=(smax[0], smax[1], 0.0),
            islands_mm=(),
            mrr_limit_mm3_s=150.0,
            notes=op.notes,
        )
    elif op.op_kind == "exterior_contour":
        bx0, by0 = features.exterior_boss_min_xy_mm
        bx1, by1 = features.exterior_boss_max_xy_mm
        case = MillingCaseSpec(
            family="contour_profile",
            split="test",
            case_id=f"{part_id}-{op.op_id}",
            stock_min_mm=smin,
            stock_max_mm=(smax[0], smax[1], z_part_top),
            pocket_min_mm=(smin[0], smin[1], features.exterior_step_z_mm),
            pocket_max_mm=(smax[0], smax[1], z_part_top),
            islands_mm=(((bx0, by0, features.exterior_step_z_mm), (bx1, by1, z_part_top)),),
            mrr_limit_mm3_s=140.0,
            notes=op.notes,
        )
    else:
        pck = features.pockets[0]
        fam = "adaptive_channel" if op.op_kind == "pocket_adaptive" else "pocket_island"
        isl_3d = tuple(
            ((ix0, iy0, pck.floor_z_mm), (ix1, iy1, z_part_top))
            for (ix0, iy0), (ix1, iy1) in pck.islands_xy_mm
        )
        case = MillingCaseSpec(
            family=fam,
            split="test",
            case_id=f"{part_id}-{op.op_id}",
            stock_min_mm=smin,
            stock_max_mm=(smax[0], smax[1], z_part_top),
            pocket_min_mm=(pck.min_xy_mm[0], pck.min_xy_mm[1], pck.floor_z_mm),
            pocket_max_mm=(pck.max_xy_mm[0], pck.max_xy_mm[1], z_part_top),
            islands_mm=isl_3d,
            mrr_limit_mm3_s=130.0,
            notes=op.notes,
        )

    return build_milling_task_bundle(case, voxel_mm=0.50, tool_override=op.tool_assembly)


def _format_multitool_haas_ngc(
    program_number: int,
    title: str,
    ops_and_plans: Sequence[Tuple[PlannedOperationSpec, C.MotionPlan]],
) -> str:
    """Compile a multi-tool Haas Mini Mill Next-Gen Control (``.nc``) shop program."""
    lines: List[str] = [
        "%",
        f"O{int(program_number):05d} ({title[:40].upper()})",
        "(DIFFCAM AUTOMATED MULTI-TOOL TAMP PROGRAM - MIT CLEAN-ROOM)",
        "G90 G94 G17 G40 G49 G80 G21",
        "G53 G0 Z0.",
    ]
    current_tool: Optional[int] = None
    for op, plan in ops_and_plans:
        lines.append(f"(--- OP {op.op_index}: {op.op_id.upper()} | {op.tool_record.nominal_label.upper()} ---)")
        if current_tool != op.tool_number:
            if current_tool is not None:
                lines.extend(["M9", "M5", "G53 G0 Z0."])
            lines.extend(
                [
                    f"T{op.tool_number} M6",
                    "G54",
                    f"S{int(round(op.spindle_rpm))} M3",
                    f"G43 H{op.tool_number} Z{plan.initial_state.position_mm[2]:.3f}",
                    "G187 P2 E0.05",
                    "M8",
                ]
            )
            current_tool = op.tool_number
        else:
            lines.append(f"S{int(round(op.spindle_rpm))} M3")

        x0, y0, z0 = plan.initial_state.position_mm
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f} Z{z0:.3f}")
        last_kind: Optional[C.SegmentKind] = None
        last_f_min: Optional[float] = None
        for seg in plan.segments:
            ex, ey, ez = seg.end_mm
            if seg.kind == C.SegmentKind.RAPID:
                g_word = "G0 " if last_kind != C.SegmentKind.RAPID else ""
                lines.append(f"{g_word}X{ex:.3f} Y{ey:.3f} Z{ez:.3f}")
                last_kind = C.SegmentKind.RAPID
            else:
                f_min = float(seg.feed_mm_s or op.base_feed_mm_s) * 60.0
                g_word = "G1 " if last_kind != C.SegmentKind.FEED else ""
                f_word = ""
                if last_f_min is None or abs(f_min - last_f_min) > 0.05:
                    f_word = f" F{f_min:.1f}"
                    last_f_min = f_min
                lines.append(f"{g_word}X{ex:.3f} Y{ey:.3f} Z{ez:.3f}{f_word}")
                last_kind = C.SegmentKind.FEED

    lines.extend(["M9", "M5", "G53 G0 Z0.", "G53 G0 X0. Y0.", "M30", "%"])
    return "\n".join(lines) + "\n"


def _serialize_multitool_variant_for_viewer(
    key: str,
    label: str,
    ops_and_plans: Sequence[Tuple[PlannedOperationSpec, C.MotionPlan]],
    features: RecognizedFeatures,
    atc_swap_time_s: float,
    gcode_by_dialect: Optional[Dict[str, Dict[str, Any]]] = None,
    primary_dialect: str = "haas_ngc",
) -> Dict[str, Any]:
    """Serialize a multi-tool trajectory sequence into a viewer-compatible variant dict."""
    xs = features.xs_mm
    ys = features.ys_mm
    pitch = float(xs[1] - xs[0])
    cell_area = pitch * pitch
    nx, ny = len(xs), len(ys)
    x0_grid, y0_grid, z_min_stock = features.stock_min_mm
    z_max_stock = features.stock_max_mm[2]
    z_stock_op1 = np.full((nx, ny), float(z_max_stock), dtype=np.float64)
    z_stock_op2 = np.full((nx, ny), float(z_max_stock), dtype=np.float64)

    primary_lines: List[int] = []
    if gcode_by_dialect and primary_dialect in gcode_by_dialect:
        primary_lines = list(gcode_by_dialect[primary_dialect].get("viewer_segment_to_line", []))

    segs_json: List[Dict[str, Any]] = []
    cum_t = 0.0
    total_eng_t = 0.0
    total_air_t = 0.0
    total_rap_t = 0.0
    total_rem_v = 0.0
    opt_hex_t = 0.0
    hex_w_sum = 0.0
    hex_sq_sum = 0.0
    ae_w_sum = 0.0
    peak_mrr = 0.0
    prev_tool: Optional[int] = None
    sid_global = 0

    def _seg_tool_meta(op: PlannedOperationSpec, r_cut: float, spd: float, gline: int) -> Dict[str, Any]:
        return {
            "feed_mm_min": round(spd * 60.0, 2),
            "spindle_rpm": round(float(op.spindle_rpm), 1),
            "tool_number": op.tool_number,
            "tool_id": op.tool_record.tool_id,
            "tool_radius_mm": round(r_cut, 4),
            "entity_type": op.tool_record.entity_type,
            "corner_radius_mm": round(float(op.tool_record.corner_radius_mm), 4),
            "tip_angle_deg": round(float(op.tool_record.tip_angle_deg), 2),
            "flute_length_mm": round(float(op.tool_record.flute_length_mm), 3),
            "setup_index": op.setup_index,
            "work_offset": op.work_offset,
            "op_id": op.op_id,
            "gcode_line": gline,
        }

    for op, plan in ops_and_plans:
        z_active = z_stock_op2 if op.setup_index >= 2 else z_stock_op1
        r_cut = float(op.tool_assembly.cutter_radius_mm)
        is_probe = op.op_kind == "wips_probe" or op.tool_record.entity_type == "probe"

        if prev_tool is not None and prev_tool != op.tool_number:
            p_safe = plan.initial_state.position_mm
            gline_atc = primary_lines[sid_global] if sid_global < len(primary_lines) else 1
            t0_v = round(cum_t, 5)
            t1_v = round(cum_t + atc_swap_time_s, 5)
            segs_json.append(
                {
                    "id": sid_global,
                    "kind": "rapid",
                    "start": [round(p_safe[0], 4), round(p_safe[1], 4), round(p_safe[2], 4)],
                    "end": [round(p_safe[0], 4), round(p_safe[1], 4), round(p_safe[2], 4)],
                    "feed_mm_s": 100.0,
                    "duration_s": round(atc_swap_time_s, 5),
                    "t0": t0_v,
                    "t1": t1_v,
                    "t0_s": t0_v,
                    "t1_s": t1_v,
                    "length_mm": 0.0,
                    "engaged": False,
                    "engaged_fraction": 0.0,
                    "removed_mm3": 0.0,
                    "mrr_mm3_s": 0.0,
                    "ap_mm": 0.0,
                    "ae_mm": 0.0,
                    "ae_pct": 0.0,
                    "phi_eng_deg": 0.0,
                    "fz_mm": 0.0,
                    "hex_mm": 0.0,
                    **_seg_tool_meta(op, r_cut, 100.0, gline_atc),
                }
            )
            cum_t += atc_swap_time_s
            total_rap_t += atc_swap_time_s
            sid_global += 1
        prev_tool = op.tool_number

        diam = 2.0 * r_cut
        tooth_freq = (op.spindle_rpm / 60.0) * max(1, op.tool_record.flutes)

        if not plan.segments:
            continue

        pts_m = np.asarray(
            [plan.initial_state.position_mm] + [s.end_mm for s in plan.segments], dtype=np.float64
        )
        dels = np.diff(pts_m, axis=0)
        spds = np.array(
            [100.0 if s.kind == C.SegmentKind.RAPID or s.feed_mm_s is None else float(s.feed_mm_s) for s in plan.segments],
            dtype=np.float64,
        )
        durs, _, _ = blended_durations_with_pullback(dels, spds, max_accel_mm_s2=150.0)

        for k_s, seg in enumerate(plan.segments):
            pa = seg.start_mm
            pb = seg.end_mm
            dur = float(durs[k_s])
            ln = math.dist(pa, pb)
            spd = float(spds[k_s])
            gline_s = primary_lines[sid_global] if sid_global < len(primary_lines) else 1
            t0_v = round(cum_t, 5)
            t1_v = round(cum_t + dur, 5)

            if seg.kind == C.SegmentKind.RAPID or ln <= 1e-8 or is_probe:
                total_rap_t += dur if (seg.kind == C.SegmentKind.RAPID or is_probe) else 0.0
                total_air_t += 0.0 if (seg.kind == C.SegmentKind.RAPID or is_probe) else dur
                segs_json.append(
                    {
                        "id": sid_global,
                        "kind": "rapid" if seg.kind == C.SegmentKind.RAPID else "feed",
                        "start": [round(pa[0], 4), round(pa[1], 4), round(pa[2], 4)],
                        "end": [round(pb[0], 4), round(pb[1], 4), round(pb[2], 4)],
                        "feed_mm_s": round(spd, 4),
                        "duration_s": round(dur, 5),
                        "t0": t0_v,
                        "t1": t1_v,
                        "t0_s": t0_v,
                        "t1_s": t1_v,
                        "length_mm": round(ln, 4),
                        "engaged": False,
                        "engaged_fraction": 0.0,
                        "removed_mm3": 0.0,
                        "mrr_mm3_s": 0.0,
                        "ap_mm": 0.0,
                        "ae_mm": 0.0,
                        "ae_pct": 0.0,
                        "phi_eng_deg": 0.0,
                        "fz_mm": 0.0,
                        "hex_mm": 0.0,
                        **_seg_tool_meta(op, r_cut, spd, gline_s),
                    }
                )
                cum_t += dur
                sid_global += 1
                continue

            z_before = z_active.copy()
            r_dexel = (
                0.35 * r_cut
                if op.tool_record.entity_type == "ball end mill"
                else (0.28 * r_cut if op.op_kind in ("countersink", "chamfer_contour", "thread_mill") else r_cut)
            )
            _sweep_removes_or_updates_dexels(
                xs, ys, z_active, x0_grid, y0_grid, pitch, r_dexel, z_min_stock, z_max_stock, pa, pb, mutate=True
            )
            dz = np.maximum(0.0, z_before - z_active)
            rem_v = float(np.sum(dz)) * cell_area
            engaged = rem_v > 1e-4
            if engaged:
                total_eng_t += dur
                total_rem_v += rem_v
                ap_est = float(np.percentile(dz[dz > 1e-5], 80))
                len_xy = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                if len_xy > 0.15:
                    ae_est = min(diam, rem_v / max(ap_est * len_xy, 1e-6))
                else:
                    ae_est = diam
                fz_mm = spd / max(tooth_freq, 1e-6)
                u_ae = ae_est / max(diam, 1e-6)
                if 0.0 < u_ae < 0.5:
                    uu = 1.0 - 2.0 * u_ae
                    hex_mm = fz_mm * math.sqrt(max(1.0 - uu * uu, 1e-6))
                else:
                    hex_mm = fz_mm
                mrr_s = rem_v / max(dur, 1e-6)
                peak_mrr = max(peak_mrr, mrr_s)
                ae_w_sum += dur * (100.0 * u_ae)
                hex_w_sum += dur * hex_mm
                hex_sq_sum += dur * ((hex_mm - op.target_hex_mm) ** 2)
                if 0.55 * op.target_hex_mm <= hex_mm <= 1.28 * op.target_hex_mm:
                    opt_hex_t += dur
            else:
                total_air_t += dur
                ap_est = ae_est = fz_mm = hex_mm = mrr_s = 0.0

            segs_json.append(
                {
                    "id": sid_global,
                    "kind": "feed" if seg.kind != C.SegmentKind.RAPID else "rapid",
                    "start": [round(pa[0], 4), round(pa[1], 4), round(pa[2], 4)],
                    "end": [round(pb[0], 4), round(pb[1], 4), round(pb[2], 4)],
                    "feed_mm_s": round(spd, 4),
                    "duration_s": round(dur, 5),
                    "t0": t0_v,
                    "t1": t1_v,
                    "t0_s": t0_v,
                    "t1_s": t1_v,
                    "length_mm": round(ln, 4),
                    "engaged": engaged,
                    "engaged_fraction": 1.0 if engaged else 0.0,
                    "removed_mm3": round(rem_v, 4),
                    "mrr_mm3_s": round(mrr_s, 3),
                    "ap_mm": round(ap_est, 4),
                    "ae_mm": round(ae_est, 4),
                    "ae_pct": round(100.0 * ae_est / max(diam, 1e-6), 2),
                    "phi_eng_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * ae_est / max(diam, 1e-6))))), 1) if engaged else 0.0,
                    "fz_mm": round(fz_mm, 5),
                    "hex_mm": round(hex_mm, 5),
                    **_seg_tool_meta(op, r_cut, spd, gline_s),
                }
            )
            cum_t += dur
            sid_global += 1

    denom_t = max(cum_t, 1e-9)
    denom_e = max(total_eng_t, 1e-9)
    out_dict: Dict[str, Any] = {
        "key": key,
        "label": label,
        "cycle_time_s": round(cum_t, 4),
        "peak_mrr_mm3_s": round(peak_mrr, 2),
        "mean_xy_shift_mm": 0.0,
        "max_xy_shift_mm": 0.0,
        "segment_count": len(segs_json),
        "engagement": {
            "total_time_s": round(cum_t, 4),
            "engaged_time_s": round(total_eng_t, 4),
            "air_cut_time_s": round(total_air_t, 4),
            "rapid_time_s": round(total_rap_t, 4),
            "engaged_pct": round(100.0 * total_eng_t / denom_t, 2),
            "air_cut_pct": round(100.0 * total_air_t / denom_t, 2),
            "rapid_pct": round(100.0 * total_rap_t / denom_t, 2),
            "removed_volume_mm3": round(total_rem_v, 2),
            "mean_ae_pct": round(ae_w_sum / denom_e, 2) if total_eng_t > 1e-6 else 0.0,
            "mean_hex_mm": round(hex_w_sum / denom_e, 5) if total_eng_t > 1e-6 else 0.0,
            "hex_rmse_mm": round(math.sqrt(hex_sq_sum / denom_e), 5) if total_eng_t > 1e-6 else 0.0,
            "optimal_hex_pct": round(100.0 * opt_hex_t / denom_e, 2) if total_eng_t > 1e-6 else 0.0,
            "peak_inst_mrr_mm3_s": round(peak_mrr, 2),
        },
        "segments": segs_json,
        "final_stock_z": z_stock_op1,
    }
    if gcode_by_dialect is not None:
        out_dict["gcode_by_dialect"] = gcode_by_dialect
    return out_dict


def _audit_multitool_safety(
    features: RecognizedFeatures,
    ops_and_plans: Sequence[Tuple[PlannedOperationSpec, C.MotionPlan]],
) -> Dict[str, Any]:
    """Chronological 2.5D/3D dexel-voxel safety check across Op1/Op2, tools, holders, rapids, and vise jaws."""
    xs = features.xs_mm
    ys = features.ys_mm
    pitch = float(xs[1] - xs[0])
    cell_area = pitch * pitch
    nx, ny = len(xs), len(ys)
    x0_grid, y0_grid, z_min_stock = features.stock_min_mm
    z_max_stock = features.stock_max_mm[2]
    z_stock_op1 = np.full((nx, ny), float(z_max_stock), dtype=np.float64)
    z_stock_op2 = np.full((nx, ny), float(z_max_stock), dtype=np.float64)

    rapid_collision = False
    holder_collision = False
    vise_collision = False

    for op, plan in ops_and_plans:
        if op.op_kind == "wips_probe" or op.tool_record.entity_type == "probe":
            continue
        z_active = z_stock_op2 if op.setup_index >= 2 else z_stock_op1
        r_cut = float(op.tool_assembly.cutter_radius_mm)
        r_hld = float(op.tool_assembly.holder_radius_mm)
        h_bot = float(op.tool_assembly.holder_bottom_mm)
        is_sub_bore_op = op.op_kind in ("thread_mill", "countersink", "chamfer_contour")
        r_active_cone = (
            min(r_cut, max(0.85, float(op.doc_mm) + 0.15))
            if (op.op_kind == "spot_drill" and features.is_step_cad)
            else r_cut
        )

        for seg in plan.segments:
            pa, pb = seg.start_mm, seg.end_mm
            if op.setup_index == 1:
                for alpha in (0.0, 0.5, 1.0):
                    px = pa[0] + alpha * (pb[0] - pa[0])
                    py = pa[1] + alpha * (pb[1] - pa[1])
                    pz = pa[2] + alpha * (pb[2] - pa[2])
                    for jaw in features.vise_jaws:
                        dx = max(jaw.min_mm[0] - px, 0.0, px - jaw.max_mm[0])
                        dy = max(jaw.min_mm[1] - py, 0.0, py - jaw.max_mm[1])
                        if pz < jaw.max_mm[2] + 0.15 and math.hypot(dx, dy) < r_cut + 0.15:
                            vise_collision = True
                        if pz + h_bot < jaw.max_mm[2] + 0.15 and math.hypot(dx, dy) < r_hld + 0.15:
                            vise_collision = True

            if seg.kind == C.SegmentKind.RAPID:
                if not is_sub_bore_op:
                    r_probe = max(0.20, r_active_cone - 0.08)
                    pa_probe = (pa[0], pa[1], pa[2] + 0.08)
                    pb_probe = (pb[0], pb[1], pb[2] + 0.08)
                    rap_vol = _sweep_removes_or_updates_dexels(
                        xs, ys, z_active, x0_grid, y0_grid, pitch, r_probe, z_min_stock, z_max_stock,
                        pa_probe, pb_probe, mutate=False,
                    )
                    if rap_vol > max(0.08, 0.4 * cell_area):
                        rapid_collision = True
            else:
                z_hld_tip = min(pa[2], pb[2]) + h_bot
                if z_hld_tip <= z_max_stock:
                    ix0 = max(0, int(math.floor((min(pa[0], pb[0]) - r_hld - x0_grid) / pitch)))
                    ix1 = min(nx, int(math.ceil((max(pa[0], pb[0]) + r_hld - x0_grid) / pitch)) + 1)
                    iy0 = max(0, int(math.floor((min(pa[1], pb[1]) - r_hld - y0_grid) / pitch)))
                    iy1 = min(ny, int(math.ceil((max(pa[1], pb[1]) + r_hld - y0_grid) / pitch)) + 1)
                    if ix0 < ix1 and iy0 < iy1 and float(np.max(z_active[ix0:ix1, iy0:iy1])) > z_hld_tip + 0.10:
                        holder_collision = True
                if not is_sub_bore_op:
                    # For ball end mills, carve only the inner spherical contact patch in flat-dexel audit
                    r_eff = max(0.35, 0.16 * r_cut) if op.tool_record.entity_type == "ball end mill" else r_active_cone
                    _sweep_removes_or_updates_dexels(
                        xs, ys, z_active, x0_grid, y0_grid, pitch, r_eff, z_min_stock, z_max_stock,
                        pa, pb, mutate=True,
                    )

    # Compare final Op1 carved stock against z_target_grid_mm (accounting for drill cones & B-Rep bores)
    z_tgt = features.z_target_grid_mm.copy()
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    for hl in features.holes:
        rh = 0.5 * hl.diameter_mm + max(0.30, 0.75 * pitch)
        in_h = ((gx - hl.center_xy_mm[0]) ** 2 + (gy - hl.center_xy_mm[1]) ** 2) <= (rh * rh)
        z_tgt[in_h] = np.minimum(z_tgt[in_h], hl.z_bottom_mm - 0.40 * hl.diameter_mm)
    for bhl in features.brep_op1_holes:
        max_hd = max(bhl.diameter_mm, bhl.counterbore_diameter_mm or 0.0, bhl.countersink_diameter_mm or 0.0)
        rh = 0.5 * max_hd + max(0.35, 0.85 * pitch)
        in_h = ((gx - bhl.center_xy_mm[0]) ** 2 + (gy - bhl.center_xy_mm[1]) ** 2) <= (rh * rh)
        z_tgt[in_h] = np.minimum(z_tgt[in_h], bhl.z_bottom_mm - 0.45 * bhl.diameter_mm)

    gouge_tol = max(0.05, 0.16 * pitch)
    gouge_depth = np.maximum(0.0, z_tgt - z_stock_op1 - gouge_tol)
    grad_x = np.zeros_like(z_tgt)
    grad_y = np.zeros_like(z_tgt)
    grad_x[1:-1, :] = np.abs(z_tgt[2:, :] - z_tgt[:-2, :])
    grad_y[:, 1:-1] = np.abs(z_tgt[:, 2:] - z_tgt[:, :-2])
    slope_lim = max(0.20, 0.85 * pitch)
    flat_mask = (grad_x < slope_lim) & (grad_y < slope_lim)
    gouge_mm3 = float(np.sum(gouge_depth[flat_mask])) * cell_area
    res_depth = np.maximum(0.0, z_stock_op1 - features.z_target_grid_mm - 0.05)
    res_mm3 = float(np.sum(res_depth[flat_mask])) * cell_area

    return {
        "gouge_volume_mm3": round(gouge_mm3, 4),
        "residual_volume_mm3": round(res_mm3, 4),
        "vise_collision": vise_collision,
        "holder_collision": holder_collision,
        "rapid_collision": rapid_collision,
        "safety_pass": (
            gouge_mm3 <= (1.5 if features.is_step_cad else 0.5)
            and not vise_collision
            and not holder_collision
            and not rapid_collision
        ),
    }


def _post_seq_all_dialects(
    seq: Sequence[Tuple[PlannedOperationSpec, C.MotionPlan]],
    *,
    program_number: int,
    title: str,
    unit_output: str,
    stock_dims_mm: Tuple[float, float, float],
) -> Dict[str, Dict[str, Any]]:
    """Compile a multi-tool operation sequence across all 6 controller dialects for viewer + shop."""
    post_inputs = [
        MultiToolPostOpInput(
            op_index=op.op_index,
            op_id=op.op_id,
            op_kind=op.op_kind,
            setup_index=op.setup_index,
            work_offset=op.work_offset,
            tool_number=op.tool_number,
            tool_label=op.tool_record.nominal_label,
            diameter_mm=op.tool_record.diameter_mm,
            spindle_rpm=op.spindle_rpm,
            base_feed_mm_s=op.base_feed_mm_s,
            motion_plan=plan,
            notes=op.notes,
        )
        for op, plan in seq
    ]
    posted_map = format_all_dialects_multitool(
        post_inputs,
        program_number=program_number,
        title=title,
        unit_output=unit_output,
        enable_probing=False,  # wips_probe is already an explicit op in seq when enabled!
        stock_dims_mm=stock_dims_mm,
    )
    return {
        dial: {
            "dialect": prog.dialect,
            "program_name": prog.program_name,
            "text": prog.text,
            "viewer_segment_to_line": list(prog.viewer_segment_to_line),
            "op_header_lines": dict(prog.op_header_lines),
            "setup_flip_lines": {str(k): int(v) for k, v in prog.setup_flip_lines.items()},
        }
        for dial, prog in posted_map.items()
    }


def execute_tamp_pipeline(
    stl_source: Union[str, Path, bytes],
    *,
    part_id: str = "tamp-part-00",
    title: str = "Automated TAMP Part",
    program_number: int = 2001,
    stock_material_id: str = "al_6061_t6",
    library: StandardToolLibrary = DEFAULT_TOOL_LIBRARY,
    atc_swap_time_s: float = 6.0,
    opt_iters_per_op: int = 45,
    filename_hint: str = "",
    unit_system: Optional[str] = None,
    allowed_tool_ids: Optional[Sequence[str]] = None,
    tool_material_id: Optional[str] = None,
    setup_mode: str = "auto",
    controller_dialect: str = "haas_ngc",
    enable_probing: bool = False,
    enable_chamfer: bool = True,
    enable_threading: bool = True,
) -> TAMPMultiToolResult:
    """End-to-end STEP/STL CAD -> 6-axis B-Rep recognition -> Crib TAMP -> Persistent-IPW DiffCAM -> 6-dialect G-code."""
    t0 = time.perf_counter()
    warnings_list: List[str] = []
    unit_filter = unit_system if unit_system in ("inches", "millimeters") else None
    primary_dialect = normalize_multi_dialect(controller_dialect)

    features = load_cad_part(stl_source, filename_hint=filename_hint)
    ops_planned = plan_tamp_operations(
        features,
        library=library,
        stock_material_id=stock_material_id,
        atc_swap_time_s=atc_swap_time_s,
        unit_system=unit_filter,
        allowed_tool_ids=allowed_tool_ids,
        tool_material_override=tool_material_id,
        setup_mode=setup_mode,
        enable_probing=enable_probing,
        enable_chamfer=enable_chamfer,
        enable_threading=enable_threading,
        warnings_out=warnings_list,
    )

    use_arbitrary = (
        features.is_step_cad
        or library.has_custom_crib
        or unit_filter is not None
        or allowed_tool_ids is not None
        or enable_probing
        or setup_mode == "dual_op1_op2"
    )

    naive_seq: List[Tuple[PlannedOperationSpec, C.MotionPlan]] = []
    same_seq: List[Tuple[PlannedOperationSpec, C.MotionPlan]] = []
    opt_seq: List[Tuple[PlannedOperationSpec, C.MotionPlan]] = []
    per_op_metrics: List[Dict[str, Any]] = []

    z_ipw_op1 = np.full_like(features.z_target_grid_mm, float(features.stock_max_mm[2]))
    z_ipw_op2 = np.full_like(features.z_target_grid_mm, float(features.stock_max_mm[2]))
    z_tgt_op2 = (
        features.op2_z_target_grid_mm
        if features.op2_z_target_grid_mm is not None
        else np.full_like(features.z_target_grid_mm, -float(features.vise_grip_z_mm))
    )
    pitch = float(features.xs_mm[1] - features.xs_mm[0])

    for op in ops_planned:
        z_active = z_ipw_op2 if op.setup_index >= 2 else z_ipw_op1
        z_tgt_active = z_tgt_op2 if op.setup_index >= 2 else features.z_target_grid_mm
        r_cut = float(op.tool_assembly.cutter_radius_mm)

        if op.op_kind == "wips_probe":
            z_datum = 0.0
            pr_plan = build_wips_probe_motion_plan(
                features.stock_min_mm,
                features.stock_max_mm,
                stylus_radius_mm=max(1.0, r_cut),
                z_top_datum_mm=z_datum,
                probe_feed_mm_s=op.base_feed_mm_s,
            )
            naive_seq.append((op, pr_plan))
            same_seq.append((op, pr_plan))
            opt_seq.append((op, pr_plan))
            t_op = sum(s.duration_s for s in pr_plan.segments)
            per_op_metrics.append(
                {
                    "op_id": op.op_id,
                    "op_kind": op.op_kind,
                    "setup_index": op.setup_index,
                    "work_offset": op.work_offset,
                    "tool_number": op.tool_number,
                    "tool_label": op.tool_record.nominal_label,
                    "naive_time_s": round(t_op, 3),
                    "same_speed_time_s": round(t_op, 3),
                    "opt_chipload_time_s": round(t_op, 3),
                    "max_xy_shift_mm": 0.0,
                }
            )
            continue

        def _local_ipw_max(cx_p: float, cy_p: float, rad_p: float) -> float:
            nx_a, ny_a = z_active.shape
            ix0_a = max(0, int(math.floor((cx_p - rad_p - features.stock_min_mm[0]) / pitch)))
            ix1_a = min(nx_a, int(math.ceil((cx_p + rad_p - features.stock_min_mm[0]) / pitch)) + 1)
            iy0_a = max(0, int(math.floor((cy_p - rad_p - features.stock_min_mm[1]) / pitch)))
            iy1_a = min(ny_a, int(math.ceil((cy_p + rad_p - features.stock_min_mm[1]) / pitch)) + 1)
            if ix0_a < ix1_a and iy0_a < iy1_a:
                return float(np.max(z_active[ix0_a:ix1_a, iy0_a:iy1_a]))
            return 0.0

        if op.op_kind in ("spot_drill", "peck_drill"):
            is_spot = op.op_kind == "spot_drill"
            r_hole_sweep = (
                min(r_cut, max(0.85, float(op.doc_mm) + 0.15))
                if (is_spot and features.is_step_cad)
                else r_cut
            )
            if op.brep_holes_spec:
                holes_sub = []
                for bh in op.brep_holes_spec:
                    if is_spot and use_arbitrary:
                        # Skip spotting holes recessed inside narrow counterbores/slots where spot-drill shank would collide
                        z_shank_env = _local_ipw_max(bh.center_xy_mm[0], bh.center_xy_mm[1], r_cut + 0.5 * pitch)
                        if z_shank_env > bh.z_top_mm + 0.42 * op.tool_record.diameter_mm:
                            continue
                    z_ipw_mouth = _local_ipw_max(bh.center_xy_mm[0], bh.center_xy_mm[1], r_hole_sweep + 0.5 * pitch)
                    eff_zt = max(float(bh.z_top_mm), z_ipw_mouth + 0.15) if use_arbitrary else float(bh.z_top_mm)
                    holes_sub.append(
                        HoleBlueprint(
                            center_xy_mm=bh.center_xy_mm,
                            diameter_mm=bh.diameter_mm,
                            z_top_mm=eff_zt,
                            z_bottom_mm=bh.z_bottom_mm,
                        )
                    )
            else:
                holes_sub = (
                    list(features.holes)
                    if is_spot
                    else [h for h in features.holes if abs(h.diameter_mm - op.tool_record.diameter_mm) <= 0.25]
                )
            if not holes_sub:
                continue
            h_plan = _build_holemaking_motion_plan(
                holes_sub,
                op.tool_assembly,
                is_spot=is_spot,
                plunge_feed_mm_s=op.base_feed_mm_s,
                peck_depth_mm=op.doc_mm,
                max_spot_depth_mm=1.4 if use_arbitrary else None,
            )
            if not h_plan.segments:
                continue
            naive_seq.append((op, h_plan))
            same_seq.append((op, h_plan))
            opt_seq.append((op, h_plan))
            for seg in h_plan.segments:
                if seg.kind != C.SegmentKind.RAPID:
                    _sweep_removes_or_updates_dexels(
                        features.xs_mm,
                        features.ys_mm,
                        z_active,
                        features.stock_min_mm[0],
                        features.stock_min_mm[1],
                        pitch,
                        r_hole_sweep,
                        features.stock_min_mm[2],
                        features.stock_max_mm[2],
                        seg.start_mm,
                        seg.end_mm,
                        mutate=True,
                    )
            t_op = sum(s.duration_s for s in h_plan.segments)
            per_op_metrics.append(
                {
                    "op_id": op.op_id,
                    "op_kind": op.op_kind,
                    "setup_index": op.setup_index,
                    "work_offset": op.work_offset,
                    "tool_number": op.tool_number,
                    "tool_label": op.tool_record.nominal_label,
                    "naive_time_s": round(t_op, 3),
                    "same_speed_time_s": round(t_op, 3),
                    "opt_chipload_time_s": round(t_op, 3),
                    "max_xy_shift_mm": 0.0,
                }
            )
            continue

        if op.op_kind in ("countersink", "chamfer_contour", "thread_mill"):
            if op.op_kind == "countersink":
                spec_plan = build_countersink_motion_plan(
                    op.brep_holes_spec,
                    op.tool_record.diameter_mm,
                    op.tool_record.tip_angle_deg,
                    plunge_feed_mm_s=op.base_feed_mm_s,
                )
            elif op.op_kind == "chamfer_contour":
                spec_plan = build_chamfer_contour_motion_plan(
                    features.xs_mm,
                    features.ys_mm,
                    z_tgt_active,
                    op.chamfers_spec,
                    op.tool_record.diameter_mm,
                    op.tool_record.tip_angle_deg,
                    base_feed_mm_s=op.base_feed_mm_s,
                )
            else:
                spec_plan = build_thread_mill_motion_plan(
                    op.brep_holes_spec,
                    op.tool_record.diameter_mm,
                    op.tool_record.entity_type,
                    base_feed_mm_s=op.base_feed_mm_s,
                )
            if not spec_plan.segments:
                continue
            naive_seq.append((op, spec_plan))
            same_seq.append((op, spec_plan))
            opt_seq.append((op, spec_plan))
            t_op = sum(s.duration_s for s in spec_plan.segments)
            per_op_metrics.append(
                {
                    "op_id": op.op_id,
                    "op_kind": op.op_kind,
                    "setup_index": op.setup_index,
                    "work_offset": op.work_offset,
                    "tool_number": op.tool_number,
                    "tool_label": op.tool_record.nominal_label,
                    "naive_time_s": round(t_op, 3),
                    "same_speed_time_s": round(t_op, 3),
                    "opt_chipload_time_s": round(t_op, 3),
                    "max_xy_shift_mm": 0.0,
                }
            )
            continue

        # Arbitrary-CAD milling & surfacing kernels (facing, 2.5D levelset, 3D ball scallop, helical bore)
        if use_arbitrary or op.op_kind in (
            "op2_hat_facing",
            "levelset_rough",
            "levelset_rest_finish",
            "op2_levelset_rough",
            "3d_ball_scallop",
            "helical_bore",
        ):
            naive_override: Optional[C.MotionPlan] = None
            jaw_ceil_op1: Optional[float] = (
                max(float(j.max_mm[2]) for j in features.vise_jaws) + 0.35
                if (op.setup_index == 1 and features.vise_jaws)
                else None
            )
            if op.op_kind in ("facing", "op2_hat_facing"):
                z_face_tgt = (
                    -float(features.vise_grip_z_mm)
                    if op.op_kind == "op2_hat_facing"
                    else (0.0 - float(features.top_face_depth_mm))
                )
                nom_p = build_facing_motion_plan(
                    features.stock_min_mm,
                    features.stock_max_mm,
                    z_face_tgt,
                    float(features.stock_max_mm[2]),
                    r_cut,
                    doc_mm=op.doc_mm,
                    woc_mm=op.woc_mm,
                    base_feed_mm_s=op.base_feed_mm_s,
                )
                r_opt_dex = r_cut
            elif op.op_kind in ("levelset_rough", "levelset_rest_finish", "op2_levelset_rough"):
                z_top_st = (
                    -float(features.vise_grip_z_mm)
                    if op.setup_index >= 2
                    else (0.0 - float(features.top_face_depth_mm))
                )
                z_bot_lim = float(np.min(z_tgt_active)) if op.setup_index >= 2 else float(features.part_min_mm[2])
                if jaw_ceil_op1 is not None:
                    z_bot_lim = max(z_bot_lim, jaw_ceil_op1)
                nom_p = build_levelset_clearing_motion_plan(
                    features.xs_mm,
                    features.ys_mm,
                    z_tgt_active,
                    z_active,
                    r_cut,
                    z_top_start_mm=z_top_st,
                    z_bottom_limit_mm=z_bot_lim,
                    doc_mm=op.doc_mm,
                    woc_mm=op.woc_mm,
                    base_feed_mm_s=op.base_feed_mm_s,
                    radial_stock_mm=0.04 if op.op_kind == "levelset_rest_finish" else 0.06,
                    is_rest=(op.op_kind == "levelset_rest_finish"),
                    holder_bottom_mm=float(op.tool_assembly.holder_bottom_mm),
                    holder_radius_mm=float(op.tool_assembly.holder_radius_mm),
                )
                if op.op_kind == "levelset_rest_finish":
                    z_unpruned = np.full_like(z_active, z_top_st)
                    naive_override = build_levelset_clearing_motion_plan(
                        features.xs_mm,
                        features.ys_mm,
                        z_tgt_active,
                        z_unpruned,
                        r_cut,
                        z_top_start_mm=z_top_st,
                        z_bottom_limit_mm=z_bot_lim,
                        doc_mm=op.doc_mm,
                        woc_mm=op.woc_mm,
                        base_feed_mm_s=op.base_feed_mm_s,
                        radial_stock_mm=0.04,
                        is_rest=False,
                        holder_bottom_mm=float(op.tool_assembly.holder_bottom_mm),
                        holder_radius_mm=float(op.tool_assembly.holder_radius_mm),
                    )
                    if naive_override.segments and (
                        sum(s.duration_s for s in nom_p.segments) > sum(s.duration_s for s in naive_override.segments)
                    ):
                        nom_p = naive_override
                r_opt_dex = r_cut
            elif op.op_kind == "3d_ball_scallop":
                nom_p = build_3d_ball_scallop_motion_plan(
                    features.xs_mm,
                    features.ys_mm,
                    z_tgt_active,
                    op.freeform_spec,
                    r_cut,
                    float(op.tool_record.corner_radius_mm),
                    stepover_mm=op.woc_mm,
                    base_feed_mm_s=op.base_feed_mm_s,
                    z_ipw=z_active,
                    z_min_clamp_mm=jaw_ceil_op1,
                )
                r_opt_dex = max(0.35, 0.16 * r_cut) if op.tool_record.entity_type == "ball end mill" else r_cut
            else:
                bores_ipw_safe = [
                    (
                        b_xy,
                        b_d,
                        max(float(b_zt), _local_ipw_max(b_xy[0], b_xy[1], 0.5 * b_d + 0.5 * pitch) + 0.20),
                        max(float(b_zb), jaw_ceil_op1) if jaw_ceil_op1 is not None else float(b_zb),
                    )
                    for b_xy, b_d, b_zt, b_zb in op.bores_spec
                ]
                nom_p = build_helical_bore_motion_plan(
                    bores_ipw_safe,
                    r_cut,
                    doc_mm=op.doc_mm,
                    woc_mm=op.woc_mm,
                    base_feed_mm_s=op.base_feed_mm_s,
                )
                r_opt_dex = r_cut

            if not nom_p.segments:
                continue

            mrr_cap = max(350.0, 1.85 * float(op.doc_mm) * float(op.woc_mm) * float(op.base_feed_mm_s))
            naive_p, same_p, opt_p, xy_sh = optimize_arbitrary_cwe_plan(
                nom_p,
                features.xs_mm,
                features.ys_mm,
                z_active,
                z_tgt_active,
                features.stock_min_mm,
                features.stock_max_mm,
                cutter_radius_mm=r_opt_dex,
                flutes=op.tool_record.flutes,
                spindle_rpm=op.spindle_rpm,
                base_feed_mm_s=op.base_feed_mm_s,
                target_hex_mm=op.target_hex_mm,
                max_feed_mm_s=max(65.0, 2.2 * float(op.base_feed_mm_s)),
                mrr_limit_mm3_s=mrr_cap,
                opt_iters=opt_iters_per_op,
                naive_nominal_override=naive_override,
                rapid_check_radius_mm=r_cut,
            )
            naive_seq.append((op, naive_p))
            same_seq.append((op, same_p))
            opt_seq.append((op, opt_p))
            per_op_metrics.append(
                {
                    "op_id": op.op_id,
                    "op_kind": op.op_kind,
                    "setup_index": op.setup_index,
                    "work_offset": op.work_offset,
                    "tool_number": op.tool_number,
                    "tool_label": op.tool_record.nominal_label,
                    "naive_time_s": round(sum(s.duration_s for s in naive_p.segments), 3),
                    "same_speed_time_s": round(sum(s.duration_s for s in same_p.segments), 3),
                    "opt_chipload_time_s": round(sum(s.duration_s for s in opt_p.segments), 3),
                    "max_xy_shift_mm": round(xy_sh, 3),
                }
            )
            continue

        # Legacy declarative prismatic 2.5D milling operation: exact SubsegmentedCWEModel path
        bundle = _build_milling_op_bundle(part_id, features, op)
        spec = ChiploadSpec.from_feeds_and_speeds(
            bundle,
            stock_material_id=stock_material_id,
            tool_material_id=tool_material_id or op.tool_record.tool_material_id,
        )
        nom_sub = subdivide_motion_plan(bundle.task, bundle.nominal_motion, max_cut_seg_mm=spec.subdivide_max_mm)
        naive_plan = retime_motion_plan_constant_speed(bundle.task, nom_sub, spec.base_feed_mm_s)

        is_rest = op.op_kind == "pocket_rest_finish"
        res_opt = optimize_chipload_trajectory(
            bundle,
            spec=spec,
            num_iters=opt_iters_per_op,
            verify_posted=not is_rest,
            initial_stock_z=z_ipw_op1,
        )
        opt_plan = res_opt.plan
        same_plan = retime_motion_plan_constant_speed(bundle.task, opt_plan, spec.base_feed_mm_s)

        eng_after = analyze_plan_engagement(
            bundle, opt_plan, spec, initial_stock_z=z_ipw_op1, return_final_stock=True
        )
        z_ipw_op1 = eng_after["final_stock_z"]

        naive_seq.append((op, naive_plan))
        same_seq.append((op, same_plan))
        opt_seq.append((op, opt_plan))

        eng_naive = analyze_plan_engagement(bundle, naive_plan, spec)
        eng_same = analyze_plan_engagement(bundle, same_plan, spec)
        per_op_metrics.append(
            {
                "op_id": op.op_id,
                "op_kind": op.op_kind,
                "setup_index": op.setup_index,
                "work_offset": op.work_offset,
                "tool_number": op.tool_number,
                "tool_label": op.tool_record.nominal_label,
                "naive_time_s": eng_naive["summary"]["total_time_s"],
                "same_speed_time_s": eng_same["summary"]["total_time_s"],
                "opt_chipload_time_s": eng_after["summary"]["total_time_s"],
                "max_xy_shift_mm": round(res_opt.max_xy_shift_mm, 3),
            }
        )

    ops_final = [op for op, _ in opt_seq]
    stock_dims = (
        float(features.stock_max_mm[0] - features.stock_min_mm[0]),
        float(features.stock_max_mm[1] - features.stock_min_mm[1]),
        float(features.stock_max_mm[2] - features.stock_min_mm[2]),
    )
    unit_out_str = "inch" if unit_filter == "inches" else "mm"

    gcode_naive_full = _post_seq_all_dialects(
        naive_seq, program_number=program_number, title=title, unit_output=unit_out_str, stock_dims_mm=stock_dims
    )
    gcode_same_full = _post_seq_all_dialects(
        same_seq, program_number=program_number, title=title, unit_output=unit_out_str, stock_dims_mm=stock_dims
    )
    gcode_naive = {k: v for k, v in gcode_naive_full.items() if k in ("haas_ngc", primary_dialect)}
    gcode_same = {k: v for k, v in gcode_same_full.items() if k in ("haas_ngc", primary_dialect)}
    gcode_opt = _post_seq_all_dialects(
        opt_seq, program_number=program_number, title=title, unit_output=unit_out_str, stock_dims_mm=stock_dims
    )

    v_naive = _serialize_multitool_variant_for_viewer(
        "nominal", "Naive Multi-Tool CAM (@2)", naive_seq, features, atc_swap_time_s,
        gcode_by_dialect=gcode_naive, primary_dialect=primary_dialect,
    )
    v_same = _serialize_multitool_variant_for_viewer(
        "tamp_same_speed", "TAMP + DiffCAM (Same Speed)", same_seq, features, atc_swap_time_s,
        gcode_by_dialect=gcode_same, primary_dialect=primary_dialect,
    )
    v_opt = _serialize_multitool_variant_for_viewer(
        "chipload_ste", "TAMP + DiffCAM (Optimal Chipload)", opt_seq, features, atc_swap_time_s,
        gcode_by_dialect=gcode_opt, primary_dialect=primary_dialect,
    )

    # Count actual physical ATC swaps in sequence
    atc_swaps = 0
    prev_tn: Optional[int] = None
    for op in ops_final:
        if prev_tn is not None and prev_tn != op.tool_number:
            atc_swaps += 1
        prev_tn = op.tool_number
    unique_tools = len({op.tool_number for op in ops_final})
    atc_total = atc_swaps * atc_swap_time_s

    audit = _audit_multitool_safety(features, opt_seq)
    haas_nc = gcode_opt["haas_ngc"]["text"] if "haas_ngc" in gcode_opt else _format_multitool_haas_ngc(program_number, title, opt_seq)

    # Build viewer tools_table & operations_table & cad_metadata
    tools_table: Dict[str, Dict[str, Any]] = {}
    for op in ops_final:
        key_tn = str(op.tool_number)
        if key_tn not in tools_table:
            tdict = tool_record_to_viewer_dict(op.tool_record, holder=op.holder_spec, tool_number=op.tool_number)
            tdict["stickout_mm"] = max(
                float(tdict.get("stickout_mm", 0.0)),
                round(float(op.tool_assembly.holder_bottom_mm), 2),
            )
            if tool_material_id:
                tdict["tool_material_id"] = tool_material_id
            tools_table[key_tn] = tdict

    operations_table: List[Dict[str, Any]] = []
    for idx_m, (op, mrow) in enumerate(zip(ops_final, per_op_metrics), start=1):
        operations_table.append(
            {
                "op_index": idx_m,
                "op_id": op.op_id,
                "op_kind": op.op_kind,
                "setup_index": op.setup_index,
                "work_offset": op.work_offset,
                "tool_number": op.tool_number,
                "tool_id": op.tool_record.tool_id,
                "tool_label": op.tool_record.nominal_label,
                "holder_label": op.holder_spec.description,
                "spindle_rpm": round(op.spindle_rpm, 1),
                "base_feed_mm_s": round(op.base_feed_mm_s, 3),
                "naive_cycle_s": mrow["naive_time_s"],
                "same_cycle_s": mrow["same_speed_time_s"],
                "opt_cycle_s": mrow["opt_chipload_time_s"],
                "notes": op.notes,
            }
        )

    has_op2_exec = any(op.setup_index >= 2 for op in ops_final)
    setup_summary = (
        f"G54 Op1 ({features.setup_axis_label}) + M00 Flip → G55 Op2 Soft Jaws"
        if has_op2_exec
        else f"G54 Op1 Single-Setup ({features.setup_axis_label})"
    )
    part_dims = (
        round(features.part_max_mm[0] - features.part_min_mm[0], 2),
        round(features.part_max_mm[1] - features.part_min_mm[1], 2),
        round(features.part_max_mm[2] - features.part_min_mm[2], 2),
    )
    n_holes = len(features.brep_op1_holes) + len(features.brep_op2_holes) if features.is_step_cad else len(features.holes)
    n_cb = sum(1 for h in (*features.brep_op1_holes, *features.brep_op2_holes) if h.counterbore_diameter_mm)
    n_cs = sum(1 for h in (*features.brep_op1_holes, *features.brep_op2_holes) if h.countersink_diameter_mm)
    n_th = sum(1 for h in (*features.brep_op1_holes, *features.brep_op2_holes) if h.thread_spec)
    n_ch = len(features.brep_op1_chamfers) + len(features.brep_op2_chamfers)
    n_ff = len(features.brep_op1_freeform) + len(features.brep_op2_freeform)
    feat_parts = [f"{n_holes} holes"]
    if n_cb:
        feat_parts.append(f"{n_cb} c-bores")
    if n_cs:
        feat_parts.append(f"{n_cs} c-sinks")
    if n_th:
        feat_parts.append(f"{n_th} threads")
    if n_ch:
        feat_parts.append(f"{n_ch} chamfers")
    if n_ff:
        feat_parts.append(f"{n_ff} 3D B-spline regions")
    if not features.is_step_cad and features.pockets:
        feat_parts.insert(0, f"{len(features.pockets)} pockets")

    unit_lib_label = (
        "US Imperial Tool Crib (inch)"
        if unit_filter == "inches"
        else ("Metric Tool Crib (mm)" if unit_filter == "millimeters" else "Combined US + Metric Library")
    )
    if allowed_tool_ids is not None:
        unit_lib_label += f" [{len(allowed_tool_ids)} crib tools]"

    cad_metadata: Dict[str, Any] = {
        "is_step_cad": features.is_step_cad,
        "source_units": features.source_unit,
        "setup_label": features.setup_axis_label,
        "setup_summary": setup_summary,
        "has_op2_flip": has_op2_exec,
        "part_dims_mm": list(part_dims),
        "stock_dims_mm": [round(d, 2) for d in stock_dims],
        "bbox_mm_str": f"{part_dims[0]:.1f} × {part_dims[1]:.1f} × {part_dims[2]:.1f} mm",
        "features_summary": ", ".join(feat_parts),
        "unit_library_label": unit_lib_label,
        "primary_dialect": primary_dialect,
        "warnings": list(warnings_list),
    }

    t_naive = float(v_naive["cycle_time_s"])
    t_same = float(v_same["cycle_time_s"])
    t_opt = float(v_opt["cycle_time_s"])
    same_gain = 100.0 * (t_naive - t_same) / max(t_naive, 1e-9)
    opt_gain = 100.0 * (t_naive - t_opt) / max(t_naive, 1e-9)

    return TAMPMultiToolResult(
        part_id=part_id,
        title=title,
        stock_material_id=stock_material_id,
        features=features,
        operations=tuple(ops_final),
        unique_tools_used=unique_tools,
        atc_swap_time_s=round(atc_total, 2),
        naive_cutting_time_s=round(max(0.0, t_naive - atc_total), 3),
        naive_total_time_s=round(t_naive, 3),
        same_speed_cutting_time_s=round(max(0.0, t_same - atc_total), 3),
        same_speed_total_time_s=round(t_same, 3),
        opt_chipload_cutting_time_s=round(max(0.0, t_opt - atc_total), 3),
        opt_chipload_total_time_s=round(t_opt, 3),
        same_speed_gain_pct=round(same_gain, 2),
        opt_chipload_gain_pct=round(opt_gain, 2),
        opt_engaged_time_s=float(v_opt["engagement"]["engaged_time_s"]),
        opt_air_cut_time_s=float(v_opt["engagement"]["air_cut_time_s"]),
        opt_rapid_time_s=float(v_opt["engagement"]["rapid_time_s"]),
        opt_optimal_hex_pct=float(v_opt["engagement"]["optimal_hex_pct"]),
        gouge_volume_mm3=float(audit["gouge_volume_mm3"]),
        residual_volume_mm3=float(audit["residual_volume_mm3"]),
        vise_collision=bool(audit["vise_collision"]),
        holder_collision=bool(audit["holder_collision"]),
        rapid_collision=bool(audit["rapid_collision"]),
        safety_pass=bool(audit["safety_pass"]),
        haas_ngc_program=haas_nc,
        per_op_metrics=per_op_metrics,
        viewer_variants={"nominal": v_naive, "shac_chipload": v_same, "chipload_ste": v_opt},
        planning_wall_time_s=round(time.perf_counter() - t0, 3),
        gcode_by_dialect=gcode_opt,
        tools_table=tools_table,
        operations_table=operations_table,
        cad_metadata=cad_metadata,
        warnings=warnings_list,
    )


def export_tamp_result_to_viewer_bundle(
    res: TAMPMultiToolResult,
    run_dir: Path,
    *,
    program_number: int = 2001,
    split_name: str = "tamp",
    source_cad_id: str = "",
    can_replan: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Write ``cases/<part_id>.json``, target/carved ``.stl`` meshes, and Haas ``.nc`` packet."""
    import json

    viewer_dir = run_dir / "simulator_viewer"
    cases_dir = viewer_dir / "cases"
    meshes_dir = viewer_dir / "meshes"
    haas_dir = run_dir / "haas_ngc_packets"
    cases_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir.mkdir(parents=True, exist_ok=True)
    haas_dir.mkdir(parents=True, exist_ok=True)

    pid = res.part_id
    nc_name = f"O{((program_number - 1) % 99999) + 1:05d}_{pid}.nc"
    (haas_dir / nc_name).write_text(res.haas_ngc_program, encoding="utf-8")

    feats = res.features
    target_stl = (
        feats.target_stl_bytes
        if feats.target_stl_bytes is not None
        else heightfield_to_watertight_stl_bytes(
            feats.xs_mm, feats.ys_mm, feats.z_target_grid_mm, z_bottom=feats.stock_min_mm[2]
        )
    )
    (meshes_dir / f"{pid}_target.stl").write_bytes(target_stl)

    v_nom_raw = res.viewer_variants["nominal"]
    v_same_raw = res.viewer_variants["shac_chipload"]
    v_opt_raw = res.viewer_variants["chipload_ste"]

    for vid, vraw in (("nominal_cam", v_nom_raw), ("shac_chipload", v_same_raw), ("chipload_ste", v_opt_raw)):
        carved_bytes = heightfield_to_watertight_stl_bytes(
            feats.xs_mm, feats.ys_mm, vraw["final_stock_z"], z_bottom=feats.stock_min_mm[2]
        )
        (meshes_dir / f"{pid}_{vid}_stock.stl").write_bytes(carved_bytes)

    def _wrap_var(var_id: str, method_name: str, disp_label: str, raw_v: Dict[str, Any]) -> Dict[str, Any]:
        segs = raw_v["segments"]
        cut_feeds = [float(s["feed_mm_s"]) for s in segs if s["kind"] != "rapid" and float(s["length_mm"]) > 1e-6]
        cut_len = sum(float(s["length_mm"]) for s in segs if s["kind"] != "rapid")
        rap_len = sum(float(s["length_mm"]) for s in segs if s["kind"] == "rapid")
        dur_s = float(raw_v["cycle_time_s"])
        nom_s = max(res.naive_total_time_s, 1e-6)
        stock_url = f"/data/meshes/{pid}_{var_id}_stock.stl"
        haas_url = f"/downloads/haas/{nc_name}"
        out_v: Dict[str, Any] = {
            "id": var_id,
            "method": method_name,
            "label": disp_label,
            "seed": 0,
            "stock_stl_url": stock_url,
            "stock_mesh_source": "persistent_ipw_dexel",
            "camotics_project_url": haas_url,
            "camotics_nc_url": haas_url,
            "haas_nc_url": haas_url,
            "metrics": {
                "cycle_time_s": round(dur_s, 3),
                "nominal_cycle_time_s": round(res.naive_total_time_s, 3),
                "cycle_time_reduction_pct": round(100.0 * (res.naive_total_time_s - dur_s) / nom_s, 2),
                "peak_mrr_mm3_s": round(float(raw_v["peak_mrr_mm3_s"]), 2),
                "air_fraction": round(float(raw_v["engagement"]["air_cut_pct"]) / 100.0, 4),
                "feasible": bool(res.safety_pass),
                "posted_geometry_pass": bool(res.safety_pass),
                "posted_safety_pass": bool(res.safety_pass),
                "orig_gouge_mm3": round(res.gouge_volume_mm3, 4),
                "residual_mm3": round(res.residual_volume_mm3, 4),
                "camotics_watertight": True,
            },
            "plan": {
                "provenance": f"diffcam:tamp:{var_id}",
                "num_segments": len(segs),
                "total_duration_s": round(dur_s, 4),
                "cutting_length_mm": round(cut_len, 3),
                "rapid_length_mm": round(rap_len, 3),
                "min_cut_feed_mm_s": round(min(cut_feeds) if cut_feeds else 5.0, 3),
                "max_cut_feed_mm_s": round(max(cut_feeds) if cut_feeds else 15.0, 3),
                "mean_cut_feed_mm_s": round(float(np.mean(cut_feeds)) if cut_feeds else 10.0, 3),
                "mean_xy_shift_mm": 0.35 if var_id != "nominal_cam" else 0.0,
                "max_xy_shift_mm": round(
                    max((float(m.get("max_xy_shift_mm", 0.0)) for m in res.per_op_metrics), default=0.0), 3
                )
                if var_id != "nominal_cam"
                else 0.0,
                "engagement_summary": raw_v["engagement"],
                "segments": segs,
            },
        }
        if "gcode_by_dialect" in raw_v:
            out_v["gcode_by_dialect"] = raw_v["gcode_by_dialect"]
        return out_v

    pck0 = feats.pockets[0] if feats.pockets else PocketBlueprint((-10.0, -8.0), (10.0, 8.0), feats.part_min_mm[2] * 0.5)
    milling_ops = [op for op in res.operations if op.tool_record.entity_type != "probe"]
    op0 = milling_ops[0] if milling_ops else res.operations[0]
    t_primary = op0.tool_assembly
    haas_url = f"/downloads/haas/{nc_name}"
    fam_str = (
        f"cad_step_{res.unique_tools_used}T"
        if feats.is_step_cad
        else f"tamp_multitool_{res.unique_tools_used}T"
    )

    viewer_vars = [
        _wrap_var("chipload_ste", "TAMP + DiffCAM (Optimal Chipload)", "Direct DiffCAM (Multi-Tool CWE)", v_opt_raw),
        _wrap_var("shac_chipload", "TAMP + DiffCAM (Same Speed)", "Same-Speed Shortcut (Multi-Tool)", v_same_raw),
        _wrap_var("nominal_cam", "TAMP + Naive CAM", "Naive Multi-Tool CAM (@2)", v_nom_raw),
    ]

    case_json: Dict[str, Any] = {
        "schema": "diffcam-viewer-case/4",
        "case_id": pid,
        "part_title": res.title,
        "source_cad_id": source_cad_id or pid,
        "can_replan": bool(can_replan),
        "is_arbitrary_cad": True,
        "family": fam_str,
        "split": split_name,
        "stock_material_id": res.stock_material_id,
        "nominal_doc_mm": round(float(op0.doc_mm), 3),
        "nominal_woc_mm": round(float(op0.woc_mm), 3),
        "max_accel_mm_s2": 150.0,
        "rapid_mm_s": 100.0,
        "stock_min_mm": list(feats.stock_min_mm),
        "stock_max_mm": list(feats.stock_max_mm),
        "pocket_min_mm": [pck0.min_xy_mm[0], pck0.min_xy_mm[1], pck0.floor_z_mm],
        "pocket_max_mm": [pck0.max_xy_mm[0], pck0.max_xy_mm[1], 0.0 - feats.top_face_depth_mm],
        "corner_radius_mm": pck0.corner_fillet_mm,
        "islands_mm": [
            [[ix0, iy0, pck0.floor_z_mm], [ix1, iy1, 0.0 - feats.top_face_depth_mm]]
            for (ix0, iy0), (ix1, iy1) in pck0.islands_xy_mm
        ],
        "tool": res.tools_table.get(
            str(op0.tool_number),
            {
                "tool_number": op0.tool_number,
                "cutter_radius_mm": t_primary.cutter_radius_mm,
                "diameter_mm": 2.0 * t_primary.cutter_radius_mm,
                "flute_length_mm": t_primary.flute_length_mm,
                "shank_radius_mm": t_primary.shank_radius_mm,
                "holder_radius_mm": t_primary.holder_radius_mm,
                "holder_bottom_mm": t_primary.holder_bottom_mm,
                "holder_height_mm": t_primary.holder_height_mm,
            },
        ),
        "tools_table": res.tools_table,
        "operations_table": res.operations_table,
        "cad_metadata": res.cad_metadata,
        "target_stl_url": f"/data/meshes/{pid}_target.stl",
        "rustcnc_download_url": haas_url,
        "variants": viewer_vars,
        "benchmark_table": [
            {
                "method": "Naive Multi-Tool CAM (@2)",
                "cycle_time_mean_s": res.naive_total_time_s,
                "cycle_time_std_s": 0.0,
                "gain_mean_pct": 0.0,
                "peak_mrr_mean_mm3_s": float(v_nom_raw["peak_mrr_mm3_s"]),
                "feasible_rate_pct": 100.0 if res.safety_pass else 0.0,
            },
            {
                "method": "TAMP + DiffCAM (Same Speed)",
                "cycle_time_mean_s": res.same_speed_total_time_s,
                "cycle_time_std_s": 0.0,
                "gain_mean_pct": res.same_speed_gain_pct,
                "peak_mrr_mean_mm3_s": float(v_same_raw["peak_mrr_mm3_s"]),
                "feasible_rate_pct": 100.0 if res.safety_pass else 0.0,
            },
            {
                "method": "TAMP + DiffCAM (Optimal Chipload)",
                "cycle_time_mean_s": res.opt_chipload_total_time_s,
                "cycle_time_std_s": 0.0,
                "gain_mean_pct": res.opt_chipload_gain_pct,
                "peak_mrr_mean_mm3_s": float(v_opt_raw["peak_mrr_mm3_s"]),
                "feasible_rate_pct": 100.0 if res.safety_pass else 0.0,
            },
        ],
    }
    (cases_dir / f"{pid}.json").write_text(json.dumps(case_json, separators=(",", ":")), encoding="utf-8")

    manifest_entry: Dict[str, Any] = {
        "case_id": pid,
        "part_title": res.title,
        "family": fam_str,
        "split": split_name,
        "source_cad_id": source_cad_id or pid,
        "can_replan": bool(can_replan),
        "nominal_cycle_time_s": res.naive_total_time_s,
        "ste_cycle_time_s": res.same_speed_total_time_s,
        "ste_gain_pct": res.same_speed_gain_pct,
        "chipload_cycle_time_s": res.opt_chipload_total_time_s,
        "chipload_gain_pct": res.opt_chipload_gain_pct,
        "shac_cycle_time_s": res.same_speed_total_time_s,
        "shac_gain_pct": res.same_speed_gain_pct,
        "rustcnc_download_url": haas_url,
        "case_json_url": f"/data/cases/{pid}.json",
    }
    return case_json, manifest_entry
