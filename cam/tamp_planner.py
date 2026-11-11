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
from cam.feeds_and_speeds import (
    MachineEnvelopeSpec,
    compute_feeds_and_speeds,
    compute_holemaking_feeds_and_speeds,
)
from cam.tool_library import (
     DEFAULT_TOOL_LIBRARY,
    CatalogToolRecord,
    HolderSpec,
    StandardToolLibrary,
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


def recognize_part_from_stl(
    stl_source: Union[str, Path, bytes],
    *,
    stock_top_margin_mm: float = 1.5,
    vise_grip_z_mm: float = 2.0,
    pitch_mm: float = 0.25,
) -> RecognizedFeatures:
    """Automatically select stock/workholding and recognize 2.5D + hole features from an ``.stl``."""
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
    op_kind: str  # "facing" | "exterior_contour" | "pocket_rough" | "pocket_rest_finish" | "spot_drill" | "peck_drill"
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
            # 90-deg spot drill chamfers hole mouth to 0.52 * D_hole depth
            spot_z = max(hl.z_bottom_mm, hl.z_top_mm - 0.52 * hl.diameter_mm)
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


def plan_tamp_operations(
    features: RecognizedFeatures,
    *,
    library: StandardToolLibrary = DEFAULT_TOOL_LIBRARY,
    stock_material_id: str = "al_6061_t6",
    machine: Optional[MachineEnvelopeSpec] = None,
    atc_swap_time_s: float = 6.0,
) -> List[PlannedOperationSpec]:
    """Task-level search selecting tools, operation sequence, and physics speeds/feeds."""
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
) -> Dict[str, Any]:
    """Serialize a multi-tool trajectory sequence into a viewer-compatible variant dict."""
    xs = features.xs_mm
    ys = features.ys_mm
    pitch = float(xs[1] - xs[0])
    cell_area = pitch * pitch
    nx, ny = len(xs), len(ys)
    x0_grid, y0_grid, z_min_stock = features.stock_min_mm
    z_max_stock = features.stock_max_mm[2]
    z_stock = np.full((nx, ny), float(z_max_stock), dtype=np.float64)

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

    for op, plan in ops_and_plans:
        if prev_tool is not None and prev_tool != op.tool_number:
            # Record non-cutting ATC tool change dwell at safe Z so timeline matches shop clock
            p_safe = plan.initial_state.position_mm
            segs_json.append(
                {
                    "id": sid_global,
                    "kind": "rapid",
                    "start": [round(p_safe[0], 4), round(p_safe[1], 4), round(p_safe[2], 4)],
                    "end": [round(p_safe[0], 4), round(p_safe[1], 4), round(p_safe[2], 4)],
                    "feed_mm_s": 100.0,
                    "duration_s": round(atc_swap_time_s, 5),
                    "t0": round(cum_t, 5),
                    "t1": round(cum_t + atc_swap_time_s, 5),
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
                    "tool_number": op.tool_number,
                    "tool_radius_mm": round(float(op.tool_assembly.cutter_radius_mm), 4),
                    "op_id": op.op_id,
                }
            )
            cum_t += atc_swap_time_s
            total_rap_t += atc_swap_time_s
            sid_global += 1
        prev_tool = op.tool_number

        r_cut = float(op.tool_assembly.cutter_radius_mm)
        diam = 2.0 * r_cut
        tooth_freq = (op.spindle_rpm / 60.0) * max(1, op.tool_record.flutes)

        # Exact blended durations for this operation
        pts_m = np.asarray(
            [plan.initial_state.position_mm] + [s.end_mm for s in plan.segments], dtype=np.float64
        )
        dels = np.diff(pts_m, axis=0)
        is_r = np.array([s.kind == C.SegmentKind.RAPID for s in plan.segments], dtype=bool)
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
            if seg.kind == C.SegmentKind.RAPID or ln <= 1e-8:
                total_rap_t += dur if seg.kind == C.SegmentKind.RAPID else 0.0
                total_air_t += 0.0 if seg.kind == C.SegmentKind.RAPID else dur
                segs_json.append(
                    {
                        "id": sid_global,
                        "kind": seg.kind.value,
                        "start": [round(pa[0], 4), round(pa[1], 4), round(pa[2], 4)],
                        "end": [round(pb[0], 4), round(pb[1], 4), round(pb[2], 4)],
                        "feed_mm_s": round(spd, 4),
                        "duration_s": round(dur, 5),
                        "t0": round(cum_t, 5),
                        "t1": round(cum_t + dur, 5),
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
                        "tool_number": op.tool_number,
                        "tool_radius_mm": round(r_cut, 4),
                        "op_id": op.op_id,
                    }
                )
                cum_t += dur
                sid_global += 1
                continue

            z_before = z_stock.copy()
            _sweep_removes_or_updates_dexels(
                xs, ys, z_stock, x0_grid, y0_grid, pitch, r_cut, z_min_stock, z_max_stock, pa, pb, mutate=True
            )
            dz = np.maximum(0.0, z_before - z_stock)
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
                    ae_est = diam  # axial plunge / drill
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
                    "kind": seg.kind.value,
                    "start": [round(pa[0], 4), round(pa[1], 4), round(pa[2], 4)],
                    "end": [round(pb[0], 4), round(pb[1], 4), round(pb[2], 4)],
                    "feed_mm_s": round(spd, 4),
                    "duration_s": round(dur, 5),
                    "t0": round(cum_t, 5),
                    "t1": round(cum_t + dur, 5),
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
                    "tool_number": op.tool_number,
                    "tool_radius_mm": round(r_cut, 4),
                    "op_id": op.op_id,
                }
            )
            cum_t += dur
            sid_global += 1

    denom_t = max(cum_t, 1e-9)
    denom_e = max(total_eng_t, 1e-9)
    return {
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
        "final_stock_z": z_stock,
    }


def _audit_multitool_safety(
    features: RecognizedFeatures,
    ops_and_plans: Sequence[Tuple[PlannedOperationSpec, C.MotionPlan]],
) -> Dict[str, Any]:
    """Chronological 2.5D/3D dexel-voxel safety check across all tools, holders, rapids, and vise jaws."""
    xs = features.xs_mm
    ys = features.ys_mm
    pitch = float(xs[1] - xs[0])
    cell_area = pitch * pitch
    nx, ny = len(xs), len(ys)
    x0_grid, y0_grid, z_min_stock = features.stock_min_mm
    z_max_stock = features.stock_max_mm[2]
    z_stock = np.full((nx, ny), float(z_max_stock), dtype=np.float64)

    rapid_collision = False
    holder_collision = False
    vise_collision = False

    for op, plan in ops_and_plans:
        r_cut = float(op.tool_assembly.cutter_radius_mm)
        r_hld = float(op.tool_assembly.holder_radius_mm)
        h_bot = float(op.tool_assembly.holder_bottom_mm)

        for seg in plan.segments:
            pa, pb = seg.start_mm, seg.end_mm
            # 1. Check vise-jaw clearance at endpoints and midpoint
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
                # Rapid must never penetrate remaining stock interior (allow 0.05 mm wall graze on vertical retract)
                r_probe = max(0.20, r_cut - 0.05)
                pa_probe = (pa[0], pa[1], pa[2] + 0.05)
                pb_probe = (pb[0], pb[1], pb[2] + 0.05)
                rap_vol = _sweep_removes_or_updates_dexels(
                    xs, ys, z_stock, x0_grid, y0_grid, pitch, r_probe, z_min_stock, z_max_stock,
                    pa_probe, pb_probe, mutate=False,
                )
                if rap_vol > 0.05:
                    rapid_collision = True
            else:
                # Check holder bottom against remaining stock before cutting
                z_hld_tip = min(pa[2], pb[2]) + h_bot
                if z_hld_tip <= z_max_stock:
                    ix0 = max(0, int(math.floor((min(pa[0], pb[0]) - r_hld - x0_grid) / pitch)))
                    ix1 = min(nx, int(math.ceil((max(pa[0], pb[0]) + r_hld - x0_grid) / pitch)) + 1)
                    iy0 = max(0, int(math.floor((min(pa[1], pb[1]) - r_hld - y0_grid) / pitch)))
                    iy1 = min(ny, int(math.ceil((max(pa[1], pb[1]) + r_hld - y0_grid) / pitch)) + 1)
                    if ix0 < ix1 and iy0 < iy1 and float(np.max(z_stock[ix0:ix1, iy0:iy1])) > z_hld_tip + 0.05:
                        holder_collision = True
                _sweep_removes_or_updates_dexels(
                    xs, ys, z_stock, x0_grid, y0_grid, pitch, r_cut, z_min_stock, z_max_stock,
                    pa, pb, mutate=True,
                )

    # Compare final carved stock against z_target_grid_mm (accounting for drill tip breakthrough cones)
    z_tgt = features.z_target_grid_mm.copy()
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    for hl in features.holes:
        rh = 0.5 * hl.diameter_mm + 0.25
        in_h = ((gx - hl.center_xy_mm[0]) ** 2 + (gy - hl.center_xy_mm[1]) ** 2) <= (rh * rh)
        z_tgt[in_h] = np.minimum(z_tgt[in_h], hl.z_bottom_mm - 0.35 * hl.diameter_mm)

    # Ignore 1-cell boundary rasterization alias at vertical feature edges when measuring true gouge
    gouge_depth = np.maximum(0.0, z_tgt - z_stock - 0.05)
    # Mask vertical walls where z_tgt jumps across adjacent pixels
    grad_x = np.zeros_like(z_tgt)
    grad_y = np.zeros_like(z_tgt)
    grad_x[1:-1, :] = np.abs(z_tgt[2:, :] - z_tgt[:-2, :])
    grad_y[:, 1:-1] = np.abs(z_tgt[:, 2:] - z_tgt[:, :-2])
    flat_mask = (grad_x < 0.20) & (grad_y < 0.20)
    gouge_mm3 = float(np.sum(gouge_depth[flat_mask])) * cell_area
    # Residual above target inside machined region (above part_min_z)
    res_depth = np.maximum(0.0, z_stock - features.z_target_grid_mm - 0.05)
    res_mm3 = float(np.sum(res_depth[flat_mask])) * cell_area

    return {
        "gouge_volume_mm3": round(gouge_mm3, 4),
        "residual_volume_mm3": round(res_mm3, 4),
        "vise_collision": vise_collision,
        "holder_collision": holder_collision,
        "rapid_collision": rapid_collision,
        "safety_pass": (
            gouge_mm3 <= 0.5
            and not vise_collision
            and not holder_collision
            and not rapid_collision
        ),
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
) -> TAMPMultiToolResult:
    """End-to-end ``.stl`` -> feature recognition -> multi-tool TAMP -> persistent-IPW DiffCAM -> Haas NGC."""
    t0 = time.perf_counter()
    features = recognize_part_from_stl(stl_source)
    ops = plan_tamp_operations(
        features,
        library=library,
        stock_material_id=stock_material_id,
        atc_swap_time_s=atc_swap_time_s,
    )

    naive_seq: List[Tuple[PlannedOperationSpec, C.MotionPlan]] = []
    same_seq: List[Tuple[PlannedOperationSpec, C.MotionPlan]] = []
    opt_seq: List[Tuple[PlannedOperationSpec, C.MotionPlan]] = []
    per_op_metrics: List[Dict[str, Any]] = []

    z_ipw_opt = np.full_like(features.z_target_grid_mm, float(features.stock_max_mm[2]))

    for op in ops:
        if op.op_kind in ("spot_drill", "peck_drill"):
            is_spot = op.op_kind == "spot_drill"
            holes_sub = (
                features.holes
                if is_spot
                else [h for h in features.holes if abs(h.diameter_mm - op.tool_record.diameter_mm) <= 0.25]
            )
            h_plan = _build_holemaking_motion_plan(
                holes_sub,
                op.tool_assembly,
                is_spot=is_spot,
                plunge_feed_mm_s=op.base_feed_mm_s,
                peck_depth_mm=op.doc_mm,
            )
            naive_seq.append((op, h_plan))
            same_seq.append((op, h_plan))
            opt_seq.append((op, h_plan))
            # Update persistent IPW for drilled holes
            r_cut = float(op.tool_assembly.cutter_radius_mm)
            for seg in h_plan.segments:
                if seg.kind != C.SegmentKind.RAPID:
                    _sweep_removes_or_updates_dexels(
                        features.xs_mm, features.ys_mm, z_ipw_opt,
                        features.stock_min_mm[0], features.stock_min_mm[1],
                        float(features.xs_mm[1] - features.xs_mm[0]),
                        r_cut, features.stock_min_mm[2], features.stock_max_mm[2],
                        seg.start_mm, seg.end_mm, mutate=True,
                    )
            t_op = sum(s.duration_s for s in h_plan.segments)
            per_op_metrics.append(
                {
                    "op_id": op.op_id,
                    "op_kind": op.op_kind,
                    "tool_number": op.tool_number,
                    "tool_label": op.tool_record.nominal_label,
                    "naive_time_s": round(t_op, 3),
                    "same_speed_time_s": round(t_op, 3),
                    "opt_chipload_time_s": round(t_op, 3),
                }
            )
            continue

        # Milling operation: build bundle & run persistent-IPW DiffCAM-Chipload!
        bundle = _build_milling_op_bundle(part_id, features, op)
        spec = ChiploadSpec.from_feeds_and_speeds(
            bundle,
            stock_material_id=stock_material_id,
            tool_material_id=op.tool_record.tool_material_id,
        )
        # 1. Naive CAM at constant feeds_and_speeds baseline speed (no cross-op IPW shortcutting)
        nom_sub = subdivide_motion_plan(bundle.task, bundle.nominal_motion, max_cut_seg_mm=spec.subdivide_max_mm)
        naive_plan = retime_motion_plan_constant_speed(bundle.task, nom_sub, spec.base_feed_mm_s)

        # 2. Persistent-IPW DiffCAM-Chipload optimization
        # Note: For rest-finishing, z_ipw_opt already has the bulk pocket cleared by T1!
        is_rest = op.op_kind == "pocket_rest_finish"
        res_opt = optimize_chipload_trajectory(
            bundle,
            spec=spec,
            num_iters=opt_iters_per_op,
            verify_posted=not is_rest,  # single-op posted check assumes fresh billet; multi-tool check audits rest ops
            initial_stock_z=z_ipw_opt,
        )
        opt_plan = res_opt.plan
        same_plan = retime_motion_plan_constant_speed(bundle.task, opt_plan, spec.base_feed_mm_s)

        # Advance persistent IPW heightfield using opt_plan
        eng_after = analyze_plan_engagement(
            bundle, opt_plan, spec, initial_stock_z=z_ipw_opt, return_final_stock=True
        )
        z_ipw_opt = eng_after["final_stock_z"]

        naive_seq.append((op, naive_plan))
        same_seq.append((op, same_plan))
        opt_seq.append((op, opt_plan))

        eng_naive = analyze_plan_engagement(bundle, naive_plan, spec)
        eng_same = analyze_plan_engagement(bundle, same_plan, spec)
        per_op_metrics.append(
            {
                "op_id": op.op_id,
                "op_kind": op.op_kind,
                "tool_number": op.tool_number,
                "tool_label": op.tool_record.nominal_label,
                "naive_time_s": eng_naive["summary"]["total_time_s"],
                "same_speed_time_s": eng_same["summary"]["total_time_s"],
                "opt_chipload_time_s": eng_after["summary"]["total_time_s"],
                "max_xy_shift_mm": round(res_opt.max_xy_shift_mm, 3),
            }
        )

    v_naive = _serialize_multitool_variant_for_viewer(
        "nominal", "Naive Multi-Tool CAM (@2)", naive_seq, features, atc_swap_time_s
    )
    v_same = _serialize_multitool_variant_for_viewer(
        "tamp_same_speed", "TAMP + DiffCAM (Same Speed)", same_seq, features, atc_swap_time_s
    )
    v_opt = _serialize_multitool_variant_for_viewer(
        "chipload_ste", "TAMP + DiffCAM (Optimal Chipload)", opt_seq, features, atc_swap_time_s
    )

    unique_tools = len({op.tool_number for op in ops})
    atc_total = max(0, unique_tools - 1) * atc_swap_time_s

    audit = _audit_multitool_safety(features, opt_seq)
    haas_nc = _format_multitool_haas_ngc(program_number, title, opt_seq)

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
        operations=tuple(ops),
        unique_tools_used=unique_tools,
        atc_swap_time_s=round(atc_total, 2),
        naive_cutting_time_s=round(t_naive - atc_total, 3),
        naive_total_time_s=round(t_naive, 3),
        same_speed_cutting_time_s=round(t_same - atc_total, 3),
        same_speed_total_time_s=round(t_same, 3),
        opt_chipload_cutting_time_s=round(t_opt - atc_total, 3),
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
    )
