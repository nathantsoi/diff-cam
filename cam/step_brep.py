# SPDX-License-Identifier: MIT
"""Clean-room ISO 10303-21 STEP B-Rep parser, trimmed UV tessellator, 6-sided setup
orienter, and semantic CNC feature extractor (``cam.step_brep``).

Uses strictly Python stdlib, ``numpy``, and ``scipy`` (100% MIT clean-room; zero GPL
dependencies). Enforces linear-time lexical scanning, path-traversal/symlink safety,
and strict byte/entity caps against untrusted CAD uploads.
"""

from __future__ import annotations

import math
import os
import stat
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import Delaunay

MAX_STEP_BYTES: int = 25 * 1024 * 1024
MAX_STEP_ENTITIES: int = 250_000


@dataclass(frozen=True)
class STEPFaceRecord:
    """Single tessellated B-Rep ``ADVANCED_FACE`` in oriented millimetres."""

    face_id: int
    surface_type: str
    same_sense: bool
    axis_origin_mm: Optional[Tuple[float, float, float]]
    axis_dir: Optional[Tuple[float, float, float]]
    radius_mm: Optional[float]
    semi_angle_rad: Optional[float]
    inward_concave: bool
    triangles_mm: np.ndarray  # (F_i, 3, 3) float64
    area_mm2: float
    bbox_min_mm: Tuple[float, float, float]
    bbox_max_mm: Tuple[float, float, float]


@dataclass(frozen=True)
class BRepHoleFeature:
    """Semantic coaxial cylindrical/counterbored/countersunk/tapped hole feature."""

    side: str  # "op1_top" | "op2_bottom"
    center_xy_mm: Tuple[float, float]
    diameter_mm: float
    z_top_mm: float
    z_bottom_mm: float
    is_through: bool
    counterbore_diameter_mm: Optional[float] = None
    counterbore_depth_mm: Optional[float] = None
    countersink_diameter_mm: Optional[float] = None
    countersink_angle_deg: Optional[float] = None
    thread_spec: Optional[str] = None
    thread_major_mm: Optional[float] = None
    thread_pitch_mm: Optional[float] = None

    def shifted(self, dx: float, dy: float, dz: float) -> BRepHoleFeature:
        return BRepHoleFeature(
            side=self.side,
            center_xy_mm=(round(self.center_xy_mm[0] + dx, 4), round(self.center_xy_mm[1] + dy, 4)),
            diameter_mm=self.diameter_mm,
            z_top_mm=round(self.z_top_mm + dz, 4),
            z_bottom_mm=round(self.z_bottom_mm + dz, 4),
            is_through=self.is_through,
            counterbore_diameter_mm=self.counterbore_diameter_mm,
            counterbore_depth_mm=self.counterbore_depth_mm,
            countersink_diameter_mm=self.countersink_diameter_mm,
            countersink_angle_deg=self.countersink_angle_deg,
            thread_spec=self.thread_spec,
            thread_major_mm=self.thread_major_mm,
            thread_pitch_mm=self.thread_pitch_mm,
        )


@dataclass(frozen=True)
class BRepChamferFeature:
    """Beveled planar or conical chamfer feature extracted from B-Rep faces."""

    side: str  # "op1_top" | "op2_bottom"
    z_top_mm: float
    z_bottom_mm: float
    width_mm: float
    angle_deg: float
    polyline_xy_mm: Tuple[Tuple[float, float], ...]
    is_circular_hole_chamfer: bool = False

    def shifted(self, dx: float, dy: float, dz: float) -> BRepChamferFeature:
        return BRepChamferFeature(
            side=self.side,
            z_top_mm=round(self.z_top_mm + dz, 4),
            z_bottom_mm=round(self.z_bottom_mm + dz, 4),
            width_mm=self.width_mm,
            angle_deg=self.angle_deg,
            polyline_xy_mm=tuple((round(x + dx, 4), round(y + dy, 4)) for x, y in self.polyline_xy_mm),
            is_circular_hole_chamfer=self.is_circular_hole_chamfer,
        )


@dataclass(frozen=True)
class BRepFreeformRegion:
    """Connected 3D sculptured surface region (B-spline / curved crown) requiring ball-endmill surfacing."""

    side: str  # "op1_top" | "op2_bottom"
    min_xy_mm: Tuple[float, float]
    max_xy_mm: Tuple[float, float]
    z_min_mm: float
    z_max_mm: float
    surface_types: Tuple[str, ...]
    area_mm2: float

    def shifted(self, dx: float, dy: float, dz: float) -> BRepFreeformRegion:
        return BRepFreeformRegion(
            side=self.side,
            min_xy_mm=(round(self.min_xy_mm[0] + dx, 4), round(self.min_xy_mm[1] + dy, 4)),
            max_xy_mm=(round(self.max_xy_mm[0] + dx, 4), round(self.max_xy_mm[1] + dy, 4)),
            z_min_mm=round(self.z_min_mm + dz, 4),
            z_max_mm=round(self.z_max_mm + dz, 4),
            surface_types=self.surface_types,
            area_mm2=self.area_mm2,
        )


@dataclass(frozen=True)
class STEPLoadResult:
    """Complete result of STEP B-Rep loading, unit scaling, 6-axis orientation, and feature extraction."""

    part_name: str
    source_unit: str
    unit_to_mm: float
    raw_bounds_min_mm: Tuple[float, float, float]
    raw_bounds_max_mm: Tuple[float, float, float]
    setup_rotation: np.ndarray  # (3, 3) SO(3)
    setup_axis_label: str  # "+Z", "+Y->+Z", etc.
    oriented_triangles_mm: np.ndarray  # (F, 3, 3) float64
    oriented_bounds_min_mm: Tuple[float, float, float]
    oriented_bounds_max_mm: Tuple[float, float, float]
    stl_bytes: bytes
    faces: Tuple[STEPFaceRecord, ...]
    op1_holes: Tuple[BRepHoleFeature, ...]
    op2_holes: Tuple[BRepHoleFeature, ...]
    op1_chamfers: Tuple[BRepChamferFeature, ...]
    op2_chamfers: Tuple[BRepChamferFeature, ...]
    op1_freeform: Tuple[BRepFreeformRegion, ...]
    op2_freeform: Tuple[BRepFreeformRegion, ...]
    requires_op2_flip: bool


def is_step_payload(raw: bytes) -> bool:
    """Return True if ``raw`` begins with an ISO-10303-21 STEP header within the first 2 KiB."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) < 32:
        return False
    head = bytes(raw[:2048]).decode("latin-1", errors="ignore").upper()
    return "ISO-10303-21" in head and ("HEADER;" in head or "FILE_SCHEMA" in head)


def triangles_to_binary_stl_bytes(triangles_mm: np.ndarray) -> bytes:
    """Serialize an ``(F, 3, 3)`` triangle array in mm to IEEE-754 little-endian binary STL."""
    arr = np.asarray(triangles_mm, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3) or len(arr) == 0:
        raise ValueError("triangles_mm must have shape (F, 3, 3) with F >= 1")
    v0, v1, v2 = arr[:, 0], arr[:, 1], arr[:, 2]
    nrm = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.maximum(nlen, 1e-12))
    dt = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")])
    out = np.zeros(len(arr), dtype=dt)
    out["normal"] = nrm
    out["verts"] = arr
    header = b"DiffCAM Clean-Room MIT STEP B-Rep Binary STL".ljust(80, b"\x00")
    return header + struct.pack("<I", len(arr)) + out.tobytes()


# ---------------------------------------------------------------------------
# Linear-Time ISO 10303-21 Scanner & Entity Store
# ---------------------------------------------------------------------------
def _strip_step_comments_and_extract_data(text: str) -> str:
    """Strip ``/* ... */`` comments in O(N) while respecting single-quoted strings, then slice DATA..ENDSEC."""
    out_chars: List[str] = []
    n = len(text)
    i = 0
    in_str = False
    in_comment = 0
    while i < n:
        ch = text[i]
        if in_comment > 0:
            if ch == "/" and i + 1 < n and text[i + 1] == "*":
                in_comment += 1
                i += 2
            elif ch == "*" and i + 1 < n and text[i + 1] == "/":
                in_comment -= 1
                i += 2
            else:
                i += 1
            continue
        if in_str:
            out_chars.append(ch)
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    out_chars.append("'")
                    i += 2
                    continue
                in_str = False
            i += 1
            continue
        if ch == "'":
            in_str = True
            out_chars.append(ch)
            i += 1
        elif ch == "/" and i + 1 < n and text[i + 1] == "*":
            in_comment = 1
            i += 2
        else:
            out_chars.append(ch)
            i += 1

    clean = "".join(out_chars)
    idx_data = clean.find("DATA;")
    if idx_data < 0:
        raise ValueError("STEP file missing DATA; section")
    idx_end = clean.find("ENDSEC;", idx_data + 5)
    if idx_end < 0:
        raise ValueError("STEP file missing ENDSEC; terminator for DATA section")
    return clean[idx_data + 5 : idx_end]


def _parse_step_arglist(arg_str: str) -> List[Any]:
    """Iterative stack-based parser for STEP parameter lists (handles nested tuples, strings, #refs, enums)."""
    stack: List[List[Any]] = [[]]
    n = len(arg_str)
    i = 0
    while i < n:
        ch = arg_str[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == ",":
            i += 1
            continue
        if ch == "(":
            new_list: List[Any] = []
            stack[-1].append(new_list)
            stack.append(new_list)
            i += 1
            continue
        if ch == ")":
            if len(stack) > 1:
                stack.pop()
            i += 1
            continue
        if ch == "'":
            j = i + 1
            buf: List[str] = []
            while j < n:
                if arg_str[j] == "'":
                    if j + 1 < n and arg_str[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    j += 1
                    break
                buf.append(arg_str[j])
                j += 1
            stack[-1].append("".join(buf))
            i = j
            continue
        # Read bare token until ',', '(', ')', or whitespace
        j = i
        while j < n and arg_str[j] not in ",() \t\r\n":
            j += 1
        tok = arg_str[i:j]
        # Check for inline typed parameter like LENGTH_MEASURE(0.0393700787401575)
        k = j
        while k < n and arg_str[k] in " \t\r\n":
            k += 1
        if k < n and arg_str[k] == "(" and tok and tok[0].isalpha():
            # Find matching ')' for inline typed value
            depth = 1
            m = k + 1
            in_q = False
            while m < n and depth > 0:
                c2 = arg_str[m]
                if in_q:
                    if c2 == "'":
                        if m + 1 < n and arg_str[m + 1] == "'":
                            m += 2
                            continue
                        in_q = False
                else:
                    if c2 == "'":
                        in_q = True
                    elif c2 == "(":
                        depth += 1
                    elif c2 == ")":
                        depth -= 1
                m += 1
            inner = _parse_step_arglist(arg_str[k + 1 : m - 1])
            stack[-1].append((tok.upper(), inner))
            i = m
            continue

        if tok == "$" or tok == "*":
            stack[-1].append(None)
        elif tok.startswith("#"):
            stack[-1].append(int(tok[1:]))
        elif tok == ".T.":
            stack[-1].append(True)
        elif tok == ".F.":
            stack[-1].append(False)
        elif tok.startswith(".") and tok.endswith("."):
            stack[-1].append(tok.upper())
        else:
            try:
                if any(c in tok for c in ".Ee+-") and not tok.lstrip("+-").isdigit():
                    val_f = float(tok)
                    if not math.isfinite(val_f):
                        raise ValueError("Non-finite float in STEP parameter")
                    stack[-1].append(val_f)
                else:
                    stack[-1].append(int(tok))
            except ValueError:
                val_f = float(tok)
                if not math.isfinite(val_f):
                    raise ValueError("Non-finite float in STEP parameter")
                stack[-1].append(val_f)
        i = j
    return stack[0]


class _STEPDocument:
    """Indexed, demand-parsed ISO 10303-21 entity graph with unit scaling."""

    def __init__(self, raw_bytes: bytes) -> None:
        if len(raw_bytes) > MAX_STEP_BYTES:
            raise ValueError(f"STEP input exceeds {MAX_STEP_BYTES} byte safety cap")
        if b"\x00" in raw_bytes:
            raise ValueError("STEP input contains forbidden NUL bytes")
        if not is_step_payload(raw_bytes):
            raise ValueError("Input is not a valid ISO-10303-21 STEP file")

        text = raw_bytes.decode("latin-1")
        data_sec = _strip_step_comments_and_extract_data(text)

        self.raw_entities: Dict[int, str] = {}
        # Linear statement splitter on ';' outside single quotes
        n = len(data_sec)
        start = 0
        in_str = False
        i = 0
        while i < n:
            ch = data_sec[i]
            if in_str:
                if ch == "'":
                    if i + 1 < n and data_sec[i + 1] == "'":
                        i += 2
                        continue
                    in_str = False
            elif ch == "'":
                in_str = True
            elif ch == ";":
                stmt = data_sec[start:i].strip()
                start = i + 1
                if stmt.startswith("#"):
                    eq = stmt.find("=")
                    if eq > 1:
                        eid = int(stmt[1:eq].strip())
                        body = stmt[eq + 1 :].strip()
                        self.raw_entities[eid] = body
                        if len(self.raw_entities) > MAX_STEP_ENTITIES:
                            raise ValueError(f"STEP entity count exceeds {MAX_STEP_ENTITIES}")
            i += 1

        if not self.raw_entities:
            raise ValueError("STEP DATA section contains zero entities")

        self._parsed: Dict[int, Tuple[str, Any]] = {}
        self._pts_mm: Dict[int, np.ndarray] = {}
        self._dirs: Dict[int, np.ndarray] = {}
        self._verts_mm: Dict[int, np.ndarray] = {}
        self._axes: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        self.source_unit, self.unit_to_mm, self.angle_to_rad = self._resolve_units()
        self.part_name = self._resolve_part_name()

    def get(self, eid: int) -> Tuple[str, Any]:
        """Return ``(TYPE_NAME, args_list)`` or ``('__COMPLEX__', {SUBTYPE: args_list})``."""
        cached = self._parsed.get(eid)
        if cached is not None:
            return cached
        body = self.raw_entities.get(eid)
        if body is None:
            raise KeyError(f"Missing STEP entity #{eid}")
        if body.startswith("(") and body.endswith(")"):
            # Complex entity instance: ( TYPE_A(...) TYPE_B(...) ... )
            items = _parse_step_arglist(body[1:-1])
            submap: Dict[str, List[Any]] = {}
            for item in items:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                    submap[item[0]] = item[1]
            res = ("__COMPLEX__", submap)
        else:
            p0 = body.find("(")
            if p0 < 0 or not body.endswith(")"):
                res = (body.upper(), [])
            else:
                ename = body[:p0].strip().upper()
                args = _parse_step_arglist(body[p0 + 1 : -1])
                res = (ename, args)
        self._parsed[eid] = res
        return res

    def _resolve_units(self) -> Tuple[str, float, float]:
        unit_name = "mm"
        unit_to_mm = 1.0
        angle_to_rad = 1.0

        si_prefix_scale = {
            ".MILLI.": 1.0,
            ".CENTI.": 10.0,
            ".DECI.": 100.0,
            None: 1000.0,
            ".MICRO.": 1e-3,
            ".KILO.": 1e6,
        }

        # Find GLOBAL_UNIT_ASSIGNED_CONTEXT if present
        unit_refs: List[int] = []
        for eid, body in self.raw_entities.items():
            if "GLOBAL_UNIT_ASSIGNED_CONTEXT" in body:
                etype, data = self.get(eid)
                if etype == "__COMPLEX__" and "GLOBAL_UNIT_ASSIGNED_CONTEXT" in data:
                    u_arg = data["GLOBAL_UNIT_ASSIGNED_CONTEXT"][0]
                    if isinstance(u_arg, list):
                        unit_refs.extend(x for x in u_arg if isinstance(x, int))
                elif etype == "GLOBAL_UNIT_ASSIGNED_CONTEXT" and data and isinstance(data[0], list):
                    unit_refs.extend(x for x in data[0] if isinstance(x, int))
                if unit_refs:
                    break

        scan_ids = unit_refs if unit_refs else list(self.raw_entities.keys())
        for uid in scan_ids:
            body = self.raw_entities.get(uid, "")
            if "UNIT" not in body:
                continue
            etype, data = self.get(uid)
            sub = data if etype == "__COMPLEX__" else {etype: data}
            if "LENGTH_UNIT" in sub:
                if "CONVERSION_BASED_UNIT" in sub:
                    c_args = sub["CONVERSION_BASED_UNIT"]
                    cname = str(c_args[0]).strip().lower() if c_args else "inch"
                    meas_ref = c_args[1] if len(c_args) > 1 and isinstance(c_args[1], int) else None
                    if "inch" in cname or cname == "in":
                        unit_name = "inch"
                        unit_to_mm = 25.4
                    elif "foot" in cname or "feet" in cname or cname == "ft":
                        unit_name = "ft"
                        unit_to_mm = 304.8
                    elif meas_ref is not None:
                        _, m_args = self.get(meas_ref)
                        if m_args:
                            val = m_args[0][1][0] if isinstance(m_args[0], tuple) else m_args[0]
                            base_ref = m_args[1] if len(m_args) > 1 and isinstance(m_args[1], int) else None
                            base_mm = 1.0
                            if base_ref is not None:
                                _, b_sub_raw = self.get(base_ref)
                                b_sub = b_sub_raw if isinstance(b_sub_raw, dict) else {"SI_UNIT": b_sub_raw}
                                if "SI_UNIT" in b_sub and b_sub["SI_UNIT"]:
                                    base_mm = si_prefix_scale.get(b_sub["SI_UNIT"][0], 1000.0)
                            unit_to_mm = float(val) * base_mm
                            unit_name = "inch" if abs(unit_to_mm - 25.4) < 0.05 else cname
                elif "SI_UNIT" in sub:
                    s_args = sub["SI_UNIT"]
                    pref = s_args[0] if s_args else None
                    unit_to_mm = si_prefix_scale.get(pref, 1.0)
                    unit_name = "mm" if abs(unit_to_mm - 1.0) < 1e-6 else ("m" if abs(unit_to_mm - 1000.0) < 1e-3 else "si")
            if "PLANE_ANGLE_UNIT" in sub:
                if "CONVERSION_BASED_UNIT" in sub:
                    c_args = sub["CONVERSION_BASED_UNIT"]
                    cname = str(c_args[0]).strip().lower() if c_args else ""
                    if "deg" in cname:
                        angle_to_rad = math.pi / 180.0
                    elif len(c_args) > 1 and isinstance(c_args[1], int):
                        _, m_args = self.get(c_args[1])
                        if m_args:
                            val = m_args[0][1][0] if isinstance(m_args[0], tuple) else m_args[0]
                            angle_to_rad = float(val)
                elif "SI_UNIT" in sub:
                    angle_to_rad = 1.0

        return unit_name, unit_to_mm, angle_to_rad

    def _resolve_part_name(self) -> str:
        for eid, body in self.raw_entities.items():
            if body.startswith("PRODUCT("):
                _, args = self.get(eid)
                for a in args[:2]:
                    if isinstance(a, str) and a.strip() and a.strip().upper() not in ("NONE", "UNSPECIFIED"):
                        return a.strip()
        return "STEP_Part"

    def point_mm(self, eid: int) -> np.ndarray:
        pt = self._pts_mm.get(eid)
        if pt is not None:
            return pt
        etype, args = self.get(eid)
        if etype != "CARTESIAN_POINT":
            raise ValueError(f"Expected CARTESIAN_POINT at #{eid}, got {etype}")
        coords = args[1]
        arr = np.array([float(coords[0]), float(coords[1]), float(coords[2])], dtype=np.float64) * self.unit_to_mm
        self._pts_mm[eid] = arr
        return arr

    def direction(self, eid: int) -> np.ndarray:
        d = self._dirs.get(eid)
        if d is not None:
            return d
        etype, args = self.get(eid)
        if etype != "DIRECTION":
            raise ValueError(f"Expected DIRECTION at #{eid}, got {etype}")
        coords = args[1]
        arr = np.array([float(coords[0]), float(coords[1]), float(coords[2])], dtype=np.float64)
        nrm = float(np.linalg.norm(arr))
        if nrm > 1e-12:
            arr = arr / nrm
        self._dirs[eid] = arr
        return arr

    def vertex_mm(self, eid: int) -> np.ndarray:
        v = self._verts_mm.get(eid)
        if v is not None:
            return v
        etype, args = self.get(eid)
        if etype != "VERTEX_POINT":
            raise ValueError(f"Expected VERTEX_POINT at #{eid}, got {etype}")
        arr = self.point_mm(int(args[1]))
        self._verts_mm[eid] = arr
        return arr

    def axis2_placement_3d(self, eid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return right-handed orthonormal ``(origin_mm, z_axis, x_axis)`` for ``AXIS2_PLACEMENT_3D``."""
        ax = self._axes.get(eid)
        if ax is not None:
            return ax
        etype, args = self.get(eid)
        if etype != "AXIS2_PLACEMENT_3D":
            raise ValueError(f"Expected AXIS2_PLACEMENT_3D at #{eid}, got {etype}")
        orig = self.point_mm(int(args[1]))
        z_dir = self.direction(int(args[2])) if (len(args) > 2 and isinstance(args[2], int)) else np.array([0.0, 0.0, 1.0])
        if len(args) > 3 and isinstance(args[3], int):
            x_raw = self.direction(int(args[3]))
        else:
            ref = np.array([1.0, 0.0, 0.0]) if abs(z_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_raw = ref
        # Gram-Schmidt orthogonalize x_dir against z_dir
        x_proj = x_raw - float(np.dot(x_raw, z_dir)) * z_dir
        xn = float(np.linalg.norm(x_proj))
        if xn <= 1e-12:
            ref = np.array([1.0, 0.0, 0.0]) if abs(z_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_proj = ref - float(np.dot(ref, z_dir)) * z_dir
            xn = float(np.linalg.norm(x_proj))
        x_dir = x_proj / max(xn, 1e-12)
        res = (orig, z_dir, x_dir)
        self._axes[eid] = res
        return res


# ---------------------------------------------------------------------------
# Exact Curve, B-Spline, and Trimmed UV Surface Tessellation
# ---------------------------------------------------------------------------
def _expand_knots(mults: Sequence[int], knots: Sequence[float]) -> np.ndarray:
    pieces: List[float] = []
    for m, kv in zip(mults, knots):
        pieces.extend([float(kv)] * int(m))
    return np.asarray(pieces, dtype=np.float64)


def _bspline_basis_and_deriv(u_vals: np.ndarray, degree: int, knot_vec: np.ndarray, n_ctrl: int) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate Cox-de Boor basis ``N_{i,p}(u)`` and first derivative ``dN_{i,p}/du`` for array ``u_vals``."""
    u_min = float(knot_vec[degree])
    u_max = float(knot_vec[n_ctrl])
    u = np.clip(np.asarray(u_vals, dtype=np.float64), u_min, u_max)
    m_pts = len(u)
    n_spans = len(knot_vec) - 1

    basis = np.zeros((m_pts, n_spans), dtype=np.float64)
    for i in range(n_spans):
        t0, t1 = knot_vec[i], knot_vec[i + 1]
        if t1 > t0:
            if i == n_ctrl - 1 or abs(t1 - u_max) <= 1e-13:
                mask = (u >= t0) & (u <= t1)
            else:
                mask = (u >= t0) & (u < t1)
            basis[mask, i] = 1.0

    prev_basis = basis.copy()
    for d in range(1, degree + 1):
        prev_basis = basis.copy()
        next_b = np.zeros((m_pts, n_spans - d), dtype=np.float64)
        for i in range(n_spans - d):
            denom1 = knot_vec[i + d] - knot_vec[i]
            if denom1 > 1e-14:
                next_b[:, i] += ((u - knot_vec[i]) / denom1) * basis[:, i]
            denom2 = knot_vec[i + d + 1] - knot_vec[i + 1]
            if denom2 > 1e-14:
                next_b[:, i] += ((knot_vec[i + d + 1] - u) / denom2) * basis[:, i + 1]
        basis = next_b

    deriv = np.zeros((m_pts, n_ctrl), dtype=np.float64)
    if degree >= 1:
        for i in range(n_ctrl):
            denom1 = knot_vec[i + degree] - knot_vec[i]
            if denom1 > 1e-14:
                deriv[:, i] += (degree / denom1) * prev_basis[:, i]
            denom2 = knot_vec[i + degree + 1] - knot_vec[i + 1]
            if denom2 > 1e-14:
                deriv[:, i] -= (degree / denom2) * prev_basis[:, i + 1]

    return basis[:, :n_ctrl], deriv


def _eval_bspline_curve(
    ctrl_pts: np.ndarray,
    degree: int,
    knot_vec: np.ndarray,
    u_vals: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate polynomial or rational NURBS curve at parameter array ``u_vals`` -> ``(M, 3)``."""
    basis, _ = _bspline_basis_and_deriv(u_vals, degree, knot_vec, len(ctrl_pts))
    if weights is not None:
        wb = basis * weights[None, :]
        denom = np.maximum(np.sum(wb, axis=1, keepdims=True), 1e-14)
        return (wb @ ctrl_pts) / denom
    return basis @ ctrl_pts


def _eval_bspline_surface_pts_and_partials(
    ctrl_grid: np.ndarray,  # (Nu, Nv, 3)
    u_deg: int,
    v_deg: int,
    u_knots: np.ndarray,
    v_knots: np.ndarray,
    u_vals: np.ndarray,
    v_vals: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate tensor-product B-spline/NURBS surface ``S(u_k, v_k)`` and partials ``(dS/du, dS/dv)``."""
    nu, nv, _ = ctrl_grid.shape
    bu, dbu = _bspline_basis_and_deriv(u_vals, u_deg, u_knots, nu)
    bv, dbv = _bspline_basis_and_deriv(v_vals, v_deg, v_knots, nv)

    if weights is None:
        pts = np.einsum("ki,kj,ijc->kc", bu, bv, ctrl_grid)
        dsu = np.einsum("ki,kj,ijc->kc", dbu, bv, ctrl_grid)
        dsv = np.einsum("ki,kj,ijc->kc", bu, dbv, ctrl_grid)
        return pts, dsu, dsv

    w_grid = ctrl_grid * weights[:, :, None]
    num = np.einsum("ki,kj,ijc->kc", bu, bv, w_grid)
    den = np.maximum(np.einsum("ki,kj,ij->k", bu, bv, weights)[:, None], 1e-14)
    pts = num / den
    dnum_u = np.einsum("ki,kj,ijc->kc", dbu, bv, w_grid)
    dden_u = np.einsum("ki,kj,ij->k", dbu, bv, weights)[:, None]
    dnum_v = np.einsum("ki,kj,ijc->kc", bu, dbv, w_grid)
    dden_v = np.einsum("ki,kj,ij->k", bu, dbv, weights)[:, None]
    dsu = (dnum_u - pts * dden_u) / den
    dsv = (dnum_v - pts * dden_v) / den
    return pts, dsu, dsv


def _discretize_edge_curve(
    doc: _STEPDocument,
    edge_curve_id: int,
    edge_same_sense: bool,
    oriented_flag: bool,
    chord_tol_mm: float = 0.08,
) -> Tuple[np.ndarray, str, float]:
    """Discretize ``EDGE_CURVE`` into ordered 3D points ``(K, 3)``, curve type, and angular span (rad)."""
    etype, args = doc.get(edge_curve_id)
    if etype != "EDGE_CURVE":
        raise ValueError(f"Expected EDGE_CURVE at #{edge_curve_id}, got {etype}")
    v_start_id, v_end_id, curve_id, curve_same = int(args[1]), int(args[2]), int(args[3]), bool(args[4])
    p_start = doc.vertex_mm(v_start_id)
    p_end = doc.vertex_mm(v_end_id)
    is_closed_edge = (v_start_id == v_end_id) or (float(np.linalg.norm(p_end - p_start)) <= 1e-6)

    # Unwrap SURFACE_CURVE / SEAM_CURVE -> underlying 3D curve
    ctype, cdata = doc.get(curve_id)
    while ctype in ("SURFACE_CURVE", "SEAM_CURVE"):
        curve_id = int(cdata[1])
        ctype, cdata = doc.get(curve_id)

    forward_curve = curve_same if edge_same_sense else (not curve_same)
    span_rad = 0.0

    if ctype == "LINE":
        pts = np.vstack([p_start, p_end])
        if not oriented_flag:
            pts = pts[::-1].copy()
        return pts, "LINE", 0.0

    if ctype in ("CIRCLE", "ELLIPSE"):
        orig, z_ax, x_ax = doc.axis2_placement_3d(int(cdata[1]))
        y_ax = np.cross(z_ax, x_ax)
        r1 = float(cdata[2]) * doc.unit_to_mm
        r2 = float(cdata[3]) * doc.unit_to_mm if ctype == "ELLIPSE" else r1
        d0 = p_start - orig
        d1 = p_end - orig
        th0 = math.atan2(float(np.dot(d0, y_ax)) / max(r2, 1e-12), float(np.dot(d0, x_ax)) / max(r1, 1e-12))
        th1 = math.atan2(float(np.dot(d1, y_ax)) / max(r2, 1e-12), float(np.dot(d1, x_ax)) / max(r1, 1e-12))
        if is_closed_edge:
            dth = 2.0 * math.pi if forward_curve else -2.0 * math.pi
        else:
            if forward_curve:
                dth = (th1 - th0) % (2.0 * math.pi)
                if dth <= 1e-7:
                    dth = 2.0 * math.pi
            else:
                dth = -((th0 - th1) % (2.0 * math.pi))
                if abs(dth) <= 1e-7:
                    dth = -2.0 * math.pi
        span_rad = abs(dth)
        r_max = max(r1, r2)
        max_step = 2.0 * math.acos(max(0.0, 1.0 - chord_tol_mm / max(r_max, chord_tol_mm + 1e-6)))
        max_step = max(0.08, min(0.32, max_step))
        n_seg = max(4 if span_rad > 0.8 else 2, min(96, int(math.ceil(span_rad / max_step))))
        ths = np.linspace(th0, th0 + dth, n_seg + 1)
        pts = (
            orig[None, :]
            + (r1 * np.cos(ths))[:, None] * x_ax[None, :]
            + (r2 * np.sin(ths))[:, None] * y_ax[None, :]
        )
        pts[0] = p_start
        pts[-1] = p_end
        if not oriented_flag:
            pts = pts[::-1].copy()
        return pts, ctype, span_rad

    # B_SPLINE_CURVE_WITH_KNOTS or complex rational B-spline curve
    sub = cdata if ctype == "__COMPLEX__" else {ctype: cdata}
    if "B_SPLINE_CURVE_WITH_KNOTS" in sub or "B_SPLINE_CURVE" in sub:
        if "B_SPLINE_CURVE" in sub:
            bs = sub["B_SPLINE_CURVE"]
            deg = int(bs[0])
            cp_ids = bs[1]
            bsk = sub["B_SPLINE_CURVE_WITH_KNOTS"]
            mults, knots = bsk[0], bsk[1]
        else:
            bsk = sub["B_SPLINE_CURVE_WITH_KNOTS"]
            deg = int(bsk[1])
            cp_ids = bsk[2]
            mults, knots = bsk[6], bsk[7]
        weights = None
        if "RATIONAL_B_SPLINE_CURVE" in sub:
            weights = np.asarray(sub["RATIONAL_B_SPLINE_CURVE"][0], dtype=np.float64)
        ctrl_pts = np.asarray([doc.point_mm(int(cid)) for cid in cp_ids], dtype=np.float64)
        kvec = _expand_knots(mults, knots)
        u_lo, u_hi = float(kvec[deg]), float(kvec[len(ctrl_pts)])
        # Project endpoints onto parameter interval
        n_dense = max(32, min(256, 8 * len(ctrl_pts)))
        u_grid = np.linspace(u_lo, u_hi, n_dense)
        c_grid = _eval_bspline_curve(ctrl_pts, deg, kvec, u_grid, weights=weights)
        i0 = int(np.argmin(np.linalg.norm(c_grid - p_start[None, :], axis=1)))
        i1 = int(np.argmin(np.linalg.norm(c_grid - p_end[None, :], axis=1)))
        u0, u1 = float(u_grid[i0]), float(u_grid[i1])
        if is_closed_edge and abs(u1 - u0) <= 1e-9:
            u0, u1 = (u_lo, u_hi) if forward_curve else (u_hi, u_lo)
        arc_len = float(np.sum(np.linalg.norm(np.diff(c_grid, axis=0), axis=1)))
        n_seg = max(3, min(64, int(math.ceil(arc_len / max(chord_tol_mm * 3.0, 0.4)))))
        u_samp = np.linspace(u0, u1, n_seg + 1)
        pts = _eval_bspline_curve(ctrl_pts, deg, kvec, u_samp, weights=weights)
        pts[0] = p_start
        pts[-1] = p_end
        if not oriented_flag:
            pts = pts[::-1].copy()
        return pts, "B_SPLINE_CURVE", 0.0

    # Fallback chord
    pts = np.vstack([p_start, p_end])
    if not oriented_flag:
        pts = pts[::-1].copy()
    return pts, ctype, 0.0


def _points_in_polygon_2d(pts_xy: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test for ``pts_xy`` ``(M, 2)`` in closed ``poly_xy`` ``(K, 2)``."""
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    px = poly_xy[:, 0]
    py = poly_xy[:, 1]
    px_next = np.roll(px, -1)
    py_next = np.roll(py, -1)
    inside = np.zeros(len(pts_xy), dtype=bool)
    for i in range(len(poly_xy)):
        y0, y1 = py[i], py_next[i]
        x0, x1 = px[i], px_next[i]
        cond = (y0 > y) != (y1 > y)
        if np.any(cond):
            x_int = x0 + (y[cond] - y0) * (x1 - x0) / (y1 - y0)
            inside[cond] ^= x[cond] < x_int
    return inside


def _min_dist_to_poly_edges_2d(pts_xy: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    """Minimum Euclidean distance from each point in ``pts_xy`` ``(M, 2)`` to segments of closed ``poly_xy``."""
    a = poly_xy
    b = np.roll(poly_xy, -1, axis=0)
    ab = b - a
    ab_len2 = np.maximum(np.sum(ab * ab, axis=1), 1e-16)
    min_d2 = np.full(len(pts_xy), np.inf, dtype=np.float64)
    for i in range(len(poly_xy)):
        ap = pts_xy - a[i][None, :]
        t = np.clip((ap @ ab[i]) / ab_len2[i], 0.0, 1.0)
        proj = a[i][None, :] + t[:, None] * ab[i][None, :]
        d2 = np.sum((pts_xy - proj) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return np.sqrt(min_d2)


def _signed_area_2d(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clean_polygon_2d_and_idx(poly: np.ndarray, orig_idx: np.ndarray, eps: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
    if len(poly) == 0:
        return poly, orig_idx
    keep_p = [poly[0]]
    keep_i = [int(orig_idx[0])]
    for k in range(1, len(poly)):
        if float(np.linalg.norm(poly[k] - keep_p[-1])) > eps:
            keep_p.append(poly[k])
            keep_i.append(int(orig_idx[k]))
    if len(keep_p) > 1 and float(np.linalg.norm(keep_p[-1] - keep_p[0])) <= eps:
        keep_p.pop()
        keep_i.pop()
    return np.asarray(keep_p, dtype=np.float64), np.asarray(keep_i, dtype=np.int64)


def _earclip_polygon_with_holes(
    outer_uv: np.ndarray,
    outer_idx: np.ndarray,
    holes_uv_and_idx: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Bridge inner hole rings into ``outer_uv`` (Eberly horizontal ray) and ear-clip to index triples."""
    poly, pidx = _clean_polygon_2d_and_idx(outer_uv, outer_idx)
    if len(poly) < 3:
        return np.zeros((0, 3), dtype=np.int64)
    if _signed_area_2d(poly) < 0.0:
        poly = poly[::-1].copy()
        pidx = pidx[::-1].copy()

    # Sort holes by rightmost X descending and splice via mutual-visibility bridge
    cleaned_holes: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for huv, hidx in holes_uv_and_idx:
        hp, hi = _clean_polygon_2d_and_idx(huv, hidx)
        if len(hp) >= 3:
            if _signed_area_2d(hp) > 0.0:
                hp = hp[::-1].copy()
                hi = hi[::-1].copy()
            cleaned_holes.append((float(np.max(hp[:, 0])), hp, hi))
    cleaned_holes.sort(key=lambda item: -item[0])

    for _, hp, hi in cleaned_holes:
        m_idx = int(np.argmax(hp[:, 0]))
        mx, my = float(hp[m_idx, 0]), float(hp[m_idx, 1])
        # Find closest intersection of horizontal ray (mx + t, my), t >= 0, with current outer poly
        best_t = float("inf")
        best_edge_v = 0
        n_p = len(poly)
        for i in range(n_p):
            p0, p1 = poly[i], poly[(i + 1) % n_p]
            if (p0[1] > my) != (p1[1] > my):
                x_int = p0[0] + (my - p0[1]) * (p1[0] - p0[0]) / (p1[1] - p0[1])
                if x_int >= mx - 1e-9 and (x_int - mx) < best_t:
                    best_t = x_int - mx
                    best_edge_v = i if p0[0] >= p1[0] else (i + 1) % n_p
        if not math.isfinite(best_t):
            best_edge_v = int(np.argmin(np.sum((poly - hp[m_idx][None, :]) ** 2, axis=1)))
        else:
            # Check reflex vertices inside triangle (M, I, P_vis)
            ix_pt = np.array([mx + best_t, my], dtype=np.float64)
            p_vis = poly[best_edge_v]
            tri = np.vstack([hp[m_idx], ix_pt, p_vis])
            if abs(_signed_area_2d(tri)) > 1e-11:
                if _signed_area_2d(tri) < 0.0:
                    tri = tri[::-1]
                best_ang = float("inf")
                for k in range(n_p):
                    if k == best_edge_v:
                        continue
                    prv = poly[(k - 1) % n_p]
                    cur = poly[k]
                    nxt = poly[(k + 1) % n_p]
                    cross = (cur[0] - prv[0]) * (nxt[1] - cur[1]) - (cur[1] - prv[1]) * (nxt[0] - cur[0])
                    if cross < 0.0 and _points_in_polygon_2d(cur[None, :], tri)[0]:
                        vec = cur - hp[m_idx]
                        ang = abs(math.atan2(vec[1], max(vec[0], 1e-12)))
                        if ang < best_ang:
                            best_ang = ang
                            best_edge_v = k

        # Splice hole starting at m_idx into poly at best_edge_v
        h_rot_p = np.roll(hp, -m_idx, axis=0)
        h_rot_i = np.roll(hi, -m_idx, axis=0)
        poly = np.vstack([poly[: best_edge_v + 1], h_rot_p, h_rot_p[:1], poly[best_edge_v:]])
        pidx = np.concatenate([pidx[: best_edge_v + 1], h_rot_i, h_rot_i[:1], pidx[best_edge_v:]])

    # Ear-clip merged simple polygon
    active = list(range(len(poly)))
    tris: List[Tuple[int, int, int]] = []
    tol_list = (1e-10, -1e-9, -1e-6)
    for tol in tol_list:
        progress = True
        while len(active) > 3 and progress:
            progress = False
            m_len = len(active)
            for pos in range(m_len):
                ia = active[(pos - 1) % m_len]
                ib = active[pos]
                ic = active[(pos + 1) % m_len]
                pa, pb, pc = poly[ia], poly[ib], poly[ic]
                cross = (pb[0] - pa[0]) * (pc[1] - pb[1]) - (pb[1] - pa[1]) * (pc[0] - pb[0])
                if cross <= tol:
                    continue
                if cross > 1e-12:
                    ear_tri = np.vstack([pa, pb, pc])
                    other_pts = np.asarray(
                        [poly[idx] for idx in active if idx not in (ia, ib, ic) and pidx[idx] not in (pidx[ia], pidx[ib], pidx[ic])],
                        dtype=np.float64,
                    )
                    if len(other_pts) > 0:
                        w0_den = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
                        if abs(w0_den) > 1e-14:
                            w0 = ((pb[1] - pc[1]) * (other_pts[:, 0] - pc[0]) + (pc[0] - pb[0]) * (other_pts[:, 1] - pc[1])) / w0_den
                            w1 = ((pc[1] - pa[1]) * (other_pts[:, 0] - pc[0]) + (pa[0] - pc[0]) * (other_pts[:, 1] - pc[1])) / w0_den
                            w2 = 1.0 - w0 - w1
                            if np.any((w0 > 1e-8) & (w1 > 1e-8) & (w2 > 1e-8)):
                                continue
                if pidx[ia] != pidx[ib] and pidx[ib] != pidx[ic] and pidx[ia] != pidx[ic]:
                    tris.append((int(pidx[ia]), int(pidx[ib]), int(pidx[ic])))
                active.pop(pos)
                progress = True
                break
        if len(active) <= 3:
            break

    if len(active) == 3:
        ia, ib, ic = active[0], active[1], active[2]
        if pidx[ia] != pidx[ib] and pidx[ib] != pidx[ic] and pidx[ia] != pidx[ic]:
            tris.append((int(pidx[ia]), int(pidx[ib]), int(pidx[ic])))
    return np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), dtype=np.int64)


def _tessellate_trimmed_uv_delaunay(
    outer_uv: np.ndarray,
    outer_3d: np.ndarray,
    holes_uv_3d: Sequence[Tuple[np.ndarray, np.ndarray]],
    eval_fn: Any,  # (u_arr, v_arr) -> (pts_3d, normals_3d)
    u_step: float,
    v_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Delaunay triangulation in scaled UV space with exact boundary 3D preservation + interior grid."""
    all_uv_list: List[np.ndarray] = [outer_uv]
    all_3d_list: List[np.ndarray] = [outer_3d]
    hole_uvs: List[np.ndarray] = []
    for huv, h3d in holes_uv_3d:
        all_uv_list.append(huv)
        all_3d_list.append(h3d)
        hole_uvs.append(huv)

    bnd_uv = np.vstack(all_uv_list)
    bnd_3d = np.vstack(all_3d_list)
    umin, vmin = np.min(outer_uv, axis=0)
    umax, vmax = np.max(outer_uv, axis=0)
    nu = max(2, min(40, int(math.ceil((umax - umin) / max(u_step, 1e-4)))))
    nv = max(2, min(40, int(math.ceil((vmax - vmin) / max(v_step, 1e-4)))))

    if nu >= 3 or nv >= 3:
        ug = np.linspace(umin, umax, nu + 1)[1:-1]
        vg = np.linspace(vmin, vmax, nv + 1)[1:-1]
        if len(ug) > 0 and len(vg) > 0:
            gu, gv = np.meshgrid(ug, vg, indexing="ij")
            cand_uv = np.column_stack([gu.ravel(), gv.ravel()])
            keep = _points_in_polygon_2d(cand_uv, outer_uv)
            min_clear = 0.28 * min(u_step, v_step)
            if np.any(keep):
                keep_idx = np.where(keep)[0]
                d_out = _min_dist_to_poly_edges_2d(cand_uv[keep_idx], outer_uv)
                ok = d_out >= min_clear
                for huv in hole_uvs:
                    ok &= ~_points_in_polygon_2d(cand_uv[keep_idx], huv)
                    ok &= _min_dist_to_poly_edges_2d(cand_uv[keep_idx], huv) >= min_clear
                cand_uv = cand_uv[keep_idx[ok]]
                if len(cand_uv) > 0:
                    int_3d, _ = eval_fn(cand_uv[:, 0], cand_uv[:, 1])
                    bnd_uv = np.vstack([bnd_uv, cand_uv])
                    bnd_3d = np.vstack([bnd_3d, int_3d])

    if len(bnd_uv) < 3:
        return np.zeros((0, 3, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    try:
        tri = Delaunay(bnd_uv)
        simplices = tri.simplices
    except Exception:
        outer_idx = np.arange(len(outer_uv), dtype=np.int64)
        off = len(outer_uv)
        h_pairs = []
        for huv, _ in holes_uv_3d:
            h_pairs.append((huv, np.arange(off, off + len(huv), dtype=np.int64)))
            off += len(huv)
        simplices = _earclip_polygon_with_holes(outer_uv, outer_idx, h_pairs)

    if len(simplices) == 0:
        return np.zeros((0, 3, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    centroids = np.mean(bnd_uv[simplices], axis=1)
    valid = _points_in_polygon_2d(centroids, outer_uv)
    for huv in hole_uvs:
        valid &= ~_points_in_polygon_2d(centroids, huv)
    simplices = simplices[valid]
    if len(simplices) == 0:
        return np.zeros((0, 3, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    cent_valid = centroids[valid]
    _, surf_normals = eval_fn(cent_valid[:, 0], cent_valid[:, 1])
    tris_3d = bnd_3d[simplices].copy()
    v0, v1, v2 = tris_3d[:, 0], tris_3d[:, 1], tris_3d[:, 2]
    tn = np.cross(v1 - v0, v2 - v0)
    flip = np.einsum("ij,ij->i", tn, surf_normals) < 0.0
    if np.any(flip):
        tmp = tris_3d[flip, 1].copy()
        tris_3d[flip, 1] = tris_3d[flip, 2]
        tris_3d[flip, 2] = tmp
    # Drop zero-area degenerate slivers
    areas = 0.5 * np.linalg.norm(np.cross(tris_3d[:, 1] - tris_3d[:, 0], tris_3d[:, 2] - tris_3d[:, 0]), axis=1)
    good = areas > 1e-9
    return tris_3d[good], surf_normals[good]


def _assemble_edge_loop(
    doc: _STEPDocument,
    bound_id: int,
    chord_tol_mm: float = 0.08,
    periodic_axis: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Assemble ``FACE_BOUND`` / ``FACE_OUTER_BOUND`` into ``(pts_3d, unwrapped_theta, total_circle_span_rad)``."""
    btype, bargs = doc.get(bound_id)
    loop_id = int(bargs[1])
    bound_ori = bool(bargs[2]) if len(bargs) > 2 and bargs[2] is not None else True
    ltype, largs = doc.get(loop_id)
    if ltype == "VERTEX_LOOP":
        vpt = doc.vertex_mm(int(largs[1]))
        return vpt[None, :], np.zeros(1, dtype=np.float64), 0.0
    if ltype != "EDGE_LOOP":
        return np.zeros((0, 3), dtype=np.float64), None, 0.0

    oe_ids = [int(x) for x in largs[1]]
    if not bound_ori:
        oe_ids = oe_ids[::-1]

    pts_out: List[np.ndarray] = []
    th_out: List[float] = []
    cur_th: Optional[float] = None
    circle_span_sum = 0.0

    for oe_id in oe_ids:
        _, oeargs = doc.get(oe_id)
        # ORIENTED_EDGE(name, edge_start, edge_end, edge_element, orientation)
        ec_id = int(oeargs[3])
        oe_flag = bool(oeargs[4])
        eff_ori = oe_flag if bound_ori else (not oe_flag)
        e_pts, ckind, span_rad = _discretize_edge_curve(doc, ec_id, True, eff_ori, chord_tol_mm=chord_tol_mm)
        if ckind in ("CIRCLE", "ELLIPSE"):
            circle_span_sum += span_rad

        if periodic_axis is not None:
            orig, z_ax, x_ax = periodic_axis
            y_ax = np.cross(z_ax, x_ax)
            rel = e_pts - orig[None, :]
            rx = rel @ x_ax
            ry = rel @ y_ax
            r_xy = np.hypot(rx, ry)
            raw_th = np.arctan2(ry, rx)
            edge_ths: List[float] = []
            for k in range(len(e_pts)):
                if r_xy[k] <= 1e-6:
                    val = cur_th if cur_th is not None else 0.0
                elif cur_th is None:
                    val = float(raw_th[k])
                else:
                    # Unwrap relative to cur_th
                    diff = (float(raw_th[k]) - cur_th + math.pi) % (2.0 * math.pi) - math.pi
                    # Preserve full +/-2*pi sweep on single-edge closed circles
                    if len(e_pts) == 2 and span_rad > 5.5 and k == 1:
                        diff = span_rad if abs(diff) < 1e-4 else diff
                    val = cur_th + diff
                edge_ths.append(val)
                cur_th = val

            start_k = 1 if len(pts_out) > 0 else 0
            for k in range(start_k, len(e_pts)):
                pts_out.append(e_pts[k])
                th_out.append(edge_ths[k])
        else:
            start_k = 1 if len(pts_out) > 0 else 0
            for k in range(start_k, len(e_pts)):
                pts_out.append(e_pts[k])

    if len(pts_out) > 1:
        if periodic_axis is not None:
            # Drop closing duplicate only if BOTH 3D and unwrapped theta match
            if float(np.linalg.norm(pts_out[-1] - pts_out[0])) <= 1e-6 and abs(th_out[-1] - th_out[0]) <= 1e-5:
                pts_out.pop()
                th_out.pop()
        else:
            if float(np.linalg.norm(pts_out[-1] - pts_out[0])) <= 1e-6:
                pts_out.pop()

    pts_arr = np.asarray(pts_out, dtype=np.float64) if pts_out else np.zeros((0, 3), dtype=np.float64)
    th_arr = np.asarray(th_out, dtype=np.float64) if (periodic_axis is not None and th_out) else None
    return pts_arr, th_arr, circle_span_sum


def _tessellate_advanced_face(doc: _STEPDocument, face_id: int, chord_tol_mm: float = 0.08) -> Optional[STEPFaceRecord]:
    """Tessellate one ``ADVANCED_FACE`` with exact boundary curves and outward solid normals."""
    ftype, fargs = doc.get(face_id)
    if ftype != "ADVANCED_FACE":
        return None
    bound_ids = [int(x) for x in fargs[1]]
    surf_id = int(fargs[2])
    same_sense = bool(fargs[3])
    normal_sign = 1.0 if same_sense else -1.0

    stype_raw, sdata = doc.get(surf_id)
    sub = sdata if stype_raw == "__COMPLEX__" else {stype_raw: sdata}
    if "B_SPLINE_SURFACE_WITH_KNOTS" in sub or "B_SPLINE_SURFACE" in sub:
        stype = "B_SPLINE_SURFACE_WITH_KNOTS"
    else:
        stype = stype_raw

    # Identify outer bound (prefer FACE_OUTER_BOUND; fallback to largest 3D bbox bound)
    outer_bid = bound_ids[0]
    for bid in bound_ids:
        bt, _ = doc.get(bid)
        if bt == "FACE_OUTER_BOUND":
            outer_bid = bid
            break
    inner_bids = [bid for bid in bound_ids if bid != outer_bid]

    axis_origin_t: Optional[Tuple[float, float, float]] = None
    axis_dir_t: Optional[Tuple[float, float, float]] = None
    radius_mm: Optional[float] = None
    semi_angle_rad: Optional[float] = None
    inward_concave = False
    tris_3d = np.zeros((0, 3, 3), dtype=np.float64)

    if stype == "PLANE":
        orig, z_ax, x_ax = doc.axis2_placement_3d(int(sdata[1]))
        y_ax = np.cross(z_ax, x_ax)
        out_nrm = normal_sign * z_ax
        axis_origin_t = (float(orig[0]), float(orig[1]), float(orig[2]))
        axis_dir_t = (float(out_nrm[0]), float(out_nrm[1]), float(out_nrm[2]))

        outer_3d, _, _ = _assemble_edge_loop(doc, outer_bid, chord_tol_mm=chord_tol_mm)
        if len(outer_3d) < 3:
            return None
        outer_uv = np.column_stack([(outer_3d - orig) @ x_ax, (outer_3d - orig) @ y_ax])
        all_3d: List[np.ndarray] = [outer_3d]
        outer_idx = np.arange(len(outer_3d), dtype=np.int64)
        offset = len(outer_3d)
        holes_uv_idx: List[Tuple[np.ndarray, np.ndarray]] = []
        for ibid in inner_bids:
            h3d, _, _ = _assemble_edge_loop(doc, ibid, chord_tol_mm=chord_tol_mm)
            if len(h3d) >= 3:
                huv = np.column_stack([(h3d - orig) @ x_ax, (h3d - orig) @ y_ax])
                hidx = np.arange(offset, offset + len(h3d), dtype=np.int64)
                offset += len(h3d)
                all_3d.append(h3d)
                holes_uv_idx.append((huv, hidx))
        verts_3d = np.vstack(all_3d)
        simplices = _earclip_polygon_with_holes(outer_uv, outer_idx, holes_uv_idx)
        if len(simplices) > 0:
            tris_3d = verts_3d[simplices].copy()
            tn = np.cross(tris_3d[:, 1] - tris_3d[:, 0], tris_3d[:, 2] - tris_3d[:, 0])
            flip = (tn @ out_nrm) < 0.0
            if np.any(flip):
                tmp = tris_3d[flip, 1].copy()
                tris_3d[flip, 1] = tris_3d[flip, 2]
                tris_3d[flip, 2] = tmp

    elif stype in ("CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "TOROIDAL_SURFACE", "SPHERICAL_SURFACE"):
        orig, z_ax, x_ax = doc.axis2_placement_3d(int(sdata[1]))
        y_ax = np.cross(z_ax, x_ax)
        r0 = float(sdata[2]) * doc.unit_to_mm
        radius_mm = r0
        axis_origin_t = (float(orig[0]), float(orig[1]), float(orig[2]))
        axis_dir_t = (float(z_ax[0]), float(z_ax[1]), float(z_ax[2]))
        inward_concave = not same_sense
        if stype == "CONICAL_SURFACE":
            semi_angle_rad = float(sdata[3]) * doc.angle_to_rad
        r_minor = float(sdata[3]) * doc.unit_to_mm if stype == "TOROIDAL_SURFACE" else 0.0

        outer_3d, outer_th, span_out = _assemble_edge_loop(
            doc, outer_bid, chord_tol_mm=chord_tol_mm, periodic_axis=(orig, z_ax, x_ax)
        )
        if len(outer_3d) < 3 or outer_th is None:
            return None

        # Handle CAD exporters that emit a closed cylinder/cone as 2 separate circle bounds without seam lines
        if (
            stype in ("CYLINDRICAL_SURFACE", "CONICAL_SURFACE")
            and len(inner_bids) == 1
            and span_out >= 6.0
            and float(np.ptp((outer_3d - orig) @ z_ax)) <= 1e-3
        ):
            inner_3d, inner_th, span_in = _assemble_edge_loop(
                doc, inner_bids[0], chord_tol_mm=chord_tol_mm, periodic_axis=(orig, z_ax, x_ax)
            )
            if len(inner_3d) >= 3 and span_in >= 6.0:
                n_ring = max(len(outer_3d), len(inner_3d), 24)
                ths = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False)
                z1 = float(np.mean((outer_3d - orig) @ z_ax))
                z2 = float(np.mean((inner_3d - orig) @ z_ax))
                r1 = r0 if stype == "CYLINDRICAL_SURFACE" else max(0.0, r0 + z1 * math.tan(semi_angle_rad or 0.0))
                r2 = r0 if stype == "CYLINDRICAL_SURFACE" else max(0.0, r0 + z2 * math.tan(semi_angle_rad or 0.0))
                ring1 = orig[None, :] + z1 * z_ax[None, :] + r1 * (np.cos(ths)[:, None] * x_ax + np.sin(ths)[:, None] * y_ax)
                ring2 = orig[None, :] + z2 * z_ax[None, :] + r2 * (np.cos(ths)[:, None] * x_ax + np.sin(ths)[:, None] * y_ax)
                q_tris: List[np.ndarray] = []
                for k in range(n_ring):
                    kn = (k + 1) % n_ring
                    q_tris.append(np.vstack([ring1[k], ring1[kn], ring2[kn]]))
                    q_tris.append(np.vstack([ring1[k], ring2[kn], ring2[k]]))
                tris_3d = np.asarray(q_tris, dtype=np.float64)
                mid_c = np.mean(tris_3d, axis=1) - orig[None, :]
                rad_vec = mid_c - (mid_c @ z_ax)[:, None] * z_ax[None, :]
                if stype == "CONICAL_SURFACE" and semi_angle_rad is not None:
                    rn = np.maximum(np.linalg.norm(rad_vec, axis=1, keepdims=True), 1e-12)
                    out_n = normal_sign * (
                        math.cos(semi_angle_rad) * (rad_vec / rn) - math.sin(semi_angle_rad) * z_ax[None, :]
                    )
                else:
                    out_n = normal_sign * rad_vec
                tn = np.cross(tris_3d[:, 1] - tris_3d[:, 0], tris_3d[:, 2] - tris_3d[:, 0])
                flip = np.einsum("ij,ij->i", tn, out_n) < 0.0
                if np.any(flip):
                    tmp = tris_3d[flip, 1].copy()
                    tris_3d[flip, 1] = tris_3d[flip, 2]
                    tris_3d[flip, 2] = tmp
                inner_bids = []

        if len(tris_3d) == 0:
            def _to_scaled_uv(pts: np.ndarray, th: np.ndarray, ref_th_center: Optional[float] = None) -> np.ndarray:
                th_adj = th.copy()
                if ref_th_center is not None:
                    shift = round((ref_th_center - float(np.mean(th_adj))) / (2.0 * math.pi)) * (2.0 * math.pi)
                    th_adj = th_adj + shift
                rel = pts - orig[None, :]
                vz = rel @ z_ax
                if stype == "CYLINDRICAL_SURFACE":
                    return np.column_stack([r0 * th_adj, vz])
                if stype == "CONICAL_SURFACE":
                    r_eff = max(0.5, float(np.mean(np.hypot(rel @ x_ax, rel @ y_ax))))
                    return np.column_stack([r_eff * th_adj, vz / max(math.cos(semi_angle_rad or 0.0), 0.2)])
                if stype == "SPHERICAL_SURFACE":
                    r_xy = np.hypot(rel @ x_ax, rel @ y_ax)
                    phi = np.arctan2(vz, r_xy)
                    return np.column_stack([r0 * th_adj, r0 * phi])
                # TOROIDAL_SURFACE
                r_xy = np.hypot(rel @ x_ax, rel @ y_ax)
                phi = np.arctan2(vz, r_xy - r0)
                return np.column_stack([r0 * th_adj, max(r_minor, 0.2) * phi])

            r_ref_scale = r0
            if stype == "CONICAL_SURFACE":
                rel_o = outer_3d - orig[None, :]
                r_ref_scale = max(0.5, float(np.mean(np.hypot(rel_o @ x_ax, rel_o @ y_ax))))
            cos_semi = max(math.cos(semi_angle_rad or 0.0), 0.2) if stype == "CONICAL_SURFACE" else 1.0

            outer_uv = _to_scaled_uv(outer_3d, outer_th)
            th_mid = float(np.mean(outer_th))
            holes_uv_3d: List[Tuple[np.ndarray, np.ndarray]] = []
            for ibid in inner_bids:
                h3d, hth, _ = _assemble_edge_loop(
                    doc, ibid, chord_tol_mm=chord_tol_mm, periodic_axis=(orig, z_ax, x_ax)
                )
                if len(h3d) >= 3 and hth is not None:
                    holes_uv_3d.append((_to_scaled_uv(h3d, hth, ref_th_center=th_mid), h3d))

            def _eval_analytic(u_sc: np.ndarray, v_sc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                if stype == "CYLINDRICAL_SURFACE":
                    th = u_sc / max(r0, 1e-9)
                    ce, se = np.cos(th), np.sin(th)
                    r_dir = ce[:, None] * x_ax[None, :] + se[:, None] * y_ax[None, :]
                    pts = orig[None, :] + v_sc[:, None] * z_ax[None, :] + r0 * r_dir
                    return pts, normal_sign * r_dir
                if stype == "CONICAL_SURFACE":
                    th = u_sc / r_ref_scale
                    vz = v_sc * cos_semi
                    sa = semi_angle_rad or 0.0
                    rad = np.maximum(0.0, r0 + vz * math.tan(sa))
                    r_dir = np.cos(th)[:, None] * x_ax[None, :] + np.sin(th)[:, None] * y_ax[None, :]
                    pts = orig[None, :] + vz[:, None] * z_ax[None, :] + rad[:, None] * r_dir
                    nrm = normal_sign * (math.cos(sa) * r_dir - math.sin(sa) * z_ax[None, :])
                    return pts, nrm
                if stype == "SPHERICAL_SURFACE":
                    th = u_sc / max(r0, 1e-9)
                    phi = v_sc / max(r0, 1e-9)
                    r_dir = np.cos(th)[:, None] * x_ax[None, :] + np.sin(th)[:, None] * y_ax[None, :]
                    n_unit = np.cos(phi)[:, None] * r_dir + np.sin(phi)[:, None] * z_ax[None, :]
                    return orig[None, :] + r0 * n_unit, normal_sign * n_unit
                th = u_sc / max(r0, 1e-9)
                phi = v_sc / max(r_minor, 1e-9)
                r_dir = np.cos(th)[:, None] * x_ax[None, :] + np.sin(th)[:, None] * y_ax[None, :]
                n_unit = np.cos(phi)[:, None] * r_dir + np.sin(phi)[:, None] * z_ax[None, :]
                pts = orig[None, :] + r0 * r_dir + r_minor * n_unit
                return pts, normal_sign * n_unit

            step_u = max(0.8, min(4.0, 0.35 * max(r_ref_scale, 2.0)))
            step_v = max(1.5, min(8.0, 0.5 * float(np.ptp(outer_uv[:, 1]) + 1e-3)))
            tris_3d, _ = _tessellate_trimmed_uv_delaunay(outer_uv, outer_3d, holes_uv_3d, _eval_analytic, step_u, step_v)

    elif stype == "B_SPLINE_SURFACE_WITH_KNOTS":
        if "B_SPLINE_SURFACE" in sub:
            bs = sub["B_SPLINE_SURFACE"]
            u_deg, v_deg = int(bs[0]), int(bs[1])
            cp_rows = bs[2]
            bsk = sub["B_SPLINE_SURFACE_WITH_KNOTS"]
            u_mults, v_mults, u_kvals, v_kvals = bsk[0], bsk[1], bsk[2], bsk[3]
        else:
            bsk = sub["B_SPLINE_SURFACE_WITH_KNOTS"]
            u_deg, v_deg = int(bsk[1]), int(bsk[2])
            cp_rows = bsk[3]
            u_mults, v_mults, u_kvals, v_kvals = bsk[8], bsk[9], bsk[10], bsk[11]
        weights = None
        if "RATIONAL_B_SPLINE_SURFACE" in sub:
            weights = np.asarray(sub["RATIONAL_B_SPLINE_SURFACE"][0], dtype=np.float64)

        ctrl_grid = np.asarray([[doc.point_mm(int(cid)) for cid in row] for row in cp_rows], dtype=np.float64)
        nu_c, nv_c, _ = ctrl_grid.shape
        u_knots = _expand_knots(u_mults, u_kvals)
        v_knots = _expand_knots(v_mults, v_kvals)
        u_lo, u_hi = float(u_knots[u_deg]), float(u_knots[nu_c])
        v_lo, v_hi = float(v_knots[v_deg]), float(v_knots[nv_c])

        # Physical arc-length spans along U and V for metric UV scaling
        len_u = max(0.5, float(np.mean(np.sum(np.linalg.norm(np.diff(ctrl_grid, axis=0), axis=2), axis=0))))
        len_v = max(0.5, float(np.mean(np.sum(np.linalg.norm(np.diff(ctrl_grid, axis=1), axis=2), axis=1))))
        du_rng = max(u_hi - u_lo, 1e-12)
        dv_rng = max(v_hi - v_lo, 1e-12)

        # Coarse UV lookup grid + 6-iter damped Gauss-Newton point-to-surface projection
        g_nu = max(16, min(72, 4 * nu_c))
        g_nv = max(16, min(96, 3 * nv_c))
        gu_1d = np.linspace(u_lo, u_hi, g_nu)
        gv_1d = np.linspace(v_lo, v_hi, g_nv)
        ggu, ggv = np.meshgrid(gu_1d, gv_1d, indexing="ij")
        flat_u, flat_v = ggu.ravel(), ggv.ravel()
        seed_pts, _, _ = _eval_bspline_surface_pts_and_partials(
            ctrl_grid, u_deg, v_deg, u_knots, v_knots, flat_u, flat_v, weights=weights
        )

        def _project_3d_to_scaled_uv(pts_3d: np.ndarray) -> np.ndarray:
            # Nearest seed for each 3D point
            dists2 = np.sum((pts_3d[:, None, :] - seed_pts[None, :, :]) ** 2, axis=2)
            best = np.argmin(dists2, axis=1)
            uk = flat_u[best].copy()
            vk = flat_v[best].copy()
            for _ in range(6):
                sp, dsu, dsv = _eval_bspline_surface_pts_and_partials(
                    ctrl_grid, u_deg, v_deg, u_knots, v_knots, uk, vk, weights=weights
                )
                r = sp - pts_3d
                j11 = np.sum(dsu * dsu, axis=1) + 1e-12
                j12 = np.sum(dsu * dsv, axis=1)
                j22 = np.sum(dsv * dsv, axis=1) + 1e-12
                g1 = np.sum(dsu * r, axis=1)
                g2 = np.sum(dsv * r, axis=1)
                det = np.maximum(j11 * j22 - j12 * j12, 1e-18)
                du_step = np.clip((j22 * g1 - j12 * g2) / det, -0.25 * du_rng, 0.25 * du_rng)
                dv_step = np.clip((j11 * g2 - j12 * g1) / det, -0.25 * dv_rng, 0.25 * dv_rng)
                uk = np.clip(uk - du_step, u_lo, u_hi)
                vk = np.clip(vk - dv_step, v_lo, v_hi)
            return np.column_stack([(uk - u_lo) / du_rng * len_u, (vk - v_lo) / dv_rng * len_v])

        def _eval_bs_scaled(u_sc: np.ndarray, v_sc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            uu = np.clip(u_lo + (u_sc / len_u) * du_rng, u_lo, u_hi)
            vv = np.clip(v_lo + (v_sc / len_v) * dv_rng, v_lo, v_hi)
            pts, dsu, dsv = _eval_bspline_surface_pts_and_partials(
                ctrl_grid, u_deg, v_deg, u_knots, v_knots, uu, vv, weights=weights
            )
            nrm = np.cross(dsu, dsv)
            nl = np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            return pts, normal_sign * (nrm / nl)

        outer_3d, _, _ = _assemble_edge_loop(doc, outer_bid, chord_tol_mm=chord_tol_mm)
        if len(outer_3d) < 3:
            return None
        outer_uv = _project_3d_to_scaled_uv(outer_3d)
        holes_uv_3d_bs: List[Tuple[np.ndarray, np.ndarray]] = []
        for ibid in inner_bids:
            h3d, _, _ = _assemble_edge_loop(doc, ibid, chord_tol_mm=chord_tol_mm)
            if len(h3d) >= 3:
                holes_uv_3d_bs.append((_project_3d_to_scaled_uv(h3d), h3d))
        step_uv = max(1.2, min(3.5, 0.12 * max(len_u, len_v)))
        tris_3d, _ = _tessellate_trimmed_uv_delaunay(
            outer_uv, outer_3d, holes_uv_3d_bs, _eval_bs_scaled, step_uv, step_uv
        )

    if len(tris_3d) == 0:
        return None

    v0, v1, v2 = tris_3d[:, 0], tris_3d[:, 1], tris_3d[:, 2]
    area = float(np.sum(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))
    flat_pts = tris_3d.reshape(-1, 3)
    bmin = tuple(float(v) for v in np.min(flat_pts, axis=0))
    bmax = tuple(float(v) for v in np.max(flat_pts, axis=0))
    return STEPFaceRecord(
        face_id=face_id,
        surface_type=stype,
        same_sense=same_sense,
        axis_origin_mm=axis_origin_t,
        axis_dir=axis_dir_t,
        radius_mm=radius_mm,
        semi_angle_rad=semi_angle_rad,
        inward_concave=inward_concave,
        triangles_mm=tris_3d,
        area_mm2=area,
        bbox_min_mm=bmin,  # type: ignore[arg-type]
        bbox_max_mm=bmax,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 6-Sided SO(3) Setup Auto-Orienter & Semantic CNC Feature Extraction
# ---------------------------------------------------------------------------
# ANSI Unified + ISO Metric standard tap-drill table: (tap_drill_mm, spec, major_mm, pitch_mm)
_TAP_DRILL_TABLE: Tuple[Tuple[float, str, float, float], ...] = (
    (2.050, "M2.5x0.45", 2.500, 0.450),
    (2.057, "#2-56 UNC", 2.184, 0.454),
    (2.261, "#4-40 UNC", 2.845, 0.635),
    (2.500, "M3x0.5", 3.000, 0.500),
    (2.705, "#6-32 UNC", 3.505, 0.794),
    (3.300, "M4x0.7", 4.000, 0.700),
    (3.454, "#8-32 UNC", 4.166, 0.794),
    (3.797, "#10-24 UNC", 4.826, 1.058),
    (4.039, "#10-32 UNF", 4.826, 0.794),
    (4.200, "M5x0.8", 5.000, 0.800),
    (5.000, "M6x1.0", 6.000, 1.000),
    (5.105, "1/4-20 UNC", 6.350, 1.270),
    (5.410, "1/4-28 UNF", 6.350, 0.907),
    (6.528, "5/16-18 UNC", 7.938, 1.411),
    (6.800, "M8x1.25", 8.000, 1.250),
    (7.938, "3/8-16 UNC", 9.525, 1.588),
    (8.500, "M10x1.5", 10.000, 1.500),
    (10.200, "M12x1.75", 12.000, 1.750),
    (10.716, "1/2-13 UNC", 12.700, 1.954),
)


def _match_tap_drill(diam_mm: float, source_unit: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    best: Optional[Tuple[float, str, float, float]] = None
    best_err = 0.095
    prefer_inch = source_unit == "inch"
    for td_mm, spec, maj_mm, pitch_mm in _TAP_DRILL_TABLE:
        err = abs(diam_mm - td_mm)
        if err <= best_err:
            is_iso = spec.startswith("M")
            adj_err = err + (0.015 if (prefer_inch and is_iso) or ((not prefer_inch) and (not is_iso)) else 0.0)
            if best is None or adj_err < best[0]:
                best = (adj_err, spec, maj_mm, pitch_mm)
    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


def _choose_setup_orientation(faces: Sequence[STEPFaceRecord], all_verts: np.ndarray) -> Tuple[np.ndarray, str]:
    """Evaluate all 6 orthogonal spindle candidates ``{+Z, -Z, +Y, -Y, +X, -X}`` -> ``(R_SO3, label)``."""
    ext = np.max(all_verts, axis=0) - np.min(all_verts, axis=0)
    ref_area = max(1.0, float(np.sum([f.area_mm2 for f in faces])) / 6.0)

    candidates: Tuple[Tuple[str, np.ndarray], ...] = (
        ("+Z", np.array([0.0, 0.0, 1.0])),
        ("-Z", np.array([0.0, 0.0, -1.0])),
        ("+Y", np.array([0.0, 1.0, 0.0])),
        ("-Y", np.array([0.0, -1.0, 0.0])),
        ("+X", np.array([1.0, 0.0, 0.0])),
        ("-X", np.array([-1.0, 0.0, 0.0])),
    )

    best_score = -1e18
    best_dir = np.array([0.0, 0.0, 1.0])
    best_name = "+Z"

    for cname, d_vec in candidates:
        ax_i = int(np.argmax(np.abs(d_vec)))
        h_span = max(1e-3, float(ext[ax_i]))
        other_spans = [float(ext[k]) for k in range(3) if k != ax_i]
        plate_score = (min(other_spans) * max(other_spans)) / (h_span * h_span + 1e-6)

        proj = all_verts @ d_vec
        z_min_c, z_max_c = float(np.min(proj)), float(np.max(proj))

        hole_axis_area = 0.0
        recessed_plane_area = 0.0
        recessed_plane_count = 0
        base_clamp_area = 0.0
        freeform_up_area = 0.0

        for f in faces:
            if f.surface_type in ("CYLINDRICAL_SURFACE", "CONICAL_SURFACE") and f.inward_concave and f.axis_dir is not None:
                ad = np.asarray(f.axis_dir, dtype=np.float64)
                if abs(float(np.dot(ad, d_vec))) >= 0.92:
                    hole_axis_area += f.area_mm2
            elif f.surface_type == "PLANE" and f.axis_dir is not None:
                nd = float(np.dot(np.asarray(f.axis_dir, dtype=np.float64), d_vec))
                f_proj = float(np.mean(f.triangles_mm.reshape(-1, 3) @ d_vec))
                if nd >= 0.94 and f_proj < z_max_c - 0.35:
                    recessed_plane_area += f.area_mm2
                    recessed_plane_count += 1
                elif nd <= -0.94 and f_proj <= z_min_c + 0.35:
                    base_clamp_area += f.area_mm2
            elif f.surface_type == "B_SPLINE_SURFACE_WITH_KNOTS":
                v0, v1, v2 = f.triangles_mm[:, 0], f.triangles_mm[:, 1], f.triangles_mm[:, 2]
                tn = np.cross(v1 - v0, v2 - v0)
                if float(np.sum(tn @ d_vec)) > 0.0:
                    freeform_up_area += f.area_mm2

        tie = 0.02 if cname == "+Z" else (0.01 if cname.startswith("+") else 0.0)
        score = (
            1.8 * math.log1p(plate_score)
            + 3.0 * (hole_axis_area / ref_area)
            + 2.5 * (recessed_plane_area / ref_area)
            + 0.35 * recessed_plane_count
            + 1.6 * (freeform_up_area / ref_area)
            + 1.0 * (base_clamp_area / ref_area)
            + tie
        )
        if score > best_score:
            best_score = score
            best_dir = d_vec
            best_name = cname

    # Choose horizontal +X from remaining world axes with largest span, then +Y = +Z x +X (det = +1!)
    z_new = best_dir
    rem_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    rem_axes = [a for a in rem_axes if abs(float(np.dot(a, z_new))) < 0.1]
    rem_axes.sort(key=lambda a: -float(np.ptp(all_verts @ a)))
    x_new = rem_axes[0]
    y_new = np.cross(z_new, x_new)
    rot = np.vstack([x_new, y_new, z_new])
    if float(np.linalg.det(rot)) < 0.0:
        x_new = -x_new
        y_new = np.cross(z_new, x_new)
        rot = np.vstack([x_new, y_new, z_new])
    label = "+Z" if np.allclose(rot, np.eye(3)) else f"{best_name}->+Z"
    return rot, label


def _rotate_faces(faces: Sequence[STEPFaceRecord], rot: np.ndarray) -> Tuple[STEPFaceRecord, ...]:
    if np.allclose(rot, np.eye(3)):
        return tuple(faces)
    rt = rot.T
    out: List[STEPFaceRecord] = []
    for f in faces:
        t_rot = f.triangles_mm @ rt
        pts = t_rot.reshape(-1, 3)
        bmin = tuple(float(v) for v in np.min(pts, axis=0))
        bmax = tuple(float(v) for v in np.max(pts, axis=0))
        ao = tuple(float(v) for v in (rot @ np.asarray(f.axis_origin_mm))) if f.axis_origin_mm is not None else None
        ad = tuple(float(v) for v in (rot @ np.asarray(f.axis_dir))) if f.axis_dir is not None else None
        out.append(
            STEPFaceRecord(
                face_id=f.face_id,
                surface_type=f.surface_type,
                same_sense=f.same_sense,
                axis_origin_mm=ao,  # type: ignore[arg-type]
                axis_dir=ad,  # type: ignore[arg-type]
                radius_mm=f.radius_mm,
                semi_angle_rad=f.semi_angle_rad,
                inward_concave=f.inward_concave,
                triangles_mm=t_rot,
                area_mm2=f.area_mm2,
                bbox_min_mm=bmin,  # type: ignore[arg-type]
                bbox_max_mm=bmax,  # type: ignore[arg-type]
            )
        )
    return tuple(out)


def _extract_brep_features(
    faces: Sequence[STEPFaceRecord],
    bounds_min_mm: Tuple[float, float, float],
    bounds_max_mm: Tuple[float, float, float],
    source_unit: str,
) -> Tuple[
    Tuple[BRepHoleFeature, ...],
    Tuple[BRepHoleFeature, ...],
    Tuple[BRepChamferFeature, ...],
    Tuple[BRepChamferFeature, ...],
    Tuple[BRepFreeformRegion, ...],
    Tuple[BRepFreeformRegion, ...],
    bool,
]:
    z_part_min, z_part_max = bounds_min_mm[2], bounds_max_mm[2]
    z_mid_part = 0.5 * (z_part_min + z_part_max)

    # 1. Coaxial vertical inward cylinders & cones -> Holes, Counterbores, Countersinks, Drill Tips
    cyl_items: List[Tuple[float, float, float, float, float]] = []  # (cx, cy, diam, z_lo, z_hi)
    cone_items: List[Tuple[float, float, float, float, float, float]] = []  # (cx, cy, semi_deg, d_min, d_max, z_mid)

    for f in faces:
        if not f.inward_concave or f.axis_dir is None or f.axis_origin_mm is None:
            continue
        ad = np.asarray(f.axis_dir, dtype=np.float64)
        if abs(ad[2]) < 0.94:
            continue
        ao = np.asarray(f.axis_origin_mm, dtype=np.float64)
        z_lo, z_hi = f.bbox_min_mm[2], f.bbox_max_mm[2]
        zm = 0.5 * (z_lo + z_hi)
        # Project potentially tilted axis onto zm for exact XY center
        t_proj = (zm - ao[2]) / ad[2]
        cx = float(ao[0] + t_proj * ad[0])
        cy = float(ao[1] + t_proj * ad[1])
        pts_xy = f.triangles_mm.reshape(-1, 3)[:, :2] - np.array([[cx, cy]])
        angs = np.mod(np.arctan2(pts_xy[:, 1], pts_xy[:, 0]), 2.0 * math.pi)
        angs_s = np.sort(angs)
        max_gap = float(np.max(np.diff(np.concatenate([angs_s, [angs_s[0] + 2.0 * math.pi]]))))
        arc_span_deg = math.degrees(2.0 * math.pi - max_gap)

        if f.surface_type == "CYLINDRICAL_SURFACE" and f.radius_mm is not None:
            # Exclude 90-deg pocket corner fillets (span < 170 deg) unless radius <= 20 mm full/split bore
            if arc_span_deg >= 170.0 and (z_hi - z_lo) >= 0.20:
                cyl_items.append((cx, cy, 2.0 * f.radius_mm, z_lo, z_hi))
        elif f.surface_type == "CONICAL_SURFACE" and f.semi_angle_rad is not None:
            if arc_span_deg >= 170.0:
                radii = np.hypot(pts_xy[:, 0], pts_xy[:, 1])
                cone_items.append(
                    (cx, cy, math.degrees(f.semi_angle_rad), 2.0 * float(np.min(radii)), 2.0 * float(np.max(radii)), zm)
                )

    # Cluster coaxial cylinders within 0.35 mm XY
    clusters: List[List[Tuple[float, float, float, float, float]]] = []
    for item in cyl_items:
        placed = False
        for grp in clusters:
            if math.hypot(item[0] - grp[0][0], item[1] - grp[0][1]) <= 0.35:
                grp.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    op1_holes: List[BRepHoleFeature] = []
    op2_holes: List[BRepHoleFeature] = []

    for grp in clusters:
        cx = round(float(np.mean([g[0] for g in grp])), 4)
        cy = round(float(np.mean([g[1] for g in grp])), 4)
        # Merge split half-cylinders of identical diameter
        by_d: Dict[float, Tuple[float, float]] = {}
        for _, _, d_val, zlo, zhi in grp:
            dk = round(d_val, 3)
            if dk in by_d:
                by_d[dk] = (min(by_d[dk][0], zlo), max(by_d[dk][1], zhi))
            else:
                by_d[dk] = (zlo, zhi)
        steps = sorted([(dk, bounds[0], bounds[1]) for dk, bounds in by_d.items()], key=lambda s: s[0])
        main_d, main_zlo, main_zhi = steps[0]

        # Match coaxial cones (drill point ~59 deg vs countersink 35..50 deg)
        coax_cones = [c for c in cone_items if math.hypot(c[0] - cx, c[1] - cy) <= 0.45]
        drill_cone_top = any(c[2] >= 52.0 and c[5] > 0.5 * (main_zlo + main_zhi) for c in coax_cones)
        drill_cone_bot = any(c[2] >= 52.0 and c[5] <= 0.5 * (main_zlo + main_zhi) for c in coax_cones)

        cbore_d: Optional[float] = None
        cbore_depth: Optional[float] = None
        side = "op1_top"

        if len(steps) >= 2:
            lg_d, lg_zlo, lg_zhi = steps[-1]
            cbore_d = round(lg_d, 4)
            cbore_depth = round(lg_zhi - lg_zlo, 4)
            # Counterbore sits on entry side!
            side = "op1_top" if 0.5 * (lg_zlo + lg_zhi) >= 0.5 * (main_zlo + main_zhi) else "op2_bottom"
            z_all_lo = min(s[1] for s in steps)
            z_all_hi = max(s[2] for s in steps)
        else:
            z_all_lo, z_all_hi = main_zlo, main_zhi
            if drill_cone_top and not drill_cone_bot:
                side = "op2_bottom"
            elif drill_cone_bot and not drill_cone_top:
                side = "op1_top"
            elif z_all_hi < z_mid_part and z_all_lo <= z_part_min + 0.5:
                side = "op2_bottom"

        # Countersink check (semi-angle 35..50 deg -> included angle 70..100 deg, e.g. 82 or 90 deg)
        csink_d: Optional[float] = None
        csink_ang: Optional[float] = None
        for _, _, semi_deg, _, d_max_c, zm_c in coax_cones:
            if 34.0 <= semi_deg <= 50.0 and d_max_c > main_d + 0.25:
                if (side == "op1_top" and zm_c >= main_zhi - 1.5) or (side == "op2_bottom" and zm_c <= main_zlo + 1.5):
                    csink_d = round(d_max_c, 4)
                    csink_ang = round(2.0 * semi_deg, 1)
                    z_all_hi = max(z_all_hi, zm_c)
                    z_all_lo = min(z_all_lo, zm_c)

        is_through = (not drill_cone_bot and not drill_cone_top) and (
            z_all_lo <= z_part_min + 0.25 or (z_all_hi - z_all_lo) >= 0.85 * (z_part_max - z_part_min)
        )
        t_spec, t_maj, t_pitch = _match_tap_drill(main_d, source_unit)

        hf = BRepHoleFeature(
            side=side,
            center_xy_mm=(cx, cy),
            diameter_mm=round(main_d, 4),
            z_top_mm=round(z_all_hi, 4),
            z_bottom_mm=round(z_all_lo, 4),
            is_through=is_through,
            counterbore_diameter_mm=cbore_d,
            counterbore_depth_mm=cbore_depth,
            countersink_diameter_mm=csink_d,
            countersink_angle_deg=csink_ang,
            thread_spec=t_spec,
            thread_major_mm=t_maj,
            thread_pitch_mm=t_pitch,
        )
        if side == "op1_top":
            op1_holes.append(hf)
        else:
            op2_holes.append(hf)

    op1_holes.sort(key=lambda h: (round(h.diameter_mm, 3), h.center_xy_mm[0], h.center_xy_mm[1]))
    op2_holes.sort(key=lambda h: (round(h.diameter_mm, 3), h.center_xy_mm[0], h.center_xy_mm[1]))

    # 2. Chamfer Features (planar & partial conical bevels at ~45 deg, width <= 4.5 mm)
    op1_chamfers: List[BRepChamferFeature] = []
    op2_chamfers: List[BRepChamferFeature] = []
    for f in faces:
        dz = f.bbox_max_mm[2] - f.bbox_min_mm[2]
        if dz < 0.12 or dz > 4.5:
            continue
        if f.surface_type == "PLANE" and f.axis_dir is not None:
            nz = float(f.axis_dir[2])
            if 0.45 <= abs(nz) <= 0.88:
                ang = round(math.degrees(math.acos(min(1.0, abs(nz)))), 1)
                c_side = "op1_top" if nz > 0.0 else "op2_bottom"
                pts = f.triangles_mm.reshape(-1, 3)
                # Extract lower lip edge segment for cutter offset reference
                z_lip = f.bbox_min_mm[2] if nz > 0.0 else f.bbox_max_mm[2]
                lip_pts = pts[np.abs(pts[:, 2] - z_lip) <= max(0.08, 0.18 * dz)][:, :2]
                if len(lip_pts) >= 2:
                    # Principal chord endpoints along lip
                    dmat = np.sum((lip_pts[:, None, :] - lip_pts[None, :, :]) ** 2, axis=2)
                    i_a, i_b = np.unravel_index(int(np.argmax(dmat)), dmat.shape)
                    pa, pb = lip_pts[i_a], lip_pts[i_b]
                    if float(np.linalg.norm(pb - pa)) >= 0.8:
                        cf = BRepChamferFeature(
                            side=c_side,
                            z_top_mm=round(f.bbox_max_mm[2], 4),
                            z_bottom_mm=round(f.bbox_min_mm[2], 4),
                            width_mm=round(dz, 4),
                            angle_deg=ang,
                            polyline_xy_mm=((round(float(pa[0]), 4), round(float(pa[1]), 4)), (round(float(pb[0]), 4), round(float(pb[1]), 4))),
                            is_circular_hole_chamfer=False,
                        )
                        (op1_chamfers if c_side == "op1_top" else op2_chamfers).append(cf)

    # 3. 3D Freeform Sculptured Regions (B-spline surfaces + horizontal curved crowns)
    raw_ff: List[Tuple[str, Tuple[float, float], Tuple[float, float], float, float, str, float]] = []
    for f in faces:
        v0, v1, v2 = f.triangles_mm[:, 0], f.triangles_mm[:, 1], f.triangles_mm[:, 2]
        tn = np.cross(v1 - v0, v2 - v0)
        area_sum = max(float(np.sum(np.linalg.norm(tn, axis=1))), 1e-12)
        mean_nz = float(np.sum(tn[:, 2])) / area_sum
        is_bs = f.surface_type == "B_SPLINE_SURFACE_WITH_KNOTS" and abs(mean_nz) >= 0.12
        is_horiz_curved = (
            f.surface_type in ("CYLINDRICAL_SURFACE", "TOROIDAL_SURFACE", "SPHERICAL_SURFACE")
            and f.axis_dir is not None
            and abs(float(f.axis_dir[2])) < 0.25
            and 0.20 <= abs(mean_nz) <= 0.96
            and f.area_mm2 >= 18.0
        )
        if is_bs or is_horiz_curved:
            f_side = "op1_top" if mean_nz >= 0.0 else "op2_bottom"
            raw_ff.append(
                (
                    f_side,
                    (f.bbox_min_mm[0], f.bbox_min_mm[1]),
                    (f.bbox_max_mm[0], f.bbox_max_mm[1]),
                    f.bbox_min_mm[2],
                    f.bbox_max_mm[2],
                    f.surface_type,
                    f.area_mm2,
                )
            )

    def _cluster_freeform(side_filter: str) -> Tuple[BRepFreeformRegion, ...]:
        items = [r for r in raw_ff if r[0] == side_filter]
        # Require at least one genuine B-spline face on the part or large sculptured crown (>80 mm2)
        has_bspline = any(r[5] == "B_SPLINE_SURFACE_WITH_KNOTS" for r in items)
        if not has_bspline and sum(r[6] for r in items) < 80.0:
            return ()
        used = [False] * len(items)
        regions: List[BRepFreeformRegion] = []
        for i in range(len(items)):
            if used[i]:
                continue
            used[i] = True
            x0, y0 = items[i][1]
            x1, y1 = items[i][2]
            z0, z1 = items[i][3], items[i][4]
            stypes = {items[i][5]}
            area_tot = items[i][6]
            changed = True
            while changed:
                changed = False
                for j in range(len(items)):
                    if used[j]:
                        continue
                    jx0, jy0 = items[j][1]
                    jx1, jy1 = items[j][2]
                    if jx0 <= x1 + 2.5 and jx1 >= x0 - 2.5 and jy0 <= y1 + 2.5 and jy1 >= y0 - 2.5:
                        used[j] = True
                        x0, y0 = min(x0, jx0), min(y0, jy0)
                        x1, y1 = max(x1, jx1), max(y1, jy1)
                        z0, z1 = min(z0, items[j][3]), max(z1, items[j][4])
                        stypes.add(items[j][5])
                        area_tot += items[j][6]
                        changed = True
            regions.append(
                BRepFreeformRegion(
                    side=side_filter,
                    min_xy_mm=(round(x0, 4), round(y0, 4)),
                    max_xy_mm=(round(x1, 4), round(y1, 4)),
                    z_min_mm=round(z0, 4),
                    z_max_mm=round(z1, 4),
                    surface_types=tuple(sorted(stypes)),
                    area_mm2=round(area_tot, 3),
                )
            )
        return tuple(regions)

    op1_ff = _cluster_freeform("op1_top")
    op2_ff = _cluster_freeform("op2_bottom")

    # Detect underside recessed planar pockets/steps facing -Z above z_part_min + 0.4 mm
    has_op2_recess = any(
        f.surface_type == "PLANE"
        and f.axis_dir is not None
        and f.axis_dir[2] <= -0.94
        and f.bbox_min_mm[2] > z_part_min + 0.45
        and f.area_mm2 >= 25.0
        for f in faces
    )
    requires_op2 = bool(op2_holes or op2_chamfers or op2_ff or has_op2_recess)
    return (
        tuple(op1_holes),
        tuple(op2_holes),
        tuple(op1_chamfers),
        tuple(op2_chamfers),
        op1_ff,
        op2_ff,
        requires_op2,
    )


def load_step_brep(
    source: Union[str, Path, bytes],
    *,
    auto_orient: bool = True,
    chord_tol_mm: float = 0.08,
) -> STEPLoadResult:
    """Load, tessellate, auto-orient, and extract CNC features from an ISO 10303-21 STEP solid."""
    if isinstance(source, (str, Path)):
        raw_str = str(source)
        if "\x00" in raw_str:
            raise ValueError("NUL byte in STEP path")
        p_in = Path(raw_str)
        if ".." in p_in.parts:
            raise ValueError("Path traversal '..' is forbidden in STEP path")
        lst = os.lstat(p_in)
        if stat.S_ISLNK(lst.st_mode) or not stat.S_ISREG(lst.st_mode):
            raise ValueError("STEP source must be a regular non-symlink file")
        if lst.st_size > MAX_STEP_BYTES:
            raise ValueError(f"STEP file exceeds {MAX_STEP_BYTES} byte safety cap")
        raw_bytes = p_in.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw_bytes = bytes(source)
    else:
        raise TypeError("source must be str, Path, or bytes")

    doc = _STEPDocument(raw_bytes)
    raw_faces: List[STEPFaceRecord] = []
    for eid, body in doc.raw_entities.items():
        if body.startswith("ADVANCED_FACE("):
            frec = _tessellate_advanced_face(doc, eid, chord_tol_mm=chord_tol_mm)
            if frec is not None:
                raw_faces.append(frec)

    if not raw_faces:
        raise ValueError("STEP solid contains zero tessellatable ADVANCED_FACE entities")

    raw_tris = np.vstack([f.triangles_mm for f in raw_faces])
    raw_pts = raw_tris.reshape(-1, 3)
    raw_bmin = tuple(float(v) for v in np.min(raw_pts, axis=0))
    raw_bmax = tuple(float(v) for v in np.max(raw_pts, axis=0))

    if auto_orient:
        rot, axis_label = _choose_setup_orientation(raw_faces, raw_pts)
        oriented_faces = _rotate_faces(raw_faces, rot)
    else:
        rot = np.eye(3, dtype=np.float64)
        axis_label = "+Z"
        oriented_faces = tuple(raw_faces)

    ori_tris = np.vstack([f.triangles_mm for f in oriented_faces])
    ori_pts = ori_tris.reshape(-1, 3)
    ori_bmin = tuple(float(v) for v in np.min(ori_pts, axis=0))
    ori_bmax = tuple(float(v) for v in np.max(ori_pts, axis=0))

    op1_h, op2_h, op1_c, op2_c, op1_ff, op2_ff, req_op2 = _extract_brep_features(
        oriented_faces,
        ori_bmin,  # type: ignore[arg-type]
        ori_bmax,  # type: ignore[arg-type]
        doc.source_unit,
    )

    return STEPLoadResult(
        part_name=doc.part_name,
        source_unit=doc.source_unit,
        unit_to_mm=doc.unit_to_mm,
        raw_bounds_min_mm=raw_bmin,  # type: ignore[arg-type]
        raw_bounds_max_mm=raw_bmax,  # type: ignore[arg-type]
        setup_rotation=rot,
        setup_axis_label=axis_label,
        oriented_triangles_mm=ori_tris,
        oriented_bounds_min_mm=ori_bmin,  # type: ignore[arg-type]
        oriented_bounds_max_mm=ori_bmax,  # type: ignore[arg-type]
        stl_bytes=triangles_to_binary_stl_bytes(ori_tris),
        faces=oriented_faces,
        op1_holes=op1_h,
        op2_holes=op2_h,
        op1_chamfers=op1_c,
        op2_chamfers=op2_c,
        op1_freeform=op1_ff,
        op2_freeform=op2_ff,
        requires_op2_flip=req_op2,
    )


__all__ = [
    "BRepChamferFeature",
    "BRepFreeformRegion",
    "BRepHoleFeature",
    "MAX_STEP_BYTES",
    "MAX_STEP_ENTITIES",
    "STEPFaceRecord",
    "STEPLoadResult",
    "is_step_payload",
    "load_step_brep",
    "triangles_to_binary_stl_bytes",
]
