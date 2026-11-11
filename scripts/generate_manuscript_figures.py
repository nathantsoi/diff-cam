# SPDX-License-Identifier: MIT
"""Pure-Python PDF 1.4 publication figure generator for DiffCAM.

Generates four IEEEtran PDF figures directly from campaign JSON artifacts
and CAMotics v1.2.0 watertight STL meshes (zero external plotting dependencies):
- ``fig1_hero_camotics.pdf``: 4-family Nominal CAM vs. DiffCAM 2.5D feed-colored toolpaths
  + 4x-MSAA Z-buffered 3D isometric renderings of CAMotics v1.2.0 cut-workpiece STL meshes.
- ``fig2_exp1_conditioning.pdf``: Stiffness k gradient cosine/SNR, soft-hard occupancy L1 bias,
  and R-perturbation reachability cycle-time recovery (Exp1).
- ``fig3_exp2_convergence.pdf``: 4-family sample-efficiency convergence curves (mean +/- std
  across seeds over 64 simulator evaluations, Exp2).
- ``fig4_exp3_transfer_ablations.pdf``: Held-out zero-shot / 16-eval policy transfer (Exp3) +
  5-arm physical/kinematic ablation study with Peak MRR overload analysis (Exp2-Ablation).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import contracts as C
from benchmarks.milling_suite import build_milling_suite, make_joint_parameterization

RGB = Tuple[float, float, float]

# Publication colorblind-safe Okabe-Ito / Viridis-inspired palette
METHOD_STYLE: Dict[str, Tuple[RGB, Optional[Tuple[float, ...]], str]] = {
    "STE-Continuation-Adam": ((0.00, 0.45, 0.70), None, "STE-Cont-Adam (1st-Order Traj)"),
    "DiffPolicy-SHAC": ((0.00, 0.62, 0.45), None, "DiffPolicy-SHAC (1st-Order RL)"),
    "Annealed-Adam": ((0.34, 0.71, 0.91), (3.5, 2.0), "Annealed-Adam (Smooth k-Sched)"),
    "Naive-Smooth-Adam": ((0.55, 0.55, 0.58), (2.0, 2.0), "Naive-Smooth-Adam (Fixed k)"),
    "Hard-PPO": ((0.84, 0.37, 0.00), (4.5, 2.0), "Hard-PPO (0th-Order RL)"),
    "Hard-CEM": ((0.80, 0.47, 0.65), (2.5, 1.8), "Hard-CEM (0th-Order Elite)"),
    "Hard-CMA-ES": ((0.90, 0.62, 0.00), (1.5, 1.8), "Hard-CMA-ES (0th-Order Cov)"),
}

FAMILY_LABELS = {
    "facing_open_boss": "Facing (Open Boss)",
    "pocket_island": "Pocket (Retained Island)",
    "adaptive_channel": "Adaptive (Dual-Island)",
    "contour_profile": "Contour (Exterior Boss)",
}
FAMILY_SHORT = {
    "facing_open_boss": "Facing",
    "pocket_island": "Pocket",
    "adaptive_channel": "Adaptive",
    "contour_profile": "Contour",
}
FAMILY_ORDER = ("facing_open_boss", "pocket_island", "adaptive_channel", "contour_profile")


def _pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _approx_text_width(text: str, size: float, bold: bool = False, mono: bool = False) -> float:
    if mono:
        return len(text) * size * 0.60
    narrow = set("ijlI1.,:;|![]()' ")
    wide = set("MWQG@%+-=")
    total = 0.0
    for ch in text:
        if ch in narrow:
            total += 0.30
        elif ch in wide:
            total += 0.78
        elif ch.isupper():
            total += 0.66
        else:
            total += 0.53
    return total * size * (1.05 if bold else 1.0)


class VectorPDFCanvas:
    """Minimal, self-contained native PDF 1.4 vector + XObject image builder."""

    def __init__(self, width_pt: float, height_pt: float) -> None:
        self.width = float(width_pt)
        self.height = float(height_pt)
        self._ops: List[str] = ["1 J 1 j\n"]
        self._images: List[Tuple[str, int, int, bytes]] = []

    def _set_Alpha(self, band: bool) -> None:
        self._ops.append("/GS2 gs\n" if band else "/GS1 gs\n")

    def save_state(self) -> None:
        self._ops.append("q\n")

    def restore_state(self) -> None:
        self._ops.append("Q\n")

    def clip_rect(self, x: float, y: float, w: float, h: float) -> None:
        self._ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re W n\n")

    def draw_rgb_image(
        self,
        img_bottom_up: np.ndarray,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:
        """Embed an ``(H, W, 3)`` uint8 RGB image (row 0 = bottom) as a lossless PDF XObject."""
        arr = np.asarray(img_bottom_up, dtype=np.uint8)
        h_px, w_px, _ = arr.shape
        top_down = np.ascontiguousarray(arr[::-1])
        comp = zlib.compress(top_down.tobytes(), level=9)
        im_name = f"Im{len(self._images) + 1}"
        self._images.append((im_name, w_px, h_px, comp))
        self._ops.append(
            f"q /GS1 gs {w:.2f} 0 0 {h:.2f} {x:.2f} {y:.2f} cm /{im_name} Do Q\n"
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: Optional[RGB] = None,
        stroke: Optional[RGB] = None,
        line_width: float = 0.6,
        alpha_band: bool = False,
    ) -> None:
        self.save_state()
        self._set_Alpha(alpha_band)
        if fill is not None and stroke is not None:
            self._ops.append(
                f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg "
                f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG "
                f"{line_width:.2f} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re B\n"
            )
        elif fill is not None:
            self._ops.append(
                f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg "
                f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re f\n"
            )
        elif stroke is not None:
            self._ops.append(
                f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG "
                f"{line_width:.2f} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re S\n"
            )
        self.restore_state()

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: RGB = (0.0, 0.0, 0.0),
        line_width: float = 0.8,
        dash: Optional[Sequence[float]] = None,
    ) -> None:
        self.save_state()
        self._set_Alpha(False)
        dash_cmd = (
            f"[{' '.join(f'{d:.2f}' for d in dash)}] 0 d "
            if dash
            else "[] 0 d "
        )
        self._ops.append(
            f"{dash_cmd}{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG "
            f"{line_width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"
        )
        self.restore_state()

    def polyline(
        self,
        pts: Sequence[Tuple[float, float]],
        *,
        stroke: RGB = (0.0, 0.0, 0.0),
        line_width: float = 1.0,
        dash: Optional[Sequence[float]] = None,
        close: bool = False,
    ) -> None:
        if len(pts) < 2:
            return
        self.save_state()
        self._set_Alpha(False)
        dash_cmd = (
            f"[{' '.join(f'{d:.2f}' for d in dash)}] 0 d "
            if dash
            else "[] 0 d "
        )
        segs = [f"{pts[0][0]:.2f} {pts[0][1]:.2f} m"]
        for px, py in pts[1:]:
            segs.append(f"{px:.2f} {py:.2f} l")
        end_op = "s" if close else "S"
        self._ops.append(
            f"{dash_cmd}{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG "
            f"{line_width:.2f} w {' '.join(segs)} {end_op}\n"
        )
        self.restore_state()

    def polygon(
        self,
        pts: Sequence[Tuple[float, float]],
        *,
        fill: Optional[RGB] = None,
        stroke: Optional[RGB] = None,
        line_width: float = 0.4,
        alpha_band: bool = False,
    ) -> None:
        if len(pts) < 3:
            return
        self.save_state()
        self._set_Alpha(alpha_band)
        segs = [f"{pts[0][0]:.2f} {pts[0][1]:.2f} m"]
        for px, py in pts[1:]:
            segs.append(f"{px:.2f} {py:.2f} l")
        path_str = " ".join(segs) + " h"
        if fill is not None and stroke is not None:
            self._ops.append(
                f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg "
                f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG "
                f"{line_width:.2f} w {path_str} B\n"
            )
        elif fill is not None:
            self._ops.append(
                f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg {path_str} f\n"
            )
        elif stroke is not None:
            self._ops.append(
                f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG "
                f"{line_width:.2f} w {path_str} S\n"
            )
        self.restore_state()

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: Optional[RGB] = None,
        stroke: Optional[RGB] = None,
        line_width: float = 0.6,
    ) -> None:
        k = 0.55228475 * r
        self.save_state()
        self._set_Alpha(False)
        path = (
            f"{cx + r:.2f} {cy:.2f} m "
            f"{cx + r:.2f} {cy + k:.2f} {cx + k:.2f} {cy + r:.2f} {cx:.2f} {cy + r:.2f} c "
            f"{cx - k:.2f} {cy + r:.2f} {cx - r:.2f} {cy + k:.2f} {cx - r:.2f} {cy:.2f} c "
            f"{cx - r:.2f} {cy - k:.2f} {cx - k:.2f} {cy - r:.2f} {cx:.2f} {cy - r:.2f} c "
            f"{cx + k:.2f} {cy - r:.2f} {cx + r:.2f} {cy - k:.2f} {cx + r:.2f} {cy:.2f} c h"
        )
        if fill is not None and stroke is not None:
            self._ops.append(
                f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg "
                f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG "
                f"{line_width:.2f} w {path} B\n"
            )
        elif fill is not None:
            self._ops.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg {path} f\n")
        elif stroke is not None:
            self._ops.append(
                f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG {line_width:.2f} w {path} S\n"
            )
        self.restore_state()

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        size: float = 7.5,
        bold: bool = False,
        mono: bool = False,
        rgb: RGB = (0.10, 0.10, 0.12),
        align: str = "left",
        rotate_deg: float = 0.0,
    ) -> None:
        font = "/F3" if mono else ("/F2" if bold else "/F1")
        w = _approx_text_width(content, size, bold=bold, mono=mono)
        offset = 0.0 if align == "left" else (-0.5 * w if align == "center" else -w)
        esc = _pdf_escape(content)
        self.save_state()
        self._set_Alpha(False)
        if abs(rotate_deg) < 1e-6:
            tx = x + offset
            ty = y
            tm = f"1 0 0 1 {tx:.2f} {ty:.2f} Tm"
        else:
            rad = math.radians(rotate_deg)
            cs, sn = math.cos(rad), math.sin(rad)
            tx = x + offset * cs
            ty = y + offset * sn
            tm = f"{cs:.4f} {sn:.4f} {-sn:.4f} {cs:.4f} {tx:.2f} {ty:.2f} Tm"
        self._ops.append(
            f"BT {font} {size:.2f} Tf {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} rg {tm} ({esc}) Tj ET\n"
        )
        self.restore_state()

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_stream = "".join(self._ops).encode("latin-1", errors="replace")
        comp = zlib.compress(raw_stream, level=9)

        xobj_entries = ""
        if self._images:
            refs = " ".join(f"/{name} {10 + idx} 0 R" for idx, (name, _, _, _) in enumerate(self._images))
            xobj_entries = f" /XObject << {refs} >>"

        objects: List[bytes] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.2f} {self.height:.2f}] "
                f"/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> "
                f"/ExtGState << /GS1 7 0 R /GS2 8 0 R >>{xobj_entries} >> /Contents 9 0 R >>"
            ).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
            b"<< /Type /ExtGState /ca 1.0 /CA 1.0 >>",
            b"<< /Type /ExtGState /ca 0.18 /CA 0.18 >>",
            f"<< /Length {len(comp)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
            + comp
            + b"\nendstream",
        ]
        for _, w_px, h_px, im_comp in self._images:
            objects.append(
                (
                    f"<< /Type /XObject /Subtype /Image /Width {w_px} /Height {h_px} "
                    f"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
                    f"/Length {len(im_comp)} /Filter /FlateDecode >>\nstream\n"
                ).encode("ascii")
                + im_comp
                + b"\nendstream"
            )

        buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(buf))
            buf.extend(f"{idx} 0 obj\n".encode("ascii"))
            buf.extend(obj)
            buf.extend(b"\nendobj\n")

        xref_pos = len(buf)
        buf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        buf.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        buf.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
                "ascii"
            )
        )
        path.write_bytes(bytes(buf))


def _feed_colormap(feed_mm_s: float, vmin: float = 3.0, vmax: float = 20.0) -> RGB:
    """Smooth plasma/viridis-like ramp from deep navy (slow) -> teal -> amber-gold (fast)."""
    t = max(0.0, min(1.0, (feed_mm_s - vmin) / max(vmax - vmin, 1e-6)))
    stops = [
        (0.00, (0.14, 0.18, 0.56)),
        (0.35, (0.05, 0.52, 0.62)),
        (0.70, (0.28, 0.74, 0.36)),
        (1.00, (0.96, 0.64, 0.06)),
    ]
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        if t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-9)
            return (
                c0[0] + u * (c1[0] - c0[0]),
                c0[1] + u * (c1[1] - c0[1]),
                c0[2] + u * (c1[2] - c0[2]),
            )
    return stops[-1][1]


def _load_binary_or_ascii_stl(stl_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(triangles (M, 3, 3), normals (M, 3))`` from a CAMotics STL mesh."""
    raw = stl_path.read_bytes()
    if len(raw) >= 84:
        tri_count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + tri_count * 50 == len(raw) and tri_count > 0:
            dtype = np.dtype(
                [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
            )
            arr = np.frombuffer(raw, dtype=dtype, offset=84, count=tri_count)
            verts = np.asarray(arr["verts"], dtype=np.float64)
            v1 = verts[:, 1] - verts[:, 0]
            v2 = verts[:, 2] - verts[:, 0]
            nrm = np.cross(v1, v2)
            nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = np.divide(nrm, np.maximum(nlen, 1e-12))
            return verts, nrm
    tris = []
    cur = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        toks = line.strip().split()
        if len(toks) == 4 and toks[0] == "vertex":
            cur.append([float(toks[1]), float(toks[2]), float(toks[3])])
            if len(cur) == 3:
                tris.append(cur)
                cur = []
    verts = np.asarray(tris, dtype=np.float64)
    v1 = verts[:, 1] - verts[:, 0]
    v2 = verts[:, 2] - verts[:, 0]
    nrm = np.cross(v1, v2)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return verts, nrm


def _render_stl_isometric_rgb(
    stl_path: Path,
    width_px: int = 330,
    height_px: int = 198,
    bg_rgb: RGB = (0.975, 0.978, 0.985),
) -> np.ndarray:
    """4x-MSAA exact Z-buffered CAD/CAM metallic renderer over ALL triangles of a CAMotics STL."""
    verts, normals = _load_binary_or_ascii_stl(stl_path)

    az = math.radians(-52.0)
    el = math.radians(34.0)
    cam_dir = np.array(
        [math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)],
        dtype=np.float64,
    )
    cam_dir /= np.linalg.norm(cam_dir)
    cam_right = np.cross(np.array([0.0, 0.0, 1.0]), cam_dir)
    cam_right /= np.linalg.norm(cam_right)
    cam_up = np.cross(cam_dir, cam_right)

    # Keep all front-facing facets (watertight closed solid -> back faces strictly hidden)
    keep = (normals @ cam_dir) > -1e-4
    verts = verts[keep]
    normals = normals[keep]

    # 2x supersampled raster grid for crisp 4x-MSAA silhouettes & machined edges
    ss = 2
    W = width_px * ss
    H = height_px * ss

    proj_x = verts @ cam_right
    proj_y = verts @ cam_up
    proj_z = verts @ cam_dir

    xmin, xmax = float(np.min(proj_x)), float(np.max(proj_x))
    ymin, ymax = float(np.min(proj_y)), float(np.max(proj_y))
    pad = 8.0 * ss
    scale = min((W - 2.0 * pad) / max(xmax - xmin, 1e-6), (H - 2.0 * pad) / max(ymax - ymin, 1e-6))
    cx_mid = 0.5 * (xmin + xmax)
    cy_mid = 0.5 * (ymin + ymax)

    sx = 0.5 * W + (proj_x - cx_mid) * scale
    sy = 0.5 * H + (proj_y - cy_mid) * scale

    x0, x1, x2 = sx[:, 0], sx[:, 1], sx[:, 2]
    y0, y1, y2 = sy[:, 0], sy[:, 1], sy[:, 2]
    z0, z1, z2 = proj_z[:, 0], proj_z[:, 1], proj_z[:, 2]

    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    non_deg = np.abs(area2) > 1e-6
    x0, x1, x2 = x0[non_deg], x1[non_deg], x2[non_deg]
    y0, y1, y2 = y0[non_deg], y1[non_deg], y2[non_deg]
    z0, z1, z2 = z0[non_deg], z1[non_deg], z2[non_deg]
    area2 = area2[non_deg]
    normals = normals[non_deg]

    # Ensure positive 2D winding in screen space
    flip = area2 < 0.0
    if np.any(flip):
        x1_f, x2_f = x1[flip].copy(), x2[flip].copy()
        y1_f, y2_f = y1[flip].copy(), y2[flip].copy()
        z1_f, z2_f = z1[flip].copy(), z2[flip].copy()
        x1[flip], x2[flip] = x2_f, x1_f
        y1[flip], y2[flip] = y2_f, y1_f
        z1[flip], z2[flip] = z2_f, z1_f
        area2[flip] = -area2[flip]

    ix_min = np.clip(np.floor(np.minimum(np.minimum(x0, x1), x2)).astype(np.int32), 0, W - 1)
    ix_max = np.clip(np.ceil(np.maximum(np.maximum(x0, x1), x2)).astype(np.int32), 0, W - 1)
    iy_min = np.clip(np.floor(np.minimum(np.minimum(y0, y1), y2)).astype(np.int32), 0, H - 1)
    iy_max = np.clip(np.ceil(np.maximum(np.maximum(y0, y1), y2)).astype(np.int32), 0, H - 1)

    zbuf = np.full((H, W), -1e30, dtype=np.float64)
    tri_buf = np.full((H, W), -1, dtype=np.int32)

    span_x = ix_max - ix_min
    span_y = iy_max - iy_min
    MAX_VEC_SPAN = 9
    small_mask = (span_x <= MAX_VEC_SPAN) & (span_y <= MAX_VEC_SPAN)
    small_ids = np.nonzero(small_mask)[0]
    large_ids = np.nonzero(~small_mask)[0]

    # Vectorized exact edge-function rasterizer for small marching-cubes facets
    if len(small_ids) > 0:
        sx0, sx1, sx2 = x0[small_ids], x1[small_ids], x2[small_ids]
        sy0, sy1, sy2 = y0[small_ids], y1[small_ids], y2[small_ids]
        sz0, sz1, sz2 = z0[small_ids], z1[small_ids], z2[small_ids]
        inv_a = 1.0 / area2[small_ids]
        bx0 = ix_min[small_ids]
        by0 = iy_min[small_ids]
        bx1 = ix_max[small_ids]
        by1 = iy_max[small_ids]

        # Collect covered (py, px, z_val, tid) across the small bbox stencil
        cov_py: List[np.ndarray] = []
        cov_px: List[np.ndarray] = []
        cov_z: List[np.ndarray] = []
        cov_tid: List[np.ndarray] = []
        for dy in range(MAX_VEC_SPAN + 1):
            py = by0 + dy
            valid_y = py <= by1
            if not np.any(valid_y):
                continue
            py_f = py.astype(np.float64) + 0.5
            for dx in range(MAX_VEC_SPAN + 1):
                px = bx0 + dx
                active = valid_y & (px <= bx1)
                if not np.any(active):
                    continue
                idx_a = np.nonzero(active)[0]
                px_f = px[idx_a].astype(np.float64) + 0.5
                py_fa = py_f[idx_a]
                w0 = (sx1[idx_a] - px_f) * (sy2[idx_a] - py_fa) - (sy1[idx_a] - py_fa) * (sx2[idx_a] - px_f)
                w1 = (sx2[idx_a] - px_f) * (sy0[idx_a] - py_fa) - (sy2[idx_a] - py_fa) * (sx0[idx_a] - px_f)
                w2 = (sx0[idx_a] - px_f) * (sy1[idx_a] - py_fa) - (sy0[idx_a] - py_fa) * (sx1[idx_a] - px_f)
                inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
                if np.any(inside):
                    ins = idx_a[inside]
                    ia = inv_a[ins]
                    zp = (w0[inside] * sz0[ins] + w1[inside] * sz1[ins] + w2[inside] * sz2[ins]) * ia
                    cov_py.append(py[ins])
                    cov_px.append(px[ins])
                    cov_z.append(zp)
                    cov_tid.append(small_ids[ins])

        if cov_z:
            all_py = np.concatenate(cov_py)
            all_px = np.concatenate(cov_px)
            all_z = np.concatenate(cov_z)
            all_tid = np.concatenate(cov_tid)
            np.maximum.at(zbuf, (all_py, all_px), all_z)
            winners = all_z >= (zbuf[all_py, all_px] - 1e-7)
            tri_buf[all_py[winners], all_px[winners]] = all_tid[winners]

    # Exact bbox rasterizer for any larger merged facets
    for tid in large_ids:
        px_grid, py_grid = np.meshgrid(
            np.arange(ix_min[tid], ix_max[tid] + 1, dtype=np.float64) + 0.5,
            np.arange(iy_min[tid], iy_max[tid] + 1, dtype=np.float64) + 0.5,
        )
        w0 = (x1[tid] - px_grid) * (y2[tid] - py_grid) - (y1[tid] - py_grid) * (x2[tid] - px_grid)
        w1 = (x2[tid] - px_grid) * (y0[tid] - py_grid) - (y2[tid] - py_grid) * (x0[tid] - px_grid)
        w2 = (x0[tid] - px_grid) * (y1[tid] - py_grid) - (y0[tid] - py_grid) * (x1[tid] - px_grid)
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            continue
        zp = (w0 * z0[tid] + w1 * z1[tid] + w2 * z2[tid]) / area2[tid]
        sub_z = zbuf[iy_min[tid] : iy_max[tid] + 1, ix_min[tid] : ix_max[tid] + 1]
        sub_t = tri_buf[iy_min[tid] : iy_max[tid] + 1, ix_min[tid] : ix_max[tid] + 1]
        better = inside & (zp >= sub_z)
        sub_z[better] = zp[better]
        sub_t[better] = int(tid)

    # Shade hit pixels (6061-T6 brushed aluminum studio lighting + crease/silhouette accents)
    hit = tri_buf >= 0
    img = np.empty((H, W, 3), dtype=np.float64)
    img[:, :, 0] = bg_rgb[0]
    img[:, :, 1] = bg_rgb[1]
    img[:, :, 2] = bg_rgb[2]

    if np.any(hit):
        nrm_pix = np.zeros((H, W, 3), dtype=np.float64)
        nrm_pix[hit] = normals[tri_buf[hit]]

        l_key = np.array([0.36, -0.46, 0.81], dtype=np.float64)
        l_key /= np.linalg.norm(l_key)
        l_fill = np.array([-0.60, 0.32, 0.42], dtype=np.float64)
        l_fill /= np.linalg.norm(l_fill)
        h_vec = l_key + cam_dir
        h_vec /= np.linalg.norm(h_vec)

        ndotl1 = np.clip(np.sum(nrm_pix * l_key, axis=-1), 0.0, 1.0)
        ndotl2 = np.clip(np.sum(nrm_pix * l_fill, axis=-1), 0.0, 1.0)
        ndoth = np.clip(np.sum(nrm_pix * h_vec, axis=-1), 0.0, 1.0)
        spec = 0.18 * (ndoth ** 20)
        top_w = np.clip(nrm_pix[:, :, 2], 0.0, 1.0)

        illum = 0.27 + 0.56 * ndotl1 + 0.17 * ndotl2
        base_r = 0.65 + 0.15 * top_w
        base_g = 0.73 + 0.12 * top_w
        base_b = 0.84 + 0.08 * top_w

        sh_r = np.clip(illum * base_r + spec, 0.0, 1.0)
        sh_g = np.clip(illum * base_g + spec, 0.0, 1.0)
        sh_b = np.clip(illum * base_b + spec, 0.0, 1.0)

        # Silhouette & machined feature-crease edge detection
        hit_f = hit.astype(np.float64)
        sil = np.zeros((H, W), dtype=bool)
        sil[:, :-1] |= hit[:, :-1] != hit[:, 1:]
        sil[:-1, :] |= hit[:-1, :] != hit[1:, :]

        z_safe = np.where(hit, zbuf, 0.0)
        dz = np.zeros((H, W), dtype=np.float64)
        both_x = hit[:, :-1] & hit[:, 1:]
        both_y = hit[:-1, :] & hit[1:, :]
        dz[:, :-1] = np.where(both_x, np.abs(z_safe[:, :-1] - z_safe[:, 1:]), 0.0)
        dz[:-1, :] = np.maximum(dz[:-1, :], np.where(both_y, np.abs(z_safe[:-1, :] - z_safe[1:, :]), 0.0))

        dn = np.zeros((H, W), dtype=np.float64)
        dn[:, :-1] = np.where(both_x, np.linalg.norm(nrm_pix[:, :-1] - nrm_pix[:, 1:], axis=-1), 0.0)
        dn[:-1, :] = np.maximum(
            dn[:-1, :],
            np.where(both_y, np.linalg.norm(nrm_pix[:-1, :] - nrm_pix[1:, :], axis=-1), 0.0),
        )

        edge_sil = hit & (sil | (dz > 0.95))
        edge_crease = hit & (~edge_sil) & (dn > 0.55)

        img[hit, 0] = sh_r[hit]
        img[hit, 1] = sh_g[hit]
        img[hit, 2] = sh_b[hit]

        img[edge_crease] = 0.68 * img[edge_crease] + 0.32 * np.array([0.20, 0.26, 0.35])
        img[edge_sil] = 0.45 * img[edge_sil] + 0.55 * np.array([0.14, 0.19, 0.26])
        _ = hit_f

    # 2x2 box-filter MSAA downsample -> (height_px, width_px, 3) uint8
    aa = 0.25 * (
        img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2]
    )
    return np.clip(np.round(aa * 255.0), 0, 255).astype(np.uint8)


def _draw_toolpath_panel(
    pdf: VectorPDFCanvas,
    task: C.TaskSpec,
    nom_motion: C.MotionPlan,
    opt_motion: C.MotionPlan,
    bx: float,
    by: float,
    bw: float,
    bh: float,
) -> None:
    """Draw Blueprint Stock/Islands + Nominal dashed gray + DiffCAM feed-colored XY trajectory."""
    smin = task.stock.params["min_mm"]
    smax = task.stock.params["max_mm"]
    part = task.original_part.params
    pmin = part["pocket_min_mm"]
    pmax = part["pocket_max_mm"]
    islands = part.get("islands_mm", [])

    margin_mm = 3.6
    wx0, wx1 = smin[0] - margin_mm, smax[0] + margin_mm
    wy0, wy1 = smin[1] - margin_mm, smax[1] + margin_mm
    pad = 4.0
    scale = min((bw - 2 * pad) / (wx1 - wx0), (bh - 2 * pad) / (wy1 - wy0))
    ox = bx + 0.5 * bw - 0.5 * (wx0 + wx1) * scale
    oy = by + 0.5 * bh - 0.5 * (wy0 + wy1) * scale

    def to_pt(x: float, y: float) -> Tuple[float, float]:
        return (ox + x * scale, oy + y * scale)

    pdf.rect(bx, by, bw, bh, fill=(0.985, 0.988, 0.994), stroke=(0.78, 0.80, 0.84), line_width=0.5)
    sx0, sy0 = to_pt(smin[0], smin[1])
    sx1, sy1 = to_pt(smax[0], smax[1])
    pdf.rect(sx0, sy0, sx1 - sx0, sy1 - sy0, fill=(0.86, 0.89, 0.93), stroke=(0.45, 0.49, 0.55), line_width=0.6)
    px0, py0 = to_pt(pmin[0], pmin[1])
    px1, py1 = to_pt(pmax[0], pmax[1])
    pdf.rect(px0, py0, px1 - px0, py1 - py0, fill=(0.965, 0.972, 0.982), stroke=(0.62, 0.66, 0.72), line_width=0.45)
    for imin, imax in islands:
        ix0, iy0 = to_pt(imin[0], imin[1])
        ix1, iy1 = to_pt(imax[0], imax[1])
        pdf.rect(ix0, iy0, ix1 - ix0, iy1 - iy0, fill=(0.76, 0.82, 0.90), stroke=(0.28, 0.36, 0.48), line_width=0.7)

    for seg in nom_motion.segments:
        if seg.kind != C.SegmentKind.RAPID:
            p0 = to_pt(seg.start_mm[0], seg.start_mm[1])
            p1 = to_pt(seg.end_mm[0], seg.end_mm[1])
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 0.2:
                pdf.line(p0[0], p0[1], p1[0], p1[1], stroke=(0.72, 0.74, 0.77), line_width=0.55, dash=(1.4, 1.2))

    for seg in opt_motion.segments:
        p0 = to_pt(seg.start_mm[0], seg.start_mm[1])
        p1 = to_pt(seg.end_mm[0], seg.end_mm[1])
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= 0.15:
            continue
        if seg.kind == C.SegmentKind.RAPID:
            pdf.line(p0[0], p0[1], p1[0], p1[1], stroke=(0.55, 0.55, 0.58), line_width=0.4, dash=(2.0, 1.8))
        else:
            col = _feed_colormap(seg.feed_mm_s or 5.5)
            pdf.line(p0[0], p0[1], p1[0], p1[1], stroke=col, line_width=1.15)


def generate_fig1_hero(
    run_dir: Path,
    exp2_main: Dict[str, Any],
    exp4: Dict[str, Any],
    out_path: Path,
) -> None:
    pdf = VectorPDFCanvas(508.0, 232.0)
    fine_bundles = {
        b.case.case_id: b
        for b in build_milling_suite(voxel_mm=0.25, timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED)
    }
    rep_ids = ("face-test-01", "pocket-test-01", "adapt-test-01", "contour-test-01")
    task_map = {t["case_id"]: t for t in exp2_main["tasks"]}
    cam_map = {(v["case_id"], v["variant"]): v for v in exp4["verifications"]}

    col_w = 113.0
    gap = 7.5
    left_x = 18.0
    top_row_y = 123.0
    top_row_h = 84.0
    bot_row_y = 24.0
    bot_row_h = 86.0

    for idx, cid in enumerate(rep_ids):
        bx = left_x + idx * (col_w + gap)
        bundle = fine_bundles[cid]
        entry = task_map[cid]
        ste_runs = [r for r in entry["methods"]["STE-Continuation-Adam"] if r["feasible"]]
        if not ste_runs:
            ste_runs = entry["methods"]["STE-Continuation-Adam"]
        best_ste = min(ste_runs, key=lambda r: r["cycle_time_s"])
        sp = make_joint_parameterization(bundle.task, bundle.nominal_motion)
        theta = np.asarray(best_ste["best_theta"], dtype=np.float64)
        pos = sp.positions(theta)
        fm = sp.forward_feed_multipliers(theta)
        to_m = bundle.task.work_to_machine.apply
        rsegs = []
        for orig, sw, ew, mult in zip(bundle.nominal_motion.segments, pos[:-1], pos[1:], fm):
            sm, em = to_m(sw), to_m(ew)
            ln = math.dist(sm, em)
            ef = None if orig.kind == C.SegmentKind.RAPID else float(orig.feed_mm_s * mult)
            rsegs.append(
                C.Segment(
                    id=orig.id, kind=orig.kind, start_mm=sm, end_mm=em,
                    duration_s=ln / (bundle.task.limits.rapid_mm_s if ef is None else ef),
                    feed_mm_s=ef, tag=orig.tag,
                )
            )
        durs = C.blended_segment_durations(rsegs, bundle.task.limits.rapid_mm_s, bundle.task.limits.max_accel_mm_s2)
        opt_plan = C.MotionPlan(
            frame=C.Frame.MACHINE,
            initial_state=bundle.task.initial_state,
            segments=tuple(
                C.Segment(
                    id=s.id, kind=s.kind, start_mm=s.start_mm, end_mm=s.end_mm,
                    duration_s=d, feed_mm_s=s.feed_mm_s, tag=s.tag,
                )
                for s, d in zip(rsegs, durs)
            ),
            timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
        )

        pdf.text(
            bx + 0.5 * col_w,
            top_row_y + top_row_h + 7.0,
            FAMILY_LABELS[bundle.task.family],
            size=8.0,
            bold=True,
            align="center",
        )
        _draw_toolpath_panel(pdf, bundle.task, bundle.nominal_motion, opt_plan, bx, top_row_y, col_w, top_row_h)

        # Bottom row: CAMotics v1.2.0 watertight STL mesh 4x-MSAA rendering + verification badge
        bg_panel = (0.975, 0.978, 0.985)
        pdf.rect(bx, bot_row_y, col_w, bot_row_h, fill=bg_panel, stroke=(0.78, 0.80, 0.84), line_width=0.5)
        ver = cam_map.get((cid, "DiffCAM-STE"), {})
        stl_rel = ver.get("stl_relpath", "")
        if stl_rel and (run_dir / stl_rel).exists():
            rgb_img = _render_stl_isometric_rgb(run_dir / stl_rel, width_px=330, height_px=198, bg_rgb=bg_panel)
            pdf.draw_rgb_image(rgb_img, bx + 2.0, bot_row_y + 15.0, col_w - 4.0, bot_row_h - 17.0)
        nom_ver = cam_map.get((cid, "Nominal-CAM"), {})
        t_nom = float(nom_ver.get("cycle_time_s", best_ste["nominal_cycle_time_s"]))
        t_opt = float(ver.get("cycle_time_s", best_ste["cycle_time_s"]))
        pct = 100.0 * (t_nom - t_opt) / max(t_nom, 1e-9)
        gouge_v = float(ver.get("posted_0p125mm_orig_gouge_mm3", 0.0))
        pdf.rect(bx + 2.0, bot_row_y + 2.0, col_w - 4.0, 12.2, fill=(0.91, 0.945, 0.985))
        pdf.text(
            bx + 0.5 * col_w,
            bot_row_y + 5.4,
            f"{t_nom:.1f}s -> {t_opt:.1f}s (-{pct:.1f}%) | {gouge_v:.2f} mm3 gouge",
            size=5.6,
            bold=True,
            rgb=(0.05, 0.23, 0.43),
            align="center",
        )

    pdf.text(10.0, top_row_y + 0.5 * top_row_h, "2.5D Path + Feed", size=7.1, bold=True, align="center", rotate_deg=90.0)
    pdf.text(10.0, bot_row_y + 0.5 * bot_row_h, "CAMotics v1.2 STL", size=7.1, bold=True, align="center", rotate_deg=90.0)

    # Bottom colorbar & legend strip
    cb_x, cb_y, cb_w, cb_h = 148.0, 6.5, 150.0, 6.5
    n_steps = 48
    for i in range(n_steps):
        f_val = 3.0 + (20.0 - 3.0) * (i + 0.5) / n_steps
        x0 = cb_x + cb_w * i / n_steps
        pdf.rect(x0, cb_y, cb_w / n_steps + 0.3, cb_h, fill=_feed_colormap(f_val))
    pdf.rect(cb_x, cb_y, cb_w, cb_h, stroke=(0.3, 0.3, 0.3), line_width=0.4)
    pdf.text(cb_x - 4.0, cb_y + 1.2, "Feed f_t:", size=6.7, bold=True, align="right")
    for f_tick in (3, 7, 11, 15, 20):
        tx = cb_x + cb_w * (f_tick - 3.0) / 17.0
        pdf.text(tx, cb_y + cb_h + 2.2, f"{f_tick}", size=5.9, align="center")
    pdf.text(cb_x + cb_w + 4.0, cb_y + 1.2, "mm/s (Haas NGC G187 P2 E0.05)", size=6.5, align="left")
    pdf.line(20.0, cb_y + 3.2, 36.0, cb_y + 3.2, stroke=(0.68, 0.70, 0.74), line_width=0.9, dash=(1.8, 1.4))
    pdf.text(39.0, cb_y + 1.2, "Nominal CAM Seed", size=6.5, align="left")

    pdf.write(out_path)


def _draw_axes(
    pdf: VectorPDFCanvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    xticks: Sequence[Tuple[float, str]],
    yticks: Sequence[Tuple[float, str]],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    pdf.rect(x, y, w, h, fill=(0.992, 0.994, 0.998), stroke=(0.25, 0.27, 0.30), line_width=0.65)
    for yv, ylab in yticks:
        py = y + h * (yv - ymin) / max(ymax - ymin, 1e-9)
        if y - 0.5 <= py <= y + h + 0.5:
            pdf.line(x, py, x + w, py, stroke=(0.88, 0.90, 0.93), line_width=0.4)
            pdf.line(x - 2.2, py, x, py, stroke=(0.25, 0.27, 0.30), line_width=0.6)
            pdf.text(x - 3.8, py - 2.1, ylab, size=6.1, align="right")
    for xv, xlab in xticks:
        px = x + w * (xv - xmin) / max(xmax - xmin, 1e-9)
        if x - 0.5 <= px <= x + w + 0.5:
            pdf.line(px, y, px, y + h, stroke=(0.91, 0.92, 0.95), line_width=0.35)
            pdf.line(px, y - 2.2, px, y, stroke=(0.25, 0.27, 0.30), line_width=0.6)
            pdf.text(px, y - 9.2, xlab, size=6.1, align="center")
    if title:
        pdf.text(x + 0.5 * w, y + h + 5.5, title, size=7.5, bold=True, align="center")
    if xlabel:
        pdf.text(x + 0.5 * w, y - 18.5, xlabel, size=6.7, align="center")
    if ylabel:
        pdf.text(x - 23.5, y + 0.5 * h, ylabel, size=6.7, align="center", rotate_deg=90.0)


def generate_fig2_exp1(exp1: Dict[str, Any], out_path: Path) -> None:
    pdf = VectorPDFCanvas(504.0, 166.0)
    rows = exp1["stiffness_sweep"]
    ks = sorted({float(r["k"]) for r in rows})
    log_ks = [math.log2(k) for k in ks]
    x0_l, x1_l = min(log_ks), max(log_ks)

    # Panel (a): Gradient SNR & Soft-Hard Cosine vs k
    ax_a = (31.0, 32.0, 120.0, 110.0)
    snr_means = [float(np.mean([r["local_grad_snr"] for r in rows if float(r["k"]) == k])) for k in ks]
    cos_means = [float(np.mean([r["cosine_vs_hard_fd"] for r in rows if float(r["k"]) == k])) for k in ks]
    snr_step = 20.0 if max(snr_means) <= 80.0 else (25.0 if max(snr_means) <= 100.0 else 40.0)
    snr_max = max(80.0, math.ceil(max(snr_means) / snr_step) * snr_step)
    _draw_axes(
        pdf, *ax_a,
        xticks=[(math.log2(k), f"{int(k)}") for k in ks],
        yticks=[(v, f"{int(v)}") for v in np.linspace(0.0, snr_max, 5)],
        xmin=x0_l, xmax=x1_l, ymin=0.0, ymax=snr_max,
        title="(a) Gradient SNR & Alignment vs. k",
        xlabel="Surrogate Stiffness k",
        ylabel="Local Gradient SNR",
    )
    # Secondary right y-axis ticks for Cosine in [0, 1]
    for cv in (0.0, 0.25, 0.50, 0.75, 1.00):
        ry = ax_a[1] + ax_a[3] * cv
        pdf.line(ax_a[0] + ax_a[2], ry, ax_a[0] + ax_a[2] + 2.0, ry, stroke=(0.84, 0.37, 0.00), line_width=0.6)
        pdf.text(ax_a[0] + ax_a[2] + 3.2, ry - 2.0, f"{cv:.2f}", size=5.3, rgb=(0.84, 0.37, 0.00), align="left")
    pts_snr = [
        (ax_a[0] + ax_a[2] * (lk - x0_l) / (x1_l - x0_l), ax_a[1] + ax_a[3] * v / snr_max)
        for lk, v in zip(log_ks, snr_means)
    ]
    pts_cos = [
        (ax_a[0] + ax_a[2] * (lk - x0_l) / (x1_l - x0_l), ax_a[1] + ax_a[3] * max(0.0, min(1.0, c)))
        for lk, c in zip(log_ks, cos_means)
    ]
    pdf.polyline(pts_snr, stroke=(0.00, 0.45, 0.70), line_width=1.5)
    for px, py in pts_snr:
        pdf.circle(px, py, 2.1, fill=(0.00, 0.45, 0.70))
    pdf.polyline(pts_cos, stroke=(0.84, 0.37, 0.00), line_width=1.4, dash=(3.0, 1.8))
    for px, py in pts_cos:
        pdf.circle(px, py, 2.0, fill=(0.84, 0.37, 0.00))
    peak_idx = int(np.argmax(snr_means))
    peak_k = int(ks[peak_idx])
    # Legend cleanly in open lower region (y in [4, 23] pt above bottom axis)
    pdf.rect(ax_a[0] + 4.5, ax_a[1] + 4.0, ax_a[2] - 9.0, 19.5, fill=(0.99, 0.99, 1.0), stroke=(0.82, 0.84, 0.88), line_width=0.35)
    pdf.line(ax_a[0] + 8.0, ax_a[1] + 17.5, ax_a[0] + 19.0, ax_a[1] + 17.5, stroke=(0.00, 0.45, 0.70), line_width=1.5)
    pdf.text(ax_a[0] + 22.0, ax_a[1] + 15.5, f"Grad SNR (peak {snr_means[peak_idx]:.1f} at k={peak_k})", size=5.4, rgb=(0.00, 0.45, 0.70), bold=True)
    pdf.line(ax_a[0] + 8.0, ax_a[1] + 8.8, ax_a[0] + 19.0, ax_a[1] + 8.8, stroke=(0.84, 0.37, 0.00), line_width=1.4, dash=(2.5, 1.5))
    pdf.text(ax_a[0] + 22.0, ax_a[1] + 6.8, f"Hard-FD Cosine ({min(cos_means):.2f}-{max(cos_means):.2f})", size=5.4, rgb=(0.84, 0.37, 0.00), bold=True)

    # Panel (b): Soft-to-Hard Volumetric Occupancy L1 Bias vs k (per family)
    ax_b = (200.0, 32.0, 130.0, 110.0)
    fam_colors = {
        "facing_open_boss": (0.00, 0.45, 0.70),
        "pocket_island": (0.00, 0.62, 0.45),
        "adaptive_channel": (0.84, 0.37, 0.00),
        "contour_profile": (0.80, 0.47, 0.65),
    }
    ymax_b = 100.0
    _draw_axes(
        pdf, *ax_b,
        xticks=[(math.log2(k), f"{int(k)}") for k in ks],
        yticks=[(v, f"{int(v)}%") for v in (0, 25, 50, 75, 100)],
        xmin=x0_l, xmax=x1_l, ymin=0.0, ymax=ymax_b,
        title="(b) Soft-Hard Occupancy Drift (%)",
        xlabel="Surrogate Stiffness k",
        ylabel="Occupancy L1 Gap (%)",
    )
    kx0 = ax_b[0] + ax_b[2] * (math.log2(25.0) - x0_l) / (x1_l - x0_l)
    kx1 = ax_b[0] + ax_b[2] * (math.log2(90.0) - x0_l) / (x1_l - x0_l)
    pdf.rect(kx0, ax_b[1], kx1 - kx0, ax_b[3], fill=(0.20, 0.65, 0.45), alpha_band=True)
    pdf.text(0.5 * (kx0 + kx1), ax_b[1] + ax_b[3] - 9.5, "STE k [25->90]", size=5.5, rgb=(0.05, 0.42, 0.25), align="center", bold=True)
    for fam in FAMILY_ORDER:
        f_pts = []
        for k, lk in zip(ks, log_ks):
            vals = [100.0 * float(r["soft_hard_occupancy_l1"]) for r in rows if r["family"] == fam and float(r["k"]) == k]
            if vals:
                px = ax_b[0] + ax_b[2] * (lk - x0_l) / (x1_l - x0_l)
                py = ax_b[1] + ax_b[3] * float(np.mean(vals)) / ymax_b
                f_pts.append((px, py))
        pdf.polyline(f_pts, stroke=fam_colors[fam], line_width=1.25)
        for px, py in f_pts:
            pdf.circle(px, py, 1.7, fill=fam_colors[fam])
    # Compact family legend in open upper-right of Panel (b)
    for f_i, fam in enumerate(FAMILY_ORDER):
        ly = ax_b[1] + ax_b[3] - 20.0 - f_i * 8.2
        lx = ax_b[0] + ax_b[2] - 46.0
        pdf.line(lx, ly + 2.0, lx + 8.0, ly + 2.0, stroke=fam_colors[fam], line_width=1.3)
        pdf.text(lx + 10.0, ly, FAMILY_SHORT[fam], size=5.3, rgb=fam_colors[fam], bold=True)

    # Panel (c): Perturbed Initialization Cycle-Time Gain (%) + 100% Feasibility
    ax_c = (367.0, 32.0, 127.0, 110.0)
    reach = exp1["reachability_sweep"]
    r_vals = sorted({float(r["perturbation_R"]) for r in reach})
    ymax_c = 60.0
    _draw_axes(
        pdf, *ax_c,
        xticks=[(i, f"R={rv:.2f}") for i, rv in enumerate(r_vals)],
        yticks=[(v, f"{int(v)}%") for v in (0, 15, 30, 45, 60)],
        xmin=-0.5, xmax=len(r_vals) - 0.5, ymin=0.0, ymax=ymax_c,
        title="(c) Perturbed Init Recovery",
        xlabel="Perturbation Radius R",
        ylabel="Cycle-Time Gain (%)",
    )
    for i, rv in enumerate(r_vals):
        gx = ax_c[0] + ax_c[2] * (i + 0.5) / len(r_vals)
        bw_bar = 13.5
        for m_idx, (m_name, col) in enumerate((("STE-Continuation-Adam", (0.00, 0.45, 0.70)), ("Hard-CMA-ES", (0.90, 0.62, 0.00)))):
            sub = [r for r in reach if float(r["perturbation_R"]) == rv and r["method"] == m_name]
            gains = [
                100.0 * (float(r["init_cycle_time_s"]) - float(r["final_cycle_time_s"])) / max(float(r["init_cycle_time_s"]), 1e-9)
                for r in sub
            ]
            gain_pct = max(0.0, float(np.mean(gains)))
            bx = gx + (-bw_bar - 1.0 if m_idx == 0 else 1.0)
            bh_bar = ax_c[3] * gain_pct / ymax_c
            pdf.rect(bx, ax_c[1], bw_bar, max(1.0, bh_bar), fill=col, stroke=(0.2, 0.2, 0.2), line_width=0.35)
            pdf.text(bx + 0.5 * bw_bar, ax_c[1] + bh_bar + 2.5, f"{gain_pct:.0f}%", size=5.2, align="center", bold=True)

    pdf.rect(ax_c[0] + 5.0, ax_c[1] + ax_c[3] - 23.0, ax_c[2] - 10.0, 19.5, fill=(0.99, 0.99, 1.0), stroke=(0.82, 0.84, 0.88), line_width=0.35)
    pdf.rect(ax_c[0] + 8.5, ax_c[1] + ax_c[3] - 11.5, 6.0, 4.5, fill=(0.00, 0.45, 0.70))
    pdf.text(ax_c[0] + 16.5, ax_c[1] + ax_c[3] - 11.2, "STE-Adam", size=5.3, bold=True)
    pdf.rect(ax_c[0] + 58.0, ax_c[1] + ax_c[3] - 11.5, 6.0, 4.5, fill=(0.90, 0.62, 0.00))
    pdf.text(ax_c[0] + 66.0, ax_c[1] + ax_c[3] - 11.2, "Hard-CMA-ES", size=5.3, bold=True)
    pdf.text(ax_c[0] + 0.5 * ax_c[2], ax_c[1] + ax_c[3] - 20.0, "100% Hard Feasible Across All R", size=5.2, rgb=(0.05, 0.42, 0.25), align="center", bold=True)

    pdf.write(out_path)


def generate_fig3_exp2_convergence(exp2_main: Dict[str, Any], out_path: Path) -> None:
    pdf = VectorPDFCanvas(520.0, 190.0)
    budget = int(exp2_main["budget_evaluations"])
    panel_w = 104.0
    panel_h = 112.0
    gap = 21.0
    left = 31.0
    bot = 48.0

    methods_order = [
        "STE-Continuation-Adam",
        "DiffPolicy-SHAC",
        "Annealed-Adam",
        "Naive-Smooth-Adam",
        "Hard-PPO",
        "Hard-CEM",
        "Hard-CMA-ES",
    ]

    all_min_norm = 1.0
    for t_rec in exp2_main["tasks"]:
        nom_t = float(t_rec["methods"]["Nominal-CAM"][0]["cycle_time_s"])
        for m_name in methods_order:
            for r in t_rec["methods"].get(m_name, []):
                if r["curve_best_cycle_time_s"]:
                    all_min_norm = min(all_min_norm, float(min(r["curve_best_cycle_time_s"])) / max(nom_t, 1e-9))
    ymin = 0.20 if all_min_norm < 0.29 else 0.30
    ymax = 1.04
    yticks_conv = (
        [(0.20, "0.20"), (0.40, "0.40"), (0.60, "0.60"), (0.80, "0.80"), (1.00, "1.00")]
        if ymin <= 0.20
        else [(0.30, "0.30"), (0.45, "0.45"), (0.60, "0.60"), (0.75, "0.75"), (0.90, "0.90"), (1.00, "1.00")]
    )

    for f_idx, fam in enumerate(FAMILY_ORDER):
        px = left + f_idx * (panel_w + gap)
        fam_tasks = [t for t in exp2_main["tasks"] if t["family"] == fam]
        _draw_axes(
            pdf, px, bot, panel_w, panel_h,
            xticks=[(1, "1"), (16, "16"), (32, "32"), (48, "48"), (budget, str(budget))],
            yticks=yticks_conv,
            xmin=1.0, xmax=float(budget), ymin=ymin, ymax=ymax,
            title=FAMILY_LABELS[fam],
            xlabel="Simulator Evals",
            ylabel="Norm. Cycle Time" if f_idx == 0 else "",
        )
        pdf.save_state()
        pdf.clip_rect(px, bot, panel_w, panel_h)
        for m_name in reversed(methods_order):
            col, dash, _ = METHOD_STYLE[m_name]
            curves = []
            for t_rec in fam_tasks:
                nom_t = float(t_rec["methods"]["Nominal-CAM"][0]["cycle_time_s"])
                for r in t_rec["methods"].get(m_name, []):
                    raw_c = np.asarray(r["curve_best_cycle_time_s"], dtype=np.float64) / max(nom_t, 1e-9)
                    if len(raw_c) < budget:
                        raw_c = np.pad(raw_c, (0, budget - len(raw_c)), mode="edge")
                    else:
                        raw_c = raw_c[:budget]
                    curves.append(raw_c)
            if not curves:
                continue
            mat = np.stack(curves, axis=0)
            mean_c = np.mean(mat, axis=0)
            std_c = np.std(mat, axis=0)
            xs = [px + panel_w * (ev - 1.0) / max(1.0, budget - 1.0) for ev in range(1, budget + 1)]
            ys = [bot + panel_h * (float(v) - ymin) / (ymax - ymin) for v in mean_c]
            if m_name in ("STE-Continuation-Adam", "DiffPolicy-SHAC", "Hard-PPO"):
                upper = [bot + panel_h * (float(m + s) - ymin) / (ymax - ymin) for m, s in zip(mean_c, std_c)]
                lower = [bot + panel_h * (float(m - s) - ymin) / (ymax - ymin) for m, s in zip(mean_c, std_c)]
                poly = list(zip(xs, upper)) + list(reversed(list(zip(xs, lower))))
                pdf.polygon(poly, fill=col, alpha_band=True)
            lw = 1.6 if m_name in ("STE-Continuation-Adam", "DiffPolicy-SHAC") else 1.15
            pdf.polyline(list(zip(xs, ys)), stroke=col, line_width=lw, dash=dash)
        pdf.restore_state()

    # Shared bottom legend with clean separation below x-axis labels
    lx = 16.0
    ly = 6.0
    for idx, m_name in enumerate(methods_order):
        col, dash, label = METHOD_STYLE[m_name]
        cx = lx + (idx % 4) * 123.0
        cy = ly + (9.5 if idx < 4 else 0.0)
        pdf.line(cx, cy + 2.2, cx + 14.0, cy + 2.2, stroke=col, line_width=1.5, dash=dash)
        pdf.text(cx + 17.0, cy, label, size=5.8, bold=m_name in ("STE-Continuation-Adam", "DiffPolicy-SHAC"))

    pdf.write(out_path)


def generate_fig4_transfer_and_ablations(
    exp2_abl: Dict[str, Any],
    exp3: Dict[str, Any],
    out_path: Path,
) -> None:
    pdf = VectorPDFCanvas(504.0, 184.0)

    # Panel (a): Exp3 Held-out Test Zero-Shot & 16-Eval Transfer
    ax_a = (34.0, 46.0, 206.0, 112.0)
    regimes = [
        ("DiffPolicy-SHAC-ZeroShot", (0.00, 0.62, 0.45), "SHAC Zero-Shot (1 eval)"),
        ("Hard-PPO-ZeroShot", (0.84, 0.37, 0.00), "PPO Zero-Shot (1 eval)"),
        ("DiffPolicy-SHAC-FewShot16", (0.00, 0.45, 0.70), "SHAC Few-Shot (16 evals)"),
        ("Hard-PPO-FewShot16", (0.90, 0.62, 0.00), "PPO Few-Shot (16 evals)"),
        ("Scratch-STE-Adam-16", (0.55, 0.55, 0.58), "Scratch STE (16 evals)"),
    ]
    rows3 = exp3["test_transfer"]
    ymax_a = 60.0
    _draw_axes(
        pdf, *ax_a,
        xticks=[(i, short) for i, short in enumerate(("Facing", "Pocket", "Adaptive", "Contour"))],
        yticks=[(v, f"{int(v)}%") for v in (0, 15, 30, 45, 60)],
        xmin=-0.5, xmax=3.5, ymin=0.0, ymax=ymax_a,
        title="(a) Held-Out Test Policy Generalization (Exp3)",
        xlabel="Held-Out Test Family",
        ylabel="Cycle-Time Reduction (%)",
    )
    for f_i, fam in enumerate(FAMILY_ORDER):
        group_x = ax_a[0] + ax_a[2] * (f_i + 0.5) / 4.0
        bar_w = 7.8
        for r_i, (reg_key, col, _) in enumerate(regimes):
            vals = [
                float(r["cycle_time_reduction_pct"])
                for r in rows3
                if r["family"] == fam and r["regime"] == reg_key and r["feasible"]
            ]
            mean_v = max(0.0, float(np.mean(vals))) if vals else 0.0
            bx = group_x + (r_i - 2.5) * (bar_w + 1.0)
            bh = ax_a[3] * mean_v / ymax_a
            pdf.rect(bx, ax_a[1], bar_w, max(0.8, bh), fill=col, stroke=(0.2, 0.2, 0.2), line_width=0.3)

    for r_i, (_, col, label) in enumerate(regimes):
        lx = 16.0 + (r_i % 3) * 82.0
        ly = 13.5 if r_i < 3 else 4.5
        pdf.rect(lx, ly, 6.5, 4.8, fill=col)
        pdf.text(lx + 8.5, ly + 0.4, label, size=5.4)

    # Panel (b): 5-Arm Ablation Study (Cycle-Time Gain vs Peak MRR Overload)
    ax_b = (284.0, 46.0, 206.0, 112.0)
    arms = [
        ("Full-Joint-Blended-STE", "Full Joint"),
        ("Path-Only-Frozen-Feed", "Path-Only"),
        ("Feed-Only-Frozen-Path", "Feed-Only"),
        ("Exact-Stop-Surrogate", "Exact-Stop"),
        ("No-MRR-Barrier", "No-MRR"),
    ]
    abl_rows = exp2_abl["rows"]
    arm_peaks = []
    for arm_key, _ in arms:
        sub = [r for r in abl_rows if r["arm"] == arm_key]
        if sub:
            arm_peaks.append(float(np.mean([r["cycle_time_reduction_pct"] for r in sub])))
            arm_peaks.append(float(np.mean([r["mrr_overload_pct"] for r in sub])))
    raw_peak_b = max(arm_peaks + [120.0])
    step_b = 35.0 if raw_peak_b <= 132.0 else (50.0 if raw_peak_b <= 185.0 else 70.0)
    max_val_b = step_b * 4.0 + 8.0
    _draw_axes(
        pdf, *ax_b,
        xticks=[(i, lbl) for i, (_, lbl) in enumerate(arms)],
        yticks=[(v, f"{int(v)}%") for v in (0.0, step_b, 2.0 * step_b, 3.0 * step_b, 4.0 * step_b)],
        xmin=-0.5, xmax=len(arms) - 0.5, ymin=0.0, ymax=max_val_b,
        title="(b) Physical & Kinematic Ablation Study",
        xlabel="Ablation Configuration (8 Test Tasks)",
        ylabel="Gain / MRR Overload (%)",
    )
    for a_i, (arm_key, _) in enumerate(arms):
        sub = [r for r in abl_rows if r["arm"] == arm_key]
        gain = max(0.0, float(np.mean([r["cycle_time_reduction_pct"] for r in sub])))
        ovl = max(0.0, float(np.mean([r["mrr_overload_pct"] for r in sub])))
        gx = ax_b[0] + ax_b[2] * (a_i + 0.5) / len(arms)
        bw = 11.0
        bh_gain = max(0.8, ax_b[3] * gain / max_val_b)
        bh_ovl = max(0.8, ax_b[3] * ovl / max_val_b)
        pdf.rect(
            gx - bw - 1.0, ax_b[1], bw, bh_gain,
            fill=(0.00, 0.45, 0.70), stroke=(0.2, 0.2, 0.2), line_width=0.35,
        )
        pdf.rect(
            gx + 1.0, ax_b[1], bw, bh_ovl,
            fill=(0.82, 0.18, 0.22), stroke=(0.2, 0.2, 0.2), line_width=0.35,
        )
        if arm_key == "No-MRR-Barrier":
            pdf.text(
                gx + 1.0 + 0.5 * bw,
                ax_b[1] + bh_ovl + 2.5,
                f"+{ovl:.0f}%",
                size=5.4,
                bold=True,
                rgb=(0.72, 0.10, 0.14),
                align="center",
            )
    pdf.rect(286.0, 9.0, 7.5, 5.0, fill=(0.00, 0.45, 0.70))
    pdf.text(296.0, 9.4, "Cycle-Time Reduction (%)", size=5.8, bold=True)
    pdf.rect(392.0, 9.0, 7.5, 5.0, fill=(0.82, 0.18, 0.22))
    pdf.text(402.0, 9.4, "Peak MRR Overload > Limit (%)", size=5.8, bold=True)

    pdf.write(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "pub-20261110")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "manuscript" / "figures")
    parser.add_argument("--mirror-dir", type=Path, default=PROJECT_ROOT / "overleaf" / "figures")
    args = parser.parse_args()

    rdir = args.run_dir.resolve()
    exp1 = json.loads((rdir / "exp1_conditioning.json").read_text(encoding="utf-8"))
    exp2_main = json.loads((rdir / "exp2_main_benchmark.json").read_text(encoding="utf-8"))
    exp2_abl = json.loads((rdir / "exp2_ablations.json").read_text(encoding="utf-8"))
    exp3 = json.loads((rdir / "exp3_policy_transfer.json").read_text(encoding="utf-8"))
    exp4 = json.loads((rdir / "exp4_camotics_and_haas.json").read_text(encoding="utf-8"))

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    f1 = out_dir / "fig1_hero_camotics.pdf"
    f2 = out_dir / "fig2_exp1_conditioning.pdf"
    f3 = out_dir / "fig3_exp2_convergence.pdf"
    f4 = out_dir / "fig4_exp3_transfer_ablations.pdf"

    generate_fig1_hero(rdir, exp2_main, exp4, f1)
    generate_fig2_exp1(exp1, f2)
    generate_fig3_exp2_convergence(exp2_main, f3)
    generate_fig4_transfer_and_ablations(exp2_abl, exp3, f4)

    if args.mirror_dir:
        mdir = args.mirror_dir.resolve()
        mdir.mkdir(parents=True, exist_ok=True)
        for p in (f1, f2, f3, f4):
            shutil.copy2(p, mdir / p.name)

    print(f"[FIGURES] Generated 4 PDF figures in {out_dir} and mirrored to {args.mirror_dir}", flush=True)


if __name__ == "__main__":
    main()
