// SPDX-License-Identifier: MIT
// Clean-room WebGL1 Live 2.5D Dexel Stock Carving, CWE & Chipload Simulator.
// TODO(security): Scoped as a read-only local 127.0.0.1 single-user viewer with zero cookies,
// zero storage APIs, and strict createElement/createElementNS/textContent DOM construction.

(function () {
  "use strict";

  const MAX_STL_TRIANGLES = 2000000;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const DEXEL_PITCH_MM = 0.35;
  const KEYFRAME_STRIDE = 16;

  const STOCK_PALETTES = {
    green: { skin: [0.22, 0.56, 0.26], alloy: [0.73, 0.79, 0.86], wall: [0.64, 0.71, 0.79] },
    silver: { skin: [0.56, 0.62, 0.69], alloy: [0.80, 0.84, 0.90], wall: [0.68, 0.73, 0.80] },
    blue: { skin: [0.20, 0.44, 0.68], alloy: [0.75, 0.80, 0.87], wall: [0.64, 0.71, 0.79] },
    amber: { skin: [0.68, 0.49, 0.18], alloy: [0.76, 0.81, 0.87], wall: [0.66, 0.72, 0.79] },
  };

  // --- Safe HTML & SVG DOM Helpers (zero innerHTML / eval / inline styles) ---
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined) continue;
        if (k === "className") {
          node.className = String(v);
        } else if (k === "text") {
          node.textContent = String(v);
        } else {
          node.setAttribute(k, String(v));
        }
      }
    }
    if (Array.isArray(children)) {
      for (const child of children) {
        if (child === null || child === undefined) continue;
        if (typeof child === "string") {
          node.appendChild(document.createTextNode(child));
        } else {
          node.appendChild(child);
        }
      }
    }
    return node;
  }

  function svgEl(tag, attrs, children) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined) continue;
        node.setAttribute(k, String(v));
      }
    }
    if (Array.isArray(children)) {
      for (const child of children) {
        if (child) node.appendChild(child);
      }
    }
    return node;
  }

  function showError(msg) {
    const banner = document.getElementById("error-banner");
    banner.textContent = String(msg);
    banner.classList.remove("hidden");
  }

  function clearError() {
    const banner = document.getElementById("error-banner");
    banner.textContent = "";
    banner.classList.add("hidden");
  }

  // --- Minimal 4x4 Column-Major Matrix Utilities (Z-up G54 Frame) ---
  function mat4Identity() {
    const m = new Float32Array(16);
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
    return m;
  }

  function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return out;
  }

  function mat4Perspective(fovRad, aspect, near, far) {
    const f = 1.0 / Math.tan(fovRad * 0.5);
    const nf = 1.0 / (near - far);
    const out = new Float32Array(16);
    out[0] = f / Math.max(aspect, 1e-4);
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1.0;
    out[14] = 2.0 * far * near * nf;
    return out;
  }

  function mat4LookAt(eye, target, up) {
    let zx = eye[0] - target[0];
    let zy = eye[1] - target[1];
    let zz = eye[2] - target[2];
    let zLen = Math.hypot(zx, zy, zz) || 1.0;
    zx /= zLen; zy /= zLen; zz /= zLen;

    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    let xLen = Math.hypot(xx, xy, xz);
    if (xLen < 1e-6) {
      xx = 1.0; xy = 0.0; xz = 0.0;
    } else {
      xx /= xLen; xy /= xLen; xz /= xLen;
    }

    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    const out = mat4Identity();
    out[0] = xx; out[4] = xy; out[8] = xz;
    out[1] = yx; out[5] = yy; out[9] = yz;
    out[2] = zx; out[6] = zy; out[10] = zz;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    return out;
  }

  function mat4Translation(tx, ty, tz) {
    const out = mat4Identity();
    out[12] = tx; out[13] = ty; out[14] = tz;
    return out;
  }

  // --- Validated Binary STL Parser ---
  function parseBinaryStl(buffer, tintRgb) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 84) {
      throw new Error("Invalid binary STL payload (< 84 bytes).");
    }
    const view = new DataView(buffer);
    const triCount = view.getUint32(80, true);
    if (triCount > MAX_STL_TRIANGLES || 84 + triCount * 50 !== buffer.byteLength) {
      throw new Error("STL facet count does not match byte length.");
    }
    const positions = new Float32Array(triCount * 9);
    const normals = new Float32Array(triCount * 9);
    const colors = new Float32Array(triCount * 9);
    const cr = tintRgb ? tintRgb[0] : 0.72;
    const cg = tintRgb ? tintRgb[1] : 0.78;
    const cb = tintRgb ? tintRgb[2] : 0.85;

    let offset = 84;
    for (let i = 0; i < triCount; i++) {
      let nx = view.getFloat32(offset, true);
      let ny = view.getFloat32(offset + 4, true);
      let nz = view.getFloat32(offset + 8, true);

      const v0x = view.getFloat32(offset + 12, true);
      const v0y = view.getFloat32(offset + 16, true);
      const v0z = view.getFloat32(offset + 20, true);
      const v1x = view.getFloat32(offset + 24, true);
      const v1y = view.getFloat32(offset + 28, true);
      const v1z = view.getFloat32(offset + 32, true);
      const v2x = view.getFloat32(offset + 36, true);
      const v2y = view.getFloat32(offset + 40, true);
      const v2z = view.getFloat32(offset + 44, true);
      offset += 50;

      let nLen = Math.hypot(nx, ny, nz);
      if (!Number.isFinite(nLen) || nLen < 1e-6) {
        const ax = v1x - v0x, ay = v1y - v0y, az = v1z - v0z;
        const bx = v2x - v0x, by = v2y - v0y, bz = v2z - v0z;
        nx = ay * bz - az * by;
        ny = az * bx - ax * bz;
        nz = ax * by - ay * bx;
        nLen = Math.hypot(nx, ny, nz) || 1.0;
      }
      nx /= nLen; ny /= nLen; nz /= nLen;

      const base = i * 9;
      positions[base] = v0x; positions[base + 1] = v0y; positions[base + 2] = v0z;
      positions[base + 3] = v1x; positions[base + 4] = v1y; positions[base + 5] = v1z;
      positions[base + 6] = v2x; positions[base + 7] = v2y; positions[base + 8] = v2z;

      normals[base] = nx; normals[base + 1] = ny; normals[base + 2] = nz;
      normals[base + 3] = nx; normals[base + 4] = ny; normals[base + 5] = nz;
      normals[base + 6] = nx; normals[base + 7] = ny; normals[base + 8] = nz;

      colors[base] = cr; colors[base + 1] = cg; colors[base + 2] = cb;
      colors[base + 3] = cr; colors[base + 4] = cg; colors[base + 5] = cb;
      colors[base + 6] = cr; colors[base + 7] = cg; colors[base + 8] = cb;
    }
    return { positions, normals, colors, vertexCount: triCount * 3 };
  }

  // --- Clean-Room Real-Time 2.5D Z-Dexel Cutting Stock Simulator ---
  class DexelStockSim {
    constructor(caseData, pitchMm) {
      this.pitch = pitchMm || DEXEL_PITCH_MM;
      this.smin = caseData.stock_min_mm.map(Number);
      this.smax = caseData.stock_max_mm.map(Number);
      this.rCut = Number((caseData.tool && caseData.tool.cutter_radius_mm) || 2.5);
      this.r2 = this.rCut * this.rCut;

      const lx = Math.max(2.0, this.smax[0] - this.smin[0]);
      const ly = Math.max(2.0, this.smax[1] - this.smin[1]);
      this.nx = Math.max(8, Math.round(lx / this.pitch));
      this.ny = Math.max(8, Math.round(ly / this.pitch));
      this.dx = lx / this.nx;
      this.dy = ly / this.ny;
      this.cellArea = this.dx * this.dy;

      const count = this.nx * this.ny;
      this.xs = new Float32Array(this.nx);
      this.ys = new Float32Array(this.ny);
      for (let i = 0; i < this.nx; i++) this.xs[i] = this.smin[0] + (i + 0.5) * this.dx;
      for (let j = 0; j < this.ny; j++) this.ys[j] = this.smin[1] + (j + 0.5) * this.dy;

      this.zTarget = new Float32Array(count);
      this._buildAnalyticTarget(caseData);

      this.zWork = new Float32Array(count);
      this.lastCutSeg = new Int32Array(count);

      // Pre-allocate worst-case triangulation buffers to avoid per-frame GC allocations
      const maxTris = (3 * count + 2 * (this.nx + this.ny) + 4) * 2;
      this.posScratch = new Float32Array(maxTris * 9);
      this.nrmScratch = new Float32Array(maxTris * 9);
      this.colScratch = new Float32Array(maxTris * 9);

      this.segments = [];
      this.keyframes = [];
      this.keyframeCutSeg = [];
      this.lastCarveKey = "";
    }

    _buildAnalyticTarget(caseData) {
      const [bx0, by0, bz0] = this.smin;
      const [bx1, by1, bz1] = this.smax;
      const pmin = (caseData.pocket_min_mm || [bx0, by0, bz1]).map(Number);
      const pmax = (caseData.pocket_max_mm || [bx1, by1, bz1]).map(Number);
      const rc = Number(caseData.corner_radius_mm || 0.0);
      const islands = Array.isArray(caseData.islands_mm) ? caseData.islands_mm : [];

      const xWalledLo = pmin[0] > bx0 + 1e-6;
      const xWalledHi = pmax[0] < bx1 - 1e-6;
      const yWalledLo = pmin[1] > by0 + 1e-6;
      const yWalledHi = pmax[1] < by1 - 1e-6;

      for (let ix = 0; ix < this.nx; ix++) {
        const px = this.xs[ix];
        for (let iy = 0; iy < this.ny; iy++) {
          const py = this.ys[iy];
          let zt = bz1;

          if (px >= pmin[0] && px <= pmax[0] && py >= pmin[1] && py <= pmax[1]) {
            let inCornerFillet = false;
            if (rc > 1e-5) {
              if (xWalledLo && yWalledLo && px < pmin[0] + rc && py < pmin[1] + rc) {
                if (Math.hypot(px - (pmin[0] + rc), py - (pmin[1] + rc)) > rc) inCornerFillet = true;
              }
              if (xWalledHi && yWalledLo && px > pmax[0] - rc && py < pmin[1] + rc) {
                if (Math.hypot(px - (pmax[0] - rc), py - (pmin[1] + rc)) > rc) inCornerFillet = true;
              }
              if (xWalledLo && yWalledHi && px < pmin[0] + rc && py > pmax[1] - rc) {
                if (Math.hypot(px - (pmin[0] + rc), py - (pmax[1] - rc)) > rc) inCornerFillet = true;
              }
              if (xWalledHi && yWalledHi && px > pmax[0] - rc && py > pmax[1] - rc) {
                if (Math.hypot(px - (pmax[0] - rc), py - (pmax[1] - rc)) > rc) inCornerFillet = true;
              }
            }
            if (!inCornerFillet) {
              zt = Math.max(bz0, pmin[2]);
            }
          }

          for (let k = 0; k < islands.length; k++) {
            const lo = islands[k][0];
            const hi = islands[k][1];
            if (px >= lo[0] && px <= hi[0] && py >= lo[1] && py <= hi[1]) {
              zt = Math.max(zt, Number(hi[2]));
            }
          }
          this.zTarget[ix * this.ny + iy] = zt;
        }
      }
    }

    setSegments(segments) {
      this.segments = Array.isArray(segments) ? segments : [];
      this.keyframes = [];
      this.keyframeCutSeg = [];

      const zCur = new Float32Array(this.nx * this.ny);
      zCur.fill(this.smax[2]);
      const cutSeg = new Int32Array(this.nx * this.ny);
      cutSeg.fill(-999);

      this.keyframes.push(zCur.slice());
      this.keyframeCutSeg.push(cutSeg.slice());

      for (let k = 0; k < this.segments.length; k++) {
        const s = this.segments[k];
        if (s.kind !== "rapid") {
          this._sweepLinear(
            zCur,
            cutSeg,
            k,
            s.start[0],
            s.start[1],
            s.start[2],
            s.end[0],
            s.end[1],
            s.end[2]
          );
        }
        if ((k + 1) % KEYFRAME_STRIDE === 0) {
          this.keyframes.push(zCur.slice());
          this.keyframeCutSeg.push(cutSeg.slice());
        }
      }
      this.lastCarveKey = "";
    }

    _sweepLinear(zGrid, cutSeg, segIdx, ax, ay, az, bx, by, bz) {
      const zMinSeg = Math.min(az, bz);
      if (zMinSeg >= this.smax[2] - 1e-6) return;

      const r = this.rCut;
      const ix0 = Math.max(0, Math.floor((Math.min(ax, bx) - r - this.smin[0]) / this.dx));
      const ix1 = Math.min(this.nx - 1, Math.ceil((Math.max(ax, bx) + r - this.smin[0]) / this.dx));
      const iy0 = Math.max(0, Math.floor((Math.min(ay, by) - r - this.smin[1]) / this.dy));
      const iy1 = Math.min(this.ny - 1, Math.ceil((Math.max(ay, by) + r - this.smin[1]) / this.dy));
      if (ix0 > ix1 || iy0 > iy1) return;

      const dx = bx - ax;
      const dy = by - ay;
      const dz = bz - az;
      const lxy2 = dx * dx + dy * dy;
      const zFloor = this.smin[2];

      if (lxy2 <= 1e-10) {
        const zCut = Math.max(zFloor, zMinSeg);
        for (let ix = ix0; ix <= ix1; ix++) {
          const dpx = this.xs[ix] - ax;
          const dpx2 = dpx * dpx;
          const rowOff = ix * this.ny;
          for (let iy = iy0; iy <= iy1; iy++) {
            const dpy = this.ys[iy] - ay;
            if (dpx2 + dpy * dpy <= this.r2) {
              const idx = rowOff + iy;
              if (zCut < zGrid[idx] - 1e-6) {
                zGrid[idx] = zCut;
                cutSeg[idx] = segIdx;
              }
            }
          }
        }
        return;
      }

      const invLxy2 = 1.0 / lxy2;
      for (let ix = ix0; ix <= ix1; ix++) {
        const px = this.xs[ix] - ax;
        const rowOff = ix * this.ny;
        for (let iy = iy0; iy <= iy1; iy++) {
          const py = this.ys[iy] - ay;
          const uc = (px * dx + py * dy) * invLxy2;
          const perpX = px - uc * dx;
          const perpY = py - uc * dy;
          const dPerp2 = perpX * perpX + perpY * perpY;
          if (dPerp2 > this.r2) continue;

          const du = Math.sqrt(Math.max(0.0, this.r2 - dPerp2) * invLxy2);
          const uLo = Math.max(0.0, uc - du);
          const uHi = Math.min(1.0, uc + du);
          if (uLo > uHi) continue;

          const uBest = dz < 0.0 ? uHi : uLo;
          const zCut = Math.max(zFloor, az + uBest * dz);
          const idx = rowOff + iy;
          if (zCut < zGrid[idx] - 1e-6) {
            zGrid[idx] = zCut;
            cutSeg[idx] = segIdx;
          }
        }
      }
    }

    carveAndTriangulate(activeSegIdx, alpha, stockMode, paletteKey) {
      const clampedSeg = Math.max(-1, Math.min(activeSegIdx, this.segments.length - 1));
      const quantAlpha = Math.round(Math.max(0.0, Math.min(1.0, alpha)) * 40);
      const cacheKey = clampedSeg + ":" + quantAlpha + ":" + stockMode + ":" + paletteKey;
      if (cacheKey === this.lastCarveKey) {
        return null;
      }
      this.lastCarveKey = cacheKey;

      const completedCount = clampedSeg < 0 ? 0 : clampedSeg;
      const kfIndex = Math.min(
        this.keyframes.length - 1,
        Math.floor(completedCount / KEYFRAME_STRIDE)
      );
      this.zWork.set(this.keyframes[kfIndex]);
      this.lastCutSeg.set(this.keyframeCutSeg[kfIndex]);

      const replayStart = kfIndex * KEYFRAME_STRIDE;
      for (let k = replayStart; k < completedCount; k++) {
        const s = this.segments[k];
        if (s.kind !== "rapid") {
          this._sweepLinear(
            this.zWork,
            this.lastCutSeg,
            k,
            s.start[0],
            s.start[1],
            s.start[2],
            s.end[0],
            s.end[1],
            s.end[2]
          );
        }
      }

      if (clampedSeg >= 0 && alpha > 1e-4) {
        const cur = this.segments[clampedSeg];
        if (cur && cur.kind !== "rapid") {
          const ex = cur.start[0] + (cur.end[0] - cur.start[0]) * alpha;
          const ey = cur.start[1] + (cur.end[1] - cur.start[1]) * alpha;
          const ez = cur.start[2] + (cur.end[2] - cur.start[2]) * alpha;
          this._sweepLinear(
            this.zWork,
            this.lastCutSeg,
            clampedSeg,
            cur.start[0],
            cur.start[1],
            cur.start[2],
            ex,
            ey,
            ez
          );
        }
      }

      return this._triangulate(clampedSeg, stockMode, paletteKey);
    }

    _cellTopColor(idx, zVal, activeSegIdx, stockMode, pal) {
      if (stockMode === "live_deviation") {
        const diff = zVal - this.zTarget[idx];
        if (diff < -0.08) return [0.94, 0.24, 0.24]; // Gouge
        if (diff <= 0.16) return [0.19, 0.72, 0.30]; // In-tolerance
        const u = Math.min(1.0, (diff - 0.16) / 4.5);
        return [0.16 * (1 - u) + 0.14 * u, 0.56 * (1 - u) + 0.72 * u, 0.92];
      }
      if (stockMode === "live_glass") {
        return pal.skin;
      }
      // live_alloy: uncut raw skin vs machined 6061 alloy + warm active cut highlight
      if (zVal >= this.smax[2] - 1e-4) {
        return pal.skin;
      }
      const age = activeSegIdx - this.lastCutSeg[idx];
      if (age >= 0 && age <= 3) {
        const glow = 1.0 - age / 4.0;
        return [
          pal.alloy[0] * (1 - glow * 0.45) + 0.98 * glow * 0.45,
          pal.alloy[1] * (1 - glow * 0.45) + 0.68 * glow * 0.45,
          pal.alloy[2] * (1 - glow * 0.45) + 0.22 * glow * 0.45,
        ];
      }
      return pal.alloy;
    }

    _triangulate(activeSegIdx, stockMode, paletteKey) {
      const pal = STOCK_PALETTES[paletteKey] || STOCK_PALETTES.green;
      const pos = this.posScratch;
      const nrm = this.nrmScratch;
      const col = this.colScratch;
      let o = 0;

      function emitQuad(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, nx, ny, nz, rgb) {
        const r = rgb[0], g = rgb[1], b = rgb[2];
        pos[o] = ax; pos[o + 1] = ay; pos[o + 2] = az;
        pos[o + 3] = bx; pos[o + 4] = by; pos[o + 5] = bz;
        pos[o + 6] = cx; pos[o + 7] = cy; pos[o + 8] = cz;
        pos[o + 9] = ax; pos[o + 10] = ay; pos[o + 11] = az;
        pos[o + 12] = cx; pos[o + 13] = cy; pos[o + 14] = cz;
        pos[o + 15] = dx; pos[o + 16] = dy; pos[o + 17] = dz;

        for (let k = 0; k < 18; k += 3) {
          nrm[o + k] = nx; nrm[o + k + 1] = ny; nrm[o + k + 2] = nz;
          col[o + k] = r; col[o + k + 1] = g; col[o + k + 2] = b;
        }
        o += 18;
      }

      const [bx0, by0, bz0] = this.smin;
      const [bx1, by1] = this.smax;
      const dx = this.dx;
      const dy = this.dy;
      const nx = this.nx;
      const ny = this.ny;
      const z = this.zWork;
      const wallCutRgb = stockMode === "live_glass" ? pal.skin : pal.wall;

      // 1. Top heightfield quads (+Z)
      for (let ix = 0; ix < nx; ix++) {
        const x0 = bx0 + ix * dx;
        const x1 = x0 + dx;
        const rowOff = ix * ny;
        for (let iy = 0; iy < ny; iy++) {
          const idx = rowOff + iy;
          const zTop = z[idx];
          const y0 = by0 + iy * dy;
          const y1 = y0 + dy;
          const cTop = this._cellTopColor(idx, zTop, activeSegIdx, stockMode, pal);
          emitQuad(x0, y0, zTop, x1, y0, zTop, x1, y1, zTop, x0, y1, zTop, 0, 0, 1, cTop);
        }
      }

      // 2. Outer -X (ix=0) and +X (ix=nx-1) billet walls + internal X cliff walls
      for (let iy = 0; iy < ny; iy++) {
        const y0 = by0 + iy * dy;
        const y1 = y0 + dy;
        const zLeft = z[iy];
        if (zLeft > bz0 + 1e-5) {
          emitQuad(bx0, y0, bz0, bx0, y0, zLeft, bx0, y1, zLeft, bx0, y1, bz0, -1, 0, 0, pal.skin);
        }
        const zRight = z[(nx - 1) * ny + iy];
        if (zRight > bz0 + 1e-5) {
          emitQuad(bx1, y0, bz0, bx1, y1, bz0, bx1, y1, zRight, bx1, y0, zRight, 1, 0, 0, pal.skin);
        }
      }

      for (let ix = 0; ix < nx - 1; ix++) {
        const xEdge = bx0 + (ix + 1) * dx;
        const offA = ix * ny;
        const offB = (ix + 1) * ny;
        for (let iy = 0; iy < ny; iy++) {
          const zA = z[offA + iy];
          const zB = z[offB + iy];
          if (Math.abs(zA - zB) <= 1e-4) continue;
          const y0 = by0 + iy * dy;
          const y1 = y0 + dy;
          if (zA > zB) {
            const cStep =
              stockMode === "live_deviation"
                ? this._cellTopColor(offA + iy, zA, activeSegIdx, stockMode, pal)
                : wallCutRgb;
            emitQuad(xEdge, y0, zB, xEdge, y1, zB, xEdge, y1, zA, xEdge, y0, zA, 1, 0, 0, cStep);
          } else {
            const cStep =
              stockMode === "live_deviation"
                ? this._cellTopColor(offB + iy, zB, activeSegIdx, stockMode, pal)
                : wallCutRgb;
            emitQuad(xEdge, y0, zA, xEdge, y0, zB, xEdge, y1, zB, xEdge, y1, zA, -1, 0, 0, cStep);
          }
        }
      }

      // 3. Outer -Y (iy=0) and +Y (iy=ny-1) billet walls + internal Y cliff walls
      for (let ix = 0; ix < nx; ix++) {
        const x0 = bx0 + ix * dx;
        const x1 = x0 + dx;
        const rowOff = ix * ny;
        const zFront = z[rowOff];
        if (zFront > bz0 + 1e-5) {
          emitQuad(x0, by0, bz0, x1, by0, bz0, x1, by0, zFront, x0, by0, zFront, 0, -1, 0, pal.skin);
        }
        const zBack = z[rowOff + ny - 1];
        if (zBack > bz0 + 1e-5) {
          emitQuad(x0, by1, bz0, x0, by1, zBack, x1, by1, zBack, x1, by1, bz0, 0, 1, 0, pal.skin);
        }

        for (let iy = 0; iy < ny - 1; iy++) {
          const zA = z[rowOff + iy];
          const zB = z[rowOff + iy + 1];
          if (Math.abs(zA - zB) <= 1e-4) continue;
          const yEdge = by0 + (iy + 1) * dy;
          if (zA > zB) {
            const cStep =
              stockMode === "live_deviation"
                ? this._cellTopColor(rowOff + iy, zA, activeSegIdx, stockMode, pal)
                : wallCutRgb;
            emitQuad(x0, yEdge, zB, x0, yEdge, zA, x1, yEdge, zA, x1, yEdge, zB, 0, 1, 0, cStep);
          } else {
            const cStep =
              stockMode === "live_deviation"
                ? this._cellTopColor(rowOff + iy + 1, zB, activeSegIdx, stockMode, pal)
                : wallCutRgb;
            emitQuad(x0, yEdge, zA, x1, yEdge, zA, x1, yEdge, zB, x0, yEdge, zB, 0, -1, 0, cStep);
          }
        }
      }

      // 4. Single bottom billet face (-Z)
      emitQuad(bx0, by0, bz0, bx0, by1, bz0, bx1, by1, bz0, bx1, by0, bz0, 0, 0, -1, pal.skin);

      return {
        positions: pos.subarray(0, o),
        normals: nrm.subarray(0, o),
        colors: col.subarray(0, o),
        vertexCount: o / 3,
      };
    }
  }

  // --- Procedural 3D Cutter + Shank + CAT40 Holder Mesh (tip at origin) ---
  function buildToolAssemblyMesh(toolSpec) {
    const rCut = Number(toolSpec.cutter_radius_mm || 2.5);
    const fluteH = Number(toolSpec.flute_length_mm || 12.0);
    const rShank = Number(toolSpec.shank_radius_mm || rCut);
    const holderBot = Math.max(Number(toolSpec.holder_bottom_mm || 14.0), fluteH + 1.5);
    const rHolder = Number(toolSpec.holder_radius_mm || 14.0);
    const holderH = Math.min(Number(toolSpec.holder_height_mm || 24.0), 26.0);

    const sections = [
      { z0: 0.0, z1: fluteH, r0: rCut, r1: rCut, color: [0.96, 0.76, 0.24] },
      { z0: fluteH, z1: holderBot, r0: rShank, r1: rShank, color: [0.68, 0.73, 0.79] },
      { z0: holderBot, z1: holderBot + holderH * 0.35, r0: rShank * 1.6, r1: rHolder, color: [0.27, 0.31, 0.37] },
      { z0: holderBot + holderH * 0.35, z1: holderBot + holderH, r0: rHolder, r1: rHolder, color: [0.22, 0.25, 0.31] },
    ];

    const pos = [];
    const nrm = [];
    const col = [];
    const slices = 28;

    function pushTri(p0, p1, p2, n0, n1, n2, c) {
      pos.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
      nrm.push(n0[0], n0[1], n0[2], n1[0], n1[1], n1[2], n2[0], n2[1], n2[2]);
      col.push(c[0], c[1], c[2], c[0], c[1], c[2], c[0], c[1], c[2]);
    }

    for (const sec of sections) {
      for (let i = 0; i < slices; i++) {
        const a0 = (i / slices) * Math.PI * 2.0;
        const a1 = ((i + 1) / slices) * Math.PI * 2.0;
        const c0 = Math.cos(a0), s0 = Math.sin(a0);
        const c1 = Math.cos(a1), s1 = Math.sin(a1);

        const p00 = [sec.r0 * c0, sec.r0 * s0, sec.z0];
        const p10 = [sec.r0 * c1, sec.r0 * s1, sec.z0];
        const p11 = [sec.r1 * c1, sec.r1 * s1, sec.z1];
        const p01 = [sec.r1 * c0, sec.r1 * s0, sec.z1];
        const n0 = [c0, s0, 0.0];
        const n1 = [c1, s1, 0.0];

        pushTri(p00, p10, p11, n0, n1, n1, sec.color);
        pushTri(p00, p11, p01, n0, n1, n0, sec.color);
      }
    }
    for (let i = 0; i < slices; i++) {
      const a0 = (i / slices) * Math.PI * 2.0;
      const a1 = ((i + 1) / slices) * Math.PI * 2.0;
      pushTri(
        [0, 0, 0],
        [rCut * Math.cos(a1), rCut * Math.sin(a1), 0],
        [rCut * Math.cos(a0), rCut * Math.sin(a0), 0],
        [0, 0, -1],
        [0, 0, -1],
        [0, 0, -1],
        [0.96, 0.76, 0.24]
      );
    }

    return {
      positions: new Float32Array(pos),
      normals: new Float32Array(nrm),
      colors: new Float32Array(col),
      vertexCount: pos.length / 3,
    };
  }

  // --- Toolpath Segment Color Ramps ---
  function feedColorRgb(feedMmS, minF, maxF) {
    const span = Math.max(maxF - minF, 0.5);
    const t = Math.max(0.0, Math.min(1.0, (feedMmS - minF) / span));
    if (t < 0.45) {
      const u = t / 0.45;
      return [0.30 * (1 - u) + 0.08 * u, 0.43 * (1 - u) + 0.67 * u, 0.96 * (1 - u) + 0.75 * u];
    }
    if (t < 0.8) {
      const u = (t - 0.45) / 0.35;
      return [0.08 * (1 - u) + 0.25 * u, 0.67 * (1 - u) + 0.78 * u, 0.75 * (1 - u) + 0.34 * u];
    }
    const u = (t - 0.8) / 0.2;
    return [0.25 * (1 - u) + 0.98 * u, 0.78 * (1 - u) + 0.69 * u, 0.34 * (1 - u) + 0.03 * u];
  }

  function cutSegmentColorRgba(seg, mode, minF, maxF) {
    if (mode === "engaged_air") {
      if (!seg.engaged) {
        return [0.94, 0.40, 0.58, 0.92]; // Air-cut G1 (rose/magenta)
      }
      return [0.20, 0.86, 0.48, 0.98]; // Engaged material removal (bright emerald)
    }
    if (mode === "chipload") {
      if (!seg.engaged) return [0.52, 0.58, 0.68, 0.45];
      const h = Number(seg.hex_mm || 0.0);
      if (h < 0.020) {
        const u = Math.max(0.0, h / 0.020);
        return [0.23 * (1 - u) + 0.14 * u, 0.51 * (1 - u) + 0.77 * u, 0.96 * (1 - u) + 0.55 * u, 0.96];
      }
      if (h <= 0.045) {
        return [0.16, 0.82, 0.42, 0.98]; // Optimal 6061 chipload band
      }
      const u = Math.min(1.0, (h - 0.045) / 0.020);
      return [0.95, 0.65 * (1 - u) + 0.24 * u, 0.15, 0.98];
    }
    if (mode === "engagement") {
      if (!seg.engaged) return [0.52, 0.58, 0.68, 0.42];
      const aePct = Math.max(0.0, Math.min(100.0, Number(seg.ae_pct || 0.0))) / 100.0;
      if (aePct < 0.5) {
        const u = aePct / 0.5;
        return [0.13 * (1 - u) + 0.25 * u, 0.72 * (1 - u) + 0.76 * u, 0.82 * (1 - u) + 0.34 * u, 0.96];
      }
      const u = (aePct - 0.5) / 0.5;
      return [0.25 * (1 - u) + 0.95 * u, 0.76 * (1 - u) + 0.24 * u, 0.34 * (1 - u) + 0.22 * u, 0.98];
    }
    const rgb = feedColorRgb(Number(seg.feed_mm_s || minF), minF, maxF);
    return [rgb[0], rgb[1], rgb[2], 0.96];
  }

  // --- Build Static Grid + G54 Triad + Billet Wireframe Lines ---
  function buildSceneGuideLines(stockMin, stockMax) {
    const verts = [];
    const cols = [];
    function addSeg(a, b, rgba) {
      verts.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      cols.push(rgba[0], rgba[1], rgba[2], rgba[3], rgba[0], rgba[1], rgba[2], rgba[3]);
    }

    const zFloor = stockMin[2] - 0.05;
    const gridExtent = 50;
    for (let g = -gridExtent; g <= gridExtent; g += 10) {
      const c = g === 0 ? [0.26, 0.33, 0.43, 0.65] : [0.18, 0.23, 0.30, 0.42];
      addSeg([-gridExtent, g, zFloor], [gridExtent, g, zFloor], c);
      addSeg([g, -gridExtent, zFloor], [g, gridExtent, zFloor], c);
    }

    const axisLen = 10.0;
    addSeg([0, 0, 0.08], [axisLen, 0, 0.08], [1.0, 0.42, 0.42, 1.0]);
    addSeg([0, 0, 0.08], [0, axisLen, 0.08], [0.45, 0.92, 0.52, 1.0]);
    addSeg([0, 0, 0.08], [0, 0, axisLen], [0.42, 0.72, 1.0, 1.0]);

    const gridVertexCount = verts.length / 3;

    const [x0, y0, z0] = stockMin;
    const [x1, y1, z1] = stockMax;
    const bc = [0.68, 0.78, 0.90, 0.68];
    const corners = [
      [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    for (const [i, j] of edges) {
      addSeg(corners[i], corners[j], bc);
    }

    return {
      positions: new Float32Array(verts),
      colors: new Float32Array(cols),
      gridCount: gridVertexCount,
      billetOffset: gridVertexCount,
      billetCount: verts.length / 3 - gridVertexCount,
    };
  }

  // --- WebGL1 Renderer Setup (Explicit Attribute Locations 0, 1, 2) ---
  function initWebGL(canvas) {
    const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
    if (!gl) {
      throw new Error("WebGL is unavailable in this browser session.");
    }

    function compile(type, src) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error("WebGL shader compilation failed.");
      }
      return sh;
    }

    function link(vsSrc, fsSrc, bindings) {
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
      for (const [loc, name] of bindings) {
        gl.bindAttribLocation(prog, loc, name);
      }
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error("WebGL program link failed.");
      }
      return prog;
    }

    const meshProg = link(
      `
      attribute vec3 aPos;
      attribute vec3 aNormal;
      attribute vec3 aColor;
      uniform mat4 uMVP;
      uniform mat4 uModel;
      uniform vec3 uBaseColor;
      uniform float uUseVertColor;
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying vec3 vColor;
      void main() {
        vec4 wp = uModel * vec4(aPos, 1.0);
        vWorldPos = wp.xyz;
        vNormal = normalize(mat3(uModel[0].xyz, uModel[1].xyz, uModel[2].xyz) * aNormal);
        vColor = mix(uBaseColor, aColor, uUseVertColor);
        gl_Position = uMVP * wp;
      }
      `,
      `
      precision mediump float;
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying vec3 vColor;
      uniform vec3 uEye;
      uniform float uAlpha;
      void main() {
        vec3 N = normalize(vNormal);
        vec3 V = normalize(uEye - vWorldPos);
        if (dot(N, V) < 0.0) {
          N = -N;
        }
        vec3 L1 = normalize(vec3(0.52, -0.62, 0.85));
        vec3 L2 = normalize(vec3(-0.68, 0.48, 0.35));
        float diff1 = max(dot(N, L1), 0.0);
        float diff2 = max(dot(N, L2), 0.0);
        vec3 H = normalize(L1 + V);
        float spec = pow(max(dot(N, H), 0.0), 36.0) * 0.24;
        float amb = 0.32 + 0.08 * (0.5 * N.z + 0.5);
        vec3 lit = vColor * (amb + 0.56 * diff1 + 0.20 * diff2) + vec3(spec);
        gl_FragColor = vec4(lit, uAlpha);
      }
      `,
      [
        [0, "aPos"],
        [1, "aNormal"],
        [2, "aColor"],
      ]
    );

    const lineProg = link(
      `
      attribute vec3 aPos;
      attribute vec4 aColor;
      uniform mat4 uMVP;
      varying vec4 vColor;
      void main() {
        vColor = aColor;
        gl_Position = uMVP * vec4(aPos, 1.0);
      }
      `,
      `
      precision mediump float;
      varying vec4 vColor;
      void main() {
        gl_FragColor = vColor;
      }
      `,
      [
        [0, "aPos"],
        [1, "aColor"],
      ]
    );

    return {
      gl,
      mesh: {
        prog: meshProg,
        aPos: 0,
        aNormal: 1,
        aColor: 2,
        uMVP: gl.getUniformLocation(meshProg, "uMVP"),
        uModel: gl.getUniformLocation(meshProg, "uModel"),
        uBaseColor: gl.getUniformLocation(meshProg, "uBaseColor"),
        uUseVertColor: gl.getUniformLocation(meshProg, "uUseVertColor"),
        uEye: gl.getUniformLocation(meshProg, "uEye"),
        uAlpha: gl.getUniformLocation(meshProg, "uAlpha"),
      },
      line: {
        prog: lineProg,
        aPos: 0,
        aColor: 1,
        uMVP: gl.getUniformLocation(lineProg, "uMVP"),
      },
      buffers: {
        liveStockPos: gl.createBuffer(),
        liveStockNrm: gl.createBuffer(),
        liveStockCol: gl.createBuffer(),
        staticStockPos: gl.createBuffer(),
        staticStockNrm: gl.createBuffer(),
        staticStockCol: gl.createBuffer(),
        targetPos: gl.createBuffer(),
        targetNrm: gl.createBuffer(),
        toolPos: gl.createBuffer(),
        toolNrm: gl.createBuffer(),
        toolCol: gl.createBuffer(),
        guidePos: gl.createBuffer(),
        guideCol: gl.createBuffer(),
        cutPos: gl.createBuffer(),
        cutCol: gl.createBuffer(),
        rapidPos: gl.createBuffer(),
        rapidCol: gl.createBuffer(),
        activePos: gl.createBuffer(),
        activeCol: gl.createBuffer(),
      },
    };
  }

  // --- Application State ---
  const state = {
    manifest: null,
    currentCase: null,
    currentVariant: null,
    dexelSim: null,
    activeTab: "chipload",
    stlCache: new Map(),
    loadSeq: 0,

    stockMode: "live_alloy",
    stockPalette: "green",
    stockAlpha: 1.0,
    pathColorMode: "engaged_air",

    liveStockCount: 0,
    staticStockCount: 0,
    targetCount: 0,
    toolCount: 0,
    guideGridCount: 0,
    guideBilletOffset: 0,
    guideBilletCount: 0,
    cutTotalSegs: 0,
    rapidTotalSegs: 0,
    cutSegIndices: [],
    rapidSegIndices: [],

    playing: true,
    firstBoot: true,
    playTimeS: 0.0,
    totalDurationS: 0.0,
    speed: 2.0,
    lastFrameMs: 0.0,
    toolPos: [0.0, 0.0, 6.0],
    activeSegIndex: -1,
    activeSegAlpha: 0.0,

    cam: {
      yaw: -0.78,
      pitch: 0.62,
      distance: 108.0,
      target: [0.0, 0.0, -2.5],
      dragging: false,
      panMode: false,
      lastX: 0,
      lastY: 0,
    },
  };

  let renderer = null;

  function uploadBuffer(gl, buf, float32Data, dynamic) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, float32Data, dynamic ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
  }

  async function fetchJson(url) {
    const resp = await fetch(url, { headers: { Accept: "application/json" } });
    if (!resp.ok) {
      throw new Error("Failed to load JSON asset (" + resp.status + ").");
    }
    return resp.json();
  }

  async function fetchParsedStl(url, tintRgb) {
    if (state.stlCache.has(url)) {
      return state.stlCache.get(url);
    }
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error("Failed to load binary STL mesh (" + resp.status + ").");
    }
    const buf = await resp.arrayBuffer();
    const parsed = parseBinaryStl(buf, tintRgb);
    state.stlCache.set(url, parsed);
    return parsed;
  }

  // --- Update HUD Color Scale Legend ---
  function refreshHudLegend() {
    const titleEl = document.getElementById("legend-title");
    const barEl = document.getElementById("legend-bar");
    const minEl = document.getElementById("legend-min");
    const midEl = document.getElementById("legend-mid");
    const maxEl = document.getElementById("legend-max");

    if (state.stockMode === "live_deviation") {
      titleEl.textContent = "Stock-to-Model Deviation (09 §3)";
      barEl.className = "legend-gradient mode-deviation";
      minEl.textContent = "In-Tol (±0.15)";
      midEl.textContent = "Excess Stock";
      maxEl.textContent = "Gouge";
      return;
    }

    const mode = state.pathColorMode;
    if (mode === "engaged_air") {
      titleEl.textContent = "Toolpath: Engaged Cut vs. Air-Cutting";
      barEl.className = "legend-gradient mode-engaged";
      minEl.textContent = "Engaged G1";
      midEl.textContent = "Air-Cut G1";
      maxEl.textContent = "Rapid G0";
    } else if (mode === "chipload") {
      titleEl.textContent = "Chip Load h_ex (6061 Target 0.035 mm)";
      barEl.className = "legend-gradient mode-chipload";
      minEl.textContent = "<0.020 Rub";
      midEl.textContent = "0.035 Optimal";
      maxEl.textContent = ">0.045 Over";
    } else if (mode === "engagement") {
      titleEl.textContent = "Radial Cutter Engagement a_e (%D)";
      barEl.className = "legend-gradient mode-engagement";
      minEl.textContent = "10%D Light";
      midEl.textContent = "50%D Step";
      maxEl.textContent = "100%D Slot";
    } else {
      const plan = state.currentVariant ? state.currentVariant.plan : {};
      const minF = Number(plan.min_cut_feed_mm_s || 3.0);
      const maxF = Number(plan.max_cut_feed_mm_s || 20.0);
      titleEl.textContent = "Programmed Cutting Feedrate (G1)";
      barEl.className = "legend-gradient mode-feed";
      minEl.textContent = minF.toFixed(1) + " mm/s";
      midEl.textContent = ((minF + maxF) * 0.5).toFixed(1) + " mm/s";
      maxEl.textContent = maxF.toFixed(1) + " mm/s";
    }
  }

  // --- Upload Toolpath Line Buffers for Current Variant & Color Mode ---
  function rebuildToolpathColors() {
    if (!state.currentVariant || !renderer) return;
    const gl = renderer.gl;
    const plan = state.currentVariant.plan;
    const segs = Array.isArray(plan.segments) ? plan.segments : [];
    const minF = Number(plan.min_cut_feed_mm_s || 3.0);
    const maxF = Number(plan.max_cut_feed_mm_s || 20.0);

    const cutPos = [];
    const cutCol = [];
    const cutIdx = [];
    const rapPos = [];
    const rapCol = [];
    const rapIdx = [];

    for (let i = 0; i < segs.length; i++) {
      const s = segs[i];
      const st = s.start;
      const en = s.end;
      if (s.kind === "rapid") {
        rapPos.push(st[0], st[1], st[2], en[0], en[1], en[2]);
        rapCol.push(0.96, 0.64, 0.12, 0.68, 0.96, 0.64, 0.12, 0.68);
        rapIdx.push(i);
      } else {
        const rgba = cutSegmentColorRgba(s, state.pathColorMode, minF, maxF);
        cutPos.push(st[0], st[1], st[2], en[0], en[1], en[2]);
        cutCol.push(rgba[0], rgba[1], rgba[2], rgba[3], rgba[0], rgba[1], rgba[2], rgba[3]);
        cutIdx.push(i);
      }
    }

    uploadBuffer(gl, renderer.buffers.cutPos, new Float32Array(cutPos), false);
    uploadBuffer(gl, renderer.buffers.cutCol, new Float32Array(cutCol), false);
    uploadBuffer(gl, renderer.buffers.rapidPos, new Float32Array(rapPos), false);
    uploadBuffer(gl, renderer.buffers.rapidCol, new Float32Array(rapCol), false);

    state.cutTotalSegs = cutIdx.length;
    state.rapidTotalSegs = rapIdx.length;
    state.cutSegIndices = cutIdx;
    state.rapidSegIndices = rapIdx;
    refreshHudLegend();
  }

  // --- Update Live Dexel Stock Mesh Buffer ---
  function syncLiveDexelMesh() {
    if (!state.dexelSim || !renderer || state.stockMode === "static_stl") return;
    const mesh = state.dexelSim.carveAndTriangulate(
      state.activeSegIndex,
      state.activeSegAlpha,
      state.stockMode,
      state.stockPalette
    );
    if (!mesh) return;
    const gl = renderer.gl;
    uploadBuffer(gl, renderer.buffers.liveStockPos, mesh.positions, true);
    uploadBuffer(gl, renderer.buffers.liveStockNrm, mesh.normals, true);
    uploadBuffer(gl, renderer.buffers.liveStockCol, mesh.colors, true);
    state.liveStockCount = mesh.vertexCount;
  }

  // --- Render Stacked Engaged vs. Air-Cutting vs. Rapid SVG Bar ---
  function renderEngagementBar(playFrac) {
    const mount = document.getElementById("eng-svg-mount");
    const legendRow = document.getElementById("eng-legend-row");
    const volEl = document.getElementById("eng-vol-summary");
    if (!state.currentVariant) return;

    const eng = state.currentVariant.plan.engagement_summary || {};
    const ePct = Number(eng.engaged_pct || 0.0);
    const aPct = Number(eng.air_cut_pct || 0.0);
    const rPct = Math.max(0.0, 100.0 - ePct - aPct);

    const wEng = Math.round(ePct * 10);
    const wAir = Math.round(aPct * 10);
    const wRap = Math.max(0, 1000 - wEng - wAir);
    const playX = Math.max(0, Math.min(996, Math.round(playFrac * 1000) - 2));

    const svg = svgEl("svg", { viewBox: "0 0 1000 14", preserveAspectRatio: "none", class: "eng-bar-svg" }, [
      svgEl("rect", { x: "0", y: "0", width: String(wEng), height: "14", class: "svg-seg-engaged" }),
      svgEl("rect", { x: String(wEng), y: "0", width: String(wAir), height: "14", class: "svg-seg-air" }),
      svgEl("rect", { x: String(wEng + wAir), y: "0", width: String(wRap), height: "14", class: "svg-seg-rapid" }),
      svgEl("rect", { x: String(playX), y: "0", width: "4", height: "14", class: "svg-playhead" }),
    ]);
    mount.replaceChildren(svg);

    volEl.textContent = Number(eng.removed_volume_mm3 || 0).toFixed(1) + " mm³ total cut";
    legendRow.replaceChildren(
      el("span", { className: "dot-pill" }, [
        el("span", { className: "dot dot-engaged" }),
        "Engaged: " + Number(eng.engaged_time_s || 0).toFixed(2) + "s (" + ePct.toFixed(1) + "%)",
      ]),
      el("span", { className: "dot-pill" }, [
        el("span", { className: "dot dot-air" }),
        "Air G1: " + Number(eng.air_cut_time_s || 0).toFixed(2) + "s (" + aPct.toFixed(1) + "%)",
      ]),
      el("span", { className: "dot-pill" }, [
        el("span", { className: "dot dot-rapid" }),
        "Rapid G0: " + Number(eng.rapid_time_s || 0).toFixed(2) + "s (" + rPct.toFixed(1) + "%)",
      ])
    );
  }

  // --- Evaluate Spindle Pose, Live Dexel Carve, & Update DRO at state.playTimeS ---
  function updateScrubberAndDro() {
    if (!state.currentVariant) return;
    const plan = state.currentVariant.plan;
    const segs = plan.segments || [];
    const t = Math.max(0.0, Math.min(state.playTimeS, state.totalDurationS));
    const playFrac = state.totalDurationS > 1e-6 ? t / state.totalDurationS : 1.0;

    const slider = document.getElementById("time-slider");
    slider.value = String(Math.round(playFrac * 1000));
    document.getElementById("time-readout").textContent =
      t.toFixed(2) + " / " + state.totalDurationS.toFixed(2) + " s";

    let activeSegIndex = -1;
    let alpha = 0.0;
    let pos = plan.initial_pos ? plan.initial_pos.slice() : [0, 0, 6];
    let activeSeg = null;

    if (t <= 1e-7 || segs.length === 0) {
      activeSegIndex = -1;
      alpha = 0.0;
    } else {
      for (let i = 0; i < segs.length; i++) {
        const s = segs[i];
        if (t <= s.t1_s + 1e-7 || i === segs.length - 1) {
          activeSegIndex = i;
          activeSeg = s;
          const span = Math.max(s.t1_s - s.t0_s, 1e-9);
          alpha = Math.max(0.0, Math.min(1.0, (t - s.t0_s) / span));
          pos = [
            s.start[0] + (s.end[0] - s.start[0]) * alpha,
            s.start[1] + (s.end[1] - s.start[1]) * alpha,
            s.start[2] + (s.end[2] - s.start[2]) * alpha,
          ];
          break;
        }
      }
    }

    state.toolPos = pos;
    state.activeSegIndex = activeSegIndex;
    state.activeSegAlpha = alpha;

    syncLiveDexelMesh();
    renderEngagementBar(playFrac);

    const engSum = plan.engagement_summary || {};
    const totalVol = Number(engSum.removed_volume_mm3 || 0.0);
    let liveVol = 0.0;
    if (activeSegIndex >= 0 && activeSeg) {
      const prevCum = activeSegIndex > 0 ? Number(segs[activeSegIndex - 1].cum_removed_mm3 || 0.0) : 0.0;
      liveVol = prevCum + alpha * Number(activeSeg.removed_mm3 || 0.0);
    }

    const badge = document.getElementById("dro-state-badge");
    if (!activeSeg) {
      badge.textContent = "BILLET READY";
      badge.className = "state-badge state-engaged";
    } else if (activeSeg.kind === "rapid") {
      badge.textContent = "RAPID G0";
      badge.className = "state-badge state-rapid";
    } else if (activeSeg.engaged) {
      badge.textContent = "ENGAGED CUT";
      badge.className = "state-badge state-engaged";
    } else {
      badge.textContent = "AIR-CUT G1";
      badge.className = "state-badge state-air";
    }

    const feedS = activeSeg ? Number(activeSeg.feed_mm_s || 0.0) : 0.0;
    const feedMin = activeSeg ? Number(activeSeg.feed_mm_min || feedS * 60.0) : 0.0;
    const aeMm = activeSeg ? Number(activeSeg.ae_mm || 0.0) : 0.0;
    const aePct = activeSeg ? Number(activeSeg.ae_pct || 0.0) : 0.0;
    const phiDeg = activeSeg ? Number(activeSeg.phi_eng_deg || 0.0) : 0.0;
    const hexMm = activeSeg ? Number(activeSeg.hex_mm || 0.0) : 0.0;

    document.getElementById("dro-x").textContent = pos[0].toFixed(3) + " mm";
    document.getElementById("dro-y").textContent = pos[1].toFixed(3) + " mm";
    document.getElementById("dro-z").textContent = pos[2].toFixed(3) + " mm";
    document.getElementById("dro-class").textContent = activeSeg ? String(activeSeg.class || activeSeg.kind) : "Start (Uncut)";
    document.getElementById("dro-feed").textContent =
      feedS.toFixed(1) + " mm/s (" + Math.round(feedMin) + " mm/min)";
    document.getElementById("dro-ae").textContent =
      aeMm.toFixed(2) + " mm (" + aePct.toFixed(0) + "%D · " + phiDeg.toFixed(0) + "°)";
    document.getElementById("dro-hex").textContent = hexMm.toFixed(4) + " mm/tooth";
    document.getElementById("dro-vol").textContent =
      liveVol.toFixed(0) + " / " + totalVol.toFixed(0) + " mm³";
    document.getElementById("dro-seg").textContent =
      (activeSegIndex + 1) + " / " + segs.length;

    if (activeSegIndex >= 0 && activeSeg) {
      const gl = renderer.gl;
      uploadBuffer(
        gl,
        renderer.buffers.activePos,
        new Float32Array([
          activeSeg.start[0], activeSeg.start[1], activeSeg.start[2],
          activeSeg.end[0], activeSeg.end[1], activeSeg.end[2],
        ]),
        true
      );
      uploadBuffer(
        gl,
        renderer.buffers.activeCol,
        new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        true
      );
    }
  }

  // --- Select Variant Within Current Case ---
  async function selectVariant(variantId, preservePlayhead) {
    if (!state.currentCase) return;
    const variant =
      state.currentCase.variants.find((v) => v.id === variantId) || state.currentCase.variants[0];
    if (!variant) return;

    const mySeq = ++state.loadSeq;
    const prevFrac =
      preservePlayhead && state.totalDurationS > 1e-6
        ? Math.max(0.0, Math.min(1.0, state.playTimeS / state.totalDurationS))
        : null;

    state.currentVariant = variant;

    const btnWrap = document.getElementById("variant-buttons");
    for (const child of Array.from(btnWrap.children)) {
      if (child.getAttribute("data-vid") === variant.id) {
        child.classList.add("btn-active");
      } else {
        child.classList.remove("btn-active");
      }
    }

    document.getElementById("dl-rustcnc").setAttribute("href", state.currentCase.rustcnc_download_url);
    document.getElementById("dl-camotics").setAttribute("href", variant.camotics_project_url);
    const dlNc = document.getElementById("dl-nc");
    if (variant.haas_nc_url) {
      dlNc.setAttribute("href", variant.haas_nc_url);
      dlNc.textContent = "Haas NGC (.nc)";
    } else {
      dlNc.setAttribute("href", variant.camotics_nc_url);
      dlNc.textContent = "RS-274 (.nc)";
    }

    const plan = variant.plan;
    state.totalDurationS = Number(plan.total_duration_s || 0.0);
    if (state.dexelSim) {
      state.dexelSim.setSegments(plan.segments || []);
    }

    if (state.firstBoot) {
      state.firstBoot = false;
      state.playTimeS = 0.0;
      state.playing = true;
    } else if (prevFrac !== null) {
      state.playTimeS = prevFrac * state.totalDurationS;
    } else {
      state.playTimeS = state.totalDurationS;
      state.playing = false;
    }
    document.getElementById("btn-play").textContent = state.playing ? "Pause" : "Play";

    rebuildToolpathColors();
    updateScrubberAndDro();
    renderKpiStrip();
    renderInspectorTab();

    // Load certified static STL in background for 'static_stl' mode comparison
    try {
      clearError();
      const stockMesh = await fetchParsedStl(variant.stock_stl_url, [0.74, 0.80, 0.87]);
      if (mySeq !== state.loadSeq) return;
      const gl = renderer.gl;
      uploadBuffer(gl, renderer.buffers.staticStockPos, stockMesh.positions, false);
      uploadBuffer(gl, renderer.buffers.staticStockNrm, stockMesh.normals, false);
      uploadBuffer(gl, renderer.buffers.staticStockCol, stockMesh.colors, false);
      state.staticStockCount = stockMesh.vertexCount;
    } catch (err) {
      if (mySeq === state.loadSeq) {
        showError(err.message || "Unable to load certified static STL mesh.");
      }
    }
  }

  // --- Load Benchmark Case ---
  async function loadCase(caseJsonUrl, preferredVariantId) {
    const mySeq = ++state.loadSeq;
    try {
      clearError();
      const caseData = await fetchJson(caseJsonUrl);
      if (mySeq !== state.loadSeq) return;
      state.currentCase = caseData;
      state.dexelSim = new DexelStockSim(caseData, DEXEL_PITCH_MM);

      const gl = renderer.gl;
      const guides = buildSceneGuideLines(caseData.stock_min_mm, caseData.stock_max_mm);
      uploadBuffer(gl, renderer.buffers.guidePos, guides.positions, false);
      uploadBuffer(gl, renderer.buffers.guideCol, guides.colors, false);
      state.guideGridCount = guides.gridCount;
      state.guideBilletOffset = guides.billetOffset;
      state.guideBilletCount = guides.billetCount;

      const toolMesh = buildToolAssemblyMesh(caseData.tool || {});
      uploadBuffer(gl, renderer.buffers.toolPos, toolMesh.positions, false);
      uploadBuffer(gl, renderer.buffers.toolNrm, toolMesh.normals, false);
      uploadBuffer(gl, renderer.buffers.toolCol, toolMesh.colors, false);
      state.toolCount = toolMesh.vertexCount;

      const targetMesh = await fetchParsedStl(caseData.target_stl_url, [0.18, 0.85, 0.64]);
      if (mySeq !== state.loadSeq) return;
      uploadBuffer(gl, renderer.buffers.targetPos, targetMesh.positions, false);
      uploadBuffer(gl, renderer.buffers.targetNrm, targetMesh.normals, false);
      state.targetCount = targetMesh.vertexCount;

      const cx = 0.5 * (caseData.stock_min_mm[0] + caseData.stock_max_mm[0]);
      const cy = 0.5 * (caseData.stock_min_mm[1] + caseData.stock_max_mm[1]);
      const cz = 0.5 * (caseData.stock_min_mm[2] + caseData.stock_max_mm[2]);
      state.cam.target = [cx, cy, cz];

      const btnWrap = document.getElementById("variant-buttons");
      btnWrap.replaceChildren();
      for (const v of caseData.variants) {
        const b = el("button", {
          type: "button",
          className: "btn",
          "data-vid": v.id,
          text: v.label + " (" + v.metrics.cycle_time_s.toFixed(1) + "s)",
        });
        b.addEventListener("click", () => {
          selectVariant(v.id, false);
        });
        btnWrap.appendChild(b);
      }

      const defaultVid =
        caseData.variants.some((v) => v.id === "chipload_ste") ? "chipload_ste" : "ste_adam";
      const targetVid =
        preferredVariantId ||
        (state.currentVariant && state.currentVariant.id) ||
        defaultVid;
      await selectVariant(targetVid, false);
    } catch (err) {
      showError(err.message || "Unable to load benchmark part.");
    }
  }

  // --- Render 6-Card KPI Strip ---
  function renderKpiStrip() {
    const strip = document.getElementById("kpi-strip");
    strip.replaceChildren();
    if (!state.currentVariant) return;
    const m = state.currentVariant.metrics;
    const p = state.currentVariant.plan;
    const eng = p.engagement_summary || {};
    const gainSign =
      m.cycle_time_reduction_pct >= 0.05
        ? "-" + m.cycle_time_reduction_pct.toFixed(1) + "% vs @2"
        : "Baseline @2";
    const safetyOk = Boolean(m.posted_geometry_pass && m.posted_safety_pass);

    const cards = [
      {
        label: "Cycle Time",
        value: m.cycle_time_s.toFixed(2) + " s",
        sub: gainSign,
        cls: "kpi-cyan",
      },
      {
        label: "Engaged Cut (G1)",
        value: Number(eng.engaged_time_s || 0).toFixed(2) + " s",
        sub: Number(eng.engaged_pct || 0).toFixed(1) + "% · " + Number(eng.engaged_dist_mm || 0).toFixed(0) + " mm",
        cls: "kpi-good",
      },
      {
        label: "Air-Cutting (G1)",
        value: Number(eng.air_cut_time_s || 0).toFixed(2) + " s",
        sub: Number(eng.air_cut_pct || 0).toFixed(1) + "% · " + Number(eng.air_cut_dist_mm || 0).toFixed(0) + " mm",
        cls: "kpi-rose",
      },
      {
        label: "Mean Chip h_ex",
        value: Number(eng.mean_hex_mm || 0).toFixed(4) + " mm",
        sub: Number(eng.optimal_hex_pct || 0).toFixed(0) + "% in [0.020, 0.045]",
        cls: Number(eng.optimal_hex_pct || 0) >= 65 ? "kpi-good" : "kpi-amber",
      },
      {
        label: "Engagement a_e",
        value: Number(eng.mean_ae_pct || 0).toFixed(1) + "% D",
        sub: "Peak " + Number(eng.peak_ae_pct || 0).toFixed(0) + "%D · " + m.peak_mrr_mm3_s.toFixed(0) + " mm³/s",
        cls: "",
      },
      {
        label: "XY Reshape / Safety",
        value: "ΔXY " + Number(p.max_xy_shift_mm || 0).toFixed(2) + " mm",
        sub: safetyOk ? "0.125mm PASS (0 gouge)" : "CHECK",
        cls: safetyOk ? "kpi-good" : "kpi-amber",
      },
    ];

    for (const c of cards) {
      strip.appendChild(
        el("div", { className: "kpi-card" }, [
          el("div", { className: "kpi-label", text: c.label }),
          el("div", { className: ("kpi-value " + c.cls).trim(), text: c.value }),
          el("div", { className: "kpi-sub", text: c.sub }),
        ])
      );
    }
  }

  function buildTable(headers, rows, highlightPred) {
    const thead = el("thead", null, [
      el("tr", null, headers.map((h) => el("th", { text: h }))),
    ]);
    const tbody = el("tbody");
    for (const r of rows) {
      const tr = el("tr", { className: highlightPred && highlightPred(r) ? "row-highlight" : "" });
      for (const cell of r.cells) {
        if (cell && typeof cell === "object" && cell.nodeType === 1) {
          const td = el("td");
          td.appendChild(cell);
          tr.appendChild(td);
        } else {
          tr.appendChild(el("td", { text: String(cell) }));
        }
      }
      tbody.appendChild(tr);
    }
    return el("table", { className: "data-table" }, [thead, tbody]);
  }

  function renderInspectorTab() {
    const container = document.getElementById("tab-content");
    container.replaceChildren();
    if (!state.currentCase || !state.manifest) return;

    if (state.activeTab === "chipload") {
      container.appendChild(
        el("div", {
          className: "section-heading",
          text: state.currentCase.case_id + " — Cutter Engagement & Chipload Co-Optimization",
        })
      );
      const activeLabel = state.currentVariant ? state.currentVariant.label : "";
      const cTable = state.currentCase.chipload_table || [];
      if (cTable.length > 0) {
        const cRows = cTable.map((r) => ({
          raw: r,
          cells: [
            r.method,
            Number(r.cycle_time_s).toFixed(2) + "s",
            Number(r.engaged_time_s).toFixed(1) + " / " + Number(r.air_cut_time_s).toFixed(1) + "s",
            Number(r.mean_hex_mm).toFixed(4),
            el("span", {
              className: Number(r.optimal_hex_pct) >= 65.0 ? "badge-pass" : "badge-warn",
              text: Number(r.optimal_hex_pct).toFixed(0) + "%",
            }),
            Number(r.max_xy_shift_mm).toFixed(2) + " mm",
          ],
        }));
        container.appendChild(
          buildTable(
            ["Method", "Cycle", "Eng / Air", "h_ex (mm)", "Opt%", "Max ΔXY"],
            cRows,
            (r) => r.raw.method === activeLabel
          )
        );
      } else {
        // Fallback live comparison across playable variants
        const fbRows = state.currentCase.variants.map((v) => {
          const es = v.plan.engagement_summary || {};
          return {
            raw: v,
            cells: [
              v.label,
              v.metrics.cycle_time_s.toFixed(2) + "s",
              Number(es.engaged_time_s || 0).toFixed(1) + " / " + Number(es.air_cut_time_s || 0).toFixed(1) + "s",
              Number(es.mean_hex_mm || 0).toFixed(4),
              Number(es.optimal_hex_pct || 0).toFixed(0) + "%",
              Number(v.plan.max_xy_shift_mm || 0).toFixed(2) + " mm",
            ],
          };
        });
        container.appendChild(
          buildTable(
            ["Variant", "Cycle", "Eng / Air", "h_ex (mm)", "Opt%", "Max ΔXY"],
            fbRows,
            (r) => state.currentVariant && r.raw.id === state.currentVariant.id
          )
        );
      }

      const agg = state.manifest.chipload_summary || [];
      if (agg.length > 0) {
        container.appendChild(
          el("div", {
            className: "section-heading",
            text: "8-Test-Part Mean — DiffCAM-Chipload vs. Canned CAM & Equal-Budget RL",
          })
        );
        const aggRows = agg.map((r) => ({
          raw: r,
          cells: [
            r.method,
            Number(r.mean_cycle_time_s).toFixed(2) + "s (-" + Number(r.mean_gain_pct).toFixed(1) + "%)",
            Number(r.mean_air_cut_time_s).toFixed(1) + "s",
            Number(r.mean_hex_mm).toFixed(4),
            Number(r.mean_optimal_hex_pct).toFixed(1) + "%",
            Number(r.mean_max_xy_shift_mm).toFixed(2) + "mm",
          ],
        }));
        container.appendChild(
          buildTable(
            ["Method", "Mean Cycle", "Air G1", "Mean h_ex", "Opt%", "Max ΔXY"],
            aggRows,
            (r) => state.currentVariant && r.raw.method === state.currentVariant.label
          )
        );
      }
      return;
    }

    if (state.activeTab === "case") {
      container.appendChild(
        el("div", {
          className: "section-heading",
          text: state.currentCase.case_id + " (" + state.currentCase.family + ") — 3-Seed Benchmark Summary",
        })
      );
      const activeMethod = state.currentVariant ? state.currentVariant.method : "";
      const rows = (state.currentCase.benchmark_table || []).map((r) => ({
        raw: r,
        cells: [
          r.method,
          r.cycle_time_mean_s.toFixed(2) + " ± " + r.cycle_time_std_s.toFixed(2),
          (r.gain_mean_pct >= 0 ? "-" : "+") + Math.abs(r.gain_mean_pct).toFixed(1) + "%",
          r.peak_mrr_mean_mm3_s.toFixed(1),
          el("span", {
            className: r.feasible_rate_pct >= 99.9 ? "badge-pass" : "badge-warn",
            text: r.feasible_rate_pct.toFixed(0) + "%",
          }),
        ],
      }));
      container.appendChild(
        buildTable(
          ["Method", "Cycle (s)", "Time Δ", "Peak MRR", "Feas."],
          rows,
          (r) => r.raw.method === activeMethod
        )
      );

      container.appendChild(
        el("div", { className: "section-heading", text: "Posted G-Code & CAMotics Verification" })
      );
      const vRows = state.currentCase.variants.map((v) => ({
        raw: v,
        cells: [
          v.label,
          v.plan.num_segments,
          v.metrics.orig_gouge_mm3.toFixed(4) + " mm³",
          v.metrics.residual_mm3.toFixed(4) + " mm³",
          el("span", {
            className: v.metrics.posted_geometry_pass && v.metrics.posted_safety_pass ? "badge-pass" : "badge-warn",
            text: v.metrics.posted_geometry_pass && v.metrics.posted_safety_pass ? "PASS" : "FAIL",
          }),
        ],
      }));
      container.appendChild(
        buildTable(
          ["Simulated Variant", "Segs", "Gouge", "Residual", "Oracle"],
          vRows,
          (r) => state.currentVariant && r.raw.id === state.currentVariant.id
        )
      );
      return;
    }

    const summary = state.manifest.paper_summary || {};
    if (state.activeTab === "exp2") {
      container.appendChild(
        el("div", {
          className: "section-heading",
          text: "Table I — 16-Task Main Benchmark (64 Evaluations x 3 Seeds)",
        })
      );
      const exp2Main = summary.exp2_main || {};
      const mRows = Object.entries(exp2Main).map(([method, d]) => ({
        method,
        cells: [
          method,
          Number(d.mean_cycle_time_s).toFixed(2) + "s",
          "-" + Number(d.overall_gain_mean).toFixed(1) + " ± " + Number(d.overall_gain_std).toFixed(1) + "%",
          Number(d.mean_peak_mrr_mm3_s).toFixed(1),
          el("span", {
            className: Number(d.feasible_rate_pct) >= 99.0 ? "badge-pass" : "badge-warn",
            text: Number(d.feasible_rate_pct).toFixed(1) + "%",
          }),
        ],
      }));
      container.appendChild(
        buildTable(
          ["Method", "Mean Cycle", "Cycle Gain", "Peak MRR", "Feas."],
          mRows,
          (r) => state.currentVariant && r.method === state.currentVariant.method
        )
      );

      container.appendChild(
        el("div", { className: "section-heading", text: "Per-Strategy Family Cycle-Time Gain (%)" })
      );
      const famRows = Object.entries(exp2Main).map(([method, d]) => {
        const bf = d.by_family_gain_pct || {};
        const fmt = (k) => (bf[k] ? "-" + Number(bf[k][0]).toFixed(1) + "%" : "—");
        return {
          method,
          cells: [
            method,
            fmt("facing_open_boss"),
            fmt("pocket_island"),
            fmt("adaptive_channel"),
            fmt("contour_profile"),
          ],
        };
      });
      container.appendChild(
        buildTable(
          ["Method", "Facing", "Pocket", "Adaptive", "Contour"],
          famRows,
          (r) => state.currentVariant && r.method === state.currentVariant.method
        )
      );
      return;
    }

    container.appendChild(
      el("div", { className: "section-heading", text: "Table II — Residual Policy Transfer (Held-Out Test)" })
    );
    const exp3 = summary.exp3_transfer || {};
    const tRows = Object.entries(exp3).map(([reg, d]) => ({
      cells: [
        reg,
        "-" + Number(d.gain_mean).toFixed(1) + " ± " + Number(d.gain_std).toFixed(1) + "%",
        Number(d.peak_mrr_mean).toFixed(1),
        el("span", {
          className: Number(d.feasible_pct) >= 99.0 ? "badge-pass" : "badge-warn",
          text: Number(d.feasible_pct).toFixed(1) + "%",
        }),
      ],
    }));
    container.appendChild(buildTable(["Transfer Regime", "Cycle Gain", "Peak MRR", "Feas."], tRows));

    container.appendChild(
      el("div", { className: "section-heading", text: "Table III — 5-Arm Formulation Ablations" })
    );
    const abl = summary.exp2_ablations || {};
    const aRows = Object.entries(abl).map(([arm, d]) => ({
      cells: [
        arm,
        "-" + Number(d.gain_mean).toFixed(1) + "%",
        Number(d.peak_mrr_mean).toFixed(1),
        Number(d.mrr_overload_pct_mean).toFixed(1) + "%",
      ],
    }));
    container.appendChild(buildTable(["Ablation Arm", "Gain", "Peak MRR", "MRR Overload"], aRows));

    const mc = (summary.exp4_verification || {}).multi_controller_summary || {};
    if (mc.by_dialect) {
      container.appendChild(
        el("div", { className: "section-heading", text: "6-Controller Clean-Room Post Audit (Exp4)" })
      );
      const dRows = Object.entries(mc.by_dialect).map(([dial, d]) => ({
        cells: [
          dial.toUpperCase(),
          d.pass_count + " / " + d.count,
          Number(d.mean_cycle_time_s).toFixed(2) + " s",
          el("span", {
            className: d.pass_count === d.count ? "badge-pass" : "badge-warn",
            text: d.pass_count === d.count ? "100% PASS" : "CHECK",
          }),
        ],
      }));
      container.appendChild(buildTable(["Controller", "Geom+Safe", "Mean Cycle", "Status"], dRows));
    }
  }

  // --- Main WebGL Render Loop (Explicit Attribute Hygiene) ---
  function drawMeshBuffer(posBuf, nrmBuf, colBuf, count, mvp, modelMat, eye, baseColor, useVertCol, alpha) {
    if (count <= 0) return;
    const gl = renderer.gl;
    const m = renderer.mesh;
    gl.useProgram(m.prog);
    gl.uniformMatrix4fv(m.uMVP, false, mvp);
    gl.uniformMatrix4fv(m.uModel, false, modelMat);
    gl.uniform3fv(m.uBaseColor, baseColor);
    gl.uniform1f(m.uUseVertColor, useVertCol ? 1.0 : 0.0);
    gl.uniform3fv(m.uEye, eye);
    gl.uniform1f(m.uAlpha, alpha);

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    if (useVertCol && colBuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
    } else {
      gl.disableVertexAttribArray(2);
      gl.vertexAttrib3f(2, baseColor[0], baseColor[1], baseColor[2]);
    }

    gl.drawArrays(gl.TRIANGLES, 0, count);
  }

  function drawLineBuffer(posBuf, colBuf, firstVertex, vertexCount, mvp) {
    if (vertexCount <= 0) return;
    const gl = renderer.gl;
    const l = renderer.line;
    gl.useProgram(l.prog);
    gl.uniformMatrix4fv(l.uMVP, false, mvp);

    gl.disableVertexAttribArray(2);

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.LINES, firstVertex, vertexCount);
  }

  function countVisibleSubset(segIndices, maxSegIndex) {
    if (maxSegIndex < 0) return 0;
    let lo = 0;
    let hi = segIndices.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (segIndices[mid] <= maxSegIndex) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function renderFrame(nowMs) {
    requestAnimationFrame(renderFrame);
    if (!renderer) return;

    if (state.playing && state.totalDurationS > 1e-6) {
      const dt = Math.min((nowMs - state.lastFrameMs) * 0.001, 0.1);
      state.playTimeS += dt * state.speed;
      if (state.playTimeS >= state.totalDurationS) {
        state.playTimeS = state.totalDurationS;
        state.playing = false;
        document.getElementById("btn-play").textContent = "Play";
      }
      updateScrubberAndDro();
    }
    state.lastFrameMs = nowMs;

    const canvas = renderer.gl.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    const gl = renderer.gl;
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.055, 0.075, 0.105, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    const cp = Math.cos(state.cam.pitch);
    const sp = Math.sin(state.cam.pitch);
    const cy = Math.cos(state.cam.yaw);
    const sy = Math.sin(state.cam.yaw);
    const eye = [
      state.cam.target[0] + state.cam.distance * cp * cy,
      state.cam.target[1] + state.cam.distance * cp * sy,
      state.cam.target[2] + state.cam.distance * sp,
    ];

    const proj = mat4Perspective((40.0 * Math.PI) / 180.0, w / h, 0.5, 900.0);
    const view = mat4LookAt(eye, state.cam.target, [0.0, 0.0, 1.0]);
    const vp = mat4Multiply(proj, view);
    const ident = mat4Identity();

    const showStock = document.getElementById("chk-stock").checked;
    const showTarget = document.getElementById("chk-target").checked || state.stockMode === "live_glass";
    const showBillet = document.getElementById("chk-billet").checked;
    const showCuts = document.getElementById("chk-cuts").checked;
    const showRapids = document.getElementById("chk-rapids").checked;
    const showTool = document.getElementById("chk-tool").checked;
    const clipToTime = document.getElementById("chk-clip").checked;

    const effectiveStockAlpha =
      state.stockMode === "live_glass" ? Math.min(state.stockAlpha, 0.52) : state.stockAlpha;
    const stockOpaque = effectiveStockAlpha >= 0.995;

    const useStaticStl = state.stockMode === "static_stl";
    const sPos = useStaticStl ? renderer.buffers.staticStockPos : renderer.buffers.liveStockPos;
    const sNrm = useStaticStl ? renderer.buffers.staticStockNrm : renderer.buffers.liveStockNrm;
    const sCol = useStaticStl ? renderer.buffers.staticStockCol : renderer.buffers.liveStockCol;
    const sCnt = useStaticStl ? state.staticStockCount : state.liveStockCount;

    gl.disable(gl.BLEND);
    gl.depthMask(true);

    // 1a. In Glass mode, draw solid Target CAD part underneath translucent billet
    if (state.stockMode === "live_glass" && state.targetCount > 0) {
      drawMeshBuffer(
        renderer.buffers.targetPos,
        renderer.buffers.targetNrm,
        null,
        state.targetCount,
        vp,
        ident,
        eye,
        [0.72, 0.80, 0.88],
        false,
        1.0
      );
    }

    // 1b. Opaque Stock Mesh (default)
    if (showStock && stockOpaque && sCnt > 0) {
      drawMeshBuffer(sPos, sNrm, sCol, sCnt, vp, ident, eye, [0.72, 0.78, 0.85], true, 1.0);
    }

    // 2. Opaque Animated 3D Endmill + Shank + CAT40 Holder
    if (showTool && state.toolCount > 0) {
      const toolModel = mat4Translation(state.toolPos[0], state.toolPos[1], state.toolPos[2]);
      drawMeshBuffer(
        renderer.buffers.toolPos,
        renderer.buffers.toolNrm,
        renderer.buffers.toolCol,
        state.toolCount,
        vp,
        toolModel,
        eye,
        [1.0, 1.0, 1.0],
        true,
        1.0
      );
    }

    // 3. Lines (Grid + G54 + Billet + Toolpaths)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    if (state.guideGridCount > 0) {
      drawLineBuffer(renderer.buffers.guidePos, renderer.buffers.guideCol, 0, state.guideGridCount, vp);
    }
    if (showBillet && state.guideBilletCount > 0) {
      drawLineBuffer(
        renderer.buffers.guidePos,
        renderer.buffers.guideCol,
        state.guideBilletOffset,
        state.guideBilletCount,
        vp
      );
    }

    const maxIdx = clipToTime ? state.activeSegIndex : 1000000000;
    if (showRapids && state.rapidTotalSegs > 0) {
      const nRap = clipToTime ? countVisibleSubset(state.rapidSegIndices, maxIdx) : state.rapidTotalSegs;
      drawLineBuffer(renderer.buffers.rapidPos, renderer.buffers.rapidCol, 0, nRap * 2, vp);
    }
    if (showCuts && state.cutTotalSegs > 0) {
      const nCut = clipToTime ? countVisibleSubset(state.cutSegIndices, maxIdx) : state.cutTotalSegs;
      drawLineBuffer(renderer.buffers.cutPos, renderer.buffers.cutCol, 0, nCut * 2, vp);
    }
    if (state.activeSegIndex >= 0 && (showCuts || showRapids)) {
      drawLineBuffer(renderer.buffers.activePos, renderer.buffers.activeCol, 0, 2, vp);
    }

    // 4. Translucent Stock Glass OR Translucent Target CAD Overlay
    gl.depthMask(false);
    if (showStock && !stockOpaque && sCnt > 0) {
      drawMeshBuffer(
        sPos,
        sNrm,
        sCol,
        sCnt,
        vp,
        ident,
        eye,
        [0.22, 0.56, 0.26],
        true,
        effectiveStockAlpha
      );
    }
    if (showTarget && state.stockMode !== "live_glass" && state.targetCount > 0) {
      drawMeshBuffer(
        renderer.buffers.targetPos,
        renderer.buffers.targetNrm,
        null,
        state.targetCount,
        vp,
        ident,
        eye,
        [0.18, 0.85, 0.64],
        false,
        0.34
      );
    }
    gl.depthMask(true);
  }

  // --- Wire Mouse / Transport / Appearance Controls ---
  function stepMove(delta) {
    if (!state.currentVariant) return;
    const segs = state.currentVariant.plan.segments || [];
    if (segs.length === 0) return;
    state.playing = false;
    document.getElementById("btn-play").textContent = "Play";

    let nextIdx = state.activeSegIndex + delta;
    if (nextIdx < 0) {
      state.playTimeS = 0.0;
    } else if (nextIdx >= segs.length) {
      state.playTimeS = state.totalDurationS;
    } else {
      state.playTimeS = Number(segs[nextIdx].t1_s || 0.0);
    }
    updateScrubberAndDro();
  }

  function bindEvents(canvas) {
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    canvas.addEventListener("mousedown", (e) => {
      state.cam.dragging = true;
      state.cam.panMode = e.button === 2 || e.button === 1 || e.shiftKey;
      state.cam.lastX = e.clientX;
      state.cam.lastY = e.clientY;
    });

    window.addEventListener("mouseup", () => {
      state.cam.dragging = false;
    });

    window.addEventListener("mousemove", (e) => {
      if (!state.cam.dragging) return;
      const dx = e.clientX - state.cam.lastX;
      const dy = e.clientY - state.cam.lastY;
      state.cam.lastX = e.clientX;
      state.cam.lastY = e.clientY;

      if (state.cam.panMode) {
        const scale = state.cam.distance * 0.0016;
        const rightX = -Math.sin(state.cam.yaw);
        const rightY = Math.cos(state.cam.yaw);
        state.cam.target[0] -= dx * scale * rightX;
        state.cam.target[1] -= dx * scale * rightY;
        state.cam.target[2] += dy * scale;
      } else {
        state.cam.yaw -= dx * 0.0075;
        state.cam.pitch = Math.max(-1.48, Math.min(1.52, state.cam.pitch + dy * 0.0075));
      }
    });

    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const factor = Math.exp(e.deltaY * 0.0012);
        state.cam.distance = Math.max(18.0, Math.min(420.0, state.cam.distance * factor));
      },
      { passive: false }
    );

    document.getElementById("cam-iso").addEventListener("click", () => {
      state.cam.yaw = -0.78;
      state.cam.pitch = 0.62;
      state.cam.distance = 108.0;
    });
    document.getElementById("cam-top").addEventListener("click", () => {
      state.cam.yaw = -Math.PI * 0.5;
      state.cam.pitch = 1.52;
    });
    document.getElementById("cam-front").addEventListener("click", () => {
      state.cam.yaw = -Math.PI * 0.5;
      state.cam.pitch = 0.06;
    });
    document.getElementById("cam-fit").addEventListener("click", () => {
      state.cam.yaw = -0.78;
      state.cam.pitch = 0.62;
      state.cam.distance = 108.0;
      if (state.currentCase) {
        state.cam.target = [
          0.5 * (state.currentCase.stock_min_mm[0] + state.currentCase.stock_max_mm[0]),
          0.5 * (state.currentCase.stock_min_mm[1] + state.currentCase.stock_max_mm[1]),
          0.5 * (state.currentCase.stock_min_mm[2] + state.currentCase.stock_max_mm[2]),
        ];
      }
    });

    document.getElementById("stock-mode-select").addEventListener("change", (e) => {
      state.stockMode = String(e.target.value || "live_alloy");
      const alphaSlider = document.getElementById("stock-opacity");
      if (state.stockMode === "live_glass" && state.stockAlpha > 0.85) {
        state.stockAlpha = 0.48;
        alphaSlider.value = "50";
      } else if (state.stockMode !== "live_glass" && state.stockAlpha < 0.85) {
        state.stockAlpha = 1.0;
        alphaSlider.value = "100";
      }
      if (state.dexelSim) state.dexelSim.lastCarveKey = "";
      syncLiveDexelMesh();
      refreshHudLegend();
    });

    document.getElementById("stock-color-select").addEventListener("change", (e) => {
      state.stockPalette = String(e.target.value || "green");
      if (state.dexelSim) state.dexelSim.lastCarveKey = "";
      syncLiveDexelMesh();
    });

    document.getElementById("stock-opacity").addEventListener("input", (e) => {
      const pct = Number(e.target.value || 100);
      state.stockAlpha = Math.max(0.15, Math.min(1.0, pct / 100.0));
    });

    document.getElementById("path-color-select").addEventListener("change", (e) => {
      state.pathColorMode = String(e.target.value || "engaged_air");
      rebuildToolpathColors();
    });

    document.getElementById("btn-play").addEventListener("click", () => {
      if (state.totalDurationS <= 1e-6) return;
      if (!state.playing && state.playTimeS >= state.totalDurationS - 1e-4) {
        state.playTimeS = 0.0;
      }
      state.playing = !state.playing;
      state.lastFrameMs = performance.now();
      document.getElementById("btn-play").textContent = state.playing ? "Pause" : "Play";
      updateScrubberAndDro();
    });

    document.getElementById("btn-reset").addEventListener("click", () => {
      state.playing = false;
      state.playTimeS = 0.0;
      document.getElementById("btn-play").textContent = "Play";
      updateScrubberAndDro();
    });

    document.getElementById("btn-prev").addEventListener("click", () => {
      stepMove(-1);
    });

    document.getElementById("btn-next").addEventListener("click", () => {
      stepMove(1);
    });

    document.getElementById("btn-end").addEventListener("click", () => {
      state.playing = false;
      state.playTimeS = state.totalDurationS;
      document.getElementById("btn-play").textContent = "Play";
      updateScrubberAndDro();
    });

    document.getElementById("speed-select").addEventListener("change", (e) => {
      const val = Number(e.target.value);
      if (Number.isFinite(val) && val > 0) {
        state.speed = val;
      }
    });

    document.getElementById("time-slider").addEventListener("input", (e) => {
      state.playing = false;
      document.getElementById("btn-play").textContent = "Play";
      const frac = Number(e.target.value) / 1000.0;
      state.playTimeS = Math.max(0.0, Math.min(1.0, frac)) * state.totalDurationS;
      updateScrubberAndDro();
    });

    document.getElementById("case-select").addEventListener("change", (e) => {
      loadCase(e.target.value);
    });

    for (const btn of Array.from(document.querySelectorAll(".tab-btn"))) {
      btn.addEventListener("click", () => {
        for (const other of Array.from(document.querySelectorAll(".tab-btn"))) {
          other.classList.remove("active");
          other.setAttribute("aria-selected", "false");
        }
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        state.activeTab = btn.getAttribute("data-tab") || "chipload";
        renderInspectorTab();
      });
    }
  }

  // --- Bootstrap ---
  async function boot() {
    try {
      const canvas = document.getElementById("gl-canvas");
      renderer = initWebGL(canvas);
      bindEvents(canvas);

      const manifest = await fetchJson("/data/manifest.json");
      state.manifest = manifest;

      const select = document.getElementById("case-select");
      select.replaceChildren();
      for (const c of manifest.cases || []) {
        const gainVal = Number(
          c.chipload_gain_pct !== undefined ? c.chipload_gain_pct : c.ste_gain_pct || 0
        );
        select.appendChild(
          el("option", {
            value: c.case_json_url,
            text:
              c.case_id +
              " (" +
              c.family +
              " · -" +
              gainVal.toFixed(1) +
              "%)",
          })
        );
      }

      if (manifest.cases && manifest.cases.length > 0) {
        const pocketDefault =
          manifest.cases.find((c) => c.case_id === "pocket-test-00") || manifest.cases[0];
        select.value = pocketDefault.case_json_url;
        await loadCase(pocketDefault.case_json_url, "chipload_ste");
      }
      state.lastFrameMs = performance.now();
      requestAnimationFrame(renderFrame);
    } catch (err) {
      showError(err.message || "Failed to initialize DiffCAM Manufacture Simulator.");
    }
  }

  window.addEventListener("DOMContentLoaded", boot);
})();
