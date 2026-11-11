// SPDX-License-Identifier: MIT
// Clean-room WebGL1 3D Toolpath, Resulting Stock Mesh & Benchmark Results Explorer.
// TODO(security): Scoped as a read-only local 127.0.0.1 single-user viewer with zero cookies,
// zero storage APIs, and strict createElement/textContent DOM construction (no innerHTML/eval).

(function () {
  "use strict";

  const MAX_STL_TRIANGLES = 2000000;

  // --- Safe DOM Helper (strictly createElement + textContent + setAttribute) ---
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

  // --- Minimal 4x4 Column-Major Matrix Utilities (Z-up CAD/CAM Coordinate Frame) ---
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
  function parseBinaryStl(buffer) {
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
    }
    return { positions, normals, vertexCount: triCount * 3 };
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
    // Bottom disk at z=0
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

  // --- Feedrate Color Ramp (3..20 mm/s -> Cobalt -> Cyan -> Emerald -> Amber) ---
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

  // --- Build Static Grid + G54 Triad + Billet Wireframe Lines ---
  function buildSceneGuideLines(stockMin, stockMax) {
    const verts = [];
    const cols = [];
    function addSeg(a, b, rgba) {
      verts.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      cols.push(rgba[0], rgba[1], rgba[2], rgba[3], rgba[0], rgba[1], rgba[2], rgba[3]);
    }

    // Floor grid at stockMin[2]
    const zFloor = stockMin[2] - 0.05;
    const gridExtent = 50;
    for (let g = -gridExtent; g <= gridExtent; g += 10) {
      const c = g === 0 ? [0.26, 0.33, 0.43, 0.65] : [0.18, 0.23, 0.30, 0.42];
      addSeg([-gridExtent, g, zFloor], [gridExtent, g, zFloor], c);
      addSeg([g, -gridExtent, zFloor], [g, gridExtent, zFloor], c);
    }

    // G54 Coordinate Triad at (0, 0, 0)
    const axisLen = 10.0;
    addSeg([0, 0, 0.05], [axisLen, 0, 0.05], [1.0, 0.42, 0.42, 1.0]);
    addSeg([0, 0, 0.05], [0, axisLen, 0.05], [0.45, 0.92, 0.52, 1.0]);
    addSeg([0, 0, 0.05], [0, 0, axisLen], [0.42, 0.72, 1.0, 1.0]);

    const gridVertexCount = verts.length / 3;

    // Billet Box Wireframe
    const [x0, y0, z0] = stockMin;
    const [x1, y1, z1] = stockMax;
    const bc = [0.64, 0.72, 0.84, 0.62];
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

  // --- WebGL1 Renderer Setup ---
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

    function link(vsSrc, fsSrc) {
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
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
        float spec = pow(max(dot(N, H), 0.0), 34.0) * 0.26;
        float amb = 0.30 + 0.08 * (0.5 * N.z + 0.5);
        vec3 lit = vColor * (amb + 0.56 * diff1 + 0.22 * diff2) + vec3(spec);
        gl_FragColor = vec4(lit, uAlpha);
      }
      `
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
      `
    );

    return {
      gl,
      mesh: {
        prog: meshProg,
        aPos: gl.getAttribLocation(meshProg, "aPos"),
        aNormal: gl.getAttribLocation(meshProg, "aNormal"),
        aColor: gl.getAttribLocation(meshProg, "aColor"),
        uMVP: gl.getUniformLocation(meshProg, "uMVP"),
        uModel: gl.getUniformLocation(meshProg, "uModel"),
        uBaseColor: gl.getUniformLocation(meshProg, "uBaseColor"),
        uUseVertColor: gl.getUniformLocation(meshProg, "uUseVertColor"),
        uEye: gl.getUniformLocation(meshProg, "uEye"),
        uAlpha: gl.getUniformLocation(meshProg, "uAlpha"),
      },
      line: {
        prog: lineProg,
        aPos: gl.getAttribLocation(lineProg, "aPos"),
        aColor: gl.getAttribLocation(lineProg, "aColor"),
        uMVP: gl.getUniformLocation(lineProg, "uMVP"),
      },
      buffers: {
        stockPos: gl.createBuffer(),
        stockNrm: gl.createBuffer(),
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
    activeTab: "case",
    stlCache: new Map(),

    stockCount: 0,
    targetCount: 0,
    toolCount: 0,
    guideGridCount: 0,
    guideBilletOffset: 0,
    guideBilletCount: 0,
    cutTotalSegs: 0,
    rapidTotalSegs: 0,
    cutSegIndices: [],
    rapidSegIndices: [],

    playing: false,
    playTimeS: 0.0,
    totalDurationS: 0.0,
    speed: 1.0,
    lastFrameMs: 0.0,
    toolPos: [0.0, 0.0, 6.0],

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

  function uploadStaticBuffer(gl, buf, float32Data) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, float32Data, gl.STATIC_DRAW);
  }

  async function fetchJson(url) {
    const resp = await fetch(url, { headers: { Accept: "application/json" } });
    if (!resp.ok) {
      throw new Error("Failed to load JSON asset (" + resp.status + ").");
    }
    return resp.json();
  }

  async function fetchParsedStl(url) {
    if (state.stlCache.has(url)) {
      return state.stlCache.get(url);
    }
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error("Failed to load binary STL mesh (" + resp.status + ").");
    }
    const buf = await resp.arrayBuffer();
    const parsed = parseBinaryStl(buf);
    state.stlCache.set(url, parsed);
    return parsed;
  }

  // --- Upload Toolpath Buffers for Current Variant ---
  function uploadVariantToolpath(variant) {
    const gl = renderer.gl;
    const plan = variant.plan;
    const segs = Array.isArray(plan.segments) ? plan.segments : [];
    const minF = Number(plan.min_cut_feed_mm_s || 3.0);
    const maxF = Number(plan.max_cut_feed_mm_s || 20.0);

    document.getElementById("legend-min").textContent = minF.toFixed(1) + " mm/s";
    document.getElementById("legend-mid").textContent = ((minF + maxF) * 0.5).toFixed(1) + " mm/s";
    document.getElementById("legend-max").textContent = maxF.toFixed(1) + " mm/s";

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
        rapCol.push(0.95, 0.65, 0.18, 0.72, 0.95, 0.65, 0.18, 0.72);
        rapIdx.push(i);
      } else {
        const rgb = feedColorRgb(Number(s.feed_mm_s || minF), minF, maxF);
        cutPos.push(st[0], st[1], st[2], en[0], en[1], en[2]);
        cutCol.push(rgb[0], rgb[1], rgb[2], 0.96, rgb[0], rgb[1], rgb[2], 0.96);
        cutIdx.push(i);
      }
    }

    uploadStaticBuffer(gl, renderer.buffers.cutPos, new Float32Array(cutPos));
    uploadStaticBuffer(gl, renderer.buffers.cutCol, new Float32Array(cutCol));
    uploadStaticBuffer(gl, renderer.buffers.rapidPos, new Float32Array(rapPos));
    uploadStaticBuffer(gl, renderer.buffers.rapidCol, new Float32Array(rapCol));

    state.cutTotalSegs = cutIdx.length;
    state.rapidTotalSegs = rapIdx.length;
    state.cutSegIndices = cutIdx;
    state.rapidSegIndices = rapIdx;
    state.totalDurationS = Number(plan.total_duration_s || 0.0);
    state.playTimeS = state.totalDurationS;
    state.playing = false;
    document.getElementById("btn-play").textContent = "Play";
    updateScrubberAndDro();
  }

  // --- Evaluate Spindle Pose & Update DRO at state.playTimeS ---
  function updateScrubberAndDro() {
    if (!state.currentVariant) return;
    const plan = state.currentVariant.plan;
    const segs = plan.segments || [];
    const t = Math.max(0.0, Math.min(state.playTimeS, state.totalDurationS));

    const slider = document.getElementById("time-slider");
    const frac = state.totalDurationS > 1e-6 ? Math.round((t / state.totalDurationS) * 1000) : 1000;
    slider.value = String(frac);
    document.getElementById("time-readout").textContent =
      t.toFixed(2) + " / " + state.totalDurationS.toFixed(2) + " s";

    let activeSegIndex = segs.length > 0 ? segs.length - 1 : -1;
    let pos = plan.initial_pos ? plan.initial_pos.slice() : [0, 0, 6];
    let moveClass = "Idle";
    let feedS = 0.0;
    let feedMin = 0.0;

    for (let i = 0; i < segs.length; i++) {
      const s = segs[i];
      if (t <= s.t1_s + 1e-7 || i === segs.length - 1) {
        activeSegIndex = i;
        const span = Math.max(s.t1_s - s.t0_s, 1e-9);
        const alpha = Math.max(0.0, Math.min(1.0, (t - s.t0_s) / span));
        pos = [
          s.start[0] + (s.end[0] - s.start[0]) * alpha,
          s.start[1] + (s.end[1] - s.start[1]) * alpha,
          s.start[2] + (s.end[2] - s.start[2]) * alpha,
        ];
        moveClass = String(s.class || s.kind);
        feedS = Number(s.feed_mm_s || 0.0);
        feedMin = Number(s.feed_mm_min || feedS * 60.0);
        break;
      }
    }

    state.toolPos = pos;
    state.activeSegIndex = activeSegIndex;

    document.getElementById("dro-x").textContent = pos[0].toFixed(3) + " mm";
    document.getElementById("dro-y").textContent = pos[1].toFixed(3) + " mm";
    document.getElementById("dro-z").textContent = pos[2].toFixed(3) + " mm";
    document.getElementById("dro-class").textContent = moveClass;
    document.getElementById("dro-feed").textContent =
      feedS.toFixed(1) + " mm/s (" + Math.round(feedMin) + " mm/min)";
    document.getElementById("dro-seg").textContent =
      (activeSegIndex + 1) + " / " + segs.length;
    document.getElementById("dro-mesh-src").textContent =
      String(state.currentVariant.stock_mesh_source || "voxel_0.25mm");

    if (activeSegIndex >= 0 && segs[activeSegIndex]) {
      const s = segs[activeSegIndex];
      const gl = renderer.gl;
      uploadStaticBuffer(
        gl,
        renderer.buffers.activePos,
        new Float32Array([s.start[0], s.start[1], s.start[2], s.end[0], s.end[1], s.end[2]])
      );
      uploadStaticBuffer(
        gl,
        renderer.buffers.activeCol,
        new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
      );
    }
  }

  // --- Select Variant Within Current Case ---
  async function selectVariant(variantId) {
    if (!state.currentCase) return;
    const variant = state.currentCase.variants.find((v) => v.id === variantId) || state.currentCase.variants[0];
    if (!variant) return;
    state.currentVariant = variant;

    // Update variant button active states
    const btnWrap = document.getElementById("variant-buttons");
    for (const child of Array.from(btnWrap.children)) {
      if (child.getAttribute("data-vid") === variant.id) {
        child.classList.add("btn-active");
      } else {
        child.classList.remove("btn-active");
      }
    }

    // Update download links
    const dlRustcnc = document.getElementById("dl-rustcnc");
    dlRustcnc.setAttribute("href", state.currentCase.rustcnc_download_url);

    const dlCamotics = document.getElementById("dl-camotics");
    dlCamotics.setAttribute("href", variant.camotics_project_url);

    const dlNc = document.getElementById("dl-nc");
    if (variant.haas_nc_url) {
      dlNc.setAttribute("href", variant.haas_nc_url);
      dlNc.textContent = "Haas NGC (.nc)";
    } else {
      dlNc.setAttribute("href", variant.camotics_nc_url);
      dlNc.textContent = "RS-274 (.nc)";
    }

    uploadVariantToolpath(variant);
    renderKpiStrip();
    renderInspectorTab();

    // Load resulting cut stock STL
    try {
      clearError();
      const stockMesh = await fetchParsedStl(variant.stock_stl_url);
      const gl = renderer.gl;
      uploadStaticBuffer(gl, renderer.buffers.stockPos, stockMesh.positions);
      uploadStaticBuffer(gl, renderer.buffers.stockNrm, stockMesh.normals);
      state.stockCount = stockMesh.vertexCount;
    } catch (err) {
      showError(err.message || "Unable to load resulting stock mesh.");
    }
  }

  // --- Load Benchmark Case ---
  async function loadCase(caseJsonUrl, preferredVariantId) {
    try {
      clearError();
      const caseData = await fetchJson(caseJsonUrl);
      state.currentCase = caseData;

      const gl = renderer.gl;
      const guides = buildSceneGuideLines(caseData.stock_min_mm, caseData.stock_max_mm);
      uploadStaticBuffer(gl, renderer.buffers.guidePos, guides.positions);
      uploadStaticBuffer(gl, renderer.buffers.guideCol, guides.colors);
      state.guideGridCount = guides.gridCount;
      state.guideBilletOffset = guides.billetOffset;
      state.guideBilletCount = guides.billetCount;

      const toolMesh = buildToolAssemblyMesh(caseData.tool || {});
      uploadStaticBuffer(gl, renderer.buffers.toolPos, toolMesh.positions);
      uploadStaticBuffer(gl, renderer.buffers.toolNrm, toolMesh.normals);
      uploadStaticBuffer(gl, renderer.buffers.toolCol, toolMesh.colors);
      state.toolCount = toolMesh.vertexCount;

      const targetMesh = await fetchParsedStl(caseData.target_stl_url);
      uploadStaticBuffer(gl, renderer.buffers.targetPos, targetMesh.positions);
      uploadStaticBuffer(gl, renderer.buffers.targetNrm, targetMesh.normals);
      state.targetCount = targetMesh.vertexCount;

      // Center camera on stock center
      const cx = 0.5 * (caseData.stock_min_mm[0] + caseData.stock_max_mm[0]);
      const cy = 0.5 * (caseData.stock_min_mm[1] + caseData.stock_max_mm[1]);
      const cz = 0.5 * (caseData.stock_min_mm[2] + caseData.stock_max_mm[2]);
      state.cam.target = [cx, cy, cz];

      // Rebuild variant switcher buttons
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
          selectVariant(v.id);
        });
        btnWrap.appendChild(b);
      }

      const targetVid = preferredVariantId || (state.currentVariant && state.currentVariant.id) || "ste_adam";
      await selectVariant(targetVid);
    } catch (err) {
      showError(err.message || "Unable to load benchmark part.");
    }
  }

  // --- Render KPI Strip & Side-by-Side Tabular Results ---
  function renderKpiStrip() {
    const strip = document.getElementById("kpi-strip");
    strip.replaceChildren();
    if (!state.currentVariant) return;
    const m = state.currentVariant.metrics;
    const gainSign = m.cycle_time_reduction_pct >= 0.05 ? "-" + m.cycle_time_reduction_pct.toFixed(1) + "%" : "0.0%";
    const safetyOk = Boolean(m.posted_geometry_pass && m.posted_safety_pass);

    const cards = [
      { label: "Cycle Time", value: m.cycle_time_s.toFixed(2) + " s", cls: "kpi-cyan" },
      { label: "vs Nominal (@2)", value: gainSign, cls: "kpi-good" },
      { label: "Peak MRR", value: m.peak_mrr_mm3_s.toFixed(1) + " mm³/s", cls: "" },
      { label: "0.125mm Oracle", value: safetyOk ? "PASS (0 mm³)" : "CHECK", cls: safetyOk ? "kpi-good" : "" },
    ];
    for (const c of cards) {
      strip.appendChild(
        el("div", { className: "kpi-card" }, [
          el("div", { className: "kpi-label", text: c.label }),
          el("div", { className: ("kpi-value " + c.cls).trim(), text: c.value }),
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

      // Verification details for playable 3D variants
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

    // Tab 3: Transfer, Ablations, and 6-Controller Audit
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

  // --- Main WebGL Render Loop ---
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
    gl.enableVertexAttribArray(m.aPos);
    gl.vertexAttribPointer(m.aPos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
    gl.enableVertexAttribArray(m.aNormal);
    gl.vertexAttribPointer(m.aNormal, 3, gl.FLOAT, false, 0, 0);

    if (useVertCol && colBuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
      gl.enableVertexAttribArray(m.aColor);
      gl.vertexAttribPointer(m.aColor, 3, gl.FLOAT, false, 0, 0);
    } else {
      gl.disableVertexAttribArray(m.aColor);
      gl.vertexAttrib3f(m.aColor, baseColor[0], baseColor[1], baseColor[2]);
    }

    gl.drawArrays(gl.TRIANGLES, 0, count);
  }

  function drawLineBuffer(posBuf, colBuf, firstVertex, vertexCount, mvp) {
    if (vertexCount <= 0) return;
    const gl = renderer.gl;
    const l = renderer.line;
    gl.useProgram(l.prog);
    gl.uniformMatrix4fv(l.uMVP, false, mvp);

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(l.aPos);
    gl.vertexAttribPointer(l.aPos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.enableVertexAttribArray(l.aColor);
    gl.vertexAttribPointer(l.aColor, 4, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.LINES, firstVertex, vertexCount);
  }

  function countVisibleSubset(segIndices, maxSegIndex) {
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
    const showTarget = document.getElementById("chk-target").checked;
    const showBillet = document.getElementById("chk-billet").checked;
    const showCuts = document.getElementById("chk-cuts").checked;
    const showRapids = document.getElementById("chk-rapids").checked;
    const showTool = document.getElementById("chk-tool").checked;
    const clipToTime = document.getElementById("chk-clip").checked;

    // 1. Opaque Resulting Stock Mesh
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    if (showStock && state.stockCount > 0) {
      drawMeshBuffer(
        renderer.buffers.stockPos,
        renderer.buffers.stockNrm,
        null,
        state.stockCount,
        vp,
        ident,
        eye,
        [0.72, 0.78, 0.85],
        false,
        1.0
      );
    }

    // 2. Opaque Animated 3D Tool + Shank + CAT40 Holder
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

    // 3. Lines (Grid + G54 + Billet + Toolpaths) with alpha blending
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

    // 4. Translucent Target CAD Part Overlay (depth write off so interior toolpaths remain crisp)
    if (showTarget && state.targetCount > 0) {
      gl.depthMask(false);
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
      gl.depthMask(true);
    }
  }

  // --- Wire Mouse / Touch / UI Controls ---
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
      state.cam.target = [0.0, 0.0, -2.5];
    });

    document.getElementById("btn-play").addEventListener("click", () => {
      if (state.totalDurationS <= 1e-6) return;
      if (!state.playing && state.playTimeS >= state.totalDurationS - 1e-4) {
        state.playTimeS = 0.0;
      }
      state.playing = !state.playing;
      state.lastFrameMs = performance.now();
      document.getElementById("btn-play").textContent = state.playing ? "Pause" : "Play";
    });

    document.getElementById("btn-reset").addEventListener("click", () => {
      state.playing = false;
      state.playTimeS = 0.0;
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
        state.activeTab = btn.getAttribute("data-tab") || "case";
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
        select.appendChild(
          el("option", {
            value: c.case_json_url,
            text:
              c.case_id +
              " (" +
              c.family +
              " · STE -" +
              Number(c.ste_gain_pct || 0).toFixed(1) +
              "%)",
          })
        );
      }

      if (manifest.cases && manifest.cases.length > 0) {
        await loadCase(manifest.cases[0].case_json_url, "ste_adam");
      }
      state.lastFrameMs = performance.now();
      requestAnimationFrame(renderFrame);
    } catch (err) {
      showError(err.message || "Failed to initialize DiffCAM Manufacture Simulator.");
    }
  }

  window.addEventListener("DOMContentLoaded", boot);
})();
