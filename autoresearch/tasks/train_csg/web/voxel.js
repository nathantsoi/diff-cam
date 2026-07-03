// WebGPU voxel-cutting viewer for the train_csg preview page.
//
// Faithful port of the aim3d WebGPU voxelizer + renderer:
//   aim3d/ui/frontend/src/services/webgpuVoxelizer.js   (init / cut / marching-cubes WGSL)
//   aim3d/ui/frontend/src/services/webgpuRenderer.js    (lit solid render pipeline)
//   aim3d/ui/frontend/src/services/viewportProjection.js (perspective + lookAt MVP)
//   aim3d/ui/frontend/src/services/viewportControls.js  (orbit camera basis)
//
// The voxel volume is a 1D f32 storage buffer (SDF: density<0 = inside material,
// density>=0 = carved). A flat-endmill swept segment ("cut") subtracts material via
// the sdSweptTool SDF; marching cubes extracts a mesh each time the grid changes.
//
// Coordinate space: stock-normalized [0,1]^3 (the same space as trajectory.npy), so
// stockSize=[1,1,1], stockLocation=[0,0,0], tool radius in normalized units. The orbit
// camera (target/distance/yaw/pitch, Z-up) is shared with the page's D3 viewer.
//
// Exports: createVoxelViewer(canvas, { gridResolution, toolRadius, onStatus }) -> viewer

import { edgeTable, triTable } from "./marchingCubesTables.js";

// ---- camera math (ported from viewportControls.js / viewportProjection.js) ----
const camFwd = (y, p) => { const cp = Math.cos(p); return [-cp*Math.sin(y), -cp*Math.cos(y), -Math.sin(p)]; };
const camUp  = (y, p) => { const sp = Math.sin(p); return [-sp*Math.sin(y), -sp*Math.cos(y), Math.cos(p)]; };
const camRight = (y) => [-Math.cos(y), Math.sin(y), 0];
const camEye = (c) => {
  const f = camFwd(c.yaw, c.pitch);
  return [c.target[0]-f[0]*c.distance, c.target[1]-f[1]*c.distance, c.target[2]-f[2]*c.distance];
};
const vlen = (v) => Math.hypot(v[0], v[1], v[2]);
const vsub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const vdot = (a, b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const vcross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const vnorm = (v) => { const l = vlen(v) || 1; return [v[0]/l, v[1]/l, v[2]/l]; };

// Build a vertical (+Z) cylinder mesh on the axis (cx, cy) from bottomZ to topZ.
// Returns a Float32Array of solid-pipeline vertices (pos3 + normal3 + color4 = 10
// floats/vertex) and the vertex count. Closed top cap, open bottom. Used for both the
// cutting tool (flat endmill) and the toolholder (the wider cylinder stacked atop it).
function buildCylinder(cx, cy, bottomZ, topZ, radius, segments, color) {
  const r = Math.max(0.001, radius);
  const cols = segments;
  const verts = [];
  const pushV = (x, y, z, nx, ny, nz) => { verts.push(x, y, z, nx, ny, nz, color[0], color[1], color[2], color[3]); };
  // Side wall: two CCW triangles per quad so cullMode "back" keeps only the outer
  // surface (consistent winding = (bot[i], bot[j], top[j]) + (bot[i], top[j], top[i])).
  const ring = (zz) => { const out = []; for (let i=0;i<cols;i++){ const t=(i/cols)*Math.PI*2; out.push([cx+r*Math.cos(t), cy+r*Math.sin(t), zz]); } return out; };
  const bot = ring(bottomZ), top = ring(topZ);
  for (let i=0;i<cols;i++){
    const j = (i+1)%cols;
    const n0 = [Math.cos((i/cols)*Math.PI*2), Math.sin((i/cols)*Math.PI*2), 0];
    const n1 = [Math.cos((j/cols)*Math.PI*2), Math.sin((j/cols)*Math.PI*2), 0];
    pushV(bot[i][0],bot[i][1],bot[i][2], n0[0],n0[1],n0[2]);
    pushV(bot[j][0],bot[j][1],bot[j][2], n1[0],n1[1],n1[2]);
    pushV(top[j][0],top[j][1],top[j][2], n1[0],n1[1],n1[2]);
    pushV(bot[i][0],bot[i][1],bot[i][2], n0[0],n0[1],n0[2]);
    pushV(top[j][0],top[j][1],top[j][2], n1[0],n1[1],n1[2]);
    pushV(top[i][0],top[i][1],top[i][2], n0[0],n0[1],n0[2]);
  }
  // Top cap (fan), CCW viewed from above.
  for (let i=0;i<cols;i++){
    const j = (i+1)%cols;
    pushV(cx, cy, topZ, 0, 0, 1);
    pushV(top[i][0], top[i][1], topZ, 0, 0, 1);
    pushV(top[j][0], top[j][1], topZ, 0, 0, 1);
  }
  return { data: new Float32Array(verts), count: verts.length / 10 };
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far+near)*nf, -1,
    0, 0, (2*far*near)*nf, 0,
  ]);
}
function lookAt(eye, center, up) {
  const z = vnorm(vsub(eye, center));
  const x = vnorm(vcross(up, z));
  const y = vcross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -vdot(x,eye), -vdot(y,eye), -vdot(z,eye), 1,
  ]);
}
function mat4Mul(a, b) {
  const o = new Float32Array(16);
  for (let r=0; r<4; r++) for (let c=0; c<4; c++) {
    o[r*4+c] = a[r*4]*b[c] + a[r*4+1]*b[4+c] + a[r*4+2]*b[8+c] + a[r*4+3]*b[12+c];
  }
  return o;
}
function viewProj(cam, w, h) {
  const eye = camEye(cam);
  const up = camUp(cam.yaw, cam.pitch);
  const aspect = Math.max(1, w) / Math.max(1, h);
  return mat4Mul(lookAt(eye, cam.target, up), perspective(Math.PI/4, aspect, 0.01, 100));
}

// ---- WGSL shaders (verbatim from aim3d webgpuVoxelizer.js) ----
const PARAMS_STRUCT = `
  struct Params {
    gridSize: vec3<u32>, numCuts: u32,
    voxelSize: vec3<f32>, pad1: f32,
    gridOffset: vec3<f32>, pad2: f32,
    stockSize: vec3<f32>, pad3: f32,
    stockLocation: vec3<f32>, uiScale: f32,
  };
`;

const INIT_CODE = `
${PARAMS_STRUCT}
@group(0) @binding(0) var<storage, read_write> grid: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
  let d = abs(p) - b;
  return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x || global_id.y >= params.gridSize.y || global_id.z >= params.gridSize.z) { return; }
  let idx = global_id.x + global_id.y * params.gridSize.x + global_id.z * params.gridSize.x * params.gridSize.y;
  let pos = params.gridOffset + vec3<f32>(global_id) * params.voxelSize;
  let boxSize = params.stockSize * 0.5;
  let center = params.stockLocation + boxSize;
  grid[idx] = sdBox(pos - center, boxSize);
}
`;

const CUT_CODE = `
${PARAMS_STRUCT}
struct Cut {
  start: vec3<f32>, radius: f32,
  end: vec3<f32>, pad: f32,
};
@group(0) @binding(0) var<storage, read_write> grid: array<f32>;
@group(0) @binding(1) var<storage, read> cuts: array<Cut>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> volumeCounter: atomic<u32>;
fn sdSweptTool(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
  let pa = p.xy - a.xy;
  let ba = b.xy - a.xy;
  let ba2 = dot(ba, ba);
  var h: f32 = 0.0;
  if (ba2 > 0.000001) {
    h = clamp(dot(pa, ba) / ba2, 0.0, 1.0);
  }
  let d_xy = length(pa - ba * h) - r;
  let z_tool = a.z + h * (b.z - a.z);
  let d_z = z_tool - p.z;
  let d_out = length(vec2<f32>(max(d_xy, 0.0), max(d_z, 0.0)));
  let d_in = min(max(d_xy, d_z), 0.0);
  return d_out + d_in;
}
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x || global_id.y >= params.gridSize.y || global_id.z >= params.gridSize.z) { return; }
  let idx = global_id.x + global_id.y * params.gridSize.x + global_id.z * params.gridSize.x * params.gridSize.y;
  let pos = params.gridOffset + vec3<f32>(global_id) * params.voxelSize;
  var density = grid[idx];
  let originalDensity = density;
  for (var i = 0u; i < params.numCuts; i = i + 1u) {
    let cut = cuts[i];
    let d = sdSweptTool(pos, cut.start, cut.end, cut.radius);
    density = max(density, -d);
  }
  if (originalDensity < 0.0 && density >= 0.0) {
    atomicAdd(&volumeCounter, 1u);
  }
  grid[idx] = density;
}
`;

const MC_CODE = `
${PARAMS_STRUCT}
struct Vertex {
  px: f32, py: f32, pz: f32,
  nx: f32, ny: f32, nz: f32,
  cr: f32, cg: f32, cb: f32, ca: f32,
};
@group(0) @binding(0) var<storage, read> grid: array<f32>;
@group(0) @binding(1) var<storage, read> edgeTable: array<i32>;
@group(0) @binding(2) var<storage, read> triTable: array<i32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(6) var<uniform> params: Params;
fn getDensity(x: u32, y: u32, z: u32) -> f32 {
  let idx = x + y * params.gridSize.x + z * params.gridSize.x * params.gridSize.y;
  return grid[idx];
}
fn getPos(x: u32, y: u32, z: u32) -> vec3<f32> {
  return params.gridOffset + vec3<f32>(f32(x), f32(y), f32(z)) * params.voxelSize;
}
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (global_id.x >= params.gridSize.x - 1u || global_id.y >= params.gridSize.y - 1u || global_id.z >= params.gridSize.z - 1u) { return; }
  let x = global_id.x; let y = global_id.y; let z = global_id.z;
  var val = array<f32, 8>(
    getDensity(x, y, z), getDensity(x+1u, y, z),
    getDensity(x+1u, y+1u, z), getDensity(x, y+1u, z),
    getDensity(x, y, z+1u), getDensity(x+1u, y, z+1u),
    getDensity(x+1u, y+1u, z+1u), getDensity(x, y+1u, z+1u)
  );
  var p = array<vec3<f32>, 8>(
    getPos(x, y, z), getPos(x+1u, y, z),
    getPos(x+1u, y+1u, z), getPos(x, y+1u, z),
    getPos(x, y, z+1u), getPos(x+1u, y, z+1u),
    getPos(x+1u, y+1u, z+1u), getPos(x, y+1u, z+1u)
  );
  var cubeIndex = 0u;
  if (val[0] < 0.0) { cubeIndex |= 1u; }
  if (val[1] < 0.0) { cubeIndex |= 2u; }
  if (val[2] < 0.0) { cubeIndex |= 4u; }
  if (val[3] < 0.0) { cubeIndex |= 8u; }
  if (val[4] < 0.0) { cubeIndex |= 16u; }
  if (val[5] < 0.0) { cubeIndex |= 32u; }
  if (val[6] < 0.0) { cubeIndex |= 64u; }
  if (val[7] < 0.0) { cubeIndex |= 128u; }
  let edges = edgeTable[cubeIndex];
  if (edges == 0) { return; }
  var vertList = array<vec3<f32>, 12>();
  if ((edges & 1) != 0) { vertList[0] = mix(p[0], p[1], val[0] / (val[0] - val[1])); }
  if ((edges & 2) != 0) { vertList[1] = mix(p[1], p[2], val[1] / (val[1] - val[2])); }
  if ((edges & 4) != 0) { vertList[2] = mix(p[2], p[3], val[2] / (val[2] - val[3])); }
  if ((edges & 8) != 0) { vertList[3] = mix(p[3], p[0], val[3] / (val[3] - val[0])); }
  if ((edges & 16) != 0) { vertList[4] = mix(p[4], p[5], val[4] / (val[4] - val[5])); }
  if ((edges & 32) != 0) { vertList[5] = mix(p[5], p[6], val[5] / (val[5] - val[6])); }
  if ((edges & 64) != 0) { vertList[6] = mix(p[6], p[7], val[6] / (val[6] - val[7])); }
  if ((edges & 128) != 0) { vertList[7] = mix(p[7], p[4], val[7] / (val[7] - val[4])); }
  if ((edges & 256) != 0) { vertList[8] = mix(p[0], p[4], val[0] / (val[0] - val[4])); }
  if ((edges & 512) != 0) { vertList[9] = mix(p[1], p[5], val[1] / (val[1] - val[5])); }
  if ((edges & 1024) != 0) { vertList[10] = mix(p[2], p[6], val[2] / (val[2] - val[6])); }
  if ((edges & 2048) != 0) { vertList[11] = mix(p[3], p[7], val[3] / (val[3] - val[7])); }
  for (var i = 0u; i < 16u; i = i + 3u) {
    let e0 = triTable[cubeIndex * 16u + i];
    if (e0 == -1) { break; }
    let e1 = triTable[cubeIndex * 16u + i + 1u];
    let e2 = triTable[cubeIndex * 16u + i + 2u];
    let tIdx = atomicAdd(&counter, 3u);
    let n = normalize(cross(vertList[e1] - vertList[e0], vertList[e2] - vertList[e0]));
    vertices[tIdx] = Vertex(vertList[e0].x * params.uiScale, vertList[e0].y * params.uiScale, vertList[e0].z * params.uiScale, n.x, n.y, n.z, 0.5, 0.5, 0.5, 1.0);
    vertices[tIdx+1u] = Vertex(vertList[e1].x * params.uiScale, vertList[e1].y * params.uiScale, vertList[e1].z * params.uiScale, n.x, n.y, n.z, 0.5, 0.5, 0.5, 1.0);
    vertices[tIdx+2u] = Vertex(vertList[e2].x * params.uiScale, vertList[e2].y * params.uiScale, vertList[e2].z * params.uiScale, n.x, n.y, n.z, 0.5, 0.5, 0.5, 1.0);
    indices[tIdx] = tIdx;
    indices[tIdx+1u] = tIdx+1u;
    indices[tIdx+2u] = tIdx+2u;
  }
}
`;

// Lit solid render shader (ported from webgpuRenderer.js VERTEX_SHADER).
const RENDER_CODE = `
struct Uniforms { viewProj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct SolidIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
};
struct LineIn {
  @location(0) position: vec3<f32>,
  @location(1) color: vec4<f32>,
};
struct Out {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};
@vertex
fn solid_main(input: SolidIn) -> Out {
  var out: Out;
  let light = normalize(vec3<f32>(0.4, 0.7, 0.9));
  let shade = max(dot(normalize(input.normal), light), 0.18);
  out.position = uniforms.viewProj * vec4<f32>(input.position, 1.0);
  out.color = vec4<f32>(input.color.rgb * shade, input.color.a);
  return out;
}
@vertex
fn line_main(input: LineIn) -> Out {
  var out: Out;
  out.position = uniforms.viewProj * vec4<f32>(input.position, 1.0);
  out.color = input.color;
  return out;
}
@fragment
fn fragment_main(input: Out) -> @location(0) vec4<f32> { return input.color; }
`;

// Toolholder shader: samples the carved-stock SDF at each fragment's world position
// and colors the holder red where density < 0 (inside remaining material), i.e. where
// the holder gouges the stock (the sim's holder_overlap condition). Elsewhere it keeps
// the holder's base gray. Shares the solid vertex input layout (pos3+normal3+color4).
const HOLDER_CODE = `
struct HolderUniforms { viewProj: mat4x4<f32> };
struct HParams {
  gridSize: vec3<u32>, numCuts: u32,
  voxelSize: vec3<f32>, pad1: f32,
  gridOffset: vec3<f32>, pad2: f32,
  stockSize: vec3<f32>, pad3: f32,
  stockLocation: vec3<f32>, uiScale: f32,
};
@group(0) @binding(0) var<uniform> uniforms: HolderUniforms;
@group(0) @binding(1) var<storage, read> grid: array<f32>;
@group(0) @binding(2) var<uniform> params: HParams;
struct HolderIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
};
struct HOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) baseColor: vec4<f32>,
};
@vertex
fn holder_vert(input: HolderIn) -> HOut {
  var out: HOut;
  out.position = uniforms.viewProj * vec4<f32>(input.position, 1.0);
  out.worldPos = input.position;
  out.baseColor = input.color;
  return out;
}
fn sampleStock(p: vec3<f32>) -> f32 {
  let g = (p - params.gridOffset) / params.voxelSize;
  let ix = u32(clamp(g.x, 0.0, f32(params.gridSize.x) - 1.0));
  let iy = u32(clamp(g.y, 0.0, f32(params.gridSize.y) - 1.0));
  let iz = u32(clamp(g.z, 0.0, f32(params.gridSize.z) - 1.0));
  let idx = ix + iy * params.gridSize.x + iz * params.gridSize.x * params.gridSize.y;
  return grid[idx];
}
@fragment
fn holder_frag(input: HOut) -> @location(0) vec4<f32> {
  let d = sampleStock(input.worldPos);
  let inside = d < 0.0;
  let red = vec3<f32>(0.9, 0.2, 0.2);
  let col = select(input.baseColor.rgb, red, inside);
  return vec4<f32>(col, input.baseColor.a);
}
`;

export async function createVoxelViewer(canvas, options = {}) {
  const onStatus = options.onStatus || (() => {});
  if (!globalThis.navigator?.gpu) {
    onStatus({ available: false, reason: "WebGPU is unavailable in this browser." });
    return { available: false };
  }
  let adapter, device;
  try {
    adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { onStatus({ available: false, reason: "No WebGPU adapter." }); return { available: false }; }
    device = await adapter.requestDevice();
  } catch (e) {
    onStatus({ available: false, reason: "Device init failed: " + e.message });
    return { available: false };
  }

  // Stock in normalized [0,1]^3. toolRadius normalized (3.175mm / 25.4mm).
  const stockSize = [1, 1, 1];
  const stockLocation = [0, 0, 0];
  const uiScale = 1.0;
  // Tool + holder geometry, in stock-normalized units, matching the Taichi
  // CSGSimulatorDelta (simulator/csg_simulator.py) — the same sim that renders the
  // run.mp4 video via record_video. The sim measures SDFs in voxel space
  // (r_vox = radius_mm / voxel_size_mm) over a grid of Nx = stock_mm / voxel_size_mm
  // cells, so a normalized [0,1] radius is r_vox / Nx = radius_mm / stock_mm. The
  // defaults below are the sim's own defaults (tool_radius_mm=3.175, tool_height_mm=25.0,
  // 2.5"-diameter holder, 10"-Z work volume) on a 1" stock → radius 0.125, height 0.984,
  // holderR 1.25, holderH 10.0. index.html overrides these per-run from args.json via
  // setToolGeometry() so the WebGPU tool matches the run's recorded Taichi video.
  let toolRadius = options.toolRadius ?? 0.125;
  let TOOL_HEIGHT = options.toolHeight ?? 0.984;
  let holderRadius = options.holderRadius ?? 1.25;
  let HOLDER_HEIGHT = options.holderHeight ?? 10.0;
  const baseRes = options.gridResolution ?? 96;
  const userGridSize = [baseRes, baseRes, baseRes];
  const padding = 2;
  const gridSize = [userGridSize[0]+padding*2, userGridSize[1]+padding*2, userGridSize[2]+padding*2];
  const numVoxels = gridSize[0]*gridSize[1]*gridSize[2];
  const maxTriangles = Math.floor(numVoxels * 0.2);
  const maxVertices = maxTriangles * 3;
  const voxelSize = [stockSize[0]/userGridSize[0], stockSize[1]/userGridSize[1], stockSize[2]/userGridSize[2]];
  const gridOffset = [stockLocation[0]-padding*voxelSize[0], stockLocation[1]-padding*voxelSize[1], stockLocation[2]-padding*voxelSize[2]];

  const U = GPUBufferUsage;
  const gridBuffer = device.createBuffer({ size: numVoxels*4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const edgeTableBuffer = device.createBuffer({ size: 256*4, usage: U.STORAGE | U.COPY_DST, mappedAtCreation: true });
  new Int32Array(edgeTableBuffer.getMappedRange()).set(edgeTable); edgeTableBuffer.unmap();
  const triTableBuffer = device.createBuffer({ size: 4096*4, usage: U.STORAGE | U.COPY_DST, mappedAtCreation: true });
  new Int32Array(triTableBuffer.getMappedRange()).set(triTable); triTableBuffer.unmap();
  const vertexBuffer = device.createBuffer({ size: maxVertices*10*4, usage: U.STORAGE | U.VERTEX | U.COPY_SRC });
  const indexBuffer = device.createBuffer({ size: maxVertices*4, usage: U.STORAGE | U.INDEX | U.COPY_SRC });
  const counterBuffer = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const volumeCounterBuffer = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  device.queue.writeBuffer(volumeCounterBuffer, 0, new Uint32Array([0]));
  const paramsBuffer = device.createBuffer({ size: 128, usage: U.UNIFORM | U.COPY_DST });
  const cutBuffer = device.createBuffer({ size: 1000*32, usage: U.STORAGE | U.COPY_DST });

  const updateParams = (numCuts) => {
    const data = new Float32Array(32); const u = new Uint32Array(data.buffer);
    u[0]=gridSize[0]; u[1]=gridSize[1]; u[2]=gridSize[2]; u[3]=numCuts;
    data[4]=voxelSize[0]; data[5]=voxelSize[1]; data[6]=voxelSize[2];
    data[8]=gridOffset[0]; data[9]=gridOffset[1]; data[10]=gridOffset[2];
    data[12]=stockSize[0]; data[13]=stockSize[1]; data[14]=stockSize[2];
    data[16]=stockLocation[0]; data[17]=stockLocation[1]; data[18]=stockLocation[2]; data[19]=uiScale;
    device.queue.writeBuffer(paramsBuffer, 0, data);
  };
  updateParams(0);

  const mkCompute = (code, layoutEntries) => {
    const layout = device.createBindGroupLayout({ entries: layoutEntries });
    const mod = device.createShaderModule({ code });
    const pipe = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module: mod, entryPoint: "main" } });
    return { pipe, layout };
  };

  const initBGL = mkCompute(INIT_CODE, [
    { binding:0, visibility:GPUShaderStage.COMPUTE, buffer:{type:"storage"} },
    { binding:1, visibility:GPUShaderStage.COMPUTE, buffer:{type:"uniform"} },
  ]);
  const initBG = device.createBindGroup({ layout:initBGL.pipe.getBindGroupLayout(0), entries:[{binding:0,resource:{buffer:gridBuffer}},{binding:1,resource:{buffer:paramsBuffer}}] });

  const cutBGL = mkCompute(CUT_CODE, [
    { binding:0, visibility:GPUShaderStage.COMPUTE, buffer:{type:"storage"} },
    { binding:1, visibility:GPUShaderStage.COMPUTE, buffer:{type:"read-only-storage"} },
    { binding:2, visibility:GPUShaderStage.COMPUTE, buffer:{type:"uniform"} },
    { binding:3, visibility:GPUShaderStage.COMPUTE, buffer:{type:"storage"} },
  ]);
  const cutBG = device.createBindGroup({ layout:cutBGL.pipe.getBindGroupLayout(0), entries:[{binding:0,resource:{buffer:gridBuffer}},{binding:1,resource:{buffer:cutBuffer}},{binding:2,resource:{buffer:paramsBuffer}},{binding:3,resource:{buffer:volumeCounterBuffer}}] });

  const mcBGL = mkCompute(MC_CODE, [
    { binding:0, visibility:GPUShaderStage.COMPUTE, buffer:{type:"read-only-storage"} },
    { binding:1, visibility:GPUShaderStage.COMPUTE, buffer:{type:"read-only-storage"} },
    { binding:2, visibility:GPUShaderStage.COMPUTE, buffer:{type:"read-only-storage"} },
    { binding:3, visibility:GPUShaderStage.COMPUTE, buffer:{type:"storage"} },
    { binding:4, visibility:GPUShaderStage.COMPUTE, buffer:{type:"storage"} },
    { binding:5, visibility:GPUShaderStage.COMPUTE, buffer:{type:"storage"} },
    { binding:6, visibility:GPUShaderStage.COMPUTE, buffer:{type:"uniform"} },
  ]);
  const mcBG = device.createBindGroup({ layout:mcBGL.pipe.getBindGroupLayout(0), entries:[
    {binding:0,resource:{buffer:gridBuffer}},{binding:1,resource:{buffer:edgeTableBuffer}},{binding:2,resource:{buffer:triTableBuffer}},
    {binding:3,resource:{buffer:vertexBuffer}},{binding:4,resource:{buffer:indexBuffer}},{binding:5,resource:{buffer:counterBuffer}},{binding:6,resource:{buffer:paramsBuffer}},
  ] });

  const dispatch = (pass) => pass.dispatchWorkgroups(Math.ceil(gridSize[0]/8), Math.ceil(gridSize[1]/8), Math.ceil(gridSize[2]/4));

  // Init the SDF grid (solid stock box).
  const initGrid = () => {
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(initBGL.pipe); pass.setBindGroup(0, initBG); dispatch(pass); pass.end();
    device.queue.submit([enc.finish()]);
  };
  initGrid();

  // ---- render pipeline (lit solid + lines) ----
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });
  const uniformBuffer = device.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST });
  const bindGroupLayout = device.createBindGroupLayout({ entries: [{ binding:0, visibility:GPUShaderStage.VERTEX, buffer:{type:"uniform"} }] });
  const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: [{ binding:0, resource:{buffer:uniformBuffer} }] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const shader = device.createShaderModule({ code: RENDER_CODE });

  const solidPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shader, entryPoint: "solid_main", buffers: [{
      arrayStride: 40,
      attributes: [
        { shaderLocation:0, offset:0,  format:"float32x3" },
        { shaderLocation:1, offset:12, format:"float32x3" },
        { shaderLocation:2, offset:24, format:"float32x4" },
      ],
    }]},
    fragment: { module: shader, entryPoint: "fragment_main", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
  });
  const linePipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shader, entryPoint: "line_main", buffers: [{
      arrayStride: 28,
      attributes: [
        { shaderLocation:0, offset:0,  format:"float32x3" },
        { shaderLocation:1, offset:12, format:"float32x4" },
      ],
    }]},
    fragment: { module: shader, entryPoint: "fragment_main", targets: [{ format }] },
    primitive: { topology: "line-list", cullMode: "none" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less-equal" },
  });
  // Transparent solid pipeline for the cutting-tool cylinder. Same vertex layout as the
  // opaque solid pipeline, but drawn with depthCompare "always" so the tool is never
  // occluded by the carved stock (the tool tip sits inside the stock, so a depth-tested
  // cylinder would be hidden by the stock surface in front of it). depthWrite stays off
  // so the tool doesn't mask the line overlay drawn afterward.
  const transPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: shader, entryPoint: "solid_main", buffers: [{
      arrayStride: 40,
      attributes: [
        { shaderLocation:0, offset:0,  format:"float32x3" },
        { shaderLocation:1, offset:12, format:"float32x3" },
        { shaderLocation:2, offset:24, format:"float32x4" },
      ],
    }]},
    fragment: { module: shader, entryPoint: "fragment_main", targets: [{ format, blend: {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    }}] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
  });
  // Toolholder pipeline: its own shader module (HOLDER_CODE) that binds the carved-stock
  // SDF grid + params so the fragment can recolor the holder red where it gouges the
  // stock (density < 0). Separate bind group: viewProj uniform + grid (read-only
  // storage) + params uniform. Same vertex layout and blend/depth setup as the tool.
  const holderShader = device.createShaderModule({ code: HOLDER_CODE });
  const holderBGL = device.createBindGroupLayout({ entries: [
    { binding:0, visibility:GPUShaderStage.VERTEX, buffer:{type:"uniform"} },
    { binding:1, visibility:GPUShaderStage.FRAGMENT, buffer:{type:"read-only-storage"} },
    { binding:2, visibility:GPUShaderStage.FRAGMENT, buffer:{type:"uniform"} },
  ]});
  const holderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [holderBGL] }),
    vertex: { module: holderShader, entryPoint: "holder_vert", buffers: [{
      arrayStride: 40,
      attributes: [
        { shaderLocation:0, offset:0,  format:"float32x3" },
        { shaderLocation:1, offset:12, format:"float32x3" },
        { shaderLocation:2, offset:24, format:"float32x4" },
      ],
    }]},
    fragment: { module: holderShader, entryPoint: "holder_frag", targets: [{ format, blend: {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    }}] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
  });
  const holderBG = device.createBindGroup({ layout: holderBGL, entries: [
    { binding:0, resource:{buffer:uniformBuffer} },
    { binding:1, resource:{buffer:gridBuffer} },
    { binding:2, resource:{buffer:paramsBuffer} },
  ]});

  let depthTex = null, width = 1, height = 1;
  const resize = () => {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height;
    }
    if (depthTex) depthTex.destroy();
    depthTex = device.createTexture({ size:[width,height], format:"depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  };

  // ---- line overlay (tool path + stock wireframe) ----
  // Layout per vertex: pos(3 f32) + color(4 f32) = 28 bytes (matches linePipeline).
  const STOCK_WIRE = [
    [[0,0,0],[1,0,0]],[[1,0,0],[1,1,0]],[[1,1,0],[0,1,0]],[[0,1,0],[0,0,0]],
    [[0,0,1],[1,0,1]],[[1,0,1],[1,1,1]],[[1,1,1],[0,1,1]],[[0,1,1],[0,0,1]],
    [[0,0,0],[0,0,1]],[[1,0,0],[1,0,1]],[[1,1,0],[1,1,1]],[[0,1,0],[0,1,1]],
  ];
  let lineBuffer = null, lineCount = 0;
  let toolBuffer = null, toolVertexCount = 0;
  const uploadLines = (segments) => {
    const n = segments.length;
    const data = new Float32Array(n * 2 * 7);
    for (let i=0;i<n;i++){
      const s = segments[i];
      data.set([...s.a, ...s.color], i*14);
      data.set([...s.b, ...s.color], i*14+7);
    }
    const size = Math.max(4, data.byteLength);
    if (!lineBuffer || lineBuffer.size < size) {
      lineBuffer?.destroy();
      lineBuffer = device.createBuffer({ size, usage: U.VERTEX | U.COPY_DST });
    }
    device.queue.writeBuffer(lineBuffer, 0, data);
    lineCount = n;
  };
  // Upload the cutting-tool cylinder vertices (solid-pipeline layout: 10 floats/vtx).
  let toolLogged = false;
  const uploadTool = (data) => {
    const size = Math.max(4, data.byteLength);
    if (!toolBuffer || toolBuffer.size < size) {
      toolBuffer?.destroy();
      toolBuffer = device.createBuffer({ size, usage: U.VERTEX | U.COPY_DST });
    }
    device.queue.writeBuffer(toolBuffer, 0, data);
    toolVertexCount = data.length / 10;
  };
  // Separate buffer for the toolholder (drawn with the holder pipeline so its color
  // can react to the stock SDF). Solid-pipeline layout: 10 floats/vtx.
  let holderBuffer = null, holderVertexCount = 0;
  const uploadHolder = (data) => {
    const size = Math.max(4, data.byteLength);
    if (!holderBuffer || holderBuffer.size < size) {
      holderBuffer?.destroy();
      holderBuffer = device.createBuffer({ size, usage: U.VERTEX | U.COPY_DST });
    }
    device.queue.writeBuffer(holderBuffer, 0, data);
    holderVertexCount = data.length / 10;
  };

  // ---- cut application + mesh extraction ----
  let vertexCount = 0;
  const countRead = device.createBuffer({ size: 8, usage: U.MAP_READ | U.COPY_DST });
  let extracting = false, dirty = false;

  const applyCuts = (cuts) => {
    if (!cuts || !cuts.length) return;
    const data = new Float32Array(cuts.length * 8);
    for (let i=0;i<cuts.length;i++){
      data[i*8+0]=cuts[i].startX; data[i*8+1]=cuts[i].startY; data[i*8+2]=cuts[i].startZ; data[i*8+3]=cuts[i].radius;
      data[i*8+4]=cuts[i].endX;   data[i*8+5]=cuts[i].endY;   data[i*8+6]=cuts[i].endZ;   data[i*8+7]=0;
    }
    device.queue.writeBuffer(cutBuffer, 0, data);
    updateParams(cuts.length);
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(cutBGL.pipe); pass.setBindGroup(0, cutBG); dispatch(pass); pass.end();
    device.queue.submit([enc.finish()]);
  };

  const extractMesh = async () => {
    if (extracting) { dirty = true; return; }
    extracting = true;
    device.queue.writeBuffer(counterBuffer, 0, new Uint32Array([0]));
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(mcBGL.pipe); pass.setBindGroup(0, mcBG); dispatch(pass); pass.end();
    enc.copyBufferToBuffer(counterBuffer, 0, countRead, 0, 4);
    device.queue.submit([enc.finish()]);
    try {
      await countRead.mapAsync(GPUMapMode.READ);
      const counts = new Uint32Array(countRead.getMappedRange());
      vertexCount = counts[0];
      countRead.unmap();
    } catch (e) { /* ignore */ }
    extracting = false;
    if (dirty) { dirty = false; extractMesh(); }
  };

  // ---- public API ----
  // Trajectory points are normalized [0,1]^3. Re-init stock and carve 0..step-1.
  let carvedStep = 0, trajectoryPts = null, stageBoundaryIdx = null;
  const cutsFor = (upto) => {
    const cuts = [];
    for (let i=0; i<upto && i+1<trajectoryPts.length; i++){
      cuts.push({ startX:trajectoryPts[i][0], startY:trajectoryPts[i][1], startZ:trajectoryPts[i][2], radius:toolRadius,
                  endX:trajectoryPts[i+1][0], endY:trajectoryPts[i+1][1], endZ:trajectoryPts[i+1][2] });
    }
    return cuts;
  };
  const carveToStep = (step) => {
    if (!trajectoryPts) return;
    const upto = Math.max(0, Math.min(step, trajectoryPts.length-1));
    if (upto >= carvedStep) {
      applyCuts(cutsForRange(carvedStep, upto));
    } else {
      initGrid(); carvedStep = 0; applyCuts(cutsForRange(0, upto));
    }
    carvedStep = upto;
    extractMesh();
  };
  const cutsForRange = (from, to) => {
    const cuts = [];
    for (let i=from; i<to && i+1<trajectoryPts.length; i++){
      cuts.push({ startX:trajectoryPts[i][0], startY:trajectoryPts[i][1], startZ:trajectoryPts[i][2], radius:toolRadius,
                  endX:trajectoryPts[i+1][0], endY:trajectoryPts[i+1][1], endZ:trajectoryPts[i+1][2] });
    }
    return cuts;
  };

  const setTrajectory = (pts, stageBoundary) => {
    trajectoryPts = pts ? pts.map(p => [p[0], p[1], p[2]]) : null;
    stageBoundaryIdx = (typeof stageBoundary === "number") ? stageBoundary : null;
    initGrid(); carvedStep = 0; vertexCount = 0;
    if (pts && pts.length) { applyCuts(cutsFor(pts.length-1)); carvedStep = pts.length-1; extractMesh(); }
  };

  // Update tool + holder geometry (stock-normalized units) to match the selected
  // run's args.json — the same dimensions the Taichi sim uses to render run.mp4.
  // The cut radius (toolRadius) changes the carved groove width, so re-carve the
  // current trajectory from scratch at the new radius; tool/holder mesh height and
  // holder radius only affect the drawn cylinder, which the render loop picks up.
  const setToolGeometry = (g = {}) => {
    if (typeof g.toolRadius === "number") toolRadius = g.toolRadius;
    if (typeof g.toolHeight === "number") TOOL_HEIGHT = g.toolHeight;
    if (typeof g.holderRadius === "number") holderRadius = g.holderRadius;
    if (typeof g.holderHeight === "number") HOLDER_HEIGHT = g.holderHeight;
    toolLogged = false;
    // The cut radius may have changed, so re-carve the already-played prefix from
    // a fresh stock up to the current step.
    if (trajectoryPts && trajectoryPts.length) {
      const upto = Math.max(0, Math.min(carvedStep, trajectoryPts.length-1));
      initGrid(); carvedStep = 0;
      applyCuts(cutsForRange(0, upto)); carvedStep = upto;
      extractMesh();
    }
  };

  // Render one frame with the given orbit camera. opts:
  //   showCube : bool        — draw the stock wireframe (default true)
  //   showAxes : bool        — draw the XYZ triad at the stock origin (default false)
  //   cmdPts   : [[x,y,z]…]  — commanded (pre-clip) path to draw dimly (default null)
  // The carved stock mesh, the reached tool path (bright), the not-yet-reached suffix
  // (dim), and the current tool-tip marker are always drawn when a trajectory is set.
  const render = (cam, reachedStep, opts = {}) => {
    resize();
    device.queue.writeBuffer(uniformBuffer, 0, viewProj(cam, width, height));
    // Build the cutting-tool cylinder at the current tip (vertical +Z shank rising
    // above the stock), uploaded before the pass so it can be drawn transparently over
    // the opaque stock.
    let hasTool = false, hasHolder = false;
    if (trajectoryPts) {
      const rstep = Math.max(0, Math.min(reachedStep, trajectoryPts.length-1));
      const tip = trajectoryPts[rstep];
      // Match the Taichi CSGSimulatorDelta tool model (simulator/csg_simulator.py):
      //   tool:   flat-end Z cylinder, bottom at tip, height TOOL_HEIGHT.
      //   holder: Z cylinder stacked on the tool's top (bottom = tip + tool_height),
      //           radius = holder_radius (2.5"-dia spindle), height = holder_height
      //           (machine Z travel). Both are set per-run from args.json.
      const toolTop = tip[2] + TOOL_HEIGHT;
      const toolCyl = buildCylinder(tip[0], tip[1], tip[2], toolTop, toolRadius, 24, [0.90, 0.20, 0.20, 0.5]);
      const holderCyl = buildCylinder(tip[0], tip[1], toolTop, toolTop + HOLDER_HEIGHT, holderRadius, 24, [0.55, 0.58, 0.62, 0.5]);
      uploadTool(toolCyl.data);
      uploadHolder(holderCyl.data);
      hasTool = true; hasHolder = true;
      if (!toolLogged) { toolLogged = true; console.log("[voxel] tool+holder:", { tip, toolR: toolRadius, toolH: TOOL_HEIGHT, toolTop, holderR: holderRadius, holderH: HOLDER_HEIGHT, holderTop: toolTop + HOLDER_HEIGHT }); }
    }
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: context.getCurrentTexture().createView(), clearValue:{r:0.07,g:0.09,b:0.11,a:1}, loadOp:"clear", storeOp:"store" }],
      depthStencilAttachment: { view: depthTex.createView(), depthClearValue:1.0, depthLoadOp:"clear", depthStoreOp:"store" },
    });
    pass.setBindGroup(0, bindGroup);
    // Carved stock mesh.
    if (vertexCount > 0) {
      pass.setPipeline(solidPipeline);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.setIndexBuffer(indexBuffer, "uint32");
      pass.drawIndexed(vertexCount);
    }
    // Cutting-tool cylinder (50% transparent, drawn after the opaque stock).
    if (hasTool) {
      pass.setPipeline(transPipeline);
      pass.setVertexBuffer(0, toolBuffer);
      pass.draw(toolVertexCount);
    }
    // Toolholder: drawn with the holder pipeline so its fragment shader can recolor it
    // red where it gouges the carved stock (stock SDF < 0 at the fragment's world pos).
    if (hasHolder) {
      pass.setPipeline(holderPipeline);
      pass.setBindGroup(0, holderBG);
      pass.setVertexBuffer(0, holderBuffer);
      pass.draw(holderVertexCount);
    }
    // Line overlay: stock wireframe + axes + commanded path + tool path + tool-tip marker.
    const showCube = opts.showCube !== false;
    const showAxes = !!opts.showAxes;
    const cmdPts = opts.cmdPts || null;
    const segs = [];
    if (showCube) {
      STOCK_WIRE.forEach(([a,b]) => segs.push({ a, b, color:[0.22,0.26,0.30,0.5] }));
    }
    if (showAxes) {
      const L = 0.5;
      [[1,0,0,0.90,0.33,0.29],[0,1,0,0.34,0.83,0.39],[0,0,1,0.35,0.63,1.0]].forEach(([dx,dy,dz,r,g,b])=>{
        segs.push({ a:[0,0,0], b:[dx*L,dy*L,dz*L], color:[r,g,b,0.9] });
      });
    }
    if (cmdPts && cmdPts.length > 1) {
      for (let i=1;i<cmdPts.length;i++){
        segs.push({ a:cmdPts[i-1], b:cmdPts[i], color:[0.6,0.65,0.7,0.35] });
      }
    }
    if (trajectoryPts) {
      const rstep = Math.max(0, Math.min(reachedStep, trajectoryPts.length-1));
      // Not-yet-reached suffix: dim, so playback progress is visible.
      for (let i=1;i<trajectoryPts.length;i++){
        if (i <= rstep) continue;
        segs.push({ a:trajectoryPts[i-1], b:trajectoryPts[i], color:[0.22,0.45,0.66,0.25] });
      }
      // Reached prefix: bright; stage 2 (staged runs) in amber to distinguish
      // the second trajectory from the first.
      for (let i=1;i<=rstep;i++){
        const c = (stageBoundaryIdx != null && i > stageBoundaryIdx)
          ? [0.98,0.75,0.14,0.95] : [0.36,0.75,1.0,0.95];
        segs.push({ a:trajectoryPts[i-1], b:trajectoryPts[i], color:c });
      }
      // Stage boundary marker (staged runs): amber 3-axis cross where stage 1
      // ends and stage 2 begins.
      if (stageBoundaryIdx != null && stageBoundaryIdx < trajectoryPts.length) {
        const bp = trajectoryPts[stageBoundaryIdx], L = 0.06;
        [[1,0,0],[0,1,0],[0,0,1]].forEach(([dx,dy,dz])=>{
          segs.push({ a:bp, b:[bp[0]+dx*L,bp[1]+dy*L,bp[2]+dz*L], color:[0.98,0.75,0.14,1] });
          segs.push({ a:bp, b:[bp[0]-dx*L,bp[1]-dy*L,bp[2]-dz*L], color:[0.98,0.75,0.14,1] });
        });
      }
      // tool-tip marker: small 3-axis cross at the current tip.
      const tip = trajectoryPts[rstep], L = 0.05;
      [[1,0,0,1.0,0.82,0.47],[0,1,0,0.34,0.83,0.39],[0,0,1,0.35,0.63,1.0]].forEach(([dx,dy,dz,r,g,b])=>{
        segs.push({ a:tip, b:[tip[0]+dx*L,tip[1]+dy*L,tip[2]+dz*L], color:[r,g,b,1] });
        segs.push({ a:tip, b:[tip[0]-dx*L,tip[1]-dy*L,tip[2]-dz*L], color:[r,g,b,1] });
      });
    }
    if (segs.length) {
      uploadLines(segs);
      pass.setPipeline(linePipeline);
      // The holder draw above rebinds group 0 to holderBG; restore the solid/line
      // bind group before drawing the line overlay (linePipeline expects pipelineLayout).
      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, lineBuffer);
      pass.draw(lineCount * 2);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  };

  onStatus({ available: true, gridResolution: baseRes, toolRadius });
  return {
    available: true,
    setTrajectory,
    setToolGeometry,
    carveToStep,
    render,
    resize,
    get vertexCount() { return vertexCount; },
    destroy() {
      [gridBuffer, edgeTableBuffer, triTableBuffer, vertexBuffer, indexBuffer, counterBuffer, volumeCounterBuffer, paramsBuffer, cutBuffer, countRead, uniformBuffer, lineBuffer, toolBuffer, holderBuffer].forEach(b => b?.destroy?.());
      depthTex?.destroy();
    },
  };
}
