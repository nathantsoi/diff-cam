# CAMotics 1.2.0 headless (`camsim`) for the T1.3 external cross-check

Upstream binary release (GPL-2.0-or-later), not built from the pinned source checkout
(`external/camotics` @ e84665f needs cbang/Qt5/SCons/V8; not attempted).

```bash
cd docker/camotics
curl -L -o camotics_1.2.0_amd64.deb \
  https://github.com/CauldronDevelopmentLLC/CAMotics/releases/download/1.2.0-release/camotics_1.2.0_amd64.deb
sha256sum camotics_1.2.0_amd64.deb   # expect 5dc5ba0c53a8eb8ce6b3bd2c3bd903465fde860742a7764f2511b188a507c2c0
docker build -t diffcam-camotics:1.2.0 .
docker run --rm -u $(id -u):$(id -g) -v $PWD/work:/work diffcam-camotics:1.2.0 camsim --binary p.camotics out.stl
```

Project file (`.camotics`, JSON): `units`, `resolution-mode: manual`, `resolution` (mm), `tools` (shape `cylindrical`, `diameter`, `length`),
`workpiece.bounds`, `files` (NC). Output STL is the cut workpiece.

Smoke 2026-11-08 (coordinator): 30x30x10 mm stock, 20 mm slot, 6.35 mm flat, 2 mm deep. Analytic remaining volume 8682.7 mm^3.
camsim mesh volume 9277.1 (res 0.25, +594) and 8761.0 (res 0.1, +78): converges, but the surface is computed over bounds padded by one
cell, so the mesh is inflated at coarse resolution. T1.3 must quantify this (and arc/rapid handling) before using it as a cross-check.
