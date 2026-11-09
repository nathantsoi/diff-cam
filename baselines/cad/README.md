# Isolated native STEP backend (T5.2)

Only the Python **bindings** are installed here. This is not the full
`cadquery` distribution, and no CAD package is a core project dependency.
The environment and derived geometry stay under ignored paths; they must not
be added to `pyproject.toml`, `uv.lock`, or source distributions automatically.

## Pinned environment

- `cadquery-ocp==7.7.2`, CPython 3.12 (`cp312`), Linux x86_64,
  `manylinux_2_35_x86_64` wheel.
- Wheel SHA-256:
  `7ac2d83b4f4b3d7c35421a1d9f8fec2adc73b6ed6cac50c1ffcede5552e38e9b`.
- The checked runtime used CPython 3.12.13 and reported wheel version 7.7.2.
- Distribution metadata: <https://github.com/CadQuery/OCP>; its download URL
  points to <https://github.com/jmwright/ocp-build-system/releases>.

Create/recreate this in a **fresh** worktree (do not overwrite an existing
CAD venv you still need):

```bash
uv venv --python 3.12 external/cad-venv
uv pip install --python external/cad-venv/bin/python \
  --require-hashes -r baselines/cad/requirements.txt
```

`requirements.txt` pins the version and wheel hash. The coordinator approved
this isolated environment. `benchmarks.external_tasks` uses that interpreter
as a bounded subprocess, with `-I`, cleared Python path overrides, a timeout,
exact-PID timeout cleanup, and a checked JSON response. `DIFFCAM_CAD_PYTHON`
can select another interpreter, but it must still pass the version, platform,
package and import checks. There is no implicit package installation or mesh
fallback. Missing/unqualified packages or a wrong interpreter return:

```text
blocked: cad env unavailable: <specific reason>
```

The subprocess is an engineering boundary, **not a security sandbox** for
native CAD code. Source file size and SHA-256 are bounded/checked before STEP
loading, and no network or arbitrary mesh source is used by this adapter.

## STEP validation before any sampling

The adapter uses `STEPControl_Reader`, selects native model unit `MM`, calls
`ReadFile`, and then **calls `TransferRoots()`**. This binding returns the
integer number of successful translations, not a boolean. Only after a valid
one-root/one-solid result, a valid B-rep, one shell, finite physical bounds,
positive native volume and a measured reference cylindrical radius may the
sampler run. OCP fulfills the inch-to-mm conversion; the adapter verifies the
observed dimension rather than multiplying already-normalized geometry again.

Mesh/STL parsing is not a STEP parser and does not establish STEP validity.
The point sampler uses native solid classification for its sign and native
point-to-boundary distances for magnitudes. Converted NumPy float64 fields
carry their grid, millimetre units, sign convention, and content hashes.

For the inspected local Titan-M1 source, the verified import reports one root,
one transferred shape, one valid solid, one shell and 67 faces; its extent is
approximately 101.6 × 48.26 × 19.05 mm and volume is 60736.926417 mm³. The
0.195-inch reference cylinder is observed at radius 4.953 mm. These checks do
not certify the design, machining or provenance.

**Titan source restriction:** source SHA-256 is
`3ba16d17de50526b452f844a7faf66a624ddd9509fcc3eac2087fdbdbc2066e3`.
Its origin/license is `"unknown / provenance unrecorded"`. It is local-only
input. Do not copy or redistribute the STEP or derived payloads as though
permission were established; only hashes and bounded derived evidence are
reported to the coordinator.

## Inspected dependency and license records

Records below were inspected in the installed wheel, not inferred from the
package name:

| Component | Record found |
|---|---|
| Python/OCP binding `LICENSE` and metadata classifier | Apache-2.0 |
| Bundled `occt` notice | LGPL-2.1-only |
| Bundled `ocp` notice | Apache-2.0 |
| Bundled `vtk` and `vtk-base` notices | BSD-3-Clause |
| `LICENSES_bundled` | 22 conda-package license notice sections |

The bundled notices additionally record fontconfig, freeimage, freetype,
imath, jxrlib, lcms2, lerc, libdeflate, libgomp, libjpeg-turbo, libpng,
libraw, libtiff, libwebp-base, openexr, openjpeg, xz and zstd, with their
respective terms. Other license text includes GPL/LGPL notices and
GCC-exception terms; do not summarize the whole wheel as Apache-licensed.
Retain the wheel's notices if redistributing its files. This inventory is a
record of the supplied metadata, not legal advice or a blanket relicensing.

On this host, the wheel contains an `OCP.cpython-312-x86_64-linux-gnu.so`
extension and 218 `.so*` entries under `cadquery_ocp.libs` (entries, not 218
distinct licensing packages). `readelf -d` on the extension shows DT_NEEDED
entries for bundled OCCT/VTK libraries and:

```text
RPATH [$ORIGIN/cadquery_ocp.libs]
```

For example, the hashed bundled `libTKernel` has a relative `cadquery_ocp.libs`
RPATH and DT_NEEDED entries for pthread, rt, stdc++, dl, m, gcc_s and c.
The successful isolated real STEP imports also demonstrate that this wheel's
native imports work in the checked environment. No unbundled GPU or legacy
CNC-Net TORCH/CUDA dependency is silently used by this route.

Useful local inspection commands (adapt paths to the active venv):

```bash
readelf -d external/cad-venv/lib/python3.12/site-packages/OCP.cpython-312-x86_64-linux-gnu.so
# Wheel-supplied notices:
less external/cad-venv/lib/python3.12/site-packages/cadquery_ocp-7.7.2.dist-info/LICENSES_bundled
```

## Pytest policy

Ordinary `tests/external_tasks_test.py` tests import no OCP and need no CAD
installation. The separately opt-in real-backend path requires
`DIFFCNCAD_REAL_TITAN=1`, the pinned interpreter, and the exact local source
hash. A missing/unqualified explicitly requested backend is blocked/failed,
not converted into a synthetic CAD pass or an analytic-target replacement.

To produce fresh local pilot artifacts (an existing output path is refused):

```bash
TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-diffcam-worker-3 \
UV_CACHE_DIR=/tmp/diffcam-uv-diffcam-worker-3 \
uv run python -m benchmarks.titan_setups --output data/external/titan-m1-pilot
```
