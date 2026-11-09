# CNC-Net modernized-stack native smoke

**Label:** `CNC-Net @ba0652f on modernized stack; one-step native smoke; NOT faithful reproduction`.

This is a persisted recipe for the bounded native-reconstruction probe verified in T5.0/T5.0b. It is **not** the original Python-3.8/Torch-1.12 stack, a full training reproduction, a published pretrained checkpoint, a model-quality benchmark, or a fixed-tool 3-axis NC implementation. The original stack remains **blocked** on the RTX 5070 Ti (`sm_120`).

## Files and ownership of changes

| File | Purpose |
|---|---|
| `requirements.txt` | Every package version from the verified native environment, including direct official Torch **2.9.1+cu128 cp311/Linux x86_64 wheel**. PyTorch3D is deliberately separate below. These are version/URL pins, not a claim of hashes for every wheel. |
| `run_smoke.sh` | Acquire only the exact `configs/references.json` CNC-Net pin, fetch the separately declared exact PyTorch3D commit if absent, build in a **new** venv, run and inspect artifacts. Never switch branches/core or update the workspace lock. |
| `run_modernized_smoke.py` | Verify source/sample identity, execute native `run.py` with disclosed bounded patches, and retain the source untouched. Converts native parameters into nothing different; there is no NC exporter. |
| `overrides/pymesh/__init__.py` | Small Open3D-backed subset of legacy PyMesh `form_mesh`, `save_mesh`, optional `separate_mesh`. It does **not** implement model, loss, KNN, target, rotation or CSG operations and is not full-PyMesh equivalence. |
| `inspect_native_result.py` | Hash/read meshes, parse native IoU from its log, compare with `reference_smoke.json`, and require a complete valid smoke. Missing/incomplete evidence returns nonzero. |
| `reference_smoke.json` | Earlier verified one-step native inventory: native IoU, file hashes and counts. Comparison uses these recorded values; it does not require the old `/tmp` files to remain available. |

Tracked text only lives here. Environments, clones, native checkpoint/mesh outputs and logs belong under git-ignored `external/` or an explicit user-chosen outside-core path. Nothing is added to core `pyproject.toml`/`uv.lock`. CNC-Net itself remains at `ba0652f9941686058c07c98617b9748e72f5f019`, with its upstream **missing/unclear license**; don't treat this recipe as permission to redistribute it or its sample.

## Required host and pins

- Linux x86_64; CPython **3.11.13** managed through `uv`; shared **RTX 5070 Ti / sm_120**. No CPU fallback is acceptable for this condition.
- Actual `uv`, `git`, `timeout`, and the C/C++ compilers at `/usr/bin/gcc`, `/usr/bin/g++`.
- CUDA compiler toolkit at `/usr/local/cuda-12.9` (override `CUDA_HOME` only deliberately). Original verified build used **nvcc 12.9.41** and driver 610.57.04. Torch runtime itself is **CUDA 12.8**, which is intentionally recorded, not conflated with the compiler version.
- `sm_120`-supporting `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS`/`CMAKE_BUILD_PARALLEL_LEVEL` = `CPU_THREADS` (default 4). Increase parallelism only with regard to other users; keep the native GPU invocation short.
- CNC-Net reference: `ba0652f9941686058c07c98617b9748e72f5f019`, with bundled `samples/0.off` SHA-256 `8e80d539b1c593501a55bd9f7b3c6b9099ce57fdb556e298b0bd3b744f690b7c`.
- PyTorch3D **v0.7.9** source: exact `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`, repository `https://github.com/facebookresearch/pytorch3d.git`. No branch or moving tag is fetched.
- Full package tour is in `requirements.txt`. No PyTorch3D wheel is downloaded: it is built from the pinned source by the command below.

The script's only reference acquisition is the canonical helper:

```bash
uv run --locked python -I scripts/pull_references.py \
  --subset cnc-net --root "$PWD" --manifest "$PWD/configs/references.json"
```

Its source-build phase is equivalent to:

```bash
CUDA_HOME=/usr/local/cuda-12.9 CUB_HOME=/usr/local/cuda-12.9/include \
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 \
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
uv --no-config pip install --python "$VENV_DIR/bin/python" \
  --no-build-isolation --no-deps "$PWD/external/pytorch3d-0.7.9"
```

The script checks source revisions before executing and preserves any dirty source tree by refusing to continue. If CUDA or the exact required source is missing, the result is **blocked**, not a substituted result.

## Run once in a fresh environment

From the repository root, using new paths that **must not already exist**:

```bash
TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-diffcam-worker-3 \
UV_CACHE_DIR=/tmp/diffcam-uv-diffcam-worker-3 \
baselines/cncnet/run_smoke.sh \
  external/cncnet-smoke-venvs/proof-1 \
  external/cncnet-smoke-runs/proof-1 \
  0
```

Use a different suffix for every reproduction so earlier evidence is never overwritten. The optional third argument is a nonnegative CUDA device index, default `0`.

The script:

1. Acquires/verifies only the pinned CNC-Net source and bundled OFF input from `configs/references.json`.
2. Creates the specified fresh venv (`--python 3.11.13`) and installs `requirements.txt` exactly, outside core `uv.lock`.
3. Fetches **only the exact PyTorch3D commit** if necessary and builds it with the CUDA configuration above.
4. Runs a maximum-480-second bounded smoke, normally approximately 23 seconds of driver/native run time on the observed host (not a promised timing budget or comparable benchmark measurement). Uses `PYTHONDONTWRITEBYTECODE=1` and zero data workers.
5. Independently checks the actual log and both raw/smooth OBJ meshes, hashes them and compares them to the recorded earlier run. A nonzero condition indicates missing/incomplete artifacts rather than silently passing.
6. Writes `${RUN_DIR}.freeze.txt`, `${RUN_DIR}.stdout.log`, and `${RUN_DIR}/artifact-inventory.json`, plus `smoke-status.json` and the actual native outputs inside the new run dir.

The working directory links `configs/` and `samples/` back to the **read-only tracked source checkout**, so `run.py` resolves its local resources while writing `checkpoints/` and `output/` under the fresh run directory. It neither overwrites an existing environment/run nor writes new metrics into shared source.

## Exact compatibility/budget changes

1. Dataset length and numeric epoch cap are each forced to **1**, so **one optimizer update** is performed; the normal 100-epoch/1000-items-per-shape run is not performed.
2. The original `Model`, `IOU`, `create_mesh_mc`, network, optimizer, target and loss are used. Source `git diff` must remain empty; the disclosed driver changes are outside source and recorded separately.
3. High-resolution evaluation wraps the original CSG to process independent **point blocks** (at most 1,048,576), then concatenates in order. It does not split paths or times or change the CSG equations. The high-resolution broadcast otherwise requires far more than the 16 GB GPU. Before the native run, a separate 2003-point/257-block CPU test compares both returned tensors and requires **bit-identical** results. This does not prove full-model equivalence to an unmodified high-memory run.
4. Legacy mesh I/O uses the above small Open3D shim. Files are **not promised byte-identical** to full-PyMesh output. The comparison explicitly records whether raw/smooth hashes match the earlier verified result, not just whether meshes exist.
5. Native `run.py` calls `torch.manual_seed(0)`, but cross-build/GPU nondeterminism remains possible. Do not silently change either IoU or hashes. Read the comparison section's delta/match fields and record the actual result. Differences may arise from nondeterminism or changed environment details; their cause is not inferred by the checker.

## What success means and does not mean

A valid smoke requires exactly one finite native IoU in `[0,1]`, nonempty finite raw and smoothed mesh output read back independently, and the actual native log/artifact files. The earlier one-step reference produced **64³ IoU = 0.423761** in **22.4588627771 seconds**, raw and smoothed meshes each with **455,374 vertices/910,748 triangles**. It is reproduced for comparison in `reference_smoke.json`; these are not a quality threshold or held-out result.

There is no native ordered waypoint/NC export. CNC-Net's learned operation tensors include radius choices and rotations, and their fixed-tool +Z feasibility remains a separate T5.1 decision. Any future comparison must preserve distinct pretraining/refinement budgets, units/frame conversion, and chronological safety checks. The one-step generated checkpoint is **not** a published pretrained checkpoint or a quality-selected trained baseline.

For the original failed legacy GPU/kernel evidence, matrix, artifact digests, and downstream T5.1/T5.2 restrictions, see `reports/external_compatibility.md` and `configs/external_compatibility.json`.

## Reproduction from this tracked recipe (T5.0b)

A new venv and new runtime directory were built with this exact tracked script, rather than reusing the original `/tmp` interpreter or any prior native outputs. The recipe/build/native-run/inspection pipeline returned **exit 0**, and the source `git diff --exit-code` also returned 0.

Command actually executed:

```bash
TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-diffcam-worker-3 \
UV_CACHE_DIR=/tmp/diffcam-uv-diffcam-worker-3 CPU_THREADS=4 \
baselines/cncnet/run_smoke.sh \
  external/cncnet-tracked-venvs/proof-1 \
  external/cncnet-tracked-runs/proof-1 0
```

| Measure | Earlier `/tmp` smoke | Fresh tracked-recipe proof |
|---|---:|---:|
| Native 64³ IoU | 0.423761 | **0.423786** |
| Absolute IoU difference | — | **0.000025** |
| Driver/native elapsed seconds | 22.4588627771 | **24.7514181710** |
| Mesh vertices/triangles (each of raw/smooth) | 455,374 / 910,748 | **455,284 / 910,580** |

Both fresh meshes read non-empty with all vertices finite. Raw/smooth file sizes are 34,673,515 and 34,679,974 bytes. Artifact hashes from the fresh run are:

- raw OBJ: `210cd06d08e34f1cca8c32fc4a6a934e9bcd0b92f511fb99e4be33efdcb878c7`
- smooth OBJ: `9a9fdc7af30787644789e3d376f864bec527f36249429a7703f9ba88ebaf1bc7`
- one-update checkpoint: `663320cbe1626abe7e8190b7688b4b62e3a78ec1e13cd7218a699e5f31e76397`
- native log: `794b5d645fbdad9a83c7663170ece734c30c39a98e751b553785dc24e6830ae5`

The new IoU differs slightly and **all four hashes differ** from the previous run: this is recorded explicitly, not presented as deterministic byte identity. GPU nondeterminism and the changed runtime/build paths are possible influences; the exact cause of every difference was not isolated. Mesh topology/count differences mean that changing only an experiment name cannot explain all raw-mesh hash differences. The one-step score is still not a quality or training-performance comparison.

Independent result: `external/cncnet-tracked-runs/proof-1/artifact-inventory.json`, SHA-256 `c34096d2c657232b363a5c55c8bc441a0a214b4b184a88d487e7772f9b72ea40`, with `valid_complete_smoke=true`. Its comparison records `iou_absolute_delta=0.000025` and `all_matched_artifacts_byte_identical=false`. Native stdout metadata is in `proof-1.stdout.log`, SHA-256 `5272d521ad70900f27d6103a5de1034b075fd3aa4df518fa05e91be8b6f3cb72`. The new interpreter's full `pip freeze` is `proof-1.freeze.txt` (SHA-256 `a5b8acd5b55be10427096b8fdf139972e496fa0c2820a49de96ab7195186cac6`); its source-path entry for the separately built PyTorch3D naturally changes between recipe locations.
