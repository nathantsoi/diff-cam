# External compatibility matrix (T5.0)

**Input DiffCAM revision:** `f4d0a644c22496720d19d9c49703f932d848d3e6`
**Reference acquisition manifest:** [`configs/references.json`](../configs/references.json)
**Machine-readable evidence and reproduction commands:** [`configs/external_compatibility.json`](../configs/external_compatibility.json)
**Earlier read-only survey:** `dispatch/diffcam/T5.0-recon-diffcam-worker-3.reply.md` (not a performance result).

## Status at a glance

This is a compatibility and provenance assessment, not a solver comparison. A source checkout, schema inspection, or stored QA pass is not a successful reconstruction or executable-task qualification.

| Resource | Native / geometry route | Executable fixed-single-tool +Z NC route | Current disposition |
|---|---|---|---|
| CNC-Net @ `ba0652f9941686058c07c98617b9748e72f5f019` | 64³ occupancy / negative-voxel IoU plus 256³ marching-cubes mesh from bundled OFF input. Original-era stack blocked on this GPU. **Modernized one-step smoke passed with native log and mesh artifacts; it is not a faithful reproduction.** | No native operation-path or NC output. Radius and effective orientation vary by operation. | Original stack **BLOCKED**; executable mapping **unsupported natively**. Do not equate a future modernized smoke with faithful reproduction or a complete trained baseline. |
| CAMotics source @ `e84665f2fa9d1151f03282ac7e01320bc65e015b`; binary 1.2.0 | Qualified external `camsim` Docker route exists. It independently evaluates posted G-code, not a candidate optimizer. | Useful to check an already-posted program after its own step/arc/rapid semantics are understood. | Binary Docker route **qualified by coordinator**; source build remains **blocked/not attempted**. This card did not repeat T1.3's verifier cross-check. |
| Local Titan M1–M10 @ input revision | STEP B-reps are present. M1 is inch-unit AP214 from Autodesk Translation Framework. | Requires an explicitly reachable +Z operation, not whole-part machining by assumption. | **BLOCKED:** no real STEP backend, ready-made single-tool setup, external-source provenance or license was found. No import or nominal NC was run. |
| CNCGEN preview @ `995e1944609248ed8d9beda5178cf785a15c8392` | Verified 20-sample schema, Step/BREP, final voxels and recorded reference paths. | Conditional adapter only after original target, actual near-net stock, operation exclusions, tool/holder feasibility and posted checks are settled. | Payload **verified** for descriptive schema use. Full benchmark reproduction and model-output comparison **unsupported**. No record yet qualifies as a validated executable task. |

## CNC-Net: actual input, actions and output semantics

Evidence paths below are in `external/cnc-net` at `ba0652f9941686058c07c98617b9748e72f5f019`, acquired with the T0.1 helper; no source edits were made there.

### Input representations, not just 64³

- `utils/dataloader.py:23-99` reads a triangle mesh with Open3D, sets `N=64`, generates a **64³ low-resolution** occupancy/point set with `pysdf.SDF.contains`, and additionally allocates **256³ high-resolution points**. `utils/eval_metric.py:44-60` scores negative-voxel intersection over union on the low state; the outside-region term is commented out.
- Each grid point is mapped separately along each axis from the corresponding mesh minimum to maximum. This is anisotropic extent scaling rather than an isotropic normalized-to-mm unit convention. `dimension=max_corner-min_corner` is returned; no millimetre declaration is present. A physical adapter must establish this scaling and the mesh-frame/G54 transforms.
- `networks/model.py:237,251-253` feeds 64³ residual occupancy to the encoder. `run.py:249-252,287-299` separately allocates the 256³ high-resolution state used for output. Both resolutions therefore matter; checking an import alone is not native scoring coverage.
- `utils/cad_meshing.py:19-99` exports raw and smoothed marching-cubes OBJ using mesh extent, but does not use the mesh minimum corner. Preserve and independently check that origin limitation rather than silently assuming physical alignment.

### Operation/action compatibility

- `networks/model.py:246-326` predicts up to **20 mill operations**, each with 100 XY samples for the low path, 500 for the high path, a single shared Z and a Gumbel-softmax radius from **0.025, 0.05, 0.075, 0.1** in object units.
- `networks/model.py:344-432` has a separate conditional drilling loop (`best_iou > 0.92`) capped at 20 single-XYZ operations with radii **0.01, 0.02, 0.03, 0.04**. It is not a fixed-length or fixed-tool program.
- Both paths predict two axis-angle components with the third zero (`model.py:269-299,365-398`). `networks/primitives.py:16-31` applies an XY cylinder and Z bound after rotating the points. The effective tool axis can therefore tilt; it is not a declared fixed-+Z three-axis action.
- Although operation XY, Z, radius and rotation tensors are assembled, `Model.forward` returns only loss/current fields. `run.py:296-299` exports meshes. There is no native ordered waypoint file, feed/rapid timing, holder model, G54 convention, retract policy or NC post. Converting a reconstructed mesh back into a path would not be reproducing the learned operation sequence.

### Native training/checkpoint evidence

The documented command is `python run.py --input_object '0.off' --gpu 0 --experiment exp_0` (`README.md:21-38`). At this revision it constructs a new model, uses `configs/specs.json` (100 epochs, step LR, nominal snapshots), and the dataset reports length 1000 for the same object (`run.py:161-179`; `utils/dataloader.py:92-99`). There is no supplied multi-shape pretraining loader in the runnable path; setting `--input_object all` makes the object `None` even though dataset construction is conditional on a non-`None` object. Do not describe the per-instance command as a published pretraining recipe.

No `.pth`, `.pt` or checkpoint is tracked. During recon, `git ls-remote --heads --tags` showed only `main` and no tags and the official GitHub `releases` endpoint returned `[]`. This supports **no weights found in the official source/release locations inspected**, not a claim that no author has any weights anywhere.

No `LICENSE`, `COPYING`, or `NOTICE` is present and the GitHub repository API returned `"license": null` on 2026-11-08. Code/sample redistribution permission is consequently **missing/unclear** (`configs/references.json` agrees). Public visibility is not a license to redistribute.

### Original pinned-stack native smoke: **BLOCKED**

The exact pinned OFF input was verified at SHA-256 `8e80d539b1c593501a55bd9f7b3c6b9099ce57fdb556e298b0bd3b744f690b7c` (1,020,767 bytes). The external checkout remains unmodified. A separate shared clone was used for any native invocation.

1. `uv run scripts/pull_references.sh --subset cnc-net,camotics`: **exit 0**, exact CNC-Net and CAMotics revisions and sample digest verified.
2. `uv venv --python 3.8 /tmp/diffcam-worker-3-t50.gKY5mr/venv-py38`: **exit 0**, CPython **3.8.20**, deliberately outside core `uv.lock`. This differs slightly from the README's Python 3.8.18 and is recorded, not hidden.
3. The first `uv pip install ... --index-url https://download.pytorch.org/whl/cu113 ... 'torch==1.12.1+cu113' ...`: **exit 1**, index-selection resolution failure; no native run followed from that failed step.
4. An exact official-wheel install using `torch @ https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl`: **exit 0**, installed `torch==1.12.1+cu113`, `numpy==1.23.5`, and `typing-extensions==4.13.2`. (The initially guessed `manylinux1_x86_64` URL was wrong and returned HTTP 403; the official index's actual `linux_x86_64` filename was used instead.)
5. The decisive short GPU allocation, corresponding to the runner's `current_high` tensor, was:

   ```bash
   /tmp/diffcam-worker-3-t50.gKY5mr/venv-py38/bin/python -I -c \
     'import torch; print(torch.cuda.get_arch_list()); x=torch.ones(1,256*256*256,device="cuda").float()'
   ```

   **Exit 1.** The host is an NVIDIA GeForce RTX 5070 Ti with device capability **(12,0)**. `torch.cuda.is_available()` returned true, but `get_arch_list()` was `sm_37, sm_50, sm_60, sm_70, sm_75, sm_80, sm_86`. Torch emitted `sm_120 is not compatible` and the tensor allocation raised **`RuntimeError: CUDA error: no kernel image is available for execution on the device`**. Compiled CUDA was 11.3.
6. A broader Python-3.8 dependency batch including `open3d==0.16.0`, `pysdf==0.1.9` and `PyMCubes==0.1.6` was attempted: **exit 1** before installing that batch. Its build resolver required `numpy~=2.0`, which requires newer Python than 3.8. Separately installed small import dependencies were `trimesh==3.23.5`, `h5py==3.10.0`, `tqdm==4.67.1`, `plyfile==0.9` (**exit 0**).
7. In the pinned, isolated source clone, invocation with `python -I run.py --input_object '0.off' --gpu 0 --experiment t50-native-smoke` returned **exit 1** at `import utils`; isolated mode removes the source directory from import lookup and thus did not reach the model. The required normal invocation was then run without `-I` under `timeout 90`:

   ```bash
   cd /tmp/diffcam-worker-3-t50.gKY5mr/runtime/cnc-net
   timeout 90 /tmp/diffcam-worker-3-t50.gKY5mr/venv-py38/bin/python \
     run.py --input_object '0.off' --gpu 0 --experiment t50-native-smoke
   ```

   It reached `utils/eval_metric.py:5` and failed with **`ModuleNotFoundError: No module named 'pytorch3d'`**, **exit 1**, before OFF loading or model allocation. The older PyTorch3D 0.6.2 PyPI release examined had macOS wheels only; a usable Linux/Python-3.8/CUDA-11.3 binary was not obtained. No source model or loss was substituted to bypass this.

**Original-stack result:** no native 64³ IoU, mesh, checkpoint or training score was obtained. A sample checksum is not an output hash. These failures are recorded instead: `/tmp/diffcam-worker-3-t50.gKY5mr/logs/uv-torch-install.log`, `uv-torch-direct-wheel-install.log`, `cuda-allocation-probe.log`, `uv-other-native-deps.log`, `uv-lightweight-native-deps.log`, `native-entrypoint.log`, and `native-entrypoint-no-isolation.log`.

### Modernized-stack bounded attempt

Started `2026-11-08T19:10:17-06:00`; authorized hard stop `2026-11-08T20:40:00-06:00`. The outcome must be called **“CNC-Net @ba0652f on modernized stack”**, not faithful reproduction.

**Outcome: native one-step smoke passed on the disclosed modernized condition; it is not a faithful reproduction or a completed trained baseline.** The actual native invocation returned at `2026-11-09T01:23:41Z`, well within the time box, after 22.458863 seconds (including preprocessing, one training/evaluation pass and mesh export). This does not erase the independently observed original-stack blocker above.

#### Environment and extension evidence

An isolated CPython 3.11.13 environment at `/tmp/diffcam-worker-3-t50.gKY5mr/modern/venv` was created with `uv --no-config venv --python 3.11`, **not** by changing core `pyproject.toml` or `uv.lock`. The exact package setup was a direct `torch @ https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl` install plus NumPy 2.3.4, Open3D 0.19.0, PySDF 0.1.9, PyMCubes 0.1.6, trimesh 5.2.0, h5py 3.16.0, plyfile 1.1.5 and their dependencies. `uv pip freeze --python ...` is retained at `/tmp/diffcam-worker-3-t50.gKY5mr/modern-pip-freeze.txt`, SHA-256 `767c25a6ca520387323df3aa8cb4ed2e5df93f1bb3d0336322833daff2f5b896`.

The short GPU/dependency probe (**exit 0**) showed `torch==2.9.1+cu128`, Torch CUDA 12.8, arch list `sm_70,75,80,86,90,100,120` and successful allocation/sum of the original `1×256³` float32 tensor (`16777216.0`). Open3D, PySDF, mcubes, trimesh, h5py and plyfile all imported. This is an availability check, not a CNC-Net result.

PyTorch3D **v0.7.9**, exact revision `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`, was then shallow-cloned and source-built instead of replacing the library's operations. The build used local **nvcc 12.9.41**, `CUDA_HOME=/usr/local/cuda-12.9`, `CUB_HOME=/usr/local/cuda-12.9/include`, `FORCE_CUDA=1`, `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS=4`, `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, and `uv --no-config pip install --no-build-isolation --no-deps -v /tmp/diffcam-worker-3-t50.gKY5mr/modern/pytorch3d-src` in the isolated interpreter. Build **exit 0**; compiler is intentionally recorded as different from Torch's CUDA 12.8 runtime. The compiled `_C...so` hash is `6d377cdc8d344e152aee4ca17a6f731043f7d4b47ad0d0b1851028227ef450b0`; `cuobjdump --list-elf` showed `sm_120` cubins.

CNC-Net's used symbols were checked by source grep: `knn_gather`, `knn_points`, `Transform3d`, and `axis_angle_to_matrix`. A direct probe of the built extension (**exit 0**) ran `knn_points(x,x,K=1)` with the expected self-neighbor indices, and `Transform3d().cuda().rotate(axis_angle_to_matrix(...)).transform_points(x)` returned the expected finite tensor shape. No KNN or rotation substitute was introduced.

#### Exact compatibility/budget patches

All patches are outside the pinned source in `/tmp/diffcam-worker-3-t50.gKY5mr/modern/overrides`; `git diff --exit-code` in the runtime clone returned 0 after the run, and the reference external checkout remained clean. Native model/loss/target/path code was not changed.

1. The separate driver `run_cnc_net_smoke.py` (SHA-256 `fdbb9d170eac4b03c1dab7616a1b7883614a33a923ad75509d97809cec894ac5`) invokes CNC-Net's own `run.py` with its own `Model`, `IOU` and `create_mesh_mc` on bundled `samples/0.off`. It overrides dataset length and configured `NumEpochs` **each to 1**, yields **one optimizer update**, and sets `num_workers=0` to isolate data loading. This is a smoke budget, not the original 100-epoch × 1000-samples/same-object procedure.
2. Because a full high-resolution `500×256³×2` broadcast is far larger than the 256³ state, the driver wraps the native CSG into independent **point blocks of at most 1,048,576 points**, using the unchanged native CSG inside each block and concatenating in original order. It never blocks across time/path samples or operations. A separate small check **before** the model run compared native CSG with 257-point chunks on 2003 points and both returned tensors were **bit-identical**. This is evidence for the chunking mechanism, not a claim that large-GPU numerical equivalence was independently compared to an unmodified full-memory run.
3. Legacy PyMesh's available distribution was not suitable for the new interpreter. A small Open3D-backed subset `overrides/pymesh/__init__.py` (SHA-256 `19223c46a252cf5795105404c9e81957bb92937e15687fec177bf8cca2e95faa`) implements `form_mesh`, `save_mesh` and optional `separate_mesh`. It leaves marching-cubes coordinates and native geometry untouched but is **not** full PyMesh or a guarantee of identical serialization bytes. No model, loss, rotation or target operation was reimplemented.

The first driver start returned **exit 1** before native invocation with a `NameError` (`native_loader_init`) in this wrapper; it was corrected to the already-defined `native_torch_loader_init`. The first attempt's log is `modern-native-run.log` (SHA-256 `a0fe6952cea9d3dc5bd9e86bf35c45584f6738ca30a3a8792d520d0115cfccac`). This was explicitly a harness bug, not a CNC-Net/CUDA result. The successful bounded invocation was:

```bash
PYTHONPATH=/tmp/diffcam-worker-3-t50.gKY5mr/modern/overrides \
CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
timeout -k 10s 480s \
/tmp/diffcam-worker-3-t50.gKY5mr/modern/venv/bin/python \
/tmp/diffcam-worker-3-t50.gKY5mr/modern/overrides/run_cnc_net_smoke.py \
  --input_object '0.off' --gpu 0 --experiment t50-modern-smoke
```

**Successful-run exit 0.** Native `run.py` logged one shape, one completed optimizer update, `Average IOU: 0.423761` and `BEST IOU: 0.423761`, followed by mesh creation. It emitted a newer-Torch warning at `run.py:255` about `math.isnan` on a tensor requiring grad, then completed normally; the warning is retained rather than hidden. The stdout log is `/tmp/diffcam-worker-3-t50.gKY5mr/logs/modern-native-run-v2.log`, SHA-256 `48191c8edfb2ebe1f45799fa217a82a5d797fc1cb39a82cc2d565f36b6a2d573`.

#### Actual artifacts and independent read-back

| Artifact (root `/tmp/diffcam-worker-3-t50.gKY5mr/runtime/cnc-net/`) | Bytes | SHA-256 |
|---|---:|---|
| `output/t50-modern-smoke/0.obj` | 34,680,789 | `c2395ef6f83a22604989e7b2ed67e328100a732cc3fd6ac2b8c6b3fe00147e83` |
| `output/t50-modern-smoke/0_smooth.obj` | 34,687,452 | `c9787b150a2978183557ae597effd27c648af3f68768826f52c434cecdbad930` |
| `checkpoints/t50-modern-smoke/best_checkpoint.pth` | 139,691,301 | `dbe446656cd1dc4399fd330cd128fb4c722bf0a22c4ce0e5008150e4da535169` |
| `checkpoints/t50-modern-smoke/logfile.log` | 8,543 | `69dc4fff8d0a44ea72c67041c055d916b388973349fc2a467f0d5394f9955a73` |

A separate Open3D inspector (`inspect_native_result.py`, SHA-256 `f0103d8f784c21af594f8ed33ee0056f33ac917f876c3d7f6613b2f8137bc9c6`) reread both mesh files. Each has **455,374 vertices and 910,748 triangles**, all vertices finite. Raw and smooth OBJ bounds both read as `[-0.3637160063,-0.3147540092,-0.1049180031]` to `[0.3637160063,0.3147540092,0.1049180031]`; this does not assert physical-mm or original-mesh-frame accuracy. The native logged 64³ IoU value is 0.423761 as printed. Inspection returned **exit 0** with `valid_complete_smoke=true`; its JSON report is `/tmp/diffcam-worker-3-t50.gKY5mr/modern/evidence/native-artifacts.json`, SHA-256 `b40cf8d3e6a1d9f5370c9e4c2dce84afe7ad79dda603fc765138951353a972ff`.

The checkpoint is generated from the **one-run, one-update model**, not previously published pretrained weights and not a quality-selected checkpoint across a training campaign. No held-out performance, full-corpus fitting, legacy numerical equivalence, NC waypoint generation, fixed-tool motion mapping, independent machining evaluation or safety check is established. A post-run GPU process query showed only the pre-existing desktop-related processes, not an active CNC-Net compute job.

### T5.1 adaptation decision

Preserve a native geometry condition whenever a native run is actually available. Independent of that, restrict executable comparison **before** performance inspection to cases that pass all of these:

1. A declared mesh-unit/mm and frame transform is verified.
2. Effective cutter orientation is fixed +Z, one declared physical cutter is used, and actual path/stock/tool/holder semantics are recorded.
3. The ordered internal operation tensors are captured without modifying the native reconstruction objective.
4. Added links, lifts, retracts, feeds and operation sequencing are explicit and separately validated.
5. All boundaries, stock containment, clearance and posted motion pass the shared hard/NC evaluator.

Otherwise return `unsupported` or `blocked`, not a fabricated NC path or a nominal IoU passed off as executable geometry. Pretraining/checkpoint acquisition and one-instance fitting must have separate budgets. Whether such a faithful constrained subset exists remains **unresolved**, not a coverage claim.

## Titan: local source and STEP backend

`reports/experimental_plan.md` identifies Titan as the ten local files `parts/Titan-M1.step` through `Titan-M10.step`, not an external benchmark downloaded by T0.1. They were introduced in `a94cfde29cbda7a116125d890a1fd847c09050a4` (`added some reports, parts`). No external-origin URL, setup manifest or source/redistribution license is supplied with the parts or that change.

For the inspected `parts/Titan-M1.step` at the input revision:

- SHA-256: `3ba16d17de50526b452f844a7faf66a624ddd9509fcc3eac2087fdbdbc2066e3`.
- Header lines 11–20: `Titan-M1.step`, ST-Developer v20.2, Autodesk Translation Framework v15.17.0.0, AP214. Line 28 contains `MANIFOLD_SOLID_BREP('Body1',...)`.
- Unit context lines 2081–2113 specifies `CONVERSION_BASED_UNIT('inch',#2057)` and `LENGTH_MEASURE(25.4)` mm; the declared uncertainty is `0.000393700787401575` inch. Product and representation references appear at lines 2124–2136. **A mm adapter must apply and verify the inch-to-mm conversion.**

The T5.0 probe `uv run python -I -c '<find_spec/which probe>'` (exit 0) found no `OCP`, `OCC`, `cadquery`, or `trimesh` modules, no FreeCAD executable, and the project lists only NumPy/SciPy/Taichi at input `pyproject.toml`. Thus STEP import is **blocked/unavailable**, not tested. T5.2 must select and pin an actual OpenCascade/FreeCAD backend in isolation; a mesh library alone is not a STEP importer. Import must validate solid topology, a known dimension and bounds before any mesh/SDF sampling.

No ready-made Titan single-tool setup was found. A first adapter needs original target, predeclared reachable +Z operation target with excluded volume, actual initial stock, G54 top-centre transform, finite cutter/flute/holder geometry, accessibility checks, nominal CAM seed and independently verified NC. Mere B-rep presence does not establish any of these, so no Titan operation is claimed compatible yet.

## CNCGEN: verified preview, not CNCGEN model output

### Payload and integrity

The pinned Git revision observed in a scratch LFS-skipping clone is `995e1944609248ed8d9beda5178cf785a15c8392`. T0.1's current entry is still optional/not acquired with `revision: null`; promotion of the observed source/payload pin is a coordinator action, not silently performed by this file.

`CNCGEN_synthetic_preview_v0.1.zip` is a 132-byte LFS pointer before payload acquisition. The real pinned download is 3,618,025 bytes and SHA-256 `03b7abce467166b22e0c7e64a7bd6ba0b8059b7205010afb2f37634388d22c4e`, matching the pointer. `unzip -l` and an independently written read-only parser found **181 members**: `checksums.json` plus 20 samples with nine files each:

`design.json`, `manufacturing.json`, `toolpath.json`, `meta.json`, `validation_mds3.json`, `part.brep`, `part.stp`, `final_voxel.h5`, `pointcloud.ply`.

The sample IDs are `0000/00000000`, `0000/00000001`, `0000/00000002`, `0000/00000003`, `0000/00000004`, `0000/00000005`, `0000/00000006`, `0000/00000008`, `0000/00000009`, `0000/00000010`, `0000/00000011`, `0000/00000012`, `0000/00000013`, `0000/00000014`, `0000/00000015`, `0000/00000016`, `0000/00001647`, `0000/00002163`, `0000/00006345`, and `0004/00045852`. Prepend `samples/` to each sample ID to obtain its archive directory. All 180 declared file hashes and byte counts matched (the checksum manifest covers sample files, not itself). This establishes payload integrity only. The archive itself has no license file, so keep the repository notices with the payload.

### Meaningful fields and provenance

- Manufacturing records link feature IDs, ordered action targets and toolpath IDs. Features contain type/depth/bounds/geometries, suggested tools, axis/undercut fields, plus semantic status traces. A `status_trace` is **not** a dense intermediate stock sequence.
- Toolpaths contain `units="mm"`, `coord_system="WCS_TOP_ZPLUS"`, actual stock/preprocessing records, tool diameter/length/flutes/class, recorded conditions, canonical waypoint records and QA history. Observed waypoint field union is `x,y,z,mode,pass,layer_id,segment_id,op_id,feed,spindle`; modes are `RAPID`, `PLUNGE`, `CUT`, `RETRACT`. These are exports of recorded paths, not proof of complete physical timing/holder semantics.
- Every final voxel HDF5 has dataset `voxel`, bool `(128,128,128)`, with root attributes `bbox_min`, `bbox_max`, `stock_dims`, `pitch`, `units`, `design_id`, `n_features`. It is the final part only. Historical voxel feature counts can differ from the curated plan after an external contour is removed.
- The 20 non-random samples are `split="preview"`, not paper-aligned train/dev/test assignments. Counts are 55 objects, 55 operations/toolpaths and 370 waypoints: 27 Mill, 23 Drill and 5 Slant labels. All 55 selected features carry `min_axes="3X"`, but that is metadata, not a safe-trajectory certificate.
- All sample/source records say DeepMS; all paths say `external_explicit`. Collection counts are `Cyliner-data` (5), `Path-data` (6), `ir-data` (8), `new-data` (1). Original IDs are retained separately from merged IDs. They are **reference paths, not CNCGEN model predictions, optimal NC ground truth or benchmark scores**. No NC, model checkpoint or dense history is included.

Licenses at the pinned HuggingFace revision are explicit: preview data/manifest **CC BY 4.0** in `LICENSE-DATA.txt` and README, and the two Python utilities **MIT** in `LICENSE-TOOLS.txt`. `THIRD_PARTY_NOTICES.md` preserves DeepMS attribution and explains that upstream generator software's MIT license is not blanket dataset permission. Preserve all of these if redistributing within their terms.

### Provisional candidate, not an executable qualification

`samples/0000/00000016` is a useful first **schema-loader** example: one curated 28 mm-deep pocket, +Z/3X/no-undercut declarations, Ø10 × 50 mm endmill, eight waypoints, no retained-path QA warning and one removed external contour. It is still **not qualified for execution**: `initial_occupancy_mode=external_contour_prism` and `internal_only_near_net_stock=true` describe different geometry from the loose declared 103 × 100 × 61 mm box; three generation failures are recorded; it uses one depth layer and `is_expert_trajectory=false`; and voxel/plan feature counts need interpretation after curation.

Consequently an adapter must retain the original full target, explicitly select/reconstruct near-net initial stock, define the selected operation target and exclusions, and check all current chronological/hard and posted conditions. It must not silently replace near-net stock with the loose bounding box, nor hide the removed contour. For counterexamples, sample `0000/00000000` records 75 mm pocket depth with a 50 mm tool despite stored QA validity; `0000/00000011` has two large-jump warnings plus a removed contour. These are reasons to evaluate, not adopt, stored paths.

The reproducible JSON/HDF5 inspection evidence is `/tmp/diffcam-worker-3-recon.qcN4Z6/cncgen-inspection.json`, SHA-256 `a66924fdec45f5b9f2262e43c6894805b920fdc8f0543b74588a4da6a82b6f37`; no bundled CNCGEN utility was executed.

## CAMotics: independent verifier, distinct source/binary routes

The exact source pin `e84665f2fa9d1151f03282ac7e01320bc65e015b` was acquired successfully. It has `LICENSE` and `COPYING` with **GPL-2.0-or-later**. Its source reproduction remains **blocked/not attempted**: T0.1 recorded absent C!/cbang, Qt5, SCons and V8 prerequisites, no source `cbang` tree and no PATH binary.

The coordinator separately qualified the upstream **1.2.0 binary release** using [`docker/camotics/README.md`](../docker/camotics/README.md), not the missing source build. The binary package URL and expected SHA-256 are recorded in the JSON manifest. The pre-existing image inspected by this card is `diffcam-camotics:1.2.0`, image ID `sha256:e680bff023d9190ed16fe1bee21d9c3c72f0e35bfbceb0a19e704c62f6ddca80`. A documented invocation is:

```bash
docker run --rm -u $(id -u):$(id -g) -v $PWD/work:/work \
  diffcam-camotics:1.2.0 camsim --binary p.camotics out.stl
```

Project files provide units, manual resolution in mm, cylindrical tool diameter/length, workpiece bounds and input NC files; output STL is the cut workpiece. The coordinator's smoke used 30 × 30 × 10 mm stock, 20 mm slot, 6.35 mm flat cutter and 2 mm depth. Analytic remaining volume 8682.7 mm³ compared with mesh volumes 9277.1 at resolution 0.25 and 8761.0 at 0.1 mm. The output surface is computed over bounds padded by one cell; T1.3 must quantify that bias as well as arc/rapid behavior before using numeric comparisons. This card checked the route/provenance but did **not** rerun the binary or claim an independent T1.3 cross-check.

## Bounded downstream plan and unresolved questions

1. **T0.1/coordinator:** promote CNCGEN immutable source/payload revision `995e194` and its separate licenses before routine acquisition. Keep CNC-Net's missing license and no-published-checkpoint findings explicit. Any CAD or CNC-Net native dependency stays isolated from the core lock.
2. **T5.1:** decide whether a current-CUDA native condition can be maintained without changing model/loss semantics; separately gate a fixed-+Z/single-tool executable subset as defined above. If not, native reconstruction and executable exclusion are valid explicit results; neither is a passing performance comparison.
3. **T5.2:** qualify the STEP backend and one original/operation target pair. Preserve CNCGEN provenance, near-net stock and removed-contour semantics; do not treat reference QA as physical clearance.
4. **T1.3:** use the qualified Docker binary route for independent posted checks, retaining resolution/padding and coverage limitations. Do not wait for the blocked source build.

All detailed facts, machine-readable flags, revision/license evidence and exact smoke attempts are in `configs/external_compatibility.json`. No physical machining or held-out performance experiment was performed for T5.0.
