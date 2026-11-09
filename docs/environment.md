# Environment

## Core (locked)

`uv.lock` pins the core stack: Python 3.12.13 (requires-python `>=3.10,<3.13`), Taichi 1.7.4
(LLVM 15.0.4), NumPy 2.5.4, SciPy 1.18.1, pytest (dev group). Optional extra `optim` adds
`cma` 4.5.0 (pycma) for T3.1; its upstream source revision is pinned separately by T0.1.

```bash
touch .env   # the user shell exports UV_ENV_FILE=.env; .env is git-ignored, so new worktrees lack it
export UV_CACHE_DIR=/tmp/diffcam-uv-<agent> TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-<agent>
uv sync --locked                 # core + dev
uv sync --locked --extra optim   # + pycma
uv run python -m pytest -q
uv run python -m algorithms.optimize --cpu --iters 2 --steps 4 --resolution 10 --output <new-dir>
```

Validated 2026-11-08 on branch `task/T0.2-environment` with a fresh online lock:
`uv lock` (22 packages), `uv sync --locked --extra optim`, `pytest -q` → 20 passed, 1 Taichi
DeprecationWarning (exit 0); CPU optimizer smoke exit 0, wrote config/history/metrics/trajectory
arrays and both NC posts. The smoke is wiring only, not a geometry or safety result.

## Isolated environments (not in the core lock)

- **CNC-Net** (T5.0/T5.1): PyTorch stack in its own venv/process under `external/`; never added to
  this lock. Communicate through files (task/target grids in, reconstructions/actions out).
- **CAD / STEP** (T5.2): STEP-capable backend (e.g. OCP/pythonocc or FreeCAD) in a separate env;
  mesh-only tools are not a STEP importer.
- **CAMotics** (T1.3 cross-check): external binary; absence is recorded as *blocked*, never pass.
- **Policy learning** (T7.1): separate extra/env when that track starts.

Workers request dependency changes from the coordinator; one combined patch updates
`pyproject.toml` + `uv.lock`.

## Hardware fingerprint (shared box)

- CPU: AMD Ryzen 9 9950X, 32 threads; RAM 121 GiB (shared with other sessions).
- GPU: NVIDIA GeForce RTX 5070 Ti 16 GB, driver 610.57.04; kernel 7.0.0-42-generic.
- GPU backend: **not yet validated** for this lock. Any GPU claim needs a separate recorded check;
  timing runs are serialized on the single GPU.

## Taichi

`ti.init()` invalidates existing fields: initialise once per process (`init_taichi=False` for
additional simulators) or use separate processes. Use distinct offline-cache and output dirs per agent.
