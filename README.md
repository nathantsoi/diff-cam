# DiffCAM differentiable core

Taichi material-removal kernels, delta-trajectory optimization, hard-carve
geometry metrics, and a G-code round trip. This branch distills `autoresearch`
at `41b7355f05a6315bf2af47ee478501e519ca9232` into a small research core.

## Run

Python 3.10–3.12 is supported. Install with `uv sync`, then run:

```bash
uv run python -m algorithms.optimize --cpu --iters 20 --steps 16 \
  --resolution 16 --output runs/core-demo
uv run python -m pytest -q
```

Use a new output directory for each run. The random initialization is a smoke
example. To refine an existing CAM seed, add `--initial seed.npy`: a finite
`(N, 3)` array of tool-tip positions normalized by the one-inch stock box.
The first position is fixed. Internal physical dimensions are millimetres;
G54 is at stock top-centre. The minimal runner uses a 3.175 mm radius cutter,
25 mm cutter height, and explicitly recorded analytic target dimensions.

The runner uses NumPy Adam, gradient clipping, linear geometry-sharpness
annealing, and best hard-Dice selection. It writes configuration, score history,
selected metrics, parameter deltas, executed hard-path positions, and RS274/Haas
NC files. Hard selection is geometric, not a safety certificate. Training and
hard execution can follow different paths because speed regimes depend on stock.
The emitted NC still requires independent swept-volume and machine validation.

## Layout

- `simulator/`: analytic primitives, dense stock history, differentiable and hard
  removal, optional penalties, saved-state APIs, and geometry metrics.
- `algorithms/optimize.py`: minimal optimization and export entry point.
- `cam/`: units, work coordinates, posts, parser, exact-stop planner, path metrics.
- `tests/`: CPU gradient and hard-carve tests plus CAM round-trip regressions.
- `report/status-and-rss-plan.tex`: implementation audit, evidence limitations,
  related work, research hypotheses, experiments, and RSS publication gates.
- `docs/cleanup-manifest.txt`: removed-file inventory and archive location.

Build the report with `make -C report` (requires `pdflatex`); the PDF goes to
`report/build/status-and-rss-plan.pdf`. Validation details are in
`report/validation.txt`.

## Scientific status

Analytic tool/target fields are sampled on a dense, time-indexed stock grid.
This is not a voxel-free or arbitrary-precision implementation. Known priorities
are vertical/sloped sweep correctness, holder checking against chronological
remaining stock, and independent posted-NC evaluation. The optional force and
breakage proxies are uncalibrated. No physical validation is claimed here.

The report recommends studying the relaxation-to-execution gap under structured
CAM initialization, with matched hard-search baselines and resolution/segmentation
ablations. Historical contour heuristics and results remain on `autoresearch`;
this minimal runner is not a reproduction of that experiment harness.

RL, dashboards, preference collection, cluster provisioning, GUI rendering and
obsolete experiments have been removed from this branch. The complete local
experiment archive is preserved at
`../backups/diffcam-core-2026-11-08/autoresearch/`. Local run data, credentials and
nested paper repositories are preserved. The old dependency lock was removed
because it describes the discarded stack; `uv sync` generates a new lock.
