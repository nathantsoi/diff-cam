# Phase 2 Baseline Record

Phase 2 was executed with no source-code or evaluation-metric changes on branch
`ethan-direct-freeform-feedback` at commit
`6b7aa285643e8205d2da7f79afd3f6403b21e76a`. W&B was explicitly disabled by
the pipeline (`--no-track`); no LLM or external paid API was invoked.

## Environment

- Host: `robovision.csres.utexas.edu`
- GPU: physical GPU 7, Quadro RTX 6000 24 GB
- Python: 3.12.3
- PyTorch: 2.10.0+cu128
- Taichi: 1.7.4
- Seed: 1

## Smoke test

Command:

```bash
CUDA_VISIBLE_DEVICES=7 uv run python scripts/run_pipeline.py --stages train --iters 2 --max-steps 32 --stock-size-in 1 1 1 --voxel-size-mm 1.0 --target-shape sphere --target-radius-mm 11.43 --eval-freq 1 --seed 1 --runs-subdir phase2-smoke
```

Run: `runs/phase2-smoke/CamEnvDiff-v0__train_csg__1__1788836573868`

Result: exit 0 in 24.2 seconds. CUDA initialization, forward/backward
optimization, hard evaluation, best-checkpoint selection, STL export,
trajectory saving, and metrics saving all succeeded. Headline hard Dice was
0.565112. This two-iteration coarse-grid run is a plumbing check, not research
evidence.

## Unchanged baseline

Command:

```bash
CUDA_VISIBLE_DEVICES=7 uv run python scripts/run_pipeline.py --stages train --iters 5000 --max-steps 128 --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere --target-radius-mm 11.43 --post haas --eval-freq 10 --seed 1 --runs-subdir phase2-baseline
```

Run: `runs/phase2-baseline/CamEnvDiff-v0__train_csg__1__1788836645823`

The run exited 0 in 2433.5 seconds; reported optimization time was 2426.27
seconds. The composite selector restored the checkpoint from iteration 4280.
The final iteration's hard Dice was 0.592550, while the selected checkpoint's
hard Dice was 0.643764.

| Metric | Selected baseline |
|---|---:|
| hard Dice | 0.643764 |
| soft Dice | 0.759218 |
| uncut-stock Dice baseline | 0.547977 |
| normalized Dice improvement | 0.211907 |
| ASD | 1.819887 |
| HD95 | 5.338539 |
| residual | 6631.875 |
| gouge | 139.375 |
| holder overlap | 0.0 |
| air-cut fraction | 0.226508 |
| air time | 4.226943 s |
| total time | 5.092358 s |
| break probability | 0.000084 |
| best composite score | 0.635670 |

The unchanged trainer read existing local feedback and included a digest in
its log/metrics, but `feedback_used` is false and no feedback warm start was
applied. The pairwise reader also identifies its output as a steering signal,
not part of the loss.

## Visualization verification

Command:

```bash
CUDA_VISIBLE_DEVICES=7 uv run python scripts/run_pipeline.py --stages viz --run-dir runs/phase2-baseline/CamEnvDiff-v0__train_csg__1__1788836645823
```

This exited 0 in 35.0 seconds and wrote `trajectory_viz_haas.png`. The visualizer
reported:

- 128 optimized trajectory points
- 132 Haas waypoints after approach/retract insertion
- 9,251 interpolated execution samples and 91.83 s of postprocessed motion
- G-code-vs-simulator hard-carve Dice: 1.00000
- carved-vs-target hard Dice: 0.64376
- identical simulated/G-code carved-voxel counts: 102,001

## Artifact manifest

All paths are relative to the repository root. Raw artifacts remain ignored by
Git; this record is the small tracked manifest required by the handoff.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `runs/phase2-baseline/CamEnvDiff-v0__train_csg__1__1788836645823/args.json` | 2,543 | `9e1a1614129b7a31a2d3a1be0014fe74a29f8b1627536c9150f1962589a207d7` |
| `runs/phase2-baseline/CamEnvDiff-v0__train_csg__1__1788836645823/metrics.json` | 6,126 | `dac00996eb08179b2921a4af2530b2d731225bfcc829fcfe18ede4d8f028fe7b` |
| `runs/phase2-baseline/CamEnvDiff-v0__train_csg__1__1788836645823/trajectory.npy` | 1,664 | `7523a0afe1381e8b6ded20d600034bc053f131b34ec5ac3ec970315762db6f41` |
| `runs/phase2-baseline/CamEnvDiff-v0__train_csg__1__1788836645823/trajectory_deltas.npy` | 1,664 | `5a121ac48dfe229bad875b9e172f04358c6bf7fcefc3dcaa0e66ca6a62a7d62d` |
| `runs/phase2-baseline/CamEnvDiff-v0__train_csg__1__1788836645823/trajectory_viz_haas.png` | 1,297,093 | `a413daa29661b6cd6fb71cb77918cd89d41960df867f11c535d7b64690975755` |
| `runs/phase2-baseline/CamEnvDiff-v0__train_csg__1__1788836645823/reproduce_command.sh` | 1,260 | `505a9c14f133aa68e6ae0a2e5bdbbb85bdea93836b01abbc114814b7ea5caf9f` |

## Phase 2 disposition

Phase 2 passes: the existing environment runs the unchanged code on CUDA and
produces metrics, a saved trajectory, STL meshes, a reproducible visualization,
and a machine-readable run configuration. This establishes the control point
for the direct free-form-feedback implementation and experiments.
