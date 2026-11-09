"""Bounded CNC-Net native entrypoint on the disclosed modernized stack.

Label: CNC-Net @ba0652f on modernized stack, one-step smoke. This is NOT faithful
legacy reproduction, full training, or a machining/NC result. The pinned source
is never edited. Only the following out-of-source compatibility/budget changes:

* Cap dataset length and NumEpochs at 1 (one optimizer update).
* Use zero DataLoader workers to isolate CPU/CUDA state.
* Split independent point queries into <=1,048,576-point blocks around the
  unchanged native CSG, concatenating in their original order. No path samples,
  time steps, geometry equations or loss semantics are reimplemented.
* Use the adjacent Open3D mesh-I/O subset for legacy PyMesh imports.

A separate CPU fixture verifies that point blocking is bit-identical to native
CSG. It is not a proof of full-model floating-point or PyMesh-file equivalence.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REVISION = "ba0652f9941686058c07c98617b9748e72f5f019"
EXPECTED_SAMPLE_SHA256 = "8e80d539b1c593501a55bd9f7b3c6b9099ce57fdb556e298b0bd3b744f690b7c"
MAX_POINTS = 1_048_576


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "external/cnc-net")
    parser.add_argument("--source-revision", default=EXPECTED_REVISION, choices=[EXPECTED_REVISION])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def verify_source(source: Path, revision: str) -> None:
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if result.stdout.strip() != revision:
        raise RuntimeError(f"CNC-Net source must be exactly {revision}; found {result.stdout.strip()}")
    dirty = subprocess.run(["git", "-C", str(source), "diff", "--quiet"], check=False)
    if dirty.returncode:
        raise RuntimeError(f"Refusing a source with tracked modifications: {source}")
    sample = source / "samples/0.off"
    if sha256(sample.read_bytes()).hexdigest() != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError(f"Bundled sample digest mismatch: {sample}")


def main() -> None:
    args = parse_args()
    if args.gpu < 0:
        raise ValueError("--gpu must be a nonnegative device index")
    source = args.source.resolve(strict=True)
    work_dir = args.work_dir.expanduser().resolve(strict=False)
    verify_source(source, args.source_revision)
    # Never overwrite an existing experiment, its metadata or native outputs.
    work_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(work_dir)
    for name in ("configs", "samples"):
        os.symlink(source / name, work_dir / name, target_is_directory=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    import torch
    import utils.dataloader as native_dataloader
    import utils.workspace as native_workspace
    import networks.model as native_model
    from networks.primitives import CSG as native_csg
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(args.gpu) != (12, 0):
        raise RuntimeError("This disclosed bounded recipe targets an RTX 5070 Ti (sm_120); refusing a CPU/unknown-device fallback")

    native_specs = native_workspace.load_experiment_specifications
    native_torch_loader_init = torch.utils.data.DataLoader.__init__

    def chunked_csg(current, xyz, radius, points, max_points=MAX_POINTS):
        if points.shape[-2] <= max_points:
            return native_csg(current, xyz, radius, points)
        current_blocks, cylinder_blocks = [], []
        for start in range(0, points.shape[-2], max_points):
            stop = min(start + max_points, points.shape[-2])
            current_block, cylinder_block = native_csg(
                current[..., start:stop], xyz, radius, points[:, start:stop]
            )
            current_blocks.append(current_block)
            cylinder_blocks.append(cylinder_block)
        return torch.cat(current_blocks, dim=-1), torch.cat(cylinder_blocks, dim=-1)

    rng = torch.Generator().manual_seed(501)
    points = torch.rand((1, 2003, 3), dtype=torch.float32, generator=rng) - 0.5
    xy = torch.linspace(-0.2, 0.2, 100).repeat(2, 1).T.unsqueeze(0)
    z = torch.tensor([[0.05]], dtype=torch.float32)
    radius = torch.tensor([[0.025]], dtype=torch.float32)
    current = -torch.ones((1, points.shape[1]), dtype=torch.float32)
    expected = native_csg(current, [xy, z], radius, points)
    actual = chunked_csg(current, [xy, z], radius, points, max_points=257)
    for reference, blocked in zip(expected, actual):
        torch.testing.assert_close(blocked, reference, rtol=0.0, atol=0.0)
    native_model.CSG = chunked_csg
    native_dataloader.DataLoader.__len__ = lambda self: 1

    def bounded_specs(directory):
        specs = copy.deepcopy(native_specs(directory))
        specs["NumEpochs"] = 1
        return specs

    native_workspace.load_experiment_specifications = bounded_specs

    def no_worker_loader(self, dataset, *args, **kwargs):
        kwargs["num_workers"] = 0
        return native_torch_loader_init(self, dataset, *args, **kwargs)

    torch.utils.data.DataLoader.__init__ = no_worker_loader
    native_argv = [str(source / "run.py"), "--input_object", "0.off", "--gpu", str(args.gpu),
                   "--experiment", "t50-tracked-smoke"]
    manifest = {
        "label": "CNC-Net @ba0652f on modernized stack, one-step native smoke; not faithful reproduction",
        "source_revision": args.source_revision,
        "sample_sha256": EXPECTED_SAMPLE_SHA256,
        "source": str(source),
        "work_dir": str(work_dir),
        "native_argv": native_argv,
        "native_num_epochs": 1,
        "native_dataset_len": 1,
        "num_data_workers": 0,
        "max_independent_point_block": MAX_POINTS,
        "chunked_csg_cpu_selfcheck": "bit-identical for 2003 points split at 257, both returned tensors",
        "mesh_io_override": str(PROJECT_ROOT / "baselines/cncnet/overrides/pymesh/__init__.py"),
        "started_at": started_at,
        "torch_version": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda,
        "status": "running",
    }
    (work_dir / "smoke-status.json").write_text(json.dumps(manifest, indent=2) + "\n")
    old_argv = sys.argv
    sys.argv = native_argv
    try:
        runpy.run_path(str(source / "run.py"), run_name="__main__")
        diff = subprocess.run(["git", "-C", str(source), "diff", "--quiet"], check=False)
        if diff.returncode:
            raise RuntimeError("Native run altered tracked source files")
        manifest["status"] = "native-entrypoint-returned"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["exception"] = repr(error)
        raise
    finally:
        sys.argv = old_argv
        manifest["elapsed_seconds"] = time.monotonic() - started
        (work_dir / "smoke-status.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("MODERN_NATIVE_ENTRYPOINT_RETURNED", json.dumps({
        "label": manifest["label"], "elapsed_seconds": manifest["elapsed_seconds"],
        "native_argv": native_argv, "work_dir": str(work_dir),
    }))


if __name__ == "__main__":
    main()
