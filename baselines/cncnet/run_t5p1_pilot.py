"""One bounded native CNC-Net reconstruction with immutable artifact capture.

Requires the already-built, separately isolated torch 2.9.1cu128/CPython 3.11
runtime. The pinned model, loss, IOU, CSG and marching-cubes operations are
invoked, not substituted. The only compatibility wrappers are budget limiting,
zero data workers, independent CSG point blocking, and artifact observation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

REVISION = "ba0652f9941686058c07c98617b9748e72f5f019"
EXPECTED_SAMPLE_SHA256 = "8e80d539b1c593501a55bd9f7b3c6b9099ce57fdb556e298b0bd3b744f690b7c"
MAX_POINTS = 1_048_576


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source):
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if result.returncode or result.stdout.strip() != REVISION:
        raise RuntimeError(f"CNC-Net source revision mismatch: {result.stdout.strip()!r}")
    dirty = subprocess.run(["git", "-C", str(source), "diff", "--quiet"], check=False)
    if dirty.returncode:
        raise RuntimeError("CNC-Net tracked source is modified")
    sample = source / "samples/0.off"
    if sha256_file(sample) != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError("CNC-Net bundled sample digest mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        raise ValueError("GPU index must be nonnegative")
    source = args.source.resolve(strict=True)
    verify_source(source)
    prepared = args.prepared.resolve(strict=True)
    manifest = json.loads((prepared / "summary.json").read_text())
    off = prepared / "dev_box_pocket.off"
    if "sha256:" + sha256_file(off) != manifest["off_hash"]:
        raise RuntimeError("prepared OFF hash differs from its immutable summary")
    work_dir = args.work_dir.expanduser().resolve(strict=False)
    if work_dir.exists():
        raise ValueError(f"refusing existing native output directory: {work_dir}")
    experiment = "t51p1-dev-pocket"
    work_dir.mkdir(parents=True)
    os.chdir(work_dir)
    os.symlink(source / "configs", work_dir / "configs", target_is_directory=True)
    (work_dir / "samples").mkdir()
    os.symlink(off, work_dir / "samples" / "dev_box_pocket.off")
    sys.dont_write_bytecode = True
    # -I deliberately ignores PYTHONPATH. Load only the reviewed compatibility
    # mesh-I/O subset by its explicit trusted path and verified content hash.
    mesh_override = Path(__file__).resolve().parent / "overrides" / "pymesh" / "__init__.py"
    if sha256_file(mesh_override) != "19223c46a252cf5795105404c9e81957bb92937e15687fec177bf8cca2e95faa":
        raise RuntimeError("reviewed mesh-I/O override hash mismatch")
    sys.path.insert(0, str(mesh_override.parent.parent))
    sys.path.insert(0, str(source))

    started = time.monotonic()
    state = {
        "schema": "diffcam-cncnet-native/1",
        "label": "CNC-Net @ba0652f modernized stack; one-update reconstruction; NOT faithful legacy reproduction",
        "original_revision": REVISION,
        "source_path": str(source), "work_dir": str(work_dir),
        "preparing_task_hash": manifest["task_hash"], "input_off_hash": manifest["off_hash"],
        "native_object_file": "dev_box_pocket.off",
        "epoch_count": 1, "dataset_length": 1, "native_data_workers": 0,
        "optimizer_step_limit": 1,
        "pretraining_checkpoint_loaded": False,
        "independent_csg_max_point_block": MAX_POINTS,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "started", "gpu_index": args.gpu,
        "native_export_input": "dev_box_pocket.off",
        "experiment": experiment,
    }
    (work_dir / "capture-status.json").write_text(json.dumps(state, indent=2) + "\n")

    import numpy as np
    import torch
    import pytorch3d
    import utils.dataloader as native_dataloader
    import utils.workspace as native_workspace
    import networks.model as native_model
    from networks.primitives import CSG as native_csg
    from utils.cad_meshing import create_mesh_mc as native_mesh_mc
    from utils.eval_metric import IOU as native_iou

    if torch.__version__ != "2.9.1+cu128" or pytorch3d.__version__ != "0.7.9":
        raise RuntimeError(f"unverified native stack torch={torch.__version__}, pytorch3d={pytorch3d.__version__}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(args.gpu) != (12, 0):
        raise RuntimeError("bounded native pilot requires the already-qualified RTX 5070 Ti / sm_120")
    state["torch_version"] = torch.__version__
    state["torch_compiled_cuda"] = torch.version.cuda
    state["pytorch3d_version"] = pytorch3d.__version__
    state["numpy_version"] = np.__version__
    import pytorch3d._C as native_extension
    state["pytorch3d_compiled_extension_sha256"] = sha256_file(native_extension.__file__)
    if state["pytorch3d_compiled_extension_sha256"] != "6d377cdc8d344e152aee4ca17a6f731043f7d4b47ad0d0b1851028227ef450b0":
        raise RuntimeError("compiled PyTorch3D extension differs from the verified pinned build")
    counters = {"optimizer_steps": 0, "iou_calls": 0, "mesh_calls": 0, "dataset_inits": 0}

    state["native_measurements"] = counters

    def chunked_csg(current, xyz, radius, points, max_points=MAX_POINTS):
        if points.shape[-2] <= max_points:
            return native_csg(current, xyz, radius, points)
        current_blocks, cylinder_blocks = [], []
        for start in range(0, points.shape[-2], max_points):
            stop = min(start + max_points, points.shape[-2])
            current_block, cylinder_block = native_csg(
                current[..., start:stop], xyz, radius, points[:, start:stop])
            current_blocks.append(current_block)
            cylinder_blocks.append(cylinder_block)
        return torch.cat(current_blocks, dim=-1), torch.cat(cylinder_blocks, dim=-1)

    rng = torch.Generator().manual_seed(501)
    points = torch.rand((1, 2003, 3), generator=rng) - 0.5
    xy = torch.linspace(-0.2, 0.2, 100).repeat(2, 1).T.unsqueeze(0)
    z = torch.tensor([[0.05]], dtype=torch.float32)
    radius = torch.tensor([[0.025]], dtype=torch.float32)
    current = -torch.ones((1, points.shape[1]), dtype=torch.float32)
    expected = native_csg(current, [xy, z], radius, points)
    actual = chunked_csg(current, [xy, z], radius, points, max_points=257)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(b, a, rtol=0, atol=0)
    state["chunk_cpu_check"] = "2003-point native CSG versus 257-point blocks: bit-identical both outputs"
    native_model.CSG = chunked_csg

    specs_function = native_workspace.load_experiment_specifications
    def bounded_specs(directory):
        specs = copy.deepcopy(specs_function(directory))
        specs["NumEpochs"] = 1
        state["source_epoch_count"] = specs_function(directory)["NumEpochs"]
        return specs
    native_workspace.load_experiment_specifications = bounded_specs

    original_loader_init = torch.utils.data.DataLoader.__init__
    def no_worker_loader(self, dataset, *args, **kwargs):
        kwargs["num_workers"] = 0
        return original_loader_init(self, dataset, *args, **kwargs)
    torch.utils.data.DataLoader.__init__ = no_worker_loader

    dataset_init = native_dataloader.DataLoader.__init__
    def audited_dataset(self, *args, **kwargs):
        result = dataset_init(self, *args, **kwargs)
        counters["dataset_inits"] += 1
        if counters["dataset_inits"] == 1:
            np.save(work_dir / "native_target_low.npy", np.asarray(self.indices_low, dtype=np.float32))
            dimensions = np.asarray(self.dimension, dtype=np.float64)
            first = np.asarray(self.all_points_low[0], dtype=np.float64)
            state["native_input"] = {
                "dimensions": dimensions.tolist(),
                "min_grid_point": first.tolist(),
                "expected_centred_minimum": (-dimensions / 2.0).tolist(),
                "low_points_min": self.all_points_low.min(axis=0).tolist(),
                "low_points_max": self.all_points_low.max(axis=0).tolist(),
                "high_first_point": self.all_points_high[0].tolist(),
                "high_last_point": self.all_points_high[-1].tolist(),
            }
            if not np.allclose(first, -dimensions / 2, rtol=0.0, atol=1e-10):
                raise RuntimeError("native loader origin disagrees with declared centred frame")
        return result
    native_dataloader.DataLoader.__init__ = audited_dataset
    native_dataloader.DataLoader.__len__ = lambda self: 1

    original_adam_step = torch.optim.Adam.step
    def audited_step(self, *args, **kwargs):
        result = original_adam_step(self, *args, **kwargs)
        counters["optimizer_steps"] += 1
        return result
    torch.optim.Adam.step = audited_step

    def audited_iou(current, target):
        result = native_iou(current, target)
        counters["iou_calls"] += 1
        np.save(work_dir / "native_current_low.npy", current.detach().cpu().numpy())
        np.save(work_dir / "native_target_low.npy", target.detach().cpu().numpy())
        state["last_native_iou"] = float(result)
        if not np.isfinite(result):
            raise RuntimeError("native IOU is non-finite")
        return result
    import utils.eval_metric
    utils.eval_metric.IOU = audited_iou

    def audited_mesh(all_points, current, dimension, filename, *args, **kwargs):
        counters["mesh_calls"] += 1
        raw = current.detach().cpu().numpy()
        if raw.shape != (1, 256 ** 3):
            raise RuntimeError(f"unexpected native high reconstruction shape: {raw.shape}")
        dims = dimension.detach().cpu().numpy()[0]
        expected = np.asarray(state["native_input"]["dimensions"], dtype=np.float64)
        if not np.allclose(dims, expected, rtol=0.0, atol=1e-10):
            raise RuntimeError("native reconstruction dimensions differ from loader")
        np.save(work_dir / "native_current_high.npy", raw.reshape(256, 256, 256).astype(np.float32))
        state["reconstruction"] = {
            "shape": [256, 256, 256], "dimensions": dims.tolist(),
            "positive_is_occupancy": True,
            "data_source": "exact pre-mesh outputs_high (0.5*(1-sign(x))) captured without changing native mesh creation",
        }
        return native_mesh_mc(all_points, current, dimension, filename, *args, **kwargs)
    import utils.cad_meshing
    utils.cad_meshing.create_mesh_mc = audited_mesh

    native_argv = [str(source / "run.py"), "--input_object", "dev_box_pocket.off", "--gpu", str(args.gpu),
                   "--experiment", experiment]
    state["native_argv"] = native_argv
    original_argv = sys.argv
    sys.argv = native_argv
    try:
        runpy.run_path(str(source / "run.py"), run_name="__main__")
        verify_source(source)
        state["status"] = "native-entrypoint-returned"
        if counters["optimizer_steps"] != 1:
            raise RuntimeError(f"expected exactly one optimizer update, observed {counters['optimizer_steps']}")
        if counters["iou_calls"] != 1 or counters["mesh_calls"] != 1:
            raise RuntimeError(f"incomplete or duplicated native evaluation counts: {counters}")
        mesh_stem = off.stem
        for suffix in (".obj", "_smooth.obj"):
            mesh = work_dir / "output" / experiment / f"{mesh_stem}{suffix}"
            if not mesh.is_file():
                raise RuntimeError(f"native mesh missing: {mesh}")
        state["artifact_files"] = {}
        for relative in (
            "native_current_high.npy", "native_current_low.npy", "native_target_low.npy",
            f"output/{experiment}/{mesh_stem}.obj",
            f"output/{experiment}/{mesh_stem}_smooth.obj",
            f"checkpoints/{experiment}/logfile.log",
        ):
            path = work_dir / relative
            if not path.is_file():
                raise RuntimeError(f"missing output artifact: {path}")
            state["artifact_files"][relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        state["native_measurements"] = counters
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.argv = original_argv
        state["elapsed_seconds"] = time.monotonic() - started
        (work_dir / "capture-status.json").write_text(json.dumps(state, indent=2) + "\n")
    print("T5P1_NATIVE_CAPTURE", json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # Import/setup failures precede main's native-entrypoint try/finally.
        # Never leave a stale "started" status behind for a failed subprocess.
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--work-dir", required=True, type=Path)
            args, _ = parser.parse_known_args()
            status_path = args.work_dir.expanduser().resolve(strict=False) / "capture-status.json"
            if status_path.is_file():
                state = json.loads(status_path.read_text())
                if state.get("status") == "started":
                    state["status"] = "failed"
                    state["error"] = f"{type(error).__name__}: {error}"
                    status_path.write_text(json.dumps(state, indent=2) + "\n")
        except Exception:
            import traceback
            traceback.print_exc()
        raise
