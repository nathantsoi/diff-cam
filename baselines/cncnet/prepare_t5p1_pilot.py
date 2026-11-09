"""Prepare the frozen dev pocket OFF and transform, without native execution.

Run with the core interpreter. The only git operation on the external source is
identity verification. The destination must be a new, private staging directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from benchmarks import contracts as C
from benchmarks import cncnet_adapter as A

REVISION = "ba0652f9941686058c07c98617b9748e72f5f019"


def verify_source(source: Path):
    actual = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False,
                            env={"GIT_TERMINAL_PROMPT": "0"})
    if actual.returncode or actual.stdout.strip() != REVISION:
        raise RuntimeError(f"expected CNC-Net {REVISION}, found {actual.stdout.strip()!r}")
    clean = subprocess.run(["git", "-C", str(source), "diff", "--quiet"], check=False)
    if clean.returncode:
        raise RuntimeError("untracked/dirty tracked native source is refused")


def prepare(directory: Path, source: Path):
    verify_source(source)
    destination = directory.expanduser().resolve(strict=False)
    if destination.exists():
        raise ValueError(f"refusing existing staging directory: {destination}")
    destination.mkdir(parents=True)
    task, frame = A.make_dev_box_pocket_task(), A.unit_frame_for_pilot(A.make_dev_box_pocket_task())
    vertices, faces = A.prepare_native_mesh(task, frame)
    audit = A.audit_mesh(vertices, faces)
    off = A.write_off(destination / "dev_box_pocket.off", vertices, faces)
    A.write_json(destination / "task.json", json.loads(C.dumps(task)))
    A.write_json(destination / "frame.json", frame.to_dict())
    A.write_json(destination / "source.json", {"revision": REVISION, "path": str(source),
                                                "license": "missing/unclear"})
    A.write_json(destination / "summary.json", {
        "schema": "diffcam-cncnet-prepared/1", "status": "prepared", "task_id": task.task_id,
        "task_hash": task.task_hash, "off_hash": off["sha256"], "mesh_audit": audit,
        "frame": frame.to_dict(), "frozen_budget": {"epoch_count": 1, "per_instance_optimizer_steps": 1},
        "native_data_scope": "development; not held-out evidence or full training",
        "executable_nc_disposition": A.executable_disposition(),
    })
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    output = prepare(args.prepare_dir, args.source.resolve(strict=True))
    print(json.dumps({"status": "prepared", "prepare_dir": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
