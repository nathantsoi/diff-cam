"""Read-only inventory of native CNC-Net smoke artifacts using an explicit path."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import numpy as np
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=Path, required=True)
parser.add_argument("--experiment", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--compare-to", type=Path, default=Path(__file__).with_name("reference_smoke.json"),
                    help="Earlier verified inventory; differences are reported, not silently cast as score failures")
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite existing evidence: {args.output}")
artifacts = {}
for relative in (
    Path(f"output/{args.experiment}/0.obj"),
    Path(f"output/{args.experiment}/0_smooth.obj"),
    Path(f"checkpoints/{args.experiment}/logfile.log"),
    Path(f"checkpoints/{args.experiment}/best_checkpoint.pth"),
):
    path = args.runtime / relative
    if not path.is_file() or path.is_symlink():
        continue
    record = {"path": str(path), "size_bytes": path.stat().st_size}
    hasher = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    record["sha256"] = hasher.hexdigest()
    if path.suffix == ".obj":
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        record.update({
            "readable_mesh": True,
            "vertices": len(vertices),
            "triangles": len(faces),
            "finite_vertices": bool(vertices.size and np.isfinite(vertices).all()),
            "bbox_min": vertices.min(axis=0).tolist() if len(vertices) else None,
            "bbox_max": vertices.max(axis=0).tolist() if len(vertices) else None,
        })
    artifacts[str(relative)] = record

iou_values = []
logfile = args.runtime / f"checkpoints/{args.experiment}/logfile.log"
if logfile.is_file():
    for line in logfile.read_text(errors="replace").splitlines():
        if "Average IOU:" in line:
            suffix = line.split("Average IOU:", 1)[1]
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", suffix)
            if match:
                iou_values.append(float(match.group(0)))
report = {
    "experiment": args.experiment,
    "sample": "samples/0.off",
    "sample_sha256": "8e80d539b1c593501a55bd9f7b3c6b9099ce57fdb556e298b0bd3b744f690b7c",
    "native_logged_iou_values": iou_values,
    "native_artifacts": artifacts,
    "inspected_at": datetime.now(timezone.utc).isoformat(),
    "valid_complete_smoke": bool(
        len(iou_values) == 1
        and all(math.isfinite(v) and 0 <= v <= 1 for v in iou_values)
        and f"output/{args.experiment}/0.obj" in artifacts
        and artifacts.get(f"output/{args.experiment}/0.obj", {}).get("readable_mesh")
        and artifacts.get(f"output/{args.experiment}/0.obj", {}).get("finite_vertices")
        and artifacts.get(f"output/{args.experiment}/0.obj", {}).get("triangles", 0) > 0
        and f"output/{args.experiment}/0_smooth.obj" in artifacts
        and artifacts.get(f"output/{args.experiment}/0_smooth.obj", {}).get("readable_mesh")
        and artifacts.get(f"output/{args.experiment}/0_smooth.obj", {}).get("finite_vertices")
        and artifacts.get(f"output/{args.experiment}/0_smooth.obj", {}).get("triangles", 0) > 0
    ),
    "note": "Native IoU is the value emitted by CNC-Net's own logging. Mesh readability/finite vertices were checked independently from the downloaded model; this is not a quality or physical feasibility gate.",
}
if args.compare_to is not None:
    reference_bytes = args.compare_to.read_bytes()
    reference = json.loads(reference_bytes)
    reference_iou = reference["native_logged_iou_values"]
    if len(reference_iou) != 1:
        raise ValueError("comparison inventory must contain exactly one bounded-run IoU")
    comparison = {
        "reference_report": str(args.compare_to),
        "reference_report_sha256": sha256(reference_bytes).hexdigest(),
        "reference_iou": reference_iou[0],
        "current_iou": iou_values[0] if len(iou_values) == 1 else None,
        "iou_absolute_delta": abs(iou_values[0] - reference_iou[0]) if len(iou_values) == 1 else None,
        "artifact_hash_matches": {},
    }
    for basename in ("0.obj", "0_smooth.obj", "best_checkpoint.pth", "logfile.log"):
        reference_records = [record for path, record in reference["native_artifacts"].items() if path.endswith("/" + basename)]
        current_records = [record for path, record in artifacts.items() if path.endswith("/" + basename)]
        if len(reference_records) == 1 and len(current_records) == 1:
            comparison["artifact_hash_matches"][basename] = current_records[0]["sha256"] == reference_records[0]["sha256"]
    comparison["all_matched_artifacts_byte_identical"] = all(comparison["artifact_hash_matches"].values()) if comparison["artifact_hash_matches"] else None
    report["comparison_to_verified_one_step_smoke"] = comparison
    report["comparison_policy"] = "Differences are preserved and reviewed as possible GPU/environment nondeterminism or changed recipe details; this checker does not infer their cause or quality significance."
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not report["valid_complete_smoke"]:
    raise SystemExit("Incomplete native smoke: missing one-step IoU, readable nonempty/finite raw/smooth meshes, or both")
