# SPDX-License-Identifier: MIT
"""Export RustCNC (.rustcnc), CAMotics (.camotics), and WebGL 3D viewer bundles from a campaign run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import contracts as C
from benchmarks.milling_suite import CORNER_RADIUS_KEY, MillingTaskBundle, build_milling_suite
from scripts.summarize_campaign_tables import METHODS, summarize
from verification.simulator_export import (
    build_target_part_stl_bytes,
    carve_resulting_stock_stl_bytes,
    export_camotics_project,
    export_rustcnc_project,
    reconstruct_motion_plan,
    safe_resolve_under,
    serialize_plan_for_viewer,
    translate_binary_stl_bytes,
)

PLAYABLE_VARIANTS: Tuple[Tuple[str, str, str, Optional[str]], ...] = (
    ("ste_adam", "STE-Continuation-Adam", "DiffCAM STE-Adam", "DiffCAM-STE"),
    ("nominal_cam", "Nominal-CAM", "Nominal-CAM (@2)", "Nominal-CAM"),
    ("shac_policy", "DiffPolicy-SHAC", "DiffPolicy-SHAC", None),
)


def _select_best_seed_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select the fastest feasible seed record, falling back to lowest objective."""
    feasible = [r for r in records if bool(r.get("feasible", False))]
    if feasible:
        return min(feasible, key=lambda r: float(r.get("cycle_time_s", 1e12)))
    return min(records, key=lambda r: float(r.get("objective", 1e12)))


def _summarize_case_benchmark_table(task_entry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Aggregate per-method tabular rows for a single benchmark task across seeds."""
    rows: List[Dict[str, Any]] = []
    methods_map = task_entry.get("methods", {})
    for method in METHODS:
        recs = methods_map.get(method, [])
        if not recs:
            continue
        times = [float(r["cycle_time_s"]) for r in recs]
        gains = [float(r["cycle_time_reduction_pct"]) for r in recs]
        mrrs = [float(r["peak_mrr_mm3_s"]) for r in recs]
        feas = [1.0 if bool(r["feasible"]) else 0.0 for r in recs]
        walls = [float(r["wall_time_s"]) for r in recs]
        posted_rec = next((r.get("posted") for r in recs if r.get("posted") is not None), None)
        rows.append(
            {
                "method": method,
                "cycle_time_mean_s": round(float(np.mean(times)), 2),
                "cycle_time_std_s": round(float(np.std(times)), 2),
                "gain_mean_pct": round(float(np.mean(gains)), 1),
                "gain_std_pct": round(float(np.std(gains)), 1),
                "peak_mrr_mean_mm3_s": round(float(np.mean(mrrs)), 1),
                "feasible_rate_pct": round(100.0 * float(np.mean(feas)), 1),
                "wall_mean_s": round(float(np.mean(walls)), 2),
                "posted_pass": (
                    bool(posted_rec["geometry_pass"] and posted_rec["safety_pass"])
                    if posted_rec is not None
                    else None
                ),
            }
        )
    return rows


def export_all_simulator_bundles(
    run_dir: Path,
    *,
    use_camotics_docker: bool = False,
    all_splits: bool = False,
) -> Dict[str, Any]:
    """Generate `.rustcnc`, `.camotics`, and `simulator_viewer/` assets inside `run_dir`."""
    run_dir = Path(run_dir).resolve()
    exp2_path = safe_resolve_under(run_dir, "exp2_main_benchmark.json")
    if not exp2_path.is_file():
        raise FileNotFoundError(f"Missing exp2_main_benchmark.json in {run_dir}")

    exp2 = json.loads(exp2_path.read_text(encoding="utf-8"))
    exp4_path = safe_resolve_under(run_dir, "exp4_camotics_and_haas.json")
    exp4 = json.loads(exp4_path.read_text(encoding="utf-8")) if exp4_path.is_file() else {}

    summary_path = safe_resolve_under(run_dir, "summary_metrics.json")
    if summary_path.is_file():
        summary_metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    elif (
        safe_resolve_under(run_dir, "exp1_conditioning.json").is_file()
        and safe_resolve_under(run_dir, "exp2_ablations.json").is_file()
        and safe_resolve_under(run_dir, "exp3_policy_transfer.json").is_file()
        and exp4_path.is_file()
    ):
        summary_metrics = summarize(run_dir)
    else:
        summary_metrics = {}

    task_by_id: Dict[str, Mapping[str, Any]] = {
        str(t["case_id"]): t for t in exp2.get("tasks", [])
    }
    exp4_ver_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {
        (str(v["case_id"]), str(v["variant"])): v for v in exp4.get("verifications", [])
    }
    haas_pkt_by_id: Dict[str, Mapping[str, Any]] = {
        str(p["case_id"]): p for p in exp4.get("haas_packets", [])
    }

    rustcnc_dir = safe_resolve_under(run_dir, "rustcnc_projects")
    camotics_dir = safe_resolve_under(run_dir, "camotics_projects")
    viewer_dir = safe_resolve_under(run_dir, "simulator_viewer")
    cases_dir = safe_resolve_under(viewer_dir, "cases")
    meshes_dir = safe_resolve_under(viewer_dir, "meshes")
    for d in (rustcnc_dir, camotics_dir, viewer_dir, cases_dir, meshes_dir):
        d.mkdir(parents=True, exist_ok=True)

    fine_bundles = build_milling_suite(
        voxel_mm=0.25,
        timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
    )
    selected_bundles = (
        list(fine_bundles)
        if all_splits
        else [b for b in fine_bundles if b.task.split == C.Split.TEST]
    )

    manifest_cases: List[Dict[str, Any]] = []
    sim_to_work_cache: Dict[str, Tuple[float, float, float]] = {}

    for bundle in selected_bundles:
        cid = bundle.case.case_id
        task_entry = task_by_id.get(cid)
        if task_entry is None:
            continue

        sim_to_work = bundle.task.work_to_machine.inverse().apply(
            bundle.task.initial_state.position_mm
        )
        sim_to_work_cache[cid] = sim_to_work

        # 1. Target CAD part STL in C.Frame.WORK
        target_stl_name = f"{cid}_target.stl"
        target_stl_path = safe_resolve_under(meshes_dir, target_stl_name)
        target_stl_path.write_bytes(build_target_part_stl_bytes(bundle))

        # 2. Reconstruct playable variants + export .camotics + stock .stl
        rustcnc_variants: List[Tuple[str, str, C.MotionPlan]] = []
        viewer_variants: List[Dict[str, Any]] = []
        haas_pkt = haas_pkt_by_id.get(cid, {})

        for var_id, method_key, display_label, exp4_variant_label in PLAYABLE_VARIANTS:
            method_recs = task_entry.get("methods", {}).get(method_key, [])
            if not method_recs:
                continue
            best_rec = _select_best_seed_record(method_recs)
            best_theta = best_rec.get("best_theta")
            plan = reconstruct_motion_plan(bundle, method_key, best_theta)
            rustcnc_variants.append((var_id, display_label, plan))

            # Export .camotics project + .nc
            cam_info = export_camotics_project(
                camotics_dir,
                bundle,
                var_id,
                plan,
                resolution_mm=0.25,
            )

            # Reuse CAMotics Docker STL from Exp4 when available (translated to C.Frame.WORK),
            # otherwise carve via CamoticsAdapter (if requested) or exact 0.25 mm voxel carve.
            stock_stl_bytes: Optional[bytes] = None
            mesh_source = ""
            if exp4_variant_label is not None:
                ver_row = exp4_ver_by_key.get((cid, exp4_variant_label))
                if ver_row and ver_row.get("stl_relpath"):
                    existing_stl = safe_resolve_under(run_dir, str(ver_row["stl_relpath"]))
                    if existing_stl.is_file():
                        stock_stl_bytes = translate_binary_stl_bytes(
                            existing_stl.read_bytes(),
                            offset_mm=sim_to_work,
                        )
                        mesh_source = "camotics_v1.2.0"

            if stock_stl_bytes is None:
                scratch_sub = (
                    safe_resolve_under(run_dir, f"camotics_stls/{cid}_{var_id}_viewer")
                    if use_camotics_docker
                    else None
                )
                stock_stl_bytes, mesh_source = carve_resulting_stock_stl_bytes(
                    bundle,
                    plan,
                    use_camotics_docker=use_camotics_docker,
                    scratch_dir=scratch_sub,
                )
                if scratch_sub is not None and scratch_sub.exists():
                    shutil.rmtree(scratch_sub, ignore_errors=True)

            stock_stl_name = f"{cid}_{var_id}_stock.stl"
            safe_resolve_under(meshes_dir, stock_stl_name).write_bytes(stock_stl_bytes)

            # Link Haas NGC shop file when present
            haas_url: Optional[str] = None
            if var_id == "ste_adam" and haas_pkt.get("file"):
                haas_url = f"/downloads/haas/{haas_pkt['file']}"
            elif var_id == "nominal_cam" and haas_pkt.get("nominal_file"):
                haas_url = f"/downloads/haas/{haas_pkt['nominal_file']}"

            ver_meta = (
                exp4_ver_by_key.get((cid, exp4_variant_label), {})
                if exp4_variant_label is not None
                else {}
            )
            posted_meta = best_rec.get("posted") or {}

            viewer_variants.append(
                {
                    "id": var_id,
                    "method": method_key,
                    "label": display_label,
                    "seed": int(best_rec.get("seed", 0)),
                    "stock_stl_url": f"/data/meshes/{stock_stl_name}",
                    "stock_mesh_source": mesh_source,
                    "camotics_project_url": f"/downloads/camotics/{cam_info['project_file']}",
                    "camotics_nc_url": f"/downloads/camotics/{cam_info['nc_file']}",
                    "haas_nc_url": haas_url,
                    "metrics": {
                        "cycle_time_s": round(float(sum(s.duration_s for s in plan.segments)), 3),
                        "nominal_cycle_time_s": round(
                            float(sum(s.duration_s for s in bundle.nominal_motion.segments)), 3
                        ),
                        "cycle_time_reduction_pct": round(
                            float(best_rec.get("cycle_time_reduction_pct", 0.0)), 2
                        ),
                        "peak_mrr_mm3_s": round(float(best_rec.get("peak_mrr_mm3_s", 0.0)), 2),
                        "air_fraction": round(float(best_rec.get("air_fraction", 0.0)), 4),
                        "feasible": bool(best_rec.get("feasible", True)),
                        "posted_geometry_pass": bool(
                            ver_meta.get(
                                "posted_0p125mm_geometry_pass",
                                posted_meta.get("geometry_pass", True),
                            )
                        ),
                        "posted_safety_pass": bool(
                            ver_meta.get(
                                "posted_0p125mm_safety_pass",
                                posted_meta.get("safety_pass", True),
                            )
                        ),
                        "orig_gouge_mm3": round(
                            float(
                                ver_meta.get(
                                    "posted_0p125mm_orig_gouge_mm3",
                                    posted_meta.get("original_part_gouge_mm3", 0.0),
                                )
                            ),
                            4,
                        ),
                        "residual_mm3": round(
                            float(
                                ver_meta.get(
                                    "posted_0p125mm_residual_mm3",
                                    posted_meta.get("residual_mm3", 0.0),
                                )
                            ),
                            4,
                        ),
                        "camotics_watertight": ver_meta.get("camotics_watertight", None),
                    },
                    "plan": serialize_plan_for_viewer(bundle, plan),
                }
            )

        # 3. Export native .rustcnc Manufacture Workspace archive
        rustcnc_filename = f"{cid}.rustcnc"
        rustcnc_path = safe_resolve_under(rustcnc_dir, rustcnc_filename)
        rustcnc_info = export_rustcnc_project(rustcnc_path, bundle, rustcnc_variants)

        geom_params = bundle.task.operation_target.geometry.params
        tool = bundle.task.tool
        case_payload = {
            "schema": "diffcam-viewer-case/1",
            "case_id": cid,
            "family": bundle.case.family,
            "split": bundle.task.split.value,
            "stock_min_mm": list(bundle.case.stock_min_mm),
            "stock_max_mm": list(bundle.case.stock_max_mm),
            "pocket_min_mm": [float(v) for v in geom_params["pocket_min_mm"]],
            "pocket_max_mm": [float(v) for v in geom_params["pocket_max_mm"]],
            "corner_radius_mm": float(geom_params.get(CORNER_RADIUS_KEY, 0.0)),
            "islands_mm": [
                [[float(x) for x in lo], [float(x) for x in hi]]
                for lo, hi in geom_params.get("islands_mm", ())
            ],
            "tool": {
                "tool_number": int(tool.tool_number),
                "cutter_radius_mm": float(tool.cutter_radius_mm),
                "diameter_mm": 2.0 * float(tool.cutter_radius_mm),
                "flute_length_mm": float(tool.flute_length_mm),
                "shank_radius_mm": float(tool.shank_radius_mm),
                "holder_radius_mm": float(tool.holder_radius_mm),
                "holder_bottom_mm": float(tool.holder_bottom_mm),
                "holder_height_mm": float(tool.holder_height_mm),
            },
            "target_stl_url": f"/data/meshes/{target_stl_name}",
            "rustcnc_download_url": f"/downloads/rustcnc/{rustcnc_filename}",
            "rustcnc_sha256": rustcnc_info["sha256"],
            "variants": viewer_variants,
            "benchmark_table": _summarize_case_benchmark_table(task_entry),
        }
        case_json_name = f"{cid}.json"
        safe_resolve_under(cases_dir, case_json_name).write_text(
            json.dumps(case_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        by_var = {v["id"]: v for v in viewer_variants}
        nom_s = by_var.get("nominal_cam", {}).get("metrics", {}).get("cycle_time_s", 0.0)
        ste_s = by_var.get("ste_adam", {}).get("metrics", {}).get("cycle_time_s", 0.0)
        ste_g = by_var.get("ste_adam", {}).get("metrics", {}).get("cycle_time_reduction_pct", 0.0)
        shac_s = by_var.get("shac_policy", {}).get("metrics", {}).get("cycle_time_s", 0.0)
        shac_g = by_var.get("shac_policy", {}).get("metrics", {}).get("cycle_time_reduction_pct", 0.0)

        manifest_cases.append(
            {
                "case_id": cid,
                "family": bundle.case.family,
                "split": bundle.task.split.value,
                "nominal_cycle_time_s": nom_s,
                "ste_cycle_time_s": ste_s,
                "ste_gain_pct": ste_g,
                "shac_cycle_time_s": shac_s,
                "shac_gain_pct": shac_g,
                "rustcnc_download_url": f"/downloads/rustcnc/{rustcnc_filename}",
                "case_json_url": f"/data/cases/{case_json_name}",
            }
        )
        print(
            f"[SimulatorExport] {cid}: .rustcnc ({rustcnc_info['size_bytes'] // 1024} KiB), "
            f"{len(viewer_variants)} .camotics bundles, target+stock .stl meshes.",
            flush=True,
        )

    manifest_payload = {
        "schema": "diffcam-simulator-viewer/1",
        "run_id": run_dir.name,
        "cases": manifest_cases,
        "paper_summary": summary_metrics,
    }
    safe_resolve_under(viewer_dir, "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "pub-v2-20261110",
        help="Campaign directory containing exp2_main_benchmark.json and exp4_camotics_and_haas.json.",
    )
    parser.add_argument(
        "--use-camotics",
        action="store_true",
        help="Invoke CAMotics Docker image when a pre-carved CAMotics STL is not already cached.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Export dev + test splits (16 tasks) instead of the 8 held-out test parts.",
    )
    args = parser.parse_args()
    manifest = export_all_simulator_bundles(
        args.run_dir,
        use_camotics_docker=args.use_camotics,
        all_splits=args.all_splits,
    )
    print(f"[DONE] Exported simulator bundles for {len(manifest['cases'])} cases in {args.run_dir.resolve()}.")


if __name__ == "__main__":
    main()
