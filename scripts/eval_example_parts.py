#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evaluate all 10 TITAN Building-Block STEP parts (``Titan-M1``..``Titan-M10``) and export viewer bundles.

Runs the full clean-room arbitrary-CAD pipeline:
  1. Native ISO 10303-21 STEP B-Rep parsing + 6-axis automatic setup orientation + feature extraction
  2. US Imperial (inch) Tool-Library TAMP task planning (including Op1 G54 + Op2 G55 soft-jaw flip & 3D ball scallop)
  3. Persistent-IPW DiffCAM CWE trajectory shortcutting & chipload optimization
  4. Line-mapped 6-dialect G-code post-processing + 3D WebGL viewer bundle export
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cam.tamp_planner import (
    HoleBlueprint,
    PocketBlueprint,
    PrismaticPartBlueprint,
    build_stl_from_blueprint,
    execute_tamp_pipeline,
    export_tamp_result_to_viewer_bundle,
)
from cam.tool_library import DEFAULT_TOOL_LIBRARY

EXAMPLE_PARTS_DIR = WORKSPACE_ROOT / "example_parts"
RUN_DIR = REPO_ROOT / "runs" / "pub-v2-20261110"

TITAN_EXAMPLE_SPECS: List[Dict[str, Any]] = [
    {"id": "titan-m1", "filename": "Titan-M1.step", "prog": 2101, "title": "TITAN-1M (Stepped Pocket + #2-56 Holes)"},
    {"id": "titan-m2", "filename": "Titan-M2.step", "prog": 2102, "title": "TITAN-2M (Angled Slots + C-Bores + Chamfers)"},
    {"id": "titan-m3", "filename": "Titan-M3.step", "prog": 2103, "title": "TITAN-3M (Circular Island Boss + #10-24)"},
    {"id": "titan-m4", "filename": "Titan-M4.step", "prog": 2104, "title": "TITAN-4M (Dual-Setup Op1/Op2 + #10-24)"},
    {"id": "titan-m5", "filename": "Titan-M5.step", "prog": 2105, "title": "TITAN-5M (Hex Hub + Radial Slots + 1/4-20)"},
    {"id": "titan-m6", "filename": "Titan-M6.step", "prog": 2106, "title": "TITAN-6M (Webbed Star Pocket + C-Sinks)"},
    {"id": "titan-m7", "filename": "Titan-M7.step", "prog": 2107, "title": "TITAN-7M (Dual-Setup Tri-Lobe + C-Bores)"},
    {"id": "titan-m8", "filename": "Titan-M8.step", "prog": 2108, "title": "TITAN-8M (Dual-Setup Twin Stepped Bores)"},
    {"id": "titan-m9", "filename": "Titan-M9.step", "prog": 2109, "title": "TITAN-9M (3D B-Spline Sculptured Surface)"},
    {"id": "titan-m10", "filename": "Titan-M10.step", "prog": 2110, "title": "TITAN-10M (Dual-Setup 3D Freeform + Threads)"},
]


def _refresh_tamp_stl_cases() -> List[Dict[str, Any]]:
    """Refresh the 4 prismatic TAMP benchmark cases with full 6-dialect G-code & 3D tool tables."""
    from scripts.run_tamp_and_overfit_experiments import _build_tamp_benchmark_blueprints

    tamp_dir = RUN_DIR / "tamp_shop_packets"
    tamp_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    for bp, mat_id, prog_no in _build_tamp_benchmark_blueprints():
        in_stl = tamp_dir / f"{bp.part_id}_input.stl"
        if not in_stl.exists():
            in_stl.write_bytes(build_stl_from_blueprint(bp, pitch_mm=0.25))
        res = execute_tamp_pipeline(
            in_stl,
            part_id=bp.part_id,
            title=bp.title,
            program_number=prog_no,
            stock_material_id=mat_id,
            opt_iters_per_op=35,
            filename_hint=in_stl.name,
        )
        _, m_entry = export_tamp_result_to_viewer_bundle(
            res, RUN_DIR, program_number=prog_no, split_name="tamp", source_cad_id=bp.part_id
        )
        entries.append(m_entry)
        print(
            f"  [TAMP-STL] {bp.part_id:22s} | {res.unique_tools_used}T {len(res.operations)} ops | "
            f"Naive={res.naive_total_time_s:6.2f}s -> Opt={res.opt_chipload_total_time_s:6.2f}s "
            f"({res.opt_chipload_gain_pct:+5.1f}%) | pass={res.safety_pass}"
        )
    return entries


def main() -> int:
    t_all = time.perf_counter()
    print("=== Evaluating All 10 TITAN Building-Block STEP Parts (US Library + Multi-Setup + 3D) ===")

    tamp_entries = _refresh_tamp_stl_cases()
    titan_manifest_entries: List[Dict[str, Any]] = []
    titan_eval_rows: List[Dict[str, Any]] = []
    example_parts_catalog: List[Dict[str, Any]] = []

    for spec in TITAN_EXAMPLE_SPECS:
        step_path = EXAMPLE_PARTS_DIR / spec["filename"]
        if not step_path.is_file():
            raise FileNotFoundError(f"Missing example STEP file: {step_path}")

        res = execute_tamp_pipeline(
            step_path,
            part_id=spec["id"],
            title=spec["title"],
            program_number=int(spec["prog"]),
            stock_material_id="al_6061_t6",
            unit_system="inches",
            setup_mode="auto",
            controller_dialect="haas_ngc",
            opt_iters_per_op=28,
            filename_hint=spec["filename"],
            enable_probing=False,
            enable_chamfer=True,
            enable_threading=True,
        )
        _, m_entry = export_tamp_result_to_viewer_bundle(
            res,
            RUN_DIR,
            program_number=int(spec["prog"]),
            split_name="example_cad",
            source_cad_id=spec["id"],
            can_replan=True,
        )
        titan_manifest_entries.append(m_entry)
        example_parts_catalog.append(
            {
                "id": spec["id"],
                "filename": spec["filename"],
                "label": spec["title"],
                "description": f"{res.cad_metadata['setup_summary']} · {res.cad_metadata['features_summary']}",
            }
        )

        dialect_lines = {d: len(p["lines"]) for d, p in res.gcode_by_dialect.items()}
        row = {
            "part_id": res.part_id,
            "filename": spec["filename"],
            "title": res.title,
            "source_units": res.features.source_unit,
            "setup_axis": res.features.setup_axis_label,
            "has_op2_flip": bool(res.cad_metadata["has_op2_flip"]),
            "part_dims_mm": res.cad_metadata["part_dims_mm"],
            "stock_dims_mm": res.cad_metadata["stock_dims_mm"],
            "features_summary": res.cad_metadata["features_summary"],
            "num_operations": len(res.operations),
            "unique_tools_used": res.unique_tools_used,
            "tools_used": [
                {
                    "tool_number": op.tool_number,
                    "work_offset": op.work_offset,
                    "op_kind": op.op_kind,
                    "tool_id": op.tool_record.tool_id,
                    "tool_label": op.tool_record.nominal_label,
                    "entity_type": op.tool_record.entity_type,
                }
                for op in res.operations
            ],
            "naive_total_time_s": res.naive_total_time_s,
            "same_speed_total_time_s": res.same_speed_total_time_s,
            "opt_chipload_total_time_s": res.opt_chipload_total_time_s,
            "same_speed_gain_pct": res.same_speed_gain_pct,
            "opt_chipload_gain_pct": res.opt_chipload_gain_pct,
            "opt_optimal_hex_pct": res.opt_optimal_hex_pct,
            "gouge_volume_mm3": res.gouge_volume_mm3,
            "residual_volume_mm3": res.residual_volume_mm3,
            "vise_collision": res.vise_collision,
            "holder_collision": res.holder_collision,
            "rapid_collision": res.rapid_collision,
            "safety_pass": res.safety_pass,
            "dialect_line_counts": dialect_lines,
            "planning_wall_time_s": res.planning_wall_time_s,
            "warnings": res.warnings,
        }
        titan_eval_rows.append(row)
        print(
            f"  [{spec['id']:9s}] rot={res.features.setup_axis_label:7s} op2={str(row['has_op2_flip']):5s} | "
            f"{res.unique_tools_used}T {len(res.operations):2d} ops | "
            f"Naive={res.naive_total_time_s:6.2f}s -> Same={res.same_speed_total_time_s:6.2f}s -> "
            f"Opt={res.opt_chipload_total_time_s:6.2f}s ({res.opt_chipload_gain_pct:+5.1f}%) | "
            f"gouge={res.gouge_volume_mm3:.3f}mm3 pass={res.safety_pass} ({res.planning_wall_time_s:.2f}s)"
        )

    # Update simulator_viewer/manifest.json with TITAN cases, refreshed TAMP cases, and full US+Metric tool_catalog
    manifest_path = RUN_DIR / "simulator_viewer" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema": "diffcam-viewer-manifest/2", "cases": []}

    new_ids = {e["case_id"] for e in (*tamp_entries, *titan_manifest_entries)}
    kept_cases = [c for c in manifest.get("cases", []) if c.get("case_id") not in new_ids]
    manifest["cases"] = [*titan_manifest_entries, *tamp_entries, *kept_cases]
    manifest["tool_catalog"] = DEFAULT_TOOL_LIBRARY.export_viewer_catalog()
    manifest["example_parts"] = example_parts_catalog

    mean_gain = sum(r["opt_chipload_gain_pct"] for r in titan_eval_rows) / max(1, len(titan_eval_rows))
    mean_same_gain = sum(r["same_speed_gain_pct"] for r in titan_eval_rows) / max(1, len(titan_eval_rows))
    pass_count = sum(1 for r in titan_eval_rows if r["safety_pass"])

    eval_summary = {
        "schema": "diffcam-titan-step-eval/1",
        "num_parts": len(titan_eval_rows),
        "safety_pass_count": pass_count,
        "safety_pass_rate_pct": round(100.0 * pass_count / max(1, len(titan_eval_rows)), 1),
        "mean_same_speed_gain_pct": round(mean_same_gain, 2),
        "mean_opt_chipload_gain_pct": round(mean_gain, 2),
        "total_eval_wall_time_s": round(time.perf_counter() - t_all, 2),
        "parts": titan_eval_rows,
    }
    manifest["titan_step_eval_summary"] = {
        k: v for k, v in eval_summary.items() if k != "parts"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    eval_out_path = RUN_DIR / "titan_example_parts_eval.json"
    eval_out_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    print(
        f"\n=== Completed 10/10 TITAN STEP Evaluation in {eval_summary['total_eval_wall_time_s']:.2f}s: "
        f"Safety={pass_count}/{len(titan_eval_rows)}, Mean Opt Gain={mean_gain:+.2f}% ==="
    )
    print(f"Updated {manifest_path} and {eval_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
