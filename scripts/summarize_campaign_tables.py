# SPDX-License-Identifier: MIT
"""Deterministic statistical summary & LaTeX table verifier for DiffCAM campaign artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("facing_open_boss", "pocket_island", "adaptive_channel", "contour_profile")
METHODS = (
    "Nominal-CAM",
    "Naive-Smooth-Adam",
    "Annealed-Adam",
    "STE-Continuation-Adam",
    "DiffPolicy-SHAC",
    "Hard-PPO",
    "Hard-CEM",
    "Hard-CMA-ES",
)


def summarize(run_dir: Path) -> Dict[str, Any]:
    exp1 = json.loads((run_dir / "exp1_conditioning.json").read_text(encoding="utf-8"))
    exp2 = json.loads((run_dir / "exp2_main_benchmark.json").read_text(encoding="utf-8"))
    abl = json.loads((run_dir / "exp2_ablations.json").read_text(encoding="utf-8"))
    exp3 = json.loads((run_dir / "exp3_policy_transfer.json").read_text(encoding="utf-8"))
    exp4 = json.loads((run_dir / "exp4_camotics_and_haas.json").read_text(encoding="utf-8"))

    out: Dict[str, Any] = {}

    # 1. Exp1 Stiffness Summary
    k_summary = {}
    for k in sorted({float(r["k"]) for r in exp1["stiffness_sweep"]}):
        sub = [r for r in exp1["stiffness_sweep"] if float(r["k"]) == k]
        k_summary[str(int(k))] = {
            "cosine_mean": round(float(np.mean([r["cosine_vs_hard_fd"] for r in sub])), 3),
            "snr_mean": round(float(np.mean([r["local_grad_snr"] for r in sub])), 2),
            "occ_l1_pct": round(100.0 * float(np.mean([r["soft_hard_occupancy_l1"] for r in sub])), 3),
        }
    out["exp1_stiffness"] = k_summary

    reach_summary = {}
    for r_val in sorted({float(r["perturbation_R"]) for r in exp1["reachability_sweep"]}):
        reach_summary[f"R={r_val:.2f}"] = {}
        for m in ("STE-Continuation-Adam", "Hard-CMA-ES"):
            sub = [x for x in exp1["reachability_sweep"] if float(x["perturbation_R"]) == r_val and x["method"] == m]
            gains = [100.0 * (x["init_cycle_time_s"] - x["final_cycle_time_s"]) / x["init_cycle_time_s"] for x in sub]
            reach_summary[f"R={r_val:.2f}"][m] = {
                "feasible_pct": round(100.0 * float(np.mean([1.0 if x["final_feasible"] else 0.0 for x in sub])), 1),
                "gain_pct_mean": round(float(np.mean(gains)), 1),
            }
    out["exp1_reachability"] = reach_summary

    # 2. Exp2 Main Benchmark (All 16 tasks + per family + test split)
    exp2_table: Dict[str, Any] = {}
    for m in METHODS:
        all_gains: List[float] = []
        all_times: List[float] = []
        all_mrr: List[float] = []
        all_feas: List[float] = []
        all_wall: List[float] = []
        fam_gains: Dict[str, Tuple[float, float]] = {}
        for fam in FAMILIES:
            fg = []
            for t in exp2["tasks"]:
                if t["family"] != fam:
                    continue
                for r in t["methods"][m]:
                    fg.append(float(r["cycle_time_reduction_pct"]))
                    all_gains.append(float(r["cycle_time_reduction_pct"]))
                    all_times.append(float(r["cycle_time_s"]))
                    all_mrr.append(float(r["peak_mrr_mm3_s"]))
                    all_feas.append(1.0 if r["feasible"] else 0.0)
                    all_wall.append(float(r["wall_time_s"]))
            fam_gains[fam] = (round(float(np.mean(fg)), 1), round(float(np.std(fg)), 1))
        exp2_table[m] = {
            "overall_gain_mean": round(float(np.mean(all_gains)), 1),
            "overall_gain_std": round(float(np.std(all_gains)), 1),
            "mean_cycle_time_s": round(float(np.mean(all_times)), 2),
            "mean_peak_mrr_mm3_s": round(float(np.mean(all_mrr)), 1),
            "feasible_rate_pct": round(100.0 * float(np.mean(all_feas)), 1),
            "mean_wall_s": round(float(np.mean(all_wall)), 2),
            "by_family_gain_pct": fam_gains,
        }
    out["exp2_main"] = exp2_table

    # 3. Exp2 Ablations
    abl_table: Dict[str, Any] = {}
    for arm in (
        "Full-Joint-Blended-STE",
        "Path-Only-Frozen-Feed",
        "Feed-Only-Frozen-Path",
        "Exact-Stop-Surrogate",
        "No-MRR-Barrier",
    ):
        sub = [r for r in abl["rows"] if r["arm"] == arm]
        abl_table[arm] = {
            "gain_mean": round(float(np.mean([r["cycle_time_reduction_pct"] for r in sub])), 1),
            "gain_std": round(float(np.std([r["cycle_time_reduction_pct"] for r in sub])), 1),
            "peak_mrr_mean": round(float(np.mean([r["peak_mrr_mm3_s"] for r in sub])), 1),
            "mrr_overload_pct_mean": round(float(np.mean([r["mrr_overload_pct"] for r in sub])), 1),
            "feasible_pct": round(100.0 * float(np.mean([1.0 if r["feasible"] else 0.0 for r in sub])), 1),
        }
    out["exp2_ablations"] = abl_table

    # 4. Exp3 Transfer
    exp3_table: Dict[str, Any] = {}
    for reg in (
        "DiffPolicy-SHAC-ZeroShot",
        "Hard-PPO-ZeroShot",
        "DiffPolicy-SHAC-FewShot16",
        "Hard-PPO-FewShot16",
        "Scratch-STE-Adam-16",
    ):
        sub = [r for r in exp3["test_transfer"] if r["regime"] == reg]
        ovl_vals = [float(r.get("mrr_overload_pct", 0.0)) for r in sub]
        fam_g3 = {
            fam: round(float(np.mean([r["cycle_time_reduction_pct"] for r in sub if r["family"] == fam])), 1)
            for fam in FAMILIES
        }
        exp3_table[reg] = {
            "gain_mean": round(float(np.mean([r["cycle_time_reduction_pct"] for r in sub])), 1),
            "gain_std": round(float(np.std([r["cycle_time_reduction_pct"] for r in sub])), 1),
            "feasible_pct": round(100.0 * float(np.mean([1.0 if r["feasible"] else 0.0 for r in sub])), 1),
            "peak_mrr_mean": round(float(np.mean([r["peak_mrr_mm3_s"] for r in sub])), 1),
            "mrr_overload_pct_mean": round(float(np.mean(ovl_vals)), 1),
            "by_family_gain_pct": fam_g3,
        }
    out["exp3_transfer"] = exp3_table

    # 5. Exp4 CAMotics + Haas NGC + 6-Controller Audit
    ver = exp4["verifications"]
    diff_v = [v for v in ver if v["variant"] == "DiffCAM-STE"]
    nom_v = [v for v in ver if v["variant"] == "Nominal-CAM"]
    mc_rows = exp4.get("multi_controller_audit", [])
    mc_summary: Dict[str, Any] = {}
    if mc_rows:
        dialects = sorted({str(r["dialect"]) for r in mc_rows})
        mc_summary = {
            "total_programs": len(mc_rows),
            "geometry_pass_count": sum(1 for r in mc_rows if r["geometry_pass"]),
            "safety_pass_count": sum(1 for r in mc_rows if r["safety_pass"]),
            "max_orig_gouge_mm3": round(max(float(r["orig_gouge_mm3"]) for r in mc_rows), 4),
            "by_dialect": {
                d: {
                    "count": sum(1 for r in mc_rows if r["dialect"] == d),
                    "pass_count": sum(1 for r in mc_rows if r["dialect"] == d and r["geometry_pass"] and r["safety_pass"]),
                    "mean_cycle_time_s": round(float(np.mean([r["cycle_time_s"] for r in mc_rows if r["dialect"] == d])), 2),
                }
                for d in dialects
            },
        }
    out["exp4_verification"] = {
        "num_test_cases": len(diff_v),
        "posted_0p125mm_geom_pass_count": sum(1 for v in diff_v if v["posted_0p125mm_geometry_pass"]),
        "posted_0p125mm_safety_pass_count": sum(1 for v in diff_v if v["posted_0p125mm_safety_pass"]),
        "max_posted_orig_gouge_mm3": round(max(float(v["posted_0p125mm_orig_gouge_mm3"]) for v in diff_v), 4),
        "max_posted_residual_mm3": round(max(float(v["posted_0p125mm_residual_mm3"]) for v in diff_v), 4),
        "camotics_watertight_count": sum(1 for v in diff_v if v["camotics_watertight"]),
        "mean_nominal_time_s": round(float(np.mean([v["cycle_time_s"] for v in nom_v])), 2),
        "mean_diffcam_time_s": round(float(np.mean([v["cycle_time_s"] for v in diff_v])), 2),
        "multi_controller_summary": mc_summary,
        "per_case": exp4["haas_packets"],
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "pub-20261110")
    args = parser.parse_args()
    summary = summarize(args.run_dir.resolve())
    out_file = args.run_dir.resolve() / "summary_metrics.json"
    out_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
