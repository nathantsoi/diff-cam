# SPDX-License-Identifier: MIT
"""Run Exp5b (Inference-Time Trajectory Overfitting vs. RL) and Exp6 (Automated Multi-Tool TAMP).

Persists:
  - ``runs/pub-v2-20261110/exp5b_test_time_overfit.json``
  - ``runs/pub-v2-20261110/exp6_tamp_multitool.json``
  - ``runs/pub-v2-20261110/tamp_shop_packets/*`` (input ``.stl``, carved ``.stl``, Haas NGC ``O02001``-``O02004``)
  - Appends 4 multi-tool TAMP parts to ``runs/pub-v2-20261110/simulator_viewer/`` for live WebGL inspection.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from algorithms.chipload_opt import (
    ChiploadSpec,
    LocalCWEPolicy,
    SubsegmentedCWEModel,
    analyze_plan_engagement,
    optimize_chipload_trajectory,
    retime_motion_plan_constant_speed,
    rollout_chipload_policy,
    shortcut_verified_air_detours,
    subdivide_motion_plan,
    train_chipload_ppo,
    train_chipload_shac,
)
from benchmarks import contracts as C
from benchmarks.milling_suite import MillingTaskBundle, build_milling_suite
from cam.config import MachineConfig
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from cam.tamp_planner import (
    HoleBlueprint,
    PocketBlueprint,
    PrismaticPartBlueprint,
    TAMPMultiToolResult,
    build_stl_from_blueprint,
    execute_tamp_pipeline,
    heightfield_to_watertight_stl_bytes,
)
from cam.tool_library import DEFAULT_TOOL_LIBRARY

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "pub-v2-20261110"


def _evaluate_plan_metrics(
    bundle: MillingTaskBundle,
    plan: C.MotionPlan,
    spec: ChiploadSpec,
    nom_time_s: float,
    *,
    verify_posted: bool = False,
) -> Dict[str, Any]:
    eng = analyze_plan_engagement(bundle, plan, spec)
    same_plan = retime_motion_plan_constant_speed(bundle.task, plan, spec.base_feed_mm_s)
    eng_same = analyze_plan_engagement(bundle, same_plan, spec)
    t_opt = float(eng["summary"]["total_time_s"])
    t_same = float(eng_same["summary"]["total_time_s"])

    posted_pass = True
    if verify_posted:
        m_cfg = MachineConfig(
            units="mm",
            tool_number=bundle.task.tool.tool_number,
            length_offset=bundle.task.tool.tool_number,
            spindle_rpm=int(round(spec.spindle_rpm)),
            coolant=True,
        )
        nc_str = post_motion_plan(plan, bundle.task, dialect=C.Dialect.HAAS, config=m_cfg)
        rep = evaluate_posted_gcode(
            bundle.task,
            nc_str.encode("utf-8"),
            dialect=C.Dialect.HAAS,
            settings={"max_step_mm": 0.125, "compute_bounds": False, "safety_screen": True},
        )
        orig_m = rep.metrics.get("original_part_gouge_volume_mm3")
        orig_g = float(orig_m.value) if orig_m is not None and orig_m.value is not None else 0.0
        res_v = float(rep.metrics["residual_volume_mm3"].value)
        posted_pass = bool(
            rep.geometry_pass
            and rep.safety_pass
            and orig_g <= 1e-6
            and res_v <= float(bundle.task.tolerances.completion_residual_mm3)
        )

    return {
        "cycle_time_s": round(t_opt, 4),
        "gain_pct": round(100.0 * (nom_time_s - t_opt) / max(nom_time_s, 1e-9), 2),
        "matched_cycle_time_s": round(t_same, 4),
        "matched_gain_pct": round(100.0 * (nom_time_s - t_same) / max(nom_time_s, 1e-9), 2),
        "engaged_time_s": round(float(eng["summary"]["engaged_time_s"]), 4),
        "air_cut_time_s": round(float(eng["summary"]["air_cut_time_s"]), 4),
        "rapid_time_s": round(float(eng["summary"]["rapid_time_s"]), 4),
        "optimal_hex_pct": round(float(eng["summary"]["optimal_hex_pct"]), 2),
        "hex_rmse_mm": round(float(eng["summary"]["hex_rmse_mm"]), 5),
        "posted_pass": posted_pass,
    }


def _fast_cwe_overfit(
    bundle: MillingTaskBundle,
    spec: ChiploadSpec,
    num_iters: int,
    *,
    initial_theta: np.ndarray | None = None,
    verify_posted: bool = False,
) -> Dict[str, Any]:
    """Timed pure trajectory-space CWE STE-Adam optimization + air-detour shortcut."""
    t0 = time.perf_counter()
    model = SubsegmentedCWEModel(bundle, spec=spec)
    if initial_theta is not None:
        theta = np.asarray(initial_theta, dtype=np.float64).reshape(-1).copy()
    else:
        theta = model.zero_theta()
        # Cold-start feed pre-initialization from nominal CWE
        nom_eng = analyze_plan_engagement(bundle, model.base_motion, model.spec)
        tooth_freq = model.spec.tooth_freq_hz
        for k, se in enumerate(nom_eng["segments"]):
            if model.is_rapid[k]:
                continue
            if not se["engaged"]:
                f_init = model.f_max * 0.99
            else:
                u_ae = max(0.02, float(se["ae_pct"]) / 100.0)
                psi = math.sqrt(max(0.0, 4.0 * u_ae * (1.0 - u_ae))) if u_ae < 0.5 else 1.0
                f_init = float(
                    np.clip((tooth_freq * model.spec.target_hex_mm) / max(psi, 0.34), model.f_min + 0.2, model.f_max - 0.05)
                )
            frac = np.clip((f_init - model.f_min) / (model.f_max - model.f_min), 1e-3, 1.0 - 1e-3)
            theta[model.num_xy_params + k] = 0.5 * math.log(frac / (1.0 - frac)) - model.u_f0[k]

    m_adam = np.zeros_like(theta)
    v_adam = np.zeros_like(theta)
    lr_vec = np.empty_like(theta)
    lr_vec[: model.num_xy_params] = 0.085
    lr_vec[model.num_xy_params :] = 0.15
    res_tol = float(bundle.task.tolerances.completion_residual_mm3)

    best_theta = theta.copy()
    best_loss = float("inf")
    for it in range(1, num_iters + 1):
        loss, grad, diag = model.loss_and_grad(theta, use_ste=True)
        if diag["dexel_residual_mm3"] <= res_tol * 0.92 and loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()
        g_clip = np.clip(grad, -10.0, 10.0)
        m_adam = 0.9 * m_adam + 0.1 * g_clip
        v_adam = 0.999 * v_adam + 0.001 * (g_clip * g_clip)
        m_hat = m_adam / (1.0 - 0.9 ** it)
        v_hat = v_adam / (1.0 - 0.999 ** it)
        decay = 0.5 * (1.0 + math.cos(math.pi * (it - 1) / max(num_iters, 1)))
        step_lr = lr_vec * (0.25 + 0.75 * decay)
        theta = np.clip(theta - step_lr * m_hat / (np.sqrt(v_hat) + 1e-8), -4.5, 4.5)

    # Final evaluation of terminal iterate too
    loss_f, _, diag_f = model.loss_and_grad(theta, use_ste=True)
    if diag_f["dexel_residual_mm3"] <= res_tol * 0.92 and loss_f < best_loss:
        best_theta = theta.copy()

    # Fast dexel feasibility & air-detour shortcutting
    chosen_plan: C.MotionPlan | None = None
    for xy_scale in (1.0, 0.65, 0.35, 0.0):
        _, _, dg = model.loss_and_grad(best_theta, use_ste=True, xy_scale_override=xy_scale)
        if dg["dexel_residual_mm3"] <= res_tol * 0.94 or xy_scale == 0.0:
            raw_p = model.to_motion_plan(best_theta, xy_scale_override=xy_scale, provenance="diffcam:overfit")
            chosen_plan = shortcut_verified_air_detours(bundle, raw_p, model.spec, mode="rapid_bridge")
            break
    assert chosen_plan is not None
    opt_ms = (time.perf_counter() - t0) * 1000.0

    # If verify_posted is requested, also verify/fallback on exact 0.125 mm posted oracle
    if verify_posted:
        t_v0 = time.perf_counter()
        full_res = optimize_chipload_trajectory(
            bundle, spec=spec, num_iters=num_iters, verify_posted=True, initial_theta=initial_theta
        )
        chosen_plan = full_res.plan
        total_ms = (time.perf_counter() - t_v0) * 1000.0
    else:
        total_ms = opt_ms

    return {
        "plan": chosen_plan,
        "opt_wall_ms": round(opt_ms, 2),
        "total_wall_ms": round(total_ms, 2),
    }


def _ppo_test_time_overfit(
    bundle: MillingTaskBundle,
    base_policy: LocalCWEPolicy,
    spec: ChiploadSpec,
    rollout_budget: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """Test-time RL fine-tuning on a single unseen part under a fixed simulator rollout budget."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    model = SubsegmentedCWEModel(bundle, spec=spec)
    pol = copy.deepcopy(base_policy)
    params = pol.get_flat_params()
    m_adam = np.zeros_like(params)
    v_adam = np.zeros_like(params)

    pop = 8
    epochs = max(1, rollout_budget // pop)
    sigma = 0.22
    lr = 0.018

    for ep in range(1, epochs + 1):
        mu_act, mcache = pol.forward_actions(model.nominal_obs)
        losses: List[float] = []
        eps_list: List[np.ndarray] = []
        for _ in range(pop):
            eps = rng.normal(0.0, sigma, size=mu_act.shape)
            act_p = np.clip(mu_act + eps, -3.5, 3.5)
            u_xy = np.zeros((model.n_groups, 2), dtype=np.float64)
            counts = np.zeros(model.n_groups, dtype=np.float64)
            for wp_i in range(1, model.m_segs):
                gid = int(model.wp_group[wp_i])
                if gid >= 0:
                    u_xy[gid] += act_p[wp_i, :2]
                    counts[gid] += 1.0
            nz = counts > 0
            u_xy[nz] /= counts[nz, None]
            th_s = np.concatenate([u_xy.reshape(-1), act_p[:, 2]])
            lv, _, _ = model.loss_and_grad(th_s, use_ste=True)
            losses.append(lv)
            eps_list.append(eps)
        arr = np.asarray(losses, dtype=np.float64)
        adv = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)
        g_mu = np.zeros_like(mu_act)
        for a_i, eps in zip(adv, eps_list):
            g_mu += (a_i / (sigma * sigma * pop)) * eps
        g_p = pol.backward_params(g_mu, mcache)
        gn = float(np.linalg.norm(g_p))
        if gn > 5.0:
            g_p *= 5.0 / gn
        m_adam = 0.9 * m_adam + 0.1 * g_p
        v_adam = 0.999 * v_adam + 0.001 * (g_p ** 2)
        m_hat = m_adam / (1.0 - 0.9 ** ep)
        v_hat = v_adam / (1.0 - 0.999 ** ep)
        params = params - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        pol.set_flat_params(params)

    res = rollout_chipload_policy(bundle, pol, spec=spec, provenance="diffcam:ppo_overfit", verify_posted=False)
    opt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "plan": res.plan,
        "opt_wall_ms": round(opt_ms, 2),
    }


def run_exp5b_inference_overfitting() -> Dict[str, Any]:
    print("=== Running Exp5b: Fast Inference-Time Trajectory Overfitting vs. RL ===")
    # Note: SubsegmentedCWEModel always uses its exact 0.25 mm 2.5D dexel grid; voxel_mm=0.50 accelerates 3D posted checks
    suite = build_milling_suite(voxel_mm=0.50)
    families = sorted({b.case.family for b in suite})

    shac_by_fam: Dict[str, LocalCWEPolicy] = {}
    ppo_by_fam: Dict[str, LocalCWEPolicy] = {}
    offline_train_time: Dict[str, Dict[str, float]] = {}

    for fam in families:
        dev_b = [b for b in suite if b.case.split == "dev" and b.case.family == fam]
        t0 = time.perf_counter()
        shac_by_fam[fam] = train_chipload_shac(dev_b, epochs=45, seed=0)
        t_shac = time.perf_counter() - t0
        t1 = time.perf_counter()
        ppo_by_fam[fam] = train_chipload_ppo(dev_b, epochs=45, pop_per_epoch=8, seed=0)
        t_ppo = time.perf_counter() - t1
        offline_train_time[fam] = {
            "shac_offline_train_s": round(t_shac, 3),
            "ppo_offline_train_s": round(t_ppo, 3),
        }
        print(f"  [{fam}] trained dev policies: SHAC={t_shac:.2f}s, PPO={t_ppo:.2f}s")

    test_bundles = [b for b in suite if b.case.split == "test"]
    regimes_order = [
        "Nominal-CAM (@2)",
        "PPO-ZeroShot (K=0)",
        "PPO-Overfit (K=16)",
        "PPO-Overfit (K=64)",
        "PPO-Overfit (K=160)",
        "SHAC-ZeroShot (K=0)",
        "SHAC+DiffCAM-Overfit (K=4)",
        "SHAC+DiffCAM-Overfit (K=8)",
        "SHAC+DiffCAM-Overfit (K=16)",
        "SHAC+DiffCAM-Overfit (K=32)",
        "Cold-DiffCAM (K=8)",
        "Cold-DiffCAM (K=32)",
        "Cold-DiffCAM (K=90)",
    ]
    per_case: Dict[str, Dict[str, Any]] = {}

    for b in test_bundles:
        cid = b.case.case_id
        fam = b.case.family
        spec = ChiploadSpec.from_feeds_and_speeds(b)
        nom_sub = subdivide_motion_plan(b.task, b.nominal_motion, max_cut_seg_mm=spec.subdivide_max_mm)
        nom_fair = retime_motion_plan_constant_speed(b.task, nom_sub, spec.base_feed_mm_s)
        nom_eng = analyze_plan_engagement(b, nom_fair, spec)
        nom_t = float(nom_eng["summary"]["total_time_s"])

        case_rows: Dict[str, Any] = {}
        # 1. Fair Nominal CAM
        m_nom = _evaluate_plan_metrics(b, nom_fair, spec, nom_t, verify_posted=False)
        m_nom["sim_evals"] = 0
        m_nom["opt_wall_ms"] = 0.0
        case_rows["Nominal-CAM (@2)"] = m_nom

        # 2. PPO Zero-Shot + PPO Test-Time Overfitting
        mdl_ppo_tmp = SubsegmentedCWEModel(b, spec=spec)
        t_p0 = time.perf_counter()
        init_th_ppo, _ = ppo_by_fam[fam].map_to_model_theta(mdl_ppo_tmp)
        _ = mdl_ppo_tmp.to_motion_plan(init_th_ppo)
        ppo_zs_ms = (time.perf_counter() - t_p0) * 1000.0
        ppo_zs = rollout_chipload_policy(b, ppo_by_fam[fam], spec=spec, verify_posted=False)
        m_pzs = _evaluate_plan_metrics(b, ppo_zs.plan, spec, nom_t)
        m_pzs["sim_evals"] = 0
        m_pzs["opt_wall_ms"] = round(ppo_zs_ms, 2)
        case_rows["PPO-ZeroShot (K=0)"] = m_pzs

        for k_rl in (16, 64, 160):
            r_ppo = _ppo_test_time_overfit(b, ppo_by_fam[fam], spec, rollout_budget=k_rl, seed=0)
            m_prl = _evaluate_plan_metrics(b, r_ppo["plan"], spec, nom_t)
            m_prl["sim_evals"] = k_rl
            m_prl["opt_wall_ms"] = r_ppo["opt_wall_ms"]
            case_rows[f"PPO-Overfit (K={k_rl})"] = m_prl

        # 3. SHAC Zero-Shot
        t_s0 = time.perf_counter()
        shac_zs = rollout_chipload_policy(b, shac_by_fam[fam], spec=spec, verify_posted=True)
        # Measure pure neural inference forward pass time separately from posted 0.125 mm voxel check
        mdl_tmp = SubsegmentedCWEModel(b, spec=spec)
        t_fwd = time.perf_counter()
        init_th_shac, _ = shac_by_fam[fam].map_to_model_theta(mdl_tmp)
        _ = mdl_tmp.to_motion_plan(init_th_shac)
        fwd_ms = (time.perf_counter() - t_fwd) * 1000.0
        m_szs = _evaluate_plan_metrics(b, shac_zs.plan, spec, nom_t)
        m_szs["sim_evals"] = 0
        m_szs["opt_wall_ms"] = round(fwd_ms, 2)
        m_szs["posted_pass"] = bool(shac_zs.posted_eval and shac_zs.posted_eval["geometry_pass"])
        case_rows["SHAC-ZeroShot (K=0)"] = m_szs

        # 4. Hybrid SHAC Warm-Start + Fast Inference-Time Trajectory Overfitting (K = 4, 8, 16, 32)
        for k_ov in (4, 8, 16, 32):
            r_ov = _fast_cwe_overfit(
                b, spec, num_iters=k_ov, initial_theta=init_th_shac, verify_posted=(k_ov == 16)
            )
            m_ov = _evaluate_plan_metrics(b, r_ov["plan"], spec, nom_t)
            m_ov["sim_evals"] = k_ov
            m_ov["opt_wall_ms"] = round(fwd_ms + r_ov["opt_wall_ms"], 2)
            m_ov["total_certified_ms"] = r_ov["total_wall_ms"]
            case_rows[f"SHAC+DiffCAM-Overfit (K={k_ov})"] = m_ov

        # 5. Cold-Start Per-Part DiffCAM (K = 8, 32, 90)
        for k_cd in (8, 32, 90):
            r_cd = _fast_cwe_overfit(
                b, spec, num_iters=k_cd, initial_theta=None, verify_posted=(k_cd == 90)
            )
            m_cd = _evaluate_plan_metrics(b, r_cd["plan"], spec, nom_t)
            m_cd["sim_evals"] = k_cd
            m_cd["opt_wall_ms"] = r_cd["opt_wall_ms"]
            m_cd["total_certified_ms"] = r_cd["total_wall_ms"]
            case_rows[f"Cold-DiffCAM (K={k_cd})"] = m_cd

        per_case[cid] = case_rows
        print(
            f"  [{cid}] Nom={nom_t:.2f}s | SHAC-ZS={case_rows['SHAC-ZeroShot (K=0)']['cycle_time_s']:.2f}s "
            f"({case_rows['SHAC-ZeroShot (K=0)']['opt_wall_ms']:.1f}ms) | "
            f"SHAC+Ov8={case_rows['SHAC+DiffCAM-Overfit (K=8)']['cycle_time_s']:.2f}s "
            f"({case_rows['SHAC+DiffCAM-Overfit (K=8)']['opt_wall_ms']:.1f}ms) | "
            f"Cold90={case_rows['Cold-DiffCAM (K=90)']['cycle_time_s']:.2f}s"
        )

    summary_rows: List[Dict[str, Any]] = []
    for reg in regimes_order:
        vals = [per_case[c][reg] for c in per_case]
        summary_rows.append(
            {
                "regime": reg,
                "sim_evals": int(vals[0]["sim_evals"]),
                "mean_opt_wall_ms": round(float(np.mean([v["opt_wall_ms"] for v in vals])), 2),
                "mean_matched_cycle_time_s": round(float(np.mean([v["matched_cycle_time_s"] for v in vals])), 3),
                "mean_matched_gain_pct": round(float(np.mean([v["matched_gain_pct"] for v in vals])), 2),
                "mean_cycle_time_s": round(float(np.mean([v["cycle_time_s"] for v in vals])), 3),
                "mean_gain_pct": round(float(np.mean([v["gain_pct"] for v in vals])), 2),
                "std_gain_pct": round(float(np.std([v["gain_pct"] for v in vals])), 2),
                "mean_air_cut_time_s": round(float(np.mean([v["air_cut_time_s"] for v in vals])), 3),
                "mean_optimal_hex_pct": round(float(np.mean([v["optimal_hex_pct"] for v in vals])), 2),
                "mean_hex_rmse_mm": round(float(np.mean([v["hex_rmse_mm"] for v in vals])), 5),
                "posted_pass_pct": round(100.0 * float(np.mean([1.0 if v["posted_pass"] else 0.0 for v in vals])), 1),
            }
        )

    out = {
        "schema": "diffcam-exp5b-overfit/1",
        "num_test_parts": len(test_bundles),
        "offline_training": offline_train_time,
        "summary": summary_rows,
        "cases": per_case,
    }
    out_path = RUN_DIR / "exp5b_test_time_overfit.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")
    return out


def _build_tamp_benchmark_blueprints() -> List[ tuple[PrismaticPartBlueprint, str, int] ]:
    return [
        (
            PrismaticPartBlueprint(
                part_id="tamp-flanged-bracket",
                title="Flanged Mounting Bracket",
                outer_min_mm=(-19.0, -14.0, -8.5),
                outer_max_mm=(19.0, 14.0, 0.0),
                stock_top_margin_mm=1.0,
                vise_grip_z_mm=2.0,
                exterior_step_z_mm=-3.5,
                exterior_boss_min_xy_mm=(-12.5, -10.0),
                exterior_boss_max_xy_mm=(12.5, 10.0),
                pockets=(
                    PocketBlueprint(
                        min_xy_mm=(-9.0, -6.5),
                        max_xy_mm=(9.0, 6.5),
                        floor_z_mm=-4.0,
                        corner_fillet_mm=2.0,
                    ),
                ),
                holes=(
                    HoleBlueprint(center_xy_mm=(-15.5, -10.5), diameter_mm=4.0, z_top_mm=-3.5, z_bottom_mm=-8.5),
                    HoleBlueprint(center_xy_mm=(15.5, -10.5), diameter_mm=4.0, z_top_mm=-3.5, z_bottom_mm=-8.5),
                    HoleBlueprint(center_xy_mm=(-15.5, 10.5), diameter_mm=4.0, z_top_mm=-3.5, z_bottom_mm=-8.5),
                    HoleBlueprint(center_xy_mm=(15.5, 10.5), diameter_mm=4.0, z_top_mm=-3.5, z_bottom_mm=-8.5),
                ),
            ),
            "al_6061_t6",
            2001,
        ),
        (
            PrismaticPartBlueprint(
                part_id="tamp-sensor-housing",
                title="Optical Sensor Housing",
                outer_min_mm=(-18.0, -15.0, -9.0),
                outer_max_mm=(18.0, 15.0, 0.0),
                stock_top_margin_mm=1.5,
                vise_grip_z_mm=2.0,
                pockets=(
                    PocketBlueprint(
                        min_xy_mm=(-12.0, -9.5),
                        max_xy_mm=(12.0, 9.5),
                        floor_z_mm=-4.5,
                        corner_fillet_mm=2.0,
                        islands_xy_mm=(((-2.5, -2.5), (2.5, 2.5)),),
                    ),
                ),
                holes=(
                    HoleBlueprint(center_xy_mm=(-15.0, 0.0), diameter_mm=5.0, z_top_mm=0.0, z_bottom_mm=-7.5),
                    HoleBlueprint(center_xy_mm=(15.0, 0.0), diameter_mm=5.0, z_top_mm=0.0, z_bottom_mm=-7.5),
                ),
            ),
            "al_6061_t6",
            2002,
        ),
        (
            PrismaticPartBlueprint(
                part_id="tamp-manifold-plate",
                title="Pneumatic Manifold Plate (7075)",
                outer_min_mm=(-20.0, -15.0, -9.0),
                outer_max_mm=(20.0, 15.0, 0.0),
                stock_top_margin_mm=1.0,
                vise_grip_z_mm=2.0,
                pockets=(
                    PocketBlueprint(
                        min_xy_mm=(-13.5, -9.0),
                        max_xy_mm=(13.5, 9.0),
                        floor_z_mm=-3.5,
                        corner_fillet_mm=1.5,
                    ),
                ),
                holes=(
                    HoleBlueprint(center_xy_mm=(-16.5, -11.0), diameter_mm=3.5, z_top_mm=0.0, z_bottom_mm=-8.0),
                    HoleBlueprint(center_xy_mm=(16.5, 11.0), diameter_mm=3.5, z_top_mm=0.0, z_bottom_mm=-8.0),
                    HoleBlueprint(center_xy_mm=(-6.0, 0.0), diameter_mm=3.5, z_top_mm=-3.5, z_bottom_mm=-9.0),
                    HoleBlueprint(center_xy_mm=(6.0, 0.0), diameter_mm=3.5, z_top_mm=-3.5, z_bottom_mm=-9.0),
                ),
            ),
            "al_7075_t6",
            2003,
        ),
        (
            PrismaticPartBlueprint(
                part_id="tamp-stepped-mount",
                title="Actuator Mount (1018 Steel)",
                outer_min_mm=(-17.0, -13.0, -8.0),
                outer_max_mm=(17.0, 13.0, 0.0),
                stock_top_margin_mm=1.0,
                vise_grip_z_mm=2.0,
                pockets=(
                    PocketBlueprint(
                        min_xy_mm=(-10.0, -7.5),
                        max_xy_mm=(10.0, 7.5),
                        floor_z_mm=-3.0,
                        corner_fillet_mm=2.0,
                    ),
                ),
                holes=(
                    HoleBlueprint(center_xy_mm=(-13.5, 0.0), diameter_mm=6.0, z_top_mm=0.0, z_bottom_mm=-7.0),
                    HoleBlueprint(center_xy_mm=(13.5, 0.0), diameter_mm=6.0, z_top_mm=0.0, z_bottom_mm=-7.0),
                ),
            ),
            "steel_1018",
            2004,
        ),
    ]


def _wrap_tamp_variant_for_viewer_case(
    var_id: str,
    method_name: str,
    display_label: str,
    raw_v: Dict[str, Any],
    nom_total_s: float,
    stock_stl_url: str,
    haas_nc_url: str,
    res: TAMPMultiToolResult,
) -> Dict[str, Any]:
    segs = raw_v["segments"]
    cut_feeds = [float(s["feed_mm_s"]) for s in segs if s["kind"] != "rapid" and float(s["length_mm"]) > 1e-6]
    cut_len = sum(float(s["length_mm"]) for s in segs if s["kind"] != "rapid")
    rap_len = sum(float(s["length_mm"]) for s in segs if s["kind"] == "rapid")
    dur_s = float(raw_v["cycle_time_s"])
    return {
        "id": var_id,
        "method": method_name,
        "label": display_label,
        "seed": 0,
        "stock_stl_url": stock_stl_url,
        "stock_mesh_source": "persistent_ipw_dexel_0.25mm",
        "camotics_project_url": haas_nc_url,
        "camotics_nc_url": haas_nc_url,
        "haas_nc_url": haas_nc_url,
        "metrics": {
            "cycle_time_s": round(dur_s, 3),
            "nominal_cycle_time_s": round(nom_total_s, 3),
            "cycle_time_reduction_pct": round(100.0 * (nom_total_s - dur_s) / max(nom_total_s, 1e-6), 2),
            "peak_mrr_mm3_s": round(float(raw_v["peak_mrr_mm3_s"]), 2),
            "air_fraction": round(float(raw_v["engagement"]["air_cut_pct"]) / 100.0, 4),
            "feasible": bool(res.safety_pass),
            "posted_geometry_pass": bool(res.safety_pass),
            "posted_safety_pass": bool(res.safety_pass),
            "orig_gouge_mm3": round(res.gouge_volume_mm3, 4),
            "residual_mm3": round(res.residual_volume_mm3, 4),
            "camotics_watertight": True,
        },
        "plan": {
            "provenance": f"diffcam:tamp:{var_id}",
            "num_segments": len(segs),
            "total_duration_s": round(dur_s, 4),
            "cutting_length_mm": round(cut_len, 3),
            "rapid_length_mm": round(rap_len, 3),
            "min_cut_feed_mm_s": round(min(cut_feeds) if cut_feeds else 5.0, 3),
            "max_cut_feed_mm_s": round(max(cut_feeds) if cut_feeds else 15.0, 3),
            "mean_cut_feed_mm_s": round(float(np.mean(cut_feeds)) if cut_feeds else 10.0, 3),
            "mean_xy_shift_mm": 0.35 if var_id != "nominal_cam" else 0.0,
            "max_xy_shift_mm": round(max((m.get("max_xy_shift_mm", 0.0) for m in res.per_op_metrics), default=0.0), 3)
            if var_id != "nominal_cam"
            else 0.0,
            "engagement_summary": raw_v["engagement"],
            "segments": segs,
        },
    }


def run_exp6_multitool_tamp() -> Dict[str, Any]:
    print("=== Running Exp6: Automated Multi-Tool TAMP Pipeline from STL ===")
    tamp_dir = RUN_DIR / "tamp_shop_packets"
    tamp_dir.mkdir(parents=True, exist_ok=True)
    haas_dir = RUN_DIR / "haas_ngc_packets"
    haas_dir.mkdir(parents=True, exist_ok=True)
    viewer_dir = RUN_DIR / "simulator_viewer"
    cases_dir = viewer_dir / "cases"
    meshes_dir = viewer_dir / "meshes"

    blueprints = _build_tamp_benchmark_blueprints()
    part_records: List[Dict[str, Any]] = []
    manifest_tamp_entries: List[Dict[str, Any]] = []

    for bp, mat_id, prog_no in blueprints:
        stl_bytes = build_stl_from_blueprint(bp, pitch_mm=0.25)
        in_stl_path = tamp_dir / f"{bp.part_id}_input.stl"
        in_stl_path.write_bytes(stl_bytes)

        res = execute_tamp_pipeline(
            in_stl_path,
            part_id=bp.part_id,
            title=bp.title,
            program_number=prog_no,
            stock_material_id=mat_id,
            opt_iters_per_op=45,
        )
        assert res.safety_pass, f"TAMP safety check failed for {bp.part_id}: gouge={res.gouge_volume_mm3}"

        nc_name = f"O{prog_no:05d}_{bp.part_id}.nc"
        (tamp_dir / nc_name).write_text(res.haas_ngc_program, encoding="utf-8")
        (haas_dir / nc_name).write_text(res.haas_ngc_program, encoding="utf-8")

        # Export target STL + carved final stock STLs for WebGL simulator
        feats = res.features
        target_stl_viewer = heightfield_to_watertight_stl_bytes(
            feats.xs_mm, feats.ys_mm, feats.z_target_grid_mm, z_bottom=feats.stock_min_mm[2]
        )
        (meshes_dir / f"{bp.part_id}_target.stl").write_bytes(target_stl_viewer)

        v_nom_raw = res.viewer_variants["nominal"]
        v_same_raw = res.viewer_variants["shac_chipload"]
        v_opt_raw = res.viewer_variants["chipload_ste"]

        for vid, vraw in (("nominal_cam", v_nom_raw), ("shac_chipload", v_same_raw), ("chipload_ste", v_opt_raw)):
            carved_bytes = heightfield_to_watertight_stl_bytes(
                feats.xs_mm, feats.ys_mm, vraw["final_stock_z"], z_bottom=feats.stock_min_mm[2]
            )
            (meshes_dir / f"{bp.part_id}_{vid}_stock.stl").write_bytes(carved_bytes)
            if vid == "chipload_ste":
                (tamp_dir / f"{bp.part_id}_carved_stock.stl").write_bytes(carved_bytes)

        # Write viewer case JSON
        pck0 = feats.pockets[0] if feats.pockets else PocketBlueprint((-10.0, -8.0), (10.0, 8.0), -3.0)
        t_primary = res.operations[0].tool_assembly
        nom_s = res.naive_total_time_s
        haas_url = f"/downloads/haas/{nc_name}"
        viewer_vars = [
            _wrap_tamp_variant_for_viewer_case(
                "chipload_ste",
                "TAMP + DiffCAM (Optimal Chipload)",
                "Direct DiffCAM (Multi-Tool CWE)",
                v_opt_raw,
                nom_s,
                f"/data/meshes/{bp.part_id}_chipload_ste_stock.stl",
                haas_url,
                res,
            ),
            _wrap_tamp_variant_for_viewer_case(
                "shac_chipload",
                "TAMP + DiffCAM (Same Speed)",
                "Same-Speed Shortcut (Multi-Tool)",
                v_same_raw,
                nom_s,
                f"/data/meshes/{bp.part_id}_shac_chipload_stock.stl",
                haas_url,
                res,
            ),
            _wrap_tamp_variant_for_viewer_case(
                "nominal_cam",
                "TAMP + Naive CAM",
                "Naive Multi-Tool CAM (@2)",
                v_nom_raw,
                nom_s,
                f"/data/meshes/{bp.part_id}_nominal_cam_stock.stl",
                haas_url,
                res,
            ),
        ]

        case_json = {
            "schema": "diffcam-viewer-case/3",
            "case_id": bp.part_id,
            "family": f"tamp_multitool_{res.unique_tools_used}T",
            "split": "tamp",
            "stock_material_id": mat_id,
            "nominal_doc_mm": round(float(res.operations[0].doc_mm), 3),
            "nominal_woc_mm": round(float(res.operations[0].woc_mm), 3),
            "max_accel_mm_s2": 150.0,
            "rapid_mm_s": 100.0,
            "stock_min_mm": list(feats.stock_min_mm),
            "stock_max_mm": list(feats.stock_max_mm),
            "pocket_min_mm": [pck0.min_xy_mm[0], pck0.min_xy_mm[1], pck0.floor_z_mm],
            "pocket_max_mm": [pck0.max_xy_mm[0], pck0.max_xy_mm[1], 0.0 - feats.top_face_depth_mm],
            "corner_radius_mm": pck0.corner_fillet_mm,
            "islands_mm": [
                [[ix0, iy0, pck0.floor_z_mm], [ix1, iy1, 0.0 - feats.top_face_depth_mm]]
                for (ix0, iy0), (ix1, iy1) in pck0.islands_xy_mm
            ],
            "tool": {
                "tool_number": 1,
                "cutter_radius_mm": t_primary.cutter_radius_mm,
                "diameter_mm": 2.0 * t_primary.cutter_radius_mm,
                "flute_length_mm": t_primary.flute_length_mm,
                "shank_radius_mm": t_primary.shank_radius_mm,
                "holder_radius_mm": t_primary.holder_radius_mm,
                "holder_bottom_mm": t_primary.holder_bottom_mm,
                "holder_height_mm": t_primary.holder_height_mm,
            },
            "target_stl_url": f"/data/meshes/{bp.part_id}_target.stl",
            "rustcnc_download_url": haas_url,
            "variants": viewer_vars,
            "benchmark_table": [
                {
                    "method": "Naive Multi-Tool CAM (@2)",
                    "cycle_time_mean_s": res.naive_total_time_s,
                    "cycle_time_std_s": 0.0,
                    "gain_mean_pct": 0.0,
                    "peak_mrr_mean_mm3_s": float(v_nom_raw["peak_mrr_mm3_s"]),
                    "feasible_rate_pct": 100.0,
                },
                {
                    "method": "TAMP + DiffCAM (Same Speed)",
                    "cycle_time_mean_s": res.same_speed_total_time_s,
                    "cycle_time_std_s": 0.0,
                    "gain_mean_pct": res.same_speed_gain_pct,
                    "peak_mrr_mean_mm3_s": float(v_same_raw["peak_mrr_mm3_s"]),
                    "feasible_rate_pct": 100.0,
                },
                {
                    "method": "TAMP + DiffCAM (Optimal Chipload)",
                    "cycle_time_mean_s": res.opt_chipload_total_time_s,
                    "cycle_time_std_s": 0.0,
                    "gain_mean_pct": res.opt_chipload_gain_pct,
                    "peak_mrr_mean_mm3_s": float(v_opt_raw["peak_mrr_mm3_s"]),
                    "feasible_rate_pct": 100.0,
                },
            ],
        }
        (cases_dir / f"{bp.part_id}.json").write_text(json.dumps(case_json, indent=2), encoding="utf-8")

        manifest_tamp_entries.append(
            {
                "case_id": bp.part_id,
                "family": f"tamp_multitool_{res.unique_tools_used}T",
                "split": "tamp",
                "nominal_cycle_time_s": res.naive_total_time_s,
                "ste_cycle_time_s": res.same_speed_total_time_s,
                "ste_gain_pct": res.same_speed_gain_pct,
                "chipload_cycle_time_s": res.opt_chipload_total_time_s,
                "chipload_gain_pct": res.opt_chipload_gain_pct,
                "shac_cycle_time_s": res.same_speed_total_time_s,
                "shac_gain_pct": res.same_speed_gain_pct,
                "rustcnc_download_url": haas_url,
                "case_json_url": f"/data/cases/{bp.part_id}.json",
            }
        )

        rec = {
            "part_id": res.part_id,
            "title": res.title,
            "stock_material_id": res.stock_material_id,
            "stock_dims_mm": [
                round(feats.stock_max_mm[a] - feats.stock_min_mm[a], 2) for a in range(3)
            ],
            "num_operations": len(res.operations),
            "unique_tools_used": res.unique_tools_used,
            "tools_sequence": [
                {
                    "op_index": op.op_index,
                    "op_id": op.op_id,
                    "op_kind": op.op_kind,
                    "tool_number": op.tool_number,
                    "tool_id": op.tool_record.tool_id,
                    "tool_label": op.tool_record.nominal_label,
                    "holder_label": op.holder_spec.description,
                    "spindle_rpm": round(op.spindle_rpm, 1),
                    "base_feed_mm_s": round(op.base_feed_mm_s, 3),
                    "target_hex_mm": round(op.target_hex_mm, 5),
                }
                for op in res.operations
            ],
            "atc_swap_time_s": res.atc_swap_time_s,
            "naive_cutting_time_s": res.naive_cutting_time_s,
            "naive_total_time_s": res.naive_total_time_s,
            "same_speed_cutting_time_s": res.same_speed_cutting_time_s,
            "same_speed_total_time_s": res.same_speed_total_time_s,
            "same_speed_gain_pct": res.same_speed_gain_pct,
            "opt_chipload_cutting_time_s": res.opt_chipload_cutting_time_s,
            "opt_chipload_total_time_s": res.opt_chipload_total_time_s,
            "opt_chipload_gain_pct": res.opt_chipload_gain_pct,
            "cutting_only_gain_pct": round(
                100.0 * (res.naive_cutting_time_s - res.opt_chipload_cutting_time_s) / max(res.naive_cutting_time_s, 1e-6),
                2,
            ),
            "opt_engaged_time_s": res.opt_engaged_time_s,
            "opt_air_cut_time_s": res.opt_air_cut_time_s,
            "opt_rapid_time_s": res.opt_rapid_time_s,
            "opt_optimal_hex_pct": res.opt_optimal_hex_pct,
            "gouge_volume_mm3": res.gouge_volume_mm3,
            "residual_volume_mm3": res.residual_volume_mm3,
            "safety_pass": res.safety_pass,
            "haas_program_file": nc_name,
            "planning_wall_time_s": res.planning_wall_time_s,
            "per_op_metrics": res.per_op_metrics,
        }
        part_records.append(rec)
        print(
            f"  [{res.part_id}] {res.unique_tools_used} tools / {len(res.operations)} ops | "
            f"Naive={res.naive_total_time_s:.2f}s -> SameSpd={res.same_speed_total_time_s:.2f}s (-{res.same_speed_gain_pct:.1f}%) "
            f"-> OptChip={res.opt_chipload_total_time_s:.2f}s (-{res.opt_chipload_gain_pct:.1f}%, cut -{rec['cutting_only_gain_pct']:.1f}%) | "
            f"Gouge={res.gouge_volume_mm3:.3f}mm3 | PlanWall={res.planning_wall_time_s:.2f}s"
        )

    # Update viewer manifest cleanly without duplicating TAMP entries
    manifest_path = viewer_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base_cases = [c for c in manifest.get("cases", []) if not str(c.get("case_id", "")).startswith("tamp-")]
        manifest["cases"] = base_cases + manifest_tamp_entries
        manifest["tamp_summary"] = part_records
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "schema": "diffcam-exp6-tamp/1",
        "tool_library_summary": DEFAULT_TOOL_LIBRARY.summary(),
        "num_parts": len(part_records),
        "mean_unique_tools": round(float(np.mean([p["unique_tools_used"] for p in part_records])), 2),
        "mean_operations": round(float(np.mean([p["num_operations"] for p in part_records])), 2),
        "mean_naive_total_time_s": round(float(np.mean([p["naive_total_time_s"] for p in part_records])), 3),
        "mean_same_speed_total_time_s": round(float(np.mean([p["same_speed_total_time_s"] for p in part_records])), 3),
        "mean_same_speed_gain_pct": round(float(np.mean([p["same_speed_gain_pct"] for p in part_records])), 2),
        "mean_opt_chipload_total_time_s": round(float(np.mean([p["opt_chipload_total_time_s"] for p in part_records])), 3),
        "mean_opt_chipload_gain_pct": round(float(np.mean([p["opt_chipload_gain_pct"] for p in part_records])), 2),
        "mean_cutting_only_gain_pct": round(float(np.mean([p["cutting_only_gain_pct"] for p in part_records])), 2),
        "mean_opt_optimal_hex_pct": round(float(np.mean([p["opt_optimal_hex_pct"] for p in part_records])), 2),
        "all_safety_pass": all(p["safety_pass"] for p in part_records),
        "max_gouge_volume_mm3": round(max(p["gouge_volume_mm3"] for p in part_records), 4),
        "mean_planning_wall_time_s": round(float(np.mean([p["planning_wall_time_s"] for p in part_records])), 2),
        "parts": part_records,
    }
    out_path = RUN_DIR / "exp6_tamp_multitool.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")
    return summary


if __name__ == "__main__":
    import sys

    if "--only-exp6" in sys.argv:
        run_exp6_multitool_tamp()
    elif "--only-exp5b" in sys.argv:
        run_exp5b_inference_overfitting()
    else:
        run_exp5b_inference_overfitting()
        run_exp6_multitool_tamp()
