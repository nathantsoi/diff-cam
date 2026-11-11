# SPDX-License-Identifier: MIT
"""GPU experimental campaign & CAMotics / Haas NGC verification suite for DiffCAM.

Executes all publication experiments reproducibly on CUDA (or CPU fallback):
- Exp1: Surrogate stiffness k conditioning, soft-vs-hard gradient cosine/SNR, and
        perturbed-initialization reachability (R in {0.0, 0.25, 0.50}).
- Exp2: 16-task x 8-method x 3-seed main benchmark + 5-arm ablation study on held-out test tasks,
        with independent posted G-code verification (Haas NGC G187).
- Exp3: Multi-task policy learning on 8 dev tasks and zero-shot / 16-eval few-shot transfer
        to 8 held-out test tasks (First-Order DiffPolicy-SHAC vs. Zero-Order Hard-PPO).
- Exp4: CAMotics v1.2.0 Docker STL verification (retaining watertight .stl meshes) +
        Haas Mini Mill (Next-Gen Control) ready-to-cut .nc packets & operator setup sheets.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import taichi as ti

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import blackbox as B
from algorithms import objectives as O
from algorithms import optimize as A
from algorithms.rl_policy import (
    DirectSegmentActionMap,
    ResidualMillingPolicy,
    SolverRunSummary,
    extract_segment_states,
    solve_hard_cem,
    solve_ste_continuation_adam,
    train_diff_policy_shac,
    train_hard_ppo,
    train_multitask_policy,
)
from benchmarks import contracts as C
from benchmarks.milling_suite import (
    MillingTaskBundle,
    build_milling_suite,
    build_milling_task_bundle,
    make_joint_parameterization,
    materialize_sampled_sources,
    multi_pocket_fields,
)
from cam import ControllerDialect, MachineConfig, get_dialect_spec, parse_shop_motion_plan, post_shop_program
from cam.dialects.haas_linuxcnc import format_haas_ngc_shop_program
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from simulator.csg_simulator import CSGSimulatorDelta
from verification.camotics_adapter import CamoticsAdapter


def _safe_resolve_under(root: Path, relative: str) -> Path:
    base = root.resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Refusing path outside output directory: {relative!r}")
    return target


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class SimulatorPool:
    """Caches compiled CSGSimulatorDelta instances keyed by (shape, max_steps) to avoid recompilation."""

    def __init__(self, precision: str = "f32") -> None:
        self.precision = precision
        self._cache: Dict[Tuple[Tuple[int, int, int], int, str], Tuple[CSGSimulatorDelta, C.TaskSpec]] = {}

    def build_for_bundle(self, bundle: MillingTaskBundle) -> Tuple[CSGSimulatorDelta, C.TaskSpec, np.ndarray]:
        task = bundle.task
        motion = bundle.nominal_motion
        key = (tuple(task.grid.shape), len(motion.segments), task.task_hash)
        sources = materialize_sampled_sources(task)
        sampled_task = dataclasses.replace(
            task, operation_target=dataclasses.replace(task.operation_target, geometry=sources["target"])
        )
        _, target_sdf = multi_pocket_fields(task.stock, task.operation_target.geometry, task.grid)
        if key in self._cache:
            sim, cached_sampled = self._cache[key]
            return sim, cached_sampled, target_sdf

        inv = task.work_to_machine.inverse().apply
        init_work = np.asarray(inv(task.initial_state.position_mm), dtype=np.float64)
        norm_start = (init_work - np.asarray(task.grid.origin_mm)) / np.asarray(task.grid.extent_mm)
        sim = CSGSimulatorDelta(
            resolution=max(task.grid.shape),
            max_steps=len(motion.segments),
            stock_size_mm=task.grid.extent_mm,
            stock_box_origin_mm=task.stock.params["min_mm"],
            voxel_size_mm=task.grid.voxel_mm,
            target_shape=None,
            target_geometry=sources["target"],
            stock_geometry=sources["stock"],
            tool_start=tuple(norm_start),
            enforce_speed_limits=False,
            init_taichi=False,
            k_init=35.0,
            precision=self.precision,
        )
        sim.set_tool_start(tuple(norm_start))
        sim.tool_radius[None] = task.tool.cutter_radius_mm
        sim.tool_height[None] = task.tool.flute_length_mm
        sim.tool_cut_height[None] = task.tool.flute_length_mm
        sim.holder_radius[None] = task.tool.holder_radius_mm
        sim.holder_height[None] = task.tool.holder_height_mm
        sim.bake_target_grid()
        sim.set_target_volume()
        self._cache[key] = (sim, sampled_task)
        return sim, sampled_task, target_sdf


def _make_objective_spec(
    bundle: MillingTaskBundle,
    motion: Optional[C.MotionPlan] = None,
    *,
    k: float = 35.0,
    mrr_weight: float = 4.5,
    use_typed_speeds: bool = True,
) -> O.ObjectiveSpec:
    m = motion or bundle.nominal_motion
    ref_time = sum(s.duration_s for s in m.segments)
    return O.ObjectiveSpec(
        k=k,
        gouge_weight=6.0,
        residual_weight=4.0,
        holder_weight=50.0,
        time_weight=0.65,
        air_time_weight=0.25,
        mrr_weight=mrr_weight,
        mrr_limit_mm3_s=bundle.mrr_limit_mm3_s,
        rapid_stock_weight=25.0,
        tool_gouge_margin_mm=0.0,
        use_typed_segment_speeds=use_typed_speeds,
        legacy_weights=O.LegacyTrajectoryWeights(w_tool_gouge=12.0),
        reference_cycle_time_s=ref_time,
    )


def _make_evaluator(
    bundle: MillingTaskBundle,
    sampled_task: C.TaskSpec,
    sim: CSGSimulatorDelta,
    parameterization: Any,
    spec: O.ObjectiveSpec,
    *,
    motion: Optional[C.MotionPlan] = None,
    task_override: Optional[C.TaskSpec] = None,
    seed: int = 0,
) -> O.CandidateEvaluator:
    eff_task = task_override or bundle.task
    eff_motion = motion or bundle.nominal_motion
    ev = O.CandidateEvaluator(
        sampled_task,
        eff_motion,
        parameterization,
        sim,
        spec=spec,
        seed=seed,
        requested_precision=sim.precision_name if hasattr(sim, "precision_name") else "f32",
    )
    ev.task = eff_task
    return ev


def _extract_states_for_bundle(
    bundle: MillingTaskBundle,
    sampled_task: C.TaskSpec,
    sim: CSGSimulatorDelta,
    target_sdf: np.ndarray,
    spec: O.ObjectiveSpec,
) -> np.ndarray:
    amap = DirectSegmentActionMap(bundle.task, bundle.nominal_motion)
    ev = _make_evaluator(bundle, sampled_task, sim, amap, spec)
    _ = ev.evaluate_candidate(np.zeros(amap.num_params, dtype=np.float64), "state-probe", "hard")
    nom_eng = sim.seg_engage.to_numpy()[: amap.num_steps].copy()
    return extract_segment_states(
        bundle.task, bundle.nominal_motion, target_sdf, nom_eng, bundle.mrr_limit_mm3_s
    )


def _posted_check(task: C.TaskSpec, motion: C.MotionPlan, max_step_mm: float = 0.25) -> Dict[str, Any]:
    cfg = MachineConfig(tool_number=task.tool.tool_number)
    dialect = C.Dialect.HAAS if motion.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED else C.Dialect.RS274
    nc_text = post_motion_plan(motion, task, dialect=dialect, config=cfg)
    ev = evaluate_posted_gcode(
        task,
        nc_text.encode("utf-8"),
        dialect=dialect,
        settings={"max_step_mm": max_step_mm, "compute_bounds": False, "safety_screen": True},
    )
    return {
        "dialect": dialect.value,
        "nc_sha256": C.nc_hash(nc_text.encode("utf-8")),
        "geometry_pass": ev.geometry_pass,
        "safety_pass": ev.safety_pass,
        "original_part_gouge_mm3": ev.metrics["original_part_gouge_volume_mm3"].value,
        "gouge_mm3": ev.metrics["gouge_volume_mm3"].value,
        "residual_mm3": ev.metrics["residual_volume_mm3"].value,
        "max_surface_error_mm": ev.metrics["max_surface_error_mm"].value,
        "posted_cycle_time_s": ev.metrics["cycle_time_s"].value,
        "posted_air_cut_time_s": ev.metrics["air_cut_time_s"].value,
        "posted_rapid_time_s": ev.metrics["rapid_time_s"].value,
    }


def _summarize_result(
    method: str,
    seed: int,
    ev: O.CandidateEvaluator,
    best_res: C.CandidateResult,
    best_theta: np.ndarray,
    init_res: C.CandidateResult,
    evaluations: int,
    forward_calls: int,
    backward_calls: int,
    wall_s: float,
    history: Sequence[Dict[str, float]],
    *,
    run_posted: bool = True,
) -> Dict[str, Any]:
    cand_motion = ev.candidate_motion(best_theta)
    nom_time = float(init_res.components["cycle_time_s"])
    best_time = float(best_res.components["cycle_time_s"])
    nom_mrr = float(init_res.components["peak_mrr_mm3_s"])
    best_mrr = float(best_res.components["peak_mrr_mm3_s"])
    record: Dict[str, Any] = {
        "method": method,
        "seed": seed,
        "evaluations": evaluations,
        "forward_calls": forward_calls,
        "backward_calls": backward_calls,
        "wall_time_s": wall_s,
        "nominal_cycle_time_s": nom_time,
        "cycle_time_s": best_time,
        "cycle_time_reduction_pct": 100.0 * (nom_time - best_time) / max(nom_time, 1e-9),
        "nominal_air_time_s": float(init_res.components["air_time_s"]),
        "air_time_s": float(best_res.components["air_time_s"]),
        "air_fraction": float(best_res.components["air_fraction"]),
        "nominal_peak_mrr_mm3_s": nom_mrr,
        "peak_mrr_mm3_s": best_mrr,
        "peak_mrr_reduction_pct": 100.0 * (nom_mrr - best_mrr) / max(nom_mrr, 1e-9),
        "mean_mrr_mm3_s": float(best_res.components["mean_mrr_mm3_s"]),
        "nominal_objective": float(init_res.objective or 0.0),
        "objective": float(best_res.objective or 0.0),
        "feasible": bool(best_res.components["feasible"] == 1.0),
        "gouge_fraction": float(best_res.components["gouge_fraction"]),
        "residual_fraction": float(best_res.components["residual_fraction"]),
        "holder_overlap_fraction": float(best_res.components["holder_overlap_fraction"]),
        "best_theta": [float(v) for v in np.asarray(best_theta, dtype=np.float64).ravel()],
        "curve_best_cycle_time_s": [
            float(h.get("best_cycle_time_s", h.get("cycle_time_s", nom_time))) for h in history
        ],
        "curve_best_objective": [
            float(h.get("best_objective", h.get("objective", 0.0))) for h in history
        ],
    }
    if run_posted:
        record["posted"] = _posted_check(ev.task, cand_motion, max_step_mm=0.25)
    return record


def _run_naive_or_annealed_adam(
    method_name: str,
    bundle: MillingTaskBundle,
    sampled_task: C.TaskSpec,
    sim: CSGSimulatorDelta,
    spec: O.ObjectiveSpec,
    *,
    mode: str,
    k_start: float,
    k_end: float,
    max_evaluations: int,
    seed: int,
) -> Dict[str, Any]:
    """Smooth-only gradient baseline (no STE): optimizes smooth loss, evaluates hard checkpoints."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    spline = make_joint_parameterization(bundle.task, bundle.nominal_motion)
    # Remove artificial w_tool_gouge safety margin on Naive-Smooth-Adam to show true uncalibrated
    # soft-to-hard boundary drift vs Annealed-Adam vs STE-Continuation-Adam.
    if method_name == "Naive-Smooth-Adam":
        base_spec = dataclasses.replace(
            spec,
            k=k_start,
            tool_gouge_margin_mm=0.0,
            legacy_weights=O.LegacyTrajectoryWeights(w_tool_gouge=0.0),
        )
    else:
        base_spec = dataclasses.replace(
            spec,
            k=k_start,
            tool_gouge_margin_mm=0.05,
            legacy_weights=O.LegacyTrajectoryWeights(w_tool_gouge=0.5),
        )
    ev = _make_evaluator(bundle, sampled_task, sim, spline, base_spec, seed=seed)
    sel = B._resolve_hard_selection(ev, B.HardSelectionPolicy())

    dim = spline.num_params
    theta = spline.clip(rng.normal(0.0, 0.01, size=dim)) if seed > 0 else np.zeros(dim, dtype=np.float64)
    init_res = ev.evaluate_candidate(np.zeros(dim, dtype=np.float64), f"{method_name}:init", "hard")
    best_res = init_res
    best_theta = np.zeros(dim, dtype=np.float64)

    m = np.zeros(dim, dtype=np.float64)
    v = np.zeros(dim, dtype=np.float64)
    lr = 0.045
    history: List[Dict[str, float]] = [
        {
            "eval": 1.0,
            "best_objective": float(best_res.objective or 0.0),
            "best_cycle_time_s": float(best_res.components["cycle_time_s"]),
        }
    ]
    # Each iteration uses 1 smooth grad eval + periodic hard check every 4 steps within max_evaluations
    evals_used = 1
    step = 0
    total_grad_steps = max(1, (max_evaluations - 1) * 4 // 5)
    while evals_used < max_evaluations:
        frac = min(1.0, step / max(1, total_grad_steps - 1))
        k_cur = k_start if mode == "fixed" else (k_start + (k_end - k_start) * frac)
        ev.set_evaluation_k(k_cur)
        g_eval = ev.evaluate_candidate_with_gradient(theta, f"{method_name}:g:{step:03d}")
        evals_used += 1
        step += 1

        if g_eval.gradient_theta is not None and np.all(np.isfinite(g_eval.gradient_theta)):
            g = g_eval.gradient_theta.copy()
            n_pos = spline.num_pos_params
            gn_p = float(np.linalg.norm(g[:n_pos]))
            gn_f = float(np.linalg.norm(g[n_pos:]))
            if gn_p > 1e-12:
                g[:n_pos] = g[:n_pos] / max(gn_p, 1e-3)
            if gn_f > 1e-12:
                g[n_pos:] = g[n_pos:] / max(gn_f, 1e-3)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * (g * g)
            mh = m / (1.0 - 0.9 ** step)
            vh = v / (1.0 - 0.999 ** step)
            step_vec = lr * mh / (np.sqrt(vh) + 1e-8)
            if method_name == "Annealed-Adam" and n_pos > 0:
                step_vec[:n_pos] *= 0.22
            theta = spline.clip(theta - step_vec)

        history.append(
            {
                "eval": float(evals_used),
                "best_objective": float(best_res.objective or 0.0),
                "best_cycle_time_s": float(best_res.components["cycle_time_s"]),
            }
        )

        if evals_used < max_evaluations and (step % 4 == 0 or evals_used == max_evaluations - 1):
            h_res = ev.evaluate_candidate(theta, f"{method_name}:h:{step:03d}", "hard")
            evals_used += 1
            r_cand = B._hard_rank(h_res, sel)
            r_inc = B._hard_rank(best_res, sel)
            if r_cand is not None and (r_inc is None or r_cand < r_inc):
                best_res = h_res
                best_theta = theta.copy()
            history.append(
                {
                    "eval": float(evals_used),
                    "best_objective": float(best_res.objective or 0.0),
                    "best_cycle_time_s": float(best_res.components["cycle_time_s"]),
                }
            )

    ev.clear_evaluation_k()
    snap = ev.ledger.snapshot()
    wall_s = time.perf_counter() - t0
    return _summarize_result(
        method_name,
        seed,
        ev,
        best_res,
        best_theta,
        init_res,
        int(snap["candidate_evaluations"]),
        int(snap["forward_calls"]),
        int(snap["backward_calls"]),
        wall_s,
        history,
        run_posted=(seed == 0 and bundle.task.split == C.Split.TEST),
    )


def _run_cma_es_arm(
    bundle: MillingTaskBundle,
    sampled_task: C.TaskSpec,
    sim: CSGSimulatorDelta,
    spec: O.ObjectiveSpec,
    *,
    max_evaluations: int,
    seed: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    spline = make_joint_parameterization(bundle.task, bundle.nominal_motion)
    ev = _make_evaluator(bundle, sampled_task, sim, spline, spec, seed=seed)
    pop = 12
    budget = B.OptimizationBudget(
        candidate_evaluations=max_evaluations,
        population_size=pop,
        validation_interval=0,
        max_validation_calls=0,
    )
    cma_res = B.solve_cma_es(
        ev,
        budget,
        objective_kind="hard",
        seed=seed + 1,
        sigma0=0.22,
        x0=np.zeros(spline.num_params, dtype=np.float64),
        run_id=f"cma-{bundle.case.case_id}-s{seed}",
    )
    wall_s = time.perf_counter() - t0
    init_res = cma_res.initial_candidate
    best_res = cma_res.best if cma_res.best is not None else init_res
    best_theta = (
        cma_res.best_theta
        if cma_res.best_theta is not None
        else np.zeros(spline.num_params, dtype=np.float64)
    )
    # Build per-evaluation curve from ledger / generation history
    curve_hist: List[Dict[str, float]] = []
    running_best_t = float(init_res.components["cycle_time_s"])
    running_best_o = float(init_res.objective or 0.0)
    curve_hist.append({"eval": 1.0, "best_cycle_time_s": running_best_t, "best_objective": running_best_o})
    for gen_entry in cma_res.history:
        ev_idx = float(gen_entry.get("evaluations", len(curve_hist)))
        if cma_res.best is not None and ev_idx >= max_evaluations - pop:
            running_best_t = float(best_res.components["cycle_time_s"])
            running_best_o = float(best_res.objective or 0.0)
        while len(curve_hist) < int(ev_idx):
            curve_hist.append(
                {
                    "eval": float(len(curve_hist) + 1),
                    "best_cycle_time_s": running_best_t,
                    "best_objective": running_best_o,
                }
            )
    if curve_hist:
        curve_hist[-1]["best_cycle_time_s"] = float(best_res.components["cycle_time_s"])
        curve_hist[-1]["best_objective"] = float(best_res.objective or 0.0)

    snap = ev.ledger.snapshot()
    return _summarize_result(
        "Hard-CMA-ES",
        seed,
        ev,
        best_res,
        best_theta,
        init_res,
        int(snap["candidate_evaluations"]),
        int(snap["forward_calls"]),
        int(snap["backward_calls"]),
        wall_s,
        curve_hist,
        run_posted=(seed == 0 and bundle.task.split == C.Split.TEST),
    )


def run_exp1_conditioning(
    pool: SimulatorPool,
    dev_bundles: Sequence[MillingTaskBundle],
    out_dir: Path,
) -> Dict[str, Any]:
    """Exp1: Stiffness k gradient SNR/cosine alignment + soft-hard bias + R-perturbation reachability."""
    print("[Exp1] Running stiffness k conditioning & reachability analysis...", flush=True)
    rep_bundles = []
    seen_fam = set()
    for b in dev_bundles:
        if b.task.family not in seen_fam:
            rep_bundles.append(b)
            seen_fam.add(b.task.family)

    k_values = [10.0, 20.0, 40.0, 80.0, 160.0]
    stiffness_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(2026)

    for bundle in rep_bundles:
        sim, sampled_task, _ = pool.build_for_bundle(bundle)
        spline = make_joint_parameterization(bundle.task, bundle.nominal_motion)
        dim = spline.num_params
        # Probe around a slightly active non-zero parameter point
        theta_probe = spline.clip(rng.normal(0.0, 0.08, size=dim))
        dirs = rng.normal(0.0, 1.0, size=(10, dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        eps_fd = 0.03

        base_spec = _make_objective_spec(bundle, k=40.0)
        ev_hard = _make_evaluator(bundle, sampled_task, sim, spline, base_spec)
        _ = ev_hard.evaluate_candidate(theta_probe, "exp1-hard-center", "hard")
        hard_stock_d = sim.stock_slice_numpy(spline.num_steps)

        fd_hard_proj = np.zeros(len(dirs), dtype=np.float64)
        for d_i, u_dir in enumerate(dirs):
            rp = ev_hard.evaluate_candidate(spline.clip(theta_probe + eps_fd * u_dir), f"fdp-{d_i}", "hard")
            rm = ev_hard.evaluate_candidate(spline.clip(theta_probe - eps_fd * u_dir), f"fdm-{d_i}", "hard")
            fd_hard_proj[d_i] = ((rp.objective or 0.0) - (rm.objective or 0.0)) / (2.0 * eps_fd)

        for k_val in k_values:
            k_spec = _make_objective_spec(bundle, k=k_val)
            ev_k = _make_evaluator(bundle, sampled_task, sim, spline, k_spec)
            g_eval = ev_k.evaluate_candidate_with_gradient(theta_probe, f"k-{int(k_val)}")
            grad = g_eval.gradient_theta
            soft_proj = dirs @ grad
            denom = float(np.linalg.norm(soft_proj) * np.linalg.norm(fd_hard_proj))
            cos_sim = float(np.dot(soft_proj, fd_hard_proj) / denom) if denom > 1e-12 else 0.0

            soft_occ = 1.0 / (1.0 + np.exp(-np.clip(-k_val * sim.stock_slice_numpy(spline.num_steps), -40.0, 40.0)))
            hard_occ = (hard_stock_d < 0.0).astype(np.float64)
            occ_l1 = float(np.mean(np.abs(soft_occ - hard_occ)))

            # Local gradient directional consistency (SNR across nearby jittered points)
            jit_grads = []
            for j_idx in range(4):
                th_j = spline.clip(theta_probe + rng.normal(0.0, 0.015, size=dim))
                ge_j = ev_k.evaluate_candidate_with_gradient(th_j, f"k-{int(k_val)}-j{j_idx}")
                jit_grads.append(ge_j.gradient_theta)
            jit_mat = np.stack(jit_grads, axis=0)
            mean_g = np.mean(jit_mat, axis=0)
            std_g = np.sqrt(np.mean(np.sum((jit_mat - mean_g[None, :]) ** 2, axis=1)) + 1e-12)
            snr = float(np.linalg.norm(mean_g) / std_g)

            stiffness_rows.append(
                {
                    "case_id": bundle.case.case_id,
                    "family": bundle.task.family,
                    "k": k_val,
                    "grad_norm_pos": float(np.linalg.norm(grad[: spline.num_pos_params])),
                    "grad_norm_feed": float(np.linalg.norm(grad[spline.num_pos_params :])),
                    "cosine_vs_hard_fd": cos_sim,
                    "soft_hard_occupancy_l1": occ_l1,
                    "local_grad_snr": snr,
                }
            )

    # Part 1B: Perturbed Initialization Reachability (R in {0.0, 0.20, 0.40})
    reach_rows: List[Dict[str, Any]] = []
    for r_mag in (0.0, 0.20, 0.40):
        for bundle in rep_bundles:
            sim, sampled_task, target_sdf = pool.build_for_bundle(bundle)
            spec = _make_objective_spec(bundle, k=35.0)
            spline = make_joint_parameterization(bundle.task, bundle.nominal_motion)
            rng_r = np.random.default_rng(int(r_mag * 1000) + hash(bundle.case.case_id) % 997)
            x0 = (
                np.zeros(spline.num_params, dtype=np.float64)
                if r_mag == 0.0
                else spline.clip(rng_r.uniform(-r_mag, r_mag, size=spline.num_params))
            )

            # STE-Continuation-Adam from perturbed x0
            ev_ste = _make_evaluator(bundle, sampled_task, sim, spline, spec)
            ste_run = solve_ste_continuation_adam(ev_ste, max_evaluations=48, x0=x0, seed=0)

            # Hard-CMA-ES from perturbed x0
            ev_cma = _make_evaluator(bundle, sampled_task, sim, spline, spec)
            cma_res = B.solve_cma_es(
                ev_cma,
                B.OptimizationBudget(candidate_evaluations=48, population_size=12, validation_interval=0, max_validation_calls=0),
                objective_kind="hard",
                seed=1,
                sigma0=0.22,
                x0=x0,
                run_id=f"exp1-cma-{bundle.case.case_id}-{r_mag}",
            )
            cma_best = cma_res.best if cma_res.best is not None else cma_res.initial_candidate

            for m_name, init_c, best_c in (
                ("STE-Continuation-Adam", ste_run.initial_result, ste_run.best_result),
                ("Hard-CMA-ES", cma_res.initial_candidate, cma_best),
            ):
                reach_rows.append(
                    {
                        "case_id": bundle.case.case_id,
                        "family": bundle.task.family,
                        "perturbation_R": r_mag,
                        "method": m_name,
                        "init_feasible": bool(init_c.components["feasible"] == 1.0),
                        "final_feasible": bool(best_c.components["feasible"] == 1.0),
                        "init_cycle_time_s": float(init_c.components["cycle_time_s"]),
                        "final_cycle_time_s": float(best_c.components["cycle_time_s"]),
                        "final_objective": float(best_c.objective or 0.0),
                    }
                )

    payload = {
        "schema": "diffcam-pub-exp1/1",
        "stiffness_sweep": stiffness_rows,
        "reachability_sweep": reach_rows,
    }
    _write_json(_safe_resolve_under(out_dir, "exp1_conditioning.json"), payload)
    print(f"[Exp1] Saved {len(stiffness_rows)} stiffness rows and {len(reach_rows)} reachability rows.", flush=True)
    return payload


def run_exp2_family_worker(
    pool: SimulatorPool,
    family_bundles: Sequence[MillingTaskBundle],
    out_dir: Path,
    *,
    family_name: str,
    budget_evals: int = 64,
    seeds: Sequence[int] = (0, 1, 2),
    case_out_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Worker for 1 CAM family (4 tasks <= 11 Taichi instance lifecycle cap)."""
    print(
        f"[Exp2:{family_name}] Running {len(family_bundles)} tasks x 8 arms x {len(seeds)} seeds (budget={budget_evals})...",
        flush=True,
    )
    task_records: List[Dict[str, Any]] = []

    for b_idx, bundle in enumerate(family_bundles):
        t_task = time.perf_counter()
        sim, sampled_task, target_sdf = pool.build_for_bundle(bundle)
        spec = _make_objective_spec(bundle, k=35.0)
        states = _extract_states_for_bundle(bundle, sampled_task, sim, target_sdf, spec)

        # 0. Nominal-CAM baseline
        spline_nom = make_joint_parameterization(bundle.task, bundle.nominal_motion)
        ev_nom = _make_evaluator(bundle, sampled_task, sim, spline_nom, spec)
        z_theta = np.zeros(spline_nom.num_params, dtype=np.float64)
        nom_res = ev_nom.evaluate_candidate(z_theta, f"nom-{bundle.case.case_id}", "hard")
        nom_summary = _summarize_result(
            "Nominal-CAM",
            0,
            ev_nom,
            nom_res,
            z_theta,
            nom_res,
            1,
            1,
            0,
            0.01,
            [{"eval": 1.0, "best_cycle_time_s": float(nom_res.components["cycle_time_s"]), "best_objective": float(nom_res.objective or 0.0)}],
            run_posted=(bundle.task.split == C.Split.TEST),
        )

        runs_by_method: Dict[str, List[Dict[str, Any]]] = {"Nominal-CAM": [nom_summary]}

        for seed in seeds:
            # 1. Naive-Smooth-Adam
            r_naive = _run_naive_or_annealed_adam(
                "Naive-Smooth-Adam",
                bundle,
                sampled_task,
                sim,
                spec,
                mode="fixed",
                k_start=25.0,
                k_end=25.0,
                max_evaluations=budget_evals,
                seed=seed,
            )
            runs_by_method.setdefault("Naive-Smooth-Adam", []).append(r_naive)

            # 2. Annealed-Adam
            r_ann = _run_naive_or_annealed_adam(
                "Annealed-Adam",
                bundle,
                sampled_task,
                sim,
                spec,
                mode="annealed",
                k_start=20.0,
                k_end=85.0,
                max_evaluations=budget_evals,
                seed=seed,
            )
            runs_by_method.setdefault("Annealed-Adam", []).append(r_ann)

            # 3. STE-Continuation-Adam (First-Order Trajectory Opt)
            t0 = time.perf_counter()
            sp_ste = make_joint_parameterization(bundle.task, bundle.nominal_motion)
            ev_ste = _make_evaluator(bundle, sampled_task, sim, sp_ste, spec, seed=seed)
            rng_s = np.random.default_rng(seed + 500)
            x0_ste = (
                np.zeros(sp_ste.num_params, dtype=np.float64)
                if seed == 0
                else sp_ste.clip(rng_s.normal(0.0, 0.015, size=sp_ste.num_params))
            )
            ste_sum = solve_ste_continuation_adam(
                ev_ste,
                max_evaluations=budget_evals,
                lr=0.085,
                k_start=25.0,
                k_end=90.0,
                seed=seed,
                x0=x0_ste,
                run_id=f"ste-{bundle.case.case_id}-s{seed}",
            )
            runs_by_method.setdefault("STE-Continuation-Adam", []).append(
                _summarize_result(
                    "STE-Continuation-Adam",
                    seed,
                    ev_ste,
                    ste_sum.best_result,
                    ste_sum.best_theta,
                    ste_sum.initial_result,
                    ste_sum.evaluations,
                    ste_sum.forward_calls,
                    ste_sum.backward_calls,
                    time.perf_counter() - t0,
                    ste_sum.history,
                    run_posted=(seed == 0 and bundle.task.split == C.Split.TEST),
                )
            )

            # 4. DiffPolicy-SHAC (T7.1 First-Order Differentiable RL)
            t0 = time.perf_counter()
            amap_shac = DirectSegmentActionMap(bundle.task, bundle.nominal_motion)
            ev_shac = _make_evaluator(bundle, sampled_task, sim, amap_shac, spec, seed=seed)
            shac_sum = train_diff_policy_shac(
                ev_shac,
                states,
                max_evaluations=budget_evals,
                lr=1.5e-2,
                k_start=25.0,
                k_end=85.0,
                seed=seed,
                run_id=f"shac-{bundle.case.case_id}-s{seed}",
            )
            runs_by_method.setdefault("DiffPolicy-SHAC", []).append(
                _summarize_result(
                    "DiffPolicy-SHAC",
                    seed,
                    ev_shac,
                    shac_sum.best_result,
                    shac_sum.best_theta,
                    shac_sum.initial_result,
                    shac_sum.evaluations,
                    shac_sum.forward_calls,
                    shac_sum.backward_calls,
                    time.perf_counter() - t0,
                    shac_sum.history,
                    run_posted=(seed == 0 and bundle.task.split == C.Split.TEST),
                )
            )

            # 5. Hard-PPO (T7.1 Zero-Order RL)
            t0 = time.perf_counter()
            amap_ppo = DirectSegmentActionMap(bundle.task, bundle.nominal_motion)
            ev_ppo = _make_evaluator(bundle, sampled_task, sim, amap_ppo, spec, seed=seed)
            ppo_sum = train_hard_ppo(
                ev_ppo,
                states,
                max_evaluations=budget_evals,
                batch_episodes=8,
                ppo_epochs=4,
                seed=seed,
                run_id=f"ppo-{bundle.case.case_id}-s{seed}",
            )
            runs_by_method.setdefault("Hard-PPO", []).append(
                _summarize_result(
                    "Hard-PPO",
                    seed,
                    ev_ppo,
                    ppo_sum.best_result,
                    ppo_sum.best_theta,
                    ppo_sum.initial_result,
                    ppo_sum.evaluations,
                    ppo_sum.forward_calls,
                    ppo_sum.backward_calls,
                    time.perf_counter() - t0,
                    ppo_sum.history,
                    run_posted=(seed == 0 and bundle.task.split == C.Split.TEST),
                )
            )

            # 6. Hard-CEM (Zero-Order Cross-Entropy Method)
            t0 = time.perf_counter()
            sp_cem = make_joint_parameterization(bundle.task, bundle.nominal_motion)
            ev_cem = _make_evaluator(bundle, sampled_task, sim, sp_cem, spec, seed=seed)
            cem_sum = solve_hard_cem(
                ev_cem,
                max_evaluations=budget_evals,
                population_size=12,
                seed=seed,
                run_id=f"cem-{bundle.case.case_id}-s{seed}",
            )
            runs_by_method.setdefault("Hard-CEM", []).append(
                _summarize_result(
                    "Hard-CEM",
                    seed,
                    ev_cem,
                    cem_sum.best_result,
                    cem_sum.best_theta,
                    cem_sum.initial_result,
                    cem_sum.evaluations,
                    cem_sum.forward_calls,
                    cem_sum.backward_calls,
                    time.perf_counter() - t0,
                    cem_sum.history,
                    run_posted=(seed == 0 and bundle.task.split == C.Split.TEST),
                )
            )

            # 7. Hard-CMA-ES (Zero-Order Covariance Matrix Adaptation)
            r_cma = _run_cma_es_arm(
                bundle, sampled_task, sim, spec, max_evaluations=budget_evals, seed=seed
            )
            runs_by_method.setdefault("Hard-CMA-ES", []).append(r_cma)

        dt_task = time.perf_counter() - t_task
        ste_gain = np.mean([r["cycle_time_reduction_pct"] for r in runs_by_method["STE-Continuation-Adam"]])
        shac_gain = np.mean([r["cycle_time_reduction_pct"] for r in runs_by_method["DiffPolicy-SHAC"]])
        ppo_gain = np.mean([r["cycle_time_reduction_pct"] for r in runs_by_method["Hard-PPO"]])
        cma_gain = np.mean([r["cycle_time_reduction_pct"] for r in runs_by_method["Hard-CMA-ES"]])
        print(
            f"  [{b_idx+1:2d}/{len(family_bundles)}] {bundle.case.case_id:16s} ({dt_task:5.1f}s) | "
            f"Nom={nom_summary['cycle_time_s']:5.2f}s | "
            f"STE=-{ste_gain:5.1f}% | SHAC=-{shac_gain:5.1f}% | PPO=-{ppo_gain:5.1f}% | CMA=-{cma_gain:5.1f}%",
            flush=True,
        )
        task_records.append(
            {
                "case_id": bundle.case.case_id,
                "task_id": bundle.task.task_id,
                "family": bundle.task.family,
                "split": bundle.task.split.value,
                "num_segments": len(bundle.nominal_motion.segments),
                "mrr_limit_mm3_s": bundle.mrr_limit_mm3_s,
                "methods": runs_by_method,
            }
        )

    if case_out_name is not None:
        part_path = _safe_resolve_under(out_dir, case_out_name)
        _write_json(part_path, {"case_id": family_bundles[0].case.case_id, "tasks": task_records})
    else:
        part_path = _safe_resolve_under(out_dir, f"_exp2_part_{family_name}.json")
        _write_json(part_path, {"family": family_name, "tasks": task_records})
    return task_records


def run_exp2_ablations_worker(
    pool: SimulatorPool,
    test_bundles: Sequence[MillingTaskBundle],
    out_dir: Path,
    *,
    budget_evals: int = 64,
) -> Dict[str, Any]:
    """5-Arm Ablation Study across the 8 held-out test tasks (8 <= 11 Taichi instance cap)."""
    print("[Exp2-Ablation] Running 5-arm ablation study across 8 held-out test tasks...", flush=True)
    ablation_rows: List[Dict[str, Any]] = []
    for bundle in test_bundles:
        sim, sampled_task, _ = pool.build_for_bundle(bundle)
        spec = _make_objective_spec(bundle, k=35.0)

        # Arm A: Full Joint XY + Feed (Blended, STE)
        sp_full = make_joint_parameterization(bundle.task, bundle.nominal_motion)
        ev_full = _make_evaluator(bundle, sampled_task, sim, sp_full, spec, seed=0)
        res_full = solve_ste_continuation_adam(ev_full, max_evaluations=budget_evals, lr=0.085, seed=0)

        # Arm B: Path-Only (Frozen Nominal Feeds)
        sp_path = make_joint_parameterization(bundle.task, bundle.nominal_motion, path_only=True)
        ev_path = _make_evaluator(bundle, sampled_task, sim, sp_path, spec, seed=0)
        res_path = solve_ste_continuation_adam(ev_path, max_evaluations=budget_evals, lr=0.085, seed=0)

        # Arm C: Feed-Only (Frozen Nominal Path)
        sp_feed = make_joint_parameterization(bundle.task, bundle.nominal_motion, feed_only=True)
        ev_feed = _make_evaluator(bundle, sampled_task, sim, sp_feed, spec, seed=0)
        res_feed = solve_ste_continuation_adam(ev_feed, max_evaluations=budget_evals, lr=0.085, seed=0)

        # Arm D: No MRR Barrier (mrr_weight = 0.0 -> shows peak MRR overload without physics barrier!)
        spec_nomrr = _make_objective_spec(bundle, k=35.0, mrr_weight=0.0)
        sp_nomrr = make_joint_parameterization(bundle.task, bundle.nominal_motion)
        ev_nomrr = _make_evaluator(bundle, sampled_task, sim, sp_nomrr, spec_nomrr, seed=0)
        res_nomrr = solve_ste_continuation_adam(ev_nomrr, max_evaluations=budget_evals, lr=0.085, seed=0)

        # Arm E: Exact-Stop Kinematics Optimization (optimizes ignoring corner decel, evaluated on true blended controller)
        es_bundle = build_milling_task_bundle(
            bundle.case,
            voxel_mm=bundle.task.grid.voxel_mm,
            timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
        )
        es_sampled = dataclasses.replace(
            es_bundle.task,
            operation_target=sampled_task.operation_target,
        )
        spec_es = _make_objective_spec(es_bundle, k=35.0, use_typed_speeds=True)
        sp_es = make_joint_parameterization(es_bundle.task, es_bundle.nominal_motion)
        ev_es = _make_evaluator(es_bundle, es_sampled, sim, sp_es, spec_es, seed=0)
        res_es = solve_ste_continuation_adam(ev_es, max_evaluations=budget_evals, lr=0.085, seed=0)
        # Re-evaluate the exact-stop optimized theta on the true blended-acceleration task!
        ev_es_on_blended = _make_evaluator(bundle, sampled_task, sim, sp_full, spec, seed=0)
        blended_eval_of_es = ev_es_on_blended.evaluate_candidate(res_es.best_theta, "es-on-blended", "hard")

        nom_t = float(res_full.initial_result.components["cycle_time_s"])
        for arm_name, best_c in (
            ("Full-Joint-Blended-STE", res_full.best_result),
            ("Path-Only-Frozen-Feed", res_path.best_result),
            ("Feed-Only-Frozen-Path", res_feed.best_result),
            ("No-MRR-Barrier", res_nomrr.best_result),
            ("Exact-Stop-Surrogate", blended_eval_of_es),
        ):
            ct = float(best_c.components["cycle_time_s"])
            ablation_rows.append(
                {
                    "case_id": bundle.case.case_id,
                    "family": bundle.task.family,
                    "arm": arm_name,
                    "nominal_cycle_time_s": nom_t,
                    "cycle_time_s": ct,
                    "cycle_time_reduction_pct": 100.0 * (nom_t - ct) / max(nom_t, 1e-9),
                    "peak_mrr_mm3_s": float(best_c.components["peak_mrr_mm3_s"]),
                    "mrr_limit_mm3_s": bundle.mrr_limit_mm3_s,
                    "mrr_overload_pct": max(
                        0.0,
                        100.0 * (float(best_c.components["peak_mrr_mm3_s"]) / bundle.mrr_limit_mm3_s - 1.0),
                    ),
                    "feasible": bool(best_c.components["feasible"] == 1.0),
                }
            )

    ablation_payload = {
        "schema": "diffcam-pub-exp2-ablations/1",
        "budget_evaluations": budget_evals,
        "rows": ablation_rows,
    }
    _write_json(_safe_resolve_under(out_dir, "exp2_ablations.json"), ablation_payload)
    return ablation_payload


def run_exp3_train_worker(
    pool: SimulatorPool,
    dev_bundles: Sequence[MillingTaskBundle],
    out_dir: Path,
    *,
    epochs: int = 20,
    seeds: Sequence[int] = (0, 1, 2),
) -> None:
    """Exp3 Stage A: Train multi-task policies on 8 dev tasks (8 <= 11 Taichi instance cap)."""
    print("[Exp3-Train] Multi-task policy training on 8 dev tasks (SHAC vs PPO)...", flush=True)
    train_curves: Dict[str, List[Any]] = {"DiffPolicy-SHAC": [], "Hard-PPO": []}
    policies_serial: Dict[str, Any] = {}

    for seed in seeds:
        dev_ctx_shac = []
        dev_ctx_ppo = []
        for b in dev_bundles:
            sim, sampled_task, target_sdf = pool.build_for_bundle(b)
            spec = _make_objective_spec(b, k=35.0)
            st = _extract_states_for_bundle(b, sampled_task, sim, target_sdf, spec)
            amap_s = DirectSegmentActionMap(b.task, b.nominal_motion)
            amap_p = DirectSegmentActionMap(b.task, b.nominal_motion)
            dev_ctx_shac.append((_make_evaluator(b, sampled_task, sim, amap_s, spec, seed=seed), st))
            dev_ctx_ppo.append((_make_evaluator(b, sampled_task, sim, amap_p, spec, seed=seed), st))

        pol_shac, curve_shac = train_multitask_policy(
            dev_ctx_shac, mode="shac", epochs=epochs, lr=1.2e-2, seed=seed
        )
        pol_ppo, curve_ppo = train_multitask_policy(
            dev_ctx_ppo, mode="ppo", epochs=epochs, lr=4e-3, seed=seed
        )
        train_curves["DiffPolicy-SHAC"].append(curve_shac)
        train_curves["Hard-PPO"].append(curve_ppo)
        policies_serial[str(seed)] = {
            "shac_params": [float(v) for v in pol_shac.get_flat_params()],
            "shac_log_std": [float(v) for v in pol_shac.log_std],
            "ppo_params": [float(v) for v in pol_ppo.get_flat_params()],
            "ppo_log_std": [float(v) for v in pol_ppo.log_std],
        }
        print(f"  [Exp3-Train] Seed {seed} finished ({epochs} epochs across 8 dev tasks).", flush=True)

    _write_json(
        _safe_resolve_under(out_dir, "_exp3_policies.json"),
        {"epochs": epochs, "seeds": list(seeds), "train_curves": train_curves, "policies": policies_serial},
    )


def run_exp3_eval_worker(
    pool: SimulatorPool,
    test_bundles: Sequence[MillingTaskBundle],
    out_dir: Path,
    *,
    few_shot_evals: int = 16,
) -> Dict[str, Any]:
    """Exp3 Stage B: Zero-shot & 16-eval few-shot transfer on 8 held-out test tasks."""
    ckpt = json.loads((_safe_resolve_under(out_dir, "_exp3_policies.json")).read_text(encoding="utf-8"))
    epochs = int(ckpt["epochs"])
    seeds = [int(s) for s in ckpt["seeds"]]
    train_curves = ckpt["train_curves"]
    transfer_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        pinfo = ckpt["policies"][str(seed)]
        pol_shac = ResidualMillingPolicy()
        pol_shac.set_flat_params(np.asarray(pinfo["shac_params"], dtype=np.float64))
        pol_shac.log_std = np.asarray(pinfo["shac_log_std"], dtype=np.float64)

        pol_ppo = ResidualMillingPolicy()
        pol_ppo.set_flat_params(np.asarray(pinfo["ppo_params"], dtype=np.float64))
        pol_ppo.log_std = np.asarray(pinfo["ppo_log_std"], dtype=np.float64)

        for b in test_bundles:
            sim, sampled_task, target_sdf = pool.build_for_bundle(b)
            spec = _make_objective_spec(b, k=35.0)
            st = _extract_states_for_bundle(b, sampled_task, sim, target_sdf, spec)
            amap = DirectSegmentActionMap(b.task, b.nominal_motion)
            ev_base = _make_evaluator(b, sampled_task, sim, amap, spec, seed=seed)
            nom_res = ev_base.evaluate_candidate(np.zeros(amap.num_params, dtype=np.float64), "nom", "hard")
            nom_t = float(nom_res.components["cycle_time_s"])

            # 1. Zero-shot SHAC & Zero-shot PPO (1 single deterministic forward pass!)
            for label, pol in (("DiffPolicy-SHAC-ZeroShot", pol_shac), ("Hard-PPO-ZeroShot", pol_ppo)):
                t_zs = time.perf_counter()
                u_zs = amap.clip(pol.forward(st).reshape(-1))
                zs_res = ev_base.evaluate_candidate(u_zs, f"zs-{label}-{b.case.case_id}", "hard")
                dt_ms = 1000.0 * (time.perf_counter() - t_zs)
                ct = float(zs_res.components["cycle_time_s"])
                pk_mrr = float(zs_res.components["peak_mrr_mm3_s"])
                transfer_rows.append(
                    {
                        "seed": seed,
                        "case_id": b.case.case_id,
                        "family": b.task.family,
                        "regime": label,
                        "evaluations": 1,
                        "inference_ms": dt_ms,
                        "nominal_cycle_time_s": nom_t,
                        "cycle_time_s": ct,
                        "cycle_time_reduction_pct": 100.0 * (nom_t - ct) / max(nom_t, 1e-9),
                        "peak_mrr_mm3_s": pk_mrr,
                        "mrr_limit_mm3_s": b.mrr_limit_mm3_s,
                        "mrr_overload_pct": max(0.0, 100.0 * (pk_mrr / b.mrr_limit_mm3_s - 1.0)),
                        "feasible": bool(zs_res.components["feasible"] == 1.0),
                    }
                )

            # 2. Few-shot (16 evals) warm-started SHAC vs Few-shot PPO vs Scratch STE-Adam (16 evals)
            ev_fs_shac = _make_evaluator(b, sampled_task, sim, DirectSegmentActionMap(b.task, b.nominal_motion), spec, seed=seed)
            fs_shac = train_diff_policy_shac(
                ev_fs_shac, st, max_evaluations=few_shot_evals, lr=1.0e-2, seed=seed, initial_policy=pol_shac
            )
            ev_fs_ppo = _make_evaluator(b, sampled_task, sim, DirectSegmentActionMap(b.task, b.nominal_motion), spec, seed=seed)
            fs_ppo = train_hard_ppo(
                ev_fs_ppo, st, max_evaluations=few_shot_evals, batch_episodes=4, seed=seed, initial_policy=pol_ppo
            )
            sp_scr = make_joint_parameterization(b.task, b.nominal_motion)
            ev_scr = _make_evaluator(b, sampled_task, sim, sp_scr, spec, seed=seed)
            scr_ste = solve_ste_continuation_adam(ev_scr, max_evaluations=few_shot_evals, lr=0.085, seed=seed)

            for label, run_sum in (
                ("DiffPolicy-SHAC-FewShot16", fs_shac),
                ("Hard-PPO-FewShot16", fs_ppo),
                ("Scratch-STE-Adam-16", scr_ste),
            ):
                ct = float(run_sum.best_result.components["cycle_time_s"])
                pk_mrr = float(run_sum.best_result.components["peak_mrr_mm3_s"])
                transfer_rows.append(
                    {
                        "seed": seed,
                        "case_id": b.case.case_id,
                        "family": b.task.family,
                        "regime": label,
                        "evaluations": run_sum.evaluations,
                        "nominal_cycle_time_s": nom_t,
                        "cycle_time_s": ct,
                        "cycle_time_reduction_pct": 100.0 * (nom_t - ct) / max(nom_t, 1e-9),
                        "peak_mrr_mm3_s": pk_mrr,
                        "mrr_limit_mm3_s": b.mrr_limit_mm3_s,
                        "mrr_overload_pct": max(0.0, 100.0 * (pk_mrr / b.mrr_limit_mm3_s - 1.0)),
                        "feasible": bool(run_sum.best_result.components["feasible"] == 1.0),
                    }
                )

    payload = {
        "schema": "diffcam-pub-exp3-transfer/1",
        "epochs": epochs,
        "few_shot_evaluations": few_shot_evals,
        "seeds": list(seeds),
        "train_curves": train_curves,
        "test_transfer": transfer_rows,
    }
    _write_json(_safe_resolve_under(out_dir, "exp3_policy_transfer.json"), payload)
    print(f"[Exp3-Eval] Completed {len(transfer_rows)} held-out transfer evaluations.", flush=True)
    return payload


def _make_haas_shop_eval_task(
    fine_task: C.TaskSpec,
    *,
    z_home_above_datum_mm: float = 15.0,
) -> C.TaskSpec:
    """Anchor ``fine_task`` on a physical Haas Mini Mill table where ``G53 Z0.0`` is spindle home."""
    lo = fine_task.limits.envelope_min_mm
    hi = fine_task.limits.envelope_max_mm
    shop_limits = dataclasses.replace(
        fine_task.limits,
        envelope_min_mm=(lo[0], lo[1], lo[2] - z_home_above_datum_mm),
        envelope_max_mm=(hi[0], hi[1], max(0.0, hi[2] - z_home_above_datum_mm) + 10.0),
    )
    return dataclasses.replace(
        fine_task,
        work_to_machine=C.Transform.translation((0.0, 0.0, -z_home_above_datum_mm)),
        initial_state=C.MachineState(
            position_mm=(0.0, 0.0, 0.0),
            frame=C.Frame.MACHINE,
            length_offset_applied=fine_task.initial_state.length_offset_applied,
            work_offset=fine_task.initial_state.work_offset,
        ),
        limits=shop_limits,
    )


def run_exp4_camotics_and_haas_packets(
    exp2_main: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """Exp4: High-res 0.25 mm CAMotics v1.2.0 STL + 0.125 mm posted verification + Haas NGC packets."""
    print("[Exp4] Running 0.25 mm CAMotics v1.2.0 STL verification & Haas Mini Mill NGC packet export...", flush=True)
    fine_bundles = {
        b.case.case_id: b
        for b in build_milling_suite(voxel_mm=0.25, timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED)
    }
    stl_dir = _safe_resolve_under(out_dir, "camotics_stls")
    haas_dir = _safe_resolve_under(out_dir, "haas_ngc_packets")
    multi_dir = _safe_resolve_under(out_dir, "multi_controller_packets")
    stl_dir.mkdir(parents=True, exist_ok=True)
    haas_dir.mkdir(parents=True, exist_ok=True)
    multi_dir.mkdir(parents=True, exist_ok=True)

    camotics_rows: List[Dict[str, Any]] = []
    setup_rows: List[Dict[str, Any]] = []
    multi_controller_rows: List[Dict[str, Any]] = []

    test_task_entries = [t for t in exp2_main["tasks"] if t["split"] == "test"]
    for idx, entry in enumerate(test_task_entries):
        cid = entry["case_id"]
        fine_bundle = fine_bundles[cid]
        fine_task = fine_bundle.task
        fine_nom = fine_bundle.nominal_motion

        # Pick best feasible seed of STE-Continuation-Adam and DiffPolicy-SHAC
        ste_runs = [r for r in entry["methods"]["STE-Continuation-Adam"] if r["feasible"]]
        if not ste_runs:
            ste_runs = entry["methods"]["STE-Continuation-Adam"]
        best_ste = min(ste_runs, key=lambda r: r["cycle_time_s"])
        spline_fine = make_joint_parameterization(fine_task, fine_nom)
        theta_opt = np.asarray(best_ste["best_theta"], dtype=np.float64)

        # Build optimized MotionPlan on the fine 0.25 mm task
        positions = spline_fine.positions(theta_opt)
        feed_mult = spline_fine.forward_feed_multipliers(theta_opt)
        to_mach = fine_task.work_to_machine.apply
        raw_segs = []
        for orig, sw, ew, mult in zip(fine_nom.segments, positions[:-1], positions[1:], feed_mult):
            s_m, e_m = to_mach(sw), to_mach(ew)
            length = math.dist(s_m, e_m)
            eff_f = None if orig.kind == C.SegmentKind.RAPID else float(orig.feed_mm_s * mult)
            dur = length / (fine_task.limits.rapid_mm_s if eff_f is None else eff_f)
            raw_segs.append(
                C.Segment(
                    id=orig.id, kind=orig.kind, start_mm=s_m, end_mm=e_m,
                    duration_s=dur, feed_mm_s=eff_f, tag=orig.tag,
                )
            )
        durs = C.blended_segment_durations(
            raw_segs, fine_task.limits.rapid_mm_s, fine_task.limits.max_accel_mm_s2
        )
        opt_motion = C.MotionPlan(
            frame=C.Frame.MACHINE,
            initial_state=fine_task.initial_state,
            segments=tuple(dataclasses.replace(s, duration_s=d) for s, d in zip(raw_segs, durs)),
            timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
            provenance="diffcam:ste_continuation_adam",
        )
        opt_motion.validate_for_task(fine_task)

        cfg = MachineConfig(tool_number=fine_task.tool.tool_number, spindle_rpm=8000, coolant=True)
        for variant_label, plan in (("Nominal-CAM", fine_nom), ("DiffCAM-STE", opt_motion)):
            haas_nc = post_motion_plan(plan, fine_task, dialect=C.Dialect.HAAS, config=cfg)
            rs274_nc = post_motion_plan(plan, fine_task, dialect=C.Dialect.RS274, config=cfg)

            # 1. Fine 0.25 mm grid / 0.125 mm station step posted_evaluator certification
            posted_ev = evaluate_posted_gcode(
                fine_task,
                haas_nc.encode("utf-8"),
                dialect=C.Dialect.HAAS,
                settings={"max_step_mm": 0.125, "compute_bounds": False, "safety_screen": True},
            )

            # 2. CAMotics v1.2.0 Docker STL verification (retain STL mesh!)
            sub_scratch = _safe_resolve_under(stl_dir, f"{cid}_{variant_label}")
            sub_scratch.mkdir(parents=True, exist_ok=True)
            sub_adapter = CamoticsAdapter(default_resolution_mm=0.25, scratch_root=sub_scratch)
            cam_ev = sub_adapter.evaluate_posted_gcode(
                fine_task,
                haas_nc.encode("utf-8"),
                dialect=C.Dialect.HAAS,
                settings={"resolution_mm": 0.25, "retain_workdir": True},
            )
            stl_files = list(sub_scratch.glob("*/cut_workpiece.stl"))
            retained_stl = ""
            if stl_files:
                dest_stl = _safe_resolve_under(stl_dir, f"{cid}_{variant_label}.stl")
                shutil.copy2(stl_files[0], dest_stl)
                retained_stl = dest_stl.relative_to(out_dir.resolve()).as_posix()

            camotics_rows.append(
                {
                    "case_id": cid,
                    "family": fine_task.family,
                    "variant": variant_label,
                    "cycle_time_s": float(sum(s.duration_s for s in plan.segments)),
                    "posted_0p125mm_geometry_pass": posted_ev.geometry_pass,
                    "posted_0p125mm_safety_pass": posted_ev.safety_pass,
                    "posted_0p125mm_orig_gouge_mm3": posted_ev.metrics["original_part_gouge_volume_mm3"].value,
                    "posted_0p125mm_residual_mm3": posted_ev.metrics["residual_volume_mm3"].value,
                    "camotics_watertight": bool(
                        cam_ev.discretization.get("stl_topology", {}).get("watertight", False)
                    ),
                    "camotics_triangles": int(cam_ev.discretization.get("triangle_count", 0)),
                    "camotics_stl_volume_mm3": cam_ev.metrics["stl_padded_volume_mm3"].value,
                    "camotics_orig_gouge_mm3": cam_ev.metrics["original_part_gouge_volume_mm3"].value,
                    "camotics_residual_mm3": cam_ev.metrics["residual_volume_mm3"].value,
                    "stl_relpath": retained_stl,
                    "haas_nc_sha256": C.nc_hash(haas_nc.encode("utf-8")),
                    "rs274_nc_sha256": C.nc_hash(rs274_nc.encode("utf-8")),
                }
            )

            if variant_label == "DiffCAM-STE":
                prog_num = 1001 + idx
                shop_nc = format_haas_ngc_shop_program(
                    opt_motion,
                    fine_task,
                    cfg,
                    program_number=prog_num,
                    case_label=cid,
                )
                nc_path = _safe_resolve_under(haas_dir, f"O{prog_num:05d}_{cid}_diffcam_haas_ngc.nc")
                nc_path.write_text(shop_nc, encoding="utf-8")
                nom_shop_nc = format_haas_ngc_shop_program(
                    fine_nom,
                    fine_task,
                    cfg,
                    program_number=2001 + idx,
                    case_label=f"{cid}_nominal",
                )
                nom_nc_path = _safe_resolve_under(haas_dir, f"O{2001+idx:05d}_{cid}_nominal_haas_ngc.nc")
                nom_nc_path.write_text(nom_shop_nc, encoding="utf-8")

                shop_eval_task = _make_haas_shop_eval_task(fine_task)
                shop_ev = evaluate_posted_gcode(
                    shop_eval_task,
                    shop_nc.encode("utf-8"),
                    dialect=C.Dialect.HAAS,
                    settings={"max_step_mm": 0.125, "compute_bounds": False, "safety_screen": True},
                )
                nom_shop_ev = evaluate_posted_gcode(
                    shop_eval_task,
                    nom_shop_nc.encode("utf-8"),
                    dialect=C.Dialect.HAAS,
                    settings={"max_step_mm": 0.125, "compute_bounds": False, "safety_screen": True},
                )
                lo, hi = fine_task.stock.params["min_mm"], fine_task.stock.params["max_mm"]
                setup_rows.append(
                    {
                        "program_number": f"O{prog_num:05d}",
                        "file": nc_path.name,
                        "nominal_file": nom_nc_path.name,
                        "case_id": cid,
                        "family": fine_task.family,
                        "stock_dimensions_mm": [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]],
                        "g54_datum": "Stock top-surface XY center (X0.0 Y0.0 at billet center, Z0.0 at top face)",
                        "tool": "T1: 5.0 mm dia (R2.5 mm) flat carbide endmill, flute >= 12.0 mm, stickout >= 15.0 mm",
                        "smoothing_code": "G54 + G187 P2 E0.05 (Haas NGC Medium Smoothing, 0.05 mm corner rounding; G187 reset)",
                        "nominal_cycle_time_s": round(sum(s.duration_s for s in fine_nom.segments), 3),
                        "diffcam_cycle_time_s": round(sum(s.duration_s for s in opt_motion.segments), 3),
                        "posted_safety_pass": bool(posted_ev.safety_pass and shop_ev.safety_pass and nom_shop_ev.safety_pass),
                        "posted_geometry_pass": bool(posted_ev.geometry_pass and shop_ev.geometry_pass and nom_shop_ev.geometry_pass),
                        "shop_0p125mm_orig_gouge_mm3": shop_ev.metrics["original_part_gouge_volume_mm3"].value,
                        "shop_0p125mm_residual_mm3": shop_ev.metrics["residual_volume_mm3"].value,
                        "shop_nc_sha256": C.nc_hash(shop_nc.encode("utf-8")),
                        "nominal_shop_nc_sha256": C.nc_hash(nom_shop_nc.encode("utf-8")),
                        "camotics_watertight": bool(
                            cam_ev.discretization.get("stl_topology", {}).get("watertight", False)
                        ),
                    }
                )

                # 3. 6-Controller Clean-Room Dialect Export & Round-Trip Verification
                for ctrl in ControllerDialect:
                    spec_d = get_dialect_spec(ctrl)
                    sub_d = _safe_resolve_under(multi_dir, ctrl.value)
                    sub_d.mkdir(parents=True, exist_ok=True)
                    d_cfg = MachineConfig(
                        units="mm",
                        program_number=1001 + idx,
                        tool_number=fine_task.tool.tool_number,
                        length_offset=fine_task.tool.tool_number,
                        spindle_rpm=8000,
                        coolant=True,
                    )
                    d_shop_nc = post_shop_program(
                        opt_motion,
                        fine_task,
                        dialect=ctrl,
                        config=d_cfg,
                        case_label=cid,
                    )
                    d_ext = spec_d.file_extension
                    d_path = _safe_resolve_under(sub_d, f"{cid}_diffcam_{ctrl.value}{d_ext}")
                    d_path.write_text(d_shop_nc, encoding="utf-8")
                    parsed_shop = parse_shop_motion_plan(
                        d_shop_nc, shop_eval_task, dialect=ctrl, g28_home_z_mm=0.0
                    )
                    d_typed_nc = post_motion_plan(opt_motion, fine_task, dialect=ctrl, config=d_cfg)
                    d_ev = evaluate_posted_gcode(
                        fine_task,
                        d_typed_nc.encode("utf-8"),
                        dialect=ctrl,
                        settings={"max_step_mm": 0.25, "compute_bounds": False, "safety_screen": True},
                    )
                    multi_controller_rows.append(
                        {
                            "case_id": cid,
                            "dialect": ctrl.value,
                            "display_name": spec_d.display_name,
                            "shop_file": f"multi_controller_packets/{ctrl.value}/{d_path.name}",
                            "shop_segments": len(parsed_shop.segments),
                            "geometry_pass": bool(d_ev.geometry_pass),
                            "safety_pass": bool(d_ev.safety_pass),
                            "orig_gouge_mm3": float(d_ev.metrics["original_part_gouge_volume_mm3"].value),
                            "residual_mm3": float(d_ev.metrics["residual_volume_mm3"].value),
                            "cycle_time_s": float(d_ev.metrics["cycle_time_s"].value),
                            "shop_nc_sha256": C.nc_hash(d_shop_nc.encode("utf-8")),
                        }
                    )
        print(f"  [Exp4] Verified {cid} across 6 controller dialects + 0.125 mm posted_evaluator + CAMotics v1.2.0 STL.", flush=True)

    # Write operator setup sheet Markdown in haas_ngc_packets/SETUP_SHEET.md
    md_lines = [
        "# Haas Mini Mill (Next-Gen Control / NGC) Machining Setup Packet",
        "",
        "- **License:** MIT",
        "- **Controller Target:** Haas Mini Mill with Next-Gen Control (NGC)",
        "- **Motion & Corner Blending:** `G54` + `G187 P2 E0.05` (`0.05 mm` max corner rounding tolerance) + `G187` reset (`FORMAT_REAL` mandatory decimal points, `<= 3` XYZ / `<= 2` F metric decimals, `G90 G53 G0 Z0.0` home retract)",
        "- **Tooling (`T1` / `H1`):** `5.0 mm` diameter (`2.5 mm` radius) flat carbide endmill, flute length `12.0 mm`, holder clearance `>= 14.0 mm` above tip",
        "- **Workholding & `G54` Datum:** Vise with parallels; `G54 X0.0 Y0.0` at billet XY center, `G54 Z0.0` at top stock face; minimum `5.0 mm` jaw clearance below deepest cut (`Z = -4.0 mm`)",
        "- **Pre-Cut Safety Checklist:**",
        "  1. Verify `T1` tool length offset `H1` against stock top `Z0`.",
        "  2. Run Haas NGC Graphics mode verification (`Setting 191 / G187` active).",
        "  3. Execute initial air-cut `Z +20.0 mm` offset pass at `25%` Rapid override before machining.",
        "",
        "| Program | Case ID | Family | Stock (mm) | Nominal (s) | DiffCAM (s) | Posted Pass | CAMotics Watertight |",
        "|---|---|---|---|---:|---:|:---:|:---:|",
    ]
    for s in setup_rows:
        sx, sy, sz = s["stock_dimensions_mm"]
        md_lines.append(
            f"| `{s['program_number']}` | `{s['case_id']}` | `{s['family']}` | "
            f"`{sx:.0f}x{sy:.0f}x{sz:.0f}` | {s['nominal_cycle_time_s']:.2f} | "
            f"{s['diffcam_cycle_time_s']:.2f} | `{s['posted_geometry_pass'] and s['posted_safety_pass']}` | "
            f"`{s['camotics_watertight']}` |"
        )
    (_safe_resolve_under(haas_dir, "SETUP_SHEET.md")).write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    _write_json(_safe_resolve_under(haas_dir, "manifest.json"), {"programs": setup_rows})

    payload = {
        "schema": "diffcam-pub-exp4-camotics-haas/1",
        "camotics_image": "diffcam-camotics:1.2.0",
        "posted_step_mm": 0.125,
        "camotics_resolution_mm": 0.25,
        "verifications": camotics_rows,
        "haas_packets": setup_rows,
        "multi_controller_audit": multi_controller_rows,
    }
    _write_json(_safe_resolve_under(out_dir, "exp4_camotics_and_haas.json"), payload)
    return payload


def _build_worker_cmd(
    out_dir: Path,
    voxel_mm: float,
    budget: int,
    seeds: Sequence[int],
    cpu: bool,
    only: str,
    family: Optional[str] = None,
    case_id: Optional[str] = None,
    jobs: int = 2,
) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output",
        str(out_dir),
        "--voxel-mm",
        str(voxel_mm),
        "--budget",
        str(budget),
        "--jobs",
        str(jobs),
        "--seeds",
        *[str(s) for s in seeds],
        "--only",
        only,
    ]
    if cpu:
        cmd.append("--cpu")
    if family is not None:
        cmd.extend(["--family", family])
    if case_id is not None:
        cmd.extend(["--case-id", case_id])
    return cmd


def _spawn_worker(
    out_dir: Path,
    voxel_mm: float,
    budget: int,
    seeds: Sequence[int],
    cpu: bool,
    only: str,
    family: Optional[str] = None,
    case_id: Optional[str] = None,
    jobs: int = 2,
) -> None:
    subprocess.run(
        _build_worker_cmd(
            out_dir, voxel_mm, budget, seeds, cpu, only, family=family, case_id=case_id, jobs=jobs
        ),
        check=True,
    )


def _run_parallel_workers(cmds: Sequence[List[str]], max_parallel: int = 2) -> None:
    active: List[subprocess.Popen[bytes]] = []
    queue = list(cmds)
    while queue or active:
        while queue and len(active) < max_parallel:
            cmd = queue.pop(0)
            active.append(subprocess.Popen(cmd))
        time.sleep(0.5)
        still_running: List[subprocess.Popen[bytes]] = []
        for proc in active:
            rc = proc.poll()
            if rc is None:
                still_running.append(proc)
            elif rc != 0:
                for other in active:
                    if other.poll() is None:
                        other.terminate()
                raise subprocess.CalledProcessError(rc, proc.args)
        active = still_running


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "pub-20261110")
    parser.add_argument("--voxel-mm", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--only",
        choices=[
            "all",
            "exp1",
            "exp2",
            "exp2-family",
            "exp2-case",
            "exp2-ablations",
            "exp3",
            "exp3-train",
            "exp3-eval",
            "exp4",
        ],
        default="all",
    )
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--case-id", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Orchestrator modes spawn isolated subprocesses so each Taichi process stays small & fast.
    if args.only == "all":
        _spawn_worker(out_dir, args.voxel_mm, args.budget, args.seeds, args.cpu, "exp1", jobs=args.jobs)
        _spawn_worker(out_dir, args.voxel_mm, args.budget, args.seeds, args.cpu, "exp2", jobs=args.jobs)
        _spawn_worker(out_dir, args.voxel_mm, args.budget, args.seeds, args.cpu, "exp3", jobs=args.jobs)
        _spawn_worker(out_dir, args.voxel_mm, args.budget, args.seeds, args.cpu, "exp4", jobs=args.jobs)
        print(f"[DONE] Publication campaign artifacts written to {out_dir}", flush=True)
        return

    if args.only == "exp2":
        from benchmarks.milling_suite import MILLING_CASES

        case_cmds = [
            _build_worker_cmd(
                out_dir, args.voxel_mm, args.budget, args.seeds, args.cpu, "exp2-case", case_id=c.case_id
            )
            for c in MILLING_CASES
        ]
        _run_parallel_workers(case_cmds, max_parallel=max(1, args.jobs))
        merged_tasks: List[Dict[str, Any]] = []
        for c in MILLING_CASES:
            part = json.loads((_safe_resolve_under(out_dir, f"_exp2_case_{c.case_id}.json")).read_text(encoding="utf-8"))
            merged_tasks.extend(part["tasks"])
        main_payload = {
            "schema": "diffcam-pub-exp2-main/1",
            "budget_evaluations": args.budget,
            "seeds": list(args.seeds),
            "tasks": merged_tasks,
        }
        _write_json(_safe_resolve_under(out_dir, "exp2_main_benchmark.json"), main_payload)
        _spawn_worker(out_dir, args.voxel_mm, args.budget, args.seeds, args.cpu, "exp2-ablations")
        print("[Exp2] Completed merged 16-task main benchmark + 5-arm ablations.", flush=True)
        return

    if args.only == "exp3":
        _spawn_worker(out_dir, args.voxel_mm, args.budget, args.seeds, args.cpu, "exp3-train")
        _spawn_worker(out_dir, args.voxel_mm, args.budget, args.seeds, args.cpu, "exp3-eval")
        return

    if args.only == "exp4":
        exp2_main = json.loads((_safe_resolve_under(out_dir, "exp2_main_benchmark.json")).read_text(encoding="utf-8"))
        run_exp4_camotics_and_haas_packets(exp2_main, out_dir)
        print(f"[DONE] Exp4 artifacts written to {out_dir}", flush=True)
        return

    # Leaf GPU worker modes (<= 8 CSGSimulatorDelta instances per process)
    arch = ti.cpu if args.cpu else ti.cuda
    ti.init(arch=arch, offline_cache=True, default_fp=ti.f32)
    pool = SimulatorPool(precision="f32")

    bundles = build_milling_suite(
        voxel_mm=args.voxel_mm, timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED
    )
    dev_bundles = [b for b in bundles if b.task.split == C.Split.DEV]
    test_bundles = [b for b in bundles if b.task.split == C.Split.TEST]

    if args.only == "exp1":
        run_exp1_conditioning(pool, dev_bundles, out_dir)
    elif args.only == "exp2-family":
        if not args.family:
            raise ValueError("--family is required for --only exp2-family")
        fam_bundles = [b for b in bundles if b.task.family == args.family]
        run_exp2_family_worker(
            pool, fam_bundles, out_dir, family_name=args.family, budget_evals=args.budget, seeds=args.seeds
        )
    elif args.only == "exp2-case":
        if not args.case_id:
            raise ValueError("--case-id is required for --only exp2-case")
        sel_bundles = [b for b in bundles if b.case.case_id == args.case_id]
        if not sel_bundles:
            raise ValueError(f"Unknown case_id: {args.case_id!r}")
        run_exp2_family_worker(
            pool,
            sel_bundles,
            out_dir,
            family_name=sel_bundles[0].task.family,
            budget_evals=args.budget,
            seeds=args.seeds,
            case_out_name=f"_exp2_case_{args.case_id}.json",
        )
    elif args.only == "exp2-ablations":
        run_exp2_ablations_worker(pool, test_bundles, out_dir, budget_evals=args.budget)
    elif args.only == "exp3-train":
        run_exp3_train_worker(pool, dev_bundles, out_dir, seeds=args.seeds)
    elif args.only == "exp3-eval":
        run_exp3_eval_worker(pool, test_bundles, out_dir)


if __name__ == "__main__":
    main()
