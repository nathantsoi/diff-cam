"""Unit and finite-difference gradient tests for T7.1 RL policies and STE/CEM solvers."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import taichi as ti

from algorithms import objectives as O
from algorithms.rl_policy import (
    ACTION_DIM,
    STATE_DIM,
    DirectSegmentActionMap,
    ResidualMillingPolicy,
    ValueCritic,
    extract_segment_states,
    solve_hard_cem,
    solve_ste_continuation_adam,
    train_diff_policy_shac,
    train_hard_ppo,
)
from benchmarks.milling_suite import (
    build_milling_suite,
    make_joint_parameterization,
    materialize_sampled_sources,
    multi_pocket_fields,
)
from simulator.csg_simulator import CSGSimulatorDelta


def test_direct_segment_action_map_jacobian_and_invariants():
    bundle = build_milling_suite(voxel_mm=0.5, families=("pocket_island",), split="dev")[0]
    amap = DirectSegmentActionMap(bundle.task, bundle.nominal_motion)
    rng = np.random.default_rng(11)
    u = rng.uniform(-0.8, 0.8, size=amap.num_params)

    # Start and terminal positions must match nominal exactly, and Z must be identical at every step
    pos_cand = amap.positions(u)
    pos_nom = amap.reference_positions_mm
    np.testing.assert_allclose(pos_cand[0], pos_nom[0], atol=1e-12)
    np.testing.assert_allclose(pos_cand[-1], pos_nom[-1], atol=1e-12)
    np.testing.assert_allclose(pos_cand[:, 2], pos_nom[:, 2], atol=1e-12)

    w_d = rng.normal(size=(amap.num_steps, 3))
    w_m = rng.normal(size=(amap.num_steps,))

    def f_scalar(vec):
        return float(np.sum(w_d * amap.forward(vec)) + np.sum(w_m * amap.forward_feed_multipliers(vec)))

    analytic = amap.pullback_joint(w_d, w_m, u)
    eps = 1e-6
    fd = np.zeros_like(u)
    for i in range(0, u.size, 3):
        for off in range(3):
            idx = i + off
            if idx >= u.size:
                break
            e = np.zeros_like(u)
            e[idx] = eps
            fd[idx] = (f_scalar(u + e) - f_scalar(u - e)) / (2.0 * eps)

    np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=1e-8)


def test_mlp_policy_and_critic_backward_match_finite_differences():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(12, STATE_DIM))
    policy = ResidualMillingPolicy(STATE_DIM, hidden_dim=16, action_dim=ACTION_DIM, seed=5)
    weights_out = rng.normal(size=(12, ACTION_DIM))

    mu, cache = policy.forward_with_cache(X)
    grad_analytic = policy.backward(cache, weights_out)
    p0 = policy.get_flat_params()
    eps = 1e-6
    grad_fd = np.zeros_like(p0)
    for i in range(p0.size):
        p_plus = p0.copy()
        p_minus = p0.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        policy.set_flat_params(p_plus)
        v_plus = float(np.sum(weights_out * policy.forward(X)))
        policy.set_flat_params(p_minus)
        v_minus = float(np.sum(weights_out * policy.forward(X)))
        grad_fd[i] = (v_plus - v_minus) / (2.0 * eps)
    policy.set_flat_params(p0)
    np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-6, atol=1e-8)

    critic = ValueCritic(STATE_DIM, hidden_dim=16, seed=7)
    w_v = rng.normal(size=12)
    _, c_cache = critic.forward_with_cache(X)
    cg_analytic = critic.backward(c_cache, w_v)
    c0 = critic.get_flat_params()
    cg_fd = np.zeros_like(c0)
    for i in range(c0.size):
        cp = c0.copy()
        cm = c0.copy()
        cp[i] += eps
        cm[i] -= eps
        critic.set_flat_params(cp)
        vp = float(np.sum(w_v * critic.forward_with_cache(X)[0]))
        critic.set_flat_params(cm)
        vm = float(np.sum(w_v * critic.forward_with_cache(X)[0]))
        cg_fd[i] = (vp - vm) / (2.0 * eps)
    critic.set_flat_params(c0)
    np.testing.assert_allclose(cg_analytic, cg_fd, rtol=1e-6, atol=1e-8)


def test_shac_ppo_ste_and_cem_execute_and_improve_feasible_cycle_time(tmp_path):
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f32)
    bundle = build_milling_suite(voxel_mm=0.5, families=("contour_profile",), split="dev")[0]
    task = bundle.task
    motion = bundle.nominal_motion
    sources = materialize_sampled_sources(task, cache_root=tmp_path)
    sampled_task = dataclasses.replace(
        task, operation_target=dataclasses.replace(task.operation_target, geometry=sources["target"])
    )
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
        precision="f32",
    )
    sim.set_tool_start(tuple(norm_start))
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.tool_cut_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    sim.bake_target_grid()
    sim.set_target_volume()

    ref_time = sum(s.duration_s for s in motion.segments)
    spec = O.ObjectiveSpec(
        k=35.0,
        gouge_weight=4.0,
        residual_weight=2.0,
        holder_weight=50.0,
        time_weight=0.6,
        air_time_weight=0.25,
        mrr_weight=0.5,
        mrr_limit_mm3_s=bundle.mrr_limit_mm3_s,
        reference_cycle_time_s=ref_time,
    )

    # 1. Spline STE-Continuation-Adam & Hard-CEM
    spline = make_joint_parameterization(task, motion)
    ev_ste = O.CandidateEvaluator(sampled_task, motion, spline, sim, spec=spec, requested_precision="f32")
    ev_ste.task = task
    ste_res = solve_ste_continuation_adam(ev_ste, max_evaluations=8, seed=0)
    assert ste_res.evaluations == 8
    assert ste_res.best_result.components["feasible"] == 1.0
    assert ste_res.best_result.objective <= ste_res.initial_result.objective

    ev_cem = O.CandidateEvaluator(sampled_task, motion, spline, sim, spec=spec, requested_precision="f32")
    ev_cem.task = task
    cem_res = solve_hard_cem(ev_cem, max_evaluations=9, population_size=4, seed=0)
    assert cem_res.evaluations == 9
    assert cem_res.best_result.components["feasible"] == 1.0

    # 2. Policy DiffPolicy-SHAC & Hard-PPO
    amap = DirectSegmentActionMap(task, motion)
    ev_shac = O.CandidateEvaluator(sampled_task, motion, amap, sim, spec=spec, requested_precision="f32")
    ev_shac.task = task
    # Probe nominal engagement once
    _ = ev_shac.evaluate_candidate(np.zeros(amap.num_params), "probe", "hard")
    nom_eng = sim.seg_engage.to_numpy()[: amap.num_steps].copy()
    _, target_sdf = multi_pocket_fields(task.stock, task.operation_target.geometry, task.grid)
    states = extract_segment_states(task, motion, target_sdf, nom_eng, bundle.mrr_limit_mm3_s)

    ev_shac = O.CandidateEvaluator(sampled_task, motion, amap, sim, spec=spec, requested_precision="f32")
    ev_shac.task = task
    shac_res = train_diff_policy_shac(ev_shac, states, max_evaluations=8, seed=0)
    assert shac_res.evaluations == 8
    assert shac_res.best_result.components["feasible"] == 1.0
    assert shac_res.best_result.objective < shac_res.initial_result.objective

    ev_ppo = O.CandidateEvaluator(sampled_task, motion, amap, sim, spec=spec, requested_precision="f32")
    ev_ppo.task = task
    ppo_res = train_hard_ppo(ev_ppo, states, max_evaluations=10, batch_episodes=4, seed=0)
    assert ppo_res.evaluations == 10
    assert ppo_res.best_result.components["feasible"] == 1.0
