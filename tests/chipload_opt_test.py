# SPDX-License-Identifier: MIT
"""Unit tests for 2.5D differentiable CWE, radial chip thinning, and chipload co-optimization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.chipload_opt import (
    ChiploadSpec,
    SubsegmentedCWEModel,
    _radial_chip_thinning_and_grad,
    analyze_plan_engagement,
    blended_durations_with_pullback,
    optimize_chipload_trajectory,
    radial_chip_thinning_factor,
    rollout_chipload_policy,
    train_chipload_ppo,
    train_chipload_shac,
)
from benchmarks import contracts as C
from benchmarks.milling_suite import build_milling_suite


@pytest.fixture(scope="module")
def suite_bundles():
    return build_milling_suite(
        voxel_mm=0.5,
        timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
    )


def test_radial_chip_thinning_formula_and_derivative() -> None:
    # At 50% stepover..full slotting (u >= 0.5), thinning factor saturates at 1.0
    assert float(radial_chip_thinning_factor(0.5)) == pytest.approx(1.0, abs=1e-4)
    assert float(radial_chip_thinning_factor(0.85)) == pytest.approx(1.0, abs=1e-4)
    # At 10% stepover, 2 * sqrt(0.1 * 0.9) = 0.6
    assert float(radial_chip_thinning_factor(0.10)) == pytest.approx(0.60, abs=2e-3)

    u_grid = np.array([0.04, 0.12, 0.25, 0.41, 0.65, 0.95], dtype=np.float64)
    psi, dpsi = _radial_chip_thinning_and_grad(u_grid)
    eps = 1e-6
    for i, u0 in enumerate(u_grid):
        p_hi, _ = _radial_chip_thinning_and_grad(np.array([u0 + eps]))
        p_lo, _ = _radial_chip_thinning_and_grad(np.array([u0 - eps]))
        fd = float(p_hi[0] - p_lo[0]) / (2.0 * eps)
        assert float(dpsi[i]) == pytest.approx(fd, abs=1e-6)
    assert float(psi[2]) == pytest.approx(2.0 * np.sqrt(0.25 * 0.75), abs=2e-3)


def test_blended_durations_with_pullback_exact_contract_and_fd_parity(suite_bundles) -> None:
    bundle = next(b for b in suite_bundles if b.case.case_id == "face-test-00")
    mdl = SubsegmentedCWEModel(bundle)
    _, deltas, speeds = mdl.decode(mdl.zero_theta())

    ref_durs = C.blended_segment_durations(
        mdl.base_motion.segments,
        bundle.task.limits.rapid_mm_s,
        bundle.task.limits.max_accel_mm_s2,
    )
    rng = np.random.default_rng(19)
    weights = rng.uniform(0.2, 1.5, size=len(deltas))
    tau, g_deltas, g_speeds = blended_durations_with_pullback(
        deltas, speeds, bundle.task.limits.max_accel_mm_s2, g_tau=weights
    )
    np.testing.assert_allclose(tau, np.asarray(ref_durs, dtype=np.float64), rtol=1e-11, atol=1e-11)
    assert g_deltas is not None and g_speeds is not None

    eps = 1e-6
    for k in (0, 3, len(deltas) // 2, len(deltas) - 1):
        for ax in range(3):
            dp = deltas.copy()
            dm = deltas.copy()
            dp[k, ax] += eps
            dm[k, ax] -= eps
            tp, _, _ = blended_durations_with_pullback(dp, speeds, bundle.task.limits.max_accel_mm_s2)
            tm, _, _ = blended_durations_with_pullback(dm, speeds, bundle.task.limits.max_accel_mm_s2)
            fd = float(np.dot(weights, tp - tm)) / (2.0 * eps)
            assert float(g_deltas[k, ax]) == pytest.approx(fd, rel=1e-6, abs=1e-7)

        sp = speeds.copy()
        sm = speeds.copy()
        sp[k] += eps
        sm[k] -= eps
        tp, _, _ = blended_durations_with_pullback(deltas, sp, bundle.task.limits.max_accel_mm_s2)
        tm, _, _ = blended_durations_with_pullback(deltas, sm, bundle.task.limits.max_accel_mm_s2)
        fd_s = float(np.dot(weights, tp - tm)) / (2.0 * eps)
        assert float(g_speeds[k]) == pytest.approx(fd_s, rel=1e-6, abs=1e-7)


def test_subsegmented_cwe_model_reverse_ad_matches_finite_differences(suite_bundles) -> None:
    bundle = next(b for b in suite_bundles if b.case.case_id == "face-test-00")
    mdl = SubsegmentedCWEModel(bundle)
    rng = np.random.default_rng(7)
    theta0 = rng.normal(0.0, 0.25, size=mdl.num_params)
    _, grad0, _ = mdl.loss_and_grad(theta0, use_ste=False)

    eps = 1e-6
    check_indices = [
        0,
        1,
        57,
        min(128, mdl.num_xy_params - 2),
        mdl.num_xy_params - 1,
        mdl.num_xy_params,
        mdl.num_xy_params + 5,
        mdl.num_params - 1,
    ]
    for idx in check_indices:
        th_p = theta0.copy()
        th_m = theta0.copy()
        th_p[idx] += eps
        th_m[idx] -= eps
        lp, _, _ = mdl.loss_and_grad(th_p, use_ste=False)
        lm, _, _ = mdl.loss_and_grad(th_m, use_ste=False)
        fd = (lp - lm) / (2.0 * eps)
        assert float(grad0[idx]) == pytest.approx(fd, rel=2e-6, abs=1e-7)


def test_analyze_plan_engagement_invariants_and_chipload_optimization(suite_bundles) -> None:
    bundle = next(b for b in suite_bundles if b.case.case_id == "pocket-test-00")
    nom_rep = analyze_plan_engagement(bundle, bundle.nominal_motion)
    nom_sum = nom_rep["summary"]

    assert nom_sum["engaged_time_s"] + nom_sum["air_cut_time_s"] + nom_sum["rapid_time_s"] == pytest.approx(
        nom_sum["total_time_s"], abs=2e-3
    )
    assert nom_sum["engaged_pct"] + nom_sum["air_cut_pct"] + nom_sum["rapid_pct"] == pytest.approx(
        100.0, abs=0.05
    )
    assert nom_sum["removed_volume_mm3"] > 1000.0
    assert nom_sum["dexel_gouge_mm3"] == 0.0
    for seg_rec, orig_seg in zip(nom_rep["segments"], bundle.nominal_motion.segments):
        if orig_seg.kind == C.SegmentKind.RAPID:
            assert seg_rec["engaged"] is False
            assert seg_rec["removed_mm3"] == 0.0

    opt_res = optimize_chipload_trajectory(bundle, num_iters=65, verify_posted=True)
    opt_sum = opt_res.telemetry["summary"]
    assert opt_res.posted_eval is not None
    assert opt_res.posted_eval["geometry_pass"] is True
    assert opt_res.posted_eval["safety_pass"] is True
    assert opt_res.posted_eval["original_part_gouge_mm3"] == 0.0

    # Must visibly reshape trajectory in safe zones, cut cycle + air time, and regulate chipload
    assert opt_res.max_xy_shift_mm > 0.75
    assert opt_sum["total_time_s"] < nom_sum["total_time_s"] * 0.55
    assert opt_sum["air_cut_time_s"] < nom_sum["air_cut_time_s"] * 0.65
    assert opt_sum["hex_rmse_mm"] < nom_sum["hex_rmse_mm"] * 0.55
    assert opt_sum["optimal_hex_pct"] > 65.0


def test_equal_budget_shac_beats_ppo_on_chipload_policy(suite_bundles) -> None:
    dev_pair = [b for b in suite_bundles if b.case.case_id in ("pocket-dev-00", "contour-dev-00")]
    test_case = next(b for b in suite_bundles if b.case.case_id == "pocket-test-01")
    spec = ChiploadSpec()

    # Equal budget of 24 simulator evaluations:
    # SHAC: 12 epochs * 2 tasks = 24 evals; PPO: 3 epochs * 8 rollouts = 24 evals
    shac_pol = train_chipload_shac(dev_pair, spec=spec, epochs=12, lr=0.04, seed=3)
    ppo_pol = train_chipload_ppo(dev_pair, spec=spec, epochs=3, pop_per_epoch=8, lr=0.02, seed=3)

    shac_out = rollout_chipload_policy(test_case, shac_pol, spec=spec, provenance="diffcam:shac_chipload")
    ppo_out = rollout_chipload_policy(test_case, ppo_pol, spec=spec, provenance="diffcam:ppo_chipload")

    assert shac_out.telemetry["summary"]["total_time_s"] < ppo_out.telemetry["summary"]["total_time_s"]
    assert shac_out.telemetry["summary"]["hex_rmse_mm"] <= ppo_out.telemetry["summary"]["hex_rmse_mm"] + 1e-3
