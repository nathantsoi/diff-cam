"""Unit checks for the experimental T3.4 mechanisms.

Most trial states are explicitly synthetic fixtures: they exercise mechanism
state/budget semantics, not a manufacturing result or solver advantage. The
last objective test uses the actual merged small CPU objective and captures its
executed tool positions. Its local-surface/safety availability stays unknown.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import runpy

import numpy as np
import pytest

from algorithms import continuation as M
from algorithms.blackbox import OptimizationBudget
from benchmarks import contracts as C


GRID = C.GridSpec(shape=(4, 4, 4), origin_mm=(-2.0, -2.0, -2.0), voxel_mm=1.0)


def context(k=20.0, shift=0.0, validator="synthetic-hard-revision/1"):
    return M.EvaluationContext(
        GRID, k, shift, validator,
        {"fixture_revision": "known-synthetic-metrics-v1", "limits_mm": 0.5},
    )


def path_for(theta):
    """Synthetic executed coordinate array (not a physical machining claim)."""
    values = np.asarray(theta, dtype=float).ravel()
    return np.array([[0.0, 0.0, 0.0], [values[0], 0.0, 0.0], [values[-1], 0.0, 0.0]])


def pair(soft=0.5, hard=0.8, *, epoch=None, gouge=0.0, residual=0.0,
         local_mm=0.0, holder=0.0, safety=True, theta=None, path=None,
         coverage=True, source_valid=True):
    epoch = context() if epoch is None else epoch
    theta = np.zeros(2) if theta is None else np.asarray(theta)
    path = path_for(theta) if path is None else path
    return M.PairedObservation(
        M.SoftObservation(epoch, soft, path=path, gradient=np.ones_like(theta),
                          theta_hash=M.parameter_hash(theta)),
        M.HardObservation(epoch, coverage, gouge, residual, local_mm, holder, safety,
                          hard, path=path, theta_hash=M.parameter_hash(theta),
                          source_valid=source_valid),
    )


def make_controller(events=None, *, max_retries=0, initial_lr=0.005, min_lr=0.0000625,
                    k_max=1e6, shrink=0.5, candidate_budget=50, hard_budget=60):
    events = {
        0.0: {"soft": 0.5, "hard": 0.75, "gouge_fraction": 0.0,
              "residual_fraction": 0.0, "local_surface_error_mm": 0.0,
              "holder_overlap_fraction": 0.0, "safety_pass": True},
        **(events or {})
    }
    seen = {"soft": [], "hard": []}
    epoch = context(k=min(20.0, k_max))

    def soft(theta, received, require_gradient):
        event = events.get(float(theta[0]), events[0.0])
        seen["soft"].append((float(theta[0]), received.k, received.loss_shift, require_gradient))
        return M.SoftObservation(received, event["soft"], path_for(theta),
                                 gradient=np.ones_like(theta), theta_hash=M.parameter_hash(theta))

    def hard(theta, received):
        event = events.get(float(theta[0]), events[0.0])
        seen["hard"].append((float(theta[0]), received.k, received.loss_shift))
        return M.HardObservation(
            received, True, event["gouge_fraction"], event["residual_fraction"],
            event["local_surface_error_mm"], event["holder_overlap_fraction"],
            event["safety_pass"], event["hard"], path_for(theta),
            theta_hash=M.parameter_hash(theta),
        )

    budget = OptimizationBudget(
        candidate_evaluations=candidate_budget, validation_interval=1,
        max_validation_calls=hard_budget,
    )
    config = M.BacktrackingConfig(
        learning_rate_min=min_lr, learning_rate_max=max(0.01, initial_lr),
        learning_rate_shrink=shrink, max_retries_per_step=max_retries,
        k_max=k_max,
    )
    policy = M.ConstraintPolicy(fraction_atol=1e-10, objective_atol=1e-10, objective_rtol=0.0)
    mechanism = M.BacktrackingMechanism(config, policy, budget, soft, hard)
    start = mechanism.initialize(np.array([0.0]), epoch, {"learning_rate": initial_lr})
    assert start.snapshot.observation is not None
    return mechanism, start.snapshot, seen


def advance(boundary, absolute=False):
    def proposal(snapshot, learning_rate):
        theta = np.full_like(snapshot.theta, boundary) if absolute else snapshot.theta + boundary
        return M.Proposal(
            theta, np.ones_like(snapshot.theta) * (learning_rate + 0.3),
            np.ones_like(snapshot.theta) * (learning_rate + 0.4),
            dict(snapshot.config, learning_rate=learning_rate), snapshot.context,
            snapshot.optimizer_step + 1,
        )
    return proposal


def assert_same_state(left, right):
    np.testing.assert_array_equal(left.theta, right.theta)
    np.testing.assert_array_equal(left.first_moment, right.first_moment)
    np.testing.assert_array_equal(left.second_moment, right.second_moment)
    assert left.config == right.config
    assert left.context == right.context
    assert left.optimizer_step == right.optimizer_step


def test_signed_ema_clip_and_median_precede_clipping():
    calibration = M.SignedLossCalibrator(eta=0.25, delta_max=0.1, disagreement_floor=1e-6)
    band = M.PhysicalBand("gouge_fraction", 0.0, 1.0, "explicit synthetic overcut band")
    overcut = calibration.update([pair(soft=0.5, hard=0.2)], band, 0.2)
    assert overcut.median_delta == pytest.approx(-0.3)
    assert overcut.shift == pytest.approx(0.2 * 0.75 - 0.1 * 0.25)
    assert overcut.shift < 0.2
    clipped = calibration.update([pair(soft=0.2, hard=0.8)], band, 0.0)
    assert clipped.median_delta == pytest.approx(0.6)
    assert clipped.clipped_delta == pytest.approx(0.1)
    assert clipped.shift == pytest.approx(0.025)
    differing = [pair(soft=100.0, hard=0.0), pair(soft=2.0, hard=3.0),
                 pair(soft=1.0, hard=201.0)]
    median = calibration.update(differing, band, 0.0)
    assert median.median_delta == pytest.approx(1.0)
    assert median.shift == pytest.approx(0.025)


def test_declared_physical_band_selection_and_empty_or_unknown_band():
    calibration = M.SignedLossCalibrator(0.5, 1.0, 1e-6)
    observations = [pair(soft=0.1, hard=0.5, gouge=0.0),
                    pair(soft=0.2, hard=0.8, gouge=0.1),
                    pair(soft=0.6, hard=1.4, gouge=0.2),
                    pair(soft=0.0, hard=100.0, gouge=0.3)]
    band = M.PhysicalBand("gouge_fraction", 0.1, 0.2,
                          "inclusive known near-overcut fixture band 0.1 to 0.2")
    result = calibration.update(observations, band, 0.1)
    assert result.selected_count == 2 and result.source_count == 4
    assert result.median_delta == pytest.approx((0.6 + 0.8) / 2)
    assert result.shift == pytest.approx(0.1 * 0.5 + 0.7 * 0.5)
    empty = calibration.update(observations,
                               M.PhysicalBand("residual_fraction", 1.0, 1.0,
                                              "declared residual near-completion band"),
                               0.3)
    assert not empty.updated and empty.shift == 0.3
    assert empty.reason == "empty_declared_physical_band"


def test_calibration_never_treats_equal_theta_as_identical_motion_or_grid():
    calibration = M.SignedLossCalibrator(0.25, 1.0, 1e-6)
    band = M.PhysicalBand("gouge_fraction", 0.0, 1.0, "physical diagnostic band")
    different = pair(path=np.zeros((3, 3)))
    changed = M.PairedObservation(
        different.soft,
        dataclasses.replace(different.hard, path=np.ones((3, 3))),
    )
    unequal = calibration.update([changed], band, 0.5)
    assert not unequal.updated and unequal.shift == 0.5
    without_path = M.PairedObservation(
        dataclasses.replace(different.soft, path=None), different.hard
    )
    missing = calibration.update([without_path], band, 0.5)
    assert missing.reason == "identical_executed_path_unavailable_or_mismatch"
    other_grid = C.GridSpec((4, 4, 4), (-1.0, -2.0, -2.0), 1.0)
    other_context = M.EvaluationContext(other_grid, 20.0, 0.0, "synthetic-hard-revision/1")
    with pytest.raises(ValueError, match="identical k/shift/grid/validator"):
        calibration.update([pair(path=None), pair(epoch=other_context, path=None)], band, 0.5)
    with pytest.raises(ValueError, match="explicit gouge/residual fraction"):
        M.PhysicalBand("signed_dice_gap", 0.0, 0.0, "reject a nonphysical Dice-gap selector")


def test_zero_gap_and_bounded_disagreement_without_zero_dice_denominator():
    calibration = M.SignedLossCalibrator(0.5, 0.2, 1e-8)
    band = M.PhysicalBand("gouge_fraction", 0.0, 0.0, "known zero-overcut band")
    result = calibration.update([pair(soft=0.5, hard=0.5)] * 10, band, 0.0)
    assert result.shift == 0.0 and result.median_delta == 0.0
    assert result.disagreement["bounded_gap"] == 0.0
    assert M.bounded_disagreement(0.0, 0.0, 1e-8)["absolute_bounded_gap"] == 0.0
    assert M.bounded_disagreement(10.0, 4.0, 1e-8)["bounded_gap"] == pytest.approx(6.0 / 14.0)
    assert M.bounded_disagreement(-1.0, 1.0, 1e-8)["bounded_gap"] == pytest.approx(-1.0)
    extreme = M.bounded_disagreement(1e300, -1e300, 1e-8)
    assert np.isfinite(extreme["bounded_gap"]) and extreme["bounded_gap"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        M.bounded_disagreement(1.0, 0.0, 0.0)


def test_unknown_hard_state_is_not_known_infeasible_or_safety_pass():
    incomplete = M.HardObservation(context(), False, 0.0, 0.0, None, 0.0, None,
                                   source_valid=True, raw_phi_hard=0.2)
    result = M.assess_constraints(incomplete, M.ConstraintPolicy())
    assert result.status is M.Feasibility.UNKNOWN
    assert "coverage_incomplete" in result.missing
    assert "local_surface_error_mm_unavailable" in result.missing
    unsafe = M.HardObservation(context(), True, 0.0, 0.0, 0.0, 0.0, False)
    known_bad = M.assess_constraints(unsafe, M.ConstraintPolicy())
    assert known_bad.status is M.Feasibility.INFEASIBLE
    assert known_bad.violation == 1.0
    strict_zero = M.ConstraintPolicy(max_local_surface_error_mm=0.0)
    assert M.assess_constraints(unsafe, strict_zero).violation == 1.0
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        M.HardObservation(context(), True, gouge_fraction=-0.01)
    with pytest.raises(ValueError, match="finite"):
        M.SoftObservation(context(), float("nan"))


def test_rejected_step_restores_exact_optimizer_state_and_retries_from_incumbent():
    events = {x: {"soft": x, "hard": 0.4, "gouge_fraction": 0.0,
                  "residual_fraction": 0.0, "local_surface_error_mm": 0.0,
                  "holder_overlap_fraction": 0.0, "safety_pass": True}
              for x in (0.0, 1.0, 0.5, 0.25)}
    mechanism, start, seen = make_controller(max_retries=2, events=events)
    original_theta = start.theta.copy()
    proposal_calls = []

    def propose(snapshot, learning_rate):
        proposal_calls.append((snapshot.theta.copy(), learning_rate))
        return advance(1.0)(snapshot, learning_rate)

    before = mechanism.budget_snapshot()
    result = mechanism.step(start, propose)
    assert not result.accepted and result.reason == "bounded_retries_exhausted"
    assert result.attempts == 3 and result.retries == 2
    assert len(proposal_calls) == 3
    assert all(np.array_equal(theta, original_theta) for theta, _ in proposal_calls)
    assert_same_state(result.snapshot, start)
    assert mechanism.budget_snapshot()["proposals"] == 3
    assert mechanism.budget_snapshot()["retries"] == 2
    # Initial observation plus three candidate observations, not a free incumbent.
    assert len(seen["soft"]) == len(seen["hard"]) == 4
    assert mechanism.budget_snapshot()["candidate_evaluations"] == 4
    assert mechanism.budget_snapshot()["hard_validation_calls"] == 4
    assert result.spent["candidate_evaluations"] == 3


def test_backtracking_shrinks_bounded_learning_rate_and_keeps_accepted_config():
    events = {0.0: {"soft": 0.16, "hard": 0.8, "gouge_fraction": 0.0,
                    "residual_fraction": 0.0, "local_surface_error_mm": 0.0,
                    "holder_overlap_fraction": 0.0, "safety_pass": True}}
    events[0.005] = dict(events[0.0], soft=0.25)    # first trial: worse
    events[0.0025] = dict(events[0.0], soft=0.0225)  # retry: better, smaller step
    mechanism, start, _ = make_controller(events=events, max_retries=2)
    rates = []

    def proposal(snapshot, rate):
        rates.append(rate)
        return advance(rate)(snapshot, rate)

    result = mechanism.step(start, proposal)
    assert result.accepted and result.acceptance_kind == "improvement"
    assert rates == [0.005, 0.0025]
    np.testing.assert_allclose(result.snapshot.theta, [0.0025])
    assert result.snapshot.learning_rate == 0.0025
    assert result.snapshot.config["learning_rate"] == 0.0025
    assert result.snapshot.optimizer_step == 1
    # Both actual Adam moments came from the accepted retry, not rejected 0.5.
    np.testing.assert_allclose(result.snapshot.first_moment, np.ones(1) * 0.3025)
    np.testing.assert_allclose(result.snapshot.second_moment, np.ones(1) * 0.4025)
    assert mechanism.budget_snapshot()["retries"] == 1


def test_infeasible_start_can_restore_or_least_violation_but_never_promote_bad_safety():
    events = {
        0.0: {"soft": 0.2, "hard": 0.5, "gouge_fraction": 0.4,
              "residual_fraction": 0.0, "local_surface_error_mm": 0.0,
              "holder_overlap_fraction": 0.0, "safety_pass": True},
        0.1: {"soft": 0.3, "hard": 0.5, "gouge_fraction": 0.2,
              "residual_fraction": 0.0, "local_surface_error_mm": 0.0,
              "holder_overlap_fraction": 0.0, "safety_pass": True},
        0.5: {"soft": 0.8, "hard": 0.5, "gouge_fraction": 0.0,
              "residual_fraction": 0.0, "local_surface_error_mm": 0.0,
              "holder_overlap_fraction": 0.0, "safety_pass": True},
    }
    mechanism, start, _ = make_controller(events=events)
    assert M.assess_constraints(start.observation.hard, mechanism.constraints).status is M.Feasibility.INFEASIBLE
    least = mechanism.step(start, advance(0.1, absolute=True))
    assert least.accepted and least.acceptance_kind == "least_violation"
    assert least.snapshot.observation.hard.gouge_fraction == 0.2
    recovered = mechanism.step(least.snapshot, advance(0.5, absolute=True))
    assert recovered.accepted and recovered.acceptance_kind == "recovered_feasible"
    hard = recovered.snapshot.observation.hard
    assert M.assess_constraints(hard, mechanism.constraints).status is M.Feasibility.FEASIBLE

    unsafe_events = {
        0.0: dict(events[0.5]),
        1.0: dict(events[0.5], soft=-100.0, safety_pass=False),
    }
    second, feasible, _ = make_controller(events=unsafe_events)
    rejected = second.step(feasible, advance(1.0))
    assert not rejected.accepted
    assert_same_state(rejected.snapshot, feasible)


def test_infeasible_local_surface_constraint_is_explicit_not_replaced_by_soft_loss():
    events = {
        0.0: {"soft": 1.0, "hard": 1.0, "gouge_fraction": 0.0,
              "residual_fraction": 0.0, "local_surface_error_mm": 0.0,
              "holder_overlap_fraction": 0.0, "safety_pass": True},
        0.1: {"soft": -20.0, "hard": -20.0, "gouge_fraction": 0.0,
              "residual_fraction": 0.0, "local_surface_error_mm": 1.0,
              "holder_overlap_fraction": 0.0, "safety_pass": True},
    }
    mechanism, start, _ = make_controller(events=events)
    result = mechanism.step(start, advance(0.1))
    assert not result.accepted
    assert_same_state(result.snapshot, start)


def test_changed_k_comparison_is_rejected_without_advancing_state():
    mechanism, start, seen = make_controller()
    original_soft = mechanism.soft_evaluator

    def changed_k(theta, received, require_gradient):
        if theta[0] != 2.0:
            return original_soft(theta, received, require_gradient)
        changed = dataclasses.replace(received, k=received.k + 1)
        observed = original_soft(theta, received, require_gradient)
        return dataclasses.replace(observed, context=changed)

    mechanism.soft_evaluator = changed_k
    result = mechanism.step(start, advance(2.0))
    assert not result.accepted
    assert "different k/shift/grid/validator" in result.reason
    assert_same_state(result.snapshot, start)
    # The invalid comparison was consumed honestly rather than hidden as free.
    assert len(seen["soft"]) == 2 and len(seen["hard"]) == 2


def test_mismatched_proposal_epoch_does_not_spend_an_evaluation():
    mechanism, start, seen = make_controller()
    other = dataclasses.replace(start.context, k=40.0)

    def proposal(snapshot, rate):
        return M.Proposal(snapshot.theta + 1.0, snapshot.first_moment,
                          snapshot.second_moment, dict(snapshot.config, learning_rate=rate),
                          other, snapshot.optimizer_step + 1)

    result = mechanism.step(start, proposal)
    assert not result.accepted and result.reason == "comparison_context_mismatch"
    assert len(seen["soft"]) == len(seen["hard"]) == 1
    assert_same_state(result.snapshot, start)


def test_unknown_start_blocks_repair_instead_of_relabeling_missing_state_infeasible():
    mechanism, start, _ = make_controller(events={
        0.0: {"soft": 0.2, "hard": 0.5, "gouge_fraction": 0.4,
              "residual_fraction": 0.0, "local_surface_error_mm": 0.0,
              "holder_overlap_fraction": 0.0, "safety_pass": True},
    })
    unknown = dataclasses.replace(start.observation.hard,
                                  coverage_complete=False, source_valid=False)
    start = start.with_observation(
        M.PairedObservation(start.observation.soft, unknown)
    )
    outcome = mechanism.step(start, advance(1.0))
    assert not outcome.accepted and outcome.reason == "initial_hard_validation_unknown"
    assert_same_state(outcome.snapshot, start)


def test_initial_gradient_refresh_is_budgeted_and_not_a_free_calibration():
    mechanism, start, seen = make_controller()
    no_gradient = dataclasses.replace(start.observation.soft, gradient=None)
    state = start.with_observation(
        M.PairedObservation(no_gradient, start.observation.hard)
    )
    result = mechanism.step(state, advance(1.0))
    assert result.reason in {"bounded_retries_exhausted", "learning_rate_bound_exhausted"}
    # initialize (1/1) + incumbent gradient refresh (1/1) + candidate (1/1).
    assert len(seen["soft"]) == 3 and any(call[3] for call in seen["soft"][1:])
    assert mechanism.budget_snapshot()["candidate_evaluations"] == 3
    assert mechanism.budget_snapshot()["hard_validation_calls"] == 3
    assert_same_state(result.snapshot, start)


def test_learning_rate_floor_stops_before_an_unbudgeted_extra_trial():
    mechanism, start, seen = make_controller(initial_lr=0.005, min_lr=0.001,
                                             shrink=0.1, max_retries=3)
    calls = []

    def proposal(snapshot, lr):
        calls.append(lr)
        return advance(0.1)(snapshot, lr)

    result = mechanism.step(start, proposal)
    assert not result.accepted and result.reason == "learning_rate_floor_reached"
    assert calls == [0.005]
    assert len(seen["soft"]) == len(seen["hard"]) == 2
    assert_same_state(result.snapshot, start)


def test_calibration_is_separate_from_k_and_does_not_charge_a_second_evaluation():
    mechanism, start, seen = make_controller()
    measurement = mechanism._observe(np.array([0.25]), start.context, False)
    before = mechanism.budget_snapshot()
    band = M.PhysicalBand("gouge_fraction", 0.0, 1.0,
                          "whole explicit known diagnostic domain")
    result = mechanism.calibrate([measurement, pair(soft=0.5, hard=0.5)], band, 0.0)
    assert result.updated
    assert result.median_delta == pytest.approx((0.25 + 0.0) / 2)
    assert result.shift == pytest.approx(0.03125)
    assert mechanism.budget_snapshot() == before
    assert start.context.k == 20.0 and start.context.loss_shift == 0.0
    assert mechanism.config.k_min == 1.0
    assert len(seen["soft"]) == 2 and len(seen["hard"]) == 2


def test_budget_blocks_new_calls_and_does_not_roll_back_spending():
    mechanism, start, seen = make_controller(candidate_budget=1, hard_budget=10)
    result = mechanism.step(start, advance(1.0))
    assert not result.accepted and result.reason == "candidate_budget_exhausted"
    assert len(seen["soft"]) == 1 and len(seen["hard"]) == 1
    assert_same_state(result.snapshot, start)
    assert result.spent["candidate_evaluations"] == 0


def test_no_hard_validation_budget_means_blocked_atomic_paired_observation():
    mechanism, start, seen = make_controller(candidate_budget=4, hard_budget=1)
    result = mechanism.step(start, advance(1.0))
    assert not result.accepted and result.reason == "hard_validation_budget_exhausted"
    # Neither soft nor hard was called after start when either half had no budget.
    assert len(seen["soft"]) == 1 and len(seen["hard"]) == 1
    assert result.spent["hard_validation_calls"] == 0
    assert_same_state(result.snapshot, start)


def test_zero_candidate_budget_blocks_initial_measurement():
    config = M.BacktrackingConfig()
    constraints = M.ConstraintPolicy()
    epoch = context()
    budget = M.OptimizationBudget(candidate_evaluations=0, validation_interval=1, max_validation_calls=2)

    def should_not_call(*_args):
        raise AssertionError("out-of-budget evaluator was invoked")

    mechanism = M.BacktrackingMechanism(config, constraints, budget, should_not_call, should_not_call)
    result = mechanism.initialize(np.zeros(1), epoch)
    assert result.reason == "candidate_budget_exhausted"
    assert not result.accepted and result.snapshot.observation is None


def test_dev_ablation_is_a_plan_only_with_paired_settings():
    plan = M.dev_ablation_plan()
    assert plan["status"] == "planned_only_not_executed"
    assert plan["actual_execution"] is None
    assert plan["shared_configuration"]["split"] == "dev"
    assert {row["reference"] for row in plan["comparisons"]} == {"fixed_adam", "annealed_adam"}
    assert all(row["configuration_sha256"] == plan["shared_configuration_sha256"]
               for row in plan["comparisons"])
    assert "diffcam-orchestrator GO" in plan["launch_requires"]
    plan["shared_configuration"]["optimizer"]["learning_rate"] = 99.0
    assert M.dev_ablation_plan()["shared_configuration"]["optimizer"]["learning_rate"] == 0.005


def test_bounded_sharpness_schedule_is_separate_from_signed_shift():
    fixed = M.SharpnessSchedule("fixed", 20.0, 20.0, 2)
    assert [fixed.at(step) for step in (0, 1, 2)] == [20.0, 20.0, 20.0]
    schedule = M.SharpnessSchedule("annealed", 20.0, 100.0, 3)
    assert [schedule.at(step) for step in (0, 1, 2, 3)] == [20.0, 60.0, 100.0, 100.0]
    assert schedule.bounds == (20.0, 100.0)
    early = schedule.context_at(0, GRID, -0.2, "hard-validator/1")
    late = schedule.context_at(1, GRID, -0.2, "hard-validator/1")
    assert early.k == 20.0 and late.k == 60.0
    assert early.loss_shift == late.loss_shift == -0.2
    assert early.grid == late.grid and early.validator_id == late.validator_id
    assert early != late  # the mechanism must compare each pair inside only one
    for invalid in (
        {"mode": "fixed", "k_start": 20.0, "k_end": 40.0, "n_steps": 2},
        {"mode": "annealed", "k_start": 0.0, "k_end": 40.0, "n_steps": 2},
        {"mode": "annealed", "k_start": 20.0, "k_end": 40.0, "n_steps": 0},
        {"mode": "adaptive", "k_start": 20.0, "k_end": 40.0, "n_steps": 2},
    ):
        with pytest.raises(ValueError):
            M.SharpnessSchedule(**invalid)
    with pytest.raises(ValueError, match="nonnegative"):
        schedule.at(-1)


def test_merged_objective_raw_pair_uses_exact_executed_path_and_preserves_unknown_constraints():
    helper = runpy.run_path(str(Path(__file__).with_name("objectives_test.py")))
    evaluator = helper["_prototype"](seed=111)
    theta = np.array([0.001, -0.002, 0.003, -0.004, 0.002, 0.001])
    evaluator.set_evaluation_k(20.0)
    epoch = M.EvaluationContext(
        evaluator.task.grid, 20.0, 0.0, "algorithms.objectives.forward_hard/merged",
        {"spec": dataclasses.asdict(evaluator.spec), "evaluator_seed": evaluator.seed},
    )
    soft_result = evaluator.evaluate_candidate_with_gradient(theta, "t3.4:raw-soft")
    soft_path = evaluator.sim.tool_pos.to_numpy().copy()
    soft = M.from_merged_soft(soft_result, epoch, path=soft_path)
    assert soft_result.result.valid and soft_result.gradient_theta.shape == theta.shape
    hard_result = evaluator.evaluate_candidate(theta, "t3.4:raw-hard", "hard")
    hard_path = evaluator.sim.tool_pos.to_numpy().copy()
    hard = M.from_merged_hard(hard_result, epoch, path=hard_path)
    measurement = M.PairedObservation(soft, hard)
    assert measurement.identical_executed_path is True
    normalized_expected = (
        evaluator.parameterization.positions(theta) - evaluator.task.grid.origin_mm
    ) / evaluator.task.grid.extent_mm
    # Taichi field storage is f32; this is conversion precision, not an exact
    # equality assertion between machine doubles and the simulator field.
    np.testing.assert_allclose(soft_path, normalized_expected, atol=1e-5, rtol=0.0)
    assert soft.raw_phi_soft == pytest.approx(soft_result.result.objective)
    assert hard.raw_phi_hard == pytest.approx(hard_result.objective)
    assert np.all(np.isfinite(M.bounded_disagreement(hard.raw_phi_hard,
                                                      soft.raw_phi_soft, 1e-6)["bounded_gap"]))
    assessment = M.assess_constraints(hard, M.ConstraintPolicy())
    assert assessment.status is M.Feasibility.UNKNOWN
    assert "coverage_incomplete" in assessment.missing
    assert "local_surface_error_mm_unavailable" in assessment.missing
    assert "safety_pass_unavailable" in assessment.missing
    band = M.PhysicalBand("gouge_fraction", 0.0, 1.0,
                          "entire known prototype fraction domain for a raw diagnostic")
    result = M.SignedLossCalibrator(0.25, 1e6, 1e-6).update([measurement], band, 0.0)
    assert result.updated and result.selected_count == 1
    expected_delta = hard_result.objective - soft_result.result.objective
    assert result.median_delta == pytest.approx(expected_delta, abs=1e-6)
    assert result.shift == pytest.approx(0.25 * expected_delta, abs=1e-6)
    evaluator.clear_evaluation_k()
    shifted_context = dataclasses.replace(epoch, loss_shift=0.25)
    with pytest.raises(ValueError, match="applied_shift"):
        M.from_merged_soft(soft_result, shifted_context, path=soft_path)
    shifted_result = dataclasses.replace(soft_result.result,
                                         objective=soft_result.result.objective + 0.25)
    converted = M.from_merged_soft(shifted_result, shifted_context, path=soft_path,
                                   applied_shift=0.25)
    assert converted.raw_phi_soft == pytest.approx(soft_result.result.objective)