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


_FIELD_SOURCE = None


def field_source():
    """One actual hard-forward field, shared as immutable unit fixture data."""
    global _FIELD_SOURCE
    if _FIELD_SOURCE is None:
        evaluator = prototype()
        theta = unit_theta()
        result = evaluator.evaluate_candidate(theta, "field-source-hard", "hard")
        assert result.valid
        grid = evaluator.task.grid
        _FIELD_SOURCE = {
            "grid": grid,
            "hard": evaluator.sim.stock.to_numpy()[
                evaluator.parameterization.num_steps
            ].copy(),
            "target": evaluator.sim.target.to_numpy().copy(),
            "path": evaluator.sim.tool_pos.to_numpy().copy(),
            "value_hash": result.theta_hash,
            "context": M.EvaluationContext(
                grid, 20.0, 0.0, "prototype-hard-stock-d-field/1",
                {"actual_source": "forward_hard.stock_d[T]", "theta_hash": result.theta_hash},
            ),
        }
    return _FIELD_SOURCE


def stock_field_fixture(*, epoch=None, offset=0.0, soft=None, target=None, hard=None):
    """Actual hard stock_d, with only the declared nonmask field offset."""
    source = field_source()
    epoch = source["context"] if epoch is None else epoch
    raw_hard = source["hard"] if hard is None else hard
    selected_soft = source["hard"] + offset if soft is None else soft
    target_sdf = source["target"] if target is None else target
    return M.StockFieldPair(
        epoch, raw_hard, selected_soft, target_sdf,
        source["path"], source["path"],
    )


def band(region="stock", lo=0.0, hi=2.0):
    return M.SurfaceBand(region, lo, hi,
                         "explicit near-surface millimetre band on the declared test grid")


def test_true_distance_field_recovers_uniform_quarter_voxel_offset():
    field = stock_field_fixture(offset=0.25)
    assert np.unique(field.hard_phi_voxels).size > 2  # not a +/-1 occupancy indicator
    calibration = M.SignedFieldCalibrator(1.0, 0.5, 1e-6)
    result = calibration.update([field], band("stock"), 0.0)
    assert result.median_delta == pytest.approx(-0.25, abs=1e-12)
    assert result.clipped_delta == pytest.approx(-0.25, abs=1e-12)
    assert result.shift == pytest.approx(-0.25, abs=1e-12)
    assert result.delta_mm == pytest.approx(-0.25 * field.context.grid.voxel_mm)
    assert result.shift_mm == pytest.approx(-0.25 * field.context.grid.voxel_mm)
    assert result.band["maximum_voxels"] == pytest.approx(2.0 / field.context.grid.voxel_mm)
    assert result.hard_distance_definition.startswith("forward_hard.stock_d")
    with pytest.raises(ValueError, match="0.5 voxel"):
        M.SignedFieldCalibrator(0.25, 0.6, 1e-6)
    overcut = calibration.update([stock_field_fixture(offset=0.8)], band("stock"), 0.2)
    assert overcut.clipped_delta == -0.5
    assert overcut.shift == pytest.approx(-0.5)


def test_over_erosion_sign_is_negative_in_both_gouge_and_residual_bands():
    field = stock_field_fixture(offset=0.25)
    calibration = M.SignedFieldCalibrator(1.0, 0.5, 1e-6)
    for region in ("gouge", "residual"):
        selected = band(region, 0.0, 2.0)
        assert np.any(field.selection(selected))
        measured = calibration.update([field], selected, 0.0)
        assert measured.updated
        assert measured.shift < 0
        assert measured.median_delta == pytest.approx(-0.25)
        assert measured.selected_observations == 1
        assert measured.selected_count == int(field.selection(selected).sum())


def test_target_sign_regions_and_both_sided_aliases_match_exact_band_predicate():
    field = stock_field_fixture()
    pitch = field.context.grid.voxel_mm
    lo, hi = 0.0 / pitch, 3.0 / pitch
    common = (
        (np.abs(field.target_sdf) >= lo)
        & (np.abs(field.target_sdf) <= hi)
        & (np.abs(field.hard_phi_voxels) <= hi)
    )
    signs = {"gouge": field.target_sdf < 0, "residual": field.target_sdf > 0}
    for region, sign in signs.items():
        expected = common & sign
        np.testing.assert_array_equal(field.selection(band(region, 0.0, 3.0)), expected)
    either = common & ((field.target_sdf < 0) | (field.target_sdf > 0))
    for region in ("stock", "both", "surface"):
        np.testing.assert_array_equal(field.selection(band(region, 0.0, 3.0)), either)
    assert np.any(either & (field.hard_phi_voxels < 0))
    assert np.any(either & (field.hard_phi_voxels > 0))
    assert np.any(signs["gouge"] & common)
    assert np.any(signs["residual"] & common)


def test_physical_surface_band_uses_mm_distances_and_rejects_empty_regions():
    field = stock_field_fixture(offset=0.15)
    selected_band = band("stock", 0.5, 1.0)
    selected = field.selection(selected_band)
    assert selected.any()
    assert np.all(field.surface_distance_mm[selected] >= 0.5)
    assert np.all(field.surface_distance_mm[selected] <= 1.0)
    calibration = M.SignedFieldCalibrator(0.5, 0.5, 1e-6)
    updated = calibration.update([field], selected_band, 0.0)
    assert updated.updated
    assert updated.selected_count == int(selected.sum())
    assert updated.median_delta == pytest.approx(-0.15)
    assert updated.shift < 0
    between = np.indices(field.context.grid.shape, dtype=float)[2] - 0.5
    narrow_field = stock_field_fixture(target=between)
    with pytest.raises(ValueError, match="empty selected surface band"):
        calibration.update([narrow_field], band("stock", 0.0, 0.49), 0.4)
    with pytest.raises(ValueError, match="surface band region"):
        M.SurfaceBand("signed_dice_gap", 0.0, 0.0, "not a physical region")


def test_field_types_reject_indicators_and_mismatched_coordinates():
    with pytest.raises(TypeError, match="forward_hard stock_d"):
        stock_field_fixture(hard=np.ones(GRID.shape, dtype=bool))
    with pytest.raises(TypeError, match="soft phi"):
        stock_field_fixture(soft=np.ones(GRID.shape, dtype=bool))
    with pytest.raises(TypeError, match="target phi"):
        stock_field_fixture(target=np.zeros(GRID.shape, dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        stock_field_fixture(hard=np.ones((1, 1, 1), dtype=float))
    with pytest.raises(ValueError, match="material and removal"):
        stock_field_fixture(target=np.zeros(field_source()["grid"].shape, dtype=float))
    ordinary = stock_field_fixture()
    with pytest.raises(ValueError, match="not identical"):
        dataclasses.replace(ordinary, hard_path=np.ones((3, 3)))
    with pytest.raises(TypeError, match="scalar losses are not phi"):
        M.SignedFieldCalibrator(0.5, 0.5, 1e-8).update([pair()], band(), 0.0)
    origin = ordinary.context.grid.origin_mm
    other_grid = dataclasses.replace(
        ordinary.context.grid, origin_mm=(origin[0] + 1.0, origin[1], origin[2]),
    )
    other_context = dataclasses.replace(ordinary.context, grid=other_grid)
    with pytest.raises(ValueError, match="identical k/shift/grid/validator"):
        M.SignedFieldCalibrator(0.5, 0.5, 1e-8).update(
            [ordinary, stock_field_fixture(epoch=other_context)], band(), 0.0,
        )


def test_nonzero_normalized_disagreement_uses_median_fields_not_mean_gap_over_zero():
    field = stock_field_fixture(offset=0.6)
    calibration = M.SignedFieldCalibrator(1.0, 0.5, 1e-6)
    selected_band = band("stock")
    selected = field.selection(selected_band)
    hard_median = float(np.median(field.hard_phi_voxels[selected]))
    soft_median = float(np.median(field.soft_phi_voxels[selected]))
    result = calibration.update([field], selected_band, 0.0)
    assert result.median_hard_voxels == pytest.approx(hard_median)
    assert result.median_soft_voxels == pytest.approx(soft_median)
    assert result.median_delta == pytest.approx(-0.6)
    assert result.clipped_delta == -0.5
    expected = (
        (hard_median - soft_median)
        / (abs(hard_median) + abs(soft_median) + 1e-6)
    )
    assert result.disagreement["bounded_gap"] == pytest.approx(expected)
    assert result.disagreement["raw_gap"] == pytest.approx(hard_median - soft_median)
    assert abs(expected) < 1.0
    assert M.bounded_disagreement(-1.0, -0.4, 1e-6)["bounded_gap"] == pytest.approx(-3.0 / 7.0)


def test_large_k_near_equality_does_not_bind_clip():
    evaluator = prototype()
    evaluator.set_evaluation_k(256.0)
    theta = unit_theta()
    soft = evaluator.evaluate_candidate_with_gradient(theta, "t3.4b:large-k-soft")
    soft_path = evaluator.sim.tool_pos.to_numpy().copy()
    soft_field = evaluator.sim.stock.to_numpy()[evaluator.parameterization.num_steps].copy()
    hard = evaluator.evaluate_candidate(theta, "t3.4b:large-k-hard", "hard")
    hard_path = evaluator.sim.tool_pos.to_numpy().copy()
    hard_field = evaluator.sim.stock.to_numpy()[evaluator.parameterization.num_steps].copy()
    assert soft.result.valid and hard.valid
    np.testing.assert_array_equal(soft_path, hard_path)
    epoch = M.EvaluationContext(evaluator.task.grid, 256.0, 0.0, "large-k-matching-stock-fields/1")
    field = M.StockFieldPair(
        epoch, hard_field, soft_field, evaluator.sim.target.to_numpy(), hard_path, soft_path,
    )
    selected = band("stock", 0.0, 3.0)
    reconcile = M.SignedFieldCalibrator(1.0, 0.5, 1e-8).update([field], selected, 0.0)
    assert reconcile.updated and reconcile.selected_count > 0
    actual_gap = float(np.median(
        field.hard_phi_voxels[field.selection(selected)]
        - field.soft_phi_voxels[field.selection(selected)]
    ))
    assert reconcile.median_delta == pytest.approx(actual_gap, abs=1e-6)
    assert abs(actual_gap) < 0.15
    assert abs(reconcile.clipped_delta) < 0.5
    assert reconcile.shift == pytest.approx(actual_gap, abs=1e-6)
    evaluator.clear_evaluation_loss_shift()
    evaluator.clear_evaluation_k()


def test_zero_loss_gap_and_bounded_disagreement_avoid_zero_dice_denominator():
    field = stock_field_fixture(offset=0.0)
    result = M.SignedFieldCalibrator(0.5, 0.2, 1e-8).update([field], band(), 0.0)
    assert result.median_delta == 0.0 and result.shift == 0.0
    assert result.disagreement["bounded_gap"] == 0.0
    assert M.bounded_disagreement(0.0, 0.0, 1e-8)["absolute_bounded_gap"] == 0.0
    assert M.bounded_disagreement(10.0, 4.0, 1e-8)["bounded_gap"] == pytest.approx(6.0 / 14.0)
    assert M.bounded_disagreement(-1.0, 1.0, 1e-8)["bounded_gap"] == pytest.approx(-1.0)
    extreme = M.bounded_disagreement(1e300, -1e300, 1e-8)
    assert np.isfinite(extreme["bounded_gap"]) and extreme["bounded_gap"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        M.bounded_disagreement(1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        M.EvaluationContext(GRID, 20.0, 0.75, "x")


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
    events[0.005] = dict(events[0.0], soft=0.25, hard=0.8)  # worse raw hard score
    events[0.0025] = dict(events[0.0], soft=0.0225, hard=0.4)  # better raw hard score
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
    # One scalar observation is explicitly billed by the controller. The field
    # inputs supplied below are separate known calibration fixtures; their
    # construction is independent of the pure arithmetic calibration delta.
    _ = mechanism._observe(np.array([0.25]), start.context, False)
    before = mechanism.budget_snapshot()
    band = M.SurfaceBand("stock", 0.0, 2.0, "whole declared near-surface fixture domain")
    field_a = stock_field_fixture(offset=0.25)
    field_b = stock_field_fixture()
    result = mechanism.calibrate([field_a, field_b], band, 0.0)
    assert result.updated
    assert result.median_delta == pytest.approx((-0.25 + 0.0) / 2)
    assert result.shift == pytest.approx(-0.03125)
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


def test_acceptance_rank_uses_raw_hard_objective_not_scalar_additions_of_shift():
    mechanism, start, _ = make_controller()
    original_soft = mechanism.soft_evaluator

    def field_configured_soft(theta, received, require_gradient):
        observed = original_soft(theta, received, require_gradient)
        # Deliberately vary a scalar diagnostic: it must not become the hard
        # acceptance objective or reintroduce the old constant-shift artefact.
        return dataclasses.replace(
            observed, raw_phi_soft=observed.raw_phi_soft + received.loss_shift
        )

    mechanism.soft_evaluator = field_configured_soft
    rolled_back = mechanism.step(start, advance(1.0))
    assert not rolled_back.accepted
    assert rolled_back.history[-1]["incumbent_rank"][-1] == pytest.approx(0.75)
    assert rolled_back.history[-1]["candidate_rank"][-1] == pytest.approx(0.75)
    shifted = dataclasses.replace(start.context, loss_shift=0.4)
    also_rolled_back = mechanism.step(start, advance(1.0), context=shifted)
    assert not also_rolled_back.accepted
    assert also_rolled_back.history[-1]["incumbent_rank"][-1] == pytest.approx(0.75)
    assert also_rolled_back.history[-1]["candidate_rank"][-1] == pytest.approx(0.75)
    assert_same_state(also_rolled_back.snapshot, start)


def test_production_rank_override_keeps_legacy_constraints_unmodified():
    mechanism, _start, _seen = make_controller()
    ordinary = M.assess_constraints(pair().hard, M.ConstraintPolicy())
    assert ordinary.status is M.Feasibility.FEASIBLE
    assert ordinary.hard_rank is None
    assert mechanism._rank(ordinary, 0.0, 0.8) == (0, 0.0, 0.0, 0.8)
    exact = (0, 0.0, 0.0, 0.0, 0.375)
    attached = M.ConstraintAssessment(M.Feasibility.FEASIBLE, 0.0, (), exact)
    assert mechanism._rank(attached, 999.0, 0.8) == exact
    unknown = M.ConstraintAssessment(M.Feasibility.UNKNOWN, 0.0, ("missing",))
    assert mechanism._rank(unknown, 0.0, 0.1) is None


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


def prototype():
    helper = runpy.run_path(str(Path(__file__).with_name("objectives_test.py")))
    return helper["_prototype"](seed=111)


def unit_theta():
    return np.array([0.001, -0.002, 0.003, -0.004, 0.002, 0.001])


def test_merged_stock_fields_and_raw_hard_are_actual_measurements():
    evaluator = prototype()
    evaluator.set_evaluation_k(20.0)
    epoch = M.EvaluationContext(
        evaluator.task.grid, 20.0, 0.0, "algorithms.objectives.forward_hard/merged",
        {"spec": dataclasses.asdict(evaluator.spec), "evaluator_seed": evaluator.seed},
    )
    theta = unit_theta()
    soft_result = evaluator.evaluate_candidate_with_gradient(theta, "t3.4b:raw-soft")
    soft_path = evaluator.sim.tool_pos.to_numpy().copy()
    soft_stock = evaluator.sim.stock.to_numpy()[evaluator.parameterization.num_steps].copy()
    hard_result = evaluator.evaluate_candidate(theta, "t3.4b:raw-hard", "hard")
    hard_path = evaluator.sim.tool_pos.to_numpy().copy()
    hard_stock = evaluator.sim.stock.to_numpy()[evaluator.parameterization.num_steps].copy()
    field = M.StockFieldPair(
        epoch, hard_stock, soft_stock, evaluator.sim.target.to_numpy(),
        hard_path, soft_path,
    )
    assert field.identical_executed_path
    assert np.unique(field.hard_phi_voxels).size > 2
    assert np.all(np.isfinite(field.hard_phi_voxels))
    normalized_expected = (
        evaluator.parameterization.positions(theta) - evaluator.task.grid.origin_mm
    ) / evaluator.task.grid.extent_mm
    np.testing.assert_allclose(soft_path, normalized_expected, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(hard_path, normalized_expected, atol=1e-5, rtol=0.0)
    selected_band = M.SurfaceBand("stock", 0.0, 3.0,
                                  "within 3 mm of the declared target boundary")
    assert np.any(field.selection(selected_band))
    result = M.SignedFieldCalibrator(0.25, 0.5, 1e-6).update(
        [field], selected_band, 0.0
    )
    assert result.updated
    assert result.voxel_mm == evaluator.task.grid.voxel_mm
    selected = field.selection(selected_band)
    expected_delta = float(np.median(
        field.hard_phi_voxels[selected] - field.soft_phi_voxels[selected]
    ))
    assert result.median_delta == pytest.approx(expected_delta, abs=1e-6)
    assert result.shift == pytest.approx(0.25 * np.clip(expected_delta, -0.5, 0.5), abs=1e-6)
    assert result.shift_mm == pytest.approx(result.shift * evaluator.task.grid.voxel_mm)
    assert result.disagreement["raw_gap"] == pytest.approx(
        result.median_hard_voxels - result.median_soft_voxels, abs=1e-6
    )
    hard = M.from_merged_hard(hard_result, epoch, path=hard_path)
    assert hard.raw_phi_hard == pytest.approx(hard_result.objective)
    assessment = M.assess_constraints(hard, M.ConstraintPolicy())
    assert assessment.status is M.Feasibility.UNKNOWN
    assert "coverage_incomplete" in assessment.missing
    assert "local_surface_error_mm_unavailable" in assessment.missing
    assert "safety_pass_unavailable" in assessment.missing


def test_shift_changes_gradient_not_hard_and_neutral_reset_restores_baseline():
    evaluator = prototype()
    evaluator.set_evaluation_k(20.0)
    theta = unit_theta()
    baseline = evaluator.evaluate_candidate_with_gradient(theta, "t3.4b:neutral")
    baseline_path = evaluator.sim.tool_pos.to_numpy().copy()
    assert baseline.result.valid and baseline.gradient_theta.shape == theta.shape
    assert np.all(np.isfinite(baseline.gradient_theta))
    evaluator.set_evaluation_loss_shift(0.4)
    shifted = evaluator.evaluate_candidate_with_gradient(theta, "t3.4b:shifted")
    assert shifted.result.valid
    delta = shifted.gradient_theta - baseline.gradient_theta
    assert np.linalg.norm(delta) > 1e-7
    assert shifted.result.components["loss_shift_voxels"] == pytest.approx(0.4)
    assert shifted.result.components["loss_shift_mm"] == pytest.approx(0.4 * evaluator.sim.v)
    assert np.array_equal(shifted.diagnostics["final_stock"], baseline.diagnostics["final_stock"])
    shifted_repeat = evaluator.evaluate_candidate_with_gradient(theta, "t3.4b:shifted-repeat")
    evaluator.clear_evaluation_loss_shift()
    restored = evaluator.evaluate_candidate_with_gradient(theta, "t3.4b:neutral-restored")
    # Eight repeated neutral CPU forwards measured 1.1e-9--4.0e-9 maximum
    # elementwise replay variation; the shifted/cleanup return was 1.9e-9.
    # Retain a tight float32 replay tolerance, not a claim of bitwise gradients.
    np.testing.assert_allclose(restored.gradient_theta, baseline.gradient_theta,
                               rtol=2e-5, atol=1e-8)
    assert np.linalg.norm(restored.gradient_theta - baseline.gradient_theta) < 1e-7
    hard_neutral = evaluator.evaluate_candidate(theta, "t3.4b:hard-neutral", "hard")
    evaluator.set_evaluation_loss_shift(0.4)
    hard_shifted = evaluator.evaluate_candidate(theta, "t3.4b:hard-shifted", "hard")
    assert hard_shifted.objective == hard_neutral.objective
    shifted_path = evaluator.sim.tool_pos.to_numpy().copy()
    np.testing.assert_array_equal(shifted_path, baseline_path)
    for name in ("gouge_fraction", "residual_fraction", "holder_overlap_fraction"):
        assert hard_shifted.components[name] == hard_neutral.components[name]
    assert M.from_merged_hard(hard_shifted, M.EvaluationContext(
        evaluator.task.grid, 20.0, 0.4, "algorithms.objectives.forward_hard/merged"
    )).raw_phi_hard == hard_neutral.objective
    shifted_hard_path = evaluator.sim.tool_pos.to_numpy().copy()
    np.testing.assert_array_equal(shifted_hard_path, baseline_path)
    epoch = M.EvaluationContext(evaluator.task.grid, 20.0, 0.0, "neutral-scalar-check")
    with pytest.raises(ValueError, match="neutral zero shift"):
        M.from_merged_soft(baseline, dataclasses.replace(epoch, loss_shift=0.25))
    with pytest.raises(ValueError, match="scalar subtraction"):
        M.from_merged_soft(baseline, epoch, applied_shift=0.25)
    for invalid in (float("nan"), float("inf"), 0.51, -0.51, True):
        with pytest.raises(ValueError):
            evaluator.set_evaluation_loss_shift(invalid)
    evaluator.set_evaluation_loss_shift(None)
    assert evaluator.evaluation_loss_shift is None
    assert evaluator.sim.loss_shift[None] == 0.0
    evaluator.clear_evaluation_k()


def test_cma_start_and_adam_entry_exit_are_shift_neutral(tmp_path):
    from algorithms import blackbox as B
    from algorithms import optimize as adam

    budget = OptimizationBudget(candidate_evaluations=4, population_size=2)
    ev = prototype()
    neutral_hash = B._setup_signature(ev, "smooth")
    ev.set_evaluation_loss_shift(0.4)
    shifted_hash = B._setup_signature(ev, "smooth")
    assert shifted_hash != neutral_hash
    ev.clear_evaluation_loss_shift()
    assert B._setup_signature(ev, "smooth") == neutral_hash

    ev.set_evaluation_loss_shift(0.4)
    continued = B.solve_cma_es(ev, budget, seed=17, objective_kind="smooth",
                               run_id="t3.4b:continued-environment")
    assert continued.best.valid
    assert ev.evaluation_loss_shift is None
    for entry in ev.ledger.snapshot()["entries"]:
        assert entry["loss_shift_voxels"] == 0.0

    fresh = prototype()
    fresh_run = B.solve_cma_es(fresh, budget, seed=17, objective_kind="smooth",
                               run_id="t3.4b:fresh-reference")
    assert fresh_run.best.valid
    # Determinism of neutral baselines, not an optimizer benefit assertion.
    np.testing.assert_allclose(continued.best_theta, fresh_run.best_theta,
                               rtol=0.0, atol=2e-6)
    assert continued.best.objective == pytest.approx(fresh_run.best.objective, abs=2e-6)

    adam_budget = OptimizationBudget(candidate_evaluations=2)
    adam_config = adam.AdamConfig(mode="fixed", learning_rate=0.005,
                                  k_start=20.0, k_end=20.0)
    ev = prototype()
    ev.set_evaluation_loss_shift(0.4)
    continued_adam = adam.solve_adam(ev, adam_budget, adam_config, seed=17,
                                     run_id="t3.4b:continued-adam")
    assert ev.evaluation_loss_shift is None
    assert ev.ledger.snapshot()["entries"][0]["loss_shift_voxels"] == 0.0

    fresh_adam_evaluator = prototype()
    fresh_adam = adam.solve_adam(fresh_adam_evaluator, adam_budget, adam_config,
                                 seed=17, run_id="t3.4b:fresh-adam")
    np.testing.assert_allclose(continued_adam.current_theta, fresh_adam.current_theta,
                               rtol=0.0, atol=2e-6)
    assert len(continued_adam.history) == len(fresh_adam.history)
    for left, right in zip(continued_adam.history, fresh_adam.history):
        assert left["k"] == right["k"]
        assert left["soft_objective"] == pytest.approx(right["soft_objective"], abs=2e-6)

    # Neutral-checkpoint restoration accepts a matching clean setup, while a
    # stale positive ambient field shift cannot alter restoration identity.
    checkpoint_dir = tmp_path / "adam-checkpoints"
    checkpoint_evaluator = prototype()
    adam.solve_adam(checkpoint_evaluator, adam_budget, adam_config, seed=17,
                    run_id="t3.4b:checkpoint", checkpoint_dir=checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("ckpt-*.json"))
    assert checkpoints
    checkpoint = checkpoints[-1]
    restored_evaluator = prototype()
    restored_evaluator.set_evaluation_loss_shift(0.4)
    with pytest.raises(B.BlackBoxError, match="identity differs"):
        adam.resume_adam(checkpoint, restored_evaluator)
    restored_evaluator.clear_evaluation_loss_shift()
    resumed = adam.resume_adam(
        checkpoint, restored_evaluator,
        budget=OptimizationBudget(candidate_evaluations=3),
    )
    assert resumed is not None
    assert restored_evaluator.evaluation_loss_shift is None
    resumed_evaluator_current = B._setup_signature(restored_evaluator, "smooth")
    assert resumed_evaluator_current == neutral_hash