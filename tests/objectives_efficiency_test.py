"""Objective efficiency constraints, exact cycle costs and differentiable surrogates."""
import dataclasses
import math
from pathlib import Path
import runpy

import numpy as np
import pytest

from algorithms import blackbox as B
from algorithms import objectives as O
from benchmarks import contracts as C


_FIXTURES = runpy.run_path(Path(__file__).with_name("objectives_test.py"))
_prototype = _FIXTURES["_prototype"]


def _efficiency_spec(reference, *, time_weight=0.4, air_time_weight=0.2):
    """Isolate declared efficiency terms; geometry/safety weights are opt-in here."""
    return O.ObjectiveSpec(
        gouge_weight=0.0, residual_weight=0.0, holder_weight=0.0,
        time_weight=time_weight, air_time_weight=air_time_weight,
        reference_cycle_time_s=reference,
    )


def _duration(motion):
    return math.fsum(segment.duration_s for segment in motion.segments)


def test_time_air_weights_configurable_and_change_spec_hash():
    evaluator = _prototype(precision="f64")
    reference = evaluator.reference_cycle_time_s
    assert reference == pytest.approx(_duration(evaluator.motion))
    base = evaluator.spec
    weighted = _efficiency_spec(reference, time_weight=0.7, air_time_weight=0.3)
    assert base.time_weight == base.air_time_weight == 0.0
    assert base.legacy_weights == O.LegacyTrajectoryWeights()
    assert C.content_hash(dataclasses.asdict(base)) != C.content_hash(dataclasses.asdict(weighted))
    first_hash = B._setup_signature(evaluator, "hard")
    evaluator.spec = weighted
    assert B._setup_signature(evaluator, "hard") != first_hash
    evaluator._configure()
    assert evaluator.sim.w_time[None] == pytest.approx(0.7 / reference)
    assert evaluator.sim.w_air_time[None] == pytest.approx(0.3 / reference)
    assert evaluator.sim.w_air[None] == 0.0
    with pytest.raises(ValueError, match="reference_cycle_time_s"):
        O.ObjectiveSpec(reference_cycle_time_s=0.0)
    with pytest.raises(ValueError, match="time_normalization"):
        O.ObjectiveSpec(time_normalization="unrecognised")
    with pytest.raises(ValueError, match="time_weight"):
        O.ObjectiveSpec(time_weight=-1.0)
    with pytest.raises(ValueError, match="air_time_weight"):
        O.ObjectiveSpec(air_time_weight=-0.5)
    with pytest.raises(ValueError, match="post-cut stock contract"):
        O.ObjectiveSpec(legacy_weights=O.LegacyTrajectoryWeights(w_air=1.0))


def test_explicit_zero_efficiency_weights_preserve_scalar_objective_and_gradients():
    evaluator = _prototype(precision="f64")
    theta = np.array([0.001, -0.002, 0.001, 0.0005, 0.0015, -0.001])
    value_before = evaluator.evaluate_candidate(theta, "baseline-smooth")
    baseline_gradient = evaluator.evaluate_candidate_with_gradient(theta, "baseline-grad")
    hard_before = evaluator.evaluate_candidate(theta, "baseline-hard", objective_kind="hard")
    evaluator.spec = O.ObjectiveSpec(reference_cycle_time_s=evaluator.reference_cycle_time_s)
    value_after = evaluator.evaluate_candidate(theta, "zeros-smooth")
    actual_gradient = evaluator.evaluate_candidate_with_gradient(theta, "zeros-grad")
    hard_after = evaluator.evaluate_candidate(theta, "zeros-hard", objective_kind="hard")
    assert value_after.objective == pytest.approx(value_before.objective, abs=1e-12)
    assert hard_after.objective == pytest.approx(hard_before.objective, abs=1e-12)
    np.testing.assert_allclose(actual_gradient.gradient_theta, baseline_gradient.gradient_theta,
                               rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_gradient.result.objective, baseline_gradient.result.objective,
                               rtol=0.0, atol=1e-12)
    assert evaluator.sim.w_time[None] == evaluator.sim.w_air_time[None] == 0.0
    for descriptor in dataclasses.fields(evaluator.spec.legacy_weights):
        assert getattr(evaluator.sim, descriptor.name)[None] == 0.0


def test_hard_exact_cycle_efficiency_components_and_feasibility():
    evaluator = _prototype(precision="f64", spec=_efficiency_spec(1.0))
    theta = np.zeros(6)
    result = evaluator.evaluate_candidate(theta, "exact-time", objective_kind="hard")
    assert result.valid
    expected_time = _duration(evaluator.candidate_motion(theta))
    assert result.components["cycle_time_s"] == pytest.approx(expected_time)
    assert result.components["reference_cycle_time_s"] == pytest.approx(evaluator.reference_cycle_time_s)
    assert result.components["efficiency"] == result.objective
    for name in ("cycle_time_s", "air_time_s", "air_fraction", "feasible",
                 "gouge_fraction", "residual_fraction", "holder_overlap_fraction"):
        assert name in result.components and math.isfinite(result.components[name])
    stock = evaluator.sim.stock.to_numpy()
    initial, target = stock[0] < 0, evaluator.sim.target.to_numpy() < 0
    v = evaluator.task.grid.voxel_mm ** 3
    gouge_limit = evaluator.task.tolerances.gouge_volume_mm3 / (np.count_nonzero(initial & target) * v)
    residual_limit = evaluator.task.tolerances.completion_residual_mm3 / (
        np.count_nonzero(initial & ~target) * v)
    assert result.components["gouge_fraction_limit"] == pytest.approx(gouge_limit)
    assert result.components["residual_fraction_limit"] == pytest.approx(residual_limit)
    expected_feasible = float(
        result.components["holder_overlap_fraction"] == 0
        and result.components["gouge_fraction"] <= gouge_limit
        and result.components["residual_fraction"] <= residual_limit)
    assert result.components["feasible"] == expected_feasible
    engagement = evaluator.sim.seg_engage.to_numpy()
    durations = np.asarray([segment.duration_s for segment in evaluator.candidate_motion(theta).segments])
    expected_air = math.fsum(durations[engagement <= 0.5 / np.prod(initial.shape)].tolist())
    assert result.components["air_time_s"] == pytest.approx(expected_air)
    assert result.components["air_fraction"] == pytest.approx(expected_air / expected_time)


def test_slower_program_with_identical_hard_geometry_scores_worse():
    reference_time = 0.4
    spec = O.ObjectiveSpec(time_weight=0.25, air_time_weight=0.0,
                           reference_cycle_time_s=reference_time)
    evaluator = _prototype(precision="f64", spec=spec)
    theta = np.zeros(6)
    before = evaluator.evaluate_candidate(theta, "normal-rate", objective_kind="hard")
    before_stock = evaluator.sim.stock.to_numpy()[evaluator.parameterization.num_steps].copy()
    segment = evaluator.motion.segments[0]
    reduced_feed = segment.feed_mm_s / 2.0
    slower = dataclasses.replace(
        segment, feed_mm_s=reduced_feed,
        duration_s=segment.chord_length_mm / reduced_feed)
    changed_motion = C.MotionPlan(
        frame=evaluator.motion.frame, initial_state=evaluator.motion.initial_state,
        segments=(slower, *evaluator.motion.segments[1:]), timing_rule=evaluator.motion.timing_rule,
        provenance="geometry-invariant slower toolpath")
    changed_motion.validate_for_task(evaluator.task)
    evaluator.motion = changed_motion
    after = evaluator.evaluate_candidate(theta, "reduced-feed", objective_kind="hard")
    assert after.components["cycle_time_s"] > before.components["cycle_time_s"]
    assert after.components["efficiency"] > before.components["efficiency"]
    for name in ("gouge_fraction", "residual_fraction", "holder_overlap_fraction", "feasible"):
        assert after.components[name] == before.components[name]
    np.testing.assert_array_equal(after.diagnostics["final_stock"], before.diagnostics["final_stock"])
    assert math.isclose(
        after.objective,
        before.objective + spec.time_weight
        * (after.components["cycle_time_s"] - before.components["cycle_time_s"]) / reference_time,
        rel_tol=0.0, abs_tol=1e-12)


def test_native_time_is_a_documented_surrogate_not_exact_motion_duration():
    """The differentiable fixture has its native regime/dt terms, not exact-stop parity."""
    evaluator = _prototype(precision="f64", spec=_efficiency_spec(
        1.0, time_weight=1.0, air_time_weight=0.0))
    theta = np.zeros(6)
    smooth = evaluator.evaluate_candidate(theta, "native-time", objective_kind="smooth")
    assert smooth.valid
    assert smooth.components["time_source_is_simulator_surrogate"] == 1.0
    native_mean_seconds = float(evaluator.sim.seg_time.to_numpy().mean())
    assert smooth.components["cycle_time_s"] == pytest.approx(native_mean_seconds)
    assert smooth.objective == pytest.approx(native_mean_seconds)
    exact_seconds = _duration(evaluator.candidate_motion(theta))
    assert native_mean_seconds > 0.0 and exact_seconds > 0.0
    assert native_mean_seconds != pytest.approx(exact_seconds, rel=1e-6)


@pytest.mark.parametrize("time_weight,air_time_weight", [(0.35, 0.0), (0.0, 0.75)])
def test_efficiency_gradient_matches_finite_difference(time_weight, air_time_weight):
    """Each efficiency-only channel must have a meaningful f64 AD/FD signal.

    These two short above-stock moves stay below the native dt*rapid-speed cap.
    The default in-material fixture instead saturates both native seg_time terms,
    whose correctly-zero gradient would not establish gradient correctness.
    """
    evaluator = _prototype(
        precision="f64", initial_position_mm=(0.0, 0.0, 10.0),
        residual_mm=((0.0, 0.0, -1.0), (1.0, 1.0, 1.0)))
    evaluator.spec = _efficiency_spec(
        evaluator.reference_cycle_time_s,
        time_weight=time_weight, air_time_weight=air_time_weight)
    evaluator._configure()
    theta = np.array([0.0004, -0.0003, 0.0006, -0.0003, 0.0005, -0.0002])
    taped = evaluator.evaluate_candidate_with_gradient(theta, "efficiency-gradient")
    assert taped.result.valid
    assert taped.gradient_theta.shape == theta.shape
    assert np.isfinite(taped.gradient_theta).all()
    axis = int(np.argmax(np.abs(taped.gradient_theta)))
    direction = np.eye(theta.size)[axis]
    ad = float(taped.gradient_theta @ direction)
    assert abs(ad) > 1e-4  # A matching pair of near-zero gradients would prove nothing.
    epsilon = 1e-5
    plus = evaluator.evaluate_candidate(theta + epsilon * direction, "efficiency-fd-plus")
    minus = evaluator.evaluate_candidate(theta - epsilon * direction, "efficiency-fd-minus")
    assert plus.valid and minus.valid
    fd = (plus.objective - minus.objective) / (2 * epsilon)
    assert np.isfinite(fd) and abs(fd) > 1e-4
    assert ad == pytest.approx(fd, rel=1e-5, abs=1e-8)
    evaluator = _prototype(precision="f64", spec=_efficiency_spec(1.0))
    rejected = evaluator.evaluate_candidate(np.full(6, np.nan), "unknown", objective_kind="hard")
    assert not rejected.valid
    assert rejected.objective is None
    assert "feasible" not in rejected.components
    assert "cycle_time_s" not in rejected.components
    assert "efficiency" not in rejected.components
