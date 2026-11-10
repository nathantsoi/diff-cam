"""Unit tests for T3.3 estimator mechanics and its explicit cost accounting.

Analytic functions use a real ComputeLedger, not a mock optimizer or simulator.
They validate formula/accounting/scaling without masquerading as end-to-end
physical evidence. The separate opt-in test runs real Taichi pocket execution.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from algorithms import gradient_mechanics as gm
from algorithms.parameterization import DeltaResidual
from algorithms.run_artifacts import ComputeLedger
from benchmarks import contracts as C


class RecordingObjective:
    """A real ledgered scorer for known functions, deliberately not a simulator."""
    def __init__(self, function, *, bounds=None, kind="hard"):
        self.function = function
        self.bounds = bounds
        self.calls = []
        self.kind = kind
        self.ledger = ComputeLedger()

    def evaluate_candidate(self, theta, candidate_id, objective_kind="smooth"):
        theta = np.asarray(theta, dtype=np.float64)
        self.ledger.start_candidate(candidate_id, objective_kind)
        forward = 0
        valid = False
        try:
            if theta.ndim != 1 or not np.isfinite(theta).all():
                raise ValueError("known scorer expects a finite vector")
            if self.bounds is not None and not np.all((self.bounds[0] <= theta) & (theta <= self.bounds[1])):
                raise ValueError("known scorer declared theta out of bounds")
            if objective_kind != self.kind:
                raise ValueError(f"known scorer expected {self.kind}")
            objective = float(self.function(theta))
            if not np.isfinite(objective):
                raise FloatingPointError("known scorer non-finite value")
            forward = 1
            valid = True
            return C.CandidateResult(candidate_id=candidate_id, theta_hash=C.content_hash(theta.tolist()),
                                     objective_kind=objective_kind, objective=objective, valid=True)
        except (ValueError, FloatingPointError) as error:
            return C.CandidateResult(
                candidate_id=candidate_id, theta_hash=C.content_hash(theta.tolist()),
                objective_kind=objective_kind, objective=None, valid=False,
                invalid_reason=str(error), coverage=C.Coverage.UNKNOWN)
        finally:
            self.ledger.finish_candidate(valid=valid, forward_calls=forward, backward_calls=0,
                                         segment_steps=1 if forward else 0, voxel_steps=1 if forward else 0)


def quadratic(function_matrix, linear=None):
    a = np.asarray(function_matrix, dtype=np.float64)
    b = np.zeros(a.shape[0]) if linear is None else np.asarray(linear, dtype=np.float64)
    g = 0.5 * (a + a.T)
    return lambda x: 0.5 * x @ g @ x + b @ x, lambda x: g @ x + b


def test_antithetic_known_quadratic_and_exact_two_n_ledger():
    a = np.array([[2.0, -0.4, 0.0], [-0.4, 1.5, 0.3], [0.0, 0.3, 0.6]])
    f, grad = quadratic(a, np.array([0.25, -0.15, 0.8]))
    objective = RecordingObjective(f)
    theta = np.array([0.15, -0.4, 0.6])
    estimate = gm.antithetic_estimate(objective, theta, sigma=np.array([0.1, 0.25, 0.7]),
                                      pairs=3, seed=1234, ledger=objective.ledger)
    # With a vector sigma, noise also includes cross-coordinate terms:
    # v_i =[(grad(theta) dot (sigma*z_i))] z_i/sigma. The expectation is the
    # Gaussian-smoothed gradient, but a finite-pair estimate is NOT exact.
    z = np.random.default_rng(np.random.SeedSequence(1234)).standard_normal((3, 3))
    sigma = np.array([0.1, 0.25, 0.7])
    expected = np.mean([((grad(theta) * sigma) @ sample) * sample / sigma for sample in z], axis=0)
    np.testing.assert_allclose(estimate.gradient, expected, atol=1e-12, rtol=1e-12)
    assert np.linalg.norm(estimate.gradient - grad(theta)) > 0.1
    assert estimate.status == "ok"
    assert estimate.ledger_delta["candidate_evaluations"] == 6
    assert estimate.ledger_delta["forward_calls"] == 6
    assert estimate.ledger_delta["backward_calls"] == 0
    assert estimate.ledger_delta["validation_calls"] == 0
    assert sum(entry["warmup"] for entry in objective.ledger.entries) == 1
    z = np.random.default_rng(np.random.SeedSequence(1234)).standard_normal((3, 3))
    sigma = np.array([0.1, 0.25, 0.7])
    v = np.array([(f(theta + sigma * sample) - f(theta - sigma * sample)) * sample / (2 * sigma)
                  for sample in z])
    np.testing.assert_allclose(estimate.standard_error, v.std(axis=0, ddof=1) / np.sqrt(3))
    assert estimate.standard_error_rms > 0
    reloaded = gm.estimate_to_json(estimate)
    C.canonical_json(reloaded)
    assert reloaded["hard_candidate_evaluations_this_phase"] == 6


def test_one_pair_scalar_quadratic_identity_and_undefined_uncertainty():
    f, analytic_gradient = quadratic([[3.0]])
    objective = RecordingObjective(f)
    theta = np.array([0.4])
    estimate = gm.antithetic_estimate(objective, theta, 0.2, 1, seed=144, ledger=objective.ledger)
    z = np.random.default_rng(np.random.SeedSequence(144)).standard_normal()
    expected = analytic_gradient(theta)[0] * z * z
    np.testing.assert_allclose(estimate.gradient, [expected])
    assert estimate.standard_error is None
    assert estimate.standard_error_rms is None
    assert estimate.null_reason == "one_pair_has_no_sample_standard_error"


def test_step_heaviside_composite_matches_gaussian_weak_gradient():
    def f(theta):
        x, y = theta
        # L=H(x) + x^2 + 1.5*y^2 + y; the discontinuity is not assigned
        # an ordinary classical derivative by this test.
        return float(x > 0) + x*x + 1.5*y*y + y

    theta = np.array([0.0, 0.3])
    objective = RecordingObjective(f)
    sigma = 0.07
    result = gm.estimate_with_reference(objective, theta, sigma, 256, 4096,
                                        seed=1729, ledger=objective.ledger)
    expected = 1.0 / (np.sqrt(2*np.pi) * sigma)
    primary, reference = result["primary"], result["reference"]
    assert primary.standard_error[0] > 0 and reference.standard_error[0] > 0
    assert abs(reference.gradient[0] - expected) < 4 * reference.standard_error[0]
    assert abs(primary.gradient[0] - expected) < 5 * primary.standard_error[0]
    assert abs(reference.gradient[1] - 1.9) < 4 * reference.standard_error[1]
    assert abs(primary.gradient[1] - 1.9) < 5 * primary.standard_error[1]
    assert result["ledger_delta"]["candidate_evaluations"] == 2 * (256 + 4096)
    assert result["ledger_delta"]["backward_calls"] == 0
    assert primary.noise_sha256 != reference.noise_sha256
    assert primary.seed["spawn_key"] != reference.seed["spawn_key"]
    assert result["comparison"]["joint_standard_error_rms"] > 0


def test_step_central_secant_at_kink_and_reproducible_streams():
    # For relu(x)=max(x,0), the central secant at its kink has slope 1/2.
    objective = RecordingObjective(lambda x: max(float(x[0]), 0.0), kind="smooth")
    records = gm.central_differences(objective, np.zeros(1), [1.0, 0.5, 0.1], ledger=objective.ledger)
    for record in records:
        assert record["status"] == "ok"
        np.testing.assert_allclose(record["gradient"], [0.5])
    first, reference = gm.split_streams(77)
    a = np.random.default_rng(first).standard_normal(4)
    b = np.random.default_rng(reference).standard_normal(4)
    repeated = np.random.default_rng(gm.split_streams(77)[0]).standard_normal(4)
    np.testing.assert_array_equal(a, repeated)
    assert not np.array_equal(a, b)


def test_direct_high_sample_quadratic_reference_and_uncertainty():
    a = np.array([[2.0, -0.4, 0.0], [-0.4, 1.5, 0.3], [0.0, 0.3, 0.6]])
    bias = np.array([0.25, -0.15, 0.8])
    f, analytic_gradient = quadratic(a, bias)
    objective = RecordingObjective(f)
    theta = np.array([0.15, -0.4, 0.6])
    estimate = gm.reference_antithetic(objective, theta, 0.3, 4096,
                                       seed=933, ledger=objective.ledger)
    assert np.all(np.abs(estimate.gradient - analytic_gradient(theta)) <= 4 * estimate.standard_error)
    assert estimate.ledger_delta["candidate_evaluations"] == 8192
    assert estimate.seed["spawn_key"] == (1,)
    assert estimate.status == "ok"
    assert len(objective.ledger.entries) == 8192


def test_invalid_pairs_are_not_resampled_or_scored():
    f, _ = quadratic([[1.0]])
    objective = RecordingObjective(f, bounds=np.array([-0.2, 0.2]))
    ledger = objective.ledger
    estimate = gm.antithetic_estimate(objective, np.zeros(1), 0.4, 4, seed=5, ledger=ledger)
    assert estimate.status == "invalid"
    assert estimate.gradient is None and estimate.standard_error is None
    assert estimate.invalid_pairs
    assert len(ledger.entries) == 8
    assert ledger.candidate_evaluations == 8
    assert ledger.forward_calls < 8
    assert any(not record["valid"] for record in ledger.entries)
    assert not any("base" in record["candidate_id"] for record in ledger.entries)


def test_epsilon_subset_scaling_and_exact_accounting():
    f, grad = quadratic([[2.0, 0.0], [0.0, 1.0]], [0.5, -0.5])
    objective = RecordingObjective(f, kind="smooth")
    theta = np.array([0.15, -0.25])
    records = gm.central_differences(objective, theta, [0.2, [0.2, 0.1]],
                                     indices=[1], ledger=objective.ledger)
    assert len(objective.ledger.entries) == 4
    for entry, epsilon in zip(records, (0.2, 0.1)):
        assert entry["center_objective"] is None
        assert entry["evaluated_indices"] == [1]
        assert math.isnan(entry["gradient"][0])
        assert entry["gradient"][1] == pytest.approx(grad(theta)[1])
        assert entry["ledger_delta"]["candidate_evaluations"] == 2
    objective_other = RecordingObjective(f, kind="smooth")
    raw = gm.central_differences(objective_other, theta * 0.7, [0.2 / 0.7])[0]
    np.testing.assert_allclose(raw["gradient"], grad(theta * 0.7), atol=1e-13)
    assert objective_other.ledger.candidate_evaluations == 4


def test_near_zero_gradient_is_unknown_not_cosine_one():
    zero = gm.gradient_alignment(np.zeros(2), np.zeros(2))
    assert zero["cosine"] is None and zero["status"] == "both_near_zero"
    one = gm.gradient_alignment([1e-14, 0], [1.0, 0])
    assert one["cosine"] is None and one["status"] == "one_near_zero"
    known = gm.gradient_alignment([1.0, 0.5], [1.0, 0.0])
    assert known["cosine"] == pytest.approx(2 / np.sqrt(5))
    objective = RecordingObjective(lambda theta: 0.0)
    result = gm.antithetic_estimate(objective, np.zeros(2), 0.1, 4, seed=11, ledger=objective.ledger)
    assert result.status == "ok"
    assert result.gradient is not None
    assert result.standard_error_rms == 0.0
    assert gm.gradient_alignment(result.gradient, np.zeros(2))["cosine"] is None


def test_typed_fixed_path_and_frame_tolerances():
    parameterization = DeltaResidual([[1.25, -2.0, 0.1], [0.75, 0.5, -0.4]],
                                    scale_mm=0.25, initial_position_mm=[1, 2, 3])
    theta = np.ones(parameterization.parameter_shape) * 0.1
    positions = parameterization.positions(theta)
    state = C.MachineState(position_mm=tuple(positions[0]), frame=C.Frame.MACHINE)
    segments = []
    for i, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
        segments.append(C.Segment(id=f"s{i}", kind=C.SegmentKind.FEED,
                                  start_mm=tuple(start), end_mm=tuple(end),
                                  duration_s=float(np.linalg.norm(end - start)) / 2.0,
                                  feed_mm_s=2.0))
    first = C.MotionPlan(frame=C.Frame.MACHINE, initial_state=state,
                         timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED, segments=tuple(segments))
    second_segments = [replace(segment, end_mm=tuple(np.asarray(segment.end_mm) + [0, 0.001, 0]))
                       if i == 0 else segment for i, segment in enumerate(segments)]
    # Altering one endpoint without adjusting all connected positions is rejected
    # by the typed plan, not silently fixed by a tolerating comparison.
    with pytest.raises(C.ContractError):
        C.MotionPlan(frame=C.Frame.MACHINE, initial_state=state,
                     timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED, segments=tuple(second_segments))
    report = gm.compare_execution_paths(first, first, [-10, -10, -10], [20, 20, 20])
    assert report.matches and report.max_machine_error_mm == 0.0
    assert report.max_machine_error_mm == 0.0
    assert report.max_normalized_error == 0.0

    def shifted_final(displacement, recompute_duration=False):
        end = np.asarray(segments[-1].end_mm) + displacement
        start = np.asarray(segments[-1].start_mm)
        last = replace(segments[-1], end_mm=tuple(end),
                       duration_s=(segments[-1].duration_s if not recompute_duration
                                   else float(np.linalg.norm(end - start)) / 2.0))
        return C.MotionPlan(frame=C.Frame.MACHINE, initial_state=state,
                            timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
                            segments=(segments[0], last))

    tolerated = gm.compare_execution_paths(first, shifted_final([0, 0, 1e-8]),
                                           [-10, -10, -10], [20, 20, 20])
    assert tolerated.matches and tolerated.max_machine_error_mm == pytest.approx(1e-8)
    assert tolerated.typed_motion_hash != first.motion_hash
    with pytest.raises(AssertionError, match="timing changed"):
        gm.compare_execution_paths(
            first, shifted_final([0, 0, 0.02], recompute_duration=True),
            [-10, -10, -10], [20, 20, 20])
    with pytest.raises(ValueError):
        gm.compare_execution_paths(first, first, [-10, -10, -10], [20, 20, 0])


@pytest.mark.parametrize("kwargs", [
    {"sigma": 0, "pairs": 2, "seed": 1},
    {"sigma": -0.1, "pairs": 2, "seed": 1},
    {"sigma": np.nan, "pairs": 2, "seed": 1},
    {"sigma": [0.1, 0.2], "pairs": 2, "seed": 1},
    {"sigma": 0.1, "pairs": 0, "seed": 1},
    {"sigma": 0.1, "pairs": 1.5, "seed": 1},
    {"sigma": 0.1, "pairs": 1, "seed": -1},
    {"sigma": 0.1, "pairs": 1, "seed": None},
    {"sigma": True, "pairs": 1, "seed": 1},
])
def test_parameter_coordinate_and_budget_validation(kwargs):
    f, _ = quadratic([[1.0]])
    with pytest.raises(ValueError):
        gm.antithetic_estimate(RecordingObjective(f), np.array([0.1]), ledger=ComputeLedger(), **kwargs)


def test_invalid_fd_parameters_do_not_broadcast_or_change_kind():
    f, _ = quadratic([[1.0, 0.0], [0.0, 1.0]])
    objective = RecordingObjective(f, kind="smooth")
    with pytest.raises(ValueError):
        gm.central_differences(objective, np.zeros(2), [0], indices=[])
    with pytest.raises(ValueError):
        gm.central_differences(objective, np.zeros(2), [0.1], indices=[0, 0])
    with pytest.raises(ValueError):
        gm.central_differences(objective, np.zeros(2), [0.1], indices=[2])
    with pytest.raises(ValueError):
        gm.central_differences(objective, np.zeros(2), [[0.1, 0.2]], indices=[])
    with pytest.raises(TypeError):
        gm.scalar_result(lambda theta, ident, kind: None)
    before_invalid_eps = len(objective.ledger.entries)
    with pytest.raises(ValueError):
        gm.central_differences(objective, np.zeros(2), [0.1, -0.01])
    assert len(objective.ledger.entries) == before_invalid_eps
    def mislabel(theta, candidate_id, objective_kind):
        return C.CandidateResult(
            candidate_id=candidate_id, theta_hash=C.content_hash(theta.tolist()),
            objective_kind="smooth", objective=0.0, valid=True)
    with pytest.raises(TypeError, match="switched"):
        gm.antithetic_estimate(mislabel, np.zeros(2), 0.1, 1, seed=1)


def test_delta_ledger_is_numeric_and_mono_budget():
    before = gm._snapshot(ComputeLedger())
    ledger = ComputeLedger()
    ledger.start_candidate("x", "hard")
    ledger.finish_candidate(valid=True, forward_calls=1, backward_calls=0,
                            segment_steps=1, voxel_steps=1)
    after = ledger.snapshot()
    delta = gm.delta_snapshot(before, after)
    assert delta["candidate_evaluations"] == 1
    assert delta["forward_calls"] == 1 and delta["backward_calls"] == 0
    assert "entries" not in delta and "memory_scope" not in delta


def test_terminal_geometry_numpy_derivative_matches_central_differences():
    from scripts.run_gradient_mechanics import terminal_geometry_loss_numpy

    stock = np.array([-8.0, -3.0, -1.0, 0.0, 0.5, 1.0, 4.0, 9.0])
    target = np.array([0.0, 1.0, -1.0, 0.5, -0.6, 1.5, -2.0, 2.0])
    args = {"gouge_weight": 2.0, "residual_weight": 0.75, "loss_shift": 0.125}
    exact = terminal_geometry_loss_numpy(stock, target, **args)
    eps = 1e-5
    for index in range(stock.size):
        plus, minus = stock.copy(), stock.copy()
        plus[index] += eps
        minus[index] -= eps
        central = (terminal_geometry_loss_numpy(plus, target, **args)["loss"]
                   - terminal_geometry_loss_numpy(minus, target, **args)["loss"]) / (2 * eps)
        assert exact["gradient"][index] == pytest.approx(central, rel=2e-8, abs=1e-10)
    split = terminal_geometry_loss_numpy(stock, target, gouge_weight=1, residual_weight=0)
    assert split["loss"] == pytest.approx(split["gouge"])
    assert split["residual"] == 0


def test_terminal_geometry_numpy_handles_clamps_and_bad_inputs():
    from scripts.run_gradient_mechanics import terminal_geometry_loss_numpy

    result = terminal_geometry_loss_numpy([-60.0, 60.0, 0.0], [-1.0, 1.0, 2.0],
                                          gouge_weight=2, residual_weight=1)
    assert result["gradient"][:2].tolist() == [0.0, 0.0]
    assert not result["interior_to_clamp"][:2].any()
    with pytest.raises(ValueError, match="finite"):
        terminal_geometry_loss_numpy([np.nan], [0.0], gouge_weight=1, residual_weight=1)
    with pytest.raises(ValueError, match="nonnegative"):
        terminal_geometry_loss_numpy([0.0], [0.0], gouge_weight=-1, residual_weight=1)


@pytest.mark.parametrize(
    ("precision", "atol", "rtol"),
    [("f32", 1.6e-5, 1e-5), ("f64", 2e-12, 0.0)],
    ids=("f32", "f64"),
)
def test_ad_same_theta_after_other_theta_matches_fresh_process(tmp_path, precision, atol, rtol):
    """Real AD evaluation order stays neutral in each declared storage precision.

    Both contexts create the public evaluator in isolated child processes.
    The 2 mm coarse grid is independently sampled by the shared factory. This
    compares order invariance, not AD with FD or the larger f32 reverse bias.
    The f64 comparison is explicitly tighter than the f32 atomic-noise bound.
    """
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from scripts.run_gradient_mechanics import ad_order_regression
ad_order_regression(Path(sys.argv[1]), voxel_mm=2.0, ordered=sys.argv[2] == 'ordered', precision=sys.argv[4])
"""
    environment = dict(os.environ)
    environment["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path / "ti-cache")
    environment["UV_CACHE_DIR"] = "/tmp/diffcam-uv-diffcam-worker-3"

    def run_child(name, ordered):
        output = tmp_path / name
        run = subprocess.run(
            [sys.executable, "-I", "-c", code, str(output), "ordered" if ordered else "fresh",
             str(Path(__file__).resolve().parents[1]), precision],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=environment, capture_output=True, text=True, timeout=180,
        )
        assert run.returncode == 0, f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}"
        import json
        return json.loads((output / "ad_order.json").read_text(encoding="utf-8"))

    fresh = run_child(f"fresh-{precision}", False)
    ordered = run_child(f"ordered-{precision}", True)
    expected_dtype = {"f32": "<f4", "f64": "<f8"}[precision]
    for record in (fresh, ordered):
        assert record["simulator_precision"] == precision
        for field in ("loss_dtype", "stock_dtype", "stock_gradient_dtype",
                      "tool_delta_dtype", "tool_delta_gradient_dtype"):
            assert record[field] == expected_dtype
        assert record["input_delta_dispatch"].startswith("existing CandidateEvaluator")
    assert fresh["ledger"]["candidate_evaluations"] == 1
    assert fresh["ledger"]["backward_calls"] == 1
    assert ordered["ledger"]["candidate_evaluations"] == 4
    assert ordered["ledger"]["backward_calls"] == 3
    assert ordered["candidate_order"] == [
        "order-initial", "order-repeat-same", "order-different-theta",
        "order-after-different",
    ]
    assert ordered["different_theta_index"] == 35
    assert ordered["different_theta_delta"] == 0.025
    assert ordered["grid"]["shape"] == [30, 20, 6]
    assert ordered["grid"]["independently_sampled"] is True
    assert ordered["nonzero_gradient"] and fresh["nonzero_gradient"]
    reference = np.asarray(fresh["records"]["initial"]["gradient"])
    assert reference.shape == (36,)
    assert (ordered["records"]["initial"]["theta_hash"]
            == fresh["records"]["initial"]["theta_hash"])
    # Preserve the already measured f32 tolerance. For f64, a leak substantially
    # below f32 noise cannot hide: atol=2e-12, rtol=0. This makes no claim of
    # mathematical exactness of the underlying trilinear/CSG reverse operation.
    for key in ("initial", "repeat_same", "after_other"):
        actual = np.asarray(ordered["records"][key]["gradient"])
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
        assert ordered["records"][key]["actual_path"]["same_within_tolerance"]


@pytest.mark.skipif(os.environ.get("DIFFCAM_REAL_PILOT_GRADIENT") != "1",
                    reason="real Taichi pocket smoke is opt-in with DIFFCAM_REAL_PILOT_GRADIENT=1")
def test_real_taichi_reduced_pilot_smoke(tmp_path):
    """Separate real CPU backend evidence, explicitly not the full-grid cost run.

    A child process avoids invalidating another test's Taichi fields. 0.5 mm is
    an explicitly different coarse sampling of the same analytic pilot setup,
    not an interpolation of the CLI's full 0.25 mm .npy files.
    """
    output = tmp_path / "real-pilot-cost"
    code = """
import sys
from pathlib import Path
from scripts.run_gradient_mechanics import CostConfig, run_pilot
report = run_pilot(Path(sys.argv[1]), CostConfig(
    pairs=1, reference_pairs=2, sigma_theta=0.05,
    epsilon_theta=(0.025,), fd_indices=(0, 1, 2), voxel_mm=0.5))
print(report["status"])
"""
    environment = dict(os.environ)
    environment["TI_OFFLINE_CACHE_FILE_PATH"] = "/tmp/diffcam-ti-diffcam-worker-3-gradient"
    environment["UV_CACHE_DIR"] = "/tmp/diffcam-uv-diffcam-worker-3"
    run = subprocess.run(
        [sys.executable, "-c", code, str(output)],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment, capture_output=True, text=True, timeout=600,
    )
    assert run.returncode == 0, f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    import json
    path = output / "gradient_mechanics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "completed_kind_checked_cost_probe"
    assert report["configuration"]["voxel_mm"] == 0.5
    assert report["configuration"]["fd_full_gradient_claim"] is False
    assert report["task_and_geometry"]["grid"]["shape"] == [120, 80, 24]
    assert report["task_and_geometry"]["target_shape"] == "sampled"
    assert report["configuration"]["full_solver_study"] is False
    assert report["hard_base"]["same_theta_as_smooth_ad"] is True
    assert report["hard_base"]["actual_path_match"]["same_within_tolerance"] is True
    assert report["smooth_ad_fd"]["fixed_theta_execution"]["ad_vs_forward_only_same_theta"] is True
    assert report["smooth_ad_fd"]["fixed_theta_execution"]["all_actual_paths_match_their_declared_theta"] is True
    counters = report["cost"]["counters_including_explicit_hard_base"]
    assert counters["candidate_evaluations"] == 15  # 7 hard + 1 AD + 1 replay + 6 FD
    assert counters["forward_calls"] == 15
    assert counters["backward_calls"] == 1
    assert counters["validation_calls"] == 0
    assert report["randomized_hard"]["primary"]["null_reason"] == "one_pair_has_no_sample_standard_error"
    assert len(report["smooth_ad_fd"]["comparisons"]) == 1
    comparison = report["smooth_ad_fd"]["comparisons"][0]
    assert comparison["evaluated_indices"] == [0, 1, 2]
    assert len(comparison["path_checks"]) == 6
    assert all(path["matches_declared_path"] for path in comparison["path_checks"])
    assert all(path["theta_hash"] != report["theta"]["hash"] for path in comparison["path_checks"])
    alignment = comparison["gradient_alignment"]
    if alignment["status"] == "ok":
        assert -1.0 <= alignment["cosine"] <= 1.0
    else:
        assert alignment["status"] in ("both_near_zero", "one_near_zero")
        assert alignment["cosine"] is None
    for side in ("target", "stock"):
        import hashlib
        payload = output / f"{side}.npy"
        array = np.load(payload, allow_pickle=False)
        assert array.shape == (120, 80, 24)
        assert np.isfinite(array).all()
        assert report["task_and_geometry"][f"{side}_sha256"] == "sha256:" + hashlib.sha256(payload.read_bytes()).hexdigest()


def _declared_fixed_study_document():
    """Small explicit study declaration, not a frozen protocol or optimizer sweep."""
    def path(identifier, kind, **values):
        return {"id": identifier, "kind": kind, **values}
    return {
        "schema": "diffcam-fixed-path-gradient-study/1", "study_only": True,
        "task_id": "dev-pocket-island-000-v2", "voxel_mm": 0.25, "precision": "f64",
        "cpu_max_num_threads": 8, "k_values": [20.0, 40.0, 80.0, 160.0],
        "paths": [
            path("cam", "cam"),
            path("cam-perturb-17", "cam_perturbed", scale_fraction=0.001, seed=17,
                 reference_kind="cam_delta_norm", distribution="gaussian", exploratory=True),
            path("cam-perturb-29", "cam_perturbed", scale_fraction=0.002, seed=29,
                 reference_kind="cam_delta_norm", distribution="gaussian", exploratory=True),
            path("delta-line", "delta", scale_fraction=0.0005, seed=11,
                 reference_kind="stock_xy_extent", distribution="gaussian", exploratory=True),
        ],
        "gouge_weight": 2.0, "residual_weight": 1.0, "holder_weight": 50.0, "holder_margin_mm": 0.0,
        "time_weight": 0.15, "air_time_weight": 0.025,
        "efficiency_path_ids": ["cam"], "zero_weight_path_id": "cam", "study_only_efficiency": True,
        "epsilon_theta": [0.025, 0.00625], "fd_indices": [0, 1, 2, 35], "sigma_theta": 0.05,
        "pairs": 16, "reference_pairs": 64, "hard_root_seed": 2026, "path_root_seed": 101,
        "scale_sweep": {"reference_kind": "stock_xy_extent",
                        "fractions": [0.0005, 0.0025, 0.01, 0.05, 0.1],
                        "seeds": list(range(30)), "root_seed": 303},
        "max_wall_time_s": 7200.0,
    }


def test_fixed_study_config_and_minimal_case_partition():
    from scripts import run_gradient_mechanics as R
    config = R.study_config_from_json(_declared_fixed_study_document())
    cases = R._fixed_cases(config)
    assert config.pairs == 16 and config.reference_pairs == 64
    assert config.k_values == (20.0, 40.0, 80.0, 160.0)
    assert len(cases) == 18
    regular = [case for case in cases if case["objective_mode"] == "geometry"]
    assert len(regular) == 16
    assert [(case["path_id"], case["k"]) for case in regular[:4]] == [
        ("cam", 20.0), ("cam", 40.0), ("cam", 80.0), ("cam", 160.0)]
    supplemental = [case for case in cases if case["objective_mode"] != "geometry"]
    assert [(case["objective_mode"], case["k"]) for case in supplemental] == [
        ("zero_weights", 40.0), ("study_efficiency", 40.0)]
    expected = 2 + 2 * len(config.epsilon_theta) * len(config.fd_indices) + 2 * 16 + 2 * 64
    assert expected == 178
    zero = R._objective_spec(config, supplemental[0], 1.5)
    assert zero.k == 40.0
    assert zero.gouge_weight == zero.residual_weight == zero.holder_weight == 0.0
    assert zero.time_weight == zero.air_time_weight == 0.0
    efficient = R._objective_spec(config, supplemental[1], 231.5)
    assert efficient.time_weight == 0.15 and efficient.air_time_weight == 0.025
    for field, value in (("k_values", [10.0, 20.0, 40.0, 80.0, 160.0]),
                         ("pairs", 8), ("cpu_max_num_threads", 9),
                         ("voxel_mm", 0.5), ("precision", "f32")):
        document = _declared_fixed_study_document()
        document[field] = value
        with pytest.raises(ValueError):
            R.study_config_from_json(document)


def test_fixed_study_null_relative_errors_and_dtype_evidence():
    from scripts import run_gradient_mechanics as R
    first, second = np.array([0.0, 1.0, 0.0, 4.0]), np.array([0.0, 3.0, 2.0, -4.0])
    rows = R._coordinate_errors(first, second, [0, 1, 2, 3])
    assert [row["relative_error"] for row in rows] == [None, 2.0, None, 2.0]
    assert [row["absolute_error"] for row in rows] == [0.0, 2.0, 2.0, 8.0]
    assert all(row["epsilon_abs"] == 1e-12 for row in rows)
    with pytest.raises(ValueError):
        R._coordinate_errors(np.array([np.nan]), np.array([0.0]), [0])
    field = SimpleNamespace(dtype=np.dtype("<f8"), grad=SimpleNamespace(dtype=np.dtype("<f8")))
    sim = SimpleNamespace(loss=field, stock=field, tool_delta=field)
    dtypes = R._dtype_information(sim)
    assert len(dtypes) == 6 and all(row["is_float64"] for row in dtypes.values())
    assert all(row["dtype"] == "<f8" for row in dtypes.values())


def test_scale_sweep_preserves_all_declared_indices_and_physical_failures():
    from scripts import run_gradient_mechanics as R
    from types import SimpleNamespace
    from benchmarks.pilot_task import build_pilot_task
    config = R.study_config_from_json(_declared_fixed_study_document())
    calls = []

    def controlled(declaration, seed):
        fraction = declaration["random_init"]["scale"]["fraction"]
        calls.append((fraction, seed))
        valid = not (fraction >= 0.05 and seed % 7 == 0)
        sigma = fraction * 72.11102550927978
        derivation = {
            "scale": {"scale_mm": sigma},
            "deviations": {"count": 74, "max_deviation_mm": 0.02, "rms_deviation_mm": 0.003,
                           "max_abs_coordinate_deviation_mm": 0.01},
            "parameter_map": {"steps": 75},
            "noise_scale": {"filter_applied": False, "unit": "mm"},
            "seed_derivation": {"requested_seed": seed, "unit_test_only": True},
        }
        return SimpleNamespace(valid=valid, plan=(None if not valid else
                                                  SimpleNamespace(motion_hash=f"mock-{fraction}-{seed}")),
                               invalid_reason=(None if valid else "machine envelope violation; no clipping"),
                               declaration_hash="sha256:" + "a" * 64, derivation=json.dumps(derivation))

    report = R.scale_validity_sweep(build_pilot_task(), config, generator=controlled)
    assert len(report["rows"]) == len(calls) == 150
    assert len(report["summaries"]) == 5
    assert [(row["fraction"], row["seed"]) for row in report["rows"]] == [
        (fraction, seed) for fraction in config.sweep_fractions for seed in config.sweep_seeds]
    for summary in report["summaries"]:
        assert summary["attempted"] == summary["valid"] + summary["failed"] == 30
        assert 0 <= summary["valid_rate"] <= 1
    failures = [row for row in report["rows"] if not row["valid"]]
    assert failures and all("envelope" in row["invalid_reason"] for row in failures)
    assert all(row["status"] == "failed_physical_constraint" for row in failures)
    assert all(row["deviations"]["rms_deviation_mm"] == 0.003 for row in failures)
    assert all(row["T"] == 75 for row in failures)
    # This controlled generator is deliberately a unit oracle, not physics/e2e evidence.
    assert report["physical_validity_only"] is True


def test_fixed_path_source_and_fixed_theta_hashes_stay_distinct():
    from scripts import run_gradient_mechanics as R
    from benchmarks.pilot_task import build_pilot_task

    class FixedPathFactory:
        def __init__(self, task, motion, parameterization, sim, *, spec=None, ledger=None, seed=0):
            self.motion = motion

        def candidate_motion(self, theta):
            assert np.asarray(theta).shape == (36,)
            return SimpleNamespace(motion_hash="unit-fixed-theta-path",
                                   waypoints_mm=self.motion.waypoints_mm)

    config = R.study_config_from_json(_declared_fixed_study_document())
    task = build_pilot_task()  # Analytic metadata only; no simulator/runtime started.
    base = FixedPathFactory(task, None, None, object())
    base.sim, base.spec, base.ledger, base.seed = object(), None, None, 0
    nominal = R._prepare_study_path(config.paths[0], task, base, config)
    assert nominal["status"] == "ready" and nominal["declaration"] is None
    assert nominal["source_record"]["mode"] == "nominal_cam" and nominal["source_result"] is None
    ready = R._prepare_study_path(config.paths[1], task, base, config)
    assert ready["status"] == "ready"
    assert ready["source_motion"].frame == C.Frame.MACHINE
    assert ready["motion"].motion_hash == "unit-fixed-theta-path"
    assert ready["segment_count"] == 75
    assert ready["parameter_shape"] == [12, 3]
    assert ready["theta"].shape == (36,) and np.all(ready["theta"] == 0)
    assert ready["source_hash"] == ready["source_motion"].motion_hash
    assert ready["fixed_motion_hash"] == "unit-fixed-theta-path"
    assert ready["source_record"]["derivation"]["exploratory"] is True
    assert ready["source_record"]["derivation"]["excluded_from_init_axis_claims"] is True
    assert ready["declaration"]["noise_scale_metadata"]["unit"] == "normalized_theta"
    assert ready["source_to_fixed_path_max_error_mm"] == 0.0


def test_study_thread_cap_validation_happens_before_any_runtime_or_files(tmp_path):
    from scripts.run_gradient_mechanics import build_pilot_evaluator, CostConfig
    for maximum in (0, 9, 8.0):
        output = tmp_path / f"invalid-{maximum}"
        with pytest.raises(ValueError, match="max_cpu_threads"):
            build_pilot_evaluator(output, config=CostConfig(voxel_mm=2.0),
                                  precision="f64", max_cpu_threads=maximum)
        assert not output.exists()
