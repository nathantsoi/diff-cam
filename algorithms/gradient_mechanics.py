"""Gradient mechanics utilities with explicit parameter coordinates and ledgers.

Sigma and finite-difference epsilons are normalized **T2.1 theta units**
(control-point residuals scaled by ``parameterization.scale_mm``), not normalized
stock-cube coordinates. The antithetic estimator evaluates hard candidates at
``theta +/- sigma*z`` with standard normal ``z``. It never claims equivalence to
an unsmoothed hard trajectory or to a different smooth surrogate.

Raw simulator/tool path data has separate machine-mm and normalized-stock
representation. Comparisons must not mix these units. Real Taichi float32 values
use tolerances; tests never require bit-identical physical-simulation values.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from benchmarks import contracts as C
from .run_artifacts import ComputeLedger


@dataclass
class EstimateReport:
    gradient: Optional[np.ndarray]
    standard_error: Optional[np.ndarray]
    standard_error_rms: Optional[float]
    pairs: int
    sigma: np.ndarray
    seed: dict[str, Any]
    noise_sha256: str
    ledger_delta: dict[str, Any]
    invalid_pairs: list[dict[str, Any]]
    status: str
    null_reason: Optional[str] = None


@dataclass
class PathReport:
    matches: bool
    max_machine_error_mm: float
    max_normalized_error: float
    typed_motion_hash: str
    normalized_tolerance: float
    machine_tolerance_mm: float
    coordinate_checks: list[dict[str, Any]]


def _theta(theta) -> np.ndarray:
    value = np.asarray(theta, dtype=np.float64)
    if value.ndim == 0 or not np.isfinite(value).all():
        raise ValueError("theta must be a finite nonempty normalized parameter array")
    return value.reshape(-1)


def _positive_vector(value, dimension: int, name: str) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive scalar or ({dimension},) vector, not bool")
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} must not be a boolean array")
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 0:
        raw = np.full(dimension, float(raw), dtype=np.float64)
    if raw.shape != (dimension,) or not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError(f"{name} must be a finite positive scalar or ({dimension},) vector")
    return raw


def _positive_int(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _nonce(seed: int | np.random.SeedSequence | Sequence[int]) -> np.random.SeedSequence:
    if seed is None:
        raise ValueError("an explicit deterministic seed is required")
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, bool):
        raise ValueError("seed must be a nonnegative integer or SeedSequence, not bool")
    try:
        return np.random.SeedSequence(seed)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid deterministic seed: {error}") from error


def split_streams(seed) -> tuple[np.random.SeedSequence, np.random.SeedSequence]:
    """Independent reproducible streams for the small estimate and its reference."""
    return tuple(_nonce(seed).spawn(2))


def _score(evaluate, theta, candidate_id, kind):
    if hasattr(evaluate, "evaluate_candidate"):
        value = evaluate.evaluate_candidate(theta, candidate_id, kind)
    else:
        value = evaluate(theta, candidate_id, kind)
    if value is None or not hasattr(value, "valid") or not hasattr(value, "objective"):
        raise TypeError("objective callback must return a CandidateResult-like object")
    if getattr(value, "objective_kind", kind) != kind:
        raise TypeError(f"objective callback switched from {kind} to {value.objective_kind!r}")
    return value


def _snapshot(ledger: ComputeLedger) -> dict[str, Any]:
    before = ledger.snapshot()
    # Do not copy a potentially large per-candidate history into every estimate.
    return {name: value for name, value in before.items() if name != "entries"}


def delta_snapshot(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Numeric phase-counter deltas, not the immutable ledger documentation.

    Timing and process-high-water-memory fields remain explicit; descriptive
    strings and the potentially large per-candidate history are never subtracted.
    """
    keys = {
        "candidate_evaluations", "forward_calls", "backward_calls", "validation_calls",
        "segment_steps", "voxel_steps", "warmup_wall_time_s", "steady_state_wall_time_s",
        "end_to_end_candidate_wall_time_s", "process_peak_rss_bytes",
        "python_traced_memory_current_bytes", "python_traced_memory_peak_bytes",
    }
    return {name: after.get(name, 0) - before.get(name, 0) for name in keys if name in before or name in after}


def antithetic_estimate(evaluate, theta, sigma, pairs: int, *, seed,
                        ledger: Optional[ComputeLedger] = None,
                        label: str = "antithetic") -> EstimateReport:
    """Estimate a Gaussian-smoothed hard gradient with exactly 2N candidates.

    The estimator is the mean of ``(L(theta+sigma*z)-L(theta-sigma*z))z/(2*sigma)``.
    The two evaluations are consecutive sequential ledger candidates. No
    deterministic-extra re-sampling, clipping or branch-specific objective swap
    occurs. Invalid candidates are reported without inventing a finite gradient.
    """
    _positive_int(pairs, "pairs")
    ledger = ledger or ComputeLedger()
    theta = _theta(theta)
    local_sigma = _positive_vector(sigma, len(theta), "sigma")
    stream = _nonce(seed)
    seed_record = {"entropy_source": "numpy.random.SeedSequence",
                   "entropy": stream.entropy, "spawn_key": stream.spawn_key,
                   "rng": "numpy.random.Generator/PCG64"}
    z = np.random.default_rng(stream).standard_normal((pairs, len(theta)))
    if not np.isfinite(z).all():
        raise FloatingPointError("non-finite Gaussian perturbations")
    noise_sha256 = C.sha256_bytes(np.ascontiguousarray(z, dtype=np.float64).tobytes())
    before = _snapshot(ledger)
    contributions = np.empty((pairs, len(theta)), dtype=np.float64)
    invalid = []
    for i in range(pairs):
        delta = local_sigma * z[i]
        result_data = []
        for sign, suffix in ((1.0, "plus"), (-1.0, "minus")):
            candidate_id = f"{label}-{suffix}-{i:06d}"
            result = _score(evaluate, theta + sign * delta, candidate_id, "hard")
            result_data.append(result)
            if not result.valid:
                invalid.append({"pair": i, "side": suffix, "candidate_id": candidate_id,
                                "reason": result.invalid_reason or "candidate rejected"})
        if not all(result.valid for result in result_data):
            # Do not manufacture a numeric pair from an invalid objective.
            continue
        if not all(np.isfinite(float(result.objective)).all() for result in result_data):
            invalid.append({"pair": i, "side": "non-finite-value",
                            "reason": "valid candidate response was non-finite"})
            continue
        contributions[i] = (float(result_data[0].objective) - float(result_data[1].objective)) * z[i] / (2.0 * local_sigma)
    after = _snapshot(ledger)
    stats = delta_snapshot(before, after)
    if invalid:
        gradient, error, rms = None, None, None
        status = "invalid"
    else:
        used = int(stats["candidate_evaluations"])
        if used != 2 * pairs:
            raise RuntimeError(f"hard-evaluation accounting mismatch: {used} != {2*pairs}")
        gradient = contributions.mean(axis=0, dtype=np.float64)
        if not np.isfinite(gradient).all():
            gradient, error, rms = None, None, None
            status = "invalid"
        else:
            if pairs == 1:
                error, rms = None, None
            else:
                error = contributions.std(axis=0, ddof=1, dtype=np.float64) / math.sqrt(pairs)
                rms = float(np.sqrt(np.sum(error * error)))
            if error is not None and not np.isfinite(error).all():
                raise FloatingPointError("non-finite antithetic standard error")
            status = "ok"
    null_reason = (("one_pair_has_no_sample_standard_error" if pairs == 1 and not invalid else None)
                   or ("invalid candidates are retained; no estimator claimed" if invalid else None))
    return EstimateReport(gradient, error, rms, pairs, local_sigma.copy(), seed_record,
                          noise_sha256, stats, invalid, status, null_reason)


def reference_antithetic(evaluate, theta, sigma, reference_pairs: int, *, seed,
                         ledger: Optional[ComputeLedger] = None) -> EstimateReport:
    """A higher-sample estimate from an explicitly independent random stream.

    Exactly 2*reference_pairs calls are charged. There is no unreported base
    (zero-perturbation) evaluation and no reference-vs-primary reuse of samples.
    """
    primary, reference = split_streams(seed)
    return antithetic_estimate(evaluate, theta, sigma, reference_pairs, seed=reference,
                               ledger=ledger, label="reference-antithetic")


def estimate_with_reference(evaluate, theta, sigma, pairs: int, reference_pairs: int, *,
                            seed, ledger: Optional[ComputeLedger] = None) -> dict[str, Any]:
    """Primary and high-sample estimates, with independent sampling uncertainty.

    The comparison can show proximity/divergence of these estimates. It is not a
    proof that randomized hard and voxel-sigmoid smooth gradients are equivalent.
    """
    if isinstance(reference_pairs, bool) or not isinstance(reference_pairs, int) or reference_pairs <= pairs:
        raise ValueError("reference_pairs must exceed the positive primary pair count")
    ledger = ledger or ComputeLedger()
    before = _snapshot(ledger)
    primary_seed, reference_seed = split_streams(seed)
    primary = antithetic_estimate(evaluate, theta, sigma, pairs, seed=primary_seed,
                                  ledger=ledger, label="primary-antithetic")
    reference = antithetic_estimate(evaluate, theta, sigma, reference_pairs, seed=reference_seed,
                                    ledger=ledger, label="reference-antithetic")
    if primary.status == "ok" and reference.status == "ok":
        discrepancy = np.asarray(reference.gradient) - np.asarray(primary.gradient)
        summary = gradient_alignment(primary.gradient, reference.gradient)
        summary["higher_sample_difference"] = discrepancy.tolist()
        if primary.standard_error is None or reference.standard_error is None:
            summary["joint_standard_error_rms"] = None
            summary["similarity_with_uncertainty_status"] = "undefined: a single-pair estimate has no sample variance"
        else:
            joint_se = float(math.sqrt(primary.standard_error_rms ** 2 + reference.standard_error_rms ** 2))
            summary["joint_standard_error_rms"] = joint_se
            summary["discrepancy_in_joint_standard_errors"] = (
                float(np.linalg.norm(discrepancy)) / joint_se if joint_se > 0 else None)
            summary["similarity_with_uncertainty_status"] = (
                "finite-se samples compared; not a continuous-gradient-equivalence proof")
    else:
        summary = {"status": "invalid", "cosine": None,
                   "joint_standard_error_rms": None,
                   "reason": "cannot compare an undefined rejected-candidate estimate"}
    return {"primary": primary, "reference": reference,
            "comparison": summary, "ledger_delta": delta_snapshot(before, _snapshot(ledger))}


def _fd_one(evaluate, theta: np.ndarray, epsilon: np.ndarray, index: int, sign: int, kind="smooth",
            label="central", observer=None):
    displaced = theta.copy()
    displaced[index] += sign * epsilon[index]
    result = _score(evaluate, displaced, f"fd-axis-{index}-{sign}-{label}", kind)
    if not result.valid or not math.isfinite(result.objective):
        reason = result.invalid_reason or "finite-difference objective rejected direction"
        raise ValueError(f"central FD axis {index} sign {sign}: {reason}")
    if observer is not None:
        observer(displaced, result)
    return result


def central_differences(evaluate, theta, epsilons, *, indices=None,
                        ledger: Optional[ComputeLedger] = None,
                        observer=None) -> list[dict[str, Any]]:
    """Evaluate central finite differences on one identical smooth objective.

    ``indices`` may deliberately scope a cost probe to selected coordinates; the
    returned vector always contains entries for all normalised parameters,
    including ``None`` at unevaluated entries. No zero is invented for them.
    """
    ledger = ledger or ComputeLedger()
    theta = _theta(theta)
    if indices is None:
        selected = list(range(len(theta)))
    else:
        if len(indices) == 0:
            raise ValueError("finite-difference indices cannot be empty")
        selected = []
        for index in indices:
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
                raise ValueError("finite-difference coordinate indices must be integers")
            if int(index) < 0 or int(index) >= len(theta):
                raise ValueError("finite-difference coordinate index outside theta")
            if int(index) in selected:
                raise ValueError("finite-difference coordinate indices must be unique")
            selected.append(int(index))
    if (isinstance(epsilons, (int, float, np.integer, np.floating))
            or (isinstance(epsilons, np.ndarray) and epsilons.ndim == 0)):
        epsilon_values = [float(epsilons)]
    else:
        try:
            epsilon_values = list(epsilons)
        except TypeError as error:
            raise ValueError("epsilons must be a scalar or a sweep of scalars/vectors") from error
    if not epsilon_values:
        raise ValueError("epsilons must be a nonempty scalar or one-dimensional sweep")
    epsilon_vectors = [_positive_vector(value, len(theta), "finite-difference epsilon")
                       for value in epsilon_values]
    reports = []
    for epsilon in epsilon_vectors:
        before = _snapshot(ledger)
        try:
            fd = np.full(len(theta), np.nan, dtype=np.float64)
            for index in selected:
                plus = _fd_one(evaluate, theta, epsilon, index, +1, label=f"eps-{len(reports)}",
                               observer=observer)
                minus = _fd_one(evaluate, theta, epsilon, index, -1, label=f"eps-{len(reports)}",
                                observer=observer)
                fd[index] = (plus.objective - minus.objective) / (2 * epsilon[index])
            after = _snapshot(ledger)
            reports.append({"epsilon": epsilon.copy(), "gradient": fd,
                            "evaluated_indices": selected.copy(), "sample_scope": "subset",
                            "center_objective": None,
                            "ledger_delta": delta_snapshot(before, after), "status": "ok"})
        except ValueError as error:
            reports.append({"epsilon": epsilon.copy(), "gradient": None,
                            "ledger_delta": delta_snapshot(before, after),
                            "status": "invalid", "reason": str(error),
                            "evaluated_indices": selected.copy()})
    return reports


def gradient_alignment(first, second, *, norm_epsilon: float = 1e-10) -> dict[str, Any]:
    """Return cosine/discrepancy, not an all-purpose equivalence claim."""
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("gradient vectors must be finite, equal-size vectors")
    norm_epsilon = float(norm_epsilon)
    if not math.isfinite(norm_epsilon) or norm_epsilon < 0:
        raise ValueError("norm_epsilon must be finite and nonnegative")
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    discrepancy = np.linalg.norm(a - b)
    maximum = float(np.max(np.abs(a - b))) if a.size else 0.0
    result = {
        "first_norm": an, "second_norm": bn, "difference_norm": float(discrepancy),
        "max_abs_coordinate_difference": maximum,
        "near_zero_threshold": norm_epsilon,
    }
    if an <= norm_epsilon and bn <= norm_epsilon:
        result.update(status="both_near_zero", cosine=None,
                      reason="both norms are below the explicit near-zero threshold")
        return result
    if an <= norm_epsilon or bn <= norm_epsilon:
        result.update(status="one_near_zero", cosine=None,
                      reason="cosine is undefined for a near-zero vector")
        return result
    cosine = float(np.dot(a, b) / (an * bn))
    if not math.isfinite(cosine):
        raise FloatingPointError("non-finite gradient cosine")
    result.update(status="ok", cosine=max(-1.0, min(1.0, cosine)),
                  relative_difference_norm=float(discrepancy / an), reason=None)
    return result


def execute_normalized_and_machine(motion: C.MotionPlan, stock_origin_mm,
                                   extent_mm) -> tuple[np.ndarray, np.ndarray]:
    """Verify a typed plan's machine and normalized coordinates under one datum."""
    points = np.asarray(motion.waypoints_mm(), dtype=np.float64)
    initial = np.asarray(motion.initial_state.position_mm, dtype=np.float64)
    origin = np.asarray(stock_origin_mm, dtype=np.float64)
    extent = np.asarray(extent_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("typed motion needs finite (T,3) physical positions")
    if motion.frame != C.Frame.MACHINE:
        raise ValueError("path reconstruction requires machine-frame physical-mm motion")
    if initial.shape != (3,) or np.any(np.abs(points[0] - initial) > 1e-6):
        raise ValueError("plan's first position disagrees with the known initial machine point")
    if origin.shape != (3,) or extent.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("stock origin and extent must be three physical-mm vectors")
    if not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("stock extent must be finite and positive")
    if points.shape[0] != len(motion.segments) + 1:
        raise ValueError("typed segment count and waypoint count disagree")
    normalized = (points - origin) / extent
    return points, normalized


def compare_execution_paths(reference: C.MotionPlan, repeated: C.MotionPlan, stock_origin_mm,
                            extent_mm, *, normalized_tol: float = 2e-6,
                            machine_tol_mm: float = 0.002) -> PathReport:
    """Compare two typed, linear machine paths with explicit coordinate units.

    Motion plans have explicit fixed initial state, kind, duration and mm frame.
    Expected and executed normalized stock coordinates are also compared; the
    latter cannot be mistaken for T2.1 normalized theta parameters.
    """
    for name, mm_tol in (("normalized_tol", normalized_tol), ("machine_tol_mm", machine_tol_mm)):
        if not isinstance(mm_tol, (int, float)) or not math.isfinite(mm_tol) or mm_tol < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if reference.frame != repeated.frame or reference.initial_state != repeated.initial_state:
        raise AssertionError("path initial state/frame changed")
    if reference.timing_rule != repeated.timing_rule or len(reference.segments) != len(repeated.segments):
        raise AssertionError("path timing rule or segment count changed")
    source_world, source_normalized = execute_normalized_and_machine(reference, stock_origin_mm, extent_mm)
    other_world, other_normalized = execute_normalized_and_machine(repeated, stock_origin_mm, extent_mm)
    if source_world.shape != other_world.shape:
        raise AssertionError("path position shape changed")
    machine_error = float(np.linalg.norm(source_world - other_world, axis=1).max())
    normalized_error = float(np.linalg.norm(source_normalized - other_normalized, axis=1).max())
    checks = []
    for left, right in zip(reference.segments, repeated.segments):
        if left.kind == C.SegmentKind.ARC or right.kind == C.SegmentKind.ARC:
            raise ValueError("fixed-position path comparison is scoped to linear motion")
        kind_same = left.kind == right.kind
        feed_same = (left.feed_mm_s is None) == (right.feed_mm_s is None)
        feed_error = 0.0 if left.feed_mm_s is None else abs(left.feed_mm_s - right.feed_mm_s)
        time_error = abs(left.duration_s - right.duration_s)
        if not (kind_same and feed_same and feed_error <= 1e-6 and time_error <= 1e-5):
            raise AssertionError(f"execution kind/feed/timing changed on {left.id}: {left.id} / {right.id}")
        checks.append({"segment": left.id, "kind_same": kind_same, "feed_same": feed_same,
                       "feed_error_mm_s": feed_error, "time_error_s": time_error})
    return PathReport(machine_error <= machine_tol_mm and normalized_error <= normalized_tol,
                      machine_error, normalized_error, repeated.motion_hash,
                      normalized_tol, machine_tol_mm, checks)


def normalized_theta_hash(theta) -> str:
    return C.content_hash(_theta(theta).tolist())


def scalar_result(evaluate, theta, label="verify-ad") -> tuple[Any, np.ndarray]:
    """Get a CandidateEvaluation's value and flat float64 gradient without mixups."""
    if not hasattr(evaluate, "evaluate_candidate_with_gradient"):
        raise TypeError("smooth AD verification requires evaluate_candidate_with_gradient")
    value = evaluate.evaluate_candidate_with_gradient(theta, label)
    if hasattr(value, "gradient_theta"):
        result, gradient = value.result, value.gradient_theta
    else:
        result, gradient = value, None
    if not result.valid or gradient is None:
        raise ValueError(result.invalid_reason or "smooth gradient unavailable")
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if not np.isfinite(gradient).all():
        raise FloatingPointError("non-finite AD gradient")
    return result, gradient


def verify_smooth_ad(evaluate, theta, epsilons, *, indices=None,
                     ledger: Optional[ComputeLedger] = None,
                     norm_epsilon: float = 1e-10,
                     require_path_identity: bool = True) -> dict[str, Any]:
    """Compare smooth AD/FD, with correctly unit-converted simulated paths.

    AD and a separate smooth forward-only replay are evaluated at the SAME theta.
    Their actual paths are compared in both stock-normalized coordinates and
    physical millimetres. Perturbed FD paths are checked against their own typed
    trajectories, not misrepresented as identical to the base path.
    """
    ledger = ledger or ComputeLedger()
    if not hasattr(evaluate, "parameterization") or not hasattr(evaluate, "sim"):
        raise ValueError("verify_smooth_ad expects the existing CandidateEvaluator")
    theta = _theta(theta)
    parameterization = evaluate.parameterization
    raw_expected = np.asarray(parameterization.positions(theta), dtype=np.float64)
    base_typed = evaluate.candidate_motion(theta)
    base_mm, base_normalized = execute_normalized_and_machine(
        base_typed, evaluate.task.grid.origin_mm, evaluate.dimensions,
    )
    if not np.allclose(raw_expected, base_mm, atol=1e-8, rtol=0.0):
        raise AssertionError("typed reconstruction differs from parameterization positions")
    dimensions = np.asarray(evaluate.dimensions, dtype=np.float64)
    origin = np.asarray(evaluate.task.grid.origin_mm, dtype=np.float64)

    def actual_path(candidate_theta, expected_typed=None):
        expected_typed = (evaluate.candidate_motion(candidate_theta)
                          if expected_typed is None else expected_typed)
        expected_mm, expected_normalized = execute_normalized_and_machine(
            expected_typed, origin, dimensions,
        )
        actual_normalized = np.asarray(evaluate.sim.tool_pos.to_numpy(), dtype=np.float64).copy()
        if actual_normalized.shape != expected_normalized.shape or not np.isfinite(actual_normalized).all():
            raise AssertionError("executed trajectory has the wrong shape or non-finite positions")
        # CSG tool_pos is stock-normalized. Convert BEFORE physical-mm checks.
        actual_mm = origin + actual_normalized * dimensions
        record = {
            "motion_hash": expected_typed.motion_hash,
            "theta_hash": normalized_theta_hash(candidate_theta),
            "max_normalized_error": float(np.linalg.norm(actual_normalized - expected_normalized, axis=1).max()),
            "max_machine_error_mm": float(np.linalg.norm(actual_mm - expected_mm, axis=1).max()),
        }
        record["matches_declared_path"] = (record["max_normalized_error"] <= 2e-6
                                           and record["max_machine_error_mm"] <= 0.002)
        return record, actual_normalized

    before = _snapshot(ledger)
    ad_result, ad_gradient = scalar_result(evaluate, theta)
    if ad_gradient.shape != (parameterization.num_params,) or theta.size != ad_gradient.size:
        raise AssertionError("gradient and theta parameter counts differ")
    baseline_path, baseline_ad_normalized = actual_path(theta, base_typed)
    replay = _score(evaluate, theta, "fixed-path-smooth-forward-replay", "smooth")
    if not replay.valid or not math.isfinite(replay.objective):
        raise ValueError(replay.invalid_reason or "fixed-path smooth replay failed")
    replay_path, replay_normalized = actual_path(theta, base_typed)
    delta_normalized = baseline_ad_normalized - replay_normalized
    same_theta_path = {
        "ad_path": baseline_path, "forward_only_replay_path": replay_path,
        "max_normalized_difference": float(np.linalg.norm(delta_normalized, axis=1).max()),
        "max_machine_difference_mm": float(np.linalg.norm(delta_normalized * dimensions, axis=1).max()),
        "absolute_objective_difference": abs(float(replay.objective) - float(ad_result.objective)),
        "same_typed_motion_hash": base_typed.motion_hash,
    }
    same_theta_match = (
        baseline_path["matches_declared_path"] and replay_path["matches_declared_path"]
        and same_theta_path["max_normalized_difference"] <= 2e-6
        and same_theta_path["max_machine_difference_mm"] <= 0.002
    )
    # Observer executes once immediately after each actual forward evaluation,
    # rather than comparing only the last perturbation to the original base.
    sampled_paths = []

    def observe(candidate_theta, result):
        record, _ = actual_path(candidate_theta)
        sampled_paths.append({"candidate_id": result.candidate_id, **record})

    differences = central_differences(evaluate, theta, epsilons, indices=indices,
                                      ledger=ledger, observer=observe)
    path_records = []
    used = 0
    for record in differences:
        item = {"status": record["status"], "evaluated_indices": record["evaluated_indices"]}
        if record["status"] == "ok":
            count = 2 * len(record["evaluated_indices"])
            item["path_checks"] = sampled_paths[used:used + count]
            used += count
            selected = record["evaluated_indices"]
            item["epsilon_scope"] = record["epsilon"][selected].tolist()
            item["gradient_alignment"] = gradient_alignment(
                ad_gradient[selected], record["gradient"][selected], norm_epsilon=norm_epsilon,
            )
        else:
            item["detail"] = record["reason"]
        path_records.append(item)
    all_declared_paths_match = (same_theta_match and used == len(sampled_paths)
                                and all(item["matches_declared_path"] for item in sampled_paths))
    if require_path_identity and not all_declared_paths_match:
        raise AssertionError(f"path execution mismatch: same_theta={same_theta_path}, "
                             f"perturbed_paths={sampled_paths}")
    after = _snapshot(ledger)
    return {
        "ad_gradient": ad_gradient,
        "ad_objective": float(ad_result.objective),
        "same_theta_forward_replay_objective": float(replay.objective),
        "fd": differences,
        "comparisons": path_records,
        "same_theta_path_check": same_theta_path,
        "ledger_delta": delta_snapshot(before, after),
        "same_objective": "CandidateEvaluator smooth objective throughout; randomized hard J_sigma is separate",
        "fixed_theta_execution": {
            "required": require_path_identity,
            "ad_vs_forward_only_same_theta": same_theta_match,
            "all_actual_paths_match_their_declared_theta": all_declared_paths_match,
            "fd_perturbations_are_not_expected_to_equal_the_base_path": True,
            "normalized_tolerance": 2e-6,
            "machine_tolerance_mm": 0.002,
            "actual_normalized_positions_converted_to_mm_before_comparison": True,
            "bitwise_equality_required": False,
        },
    }


def estimate_to_json(report: EstimateReport) -> dict[str, Any]:
    """Bound JSON representation of an estimate, retaining null for unknowns."""
    return {
        "status": report.status,
        "pairs": report.pairs,
        "hard_candidate_evaluations_this_phase": report.ledger_delta["candidate_evaluations"],
        "forward_calls_this_phase": report.ledger_delta["forward_calls"],
        "backward_calls_this_phase": report.ledger_delta["backward_calls"],
        "gradient": None if report.gradient is None else report.gradient.tolist(),
        "standard_error": None if report.standard_error is None else report.standard_error.tolist(),
        "standard_error_rms": report.standard_error_rms,
        "null_reason": report.null_reason,
        "sigma_theta_units": report.sigma.tolist(),
        "seed": report.seed,
        "noise_sha256": report.noise_sha256,
        "invalid_pairs": report.invalid_pairs,
        "ledger_delta": report.ledger_delta,
    }
