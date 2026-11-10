"""Prototype stateless hard/smooth candidate objectives (contracts 1.1).

The current analytic simulator is reused without calling ``ti.init`` and without
editing its source/Task lease. The hard holder scalar is explicitly a normalized
overlap proxy, not a claim of exact equivalence to the differentiable squared-
penetration term. Swap ``reset_simulator`` to the official T1.2/T2.0 hook when it
lands. Speed clipping is disabled in this prototype so published typed motions and
simulated displacements share the same physical execution contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import taichi as ti

from benchmarks import contracts as C
from .run_artifacts import ComputeLedger, simulator_array, simulator_precision_metadata


def _nonnegative_finite(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class LegacyTrajectoryWeights:
    """Existing optional trajectory penalties; disabled explicitly by default."""
    w_air: float = 0.0
    w_jerk: float = 0.0
    w_step: float = 0.0
    w_len: float = 0.0
    w_air_late: float = 0.0
    w_prox: float = 0.0
    w_traj_prox: float = 0.0
    w_break: float = 0.0
    w_tool_gouge: float = 0.0

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            _nonnegative_finite(getattr(self, descriptor.name), descriptor.name)


@dataclass(frozen=True)
class ObjectiveSpec:
    k: float = 40.0
    gouge_weight: float = 2.0
    residual_weight: float = 1.0
    holder_weight: float = 50.0
    holder_margin_mm: float = 0.0
    time_weight: float = 0.0
    air_time_weight: float = 0.0
    reference_cycle_time_s: Optional[float] = None
    legacy_weights: LegacyTrajectoryWeights = field(default_factory=LegacyTrajectoryWeights)
    time_normalization: str = "reference_cycle_seconds"
    geometry_normalization: str = "1/N followed by squared per-voxel occupancy penalties"
    hard_geometry_normalization: str = ("w_gouge*gouge_fraction^2 + w_residual*residual_fraction^2; "
                                        "gouge over |initial∧T|, residual over |initial∧¬T| (csg_metrics parity)")
    hard_holder_normalization: str = "w_holder*hard_holder_overlap_fraction^2 (proxy)"
    hard_efficiency_normalization: str = (
        "geometry+safety+time_weight*cycle_time_s/reference_time_s+"
        "air_time_weight*air_time_s/reference_time_s; exact MotionPlan duration and sharp engagement")
    smooth_efficiency_normalization: str = (
        "geometry+safety+time_weight*n_segments/reference_time_s*mean(native seg_time)+"
        "air_time_weight*n_segments/reference_time_s*mean(native seg_time*sigmoid_air)+"
        "legacy trajectory terms; the factor n restores total-time scale before native means")

    prototype: str = "analytic_target_until_T2.0_custom_target_and_T1.2_chronology"

    def __post_init__(self) -> None:
        for name in ("k", "gouge_weight", "residual_weight", "holder_weight", "holder_margin_mm",
                     "time_weight", "air_time_weight"):
            _nonnegative_finite(getattr(self, name), name)
        if self.time_normalization != "reference_cycle_seconds":
            raise ValueError("time_normalization must be 'reference_cycle_seconds'")
        if self.reference_cycle_time_s is not None:
            _nonnegative_finite(self.reference_cycle_time_s, "reference_cycle_time_s")
            if self.reference_cycle_time_s == 0:
                raise ValueError("reference_cycle_time_s must be positive when supplied")
        if self.legacy_weights.w_air != 0:
            raise ValueError(
                "legacy w_air is disabled until its separate post-cut stock contract is fixed; "
                "use air_time_weight for the pre-cut engagement air term")


@dataclass
class CandidateEvaluation:
    result: C.CandidateResult
    gradient_theta: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _reference_time_seconds(spec: ObjectiveSpec, motion: C.MotionPlan, *,
                            allow_reference_none_for_test: bool = False) -> float:
    """Resolve the declared task reference, with an explicit test-only opt-out.

    Weighted production evaluation must supply ``reference_cycle_time_s``; it is
    never inferred from an uninitialized/random candidate. The narrowly named
    test flag preserves unit construction of local mocks that intentionally
    fall back to their reference motion's duration. It is not runner API usage.
    """
    if not isinstance(allow_reference_none_for_test, bool):
        raise ValueError("allow_reference_none_for_test must be a boolean")
    reference = spec.reference_cycle_time_s
    if reference is None:
        if spec.time_weight > 0 or spec.air_time_weight > 0:
            if not allow_reference_none_for_test:
                raise ValueError("positive efficiency weights require an explicit reference_cycle_time_s")
        # Preserve the legacy weight-zero fallback for unweighted objectives.
        # The flag permits only explicit test fallbacks for weighted fixtures.
        reference = math.fsum(segment.duration_s for segment in motion.segments)
    _nonnegative_finite(reference, "reference_cycle_time_s")
    if reference == 0 and (spec.time_weight > 0 or spec.air_time_weight > 0):
        raise ValueError("positive efficiency weights require a positive reference cycle time")
    return float(reference)


def reset_simulator(sim: Any) -> None:
    """Compatibility helper; become one call to the official T1.2/T2.0 hook.

    Preserves tool/target shape, k, grid and configured execution settings. Every
    current value/gradient/diagnostic accumulator that could leak across a caller
    is cleared before forward() reinitializes active stock and trajectory values.
    """
    official = getattr(sim, "reset_candidate_state", None)
    if official is not None:
        official()
        return
    # The public bulk API walks finalized roots and handles lazy AD buffers.
    ti.ad.clear_all_gradients()
    for name in ("stock", "tool_pos", "tool_delta", "seg_time", "seg_air",
                 "seg_swept", "seg_engage", "saved_stock"):
        getattr(sim, name).fill(0)
    for name in ("loss", "acc_psum"):
        getattr(sim, name)[None] = 0.0
    for name in dir(sim):
        if name.startswith("diag_"):
            getattr(sim, name)[None] = 0.0
    sim._clearance_buf[None] = 0.0
    sim.use_saved_init[None] = 0


def _theta_hash(theta: Any) -> str:
    try:
        value = np.asarray(theta)
    except Exception:
        return C.content_hash(repr(theta))
    if value.dtype.kind in "fiu" and np.isfinite(value).all():
        return C.content_hash(value.tolist())
    return C.content_hash({"shape": list(value.shape), "items": [repr(v) for v in value.flat]})


class CandidateEvaluator:
    """Reusable sequential evaluator; never resets the Taichi runtime.

    Non-finite normalized epsilon/bounds and non-finite simulator results become
    consistent invalid CandidateResults. A ledger attempt is consumed even when
    validation rejects a candidate before touching the simulator.
    """

    def __init__(self, task: C.TaskSpec, motion: C.MotionPlan, parameterization: Any,
                 simulator: Any, *, spec: Optional[ObjectiveSpec] = None,
                 ledger: Optional[ComputeLedger] = None, seed: int = 0,
                 requested_precision: Optional[str] = None) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        motion.validate_for_task(task)
        self.task, self.motion = task, motion
        if motion.frame != C.Frame.MACHINE or motion.timing_rule != C.TimingRule.EXACT_STOP_CONSTANT_SPEED:
            raise ValueError("prototype requires machine-frame exact-stop typed motion")
        self.parameterization = parameterization
        self.sim = simulator
        self.simulator_precision = simulator_precision_metadata(simulator, requested_precision=requested_precision)
        self.spec = spec or ObjectiveSpec()
        self.reference_cycle_time_s = _reference_time_seconds(self.spec, self.motion)
        self._evaluation_k_override: Optional[float] = None
        # Optional, per-run signed field offset in INTERNAL VOXEL SDF units.
        # None restores the neutral zero shift; it is not a scalar loss offset.
        self._evaluation_loss_shift: Optional[float] = None
        self.seed = seed  # no stochastic objective evaluation in this prototype.
        self.ledger = ledger or ComputeLedger()
        if motion.segments and parameterization.num_steps != len(motion.segments):
            raise ValueError("parameterization T must equal the number of typed segments")
        if any(segment.kind == C.SegmentKind.ARC for segment in motion.segments):
            raise ValueError("prototype deformation currently supports linear feed/rapid segments")
        if parameterization.num_steps > simulator.max_steps:
            raise ValueError("simulator max_steps is too small for this parameterization")
        self.dimensions = np.asarray(task.grid.extent_mm, dtype=np.float64)
        stock = task.stock.params
        if stock.get("primitive") != "box":
            raise ValueError("prototype requires analytic box stock")
        if tuple(stock["min_mm"]) != task.grid.origin_mm:
            raise ValueError("prototype grid must begin at the analytic stock min corner")
        if tuple(simulator.stock.shape[1:]) != tuple(task.grid.shape):
            raise ValueError("simulator and task voxel shapes differ")
        # The simulator's target field must be the OPERATION target T (⊇ original part P), per contracts 1.1.
        target_geometry = task.operation_target.geometry
        kind = target_geometry.kind
        primitive = target_geometry.params.get("primitive") if kind == "analytic" else kind
        if primitive != simulator.target_shape:
            raise ValueError(
                f"task target kind {primitive} and simulator target {simulator.target_shape} differ")
        initial_work = np.asarray(task.work_to_machine.inverse().apply(task.initial_state.position_mm))
        if not np.allclose(initial_work, parameterization.initial_position_mm, atol=1e-6, rtol=0):
            raise ValueError("parameterization initial position and task tool tip disagree")

    def set_evaluation_k(self, value: Optional[float]) -> None:
        """Set an explicit per-step smoothing override, or clear it with None.

        This belongs to the executor (e.g. Adam's declared schedule), not the
        stateless objective specification. Every evaluation applies this choice,
        so a previous annealed step cannot silently alter a later CMA method.
        """
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("evaluation_k must be positive or None")
        self._evaluation_k_override = value

    def clear_evaluation_k(self) -> None:
        self._evaluation_k_override = None

    @property
    def evaluation_loss_shift(self) -> Optional[float]:
        """Configured raw override, in voxel SDF units, or neutral ``None``."""
        return self._evaluation_loss_shift

    def _loss_shift_voxels(self) -> float:
        return 0.0 if self._evaluation_loss_shift is None else self._evaluation_loss_shift

    def _loss_shift_mm(self) -> float:
        return self._loss_shift_voxels() * float(self.sim.v)

    def set_evaluation_loss_shift(self, value: Optional[float]) -> None:
        """Shift the smooth stock-field iso-level before each forward, in voxels.

        Positive values move sigmoid(k*(stock_d+shift)) toward more erosion.
        This is NOT an offset added to the scalar objective or hard score. The
        hard forward ignores the field, so policy still sees raw hard results.
        No optimizer state is retained across a fresh CMA/Adam start: those
        entry points clear this override, as they do for per-step k overrides.
        """
        if value is None:
            self.clear_evaluation_loss_shift()
            return
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("evaluation_loss_shift must be finite in [-0.5,0.5] voxels or None")
        value = float(value)
        if not math.isfinite(value) or abs(value) > 0.5:
            raise ValueError("evaluation_loss_shift must be finite in [-0.5,0.5] voxels or None")
        if not hasattr(self.sim, "loss_shift"):
            raise AttributeError("signed stock-field shifts require simulator.loss_shift")
        self._evaluation_loss_shift = value

    def clear_evaluation_loss_shift(self) -> None:
        """Neutralize the per-run field override in voxels (default 0).

        CandidateEvaluator._configure reapplies this on every evaluation, while
        reset_candidate_state retains the configured field just as for other
        execution settings. The following _configure after reset is therefore
        idempotent, not an additional optimizer update.
        """
        self._evaluation_loss_shift = None
        # Clear the field immediately as well as the Python override. Subsequent
        # _configure calls keep this explicitly neutral value until changed.
        self.sim.loss_shift[None] = 0.0

    def _configure(self) -> None:
        spec, sim = self.spec, self.sim
        if not hasattr(sim, "loss_shift"):
            raise AttributeError("signed stock-field shifts require simulator.loss_shift")
        sim.k[None] = spec.k if self._evaluation_k_override is None else self._evaluation_k_override
        sim.loss_shift[None] = self._loss_shift_voxels()
        sim.w_gouge[None] = spec.gouge_weight
        sim.w_residual[None] = spec.residual_weight
        sim.holder_penalty_weight[None] = spec.holder_weight
        sim.holder_margin[None] = spec.holder_margin_mm / sim.v
        # Native smooth terms are per-segment MEANS. Restore total-time scale
        # against the task's TOTAL nominal duration before that mean is taken.
        n_segments = self.parameterization.num_steps
        sim.w_time[None] = (spec.time_weight * n_segments / self.reference_cycle_time_s
                            if spec.time_weight > 0 else 0.0)
        sim.w_air_time[None] = (spec.air_time_weight * n_segments / self.reference_cycle_time_s
                                if spec.air_time_weight > 0 else 0.0)
        for descriptor in fields(spec.legacy_weights):
            getattr(sim, descriptor.name)[None] = getattr(spec.legacy_weights, descriptor.name)
        # Every physical execution is the un-clipped parameterized delta sequence.
        sim.enforce_speed_limits[None] = 0
        sim.enforce_z_floor[None] = 0

    def _deltas(self, theta: Any) -> np.ndarray:
        if not self.parameterization.in_bounds(theta):
            raise ValueError("theta outside declared normalized parameter bounds")
        deltas = np.asarray(self.parameterization.forward(theta), dtype=np.float64)
        if deltas.shape != (self.parameterization.num_steps, 3) or not np.isfinite(deltas).all():
            raise ValueError("parameterization returned non-finite or wrong-shaped deltas")
        return deltas / self.dimensions[None, :]

    def candidate_motion(self, theta: Any) -> C.MotionPlan:
        """Preserve rapid/feed kinds and feed values; recompute exact-stop durations."""
        positions = self.parameterization.positions(theta)
        motion_transform = self.task.work_to_machine.apply
        segments = []
        for original, start_work, end_work in zip(self.motion.segments, positions[:-1], positions[1:]):
            start, end = motion_transform(start_work), motion_transform(end_work)
            length = math.dist(start, end)
            duration = (length / self.task.limits.rapid_mm_s if original.kind == C.SegmentKind.RAPID
                        else length / original.feed_mm_s)
            segments.append(C.Segment(
                id=original.id, kind=original.kind, start_mm=start, end_mm=end,
                duration_s=duration, feed_mm_s=original.feed_mm_s, tag=original.tag,
                source_line=original.source_line,
            ))
        plan = C.MotionPlan(
            frame=C.Frame.MACHINE, initial_state=self.task.initial_state, segments=tuple(segments),
            timing_rule=self.motion.timing_rule, provenance="stateless_candidate_objective",
        )
        plan.validate_for_task(self.task)
        return plan

    def _hard_time_metrics(self, theta: Any) -> tuple[float, float, float]:
        """Use exact motion durations with sharp per-segment engagement, not native feed rates."""
        motion = self.candidate_motion(theta)
        steps = len(motion.segments)
        self.sim.compute_traj_diagnostics_hard(steps)
        engagement = np.asarray(self.sim.seg_engage.to_numpy(), dtype=np.float64).reshape(-1)
        durations = np.asarray([segment.duration_s for segment in motion.segments], dtype=np.float64)
        if engagement.size != steps or durations.size != steps:
            raise ValueError("hard timing and engagement segment counts differ")
        if not np.all(np.isfinite(engagement)) or not np.all(np.isfinite(durations)):
            raise ValueError("non-finite hard timing or engagement data")
        cells = self.sim.Nx * self.sim.Ny * self.sim.Nz
        if cells <= 0:
            raise ValueError("air-time classification requires a nonempty simulator grid")
        air = engagement <= 0.5 / cells
        cycle_time_s = math.fsum(durations.tolist())
        air_time_s = math.fsum(durations[air].tolist())
        air_fraction = air_time_s / cycle_time_s if cycle_time_s > 0 else 0.0
        if (air_time_s < 0 or cycle_time_s < 0 or air_time_s > cycle_time_s
                or not all(math.isfinite(value) for value in (cycle_time_s, air_time_s, air_fraction))):
            raise ValueError("invalid exact candidate time/air-time components")
        return cycle_time_s, air_time_s, air_fraction

    def _native_time_metrics(self) -> tuple[float, float, float]:
        """Seconds from the native differentiable surrogate, including its regime/dt limits."""
        steps = self.parameterization.num_steps
        seg_time = np.asarray(self.sim.seg_time.to_numpy(), dtype=np.float64).reshape(-1)
        engagement = np.asarray(self.sim.seg_engage.to_numpy(), dtype=np.float64).reshape(-1)
        if seg_time.size != steps or engagement.size != steps:
            raise ValueError("native timing and engagement segment counts differ")
        if not np.all(np.isfinite(seg_time)) or not np.all(np.isfinite(engagement)):
            raise ValueError("non-finite native timing or engagement data")
        cells = self.sim.Nx * self.sim.Ny * self.sim.Nz
        inv_voxel = 1.0 / cells
        temperature = max(1e-5, 0.25 * inv_voxel)
        argument = np.clip((engagement - inv_voxel) / temperature, -50.0, 50.0)
        is_air = 1.0 / (1.0 + np.exp(argument))
        cycle_time_s = float(seg_time.mean()) if steps else 0.0
        air_time_s = float((seg_time * is_air).mean()) if steps else 0.0
        air_fraction = air_time_s / cycle_time_s if cycle_time_s > 0 else 0.0
        return cycle_time_s, air_time_s, air_fraction

    def _invalid(self, candidate_id: str, theta: Any, kind: str, reason: str) -> CandidateEvaluation:
        return CandidateEvaluation(C.CandidateResult(
            candidate_id=candidate_id, theta_hash=_theta_hash(theta), objective_kind=kind,
            objective=None, valid=False, invalid_reason=reason,
            coverage=C.Coverage.UNKNOWN,
            components={
                "loss_shift_voxels": self._loss_shift_voxels(),
                "loss_shift_mm": self._loss_shift_mm(),
            },
        ))

    def _gradient(self, normalized_gradient: np.ndarray) -> np.ndarray:
        physical = normalized_gradient[:self.parameterization.num_steps] / self.dimensions[None, :]
        return np.asarray(self.parameterization.pullback(physical), dtype=np.float64)

    def _evaluate(self, theta: Any, candidate_id: str, objective_kind: str,
                  with_gradient: bool) -> CandidateEvaluation:
        if objective_kind not in ("smooth", "hard"):
            raise ValueError("objective_kind must be smooth or hard")
        spec = self.spec
        self.ledger.start_candidate(candidate_id, objective_kind)
        forward_calls = backward_calls = 0
        steps = self.parameterization.num_steps
        voxels = steps * self.sim.Nx * self.sim.Ny * self.sim.Nz
        valid = False
        try:
            normalized = self._deltas(theta)
            self._configure()
            reset_simulator(self.sim)
            # Official reset preserves configured fields. Apply this explicitly
            # once more for any alternative reset implementation, not ti.init.
            self.sim.loss_shift[None] = self._loss_shift_voxels()
            self.sim.tool_delta.from_numpy(simulator_array(self.sim, normalized))
            forward_calls = 1
            if objective_kind == "smooth":
                if with_gradient:
                    backward_calls = 1
                    with ti.ad.Tape(loss=self.sim.loss):
                        self.sim.forward(steps + 1)
                    normalized_gradient = self.sim.tool_delta.grad.to_numpy().copy()
                    gradient = self._gradient(normalized_gradient)
                    if not np.all(np.isfinite(gradient)):
                        raise FloatingPointError("non-finite candidate gradient")
                else:
                    self.sim.forward(steps + 1)
                objective = float(self.sim.loss[None])
                self.sim.compute_diagnostics(steps)
                diagnostics = {
                    "gouge_soft": float(self.sim.diag_gouge[None]),
                    "residual_soft": float(self.sim.diag_residual[None]),
                    "holder_soft": float(self.sim.diag_holder[None]),
                    "final_stock": self.sim.stock.to_numpy()[steps].copy(),
                    "time_source_is_simulator_surrogate": 1.0,
                }
                native_metrics = spec.time_weight > 0 or spec.air_time_weight > 0
                native_metrics = (native_metrics or spec.legacy_weights.w_air_late > 0
                                  or spec.legacy_weights.w_break > 0)
                diagnostics["efficiency_timing_evaluated"] = float(native_metrics)
                if native_metrics:
                    native_cycle, native_air, native_fraction = self._native_time_metrics()
                    diagnostics.update(cycle_time_s=native_cycle, air_time_s=native_air,
                                       air_fraction=native_fraction)
            else:
                self.sim.forward_hard(steps + 1, clip_speeds=False)
                target = self.sim.target.to_numpy() < 0
                history = self.sim.stock.to_numpy()
                initial = history[0] < 0
                stock = history[steps] < 0
                # Same definitions/denominators as simulator.csg_metrics (contracts 1.1, coordinator decision):
                # gouge = removed ∧ T over |initial ∧ T|; residual = remaining ∧ ¬T over |initial ∧ ¬T|.
                in_target = np.count_nonzero(initial & target)
                removal_target = np.count_nonzero(initial & ~target)
                if not in_target or not removal_target:
                    # Review M5: csg_metrics reports null for these denominators; never score a degenerate
                    # task as perfect. ValueError -> invalid CandidateResult with this reason.
                    raise ValueError("degenerate task: no stock inside the operation target "
                                     f"({in_target} cells) or empty removal target ({removal_target} cells)")
                gouge_fraction = float(np.count_nonzero(initial & target & ~stock) / in_target)
                residual_fraction = float(np.count_nonzero(~target & stock) / removal_target)
                holder_fraction = float(self.sim.holder_overlap_total(steps))
                geometry = (self.spec.gouge_weight * gouge_fraction ** 2
                            + self.spec.residual_weight * residual_fraction ** 2)
                safety = self.spec.holder_weight * holder_fraction ** 2
                cycle_time_s, air_time_s, air_fraction = self._hard_time_metrics(theta)
                reference_s = self.reference_cycle_time_s
                if reference_s > 0:
                    time_term = self.spec.time_weight * cycle_time_s / reference_s
                    air_term = self.spec.air_time_weight * air_time_s / reference_s
                else:
                    time_term = air_term = 0.0
                objective = geometry + safety + time_term + air_term
                voxel_volume_mm3 = self.task.grid.voxel_mm ** 3
                gouge_limit = self.task.tolerances.gouge_volume_mm3 / (in_target * voxel_volume_mm3)
                residual_limit = self.task.tolerances.completion_residual_mm3 / (
                    removal_target * voxel_volume_mm3)
                constraint_violation = (holder_fraction + max(0.0, gouge_fraction - gouge_limit)
                                        + max(0.0, residual_fraction - residual_limit))
                feasible = float(holder_fraction == 0 and gouge_fraction <= gouge_limit
                                 and residual_fraction <= residual_limit)
                diagnostics = {
                    "gouge_fraction": gouge_fraction,
                    "residual_fraction": residual_fraction,
                    "holder_overlap_fraction": holder_fraction,
                    "geometry": geometry, "holder_proxy": safety, "final_stock": stock.copy(),
                    "cycle_time_s": cycle_time_s, "air_time_s": air_time_s,
                    "air_fraction": air_fraction, "efficiency": objective,
                    "efficiency_time_term": time_term, "efficiency_air_time_term": air_term,
                    "gouge_fraction_limit": gouge_limit,
                    "residual_fraction_limit": residual_limit,
                    "constraint_violation": constraint_violation, "feasible": feasible,
                    "time_source_is_simulator_surrogate": 0.0,
                }
            if not math.isfinite(objective):
                raise FloatingPointError("simulator returned a non-finite objective")
            components = {
                "objective_value": objective,
                "delta_l2_normalized": float(np.linalg.norm(normalized)),
                "speed_clipping_applied": 0.0,
                "evaluation_seed": float(self.seed),
                "loss_shift_voxels": self._loss_shift_voxels(),
                "loss_shift_mm": self._loss_shift_mm(),
                "prototype_incomplete_final_target_and_posted_coverage": 1.0,
            }
            components["efficiency"] = objective
            components["reference_cycle_time_s"] = self.reference_cycle_time_s
            for name, value in diagnostics.items():
                if name != "final_stock":
                    components[name] = float(value)
            result = C.CandidateResult(
                candidate_id=candidate_id, theta_hash=_theta_hash(theta),
                objective_kind=objective_kind, objective=objective, valid=True,
                coverage=C.Coverage.INCOMPLETE, components=components,
                gradient_available=with_gradient and objective_kind == "smooth",
            )
            valid = True
            return CandidateEvaluation(result, gradient if with_gradient and objective_kind == "smooth" else None,
                                       diagnostics)
        except (ValueError, FloatingPointError) as error:
            return self._invalid(candidate_id, theta, objective_kind, str(error))
        finally:
            entry = self.ledger.finish_candidate(valid=valid, forward_calls=forward_calls,
                                                 backward_calls=backward_calls,
                                                 segment_steps=steps if forward_calls else 0,
                                                 voxel_steps=voxels if forward_calls else 0)
            entry["loss_shift_voxels"] = self._loss_shift_voxels()
            entry["loss_shift_mm"] = self._loss_shift_mm()

    def evaluate_candidate(self, theta: Any, candidate_id: str,
                           objective_kind: str = "smooth") -> C.CandidateResult:
        return self._evaluate(theta, candidate_id, objective_kind, with_gradient=False).result

    def evaluate_candidate_with_gradient(self, theta: Any, candidate_id: str) -> CandidateEvaluation:
        return self._evaluate(theta, candidate_id, "smooth", with_gradient=True)
