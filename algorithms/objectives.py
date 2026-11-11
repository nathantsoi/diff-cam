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
    mrr_weight: float = 0.0
    mrr_limit_mm3_s: float = 100.0
    rapid_stock_weight: float = 0.0
    tool_gouge_margin_mm: float = 0.0
    use_typed_segment_speeds: bool = False
    enforce_mrr_constraint: bool = False
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
                     "time_weight", "air_time_weight", "mrr_weight", "mrr_limit_mm3_s",
                     "rapid_stock_weight", "tool_gouge_margin_mm"):
            _nonnegative_finite(getattr(self, name), name)
        for flag_name in ("use_typed_segment_speeds", "enforce_mrr_constraint"):
            if not isinstance(getattr(self, flag_name), bool):
                raise ValueError(f"{flag_name} must be boolean")
        if self.mrr_weight > 0 and self.mrr_limit_mm3_s <= 0:
            raise ValueError("mrr_limit_mm3_s must be positive when mrr_weight > 0")
        if self.enforce_mrr_constraint and self.mrr_limit_mm3_s <= 0:
            raise ValueError("mrr_limit_mm3_s must be positive when enforce_mrr_constraint is True")
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
        if motion.frame != C.Frame.MACHINE or motion.timing_rule not in (
            C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
            C.TimingRule.BLENDED_ACCEL_BOUNDED,
        ):
            raise ValueError("prototype requires machine-frame exact-stop or blended-accel typed motion")
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

    def _use_per_seg_speeds(self) -> bool:
        return bool(
            self.spec.use_typed_segment_speeds
            or self.motion.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED
            or hasattr(self.parameterization, "forward_feed_multipliers")
        )

    def _feed_multipliers(self, theta: Any) -> np.ndarray:
        steps = self.parameterization.num_steps
        if hasattr(self.parameterization, "forward_feed_multipliers"):
            mult = np.asarray(self.parameterization.forward_feed_multipliers(theta), dtype=np.float64)
            if mult.shape != (steps,) or not np.isfinite(mult).all() or np.any(mult <= 0):
                raise ValueError("parameterization returned non-finite or non-positive feed multipliers")
            return mult
        return np.ones(steps, dtype=np.float64)

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
        if hasattr(sim, "tool_gouge_margin"):
            sim.tool_gouge_margin[None] = spec.tool_gouge_margin_mm / sim.v
        # Native smooth terms are per-segment MEANS. Restore total-time scale
        # against the task's TOTAL nominal duration before that mean is taken.
        n_segments = self.parameterization.num_steps
        sim.w_time[None] = (spec.time_weight * n_segments / self.reference_cycle_time_s
                            if spec.time_weight > 0 else 0.0)
        sim.w_air_time[None] = (spec.air_time_weight * n_segments / self.reference_cycle_time_s
                                if spec.air_time_weight > 0 else 0.0)
        for descriptor in fields(spec.legacy_weights):
            getattr(sim, descriptor.name)[None] = getattr(spec.legacy_weights, descriptor.name)
        use_per_seg = self._use_per_seg_speeds()
        if hasattr(sim, "use_per_seg_speeds"):
            sim.use_per_seg_speeds[None] = 1 if use_per_seg else 0
            sim.use_blended_timing[None] = (
                1 if self.motion.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED else 0
            )
            sim.max_accel_mm_s2[None] = float(self.task.limits.max_accel_mm_s2)
            sim.w_mrr[None] = float(spec.mrr_weight * n_segments)
            sim.mrr_limit_mm3_s[None] = float(spec.mrr_limit_mm3_s)
            sim.w_rapid_stock[None] = float(spec.rapid_stock_weight)
            if use_per_seg:
                nom_speeds = np.zeros(sim.max_steps, dtype=np.float64)
                is_rapid = np.zeros(sim.max_steps, dtype=np.int32)
                rapid_spd = float(self.task.limits.rapid_mm_s)
                for idx, seg in enumerate(self.motion.segments):
                    if seg.kind == C.SegmentKind.RAPID:
                        nom_speeds[idx] = rapid_spd
                        is_rapid[idx] = 1
                    else:
                        nom_speeds[idx] = float(seg.feed_mm_s or rapid_spd)
                        is_rapid[idx] = 0
                sim.seg_nom_speed.from_numpy(simulator_array(sim, nom_speeds))
                sim.seg_is_rapid.from_numpy(is_rapid)
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
        """Preserve rapid/feed kinds; apply feed multipliers and exact-stop or blended durations."""
        positions = self.parameterization.positions(theta)
        feed_mult = self._feed_multipliers(theta)
        motion_transform = self.task.work_to_machine.apply
        segments = []
        for original, start_work, end_work, mult in zip(
            self.motion.segments, positions[:-1], positions[1:], feed_mult
        ):
            start, end = motion_transform(start_work), motion_transform(end_work)
            length = math.dist(start, end)
            if original.kind == C.SegmentKind.RAPID:
                eff_feed = None
                duration = length / self.task.limits.rapid_mm_s
            else:
                eff_feed = float(original.feed_mm_s * mult)
                duration = length / eff_feed
            segments.append(C.Segment(
                id=original.id, kind=original.kind, start_mm=start, end_mm=end,
                duration_s=duration, feed_mm_s=eff_feed, tag=original.tag,
                source_line=original.source_line,
            ))
        if self.motion.timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED and segments:
            durations = C.blended_segment_durations(
                segments, self.task.limits.rapid_mm_s, self.task.limits.max_accel_mm_s2
            )
            segments = [
                C.Segment(
                    id=seg.id, kind=seg.kind, start_mm=seg.start_mm, end_mm=seg.end_mm,
                    duration_s=dur, feed_mm_s=seg.feed_mm_s, tag=seg.tag,
                    source_line=seg.source_line,
                )
                for seg, dur in zip(segments, durations)
            ]
        plan = C.MotionPlan(
            frame=C.Frame.MACHINE, initial_state=self.task.initial_state, segments=tuple(segments),
            timing_rule=self.motion.timing_rule, provenance="stateless_candidate_objective",
        )
        plan.validate_for_task(self.task)
        return plan

    def _hard_time_and_mrr_metrics(
        self, theta: Any
    ) -> tuple[float, float, float, float, float, float, float]:
        """Use exact motion durations with sharp per-segment engagement, MRR, and rapid-stock check."""
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
        is_rapid = np.asarray(
            [segment.kind == C.SegmentKind.RAPID for segment in motion.segments], dtype=bool
        )
        rapid_stock_overlap = float(np.sum(engagement[is_rapid & ~air])) if steps else 0.0
        stock_box_mm3 = float(self.sim.Lx * self.sim.Ly * self.sim.Lz)
        vol_eng_mm3 = engagement * stock_box_mm3 * (~is_rapid).astype(np.float64)
        mrr_per_seg = vol_eng_mm3 / np.maximum(durations, 1e-6)
        peak_mrr_mm3_s = float(np.max(mrr_per_seg)) if steps else 0.0
        mean_mrr_mm3_s = float(np.sum(vol_eng_mm3) / max(cycle_time_s, 1e-9)) if steps else 0.0
        if self.spec.mrr_weight > 0 and steps:
            mrr_excess = np.maximum(0.0, mrr_per_seg / self.spec.mrr_limit_mm3_s - 1.0)
            mrr_term = float(self.spec.mrr_weight * np.sum(mrr_excess ** 2))
        else:
            mrr_term = 0.0
        return (
            cycle_time_s, air_time_s, air_fraction,
            peak_mrr_mm3_s, mean_mrr_mm3_s, mrr_term, rapid_stock_overlap,
        )

    def _hard_time_metrics(self, theta: Any) -> tuple[float, float, float]:
        cycle_time_s, air_time_s, air_fraction, *_ = self._hard_time_and_mrr_metrics(theta)
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

    def _gradient(
        self, normalized_gradient: np.ndarray,
        feed_mult_gradient: Optional[np.ndarray] = None,
        theta: Any = None,
    ) -> np.ndarray:
        physical = normalized_gradient[:self.parameterization.num_steps] / self.dimensions[None, :]
        if hasattr(self.parameterization, "pullback_joint") and feed_mult_gradient is not None:
            g_mult = feed_mult_gradient[:self.parameterization.num_steps]
            return np.asarray(
                self.parameterization.pullback_joint(physical, g_mult, theta), dtype=np.float64
            )
        return np.asarray(self.parameterization.pullback(physical), dtype=np.float64)

    def _load_candidate_inputs(self, normalized: np.ndarray, feed_mult: np.ndarray) -> None:
        steps = self.parameterization.num_steps
        self.sim.loss_shift[None] = self._loss_shift_voxels()
        self.sim.tool_delta.from_numpy(simulator_array(self.sim, normalized))
        if hasattr(self.sim, "seg_feed_mult") and self._use_per_seg_speeds():
            full_mult = np.ones(self.sim.max_steps, dtype=np.float64)
            full_mult[:steps] = feed_mult
            self.sim.seg_feed_mult.from_numpy(simulator_array(self.sim, full_mult))

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
        gradient: Optional[np.ndarray] = None
        grad_deltas_mm: Optional[np.ndarray] = None
        grad_feed_mult: Optional[np.ndarray] = None
        try:
            normalized = self._deltas(theta)
            feed_mult = self._feed_multipliers(theta)
            self._configure()
            reset_simulator(self.sim)
            # Official reset preserves configured fields. Apply this explicitly
            # once more for any alternative reset implementation, not ti.init.
            self._load_candidate_inputs(normalized, feed_mult)
            if objective_kind == "smooth":
                forward_calls += 1
                if with_gradient:
                    backward_calls = 1
                    with ti.ad.Tape(loss=self.sim.loss):
                        self.sim.forward(steps + 1)
                    normalized_gradient = self.sim.tool_delta.grad.to_numpy().copy()
                    grad_deltas_mm = (normalized_gradient[:steps] / self.dimensions[None, :]).astype(np.float64)
                    if hasattr(self.sim, "seg_feed_mult") and self._use_per_seg_speeds():
                        grad_feed_mult = self.sim.seg_feed_mult.grad.to_numpy().copy()[:steps].astype(np.float64)
                    gradient = self._gradient(normalized_gradient, grad_feed_mult, theta)
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
                    "final_stock": (
                        self.sim.stock_slice_numpy(steps)
                        if hasattr(self.sim, "stock_slice_numpy")
                        else self.sim.stock.to_numpy()[steps].copy()
                    ),
                    "time_source_is_simulator_surrogate": 1.0,
                }
                if grad_deltas_mm is not None:
                    diagnostics["grad_deltas_mm"] = grad_deltas_mm
                if grad_feed_mult is not None:
                    diagnostics["grad_feed_mult"] = grad_feed_mult
                native_metrics = spec.time_weight > 0 or spec.air_time_weight > 0 or spec.mrr_weight > 0
                native_metrics = (native_metrics or spec.legacy_weights.w_air_late > 0
                                  or spec.legacy_weights.w_break > 0)
                diagnostics["efficiency_timing_evaluated"] = float(native_metrics)
                if native_metrics:
                    native_cycle, native_air, native_fraction = self._native_time_metrics()
                    diagnostics.update(cycle_time_s=native_cycle, air_time_s=native_air,
                                       air_fraction=native_fraction)
            else:
                if with_gradient:
                    # Straight-Through Estimator (STE): analytical smooth Jacobian + exact hard forward.
                    forward_calls += 1
                    backward_calls = 1
                    with ti.ad.Tape(loss=self.sim.loss):
                        self.sim.forward(steps + 1)
                    normalized_gradient = self.sim.tool_delta.grad.to_numpy().copy()
                    grad_deltas_mm = (normalized_gradient[:steps] / self.dimensions[None, :]).astype(np.float64)
                    if hasattr(self.sim, "seg_feed_mult") and self._use_per_seg_speeds():
                        grad_feed_mult = self.sim.seg_feed_mult.grad.to_numpy().copy()[:steps].astype(np.float64)
                    gradient = self._gradient(normalized_gradient, grad_feed_mult, theta)
                    if not np.all(np.isfinite(gradient)):
                        raise FloatingPointError("non-finite STE candidate gradient")
                    self._configure()
                    reset_simulator(self.sim)
                    self._load_candidate_inputs(normalized, feed_mult)
                forward_calls += 1
                self.sim.forward_hard(steps + 1, clip_speeds=False)
                target = self.sim.target.to_numpy() < 0
                if hasattr(self.sim, "stock_slice_numpy"):
                    initial = self.sim.stock_slice_numpy(0) < 0
                    stock = self.sim.stock_slice_numpy(steps) < 0
                else:
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
                (cycle_time_s, air_time_s, air_fraction,
                 peak_mrr_mm3_s, mean_mrr_mm3_s, mrr_term, rapid_stock_overlap) = (
                    self._hard_time_and_mrr_metrics(theta)
                )
                rapid_term = self.spec.rapid_stock_weight * (rapid_stock_overlap ** 2)
                reference_s = self.reference_cycle_time_s
                if reference_s > 0:
                    time_term = self.spec.time_weight * cycle_time_s / reference_s
                    air_term = self.spec.air_time_weight * air_time_s / reference_s
                else:
                    time_term = air_term = 0.0
                objective = geometry + safety + time_term + air_term + mrr_term + rapid_term
                voxel_volume_mm3 = self.task.grid.voxel_mm ** 3
                gouge_limit = self.task.tolerances.gouge_volume_mm3 / (in_target * voxel_volume_mm3)
                residual_limit = self.task.tolerances.completion_residual_mm3 / (
                    removal_target * voxel_volume_mm3)
                mrr_violation = (
                    max(0.0, peak_mrr_mm3_s / self.spec.mrr_limit_mm3_s - 1.0)
                    if self.spec.enforce_mrr_constraint else 0.0
                )
                constraint_violation = (holder_fraction + rapid_stock_overlap + mrr_violation
                                        + max(0.0, gouge_fraction - gouge_limit)
                                        + max(0.0, residual_fraction - residual_limit))
                feasible = float(holder_fraction == 0
                                 and gouge_fraction <= gouge_limit
                                 and residual_fraction <= residual_limit)
                diagnostics = {
                    "gouge_fraction": gouge_fraction,
                    "residual_fraction": residual_fraction,
                    "holder_overlap_fraction": holder_fraction,
                    "rapid_stock_overlap": rapid_stock_overlap,
                    "peak_mrr_mm3_s": peak_mrr_mm3_s,
                    "mean_mrr_mm3_s": mean_mrr_mm3_s,
                    "mrr_term": mrr_term,
                    "geometry": geometry, "holder_proxy": safety, "final_stock": stock.copy(),
                    "cycle_time_s": cycle_time_s, "air_time_s": air_time_s,
                    "air_fraction": air_fraction, "efficiency": objective,
                    "efficiency_time_term": time_term, "efficiency_air_time_term": air_term,
                    "gouge_fraction_limit": gouge_limit,
                    "residual_fraction_limit": residual_limit,
                    "constraint_violation": constraint_violation, "feasible": feasible,
                    "time_source_is_simulator_surrogate": 0.0,
                }
                if grad_deltas_mm is not None:
                    diagnostics["grad_deltas_mm"] = grad_deltas_mm
                if grad_feed_mult is not None:
                    diagnostics["grad_feed_mult"] = grad_feed_mult
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
                if name not in ("final_stock", "grad_deltas_mm", "grad_feed_mult"):
                    components[name] = float(value)
            result = C.CandidateResult(
                candidate_id=candidate_id, theta_hash=_theta_hash(theta),
                objective_kind=objective_kind, objective=objective, valid=True,
                coverage=C.Coverage.INCOMPLETE, components=components,
                gradient_available=with_gradient,
            )
            valid = True
            return CandidateEvaluation(result, gradient if with_gradient else None, diagnostics)
        except (ValueError, FloatingPointError) as error:
            return self._invalid(candidate_id, theta, objective_kind, str(error))
        finally:
            entry = self.ledger.finish_candidate(valid=valid, forward_calls=forward_calls,
                                                 backward_calls=backward_calls,
                                                 segment_steps=steps * forward_calls,
                                                 voxel_steps=voxels * forward_calls)
            entry["loss_shift_voxels"] = self._loss_shift_voxels()
            entry["loss_shift_mm"] = self._loss_shift_mm()

    def evaluate_candidate(self, theta: Any, candidate_id: str,
                           objective_kind: str = "smooth") -> C.CandidateResult:
        return self._evaluate(theta, candidate_id, objective_kind, with_gradient=False).result

    def evaluate_candidate_with_gradient(self, theta: Any, candidate_id: str) -> CandidateEvaluation:
        return self._evaluate(theta, candidate_id, "smooth", with_gradient=True)

    def evaluate_ste_with_gradient(self, theta: Any, candidate_id: str) -> CandidateEvaluation:
        """Straight-Through Estimator: exact hard CandidateResult paired with smooth AD gradient."""
        return self._evaluate(theta, candidate_id, "hard", with_gradient=True)
