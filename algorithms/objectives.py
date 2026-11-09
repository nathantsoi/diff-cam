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
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import taichi as ti

from benchmarks import contracts as C
from .run_artifacts import ComputeLedger


@dataclass(frozen=True)
class ObjectiveSpec:
    k: float = 40.0
    gouge_weight: float = 2.0
    residual_weight: float = 1.0
    holder_weight: float = 50.0
    holder_margin_mm: float = 0.0
    geometry_normalization: str = "1/N followed by squared per-voxel occupancy penalties"
    hard_geometry_normalization: str = "w_gouge*gouge_fraction^2 + w_residual*residual_fraction^2"
    hard_holder_normalization: str = "w_holder*hard_holder_overlap_fraction^2 (proxy)"
    prototype: str = "analytic_target_until_T2.0_custom_target_and_T1.2_chronology"

    def __post_init__(self) -> None:
        for name in ("k", "gouge_weight", "residual_weight", "holder_weight", "holder_margin_mm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class CandidateEvaluation:
    result: C.CandidateResult
    gradient_theta: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


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
                 ledger: Optional[ComputeLedger] = None, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        motion.validate_for_task(task)
        self.task, self.motion = task, motion
        if motion.frame != C.Frame.MACHINE or motion.timing_rule != C.TimingRule.EXACT_STOP_CONSTANT_SPEED:
            raise ValueError("prototype requires machine-frame exact-stop typed motion")
        self.parameterization = parameterization
        self.sim = simulator
        self.spec = spec or ObjectiveSpec()
        self._evaluation_k_override: Optional[float] = None
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
        kind = task.original_part.kind
        primitive = task.original_part.params.get("primitive") if kind == "analytic" else kind
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

    def _configure(self) -> None:
        spec, sim = self.spec, self.sim
        sim.k[None] = spec.k if self._evaluation_k_override is None else self._evaluation_k_override
        sim.w_gouge[None] = spec.gouge_weight
        sim.w_residual[None] = spec.residual_weight
        sim.holder_penalty_weight[None] = spec.holder_weight
        sim.holder_margin[None] = spec.holder_margin_mm / sim.v
        for name in ("w_air", "w_jerk", "w_step", "w_len", "w_air_time", "w_air_late",
                     "w_prox", "w_traj_prox", "w_break", "w_time", "w_tool_gouge"):
            getattr(sim, name)[None] = 0.0
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

    def _invalid(self, candidate_id: str, theta: Any, kind: str, reason: str) -> CandidateEvaluation:
        return CandidateEvaluation(C.CandidateResult(
            candidate_id=candidate_id, theta_hash=_theta_hash(theta), objective_kind=kind,
            objective=None, valid=False, invalid_reason=reason,
            coverage=C.Coverage.UNKNOWN,
        ))

    def _gradient(self, normalized_gradient: np.ndarray) -> np.ndarray:
        physical = normalized_gradient[:self.parameterization.num_steps] / self.dimensions[None, :]
        return np.asarray(self.parameterization.pullback(physical), dtype=np.float64)

    def _evaluate(self, theta: Any, candidate_id: str, objective_kind: str,
                  with_gradient: bool) -> CandidateEvaluation:
        if objective_kind not in ("smooth", "hard"):
            raise ValueError("objective_kind must be smooth or hard")
        self.ledger.start_candidate(candidate_id, objective_kind)
        forward_calls = backward_calls = 0
        steps = self.parameterization.num_steps
        voxels = steps * self.sim.Nx * self.sim.Ny * self.sim.Nz
        valid = False
        try:
            normalized = self._deltas(theta)
            self._configure()
            reset_simulator(self.sim)
            self.sim.tool_delta.from_numpy(normalized.astype(np.float32))
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
                }
            else:
                self.sim.forward_hard(steps + 1, clip_speeds=False)
                target = self.sim.target.to_numpy() < 0
                stock = self.sim.stock.to_numpy()[steps] < 0
                n = stock.size
                gouge_fraction = float(np.count_nonzero(target & ~stock) / n)
                residual_fraction = float(np.count_nonzero(~target & stock) / n)
                holder_fraction = float(self.sim.holder_overlap_total(steps))
                geometry = (self.spec.gouge_weight * gouge_fraction ** 2
                            + self.spec.residual_weight * residual_fraction ** 2)
                safety = self.spec.holder_weight * holder_fraction ** 2
                objective = geometry + safety
                diagnostics = {
                    "gouge_fraction": gouge_fraction,
                    "residual_fraction": residual_fraction,
                    "holder_overlap_fraction": holder_fraction,
                    "geometry": geometry, "holder_proxy": safety, "final_stock": stock.copy(),
                }
            if not math.isfinite(objective):
                raise FloatingPointError("simulator returned a non-finite objective")
            components = {
                "objective_value": objective,
                "delta_l2_normalized": float(np.linalg.norm(normalized)),
                "speed_clipping_applied": 0.0,
                "evaluation_seed": float(self.seed),
                "prototype_incomplete_final_target_and_posted_coverage": 1.0,
            }
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
            self.ledger.finish_candidate(valid=valid, forward_calls=forward_calls,
                                         backward_calls=backward_calls,
                                         segment_steps=steps if forward_calls else 0,
                                         voxel_steps=voxels if forward_calls else 0)

    def evaluate_candidate(self, theta: Any, candidate_id: str,
                           objective_kind: str = "smooth") -> C.CandidateResult:
        return self._evaluate(theta, candidate_id, objective_kind, with_gradient=False).result

    def evaluate_candidate_with_gradient(self, theta: Any, candidate_id: str) -> CandidateEvaluation:
        return self._evaluate(theta, candidate_id, "smooth", with_gradient=True)
