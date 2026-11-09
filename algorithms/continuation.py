"""Experimental signed calibration and constrained backtracking (T3.4).

This module is deliberately not a trust-region method: no trust-region radius or
local quadratic model is implemented. It separates two mechanisms:

* ``SignedLossCalibrator`` measures the median per-voxel signed-stock-field
  difference ``phi_hard - phi_soft`` in VOXEL SDF units on an identical
  executed path/grid, in a physical band around the operation-target surface.
  ``phi_hard`` is a signed EDT of hard remaining-stock membership. Soft phi is
  the simulator's ``stock_d`` BEFORE the occupancy-only shift. Scalar loss
  components are never substituted for these signed spatial fields. The
  resulting signed bounded EMA shifts the smooth iso-level, not the objective.
* ``BacktrackingMechanism`` evaluates a proposed optimizer state against its
  incumbent under identical k, shift, grid and hard-validator identity. It
  enforces explicit constraints, retries with bounded learning rates, and
  restores theta/moments/config on rejection. Its score term is raw, unshifted
  HARD validation objective: no scalar loss_shift is added or compared.

It is an adapter contract. The executor must expose actual executed paths and
stock fields for calibration: equal theta values alone are not evidence of
identical executed motion. Hard coverage, geometry and safety availability are
kept distinct from a known infeasible result. Unknown constraint state cannot
win an acceptance comparison.

Expected integration callbacks are ``soft(theta, context, require_gradient)``
and ``hard(theta, context)``. One such call to each consumes one
candidate-objective budget and one hard-validation budget respectively. When a
fresh incumbent measurement is needed (changed epoch or no cached gradient),
that fresh measurement has its own explicit cost in the same ledgers. The
optimizer owner should preserve the executor's independent-forward/reset
semantics and restore its own execution configuration on rollback; this module
never initializes Taichi or mutates a simulator directly.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from algorithms.blackbox import OptimizationBudget
from benchmarks import contracts as C


def _frozen_array(value: Any, name: str, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    result = np.array(array, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _optional_number(value: Any, name: str, *, unit: str = "", nonnegative: bool = False):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric{(' in ' + unit) if unit else ''}, not boolean")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite{(' in ' + unit) if unit else ''}" +
                         (" and non-negative" if nonnegative else ""))
    return result


def _optional_fraction(value: Any, name: str):
    result = _optional_number(value, name, unit="fraction [0,1]", nonnegative=True)
    if result is not None and result > 1:
        raise ValueError(f"{name} must be in [0,1] under this adapter contract")
    return result


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    # Also proves JSON representability, rejecting NaN/Inf rather than later
    # serializing an incomparable or non-finite optimizer configuration.
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def parameter_hash(theta: Any) -> str:
    return C.content_hash(_frozen_array(theta, "theta").tolist())


def grid_signature(grid: C.GridSpec) -> str:
    return C.content_hash({
        "frame": grid.frame.value, "origin_mm": list(grid.origin_mm),
        "shape": list(grid.shape), "sign_convention": grid.sign_convention,
        "voxel_mm": grid.voxel_mm,
    })


@dataclass(frozen=True)
class EvaluationContext:
    """Exact same-state identity; no approximate k/shift equality is accepted."""

    grid: C.GridSpec
    k: float
    loss_shift: float
    validator_id: str
    # Revision and all settings that can change hard validation. Callers should
    # include the validator revision, not only a generic evaluator class name.
    validator_settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.k) or self.k <= 0:
            raise ValueError("k must be finite and positive")
        if not math.isfinite(self.loss_shift) or abs(self.loss_shift) > 0.5:
            raise ValueError("loss_shift must be finite in voxel units with |shift| <= 0.5 voxel")
        if not isinstance(self.validator_id, str) or not self.validator_id.strip():
            raise ValueError("hard-validator implementation/revision id is required")
        if not isinstance(self.grid, C.GridSpec):
            raise TypeError("evaluation context requires the actual physical GridSpec")
        object.__setattr__(self, "validator_settings", _json_mapping(
            self.validator_settings, "validator_settings"
        ))

    def as_dict(self):
        return {
            "k": self.k, "loss_shift": self.loss_shift,
            "grid_sha256": grid_signature(self.grid),
            "validator_id": self.validator_id,
            "validator_settings": dict(self.validator_settings),
            "validator_config_sha256": C.content_hash(self.validator_settings),
        }


@dataclass(frozen=True)
class SharpnessSchedule:
    """Independent fixed/bounded-annealed k schedule; never EMA-driven.

    Indices after the declared horizon retain the endpoint. This is a schedule
    helper only; it does not claim to invoke or replace the optimizer owner's
    actual Adam scheduler. Every candidate/epoch in a backtracking trial uses
    the one context returned for that trial, with no k change during retries.
    """
    mode: str
    k_start: float
    k_end: float
    n_steps: int

    def __post_init__(self):
        if self.mode not in ("fixed", "annealed"):
            raise ValueError("sharpness mode must be fixed or annealed")
        if not math.isfinite(self.k_start) or not math.isfinite(self.k_end):
            raise ValueError("sharpness endpoints must be finite")
        if self.k_start <= 0 or self.k_end <= 0:
            raise ValueError("sharpness endpoints must be positive")
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise ValueError("sharpness schedule requires a positive integer horizon")
        if self.mode == "fixed" and self.k_start != self.k_end:
            raise ValueError("fixed sharpness requires equal endpoints")

    @property
    def bounds(self):
        return min(self.k_start, self.k_end), max(self.k_start, self.k_end)

    def at(self, step: int) -> float:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("schedule index must be a nonnegative integer")
        fraction = 1.0 if self.n_steps == 1 else min(1.0, step / (self.n_steps - 1))
        return self.k_start + (self.k_end - self.k_start) * fraction

    def context_at(self, step: int, grid: C.GridSpec, loss_shift: float,
                   validator_id: str, validator_settings: Optional[Mapping[str, Any]] = None):
        return EvaluationContext(grid, self.at(step), loss_shift, validator_id,
                                 {} if validator_settings is None else validator_settings)


@dataclass(frozen=True)
class SoftObservation:
    """Scalar loss diagnostic, not a stock-distance field for calibration."""
    context: EvaluationContext
    raw_phi_soft: Optional[float]
    path: Optional[np.ndarray] = None
    gradient: Optional[np.ndarray] = None
    theta_hash: Optional[str] = None
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "raw_phi_soft", _optional_number(
            self.raw_phi_soft, "raw_phi_soft"
        ))
        if self.path is not None:
            object.__setattr__(self, "path", _frozen_array(self.path, "executed soft path", ndim=2))
        if self.gradient is not None:
            object.__setattr__(self, "gradient", _frozen_array(self.gradient, "soft gradient"))


@dataclass(frozen=True)
class HardObservation:
    """One reported hard evaluation; None remains unknown rather than becoming 0."""
    context: EvaluationContext
    coverage_complete: bool
    gouge_fraction: Optional[float] = None
    residual_fraction: Optional[float] = None
    local_surface_error_mm: Optional[float] = None
    holder_overlap_fraction: Optional[float] = None
    safety_pass: Optional[bool] = None
    raw_phi_hard: Optional[float] = None
    path: Optional[np.ndarray] = None
    theta_hash: Optional[str] = None
    source_valid: bool = True
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.coverage_complete, (bool, np.bool_)):
            raise ValueError("coverage_complete must be an explicit boolean")
        if not isinstance(self.source_valid, (bool, np.bool_)):
            raise ValueError("source_valid must be an explicit boolean")
        if self.safety_pass is not None and not isinstance(self.safety_pass, (bool, np.bool_)):
            raise ValueError("safety_pass must be true, false or None")
        object.__setattr__(self, "coverage_complete", bool(self.coverage_complete))
        object.__setattr__(self, "source_valid", bool(self.source_valid))
        if self.safety_pass is not None:
            object.__setattr__(self, "safety_pass", bool(self.safety_pass))
        for name in ("gouge_fraction", "residual_fraction", "holder_overlap_fraction"):
            object.__setattr__(self, name, _optional_fraction(getattr(self, name), name))
        object.__setattr__(self, "local_surface_error_mm", _optional_number(
            self.local_surface_error_mm, "local_surface_error_mm", unit="mm", nonnegative=True,
        ))
        object.__setattr__(self, "raw_phi_hard", _optional_number(
            self.raw_phi_hard, "raw_phi_hard"
        ))
        if self.path is not None:
            object.__setattr__(self, "path", _frozen_array(self.path, "executed hard path", ndim=2))


@dataclass(frozen=True)
class PairedObservation:
    """Hard/raw-soft measurement for ONE parameter point on one context."""
    soft: SoftObservation
    hard: HardObservation

    def __post_init__(self):
        object.__setattr__(self, "context", self.soft.context)
        if self.soft.context != self.hard.context:
            raise _ContextMismatch("soft and hard measurements used different k/shift/grid/validator")
        if self.soft.path is None or self.hard.path is None:
            object.__setattr__(self, "identical_executed_path", None)
        else:
            object.__setattr__(self, "identical_executed_path", bool(
                self.soft.path.shape == self.hard.path.shape
                and np.array_equal(self.soft.path, self.hard.path)
            ))

    context: EvaluationContext = field(init=False)
    identical_executed_path: Optional[bool] = field(init=False, default=None)

    @property
    def scalar_disagreement_known(self) -> bool:
        return self.soft.raw_phi_soft is not None and self.hard.raw_phi_hard is not None


@dataclass(frozen=True)
class SurfaceBand:
    """An inclusive physical band around the operation-target surface.

    ``gouge`` selects remaining hard stock inside operation target T;
    ``residual`` selects remaining hard stock outside T (the removal target).
    ``stock`` includes both regions. Distances are unsigned EDT distances to the
    opposite target-membership region on the declared cell-centre grid, not an
    exact continuous CAD Hausdorff distance or a signed Dice gap.
    """
    region: str
    minimum_mm: float
    maximum_mm: float
    rationale: str

    def __post_init__(self):
        if self.region not in ("gouge", "residual", "stock"):
            raise ValueError("surface band region must be gouge, residual or stock")
        if not self.rationale.strip():
            raise ValueError("physical surface-band rationale is required")
        for name in ("minimum_mm", "maximum_mm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_mm > self.maximum_mm:
            raise ValueError("surface-band millimetre bounds must be ordered")


def _frozen_mask(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype(bool):
        raise TypeError(f"{name} requires a dtype=bool membership array; indicators are not signed distances")
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} does not match declared grid {shape}")
    result = result.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class StockFieldPair:
    """One paired execution with a true hard signed EDT and an unshifted soft field.

    Soft phi is supplied from the raw simulator stock field after forward
    execution, before apply-cut loss occupancy shifts. Hard phi is always
    derived here from its explicit Boolean remaining-stock mask. This prevents
    the old error of treating scalar weighted losses or +/-1 indicators as phi.
    Paths are compared exactly, in the coordinates returned by the executor.
    """
    context: EvaluationContext
    hard_remaining_stock: np.ndarray
    soft_stock_sdf: np.ndarray
    operation_target: np.ndarray
    hard_path: np.ndarray
    soft_path: np.ndarray
    hard_phi_voxels: np.ndarray = field(init=False)
    soft_phi_voxels: np.ndarray = field(init=False)
    surface_distance_mm: np.ndarray = field(init=False)
    hard_distance_definition: str = field(init=False)
    soft_field_definition: str = field(init=False)
    identical_executed_path: bool = field(init=False)

    def __post_init__(self):
        if not isinstance(self.context, EvaluationContext):
            raise TypeError("field calibration requires physical evaluation context")
        grid = self.context.grid
        if grid.sign_convention != "negative_inside":
            raise ValueError("stock-field calibration expects negative_inside stock convention")
        hard_path = _frozen_array(self.hard_path, "executed hard path", ndim=2)
        soft_path = _frozen_array(self.soft_path, "executed soft path", ndim=2)
        if hard_path.shape != soft_path.shape or hard_path.shape[1] != 3:
            raise _ContextMismatch("hard/soft executed path dimensions differ")
        if not np.array_equal(hard_path, soft_path):
            raise _ContextMismatch("hard/soft executed paths are not identical")
        remaining = _frozen_mask(self.hard_remaining_stock, "hard remaining stock", grid.shape)
        target = _frozen_mask(self.operation_target, "operation target", grid.shape)
        if not remaining.any():
            raise ValueError("no remaining hard stock in represented grid: finite hard EDT unavailable")
        if remaining.all():
            raise ValueError("no exterior hard-stock cell in represented grid: finite hard EDT unavailable")
        if not target.any() or target.all():
            raise ValueError("operation-target band requires both target and outside region on the grid")
        soft = _frozen_array(self.soft_stock_sdf, "soft stock signed field")
        if soft.shape != grid.shape:
            raise ValueError("soft stock field shape does not match declared grid")
        hard_distance = distance_transform_edt(~remaining) - distance_transform_edt(remaining)
        # abs(unsigned region distance) is the distance to the closest target
        # boundary on this cell-centre grid. min(EDT(~T), EDT(T)) would be zero
        # on the outside and would not measure distance to the target surface.
        surface_distance = np.abs(
            distance_transform_edt(~target) - distance_transform_edt(target)
        )
        if not np.all(np.isfinite(hard_distance)) or not np.all(np.isfinite(surface_distance)):
            raise ValueError("non-finite field distance on the declared representation")
        hard_distance.flags.writeable = False
        surface_distance = surface_distance * grid.voxel_mm
        surface_distance.flags.writeable = False
        object.__setattr__(self, "hard_remaining_stock", remaining)
        object.__setattr__(self, "operation_target", target)
        object.__setattr__(self, "soft_stock_sdf", soft)
        object.__setattr__(self, "hard_path", hard_path)
        object.__setattr__(self, "soft_path", soft_path)
        object.__setattr__(self, "hard_phi_voxels", hard_distance)
        object.__setattr__(self, "soft_phi_voxels", soft)
        object.__setattr__(self, "surface_distance_mm", surface_distance)
        object.__setattr__(
            self, "hard_distance_definition",
            "signed_edt_voxels: distance_transform_edt(~remaining_stock) - distance_transform_edt(remaining_stock)",
        )
        object.__setattr__(
            self, "soft_field_definition",
            "simulator.stock_d_before_terminal_sigmoid/stock_d_shift_in_voxels",
        )
        object.__setattr__(self, "identical_executed_path", True)

    def selection(self, band: SurfaceBand) -> np.ndarray:
        if band.region == "gouge":
            region = self.operation_target & self.hard_remaining_stock
        elif band.region == "residual":
            region = ~self.operation_target & self.hard_remaining_stock
        elif band.region == "stock":
            region = self.hard_remaining_stock
        else:
            raise ValueError("unknown physical band region")
        return (
            region
            & (self.surface_distance_mm >= band.minimum_mm)
            & (self.surface_distance_mm <= band.maximum_mm)
        )


@dataclass(frozen=True)
class ConstraintPolicy:
    max_gouge_fraction: float = 0.0
    max_residual_fraction: float = 1.0
    max_local_surface_error_mm: float = 0.5
    max_holder_overlap_fraction: float = 0.0
    fraction_atol: float = 1e-7
    local_surface_error_atol_mm: float = 1e-6
    objective_atol: float = 1e-8
    objective_rtol: float = 1e-6

    def __post_init__(self):
        for name in ("max_gouge_fraction", "max_residual_fraction", "max_holder_overlap_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("max_local_surface_error_mm", "fraction_atol", "local_surface_error_atol_mm",
                     "objective_atol", "objective_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class Feasibility(str, Enum):
    UNKNOWN = "unknown"
    INFEASIBLE = "infeasible"
    FEASIBLE = "feasible"


@dataclass(frozen=True)
class ConstraintAssessment:
    status: Feasibility
    violation: float
    missing: tuple[str, ...] = ()


def assess_constraints(state: HardObservation, policy: ConstraintPolicy) -> ConstraintAssessment:
    missing = []
    if not state.source_valid:
        missing.append("source_invalid")
    if not state.coverage_complete:
        missing.append("coverage_incomplete")
    for name in ("gouge_fraction", "residual_fraction", "local_surface_error_mm",
                 "holder_overlap_fraction", "safety_pass"):
        if getattr(state, name) is None:
            missing.append(name + "_unavailable")
    if missing:
        return ConstraintAssessment(Feasibility.UNKNOWN, 0.0, tuple(missing))

    gouge_violation = max(0.0, state.gouge_fraction - policy.max_gouge_fraction
                          - policy.fraction_atol)
    residual_violation = max(0.0, state.residual_fraction - policy.max_residual_fraction
                             - policy.fraction_atol)
    local_violation = max(0.0, state.local_surface_error_mm - policy.max_local_surface_error_mm
                          - policy.local_surface_error_atol_mm)
    # Units are dimensionless by construction: fractions use a 1.0 declared
    # scale, and local surface error is scaled by the larger of 1 mm and its
    # explicitly configured limit. There is no division by a zero Dice gap.
    local_scaled = local_violation / max(1.0, policy.max_local_surface_error_mm)
    holder_violation = max(0.0, state.holder_overlap_fraction - policy.max_holder_overlap_fraction
                           - policy.fraction_atol)
    safety_violation = 0.0 if state.safety_pass else 1.0
    violation = gouge_violation + residual_violation + local_scaled + holder_violation + safety_violation
    status = Feasibility.FEASIBLE if violation == 0.0 else Feasibility.INFEASIBLE
    return ConstraintAssessment(status, violation)


@dataclass(frozen=True)
class CalibrationResult:
    shift: float
    updated: bool
    reason: str
    # selected_count is the number of selected VOXELS across all measurements;
    # selected_observations records actual paired path/field measurements.
    selected_count: int = 0
    selected_observations: int = 0
    source_count: int = 0
    median_delta: Optional[float] = None
    clipped_delta: Optional[float] = None
    median_hard_voxels: Optional[float] = None
    median_soft_voxels: Optional[float] = None
    delta_mm: Optional[float] = None
    clipped_delta_mm: Optional[float] = None
    shift_mm: Optional[float] = None
    disagreement: Mapping[str, Any] = field(default_factory=dict)
    band: Mapping[str, Any] = field(default_factory=dict)
    hard_distance_definition: str = ""
    soft_field_definition: str = ""
    voxel_mm: Optional[float] = None


def bounded_disagreement(hard_phi: float, soft_phi: float, scale_floor: float) -> dict[str, Any]:
    """Return a signed gap bounded to [-1,1], with a non-geometric floor.

    The denominator is |hard|+|soft| plus an explicitly positive *voxel-unit*
    floor. It is not an IoU/Dice denominator, a signed Dice gap, or a zero mask
    denominator. |hard-soft| <= |hard|+|soft| also bounds nonzero gap ratios at
    nonzero losses. Scaling by the largest magnitude avoids overflow in this
    normalization, including very large sparse synthetic values.
    """
    hard = _optional_number(hard_phi, "hard_phi")
    soft = _optional_number(soft_phi, "soft_phi")
    scale_floor = _optional_number(scale_floor, "scale_floor", nonnegative=True)
    if scale_floor <= 0:
        raise ValueError("scale_floor must be a positive voxel-SDF-unit floor, not zero")
    scale = max(abs(hard), abs(soft), scale_floor)
    gap = hard - soft
    bounded = (hard / scale - soft / scale) / (
        abs(hard) / scale + abs(soft) / scale + scale_floor / scale
    )
    bounded = min(1.0, max(-1.0, bounded))
    return {
        "raw_gap": gap if math.isfinite(gap) else None,
        "bounded_gap": bounded,
        "absolute_bounded_gap": abs(bounded),
        "scale_floor": scale_floor,
        "normalizer": "abs(hard_phi) + abs(soft_phi) + explicit_positive_voxel_sdf_unit_scale_floor",
    }


FieldObservation = StockFieldPair
PhysicalBand = SurfaceBand


class SignedLossCalibrator:
    """Calibrate stock SDF offsets in voxel units, never scalar loss offsets.

    This keeps the historical class name for adapters, but phi_hard/soft are
    spatial signed fields. The name ``SignedFieldCalibrator`` is preferred.
    """

    def __init__(self, eta: float, delta_max: float, disagreement_floor: float):
        if not math.isfinite(eta) or not 0 < eta <= 1:
            raise ValueError("eta must be in (0,1]")
        if not math.isfinite(delta_max) or not 0 < delta_max <= 0.5:
            raise ValueError("delta_max must be finite, positive and <= 0.5 voxel")
        if not math.isfinite(disagreement_floor) or disagreement_floor <= 0:
            raise ValueError("disagreement_floor must be finite and positive in voxel units")
        self.eta, self.delta_max, self.floor = eta, delta_max, disagreement_floor

    def _unchanged(self, current_shift: float, reason: str, count: int) -> CalibrationResult:
        if not math.isfinite(current_shift):
            raise ValueError("current shift must be finite")
        return CalibrationResult(current_shift, False, reason, source_count=count)

    def update(self, observations: Sequence[StockFieldPair], band: SurfaceBand,
               current_shift: float) -> CalibrationResult:
        observations = tuple(observations)
        if not observations:
            return self._unchanged(current_shift, "no_measurements", 0)
        if any(not isinstance(item, StockFieldPair) for item in observations):
            raise TypeError("field calibration requires StockFieldPair records; scalar losses are not phi")
        if not isinstance(band, SurfaceBand):
            raise TypeError("field calibration requires a physical SurfaceBand")
        contexts = [item.context for item in observations]
        first = contexts[0]
        if any(context != first for context in contexts[1:]):
            raise ValueError("calibration band requires identical k/shift/grid/validator context")
        if not math.isfinite(current_shift):
            raise ValueError("current shift must be finite in voxel units")

        masks = [item.selection(band) for item in observations]
        if any(not np.any(mask) for mask in masks):
            return self._unchanged(current_shift, "selected_surface_band_has_no_field_cells",
                                   len(observations))
        hard_parts = [item.hard_phi_voxels[mask] for item, mask in zip(observations, masks)]
        soft_parts = [item.soft_phi_voxels[mask] for item, mask in zip(observations, masks)]
        hard_values = np.concatenate(hard_parts)
        soft_values = np.concatenate(soft_parts)
        differences = hard_values - soft_values
        if not (np.all(np.isfinite(differences)) and np.all(np.isfinite(hard_values))
                and np.all(np.isfinite(soft_values))):
            return self._unchanged(current_shift, "nonfinite_signed_field_difference", len(observations))
        median = float(np.median(differences))
        median_hard = float(np.median(hard_values))
        median_soft = float(np.median(soft_values))
        clipped = float(np.clip(median, -self.delta_max, self.delta_max))
        new_shift = (1.0 - self.eta) * current_shift + self.eta * clipped
        if not math.isfinite(new_shift):
            raise ValueError("EMA shift update became non-finite")
        pitch = first.grid.voxel_mm
        selected_cells = sum(int(np.count_nonzero(mask)) for mask in masks)
        return CalibrationResult(
            shift=new_shift, updated=True, reason="signed_stock_field_surface_band_calibration",
            selected_count=selected_cells,
            selected_observations=len(observations), source_count=len(observations),
            median_delta=median, clipped_delta=clipped,
            median_hard_voxels=median_hard, median_soft_voxels=median_soft,
            delta_mm=median * pitch, clipped_delta_mm=clipped * pitch,
            shift_mm=new_shift * pitch,
            disagreement=bounded_disagreement(median_hard, median_soft, self.floor),
            band={
                "region": band.region, "minimum_mm": band.minimum_mm,
                "maximum_mm": band.maximum_mm,
                "minimum_voxels": band.minimum_mm / pitch,
                "maximum_voxels": band.maximum_mm / pitch,
                "rationale": band.rationale,
            },
            hard_distance_definition=observations[0].hard_distance_definition,
            soft_field_definition=observations[0].soft_field_definition,
            voxel_mm=pitch,
        )


SignedFieldCalibrator = SignedLossCalibrator


@dataclass(frozen=True)
class OptimizerSnapshot:
    """Immutable *copy* of state; no candidate optimizer fields are aliased."""
    theta: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray
    optimizer_config: Mapping[str, Any]
    context: EvaluationContext
    optimizer_step: int
    observation: Optional[PairedObservation] = None
    _config: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self):
        theta = _frozen_array(self.theta, "theta")
        first = _frozen_array(self.first_moment, "first moment")
        second = _frozen_array(self.second_moment, "second moment")
        if theta.shape != first.shape or theta.shape != second.shape:
            raise ValueError("theta and both Adam moment tensors must have the same shape")
        if np.any(second < 0):
            raise ValueError("second moments must be nonnegative")
        if isinstance(self.optimizer_step, bool) or not isinstance(self.optimizer_step, int) or self.optimizer_step < 0:
            raise ValueError("optimizer_step must be a nonnegative integer")
        config = _json_mapping(self.optimizer_config, "optimizer_config")
        rate = config.get("learning_rate")
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("optimizer config must include positive finite learning_rate")
        if "loss_shift" in config and config["loss_shift"] != self.context.loss_shift:
            raise ValueError("optimizer config and measured loss_shift context differ")
        if "k" in config and config["k"] != self.context.k:
            raise ValueError("optimizer config and measured k context differ")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "first_moment", first)
        object.__setattr__(self, "second_moment", second)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "optimizer_config", copy.deepcopy(config))

    @property
    def learning_rate(self) -> float:
        return self._config["learning_rate"]

    @property
    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    def clone(self) -> "OptimizerSnapshot":
        return OptimizerSnapshot(self.theta, self.first_moment, self.second_moment,
                                 self._config, self.context, self.optimizer_step, self.observation)

    def with_observation(self, observation: PairedObservation,
                         context: Optional[EvaluationContext] = None) -> "OptimizerSnapshot":
        return OptimizerSnapshot(self.theta, self.first_moment, self.second_moment,
                                 self._config, self.context if context is None else context,
                                 self.optimizer_step, observation)

    @classmethod
    def initial(cls, theta: Any, context: EvaluationContext,
                optimizer_config: Optional[Mapping[str, Any]] = None):
        theta = _frozen_array(theta, "theta")
        config = {"optimizer": "adam", "mode": "fixed", "learning_rate": 0.005,
                  "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
        if optimizer_config:
            config.update(_json_mapping(optimizer_config, "optimizer_config"))
        return cls(theta, np.zeros_like(theta), np.zeros_like(theta),
                   config, context, 0, None)


@dataclass(frozen=True)
class Proposal:
    theta: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray
    optimizer_config: Mapping[str, Any]
    context: EvaluationContext
    optimizer_step: int

    def __post_init__(self):
        θ = _frozen_array(self.theta, "proposal theta")
        first = _frozen_array(self.first_moment, "proposal first moment")
        second = _frozen_array(self.second_moment, "proposal second moment")
        if θ.shape != first.shape or θ.shape != second.shape:
            raise ValueError("proposed theta and moments must have the same shape")
        if np.any(second < 0):
            raise ValueError("proposed second moments must be nonnegative")
        if isinstance(self.optimizer_step, bool) or not isinstance(self.optimizer_step, int) or self.optimizer_step < 0:
            raise ValueError("proposal optimizer_step must be nonnegative")
        object.__setattr__(self, "theta", θ)
        object.__setattr__(self, "first_moment", first)
        object.__setattr__(self, "second_moment", second)
        object.__setattr__(self, "optimizer_config", _json_mapping(self.optimizer_config, "proposal config"))


@dataclass(frozen=True)
class BacktrackingConfig:
    learning_rate_min: float = 0.0000625
    learning_rate_max: float = 0.01
    learning_rate_shrink: float = 0.5
    max_retries_per_step: int = 2
    k_min: float = 1.0
    k_max: float = 1e6
    eta: float = 0.25
    delta_max: float = 0.25
    disagreement_floor: float = 1e-6

    def __post_init__(self):
        for name in ("learning_rate_min", "learning_rate_max", "learning_rate_shrink",
                     "k_min", "k_max", "eta", "delta_max", "disagreement_floor"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.learning_rate_min <= self.learning_rate_max:
            raise ValueError("learning-rate bounds must be positive and ordered")
        if not 0 < self.learning_rate_shrink < 1:
            raise ValueError("learning_rate_shrink must be in (0,1)")
        if not 0 < self.k_min <= self.k_max:
            raise ValueError("sharpness bounds must be positive and ordered")
        if isinstance(self.max_retries_per_step, bool) or not isinstance(self.max_retries_per_step, int) \
                or not 0 <= self.max_retries_per_step <= 16:
            raise ValueError("max_retries_per_step must be an integer in [0,16]")
        if not 0 < self.eta <= 1 or not 0 < self.delta_max <= 0.5 or self.disagreement_floor <= 0:
            raise ValueError("EMA rate must be in (0,1]; delta_max must be positive and <= 0.5 voxel")
        if self.disagreement_floor <= 0:
            raise ValueError("disagreement_floor must be positive in voxel units")


class _BudgetExhausted(RuntimeError):
    pass


class _ContextMismatch(ValueError):
    pass


@dataclass(frozen=True)
class StepOutcome:
    snapshot: OptimizerSnapshot
    status: str
    reason: str
    accepted: bool = False
    acceptance_kind: Optional[str] = None
    attempts: int = 0
    retries: int = 0
    spent: Mapping[str, int] = field(default_factory=dict)
    history: tuple[Mapping[str, Any], ...] = ()


class BacktrackingMechanism:
    """Evaluate safe same-epoch trials, with feasible/infeasible restoration.

    A known infeasible start can move to the least-violation state (or a feasible
    state). A feasible incumbent is never replaced by an infeasible candidate,
    even when that candidate has a much lower soft objective. Missing validation
    is unknown/blocked, not a safety discourtesy that can be optimized past.
    """

    def __init__(self, config: BacktrackingConfig, constraints: ConstraintPolicy,
                 budget: OptimizationBudget,
                 soft_evaluator: Callable[[np.ndarray, EvaluationContext, bool], SoftObservation],
                 hard_validator: Callable[[np.ndarray, EvaluationContext], HardObservation]):
        if budget.population_size is not None:
            raise ValueError("single-candidate backtracking cannot use a population budget")
        self.config, self.constraints, self.budget = config, constraints, budget
        self.soft_evaluator, self.hard_validator = soft_evaluator, hard_validator
        self._spent = {"candidate_evaluations": 0, "hard_validation_calls": 0,
                       "proposals": 0, "retries": 0, "accepted": 0}
        self.calibrator = SignedLossCalibrator(config.eta, config.delta_max,
                                               config.disagreement_floor)

    def budget_snapshot(self) -> dict[str, int]:
        return dict(self._spent)

    def _check_epoch(self, context: EvaluationContext):
        if not self.config.k_min <= context.k <= self.config.k_max:
            raise ValueError("requested sharpness is outside declared k bounds")

    def _observe(self, theta: np.ndarray, context: EvaluationContext, with_gradient: bool):
        self._check_epoch(context)
        if self._spent["candidate_evaluations"] >= self.budget.candidate_evaluations:
            raise _BudgetExhausted("candidate_budget_exhausted")
        if self._spent["hard_validation_calls"] >= self.budget.max_validation_calls:
            raise _BudgetExhausted("hard_validation_budget_exhausted")
        theta = _frozen_array(theta, "evaluation theta")
        # Reserve the actual two independent operations together: no soft
        # result can escape hard validation because validation has no budget.
        self._spent["candidate_evaluations"] += 1
        soft = None
        try:
            soft = self.soft_evaluator(theta, context, with_gradient)
            if not isinstance(soft, SoftObservation):
                raise TypeError("soft callback must return SoftObservation")
        except (ValueError, FloatingPointError, RuntimeError) as error:
            soft = SoftObservation(context, None, notes=(f"soft_evaluation_error:{error}",))
        self._spent["hard_validation_calls"] += 1
        try:
            hard = self.hard_validator(theta, context)
            if not isinstance(hard, HardObservation):
                raise TypeError("hard callback must return HardObservation")
        except (ValueError, FloatingPointError, RuntimeError) as error:
            hard = HardObservation(context, False, source_valid=False,
                                   notes=(f"hard_evaluation_error:{error}",))
        expected_hash = parameter_hash(theta)
        if ((soft.theta_hash is not None and soft.theta_hash != expected_hash)
                or (hard.theta_hash is not None and hard.theta_hash != expected_hash)):
            raise _ContextMismatch("returned observation belongs to different parameter values")
        pair = PairedObservation(soft, hard)
        if with_gradient and soft.gradient is not None and soft.gradient.shape != theta.shape:
            raise ValueError("soft gradient shape does not match evaluated theta")
        return pair

    def initialize(self, theta: Any, context: EvaluationContext,
                 optimizer_config: Optional[Mapping[str, Any]] = None,
                 with_gradient: bool = True) -> StepOutcome:
        base = OptimizerSnapshot.initial(theta, context, optimizer_config)
        before = self._spent.copy()
        try:
            observation = self._observe(base.theta, context, with_gradient)
        except _BudgetExhausted as error:
            return StepOutcome(base, "blocked", str(error), spent={})
        snapshot = base.with_observation(observation)
        spent = {name: self._spent[name] - before[name] for name in self._spent}
        assessment = assess_constraints(observation.hard, self.constraints)
        if assessment.status is Feasibility.UNKNOWN:
            return StepOutcome(snapshot, "blocked", "initial_hard_validation_unknown", spent=spent)
        status = "initialized_feasible" if assessment.status is Feasibility.FEASIBLE else "initialized_infeasible"
        return StepOutcome(snapshot, status, status, spent=spent)

    def _incumbent(self, snapshot: OptimizerSnapshot, context: EvaluationContext):
        previous = snapshot.observation
        if (previous is not None and previous.context == context
                and previous.soft.gradient is not None):
            return previous
        if previous is None:
            raise ValueError("initialize the observed initial state before stepping")
        return self._observe(snapshot.theta, context, True)

    def _rank(self, assessment: ConstraintAssessment, residual: float, loss: float):
        # Domain feasibility dominates the least-violation and objective terms.
        # The latter terms avoid treating a signed Dice gap as a safe denominator.
        domain = 0 if assessment.status is Feasibility.FEASIBLE else 1
        return (domain, assessment.violation, residual, loss)

    def _strictly_better(self, left: tuple, right: tuple) -> bool:
        if left[0] != right[0]:
            return left[0] < right[0]
        for a, b in zip(left[1:], right[1:]):
            tolerance = max(self.constraints.fraction_atol, self.constraints.objective_atol)
            tolerance += self.constraints.objective_rtol * max(abs(a), abs(b))
            if abs(a - b) <= tolerance:
                continue
            return a < b
        return False

    def step(self, snapshot: OptimizerSnapshot,
             proposal_fn: Callable[[OptimizerSnapshot, float], Proposal],
             context: Optional[EvaluationContext] = None) -> StepOutcome:
        """Try a proposal from the incumbent, never a rejected branch's moments.

        ``proposal_fn`` must construct theta/moments from its supplied incumbent
        and trial learning rate. Retrying therefore cannot cumulatively move the
        rejected theta. Rejection returns a new exact copy of the old optimizer
        state; consumed objective/validation budgets and attempt logs remain.
        """
        base = snapshot.clone()
        if base.context.loss_shift != base.config.get("loss_shift", base.context.loss_shift):
            raise ValueError("shift/config mismatch in incumbent snapshot")
        context = base.context if context is None else context
        start_spent = self._spent.copy()
        history: list[Mapping[str, Any]] = []
        if (self.budget.max_generations is not None
                and base.optimizer_step >= self.budget.max_generations):
            return StepOutcome(base, "blocked", "generation_budget_exhausted")
        if not self.config.learning_rate_min <= base.learning_rate <= self.config.learning_rate_max:
            raise ValueError("incumbent learning rate is outside its declared bounds")

        def delta() -> dict[str, int]:
            return {name: self._spent[name] - start_spent[name] for name in self._spent}

        try:
            incumbent = self._incumbent(base, context)
        except _BudgetExhausted as error:
            return StepOutcome(base, "blocked", str(error), spent=delta())
        if incumbent.soft.gradient is None:
            return StepOutcome(base, "blocked", "gradient_unavailable", spent=delta())
        initial_assessment = assess_constraints(incumbent.hard, self.constraints)
        if initial_assessment.status is Feasibility.UNKNOWN:
            return StepOutcome(base, "blocked", "initial_hard_validation_unknown",
                               spent=delta(), history=(initial_assessment.missing,))
        work = base.with_observation(incumbent, context)
        learning_rate = base.learning_rate
        for attempt in range(self.config.max_retries_per_step + 1):
            if not self.config.learning_rate_min <= learning_rate <= self.config.learning_rate_max:
                return StepOutcome(base, "rolled_back", "learning_rate_bound_exhausted",
                                   attempts=attempt, history=tuple(history), spent=delta())
            proposal = proposal_fn(work, learning_rate)
            if not isinstance(proposal, Proposal):
                raise TypeError("proposal_fn must return a Proposal")
            if proposal.context != context:
                return StepOutcome(base, "rolled_back", "comparison_context_mismatch",
                                   attempts=attempt, history=tuple(history), spent=delta())
            if proposal.optimizer_config["learning_rate"] != learning_rate:
                raise ValueError("proposal must use the trial learning rate in its full config")
            if proposal.optimizer_step <= base.optimizer_step:
                raise ValueError("accepted proposal must advance the optimizer step; retry from incumbent")
            self._spent["proposals"] += 1
            if attempt:
                self._spent["retries"] += 1
            try:
                candidate = self._observe(proposal.theta, context, False)
            except (_BudgetExhausted, _ContextMismatch) as error:
                return StepOutcome(base, "rolled_back" if not isinstance(error, _ContextMismatch) else "blocked",
                                   str(error), attempts=attempt + 1, retries=attempt,
                                   history=tuple(history), spent=delta())
            if candidate.hard.raw_phi_hard is None or incumbent.hard.raw_phi_hard is None:
                return StepOutcome(base, "blocked", "raw_hard_validation_objective_unavailable",
                                   attempts=attempt + 1, retries=attempt,
                                   history=tuple(history), spent=delta())
            candidate_assessment = assess_constraints(candidate.hard, self.constraints)
            if candidate_assessment.status is Feasibility.UNKNOWN:
                history.append({"attempt": attempt, "lr": learning_rate,
                                "candidate": "unknown_constraints",
                                "missing": list(candidate_assessment.missing)})
                return StepOutcome(base, "blocked", "candidate_hard_validation_unknown",
                                   attempts=attempt + 1, retries=attempt,
                                   history=tuple(history), spent=delta())
            # Score and tolerance use raw hard validation objectives. A signed
            # stock-field shift shapes the smooth surrogate only; it is not a
            # scalar addition to either side of a comparison (and cannot alter
            # a relative-tolerance outcome through a fake scalar offset).
            incumbent_loss = incumbent.hard.raw_phi_hard
            candidate_loss = candidate.hard.raw_phi_hard
            incumbent_rank = self._rank(initial_assessment, incumbent.hard.residual_fraction,
                                        incumbent_loss)
            candidate_rank = self._rank(candidate_assessment, candidate.hard.residual_fraction,
                                        candidate_loss)
            accepted = self._strictly_better(candidate_rank, incumbent_rank)
            history.append({
                "attempt": attempt, "lr": learning_rate,
                "incumbent_rank": incumbent_rank, "candidate_rank": candidate_rank,
                "candidate_feasibility": candidate_assessment.status.value,
                "accepted": accepted,
            })
            if accepted:
                if initial_assessment.status is Feasibility.INFEASIBLE:
                    kind = ("recovered_feasible" if candidate_assessment.status is Feasibility.FEASIBLE
                            else "least_violation")
                else:
                    kind = "improvement"
                accepted_snapshot = OptimizerSnapshot(
                    proposal.theta, proposal.first_moment, proposal.second_moment,
                    proposal.optimizer_config, context, proposal.optimizer_step, candidate,
                )
                self._spent["accepted"] += 1
                return StepOutcome(accepted_snapshot, "accepted", kind, accepted=True,
                                   acceptance_kind=kind, attempts=attempt + 1, retries=attempt,
                                   history=tuple(history), spent=delta())
            if attempt < self.config.max_retries_per_step:
                next_rate = learning_rate * self.config.learning_rate_shrink
                if next_rate < self.config.learning_rate_min:
                    return StepOutcome(base, "rolled_back", "learning_rate_floor_reached",
                                       attempts=attempt + 1, retries=attempt,
                                       history=tuple(history), spent=delta())
                learning_rate = next_rate
        return StepOutcome(base, "rolled_back", "bounded_retries_exhausted",
                           attempts=self.config.max_retries_per_step + 1,
                           retries=self.config.max_retries_per_step,
                           history=tuple(history), spent=delta())

    def calibrate(self, observations: Sequence[StockFieldPair], band: SurfaceBand,
                  current_shift: float) -> CalibrationResult:
        # Pure arithmetic here; backend observations and their already-accounted
        # costs are supplied by the caller. No k schedule, moments or theta edits.
        return self.calibrator.update(observations, band, current_shift)


def from_merged_soft(evaluation: Any, context: EvaluationContext, *,
                     path: Optional[np.ndarray] = None,
                     applied_shift: Optional[float] = None) -> SoftObservation:
    """Adapt the SCALAR result at the neutral zero field shift.

    This helper cannot convert a shifted scalar loss into an unshifted stock
    field. ``phi_soft + voxel_shift`` is NOT how occupancy shifts work; use the
    raw ``stock_d`` array in StockFieldPair for signed-field calibration.
    """
    if applied_shift is not None:
        raise ValueError("a stock-field shift cannot be removed by scalar subtraction; use raw stock_d")
    if context.loss_shift != 0.0:
        raise ValueError("merged scalar adapter requires the neutral zero shift")
    result = evaluation.result if hasattr(evaluation, "result") else evaluation
    if result.objective_kind != "smooth":
        raise ValueError("from_merged_soft requires the smooth objective result")
    value = result.objective
    raw_phi = None
    if result.valid and value is not None:
        raw_phi = _optional_number(value, "merged soft objective at zero field shift")
    gradient = getattr(evaluation, "gradient_theta", None)
    return SoftObservation(context, raw_phi, path=path, gradient=gradient,
                           theta_hash=result.theta_hash)


def from_merged_hard(result: C.CandidateResult, context: EvaluationContext, *,
                     path: Optional[np.ndarray] = None,
                     local_surface_error_mm: Optional[float] = None,
                     safety_pass: Optional[bool] = None) -> HardObservation:
    """Map only known merged hard fields; absent local error/safety stays unknown."""
    if result.objective_kind != "hard":
        raise ValueError("from_merged_hard requires the hard objective result")
    raw_phi = None
    if result.valid and result.objective is not None:
        raw_phi = _optional_number(result.objective, "merged hard objective")
    notes = []
    if local_surface_error_mm is None:
        notes.append("local_surface_error_mm_not_available_in_merged_objective")
    if safety_pass is None:
        notes.append("explicit_safety_pass_not_available_in_merged_objective")
    if result.coverage is not C.Coverage.COMPLETE:
        notes.append(f"coverage_is_{result.coverage.value}_not_complete")
    return HardObservation(
        context=context, coverage_complete=result.coverage is C.Coverage.COMPLETE,
        gouge_fraction=result.components.get("gouge_fraction"),
        residual_fraction=result.components.get("residual_fraction"),
        local_surface_error_mm=local_surface_error_mm,
        holder_overlap_fraction=result.components.get("holder_overlap_fraction"),
        safety_pass=safety_pass, raw_phi_hard=raw_phi, path=path,
        theta_hash=result.theta_hash, source_valid=result.valid,
        notes=tuple(notes),
    )


def dev_ablation_plan() -> dict[str, Any]:
    """Planned, not executed: dev-only comparisons after coordinator pilot GO."""
    shared = {
        "split": "dev",
        "freeze_task_grid_and_hashes_before_comparison": True,
        "initial_theta": "identical in-repo seed residuals for every method",
        "initial_moments": "identical Adam state for every method",
        "optimizer": {"learning_rate": 0.005, "beta1": 0.9, "beta2": 0.999,
                      "epsilon": 1e-8, "max_gradient_norm": 0.5},
        "budget_rule": "same declared soft-objective/hard-validation and generation caps",
        "mechanism_cost_rule": "calibration/fresh-incumbent/retry calls count and are also reported separately",
        "acceptance": "same k, shift, physical grid and hard-validator state for any compared pair",
        "constraints": "explicit gouge, local surface error, residual and safety state; unknown remains unknown",
        "outputs": "retain accepted and rejected states, moments/config, hashes and actual compute ledgers",
        "interpretation": "no-preferred-outcome; no-useful-advantage is a reason to omit this mechanism",
        "holdout": "do not launch or tune any test-split/held-out cases from this study",
    }
    shared_hash = C.content_hash(shared)
    return {
        "schema_version": "t3.4-dev-ablation-plan/1",
        "status": "planned_only_not_executed",
        "launch_requires": "measured T4.1 pilot plus explicit diffcam-orchestrator GO",
        "shared_configuration": shared,
        "shared_configuration_sha256": shared_hash,
        "comparisons": [
            {"candidate": "signed_ema_backtracking", "reference": "fixed_adam",
             "schedule": "same fixed pilot-selected k for both", "configuration_sha256": shared_hash},
            {"candidate": "signed_ema_backtracking", "reference": "annealed_adam",
             "schedule": "same bounded pilot-selected annealed k schedule for both",
             "configuration_sha256": shared_hash},
        ],
        "shift_policy": "candidate-only signed EMA over median(hard signed-EDT - raw simulator stock_d) on the identical executed path/grid in a declared millimetre target-surface band; in_stock_d voxels with <=0.5_voxel clip and recorded mm equivalent",
        "hard_acceptance": "raw hard-validation objectives, without scalar shift addition or changed-k comparison",
        "outcomes": ["retain", "no_useful_advantage_omit", "inconclusive_unknown_coverage", "implementation_reject"],
        "actual_execution": None,
    }


__all__ = [
    "PhysicalBand", "SurfaceBand", "StockFieldPair", "FieldObservation",
    "ConstraintPolicy", "Feasibility", "EvaluationContext", "SharpnessSchedule",
    "SoftObservation", "HardObservation", "PairedObservation",
    "ConstraintAssessment", "assess_constraints", "bounded_disagreement",
    "SignedLossCalibrator", "SignedFieldCalibrator", "CalibrationResult",
    "OptimizerSnapshot", "Proposal",
    "BacktrackingConfig", "StepOutcome", "BacktrackingMechanism",
    "from_merged_soft", "from_merged_hard", "parameter_hash", "grid_signature",
    "dev_ablation_plan",
]
