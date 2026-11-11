"""Float64 trajectory maps shared by trajectory optimizers.

Every class implements :class:`benchmarks.contracts.Parameterization`:

``forward(theta) -> (T, 3)`` commanded segment displacements in millimetres;
``pullback(grad_deltas) -> (num_params,)`` the exact transpose-Jacobian product.

``theta`` is normalized and dimensionless, flattened in ``(point, xyz)`` C-order.
``scale_mm`` is mm per normalized parameter unit and is deliberately explicit.
Bounds, when supplied, constrain *free parameters*, not dependent control points or
commanded displacements. ``forward`` rejects out-of-bounds parameters; ``clip``
provides an explicit projection for bounded solvers. These numerical maps do not
perform speed clipping, timing, work-coordinate conversion, collision checks, or
other physical execution operations.

The initial position is never a parameter. ``anchor_end=True`` instead preserves
the reference terminal position. For delta modes this means zero total residual
displacement. The final control point is dependent and is not exposed as a free
parameter. For absolute residuals, the terminal residual is fixed at zero. T
always denotes *segments* (hence T+1 positions), never the number of positions.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


Array = np.ndarray
_Bounds = Optional[Tuple[float, float]]


def _readonly(array: Array) -> Array:
    result = np.asarray(array, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


def _int_at_least(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name}: expected an integer >= {minimum}, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name}: expected an integer >= {minimum}, got {result}")
    return result


def _finite_array(name: str, values, shape_tail: Tuple[int, ...]) -> Array:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a real numeric array") from exc
    if array.ndim < 2 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name}: expected shape (N, {', '.join(map(str, shape_tail))}), got {array.shape}")
    if array.shape[0] < 1:
        raise ValueError(f"{name}: must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: all entries must be finite")
    return array


def _initial_position(values) -> Array:
    if values is None:
        return np.zeros(3, dtype=np.float64)
    try:
        position = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_position_mm: expected three finite mm coordinates") from exc
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("initial_position_mm: expected three finite mm coordinates")
    return position


def _scale(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("scale_mm: must be a finite positive mm/unit scalar") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError("scale_mm: must be a finite positive mm/unit scalar")
    return result


def _endpoint_matrices(basis: Array, anchor_end: bool) -> tuple[Array, Array]:
    """Return (basis-on-free-original-controls, exact delta Jacobian).

    If S = ones(T) @ basis, setting the final original control to
    ``-sum(S[:-1] * theta) / S[-1]`` makes ``ones(T) @ basis @ controls == 0``.
    S is the basis's time-column sum, so it is all ones for the raw delta map
    but generally not for a spline. Free parameters remain scaled by exactly
    scale_mm; the dependent final point is intentionally not bounded.
    """
    if not anchor_end:
        return basis, basis
    steps, controls = basis.shape
    if controls < 2 or steps < 2:
        raise ValueError("anchor_end=True requires at least two segments and two controls")
    sums = basis.sum(axis=0)
    # Cox-de Boor evaluates the partition-of-unity identity with small rounding
    # discrepancies. Preserve that exact algebraic identity when choosing the
    # dependent control instead of introducing a spurious 1+epsilon ratio.
    sums[np.abs(sums - 1.0) <= 8 * np.finfo(np.float64).eps] = 1.0
    if not np.isfinite(sums).all() or sums[-1] <= np.finfo(np.float64).eps:
        raise ValueError("anchor_end=True requires a nonzero terminal control coefficient sum")
    dependent = sums[:-1] / sums[-1]
    position_basis = np.zeros((controls, controls - 1), dtype=np.float64)
    position_basis[:-1] = np.eye(controls - 1, dtype=np.float64)
    position_basis[-1] = -dependent
    return position_basis, basis @ position_basis


def cubic_open_uniform_basis(num_steps: int, num_control_points: int) -> Array:
    """Clamped cubic B-spline collocation matrix of shape ``(T, K)``.

    Requires ``4 <= K <= T``. The knot vector is clamped to [0,1], with internal
    knots uniformly spaced. Evaluation points are exactly ``linspace(0,1,T)``;
    there is no midpoint convention hidden here. At the right endpoint evaluation
    is the continuous limiting value, so row zero is e_0 and row T-1 is e_(K-1).
    This is a basis on segment displacements, not a physical time grid.
    K=T supplies 3T parameters but is not an identity basis; use ``DeltaResidual``
    when literal raw per-step control is required.
    """
    steps = _int_at_least("T", num_steps, 1)
    controls = _int_at_least("K", num_control_points, 4)
    if controls > steps:
        raise ValueError(f"K must satisfy 4 <= K <= T; got K={controls}, T={steps}")
    degree = 3
    internal = (np.arange(1, controls - degree, dtype=np.float64) / (controls - degree))
    knots = np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))
    basis = np.empty((steps, controls), dtype=np.float64)
    for row, u in enumerate(np.linspace(0.0, 1.0, steps)):
        # Cox-de Boor, including the otherwise zero-measure right endpoint.
        values = np.zeros(controls + degree + 1, dtype=np.float64)
        for j in range(controls + degree):
            if knots[j] <= u < knots[j + 1] or (u == 1.0 and knots[j] < 1.0 and knots[j + 1] == 1.0):
                values[j] = 1.0
        for order in range(1, degree + 1):
            for j in range(controls + degree - order):
                left_den = knots[j + order] - knots[j]
                right_den = knots[j + order + 1] - knots[j + 1]
                left = 0.0 if left_den == 0 else (u - knots[j]) / left_den * values[j]
                right = 0.0 if right_den == 0 else (knots[j + order + 1] - u) / right_den * values[j + 1]
                values[j] = left + right
        basis[row] = values[:controls]
    return basis


class _LinearDeltaParameterization:
    """Common immutable affine map, with exact analytic adjoint."""

    mode = "linear_delta"

    def __init__(self, reference_deltas_mm, linear_map, scale_mm, initial_position_mm=None,
                 bounds=None, mode=None, reference_positions_mm=None):
        reference = _finite_array("reference_deltas_mm", reference_deltas_mm, (3,))
        matrix = _finite_array("linear_map (before final dimension check)", linear_map, (linear_map.shape[-1],))
        if matrix.ndim != 2 or matrix.shape[0] != reference.shape[0]:
            raise ValueError("internal linear map must have shape (T, free_controls)")
        self._reference_deltas = _readonly(reference)
        self._linear_map = _readonly(matrix)
        self._scale_mm = _scale(scale_mm)
        self._initial_position = _readonly(_initial_position(initial_position_mm))
        if reference_positions_mm is None:
            positions = np.vstack((self._initial_position,
                                   self._initial_position + np.cumsum(reference, axis=0)))
        else:
            positions = _finite_array("reference_positions_mm", reference_positions_mm, (3,))
            if positions.shape != (reference.shape[0] + 1, 3):
                raise ValueError("reference positions must have shape (T+1, 3)")
            np.testing.assert_allclose(np.diff(positions, axis=0), reference, rtol=0, atol=1e-12,
                                       err_msg="internal reference position/delta mismatch")
        self._reference_positions = _readonly(positions)
        self._num_controls = matrix.shape[1]
        self._mode = mode or self.mode
        self._lower_bounds = None
        self._upper_bounds = None
        if bounds is not None:
            self._set_bounds(bounds)

    @property
    def num_params(self) -> int:
        return self._num_controls * 3

    @property
    def num_steps(self) -> int:
        return self._reference_deltas.shape[0]

    @property
    def parameter_shape(self) -> Tuple[int, int]:
        return self._num_controls, 3

    @property
    def scale_mm(self) -> float:
        return self._scale_mm

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def initial_position_mm(self) -> Array:
        return self._initial_position

    @property
    def reference_deltas_mm(self) -> Array:
        return self._reference_deltas

    @property
    def reference_positions_mm(self) -> Array:
        return self._reference_positions

    @property
    def linear_map(self) -> Array:
        """Read-only Jacobian in normalized parameter units, shape ``(T,Kfree)``."""
        return self._linear_map

    @property
    def bounds(self):
        if self._lower_bounds is None:
            return None
        return self._lower_bounds, self._upper_bounds

    def _set_bounds(self, bounds) -> None:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("bounds: expected (lower, upper) scalars or arrays, or None")
        converted = []
        for name, value in zip(("lower", "upper"), bounds):
            try:
                array = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bounds {name}: expected a scalar or flattened numeric array") from exc
            if array.ndim == 0:
                result = np.full(self.num_params, float(array), dtype=np.float64)
            if array.ndim == 0:
                result = np.full(self.num_params, float(array), dtype=np.float64)
            elif array.size == self.num_params:
                # Free-control count can differ from the raw trajectory length
                # (anchored/spline/absolute modes), so reshape by num_params.
                result = array.reshape(-1).copy()
            elif array.shape == (3,):
                try:
                    result = np.broadcast_to(array, self.parameter_shape).reshape(-1).copy()
                except ValueError as exc:
                    raise ValueError(f"bounds {name}: cannot broadcast to {self.parameter_shape}") from exc
            else:
                raise ValueError(f"bounds {name}: expected scalar or {self.num_params} entries, got {array.size}")
            if not np.isfinite(result).all():
                raise ValueError(f"bounds {name}: all entries must be finite (use bounds=None for unbounded)")
            converted.append(result)
        lower, upper = converted
        if np.any(lower >= upper):
            raise ValueError("bounds: every lower entry must be strictly less than upper")
        if np.any(lower > 0) or np.any(upper < 0):
            raise ValueError("bounds: must contain theta=0, the shared CAM-zero-residual initialization")
        self._lower_bounds = _readonly(lower)
        self._upper_bounds = _readonly(upper)

    def _theta(self, theta, check_bounds=True) -> tuple[Array, Array]:
        try:
            array = np.asarray(theta, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"theta: expected {self.parameter_shape} or ({self.num_params},)") from exc
        if array.shape == (self.num_params,):
            flat = array
        elif array.shape == self.parameter_shape:
            flat = array.reshape(-1)
        else:
            raise ValueError(f"theta: expected {self.parameter_shape} or ({self.num_params},), got {array.shape}")
        if not np.isfinite(flat).all():
            raise ValueError("theta: all entries must be finite")
        flat = np.asarray(flat, dtype=np.float64)
        if check_bounds and self._lower_bounds is not None:
            if np.any(flat < self._lower_bounds) or np.any(flat > self._upper_bounds):
                raise ValueError("theta: outside declared normalized bounds; use clip() explicitly if projection is intended")
        return flat, flat.reshape(self.parameter_shape)

    def in_bounds(self, theta, tolerance: float = 0.0) -> bool:
        flat, _ = self._theta(theta, check_bounds=False)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance: must be finite and nonnegative")
        if self._lower_bounds is None:
            return True
        return bool(np.all(flat >= self._lower_bounds - tolerance)
                    and np.all(flat <= self._upper_bounds + tolerance))

    def clip(self, theta) -> Array:
        """Return a flattened copy projected onto declared numerical bounds."""
        flat, _ = self._theta(theta, check_bounds=False)
        if self._lower_bounds is None:
            return flat.copy()
        return np.clip(flat, self._lower_bounds, self._upper_bounds)

    def forward(self, theta) -> Array:
        """Map normalized parameters to an independent float64 ``(T,3)`` mm array."""
        _, matrix_theta = self._theta(theta)
        return self._reference_deltas + self._scale_mm * (self._linear_map @ matrix_theta)

    def pullback(self, grad_deltas) -> Array:
        """Return exact ``J.T @ grad``; no numerical differentiation is used."""
        gradient = _finite_array("grad_deltas", grad_deltas, (3,))
        if gradient.shape != (self.num_steps, 3):
            raise ValueError(f"grad_deltas: expected ({self.num_steps}, 3), got {gradient.shape}")
        # scale_mm multiplies BOTH forward and its exact transpose Jacobian.
        return (self._scale_mm * (self._linear_map.T @ gradient)).reshape(-1)

    def positions(self, theta) -> Array:
        """Reconstruct ``(T+1,3)`` positions with the reference start held fixed."""
        _, matrix_theta = self._theta(theta)
        residual = self._scale_mm * (self._linear_map @ matrix_theta)
        if self._reference_positions.shape[0] != self.num_steps + 1:
            raise RuntimeError("internal reference position count mismatch")
        # For absolute residuals use the independent reference supplied by the
        # caller (avoiding a needless reference-diff/cumsum rounding round trip).
        positions = self._reference_positions.copy()
        if hasattr(self, "_position_residual_map"):
            # Absolute-position residuals are independent residuals on P1..PT,
            # not increments of successive residuals. The D map is in forward().
            positions[1:] += self._scale_mm * (self._position_residual_map @ matrix_theta)
        else:
            positions[1:] += np.cumsum(residual, axis=0)
        return positions


class DeltaResidual(_LinearDeltaParameterization):
    """Raw segment residuals: ``delta = CAM_delta + scale_mm * theta``.

    With ``anchor_end=True``, theta has ``(T-1,3)`` entries and the final residual
    is the negative sum of the preceding residuals. This is a rank T-1 map per
    coordinate, not a redundant parameter vector with an equality constraint.
    """

    def __init__(self, cam_deltas_mm, scale_mm: float, bounds: _Bounds = None,
                 anchor_end: bool = False, initial_position_mm=None):
        deltas = _finite_array("cam_deltas_mm", cam_deltas_mm, (3,))
        steps = deltas.shape[0]
        if anchor_end and steps < 2:
            raise ValueError("DeltaResidual anchor_end=True requires T >= 2")
        identity = np.eye(steps, dtype=np.float64)
        _, mapping = _endpoint_matrices(identity, anchor_end)
        self.anchor_end = bool(anchor_end)
        self.num_control_points = steps
        super().__init__(deltas, mapping, scale_mm, initial_position_mm, bounds, "delta_residual")

    @property
    def basis(self) -> Array:
        return self.linear_map


class DeltaSpline(_LinearDeltaParameterization):
    """Cubic B-spline segment residuals: ``delta = CAM_delta + B theta``.

    K is the number of original controls; ``4 <= K <= T``. The free case has
    3K parameters. An anchored end has 3(K-1) free parameters and an effective
    basis ``B @ A`` with zero row sums. The dependent K-th original control is
    computed analytically, so terminal CAM position is preserved exactly.
    """

    def __init__(self, cam_deltas_mm, scale_mm: float, num_control_points: Optional[int] = None,
                 bounds: _Bounds = None, anchor_end: bool = False, initial_position_mm=None):
        deltas = _finite_array("cam_deltas_mm", cam_deltas_mm, (3,))
        steps = deltas.shape[0]
        controls = steps if num_control_points is None else num_control_points
        basis = cubic_open_uniform_basis(steps, controls)
        position_basis, mapping = _endpoint_matrices(basis, anchor_end)
        self.anchor_end = bool(anchor_end)
        self.num_control_points = int(controls)
        self._control_point_basis = _readonly(basis)
        self._spline_free_map = _readonly(position_basis)
        super().__init__(deltas, mapping, scale_mm, initial_position_mm, bounds, "delta_spline")

    @property
    def basis(self) -> Array:
        """Effective basis from exposed free theta to delta residuals."""
        return self.linear_map

    @property
    def control_point_basis(self) -> Array:
        """Original (T,K) clamped cubic B-spline collocation basis B."""
        return self._control_point_basis


class AbsolutePositionResidual(_LinearDeltaParameterization):
    """Residual absolute positions with fixed P0 and exact differencing.

    Reference positions have shape ``(T+1,3)``. Free theta describes residual P1..PT;
    anchored theta describes P1..P_(T-1), with residual PT fixed to zero. In both
    cases commanded deltas are ``diff(P_CAM + scale_mm * position_residual)``, and
    pullback uses that exact differencing operator's transpose. P0 does not carve.
    """

    def __init__(self, cam_positions_mm, scale_mm: float, bounds: _Bounds = None,
                 anchor_end: bool = False):
        positions = _finite_array("cam_positions_mm", cam_positions_mm, (3,))
        if positions.shape[0] < 2:
            raise ValueError("cam_positions_mm: expected at least P0 and P1")
        steps = positions.shape[0] - 1
        if anchor_end and steps < 2:
            raise ValueError("AbsolutePositionResidual anchor_end=True requires T >= 2")
        position_map = np.zeros((steps, steps), dtype=np.float64)
        position_map[:] = np.eye(steps, dtype=np.float64)
        if anchor_end:
            position_map = position_map[:, :-1]
        # D differentiates the residual positions P1..PT (P0 residual is zero).
        difference = np.zeros((steps, steps), dtype=np.float64)
        difference[0, 0] = 1.0
        for row in range(1, steps):
            difference[row, row - 1] = -1.0
            difference[row, row] = 1.0
        mapping = difference @ position_map
        self.anchor_end = bool(anchor_end)
        self.num_control_points = steps
        self._position_residual_map = _readonly(position_map)
        self._differencing_matrix = _readonly(difference)
        super().__init__(np.diff(positions, axis=0), mapping, scale_mm, positions[0], bounds,
                         "absolute_position_residual", positions)

    @property
    def position_parameter_matrix(self) -> Array:
        """Embedding from theta to residual P1..PT, including fixed terminal row."""
        return self._position_residual_map

    @property
    def differencing_matrix(self) -> Array:
        """Exact D such that delta residual = D @ position residual P1..PT."""
        return self._differencing_matrix


class PositionSpline(_LinearDeltaParameterization):
    """Cubic B-spline residuals directly on interior waypoint positions ``P1..P_(T-1)``.

    Unlike ``DeltaSpline`` (which Perturbs segment deltas and accumulates ``O(T)`` drift via
    ``cumsum``), ``PositionSpline`` places a clamped cubic B-spline basis ``B`` on interior
    waypoint positions ``delta_P[1..T-1] = scale_mm * diag(w) @ B @ theta`` with ``delta_P[0] = 0``
    and ``delta_P[T] = 0`` (when ``anchor_end=True``), and maps to segment delta residuals via
    the exact first-difference operator ``D``. Every waypoint displacement is strictly bounded
    by ``scale_mm`` regardless of horizon length ``T``.
    """

    def __init__(
        self,
        cam_deltas_mm,
        scale_mm: float,
        num_control_points: Optional[int] = None,
        bounds: _Bounds = None,
        anchor_end: bool = True,
        waypoint_weights=None,
        initial_position_mm=None,
    ):
        deltas = _finite_array("cam_deltas_mm", cam_deltas_mm, (3,))
        steps = deltas.shape[0]
        if anchor_end and steps < 2:
            raise ValueError("PositionSpline anchor_end=True requires T >= 2")
        n_active = steps - 1 if anchor_end else steps
        controls = n_active if num_control_points is None else min(int(num_control_points), n_active)
        if n_active < 4 or controls == n_active:
            active_basis = np.eye(n_active, dtype=np.float64)
            controls = n_active
        else:
            controls = max(4, controls)
            active_basis = cubic_open_uniform_basis(n_active, controls)

        if waypoint_weights is not None:
            w = np.asarray(waypoint_weights, dtype=np.float64).reshape(-1)
            if w.shape != (n_active,) or not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError(f"waypoint_weights must be non-negative finite shape ({n_active},)")
            active_basis = w[:, None] * active_basis

        position_map = np.zeros((steps, controls), dtype=np.float64)
        position_map[:n_active, :] = active_basis

        difference = np.zeros((steps, steps), dtype=np.float64)
        difference[0, 0] = 1.0
        for row in range(1, steps):
            difference[row, row - 1] = -1.0
            difference[row, row] = 1.0
        mapping = difference @ position_map
        self.anchor_end = bool(anchor_end)
        self.num_control_points = int(controls)
        self._position_residual_map = _readonly(position_map)
        self._differencing_matrix = _readonly(difference)
        base_init = np.zeros(3, dtype=np.float64) if initial_position_mm is None else np.asarray(initial_position_mm, dtype=np.float64)
        ref_positions = np.vstack([base_init[None, :], base_init[None, :] + np.cumsum(deltas, axis=0)])
        super().__init__(
            deltas,
            mapping,
            scale_mm,
            base_init,
            bounds,
            "position_spline",
            ref_positions,
        )

    @property
    def basis(self) -> Array:
        return self.linear_map


class JointPathFeedSpline:
    """Co-parameterization of spatial B-spline residuals and per-segment log-feed splines.

    ``theta`` has length ``num_pos_params + num_feed_params``.
    * ``forward(theta) -> (T, 3)`` commanded segment displacements in mm.
    * ``forward_feed_multipliers(theta) -> (T,)`` positive feed multipliers ``exp(log_scale * B_f @ theta_f)``
      on cutting segments, and exact ``1.0`` on rapid segments (where ``is_rapid[t] == True``).
    * ``pullback(grad_deltas)`` preserves the spatial-only interface (zero feed gradient).
    * ``pullback_joint(grad_deltas, grad_feed_mult, theta)`` returns the exact full gradient
      ``(num_params,)`` combining spatial transpose-Jacobian and ``d exp(u)/d theta_f``.
    """

    mode = "joint_path_feed_spline"

    def __init__(
        self,
        cam_deltas_mm,
        scale_mm: float,
        *,
        num_pos_controls: Optional[int] = None,
        num_feed_controls: Optional[int] = None,
        log_feed_scale: float = 0.35,
        pos_bounds: _Bounds = (-1.0, 1.0),
        feed_bounds: Optional[Tuple[float, float]] = (-1.0, 1.5),
        is_rapid=None,
        max_feed_mult_per_seg=None,
        anchor_end: bool = False,
        xy_only: bool = False,
        spatial_mode: str = "delta_spline",
        waypoint_weights=None,
        initial_position_mm=None,
    ):
        if spatial_mode == "position_spline":
            self.spatial = PositionSpline(
                cam_deltas_mm,
                scale_mm=scale_mm,
                num_control_points=num_pos_controls,
                bounds=pos_bounds,
                anchor_end=anchor_end,
                waypoint_weights=waypoint_weights,
                initial_position_mm=initial_position_mm,
            )
        elif spatial_mode == "delta_spline":
            self.spatial = DeltaSpline(
                cam_deltas_mm,
                scale_mm=scale_mm,
                num_control_points=num_pos_controls,
                bounds=pos_bounds,
                anchor_end=anchor_end,
                initial_position_mm=initial_position_mm,
            )
        else:
            raise ValueError(f"unknown JointPathFeedSpline spatial_mode {spatial_mode!r}")
        self.xy_only = bool(xy_only)
        self._num_free_pos_pts = self.spatial.num_params // 3
        steps = self.spatial.num_steps
        f_controls = steps if num_feed_controls is None else int(num_feed_controls)
        if f_controls == steps:
            feed_basis = np.eye(steps, dtype=np.float64)
        else:
            feed_basis = cubic_open_uniform_basis(steps, max(4, min(f_controls, steps)))
            f_controls = feed_basis.shape[1]
        self._feed_basis = _readonly(feed_basis)
        self.num_feed_controls = int(f_controls)
        self.log_feed_scale = _scale(log_feed_scale)
        if is_rapid is None:
            rapid_mask = np.zeros(steps, dtype=bool)
        else:
            rapid_mask = np.asarray(is_rapid, dtype=bool)
            if rapid_mask.shape != (steps,):
                raise ValueError(f"is_rapid: expected shape ({steps},), got {rapid_mask.shape}")
        self._is_rapid = rapid_mask.copy()
        self._is_rapid.flags.writeable = False
        self._cut_mask = (~self._is_rapid).astype(np.float64)
        self._cut_mask.flags.writeable = False

        if max_feed_mult_per_seg is None:
            self._max_mult = None
        else:
            mm = np.asarray(max_feed_mult_per_seg, dtype=np.float64)
            if mm.shape != (steps,) or np.any(mm <= 0) or not np.all(np.isfinite(mm)):
                raise ValueError("max_feed_mult_per_seg must be positive finite shape (T,)")
            self._max_mult = _readonly(mm)

        self.anchor_end = self.spatial.anchor_end
        self.num_control_points = self.spatial.num_control_points

        if pos_bounds is None and feed_bounds is None:
            self._lower_bounds = None
            self._upper_bounds = None
        else:
            if self.spatial.bounds is None:
                p_lo = -np.inf * np.ones(self.num_pos_params, dtype=np.float64)
                p_hi = np.inf * np.ones(self.num_pos_params, dtype=np.float64)
            elif self.xy_only:
                sp_lo, sp_hi = self.spatial.bounds
                p_lo = sp_lo.reshape(self._num_free_pos_pts, 3)[:, :2].reshape(-1)
                p_hi = sp_hi.reshape(self._num_free_pos_pts, 3)[:, :2].reshape(-1)
            else:
                p_lo, p_hi = self.spatial.bounds
            if feed_bounds is None:
                f_lo = np.full(self.num_feed_controls, -2.0, dtype=np.float64)
                f_hi = np.full(self.num_feed_controls, 2.0, dtype=np.float64)
            else:
                f_lo = np.full(self.num_feed_controls, float(feed_bounds[0]), dtype=np.float64)
                f_hi = np.full(self.num_feed_controls, float(feed_bounds[1]), dtype=np.float64)
            lower = np.concatenate([p_lo, f_lo])
            upper = np.concatenate([p_hi, f_hi])
            if np.any(lower >= upper) or np.any(lower > 0) or np.any(upper < 0):
                raise ValueError("JointPathFeedSpline bounds must be ordered and contain theta=0")
            self._lower_bounds = _readonly(lower)
            self._upper_bounds = _readonly(upper)

    @property
    def num_pos_params(self) -> int:
        return 2 * self._num_free_pos_pts if self.xy_only else self.spatial.num_params

    @property
    def num_feed_params(self) -> int:
        return self.num_feed_controls

    @property
    def num_params(self) -> int:
        return self.num_pos_params + self.num_feed_params

    @property
    def num_steps(self) -> int:
        return self.spatial.num_steps

    @property
    def parameter_shape(self) -> Tuple[int, ...]:
        return (self.num_params,)

    @property
    def scale_mm(self) -> float:
        return self.spatial.scale_mm

    @property
    def initial_position_mm(self) -> Array:
        return self.spatial.initial_position_mm

    @property
    def reference_deltas_mm(self) -> Array:
        return self.spatial.reference_deltas_mm

    @property
    def reference_positions_mm(self) -> Array:
        return self.spatial.reference_positions_mm

    @property
    def basis(self) -> Array:
        return self.spatial.basis

    @property
    def feed_basis(self) -> Array:
        return self._feed_basis

    @property
    def bounds(self):
        if self._lower_bounds is None:
            return None
        return self._lower_bounds, self._upper_bounds

    def _split(self, theta, check_bounds: bool = True) -> Tuple[Array, Array]:
        arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if arr.shape != (self.num_params,) or not np.isfinite(arr).all():
            raise ValueError(f"theta: expected finite ({self.num_params},), got {arr.shape}")
        if check_bounds and self._lower_bounds is not None:
            if np.any(arr < self._lower_bounds - 1e-12) or np.any(arr > self._upper_bounds + 1e-12):
                raise ValueError("theta outside declared normalized bounds")
        return arr[: self.num_pos_params], arr[self.num_pos_params :]

    def in_bounds(self, theta, tolerance: float = 0.0) -> bool:
        arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if arr.shape != (self.num_params,) or not np.isfinite(arr).all():
            raise ValueError("theta must be finite with shape (num_params,)")
        if self._lower_bounds is None:
            return True
        return bool(np.all(arr >= self._lower_bounds - tolerance)
                    and np.all(arr <= self._upper_bounds + tolerance))

    def clip(self, theta) -> Array:
        arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if self._lower_bounds is None:
            return arr.copy()
        return np.clip(arr, self._lower_bounds, self._upper_bounds)

    def _expand_pos(self, theta_pos: Array) -> Array:
        if not self.xy_only:
            return theta_pos
        pts2 = theta_pos.reshape(self._num_free_pos_pts, 2)
        return np.column_stack([pts2, np.zeros(self._num_free_pos_pts, dtype=np.float64)]).reshape(-1)

    def _reduce_pos_grad(self, grad_pos_3d: Array) -> Array:
        if not self.xy_only:
            return grad_pos_3d
        return grad_pos_3d.reshape(self._num_free_pos_pts, 3)[:, :2].reshape(-1)

    def forward(self, theta) -> Array:
        theta_pos, _ = self._split(theta)
        return self.spatial.forward(self._expand_pos(theta_pos))

    def positions(self, theta) -> Array:
        theta_pos, _ = self._split(theta)
        return self.spatial.positions(self._expand_pos(theta_pos))

    def _smooth_feed_cap(self, raw_mult: Array) -> Tuple[Array, Array]:
        """Smoothly cap feed multiplier below max_feed_mult_per_seg if configured."""
        if self._max_mult is None:
            return raw_mult, np.ones_like(raw_mult)
        capped = np.minimum(raw_mult, self._max_mult)
        d_cap = (raw_mult < self._max_mult).astype(np.float64)
        return capped, d_cap

    def forward_feed_multipliers(self, theta) -> Array:
        _, theta_feed = self._split(theta)
        log_u = self.log_feed_scale * (self._feed_basis @ theta_feed)
        raw_mult = np.exp(log_u)
        capped, _ = self._smooth_feed_cap(raw_mult)
        return 1.0 + self._cut_mask * (capped - 1.0)

    def pullback(self, grad_deltas) -> Array:
        grad_pos = self._reduce_pos_grad(self.spatial.pullback(grad_deltas))
        return np.concatenate([grad_pos, np.zeros(self.num_feed_params, dtype=np.float64)])

    def pullback_joint(self, grad_deltas, grad_feed_mult, theta) -> Array:
        _, theta_feed = self._split(theta)
        grad_pos = self._reduce_pos_grad(self.spatial.pullback(grad_deltas))
        g_mult = np.asarray(grad_feed_mult, dtype=np.float64).reshape(-1)
        if g_mult.shape != (self.num_steps,) or not np.isfinite(g_mult).all():
            raise ValueError(f"grad_feed_mult: expected finite ({self.num_steps},)")
        log_u = self.log_feed_scale * (self._feed_basis @ theta_feed)
        raw_mult = np.exp(log_u)
        _, d_cap = self._smooth_feed_cap(raw_mult)
        d_theta_f = self.log_feed_scale * (self._feed_basis.T @ (g_mult * self._cut_mask * d_cap * raw_mult))
        return np.concatenate([grad_pos, d_theta_f])


__all__ = ["AbsolutePositionResidual", "DeltaResidual", "DeltaSpline",
           "PositionSpline", "JointPathFeedSpline", "cubic_open_uniform_basis"]
