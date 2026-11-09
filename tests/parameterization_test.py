"""Float64 unit tests for shared trajectory parameterizations (no simulator)."""
import numpy as np
import pytest

from algorithms.parameterization import (
    AbsolutePositionResidual,
    DeltaResidual,
    DeltaSpline,
    cubic_open_uniform_basis,
)


RNG = np.random.default_rng(2171)


def make_cases():
    rng = np.random.default_rng(8021)
    delta = rng.normal(0.0, 3.0, (5, 3))
    start = np.array([1.25, -2.0, 8.0])
    positions = np.vstack((start, start + np.cumsum(delta, axis=0)))
    return [
        DeltaResidual(delta, scale_mm=2.75, initial_position_mm=start),
        DeltaResidual(delta, scale_mm=2.75, initial_position_mm=start, anchor_end=True),
        DeltaSpline(delta, scale_mm=2.75, num_control_points=4, initial_position_mm=start),
        DeltaSpline(delta, scale_mm=2.75, num_control_points=4, initial_position_mm=start, anchor_end=True),
        AbsolutePositionResidual(positions, scale_mm=2.75),
        AbsolutePositionResidual(positions, scale_mm=2.75, anchor_end=True),
    ]


def test_raw_delta_zero_shapes_scale_and_start():
    delta = np.array([[1.0, -2.0, 0.5], [0.25, 4.0, -1.0], [2.0, 0.0, 3.0]])
    start = np.array([7.0, -8.0, 9.0])
    parameterization = DeltaResidual(delta, scale_mm=25.4, initial_position_mm=start)
    assert parameterization.mode == "delta_residual"
    assert parameterization.num_steps == 3
    assert parameterization.num_params == 9
    assert parameterization.parameter_shape == (3, 3)
    np.testing.assert_array_equal(parameterization.forward(np.zeros(9)), delta)
    np.testing.assert_array_equal(parameterization.forward(np.zeros((3, 3))), delta)
    np.testing.assert_array_equal(parameterization.positions(np.zeros(9)),
                                  np.vstack((start, start + np.cumsum(delta, axis=0))))
    theta = np.arange(9, dtype=np.float64).reshape(3, 3) / 100
    np.testing.assert_allclose(parameterization.forward(theta), delta + 25.4 * theta, rtol=0, atol=2e-14)
    gradient = np.arange(9, dtype=np.float64).reshape(3, 3) + 1
    np.testing.assert_allclose(parameterization.pullback(gradient), (25.4 * gradient).ravel(), rtol=0, atol=1e-13)
    # Public numeric arrays cannot silently change the shared affine map.
    assert not parameterization.linear_map.flags.writeable
    assert not parameterization.reference_deltas_mm.flags.writeable


def test_raw_delta_anchored_end_is_last_position_preserving_map():
    delta = np.array([[2.0, 0.0, -1.0], [1.0, -3.0, 2.0], [-4.0, 1.0, 0.5]])
    parameterization = DeltaResidual(delta, scale_mm=4.0, anchor_end=True)
    assert parameterization.parameter_shape == (2, 3)
    theta = np.array([[0.1, 0.2, -0.3], [-0.25, 0.5, 0.75]])
    expected_residual = np.vstack((theta, -theta.sum(axis=0)))
    np.testing.assert_allclose(parameterization.forward(theta), delta + 4 * expected_residual, rtol=0, atol=1e-14)
    reference_positions = parameterization.positions(np.zeros(6))
    positions = parameterization.positions(theta)
    np.testing.assert_array_equal(positions[0], reference_positions[0])
    np.testing.assert_allclose(positions[-1], reference_positions[-1], rtol=0, atol=1e-14)
    gradient = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 1.5], [4.0, -2.0, 8.0]])
    np.testing.assert_allclose(parameterization.pullback(gradient),
                               (4 * (gradient[:-1] - gradient[-1])).ravel(), rtol=0, atol=1e-14)


def test_k4_cubic_basis_is_tensor_bernstein_at_segment_locations():
    basis = cubic_open_uniform_basis(5, 4)
    u = np.linspace(0, 1, 5)[:, None]
    expected = np.hstack(((1-u)**3, 3*u*(1-u)**2, 3*u**2*(1-u), u**3))
    np.testing.assert_allclose(basis, expected, rtol=0, atol=2e-16)
    assert basis.dtype == np.float64
    np.testing.assert_array_equal(basis[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(basis[-1], [0, 0, 0, 1])


@pytest.mark.parametrize("steps,controls", [(4, 4), (8, 5), (32, 16), (16, 16), (64, 32)])
def test_cubic_basis_partition_nonnegative_and_rank(steps, controls):
    basis = cubic_open_uniform_basis(steps, controls)
    assert basis.shape == (steps, controls)
    assert np.all(basis >= 0)
    np.testing.assert_allclose(basis.sum(axis=1), np.ones(steps), rtol=0, atol=4e-15)
    np.testing.assert_array_equal(basis[0], np.eye(controls)[0])
    np.testing.assert_array_equal(basis[-1], np.eye(controls)[-1])
    assert np.linalg.matrix_rank(basis) == controls


def test_spline_forward_and_exact_basis_pullback():
    rng = np.random.default_rng(100)
    delta = rng.normal(size=(9, 3))
    parameterization = DeltaSpline(delta, scale_mm=17.0, num_control_points=5)
    basis = parameterization.control_point_basis
    theta = rng.normal(scale=0.1, size=(5, 3))
    np.testing.assert_allclose(parameterization.forward(theta), delta + 17 * basis @ theta,
                               rtol=2e-15, atol=2e-14)
    gradient = rng.normal(size=(9, 3))
    np.testing.assert_allclose(parameterization.pullback(gradient), 17 * (basis.T @ gradient).ravel(),
                               rtol=2e-15, atol=2e-14)
    np.testing.assert_allclose(parameterization.positions(np.zeros(15))[-1],
                               parameterization.initial_position_mm + delta.sum(axis=0), rtol=0, atol=1e-14)


def test_spline_anchored_end_has_dependent_final_control():
    delta = np.arange(21, dtype=np.float64).reshape(7, 3) / 10
    parameterization = DeltaSpline(delta, scale_mm=3.0, num_control_points=4, anchor_end=True)
    theta = np.array([[0.1, 0.2, 0.3], [-0.5, 0.25, 0.125], [0.75, -1.0, 0.4]])
    basis = parameterization.control_point_basis
    # Total residual is ones(T) @ B @ controls. With S=ones(T) @ B, the
    # dependent final control is -sum(S[:-1]*C[:-1])/S[-1].
    sums = basis.sum(axis=0)
    dependent = sums[:-1] / sums[-1]
    np.testing.assert_allclose(parameterization.basis,
                               basis[:, :-1] - basis[:, [-1]] * dependent,
                               rtol=0, atol=4e-16)
    np.testing.assert_allclose(parameterization.basis.sum(axis=0), np.zeros(3), rtol=0, atol=6e-16)
    np.testing.assert_allclose(parameterization.forward(theta), delta + 3 * parameterization.basis @ theta,
                               rtol=0, atol=2e-14)
    positions = parameterization.positions(theta)
    np.testing.assert_allclose(positions[-1], parameterization.reference_positions_mm[-1], rtol=0, atol=3e-14)


def test_absolute_zero_free_map_and_exact_differencing_transpose():
    delta = np.array([[2.0, -1.0, 3.0], [1.0, 0.5, -2.0], [-3.0, 4.0, 0.25]])
    start = np.array([2.0, -4.0, 8.0])
    positions = np.vstack((start, start + np.cumsum(delta, axis=0)))
    parameterization = AbsolutePositionResidual(positions, scale_mm=10.0)
    np.testing.assert_array_equal(parameterization.positions(np.zeros(9)), positions)
    np.testing.assert_array_equal(parameterization.forward(np.zeros(9)), delta)
    theta = np.array([[0.1, 0.2, -0.3], [0.25, -0.5, 0.75], [-0.8, 1.0, 0.4]])
    shifted = positions.copy()
    shifted[1:] += 10 * theta
    np.testing.assert_allclose(parameterization.positions(theta), shifted, rtol=0, atol=4e-15)
    np.testing.assert_allclose(parameterization.forward(theta), np.diff(shifted, axis=0), rtol=0, atol=4e-15)
    np.testing.assert_array_equal(parameterization.positions(theta)[0], start)
    np.testing.assert_array_equal(parameterization.differencing_matrix,
                                  [[1, 0, 0], [-1, 1, 0], [0, -1, 1]])
    gradient = np.array([[1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [-0.5, 1.5, 8.0]])
    np.testing.assert_allclose(parameterization.pullback(gradient),
                               10 * (parameterization.differencing_matrix.T @ gradient).ravel(),
                               rtol=0, atol=4e-14)


def test_absolute_anchored_end_and_transpose_drop_terminal_residual():
    positions = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 0.0], [3.0, -1.0, -2.0], [8.0, 5.0, 6.0]])
    parameterization = AbsolutePositionResidual(positions, scale_mm=5.0, anchor_end=True)
    assert parameterization.parameter_shape == (2, 3)
    np.testing.assert_array_equal(parameterization.positions(np.zeros(6)), positions)
    theta = np.array([[0.2, -0.1, 0.5], [0.3, 0.4, -0.25]])
    shifted = positions.copy()
    shifted[1:3] += 5 * theta
    np.testing.assert_allclose(parameterization.positions(theta), shifted, rtol=0, atol=2e-15)
    np.testing.assert_allclose(parameterization.forward(theta), np.diff(shifted, axis=0), rtol=0, atol=2e-15)
    np.testing.assert_array_equal(parameterization.positions(theta)[[0, -1]], positions[[0, -1]])
    gradient = np.arange(9, dtype=np.float64).reshape(3, 3) / 4
    np.testing.assert_allclose(parameterization.pullback(gradient),
                               (5 * parameterization.differencing_matrix.T @ gradient)[:-1].ravel(),
                               rtol=0, atol=5e-15)


@pytest.mark.parametrize("parameterization", make_cases(), ids=lambda case: f"{case.mode}-anchor{case.anchor_end}")
def test_pullback_matches_central_finite_differences(parameterization):
    rng = np.random.default_rng(998)
    theta = rng.uniform(-0.15, 0.15, size=parameterization.parameter_shape)
    weights = rng.uniform(0.3, 1.3, size=(parameterization.num_steps, 3))
    coupling = rng.uniform(-0.2, 0.2, size=(parameterization.num_steps, 3))

    def objective(values):
        deltas = parameterization.forward(values)
        return float(np.sum(weights * deltas**2) + np.sum(coupling * np.sin(deltas)))

    deltas = parameterization.forward(theta)
    grad_deltas = 2 * weights * deltas + coupling * np.cos(deltas)
    analytic = parameterization.pullback(grad_deltas)
    numerical = np.empty(parameterization.num_params)
    epsilon = 1e-6
    flat = theta.ravel()
    for index in range(parameterization.num_params):
        plus = flat.copy()
        minus = flat.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (objective(plus) - objective(minus)) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-8, atol=3e-8)


@pytest.mark.parametrize("parameterization", make_cases(), ids=lambda case: f"{case.mode}-anchor{case.anchor_end}")
def test_transpose_adjoint_identity(parameterization):
    theta = RNG.normal(scale=0.1, size=parameterization.parameter_shape)
    gradient = RNG.normal(size=(parameterization.num_steps, 3))
    tangent = parameterization.forward(theta) - parameterization.forward(np.zeros(parameterization.num_params))
    adjoint = parameterization.pullback(gradient)
    np.testing.assert_allclose(np.vdot(gradient, tangent), np.vdot(adjoint, theta.ravel()),
                               rtol=3e-14, atol=2e-12)


def test_bounds_are_shared_explicit_and_projection_is_not_silent():
    delta = np.zeros((2, 3))
    parameterization = DeltaResidual(delta, scale_mm=1.0, bounds=(-0.5, 0.25))
    lower, upper = parameterization.bounds
    np.testing.assert_array_equal(lower, np.full(6, -0.5))
    np.testing.assert_array_equal(upper, np.full(6, 0.25))
    assert not lower.flags.writeable and not upper.flags.writeable
    assert parameterization.in_bounds(np.zeros(6))
    assert not parameterization.in_bounds(np.full(6, 0.3))
    np.testing.assert_array_equal(parameterization.clip(np.full((2, 3), 2.0)), np.full(6, 0.25))
    with pytest.raises(ValueError, match="outside declared normalized bounds"):
        parameterization.forward(np.full(6, 2.0))
    np.testing.assert_array_equal(parameterization.forward(parameterization.clip(np.full(6, 2.0))), np.full((2, 3), 0.25))


def test_per_parameter_bounds_accept_matrix_or_three_column_vector():
    parameterization = DeltaResidual(np.zeros((2, 3)), scale_mm=2,
                                     bounds=(np.full((2, 3), -1), np.array([1, 2, 3])))
    np.testing.assert_array_equal(parameterization.bounds[1], [1, 2, 3, 1, 2, 3])
    assert DeltaResidual(np.zeros((2, 3)), scale_mm=2, bounds=None).bounds is None
    assert DeltaResidual(np.zeros((2, 3)), scale_mm=2, bounds=None).in_bounds(np.full(6, 1e30))


@pytest.mark.parametrize("mode", ["raw", "spline", "absolute"])
def test_bound_array_length_uses_free_parameter_count(mode):
    if mode == "raw":
        parameterization = DeltaResidual(np.zeros((4, 3)), 1.0, anchor_end=True)
    elif mode == "spline":
        parameterization = DeltaSpline(np.zeros((5, 3)), 1.0, num_control_points=4, anchor_end=True)
    else:
        parameterization = AbsolutePositionResidual(np.zeros((4, 3)), 1.0, anchor_end=True)
    count = parameterization.num_params
    lower = -np.arange(1, count + 1, dtype=float)
    upper = np.arange(1, count + 1, dtype=float)
    if mode == "raw":
        parameterization = DeltaResidual(np.zeros((4, 3)), 1.0, bounds=(lower, upper), anchor_end=True)
    elif mode == "spline":
        parameterization = DeltaSpline(np.zeros((5, 3)), 1.0, num_control_points=4,
                                       bounds=(lower, upper), anchor_end=True)
    else:
        parameterization = AbsolutePositionResidual(np.zeros((4, 3)), 1.0, bounds=(lower, upper), anchor_end=True)
    np.testing.assert_array_equal(parameterization.bounds[0], lower)
    np.testing.assert_array_equal(parameterization.bounds[1], upper)
    assert parameterization.in_bounds(lower)


@pytest.mark.parametrize("steps,controls,match", [
    (4, 3, "K"), (4, 5, "K"), (2, 4, "K"), (4, 4.0, "K"), (0, 4, "T"),
])
def test_invalid_k_t_are_rejected(steps, controls, match):
    with pytest.raises(ValueError, match=match):
        cubic_open_uniform_basis(steps, controls)


@pytest.mark.parametrize("mode", ["delta", "spline", "absolute"])
def test_one_segment_anchor_is_rejected_but_free_is_valid(mode):
    delta = np.ones((1, 3))
    if mode == "delta":
        with pytest.raises(ValueError, match="T >= 2"):
            DeltaResidual(delta, 1, anchor_end=True)
        assert DeltaResidual(delta, 1).forward(np.zeros(3)).shape == (1, 3)
    elif mode == "absolute":
        with pytest.raises(ValueError, match="T >= 2"):
            AbsolutePositionResidual(np.vstack((np.zeros(3), delta)), 1, anchor_end=True)
        assert AbsolutePositionResidual(np.vstack((np.zeros(3), delta)), 1).forward(np.zeros(3)).shape == (1, 3)
    else:
        # A cubic spline requires K>=4 anyway; this is a clear K/T validation.
        with pytest.raises(ValueError, match="K"):
            DeltaSpline(delta, 1, num_control_points=1, anchor_end=True)


@pytest.mark.parametrize("bad_scale", [0, -1, np.inf, np.nan])
def test_scale_validation(bad_scale):
    with pytest.raises(ValueError, match="scale_mm"):
        DeltaResidual(np.zeros((2, 3)), scale_mm=bad_scale)


@pytest.mark.parametrize("bad_bounds,match", [
    ((-1, 1, 2), "bounds"),
    ((1, -1), "lower"),
    ((0.5, 1), "theta=0"),
    ((np.nan, 1), "finite"),
    ((-np.inf, 1), "finite"),
    (([0, 0], 1), "6 entries"),
])
def test_bounds_validation(bad_bounds, match):
    with pytest.raises(ValueError, match=match):
        DeltaResidual(np.zeros((2, 3)), scale_mm=1, bounds=bad_bounds)


def test_bad_shapes_and_nonfinite_inputs_are_actionable():
    with pytest.raises(ValueError, match="expected shape"):
        DeltaResidual(np.zeros((2, 2)), 1)
    with pytest.raises(ValueError, match="finite"):
        DeltaResidual(np.array([[np.nan, 0, 0]]), 1)
    with pytest.raises(ValueError, match="at least P0 and P1"):
        AbsolutePositionResidual(np.zeros((1, 3)), 1)
    parameterization = DeltaResidual(np.zeros((2, 3)), 1)
    with pytest.raises(ValueError, match="theta"):
        parameterization.forward(np.zeros(5))
    with pytest.raises(ValueError, match="finite"):
        parameterization.forward(np.full(6, np.inf))
    with pytest.raises(ValueError, match="grad_deltas"):
        parameterization.pullback(np.zeros((3, 3)))
