"""T1.1 production hard-sweep and smooth-gradient validation.

Reference geometry runs only in this TEST module. Expected analytic volumes and
all physical grids come from T1.0's independent fixtures; production imports no
verification code. Float32 occupancy equality is asserted on declared general
position grids, not on every mathematically coincident boundary projection.
"""

from functools import lru_cache

import numpy as np
import pytest
import taichi as ti

from simulator.csg_simulator import CSGSimulatorDelta
from verification import fixtures as F
from verification.reference_carve import run_reference


STOCK_LO = np.array(F.B_MIN, dtype=np.float64)
STOCK_SIZE = np.array(F.B_MAX, dtype=np.float64) - STOCK_LO
MAX_TEST_STEPS = 8


@pytest.fixture(scope="session", autouse=True)
def taichi_cpu_once():
    ti.init(arch=ti.cpu, offline_cache=False)


@lru_cache(maxsize=4)
def shared_sim(voxel_mm):
    # Reusing fields avoids repeated Taichi allocation; init happens once above.
    return CSGSimulatorDelta(
        voxel_size_mm=voxel_mm,
        stock_size_mm=STOCK_SIZE,
        max_steps=MAX_TEST_STEPS,
        target_shape="sphere",
        enforce_speed_limits=False,
        init_taichi=False,
    )


def production_occupancy(fixture, path_mm=None):
    motion = fixture.motion
    if path_mm is None:
        path_mm = np.asarray(
            [motion.initial_state.position_mm]
            + [segment.end_mm for segment in motion.segments],
            dtype=np.float64,
        )
    else:
        path_mm = np.asarray(path_mm, dtype=np.float64)
    assert 1 <= len(path_mm) <= MAX_TEST_STEPS + 1
    normalized = (path_mm - STOCK_LO) / STOCK_SIZE
    sim = shared_sim(fixture.task.grid.voxel_mm)
    assert sim.stock.shape[1:] == fixture.task.grid.shape
    sim.tool_start[None] = ti.Vector(normalized[0].tolist())
    deltas = np.zeros((MAX_TEST_STEPS, 3), dtype=np.float32)
    deltas[: len(path_mm) - 1] = np.diff(normalized, axis=0)
    sim.tool_delta.from_numpy(deltas)
    sim.tool_radius[None] = fixture.task.tool.cutter_radius_mm
    sim.tool_height[None] = fixture.task.tool.flute_length_mm
    sim.forward_hard(len(path_mm), clip_speeds=False)
    stock = sim.stock.to_numpy()
    initial = stock[0] < 0.0
    remaining = stock[len(path_mm) - 1] < 0.0
    # Reconstructed endpoint must not drift beyond the documented float32
    # conversion/accumulation tolerance. No geometric boundary tolerance is
    # hidden in the occupancy comparison itself.
    np.testing.assert_allclose(
        sim.tool_pos.to_numpy()[len(path_mm) - 1], normalized[-1],
        atol=2e-6, rtol=0.0,
    )
    return initial, initial & ~remaining


def volume(fixture, removed):
    return float(np.count_nonzero(removed) * fixture.task.grid.voxel_mm ** 3)


@pytest.mark.parametrize(
    "factory",
    [F.empty_preengaged, F.plunge_above, F.plunge_preengaged,
     F.through_hole, F.zero_length_feed],
    ids=lambda factory: factory().fixture_id,
)
def test_vertical_degenerate_and_empty_cases_match_independent_occupancy(factory):
    fixture = factory()
    _, removed = production_occupancy(fixture)
    trace = run_reference(
        fixture.task, fixture.motion,
        {"subdivisions": 64, "compute_bounds": False},
    )
    assert np.array_equal(removed, trace.removed)
    assert volume(fixture, removed) == pytest.approx(
        trace.result.metrics["removed_volume_mm3"].value
    )
    if fixture.fixture_id == "empty_preengaged":
        assert not removed.any()
    elif fixture.fixture_id == "through_hole":
        # Exact physical grid hand-check: centres (19,19,z) lie within radius 4
        # of the (20,20) plunge at every z=1,3,5,7,9,11 mm; no floor remains.
        assert removed[9, 9, :].all()


@pytest.mark.parametrize("factory", [F.horizontal_slot, F.sloped_ramp, F.near_vertical])
def test_reference_station_unions_converge_into_exact_production_sweep(factory):
    fixture = factory()
    _, removed = production_occupancy(fixture)
    previous = None
    previous_volume = -1.0
    errors = []
    for subdivisions in (1, 4, 16, 128):
        trace = run_reference(
            fixture.task, fixture.motion,
            {"subdivisions": subdivisions, "compute_bounds": False},
        )
        # Every static sampled pose is contained in the true continuous sweep.
        # Boundary points do not receive a false 100% convergence guarantee:
        # a boundary curve can require the limit of denser station projections.
        assert not np.any(trace.removed & ~removed)
        if previous is not None:
            assert not np.any(previous & ~trace.removed)
        sampled = trace.result.metrics["removed_volume_mm3"].value
        assert sampled >= previous_volume
        previous_volume = sampled
        previous = trace.removed.copy()
        errors.append(abs(volume(fixture, removed) - sampled))
    assert errors[-1] < errors[0]


@pytest.mark.parametrize(
    "factory,radius",
    [(F.plunge_above, 2.35), (F.horizontal_slot, 2.35),
     (F.sloped_ramp, 1.35), (F.near_vertical, 2.35),
     (F.through_hole, 3.35), (F.zero_length_feed, 2.35)],
)
def test_fractional_phase_radius_matches_dense_reference(factory, radius):
    fixture = factory(radius=radius, phase_xy=(0.26, 0.58))
    _, removed = production_occupancy(fixture)
    trace = run_reference(
        fixture.task, fixture.motion,
        {"subdivisions": 1024, "compute_bounds": False},
    )
    # Fractional phase avoids coincident boundary curves at these test radii.
    assert np.array_equal(removed, trace.removed)


@pytest.mark.parametrize(
    "fixture_id", ["plunge_above", "horizontal_slot", "sloped_ramp",
                   "near_vertical", "through_hole"]
)
def test_spatial_refinement_error_decreases_inside_certified_bracket(fixture_id):
    errors = []
    previous_width = float("inf")
    for voxel in (2.0, 1.0):
        fixture = F.with_voxel(F.fixture_by_id(fixture_id), voxel)
        _, removed = production_occupancy(fixture)
        sampled_volume = volume(fixture, removed)
        trace = run_reference(fixture.task, fixture.motion)
        bounds = trace.bounds
        assert bounds is not None
        # The strict cell brackets enclose the continuous analytic AND every
        # true-sweep centre estimate. No universal percentage tolerance.
        epsilon = 1e-8
        assert bounds.lower_mm3 - epsilon <= sampled_volume <= bounds.upper_mm3 + epsilon
        assert bounds.lower_mm3 - epsilon <= fixture.removed_volume_mm3 <= bounds.upper_mm3 + epsilon
        errors.append(abs(sampled_volume - fixture.removed_volume_mm3))
        width = bounds.upper_mm3 - bounds.lower_mm3
        assert width <= previous_width
        previous_width = width
    # Measured two-level refinement evidence for each fixture, not an asymptotic
    # assertion based on one coarse voxel count.
    assert errors[1] < errors[0]


@pytest.mark.parametrize("fixture_id", ["horizontal_slot", "sloped_ramp", "near_vertical"])
def test_production_contiguous_geometry_is_subdivision_invariant(fixture_id):
    fixture = F.fixture_by_id(fixture_id)
    start = np.asarray(fixture.motion.segments[0].start_mm)
    end = np.asarray(fixture.motion.segments[0].end_mm)
    _, one = production_occupancy(fixture, np.stack([start, end]))
    split_path = np.linspace(start, end, 5)
    _, split = production_occupancy(fixture, split_path)
    # General-position fixtures avoid float32 exact-on-gate ambiguity.
    assert np.array_equal(one, split)


@pytest.mark.parametrize("factory", [F.repeated_plunge, F.reversal])
def test_repeated_and_reversed_sweeps_are_set_unions(factory):
    fixture = factory()
    _, repeated = production_occupancy(fixture)
    if fixture.fixture_id == "repeated_plunge":
        single_fixture = F.plunge_above()
    else:
        single_fixture = F.make_fixture(
            "one_leg",
            [F.feed("outward", (12, 20, 0), (40, 20, 0))],
            F.flat_tool(3, 20),
            fixture.removed_volume_mm3,
            fixture.derivation,
        )
    _, single = production_occupancy(single_fixture)
    assert np.array_equal(repeated, single)


@pytest.mark.parametrize("motion_kind", ["sloped", "pure_vertical"])
def test_smooth_objective_gradient_matches_epsilon_sweep_away_from_gates(motion_kind):
    # Use a genuinely cubic 2 mm grid: stock (40,24,16) mm -> (20,12,8)
    # cells. The legacy resolution=16 test silently produced pitches
    # 2.5/2.4/2.666... mm, incompatible with the isotropic sweep contract.
    # A larger stock-relative displacement exercises Z interpolation and the
    # restricted interval; no probe epsilon reaches a segment or cap gate.
    sim = CSGSimulatorDelta(
        voxel_size_mm=2.0, stock_size_mm=(40.0, 24.0, 16.0),
        max_steps=1, target_shape="sphere", tool_start=(0.25, 0.5, 0.75),
        enforce_speed_limits=False, init_taichi=False, k_init=40,
    )
    sim.tool_radius[None] = 2.1
    sim.tool_height[None] = 7.0
    sim.set_target_params(radius_mm=5.0)
    delta = np.array([[0.3, 0.1, -0.35 if motion_kind == "sloped" else -0.35]],
                     dtype=np.float32)
    if motion_kind == "pure_vertical":
        delta[0, :2] = 0.0

    sim.tool_delta.from_numpy(delta)
    with ti.ad.Tape(loss=sim.loss):
        sim.forward(2)
    analytic = float(sim.tool_delta.grad.to_numpy()[0, 2])
    assert abs(analytic) > 1e-4

    numerical = []
    for epsilon in (2e-4, 5e-4, 1e-3):
        values = []
        for sign in (1.0, -1.0):
            perturbed = delta.copy()
            perturbed[0, 2] += sign * epsilon
            sim.tool_delta.from_numpy(perturbed)
            sim.loss[None] = 0.0
            sim.forward(2)
            values.append(float(sim.loss[None]))
        numerical.append((values[0] - values[1]) / (2.0 * epsilon))
    # Float32 member fields, smoothed CSG and finite differences: use explicit
    # absolute + relative checks, not a universal percentage/voxel-volume rule.
    np.testing.assert_allclose(numerical, analytic, rtol=0.08, atol=3e-4)
