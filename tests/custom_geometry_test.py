"""T2.0 sampled GeometrySource integration with physical grids and gradients."""

from pathlib import Path

import numpy as np
import pytest
import taichi as ti

from benchmarks import contracts as C
from simulator.csg_simulator import CSGSimulatorDelta


@pytest.fixture(scope="session", autouse=True)
def taichi_cpu_once():
    ti.init(arch=ti.cpu, offline_cache=False)


@ti.kernel
def target_value(sim: ti.template(), px: ti.f32, py: ti.f32, pz: ti.f32, scalar: ti.template()):
    p = ti.Vector([px, py, pz])
    if ti.static(scalar):
        sim.loss[None] += sim.target_sdf_scalar(p)
    else:
        sim.loss[None] += sim.target_sdf(p)


def point_value(sim, point, scalar=True):
    sim.loss[None] = 0.0
    target_value(sim, *point, scalar=scalar)
    return float(sim.loss[None])


@ti.kernel
def midpoint_loss(sim: ti.template()):
    mx = 0.5 * (sim.tool_pos[0].x + sim.tool_pos[1].x)
    my = 0.5 * (sim.tool_pos[0].y + sim.tool_pos[1].y)
    mz = 0.5 * (sim.tool_pos[0].z + sim.tool_pos[1].z)
    sim.loss[None] += sim.target_sdf_scalar(ti.Vector([mx, my, mz]))


def midpoint_value(sim, delta, tape):
    sim.tool_delta.from_numpy(delta)
    sim.loss[None] = 0.0
    if tape:
        with ti.ad.Tape(loss=sim.loss):
            sim.reconstruct_positions(len(delta))
            midpoint_loss(sim)
    else:
        sim.reconstruct_positions(len(delta))
        midpoint_loss(sim)
    return float(sim.loss[None])


def regular_axes(shape, origin, voxel):
    return [origin[a] + (np.arange(shape[a]) + 0.5) * voxel for a in range(3)]


def write_source(tmp_path, values, origin, voxel, name="field.npy"):
    values = np.asarray(values, dtype=np.float64)
    path = Path(tmp_path) / name
    np.save(path, values)
    grid = C.GridSpec(shape=values.shape, origin_mm=origin, voxel_mm=voxel)
    return C.GeometrySource(
        kind="sampled", path=str(path), sha256=C.sha256_bytes(path.read_bytes()), grid=grid
    )


def linear_field(shape, origin, voxel, coefficients, bias=0.0):
    xyz = np.meshgrid(*regular_axes(shape, origin, voxel), indexing="ij")
    return sum(coefficients[a] * xyz[a] for a in range(3)) - bias


def new_sim(**kwargs):
    kwargs.setdefault("target_shape", "sphere")
    kwargs.setdefault("max_steps", 1)
    kwargs.setdefault("enforce_speed_limits", False)
    kwargs.setdefault("init_taichi", False)
    return CSGSimulatorDelta(**kwargs)


def test_physical_trilinear_and_clamped_out_of_grid_values(tmp_path):
    origin = (1.0, -2.0, 0.5)
    coefficients = (2.0, -0.5, 1.0)
    values = linear_field((3, 3, 3), origin, 2.0, coefficients, bias=3.0)
    source = write_source(tmp_path, values, origin, 2.0)
    sim = new_sim(
        voxel_size_mm=2.0, stock_size_mm=(6.0, 6.0, 6.0),
        stock_box_origin_mm=origin, target_geometry=source,
    )
    cases = [
        # (1, .5, .5) normalized -> physical (2.5, 1.0, 3.5): 5/2 voxel.
        ((0.25, 0.5, 0.5), 2.5),
        # Far-left x=-5 clamps to the first x centre (source x=2), not extrapolation.
        # Far-left x=-5 clamps to source x=2: (4-0.5+3.5-3)/2 = 2.
        ((-1.0, 0.5, 0.5), 2.0),
        # Far-right x=8.2 clamps to source x=6: (12-0.5+3.5-3)/2 = 6.
        ((1.2, 0.5, 0.5), 6.0),
    ]
    for point, expected in cases:
        actual = point_value(sim, point, scalar=True)
        assert actual == pytest.approx(expected, abs=2e-6)
        assert point_value(sim, point, scalar=False) == pytest.approx(actual, abs=2e-7)
    sim.bake_target_grid()
    sim.set_target_volume()
    grid = sim.target.to_numpy()
    assert sim.target_volume[None] == pytest.approx(float(np.count_nonzero(grid < 0.0)) / grid.size)


def test_custom_non_box_stock_is_not_replaced_by_full_box(tmp_path):
    origin = (-10.0, 20.0, -1.0)
    centre = np.array([-4.0, 24.0, 2.0])
    shape, voxel = (6, 4, 3), 2.0
    xyz = np.meshgrid(*regular_axes(shape, origin, voxel), indexing="ij")
    radius = 3.0
    values = np.sqrt(sum((xyz[a] - centre[a]) ** 2 for a in range(3))) - radius
    source = write_source(tmp_path, values, origin, voxel)
    sim = new_sim(
        voxel_size_mm=voxel, stock_geometry=source, target_geometry=source,
        tool_start=(0.5, 0.5, 0.5), max_steps=1,
    )
    # Large stationary cutter opens the centre cells through the sphere boundary.
    sim.tool_radius[None] = 3.0
    sim.tool_height[None] = 1.0
    sim.init_stock()
    initial = sim.stock.to_numpy()[0]
    np.testing.assert_allclose(initial, values.astype(np.float32) / voxel, atol=2e-6, rtol=0.0)
    assert np.any(initial < 0.0)
    assert np.any(initial >= 0.0)  # non-box outside corners are not fabricated stock
    sim.tool_delta.from_numpy(np.zeros((1, 3), dtype=np.float32))
    sim.forward_hard(2, clip_speeds=False)
    remaining = sim.stock.to_numpy()[1]
    removed = (initial < 0.0) & ~(remaining < 0.0)
    assert np.any(removed)
    # A far corner outside the analytic sphere stays outside/remains intact.
    assert remaining[-1, -1, -1] >= 0.0


def sphere_source(tmp_path, voxel):
    side = 20.0
    shape = (int(round(side / voxel)),) * 3
    origin = (0.0, 0.0, 0.0)
    xyz = np.meshgrid(*regular_axes(shape, origin, voxel), indexing="ij")
    values = np.sqrt(sum((xyz[a] - 10.0) ** 2 for a in range(3))) - 6.0
    return write_source(tmp_path, values, origin, voxel, name=f"sphere-{voxel}.npy")


def test_analytic_sphere_agreement_converges_with_sampling_resolution(tmp_path):
    analytic = new_sim(voxel_size_mm=0.625, stock_size_mm=20.0, resolution=32)
    analytic.set_target_params(radius_mm=6.0)
    analytic.bake_target_grid()
    reference = analytic.target.to_numpy()
    errors = []
    for source_voxel in (2.0, 1.0, 0.5):
        source = sphere_source(tmp_path, source_voxel)
        sampled = new_sim(
            voxel_size_mm=0.625, stock_size_mm=20.0,
            target_geometry=source,
        )
        sampled.bake_target_grid()
        # Both arrays are on the same physical runtime grid; compare mm SDFFs.
        assert sampled.target.to_numpy().shape == reference.shape
        error = float(np.sqrt(np.mean((sampled.target.to_numpy() - reference) ** 2))
                      * sampled.v)
        errors.append(error)
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


def test_changing_target_grid_changes_terminal_and_position_losses(tmp_path):
    origin = (0.0, 0.0, 0.0)
    # Affine fields make the intended comparison unambiguous: target x=X vs X+5.
    a = write_source(tmp_path, linear_field((2, 2, 2), origin, 2.0, (1.0, 0.0, 0.0)), origin, 2.0, "a.npy")
    b = write_source(tmp_path, linear_field((2, 2, 2), origin, 2.0, (1.0, 0.0, 0.0), bias=-5.0), origin, 2.0, "b.npy")
    delta = np.zeros((1, 3), dtype=np.float32)
    losses = []
    positions = []
    for source in (a, b):
        sim = new_sim(voxel_size_mm=2.0, stock_size_mm=4.0,
                      target_geometry=source)
        sim.w_tool_gouge[None] = 1.0
        sim.w_traj_prox[None] = 1.0
        sim.tool_radius[None] = 0.5
        sim.tool_delta.from_numpy(delta)
        sim.forward(2)
        losses.append(float(sim.loss[None]))
        sim.loss[None] = 0.0
        sim.compute_tool_gouge_penalty(0, 1)
        positions.append(float(sim.loss[None]))
        sim.loss[None] = 0.0
        sim.compute_traj_prox_penalty(0, 1)
        positions[-1] += float(sim.loss[None])
        # The complete position objective and diagnostic path must agree even
        # for the terminal (and here only) motion segment.
        sim.compute_diagnostics(1)
        diagnostic_total = float(sim.diag_tool_gouge[None] + sim.diag_traj_prox[None])
        assert diagnostic_total == pytest.approx(positions[-1], abs=2e-5)
        assert positions[-1] > 0.0
    assert losses[0] != pytest.approx(losses[1], abs=1e-5)
    assert positions[0] != pytest.approx(positions[1], abs=1e-5)


def test_inside_cell_gradients_match_analytic_plane_and_finite_difference(tmp_path):
    origin = (1.0, -2.0, 0.5)
    coefficients = (2.0, -0.5, 1.0)
    source = write_source(
        tmp_path, linear_field((3, 3, 3), origin, 2.0, coefficients), origin, 2.0
    )
    sim = new_sim(
        voxel_size_mm=2.0, stock_size_mm=(6.0, 6.0, 6.0),
        stock_box_origin_mm=origin, target_geometry=source,
    )
    sim.set_tool_start((0.5, 0.5, 0.5))
    delta = np.array([[0.2, 0.15, 0.1]], dtype=np.float32)
    midpoint_value(sim, delta, tape=True)
    analytic = sim.tool_delta.grad.to_numpy().copy()
    # Chain rule: coefficient/runtime_voxel * stock_extent/source_voxel / 2.
    # = coefficient * 6/2/2 here. Dividing by runtime voxel AGAIN would be wrong.
    np.testing.assert_allclose(analytic[0], [3.0, -0.75, 1.5], rtol=2e-5, atol=1e-6)
    for axis in range(3):
        for epsilon in (2e-3, 4e-3, 8e-3):
            values = []
            for sign in (1.0, -1.0):
                variant = delta.copy()
                variant[0, axis] += sign * epsilon
                values.append(midpoint_value(sim, variant, tape=False))
            numerical = (values[0] - values[1]) / (2.0 * epsilon)
            assert numerical == pytest.approx(analytic[0, axis], rel=1e-4, abs=1e-6)


def test_custom_reset_preserves_sources_and_analytic_switch_is_explicit(tmp_path):
    origin = (0.0, 0.0, 0.0)
    source = write_source(
        tmp_path, linear_field((2, 2, 2), origin, 2.0, (1.0, 0.0, 0.0)), origin, 2.0
    )
    delta = np.array([[0.02, 0.0, 0.0]], dtype=np.float32)
    sim = new_sim(voxel_size_mm=2.0, stock_size_mm=4.0,
                  target_geometry=source, stock_geometry=source)
    sim.tool_delta.from_numpy(delta)
    with ti.ad.Tape(loss=sim.loss):
        sim.forward(2)
    first = sim.stock.to_numpy()[:2].copy()
    first_loss = float(sim.loss[None])
    target_before = sim.target_source_metadata.copy()
    sim.bake_target_grid()
    baked_before = sim.target.to_numpy().copy()
    shape_before = sim.target_shape
    stock_before = sim.stock.to_numpy()[0].copy()
    sim.load_saved_init(np.ones_like(sim.saved_stock.to_numpy()), (0.9, 0.9, 0.9))
    sim.reset_candidate_state()
    assert sim.use_sampled_target[None] == 1 and sim.use_sampled_stock[None] == 1
    assert sim.target_source_metadata == target_before
    assert np.array_equal(sim.tool_start.to_numpy(), (0.5, 0.5, 1.0))
    sim.tool_delta.from_numpy(delta)
    with ti.ad.Tape(loss=sim.loss):
        sim.forward(2)
    np.testing.assert_allclose(first, sim.stock.to_numpy()[:2], rtol=2e-6, atol=2e-7)
    assert first_loss == pytest.approx(sim.loss[None], rel=2e-6, abs=2e-7)
    sim.bake_target_grid()
    assert np.array_equal(baked_before, sim.target.to_numpy())
    assert shape_before == "sampled"
    assert np.array_equal(stock_before, sim.stock.to_numpy()[0])

    # This is a deliberate analytic target change, not a fallback to an old custom field.
    sim.set_target_params(radius_mm=1.0)
    assert sim.use_sampled_target[None] == 0
    assert sim.target_shape == "sphere"
    expected = point_value(sim, (0.5, 0.5, 0.5), scalar=True)
    np_x = np.sqrt(3 * (0.5 - 0.5) ** 2) - 1.0 / sim.Nx
    assert expected == pytest.approx(np_x, abs=1e-6)


@pytest.mark.parametrize("defect", ["hash", "shape", "nan", "frame"])
def test_invalid_sampled_inputs_fail_without_hidden_geometry(tmp_path, defect):
    origin = (0.0, 0.0, 0.0)
    values = linear_field((2, 2, 2), origin, 2.0, (1.0, 0.0, 0.0))
    source = write_source(tmp_path, values, origin, 2.0)
    if defect == "hash":
        source = C.GeometrySource(
            kind="sampled", path=source.path, sha256="sha256:" + "0" * 64, grid=source.grid
        )
    elif defect == "shape":
        source = C.GeometrySource(
            kind="sampled", path=source.path, sha256=source.sha256,
            grid=C.GridSpec(shape=(3, 2, 2), origin_mm=origin, voxel_mm=2.0),
        )
    elif defect == "nan":
        values[0, 0, 0] = np.nan
        source = write_source(tmp_path, values, origin, 2.0)
    else:
        source = C.GeometrySource(
            kind="sampled", path=source.path, sha256=source.sha256,
            grid=C.GridSpec(shape=values.shape, origin_mm=origin, voxel_mm=2.0,
                            frame=C.Frame.MACHINE),
        )
    with pytest.raises((ValueError, TypeError, OSError)):
        new_sim(voxel_size_mm=2.0, stock_size_mm=4.0, target_geometry=source)


def test_custom_stock_bounds_mismatch_is_rejected(tmp_path):
    origin = (0.0, 0.0, 0.0)
    source = write_source(
        tmp_path, linear_field((2, 2, 2), origin, 2.0, (1.0, 0.0, 0.0)), origin, 2.0
    )
    with pytest.raises(ValueError, match="bounds mismatch"):
        new_sim(voxel_size_mm=2.0, stock_size_mm=(4.0, 4.0, 8.0), stock_geometry=source)


def test_sampled_source_cannot_be_added_after_materialization(tmp_path):
    origin = (0.0, 0.0, 0.0)
    source = write_source(
        tmp_path, linear_field((1, 1, 1), origin, 4.0, (1.0, 0.0, 0.0)), origin, 4.0
    )
    sim = new_sim(voxel_size_mm=4.0, stock_size_mm=4.0)
    with pytest.raises(RuntimeError, match="declared during __init__"):
        sim.set_custom_target(source)
    assert sim.use_sampled_target[None] == 0
    assert sim.target_source_metadata is None
