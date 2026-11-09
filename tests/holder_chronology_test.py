"""T1.2 conservative holder chronology, candidate reset, and memory lifecycle.

Production imports the independent reference only in this test file. A positive
whole-segment pre-cut screen is explicitly NOT equated with a safety collision:
T1.0's time-resolved evaluation separately classifies the physical fixture.
"""

from dataclasses import replace
from functools import lru_cache
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import taichi as ti

from simulator.csg_simulator import CSGSimulatorDelta
from verification import fixtures as F
from verification.reference_carve import run_reference


@ti.kernel
def smooth_z_point_loss(sim: ti.template(), pz: ti.f32):
    sim.loss[None] += sim._tool_sweep_smooth_z(
        ti.Vector([0.58, 0.55, pz]), 0, sim.tool_height[None] / sim.v
    )


@ti.kernel
def hard_point_loss(sim: ti.template(), pz: ti.f32):
    sim.loss[None] = sim.tool_sdf_sharp(ti.Vector([0.5, 0.5, pz]), 0)


@ti.kernel
def smooth_tip_point_loss(sim: ti.template(), pz: ti.f32):
    sim.loss[None] += sim.tool_sdf_tip(ti.Vector([0.58, 0.55, pz]), 0)


@ti.kernel
def smooth_point_loss(sim: ti.template(), pz: ti.f32):
    # Direct scalar loss for the differentiable cylinder, not an aggregate
    # simulator objective that could hide kinks in one voxel/single component.
    sim.loss[None] += sim.tool_sdf(ti.Vector([0.58, 0.55, pz]), 0)


def evaluate_smooth_point(sim, delta, pz, with_tape):
    sim.loss[None] = 0.0
    sim.tool_delta.from_numpy(delta)
    if with_tape:
        with ti.ad.Tape(loss=sim.loss):
            sim.reconstruct_positions(1)
            smooth_point_loss(sim, pz)
    else:
        sim.reconstruct_positions(1)
        smooth_point_loss(sim, pz)
    return float(sim.loss[None])


@pytest.fixture(scope="session", autouse=True)
def taichi_cpu_once():
    ti.init(arch=ti.cpu, offline_cache=False)


@lru_cache(maxsize=4)
def column_sim(lo, hi, voxel):
    lo_array = np.asarray(lo, dtype=np.float64)
    hi_array = np.asarray(hi, dtype=np.float64)
    return CSGSimulatorDelta(
        voxel_size_mm=voxel,
        stock_size_mm=hi_array - lo_array,
        max_steps=16,
        target_shape="sphere",
        enforce_speed_limits=False,
        init_taichi=False,
    )


def screen_for(fixture):
    task, motion = fixture.task, fixture.motion
    lo = np.asarray(task.stock.params["min_mm"], dtype=np.float64)
    hi = np.asarray(task.stock.params["max_mm"], dtype=np.float64)
    points = np.asarray(
        [motion.initial_state.position_mm] + [s.end_mm for s in motion.segments],
        dtype=np.float64,
    )
    assert 1 < len(points) <= 17
    normalized = (points - lo) / (hi - lo)
    sim = column_sim(tuple(lo), tuple(hi), task.grid.voxel_mm)
    assert sim.stock.shape[1:] == task.grid.shape
    sim.tool_start[None] = ti.Vector(normalized[0].tolist())
    deltas = np.zeros((16, 3), dtype=np.float32)
    deltas[: len(points) - 1] = np.diff(normalized, axis=0)
    sim.tool_delta.from_numpy(deltas)
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm

    # Physical model here: cutter occupies base..base+4 mm. A flush holder
    # occupies base+4..base+24, radius 10 mm. There is no intermediate shank.
    sim.forward_hard(len(points), clip_speeds=False)
    segment_count = len(points) - 1
    legacy_volume = float(sim.holder_overlap_total(segment_count))
    physical_volume = float(sim.holder_overlap_total_mm3(segment_count))
    np.testing.assert_allclose(
        legacy_volume * sim.Nx * sim.Ny * sim.Nz * sim.v ** 3,
        physical_volume,
        rtol=2e-6,
        atol=2e-3,  # independent float32 volume reductions, not geometry tolerance
    )
    sim.loss[None] = 0.0
    sim.compute_holder_penalty(0, segment_count)
    # Check every active segment: a shallow first sub-segment can clear while
    # a later sub-segment genuinely penetrates. First-segment-only clearance
    # would miss that collision even though overlap_total sees it.
    clearance_vox = min(
        float(sim.holder_min_clearance_at(t)) for t in range(segment_count)
    )
    clearance_mm = min(
        float(sim.holder_min_clearance_mm_at(t)) for t in range(segment_count)
    )
    np.testing.assert_allclose(clearance_vox * sim.v, clearance_mm, rtol=0.0, atol=1e-6)
    return {
        "penalty": float(sim.loss[None]),
        "conservative_overlap_mm3": physical_volume,
        "conservative_clearance_mm": clearance_mm,
    }


def subdivided_fixture(source, subdivisions):
    assert len(source.motion.segments) == 1
    start = np.asarray(source.motion.segments[0].start_mm)
    end = np.asarray(source.motion.segments[0].end_mm)
    points = np.linspace(start, end, subdivisions + 1)
    segments = [
        F.feed(f"segment-{index}", points[index], points[index + 1])
        for index in range(subdivisions)
    ]
    return F.make_fixture(
        f"{source.fixture_id}-split-{subdivisions}", segments, source.task.tool,
        source.removed_volume_mm3, source.derivation,
        stock_lo=tuple(source.task.stock.params["min_mm"]),
        stock_hi=tuple(source.task.stock.params["max_mm"]),
        voxel=source.task.grid.voxel_mm,
    )


def test_cutter_leading_safe_screen_converges_while_reference_proves_safe():
    source = F.cutter_leading_holder_safe()
    records = []
    for subdivisions in (1, 2, 3, 4, 8, 16):
        fixture = subdivided_fixture(source, subdivisions)
        screen = screen_for(fixture)
        reference = run_reference(
            fixture.task, fixture.motion,
            {"subdivisions": subdivisions, "compute_bounds": False},
        )
        records.append((subdivisions, screen, reference))
        # The continuous exact cutter is the same under subdivision, sampled
        # aside from float32 boundary phase. Conservative holder overscreen is
        # what must decrease as pre-cut stock becomes progressively available.
        volumes = [
            record[1]["conservative_overlap_mm3"] for record in records
        ]
        assert volumes[-1] <= volumes[0] + 1e-6
    assert records[0][1]["conservative_overlap_mm3"] > 0.0
    assert records[0][1]["penalty"] > 0.0
    # At finer pre-cut refinement no conservative holder stock overlap remains;
    # the independent time-resolved evaluation also becomes definitively safe.
    assert records[-1][1]["conservative_overlap_mm3"] == 0.0
    # Smoothed CSG contact can retain a finite tangent residual even with no
    # hard overlap. Physical screen convergence is zero; the surrogate penalty
    # must decrease, not be misrepresented as an exact zero collision metric.
    assert records[-1][1]["penalty"] < records[0][1]["penalty"]
    assert records[-1][2].result.safety_pass is True
    assert records[-1][2].result.metrics["unique_contacted_volume_mm3"].value == 0.0
    # At least one coarse independent evaluation records the uncertainty/false
    # positive that the conservative training screen is deliberately avoiding
    # being mistaken for a genuine chronological collision.
    assert any(record[2].result.safety_pass is not True for record in records[:-1])


def test_genuine_holder_collision_remains_positive_under_refinement():
    source = F.genuine_holder_collision()
    for subdivisions in (1, 2, 3, 6, 12):
        fixture = subdivided_fixture(source, subdivisions)
        screen = screen_for(fixture)
        reference = run_reference(
            fixture.task, fixture.motion,
            {"subdivisions": subdivisions, "compute_bounds": False},
        )
        assert screen["conservative_overlap_mm3"] > 0.0
        assert screen["conservative_clearance_mm"] < 0.0
        assert screen["penalty"] > 0.0
        assert reference.result.safety_pass is False
        assert reference.result.metrics["unique_contacted_volume_mm3"].value > 0.0


def test_negative_holder_gradient_step_lowers_penetration_penalty():
    fixture = subdivided_fixture(F.cutter_leading_holder_safe(), 1)
    task = fixture.task
    lo = np.asarray(task.stock.params["min_mm"])
    hi = np.asarray(task.stock.params["max_mm"])
    sim = CSGSimulatorDelta(
        voxel_size_mm=task.grid.voxel_mm, stock_size_mm=hi - lo,
        max_steps=1, target_shape="sphere", tool_start=(0.5, 0.5, 14 / 12),
        enforce_speed_limits=False, init_taichi=False,
    )
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    delta = np.array([[0.0, 0.0, -9.0 / 12.0]], dtype=np.float32)

    def value_and_gradient(with_tape):
        sim.loss[None] = 0.0
        sim.tool_delta.from_numpy(delta)
        if with_tape:
            with ti.ad.Tape(loss=sim.loss):
                sim.reconstruct_positions(1)
                sim.init_stock()
                sim.apply_cut(0)
                sim.compute_holder_penalty(0, 1)
        else:
            sim.reconstruct_positions(1)
            sim.init_stock()
            sim.apply_cut(0)
            sim.compute_holder_penalty(0, 1)
        return float(sim.loss[None])

    before = value_and_gradient(True)
    gradient = sim.tool_delta.grad.to_numpy().copy()
    assert np.isfinite(gradient).all()
    norm = np.linalg.norm(gradient)
    assert norm > 1e-8
    # Step in the measured descent direction, not one chosen by an assumed sign.
    delta[:] -= (0.001 / norm) * gradient
    after = value_and_gradient(False)
    assert before > 1e-6
    assert after < before


def test_reset_candidate_state_preserves_config_and_ab_a_is_deterministic():
    sim = CSGSimulatorDelta(
        resolution=10, max_steps=2, stock_size_in=(1, 1, 1),
        target_shape="sphere", tool_start=(0.2, 0.25, 0.5),
        enforce_speed_limits=False, init_taichi=False, k_init=40,
    )
    sim.tool_radius[None] = 3.175
    sim.tool_height[None] = 25.0
    sim.set_target_params(radius_mm=9.0)
    sim.bake_target_grid()
    sim.w_time[None] = 0.3
    sim.w_air_time[None] = 0.2
    sim.w_air_late[None] = 0.1
    sim.w_break[None] = 0.01
    target_before = sim.target.to_numpy().copy()
    shape_before = sim.target_shape
    k_before = sim.k[None]
    rapid_before = sim.rapid_speed[None]
    speed_before = sim.feed_speed[None]
    start_before = sim.tool_start.to_numpy().copy()

    def evaluate(deltas):
        sim.tool_delta.from_numpy(deltas)
        with ti.ad.Tape(loss=sim.loss):
            sim.forward(len(deltas) + 1)
        sim.compute_diagnostics(len(deltas))
        return (
            sim.stock.to_numpy()[: len(deltas) + 1].copy(),
            sim.stock.grad.to_numpy()[: len(deltas) + 1].copy(),
            sim.tool_pos.to_numpy()[: len(deltas) + 1].copy(),
            sim.tool_delta.grad.to_numpy().copy(),
            float(sim.loss[None]),
            float(sim.loss.grad[None]),
            {name: float(getattr(sim, name)[None]) for name in (
                "diag_holder", "diag_air", "diag_gouge", "diag_residual",
            )},
            sim.seg_time.to_numpy().copy(),
            sim.seg_time.grad.to_numpy().copy(),
            float(sim.acc_psum[None]),
        )

    a = np.array([[0.08, 0.02, -0.01], [0.10, 0.01, 0.0]], dtype=np.float32)
    b = np.array([[-0.12, 0.10, -0.02], [0.02, -0.09, -0.04]], dtype=np.float32)
    first = evaluate(a)
    assert first[4] > 0.0
    sim.saved_stock.from_numpy(np.ones_like(sim.saved_stock.to_numpy()))
    sim.use_saved_init[None] = 1
    sim.loss.grad[None] = 123.0
    sim.reset_candidate_state()
    assert not saved_stock_nonzero(sim)
    assert sim.use_saved_init[None] == 0
    assert sim.loss[None] == 0.0 and sim.loss.grad[None] == 0.0
    assert not np.any(sim.stock.to_numpy()) and not np.any(sim.stock.grad.to_numpy())
    assert not np.any(sim.tool_pos.to_numpy()) and not np.any(sim.tool_pos.grad.to_numpy())
    assert not np.any(sim.tool_delta.to_numpy())
    assert not np.any(sim.tool_delta.grad.to_numpy())
    assert not np.any(sim.seg_time.to_numpy()) and not np.any(sim.seg_time.grad.to_numpy())
    assert not np.any(sim.seg_air.to_numpy()) and not np.any(sim.seg_air.grad.to_numpy())
    assert not np.any(sim.seg_swept.to_numpy()) and not np.any(sim.seg_swept.grad.to_numpy())
    assert not np.any(sim.seg_engage.to_numpy()) and not np.any(sim.seg_engage.grad.to_numpy())
    assert sim.acc_psum[None] == 0.0 and sim.acc_psum.grad[None] == 0.0
    evaluate(b)
    sim.reset_candidate_state()
    again = evaluate(a)

    # Small controlled-fixture tolerance: Taichi CPU scalar reductions can
    # change one float32 ULP across allocation/reinitialization histories.
    # This is the declared A/B/A tolerance, not hidden bitwise placebo.
    np.testing.assert_allclose(first[0], again[0], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(first[1], again[1], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(first[2], again[2], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(first[3], again[3], rtol=2e-6, atol=2e-7)
    assert first[4] == pytest.approx(again[4], rel=2e-6, abs=2e-7)
    assert first[5] == pytest.approx(again[5], rel=2e-6, abs=2e-7)
    assert first[6].keys() == again[6].keys()
    for name in first[6]:
        assert first[6][name] == pytest.approx(again[6][name], rel=2e-6, abs=2e-7)
    np.testing.assert_allclose(first[7], again[7], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(first[8], again[8], rtol=2e-6, atol=2e-7)
    assert first[9] == pytest.approx(again[9], rel=2e-6, abs=2e-7)
    assert sim.target_shape == shape_before
    assert sim.k[None] == k_before
    assert sim.rapid_speed[None] == rapid_before
    assert sim.feed_speed[None] == speed_before
    assert np.array_equal(sim.tool_start.to_numpy(), start_before)
    assert np.array_equal(sim.target.to_numpy(), target_before)


def saved_stock_nonzero(sim):
    return bool(np.any(sim.saved_stock.to_numpy()))


def test_taichi_cache_limit_raises_clear_error_instead_of_segfault():
    code = textwrap.dedent(
        """
        import numpy as np
        import taichi as ti
        from simulator.csg_simulator import CSGSimulatorDelta

        ti.init(arch=ti.cpu, offline_cache=False)
        for index in range(20):
            try:
                sim = CSGSimulatorDelta(
                    resolution=6, stock_size_in=(1, 1, 1), max_steps=1,
                    target_shape="sphere", enforce_speed_limits=False,
                    init_taichi=False,
                )
                sim.tool_radius[None] = 3.175
                sim.tool_height[None] = 25.0
                sim.tool_delta.from_numpy(np.zeros((1, 3), dtype=np.float32))
                sim.forward_hard(2, clip_speeds=False)
            except RuntimeError as error:
                assert "simulator instance lifecycle budget" in str(error), str(error)
                assert "fresh process" in str(error), str(error)
                print(f"GUARDED_AT_INDEX={index}")
                break
        else:
            raise AssertionError("repeated construction did not reach the guarded cache budget")
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True,
        check=False, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARDED_AT_INDEX=" in result.stdout


def test_noncutting_shank_is_not_invented_by_the_simulator():
    contract_tool = replace(
        F.flat_tool(2.0, 4.0, holder_radius=4.0),
        shank_radius_mm=4.0,
        holder_bottom_mm=8.0,
    )
    assert contract_tool.flute_length_mm < contract_tool.holder_bottom_mm
    fixture = subdivided_fixture(F.genuine_holder_collision(), 1)
    sim = column_sim(
        tuple(fixture.task.stock.params["min_mm"]),
        tuple(fixture.task.stock.params["max_mm"]),
        fixture.task.grid.voxel_mm,
    )
    assert not hasattr(sim, "shank_radius")
    assert not hasattr(sim, "holder_bottom")
    # The modeled assembly begins the holder directly above the cutting flute.
    # A separate contract shank gap must be evaluated by T1.3/reference instead.


def test_anisotropic_rounded_grid_is_rejected_before_allocation():
    with pytest.raises(ValueError, match="anisotropic"):
        CSGSimulatorDelta(
            voxel_size_mm=3.0, stock_size_mm=(40.0, 24.0, 16.0),
            max_steps=1, target_shape="sphere", init_taichi=False,
        )


def test_near_vertical_interval_excludes_exterior_cell():
    # Independently restricted near-vertical interval: b_z=-1e-4, h=0.5,
    # q_z=0.499975=h-2.5e-5. With dz<0, the admissible interval is
    # lambda in [0, (h-q_z)/1e-4] = [0,1/4]. At q_x=1/2 the nearest
    # horizontal offset is 1/4 > r=1/10; vertical cover does not permit an
    # unrestricted XY capsule to erase this exterior cell. The values are
    # representable away from the float32 cap-rounding asymptote.
    sim = CSGSimulatorDelta(
        resolution=1, stock_size_mm=1.0, max_steps=1,
        tool_start=(0.0, 0.5, 0.0), target_shape="sphere",
        enforce_speed_limits=False, init_taichi=False,
    )
    sim.tool_radius[None] = 0.1
    sim.tool_height[None] = 0.5
    sim.tool_delta.from_numpy(np.array([[0.5, 0.0, -1e-4]], dtype=np.float32))
    sim.reconstruct_positions(1)
    sim.loss[None] = 0.0
    hard_point_loss(sim, 0.499975)
    assert float(sim.loss[None]) > 0.0
    sim.init_stock()
    before = sim.stock.to_numpy()[0, 0, 0, 0].copy()
    sim.forward_hard(2, clip_speeds=False)
    after = sim.stock.to_numpy()[1, 0, 0, 0]
    # Hard membership is evaluated by occupancy, not by demanding the same
    # approximated outside SDF value. No stock may transition out of occupancy.
    assert before < 0.0
    assert after < 0.0


def smooth_probe_sim(tool_start, tool_height, tool_radius):
    sim = CSGSimulatorDelta(
        voxel_size_mm=1.0, stock_size_mm=(1.0, 1.0, 8.0), max_steps=1,
        tool_start=tool_start, target_shape="sphere",
        enforce_speed_limits=False, init_taichi=False, k_init=40,
    )
    sim.tool_radius[None] = tool_radius
    sim.tool_height[None] = tool_height
    return sim


def test_smooth_z_cylinder_centre_has_zero_autodiff_kink_error():
    sim = smooth_probe_sim((0.5, 0.5, 0.0), 5.0, 0.1)
    delta = np.zeros((1, 3), dtype=np.float32)
    # Nz=8, height=5, pz=2.5/8: the exact cylinder-centre Z query.
    pz = 2.5 / 8.0
    sim.loss[None] = 0.0
    with ti.ad.Tape(loss=sim.loss):
        sim.reconstruct_positions(1)
        smooth_z_point_loss(sim, pz)
    gradient = sim.tool_delta.grad.to_numpy().copy()
    assert np.isfinite(gradient).all()
    # The centred smooth norm is differentiable at this centre; its true
    # derivative is zero. This is a DIRECT Z probe, not the unrelated hard-XY
    # branch of max(horizontal,vertical) that dominates a full-cylinder probe.
    assert np.max(np.abs(gradient)) < 1e-5
    for sign in (1.0, -1.0):
        perturbed = delta.copy()
        perturbed[0, 2] += sign * 0.001
        actual = evaluate_smooth_point(sim, perturbed, pz, with_tape=False)
        # Also compare a completely independent float64 evaluation of the
        # declared centred norm using actual reconstructed Taichi positions.
        positions = sim.tool_pos.to_numpy()[:2] * np.asarray([sim.Nx, sim.Ny, sim.Nz])
        a, b = positions[0, 2], positions[1, 2]
        centre = 0.5 * (a + b + 5.0)
        extent = 5.0 + np.sqrt((b - a) ** 2 + 1e-8)
        expected = np.sqrt((pz * sim.Nz - centre) ** 2 + 1e-8) - 0.5 * extent
        # Reestablish the direct-Z result after the full-cylinder helper call.
        sim.loss[None] = 0.0
        smooth_z_point_loss(sim, pz)
        direct_z = float(sim.loss[None])
        assert direct_z == pytest.approx(expected, abs=2e-4)
        assert np.isfinite(actual)


def test_smooth_sweep_xyz_components_match_finite_differences():
    sim = smooth_probe_sim((0.5, 0.5, 0.0125), 5.0, 0.1)
    delta = np.array([[0.2, 0.15, 0.00625]], dtype=np.float32)
    pz = 0.4
    evaluate_smooth_point(sim, delta, pz, with_tape=True)
    analytic = sim.tool_delta.grad.to_numpy().copy()
    assert np.isfinite(analytic).all()
    for axis in range(3):
        assert abs(analytic[0, axis]) > 1e-6
        for epsilon in (5e-4, 1e-3, 2e-3):
            values = []
            for sign in (1.0, -1.0):
                perturbed = delta.copy()
                perturbed[0, axis] += sign * epsilon
                values.append(evaluate_smooth_point(sim, perturbed, pz, with_tape=False))
            numerical = (values[0] - values[1]) / (2.0 * epsilon)
            assert numerical == pytest.approx(analytic[0, axis], rel=0.06, abs=2e-5)


def test_tip_tangent_removal_is_engagement_not_air():
    sim = CSGSimulatorDelta(
        resolution=1, stock_size_mm=1.0, max_steps=1,
        tool_start=(-0.2, 0.5, 0.5), target_shape="sphere",
        enforce_speed_limits=False, init_taichi=False,
    )
    sim.tool_radius[None] = 0.5
    sim.tool_height[None] = 1.0
    sim.tool_cut_height[None] = 1.0  # explicit 1 mm band, coincident flute here
    sim.tool_delta.from_numpy(np.array([[0.2, 0.0, -0.4]], dtype=np.float32))
    sim.forward_hard(2, clip_speeds=False)
    assert sim.stock.to_numpy()[1, 0, 0, 0] == pytest.approx(0.0, abs=0.0)
    sim.compute_traj_diagnostics_hard(1)
    assert sim.seg_engage[0] > 0.0
    assert sim.diag_air_time[None] == pytest.approx(0.0, abs=0.0)


def test_tip_band_height_is_not_silently_replaced_by_full_flute_height():
    sim = smooth_probe_sim((0.5, 0.5, 0.25), 8.0, 1.0)
    sim.tool_cut_height[None] = 2.0  # explicit 2 mm tip band, 8 mm flute
    sim.tool_cut_height[None] = 2.0  # explicit 2 mm tip band, 8 mm flute
    sim.loss[None] = 0.0
    smooth_tip_point_loss(sim, 0.5)
    narrow_band = float(sim.loss[None])
    sim.tool_cut_height[None] = 0.0  # fallback full 8 mm flute
    sim.loss[None] = 0.0
    smooth_tip_point_loss(sim, 0.5)
    full_flute = float(sim.loss[None])
    assert narrow_band > full_flute


def test_hard_holder_cell_inflation_detects_interior_contact():
    # One cubic millimetre cell at [0,1]^3. Holder centre x=.9, radius .1:
    # the holder covers x=.8..1.0 and intersects the cell interior by .9 mm.
    # Its centre starts .1 mm from the boundary, but the closest point at cell
    # centre x=.5 reads +.4 cell from the holder. A centre-only test misses it.
    sim = CSGSimulatorDelta(
        voxel_size_mm=1.0, stock_size_mm=1.0, max_steps=1,
        tool_start=(0.9, 0.5, 0.0), target_shape="sphere",
        enforce_speed_limits=False, init_taichi=False,
    )
    sim.tool_radius[None] = 0.01  # stationary tiny cutter does not remove cell
    sim.tool_height[None] = 0.25
    sim.holder_radius[None] = 0.1
    sim.holder_height[None] = 1.0
    sim.tool_delta.from_numpy(np.zeros((1, 3), dtype=np.float32))
    sim.forward_hard(2, clip_speeds=False)
    assert sim.stock.to_numpy()[1, 0, 0, 0] < 0.0
    assert sim.holder_overlap_at(0) > 0.0
    assert sim.holder_overlap_mm3_at(0) > 0.0
    assert sim.holder_overlap_total(1) > 0.0
    assert sim.holder_overlap_total_mm3(1) > 0.0
    assert sim.holder_min_clearance_at(0) < 0.0
    assert sim.holder_min_clearance_mm_at(0) < 0.0
    # The separate smooth objective is deliberately not the inflated hard metric.
    sim.loss[None] = 0.0
    sim.compute_holder_penalty(0, 1)
    assert sim.loss[None] == pytest.approx(0.0, abs=0.0)
