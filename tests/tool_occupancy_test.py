"""Regress the smooth occupancy sign against physical inside/outside tool membership."""
from pathlib import Path
import tempfile
import numpy as np
import pytest
import taichi as ti

from algorithms.run_artifacts import simulator_array
from benchmarks import contracts as C
from simulator.csg_simulator import CSGSimulatorDelta


def _sampled_material(tmp_path, *, clear_air=False):
    """A real, hash-bound sampled material field; every voxel is solid, not an analytic cube."""
    shape = (8, 8, 8)
    path = tmp_path / "stock-field.npy"
    if clear_air:
        values = np.full(shape, 50.0, dtype=np.float32)  # Empty space above the tool.
    else:
        values = np.full(shape, -50.0, dtype=np.float32)  # Full material.
    if path.exists():
        assert np.array_equal(np.load(path, allow_pickle=False), values)
    else:
        np.save(path, values, allow_pickle=False)
    grid = C.GridSpec(shape=shape, origin_mm=(0.0, 0.0, 0.0), voxel_mm=1.0)
    return C.GeometrySource(kind="sampled", path=str(path),
                            sha256=C.sha256_bytes(path.read_bytes()), grid=grid)


def _simulator(label, start_z):
    tmp_path = Path(tempfile.mkdtemp(prefix=f"diffcam-occ-{label}-"))
    source = _sampled_material(tmp_path, clear_air=(label == "above"))
    sim = CSGSimulatorDelta(
        resolution=8, max_steps=1, stock_size_mm=source.grid.extent_mm,
        stock_box_origin_mm=source.grid.origin_mm,
        voxel_size_mm=source.grid.voxel_mm, target_shape="box",
        stock_geometry=source, target_geometry=source,
        tool_start=(0.5, 0.5, start_z), enforce_speed_limits=False,
        init_taichi=False, precision="f64",
    )
    sim.tool_radius[None] = 6.0  # The shallow tip/swept footprint covers this small test grid.
    sim.tool_height[None] = 12.0
    sim.tool_cut_height[None] = 12.0
    sim.holder_radius[None] = 10.0
    sim.holder_height[None] = 40.0
    for name in ("w_gouge", "w_residual", "holder_penalty_weight", "w_air_time",
                 "w_air_late", "w_time", "w_prox", "w_break"):
        getattr(sim, name)[None] = 0.0
    sim.w_air[None] = 1.0
    sim.bake_target_grid()
    return sim


def _forward_metrics(sim):
    sim.reset_candidate_state()
    sim.tool_delta.fill(0)
    sim.forward(2)
    sim.zero_seg_volumes(0, 1)
    sim.compute_seg_volumes(0, 1)
    sim.compute_diagnostics(1)


def test_smooth_above_stock_and_inside_material_membership_and_penalty():
    """Hard fields also require an explicit compute_traj_diagnostics_hard call."""
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f64)
    above = _simulator("above", 0.5)
    above.tool_radius[None] = 1.0
    _forward_metrics(above)
    swept = float(above.seg_swept[0])
    air = float(above.seg_air[0])
    engaged = float(above.seg_engage[0])
    assert swept > 0.0
    assert engaged / swept < 0.005  # Smooth tools have small finite tails; don't demand exact zero.
    assert air/swept == pytest.approx(1.0, abs=0.003)
    assert above.loss[None] > 0.0
    assert above.diag_air_unweighted[None] == pytest.approx(air, abs=1e-9)
    assert above.diag_tool_swept[None] == pytest.approx(swept, abs=1e-9)
    above.compute_traj_diagnostics_hard(1)
    assert above.seg_air[0] == pytest.approx(above.seg_swept[0], abs=1e-12)
    assert above.diag_air_unweighted[None] == pytest.approx(air, abs=1e-9)
    assert above.diag_tool_swept[None] == pytest.approx(swept, abs=1e-9)
    assert above.seg_engage[0] == pytest.approx(0.0, abs=1e-12)
    assert above.diag_air_time[None] > 0.0

    inside = _simulator("inside", 0.5)
    inside.tool_radius[None] = 1.0  # Preserve stock after this representative in-material cut.
    _forward_metrics(inside)
    swept = float(inside.seg_swept[0])
    assert swept > 0.0
    assert inside.seg_engage[0] == pytest.approx(swept, abs=1e-8)
    assert inside.seg_air[0] == pytest.approx(0.0, abs=1e-9)
    # The separate legacy pre/post-stock contract intentionally remains intact:
    # compute_air_penalty can charge productive cutting via post-cut empty stock.
    assert inside.loss[None] > 0.0
    assert inside.diag_air_unweighted[None] == pytest.approx(0.0, abs=1e-9)
    inside.compute_traj_diagnostics_hard(1)
    assert inside.seg_engage[0] == pytest.approx(inside.seg_swept[0], abs=1e-12)
    assert inside.seg_air[0] == pytest.approx(0.0, abs=1e-12)
    assert inside.diag_air_time[None] == pytest.approx(0.0, abs=1e-12)


def test_hard_diagnostic_call_does_not_happen_in_forward_hard():
    """forward_hard only carves/reconstructs; callers must request hard trajectory diagnostics."""
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f64)
    sim = _simulator("hard-call", 2.0)
    assert sim.diag_time[None] == sim.diag_air_time[None] == 0.0
    sim.forward_hard(2, clip_speeds=False)
    assert sim.diag_time[None] == sim.diag_air_time[None] == 0.0
    sim.compute_traj_diagnostics_hard(1)
    assert sim.diag_time[None] > 0.0
    assert sim.diag_air_time[None] == pytest.approx(sim.diag_time[None])
    assert sim.seg_engage[0] == 0.0


def test_smooth_air_time_has_finite_derivative_at_clear_air_moves():
    """The clear-air branch exercises signed tool occupancy and differentiable pre-cut engagement."""
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f64)
    sim = _simulator("air-time-gradient", 2.0)
    sim.w_air[None] = 0.0
    sim.w_air_time[None] = 0.75

    def evaluate(z_delta, gradient=False):
        sim.reset_candidate_state()
        sim.tool_delta.from_numpy(simulator_array(sim, [[0.0, 0.0, z_delta]]))
        if gradient:
            with ti.ad.Tape(loss=sim.loss):
                sim.forward(2)
            return float(sim.tool_delta.grad.to_numpy()[0, 2])
        sim.forward(2)
        return float(sim.loss[None])

    theta = 0.005
    epsilon = 1e-5
    assert evaluate(theta) == pytest.approx(0.006948426242507034, rel=1e-6)
    ad = evaluate(theta, gradient=True)
    plus = evaluate(theta + epsilon)
    minus = evaluate(theta - epsilon)
    fd = (plus - minus) / (2 * epsilon)
    assert np.isfinite(ad) and np.isfinite(fd)
    assert abs(ad) > 1e-2 and abs(fd) > 1e-2
    assert ad == pytest.approx(fd, rel=1e-6, abs=1e-10)
