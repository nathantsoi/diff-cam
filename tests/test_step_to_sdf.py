import os
import tempfile
import numpy as np
import pytest
import taichi as ti

from utils.step_to_sdf import step_to_sdf
from simulator.csg_simulator import CSGSimulatorDelta
from cam_env.cam_env import CamEnvDiff

def test_physical_stock_grid_math():
    # The converter keeps physical dimensions: the grid spans a rectangular
    # stock that defaults to the part's bounding box plus a padding margin,
    # with cubic voxels of side v = longest_axis / resolution and per-axis
    # counts round(L / v).
    bbox_min = np.array([-10.0, -5.0, 0.0])
    bbox_max = np.array([10.0, 5.0, 30.0])
    resolution = 64
    padding = 0.05

    part_size = bbox_max - bbox_min                # [20, 10, 30] mm
    pad_mm = padding * np.max(part_size)           # 1.5 mm per side
    stock_size = part_size + 2.0 * pad_mm          # [23, 13, 33] mm
    v = np.max(stock_size) / resolution            # 33 / 64 mm per voxel
    N = np.maximum(1, np.round(stock_size / v).astype(int))

    # Longest axis gets exactly `resolution` voxels; the others follow the
    # aspect ratio, so each axis's physical span is within half a voxel.
    assert N[2] == resolution
    assert np.all(np.abs(N * v - stock_size) <= v / 2 + 1e-9)

    # Part centered in the stock: the grid origin sits `pad_mm` outside the
    # bbox min corner, and voxel centers stay strictly inside the stock.
    center = (bbox_min + bbox_max) / 2.0
    origin = center - stock_size / 2.0
    assert np.allclose(origin, bbox_min - pad_mm)
    xs = origin[0] + (np.arange(N[0]) + 0.5) * v
    assert xs[0] > origin[0] and xs[-1] < origin[0] + stock_size[0]

def _write_physical_npz(path, sdf_mm, stock_size_mm, voxel_size_mm):
    np.savez_compressed(
        path,
        sdf=sdf_mm.astype(np.float32),
        resolution=np.array(max(sdf_mm.shape), dtype=np.int32),
        padding=np.array(0.0, dtype=np.float32),
        voxel_size_mm=np.array(voxel_size_mm, dtype=np.float32),
        stock_size_mm=np.asarray(stock_size_mm, dtype=np.float32),
        grid_origin_mm=np.zeros(3, dtype=np.float32),
        bbox_min=np.zeros(3, dtype=np.float32),
        bbox_max=np.asarray(stock_size_mm, dtype=np.float32),
        sdf_units="mm",
        source_step="synthetic",
        num_solids=np.array(1, dtype=np.int32),
        sign_convention="negative_inside",
    )

def test_simulator_grid_target_physical_dims(tmp_path):
    # New-format NPZ (utils/step_to_sdf.py): anisotropic grid, SDF in mm, and
    # the stock box carried in the file. The simulator must pick up the
    # physical dimensions with NO stock_size_in from the caller.
    try:
        ti.init(arch=ti.cpu)
    except Exception:
        pass

    voxel_size_mm = 2.0
    stock_size_mm = np.array([32.0, 16.0, 48.0])
    Nx, Ny, Nz = 16, 8, 24

    # Synthetic target: sphere of radius 6 mm at the stock center, SDF in mm.
    xs = (np.arange(Nx) + 0.5) * voxel_size_mm
    ys = (np.arange(Ny) + 0.5) * voxel_size_mm
    zs = (np.arange(Nz) + 0.5) * voxel_size_mm
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1)
    sdf_mm = np.linalg.norm(centers - stock_size_mm / 2.0, axis=-1) - 6.0

    npz_path = os.path.join(tmp_path, "physical_target.npz")
    _write_physical_npz(npz_path, sdf_mm, stock_size_mm, voxel_size_mm)

    sim = CSGSimulatorDelta(
        max_steps=10,
        target_shape="grid",
        target_sdf_path=npz_path,
        # NOTE: no stock_size_in -- dimensions must come from the NPZ.
    )

    # Physical stock box, voxel size, and grid shape all from the NPZ.
    assert (sim.Lx, sim.Ly, sim.Lz) == tuple(stock_size_mm)
    assert sim.v == voxel_size_mm
    assert (sim.Nx, sim.Ny, sim.Nz) == (Nx, Ny, Nz)
    assert sim.resolution == Nz

    # The loaded target is converted mm -> voxels (matching bake_target_grid).
    target_loaded = sim.target.to_numpy()
    assert np.allclose(target_loaded, sdf_mm / voxel_size_mm, atol=1e-5)

    # Target volume fraction matches the analytic voxel count.
    sim.set_target_volume()
    expected_volume = np.sum(sdf_mm < 0) / float(Nx * Ny * Nz)
    assert np.isclose(sim.target_volume[None], expected_volume)

    # The env also needs no stock size for grid targets.
    env = CamEnvDiff(max_steps=10, target_shape="grid", target_sdf_path=npz_path)
    obs, info = env.reset()
    n_vox = Nx * Ny * Nz
    assert (env.Nx, env.Ny, env.Nz) == (Nx, Ny, Nz)
    assert obs.shape[0] == 3 + 2 + n_vox + n_vox
    target_obs = obs[-n_vox:]
    assert np.allclose(target_obs, (sdf_mm / voxel_size_mm).ravel(), atol=1e-5)

def test_simulator_grid_target_conflicting_stock_warns(tmp_path):
    # An explicit stock box that disagrees with the NPZ's is a geometry error:
    # the simulator warns and uses the NPZ's box.
    try:
        ti.init(arch=ti.cpu)
    except Exception:
        pass

    voxel_size_mm = 2.0
    stock_size_mm = np.array([16.0, 16.0, 16.0])
    sdf_mm = np.full((8, 8, 8), 1.0, dtype=np.float32)
    sdf_mm[3:5, 3:5, 3:5] = -1.0
    npz_path = os.path.join(tmp_path, "conflict_target.npz")
    _write_physical_npz(npz_path, sdf_mm, stock_size_mm, voxel_size_mm)

    with pytest.warns(UserWarning, match="conflicts"):
        sim = CSGSimulatorDelta(
            max_steps=10,
            target_shape="grid",
            target_sdf_path=npz_path,
            stock_size_in=(1.0, 1.0, 1.0),  # 25.4 mm cube != 16 mm cube
        )
    assert (sim.Lx, sim.Ly, sim.Lz) == tuple(stock_size_mm)

def test_simulator_rejects_npz_without_dimensions(tmp_path):
    # NPZs without physical dimensions (the old normalized-cube format) are
    # rejected outright: the grid's physical scale would be a guess.
    try:
        ti.init(arch=ti.cpu)
    except Exception:
        pass

    sdf = np.full((8, 8, 8), 1.0, dtype=np.float32)
    npz_path = os.path.join(tmp_path, "legacy_target.npz")
    np.savez_compressed(npz_path, sdf=sdf)

    with pytest.raises(ValueError, match="physical dimensions"):
        CSGSimulatorDelta(
            max_steps=10,
            target_shape="grid",
            target_sdf_path=npz_path,
            stock_size_in=(1.0, 1.0, 1.0),
        )

def test_simulator_grid_target_raw_array():
    # A raw target_sdf_array carries no metadata: the caller must supply the
    # stock box, and the SDF values are millimetres (same as the NPZ format).
    try:
        ti.init(arch=ti.cpu)
    except Exception:
        pass

    voxel_size_mm = 2.0
    stock_size_mm = (16.0, 16.0, 16.0)
    xs = (np.arange(8) + 0.5) * voxel_size_mm
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    sdf_mm = np.linalg.norm(np.stack([gx, gy, gz], -1) - 8.0, axis=-1) - 5.0

    sim = CSGSimulatorDelta(
        max_steps=10,
        target_shape="grid",
        target_sdf_array=sdf_mm,
        stock_size_mm=stock_size_mm,
    )
    assert sim.v == voxel_size_mm
    assert np.allclose(sim.target.to_numpy(), sdf_mm / voxel_size_mm, atol=1e-5)

    # Without a stock box, a raw array cannot define the physical scale.
    with pytest.raises(ValueError, match="stock_size_in"):
        CSGSimulatorDelta(max_steps=10, target_shape="grid", target_sdf_array=sdf_mm)

def test_occ_step_to_sdf_box(tmp_path):
    # Runs with either OpenCascade binding: pythonocc-core (conda) or
    # cadquery-ocp (pip); skips when neither is installed.
    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IFSelect import IFSelect_RetDone
    except ImportError:
        try:
            from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
            from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType
            from OCP.IFSelect import IFSelect_ReturnStatus
            STEPControl_AsIs = STEPControl_StepModelType.STEPControl_AsIs
            IFSelect_RetDone = IFSelect_ReturnStatus.IFSelect_RetDone
        except ImportError:
            pytest.skip("no OpenCascade bindings (pythonocc-core or cadquery-ocp)")

    # Generate a simple STEP box shape on the fly
    box_maker = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0)
    box_shape = box_maker.Shape()

    step_path = os.path.join(tmp_path, "box.step")
    writer = STEPControl_Writer()
    writer.Transfer(box_shape, STEPControl_AsIs)
    status = writer.Write(step_path)
    assert status == IFSelect_RetDone

    # Perform STEP to SDF conversion
    output_npz = os.path.join(tmp_path, "box_target.npz")
    resolution = 32

    sdf_grid, metadata = step_to_sdf(
        step_path=step_path,
        output_path=output_npz,
        resolution=resolution,
        allow_non_watertight=False,
    )

    # Physical dimensions are preserved: the stock is the part's bounding box
    # plus the default 5% padding margin (1.5 mm per side for a 30 mm part),
    # and the grid is anisotropic with cubic voxels.
    stock = np.array([13.0, 23.0, 33.0])
    v = stock[2] / resolution
    assert np.allclose(metadata["stock_size_mm"], stock, atol=1e-3)
    assert np.isclose(metadata["voxel_size_mm"], v)
    assert metadata["sdf_units"] == "mm"
    expected_shape = tuple(int(round(s / v)) for s in stock)
    assert sdf_grid.shape == expected_shape
    assert os.path.exists(output_npz)

    assert metadata["num_solids"] == 1
    assert metadata["sign_convention"] == "negative_inside"

    # SDF values are millimetres: the box center is 5 mm from the nearest
    # face (the 10 mm axis); grid corners sit in the padding margin, outside.
    ci, cj, ck = (s // 2 for s in sdf_grid.shape)
    assert -5.5 < sdf_grid[ci, cj, ck] < -4.0
    assert sdf_grid[0, 0, 0] > 0.0
    assert sdf_grid[-1, -1, -1] > 0.0

    # Round-trip: the simulator picks up the physical box from the NPZ.
    try:
        ti.init(arch=ti.cpu)
    except Exception:
        pass
    sim = CSGSimulatorDelta(max_steps=10, target_shape="grid", target_sdf_path=output_npz)
    assert np.allclose([sim.Lx, sim.Ly, sim.Lz], stock, atol=1e-3)
    assert np.isclose(sim.v, v)
