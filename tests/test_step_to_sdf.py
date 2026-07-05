import os
import tempfile
import numpy as np
import pytest
import taichi as ti

from utils.step_to_sdf import step_to_sdf
from simulator.csg_simulator import CSGSimulatorDelta
from cam_env.cam_env import CamEnvDiff

def test_voxel_centers_and_normalization():
    # Test normalization math: centering and uniform fit with margin
    bbox_min = np.array([-10.0, -5.0, 0.0])
    bbox_max = np.array([10.0, 5.0, 30.0])
    padding = 0.05

    # Compute expected scaling / offsets
    size = bbox_max - bbox_min
    max_dim = np.max(size) # 30.0

    scale = (1.0 - 2.0 * padding) / max_dim # 0.9 / 30.0 = 0.03
    center = (bbox_min + bbox_max) / 2.0 # [0.0, 0.0, 15.0]
    offset = 0.5 - center * scale # [0.5, 0.5, 0.5 - 15.0 * 0.03] = [0.5, 0.5, 0.05]

    # Transform corners to verify they are within [padding, 1 - padding]
    p1 = bbox_min * scale + offset
    p2 = bbox_max * scale + offset

    assert np.all(p1 >= padding - 1e-6)
    assert np.all(p2 <= 1.0 - padding + 1e-6)

    # Verify largest dimension spans exactly 1 - 2*padding
    assert np.isclose(np.max(p2 - p1), 1.0 - 2.0 * padding)

def test_simulator_grid_target_loading(tmp_path):
    # Setup Taichi CPU arch for test
    try:
        ti.init(arch=ti.cpu)
    except Exception:
        pass

    resolution = 16
    dx = 1.0 / resolution
    # Create a synthetic target SDF: sphere centered at (0.5, 0.5, 0.5) with radius 0.3
    # standard SDF: length(p - center) - radius
    # negative inside, positive outside
    x_coords = (np.arange(resolution) + 0.5) * dx
    grid_x, grid_y, grid_z = np.meshgrid(x_coords, x_coords, x_coords, indexing="ij")
    centers = np.stack([grid_x, grid_y, grid_z], axis=-1)

    dist_from_center = np.linalg.norm(centers - np.array([0.5, 0.5, 0.5]), axis=-1)
    sdf_sphere = dist_from_center - 0.3

    npz_path = os.path.join(tmp_path, "synthetic_target.npz")
    np.savez_compressed(
        npz_path,
        sdf=sdf_sphere.astype(np.float32),
        resolution=np.array(resolution, dtype=np.int32),
        padding=np.array(0.0, dtype=np.float32),
        bbox_min=np.array([0.2, 0.2, 0.2], dtype=np.float32),
        bbox_max=np.array([0.8, 0.8, 0.8], dtype=np.float32),
        scale=np.array(1.0, dtype=np.float32),
        offset=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        source_step="synthetic",
        num_solids=np.array(1, dtype=np.int32),
        sign_convention="negative_inside",
    )

    # Initialize simulator with grid target
    sim = CSGSimulatorDelta(
        resolution=resolution,
        max_steps=10,
        target_shape="grid",
        target_sdf_path=npz_path,
        stock_size_in=(1.0, 1.0, 1.0),
    )

    assert sim.target_shape == "grid"
    # The simulator converts the npz's normalized-cube distances to VOXEL
    # units at load (matching stock/tool SDFs and bake_target_grid).
    target_loaded = sim.target.to_numpy()
    assert np.allclose(target_loaded, sdf_sphere * resolution)

    # Compute target volume and verify matches
    sim.set_target_volume()
    volume = sim.target_volume[None]
    expected_volume = np.sum(sdf_sphere < 0) * (dx**3)
    assert np.isclose(volume, expected_volume)

    # Initialize environment with grid target
    env = CamEnvDiff(
        resolution=resolution,
        max_steps=10,
        target_shape="grid",
        target_sdf_path=npz_path,
    )
    obs, info = env.reset()

    # Verify environment observation layout and content
    # obs_dims = 3 + 2 + res^3 + res^3
    assert obs.shape[0] == 3 + 2 + resolution**3 + resolution**3
    target_obs = obs[-(resolution**3) :]
    assert np.allclose(target_obs, sdf_sphere.ravel() * resolution)

def test_occ_step_to_sdf_box(tmp_path):
    # Skip if occwl or OCC is not installed in the running environment
    pytest.importorskip("occwl")
    pytest.importorskip("OCC.Core")

    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone

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
    padding = 0.05

    sdf_grid, metadata = step_to_sdf(
        step_path=step_path,
        output_path=output_npz,
        resolution=resolution,
        padding=padding,
        allow_non_watertight=False,
    )

    # Verify grid properties
    assert sdf_grid.shape == (resolution, resolution, resolution)
    assert os.path.exists(output_npz)

    # Verify metadata fields
    assert metadata["resolution"] == resolution
    assert metadata["padding"] == padding
    assert metadata["num_solids"] == 1
    assert metadata["sign_convention"] == "negative_inside"

    # Center is inside the box, corners are outside
    center_idx = resolution // 2
    assert sdf_grid[center_idx, center_idx, center_idx] < 0.0
    assert sdf_grid[0, 0, 0] > 0.0
    assert sdf_grid[-1, -1, -1] > 0.0
