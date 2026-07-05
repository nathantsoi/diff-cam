import os
import sys
import argparse
import time
import numpy as np

def _import_cad():
    # Attempt to locate and add the conda environment's Library\bin to DLL search path on Windows.
    # This prevents the DLL load failure for _XCAFDoc and other pythonocc C++ modules.
    if os.name == "nt":
        dll_dir = os.path.join(sys.prefix, "Library", "bin")
        if os.path.exists(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass
    
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopAbs import TopAbs_SOLID
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods_Solid
        from OCC.Core.ShapeFix import ShapeFix_Shape
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from occwl.solid import Solid
        
        return {
            "STEPControl_Reader": STEPControl_Reader,
            "IFSelect_RetDone": IFSelect_RetDone,
            "TopAbs_SOLID": TopAbs_SOLID,
            "TopExp_Explorer": TopExp_Explorer,
            "topods_Solid": topods_Solid,
            "ShapeFix_Shape": ShapeFix_Shape,
            "StlAPI_Writer": StlAPI_Writer,
            "BRepMesh_IncrementalMesh": BRepMesh_IncrementalMesh,
            "Solid": Solid,
        }
    except ImportError as e:
        raise ImportError(
            "CAD dependencies (occwl or pythonocc-core) are missing.\n"
            "Please ensure you are running in a conda environment where occwl is installed:\n"
            "  conda activate diffcam-occwl"
        ) from e

def _load_step_solids(step_path, cad):
    # Try importing occwl's load_step
    try:
        from occwl.io import load_step
        return load_step(str(step_path))
    except Exception:
        # Fallback to direct load without occwl.io (avoids XCAFDoc DLL loading bug on Windows conda)
        reader = cad["STEPControl_Reader"]()
        status = reader.ReadFile(str(step_path))
        if status != cad["IFSelect_RetDone"]:
            raise ValueError(f"Failed to load STEP file: {step_path}")
        reader.TransferRoots()
        shape = reader.OneShape()
        
        solids = []
        explorer = cad["TopExp_Explorer"](shape, cad["TopAbs_SOLID"])
        while explorer.More():
            solid_shape = cad["topods_Solid"](explorer.Current())
            solids.append(cad["Solid"](solid_shape))
            explorer.Next()
        return solids

def step_to_sdf(
    step_path,
    output_path=None,
    resolution=64,
    padding=0.05,
    solid_indices=None,
    clip_distance=None,
    allow_non_watertight=False,
):
    """
    Convert a STEP file containing B-Rep solids into an SDF voxel grid.
    
    Args:
        step_path (str): Path to the input STEP file.
        output_path (str, optional): Path to save the compressed .npz target grid.
        resolution (int): Dimension of the voxel grid (resolution x resolution x resolution).
        padding (float): Uniform margin fraction to scale/fit the shape inside the [0, 1]^3 stock.
        solid_indices (list of int, optional): Subset of solid indices to convert.
        clip_distance (float, optional): Maximum absolute distance to clip the output SDF values.
        allow_non_watertight (bool): If True, process non-watertight models instead of raising a ValueError.
        
    Returns:
        tuple: (sdf_grid, metadata)
            sdf_grid: np.ndarray of shape (resolution, resolution, resolution) with negative values inside.
            metadata: dict containing transformation and source info.
    """
    cad = _import_cad()
    import trimesh
    
    # Patch rtree Index class with pure Python/Numpy equivalent to prevent buggy C++ libspatialindex bad allocation errors on Windows
    try:
        import rtree
        class PurePythonRtree:
            def __init__(self, data, properties=None):
                self.ids = []
                self.bounds = []
                for item in data:
                    self.ids.append(item[0])
                    self.bounds.append(item[1])
                self.bounds = np.array(self.bounds, dtype=np.float64)
                self.ids = np.array(self.ids, dtype=np.int64)

            def intersection(self, b):
                mask = (
                    (self.bounds[:, 0] <= b[3]) & (self.bounds[:, 3] >= b[0]) &
                    (self.bounds[:, 1] <= b[4]) & (self.bounds[:, 4] >= b[1]) &
                    (self.bounds[:, 2] <= b[5]) & (self.bounds[:, 5] >= b[2])
                )
                return self.ids[mask].tolist()
        rtree.index.Index = PurePythonRtree
    except ImportError:
        pass

    import tempfile

    # Load all solids
    solids = _load_step_solids(step_path, cad)
    if not solids:
        raise ValueError(f"No solids found in STEP file: {step_path}")

    # Filter solids if requested
    if solid_indices is not None:
        solids = [solids[i] for i in solid_indices if i < len(solids)]
        if not solids:
            raise ValueError(f"No solids remained after filtering with indices {solid_indices}")

    # ---- Step 1: Compute bounding box from BRep for adaptive deflection ----
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    # Get combined bounding box across all solids BEFORE meshing
    combined_bbox = Bnd_Box()
    for solid in solids:
        brepbndlib.Add(solid.topods_shape(), combined_bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = combined_bbox.Get()
    brep_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    max_dim_model = float(np.max(brep_size))

    # Adaptive deflection: 0.1% of the largest dimension gives good tessellation
    # quality. For a 100mm part this is 0.1mm; for a 10mm part it's 0.01mm.
    adaptive_deflection = max(max_dim_model * 0.001, 1e-4)
    print(f"  Geometry size: {brep_size} (max_dim={max_dim_model:.2f})")
    print(f"  Adaptive mesh deflection: {adaptive_deflection:.6f}")

    # ---- Step 2: Heal, tessellate, and validate each solid ----
    meshes = []
    for idx, solid in enumerate(solids):
        topo_shape = solid.topods_shape()
        fixer = cad["ShapeFix_Shape"](topo_shape)
        fixer.SetPrecision(1e-3)
        fixer.Perform()
        healed_shape = fixer.Shape()

        temp_fd, temp_stl_path = tempfile.mkstemp(suffix=".stl")
        os.close(temp_fd)
        
        try:
            mesh_tool = cad["BRepMesh_IncrementalMesh"](healed_shape, adaptive_deflection)
            mesh_tool.Perform()
            
            writer = cad["StlAPI_Writer"]()
            writer.SetASCIIMode(False)
            writer.Write(healed_shape, temp_stl_path)

            mesh = trimesh.load(temp_stl_path)
        finally:
            if os.path.exists(temp_stl_path):
                try:
                    os.remove(temp_stl_path)
                except Exception:
                    pass

        print(f"  Solid {idx}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
              f"watertight={mesh.is_watertight}, volume={mesh.volume:.2f}")

        if not mesh.is_watertight:
            if not allow_non_watertight:
                raise ValueError(
                    f"Solid at index {idx} is not watertight. Ensure it represents a closed solid, "
                    "or pass allow_non_watertight=True to force processing."
                )
            else:
                print(f"  WARNING: Solid {idx} is non-watertight but proceeding anyway.")

        meshes.append(mesh)

    # ---- Step 3: Compute normalization transform ----
    all_vertices = np.concatenate([m.vertices for m in meshes], axis=0)
    bbox_min = all_vertices.min(axis=0)
    bbox_max = all_vertices.max(axis=0)

    size = bbox_max - bbox_min
    max_dim = np.max(size)
    scale = (1.0 - 2.0 * padding) / max_dim if max_dim > 0 else 1.0
    center = (bbox_min + bbox_max) / 2.0
    offset = 0.5 - center * scale

    for m in meshes:
        m.apply_translation(-center)
        m.apply_scale(scale)
        m.apply_translation([0.5, 0.5, 0.5])

    # ---- Step 4: Sample SDF at voxel centers ----
    dx = 1.0 / resolution
    x_coords = (np.arange(resolution) + 0.5) * dx
    grid_x, grid_y, grid_z = np.meshgrid(x_coords, x_coords, x_coords, indexing="ij")
    query_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    total_points = len(query_points)

    try:
        import open3d as o3d
        use_open3d = True
    except ImportError:
        use_open3d = False

    sdfs = []
    for mesh_idx, m in enumerate(meshes):
        t_start = time.time()
        
        if use_open3d:
            # Open3D is highly optimized (C++ KDTree/Embree) and can process all points at once
            print(f"  Solid {mesh_idx}: Sampling SDF using Open3D raycaster...")
            scene = o3d.t.geometry.RaycastingScene()
            vertices = o3d.core.Tensor(m.vertices, o3d.core.float32)
            faces = o3d.core.Tensor(m.faces, o3d.core.int32)
            o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, faces)
            scene.add_triangles(o3d_mesh)
            
            query_tensor = o3d.core.Tensor(query_points, o3d.core.float32)
            # Open3D convention: negative inside, positive outside
            sdf_vals = scene.compute_signed_distance(query_tensor).numpy()
        else:
            # Fallback to pure Python Trimesh sampling
            batch_size = 5000
            sdf_vals = np.empty(total_points, dtype=np.float32)
            for i in range(0, total_points, batch_size):
                batch = query_points[i : i + batch_size]
                # trimesh signed_distance uses positive-inside. We negate to get negative-inside.
                sdf_vals[i : i + batch_size] = -trimesh.proximity.signed_distance(m, batch)
                # Progress reporting for large grids
                if total_points > 50000 and (i + batch_size) % (batch_size * 20) == 0:
                    pct = min(100, 100 * (i + batch_size) / total_points)
                    elapsed = time.time() - t_start
                    print(f"  Solid {mesh_idx}: {pct:.0f}% sampled ({elapsed:.1f}s)")
                    
        elapsed = time.time() - t_start
        print(f"  Solid {mesh_idx}: SDF sampling complete ({elapsed:.1f}s)")
        sdfs.append(sdf_vals.reshape(resolution, resolution, resolution))

    sdf_grid = np.minimum.reduce(sdfs)

    # ---- Step 5: Quality validation ----
    min_sdf = float(sdf_grid.min())
    n_inside = int(np.sum(sdf_grid < 0))
    n_total = sdf_grid.size
    pct_inside = 100 * n_inside / n_total

    print(f"  SDF stats: min={min_sdf:.6f}, inside={n_inside}/{n_total} ({pct_inside:.1f}%)")

    if abs(min_sdf) < 2 * dx:
        print(f"  WARNING: SDF minimum ({min_sdf:.6f}) is shallower than 2 voxels ({2*dx:.6f}).")
        print(f"  The geometry may be a thin shell or the resolution ({resolution}) is too low.")
        print(f"  Consider increasing --resolution or verifying the STEP file contains solid bodies.")

    if pct_inside < 1.0:
        print(f"  WARNING: Only {pct_inside:.1f}% of voxels are inside the shape.")
        print(f"  This usually indicates a thin-walled shell rather than a solid body.")

    # Verify sign convention: the centroid of negative voxels should be near the geometric center
    if n_inside > 0:
        inside_mask = sdf_grid < 0
        inside_coords = np.argwhere(inside_mask)
        centroid_voxel = inside_coords.mean(axis=0)
        centroid_normalized = (centroid_voxel + 0.5) / resolution
        print(f"  Interior centroid (normalized): {centroid_normalized}")

    if clip_distance is not None:
        sdf_grid = np.clip(sdf_grid, -clip_distance, clip_distance)

    metadata = {
        "resolution": int(resolution),
        "padding": float(padding),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "scale": float(scale),
        "offset": offset.tolist(),
        "source_step": os.path.basename(step_path),
        "num_solids": int(len(solids)),
        "sign_convention": "negative_inside",
    }

    if output_path is not None:
        np.savez_compressed(
            output_path,
            sdf=sdf_grid.astype(np.float32),
            resolution=np.array(resolution, dtype=np.int32),
            padding=np.array(padding, dtype=np.float32),
            bbox_min=bbox_min.astype(np.float32),
            bbox_max=bbox_max.astype(np.float32),
            scale=np.array(scale, dtype=np.float32),
            offset=offset.astype(np.float32),
            source_step=os.path.basename(step_path),
            num_solids=np.array(len(solids), dtype=np.int32),
            sign_convention="negative_inside",
        )

    return sdf_grid.astype(np.float32), metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert STEP BREP CAD models to simulator-ready SDF target grids."
    )
    parser.add_argument("input", help="Path to input .step/.stp file")
    parser.add_argument("--output", "-o", required=True, help="Path to write the target .npz grid")
    parser.add_argument(
        "--resolution", "-r", type=int, default=64, help="Grid voxel resolution (default 64)"
    )
    parser.add_argument(
        "--padding", "-p", type=float, default=0.05, help="Boundary padding fraction (default 0.05)"
    )
    parser.add_argument(
        "--solid-indices",
        help="Comma-separated solid indices to load (default all)",
    )
    parser.add_argument(
        "--clip-distance", type=float, help="Maximum SDF distance absolute value clip"
    )
    parser.add_argument(
        "--allow-non-watertight",
        action="store_true",
        help="Proceed with conversion even if geometry is not watertight",
    )
    args = parser.parse_args()

    indices = None
    if args.solid_indices:
        indices = [int(x.strip()) for x in args.solid_indices.split(",")]

    print(f"Loading {args.input}...")
    t0 = time.time()
    try:
        grid, meta = step_to_sdf(
            step_path=args.input,
            output_path=args.output,
            resolution=args.resolution,
            padding=args.padding,
            solid_indices=indices,
            clip_distance=args.clip_distance,
            allow_non_watertight=args.allow_non_watertight,
        )
        elapsed = time.time() - t0
        print(f"Successfully converted and saved SDF to {args.output} ({elapsed:.1f}s).")
        print("Metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
