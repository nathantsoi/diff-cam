import os
import sys
import argparse
import time
import numpy as np

def _import_cad():
    """Import OpenCascade bindings from either provider:

    - pythonocc-core (``OCC.Core``): conda-forge only.
    - cadquery-ocp (``OCP``): pip-installable wheel (``uv pip install cadquery-ocp``).

    Returns a dict of classes and small adapters normalizing the two bindings'
    API differences (module-level vs class-scoped enums, ``_s`` static methods,
    ``ASCIIMode`` attribute vs ``SetASCIIMode``).
    """
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
        from OCC.Core.ShapeFix import ShapeFix_Shape
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.Bnd import Bnd_Box
        try:
            from OCC.Core.TopoDS import topods
            to_solid = topods.Solid
        except ImportError:
            from OCC.Core.TopoDS import topods_Solid as to_solid
        try:
            from OCC.Core.BRepBndLib import brepbndlib
            bbox_add = brepbndlib.Add
        except ImportError:
            from OCC.Core.BRepBndLib import brepbndlib_Add as bbox_add

        def make_stl_writer():
            writer = StlAPI_Writer()
            writer.SetASCIIMode(False)
            return writer

        return {
            "STEPControl_Reader": STEPControl_Reader,
            "RetDone": IFSelect_RetDone,
            "TopAbs_SOLID": TopAbs_SOLID,
            "TopExp_Explorer": TopExp_Explorer,
            "to_solid": to_solid,
            "ShapeFix_Shape": ShapeFix_Shape,
            "make_stl_writer": make_stl_writer,
            "BRepMesh_IncrementalMesh": BRepMesh_IncrementalMesh,
            "Bnd_Box": Bnd_Box,
            "bbox_add": bbox_add,
        }
    except ImportError:
        pass

    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.TopAbs import TopAbs_ShapeEnum
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        from OCP.ShapeFix import ShapeFix_Shape
        from OCP.StlAPI import StlAPI_Writer
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        def make_stl_writer():
            writer = StlAPI_Writer()
            writer.ASCIIMode = False
            return writer

        return {
            "STEPControl_Reader": STEPControl_Reader,
            "RetDone": IFSelect_ReturnStatus.IFSelect_RetDone,
            "TopAbs_SOLID": TopAbs_ShapeEnum.TopAbs_SOLID,
            "TopExp_Explorer": TopExp_Explorer,
            "to_solid": TopoDS.Solid_s,
            "ShapeFix_Shape": ShapeFix_Shape,
            "make_stl_writer": make_stl_writer,
            "BRepMesh_IncrementalMesh": BRepMesh_IncrementalMesh,
            "Bnd_Box": Bnd_Box,
            "bbox_add": BRepBndLib.Add_s,
        }
    except ImportError as e:
        raise ImportError(
            "OpenCascade bindings are missing. Either activate the conda env\n"
            "(scripts/setup_occ_env.sh creates it):\n"
            "  conda activate diffcam-occwl\n"
            "or install the pip wheel into the current environment:\n"
            "  uv pip install cadquery-ocp open3d"
        ) from e

def _load_step_solids(step_path, cad):
    """Load all TopoDS_Solid shapes from a STEP file."""
    reader = cad["STEPControl_Reader"]()
    status = reader.ReadFile(str(step_path))
    if status != cad["RetDone"]:
        raise ValueError(f"Failed to load STEP file: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    solids = []
    explorer = cad["TopExp_Explorer"](shape, cad["TopAbs_SOLID"])
    while explorer.More():
        solids.append(cad["to_solid"](explorer.Current()))
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
    voxel_size_mm=None,
    stock_size_mm=None,
):
    """
    Convert a STEP file containing B-Rep solids into an SDF voxel grid,
    PRESERVING the model's physical dimensions (STEP units, normally mm).

    The grid spans a rectangular STOCK box that defaults to the part's
    bounding box plus a small waste margin (``padding``), with the part
    centered inside it. Voxels are physical cubes, so the grid is anisotropic
    (Nx, Ny, Nz) matching the stock's aspect ratio, and SDF values are signed
    distances in millimetres.

    Args:
        step_path (str): Path to the input STEP file.
        output_path (str, optional): Path to save the compressed .npz target grid.
        resolution (int): Voxel count along the stock's LONGEST axis (ignored
            when ``voxel_size_mm`` is given).
        padding (float): Extra stock margin on every side, as a fraction of the
            part's longest dimension. Default 0.05 (5% waste material around
            the part); 0 makes the stock the exact part bounding box.
        solid_indices (list of int, optional): Subset of solid indices to convert.
        clip_distance (float, optional): Maximum absolute distance (mm) to clip the output SDF values.
        allow_non_watertight (bool): If True, process non-watertight models instead of raising a ValueError.
        voxel_size_mm (float, optional): Physical voxel edge in mm. Overrides ``resolution``.
        stock_size_mm (float or (x, y, z), optional): Explicit stock box in mm
            (scalar = cube). Must be at least the part's bounding box; the part
            is centered inside. Default: the part bounding box (+ padding).

    Returns:
        tuple: (sdf_grid, metadata)
            sdf_grid: np.ndarray of shape (Nx, Ny, Nz), signed distance in mm,
                negative inside.
            metadata: dict containing physical dimensions and source info.
    """
    cad = _import_cad()
    import trimesh
    
    # Patch rtree Index class with pure Python/Numpy equivalent to prevent buggy C++ libspatialindex bad allocation errors on Windows
    try:
        if os.name != "nt":
            raise ImportError("rtree patch is Windows-only")
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
    # Get combined bounding box across all solids BEFORE meshing
    combined_bbox = cad["Bnd_Box"]()
    for solid in solids:
        cad["bbox_add"](solid, combined_bbox)
    cmin, cmax = combined_bbox.CornerMin(), combined_bbox.CornerMax()
    xmin, ymin, zmin = cmin.X(), cmin.Y(), cmin.Z()
    xmax, ymax, zmax = cmax.X(), cmax.Y(), cmax.Z()
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
        fixer = cad["ShapeFix_Shape"](solid)
        fixer.SetPrecision(1e-3)
        fixer.Perform()
        healed_shape = fixer.Shape()

        temp_fd, temp_stl_path = tempfile.mkstemp(suffix=".stl")
        os.close(temp_fd)

        try:
            mesh_tool = cad["BRepMesh_IncrementalMesh"](healed_shape, adaptive_deflection)
            mesh_tool.Perform()

            writer = cad["make_stl_writer"]()
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

    # ---- Step 3: Physical stock box (dimensions are KEPT, no normalization) ----
    all_vertices = np.concatenate([m.vertices for m in meshes], axis=0)
    bbox_min = all_vertices.min(axis=0)
    bbox_max = all_vertices.max(axis=0)
    part_size = bbox_max - bbox_min
    part_center = (bbox_min + bbox_max) / 2.0

    if stock_size_mm is not None:
        if np.isscalar(stock_size_mm):
            stock_size = np.full(3, float(stock_size_mm))
        else:
            stock_size = np.asarray([float(c) for c in stock_size_mm], dtype=np.float64)
        if np.any(stock_size < part_size - 1e-6):
            print(f"  WARNING: stock {stock_size.tolist()} mm is smaller than the "
                  f"part bounding box {part_size.tolist()} mm on at least one axis; "
                  "the part will be truncated by the stock.")
    else:
        # Default stock: the part's bounding box, plus an optional uniform
        # margin (fraction of the longest part dimension) on every side.
        pad_mm = padding * float(np.max(part_size))
        stock_size = part_size + 2.0 * pad_mm

    # The part sits centered in the stock box; the grid keeps the STEP file's
    # coordinate frame (origin at the stock's min corner).
    grid_origin = part_center - stock_size / 2.0

    # Cubic voxels: explicit physical edge, or `resolution` voxels along the
    # stock's longest axis. Per-axis counts follow the stock's aspect ratio.
    if voxel_size_mm is not None:
        v = float(voxel_size_mm)
    else:
        v = float(np.max(stock_size)) / float(resolution)
    Nx, Ny, Nz = (max(1, int(round(s / v))) for s in stock_size)
    print(f"  Stock: {stock_size.tolist()} mm, voxel {v:.4f} mm -> grid ({Nx}, {Ny}, {Nz})")

    # ---- Step 4: Sample SDF at voxel centers (physical mm coordinates) ----
    x_coords = grid_origin[0] + (np.arange(Nx) + 0.5) * v
    y_coords = grid_origin[1] + (np.arange(Ny) + 0.5) * v
    z_coords = grid_origin[2] + (np.arange(Nz) + 0.5) * v
    grid_x, grid_y, grid_z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
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
        sdfs.append(sdf_vals.reshape(Nx, Ny, Nz))

    sdf_grid = np.minimum.reduce(sdfs)

    # ---- Step 5: Quality validation (distances in mm) ----
    min_sdf = float(sdf_grid.min())
    n_inside = int(np.sum(sdf_grid < 0))
    n_total = sdf_grid.size
    pct_inside = 100 * n_inside / n_total

    print(f"  SDF stats: min={min_sdf:.4f} mm, inside={n_inside}/{n_total} ({pct_inside:.1f}%)")

    if abs(min_sdf) < 2 * v:
        print(f"  WARNING: SDF minimum ({min_sdf:.4f} mm) is shallower than 2 voxels ({2*v:.4f} mm).")
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
        centroid_normalized = (centroid_voxel + 0.5) / np.array([Nx, Ny, Nz])
        print(f"  Interior centroid (normalized): {centroid_normalized}")

    if clip_distance is not None:
        sdf_grid = np.clip(sdf_grid, -clip_distance, clip_distance)

    metadata = {
        "resolution": int(max(Nx, Ny, Nz)),
        "grid_shape": (int(Nx), int(Ny), int(Nz)),
        "padding": float(padding),
        "voxel_size_mm": float(v),
        "stock_size_mm": stock_size.tolist(),
        "grid_origin_mm": grid_origin.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "sdf_units": "mm",
        "source_step": os.path.basename(step_path),
        "num_solids": int(len(solids)),
        "sign_convention": "negative_inside",
    }

    if output_path is not None:
        np.savez_compressed(
            output_path,
            sdf=sdf_grid.astype(np.float32),
            resolution=np.array(max(Nx, Ny, Nz), dtype=np.int32),
            padding=np.array(padding, dtype=np.float32),
            voxel_size_mm=np.array(v, dtype=np.float32),
            stock_size_mm=stock_size.astype(np.float32),
            grid_origin_mm=grid_origin.astype(np.float32),
            bbox_min=bbox_min.astype(np.float32),
            bbox_max=bbox_max.astype(np.float32),
            sdf_units="mm",
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
        "--resolution", "-r", type=int, default=64,
        help="Voxel count along the stock's longest axis (default 64; ignored with --voxel-size-mm)"
    )
    parser.add_argument(
        "--voxel-size-mm", type=float, default=None,
        help="Physical voxel edge in mm (overrides --resolution)"
    )
    parser.add_argument(
        "--stock-size-mm", type=float, nargs="+", default=None, metavar="MM",
        help="Explicit stock box in mm: one value (cube) or three (x y z). "
             "Default: the part's bounding box"
    )
    parser.add_argument(
        "--padding", "-p", type=float, default=0.05,
        help="Extra stock margin per side, as a fraction of the part's longest "
             "dimension (default 0.05; 0 = stock is the exact part bounding box)"
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

    stock = None
    if args.stock_size_mm is not None:
        if len(args.stock_size_mm) == 1:
            stock = args.stock_size_mm[0]
        elif len(args.stock_size_mm) == 3:
            stock = tuple(args.stock_size_mm)
        else:
            parser.error("--stock-size-mm takes 1 or 3 values")

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
            voxel_size_mm=args.voxel_size_mm,
            stock_size_mm=stock,
        )
        elapsed = time.time() - t0
        print(f"Successfully converted and saved SDF to {args.output} ({elapsed:.1f}s).")
        print("Metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
