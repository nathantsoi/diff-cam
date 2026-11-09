"""T5.0 modern native-smoke-only subset of the legacy PyMesh interface.

External to CNC-Net. No model, path, CSG or loss operation is implemented here.
The bundled run uses form_mesh and save_mesh for its raw marching-cubes OBJ.
Only these functions and optionally separate_mesh are exposed. They are NOT a
claimed replacement or drop-in equivalence proof for the full PyMesh library.
"""
from pathlib import Path

import numpy as np
import open3d as o3d


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N,3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("triangle faces must have shape (N,3)")
        if not np.isfinite(self.vertices).all():
            raise ValueError("non-finite vertices")

    @property
    def num_vertices(self):
        return len(self.vertices)

    def as_open3d(self):
        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices),
            o3d.utility.Vector3iVector(self.faces),
        )


def form_mesh(vertices, faces):
    return Mesh(vertices, faces)


def save_mesh(filename, mesh, ascii=True):
    """Serialize vertices/faces; file formatting is not asserted byte-equivalent."""
    destination = Path(filename)
    if not isinstance(mesh, Mesh):
        raise TypeError("this smoke shim accepts meshes returned by form_mesh only")
    ok = o3d.io.write_triangle_mesh(
        str(destination), mesh.as_open3d(), write_ascii=bool(ascii), compressed=False
    )
    if not ok:
        raise OSError(f"Open3D could not write triangle mesh: {destination}")


def separate_mesh(mesh):
    """Connected-component support for the optional largest-component helper."""
    if not isinstance(mesh, Mesh):
        raise TypeError("this smoke shim accepts meshes returned by form_mesh only")
    labels, _, _ = mesh.as_open3d().cluster_connected_triangles()
    groups = [Mesh(mesh.vertices, mesh.faces[labels == label]) for label in np.unique(labels)]
    return sorted(groups, key=lambda item: item.num_vertices, reverse=True)
