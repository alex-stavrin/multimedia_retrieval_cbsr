import vedo
import pymeshfix
import numpy as np


def repair_mesh(mesh: vedo.Mesh) -> vedo.Mesh:
    # only accepts np.ndarray...
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int32)

    fixer = pymeshfix.MeshFix(verts, cells)
    fixer.repair()

    return vedo.Mesh([fixer.points, fixer.faces])


def close_shape(mesh: vedo.Mesh) -> tuple[vedo.Mesh, str]:
    # Cheap method first
    mesh.fill_holes(size=1000)
    if mesh.is_closed():
        # Type of filling for debugging/optimizing
        return mesh, "filled"

    # Less cheap
    repaired = repair_mesh(mesh)
    if repaired.is_closed():
        return repaired, "pymeshfix"

    # Most expensive
    vol = mesh.binarize().dilate().erode()
    voxel_mesh = vol.isosurface()
    return voxel_mesh, "voxelized"


def standardize_wireframe(mesh: vedo.Mesh) -> vedo.Mesh:
    """
    Triangulates the mesh and closes its holes
    """
    mesh = mesh.triangulate()

    mesh, _ = close_shape(mesh)

    return mesh
