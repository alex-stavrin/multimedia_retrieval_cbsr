import math
import os

import numpy as np
import pandas as pd

import vedo

from src.helper.mesh_types import FaceType

DATABASE: pd.DataFrame | None = None


def __prepare_and_check_file_path(shape: str) -> vedo.Mesh:
    # Type check
    if not isinstance(shape, str):
        raise TypeError(
            "Shape argument must be of type string "
            f"{shape} with type: {type(shape)}"
        )

    if not shape.endswith(".obj"):
        raise ValueError(f"Mesh must be of filetype .obj: {shape}")

    if not os.path.exists(shape):
        raise ValueError(f"Path to mesh does not exist: {shape}")

    shape = vedo.Mesh(shape)

    return shape


def __get_data_from_database(
    file_path: str, database_path: str
) -> pd.DataFrame:
    if file_path is None:
        raise TypeError(f"File path is None: {file_path}")

    if database_path is None:
        raise TypeError(f"Database path is None: {database_path}")

    if not file_path.endswith(".obj"):
        raise ValueError(f"File must be of .obj type: {file_path}")

    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    if not database_path.endswith(".csv"):
        raise ValueError(f"Database file must be csv: {database_path}")

    if not os.path.exists(database_path):
        raise ValueError(f"Database file does not exist: {database_path}")

    split_path = file_path.split("/")

    if len(split_path) < 2:
        raise ValueError(
            "File path must contain shape class"
            f" superdirectory: {file_path}"
        )

    # Retrieve shape class and name
    relative_path_split = split_path[-2:]
    relative_joined = "/".join(relative_path_split)

    # Get analysis data
    global DATABASE
    if DATABASE is None:
        DATABASE = pd.read_csv(database_path)

    matching_rows = DATABASE[DATABASE["relative_file_path"] == relative_joined]

    return matching_rows


def _approx_volume(shape: vedo.Mesh) -> float:
    vertices = np.asarray(shape.vertices, dtype=np.float64)
    vertices = vertices - vertices.mean(axis=0)
    cells = np.asarray(shape.cells, dtype=np.int64)

    v0, v1, v2 = (
        vertices[cells[:, 0]],
        vertices[cells[:, 1]],
        vertices[cells[:, 2]],
    )
    return abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6)


def _oriented_bounding_box(shape: vedo.Mesh, visualise: bool = False):
    """
    Computes an oriented bounding box (OBB) of a mesh using PCA.

    The box axes are the principal axes of the vertex cloud, and the box is
    the tightest box around the vertices along those axes. This is close to
    the minimum-volume OBB but not guaranteed to be minimal (see Notes).

    Args:
        shape: The mesh to bound.
        visualise: If True, opens a vedo window showing the mesh and its
            box. This blocks until the window is closed.

    Returns:
        box_center: (3,) array, the center of the box in world space.
        axes: (3, 3) array, the box axes as columns (right-handed,
            orthonormal). Column i is the direction of side i.
        extents: (3,) array, the side lengths along each axis.
            The box volume is extents.prod().
    """
    pts = np.asarray(shape.vertices, dtype=np.float64)
    center = pts.mean(axis=0)

    # Principal axes of the point cloud (columns of `axes`)
    _, axes = np.linalg.eigh(np.cov((pts - center).T))
    if np.linalg.det(axes) < 0:  # make it a proper rotation, not a mirror
        axes[:, 2] *= -1

    # Express the points in the box frame and take the axis-aligned extent
    local = (pts - center) @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)

    extents = hi - lo
    box_center = center + axes @ ((lo + hi) / 2)

    if visualise:
        # 8 corners in the box frame (index = 4*x + 2*y + z, 0=lo / 1=hi)
        corners_local = np.array(
            [
                [x, y, z]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ]
        )
        # Back to world space (inverse of `local = (pts - center) @ axes`)
        corners = center + corners_local @ axes.T

        # An edge connects two corners whose indices differ in exactly one bit
        edges = [
            (i, i | bit) for i in range(8) for bit in (1, 2, 4) if not i & bit
        ]
        starts = corners[[i for i, _ in edges]]
        ends = corners[[j for _, j in edges]]

        box = vedo.Lines(starts, ends, c="black", lw=2)
        vedo.show(shape, box, axes=1)

    return box_center, axes, extents


def calculate_surface_area(
    file_path: str,
    database_path: str,
    shape: vedo.Mesh | None = None,
    row_entry: pd.DataFrame | None = None,
) -> float:
    """
    Calculates the provided shapes's surface area.

    If shape provided, uses that shape object instead of recalculating via
    file_path. Does not check whether shape and file_path refer to same object.
    If row_entry provided, uses dataframe instead of recalculating dataframe
    from database.

    Only accurate for comparison if shapes have been size normalized.
    """

    # Type check
    if shape is None:
        shape = __prepare_and_check_file_path(file_path)

    data_entry = row_entry
    if data_entry is None:
        data_entry = __get_data_from_database(file_path, database_path)

    # Triangulate mesh before calculating the area
    face_type = data_entry["face_type"]
    if face_type is not FaceType.TRIANGLES:
        shape = shape.triangulate(verts=True, lines=True)

    return shape.area()


def calculate_compactness(
    file_path: str,
    shape: vedo.Mesh | None = None,
) -> float:

    # Check if not None, it is correct class
    if shape is not None and not isinstance(shape, vedo.Mesh):
        raise ValueError(
            "shape must be of type vedo.mesh: "
            f"{shape} with type {type(shape)}"
        )

    shape_object = shape
    if shape_object is None:
        shape_object: vedo.Mesh = __prepare_and_check_file_path(file_path)

    volume = _approx_volume(shape_object)
    area = shape_object.area()

    print(
        f"area: {area}, volume: {volume}, volume vedo: {shape_object.volume()}"
    )

    return (36 * math.pi * volume**2) / area**3


# MAYBE TRY SUBSAMPLING FOR APPROX VOLUME AND OOBB LATER
def calculate_rectangularity(
    file_path: str, shape: vedo.Mesh | None = None
) -> float:
    """
    Rectangularity = mesh volume / volume of its oriented bounding box.

    1.0 means the shape fills its box completely (e.g. a cuboid); lower
    values mean more empty space inside the box.
    """
    if shape is None:
        shape = __prepare_and_check_file_path(file_path)

    shape_volume = _approx_volume(shape)

    _, _, extents = _oriented_bounding_box(shape)
    box_volume = float(np.prod(extents))

    if box_volume <= 0:
        return 0.0

    rectangularity = shape_volume / box_volume

    return rectangularity


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Get descriptors from a 3D shape from an OBJ file"
        " + database statistics."
    )

    parser.add_argument(
        "-i", "--input", type=str, help="Path to the OBJ file to analyze."
    )
    parser.add_argument(
        "-d", "--database_path", type=str, help="File path to the database."
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Visualises some steps for verification.",
    )

    args = parser.parse_args()

    visualize: bool = args.visualize

    if args.visualize:
        if os.path.isdir(args.input):
            raise ValueError(
                f"Visualising only works for directories: {args.input}"
            )

        _oriented_bounding_box(vedo.Mesh(args.input), visualize)
    else:
        # CHECK IF CORRECT
        area = calculate_surface_area(args.input, args.database_path)
        compactness = calculate_compactness(args.input)
        rectangularity = calculate_rectangularity(args.input)

        print(
            f"Was closed: {vedo.Mesh(args.input).is_closed()}\n"
            f"Area: {round(area, 3)}\n"
            f"Compactness: {round(compactness, 3)}\n"
            f"Rectangularity: {round(rectangularity, 3)}\n"
        )


if __name__ == "__main__":
    main()
