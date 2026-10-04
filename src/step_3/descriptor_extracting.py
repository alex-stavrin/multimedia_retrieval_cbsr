import vedo
import trimesh
import numpy as np
import os
import pandas as pd
from src.helper.mesh_types import FaceType
from src.step_2.standardize_wireframe import standardize_wireframe
import math

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

    if shape is None:
        shape: vedo.Mesh = __prepare_and_check_file_path(file_path)

    # REMOVE LATER                                  SHOULD BE DONE OVER WHOLE DATABASE
    shape = standardize_wireframe(shape)

    if not shape.is_closed():
        raise ValueError(
            f"Mesh must be closed to calculate compactness: {file_path}"
        )

    volume = shape.volume()
    area = shape.area()

    return (36 * math.pi * volume**2) / area**3


def calculate_rectangularity(
    file_path: str, shape: vedo.Mesh | None = None, max_points: int = 20000
):
    if shape is None:
        shape = __prepare_and_check_file_path(file_path)

    converted_mesh = trimesh.Trimesh(
        vertices=shape.vertices, faces=shape.cells
    )

    # MAKE SUR ETO CLEAN MESH                                               AAAAAA
    shape = standardize_wireframe(shape)
    oriented_bounding_box = converted_mesh.bounding_box_oriented

    shape_volume = abs(shape.volume())

    vertices = np.asarray(shape.vertices)
    if len(vertices) > max_points:
        rng = np.random.default_rng(0)
        vertices_fit = vertices[
            rng.choice(len(vertices), max_points, replace=False)
        ]
    else:
        vertices_fit = vertices

    _, _, _, R = min_obb(vertices_fit, restarts=n_restarts, seed=0)

    print(oriented_bounding_box.volume)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Get descriptors from a 3D shape from an OBJ file"
        " + database statistics."
    )

    parser.add_argument(
        "-i", "--input", type=str, help="Path to the OBJ file to analyze"
    )
    parser.add_argument(
        "-d", "--database_path", type=str, help="File path to the database"
    )

    args = parser.parse_args()

    # # CHECK IF CORRECT
    area = calculate_surface_area(args.input, args.database_path)
    compactness = calculate_compactness(args.input)
    rectangularity = calculate_rectangularity(args.input)

    print(f"Area: {area}\n" f"Compactness: {compactness}\n")


if __name__ == "__main__":
    main()
