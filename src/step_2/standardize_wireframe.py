import vedo
import pymeshfix
import numpy as np
import os
from src.helper.database_functions import retrieve_all_shape_files


def repair_mesh(shape: vedo.Mesh) -> vedo.Mesh:
    # only accepts np.ndarray...
    verts = np.asarray(shape.vertices, dtype=np.float64)
    cells = np.asarray(shape.cells, dtype=np.int32)

    fixer = pymeshfix.MeshFix(verts, cells)
    fixer.repair()

    return vedo.Mesh([fixer.points, fixer.faces])


def close_shape(shape: vedo.Mesh) -> tuple[vedo.Mesh, str]:
    """
    Closes shapes trying varying methods
    """
    # Cheap method first
    shape.fill_holes(size=1000)
    if shape.is_closed():
        # Type of filling for debugging/optimizing
        return shape, "filled"

    # Less cheap
    repaired = repair_mesh(shape)
    if repaired.is_closed():
        return repaired, "pymeshfix"

    # Most expensive
    vol = shape.binarize().dilate().erode()
    voxel_mesh = vol.isosurface()
    return voxel_mesh, "voxelized"


def standardize_wireframe(
    shape: vedo.Mesh | None, shape_path: str | None = None
) -> vedo.Mesh:
    """
    Triangulates the shape and closes its holes
    """
    if shape is None:
        if shape_path is None:
            raise ValueError("Either shape or shape_path must be non-None")

        if not os.path.exists(shape_path):
            raise ValueError(f"Shape file path does not exist: {shape_path}")

        if not shape_path.endswith(".obj"):
            raise ValueError(
                f"Shape file path is not of type .obj: {shape_path}"
            )

        shape = vedo.Mesh(shape_path)

    shape = shape.clone().triangulate().clean()

    shape, _ = close_shape(shape)

    return shape


def apply_wireframe_standardisation_to_database(
    database_directory_path: str, output_dir_path: str
) -> None:
    """
    Applies standardisation of a database directory containing .obj files.

    Output is a database with same file structure, containing
    wireframe standardized object.
    """
    all_shape_files: np.typing.NDArray[np.str_] = retrieve_all_shape_files(
        database_directory_path
    )

    print("First shape file:", all_shape_files[0])

    for shape_file in all_shape_files:
        shape_object = vedo.Mesh(shape_file)
        standardized_shape_object = standardize_wireframe(shape_object)

        relative_path = os.path.relpath(shape_file, database_directory_path)
        new_path = os.path.join(output_dir_path, relative_path)

        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        vedo.write(standardized_shape_object, new_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Wireframe standardize database."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the root directory of the database",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="New root directory for standardized database",
    )
    args = parser.parse_args()

    apply_wireframe_standardisation_to_database(args.input, args.output)


if __name__ == "__main__":
    main()
