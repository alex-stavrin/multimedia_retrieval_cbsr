import vedo
import os
from dataclasses import dataclass
from src.helper.mesh_types import Box, FaceType


@dataclass
class ShapeAnalysis:
    shape_class: str
    n_faces: int
    n_points: int
    bounding_box: Box
    face_type: FaceType


def analyze_shape_from_file(shape_path: str) -> ShapeAnalysis:
    # Retrieve from file with cheks
    if shape_path is not str:
        ValueError(f"File path must be string: {type(shape_path)}")

    if not os.path.exists(shape_path):
        ValueError(f"Shape file path does not exist: {shape_path}.")

    if not shape_path.endswith(".obj"):
        ValueError(f"Shape file path is of invalid type: {shape_path}.")

    # Get class of object
    split_path = shape_path.split("/")
    if len(split_path) <= 1:
        ValueError(
            f"File path does not contain class superfolder: {shape_path}"
        )
    shape_class = shape_path.split("/")[-2]
    print(shape_class)

    # Create mesh object from file
    mesh = vedo.Mesh(shape_path)

    # Get bounding box of mesh
    bounds_array = mesh.bounds()
    if bounds_array is None:
        ValueError(
            f"Could not retrieve bounding box coords from shape: {shape_path},"
            + f", bounding cords: {bounds_array}"
        )

    # Put into more readable class
    bounding_box = Box(*bounds_array)

    # Retrieve cells for face/cell type
    cells: list = mesh.cells
    if cells is None or len(cells) == 0:
        ValueError(f"Could not retrieve any cells from shape: {shape_path}")

    # Classify cells
    face_type = FaceType.MIXED
    if all(len(cell) == 3 for cell in cells):
        face_type = FaceType.TRIANGLES
    elif all(len(cell) == 4 for cell in cells):
        face_type = FaceType.QUADS

    return ShapeAnalysis(
        shape_class, mesh.ncells, mesh.npoints, bounding_box, face_type
    )


def main():
    # Only import when running main, to import necessarily when we import
    # it as a module
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a 3D shape from an OBJ file."
    )

    parser.add_argument(
        "-i", "--input", type=str, help="Path to the OBJ file to analyze"
    )

    args = parser.parse_args()

    analysis = analyze_shape_from_file(args.input)

    print(analysis)


if __name__ == "__main__":
    main()


if __name__ == "__init___":
    main()
