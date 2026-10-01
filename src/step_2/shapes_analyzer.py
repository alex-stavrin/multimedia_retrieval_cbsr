import vedo
import os
from dataclasses import dataclass
from src.helper.mesh_types import Box, FaceType
from src.helper.database_functions import retrieve_all_shape_files
import numpy as np
import csv


@dataclass
class ShapeAnalysis:
    relative_file_path: str
    shape_class: str
    n_faces: int
    n_points: int
    bounding_box: Box
    face_type: FaceType

    @staticmethod
    def write_analyses_to_csv(
        shapes: list["ShapeAnalysis"],
        filename: str,
    ) -> None:
        with open(filename, "w", newline="") as f:
            fieldnames = [
                "relative_file_path",
                "shape_class",
                "n_faces",
                "n_points",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "z_min",
                "z_max",
                "face_type",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for shape in shapes:
                writer.writerow(
                    {
                        "relative_file_path": shape.relative_file_path,
                        "shape_class": shape.shape_class,
                        "n_faces": shape.n_faces,
                        "n_points": shape.n_points,
                        "x_min": shape.bounding_box.x_min,
                        "x_max": shape.bounding_box.x_max,
                        "y_min": shape.bounding_box.y_min,
                        "y_max": shape.bounding_box.y_max,
                        "z_min": shape.bounding_box.z_min,
                        "z_max": shape.bounding_box.z_max,
                        "face_type": shape.face_type.value,
                    }
                )

    def read_from_csv(self, file_path: str):
        pass


def analyze_shape_from_file(shape_path: str) -> ShapeAnalysis:
    # Retrieve from file with cheks
    if shape_path is not str:
        ValueError(f"File path must be string: {type(shape_path)}")

    if not os.path.exists(shape_path):
        ValueError(f"{shape_path} does not exist: {shape_path}.")

    if not shape_path.endswith(".obj"):
        ValueError(f"{shape_path} is of invalid type: {shape_path}.")

    # Get class of object
    split_path = shape_path.split("/")
    if len(split_path) <= 1:
        ValueError(
            f"File path does not contain class superfolder: {shape_path}"
        )
    shape_class = shape_path.split("/")[-2]

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

    # Retrieve celss for face/cell type
    cells: list = mesh.cells
    if cells is None or len(cells) == 0:
        ValueError(f"Could not retrieve any cells from shape: {shape_path}")

    # Classify cells
    face_type = FaceType.POLY
    if all(len(cell) == 3 for cell in cells):
        face_type = FaceType.TRIANGLES

    if all(len(cell) == 4 for cell in cells):
        if face_type == FaceType.TRIANGLES:
            face_type = FaceType.MIXED
        else:
            face_type = FaceType.QUADS

    relative_path = split_path[-2] + "/" + split_path[-1]

    return ShapeAnalysis(
        relative_path,
        shape_class,
        mesh.ncells,
        mesh.npoints,
        bounding_box,
        face_type,
    )


def analyze_database(root_dir: str, output_path: str):
    if not isinstance(root_dir, str) or output_path is not str:
        TypeError("File paths must be string")

    if not os.path.exists(root_dir):
        ValueError(f"Root dir does not exist: {root_dir}")

    if not os.path.exists(output_path):
        ValueError(f"Output path does not exist: {output_path}")

    all_shape_paths: np.ndarray = retrieve_all_shape_files(root_dir)

    all_analyses = []
    n_total_shapes = len(all_shape_paths)
    last_percentage = 0

    bar_length = 40
    print("Starting to analyse shapes in database")
    for i, shape_path in enumerate(all_shape_paths, start=1):
        analysis: ShapeAnalysis = analyze_shape_from_file(shape_path)
        all_analyses.append(analysis)

        percentage_done: float = int(i / n_total_shapes * 100)

        if percentage_done > last_percentage:
            last_percentage = percentage_done

            filled = int(bar_length * percentage_done / 100)
            bar = "#" * filled + "-" * (bar_length - filled)

            print(
                f"\r[{bar}] {percentage_done:3d}% ({i}/{n_total_shapes})",
                end="",
                flush=True,
            )
    # Newline at end
    print()

    ShapeAnalysis.write_analyses_to_csv(all_analyses, output_path)


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
    parser.add_argument(
        "-o", "--output", type=str, help="Where to store CSV output"
    )
    parser.add_argument(
        "-s",
        "--single",
        type=bool,
        default=False,
        help="Whether the given input is a single file",
    )

    args = parser.parse_args()

    if args.single:
        analysis = analyze_shape_from_file(args.input)
        print(analysis)
    else:
        analyze_database(args.input, args.output)


if __name__ == "__main__":
    main()


if __name__ == "__init___":
    main()
