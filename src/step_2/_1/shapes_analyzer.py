import csv
import os
import sys
import vedo

from tqdm import tqdm
from dataclasses import dataclass
from PySide6.QtWidgets import QApplication

from src.helper.database_functions import retrieve_all_shape_files
from src.helper.mesh_types import Box, FaceType
from src.step_1.model_viewer import ModelViewer


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
    if not isinstance(shape_path, str):
        raise TypeError(f"File path must be string: {type(shape_path)}")

    if not os.path.isfile(shape_path):
        raise ValueError(f"File does not exist: {shape_path}")

    if not shape_path.lower().endswith(".obj"):
        raise ValueError(f"File is not an .obj file: {shape_path}")

    # Class of the object is the name of the folder containing it
    parent_dir = os.path.basename(os.path.dirname(os.path.abspath(shape_path)))
    if not parent_dir:
        raise ValueError(
            f"File path does not contain class superfolder: {shape_path}"
        )
    shape_class = parent_dir

    # Create mesh object from file
    mesh = vedo.Mesh(shape_path)

    # Get bounding box of mesh
    bounds_array = mesh.bounds()
    if bounds_array is None:
        raise ValueError(
            f"Could not retrieve bounding box from shape: {shape_path}"
        )

    bounding_box = Box(*bounds_array)

    # Retrieve cells for face/cell type
    cells = mesh.cells
    if cells is None or len(cells) == 0:
        raise ValueError(
            f"Could not retrieve any cells from shape: {shape_path}"
        )

    # Classify cells
    cell_sizes = {len(cell) for cell in cells}
    if cell_sizes == {3}:
        face_type = FaceType.TRIANGLES
    elif cell_sizes == {4}:
        face_type = FaceType.QUADS
    elif cell_sizes == {3, 4}:
        face_type = FaceType.MIXED
    else:
        face_type = FaceType.POLY

    relative_path = f"{shape_class}/{os.path.basename(shape_path)}"

    return ShapeAnalysis(
        relative_path,
        shape_class,
        mesh.ncells,
        mesh.npoints,
        bounding_box,
        face_type,
    )


def collect_shape_paths(inputs: list[str]) -> list[str]:
    """Expand a list of inputs into individual .obj file paths.

    Each input can be either a single .obj file or a directory (searched
    recursively). Duplicates are removed while keeping the original order.
    """
    all_paths: list[str] = []

    for path in inputs:
        if os.path.isdir(path):
            all_paths.extend(retrieve_all_shape_files(path))
        elif os.path.isfile(path):
            if not path.lower().endswith(".obj"):
                raise ValueError(f"Input is not an .obj file: {path}")
            all_paths.append(os.path.abspath(path))
        else:
            raise ValueError(f"Input does not exist: {path}")

    return list(dict.fromkeys(all_paths))


def analyze_shapes(shape_paths: list[str]) -> list[ShapeAnalysis]:
    """Analyze the given .obj files,
    showing a progress bar for larger batches."""
    all_analyses: list[ShapeAnalysis] = []

    for shape_path in tqdm(shape_paths):
        all_analyses.append(analyze_shape_from_file(shape_path))

    return all_analyses


def analyze_inputs(inputs: list[str], output_path: str | None = None):
    """Analyze any mix of single .obj files and database directories.

    Writes a CSV to output_path if given, otherwise prints the analyses.
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    shape_paths = collect_shape_paths(inputs)
    if len(shape_paths) == 0:
        raise ValueError(f"No .obj files found in inputs: {inputs}")

    analyses = analyze_shapes(shape_paths)

    if output_path is None:
        for analysis in analyses:
            print(analysis)
    else:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")
        ShapeAnalysis.write_analyses_to_csv(analyses, output_path)
        print(f"Wrote {len(analyses)} analyses to {output_path}")

    return analyses


def analyze_database(root_dir: str, output_path: str):
    """Kept for backwards compatibility; analyzes a whole directory."""
    return analyze_inputs([root_dir], output_path)


def visualize_analyses(
    analyses: list[ShapeAnalysis], shape_paths: list[str]
) -> int:
    meshes: list[vedo.Mesh] = []

    for analysis, shape_path in zip(analyses, shape_paths):
        bounds = analysis.bounding_box

        bounding_box = (
            vedo.Box(
                size=(
                    bounds.x_max - bounds.x_min,
                    bounds.y_max - bounds.y_min,
                    bounds.z_max - bounds.z_min,
                ),
                pos=(
                    (bounds.x_min + bounds.x_max) / 2,
                    (bounds.y_min + bounds.y_max) / 2,
                    (bounds.z_min + bounds.z_max) / 2,
                ),
            )
            .wireframe()
            .color("black")
            .linewidth(2)
        )
        meshes.extend([vedo.Mesh(shape_path), bounding_box])

    app = QApplication(sys.argv)
    model_viewer = ModelViewer(qt_app=app, meshes=meshes)
    return model_viewer.exec()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze 3D shapes from OBJ files and/or "
        "database directories."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="One or more .obj files and/or directories to analyze "
        "(directories are searched recursively).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Where to store CSV output. If omitted, results are printed.",
    )
    parser.add_argument(
        "-v",
        "--visual",
        action="store_true",
        help="Visualize the shapes with their bounding boxes.",
    )

    args = parser.parse_args()

    analyses = analyze_inputs(args.input, args.output)

    if args.visual:
        return visualize_analyses(analyses, collect_shape_paths(args.input))

    return 0


if __name__ == "__main__":
    sys.exit(main())
