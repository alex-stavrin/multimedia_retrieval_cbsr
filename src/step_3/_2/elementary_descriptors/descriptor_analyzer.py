import os
import csv
import argparse
import vedo
from tqdm import tqdm
import vtk

from src.step_3._2.elementary_descriptors.descriptor_extracting import (
    calculate_surface_area,
    calculate_compactness,
    calculate_rectangularity,
    calculate_diameter,
    calculate_convexity,
    calculate_eccentricity,
    __prepare_and_check_file_path,
    __approx_volume,
    __calculate_pca,
)

DESCRIPTOR_FIELDS = [
    "file_path",
    "was_closed",
    "surface_area",
    "compactness",
    "rectangularity",
    "diameter",
    "convexity",
    "eccentricity",
    "reason",
]


#   FIX LATER
class VTKWarningCapture(vtk.vtkOutputWindow):
    """Capture VTK warning messages for the current mesh."""

    def __init__(self):
        super().__init__()
        self.warnings = []

    def DisplayWarningText(self, text):
        message = text.strip()
        if message:
            self.warnings.append(message)

    def DisplayErrorText(self, text):
        message = text.strip()
        if message:
            self.warnings.append(f"VTK ERROR: {message}")

    def DisplayGenericWarningText(self, text):
        message = text.strip()
        if message:
            self.warnings.append(message)


def retrieve_all_shape_files(input_dir_path: str) -> list[str]:
    """Find every OBJ file recursively under the input directory."""
    shape_files = []

    for root, _, files in os.walk(input_dir_path):
        for filename in files:
            if filename.lower().endswith(".obj"):
                shape_files.append(os.path.join(root, filename))

    return sorted(shape_files)


def calculate_file_descriptors(
    file_path: str,
) -> dict:
    """Calculate all descriptors for one mesh."""

    shape = __prepare_and_check_file_path(file_path)

    # Calculate shared quantities once.
    volume = __approx_volume(shape)
    area = calculate_surface_area(
        file_path,
        shape=shape,
    )
    pca = __calculate_pca(shape)
    convex_hull = vedo.ConvexHull(shape)

    # Reuse the precomputed values when possible.
    compactness = calculate_compactness(
        file_path,
        shape=shape,
        volume=volume,
        area=area,
    )

    rectangularity = calculate_rectangularity(
        file_path,
        shape=shape,
        shape_volume=volume,
    )

    diameter = calculate_diameter(
        file_path,
        shape=shape,
        convex_hull=convex_hull,
    )

    convexity = calculate_convexity(
        file_path,
        shape=shape,
        volume=volume,
        convex_hull=convex_hull,
    )

    eccentricity = calculate_eccentricity(
        file_path,
        shape=shape,
        pca=pca,
    )

    return {
        "was_closed": bool(shape.is_closed()),
        "surface_area": float(area),
        "compactness": float(compactness),
        "rectangularity": float(rectangularity),
        "diameter": float(diameter),
        "convexity": float(convexity),
        "eccentricity": float(eccentricity),
        "reason": "",
    }


def calculate_database(
    input_dir_path: str,
    database_path: str,
    output_path: str,
) -> None:
    """
    Calculate all descriptors for every OBJ file in a directory.

    The input directory structure is preserved in the CSV through the
    relative file_path column. The input meshes are not modified.
    """

    if not os.path.isdir(input_dir_path):
        raise ValueError(f"Input directory does not exist: {input_dir_path}")

    if not os.path.isfile(database_path):
        raise ValueError(f"Database file does not exist: {database_path}")

    if not database_path.lower().endswith(".csv"):
        raise ValueError("Database file must be a CSV file")

    shape_files = retrieve_all_shape_files(input_dir_path)

    if not shape_files:
        raise ValueError(f"No OBJ files found in {input_dir_path}")

    root_abs = os.path.abspath(input_dir_path)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    warnings_path = os.path.splitext(output_path)[0] + "_warnings.csv"

    n_success = 0
    n_errors = 0
    n_warnings = 0

    warning_capture = VTKWarningCapture()
    previous_output_window = vtk.vtkOutputWindow.GetInstance()
    #           FIX LATER
    vtk.vtkOutputWindow.SetInstance(warning_capture)

    try:
        with (
            open(output_path, "w", newline="", encoding="utf-8") as f,
            open(warnings_path, "w", newline="", encoding="utf-8") as wf,
        ):
            writer = csv.DictWriter(
                f,
                fieldnames=DESCRIPTOR_FIELDS,
            )
            writer.writeheader()

            warning_writer = csv.DictWriter(
                wf,
                fieldnames=["file_path", "warning"],
            )
            warning_writer.writeheader()

            for shape_file in tqdm(
                shape_files,
                desc="Calculating descriptors",
            ):
                relative_path = os.path.relpath(shape_file, root_abs)
                row = {"file_path": relative_path}

                # Clear warnings from the previous mesh.
                warning_capture.warnings.clear()

                try:
                    descriptors = calculate_file_descriptors(
                        shape_file,
                    )
                    row.update(descriptors)
                    n_success += 1

                except Exception as e:
                    row.update(
                        {
                            "was_closed": "",
                            "surface_area": "",
                            "compactness": "",
                            "rectangularity": "",
                            "diameter": "",
                            "convexity": "",
                            "eccentricity": "",
                            "reason": f"{type(e).__name__}: {e}",
                        }
                    )
                    n_errors += 1

                finally:
                    # Associate each captured warning with this mesh.
                    for warning in warning_capture.warnings:
                        warning_writer.writerow(
                            {
                                "file_path": relative_path,
                                "warning": warning,
                            }
                        )
                        n_warnings += 1

                writer.writerow(row)

    finally:
        # Restore the original VTK output window.
        vtk.vtkOutputWindow.SetInstance(previous_output_window)

    summary = (
        f"Descriptor calculation complete.\\n"
        f"Total meshes: {len(shape_files)}\\n"
        f"Successful: {n_success}\\n"
        f"Errors: {n_errors}\\n"
        f"VTK warnings/errors captured: {n_warnings}\\n"
        f"Output CSV: {output_path}\\n"
        f"Warnings CSV: {warnings_path}\\n"
    )

    summary_path = os.path.splitext(output_path)[0] + "_conclusion.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate 3D shape descriptors for an OBJ database."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Root directory containing OBJ meshes.",
    )

    parser.add_argument(
        "-d",
        "--database_path",
        required=True,
        help="CSV database containing relative_file_path and face_type.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="shape_descriptors.csv",
        help="Output CSV path (default: shape_descriptors.csv).",
    )

    args = parser.parse_args()

    calculate_database(
        input_dir_path=args.input,
        database_path=args.database_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
