import os
import csv
import argparse
import vedo
from tqdm import tqdm
import vtk
from concurrent.futures import ProcessPoolExecutor

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


def _silence_vtk() -> None:
    """Suppress VTK warnings and logger output to console."""
    try:
        vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    except Exception:
        pass
    try:
        vtk.vtkObject.GlobalWarningDisplayOff()
    except Exception:
        pass


# Silence VTK warnings on import
_silence_vtk()


def retrieve_all_shape_files(input_dir_path: str) -> list[str]:
    """Find every OBJ file recursively under the input directory."""
    shape_files = []

    for root, _, files in os.walk(input_dir_path):
        for filename in files:
            if filename.lower().endswith(".obj"):
                shape_files.append(os.path.join(root, filename))

    return sorted(shape_files)


def calculate_mesh_descriptors(shape: vedo.Mesh) -> dict:
    # Calculate shared quantities once.
    volume = __approx_volume(shape)
    area = calculate_surface_area(
        shape=shape,
    )
    pca = __calculate_pca(shape)
    convex_hull = vedo.ConvexHull(shape)

    # Reuse the precomputed values when possible.
    compactness = calculate_compactness(
        shape=shape,
        volume=volume,
        area=area,
    )

    rectangularity = calculate_rectangularity(
        shape=shape,
        shape_volume=volume,
    )

    diameter = calculate_diameter(
        shape=shape,
        convex_hull=convex_hull,
    )

    convexity = calculate_convexity(
        shape=shape,
        volume=volume,
        convex_hull=convex_hull,
    )

    eccentricity = calculate_eccentricity(
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


def calculate_file_descriptors(file_path: str) -> dict:
    shape = __prepare_and_check_file_path(file_path)
    return calculate_mesh_descriptors(shape)


def _process_shape_worker(args: tuple[str, str]) -> tuple[dict, list[str]]:
    """Worker task for multiprocessing descriptor calculation."""
    shape_file, root_abs = args
    relative_path = os.path.relpath(shape_file, root_abs)
    row = {"file_path": relative_path}
    warnings: list[str] = []

    _silence_vtk()

    try:
        descriptors = calculate_file_descriptors(shape_file)
        row.update(descriptors)
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

    return row, warnings


def calculate_database(
    input_dir_path: str,
    output_path: str,
    max_workers: int | None = None,
) -> None:
    """
    Calculate all descriptors for every OBJ file in a directory.

    The input directory structure is preserved in the CSV through the
    relative file_path column. The input meshes are not modified.
    """

    if not os.path.isdir(input_dir_path):
        raise ValueError(f"Input directory does not exist: {input_dir_path}")

    shape_files = retrieve_all_shape_files(input_dir_path)

    if not shape_files:
        raise ValueError(f"No OBJ files found in {input_dir_path}")

    root_abs = os.path.abspath(input_dir_path)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    warnings_path = os.path.splitext(output_path)[0] + "_warnings.csv"

    n_success = 0
    n_errors = 0
    n_warnings = 0

    _silence_vtk()

    tasks = [(shape_file, root_abs) for shape_file in shape_files]
    n_workers = max_workers or os.cpu_count() or 1

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

        if n_workers == 1 or len(tasks) <= 1:
            for task in tqdm(tasks, desc="Calculating descriptors"):
                row, warnings = _process_shape_worker(task)
                if row.get("reason"):
                    n_errors += 1
                else:
                    n_success += 1

                for warning in warnings:
                    warning_writer.writerow(
                        {
                            "file_path": row["file_path"],
                            "warning": warning,
                        }
                    )
                    n_warnings += 1

                writer.writerow(row)
        else:
            print(f"Using {n_workers} worker processes")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                chunksize = max(1, min(16, len(tasks) // (n_workers * 2)))
                for row, warnings in tqdm(
                    executor.map(_process_shape_worker, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc="Calculating descriptors",
                ):
                    if row.get("reason"):
                        n_errors += 1
                    else:
                        n_success += 1

                    for warning in warnings:
                        warning_writer.writerow(
                            {
                                "file_path": row["file_path"],
                                "warning": warning,
                            }
                        )
                        n_warnings += 1

                    writer.writerow(row)

    summary = (
        f"Descriptor calculation complete.\n"
        f"Total meshes: {len(shape_files)}\n"
        f"Successful: {n_success}\n"
        f"Errors: {n_errors}\n"
        f"VTK warnings/errors captured: {n_warnings}\n"
        f"Output CSV: {output_path}\n"
        f"Warnings CSV: {warnings_path}\n"
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
        "-o",
        "--output",
        default="shape_descriptors.csv",
        help="Output CSV path (default: shape_descriptors.csv).",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: all CPU cores).",
    )

    args = parser.parse_args()

    calculate_database(
        input_dir_path=args.input,
        output_path=args.output,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
