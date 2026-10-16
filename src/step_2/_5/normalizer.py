import os
from dataclasses import dataclass

import numpy as np
import vedo

from src.step_2.mesh_types import Box


@dataclass
class NormalizationReport:
    shape_path: str
    barycenter_before: np.ndarray
    barycenter_after: np.ndarray
    bounding_box_before: Box
    bounding_box_after: Box
    scale_factor: float


def normalize_mesh(mesh: vedo.Mesh) -> NormalizationReport:
    """Translate the mesh's barycenter to the origin, then uniformly scale
    it so its largest bounding-box dimension fits a unit cube. Mutates
    and returns info about `mesh`."""
    barycenter_before = mesh.center_of_mass()
    bounding_box_before = Box(*mesh.bounds())

    mesh.shift(*(-barycenter_before))

    bounds = mesh.bounds()
    extents = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    max_extent = max(extents)
    if max_extent <= 0:
        raise ValueError(
            "Mesh has zero extent on every axis, cannot scale to a unit cube."
        )

    scale_factor = 1.0 / max_extent
    # origin=False scales about world (0,0,0), i.e. the barycenter we just
    # shifted there. origin=True (the default) would instead pivot around
    # the mesh's own transform position and re-introduce an offset.
    mesh.scale(scale_factor, origin=False)

    return NormalizationReport(
        shape_path="",
        barycenter_before=barycenter_before,
        barycenter_after=mesh.center_of_mass(),
        bounding_box_before=bounding_box_before,
        bounding_box_after=Box(*mesh.bounds()),
        scale_factor=scale_factor,
    )


def normalize_shape_file(input_path: str, output_path: str) -> NormalizationReport:
    """Load a mesh from disk, normalize it, and write the result to `output_path`."""
    if not os.path.exists(input_path):
        raise ValueError(f"File does not exist: {input_path}")

    mesh = vedo.Mesh(input_path)
    report = normalize_mesh(mesh)
    report.shape_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.write(output_path)

    return report


def normalize_database(input_root: str, output_root: str) -> list[NormalizationReport]:
    """Normalize every .obj file under `input_root`, mirroring the
    class-subfolder layout into `output_root`. Skips files that fail to
    load or normalize rather than aborting the whole run."""
    reports: list[NormalizationReport] = []

    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if not filename.endswith(".obj"):
                continue

            input_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path)

            try:
                reports.append(normalize_shape_file(input_path, output_path))
            except Exception as e:
                print(f"Skipping {input_path}: {e}")

    return reports


def summarize_reports(reports: list[NormalizationReport]) -> None:
    """Print aggregate stats confirming normalization worked across the
    database: barycenters should sit at the origin and every shape's
    largest bounding-box dimension should be 1 (a unit cube)."""
    if not reports:
        print("No shapes were normalized.")
        return

    barycenter_distances = [float(np.linalg.norm(r.barycenter_after)) for r in reports]
    max_extents = [
        max(
            r.bounding_box_after.x_max - r.bounding_box_after.x_min,
            r.bounding_box_after.y_max - r.bounding_box_after.y_min,
            r.bounding_box_after.z_max - r.bounding_box_after.z_min,
        )
        for r in reports
    ]

    print(f"Normalized {len(reports)} shapes")
    print(
        f"Barycenter distance from origin: mean={np.mean(barycenter_distances):.2e}, max={np.max(barycenter_distances):.2e}"
    )
    print(
        f"Largest bounding-box dimension:  mean={np.mean(max_extents):.6f}, max={np.max(max_extents):.6f}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalize 3D shapes: center each barycenter at the origin and scale to fit a unit cube."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to an OBJ file, or a database root directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to write the normalized OBJ file, or output root directory",
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        reports = normalize_database(args.input, args.output)
        summarize_reports(reports)
    else:
        report = normalize_shape_file(args.input, args.output)
        print(report)


if __name__ == "__main__":
    main()
