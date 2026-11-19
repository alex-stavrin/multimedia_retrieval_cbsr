import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import vedo
from tqdm import tqdm

from src.types.mesh_types import Box
from src.step_2._5.normalizer import (
    TOLERANCE,
    center,
    scale_to_unit_cube,
    verify_saved_database,
)


@dataclass
class FullNormalizationReport:
    shape_path: str
    barycenter_before: np.ndarray
    barycenter_after: np.ndarray
    bounding_box_before: Box
    bounding_box_after: Box
    eigenvalues: (
        np.ndarray
    )  # descending, from the PCA alignment step (pre-scaling units)
    flip_signs: np.ndarray  # +1/-1 applied to x, y, z after alignment
    scale_factor: float = None
    variance_after: np.ndarray = field(
        default=None
    )  # var(x), var(y), var(z) after full normalization


def align_mesh(mesh: vedo.Mesh) -> np.ndarray:
    """Rotate the mesh so its two largest-variance directions (from PCA on
    the vertex covariance matrix) become the x and y axes. Mutates `mesh`
    and returns the sorted eigenvalues (largest first)."""
    vertices = mesh.vertices

    covariance = np.cov(vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # eigh returns ascending eigenvalues; we want largest-variance axis first.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    e1, e2 = eigenvectors[:, order[0]], eigenvectors[:, order[1]]

    # Recompute the third axis as e1 x e2 rather than trusting eigh's third
    # eigenvector: eigh's per-vector sign is arbitrary, and using the raw
    # third eigenvector can make the resulting basis left-handed (a mirror
    # rather than a rotation), which would flip the shape's chirality.
    e3 = np.cross(e1, e2)

    rotation = np.vstack([e1, e2, e3])  # rows = new x, y, z axes in old coords
    mesh.vertices = vertices @ rotation.T

    return eigenvalues


def flip_mesh(mesh: vedo.Mesh) -> np.ndarray:
    """Flip the mesh along any of the x, y, z axes whose moment test is
    negative, so mass consistently lands on the positive side of every
    axis. Mutates `mesh` and returns the +1/-1 sign applied per axis."""
    vertices = mesh.vertices

    moments = np.sum(np.sign(vertices) * vertices**2, axis=0)
    flip_signs = np.where(moments < 0, -1.0, 1.0)

    mesh.vertices = vertices * flip_signs

    # An odd number of flips is a mirror image, which reverses the corner
    # order of every triangle and turns the surface inside-out (negative
    # volume, inward normals). Reversing the faces restores the orientation.
    if np.prod(flip_signs) < 0:
        mesh.reverse()

    return flip_signs


def full_normalize(mesh: vedo.Mesh) -> FullNormalizationReport:
    """Run the complete four-step normalization pipeline on `mesh`, in the
    order translate -> align -> flip -> scale. Mutates `mesh` in place.
    """
    barycenter_before = mesh.center_of_mass()
    bounding_box_before = Box(*mesh.bounds())

    center(mesh)
    eigenvalues = align_mesh(mesh)
    flip_signs = flip_mesh(mesh)
    scale_factor = scale_to_unit_cube(mesh)

    variance_after = np.var(mesh.vertices, axis=0)

    return FullNormalizationReport(
        shape_path="",
        barycenter_before=barycenter_before,
        barycenter_after=mesh.center_of_mass(),
        bounding_box_before=bounding_box_before,
        bounding_box_after=Box(*mesh.bounds()),
        eigenvalues=eigenvalues,
        flip_signs=flip_signs,
        scale_factor=scale_factor,
        variance_after=variance_after,
    )


def full_normalize_file(input_path: str, output_path: str) -> FullNormalizationReport:
    """Load a mesh from disk, fully normalize it, and write the result to `output_path`."""
    if not os.path.exists(input_path):
        raise ValueError(f"File does not exist: {input_path}")

    mesh = vedo.Mesh(input_path)
    report = full_normalize(mesh)
    report.shape_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.write(output_path)

    return report


def _normalize_worker(
    task: tuple[str, str],
) -> tuple[FullNormalizationReport | None, str, str | None]:
    """Worker task to normalize a single file in a background process."""
    input_path, output_path = task
    try:
        report = full_normalize_file(input_path, output_path)
        return report, input_path, None
    except Exception as e:
        return None, input_path, str(e)


def full_normalize_database(
    input_root: str,
    output_root: str,
    max_workers: int | None = None,
) -> list[FullNormalizationReport]:
    """Fully normalize every .obj file under `input_root` in parallel, mirroring the
    class-subfolder layout into `output_root`. Skips files that fail to
    load or normalize rather than aborting the whole run."""
    tasks: list[tuple[str, str]] = []

    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if not filename.endswith(".obj"):
                continue

            input_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path)
            tasks.append((input_path, output_path))

    if not tasks:
        print(f"No .obj files found under {input_root}")
        return []

    tasks.sort()

    n_workers = max_workers or os.cpu_count() or 1
    print(f"Using {n_workers} worker processes")

    reports: list[FullNormalizationReport] = []

    if max_workers == 1:
        for task in tqdm(tasks, desc="Fully normalizing meshes"):
            report, input_path, error = _normalize_worker(task)
            if error is not None:
                tqdm.write(f"Skipping {input_path}: {error}")
            elif report is not None:
                reports.append(report)
        return reports

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_normalize_worker, task) for task in tasks]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fully normalizing meshes",
        ):
            try:
                report, input_path, error = future.result()
            except Exception as e:
                tqdm.write(f"Worker failure: {e}")
                continue

            if error is not None:
                tqdm.write(f"Skipping {input_path}: {error}")
            elif report is not None:
                reports.append(report)

    reports.sort(key=lambda r: r.shape_path)
    return reports


def summarize_reports(reports: list[FullNormalizationReport]) -> None:
    """Recompute, for every shape, the value each normalization step is
    supposed to fix and print how close it landed to its target: barycenter
    at the origin (translation), largest bounding-box dimension equal to 1
    (scaling), and variance ordered x >= y >= z (alignment)."""
    if not reports:
        print("No shapes were normalized.")
        return

    barycenter_distances = np.array(
        [np.linalg.norm(r.barycenter_after) for r in reports]
    )
    max_extents = np.array(
        [
            max(
                r.bounding_box_after.x_max - r.bounding_box_after.x_min,
                r.bounding_box_after.y_max - r.bounding_box_after.y_min,
                r.bounding_box_after.z_max - r.bounding_box_after.z_min,
            )
            for r in reports
        ]
    )

    variances = np.array([r.variance_after for r in reports])
    correctly_ordered = np.sum(
        (variances[:, 0] >= variances[:, 1] - 1e-5)
        & (variances[:, 1] >= variances[:, 2] - 1e-5)
    )

    n = len(reports)
    print(f"Normalized {n} shapes")
    print(
        f"Barycenter distance from origin: mean={barycenter_distances.mean():.2e}, "
        f"max={barycenter_distances.max():.2e}, "
        f"within {TOLERANCE:g}: {np.sum(barycenter_distances < TOLERANCE)}/{n}"
    )
    print(
        f"Largest bounding-box dimension:  min={max_extents.min():.9f}, "
        f"max={max_extents.max():.9f}, "
        f"within {TOLERANCE:g} of 1: {np.sum(np.abs(max_extents - 1) < TOLERANCE)}/{n}"
    )
    print(f"Variance ordered (x>=y>=z) after alignment: {correctly_ordered}/{n}")
    print(f"Mean variance after normalization (x, y, z): {variances.mean(axis=0)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fully normalize 3D shapes: center, PCA-align, flip via "
        "the moment test, then scale."
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
    parser.add_argument(
        "-j",
        "-w",
        "--workers",
        dest="workers",
        type=int,
        default=None,
        help="Number of worker processes to use (default: CPU count)",
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        reports = full_normalize_database(
            args.input, args.output, max_workers=args.workers
        )
        summarize_reports(reports)
        verify_saved_database(args.output, check_alignment=True)
    else:
        report = full_normalize_file(args.input, args.output)
        print(report)


if __name__ == "__main__":
    main()
