import os
from dataclasses import dataclass

import numpy as np
import vedo

from src.types.mesh_types import Box
from src.helper.normalization_utils import ORDER_SLACK, run_cli
from src.step_2._5.normalizer import center, scale_to_unit_cube


@dataclass
class FullNormalizationReport:
    barycenter_before: np.ndarray
    barycenter_after: np.ndarray
    bounding_box_before: Box
    bounding_box_after: Box
    eigenvalues: np.ndarray  # PCA eigenvalues, largest first, before scaling
    flip_signs: np.ndarray  # +1 or -1 applied to x, y, z
    scale_factor: float
    variance_after: np.ndarray  # variance of x, y, z after normalization
    shape_path: str = ""


def align_mesh(mesh: vedo.Mesh) -> np.ndarray:
    """Rotate the mesh so its two largest-variance directions (from PCA on
    the vertex covariance matrix) become the x and y axes. Mutates `mesh`
    and returns the sorted eigenvalues (largest first)."""
    vertices = mesh.vertices

    covariance = np.cov(vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # eigh returns ascending eigenvalues; we want the largest-variance axis first.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    e1, e2 = eigenvectors[:, order[0]], eigenvectors[:, order[1]]

    # Recompute the third axis as e1 x e2 rather than trusting eigh's third
    # eigenvector: its sign is arbitrary, and the raw vector can make the
    # basis left-handed (a mirror rather than a rotation).
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
    """Run the four normalization steps on `mesh`, in this order: translate,
    align, flip, scale. Mutates `mesh`.

    Scaling is last because it uses the bounding box, which only stops
    depending on the shape's original rotation once the shape has been
    aligned."""
    barycenter_before = mesh.center_of_mass()
    bounding_box_before = Box(*mesh.bounds())

    center(mesh)
    eigenvalues = align_mesh(mesh)
    flip_signs = flip_mesh(mesh)
    scale_factor = scale_to_unit_cube(mesh)

    return FullNormalizationReport(
        barycenter_before=barycenter_before,
        barycenter_after=mesh.center_of_mass(),
        bounding_box_before=bounding_box_before,
        bounding_box_after=Box(*mesh.bounds()),
        eigenvalues=eigenvalues,
        flip_signs=flip_signs,
        scale_factor=scale_factor,
        variance_after=np.var(mesh.vertices, axis=0),
    )


def full_normalize_file(input_path: str, output_path: str) -> FullNormalizationReport:
    """Load a mesh, fully normalize it, and write it to `output_path`."""
    if not os.path.exists(input_path):
        raise ValueError(f"File does not exist: {input_path}")

    mesh = vedo.Mesh(input_path)
    report = full_normalize(mesh)
    report.shape_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.write(output_path)

    return report


def print_alignment_summary(reports: list[FullNormalizationReport]) -> None:
    """Print the part of the summary that only exists after alignment."""
    variances = np.array([r.variance_after for r in reports])
    ordered = np.sum(
        (variances[:, 0] >= variances[:, 1] - ORDER_SLACK)
        & (variances[:, 1] >= variances[:, 2] - ORDER_SLACK)
    )
    print(f"Variance ordered (x>=y>=z) after alignment: {ordered}/{len(reports)}")
    print(f"Mean variance after normalization (x, y, z): {variances.mean(axis=0)}")


def main():
    run_cli(
        description="Fully normalize 3D shapes: center, PCA-align, flip via "
        "the moment test, then scale.",
        process_file=full_normalize_file,
        action="Normalized",
        check_alignment=True,
        extra_summary=print_alignment_summary,
    )


if __name__ == "__main__":
    main()
