import os
import re
from dataclasses import dataclass

import numpy as np
import vedo

from src.helper.mesh_types import Box

# A recomputed quantity counts as correct if it is within this distance of its
# target. Far above the ~1e-7 rounding from single-precision/text storage,
# far below any real normalization error.
TOLERANCE = 1e-6


@dataclass
class CenteringReport:
    shape_path: str
    barycenter_before: np.ndarray
    barycenter_after: np.ndarray
    bounding_box_before: Box
    bounding_box_after: Box
    scale_factor: float


def center(mesh: vedo.Mesh) -> np.ndarray:
    """Translate the mesh's barycenter to the origin. Mutates `mesh` and
    returns the barycenter that was subtracted."""
    barycenter = mesh.center_of_mass()
    mesh.shift(*(-barycenter))
    return barycenter


def scale_to_unit_cube(mesh: vedo.Mesh) -> float:
    """Uniformly scale the mesh, about the world origin, so its current
    axis-aligned bounding box's largest dimension becomes exactly 1.

    This must run AFTER any rotation/alignment step (e.g. Step 3.1's PCA
    alignment), not before: the bounding box computed here depends on the
    mesh's current orientation, so scaling before alignment would give two
    instances of the same shape -- differing only by their original
    rotation -- different scale factors. Mutates `mesh` and returns the
    scale factor applied."""
    bounds = mesh.bounds()
    extents = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    max_extent = max(extents)
    if max_extent <= 0:
        raise ValueError(
            "Mesh has zero extent on every axis, cannot scale to a unit cube."
        )

    scale_factor = 1.0 / max_extent
    # origin=False scales about world (0,0,0) rather than the mesh's own
    # transform position (vedo's default, origin=True), which would pivot
    # around the wrong point and reintroduce an offset.
    mesh.scale(scale_factor, origin=False)
    return scale_factor


def center_and_scale(mesh: vedo.Mesh) -> CenteringReport:
    """Translate the mesh's barycenter to the origin, then uniformly scale
    it so its largest bounding-box dimension fits a unit cube. Mutates
    and returns info about `mesh`.

    This is Step 2.5 in isolation (centering immediately followed by
    scaling, with no alignment in between). For the full, correctly-ordered
    pipeline used from Step 3.1 onwards, call `center()`, then alignment
    and flipping, then `scale_to_unit_cube()` separately -- see
    `src.step_3.full_normalization.full_normalize`.
    """
    barycenter_before = mesh.center_of_mass()
    bounding_box_before = Box(*mesh.bounds())

    center(mesh)
    scale_factor = scale_to_unit_cube(mesh)

    return CenteringReport(
        shape_path="",
        barycenter_before=barycenter_before,
        barycenter_after=mesh.center_of_mass(),
        bounding_box_before=bounding_box_before,
        bounding_box_after=Box(*mesh.bounds()),
        scale_factor=scale_factor,
    )


def center_and_scale_file(input_path: str, output_path: str) -> CenteringReport:
    """Load a mesh from disk, center and scale it, and write the result to
    `output_path`."""
    if not os.path.exists(input_path):
        raise ValueError(f"File does not exist: {input_path}")

    mesh = vedo.Mesh(input_path)
    report = center_and_scale(mesh)
    report.shape_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.write(output_path)

    return report


def center_and_scale_database(
    input_root: str, output_root: str
) -> list[CenteringReport]:
    """Center and scale every .obj file under `input_root`, mirroring the
    class-subfolder layout into `output_root`. Skips files that fail to
    load or process rather than aborting the whole run."""
    reports: list[CenteringReport] = []

    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if not filename.endswith(".obj"):
                continue

            input_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path)

            try:
                reports.append(center_and_scale_file(input_path, output_path))
            except Exception as e:
                print(f"Skipping {input_path}: {e}")

    return reports


def summarize_reports(reports: list[CenteringReport]) -> None:
    """Print, over every processed shape, how close the barycenter landed to
    the origin and the largest bounding-box dimension to 1."""
    if not reports:
        print("No shapes were centered/scaled.")
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

    n = len(reports)
    print(f"Centered and scaled {n} shapes")
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


def read_obj_vertices(path: str) -> np.ndarray:
    """Read only the vertex positions of an .obj file, with no mesh library,
    so checks built on it don't share any code (or bugs) with the code they
    verify."""
    with open(path, "rb") as f:
        data = f.read()
    rows = re.findall(rb"^v[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)", data, re.M)
    return np.array(rows).astype(np.float64)


def verify_saved_database(root: str, check_alignment: bool = False) -> bool:
    """Re-read every normalized .obj file under `root` and recompute, from
    the saved vertices alone, the quantities normalization is meant to fix:
    barycenter at the origin, largest bounding-box dimension equal to 1 and,
    if `check_alignment` (Step 3.1): the covariance matrix of the vertices is
    diagonal with entries ordered x >= y >= z (PCA alignment) and the moment
    test sum(sign(v) * v^2) is non-negative on every axis (flipping). Returns
    True only if every shape passes every check."""
    barycenter_distances, max_extents, ordered = [], [], []
    off_diagonal, flipped_ok = [], []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".obj"):
                continue
            vertices = read_obj_vertices(os.path.join(dirpath, filename))
            barycenter_distances.append(np.linalg.norm(vertices.mean(axis=0)))
            max_extents.append((vertices.max(axis=0) - vertices.min(axis=0)).max())
            variance = vertices.var(axis=0)
            # Near-equal variances (e.g. a table with width ~ depth) can swap
            # by rounding noise, so allow a small slack on the ordering.
            ordered.append(
                variance[0] >= variance[1] - 1e-5 and variance[1] >= variance[2] - 1e-5
            )
            # Off-diagonal covariance relative to the largest variance: zero
            # exactly when the coordinate axes are the principal axes.
            covariance = np.cov(vertices, rowvar=False)
            off_diagonal.append(
                max(abs(covariance[0, 1]), abs(covariance[0, 2]), abs(covariance[1, 2]))
                / max(covariance[0, 0], 1e-300)
            )
            # Moment test per axis, relative to the total second moment so the
            # check is independent of how the shape happens to be sized.
            moments = np.sum(np.sign(vertices) * vertices**2, axis=0)
            flipped_ok.append(bool(np.all(moments >= -TOLERANCE * np.sum(vertices**2))))

    n = len(barycenter_distances)
    if n == 0:
        print(f"No .obj files found under {root}")
        return False

    barycenter_distances = np.array(barycenter_distances)
    max_extents = np.array(max_extents)
    barycenter_ok = int(np.sum(barycenter_distances < TOLERANCE))
    extent_ok = int(np.sum(np.abs(max_extents - 1) < TOLERANCE))

    print(f"Re-read {n} saved shapes and recomputed from the files alone:")
    print(
        f"  barycenter distance from origin: max={barycenter_distances.max():.2e}, "
        f"within {TOLERANCE:g}: {barycenter_ok}/{n}"
    )
    print(
        f"  largest bounding-box dimension:  min={max_extents.min():.9f}, "
        f"max={max_extents.max():.9f}, within {TOLERANCE:g} of 1: {extent_ok}/{n}"
    )
    all_ok = barycenter_ok == n and extent_ok == n

    if check_alignment:
        off_diagonal = np.array(off_diagonal)
        diagonal_ok = int(np.sum(off_diagonal < TOLERANCE))
        ordered_ok = int(np.sum(ordered))
        flip_ok = int(np.sum(flipped_ok))
        print(
            f"  off-diagonal covariance / largest variance: max={off_diagonal.max():.2e}, "
            f"within {TOLERANCE:g}: {diagonal_ok}/{n}"
        )
        print(f"  variance ordered x>=y>=z: {ordered_ok}/{n}")
        print(f"  moment test non-negative on all axes: {flip_ok}/{n}")
        all_ok = all_ok and diagonal_ok == n and ordered_ok == n and flip_ok == n

    return all_ok


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Center and scale 3D shapes: barycenter to the origin, "
        "scale to fit a unit cube."
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
        help="Path to write the result, or output root directory",
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        reports = center_and_scale_database(args.input, args.output)
        summarize_reports(reports)
        verify_saved_database(args.output)
    else:
        report = center_and_scale_file(args.input, args.output)
        print(report)


if __name__ == "__main__":
    main()
