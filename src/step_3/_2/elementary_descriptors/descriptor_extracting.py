import math
import os

import numpy as np
import pandas as pd

import vedo

from dataclasses import dataclass

DATABASE: pd.DataFrame | None = None


@dataclass
class PCA:
    center: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def __prepare_and_check_file_path(shape: str) -> vedo.Mesh:
    # Type check
    if not isinstance(shape, str):
        raise TypeError(
            "Shape argument must be of type string " f"{shape} with type: {type(shape)}"
        )

    if not shape.endswith(".obj"):
        raise ValueError(f"Mesh must be of filetype .obj: {shape}")

    if not os.path.exists(shape):
        raise ValueError(f"Path to mesh does not exist: {shape}")

    shape = vedo.Mesh(shape)

    return shape


def __approx_volume(shape: vedo.Mesh) -> float:
    vertices = np.asarray(shape.vertices, dtype=np.float64)
    vertices = vertices - vertices.mean(axis=0)
    cells = np.asarray(shape.cells, dtype=np.int64)

    v0, v1, v2 = (
        vertices[cells[:, 0]],
        vertices[cells[:, 1]],
        vertices[cells[:, 2]],
    )
    return abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6)


def __calculate_pca(shape: vedo.Mesh) -> PCA:
    """
    Computes principal component analysis for a 3D point cloud.

    Args:
        points: (N, 3) array of points.

    Returns:
        PCA object containing the center, eigenvalues, and eigenvectors.
    """
    points = np.asarray(shape.points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    if len(points) < 2:
        raise ValueError("PCA requires at least two points")

    if not np.isfinite(points).all():
        raise ValueError("Points must contain only finite values")

    center = points.mean(axis=0)
    centered = points - center

    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort eigenvalues and eigenvectors by descending variance.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Ensure a right-handed coordinate system.
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    return PCA(
        center=center,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def __oriented_bounding_box(
    shape: vedo.Mesh,
    visualise: bool = False,
    pca: PCA | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes an oriented bounding box (OBB) of a mesh using PCA.

    The box axes are the principal axes of the vertex cloud, and the
    box tightly bounds the vertices along those axes.

    Args:
        shape: The mesh to bound.
        visualise: If True, displays the mesh and its bounding box.
        pca: Optional precomputed PCA result. If omitted, PCA is computed
            from the mesh vertices.

    Returns:
        A tuple containing:
            box_center: (3,) array, the center of the box in world space.
            axes: (3, 3) array, the box axes as columns.
            extents: (3,) array, the side lengths along each axis.
    """
    if pca is None:
        pca = __calculate_pca(shape)

    points = np.asarray(shape.points)

    # Use the PCA instance consistently.
    center = pca.center
    axes = pca.eigenvectors

    # Express the points in the PCA coordinate frame.
    local = (points - center) @ axes
    lo = local.min(axis=0)
    hi = local.max(axis=0)

    extents = hi - lo
    box_center = center + axes @ ((lo + hi) / 2)

    if visualise:
        corners_local = np.array(
            [
                [x, y, z]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ]
        )

        # Transform the corners back into world coordinates.
        corners = center + corners_local @ axes.T

        # Connect corners differing in exactly one bit.
        edges = [(i, i | bit) for i in range(8) for bit in (1, 2, 4) if not i & bit]

        starts = corners[[i for i, _ in edges]]
        ends = corners[[j for _, j in edges]]

        box = vedo.Lines(starts, ends, c="black", lw=2)
        vedo.show(shape, box, axes=1)

    return box_center, axes, extents


def calculate_surface_area(
    file_path: str,
    shape: vedo.Mesh | None = None,
) -> float:
    """
    Calculates the provided shapes's surface area.

    If shape provided, uses that shape object instead of recalculating via
    file_path. Does not check whether shape and file_path refer to same object.
    If row_entry provided, uses dataframe instead of recalculating dataframe
    from database.

    Only accurate for comparison if shapes have been size normalized.
    """

    # Type check
    if shape is None:
        shape = __prepare_and_check_file_path(file_path)

    return shape.area()


def calculate_compactness(
    file_path: str,
    shape: vedo.Mesh | None = None,
    volume: float | None = None,
    area: float | None = None,
) -> float:

    # Check if not None, it is correct class
    if shape is not None and not isinstance(shape, vedo.Mesh):
        raise ValueError(
            "shape must be of type vedo.mesh: " f"{shape} with type {type(shape)}"
        )

    shape_object = shape
    if shape_object is None:
        shape_object: vedo.Mesh = __prepare_and_check_file_path(file_path)

    if volume is None:
        volume = __approx_volume(shape_object)
    if area is None:
        area = shape_object.area()

    return (36 * math.pi * volume**2) / area**3


# MAYBE TRY SUBSAMPLING FOR APPROX VOLUME AND OOBB LATER
def calculate_rectangularity(
    file_path: str,
    shape: vedo.Mesh | None = None,
    shape_volume: float | None = None,
) -> float:
    """
    Rectangularity = mesh volume / volume of its oriented bounding box.

    1.0 means the shape fills its box completely (e.g. a cuboid); lower
    values mean more empty space inside the box.
    """
    if shape is None:
        shape = __prepare_and_check_file_path(file_path)

    if shape_volume is None:
        shape_volume = __approx_volume(shape)

    _, _, extents = __oriented_bounding_box(shape)
    box_volume = float(np.prod(extents))

    if box_volume <= 0:
        return 0.0

    rectangularity = shape_volume / box_volume

    return rectangularity


def calculate_diameter(
    file_path: str,
    shape: vedo.Mesh | None = None,
    convex_hull: vedo.ConvexHull | None = None,
) -> float:
    """
    Calculates the diameter of the given shape, which is the maximum euclidean
    distance between two points.

    Uses the convex hull for optimization.
    """
    if shape is None:
        shape = __prepare_and_check_file_path(file_path)

    if convex_hull is None:
        convex_hull = vedo.ConvexHull(shape)
    points = convex_hull.points

    max_squared_distance = 0.0

    for i in range(len(points) - 1):
        differences = points[i + 1 :] - points[i]
        squared_distances = np.einsum("ij,ij->i", differences, differences)

        max_squared_distance = max(
            max_squared_distance,
            float(squared_distances.max()),
        )

    return math.sqrt(max_squared_distance)


def calculate_convexity(
    file_path: str,
    shape: vedo.Mesh | None = None,
    volume: float | None = None,
    convex_hull: vedo.ConvexHull | None = None,
) -> float:
    if shape is None:
        shape = vedo.Mesh(file_path)

    if volume is None:
        volume = __approx_volume(shape)

    if convex_hull is None:
        convex_hull = vedo.ConvexHull(shape)

    convexity = volume / convex_hull.volume()

    return float(convexity)


def calculate_eccentricity(
    file_path: str,
    shape: vedo.Mesh | None = None,
    pca: PCA | None = None,
) -> float:
    if shape is None:
        shape = vedo.Mesh(file_path)

    if pca is None:
        pca = __calculate_pca(shape)

    eccentricity = pca.eigenvalues[0] / pca.eigenvalues[2]

    return float(eccentricity)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Get descriptors from a 3D shape from an OBJ file"
        " + database statistics."
    )

    parser.add_argument(
        "-i", "--input", type=str, help="Path to the OBJ file to analyze."
    )
    parser.add_argument(
        "-d", "--database_path", type=str, help="File path to the database."
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Visualises some steps for verification.",
    )

    args = parser.parse_args()

    visualize: bool = args.visualize

    if args.visualize:
        if os.path.isdir(args.input):
            raise ValueError(f"Visualising only works for directories: {args.input}")

        __oriented_bounding_box(vedo.Mesh(args.input), visualize)
    else:
        # CHECK IF CORRECT
        shape = __prepare_and_check_file_path(args.input)
        area = calculate_surface_area(args.input, args.database_path)
        compactness = calculate_compactness(args.input, shape)
        rectangularity = calculate_rectangularity(args.input, shape)
        diameter = calculate_diameter(args.input, shape)
        convexity = calculate_convexity(args.input, shape)
        eccentricity = calculate_eccentricity(args.input, shape)

        print(
            f"Was closed: {vedo.Mesh(args.input).is_closed()}\n"
            f"Area: {round(area, 3)}\n"
            f"Compactness: {round(compactness, 3)}\n"
            f"Rectangularity: {round(rectangularity, 3)}\n"
            f"Diameter: {round(diameter, 3)}\n"
            f"Convexity: {round(convexity, 3)}\n"
            f"Eccentricity: {round(eccentricity, 3)}\n"
        )


if __name__ == "__main__":
    main()
