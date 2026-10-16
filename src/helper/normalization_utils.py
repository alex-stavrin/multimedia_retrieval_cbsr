"""Pieces shared by the normalization scripts (steps 2.5 and 3.1): running a
script on a file or a whole database, summarizing the result, and re-checking
the saved files."""

import argparse
import os
import re
from typing import Callable

import numpy as np

from src.helper.mesh_types import Box

# A recomputed quantity counts as correct if it is within this distance of its
# target. Far above the ~1e-7 rounding from single-precision/text storage,
# far below any real normalization error.
TOLERANCE = 1e-6

# Near-equal variances (e.g. a table whose width is about its depth) can swap
# by rounding noise, so the x >= y >= z check allows this much slack.
ORDER_SLACK = 1e-5


def largest_side(box: Box) -> float:
    return max(box.x_max - box.x_min, box.y_max - box.y_min, box.z_max - box.z_min)


def process_database(
    input_root: str, output_root: str, process_file: Callable[[str, str], object]
) -> list:
    """Call `process_file(input_path, output_path)` for every .obj file under
    `input_root`, mirroring the class sub-folders into `output_root`. A shape
    that fails is reported and skipped instead of stopping the whole run.
    Returns what `process_file` returned for the shapes that worked."""
    results = []

    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if not filename.endswith(".obj"):
                continue

            input_path = os.path.join(dirpath, filename)
            output_path = os.path.join(
                output_root, os.path.relpath(input_path, input_root)
            )
            try:
                results.append(process_file(input_path, output_path))
            except Exception as e:
                print(f"Skipping {input_path}: {e}")

    return results


def summarize_normalization(reports: list, action: str) -> None:
    """Print how close the barycenter of every processed shape landed to the
    origin and its largest bounding-box side to 1."""
    if not reports:
        print("No shapes were processed.")
        return

    distances = np.array([np.linalg.norm(r.barycenter_after) for r in reports])
    sides = np.array([largest_side(r.bounding_box_after) for r in reports])
    n = len(reports)

    print(f"{action} {n} shapes")
    print(
        f"Barycenter distance from origin: mean={distances.mean():.2e}, "
        f"max={distances.max():.2e}, "
        f"within {TOLERANCE:g}: {np.sum(distances < TOLERANCE)}/{n}"
    )
    print(
        f"Largest bounding-box dimension:  min={sides.min():.9f}, "
        f"max={sides.max():.9f}, "
        f"within {TOLERANCE:g} of 1: {np.sum(np.abs(sides - 1) < TOLERANCE)}/{n}"
    )


def read_obj_vertices(path: str) -> np.ndarray:
    """Read only the vertex positions of an .obj file, with no mesh library,
    so checks built on it share no code (or bugs) with the code they verify."""
    with open(path, "rb") as f:
        data = f.read()
    rows = re.findall(rb"^v[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)", data, re.M)
    return np.array(rows).astype(np.float64)


def measure(vertices: np.ndarray) -> dict:
    """Recompute, from the vertices alone, what normalization should fix."""
    variance = vertices.var(axis=0)
    covariance = np.cov(vertices, rowvar=False)
    moments = np.sum(np.sign(vertices) * vertices**2, axis=0)
    off_diagonal = max(
        abs(covariance[0, 1]), abs(covariance[0, 2]), abs(covariance[1, 2])
    )
    return {
        "barycenter": np.linalg.norm(vertices.mean(axis=0)),
        "side": (vertices.max(axis=0) - vertices.min(axis=0)).max(),
        # zero exactly when the coordinate axes are the principal axes
        "off_diagonal": off_diagonal / max(covariance[0, 0], 1e-300),
        "ordered": variance[0] >= variance[1] - ORDER_SLACK
        and variance[1] >= variance[2] - ORDER_SLACK,
        # relative to the total second moment, so it doesn't depend on size
        "flipped": bool(np.all(moments >= -TOLERANCE * np.sum(vertices**2))),
    }


def verify_saved_database(root: str, check_alignment: bool = False) -> bool:
    """Re-read every normalized .obj file under `root` and recompute, from the
    saved vertices alone, that the barycenter is at the origin and the largest
    bounding-box side is 1. With `check_alignment` (Step 3.1) also that the
    covariance matrix is diagonal with entries ordered x >= y >= z (PCA
    alignment) and that the moment test is non-negative on every axis
    (flipping). Returns True only if every shape passes every check."""
    paths = [
        os.path.join(dirpath, f)
        for dirpath, _, filenames in os.walk(root)
        for f in filenames
        if f.endswith(".obj")
    ]
    if not paths:
        print(f"No .obj files found under {root}")
        return False

    measured = [measure(read_obj_vertices(path)) for path in paths]
    n = len(measured)

    def column(key: str) -> np.ndarray:
        return np.array([m[key] for m in measured])

    distances, sides = column("barycenter"), column("side")
    barycenter_ok = int(np.sum(distances < TOLERANCE))
    side_ok = int(np.sum(np.abs(sides - 1) < TOLERANCE))

    print(f"Re-read {n} saved shapes and recomputed from the files alone:")
    print(
        f"  barycenter distance from origin: max={distances.max():.2e}, "
        f"within {TOLERANCE:g}: {barycenter_ok}/{n}"
    )
    print(
        f"  largest bounding-box dimension:  min={sides.min():.9f}, "
        f"max={sides.max():.9f}, within {TOLERANCE:g} of 1: {side_ok}/{n}"
    )
    all_ok = barycenter_ok == n and side_ok == n

    if check_alignment:
        off_diagonal = column("off_diagonal")
        diagonal_ok = int(np.sum(off_diagonal < TOLERANCE))
        ordered_ok = int(np.sum(column("ordered")))
        flipped_ok = int(np.sum(column("flipped")))
        print(
            f"  off-diagonal covariance / largest variance: "
            f"max={off_diagonal.max():.2e}, within {TOLERANCE:g}: {diagonal_ok}/{n}"
        )
        print(f"  variance ordered x>=y>=z: {ordered_ok}/{n}")
        print(f"  moment test non-negative on all axes: {flipped_ok}/{n}")
        all_ok = all_ok and diagonal_ok == n and ordered_ok == n and flipped_ok == n

    return all_ok


def run_cli(
    description: str,
    process_file: Callable[[str, str], object],
    action: str,
    check_alignment: bool = False,
    extra_summary: Callable[[list], None] | None = None,
) -> None:
    """Command line shared by the normalization scripts: `-i` and `-o` are
    either a single .obj file or a whole database folder. For a folder it also
    prints a summary and re-checks the saved files."""
    parser = argparse.ArgumentParser(description=description)
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

    if not os.path.isdir(args.input):
        print(process_file(args.input, args.output))
        return

    reports = process_database(args.input, args.output, process_file)
    summarize_normalization(reports, action)
    if reports and extra_summary is not None:
        extra_summary(reports)
    verify_saved_database(args.output, check_alignment)
