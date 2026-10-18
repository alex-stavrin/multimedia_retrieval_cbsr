import argparse
import csv
import os
from typing import Callable

import numpy as np
import trimesh
import vedo

from tqdm import tqdm
from src.helper.database_functions import retrieve_all_shape_files

HOLE_SIZE = 100

# ---
# Loading


def load_mesh(shape_path: str) -> vedo.Mesh:
    if not os.path.exists(shape_path):
        raise ValueError(f"Shape file path does not exist: {shape_path}")

    if not shape_path.endswith(".obj"):
        raise ValueError(f"Shape file path is not of type .obj: {shape_path}")

    return vedo.Mesh(shape_path)


# ---
# Diagnostics


def edge_stats(shape: vedo.Mesh) -> dict:
    """Counts boundary edges (open holes) and non-manifold edges."""
    faces: np.ndarray = np.asarray(shape.cells, dtype=np.int64)
    if len(faces) == 0 or faces.ndim != 2 or faces.shape[1] != 3:
        return {"boundary_edges": -1, "nonmanifold_edges": -1}

    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
    }


def failure_reason(stats: dict) -> str:
    if stats["boundary_edges"] == -1:
        return "mesh is not purely triangles"
    reasons = []
    if stats["nonmanifold_edges"] > 0:
        reasons.append(
            f"{stats['nonmanifold_edges']} non-manifold edges "
            "(hole filling cannot fix these)"
        )
    if stats["boundary_edges"] > 0:
        reasons.append(
            f"{stats['boundary_edges']} boundary edges remain "
            "(holes larger than HOLE_SIZE, or fill failed)"
        )
    return "; ".join(reasons) or "unknown (is_closed false but no bad edges)"


# ---
# Standardization steps
#
# Every step has the signature  vedo.Mesh -> vedo.Mesh  and must not modify
# its input. To add a step: write the function, add it to STEPS.


def preprocess(shape: vedo.Mesh) -> vedo.Mesh:
    """
    Triangulates and merges duplicate vertices. Without this, UV/normal
    seams in the .obj look like holes and fill_holes cannot close them.
    """
    return shape.clone().triangulate().clean()


def close_holes(shape: vedo.Mesh) -> vedo.Mesh:
    """Fills holes. Never raises; the result may still be open."""
    if shape.is_closed():
        return shape
    return shape.fill_holes(size=HOLE_SIZE)


def correct_winding(shape: vedo.Mesh) -> vedo.Mesh:
    """
    Makes triangle winding consistent and the normals point outwards.
    Works best on closed meshes (outward is ambiguous otherwise).
    """
    mesh = trimesh.Trimesh(
        vertices=np.asarray(shape.vertices),
        faces=np.asarray(shape.cells, dtype=np.int64),
        process=False,
    )
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    return vedo.Mesh([mesh.vertices, mesh.faces])


STEPS: dict[str, Callable[[vedo.Mesh], vedo.Mesh]] = {
    "preprocess": preprocess,
    "close_holes": close_holes,
    "correct_winding": correct_winding,
}

DEFAULT_STEPS = ["preprocess", "close_holes", "correct_winding"]


def run_steps(
    shape: vedo.Mesh, steps: dict[str, Callable]
) -> tuple[vedo.Mesh, str]:
    """
    Applies the named steps in the given order. Returns the mesh and a
    trace of the mesh being closed after each step, e.g.
    "preprocess:open > close_holes:closed".
    """
    trace = []
    for name in steps:
        shape = STEPS[name](shape)
        state = "closed" if shape.is_closed() else "open"
        trace.append(f"{name}:{state}")
    return shape, " > ".join(trace)


# ---
# File / directory handling

REPORT_FIELDS = [
    "file_path",
    "closed_raw",
    "closed_after",
    "repaired",
    "boundary_edges_after",
    "nonmanifold_edges_after",
    "trace",
    "reason",
]


def standardize_file(
    input_path: str, output_path: str, steps: dict[str, Callable]
) -> dict:
    """Loads one .obj, applies the steps, writes it. Returns a report row."""
    raw = load_mesh(input_path)
    closed_raw = bool(raw.is_closed())

    result, trace = run_steps(raw, steps)
    closed_after = bool(result.is_closed())
    stats = edge_stats(result)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    vedo.write(result, output_path)

    return {
        "closed_raw": closed_raw,
        "closed_after": closed_after,
        "repaired": (not closed_raw) and closed_after,
        "boundary_edges_after": stats["boundary_edges"],
        "nonmanifold_edges_after": stats["nonmanifold_edges"],
        "trace": trace,
        "reason": "" if closed_after else failure_reason(stats),
    }


def standardize_directory(
    input_dir_path: str,
    output_dir_path: str,
    steps: dict[str, Callable] = STEPS,
) -> None:
    """Processes every shape under input_dir_path, mirroring the folders."""
    all_shape_files = retrieve_all_shape_files(input_dir_path)
    root_abs = os.path.abspath(input_dir_path)

    os.makedirs(output_dir_path, exist_ok=True)
    report_path = os.path.join(output_dir_path, "report.csv")

    n_closed = n_open = n_errors = n_closed_raw = n_repaired = 0

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()

        for shape_file in tqdm(all_shape_files):
            relative_path = os.path.relpath(shape_file, root_abs)
            new_path = os.path.join(output_dir_path, relative_path)
            row = {"file_path": relative_path}

            try:
                row.update(standardize_file(shape_file, new_path, steps))
            except Exception as e:
                row["reason"] = f"error: {type(e).__name__}: {e}"
                n_errors += 1
            else:
                n_closed_raw += int(row["closed_raw"])
                n_repaired += int(row["repaired"])
                if row["closed_after"]:
                    n_closed += 1
                else:
                    n_open += 1

            writer.writerow(row)

    summary = (
        f"Steps: {' > '.join(steps)}\n"
        f"Done. {len(all_shape_files)} shapes: "
        f"{n_closed} closed, {n_open} still open, {n_errors} errors.\n"
        f"{n_closed_raw} were closed as loaded and {n_repaired} became "
        f"closed through the steps.\n"
    )
    with open(
        os.path.join(output_dir_path, "report_conclusion.txt"), "w"
    ) as f:
        f.write(summary)
    print(summary + f"Report written to {report_path}")


def standardize_path(
    input_path: str, output_path: str, steps: list[str] | None = None
) -> None:
    """
    Applies the standardization steps (in order) to a file or a whole
    directory, writing to output_path with the same structure.

    - file in -> output_path is a file path, or a directory to put it in
    - dir in  -> output_path is the root directory of the new database
    """
    steps = DEFAULT_STEPS if steps is None else steps

    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"Unknown steps {unknown}. Available: {list(STEPS)}")

    if os.path.isdir(input_path):
        standardize_directory(input_path, output_path, steps)

    elif os.path.isfile(input_path):
        if os.path.isdir(output_path) or not output_path.endswith(".obj"):
            output_path = os.path.join(
                output_path, os.path.basename(input_path)
            )

        single_stats_row = standardize_file(input_path, output_path, steps)
        print(f"Wrote {output_path}: {single_stats_row}")

    else:
        raise ValueError(f"Input path does not exist: {input_path}")


# ---
# main


def main():
    parser = argparse.ArgumentParser(
        description="Standardize a mesh file or a whole database of meshes."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to a .obj file or the root directory of the database",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file/directory (new root directory for a database)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        nargs="+",
        choices=list(STEPS),
        default=DEFAULT_STEPS,
        help="Steps to apply, in order (default: %(default)s)",
    )
    args = parser.parse_args()
    standardize_path(args.input, args.output, args.steps)


if __name__ == "__main__":
    main()
