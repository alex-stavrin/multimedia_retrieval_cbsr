import argparse
import csv
import os

import numpy as np
import pymeshfix
import trimesh
import vedo

from tqdm import tqdm
from src.helper.database_functions import retrieve_all_shape_files

# ---
# Diagnostics


def diagnose(path):
    m = trimesh.load(path, process=True, force="mesh")
    _, counts = np.unique(m.edges_sorted, axis=0, return_counts=True)
    return {
        "watertight": m.is_watertight,
        "winding_consistent": m.is_winding_consistent,
        "components": len(m.split(only_watertight=False)),
        "boundary_edges": int((counts == 1).sum()),  # open holes
        "nonmanifold_edges": int(
            (counts > 2).sum()
        ),  # hole filling can't fix these
        "faces": len(m.faces),
    }


# ---
# Individual mesh steps (each does exactly one thing)


def load_mesh(shape_path: str) -> vedo.Mesh:
    if not os.path.exists(shape_path):
        raise ValueError(f"Shape file path does not exist: {shape_path}")

    if not shape_path.endswith(".obj"):
        raise ValueError(f"Shape file path is not of type .obj: {shape_path}")

    return vedo.Mesh(shape_path)


def triangulate_mesh(shape: vedo.Mesh) -> vedo.Mesh:
    """Converts all faces (quads, polygons) to triangles."""
    return shape.clone().triangulate()


def clean_mesh(shape: vedo.Mesh) -> vedo.Mesh:
    """Merges duplicate vertices and drops degenerate faces."""
    return shape.clean()


def is_closed(shape: vedo.Mesh) -> bool:
    return bool(shape.is_closed())


def close_holes_vedo(shape: vedo.Mesh) -> vedo.Mesh:
    """Fills holes with vedo/VTK. Returns the (possibly still open) mesh."""
    return shape.fill_holes(size=1000000)


def close_holes_trimesh(shape: vedo.Mesh) -> vedo.Mesh:
    """
    Fixes winding/normals and fills holes with trimesh.
    Returns a vedo mesh (possibly still open).
    """
    mesh = trimesh.Trimesh(
        vertices=np.asarray(shape.vertices),
        faces=np.asarray(shape.cells),
        process=True,
    )
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    return vedo.Mesh([mesh.vertices, mesh.faces])


def close_holes_pymeshfix(
    shape: vedo.Mesh, max_face_loss: float = 0.2
) -> vedo.Mesh:
    """
    Repairs the mesh with pymeshfix while keeping all components.
    Raises RuntimeError if the repair removes more than `max_face_loss`
    (fraction) of the faces, since that likely destroyed the shape.
    """
    verts = np.asarray(shape.vertices, dtype=np.float64)
    faces = np.asarray(shape.cells, dtype=np.int32)

    fixer = pymeshfix.MeshFix(verts, faces)
    fixer.repair(remove_smallest_components=False)

    new_verts = np.asarray(fixer.points)
    new_faces = np.asarray(fixer.faces)
    if new_faces.ndim == 1:
        new_faces = new_faces.reshape(-1, 4)[:, 1:]

    if len(new_faces) == 0:
        raise RuntimeError("pymeshfix removed all faces")

    face_loss = 1 - len(new_faces) / len(faces)
    if face_loss > max_face_loss:
        raise RuntimeError(
            f"pymeshfix removed {face_loss:.0%} of the faces; keeping original"
        )

    return vedo.Mesh([new_verts, new_faces])


def close_holes(shape: vedo.Mesh) -> vedo.Mesh:
    """
    Tries to close holes with increasingly heavy methods, stopping as soon
    as the mesh is closed. Never raises: if nothing manages to close the
    mesh, the best attempt so far is returned (check is_closed on it).
    """
    if is_closed(shape):
        return shape

    shape = close_holes_vedo(shape)
    if is_closed(shape):
        return shape

    shape = close_holes_trimesh(shape)
    if is_closed(shape):
        return shape

    try:
        shape = close_holes_pymeshfix(shape)
    except Exception:
        pass

    return shape


# ---
# Pipeline for one mesh


def repair_mesh(shape: vedo.Mesh) -> vedo.Mesh:
    """
    Applies repair steps one by one, only while the mesh is not closed.
    Never raises; the result may still be open.
    Add new steps to the list (cheapest first).
    """
    steps = [clean_mesh, close_holes]

    for step in steps:
        if is_closed(shape):
            break
        shape = step(shape)

    return shape


def standardize_mesh(shape: vedo.Mesh) -> tuple[vedo.Mesh, dict]:
    """
    Triangulates the mesh, then repairs it if it is not closed.
    Returns the mesh and a small report on what happened.
    """
    shape = triangulate_mesh(shape)
    closed_before = is_closed(shape)

    shape = repair_mesh(shape)
    closed_after = is_closed(shape)

    report = {
        "closed_before": closed_before,
        "closed_after": closed_after,
        "repaired": (not closed_before) and closed_after,
    }
    return shape, report


def standardize_file(input_path: str, output_path: str) -> dict:
    """
    Loads one .obj, standardizes it and always writes the result,
    whether or not it ended up closed. Returns the report.
    """
    shape = load_mesh(input_path)
    shape, report = standardize_mesh(shape)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    vedo.write(shape, output_path)
    return report


# ---
# Database level

REPORT_FIELDS = [
    "file_path",
    "closed_before",
    "closed_after",
    "repaired",
    "error",
]


def standardize_directory(input_dir_path: str, output_dir_path: str) -> str:
    """
    Standardizes every shape under input_dir_path, mirroring the folder
    structure in output_dir_path, and writes standardization_report.csv.
    Returns the path to the report. Only files that cannot be loaded or
    written at all are listed with an error (and not written).
    """
    all_shape_files = retrieve_all_shape_files(input_dir_path)
    root_abs = os.path.abspath(input_dir_path)

    os.makedirs(output_dir_path, exist_ok=True)
    report_path = os.path.join(output_dir_path, "standardization_report.csv")

    n_closed = n_repaired = n_open = n_errors = 0

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()

        for shape_file in tqdm(all_shape_files):
            relative_path = os.path.relpath(shape_file, root_abs)
            new_path = os.path.join(output_dir_path, relative_path)
            row = {"file_path": relative_path, "error": ""}

            try:
                row.update(standardize_file(shape_file, new_path))
            except Exception as e:  # vtk/trimesh can raise many types
                row["error"] = f"{type(e).__name__}: {e}"
                n_errors += 1
            else:
                if row["closed_after"]:
                    n_closed += 1
                    n_repaired += int(row["repaired"])
                else:
                    n_open += 1

            writer.writerow(row)

    print(
        f"Done. {len(all_shape_files)} shapes: "
        f"{n_closed} closed ({n_repaired} of them repaired), "
        f"{n_open} still open, {n_errors} errors."
    )
    print(f"Report written to {report_path}")
    return report_path


def standardize_path(input_path: str, output_path: str) -> None:
    """
    Takes either a single .obj file or a database directory and
    standardizes accordingly.

    - file in  -> output_path is a file path, or a directory to put it in
    - dir in   -> output_path is the root directory of the new database
    """
    if os.path.isdir(input_path):
        standardize_directory(input_path, output_path)

    elif os.path.isfile(input_path):
        if os.path.isdir(output_path) or not output_path.endswith(".obj"):
            output_path = os.path.join(
                output_path, os.path.basename(input_path)
            )

        report = standardize_file(input_path, output_path)
        print(f"Wrote {output_path}: {report}")

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
        help="Output file/directory (new root directory for a database)",
    )
    parser.add_argument(
        "-d",
        "--diagnose",
        action="store_true",
        help="Diagnose a mesh from file",
    )
    args = parser.parse_args()

    if args.diagnose:
        print(diagnose(args.input))
    else:
        if args.output is None:
            parser.error("--output is required unless --diagnose is given")
        standardize_path(args.input, args.output)


if __name__ == "__main__":
    main()
