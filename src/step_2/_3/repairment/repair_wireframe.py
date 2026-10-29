import argparse
import csv
import os
from typing import Callable

import numpy as np
import trimesh
import vedo

from tqdm import tqdm
from src.helper.database_functions import retrieve_all_shape_files

from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

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


def _boundary_loops(shape: vedo.Mesh) -> list[list[int]]:
    """Extracts simple closed boundary loops from a triangular mesh."""
    faces = np.asarray(shape.cells, dtype=np.int64)

    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        return []

    edges = np.sort(
        np.concatenate(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ]
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    if len(boundary_edges) == 0:
        return []

    # Each vertex in a simple boundary loop must have degree two.
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return []

    loops = []
    remaining = {tuple(edge) for edge in boundary_edges.tolist()}

    while remaining:
        start_edge = next(iter(remaining))
        start, current = start_edge
        loop = [start, current]
        remaining.remove(start_edge)
        previous = start

        while current != start:
            neighbors = adjacency[current]
            nxt = neighbors[0] if neighbors[0] != previous else neighbors[1]

            if nxt == start:
                edge = tuple(sorted((current, nxt)))
                if edge not in remaining:
                    break
                remaining.remove(edge)
                loops.append(loop)
                break

            edge = tuple(sorted((current, nxt)))
            if edge not in remaining:
                break

            remaining.remove(edge)
            loop.append(nxt)
            previous, current = current, nxt

    return loops


def _fan_boundary_loops(shape: vedo.Mesh) -> vedo.Mesh:
    """Closes simple boundary loops by adding a triangle fan to each."""
    vertices = np.asarray(shape.vertices)
    faces = np.asarray(shape.cells, dtype=np.int64)

    loops = _boundary_loops(shape)
    if not loops:
        return shape

    new_vertices = [v for v in vertices]
    new_faces = [f for f in faces]

    for loop in loops:
        if len(loop) < 3:
            continue

        # Add a center vertex and connect it to every boundary edge.
        center = vertices[loop].mean(axis=0)
        center_id = len(new_vertices)
        new_vertices.append(center)

        for i, vertex_id in enumerate(loop):
            next_id = loop[(i + 1) % len(loop)]
            new_faces.append([vertex_id, next_id, center_id])

    result = vedo.Mesh(
        [
            np.asarray(new_vertices),
            np.asarray(new_faces, dtype=np.int64),
        ]
    )

    return result


def close_holes(shape: vedo.Mesh) -> vedo.Mesh:
    """Attempts normal hole filling, then fans remaining simple
    boundary loops."""
    if shape.is_closed():
        return shape

    # Ensure that the fan algorithm operates on triangles.
    result = shape.clone().triangulate().clean()

    # First try Vedo's built-in hole filling.
    try:
        result = result.fill_holes(size=HOLE_SIZE)
    except Exception:
        pass

    if result.is_closed():
        return result

    # Fallback: cap remaining simple boundary loops with triangle fans.
    result = _fan_boundary_loops(result)

    return result


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
    shape: vedo.Mesh, steps: dict[str, Callable] = STEPS
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


def repair_file(
    input_path: str, output_path: str, steps: dict[str, Callable] = STEPS,
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


def _repair_worker(
    job: tuple[str, str, str, list[str]],
) -> dict:
    """Repairs one mesh and returns a CSV-ready report row."""
    input_path, output_path, relative_path, steps = job

    try:
        return {
            "file_path": relative_path,
            **repair_file(input_path, output_path, steps),
        }
    except Exception as e:
        return {
            "file_path": relative_path,
            "reason": f"error: {type(e).__name__}: {e}",
        }


def repair_directory(
    input_dir_path: str,
    output_dir_path: str,
    steps: list[str] = DEFAULT_STEPS,
    max_workers: int | None = None,
) -> None:
    """Repairs a directory of meshes in parallel, preserving its structure."""
    all_shape_files = retrieve_all_shape_files(input_dir_path)
    root_abs = os.path.abspath(input_dir_path)

    os.makedirs(output_dir_path, exist_ok=True)
    report_path = os.path.join(output_dir_path, "report.csv")

    n_closed = n_open = n_errors = n_closed_raw = n_repaired = 0

    # Bound the number of submitted jobs to limit memory usage.
    workers = max_workers or (os.cpu_count() or 1)
    workers = max(1, workers)
    max_pending = workers * 2

    def make_jobs():
        for shape_file in all_shape_files:
            relative_path = os.path.relpath(shape_file, root_abs)
            output_path = os.path.join(output_dir_path, relative_path)
            yield (shape_file, output_path, relative_path, steps)

    def record_result(writer, row):
        nonlocal n_closed, n_open, n_errors, n_closed_raw, n_repaired

        if "closed_raw" not in row:
            n_errors += 1
        else:
            n_closed_raw += int(row["closed_raw"])
            n_repaired += int(row["repaired"])

            if row["closed_after"]:
                n_closed += 1
            else:
                n_open += 1

        writer.writerow(row)

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()

        jobs = iter(make_jobs())

        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = set()

            # Initially submit only a bounded number of jobs.
            for _ in range(max_pending):
                job = next(jobs, None)
                if job is None:
                    break
                pending.add(executor.submit(_repair_worker, job))

            with tqdm(
                total=len(all_shape_files), desc="Repairing meshes"
            ) as pbar:
                while pending:
                    completed, pending = wait(
                        pending, return_when=FIRST_COMPLETED
                    )

                    for future in completed:
                        try:
                            row = future.result()
                        except Exception as e:
                            # Handles unexpected worker/process failures.
                            row = {
                                "file_path": "",
                                "reason": "worker error: "
                                f"{type(e).__name__}: {e}",
                            }

                        record_result(writer, row)
                        pbar.update(1)

                        job = next(jobs, None)
                        if job is not None:
                            pending.add(executor.submit(_repair_worker, job))

                    # Make progress visible even during long runs.
                    f.flush()

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


def repair_from_path(
    input_path: str,
    output_path: str,
    steps: list[str] | None = None,
    max_workers: int | None = None,
) -> None:
    """Standardizes a mesh file or a directory of meshes."""
    steps = DEFAULT_STEPS if steps is None else steps

    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"Unknown steps {unknown}. Available: {list(STEPS)}")

    if os.path.isdir(input_path):
        repair_directory(
            input_path, output_path, steps, max_workers=max_workers
        )

    elif os.path.isfile(input_path):
        if os.path.isdir(output_path) or not output_path.endswith(".obj"):
            output_path = os.path.join(
                output_path, os.path.basename(input_path)
            )

        row = repair_file(input_path, output_path, steps)
        print(f"Wrote {output_path}: {row}")

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
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=6,
        help="Number of parallel workers (default: CPU count)",
    )
    args = parser.parse_args()
    repair_from_path(
        args.input, args.output, args.steps, max_workers=args.workers
    )


if __name__ == "__main__":
    main()
