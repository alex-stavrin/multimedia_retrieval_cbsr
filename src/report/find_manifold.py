"""Find non-manifold 3D objects in a database directory.

This script scans a database folder containing 3D meshes (.obj) and finds N
non-manifold objects, providing detailed diagnostic reports on non-manifold edges,
non-manifold (pinched) vertices, boundary edges, and failure reasons.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Ensure repo root is on sys.path so imports work regardless of execution directory
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import vedo
from tqdm import tqdm

from src.helper.database_functions import retrieve_all_shape_files


@dataclass
class NonManifoldReport:
    """Diagnostic details for a non-manifold mesh."""

    file_path: str
    relative_path: str
    shape_class: str
    n_vertices: int
    n_faces: int
    is_manifold: bool
    non_manifold_edges: int
    non_manifold_vertices: int
    boundary_edges: int
    non_manifold_vertex_ids: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def summary_reason(self) -> str:
        if self.reasons:
            return "; ".join(self.reasons)
        if not self.is_manifold:
            return "Non-manifold geometry detected"
        return "Manifold (valid 2-manifold surface)"


def detect_non_manifold_edges(mesh: vedo.Mesh) -> tuple[int, np.ndarray, int]:
    """Counts non-manifold edges (>2 incident faces) and boundary edges (1 incident face).

    Returns:
        (num_non_manifold_edges, non_manifold_edges_array, num_boundary_edges)
    """
    faces = np.asarray(mesh.cells, dtype=np.int64)
    if len(faces) == 0:
        return 0, np.empty((0, 2), dtype=np.int64), 0

    # Ensure mesh is triangulated for consistent edge pairing
    if faces.ndim != 2 or faces.shape[1] != 3:
        tri_mesh = mesh.clone().triangulate()
        faces = np.asarray(tri_mesh.cells, dtype=np.int64)
        if len(faces) == 0 or faces.ndim != 2 or faces.shape[1] != 3:
            return -1, np.empty((0, 2), dtype=np.int64), -1

    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

    nm_mask = counts > 2
    boundary_mask = counts == 1

    nm_edges = unique_edges[nm_mask]
    num_nm_edges = int(nm_mask.sum())
    num_boundary_edges = int(boundary_mask.sum())

    return num_nm_edges, nm_edges, num_boundary_edges


def detect_non_manifold_vertices(faces: np.ndarray) -> tuple[int, list[int]]:
    """Identifies vertices whose local neighborhood (link) is not a 2D disk or half-disk.

    In a valid 2-manifold with boundary:
      - The link graph of an interior vertex is a single simple cycle (all vertices degree 2, 1 component).
      - The link graph of a boundary vertex is a single simple path (two vertices degree 1, others degree 2, 1 component).

    Any vertex violating these conditions is topologically non-manifold (e.g. pinched / hourglass points).
    """
    if len(faces) == 0 or faces.ndim != 2 or faces.shape[1] != 3:
        return 0, []

    # Map each vertex to opposite edges in adjacent faces
    link = defaultdict(list)
    for f in faces:
        u, v, w = int(f[0]), int(f[1]), int(f[2])
        link[u].append((min(v, w), max(v, w)))
        link[v].append((min(u, w), max(u, w)))
        link[w].append((min(u, v), max(u, v)))

    nm_vertices: list[int] = []
    for v, edges in link.items():
        adj = defaultdict(set)
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)

        deg_counts = defaultdict(int)
        has_high_degree = False
        for nbrs in adj.values():
            deg = len(nbrs)
            deg_counts[deg] += 1
            if deg > 2:
                has_high_degree = True
                break

        if has_high_degree:
            nm_vertices.append(v)
            continue

        deg1 = deg_counts[1]
        # Valid manifold vertex link: either cycle (deg1 == 0) or path (deg1 == 2)
        if deg1 not in (0, 2):
            nm_vertices.append(v)
            continue

        # Count connected components in link graph
        visited = set()
        components = 0
        for node in adj:
            if node not in visited:
                components += 1
                q = [node]
                visited.add(node)
                while q:
                    curr = q.pop()
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            q.append(nbr)

        if components != 1:
            # Disconnected fans meeting at vertex v (pinched vertex)
            nm_vertices.append(v)

    return len(nm_vertices), nm_vertices


def analyze_mesh(
    shape_path: str | Path,
    base_dir: str | Path = "",
) -> NonManifoldReport | None:
    """Analyzes a 3D mesh file and returns its NonManifoldReport.

    Returns None if the file cannot be loaded or is invalid.
    """
    shape_path_obj = Path(shape_path).resolve()
    base_dir_obj = Path(base_dir).resolve() if base_dir else shape_path_obj.parent

    try:
        relative_path = str(shape_path_obj.relative_to(base_dir_obj))
    except ValueError:
        relative_path = shape_path_obj.name

    shape_class = (
        shape_path_obj.parent.name
        if shape_path_obj.parent != base_dir_obj
        else "default"
    )

    try:
        mesh = vedo.Mesh(str(shape_path_obj))
    except Exception as exc:
        return None

    if mesh is None or mesh.npoints == 0 or mesh.ncells == 0:
        return None

    n_vertices = mesh.npoints
    n_faces = mesh.ncells

    # Check non-manifold edges
    num_nm_edges, _, num_boundary_edges = detect_non_manifold_edges(mesh)

    # Check non-manifold vertices
    faces = np.asarray(mesh.cells, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        tri_mesh = mesh.clone().triangulate()
        faces = np.asarray(tri_mesh.cells, dtype=np.int64)

    num_nm_verts, nm_vert_ids = detect_non_manifold_vertices(faces)

    # vedo's internal VTK manifold test
    vedo_is_manifold = mesh.is_manifold()

    reasons: list[str] = []
    if num_nm_edges > 0:
        reasons.append(f"{num_nm_edges} non-manifold edges (>2 incident faces)")
    if num_nm_verts > 0:
        reasons.append(f"{num_nm_verts} non-manifold vertices (pinched points)")
    if not vedo_is_manifold and not reasons:
        reasons.append("vedo/VTK reported non-manifold topology")

    is_manifold = vedo_is_manifold and (num_nm_edges == 0) and (num_nm_verts == 0)

    return NonManifoldReport(
        file_path=str(shape_path_obj),
        relative_path=relative_path,
        shape_class=shape_class,
        n_vertices=n_vertices,
        n_faces=n_faces,
        is_manifold=is_manifold,
        non_manifold_edges=num_nm_edges,
        non_manifold_vertices=num_nm_verts,
        boundary_edges=num_boundary_edges,
        non_manifold_vertex_ids=nm_vert_ids,
        reasons=reasons,
    )


def order_files_by_diversity(files: Sequence[str | Path]) -> list[str]:
    """Interleaves files from different categories so sampling is diverse."""
    class_groups = defaultdict(list)
    for f in files:
        p = Path(f)
        shape_class = p.parent.name
        class_groups[shape_class].append(str(p))

    diverse_order: list[str] = []
    max_len = max(len(grp) for grp in class_groups.values()) if class_groups else 0
    class_names = sorted(class_groups.keys())

    for idx in range(max_len):
        for cname in class_names:
            if idx < len(class_groups[cname]):
                diverse_order.append(class_groups[cname][idx])

    return diverse_order


def find_non_manifold_objects(
    database_path: str | Path,
    n: int = 5,
    diverse: bool = False,
    verbose: bool = True,
) -> list[NonManifoldReport]:
    """Scans the given database directory and returns up to N non-manifold objects."""
    db_path = Path(database_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Database directory does not exist: {db_path}")

    if db_path.is_file():
        shape_files = [str(db_path)]
    else:
        shape_files = list(retrieve_all_shape_files(str(db_path)))

    if not shape_files:
        raise ValueError(f"No .obj files found in database path: {db_path}")

    shape_files = sorted(shape_files)

    if diverse:
        shape_files = order_files_by_diversity(shape_files)

    non_manifold_results: list[NonManifoldReport] = []

    if verbose:
        print(f"\n[INFO] Database path: {db_path}")
        print(f"[INFO] Total candidate files: {len(shape_files)}")
        print(f"[INFO] Target non-manifold objects: {n}")
        if diverse:
            print("[INFO] Diverse category sampling: ON")
        print("[INFO] Scanning for non-manifold meshes...\n")

    pbar = tqdm(shape_files, desc="Scanning meshes", disable=not verbose)

    for file_path in pbar:
        report = analyze_mesh(file_path, base_dir=db_path)
        if report is not None and not report.is_manifold:
            non_manifold_results.append(report)
            pbar.set_postfix(found=len(non_manifold_results))
            if verbose:
                pbar.write(
                    f"  -> [{len(non_manifold_results)}/{n}] Found non-manifold: "
                    f"{report.relative_path} ({report.summary_reason})"
                )

            if len(non_manifold_results) >= n:
                break

    pbar.close()
    return non_manifold_results


def print_summary_table(results: list[NonManifoldReport]) -> None:
    """Prints a structured ASCII summary table of the non-manifold objects found."""
    if not results:
        print("\nNo non-manifold objects were found.")
        return

    print("\n" + "=" * 110)
    print(f" FOUND {len(results)} NON-MANIFOLD OBJECTS")
    print("=" * 110)

    header = (
        f"{'#':<3} | {'Class':<18} | {'Relative Path':<30} | "
        f"{'Vertices':<8} | {'Faces':<8} | {'NM Edges':<8} | {'NM Verts':<8} | {'Open Edges':<10}"
    )
    print(header)
    print("-" * 110)

    for i, r in enumerate(results, start=1):
        rel_path = (
            r.relative_path
            if len(r.relative_path) <= 30
            else "..." + r.relative_path[-27:]
        )
        shape_class = (
            r.shape_class if len(r.shape_class) <= 18 else r.shape_class[:15] + "..."
        )
        row = (
            f"{i:<3} | {shape_class:<18} | {rel_path:<30} | "
            f"{r.n_vertices:<8} | {r.n_faces:<8} | {r.non_manifold_edges:<8} | "
            f"{r.non_manifold_vertices:<8} | {r.boundary_edges:<10}"
        )
        print(row)
        print(f"      Details: {r.summary_reason}")

    print("=" * 110 + "\n")


def export_to_csv(results: list[NonManifoldReport], output_path: str | Path) -> None:
    """Exports non-manifold analysis results to a CSV file."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "file_path",
        "relative_path",
        "shape_class",
        "n_vertices",
        "n_faces",
        "is_manifold",
        "non_manifold_edges",
        "non_manifold_vertices",
        "boundary_edges",
        "reason",
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(results, start=1):
            writer.writerow(
                {
                    "index": idx,
                    "file_path": r.file_path,
                    "relative_path": r.relative_path,
                    "shape_class": r.shape_class,
                    "n_vertices": r.n_vertices,
                    "n_faces": r.n_faces,
                    "is_manifold": r.is_manifold,
                    "non_manifold_edges": r.non_manifold_edges,
                    "non_manifold_vertices": r.non_manifold_vertices,
                    "boundary_edges": r.boundary_edges,
                    "reason": r.summary_reason,
                }
            )

    print(f"[INFO] Exported {len(results)} results to CSV: {out_file}")


def copy_objects(results: list[NonManifoldReport], dest_dir: str | Path) -> None:
    """Copies found non-manifold mesh files (.obj and .mtl) into a destination directory."""
    destination = Path(dest_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    for r in results:
        src_path = Path(r.file_path)
        dest_filename = (
            f"{r.shape_class}_{src_path.name}"
            if r.shape_class != "default"
            else src_path.name
        )
        dest_path = destination / dest_filename
        shutil.copy2(src_path, dest_path)

        # Check for accompanying .mtl file
        mtl_file = src_path.with_suffix(".mtl")
        if mtl_file.exists():
            shutil.copy2(mtl_file, destination / mtl_file.name)

    print(f"[INFO] Copied {len(results)} models to directory: {destination}")


def visualize_objects(results: list[NonManifoldReport]) -> None:
    """Interactively visualizes found non-manifold models with bad edges and vertices highlighted."""
    for idx, r in enumerate(results, start=1):
        print(f"\n[VISUALIZE {idx}/{len(results)}] Opening {r.relative_path}...")
        print(" -> Close window or press 'q' in the 3D viewer to view next model.")

        mesh = vedo.Mesh(r.file_path).c("silver").alpha(0.7).wireframe(False)
        actors: list[vedo.Points | vedo.Mesh] = [mesh]

        # Highlight non-manifold boundary lines
        nm_lines = mesh.boundaries(boundary_edges=False, non_manifold_edges=True)
        if nm_lines.dataset.GetNumberOfCells() > 0:
            nm_lines.c("red").lw(6)
            actors.append(nm_lines)

        # Highlight non-manifold vertices if available
        if r.non_manifold_vertex_ids and mesh.npoints > 0:
            valid_ids = [vid for vid in r.non_manifold_vertex_ids if vid < mesh.npoints]
            if valid_ids:
                pts = mesh.points[valid_ids]
                nm_pts = vedo.Points(pts, r=10, c="crimson")
                actors.append(nm_pts)

        title = f"[{idx}/{len(results)}] {r.relative_path} | {r.summary_reason}"
        vedo.show(actors, title=title, axes=1, bg="white")


def parse_arguments() -> tuple[str, int, argparse.Namespace]:
    """Parses command-line arguments, supporting both positional and flagged formats."""
    parser = argparse.ArgumentParser(
        description="Search a database folder and find N non-manifold 3D objects (.obj)."
    )

    # Positional arguments (optional so flags can also be used)
    parser.add_argument(
        "database",
        nargs="?",
        default=None,
        help="Path to database folder containing 3D mesh files (.obj).",
    )
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=None,
        help="Number of non-manifold objects to find (e.g. 5).",
    )

    # Flag options
    parser.add_argument(
        "-d",
        "--database",
        "--input",
        "-i",
        dest="database_flag",
        type=str,
        default=None,
        help="Path to database folder containing 3D mesh files (.obj).",
    )
    parser.add_argument(
        "-n",
        "--count",
        "--number",
        dest="n_flag",
        type=int,
        default=None,
        help="Number of non-manifold objects to find.",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Sample non-manifold meshes from diverse categories instead of sequentially.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional CSV file path to save results.",
    )
    parser.add_argument(
        "--copy-to",
        type=str,
        default=None,
        help="Optional directory to copy the found non-manifold models into.",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Interactively display each found non-manifold mesh with highlighted non-manifold edges/vertices.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-essential console logs and progress bar.",
    )

    args = parser.parse_args()

    # Resolve database path
    db_arg = args.database_flag or args.database
    n_arg = args.n_flag if args.n_flag is not None else args.n

    # Support graceful reversal if user ran: find_manifold.py <N> <database_path>
    if db_arg is not None and not os.path.exists(db_arg) and str(db_arg).isdigit():
        if n_arg is not None and isinstance(n_arg, str) and os.path.exists(n_arg):
            db_arg, n_arg = n_arg, int(db_arg)

    if not db_arg:
        parser.error(
            "A database directory path must be provided. "
            "Example: python -m src.report.find_manifold 3d_models 5"
        )

    target_n = int(n_arg) if n_arg is not None else 5
    if target_n <= 0:
        parser.error(f"N must be a positive integer, got: {target_n}")

    return db_arg, target_n, args


def main() -> None:
    db_path, count, args = parse_arguments()

    results = find_non_manifold_objects(
        database_path=db_path,
        n=count,
        diverse=args.diverse,
        verbose=not args.quiet,
    )

    print_summary_table(results)

    if args.output:
        export_to_csv(results, args.output)

    if args.copy_to:
        copy_objects(results, args.copy_to)

    if args.visualize and results:
        visualize_objects(results)


if __name__ == "__main__":
    main()
