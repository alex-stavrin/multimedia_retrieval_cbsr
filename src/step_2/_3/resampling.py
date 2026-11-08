import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy
from tqdm import tqdm
from trimesh import grouping
from trimesh.geometry import faces_to_edges
import vedo

MIN_VERTEX_COUNT: Final = 5_000
TARGET_VERTEX_COUNT: Final = 7_500
MAX_VERTEX_COUNT: Final = 10_000
MAX_RESAMPLING_ATTEMPTS: Final = 15
MAX_TOP_1PCT_AREA_SHARE: Final = 0.10
DECIMATE_REGULARIZATION: Final = 1e-3

SUBDIVISION_METHOD_NAMES: Final = {
    0: "Loop",
    1: "Linear",
    2: "Adaptive",
    3: "Butterfly",
    4: "Centroid",
}


def format_subdivision_methods(methods: list[str]) -> str:
    if not methods:
        return "none"
    if len(methods) > 1 and len(set(methods)) == 1:
        return f"{methods[0]} ({len(methods)} iterations)"
    return ", ".join(methods)


@dataclass
class ResamplingRecord:
    relative_path: str
    before_vertices: int
    after_vertices: int
    subdivision_methods: list[str] = field(default_factory=list)
    error: str | None = None


def _compute_triangle_areas(
    vertices: numpy.ndarray, faces: numpy.ndarray
) -> numpy.ndarray:
    if len(faces) == 0:
        return numpy.empty(0, dtype=numpy.float64)
    v_0 = vertices[faces[:, 0]]
    v_1 = vertices[faces[:, 1]]
    v_2 = vertices[faces[:, 2]]
    return 0.5 * numpy.linalg.norm(numpy.cross(v_1 - v_0, v_2 - v_0), axis=1)


def _is_area_distribution_uneven(
    mesh: vedo.Mesh, threshold: float = MAX_TOP_1PCT_AREA_SHARE
) -> bool:
    faces = numpy.asarray(mesh.cells, dtype=numpy.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:  # type: ignore
        return True
    areas = _compute_triangle_areas(
        numpy.asarray(mesh.vertices, dtype=numpy.float64), faces
    )
    total_area = float(areas.sum())
    if total_area <= 1e-12:
        return False

    top_1pct_count = max(1, len(areas) // 100)
    top_1pct_area = float(numpy.sort(areas)[-top_1pct_count:].sum())
    return (top_1pct_area / total_area) > threshold


def _adaptive_subdivide(
    mesh: vedo.Mesh,
    target_vertices: int = TARGET_VERTEX_COUNT,
    subdivision_methods: list[str] | None = None,
) -> vedo.Mesh:
    """Iteratively bisect the longest edges until reaching `target_vertices`."""
    vertices = numpy.array(mesh.vertices, dtype=numpy.float64, copy=True)
    faces = numpy.asarray(mesh.cells, dtype=numpy.int64).copy()

    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:  # type: ignore
        return mesh

    goal_vertices = max(target_vertices, min(9_800, len(vertices) + 2_500))
    added_any = False

    for _ in range(25):
        needed = goal_vertices - len(vertices)
        if needed <= 0:
            break

        edges = numpy.sort(faces_to_edges(faces), axis=1)
        unique, inverse = grouping.unique_rows(edges)
        edges_unique = edges[unique]
        if len(edges_unique) == 0:
            break

        lengths = numpy.linalg.norm(
            vertices[edges_unique[:, 0]] - vertices[edges_unique[:, 1]], axis=1
        )
        max_len = float(lengths.max())
        if max_len <= 1e-12:
            break

        candidates = numpy.flatnonzero(lengths >= 0.5 * max_len)
        if len(candidates) > needed:
            top_idx = numpy.argpartition(lengths[candidates], -needed)[-needed:]
            selected = candidates[top_idx]
        else:
            selected = candidates

        long_edge = numpy.zeros(len(unique), dtype=bool)
        long_edge[selected] = True

        midpoint = numpy.full(len(unique), -1, dtype=numpy.int64)
        midpoint[long_edge] = numpy.arange(int(long_edge.sum())) + len(vertices)
        vertices = numpy.vstack(
            (vertices, vertices[edges_unique[long_edge]].mean(axis=1))
        )
        added_any = True

        face_mid = midpoint[inverse].reshape((-1, 3))
        split = face_mid >= 0
        count = split.sum(axis=1)

        keep = count == 0
        faces_keep = faces[keep]

        one = count == 1
        j = numpy.argmax(split[one], axis=1)
        r = numpy.arange(len(j))
        f1, mid1 = faces[one], face_mid[one]
        a, b, c = f1[r, j], f1[r, (j + 1) % 3], f1[r, (j + 2) % 3]
        p = mid1[r, j]
        faces_one = numpy.column_stack((a, p, c, p, b, c)).reshape((-1, 3))

        two = count == 2
        j = (numpy.argmin(split[two], axis=1) + 1) % 3
        r = numpy.arange(len(j))
        f2, mid2 = faces[two], face_mid[two]
        a, b, c = f2[r, j], f2[r, (j + 1) % 3], f2[r, (j + 2) % 3]
        p, q = mid2[r, j], mid2[r, (j + 1) % 3]
        use_aq = (
            ((vertices[a] - vertices[q]) ** 2).sum(axis=1)
            <= ((vertices[p] - vertices[c]) ** 2).sum(axis=1)
        )[:, None]
        corner = numpy.column_stack((p, b, q))
        quad_a = numpy.where(
            use_aq, numpy.column_stack((a, p, q)), numpy.column_stack((a, p, c))
        )
        quad_b = numpy.where(
            use_aq, numpy.column_stack((a, q, c)), numpy.column_stack((p, q, c))
        )
        faces_two = numpy.vstack((corner, quad_a, quad_b))

        three = count == 3
        f3, mid3 = faces[three], face_mid[three]
        v0, v1, v2 = f3.T
        m0, m1, m2 = mid3.T
        faces_three = numpy.column_stack(
            (v0, m0, m2, m0, v1, m1, m2, m1, v2, m0, m1, m2)
        ).reshape((-1, 3))

        faces = numpy.vstack(
            (faces_keep, faces_one, faces_two, faces_three)
        ).astype(numpy.int64)

    if added_any and subdivision_methods is not None:
        subdivision_methods.append("AdaptiveEdge")

    result = vedo.Mesh([vertices, faces])
    result.clean()
    return result


def resample_mesh(
    mesh: vedo.Mesh,
    max_iterations: int = MAX_RESAMPLING_ATTEMPTS,
    subdivision_methods: list[str] | None = None,
) -> vedo.Mesh:
    needs_vertices_count_fix = not (
        MIN_VERTEX_COUNT <= mesh.npoints <= MAX_VERTEX_COUNT
    )
    needs_area_fix = _is_area_distribution_uneven(mesh)

    if not needs_vertices_count_fix and not needs_area_fix:
        return mesh

    if mesh.npoints > MAX_VERTEX_COUNT:
        mesh.decimate(n=6_500, regularization=DECIMATE_REGULARIZATION)
        mesh.clean()
        needs_area_fix = _is_area_distribution_uneven(mesh)

    if mesh.npoints < MIN_VERTEX_COUNT or (
        needs_area_fix and mesh.npoints < 9_500
    ):
        mesh = _adaptive_subdivide(
            mesh,
            target_vertices=TARGET_VERTEX_COUNT,
            subdivision_methods=subdivision_methods,
        )

    for _ in range(max_iterations):
        before_vertices = mesh.npoints
        if before_vertices >= MIN_VERTEX_COUNT:
            break

        method_id = 1 if mesh.is_manifold() else 4
        method_name = SUBDIVISION_METHOD_NAMES.get(method_id, f"method {method_id}")
        if subdivision_methods is not None:
            subdivision_methods.append(f"{method_name} (method {method_id})")

        mesh.subdivide(n=1, method=method_id)
        mesh.clean()

        if mesh.npoints == before_vertices:
            break

    for _ in range(5):
        if mesh.npoints <= MAX_VERTEX_COUNT:
            break
        before_vertices = mesh.npoints
        mesh.decimate(
            n=TARGET_VERTEX_COUNT, regularization=DECIMATE_REGULARIZATION
        )
        mesh.clean()
        if mesh.npoints == before_vertices:
            break

    return mesh


def resample_mesh_file(
    relative_path: str | Path,
    input_path: Path | str,
    output_path: Path | str,
) -> tuple[str, ResamplingRecord]:
    relative_path = Path(relative_path)
    mesh_path = Path(input_path) / relative_path
    output_p = Path(output_path)
    if output_p.suffix.lower() == ".obj":
        destination = output_p
    else:
        destination = output_p / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        mesh = vedo.Mesh(str(mesh_path))
        before_vertices = mesh.npoints

        subdivision_methods: list[str] = []

        if (
            MIN_VERTEX_COUNT <= before_vertices <= MAX_VERTEX_COUNT
            and not _is_area_distribution_uneven(mesh)
        ):
            mesh.write(str(destination))
            return "fine", ResamplingRecord(
                relative_path=str(relative_path.as_posix()),
                before_vertices=before_vertices,
                after_vertices=before_vertices,
                subdivision_methods=[],
            )

        category = "subdivided" if before_vertices < MIN_VERTEX_COUNT else "decimated"

        mesh = resample_mesh(mesh, subdivision_methods=subdivision_methods)

        resampling_record = ResamplingRecord(
            relative_path=str(relative_path.as_posix()),
            before_vertices=before_vertices,
            after_vertices=mesh.npoints,
            subdivision_methods=subdivision_methods,
        )
        mesh.write(str(destination))

        return category, resampling_record
    except Exception as exc:
        return "error", ResamplingRecord(
            relative_path=str(relative_path.as_posix()),
            before_vertices=-1,
            after_vertices=-1,
            subdivision_methods=[],
            error=str(exc),
        )


def resample_meshes(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    print_outlier_stats: bool = False,
    print_decimating_logs: bool = False,
    print_subdivision_logs: bool = False,
    print_failed_logs: bool = True,
    *,
    models_path: Path | str | None = None,
    max_workers: int | None = None,
) -> None:
    if input_path is None:
        if models_path is not None:
            input_path = models_path
        else:
            raise ValueError("input_path must be specified.")

    if output_path is None:
        raise ValueError("output_path must be specified.")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    subdivided_models: list[ResamplingRecord] = []
    decimated_models: list[ResamplingRecord] = []
    fine_models: list[ResamplingRecord] = []
    failed_models_decimation: list[ResamplingRecord] = []
    failed_models_subdivision: list[ResamplingRecord] = []
    error_models: list[ResamplingRecord] = []

    groups = {
        "subdivided": subdivided_models,
        "decimated": decimated_models,
        "fine": fine_models,
        "error": error_models,
    }

    if input_path.is_file():
        if input_path.suffix.lower() != ".obj":
            raise ValueError(f"Input file is not an .obj file: {input_path}")
        mesh_files = [Path(input_path.name)]
        input_root = input_path.parent
        output_root = output_path
    else:
        input_root = input_path
        output_root = output_path
        mesh_files = [
            p.relative_to(input_path)
            for p in sorted(input_path.rglob("*"))
            if p.is_file() and p.suffix.lower() == ".obj"
        ]

    if not mesh_files:
        print(f"No .obj files found in input path: {input_path}")
        return

    n_workers = max_workers or os.cpu_count() or 1
    print(f"Using {n_workers} worker processes")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        resample_futures = [
            executor.submit(
                resample_mesh_file,
                relative_path,
                input_root,
                output_root,
            )
            for relative_path in mesh_files
        ]

        for future in tqdm(
            as_completed(resample_futures),
            total=len(resample_futures),
            desc="Resampling meshes",
        ):
            category, record = future.result()
            if category in groups:
                groups[category].append(record)

            if category == "error":
                tqdm.write(f"Error on {record.relative_path}: {record.error}")
                continue

            outside_range = not (
                MIN_VERTEX_COUNT <= record.after_vertices <= MAX_VERTEX_COUNT
            )

            methods_str = format_subdivision_methods(record.subdivision_methods)
            should_print = (
                (category == "subdivided" and print_subdivision_logs)
                or (category == "decimated" and print_decimating_logs)
                or (category == "fine" and print_outlier_stats)
            )

            if outside_range:
                if category == "subdivided":
                    failed_models_subdivision.append(record)
                elif category == "decimated":
                    failed_models_decimation.append(record)

                if print_failed_logs:
                    if category == "subdivided":
                        tqdm.write(
                            f"FAILED subdivision on {record.relative_path}: "
                            f"{record.before_vertices} -> {record.after_vertices} "
                            f"(outside range [{MIN_VERTEX_COUNT}, {MAX_VERTEX_COUNT}], method: {methods_str})"
                        )
                    elif category == "decimated":
                        methods_info = (
                            f", subdivision method: {methods_str}"
                            if record.subdivision_methods
                            else ""
                        )
                        tqdm.write(
                            f"FAILED decimation on {record.relative_path}: "
                            f"{record.before_vertices} -> {record.after_vertices} "
                            f"(outside range [{MIN_VERTEX_COUNT}, {MAX_VERTEX_COUNT}]{methods_info})"
                        )
                elif should_print:
                    if category == "subdivided":
                        tqdm.write(
                            f"{record.relative_path} (subdivision method: {methods_str}): "
                            f"{record.before_vertices} -> {record.after_vertices}"
                        )
                    else:
                        tqdm.write(
                            f"{record.relative_path}: "
                            f"{record.before_vertices} -> {record.after_vertices}"
                        )
            else:
                if should_print:
                    if category == "subdivided":
                        tqdm.write(
                            f"{record.relative_path} (subdivision method: {methods_str}): "
                            f"{record.before_vertices} -> {record.after_vertices}"
                        )
                    else:
                        tqdm.write(
                            f"{record.relative_path}: "
                            f"{record.before_vertices} -> "
                            f"{record.after_vertices}"
                        )

    log_folder = Path("./logs")
    log_folder.mkdir(parents=True, exist_ok=True)

    log_path = log_folder / "resampled_outliers.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"SUBDIVIDED MODELS ({len(subdivided_models)} total)\n")
        log_file.write("=================\n")

        for record in subdivided_models:
            methods_str = format_subdivision_methods(record.subdivision_methods)
            methods_info = (
                f" using method: {methods_str}" if record.subdivision_methods else ""
            )
            log_file.write(
                f"{record.relative_path} had {record.before_vertices} "
                f"and now has {record.after_vertices} vertices{methods_info}\n"
            )

        log_file.write(f"\nDECIMATED MODELS ({len(decimated_models)} total)\n")
        log_file.write("================\n")

        for record in decimated_models:
            methods_str = format_subdivision_methods(record.subdivision_methods)
            methods_info = (
                f" using method: {methods_str}" if record.subdivision_methods else ""
            )
            log_file.write(
                f"{record.relative_path} had {record.before_vertices} "
                f"and now has {record.after_vertices} vertices{methods_info}\n"
            )

        log_file.write(f"\nFINE MODELS ({len(fine_models)} total)\n")
        log_file.write("================\n")

        for record in fine_models:
            log_file.write(f"{record}\n")

        log_file.write(
            f"\nFAILED MODELS DECIMATION ({len(failed_models_decimation)} total)\n"
        )
        log_file.write("================\n")

        for record in failed_models_decimation:
            methods_str = format_subdivision_methods(record.subdivision_methods)
            methods_info = (
                f" using method: {methods_str}" if record.subdivision_methods else ""
            )
            log_file.write(
                f"{record.relative_path} had {record.before_vertices} "
                f"and now has {record.after_vertices} vertices "
                f"(outside range [{MIN_VERTEX_COUNT}, {MAX_VERTEX_COUNT}]){methods_info}\n"
            )

        log_file.write(
            f"\nFAILED MODELS SUBDIVISION ({len(failed_models_subdivision)} total)\n"
        )
        log_file.write("================\n")

        for record in failed_models_subdivision:
            methods_str = format_subdivision_methods(record.subdivision_methods)
            methods_info = (
                f" using method: {methods_str}" if record.subdivision_methods else ""
            )
            log_file.write(
                f"{record.relative_path} had {record.before_vertices} "
                f"and now has {record.after_vertices} vertices "
                f"(outside range [{MIN_VERTEX_COUNT}, {MAX_VERTEX_COUNT}]){methods_info}\n"
            )

        if error_models:
            log_file.write(f"\nERROR MODELS ({len(error_models)} total)\n")
            log_file.write("================\n")

            for record in error_models:
                log_file.write(f"{record.relative_path}: {record.error}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Resample meshes so that vertex counts are within [5000, 10000]."
    )

    parser.add_argument(
        "-i",
        "--input",
        "--input_path",
        "--models_path",
        type=Path,
        required=True,
        dest="input_path",
        help="Input folder or file containing 3D meshes.",
    )

    parser.add_argument(
        "-o",
        "--output",
        "--output_path",
        type=Path,
        required=True,
        dest="output_path",
        help="Output folder where resampled meshes will be saved.",
    )

    parser.add_argument(
        "-pos",
        "--print_outliers_stats",
        action="store_true",
        help="Prints outliers stats",
    )

    parser.add_argument(
        "-pdl",
        "--print_decimate_logs",
        action="store_true",
        help="Print decimate logs",
    )

    parser.add_argument(
        "-psl",
        "--print_subdivision_logs",
        action="store_true",
        help="Print subdivision logs",
    )

    parser.add_argument(
        "-pfl",
        "--print_failed_logs",
        action="store_true",
        default=True,
        help="Print failed subdivision and decimation logs (default: True)",
    )

    parser.add_argument(
        "--no_print_failed_logs",
        action="store_false",
        dest="print_failed_logs",
        help="Do not print failed subdivision and decimation logs",
    )

    args = parser.parse_args()

    print("Starting resampling. This will take some time...")

    resample_meshes(
        input_path=args.input_path,
        output_path=args.output_path,
        print_outlier_stats=args.print_outliers_stats,
        print_decimating_logs=args.print_decimate_logs,
        print_subdivision_logs=args.print_subdivision_logs,
        print_failed_logs=args.print_failed_logs,
    )


if __name__ == "__main__":
    main()
