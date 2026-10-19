import pandas
import vedo
from typing import Final
from pathlib import Path
from shutil import copy2
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

MIN_VERTEX_COUNT: Final = 5_000
TARGET_VERTEX_COUNT: Final = 7_500
MAX_VERTEX_COUNT: Final = 10_000
MAX_RESAMPLING_ATTEMPTS: Final = 15


@dataclass
class ResamplingRecord:
    relative_path: str
    before_vertices: int
    after_vertices: int


def resample_one_mesh(
    relative_path: str, vertex_count: int, models_path: Path, output_path: Path
) -> tuple[str, ResamplingRecord]:
    mesh_path = models_path / relative_path
    destination = output_path / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    resampling_record = ResamplingRecord(
        relative_path=relative_path,
        before_vertices=vertex_count,
        after_vertices=vertex_count,
    )

    if MIN_VERTEX_COUNT <= vertex_count <= MAX_VERTEX_COUNT:
        copy2(mesh_path, destination)
        return "fine", resampling_record

    category = "subdivided" if vertex_count < MIN_VERTEX_COUNT else "decimated"

    mesh = vedo.Mesh(str(mesh_path))
    resampling_record.before_vertices = mesh.npoints
    mesh.clean()

    for _ in range(MAX_RESAMPLING_ATTEMPTS):
        before_vertices = mesh.npoints
        if MIN_VERTEX_COUNT <= before_vertices <= MAX_VERTEX_COUNT:
            break

        if before_vertices < MIN_VERTEX_COUNT:
            mesh.subdivide(n=1, method=4)
        else:
            mesh.decimate(n=TARGET_VERTEX_COUNT)

        if mesh.npoints == before_vertices:
            break

    resampling_record.after_vertices = mesh.npoints
    mesh.write(str(destination))

    return category, resampling_record


def resample_meshes(
    models_path: Path,
    input_csv_path: Path,
    output_path: Path,
    print_outlier_stats=False,
    print_decimating_logs=False,
    print_subdivision_logs=False,
) -> None:
    data_frame = pandas.read_csv(input_csv_path)

    subdivided_models: list[ResamplingRecord] = []
    decimated_models: list[ResamplingRecord] = []
    fine_models: list[ResamplingRecord] = []
    failed_models_decimation: list[ResamplingRecord] = []
    failed_models_subdivision: list[ResamplingRecord] = []

    groups = {
        "subdivided": subdivided_models,
        "decimated": decimated_models,
        "fine": fine_models,
    }

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                resample_one_mesh,
                str(row.relative_file_path),
                int(row.n_points),
                Path(models_path),
                Path(output_path),
            )
            for row in data_frame.itertuples()
        ]

        for future in as_completed(futures):
            category, record = future.result()
            groups[category].append(record)

            outside_range = not (
                MIN_VERTEX_COUNT <= record.after_vertices <= MAX_VERTEX_COUNT
            )

            if outside_range:
                if category == "subdivided":
                    failed_models_subdivision.append(record)
                elif category == "decimated":
                    failed_models_decimation.append(record)

            should_print = (
                (category == "subdivided" and print_subdivision_logs)
                or (category == "decimated" and print_decimating_logs)
                or (category == "fine" and print_outlier_stats)
            )

            if should_print:
                print(
                    f"{record.relative_path}: {record.before_vertices} -> {record.after_vertices}"
                )

    # Logging
    log_folder = Path("./") / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_path = log_folder / "resampled_outliers.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"SUBDIVIDED MODELS ({len(subdivided_models)} total)\n")
        log_file.write("=================\n")

        for resampling_record in subdivided_models:
            log_file.write(
                f"{resampling_record.relative_path} had {resampling_record.before_vertices} and now has {resampling_record.after_vertices} vertices\n"
            )

        log_file.write(f"\nDECIMATED MODELS ({len(decimated_models)} total)\n")
        log_file.write("================\n")

        for resampling_record in decimated_models:
            log_file.write(
                f"{resampling_record.relative_path} had {resampling_record.before_vertices} and now has {resampling_record.after_vertices} vertices\n"
            )

        log_file.write(f"\nFINE MODELS\n ({len(fine_models)} total)")
        log_file.write("================\n")

        for resampling_record in fine_models:
            log_file.write(f"{resampling_record}\n")

        log_file.write(
            f"\nFAILED MODELS DECIMATION ({len(failed_models_decimation)} total)\n"
        )
        log_file.write("================\n")

        for resampling_record in failed_models_decimation:
            log_file.write(
                f"{resampling_record.relative_path} had {resampling_record.before_vertices} and now has {resampling_record.after_vertices} vertices\n"
            )

        log_file.write(
            f"\nFAILED MODELS SUBDIVSION ({len(failed_models_subdivision)} total)\n"
        )
        log_file.write("================\n")

        for resampling_record in failed_models_subdivision:
            log_file.write(
                f"{resampling_record.relative_path} had {resampling_record.before_vertices} and now has {resampling_record.after_vertices} vertices\n"
            )


def main() -> None:
    import argparse

    arguements_parser = argparse.ArgumentParser()

    arguements_parser.add_argument(
        "-icsv", "--input_csv_path", type=Path, required=True
    )

    arguements_parser.add_argument("-m", "--models_path", type=Path, required=True)

    arguements_parser.add_argument("-o", "--output_path", type=Path, required=True)

    arguements_parser.add_argument(
        "-pos",
        "--print_outliers_stats",
        action="store_true",
        help="Prints outliers stats",
    )

    arguements_parser.add_argument(
        "-pdl",
        "--print_decimate_logs",
        action="store_true",
        help="Print decimate logs",
    )

    arguements_parser.add_argument(
        "-psl",
        "--print_subdivision_logs",
        action="store_true",
        help="Print subdivision logs",
    )

    arguements = arguements_parser.parse_args()

    print("Starting resampling. This will take some time...")

    resample_meshes(
        arguements.print_outliers_stats,
        arguements.print_decimate_logs,
        arguements.print_subdivision_logs,
        arguements.models_path,
        arguements.input_csv_path,
        arguements.output_path,
    )


if __name__ == "__main__":
    main()
