import pandas
import vedo
from typing import Final
from src.step_2._2.database_analyzer import find_outliers
from pathlib import Path
from shutil import copy2


def resample_outliers(
    print_outlier_stats=False,
    print_decimating_logs=False,
    print_subdivision_logs=False,
    models_path=Path,
    input_csv_path=Path,
    output_path=Path,
) -> None:
    data_frame = pandas.read_csv(input_csv_path)

    subdivided_models: list[str] = []
    decimated_models: list[str] = []

    # Vertex outliers
    vertex_outliers_data_frame = find_outliers(data_frame, "n_points")
    if print_outlier_stats:
        print("Vertex outliers count: " + str(vertex_outliers_data_frame.shape[0]))

    # Face outliers
    face_outliers_data_frame = find_outliers(data_frame, "n_faces")
    if print_outlier_stats:
        print("Face outliers count: " + str(face_outliers_data_frame.shape[0]))

    # Intersection of face and vertex outliers. We care about intersection only to check the stats
    if print_outlier_stats:
        intersection_outliers_data_frame = vertex_outliers_data_frame[
            vertex_outliers_data_frame.index.isin(face_outliers_data_frame.index)
        ]
        print(
            "Intersection outliers count: "
            + str(intersection_outliers_data_frame.shape[0])
        )

    # Combined face and vertex outliers (no duplicates)
    all_outliers_indices = vertex_outliers_data_frame.index.union(
        face_outliers_data_frame.index
    )
    all_outliers_data_frame = data_frame.loc[all_outliers_indices]
    if print_outlier_stats:
        print("Combined outliers count: " + str(all_outliers_data_frame.shape[0]))

    MIN_VERTEX_COUNT: Final = 5_000
    TARGET_VERTEX_COUNT: Final = 7_500
    MAX_VERTEX_COUNT: Final = 10_000
    for row in all_outliers_data_frame.itertuples():
        vertex_count = int(row.n_points)
        mesh_path = models_path / row.relative_file_path
        final_output_path = Path(output_path) / str(row.relative_file_path)
        # If this is true that means we want to increase the amount of vertices to the 5k-10k range. So we subdivide.
        if vertex_count < MIN_VERTEX_COUNT:
            if print_subdivision_logs:
                print(
                    f"Mesh {row.relative_file_path} has {row.n_points} vertices. We are subdividing it to increase the vertices count."
                )

            mesh = vedo.Mesh(str(mesh_path))

            while mesh.npoints < MIN_VERTEX_COUNT:
                mesh.subdivide(n=1, method=4)

            subdivided_models.append(str(row.relative_file_path))

            if mesh.npoints > MAX_VERTEX_COUNT:
                mesh.decimate(n=TARGET_VERTEX_COUNT)

            if print_subdivision_logs:
                print(f"Mesh {row.relative_file_path} has now {mesh.npoints}\n")

            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.write(str(final_output_path))
        # If this is true that means we want to decrease the amount of vertices to the 5k-10k range. So we decimate.
        elif vertex_count > MAX_VERTEX_COUNT:
            if print_decimating_logs:
                print(
                    f"Mesh {row.relative_file_path} has {row.n_points} vertices. We are decimating it to decrease the vertices count."
                )
            mesh_path = models_path / row.relative_file_path
            mesh = vedo.Mesh(str(mesh_path))
            mesh.decimate(n=TARGET_VERTEX_COUNT)
            decimated_models.append(str(row.relative_file_path))
            if print_decimating_logs:
                print(
                    f"Mesh {row.relative_file_path} has now {mesh.npoints} vertices\n"
                )

            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.write(str(final_output_path))
        else:
            # just copy the file
            if print_outlier_stats:
                print("Found outlier that is not in the vertex range.\n")
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(mesh_path, final_output_path)

    # We also need to copy from models_path to output_path, the models that are not outliers, so that are our dataset is complete
    not_outliers_models_data_frame = data_frame.drop(index=all_outliers_indices)
    if print_outlier_stats:
        print("Remaining meshes count: " + str(not_outliers_models_data_frame.shape[0]))
    for row in not_outliers_models_data_frame.itertuples():
        source_path = Path(models_path) / row.relative_file_path
        destination_path = Path(output_path) / row.relative_file_path

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(source_path, destination_path)

    # Logging
    log_folder = Path("./") / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_path = log_folder / "resampled_outliers.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("SUBDIVIDED MODELS\n")
        log_file.write("=================\n")

        for relative_path in subdivided_models:
            log_file.write(f"{relative_path}\n")

        log_file.write("\nDECIMATED MODELS\n")
        log_file.write("================\n")

        for relative_path in decimated_models:
            log_file.write(f"{relative_path}\n")


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

    resample_outliers(
        arguements.print_outliers_stats,
        arguements.print_decimate_logs,
        arguements.print_subdivision_logs,
        arguements.models_path,
        arguements.input_csv_path,
        arguements.output_path,
    )


main()
