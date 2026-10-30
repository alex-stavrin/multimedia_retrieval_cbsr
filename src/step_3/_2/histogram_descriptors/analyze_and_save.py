import argparse
from pathlib import Path
import vedo
from src.step_3._2.histogram_descriptors.histogram_descriptors import (
    calculate_angle_histogram,
    calculate_distance_from_barycenter,
    calculate_distance_two_vertices,
    calculate_area_triangle,
    calculate_volume_tetrahedron,
)
from tqdm import tqdm
import csv
from concurrent.futures import ProcessPoolExecutor
import os


def proccess_file(args: tuple[Path, Path, int, int]) -> list:

    file, input_root, n_samples, n_bins = args
    relative_path = file.relative_to(input_root).as_posix()
    mesh = vedo.Mesh(str(file))

    A3, _ = calculate_angle_histogram(
        mesh=mesh,
        n_samples=n_samples,
        n_bins_histogram=n_bins,
    )
    D1, _ = calculate_distance_from_barycenter(
        mesh=mesh,
        n_bins_histogram=n_bins,
    )
    D2, _ = calculate_distance_two_vertices(
        mesh=mesh,
        n_samples=n_samples,
        n_bins_histogram=n_bins,
    )
    D3, _ = calculate_area_triangle(
        mesh=mesh,
        n_samples=n_samples,
        n_bins_histogram=n_bins,
    )
    D4, _ = calculate_volume_tetrahedron(
        mesh=mesh,
        n_samples=n_samples,
        n_bins_histogram=n_bins,
    )

    return [
        relative_path,
        *A3.tolist(),
        *D1.tolist(),
        *D2.tolist(),
        *D3.tolist(),
        *D4.tolist(),
    ]


def analyze_and_save(
    input: str | Path,
    output: str | Path,
    bins: int = 64,
    samples: int = 100_000,
):
    if isinstance(input, str):
        input = Path(input)

    if isinstance(output, str):
        output = Path(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    histogram_descriptors = ["A3", "D1", "D2", "D3", "D4"]

    header = ["file_path"] + [
        f"{h_desc}_{i}"
        for h_desc in histogram_descriptors
        for i in range(bins)
    ]

    # get all 3D files (.obj)
    files = [item for item in input.rglob("*.obj") if item.is_file()]

    # Loop through files and map them into their corresponding subplot
    with open(output, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        tasks = [(file, input, samples, bins) for file in files]

        with ProcessPoolExecutor() as executor:
            n_workers = os.cpu_count() or 1
            print(f"Using {n_workers} worker processes")
            for row in tqdm(
                executor.map(proccess_file, tasks, chunksize=16),
                total=len(tasks),
                desc="Calculating Histogram Descriptors",
            ):
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input folder of 3D shapes to analyze.",
    )

    parser.add_argument(
        "-o", "--output", type=Path, help="Output .csv folder of results."
    )

    parser.add_argument("-s", "--samples", type=int, default=100_000)

    parser.add_argument("-b", "--bins", type=int, default=64)

    arguements = parser.parse_args()

    analyze_and_save(
        input=arguements.input,
        output=arguements.output,
        samples=arguements.samples,
        bins=arguements.bins,
    )


if __name__ == "__main__":
    main()
