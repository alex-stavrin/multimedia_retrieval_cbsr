import argparse
from pathlib import Path
import vedo
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from src.step_3._2.histogram_descriptors.histogram_descriptors import (
    calculate_angle_histogram,
    calculate_distance_from_barycenter,
    calculate_distance_two_vertices,
    calculate_area_triangle,
    calculate_volume_tetrahedron,
    DEFAULT_SEED,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input", type=Path, help="Input folder of 3D shapes to analyze."
    )

    parser.add_argument(
        "-d",
        "--descriptor",
        type=str,
        choices=["A3", "D1", "D2", "D3", "D4"],
        default="A3",
    )

    parser.add_argument("-s", "--samples", type=int, default=100_000)

    parser.add_argument("-b", "--bins", type=int, default=64)

    parser.add_argument("-cl", "--classes", type=int, default=12)

    parser.add_argument("-c", "--columns", type=int, default=4)

    parser.add_argument("-pdf", "--pdf", action="store_true")

    parser.add_argument(
        "-ym",
        "--ymax",
        "--y-max",
        dest="ymax",
        type=float,
        default=None,
        help="Maximum value for the y-axis. If not provided, each subplot scales dynamically to fit its data.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible descriptor sampling (default: 42).",
    )

    arguements = parser.parse_args()

    # get all 3D files (.obj)
    files = [item for item in arguements.input.rglob("*.obj") if item.is_file()]

    classes = sorted({file.parent.name for file in files})[: arguements.classes]

    number_of_rows_plot: int = math.ceil(len(classes) / arguements.columns)

    figure, axes = plt.subplots(
        number_of_rows_plot,
        arguements.columns,
        figsize=(12, 8),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.flatten()

    # Map each class to its subplot
    class_to_axis = {}
    class_max_y = {model_class: 0.0 for model_class in classes}
    for i, model_class in enumerate(classes):
        axis = axes[i]
        axis.set_title(model_class)
        axis.set_xlim(0, 1)
        class_to_axis[model_class] = axis

    # Hide empty subplots in the grid
    for j in range(len(classes), len(axes)):
        axes[j].set_visible(False)

    titles = {
        "A3": "A3 : Angle histograms for each Class",
        "D1": "D1 : Barycenter distance from random vertex histograms for each Class",
        "D2": "D2 : Distance between 2 random vertices for each Class",
        "D3": "D3 : Square root of area of triangle given by 3 random vertices",
        "D4": "D4 : Cube root of volume of tetrahedron formed by 4 random vertices",
    }
    base_title = titles.get(
        arguements.descriptor, f"{arguements.descriptor} histograms for each Class"
    )
    figure.suptitle(
        f"{base_title} (Samples: {arguements.samples}, Bins: {arguements.bins})"
    )

    max_y = -1

    # Filter files belonging to the selected classes
    target_files = [file for file in files if file.parent.name in class_to_axis]

    # Loop through files and map them into their corresponding subplot
    for file in tqdm(target_files, desc=f"Processing {arguements.descriptor} descriptors"):
        model_class = file.parent.name
        mesh = vedo.Mesh(str(file))

        if arguements.descriptor == "A3":
            histogram, bin_centers = calculate_angle_histogram(
                mesh=mesh,
                n_samples=arguements.samples,
                n_bins_histogram=arguements.bins,
                seed=arguements.seed,
            )
        elif arguements.descriptor == "D1":
            histogram, bin_centers = calculate_distance_from_barycenter(
                mesh=mesh,
                n_bins_histogram=arguements.bins,
                seed=arguements.seed,
            )
        elif arguements.descriptor == "D2":
            histogram, bin_centers = calculate_distance_two_vertices(
                mesh=mesh,
                n_samples=arguements.samples,
                n_bins_histogram=arguements.bins,
                seed=arguements.seed,
            )
        elif arguements.descriptor == "D3":
            histogram, bin_centers = calculate_area_triangle(
                mesh=mesh,
                n_samples=arguements.samples,
                n_bins_histogram=arguements.bins,
                seed=arguements.seed,
            )
        elif arguements.descriptor == "D4":
            histogram, bin_centers = calculate_volume_tetrahedron(
                mesh=mesh,
                n_samples=arguements.samples,
                n_bins_histogram=arguements.bins,
                seed=arguements.seed,
            )
        else:
            raise ValueError(f"Unsupported descriptor: {arguements.descriptor}")

        current_max = float(histogram.max())
        max_y = max(max_y, current_max)
        class_max_y[model_class] = max(class_max_y[model_class], current_max)

        axis = class_to_axis[model_class]
        axis.plot(bin_centers, histogram, alpha=0.7)

    # Set y-limits: custom ymax if provided, otherwise scale each subplot to fit its class data
    for model_class, axis in class_to_axis.items():
        if arguements.ymax is not None:
            axis.set_ylim(0, arguements.ymax)
        else:
            local_max = class_max_y[model_class]
            if local_max > 0:
                axis.set_ylim(0, local_max * 1.05)

    if arguements.pdf:
        figure.savefig(
            f"{arguements.descriptor}_histogram_{arguements.samples}_{arguements.bins}.pdf",
            bbox_inches="tight",
        )
    plt.show()


if __name__ == "__main__":
    main()
