import argparse
from pathlib import Path
import vedo
import numpy
import matplotlib.pyplot as plt
import math


def calculate_angle_histogram(
    mesh: vedo.Mesh, n_samples: int, n_bins_histogram: int
) -> tuple[numpy.ndarray, numpy.ndarray]:

    # --- 1. Take n_samples triplets of vertices ---

    # turn vertices to an array
    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)
    # take random samples of 3 (vertex_1, vertex_2, vertex_3)
    samples_indices: numpy.ndarray = numpy.random.randint(
        low=0, high=len(vertices), size=(n_samples, 3)
    )

    first_vertices: numpy.ndarray = vertices[samples_indices[:, 0]]
    second_vertices: numpy.ndarray = vertices[samples_indices[:, 1]]
    third_vertices: numpy.ndarray = vertices[samples_indices[:, 2]]

    # --- 2. Turn our 3 vertices into two vectors starting at the first vertex and ending in second and third one ---
    vectors_1: numpy.ndarray = second_vertices - first_vertices
    vectors_2: numpy.ndarray = third_vertices - first_vertices

    # norm() stands for Euclidean norm
    lengths_1: numpy.ndarray = numpy.linalg.norm(vectors_1, axis=1)
    lengths_2: numpy.ndarray = numpy.linalg.norm(vectors_2, axis=1)

    # keep only samples were the length of the vectors are both not zero (filter out duplicate vertices)
    # this basically says, keep only vectors that have higher than 0 length and vertex 2 and 3 are different
    mask: numpy.ndarray = (
        (lengths_1 > 0)
        & (lengths_2 > 0)
        & numpy.any(second_vertices != third_vertices, axis=1)
    )
    vectors_1 = vectors_1[mask]
    vectors_2 = vectors_2[mask]
    lengths_1 = lengths_1[mask]
    lengths_2 = lengths_2[mask]

    # --- 3. Calculate the angles of all the pairs of vectors ---
    dot_products: numpy.ndarray = numpy.sum(vectors_1 * vectors_2, axis=1)
    cosines = dot_products / (lengths_1 * lengths_2)
    # arccos() returns radians so in the range [0, π]. We normalize so that its in [0, 1]
    angles = numpy.arccos(numpy.clip(cosines, -1.0, 1.0)) / numpy.pi

    # --- 4. Make the histogram from the angles ---
    histogram, bin_edges = numpy.histogram(
        angles, bins=n_bins_histogram, range=(0, numpy.pi)
    )
    # normalize the histogram
    histogram = histogram / histogram.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


def calculate_distance_from_barycenter():
    pass


def calculate_distance_two_vertices():
    pass


def calculate_area_triangle():
    pass


def calculate_volume_tetrahedron():
    pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input", type=Path, help="Input folder of 3D shapes to analyze."
    )

    parser.add_argument(
        "-o", "--output", type=Path, help="Output .csv folder of results."
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

    arguements = parser.parse_args()

    # get all 3D files (.obj)
    files = [item for item in arguements.input.rglob("*.obj") if item.is_file()]

    classes = sorted({file.parent.name for file in files})[: arguements.classes]

    number_of_rows: int = math.ceil(len(classes) / arguements.columns)

    figure, axes = plt.subplots(
        number_of_rows, arguements.columns, figsize=(20, 12), constrained_layout=True
    )
    axes = axes.flatten()

    # Map each class to its subplot
    class_to_axis = {}
    for i, model_class in enumerate(classes):
        axis = axes[i]
        axis.set_title(model_class)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 0.7)
        class_to_axis[model_class] = axis

    # Hide empty subplots in the grid
    for j in range(len(classes), len(axes)):
        axes[j].set_visible(False)

    # Loop through files and map them into their corresponding subplot
    for file in files:
        model_class = file.parent.name

        if model_class in class_to_axis:
            mesh = vedo.Mesh(str(file))

            if arguements.descriptor == "A3":
                histogram, bin_centers = calculate_angle_histogram(
                    mesh=mesh,
                    n_samples=arguements.samples,
                    n_bins_histogram=arguements.bins,
                )
            elif arguements.descriptor == "D1":
                histogram, bin_centers = calculate_angle_histogram(
                    mesh=mesh,
                    n_samples=arguements.samples,
                    n_bins_histogram=arguements.bins,
                )
            else:
                raise ValueError(f"Unsupported descriptor: {arguements.descriptor}")

            axis = class_to_axis[model_class]
            axis.plot(bin_centers, histogram, alpha=0.7)

    figure.suptitle("Angle Histograms for each Class")
    plt.show()


if __name__ == "__main__":
    main()
