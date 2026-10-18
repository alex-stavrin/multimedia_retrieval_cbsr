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
    histogram, bin_edges = numpy.histogram(angles, bins=n_bins_histogram, range=(0, 1))
    # normalize the histogram
    histogram = histogram / histogram.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


def calculate_distance_from_barycenter(
    mesh: vedo.Mesh, n_samples: int, n_bins_histogram: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)
    barycenter: numpy.ndarray = numpy.mean(vertices, axis=0)

    # pick a total of n_samples samples
    samples_indices: numpy.ndarray = numpy.random.randint(
        low=0, high=len(vertices), size=n_samples
    )

    samples_vertices: numpy.ndarray = vertices[samples_indices]

    # calculate distances from barycenter to vertices
    vectors: numpy.ndarray = samples_vertices - barycenter
    distances: numpy.ndarray = numpy.linalg.norm(vectors, axis=1)

    # because the shapes are already normalized in size. The furthest a vertex can be is
    # \sqrt{0.5^2 + 0.5^2 + 0.5^2} = \sqrt{1/4 + 1/4 + 1/4} = \sqrt{3/4} = \sqrt{3} / 2
    maximum_possible_distance: float = numpy.sqrt(3) / 2
    distances = numpy.clip(distances / maximum_possible_distance, 0.0, 1.0)

    # make the histogram
    histogram, bin_edges = numpy.histogram(
        distances, bins=n_bins_histogram, range=(0, 1)
    )

    # normalize histogram
    histogram = histogram / histogram.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


def calculate_distance_two_vertices(
    mesh: vedo.Mesh, n_samples: int, n_bins_histogram: int
) -> tuple[numpy.ndarray, numpy.ndarray]:

    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)

    samples_indices: numpy.ndarray = numpy.random.randint(
        low=0, high=len(vertices), size=(n_samples, 2)
    )

    first_vertices: numpy.ndarray = vertices[samples_indices[:, 0]]
    second_vertices: numpy.ndarray = vertices[samples_indices[:, 1]]

    vectors: numpy.ndarray = second_vertices - first_vertices
    distances: numpy.ndarray = numpy.linalg.norm(vectors, axis=1)

    # filter out 0 distances. That means both vertices of the samples are the same
    distances = distances[distances > 0]

    # here again because we have size-normalized the shape. The maximum distnace could be \sqrt{1 + 1 + 1} = \sqrt{3}
    maximum_possible_distance: float = numpy.sqrt(3)
    distances = numpy.clip(distances / maximum_possible_distance, 0.0, 1.0)

    histogram, bin_edges = numpy.histogram(
        distances, bins=n_bins_histogram, range=(0, 1)
    )

    histogram = histogram / histogram.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


def calculate_area_triangle(
    mesh: vedo.Mesh, n_samples: int, n_bins_histogram: int
) -> tuple[numpy.ndarray, numpy.ndarray]:

    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)

    # here we take 3 vertices per sample
    samples_indices: numpy.ndarray = numpy.random.randint(
        low=0, high=len(vertices), size=(n_samples, 3)
    )

    first_vertices: numpy.ndarray = vertices[samples_indices[:, 0]]
    second_vertices: numpy.ndarray = vertices[samples_indices[:, 1]]
    third_vertices: numpy.ndarray = vertices[samples_indices[:, 2]]

    vectors_1: numpy.ndarray = second_vertices - first_vertices
    vectors_2: numpy.ndarray = third_vertices - first_vertices

    cross_products: numpy.ndarray = numpy.cross(vectors_1, vectors_2)
    areas: numpy.ndarray = numpy.sqrt(0.5 * numpy.linalg.norm(cross_products, axis=1))

    areas = areas[areas > 0]
    maximum_possible_area: float = numpy.sqrt(numpy.sqrt(3) / 2)
    areas = numpy.clip(areas / maximum_possible_area, 0.0, 1.0)

    histogram, bin_edges = numpy.histogram(areas, bins=n_bins_histogram, range=(0, 1))
    histogram = histogram / histogram.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


def calculate_volume_tetrahedron(
    mesh: vedo.Mesh, n_samples: int, n_bins_histogram: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)

    sample_indices: numpy.ndarray = numpy.random.randint(
        low=0, high=len(vertices), size=(n_samples, 4)
    )

    first_vertices: numpy.ndarray = vertices[sample_indices[:, 0]]
    second_vertices: numpy.ndarray = vertices[sample_indices[:, 1]]
    third_vertices: numpy.ndarray = vertices[sample_indices[:, 2]]
    fourth_vertices: numpy.ndarray = vertices[sample_indices[:, 3]]

    vectors_1: numpy.ndarray = second_vertices - first_vertices
    vectors_2: numpy.ndarray = third_vertices - first_vertices
    vectors_3: numpy.ndarray = fourth_vertices - first_vertices

    cross_products: numpy.ndarray = numpy.cross(vectors_1, vectors_2)
    dot_products: numpy.ndarray = numpy.sum(cross_products * vectors_3, axis=1)

    volumes: numpy.ndarray = numpy.cbrt(numpy.abs(dot_products) / 6.0)

    volumes = volumes[volumes > 0]

    maximum_possible_volume: float = numpy.cbrt(1.0 / 3.0)
    volumes = numpy.clip(volumes / maximum_possible_volume, 0.0, 1.0)

    histogram, bin_edges = numpy.histogram(volumes, bins=n_bins_histogram, range=(0, 1))
    histogram = histogram / histogram.sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


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

    parser.add_argument("-pdf", "--pdf", action="store_true")

    arguements = parser.parse_args()

    # get all 3D files (.obj)
    files = [item for item in arguements.input.rglob("*.obj") if item.is_file()]

    classes = sorted({file.parent.name for file in files})[: arguements.classes]

    number_of_rows: int = math.ceil(len(classes) / arguements.columns)

    figure, axes = plt.subplots(
        number_of_rows, arguements.columns, figsize=(12, 8), constrained_layout=True
    )
    axes = axes.flatten()

    # Map each class to its subplot
    class_to_axis = {}
    for i, model_class in enumerate(classes):
        axis = axes[i]
        axis.set_title(model_class)
        axis.set_xlim(0, 1)
        class_to_axis[model_class] = axis

    # Hide empty subplots in the grid
    for j in range(len(classes), len(axes)):
        axes[j].set_visible(False)

    max_y = -1

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
                figure.suptitle("A3 : Angle histograms for each Class")

            elif arguements.descriptor == "D1":
                histogram, bin_centers = calculate_distance_from_barycenter(
                    mesh=mesh,
                    n_samples=arguements.samples,
                    n_bins_histogram=arguements.bins,
                )
                figure.suptitle(
                    "D1 : Barycenter distance from random vertex histograms for each Class"
                )
            elif arguements.descriptor == "D2":
                histogram, bin_centers = calculate_distance_two_vertices(
                    mesh=mesh,
                    n_samples=arguements.samples,
                    n_bins_histogram=arguements.bins,
                )
                figure.suptitle(
                    "D2 : Distance between 2 random vertices for each Class"
                )
            elif arguements.descriptor == "D3":
                histogram, bin_centers = calculate_area_triangle(
                    mesh=mesh,
                    n_samples=arguements.samples,
                    n_bins_histogram=arguements.bins,
                )
                figure.suptitle(
                    "D3 : Square root of area of triangle given by 3 random vertices"
                )
            elif arguements.descriptor == "D4":
                histogram, bin_centers = calculate_volume_tetrahedron(
                    mesh=mesh,
                    n_samples=arguements.samples,
                    n_bins_histogram=arguements.bins,
                )
                figure.suptitle(
                    "D4 : Cube root of volume of tetrahedron formed by 4 random vertices"
                )
            else:
                raise ValueError(f"Unsupported descriptor: {arguements.descriptor}")

            max_y = max(max_y, histogram.max())

            axis = class_to_axis[model_class]
            axis.plot(bin_centers, histogram, alpha=0.7)

    # set max y for all subplots
    for axis in class_to_axis.values():
        axis.set_ylim(0, max_y)

    if arguements.pdf:
        figure.savefig(f"{arguements.descriptor}_histograms.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
