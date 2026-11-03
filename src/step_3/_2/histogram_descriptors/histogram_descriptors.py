import numpy
import vedo

DEFAULT_SEED: int = 42


def _normalize_histogram(
    histogram: numpy.ndarray,
    bin_edges: numpy.ndarray,
    normalize_by_sum: bool = False,
) -> numpy.ndarray:
    """
    Normalize histogram either:
    - by sum (sum of bins = 1): histogram / histogram.sum()
    - by area / integral (integral over bin edges = 1): the default area method
    """
    if normalize_by_sum:
        total = float(numpy.sum(histogram))
        if total <= 0 or not numpy.isfinite(total):
            raise ValueError("Cannot normalize an empty or invalid histogram.")
        return histogram / total

    area = float(numpy.sum(histogram * numpy.diff(bin_edges)))
    if area <= 0 or not numpy.isfinite(area):
        raise ValueError("Cannot normalize an empty or invalid histogram.")
    return histogram / area


def _normalize_histogram_area(
    histogram: numpy.ndarray,
    bin_edges: numpy.ndarray,
) -> numpy.ndarray:
    """Scale bin heights so their integral across the bin edges is one (legacy compatibility)."""
    return _normalize_histogram(histogram, bin_edges, normalize_by_sum=False)


def calculate_angle_histogram(
    mesh: vedo.Mesh,
    n_samples: int,
    n_bins_histogram: int,
    seed: int | numpy.random.Generator | None = DEFAULT_SEED,
    normalize_by_sum: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]:

    # --- 1. Take n_samples triplets of vertices ---
    rng = numpy.random.default_rng(seed)

    # turn vertices to an array
    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)
    # take random samples of 3 (vertex_1, vertex_2, vertex_3)
    samples_indices: numpy.ndarray = rng.integers(
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
        angles, bins=n_bins_histogram, range=(0, 1)
    )
    histogram = _normalize_histogram(
        histogram, bin_edges, normalize_by_sum=normalize_by_sum
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


def calculate_distance_from_barycenter(
    mesh: vedo.Mesh,
    n_bins_histogram: int,
    seed: int | numpy.random.Generator | None = None,
    normalize_by_sum: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)
    barycenter: numpy.ndarray = numpy.mean(vertices, axis=0)

    # calculate distances from barycenter to vertices
    vectors: numpy.ndarray = vertices - barycenter
    distances: numpy.ndarray = numpy.linalg.norm(vectors, axis=1)

    # because the shapes are already normalized in size. The furthest a vertex can be is
    # \sqrt{0.5^2 + 0.5^2 + 0.5^2} = \sqrt{1/4 + 1/4 + 1/4} = \sqrt{3/4} = \sqrt{3} / 2
    maximum_possible_distance: float = numpy.sqrt(3) / 2
    distances = numpy.clip(distances / maximum_possible_distance, 0.0, 1.0)

    # make the histogram
    histogram, bin_edges = numpy.histogram(
        distances, bins=n_bins_histogram, range=(0, 1)
    )

    histogram = _normalize_histogram(
        histogram, bin_edges, normalize_by_sum=normalize_by_sum
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


def calculate_distance_two_vertices(
    mesh: vedo.Mesh,
    n_samples: int,
    n_bins_histogram: int,
    seed: int | numpy.random.Generator | None = DEFAULT_SEED,
    normalize_by_sum: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    rng = numpy.random.default_rng(seed)
    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)

    samples_indices: numpy.ndarray = rng.integers(
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

    histogram = _normalize_histogram(
        histogram, bin_edges, normalize_by_sum=normalize_by_sum
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


def calculate_area_triangle(
    mesh: vedo.Mesh,
    n_samples: int,
    n_bins_histogram: int,
    seed: int | numpy.random.Generator | None = DEFAULT_SEED,
    normalize_by_sum: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    rng = numpy.random.default_rng(seed)
    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)

    # here we take 3 vertices per sample
    samples_indices: numpy.ndarray = rng.integers(
        low=0, high=len(vertices), size=(n_samples, 3)
    )

    first_vertices: numpy.ndarray = vertices[samples_indices[:, 0]]
    second_vertices: numpy.ndarray = vertices[samples_indices[:, 1]]
    third_vertices: numpy.ndarray = vertices[samples_indices[:, 2]]

    vectors_1: numpy.ndarray = second_vertices - first_vertices
    vectors_2: numpy.ndarray = third_vertices - first_vertices

    cross_products: numpy.ndarray = numpy.cross(vectors_1, vectors_2)
    areas: numpy.ndarray = numpy.sqrt(
        0.5 * numpy.linalg.norm(cross_products, axis=1)
    )

    areas = areas[areas > 0]
    maximum_possible_area: float = numpy.sqrt(numpy.sqrt(3) / 2)
    areas = numpy.clip(areas / maximum_possible_area, 0.0, 1.0)

    histogram, bin_edges = numpy.histogram(
        areas, bins=n_bins_histogram, range=(0, 1)
    )
    histogram = _normalize_histogram(
        histogram, bin_edges, normalize_by_sum=normalize_by_sum
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers


def calculate_volume_tetrahedron(
    mesh: vedo.Mesh,
    n_samples: int,
    n_bins_histogram: int,
    seed: int | numpy.random.Generator | None = DEFAULT_SEED,
    normalize_by_sum: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    rng = numpy.random.default_rng(seed)
    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)

    sample_indices: numpy.ndarray = rng.integers(
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

    histogram, bin_edges = numpy.histogram(
        volumes, bins=n_bins_histogram, range=(0, 1)
    )
    histogram = _normalize_histogram(
        histogram, bin_edges, normalize_by_sum=normalize_by_sum
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return histogram, bin_centers
