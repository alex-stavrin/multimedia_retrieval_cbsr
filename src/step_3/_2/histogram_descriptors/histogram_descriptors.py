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
    """Scale bin heights so their integral across the bin edges is one
    (legacy compatibility)."""
    return _normalize_histogram(histogram, bin_edges, normalize_by_sum=False)


def _sample_surface_points(
    mesh: vedo.Mesh,
    n_samples: int,
    rng: numpy.random.Generator,
) -> numpy.ndarray:
    """Sample points uniformly over a triangular mesh's surface area."""
    vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
    faces = numpy.asarray(mesh.cells, dtype=numpy.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Surface sampling requires a mesh of triangle faces.")

    triangles = vertices[faces]
    cross_products = numpy.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * numpy.linalg.norm(cross_products, axis=1)
    valid_faces = numpy.isfinite(areas) & (areas > 0)
    if not valid_faces.any():
        raise ValueError("Cannot sample a mesh with no positive-area faces.")

    face_indices = numpy.flatnonzero(valid_faces)
    probabilities = areas[valid_faces] / areas[valid_faces].sum()
    selected_faces = face_indices[
        rng.choice(len(face_indices), size=n_samples, p=probabilities)
    ]
    selected_triangles = triangles[selected_faces]

    # Square-root barycentric sampling is uniform within each selected face.
    u = numpy.sqrt(rng.random(n_samples))
    v = rng.random(n_samples)
    weights = numpy.column_stack((1.0 - u, u * (1.0 - v), u * v))
    return numpy.sum(selected_triangles * weights[:, :, None], axis=1)


def calculate_angle_histogram(
    mesh: vedo.Mesh,
    n_samples: int,
    n_bins_histogram: int,
    seed: int | numpy.random.Generator | None = DEFAULT_SEED,
    normalize_by_sum: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]:

    # --- 1. Take n_samples triplets of surface points ---
    rng = numpy.random.default_rng(seed)

    # Previous vertex-index sampling:
    # vertices = numpy.asarray(mesh.vertices)
    # samples_indices = rng.integers(
    #     low=0, high=len(vertices), size=(n_samples, 3)
    # )
    # first_vertices = vertices[samples_indices[:, 0]]
    # second_vertices = vertices[samples_indices[:, 1]]
    # third_vertices = vertices[samples_indices[:, 2]]
    sampled_points = _sample_surface_points(mesh, n_samples * 3, rng).reshape(
        n_samples, 3, 3
    )
    first_vertices = sampled_points[:, 0]
    second_vertices = sampled_points[:, 1]
    third_vertices = sampled_points[:, 2]

    # --- 2. Turn our 3 vertices into two vectors starting at the first vertex
    # and ending in second and third one ---
    vectors_1: numpy.ndarray = second_vertices - first_vertices
    vectors_2: numpy.ndarray = third_vertices - first_vertices

    # norm() stands for Euclidean norm
    lengths_1: numpy.ndarray = numpy.linalg.norm(vectors_1, axis=1)
    lengths_2: numpy.ndarray = numpy.linalg.norm(vectors_2, axis=1)

    # keep only samples were the length of the vectors are both not zero
    # (filter out duplicate vertices)
    # this basically says, keep only vectors that have higher than 0 length
    # and vertex 2 and 3 are different
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
    # arccos() returns radians so in the range [0, π].
    # We normalize so that its in [0, 1]
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
    seed: int | numpy.random.Generator | None = DEFAULT_SEED,
    normalize_by_sum: bool = False,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    vertices: numpy.ndarray = numpy.asarray(mesh.vertices)
    barycenter: numpy.ndarray = numpy.mean(vertices, axis=0)

    # Previous vertex-set sampling:
    # vectors = vertices - barycenter
    sampled_points = _sample_surface_points(
        mesh,
        len(vertices),
        numpy.random.default_rng(seed),
    )
    vectors: numpy.ndarray = sampled_points - barycenter

    # Alex I removed clipping since the final meshes
    # aren't inside a 1x1x1 box actually
    distances: numpy.ndarray = numpy.linalg.norm(vectors, axis=1)

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

    # Previous vertex-index sampling:
    # vertices = numpy.asarray(mesh.vertices)
    # samples_indices = rng.integers(
    #     low=0, high=len(vertices), size=(n_samples, 2)
    # )
    # first_vertices = vertices[samples_indices[:, 0]]
    # second_vertices = vertices[samples_indices[:, 1]]
    sampled_points = _sample_surface_points(mesh, n_samples * 2, rng).reshape(
        n_samples, 2, 3
    )
    first_vertices = sampled_points[:, 0]
    second_vertices = sampled_points[:, 1]

    vectors: numpy.ndarray = second_vertices - first_vertices
    distances: numpy.ndarray = numpy.linalg.norm(vectors, axis=1)

    # filter out 0 distances.
    # That means both vertices of the samples are the same
    distances = distances[distances > 0]

    # here again because we have size-normalized the shape.
    # The maximum distnace could be \sqrt{1 + 1 + 1} = \sqrt{3}
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

    # Previous vertex-index sampling:
    # vertices = numpy.asarray(mesh.vertices)
    # samples_indices = rng.integers(
    #     low=0, high=len(vertices), size=(n_samples, 3)
    # )
    # first_vertices = vertices[samples_indices[:, 0]]
    # second_vertices = vertices[samples_indices[:, 1]]
    # third_vertices = vertices[samples_indices[:, 2]]
    sampled_points = _sample_surface_points(mesh, n_samples * 3, rng).reshape(
        n_samples, 3, 3
    )
    first_vertices = sampled_points[:, 0]
    second_vertices = sampled_points[:, 1]
    third_vertices = sampled_points[:, 2]

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

    # Previous vertex-index sampling:
    # vertices = numpy.asarray(mesh.vertices)
    # sample_indices = rng.integers(
    #     low=0, high=len(vertices), size=(n_samples, 4)
    # )
    # first_vertices = vertices[sample_indices[:, 0]]
    # second_vertices = vertices[sample_indices[:, 1]]
    # third_vertices = vertices[sample_indices[:, 2]]
    # fourth_vertices = vertices[sample_indices[:, 3]]
    sampled_points = _sample_surface_points(mesh, n_samples * 4, rng).reshape(
        n_samples, 4, 3
    )
    first_vertices = sampled_points[:, 0]
    second_vertices = sampled_points[:, 1]
    third_vertices = sampled_points[:, 2]
    fourth_vertices = sampled_points[:, 3]

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
