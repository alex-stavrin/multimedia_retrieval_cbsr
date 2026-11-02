"""Use the KD-tree with a query shape, so the Step 4 interface can show the
results of the ANN search in the same way as the exact search."""

import numpy

from src.step_4.distance_calculations.calculate_total_distance import (
    ShapeDistanceCalculator,
)
from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine


def build_query_vector(
    engine: SpatialSearchEngine,
    elementary: dict[str, float],
    histograms: dict[str, numpy.ndarray],
) -> numpy.ndarray:
    """Put the descriptors of a query shape in the column order of the index."""
    # the extractor gives a density, but the database has the share of every
    # bin, so that the bins of a histogram sum to 1
    shares: dict[str, numpy.ndarray] = {}
    for histogram_name, bins in histograms.items():
        bins = numpy.asarray(bins, dtype=numpy.float64)
        shares[histogram_name] = bins / bins.sum() if bins.sum() > 0 else bins

    values: list[float] = []
    for column in engine.feature_columns:
        if column in elementary:
            values.append(float(elementary[column]))
        else:
            # the bins of a histogram are called A3_0, A3_1, ...
            histogram_name, bin_index = column.rsplit("_", 1)
            values.append(float(shares[histogram_name][int(bin_index)]))
    return numpy.array(values, dtype=numpy.float64)


def knn_from_vector(
    engine: SpatialSearchEngine,
    vector: numpy.ndarray,
    k: int,
    eps: float = 0.0,
    p: float = 2.0,
) -> dict[str, float]:
    """The k nearest shapes to a vector as {shape id: distance}, closest first."""
    k = min(k, len(engine.shape_ids))
    distances, indices = engine.kd_tree.query(vector, k=k, eps=eps, p=p)

    return {
        engine.shape_ids[int(index)]: float(distance)
        for distance, index in zip(
            numpy.atleast_1d(distances), numpy.atleast_1d(indices)
        )
    }


def search(
    engine: SpatialSearchEngine,
    calculator: ShapeDistanceCalculator,
    shape_path: str,
    k: int,
    eps: float = 0.0,
) -> dict[str, float]:
    """KNN search for a shape from the database or from outside of it."""
    try:
        # a shape of the database already has its descriptors in the index
        vector = engine.feature_matrix[engine.resolve_query_index(shape_path)]
    except ValueError:
        # a new shape is repaired, resampled and normalised first, and its
        # descriptors get the same transform as the ones of the database
        elementary, histograms = calculator.process_mesh_and_extract_descriptors(
            shape_path
        )
        vector = build_query_vector(engine, elementary, histograms)

    return knn_from_vector(engine, vector, k, eps)
