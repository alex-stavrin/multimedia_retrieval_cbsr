"""Extract configured scalar and histogram descriptors from a mesh."""

from collections.abc import Mapping

import numpy as np
import vedo

from src.step_3._2.elementary_descriptors.descriptor_analyzer import (
    calculate_mesh_descriptors,
)
from src.step_3._2.histogram_descriptors.histogram_descriptors import (
    DEFAULT_SEED,
    calculate_angle_histogram,
    calculate_distance_from_barycenter,
    calculate_distance_two_vertices,
    calculate_area_triangle,
    calculate_volume_tetrahedron,
)


def calculate_all_descriptors(
    mesh: vedo.Mesh,
    scalar_descriptors: set[str],
    histogram_bins: Mapping[str, int],
    normalize_by_sum: bool = True,
    seed: int | None = DEFAULT_SEED,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Calculate requested descriptors without modifying or preprocessing mesh."""
    raw_scalars = calculate_mesh_descriptors(mesh) if scalar_descriptors else {}
    scalar_values = {
        descriptor: float(raw_scalars[descriptor]) for descriptor in scalar_descriptors
    }
    histogram_functions = {
        "A3": lambda bins: calculate_angle_histogram(
            mesh,
            n_samples=100_000,
            n_bins_histogram=bins,
            seed=seed,
            normalize_by_sum=normalize_by_sum,
        )[0],
        "D1": lambda bins: calculate_distance_from_barycenter(
            mesh,
            n_bins_histogram=bins,
            seed=seed,
            normalize_by_sum=normalize_by_sum,
        )[0],
        "D2": lambda bins: calculate_distance_two_vertices(
            mesh,
            n_samples=100_000,
            n_bins_histogram=bins,
            seed=seed,
            normalize_by_sum=normalize_by_sum,
        )[0],
        "D3": lambda bins: calculate_area_triangle(
            mesh,
            n_samples=100_000,
            n_bins_histogram=bins,
            seed=seed,
            normalize_by_sum=normalize_by_sum,
        )[0],
        "D4": lambda bins: calculate_volume_tetrahedron(
            mesh,
            n_samples=100_000,
            n_bins_histogram=bins,
            seed=seed,
            normalize_by_sum=normalize_by_sum,
        )[0],
    }
    histogram_values: dict[str, np.ndarray] = {}
    for descriptor, bins in histogram_bins.items():
        extractor = histogram_functions.get(descriptor)
        if extractor is None:
            raise ValueError(
                f"No mesh extractor is registered for histogram " f"'{descriptor}'."
            )
        histogram_values[descriptor] = extractor(bins)

    return scalar_values, histogram_values
