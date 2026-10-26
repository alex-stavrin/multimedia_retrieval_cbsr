from src.step_4.distance_calculations.distance_functions import (
    euclidean_distance,
    earth_movers_distance,
    cosine_distance,
)

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import numpy as np

IDENTIFIER_COLUMN = "file_path"

DistanceFunction = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class DistanceConfig:
    function: DistanceFunction
    descriptors: list[str]
    histogram: bool = False


EUCLIDEAN_DESCRIPTORS = [
    "surface_area",
    "compactness",
    "rectangularity",
    "diameter",
    "convexity",
    "eccentricity",
]

COSINE_DESCRIPTORS = []

EMD_DESCRIPTORS = ["A3", "D1", "D2", "D3", "D4"]

DISTANCE_CONFIG = {
    "euclidean": DistanceConfig(
        euclidean_distance, EUCLIDEAN_DESCRIPTORS, histogram=False
    ),
    "cosine": DistanceConfig(
        cosine_distance, COSINE_DESCRIPTORS, histogram=True
    ),
    "emd": DistanceConfig(
        earth_movers_distance, EMD_DESCRIPTORS, histogram=True
    ),
}


class ShapeDistanceCalculator:
    """Calculate mesh distances using reusable descriptor CSV databases."""

    def __init__(
        self,
        csv_databases: list[str | Path],
        distance_config: dict[str, DistanceConfig] | None = None,
        identifier_column: str = IDENTIFIER_COLUMN,
    ) -> None:
        self.csv_databases = [Path(path) for path in csv_databases]
        self.distance_config = (
            distance_config.copy()
            if distance_config is not None
            else DISTANCE_CONFIG.copy()
        )
        self.identifier_column = identifier_column

        # Load CSV databases once so the object can be reused.
        self.databases = {
            path: pd.read_csv(path) for path in self.csv_databases
        }

        self.all_distances: dict[str, dict[str, dict[str, float]]] = {}

    def _find_database(self, descriptor: str, histogram: bool) -> Path:
        """Find the CSV containing the requested descriptor."""
        for path, df in self.databases.items():
            if self.identifier_column not in df.columns:
                continue

            if histogram:
                # Match columns such as A3_0, A3_1, D2_0, D2_1, etc.
                matching_columns = [
                    column
                    for column in df.columns
                    if column.startswith(f"{descriptor}_")
                    and column[len(descriptor) + 1 :].isdigit()
                ]

                if matching_columns:
                    return path

            elif descriptor in df.columns:
                return path

        descriptor_type = "histogram" if histogram else "elemental"
        searched_files = ", ".join(str(path) for path in self.databases)

        raise ValueError(
            f"Could not find {descriptor_type} descriptor '{descriptor}' "
            f"in any supplied CSV file. Searched: {searched_files}"
        )

    def _find_weights(self) -> dict[tuple[str, str], float]:
        """Find and validate the descriptor weights CSV."""

        required_columns = {
            "distance_function",
            "descriptor",
            "weight",
        }

        weights_df = next(
            (
                df
                for df in self.databases.values()
                if required_columns.issubset(df.columns)
            ),
            None,
        )

        if weights_df is None:
            raise ValueError(
                "No weight CSV found. Expected columns: "
                "'distance_function', 'descriptor', and 'weight'."
            )

        weights_df = weights_df.copy()
        weights_df["weight"] = pd.to_numeric(
            weights_df["weight"],
            errors="coerce",
        )

        if weights_df["weight"].isna().any():
            raise ValueError(
                "The weight CSV contains missing or invalid weights."
            )

        weight_values = weights_df["weight"].to_numpy(dtype=np.float64)

        if not np.isfinite(weight_values).all():
            raise ValueError("The weight CSV contains non-finite weights.")

        if (weight_values < 0).any():
            raise ValueError("Weights must be non-negative.")

        if weights_df.duplicated(
            subset=["distance_function", "descriptor"]
        ).any():
            raise ValueError(
                "Duplicate (distance_function, descriptor) entries "
                "found in the weight CSV."
            )

        weights: dict[tuple[str, str], float] = {
            (str(function_name), str(descriptor)): float(weight)
            for function_name, descriptor, weight in zip(
                weights_df["distance_function"].astype(str),
                weights_df["descriptor"].astype(str),
                weight_values,
            )
        }

        return weights

    @staticmethod
    def _normalize_shape_path(shape: str | Path) -> str:
        """Normalize a mesh path to its category/filename identifier."""
        return (
            Path(shape).as_posix().split("4_models_fully_normalized/", 1)[-1]
        )

    def calculate_all_distances(
        self,
        query: str,
        shapes_to_compare: list[str],
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Calculate distances and store results using normalized shape paths."""
        all_distances = {}

        for function_name, config in self.distance_config.items():
            all_distances[function_name] = {}

            for descriptor in config.descriptors:
                csv_database = self._find_database(
                    descriptor=descriptor,
                    histogram=config.histogram,
                )

                distances = config.function(
                    csv_database=csv_database,
                    query=query,
                    shapes_to_compare=shapes_to_compare,
                    descriptor=descriptor,
                    identifier_column=self.identifier_column,
                    histogram=config.histogram,
                )

                # Normalize candidate paths before storing the results.
                normalized_distances = {}

                for shape, distance in distances.items():
                    normalized_shape = self._normalize_shape_path(shape)

                    if normalized_shape in normalized_distances:
                        raise ValueError(
                            f"Multiple shapes normalize to the same "
                            f"identifier '{normalized_shape}' for descriptor "
                            f"'{descriptor}' using '{function_name}'"
                        )

                    normalized_distances[normalized_shape] = distance

                all_distances[function_name][descriptor] = normalized_distances

        self.all_distances = all_distances
        return self.all_distances

    def calculate_total_distance(
        self,
        query: str,
        shapes_to_compare: list[str],
    ) -> dict[str, float]:
        """
        Calculate weighted total distances for candidate meshes.

        Distances are calculated for the supplied query and candidates.
        The results are sorted from smallest to largest total distance.
        Result keys use normalized mesh paths.
        """

        self.calculate_all_distances(
            query=query,
            shapes_to_compare=shapes_to_compare,
        )

        weights = self._find_weights()

        calculated_pairs = {
            (function_name, descriptor)
            for function_name, descriptors in self.all_distances.items()
            for descriptor in descriptors
        }

        missing_weights = calculated_pairs - weights.keys()

        if missing_weights:
            raise ValueError(
                "Missing weights for calculated descriptors: "
                f"{sorted(missing_weights)}"
            )

        normalized_shapes = {
            shape: self._normalize_shape_path(shape)
            for shape in shapes_to_compare
        }

        # Use normalized paths as the result keys.
        total_distances = {
            normalized_shape: 0.0
            for normalized_shape in normalized_shapes.values()
        }

        for function_name, descriptors in self.all_distances.items():
            for descriptor, shape_distances in descriptors.items():
                weight = weights[(function_name, descriptor)]

                for normalized_shape in normalized_shapes.values():
                    if normalized_shape not in shape_distances:
                        raise ValueError(
                            f"Missing distance for '{normalized_shape}', "
                            f"descriptor '{descriptor}', "
                            f"function '{function_name}'."
                        )

                    distance = shape_distances[normalized_shape]

                    if not np.isfinite(distance):
                        raise ValueError(
                            f"Non-finite distance for shape"
                            f" '{normalized_shape}', "
                            f"descriptor '{descriptor}', "
                            f"function '{function_name}'."
                        )

                    total_distances[normalized_shape] += weight * distance

        return dict(
            sorted(
                total_distances.items(),
                key=lambda item: item[1],
            )
        )

    def get_distances(
        self,
        function_name: str,
        descriptor: str,
    ) -> dict[str, float]:
        """Return the most recently calculated descriptor distances."""

        try:
            return self.all_distances[function_name][descriptor]
        except KeyError as exc:
            raise KeyError(
                f"No calculated results for function '{function_name}' "
                f"and descriptor '{descriptor}'. "
                "Call calculate_all_distances() first."
            ) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_paths",
        nargs="+",
        metavar="CSV",
        required=True,
        type=str,
        help="File path(s) to csv file(s) containing the descriptor values",
    )

    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help=("Query mesh identifier, filename, or relative path."),
    )

    parser.add_argument(
        "-s",
        "--shapes",
        nargs="+",
        metavar="MESH",
        required=True,
        help=("Comparison mesh identifiers, filenames, " "or relative paths."),
    )

    args = parser.parse_args()

    parser.add_argument
    calculator = ShapeDistanceCalculator(
        csv_databases=args.csv_paths,
        distance_config=DISTANCE_CONFIG,
    )

    print(calculator.calculate_all_distances(args.query, args.shapes))
    print(calculator.calculate_total_distance(args.query, args.shapes))


if __name__ == "__main__":
    main()
