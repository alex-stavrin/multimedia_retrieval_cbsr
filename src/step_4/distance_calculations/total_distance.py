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
    """Calculate and store mesh distances using multiple CSV databases."""

    def __init__(
        self,
        csv_databases: list[str | Path],
        query: str,
        shapes_to_compare: list[str],
        distance_config: dict[str, DistanceConfig] | None = None,
        identifier_column: str = IDENTIFIER_COLUMN,
    ) -> None:
        self.csv_databases = [Path(path) for path in csv_databases]
        self.query = query
        self.shapes_to_compare = shapes_to_compare
        self.distance_config = (
            distance_config.copy()
            if distance_config is not None
            else DISTANCE_CONFIG.copy()
        )
        self.identifier_column = identifier_column

        self.results: dict[str, dict[str, dict[str, float]]] = {}

        # Load the CSV files once.
        self.databases = {
            path: pd.read_csv(path) for path in self.csv_databases
        }

    def _find_database(self, descriptor: str, histogram: bool) -> Path:
        """Find the CSV containing the requested descriptor."""

        for path, df in self.databases.items():
            if self.identifier_column not in df.columns:
                continue

            if histogram:
                # Find histogram columns such as A3_0, A3_1, A3_2.
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

        raise ValueError(
            f"Could not find {descriptor_type} descriptor "
            f"'{descriptor}' in any of the supplied CSV files."
        )

    def calculate(self) -> dict[str, dict[str, dict[str, float]]]:
        """Calculate distances using the CSV containing each descriptor."""

        self.results = {}

        for function_name, config in self.distance_config.items():
            self.results[function_name] = {}

            for descriptor in config.descriptors:
                csv_database = self._find_database(
                    descriptor=descriptor,
                    histogram=config.histogram,
                )

                distances = config.function(
                    csv_database=csv_database,
                    query=self.query,
                    shapes_to_compare=self.shapes_to_compare,
                    descriptor=descriptor,
                    identifier_column=self.identifier_column,
                    histogram=config.histogram,
                )

                self.results[function_name][descriptor] = distances

        return self.results

    def get_distances(
        self,
        function_name: str,
        descriptor: str,
    ) -> dict[str, float]:
        """Return distances for a calculated function and descriptor."""

        try:
            return self.results[function_name][descriptor]
        except KeyError as exc:
            raise KeyError(
                f"No calculated results for function '{function_name}' "
                f"and descriptor '{descriptor}'. Call calculate() first."
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
        query=args.query,
        shapes_to_compare=args.shapes,
        distance_config=DISTANCE_CONFIG,
    )

    print(calculator.calculate())


if __name__ == "__main__":
    main()
