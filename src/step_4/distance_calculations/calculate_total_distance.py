from src.step_4.distance_calculations.distance_functions import (
    euclidean_distance,
    earth_movers_distance,
    cosine_distance,
)
from src.step_4.distance_calculations.visualisations import (
    visualize_mesh_distances,
)

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Sequence, Any
import json
from tqdm import tqdm

import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

_DIR_EXPANSION_CACHE: dict[Path, list[str]] = {}


class ShapeDistanceCalculator:
    """Calculate mesh distances using reusable descriptor CSV databases."""

    def __init__(
        self,
        csv_databases: list[str | Path],
        distance_config: dict[str, DistanceConfig] | None = DISTANCE_CONFIG,
        identifier_column: str = IDENTIFIER_COLUMN,
        max_workers: int | None = None,
    ) -> None:
        self.csv_databases = [Path(path) for path in csv_databases]
        self.distance_config = (
            distance_config.copy()
            if distance_config is not None
            else DISTANCE_CONFIG.copy()
        )
        self.identifier_column = identifier_column
        self.max_workers = (
            max_workers
            if max_workers is not None
            else min(32, (os.cpu_count() or 1) + 4)
        )

        if self.max_workers < 1:
            raise ValueError(
                f"max_workers must be at least 1: `{self.max_workers}`"
            )

        self.databases = {
            path: pd.read_csv(path) for path in self.csv_databases
        }

        self.all_distances: dict[str, dict[str, dict[str, float]]] = {}
        self.total_distances: dict[str, float] = {}
        self.last_query: str = ""

        # Lazy caches populated when descriptors or weights are first looked up
        self._database_cache: dict[tuple[str, bool], pd.DataFrame] = {}
        self._weights_cache: dict[tuple[str, str], float] | None = None

    @staticmethod
    def _has_histogram_columns(df: pd.DataFrame, descriptor: str) -> bool:
        prefix = f"{descriptor}_"
        return any(
            str(col).startswith(prefix) and str(col)[len(prefix) :].isdigit()
            for col in df.columns
        )

    def _find_database(
        self, descriptor: str, histogram: bool = False
    ) -> pd.DataFrame:
        """Find the CSV database containing the requested descriptor and cache it."""
        cache_key = (descriptor, histogram)
        cached_df = self._database_cache.get(cache_key)
        if cached_df is not None:
            return cached_df

        # Primary search matching the requested histogram mode
        for df in self.databases.values():
            if self.identifier_column not in df.columns:
                continue

            if histogram:
                if self._has_histogram_columns(df, descriptor):
                    self._database_cache[cache_key] = df
                    return df
            else:
                if descriptor in df.columns:
                    self._database_cache[cache_key] = df
                    return df

        # Fallback: check if the descriptor is stored in the opposite format
        # (e.g. a histogram family listed under a non-histogram DistanceConfig)
        for df in self.databases.values():
            if self.identifier_column not in df.columns:
                continue

            if not histogram and self._has_histogram_columns(df, descriptor):
                self._database_cache[cache_key] = df
                return df
            if histogram and descriptor in df.columns:
                self._database_cache[cache_key] = df
                return df

        descriptor_type = "histogram" if histogram else "elemental"
        searched_files = ", ".join(str(path) for path in self.databases)
        raise ValueError(
            f"Could not find {descriptor_type} descriptor '{descriptor}' "
            f"in any supplied CSV file. Searched: {searched_files}"
        )

    def _find_weights(self) -> dict[tuple[str, str], float]:
        """Find, validate, and cache the descriptor weights CSV."""
        if self._weights_cache is not None:
            return self._weights_cache

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

        self._weights_cache = weights
        return weights

    @staticmethod
    def _normalize_shape_path(shape: str | Path) -> str:
        """Normalize a mesh path to its category/filename identifier."""
        posix_path = str(shape).replace("\\", "/").strip("/")
        if "4_models_fully_normalized/" in posix_path:
            return posix_path.split("4_models_fully_normalized/", 1)[-1]
        parts = [part for part in posix_path.split("/") if part and part != "."]
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return posix_path

    @staticmethod
    def _expand_mesh_paths(paths: Sequence[str | Path]) -> list[str]:
        """
        Expand explicit mesh files, directories, or database identifiers into a
        list of OBJ mesh paths/identifiers.

        Directories are searched recursively and cached. Duplicate entries are
        removed while preserving order.
        """
        mesh_paths: list[str] = []
        seen_paths: set[str] = set()

        for item in paths:
            path = Path(item)

            if path.is_file():
                if path.suffix.lower() != ".obj":
                    raise ValueError(
                        "Expected an OBJ mesh file or directory, got: "
                        f"'{path}'"
                    )
                posix_candidate = path.as_posix()
                norm_key = os.path.normcase(os.path.abspath(path))
                if norm_key not in seen_paths:
                    seen_paths.add(norm_key)
                    mesh_paths.append(posix_candidate)

            elif path.is_dir():
                resolved_dir = path.resolve()
                cached_files = _DIR_EXPANSION_CACHE.get(resolved_dir)
                if cached_files is None:
                    discovered: list[str] = []
                    for root, _, files in os.walk(path):
                        for fname in files:
                            if fname.lower().endswith(".obj"):
                                discovered.append(Path(root, fname).as_posix())
                    discovered.sort()
                    cached_files = discovered
                    _DIR_EXPANSION_CACHE[resolved_dir] = cached_files

                if not cached_files:
                    print(f"Warning: no OBJ meshes found in '{path}'")

                for posix_candidate in cached_files:
                    norm_key = os.path.normcase(
                        os.path.abspath(posix_candidate)
                    )
                    if norm_key not in seen_paths:
                        seen_paths.add(norm_key)
                        mesh_paths.append(posix_candidate)

            else:
                # Allow mesh identifiers / filenames that refer to entries in
                # the CSV database rather than local filesystem directories.
                identifier = str(item).replace("\\", "/").strip("/")
                if identifier and identifier not in seen_paths:
                    seen_paths.add(identifier)
                    mesh_paths.append(identifier)

        if not mesh_paths:
            raise ValueError("No OBJ meshes were found in the supplied paths.")

        return mesh_paths

    @staticmethod
    def _calculate_descriptor_task(
        function_name: str,
        descriptor: str,
        config: DistanceConfig,
        df: pd.DataFrame,
        query: str,
        shapes_to_compare: list[str],
        identifier_column: str,
    ) -> tuple[str, str, dict[str, float]]:
        """Calculate distances for one descriptor."""
        distances = config.function(
            df=df,
            query=query,
            shapes_to_compare=shapes_to_compare,
            descriptor=descriptor,
            identifier_column=identifier_column,
            histogram=config.histogram,
        )

        normalized_distances = {}

        for shape, distance in distances.items():
            normalized_shape = ShapeDistanceCalculator._normalize_shape_path(
                shape
            )

            if normalized_shape in normalized_distances:
                raise ValueError(
                    f"Multiple shapes normalize to the same identifier "
                    f"'{normalized_shape}' for descriptor '{descriptor}' "
                    f"using '{function_name}'"
                )

            normalized_distances[normalized_shape] = distance

        return function_name, descriptor, normalized_distances

    def calculate_all_distances(
        self,
        query: str,
        shapes_to_compare: list[str],
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Calculate descriptor distances concurrently.
        Public call that always expands list to avoid error."""

        self.last_query = query
        expanded_shapes_to_compare = self._expand_mesh_paths(shapes_to_compare)

        return self._calculate_all_distances_expanded(
            query, expanded_shapes_to_compare
        )

    def _calculate_all_distances_expanded(
        self,
        query: str,
        expanded_shapes_to_compare: list[str],
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Calculates descriptor distances concurrently from given expanded
        file paths to avoid repeating path expansion."""
        tasks = []
        for function_name, config in self.distance_config.items():
            for descriptor in config.descriptors:
                csv_database = self._find_database(
                    descriptor=descriptor,
                    histogram=config.histogram,
                )

                tasks.append(
                    (
                        function_name,
                        descriptor,
                        config,
                        csv_database,
                        query,
                        expanded_shapes_to_compare,
                        self.identifier_column,
                    )
                )

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._calculate_descriptor_task, *task)
                for task in tasks
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Calculating distances",
            ):
                function_name, descriptor, distances = future.result()
                results[(function_name, descriptor)] = distances

        # Reconstruct the original nested dictionary and ordering.
        all_distances = {}

        for function_name, config in self.distance_config.items():
            all_distances[function_name] = {}

            for descriptor in config.descriptors:
                all_distances[function_name][descriptor] = results[
                    (function_name, descriptor)
                ]

        self.all_distances = all_distances
        return self.all_distances

    @staticmethod
    def _resolve_candidate_key(
        shape: str,
        available_keys: dict[str, str],
        filename_to_key: dict[str, str | None],
        stem_to_key: dict[str, str | None],
    ) -> str:
        norm = ShapeDistanceCalculator._normalize_shape_path(shape)
        if norm in available_keys:
            return norm

        fname = Path(norm).name
        if "." in fname:
            matched = filename_to_key.get(fname)
            if matched is not None:
                return matched
        else:
            matched = stem_to_key.get(fname)
            if matched is not None:
                return matched

        return norm

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

        self.last_query = query

        shapes_to_compare = self._expand_mesh_paths(shapes_to_compare)

        self._calculate_all_distances_expanded(
            query=query,
            expanded_shapes_to_compare=shapes_to_compare,
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

        # Build fast lookup from the calculated distance keys so that even
        # filename-only or stem-only shapes_to_compare entries match cleanly.
        first_distances: dict[str, float] = next(
            (
                shape_distances
                for descriptors in self.all_distances.values()
                for shape_distances in descriptors.values()
            ),
            {},
        )
        available_keys = {k: k for k in first_distances}
        filename_to_key: dict[str, str | None] = {}
        stem_to_key: dict[str, str | None] = {}
        for k in first_distances:
            fname = Path(k).name
            filename_to_key[fname] = (
                None if fname in filename_to_key else k
            )
            stem = Path(fname).stem
            stem_to_key[stem] = None if stem in stem_to_key else k

        normalized_shape_list = [
            self._resolve_candidate_key(
                shape, available_keys, filename_to_key, stem_to_key
            )
            for shape in shapes_to_compare
        ]

        # Use normalized paths as the result keys.
        total_distances = {
            normalized_shape: 0.0
            for normalized_shape in normalized_shape_list
        }

        for function_name, descriptors in self.all_distances.items():
            for descriptor, shape_distances in descriptors.items():
                weight = weights[(function_name, descriptor)]

                for normalized_shape in normalized_shape_list:
                    if normalized_shape not in shape_distances:
                        raise ValueError(
                            f"Missing distance for '{normalized_shape}', "
                            f"descriptor '{descriptor}', "
                            f"function '{function_name}'."
                        )

                    distance = shape_distances[normalized_shape]

                    if not np.isfinite(distance):
                        raise ValueError(
                            f"Non-finite distance for shape "
                            f"'{normalized_shape}', "
                            f"descriptor '{descriptor}', "
                            f"function '{function_name}'."
                        )

                    total_distances[normalized_shape] += weight * distance

        self.total_distances = dict(
            sorted(
                total_distances.items(),
                key=lambda item: item[1],
            )
        )

        return self.total_distances

    def _format_results(
        self,
        query: str,
        total_distances: dict[str, float],
        all_distances: dict[str, dict[str, dict[str, float]]],
    ) -> dict[Any, Any]:
        """
        Combine total distances and individual descriptor distances
        into a JSON-serializable results dictionary.
        """

        query_path = self._normalize_shape_path(query)
        results = {}

        for mesh_path, total_distance in total_distances.items():
            descriptor_distances = {}

            for descriptors in all_distances.values():
                for descriptor, distances in descriptors.items():
                    if mesh_path in distances:
                        descriptor_distances[descriptor] = distances[mesh_path]

            results[mesh_path] = {
                "total_distance": total_distance,
                "descriptor_distances": descriptor_distances,
            }

        return {
            "query": query_path,
            "results": results,
        }

    def save_results(
        self,
        output_path: str | Path,
        query: str | None = None,
        results: dict[Any, Any] | None = None,
    ) -> None:
        """Save formatted distance results to a JSON file."""

        if not results:
            if self.total_distances == {}:
                raise ValueError(
                    "Results not given, nor latest total distances saved."
                )
            if self.all_distances == {}:
                raise ValueError(
                    "Results not given, nor latest all distances saved."
                )

            if not query:
                results = self._format_results(
                    query=self.last_query,
                    all_distances=self.all_distances,
                    total_distances=self.total_distances,
                )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

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

    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Visualises the distances between shapes in interacrtive 3D "
        "renderder",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file directory for json file storing calculation result",
    )

    args = parser.parse_args()

    calculator = ShapeDistanceCalculator(
        csv_databases=args.csv_paths,
        distance_config=DISTANCE_CONFIG,
    )

    args.shapes = ShapeDistanceCalculator._expand_mesh_paths(args.shapes)

    total_distances = calculator.calculate_total_distance(
        args.query, args.shapes
    )

    if args.output:
        calculator.save_results(args.output)

    if args.visualise:
        visualize_mesh_distances(
            query=args.query,
            shapes_to_compare=args.shapes,
            descriptor="total distance",
            distances=total_distances,
        )


if __name__ == "__main__":
    main()
