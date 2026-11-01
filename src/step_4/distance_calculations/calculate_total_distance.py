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
            distance_config if distance_config is not None else DISTANCE_CONFIG
        )

        self.databases: dict[Path, pd.DataFrame] = {}
        for path in self.csv_databases:
            if not path.is_file():
                raise FileNotFoundError(f"CSV database not found: {path}")
            self.databases[path] = pd.read_csv(path)

        self.simple_df: pd.DataFrame | None = None
        self.hist_df: pd.DataFrame | None = None
        self.weights_df: pd.DataFrame | None = None
        self._descriptor_sources: dict[
            tuple[str, str], tuple[pd.DataFrame, list[str], bool]
        ] = {}
        self._descriptor_source_paths: dict[tuple[str, str], Path] = {}
        self.all_distances: dict[str, dict[str, dict[str, float]]] = {}
        self.total_distances: dict[str, float] = {}
        self.last_query: str | None = None

        for path, df in self.databases.items():
            cols = set(df.columns)
            if {"distance_function", "descriptor", "weight"}.issubset(cols):
                self.weights_df = df
                continue
            if self.identifier_column not in cols:
                raise ValueError(
                    f"CSV database '{path}' has no identifier column "
                    f"'{self.identifier_column}'."
                )

        feature_databases = {
            path: df
            for path, df in self.databases.items()
            if not {"distance_function", "descriptor", "weight"}.issubset(
                df.columns
            )
        }
        if not feature_databases:
            raise ValueError("No descriptor databases were supplied.")

        self._histogram_columns: dict[tuple[Path, str], list[str]] = {}
        for path, df in feature_databases.items():
            for descriptor in self._available_histograms(df):
                self._histogram_columns[(path, descriptor)] = (
                    self._get_histogram_columns(df, descriptor)
                )

        for function_name, config in self.distance_config.items():
            for descriptor in config.descriptors:
                source_candidates = []
                for path, df in feature_databases.items():
                    scalar_present = descriptor in df.columns
                    histogram_columns = self._histogram_columns.get(
                        (path, descriptor), []
                    )
                    if config.histogram:
                        if histogram_columns:
                            source_candidates.append(
                                (path, df, histogram_columns, True)
                            )
                    elif scalar_present:
                        source_candidates.append(
                            (path, df, [descriptor], False)
                        )
                    elif histogram_columns:
                        # Infer histogram input from its numbered bin columns
                        # when the family is not stored as a scalar column.
                        source_candidates.append(
                            (path, df, histogram_columns, True)
                        )

                if not source_candidates:
                    raise ValueError(
                        f"No supplied CSV contains descriptor '{descriptor}' "
                        "in the form required by the distance configuration."
                    )

                if len(source_candidates) > 1:
                    processed_candidates = [
                        item
                        for item in source_candidates
                        if item[0].stem.endswith("_processed")
                    ]
                    if len(processed_candidates) == 1:
                        source_candidates = processed_candidates
                    else:
                        paths = [str(item[0]) for item in source_candidates]
                        raise ValueError(
                            f"Descriptor '{descriptor}' is present in more "
                            f"than one supplied CSV: {paths}"
                        )

                path, source_df, columns, is_histogram = source_candidates[0]
                self._descriptor_sources[(function_name, descriptor)] = (
                    source_df,
                    columns,
                    is_histogram,
                )
                self._descriptor_source_paths[(function_name, descriptor)] = (
                    path
                )
                if is_histogram and self.hist_df is None:
                    self.hist_df = source_df
                if not is_histogram and self.simple_df is None:
                    self.simple_df = source_df

        if not self._descriptor_sources:
            raise ValueError(
                "The active distance configuration has no descriptors."
            )

        if self.simple_df is None:
            # Preserve a common candidate index even for histogram-only
            # configurations.
            first_source = next(iter(self._descriptor_sources.values()))[0]
            self.simple_df = first_source

        # Parse descriptor weights
        self.weights: dict[tuple[str, str], float] = {}
        if self.weights_df is not None:
            self.weights = {
                (str(r["distance_function"]), str(r["descriptor"])): float(
                    r["weight"]
                )
                for _, r in self.weights_df.iterrows()
            }

        # Pairwise descriptor distances have very different units (for
        # example, scalar z-score distances versus histogram-bin distances).
        # Use the pairwise mean from the same calibration CSV as a scale so a
        # typical distance for each descriptor is approximately one.
        self._distance_scales: dict[tuple[str, str], float] = {}
        if self.weights_df is not None and "mean" in self.weights_df.columns:
            for _, row in self.weights_df.iterrows():
                key = (str(row["distance_function"]), str(row["descriptor"]))
                scale = pd.to_numeric(row["mean"], errors="coerce")
                if pd.notna(scale) and np.isfinite(scale) and scale > 0:
                    self._distance_scales[key] = float(scale)

        self._query_transforms: dict[str, tuple[float, float]] = {}
        for key, (
            _,
            columns,
            is_histogram,
        ) in self._descriptor_sources.items():
            if is_histogram or len(columns) != 1:
                continue
            source_path = self._descriptor_source_paths[key]
            if not source_path.stem.endswith("_processed"):
                continue
            descriptor = columns[0]
            raw_sources = [
                (path, df)
                for path, df in self.databases.items()
                if path != source_path
                and descriptor in df.columns
                and self.identifier_column in df.columns
                and not path.stem.endswith("_processed")
            ]
            sibling_path = source_path.with_name(
                f"{source_path.stem.removesuffix('_processed')}.csv"
            )
            if sibling_path.is_file():
                raw_df = pd.read_csv(sibling_path)
            elif len(raw_sources) == 1:
                raw_df = raw_sources[0][1]
            else:
                continue

            raw_values = raw_df.set_index(self.identifier_column)[descriptor]
            processed_values = self.databases[source_path].set_index(
                self.identifier_column
            )[descriptor]
            paired = pd.concat(
                [raw_values.rename("raw"), processed_values.rename("fit")],
                axis=1,
                join="inner",
            ).dropna()
            x = paired["raw"].to_numpy(dtype=float)
            y = paired["fit"].to_numpy(dtype=float)
            variance = float(np.var(x))
            if variance == 0.0:
                scale = 0.0
                offset = float(np.mean(y))
            else:
                scale = float(
                    np.mean((x - x.mean()) * (y - y.mean())) / variance
                )
                offset = float(y.mean() - scale * x.mean())

            if not np.allclose(
                y,
                scale * x + offset,
                rtol=1e-5,
                atol=1e-7,
            ):
                raise ValueError(
                    f"The transformation from raw '{descriptor}' values to "
                    "the supplied descriptor database is not affine. Pass "
                    "query descriptors in the database's transformed space "
                    "or provide an explicit transform."
                )
            self._query_transforms[descriptor] = (scale, offset)

        identifiers = [
            set(df[self.identifier_column].dropna().astype(str))
            for df, _, _ in self._descriptor_sources.values()
        ]
        self.shape_identifiers = sorted(set.intersection(*identifiers))
        if not self.shape_identifiers:
            raise ValueError(
                "The supplied descriptor databases have no identifiers "
                "in common for the active distance configuration."
            )

        self._filename_to_keys: dict[str, set[str]] = {}
        for key in self.shape_identifiers:
            self._filename_to_keys.setdefault(key, set()).add(key)
            normalized_key = key.replace("\\", "/")
            self._filename_to_keys.setdefault(normalized_key, set()).add(key)
            self._filename_to_keys.setdefault(Path(key).name, set()).add(key)

    @staticmethod
    def _get_histogram_columns(
        df: pd.DataFrame,
        descriptor: str,
    ) -> list[str]:
        bins = []
        for column in df.columns:
            prefix, separator, suffix = str(column).rpartition("_")
            if separator and prefix == descriptor and suffix.isdigit():
                bins.append((int(suffix), str(column)))
        bins.sort(key=lambda item: item[0])
        indices = [index for index, _ in bins]
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Histogram '{descriptor}' has missing or non-consecutive "
                "bin columns."
            )
        return [column for _, column in bins]

    @classmethod
    def _available_histograms(cls, df: pd.DataFrame) -> set[str]:
        families = set()
        for column in df.columns:
            prefix, separator, suffix = str(column).rpartition("_")
            if separator and suffix.isdigit():
                families.add(prefix)
        return families

    def _calculate_configured_distances(
        self,
        config: DistanceConfig,
        source_df: pd.DataFrame,
        columns: list[str],
        is_histogram: bool,
        query_features: Sequence[float] | np.ndarray,
        descriptor: str,
        query_identifier: str | None = None,
    ) -> dict[str, float]:
        """Use vectorized built-in distances, with a generic fallback."""
        indexed = source_df.set_index(self.identifier_column)
        candidates = indexed.loc[self.shape_identifiers, columns].to_numpy(
            dtype=float
        )
        query_vector = np.asarray(query_features, dtype=float)

        if config.function is euclidean_distance:
            values = (
                np.linalg.norm(candidates - query_vector, axis=1)
                if is_histogram
                else np.abs(candidates[:, 0] - query_vector[0])
            )
            return dict(zip(self.shape_identifiers, values.astype(float)))

        if config.function is earth_movers_distance and is_histogram:
            candidate_sums = candidates.sum(axis=1, keepdims=True)
            candidate_sums[candidate_sums == 0] = 1.0
            query_sum = query_vector.sum()
            if query_sum == 0:
                query_sum = 1.0
            candidate_cdfs = np.cumsum(candidates / candidate_sums, axis=1)[
                :, :-1
            ]
            query_cdf = np.cumsum(query_vector / query_sum)[:-1]
            values = np.abs(candidate_cdfs - query_cdf).sum(axis=1)
            return dict(zip(self.shape_identifiers, values.astype(float)))

        if config.function is cosine_distance:
            candidate_norms = np.linalg.norm(candidates, axis=1)
            query_norm = float(np.linalg.norm(query_vector))
            if query_norm == 0 or np.any(candidate_norms == 0):
                raise ValueError(
                    f"Cosine distance is undefined for descriptor "
                    f"'{descriptor}' with a zero vector."
                )
            similarities = (
                candidates @ query_vector / (candidate_norms * query_norm)
            )
            values = 1.0 - np.clip(similarities, -1.0, 1.0)
            return dict(zip(self.shape_identifiers, values.astype(float)))

        if query_identifier is None:
            raise ValueError(
                "A query identifier is required for a custom distance "
                f"function on descriptor '{descriptor}'."
            )

        query_row = {
            self.identifier_column: query_identifier,
            **dict(zip(columns, query_features)),
        }
        augmented_df = pd.concat(
            [source_df, pd.DataFrame([query_row])],
            ignore_index=True,
        )
        distances = config.function(
            df=augmented_df,
            query=query_identifier,
            shapes_to_compare=self.shape_identifiers,
            descriptor=descriptor,
            identifier_column=self.identifier_column,
            histogram=is_histogram,
        )
        return {
            self._normalize_shape_path(shape): float(distance)
            for shape, distance in distances.items()
        }

    def _resolve_query_key(self, query: str) -> str:
        """Resolve an identifier or unambiguous filename in the databases."""
        normalized_query = str(query).replace("\\", "/").strip("/")
        exact_matches = self._filename_to_keys.get(normalized_query, set())
        if len(exact_matches) == 1:
            return next(iter(exact_matches))

        filename = Path(normalized_query).name
        filename_matches = self._filename_to_keys.get(filename, set())
        if len(filename_matches) == 1:
            return next(iter(filename_matches))
        if len(filename_matches) > 1:
            raise ValueError(
                f"Query filename '{filename}' is ambiguous; pass its full "
                "database identifier."
            )

        raise ValueError(
            f"Query '{query}' was not found in the supplied descriptor "
            "databases."
        )

    def process_mesh_and_extract_descriptors(
        self,
        mesh_or_path: str | Path | vedo.Mesh,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """
        Run the complete pipeline on a query mesh:
        1. Load mesh
        2. Repair wireframe holes and edges
        3. Resample mesh vertices
        4. Center, PCA-align and scale normalize
        5. Extract elementary descriptors & standardize with dataset stats
        6. Extract histogram descriptors (A3, D1, D2, D3, D4)
        """
        is_fully_normalized_file = False
        if isinstance(mesh_or_path, (str, Path)):
            mesh = vedo.Mesh(str(mesh_or_path))
            is_fully_normalized_file = any(
                part.endswith("_models_fully_normalized")
                for part in Path(mesh_or_path).parts
            )
        else:
            mesh = mesh_or_path.clone()

        # Database descriptor CSVs are extracted directly from the saved
        # fully-normalized meshes. Apply preprocessing only to raw queries.
        if not is_fully_normalized_file:
            # Step 2: Repair
            repaired_mesh, _ = repair(mesh)

            # Step 2.3: Resample
            mesh = resample_mesh(repaired_mesh)

            # Step 3.1: Full normalize
            full_normalize(mesh)

        scalar_descriptors = {
            columns[0]
            for _, columns, is_histogram in self._descriptor_sources.values()
            if not is_histogram and len(columns) == 1
        }
        histogram_bins = {
            descriptor: len(columns)
            for (_, descriptor), (
                _,
                columns,
                is_histogram,
            ) in self._descriptor_sources.items()
            if is_histogram
        }

        elem_raw = (
            calculate_mesh_descriptors(mesh) if scalar_descriptors else {}
        )
        elem_values = {
            descriptor: float(elem_raw[descriptor])
            for descriptor in scalar_descriptors
        }

        processed_scalar_descriptors = {
            columns[0]
            for key, (
                _,
                columns,
                is_histogram,
            ) in self._descriptor_sources.items()
            if not is_histogram
            and len(columns) == 1
            and self._descriptor_source_paths[key].stem.endswith("_processed")
        }
        missing_transforms = (
            processed_scalar_descriptors - self._query_transforms.keys()
        )
        if missing_transforms:
            raise ValueError(
                "No weight CSV found. Expected columns: "
                "'distance_function', 'descriptor', and 'weight'."
            )

        for descriptor, (scale, offset) in self._query_transforms.items():
            if descriptor in elem_values:
                elem_values[descriptor] = (
                    scale * elem_values[descriptor] + offset
                )

        histogram_functions = {
            "A3": lambda bins: calculate_angle_histogram(
                mesh, n_samples=100_000, n_bins_histogram=bins
            )[0],
            "D1": lambda bins: calculate_distance_from_barycenter(
                mesh, n_bins_histogram=bins
            )[0],
            "D2": lambda bins: calculate_distance_two_vertices(
                mesh, n_samples=100_000, n_bins_histogram=bins
            )[0],
            "D3": lambda bins: calculate_area_triangle(
                mesh, n_samples=100_000, n_bins_histogram=bins
            )[0],
            "D4": lambda bins: calculate_volume_tetrahedron(
                mesh, n_samples=100_000, n_bins_histogram=bins
            )[0],
        }
        hist_values: dict[str, np.ndarray] = {}
        for descriptor, bins in histogram_bins.items():
            extractor = histogram_functions.get(descriptor)
            if extractor is None:
                raise ValueError(
                    f"No mesh extractor is registered for histogram "
                    f"'{descriptor}'."
                )
            hist_values[descriptor] = extractor(bins)

        return elem_values, hist_values

    def calculate_total_distance_from_mesh(
        self,
        mesh_or_path: str | Path | vedo.Mesh,
    ) -> dict[str, float]:
        """
        Run the whole pipeline on the query mesh and compute total weighted
        distances against the precomputed database in < 1 second.
        """
        # If the query is one of the indexed database meshes, use its saved
        # descriptor row as the query vector. Re-extracting it from the raw
        # mesh would apply repair/resampling/normalization that was not used
        # to build the database descriptors, so even a self-match could be
        # non-zero.
        if isinstance(mesh_or_path, (str, Path)):
            try:
                query_key = self._resolve_query_key(str(mesh_or_path))
            except ValueError:
                query_key = None
            if query_key is not None:
                self.last_query = str(mesh_or_path)
                self.all_distances = self.calculate_all_distances(query_key)
                self.total_distances = self._sum_weighted_distances(
                    self.all_distances
                )
                return self.total_distances

        elem_descriptors, hist_descriptors = (
            self.process_mesh_and_extract_descriptors(mesh_or_path)
        )

        query_identifier = "__query_mesh__"
        while query_identifier in self.shape_identifiers:
            query_identifier += "_"

        all_distances: dict[str, dict[str, dict[str, float]]] = {}
        for function_name, config in self.distance_config.items():
            all_distances[function_name] = {}
            for descriptor in config.descriptors:
                source_df, columns, is_histogram = self._descriptor_sources[
                    (function_name, descriptor)
                ]
                query_features = (
                    hist_descriptors[descriptor]
                    if is_histogram
                    else [elem_descriptors[descriptor]]
                )
                all_distances[function_name][descriptor] = (
                    self._calculate_configured_distances(
                        config=config,
                        source_df=source_df,
                        columns=columns,
                        is_histogram=is_histogram,
                        query_features=query_features,
                        descriptor=descriptor,
                        query_identifier=query_identifier,
                    )
                )

        self.all_distances = all_distances
        return self._sum_weighted_distances(all_distances)

    def _sum_weighted_distances(
        self,
        all_distances: dict[str, dict[str, dict[str, float]]],
    ) -> dict[str, float]:
        totals = {shape: 0.0 for shape in self.shape_identifiers}
        for function_name, descriptors in all_distances.items():
            for descriptor, distances in descriptors.items():
                weight = self.weights.get((function_name, descriptor))
                if weight is None:
                    raise ValueError(
                        "Missing weight for configured descriptor "
                        f"('{function_name}', '{descriptor}')."
                    )
                scale = self._distance_scales.get(
                    (function_name, descriptor), 1.0
                )
                for shape in totals:
                    if shape not in distances:
                        raise ValueError(
                            f"No distance for '{shape}' and descriptor "
                            f"'{descriptor}'."
                        )
                    totals[shape] += weight * (distances[shape] / scale)
        return dict(sorted(totals.items(), key=lambda item: item[1]))

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
        query_key = self._resolve_query_key(query)
        all_distances: dict[str, dict[str, dict[str, float]]] = {}
        for function_name, config in self.distance_config.items():
            all_distances[function_name] = {}
            for descriptor in config.descriptors:
                source_df, _, is_histogram = self._descriptor_sources[
                    (function_name, descriptor)
                ]
                columns = self._descriptor_sources[
                    (function_name, descriptor)
                ][1]
                query_features = source_df.set_index(
                    self.identifier_column
                ).loc[query_key, columns]
                if isinstance(query_features, pd.Series):
                    query_features = query_features.to_numpy(dtype=float)
                else:
                    query_features = np.asarray([query_features], dtype=float)

                if config.function in {
                    euclidean_distance,
                    earth_movers_distance,
                    cosine_distance,
                }:
                    distances = self._calculate_configured_distances(
                        config=config,
                        source_df=source_df,
                        columns=columns,
                        is_histogram=is_histogram,
                        query_features=query_features,
                        descriptor=descriptor,
                    )
                else:
                    distances = config.function(
                        df=source_df,
                        query=query_key,
                        shapes_to_compare=self.shape_identifiers,
                        descriptor=descriptor,
                        identifier_column=self.identifier_column,
                        histogram=is_histogram,
                    )
                    distances = {
                        self._normalize_shape_path(shape): float(distance)
                        for shape, distance in distances.items()
                    }
                all_distances[function_name][descriptor] = distances

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
        """Combine total distances and descriptor distances into
        a dictionary."""
        if Path(query).is_file():
            query_parts = Path(query).parts
            query_path = "/".join(query_parts[-2:])
        else:
            query_path = self._resolve_query_key(query)
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
            if not self.all_distances:
                raise ValueError("No distances calculated to save.")

            total_distances = self.total_distances
            if not total_distances:
                # Mesh queries populate descriptor distances but return the
                # weighted totals without retaining them on the instance.
                total_distances = self._sum_weighted_distances(
                    self.all_distances
                )

            results = self._format_results(
                query=query or self.last_query or "",
                all_distances=self.all_distances,
                total_distances=total_distances,
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
