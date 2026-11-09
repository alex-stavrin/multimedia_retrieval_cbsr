from src.step_4.distance_calculations.distance_functions import (
    cdf_earth_movers_distance,
    euclidean_distance,
    earth_movers_distance,
    cosine_distance,
)
from src.step_4.distance_calculations.visualisations import (
    visualize_mesh_distances,
)

from src.helper.mesh_descriptor_extraction import calculate_all_descriptors
from src.helper.mesh_preprocessing import preprocess_mesh
from src.helper.path_helpers import normalize_path_separators
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Sequence, Any
import json
import vedo
import random

import pandas as pd
import numpy as np

IDENTIFIER_COLUMN = "file_path"

DistanceFunction = Callable[..., dict[str, float]]

@dataclass(frozen=True)
class DistanceConfig:
    function: DistanceFunction
    descriptors: list[str]
    histogram: bool = False


ELEMENTAL_DESCRIPTORS = [
    "surface_area",
    "compactness",
    "rectangularity",
    "diameter",
    "convexity",
    "eccentricity",
]

HISTOGRAM_DESCRIPTORS = ["A3", "D1", "D2", "D3", "D4"]

DISTANCE_CONFIG_EMD = {
    "euclidean": DistanceConfig(
        euclidean_distance, ELEMENTAL_DESCRIPTORS, histogram=False
    ),
    "emd": DistanceConfig(
        earth_movers_distance, HISTOGRAM_DESCRIPTORS, histogram=True
    ),
}

DISTANCE_CONFIG_COSINE = {
    "euclidean": DistanceConfig(
        euclidean_distance, ELEMENTAL_DESCRIPTORS, histogram=False
    ),
    "cosine": DistanceConfig(
        cosine_distance, HISTOGRAM_DESCRIPTORS, histogram=True
    ),
}


DISTANCE_CONFIG_L2 = {
    "euclidean_elemental": DistanceConfig(
        euclidean_distance, ELEMENTAL_DESCRIPTORS, histogram=False
    ),
    "euclidean_hist": DistanceConfig(
        euclidean_distance,
        HISTOGRAM_DESCRIPTORS,
        histogram=True,
    ),
}


class ShapeDistanceCalculator:
    """Calculate mesh distances using
    fast vectorized descriptor comparisons."""

    def __init__(
        self,
        csv_databases: list[str | Path],
        distance_config: dict[str, DistanceConfig] = DISTANCE_CONFIG_EMD,
        identifier_column: str = IDENTIFIER_COLUMN,
    ) -> None:
        self.csv_databases = [Path(path) for path in csv_databases]
        self.identifier_column = identifier_column
        self.distance_config = distance_config

        self.databases: dict[Path, pd.DataFrame] = {}
        for path in self.csv_databases:
            if not path.is_file():
                raise FileNotFoundError(f"CSV database not found: {path}")
            df = pd.read_csv(path)
            if self.identifier_column in df.columns:
                df[self.identifier_column] = (
                    df[self.identifier_column]
                    .astype(str)
                    .map(normalize_path_separators)
                )
            self.databases[path] = df

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
        self.distance_scales: dict[tuple[str, str], float] = {}
        if self.weights_df is not None and "distance_scale" in self.weights_df:
            for _, row in self.weights_df.iterrows():
                key = (
                    str(row["distance_function"]),
                    str(row["descriptor"]),
                )
                scale = pd.to_numeric(row["distance_scale"], errors="coerce")
                if pd.isna(scale) or not np.isfinite(scale) or scale <= 0:
                    raise ValueError(
                        "Invalid distance_scale for "
                        f"('{key[0]}', '{key[1]}'): {row['distance_scale']}"
                    )
                self.distance_scales[key] = float(scale)

        # A distance-function change should not require editing the config
        # and weights CSV in lockstep. Reuse a descriptor weight only when
        # the CSV has exactly one unambiguous weight for that descriptor.
        for function_name, config in self.distance_config.items():
            for descriptor in config.descriptors:
                key = (function_name, descriptor)
                if key not in self.weights:
                    matches = [
                        weight
                        for (_, candidate), weight in self.weights.items()
                        if candidate == descriptor
                    ]
                    if len(matches) == 1:
                        self.weights[key] = matches[0]
                if key not in self.distance_scales:
                    matches = [
                        scale
                        for (
                            _,
                            candidate,
                        ), scale in self.distance_scales.items()
                        if candidate == descriptor
                    ]
                    if len(matches) == 1:
                        self.distance_scales[key] = matches[0]

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
        self._shape_id_to_pos: dict[str, int] = {
            key: idx for idx, key in enumerate(self.shape_identifiers)
        }
        self._candidate_cache: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}

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
        cache_key = (id(source_df), tuple(columns))
        cached = self._candidate_cache.get(cache_key)
        if cached is None:
            indexed = source_df.set_index(self.identifier_column)
            candidates = indexed.loc[self.shape_identifiers, columns].to_numpy(
                dtype=float
            )
            cached = {"candidates": candidates}
            self._candidate_cache[cache_key] = cached
        else:
            candidates = cached["candidates"]
        query_vector = np.asarray(query_features, dtype=float)

        if config.function is euclidean_distance:
            values = (
                np.linalg.norm(candidates - query_vector, axis=1)
                if is_histogram
                else np.abs(candidates[:, 0] - query_vector[0])
            )
            return dict(zip(self.shape_identifiers, values.astype(float)))

        if config.function is earth_movers_distance and is_histogram:
            candidate_cdfs = cached.get("candidate_cdfs")
            if candidate_cdfs is None:
                candidate_sums = candidates.sum(axis=1, keepdims=True)
                candidate_sums[candidate_sums == 0] = 1.0
                candidate_cdfs = np.cumsum(candidates / candidate_sums, axis=1)[
                    :, :-1
                ]
                cached["candidate_cdfs"] = candidate_cdfs
            query_sum = query_vector.sum()
            if query_sum == 0:
                query_sum = 1.0
            query_cdf = np.cumsum(query_vector / query_sum)[:-1]
            values = cdf_earth_movers_distance(
                candidate_cdfs,
                query_cdf,
                number_of_bins=candidates.shape[1],
            )
            return dict(zip(self.shape_identifiers, values.astype(float)))

        if config.function is cosine_distance:
            candidate_norms = cached.get("candidate_norms")
            if candidate_norms is None:
                candidate_norms = np.linalg.norm(candidates, axis=1)
                cached["candidate_norms"] = candidate_norms
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
            normalize_path_separators(shape): float(distance)
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
        """Delegate query preprocessing and descriptor extraction."""
        is_fully_normalized_file = False
        if isinstance(mesh_or_path, (str, Path)):
            mesh = vedo.Mesh(str(mesh_or_path))
            is_fully_normalized_file = any(
                part.endswith("_models_fully_normalized")
                for part in Path(mesh_or_path).parts
            )
        else:
            mesh = mesh_or_path
        if not is_fully_normalized_file:
            mesh = preprocess_mesh(mesh)

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
        processed_scalars = {
            columns[0]
            for key, (
                _,
                columns,
                is_histogram,
            ) in self._descriptor_sources.items()
            if (
                not is_histogram
                and len(columns) == 1
                and self._descriptor_source_paths[key].stem.endswith(
                    "_processed"
                )
            )
        }
        missing_transforms = processed_scalars - self._query_transforms.keys()
        if missing_transforms:
            raise ValueError(
                "Cannot transform query descriptors to match the supplied "
                "processed database. Could not infer raw-to-database "
                f"transforms for: {sorted(missing_transforms)}. "
                "Supply matching raw "
                "descriptor data or query descriptors in the database's "
                "transformed space."
            )

        scalar_values, histogram_values = calculate_all_descriptors(
            mesh,
            scalar_descriptors,
            histogram_bins,
        )
        for descriptor, (scale, offset) in self._query_transforms.items():
            if descriptor in scalar_values:
                scalar_values[descriptor] = (
                    scale * scalar_values[descriptor] + offset
                )
        return scalar_values, histogram_values

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
                distance_scale = self.distance_scales.get(
                    (function_name, descriptor), 1.0
                )
                # print(
                #     f"Descriptor: {descriptor}\n"
                #     f"Distance scale: {distance_scale}\n"
                #     f"Wieght: {weight}\n\n"
                # )
                for shape in totals:
                    if shape not in distances:
                        raise ValueError(
                            f"No distance for '{shape}' and descriptor "
                            f"'{descriptor}'."
                        )
                    totals[shape] += weight * distance_scale * distances[shape]
        return dict(sorted(totals.items(), key=lambda item: item[1]))

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
            normalized_shape = normalize_path_separators(shape)

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
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Calculate distances using the active distance configuration."""
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
                cache_key = (id(source_df), tuple(columns))
                cached = self._candidate_cache.get(cache_key)
                if cached is None:
                    indexed = source_df.set_index(self.identifier_column)
                    candidates = indexed.loc[
                        self.shape_identifiers, columns
                    ].to_numpy(dtype=float)
                    cached = {"candidates": candidates}
                    self._candidate_cache[cache_key] = cached
                else:
                    candidates = cached["candidates"]
                query_pos = self._shape_id_to_pos.get(query_key)
                if query_pos is not None:
                    query_features = candidates[query_pos]
                else:
                    qf = source_df.set_index(
                        self.identifier_column
                    ).loc[query_key, columns]
                    if isinstance(qf, pd.Series):
                        query_features = qf.to_numpy(dtype=float)
                    else:
                        query_features = np.asarray([qf], dtype=float)

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
                        normalize_path_separators(shape): float(distance)
                        for shape, distance in distances.items()
                    }
                if query_key in distances:
                    distances[query_key] = 0.0
                else:
                    normalized_query = normalize_path_separators(query_key)
                    if normalized_query in distances:
                        distances[normalized_query] = 0.0
                all_distances[function_name][descriptor] = distances

        self.all_distances = all_distances
        return self.all_distances

    def calculate_total_distance(
        self,
        query: str,
        shapes_to_compare: list[str] | None = None,
        run_pipeline: bool = True,
    ) -> dict[str, float]:
        """
        Calculate weighted total distances for candidate meshes.
        If run_pipeline=True, executes the complete pipeline on the query mesh.
        If run_pipeline=False, looks up precomputed values directly.
        """
        self.last_query = str(query)

        if run_pipeline and (
            isinstance(query, vedo.Mesh)
            or (isinstance(query, (str, Path)) and Path(query).is_file())
        ):
            return self.calculate_total_distance_from_mesh(query)

        all_distances = self.calculate_all_distances(str(query))
        self.total_distances = self._sum_weighted_distances(all_distances)
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

        weighting = {
            function_name: {
                descriptor: {
                    "weight": self.weights.get((function_name, descriptor)),
                    "distance_scale": self.distance_scales.get(
                        (function_name, descriptor), 1.0
                    ),
                }
                for descriptor in config.descriptors
            }
            for function_name, config in self.distance_config.items()
        }

        return {
            "query": query_path,
            "weighting": weighting,
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

    available_configs = {
        "emd": DISTANCE_CONFIG_EMD,
        "emd-equal": DISTANCE_CONFIG_EMD,
        "emd-rec": DISTANCE_CONFIG_EMD,
        "l2": DISTANCE_CONFIG_L2,
        "l2-equal": DISTANCE_CONFIG_L2,
        "l2-rec": DISTANCE_CONFIG_L2,
        "cosine": DISTANCE_CONFIG_COSINE,
        "cosine-equal": DISTANCE_CONFIG_COSINE,
        "cosine-rec": DISTANCE_CONFIG_COSINE,
    }

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
        help="Query mesh identifier, filename, or path.",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        nargs="*",
        metavar="MESH",
        default=None,
        help="Comparison mesh identifiers or paths.",
    )
    parser.add_argument(
        "-c",
        "--config",
        choices=tuple(available_configs.keys()),
        default="mixed",
        help=(
            "Distance configuration: 'mixed' uses Euclidean elemental and "
            "EMD histogram distances; 'l2' uses Euclidean for both. "
            "The '-equal' variants use equal descriptor weights; '*-rec' "
            "selects a recommended-weight configuration."
        ),
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Visualises the distances between shapes in interactive"
        "3D renderer",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file directory for json file storing calculation result",
    )
    parser.add_argument(
        "--max_top_results_printing",
        type=int,
        default=10,
        help="The final result printing in CLI"
    )

    args = parser.parse_args()

    distance_config = available_configs[args.config]

    calculator = ShapeDistanceCalculator(
        csv_databases=args.csv_paths,
        distance_config=distance_config,
    )
    total_distances = calculator.calculate_total_distance(
        args.query, args.shapes, run_pipeline=True
    )
    print(f"Total distances calculated. Top {args.max_top_results_printing}:")
    for k, v in list(total_distances.items())[:args.max_top_results_printing]:
        print(f"  {k}: {v:.4f}")

    if args.output:
        calculator.save_results(args.output)

    # Visualise shapes
    if args.visualise:
        shapes = args.shapes or list(total_distances.keys())[:10]
        if len(shapes) == 1 and Path(shapes[0]).is_dir():
            mesh_root = Path(shapes[0])
            shapes = [
                str(mesh_root / shape)
                for shape in list(total_distances.keys())[:10]
            ]
        else:
            # Distance keys are usually database-relative identifiers such as
            # ``Car/m1496.obj`` rather than paths relative to the working
            # directory. When the query is a mesh file, its class directory's
            # parent is the likely database root; use it to resolve those keys.
            query_path = Path(args.query)
            if query_path.is_file():
                mesh_root = query_path.parent.parent
                shapes = [
                    shape if Path(shape).is_file()
                    else str(mesh_root / shape)
                    for shape in shapes
                ]
        visualize_mesh_distances(
            query=args.query,
            shapes_to_compare=shapes,
            distances=total_distances,
            config=(
                "Euclidean (equal weights)"
                if args.config == "l2-equal"
                else (
                    "Euclidean"
                    if args.config.startswith("l2")
                    else (
                        "Cosine" if args.config.startswith("cosine") else "EMD"
                    )
                )
            ),
        )


if __name__ == "__main__":
    main()
