import argparse
from dataclasses import dataclass, field
from typing import Callable, Sequence

from pathlib import Path
import re

import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean, cosine as scipy_cosine
from scipy.stats import wasserstein_distance

from src.step_4.distance_calculations.visualisations import (
    visualize_mesh_distances,
)

IDENTIFIER_COLUMN = "file_path"


@dataclass
class _DataFrameShapeLookup:
    raw_identifiers: list[str]
    exact_to_idx: dict[str, int]
    filename_to_idx: dict[str, int | None]
    stem_to_idx: dict[str, int | None]
    batch_cache: dict[tuple[str, ...], tuple[np.ndarray, list[str]]] = field(
        default_factory=dict
    )

    def resolve_index(self, shape_identifier: str) -> int:
        idx = self.exact_to_idx.get(shape_identifier)
        if idx is not None:
            return idx

        identifier = shape_identifier.replace("\\", "/").strip("/")
        idx = self.exact_to_idx.get(identifier)
        if idx is not None:
            return idx

        if "/" in identifier:
            last_two = "/".join(identifier.rsplit("/", 2)[-2:])
            idx = self.exact_to_idx.get(last_two)
            if idx is not None:
                return idx

        filename = identifier.rsplit("/", 1)[-1]
        if "." in filename:
            if filename in self.filename_to_idx:
                resolved = self.filename_to_idx[filename]
                if resolved is None:
                    raise ValueError(
                        f"Ambiguous mesh identifier '{shape_identifier}'. "
                        "Please provide its full relative file path."
                    )
                return resolved
        else:
            if filename in self.stem_to_idx:
                resolved = self.stem_to_idx[filename]
                if resolved is None:
                    raise ValueError(
                        f"Ambiguous mesh identifier '{shape_identifier}'. "
                        "Please provide its full relative file path."
                    )
                return resolved

        raise ValueError(f"Mesh not found in CSV database: {shape_identifier}")

    def resolve_many(
        self, shapes_to_compare: Sequence[str]
    ) -> tuple[np.ndarray, list[str]]:
        cache_key = tuple(shapes_to_compare)
        cached = self.batch_cache.get(cache_key)
        if cached is not None:
            return cached

        indices = np.empty(len(shapes_to_compare), dtype=np.intp)
        keys: list[str] = []
        raw_ids = self.raw_identifiers

        for i, shape in enumerate(shapes_to_compare):
            idx = self.resolve_index(shape)
            indices[i] = idx
            keys.append(raw_ids[idx])

        result = (indices, keys)
        if len(self.batch_cache) < 16:
            self.batch_cache[cache_key] = result
        return result


# Per-DataFrame caches keyed by id(df)
_LOOKUP_CACHE: dict[tuple[int, str], tuple[pd.DataFrame, _DataFrameShapeLookup]] = {}
_HISTOGRAM_GROUPS_CACHE: dict[int, tuple[pd.DataFrame, dict[str, list[str]]]] = {}
_FEATURE_MATRIX_CACHE: dict[
    tuple[int, str, bool, str],
    tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray],
] = {}
_EMD_CDF_CACHE: dict[
    tuple[int, str, bool, str],
    tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
] = {}


def _ensure_dataframe(csv_database: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(csv_database, pd.DataFrame):
        return csv_database
    csv_path = Path(csv_database)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV database not found: {csv_database}")
    return pd.read_csv(csv_path)


def _get_shape_lookup(
    df: pd.DataFrame,
    identifier_column: str = IDENTIFIER_COLUMN,
) -> _DataFrameShapeLookup:
    cache_key = (id(df), identifier_column)
    cached = _LOOKUP_CACHE.get(cache_key)
    if cached is not None and cached[0] is df:
        return cached[1]

    if identifier_column not in df.columns:
        raise ValueError(
            f"Required identifier column '{identifier_column}' "
            "not found in CSV database."
        )

    if df[identifier_column].duplicated().any():
        duplicates = df.loc[
            df[identifier_column].duplicated(keep=False),
            identifier_column,
        ].tolist()
        raise ValueError(
            f"Duplicate entries found in '{identifier_column}': {duplicates}"
        )

    raw_identifiers = df[identifier_column].astype(str).tolist()
    exact_to_idx: dict[str, int] = {}
    filename_to_idx: dict[str, int | None] = {}
    stem_to_idx: dict[str, int | None] = {}

    for idx, raw_id in enumerate(raw_identifiers):
        exact_to_idx[raw_id] = idx
        norm_id = raw_id.replace("\\", "/").strip("/")
        exact_to_idx[norm_id] = idx

        filename = norm_id.rsplit("/", 1)[-1]
        if filename in filename_to_idx and filename_to_idx[filename] != idx:
            filename_to_idx[filename] = None
        else:
            filename_to_idx[filename] = idx

        stem = Path(filename).stem
        if stem in stem_to_idx and stem_to_idx[stem] != idx:
            stem_to_idx[stem] = None
        else:
            stem_to_idx[stem] = idx

    lookup = _DataFrameShapeLookup(
        raw_identifiers=raw_identifiers,
        exact_to_idx=exact_to_idx,
        filename_to_idx=filename_to_idx,
        stem_to_idx=stem_to_idx,
    )
    _LOOKUP_CACHE[cache_key] = (df, lookup)
    return lookup


def _get_histogram_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Automatically identify histogram families and sort their bins."""
    cached = _HISTOGRAM_GROUPS_CACHE.get(id(df))
    if cached is not None and cached[0] is df:
        return cached[1]

    groups: dict[str, list[tuple[int, str]]] = {}

    for column in df.columns:
        match = re.fullmatch(r"(.+)_([0-9]+)", str(column))

        if match is None:
            continue

        histogram_name = match.group(1)
        bin_index = int(match.group(2))

        groups.setdefault(histogram_name, []).append((bin_index, column))

    result = {}

    for histogram_name, bins in groups.items():
        bins.sort(key=lambda item: item[0])
        indices = [index for index, _ in bins]

        # Ensure bins are numbered consecutively from zero.
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Histogram '{histogram_name}' has missing or "
                "non-consecutive bin indices."
            )

        result[histogram_name] = [column for _, column in bins]

    if not result:
        raise ValueError(
            "No histogram columns found. "
            "Expected names such as A3_0, A3_1, etc."
        )

    _HISTOGRAM_GROUPS_CACHE[id(df)] = (df, result)
    return result


def _get_feature_columns(
    df: pd.DataFrame,
    descriptor: str,
    histogram: bool,
    identifier_column: str = IDENTIFIER_COLUMN,
) -> list[str]:
    """
    Select the columns used for distance calculation.

    When histogram=False, select a single descriptor column, e.g. 'convexity'
    (falling back to histogram bin columns if the descriptor is stored as a
    histogram family in the DataFrame).

    When histogram=True, select all bins for one histogram family,
    e.g. 'A3_0', 'A3_1', 'A3_2', etc.
    """
    if not histogram:
        if descriptor in df.columns:
            return [descriptor]

        try:
            histogram_groups = _get_histogram_groups(df)
            if descriptor in histogram_groups:
                return histogram_groups[descriptor]
        except ValueError:
            pass

        raise ValueError(
            f"Descriptor '{descriptor}' was not found in the CSV database."
        )

    # Histogram mode: descriptor must identify a histogram family,
    # such as A1, A3, D1, or D2.
    histogram_groups = _get_histogram_groups(df)

    if descriptor not in histogram_groups:
        if descriptor in df.columns:
            return [descriptor]

        available = ", ".join(sorted(histogram_groups))

        raise ValueError(
            f"Histogram '{descriptor}' was not found in the CSV database. "
            f"Available histograms: {available}"
        )

    return histogram_groups[descriptor]


def _get_cached_feature_matrix(
    df: pd.DataFrame,
    descriptor: str,
    histogram: bool,
    identifier_column: str = IDENTIFIER_COLUMN,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract and cache the numeric feature matrix for a descriptor."""
    cache_key = (id(df), descriptor, histogram, identifier_column)
    cached = _FEATURE_MATRIX_CACHE.get(cache_key)
    if cached is not None and cached[0] is df:
        return cached[1], cached[2], cached[3]

    columns = _get_feature_columns(
        df, descriptor, histogram, identifier_column
    )
    sub_df = df[columns]
    matrix = np.empty((len(df), len(columns)), dtype=np.float64)
    for col_idx, col_name in enumerate(columns):
        matrix[:, col_idx] = pd.to_numeric(
            sub_df[col_name], errors="coerce"
        ).to_numpy(dtype=np.float64)

    finite_mask = np.isfinite(matrix).all(axis=1)
    _FEATURE_MATRIX_CACHE[cache_key] = (df, columns, matrix, finite_mask)
    return columns, matrix, finite_mask


def _get_numeric_vector(
    row: pd.Series,
    columns: list[str],
    mesh_identifier: str,
) -> np.ndarray:
    """Extract a validated numeric feature vector from one mesh row."""
    values = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Missing or non-finite feature values for mesh "
            f"'{mesh_identifier}'."
        )

    return values


def _find_shape(
    df: pd.DataFrame,
    shape_identifier: str,
    identifier_column=IDENTIFIER_COLUMN,
) -> pd.Series:
    """Find a mesh by its path, filename, or filename stem."""
    lookup = _get_shape_lookup(df, identifier_column)
    idx = lookup.resolve_index(shape_identifier)
    return df.iloc[idx]


def euclidean_distance(
    df: pd.DataFrame,
    query: str,
    shapes_to_compare: list[str],
    descriptor: str,
    histogram: bool = False,
    identifier_column=IDENTIFIER_COLUMN,
) -> dict[str, float]:
    """Calculate Euclidean distances between meshes."""
    df = _ensure_dataframe(df)

    if not shapes_to_compare:
        raise ValueError("At least one comparison mesh must be provided.")

    lookup = _get_shape_lookup(df, identifier_column)
    _, matrix, finite_mask = _get_cached_feature_matrix(
        df, descriptor, histogram, identifier_column
    )

    query_idx = lookup.resolve_index(query)
    if not finite_mask[query_idx]:
        raise ValueError(
            f"Missing or non-finite feature values for mesh '{query}'."
        )

    comp_indices, comp_keys = lookup.resolve_many(shapes_to_compare)
    if not finite_mask[comp_indices].all():
        bad_pos = int(np.flatnonzero(~finite_mask[comp_indices])[0])
        raise ValueError(
            f"Missing or non-finite feature values for mesh "
            f"'{shapes_to_compare[bad_pos]}'."
        )

    query_vector = matrix[query_idx]
    comp_matrix = matrix[comp_indices]

    if comp_matrix.shape[1] == 1:
        dists = np.abs(comp_matrix[:, 0] - query_vector[0])
    else:
        dists = np.linalg.norm(comp_matrix - query_vector, axis=1)

    return dict(zip(comp_keys, dists.tolist()))


def cosine_distance(
    df: pd.DataFrame,
    query: str,
    shapes_to_compare: list[str],
    descriptor: str,
    identifier_column=IDENTIFIER_COLUMN,
    histogram: bool = False,
) -> dict[str, float]:
    """Calculate cosine distances between meshes."""
    df = _ensure_dataframe(df)

    if not shapes_to_compare:
        raise ValueError("At least one comparison mesh must be provided.")

    lookup = _get_shape_lookup(df, identifier_column)
    _, matrix, finite_mask = _get_cached_feature_matrix(
        df, descriptor, histogram, identifier_column
    )

    query_idx = lookup.resolve_index(query)
    if not finite_mask[query_idx]:
        raise ValueError(
            f"Missing or non-finite feature values for mesh '{query}'."
        )

    comp_indices, comp_keys = lookup.resolve_many(shapes_to_compare)
    if not finite_mask[comp_indices].all():
        bad_pos = int(np.flatnonzero(~finite_mask[comp_indices])[0])
        raise ValueError(
            f"Missing or non-finite feature values for mesh "
            f"'{shapes_to_compare[bad_pos]}'."
        )

    query_vector = matrix[query_idx]
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0:
        raise ValueError(
            f"Cosine distance is undefined for zero vector: '{query}'."
        )

    comp_matrix = matrix[comp_indices]
    comp_norms = np.linalg.norm(comp_matrix, axis=1)
    if np.any(comp_norms == 0):
        bad_pos = int(np.flatnonzero(comp_norms == 0)[0])
        raise ValueError(
            f"Cosine distance is undefined for zero vector: "
            f"'{shapes_to_compare[bad_pos]}'."
        )

    dots = comp_matrix @ query_vector
    dists = 1.0 - (dots / (comp_norms * query_norm))
    return dict(zip(comp_keys, dists.tolist()))


def earth_movers_distance(
    df: pd.DataFrame,
    query: str,
    shapes_to_compare: list[str],
    descriptor: str,
    identifier_column: str = IDENTIFIER_COLUMN,
    histogram: bool = False,
) -> dict[str, float]:
    """
    Calculate Earth Mover's Distance (EMD) between a query mesh and
    comparison meshes.
    """
    df = _ensure_dataframe(df)

    if not shapes_to_compare:
        raise ValueError("No comparison meshes were provided.")

    lookup = _get_shape_lookup(df, identifier_column)
    _, matrix, finite_mask = _get_cached_feature_matrix(
        df, descriptor, histogram, identifier_column
    )

    query_idx = lookup.resolve_index(query)
    if not finite_mask[query_idx]:
        raise ValueError(
            f"Missing or non-finite feature values for mesh '{query}'."
        )

    query_values = matrix[query_idx]
    if (query_values < 0).any():
        raise ValueError(
            f"Query mesh '{query}' has negative values. "
            "EMD requires non-negative histogram weights."
        )

    comp_indices, comp_keys = lookup.resolve_many(shapes_to_compare)
    if not finite_mask[comp_indices].all():
        bad_pos = int(np.flatnonzero(~finite_mask[comp_indices])[0])
        raise ValueError(
            f"Missing or non-finite feature values for mesh "
            f"'{shapes_to_compare[bad_pos]}'."
        )

    # Check or build cached CDF matrix for EMD
    emd_cache_key = (id(df), descriptor, histogram, identifier_column)
    cached_emd = _EMD_CDF_CACHE.get(emd_cache_key)
    if cached_emd is not None and cached_emd[0] is df:
        _, non_neg_mask, row_sums, cdf_matrix = cached_emd
    else:
        non_neg_mask = (matrix >= 0).all(axis=1)
        row_sums = matrix.sum(axis=1)
        safe_sums = np.where(row_sums == 0, 1.0, row_sums)[:, np.newaxis]
        if matrix.shape[1] > 1:
            cdf_matrix = np.cumsum(matrix / safe_sums, axis=1)[:, :-1]
        else:
            cdf_matrix = np.zeros((len(df), 1), dtype=np.float64)
        _EMD_CDF_CACHE[emd_cache_key] = (
            df,
            non_neg_mask,
            row_sums,
            cdf_matrix,
        )

    if not non_neg_mask[comp_indices].all():
        bad_pos = int(np.flatnonzero(~non_neg_mask[comp_indices])[0])
        raise ValueError(
            f"Comparison mesh '{shapes_to_compare[bad_pos]}' has negative values. "
            "EMD requires non-negative histogram weights."
        )

    query_sum = float(row_sums[query_idx])
    comp_sums = row_sums[comp_indices]

    if query_sum == 0:
        if np.any(comp_sums != 0):
            bad_pos = int(np.flatnonzero(comp_sums != 0)[0])
            raise ValueError(
                f"Cannot compare a zero distribution with a non-zero "
                f"distribution: '{query}' vs '{shapes_to_compare[bad_pos]}'."
            )
        return dict.fromkeys(comp_keys, 0.0)

    if np.any(comp_sums == 0):
        bad_pos = int(np.flatnonzero(comp_sums == 0)[0])
        raise ValueError(
            f"Cannot compare a zero distribution with a non-zero "
            f"distribution: '{query}' vs '{shapes_to_compare[bad_pos]}'."
        )

    if matrix.shape[1] == 1:
        return dict.fromkeys(comp_keys, 0.0)

    query_cdf = cdf_matrix[query_idx]
    comp_cdfs = cdf_matrix[comp_indices]
    dists = np.sum(np.abs(comp_cdfs - query_cdf), axis=1)

    return dict(zip(comp_keys, dists.tolist()))


DISTANCE_FUNCTIONS: dict[str, Callable] = {
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
    "emd": earth_movers_distance,
}


def main() -> None:
    """Parse arguments and calculate descriptor distances."""
    parser = argparse.ArgumentParser(
        description=(
            "Distance functions for comparing descriptor values between shapes"
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        "--csv_descriptors",
        dest="input",
        required=True,
        help="Path to the CSV containing mesh descriptors.",
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
        "--descriptor",
        type=str,
        required=True,
        help=("Descriptor to use, e.g. compactness or convexity."),
    )

    parser.add_argument(
        "--distance_function",
        type=str,
        required=True,
        help=(
            "The distance function to be used, functions available: "
            f"{DISTANCE_FUNCTIONS.keys()}"
        ),
    )

    parser.add_argument(
        "--histogram",
        action="store_true",
        help=(
            "Compare all automatically detected histogram bins "
            "instead of a single descriptor."
        ),
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualise query with VEDO"
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        distances = DISTANCE_FUNCTIONS[args.distance_function](
            df=df,
            query=args.query,
            shapes_to_compare=args.shapes,
            descriptor=args.descriptor,
            histogram=args.histogram,
        )

        print(f"\nQuery mesh: {args.query}")
        print(f"Descriptor: {args.descriptor}")
        print("\nDistances (ascending):")

        sorted_distances = sorted(
            distances.items(),
            key=lambda item: item[1],
        )

        for mesh_path, distance in sorted_distances:
            print(f"{distance:.6f}  {mesh_path}")

        if args.visualize:
            visualize_mesh_distances(
                args.query,
                args.shapes,
                descriptor=args.descriptor,
                distance_measure=args.distance_function,
                distances=distances,
            )

    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
