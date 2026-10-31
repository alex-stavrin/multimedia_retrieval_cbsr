import argparse
from typing import Callable

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


def _get_histogram_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Automatically identify histogram families and sort their bins."""
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

    return result


def _get_feature_columns(
    df: pd.DataFrame,
    descriptor: str,
    histogram: bool,
    identifier_column: str = IDENTIFIER_COLUMN,
) -> list[str]:
    """
    Select the columns used for distance calculation.

    When histogram=False, select a single descriptor column, e.g. 'convexity'.

    When histogram=True, select all bins for one histogram family,
    e.g. 'A3_0', 'A3_1', 'A3_2', etc.
    """
    if not histogram:
        if descriptor not in df.columns:
            raise ValueError(
                f"Descriptor '{descriptor}' was not found in the CSV database."
            )

        return [descriptor]

    # Histogram mode: descriptor must identify a histogram family,
    # such as A1, A3, D1, or D2.
    histogram_groups = _get_histogram_groups(df)

    if descriptor not in histogram_groups:
        available = ", ".join(sorted(histogram_groups))

        raise ValueError(
            f"Histogram '{descriptor}' was not found in the CSV database. "
            f"Available histograms: {available}"
        )

    return histogram_groups[descriptor]


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
    identifier = shape_identifier.replace("\\", "/").strip("/")
    identifier_path = Path(identifier)

    # Attempt an exact match against file_path.
    normalized_paths = (
        df[identifier_column].astype(str).str.replace("\\", "/", regex=False)
    )
    matches = df[normalized_paths == identifier]

    if len(matches) == 1:
        return matches.iloc[0]

    if len(matches) > 1:
        raise ValueError(f"Multiple meshes found for: {shape_identifier}")

    # Fall back to matching the filename or filename stem.
    filenames = []

    for path in df[identifier_column].astype(str):
        normalized_path = path.replace("\\", "/")
        filename = normalized_path.split("/")[-1]
        filenames.append(filename)

    if identifier_path.suffix:
        matches = df[
            pd.Series(filenames, index=df.index) == identifier_path.name
        ]
    else:
        stems = []

        for filename in filenames:
            stem = Path(filename).stem
            stems.append(stem)

        matches = df[pd.Series(stems, index=df.index) == identifier]

    if matches.empty:
        raise ValueError(f"Mesh not found in CSV database: {shape_identifier}")

    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous mesh identifier '{shape_identifier}'. "
            "Please provide its full relative file path."
        )

    return matches.iloc[0]


def euclidean_distance(
    df: pd.DataFrame,
    query: str,
    shapes_to_compare: list[str],
    descriptor: str,
    histogram: bool = False,
    identifier_column=IDENTIFIER_COLUMN,
) -> dict[str, float]:
    """Calculate Euclidean distances between meshes."""
    if identifier_column not in df.columns:
        raise ValueError(
            f"Required identifier column '{identifier_column}' "
            "not found in CSV database."
        )

    if not shapes_to_compare:
        raise ValueError("At least one comparison mesh must be provided.")

    if df[identifier_column].duplicated().any():
        raise ValueError(f"Duplicate entries found in '{identifier_column}'.")

    columns = _get_feature_columns(
        df, descriptor, histogram, identifier_column
    )

    query_shape = _find_shape(df, query, identifier_column)
    query_vector = _get_numeric_vector(query_shape, columns, query)

    distances = {}

    for shape in shapes_to_compare:
        comparison_shape = _find_shape(df, shape, identifier_column)
        comparison_vector = _get_numeric_vector(
            comparison_shape, columns, shape
        )

        distance = float(euclidean(query_vector, comparison_vector))
        mesh_path = str(comparison_shape[identifier_column])
        distances[mesh_path] = distance

    return distances


def cosine_distance(
    df: pd.DataFrame,
    query: str,
    shapes_to_compare: list[str],
    descriptor: str,
    identifier_column=IDENTIFIER_COLUMN,
    histogram: bool = False,
) -> dict[str, float]:
    """Calculate cosine distances between meshes."""
    if identifier_column not in df.columns:
        raise ValueError(
            f"Required identifier column '{identifier_column}' "
            "not found in CSV database."
        )

    if not shapes_to_compare:
        raise ValueError("At least one comparison mesh must be provided.")

    if df[identifier_column].duplicated().any():
        raise ValueError(f"Duplicate entries found in '{identifier_column}'.")

    columns = _get_feature_columns(
        df, descriptor, histogram, identifier_column
    )

    query_shape = _find_shape(df, query, identifier_column)
    query_vector = _get_numeric_vector(query_shape, columns, query)

    if np.linalg.norm(query_vector) == 0:
        raise ValueError(
            f"Cosine distance is undefined for zero vector: '{query}'."
        )

    distances = {}

    for shape in shapes_to_compare:
        comparison_shape = _find_shape(df, shape, identifier_column)
        comparison_vector = _get_numeric_vector(
            comparison_shape, columns, shape
        )

        if np.linalg.norm(comparison_vector) == 0:
            raise ValueError(
                f"Cosine distance is undefined for zero vector: '{shape}'."
            )

        distance = float(scipy_cosine(query_vector, comparison_vector))

        mesh_path = str(comparison_shape[identifier_column])
        distances[mesh_path] = distance

    return distances


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
    comparison meshes using scipy.stats.wasserstein_distance.

    When histogram=False:
        Compares a single elemental descriptor, e.g. 'convexity'.
        For scalar values, the result is equivalent to the absolute
        difference between the two values.

    When histogram=True:
        Compares the bins of one histogram, e.g. 'A3'.
        The corresponding CSV columns must be named A3_0, A3_1, etc.

    Returns:
        Dictionary mapping each comparison mesh's identifier to its distance.
    """
    if identifier_column not in df.columns:
        raise ValueError(
            f"Identifier column '{identifier_column}' "
            "was not found in the CSV database."
        )

    if not shapes_to_compare:
        raise ValueError("No comparison meshes were provided.")

    # Ensure identifiers are unique
    if df[identifier_column].duplicated().any():
        duplicates = df.loc[
            df[identifier_column].duplicated(keep=False),
            identifier_column,
        ].tolist()

        raise ValueError(
            f"Duplicate mesh identifiers found in CSV: {duplicates}"
        )

    # Select either one elemental descriptor or one histogram family
    feature_columns = _get_feature_columns(
        df=df,
        descriptor=descriptor,
        histogram=histogram,
        identifier_column=identifier_column,
    )

    # Find the query mesh in the database
    query_row = _find_shape(
        df,
        query,
        identifier_column=identifier_column,
    )

    # Extract query values
    query_values = _get_numeric_vector(
        query_row,
        feature_columns,
        query,
    )

    # EMD requires non-negative histogram weights
    if (query_values < 0).any():
        raise ValueError(
            f"Query mesh '{query}' has negative values. "
            "EMD requires non-negative histogram weights."
        )

    # Histogram bin positions
    bin_positions = np.arange(len(feature_columns))

    distances: dict[str, float] = {}

    for comparison in shapes_to_compare:
        comparison_row = _find_shape(
            df,
            comparison,
            identifier_column=identifier_column,
        )

        comparison_values = _get_numeric_vector(
            comparison_row,
            feature_columns,
            comparison,
        )

        if (comparison_values < 0).any():
            raise ValueError(
                f"Comparison mesh '{comparison}' has negative values. "
                "EMD requires non-negative histogram weights."
            )

        query_sum = query_values.sum()
        comparison_sum = comparison_values.sum()

        # Handle empty distributions explicitly
        if query_sum == 0 and comparison_sum == 0:
            distance = 0.0

        elif query_sum == 0 or comparison_sum == 0:
            raise ValueError(
                f"Cannot compare a zero distribution with a non-zero "
                f"distribution: '{query}' vs '{comparison}'."
            )

        else:
            # For a scalar descriptor, this compares the values as
            # single-point distributions. For histograms, it compares
            # the weights across their bin positions.
            distance = float(
                wasserstein_distance(
                    bin_positions,
                    bin_positions,
                    u_weights=query_values,
                    v_weights=comparison_values,
                )
            )

        distances[str(comparison_row[identifier_column])] = distance

    return distances


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
