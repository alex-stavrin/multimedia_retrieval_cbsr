"""Calculate descriptor weights from pairwise distance data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.step_4.distance_calculations.calculate_total_distance import (
    DISTANCE_CONFIG_EMD,
    DISTANCE_CONFIG_COSINE,
    DISTANCE_CONFIG_L2,
    DistanceConfig,
)

PAIRWISE_COLUMNS = ["distance_function", "descriptor", "distance"]


@dataclass
class RunningStatistics:
    """Mean and sum of squared deviations, updated in chunks."""

    count: int = 0
    mean: float = 0.0
    squared_deviations: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """Merge a batch of values into these running statistics."""
        batch_count = len(values)
        if batch_count == 0:
            return

        batch_mean = float(values.mean())
        batch_squared_deviations = float(np.square(values - batch_mean).sum())

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.squared_deviations = batch_squared_deviations
            return

        combined_count = self.count + batch_count
        mean_delta = batch_mean - self.mean
        self.squared_deviations += (
            batch_squared_deviations
            + mean_delta**2 * self.count * batch_count / combined_count
        )
        self.mean += mean_delta * batch_count / combined_count
        self.count = combined_count

    @property
    def sample_std(self) -> float:
        """Return the sample standard deviation (ddof=1)."""
        if self.count < 2:
            return float("nan")
        return float(np.sqrt(self.squared_deviations / (self.count - 1)))


def _expected_descriptor_pairs(
    distance_config: dict[str, DistanceConfig],
) -> set[tuple[str, str]]:
    """Return descriptor/function combinations included by the config."""
    return {
        (function_name, descriptor)
        for function_name, config in distance_config.items()
        for descriptor in config.descriptors
    }


def _update_statistics_from_chunk(
    chunk: pd.DataFrame,
    expected_pairs: set[tuple[str, str]],
    statistics: dict[tuple[str, str], RunningStatistics],
) -> None:
    """Accumulate valid pairwise distances from one CSV chunk."""
    chunk["distance"] = pd.to_numeric(chunk["distance"], errors="coerce")
    valid_rows = chunk.dropna(subset=["distance"])
    pair_index = pd.MultiIndex.from_arrays(
        [valid_rows["distance_function"], valid_rows["descriptor"]]
    )
    valid_rows = valid_rows.loc[pair_index.isin(expected_pairs)]

    for pair, group in valid_rows.groupby(
        ["distance_function", "descriptor"], sort=False
    ):
        key = (str(pair[0]), str(pair[1]))
        values = group["distance"].to_numpy(dtype=np.float64)
        statistics[key].update(values)


def _build_statistics_frame(
    statistics: dict[tuple[str, str], RunningStatistics],
) -> pd.DataFrame:
    """Create a sorted table of distance statistics."""
    records = []
    for (function_name, descriptor), values in sorted(statistics.items()):
        records.append(
            {
                "distance_function": function_name,
                "descriptor": descriptor,
                "count": values.count,
                "pairwise_mean": (
                    values.mean if values.count else float("nan")
                ),
                "std": values.sample_std,
            }
        )
    return pd.DataFrame(records)


def _histogram_pairs(
    distance_config: dict[str, DistanceConfig],
) -> set[tuple[str, str]]:
    """Return the configured function/descriptor pairs that are histograms."""
    return {
        (function_name, descriptor)
        for function_name, config in distance_config.items()
        if config.histogram
        for descriptor in config.descriptors
    }


def calculate_weights(stats: pd.DataFrame) -> pd.DataFrame:
    """Calculate equal descriptor weights independently of distance scales."""
    if stats.empty:
        raise ValueError("No descriptor statistics were calculated.")

    number_of_descriptors = len(stats)
    return stats[["distance_function", "descriptor"]].assign(
        weight=1.0 / number_of_descriptors
    )


def calculate_distance_scales(
    stats: pd.DataFrame,
    distance_config: dict[str, DistanceConfig],
) -> pd.DataFrame:
    """Scale histogram-distance spread to the median scalar-distance spread.

    Scalar descriptors already use standardized values, so their distance
    scales remain one. This calculation does not read or modify weights.
    """
    if stats.empty:
        raise ValueError("No descriptor statistics were calculated.")

    histogram_pairs = _histogram_pairs(distance_config)
    scales = stats[["distance_function", "descriptor"]].copy()
    scales["distance_scale"] = 1.0
    if not histogram_pairs:
        return scales

    is_histogram = pd.Series(
        [
            (str(row.distance_function), str(row.descriptor))
            in histogram_pairs
            for row in stats.itertuples(index=False)
        ],
        index=stats.index,
    )
    scalar_stds = pd.to_numeric(
        stats.loc[~is_histogram, "std"], errors="coerce"
    )
    scalar_stds = scalar_stds[np.isfinite(scalar_stds) & (scalar_stds > 0)]
    if scalar_stds.empty:
        raise ValueError(
            "Cannot scale histogram distances: no positive scalar-distance "
            "standard deviations are available."
        )

    reference_std = float(scalar_stds.median())
    for index in stats.index[is_histogram]:
        histogram_std = pd.to_numeric(stats.at[index, "std"], errors="coerce")
        if pd.isna(histogram_std) or not np.isfinite(histogram_std):
            raise ValueError(
                "Cannot scale histogram distance for "
                f"('{stats.at[index, 'distance_function']}', "
                f"'{stats.at[index, 'descriptor']}'): "
                "invalid standard deviation."
            )
        if histogram_std > 0:
            scales.at[index, "distance_scale"] = reference_std / float(
                histogram_std
            )

    return scales


def _output_csv_path(output_path: str | Path) -> Path:
    """Return an output path with a CSV extension."""
    path = Path(output_path)
    return path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")


def _calculate_pairwise_statistics(
    csv_path: str | Path,
    distance_config: dict[str, DistanceConfig] = DISTANCE_CONFIG_EMD,
    chunksize: int = 50_000,
) -> tuple[pd.DataFrame, int]:
    """Calculate pairwise means and standard deviations from a CSV."""
    input_path = Path(csv_path)
    if not input_path.is_file():
        raise ValueError(f"CSV path '{input_path}' does not exist.")
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")

    expected_pairs = _expected_descriptor_pairs(distance_config)
    statistics = {pair: RunningStatistics() for pair in expected_pairs}
    rows_processed = 0
    reader = pd.read_csv(
        input_path,
        usecols=PAIRWISE_COLUMNS,
        chunksize=chunksize,
    )

    for chunk in tqdm(reader, desc="Reading pairwise distances", unit="chunk"):
        rows_processed += len(chunk)
        _update_statistics_from_chunk(chunk, expected_pairs, statistics)

    return _build_statistics_frame(statistics), rows_processed


def main() -> None:
    available_configs = {
        "emd": DISTANCE_CONFIG_EMD,
        "emd-equal": DISTANCE_CONFIG_EMD,
        "l2": DISTANCE_CONFIG_L2,
        "l2-equal": DISTANCE_CONFIG_L2,
        "cosine": DISTANCE_CONFIG_COSINE,
        "cosine-equal": DISTANCE_CONFIG_COSINE,
    }

    parser = argparse.ArgumentParser(
        description=(
            "Calculate descriptor statistics and weights from "
            "a pairwise-distance CSV."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to the pairwise-distance CSV.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="The distance config used to calculate the pairwise differences."
        f"\nConfigs available: {available_configs.keys()}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("weights_descriptors.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Number of CSV rows to process per chunk.",
    )

    args = parser.parse_args()

    distance_config = available_configs[args.config]
    pairwise_stats, rows_processed = _calculate_pairwise_statistics(
        csv_path=args.csv_path,
        chunksize=args.chunksize,
        distance_config=distance_config,
    )
    weights = calculate_weights(pairwise_stats)
    distance_scales = calculate_distance_scales(
        pairwise_stats,
        distance_config,
    )
    stats = pairwise_stats.merge(
        weights,
        on=["distance_function", "descriptor"],
    ).merge(
        distance_scales,
        on=["distance_function", "descriptor"],
    )

    output_file = _output_csv_path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_file, index=False)
    print(f"Processed {rows_processed:,} CSV rows.")
    print(f"Saved statistics and weights to: {output_file}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
