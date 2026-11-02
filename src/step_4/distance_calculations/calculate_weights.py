"""Calculate descriptor weights from pairwise distance data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.step_4.distance_calculations.calculate_total_distance import (
    DISTANCE_CONFIG_MIXED,
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
        batch_squared_deviations = float(
            np.square(values - batch_mean).sum()
        )

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


def _add_inverse_std_weights(stats: pd.DataFrame) -> pd.DataFrame:
    """Assign global weights proportional to inverse standard deviation."""
    stats = stats.copy()
    stats["weight"] = 0.0
    valid = stats["std"].notna() & np.isfinite(stats["std"])
    valid &= stats["std"] > 0

    if valid.any():
        inverse_std = 1.0 / stats.loc[valid, "std"]
        stats.loc[valid, "weight"] = inverse_std / inverse_std.sum()
    return stats


def _output_csv_path(output_path: str | Path) -> Path:
    """Return an output path with a CSV extension."""
    path = Path(output_path)
    return path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")


def calculate_weights(
    csv_path: str | Path,
    output_path: str | Path,
    distance_config: dict[str, DistanceConfig] = DISTANCE_CONFIG_MIXED,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Calculate descriptor statistics and weights from a pairwise CSV.

    Weights are inversely proportional to each descriptor's pairwise-distance
    standard deviation, then normalized globally to sum to one. The mean is
    saved as ``pairwise_mean`` for reference; it is not used as a distance
    scale by the query calculator.
    """
    input_path = Path(csv_path)
    if not input_path.is_file():
        raise ValueError(f"CSV path '{input_path}' does not exist.")
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")

    expected_pairs = _expected_descriptor_pairs(distance_config)
    statistics = {
        pair: RunningStatistics() for pair in expected_pairs
    }
    rows_processed = 0
    reader = pd.read_csv(
        input_path,
        usecols=PAIRWISE_COLUMNS,
        chunksize=chunksize,
    )

    for chunk in tqdm(reader, desc="Reading pairwise distances", unit="chunk"):
        rows_processed += len(chunk)
        _update_statistics_from_chunk(chunk, expected_pairs, statistics)

    stats = _add_inverse_std_weights(_build_statistics_frame(statistics))
    output_file = _output_csv_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_file, index=False)

    print(f"Processed {rows_processed:,} CSV rows.")
    print(f"Saved statistics and weights to: {output_file}")
    print(stats.to_string(index=False))
    return stats


def main() -> None:
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

    calculate_weights(
        csv_path=args.csv_path,
        output_path=args.output,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
