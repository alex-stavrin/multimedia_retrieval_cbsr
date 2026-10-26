import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.step_4.distance_calculations.calculate_total_distance import (
    DISTANCE_CONFIG,
    DistanceConfig,
)


def calculate_weights(
    csv_path: str,
    output_path: str,
    distance_config: dict[str, DistanceConfig] = DISTANCE_CONFIG,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Calculate descriptor statistics and effective weights from a large
    pairwise-distance CSV without loading the entire file into memory.

    Expected CSV columns:
        mesh_a, mesh_b, distance_function, descriptor, distance

    Effective weights are inversely proportional to standard deviation
    and normalized to sum to 1 within each distance function.
    """
    input_path = Path(csv_path)

    if not input_path.is_file():
        raise ValueError(f"CSV path `{csv_path}` does not exist.")

    # Precompute the valid combinations from the small configuration.
    expected_pairs = {
        (function_name, descriptor)
        for function_name, config in distance_config.items()
        for descriptor in config.descriptors
    }

    # Running statistics per descriptor/function:
    # count, mean, and M2 (sum of squared deviations).
    running_stats = {
        pair: {"count": 0, "mean": 0.0, "M2": 0.0} for pair in expected_pairs
    }

    columns = ["distance_function", "descriptor", "distance"]
    rows_processed = 0

    for chunk in pd.read_csv(
        input_path,
        usecols=columns,
        chunksize=chunksize,
    ):
        rows_processed += len(chunk)

        chunk["distance"] = pd.to_numeric(chunk["distance"], errors="coerce")
        chunk = chunk.dropna(subset=["distance"])

        # Ignore unexpected descriptor/function combinations.
        chunk = chunk[
            [
                pair in expected_pairs
                for pair in zip(
                    chunk["distance_function"],
                    chunk["descriptor"],
                )
            ]
        ]

        if chunk.empty:
            continue

        # Aggregate each chunk before combining with running statistics.
        for pair, group in chunk.groupby(
            ["distance_function", "descriptor"],
            sort=False,
        ):
            function_name, descriptor = cast(tuple[str, str], pair)
            key = (function_name, descriptor)

            values = group["distance"].to_numpy(dtype=np.float64)
            n = len(values)
            chunk_mean = float(values.mean())
            chunk_M2 = float(((values - chunk_mean) ** 2).sum())

            current = running_stats[key]
            old_n = current["count"]

            if old_n == 0:
                current["count"] = n
                current["mean"] = chunk_mean
                current["M2"] = chunk_M2
            else:
                old_mean = current["mean"]
                total_n = old_n + n
                delta = chunk_mean - old_mean

                current["mean"] = old_mean + delta * n / total_n
                current["M2"] += chunk_M2 + delta**2 * old_n * n / total_n
                current["count"] = total_n

    print(f"Processed {rows_processed:,} CSV rows.")

    records = []

    for (function_name, descriptor), values in running_stats.items():
        count = values["count"]
        mean = values["mean"]
        M2 = values["M2"]

        # Match pandas' default sample standard deviation (ddof=1).
        std = float(np.sqrt(M2 / (count - 1))) if count > 1 else float("nan")

        records.append(
            {
                "distance_function": function_name,
                "descriptor": descriptor,
                "count": count,
                "mean": mean if count else float("nan"),
                "std": std,
                "weight": 0.0,
            }
        )

    stats = pd.DataFrame(records)

    # Calculate inverse-mean weights across all descriptors.
    valid = stats["mean"].notna() & (stats["mean"] > 0)

    stats["weight"] = 0.0

    inverse_mean = 1.0 / stats.loc[valid, "mean"]
    stats.loc[valid, "weight"] = inverse_mean / inverse_mean.sum()

    output_file = Path(output_path)

    if output_file.suffix.lower() != ".csv":
        output_file = output_file.with_suffix(".csv")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_file, index=False)

    print(f"Saved statistics and weights to: {output_file}")
    print(stats.to_string(index=False))

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate descriptor statistics and weights from "
            "a large pairwise-distance CSV."
        )
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the pairwise-distance CSV.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="weights_descriptors.csv",
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
