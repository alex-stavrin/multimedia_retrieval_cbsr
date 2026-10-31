import argparse
from pathlib import Path

import pandas as pd

META_COLUMNS = ["file_path", "was_closed", "reason"]


def calculate_standardization_statistics(
    df: pd.DataFrame,
    ignorable_columns: list[str] = META_COLUMNS,
) -> dict[str, tuple[float, float]]:
    """Fit the per-column mean and sample standard deviation."""
    statistics: dict[str, tuple[float, float]] = {}

    for column in df.columns:
        if column in ignorable_columns:
            continue

        values = pd.to_numeric(df[column], errors="raise")
        mean = float(values.mean())
        std = float(values.std())
        statistics[column] = (mean, std)

    return statistics


def standardize_descriptors(
    df: pd.DataFrame,
    ignorable_columns: list[str] = META_COLUMNS,
    statistics: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Standardize columns using fitted or input-data z-score statistics."""
    if statistics is None:
        statistics = calculate_standardization_statistics(
            df,
            ignorable_columns,
        )

    for cur_column in df.columns:
        if cur_column in ignorable_columns:
            continue

        if cur_column not in statistics:
            raise ValueError(
                f"No standardization statistics found for '{cur_column}'."
            )

        mean, std = statistics[cur_column]

        if std == 0 or pd.isna(std):
            df[cur_column] = 0.0
        else:
            df[cur_column] = (df[cur_column] - mean) / std

    return df


def process_descriptors(
    csv_path: str,
    output_path: str,
    ignorable_columns: list[str] = META_COLUMNS,
) -> pd.DataFrame:
    """Normalize and standardize selected CSV columns, then save."""
    input_path = Path(csv_path)
    output_file = Path(output_path)

    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {csv_path}")

    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV: {csv_path}")

    if output_file.suffix.lower() != ".csv":
        raise ValueError(f"Output file must be a CSV: {output_path}")

    df = pd.read_csv(input_path)

    df = standardize_descriptors(df, ignorable_columns)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Processed descriptors saved to {output_file}")

    return df


def main() -> None:
    """Parse arguments and process the descriptor CSV."""
    parser = argparse.ArgumentParser(
        description="Normalize and standardize shape descriptors."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input descriptor CSV.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path for the output CSV.",
    )

    parser.add_argument(
        "--ignore",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help=(
            "Columns to ignore for standardization. If omitted, uses "
            f"{META_COLUMNS}."
        ),
    )

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        input_path = Path(args.input)
        output_path = str(
            input_path.with_name(f"{input_path.stem}_processed.csv")
        )

    process_descriptors(
        csv_path=args.input,
        output_path=output_path,
        ignorable_columns=args.ignore,
    )


if __name__ == "__main__":
    main()
