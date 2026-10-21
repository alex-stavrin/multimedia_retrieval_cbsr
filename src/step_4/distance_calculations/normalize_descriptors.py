import math
import argparse
from pathlib import Path

import pandas as pd

NORMALIZABLE = {
    "compactness": (0.0, 1.0),
    "rectangularity": (0.0, 1.0),
    "diameter": (0.0, math.sqrt(3)),
}

STANDARDIZABLE = [
    "convexity",
    "eccentricity",
    "surface_area",
]


def standardize_descriptors(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Standardize selected columns using z-score scaling."""
    for column in columns:
        std = df[column].std()

        if std == 0 or pd.isna(std):
            df[column] = 0.0
        else:
            df[column] = (df[column] - df[column].mean()) / std

    return df


def normalize_descriptors(
    df: pd.DataFrame,
    columns: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Normalize selected columns to the range [0, 1]."""
    for column_key in columns.keys():
        minimum = columns[column_key][0]
        maximum = columns[column_key][1]
        value_range = maximum - minimum

        if value_range == 0 or pd.isna(value_range):
            df[column_key] = 0.0
        else:
            df[column_key] = (df[column_key] - minimum) / value_range

    return df


def process_descriptors(
    csv_path: str,
    output_path: str,
    normalize_columns: dict[str, tuple[float, float]] | None = None,
    standardize_columns: list[str] | None = None,
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

    # Use defaults independently when a flag was not provided.
    if normalize_columns is None:
        normalize_columns = NORMALIZABLE

    if standardize_columns is None:
        standardize_columns = STANDARDIZABLE

    df = pd.read_csv(input_path)

    selected_columns = set(
        list(normalize_columns.keys()) + standardize_columns
    )
    missing_columns = selected_columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Columns not found in CSV: {missing}")

    df = normalize_descriptors(df, normalize_columns)
    df = standardize_descriptors(df, standardize_columns)

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
        "--normalize",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help=("Columns to normalize. If omitted, uses " f"{NORMALIZABLE}."),
    )

    parser.add_argument(
        "--standardize",
        nargs="+",
        default=None,
        metavar="COLUMN",
        help=(
            "Columns to standardize. If omitted, uses " f"{STANDARDIZABLE}."
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
        normalize_columns=args.normalize,
        standardize_columns=args.standardize,
    )


if __name__ == "__main__":
    main()
