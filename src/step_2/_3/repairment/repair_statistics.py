"""Plot mesh-closure statistics before and after the repair attempt."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd


def parse_bool_column(series: pd.Series, column_name: str) -> pd.Series:
    """Parse boolean values while allowing empty values for failed repairs."""
    normalized = series.astype("string").str.strip().str.lower()

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    converted = normalized.map(mapping)

    # Empty values are allowed because failed repairs may not have
    # produced closure statistics.
    missing = normalized.isna() | normalized.isin(["", "nan", "none", "<na>"])

    invalid = converted.isna() & ~missing

    if invalid.any():
        examples = series[invalid].head(5).tolist()
        raise ValueError(
            f"Column '{column_name}' contains unrecognized boolean "
            f"values, e.g. {examples}"
        )

    return converted.astype("boolean")


def load_statistics(csv_path: Path) -> pd.DataFrame:
    """Read the CSV and identify successful and failed repair attempts."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    data = pd.read_csv(csv_path)

    required_columns = {"closed_raw", "closed_after"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        available = ", ".join(data.columns)

        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Available columns: {available}"
        )

    data["closed_raw"] = parse_bool_column(data["closed_raw"], "closed_raw")
    data["closed_after"] = parse_bool_column(data["closed_after"], "closed_after")

    # Explicitly identify processing errors from the report.
    if "reason" in data.columns:
        reason = data["reason"].fillna("").astype(str).str.strip()
        processing_error = reason.str.startswith(("error:", "worker error:"), na=False)
    else:
        processing_error = pd.Series(False, index=data.index)

    # Missing closure values also indicate an unsuccessful processing
    # attempt, even if no error message was recorded.
    missing_status = data["closed_raw"].isna() | data["closed_after"].isna()

    data["processing_error"] = processing_error | missing_status

    return data


def calculate_closure_counts(
    data: pd.DataFrame,
) -> tuple[list[str], list[int], list[float], int]:
    """Calculate counts and percentages for all repair outcomes."""
    failed = data["processing_error"]
    successful = ~failed

    closed_raw = data["closed_raw"].fillna(False).astype(bool)
    closed_after = data["closed_after"].fillna(False).astype(bool)

    # These categories are mutually exclusive.
    already_closed = int((successful & closed_raw & closed_after).sum())
    closed_by_repair = int((successful & ~closed_raw & closed_after).sum())
    still_open = int((successful & ~closed_after).sum())
    errors = int(failed.sum())

    labels = [
        "Already closed\nand stayed closed",
        "Closed by\nrepair attempt",
        "Still open\nafter repair",
        "Repair\nfailed",
    ]

    counts = [
        already_closed,
        closed_by_repair,
        still_open,
        errors,
    ]

    total = len(data)

    percentages = [count / total * 100 if total else 0.0 for count in counts]

    return labels, counts, percentages, total


def create_plot(
    labels: list[str],
    counts: list[int],
    percentages: list[float],
    total: int,
) -> Figure:
    """Create a bar chart showing closure outcomes and processing errors."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, counts)

    ax.set_title("Mesh closure before and after repair")
    ax.set_ylabel("Number of shapes")
    ax.set_ylim(0, max(counts, default=0) * 1.18 + 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, count, percentage in zip(bars, counts, percentages):
        ax.annotate(
            f"{count:,}\n({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    fig.text(
        0.99,
        0.01,
        f"Total shapes: {total:,}",
        ha="right",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def print_statistics(
    labels: list[str],
    counts: list[int],
    percentages: list[float],
    total: int,
    output_path: Path,
) -> None:
    """Print closure statistics and the output path."""
    print(f"Total shapes: {total:,}")

    for label, count, percentage in zip(labels, counts, percentages):
        print(f"{label.replace(chr(10), ' ')}: " f"{count:,} ({percentage:.1f}%)")

    print(f"Plot saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Plot how many meshes were closed before and after repair.")
    )

    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Path to the repair-statistics CSV",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: next to the CSV)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively as well as saving it",
    )

    return parser.parse_args()


def main() -> None:
    """Load data, calculate statistics, and save the plot."""
    args = parse_arguments()

    try:
        data = load_statistics(args.csv_path)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    labels, counts, percentages, total = calculate_closure_counts(data)
    fig = create_plot(labels, counts, percentages, total)

    output_path = args.output or args.csv_path.with_name(
        f"{args.csv_path.stem}_closure_statistics.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Match the actual image format to the output extension.
    image_format = output_path.suffix.lstrip(".").lower() or "png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        format=image_format,
    )

    print_statistics(labels, counts, percentages, total, output_path)

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
