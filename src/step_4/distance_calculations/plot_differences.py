import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METADATA_COLUMNS = {"file_path", "reason", "was_closed"}


def load_descriptors(csv_path: str) -> pd.DataFrame:
    """Load a descriptor CSV and retain numeric descriptor columns."""
    path = Path(csv_path)
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {csv_path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV: {csv_path}")

    df = pd.read_csv(path)
    numeric = df.select_dtypes(include="number")
    return numeric.drop(
        columns=[col for col in METADATA_COLUMNS if col in numeric.columns],
        errors="ignore",
    )


def compare_ranges(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize ranges for numeric columns shared by both datasets."""
    common_columns = [
        column for column in raw_df.columns if column in processed_df.columns
    ]
    if not common_columns:
        raise ValueError(
            "The CSV files have no shared numeric descriptor columns."
        )

    rows = []
    for column in common_columns:
        raw_values = raw_df[column].dropna()
        processed_values = processed_df[column].dropna()

        if raw_values.empty or processed_values.empty:
            continue

        rows.append(
            {
                "descriptor": column,
                "raw_min": raw_values.min(),
                "raw_max": raw_values.max(),
                "processed_min": processed_values.min(),
                "processed_max": processed_values.max(),
            }
        )

    if not rows:
        raise ValueError(
            "No shared descriptors contain usable numeric values."
        )

    return pd.DataFrame(rows)


def plot_ranges(
    summary: pd.DataFrame,
    output_path: str | None = None,
):
    """Plot raw and processed ranges with an independent x-axis
    per descriptor."""
    descriptors = summary["descriptor"].tolist()
    fig_height = max(4.5, 0.85 * len(descriptors))
    fig, axes = plt.subplots(
        nrows=len(descriptors),
        ncols=2,
        figsize=(13, fig_height),
        squeeze=False,
    )

    raw_color = "tab:blue"
    processed_color = "tab:orange"

    for position, (_, row) in enumerate(summary.iterrows()):
        raw_ax = axes[position, 0]
        processed_ax = axes[position, 1]

        raw_min = float(row["raw_min"])
        raw_max = float(row["raw_max"])
        processed_min = float(row["processed_min"])
        processed_max = float(row["processed_max"])

        raw_ax.hlines(
            0,
            raw_min,
            raw_max,
            color=raw_color,
            linewidth=6,
            linestyles="solid",
        )
        raw_ax.plot(
            [raw_min, raw_max],
            [0, 0],
            "|",
            color=raw_color,
            markersize=14,
        )
        processed_ax.hlines(
            0,
            processed_min,
            processed_max,
            color=processed_color,
            linewidth=6,
            linestyles="dashed",
        )
        processed_ax.plot(
            [processed_min, processed_max],
            [0, 0],
            "|",
            color=processed_color,
            markersize=14,
        )

        # Add a small margin around each descriptor's own range.
        raw_padding = (raw_max - raw_min) * 0.08
        if raw_padding == 0:
            raw_padding = abs(raw_min) * 0.05 or 1.0
        processed_padding = (processed_max - processed_min) * 0.08
        if processed_padding == 0:
            processed_padding = abs(processed_min) * 0.05 or 1.0

        raw_ax.set_xlim(raw_min - raw_padding, raw_max + raw_padding)
        processed_ax.set_xlim(
            processed_min - processed_padding,
            processed_max + processed_padding,
        )

        raw_ax.set_yticks([])
        processed_ax.set_yticks([])
        raw_ax.set_ylabel(descriptors[position], rotation=0, ha="right")
        raw_ax.grid(axis="x", linestyle="--", alpha=0.4)
        processed_ax.grid(axis="x", linestyle="--", alpha=0.4)

        if position == 0:
            raw_ax.set_title("Raw ranges")
            processed_ax.set_title("Processed ranges")

    axes[-1, 0].set_xlabel("Raw descriptor value")
    axes[-1, 1].set_xlabel("Processed descriptor value")
    fig.suptitle("Raw vs. Processed Descriptor Ranges")
    fig.tight_layout()

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight", format="pdf")
        print(f"Saved plot to {output}")
    else:
        plt.show()

    return fig


def plot_descriptor_histograms(
    raw_csv: str,
    processed_csv: str,
    descriptor: str,
    output_path: str | None = None,
    bins: int = 50,
):
    """Plot raw and processed distributions for one descriptor side by side."""
    raw_df = pd.read_csv(raw_csv)
    processed_df = pd.read_csv(processed_csv)

    for label, df in (("raw", raw_df), ("processed", processed_df)):
        if descriptor not in df.columns:
            raise ValueError(
                f"Descriptor '{descriptor}' not found in {label} CSV."
            )

    raw_values = pd.to_numeric(raw_df[descriptor], errors="coerce").dropna()
    processed_values = pd.to_numeric(
        processed_df[descriptor], errors="coerce"
    ).dropna()

    if raw_values.empty or processed_values.empty:
        raise ValueError(
            f"Descriptor '{descriptor}' has no usable numeric values "
            "in one or both CSV files."
        )

    fig, (raw_ax, processed_ax) = plt.subplots(
        ncols=2,
        figsize=(12, 4.5),
    )

    raw_ax.hist(raw_values, bins=bins, color="tab:blue", edgecolor="black")
    raw_ax.set_title(f"Raw {descriptor}")
    raw_ax.set_xlabel(descriptor)
    raw_ax.set_ylabel("Frequency")
    raw_ax.grid(axis="y", linestyle="--", alpha=0.35)

    processed_ax.hist(
        processed_values,
        bins=bins,
        color="tab:orange",
        edgecolor="black",
    )
    processed_ax.set_title(f"Processed {descriptor}")
    processed_ax.set_xlabel(descriptor)
    processed_ax.set_ylabel("Frequency")
    processed_ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(f"Distribution of {descriptor}: Raw vs. Processed")
    fig.tight_layout()

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight", format="pdf")
        print(f"Saved histogram to {output}")
    else:
        plt.show()

    return fig


def main() -> None:
    """Parse arguments, compare descriptor ranges, and plot the result."""
    parser = argparse.ArgumentParser(
        description="Plot raw and processed descriptor ranges."
    )
    parser.add_argument(
        "--raw",
        required=True,
        help="Path to the raw descriptor CSV.",
    )
    parser.add_argument(
        "--processed",
        required=True,
        help="Path to the processed descriptor CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output plot path (default: "
        "descriptor_[descriptor]_range_comparison.pdf).",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path to save the min/max comparison as a CSV.",
    )
    parser.add_argument(
        "--descriptor",
        metavar="DESCRIPTOR",
        help=(
            "Plot side-by-side raw and processed histograms for this "
            "descriptor, e.g. --histogram eccentricity."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"descriptor_{args.descriptor}_range_comparison.pdf"

    if args.descriptor:
        if args.bins < 1:
            parser.error("--bins must be at least 1")
        plot_descriptor_histograms(
            raw_csv=args.raw,
            processed_csv=args.processed,
            descriptor=args.descriptor,
            output_path=args.output,
            bins=args.bins,
        )
        return

    raw_df = load_descriptors(args.raw)
    processed_df = load_descriptors(args.processed)
    summary = compare_ranges(raw_df, processed_df)

    plot_ranges(summary, args.output)

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Saved range summary to {summary_path}")


if __name__ == "__main__":
    main()
