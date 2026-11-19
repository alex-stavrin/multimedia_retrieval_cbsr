import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---
# Matplotlib configuration
plt.rcParams.update(
    {
        # General font settings
        "font.family": "sans-serif",
        "font.size": 20,
        # Titles
        "axes.titlesize": 15,
        "axes.titleweight": "normal",
        "figure.titlesize": 17,
        "figure.titleweight": "normal",
        # Axis labels
        "axes.labelsize": 15,
        "axes.labelweight": "normal",
        # Tick labels
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        # Legend
        "legend.fontsize": 14,
        "legend.title_fontsize": 15,
        # Lines and markers
        "lines.linewidth": 1.5,
        "lines.markersize": 8,
        # Histogram patches
        "patch.linewidth": 0.8,
        # Grid
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.4,
        # Figure defaults
        "figure.dpi": 100,
        "savefig.dpi": 200,
        # Embed TrueType fonts in PDF output
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---
# Plot style and layout constants

# Colors
RAW_COLOR = "tab:blue"
PROCESSED_COLOR = "tab:orange"
HISTOGRAM_EDGE_COLOR = "black"

# Range plot
RANGE_LINEWIDTH = 6
RANGE_LINESTYLE_RAW = "solid"
RANGE_LINESTYLE_PROCESSED = "solid"
RANGE_ENDPOINT_MARKER = "|"
RANGE_PADDING_FRACTION = 0.08
ZERO_RANGE_PADDING_FRACTION = 0.05
ZERO_RANGE_MIN_PADDING = 1.0

# Range plot layout
RANGE_FIGURE_WIDTH = 13
RANGE_FIGURE_MIN_HEIGHT = 4.5
RANGE_ROW_HEIGHT = 0.85

# Histogram layout
HISTOGRAM_FIGURE_WIDTH = 12
HISTOGRAM_FIGURE_HEIGHT = 4.5
DEFAULT_HISTOGRAM_BINS = 50

# Figure layout
FIGURE_LAYOUT = "tight"
FIGURE_BBOX = "tight"

# Data handling
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
        raise ValueError("The CSV files have no shared numeric descriptor columns.")

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
                "raw_mean": raw_values.mean(),
                "processed_min": processed_values.min(),
                "processed_max": processed_values.max(),
                "processed_mean": processed_values.mean(),
            }
        )

    if not rows:
        raise ValueError("No shared descriptors contain usable numeric values.")

    return pd.DataFrame(rows)


def plot_ranges(
    summary: pd.DataFrame,
    output_path: str | None = None,
):
    """Plot raw and processed ranges with an independent x-axis
    per descriptor."""
    descriptors = summary["descriptor"].tolist()
    fig_height = max(
        RANGE_FIGURE_MIN_HEIGHT,
        RANGE_ROW_HEIGHT * len(descriptors) * 1.3,
    )

    fig, axes = plt.subplots(
        nrows=len(descriptors),
        ncols=2,
        figsize=(RANGE_FIGURE_WIDTH, fig_height),
        squeeze=False,
    )

    for position, (_, row) in enumerate(summary.iterrows()):
        raw_ax = axes[position, 0]
        processed_ax = axes[position, 1]

        raw_min = float(row["raw_min"])
        raw_max = float(row["raw_max"])
        processed_min = float(row["processed_min"])
        processed_max = float(row["processed_max"])

        print(
            f"Descriptor: {descriptors[position]}\n"
            f"Raw: min: {raw_min}, max: {raw_max}, mean: {row["raw_mean"]}"
            f"Processed: min: {processed_min}, max: {processed_max}, mean: {row["processed_mean"]}"
        )

        raw_ax.hlines(
            0,
            raw_min,
            raw_max,
            color=RAW_COLOR,
            linewidth=RANGE_LINEWIDTH,
            linestyles=RANGE_LINESTYLE_RAW,
        )
        raw_ax.plot(
            [raw_min, raw_max],
            [0, 0],
            RANGE_ENDPOINT_MARKER,
            color=RAW_COLOR,
        )

        processed_ax.hlines(
            0,
            processed_min,
            processed_max,
            color=PROCESSED_COLOR,
            linewidth=RANGE_LINEWIDTH,
            linestyles=RANGE_LINESTYLE_PROCESSED,
        )
        processed_ax.plot(
            [processed_min, processed_max],
            [0, 0],
            RANGE_ENDPOINT_MARKER,
            color=PROCESSED_COLOR,
        )

        # Add a small margin around each descriptor's own range.
        raw_padding = (raw_max - raw_min) * RANGE_PADDING_FRACTION
        if raw_padding == 0:
            raw_padding = (
                abs(raw_min) * ZERO_RANGE_PADDING_FRACTION or ZERO_RANGE_MIN_PADDING
            )

        processed_padding = (processed_max - processed_min) * RANGE_PADDING_FRACTION
        if processed_padding == 0:
            processed_padding = (
                abs(processed_min) * ZERO_RANGE_PADDING_FRACTION
                or ZERO_RANGE_MIN_PADDING
            )

        raw_ax.set_xlim(
            raw_min - raw_padding,
            raw_max + raw_padding,
        )
        processed_ax.set_xlim(
            processed_min - processed_padding,
            processed_max + processed_padding,
        )

        raw_ax.set_yticks([])
        processed_ax.set_yticks([])

        raw_ax.set_ylabel(
            descriptors[position],
            rotation=0,
            ha="right",
        )

        raw_ax.grid(axis="x")
        processed_ax.grid(axis="x")

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
        fig.savefig(
            output,
            bbox_inches=FIGURE_BBOX,
            format="pdf",
        )
        print(f"Saved plot to {output}")
    else:
        plt.show()

    return fig


def plot_descriptor_histograms(
    raw_csv: str,
    processed_csv: str,
    descriptor: str,
    output_path: str | None = None,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    verbose: bool = False,
):
    """Plot raw and processed distributions within their 5th–95th
    percentile ranges."""
    raw_df = pd.read_csv(raw_csv)
    processed_df = pd.read_csv(processed_csv)

    for label, df in (("raw", raw_df), ("processed", processed_df)):
        if descriptor not in df.columns:
            raise ValueError(f"Descriptor '{descriptor}' not found in {label} CSV.")

    raw_values = pd.to_numeric(raw_df[descriptor], errors="coerce").dropna()

    processed_values = pd.to_numeric(processed_df[descriptor], errors="coerce").dropna()

    if raw_values.empty or processed_values.empty:
        raise ValueError(
            f"Descriptor '{descriptor}' has no usable numeric values "
            "in one or both CSV files."
        )

    # Calculate the 5th and 95th percentiles independently.
    raw_lower, raw_upper = raw_values.quantile([0.05, 0.95])
    processed_lower, processed_upper = processed_values.quantile([0.05, 0.95])

    # Retain only observations within the percentile ranges.
    raw_plot_values = raw_values[raw_values.between(raw_lower, raw_upper)]

    processed_plot_values = processed_values[
        processed_values.between(processed_lower, processed_upper)
    ]

    if verbose:
        print(f"\nDescriptor: {descriptor}")
        print("Raw:")
        print(raw_values.describe())
        print(f"Displayed range: [{raw_lower:.4f}, {raw_upper:.4f}]")
        print(f"Displayed observations: {len(raw_plot_values)} " f"/ {len(raw_values)}")

        print("\nProcessed:")
        print(processed_values.describe())
        print(f"Displayed range: " f"[{processed_lower:.4f}, {processed_upper:.4f}]")
        print(
            f"Displayed observations: {len(processed_plot_values)} "
            f"/ {len(processed_values)}"
        )

    fig, (raw_ax, processed_ax) = plt.subplots(
        ncols=2,
        figsize=(
            HISTOGRAM_FIGURE_WIDTH,
            HISTOGRAM_FIGURE_HEIGHT,
        ),
    )

    raw_ax.hist(
        raw_plot_values,
        bins=bins,
        color=RAW_COLOR,
        edgecolor=HISTOGRAM_EDGE_COLOR,
    )
    raw_ax.set_title(f"Raw {descriptor}")
    raw_ax.set_xlabel(descriptor)
    raw_ax.set_ylabel("Frequency")
    raw_ax.grid(axis="y")

    processed_ax.hist(
        processed_plot_values,
        bins=bins,
        color=PROCESSED_COLOR,
        edgecolor=HISTOGRAM_EDGE_COLOR,
    )
    processed_ax.set_title(f"Processed {descriptor}")
    processed_ax.set_xlabel(descriptor)
    processed_ax.set_ylabel("Frequency")
    processed_ax.grid(axis="y")

    processed_ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Zero (mean)",
    )
    processed_ax.legend()

    fig.suptitle(
        f"Distribution of {descriptor}: Raw vs. Processed " "(5th–95th percentile)"
    )
    fig.tight_layout()

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            bbox_inches=FIGURE_BBOX,
            format="pdf",
        )
        print(f"Saved plot to {output}")
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
            "descriptor, e.g. --descriptor eccentricity."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_HISTOGRAM_BINS,
        help=(f"Number of histogram bins " f"(default: {DEFAULT_HISTOGRAM_BINS})."),
    )

    args = parser.parse_args()

    if args.output is None:
        if args.descriptor:
            args.output = f"descriptor_{args.descriptor}_range_comparison.pdf"
        else:
            args.output = "descriptor_all_range_comparison.pdf"

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
