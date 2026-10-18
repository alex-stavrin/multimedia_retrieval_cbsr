import pandas as pd
import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import rc
from pathlib import Path

# Set up global Matplotlib variables
FONT = {"size": 20}
FIGURE_SIZE = (25, 7)
RWIDTH = 0.8
rc("font", **FONT)
rc("xtick", labelsize=7)


def _save_figure(fig: Figure, output: str, column: str) -> None:
    """Save a figure to `output`, or to a default name if none is given.

    The file format is inferred from the extension (.png, .pdf, ...).
    Falls back to PDF if there is no extension.
    """
    path = Path(output) if output else Path(f"histogram_plot_{column}.pdf")
    if path.suffix == "":
        path = path.with_suffix(".pdf")

    fig.savefig(path, bbox_inches="tight")


def _set_ticks_to_ax(ax: Axes, target_ticks, min_val, max_val):
    # Nothing to tick when there are no values (e.g. no outliers).
    if not (np.isfinite(min_val) and np.isfinite(max_val)):
        return ax

    # A practically constant column (e.g. the largest bounding-box side after
    # normalization) has no range to divide into ticks: the spacing would be
    # so small that np.arange cannot build it.
    if max_val - min_val <= 1e-6 * max(abs(min_val), abs(max_val), 1.0):
        middle = (min_val + max_val) / 2
        ax.set_xticks([middle])
        ax.set_xticklabels([f"{middle:.6g}"], fontsize=16)
        return ax

    raw_interval = (max_val - min_val) / target_ticks

    magnitude = 10 ** np.floor(np.log10(raw_interval))
    normalized = raw_interval / magnitude

    if normalized <= 1:
        tick_interval = 1 * magnitude
    elif normalized <= 2:
        tick_interval = 2 * magnitude
    elif normalized <= 5:
        tick_interval = 5 * magnitude
    else:
        tick_interval = 10 * magnitude

    # Nice round ticks
    tick_positions = np.arange(
        np.ceil(min_val / tick_interval) * tick_interval,
        max_val,
        tick_interval,
    )

    # Always include the maximum
    tick_positions = np.insert(tick_positions, 0, min_val)
    tick_positions = np.append(tick_positions, max_val)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [f"{x:g}" for x in tick_positions],
        fontsize=16,
    )

    return ax


def show_histogram_from_csv(
    csv_path: str,
    n_bins: int,
    column: str = "",
    save=False,
    output: str = "",
) -> pd.DataFrame | None:
    """Load a CSV file and display its distribution and detected outliers.

    Returns the outliers DataFrame for numeric columns, otherwise None.
    """
    if not csv_path.endswith(".csv"):
        raise ValueError("Input file must end on .csv")

    df = pd.read_csv(csv_path)
    numeric = pd.to_numeric(df[column], errors="coerce").notnull().all()

    fig, axes = [], []

    if numeric:
        fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
        plot_histogram(
            df,
            column,
            bins=n_bins,
            ylabel="Number of shapes",
            ax=axes[0],
        )
    else:
        fig, axis = plt.subplots(1, 1, figsize=FIGURE_SIZE)

        plot_histogram(
            df,
            column,
            bins=n_bins,
            ylabel="Number of shapes",
            ax=axis,
        )

    # Quit early, do not calculate outliers
    if not numeric:
        if save:
            _save_figure(fig, output, column)
        else:
            plt.show()

        return None

    iqr_strictness = 1.5
    percentile = 0.05

    outliers = find_outliers(
        df,
        column,
        interquartile_strictness=iqr_strictness,
        percentile=percentile,
    )

    plot_histogram(
        outliers,
        column,
        bins=n_bins,
        ylabel="Outliers",
        title=(
            f"Outliers of {column} \n"
            f"(IQR strictness: {iqr_strictness}, "
            f"percentile: {percentile:.0%})"
        ),
        ax=axes[2],
        color="red",
    )

    plot_combined_histogram(
        df,
        outliers,
        column,
        n_bins,
        ax=axes[1],
        iqr_strictness=iqr_strictness,
        percentile=percentile,
    )

    if save:
        _save_figure(fig, output, column)
    else:
        plt.show()

    return outliers


def find_outliers(
    df: pd.DataFrame,
    column: str,
    interquartile_strictness: float = 1.5,
    percentile: float = 0.05,
) -> pd.DataFrame:
    """
    Find outliers using both the IQR and percentile methods.

    A row is considered an outlier if its value falls outside
    either the IQR-based range or the percentile-based range.

    Returns:
        A DataFrame containing the detected outlier rows.
    """
    # Interquartile outliers
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    interquartile = q3 - q1

    lower_iqr = q1 - interquartile_strictness * interquartile
    upper_iqr = q3 + interquartile_strictness * interquartile

    iqr_outliers = df[(df[column] < lower_iqr) | (df[column] > upper_iqr)]

    # Percentile outliers
    lower_percentile = df[column].quantile(percentile)
    upper_percentile = df[column].quantile(1 - percentile)

    percentile_outliers = df[
        (df[column] < lower_percentile) | (df[column] > upper_percentile)
    ]

    # Combine
    outliers = pd.concat([iqr_outliers, percentile_outliers]).drop_duplicates(
        subset="relative_file_path"
    )

    return outliers


def export_sorted_csv(df: pd.DataFrame, column: str, output: str) -> None:
    """Write a DataFrame to a CSV file sorted by the specified column."""
    if not output.endswith(".csv"):
        raise ValueError("Export file must end in .csv")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    # Make sorted property first
    columns_order = [column] + [col for col in df.columns if col != column]
    df = df.loc[:, columns_order]

    # Sort on values1
    df = df.sort_values(by=column, ascending=False)

    df.to_csv(output, index=False)



def plot_histogram(
    df,
    column: str,
    bins: int = 20,
    ylabel: str = "",
    ax=None,
    title="",
    save=False,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot a histogram for numeric data or a class-count bar chart
    for non-numeric data.

    Numeric columns:
        - Plot a histogram.
        - Mark mean and median.

    Non-numeric columns:
        - Plot the number of occurrences per class.
        - Display every class as an x-tick using a small font.
        - Do not calculate or plot mean/median.

    Returns:
        The figure and axes containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    else:
        fig = ax.get_figure()

    kwargs.setdefault("edgecolor", "black")

    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric data: histogram + mean/median
        mean = df[column].mean()
        median = df[column].median()

        ax.hist(df[column].dropna(), bins=bins, rwidth=RWIDTH, **kwargs)

        ax.axvline(
            mean,
            color="magenta",
            linestyle="--",
            linewidth=4,
            label=f"Mean: {mean:.2f}",
        )

        ax.axvline(
            median,
            color="lime",
            linestyle="-.",
            linewidth=4,
            label=f"Median: {median:.2f}",
        )

        _set_ticks_to_ax(ax, 6, df[column].min(), df[column].max())

        ax.set_xlabel(column)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.legend()

    else:
        # Non-numeric data: count occurrences per class
        counts = df[column].value_counts(dropna=False)

        # Convert labels to strings for display
        labels = counts.index.astype(str)

        ax.bar(
            range(len(counts)),
            counts.values,
            **kwargs,
        )

        # Explicitly display every class
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(
            labels,
            fontsize=19,
            rotation=90,
        )

        ax.set_xlabel(column)
        ax.set_ylabel(ylabel or "Count")

    if title == "":
        ax.set_title(f"Distribution of {column},\nTotal: {len(df)}")
    else:
        ax.set_title(title + f",\nTotal: {len(df)}")

    if save:
        fig.savefig(
            fname=f"histogram_plot_{column}.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    return fig, ax


def plot_combined_histogram(
    df: pd.DataFrame,
    outliers: pd.DataFrame,
    column: str,
    bins: int = 20,
    ax=None,
    save=False,
    iqr_strictness: float = 1.5,
    percentile: float = 0.05,
) -> Tuple[Figure, Axes]:
    """Plot the full distribution with detected outliers overlaid."""

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    else:
        fig = ax.get_figure()

    # Make sure the column is numeric
    values = df[column]
    outlier_values = outliers[column]

    if len(values) == 0:
        raise ValueError(f"Column '{column}' contains no numeric values.")

    ax.hist(
        values,
        bins=bins,
        color="steelblue",
        alpha=0.5,
        edgecolor="black",
        label=f"Distribution (n={len(values)})",
        rwidth=RWIDTH,
    )

    ax.hist(
        outlier_values,
        bins=bins,
        color="red",
        alpha=0.75,
        edgecolor="black",
        label=f"Outliers (n={len(outlier_values)})",
        rwidth=RWIDTH,
    )

    # Mean and median of the complete distribution
    mean = values.mean()
    median = values.median()

    ax.axvline(
        mean,
        color="magenta",
        linestyle="--",
        linewidth=4,
        label=f"Mean: {mean:.2f}",
    )

    ax.axvline(
        median,
        color="lime",
        linestyle="-.",
        linewidth=4,
        label=f"Median: {median:.2f}",
    )

    _set_ticks_to_ax(ax, 6, values.min(), values.max())

    ax.set_xlabel(column)
    ax.set_ylabel("Number of shapes")

    ax.set_yscale("log")

    ax.set_title(
        f"Distribution and Outliers of {column}\n"
        f"(IQR strictness: {iqr_strictness}, "
        f"percentile: {percentile:.0%})\n"
        f"total: {len(values)}"
    )

    ax.legend()

    fig.tight_layout()

    if save:
        fig.savefig(
            fname=f"histogram_combined_{column}.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    return fig, ax


def main():
    # Only import when using main, to avoid lib cluttering
    import argparse

    parser = argparse.ArgumentParser(description="Analyze CSV database.")

    parser.add_argument(
        "-i", "--input", type=str, help="Path to the CSV file to analyze"
    )
    parser.add_argument(
        "-p",
        "--property",
        type=str,
        help="Header property wanted from CSV file",
    )
    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        default=20,
        help="Number of bins for a histogram",
    )
    parser.add_argument(
        "-s",
        "--save",
        default=False,
        action="store_true",
        help="If given, save image to file",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="If active, export found outliers to CSV, output argument "
        "needed too",
    )
    parser.add_argument(
        "-o", "--output", help="Output for either saved png or exported CSV"
    )

    args = parser.parse_args()

    if args.export and not args.output:
        parser.error("--export requires --output to be set")

    image_output = "" if args.export else args.output

    outliers = show_histogram_from_csv(
        args.input,
        args.bins,
        args.property,
        args.save,
        image_output,
    )

    if args.export:
        if outliers is None:
            parser.error("Cannot export outliers for a non-numeric column")
        print(f"Exporting outliers to {args.output}")
        export_sorted_csv(outliers, args.property, args.output)


if __name__ == "__main__":
    main()
