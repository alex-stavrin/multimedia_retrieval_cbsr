import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import rc

# Set up global Matplotlib variables
FONT = {"weight": "bold", "size": 20}
FIGURE_SIZE = (21, 7)
rc("font", **FONT)
rc("xtick", labelsize=14)


def show_histogram_from_csv(
    csv_path: str, n_bins: int, column: str = "", save=False
) -> None:
    """Load a CSV file and display its distribution and detected outliers."""
    if not csv_path.endswith(".csv"):
        raise ValueError("Input file must end on .csv")

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)

    plot_histogram(
        df,
        column,
        bins=n_bins,
        ylabel="Number of shapes",
        ax=axes[0],
    )

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
        fig.savefig(
            fname=f"histogram_plot_{column}.pdf",
            format="pdf",
            bbox_inches="tight",
        )

    if not save:
        plt.show()


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
    Plot a histogram and mark its mean and median.

    If an existing Axes object is provided, the histogram is drawn
    on that axis. Otherwise, a new figure and axis are created.

    Returns:
        The figure and axes containing the histogram.
    """

    mean = df[column].mean()
    median = df[column].median()

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    else:
        fig = ax.get_figure()

    kwargs.setdefault("edgecolor", "black")

    ax.hist(df[column], bins=bins, **kwargs)

    ax.axvline(
        mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean:.2f}",
    )

    ax.axvline(
        median,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median:.2f}",
    )

    ax.set_xlabel(column)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    if title == "":
        ax.set_title(f"Distribution of {column},\nTotal: {len(df)}")
    else:
        ax.set_title(title + f",\nTotal: {len(df)}")
    ax.legend()

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
    )

    ax.hist(
        outlier_values,
        bins=bins,
        color="red",
        alpha=0.75,
        edgecolor="black",
        label=f"Outliers (n={len(outlier_values)})",
    )

    # Mean and median of the complete distribution
    mean = values.mean()
    median = values.median()

    ax.axvline(
        mean,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean:.2f}",
    )

    ax.axvline(
        median,
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median:.2f}",
    )

    ax.set_xlabel(column)
    ax.set_ylabel("Number of shapes")

    ax.set_yscale("log")

    ax.set_title(
        f"Distribution and Outliers of {column}\n"
        f"IQR strictness: {iqr_strictness}, "
        f"percentile: {percentile:.0%}"
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

    args = parser.parse_args()

    show_histogram_from_csv(args.input, args.bins, args.property, args.save)


if __name__ == "__main__":
    main()
