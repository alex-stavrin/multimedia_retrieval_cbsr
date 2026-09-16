import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def show_histogram_from_csv(
    csv_path: str, n_bins: int, column: str = ""
) -> None:
    """Load a CSV file and display its distribution and detected outliers."""
    if not csv_path.endswith(".csv"):
        ValueError("Input file must end on .csv")

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_histogram(df, column, bins=n_bins, ylabel=column, ax=axes[0])

    outliers = find_outliers(df, column, interquartile_strictness=1.5)
    plot_histogram(
        outliers,
        column,
        bins=n_bins,
        ylabel="Outliers",
        ax=axes[1],
        color="red",
    )

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
    df, column: str, bins: int = 20, ylabel: str = "", ax=None, **kwargs
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
        fig, ax = plt.subplots(figsize=(10, 6))
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
    ax.set_title(f"Distribution of {column}, total: {len(df)}")
    ax.legend()

    fig.tight_layout()

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

    args = parser.parse_args()

    show_histogram_from_csv(args.input, args.bins, args.property)


if __name__ == "__main__":
    main()
