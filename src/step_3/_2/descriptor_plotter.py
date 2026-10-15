import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import rc
import numpy as np
import pandas as pd

NON_FEATURE_COLUMNS = {"file_path", "was_closed", "reason"}
CLASS_COLUMN = "class"
FONT = {"weight": "bold", "size": 20}
FIGURE_SIZE = (25, 15)
RWIDTH = 0.8
rc("font", **FONT)
rc("xtick", labelsize=16)


def load_features(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load feature data from a CSV into a DataFrame.

    Adds:
        class: class name extracted from the first folder in file_path.
        was_closed: boolean indicating whether the mesh was closed.

    Returns:
        df: DataFrame containing metadata and numeric feature columns.
        feature_names: names of the numeric feature columns.
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"No rows in {csv_path}")

    # Extract the class name from the first folder in file_path.
    df[CLASS_COLUMN] = (
        df["file_path"]
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[0]
    )

    # Convert the closed column to boolean.
    df["was_closed"] = (
        df["was_closed"].astype(str).str.strip().str.lower().eq("true")
    )

    # Identify feature columns and convert them to numeric values.
    feature_names = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and c != CLASS_COLUMN
    ]

    for feature in feature_names:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    return df, feature_names


def grid_shape(n: int) -> tuple[int, int]:
    cols = min(3, n)
    return math.ceil(n / cols), cols


def plot_histograms(
    df: pd.DataFrame,
    feature_names: list[str],
    bins: int = 30,
    axes=None,
) -> Figure:
    """Plot one histogram per feature, with closed and open meshes overlaid."""
    n_rows, n_cols = grid_shape(len(feature_names))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    closed = df["was_closed"]

    for ax, feature in zip(axes.ravel(), feature_names):
        values = df[feature]
        finite = values[np.isfinite(values)]

        if finite.empty:
            ax.set_title(feature)
            ax.set_ylabel("Shapes")
            continue

        edges = np.histogram_bin_edges(finite, bins=bins)

        for mask, label, color in (
            (closed, f"Closed ({closed.sum()})", "tab:blue"),
            (~closed, f"Open ({(~closed).sum()})", "tab:orange"),
        ):
            group = values[mask]
            group = group[np.isfinite(group)]

            ax.hist(
                group,
                bins=edges,
                alpha=0.6,
                label=label,
                color=color,
            )

        ax.set_title(feature)
        ax.set_ylabel("Shapes")
        ax.legend()

    for ax in axes.ravel()[len(feature_names) :]:
        ax.axis("off")

    fig.tight_layout()
    return axes, fig


def plot_by_class(
    df: pd.DataFrame,
    feature_names: list[str],
    ax: Axes | None = None,
    class_limit: int = 10,
    grid_size: tuple[int, int] = (3, 2),
) -> tuple[Axes, Figure]:

    classes = df[CLASS_COLUMN].unique()[:class_limit]
    n_features = len(feature_names)

    if ax is None:
        fig, axes = plt.subplots(
            *grid_size,
            figsize=FIGURE_SIZE,
            squeeze=False,
            constrained_layout=True,
        )
        axes = axes.flatten()
    else:
        fig = ax.get_figure()
        axes = [ax]

    for i, feature in enumerate(feature_names):
        if i >= len(axes):
            break

        feature_ax = axes[i]

        data = [
            df.loc[df[CLASS_COLUMN] == cls, feature].dropna()
            for cls in classes
        ]

        feature_ax.boxplot(
            data, tick_labels=[str(cls) for cls in classes], sym=""
        )

        feature_ax.set_title(feature)
        feature_ax.set_xlabel("Class")
        feature_ax.set_ylabel("Descriptor value")
        feature_ax.tick_params(axis="x", rotation=25)

    # Hide unused subplots
    for j in range(min(n_features, len(axes)), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()

    return axes, fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the shape features from a feature CSV."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to CSV")
    parser.add_argument(
        "-o",
        "--output",
        help="Save the figure to this file (png/pdf/...) instead of showing",
    )
    parser.add_argument(
        "--by_class",
        action="store_true",
        help="Boxplot per class (first folder in file_path) instead of "
        "histograms",
    )
    parser.add_argument(
        "--closed_only",
        action="store_true",
        help="Only use meshes that were closed (volume-based features such "
        "as compactness are unreliable on open meshes)",
    )
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    df, feature_names = load_features(args.input)

    if args.closed_only:
        df = df.loc[df["was_closed"]].copy()

        if df.empty:
            raise ValueError("No closed meshes found in the input CSV.")

    if args.by_class:
        axes, fig = plot_by_class(df, feature_names)
    else:
        axes, fig = plot_histograms(df, feature_names, bins=args.bins)

    if args.output:
        fig.savefig(args.output, dpi=150, format="pdf")
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
