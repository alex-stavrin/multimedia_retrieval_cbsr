import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import rc
import numpy as np
import pandas as pd
import vedo

NON_FEATURE_COLUMNS = {"file_path", "was_closed", "reason"}
CLASS_COLUMN = "class"
FONT = {"size": 20}
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
        df["file_path"].str.replace("\\", "/", regex=False).str.split("/").str[0]
    )

    # Convert the closed column to boolean.
    df["was_closed"] = df["was_closed"].astype(str).str.strip().str.lower().eq("true")

    # Identify feature columns and convert them to numeric values.
    feature_names = [
        c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != CLASS_COLUMN
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

        data = [df.loc[df[CLASS_COLUMN] == cls, feature].dropna() for cls in classes]

        feature_ax.boxplot(data, tick_labels=[str(cls) for cls in classes], sym="")

        feature_ax.set_title(feature)
        feature_ax.set_xlabel("Class")
        feature_ax.set_ylabel("Descriptor value")
        feature_ax.tick_params(axis="x", rotation=25)

    # Hide unused subplots
    for j in range(min(n_features, len(axes)), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()

    return axes, fig


# ---
# Comparing individual shapes


def _normalize(path: str) -> str:
    return str(path).replace("\\", "/")


def _csv_filter(value: str, target: str) -> bool:
    return (value == target) or target.endswith("/" + value)


def _file_name_matches(path: str, file_name: str) -> bool:
    return path.split("/")[-1] == file_name


def find_shape_row(df: pd.DataFrame, shape_path: str) -> pd.Series:
    """
    Finds the CSV row that belongs to an .obj file.

    The CSV stores paths relative to the database root (e.g.
    "Apartment/D00086.obj"), while shape_path may be absolute or relative
    to anywhere. A row matches when its path is the end of shape_path. If
    nothing matches, a bare file name is matched against the file names in
    the CSV, as long as that is unambiguous.
    """
    target: str = _normalize(shape_path)
    csv_paths = df["file_path"].map(_normalize)

    ends_with = [_csv_filter(value, target) for value in csv_paths]
    matches = df.loc[ends_with]

    if matches.empty:
        file_name = target.split("/")[-1]
        matches = df.loc[[_file_name_matches(path, file_name) for path in csv_paths]]

    if matches.empty:
        raise ValueError(f"{shape_path} was not found in the CSV")

    if len(matches) > 1:
        # Several rows end the same way: take the most specific (longest)
        lengths = matches["file_path"].str.len()
        best = matches.loc[lengths == lengths.max()]
        if len(best) > 1:
            raise ValueError(
                f"{shape_path} is ambiguous, it matches: "
                f"{list(best['file_path'])}. Give more of the path."
            )
        matches = best

    return matches.iloc[0]


def render_mesh(shape_path: str, size: tuple[int, int] = (700, 700)):
    """Renders an .obj file off-screen and returns the image as an array."""
    mesh = vedo.Mesh(shape_path).color("lightsteelblue")

    plotter = vedo.Plotter(offscreen=True, size=size, axes=0, bg="white")
    plotter.show(mesh, interactive=False)
    plotter.camera.Azimuth(35)
    plotter.camera.Elevation(25)
    plotter.render()

    image = plotter.screenshot(asarray=True)
    plotter.close()

    return image


def _format_value(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.4g}"
    return str(value)


def _save_comparison_table(
    rows: list[pd.Series],
    titles: list[str],
    feature_names: list[str],
    output_path: str,
) -> None:
    """Save shape descriptors as a LaTeX table."""
    table_rows = {
        name: [_format_value(row[name]) for row in rows] for name in feature_names
    }

    table_df = pd.DataFrame(table_rows, index=titles).T
    table_df.index.name = "Descriptor"

    latex = table_df.to_latex(
        caption="Comparison of shape descriptors",
        label="tab:shape-comparison",
        column_format="l|" + "c" * len(titles),
        escape=True,
        index=True,
        header=True,
        index_names=True,
        bold_rows=False,
    )

    output_path = str(output_path)
    if not output_path.endswith(".tex"):
        output_path += ".tex"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(latex)


def compare_shapes(
    df: pd.DataFrame,
    feature_names: list[str],
    shape_paths: list[str],
    output_path: str | None = None,
) -> Figure:
    """
    Compare 3D shapes, optionally displaying only the shapes for export.

    When output_path is None, displays the shapes and descriptor table.
    When output_path is given, displays only the shapes with titles.
    """
    if len(shape_paths) < 2:
        raise ValueError("Give at least two shapes to compare")

    rows = [find_shape_row(df, path) for path in shape_paths]
    titles = [
        f"{row[CLASS_COLUMN]} / {_normalize(row['file_path']).split('/')[-1]}"
        for row in rows
    ]

    n = len(shape_paths)

    if output_path is not None:
        # Compact, single-row figure for export.
        fig, axes = plt.subplots(
            1,
            n,
            figsize=(max(4 * n, 8), 4.5),
            squeeze=False,
        )
        axes = axes[0]

        for ax, path, title in zip(axes, shape_paths, titles):
            ax.imshow(render_mesh(path))
            ax.set_title(title, fontsize=22, pad=6)
            ax.axis("off")

        fig.tight_layout(pad=0.5)

        # Export to latex file
        _save_comparison_table(rows, titles, feature_names, output_path)
        return fig

    # Full comparison figure, including the descriptor table.
    fig = plt.figure(
        figsize=(max(8, 5 * n), 10),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2, n, height_ratios=[3, 2])

    for i, (path, title) in enumerate(zip(shape_paths, titles)):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(render_mesh(path))
        ax.set_title(title)
        ax.axis("off")

    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis("off")

    cell_text = [[_format_value(row[name]) for row in rows] for name in feature_names]

    table = table_ax.table(
        cellText=cell_text,
        rowLabels=feature_names,
        colLabels=titles,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.2)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the shape features from a feature CSV."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to CSV")
    parser.add_argument(
        "-o",
        "--output",
        help="Save the figure to this file (png/pdf/...) instead of showing",
        default=None,
    )
    parser.add_argument(
        "--by_class",
        action="store_true",
        help="Boxplot per class (first folder in file_path) instead of " "histograms",
    )
    parser.add_argument(
        "--closed_only",
        action="store_true",
        help="Only use meshes that were closed (volume-based features such "
        "as compactness are unreliable on open meshes)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="OBJ",
        help="Two or more .obj files to show side by side with their "
        "descriptors from the CSV (ignores the other plot options)",
    )
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    df, feature_names = load_features(args.input)

    if args.compare:
        if len(args.compare) < 2:
            parser.error("--compare needs at least two .obj files")
        fig = compare_shapes(df, feature_names, args.compare, args.output)
    else:
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
