import argparse
import csv
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import vedo

from src.helper.database_functions import (
    map_by_relative_path,
)

FONT = {"size": 25}
FIGSIZE = (20, 10)
rc("font", **FONT)


def compare_hole_fill_stats(
    original_database_path: str,
    filled_database_path: str,
    output_path: str,
):
    if not isinstance(original_database_path, str):
        raise TypeError(
            f"Database path must be string: {original_database_path}"
        )

    if not isinstance(filled_database_path, str):
        raise TypeError(
            f"Database path must be string: {filled_database_path}"
        )

    original_files = map_by_relative_path(original_database_path)
    filled_files = map_by_relative_path(filled_database_path)

    # Report files that exist in only one database
    only_original = sorted(original_files.keys() - filled_files.keys())
    only_filled = sorted(filled_files.keys() - original_files.keys())

    if only_original or only_filled:
        raise ValueError(
            "The databases do not contain the same files.\n"
            f"Only in original ({len(only_original)}): {only_original[:10]}\n"
            f"Only in filled ({len(only_filled)}): {only_filled[:10]}"
        )

    with open(output_path, "w", newline="") as f:
        field_names = [
            "file_path",
            "original_n_points",
            "filled_n_points",
            "original_n_faces",
            "filled_n_faces",
        ]

        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()

        # Sorted so the CSV output is deterministic
        for relative_path in sorted(original_files):
            original_shape = vedo.Mesh(original_files[relative_path])
            filled_shape = vedo.Mesh(filled_files[relative_path])

            writer.writerow(
                {
                    "file_path": relative_path,
                    "original_n_points": len(original_shape.points),
                    "filled_n_points": len(filled_shape.points),
                    "original_n_faces": len(original_shape.cells),
                    "filled_n_faces": len(filled_shape.cells),
                }
            )


def plot_differences(
    csv_path: str,
    property_before: str = "original_n_faces",
    property_after: str = "filled_n_faces",
    figure_path: str | None = None,
) -> None:
    """
    Compare two properties before and after standardisation.

    Produces:
      1. Scatter plot comparing the two properties.
      2. Distribution of relative changes for changed meshes.
    """
    df = pd.read_csv(csv_path)

    # Validate the requested columns.
    for column in (property_before, property_after):
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

    before = pd.to_numeric(df[property_before], errors="coerce")
    after = pd.to_numeric(df[property_after], errors="coerce")

    valid = before.notna() & after.notna()
    df = df.loc[valid].copy()
    before = before.loc[valid]
    after = after.loc[valid]

    delta = after - before
    rel_pct = 100 * delta / before.replace(0, np.nan)
    changed = delta != 0

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.suptitle(
        f"Effects of standardisation: "
        f"{property_before} vs. {property_after}"
    )

    # 1. Before vs. after scatter plot.
    ax = axes[0]

    unchanged = ~changed

    ax.scatter(
        before[unchanged],
        after[unchanged],
        s=8,
        c="lightgray",
        label=f"Unchanged ({unchanged.sum()})",
    )
    ax.scatter(
        before[changed],
        after[changed],
        s=10,
        c="tab:red",
        alpha=0.6,
        label=f"Changed ({changed.sum()})",
    )

    # Reference diagonal: no change.
    min_val = min(before.min(), after.min())
    max_val = max(before.max(), after.max())

    if min_val == max_val:
        padding = abs(min_val) * 0.05 or 1
        min_val -= padding
        max_val += padding

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        lw=1,
        label="No change (y = x)",
    )

    ax.set_xlabel(f"Before: {property_before}")
    ax.set_ylabel(f"After: {property_after}")
    ax.set_title("Before vs. after")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)

    # 2. Distribution of relative changes.
    ax = axes[1]
    rel = rel_pct[changed].dropna()

    if len(rel) > 0:
        lo, hi = np.percentile(rel, [1, 99])
        ax.hist(rel.clip(lo, hi), bins=50, color="tab:blue")
        ax.set_yscale("log")
        ax.axvline(0, color="k", lw=1)

    ax.set_xlabel("Relative change (%)")
    ax.set_ylabel("Number of meshes (log)")
    ax.set_title("Relative change in modified meshes")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    if figure_path is not None:
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {figure_path}")
    else:
        plt.show()

    print(f"\nMeshes analysed: {len(df)}")
    print(
        f"Meshes changed: {changed.sum()} " f"({100 * changed.mean():.2f}%)"
        if len(df)
        else "Meshes changed: 0"
    )

    print("\nTop 10 largest relative changes:")
    results = pd.DataFrame(
        {
            "file_path": (
                df["file_path"].values
                if "file_path" in df.columns
                else df.index
            ),
            "before": before.values,
            "after": after.values,
            "change": delta.values,
            "relative_change_pct": rel_pct.values,
        }
    )

    print(
        results.reindex(
            results["relative_change_pct"]
            .abs()
            .sort_values(ascending=False)
            .index
        )
        .head(10)
        .to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare mesh statistics before and after hole filling."
    )
    parser.add_argument(
        "-b",
        "--before",
        required=True,
        type=str,
        help="The before datbase, "
        "or before property of analysis when using --plot",
    )
    parser.add_argument(
        "-a",
        "--after",
        required=True,
        type=str,
        help="The before database, "
        "or before property of analysis when using --plot",
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        type=str,
        help="CSV path: written when comparing, read when using --plot.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot an existing CSV (given by --output) "
        "instead of calculating.",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default=None,
        help="Optional path to save the figure instead of showing it.",
    )
    parser.add_argument("--property")
    args = parser.parse_args()

    if not args.csv_path:
        raise ValueError("CSV path must be given")

    before: str = args.before
    after: str = args.after

    if args.plot:
        if os.path.exists(before):
            raise ValueError(
                "Before flag should be a dataframe property/csv column, "
                "not a file"
            )
        if os.path.exists(before):
            raise ValueError(
                "After flag should be a dataframe property/csv column, "
                "not a file"
            )

        plot_differences(
            csv_path=args.csv_path,
            property_before=before,
            property_after=after,
            figure_path=args.figure,
        )
    else:
        if not os.path.isdir(before):
            raise ValueError(
                "Before flag should be a directory for comparing datbases"
            )
        if not os.path.isdir(after):
            raise ValueError(
                "After flag should be a directory for comparing datbases"
            )

        compare_hole_fill_stats(
            original_database_path=before,
            filled_database_path=after,
            output_path=args.csv_path,
        )


if __name__ == "__main__":
    main()
