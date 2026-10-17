import argparse
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vedo

from src.helper.database_functions import (
    map_by_relative_path,
)


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
    figure_path: str | None = None,
    top_n_classes: int = 15,
) -> None:
    """
    Visualises the CSV written by compare_hole_fill_stats.
    Shows what changed, how much it changed, and where (per class).
    """
    df = pd.read_csv(csv_path)

    df["delta_points"] = df["filled_n_points"] - df["original_n_points"]
    df["delta_faces"] = df["filled_n_faces"] - df["original_n_faces"]
    df["rel_faces_pct"] = (
        100 * df["delta_faces"] / df["original_n_faces"].replace(0, np.nan)
    )
    df["changed"] = (df["delta_points"] != 0) | (df["delta_faces"] != 0)

    # Class = first directory of the relative path (e.g. "Chair/m123.obj")
    def get_class(path: str) -> str:
        parts = os.path.normpath(path).split(os.sep)
        return parts[0] if len(parts) > 1 else "(root)"

    df["shape_class"] = df["file_path"].map(get_class)

    changed = df[df["changed"]]
    unchanged = df[~df["changed"]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Effect of hole filling / standardisation", fontsize=15)

    # 1. Original vs. filled face count (log-log). Off the diagonal = changed.
    ax = axes[0, 0]
    ax.scatter(
        unchanged["original_n_faces"],
        unchanged["filled_n_faces"],
        s=8,
        c="lightgray",
        label=f"unchanged ({len(unchanged)})",
    )
    ax.scatter(
        changed["original_n_faces"],
        changed["filled_n_faces"],
        s=10,
        c="tab:red",
        alpha=0.6,
        label=f"changed ({len(changed)})",
    )
    lims = [
        max(1, df[["original_n_faces", "filled_n_faces"]].min().min()),
        df[["original_n_faces", "filled_n_faces"]].max().max(),
    ]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Original faces")
    ax.set_ylabel("Filled faces")
    ax.set_title("Face count before vs. after")
    ax.legend()

    # 2. Distribution of relative face change (changed meshes only)
    ax = axes[0, 1]
    rel = changed["rel_faces_pct"].dropna()
    if len(rel) > 0:
        lo, hi = np.percentile(rel, [1, 99])
        ax.hist(rel.clip(lo, hi), bins=50, color="tab:blue")
        ax.set_yscale("log")
        ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Face count change (%), clipped to 1st-99th percentile")
    ax.set_ylabel("Number of meshes (log)")
    ax.set_title("Relative face change (changed meshes)")

    # 3. ECDF of absolute relative change, all meshes
    ax = axes[0, 2]
    abs_rel = np.sort(df["rel_faces_pct"].abs().dropna().to_numpy())
    if len(abs_rel) > 0:
        ax.plot(abs_rel, np.arange(1, len(abs_rel) + 1) / len(abs_rel))
        ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("|Face count change| (%)")
    ax.set_ylabel("Fraction of meshes")
    ax.set_title("Cumulative distribution of change")
    ax.grid(alpha=0.3)

    # 4. Delta points vs. delta faces (both can be negative due to clean())
    ax = axes[1, 0]
    ax.scatter(
        df["delta_points"],
        df["delta_faces"],
        s=10,
        c=df["changed"].map({True: "tab:red", False: "lightgray"}),
        alpha=0.6,
    )
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Change in points")
    ax.set_ylabel("Change in faces")
    ax.set_title("Points vs. faces change")

    # 5. Per-class share of changed meshes
    ax = axes[1, 1]
    per_class = (
        df.groupby("shape_class")["changed"]
        .agg(["sum", "count"])
        .assign(pct=lambda d: 100 * d["sum"] / d["count"])
        .sort_values("sum", ascending=False)
        .head(top_n_classes)
        .iloc[::-1]
    )
    ax.barh(per_class.index, per_class["pct"], color="tab:orange")
    for y, (n_changed, n_total) in enumerate(
        zip(per_class["sum"], per_class["count"])
    ):
        ax.text(1, y, f"{n_changed}/{n_total}", va="center", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of meshes changed")
    ax.set_title(f"Top {len(per_class)} classes by number of changed meshes")

    # 6. Text summary
    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        f"Total meshes:        {len(df)}\n"
        f"Changed:             {len(changed)} "
        f"({100 * len(changed) / max(len(df), 1):.1f}%)\n"
        f"Faces added (total): {int(df['delta_faces'].clip(lower=0).sum())}\n"
        f"Faces removed:       {int((-df['delta_faces']).clip(lower=0).sum())}\n"
        f"Points added:        {int(df['delta_points'].clip(lower=0).sum())}\n"
        f"Points removed:      {int((-df['delta_points']).clip(lower=0).sum())}\n"
        f"Median |rel. face change| (changed): "
        f"{changed['rel_faces_pct'].abs().median():.2f}%\n"
        f"Max rel. face change: {df['rel_faces_pct'].abs().max():.1f}%"
    )
    ax.text(0, 1, summary, va="top", family="monospace", fontsize=11)

    fig.tight_layout()

    if figure_path is not None:
        fig.savefig(figure_path, dpi=150)
        print(f"Figure saved to {figure_path}")
    else:
        plt.show()

    # Largest changes, useful for manual inspection
    print("\nTop 10 largest relative face changes:")
    print(
        df.reindex(
            df["rel_faces_pct"].abs().sort_values(ascending=False).index
        )
        .head(10)[
            [
                "file_path",
                "original_n_faces",
                "filled_n_faces",
                "rel_faces_pct",
            ]
        ]
        .to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare mesh statistics before and after hole filling."
    )
    parser.add_argument("-d", "--original_database", type=str)
    parser.add_argument("-p", "--processed_database", type=str)
    parser.add_argument(
        "-o",
        "--output",
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
    args = parser.parse_args()

    if args.plot:
        plot_differences(args.output, args.figure)
    else:
        compare_hole_fill_stats(
            original_database_path=args.original_database,
            filled_database_path=args.processed_database,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
