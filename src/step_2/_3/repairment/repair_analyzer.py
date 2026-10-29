import argparse
import csv
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import vedo
from typing import Callable
from tqdm import tqdm

from src.helper.database_functions import (
    map_by_relative_path,
)
from src.step_2._3.repairment.repair_wireframe import (
    STEPS,
    retrieve_all_shape_files,
    REPORT_FIELDS,
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


def generate_repair_report(
    input_dir_path: str,
    output_dir_path: str,
    report_output_path: str | None = None,
    steps: dict[str, Callable] = STEPS,
) -> None:
    """Generates a repair report for an already processed database."""
    if not isinstance(input_dir_path, str):
        raise TypeError("input_dir_path must be a str")
    if not isinstance(output_dir_path, str):
        raise TypeError("output_dir_path must be a str")
    if report_output_path is not None and not isinstance(
        report_output_path, str
    ):
        raise TypeError("report_output_path must be a str or None")
    if not isinstance(steps, dict):
        raise TypeError("steps must be a dict")

    all_shape_files = retrieve_all_shape_files(input_dir_path)
    root_abs = os.path.abspath(input_dir_path)

    os.makedirs(output_dir_path, exist_ok=True)
    report_path = os.path.join(output_dir_path, "report.csv")

    n_closed = n_open = n_errors = n_closed_raw = n_repaired = 0

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()

        for shape_file in tqdm(
            all_shape_files, desc="Generating repair report"
        ):
            if not isinstance(shape_file, str):
                raise TypeError(
                    "shape_file must be a str, got "
                    f"{type(shape_file).__name__}"
                )

            relative_path = os.path.relpath(shape_file, root_abs)

            closed_raw = False
            repaired = False
            closed_after = False
            reason = ""

            try:
                # Inspect the mesh without applying repairs.
                shape = vedo.Mesh(shape_file)
                closed_after_result = shape.is_closed()

                if not isinstance(closed_after_result, (bool, int)):
                    raise TypeError(
                        "shape.is_closed() must return a bool or int, "
                        f"got {type(closed_after_result).__name__}"
                    )

                closed_after = bool(closed_after_result)

            except Exception as e:
                reason = f"error: {type(e).__name__}: {e}"
                n_errors += 1
            else:
                n_closed_raw += int(closed_raw)
                n_repaired += int(repaired)

                if closed_after:
                    n_closed += 1
                else:
                    n_open += 1

            row: dict[str, str | bool] = {
                "file_path": relative_path,
                "closed_raw": closed_raw,
                "repaired": repaired,
                "closed_after": closed_after,
                "reason": reason,
            }

            writer.writerow(row)

    summary = (
        f"Steps: {' > '.join(steps)}\n"
        f"Done. {len(all_shape_files)} shapes: "
        f"{n_closed} closed, {n_open} still open, {n_errors} errors.\n"
        f"{n_closed_raw} were closed as loaded and {n_repaired} became "
        f"closed through the steps.\n"
    )

    if report_output_path is None:
        report_output_path = os.path.join(
            output_dir_path, "report_conclusion.txt"
        )

    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary + f"Report written to {report_path}")


def generate_closing_result_csv(
    root_dir_raw_database: str,
    root_dir_repaired_database: str,
    result_output_path: str,
) -> None:
    """Compare raw and repaired meshes and write a CSV report."""

    raw_root = os.path.abspath(root_dir_raw_database)
    repaired_root = os.path.abspath(root_dir_repaired_database)

    raw_files = retrieve_all_shape_files(raw_root)
    repaired_files = retrieve_all_shape_files(repaired_root)

    raw_by_relative_path = {
        os.path.relpath(path, raw_root): path for path in raw_files
    }
    repaired_by_relative_path = {
        os.path.relpath(path, repaired_root): path for path in repaired_files
    }

    # Include every mesh found in either database.
    all_relative_paths = sorted(
        set(raw_by_relative_path) | set(repaired_by_relative_path)
    )

    os.makedirs(
        os.path.dirname(os.path.abspath(result_output_path)),
        exist_ok=True,
    )

    with open(
        result_output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()

        for relative_path in tqdm(
            all_relative_paths,
            desc="Comparing raw and repaired meshes",
        ):
            row = {
                "file_path": relative_path,
                "closed_raw": "",
                "closed_after": "",
                "repaired": "",
                "boundary_edges_after": "",
                "nonmanifold_edges_after": "",
                "trace": "",
                "reason": "",
            }

            errors: list[str] = []

            raw_path = raw_by_relative_path.get(relative_path)
            repaired_path = repaired_by_relative_path.get(relative_path)

            # Inspect the original mesh independently.
            if raw_path is None:
                errors.append("Mesh is missing from raw database")
            else:
                try:
                    raw_mesh = vedo.Mesh(raw_path)
                    row["closed_raw"] = bool(raw_mesh.is_closed())
                except Exception as e:
                    errors.append(f"Raw mesh error: {type(e).__name__}: {e}")

            # Inspect the repaired mesh independently.
            if repaired_path is None:
                errors.append("Mesh is missing from repaired database")
            else:
                try:
                    repaired_mesh = vedo.Mesh(repaired_path)
                    closed_after = bool(repaired_mesh.is_closed())

                    row["closed_after"] = closed_after

                    stats = edge_stats(repaired_mesh)
                    row["boundary_edges_after"] = stats["boundary_edges"]
                    row["nonmanifold_edges_after"] = stats["nonmanifold_edges"]

                except Exception as e:
                    errors.append(
                        f"Repaired mesh error: {type(e).__name__}: {e}"
                    )

            # Only determine whether repair succeeded when both
            # closure statuses are known.
            if isinstance(row["closed_raw"], bool) and isinstance(
                row["closed_after"], bool
            ):
                row["repaired"] = not row["closed_raw"] and row["closed_after"]

            row["reason"] = "; ".join(errors)

            writer.writerow(row)


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
        fig.savefig(figure_path, dpi=150, bbox_inches="tight", format="pdf")
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
    parser.add_argument(
        "--create_points_and_faces_report",
        type=str,
        required=False,
        metavar="REPORT_OUTPUT_PATH",
        help="The output path for the report, input and output become before "
        "and after database",
    )
    parser.add_argument(
        "--create_closed_report",
        type=str,
        required=False,
        metavar="REPORT_OUTPUT_PATH",
        help="The output path for the report, input and output become before "
        "and after database",
    )
    parser.add_argument("--property")
    args = parser.parse_args()

    before: str = args.before
    after: str = args.after

    if args.plot:
        if os.path.exists(before):
            raise ValueError(
                "Before flag should be a dataframe property/csv column, "
                "not a file"
            )
        if os.path.exists(after):
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

        if args.create_closed_report:
            generate_closing_result_csv(
                args.before, args.after, args.create_closed_report
            )
        elif args.create_points_and_faces_report:
            generate_repair_report(
                args.before, args.after, args.create_points_and_faces_report
            )
        else:
            compare_hole_fill_stats(
                original_database_path=before,
                filled_database_path=after,
                output_path=args.csv_path,
            )


if __name__ == "__main__":
    main()
