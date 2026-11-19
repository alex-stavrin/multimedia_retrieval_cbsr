"""Analyze triangle area distributions (using histograms) before and after resampling.

Evaluates whether adaptive subdivision and resampling produced a more even
triangle area distribution by comparing per-mesh Coefficient of Variation
(CV = std / mean of triangle areas) and top-1% area share before and after
resampling across:
  1. All meshes in the database.
  2. The subset of meshes that needed/underwent adaptive subdivision.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vedo
from matplotlib import rc
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm

import src.helper.silence_vtk  # noqa: F401
from src.helper.database_functions import map_by_relative_path
from src.step_2._3.resampling import (
    MAX_TOP_1PCT_AREA_SHARE,
    MAX_VERTEX_COUNT,
    MIN_VERTEX_COUNT,
    _compute_triangle_areas,
)

FONT = {"size": 14}
rc("font", **FONT)


@dataclass
class MeshAreaStats:
    n_vertices: int
    n_faces: int
    total_area: float
    mean_area: float
    std_area: float
    median_area: float
    min_area: float
    max_area: float
    cv_area: float
    top_1pct_area_share: float


@dataclass
class TriangleAreaComparisonRecord:
    relative_file_path: str
    shape_class: str
    before_n_vertices: int
    after_n_vertices: int
    before_n_faces: int
    after_n_faces: int
    before_mean_area: float
    after_mean_area: float
    before_std_area: float
    after_std_area: float
    before_median_area: float
    after_median_area: float
    before_cv_area: float
    after_cv_area: float
    cv_area_diff: float
    cv_area_rel_change_pct: float
    before_top_1pct_share: float
    after_top_1pct_share: float
    needed_vertex_subdivision: bool
    needed_area_subdivision: bool
    underwent_subdivision: bool
    resampling_category: str


def compute_mesh_area_stats(mesh_path: str | Path) -> MeshAreaStats:
    """Load a mesh and compute its triangle area distribution statistics."""
    mesh = vedo.Mesh(str(mesh_path))
    faces = np.asarray(mesh.cells, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        tri_mesh = mesh.clone().triangulate()
        faces = np.asarray(tri_mesh.cells, dtype=np.int64)
        vertices = np.asarray(tri_mesh.vertices, dtype=np.float64)
        n_vertices = int(tri_mesh.npoints)
    else:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        n_vertices = int(mesh.npoints)

    if len(faces) == 0:
        return MeshAreaStats(
            n_vertices=n_vertices,
            n_faces=0,
            total_area=0.0,
            mean_area=0.0,
            std_area=0.0,
            median_area=0.0,
            min_area=0.0,
            max_area=0.0,
            cv_area=0.0,
            top_1pct_area_share=0.0,
        )

    areas = _compute_triangle_areas(vertices, faces)
    total_area = float(areas.sum())
    mean_area = float(areas.mean())
    std_area = float(areas.std())
    median_area = float(np.median(areas))
    min_area = float(areas.min())
    max_area = float(areas.max())

    cv_area = (std_area / mean_area) if mean_area > 1e-12 else 0.0

    if total_area > 1e-12:
        top_1pct_count = max(1, len(areas) // 100)
        top_1pct_area = float(np.sort(areas)[-top_1pct_count:].sum())
        top_1pct_share = top_1pct_area / total_area
    else:
        top_1pct_share = 0.0

    return MeshAreaStats(
        n_vertices=n_vertices,
        n_faces=len(faces),
        total_area=total_area,
        mean_area=mean_area,
        std_area=std_area,
        median_area=median_area,
        min_area=min_area,
        max_area=max_area,
        cv_area=cv_area,
        top_1pct_area_share=top_1pct_share,
    )


def _compare_single_mesh_pair(
    relative_path: str,
    before_file: str,
    after_file: str,
) -> TriangleAreaComparisonRecord:
    """Worker function to compute before/after area stats for a single mesh."""
    before_stats = compute_mesh_area_stats(before_file)
    after_stats = compute_mesh_area_stats(after_file)

    norm_rel = relative_path.replace("\\", "/")
    parts = norm_rel.split("/")
    shape_class = parts[0] if len(parts) > 1 else "Unknown"

    needed_vertex_sub = before_stats.n_vertices < MIN_VERTEX_COUNT
    needed_area_sub = before_stats.top_1pct_area_share > MAX_TOP_1PCT_AREA_SHARE
    needed_decimation = before_stats.n_vertices > MAX_VERTEX_COUNT

    # Determine whether the mesh underwent subdivision during resampling:
    # In resampling.py, _adaptive_subdivide runs when:
    #   npoints < MIN_VERTEX_COUNT or (needs_area_fix and npoints < 9500)
    # (including after initial decimation to 6500 if it was > MAX_VERTEX_COUNT).
    underwent_subdivision = (
        needed_vertex_sub
        or (needed_area_sub and before_stats.n_vertices < 9_500)
        or (after_stats.n_vertices > before_stats.n_vertices)
    )

    if needed_vertex_sub:
        category = "subdivided_low_vertices"
    elif needed_decimation:
        category = "decimated"
    elif needed_area_sub and underwent_subdivision:
        category = "subdivided_uneven_area"
    elif after_stats.n_vertices != before_stats.n_vertices:
        category = "modified"
    else:
        category = "unchanged"

    cv_diff = after_stats.cv_area - before_stats.cv_area
    if before_stats.cv_area > 1e-12:
        cv_rel_pct = 100.0 * cv_diff / before_stats.cv_area
    else:
        cv_rel_pct = 0.0

    return TriangleAreaComparisonRecord(
        relative_file_path=norm_rel,
        shape_class=shape_class,
        before_n_vertices=before_stats.n_vertices,
        after_n_vertices=after_stats.n_vertices,
        before_n_faces=before_stats.n_faces,
        after_n_faces=after_stats.n_faces,
        before_mean_area=before_stats.mean_area,
        after_mean_area=after_stats.mean_area,
        before_std_area=before_stats.std_area,
        after_std_area=after_stats.std_area,
        before_median_area=before_stats.median_area,
        after_median_area=after_stats.median_area,
        before_cv_area=before_stats.cv_area,
        after_cv_area=after_stats.cv_area,
        cv_area_diff=cv_diff,
        cv_area_rel_change_pct=cv_rel_pct,
        before_top_1pct_share=before_stats.top_1pct_area_share,
        after_top_1pct_share=after_stats.top_1pct_area_share,
        needed_vertex_subdivision=needed_vertex_sub,
        needed_area_subdivision=needed_area_sub,
        underwent_subdivision=underwent_subdivision,
        resampling_category=category,
    )


def analyze_triangle_areas(
    before_dir: str | Path,
    after_dir: str | Path,
    output_csv: str | Path | None = None,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compare triangle area distributions before and after resampling across a database."""
    before_path = Path(before_dir).resolve()
    after_path = Path(after_dir).resolve()

    if not before_path.is_dir():
        raise NotADirectoryError(f"Before directory not found: {before_path}")
    if not after_path.is_dir():
        raise NotADirectoryError(f"After directory not found: {after_path}")

    before_files = map_by_relative_path(str(before_path))
    after_files = map_by_relative_path(str(after_path))

    common_keys = sorted(set(before_files.keys()) & set(after_files.keys()))
    if not common_keys:
        raise ValueError(
            f"No matching .obj files found between {before_path} and {after_path}."
        )

    records: list[TriangleAreaComparisonRecord] = []
    n_workers = max_workers or os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _compare_single_mesh_pair,
                key,
                before_files[key],
                after_files[key],
            ): key
            for key in common_keys
        }

        iterator = (
            tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Analyzing triangle areas (before vs. after)",
            )
            if show_progress
            else as_completed(futures)
        )

        for future in iterator:
            records.append(future.result())

    records.sort(key=lambda r: r.relative_file_path)
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)

    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(df.columns))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[TriangleAreaAnalyzer] Saved {len(df)} mesh comparisons to {csv_path}")

    return df


def _plot_before_after_cv_overlay(
    ax: Axes,
    before_cv: pd.Series,
    after_cv: pd.Series,
    title: str,
    bins: int = 35,
    clip_percentile: float = 99.0,
) -> None:
    """Plot overlaid before vs. after histograms of per-mesh triangle area CV."""
    b_vals = before_cv.dropna().to_numpy(dtype=np.float64)
    a_vals = after_cv.dropna().to_numpy(dtype=np.float64)

    if len(b_vals) == 0 or len(a_vals) == 0:
        ax.set_title(f"{title} (no data)")
        return

    combined = np.concatenate([b_vals, a_vals])
    upper = float(np.percentile(combined, clip_percentile))
    upper = max(upper, float(np.max(a_vals)), 1.0)
    bin_edges = np.linspace(0.0, upper, bins + 1)

    b_clipped = np.clip(b_vals, 0.0, upper)
    a_clipped = np.clip(a_vals, 0.0, upper)

    ax.hist(
        b_clipped,
        bins=bin_edges,
        alpha=0.55,
        color="tab:orange",
        edgecolor="black",
        label="Before resampling",
    )
    ax.hist(
        a_clipped,
        bins=bin_edges,
        alpha=0.60,
        color="tab:blue",
        edgecolor="black",
        label="After resampling",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Triangle Area Coefficient of Variation (std / mean)")
    ax.set_ylabel("Number of Meshes")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, loc="upper right")


def plot_triangle_area_histograms(
    df: pd.DataFrame,
    output_plot: str | Path | None = None,
    bins: int = 35,
    show: bool = False,
) -> Figure:
    """Create a histogram figure comparing triangle area CV before & after resampling."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    _plot_before_after_cv_overlay(
        ax,
        df["before_cv_area"],
        df["after_cv_area"],
        title="Triangle Area CV Before vs. After Resampling",
        bins=bins,
    )

    fig.tight_layout()

    if output_plot is not None:
        plot_path = Path(output_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[TriangleAreaAnalyzer] Saved histogram plot to {plot_path}")

        alt_suffix = ".pdf" if plot_path.suffix.lower() != ".pdf" else ".png"
        alt_path = plot_path.with_suffix(alt_suffix)
        fig.savefig(alt_path, dpi=150, bbox_inches="tight")
        print(f"[TriangleAreaAnalyzer] Saved histogram plot to {alt_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def generate_summary_text(df: pd.DataFrame) -> str:
    """Format a human-readable summary checking whether adaptive subdivision worked."""
    sub_df = df[df["underwent_subdivision"] == True]  # noqa: E712
    uneven_before = df[df["needed_area_subdivision"] == True]  # noqa: E712
    uneven_after = df[df["after_top_1pct_share"] > MAX_TOP_1PCT_AREA_SHARE]

    all_improved = (df["after_cv_area"] < df["before_cv_area"]).sum()
    sub_improved = (sub_df["after_cv_area"] < sub_df["before_cv_area"]).sum()
    uneven_improved = (
        uneven_before["after_cv_area"] < uneven_before["before_cv_area"]
    ).sum()

    lines = [
        "======================================================================",
        "TRIANGLE AREA DISTRIBUTION ANALYSIS (BEFORE vs. AFTER RESAMPLING)",
        "======================================================================",
        f"Total meshes analyzed: {len(df)}",
        f"  - Needed vertex subdivision (< {MIN_VERTEX_COUNT} vertices): "
        f"{int(df['needed_vertex_subdivision'].sum())}",
        f"  - Had uneven triangle area distribution (top 1% area > {MAX_TOP_1PCT_AREA_SHARE:.0%}): "
        f"{len(uneven_before)}",
        f"  - Underwent subdivision total: {len(sub_df)}",
        "",
        "1. ALL MESHES - Coefficient of Variation (CV = std / mean of triangle areas):",
        f"  - Before resampling: mean = {df['before_cv_area'].mean():.4f}, "
        f"median = {df['before_cv_area'].median():.4f}, "
        f"std = {df['before_cv_area'].std():.4f}",
        f"  - After resampling:  mean = {df['after_cv_area'].mean():.4f}, "
        f"median = {df['after_cv_area'].median():.4f}, "
        f"std = {df['after_cv_area'].std():.4f}",
        f"  - Meshes with reduced (more even) CV: {all_improved}/{len(df)} "
        f"({100.0 * all_improved / max(1, len(df)):.1f}%)",
        "",
        "2. SUBDIVIDED MESHES ONLY - Coefficient of Variation (CV = std / mean):",
        f"  - Before resampling: mean = {sub_df['before_cv_area'].mean():.4f}, "
        f"median = {sub_df['before_cv_area'].median():.4f}",
        f"  - After resampling:  mean = {sub_df['after_cv_area'].mean():.4f}, "
        f"median = {sub_df['after_cv_area'].median():.4f}",
        f"  - Median relative CV change: {sub_df['cv_area_rel_change_pct'].median():+.2f}%",
        f"  - Subdivided meshes with reduced (more even) CV: "
        f"{sub_improved}/{len(sub_df)} "
        f"({100.0 * sub_improved / max(1, len(sub_df)):.1f}%)",
        "",
        "3. UNEVEN-AREA MESHES CHECK (Adaptive Subdivision Target Group):",
        f"  - Meshes exceeding top-1% area share threshold ({MAX_TOP_1PCT_AREA_SHARE:.0%}) "
        f"before resampling: {len(uneven_before)}",
        f"  - Meshes exceeding top-1% area share threshold ({MAX_TOP_1PCT_AREA_SHARE:.0%}) "
        f"after resampling:  {len(uneven_after)}",
        f"  - Uneven meshes with improved CV after resampling: "
        f"{uneven_improved}/{len(uneven_before)} "
        f"({100.0 * uneven_improved / max(1, len(uneven_before)):.1f}%)",
        "======================================================================",
    ]
    return "\n".join(lines) + "\n"


def analyze_and_plot_triangle_areas(
    before_dir: str | Path,
    after_dir: str | Path,
    output_csv: str | Path,
    output_plot: str | Path | None = None,
    output_summary: str | Path | None = None,
    bins: int = 35,
    max_workers: int | None = None,
    show_progress: bool = True,
    show_plot: bool = False,
) -> pd.DataFrame:
    """Run the full before/after triangle area analysis, save CSV, summary, and histogram plots."""
    csv_path = Path(output_csv)
    if output_plot is None:
        output_plot = csv_path.parent / "triangle_area_cv_histograms.png"
    if output_summary is None:
        output_summary = csv_path.parent / "triangle_area_summary.txt"

    df = analyze_triangle_areas(
        before_dir=before_dir,
        after_dir=after_dir,
        output_csv=csv_path,
        max_workers=max_workers,
        show_progress=show_progress,
    )

    summary = generate_summary_text(df)
    print(summary)

    summary_path = Path(output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    print(f"[TriangleAreaAnalyzer] Saved summary report to {summary_path}")

    plot_triangle_area_histograms(
        df=df,
        output_plot=output_plot,
        bins=bins,
        show=show_plot,
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze triangle area distribution (using histograms of CV = std/mean) "
            "before and after resampling."
        )
    )
    parser.add_argument(
        "-b",
        "--before",
        type=Path,
        help="Path to the database directory before resampling (e.g. 1_models_reparation).",
    )
    parser.add_argument(
        "-a",
        "--after",
        type=Path,
        help="Path to the database directory after resampling (e.g. 2_models_resampled).",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=Path(
            "3d_models_prepared/analysis/2_models_resampled/triangle_area_distribution.csv"
        ),
        help="Output CSV path (or input CSV path when --from_csv is used).",
    )
    parser.add_argument(
        "-p",
        "--output_plot",
        type=Path,
        default=None,
        help="Output path for the histogram figure (.png or .pdf).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=35,
        help="Number of histogram bins (default: 35).",
    )
    parser.add_argument(
        "--from_csv",
        action="store_true",
        help="Plot histograms directly from an existing CSV without re-scanning meshes.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the matplotlib figure interactively.",
    )
    args = parser.parse_args()

    if args.from_csv:
        if not args.output_csv.is_file():
            parser.error(f"CSV file not found: {args.output_csv}")
        df = pd.read_csv(args.output_csv)
        print(generate_summary_text(df))
        plot_path = (
            args.output_plot
            or args.output_csv.parent / "triangle_area_cv_histograms.png"
        )
        plot_triangle_area_histograms(
            df=df,
            output_plot=plot_path,
            bins=args.bins,
            show=args.show,
        )
    else:
        if args.before is None or args.after is None:
            parser.error("--before and --after are required unless --from_csv is set.")
        analyze_and_plot_triangle_areas(
            before_dir=args.before,
            after_dir=args.after,
            output_csv=args.output_csv,
            output_plot=args.output_plot,
            bins=args.bins,
            show_plot=args.show,
        )


if __name__ == "__main__":
    main()
