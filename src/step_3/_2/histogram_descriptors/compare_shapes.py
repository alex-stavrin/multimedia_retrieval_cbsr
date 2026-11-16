"""Plot histogram descriptors (A3, D1, D2, D3, D4) and 3D model comparisons for 2 similar shapes and 1 different shape.

Supports exporting:
  - Each descriptor (A3, D1, D2, D3, D4) as its own individual LaTeX-ready PDF
  - A standalone 1x3 PDF of the 3 compared 3D models (shapes_3d_comparison.pdf)
  - Or a single chosen descriptor (-d A3, -d models, etc.)

Usage examples:
    # Save all 5 descriptor PDFs + the 3D models comparison PDF (shapes_3d_comparison.pdf):
    python -m src.step_3._2.histogram_descriptors.compare_shapes -pdf --no-show

    # Generate ONLY the 3D models comparison PDF:
    python -m src.step_3._2.histogram_descriptors.compare_shapes -d models -pdf --no-show

    # Plot/save just one specific descriptor (e.g. A3):
    python -m src.step_3._2.histogram_descriptors.compare_shapes -d A3 -o A3_comparison.pdf

    # Specify custom 3 shapes (2 similar + 1 different) and save all PDFs:
    python -m src.step_3._2.histogram_descriptors.compare_shapes Mug/D00747.obj Mug/m507.obj Sword/D00812.obj -pdf --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import vedo

import src.helper.silence_vtk as _silence_vtk_mod
from src.step_3._2.histogram_descriptors.histogram_descriptors import (
    DEFAULT_SEED,
    calculate_angle_histogram,
    calculate_area_triangle,
    calculate_distance_from_barycenter,
    calculate_distance_two_vertices,
    calculate_volume_tetrahedron,
)

_silence_vtk_mod.silence_vtk()

# Embed TrueType fonts in PDF output for LaTeX / Overleaf compatibility
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

DESCRIPTORS: list[str] = ["A3", "D1", "D2", "D3", "D4"]

DESCRIPTOR_TITLES: dict[str, str] = {
    "A3": "A3: Angle Between 3 Random Points",
    "D1": "D1: Distance from Barycenter",
    "D2": "D2: Distance Between 2 Random Points",
    "D3": r"D3: $\sqrt{\mathrm{Area}}$ of Random Triangle (3 pts)",
    "D4": r"D4: $\sqrt[3]{\mathrm{Volume}}$ of Random Tetrahedron (4 pts)",
}

DESCRIPTOR_XLABELS: dict[str, str] = {
    "A3": r"Normalized Angle ($\mathrm{rad} / \pi$)",
    "D1": "Normalized Distance from Barycenter",
    "D2": r"Normalized Pairwise Distance ($d / \sqrt{3}$)",
    "D3": r"Normalized $\sqrt{\mathrm{Triangle\ Area}}$",
    "D4": r"Normalized $\sqrt[3]{\mathrm{Tetrahedron\ Volume}}$",
}

DEFAULT_SHAPES: list[str] = [
    "Guitar/D00374.obj",
    "Guitar/D00655.obj",
    "Mug/m507.obj",
]

DEFAULT_CSV_CANDIDATES: list[Path] = [
    Path("3d_models_final/analysis/4_descriptors/histogram_descriptors.csv"),
    Path("analysis_output/descriptors/histogram_new_sampling_descriptors.csv"),
    Path("analysis_output/descriptors/histogram_descriptors.csv"),
]

DEFAULT_MODEL_ROOTS: list[Path] = [
    Path("3d_models_final/3d_models_steps/3_models_fully_normalized"),
    Path("3d_models"),
]

# Distinct visual styling for [Similar 1, Similar 2, Different]
SHAPE_STYLES: list[dict[str, object]] = [
    {
        "role": "Similar 1",
        "color": "#1f77b4",  # Royal blue
        "mesh_color": "#5b9bd5",
        "linestyle": "-",
        "linewidth": 2.5,
        "alpha_fill": 0.14,
    },
    {
        "role": "Similar 2",
        "color": "#009e73",  # Bluish green / teal
        "mesh_color": "#4db6ac",
        "linestyle": "--",
        "linewidth": 2.5,
        "alpha_fill": 0.14,
    },
    {
        "role": "Different",
        "color": "#d62728",  # Crimson red
        "mesh_color": "#e57373",
        "linestyle": "-",
        "linewidth": 2.5,
        "alpha_fill": 0.14,
    },
]


def _find_default_csv() -> Path | None:
    for candidate in DEFAULT_CSV_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _resolve_mesh_path(
    shape_id: str,
    model_roots: list[Path] | None = None,
) -> Path:
    """Resolve a shape identifier (e.g. 'Mug/D00747.obj' or full path) to an existing .obj file."""
    direct = Path(shape_id)
    if direct.is_file():
        return direct

    roots = model_roots or DEFAULT_MODEL_ROOTS
    norm_id = shape_id.replace("\\", "/").strip("/")
    parts = norm_id.split("/")
    rel_two = Path(*parts[-2:]) if len(parts) >= 2 else Path(parts[-1])

    for root in roots:
        candidate = root / rel_two
        if candidate.is_file():
            return candidate

    filename = parts[-1]
    if not filename.endswith(".obj"):
        filename += ".obj"
    for root in roots:
        if root.is_dir():
            matches = list(root.rglob(filename))
            if len(matches) == 1:
                return matches[0]

    raise FileNotFoundError(
        f"Could not find mesh file for '{shape_id}'. Checked {[str(r) for r in roots]}."
    )


def _normalize_id(path_str: str) -> str:
    norm = str(path_str).replace("\\", "/").strip("/")
    parts = norm.split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return norm


def _extract_histograms_from_csv(
    df: pd.DataFrame,
    shape_id: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract (histogram, bin_centers) for all 5 descriptors from a precomputed CSV."""
    target = _normalize_id(shape_id)
    norm_col = df["file_path"].astype(str).str.replace("\\", "/", regex=False).str.strip("/")

    matches = df[norm_col == target]
    if matches.empty:
        filename = target.split("/")[-1]
        matches = df[norm_col.str.endswith("/" + filename) | (norm_col == filename)]

    if matches.empty:
        raise KeyError(f"Shape '{shape_id}' not found in histogram CSV.")

    row = matches.iloc[0]
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for desc in DESCRIPTORS:
        pattern = re.compile(rf"^{re.escape(desc)}_(\d+)$")
        bin_cols: list[tuple[int, str]] = []
        for col in df.columns:
            m = pattern.match(str(col))
            if m:
                bin_cols.append((int(m.group(1)), str(col)))
        bin_cols.sort(key=lambda item: item[0])
        values = np.array([float(row[col]) for _, col in bin_cols], dtype=np.float64)
        n_bins = len(values)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        result[desc] = (values, centers)

    return result


def _compute_histograms_from_mesh(
    mesh_path: Path,
    n_samples: int = 100_000,
    n_bins: int = 64,
    seed: int = DEFAULT_SEED,
    normalize_by_sum: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute all 5 histogram descriptors directly from a 3D mesh."""
    mesh = vedo.Mesh(str(mesh_path))
    return {
        "A3": calculate_angle_histogram(
            mesh, n_samples=n_samples, n_bins_histogram=n_bins, seed=seed, normalize_by_sum=normalize_by_sum
        ),
        "D1": calculate_distance_from_barycenter(
            mesh, n_bins_histogram=n_bins, seed=seed, normalize_by_sum=normalize_by_sum
        ),
        "D2": calculate_distance_two_vertices(
            mesh, n_samples=n_samples, n_bins_histogram=n_bins, seed=seed, normalize_by_sum=normalize_by_sum
        ),
        "D3": calculate_area_triangle(
            mesh, n_samples=n_samples, n_bins_histogram=n_bins, seed=seed, normalize_by_sum=normalize_by_sum
        ),
        "D4": calculate_volume_tetrahedron(
            mesh, n_samples=n_samples, n_bins_histogram=n_bins, seed=seed, normalize_by_sum=normalize_by_sum
        ),
    }


def _load_or_compute_histograms(
    shape_ids: list[str],
    csv_path: Path | None = None,
    model_roots: list[Path] | None = None,
    compute_on_fly: bool = False,
    n_samples: int = 100_000,
    n_bins: int = 64,
    seed: int = DEFAULT_SEED,
    normalize_by_sum: bool = False,
) -> tuple[list[str], list[dict[str, tuple[np.ndarray, np.ndarray]]]]:
    if len(shape_ids) != 3:
        raise ValueError("Please provide exactly 3 shapes (2 similar shapes and 1 different shape).")

    df = None
    if not compute_on_fly:
        resolved_csv = csv_path or _find_default_csv()
        if resolved_csv is not None and resolved_csv.is_file():
            df = pd.read_csv(resolved_csv)

    short_labels = [_normalize_id(sid) for sid in shape_ids]
    histograms_per_shape: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []

    for sid in shape_ids:
        if df is not None and not compute_on_fly:
            try:
                hists = _extract_histograms_from_csv(df, sid)
                if normalize_by_sum:
                    hists = {
                        k: (v / np.sum(v), c) for k, (v, c) in hists.items()
                    }
                histograms_per_shape.append(hists)
                continue
            except KeyError:
                pass
        mpath = _resolve_mesh_path(sid, model_roots=model_roots)
        histograms_per_shape.append(
            _compute_histograms_from_mesh(
                mpath,
                n_samples=n_samples,
                n_bins=n_bins,
                seed=seed,
                normalize_by_sum=normalize_by_sum,
            )
        )

    return short_labels, histograms_per_shape


def find_best_trio(
    df: pd.DataFrame,
    similar_class: str | None = None,
    different_class: str | None = None,
) -> list[str]:
    """Automatically select 2 similar shapes (same class) and 1 different shape (other class)."""
    work = df.copy()
    work["class"] = work["file_path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]
    bins = 64
    cdfs: dict[str, np.ndarray] = {}
    for d in DESCRIPTORS:
        cols = [f"{d}_{i}" for i in range(bins)]
        mat = work[cols].to_numpy(dtype=float)
        mat_norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-12)
        cdfs[d] = np.cumsum(mat_norm, axis=1)[:, :-1]

    candidate_classes = [similar_class] if similar_class else sorted(work["class"].unique())
    best_pair = None

    for cls in candidate_classes:
        grp = work[work["class"].str.lower() == cls.lower()]
        if len(grp) < 2:
            continue
        idxs = grp.index.to_numpy()
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                i, j = idxs[a_i], idxs[b_i]
                emds = [float(np.sum(np.abs(cdfs[d][i] - cdfs[d][j])) / bins) for d in DESCRIPTORS]
                if min(emds) > 0.003 and max(emds) < 0.025:
                    score = max(emds)
                    if best_pair is None or score < best_pair[0]:
                        best_pair = (score, work.loc[i, "class"], i, j, emds)

    if best_pair is None:
        raise ValueError(f"Could not find a suitable similar pair for class '{similar_class}'.")

    _, chosen_sim_cls, i, j, _ = best_pair
    if different_class:
        other_mask = work["class"].str.lower() == different_class.lower()
    else:
        other_mask = work["class"] != chosen_sim_cls

    other_idxs = work.index[other_mask].to_numpy()
    if len(other_idxs) == 0:
        raise ValueError(f"No shapes found for different class '{different_class}'.")

    min_diff_emd = np.full(len(other_idxs), np.inf)
    for d in DESCRIPTORS:
        emd_ik = np.sum(np.abs(cdfs[d][other_idxs] - cdfs[d][i]), axis=1) / bins
        emd_jk = np.sum(np.abs(cdfs[d][other_idxs] - cdfs[d][j]), axis=1) / bins
        min_diff_emd = np.minimum(min_diff_emd, np.minimum(emd_ik, emd_jk))

    k = int(other_idxs[int(np.argmax(min_diff_emd))])
    return [
        str(work.loc[i, "file_path"]),
        str(work.loc[j, "file_path"]),
        str(work.loc[k, "file_path"]),
    ]


def render_mesh_image(
    mesh_path: Path,
    mesh_color: str = "lightsteelblue",
    size: tuple[int, int] = (750, 750),
) -> np.ndarray:
    """Render a 3D mesh off-screen and crop excess white margins into a square image."""
    mesh = vedo.Mesh(str(mesh_path))
    center = mesh.center_of_mass()
    mesh.shift(-center[0], -center[1], -center[2])
    mesh.color(mesh_color).lighting("default")

    plotter = vedo.Plotter(offscreen=True, size=size, axes=0, bg="white")
    plotter.show(mesh, interactive=False, resetcam=True)
    plotter.camera.Azimuth(35)
    plotter.camera.Elevation(22)
    plotter.reset_camera()
    plotter.camera.Zoom(1.15)
    plotter.render()

    image = plotter.screenshot(asarray=True)
    plotter.close()

    if image is not None and image.ndim == 3:
        non_white = np.any(image[:, :, :3] < 250, axis=2)
        if np.any(non_white):
            ys, xs = np.where(non_white)
            pad = 28
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(image.shape[0], int(ys.max()) + pad)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(image.shape[1], int(xs.max()) + pad)
            h, w = y1 - y0, x1 - x0
            side = max(h, w)
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            y0 = max(0, cy - side // 2)
            y1 = min(image.shape[0], y0 + side)
            x0 = max(0, cx - side // 2)
            x1 = min(image.shape[1], x0 + side)
            image = image[y0:y1, x0:x1]

    return image


def plot_3d_models_comparison(
    shape_ids: list[str],
    model_roots: list[Path] | None = None,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    save_pdf: bool = False,
    show: bool = True,
) -> plt.Figure:
    """Create a standalone 1x3 figure of the 3 compared 3D meshes for LaTeX/Overleaf."""
    if len(shape_ids) != 3:
        raise ValueError("Please provide exactly 3 shapes (2 similar shapes and 1 different shape).")

    mesh_paths = [_resolve_mesh_path(sid, model_roots=model_roots) for sid in shape_ids]
    short_labels = [_normalize_id(sid) for sid in shape_ids]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), dpi=150, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.08, wspace=0.04)

    for idx, ax_mesh in enumerate(axes):
        style = SHAPE_STYLES[idx]
        img = render_mesh_image(mesh_paths[idx], mesh_color=str(style["mesh_color"]))
        ax_mesh.imshow(img)
        ax_mesh.set_xticks([])
        ax_mesh.set_yticks([])

        for spine in ax_mesh.spines.values():
            spine.set_visible(True)
            spine.set_color(str(style["color"]))
            spine.set_linewidth(3.0)
            if style["linestyle"] == "--":
                spine.set_linestyle("--")

        ax_mesh.set_title(
            f"{style['role']}: {short_labels[idx]}",
            fontsize=13.5,
            fontweight="bold",
            color=str(style["color"]),
            pad=8,
        )

    save_target: Path | None = None
    if output_path is not None:
        base = Path(output_path)
        if base.suffix:
            save_target = base
        else:
            save_target = base / "shapes_3d_comparison.pdf"
    elif output_dir is not None or save_pdf:
        target_dir = Path(output_dir) if output_dir is not None else Path(".")
        save_target = target_dir / "shapes_3d_comparison.pdf"

    if save_target is not None:
        save_target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, bbox_inches="tight", dpi=200)
        print(f"Saved 3D models comparison figure to: {save_target}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _draw_single_descriptor_on_ax(
    ax: plt.Axes,
    desc: str,
    short_labels: list[str],
    histograms_per_shape: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    normalize_by_sum: bool = False,
    show_legend: bool = True,
) -> tuple[list, list[str]]:
    """Plot a single descriptor's 3-shape histogram comparison onto a Matplotlib Axes."""
    y_label = "Probability (Sum = 1)" if normalize_by_sum else "Probability Density"
    max_y = 0.0
    handles = []
    labels = []

    for idx in range(3):
        style = SHAPE_STYLES[idx]
        values, centers = histograms_per_shape[idx][desc]
        max_y = max(max_y, float(np.max(values)))
        label_str = f"{style['role']}: {short_labels[idx]}"
        (line_handle,) = ax.plot(
            centers,
            values,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=float(style["linewidth"]),
            label=label_str,
        )
        ax.fill_between(
            centers,
            values,
            color=str(style["color"]),
            alpha=float(style["alpha_fill"]),
        )
        handles.append(line_handle)
        labels.append(label_str)

    ax.set_title(DESCRIPTOR_TITLES[desc], fontsize=14, fontweight="bold", pad=8)
    ax.set_xlabel(DESCRIPTOR_XLABELS[desc], fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max_y * 1.15 if max_y > 0 else 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)

    if show_legend:
        ax.legend(loc="best", fontsize=10.5, framealpha=0.92, edgecolor="#cccccc")

    return handles, labels


def plot_individual_descriptor_figures(
    shape_ids: list[str],
    descriptors: list[str] | None = None,
    csv_path: Path | None = None,
    model_roots: list[Path] | None = None,
    compute_on_fly: bool = False,
    n_samples: int = 100_000,
    n_bins: int = 64,
    seed: int = DEFAULT_SEED,
    normalize_by_sum: bool = False,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    save_pdf: bool = False,
    show: bool = True,
    show_legend: bool = True,
) -> dict[str, plt.Figure]:
    """Create separate individual Matplotlib figures for each requested descriptor (A3, D1, D2, D3, D4)."""
    target_descriptors = descriptors or DESCRIPTORS
    short_labels, histograms_per_shape = _load_or_compute_histograms(
        shape_ids=shape_ids,
        csv_path=csv_path,
        model_roots=model_roots,
        compute_on_fly=compute_on_fly,
        n_samples=n_samples,
        n_bins=n_bins,
        seed=seed,
        normalize_by_sum=normalize_by_sum,
    )

    figures: dict[str, plt.Figure] = {}
    for desc in target_descriptors:
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150, constrained_layout=True)
        _draw_single_descriptor_on_ax(
            ax=ax,
            desc=desc,
            short_labels=short_labels,
            histograms_per_shape=histograms_per_shape,
            normalize_by_sum=normalize_by_sum,
            show_legend=show_legend,
        )
        figures[desc] = fig

        save_target: Path | None = None
        if output_path is not None and len(target_descriptors) == 1:
            save_target = Path(output_path)
        elif output_path is not None and len(target_descriptors) > 1:
            base = Path(output_path)
            if base.suffix:
                save_target = base.with_name(f"{base.stem}_{desc}{base.suffix}")
            else:
                save_target = base / f"{desc}_comparison.pdf"
        elif output_dir is not None or save_pdf:
            target_dir = Path(output_dir) if output_dir is not None else Path(".")
            save_target = target_dir / f"{desc}_comparison.pdf"

        if save_target is not None:
            save_target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_target, bbox_inches="tight", dpi=200)
            print(f"Saved {desc} figure to: {save_target}")

        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return figures


def plot_shape_histogram_comparison(
    shape_ids: list[str],
    csv_path: Path | None = None,
    model_roots: list[Path] | None = None,
    compute_on_fly: bool = False,
    n_samples: int = 100_000,
    n_bins: int = 64,
    seed: int = DEFAULT_SEED,
    normalize_by_sum: bool = False,
    output_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create the combined figure showing all 5 histogram descriptors (A3, D1, D2, D3, D4) on one canvas."""
    short_labels, histograms_per_shape = _load_or_compute_histograms(
        shape_ids=shape_ids,
        csv_path=csv_path,
        model_roots=model_roots,
        compute_on_fly=compute_on_fly,
        n_samples=n_samples,
        n_bins=n_bins,
        seed=seed,
        normalize_by_sum=normalize_by_sum,
    )

    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = GridSpec(2, 6, figure=fig, hspace=0.36, wspace=0.55)

    fig.suptitle(
        "Histogram Descriptors Comparison (2 Similar Shapes vs. 1 Different Shape)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    subplot_specs = [
        ("A3", gs[0, 0:2]),
        ("D1", gs[0, 2:4]),
        ("D2", gs[0, 4:6]),
        ("D3", gs[1, 1:3]),
        ("D4", gs[1, 3:5]),
    ]

    legend_handles = []
    legend_labels = []

    for desc, spec in subplot_specs:
        ax = fig.add_subplot(spec)
        handles, labels = _draw_single_descriptor_on_ax(
            ax=ax,
            desc=desc,
            short_labels=short_labels,
            histograms_per_shape=histograms_per_shape,
            normalize_by_sum=normalize_by_sum,
            show_legend=False,
        )
        if desc == "A3":
            legend_handles = handles
            legend_labels = labels

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        fontsize=11.5,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    fig.subplots_adjust(top=0.83, bottom=0.08, left=0.06, right=0.98)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        print(f"Saved combined figure to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot histogram descriptors (A3, D1, D2, D3, D4) and 3D model comparisons "
            "for 2 similar shapes and 1 different shape as LaTeX/Overleaf PDFs."
        )
    )
    parser.add_argument(
        "shapes",
        nargs="*",
        help=(
            "Three shape paths or identifiers: [similar_1, similar_2, different]. "
            f"Defaults to {DEFAULT_SHAPES}."
        ),
    )
    parser.add_argument(
        "--compare",
        nargs=3,
        metavar=("SIMILAR_1", "SIMILAR_2", "DIFFERENT"),
        help="Alternative flag to specify the 3 shapes to compare.",
    )
    parser.add_argument(
        "-d",
        "--descriptor",
        type=str,
        choices=["A3", "D1", "D2", "D3", "D4", "models", "all"],
        default="all",
        help="Which figure(s) to plot: A3, D1, D2, D3, D4, 'models' (3D models only), or 'all' (default: all).",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Plot/save only the 3D models comparison figure (shapes_3d_comparison.pdf).",
    )
    parser.add_argument(
        "-pdf",
        "--pdf",
        action="store_true",
        help="Save each plotted descriptor and the 3D models comparison as individual PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the individual PDF files (e.g. --output-dir ./overleaf_figs).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (or base path when plotting multiple figures).",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Plot all 5 descriptors together in a single combined grid instead of separate figures.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Omit the legend from individual descriptor plots.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to precomputed histogram_descriptors.csv (auto-detected if omitted).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Root directory of 3D .obj models (used when rendering 3D models or computing on the fly).",
    )
    parser.add_argument(
        "--similar-class",
        type=str,
        default=None,
        help="Automatically pick 2 similar shapes from this class (e.g. Guitar, Mug, Drum).",
    )
    parser.add_argument(
        "--different-class",
        type=str,
        default=None,
        help="Automatically pick 1 different shape from this class (e.g. Sword, Vase, Gun).",
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Recompute histograms directly from .obj meshes instead of reading from CSV.",
    )
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=100_000,
        help="Number of random surface samples when computing on the fly (default: 100,000).",
    )
    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        default=64,
        help="Number of histogram bins when computing on the fly (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed when computing on the fly (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--normalize-by-sum",
        "--normalize_by_sum",
        action="store_true",
        default=False,
        help="Normalize histogram bins so their sum equals 1 instead of area integral = 1.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive matplotlib windows (useful when exporting PDFs).",
    )

    args = parser.parse_args()

    if args.compare:
        chosen_shapes = list(args.compare)
    elif args.shapes:
        if len(args.shapes) != 3:
            parser.error("Please provide exactly 3 shapes: <similar_1> <similar_2> <different>.")
        chosen_shapes = list(args.shapes)
    elif args.similar_class or args.different_class:
        csv_file = args.csv or _find_default_csv()
        if csv_file is None or not csv_file.is_file():
            parser.error("Cannot auto-select shapes by class without a histogram_descriptors.csv file.")
        df = pd.read_csv(csv_file)
        chosen_shapes = find_best_trio(
            df,
            similar_class=args.similar_class,
            different_class=args.different_class,
        )
        print(f"Auto-selected shapes: {chosen_shapes}")
    else:
        chosen_shapes = list(DEFAULT_SHAPES)

    roots = [args.models_dir] + DEFAULT_MODEL_ROOTS if args.models_dir else None

    if args.models_only or args.descriptor == "models":
        plot_3d_models_comparison(
            shape_ids=chosen_shapes,
            model_roots=roots,
            output_dir=args.output_dir,
            output_path=args.output,
            save_pdf=args.pdf,
            show=not args.no_show,
        )
        return

    if args.combined:
        out_path = args.output
        if out_path is None and args.pdf:
            out_path = Path("histogram_descriptors_combined.pdf")
        plot_shape_histogram_comparison(
            shape_ids=chosen_shapes,
            csv_path=args.csv,
            model_roots=roots,
            compute_on_fly=args.compute,
            n_samples=args.samples,
            n_bins=args.bins,
            seed=args.seed,
            normalize_by_sum=args.normalize_by_sum,
            output_path=out_path,
            show=not args.no_show,
        )
        return

    if args.descriptor == "all":
        # Also generate the 3D models comparison figure alongside the 5 descriptor figures
        models_out = None
        if args.output is not None:
            base = Path(args.output)
            if base.suffix:
                models_out = base.with_name(f"{base.stem}_models{base.suffix}")
        plot_3d_models_comparison(
            shape_ids=chosen_shapes,
            model_roots=roots,
            output_dir=args.output_dir,
            output_path=models_out,
            save_pdf=args.pdf,
            show=not args.no_show,
        )

    selected_descs = DESCRIPTORS if args.descriptor == "all" else [args.descriptor]
    plot_individual_descriptor_figures(
        shape_ids=chosen_shapes,
        descriptors=selected_descs,
        csv_path=args.csv,
        model_roots=roots,
        compute_on_fly=args.compute,
        n_samples=args.samples,
        n_bins=args.bins,
        seed=args.seed,
        normalize_by_sum=args.normalize_by_sum,
        output_dir=args.output_dir,
        output_path=args.output,
        save_pdf=args.pdf,
        show=not args.no_show,
        show_legend=not args.no_legend,
    )


if __name__ == "__main__":
    main()
