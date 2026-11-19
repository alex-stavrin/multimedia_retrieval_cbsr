import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.step_4.user_interface.class_colors import get_class_color
from src.step_5.ANN.table_renderer import find_mesh_path, render_mesh_thumbnail


METHODS_SPEC = [
    (
        "KNN (k=6)",
        "knn",
        "knn/knn_k6_per_class.csv",
        "knn/knn_k6_auroc_queries.csv",
        "knn/knn_k6_roc.json",
    ),
    (
        "RNN (r=0.85)",
        "rnn",
        "rnn/rnn_r0.85_per_class.csv",
        "rnn/rnn_r0.85_auroc_queries.csv",
        "rnn/rnn_r0.85_roc.json",
    ),
    (
        "Step 4: Weighted (EMD, k=6)",
        "step4_weighted",
        "weighted/weighted_emd_k6_per_class.csv",
        "weighted/weighted_emd_k6_auroc_queries.csv",
        "weighted/weighted_emd_k6_roc.json",
    ),
]


def load_evaluation_data(eval_root: Path) -> list[dict]:
    loaded = []
    for label, key, pc_rel, q_rel, roc_rel in METHODS_SPEC:
        pc_path = eval_root / pc_rel
        q_path = eval_root / q_rel
        roc_path = eval_root / roc_rel

        with open(roc_path, "r", encoding="utf-8") as f:
            roc = json.load(f)
        pc_df = pd.read_csv(pc_path)
        q_df = pd.read_csv(q_path)

        # Sort classes by mean AUROC descending
        pc_sorted = (
            pc_df[["class", "number_of_queries", "auroc"]]
            .sort_values(["auroc", "class"], ascending=[False, True])
            .reset_index(drop=True)
        )
        pc_sorted["rank"] = pc_sorted.index + 1

        # Top 10 best and 10 worst individual shapes
        best_10 = (
            q_df.sort_values(["auroc", "query_id"], ascending=[False, True])
            .head(10)
            .reset_index(drop=True)
        )
        best_10["rank"] = best_10.index + 1

        worst_10 = (
            q_df.sort_values(["auroc", "query_id"], ascending=[True, True])
            .head(10)
            .reset_index(drop=True)
        )
        worst_10["rank"] = worst_10.index + 1

        loaded.append(
            {
                "label": label,
                "key": key,
                "macro_auroc": float(roc["macro_auroc"]),
                "micro_auroc": float(roc["micro_auroc"]),
                "classes": pc_sorted,
                "best_shapes": best_10,
                "worst_shapes": worst_10,
            }
        )
    return loaded


def _save_fig(fig: plt.Figure, out_base: Path) -> tuple[Path, Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {png_path.resolve()}")
    print(f"Saved: {pdf_path.resolve()}")
    return png_path, pdf_path


def plot_all_classes_ranking(data: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """
    3-column horizontal bar chart showing all 69 classes ranked from Best (#1)
    to Worst (#69) by Mean AUROC for KNN, RNN, and Step 4 (Weighted EMD).
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 19), dpi=300, sharex=True)
    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0.45, vmax=1.0)

    for idx, m in enumerate(data):
        ax = axes[idx]
        df = m["classes"]
        n = len(df)
        y_pos = np.arange(n)

        # Invert so Rank #1 is at the top
        vals = df["auroc"].to_numpy()[::-1]
        labels = [
            f"#{int(r['rank']):02d}  {r['class']} (n={int(r['number_of_queries'])})"
            for _, r in df.iloc[::-1].iterrows()
        ]
        colors = [cmap(norm(v)) for v in vals]

        bars = ax.barh(
            y_pos,
            vals,
            color=colors,
            edgecolor="#334155",
            linewidth=0.45,
            height=0.76,
            alpha=0.92,
        )

        # Macro AUROC reference line & random baseline
        ax.axvline(
            m["macro_auroc"],
            color="#1E293B",
            linestyle="--",
            linewidth=1.4,
            label=f"Macro AUROC = {m['macro_auroc']:.4f}",
        )
        ax.axvline(
            0.5,
            color="#94A3B8",
            linestyle=":",
            linewidth=1.1,
            label="Random (0.50)",
        )

        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.008,
                bar.get_y() + bar.get_height() / 2.0,
                f"{v:.4f}",
                va="center",
                ha="left",
                fontsize=7.2,
                fontweight="bold",
                color="#0F172A",
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7.8, fontweight="normal", color="#0F172A")
        ax.set_xlim(0.0, 1.14)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlabel("Class Mean AUROC", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{m['label']}\nMacro AUROC: {m['macro_auroc']:.4f}",
            fontsize=12.5,
            fontweight="bold",
            color="#0F172A",
            pad=10,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Complete Ranking of All 69 Shape Classes by Mean AUROC (Best to Worst) — KNN, RNN, Step 4",
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    return _save_fig(fig, out_dir / "auroc_class_rankings_all_methods")


def plot_top_bottom_classes(data: list[dict], out_dir: Path, k: int = 10) -> tuple[Path, Path]:
    """
    2x3 grid of horizontal bar charts showing the Top 10 Best and Top 10 Worst
    classes by Mean AUROC for each method (KNN, RNN, Step 4).
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10.5), dpi=300, sharex=True)

    for col_idx, m in enumerate(data):
        df = m["classes"]
        best_df = df.head(k).iloc[::-1]  # #1 at top
        worst_df = df.tail(k).sort_values("auroc", ascending=True).iloc[::-1]  # #1 worst at top

        # Top row: 10 Best Classes
        ax_top = axes[0, col_idx]
        y_top = np.arange(len(best_df))
        vals_top = best_df["auroc"].to_numpy()
        labels_top = [
            f"#{int(r['rank'])}  {r['class']}" for _, r in best_df.iterrows()
        ]
        bars_top = ax_top.barh(
            y_top,
            vals_top,
            color="#10B981",
            edgecolor="#065F46",
            linewidth=0.7,
            height=0.7,
            alpha=0.9,
        )
        for bar, v in zip(bars_top, vals_top):
            ax_top.text(
                v + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{v:.4f}",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#065F46",
            )
        ax_top.axvline(
            m["macro_auroc"],
            color="#1E293B",
            linestyle="--",
            linewidth=1.2,
            label=f"Macro AUROC ({m['macro_auroc']:.3f})",
        )
        ax_top.set_yticks(y_top)
        ax_top.set_yticklabels(labels_top, fontsize=9, fontweight="bold", color="#0F172A")
        ax_top.set_xlim(0.0, 1.13)
        ax_top.set_title(
            f"{m['label']} — 10 Best Classes",
            fontsize=11.5,
            fontweight="bold",
            color="#065F46",
        )
        ax_top.grid(axis="x", linestyle=":", alpha=0.6)
        ax_top.legend(loc="lower right", fontsize=8)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        # Bottom row: 10 Worst Classes
        ax_bot = axes[1, col_idx]
        y_bot = np.arange(len(worst_df))
        vals_bot = worst_df["auroc"].to_numpy()
        labels_bot = [
            f"#{int(r['rank'])}  {r['class']}" for _, r in worst_df.iterrows()
        ]
        bars_bot = ax_bot.barh(
            y_bot,
            vals_bot,
            color="#EF4444",
            edgecolor="#991B1B",
            linewidth=0.7,
            height=0.7,
            alpha=0.88,
        )
        for bar, v in zip(bars_bot, vals_bot):
            ax_bot.text(
                v + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{v:.4f}",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#991B1B",
            )
        ax_bot.axvline(
            0.5,
            color="#64748B",
            linestyle=":",
            linewidth=1.2,
            label="Random (0.500)",
        )
        ax_bot.axvline(
            m["macro_auroc"],
            color="#1E293B",
            linestyle="--",
            linewidth=1.2,
            label=f"Macro AUROC ({m['macro_auroc']:.3f})",
        )
        ax_bot.set_yticks(y_bot)
        ax_bot.set_yticklabels(labels_bot, fontsize=9, fontweight="bold", color="#0F172A")
        ax_bot.set_xlim(0.0, 1.15)
        ax_bot.set_xlabel("Class Mean AUROC", fontsize=10.5, fontweight="bold")
        ax_bot.set_title(
            f"{m['label']} — 10 Worst Classes",
            fontsize=11.5,
            fontweight="bold",
            color="#991B1B",
        )
        ax_bot.grid(axis="x", linestyle=":", alpha=0.6)
        ax_bot.legend(loc="upper right", fontsize=8, framealpha=0.92)
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)

    fig.suptitle(
        "Top 10 Best vs. Top 10 Worst Performing Classes by Mean AUROC Across All Methods",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    return _save_fig(fig, out_dir / "auroc_top10_best_worst_classes")


def plot_top_bottom_shapes(data: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """
    2x3 grid of horizontal bar charts showing the 10 Best and 10 Worst individual
    shapes by AUROC for KNN, RNN, and Step 4 (Weighted EMD).
    """
    fig, axes = plt.subplots(2, 3, figsize=(21, 11), dpi=300, sharex=True)

    for col_idx, m in enumerate(data):
        best_df = m["best_shapes"].iloc[::-1]  # #1 at top
        worst_df = m["worst_shapes"].iloc[::-1]  # #1 worst at top

        # Top row: 10 Best Shapes
        ax_top = axes[0, col_idx]
        y_top = np.arange(len(best_df))
        vals_top = best_df["auroc"].to_numpy()
        labels_top = [
            f"#{int(r['rank'])}  {r['query_id']}" for _, r in best_df.iterrows()
        ]
        bars_top = ax_top.barh(
            y_top,
            vals_top,
            color="#059669",
            edgecolor="#064E3B",
            linewidth=0.7,
            height=0.7,
            alpha=0.9,
        )
        for bar, v in zip(bars_top, vals_top):
            ax_top.text(
                v + 0.008,
                bar.get_y() + bar.get_height() / 2.0,
                f"{v:.4f}",
                va="center",
                ha="left",
                fontsize=8.3,
                fontweight="bold",
                color="#064E3B",
            )
        ax_top.set_yticks(y_top)
        ax_top.set_yticklabels(labels_top, fontsize=8.5, fontweight="bold", color="#0F172A")
        ax_top.set_xlim(0.0, 1.14)
        ax_top.set_title(
            f"{m['label']} — 10 Best Shapes",
            fontsize=11.5,
            fontweight="bold",
            color="#064E3B",
        )
        ax_top.grid(axis="x", linestyle=":", alpha=0.6)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        # Bottom row: 10 Worst Shapes
        ax_bot = axes[1, col_idx]
        y_bot = np.arange(len(worst_df))
        vals_bot = worst_df["auroc"].to_numpy()
        labels_bot = [
            f"#{int(r['rank'])}  {r['query_id']}" for _, r in worst_df.iterrows()
        ]
        bars_bot = ax_bot.barh(
            y_bot,
            vals_bot,
            color="#DC2626",
            edgecolor="#7F1D1D",
            linewidth=0.7,
            height=0.7,
            alpha=0.88,
        )
        for bar, v in zip(bars_bot, vals_bot):
            ax_bot.text(
                v + 0.008,
                bar.get_y() + bar.get_height() / 2.0,
                f"{v:.4f}",
                va="center",
                ha="left",
                fontsize=8.3,
                fontweight="bold",
                color="#7F1D1D",
            )
        ax_bot.axvline(
            0.5,
            color="#64748B",
            linestyle=":",
            linewidth=1.2,
            label="Random (0.500)",
        )
        ax_bot.set_yticks(y_bot)
        ax_bot.set_yticklabels(labels_bot, fontsize=8.5, fontweight="bold", color="#0F172A")
        ax_bot.set_xlim(0.0, 1.14)
        ax_bot.set_xlabel("Shape Query AUROC", fontsize=10.5, fontweight="bold")
        ax_bot.set_title(
            f"{m['label']} — 10 Worst Shapes",
            fontsize=11.5,
            fontweight="bold",
            color="#7F1D1D",
        )
        ax_bot.grid(axis="x", linestyle=":", alpha=0.6)
        ax_bot.legend(loc="lower right", fontsize=8)
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)

    fig.suptitle(
        "Top 10 Best vs. Top 10 Worst Performing Individual Shapes by AUROC Across All Methods",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    return _save_fig(fig, out_dir / "auroc_top10_best_worst_shapes")


SHORT_ROW_LABELS = {
    "knn": "KNN\n(k=6)",
    "rnn": "RNN\n(r=0.85)",
    "step4_weighted": "Step 4\n(Weighted)",
}


def _add_auroc_cell_caption(
    ax: Any,
    class_name: str,
    model_name: str,
    auroc: float,
    is_best: bool = True,
) -> None:
    pill_color = get_class_color(class_name)
    pill_fontsize = 7.0 if len(class_name) > 16 else 7.8
    ax.text(
        0.5,
        -0.11,
        class_name,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=pill_fontsize,
        fontweight="bold",
        color="#000000",
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.45",
            "facecolor": pill_color,
            "edgecolor": "#666666",
            "linewidth": 0.5,
        },
        clip_on=False,
    )
    ax.text(
        0.5,
        -0.21,
        model_name,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#0F172A",
        clip_on=False,
    )
    metric_color = "#065F46" if is_best else "#991B1B"
    ax.text(
        0.5,
        -0.31,
        f"AUROC={auroc:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=metric_color,
        clip_on=False,
    )


def _render_auroc_shapes_grid(
    rows_spec: list[dict[str, Any]],
    model_dir: Path,
    out_base: Path,
    title: str,
    cell_size: float = 2.2,
) -> tuple[Path, Path]:
    num_rows = len(rows_spec)
    num_cols = max(len(r["items"]) for r in rows_spec)

    row_height = cell_size * 1.34
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(num_cols * cell_size, num_rows * row_height),
        gridspec_kw={"wspace": 0.36, "hspace": 0.46},
        dpi=300,
    )

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r_idx, row_entry in enumerate(rows_spec):
        row_label = row_entry["row_label"]
        is_best = row_entry.get("is_best", True)
        items = row_entry["items"]

        for c_idx in range(num_cols):
            ax = axes[r_idx, c_idx]
            if r_idx == 0:
                ax.set_title(
                    f"{c_idx + 1:02d}",
                    fontsize=11,
                    fontweight="bold",
                    pad=8,
                    color="#0F172A",
                )

            if c_idx == 0 and row_label:
                ax.set_ylabel(
                    row_label,
                    fontsize=11,
                    fontweight="bold",
                    labelpad=12,
                    color="#0F172A",
                )

            if c_idx < len(items):
                item = items[c_idx]
                mesh_path = find_mesh_path(model_dir, item["query_id"])
                img = render_mesh_thumbnail(mesh_path)
                cls_name = mesh_path.parent.name
                model_name = mesh_path.name

                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                _add_auroc_cell_caption(
                    ax,
                    class_name=cls_name,
                    model_name=model_name,
                    auroc=float(item["auroc"]),
                    is_best=is_best,
                )
                for spine in ax.spines.values():
                    spine.set_color("black")
                    spine.set_linewidth(1.0)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.8)

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=0.98 if num_rows <= 3 else 0.96,
    )
    return _save_fig(fig, out_base)


def plot_rendered_shapes_tables(
    data: list[dict],
    model_dir: Path,
    out_dir: Path,
) -> dict[str, tuple[Path, Path]]:
    """
    Renders 3D shape thumbnail tables (with class color pill, shape filename,
    and AUROC score) for the 10 Best and 10 Worst performing shapes across
    KNN, RNN, and Step 4 (Weighted EMD).
    """
    best_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []

    for m in data:
        short_label = SHORT_ROW_LABELS.get(m["key"], m["label"])
        best_items = m["best_shapes"].to_dict(orient="records")
        worst_items = m["worst_shapes"].to_dict(orient="records")

        best_rows.append(
            {
                "row_label": short_label,
                "is_best": True,
                "items": best_items,
            }
        )
        worst_rows.append(
            {
                "row_label": short_label,
                "is_best": False,
                "items": worst_items,
            }
        )

    for m in data:
        short_label = SHORT_ROW_LABELS.get(m["key"], m["label"]).replace("\n", " ")
        combined_rows.append(
            {
                "row_label": f"{short_label}\n(10 Best)",
                "is_best": True,
                "items": m["best_shapes"].to_dict(orient="records"),
            }
        )
    for m in data:
        short_label = SHORT_ROW_LABELS.get(m["key"], m["label"]).replace("\n", " ")
        combined_rows.append(
            {
                "row_label": f"{short_label}\n(10 Worst)",
                "is_best": False,
                "items": m["worst_shapes"].to_dict(orient="records"),
            }
        )

    best_paths = _render_auroc_shapes_grid(
        rows_spec=best_rows,
        model_dir=model_dir,
        out_base=out_dir / "auroc_top10_best_shapes_table",
        title="Top 10 Best Performing Shapes by AUROC — KNN, RNN, Step 4",
    )
    worst_paths = _render_auroc_shapes_grid(
        rows_spec=worst_rows,
        model_dir=model_dir,
        out_base=out_dir / "auroc_top10_worst_shapes_table",
        title="Top 10 Worst Performing Shapes by AUROC — KNN, RNN, Step 4",
    )
    combined_paths = _render_auroc_shapes_grid(
        rows_spec=combined_rows,
        model_dir=model_dir,
        out_base=out_dir / "auroc_top10_best_worst_shapes_table",
        title="Top 10 Best & 10 Worst Performing Shapes by AUROC — KNN, RNN, Step 4",
    )
    return {
        "best_table": best_paths,
        "worst_table": worst_paths,
        "combined_table": combined_paths,
    }


def main() -> None:
    default_eval = Path("3d_models_final/evaluation")
    if not default_eval.is_dir():
        default_eval = Path("analysis_output/evaluation")

    default_model_dir = Path("3d_models_final/3d_models_steps/3_models_fully_normalized")
    if not default_model_dir.is_dir():
        default_model_dir = Path("resampled_models")

    parser = argparse.ArgumentParser(
        description="Plot class mean AUROC rankings and 10 best / 10 worst shapes by AUROC across KNN, RNN, and Step 4."
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=str(default_eval),
        help="Root evaluation directory containing knn/, rnn/, and weighted/ subdirectories.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(default_model_dir),
        help="Path to 3D models directory for rendering shape thumbnails.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save generated plots (defaults to <eval_dir>/comparison).",
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (eval_root / "comparison")

    data = load_evaluation_data(eval_root)
    plot_all_classes_ranking(data, out_dir)
    plot_top_bottom_classes(data, out_dir)
    plot_top_bottom_shapes(data, out_dir)
    if model_dir.is_dir():
        plot_rendered_shapes_tables(data, model_dir, out_dir)
    else:
        print(f"[WARN] Model directory not found ({model_dir}); skipping 3D shape render tables.")


if __name__ == "__main__":
    main()
