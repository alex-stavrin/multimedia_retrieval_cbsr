import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.step_6.run_and_evaluate_all import (
    METRIC_COLUMNS,
    compute_ranking_table,
)

# Human-readable labels for Macro-Averaged retrieval metrics.
MACRO_METRIC_LABELS: dict[str, str] = {
    "macro_precision": "Precision",
    "macro_recall": "Recall (Sensitivity)",
    "macro_specificity": "Specificity",
    "macro_accuracy": "Accuracy",
    "macro_f1": "F1 Score",
    "macro_map": "MAP",
    "macro_last_rank": "Last Rank",
    "macro_first_tier": "First Tier",
    "macro_second_tier": "Second Tier",
    "macro_is_first_correct_accuracy": "Top-1 Accuracy",
    "macro_auroc": "AUROC",
}

HEADER_BG = "#1E293B"
HEADER_FG = "#FFFFFF"
SECTION_BG = "#E2E8F0"
SECTION_FG = "#0F172A"
WIN_BG = "#D1FAE5"
WIN_FG = "#065F46"
ROW_EVEN_BG = "#FFFFFF"
ROW_ODD_BG = "#F8FAFC"
SUMMARY_BG = "#F1F5F9"
BORDER_COLOR = "#CBD5E1"
TEXT_COLOR = "#1E293B"


def _wrap_method_name(name: str, max_len: int = 12) -> str:
    """Insert a clean line break for method labels such as 'KNN (k=6, ...)'."""
    if len(name) <= max_len:
        return name
    if " (" in name:
        prefix, rest = name.split(" (", 1)
        return f"{prefix}\n({rest}"
    return name


def build_table_rows(
    comparison_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build macro-only metric rows and summary rows sorted left-to-right by average rank."""
    macro_cols = [
        c
        for c in METRIC_COLUMNS
        if c.startswith("macro_") and c in comparison_df.columns
    ]
    if not macro_cols:
        raise ValueError("No macro_* metric columns found in comparison CSV.")

    macro_df = comparison_df[["method"] + macro_cols].copy()
    ranking_df = compute_ranking_table(macro_df)

    ordered_methods = ranking_df["method"].tolist()
    comp_by_method = macro_df.set_index("method")
    rank_by_method = ranking_df.set_index("method")

    rows: list[dict[str, Any]] = []
    rows.append({"kind": "section", "label": "Macro-Averaged Retrieval Metrics"})

    for col in macro_cols:
        label = MACRO_METRIC_LABELS.get(col, col)
        rank_col = f"{col}_rank"
        cells = []
        for method in ordered_methods:
            val = float(comp_by_method.loc[method, col])
            rk = int(rank_by_method.loc[method, rank_col])
            cells.append(
                {
                    "text": f"{val:.4f}",
                    "is_winner": rk == 1,
                }
            )
        rows.append({"kind": "metric", "label": label, "cells": cells})

    rows.append({"kind": "section", "label": "System Ranking Summary"})

    avg_vals = [float(rank_by_method.loc[m, "average_rank"]) for m in ordered_methods]
    best_avg = min(avg_vals)
    rows.append(
        {
            "kind": "summary",
            "is_overall": True,
            "label": "Average Rank",
            "cells": [
                {
                    "text": f"{val:.2f}",
                    "is_winner": abs(val - best_avg) < 1e-12,
                }
                for val in avg_vals
            ],
        }
    )

    return rows, ordered_methods


def export_ranking_table_pdf(
    comparison_csv: str | Path,
    output_pdf: str | Path | None = None,
    title: str = "Retrieval Methods Average-Rank Aggregation",
    save_png: bool = False,
) -> Path:
    """Render the macro comparison and system ranking table as a formatted PDF."""
    comp_path = Path(comparison_csv)
    if not comp_path.is_file():
        raise FileNotFoundError(f"Comparison CSV not found: {comp_path}")

    comparison_df = pd.read_csv(comp_path)
    if comparison_df.empty:
        raise ValueError(f"Comparison CSV is empty: {comp_path}")

    rows, ordered_methods = build_table_rows(comparison_df)

    if output_pdf is None:
        out_path = comp_path.parent / "all_methods_ranking_table.pdf"
    else:
        out_path = Path(output_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_methods = len(ordered_methods)
    fig_width = max(9.0, 3.6 + 2.1 * num_methods)
    fig_height = max(5.8, 1.4 + 0.34 * len(rows))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_position([0.02, 0.04, 0.96, 0.85])
    ax.axis("off")

    fig.text(
        0.5,
        0.95,
        title,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    col_labels = ["Metric"] + [_wrap_method_name(m) for m in ordered_methods]

    metric_col_w = 0.28
    method_col_w = (1.0 - metric_col_w) / max(1, num_methods)
    col_widths = [metric_col_w] + [method_col_w] * num_methods

    cell_text: list[list[str]] = []
    for row in rows:
        if row["kind"] == "section":
            cell_text.append([row["label"]] + [""] * num_methods)
        else:
            cell_text.append([row["label"]] + [c["text"] for c in row["cells"]])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)

    # Style header cells (row 0)
    for col_idx in range(len(col_labels)):
        cell = table[0, col_idx]
        cell.set_facecolor(HEADER_BG)
        cell.set_edgecolor(HEADER_BG)
        cell.set_linewidth(0.8)
        cell.set_height(0.08)
        txt = cell.get_text()
        txt.set_color(HEADER_FG)
        txt.set_fontweight("bold")
        txt.set_fontsize(9.0)
        if col_idx == 0:
            txt.set_ha("left")
            cell.PAD = 0.04

    # Style body cells (rows 1..N)
    metric_stripe = 0
    for r_idx, row in enumerate(rows, start=1):
        kind = row["kind"]

        if kind == "section":
            metric_stripe = 0
            for c_idx in range(len(col_labels)):
                cell = table[r_idx, c_idx]
                cell.set_facecolor(SECTION_BG)
                cell.set_edgecolor(SECTION_BG)
                cell.set_linewidth(0.5)
                cell.set_height(0.05)
                txt = cell.get_text()
                txt.set_color(SECTION_FG)
                txt.set_fontweight("bold")
                txt.set_fontsize(9.0)
                if c_idx == 0:
                    txt.set_ha("left")
                    cell.PAD = 0.04
            continue

        is_summary = kind == "summary"
        is_overall = bool(row.get("is_overall", False))
        base_bg = (
            SUMMARY_BG
            if is_summary
            else (ROW_EVEN_BG if metric_stripe % 2 == 0 else ROW_ODD_BG)
        )
        if not is_summary:
            metric_stripe += 1

        row_height = 0.055 if is_overall else (0.05 if is_summary else 0.048)

        # Metric label column
        label_cell = table[r_idx, 0]
        label_cell.set_facecolor(base_bg)
        label_cell.set_edgecolor(BORDER_COLOR)
        label_cell.set_linewidth(0.5 if not is_overall else 0.9)
        label_cell.set_height(row_height)
        label_txt = label_cell.get_text()
        label_txt.set_fontsize(8.8 if is_summary else 8.5)
        label_txt.set_color(TEXT_COLOR)
        label_txt.set_ha("left")
        label_cell.PAD = 0.04
        if is_summary:
            label_txt.set_fontweight("bold")

        # Method value cells
        for m_idx, cell_spec in enumerate(row["cells"]):
            c_idx = 1 + m_idx
            cell = table[r_idx, c_idx]
            is_win = cell_spec["is_winner"]
            bg = WIN_BG if is_win else base_bg
            fg = WIN_FG if is_win else TEXT_COLOR

            cell.set_facecolor(bg)
            cell.set_edgecolor(BORDER_COLOR)
            cell.set_linewidth(0.5 if not is_overall else 0.9)
            cell.set_height(row_height)
            txt = cell.get_text()
            txt.set_color(fg)
            txt.set_fontsize(8.8 if is_summary else 8.5)
            txt.set_fontweight("bold" if (is_win or is_overall) else "normal")

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    print(f"Saved ranking table PDF to: {out_path.resolve()}")

    if save_png:
        png_path = out_path.with_suffix(".png")
        plt.savefig(png_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
        print(f"Saved ranking table PNG to: {png_path.resolve()}")

    plt.close(fig)
    return out_path


def main() -> None:
    default_comp = Path(
        "3d_models_final/evaluation/comparison/all_methods_comparison_table.csv"
    )
    if not default_comp.is_file():
        default_comp = Path(
            "analysis_output/evaluation/comparison/all_methods_comparison_table.csv"
        )

    parser = argparse.ArgumentParser(
        description="Export a formatted PDF ranking table highlighting metric winners in green."
    )
    parser.add_argument(
        "--comparison_csv",
        type=str,
        default=str(default_comp),
        help="Path to all_methods_comparison_table.csv.",
    )
    parser.add_argument(
        "-o",
        "--output_pdf",
        type=str,
        default=None,
        help="Output PDF path (defaults to all_methods_ranking_table.pdf alongside comparison_csv).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Retrieval Methods Average-Rank Aggregation",
        help="Title shown at the top of the PDF table.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also save a sibling PNG preview next to the PDF.",
    )
    args = parser.parse_args()

    export_ranking_table_pdf(
        comparison_csv=args.comparison_csv,
        output_pdf=args.output_pdf,
        title=args.title,
        save_png=args.png,
    )


if __name__ == "__main__":
    main()
