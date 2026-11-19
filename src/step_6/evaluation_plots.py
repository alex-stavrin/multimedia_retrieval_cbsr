from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_plot(
    output_path: str | Path,
    label: str,
    save_pdf: bool = False,
) -> None:
    """Save the active matplotlib figure to output_path and optionally a sibling PDF."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    print(f"Saved {label} to: {out_file.resolve()}")
    if save_pdf and out_file.suffix.lower() != ".pdf":
        pdf_file = out_file.with_suffix(".pdf")
        plt.savefig(pdf_file)
        print(f"Saved {label} (PDF) to: {pdf_file.resolve()}")
    plt.close()


def plot_roc_curves(
    evaluated_methods: list[dict],
    output_path: str | Path,
    title: str = "ROC Curves (Sensitivity vs Specificity)",
    save_pdf: bool = False,
) -> None:
    plt.figure(figsize=(9, 7), dpi=300)

    plt.plot(
        [0.0, 1.0],
        [1.0, 0.0],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Random Choice (AUROC = 0.500)",
    )

    linestyles = ["-", "--", "-.", ":"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for idx, eval_res in enumerate(evaluated_methods):
        name = eval_res["method"]
        auc = eval_res["macro_auroc"]
        sens = eval_res["curve"]["sensitivity"]
        spec = eval_res["curve"]["specificity"]
        color = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]

        plt.plot(
            sens,
            spec,
            linewidth=2.2,
            linestyle=ls,
            color=color,
            label=f"{name} (AUROC = {auc:.4f})",
        )

    plt.xlim([-0.02, 1.02])  # type: ignore
    plt.ylim([-0.02, 1.02])  # type: ignore
    plt.xlabel("Sensitivity", fontsize=12)
    plt.ylabel("Specificity", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    _save_plot(output_path, "ROC plot", save_pdf=save_pdf)


def plot_metrics_comparison(
    all_stats: list[dict],
    output_path: str | Path,
    title: str = "Comparison of Retrieval Methods Across Metrics",
    save_pdf: bool = False,
) -> None:
    """Plots a grouped bar chart comparing all methods across key retrieval metrics."""
    metrics_to_plot = [
        ("macro_precision", "Precision"),
        ("macro_recall", "Recall (Sens)"),
        ("macro_specificity", "Specificity"),
        ("macro_accuracy", "Accuracy"),
        ("macro_f1", "F1 Score"),
        ("macro_map", "MAP"),
        ("macro_last_rank", "Last Rank"),
        ("macro_first_tier", "1st Tier"),
        ("macro_second_tier", "2nd Tier"),
        ("macro_auroc", "AUROC"),
        ("macro_is_first_correct_accuracy", "Top-1 Acc"),
    ]

    available_metrics = [
        (key, label)
        for key, label in metrics_to_plot
        if all(key in stat for stat in all_stats)
    ]

    x = np.arange(len(available_metrics))
    num_methods = max(1, len(all_stats))
    width = min(0.8 / num_methods, 0.22)

    plt.figure(figsize=(14, 6.5), dpi=300)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for idx, stat in enumerate(all_stats):
        method_name = stat.get("method", f"Method {idx + 1}")
        values = [float(stat[key]) for key, _ in available_metrics]
        offset = (idx - (num_methods - 1) / 2) * width
        bars = plt.bar(
            x + offset,
            values,
            width=width,
            label=method_name,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90 if num_methods > 3 else 0,
            )

    plt.xticks(
        x,
        [label for _, label in available_metrics],
        rotation=20,
        ha="right",
        fontsize=10,
    )
    plt.ylim([0.0, 1.12])  # type: ignore
    plt.ylabel("Score [0, 1]", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    _save_plot(output_path, "metrics comparison plot", save_pdf=save_pdf)


def plot_time_comparison(
    all_stats: list[dict],
    output_path: str | Path,
    title: str = "Mean Retrieval Time per Query Across Methods",
    save_pdf: bool = False,
) -> None:
    """Plots a bar chart comparing mean per-query retrieval time (ms) across methods."""
    methods = [
        stat.get("method", f"Method {idx + 1}") for idx, stat in enumerate(all_stats)
    ]
    mean_query_ms = [float(stat.get("mean_query_time_ms", 0.0)) for stat in all_stats]

    x = np.arange(len(methods))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]

    plt.figure(figsize=(8.5, 6), dpi=300)
    bars = plt.bar(
        x,
        mean_query_ms,
        width=0.5,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.3f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x, methods, rotation=20, ha="right", fontsize=10)
    if len(methods) == 1:
        plt.xlim([-0.7, 0.7])  # type: ignore
    plt.ylabel("Mean Query Time (ms / query)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    if max(mean_query_ms, default=0.0) > 0:
        plt.ylim([0, max(mean_query_ms) * 1.18])  # type: ignore
    plt.tight_layout()

    _save_plot(output_path, "time comparison plot", save_pdf=save_pdf)
