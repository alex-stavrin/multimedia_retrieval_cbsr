import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.step_6.roc_evaluator import ROCEvaluator
from src.step_6.run_and_evaluate_knn import run_and_evaluate_knn
from src.step_6.run_and_evaluate_rnn import run_and_evaluate_rnn
from src.step_6.run_and_evaluate_weighted import run_and_evaluate_weighted


METRIC_COLUMNS = [
    "macro_precision",
    "macro_recall",
    "macro_specificity",
    "macro_accuracy",
    "macro_f1",
    "macro_map",
    "macro_last_rank",
    "macro_first_tier",
    "macro_second_tier",
    "macro_is_first_correct_accuracy",
    "macro_auroc",
    "micro_precision",
    "micro_recall",
    "micro_specificity",
    "micro_accuracy",
    "micro_f1",
    "micro_map",
    "micro_last_rank",
    "micro_first_tier",
    "micro_second_tier",
    "micro_is_first_correct_accuracy",
    "micro_auroc",
]


def _try_load_existing(stats_file: Path, roc_file: Path) -> tuple[dict, dict] | None:
    if stats_file.is_file() and roc_file.is_file():
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        roc = ROCEvaluator.load_roc_results(roc_file)
        if "macro_auroc" not in stats:
            stats["macro_auroc"] = roc["macro_auroc"]
        if "micro_auroc" not in stats:
            stats["micro_auroc"] = roc["micro_auroc"]
        return stats, roc
    return None


def compute_ranking_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes integer ranks (1 = best/highest score) per metric and average rank
    per row, following Slide 31 of Module 7.
    """
    rank_df = pd.DataFrame()
    rank_df["method"] = comparison_df["method"]

    present_metrics = [col for col in METRIC_COLUMNS if col in comparison_df.columns]
    for col in present_metrics:
        # Higher score is better -> ascending=False, method="min"
        rank_df[f"{col}_rank"] = (
            comparison_df[col].rank(ascending=False, method="min").astype(int)
        )

    rank_cols = [f"{col}_rank" for col in present_metrics]
    rank_df["average_rank"] = rank_df[rank_cols].mean(axis=1)
    rank_df = rank_df.sort_values("average_rank", ascending=True).reset_index(drop=True)
    return rank_df


def run_and_evaluate_all(
    elementary_csv: str | Path,
    histogram_csv: str | Path,
    weights_csv: str | Path | None = None,
    output_dir: str | Path = "analysis_output/evaluation",
    methods: list[str] | None = None,
    weighted_configs: list[str] | None = None,
    knn_k: int = 6,
    knn_p: float = 2.0,
    knn_eps: float = 0.0,
    rnn_radius: float = 0.85,
    rnn_p: float = 2.0,
    rnn_eps: float = 0.0,
    weighted_k: int = 6,
    exclude_self: bool = True,
    reuse_existing: bool = False,
) -> dict:
    if methods is None:
        methods = ["knn", "rnn", "weighted"]
    if weighted_configs is None:
        weighted_configs = ["emd"]

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    comparison_dir = out_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_total_stats: list[dict] = []
    all_roc_results: list[dict] = []

    # 1. kNN Evaluation
    if "knn" in methods:
        knn_dir = out_root / "knn"
        stats_file = knn_dir / f"knn_k{knn_k}_total_stats.json"
        roc_file = knn_dir / f"knn_k{knn_k}_roc.json"

        loaded = _try_load_existing(stats_file, roc_file) if reuse_existing else None
        if loaded is not None:
            print(f"[REUSE] Loaded existing kNN evaluation from {knn_dir}")
            stats, roc = loaded
        else:
            knn_eval = run_and_evaluate_knn(
                elementary_csv=elementary_csv,
                histogram_csv=histogram_csv,
                output_dir=knn_dir,
                k=knn_k,
                p=knn_p,
                eps=knn_eps,
                exclude_self=exclude_self,
                eval_roc=True,
            )
            stats = knn_eval["total_stats"]
            roc = knn_eval["roc"]

        all_total_stats.append(stats)
        all_roc_results.append(roc)

    # 2. rNN Evaluation
    if "rnn" in methods:
        rnn_dir = out_root / "rnn"
        stats_file = rnn_dir / f"rnn_r{rnn_radius}_total_stats.json"
        roc_file = rnn_dir / f"rnn_r{rnn_radius}_roc.json"

        loaded = _try_load_existing(stats_file, roc_file) if reuse_existing else None
        if loaded is not None:
            print(f"[REUSE] Loaded existing rNN evaluation from {rnn_dir}")
            stats, roc = loaded
        else:
            rnn_eval = run_and_evaluate_rnn(
                elementary_csv=elementary_csv,
                histogram_csv=histogram_csv,
                output_dir=rnn_dir,
                radius=rnn_radius,
                p=rnn_p,
                eps=rnn_eps,
                exclude_self=exclude_self,
                eval_roc=True,
            )
            stats = rnn_eval["total_stats"]
            roc = rnn_eval["roc"]

        all_total_stats.append(stats)
        all_roc_results.append(roc)

    # 3. Weighted Evaluation(s)
    if "weighted" in methods:
        resolved_weights: Path | None = None
        if weights_csv is not None and Path(weights_csv).is_file():
            resolved_weights = Path(weights_csv)
        else:
            for candidate in [
                Path("analysis_output/distances/weights_emd_hist_sampling_descriptors.csv"),
                Path("analysis_output/distances/weights_emd_descriptors.csv"),
                Path("analysis_output/distances/weights_equal.csv"),
            ]:
                if candidate.is_file():
                    resolved_weights = candidate
                    break

        if resolved_weights is None or not resolved_weights.is_file():
            print(
                "[WARN] Skipping 'weighted' evaluation: no valid weights CSV found."
            )
        else:
            weighted_dir = out_root / "weighted"
            for cfg in weighted_configs:
                stats_file = weighted_dir / f"weighted_{cfg}_k{weighted_k}_total_stats.json"
                roc_file = weighted_dir / f"weighted_{cfg}_k{weighted_k}_roc.json"

                loaded = (
                    _try_load_existing(stats_file, roc_file)
                    if reuse_existing
                    else None
                )
                if loaded is not None:
                    print(
                        f"[REUSE] Loaded existing Weighted ({cfg}) evaluation from {weighted_dir}"
                    )
                    stats, roc = loaded
                else:
                    w_eval = run_and_evaluate_weighted(
                        elementary_csv=elementary_csv,
                        histogram_csv=histogram_csv,
                        weights_csv=resolved_weights,
                        output_dir=weighted_dir,
                        config=cfg,
                        k=weighted_k,
                        exclude_self=exclude_self,
                        eval_roc=True,
                    )
                    stats = w_eval["total_stats"]
                    roc = w_eval["roc"]

                all_total_stats.append(stats)
                all_roc_results.append(roc)

    if not all_total_stats:
        raise RuntimeError("No retrieval methods were evaluated.")

    # 4. Build Combined Comparison Table (Slide 30) & Ranking Table (Slide 31)
    comparison_df = pd.DataFrame(all_total_stats)
    ordered_cols = ["method", "total_queries"] + [
        c for c in METRIC_COLUMNS if c in comparison_df.columns
    ]
    comparison_df = comparison_df[ordered_cols]

    ranking_df = compute_ranking_table(comparison_df)

    comparison_csv = comparison_dir / "all_methods_comparison_table.csv"
    comparison_json = comparison_dir / "all_methods_comparison_table.json"
    ranking_csv = comparison_dir / "all_methods_ranking_table.csv"

    comparison_df.to_csv(comparison_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(all_total_stats, f, indent=2)

    # 5. Plot Combined ROC Curve (Slide 26) & Combined Metrics Bar Chart
    roc_plot_path = comparison_dir / "all_methods_roc_comparison.png"
    ROCEvaluator.plot_roc_curves(
        all_roc_results,
        output_path=roc_plot_path,
        title="ROC Curves Comparison - All Retrieval Methods",
    )

    metrics_plot_path = comparison_dir / "all_methods_metrics_comparison.png"
    ROCEvaluator.plot_metrics_comparison(
        all_total_stats,
        output_path=metrics_plot_path,
        title="Metrics Comparison - All Retrieval Methods",
    )

    # 6. Print Comparison & Ranking Tables to Console
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + "=" * 95)
    print("COMBINED METRICS COMPARISON TABLE (Slide 30)")
    print("=" * 95)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 95)
    print("SYSTEM RANKING TABLE (1 = Best, Slide 31)")
    print("=" * 95)
    print(ranking_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=" * 95)
    print(f"Saved combined tables and plots to: {comparison_dir.resolve()}")

    return {
        "comparison_table": comparison_df,
        "ranking_table": ranking_df,
        "roc_results": all_roc_results,
        "roc_plot": roc_plot_path,
        "metrics_plot": metrics_plot_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and evaluate all retrieval methods (kNN, rNN, Weighted), compare metrics in one table, and plot combined ROC curves."
    )
    parser.add_argument(
        "--elementary_csv",
        type=str,
        default="analysis_output/descriptors/simple_descriptors_normalized.csv",
        help="Path to normalized elementary descriptors CSV.",
    )
    parser.add_argument(
        "--histogram_csv",
        type=str,
        default="analysis_output/descriptors/histogram_descriptors.csv",
        help="Path to histogram descriptors CSV.",
    )
    parser.add_argument(
        "--weights_csv",
        type=str,
        default="analysis_output/distances/weights_emd_hist_sampling_descriptors.csv",
        help="Path to descriptor weights CSV for weighted retrieval.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="analysis_output/evaluation",
        help="Root directory to save individual and combined evaluation results.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["knn", "rnn", "weighted"],
        default=["knn", "rnn", "weighted"],
        help="Which retrieval methods to evaluate and compare (default: knn rnn weighted).",
    )
    parser.add_argument(
        "--weighted_configs",
        nargs="+",
        choices=["emd", "l2", "cosine"],
        default=["emd"],
        help="Weighted distance configurations to include (e.g. --weighted_configs emd l2 cosine).",
    )
    parser.add_argument(
        "-k",
        "--knn_k",
        type=int,
        default=6,
        help="k for kNN evaluation (default: 6).",
    )
    parser.add_argument(
        "--knn_p",
        type=float,
        default=2.0,
        help="Minkowski p for kNN (default: 2.0).",
    )
    parser.add_argument(
        "--knn_eps",
        type=float,
        default=0.0,
        help="KD-Tree eps for kNN (default: 0.0).",
    )
    parser.add_argument(
        "-r",
        "--rnn_radius",
        type=float,
        default=0.85,
        help="Radius r for rNN evaluation (default: 0.85).",
    )
    parser.add_argument(
        "--rnn_p",
        type=float,
        default=2.0,
        help="Minkowski p for rNN (default: 2.0).",
    )
    parser.add_argument(
        "--rnn_eps",
        type=float,
        default=0.0,
        help="KD-Tree eps for rNN (default: 0.0).",
    )
    parser.add_argument(
        "--weighted_k",
        type=int,
        default=6,
        help="k for weighted distance evaluation (default: 6).",
    )
    parser.add_argument(
        "--include_self",
        action="store_true",
        help="Include query shape itself in retrieved results (default: excluded).",
    )
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Reuse existing evaluation JSONs if already present in output_dir.",
    )

    args = parser.parse_args()
    run_and_evaluate_all(
        elementary_csv=args.elementary_csv,
        histogram_csv=args.histogram_csv,
        weights_csv=args.weights_csv,
        output_dir=args.output_dir,
        methods=args.methods,
        weighted_configs=args.weighted_configs,
        knn_k=args.knn_k,
        knn_p=args.knn_p,
        knn_eps=args.knn_eps,
        rnn_radius=args.rnn_radius,
        rnn_p=args.rnn_p,
        rnn_eps=args.rnn_eps,
        weighted_k=args.weighted_k,
        exclude_self=not args.include_self,
        reuse_existing=args.reuse_existing,
    )


if __name__ == "__main__":
    main()
