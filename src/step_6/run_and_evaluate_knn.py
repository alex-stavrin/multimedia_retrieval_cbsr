import argparse
import json
from pathlib import Path

from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine
from src.step_6.retrieval_evalulator import RetrievalEvaluator
from src.step_6.roc_evaluator import ROCEvaluator


def run_and_evaluate_knn(
    elementary_csv: str | Path,
    histogram_csv: str | Path,
    output_dir: str | Path,
    k: int = 6,
    p: float = 2.0,
    eps: float = 0.0,
    exclude_self: bool = True,
    eval_roc: bool = True,
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    spatial_search_engine = SpatialSearchEngine(
        elementary_csv=elementary_csv, histogram_csv=histogram_csv
    )

    # Build ground truth: shape_id -> class_name
    ground_truth = {
        sid: sid.replace("\\", "/").split("/")[0]
        for sid in spatial_search_engine.shape_ids
    }
    evaluator = RetrievalEvaluator(ground_truth=ground_truth)

    total_db_shapes = len(spatial_search_engine.shape_ids)
    results: dict[str, list[str]] = {}
    full_rankings: dict[str, list[str]] = {}

    for shape_id in spatial_search_engine.shape_ids:
        if eval_roc and eps == 0.0:
            full_matches = spatial_search_engine.query_knn(
                query=shape_id,
                k=total_db_shapes,
                eps=eps,
                p=p,
                exclude_self=exclude_self,
            )
            ranked_all = list(full_matches.keys())
            full_rankings[shape_id] = ranked_all
            results[shape_id] = ranked_all[:k]
        else:
            knn_matches = spatial_search_engine.query_knn(
                query=shape_id,
                k=k,
                eps=eps,
                p=p,
                exclude_self=exclude_self,
            )
            results[shape_id] = list(knn_matches.keys())
            if eval_roc:
                full_matches = spatial_search_engine.query_knn(
                    query=shape_id,
                    k=total_db_shapes,
                    eps=eps,
                    p=p,
                    exclude_self=exclude_self,
                )
                full_rankings[shape_id] = list(full_matches.keys())

    method_label = f"KNN (k={k}, p={p}, eps={eps})"
    eval_results = evaluator.evaluate(
        results=results,
        method_name=method_label,
        full_rankings=full_rankings if full_rankings else None,
    )

    if eval_roc:
        roc_eval = ROCEvaluator(ground_truth=ground_truth)
        roc_res = roc_eval.evaluate_method(
            full_rankings, method_name=f"kNN (p={p}, eps={eps})"
        )
        roc_res["operating_point"] = (
            eval_results["total_stats"]["macro_recall"],
            eval_results["total_stats"]["macro_specificity"],
            f"k={k}",
        )
        eval_results["total_stats"]["macro_auroc"] = roc_res["macro_auroc"]
        eval_results["total_stats"]["micro_auroc"] = roc_res["micro_auroc"]
        eval_results["per_class"]["auroc"] = eval_results["per_class"]["class"].map(
            roc_res["per_class_auroc"]
        )
        eval_results["roc"] = roc_res

        ROCEvaluator.save_roc_results(roc_res, out_path, prefix=f"knn_k{k}")
        ROCEvaluator.plot_roc_curves(
            [roc_res],
            out_path / f"knn_k{k}_roc.png",
            title=f"ROC Curve - kNN (p={p}, eps={eps})",
        )

    # Save all results to disk
    per_class_file = out_path / f"knn_k{k}_per_class.csv"
    queries_file = out_path / f"knn_k{k}_queries.csv"
    confusion_file = out_path / f"knn_k{k}_confusion_matrix.csv"
    total_stats_file = out_path / f"knn_k{k}_total_stats.json"
    eval_results["per_class"].to_csv(per_class_file, index=False)
    eval_results["queries"].to_csv(queries_file, index=False)
    eval_results["confusion_matrix"].to_csv(confusion_file)

    with open(total_stats_file, "w", encoding="utf-8") as f:
        json.dump(eval_results["total_stats"], f, indent=2)

    # Display summary statistics to console
    print("=" * 60)
    print(f"kNN Evaluation (k={k}, p={p}, eps={eps})")
    print("=" * 60)
    for key, val in eval_results["total_stats"].items():
        if isinstance(val, float):
            print(f"  {key:<26}: {val:.4f}")
        else:
            print(f"  {key:<26}: {val}")
    print("=" * 60)
    print(f"Saved results to: {out_path.resolve()}")
    return eval_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and evaluate kNN retrieval using SpatialSearchEngine and RetrievalEvaluator."
    )
    parser.add_argument(
        "--elementary_csv",
        type=str,
        default="analysis_output/descriptors/simple_descriptors_normalized.csv",
        help="Path to elementary/simple descriptors CSV.",
    )
    parser.add_argument(
        "--histogram_csv",
        type=str,
        default="analysis_output/descriptors/histogram_descriptors.csv",
        help="Path to histogram descriptors CSV.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="analysis_output/evaluation/knn",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=6,
        help="Number of nearest neighbors to retrieve (default: 6).",
    )
    parser.add_argument(
        "-p",
        type=float,
        default=2.0,
        help="Minkowski distance order (default: 2.0 = Euclidean).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Approximation factor for KD-Tree (default: 0.0 = exact).",
    )
    parser.add_argument(
        "--include_self",
        action="store_true",
        help="Include query shape itself in retrieved results (default: excluded).",
    )
    parser.add_argument(
        "--eval_roc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute full ROC curve, AUROC and plot (default: True).",
    )
    args = parser.parse_args()
    run_and_evaluate_knn(
        elementary_csv=args.elementary_csv,
        histogram_csv=args.histogram_csv,
        output_dir=args.output_dir,
        k=args.k,
        p=args.p,
        eps=args.eps,
        exclude_self=not args.include_self,
        eval_roc=args.eval_roc,
    )


if __name__ == "__main__":
    main()
