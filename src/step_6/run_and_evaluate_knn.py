from pathlib import Path
from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine
from src.step_6.retrieval_evalulator import RetrievalEvaluator
import json
import argparse

def run_and_evaluate_knn(
    elementary_csv: str,
    histogram_csv: str,
    output_dir: str,
    k: int = 10,
    p: float = 2.0,
    eps: float = 0.0,
    exclude_self : bool = True
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

    # Query KNN for every shape in the data base
    results : dict[str, list[str]] = {}
    for shape_id in spatial_search_engine.shape_ids:
        knn_matches = spatial_search_engine.query_knn(
            query=shape_id,
            k=k,
            eps=eps,
            p=p,
            exclude_self=exclude_self
        )
        results[shape_id] = list(knn_matches.keys())

    eval_results = evaluator.evaluate(
        results=results,
        method_name=f"KNN k={k}, p={p}, eps={eps}"
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
        default=10,
        help="Number of nearest neighbors to retrieve (default: 10).",
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
    args = parser.parse_args()
    run_and_evaluate_knn(
        elementary_csv=args.elementary_csv,
        histogram_csv=args.histogram_csv,
        output_dir=args.output_dir,
        k=args.k,
        p=args.p,
        eps=args.eps,
        exclude_self=not args.include_self,
    )
if __name__ == "__main__":
    main()