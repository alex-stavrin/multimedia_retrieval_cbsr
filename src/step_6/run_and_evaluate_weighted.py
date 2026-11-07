import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.step_4.distance_calculations.calculate_total_distance import (
    ShapeDistanceCalculator,
    DISTANCE_CONFIG_EMD,
    DISTANCE_CONFIG_L2,
    DISTANCE_CONFIG_COSINE,
)
from src.step_6.retrieval_evalulator import RetrievalEvaluator
from src.step_6.roc_evaluator import ROCEvaluator

CONFIG_MAP = {
    "emd": DISTANCE_CONFIG_EMD,
    "l2": DISTANCE_CONFIG_L2,
    "cosine": DISTANCE_CONFIG_COSINE,
}


def run_and_evaluate_weighted(
    elementary_csv: str | Path,
    histogram_csv: str | Path,
    weights_csv: str | Path,
    output_dir: str | Path,
    config: str = "emd",
    k: int = 10,
    exclude_self: bool = True,
    eval_roc: bool = True,
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if config not in CONFIG_MAP:
        raise ValueError(
            f"Unknown config '{config}'. Choose from: {list(CONFIG_MAP.keys())}"
        )

    distance_config = CONFIG_MAP[config]
    calculator = ShapeDistanceCalculator(
        csv_databases=[str(elementary_csv), str(histogram_csv), str(weights_csv)],
        distance_config=distance_config,
    )

    ground_truth = {
        sid: sid.replace("\\", "/").split("/")[0]
        for sid in calculator.shape_identifiers
    }
    evaluator = RetrievalEvaluator(ground_truth=ground_truth)

    results: dict[str, list[str]] = {}
    full_rankings: dict[str, list[str]] = {}
    print(
        f"Calculating weighted distances ({config}) for {len(calculator.shape_identifiers)} shapes..."
    )
    for shape_id in tqdm(calculator.shape_identifiers, desc="Evaluating Queries"):
        total_distances = calculator.calculate_total_distance(
            query=shape_id,
            run_pipeline=False,
        )

        ranked = [
            sid
            for sid in total_distances.keys()
            if not (exclude_self and sid == shape_id)
        ]
        results[shape_id] = ranked[:k]
        full_rankings[shape_id] = ranked

    eval_results = evaluator.evaluate(
        results=results,
        method_name=f"Weighted ({config}, k={k})",
        full_rankings=full_rankings,
    )

    if eval_roc:
        roc_eval = ROCEvaluator(ground_truth=ground_truth)
        roc_res = roc_eval.evaluate_method(
            full_rankings, method_name=f"Weighted ({config.upper()})"
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

        ROCEvaluator.save_roc_results(
            roc_res, out_path, prefix=f"weighted_{config}_k{k}"
        )
        ROCEvaluator.plot_roc_curves(
            [roc_res],
            out_path / f"weighted_{config}_k{k}_roc.png",
            title=f"ROC Curve - Weighted ({config.upper()})",
        )

    per_class_file = out_path / f"weighted_{config}_k{k}_per_class.csv"
    queries_file = out_path / f"weighted_{config}_k{k}_queries.csv"
    confusion_file = out_path / f"weighted_{config}_k{k}_confusion_matrix.csv"
    total_stats_file = out_path / f"weighted_{config}_k{k}_total_stats.json"

    eval_results["per_class"].to_csv(per_class_file, index=False)
    eval_results["queries"].to_csv(queries_file, index=False)
    eval_results["confusion_matrix"].to_csv(confusion_file)

    with open(total_stats_file, "w", encoding="utf-8") as f:
        json.dump(eval_results["total_stats"], f, indent=2)

    print("=" * 60)
    print(f"Weighted Evaluation ({config}, k={k})")
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
        description="Run and evaluate weighted distance retrieval using ShapeDistanceCalculator and RetrievalEvaluator."
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
        "--weights_csv",
        type=str,
        default="analysis_output/distances/weights_emd_descriptors.csv",
        help="Path to descriptor weights CSV.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="analysis_output/evaluation/weighted",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        choices=["emd", "l2", "cosine"],
        default="emd",
        help="Distance metric configuration (default: emd).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve (default: 10).",
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

    run_and_evaluate_weighted(
        elementary_csv=args.elementary_csv,
        histogram_csv=args.histogram_csv,
        weights_csv=args.weights_csv,
        output_dir=args.output_dir,
        config=args.config,
        k=args.k,
        exclude_self=not args.include_self,
        eval_roc=args.eval_roc,
    )


if __name__ == "__main__":
    main()
