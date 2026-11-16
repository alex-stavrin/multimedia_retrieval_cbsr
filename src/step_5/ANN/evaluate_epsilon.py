import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine
from src.step_6.retrieval_evalulator import RetrievalEvaluator


DEFAULT_EPS_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

_WORKER_ENGINE: SpatialSearchEngine | None = None
_WORKER_EVALUATOR: RetrievalEvaluator | None = None


def _init_worker(elementary_csv: str, histogram_csv: str) -> None:
    global _WORKER_ENGINE, _WORKER_EVALUATOR
    _WORKER_ENGINE = SpatialSearchEngine(
        elementary_csv=elementary_csv,
        histogram_csv=histogram_csv,
    )
    ground_truth = {
        sid: sid.replace("\\", "/").split("/")[0] for sid in _WORKER_ENGINE.shape_ids
    }
    _WORKER_EVALUATOR = RetrievalEvaluator(ground_truth=ground_truth)
    # Warmup KD-Tree in this worker process
    for sid in _WORKER_ENGINE.shape_ids[:25]:
        _WORKER_ENGINE.query_knn(sid, k=6, eps=0.0, exclude_self=True)
        _WORKER_ENGINE.query_range(sid, radius=0.85, eps=0.0, exclude_self=True)


def _run_knn_for_eps(
    eps: float,
    k: int,
    p: float,
    n_repeats: int,
    exclude_self: bool,
) -> tuple[float, dict[str, list[str]], float, dict]:
    assert _WORKER_ENGINE is not None and _WORKER_EVALUATOR is not None
    shape_ids = _WORKER_ENGINE.shape_ids

    pass_times: list[float] = []
    results: dict[str, list[str]] = {}
    for r in range(n_repeats):
        t0 = time.perf_counter()
        current: dict[str, list[str]] = {}
        for sid in shape_ids:
            matches = _WORKER_ENGINE.query_knn(
                sid, k=k, eps=eps, p=p, exclude_self=exclude_self
            )
            current[sid] = list(matches.keys())
        pass_times.append(time.perf_counter() - t0)
        if r == 0:
            results = current

    elapsed_s = float(np.median(pass_times))
    eval_res = _WORKER_EVALUATOR.evaluate(
        results=results, method_name=f"KNN (k={k}, eps={eps})"
    )
    return eps, results, elapsed_s, eval_res["total_stats"]


def _run_rnn_for_eps(
    eps: float,
    radius: float,
    p: float,
    n_repeats: int,
    exclude_self: bool,
) -> tuple[float, dict[str, list[str]], float, dict]:
    assert _WORKER_ENGINE is not None and _WORKER_EVALUATOR is not None
    shape_ids = _WORKER_ENGINE.shape_ids

    pass_times: list[float] = []
    results: dict[str, list[str]] = {}
    for r in range(n_repeats):
        t0 = time.perf_counter()
        current: dict[str, list[str]] = {}
        for sid in shape_ids:
            matches = _WORKER_ENGINE.query_range(
                sid, radius=radius, eps=eps, p=p, exclude_self=exclude_self
            )
            current[sid] = list(matches.keys())
        pass_times.append(time.perf_counter() - t0)
        if r == 0:
            results = current

    elapsed_s = float(np.median(pass_times))
    eval_res = _WORKER_EVALUATOR.evaluate(
        results=results, method_name=f"RNN (r={radius}, eps={eps})"
    )
    return eps, results, elapsed_s, eval_res["total_stats"]


def run_parallel_benchmarks(
    elementary_csv: str | Path,
    histogram_csv: str | Path,
    eps_values: list[float],
    k: int = 6,
    radius: float = 0.85,
    p: float = 2.0,
    n_repeats: int = 1,
    workers: int = 4,
    exclude_self: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_eps = list(dict.fromkeys([0.0] + list(eps_values)))
    max_procs = max(1, min(workers, len(ordered_eps)))

    knn_raw: dict[float, tuple[dict[str, list[str]], float, dict]] = {}
    rnn_raw: dict[float, tuple[dict[str, list[str]], float, dict]] = {}

    total_tasks = len(ordered_eps) * 2
    with tqdm(
        total=total_tasks,
        desc="Evaluating KNN & RNN across ε (parallel processes)",
        unit="task",
        dynamic_ncols=True,
    ) as pbar:
        with ProcessPoolExecutor(
            max_workers=max_procs,
            initializer=_init_worker,
            initargs=(str(elementary_csv), str(histogram_csv)),
        ) as executor:
            knn_futures = {
                executor.submit(
                    _run_knn_for_eps, eps, k, p, n_repeats, exclude_self
                ): eps
                for eps in ordered_eps
            }
            for fut in as_completed(knn_futures):
                eps, results, elapsed_s, stats = fut.result()
                knn_raw[eps] = (results, elapsed_s, stats)
                pbar.set_postfix_str(f"done KNN eps={eps}")
                pbar.update(1)

            rnn_futures = {
                executor.submit(
                    _run_rnn_for_eps, eps, radius, p, n_repeats, exclude_self
                ): eps
                for eps in ordered_eps
            }
            for fut in as_completed(rnn_futures):
                eps, results, elapsed_s, stats = fut.result()
                rnn_raw[eps] = (results, elapsed_s, stats)
                pbar.set_postfix_str(f"done RNN eps={eps}")
                pbar.update(1)

    # Assemble KNN DataFrame
    exact_knn_results = knn_raw[0.0][0]
    shape_ids = list(exact_knn_results.keys())
    num_shapes = len(shape_ids)

    knn_rows = []
    for eps in eps_values:
        results, elapsed_s, stats = knn_raw[eps]
        mean_query_ms = (elapsed_s / num_shapes) * 1000.0
        overlaps = [
            len(set(exact_knn_results[sid]) & set(results[sid]))
            / len(exact_knn_results[sid])
            if exact_knn_results[sid]
            else 1.0
            for sid in shape_ids
        ]
        knn_rows.append(
            {
                "eps": float(eps),
                "total_time_ms": elapsed_s * 1000.0,
                "mean_query_ms": mean_query_ms,
                "exact_overlap": float(np.mean(overlaps)),
                "macro_precision": float(stats["macro_precision"]),
                "macro_recall": float(stats["macro_recall"]),
                "macro_accuracy": float(stats["macro_accuracy"]),
                "macro_f1": float(stats["macro_f1"]),
                "macro_map": float(stats["macro_map"]),
                "micro_precision": float(stats["micro_precision"]),
            }
        )
    knn_df = pd.DataFrame(knn_rows)
    base_knn_ms = (knn_raw[0.0][1] / num_shapes) * 1000.0
    knn_df["speedup"] = base_knn_ms / knn_df["mean_query_ms"]

    # Assemble RNN DataFrame
    exact_rnn_results = rnn_raw[0.0][0]
    rnn_rows = []
    for eps in eps_values:
        results, elapsed_s, stats = rnn_raw[eps]
        mean_query_ms = (elapsed_s / num_shapes) * 1000.0
        overlaps = []
        counts = []
        for sid in shape_ids:
            s_exact = set(exact_rnn_results[sid])
            s_approx = set(results[sid])
            counts.append(len(s_approx))
            if not s_exact and not s_approx:
                overlaps.append(1.0)
            elif not s_exact:
                overlaps.append(0.0)
            else:
                overlaps.append(len(s_exact & s_approx) / len(s_exact))

        rnn_rows.append(
            {
                "eps": float(eps),
                "total_time_ms": elapsed_s * 1000.0,
                "mean_query_ms": mean_query_ms,
                "mean_retrieved": float(np.mean(counts)),
                "exact_overlap": float(np.mean(overlaps)),
                "macro_precision": float(stats["macro_precision"]),
                "macro_recall": float(stats["macro_recall"]),
                "macro_accuracy": float(stats["macro_accuracy"]),
                "macro_f1": float(stats["macro_f1"]),
                "macro_map": float(stats["macro_map"]),
                "micro_precision": float(stats["micro_precision"]),
            }
        )
    rnn_df = pd.DataFrame(rnn_rows)
    base_rnn_ms = (rnn_raw[0.0][1] / num_shapes) * 1000.0
    rnn_df["speedup"] = base_rnn_ms / rnn_df["mean_query_ms"]

    return knn_df, rnn_df


def plot_epsilon_results(
    knn_df: pd.DataFrame,
    rnn_df: pd.DataFrame,
    output_dir: Path,
    k: int,
    radius: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)

    ax.plot(
        knn_df["eps"],
        knn_df["mean_query_ms"],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label=f"KNN (k={k})",
    )
    ax.plot(
        rnn_df["eps"],
        rnn_df["mean_query_ms"],
        marker="s",
        linewidth=2.2,
        color="#d62728",
        label=f"RNN (r={radius})",
    )
    ax.set_xlabel("Approximation Factor (ε)", fontsize=11)
    ax.set_ylabel("Mean Query Time (ms)", fontsize=11)
    ax.set_title(
        "Effect of KD-Tree Approximation Factor (ε) on Mean Query Time",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)

    fig.tight_layout()
    pdf_path = output_dir / "epsilon_query_time.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    saved_plots["query_time_pdf"] = pdf_path

    return saved_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark KNN and RNN speed and accuracy across KD-Tree approximation factors (eps)."
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
        "-o",
        "--output_dir",
        type=str,
        default="analysis_output/evaluation/epsilon",
        help="Directory to save CSV tables and plots.",
    )
    parser.add_argument(
        "--eps_values",
        type=float,
        nargs="+",
        default=DEFAULT_EPS_VALUES,
        help="List of epsilon values to benchmark.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=6,
        help="Number of neighbors for KNN (default: 6).",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=0.85,
        help="Search radius for RNN (default: 0.85).",
    )
    parser.add_argument(
        "-p",
        type=float,
        default=2.0,
        help="Minkowski distance p (default: 2.0).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of full query passes per epsilon (default: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(6, max(1, (os.cpu_count() or 4) // 2)),
        help="Number of parallel worker processes (default: up to 6 physical cores).",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    knn_df, rnn_df = run_parallel_benchmarks(
        elementary_csv=args.elementary_csv,
        histogram_csv=args.histogram_csv,
        eps_values=args.eps_values,
        k=args.k,
        radius=args.radius,
        p=args.p,
        n_repeats=args.repeats,
        workers=args.workers,
    )

    knn_csv = out_dir / "knn_epsilon_benchmark.csv"
    rnn_csv = out_dir / "rnn_epsilon_benchmark.csv"
    knn_df.to_csv(knn_csv, index=False)
    rnn_df.to_csv(rnn_csv, index=False)

    plots = plot_epsilon_results(
        knn_df=knn_df,
        rnn_df=rnn_df,
        output_dir=out_dir,
        k=args.k,
        radius=args.radius,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + "=" * 90)
    print(f"KNN (k={args.k}) EPSILON BENCHMARK")
    print("=" * 90)
    print(knn_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 90)
    print(f"RNN (r={args.radius}) EPSILON BENCHMARK")
    print("=" * 90)
    print(rnn_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 90)
    for name, path in plots.items():
        print(f"Saved plot ({name}): {path.resolve()}")


if __name__ == "__main__":
    main()
