import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine
from src.step_6.retrieval_evalulator import RetrievalEvaluator

DEFAULT_LEAF_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))

# Worker-local state initialized once per process to avoid pickling overhead
_WORKER_ENGINE: SpatialSearchEngine | None = None
_WORKER_TREES: dict[int, KDTree] = {}


def _init_worker(
    elementary_csv: str,
    histogram_csv: str,
    leaf_sizes: list[int],
    k: int,
    eps: float,
    p: float,
    exclude_self: bool,
) -> None:
    global _WORKER_ENGINE, _WORKER_TREES
    _WORKER_ENGINE = SpatialSearchEngine(
        elementary_csv=elementary_csv,
        histogram_csv=histogram_csv,
    )
    _WORKER_TREES = {
        ls: KDTree(_WORKER_ENGINE.feature_matrix, leafsize=ls) for ls in leaf_sizes
    }
    # Warmup worker process
    for sid in _WORKER_ENGINE.shape_ids[:50]:
        _WORKER_ENGINE.query_knn(sid, k=k, eps=eps, p=p, exclude_self=exclude_self)


def _run_single_pass(
    task: tuple[int, int, int, float, float, bool, bool],
) -> tuple[int, int, float, float, dict[str, list[str]] | None]:
    repeat_idx, ls, k, eps, p, exclude_self, collect_results = task
    assert _WORKER_ENGINE is not None

    # Measure KDTree construction time
    gc.collect()
    gc.disable()
    t_build_0 = time.perf_counter()
    tree = KDTree(_WORKER_ENGINE.feature_matrix, leafsize=ls)
    build_elapsed_s = time.perf_counter() - t_build_0
    gc.enable()

    _WORKER_ENGINE.leaf_size = ls
    _WORKER_ENGINE.kd_tree = tree

    # Measure full KNN query pass across all shapes
    shape_ids = _WORKER_ENGINE.shape_ids
    gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    current_results: dict[str, list[str]] = {}
    for sid in shape_ids:
        matches = _WORKER_ENGINE.query_knn(
            sid, k=k, eps=eps, p=p, exclude_self=exclude_self
        )
        current_results[sid] = list(matches.keys())
    elapsed_s = time.perf_counter() - t0
    gc.enable()

    return (
        repeat_idx,
        ls,
        build_elapsed_s,
        elapsed_s,
        current_results if collect_results else None,
    )


def benchmark_knn_leaf_size(
    engine: SpatialSearchEngine,
    evaluator: RetrievalEvaluator,
    leaf_sizes: list[int],
    k: int = 6,
    eps: float = 0.0,
    p: float = 2.0,
    n_repeats: int = 5,
    exclude_self: bool = True,
    max_workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    shape_ids = engine.shape_ids

    # Reference baseline (default leaf_size = 16 or first leaf_size)
    ref_ls = 16 if 16 in leaf_sizes else leaf_sizes[0]
    engine.leaf_size = ref_ls
    engine.kd_tree = KDTree(engine.feature_matrix, leafsize=ref_ls)
    baseline_results = {
        sid: list(
            engine.query_knn(sid, k=k, eps=eps, p=p, exclude_self=exclude_self).keys()
        )
        for sid in shape_ids
    }

    results_by_ls: dict[int, dict[str, list[str]]] = {}
    pass_times: dict[int, list[float]] = {ls: [] for ls in leaf_sizes}
    build_times: dict[int, list[float]] = {ls: [] for ls in leaf_sizes}

    tasks = [
        (r, ls, k, eps, p, exclude_self, r == 0)
        for r in range(n_repeats)
        for ls in leaf_sizes
    ]

    effective_workers = max(1, min(max_workers, len(tasks)))
    print(
        f"Running {len(tasks)} benchmark passes ({len(leaf_sizes)} leaf sizes x {n_repeats} repeats) "
        f"across {effective_workers} parallel workers..."
    )

    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_init_worker,
        initargs=(
            str(engine.elementary_csv),
            str(engine.histogram_csv),
            leaf_sizes,
            k,
            eps,
            p,
            exclude_self,
        ),
    ) as executor:
        futures = [executor.submit(_run_single_pass, t) for t in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Benchmarking leaf sizes",
            unit="pass",
        ):
            repeat_idx, ls, build_elapsed_s, elapsed_s, res = future.result()
            build_times[ls].append(build_elapsed_s)
            pass_times[ls].append(elapsed_s)
            if res is not None:
                results_by_ls[ls] = res

    rows = []
    for ls in tqdm(leaf_sizes, desc="Evaluating retrieval metrics", unit="leaf_size"):
        elapsed_s = float(np.median(pass_times[ls]))
        mean_query_ms = (elapsed_s / len(shape_ids)) * 1000.0
        std_query_ms = (float(np.std(pass_times[ls])) / len(shape_ids)) * 1000.0
        tree_build_ms = float(np.median(build_times[ls])) * 1000.0
        results = results_by_ls[ls]

        eval_res = evaluator.evaluate(
            results=results, method_name=f"KNN (k={k}, leaf_size={ls})"
        )
        stats = eval_res["total_stats"]

        overlaps = [
            (
                len(set(baseline_results[sid]) & set(results[sid]))
                / len(baseline_results[sid])
                if baseline_results[sid]
                else 1.0
            )
            for sid in shape_ids
        ]

        rows.append(
            {
                "leaf_size": int(ls),
                "tree_build_ms": tree_build_ms,
                "total_time_ms": elapsed_s * 1000.0,
                "mean_query_ms": mean_query_ms,
                "std_query_ms": std_query_ms,
                "exact_overlap": float(np.mean(overlaps)),
                "macro_precision": float(stats["macro_precision"]),
                "macro_recall": float(stats["macro_recall"]),
                "macro_accuracy": float(stats["macro_accuracy"]),
                "macro_f1": float(stats["macro_f1"]),
                "macro_map": float(stats["macro_map"]),
                "micro_precision": float(stats["micro_precision"]),
            }
        )

    df = pd.DataFrame(rows)
    base_row = df.loc[df["leaf_size"] == ref_ls, "mean_query_ms"]
    base_time = (
        float(base_row.iloc[0])
        if not base_row.empty
        else float(df["mean_query_ms"].iloc[0])
    )
    df["speedup_vs_default"] = base_time / df["mean_query_ms"]
    return df


def plot_leaf_size_results(
    knn_df: pd.DataFrame,
    output_dir: Path,
    k: int,
    eps: float,
    elbow_ls: int = 16,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: dict[str, Path] = {}

    if elbow_ls not in knn_df["leaf_size"].values:
        elbow_ls = int(
            knn_df.loc[(knn_df["leaf_size"] - elbow_ls).abs().idxmin(), "leaf_size"]
        )
    elbow_time = float(
        knn_df.loc[knn_df["leaf_size"] == elbow_ls, "mean_query_ms"].iloc[0]
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
    ax.plot(
        knn_df["leaf_size"],
        knn_df["mean_query_ms"],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label=f"Mean Retrieval Time (k={k})",
    )
    ax.scatter(
        [elbow_ls],
        [elbow_time],
        color="#2ca02c",
        s=90,
        zorder=5,
        label=f"Selected Value (leaf_size={elbow_ls}: {elbow_time:.3f} ms)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Leaf Size (log₂ scale)", fontsize=11)
    ax.set_ylabel("Mean Retrieval Time (ms)", fontsize=11)
    ax.set_title(
        f"KD-Tree Leaf Size Analysis for KNN Search (k={k}, ε={eps})",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9.5, loc="upper right")
    fig.tight_layout()

    for stem in ("knn_leaf_size_retrieval_time", "knn_leaf_size_detailed_analysis"):
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        saved_plots[f"{stem}_png"] = png_path
        saved_plots[f"{stem}_pdf"] = pdf_path

    plt.close(fig)
    return saved_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark KNN retrieval speed across different KD-Tree leaf sizes."
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
        default="analysis_output/evaluation/leaf_size",
        help="Directory to save CSV benchmark results and plots.",
    )
    parser.add_argument(
        "--leaf_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_LEAF_SIZES,
        help="List of KD-Tree leaf sizes to benchmark.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=6,
        help="Number of neighbors for KNN (default: 6).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Approximation factor for KD-Tree (default: 0.0).",
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
        default=5,
        help="Number of full query passes per leaf size (default: 5).",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel worker processes (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--elbow_leaf_size",
        type=int,
        default=16,
        help="Leaf size after diminishing returns to highlight on the plot (default: 16).",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip benchmarking and regenerate plots from existing CSV.",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "knn_leaf_size_benchmark.csv"

    if args.plot_only and csv_path.exists():
        knn_df = pd.read_csv(csv_path)
    else:
        engine = SpatialSearchEngine(
            elementary_csv=args.elementary_csv,
            histogram_csv=args.histogram_csv,
        )
        ground_truth = {
            sid: sid.replace("\\", "/").split("/")[0] for sid in engine.shape_ids
        }
        evaluator = RetrievalEvaluator(ground_truth=ground_truth)

        knn_df = benchmark_knn_leaf_size(
            engine=engine,
            evaluator=evaluator,
            leaf_sizes=args.leaf_sizes,
            k=args.k,
            eps=args.eps,
            p=args.p,
            n_repeats=args.repeats,
            max_workers=args.workers,
        )
        knn_df.to_csv(csv_path, index=False)

    plots = plot_leaf_size_results(
        knn_df=knn_df,
        output_dir=out_dir,
        k=args.k,
        eps=args.eps,
        elbow_ls=args.elbow_leaf_size,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n" + "=" * 95)
    print(f"KNN (k={args.k}, eps={args.eps}) LEAF SIZE BENCHMARK")
    print("=" * 95)
    print(knn_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 95)
    print(f"Saved benchmark CSV: {csv_path.resolve()}")
    for name, path in plots.items():
        print(f"Saved plot ({name}): {path.resolve()}")


if __name__ == "__main__":
    main()
