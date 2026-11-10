import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ROCEvaluator:
    """
    Computes ROC curves (Sensitivity vs Specificity) and AUROC as defined
    in Multimedia Retrieval Module 7 (Slides 21-26).
    """

    def __init__(self, ground_truth: dict[str, str]):
        self.ground_truth = ground_truth
        self.class_counts = defaultdict(int)
        for cls in ground_truth.values():
            self.class_counts[cls] += 1
        self.total_db_size = max(1, len(ground_truth) - 1)

    def compute_query_roc(
        self, query_id: str, ranked_ids: list[str]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Computes (Sensitivity, Specificity, AUROC) for a single query's ranked results.
        """
        query_class = self.ground_truth.get(query_id)
        c = max(1, self.class_counts[query_class] - 1)
        negative_count = max(1, self.total_db_size - c)

        # Exclude self
        filtered_ranks = [rid for rid in ranked_ids if rid != query_id]
        total_ranked = len(filtered_ranks)

        if total_ranked == 0:
            return np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.5

        is_relevant = np.array(
            [self.ground_truth.get(rid) == query_class for rid in filtered_ranks],
            dtype=bool,
        )

        true_positives = np.cumsum(is_relevant)
        false_positives = np.arange(1, total_ranked + 1) - true_positives

        sensitivities = np.concatenate(([0.0], true_positives / c))
        specificities = np.concatenate(
            ([1.0], np.maximum(0, negative_count - false_positives) / negative_count)
        )

        if sensitivities[-1] < 1.0 or specificities[-1] > 0.0:
            sensitivities = np.append(sensitivities, 1.0)
            specificities = np.append(specificities, 0.0)

        trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auroc = float(trap_fn(specificities, sensitivities))

        return sensitivities, specificities, auroc

    def evaluate_method(
        self,
        results: dict[str, list[str]],
        method_name: str = "Retrieval",
        grid_points: int = 101,
    ) -> dict:
        """
        Evaluates query-level AUROC and interpolates a mean ROC curve across all queries
        by sweeping rank cutoffs k in [1, d].
        """
        common_sensitivity = np.linspace(0.0, 1.0, grid_points)
        interpolated_specificities = []
        query_records = []
        class_aurocs = defaultdict(list)

        for query_id, ranked_ids in results.items():
            query_class = self.ground_truth.get(query_id)
            sensitivity, specificity, auroc = self.compute_query_roc(
                query_id, ranked_ids
            )
            query_records.append(
                {"query_id": query_id, "class": query_class, "auroc": auroc}
            )
            class_aurocs[query_class].append(auroc)

            interpolated_specification = np.interp(
                common_sensitivity, sensitivity, specificity
            )
            interpolated_specificities.append(interpolated_specification)

        data_frame_queries = pd.DataFrame(query_records)
        mean_specificity = np.mean(interpolated_specificities, axis=0)

        class_mean_auroc = {
            cls: float(np.mean(vals)) for cls, vals in class_aurocs.items()
        }
        macro_auroc = float(np.mean(list(class_mean_auroc.values())))
        micro_auroc = float(data_frame_queries["auroc"].mean())

        return {
            "method": method_name,
            "macro_auroc": macro_auroc,
            "micro_auroc": micro_auroc,
            "per_class_auroc": class_mean_auroc,
            "queries": data_frame_queries,
            "curve": {
                "sensitivity": common_sensitivity.tolist(),
                "specificity": mean_specificity.tolist(),
            },
        }

    def evaluate_radius_sweep(
        self,
        distance_results: dict[str, dict[str, float]],
        method_name: str = "rNN",
        grid_points: int = 101,
    ) -> dict:
        """
        Evaluates rNN ROC curve by sweeping a global distance threshold r across
        the dataset (Slide 22: ||I - Q|| < threshold).
        """
        all_dists = []
        prepared_queries = []

        for query_id, matches in distance_results.items():
            query_class = self.ground_truth.get(query_id)
            c = max(1, self.class_counts[query_class] - 1)
            negative_count = max(1, self.total_db_size - c)

            # Ensure sorted by distance ascending, excluding self
            pairs = [
                (rid, float(dist))
                for rid, dist in matches.items()
                if rid != query_id
            ]
            pairs.sort(key=lambda item: item[1])

            rids = [p[0] for p in pairs]
            dists = np.array([p[1] for p in pairs], dtype=float)
            is_rel = np.array(
                [self.ground_truth.get(rid) == query_class for rid in rids],
                dtype=bool,
            )
            cum_tp = np.cumsum(is_rel)

            if len(dists) > 0:
                all_dists.append(dists)

            prepared_queries.append(
                (query_id, query_class, dists, cum_tp, c, negative_count)
            )

        if all_dists:
            flat_dists = np.concatenate(all_dists)
            radii = np.concatenate(
                ([0.0], np.quantile(flat_dists, np.linspace(0.0, 1.0, grid_points - 1)))
            )
        else:
            radii = np.linspace(0.0, 1.0, grid_points)

        mean_sens_curve = []
        mean_spec_curve = []
        class_sens_curves = defaultdict(lambda: [[] for _ in range(len(radii))])
        class_spec_curves = defaultdict(lambda: [[] for _ in range(len(radii))])

        query_sens_trajectories = defaultdict(list)
        query_spec_trajectories = defaultdict(list)

        for r_idx, r in enumerate(radii):
            sens_at_r = []
            spec_at_r = []
            for q_id, q_cls, dists, cum_tp, c, neg_count in prepared_queries:
                count_retrieved = int(np.searchsorted(dists, r, side="right"))
                tp = int(cum_tp[count_retrieved - 1]) if count_retrieved > 0 else 0
                fp = count_retrieved - tp

                s = min(1.0, tp / c)
                sp = max(0.0, (neg_count - fp) / neg_count)

                sens_at_r.append(s)
                spec_at_r.append(sp)
                class_sens_curves[q_cls][r_idx].append(s)
                class_spec_curves[q_cls][r_idx].append(sp)
                query_sens_trajectories[q_id].append(s)
                query_spec_trajectories[q_id].append(sp)

            mean_sens_curve.append(float(np.mean(sens_at_r)))
            mean_spec_curve.append(float(np.mean(spec_at_r)))

        # Ensure endpoints (0, 1) and (1, 0)
        sens_arr = np.concatenate(([0.0], np.array(mean_sens_curve), [1.0]))
        spec_arr = np.concatenate(([1.0], np.array(mean_spec_curve), [0.0]))

        # Sort by sensitivity ascending for integration and interpolation
        order = np.argsort(sens_arr, kind="mergesort")
        sens_arr = sens_arr[order]
        spec_arr = spec_arr[order]

        common_sensitivity = np.linspace(0.0, 1.0, grid_points)
        interp_spec = np.interp(common_sensitivity, sens_arr, spec_arr)

        trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        micro_auroc = float(trap_fn(spec_arr, sens_arr))

        # Per-class AUROC under radius sweep
        class_mean_auroc = {}
        for cls in class_sens_curves:
            c_sens = np.concatenate(
                ([0.0], [float(np.mean(vals)) for vals in class_sens_curves[cls]], [1.0])
            )
            c_spec = np.concatenate(
                ([1.0], [float(np.mean(vals)) for vals in class_spec_curves[cls]], [0.0])
            )
            c_ord = np.argsort(c_sens, kind="mergesort")
            class_mean_auroc[cls] = float(trap_fn(c_spec[c_ord], c_sens[c_ord]))

        macro_auroc = float(np.mean(list(class_mean_auroc.values())))

        # Per-query AUROC across global radius sweep
        query_records = []
        for q_id, q_cls, _, _, _, _ in prepared_queries:
            q_s = np.concatenate(([0.0], query_sens_trajectories[q_id], [1.0]))
            q_sp = np.concatenate(([1.0], query_spec_trajectories[q_id], [0.0]))
            q_ord = np.argsort(q_s, kind="mergesort")
            q_auc = float(trap_fn(q_sp[q_ord], q_s[q_ord]))
            query_records.append(
                {"query_id": q_id, "class": q_cls, "auroc": q_auc}
            )

        data_frame_queries = pd.DataFrame(query_records)

        return {
            "method": method_name,
            "macro_auroc": macro_auroc,
            "micro_auroc": micro_auroc,
            "per_class_auroc": class_mean_auroc,
            "queries": data_frame_queries,
            "curve": {
                "sensitivity": common_sensitivity.tolist(),
                "specificity": interp_spec.tolist(),
            },
        }

    @staticmethod
    def plot_roc_curves(
        evaluated_methods: list[dict],
        output_path: str | Path,
        title: str = "ROC Curves (Sensitivity vs Specificity)",
    ) -> None:
        """
        Plots one or more ROC curves on the same figure (Slides 22-26).
        """
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

            # Plot the operating point (e.g. chosen k or radius r) if provided
            op = eval_res.get("operating_point")
            if op is not None:
                op_sens, op_spec, op_label = op
                plt.scatter(
                    [op_sens],
                    [op_spec],
                    color=color,
                    s=65,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.8,
                    label=f"  • {op_label} (Sens={op_sens:.2f}, Spec={op_spec:.2f})",
                )

        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("Sensitivity (Recall / TPR)", fontsize=12)
        plt.ylabel("Specificity (TNR)", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="lower left", fontsize=9, framealpha=0.9)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
        plt.close()
        print(f"Saved ROC plot to: {out_file.resolve()}")

    @staticmethod
    def plot_metrics_comparison(
        all_stats: list[dict],
        output_path: str | Path,
        title: str = "Comparison of Retrieval Methods Across Metrics",
    ) -> None:
        """
        Plots a grouped bar chart comparing all methods across key retrieval metrics.
        """
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

        plt.xticks(x, [label for _, label in available_metrics], rotation=20, ha="right", fontsize=10)
        plt.ylim([0.0, 1.12])
        plt.ylabel("Score [0, 1]", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="upper right", fontsize=9, framealpha=0.9)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
        plt.close()
        print(f"Saved metrics comparison plot to: {out_file.resolve()}")

    @staticmethod
    def plot_time_comparison(
        all_stats: list[dict],
        output_path: str | Path,
        title: str = "Execution & Retrieval Time Comparison Across Methods",
    ) -> None:
        """
        Plots side-by-side bar charts comparing total execution time, retrieval query time,
        and mean per-query time across retrieval methods.
        """
        methods = [
            stat.get("method", f"Method {idx + 1}")
            for idx, stat in enumerate(all_stats)
        ]
        init_times = [float(stat.get("init_time_s", 0.0)) for stat in all_stats]
        retrieval_times = [
            float(stat.get("retrieval_time_s", 0.0)) for stat in all_stats
        ]
        total_times = [float(stat.get("total_time_s", 0.0)) for stat in all_stats]
        mean_query_ms = [
            float(stat.get("mean_query_time_ms", 0.0)) for stat in all_stats
        ]

        x = np.arange(len(methods))
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
        bar_colors = [colors[i % len(colors)] for i in range(len(methods))]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

        # Panel 1: Total Time to Finish vs Retrieval Time (seconds)
        width = 0.36
        bars_ret = axes[0].bar(
            x - width / 2,
            retrieval_times,
            width=width,
            label="Retrieval Queries Time (s)",
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        bars_tot = axes[0].bar(
            x + width / 2,
            total_times,
            width=width,
            label="Total Time to Finish (s)",
            color="#ff7f0e",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        for bar in list(bars_ret) + list(bars_tot):
            h = bar.get_height()
            axes[0].annotate(
                f"{h:.2f}s",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        axes[0].set_ylabel("Time (seconds)", fontsize=11)
        axes[0].set_title("Total Time to Finish & Retrieval Time (s)", fontsize=12, fontweight="bold")
        axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)
        if max(total_times + retrieval_times, default=0.0) > 0:
            axes[0].set_ylim([0, max(total_times + retrieval_times) * 1.18])

        # Panel 2: Mean Query Time per Shape (milliseconds)
        bars_q = axes[1].bar(
            x,
            mean_query_ms,
            width=0.5,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        for bar, init_t in zip(bars_q, init_times):
            h = bar.get_height()
            axes[1].annotate(
                f"{h:.3f} ms\n(init: {init_t:.2f}s)",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        axes[1].set_ylabel("Mean Query Time (ms / query)", fontsize=11)
        axes[1].set_title("Mean Retrieval Time per Query (ms)", fontsize=12, fontweight="bold")
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)
        if max(mean_query_ms, default=0.0) > 0:
            axes[1].set_ylim([0, max(mean_query_ms) * 1.22])

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
        plt.close()
        print(f"Saved time comparison plot to: {out_file.resolve()}")

    @staticmethod
    def save_roc_results(
        eval_result: dict, output_dir: str | Path, prefix: str
    ) -> None:
        """Helper to save ROC metrics and curves to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if "queries" in eval_result and isinstance(eval_result["queries"], pd.DataFrame):
            eval_result["queries"].to_csv(
                out_path / f"{prefix}_auroc_queries.csv", index=False
            )

        summary = {
            "method": eval_result["method"],
            "macro_auroc": eval_result["macro_auroc"],
            "micro_auroc": eval_result["micro_auroc"],
            "per_class_auroc": eval_result["per_class_auroc"],
            "curve": eval_result["curve"],
        }
        if "operating_point" in eval_result and eval_result["operating_point"] is not None:
            summary["operating_point"] = list(eval_result["operating_point"])

        with open(out_path / f"{prefix}_roc.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def load_roc_results(roc_json_path: str | Path) -> dict:
        """Loads a previously saved ROC JSON summary."""
        with open(roc_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "operating_point" in data and data["operating_point"] is not None:
            data["operating_point"] = tuple(data["operating_point"])
        return data
