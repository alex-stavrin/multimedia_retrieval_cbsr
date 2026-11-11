import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.step_6.evaluation_plots import (
    plot_metrics_comparison,
    plot_roc_curves,
    plot_time_comparison,
)


class ROCEvaluator:
    def __init__(self, ground_truth: dict[str, str]):
        self.ground_truth = ground_truth
        self.class_counts = defaultdict(int)
        for cls in ground_truth.values():
            self.class_counts[cls] += 1
        self.total_db_size = max(1, len(ground_truth) - 1)

    def compute_query_roc(
        self, query_id: str, ranked_ids: list[str]
    ) -> tuple[np.ndarray, np.ndarray, float]:
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
        auroc = float(trap_fn(specificities, sensitivities))  # type: ignore

        return sensitivities, specificities, auroc

    def evaluate_method(
        self,
        results: dict[str, list[str]],
        method_name: str = "Retrieval",
        grid_points: int = 101,
    ) -> dict:
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
        all_dists = []
        prepared_queries = []

        for query_id, matches in distance_results.items():
            query_class = self.ground_truth.get(query_id)
            c = max(1, self.class_counts[query_class] - 1)
            negative_count = max(1, self.total_db_size - c)

            # Ensure sorted by distance ascending, excluding self
            pairs = [
                (rid, float(dist)) for rid, dist in matches.items() if rid != query_id
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
        micro_auroc = float(trap_fn(spec_arr, sens_arr))  # type: ignore

        # Per-class AUROC under radius sweep
        class_mean_auroc = {}
        for cls in class_sens_curves:
            c_sens = np.concatenate(
                (
                    [0.0],
                    [float(np.mean(vals)) for vals in class_sens_curves[cls]],
                    [1.0],
                )
            )
            c_spec = np.concatenate(
                (
                    [1.0],
                    [float(np.mean(vals)) for vals in class_spec_curves[cls]],
                    [0.0],
                )
            )
            c_ord = np.argsort(c_sens, kind="mergesort")
            class_mean_auroc[cls] = float(trap_fn(c_spec[c_ord], c_sens[c_ord]))  # type: ignore

        macro_auroc = float(np.mean(list(class_mean_auroc.values())))

        # Per-query AUROC across global radius sweep
        query_records = []
        for q_id, q_cls, _, _, _, _ in prepared_queries:
            q_s = np.concatenate(([0.0], query_sens_trajectories[q_id], [1.0]))
            q_sp = np.concatenate(([1.0], query_spec_trajectories[q_id], [0.0]))
            q_ord = np.argsort(q_s, kind="mergesort")
            q_auc = float(trap_fn(q_sp[q_ord], q_s[q_ord]))  # type: ignore
            query_records.append({"query_id": q_id, "class": q_cls, "auroc": q_auc})

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

    # Plotting helper aliases (delegated to evaluation_plots)
    plot_roc_curves = staticmethod(plot_roc_curves)
    plot_metrics_comparison = staticmethod(plot_metrics_comparison)
    plot_time_comparison = staticmethod(plot_time_comparison)

    @staticmethod
    def save_roc_results(
        eval_result: dict, output_dir: str | Path, prefix: str
    ) -> None:
        """Helper to save ROC metrics and curves to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if "queries" in eval_result and isinstance(
            eval_result["queries"], pd.DataFrame
        ):
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
        if (
            "operating_point" in eval_result
            and eval_result["operating_point"] is not None
        ):
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
