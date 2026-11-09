import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


class RetrievalEvaluator:
    # ground truch is dictionary in style : "shape" -> "class"
    def __init__(self, ground_truth: dict[str, str]):
        self.ground_truth = ground_truth
        # creates a dictionary in the style : "class" -> "shapes_in_class_counter" with initialized 0 counter
        self.class_counts = defaultdict(int)
        for current_class in ground_truth.values():
            self.class_counts[current_class] += 1

    def evaluate(
        self,
        results: dict[str, list[str]],
        method_name: str = "Retrieval",
        full_rankings: dict[str, list[str]] | None = None,
    ) -> dict:
        # this will collect the stats and data for individual query
        query_records = []
        # this is our confusion matrix
        # If we queried for a car and got a car confusion_matrix["Car"]["Car"] += 1
        # If we queried for a car and got a sedan confusion_matrix["Car"]["Sedan"] += 1
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        # exclude self with -1
        total_database_size = max(1, len(self.ground_truth) - 1)

        # do every query
        for query_id, retrieval_ids in results.items():
            query_class = self.ground_truth.get(query_id)

            # -1 to exclude self
            same_class_as_query_count = max(1, self.class_counts[query_class] - 1)
            negative_class_count = max(
                1, total_database_size - same_class_as_query_count
            )
            number_of_retrieved = len(retrieval_ids)

            # true positives are retrieved shapes that belong to the same class as the query
            true_positive = 0
            is_first_correct = False
            precision_i = []
            consecutive_correct = 0
            hit_error = False

            # loop through the results
            for rank, retrieved_id in enumerate(retrieval_ids):
                retrieved_class = self.ground_truth.get(retrieved_id)
                confusion_matrix[query_class][retrieved_class] += 1

                if query_class == retrieved_class:
                    true_positive += 1
                    if rank == 0:
                        is_first_correct = True
                    if not hit_error:
                        consecutive_correct += 1
                else:
                    hit_error = True

                precision_i.append(true_positive / (rank + 1))

            false_positives = number_of_retrieved - true_positive
            false_negative = max(0, same_class_as_query_count - true_positive)
            true_negative = max(0, negative_class_count - false_positives)

            precision = (
                (true_positive / number_of_retrieved)
                if number_of_retrieved > 0
                else 0.0
            )
            recall = (
                (true_positive / same_class_as_query_count)
                if same_class_as_query_count > 0
                else 0.0
            )
            specificity = (
                (true_negative / negative_class_count)
                if negative_class_count > 0
                else 0.0
            )
            accuracy = (
                (true_positive + true_negative) / total_database_size
                if total_database_size > 0
                else 0.0
            )
            f_1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            mean_precision = float(np.mean(precision_i)) if precision_i else 0.0
            last_rank = min(1.0, consecutive_correct / same_class_as_query_count)

            c = same_class_as_query_count
            tier_source = (
                full_rankings.get(query_id, retrieval_ids)
                if full_rankings is not None
                else retrieval_ids
            )
            tp_first_tier = sum(
                1
                for rid in tier_source[:c]
                if self.ground_truth.get(rid) == query_class
            )
            first_tier = tp_first_tier / c
            tp_second_tier = sum(
                1
                for rid in tier_source[: 2 * c]
                if self.ground_truth.get(rid) == query_class
            )
            second_tier = tp_second_tier / c

            query_records.append(
                {
                    "query_id": query_id,
                    "class": query_class,
                    "retrieval_count": number_of_retrieved,
                    "true_positives": true_positive,
                    "false_positives": false_positives,
                    "true_negatives": true_negative,
                    "false_negatives": false_negative,
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "accuracy": accuracy,
                    "f1": f_1,
                    "average_precision": mean_precision,
                    "last_rank": last_rank,
                    "first_tier": first_tier,
                    "second_tier": second_tier,
                    "is_first_correct": 1.0 if is_first_correct else 0.0,
                }
            )

        data_frame_queries = pd.DataFrame(query_records)

        per_class = (
            data_frame_queries.groupby("class")
            .agg(
                number_of_queries=("query_id", "count"),
                mean_retrieved=("retrieval_count", "mean"),
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                specificity=("specificity", "mean"),
                accuracy=("accuracy", "mean"),
                last_rank=("last_rank", "mean"),
                first_tier=("first_tier", "mean"),
                second_tier=("second_tier", "mean"),
                f1=("f1", "mean"),
                ap=("average_precision", "mean"),
                is_first_correct=("is_first_correct", "mean"),
            )
            .reset_index()
            .sort_values("f1", ascending=False)
        )

        total_stats = {
            "method": method_name,
            "total_queries": len(data_frame_queries),
            "macro_precision": float(per_class["precision"].mean()),
            "macro_recall": float(per_class["recall"].mean()),
            "macro_specificity": float(per_class["specificity"].mean()),
            "macro_accuracy": float(per_class["accuracy"].mean()),
            "macro_f1": float(per_class["f1"].mean()),
            "macro_map": float(per_class["ap"].mean()),
            "macro_last_rank": float(per_class["last_rank"].mean()),
            "macro_first_tier": float(per_class["first_tier"].mean()),
            "macro_second_tier": float(per_class["second_tier"].mean()),
            "micro_precision": float(data_frame_queries["precision"].mean()),
            "micro_recall": float(data_frame_queries["recall"].mean()),
            "micro_specificity": float(data_frame_queries["specificity"].mean()),
            "micro_accuracy": float(data_frame_queries["accuracy"].mean()),
            "micro_f1": float(data_frame_queries["f1"].mean()),
            "micro_map": float(data_frame_queries["average_precision"].mean()),
            "micro_last_rank": float(data_frame_queries["last_rank"].mean()),
            "micro_first_tier": float(data_frame_queries["first_tier"].mean()),
            "micro_second_tier": float(data_frame_queries["second_tier"].mean()),
            "is_first_correct_accuracy": float(
                data_frame_queries["is_first_correct"].mean()
            ),
        }

        confusion_data_frame = pd.DataFrame(confusion_matrix).fillna(0).T

        return {
            "per_class": per_class,
            "total_stats": total_stats,
            "queries": data_frame_queries,
            "confusion_matrix": confusion_data_frame,
        }
