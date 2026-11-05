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
        self, results: dict[str, list[str]], method_name: str = "Retrieval"
    ) -> dict:
        # this will collect the stats and data for individual query
        query_records = []
        # this is our confusion matrix
        # If we queried for a car and got a car confusion_matrix["Car"]["Car"] += 1
        # If we queried for a car and got a sedan confusion_matrix["Car"]["Sedan"] += 1
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for query_id, retrieval_ids in results.items():
            query_class = self.ground_truth.get(query_id)

            # -1 to exclude self
            same_class_as_query_count = max(1, self.class_counts[query_class] - 1)
            number_of_retrieved = len(retrieval_ids)

            # true positives are retrieved shapes that belong to the same class as the query
            true_positive = 0
            is_first_correct = False

            for rank, retrieved_id in enumerate(retrieved_id):  # type: ignore
                retrieved_class = self.ground_truth.get(retrieved_id)
                confusion_matrix[query_class][retrieved_class] += 1

                if query_class == retrieved_class:
                    true_positive += 1
                    if rank == 0:
                        is_first_correct = True

            false_positives = number_of_retrieved - true_positive
            false_negative = same_class_as_query_count - true_positive

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
            f_1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            query_records.append(
                {
                    "query_id": query_id,
                    "class": query_class,
                    "retrieval_count": number_of_retrieved,
                    "true_positives": true_positive,
                    "false_positives": false_positives,
                    "precision": precision,
                    "recall": recall,
                    "f1": f_1,
                    "is_first_correct": 1.0 if is_first_correct else 0.0,
                }
            )

        data_frame_queries = pd.DataFrame(query_records)

        per_class = (
            data_frame_queries.groupby("class")
            .agg(
                number_of_queries=("query_id", "count"),
                mean_retrieved=("retrieved_count", "mean"),
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                f1=("f1", "mean"),
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
            "macro_f1": float(per_class["f1"].mean()),
            "micro_precision": float(data_frame_queries["precision"].mean()),
            "micro_recall": float(data_frame_queries["recall"].mean()),
            "micro_f1": float(data_frame_queries["f1"].mean()),
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
