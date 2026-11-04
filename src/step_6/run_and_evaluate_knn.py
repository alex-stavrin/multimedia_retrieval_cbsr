from pathlib import Path
from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine
from src.step_6.retrieval_evalulator import RetrievalEvaluator


def run_and_evaluate_knn(
    elementary_csv: str,
    histogram_csv: str,
    output_dir: str,
    k: int = 10,
    p: float = 2.0,
    eps: float = 0.0,
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

    return dict()
