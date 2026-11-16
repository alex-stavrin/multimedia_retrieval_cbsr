import argparse
from pathlib import Path

from src.step_4.distance_calculations.calculate_total_distance import (
    DISTANCE_CONFIG_EMD,
    ShapeDistanceCalculator,
)
from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine
from src.step_5.ANN.table_renderer import find_mesh_path, render_meshes_grid

DEFAULT_QUERIES = [
    "Chess/D01049.obj",
    "Door/D01005.obj",
    "HumanHead/D00013.obj",
    "HumanHead/m356.obj",
]

QUERY_OUTPUT_NAMES = {
    "Chess/D01049.obj": "result_comparison_chess_D01049.pdf",
    "Door/D01005.obj": "result_comparison_door_D01005.pdf",
    "HumanHead/D00013.obj": "result_comparison_human_D00013.pdf",
    "HumanHead/m356.obj": "result_comparison_human_m356.pdf",
}


def _output_filename_for_query(query_str: str, resolved_id: str | None) -> str:
    if resolved_id is not None:
        norm_id = resolved_id.replace("\\", "/")
        if norm_id in QUERY_OUTPUT_NAMES:
            return QUERY_OUTPUT_NAMES[norm_id]
        cls_name, stem = norm_id.split("/")[0].lower(), Path(norm_id).stem
        if cls_name == "humanhead":
            cls_name = "human"
        return f"result_comparison_{cls_name}_{stem}.pdf"
    stem = Path(query_str).stem
    return f"result_comparison_{stem}.pdf"


def generate_comparison_tables(
    queries: list[str],
    elementary_csv: str | Path,
    histogram_csv: str | Path,
    weights_csv: str | Path,
    model_dir: str | Path,
    output_dir: str | Path = ".",
    k: int = 6,
    radius: float = 500.0,
    eps: float = 0.0,
    p: float = 2.0,
) -> list[Path]:
    engine = SpatialSearchEngine(
        elementary_csv=elementary_csv,
        histogram_csv=histogram_csv,
    )
    calculator = ShapeDistanceCalculator(
        csv_databases=[elementary_csv, histogram_csv, weights_csv],
        distance_config=DISTANCE_CONFIG_EMD,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    for query_str in queries:
        resolved_id: str | None = None
        try:
            resolved_id = engine.shape_ids[engine.resolve_query_index(query_str)]
            query_file: str | Path = find_mesh_path(model_dir, resolved_id)
        except ValueError:
            query_file = Path(query_str)

        norm_resolved = (
            resolved_id.replace("\\", "/") if resolved_id is not None else None
        )

        # 1. Step 4 Weighted Custom Distance (EMD + Euclidean)
        weighted_all = calculator.calculate_total_distance(
            query=resolved_id if resolved_id is not None else query_str,
            run_pipeline=(resolved_id is None),
        )
        weighted_matches = [
            (find_mesh_path(model_dir, shape_id), float(dist))
            for shape_id, dist in weighted_all.items()
            if norm_resolved is None or shape_id.replace("\\", "/") != norm_resolved
        ][:k]

        # 2. Step 5 kNN
        knn_results = engine.query_knn(
            query=query_str,
            k=k,
            eps=eps,
            p=p,
            exclude_self=True,
        )
        if resolved_id is None and engine.last_preprocessed_mesh is not None:
            query_file = engine.last_preprocessed_mesh

        knn_matches = [
            (find_mesh_path(model_dir, shape_id), float(dist))
            for shape_id, dist in knn_results.items()
        ]

        # 3. Step 5 rNN (ensure radius is large enough so rNN returns at least k matches)
        min_required_radius = (
            max(knn_results.values()) + 1.0 if knn_results else radius
        )
        effective_radius = max(radius, min_required_radius)
        rnn_results = engine.query_range(
            query=query_str,
            radius=effective_radius,
            eps=eps,
            p=p,
            exclude_self=True,
        )
        rnn_matches = [
            (find_mesh_path(model_dir, shape_id), float(dist))
            for shape_id, dist in list(rnn_results.items())[:k]
        ]

        rows_data = [
            {
                "query_file": query_file,
                "matches": weighted_matches,
                "row_label": "Custom Distance",
            },
            {
                "query_file": query_file,
                "matches": knn_matches,
                "row_label": "kNN",
            },
            {
                "query_file": query_file,
                "matches": rnn_matches,
                "row_label": "rNN",
            },
        ]

        out_file = out_dir / _output_filename_for_query(query_str, resolved_id)
        render_meshes_grid(rows_data=rows_data, output_path=str(out_file))
        saved_files.append(out_file)

    return saved_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PDF comparison tables with 3 rows (Step 4 Custom Distance, "
            "Step 5 kNN, Step 5 rNN) for each query shape."
        )
    )
    parser.add_argument(
        "-q",
        "--queries",
        nargs="+",
        default=DEFAULT_QUERIES,
        help="Query shapes to generate comparison tables for.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=6,
        help="Number of neighbors to display per method (default: 6).",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=500.0,
        help="Search radius for rNN (default: 500.0, big enough to return k matches).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="KD-Tree approximation factor eps (default: 0.0).",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=2.0,
        help="Minkowski distance order p (default: 2.0).",
    )
    parser.add_argument(
        "--elementary_csv",
        type=str,
        default="3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv",
        help="Path to normalized elementary descriptors CSV.",
    )
    parser.add_argument(
        "--histogram_csv",
        type=str,
        default="3d_models_final/analysis/4_descriptors/histogram_descriptors.csv",
        help="Path to histogram descriptors CSV.",
    )
    parser.add_argument(
        "--weights_csv",
        type=str,
        default="analysis_output/distances/weights_emd_hist_sampling_descriptors.csv",
        help="Path to descriptor weights CSV for Step 4 custom distance.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="3d_models_final/3d_models_steps/3_models_fully_normalized",
        help="Path to 3D models directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output PDF tables (default: current directory).",
    )

    args = parser.parse_args()
    generate_comparison_tables(
        queries=args.queries,
        elementary_csv=args.elementary_csv,
        histogram_csv=args.histogram_csv,
        weights_csv=args.weights_csv,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        k=args.k,
        radius=args.radius,
        eps=args.eps,
        p=args.p,
    )


if __name__ == "__main__":
    main()
