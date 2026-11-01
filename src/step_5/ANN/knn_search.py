import argparse
from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine
from src.step_5.ANN.table_renderer import find_mesh_path, render_meshes_grid


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-q",
        "--queries",
        nargs="+",
        required=True,
    )
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument(
        "--p", type=float, default=2.0
    )  # Default is 2.0 (Euclidean distance)
    parser.add_argument("--elementary_csv", type=str, required=True)
    parser.add_argument("--histogram_csv", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="3d_models")
    parser.add_argument("-o", "--output", type=str, default="knn_result_table.png")

    arguments = parser.parse_args()
    engine = SpatialSearchEngine(
        elementary_csv=arguments.elementary_csv, histogram_csv=arguments.histogram_csv
    )

    rows_data = []
    for query_str in arguments.queries:
        resolved_id = engine.shape_ids[engine.resolve_query_index(query_str)]
        query_file = find_mesh_path(arguments.model_dir, resolved_id)

        results = engine.query_knn(
            query=query_str,
            k=arguments.k,
            eps=arguments.eps,
            p=arguments.p,
            exclude_self=True,
        )

        matches = [
            (find_mesh_path(arguments.model_dir, shape_id), dist)
            for shape_id, dist in results.items()
        ]

        rows_data.append({"query_file": query_file, "matches": matches})

    render_meshes_grid(rows_data=rows_data, output_path=arguments.output)


if __name__ == "__main__":
    main()
