import argparse
from pathlib import Path
from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine
from src.step_5.ANN.table_renderer import find_mesh_path, render_meshes_grid


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("-q", "--queries", nargs="+", required=True)
    parser.add_argument("-r", "--radius", type=float, required=True)
    parser.add_argument("--max_matches", type=int, default=6)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument(
        "--p", type=float, default=2.0
    )  # default is p=2 (euclidean distance)
    parser.add_argument("--elementary_csv", type=str, required=True)
    parser.add_argument("--histogram_csv", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="rnn_table.png")

    arguments = parser.parse_args()

    spatial_search_engine = SpatialSearchEngine(
        elementary_csv=arguments.elementary_csv, histogram_csv=arguments.histogram_csv
    )

    rows_data = []
    for query_string in arguments.queries:
        results = spatial_search_engine.query_range(
            query=query_string,
            radius=arguments.radius,
            eps=arguments.eps,
            p=arguments.p,
            exclude_self=True,
        )

        try:
            resolved_id = spatial_search_engine.shape_ids[
                spatial_search_engine.resolve_query_index(query_string)
            ]
            query_file = find_mesh_path(arguments.model_dir, resolved_id)
        except ValueError:
            query_file = (
                spatial_search_engine.last_preprocessed_mesh
                if spatial_search_engine.last_preprocessed_mesh is not None
                else Path(query_string)
            )

        matches = [
            (find_mesh_path(arguments.model_dir, shape_id), dist)
            for shape_id, dist in list(results.items())[: arguments.max_matches]
        ]

        rows_data.append({"query_file": query_file, "matches": matches})

    render_meshes_grid(rows_data=rows_data, output_path=arguments.output)


if __name__ == "__main__":
    main()
