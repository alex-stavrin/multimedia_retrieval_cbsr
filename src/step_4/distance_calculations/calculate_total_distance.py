from src.step_4.distance_calculations.distance_functions import (
    euclidean_distance,
    earth_movers_distance,
    cosine_distance,
)
from src.step_4.distance_calculations.visualisations import (
    visualize_mesh_distances,
)

from src.step_2.repairment.repair_wireframe import run_steps as repair
from src.step_2._3.resampling import resample_mesh
from src.step_3._1.full_normalization import full_normalize
from src.step_3._2.elementary_descriptors.descriptor_analyzer import (
    calculate_mesh_descriptors,
)
from src.step_3._2.histogram_descriptors.histogram_descriptors import (
    calculate_angle_histogram,
    calculate_distance_from_barycenter,
    calculate_distance_two_vertices,
    calculate_area_triangle,
    calculate_volume_tetrahedron,
)

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Sequence, Any
import json
import math
import vedo

import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

IDENTIFIER_COLUMN = "file_path"

DistanceFunction = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class DistanceConfig:
    function: DistanceFunction
    descriptors: list[str]
    histogram: bool = False


EUCLIDEAN_DESCRIPTORS = [
    "surface_area",
    "compactness",
    "rectangularity",
    "diameter",
    "convexity",
    "eccentricity",
]

COSINE_DESCRIPTORS = []

EMD_DESCRIPTORS = ["A3", "D1", "D2", "D3", "D4"]

DISTANCE_CONFIG = {
    "euclidean": DistanceConfig(
        euclidean_distance, EUCLIDEAN_DESCRIPTORS, histogram=False
    ),
    "cosine": DistanceConfig(
        cosine_distance, COSINE_DESCRIPTORS, histogram=True
    ),
    "emd": DistanceConfig(
        earth_movers_distance, EMD_DESCRIPTORS, histogram=True
    ),
}


class ShapeDistanceCalculator:
    """Calculate mesh distances using fast vectorized descriptor comparisons."""

    def __init__(
        self,
        csv_databases: list[str | Path],
        distance_config: dict[str, DistanceConfig] | None = None,
        identifier_column: str = IDENTIFIER_COLUMN,
        max_workers: int | None = None,
    ) -> None:
        self.csv_databases = [Path(path) for path in csv_databases]
        self.identifier_column = identifier_column

        # Load supplied CSVs
        self.databases: dict[Path, pd.DataFrame] = {}
        for path in self.csv_databases:
            if path.is_file():
                self.databases[path] = pd.read_csv(path)

        self.simple_df: pd.DataFrame | None = None
        self.hist_df: pd.DataFrame | None = None
        self.weights_df: pd.DataFrame | None = None

        for df in self.databases.values():
            cols = set(df.columns)
            if {"distance_function", "descriptor", "weight"}.issubset(cols):
                self.weights_df = df
            elif any(c.startswith("A3_") for c in cols):
                self.hist_df = df
            elif any(d in cols for d in EUCLIDEAN_DESCRIPTORS):
                self.simple_df = df

        # Fallback to standard paths if not found
        if self.simple_df is None:
            self.simple_df = pd.read_csv("analysis_output/simple_descriptors_processed.csv")
        if self.hist_df is None:
            self.hist_df = pd.read_csv("analysis_output/histogram_descriptors.csv")
        if self.weights_df is None:
            self.weights_df = pd.read_csv("analysis_output/weights_descriptors.csv")

        # Index both databases by file_path and align rows
        self.simple_df = self.simple_df.set_index(self.identifier_column)
        self.hist_df = self.hist_df.set_index(self.identifier_column).reindex(self.simple_df.index)

        # Parse descriptor weights
        self.weights: dict[tuple[str, str], float] = {
            (str(r["distance_function"]), str(r["descriptor"])): float(r["weight"])
            for _, r in self.weights_df.iterrows()
        }

        # Load normalization parameters (mean, std) from raw simple descriptors for pipeline standardization
        raw_csv = Path("analysis_output/simple_descriptors.csv")
        if raw_csv.is_file():
            raw_df = pd.read_csv(raw_csv)
            self._norm_stats = {
                desc: (float(raw_df[desc].mean()), float(raw_df[desc].std()))
                for desc in ["surface_area", "convexity", "eccentricity"]
                if desc in raw_df.columns
            }
        else:
            self._norm_stats = {
                "surface_area": (2.212288, 2.686136),
                "convexity": (0.418892, 0.576090),
                "eccentricity": (133.8659, 1284.303),
            }

        # Direct O(1) query key lookup map: filename or relative path -> exact index key
        self._filename_to_key: dict[str, str] = {}
        for key in self.simple_df.index:
            key_str = str(key)
            self._filename_to_key[key_str] = key_str
            self._filename_to_key[key_str.replace("\\", "/")] = key_str
            self._filename_to_key[Path(key_str).name] = key_str

        # Precompute histogram columns and normalized CDF matrices
        self._hist_cdfs: dict[str, np.ndarray] = {}
        for desc in EMD_DESCRIPTORS:
            cols = [
                c for c in self.hist_df.columns
                if c.startswith(f"{desc}_") and c[len(desc) + 1:].isdigit()
            ]
            cols.sort(key=lambda x: int(x.split("_")[1]))

            mat = self.hist_df[cols].to_numpy(dtype=float)
            mat_sum = mat.sum(axis=1, keepdims=True)
            mat_sum[mat_sum == 0] = 1.0
            # Normalized cumulative sum across bins (excluding last bin where CDF is always 1.0)
            self._hist_cdfs[desc] = np.cumsum(mat / mat_sum, axis=1)[:, :-1]

        self.all_distances: dict[str, dict[str, dict[str, float]]] = {}
        self.total_distances: dict[str, float] = {}
        self.last_query: str | None = None

    def _resolve_query_key(self, query: str) -> str:
        """Resolve a full UI path, relative path, or filename to the CSV row key in O(1)."""
        filename = Path(query).name
        if filename in self._filename_to_key:
            return self._filename_to_key[filename]

        norm_query = str(query).replace("\\", "/").strip("/")
        if norm_query in self._filename_to_key:
            return self._filename_to_key[norm_query]

        raise ValueError(
            f"Query mesh '{query}' (filename '{filename}') was not found in the descriptor database."
        )

    def process_mesh_and_extract_descriptors(
        self,
        mesh_or_path: str | Path | vedo.Mesh,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """
        Run the complete pipeline on a query mesh:
        1. Load mesh
        2. Repair wireframe holes and edges
        3. Resample mesh vertices
        4. Center, PCA-align and scale normalize
        5. Extract elementary descriptors & standardize with dataset stats
        6. Extract histogram descriptors (A3, D1, D2, D3, D4)
        """
        if isinstance(mesh_or_path, (str, Path)):
            mesh = vedo.Mesh(str(mesh_or_path))
        else:
            mesh = mesh_or_path.clone()

        # Step 2: Repair
        repaired_mesh, _ = repair(mesh)

        # Step 2.3: Resample
        resampled_mesh = resample_mesh(repaired_mesh)

        # Step 3.1: Full normalize
        full_normalize(resampled_mesh)

        # Step 3.2: Elementary descriptors
        elem_raw = calculate_mesh_descriptors(resampled_mesh)

        elem_standardized: dict[str, float] = {}
        for k in EUCLIDEAN_DESCRIPTORS:
            val = float(elem_raw[k])
            if k == "diameter":
                elem_standardized[k] = val / math.sqrt(3)
            elif k in ["compactness", "rectangularity"]:
                elem_standardized[k] = val
            elif k in self._norm_stats:
                mean, std = self._norm_stats[k]
                elem_standardized[k] = (val - mean) / std if std > 0 else 0.0
            else:
                elem_standardized[k] = val

        # Step 3.2: Histogram descriptors
        hist_descriptors: dict[str, np.ndarray] = {
            "A3": calculate_angle_histogram(resampled_mesh, n_samples=100_000, n_bins_histogram=64)[0],
            "D1": calculate_distance_from_barycenter(resampled_mesh, n_bins_histogram=64)[0],
            "D2": calculate_distance_two_vertices(resampled_mesh, n_samples=100_000, n_bins_histogram=64)[0],
            "D3": calculate_area_triangle(resampled_mesh, n_samples=100_000, n_bins_histogram=64)[0],
            "D4": calculate_volume_tetrahedron(resampled_mesh, n_samples=100_000, n_bins_histogram=64)[0],
        }

        return elem_standardized, hist_descriptors

    def calculate_total_distance_from_mesh(
        self,
        mesh_or_path: str | Path | vedo.Mesh,
    ) -> dict[str, float]:
        """
        Run the whole pipeline on the query mesh and compute total weighted
        distances against the precomputed database in < 1 second.
        """
        elem_descriptors, hist_descriptors = self.process_mesh_and_extract_descriptors(mesh_or_path)

        total_distances = np.zeros(len(self.simple_df), dtype=float)

        # 1. Vectorized Euclidean distances
        for desc in EUCLIDEAN_DESCRIPTORS:
            weight = self.weights.get(("euclidean", desc), 0.0)
            if weight == 0.0 or desc not in self.simple_df.columns:
                continue

            query_val = elem_descriptors[desc]
            candidates = self.simple_df[desc].to_numpy(dtype=float)
            total_distances += weight * np.abs(candidates - query_val)

        # 2. Vectorized EMD distances via precalculated CDF differences
        for desc in EMD_DESCRIPTORS:
            weight = self.weights.get(("emd", desc), 0.0)
            if weight == 0.0 or desc not in self._hist_cdfs:
                continue

            cdf_matrix = self._hist_cdfs[desc]
            q_hist = hist_descriptors[desc]
            q_sum = q_hist.sum()
            if q_sum == 0:
                q_sum = 1.0
            q_cdf = np.cumsum(q_hist / q_sum)[:-1]

            emd_dists = np.sum(np.abs(cdf_matrix - q_cdf), axis=1)
            total_distances += weight * emd_dists

        sorted_order = np.argsort(total_distances)
        all_keys = self.simple_df.index.to_numpy()

        self.total_distances = {
            all_keys[i]: float(total_distances[i])
            for i in sorted_order
        }

        return self.total_distances

    @staticmethod
    def _normalize_shape_path(shape: str | Path) -> str:
        """Normalize a mesh path to its category/filename identifier."""
        return Path(shape).as_posix().split("4_models_fully_normalized/", 1)[-1]

    @staticmethod
    def _expand_mesh_paths(paths: Sequence[str | Path]) -> list[str]:
        """Expand explicit mesh files and directories into a list of OBJ mesh paths."""
        mesh_paths: list[str] = []
        seen_paths: set[Path] = set()

        for item in paths:
            path = Path(item)
            if not path.exists():
                raise FileNotFoundError(f"Mesh path does not exist: '{path}'")

            if path.is_file():
                if path.suffix.lower() != ".obj":
                    raise ValueError(f"Expected an OBJ mesh file, got: '{path}'")
                candidates = [path]
            elif path.is_dir():
                candidates = sorted(
                    candidate for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix.lower() == ".obj"
                )
            else:
                raise ValueError(f"Not a file or directory: '{path}'")

            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    mesh_paths.append(candidate.as_posix())

        if not mesh_paths:
            raise ValueError("No OBJ meshes were found in the supplied paths.")

        return mesh_paths

    @staticmethod
    def _calculate_descriptor_task(
        function_name: str,
        descriptor: str,
        config: DistanceConfig,
        df: pd.DataFrame,
        query: str,
        shapes_to_compare: list[str],
        identifier_column: str,
    ) -> tuple[str, str, dict[str, float]]:
        """Calculate distances for one descriptor."""
        distances = config.function(
            df=df,
            query=query,
            shapes_to_compare=shapes_to_compare,
            descriptor=descriptor,
            identifier_column=identifier_column,
            histogram=config.histogram,
        )

        normalized_distances = {}

        for shape, distance in distances.items():
            normalized_shape = ShapeDistanceCalculator._normalize_shape_path(
                shape
            )

            if normalized_shape in normalized_distances:
                raise ValueError(
                    f"Multiple shapes normalize to the same identifier "
                    f"'{normalized_shape}' for descriptor '{descriptor}' "
                    f"using '{function_name}'"
                )

            normalized_distances[normalized_shape] = distance

        return function_name, descriptor, normalized_distances

    def calculate_all_distances(
        self,
        query: str,
        shapes_to_compare: list[str] | None = None,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Calculate and cache all descriptor distances vectorized."""
        self.last_query = query
        query_key = self._resolve_query_key(query)
        q_idx = self.simple_df.index.get_loc(query_key)

        all_distances: dict[str, dict[str, dict[str, float]]] = {
            "euclidean": {},
            "emd": {},
        }

        # Euclidean descriptors
        for desc in EUCLIDEAN_DESCRIPTORS:
            if desc in self.simple_df.columns:
                query_val = float(self.simple_df.loc[query_key, desc])
                candidates = self.simple_df[desc].to_numpy(dtype=float)
                dists = np.abs(candidates - query_val)
                all_distances["euclidean"][desc] = dict(zip(self.simple_df.index, dists))

        # EMD descriptors
        for desc in EMD_DESCRIPTORS:
            if desc in self._hist_cdfs:
                cdf_mat = self._hist_cdfs[desc]
                cdf_q = cdf_mat[q_idx]
                emd_dists = np.sum(np.abs(cdf_mat - cdf_q), axis=1)
                all_distances["emd"][desc] = dict(zip(self.simple_df.index, emd_dists))

        self.all_distances = all_distances
        return self.all_distances

    def calculate_total_distance(
        self,
        query: str,
        shapes_to_compare: list[str] | None = None,
        run_pipeline: bool = True,
    ) -> dict[str, float]:
        """
        Calculate weighted total distances for candidate meshes.
        If run_pipeline=True, executes the complete pipeline on the query mesh.
        If run_pipeline=False, looks up precomputed values directly.
        """
        self.last_query = query

        if run_pipeline and (Path(query).is_file() or isinstance(query, vedo.Mesh)):
            return self.calculate_total_distance_from_mesh(query)

        query_key = self._resolve_query_key(query)
        q_idx = self.simple_df.index.get_loc(query_key)

        total_distances = np.zeros(len(self.simple_df), dtype=float)

        # 1. Vectorized Euclidean distances
        for desc in EUCLIDEAN_DESCRIPTORS:
            weight = self.weights.get(("euclidean", desc), 0.0)
            if weight == 0.0 or desc not in self.simple_df.columns:
                continue

            query_val = float(self.simple_df.loc[query_key, desc])
            candidates = self.simple_df[desc].to_numpy(dtype=float)
            total_distances += weight * np.abs(candidates - query_val)

        # 2. Vectorized EMD distances via precalculated CDF differences
        for desc in EMD_DESCRIPTORS:
            weight = self.weights.get(("emd", desc), 0.0)
            if weight == 0.0 or desc not in self._hist_cdfs:
                continue

            cdf_mat = self._hist_cdfs[desc]
            cdf_q = cdf_mat[q_idx]
            emd_dists = np.sum(np.abs(cdf_mat - cdf_q), axis=1)
            total_distances += weight * emd_dists

        sorted_order = np.argsort(total_distances)
        all_keys = self.simple_df.index.to_numpy()

        self.total_distances = {
            all_keys[i]: float(total_distances[i])
            for i in sorted_order
        }

        return self.total_distances

    def _format_results(
        self,
        query: str,
        total_distances: dict[str, float],
        all_distances: dict[str, dict[str, dict[str, float]]],
    ) -> dict[Any, Any]:
        """Combine total distances and descriptor distances into a dictionary."""
        query_path = self._resolve_query_key(query) if not Path(query).is_file() else Path(query).name
        results = {}

        for mesh_path, total_distance in total_distances.items():
            descriptor_distances = {}
            for descriptors in all_distances.values():
                for descriptor, distances in descriptors.items():
                    if mesh_path in distances:
                        descriptor_distances[descriptor] = distances[mesh_path]

            results[mesh_path] = {
                "total_distance": total_distance,
                "descriptor_distances": descriptor_distances,
            }

        return {
            "query": query_path,
            "results": results,
        }

    def save_results(
        self,
        output_path: str | Path,
        query: str | None = None,
        results: dict[Any, Any] | None = None,
    ) -> None:
        """Save formatted distance results to a JSON file."""
        if not results:
            if not self.total_distances:
                raise ValueError("No total distances calculated to save.")
            if not self.all_distances:
                self.calculate_all_distances(query or self.last_query or "")

            results = self._format_results(
                query=query or self.last_query or "",
                all_distances=self.all_distances,
                total_distances=self.total_distances,
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

    def get_distances(
        self,
        function_name: str,
        descriptor: str,
    ) -> dict[str, float]:
        """Return the most recently calculated descriptor distances."""
        try:
            return self.all_distances[function_name][descriptor]
        except KeyError as exc:
            raise KeyError(
                f"No calculated results for function '{function_name}' "
                f"and descriptor '{descriptor}'. "
                "Call calculate_all_distances() first."
            ) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_paths",
        nargs="+",
        metavar="CSV",
        required=True,
        type=str,
        help="File path(s) to csv file(s) containing the descriptor values",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Query mesh identifier, filename, or path.",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        nargs="*",
        metavar="MESH",
        default=None,
        help="Comparison mesh identifiers or paths.",
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Visualises the distances between shapes in interactive 3D renderer",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file directory for json file storing calculation result",
    )

    args = parser.parse_args()

    calculator = ShapeDistanceCalculator(csv_databases=args.csv_paths)
    total_distances = calculator.calculate_total_distance(args.query, args.shapes, run_pipeline=True)
    print("Total distances calculated. Top 10:")
    for k, v in list(total_distances.items())[:10]:
        print(f"  {k}: {v:.4f}")

    if args.output:
        calculator.save_results(args.output)

    if args.visualise:
        shapes = args.shapes or list(total_distances.keys())[:10]
        visualize_mesh_distances(
            query=args.query,
            shapes_to_compare=shapes,
            descriptor="total distance",
            distances=total_distances,
        )


if __name__ == "__main__":
    main()
