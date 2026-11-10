import argparse
import numpy
import pandas
from pathlib import Path
import vedo
from src.helper.mesh_descriptor_extraction import calculate_all_descriptors
from src.helper.mesh_preprocessing import preprocess_mesh
from src.helper.path_helpers import normalize_shape_path
from scipy.spatial import KDTree

from src.step_4.distance_calculations.calculate_total_distance import (
    IDENTIFIER_COLUMN,
    ELEMENTAL_DESCRIPTORS,
    HISTOGRAM_DESCRIPTORS,
)


class SpatialSearchEngine:

    _PROCESSED_SUFFIXES = ("_processed", "_normalized")

    @classmethod
    def _is_processed_stem(cls, stem: str) -> bool:
        return any(stem.endswith(suffix) for suffix in cls._PROCESSED_SUFFIXES)

    @classmethod
    def _raw_stem(cls, stem: str) -> str:
        for suffix in cls._PROCESSED_SUFFIXES:
            if stem.endswith(suffix):
                return stem.removesuffix(suffix)
        return stem

    def is_in_database(self, query: str | Path) -> bool:
        """Return True if the query resolves to a precomputed shape in the database."""
        try:
            self.resolve_query_index(query)
            return True
        except ValueError:
            return False

    def get_feature_vector(self, query: str | Path | vedo.Mesh) -> numpy.ndarray:
        _, vector = self._resolve_query_vector(query)
        return vector

    def resolve_query_index(self, query: str | Path) -> int:
        """Resolve a shape path, 'Class/file.obj', filename, or stem to its row index."""
        raw = str(query).replace("\\", "/").strip("/")
        if raw in self._id_to_index:
            return self._id_to_index[raw]

        if Path(query).is_file():
            resolved_norm = normalize_shape_path(Path(query).resolve())
            if resolved_norm in self._id_to_index:
                return self._id_to_index[resolved_norm]
            fname = Path(query).name
            if fname in self._id_to_index:
                return self._id_to_index[fname]
            raise ValueError(
                f"Query shape '{query}' not found in feature database."
            )

        norm = normalize_shape_path(raw)
        if norm in self._id_to_index:
            return self._id_to_index[norm]

        fname = Path(norm).name
        if "." in fname:
            if fname in self._filename_to_index:
                idx = self._filename_to_index[fname]
                if idx is None:
                    raise ValueError(
                        f"Ambiguous filename '{query}'. Provide 'Class/{fname}'."
                    )
                return idx
        else:
            if fname in self._stem_to_index:
                idx = self._stem_to_index[fname]
                if idx is None:
                    raise ValueError(
                        f"Ambiguous mesh stem '{query}'. Provide 'Class/{fname}.obj'."
                    )
                return idx

        raise ValueError(f"Query shape '{query}' not found in feature database.")

    def _load_and_merge_features(self) -> tuple[list[str], list[str], numpy.ndarray]:
        elementary_data_frame = pandas.read_csv(self.elementary_csv).copy()
        histogram_data_frame = pandas.read_csv(self.histogram_csv).copy()

        elementary_data_frame["_norm_id"] = elementary_data_frame[
            self.identifier_column
        ].map(normalize_shape_path)
        histogram_data_frame["_norm_id"] = histogram_data_frame[
            self.identifier_column
        ].map(normalize_shape_path)

        merged_data_frame = pandas.merge(
            elementary_data_frame,
            histogram_data_frame,
            on="_norm_id",
            how="inner",
            suffixes=("", "_hist"),
        )

        elementary_columns = [
            col
            for col in ELEMENTAL_DESCRIPTORS
            if col in merged_data_frame.columns
        ]
        histogram_columns: list[str] = []
        for histogram_prefix in HISTOGRAM_DESCRIPTORS:
            bins = [
                col
                for col in merged_data_frame.columns
                if col.startswith(f"{histogram_prefix}_")
                and col[len(histogram_prefix) + 1 :].isdigit()
            ]
            bins.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
            histogram_columns.extend(bins)

        self.elementary_columns = elementary_columns
        self.histogram_columns = histogram_columns
        self._normalize_by_sum = True
        for histogram_prefix in HISTOGRAM_DESCRIPTORS:
            bins = [
                col
                for col in histogram_columns
                if col.startswith(f"{histogram_prefix}_")
            ]
            if bins:
                row_sums = (
                    merged_data_frame[bins]
                    .to_numpy(dtype=numpy.float64)
                    .sum(axis=1)
                )
                mean_sum = float(numpy.nanmean(row_sums))
                self._normalize_by_sum = abs(mean_sum - 1.0) < abs(
                    mean_sum - len(bins)
                )
                break

        self._query_transforms: dict[str, tuple[float, float]] = {}
        if self._is_processed_stem(self.elementary_csv.stem):
            sibling_path = self.elementary_csv.with_name(
                f"{self._raw_stem(self.elementary_csv.stem)}.csv"
            )
            if sibling_path.is_file():
                raw_df = pandas.read_csv(sibling_path).copy()
                raw_df["_norm_id"] = raw_df[self.identifier_column].map(
                    normalize_shape_path
                )
                raw_indexed = raw_df.set_index("_norm_id")
                norm_indexed = elementary_data_frame.set_index("_norm_id")
                for descriptor in elementary_columns:
                    if descriptor not in raw_indexed.columns:
                        continue
                    paired = pandas.concat(
                        [
                            raw_indexed[descriptor].rename("raw"),
                            norm_indexed[descriptor].rename("fit"),
                        ],
                        axis=1,
                        join="inner",
                    ).dropna()
                    x = paired["raw"].to_numpy(dtype=numpy.float64)
                    y = paired["fit"].to_numpy(dtype=numpy.float64)
                    variance = float(numpy.var(x))
                    if variance == 0.0:
                        scale = 0.0
                        offset = float(numpy.mean(y))
                    else:
                        scale = float(
                            numpy.mean((x - x.mean()) * (y - y.mean()))
                            / variance
                        )
                        offset = float(y.mean() - scale * x.mean())
                    self._query_transforms[descriptor] = (scale, offset)

        feature_columns = elementary_columns + histogram_columns

        # make the matrix NxM. Where N is the number of shapes and M is the number of descriptor columns (elementary + histogram)
        matrix = merged_data_frame[feature_columns].to_numpy(
            dtype=numpy.float64
        )
        shape_ids = merged_data_frame["_norm_id"].astype(str).tolist()
        return shape_ids, feature_columns, matrix

    @staticmethod
    def _compute_feature_groups(
        matrix: numpy.ndarray,
        feature_columns: list[str],
    ) -> tuple[dict[str, list[int]], dict[str, float]]:
        groups: dict[str, list[int]] = {}
        for index, column in enumerate(feature_columns):
            prefix, _, bin_index = column.rpartition("_")
            name = prefix if bin_index.isdigit() and prefix else "elementary"
            groups.setdefault(name, []).append(index)

        scales: dict[str, float] = {}
        for name, indices in groups.items():
            block = matrix[:, indices]
            total_variance = float(block.var(axis=0).sum())
            scales[name] = (
                float(numpy.sqrt(total_variance))
                if total_variance > 0
                else 1.0
            )
        return groups, scales

    @staticmethod
    def balance_feature_matrix(
        matrix: numpy.ndarray,
        feature_columns: list[str],
    ) -> numpy.ndarray:
        groups, scales = SpatialSearchEngine._compute_feature_groups(
            matrix, feature_columns
        )
        balanced = numpy.empty_like(matrix)
        for name, indices in groups.items():
            scale = scales[name]
            block = matrix[:, indices]
            balanced[:, indices] = block / scale if scale > 0 else block
        return balanced

    def _build_lookup_tables(self) -> None:
        for idx, shape_id in enumerate(self.shape_ids):
            self._id_to_index[shape_id] = idx
            norm = shape_id.replace("\\", "/").strip("/")
            self._id_to_index[norm] = idx

            fname = Path(norm).name
            if fname in self._filename_to_index:
                self._filename_to_index[fname] = None
            else:
                self._filename_to_index[fname] = idx

            stem = Path(fname).stem
            if stem in self._stem_to_index:
                self._stem_to_index[stem] = None
            else:
                self._stem_to_index[stem] = idx

    def __init__(
        self,
        elementary_csv: str | Path,
        histogram_csv: str | Path,
        balance: bool = True,
    ) -> None:
        self.elementary_csv = Path(elementary_csv)
        self.histogram_csv = Path(histogram_csv)
        self.identifier_column = IDENTIFIER_COLUMN
        self.leaf_size = 16
        self.balance = balance
        self.elementary_columns: list[str] = []
        self.histogram_columns: list[str] = []
        self._normalize_by_sum: bool = True
        self._query_transforms: dict[str, tuple[float, float]] = {}
        self.last_preprocessed_mesh: vedo.Mesh | None = None

        self.shape_ids, self.feature_columns, self.raw_feature_matrix = (
            self._load_and_merge_features()
        )
        self._group_indices, self._group_scales = self._compute_feature_groups(
            self.raw_feature_matrix, self.feature_columns
        )
        self.feature_matrix = (
            self.balance_feature_matrix(
                self.raw_feature_matrix, self.feature_columns
            )
            if balance
            else self.raw_feature_matrix
        )

        self.kd_tree = KDTree(self.feature_matrix, self.leaf_size)

        # here index = row number of the KD tree
        self._id_to_index: dict[str, int] = {}
        self._filename_to_index: dict[str, int | None] = {}
        self._stem_to_index: dict[str, int | None] = {}
        self._build_lookup_tables()

    def process_mesh_and_extract_feature_vector(
        self,
        mesh_or_path: str | Path | vedo.Mesh,
    ) -> numpy.ndarray:
        """Run the preprocessing and descriptor pipeline on an out-of-database mesh."""
        is_fully_normalized_file = False
        if isinstance(mesh_or_path, (str, Path)):
            mesh_path = Path(mesh_or_path)
            if not mesh_path.is_file():
                raise ValueError(
                    f"Query shape '{mesh_or_path}' not found in feature database "
                    "and does not exist as a mesh file."
                )
            mesh = vedo.Mesh(str(mesh_path))
            is_fully_normalized_file = any(
                part.endswith("_models_fully_normalized")
                for part in mesh_path.parts
            )
        elif isinstance(mesh_or_path, vedo.Mesh):
            mesh = mesh_or_path
        else:
            raise TypeError(f"Unsupported query type: {type(mesh_or_path)}")

        if not is_fully_normalized_file:
            mesh = preprocess_mesh(mesh)
        self.last_preprocessed_mesh = mesh

        scalar_descriptors = set(self.elementary_columns)
        histogram_bins: dict[str, int] = {}
        for prefix in HISTOGRAM_DESCRIPTORS:
            bins = [
                col
                for col in self.histogram_columns
                if col.startswith(f"{prefix}_")
            ]
            if bins:
                histogram_bins[prefix] = len(bins)

        if (
            self._is_processed_stem(self.elementary_csv.stem)
            and scalar_descriptors - self._query_transforms.keys()
        ):
            missing = sorted(scalar_descriptors - self._query_transforms.keys())
            raise ValueError(
                "Cannot standardize query elementary descriptors for "
                f"{missing}; raw simple descriptors CSV was not found."
            )

        scalar_values, histogram_values = calculate_all_descriptors(
            mesh,
            scalar_descriptors,
            histogram_bins,
            normalize_by_sum=self._normalize_by_sum,
        )

        for descriptor, (scale, offset) in self._query_transforms.items():
            if descriptor in scalar_values:
                scalar_values[descriptor] = (
                    scale * scalar_values[descriptor] + offset
                )

        raw_vector = numpy.zeros(len(self.feature_columns), dtype=numpy.float64)
        for idx, col in enumerate(self.feature_columns):
            if col in scalar_values:
                raw_vector[idx] = scalar_values[col]
            else:
                prefix, _, bin_idx_str = col.rpartition("_")
                raw_vector[idx] = histogram_values[prefix][int(bin_idx_str)]

        if self.balance:
            balanced_vector = numpy.empty_like(raw_vector)
            for name, indices in self._group_indices.items():
                scale = self._group_scales[name]
                balanced_vector[indices] = (
                    raw_vector[indices] / scale
                    if scale > 0
                    else raw_vector[indices]
                )
            return balanced_vector

        return raw_vector

    def _resolve_query_vector(
        self,
        query: str | Path | vedo.Mesh,
    ) -> tuple[str | None, numpy.ndarray]:
        """Return (database_shape_id, feature_vector), running the pipeline if outside DB."""
        if not isinstance(query, vedo.Mesh):
            try:
                query_index = self.resolve_query_index(query)
                self.last_preprocessed_mesh = None
                return (
                    self.shape_ids[query_index],
                    self.feature_matrix[query_index],
                )
            except ValueError:
                if not Path(query).is_file():
                    raise
        return None, self.process_mesh_and_extract_feature_vector(query)

    # p is for the distance. p=2 is euclidean. p=1 is manhattan.
    # eps will be how accurate the approximation in ANN (Approximate Nearest Neighbors).
    # eps = 0 is exact. eps = 1 100% tolerance or something.
    def query_range(
        self,
        query: str | Path | vedo.Mesh,
        radius: float,
        eps: float = 0.0,
        p: float = 2.0,
        exclude_self: bool = False,
    ) -> dict[str, float]:
        query_id, query_vector = self._resolve_query_vector(query)

        result_indices: list[int] = self.kd_tree.query_ball_point(
            query_vector, r=radius, p=p, eps=eps
        )

        result_matrix = self.feature_matrix[result_indices]
        distances = numpy.linalg.norm(result_matrix - query_vector, ord=p, axis=1)

        # make the dictionary of (result_shape, distance)
        pairs: list[tuple[str, float]] = []
        for index, distance in zip(result_indices, distances):
            result_id = self.shape_ids[int(index)]
            if exclude_self and query_id is not None and result_id == query_id:
                continue
            pairs.append((result_id, distance))

        pairs.sort(key=lambda item: item[1])
        return dict(pairs)

    def query_knn(
        self,
        query: str | Path | vedo.Mesh,
        k: int = 6,
        eps: float = 0.0,
        p: float = 2.0,
        exclude_self: bool = False,
    ) -> dict[str, float]:
        query_id, query_vector = self._resolve_query_vector(query)

        total_number_of_shapes = len(self.shape_ids)
        should_exclude_self = exclude_self and query_id is not None
        k_return = min(
            k + 1 if should_exclude_self else k, total_number_of_shapes
        )

        distances, indices = self.kd_tree.query(query_vector, k=k_return, eps=eps, p=p)

        distances_array = numpy.atleast_1d(distances)
        indices_array = numpy.atleast_1d(indices)

        results: dict[str, float] = {}
        for distance, index in zip(distances_array, indices_array):
            index_int = int(index)
            # check if sci pi returns sometihng stupid if it cannot find k neighbors
            if index_int >= total_number_of_shapes:
                continue
            result_id = self.shape_ids[index_int]
            if should_exclude_self and result_id == query_id:
                continue
            results[result_id] = float(distance)
            if len(results) >= k:
                break

        return results



def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--elementary_csv", type=str, required=True)
    parser.add_argument("--histogram_csv", type=str, required=True)
    parser.add_argument("-q", "--query", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-k", type=int)
    mode_group.add_argument("-r", "--radius", type=float)

    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--p", type=float, default=2.0)

    arguments = parser.parse_args()

    spatial_search_engine = SpatialSearchEngine(
        elementary_csv=arguments.elementary_csv,
        histogram_csv=arguments.histogram_csv,
    )


if __name__ == "__main__":
    main()
