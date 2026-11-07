import argparse
import numpy
import pandas
from pathlib import Path
from src.helper.path_helpers import normalize_shape_path
from scipy.spatial import KDTree

from src.step_4.distance_calculations.calculate_total_distance import (
    IDENTIFIER_COLUMN,
    ELEMENTAL_DESCRIPTORS,
    HISTOGRAM_DESCRIPTORS,
)


class SpatialSearchEngine:

    def get_feature_vector(self, query: str | Path) -> numpy.ndarray:
        idx = self.resolve_query_index(query)
        return self.feature_matrix[idx]

    def resolve_query_index(self, query: str | Path) -> int:
        """Resolve a shape path, 'Class/file.obj', filename, or stem to its row index."""
        raw = str(query).replace("\\", "/").strip("/")
        if raw in self._id_to_index:
            return self._id_to_index[raw]

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

        feature_columns = elementary_columns + histogram_columns

        # make the matrix NxM. Where N is the number of shapes and M is the number of descriptor columns (elementary + histogram)
        matrix = merged_data_frame[feature_columns].to_numpy(
            dtype=numpy.float64
        )
        shape_ids = merged_data_frame["_norm_id"].astype(str).tolist()
        return shape_ids, feature_columns, matrix

    @staticmethod
    def balance_feature_matrix(
        matrix: numpy.ndarray,
        feature_columns: list[str],
    ) -> numpy.ndarray:
        groups: dict[str, list[int]] = {}
        for index, column in enumerate(feature_columns):
            prefix, _, bin_index = column.rpartition("_")
            name = prefix if bin_index.isdigit() and prefix else "elementary"
            groups.setdefault(name, []).append(index)

        balanced = numpy.empty_like(matrix)
        for indices in groups.values():
            block = matrix[:, indices]
            total_variance = block.var(axis=0).sum()
            balanced[:, indices] = (
                block / numpy.sqrt(total_variance) if total_variance > 0 else block
            )
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
        self.shape_ids, self.feature_columns, self.raw_feature_matrix = (
            self._load_and_merge_features()
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

    # p is for the distance. p=2 is euclidean. p=1 is manhattan.
    # eps will be how accurate the approximation in ANN (Approximate Nearest Neighbors).
    # eps = 0 is exact. eps = 1 100% tolerance or something.
    def query_range(
        self,
        query: str | Path,
        radius: float,
        eps: float = 0.0,
        p: float = 2.0,
        exclude_self: bool = False,
    ) -> dict[str, float]:
        query_index = self.resolve_query_index(query)
        query_id = self.shape_ids[query_index]
        query_vector = self.feature_matrix[query_index]

        result_indices: list[int] = self.kd_tree.query_ball_point(
            query_vector, r=radius, p=p, eps=eps
        )

        result_matrix = self.feature_matrix[result_indices]
        distances = numpy.linalg.norm(result_matrix - query_vector, ord=p, axis=1)

        # make the dictionary of (result_shape, distance)
        pairs: list[tuple[str, float]] = []
        for index, distance in zip(result_indices, distances):
            result_id = self.shape_ids[int(index)]
            if exclude_self and result_id == query_id:
                continue
            pairs.append((result_id, distance))

        pairs.sort(key=lambda item: item[1])
        return dict(pairs)

    def query_knn(
        self,
        query: str | Path,
        k: int = 6,
        eps: float = 0.0,
        p: float = 2.0,
        exclude_self: bool = False,
    ) -> dict[str, float]:
        query_index = self.resolve_query_index(query)
        query_id = self.shape_ids[query_index]
        query_vector = self.feature_matrix[query_index]

        total_number_of_shapes = len(self.shape_ids)
        k_return = min(k + 1 if exclude_self else k, total_number_of_shapes)

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
            if exclude_self and result_id == query_id:
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
