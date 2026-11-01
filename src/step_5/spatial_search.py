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

    def _load_and_merge_features(
        self,
    ) -> tuple[list[str], list[str], numpy.ndarray]:
        elementary_data_frame = pandas.read_csv(self.elementary_csv)
        histogram_data_frame = pandas.read_csv(self.histogram_csv)

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
    ) -> None:
        self.elementary_csv = Path(elementary_csv)
        self.histogram_csv = Path(histogram_csv)
        self.identifier_column = IDENTIFIER_COLUMN
        self.leaf_size = 16
        self.shape_ids, self.feature_columns, self.feature_matrix = (
            self._load_and_merge_features()
        )

        self.kd_tree = KDTree(self.feature_matrix, self.leaf_size)

        # here index = row number of the KD tree
        self._id_to_index: dict[str, int] = {}
        self._filename_to_index: dict[str, int | None] = {}
        self._stem_to_index: dict[str, int | None] = {}
        self._build_lookup_tables()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--elementary_csv", type=str, required=True)
    parser.add_argument("--histogram_csv", type=str, required=True)
    parser.add_argument("-q", "--query", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    mode_group = parser.add_mutually_exclusive_group()
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
