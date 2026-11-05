"""Dimensionality reduction engine of Step 5: reduces the feature vectors of the
database to 2D with t-SNE, so they can be shown in a scatterplot."""

import argparse
from pathlib import Path

import numpy
import pandas
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from src.step_5.ANN.spatial_search_engine import SpatialSearchEngine

DEFAULT_OUTPUT = "analysis_output/embeddings/tsne.csv"


class DimensionalityReductionEngine:
    """Holds the feature vectors of the database and reduces them to 2D."""

    def __init__(
        self,
        elementary_csv: str | Path,
        histogram_csv: str | Path,
    ) -> None:
        # the search engine already merges the two csv files, so the points in
        # the plot are made from the same vectors as the ones in the KD-tree
        search_engine = SpatialSearchEngine(elementary_csv, histogram_csv)
        self.shape_ids: list[str] = search_engine.shape_ids
        self.feature_columns: list[str] = search_engine.feature_columns
        self.feature_matrix: numpy.ndarray = search_engine.feature_matrix
        self.shape_classes: numpy.ndarray = numpy.array(
            [shape_id.split("/")[0] for shape_id in self.shape_ids]
        )

    def balanced_features(self) -> numpy.ndarray:
        """Scale every group of columns (the scalars together, and each histogram
        on its own) so that the group has a total variance of 1.

        With 320 histogram columns and only 6 scalars, the Euclidean distance
        would otherwise be decided by whichever group has the largest values."""
        groups: dict[str, list[int]] = {}
        for index, column in enumerate(self.feature_columns):
            prefix, _, bin_index = column.rpartition("_")
            name = prefix if bin_index.isdigit() and prefix else "elementary"
            groups.setdefault(name, []).append(index)

        balanced = numpy.empty_like(self.feature_matrix)
        for indices in groups.values():
            block = self.feature_matrix[:, indices]
            total_variance = block.var(axis=0).sum()
            balanced[:, indices] = (
                block / numpy.sqrt(total_variance) if total_variance > 0 else block
            )
        return balanced

    def reduce(
        self,
        perplexity: float = 30.0,
        random_state: int = 0,
        balance: bool = True,
    ) -> numpy.ndarray:
        """t-SNE of all shapes as an (N, 2) array, in the order of shape_ids."""
        features = self.balanced_features() if balance else self.feature_matrix
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            random_state=random_state,
        )
        return tsne.fit_transform(features)

    def class_agreement(self, points: numpy.ndarray, k: int = 10) -> float:
        """Share of the k nearest neighbours of a point that have the same class,
        averaged over all points. It tells how well the classes are kept
        together, and is used to compare the 2D plot with the full vectors."""
        neighbour_indices = (
            NearestNeighbors(n_neighbors=k + 1)
            .fit(points)
            .kneighbors(points, return_distance=False)[:, 1:]
        )
        same_class = (
            self.shape_classes[neighbour_indices] == self.shape_classes[:, None]
        )
        return float(same_class.mean())

    def save(self, points: numpy.ndarray, output_csv: str | Path) -> None:
        """Write one row per shape, so the scatterplot does not need to rerun t-SNE."""
        table = pandas.DataFrame(
            {
                "file_path": self.shape_ids,
                "shape_class": self.shape_classes,
                "x": points[:, 0],
                "y": points[:, 1],
            }
        )
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)


def load_embedding(embedding_csv: str | Path = DEFAULT_OUTPUT) -> pandas.DataFrame:
    """The saved 2D points, with the columns file_path, shape_class, x and y."""
    return pandas.read_csv(embedding_csv)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--elementary_csv", type=str, required=True)
    parser.add_argument("--histogram_csv", type=str, required=True)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="use the feature vectors of the KD-tree as they are",
    )
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT)

    arguments = parser.parse_args()

    engine = DimensionalityReductionEngine(
        arguments.elementary_csv, arguments.histogram_csv
    )
    points = engine.reduce(
        perplexity=arguments.perplexity,
        random_state=arguments.seed,
        balance=not arguments.no_balance,
    )
    engine.save(points, arguments.output)

    features = (
        engine.feature_matrix if arguments.no_balance else engine.balanced_features()
    )
    print(
        f"{len(points)} shapes written to {arguments.output}\n"
        f"10 nearest neighbours with the same class: "
        f"{engine.class_agreement(features):.3f} in the feature space, "
        f"{engine.class_agreement(points):.3f} in the 2D plot"
    )


if __name__ == "__main__":
    main()
