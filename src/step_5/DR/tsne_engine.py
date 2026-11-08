"""Dimensionality reduction engine of Step 5: reduces the feature vectors of the
database to 2D with t-SNE, so they can be shown in a scatterplot."""

import argparse
from pathlib import Path

import numpy
import pandas
from tqdm import tqdm

from src.helper.path_helpers import normalize_shape_path
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
        search_engine = SpatialSearchEngine(
            elementary_csv, histogram_csv, balance=False
        )
        self.shape_ids: list[str] = search_engine.shape_ids
        self.feature_columns: list[str] = search_engine.feature_columns
        self.feature_matrix: numpy.ndarray = search_engine.raw_feature_matrix
        self.shape_classes: numpy.ndarray = numpy.array(
            [shape_id.split("/")[0] for shape_id in self.shape_ids]
        )

    def balanced_features(self) -> numpy.ndarray:
        """Scale every group of columns (the scalars together, and each histogram
        on its own) so that the group has a total variance of 1.

        With 320 histogram columns and only 6 scalars, the Euclidean distance
        would otherwise be decided by whichever group has the largest values."""
        return SpatialSearchEngine.balance_feature_matrix(
            self.feature_matrix, self.feature_columns
        )

    def step4_distances(self, calculator) -> numpy.ndarray:
        """The distances between all pairs of shapes, as the search of Step 4 gives
        them (weighted descriptor distances), in the order of shape_ids.

        calculator is a ShapeDistanceCalculator that has the shapes of this
        engine in its csv files."""
        positions = {shape_id: row for row, shape_id in enumerate(self.shape_ids)}
        distances = numpy.zeros((len(self.shape_ids), len(self.shape_ids)))
        for row, shape_id in enumerate(tqdm(self.shape_ids, desc="Step 4 distances")):
            # the shape is looked up in the csv files, and not run through the
            # pipeline again
            for other, distance in calculator.calculate_total_distance(
                shape_id, run_pipeline=False
            ).items():
                column = positions.get(normalize_shape_path(other))
                if column is not None:
                    distances[row, column] = distance
        # the distance of A to B can differ a little from B to A, t-SNE needs one
        return (distances + distances.T) / 2

    def reduce(
        self,
        perplexity: float = 30.0,
        random_state: int = 0,
        balance: bool = True,
        distances: numpy.ndarray | None = None,
    ) -> numpy.ndarray:
        """t-SNE of all shapes as an (N, 2) array, in the order of shape_ids.

        By default the Euclidean distance between the feature vectors is used.
        With a matrix of distances (N, N) these are used instead."""
        # imported here so that the scatterplot, which only reads the csv file
        # of the points, does not need scikit-learn
        from sklearn.manifold import TSNE

        if distances is not None:
            # scikit-learn does not allow the pca start for given distances
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                metric="precomputed",
                init="random",
                random_state=random_state,
            )
            return tsne.fit_transform(distances)

        features = self.balanced_features() if balance else self.feature_matrix
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            random_state=random_state,
        )
        return tsne.fit_transform(features)

    def class_agreement(
        self,
        points: numpy.ndarray | None = None,
        k: int = 10,
        distances: numpy.ndarray | None = None,
    ) -> float:
        """Share of the k nearest neighbours of a point that have the same class,
        averaged over all points. It tells how well the classes are kept
        together, and is used to compare the 2D plot with the full vectors.

        The neighbours are found with the Euclidean distance between the points,
        or with a matrix of distances (N, N) when it is given."""
        if distances is None:
            from sklearn.neighbors import NearestNeighbors

            neighbour_indices = (
                NearestNeighbors(n_neighbors=k + 1)
                .fit(points)
                .kneighbors(points, return_distance=False)[:, 1:]
            )
        else:
            # the shape itself has distance 0, so it is the first one
            neighbour_indices = numpy.argsort(distances, axis=1)[:, 1 : k + 1]
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
    parser.add_argument(
        "--distance",
        choices=["euclidean", "step4"],
        default="step4",
        help="step4: the weighted distance of the Step 4 search (needs "
        "--weights_csv), euclidean: between the feature vectors",
    )
    parser.add_argument(
        "--weights_csv",
        type=str,
        default="analysis_output/distances/weights_emd_hist_sampling_descriptors.csv",
    )
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT)

    arguments = parser.parse_args()

    engine = DimensionalityReductionEngine(
        arguments.elementary_csv, arguments.histogram_csv
    )

    distances = None
    if arguments.distance == "step4":
        # imported here because the Step 4 code loads the mesh libraries
        from src.step_4.distance_calculations.calculate_total_distance import (
            DISTANCE_CONFIG_EMD,
            ShapeDistanceCalculator,
        )

        calculator = ShapeDistanceCalculator(
            csv_databases=[
                arguments.elementary_csv,
                arguments.histogram_csv,
                arguments.weights_csv,
            ],
            distance_config=DISTANCE_CONFIG_EMD,
        )
        distances = engine.step4_distances(calculator)

    points = engine.reduce(
        perplexity=arguments.perplexity,
        random_state=arguments.seed,
        balance=not arguments.no_balance,
        distances=distances,
    )
    engine.save(points, arguments.output)

    if distances is None:
        features = (
            engine.feature_matrix
            if arguments.no_balance
            else engine.balanced_features()
        )
        before = engine.class_agreement(features)
    else:
        before = engine.class_agreement(distances=distances)
    print(
        f"{len(points)} shapes written to {arguments.output}\n"
        f"10 nearest neighbours with the same class: "
        f"{before:.3f} before the reduction, "
        f"{engine.class_agreement(points):.3f} in the 2D plot"
    )


if __name__ == "__main__":
    main()
