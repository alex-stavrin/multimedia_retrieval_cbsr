import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.step_4.distance_calculations.calculate_total_distance import (
    ShapeDistanceCalculator,
    DISTANCE_CONFIG,
    IDENTIFIER_COLUMN,
)

CHECKPOINT_INTERVAL = 2

OUTPUT_COLUMNS = [
    "mesh_a",
    "mesh_b",
    "distance_function",
    "descriptor",
    "distance",
]


def get_all_mesh_identifiers(
    csv_paths: list[str | Path],
    identifier_column: str = IDENTIFIER_COLUMN,
) -> list[str]:
    """Collect unique mesh identifiers from all descriptor CSV files."""
    identifiers = set()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, usecols=[identifier_column])
        identifiers.update(df[identifier_column].dropna().astype(str).unique())

    return sorted(identifiers)


def save_checkpoint(
    checkpoint_path: Path,
    completed_queries: set[str],
    total_meshes: int,
) -> None:
    """Save progress to a pickle file, using atomic replacement."""
    checkpoint = {
        "completed_queries": completed_queries,
        "total_meshes": total_meshes,
    }

    temporary_path = checkpoint_path.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(checkpoint, file)
        file.flush()

        # Ensure buffered data reaches the operating system.
        import os

        os.fsync(file.fileno())

    # Avoid leaving a partially written checkpoint if interrupted.
    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    total_meshes: int,
) -> set[str]:
    """Load previously saved checkpoint progress."""
    if not checkpoint_path.exists():
        return set()

    try:
        with checkpoint_path.open("rb") as file:
            checkpoint = pickle.load(file)

        if checkpoint.get("total_meshes") != total_meshes:
            print("Checkpoint database size changed; rebuilding progress.")
            return set()

        return set(checkpoint.get("completed_queries", set()))

    except (
        OSError,
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        ValueError,
    ):
        print("Checkpoint is unreadable; recovering from the CSV instead.")
        return set()


def get_completed_queries_from_csv(
    output_path: Path,
) -> set[str]:
    """Recover completed query meshes from the existing output CSV."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    try:
        df = pd.read_csv(output_path, usecols=["mesh_a"])
        return set(df["mesh_a"].dropna().astype(str).unique())

    except (OSError, ValueError, pd.errors.ParserError):
        raise RuntimeError(
            f"Could not read {output_path}. The CSV may be corrupted. "
            "Back it up and repair or remove it before resuming."
        )


def calculate_all_pairwise_distances(
    csv_paths: list[str | Path],
    output_path: str | Path = "all_pairwise_distances.csv",
    distance_config=None,
    identifier_column: str = IDENTIFIER_COLUMN,
) -> None:
    """Calculate unique mesh pairs and resume safely from saved progress."""

    if distance_config is None:
        distance_config = DISTANCE_CONFIG

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path.with_name(
        f"{output_path.stem}_checkpoint.pkl"
    )

    meshes = get_all_mesh_identifiers(
        csv_paths,
        identifier_column=identifier_column,
    )

    total_meshes = len(meshes)

    if total_meshes < 2:
        raise ValueError("At least two unique meshes are required.")

    total_pairs = total_meshes * (total_meshes - 1) // 2

    print(f"Found {total_meshes:,} unique meshes.")
    print(f"Total unique pairs: {total_pairs:,}")

    # Recover progress from both the checkpoint and the CSV.
    checkpoint_queries = load_checkpoint(
        checkpoint_path,
        total_meshes,
    )

    csv_queries = get_completed_queries_from_csv(output_path)

    completed_queries = checkpoint_queries | csv_queries

    # Only identifiers that can act as query meshes are relevant.
    query_meshes = meshes[:-1]
    valid_queries = set(query_meshes)
    completed_queries &= valid_queries

    # Reconcile checkpoint progress with the CSV. The CSV is the
    # source of truth for query batches already written to disk.
    if completed_queries:
        save_checkpoint(
            checkpoint_path,
            completed_queries,
            total_meshes,
        )

    remaining_queries = [
        mesh for mesh in query_meshes if mesh not in completed_queries
    ]

    completed_pairs = sum(
        total_meshes - i - 1
        for i, mesh in enumerate(query_meshes)
        if mesh in completed_queries
    )

    print(f"Already completed queries: {len(completed_queries):,}")
    print(f"Queries remaining: {len(remaining_queries):,}")
    print(f"Pairs accounted for: {completed_pairs:,}/{total_pairs:,}")

    if not remaining_queries:
        print("All query meshes have already been processed.")
        return

    # Initialize the output only if it doesn't exist yet.
    if not output_path.exists() or output_path.stat().st_size == 0:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(
            output_path,
            index=False,
        )

    # Initialize the calculator once so its CSV databases are loaded once.
    calculator = ShapeDistanceCalculator(
        csv_databases=csv_paths,
        distance_config=distance_config,
        identifier_column=identifier_column,
    )

    queries_since_checkpoint = 0

    for query_index, query_mesh in enumerate(remaining_queries, start=1):
        mesh_index = meshes.index(query_mesh)
        comparison_meshes = meshes[mesh_index + 1 :]

        results = calculator.calculate_all_distances(
            query=query_mesh,
            shapes_to_compare=comparison_meshes,
        )
        comparison_mesh_set = set(comparison_meshes)

        batch_rows = []

        for function_name, descriptor_results in results.items():
            for descriptor, distances in descriptor_results.items():
                for comparison_mesh, distance in distances.items():
                    if comparison_mesh not in comparison_mesh_set:
                        continue

                    batch_rows.append(
                        {
                            "mesh_a": query_mesh,
                            "mesh_b": comparison_mesh,
                            "distance_function": function_name,
                            "descriptor": descriptor,
                            "distance": distance,
                        }
                    )

        # Write this query's results before marking it complete.
        if batch_rows:
            pd.DataFrame(batch_rows, columns=OUTPUT_COLUMNS).to_csv(
                output_path,
                mode="a",
                header=False,
                index=False,
            )

        completed_queries.add(query_mesh)
        queries_since_checkpoint += 1

        completed_pairs += len(comparison_meshes)

        print(
            f"Query {query_index}/{len(remaining_queries)}: {query_mesh} | "
            f"{completed_pairs:,}/{total_pairs:,} pairs accounted for"
        )

        # Save progress every two completed query meshes.
        if queries_since_checkpoint >= CHECKPOINT_INTERVAL:
            save_checkpoint(
                checkpoint_path,
                completed_queries,
                total_meshes,
            )
            queries_since_checkpoint = 0
            print(f"Checkpoint saved: {checkpoint_path}")

    # Always save the final progress, even if fewer than two queries
    # have been processed since the last checkpoint.
    save_checkpoint(
        checkpoint_path,
        completed_queries,
        total_meshes,
    )

    print("\nDistance collection complete.")
    print(f"Results: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate all unique pairwise mesh distances."
    )

    parser.add_argument(
        "-i",
        "--csv_paths",
        "--input",
        nargs="+",
        required=True,
        metavar="CSV's",
        help="Descriptor CSV files.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="all_pairwise_distances.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    calculate_all_pairwise_distances(
        csv_paths=args.csv_paths,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
