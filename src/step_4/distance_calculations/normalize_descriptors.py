import pandas as pd
import os

NORMALIZABLE = ["compactness", "rectangularity", "diameter"]

STANDARDIZABLE = [
    "convexity",
    "eccentricity",
    "surface_area",
]


def standardize_descriptors(
    csv_path: str, columns: list[str], df: pd.DataFrame | None = None
):
    if not os.path.exists(csv_path):
        raise ValueError(f"File does not exist: {csv_path}")

    if not csv_path.endswith(".csv"):
        raise ValueError(f"File must be a csv file: {csv_path}")

    if df is None:
        df = pd.read_csv(csv_path)

    for cur_column in columns:
        df[cur_column] = (df[cur_column] - df[cur_column].mean()) / df[
            cur_column
        ].std()

    return df


def normalize_descriptor(
    csv_path: str, columns: list[str], df: pd.DataFrame | None = None
):
    if not os.path.exists(csv_path):
        raise ValueError(f"File does not exist: {csv_path}")

    if not csv_path.endswith(".csv"):
        raise ValueError(f"File must be a csv file: {csv_path}")

    if df is None:
        df = pd.read_csv(csv_path)

    for cur_column in columns:
        df[cur_column] = (df[cur_column] - df[cur_column].min()) / (
            df[cur_column].max() - df[cur_column].min()
        )

    return df


def main():
    


if __name__ == "__main__":
    main()
