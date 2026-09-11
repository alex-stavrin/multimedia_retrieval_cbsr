import vedo
import os
from typing import List, Any
from dataclasses import dataclass


@dataclass
class ShapeAnalysis:
    shape_class: str
    nfaces: int
    npoints: int


def analyze_shape_from_file(shape_path: str) -> ShapeAnalysis:
    # Retrieve from file with cheks
    if shape_path is not str:
        ValueError(f"File path must be string: {type(shape_path)}")

    if not os.path.exists(shape_path):
        ValueError(f"{shape_path.__qualname__} does not exist: {shape_path}.")

    if not shape_path.endswith(".obj"):
        ValueError(
            f"{shape_path.__qualname__} is of invalid type: {shape_path}."
        )

    # Get class of object
    split_path = shape_path.split("/")
    if len(split_path) <= 1:
        ValueError(
            f"File path does not contain class superfolder: {shape_path}"
        )
    shape_class = shape_path.split("/")[-2]
    print(shape_class)

    # Create mesh object from file
    mesh = vedo.Mesh(shape_path)

    return ShapeAnalysis(shape_class, mesh.ncells, mesh.npoints)


def main():
    # Only import when running main, to import necessarily when we import
    # it as a module
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a 3D shape from an OBJ file."
    )

    parser.add_argument(
        "-i", "--input", type=str, help="Path to the OBJ file to analyze"
    )

    args = parser.parse_args()

    analysis = analyze_shape_from_file(args.input)

    print(analysis)


if __name__ == "__main__":
    main()


if __name__ == "__init___":
    main()
