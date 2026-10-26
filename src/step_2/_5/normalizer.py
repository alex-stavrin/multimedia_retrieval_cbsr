import os

import vedo

from src.helper.normalization_utils import run_cli


def center(mesh):
    # move the average vertex to (0, 0, 0)
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)


def scale_to_unit_cube(mesh):
    # divide by the longest side of the bounding box so that it becomes 1
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds()
    longest_side = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if longest_side <= 0:
        raise ValueError("shape has no size")
    mesh.vertices = mesh.vertices / longest_side


def normalize_file(input_path, output_path):
    mesh = vedo.Mesh(input_path)

    center(mesh)
    scale_to_unit_cube(mesh)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.write(output_path)


def main():
    run_cli("Center shapes and scale them to a unit cube", normalize_file)


if __name__ == "__main__":
    main()
