import os

import numpy as np
import vedo

from src.helper.normalization_utils import run_cli, run_on_database
from src.step_2._5.normalizer import center, scale_to_unit_cube


def align(mesh):
    vertices = mesh.vertices

    # PCA: the eigenvectors of the covariance matrix are the main directions
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(vertices, rowvar=False))

    # eigh sorts from small to large, we want the largest first
    order = np.argsort(eigenvalues)[::-1]
    e1 = eigenvectors[:, order[0]]
    e2 = eigenvectors[:, order[1]]

    # the third axis is e1 x e2, so we get a rotation and not a mirror
    e3 = np.cross(e1, e2)

    mesh.vertices = vertices @ np.vstack([e1, e2, e3]).T


def flip(mesh):
    vertices = mesh.vertices

    # moment test: mirror every axis where most of the shape is on the negative side
    moments = np.sum(np.sign(vertices) * vertices**2, axis=0)
    signs = np.where(moments < 0, -1.0, 1.0)
    mesh.vertices = vertices * signs

    # mirroring one or three axes turns the triangles inside out
    if np.prod(signs) < 0:
        mesh.reverse()


def full_normalize(mesh):
    center(mesh)
    align(mesh)
    flip(mesh)
    # scaling is last because the bounding box depends on the rotation
    scale_to_unit_cube(mesh)


def full_normalize_file(input_path, output_path):
    mesh = vedo.Mesh(input_path)
    full_normalize(mesh)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.write(output_path)


def full_normalize_database(input_root, output_root, max_workers=None):
    run_on_database(input_root, output_root, full_normalize_file, max_workers)


def main():
    run_cli("Center, align, flip and scale shapes", full_normalize_file, aligned=True)


if __name__ == "__main__":
    main()
