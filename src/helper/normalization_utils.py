import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

TOLERANCE = 1e-6


def find_obj_files(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".obj"):
                paths.append(os.path.join(root, name))
    return paths


def run_on_database(input_folder, output_folder, process_file, workers=None):
    paths = find_obj_files(input_folder)

    # numpy runs out of memory on some pcs if every process uses many threads
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for path in paths:
            output_path = os.path.join(
                output_folder, os.path.relpath(path, input_folder)
            )
            futures[executor.submit(process_file, path, output_path)] = path

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
                done += 1
            except Exception as error:
                print(f"Skipping {futures[future]}: {error}")

    print(f"Normalized {done} of {len(paths)} shapes")


def read_vertices(path):
    # only reads the "v x y z" lines, so the check does not need vedo
    vertices = []
    with open(path) as file:
        for line in file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    return np.array(vertices)


def check_shapes(folder, aligned=False):
    paths = find_obj_files(folder)
    center_ok = size_ok = diagonal_ok = order_ok = flip_ok = 0

    for path in tqdm(paths, desc="Checking"):
        vertices = read_vertices(path)

        if np.linalg.norm(vertices.mean(axis=0)) < TOLERANCE:
            center_ok += 1

        longest_side = (vertices.max(axis=0) - vertices.min(axis=0)).max()
        if abs(longest_side - 1) < TOLERANCE:
            size_ok += 1

        if aligned:
            cov = np.cov(vertices, rowvar=False)
            off_diagonal = max(abs(cov[0, 1]), abs(cov[0, 2]), abs(cov[1, 2]))
            if off_diagonal / cov[0, 0] < TOLERANCE:
                diagonal_ok += 1

            # small slack, because two variances can be almost equal
            if cov[0, 0] >= cov[1, 1] - 1e-5 and cov[1, 1] >= cov[2, 2] - 1e-5:
                order_ok += 1

            moments = np.sum(np.sign(vertices) * vertices**2, axis=0)
            if np.all(moments >= -TOLERANCE * np.sum(vertices**2)):
                flip_ok += 1

    n = len(paths)
    print(f"Barycenter at the origin: {center_ok}/{n}")
    print(f"Longest side is 1: {size_ok}/{n}")
    if aligned:
        print(f"Covariance is diagonal: {diagonal_ok}/{n}")
        print(f"Variance ordered x >= y >= z: {order_ok}/{n}")
        print(f"Moment test positive on all axes: {flip_ok}/{n}")


def run_cli(description, process_file, aligned=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", required=True, help="obj file or folder")
    parser.add_argument("-o", "--output", required=True, help="where to save")
    parser.add_argument(
        "-j", "--workers", type=int, default=None, help="number of processes"
    )
    parser.add_argument(
        "-v", "--verify", action="store_true", help="check the saved shapes"
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        run_on_database(args.input, args.output, process_file, args.workers)
        if args.verify:
            check_shapes(args.output, aligned)
    else:
        process_file(args.input, args.output)
