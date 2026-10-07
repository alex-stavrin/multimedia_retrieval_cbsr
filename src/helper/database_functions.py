import numpy as np
import os


def retrieve_all_shape_files(root_dir: str):
    resulting_shape_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        obj_files = [f for f in filenames if f.endswith(".obj")]

        if not obj_files:
            print(f"No .obj files found in directory: {dirpath}")
            continue

        for file in obj_files:
            resulting_shape_paths.append(
                os.path.abspath(os.path.join(dirpath, file))
            )

    return np.array(resulting_shape_paths)


def map_by_relative_path(root_dir: str) -> dict:
    """Maps 'AquaticAnimal/m51.obj' -> absolute path,
    for every shape in root_dir."""
    root_abs = os.path.abspath(root_dir)
    return {
        os.path.relpath(path, root_abs): path
        for path in retrieve_all_shape_files(root_dir)
    }
