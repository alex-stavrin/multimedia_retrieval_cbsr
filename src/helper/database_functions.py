import numpy as np
import os


def retrieve_all_shape_files(root_dir: str):
    resulting_shape_paths = np.array([])

    cwd = os.getcwd() + "/"

    # Go through all directories and files
    for file_system_info in os.walk(root_dir):
        # Check if the first file, of the files in a directory, is of type .obj
        if not file_system_info[2][0].endswith(".obj"):
            continue

        cur_path = file_system_info[0] + "/"

        for file in file_system_info[2]:
            # Get absolute directory to avoid issues
            shape_path = cwd + cur_path + file
            resulting_shape_paths = np.append(
                resulting_shape_paths, shape_path
            )

    return resulting_shape_paths
