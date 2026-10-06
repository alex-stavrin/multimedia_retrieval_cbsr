from src.step_1.model_viewer import ModelViewer
from vedo import Mesh
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
import numpy


def main():

    import argparse

    arguements_parser = argparse.ArgumentParser()
    arguements_parser.add_argument("mesh_paths", nargs="+")
    arguements_parser.add_argument(
        "--delta",
        "-d",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
    )

    arguements = arguements_parser.parse_args()

    mesh_path_names = arguements.mesh_paths

    if not mesh_path_names:
        raise ValueError("Please provide at least one 3d shape")

    meshes: list[Mesh] = []
    for mesh_path_name in mesh_path_names:
        mesh_path = Path(mesh_path_name).resolve()

        if not mesh_path.is_file():
            raise FileNotFoundError(f"File {mesh_path} not found")

        meshes.append(Mesh(mesh_path))

    app = QApplication(sys.argv)
    model_viewer: ModelViewer = ModelViewer(
        qt_app=app, meshes=meshes, models_delta=numpy.array(arguements.delta)
    )

    return model_viewer.exec()


# checks if the current file is being run directly using "python main.py"
# and not imported
if __name__ == "__main__":
    sys.exit(main())
