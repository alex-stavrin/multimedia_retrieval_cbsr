from src.step_1.model_viewer import ModelViewer
from vedo import Mesh
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
import numpy


def main():

    mesh_path_names: list[str] = sys.argv[1:]

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
        qt_app=app, meshes=meshes, models_delta=numpy.array([0.5, 0, 0])
    )

    return model_viewer.exec()


# checks if the current file is being run directly using "python main.py"
# and not imported
if __name__ == "__main__":
    sys.exit(main())
