from src.step_1.model_viewer import ModelViewer
from vedo import Mesh
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication


def main():

    # Look if the mesh path we were given as command line arguement actually exists
    mesh_path_name: str = sys.argv[1]
    mesh_path: Path = Path(mesh_path_name).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found : {mesh_path}")

    meshes: list[Mesh] = [Mesh(mesh_path)]

    app = QApplication(sys.argv)
    model_viewer: ModelViewer = ModelViewer(
        qt_app=app,
        meshes=meshes,
    )

    return model_viewer.exec()


# checks if the current file is being run directly using "python main.py"
# and not imported
if __name__ == "__main__":
    sys.exit(main())
