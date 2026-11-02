from src.step_1.model_viewer import ModelViewer
from vedo import Mesh
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
import numpy

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():

    import argparse

    arguements_parser = argparse.ArgumentParser(
        description="View 3D shapes. Shapes can also be opened from the "
        "content browser inside the viewer."
    )
    arguements_parser.add_argument(
        "mesh_paths",
        nargs="*",
        help="Shapes to show at the start (optional)",
    )
    arguements_parser.add_argument(
        "--delta",
        "-d",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Distance between the shapes",
    )
    arguements_parser.add_argument(
        "--browse",
        "-b",
        type=str,
        default=None,
        help="Folder shown in the content browser (default: ./3d_models)",
    )

    arguements = arguements_parser.parse_args()

    meshes: list[Mesh] = []
    for mesh_path_name in arguements.mesh_paths:
        mesh_path = Path(mesh_path_name).resolve()

        if not mesh_path.is_file():
            raise FileNotFoundError(f"File {mesh_path} not found")

        meshes.append(Mesh(mesh_path))

    browse_root = Path(arguements.browse) if arguements.browse else None
    if browse_root is None:
        browse_root = PROJECT_ROOT / "3d_models"
    if not browse_root.is_dir():
        browse_root = None

    app = QApplication(sys.argv)
    model_viewer: ModelViewer = ModelViewer(
        qt_app=app,
        meshes=meshes,
        models_delta=numpy.array(arguements.delta),
        browse_root=browse_root,
    )

    return model_viewer.exec()


# checks if the current file is being run directly using "python main.py"
# and not imported
if __name__ == "__main__":
    sys.exit(main())
