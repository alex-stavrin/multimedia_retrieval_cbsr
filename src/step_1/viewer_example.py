from model_viewer import ModelViewer
from vedo import Mesh, Points
from pathlib import Path
import ctypes
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

# Wayland fix
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"


def main():

    # Give the viewer its own Windows taskbar identity.
    if sys.platform == "win32":
        ctypes.OleDLL("shell32").SetCurrentProcessExplicitAppUserModelID(
            ctypes.c_wchar_p("MultimediaRetrieval.CBSR.Viewer")
        )

    # Make the QT Application
    app = QApplication(sys.argv)

    # Add an application icon
    app_icon_path = Path(__file__).resolve().parent / "assets/icon.png"
    app.setWindowIcon(QIcon(str(app_icon_path)))

    # Look if the mesh path we were given as command line arguement
    # actually exists
    mesh_path_name: str = sys.argv[1]
    mesh_path: Path = Path(__file__).resolve().parent / mesh_path_name
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found : {mesh_path}")

    meshes: list[Mesh] = [Mesh(mesh_path)]
    vertices: list[Points] = []

    for mesh in meshes:
        # add shading and lighting to meshes
        mesh.compute_normals(points=True, cells=True)
        mesh.phong()
        mesh.lighting("plastic")

        # make edges of mesh stand out
        mesh.linewidth(1)
        mesh.linecolor("white")

        # make vertices of mesh stand out
        points: Points = Points(mesh.vertices, r=5, c="magenta")
        vertices.append(points)

    model_viewer: ModelViewer = ModelViewer(
        window_title="Multimedia Retrieval CBSR",
        window_size=(1000, 1000),
        background_color="black",
        meshes=meshes,
        points=vertices,
    )

    model_viewer.add_mesh_name_text(mesh_path_name)
    model_viewer.run()

    return app.exec()


# checks if the current file is being run directly using "python main.py"
# and not imported
if __name__ == "__main__":
    sys.exit(main())
