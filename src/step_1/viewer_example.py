from model_viewer import ModelViewer
from vedo import Mesh, Points
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication


def main():

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
        mesh.c("#A0A0A0")
        mesh.lighting("plastic")

        # make edges of mesh stand out
        mesh.linewidth(3)
        mesh.linecolor("#000000")

        # make vertices of mesh stand out
        points: Points = Points(mesh.vertices, r=10, c="#000000")
        vertices.append(points)

    app = QApplication(sys.argv)
    model_viewer: ModelViewer = ModelViewer(
        qt_app=app,
        window_title="Multimedia Retrieval CBSR",
        window_size=(1000, 1000),
        background_color="#FFFFFF",
        meshes=meshes,
        points=vertices,
    )

    model_viewer.add_mesh_name_text(mesh_path_name)

    return model_viewer.exec()


# checks if the current file is being run directly using "python main.py"
# and not imported
if __name__ == "__main__":
    sys.exit(main())
