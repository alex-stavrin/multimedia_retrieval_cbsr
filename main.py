from model_viewer import ModelViewer
from vedo import Mesh, Points
from pathlib import Path


def main():
    model_viewer: ModelViewer = ModelViewer(
        "Multimedia Retrieval CBSR",
        window_size=(1000, 1000),
        background_color="black",
    )

    mesh_path_name: str = "3d_models/Bicycle/D00033.obj"
    mesh_path: Path = Path(__file__).resolve().parent / mesh_path_name
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found : {mesh_path}")

    meshes: list[Mesh] = [Mesh(mesh_path)]
    vertices: list[Points] = []

    for mesh in meshes:
        # add shading and lighting to meshes
        mesh.phong()
        mesh.lighting("plastic")

        # make edges of mesh stand out
        mesh.linewidth(1)
        mesh.linecolor("white")

        # make vertices of mesh stand out
        points: Points = Points(mesh.vertices, r=4, c="magenta")
        vertices.append(points)

    meshes_and_points: list[Mesh | Points] = meshes + vertices
    model_viewer.add_mesh_name_text(mesh_path_name)
    model_viewer.run(meshes_and_points)
    model_viewer.close()


# checks if the current file is being run directly using "python main.py" and not imported
if __name__ == "__main__":
    main()
