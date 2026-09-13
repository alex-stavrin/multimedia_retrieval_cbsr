from model_viewer import ModelViewer
from vedo import Mesh
from pathlib import Path

def main():
    model_viewer : ModelViewer = ModelViewer("Multimedia Retrieval CBSR", window_size=(1000, 1000), background_color="black")

    mesh_path_name : str = "3d_models/Bicycle/D00033.obj"
    mesh_path : Path = Path(__file__).resolve().parent / mesh_path_name
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found : {mesh_path}")

    print("hello?")

    models : list[Mesh] = [Mesh(mesh_path)]

    model_viewer.run(models)
    model_viewer.close()
    

# checks if the current file is being run directly using "python main.py" and not imported
if __name__ == "__main__":
    main()