from vedo import Plotter, Mesh, Text2D

# This is a abstraction class to handle model/mesh viewing using the Plotter class of vedo
class ModelViewer:
    def __init__(self, window_title : str, window_size : tuple[int, int], background_color : str):
        self.viewer : Plotter = Plotter(
            title=window_title,
            size=window_size,
            bg=background_color
        )   

        # We have to implement scroll ourself in this viewer mode (mode=8)
        self.viewer.interactor.AddObserver("MouseWheelForwardEvent", self.on_scroll)
        self.viewer.interactor.AddObserver("MouseWheelBackwardEvent", self.on_scroll)

    def on_scroll(self, interactor, event) -> None:
        factor = 1.15 if event == "MouseWheelForwardEvent" else 1 / 1.15
        camera = self.viewer.camera

        if camera.GetParallelProjection():
            camera.SetParallelScale(camera.GetParallelScale() / factor)
        else:
            camera.Dolly(factor)

        self.viewer.renderer.ResetCameraClippingRange()
        self.viewer.render()
        

    def run(self, meshes : list[Mesh]) -> None:
        self.viewer.show(
            *meshes,
            axes=8,
            mode=8,
            interactive=True
        )

    def close(self) -> None:
        self.viewer.close()

    def add_mesh_name_text(self, mesh_name):
        mesh_name_text = Text2D(
            mesh_name,
            pos="bottom-right",
            c="white",
            s=1.2
        )
        self.viewer.add(mesh_name_text)