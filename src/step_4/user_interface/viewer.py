from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)
import vedo
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class EmbeddedViewer(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.plotter = vedo.Plotter(qt_widget=self.vtk_widget, bg="#FFFFFF")
        self.vtk_widget.Initialize()

        # World axes
        line_size = 1000
        self.world_axes = [
            vedo.Line((-line_size, 0, 0), (line_size, 0, 0), c="red", lw=2),
            vedo.Line((0, -line_size, 0), (0, line_size, 0), c="green", lw=2),
            vedo.Line((0, 0, -line_size), (0, 0, line_size), c="blue", lw=2),
        ]
        for axis in self.world_axes:
            axis.use_bounds(False)
            axis.pickable(False)

        # World origin marker
        self.world_origin = vedo.Points([(0, 0, 0)], r=12, c="red").clone2d(size=1)
        position = self.world_origin.GetPositionCoordinate()
        position.SetCoordinateSystemToWorld()
        position.SetValue(0, 0, 0)
        self.world_origin.ontop()

        self.loaded_model: str | None = None

        # Custom scroll zoom
        self.plotter.interactor.AddObserver("MouseWheelForwardEvent", self.on_scroll)
        self.plotter.interactor.AddObserver("MouseWheelBackwardEvent", self.on_scroll)

        self.plotter.show(
            *self.world_axes,
            self.world_origin,
            mode="Terrain",
            interactive=False,
        )
        self._configure_camera()

    def load_mesh(self, file_path: str | vedo.Mesh) -> None:
        mesh = (
            file_path.clone()
            if isinstance(file_path, vedo.Mesh)
            else vedo.Mesh(file_path)
        )
        mesh.phong()
        mesh.c("#A0A0A0")
        mesh.lighting("plastic")
        self.plotter.clear()
        self.plotter.add(
            mesh,
            *self.world_axes,
            self.world_origin,
        )
        self.plotter.reset_camera()
        # Zoom in closer to the object
        self.plotter.camera.Dolly(3)
        self._configure_camera()

    def _configure_camera(self) -> None:
        camera = self.plotter.camera
        camera.ParallelProjectionOff()
        camera.SetViewAngle(90)
        camera.SetClippingRange(0.01, 10000)
        style = self.plotter.interactor.GetInteractorStyle()
        style.AutoAdjustCameraClippingRangeOff()
        self.plotter.render()

    def on_scroll(self, interactor, event) -> None:
        factor = 1.15 if event == "MouseWheelForwardEvent" else 1 / 1.15
        camera = self.plotter.camera
        if camera.GetParallelProjection():
            camera.SetParallelScale(camera.GetParallelScale() / factor)
        else:
            camera.Dolly(factor)
        self.plotter.render()

    def cleanup(self) -> None:
        self.vtk_widget.Disable()
        self.vtk_widget.close()
