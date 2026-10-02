from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QCheckBox,
    QComboBox,
)
from PySide6.QtGui import QIcon
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Mesh, Points, Line
import ctypes
import sys
import os
from pathlib import Path


class ModelViewer(QMainWindow):
    def __init__(
        self,
        qt_app: QApplication,
        window_title: str,
        window_size: tuple[int, int],
        background_color: str,
        meshes: list[Mesh],
    ):

        super().__init__()

        # Give the viewer its own Windows taskbar identity.
        if sys.platform == "win32":
            ctypes.OleDLL("shell32").SetCurrentProcessExplicitAppUserModelID(
                ctypes.c_wchar_p("MultimediaRetrieval.CBSR.Viewer")
            )
        else:
            os.environ["QT_QPA_PLATFORM"] = "xcb"

        # Make the Application
        self.app = qt_app

        # Add an application icon
        app_icon_path = Path(__file__).resolve().parent / "assets/icon.png"
        self.app.setWindowIcon(QIcon(str(app_icon_path)))

        self.setWindowTitle(window_title)
        self.resize(*window_size)

        # pass meshes and points and store them
        self.meshes = meshes
        self.points = []

        for mesh in self.meshes:
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
            self.points.append(points)

        self.world_origin = Points([(0, 0, 0)], r=12, c="red").clone2d(size=1)
        position = self.world_origin.GetPositionCoordinate()

        # Set point to world origin
        position.SetCoordinateSystemToWorld()
        position.SetValue(0, 0, 0)

        self.world_origin.ontop()

        line_size = 1000
        self.world_axes = [
            Line((-line_size, 0, 0), (line_size, 0, 0), c="red", lw=2),
            Line((0, -line_size, 0), (0, line_size, 0), c="green", lw=2),
            Line((0, 0, -line_size), (0, 0, line_size), c="blue", lw=2),
        ]

        for axis in self.world_axes:
            axis.use_bounds(False)
            axis.pickable(False)

        # this is the main container. It will hold the viewer and
        # sidebar horizontally
        container = QWidget()
        self.setCentralWidget(container)

        # this will be the 3d model viewer
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # this will be the side bar with the controls
        side_bar = QFrame()
        side_bar.setFixedWidth(240)

        # this is the vertical box for the controls
        controls = QVBoxLayout(side_bar)
        controls.setContentsMargins(16, 16, 16, 16)
        controls.setSpacing(12)

        # this is the name of the 3d model path
        self.mesh_name_label = QLabel()
        self.mesh_name_label.setWordWrap(True)
        controls.addWidget(self.mesh_name_label)

        # checkbox to toggle edges
        self.edges_checkbox = QCheckBox("Show Edges")
        self.edges_checkbox.setChecked(True)
        controls.addWidget(self.edges_checkbox)

        # checkbox to toggle vertices
        self.vertices_checkbox = QCheckBox("Show Vertices")
        self.vertices_checkbox.setChecked(True)
        controls.addWidget(self.vertices_checkbox)

        # input dropdown for selecting shading
        controls.addWidget(QLabel("Shading"))
        self.shading_dropdown = QComboBox()
        self.shading_dropdown.addItems(
            [
                "Smooth Shading",
                "Flat Shading",
            ]
        )
        self.shading_dropdown.currentTextChanged.connect(self.on_shading_option_changed)
        controls.addWidget(self.shading_dropdown)

        # Add empty space to keep all controls at the top
        controls.addStretch()

        layout.addWidget(side_bar)

        # VTK is the library for rendering 3d models
        # Add VTK inside the Qt layout
        self.vtk_widget = QVTKRenderWindowInteractor(container)
        layout.addWidget(self.vtk_widget, 1)

        self.viewer = Plotter(qt_widget=self.vtk_widget, bg=background_color)

        self.viewer.interactor.AddObserver("MouseWheelForwardEvent", self.on_scroll)
        self.viewer.interactor.AddObserver("MouseWheelBackwardEvent", self.on_scroll)

        self.edges_checkbox.toggled.connect(self.set_edges_visible)
        self.vertices_checkbox.toggled.connect(self.set_vertices_visible)

    def on_scroll(self, interactor, event) -> None:
        factor = 1.15 if event == "MouseWheelForwardEvent" else 1 / 1.15
        camera = self.viewer.camera

        if camera.GetParallelProjection():
            camera.SetParallelScale(camera.GetParallelScale() / factor)
        else:
            camera.Dolly(factor)

        self.viewer.render()

    def set_edges_visible(self, visible: bool) -> None:

        for mesh in self.meshes:
            mesh.linewidth(1 if visible else 0)

        self.viewer.render()

    def set_vertices_visible(self, visible: bool) -> None:

        for points in self.points:
            points.actor.SetVisibility(visible)

        self.viewer.render()

    def closeEvent(self, event) -> None:

        # clean up vtk widget
        self.vtk_widget.Disable()
        self.vtk_widget.close()

        super().closeEvent(event)

    def add_mesh_name_text(self, mesh_name: str) -> None:
        self.mesh_name_label.setText(mesh_name)

    def on_shading_option_changed(self, input_selection_text: str) -> None:
        for mesh in self.meshes:
            if input_selection_text == "Flat Shading":
                mesh.flat()
            elif input_selection_text == "Smooth Shading":
                mesh.phong()
        self.viewer.render()

    def exec(self):
        self.show()
        self.vtk_widget.Initialize()

        self.viewer.show(
            *self.meshes,
            *self.points,
            *self.world_axes,
            self.world_origin,
            mode=8,
            interactive=False
        )

        camera = self.viewer.camera

        camera.ParallelProjectionOff()
        # Set fov to 90
        camera.SetViewAngle(90)
        # Set near plane to 0.01 and far plane to 10.000
        camera.SetClippingRange(0.01, 10000)

        style = self.viewer.interactor.GetInteractorStyle()
        style.AutoAdjustCameraClippingRangeOff()

        self.viewer.render()

        return self.app.exec()
