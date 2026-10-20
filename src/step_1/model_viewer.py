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
    QGroupBox,
    QSlider,
)
from PySide6.QtGui import QIcon
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Mesh, Points, Line
import ctypes
import sys
import os
from pathlib import Path
import numpy
from PySide6.QtCore import Qt

# Configure platform taskbar/window icon support before QApplication is initialized.
if sys.platform == "win32":
    ctypes.OleDLL("shell32").SetCurrentProcessExplicitAppUserModelID(
        ctypes.c_wchar_p("MultimediaRetrieval.CBSR.Viewer")
    )
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


class ModelViewer(QMainWindow):
    def __init__(
        self,
        qt_app: QApplication,
        window_title: str = "Multimedia Retrieval CBSR",
        window_size: tuple[int, int] = (1000, 1000),
        background_color: str = "#FFFFFF",
        meshes: list[Mesh] = []],
        models_delta: numpy.array = numpy.zeros(3), # type: ignore
    ):

        super().__init__()

        # Make the Application
        self.app = qt_app

        # Add an application icon (assets/icon.png is at the repository root)
        app_icon_path = Path(__file__).resolve().parents[2] / "assets" / "icon.png"
        app_icon = QIcon(str(app_icon_path))
        self.app.setWindowIcon(app_icon)
        self.setWindowIcon(app_icon)

        self.setWindowTitle(window_title)
        self.resize(*window_size)

        # pass meshes and points and store them
        self.meshes = meshes
        self.points = []

        shift_amount: numpy.array = numpy.zeros(3) # type: ignore
        for mesh in self.meshes:
            # add shading and lighting to meshes
            mesh.compute_normals(points=True, cells=True)
            mesh.phong()
            mesh.c("#A0A0A0")
            mesh.lighting("plastic")

            # make edges of mesh stand out
            mesh.linewidth(3)
            mesh.linecolor("#000000")

            mesh.shift(*shift_amount)
            shift_amount = numpy.add(shift_amount, models_delta)

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

        # World group. Options for the world
        world_group = QGroupBox("World")
        world_layout = QVBoxLayout(world_group)

        # checkbox for global origin
        self.global_origin_checkbox = QCheckBox("Show Global Origin")
        self.global_origin_checkbox.setChecked(True)
        world_layout.addWidget(self.global_origin_checkbox)

        self.world_axes_checkbox = QCheckBox("Show World Axes")
        self.world_axes_checkbox.setChecked(True)
        world_layout.addWidget(self.world_axes_checkbox)

        controls.addWidget(world_group)

        # Camera Group. Options for camera
        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout(camera_group)

        # input selector for camera mode
        self.camera_type_dropdown = QComboBox()
        self.camera_type_dropdown.addItems(
            [
                "TrackballCamera",
                "TrackballActor",
                "JoystickCamera",
                "JoystickActor",
                "Flight",
                "RubberBand2D",
                "RubberBand3D",
                "RubberBandZoom",
                "Terrain",
                "Unicam",
                "Image",
            ]
        )
        self.camera_type_dropdown.setCurrentText("Terrain")
        self.camera_type_dropdown.currentTextChanged.connect(
            self.on_camera_type_changed
        )
        camera_layout.addWidget(self.camera_type_dropdown)

        controls.addWidget(camera_group)

        # Edges group. Options for edges visualization
        edges_group = QGroupBox("Edges")
        edges_layout = QVBoxLayout(edges_group)

        # slider to change line-width of edges
        EDGE_STARTING_VALUE = 3
        self.edge_width_label = QLabel(f"Width = {EDGE_STARTING_VALUE}")
        edges_layout.addWidget(self.edge_width_label)

        self.edge_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_width_slider.setRange(0, 15)
        self.edge_width_slider.setValue(3)
        edges_layout.addWidget(self.edge_width_slider)

        controls.addWidget(edges_group)

        # Vertices group. Options for vertices visualization
        vertices_group = QGroupBox("Vertices")
        vertices_layout = QVBoxLayout(vertices_group)

        VERTEX_STARTING_VALUE = 10
        self.vertex_radius_label = QLabel(f"Point size = {VERTEX_STARTING_VALUE}")
        vertices_layout.addWidget(self.vertex_radius_label)

        self.vertex_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.vertex_radius_slider.setRange(0, 20)
        self.vertex_radius_slider.setValue(VERTEX_STARTING_VALUE)
        vertices_layout.addWidget(self.vertex_radius_slider)

        controls.addWidget(vertices_group)

        shading_group = QGroupBox("Shading")
        shading_layout = QVBoxLayout(shading_group)

        # input dropdown for selecting shading
        self.shading_dropdown = QComboBox()
        self.shading_dropdown.addItems(
            [
                "Smooth Shading",
                "Flat Shading",
            ]
        )
        self.shading_dropdown.currentTextChanged.connect(self.on_shading_option_changed)
        shading_layout.addWidget(self.shading_dropdown)

        controls.addWidget(shading_group)

        # Add empty space to keep all controls at the top
        controls.addStretch()

        layout.addWidget(side_bar)

        # VTK is the library for rendering 3d models
        # Add VTK inside the Qt layout
        self.vtk_widget = QVTKRenderWindowInteractor(container)
        layout.addWidget(self.vtk_widget, 1)

        self.viewer = Plotter(qt_widget=self.vtk_widget, bg=background_color) # type: ignore

        self.viewer.interactor.AddObserver("MouseWheelForwardEvent", self.on_scroll)
        self.viewer.interactor.AddObserver("MouseWheelBackwardEvent", self.on_scroll)

        self.global_origin_checkbox.toggled.connect(self.set_global_origin_visible)
        self.world_axes_checkbox.toggled.connect(self.set_world_axes_visible)
        self.edge_width_slider.valueChanged.connect(self.on_edge_width_changed)
        self.vertex_radius_slider.valueChanged.connect(self.on_vertex_radius_changed)

    def on_vertex_radius_changed(self, radius: int) -> None:
        self.vertex_radius_label.setText(f"Size = {radius}")
        visible = radius > 0
        for points in self.points:
            points.actor.SetVisibility(visible)
            if radius > 0:
                points.properties.SetPointSize(radius)

        self.viewer.render()

    def on_edge_width_changed(self, line_width: int) -> None:
        for mesh in self.meshes:
            self.edge_width_label.setText(f"Width = {line_width}")
            if line_width == 0:
                mesh.properties.SetEdgeVisibility(False)
            else:
                mesh.properties.SetEdgeVisibility(True)
                mesh.properties.SetLineWidth(line_width)

        self.viewer.render()

    def on_scroll(self, interactor, event) -> None:
        factor = 1.15 if event == "MouseWheelForwardEvent" else 1 / 1.15
        camera = self.viewer.camera

        if camera.GetParallelProjection():
            camera.SetParallelScale(camera.GetParallelScale() / factor)
        else:
            camera.Dolly(factor)

        self.viewer.render()

    def set_global_origin_visible(self, visible: bool) -> None:
        self.world_origin.SetVisibility(visible)
        self.viewer.render()

    def set_world_axes_visible(self, visible: bool) -> None:
        for axis in self.world_axes:
            axis.actor.SetVisibility(visible)
        self.viewer.render()

    def closeEvent(self, event) -> None:

        # clean up vtk widget
        self.vtk_widget.Disable()
        self.vtk_widget.close()

        super().closeEvent(event)

    def on_shading_option_changed(self, input_selection_text: str) -> None:
        for mesh in self.meshes:
            if input_selection_text == "Flat Shading":
                mesh.flat()
            elif input_selection_text == "Smooth Shading":
                mesh.phong()
        self.viewer.render()

    def on_camera_type_changed(self, camera_type_selection_text: str) -> None:
        self.viewer.user_mode(camera_type_selection_text)
        style = self.viewer.interactor.GetInteractorStyle()
        style.AutoAdjustCameraClippingRangeOff()
        self.viewer.render()

    def exec(self):
        self.show()
        self.vtk_widget.Initialize()

        self.viewer.show(
            *self.meshes,
            *self.points,
            *self.world_axes,
            self.world_origin,
            mode=self.camera_type_dropdown.currentText(),
            interactive=False,
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
