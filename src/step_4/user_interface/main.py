from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
import ctypes
from pathlib import Path
import sys
from src.step_4.user_interface.viewer import EmbeddedViewer
from src.step_4.user_interface.class_colors import get_class_color, get_shape_class
from src.helper.database_functions import retrieve_all_shape_files
import random
import os

if sys.platform == "win32":
    ctypes.OleDLL("shell32").SetCurrentProcessExplicitAppUserModelID(
        ctypes.c_wchar_p("MultimediaRetrieval.CBSR.Viewer")
    )
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def query_random_shapes(
    query_shape_path: str,
    database_directory: str,
    k: int,
) -> list[tuple[str, float]]:
    all_shapes = retrieve_all_shape_files(database_directory)
    results: list[tuple[str, float]] = []

    for shape_path in all_shapes:
        if shape_path == query_shape_path:
            continue

        distance = random.random()
        results.append((str(shape_path), distance))

    results.sort(key=lambda item: item[1])
    return results[:k]


def style_class_pill(pill: QLabel, shape_path: str) -> None:
    shape_class = get_shape_class(shape_path)
    pill_color = get_class_color(shape_class)
    pill.setText(shape_class)
    pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
    pill.setStyleSheet(f"""
        QLabel {{
            background-color: {pill_color};
            color: #000000;
            border-radius: 10px;
            padding: 2px 10px;
            font-weight: bold;
        }}
        """)


class MainWindow(QMainWindow):
    def __init__(
        self,
    ):
        super().__init__()

        app_icon_path = Path(__file__).resolve().parents[3] / "assets" / "icon.png"
        app_icon = QIcon(str(app_icon_path))
        self.setWindowIcon(app_icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(app_icon)  # type: ignore

        self.setWindowTitle("Multimedia Retrieval CBSR")
        self.resize(1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.selected_file: str | None = None
        self.result_viewers: list[EmbeddedViewer] = []

        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.selected_class_pill = QLabel()
        self.selected_class_pill.hide()
        left_layout.addWidget(
            self.selected_class_pill, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.file_label)

        button_holder = QHBoxLayout()

        self.pick_button = QPushButton("Pick 3D Model")
        self.pick_button.setFixedWidth(200)
        self.pick_button.clicked.connect(self.open_file_picker)
        button_holder.addWidget(
            self.pick_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.query_button = QPushButton("Query")
        self.query_button.setFixedWidth(200)
        self.query_button.clicked.connect(self.run_query)
        self.query_button.setEnabled(False)
        button_holder.addWidget(
            self.query_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        left_layout.addLayout(button_holder)

        self.left_viewer = EmbeddedViewer(left_panel)
        left_layout.addWidget(self.left_viewer, stretch=1)

        self.results_list = QListWidget()
        self.results_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.results_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.results_list.setMovement(QListWidget.Movement.Static)
        self.results_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.results_list.setSpacing(10)
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(self.results_list, stretch=2)

    def open_file_picker(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, filter="(*.obj)")
        if file_path:
            self.selected_file = file_path
            style_class_pill(self.selected_class_pill, file_path)
            self.selected_class_pill.show()
            self.file_label.setText(file_path)
            self.left_viewer.load_mesh(file_path)
            self.query_button.setEnabled(True)

    def run_query(self) -> None:
        if not self.selected_file:
            return

        k: int = 6
        matches = query_random_shapes(
            query_shape_path=self.selected_file,
            database_directory="3d_models_full_normalized",
            k=k,
        )

        for viewer in self.result_viewers:
            viewer.cleanup()
        self.result_viewers.clear()
        self.results_list.clear()
        viewer_width = 300
        viewer_height = 300
        for shape_path, distance in matches:
            card = QWidget()
            card.setFixedWidth(viewer_width)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(0, 8, 0, 0)

            class_pill = QLabel()
            style_class_pill(class_pill, shape_path)
            card_layout.addWidget(class_pill, alignment=Qt.AlignmentFlag.AlignCenter)

            label = QLabel(f"{os.path.basename(shape_path)}\ndist: {distance:.3f}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(label)
            viewer = EmbeddedViewer(card)
            viewer.setFixedSize(viewer_width, viewer_height)
            viewer.load_mesh(shape_path)
            card_layout.addWidget(viewer, alignment=Qt.AlignmentFlag.AlignCenter)
            card_layout.activate()
            item = QListWidgetItem(self.results_list)
            item.setSizeHint(card.sizeHint())
            self.results_list.addItem(item)
            self.results_list.setItemWidget(item, card)
            self.result_viewers.append(viewer)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
