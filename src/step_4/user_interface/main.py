from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QGridLayout,
)
from PySide6.QtCore import Qt
import sys
from src.step_4.user_interface.viewer import EmbeddedViewer
from src.helper.database_functions import retrieve_all_shape_files
import random
import os


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


class MainWindow(QMainWindow):
    def __init__(
        self,
    ):
        super().__init__()

        self.setWindowTitle("Multimedia Retrieval CBSR")
        self.resize(1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.selected_file: str | None = None
        self.result_viewers: list[EmbeddedViewer] = []

        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

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

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.right_panel = QWidget()
        self.right_layout = QGridLayout(self.right_panel)
        self.right_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

        right_placeholder = QLabel("Matched models will appear here")
        right_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(right_placeholder, 0, 0)

        self.scroll_area.setWidget(self.right_panel)

        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(self.scroll_area, stretch=2)

    def open_file_picker(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, filter="(*.obj)")
        if file_path:
            self.selected_file = file_path
            self.file_label.setText(file_path)
            self.left_viewer.load_mesh(file_path)
            self.query_button.setEnabled(True)

    def run_query(self) -> None:
        if not self.selected_file:
            return

        k: int = 5
        matches = query_random_shapes(
            query_shape_path=self.selected_file,
            database_directory="3d_models_full_normalized",
            k=k,
        )

        for viewer in self.result_viewers:
            viewer.cleanup()
        self.result_viewers.clear()
        while self.right_layout.count():
            item = self.right_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        columns: int = 4
        viewer_width = 300
        viewer_height = 300

        for index, (shape_path, distance) in enumerate(matches):

            row, col = divmod(index, columns)

            card = QWidget(self.right_panel)
            card.setFixedWidth(viewer_width)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"{os.path.basename(shape_path)}\ndist: {distance:.3f}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(label)

            viewer = EmbeddedViewer(card)
            viewer.setFixedSize(viewer_width, viewer_height)
            viewer.load_mesh(shape_path)
            card_layout.addWidget(viewer, alignment=Qt.AlignmentFlag.AlignCenter)

            self.right_layout.addWidget(
                card, row, col, alignment=Qt.AlignmentFlag.AlignCenter
            )
            self.result_viewers.append(viewer)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
