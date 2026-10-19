from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
)
from PySide6.QtCore import Qt
import sys
import vedo
from src.step_4.user_interface.viewer import EmbeddedViewer


class MainWindow(QMainWindow):
    def __init__(
        self,
    ):
        super().__init__()

        self.setWindowTitle("Multimedia Retrieval CBSR")
        self.resize(1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

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
        self.query_button.clicked.connect(self.open_file_picker)
        self.query_button.setEnabled(False)
        button_holder.addWidget(
            self.query_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        left_layout.addLayout(button_holder)

        self.left_viewer = EmbeddedViewer(left_panel)
        left_layout.addWidget(self.left_viewer, stretch=1)

        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_placeholder = QLabel("Matched models will appear here.")
        right_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(right_placeholder)

        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(self.right_panel, stretch=2)

    def open_file_picker(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, filter="(*.obj)")
        if file_path:
            self.file_label.setText(file_path)
            mesh = vedo.Mesh(file_path).c("lightgray").lighting("default")
            self.left_viewer.load_mesh(mesh)
            self.query_button.setEnabled(True)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
