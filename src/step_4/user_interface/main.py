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
    QSpinBox,
    QProgressBar,
    QCheckBox,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon
import ctypes
from pathlib import Path
import sys
from src.step_4.user_interface.viewer import EmbeddedViewer
from src.step_4.user_interface.class_colors import (
    SHAPE_CLASS_COLORS,
    get_class_color,
    get_shape_class,
)
import os
from src.step_4.distance_calculations.calculate_total_distance import (
    ShapeDistanceCalculator,
    DISTANCE_CONFIG_L2,
    DISTANCE_CONFIG_EMD,
)
from src.step_5.ANN.table_renderer import find_mesh_path

if sys.platform == "win32":
    ctypes.OleDLL("shell32").SetCurrentProcessExplicitAppUserModelID(
        ctypes.c_wchar_p("MultimediaRetrieval.CBSR.Viewer")
    )
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


from dotenv import load_dotenv

load_dotenv()


def style_class_pill(
    pill: QLabel,
    shape_path: str,
    class_override: str | None = None,
) -> None:
    shape_class = (
        class_override
        if class_override is not None
        else get_shape_class(shape_path)
    )
    if class_override is None and shape_class not in SHAPE_CLASS_COLORS:
        shape_class = "Outside Database"
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


class QueryWorker(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        calculator: ShapeDistanceCalculator,
        selected_file: str,
        database_directory: str,
    ) -> None:
        super().__init__()
        self.calculator = calculator
        self.selected_file = selected_file
        self.database_directory = database_directory

    def run(self) -> None:
        try:
            results = self.calculator.calculate_total_distance(
                self.selected_file,
                shapes_to_compare=[str(self.database_directory)],
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        app_icon_path = (
            Path(__file__).resolve().parents[3] / "assets" / "icon.png"
        )
        app_icon = QIcon(str(app_icon_path))
        self.setWindowIcon(app_icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(app_icon)  # type: ignore

        self.setWindowTitle("Multimedia Retrieval CBSR")
        self.resize(1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        default_db = Path(
            "3d_models_final/3d_models_steps/3_models_fully_normalized"
        )
        if not default_db.is_dir():
            default_db = Path("3d_models")

        self.selected_file: str | None = None
        self.database_directory: str | None = (
            str(default_db.resolve()) if default_db.is_dir() else None
        )
        self.result_viewers: list[EmbeddedViewer] = []
        self.query_worker: QueryWorker | None = None

        # Get csv's from .env
        simple_descriptors_csv = os.getenv("SIMPLE_DESCRIPTORS_CSV")
        histogram_descriptors_csv = os.getenv("HISTOGRAM_DESCRIPTORS_CSV")
        weights_emd_csv = os.getenv("WEIGHTS_EMD_CSV")

        if (
            simple_descriptors_csv is None
            or histogram_descriptors_csv is None
            or weights_emd_csv is None
        ):
            raise ValueError(
                "Missing CSV file paths in .env. "
                "Check SIMPLE_DESCRIPTORS_CSV, "
                "HISTOGRAM_DESCRIPTORS_CSV, and WEIGHTS_EMD_CSV."
            )

        csv_databases = [
            simple_descriptors_csv,
            histogram_descriptors_csv,
            weights_emd_csv,
        ]

        self.shape_distance_calculator = ShapeDistanceCalculator(
            csv_databases=csv_databases,
            distance_config=DISTANCE_CONFIG_EMD,
        )

        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Database folder name label
        self.database_label = QLabel(f"Database: {self.database_directory}")
        self.database_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.database_label.setWordWrap(True)
        left_layout.addWidget(self.database_label)

        # Pick database folder button
        self.database_button = QPushButton("Select Database Folder")
        self.database_button.setFixedWidth(200)
        self.database_button.clicked.connect(self.open_database_picker)
        left_layout.addWidget(
            self.database_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.selected_class_pill = QLabel()
        self.selected_class_pill.hide()
        left_layout.addWidget(
            self.selected_class_pill, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.file_label)

        # Top-k results selector and exclude-self checkbox
        k_holder = QHBoxLayout()
        k_holder.addStretch()
        self.k_label = QLabel("Top k results:")
        k_holder.addWidget(self.k_label)
        self.k_spinbox = QSpinBox()
        max_k = max(1, len(self.shape_distance_calculator.shape_identifiers))
        self.k_spinbox.setRange(1, max_k)
        self.k_spinbox.setValue(min(10, max_k))
        self.k_spinbox.setFixedWidth(80)
        k_holder.addWidget(self.k_spinbox)
        self.exclude_self_checkbox = QCheckBox("Exclude self")
        k_holder.addWidget(self.exclude_self_checkbox)
        k_holder.addStretch()
        left_layout.addLayout(k_holder)

        button_holder = QHBoxLayout()

        # Pick 3D model button
        self.pick_button = QPushButton("Pick 3D Model")
        self.pick_button.setFixedWidth(200)
        self.pick_button.clicked.connect(self.open_file_picker)
        button_holder.addWidget(
            self.pick_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        # Query button
        self.query_button = QPushButton("Query")
        self.query_button.setFixedWidth(200)
        self.query_button.clicked.connect(self.run_query)
        self.query_button.setEnabled(False)
        button_holder.addWidget(
            self.query_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        left_layout.addLayout(button_holder)

        # Loading indicator shown while querying
        self.loading_widget = QWidget()
        loading_layout = QVBoxLayout(self.loading_widget)
        loading_layout.setContentsMargins(0, 4, 0, 4)
        self.loading_label = QLabel("Querying...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 0)
        self.loading_progress.setTextVisible(True)
        loading_layout.addWidget(self.loading_label)
        loading_layout.addWidget(self.loading_progress)
        self.loading_widget.hide()
        left_layout.addWidget(self.loading_widget)

        self.left_viewer = EmbeddedViewer(left_panel)
        left_layout.addWidget(self.left_viewer, stretch=1)

        self.results_list = QListWidget()
        self.results_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.results_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.results_list.setMovement(QListWidget.Movement.Static)
        self.results_list.setSelectionMode(
            QListWidget.SelectionMode.NoSelection
        )
        self.results_list.setSpacing(10)
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(self.results_list, stretch=2)

    def open_file_picker(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, filter="(*.obj)")
        if file_path:
            self.selected_file = file_path
            class_override: str | None = None
            if not self.shape_distance_calculator.is_in_database(file_path):
                candidate_class = get_shape_class(file_path)
                if candidate_class not in SHAPE_CLASS_COLORS:
                    class_override = "Outside Database"
            style_class_pill(
                self.selected_class_pill,
                file_path,
                class_override=class_override,
            )
            self.selected_class_pill.show()
            self.file_label.setText(file_path)
            self.left_viewer.load_mesh(file_path)
            self.query_button.setEnabled(True)

    def open_database_picker(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select 3D Database Directory"
        )
        if folder:
            self.database_directory = folder
            self.database_label.setText(f"Database: {folder}")

    def _set_loading_state(
        self, loading: bool, message: str = "Querying..."
    ) -> None:
        self.query_button.setEnabled(
            not loading and self.selected_file is not None
        )
        self.pick_button.setEnabled(not loading)
        self.database_button.setEnabled(not loading)
        self.k_spinbox.setEnabled(not loading)
        self.exclude_self_checkbox.setEnabled(not loading)
        self.query_button.setText("Querying..." if loading else "Query")
        if loading:
            self.loading_label.setText(message)
            self.loading_progress.setRange(0, 0)
            self.loading_widget.show()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            self.loading_widget.hide()
            QApplication.restoreOverrideCursor()

    def run_query(self) -> None:
        if not self.selected_file or not self.database_directory:
            return
        if self.query_worker is not None and self.query_worker.isRunning():
            return

        for viewer in self.result_viewers:
            viewer.cleanup()
        self.result_viewers.clear()
        self.results_list.clear()

        self._set_loading_state(True, "Calculating distances...")
        QApplication.processEvents()

        self.query_worker = QueryWorker(
            self.shape_distance_calculator,
            self.selected_file,
            self.database_directory,
        )
        self.query_worker.finished.connect(self._on_query_finished)
        self.query_worker.error.connect(self._on_query_error)
        self.query_worker.start()

    def _on_query_error(self, error_msg: str) -> None:
        self._set_loading_state(False)
        self.file_label.setText(f"Query error: {error_msg}")

    def _on_query_finished(self, results: dict[str, float]) -> None:
        if not self.database_directory:
            self._set_loading_state(False)
            return

        try:
            if (
                self.shape_distance_calculator.last_preprocessed_mesh
                is not None
            ):
                self.left_viewer.load_mesh(
                    self.shape_distance_calculator.last_preprocessed_mesh
                )

            k = self.k_spinbox.value()
            sorted_items = sorted(results.items(), key=lambda item: item[1])
            if self.exclude_self_checkbox.isChecked() and self.selected_file:
                query_key: str | None = None
                if self.shape_distance_calculator.is_in_database(
                    self.selected_file
                ):
                    query_key = (
                        self.shape_distance_calculator._resolve_query_key(
                            self.selected_file
                        )
                    )
                selected_name = Path(self.selected_file).name
                sorted_items = [
                    (shape_path, dist)
                    for shape_path, dist in sorted_items
                    if shape_path != query_key
                    and Path(shape_path).name != selected_name
                ]
            matches = sorted_items[:k]

            self.loading_progress.setRange(0, max(1, len(matches)))
            self.loading_progress.setValue(0)

            viewer_width = 300
            viewer_height = 300
            for idx, (shape_path, distance) in enumerate(matches, start=1):
                self.loading_label.setText(
                    f"Loading result {idx}/{len(matches)}..."
                )
                self.loading_progress.setValue(idx)
                QApplication.processEvents()

                direct_path = Path(self.database_directory) / shape_path
                if direct_path.is_file():
                    full_shape_path = str(direct_path)
                else:
                    full_shape_path = str(
                        find_mesh_path(self.database_directory, shape_path)
                    )
                card = QWidget()
                card.setFixedWidth(viewer_width)
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(0, 8, 0, 0)

                class_pill = QLabel()
                style_class_pill(class_pill, full_shape_path)
                card_layout.addWidget(
                    class_pill, alignment=Qt.AlignmentFlag.AlignCenter
                )

                label = QLabel(
                    f"{os.path.basename(shape_path)}\ndist: {distance:.3f}"
                )
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                card_layout.addWidget(label)
                viewer = EmbeddedViewer(card)
                viewer.setFixedSize(viewer_width, viewer_height)
                viewer.load_mesh(full_shape_path)
                card_layout.addWidget(
                    viewer, alignment=Qt.AlignmentFlag.AlignCenter
                )
                card_layout.activate()
                item = QListWidgetItem(self.results_list)
                item.setSizeHint(card.sizeHint())
                self.results_list.addItem(item)
                self.results_list.setItemWidget(item, card)
                self.result_viewers.append(viewer)
        finally:
            self._set_loading_state(False)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
