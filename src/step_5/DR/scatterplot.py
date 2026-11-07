"""Scatterplot of the t-SNE points of Step 5. Hovering over a point shows the
name and class of the shape, and clicking a point highlights its whole class."""

import argparse
from pathlib import Path

import matplotlib.pyplot as pyplot
import numpy
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import to_rgba

from src.step_4.user_interface.class_colors import get_class_color
from src.step_5.DR.tsne_engine import DEFAULT_OUTPUT, load_embedding

BACKGROUND = "#111111"
# how close (in pixels) the mouse has to be to a point to select it
PICK_DISTANCE = 8
DIMMED_ALPHA = 0.08


class Scatterplot:
    def __init__(self, embedding_csv: str = DEFAULT_OUTPUT) -> None:
        embedding = load_embedding(embedding_csv)
        self.shape_paths: list[str] = embedding["file_path"].tolist()
        self.shape_classes: numpy.ndarray = embedding["shape_class"].to_numpy()
        self.points: numpy.ndarray = embedding[["x", "y"]].to_numpy()
        self.colors: numpy.ndarray = numpy.array(
            [to_rgba(get_class_color(name)) for name in self.shape_classes]
        )
        self.selected_class: str | None = None

        self.figure, self.axes = pyplot.subplots(
            figsize=(10, 8), facecolor=BACKGROUND, num="t-SNE of the shape database"
        )
        self.axes.set_facecolor(BACKGROUND)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for spine in self.axes.spines.values():
            spine.set_visible(False)

        self.scatter = self.axes.scatter(
            self.points[:, 0], self.points[:, 1], s=14, c=self.colors
        )
        self.annotation = self.axes.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            color="white",
            bbox={"boxstyle": "round", "fc": "#222222", "ec": "#888888"},
            annotation_clip=False,
        )
        self.annotation.set_visible(False)
        self.title = self.axes.set_title(self._title(), color="white")

        self.figure.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)

    def _title(self) -> str:
        if self.selected_class is None:
            return f"{len(self.points)} shapes, click a point to show its class"
        count = int((self.shape_classes == self.selected_class).sum())
        return f"{self.selected_class} ({count} shapes), click empty space to clear"

    def nearest_point(self, event: MouseEvent) -> int | None:
        """Index of the point under the mouse, or None when none is close enough."""
        if event.inaxes is not self.axes:
            return None
        # compare in pixels, so the distance is the same after zooming
        pixels = self.axes.transData.transform(self.points)
        distances = numpy.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
        index = int(distances.argmin())
        return index if distances[index] <= PICK_DISTANCE else None

    def on_hover(self, event: MouseEvent) -> None:
        index = self.nearest_point(event)
        if index is None:
            if self.annotation.get_visible():
                self.annotation.set_visible(False)
                self.figure.canvas.draw_idle()
            return
        self.annotation.xy = tuple(self.points[index])
        self.annotation.set_text(
            f"{self.shape_classes[index]}\n{self.shape_paths[index].split('/')[-1]}"
        )
        self.annotation.set_visible(True)
        self.figure.canvas.draw_idle()

    def on_click(self, event: MouseEvent) -> None:
        # a click while zooming or panning is not a selection
        toolbar = self.figure.canvas.toolbar
        if toolbar is not None and toolbar.mode != "":
            return
        index = self.nearest_point(event)
        clicked_class = None if index is None else str(self.shape_classes[index])
        # clicking the selected class again, or empty space, clears the selection
        self.selected_class = (
            None if clicked_class == self.selected_class else clicked_class
        )
        self.highlight()

    def highlight(self) -> None:
        colors = self.colors.copy()
        if self.selected_class is not None:
            colors[self.shape_classes != self.selected_class, 3] = DIMMED_ALPHA
        self.scatter.set_facecolor(colors)
        self.title.set_text(self._title())
        self.figure.canvas.draw_idle()

    def save_image(self, output_path: str | Path, dpi: int = 300) -> None:
        """Saves a static image of the scatterplot."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(out, dpi=dpi, facecolor=BACKGROUND, bbox_inches="tight")
        print(f"Saved scatterplot image to: {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="Optional path to save a PNG image of the plot.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display GUI window (useful when only saving image).",
    )
    arguments = parser.parse_args()

    plot = Scatterplot(arguments.input)
    if arguments.save:
        plot.save_image(arguments.save)
    if not arguments.no_show:
        pyplot.show()


if __name__ == "__main__":
    main()
