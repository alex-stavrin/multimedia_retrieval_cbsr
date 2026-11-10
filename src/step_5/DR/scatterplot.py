"""Scatterplot of the t-SNE points of Step 5. Hovering over a point shows the
name and class of the shape, and clicking a point or legend entry highlights
its whole class."""

import argparse
from pathlib import Path

import matplotlib.pyplot as pyplot
import numpy
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from src.step_4.user_interface.class_colors import get_class_color
from src.step_5.DR.tsne_engine import DEFAULT_OUTPUT, load_embedding

BACKGROUND = "#FFFFFF"
LEGEND_BACKGROUND = "#F8F9FA"
TEXT_COLOR = "#111111"
BORDER_COLOR = "#CCCCCC"
POINT_EDGE_COLOR = "#333333"
# how close (in pixels) the mouse has to be to a point to select it
PICK_DISTANCE = 8
DIMMED_ALPHA = 0.08
VISIBLE_LEGEND_ITEMS = 25
SCROLL_STEP = 3


class Scatterplot:
    def __init__(self, embedding_csv: str = DEFAULT_OUTPUT) -> None:
        embedding = load_embedding(embedding_csv)
        self.shape_paths: list[str] = embedding["file_path"].tolist()
        self.shape_classes: numpy.ndarray = embedding["shape_class"].to_numpy()
        self.points: numpy.ndarray = embedding[["x", "y"]].to_numpy()
        self.colors: numpy.ndarray = numpy.array(
            [to_rgba(get_class_color(name)) for name in self.shape_classes]
        )
        self.edge_colors: numpy.ndarray = numpy.tile(
            to_rgba(POINT_EDGE_COLOR, alpha=0.55), (len(self.points), 1)
        )
        self.selected_class: str | None = None

        self.classes: list[str] = sorted(
            str(c) for c in numpy.unique(self.shape_classes)
        )
        self.visible_items: int = min(VISIBLE_LEGEND_ITEMS, max(1, len(self.classes)))
        self.max_scroll: float = float(max(0, len(self.classes) - self.visible_items))
        self.scroll_offset: float = 0.0
        self._dragging_scrollbar: bool = False
        self._drag_start_y: float = 0.0
        self._drag_start_offset: float = 0.0

        self.figure = pyplot.figure(
            figsize=(12, 8), facecolor=BACKGROUND, num="t-SNE of the shape database"
        )
        self.axes = self.figure.add_axes(
            (0.03, 0.05, 0.71, 0.88), facecolor=BACKGROUND
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for spine in self.axes.spines.values():
            spine.set_visible(False)

        self.scatter = self.axes.scatter(
            self.points[:, 0],
            self.points[:, 1],
            s=18,
            c=self.colors,
            edgecolors=self.edge_colors,
            linewidths=0.45,
        )
        self.annotation = self.axes.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            color=TEXT_COLOR,
            bbox={"boxstyle": "round", "fc": "#FFFFFF", "ec": "#666666"},
            annotation_clip=False,
        )
        self.annotation.set_visible(False)
        self.title = self.axes.set_title(self._title(), color=TEXT_COLOR, fontsize=11, pad=8)

        self._init_legend()

        self.figure.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.figure.canvas.mpl_connect("scroll_event", self.on_scroll)

    def _init_legend(self) -> None:
        self.legend_axes = self.figure.add_axes(
            (0.76, 0.05, 0.195, 0.88), facecolor=LEGEND_BACKGROUND
        )
        self.legend_axes.set_navigate(False)
        self.legend_axes.set_title(
            f"Classes ({len(self.classes)})",
            color=TEXT_COLOR,
            fontsize=10,
            fontweight="bold",
            pad=8,
        )
        self.legend_axes.set_xlim(0, 1)
        self.legend_axes.set_xticks([])
        self.legend_axes.set_yticks([])
        for spine in self.legend_axes.spines.values():
            spine.set_color(BORDER_COLOR)

        self.legend_colors: numpy.ndarray = numpy.array(
            [to_rgba(get_class_color(name)) for name in self.classes]
        )
        self.legend_edge_colors: numpy.ndarray = numpy.tile(
            to_rgba(POINT_EDGE_COLOR, alpha=0.7), (len(self.classes), 1)
        )

        self.legend_row_patches: list[Rectangle] = []
        self.legend_texts = []
        for idx, class_name in enumerate(self.classes):
            row_patch = Rectangle(
                (0, idx),
                1,
                1,
                facecolor="#DCE6F2",
                edgecolor="none",
                visible=False,
                clip_on=True,
                zorder=1,
            )
            self.legend_axes.add_patch(row_patch)
            self.legend_row_patches.append(row_patch)

            text = self.legend_axes.text(
                0.18,
                idx + 0.5,
                class_name,
                va="center",
                ha="left",
                fontsize=8.5,
                color=TEXT_COLOR,
                clip_on=True,
                zorder=3,
            )
            self.legend_texts.append(text)

        y_positions = numpy.arange(len(self.classes), dtype=float) + 0.5
        self.legend_scatter = self.legend_axes.scatter(
            numpy.full(len(self.classes), 0.09),
            y_positions,
            s=55,
            c=self.legend_colors,
            edgecolors=self.legend_edge_colors,
            linewidths=0.6,
            clip_on=True,
            zorder=3,
        )

        self.scrollbar_axes = self.figure.add_axes(
            (0.958, 0.05, 0.014, 0.88), facecolor="#EEEEEE"
        )
        self.scrollbar_axes.set_navigate(False)
        self.scrollbar_axes.set_xlim(0, 1)
        self.scrollbar_axes.set_ylim(1, 0)
        self.scrollbar_axes.set_xticks([])
        self.scrollbar_axes.set_yticks([])
        for spine in self.scrollbar_axes.spines.values():
            spine.set_color(BORDER_COLOR)

        self.thumb_height: float = max(
            0.06, self.visible_items / max(1, len(self.classes))
        )
        self.scrollbar_thumb = Rectangle(
            (0.12, 0.0),
            0.76,
            self.thumb_height,
            facecolor="#A8A8A8",
            edgecolor="#777777",
            linewidth=0.6,
        )
        self.scrollbar_axes.add_patch(self.scrollbar_thumb)
        self._update_legend_scroll()

    def _update_legend_scroll(self) -> None:
        self.scroll_offset = float(
            numpy.clip(self.scroll_offset, 0.0, self.max_scroll)
        )
        top = self.scroll_offset
        bottom = self.scroll_offset + self.visible_items
        self.legend_axes.set_ylim(bottom, top)

        for idx, text in enumerate(self.legend_texts):
            center_y = idx + 0.5
            text.set_visible(top <= center_y <= bottom)

        if self.max_scroll > 0:
            thumb_top = (self.scroll_offset / self.max_scroll) * (
                1.0 - self.thumb_height
            )
        else:
            thumb_top = 0.0
        self.scrollbar_thumb.set_y(thumb_top)

    def _scroll_to_class(self, class_name: str) -> None:
        if class_name not in self.classes:
            return
        idx = self.classes.index(class_name)
        if idx < self.scroll_offset:
            self.scroll_offset = float(idx)
        elif idx >= self.scroll_offset + self.visible_items:
            self.scroll_offset = float(idx - self.visible_items + 1)
        self._update_legend_scroll()

    def _title(self) -> str:
        if self.selected_class is None:
            return ""
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
        if self._dragging_scrollbar:
            if event.y is not None:
                _, axes_y = self.scrollbar_axes.transAxes.inverted().transform(
                    (event.x, event.y)
                )
                # axes_y in transAxes is 0 at bottom, 1 at top; convert to top-down [0, 1]
                top_down_y = 1.0 - axes_y
                delta_y = top_down_y - self._drag_start_y
                usable_track = max(1e-6, 1.0 - self.thumb_height)
                self.scroll_offset = (
                    self._drag_start_offset + (delta_y / usable_track) * self.max_scroll
                )
                self._update_legend_scroll()
                self.figure.canvas.draw_idle()
            return

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
        if event.inaxes is self.scrollbar_axes and event.ydata is not None:
            thumb_top = self.scrollbar_thumb.get_y()
            thumb_bottom = thumb_top + self.thumb_height
            if not (thumb_top <= event.ydata <= thumb_bottom) and self.max_scroll > 0:
                usable_track = max(1e-6, 1.0 - self.thumb_height)
                target_top = numpy.clip(
                    event.ydata - self.thumb_height / 2.0, 0.0, usable_track
                )
                self.scroll_offset = (target_top / usable_track) * self.max_scroll
                self._update_legend_scroll()
                self.figure.canvas.draw_idle()
            self._dragging_scrollbar = True
            self._drag_start_y = float(event.ydata)
            self._drag_start_offset = self.scroll_offset
            return

        if event.inaxes is self.legend_axes and event.ydata is not None:
            idx = int(numpy.floor(event.ydata))
            if 0 <= idx < len(self.classes):
                clicked_class = self.classes[idx]
                self.selected_class = (
                    None if clicked_class == self.selected_class else clicked_class
                )
                self.highlight()
            return

        if event.inaxes is not self.axes:
            return

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
        if self.selected_class is not None:
            self._scroll_to_class(self.selected_class)
        self.highlight()

    def on_release(self, _event: MouseEvent) -> None:
        self._dragging_scrollbar = False

    def on_scroll(self, event: MouseEvent) -> None:
        step = -SCROLL_STEP if event.button == "up" else SCROLL_STEP
        self.scroll_offset += step
        self._update_legend_scroll()
        self.figure.canvas.draw_idle()

    def highlight(self) -> None:
        colors = self.colors.copy()
        edge_colors = self.edge_colors.copy()
        if self.selected_class is not None:
            mask = self.shape_classes != self.selected_class
            colors[mask, 3] = DIMMED_ALPHA
            edge_colors[mask, 3] = DIMMED_ALPHA
            edge_colors[~mask, 3] = 0.85
        self.scatter.set_facecolor(colors)
        self.scatter.set_edgecolor(edge_colors)

        legend_colors = self.legend_colors.copy()
        legend_edges = self.legend_edge_colors.copy()
        for idx, class_name in enumerate(self.classes):
            is_selected = self.selected_class == class_name
            self.legend_row_patches[idx].set_visible(is_selected)
            if self.selected_class is not None and not is_selected:
                legend_colors[idx, 3] = 0.25
                legend_edges[idx, 3] = 0.25
                self.legend_texts[idx].set_color("#888888")
                self.legend_texts[idx].set_fontweight("normal")
            else:
                self.legend_texts[idx].set_color(TEXT_COLOR)
                self.legend_texts[idx].set_fontweight(
                    "bold" if is_selected else "normal"
                )
        self.legend_scatter.set_facecolor(legend_colors)
        self.legend_scatter.set_edgecolor(legend_edges)

        self.title.set_text(self._title())
        self.figure.canvas.draw_idle()

    def save_image(self, output_path: str | Path, dpi: int = 300) -> None:
        """Saves a static image of the scatterplot."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        prev_visible = self.title.get_visible()
        self.title.set_visible(False)
        self.figure.savefig(out, dpi=dpi, facecolor=BACKGROUND, bbox_inches="tight")
        self.title.set_visible(prev_visible)
        print(f"Saved scatterplot image to: {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "-s",
        "--save",
        nargs="?",
        const="",
        default=None,
        help="Optional path to save a PNG image of the plot. Defaults to tsne_scatterplot.png next to input if passed without path.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display GUI window (useful when only saving image).",
    )
    arguments = parser.parse_args()

    plot = Scatterplot(arguments.input)
    if arguments.save is not None:
        save_path = (
            arguments.save
            if arguments.save != ""
            else Path(arguments.input).parent / "tsne_scatterplot.png"
        )
        plot.save_image(save_path)
    if not arguments.no_show:
        pyplot.show()


if __name__ == "__main__":
    main()
