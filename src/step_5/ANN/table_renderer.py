from typing import Any
import matplotlib.pyplot as plt
import numpy
from pathlib import Path
import vedo


def render_mesh_thumbnail(
    mesh_path: str | Path | vedo.Mesh,
    image_size: tuple[int, int] = (300, 300),
    mesh_color: tuple[float, float, float] = (0.4, 0.3, 0.8),
) -> numpy.ndarray:
    if isinstance(mesh_path, vedo.Mesh):
        mesh = mesh_path.clone()
    else:
        path = Path(mesh_path)
        mesh = vedo.Mesh(str(path))
    mesh.shift(-mesh.center_of_mass())  # type: ignore
    mesh.color(mesh_color)

    plotter = vedo.Plotter(offscreen=True, size=image_size, bg="white")
    plotter.show(mesh, zoom=1.1, resetcam=True)
    image = plotter.screenshot(asarray=True)
    plotter.close()

    return image


def find_mesh_path(model_dir: str | Path, shape_id: str) -> Path:
    """Finds the actual .obj file given the relative identifier (e.g. Category/m123.obj)."""
    root = Path(model_dir)
    direct = root / shape_id
    if direct.is_file():
        return direct

    # Recursive search if stored under different subdirectories
    matched = list(root.rglob(f"*{Path(shape_id).name}"))
    if matched:
        return matched[0]

    raise FileNotFoundError(
        f"Could not find model file for '{shape_id}' in '{model_dir}'"
    )


def render_meshes_grid(
    rows_data: list[dict[str, Any]], output_path: str, cell_size: float = 2.2
) -> None:
    number_of_rows = len(rows_data)

    # figure out how many number of columns we need
    max_matches = max(len(r["matches"]) for r in rows_data)
    number_of_collumns = max(max_matches, 1) + 1

    fig, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=number_of_collumns,
        figsize=(number_of_collumns * cell_size, number_of_rows * cell_size),
        gridspec_kw={"wspace": 0.4, "hspace": 0.04},
    )

    # Ensure 2D array of axes even if single row or single column
    if number_of_rows == 1:
        axes = numpy.expand_dims(axes, axis=0)
    if number_of_collumns == 1:
        axes = numpy.expand_dims(axes, axis=1)

    for row_index, row_entry in enumerate(rows_data):
        query_mesh_file = row_entry["query_file"]
        matches = row_entry["matches"]

        query_ax = axes[row_index, 0]
        query_img = render_mesh_thumbnail(query_mesh_file)
        query_ax.imshow(query_img)
        query_ax.set_xticks([])
        query_ax.set_yticks([])

        if row_index == 0:
            query_ax.set_title(
                "Query shape", fontsize=11, fontweight="bold", pad=8
            )

        for spine in query_ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)
        query_ax.spines["right"].set_linewidth(2.8)

        for column_index in range(1, number_of_collumns):
            match_ax = axes[row_index, column_index]
            match_index = column_index - 1

            if row_index == 0:
                match_ax.set_title(
                    f"{column_index:02d}",
                    fontsize=11,
                    fontweight="bold",
                    pad=8,
                )

            if match_index < len(matches):
                match_file, dist = matches[match_index]
                match_img = render_mesh_thumbnail(match_file)

                match_ax.imshow(match_img)
                match_ax.set_xticks([])
                match_ax.set_yticks([])
                match_ax.set_xlabel(
                    f"dis={dist:.6f}", fontsize=8.5, labelpad=4
                )

                for spine in match_ax.spines.values():
                    spine.set_color("black")
                    spine.set_linewidth(1.0)
                if column_index == 1:
                    match_ax.spines["left"].set_linewidth(2.8)
            else:  # fewer matches than this row
                match_ax.set_xticks([])
                match_ax.set_yticks([])
                for spine in match_ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.8)
                if column_index == 1:
                    match_ax.spines["left"].set_linewidth(2.8)
                    match_ax.spines["left"].set_color("black")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\nSaved retrieval table to: {out_file.resolve()}")
