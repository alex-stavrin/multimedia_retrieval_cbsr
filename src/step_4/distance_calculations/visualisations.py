from pathlib import Path
import vedo
import numpy as np


def visualize_mesh_distances(
    query: str,
    shapes_to_compare: list[str],
    descriptor: str | None = None,
    distance_measure: str | None = None,
    distances: dict[str, float] | None = None,
    config: str | None = None,
) -> None:
    """Display meshes side by side, coloured by distance to the query."""
    distances = distances or {}

    # Match command-line mesh paths to the paths used as distance keys.
    def get_distance_key(shape: str) -> str | None:
        normalized_shape = shape.replace("\\", "/")

        for key in distances:
            normalized_key = key.replace("\\", "/")

            if normalized_shape == normalized_key:
                return key

            if normalized_shape.endswith("/" + normalized_key):
                return key

            if normalized_key.endswith("/" + normalized_shape):
                return key

        return None

    # Pair each mesh path with its calculated distance.
    comparison_meshes = []

    for shape in shapes_to_compare:
        distance_key = get_distance_key(shape)

        if distance_key is None:
            distance = None
        else:
            distance = distances[distance_key]

        comparison_meshes.append((shape, distance))

    # Sort available distances first; missing distances go last.
    comparison_meshes.sort(
        key=lambda item: (
            item[1] is None,
            item[1] if item[1] is not None else float("inf"),
        )
    )

    mesh_paths = [query]

    for shape, distance in comparison_meshes:
        mesh_paths.append(shape)

    available_distances = []

    for shape, distance in comparison_meshes:
        if distance is not None:
            available_distances.append(distance)

    min_distance = min(available_distances, default=0.0)
    max_distance = max(available_distances, default=1.0)

    meshes = []
    labels = []
    spacing = 0.0
    width = 0.0

    for index, mesh_path in enumerate(mesh_paths):
        path = Path(mesh_path)

        if not path.is_file():
            raise ValueError(f"Mesh file does not exist: {mesh_path}")

        mesh = vedo.Mesh(str(path))

        # Centre the mesh around its centre of mass.
        center = mesh.center_of_mass()
        mesh.shift(-center[0], -center[1], -center[2])

        bounds = mesh.bounds()
        width = bounds[1] - bounds[0]
        height = bounds[5] - bounds[4]

        # Position meshes side by side.
        mesh.shift(spacing - bounds[0], 0, 0)

        if index == 0:
            mesh.color("dodgerblue")
            label_text = "QUERY"
        else:
            shape, distance = comparison_meshes[index - 1]

            if distance is None:
                mesh.color("lightgray")
                label_text = f"MESH {index}\nDistance: N/A"
            else:
                if max_distance == min_distance:
                    normalized = 0.0
                else:
                    normalized = (distance - min_distance) / (
                        max_distance - min_distance
                    )

                # Green means closer; red means farther.
                red = normalized
                green = 1.0 - normalized

                mesh.color((red, green, 0.0))
                mesh_label = mesh_path.split("/")[-2:]
                label_text = (
                    f"CLASS: {mesh_label[0]}\nOBJECT: {mesh_label[1]}\n"
                    f"Distance: {distance:.6f}"
                )

        meshes.append(mesh)

        labels.append(
            vedo.Text3D(
                label_text,
                pos=(
                    spacing + width / 2,
                    0,
                    height / 2 + 0.2,
                ),
                s=max(width * 0.08, 0.05),
            )
        )

        spacing += width + 1.0

    plotter = vedo.Plotter(
        title="3D Mesh Similarity Comparison",
        axes=0,
        bg="white",
    )

    bounds = np.array([mesh.bounds() for mesh in meshes])

    x_center = (bounds[:, 0].min() + bounds[:, 1].max()) / 2
    y_center = (bounds[:, 2].min() + bounds[:, 3].max()) / 2
    z_center = (bounds[:, 4].min() + bounds[:, 5].max()) / 2
    y_top = bounds[:, 3].max()

    title_text = f"Config: {config}" if config else None
    if title_text is None:
        title_text = f"Descriptor: {descriptor or 'unknown'}"

    title = vedo.Text3D(
        title_text,
        pos=(x_center, y_top + 0.6, z_center),
        s=0.3,
        c="black",
        justify="center",
    )

    subtitle = None

    if distance_measure:
        subtitle = vedo.Text3D(
            f"Measure: {distance_measure}",
            pos=(x_center, y_top + 0.3, z_center),
            s=0.15,
            c="black",
            justify="center",
        )

    plotter.show(
        *meshes,
        *labels,
        title,
        subtitle,
        viewup="z",
        camera={
            "pos": (x_center, y_center, 10),
            "focal_point": (x_center, y_center, 0),
            "viewup": (0, 1, 0),
        },
        interactive=True,
    )
