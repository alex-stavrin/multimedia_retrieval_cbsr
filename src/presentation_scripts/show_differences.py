"""Render two mesh files side by side for visual comparison.

Usage:
    python -m src.presentation_scripts.show_differences processed/Category/model.obj
    python -m src.presentation_scripts.show_differences processed/Category/model.obj original/Category/model.obj
"""

import argparse
from pathlib import Path

import vedo


def load_mesh(mesh_path: str | Path) -> vedo.Mesh:
    """Load and center a wireframe mesh for interactive viewing."""
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file does not exist: {path}")

    mesh = vedo.Mesh(str(path))
    mesh.shift(-mesh.center_of_mass())  # type: ignore
    mesh.wireframe()
    mesh.color("#858585")

    return mesh


def show_meshes_side_by_side(
    first_mesh: str | Path,
    second_mesh: str | Path | None = None,
    output_path: str | Path | None = None,
    database_root: str | Path = "3d_models",
) -> None:
    """Display two mesh renderings, optionally saving the comparison image."""
    first_mesh = Path(first_mesh)
    if second_mesh is None:
        second_path = find_matching_mesh(first_mesh, database_root)
    else:
        second_path = Path(second_mesh)

    paths = (first_mesh, second_path)
    if _is_under_database_root(
        second_path, database_root
    ) and not _is_under_database_root(first_mesh, database_root):
        paths = (second_path, first_mesh)
    meshes = [load_mesh(path) for path in paths]
    titles = [
        f"{_database_root_name(path, database_root)}\n{path.name}"
        for path in paths
    ]
    plotter = vedo.Plotter(shape=(1, 2), sharecam=True, bg="white")
    plotter.show(meshes[0], at=0, resetcam=True, interactive=False)
    plotter.add(
        vedo.Text2D(titles[0], pos="top-center", c="black", s=1.5), at=0
    )
    plotter.show(meshes[1], at=1, resetcam=False, interactive=False)
    plotter.add(
        vedo.Text2D(titles[1], pos="top-center", c="black", s=1.5), at=1
    )
    plotter.interactive()


def _is_under_database_root(
    mesh_path: Path, database_root: str | Path
) -> bool:
    """Whether a mesh path belongs to the named database root."""
    return _database_root_name(mesh_path, database_root) == "3d_models"


def _database_root_name(mesh_path: Path, database_root: str | Path) -> str:
    """Return the database directory name for a displayed mesh."""
    # Mesh layout is <database-root>/<category>/<mesh-file>.
    inferred_root = mesh_path.parent.parent.name
    return inferred_root or Path(database_root).name


def find_matching_mesh(
    first_mesh: str | Path, database_root: str | Path
) -> Path:
    """Find the same category and filename beneath another database root."""
    first_path = Path(first_mesh)
    root = Path(database_root)
    relative_mesh = Path(first_path.parent.name) / first_path.name
    candidate = root / relative_mesh
    if candidate.is_file():
        return candidate

    matches = (
        [
            path
            for path in root.rglob(first_path.name)
            if path.is_file() and path.parent.name == first_path.parent.name
        ]
        if root.is_dir()
        else []
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple meshes match '{relative_mesh}' under '{root}'. "
            "Pass the second mesh explicitly."
        )
    raise FileNotFoundError(
        f"Could not find '{relative_mesh}' under database root '{root}'. "
        "Pass the second mesh explicitly or set --database-root."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_mesh", type=Path, help="Path to the first mesh")
    parser.add_argument(
        "second_mesh",
        type=Path,
        nargs="?",
        help="Path to the second mesh (optional)",
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("3d_models"),
        help="Database root for automatic matching (default: 3d_models)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Save the comparison image instead of opening a window",
    )
    args = parser.parse_args()
    show_meshes_side_by_side(
        args.first_mesh,
        args.second_mesh,
        output_path=args.output,
        database_root=args.database_root,
    )


if __name__ == "__main__":
    main()
