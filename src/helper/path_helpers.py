from pathlib import Path
from collections.abc import Sequence


def normalize_path_separators(shape: str | Path) -> str:
    """Normalize Windows path separators to POSIX separators."""
    return str(shape).replace("\\", "/")


def expand_mesh_paths(paths: Sequence[str | Path]) -> list[str]:
    """Expand OBJ files and directories, preserving order and removing duplicates."""
    mesh_paths: list[str] = []
    seen_paths: set[Path] = set()

    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"Mesh path does not exist: '{path}'")

        if path.is_file():
            if path.suffix.lower() != ".obj":
                raise ValueError(f"Expected an OBJ mesh file, got: '{path}'")
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() == ".obj"
            )
        else:
            raise ValueError(f"Not a file or directory: '{path}'")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                mesh_paths.append(candidate.as_posix())

    if not mesh_paths:
        raise ValueError("No OBJ meshes were found in the supplied paths.")
    return mesh_paths


def normalize_shape_path(shape: str | Path) -> str:
    posix_path = str(shape).replace("\\", "/").strip("/")
    parts = [part for part in posix_path.split("/") if part and part != "."]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return posix_path
