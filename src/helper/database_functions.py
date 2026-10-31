import numpy as np
import os
from pathlib import Path


def retrieve_all_shape_files(root_dir: str, verbose=False):
    resulting_shape_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        obj_files = [f for f in filenames if f.endswith(".obj")]

        if not obj_files and verbose:
            print(f"No .obj files found in directory: {dirpath}")
            continue

        for file in obj_files:
            resulting_shape_paths.append(os.path.abspath(os.path.join(dirpath, file)))

    return np.array(resulting_shape_paths)


def map_by_relative_path(root_dir: str) -> dict:
    """Maps 'AquaticAnimal/m51.obj' -> absolute path,
    for every shape in root_dir."""
    root_abs = os.path.abspath(root_dir)
    return {
        os.path.relpath(path, root_abs): path
        for path in retrieve_all_shape_files(root_dir)
    }


def _obj_files_under(root: Path) -> set[Path]:
    """Return .obj files below a directory, relative to that directory."""
    if not root.is_dir():
        return set()
    return {
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".obj"
    }


def verify_database_complete(
    raw_path: Path, processed_path: Path, step_name: str
) -> None:
    """Require every raw file to exist in the processed database."""
    raw_files = _obj_files_under(raw_path)
    processed_files = _obj_files_under(processed_path)
    if not raw_files:
        raise RuntimeError(f"No files found in raw database: {raw_path}")

    missing = sorted(raw_files - processed_files, key=str)
    if missing:
        preview = "\n".join(f"  - {p}" for p in missing[:25])
        if len(missing) > 25:
            preview += f"\n  ... and {len(missing) - 25} more"
        raise RuntimeError(
            f"Step '{step_name}' produced an incomplete database. "
            f"Missing {len(missing)} of {len(raw_files)} raw files in "
            f"{processed_path}:\n{preview}"
        )
    print(f"[CHECK] {step_name}: all {len(raw_files)} raw files are present.")
