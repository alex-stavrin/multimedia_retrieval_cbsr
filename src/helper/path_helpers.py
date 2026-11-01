from pathlib import Path


def normalize_shape_path(shape: str | Path) -> str:
    posix_path = str(shape).replace("\\", "/").strip("/")
    parts = [part for part in posix_path.split("/") if part and part != "."]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return posix_path
