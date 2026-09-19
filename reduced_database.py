import argparse
import os
import random
import shutil

DEFAULT_SHAPES_PER_CLASS = 4


def list_classes(root: str) -> dict[str, list[str]]:
    """Map each class (subfolder of `root`) to its .obj file paths."""
    classes: dict[str, list[str]] = {}

    for entry in sorted(os.listdir(root)):
        class_dir = os.path.join(root, entry)
        if not os.path.isdir(class_dir):
            continue

        obj_files = sorted(f for f in os.listdir(class_dir) if f.endswith(".obj"))
        if obj_files:
            classes[entry] = [os.path.join(class_dir, f) for f in obj_files]

    return classes


def select_balanced_subset(
    root: str, shapes_per_class: int = DEFAULT_SHAPES_PER_CLASS, seed: int = 42
) -> dict[str, list[str]]:
    """Randomly select up to `shapes_per_class` shapes from every class, so
    the reduced database stays balanced even though class sizes in the full
    database vary widely (15 to 143 shapes)."""
    rng = random.Random(seed)
    classes = list_classes(root)

    selection: dict[str, list[str]] = {}
    for shape_class, paths in classes.items():
        n = min(shapes_per_class, len(paths))
        selection[shape_class] = rng.sample(paths, n)

    return selection


def build_reduced_database(
    input_root: str,
    output_root: str,
    shapes_per_class: int = DEFAULT_SHAPES_PER_CLASS,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Copy a class-balanced subset of `input_root` into `output_root`,
    mirroring the class-subfolder layout, per the assignment's guidance to
    start development on a reduced database (>=200 shapes, most/all classes,
    roughly balanced) before scaling up to the full one."""
    selection = select_balanced_subset(input_root, shapes_per_class, seed)

    for shape_class, paths in selection.items():
        class_out_dir = os.path.join(output_root, shape_class)
        os.makedirs(class_out_dir, exist_ok=True)
        for path in paths:
            shutil.copy2(path, os.path.join(class_out_dir, os.path.basename(path)))

    return selection


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a reduced, class-balanced shape database for early development "
            "(assignment guidance: start small, then scale up once the pipeline works)."
        )
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the full shape database root")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to write the reduced database to")
    parser.add_argument(
        "-n",
        "--shapes-per-class",
        type=int,
        default=DEFAULT_SHAPES_PER_CLASS,
        help=f"Maximum shapes to take per class (default: {DEFAULT_SHAPES_PER_CLASS})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible selection")
    args = parser.parse_args()

    selection = build_reduced_database(args.input, args.output, args.shapes_per_class, args.seed)

    total = sum(len(paths) for paths in selection.values())
    counts = sorted((len(paths), shape_class) for shape_class, paths in selection.items())

    print(f"Selected {total} shapes across {len(selection)} classes")
    print(f"Smallest class: {counts[0][1]} ({counts[0][0]} shapes)")
    print(f"Largest class:  {counts[-1][1]} ({counts[-1][0]} shapes)")


if __name__ == "__main__":
    main()
