import os
from dataclasses import dataclass

import numpy as np
import vedo

from src.helper.mesh_types import Box
from src.helper.normalization_utils import run_cli


@dataclass
class CenteringReport:
    barycenter_before: np.ndarray
    barycenter_after: np.ndarray
    bounding_box_before: Box
    bounding_box_after: Box
    scale_factor: float
    shape_path: str = ""


def center(mesh: vedo.Mesh) -> np.ndarray:
    """Translate the mesh's barycenter to the origin. Mutates `mesh` and
    returns the barycenter that was subtracted."""
    barycenter = mesh.center_of_mass()
    mesh.shift(*(-barycenter))
    return barycenter


def scale_to_unit_cube(mesh: vedo.Mesh) -> float:
    """Uniformly scale the mesh, about the world origin, so the largest side
    of its axis-aligned bounding box becomes exactly 1.

    This must run AFTER any rotation (e.g. the PCA alignment of Step 3.1): the
    bounding box depends on the orientation, so scaling before aligning would
    give copies of the same shape that are rotated differently different
    scale factors. Mutates `mesh` and returns the scale factor applied."""
    bounds = mesh.bounds()
    largest_side = max(
        bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
    )
    if largest_side <= 0:
        raise ValueError(
            "Mesh has zero extent on every axis, cannot scale to a unit cube."
        )

    scale_factor = 1.0 / largest_side
    # origin=False scales about world (0,0,0) rather than the mesh's own
    # transform position (vedo's default, origin=True), which would pivot
    # around the wrong point and reintroduce an offset.
    mesh.scale(scale_factor, origin=False)
    return scale_factor


def center_and_scale(mesh: vedo.Mesh) -> CenteringReport:
    """Step 2.5: center the mesh, then scale it so its largest bounding-box
    side is 1. Mutates `mesh`. The full pipeline of Step 3.1 uses `center()`
    and `scale_to_unit_cube()` separately, with alignment in between."""
    barycenter_before = mesh.center_of_mass()
    bounding_box_before = Box(*mesh.bounds())

    center(mesh)
    scale_factor = scale_to_unit_cube(mesh)

    return CenteringReport(
        barycenter_before=barycenter_before,
        barycenter_after=mesh.center_of_mass(),
        bounding_box_before=bounding_box_before,
        bounding_box_after=Box(*mesh.bounds()),
        scale_factor=scale_factor,
    )


def center_and_scale_file(input_path: str, output_path: str) -> CenteringReport:
    """Load a mesh, center and scale it, and write it to `output_path`."""
    if not os.path.exists(input_path):
        raise ValueError(f"File does not exist: {input_path}")

    mesh = vedo.Mesh(input_path)
    report = center_and_scale(mesh)
    report.shape_path = input_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh.write(output_path)

    return report


def main():
    run_cli(
        description="Center and scale 3D shapes: barycenter to the origin, "
        "scale to fit a unit cube.",
        process_file=center_and_scale_file,
        action="Centered and scaled",
    )


if __name__ == "__main__":
    main()
