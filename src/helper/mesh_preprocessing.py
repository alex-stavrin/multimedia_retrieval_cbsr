"""Preprocessing steps for query meshes before descriptor extraction."""

import vedo

from src.step_2.repairment.repair_wireframe import run_steps as repair
from src.step_2._3.resampling import resample_mesh
from src.step_3._1.full_normalization import full_normalize


def preprocess_mesh(mesh: vedo.Mesh) -> vedo.Mesh:
    """Repair, resample, and normalize a copy of a query mesh."""
    processed_mesh = mesh.clone()
    repaired_mesh, _ = repair(processed_mesh)
    processed_mesh = resample_mesh(repaired_mesh)
    full_normalize(processed_mesh)
    return processed_mesh
