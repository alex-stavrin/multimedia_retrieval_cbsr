from dataclasses import dataclass
from enum import Enum


@dataclass
class Box:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


class FaceType(Enum):
    TRIANGLES = 1
    QUADS = 2
    MIXED = 3
