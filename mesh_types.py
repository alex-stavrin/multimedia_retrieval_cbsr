from dataclasses import dataclass
from numpy import float64
from enum import Enum


@dataclass
class Box:
    x_min: float64
    x_max: float64
    y_min: float64
    y_max: float64
    z_min: float64
    z_max: float64


class FaceType(Enum):
    TRIANGLES = 1
    QUADS = 2
    MIXED = 3
