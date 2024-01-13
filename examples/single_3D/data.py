from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from parameters import Ellipsoid

@dataclass
class Solution:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    sigma: np.ndarray
    T: float
    optimisation_time: float
    cost: float
    feasible: bool

@dataclass
class Environment:
    env_boundaries: np.ndarray # [x_min, x_max; y_min, y_max; z_min, z_max]
    start_pos: List[np.ndarray]
    end_pos: List[np.ndarray]
    obstacles: Optional[List[Ellipsoid]]
    solution: Optional[List[Solution]]
