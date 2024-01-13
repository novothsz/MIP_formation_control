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

    def to_dict(self):
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "z": self.z.tolist(),
            "ux": self.ux.tolist(),
            "uy": self.uy.tolist(),
            "uz": self.uz.tolist(),
            "sigma": self.sigma.tolist(),
            "T": self.T,
            "optimisation_time": self.optimisation_time,
            "cost": self.cost,
            "feasible": self.feasible
        }
    @classmethod
    def from_dict(cls, data):
        return cls(
            x=np.array(data["x"]),
            y=np.array(data["y"]),
            z=np.array(data["z"]),
            ux=np.array(data["ux"]),
            uy=np.array(data["uy"]),
            uz=np.array(data["uz"]),
            sigma=np.array(data["sigma"]),
            T=data["T"],
            optimisation_time=data["optimisation_time"],
            cost=data["cost"],
            feasible=data["feasible"]
        )
@dataclass
class Environment:
    env_boundaries: np.ndarray # [x_min, x_max; y_min, y_max; z_min, z_max]
    start_pos: List[np.ndarray]
    end_pos: List[np.ndarray]
    obstacles: Optional[List[Ellipsoid]]
    solution: Optional[List[Solution]]

    def to_dict(self):
        return {
            "env_boundaries": self.env_boundaries.tolist(),
            "start_pos": [pos.tolist() for pos in self.start_pos],
            "end_pos": [pos.tolist() for pos in self.end_pos],
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles] if self.obstacles else None,
            "solution": [sol.to_dict() for sol in self.solution] if self.solution else None
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            env_boundaries=np.array(data["env_boundaries"]),
            start_pos=[np.array(pos) for pos in data["start_pos"]],
            end_pos=[np.array(pos) for pos in data["end_pos"]],
            obstacles=[Ellipsoid.from_dict(obstacle) for obstacle in data["obstacles"]] if data["obstacles"] else None,
            solution=[Solution.from_dict(sol) for sol in data["solution"]] if data["solution"] else None
        )
