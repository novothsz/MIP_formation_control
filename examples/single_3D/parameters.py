from typing import Any, Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class Ellipsoid:
    H:np.ndarray
    c:np.ndarray

@dataclass
class QuadrotorParameters:
    id_r: List[int]
    id_v: List[int]
    id_u: List[int]
    id_sigma: int
    id_t: int
    u_max: float
    u_min: float
    tilt_max: float

@dataclass 
class QuadrotorEnvironmentParameters:
    g: np.ndarray
    obs:List[Ellipsoid]
    n_obs:int


@dataclass
class QuadrotorTrajectoryParameters:
    r0: np.ndarray
    rf: np.ndarray
    v0: np.ndarray
    vf: np.ndarray
    tf_min: float
    tf_max: float
    gamma: float

@dataclass
class QuadrotorProblem:
    vehicle: QuadrotorParameters
    env: QuadrotorEnvironmentParameters
    traj: QuadrotorTrajectoryParameters


# Parameter setup
def get_quad_pbm() -> QuadrotorProblem:
    quad_params = QuadrotorParameters(
        id_r = [0, 1, 2],
        id_v = [3, 4, 5],
        id_u = [0, 1, 2],
        id_sigma = 4,
        id_t = 0,
        u_max = 23.2,
        u_min = 0.6,
        tilt_max = 60 * np.pi / 180,
    )
    obstacles = [Ellipsoid(H = np.diag([1.0, 1.0, 100.0]), c = np.array([3.0, 3.0, 0.0])),
                Ellipsoid(H = np.diag([1.0, 1.0, 100.0]), c = np.array([1.1, 3.0, 0.0]))]
    quad_env_params = QuadrotorEnvironmentParameters(
        g = np.array([0.0, 0.0, -9.81]).reshape((3, 1)),
        obs = obstacles,
        n_obs = len(obstacles),
    )

    traj = QuadrotorTrajectoryParameters(r0 = np.zeros((3, 1)),
                                        rf = np.array([2.5, 6.0, 0.0]).reshape((3, 1)),
                                        v0 = np.zeros((3, 1)),
                                        vf = np.zeros((3, 1)),
                                        tf_min = 0.0,
                                        tf_max = 2.5,
                                        gamma = 0.0,)

    quad_problem = QuadrotorProblem(vehicle = quad_params,
                                    env = quad_env_params,
                                    traj = traj)

    return quad_problem