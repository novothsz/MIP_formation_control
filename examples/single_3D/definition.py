from typing import Any, Dict, List
from dataclasses import dataclass
import numpy as np
from parameters import *

from gurobipy import Model
from gurobipy import *
from gurobipy import GRB, quicksum

@dataclass
class TrajectoryProblem():
    n_states:int
    n_inputs:int
    n_parameters:int
    t_init:float
    p_init:np.ndarray
    x_init:np.ndarray
    u_init:np.ndarray
    p_init:np.ndarray


def set_dim(traj_pbm: Dict[str, Any] = {}) -> Dict[str, Any]:
    traj_pbm["n_states"] = 6
    traj_pbm["n_inputs"] = 4
    traj_pbm["n_parameters"] = 1
    return traj_pbm

# def set_scale(pbm:QuadrotorProblem, params: Dict[str, Any] = {}) -> Dict[str, Any]:
#     tdil_min = pbm.traj.tf_min
#     tdil_max = pbm.traj.tf_max
#     tdil_max_adj = tdil_min + 1.0 * (tdil_max - tdil_min)
#     params["t_init"] = tdil_max_adj
#     return params

def set_quess(quad_pbm:QuadrotorProblem, traj_pbm: Dict[str, Any] = {}) -> Dict[str, Any]:
    veh = quad_pbm.vehicle
    traj = quad_pbm.traj
    g = quad_pbm.env.g

    # Parameter guess
    p = np.zeros((traj_pbm.n_parameters, 1))
    p[veh.id_t] = 0.5 * (traj.tf_min + traj.tf_max)

    # State guess
    x0 = np.zeros((traj_pbm.n_states, 1))
    xf = np.zeros((traj_pbm.n_states, 1))
    x0[veh.id_r] = traj.r0
    xf[veh.id_r] = traj.rf
    x0[veh.id_v] = traj.v0
    xf[veh.id_v] = traj.vf
    x_init = np.linspace(x0, xf, traj_pbm.n_states).reshape(-1, 1)
    traj_pbm["x_init"] = x_init

    # Input guess
    hover = np.zeros((traj_pbm.n_inputs, 1))
    hover[veh.id_u] = -g
    hover[veh.id_sigma] = np.norm(g)
    u_init = np.tile(hover, (traj_pbm.n_states, 1))
    traj_pbm["u_init"] = u_init

    # Time guess
    p_guess = np.zeros((traj_pbm.n_parameters, 1))
    p_guess[veh.id_t] = 0.5 * (traj.tf_min + traj.tf_max)
    traj_pbm["p_init"] = p_guess

    return traj_pbm

# Now we need to create the MIP problem.
# We will use Gurobi as the solver.
def create_problem(quad_pbm:QuadrotorProblem, traj_pbm: Dict[str, Any] = {})
    model = Model("ppl")
    J = 0
    R = 1e5

    x, y, z = [], [], []
    vx, vy, vz = [], [], []
    ux, uy, uz = [], [], []
    sigma = []
    t = []

    N = 30
    def f()





