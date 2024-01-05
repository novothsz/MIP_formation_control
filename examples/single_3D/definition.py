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
    p = np.zeros((traj_pbm["n_parameters"], 1))
    p[veh.id_t] = 0.5 * (traj.tf_min + traj.tf_max)

    # State guess
    x0 = np.zeros((traj_pbm["n_states"], 1))
    xf = np.zeros((traj_pbm["n_states"], 1))
    x0[veh.id_r] = traj.r0
    xf[veh.id_r] = traj.rf
    x0[veh.id_v] = traj.v0
    xf[veh.id_v] = traj.vf
    x_init = np.linspace(x0, xf, traj_pbm["n_states"]).reshape(-1, 1)
    traj_pbm["x_init"] = x_init

    # Input guess
    hover = np.zeros((traj_pbm["n_inputs"]+1, 1))
    hover[veh.id_u] = -g
    hover[veh.id_sigma] = np.linalg.norm(g)
    u_init = np.tile(hover, (traj_pbm["n_states"], 1))
    traj_pbm["u_init"] = u_init

    # Time guess
    p_guess = np.zeros((traj_pbm["n_parameters"], 1))
    p_guess[veh.id_t] = 0.5 * (traj.tf_min + traj.tf_max)
    traj_pbm["p_init"] = p_guess

    return traj_pbm

# Now we need to create the MIP problem.
# We will use Gurobi as the solver.
def create_problem(quad_pbm:QuadrotorProblem, traj_pbm: Dict[str, Any] = {}):
    model = Model("ppl")
    J = 0
    R = 1e5

    x, y, z = [], [], []
    vx, vy, vz = [], [], []
    ux, uy, uz = [], [], []
    sigma = []
    t = []

    N = 30
    dt = 1 / 30
    x = model.addVars(N, lb=-np.inf, ub=np.inf, name='x')
    y = model.addVars(N, lb=-np.inf, ub=np.inf, name='y')
    z = model.addVars(N, lb=-np.inf, ub=np.inf, name='z')
    vx = model.addVars(N, lb=-np.inf, ub=np.inf, name='vx')
    vy = model.addVars(N, lb=-np.inf, ub=np.inf, name='vy')
    vz = model.addVars(N, lb=-np.inf, ub=np.inf, name='vz')
    ux = model.addVars(N-1, lb=-np.inf, ub=np.inf, name='ux')
    uy = model.addVars(N-1, lb=-np.inf, ub=np.inf, name='uy')
    uz = model.addVars(N-1, lb=-np.inf, ub=np.inf, name='uz')
    # sigma = model.addVars(N-1, lb=-np.inf, ub=np.inf, name='sigma')

    g = quad_pbm.env.g

    # Initial conditions
    model.addConstr(x[0] == quad_pbm.traj.r0[0])
    model.addConstr(y[0] == quad_pbm.traj.r0[1])
    model.addConstr(z[0] == quad_pbm.traj.r0[2])
    model.addConstr(vx[0] == quad_pbm.traj.v0[0])
    model.addConstr(vy[0] == quad_pbm.traj.v0[1])
    model.addConstr(vz[0] == quad_pbm.traj.v0[2])
    # Final conditions
    model.addConstr(x[N-1] == quad_pbm.traj.rf[0])
    model.addConstr(y[N-1] == quad_pbm.traj.rf[1])
    model.addConstr(z[N-1] == quad_pbm.traj.rf[2])
    model.addConstr(vx[N-1] == quad_pbm.traj.vf[0])
    model.addConstr(vy[N-1] == quad_pbm.traj.vf[1])
    model.addConstr(vz[N-1] == quad_pbm.traj.vf[2])

    # Dynamics
    for i in range(N-1):
        model.addConstr(x[i+1] == x[i] + vx[i] * dt)
        model.addConstr(y[i+1] == y[i] + vy[i] * dt)
        model.addConstr(z[i+1] == z[i] + vz[i] * dt)
        model.addConstr(vx[i+1] == vx[i] + (uz[i] + g[0]) * dt)
        model.addConstr(vy[i+1] == vy[i] + (uz[i] + g[1]) * dt)
        model.addConstr(vz[i+1] == vz[i] + (uz[i] + g[2]) * dt)

    # Control constraints
    for i in range(N-1):
        model.addConstr(ux[i] <= quad_pbm.vehicle.u_max)
        model.addConstr(ux[i] >= quad_pbm.vehicle.u_min)
        model.addConstr(uy[i] <= quad_pbm.vehicle.u_max)
        model.addConstr(uy[i] >= quad_pbm.vehicle.u_min)
        model.addConstr(uz[i] <= quad_pbm.vehicle.u_max)
        model.addConstr(uz[i] >= quad_pbm.vehicle.u_min)

    # Tilt constraints
    # model.addConstr(σ * np.cos(veh.tilt_max) - a[3] <= 0, name="max_tilt")
    # TODO: Add constraints for tilt
        
    # Obstacle avoidance constraints
    for i in range(N):
        for j in range(quad_pbm.env.n_obs):

            obs_dx = quad_pbm.env.obs[j].c[0]
            obs_dy = quad_pbm.env.obs[j].c[1]
            obs_dz = quad_pbm.env.obs[j].c[2]
            d_obs = 0.1

            b_avoidance = model.addVars(6, vtype=GRB.BINARY, name='b_avoidance')
            model.addConstr(  x[i] - obs_dx >=  d_obs - R * b_avoidance[1] )
            model.addConstr(  x[i] + obs_dx >=  d_obs - R * b_avoidance[0] )
            model.addConstr(  y[i] - obs_dy >=  d_obs - R * b_avoidance[3] )
            model.addConstr(  y[i] + obs_dy >=  d_obs - R * b_avoidance[2] )
            model.addConstr(  z[i] - obs_dz >=  d_obs - R * b_avoidance[5] )
            model.addConstr(  z[i] + obs_dz >=  d_obs - R * b_avoidance[4] )
            model.addConstr(   quicksum(b_avoidance[q_sum_idx] for q_sum_idx in range(6)) <= 5)


    # Minimum time constraint
    b_time = model.addVars(N, vtype=GRB.BINARY, name='b_time')
    for i in range(N):
        model.addConstr( x[i] - quad_pbm.traj.rf[0] <=  + R * (1 - b_time[i]) )
        model.addConstr( x[i] - quad_pbm.traj.rf[0] >=  - R * (1 - b_time[i]) )
        model.addConstr( y[i] - quad_pbm.traj.rf[1] <=  + R * (1 - b_time[i]) )
        model.addConstr( y[i] - quad_pbm.traj.rf[1] >= - R * (1 - b_time[i]) )
        model.addConstr( z[i] - quad_pbm.traj.rf[2] <= + R * (1 - b_time[i]) )
        model.addConstr( z[i] - quad_pbm.traj.rf[2] >= - R * (1 - b_time[i]) )
        model.addConstr( quicksum(b_time[q_sum_idx] for q_sum_idx in range(N)) == 1 )

    # Minimum time objective
    J += quicksum((idx+1) * b_time[q_sum_idx] for idx, q_sum_idx in enumerate(range(N)))
                 
    # Minimum energy objective
    J += quicksum(ux[q_sum_idx] ** 2 + uy[q_sum_idx] ** 2 + uz[q_sum_idx] ** 2 for q_sum_idx in range(N-1))

    model.setObjective(J, GRB.MINIMIZE)
    model.Params.Threads = 8
    model.Params.TimeLimit = 200
    model.Params.NonConvex = 2
    model.optimize()
    model.getVars()
    feasible = True
    try:   
        model.getVars()[0].x
        feasible = True
    except:
        feasible = False
    print(type(model))
    return model, feasible

class define_problem():
    quad_pbm = get_quad_pbm()
    traj_pbm = set_dim()
    traj_pbm = set_quess(quad_pbm, traj_pbm)
    model, feasible = create_problem(quad_pbm, traj_pbm)