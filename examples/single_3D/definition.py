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
def create_problem(quad_pbm:QuadrotorProblem, traj_pbm: Dict[str, Any] = {}) -> Model:
    model = Model("ppl")
    J = 0
    R = 1e5

    x, y, z = [], [], []
    vx, vy, vz = [], [], []
    ux, uy, uz = [], [], []
    sigma = []
    t = []

    N = 50
    dt = 1 / N
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
        model.addConstr(vx[i+1] == vx[i] + (ux[i] + g[0]) * dt)
        model.addConstr(vy[i+1] == vy[i] + (uy[i] + g[1]) * dt)
        model.addConstr(vz[i+1] == vz[i] + (uz[i] + g[2]) * dt)

    # Control constraints
    for i in range(N-1):
        model.addConstr(ux[i] <= quad_pbm.vehicle.u_max)
        # model.addConstr(ux[i] >= quad_pbm.vehicle.u_min)
        model.addConstr(uy[i] <= quad_pbm.vehicle.u_max)
        # model.addConstr(uy[i] >= quad_pbm.vehicle.u_min)
        model.addConstr(uz[i] <= quad_pbm.vehicle.u_max)
        # model.addConstr(uz[i] >= quad_pbm.vehicle.u_min)

    # Tilt constraints
    # model.addConstr(σ * np.cos(veh.tilt_max) - a[3] <= 0, name="max_tilt")
    # TODO: Add constraints for tilt
        
    # Obstacle avoidance constraints
    for i in range(N):
        for j in range(quad_pbm.env.n_obs):

            obs_dx = quad_pbm.env.obs[j].c[0]
            obs_dy = quad_pbm.env.obs[j].c[1]
            obs_dz = quad_pbm.env.obs[j].c[2]
            d_obs = 1.0

            b_avoidance = model.addVars(6, vtype=GRB.BINARY, name='b_avoidance')
            model.addConstr(  x[i] - obs_dx >=  d_obs - R * b_avoidance[1] )
            model.addConstr( -x[i] + obs_dx >=  d_obs - R * b_avoidance[0] )
            model.addConstr(  y[i] - obs_dy >=  d_obs - R * b_avoidance[3] )
            model.addConstr( -y[i] + obs_dy >=  d_obs - R * b_avoidance[2] )
            # model.addConstr(  z[i] - obs_dz >=  d_obs - R * b_avoidance[5] )
            # model.addConstr(  z[i] + obs_dz >=  d_obs - R * b_avoidance[4] )
            # model.addConstr(   quicksum(b_avoidance[q_sum_idx] for q_sum_idx in range(6)) <= 5)
            model.addConstr(   quicksum(b_avoidance[q_sum_idx] for q_sum_idx in range(4)) <= 3)


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
        print("Model is feasible")
    except:
        feasible = False
        print("Model is infeasible")
    return model

def define_problem() -> Model:
    quad_pbm = get_quad_pbm()
    traj_pbm = set_dim()
    traj_pbm = set_quess(quad_pbm, traj_pbm)
    model = create_problem(quad_pbm, traj_pbm)


    N = 50

    sol_x = []
    sol_y = []
    sol_z = []
    sol_vx = []
    sol_vy = []
    sol_vz = []
    sol_ux = []
    sol_uy = []
    sol_uz = []

    solutions = model.getVars()
    for solution in solutions:
        if solution.VarName[0:len('x')] == 'x':
            sol_x += [solution.x]
        if solution.VarName[0:len('y')] == 'y':
            sol_y += [solution.x]
        if solution.VarName[0:len('z')] == 'z':
            sol_z += [solution.x]
        if solution.VarName[0:len('vx')] == 'vx':
            sol_vx += [solution.x]
        if solution.VarName[0:len('vy')] == 'vy':
            sol_vy += [solution.x]
        if solution.VarName[0:len('vz')] == 'vz':
            sol_vz += [solution.x]
        if solution.VarName[0:len('ux')] == 'ux':
            sol_ux += [solution.x]
        if solution.VarName[0:len('uy')] == 'uy':
            sol_uy += [solution.x]
        if solution.VarName[0:len('uz')] == 'uz':
            sol_uz += [solution.x]

    sol_x = np.array(sol_x).reshape((N, 1))
    sol_y = np.array(sol_y).reshape((N, 1))
    sol_z = np.array(sol_z).reshape((N, 1))

    # Plot solution
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(sol_x, sol_y)
    ax.plot(sol_x, sol_y, "k.")
    ax.grid(True) 
    # Plot the obstacle
    for obs in quad_pbm.env.obs:

        obs_dx = obs.c[0]
        obs_dy = obs.c[1]
        obs_dz = obs.c[2]

        # Square obstacle with poligon
        from matplotlib.patches import Polygon
        import matplotlib.patches as patches
        from matplotlib.lines import Line2D

        r = 1.0
        corners = np.array([[-r, -r], [-r, r], [r, r], [r, -r]])
        corners = corners + np.array([obs_dx, obs_dy])
        polygon = Polygon(corners, closed=True, fill=True, fc=(0,0,1,0.1), ec=(0,0,0,1), ls=':', lw=1, zorder = 2)
        ax.add_patch(polygon)
        # Plot center
        ax.plot(obs_dx, obs_dy, "ro")

        
    plt.show()

    # Plot z, ux, uy, uz on subplots
    fig, axs = plt.subplots(2)
    axs[0].plot(sol_z)
    axs[0].set_title('z')
    axs[0].grid(True) 
    axs[1].plot(sol_ux, label = 'ux')
    axs[1].plot(sol_uy, label = 'uy')
    axs[1].plot(sol_uz, label = 'uz')
    axs[1].legend() 
    axs[1].grid(True) 
    plt.show()
        
        
