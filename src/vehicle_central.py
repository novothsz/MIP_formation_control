import numpy as np
# from numpy import interp
import math
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# from casadi import MX, SX, Function, vertcat, nlpsol, cos, sin, norm_2 #, dot
# from casadi import dot
from casadi import sin, cos, MX
from .spline import BSpline, BSplineBasis


from .vehicle_basis import VehicleBasis
# from .param import ParamValX, ParamValZ, DecisionVarX, DecisionVarZ
# import time

# from matplotlib.collections import LineCollection
# from matplotlib.pyplot import cm
import csv

from gurobipy import Model
from gurobipy import *
from gurobi import GRB, quicksum
from matplotlib.patches import Polygon

from .gurobi_spline import define_gurobi_spline, define_constraint


class VehicleCentral(VehicleBasis):
    def __init__(self):
        super().__init__()
        
        
        
        
        self.plot_self_ID = 2
        self.plot_obst_IDX = 0 # 4
        
        self.t_start = 0.0
        self.t_step = 0.04 # 0.04 # Changed in simulation_step() upon first call
        self.t_window_size = 0.2
        self.t_end = self.t_start + self.t_window_size
        self.simulation = False
        self.shift_enabled = False
        
        self.current_configuration_position = []
        self.vehicles = []
        
        self.model = []
        
        
    def plot_model(self, stage):
        """Plotting the solution"""
        
        ##########################
        # --- Reconstruction --- #
        ##########################
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        N = len(basis)
        res = self.model.getVars()
        
        
        sol_x, sol_y, sol_c, sol_s, sol_sigma, sol_b_collection = [], [], [], [], [], []
        sol_dx, sol_ddx, sol_dy, sol_ddy = [], [], [], []
        sol_ux, sol_uy = [], []
        sol_x_tmp, sol_y_tmp = [], []
        for i, sol_ in enumerate(res):
            
            if sol_.VarName[0:len('sigma')] == 'sigma':
                sol_sigma += [sol_.x]
                
            elif sol_.VarName[0] == 'c':
                sol_c += [sol_.x]
                
            elif sol_.VarName[0] == 's':
                sol_s += [sol_.x]
                
            elif sol_.VarName[0:len('dx')] == 'dx':
                sol_dx += [sol_.x]
                print(sol_.VarName)
            elif sol_.VarName[0:len('dy')] == 'dy':
                sol_dy += [sol_.x]
            elif sol_.VarName[0:len('ddx')] == 'ddx':
                sol_ddx += [sol_.x]
            elif sol_.VarName[0:len('ddy')] == 'ddy':
                sol_ddy += [sol_.x]
            elif sol_.VarName[0:len('ux')] == 'ux':
                sol_ux += [sol_.x]
            elif sol_.VarName[0:len('uy')] == 'uy':
                sol_uy += [sol_.x]
                
                
            # tbh, we don't care about this:
            elif sol_.VarName[0] == 'b':
                sol_b_collection += [sol_.x]
                
            elif sol_.VarName[0] == 'x':
                sol_x_tmp += [sol_.x]
                if len(sol_x_tmp) == N:
                    sol_x += [sol_x_tmp]
                    sol_x_tmp = []
                    
            elif sol_.VarName[0] == 'y':
                sol_y_tmp += [sol_.x]
                if len(sol_y_tmp) == N:
                    sol_y += [sol_y_tmp]
                    sol_y_tmp = []


            
        ####################
        # --- Plotting --- #
        ####################
        fig, ax = plt.subplots()
        self.vehicles[0].plot_environment(ax, 0)
        
        # plotting obstacles
        for obstacle in self.vehicles[0].obstacles:
            obstacle.plot_obstacle(ax)
            
        # pltting trajectory & position at time-instance
        for veh_idx in range(len(self.vehicles)):
            # t = np.linspace(0, 1, 100)
            t = np.linspace(0, 1, len(basis))
            
            # trajectory
            zsol_t = [self.fp.frenet_to_inertial(sol_x[veh_idx][t_idx], sol_y[veh_idx][t_idx], t_) for t_idx, t_ in enumerate(t)]
            sol_x_t = [zsol_t_[0] for zsol_t_ in zsol_t]
            sol_y_t = [zsol_t_[1] for zsol_t_ in zsol_t]
            ax.plot(sol_x_t, sol_y_t)
            if veh_idx == 1:
                ax.plot(sol_x_t, sol_y_t, 'k')
            
            # position at time-instance
            t = np.linspace(0, 1, N)
            zsol_t = [self.fp.frenet_to_inertial(sol_x[veh_idx][t_idx], sol_y[veh_idx][t_idx], t_) for t_idx, t_ in enumerate(t)]
            sol_x_t = [zsol_t_[0] for zsol_t_ in zsol_t]
            sol_y_t = [zsol_t_[1] for zsol_t_ in zsol_t]
            # if t_idx % 2 == 1 or t_idx == 0:
            #     ax.plot(sol_x_t, sol_y_t, 'k.')
            
            
            ax = self.vehicles[0].plot_drone(ax, sol_x_t[0], sol_y_t[0], 0)
            ax = self.vehicles[0].plot_drone(ax, sol_x_t[-1], sol_y_t[-1], 0)
            basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
            half_index = int(len(basis) / 2)
            ax = self.vehicles[0].plot_drone(ax, sol_x_t[half_index], sol_y_t[half_index], math.pi/4)
            
        # plotting polygon
        for t_idx, t_ in enumerate(t):
            corners = []
            for veh_idx in range(len(self.vehicles)):
                zsol_t = self.fp.frenet_to_inertial(sol_x[veh_idx][t_idx], sol_y[veh_idx][t_idx], t_)
                # corners += [[list(zsol_t[0])[0], list(zsol_t[1])[0]]]
                corners += [list(zsol_t)]
            
            corners = np.array(corners)
            corners = np.vstack((corners, corners[0, :]))
            polygon = Polygon(corners, closed=True, fill=True, fc=(0,0,1,0.1), ec=(0,0,0,1), ls=':', lw=1, zorder = 2)
            # polygon = Polygon(corners, closed=True, fill=True, fc=(1,0,0,0.6), ec=(0,0,0,1), lw=1, zorder = 2)
            # if t_idx % 2 == 1 or t_idx == 0:
            ax.add_patch(polygon)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-1, 7])
        ax.set_ylim([-7, 1])
        ax.set_ylim([-7, 7])
        plt.show()
        
        fig.savefig('figures/' + 'MIP_formation' + '{:0>1d}'.format(stage) +'.png', dpi = 200)
       
        return self
    
    
    def optimize_model(self):
        ###############################
        # --- Optimization start! --- #
        ###############################
        self.model.optimize()
        self.model.getVars()
        feasible = True
        try:   
            self.model.getVars()[0].x
            return [self, feasible]
        except:
            return [self, False]
        # return self
        
        
    def prepare_model(self):
        """This function creates a model for the optimization
        """
        
        
        ######################################
        # --- Initialization --- #
        ######################################
        
        self.state_degree = self.vehicles[0].state_degree
        self.n_dimensions_old = self.vehicles[0].n_dimensions_old
        self.fp = self.vehicles[0].fp
        self.knot_intervals = self.vehicles[0].knot_intervals
        self.border_x = self.vehicles[0].border_x
        self.border_y = self.vehicles[0].border_y
        d_obs = self.vehicles[0].radious*2
        basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        N = len(basis)
        alpha_acc = 0.001 * 1
        alpha_transform = 0.1 * 1
        det_lbw = 0.5 # math.sqrt(2)/2 
        det_ubw = 8
        max_trig_int = 200
        
        using_integer = False
        use_warm_start = False
        
        # Get obstacle tilt
        phi_rot = self.get_obstcale_rotation(N)
        
        
        ###########################
        # --- Forming the MIP --- #
        ###########################
        model = Model("ppl")
        # model.setParam('OutputFlag', 0)
        # model.setParam('LogToConsole', 0)
        J = 0
        R = 1e5
        
        if using_integer == True:
            c = model.addVars(N, lb = 0, ub = max_trig_int, name = 'c', vtype = GRB.INTEGER)
            s = model.addVars(N, lb = 0, ub = max_trig_int, name = 's', vtype = GRB.INTEGER)
            integer_multiply = det_ubw / max_trig_int 
        else:
            max_trig = det_ubw
            c = model.addVars(N, lb = 0, ub = max_trig, name = 'c')
            s = model.addVars(N, lb = 0, ub = max_trig, name = 's')

        
        ######################
        # --- Warm-start --- #
        ######################
        # basis = self.define_knots(degree = self.state_degree, knot_intervals = self.knot_intervals)
        coeffs_x, coeffs_y = [], []
        try:
            import pickle
            if use_warm_start == True:
                pickle_in = open("MIP_warm.pickle", "rb")
                pickle_dict = pickle.load(pickle_in)
                for t_idx, t_ in enumerate(np.linspace(0, 1, N)):
                    # happy values
                    c[t_idx].start = pickle_dict["c"][t_idx]
                    s[t_idx].start = pickle_dict["s"][t_idx]
                    
                coeffs_x = pickle_dict["coeffs_x"]
                coeffs_y = pickle_dict["coeffs_y"]
        except:
            print("Loading warm-start file failed")
            pass
            
            
            
        # Setting spline trajectories
        x, y = [], []
        xd, yd = [], []
        xdd, ydd = [], []
        for veh_idx in range(len(self.vehicles)):
            lower_bound_x, upper_bound_x = self.border_x[0], self.border_x[1]
            lower_bound_y, upper_bound_y = self.border_y[0], self.border_y[1]
            initial_x = [[0] * len(basis)] if coeffs_x == [] else [coeffs_x[veh_idx]]
            initial_y = [[0] * len(basis)] if coeffs_y == [] else [coeffs_y[veh_idx]]
            x_tmp, model = define_gurobi_spline(model = model, degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = 1, lower_bound = lower_bound_x, upper_bound = upper_bound_x, initial_values = initial_x, name = "x")
            y_tmp, model = define_gurobi_spline(model = model, degree = self.state_degree, knot_intervals = self.knot_intervals, n_spl = 1, lower_bound = lower_bound_y, upper_bound = upper_bound_y, initial_values = initial_y, name = "y")
            
            x += [x_tmp[0]]
            y += [y_tmp[0]]
            xd += [x_tmp[0].derivative()]
            yd += [y_tmp[0].derivative()]
            xdd += [x_tmp[0].derivative(2)]
            ydd += [y_tmp[0].derivative(2)]
            
        # initial and final constraints
        for veh_idx, vehicle in enumerate(self.vehicles):
            model = define_constraint(model = model, constraint = [x[veh_idx], y[veh_idx]], lower_bound = np.array(vehicle.x0) - self.slack, upper_bound = np.array(vehicle.x0) + self.slack, constraint_type = 'initial', name = ['x_initial', 'y_initial'])
            model = define_constraint(model = model, constraint = [x[veh_idx], y[veh_idx]], lower_bound = np.array(vehicle.xf) - self.slack, upper_bound = np.array(vehicle.xf) + self.slack, constraint_type = 'final', name = ['x_final', 'y_final'])
        
        
        # collision avoidance
        for t_idx, t_ in enumerate(np.linspace(0, 1, N)):
            for obst_idx, obstacle in enumerate(self.vehicles[0].obstacles):
                for veh_idx, vehicle in enumerate(self.vehicles):
                    b = model.addVars(4, lb = 0, vtype = GRB.BINARY, name = 'b')
                    
                    # Let us now correct for the tilting.
                    # We will rotate the obstacle corners with phi, calculate the appropriate dx, dy values and center value,
                    # and we will multiply the vehicle position with the rotation matrix
                    corners = self.corn_eval([], obstacle.corners, self.fp, t_)
                    center = self.fp.inertial_to_frenet(obstacle.center[0], obstacle.center[1], t_)
                
                    # rotation of the obstacle
                    corners_rot = [list(np.dot(self.cs(-phi_rot[t_idx][obst_idx]), np.array(corner))) for corner in corners ]
                    center_rot = list(np.dot(self.cs(-phi_rot[t_idx][obst_idx]), np.array(center)))
                    obs_dx, obs_dy = self.get_dx_dy(center_rot, corners_rot)
                    
                    
                    rotation_matrix = self.cs(-phi_rot[t_idx][obst_idx])
                    
                    # collision avoidance constraints, inluding rotation of vehicle
                    model.addConstr(  (rotation_matrix[0][0] * x[veh_idx](t_) + rotation_matrix[0][1] * y[veh_idx](t_)) - (center_rot[0] + obs_dx) >=  d_obs - R * b[1] )
                    model.addConstr( -(rotation_matrix[0][0] * x[veh_idx](t_) + rotation_matrix[0][1] * y[veh_idx](t_)) + (center_rot[0] - obs_dx) >=  d_obs - R * b[0] )
                    model.addConstr(  (rotation_matrix[1][0] * x[veh_idx](t_) + rotation_matrix[1][1] * y[veh_idx](t_)) - (center_rot[1] + obs_dy) >=  d_obs - R * b[3] )
                    model.addConstr( -(rotation_matrix[1][0] * x[veh_idx](t_) + rotation_matrix[1][1] * y[veh_idx](t_)) + (center_rot[1] - obs_dy) >=  d_obs - R * b[2] )
                    
                    
                    
                    model.addConstr(   quicksum(b[q_sum_idx] for q_sum_idx in range(4)) <= 3)
        # assert 0            
        # Formation constraints & costs
        for t_idx, t_ in enumerate(np.linspace(0, 1, N)):
            combination_satisfied = {}
            for i in range(len(self.vehicles)):
                for j in range(len(self.vehicles)):
                    combination_satisfied[i, j] = False
                    combination_satisfied[j, i] = False
            
            for veh_idx, vehicle in enumerate(self.vehicles):
                # cost for acceleration
                J += alpha_acc * (xdd[veh_idx](t_)**2 + ydd[veh_idx](t_)**2)
                # for every neighbour
                for _, neighbour_tmp in enumerate(vehicle.neighbours):
                    # neighbour = []
                    for neigh_idx, neighbour_ in enumerate(self.vehicles):
                        if neighbour_.ID == neighbour_tmp.ID:
                            # neighbour = neighbour_
                            # and neigh_idx is well, where the loop breaks :) This is what we will use :)
                            break
                                  
                    # vec1: what it should be
                    # vec2: what we have
                    # cross: the cross product of the two vectors. It is a function of t.
                    if combination_satisfied[veh_idx, neigh_idx] == True:
                        break
                    
                    # If this combination is not handled yet, the do it now & note, that
                    # we have also handled its siblig constraint
                    combination_satisfied[veh_idx, neigh_idx] = True
                    combination_satisfied[neigh_idx, veh_idx] = True
                    
                    # Formation constraint implementation
                    vec1_x = x[veh_idx](t_) - x[neigh_idx](t_)
                    vec1_y = y[veh_idx](t_) - y[neigh_idx](t_)
                    vec1 = [vec1_x, vec1_y]
                    vec2 = np.array(self.vehicles[veh_idx].xf[:self.n_dimensions_old]) - np.array(self.vehicles[neigh_idx].xf[:self.n_dimensions_old])
                    
                    
                    if using_integer == True:
                        vec2_x_rot = self.cs_happy(integer_multiply * c[t_idx], integer_multiply * s[t_idx])[0][0] * vec2[0] + self.cs_happy(integer_multiply * c[t_idx], integer_multiply * s[t_idx])[0][1] * vec2[1]
                        vec2_y_rot = self.cs_happy(integer_multiply * c[t_idx], integer_multiply * s[t_idx])[1][0] * vec2[0] + self.cs_happy(integer_multiply * c[t_idx], integer_multiply * s[t_idx])[1][1] * vec2[1]
                    else:
                        vec2_x_rot = self.cs_happy(c[t_idx], s[t_idx])[0][0] * vec2[0] + self.cs_happy(c[t_idx], s[t_idx])[0][1] * vec2[1]
                        vec2_y_rot = self.cs_happy(c[t_idx], s[t_idx])[1][0] * vec2[0] + self.cs_happy(c[t_idx], s[t_idx])[1][1] * vec2[1]
                    
                    vec2_rot = [vec2_x_rot, vec2_y_rot]
                    
                    # model.addConstr( vec1[0] == vec2_rot[0])
                    # model.addConstr( vec1[1] == vec2_rot[1])
                    self.slack = 0.001
                    model.addConstr( vec1[0] <= vec2_rot[0] + self.slack * 1)
                    model.addConstr( vec1[0] >= vec2_rot[0] - self.slack * 1)
                    model.addConstr( vec1[1] <= vec2_rot[1] + self.slack * 1)
                    model.addConstr( vec1[1] >= vec2_rot[1] - self.slack * 1)
                    
                    if using_integer == True:
                        model.addConstr(integer_multiply * c[t_idx] + integer_multiply * s[t_idx] >= det_lbw)
                        model.addConstr(integer_multiply * c[t_idx] + integer_multiply * s[t_idx] <= det_ubw)
                        if t_idx != 0:
                            J += alpha_transform * (integer_multiply * s[t_idx]  - integer_multiply * s[t_idx - 1])**2
                            J += alpha_transform * (integer_multiply * c[t_idx]  - integer_multiply * c[t_idx - 1])**2
                    else:
                        model.addConstr(s[t_idx] >= -c[t_idx] + 2 / math.sqrt(2) * det_lbw)
                        model.addConstr(s[t_idx] <= -c[t_idx] + det_ubw)
                        
                        if t_idx != 0:
                            J += alpha_transform * (s[t_idx]  - s[t_idx - 1])**2
                            J += alpha_transform * (c[t_idx]  - c[t_idx - 1])**2
                            
                        
                        
        
        model.setObjective(J, GRB.MINIMIZE)
        model.Params.Threads = 8
        model.Params.TimeLimit = 200
        model.Params.NonConvex = 2
        self.model = model
        
        
        return self
        
        
        
    def get_obstcale_rotation(self, N):
        """This function calculates the amount ot phi rotation of the obstacles.
        This is because the obstacle avoidance MIP formulation only works for obstacles,
        that are not tilted.
        """
        #######################################
        # --- Obtaining obstacle-rotation --- #
        #######################################
        
        phi_rot = []
        for t_idx, t_ in enumerate(np.linspace(0, 1, N)):
            obst_phi = []
            for obst_idx, obstacle in enumerate(self.vehicles[0].obstacles):
                
                # Obtaining obstacle corners at t
                corners = self.corn_eval([], obstacle.corners, self.fp, t_)
                corners += [corners[0]]
                
                # Calculating phi
                tmp_phi = []
                # looping through the coners
                for j in range(1, 5): # !
                    vec = np.array(corners[j]) - np.array(corners[j-1]) # pointing from one corner to the next
                    tmp_phi += [math.atan2(vec[1], vec[0])]
                
                # getting the smallest phi value
                min_deviation = math.inf
                min_deviation_idx = []
                for j, tmp_phi_ in enumerate(tmp_phi):
                    if abs(0 - tmp_phi_) < min_deviation:
                        min_deviation = abs(0 - tmp_phi_)
                        min_deviation_idx = j
                # saving the choosen phi (for the specific time instace)
                obst_phi += [tmp_phi[min_deviation_idx]]
            phi_rot += [obst_phi]
                
        return phi_rot
        
        
    def corn_eval(self, corners_spline, corners_sample, fp, t):
        """This function calculates and returns the corners of the obstacle at the given time.
        It returns a list [[corn1_x, corn1_y], [c2_x, c2_y], ... ]"""
        if corners_spline != []:
            res = []
            for corners in corners_spline:
                res += [[list(corners_(t)[0])[0] for corners_ in corners]]
            return res
        else:
            res = []
            for corner in corners_sample:
                res += [list(fp.inertial_to_frenet(corner[0], corner[1], t))]
            return res
    
    
    def get_dx_dy(self, center, corners):
        """This function calculates the vertical and horizontal extension of an obstacle, with 
        regards to the center.
        Important: to get the correct result, make sure, that the obstacle sides are horizontal and vertical and
        the center is actually in the center of the rectangle(!).
        actually, it would be nice from me, if I would add an assert function, that does this automatically :))
        """
        dx, dy = 0, 0
        for corner in corners:
            dx = max(dx, abs(center[0] - corner[0]))
            dy = max(dy, abs(center[1] - corner[1]))
        return dx, dy
        
        
    def cs(self, gamma):
        """Rotation matrix in 2D.
            Input: gamma
            Output: 2x2 rotation matrix
        """
        mx = [[cos(gamma), -sin(gamma)],
              [sin(gamma), cos(gamma)]]
        return np.array(mx)
    
    def cs_happy(self, c_happy, s_happy):
        """Rotation matrix - but the input is already a cos and sin value : ))
        Here we just create a matrix of these values."""
        mx = [[c_happy, -s_happy],
              [s_happy, c_happy]]
        return np.array(mx)
    
      
    def write_csv_for_vehicle(self, T, px, py, vehicleID, stage):
        """This function writes a csv for only the specific stage.
        It won't include any other stages, but can be used later to assemble
        trajectories for all stages."""
        
        pz = [0] * len(px[0])
        pj = [0] * len(px[0])
        first_line = ['duration', 'x^0', 'x^1', 'x^2', 'x^3', 'x^4', 'x^5', 'x^6', 'x^7', 'y^0', 'y^1', 'y^2', 'y^3', 'y^4', 'y^5', 'y^6', 'y^7', 'z^0', 'z^1', 'z^2', 'z^3', 'z^4', 'z^5', 'z^6', 'z^7', 'yaw^0', 'yaw^1', 'yaw^2', 'yaw^3', 'yaw^4', 'yaw^5', 'yaw^6', 'yaw^7']
        mode = 'w'
        with open(self.cwd + '/csv/' + 'stage_' + str(stage) + '_vehicle' + str(vehicleID) + '.csv', mode = mode) as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(first_line)
            for i in range(len(T)):
                writer.writerow(T[i] + px[i] + py[i] + pz + pj)
                
        return self  
        
        