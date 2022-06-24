import numpy as np
from numpy import interp
from scipy.interpolate import interp1d
import math
import copy

from matplotlib.patches import Polygon
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import csv
import time
import pickle
import random



from casadi import MX, SX, Function, vertcat, dot, nlpsol, cos, sin, norm_2
from .spline import BSpline, BSplineBasis
from .spline_extra import definite_integral, shift_spline, shift_knot1_fwd, shift_knot1_bwd, shift_over_knot, extrapolate, crop_spline
from .environment import Environment


        


class VehicleBasis(Environment):
    def __init__(self):
        super().__init__()
        
        self.stage = []
        self.n_dimensions = 3 # this is considering a third state, the phi rotation angle
        # if changed, don't forget to adjust: y_min, ellipse_generator()
        self.state_len = self.n_dimensions * 2 
        
        
        self.n_dimensions_old = 2
        self.state_len_old = self.n_dimensions_old * 2 # [position_x, velocity_x] * 2 if we are in 2D
        
        
        self.state_degree = 3
        self.radious = 0.08
        self.T = 5

    
        self.x0 = []
        self.xf = []
        
        self.neighbours = []
        self.obstacles = []
        self.ID = -1
        
        # Test hyperparam
        self.slack = 0.00001
        self.feasibility_slack = 0.001
        # Hyperparams
        self.rho = 50# /5 # /50
        # self.rho_formation = 100# /5 # /50
        self.rho_input = 0.1 * 10 * 2 * 100
        self.rho_final_value = 0.1 * 10 * 100# * 1000
        self.rho_intermediate = 10
        
        self.epsilon = 0.05 # try to keep minimum epsilon distance from the obstacle
        # self.epsilon = self.radious # try to keep minimum epsilon distance from the obstacle
        self.safety_weight = 100 # cost parameter for epsilon
        
        self.knot_intervals = 15 # number of knots for the output (position) spline of the vehicle
        # self.t_resolution_length = 30
        
        self.y_min = [-2, -2, -math.pi * 2]
        self.y_max = [2, 2, math.pi * 2]
        
        
        assert len(self.y_min) == self.n_dimensions
        
        u__ = 10e-5
        u__ = 1
        self.u_min = [-50, -50]
        self.u_max = [50, 50]
        self.u_min = [-250, -250]
        self.u_max = [250, 250]
        self.u_min = [-u__, -u__]
        self.u_max = [u__, u__]
        
        # variable_history
        self.variable_history =  {'y' : [],         # x_update
                                  'y_j' : [],       # data_exchange_x_receive
                                  't_start' : [],
                                  't_end' : [],
                                  'xf': [],
                                  'x_intermediate_list' : [],
                                  'a_intermediate_list' : [],
                                  't_intermediate_list' : [],
                                  't_real_intermediate_list' : [],
                                  't_real_activation_list' : [],
                                  'current_configuration_position' : [],
                                  'x_update_time' : [],
                                  'z_update_time' : [],
                                  'a' : [],
                                  'b' : [],
                                  'solution' : [],
                                  'solver_stats' : [],
                                  'feasibility_dict' : [],
                                  'first_time_success' : [],
                                  'DvX_prior' : [],
                                  'DvX_posterior' : [],
                                  'PvX' : []
                }
        self.n_intermediate_ADMM = 1
        self.vehicle_positions_new = {'stage' : [], 'vehicle_positions_new' : []}
        self.MPC_version = []
        # message
        self.message_in = {}
        self.message_out = {}
        

    def set_position(self, position : list, position_type : str):
        # 2D
        assert len(position) == self.n_dimensions
        if position_type == 'initial':
            self.x0 = position + [0] * len(position)
            # self.x0[2] = 1 # because cos(0) = 1
            self.current_configuration_position = position
        elif position_type == 'final':
            self.xf = position + [0] * len(position)
            # self.xf[2] = 1
            self.variable_history['xf'] += [self.xf]
        else:
            raise NotImplementedError()

        return self
    
    



