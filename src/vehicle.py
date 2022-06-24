import numpy as np
from numpy import interp
import math
from time import time
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from casadi import MX, SX, Function, vertcat, nlpsol, cos, sin, norm_2 #, dot
from casadi import dot
from .spline import BSpline, BSplineBasis
from .spline_extra import definite_integral


from .vehicle_basis import VehicleBasis
# from .param import ParamValX, ParamValZ, DecisionVarX, DecisionVarZ
import time

from matplotlib.collections import LineCollection
from matplotlib.pyplot import cm
import copy

class Vehicle(VehicleBasis):
    def __init__(self):
        super().__init__()
        
        self.plot_self_ID = 2
        self.plot_obst_IDX = 0 # 4
        self.plot_stage = 5
        
        self.t_start = 0.0
        self.t_step = 0.04 # 0.04 # Changed in simulation_step() upon first call
        self.t_window_size = 0.2
        self.t_end = self.t_start + self.t_window_size
        self.simulation = False
        self.shift_enabled = False
        
        self.n_of_saved_waypoints = int(self.t_window_size / self.t_step) + 1; epsilon = 10e-10; assert self.t_window_size % self.t_step > -epsilon and \
                                                                                                        self.t_window_size % self.t_step < epsilon# math.floor(self.t_window_size / self.t_step)
        
        self.n_of_saved_waypoints = 5
        self.waypoints = []
        self.waypoint_timestamps = []
        self.current_configuration_position = []
     
    def plot_rotation(self, x, y, theta):
        x_new = x * cos(theta) - y * sin(theta)
        y_new = x * sin(theta) + y * cos(theta)
        return x_new, y_new
    
    def plot_drone(self, ax, x0, y0, theta_c):
        
        # Draw body
        radious = 0.05
        radious = self.radious * 0.5 #* 0.9
        height = radious
        width = radious


        # Draw arms
        l = 0.1
        l = radious * 2


        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4)
        x1 = x0 + rot_x
        y1 = y0 + rot_y

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4*3)
        x2 = x0 + rot_x
        y2 = y0 + rot_y

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4*5)
        x3 = x0 + rot_x
        y3 = y0 + rot_y

        rot_x, rot_y = self.plot_rotation(l, 0, theta_c + np.pi/4*7)
        x4 = x0 + rot_x
        y4 = y0 + rot_y


        # Rotors
        # https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
        r_rotor = 0.03
        r_rotor = l * 0.3

        # Adding stuff to ax

        # Lines
        line1 = Line2D([x0, x1], [y0, y1], zorder = 8)
        line2 = Line2D([x0, x2], [y0, y2], zorder = 8)
        line3 = Line2D([x0, x3], [y0, y3], zorder = 8)
        line4 = Line2D([x0, x4], [y0, y4], zorder = 8)
        ax.add_line(line1)
        ax.add_line(line2)
        ax.add_line(line3)
        ax.add_line(line4)

        # Circles
        circle1 = plt.Circle((x1, y1), r_rotor, color='k', alpha=0.5, zorder = 10)
        circle2 = plt.Circle((x2, y2), r_rotor, color='k', alpha=0.5, zorder = 10)
        circle3 = plt.Circle((x3, y3), r_rotor, color='k', alpha=0.5, zorder = 10)
        circle4 = plt.Circle((x4, y4), r_rotor, color='k', alpha=0.5, zorder = 10)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        ax.add_patch(circle4)



        rect_x = - width/2
        rect_y = - height/2
        rot_x, rot_y = self.plot_rotation(rect_x, rect_y, theta_c)

        rectangle = patches.Rectangle((x0 + rot_x, y0 + rot_y), height, width,
                                      linewidth=1, edgecolor='k', facecolor='k',
                                      zorder = 9, angle = theta_c / np.pi * 180)
        ax.add_patch(rectangle)

        return ax
        