from casadi import SX, MX, Function, vertcat, sin, cos
from .frenet_path import FrenetPath
from matplotlib.patches import Polygon
import numpy as np
from numpy import interp
from .environment import Environment
from .spline import BSpline, BSplineBasis

from .spline_extra import shift_spline, shift_knot1_fwd, crop_spline, extrapolate
import math as math
from typing import List
import copy


class Obstacle(Environment):
    def __init__(self, ID : int = 0, corners: list = [], x_limits: list = [-0.2, 0.2], y_limits: list = [-0.3, 0.3]):
        super().__init__()
        self.ID = ID
        if corners:
            self.corners = corners
        else:
            self.corners = self.random_placement(x_limits, y_limits)
            # Calculating center
        points = self.corners
        center = []
        center += [np.mean(np.array(points)[:, 0]).tolist()]
        center += [np.mean(np.array(points)[:, 1]).tolist()]
        # center += [0.1]
        self.center = center
        self.max_dist_from_center = max([np.linalg.norm(np.array([self.center]) - np.array([corn])) for corn in self.corners])
        # self.max_dist_from_center = min(self.max_dist_from_center, 1)
        
        self.scaled_corners = self.scaled_corners(size = 0.03)    
        self.corners_spline = [] # these will be splines defined in the frenet frame.
        self.center_spline = []
        self.center_t = []
        self.corners_t = [] # sampling of the corner positions in the mooving frenet frame. They represent
        # the true value at time t and can be used measure if we have correctly fitted the spline.
        
        
        self.scaled_corners_spline = []
        self.scaled_corners_t = []
            
        self.gate_pair_ID = []
        
    
    def spline_position_in_frenet(self):
        """Fit a BSpline on the trajectory of the obstacle, as seen from the frenet frame."""
        do_the_fit = True
        
        # ---- Regular corners
        corners_spline_coeffs = []
        t = np.linspace(0, 1, 100)
        self.fitter.knot_intervals = 20
        for i in range(len(self.corners)):
            corner_ = np.array([self.fp.inertial_to_frenet(self.corners[i][0], self.corners[i][1], t_) for t_ in t])
            p_ = corner_[:, 0].tolist()
            q_ = corner_[:, 1].tolist()
            corner_ = [p_, q_]
            
            if do_the_fit == True:
                fitted_splines = self.fitter.fitting_single(corner_,
                                                     y_min = [-20, -20],
                                                     y_max = [20, 20]
                                                     )
                self.corners_spline += [fitted_splines]
                corners_spline_coeffs += [ [sp.coeffs for sp in fitted_splines] ]
            self.corners_t += [corner_]
            
        # ---- Scaled corners
        scaled_corners_spline_coeffs = []
        t = np.linspace(0, 1, 100)
        self.fitter.knot_intervals = 20
        for i in range(len(self.scaled_corners)):
            corner_ = np.array([self.fp.inertial_to_frenet(self.scaled_corners[i][0], self.scaled_corners[i][1], t_) for t_ in t])
            p_ = corner_[:, 0].tolist()
            q_ = corner_[:, 1].tolist()
            corner_ = [p_, q_]
            
            if do_the_fit == True:
                fitted_splines = self.fitter.fitting_single(corner_,
                                                     y_min = [-20, -20],
                                                     y_max = [20, 20]
                                                     )
                self.scaled_corners_spline += [fitted_splines]
                scaled_corners_spline_coeffs += [ [sp.coeffs for sp in fitted_splines] ]
            self.scaled_corners_t += [corner_]
        # self.plot_corners_spline()
        
        # ---- Virtual center "corner/circle" with virtual "radious"
        center_spline_coeffs = []
        t = np.linspace(0, 1, 100)
        self.fitter.knot_intervals = 20
        center = self.center
        center_ = np.array([self.fp.inertial_to_frenet(center[0], center[1], t_) for t_ in t])
        p_ = center_[:, 0].tolist()
        q_ = center_[:, 1].tolist()
        center_ = [p_, q_]
        
        if do_the_fit == True:
            fitted_splines = self.fitter.fitting_single(center_,
                                                 y_min = [-20, -20],
                                                 y_max = [20, 20]
                                                 )
            self.center_spline += fitted_splines
            center_spline_coeffs += [ [sp.coeffs for sp in fitted_splines] ]
        self.center_t += [center_]
        
        
        # ---- Collecting all the coeffs and writing it to file
        if do_the_fit == True:
            coeffs = {
                    "corners_spline_coeffs" : corners_spline_coeffs,
                    "scaled_corners_spline_coeffs" : scaled_corners_spline_coeffs,
                    "center_spline_coeffs" : center_spline_coeffs
                    }
            import pickle
            pickle_out = open("obst_" + str(int(self.ID)) + "_coeffs.pickle", "wb")
            pickle.dump(coeffs, pickle_out)
            pickle_out.close()
        
        
        else:
            # fitter = SplineFitter()
            basis = self.fitter.define_knots(degree = 3, knot_intervals = self.fitter.knot_intervals)
            import pickle
            pickle_in = open("obst_" + str(int(self.ID)) + "_coeffs.pickle", "rb")
            coeffs = pickle.load(pickle_in)
    
            self.corners_spline = [[BSpline(basis, xy) for xy in cxy] for cxy in coeffs["corners_spline_coeffs"]]
            self.scaled_corners_spline = [[BSpline(basis, xy) for xy in cxy] for cxy in coeffs["scaled_corners_spline_coeffs"]]
            self.center_spline = [BSpline(basis, xy) for xy in coeffs["center_spline_coeffs"]]
        
        return self
    
    
    def scaled_corners(self, proportion = 1, size = 0.08 * 2):
        """We can scale by proportion or by size.
        For example: scaling by 2 will increase the distance of the
        corners from the center twofold.
        By increasing by size, the sides of the obstacle will increase
        their distance from the center by 'size'."""
        
        center = np.array(self.center)
        scaled_corners = []
        # Scaling by proportion
        for i, corner in enumerate(self.corners):
            center_to_corner = corner - center
            new_center_to_corner = center_to_corner * proportion
            scaled_corners += [(center + new_center_to_corner).reshape(1, -1).tolist()[0]]
        
        scaled_corners = tuple(scaled_corners)
        size_scaled_corners = []
        # Scaling by size
        for i, corner in enumerate(scaled_corners):
            if i == 0:
                previous_corner = scaled_corners[-1]
            else:
                previous_corner = scaled_corners[i - 1]

            vector_1 = np.array(corner) - center
            unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
            # self.corners[i] = (   center + (vector_1 + unit_vector_1 * math.sqrt(size**2 + size**2))   ).reshape(1, -1).tolist()[0]
            size_scaled_corners += [(   center + (vector_1 + unit_vector_1 * math.sqrt(size**2 + size**2))   ).reshape(1, -1).tolist()[0]]
            
        return size_scaled_corners
    
    
    
    def plot_obstacle(self, ax):
        
        # Plotting of obstacle
        corners = np.array(self.corners)
        corners = np.vstack((corners, corners[0, :]))
        polygon = Polygon(corners, closed=True, fill=True, fc=(0,0,0,0.1), ec=(0,0,0,1), lw=1, zorder = 1)
        # polygon = Polygon(corners, closed=True, fill=True, fc=(1,0,0,1), ec=(0,0,0,1), lw=1, zorder = 1)
        ax.add_patch(polygon)
        ax.plot(self.center[0], self.center[1], 'k*')

        import csv
        data_to_save = self.center + [self.max_dist_from_center]
        file_path = "./coeffs/obst_" + str(int(self.ID)) + "_coeffs.csv"
        with open(file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(data_to_save)
      
            
    def cs(self, gamma):
        """Rotation matrix in 2D.
            Input: gamma
            Output: 2x2 rotation matrix
        """
        mx = [[cos(gamma), -sin(gamma)],
              [sin(gamma), cos(gamma)]]
        return np.array(mx)
            
    
    
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
