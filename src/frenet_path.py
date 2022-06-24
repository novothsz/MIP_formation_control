import autograd.numpy as np
from autograd import grad
import math
from numpy import interp
from casadi import MX, SX, Function, vertcat, cos, sin, nlpsol
from .spline import BSpline, BSplineBasis
from .spline_extra import definite_integral, shift_spline, shift_knot1_fwd, shift_knot1_bwd, shift_over_knot, extrapolate

from .frenet_spline import SplineFitter

import matplotlib.pyplot as plt


class FrenetPath(object):
    def __init__(self, fx, fy, ft, ft_0, ft_f, T, N = 100):
        
        # path functions
        self.fx = fx
        self.fy = fy
        self.ft = ft
        
        self.fx_d = grad(self.fx)
        self.fy_d = grad(self.fy)
        
        # parameteres
        self.ft_0 = ft_0
        self.ft_f = ft_f
        self.T = T
        self.N = N
        
        
    def f_theta(self, t):
        tau = self.ft(t, self.ft_0, self.ft_f, T = 1)
        return math.atan2(self.fy_d(tau) / self.fx_d(tau), 1)
        
    def inertial_to_frenet(self, x, y, t):
        """
        Takes a position in the inertial frame and transforms it to
        pq coordinates in the frenet frame, given the frenet frame is at a
        position, defined by t.
        """
        x_frenet, y_frenet = self.fx(self.ft(t, self.ft_0, self.ft_f, T = 1)), self.fy(self.ft(t, self.ft_0, self.ft_f, T = 1))
        theta = self.f_theta(t)
        x_rel = x - x_frenet
        y_rel = y - y_frenet
        
        p = x_rel * cos(theta) + y_rel * sin(theta)
        q =-x_rel * sin(theta) + y_rel * cos(theta)
        
        return p, q
    
    def frenet_to_inertial(self, p, q, t):
        """
        Takes a position in the frenet frame and transforms it to
        xy coordinates, given the frenet frame is at a position, defined by t.
        """
        x, y = self.fx(self.ft(t, self.ft_0, self.ft_f, T = 1)), self.fy(self.ft(t, self.ft_0, self.ft_f, T = 1))
        theta = self.f_theta(t)
        
        x_ = p * cos(theta) - q * sin(theta)
        y_ = p * sin(theta) + q * cos(theta)
        
        return x + x_, y + y_
    
    
    
        
        
        
    def plot_path(self, ax, t, t_0 = [], t_f = []):
        # param processing
        t_0 = 0 if t_0 == [] else t_0
        t_f = 1 if t_f == [] else t_f
        # validation
        assert 0.0 <= t <= 1.0
        assert 0.0 <= t_0 <= 1.0
        assert 0.0 <= t_f <= 1.0
        
        # range [0, 1]
        t_samples = np.linspace(0, 1, 100)
        fx_ = [self.fx(self.ft(t_, self.ft_0, self.ft_f, T = 1)) for t_ in t_samples]
        fy_ = [self.fy(self.ft(t_, self.ft_0, self.ft_f, T = 1)) for t_ in t_samples]
        ax.plot(fx_, fy_, color = 'gray', linestyle = ':') # default color
        # range [t_0, t_f]
        t_samples = np.linspace(t_0, t_f, 100)
        fx_ = [self.fx(self.ft(t_, self.ft_0, self.ft_f, T = 1)) for t_ in t_samples]
        fy_ = [self.fy(self.ft(t_, self.ft_0, self.ft_f, T = 1)) for t_ in t_samples]
        
        ax.plot(fx_, fy_, c = 'yellow', ls = ':') # default color
        
        
        fx_t = self.fx(self.ft(t, self.ft_0, self.ft_f, T = 1))
        fy_t = self.fy(self.ft(t, self.ft_0, self.ft_f, T = 1))
        ax.plot(fx_t, fy_t, 'ko')
        # checking if inertial->frenet->inertial is working properly
        x, y = 0.3, 0.5
        ax.plot(x, y, 'yo')
        from random import random
        t_random = random()
        x_frenet, y_frenet = self.inertial_to_frenet(x, y, t_random)
        x_inertial, y_inertial = self.frenet_to_inertial(x_frenet, y_frenet, t_random)
        ax.plot(x_inertial, y_inertial, 'g.')
        ax.set_aspect('equal', adjustable='box')
        return ax
  