from .vehicle import Vehicle
import numpy as np
from math import *
import matplotlib.pyplot as plt
from .frenet_path import FrenetPath
from .environment import Environment
import yaml
from numpy import interp
import time
import csv
import random
import copy
import os

from gurobipy import Model
from gurobipy import *
from gurobipy import GRB
from matplotlib.patches import Polygon


from .obstacle import Obstacle
from .vehicle_central import VehicleCentral

class Group(Environment):
    def __init__(self, n_vehicles : int, start_position = [-0.8, 0, 0], goal_position = [0.8, 0, 0], stage = 0):
        self.vehicles = []
        for i in range(n_vehicles):
            vehicle = Vehicle()
            vehicle.ID = i
            vehicle.stage = stage
            self.vehicles += [vehicle]
        self.start_position = start_position
        self.goal_position = goal_position
        self.stage = stage
        super().__init__()
        
        self.seed = []
        
        self.obstacles = []
        self.vehicle_central = VehicleCentral()
        
        
    
    import functools
    @functools.lru_cache(maxsize=None)
    def get_obstacle_corners(self, t):
        """ This function returns the corners for all the obstacles in a list.
        (To loop through: 
        """
        corners = []
        for obstacle in self.vehicles[0].obstacles:
            corners_tmp = [ [corner[0](t).tolist()[0][0], corner[1](t).tolist()[0][0]] for corner in obstacle.corners_spline]
            corners += [corners_tmp]
        return corners
    
    @functools.lru_cache(maxsize=None)
    def get_scaled_obstacle_corners(self, t):
        """ This function returns the corners for all the obstacles in a list.
        (To loop through: 
        """
        corners = []
        for obstacle in self.vehicles[0].obstacles:
            corners_tmp = [ [corner[0](t).tolist()[0][0], corner[1](t).tolist()[0][0]] for corner in obstacle.scaled_corners_spline]
            corners += [corners_tmp]
        return corners
    
    

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    def update_fp(self, stage):
        """Updating the frenet path values."""
        
        # In this code the frenet path gives us a circular path and below we set
        # (depending on the "stage" variable), whether we want the upper or 
        # lower part of the circle.
        for vehicle in self.vehicles:
            vehicle.fp.ft_0 = (int(stage%2) * 180 + 1)  / 180 * math.pi
            vehicle.fp.ft_f = (int(stage%2 + 1) * 180 - 1) / 180 * math.pi
            vehicle.obstacles = self.obstacles
        self.vehicle_central.fp.ft_0 = (int(stage%2) * 180 + 1)  / 180 * math.pi
        self.vehicle_central.fp.ft_f = (int(stage%2 + 1) * 180 - 1) / 180 * math.pi
        self.vehicle_central.obstacles = self.obstacles
        
    
    
    def plot_model(self, stage):
        self.vehicle_central.plot_model(stage)
        return self
    
    def optimize_model(self):
        vehicle_central, feasibility = self.vehicle_central.optimize_model()
        return [self, feasibility]
        

    def set_group_position(self, position : list, targetHeight : float = 0.8,  position_type : str = 'initial'):
        """This function sets the initial/final positions of the group. It also writes and saves it to a yaml file."""
        if position_type == 'initial':
            position = self.start_position
            positions = self.ellipse_generator(centerpoint = position, n_positions = len(self.vehicles), a = self.vehicles[0].radious * 10, b = self.vehicles[0].radious * 4.5,
                                               ellipse_rotation = math.pi / 2 * 1 + math.pi * (self.stage%2))
            

        
        
            if self.stage == 0:
                yaml_dict = {'crazyflies' : []}
                for i, vehicle in enumerate(self.vehicles):
                    # initialPosition = [positions[i][j].tolist() for j in range(len(positions[i]))] + [targetHeight]
                    initialPosition = [float(pos) for pos in positions[i]]
                    initialPosition = initialPosition[:2] + [float(targetHeight)]
                    yaml_dict['crazyflies'] += [{'id' : i, 'channel' : 100,
                                                 'initialPosition' : initialPosition,
                                                 'type' : 'default'
                                                 }]
                with open(self.cwd + "/yaml/initialPosition.yaml", "w") as file_descriptor:
                    yaml.dump(yaml_dict, file_descriptor)

        elif position_type == 'final':
            position = self.goal_position
            positions = self.ellipse_generator(centerpoint = position, n_positions = len(self.vehicles), a = self.vehicles[0].radious * 10, b = self.vehicles[0].radious * 4.5,
                                               ellipse_rotation = math.pi / 2 * 1 + math.pi * (self.stage%2))
            self.og_final_positions = positions
            if self.stage == 0:
                yaml_dict = {'crazyflies' : []}
                for i, vehicle in enumerate(self.vehicles):
                    # initialPosition = [positions[i][j].tolist() for j in range(len(positions[i]))] + [targetHeight]
                    initialPosition = [float(pos) for pos in positions[i]]
                    initialPosition = initialPosition[:2] + [float(targetHeight)]
                    yaml_dict['crazyflies'] += [{'id' : i, 'channel' : 100,
                                                 'finalPosition' : initialPosition,
                                                 'type' : 'default'
                                                 }]
                with open(self.cwd + "/yaml/finalPosition.yaml", "w") as file_descriptor:
                    yaml.dump(yaml_dict, file_descriptor)
        else:
            NotImplementedError()


        for i in range(len(self.vehicles)):
            self.vehicles[i].set_position(position = positions[i], position_type = position_type)


    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################


    def ellipse_generator(self, centerpoint : list, n_positions : int, a : float, b : float, 
                          ellipse_rotation : float = 0, vehicles_rotation : float = math.pi / 4,
                          ellipse_scale_x : float = 1, ellipse_scale_y : float = 1):
        """Generates positions distributed on an ellipse"""
        
        # Rotating vehicles on the ellipse
        alpha = 0 + vehicles_rotation
        # Scaling
        a *= ellipse_scale_x
        b *= ellipse_scale_y
        positions = []
        for i in range(n_positions):
            positions += [ [centerpoint[0] + a * np.cos(alpha), centerpoint[1] + b * np.sin(alpha), centerpoint[2]] ] # [p, q, phi]
            # Because of having cos_phi and sin_phi instead of a single phi value, our position array will be 4 long
            # positions += [ [centerpoint[0] + a * np.cos(alpha), centerpoint[1] + b * np.sin(alpha), np.cos(centerpoint[2]), np.sin(centerpoint[2])] ] # [p, q, cos_phi, sin_phi]
            alpha += np.pi * 2.0 / n_positions
        
        # Rotating the ellipse itself with the vehicles already in place
        if ellipse_rotation != 0 and centerpoint[:2] == [0.0, 0.0]:
            for i, pos in enumerate(positions):
                positions[i][:2] = self.rotate_vector(pos[:2], ellipse_rotation)
                
        if ellipse_rotation != 0 and centerpoint[:2] != [0.0, 0.0]:
            # raise NotImplementedError("Please set the centerpoint to [0, 0]")
            # Actually, not an error... We can still rotate around (0, 0) and then shift the 
            # corners with the centerpoint vector.
            
            for i, pos in enumerate(positions):
                positions[i][:2] = self.shift_vector(positions[i][:2], list(- 1 * np.array(centerpoint[:2])))
                positions[i][:2] = self.rotate_vector(pos[:2], ellipse_rotation)
                positions[i][:2] = self.shift_vector(positions[i][:2], centerpoint[:2])
                
        return positions

    
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    
    def rotate_formation(self, vehicle_positions, angle):
        vehicle_positions_new = []
        for position in vehicle_positions:
            vehicle_positions_new += [self.rotate_vector(position, angle)]
            
        return vehicle_positions_new
        
    def scale_formation(self, vehicle_positions, scaling_factor):
        vehicle_positions_new = []
        for position in vehicle_positions:
            vehicle_positions_new += [self.scale_vector(position, scaling_factor)]
            
        return vehicle_positions_new
        
    def rotate_vector(self, vector, angle):
        a1, a2 = vector[0], vector[1]
        
        return (a1 * math.cos(angle) - a2 * math.sin(angle), \
                        a1 * math.sin(angle) + a2 * math.cos(angle))
            
    def shift_vector(self, vector1, vector2):
        a1, a2 = vector1[0], vector1[1]
        b1, b2 = vector2[0], vector2[1]
        
        return [a1 + b1, a2 + b2]
        
        
    def scale_vector(self, vector, scaling_factor):
        scaled_vector = [vector[0] * scaling_factor, vector[1] * scaling_factor]
        
        return scaled_vector

    def add_obstacles(self, obstacles : list):
        for i in range(len(self.vehicles)):
            self.vehicles[i].obstacles = obstacles
        return self


    def organise_neighbours(self):
        """Sets up the l_neighbours for the vehicles in the group. Every agent
        n number of neighbours are assigned. The value n is hard-coded in
        the is_this_my_neighbour() function.

        Parameters
        ----------

        Returns
        -------
        self
        """
        for i in range(len(self.vehicles)):
            neighbours = []
            for j in range(len(self.vehicles)):
                my_id = self.vehicles[i].ID
                neighbour_id = self.vehicles[j].ID
                condition = self.is_this_my_neighbour(my_id = my_id, neighbour_id = neighbour_id)
                if condition == True:
                    neighbours += [self.vehicles[j]]
            self.vehicles[i].neighbours = neighbours
        return self

    def is_this_my_neighbour(self, my_id : int, neighbour_id : int):
        """Helper function, which tells wether neighbour_id is a
           neighbour of my_id or not
           (currently 2-distance neighbourhood is hard-coded -> max_on)

        Parameters
        ----------
        my_id : int
            ID of current vehicle
        neighbour_id : int
            ID of the other vehicle, whose neighbourhood is questioned

        Returns
        -------
        res : bool
            True if neighbour_id is a neighbour, False otherwise
        """
        # First we have to create a list of numbers
        old_list = np.linspace(0, len(self.vehicles) - 1, len(self.vehicles))
        # We cut the list where I am at. Put the first cut in to the front, the rest to the back
        # Example: [0, 1, 2!, 3, 4, 5, 6] -> [2!, 3, 4, 5, 6, 0, 1]
        new_list = np.append( old_list[my_id:] , old_list[0:my_id] )
        old_list = new_list
        max_on = 4 # max_observed_neighbours. How many neighbours we have on our right and on our left. 2-> 2+2=4 neighbours
        if(max_on + 1 < len(new_list) and 0 < len(new_list)):
            if any([neighbour_id == i for i in old_list[1:max_on + 1]]) or any( [neighbour_id == i for i in old_list[-max_on:]] ):
                res = True
            else:
                res = False
        elif max_on + 1 >= len(new_list):
            # No need to check anything further, because in the outer
            # loop we are only looking at potential neighbours anyway
            if my_id != neighbour_id:
                res = True
            else:
                res = False
        else:
            res = False
        return res

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################


    def prepare_model(self):
        """Creating the optimization algorithm"""
        self.vehicle_central.vehicles = self.vehicles
        self.vehicle_central.prepare_model()
        
        return self

    def set_var(self, var):
        for i in range(len(self.vehicles)):
            if 'n_intermediate_ADMM' in var:
                self.vehicles[i].n_intermediate_ADMM = var['n_intermediate_ADMM']
                self.n_intermediate_ADMM = var['n_intermediate_ADMM']
            if 'stage' in var:
                self.vehicles[i].stage = var['stage']
                self.stage = var['stage']
                
            if 'new_positions' in var:
                self.vehicles[i].vehicle_positions_new['stage'] += [var['new_positions']['stage']]
                if var['new_positions']['vehicle_positions_new'] != []:
                    self.vehicles[i].vehicle_positions_new['vehicle_positions_new'] += [var['new_positions']['vehicle_positions_new'][i]]
                else:
                    self.vehicles[i].vehicle_positions_new['vehicle_positions_new'] += [var['new_positions']['vehicle_positions_new']]
            if 'new_times' in var:
                self.vehicles[i].vehicle_positions_new['stage'] += [var['new_times']['stage']]
                if var['new_times']['vehicle_times_new'] != []:
                    self.vehicles[i].vehicle_positions_new['vehicle_times_new'] += [var['new_times']['vehicle_times_new'][i]]
                else:
                    self.vehicles[i].vehicle_positions_new['vehicle_times_new'] += [var['new_times']['vehicle_times_new']]
                    
            if 't_step' in var:
                self.vehicles[i].t_step = var['t_step']
            if 't_window_size' in var:
                self.vehicles[i].t_window_size = var['t_window_size']
            if 't_end' in var:
                self.vehicles[i].t_end = var['t_end']
            if 'knot_intervals' in var:
                self.vehicles[i].knot_intervals = var['knot_intervals']
            if 't_resolution_length' in var:
                self.vehicles[i].t_resolution_length = var['t_resolution_length']
            if 'rho' in var:
                self.vehicles[i].rho = var['rho']
            if 'rho_input' in var:
                self.vehicles[i].rho_input = var['rho_input']
            if 'rho_final_value' in var:
                self.vehicles[i].rho_final_value = var['rho_final_value']
                
            if 'MPC_version' in var:
                self.vehicles[i].MPC_version = var['MPC_version']
                self.MPC_version = var['MPC_version']
            if 'n_of_saved_waypoints' in var:
                self.vehicles[i].n_of_saved_waypoints = var['n_of_saved_waypoints']



    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    def add_obstacles_gate(self, times, ids, stage : int = 0, seed : int = 42):
        random.seed(seed)
        # a_bound = [0.4, 0.7] # ranges for obstacle horizontal extent
        # b_bound = [0.4, 0.7] # ranges for obstacle vertical extent
        # alpha_bound = [-math.pi/2, math.pi/2] # ranges for obstacle rotation
        
        gate_gap_bound = [4 * self.vehicles[0].radious, 10 * self.vehicles[0].radious]
        gate_length_bound = 1.0
        
        obstacles = []
        for i, (t_, id_) in enumerate(zip(times, ids)):
            gate_points_tmp = random.uniform(gate_gap_bound[0], gate_gap_bound[1])
            
            # The lower part of the gate
            # corners = [top-right, top_left]
            gate1_inside_corners = [ [gate_length_bound / 2, -gate_points_tmp], \
                                      [-gate_length_bound / 2, -gate_points_tmp] ]
            
            # corners = [bottom-left, bottom-right]
            gate2_inside_corners = [ [gate_length_bound / 2, gate_points_tmp], \
                                      [-gate_length_bound / 2, gate_points_tmp] ]
            
            # transforming the frenet coordinates to inertial frame at random times
            g1 = [list(self.fp.frenet_to_inertial(corner[0], corner[1], t_)) for corner in gate1_inside_corners]
            g2 = [list(self.fp.frenet_to_inertial(corner[0], corner[1], t_)) for corner in gate2_inside_corners]
            
            
            g1_extra = []
            g1_extra += [list(self.fp.frenet_to_inertial(gate1_inside_corners[-1][0], gate1_inside_corners[-1][1] - 2, t_))]
            g1_extra += [list(self.fp.frenet_to_inertial(gate1_inside_corners[0][0], gate1_inside_corners[0][1] - 2, t_))]
            g1 += g1_extra
            g2_extra = []
            g2_extra += [list(self.fp.frenet_to_inertial(gate2_inside_corners[-1][0], gate2_inside_corners[-1][1] + 2, t_))]
            g2_extra += [list(self.fp.frenet_to_inertial(gate2_inside_corners[0][0], gate2_inside_corners[0][1] + 2, t_))]
            g2 += g2_extra
            
            obstacles += [Obstacle(ID = 3+i*2, corners = g1)]
            obstacles += [Obstacle(ID = 3+i*2 + 1, corners = g2)]
            # Sharing ID-s between gate pairs
            obstacles[-2].gate_pair_ID = obstacles[-1].ID
            obstacles[-1].gate_pair_ID = obstacles[-2].ID
            
            
            
        for i in range(len(obstacles)):
            obstacle_replaced = False
            for j in range(len(self.obstacles)):
                if self.obstacles[j].ID == obstacles[i].ID:
                    self.obstacles[j] = obstacles[i]
                    obstacle_replaced = True
                    break
            if obstacle_replaced == False:
                self.obstacles += [obstacles[i]]
                
        
            
        for i in range(len(self.vehicles)):
            self.vehicles[i].obstacles = self.obstacles
           
        
        return self
    
    def add_obstacles_single(self, times, ids, stage : int = 0, seed : int = 42):
        """ Using this function we can add obstacles by only providing their center position.
        Thereafter, its size and orientation is decided randomly.
        """
        random.seed(seed)
        a_bound = [0.3, 1] # ranges for obstacle horizontal extent
        b_bound = [0.3, 1] # ranges for obstacle vertical extent
        x_shift_bound = [0, 0]
        y_shift_bound = [-1.2, 1.2]
        alpha_bound = [-math.pi/2, math.pi/2] # ranges for obstacle rotation
        # alpha_bound = [0, 0] # ranges for obstacle rotation
        
        obstacles = []
        if stage == 14:
            kappa = True
        for obst_idx, (t_, id_) in enumerate(zip(times, ids)):
            x_shift = random.uniform(x_shift_bound[0], x_shift_bound[1])
            y_shift = random.uniform(y_shift_bound[0], y_shift_bound[1])
            center = list(self.fp.frenet_to_inertial(0 + x_shift, 0 + y_shift, t_)) + [0]
            a = random.uniform(a_bound[0], a_bound[1])
            b = random.uniform(b_bound[0], b_bound[1])
            alpha = random.uniform(alpha_bound[0], alpha_bound[1])
            # print("a and b are: " + str([a, b]))
            ellipse_corners = self.ellipse_generator( centerpoint = center, n_positions = 4, a = a, b = b, 
                                                      ellipse_rotation = alpha, vehicles_rotation = math.pi / 4)
            obstacle_corners = [corner[:2] for corner in ellipse_corners]
            obstacle = Obstacle(ID = id_, corners = obstacle_corners)

            obstacles += [obstacle]

        for i in range(len(obstacles)):
            obstacle_replaced = False
            for j in range(len(self.obstacles)):
                if self.obstacles[j].ID == obstacles[i].ID:
                    self.obstacles[j] = obstacles[i]
                    obstacle_replaced = True
                    break
            if obstacle_replaced == False:
                self.obstacles += [obstacles[i]]
                
        
            
        for i in range(len(self.vehicles)):
            self.vehicles[i].obstacles = self.obstacles
        return self
            
        