import sys
sys.path.append('../')
sys.path.append("/Users/szilard/miniforge3/envs/circular_frenet/lib/python3.12/site-packages")
import os
import time
import argparse


import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# fig.savefig('figures/' + 'suss.png', dpi = 200)

from src import Group
import math
import os
os.system("mkdir log")
os.system("mkdir urdf")
os.system("mkdir csv")
os.system("mkdir figures")
os.system("mkdir video")
os.system("mkdir yaml")
os.system("mkdir coeffs")


def optimization_loop(stage : int = 0):
    # Create group
    start_position = [0.0, 0.0, 0.0]
    goal_position = [0.0, 0.0, 0.0]
    group = Group(n_vehicles=4, start_position = start_position, goal_position = goal_position, stage = 0)
    group.set_group_position(   
        position=group.start_position,targetHeight = 0, position_type='initial')
    group.set_group_position(
        position=group.goal_position, targetHeight = 0, position_type='final')
    
        
    # Prepare model
    # Setting the start and final time of the frenet path
    group.fp.ft_0 = (int(stage%2) * 180 + 1)  / 180 * math.pi
    group.fp.ft_f = (int(stage%2 + 1) * 180 - 1) / 180 * math.pi
    group.update_fp(stage)
    
    
    # Add obstacles
    if stage % 2 == 0:
        group.add_obstacles_single([0.5], [0], stage = stage, seed=stage)
    else:
        group.add_obstacles_gate([0.5], [10], stage = stage, seed=stage)
        # group.obstacles[-2].generate_urdf_torus(group.obstacles[-1].corners)
        
        
    group.organise_neighbours()
    group.prepare_model()
    
    # Solve problem
    group, feasible = group.optimize_model()
    
    group.plot_model(stage)
    if feasible == False:
        print("INFEASIBLE!!!")
    
    return [group, feasible]

if __name__ == "__main__":
    optimization_loop(stage = 0)