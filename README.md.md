# Formation planning algorithm with collision avoidance in a moving Frenet frame using Mixed Integer Programming


# Purpose
This repository implements the method described in the work titled "---".

# Installation
- install the gurobi solver on your system
- install the gurobi wrapper for python
- install the dependencies listed in requirements.txt

# Code structure

The code is implemented in an OOP manner. The main classes are as follows.
- environment: The environment serves as the class, which holds the environment-related details, like the size of the environment and the frenet path
- frenet_path: This class implements the moving frenet frame. Various paths for the frame can be defined and helper functions are implemented to convert from inertial to frenet frame and vice versa.
- frenet_spline: A helper class to fit B-splines onto sequence of points.
- vehicle: A vehicle defines a drone, moving in the environment, implements basic plotting functionality and hold vehicle-specific data
- vehicle_central: the central unit, implementing the centralised optimization problem for multiple vehicle, that form a formation. 
- obstacle: The obstacle class defines obstacles, that must be avoided during flight. It is able to broadcast its (future) position as Bsplines, in the common Frenet frame (inherited from the environment).
- group: The group is the class, which implements actions, that requires "overseeing" all vehicles and all obstacles. It acts as a central hub for task and position distribution.

# Quick start
simply run:
```
python3 main.py
```


<img src="https://github.com/AIMotionLab-SZTAKI/MIP_formation_control/blob/main/doc/fig/payload.png" alt="Planning through a narrow corridor" width="500"/>

<img src="https://github.com/AIMotionLab-SZTAKI/MIP_formation_control/blob/main/doc/fig/simulation.gif" alt="Formation control in complex environments" width="500"/>
