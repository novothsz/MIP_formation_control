

from parameters import *
from definition import *
from data import Environment, Solution
from typing import List
from definition import Ellipsoid
import pickle
import os

def generate_obstacles(env: Environment, n_obs:int) -> List[Ellipsoid]:

    # c1=np.array([2.39978515, 7.17135522, 0.        ])
    # c2=np.array([1.85789218, 6.87768675, 0.        ])

    # obstacles=[Ellipsoid(H=np.diag([1.0, 1.0, 1.0]), c=c1),
    #                 Ellipsoid(H=np.diag([1.0, 1.0, 1.0]), c=c2)]
    
    # print(c1, np.linalg.norm(c1 - env.start_pos[0]) < 2.0, np.linalg.norm(c1 - env.end_pos[0]) < 2.0)
    # print(c2, np.linalg.norm(c2 - env.start_pos[0]) < 2.0, np.linalg.norm(c2 - env.end_pos[0]) < 2.0)
    # assert(not (np.linalg.norm(c - env.start_pos) < 2.0 or np.linalg.norm(c - env.end_pos) < 2.0))
    # assert(0)


    # obstacles=[Ellipsoid(H=np.diag([1.0, 1.0, 1.0]), c=np.array([3.0, 3.0, 0.0])),
    #                 Ellipsoid(H=np.diag([1.0, 1.0, 1.0]), c=np.array([1.1, 3.0, 0.0]))]

    obstacles=[]
    for i in range(n_obs):
        H=np.diag([1.0, 1.0, 1.0])


        def c_generator(env:Environment):
            c_x=np.random.uniform(low=env.env_boundaries[0][0], high=env.env_boundaries[0][1], size=(1,))
            c_y=np.random.uniform(low=env.env_boundaries[1][0], high=env.env_boundaries[1][1], size=(1,))
            c_z=np.random.uniform(low=env.env_boundaries[2][0], high=env.env_boundaries[2][1], size=(1,))
            return np.array([c_x, c_y, c_z]).reshape((3,))
        
        c=c_generator(env=env)
        while np.linalg.norm(c - env.start_pos[0]) < 2.0 or np.linalg.norm(c - env.end_pos[0]) < 2.0:
            c=c_generator(env=env)

        print(c, np.linalg.norm(c - env.start_pos[0]) < 2.0, np.linalg.norm(c - env.end_pos[0]) < 2.0)
        assert(not (np.linalg.norm(c - env.start_pos[0]) < 2.0 or np.linalg.norm(c - env.end_pos[0]) < 2.0))
        # assert(0)
        obstacles += [Ellipsoid(H=H, c=c)]

    return obstacles    


def generate_environment(n_obs:int) -> Environment:
    env_boundaries=np.array([[0.0, 4.0], [0.0, 8.0], [0.0, 0.0]])
    r0=np.zeros((3,))
    rf=np.array([2.5, 6.0, 0.0]).reshape((3,))          
    env=Environment(env_boundaries=env_boundaries, start_pos=[r0], end_pos=[rf], obstacles=None, solution=None)
    obstacles=generate_obstacles(env=env, n_obs=2)
    env.obstacles=obstacles
    return env


# If possible, load pickled environment
load_environment=True
if load_environment:
    # Pickle the last environment in the folder
    i = 0
    while os.path.exists(f"../AIConvexification.jl/data/{i}.pickle"):
        i += 1
    # open the last pickle file
    with open(f"../AIConvexification.jl/data/{i-1}.pickle", 'rb') as f:
        environment = pickle.load(f)

    tp=TrajectoryProblem(env=environment)
    solution: Solution = environment.solution

else:
    environment: Environment=generate_environment(2)
    tp=TrajectoryProblem(env=environment)
    tp.create_problem()
    solution: Solution
    solution=tp.get_solution()

    environment.solution=solution
    # Pickle environment
    # read files in the folder (filenames are: {integer}.pickle)
    # then, write new file, names: {max_integer+1}.pickle
    i=0
    while os.path.exists(f"../AIConvexification.jl/data/{i}.pickle"):
        i+=1
    with open(f"../AIConvexification.jl/data/{i}.pickle", 'wb') as f:
        pickle.dump(environment, f)


tp.plot_solution(env=environment)

