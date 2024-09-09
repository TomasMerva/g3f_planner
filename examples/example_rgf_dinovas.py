# 1. Read yaml [x]
# 2. Create env [x]
# 3. Create planner (it includes rollouts, QP and outputs the waypoints)
# 4. Reference tracker
# 5. Execute fabrics

import numpy as np
import copy
import yaml
import time
import os
from scipy.spatial.transform import Rotation as R


from dinovas_pybullet_env import Environment
from rgf_planner import RGF_Planner

if __name__=="__main__":
    RENDER = True
    NUM_ROBOTS = 2
    NUM_DOF = 11
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = 2000

    # Environment
    env = Environment()
    (sim, goal) = env.initialize(render=RENDER, nr_robots=NUM_ROBOTS)
    CONFIG_FILE = env.get_config_file()
    action = np.zeros(NUM_ROBOTS*NUM_DOF)
    ob, *_ = sim.step(action)

    # Planer
    fk_args = dict(
        urdf_file = env.ROBOT_URDF_FILE,
        root_link = "world",
        end_link = "arm_tool_frame",
        num_dofs = NUM_DOF-NUM_GRIPPER_FINGERS,
    )
    planner = RGF_Planner(fk_args=fk_args,
                          config_file=CONFIG_FILE)


    for timestep in range(NUM_TIMESTEPS):
        
        
        ob, *_ = sim.step(action)
    sim.close()