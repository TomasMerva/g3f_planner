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
import pybullet
from typing import Dict

from dinovas_pybullet_env import Environment
from fabrics_planner import Fabrics

def compute_static_grasp(T_W_Obj):
        T_Obj_Grasp = np.eye(4)
        T_Obj_Grasp[:3,:3] = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
        T_Grasp_Theta = np.eye(4)
        T_Grasp_Theta[:3,:3] = R.from_euler('xyz', [90, 0, 0], degrees=False).as_matrix()
        T_W_Grasp = T_W_Obj @ T_Obj_Grasp @ T_Grasp_Theta
        T_W_Grasp[2,3] += 0.02
        return T_W_Grasp

if __name__=="__main__":
    RENDER = True
    NUM_ROBOTS = 2
    NUM_DOF = 11
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = 10000
    PLANNER_FREQ = 10

    # Environment
    env = Environment()
    (sim, goal) = env.initialize(render=RENDER, nr_robots=NUM_ROBOTS)
    CONFIG_FILE_PATH = env.get_config_file_path()
    action = np.zeros(NUM_ROBOTS*NUM_DOF)
    ob, *_ = sim.step(action)
    obstacles = env.get_obstacles()

    # Fabrics
    fabrics = Fabrics(robot_urdf_path=env.ROBOT_URDF_FILE,
                      config_file_path=CONFIG_FILE_PATH,
                      degrees_of_freedom=NUM_DOF-NUM_GRIPPER_FINGERS)
    
    T_W_RedCup, T_W_GrenCup = np.eye(4), np.eye(4)
    
    # Main loop
    for timestep in range(NUM_TIMESTEPS):
        robot_states = [[ob["robot_"+str(i)]["joint_state"]["position"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)],
                         ob["robot_"+str(i)]["joint_state"]["velocity"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)]]
                        for i in range(NUM_ROBOTS)]

        T_W_RedCup[:3,3], redcup_quat = env.get_redcup_pose()
        T_W_GrenCup[:3,3], greencup_quat = env.get_greencup_pose()
        T_W_Grasp_robots = [compute_static_grasp(T_W_RedCup), 
                            compute_static_grasp(T_W_GrenCup)]

        x_obsts = [obstacles[i]["position"] for i in obstacles]
        r_obsts = [obstacles[i]["radius"] for i in obstacles]

        for robot_id in range(1):
            fabrics.update_arguments(joint_state= robot_states[robot_id],
                                     T_W_Goal=T_W_Grasp_robots[robot_id],
                                     obst_pos=x_obsts,
                                     obst_radius=None)
            action_unclipped = fabrics.compute_action()
            action[(robot_id*NUM_DOF): NUM_DOF*robot_id + (NUM_DOF-NUM_GRIPPER_FINGERS)] = fabrics.clip_action(action_unclipped)

        ob, *_ = sim.step(action)
    sim.close()