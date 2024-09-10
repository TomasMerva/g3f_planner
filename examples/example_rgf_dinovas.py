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
from rgf_planner import RGF_Planner
from fabrics_rollouts import ReferenceTracker


def transformation2dict(T : np.array) -> Dict:
    return {
        "position": T[:3,3],
        "orientation": R.from_matrix(T[:3,:3]).as_quat(),
    }


if __name__=="__main__":
    RENDER = True
    NUM_ROBOTS = 2
    NUM_DOF = 11
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = 2000
    PLANNER_FREQ = 10

    # Environment
    env = Environment()
    (sim, goal) = env.initialize(render=RENDER, nr_robots=NUM_ROBOTS)
    CONFIG_FILE_PATH = env.get_config_file_path()
    action = np.zeros(NUM_ROBOTS*NUM_DOF)
    ob, *_ = sim.step(action)

    # Planner
    fk_args = dict(
        urdf_file = env.ROBOT_URDF_FILE,
        root_link = "world",
        end_link = "arm_tool_frame",
        num_dofs = NUM_DOF-NUM_GRIPPER_FINGERS,
    )
    planner = RGF_Planner(fk_args=fk_args,
                          config_file_path=CONFIG_FILE_PATH)
    
    # Reference
    reference_tracker = ReferenceTracker()

    T_W_RedCup, T_W_GrenCup = np.eye(4), np.eye(4)

    # Main loop
    for timestep in range(NUM_TIMESTEPS):
        robot_states = [[ob["robot_"+str(i)]["joint_state"]["position"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)],
                         ob["robot_"+str(i)]["joint_state"]["velocity"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)]]
                        for i in range(NUM_ROBOTS)]
        T_W_EEFs_current = [planner.compute_fk(robot_states[i][0]) for i in range(NUM_ROBOTS)]

        T_W_RedCup[:3,3], redcup_quat = env.get_redcup_pose()
        T_W_GrenCup[:3,3], greencup_quat = env.get_greencup_pose()


        if timestep%PLANNER_FREQ == 0:
            for robot_id in range(NUM_ROBOTS):
                start_time = time.perf_counter()
                waypoints_list, planner_status = planner.solve(joint_state=robot_states[robot_id], 
                                                              T_W_Obj=T_W_RedCup,
                                                              x_obsts=None
                                                              )
                end_time = time.perf_counter()
                print(f'Computational time: {end_time-start_time} s for robot: {robot_id}')                

                if waypoints_list is None or len(waypoints_list) == 0:
                    pass
                else:
                    current_eef_pose = transformation2dict(T_W_EEFs_current[robot_id])               
                    current_goal_dict, waypoints_list, flag = reference_tracker.update_local_goal_pos_orient(current_eef_pose["position"], waypoints_list)

                if current_goal_dict is None:
                    pass
                if not flag:
                    pass
                    #update_fabrics_goal

        ob, *_ = sim.step(action)
    sim.close()