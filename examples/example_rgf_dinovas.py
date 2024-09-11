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
from fabrics_planner import Fabrics
from fabrics_rollouts import ReferenceTracker


def transformation2dict(T : np.array) -> Dict:
    return {
        "position": T[:3,3],
        "orientation": R.from_matrix(T[:3,:3]).as_quat(),
    }

def dict2transformation(pose : dict) -> np.ndarray:
    T = np.eye(4)
    T[:3,3] = pose["position"]
    T[:3,:3] = R.from_quat(pose["orientation"]).as_matrix()
    return T

if __name__=="__main__":
    RENDER = True
    NUM_ROBOTS = 2
    NUM_DOF = 11
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = 10000
    PLANNER_FREQ = 5

    # Environment
    env = Environment()
    (sim, goal) = env.initialize(render=RENDER, nr_robots=NUM_ROBOTS)
    CONFIG_FILE_PATH = env.get_config_file_path()
    action = np.zeros(NUM_ROBOTS*NUM_DOF)
    ob, *_ = sim.step(action)
    obstacles = env.get_obstacles()

    # Planner
    fk_args = dict(
        urdf_file = env.ROBOT_URDF_FILE,
        root_link = "world",
        end_link = "arm_tool_frame",
        num_dofs = NUM_DOF-NUM_GRIPPER_FINGERS,
    )
    planner = RGF_Planner(fk_args=fk_args,
                          config_file_path=CONFIG_FILE_PATH)
    
    # Fabrics
    fabrics = Fabrics(robot_urdf_path=env.ROBOT_URDF_FILE,
                      config_file_path=CONFIG_FILE_PATH,
                      degrees_of_freedom=NUM_DOF-NUM_GRIPPER_FINGERS)
    
    # Reference
    reference_tracker = ReferenceTracker(ub=1.0, lb=0.2)

    T_W_RedCup, T_W_GrenCup = np.eye(4), np.eye(4)
    T_W_Goal = np.eye(4)

    # Main loop
    for timestep in range(NUM_TIMESTEPS):
        robot_states = [[ob["robot_"+str(i)]["joint_state"]["position"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)],
                         ob["robot_"+str(i)]["joint_state"]["velocity"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)]]
                        for i in range(NUM_ROBOTS)]
        T_W_EEFs_current = [planner.compute_fk(robot_states[i][0]) for i in range(NUM_ROBOTS)]

        T_W_RedCup[:3,3], redcup_quat = env.get_redcup_pose()
        T_W_GrenCup[:3,3], greencup_quat = env.get_greencup_pose()
        T_W_Objects_robots = [T_W_RedCup, 
                            T_W_GrenCup]
        x_obsts = [obstacles[i]["position"] for i in obstacles]
        r_obsts = [obstacles[i]["radius"] for i in obstacles]

        if timestep%PLANNER_FREQ == 0:
            for robot_id in range(1):
                start_time = time.perf_counter()
                waypoints_list, planner_status = planner.solve(joint_state=robot_states[robot_id], 
                                                              T_W_Obj=T_W_Objects_robots[robot_id],
                                                              x_obsts=x_obsts
                                                              )
                end_time = time.perf_counter()
                print(f"Solver status for robot {robot_id}: {planner_status}")
                print(f'Computational time: {end_time-start_time} s for robot: {robot_id}')                

                if RENDER:
                    if planner_status:
                        for i in range(len(waypoints_list)):
                            pybullet.addUserDebugPoints([waypoints_list[i][:3, 3].tolist()], [[0, 1, 0]], 7, 1.0)


                if np.linalg.norm(T_W_EEFs_current[robot_id][:3,3] - T_W_Objects_robots[robot_id][:3,3]) < 0.2:
                    T_W_Goal = waypoints_list[-1]
                else:
                    # Reference tracker
                    if waypoints_list is None or len(waypoints_list) == 0:
                        pass
                    else:
                        current_eef_pose = transformation2dict(T_W_EEFs_current[robot_id])
                        waypoint_dict = [transformation2dict(waypoints_list[i]) for i in range(len(waypoints_list))]
                        current_goal_dict, waypoint_dict, flag = reference_tracker.update_local_goal_pos_orient(current_eef_pose["position"], waypoint_dict)

                        if current_goal_dict is not None:
                            T_W_Goal = dict2transformation(current_goal_dict)



            fabrics.update_arguments(joint_state= robot_states[robot_id],
                                     T_W_Goal=T_W_Goal,
                                     obst_pos=x_obsts,
                                     obst_radius=None)
            action[(robot_id*NUM_DOF): NUM_DOF*robot_id + (NUM_DOF-NUM_GRIPPER_FINGERS)] = fabrics.clip_action(fabrics.compute_action())

        ob, *_ = sim.step(action)
    sim.close()