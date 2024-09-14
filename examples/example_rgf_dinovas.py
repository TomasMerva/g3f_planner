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
    PLANNER_PERIOD = 100
    REFERENCE_TRACKER_PERIOD = 10


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
    reference_tracker = ReferenceTracker(ub=1.0, lb=0.1)

    T_W_RedCup, T_W_GrenCup = np.eye(4), np.eye(4)
    T_W_Goal_robots = [np.eye(4)] * NUM_ROBOTS
    waypoints_list_robots = [None]* NUM_ROBOTS
    solver_status_robots = [None] * NUM_ROBOTS

    x_obsts = [obstacles[i]["position"] for i in obstacles]
    r_obsts = [obstacles[i]["radius"] for i in obstacles]
    x_obsts_robots = [copy.deepcopy(x_obsts), copy.deepcopy(x_obsts)]
    
    # Main loop
    for timestep in range(NUM_TIMESTEPS):
        robot_states = [[ob["robot_"+str(i)]["joint_state"]["position"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)],
                         ob["robot_"+str(i)]["joint_state"]["velocity"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)]]
                        for i in range(NUM_ROBOTS)]
        T_W_EEFs_current = [planner.compute_fk(robot_states[i][0]) for i in range(NUM_ROBOTS)]

        T_W_RedCup[:3,3], redcup_quat = env.get_redcup_pose()
        T_W_GrenCup[:3,3], greencup_quat = env.get_greencup_pose()
        T_W_Objects_robots = [T_W_RedCup, T_W_GrenCup]


        # Update collision spheres for GOMP & fabrics
        T_W_chassis_robots = [fabrics.compute_fk(robot_states[i][0], "chassis_link") for i in range(NUM_ROBOTS)]
        T_W_wrist_robots = [fabrics.compute_fk(robot_states[i][0], "arm_upper_wrist_link") for i in range(NUM_ROBOTS)]

        x_obsts_robots[0][-2] =  T_W_chassis_robots[1][:3,3].tolist()
        x_obsts_robots[0][-1] =  T_W_wrist_robots[1][:3,3].tolist()
        x_obsts_robots[1][-2] =  T_W_chassis_robots[0][:3,3].tolist()
        x_obsts_robots[1][-1] =  T_W_wrist_robots[0][:3,3].tolist()

        
        # GOMP
        if timestep%PLANNER_PERIOD == 0:
            for robot_id in range(NUM_ROBOTS):
                start_time = time.perf_counter()
                waypoint_list, solver_status_robots[robot_id] = planner.solve(joint_state=robot_states[robot_id], 
                                                                                T_W_Obj=T_W_Objects_robots[robot_id],
                                                                                x_obsts=x_obsts_robots[robot_id]
                                                                                )

                end_time = time.perf_counter()
                print(f"Solver status for robot {robot_id}: {solver_status_robots[robot_id]}")
                print(f'Computational time for robot {robot_id}: {end_time-start_time} s')                
                print()
                if solver_status_robots[robot_id] == True:
                    waypoints_list_robots[robot_id] = copy.deepcopy(waypoint_list)
                if RENDER:
                    if solver_status_robots[robot_id]:
                        for i in range(len(waypoints_list_robots[robot_id])):
                            pybullet.addUserDebugPoints([waypoints_list_robots[robot_id][i][:3, 3].tolist()], [[0, 1, 0]], 7, 1.0)

        if timestep%REFERENCE_TRACKER_PERIOD == 0:
            for robot_id in range(NUM_ROBOTS):
                # Reference tracker
                if np.linalg.norm(T_W_EEFs_current[robot_id][:3,3] - T_W_Objects_robots[robot_id][:3,3]) <= 0.2:
                    T_W_Goal_robots[robot_id] = waypoints_list_robots[robot_id][-1]
                else:
                    if waypoints_list_robots[robot_id] is None or len(waypoints_list_robots[robot_id]) == 0:
                        pass
                    else:
                        current_eef_pose = transformation2dict(T_W_EEFs_current[robot_id])
                        waypoint_dict = [transformation2dict(waypoints_list_robots[robot_id][i]) for i in range(len(waypoints_list_robots[robot_id]))]
                        current_goal_dict, waypoint_dict, flag = reference_tracker.update_local_goal_pos_orient(current_eef_pose["position"], waypoint_dict)
                        if current_goal_dict is not None:
                            T_W_Goal_robots[robot_id] = dict2transformation(current_goal_dict)
     
        
        for robot_id in range(NUM_ROBOTS):
            fabrics.update_arguments(joint_state= robot_states[robot_id],
                                    T_W_Goal=T_W_Goal_robots[robot_id],
                                    obst_pos=x_obsts_robots[robot_id],
                                    obst_radius=None)
            unclipped_action = fabrics.compute_action()
            action[(robot_id*NUM_DOF): NUM_DOF*robot_id + (NUM_DOF-NUM_GRIPPER_FINGERS)] = fabrics.clip_action(unclipped_action)

        ob, *_ = sim.step(action)
    sim.close()