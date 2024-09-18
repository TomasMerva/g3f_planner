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
from tqdm import tqdm


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



def run_dinova_example(n_steps, dof, n_robots, env:Environment, render=False):
    RENDER = render
    NUM_ROBOTS = n_robots
    NUM_DOF = dof
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = n_steps
    PLANNER_PERIOD = 100
    REFERENCE_TRACKER_PERIOD = 10


    sim = env.get_sim_handler()
    CONFIG_FILE_PATH = env.get_config_file_path()
    action = np.zeros(NUM_ROBOTS*NUM_DOF)
    ob, *_ = sim.step(action)
    obstacles = env.get_obstacles()
    robots_color = [ [0, 255, 0],
                     [0, 255, 255],
                     [0,0,255],
                     [255,255,0] ]

    
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


    T_W_Objects = []
    T_W_Goals = []
    for robot_id in range(NUM_ROBOTS):
        T_W_Object = np.eye(4)
        T_W_Object[:3,3], quat = env.get_cup(robot_id)
        T_W_Objects.append(T_W_Object)
        T_W_Goals.append(np.eye(4))

    waypoints_list_robots = [None]* NUM_ROBOTS
    solver_status_robots = [None] * NUM_ROBOTS

    
    x_obsts = [obstacles[i]["position"] for i in obstacles]
    # r_obsts = [obstacles[i]["radius"] for i in obstacles]
    # r_obsts = [0.0975, 0.2, 0.55, 0.2, 0.55, 0.2, 0.55, 0.2]
    r_obsts_fabrics = [obstacles[i]["radius"] for i in obstacles]
    # rollouts
    r_obsts = [0.0975, 0.1, 0.1, 0.1, 0.1, 0.1, 0.55, 0.2]
    r_coll_links = [0.65, 0.3, 0.3, 0.3, 0.1, 0.1]
    # x_obsts_robots = []
    # for robot_id in range(NUM_ROBOTS):
    #     x_obsts_robots.append(copy.deepcopy(x_obsts))


    """
    Results metrics:
    """
    results = {"collision":[], 
               "goal_reached": 0.,
               "time_to_goal":np.nan,
               "computation_time":[], 
               }
    success_rate_per_robot = [0] * n_robots
    
    # Main loop
    for timestep in tqdm(range(NUM_TIMESTEPS)):
        robot_states = [[ob["robot_"+str(i)]["joint_state"]["position"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)],
                         ob["robot_"+str(i)]["joint_state"]["velocity"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)]]
                        for i in range(NUM_ROBOTS)]
        T_W_EEFs_current = [planner.compute_fk(robot_states[i][0]) for i in range(NUM_ROBOTS)]

        # Update collision spheres for GOMP & fabrics
        T_W_chassis_robots = [fabrics.compute_fk(robot_states[i][0], "chassis_link") for i in range(NUM_ROBOTS)]
        T_W_wrist_robots = [fabrics.compute_fk(robot_states[i][0], "arm_upper_wrist_link") for i in range(NUM_ROBOTS)]

        
        for robot_id in range(NUM_ROBOTS):
            success_rate_per_robot[robot_id] = (fabrics.error(goal_pos=T_W_Goals[robot_id][:3, 3], q_current=robot_states[robot_id][0]) <= 0.08)
        if np.all(success_rate_per_robot):
            results["goal_reached"] = 1.
            results["time_to_goal"] = timestep * sim._dt
            break

        # GOMP
        if timestep%PLANNER_PERIOD == 0:
            for robot_id in range(NUM_ROBOTS):
                # other robots as dynamic obstacles
                num_obst2replace = -2*(n_robots-1)
                counter = 0
                for i in range(n_robots):
                    if i == robot_id:
                        continue
                    else:
                        chassis_idx = num_obst2replace + 2*counter
                        wrist_idx = num_obst2replace + 2*counter + 1
                        x_obsts[chassis_idx] = T_W_chassis_robots[i][:3,3].tolist()
                        x_obsts[wrist_idx] = T_W_wrist_robots[i][:3,3].tolist()
                        r_obsts[chassis_idx] = 0.55
                        r_obsts[wrist_idx] = 0.2
                        counter += 1

                start_time = time.perf_counter()
                waypoint_list, solver_status_robots[robot_id] = planner.solve(joint_state=robot_states[robot_id], 
                                                                              T_W_Obj=T_W_Objects[robot_id],
                                                                              x_obsts=x_obsts,
                                                                              
                                                                            )
                end_time = time.perf_counter()
                # Log data
                results["computation_time"].append(end_time-start_time)


                waypoints_list_robots[robot_id] = copy.deepcopy(waypoint_list)
                if RENDER:
                    for i in range(len(waypoints_list_robots[robot_id])):
                        pybullet.addUserDebugPoints([waypoints_list_robots[robot_id][i][:3, 3].tolist()], [robots_color[robot_id]], 10, 2.0)
                    # (init_coll, init_free) = planner.get_initial_guesses()
                    # for q in init_coll:
                    #     T_W_EEE_init = planner.compute_fk(q)
                    #     pybullet.addUserDebugPoints([T_W_EEE_init[:3, 3].tolist()], [[255,0,0]], 10, 1)

        if timestep%REFERENCE_TRACKER_PERIOD == 0:
            for robot_id in range(NUM_ROBOTS):
                if waypoints_list_robots[robot_id] is None or len(waypoints_list_robots[robot_id]) == 0:
                    continue
                else:
                    current_eef_pose = transformation2dict(T_W_EEFs_current[robot_id])
                    waypoint_dict = [transformation2dict(waypoints_list_robots[robot_id][i]) for i in range(len(waypoints_list_robots[robot_id]))]
                    current_goal_dict, waypoint_dict, flag = reference_tracker.update_local_goal_pos_orient(current_eef_pose["position"], waypoint_dict)
                    waypoints_list_robots[robot_id] = [dict2transformation(waypoint_dict[i]) for i in range(len(waypoint_dict))]
                    if current_goal_dict is not None:
                        T_W_Goals[robot_id] = dict2transformation(current_goal_dict)

     
        
        for robot_id in range(NUM_ROBOTS):
            # other robots as dynamic obstacles
            num_obst2replace = -2*(n_robots-1)
            counter = 0
            for i in range(n_robots):
                if i == robot_id:
                    continue
                else:
                    chassis_idx = num_obst2replace + 2*counter
                    wrist_idx = num_obst2replace + 2*counter + 1
                    x_obsts[chassis_idx] = T_W_chassis_robots[i][:3,3].tolist()
                    x_obsts[wrist_idx] = T_W_wrist_robots[i][:3,3].tolist()
                    r_obsts[chassis_idx] = 0.4
                    r_obsts[wrist_idx] = 0.1
                    counter += 1

            fabrics.update_arguments(joint_state= robot_states[robot_id],
                                     T_W_Goal=T_W_Goals[robot_id],
                                     obst_pos=x_obsts,
                                     obst_radius=r_obsts)
            action_unclipped = fabrics.compute_action()
            action[(robot_id*NUM_DOF): NUM_DOF*robot_id + (NUM_DOF-NUM_GRIPPER_FINGERS)] = fabrics.clip_action(action_unclipped)

        ob, *_ = sim.step(action)



    sim.close()

    print("The success-rate of the scenario is: ", results["goal_reached"], ", with a time-to-goal of: ", results["time_to_goal"], " sec.")
    print("The success-rate per robot is: ", success_rate_per_robot)

    return results


if __name__=="__main__":
    RENDER = True
    NUM_ROBOTS = 2
    NUM_DOF = 11
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = 2000


    # Environment
    env = Environment()
    env.initialize(render=RENDER, nr_robots=NUM_ROBOTS)
    run_dinova_example(n_steps=NUM_TIMESTEPS,
                       dof=NUM_DOF,
                       n_robots=NUM_ROBOTS,
                       env=env,
                       render=RENDER
                       )
                       

    