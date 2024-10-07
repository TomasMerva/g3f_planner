import numpy as np
import copy
import yaml
import time
import os
from scipy.spatial.transform import Rotation as R
import pybullet
from typing import Dict
from fabrics_rollouts import DeadlockPrevention
from dinovas_pybullet_env import Environment
from rgf_planner import RGF_Planner
from fabrics_planner import Fabrics
from fabrics_rollouts import ReferenceTracker
from tqdm import tqdm

from evaluation.record_data import RecordData

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






def run_dinova_example(n_steps, dof, n_robots, env:Environment, render=False, stopping_tolerance=0.05):
    RENDER = render
    NUM_ROBOTS = n_robots
    NUM_DOF = dof
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = n_steps
    PLANNER_PERIOD = 100
    REFERENCE_TRACKER_PERIOD = 10

    robots_color = [ [0, 255, 0],
                    [0, 255, 255],
                    [0,0,255],
                    [255,255,0] ]


    sim = env.get_sim_handler()
    CONFIG_FILE_PATH = env.get_config_file_path()
    action = np.zeros(NUM_ROBOTS*NUM_DOF)
    ob, *_ = sim.step(action)

    # Deadlock resolution
    deadlock_prevention = DeadlockPrevention(dof=[NUM_DOF] * NUM_ROBOTS, n_robots=NUM_ROBOTS)
    qdot_rollout_avg = {f"robot_{i}": [] for i in range(n_robots)}
   
    #Modify obstacles based on number of robots
    obstacles = env.get_obstacles()
    x_obsts = [obstacles[i]["position"] for i in obstacles]
    r_obsts = [obstacles[i]["radius"] for i in obstacles]
    num_obst2replace = -2*(n_robots-1)
    for i in range(n_robots-1):
        chassis_idx = num_obst2replace + 2*i
        wrist_idx = num_obst2replace + 2*i + 1
        r_obsts[chassis_idx] = 0.6
        r_obsts[wrist_idx] = 0.2
    x_r_obsts_robots = {f"robot_{i}": {"x_obsts": x_obsts, "r_obsts": r_obsts} for i in range(n_robots)}
    arguments_dicts = {f"robot_{i}": [] for i in range(n_robots)}
    
    # Planner
    fk_args = dict(
        urdf_file = env.ROBOT_URDF_FILE,
        root_link = "world",
        end_link = "arm_tool_frame",
        num_dofs = NUM_DOF-NUM_GRIPPER_FINGERS,
    )
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_script_dir, '..', 'config/dinova_config_rgf_3obst.yaml')
    CONFIG_FILE_PATH_GOMP = os.path.normpath(config_path)
    planner = RGF_Planner(fk_args=fk_args,
                          config_file_path=CONFIG_FILE_PATH_GOMP)
    
    
    # Fabrics
    fabrics = Fabrics(robot_urdf_path=env.ROBOT_URDF_FILE,
                      config_file_path=CONFIG_FILE_PATH,
                      degrees_of_freedom=NUM_DOF-NUM_GRIPPER_FINGERS)
    
    # Reference
    reference_tracker = ReferenceTracker(tolerance_gripper=0.2, ub=1.0, lb=0.2)
    waypoints_list_robots = [None]* NUM_ROBOTS
    solver_status_robots = [None] * NUM_ROBOTS  

    T_W_Goals, T_W_Objects = [], []
    for robot_id in range(NUM_ROBOTS):
        T_W_Object = np.eye(4)
        T_W_Object[:3,3], quat = env.get_cup(robot_id)
        T_W_Objects.append(T_W_Object)
        T_W_Goals.append(T_W_Object)    


    """
    Results metrics:
    """
    evaluation_data = RecordData()
    success_rate_per_robot = [0] * n_robots

    
    # Main loop
    print("Starting RGF env")
    for timestep in tqdm(range(NUM_TIMESTEPS)):
        robot_states = [[ob["robot_"+str(i)]["joint_state"]["position"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)],
                         ob["robot_"+str(i)]["joint_state"]["velocity"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)]]
                        for i in range(NUM_ROBOTS)]
        T_W_EEFs_current = [planner.compute_fk(robot_states[i][0]) for i in range(NUM_ROBOTS)]

        # Update collision spheres for GOMP & fabrics
        T_W_chassis_robots = [fabrics.compute_fk(robot_states[i][0], "chassis_link") for i in range(NUM_ROBOTS)]
        T_W_wrist_robots = [fabrics.compute_fk(robot_states[i][0], "arm_upper_wrist_link") for i in range(NUM_ROBOTS)]


        # GOMP
        for robot_id in range(NUM_ROBOTS):
            if timestep%PLANNER_PERIOD == 0:
                if success_rate_per_robot[robot_id] == 0:
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
                            r_obsts[chassis_idx] = 0.7
                            r_obsts[wrist_idx] = 0.3
                            counter += 1
               
                    start_time = time.perf_counter()
                    waypoint_list, solver_status_robots[robot_id]= planner.solve(joint_state=robot_states[robot_id],
                                                                                T_W_Obj=T_W_Objects[robot_id],
                                                                                x_obsts=x_obsts[:1] + x_obsts[-2*(n_robots-1):],
                                                                                )
                    qdot_rollout_avg["robot_" + str(robot_id)] = planner.get_velocity_average()
                    end_time = time.perf_counter()
                    # Log data
                    evaluation_data.record_computational_time(end_time-start_time)

                    if timestep == 0:
                        waypoints_list_robots[robot_id] = copy.deepcopy(waypoint_list)
                    elif solver_status_robots[robot_id]:
                        if solver_status_robots[robot_id]:
                            waypoints_list_robots[robot_id] = copy.deepcopy(waypoint_list)
                    if RENDER:
                        for i in range(len(waypoints_list_robots[robot_id])):
                            pybullet.addUserDebugPoints([waypoints_list_robots[robot_id][i][:3, 3].tolist()], [robots_color[robot_id]], 10, 2.0)

            # if timestep%REFERENCE_TRACKER_PERIOD == 0:
            if success_rate_per_robot[robot_id] == 0:
                if waypoints_list_robots[robot_id] is None or len(waypoints_list_robots[robot_id]) == 0:
                    continue
                else:
                    current_eef_pose = transformation2dict(T_W_EEFs_current[robot_id])
                    waypoint_dict = [transformation2dict(waypoints_list_robots[robot_id][i]) for i in range(len(waypoints_list_robots[robot_id]))]
                    current_goal_dict, waypoint_dict, flag = reference_tracker.update_local_goal_pos_orient(current_eef_pose["position"], waypoint_dict)
                    waypoints_list_robots[robot_id] = [dict2transformation(waypoint_dict[i]) for i in range(len(waypoint_dict))]
                    if current_goal_dict is not None:
                        T_W_Goals[robot_id] = dict2transformation(current_goal_dict)

            
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
                    r_obsts[chassis_idx] = 0.45
                    r_obsts[wrist_idx] = 0.2
                    counter += 1

            fabrics.compute_dynamic_weights(q= robot_states[robot_id][0],
                                            T_W_Goal=T_W_Goals[robot_id])
            fabrics.update_arguments(joint_state= robot_states[robot_id],
                                     T_W_Goal=T_W_Goals[robot_id],
                                     obst_pos=x_obsts,
                                     obst_radius=r_obsts)
            arguments_dicts["robot_" + str(robot_id)] = fabrics.get_arguments()

        # need information of GOMP for both robots for deadlock resolution:
        deadlock_prevention.deadlock_checking(x_robots=[T_W_EEFs_current[robot_id][:3, 3] for robot_id in range(NUM_ROBOTS)],
                                              goals_final=[T_W_Goals[robot_id][:3,3] for robot_id in range(NUM_ROBOTS)],
                                              time_step=timestep,
                                              avg_sum=sum(qdot_rollout_avg.values()) / NUM_ROBOTS)

        for robot_id in range(NUM_ROBOTS):
            goal_robots, goal_weights = deadlock_prevention.deadlock_adapt_goals_weights(x_robots=T_W_EEFs_current,
                                                                                         goal_robots=[arguments_dicts["robot_" + str(j_robot)]["x_goal_0"] for j_robot in range(NUM_ROBOTS)],
                                                                                         goal_weights=[arguments_dicts["robot_" + str(j_robot)]["weight_goal_0"]for j_robot in range(NUM_ROBOTS)])
            arguments_dicts["robot_" + str(robot_id)]["x_goal_0"] = goal_robots[robot_id]
            arguments_dicts["robot_" + str(robot_id)]["weight_goal_0"] = goal_weights[robot_id]
            fabrics.set_arguments(arguments_dicts["robot_" + str(robot_id)])
            action_unclipped = fabrics.compute_action()
            action[(robot_id*NUM_DOF): NUM_DOF*robot_id + (NUM_DOF-NUM_GRIPPER_FINGERS)] = fabrics.clip_action(action_unclipped)
                    
            if planner.error(goal_pos=planner._T_W_StaticGrasp[:3, 3], q_current=robot_states[robot_id][0]) <= stopping_tolerance:
                success_rate_per_robot[robot_id] = 1
                print(f"Robot {robot_id} has finished")
            x_r_obsts_robots["robot_"+str(robot_id)]["x_obsts"] =  x_obsts
            x_r_obsts_robots["robot_"+str(robot_id)]["r_obsts"] = r_obsts

            if timestep%100 == 0:
                position = T_W_Goals[robot_id][:3, 3]  
                rotation_matrix = T_W_Goals[robot_id][:3, :3]
                axis_length = 0.2
                pybullet.addUserDebugLine(position, position + rotation_matrix[:, 0] * axis_length, [1, 0, 0], lineWidth=3, lifeTime=1.0)
                pybullet.addUserDebugLine(position, position + rotation_matrix[:, 1] * axis_length, [0, 1, 0], lineWidth=3, lifeTime=1.0)
                pybullet.addUserDebugLine(position, position + rotation_matrix[:, 2] * axis_length, [0, 0, 1], lineWidth=3, lifeTime=1.0)

 

        if np.all(success_rate_per_robot):
            evaluation_data.record_success_rate(success=100.0)
            evaluation_data.record_time_to_goal(timestep, sim._dt)
            print("RGF: Success")
            break

        evaluation_data.record_collision_violation(fabrics.collision_check(x_r_obsts_robots, robot_states[0], threshold=0.))

        ob, *_ = sim.step(action)

    sim.close()

    return evaluation_data.get_result()

def main(render=True, timesteps=2000):
    RENDER = render
    NUM_ROBOTS = 2
    NUM_DOF = 11
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = timesteps

    # Environment
    env = Environment()
    env.initialize(render=RENDER, nr_robots=NUM_ROBOTS)
    run_dinova_example(n_steps=NUM_TIMESTEPS,
                       dof=NUM_DOF,
                       n_robots=NUM_ROBOTS,
                       env=env,
                       render=RENDER
                       )
    return {}

if __name__=="__main__":
    main()

    