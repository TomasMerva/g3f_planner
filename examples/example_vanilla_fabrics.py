import numpy as np
import copy
import yaml
import time
import os
from scipy.spatial.transform import Rotation as R
import pybullet
from typing import Dict
from tqdm import tqdm

from dinovas_pybullet_env import Environment
from fabrics_planner import Fabrics

from evaluation.record_data import RecordData

def draw_coordinate_frame(T, axis_length=0.1):
    # Extract origin and rotation matrix from T
    origin = T[:3, 3]
    R_matrix = T[:3, :3]

    # Define the colors for the x, y, z axes
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue

    # Define unit vectors along x, y, z axes
    axes = np.eye(3)

    # Draw each axis using pybullet's addUserDebugLine
    for i in range(3):
        # Calculate the end point of each axis in world coordinates
        end_point = origin + R_matrix @ (axes[:, i] * axis_length)
        pybullet.addUserDebugLine(origin, end_point, colors[i], lineWidth=3)

def run_dinova_example(n_steps, dof, n_robots, env:Environment, stopping_tolerance = 0.05, grasp_goals = None):
    NUM_ROBOTS = n_robots
    NUM_DOF = dof
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = n_steps
    sim = env.get_sim_handler()
    CONFIG_FILE_PATH = env.get_config_file_path()
    action = np.zeros(NUM_ROBOTS*NUM_DOF)
    ob, *_ = sim.step(action)
    obstacles = env.get_obstacles()

    # Fabrics
    fabrics = Fabrics(robot_urdf_path=env.ROBOT_URDF_FILE,
                      config_file_path=CONFIG_FILE_PATH,
                      degrees_of_freedom=NUM_DOF-NUM_GRIPPER_FINGERS)
    goals_loaded = False
    T_W_Goals, T_W_Objects = [], []
    if grasp_goals == None:
        for robot_id in range(NUM_ROBOTS):
            T_W_Object = np.eye(4)
            T_W_Object[:3,3], quat = env.get_cup(robot_id)
            T_W_Goals.append(T_W_Object)#position vector
            T_W_Objects.append(T_W_Object)
        print("T_W_Goals:", T_W_Goals)
    else:
        print("Load grasp pose from pikle file.")
        goals_loaded = True
        # Generate transformation matrix for each grasp position
        for robot_id, grasp in enumerate(grasp_goals):
            # Create a 4x4 identity matrix
            T = np.eye(4)
            # Set the translation part (last column, first three elements)
            T[:3, 3] = grasp["position"]
            rotation_matrix = R.from_quat(grasp["orientation"]).as_matrix()
            # Set the top-left 3x3 part of T to the rotation matrix
            T[:3, :3] = rotation_matrix
            # Append the transformation matrix to the list
            T_W_Goals.append(T)
            #Looks weird
            draw_coordinate_frame(T)
            env.reset_cup(robot_id, grasp["position"])
    x_obsts = [obstacles[i]["position"] for i in obstacles]
    r_obsts = [obstacles[i]["radius"] for i in obstacles]
    
    """
    Results metrics:
    """
    evaluation_data = RecordData()
    success_rate_per_robot = [0] * n_robots

    # Main loop
    print("Starting GF env")
    for timestep in tqdm(range(NUM_TIMESTEPS)):
        robot_states = [[ob["robot_"+str(i)]["joint_state"]["position"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)],
                         ob["robot_"+str(i)]["joint_state"]["velocity"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)]]
                        for i in range(NUM_ROBOTS)]
        
        # Compute robots' position for collision avoidance
        T_W_chassis_robots = [fabrics.compute_fk(robot_states[i][0], "chassis_link") for i in range(NUM_ROBOTS)]
        T_W_wrist_robots = [fabrics.compute_fk(robot_states[i][0], "arm_upper_wrist_link") for i in range(NUM_ROBOTS)]

        # Compute static grasps at the beginning
        if timestep == 0 and not goals_loaded:
            for robot_id in range(NUM_ROBOTS):
                theta = fabrics.get_theta_preference(q=robot_states[robot_id][0], 
                                                     goal_position=T_W_Goals[robot_id][:3,3])
                T_W_Goals[robot_id] = fabrics.compute_static_grasp(T_W_Goals[robot_id], theta)

        q_robots = [] # hold the robot configurations, for further reproduction
        goal_positions = [T[:3, 3] for T in T_W_Goals]  # Extracts the x, y, z position from each matrix
        goal_orientations = [R.from_matrix(T[:3, :3]).as_quat() for T in T_W_Goals]
        for robot_id in range(NUM_ROBOTS):
            # other robots as dynamic obstacles
            counter = 0
            for i in range(NUM_ROBOTS):
                if i == robot_id:
                    continue
                else:
                    chassis_idx = counter 
                    wrist_idx = counter + 1
                    x_obsts[chassis_idx] = T_W_chassis_robots[i][:3,3].tolist()
                    x_obsts[wrist_idx] = T_W_wrist_robots[i][:3,3].tolist()
                    counter += 2

            # Planner computes new action
            start_time = time.perf_counter()
            fabrics.compute_dynamic_weights(q= robot_states[robot_id][0],
                                            T_W_Goal=T_W_Goals[robot_id])
            fabrics.update_arguments(joint_state= robot_states[robot_id],
                                    T_W_Goal=T_W_Goals[robot_id],
                                    obst_pos=x_obsts,
                                    obst_radius=r_obsts)
            
            action_unclipped = fabrics.compute_action()
            action[(robot_id*NUM_DOF): NUM_DOF*robot_id + (NUM_DOF-NUM_GRIPPER_FINGERS)] = fabrics.clip_action(action_unclipped)
            end_time = time.perf_counter()

            # Log data
            evaluation_data.record_computational_time(end_time-start_time)
            if fabrics.error(goal_pos=T_W_Goals[robot_id][:3, 3], q_current=robot_states[robot_id][0]) <= stopping_tolerance:
                success_rate_per_robot[robot_id] = 1

            q_robots.append(robot_states[robot_id][0])# hold the robot configurations, for further reproduction


        collision_flag, collision_pairs, collision_pair_names = env.check_collisions()
        if collision_flag == True:
            evaluation_data.record_success_rate(success=0.0)
            evaluation_data.record_collision_violation(collision_flag)
            evaluation_data.record_robot_configurations_in_collisions(q_robots)
            evaluation_data.record_goal_positions(goal_positions)
            evaluation_data.record_goal_orientations(goal_orientations)
            evaluation_data.record_collision_pair(collision_pairs)
            evaluation_data.record_collision_pair_names(collision_pair_names)
            print("GF failed")
            break

        if np.all(success_rate_per_robot):
            evaluation_data.record_success_rate(success=100.0)
            evaluation_data.record_time_to_goal(timestep, sim._dt)
            evaluation_data.record_collision_violation(collision_flag=False)
            print("GF succeeded")
            break  
        
        ob, *_ = sim.step(action)
    if not np.all(success_rate_per_robot):
        evaluation_data.record_robot_configurations_fail_to_reach(q_robots)
        evaluation_data.record_goal_positions(goal_positions)
        evaluation_data.record_goal_orientations(goal_orientations)

    sim.close()
    return evaluation_data.get_result()

def main(render=True, timesteps=2000):
    RENDER = render
    NUM_ROBOTS = 2
    NUM_DOF = 11
    NUM_TIMESTEPS = timesteps


    # Environment
    env = Environment()
    env.initialize(render=RENDER, nr_robots=NUM_ROBOTS)
    run_dinova_example(n_steps=NUM_TIMESTEPS,
                       dof=NUM_DOF,
                       n_robots=NUM_ROBOTS,
                       env=env
                       )
    return {}

if __name__=="__main__":
    main()

   