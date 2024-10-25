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



def run_dinova_example(n_steps, dof, n_robots, env:Environment, stopping_tolerance = 0.05):
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
    
    T_W_Goals, T_W_Objects = [], []
    for robot_id in range(NUM_ROBOTS):
        T_W_Object = np.eye(4)
        T_W_Object[:3,3], quat = env.get_cup(robot_id)
        T_W_Goals.append(T_W_Object)
        T_W_Objects.append(T_W_Object)

    x_obsts = [obstacles[i]["position"] for i in obstacles]
    r_obsts = [obstacles[i]["radius"] for i in obstacles]
    x_r_obsts_robots = {f"robot_{i}": {"x_obsts": x_obsts, "r_obsts": r_obsts} for i in range(n_robots)}
    
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
        if timestep == 0:
            for robot_id in range(NUM_ROBOTS):
                theta = fabrics.get_theta_preference(q=robot_states[robot_id][0], 
                                                     goal_position=T_W_Goals[robot_id][:3,3])
                T_W_Goals[robot_id] = fabrics.compute_static_grasp(T_W_Goals[robot_id], theta)

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
            x_r_obsts_robots["robot_" + str(robot_id)]["x_obsts"] = copy.deepcopy(x_obsts)
            x_r_obsts_robots["robot_" + str(robot_id)]["r_obsts"] = r_obsts

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

            # if timestep%100 == 0:
            #     position = T_W_Goals[robot_id][:3, 3]  
            #     rotation_matrix = T_W_Goals[robot_id][:3, :3]
            #     axis_length = 0.2
            #     pybullet.addUserDebugLine(position, position + rotation_matrix[:, 0] * axis_length, [1, 0, 0], lineWidth=3, lifeTime=1.0)
            #     pybullet.addUserDebugLine(position, position + rotation_matrix[:, 1] * axis_length, [0, 1, 0], lineWidth=3, lifeTime=1.0)
            #     pybullet.addUserDebugLine(position, position + rotation_matrix[:, 2] * axis_length, [0, 0, 1], lineWidth=3, lifeTime=1.0)

        ob, *_ = sim.step(action)

        collision_flag = fabrics.collision_check(x_r_obsts_robots, robot_states, threshold=-0.05)
        evaluation_data.record_collision_violation(collision_flag)
        if collision_flag == True:
            evaluation_data.record_success_rate(success=0.0)
            break

        if np.all(success_rate_per_robot):
            evaluation_data.record_success_rate(success=100.0)
            evaluation_data.record_time_to_goal(timestep, sim._dt)
            break

        # evaluation_data.record_collision_violation(fabrics.collision_check(x_r_obsts_robots, robot_states, threshold=-0.05))

            


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
                       env=env
                       )
    return {}

if __name__=="__main__":
    main()

   