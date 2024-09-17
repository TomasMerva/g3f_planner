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
from fabrics_planner import Fabrics
from fabrics_rollouts import RolloutFabrics

def compute_static_grasp(T_W_Obj, theta_preference):
    T_Obj_Grasp = np.eye(4)
    T_Obj_Grasp[:3,:3] = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
    T_Grasp_Theta = np.eye(4)
    T_Grasp_Theta[:3,:3] = R.from_euler('xyz', [-theta_preference, 0, 0], degrees=False).as_matrix()
    T_W_Grasp = T_W_Obj @ T_Obj_Grasp @ T_Grasp_Theta

    T_Grasp_Offset = np.eye(4)
    T_Grasp_Offset[:3, 3] = [-0, 0, -0.05]
    return T_W_Grasp @ T_Grasp_Offset


def run_dinova_example(n_steps, dof, n_robots, env:Environment):
    NUM_ROBOTS = n_robots
    NUM_DOF = dof
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = n_steps
    nr_rollout_timesteps = 200

    sim = env.get_sim_handler()
    CONFIG_FILE_PATH = env.get_config_file_path()
    action = np.zeros(NUM_ROBOTS*NUM_DOF)
    ob, *_ = sim.step(action)
    obstacles = env.get_obstacles()

    # Fabrics
    fabrics = Fabrics(robot_urdf_path=env.ROBOT_URDF_FILE,
                      config_file_path=CONFIG_FILE_PATH,
                      degrees_of_freedom=NUM_DOF-NUM_GRIPPER_FINGERS)

    # Deadlock resolution
    deadlock_prevention = DeadlockPrevention(dof=[NUM_DOF]*NUM_ROBOTS, n_robots=NUM_ROBOTS)
    qdot_rollout_avg = {f"robot_{i}": [] for i in range(n_robots)}
    arguments_dicts = {f"robot_{i}": [] for i in range(n_robots)}

    # Rollout planner
    fk_args = dict(
    urdf_file = env.ROBOT_URDF_FILE,
    root_link = "world",
    end_link = "arm_tool_frame"
    )
    dinova_vel_limits = np.asarray(env.CONFIG_PROBLEM["joint_limits"]["velocity"], dtype=np.float32)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    fabrics_lib = os.path.normpath(os.path.join(current_script_dir, '../', 'build', 'libfabrics_controller.so'))
    rollouts_planner = RolloutFabrics(fk_dict=fk_args,
                                      config_file=env.CONFIG_FILE,
                                      controller_lib=fabrics_lib,
                                      vel_limits=dinova_vel_limits,
                                      dt=0.1,
                                      alpha_filter=0.7)
    
    T_W_Goals = []
    for robot_id in range(NUM_ROBOTS):
        T_W_Object = np.eye(4)
        T_W_Object[:3,3], quat = env.get_cup(robot_id)
        T_W_Goals.append(T_W_Object)

    # Static obstacles #TODO: change to num_robots
    x_obsts = [obstacles[i]["position"] for i in obstacles]
    r_obsts = [obstacles[i]["radius"] for i in obstacles]
    x_r_obsts_robots = {f"robot_{i}": {"x_obsts": x_obsts, "r_obsts": r_obsts} for i in range(n_robots)}
    
    """
    Results metrics:
    """
    results = {"collision": 0.,
               "goal_reached": 0.,
               "time_to_goal":np.nan,
               "computation_time":[], 
               }
    success_rate_per_robot = [0] * n_robots

    # Main loop
    for timestep in range(NUM_TIMESTEPS):
        robot_states = [[ob["robot_"+str(i)]["joint_state"]["position"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)],
                         ob["robot_"+str(i)]["joint_state"]["velocity"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)]]
                        for i in range(NUM_ROBOTS)]
        
        # Compute robots' position for collision avoidance
        T_W_chassis_robots = [fabrics.compute_fk(robot_states[i][0], "chassis_link") for i in range(NUM_ROBOTS)]
        T_W_wrist_robots = [fabrics.compute_fk(robot_states[i][0], "arm_upper_wrist_link") for i in range(NUM_ROBOTS)]
        x_ee_robots = [fabrics.compute_fk(robot_states[i][0], "arm_tool_frame")[0:3, 3] for i in range(NUM_ROBOTS)]

        if timestep == 0:
            for robot_id in range(NUM_ROBOTS):
                theta = fabrics.get_theta_preference(q=robot_states[robot_id][0], 
                                                     goal_position=T_W_Goals[robot_id][:3,3])
                T_W_Goals[robot_id] = compute_static_grasp(T_W_Goals[robot_id], theta)

        for robot_id in range(NUM_ROBOTS):
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
            fabrics.update_arguments(joint_state= robot_states[robot_id],
                                    T_W_Goal=T_W_Goals[robot_id],
                                    obst_pos=x_obsts,
                                    obst_radius=r_obsts)
            arguments_dicts["robot_"+str(robot_id)] = fabrics.get_arguments()

            if timestep % 100 == 0:
                q_rollout = rollouts_planner.compute_rollout(timesteps=nr_rollout_timesteps,
                                                             arg_dict=fabrics._arguments_dict,
                                                             tolerance=0.15)
                qdot_rollout_avg["robot_" + str(robot_id)] = rollouts_planner.get_rollout_velocity_avg()

            action_unclipped = fabrics.compute_action()
            action[(robot_id*NUM_DOF): NUM_DOF*robot_id + (NUM_DOF-NUM_GRIPPER_FINGERS)] = fabrics.clip_action(action_unclipped)
            end_time = time.perf_counter()

            results["computation_time"].append(end_time-start_time)
            x_r_obsts_robots["robot_"+str(robot_id)]["x_obsts"] =  x_obsts
            x_r_obsts_robots["robot_"+str(robot_id)]["r_obsts"] = r_obsts

        deadlock_prevention.deadlock_checking(x_robots=x_ee_robots,
                                              goals_final=[T_W_Goals[robot_id][:3,3] for robot_id in range(NUM_ROBOTS)],
                                              time_step=timestep,
                                              avg_sum=sum(qdot_rollout_avg.values()) / NUM_ROBOTS)

        for robot_id in range(NUM_ROBOTS):
            goal_robots, goal_weights = deadlock_prevention.deadlock_adapt_goals_weights(x_robots=x_ee_robots,
                                                                                         goal_robots=[arguments_dicts["robot_" + str(j_robot)]["x_goal_0"] for j_robot in range(NUM_ROBOTS)],
                                                                                         goal_weights=[arguments_dicts["robot_" + str(j_robot)]["weight_goal_0"]for j_robot in range(NUM_ROBOTS)])
            arguments_dicts["robot_" + str(robot_id)]["x_goal_0"] = goal_robots[robot_id]
            arguments_dicts["robot_" + str(robot_id)]["weight_goal_0"] = goal_weights[robot_id]
            fabrics.set_arguments(arguments_dicts["robot_" + str(robot_id)])
            action_unclipped = fabrics.compute_action()
            action[(robot_id*NUM_DOF): NUM_DOF*robot_id + (NUM_DOF-NUM_GRIPPER_FINGERS)] = fabrics.clip_action(action_unclipped)

        ob, *_ = sim.step(action)

        for robot_id in range(NUM_ROBOTS):
            success_rate_per_robot[robot_id] = (fabrics.error(goal_pos=T_W_Goals[robot_id][:3, 3], q_current=robot_states[robot_id][0]) <= 0.08)
        if np.all(success_rate_per_robot):
            results["goal_reached"] = 1.
            results["time_to_goal"] = timestep * sim._dt
            break

        results["collision"] = fabrics.collision_check(x_r_obsts_robots, robot_states[0], threshold=0.)

    sim.close()

    print("The success-rate of the scenario is: ", results["goal_reached"], ", with a time-to-goal of: ", results["time_to_goal"], " sec.")
    print("The success-rate per robot is: ", success_rate_per_robot)
    print("Has a collision occurred?: ", results["collision"])

    return results


if __name__=="__main__":
    RENDER = True
    NUM_ROBOTS = 2
    NUM_DOF = 11
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = 2500
    PLANNER_FREQ = 10

    # Environment
    env = Environment()
    (sim, goal) = env.initialize(render=RENDER, nr_robots=NUM_ROBOTS)
    run_dinova_example(n_steps=NUM_TIMESTEPS,
                       dof=NUM_DOF,
                       n_robots=NUM_ROBOTS,
                       env=env
                       )