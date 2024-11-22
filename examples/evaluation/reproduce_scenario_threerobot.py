#!/usr/bin/env python3
"""Closed-loop upright simulation using Pybullet."""
import datetime
import os
import numpy as np
import pybullet as pyb
from pyb_utils.frame import debug_frame_world
import matplotlib.pyplot as plt
import yaml
from upright_core.logging import DataLogger, DataPlotter
import upright_sim as sim
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd
import pickle
import time
from dinova_multi_pybullet import MultiDinovas 
from comparison_study import link_dict_mpc
from comparison_study import Environment
from mobile_manipulation_central import BulletSimulation
from dinova_multi_pybullet import RecordData
import IPython
"""
python3 reproduce_scenario_threerobot.py --config1 $(rospack find comparison_study)/config/double_integrator/multi_3dinovas/multi_dinova1.yaml --config2 $(rospack find comparison_study)/config/double_integrator/multi_3dinovas/multi_dinova2.yaml --config3 $(rospack find comparison_study)/config/double_integrator/multi_3dinovas/multi_dinova3.yaml
"""

class MultiDinovasSim:
    def __init__(self):
        self.DINGO1_DYNAMIC_OBSTACLE_ENABLED = False
        self.DINGO2_DYNAMIC_OBSTACLE_ENABLED = False
        self.DINGO3_DYNAMIC_OBSTACLE_ENABLED = False
        self.stopping_tolerance = 0.07
        return
    def main(self, failure_data, n_robots=3, GUI=True):
        # --- for evaluation ----#
        evaluation_data = RecordData()
        success_rate_per_robot = [0] * n_robots
        collision_detected = False
        

        np.set_printoptions(precision=0, suppress=True)

        parser = cmd.cli.sim_arg_parser_multi_robots_3()
        cli_args = parser.parse_args()

        # load configuration for both robots
        config1 = core.parsing.load_config(cli_args.config1)
        sim_config = config1["simulation"]
        ctrl_config1 = config1["controller"]
        log_config = config1["logging"]
        file_storage_name = config1["file_storage_name"]
        if "dynamic" in ctrl_config1["obstacles"]:
            self.DINGO1_DYNAMIC_OBSTACLE_ENABLED = True
        
        config2 = core.parsing.load_config(cli_args.config2)
        ctrl_config2 = config2["controller"]
        if "dynamic" in ctrl_config2["obstacles"]:
            self.DINGO2_DYNAMIC_OBSTACLE_ENABLED = True
            
        config3 = core.parsing.load_config(cli_args.config3)
        ctrl_config3 = config3["controller"]
        if "dynamic" in ctrl_config3["obstacles"]:
            self.DINGO3_DYNAMIC_OBSTACLE_ENABLED = True

        timestep = sim_config["timestep"]
        duration = sim_config["duration"]
        
        urdf_robot_list = [config1["simulation"]["robot"]["urdf"], config2["simulation"]["robot"]["urdf"], config3["simulation"]["robot"]["urdf"]]#ONLY used for loading robots
        
        sim = BulletSimulation(timestep, (0,0,0), gui = GUI)
        simulated_robots = MultiDinovas(sim_config, urdf_robot_list, position=(0, 0, 0), n_robots=n_robots)
        
        env = Environment(config1, n_robots)
        env.draw_additional_static_obstacles(6)
        env.load_scene(n_robots)
        self._obst_id_list = env.get_obstacle_id_list()
        self.additional_obsts = env.get_additional_dynamic_obstacles()
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 0)  # Disable shadows if not needed
        debug_frame_ids = []
        visual_sphere_ids = []
        # Example: for each failure scenario, reset robots and visualize goals and configurations
        # print("failure_data",failure_data['collision'])
        # print("failure_data",failure_data['robot_configurations_in_collisions'])
        # print("failure_data",failure_data['robot_configurations_fail_to_reach'])
        for idx, (collision, configurations_in_collisions, robot_configurations_fail_to_reach, goal_position, goal_orientation, collision_pairs) in enumerate(
                zip(failure_data['collision'],
                    failure_data['robot_configurations_in_collisions'], 
                    failure_data['robot_configurations_fail_to_reach'], 
                    failure_data['goal_positions'], 
                    failure_data['goal_orientations'], 
                    failure_data['collision_pairs'])):
            #print("goal_position", goal_position)
            # Wait for user input to move to the next failure scenario
            
            # print("configurations_in_collisions:",configurations_in_collisions)
            # print("robot_configurations_fail_to_reach",robot_configurations_fail_to_reach)
            # Reset robot configurations based on the failure scenario
            for i, robot in enumerate(simulated_robots.dinovas):
                # q, v = robot.joint_states()
                # u = np.zeros(simulated_robots.nu)
                
                # v_cmd = np.zeros_like(v)
                
                debug_frame_ids.append(debug_frame_world(0.2, list(goal_position[i]), orientation=goal_orientation[i], line_width=3))
                visual_sphere_ids.append(env.add_debug_sphere(goal_position[i], i+1, radius = 0.03))
                if(collision):
                    print(configurations_in_collisions[i])
                    robot.reset_joint_configuration(configurations_in_collisions[i])
                    print("In collision", configurations_in_collisions[i])
                    print("Collision Pairs:", collision_pairs)
                else:
                    print(robot_configurations_fail_to_reach[i])
                    robot.reset_joint_configuration(robot_configurations_fail_to_reach[i])
                    print("fail to reach", robot_configurations_fail_to_reach[i])
                #reset_robot_configuration(robot, configurations[i])

            # # Visualize goal and robot configurations
            # draw_goal_and_configuration(goal_position[0], goal_orientation[0], configurations)

            # Simulation loop for a single failure scenario
            t = 0.0
        
            # simulation loop
            while t <= duration:
                
                #Get robot and obstacles states
                q1, v1 = simulated_robots.joint_states(uid=1, add_noise=False)
                
                q2, v2 = simulated_robots.joint_states(uid=2, add_noise=False)
                
                q3, v3 = simulated_robots.joint_states(uid=3, add_noise=False)
                
                
                # now get the noisy version for use in the controller
                # q_noisy1, v_noisy1 = simulated_robots.joint_states(uid=1,add_noise=True)
                # x_noisy1 = np.concatenate((q_noisy1, v_noisy1, x_obs1))
                # q_noisy2, v_noisy2 = simulated_robots.joint_states(uid=2,add_noise=True)
                # x_noisy2 = np.concatenate((q_noisy2, v_noisy2, x_obs2))
                # q_noisy3, v_noisy3 = simulated_robots.joint_states(uid=3,add_noise=True)
                # x_noisy3 = np.concatenate((q_noisy3, v_noisy3, x_obs3))

                v_cmd1 = np.zeros_like(v1)
                v_cmd2 = np.zeros_like(v2)
                v_cmd3 = np.zeros_like(v3)
            
                # generated velocity is in the world frame
                simulated_robots.command_velocity(1, v_cmd1, bodyframe=False)
                simulated_robots.command_velocity(2, v_cmd2, bodyframe=False)
                simulated_robots.command_velocity(3, v_cmd3, bodyframe=False)

                # log data:
                    # Log data:
                    # evaluation_data.record_computational_time((time.perf_counter() - start_time) / n_robots)

                    # Detect collisions
                for robot_id in simulated_robots.dinovas_uid:
                    contacts = pyb.getContactPoints(robot_id)
                    for contact_info in contacts:
                        if contact_info[2] in self._obst_id_list or contact_info[2] in simulated_robots.dinovas_uid:
                            if contact_info[2] != robot_id:
                                print("Collision Detected!")
                                # evaluation_data.record_collision_violation(100)
                                # evaluation_data.record_success_rate(success=0)
                                print("Collision Pairs:", robot_id, contact_info[2])

                    pyb.stepSimulation()
                    t = t + timestep
                # After finishing the simulation loop for this scenario, wait for user input to continue to the next one
            input(f"Scenario {idx + 1} finished. Press Enter to move to the next scenario...")
            for debug_tuple in debug_frame_ids:
                for id in debug_tuple:
                    pyb.removeUserDebugItem(id)
            for sphere_id in visual_sphere_ids:
                pyb.removeBody(sphere_id)
        pyb.disconnect()
        return {}

def load_failure_data():
    # Load the failure scenario data (pickle or other data sources)
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Ensure it works correctly when running directly
    #file_path = os.path.join(current_directory, 'result1311', 'dinovas_MPC_results_dinovas_two_tables_env_triple_integrator.pkl')
    # file_path = os.path.join(current_directory, 'result1311', 'dinovas_MPC_results_dinovas_threerobots_env_triple_all_dynamic.pkl')
    file_path = os.path.join(current_directory, 'result1311', 'dinovas_MPC_results_dinovas_threerobots_env_double_integrator_new.pkl')
    file_path = os.path.join(current_directory, 'result1311', 'dinovas_MPC_results_dinovas_threerobots_env_triple_integrator.pkl')
    file_path = os.path.join(current_directory, 'result1311', 'dinovas_MPC_results_dinovas_threerobots_env_triple_all_dynamic.pkl')
    
    file_path = os.path.join(current_directory, 'result1911', 'dinovas_threerobots_results.pickle')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    # print(data)
    collision_list = []
    collision_pairs_list = []
    collision_pair_names_list = []
    robot_configurations_in_collisions_list = []
    robot_configurations_fail_to_reach_list = []
    goal_positions_list = []
    goal_orientations_list = []

    for scenario in data:
        eds = scenario.get("MPC")
        if eds is not None:
            goal_reached = eds.goal_reached
            collision = eds.collision
            collision_pair = eds.collision_pair
            # collision_pair_names = eds.collision_pair_names
            robot_configurations_in_collisions = eds.robot_configurations_in_collisions
            robot_configurations_fail_to_reach = eds.robot_configurations_fail_to_reach
            goal_position = eds.goal_positions
            goal_orientation = eds.goal_orientations

            if goal_reached == 0:  # Robot failed to reach the goal
                if collision == 0:  # No collision, just didn't reach goal
                    robot_configurations_fail_to_reach_list.append(robot_configurations_fail_to_reach)
                    goal_positions_list.append(goal_position)
                    goal_orientations_list.append(goal_orientation)
                    collision_pairs_list.append([])
                    robot_configurations_in_collisions_list.append([])
                    collision_list.append([])
                else:  # Collision occurred
                    robot_configurations_fail_to_reach_list.append([])
                    collision_pairs_list.append(collision_pair)
                    robot_configurations_in_collisions_list.append(robot_configurations_in_collisions)
                    goal_positions_list.append(goal_position)
                    goal_orientations_list.append(goal_orientation)
                    collision_list.append(collision)
    # print(goal_orientations_list)
    # print(len(robot_configurations_in_collisions_list))
    # print(len(robot_configurations_fail_to_reach_list))
    return {
        'collision_pairs': collision_pairs_list,
        'collision_pair_names': collision_pair_names_list,
        'collision': collision_list,
        'robot_configurations_in_collisions': robot_configurations_in_collisions_list,
        'robot_configurations_fail_to_reach': robot_configurations_fail_to_reach_list,
        'goal_positions': goal_positions_list,
        'goal_orientations': goal_orientations_list
    }
if __name__ == "__main__":
    failure_data = load_failure_data()
    multi_dinovas = MultiDinovasSim()
    #multi_dinovas.main(robot_configurations_fail_to_reach, goal_positions, goal_orientations)
    multi_dinovas.main(failure_data)

# Texttable Latex:
# \begin{table}
#         \begin{center}
#                 \begin{tabular}{c|c|c|c|c}
#                           & Success rate [\%] & Time-to-Success [s] & Computation time[s] & Collision-rate \\
#                         \hline
#                         MPC & 60.0 \% & 19.5892 $\pm$ 4.9365 & 0.099271 $\pm$ 0.005622 & 25 \\
#                 \end{tabular}
#         \end{center}
# \end{table}
# Failed scenario indices for case 'MPC': [2, 5, 6, 7, 8, 10, 13, 16] 5,6, 16,might? or deadlock 7 might fined tuned, 13, 
# Collision scenario indices for case 'MPC': [5, 6, 7, 13, 16]
# Collision Pairs {'MPC': [[3, 11], [1, 11], [1, 10], [3, 11], [1, 11]]}
#diag: [0,0,0, 200, 0, 0, 0, 0, 0, 400, 400, 200, 300, 700, 400, 300rep3] #penalize joint5 is not a good idea because it hinders the goal reaching


#generate static
# Failed scenario indices for case 'MPC': [2, 5, 6, 7, 8, 10, 13, 15, 16]
# Collision scenario indices for case 'MPC': [6, 7, 10, 13]
# Collision Pairs {'MPC': [[1, 11], [1, 10], [1, 11], [1, 11]]}