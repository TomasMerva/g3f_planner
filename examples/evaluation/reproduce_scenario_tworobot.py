#!/usr/bin/env python3
"""Closed-loop upright simulation using Pybullet."""
import datetime
import os
import numpy as np
import pybullet as pyb
# from pyb_utils.frame import debug_frame_world
# import matplotlib.pyplot as plt
import yaml
# from pyb_utils.frame import debug_frame_world
# from upright_core.logging import DataLogger, DataPlotter
# import upright_sim as sim
# import upright_core as core
# import upright_control as ctrl
# import upright_cmd as cmd
import pickle
import time
import sys
# from dinova_multi_pybullet import MultiDinovas 
# from comparison_study import link_dict_mpc
# from comparison_study import Environment
# from mobile_manipulation_central import BulletSimulation
# from dinova_multi_pybullet import RecordData
#import IPython
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.record_data import RecordData
"""
python3 reproduce_scenario_tworobot.py --config1 $(rospack find comparison_study)/config/double_integrator/multi_3dinovas/multi_dinova1.yaml --config2 $(rospack find comparison_study)/config/double_integrator/multi_3dinovas/multi_dinova2.yaml
"""

class MultiDinovasSim:
    def __init__(self):
        self.DINGO1_DYNAMIC_OBSTACLE_ENABLED = False
        self.DINGO2_DYNAMIC_OBSTACLE_ENABLED = False
        # self.DINGO3_DYNAMIC_OBSTACLE_ENABLED = False
        self.stopping_tolerance = 0.07
        return
    def main(self, failure_data, n_robots=2, GUI=True):
        np.set_printoptions(precision=0, suppress=True)

        parser = cmd.cli.sim_arg_parser_multi_robots()
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
            
        # config3 = core.parsing.load_config(cli_args.config3)
        # ctrl_config3 = config3["controller"]
        # if "dynamic" in ctrl_config3["obstacles"]:
        #     self.DINGO3_DYNAMIC_OBSTACLE_ENABLED = True

        timestep = sim_config["timestep"]
        duration = 1#sim_config["duration"]
        
        urdf_robot_list = [config1["simulation"]["robot"]["urdf"], config2["simulation"]["robot"]["urdf"]]#ONLY used for loading robots
        
        sim = BulletSimulation(timestep, (0,0,0), gui = GUI)
        simulated_robots = MultiDinovas(sim_config, urdf_robot_list, position=(0, 0, 0), n_robots=n_robots)
        
        env = Environment(config1, n_robots)
        env.draw_additional_static_obstacles(4)
        env.load_scene(n_robots)
        self._obst_id_list = env.get_obstacle_id_list()
        self.additional_obsts = env.get_additional_dynamic_obstacles()

        debug_frame_ids = []
        visual_sphere_ids = []
        # Example: for each failure scenario, reset robots and visualize goals and configurations
        for idx, (collision, configurations_in_collisions, robot_configurations_fail_to_reach, goal_position, goal_orientation, collision_pairs) in enumerate(
                zip(failure_data['collision'],
                    failure_data['robot_configurations_in_collisions'], 
                    failure_data['robot_configurations_fail_to_reach'], 
                    failure_data['goal_positions'], 
                    failure_data['goal_orientations'], 
                    failure_data['collision_pairs'])):
            #print("goal_position", goal_position)
            # Wait for user input to move to the next failure scenario
            
            # Reset robot configurations based on the failure scenario
            for i, robot in enumerate(simulated_robots.dinovas):
                # q, v = robot.joint_states()
                # u = np.zeros(simulated_robots.nu)
                
                # v_cmd = np.zeros_like(v)
                
                debug_frame_ids.append(debug_frame_world(0.2, list(goal_position[i]), orientation=goal_orientation[i], line_width=3))
                visual_sphere_ids.append(env.add_debug_sphere(goal_position[i], i+1, radius = 0.03))
                if(collision):
                    robot.reset_joint_configuration(configurations_in_collisions[i])
                    print("In collision", configurations_in_collisions[i])
                    print("Collision Pairs:", collision_pairs)
                else:
                    robot.reset_joint_configuration(robot_configurations_fail_to_reach[i])
                    print("fail to reach", robot_configurations_fail_to_reach[i])
                #reset_robot_configuration(robot, configurations[i])

            # # Visualize goal and robot configurations
            # draw_goal_and_configuration(goal_position[0], goal_orientation[0], configurations)

            # Simulation loop for a single failure scenario
            t = 0.0
            while t <= duration:
                q1, v1 = simulated_robots.joint_states(uid=1, add_noise=False)
                q2, v2 = simulated_robots.joint_states(uid=2, add_noise=False)

                v_cmd1 = np.zeros_like(v1)
                v_cmd2 = np.zeros_like(v2)
                # Generate velocity commands for robots
                simulated_robots.command_velocity(1, v_cmd1, bodyframe=False)
                simulated_robots.command_velocity(2, v_cmd2, bodyframe=False)

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
    # file_path = os.path.join(current_directory, 'result1311', 'dinovas_MPC_results_dinovas_two_tables_env_triple_integrator.pkl')
    file_path = os.path.join(current_directory, 'datasets', 'dinovas_MPC_results_dinovas_two_tables_cross_env_double_integrator_17th.pkl')
    file_path = os.path.join(current_directory, 'results', 'dinovas_two_tables_cross_results.pickle')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        #print(data)

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
            robot_configurations_in_collisions = eds.robot_configurations_in_collisions[0:9]
            robot_configurations_fail_to_reach = eds.robot_configurations_fail_to_reach[0:9]
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
    
    # multi_dinovas = MultiDinovasSim()
    # #multi_dinovas.main(robot_configurations_fail_to_reach, goal_positions, goal_orientations)
    # multi_dinovas.main(failure_data)
