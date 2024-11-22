#!/usr/bin/python3
"""
This file generates a table of the results of several simulated experiments with varying
initial position, goal positions and obstacle positions
"""

import yaml
import numpy as np
from texttable import Texttable
import latextable
import copy
import pickle
import pybullet
import random
import time
from tqdm import tqdm
import math

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from dinovas_pybullet_env import Environment
from evaluation.record_data import RecordData, EvaluationDataStructure

from evaluation.dinovas_single_agent.example_deadlock_resolution import run_dinova_example as deadlock_dinova_example
from evaluation.computation_time_comparison.example_gf_dinova import run_dinova_example as fabrics_dinova_example
from evaluation.dinovas_single_agent.example_rgf_dinova import run_dinova_example as gomp_dinova_example

class ComparisonDinovas():
    def __init__(self, n_runs=2, cases= ["IF" ,"GF", "RF"], n_steps_per_run=1000, nr_obst = 50):
        self._scenario_name = "dinovas_single_agent"
        self.nr_robots = 1 #HARD-CODED as 1
        assert self.nr_robots <= 4, "Large number of robots. Not enough urdf files,..."
        self.dof = 11
        self._num_obst = 5
        self.n_runs = n_runs
        self._stopping_tolerance = 0.07
        self.n_steps_per_run = n_steps_per_run
        self.cases = cases
        self.nr_obst = nr_obst# new variable for generating obstacles

        self.results = [{
            case: EvaluationDataStructure() for case in self.cases
        } for _ in range(self.n_runs)]

        self._render = False
        self.scenarios = {run_id:{} for run_id in range(self.n_runs)}

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self._fabrics_config_file = os.path.normpath(os.path.join(current_script_dir, "dinova_config_fabrics.yaml"))
        self._gomp_config_file = os.path.normpath(os.path.join(current_script_dir, "dinova_config_if_5obst.yaml"))
    
    def generate_obstacles_for_scenario(self, num_obstacles, position_range = [(-5, 5), (-5, 5), (0, 2)], radius_range = (0.1, 0.5), min_distance = 2):
        """
        Generates valid obstacles for a scenario, ensuring they don't overlap with the robot's base.
        
        Args:
            scenario_id (int): The ID of the scenario to update.
            num_obstacles (int): Number of obstacles to generate.
            position_range (list of tuples): Ranges for x, y, z positions [(x_min, x_max), (y_min, y_max), (z_min, z_max)].
            radius_range (tuple): Range for obstacle radii (r_min, r_max).
            min_distance (float): Minimum distance between the robot base and obstacles (including obstacle radius).
        """

        self.scenario = {"x_grasp": {"position":[0.5, 0, 0.7],  "orientation": [0, 0, 1, 0.]},
            "q_home": [[ 0 , 0,  0 , 0, 0, 0, 1.7 , 1.57 , -1.57  ]],
            "x_obsts": [[3, 1, 0], [-10, -5, 0], [-10, -5, 0], [-10, -10, 0]],
            "r_obsts": [0.2, 0.2, 0.2, 0.2]
            }
        
        # Get the robot's base position (first three elements of q_home)
        robot_base = self.scenario["q_home"][0][:3]
        
        # Generate valid obstacles
        x_obsts = []
        r_obsts = []
        
        for _ in range(num_obstacles):
            valid_obstacle = False
            while not valid_obstacle:
                # Generate a random obstacle position and radius
                position = [
                    round(random.uniform(position_range[0][0], position_range[0][1]), 2),  # x-coordinate
                    round(random.uniform(position_range[1][0], position_range[1][1]), 2),  # y-coordinate
                    round(random.uniform(position_range[2][0], position_range[2][1]), 2)   # z-coordinate
                ]
                radius = round(random.uniform(radius_range[0], radius_range[1]), 2)
                
                # Calculate the Euclidean distance from the robot base to the obstacle
                distance = math.sqrt(
                    (robot_base[0] - position[0])**2 +
                    (robot_base[1] - position[1])**2 +
                    (robot_base[2] - position[2])**2
                )
                
                # Validate the obstacle
                if distance > min_distance + radius:
                    valid_obstacle = True
                    x_obsts.append(position)
                    r_obsts.append(radius)
        
        # Update the scenario with valid obstacles
        self.scenario["x_obsts"] = x_obsts # a list of obs
        self.scenario["r_obsts"] = r_obsts
        print(self.scenario)
        
    def set_variations_for_simulation(self, run_id=0):
        #modify yaml
        self.generate_obstacles_for_scenario(self.nr_obst)
        radius_list = self.scenario["r_obsts"]
        position_list = self.scenario["x_obsts"]
        
        # reduced_radius_list = [radius - 0.1 for radius in radius_list]
        # generate_xacro_file(reduced_radius_list, position_list)
        
        # for i_robot in range(self.nr_robots):# currently only single robot
            # data["controller"]["waypoints"][0]["position"] = self.scenarios[run_id]["x_grasp"][i_robot]["position"]
            # data["controller"]["waypoints"][0]["orientation"] = self.scenarios[run_id]["x_grasp"][i_robot]["orientation"]
            # data["simulation"]["robot"]["home"] = self.scenario["q_home"][i_robot]
            # data["file_storage_name"] = "datasets/dinovas_data"+str(run_id)+".pkl" #added
            

        with open(self._fabrics_config_file, 'r') as file:
            self.data_yaml = yaml.safe_load(file)
            
        
        self.data_yaml['problem']['environment']['number_spheres']['static'] = self.nr_obst
        # # Access `obstacles` key or create it if it doesn't exist
        obstacles = self.data_yaml['problem']['environment'].get('obstacle_definition', {})
        
        # # Clear or initialize the `static` and `collision_pairs` keys
        obstacles = {}
        
        # Add new obstacles if the list is not empty
        if self.nr_obst:
            for idx in range(self.nr_obst):
                # Generate the obstacle ID
                obstacle_id = f"obstacle{idx}"
                
                # Define obstacle_type, assuming you want to specify a type like 'sphere'
                obstacle_type = "sphere"  # Adjust this if you have more types (e.g., 'box', 'cylinder', etc.)

                # Define the obstacle structure as per the new requirement
                obstacles[obstacle_id] = {
                        "type": obstacle_type,  # Type of the obstacle, like 'sphere'
                        "position": position_list[idx],  # Position of the obstacle (from your scenario)
                        "radius": radius_list[idx]  # Radius of the obstacle (from your scenario)
                    }


                # # Append the obstacle definition to the `obstacles` list
                # obstacles.append(obstacle_definition)


                # # Add collision pairs for this obstacle
                # for link in self.robot_links:
                #     obstacles['collision_pairs'].append([link, obstacle_id])
                        
                # # # Update the data dictionary
        self.data_yaml['problem']['environment']['obstacle_definition'] = obstacles
        
        ##Should be appended
        # for i_obst in range(nr_obst):# to change the yaml file
        #     data_obst["controller"]["obstacles"]["static"][i_obst]["radius"]  = self.scenario["r_obsts"][i_obst]#- min_distance
        #     data_obst["controller"]["obstacles"]["static"][i_obst]["position"]  = list(self.scenario["x_obsts"][i_obst])
            
            
            
        with open(self._fabrics_config_file, 'w') as file:
            yaml.dump(self.data_yaml, file, default_flow_style=False)
        
    # return data["file_storage_name"]

    def create_environment(self):
        # --- create environment ---#
        self.set_variations_for_simulation()
        env = Environment(config_file=self._fabrics_config_file)
        self._home_config = self.randomize_default_home_config()
        objects_pos_noise = self.randomize_objects_pos()
        env.set_objects_pos_noise(objects_pos_noise)
        self.grasp_list = None
        return env
    
    def load_environment(self, run_id=0):
        env = Environment(config_file=self._fabrics_config_file)
        pickle_file_path = '../results/' + self._scenario_name + "_env.pickle"
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        # environment_settings = data["environment_settings"]
        # self._home_config = np.array(data[run_id]["q_home"])
                        #     obst_dict = env.get_obstacles()
        # self.scenarios[run_id] = {
        #         "x_grasp" : env.compute_init_static_grasp(self.nr_robots),
        #         "x_obsts" : [obst_dict[obst]["position"] for obst in obst_dict],
        #         "r_obsts" : [obst_dict[obst]["radius"] for obst in obst_dict]
        #         }
        self._home_config = np.array([np.concatenate((vec, [0.9, -0.9])) for vec in data[run_id]["q_home"]]) 
        
        self.scenarios[run_id] = data[run_id]
        obst_pos_dict = self.scenarios[run_id]["x_obsts"][2:6]
        #obst_radii_dict = self.scenarios[run_id]["r_obsts"][2:6]
        # objects_pos_noise = self.randomize_objects_pos()
        obsts_pos = [list(obst) for obst in obst_pos_dict]
        # obsts_r = [list(obst) for obst in obst_radii_dict]
        self.grasp_list = self.scenarios[run_id]["x_grasp"]
        # env.set_objects_pos_noise(objects_pos_noise)
        env.set_obsts_pos(pos=obsts_pos, start_idx=2) 
        # env.set_obsts_pos(radii=obsts_r, start_idx=2) 
        #attach the gripper config to robots 9dim home config

        return env

    def euclidean_distance(self, pos_0, pos_1):
        return np.linalg.norm(pos_0 - pos_1)


    def _is_within_annular_region(self, point, outer_radius, inner_radius):
        distance_from_center = self.euclidean_distance(np.zeros(2), point)
        return inner_radius < distance_from_center <= outer_radius

    def randomize_default_home_config(self):
        home_config = np.array([0, 3, -np.pi / 2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])
        x_range = [-3, 3]
        y_range = [2.0, 5.0]
        # z_range = [-3.12, 3.12]
        z_range = [-2, 2]

        xyz_random = []
        safety_counter = 0
        while len(xyz_random) < 1:
            new_point = (round(random.uniform(x_range[0], x_range[1]), 5), 
                         round(random.uniform(y_range[0], y_range[1]), 5), 
                         round(random.uniform(z_range[0], z_range[1]), 5))

            if all(self.euclidean_distance(np.array(new_point)[0:2], np.array(p)[0:2]) > 1.5 for p in xyz_random):
                xyz_random.append(new_point)
            safety_counter += 1
            if safety_counter >= 50:
                raise ValueError("Cannot find valid home configurations for so many robots")
        
        configs = np.zeros((self.nr_robots, self.dof))
        for robot in range(1):
            home_config[0:3] = xyz_random[robot][0:3]
            configs[robot] = copy.deepcopy(home_config)
        #configs[1] =  np.array([0, 0.85, -np.pi / 2, 0, 0, 1.9, 0, 0, 0, 0.9, -0.9])
        return configs


    def randomize_objects_pos(self):
        x_range = [-0.3, 0.3]
        y_range = [-0.3, 0.3]
        
        outer_radius = 0.3  # Outer radius of the circle
        inner_radius = 0.2
        tolerance = 0.35
        points = []
        points_other_cups = [[0.0, -0.25]]
        
        counter = 0
        max_number_of_objects = 1
        while len(points) < max_number_of_objects:
            # Generate random angle and radius
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(inner_radius, outer_radius)
            
            # Convert polar coordinates (radius, angle) to Cartesian coordinates (x, y)
            x = round(radius * math.cos(angle), 5)
            y = round(radius * math.sin(angle), 5)
            new_point = (x, y)
            
            # Check if the new point is far enough from all existing points and within the annular region
            if self._is_within_annular_region(new_point[:2], outer_radius, inner_radius) \
                and all(self.euclidean_distance(np.array(new_point), np.array(p)) > tolerance for p in points_other_cups):
                points.append(new_point)
            
            counter += 1
            if counter > 200:
                raise ValueError("Cannot find valid points for so many objects")
        
        points.append(points_other_cups[0])
        points.append([round(random.uniform(x_range[0], x_range[1]), 5),  round(random.uniform(y_range[0], y_range[1]))])
        points.append([round(random.uniform(x_range[0], x_range[1]), 5),  round(random.uniform(y_range[0], y_range[1]))])
        return np.asarray(points)

    def run_i(self, run_id, case="test", env=None):
        # --- run example dinovas --- #
        if case == "IF":
            self.results[run_id][case]  = gomp_dinova_example(
                                            n_steps=self.n_steps_per_run, 
                                            dof=self.dof, 
                                            n_robots=self.nr_robots, 
                                            gomp_config_file=self._gomp_config_file,
                                            env=env, 
                                            nr_obst=self._num_obst,
                                            render=self._render,
                                            stopping_tolerance=self._stopping_tolerance
                                            )     
        elif case == "GF":
            self.results[run_id][case] = fabrics_dinova_example(
                                            n_steps=self.n_steps_per_run, 
                                            dof=self.dof, 
                                            n_robots=self.nr_robots, 
                                            env=env,
                                            stopping_tolerance=self._stopping_tolerance,
                                            grasp_goals = self.grasp_list
                                            )
        elif case == "RF":
            self.results[run_id][case] = deadlock_dinova_example(
                                                n_steps=self.n_steps_per_run, 
                                                dof=self.dof, 
                                                n_robots=self.nr_robots, 
                                                gomp_config_file=self._gomp_config_file,
                                                env=env,
                                                stopping_tolerance=self._stopping_tolerance
                                                )
        elif case == "MPC":
            raise ValueError("MPC is not implemented.")
      

    def run_comparison(self, render, LOAD_SCENARIO=False, SAVE_DATA=False):
        self._render = render
        for i_run in tqdm(range(self.n_runs)):
            # if LOAD_SCENARIO:
            #     env = self.load_environment(run_id=1)
            # else:
            env = self.create_environment()
            
            for i, algorithm in enumerate(self.cases):
                env.initialize(render, nr_robots=self.nr_robots, home_config=self._home_config, nr_tables=0)    
                # if i == 0:
                #     obst_dict = env.get_obstacles()
                #     self.scenarios[i_run] = {
                #         "q_home" : env.get_home_configs(),
                #         "x_grasp" : env.compute_init_static_grasp(self.nr_robots),
                #         "x_obsts" : [obst_dict[obst]["position"] for obst in obst_dict],
                #         "r_obsts" : [obst_dict[obst]["radius"] for obst in obst_dict]
                #         }
                self.run_i(case=algorithm, env=env, run_id = i_run)
        
        if SAVE_DATA:
            pickle_file_path = 'results/fabrics_'+str(self.nr_obst)
            # with open(pickle_file_path+"_env.pickle", 'wb') as handle:
            #     pickle.dump(self.scenarios, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(pickle_file_path + '_results.pickle', 'wb') as handle:
                pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Env saved at: {pickle_file_path + '_env.pickle'}")
            print(f"Results saved at: {pickle_file_path + '_results.pickle'}")

    def table_results(self):    
        # --- create and plot table --- #
        rows = []
        title_row = [' ', "Success rate [\%]", 'Time-to-Success [s]', "Computation time[s]", "IF Computation time[s]", "Collision-rate"]
        nr_column = len(title_row)
        rows.append(title_row)
        for case in self.cases:
            rows.append([case,  
                         str(np.round(np.sum([entry[case].goal_reached for entry in self.results]) / self.n_runs, decimals=1)) + " $\%$ ", 
                         str(np.round(np.nanmean([entry[case].time_to_goal for entry in self.results]), decimals=4)) + " $\pm$ " + str(np.round(np.nanstd([entry[case].time_to_goal for entry in self.results]), decimals=4)),
                         str(np.round(np.nanmean(np.concatenate([entry[case].computation_time for entry in self.results], axis=0)),decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate([entry[case].computation_time for entry in self.results], axis=0)), decimals=6)),
                         str(np.round(np.nanmean(np.concatenate([entry[case].computation_time_qp for entry in self.results], axis=0)),decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate([entry[case].computation_time_qp for entry in self.results], axis=0)), decimals=6)),
                         str(np.round(np.sum([entry[case].collision for entry in self.results]) / self.n_runs, decimals=1)),
                         
                         ])
            
        table = Texttable()
        table.set_cols_align(["c"] * nr_column)
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(rows)
        print('\nTexttable Latex:')
        print(latextable.draw_latex(table))


def main(render=True, n_runs=20, cases= ["IF" ,"GF", "RF"], timesteps=5000, save_data=True, nr_obst = 0):
    random.seed(0)
    np.random.seed(0)
    comparison_dinovas = ComparisonDinovas(n_runs=n_runs, n_steps_per_run=timesteps, cases=cases, nr_obst=nr_obst)
    comparison_dinovas.run_comparison(render =render, LOAD_SCENARIO=False, SAVE_DATA=save_data)
    print("Results from Single agent scenario")
    print("==================================")
    comparison_dinovas.table_results()
    return {}

if __name__ == "__main__":
    # EvaluateSimulationsMPC(n_runs=1, n_obstacles=number_of_obstales).run_simulations()
    numbers_of_obstales = [35]
    # Iterate over each PCK_LOAD_FILE and run simulations
    for number_of_obstales in numbers_of_obstales:
        main(render=False, 
            n_runs=10, 
            timesteps=3500, 
            cases=["GF"], # ,"GF", "RF"
            save_data=True,
            nr_obst = number_of_obstales)



