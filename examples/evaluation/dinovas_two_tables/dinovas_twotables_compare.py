"""
This file generates a table of the results of several simulated experiments with varying
initial position, goal positions and obstacle positions
"""

import numpy as np
from texttable import Texttable
import latextable
import copy
import pickle
import random
import time
from tqdm import tqdm
import sys
import os
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from dinovas_pybullet_env import Environment
from example_deadlock_resolution import run_dinova_example as deadlock_dinova_example
from example_vanilla_fabrics import run_dinova_example as fabrics_dinova_example
from example_rgf_dinovas import run_dinova_example as gomp_dinova_example
from evaluation.record_data import RecordData, EvaluationDataStructure

class ComparisonDinovas():
    def __init__(self, n_runs=2, cases= ["IF" ,"GF", "RF"], n_steps_per_run=1000):
        self._scenario_name = "dinovas_two_tables"
        self.nr_robots = 2
        assert self.nr_robots <= 4, "Large number of robots. Not enough urdf files,..."
        self.dof = 11
        self._num_obst = 6
        self._stopping_tolerance = 0.07
        self.n_runs = n_runs
        self.n_steps_per_run = n_steps_per_run
   
        self.cases = cases
        self.results = [{
            case: EvaluationDataStructure() for case in self.cases
        } for _ in range(self.n_runs)]

        self._render = False
        self.scenarios = {run_id:{} for run_id in range(self.n_runs)}

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self._fabrics_config_file = os.path.normpath(os.path.join(current_script_dir, "dinova_config_fabrics.yaml"))
        self._gomp_config_file = os.path.normpath(os.path.join(current_script_dir, "dinova_config_if_6obst.yaml"))

    def create_environment(self):
        # --- create environment ---#
        env = Environment(config_file=self._fabrics_config_file)
        self._home_config = self.randomize_default_home_config()
        objects_pos_noise = self.randomize_objects_pos()
        env.set_objects_pos_noise(objects_pos_noise)
        return env

    def load_environment(self, run_id=0):
        env = Environment(config_file=self._fabrics_config_file)
        pickle_file_path = '../results/' + self._scenario_name + "_env.pickle"
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        environment_settings = data["environment_settings"]
        self._home_config = np.array(environment_settings[run_id]["q_home"])
        return env

    def euclidean_distance(self, pos_0, pos_1):
        return np.linalg.norm(pos_0 - pos_1)


    def _is_within_annular_region(self, point, outer_radius, inner_radius):
        distance_from_center = self.euclidean_distance(np.zeros(2), point)
        return inner_radius < distance_from_center <= outer_radius

    def randomize_default_home_config(self):
        home_config = np.array([0, 3, -np.pi / 2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])
        x_range = [-3, 0.0]
        y_range = [2.0, 5.0]
        z_range = [-2, 2]

        xyz_random = []
        safety_counter = 0
        while len(xyz_random) < self.nr_robots:
            if len(xyz_random) >= 1 :
                new_point = (-round(random.uniform(x_range[0], x_range[1]), 5), 
                            round(random.uniform(y_range[0], y_range[1]), 5), 
                            round(random.uniform(z_range[0], z_range[1]), 5))
            else:
                new_point = (round(random.uniform(x_range[0], x_range[1]), 5), 
                            round(random.uniform(y_range[0], y_range[1]), 5), 
                            round(random.uniform(z_range[0], z_range[1]), 5))

            if all(self.euclidean_distance(np.array(new_point)[0:2], np.array(p)[0:2]) > 1.5 for p in xyz_random):
                xyz_random.append(new_point)
            safety_counter += 1
            if safety_counter >= 50:
                raise ValueError("Cannot find valid home configurations for so many robots")
        
        configs = np.zeros((self.nr_robots, self.dof))
        for robot in range(self.nr_robots):
            home_config[0:3] = xyz_random[robot][0:3]
            configs[robot] = copy.deepcopy(home_config)
        return configs

    def randomize_obstacle_config(self):
        x_range = [-2, 2]
        y_range = [1, 2]

        points = []
        #  8 obstacles in total
        #  1 obstacle is the table
        #  2 obstacle spheres for other agents
        max_number_of_obsts = 8 - (self.nr_robots-1)*2 -1
        for i in range(max_number_of_obsts):
            new_point = (round(random.uniform(x_range[0], x_range[1]), 5),
                         round(random.uniform(y_range[0], y_range[1]), 5),
                         0.15)
            points.append(new_point)
        return points

        
    def randomize_objects_pos(self):
        x_range = [-0.3, 0.3]
        y_range = [-0.3, 0.3]
        
        outer_radius = 0.3  # Outer radius of the circle
        inner_radius = 0.2
        tolerance = 0.35
        points = []
        
        counter = 0
        max_number_of_objects = 2
        while len(points) < max_number_of_objects:
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(inner_radius, outer_radius)
            
            x = round(radius * math.cos(angle), 5)
            y = round(radius * math.sin(angle), 5)
            new_point = (x, y)
            
            if self._is_within_annular_region(new_point[:2], outer_radius, inner_radius) \
                and all(self.euclidean_distance(np.array(new_point), np.array(p)) > tolerance for p in points):
                points.append(new_point)
            
            counter += 1
            if counter > 200:
                raise ValueError("Cannot find valid points for so many objects")
        
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
                                            stopping_tolerance=self._stopping_tolerance
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
            if LOAD_SCENARIO:
                env = self.load_environment(run_id=1)
            else:
                env = self.create_environment()
                
            for i, algorithm in enumerate(self.cases):
                env.initialize(render, nr_robots=self.nr_robots, home_config=self._home_config, nr_tables=2)
                if i == 0:
                    obst_dict = env.get_obstacles()
                    self.scenarios[i_run] = {
                        "q_home" : env.get_home_configs(),
                        "x_grasp" : env.compute_init_static_grasp(self.nr_robots),
                        "x_obsts" : [obst_dict[obst]["position"] for obst in obst_dict],
                        "r_obsts" : [obst_dict[obst]["radius"] for obst in obst_dict]
                        }
                self.run_i(case=algorithm, env=env, run_id = i_run)
        if SAVE_DATA:
            pickle_file_path = '../results/' + self._scenario_name
            with open(pickle_file_path+"_env.pickle", 'wb') as handle:
                pickle.dump(self.scenarios, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
        
      

def main(render=True, n_runs=20, cases= ["IF" ,"GF", "RF"], timesteps=5000, save_data=True):
    random.seed(0)
    np.random.seed(0)
    comparison_dinovas = ComparisonDinovas(n_runs=n_runs, n_steps_per_run=timesteps, cases=cases)
    comparison_dinovas.run_comparison(render =render, LOAD_SCENARIO=False, SAVE_DATA=save_data)
    print("Results from Two tables scenario")
    print("==================================")
    comparison_dinovas.table_results()
    return {}

if __name__ == "__main__":
    main(render=True, 
         n_runs=20, 
         timesteps=5000, 
         cases=["GF"], # ,"GF", "RF"
         save_data=False)