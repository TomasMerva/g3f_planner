"""
This file generates a table of the results of several simulated experiments with varying
initial position, goal positions and obstacle positions
"""

import numpy as np
from texttable import Texttable
import latextable
import copy
import pickle
import pybullet
import random
import time
from tqdm import tqdm
import contextlib

# import examples:
# from examples.rollouts.example_sqp_planner_dinovas import Environment, run_dinova_example
from dinovas_pybullet_env import Environment
from example_deadlock_resolution import run_dinova_example as deadlock_dinova_example
from example_vanilla_fabrics import run_dinova_example as fabrics_dinova_example
from example_rgf_dinovas import run_dinova_example as gomp_dinova_example

class ComparisonDinovas():
    def __init__(self, n_runs=2, n_steps_per_run=1000):
        self.nr_robots = 2
        assert self.nr_robots <= 4, "Large number of robots. Not enough urdf files,..."
        self.dof = 11
        self.n_runs = n_runs
        self.n_steps_per_run = n_steps_per_run
        self.cases = ["RGF", "GF"] #["RGF" ,"GF", "RF", "MPC"]
        # self.results_struct = {"collision":[], "goal_reached":[], "time_to_goal":[], "computation_time_Rollouts":[], "computation time_GOMP":[]}
        self.results_struct = {"collision":[], "goal_reached":[], "time_to_goal":[], "computation_time":[]}
        self.results = {self.cases[0]: self.results_struct,
                        self.cases[1]: self.results_struct} #, self.cases[1]: self.results_struct, self.cases[2]: self.results_struct}
        self._render = False

    def create_environment(self, render=False):
        # --- create environment ---#
        env = Environment()
        self._home_config = self.randomize_default_home_config()
        objects_pos_noise = self.randomize_objects_pos()

        env.set_objects_pos_noise(objects_pos_noise)
        # env = self.randomize_obstacle_config(env)
        env.initialize(render, nr_robots=self.nr_robots, home_config=self._home_config)
        return env

    def euclidean_distance(self, pos_0, pos_1):
        return np.linalg.norm(pos_0 - pos_1)


    def _is_within_annular_region(self, point, outer_radius, inner_radius):
        distance_from_center = self.euclidean_distance(np.zeros(2), point)
        return inner_radius < distance_from_center <= outer_radius

    def randomize_default_home_config(self):
        home_config = np.array([0, 3, -np.pi / 2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])
        x_range = [-3, 3]
        y_range = [1.0, 3.0]
        z_range = [-3.12, 3.12]

        xyz_random = []
        safety_counter = 0
        while len(xyz_random) < self.nr_robots:
            new_point = (round(random.uniform(x_range[0], x_range[1]), 5), 
                         round(random.uniform(y_range[0], y_range[1]), 5), 
                         round(random.uniform(z_range[0], z_range[1]), 5))

            if all(self.euclidean_distance(np.array(new_point)[0:2], np.array(p)[0:2]) > 2 for p in xyz_random):
                xyz_random.append(new_point)
            safety_counter += 1
            if safety_counter >= 50:
                raise ValueError("Cannot find valid home configurations for so many robots")
        
        configs = np.zeros((self.nr_robots, self.dof))
        for robot in range(self.nr_robots):
            home_config[0:3] = xyz_random[robot][0:3]
            configs[robot] = copy.deepcopy(home_config)
        return configs

    def randomize_obstacle_config(self, env):
        # todo: make working for more than 2 obstacles!
        obst_struct = env.CONFIG_PROBLEM["environment"]["obstacle_definition"]
        xyz_random = [[random.uniform(-2, 2), random.uniform(-0.5, 1), random.uniform(0, 0.3)] for _ in
                      range(self.nr_robots)]
        while self.euclidean_distance(np.array(xyz_random[0][0:1]), np.array(xyz_random[1][0:1])) < 0.8:
            xyz_random = [[random.uniform(-2, 2), random.uniform(-0.5, 1), random.uniform(0, 0.3)] for _ in
                          range(self.nr_robots)]
            #todo: add check if obstacle is colliding with object position (or check placing better)
        for i, obstacle_name in enumerate(obst_struct.keys()):
            env.CONFIG_PROBLEM["environment"]["obstacle_definition"][obstacle_name]["position"] = xyz_random[i]
        return env

    def randomize_objects_pos(self):
        x_range = [-0.3, 0.3]
        y_range = [-0.1, 0.1]

        points = []

        max_number_of_objects = 2
        safety_counter = 0
        while len(points) < max_number_of_objects:
            new_point = (round(random.uniform(x_range[0], x_range[1]), 5),
                         round(random.uniform(y_range[0], y_range[1]), 5))
            
            if all(self.euclidean_distance(np.array(new_point), np.array(p)) > 0.2 for p in points):
                points.append(new_point)
            
            safety_counter += 1
            if safety_counter >= 2000:
                raise ValueError("Cannot find valid points for so many objects")
        points.append([round(random.uniform(x_range[0], x_range[1]), 5),  round(random.uniform(y_range[0], y_range[1]))])
        points.append([round(random.uniform(x_range[0], x_range[1]), 5),  round(random.uniform(y_range[0], y_range[1]))])
        return np.asarray(points)


    def run_i(self, case="test", env=None):
        # --- run example dinovas --- #
        #["RGF" ,"GF", "RF", "MPC"]
        if case == "RGF":
            results_i = gomp_dinova_example(n_steps=self.n_steps_per_run, dof=self.dof, n_robots=self.nr_robots, env=env, render=self._render)
            for key in results_i.keys():
                self.results[case][key].append(results_i[key])        
        elif case == "GF":
            results_i = fabrics_dinova_example(n_steps=self.n_steps_per_run, dof=self.dof, n_robots=self.nr_robots, env=env)
            for key in results_i.keys():
                self.results[case][key].append(results_i[key])
        elif case == "RF":
            results_i = deadlock_dinova_example(n_steps=self.n_steps_per_run, dof=self.dof, n_robots=self.nr_robots, env=env)
            for key in results_i.keys():
                self.results[case][key].append(results_i[key])
        elif case == "MPC":
            raise ValueError("MPC is not implemented.")
      

    def run_comparison(self, render):
        self._render = render
        for i_run in tqdm(range(self.n_runs)):
            env = self.create_environment(self._render)
            for algorithm in self.cases:
                self.run_i(case=algorithm, env=env)
                env.initialize(render, nr_robots=self.nr_robots, home_config=self._home_config)


    def table_results(self, results):
        # --- create and plot table --- #
        rows = []
        title_row = [' ', "Success rate [\%]", "Time-to-Success [s]", "Computation time[s]"]#,"Collision-rate" #'Success-Rate', 'Time-to-Success [s]',
        nr_column = len(title_row)
        rows.append(title_row)
        for case in self.cases:
            rows.append([case,
                         str(np.round(np.sum(results[case]["goal_reached"]) / self.n_runs, decimals=1)), # + "+-" + str(np.round(np.nanstd(results[case]["goal_reached"]), decimals=4)),
                         str(np.round(np.nanmean(results[case]["time_to_goal"]), decimals=4)) + " $\pm$ " + str(np.round(np.nanstd(results[case]["time_to_goal"]), decimals=4)),
                         # str(np.round(np.nanmean(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)),
                         str(np.round(np.nanmean(np.concatenate(results[case]["computation_time"], axis=0)),decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["computation_time"], axis=0)), decimals=6)),
                        #  str(np.round(np.sum(results[case]["collision"]) / self.n_runs, decimals=1)),
                         ])
        table = Texttable()
        table.set_cols_align(["c"] * nr_column)
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(rows)
        print('\nTexttable Latex:')
        print(latextable.draw_latex(table)) #, caption="\small{Statistics for 50 simulated scenarios of our proposed methods \ac{gm} and \ac{cm} compared to 50 scenarios of \ac{gf} and \ac{smp}}"))


import sys
import os
@contextlib.contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    start_time = time.perf_counter()
    # with suppress_stdout():
    comparison_dinovas = ComparisonDinovas(n_runs=4, n_steps_per_run=100)
    comparison_dinovas.run_comparison(render = False)
    end_time = time.perf_counter()
    print("Computational time: ", end_time-start_time)
    comparison_dinovas.table_results(comparison_dinovas.results)



