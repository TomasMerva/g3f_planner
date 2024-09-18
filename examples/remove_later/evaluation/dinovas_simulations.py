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

# import examples:
from examples.rollouts.example_sqp_planner_dinovas import Environment, run_dinova_example

class ComparisonDinovas():
    def __init__(self, n_runs=2):
        self.nr_robots = 2
        self.dof = 11
        self.n_runs = n_runs
        self.cases = ["RGF"] #"GF", "RF", "MPC"]
        self.render = True
        self.results_struct = {"collision":[], "goal_reached":[], "time_to_goal":[], "computation_time_Rollouts":[], "computation time_GOMP":[]}
        self.results = {self.cases[0]: self.results_struct} #, self.cases[1]: self.results_struct, self.cases[2]: self.results_struct}

    def create_environment(self):
        # --- create environment ---#
        env = Environment()
        home_config = self.randomize_default_home_config()
        env = self.randomize_obstacle_config(env)
        env.initialize(self.render, nr_robots=self.nr_robots, home_config=home_config)
        pybullet.setGravity(0, 0, 0)
        env.create_scene()
        return env

    def euclidean_distance(self, pos_0, pos_1):
        return np.linalg.norm(pos_0 - pos_1)

    def randomize_default_home_config(self):
        home_config = np.array([np.array([0, 3, -np.pi / 2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9]),
                                np.array([1.5, 3, -np.pi / 2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])])
        xyz_random = [[random.uniform(-2, 2), random.uniform(1.5, 3), random.uniform(-3.12, 3.12)] for _ in range(self.nr_robots)]
        while self.euclidean_distance(np.array(xyz_random[0][0:1]), np.array(xyz_random[1][0:1]))<0.8:
            xyz_random = [[random.uniform(-2, 2), random.uniform(1, 3), random.uniform(-3.12, 3.12)] for _ in range(self.nr_robots)]
        home_config[0][0:3] = xyz_random[0][0:3]
        home_config[1][0:3] = xyz_random[1][0:3]
        return home_config

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

    def run_i(self, case="test", env=None):
        # --- run example dinovas --- #
        results_i = run_dinova_example(n_steps=5000, render=self.render, dof=self.dof, nr_robots=self.nr_robots, env=env)
        for key in results_i.keys():
            self.results[case][key].append(results_i[key])

    def run_comparison(self):
        for i_run in range(self.n_runs):
            env = self.create_environment()
            self.run_i(case="RGF", env=env)

    def table_results(self, results):
        # --- create and plot table --- #
        rows = []
        title_row = [' ', "Computation time RF [ms]"] #'Success-Rate', 'Time-to-Success [s]',
        nr_column = len(title_row)
        rows.append(title_row)
        for case in self.cases:
            rows.append([case,
                         # str(np.round(np.sum(results[case]["goal_reached"]) / n_runs, decimals=1)), #+ "+-" + str(np.round(np.nanstd(results[case]["goal_reached"]), decimals=4)),
                         # str(np.round(np.nanmean(results[case]["time_to_goal"]), decimals=4)) + " $\pm$ " + str(np.round(np.nanstd(results[case]["time_to_goal"]), decimals=4)),
                         # str(np.round(np.nanmean(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)),
                         str(np.round(np.nanmean(np.concatenate(results[case]["computation_time_Rollouts"], axis=0)),decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["computation_time_Rollouts"], axis=0)), decimals=6)),
                         ])
        table = Texttable()
        table.set_cols_align(["c"] * nr_column)
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(rows)
        print('\nTexttable Latex:')
        print(latextable.draw_latex(table)) #, caption="\small{Statistics for 50 simulated scenarios of our proposed methods \ac{gm} and \ac{cm} compared to 50 scenarios of \ac{gf} and \ac{smp}}"))

if __name__ == "__main__":
    comparison_dinovas = ComparisonDinovas()
    comparison_dinovas.run_comparison()
    comparison_dinovas.table_results(comparison_dinovas.results)



