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

# import examples:
from examples.rollouts.example_sqp_planner_dinovas import Environment, run_dinova_example

class ComparisonDinovas():
    def __init__(self, n_runs=2):
        self.nr_robots = 2
        self.dof = 11
        self.n_runs = n_runs
        self.cases = ["RGF"] #"GF", "RF", "MPC"]
        self.render = False
        self.results_struct = {"collision":[], "goal_reached":[], "time_to_goal":[], "computation_time_Rollouts":[], "computation time_GOMP":[]}
        self.results = {self.cases[0]: self.results_struct} #, self.cases[1]: self.results_struct, self.cases[2]: self.results_struct}

    def create_environment(self):
        # --- create environment ---#
        env = Environment()
        env.initialize(self.render, nr_robots=self.nr_robots)
        pybullet.setGravity(0, 0, 0)
        env.create_scene()
        return env

    def run_i(self, case="test", env=None):
        # --- run example dinovas --- #
        results_i = run_dinova_example(n_steps=1000, render=self.render, dof=self.dof, nr_robots=self.nr_robots, env=env)
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



