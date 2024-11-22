#!/usr/bin/env python3
import numpy as np
import pickle
from texttable import Texttable
import latextable

# Define the file path of the .pkl file
# pkl_file_path = "datasets/dinovas_MPC_results_dinovas_crossover_env_double_integrator.pkl"
pkl_file_path = "results/fabrics_25_results.pickle"
# pkl_file_path = "datasets/dinovas_MPC_results_50_double_integrator.pkl"
# pkl_file_path = "datasets/dinovas_MPC_results_0_double_integrator.pkl"
# pkl_file_path = "datasets/dinovas_MPC_results_5_double_integrator.pkl"

try:
    with open(pkl_file_path, 'rb') as handle:
        results = pickle.load(handle)
    print("Data loaded successfully from:", pkl_file_path)
except Exception as e:
    print(f"An error occurred while loading the .pkl file: {e}")
    results = None

# Parameters for the table
# title_row = [' ', "Success rate [\%]", 'Time-to-Success [s]', "Computation time[s]", "Collision-rate"]
# nr_column = len(title_row)
print(len(results[0]['GF'].computation_time))
# Add data to rows if the results loaded successfully
rows = []
title_row = [' ', "Computation time[s]", "Collision-rate"]
nr_column = len(title_row)
rows.append(title_row)
cases = results[0].keys()  # Assuming all entries have the same cases
n_runs = len(results)
print("n_runs",n_runs)
for case in cases:
    print("case",case)
    rows.append([case,  
                    str(np.round(np.nanmean(np.concatenate([entry[case].computation_time for entry in results], axis=0)),decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate([entry[case].computation_time for entry in results], axis=0)), decimals=6)),
                    str(np.round(np.sum([entry[case].collision for entry in results]) / n_runs, decimals=1)),
                    ])
    
table = Texttable()
table.set_cols_align(["c"] * nr_column)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)
print('\nTexttable Latex:')
print(latextable.draw_latex(table))
