#!/usr/bin/env python3
import numpy as np
import pickle
from texttable import Texttable
import latextable
import os
import sys

#import IPython
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.record_data import RecordData
pkl_file_path = "results/dinovas_two_tables_results.pickle"
# Load the data from the .pkl file
try:
    with open(pkl_file_path, 'rb') as handle:
        results = pickle.load(handle)
    print("Data loaded successfully from:", pkl_file_path)
except Exception as e:
    print(f"An error occurred while loading the .pkl file: {e}")
    results = None

# Parameters for the table
title_row = [' ', "Success rate [\%]", 'Time-to-Success [s]', "Path [m]", "Computation time[s]", "Collision-rate"]
nr_column = len(title_row)

# Initialize table rows
rows = [title_row]

# Add data to rows if the results loaded successfully
if results:
    # Loop through each case in results to create the table rows
    n_runs = len(results)
    cases = results[0].keys()  # Assuming all entries have the same cases
    for case in cases:
        success_rate = np.round(np.sum([entry[case].goal_reached for entry in results]) / n_runs * 100, decimals=1)
        time_to_success_mean = np.round(np.nanmean([entry[case].time_to_goal for entry in results]), decimals=4)
        time_to_success_std = np.round(np.nanstd([entry[case].time_to_goal for entry in results]), decimals=4)
        computation_time_mean = np.round(np.nanmean(np.concatenate([entry[case].computation_time for entry in results], axis=0)), decimals=6)
        computation_time_std = np.round(np.nanstd(np.concatenate([entry[case].computation_time for entry in results], axis=0)), decimals=6)
        collision_rate = np.round(np.sum([entry[case].collision for entry in results]) / n_runs, decimals=1)
        path_mean = np.round(np.nanmean([entry[case].path for entry in results]), decimals=4)
        path_std = np.round(np.nanstd([entry[case].path for entry in results]), decimals=4)

        # Append a row for each case
        rows.append([
            case,
            f"{success_rate} \\%",
            f"{time_to_success_mean} $\\pm$ {time_to_success_std}",
            f"{path_mean} $\\pm$ {path_std}",
            f"{computation_time_mean} $\\pm$ {computation_time_std}",
            collision_rate
        ])

# Create the table using Texttable
table = Texttable()
table.set_cols_align(["c"] * nr_column)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

# Print the table in LaTeX format
print('\nTexttable Latex:')
print(latextable.draw_latex(table))

#############################################################################
# Collect indices of failed scenarios for each case
failed_scenarios = {case: [] for case in results[0].keys()}  # Initialize dictionary for each case

# Iterate over each entry in results
for i, entry in enumerate(results):
    for case in entry:
        if not entry[case].goal_reached:
            failed_scenarios[case].append(i)  # Append the index of the failed scenario

# Print indices of failed scenarios for each case
for case, indices in failed_scenarios.items():
    print(f"Failed scenario indices for case '{case}': {indices}")


#Failed scenario indices for case 'MPC': [2, 3, 7, 10, 11, 12, 13, 14, 15]

# Collect indices of collision scenarios for each case
collision_scenarios = {case: [] for case in results[0].keys()}  # Initialize dictionary for each case
collision_pairs = {case: [] for case in results[0].keys()}  # Initialize dictionary for each case
# Iterate over each entry in results
for i, entry in enumerate(results):
    for case in entry:
        if entry[case].collision:  # Check if a collision occurred
            collision_scenarios[case].append(i)  # Append the index of the collision scenario
            collision_pairs[case].append(entry[case].collision_pair)

# Print indices of collision scenarios for each case
for case, indices in collision_scenarios.items():
    print(f"Collision scenario indices for case '{case}': {indices}")
# for case, indices in collision_pairs.items():
    print("Collision Pairs", collision_pairs)
