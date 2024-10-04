import pickle
import numpy as np
from dataclasses import dataclass, field

# Step 2: Define the path to the pickle file
pickle_file_path = 'evaluation/results/dinovas_tworobots_without_obst_results.pickle'

# Step 3: Open the pickle file in binary read mode
with open(pickle_file_path, 'rb') as file:
    # Step 4: Use pickle.load() to load the data from the file
    data = pickle.load(file)

kkk=1
