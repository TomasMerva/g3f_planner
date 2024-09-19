import pickle
import numpy as np

import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.record_data import RecordData

data = pickle.load(open('dinovas_tworobots_result.pickle', 'rb'))


print(data)
# for case in data:
#     print(data[case].goal_reached)