import numpy as np
import random
import time
from dataclasses import dataclass, field

@dataclass
class EvaluationDataStructure:
    collision : float = 0.0
    goal_reached : float = 0.0
    time_to_goal : float = np.nan
    computation_time: list = field(default_factory=list)
    computation_time_qp : list = field(default_factory=list)


class RecordData():
    def __init__(self):
        self._result = EvaluationDataStructure()
    
    def record_collision_violation(self, collision_flag):
        self._result.collision = collision_flag

    def record_success_rate(self, success):
        self._result.goal_reached = success

    def record_computational_time(self, time):
        self._result.computation_time.append(time)
    
    def record_computational_time_qp(self, time):
        self._result.computation_time_qp.append(time)

    def record_time_to_goal(self, current_timestep, dt):
        self._result.time_to_goal= current_timestep * dt

    def get_result(self):
        return self._result



