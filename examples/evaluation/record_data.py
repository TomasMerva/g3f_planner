import numpy as np
from dataclasses import dataclass, field

@dataclass
class EvaluationDataStructure:
    collision : float = 0.0
    goal_reached : float = 0.0
    time_to_goal : float = np.nan
    computation_time: list = field(default_factory=list)
    computation_time_qp : list = field(default_factory=list)
    collision_error : list = field(default_factory=list)
    solver_success_rate : list = field(default_factory=list)


class RecordData():
    def __init__(self):
        self._result = EvaluationDataStructure()
    
    def record_collision_violation(self, collision_flag):
        if collision_flag:
            self._result.collision = 100.0
        else:
            self._result.collision = 0.0

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
    
    def record_collision_error(self, error):
        self._result.collision_error.append(error)

    def record_solver_success_rate(self, rate):
        self._result.solver_success_rate.append(rate)




