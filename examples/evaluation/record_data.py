import numpy as np
import random
import time
from dataclasses import dataclass, field
from typing import List

@dataclass
class EvaluationDataStructure:
    collision : float = 0.0
    goal_reached : float = 0.0
    time_to_goal : float = np.nan
    computation_time: list = field(default_factory=list)
    computation_time_qp : list = field(default_factory=list)
    collision_error : list = field(default_factory=list)
    
    #Extra
    collision_pair: List[int] = field(default_factory=lambda: [-1, -1])
    collision_pair_names: List[str] = field(default_factory=lambda: ["", ""])
    robot_configurations_in_collisions: List[List[np.array]] = field(default_factory=list)  
    robot_configurations_fail_to_reach: List[List[np.array]] = field(default_factory=list) 
    goal_positions: List[List[np.array]] = field(default_factory=list)  # easy for reproduction
    goal_orientations: List[List[np.array]] = field(default_factory=list) 


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

    def record_collision_pair(self, collision_pair):
        self._result.collision_pair = collision_pair
    
    def record_collision_pair_names(self, collision_pair_names):
        self._result.collision_pair_names = collision_pair_names
    
    def record_robot_configurations_in_collisions(self, robot_configurations):
        self._result.robot_configurations_in_collisions.extend(robot_configurations)
        
    def record_robot_configurations_fail_to_reach(self, robot_configurations):
        self._result.robot_configurations_fail_to_reach.extend(robot_configurations)
        
    def record_goal_positions(self, goal_positions):
        self._result.goal_positions.extend(goal_positions)
    
    def record_goal_orientations(self, goal_orientations):
        self._result.goal_orientations.extend(goal_orientations)


