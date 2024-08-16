import numpy as np

class ReferenceTracker:
    """
    Simple reference tracker to track a set of waypoints
    """
    def __init__(self, tolerance=0.5):
        self.waypoint_list = []
        self.tolerance = tolerance

    def euclidian_distance(self, pos_0, pos_1):
        return np.linalg.norm(pos_0 - pos_1)

    def update_reference(self, waypoint_list:list):
        self.waypoint_list = waypoint_list

    def update_local_goal(self, current_pos: np.ndarray, waypoint_list=None, goal_final=None) -> (list, list):
        if waypoint_list is None:
            waypoint_list = self.waypoint_list

        if goal_final is not None:
            if self.euclidian_distance(current_pos, goal_final) < self.tolerance:
                return goal_final, [goal_final]

        if self.euclidian_distance(current_pos, waypoint_list[-1][:3, 3]) < self.tolerance:
            return list(waypoint_list[-1]), [list(waypoint_list[-1])],
        else:
            for i in range(len(waypoint_list)):
                if self.euclidian_distance(current_pos, waypoint_list[i][:3, 3]) > self.tolerance:
                    waypoint_list = waypoint_list[i:]
                    return waypoint_list[0], waypoint_list
        print("warning: no waypoints on the path are closer than the tolerance")
        return waypoint_list[0], waypoint_list