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
    
    def update_local_goal_pos_orient(self, current_pos: np.ndarray, waypoint_list=None) -> (list, list):
        if waypoint_list is None:
            waypoint_list = self.waypoint_list
  
        if self.euclidian_distance(current_pos, waypoint_list[-1]["position"]) < self.tolerance:
            return waypoint_list[-1], [waypoint_list[-1]],
        else:
            for i in range(len(waypoint_list)):
                if self.euclidian_distance(current_pos, waypoint_list[i]["position"]) > self.tolerance:
                    waypoint_list = waypoint_list[i:]
                    return waypoint_list[0], waypoint_list
        print("warning: no waypoints on the path are closer than the tolerance")
        return waypoint_list[0], waypoint_list

    def update_local_goal(self, current_pos: np.ndarray, waypoint_list=None) -> (list, list):
        if waypoint_list is None:
            waypoint_list = self.waypoint_list

        if self.euclidian_distance(current_pos, waypoint_list[-1][:3, 3]) < self.tolerance:
            return waypoint_list[-1], [waypoint_list[-1]],
        else:
            for i in range(len(waypoint_list)):
                if self.euclidian_distance(current_pos, waypoint_list[i][:3, 3]) > self.tolerance:
                    waypoint_list = waypoint_list[i:]
                    return waypoint_list[0], waypoint_list
        print("warning: no waypoints on the path are closer than the tolerance")
        return waypoint_list[0], waypoint_list

    def update_arguments_subgoal(self, T_W_EEF_subgoal, x_goal_1_x, x_goal_2_z, arguments_dict):
        p_orient_rot_x = T_W_EEF_subgoal[:3, :3] @ x_goal_1_x
        p_orient_rot_z = T_W_EEF_subgoal[:3, :3] @ x_goal_2_z
        arguments_dict["x_goal_0"] = T_W_EEF_subgoal[:3, 3].tolist()
        arguments_dict["x_goal_1"] = p_orient_rot_x
        arguments_dict["x_goal_2"] = p_orient_rot_z
        return arguments_dict

    def get_local_goal(self, current_pos:np.ndarray, waypoint_list:list, arguments_dict:dict, x_goal_1_x:np.ndarray, x_goal_2_z:np.ndarray, goal_final=None):
        if goal_final is not None:
            goal_final_0 = goal_final["subgoal0"]["desired_position"]
            if self.euclidian_distance(current_pos, goal_final_0) < self.tolerance:
                arguments_dict["x_goal_0"] = goal_final["subgoal0"]["desired_position"]
                arguments_dict["x_goal_1"] = goal_final["subgoal1"]["desired_position"]
                arguments_dict["x_goal_2"] = goal_final["subgoal2"]["desired_position"]
                return arguments_dict
            else:
                T_W_EEF_subgoal, reference_poses = self.update_local_goal(current_pos=current_pos, waypoint_list=waypoint_list)
                arguments_dict = self.update_arguments_subgoal(T_W_EEF_subgoal, x_goal_1_x, x_goal_2_z, arguments_dict)
        return arguments_dict