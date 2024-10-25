import os
import gymnasium as gym
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import yaml
from scipy.spatial.transform import Rotation as R


class Fabrics():
    def __init__(self, robot_urdf_path, 
                       config_file_path, 
                       degrees_of_freedom: int = 9) -> None:
        
        self._num_dofs = degrees_of_freedom
        self._read_config_file(config_file_path)
        self._establish_fabrics(robot_urdf_path)
        self._init_arguments()

    def _read_config_file(self, config_file_path):
        self._CONFIG_FILE_PATH = config_file_path
        with open(config_file_path, 'r') as config_file:
            self._CONFIG = yaml.safe_load(config_file)
        
        _goal_config = self._CONFIG['problem']["goal"]["goal_definition"]
        self._goal_weights_offline = [_goal_config["subgoal"+str(i)]["weight"] for i in range(len(_goal_config))]
        self.num_obstacles = self._CONFIG["problem"]["environment"]["number_spheres"]["static"]

        self._collision_links = []
        self._collision_dict = self._CONFIG["problem"]["robot_representation"]["collision_links"]
        for coll_link in self._collision_dict:
            self._collision_links.append(self._collision_dict[coll_link]["sphere"]["radius"])

        self._vel_limits = np.asarray(self._CONFIG["problem"]["joint_limits"]["velocity"], dtype=np.float32)

    def _establish_fabrics(self, robot_urdf_path):
        with open(robot_urdf_path, "r", encoding="utf-8") as file:
            urdf = file.read()

        self._end_link = "arm_tool_frame"

        self._forward_kinematics = GenericURDFFk(
            urdf,
            root_link="world",
            end_links=[self._end_link , "arm_orientation_helper_link"],
        )
        

        
        base_metric = np.eye(9) * 0.3
        base_metric[0, 0] = 2
        base_metric[1, 1] = 2
        base_metric[2, 2] = 2
        base_energy = f"ca.dot(ca.mtimes(np.array({base_metric.tolist()}), xdot), xdot)"

        self._planner = ParameterizedFabricPlanner(
            self._num_dofs,
            self._forward_kinematics,
            base_energy=base_energy
        )

        self._planner.load_fabrics_configuration(self._CONFIG['fabrics'])
        self._planner.load_problem_configuration(self._CONFIG['problem'])
        self._planner.concretize()

    def set_runtime_weights(self, error):
        # weight_goal_0 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * self._goal_weights_offline[0]
        weight_goal_0 = 0.4 * (np.tanh(4 * error - 0.5) + 2.0) * self._goal_weights_offline[0]
        weight_goal_1 = 0.4 * (np.tanh(-4 * error + 2.0) + 1.5) * self._goal_weights_offline[1]
        weight_goal_2 = 0.5 * (np.tanh(-4 * error + 2.0) + 1.0) * self._goal_weights_offline[2]
        weight_goal_3 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * self._goal_weights_offline[3]
        return weight_goal_0, weight_goal_1, weight_goal_2, weight_goal_3
    
    def get_theta_preference(self, q:np.ndarray, goal_position:np.ndarray) ->  float:
        position_diff = goal_position[0:2] - q[0:2]
        theta_preference = np.arctan2(position_diff[1], position_diff[0])
        if theta_preference < -np.pi:
            theta_preference += 2 * np.pi
        if theta_preference > 1 * np.pi:
            theta_preference -= 2 * np.pi
        return float(theta_preference)
    
    def error(self, goal_pos:np.ndarray, q_current:np.ndarray) -> float:
        fk_current = self.compute_fk(q_current, self._end_link)[:3,3]
        return np.linalg.norm(goal_pos-fk_current)


    def _init_arguments(self):
        self._x_goal_1_x = np.array([0.0, 0.0, 0.13])
        self._x_goal_2_z = np.array([0.0, 0.10, 0.00])
        p_orient_rot_x = np.eye(3) @ self._x_goal_1_x
        p_orient_rot_z = np.eye(3) @ self._x_goal_2_z

        self._arguments_dict = dict(
            q= np.zeros(self._num_dofs),
            qdot = np.zeros(self._num_dofs),
            x_goal_0 = np.zeros(3),
            weight_goal_0= self._goal_weights_offline[0],
            x_goal_1 = p_orient_rot_x,
            weight_goal_1 = self._goal_weights_offline[1],
            x_goal_2 = p_orient_rot_z,
            weight_goal_2 = self._goal_weights_offline[2],
            x_goal_3 = [0.0],
            weight_goal_3 = self._goal_weights_offline[3],
            x_obsts = [np.array([20., 20., 20.]) for _ in range(self.num_obstacles)],
            radius_obsts = [0.1]*self.num_obstacles,
            radius_body_chassis_link = self._collision_links[0],
            radius_body_arm_shoulder_link = self._collision_links[1],
            radius_body_arm_forearm_link = self._collision_links[2],
            radius_body_arm_lower_wrist_link = self._collision_links[3],
            radius_body_arm_upper_wrist_link = self._collision_links[4],
            radius_body_arm_end_effector_link = self._collision_links[5],
        )


    def compute_dynamic_weights(self, q, T_W_Goal):
        position_error = self.error(goal_pos=T_W_Goal[:3,3],
                                    q_current=q)
        self.weight_goal_0, self.weight_goal_1, self.weight_goal_2, self.weight_goal_3 = self.set_runtime_weights(position_error)

    def update_arguments(self, joint_state, T_W_Goal, obst_pos=None, obst_radius=None):
        p_orient_rot_x = T_W_Goal[:3,:3] @ self._x_goal_1_x
        p_orient_rot_z = T_W_Goal[:3,:3] @ self._x_goal_2_z

        # position_error = self.error(goal_pos=T_W_Goal[:3,3],
        #                             q_current=joint_state[0])
        # weight_goal_0, weight_goal_1, weight_goal_2, weight_goal_3 = self.set_runtime_weights(position_error)
        theta_preference = self.get_theta_preference(q = joint_state[0], \
                                                     goal_position=T_W_Goal[:3,3])
        self._arguments_dict["q"] = joint_state[0]
        self._arguments_dict["qdot"] = joint_state[1]
        self._arguments_dict["x_goal_0"] = T_W_Goal[:3,3]
        self._arguments_dict["weight_goal_0"] = self.weight_goal_0
        self._arguments_dict["x_goal_1"] = p_orient_rot_x
        self._arguments_dict["weight_goal_1"] = self.weight_goal_1
        self._arguments_dict["x_goal_2"] = p_orient_rot_z
        self._arguments_dict["weight_goal_2"] = self.weight_goal_2
        self._arguments_dict["x_goal_3"] = [theta_preference]
        self._arguments_dict["weight_goal_3"] = self.weight_goal_3
        if obst_pos is not None:
            self._arguments_dict["x_obsts"] = obst_pos
        if obst_radius is not None:
            self._arguments_dict["radius_obsts"] = obst_radius

    def get_arguments(self):
        return self._arguments_dict

    def set_arguments(self, arguments_dict):
        self._arguments_dict = arguments_dict

    def compute_action(self):
        return self._planner.compute_action(**self._arguments_dict)

    
    def clip_action(self, action):
        if np.linalg.norm(action[0:2]) > self._vel_limits[0]:
            action[0:2] = action[0:2] / np.linalg.norm(action[0:2]) * self._vel_limits[:2]
        action[2:] = np.clip(action[2:], -1*self._vel_limits[2:], self._vel_limits[2:])

        return action
    
    def compute_fk(self, q, end_link=None):
        if end_link is not None:
            return self._forward_kinematics.numpy(q, end_link)
        else:
            return self._forward_kinematics.numpy(q, self._end_link)

    def collision_check(self, x_r_obsts, robot_states, threshold=-0.05):
        collision_link_names = list(self._collision_dict.keys())
        for i_robot in range(len(x_r_obsts)):
            robot_joint_positions = robot_states[i_robot][0]
            x_obsts = x_r_obsts["robot_"+str(i_robot)]["x_obsts"]
            r_obsts = x_r_obsts["robot_"+str(i_robot)]["r_obsts"]
            for i_obst in range(len(x_obsts)):
                for collision_link_name in collision_link_names:
                    x_collision_robot = self.compute_fk(robot_joint_positions, end_link=collision_link_name)[0:3, 3]
                    r_collision_robot = self._collision_dict[collision_link_name]["sphere"]["radius"]
                    error = np.linalg.norm(x_collision_robot - x_obsts[i_obst]) - r_obsts[i_obst] - r_collision_robot
                    if error <= threshold:
                        # print("A collision has occurred for robot ", str(i_robot), " with collision sphere ", str(collision_link_name), ".")
                        # print(f"Error: {error}")
                        # print(f"L2 norm: {np.linalg.norm(x_collision_robot - x_obsts[i_obst])}")
                        # print(f"R obst: {r_obsts[i_obst]}     r_robot: {r_collision_robot}")
                        # print()
                        return True
        return False


    def compute_static_grasp(self, T_W_Obj, theta_preference):
        T_Obj_Grasp = np.eye(4)
        T_Obj_Grasp[:3,:3] = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
        T_Grasp_Theta = np.eye(4)
        T_Grasp_Theta[:3,:3] = R.from_euler('xyz', [-theta_preference, 0, 0], degrees=False).as_matrix()
        T_W_Grasp = T_W_Obj @ T_Obj_Grasp @ T_Grasp_Theta

        T_Grasp_Offset = np.eye(4)
        T_Grasp_Offset[:3, 3] = [-0.05, 0, -0.05]
        return T_W_Grasp @ T_Grasp_Offset