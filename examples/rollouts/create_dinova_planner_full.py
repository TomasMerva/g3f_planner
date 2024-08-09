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
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import copy
import yaml


class Environment():
    def __init__(self) -> None:
        self.define_files_path()

    def define_files_path(self) -> None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        self.URDF_FOLDER = os.path.normpath( os.path.join(current_script_dir, '..', 'urdfs'))
        self.ROBOT_URDF_FILE = self.URDF_FOLDER + "/dinova/dinova.urdf"

        config_path = os.path.join(current_script_dir, '..', '..', 'config', 'dingo_kinova_config_full.yaml')
        self.CONFIG_FILE = os.path.normpath(config_path)

        with open(self.CONFIG_FILE, 'r') as config_file:
            self.CONFIG = yaml.safe_load(config_file)
            self.CONFIG_PROBLEM = self.CONFIG['problem']
            self.CONFIG_FABRICS = self.CONFIG['fabrics']

    
    def initialize(self, render : bool = True, nr_obst: int = 0) -> tuple:
        robots = [
        GenericUrdfReacher(urdf=self.ROBOT_URDF_FILE, mode="acc"),
        ]
        env: UrdfEnv = UrdfEnv(
            robots=robots,
            dt=0.01,
            render=render,
            observation_checking=False,
        )
        full_sensor = FullSensor(
                goal_mask=["position", "weight"],
                obstacle_mask=['position', 'size'],
                variance=0.0
        )

        goal = GoalComposition(name="goal", content_dict=self.CONFIG_PROBLEM["goal"]["goal_definition"])
        
        # Definition of the obstacle.
        self.nr_obstacles = len(self.CONFIG_PROBLEM["environment"]["obstacle_definition"])
        obstacles = []
        for obst_name, obst_param in self.CONFIG_PROBLEM["environment"]["obstacle_definition"].items():
            static_obst_dict = {
                "type": obst_param["type"],
                "geometry": {"position": obst_param["position"], "radius": obst_param["radius"]},
            }
            obstacles.append(SphereObstacle(name="staticObst", content_dict=static_obst_dict))
        
        pos0 = np.zeros((9,))
        env.reset(pos=pos0)
        env.add_sensor(full_sensor, [0])
        for obst in obstacles:
            env.add_obstacle(obst)
        for sub_goal in goal.sub_goals():
            env.add_goal(sub_goal)
        env.set_spaces()

        pybullet_links_idx = {
            'world': -1, 
            'base_link': 0, 
            'base_link_x': 1, 
            'base_link_y': 2, 
            'chassis_link': 3, 
            'front_left_wheel_link': 4,
            'front_right_wheel_link': 5, 
            'rear_left_wheel_link': 6, 
            'rear_right_wheel_link': 7, 
            'mid_mount': 8, 
            'front_c_mount': 9, 
            'front_b_mount': 10, 
            'front_mount': 11, 
            'arm_base_link': 12, 
            'arm_shoulder_link': 13, 
            'arm_arm_link': 14, 
            'arm_forearm_link': 15, 
            'arm_lower_wrist_link': 16, 
            'arm_upper_wrist_link': 17, 
            'arm_end_effector_link': 18, 
            'arm_dummy_link': 19, 
            'arm_tool_frame': 20, 
            'arm_orientation_helper_link': 21, 
            'arm_gripper_base_link': 22, 
            'arm_right_finger_prox_link': 23, 
            'arm_left_finger_prox_link': 24, 
            'rear_c_mount': 25, 
            'rear_b_mount': 26, 
            'rear_mount': 27, 
            'front_bumper_mount': 28
            }
        
        self.collision_links = {}
        for link_name, val in self.CONFIG_PROBLEM["robot_representation"]["collision_links"].items():
            env.add_collision_link(0, 
                                pybullet_links_idx[link_name], 
                                shape_type='sphere', 
                                size=[val["sphere"]["radius"]])
            self.collision_links[link_name] = val["sphere"]["radius"]

        return (env, goal)

class goalOperations():
    def __init__(self, goal_composition: GoalComposition, forward_kinematics:GenericURDFFk):
        self._goal_composition = goal_composition
        self._fk = forward_kinematics

    def goal_vector(self, q:np.ndarray, subgoal: StaticSubGoal) -> np.ndarray:
        if isinstance(q, np.ndarray):
            fk_parent = self._fk.numpy(q, subgoal.parent_link(), position_only=True)
            fk_child = self._fk.numpy(q, subgoal.child_link(), position_only=True)
            return fk_child - fk_parent
        return np.zeros(3)

    def error(self, q:np.ndarray) -> float:
        if q is not None:
            position_error = np.linalg.norm(
                self._goal_composition.primary_goal().position() \
                - self.goal_vector(q, self._goal_composition.sub_goals()[0])
            )
            # orientation_1_error = np.linalg.norm(
            #     self._goal_composition.sub_goals()[1].position() \
            #     - self.goal_vector(q, self._goal_composition.sub_goals()[1])
            # )
            # orientation_2_error = np.linalg.norm(
            #     self._goal_composition.sub_goals()[2].position() \
            #     - self.goal_vector(q, self._goal_composition.sub_goals()[2])
            # )
            index = 0
            target_position = self._goal_composition.sub_goals()[index].position()
            actual_position = self.goal_vector(q, self._goal_composition.sub_goals()[index])
            error = np.linalg.norm(target_position - actual_position)
            return position_error #+ orientation_1_error + orientation_2_error
        else:
            return 1

    def get_theta_preference(self, q:np.ndarray, goal_position:np.ndarray) ->  float:
        position_diff = goal_position[0:2] - q[0:2]
        theta_preference = np.arctan2(position_diff[1], position_diff[0])
        if theta_preference < -np.pi:
            theta_preference += 2 * np.pi
        if theta_preference > 1 * np.pi:
            theta_preference -= 2 * np.pi
        return float(theta_preference)

def set_planner(robot_urdf_path, config_dict, degrees_of_freedom: int = 9):
    with open(robot_urdf_path, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="world",
        end_links=["arm_end_effector_link"],
    )
    base_metric = np.eye(9) * 0.3
    base_metric[0, 0] = 2
    base_metric[1, 1] = 2
    base_metric[2, 2] = 2
    base_energy = f"ca.dot(ca.mtimes(np.array({base_metric.tolist()}), xdot), xdot)"

    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        base_energy=base_energy
    )

    planner.load_fabrics_configuration(config_dict['fabrics'])
    planner.load_problem_configuration(config_dict['problem'])
    planner.concretize()
    planner.export_as_c("pure_controller.c")
    return planner

def set_runtime_weights(error, goal_weights_offline=None):
    weight_goal_0 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * goal_weights_offline[0]
    weight_goal_1 = 0.4 * (np.tanh(-4 * error + 2.0) + 1.5) * goal_weights_offline[1]
    weight_goal_2 = 0.5 * (np.tanh(-4 * error + 2.0) + 1.0) * goal_weights_offline[2]
    # weight_goal_3 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * goal_weights_offline[3]
    return weight_goal_0, weight_goal_1, weight_goal_2 #, weight_goal_3


def run_kinova_example(n_steps=5000, render=True, dof=9):
    nr_fingers = 2
    """
    1. Create environment
    """
    env = Environment()
    (sim, goal) = env.initialize(render)
    nr_obst = env.nr_obstacles
    action = np.zeros(dof)
    ob, *_ = sim.step(action)

    """
    2. Create fabrics
    """
    planner_dinova = set_planner(robot_urdf_path = env.ROBOT_URDF_FILE,
                                   config_dict = env.CONFIG,
                                   degrees_of_freedom = dof-nr_fingers)

    #goal weights original:
    goal_operations = goalOperations(goal_composition=goal, forward_kinematics=planner_dinova._forward_kinematics)
    goal_weights_offline = [goal._config["subgoal"+str(i)]["weight"] for i in range(len(goal._config))]

    # max velocities:
    dinova_vel_limits = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100., 100.])

    for w in range(n_steps):
        ob_robot = ob['robot_0']
        q = ob_robot["joint_state"]["position"][0:dof-2]

        # adapt goal weights online:
        position_error = goal_operations.error(q)
        weight_goal_0, weight_goal_1, weight_goal_2 = set_runtime_weights(position_error, goal_weights_offline)
        theta_preference = goal_operations.get_theta_preference(q=q, goal_position=ob_robot['FullSensor']['goals'][nr_obst+2]['position'])

        # compute arguments for the fabrics action
        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"][0:(dof-nr_fingers)],
            qdot=ob_robot["joint_state"]["velocity"][0:(dof-nr_fingers)],
            x_goal_0=ob_robot['FullSensor']['goals'][nr_obst+2]['position'],
            weight_goal_0=weight_goal_0,
            x_goal_1=ob_robot['FullSensor']['goals'][nr_obst + 3]['position'],
            weight_goal_1=weight_goal_1,
            x_goal_2=ob_robot['FullSensor']['goals'][nr_obst + 4]['position'],
            weight_goal_2=weight_goal_2,
            # x_goal_3=[theta_preference],
            # weight_goal_3=weight_goal_3,
            x_obsts=[ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
                    ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position']],
            radius_obsts=[ob_robot['FullSensor']['obstacles'][nr_obst + 0]['size'],
                        ob_robot['FullSensor']['obstacles'][nr_obst + 1]['size']],
            radius_body_chassis_link=0.4,
            radius_body_arm_shoulder_link=0.1,
            radius_body_arm_end_effector_link = 0.1,
            radius_body_arm_upper_wrist_link = 0.1,
            radius_body_arm_lower_wrist_link = 0.1,
            radius_body_arm_forearm_link=0.1,
        )

        action[0:(dof-nr_fingers)] = planner_dinova.compute_action(**arguments_dict)
        np.clip(action, -1*dinova_vel_limits, dinova_vel_limits)
        ob, *_ = sim.step(action)
    sim.close()
    return {}


if __name__ == "__main__":
    dof = 11
    res = run_kinova_example(n_steps=5000, dof=dof)