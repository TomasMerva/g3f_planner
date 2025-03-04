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
import copy
import yaml
import pybullet
import shutil

HOME_JOINT_CONFIGS = np.array([np.array([0, 3, -np.pi/2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9]),
                              np.array([1.5, 3, -np.pi/2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])])

class Environment():
    def __init__(self) -> None:
        self.define_files_path()

    def define_files_path(self) -> None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.URDF_FOLDER = os.path.normpath( os.path.join(current_script_dir, '..', 'urdfs'))
        self.ROBOT_URDF_FILE = self.URDF_FOLDER + "/dinova/dinova.urdf"

        config_path = os.path.join(current_script_dir, '../..', 'config', 'dinova_config_rgf_5obst.yaml')
        # config_path = os.path.join('/home/tomas/repos/grasp_fabrics/config/evaluation/single_agent', 'dinova_config_rgf_3obst.yaml')
        self.CONFIG_FILE = os.path.normpath(config_path)

        with open(self.CONFIG_FILE, 'r') as config_file:
            self.CONFIG = yaml.safe_load(config_file)
            self.CONFIG_PROBLEM = self.CONFIG['problem']
            self.CONFIG_FABRICS = self.CONFIG['fabrics']

    
    def initialize(self, render : bool = True, nr_obst: int = 0, nr_robots: int = 1) -> tuple:
        robots = [GenericUrdfReacher(urdf=self.ROBOT_URDF_FILE, mode="acc") for _ in range(nr_robots)]
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
        
        pos0 = HOME_JOINT_CONFIGS
        env.reset(pos=pos0)
        env.add_sensor(full_sensor, [0])
        for obst in obstacles:
            env.add_obstacle(obst)
        # for sub_goal in goal.sub_goals():
        #     env.add_goal(sub_goal)
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
        for i_robot in range(nr_robots):
            for link_name, val in self.CONFIG_PROBLEM["robot_representation"]["collision_links"].items():
                env.add_collision_link(i_robot,
                                    pybullet_links_idx[link_name],
                                    shape_type='sphere',
                                    size=[val["sphere"]["radius"]])
                self.collision_links[link_name] = val["sphere"]["radius"]

        return (env, goal)

    def create_scene(self) -> tuple:
        # Table
        URDF_table = self.URDF_FOLDER + "/table/table.urdf"
        URDF_cup_red = self.URDF_FOLDER + "/cup/cup_red.urdf"
        URDF_cup_green = self.URDF_FOLDER + "/cup/cup_green.urdf"

        urdf_links = {"URDF_table": URDF_table,
                    "URDF_cup_red" : URDF_cup_red,
                    "URDF_cup_green" : URDF_cup_green}
        z_table = 0.65*0.3
        scene_positions = {
            "z_table" : z_table,
            "table" : [0., -1., 0.0],
            "cup_red" : [-0.05, -0.9, z_table-0.01],
            "cup_green" : [0.05, -0.9, z_table-0.01],
        }

        tableUid = pybullet.loadURDF(urdf_links["URDF_table"], basePosition=scene_positions["table"],  globalScaling=0.3)
        cup_redUid = pybullet.loadURDF(urdf_links["URDF_cup_red"], basePosition=scene_positions["cup_red"])
        cup_greenUid = pybullet.loadURDF(urdf_links["URDF_cup_green"], basePosition=scene_positions["cup_green"])

        scene_id = {
            "table" : tableUid,
            "cup_red" : cup_redUid,
            "cup_green" : cup_greenUid
        }

        return (scene_id, scene_positions)
    

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
        end_links=["arm_tool_frame", "arm_orientation_helper_link"],
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
    controller_file = "fabrics_controller_5obst.cpp"
    planner.export_as_c(controller_file)
    
    # Move to src folder
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    CONTROLLER_FOLDER_NEW = os.path.normpath(os.path.join(current_script_dir, "../../fabrics_rollouts/src/", controller_file))
    print("Controller exported to:", CONTROLLER_FOLDER_NEW)
    shutil.move(controller_file, CONTROLLER_FOLDER_NEW)
    return planner

def set_runtime_weights(error, goal_weights_offline=None):
    weight_goal_0 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * goal_weights_offline[0]
    weight_goal_1 = 0.4 * (np.tanh(-4 * error + 2.0) + 1.5) * goal_weights_offline[1]
    weight_goal_2 = 0.5 * (np.tanh(-4 * error + 2.0) + 1.0) * goal_weights_offline[2]
    weight_goal_3 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * goal_weights_offline[3]
    return weight_goal_0, weight_goal_1, weight_goal_2, weight_goal_3

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

def run_kinova_example(n_steps=5000, render=True, dof=9, nr_robots=2):
    nr_fingers = 2
    collision_links = [
        "chassis_link",
        "arm_forearm_link",
        "arm_lower_wrist_link",
        "arm_upper_wrist_link",
        "arm_end_effector_link",
        "arm_gripper_base_link"
    ]
    collision_radius = [0.4, 0.1, 0.1, 0.1, 0.1, 0.11]

    """
    1. Create environment
    """
    env = Environment()
    (sim, goal) = env.initialize(render, nr_robots=nr_robots)
    pybullet.setGravity(0,0,0)
    (objects_id, objects_position) = env.create_scene()
    redcup_pos, redcup_quat = pybullet.getBasePositionAndOrientation(objects_id["cup_red"])
    T_W_RedCup = np.eye(4)
    T_W_RedCup[:3,3] = np.asarray(redcup_pos)
    nr_obst = env.nr_obstacles
    action = np.zeros(nr_robots*dof)
    ob, *_ = sim.step(action)

    #TODO: remove
    subgoal0 = env.CONFIG_PROBLEM["goal"]["goal_definition"]["subgoal0"]["desired_position"]
    subgoal1 = env.CONFIG_PROBLEM["goal"]["goal_definition"]["subgoal1"]["desired_position"]
    subgoal2 = env.CONFIG_PROBLEM["goal"]["goal_definition"]["subgoal2"]["desired_position"]

    """
    2. Create fabrics
    """
    planner_dinova = set_planner(robot_urdf_path = env.ROBOT_URDF_FILE,
                                 config_dict = env.CONFIG,
                                 degrees_of_freedom = dof-nr_fingers)
    rot_matrix = np.array([[-0.339, -0.784306, -0.51956],
                           [-0.0851341, 0.57557, -0.813309],
                           [0.936926, -0.23148, -0.261889]])
    x_goal_1_x = np.array([0.0, 0.0, 0.13])
    x_goal_2_z = np.array([0.0, 0.10, 0.00])
    subgoal1 = rot_matrix @ x_goal_1_x
    subgoal2 = rot_matrix @ x_goal_2_z

    #goal weights original:
    goal_operations = goalOperations(goal_composition=goal, forward_kinematics=planner_dinova._forward_kinematics)
    goal_weights_offline = [goal._config["subgoal"+str(i)]["weight"] for i in range(len(goal._config))]


    # for w in range(n_steps):
    #     q_robots = []
    #     for i_robot in range(nr_robots):
    #         q_robots.append(ob["robot_"+str(i_robot)]["joint_state"]["position"][0:(dof-nr_fingers)])
    #         q_dinovas[i_robot,:] = torch.as_tensor(ob["robot_"+str(i_robot)]["joint_state"]["position"])

    #     # obstacle positions on other robot via FK
    #     FK_W = chain.forward_kinematics(q_dinovas[:,:(dof-nr_fingers)], end_only=False)
    #     obst_dinovas = {"robot_0": [], "robot_1":[]}
    #     for col_link in collision_links:
    #         for i_robot in range(nr_robots):
    #             obst_dinovas["robot_"+str(i_robot)].append( FK_W[col_link].get_matrix().numpy()[i_robot,:3,3] )

    #     # compute arguments for the fabrics action with environmental obstacles:
    #     x_obsts_env = []
    #     radius_obsts_env = []
    #     for i_obst in range(nr_obst):
    #         x_obsts_env.append(ob["robot_0"]['FullSensor']['obstacles'][nr_obst+nr_robots-1+i_obst]['position'])
    #         radius_obsts_env.append(ob["robot_0"]['FullSensor']['obstacles'][nr_obst + nr_robots-1+i_obst]['size'])

    #     for i_robot in range(nr_robots):
    #         if i_robot == 0:
    #             i_other_robot = 1
    #         else:
    #             i_other_robot = 0

    #         # adapt goal weights online:
    #         position_error = goal_operations.error(q_robots[i_robot])
    #         weight_goal_0, weight_goal_1, weight_goal_2, weight_goal_3 = set_runtime_weights(position_error,
    #                                                                                          goal_weights_offline)
    #         theta_preference = goal_operations.get_theta_preference(q=q_robots[i_robot],
    #                                                                 goal_position=[-0.24355761, -0.75252747, 0.5])

    #         arguments_dicts["robot_"+str(i_robot)] = dict(
    #             q=ob["robot_"+str(i_robot)]["joint_state"]["position"][0:(dof-nr_fingers)],
    #             qdot=ob["robot_"+str(i_robot)]["joint_state"]["velocity"][0:(dof-nr_fingers)],
    #             x_goal_0= subgoal0,
    #             weight_goal_0=weight_goal_0,
    #             x_goal_1= subgoal1,
    #             weight_goal_1=weight_goal_1,
    #             x_goal_2= subgoal2,
    #             weight_goal_2=weight_goal_2,
    #             x_goal_3=[theta_preference],
    #             weight_goal_3=weight_goal_3,
    #             x_obsts=[*x_obsts_env, *obst_dinovas["robot_"+str(i_other_robot)]],
    #             radius_obsts=[*radius_obsts_env, *collision_radius],
    #             radius_chassis_link=collision_radius[0],
    #             radius_arm_shoulder_link=collision_radius[1],
    #             radius_arm_end_effector_link = collision_radius[2],
    #             radius_arm_upper_wrist_link = collision_radius[3],
    #             radius_arm_lower_wrist_link = collision_radius[4],
    #             radius_arm_forearm_link=collision_radius[5],
    #         )

    #     # actions robot 0:
    #     action[0:(dof-nr_fingers)] = planner_dinova.compute_action(**arguments_dicts["robot_0"])
    #     if np.linalg.norm(action[0:2]) > dinova_vel_limits[0]:
    #         action[0:2] = action[0:2] / np.linalg.norm(action[0:2]) * dinova_vel_limits[:2]
    #     action[2:(dof-nr_fingers)] = np.clip(action[2:(dof-nr_fingers)], -1*dinova_vel_limits[2:], dinova_vel_limits[2:])

    #     # actions robot 1:
    #     if nr_robots>1:
    #         action[dof:(dof*2-nr_fingers)] = planner_dinova.compute_action(**arguments_dicts["robot_1"])
    #         action[dof+2:(dof*2 - nr_fingers)] = np.clip(action[dof+2:(dof*2 - nr_fingers)], -1 * dinova_vel_limits[2:],
    #                                            dinova_vel_limits[2:])
    #         if np.linalg.norm(action[dof:dof+2]) > dinova_vel_limits[0]:
    #             action[dof:dof+2] = action[dof:dof+2] / np.linalg.norm(action[dof:dof+2]) * dinova_vel_limits[0:2]

    #     ob, *_ = sim.step(action)
    # sim.close()
    # return {}


if __name__ == "__main__":
    dof = 11
    nr_robots = 2
    res = run_kinova_example(n_steps=5000, dof=dof, nr_robots=nr_robots)