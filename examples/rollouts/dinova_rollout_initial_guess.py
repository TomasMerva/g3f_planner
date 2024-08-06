import os
import gymnasium as gym
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

import copy
import yaml

import pybullet
import pytorch_kinematics as pk
import torch
import time
from scipy.spatial.transform import Rotation as R
from grasp_planning import GOMP
from fabrics_rollouts import FabricsClient


# HOME_JOINT_CONFIG = np.array([-0.75, 1, -np.pi/2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])
HOME_JOINT_CONFIG = np.array([0,0, 0, 0, 0, 0, 0, 0, 0, 0.9, -0.9])

class Environment():
    def __init__(self) -> None:
        self.define_files_path()

    def define_files_path(self) -> None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        self.URDF_FOLDER = os.path.normpath( os.path.join(current_script_dir, '..', 'urdfs'))
        self.ROBOT_URDF_FILE = self.URDF_FOLDER + "/dinova/dinova.urdf"

        config_path = os.path.join(current_script_dir, '..', '..', 'config', 'dingo_kinova_config.yaml')
        self.CONFIG_FILE = os.path.normpath(config_path)

        with open(self.CONFIG_FILE, 'r') as config_file:
            self.CONFIG = yaml.safe_load(config_file)
            self.CONFIG_PROBLEM = self.CONFIG['problem']
            self.CONFIG_FABRICS = self.CONFIG['fabrics']

    def initialize(self, render : bool = True, nr_obst: int = 0) -> tuple:
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        steps the simulation once.
        """
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
        # Definition of the obstacle.
        self.nr_obstacles = len(self.CONFIG_PROBLEM["environment"]["obstacle_definition"])
        obstacles = []
        for obst_name, obst_param in self.CONFIG_PROBLEM["environment"]["obstacle_definition"].items():
            static_obst_dict = {
                "type": obst_param["type"],
                "geometry": {"position": obst_param["position"], "radius": obst_param["radius"]},
            }
            obstacles.append(SphereObstacle(name="staticObst", content_dict=static_obst_dict))
        


        goal = GoalComposition(name="goal", content_dict=self.CONFIG_PROBLEM["goal"]["goal_definition"])

        pos0 =  np.array([
                            HOME_JOINT_CONFIG
                        ])
        
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



def set_fk(robot_urdf_path):
    with open(robot_urdf_path, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="world",
        end_links=["arm_end_effector_link"], #TODO: read this from config file?
    )
    return forward_kinematics



def compute_rollout(planner, arg_dict, timesteps, dt):
    rollout_arg_dict = copy.deepcopy(arg_dict)
    q = rollout_arg_dict["q"]
    qdot = rollout_arg_dict["qdot"]
    alpha = 0.9
    q_list = []
    for i in range(timesteps):
        
        action = planner.compute_action(**rollout_arg_dict)
        
        dingo_vel_limit = 0.5
        if np.linalg.norm(action[0:2]) > dingo_vel_limit:
            action[0:2] = action[0:2] / np.linalg.norm(action[0:2]) * dingo_vel_limit
        action[2:] = np.clip(action[2:], -3, 3)


        qdot = alpha*qdot + (1-alpha)*action

        # print(action)
        q = q + qdot*dt
        
        rollout_arg_dict["q"] = q
        rollout_arg_dict["qdot"] = qdot
        q_list.append(q)

    return rollout_arg_dict["q"], q_list

def run_kinova_example(n_steps=5000, render=True, dof=9):
    nr_robots = 1
    nr_fingers = 2
    """
    1. Create environment
    """
    env = Environment()
    (sim, goal) = env.initialize(render)
    nr_obst = env.nr_obstacles
    # (objects_id, objects_position) = env.create_scene()
    pybullet.setGravity(0,0,0)
    action = np.zeros(nr_robots*dof)
    ob, *_ = sim.step(action)


    """
    2. Create fabrics
    """
    planner_dinova_1 = FabricsClient(env.CONFIG_FILE)
    weight_pose_goal = 0.5
    weight_orient_goal = 1.0
    rot_matrix = np.array([[-0.339, -0.784306, -0.51956],
                           [-0.0851341, 0.57557, -0.813309],
                           [0.936926, -0.23148, -0.261889]])
    x_goal_1_x = np.array([0.0, 0.0, 0.13])
    x_goal_2_z = np.array([0.0, 0.10, 0.00])
    # objects_position["cup_red"][2] += 0.05
  

    # """
    # 3. Create grasp planner
    # """
    T_W_RedCup = np.eye(4)
    T_W_RedCup[:3,:3] = R.from_euler("xyz", [0, 90, 0], degrees=True).as_matrix()
    # (grasp_planner_dinova_1, g_collision_names_d1) = set_grasp_planner(HOME_JOINT_CONFIG[:(dof-nr_fingers)])
    # id_grasp_obj = None
    chain = pk.build_serial_chain_from_urdf(open(env.ROBOT_URDF_FILE).read(), "arm_end_effector_link")
    chain = chain.to(dtype=torch.float64, device="cpu")
    # q_kinovas = torch.zeros((2, dof-2), dtype=torch.float64)


    """
    4. Main loop
    """
    for w in range(n_steps):
        # Read current state
        ob_robot = ob['robot_0']
        
        # Fabrics
        arguments_dict_1 = dict(
            q=ob_robot["joint_state"]["position"][0:(dof-nr_fingers)],
            qdot=ob_robot["joint_state"]["velocity"][0:(dof-nr_fingers)],
            x_goal_0=ob_robot['FullSensor']['goals'][nr_obst+2]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][nr_obst+2]['weight'],
            x_obsts=[ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
                    ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position']],
            radius_obsts=[ob_robot['FullSensor']['obstacles'][nr_obst + 0]['size'],
                        ob_robot['FullSensor']['obstacles'][nr_obst + 1]['size']],
            # x_goal_0=T_W_RedCup[:3,3],
            # weight_goal_0 = weight_pose_goal,
            # x_goal_1 = p_orient_rot_x_red,
            # weight_goal_1 =weight_orient_goal,
            # x_goal_2 = p_orient_rot_z_red,
            # weight_goal_2 = weight_orient_goal,
            # radius_obst_0 = 0.01,
            # x_obst_0 = objects_position["table"],
            radius_body_chassis_link = env.collision_links["chassis_link"],
            radius_body_arm_shoulder_link = env.collision_links["arm_shoulder_link"],
            radius_body_arm_end_effector_link = env.collision_links["arm_end_effector_link"],
            radius_body_arm_upper_wrist_link = env.collision_links["arm_upper_wrist_link"],
            radius_body_arm_lower_wrist_link = env.collision_links["arm_lower_wrist_link"],
            radius_body_arm_forearm_link = env.collision_links["arm_forearm_link"],
            # constraint_0=np.array([0, 0, 1, objects_position["z_table"]]),

        )
        start_time = time.perf_counter()
       
        timesteps = 100
        q_rollout, q_rollout_list = compute_rollout(planner_dinova_1, 
                                    arguments_dict_1, 
                                    timesteps=timesteps,
                                    dt=0.5)
        end_time = time.perf_counter()
        print("Computational time for rollouts:", end_time-start_time)

        q_kinovas = torch.zeros((timesteps+1, dof-2), dtype=torch.float64)
        q_kinovas[0,:] = torch.as_tensor(arguments_dict_1["q"])
        q_kinovas[1:,:] = torch.as_tensor(q_rollout_list)

        T_W_EEF = chain.forward_kinematics(q_kinovas , end_only=True)
        T_W_EEFs = T_W_EEF.get_matrix().numpy()
        p_W_EEF_actual = T_W_EEFs[0,:3,3]

        # for i in range(1, timesteps):
        #     pybullet.addUserDebugPoints([T_W_EEFs[i, :3, 3]], [[1, 0, 0]], 10, 1)
     
        pybullet.addUserDebugPoints(T_W_EEFs[1:, :3, 3].tolist(), [[1, 0, 0]]*timesteps, 5, 0.1)


        action[0:(dof-nr_fingers)] = planner_dinova_1.compute_action(**arguments_dict_1)
        dingo_vel_limit = 0.5
        if np.linalg.norm(action[0:2]) > dingo_vel_limit:
            action[0:2] = action[0:2] / np.linalg.norm(action[0:2]) * dingo_vel_limit
        action[2:] = np.clip(action[2:], -3, 3)

        p_W_EEF_horizon = T_W_EEFs[-1,:3,3]
        print("Actual:    ", p_W_EEF_actual)
        print("Predicted: ", p_W_EEF_horizon)
        print("desired:   ", ob_robot['FullSensor']['goals'][nr_obst+2]['position'])
        print()

        # pybullet.addUserDebugLine(p_W_EEF_actual, p_W_EEF_horizon,  [1, 0, 0], 1, 1)
        # print("act",action)
        # print()
        ob, *_ = sim.step(action)
        # time.sleep(0.1)
    sim.close()
    return {}


if __name__ == "__main__":
    dof = 11
    res = run_kinova_example(n_steps=5000, dof=dof)

