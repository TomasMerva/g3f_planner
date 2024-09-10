import pybullet
import numpy as np
import os
import yaml

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal

class Environment():
    def __init__(self) -> None:
        self._define_files_path()
        self.home_config = np.array([np.array([0, 3, -np.pi/2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9]),
                                     np.array([1.5, 3, -np.pi/2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])])
        


    def _define_files_path(self) -> None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.URDF_FOLDER = os.path.normpath( os.path.join(current_script_dir, 'urdfs'))
        self.ROBOT_URDF_FILE = self.URDF_FOLDER + "/dinova/dinova.urdf"

        config_path = os.path.join(current_script_dir, '..', 'config/dinova_config_full.yaml')
        self.CONFIG_FILE = os.path.normpath(config_path)

        with open(self.CONFIG_FILE, 'r') as config_file:
            self.CONFIG = yaml.safe_load(config_file)
            self.CONFIG_PROBLEM = self.CONFIG['problem']
            self.CONFIG_FABRICS = self.CONFIG['fabrics']

    
    def initialize(self, render : bool = True, nr_obst: int = 0, nr_robots: int=1, home_config=None) -> tuple:
        robots = [GenericUrdfReacher(urdf=self.ROBOT_URDF_FILE, mode="acc") for _ in range(nr_robots)]
        self.env: UrdfEnv = UrdfEnv(
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

        self.goal = GoalComposition(name="goal", content_dict=self.CONFIG_PROBLEM["goal"]["goal_definition"])
        
        # Definition of the obstacle.
        self.nr_obstacles = len(self.CONFIG_PROBLEM["environment"]["obstacle_definition"])
        
        self._obstacles_dict = {}
        obstacles = []
        for obst_name, obst_param in self.CONFIG_PROBLEM["environment"]["obstacle_definition"].items():
            static_obst_dict = {
                "type": obst_param["type"],
                "geometry": {"position": obst_param["position"], "radius": obst_param["radius"]},
            }
            obstacles.append(SphereObstacle(name="staticObst", content_dict=static_obst_dict))
            self._obstacles_dict[obst_name] = {
                "position" : obst_param["position"], 
                "radius": obst_param["radius"]
            }

        
        if home_config is not None:
            self.home_config = home_config
        pos0 = self.home_config[0:nr_robots]
        self.env.reset(pos=pos0)
        self.env.add_sensor(full_sensor, [0])
        for obst in obstacles:
            self.env.add_obstacle(obst)
        self.env.set_spaces()

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
                self.env.add_collision_link(0,
                                       pybullet_links_idx[link_name],
                                       shape_type='sphere',
                                       size=[val["sphere"]["radius"]])
                self.collision_links[link_name] = val["sphere"]["radius"]

        pybullet.setGravity(0,0,0)
        self.load_scene()
        return (self.env, self.goal)
    
    def load_scene(self) -> tuple:
        # Table
        URDF_table = self.URDF_FOLDER + "/table/table.urdf"
        URDF_cup_red = self.URDF_FOLDER + "/cup/cup_red.urdf"
        URDF_cup_green = self.URDF_FOLDER + "/cup/cup_green.urdf"

        urdf_links = {"URDF_table": URDF_table,
                    "URDF_cup_red" : URDF_cup_red,
                    "URDF_cup_green" : URDF_cup_green}
        z_table = 0.65*0.3
        self.scene_positions = {
            "z_table" : z_table,
            "table" : [0., -1., 0.0],
            "cup_red" : [-0.05, -0.9, z_table-0.01],
            "cup_green" : [0.05, -0.9, z_table-0.01],
        }

        tableUid = pybullet.loadURDF(urdf_links["URDF_table"], basePosition=self.scene_positions["table"],  globalScaling=0.3)
        cup_redUid = pybullet.loadURDF(urdf_links["URDF_cup_red"], basePosition=self.scene_positions["cup_red"])
        cup_greenUid = pybullet.loadURDF(urdf_links["URDF_cup_green"], basePosition=self.scene_positions["cup_green"])

        self.scene_id = {
            "table" : tableUid,
            "cup_red" : cup_redUid,
            "cup_green" : cup_greenUid
        }

        return (self.scene_id, self.scene_positions)
    
    def get_config_file_path(self):
        return self.CONFIG_FILE
    
    
    def get_redcup_pose(self):
        return pybullet.getBasePositionAndOrientation(self.scene_id["cup_red"])
    
    def get_greencup_pose(self):
        return pybullet.getBasePositionAndOrientation(self.scene_id["cup_green"])
    
    def get_table_pose(self):
        return pybullet.getBasePositionAndOrientation(self.scene_id["table"])
    

    def get_obstacles(self):
        return self._obstacles_dict