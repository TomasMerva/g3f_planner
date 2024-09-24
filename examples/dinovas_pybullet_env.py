import pybullet
import numpy as np
import os
import yaml
import random

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal

class Environment():
    def __init__(self, config_file="dinova_config_full.yaml") -> None:
        self._define_files_path(config_file)
        self.home_config = np.array([np.array([-0.75, 1, -np.pi/2, 0, 0, 0, 0, 0, 0, 0.9, -0.9]),
                                     np.array([0.75, 1, -np.pi/2, 0, 0, 0, 0, 0, 0, 0.9, -0.9])
                                     ])
        
        self._objects_pose_noise = None

    def _define_files_path(self, env_config_file) -> None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.URDF_FOLDER = os.path.normpath( os.path.join(current_script_dir, 'urdfs'))
        self.ROBOT_URDF_FILE = self.URDF_FOLDER + "/dinova/dinova.urdf"

        config_path = os.path.join(current_script_dir, '../config', env_config_file)
        self.CONFIG_FILE = os.path.normpath(config_path)

        with open(self.CONFIG_FILE, 'r') as config_file:
            self.CONFIG = yaml.safe_load(config_file)
            self.CONFIG_PROBLEM = self.CONFIG['problem']
            self.CONFIG_FABRICS = self.CONFIG['fabrics']

    
    def initialize(self, render : bool = True, 
                   nr_robots: int=1, 
                   home_config=None) -> tuple:
        self.n_robots = nr_robots
        robots_urdf = []
        for robot_id in range(self.n_robots):
            robots_urdf.append(self.URDF_FOLDER + "/dinova/dinova_" + str(robot_id+1) +".urdf")

        robots = [GenericUrdfReacher(urdf=robots_urdf[i], mode="acc") for i in range(nr_robots)]
        self.env: UrdfEnv = UrdfEnv(
            robots=robots,
            dt=0.01,
            render=render,
            observation_checking=False,
        )
        self._full_sensor = FullSensor(
                goal_mask=["position", "weight"],
                obstacle_mask=['position', 'size'],
                variance=0.0
        )

        self.goal = GoalComposition(name="goal", content_dict=self.CONFIG_PROBLEM["goal"]["goal_definition"])
        
        # Definition of the obstacle.
        self.nr_obstacles = len(self.CONFIG_PROBLEM["environment"]["obstacle_definition"])
        
        self._obstacles_dict = {}
        self._obstacles = []
        for obst_name, obst_param in self.CONFIG_PROBLEM["environment"]["obstacle_definition"].items():
            static_obst_dict = {
                "type": obst_param["type"],
                "geometry": {"position": obst_param["position"], "radius": obst_param["radius"]},
            }
            self._obstacles.append(SphereObstacle(name="staticObst", content_dict=static_obst_dict))
            self._obstacles_dict[obst_name] = {
                "position" : obst_param["position"], 
                "radius": obst_param["radius"]
            }

        
        if home_config is not None:
            self.home_config = home_config
        # self.home_config[0][:3] = [1.22466862, -1.72433948,  1.94785656]
        # self.home_config[1] = np.array([-0.188354, 0.84605903, -1.34784925, -0.01746023, -0.90232639,  1.16808536,  -0.05370079, -0.4273152,  -0.02838291, 0.9, -0.9])
        # self.home_config[2] = np.array([-0.64654884, -0.54696915,  0.47987815,  0.00676067, -1.0327904,   0.94264399, -0.07074774,  0.16436266, -0.07603307, 0.9, -0.9])

        pos0 = self.home_config[0:nr_robots]
        self.env.reset(pos=pos0)
        self.env.add_sensor(self._full_sensor, [0])
        for obst in self._obstacles:
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
                # self.env.add_collision_link(0,
                #                        pybullet_links_idx[link_name],
                #                        shape_type='sphere',
                #                        size=[val["sphere"]["radius"]])
                self.collision_links[link_name] = val["sphere"]["radius"]
        
        pybullet.setGravity(0,0,0)
        self.load_scene()
        return (self.env, self.goal)
    
    def load_scene(self) -> tuple:
        # Table
        if self.n_robots == 3:
            URDF_table = self.URDF_FOLDER + "/table_100x100/table_square.urdf"
            table_pos = [0., 0.0, 0.0]
            z_table = 0.3
            
            if self._objects_pose_noise is None:
                objects_pos = [
                    [table_pos[0]-0.45, table_pos[1]+0.4, z_table],
                    [table_pos[0]+0.45, table_pos[1]+0.4, z_table],
                    [table_pos[0]-0.45, table_pos[1]-0.4, z_table],
                    [table_pos[0]+0.45, table_pos[1]-0.4, z_table],
                ]
            else:
                objects_pos = [
                    [table_pos[0]+self._objects_pose_noise[0][0], table_pos[1]-self._objects_pose_noise[0][1], z_table],
                    [table_pos[0]+self._objects_pose_noise[1][0], table_pos[1]-self._objects_pose_noise[1][1], z_table],
                    [table_pos[0]+self._objects_pose_noise[2][0], table_pos[1]-self._objects_pose_noise[2][1], z_table],
                    [table_pos[0]+self._objects_pose_noise[3][0], table_pos[1]-self._objects_pose_noise[3][1], z_table],
                ]
        else:
            URDF_table = self.URDF_FOLDER + "/table_50x50/table_square.urdf"

            table_pos = [0., 0.0, 0.0]
            z_table = 0.3
            
            if self._objects_pose_noise is None:
                objects_pos = [
                    [table_pos[0]-0.05, table_pos[1]+0.1, z_table],
                    [table_pos[0]+0.05, table_pos[1]+0.1, z_table],
                    [table_pos[0]-0.05, table_pos[1]-0.1, z_table],
                    [table_pos[0]+0.05, table_pos[1]-0.1, z_table],
                ]
            else:
                objects_pos = [
                    [table_pos[0]+self._objects_pose_noise[0][0], table_pos[1]-self._objects_pose_noise[0][1], z_table],
                    [table_pos[0]+self._objects_pose_noise[1][0], table_pos[1]-self._objects_pose_noise[1][1], z_table],
                    [table_pos[0]+self._objects_pose_noise[2][0], table_pos[1]-self._objects_pose_noise[2][1], z_table],
                    [table_pos[0]+self._objects_pose_noise[3][0], table_pos[1]-self._objects_pose_noise[3][1], z_table],
                ]

        self.scene_id = {}
        for object_id in range(self.n_robots):
            urdf_file = self.URDF_FOLDER + "/cup/cup_" + str(object_id+1) +".urdf"
            object_pybulletID = pybullet.loadURDF(urdf_file, basePosition=objects_pos[object_id])
            self.scene_id["cup_"+str(object_id)] = object_pybulletID
        self.scene_id["table"] = pybullet.loadURDF(URDF_table, basePosition=table_pos,  globalScaling=1)

    def get_config_file_path(self):
        return self.CONFIG_FILE
    
    def get_cup(self, cup_id):
        name = "cup_" + str(cup_id)
        return pybullet.getBasePositionAndOrientation(self.scene_id[name])


    
    def get_table_pose(self):
        return pybullet.getBasePositionAndOrientation(self.scene_id["table"])
    

    def get_obstacles(self):
        return self._obstacles_dict
    
    def get_sim_handler(self):
        return self.env
    
    def set_objects_pos_noise(self, pos):
        assert self._objects_pose_noise == None, "Objects positions have already been set"
        self._objects_pose_noise = pos

    def reset(self):
        pos0 = self.home_config[0:self.n_robots]
        self.env.reset(pos=pos0)
        # self.env.add_sensor(self._full_sensor, [0])
        for obst in self._obstacles:
            self.env.add_obstacle(obst)
        self.env.set_spaces()
        pybullet.setGravity(0,0,0)
        self.load_scene()

    def set_obsts_pos(self, pos):
        obst_struct = self.CONFIG_PROBLEM["environment"]["obstacle_definition"]
        for i, obstacle_name in enumerate(obst_struct.keys()):
            if i > 0 and i <= len(pos):
                self.CONFIG_PROBLEM["environment"]["obstacle_definition"][obstacle_name]["position"] = pos[i-1]
            else:
                continue