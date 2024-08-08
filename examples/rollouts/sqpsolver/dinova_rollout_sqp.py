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
from solver.grasp_sqp import GompSQP


# HOME_JOINT_CONFIG = np.array([-0.75, 1, -np.pi/2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])
HOME_JOINT_CONFIG = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, -0.9])

class Environment():
    def __init__(self) -> None:
        self.define_files_path()

    def define_files_path(self) -> None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        self.URDF_FOLDER = os.path.normpath( os.path.join(current_script_dir, '../../', 'urdfs'))
        self.ROBOT_URDF_FILE = self.URDF_FOLDER + "/dinova/dinova.urdf"

        config_path = os.path.join(current_script_dir, '../../..', 'config', 'dingo_kinova_config.yaml')
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


class ForwardKinematics():
    def __init__(self, urdf_file, root_link, end_link) -> None:
        with open(urdf_file, "r") as file:
            urdf = file.read()
        self.fk = GenericURDFFk(
            urdf,
            root_link = root_link,
            end_links = end_link,
        )
        self._end_link = end_link
    
    def compute(self, q, end_link=None):
        if end_link is not None:
            return self.fk.numpy(q, end_link, position_only=False)
        else:
            return self.fk.numpy(q, self._end_link, position_only=False)

class Rollout():
    def __init__(self, planner, fk_dict, timesteps, dt) -> None:
        self._planner = planner
        self._timesteps = timesteps
        self._dt = dt

        self._alpha_filter = 0.7
        self.fk = ForwardKinematics(fk_dict["urdf_file"], fk_dict["root_link"], fk_dict["end_link"])
        

    def compute_rollout(self, arg_dict, tolerance) -> list:
        rollout_arg_dict = copy.deepcopy(arg_dict)
        q = rollout_arg_dict["q"]
        qdot = rollout_arg_dict["qdot"]
        q_rollout_record = []
        for _ in range(self._timesteps):
            action = self._planner.compute_action(**rollout_arg_dict)

            action = self.clip_actions(action)

            qdot = self.apply_low_pass_filter(action, qdot)
            q = q + qdot*self._dt

            rollout_arg_dict["q"] = q
            rollout_arg_dict["qdot"] = qdot
            q_rollout_record.append(q)

            error = self.compute_error(rollout_arg_dict["x_goal_0"], q)
            if error <= tolerance:
                return q_rollout_record
        return q_rollout_record

    
    def clip_actions(self, action):
        #TODO: change it since it is tailored for dinova
        dingo_vel_limit = 0.5
        if np.linalg.norm(action[0:2]) > dingo_vel_limit:
            action[0:2] = action[0:2] / np.linalg.norm(action[0:2]) * dingo_vel_limit
        action[2:] = np.clip(action[2:], -3, 3)
        return action

    def apply_low_pass_filter(self, x, x_prev):
        return self._alpha_filter*x_prev + (1-self._alpha_filter)*x
    

    def compute_error(self, x_goal, q_current):
        T_W_EEF = self.fk.compute(q_current)
        position_error = np.linalg.norm(x_goal - T_W_EEF[:3,3])
        return position_error

    def get_initial_guess(self, num_waypoints, rollout):
        if len(rollout) < 2:
            raise ValueError("Vector must have at least 2 elements.")
        
        indices = np.linspace(0, len(rollout) - 1, num_waypoints, dtype=int)

        waypoints = [rollout[i].tolist() for i in indices]
        return waypoints


def run_kinova_example(n_steps=5000, render=True, dof=9):
    nr_robots = 1
    nr_fingers = 2
    nr_rollout_timesteps = 100
    nr_rollout_dt = 0.1
    nr_waypoints = 5
    nr_inner_optim = 5
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

    """
    3. Rollouts
    """
    fk_args = dict(
        urdf_file = env.ROBOT_URDF_FILE,
        root_link = "world",
        end_link = "arm_end_effector_link"
    )
    rollouts = Rollout(planner_dinova_1,
                       fk_args,
                       nr_rollout_timesteps,
                       nr_rollout_dt)
    
    """
    4. 
    """
    gomp_args = dict(
        urdf_file = env.ROBOT_URDF_FILE,
        root_link = "world",
        end_link = "arm_end_effector_link",
        num_waypoints = nr_waypoints,
        num_dim = dof-nr_fingers,
        joint_limits = env.CONFIG_PROBLEM["joint_limits"]
    )
    gomp_planner = GompSQP(gomp_args)
    gomp_planner.add_grasp_pos_constraint("g_grasp_pos", nr_waypoints-1, 0.0)

    for i in range(nr_waypoints):
        gomp_planner.add_collision_constraint(name="g_col1_"+str(i), 
                                            waypoint_ID=i, 
                                            child_link="chassis_link",
                                            r_link= 0.6, 
                                            r_obst=0.2)
        gomp_planner.add_collision_constraint(name="g_col2_"+str(i), 
                                            waypoint_ID=i, 
                                            child_link="chassis_link",
                                            r_link= 0.6, 
                                            r_obst=0.2)
    # gomp_planner.param_ca_dict["g_grasp_pos"]["num_param"][:3,3] = [-1.24355761, -0.75252747, 0.5]
    x_init = np.ones((nr_waypoints, dof-2))
    gomp_planner.setup_problem(x0=x_init)




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
            radius_body_chassis_link = env.collision_links["chassis_link"],
            radius_body_arm_shoulder_link = env.collision_links["arm_shoulder_link"],
            radius_body_arm_end_effector_link = env.collision_links["arm_end_effector_link"],
            radius_body_arm_upper_wrist_link = env.collision_links["arm_upper_wrist_link"],
            radius_body_arm_lower_wrist_link = env.collision_links["arm_lower_wrist_link"],
            radius_body_arm_forearm_link = env.collision_links["arm_forearm_link"],
        )

        if w == 0:
            start_time = time.perf_counter()
            q_rollouts = rollouts.compute_rollout(arguments_dict_1, tolerance=0.01)
            end_time = time.perf_counter()
            print(f"Elapsed time: {end_time-start_time} s")
            print(f"Size of rollout: {len(q_rollouts)}")

            initial_guess = np.array(rollouts.get_initial_guess(num_waypoints=nr_waypoints, rollout=q_rollouts))
            waypoints_result = copy.deepcopy(initial_guess)

            
            for i in range(nr_waypoints):
                    T_W_EEF = rollouts.fk.compute(initial_guess[i,:])
                    pybullet.addUserDebugPoints([T_W_EEF[:3, 3].tolist()], [[0, 0, 1]], 5, 0)

            for i in range(nr_inner_optim):
                gomp_planner.set_starting_state(q_start=waypoints_result[0,:])
                gomp_planner.param_ca_dict["g_grasp_pos"]["num_param"][:3,3] = ob_robot['FullSensor']['goals'][nr_obst+2]['position']
                
                for i in range(nr_waypoints):
                    gomp_planner.param_ca_dict["g_col1_"+str(i)]["num_param"] = ob_robot['FullSensor']['obstacles'][nr_obst]['position']
                    gomp_planner.param_ca_dict["g_col2_"+str(i)]["num_param"] = ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position']

                gomp_planner.change_fixed_point(x0=waypoints_result)
                waypoints_result = gomp_planner.solve(waypoints_result.flatten())

                for i in range(nr_waypoints):
                    T_W_EEF = rollouts.fk.compute(waypoints_result[i,:])
                    pybullet.addUserDebugPoints([T_W_EEF[:3, 3].tolist()], [[1, 0, 0]], 10, 5)
                time.sleep(5)
            
            # draw final
            for i in range(nr_waypoints):
                T_W_EEF = rollouts.fk.compute(waypoints_result[i,:])
                pybullet.addUserDebugPoints([T_W_EEF[:3, 3].tolist()], [[1, 0, 0]], 5, 0)
 

        action[0:(dof-nr_fingers)] = planner_dinova_1.compute_action(**arguments_dict_1)
        action = rollouts.clip_actions(action)


        ob, *_ = sim.step(action)
        # time.sleep(0.1)
    sim.close()
    return {}


if __name__ == "__main__":
    dof = 11
    res = run_kinova_example(n_steps=5000, dof=dof)

