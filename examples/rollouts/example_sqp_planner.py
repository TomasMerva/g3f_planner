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
from mpscenes.goals.static_sub_goal import StaticSubGoal

import copy
import yaml

import pybullet

import time
from scipy.spatial.transform import Rotation as R

from fabrics_rollouts import GompSQP
from fabrics_rollouts import RolloutFabrics
from fabrics_rollouts.scripts.utils.reference_tracker import ReferenceTracker


HOME_JOINT_CONFIG =  np.array([0, 3, -np.pi/2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])

class Environment():
    def __init__(self) -> None:
        self.define_files_path()

    def define_files_path(self) -> None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.URDF_FOLDER = os.path.normpath( os.path.join(current_script_dir, '..', 'urdfs'))
        self.ROBOT_URDF_FILE = self.URDF_FOLDER + "/dinova/dinova.urdf"

        config_path = os.path.join(current_script_dir, '..', '..', 'config', 'dinova_config_full.yaml')
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
        
        pos0 = np.array(HOME_JOINT_CONFIG)
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

def set_runtime_weights(error, goal_weights_offline=None):
    weight_goal_0 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * goal_weights_offline[0]
    weight_goal_1 = 0.4 * (np.tanh(-4 * error + 2.0) + 1.5) * goal_weights_offline[1]
    weight_goal_2 = 0.5 * (np.tanh(-4 * error + 2.0) + 1.0) * goal_weights_offline[2]
    weight_goal_3 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * goal_weights_offline[3]
    return weight_goal_0, weight_goal_1, weight_goal_2, weight_goal_3


def run_kinova_example(n_steps=5000, render=True, dof=9):
    nr_robots = 1
    nr_fingers = 2
    nr_rollout_timesteps = 200
    nr_rollout_dt = 0.1
    nr_waypoints = 10
    nr_inner_optim = 10

    """
    1. Create environment
    """
    env = Environment()
    (sim, goal) = env.initialize(render)
    nr_obst = env.nr_obstacles
    pybullet.setGravity(0,0,0)
    (objects_id, objects_position) = env.create_scene()
    # objects_position["cup_red"][2] += 0.05
    action = np.zeros(nr_robots*dof)
    ob, *_ = sim.step(action)

    """
    2. Fabrics
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    fabrics_lib = os.path.normpath(os.path.join(current_script_dir, '../..', 'build', 'libfabrics_controller.so'))
    #TODO: remove
    subgoal0 = env.CONFIG_PROBLEM["goal"]["goal_definition"]["subgoal0"]["desired_position"]
    subgoal1 = env.CONFIG_PROBLEM["goal"]["goal_definition"]["subgoal1"]["desired_position"]
    subgoal2 = env.CONFIG_PROBLEM["goal"]["goal_definition"]["subgoal2"]["desired_position"]

    fk_args = dict(
    urdf_file = env.ROBOT_URDF_FILE,
    root_link = "world",
    end_link = "arm_tool_frame"
    )
    dinova_vel_limits = np.asarray(env.CONFIG_PROBLEM["joint_limits"]["velocity"], dtype=np.float32)

    rollouts_planner = RolloutFabrics(fk_dict=fk_args,
                                      config_file=env.CONFIG_FILE,
                                      controller_lib=fabrics_lib,
                                      vel_limits=dinova_vel_limits,
                                      dt=nr_rollout_dt,
                                      alpha_filter=0.7)
    
    #goal weights original:
    #TODO: PASSING FK this way is wrong, we have to reimplement it (either within rollout class or here)
    #TODO: but it would be nice to have one FK object that we share with other classes
    goal_operations = goalOperations(goal_composition=goal, forward_kinematics=rollouts_planner._robot_model._robot_fk)
    goal_weights_offline = [goal._config["subgoal"+str(i)]["weight"] for i in range(len(goal._config))]


    # Red cup
    x_goal_1_x = np.array([0.0, 0.0, 0.13])
    x_goal_2_z = np.array([0.0, 0.10, 0.00])
    T_Obj_GraspRed, T_W_RedCup = np.eye(4), np.eye(4)
    T_Obj_GraspRed[:3,:3] = R.from_euler("xyz", [0, 90, -90], degrees=True).as_matrix()
    redcup_pos, redcup_quat = pybullet.getBasePositionAndOrientation(objects_id["cup_red"])
    T_W_RedCup[:3,3] = np.asarray(redcup_pos)
    # T_W_RedCup[:3,:3] = R.from_quat(np.asarray(redcup_quat)).as_matrix()
    T_W_GraspRed = T_W_RedCup @ T_Obj_GraspRed
    T_W_RedCup[2,3] += 0.1  
    p_orient_rot_x_red = T_W_GraspRed[:3,:3] @ x_goal_1_x
    p_orient_rot_z_red = T_W_GraspRed[:3,:3] @ x_goal_2_z


    
    """
    4. 
    """
    x_init = np.ones((nr_waypoints, dof-2))
    f_q_prev = 5000.0

    gomp_args = dict(
        urdf_file = env.ROBOT_URDF_FILE,
        root_link = "world",
        end_link = "arm_tool_frame",
        num_waypoints = nr_waypoints,
        num_dim = dof-nr_fingers,
        joint_limits = env.CONFIG_PROBLEM["joint_limits"]
    )
    gomp_planner = GompSQP(gomp_args)
    gomp_planner.add_grasp_pos_constraint("g_grasp_pos", nr_waypoints-1, 0.0)
    gomp_planner.add_grasp_rot_constraint("g_grasp_rot", nr_waypoints-1, 0.0)
    for i in range(1, nr_waypoints):
        # obst1
        g_name = "g_col1_chassis_"+str(i)
        gomp_planner.add_collision_constraint(name= g_name, 
                                              waypoint_ID=i, 
                                              child_link="chassis_link",
                                              r_link= 0.5, 
                                              r_obst=0.2)
        gomp_planner.param_dict[g_name]["num_param"] = ob['robot_0']['FullSensor']['obstacles'][nr_obst]['position']
        g_name = "g_col1_upper_"+str(i)
        gomp_planner.add_collision_constraint(name= g_name, 
                                              waypoint_ID=i, 
                                              child_link="arm_upper_wrist_link",
                                              r_link= 0.2, 
                                              r_obst=0.2)
        gomp_planner.param_dict[g_name]["num_param"] = ob['robot_0']['FullSensor']['obstacles'][nr_obst]['position']
        # obst2
        g_name = "g_col2_chassis_"+str(i)
        gomp_planner.add_collision_constraint(name= g_name, 
                                              waypoint_ID=i, 
                                              child_link="chassis_link",
                                              r_link= 0.5, 
                                              r_obst=0.2)
        gomp_planner.param_dict[g_name]["num_param"] = ob['robot_0']['FullSensor']['obstacles'][nr_obst+1]['position']
        g_name = "g_col2_upper_"+str(i)
        gomp_planner.add_collision_constraint(name= g_name, 
                                              waypoint_ID=i, 
                                              child_link="arm_upper_wrist_link",
                                              r_link= 0.2, 
                                              r_obst=0.2)
        gomp_planner.param_dict[g_name]["num_param"] = ob['robot_0']['FullSensor']['obstacles'][nr_obst+1]['position']


    gomp_planner.param_dict["g_grasp_pos"]["num_param"][:3,3] = subgoal0
    gomp_planner.param_dict["g_grasp_rot"]["num_param"] = T_W_GraspRed

    gomp_planner.set_starting_state(q_start=HOME_JOINT_CONFIG[:9])    
    gomp_planner.setup_problem(x0=x_init)

    # reference tracker:
    reference_tracker = ReferenceTracker()

    # start_time = time.perf_counter()
    # q_result, solver_status = gomp_planner.solve(x_init.reshape(-1,1))
    # end_time = time.perf_counter()
    # print("elapsed time:", end_time-start_time)
    # print(q_result)

    for w in range(n_steps):
        ob_robot = ob['robot_0']
        q = ob_robot["joint_state"]["position"][0:dof-2]

        # adapt goal weights online:
        position_error = goal_operations.error(q)
        weight_goal_0, weight_goal_1, weight_goal_2, weight_goal_3 = set_runtime_weights(position_error, goal_weights_offline)
        theta_preference = goal_operations.get_theta_preference(q = q, goal_position=[-0.24355761, -0.75252747, 0.5])

        # compute arguments for the fabrics action
        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"][0:(dof-nr_fingers)],
            qdot=ob_robot["joint_state"]["velocity"][0:(dof-nr_fingers)],
            x_goal_0= subgoal0,
            weight_goal_0=weight_goal_0,
            x_goal_1= p_orient_rot_x_red,
            weight_goal_1=weight_goal_1,
            x_goal_2= p_orient_rot_z_red,
            weight_goal_2=weight_goal_2,
            x_goal_3=[theta_preference],
            weight_goal_3=weight_goal_3,
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

        # rollouts
        if w % 100 == 0:
            arguments_dict["weight_goal_0"] = 10.0
            arguments_dict["weight_goal_1"] = 20.0
            arguments_dict["weight_goal_2"] = 20.0
            q_rollout = rollouts_planner.compute_rollout(timesteps=nr_rollout_timesteps,
                                                         arg_dict=arguments_dict,
                                                         tolerance=0.15)
            q_fabrics_initial_guess = rollouts_planner.get_initial_guess(num_waypoints=nr_waypoints,
                                                                         rollout=q_rollout)

            if q_fabrics_initial_guess is not None:
                q_prev_solution = copy.deepcopy(q_fabrics_initial_guess)

                for i in range(nr_waypoints):
                    T_W_EEF = rollouts_planner._robot_model.compute_fk(q_fabrics_initial_guess[i,:])
                    pybullet.addUserDebugPoints([T_W_EEF[:3, 3].tolist()], [[0, 0, 1]], 7, 1)

                for i_optim in range(nr_inner_optim):
                    gomp_planner.set_starting_state(q_start=arguments_dict["q"])
                    gomp_planner.change_fixed_point(x0=q_prev_solution)
                    q_result, solver_status = gomp_planner.solve(q_prev_solution.reshape(-1,1))
                    if solver_status != "primal infeasible" and solver_status != "primal infeasible inaccurate":
                        f_q = gomp_planner.compute_cost(q_result.reshape(-1,1))
                        if (np.linalg.norm((f_q-f_q_prev), 2) <= 1e-3):
                            print("Absolute tolerance reached")
                        if (np.linalg.norm((f_q-f_q_prev), 2) <= 1e-3) or i_optim == (nr_inner_optim-1):
                            reference_poses = []
                            for i in range(nr_waypoints):
                                T_W_EEF = rollouts_planner._robot_model.compute_fk(q_result[i,:])
                                pybullet.addUserDebugPoints([T_W_EEF[:3, 3].tolist()], [[0, 1, 0]], 7, 1)
                                reference_poses.append(T_W_EEF)
                            break
                        else:
                            f_q_prev = copy.deepcopy(f_q)
                        q_prev_solution = copy.deepcopy(q_result)
            arguments_dict["weight_goal_0"] = weight_goal_0
            arguments_dict["weight_goal_1"] = weight_goal_1
            arguments_dict["weight_goal_2"] = weight_goal_2

        if solver_status != "primal infeasible" \
            and solver_status != "primal infeasible inaccurate" \
            and solver_status != "maximum iterations reached":
            current_pose = rollouts_planner._robot_model.compute_fk(q)
            arguments_dict = reference_tracker.get_local_goal(current_pos=current_pose[:3, 3],
                                                              waypoint_list=reference_poses,
                                                              arguments_dict=arguments_dict,
                                                              x_goal_1_x=x_goal_1_x,
                                                              x_goal_2_z=x_goal_2_z,
                                                              goal_final=env.CONFIG_PROBLEM["goal"]["goal_definition"])
            pybullet.addUserDebugPoints([arguments_dict["x_goal_0"]], [[1, 0, 1]], 15, 1)

        # clip actions

        action[0:(dof-nr_fingers)] = rollouts_planner.compute_action(**arguments_dict)
        if np.linalg.norm(action[0:2]) > dinova_vel_limits[0]:
            action[0:2] = action[0:2] / np.linalg.norm(action[0:2]) * dinova_vel_limits[:2]
        action[2:(dof-nr_fingers)] = np.clip(action[2:(dof-nr_fingers)], -1*dinova_vel_limits[2:], dinova_vel_limits[2:])

        ob, *_ = sim.step(action)
    sim.close()

    return {}


if __name__ == "__main__":
    dof = 11
    res = run_kinova_example(n_steps=5000, dof=dof)