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
import pybullet
import pytorch_kinematics as pk
import torch
import time

ROBOTTYPE = 'dingo_kinova'
ROBOTMODEL = 'dingo_kinova_gripper'

def initalize_environment(render=True, nr_obst: int = 0):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robot_model = RobotModel(ROBOTTYPE, model_name=ROBOTMODEL)

    # Robot urdf
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    URDF_FILE = absolute_path + "/urdfs/dinova/dinova.urdf"
    # urdf_file = robot_model.get_urdf_path()
    robots = [
        GenericUrdfReacher(urdf=URDF_FILE, mode="acc"),
        GenericUrdfReacher(urdf=URDF_FILE, mode="acc"),
    ]
    env: UrdfEnv = UrdfEnv(
        robots=robots,
        dt=0.01,
        render=render,
        observation_checking=False,
    )

 
    full_sensor_1 = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    full_sensor_2 = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    # Definition of the obstacle.
    # static_obst_dict = {
    #     "type": "sphere",
    #     "geometry": {"position": [0.3, -0.3, 0.3], "radius": 0.1},
    # }
    # obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # static_obst_dict = {
    #     "type": "sphere",
    #     "geometry": {"position": [-0.7, 0.0, 0.5], "radius": 0.1},
    # }
    # obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    goal_dict = {
        "subgoal0": {
            "weight": 3.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "world",
            "child_link": "arm_tool_frame",
            "desired_position": [0, -1, 0.5],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 5.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "arm_dummy_link",
            "child_link": "arm_tool_frame",
            "desired_position": [0, -1, 0.5],
            "epsilon": 0.03,
            "type": "staticSubGoal",
        },
        "subgoal2": {
            "weight": 5.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "arm_dummy_link",
            "child_link": "arm_orientation_helper_link",
            "desired_position": [0, -1, 0.5],
            "epsilon": 0.03,
            "type": "staticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    # obstacles = [obst1, obst2][0:nr_obst]

    pos0 =  np.array([
                        np.array([-0.75, 1, -np.pi/2, 0, 0, 0, 0, 0, 0, 0.9, -0.9]),
                        np.array([0.75, 1, -np.pi/2, 0, 0, 0, 0, 0, 0, 0.9, -0.9]),
                ])
    
    env.reset(pos=pos0)
    env.add_sensor(full_sensor_1, [0])
    env.add_sensor(full_sensor_2, [1])

    # for obst in obstacles:
    #     env.add_obstacle(obst)
    # for sub_goal in goal.sub_goals():
    #     env.add_goal(sub_goal)
    env.set_spaces()
    # collision_keys_robot0 = [8, 12, 13, 14, 16, 17, 18]
    # # for collision_link_nr in collision_keys_robot0:
    # for i in collision_keys_robot0:
    #     env.add_collision_link(0, i, shape_type='sphere', size=[0.10])

    return (env, goal)



def set_planner(goal: GoalComposition, nr_obst: int = 0, degrees_of_freedom: int = 6, i_robot: int = 0):
    """
    Initializes the fabric planner for the kuka robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    URDF_FILE = absolute_path + "/urdfs/dinova/dinova.urdf"
    # robot_model = RobotModel(ROBOTTYPE, model_name=ROBOTMODEL)
    # urdf_file = robot_model.get_urdf_path()
    with open(URDF_FILE, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="world",
        end_links=["arm_tool_frame", "arm_orientation_helper_link"],
    )
    collision_geometry =  "-0.01 / (x ** 1) * xdot ** 2"
    collision_finsler = "0.01/(x**2) * xdot**2"
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_finsler=collision_finsler,
        collision_geometry=collision_geometry
    )
    collision_links = [
        "chassis_link",
        "arm_forearm_link",
        "arm_lower_wrist_link",
        "arm_upper_wrist_link",
        "arm_end_effector_link"
    ]
    dingo_limits = np.array([
        [-10, 10],
        [-10, 10],
        [-10, 10]]
    )
    gen3lite_limits = np.array([
        [-154.1, 154.1],
        [150.1, 150.1],
        [150.1, 150.1],
        [-148.98, 148.98],
        [-144.97, 145.0],
        [-148.98, 148.98]
    ]) * np.pi/180
    dingo_kinova_limits = list(np.concatenate((dingo_limits, gen3lite_limits)))
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=nr_obst,
        number_plane_constraints=0,
        limits=dingo_kinova_limits,
    )
    planner.concretize()
    return planner


def run_kinova_example(n_steps=5000, render=True, dof=9):
    nr_obst = 5
    (env, goal) = initalize_environment(render, nr_obst=nr_obst)

    planner_dinova_1 = set_planner(goal, nr_obst, degrees_of_freedom=dof, i_robot=0)
    planner_dinova_2 = set_planner(goal, nr_obst, degrees_of_freedom=dof,  i_robot=1)
    action = np.zeros(2*dof)
    ob, *_ = env.step(action)

    # Table
    URDF_table = os.path.dirname(os.path.abspath(__file__)) + "/urdfs/table/table.urdf"
    URDF_cup_red = os.path.dirname(os.path.abspath(__file__)) + "/urdfs/cup/cup_red.urdf"
    URDF_cup_green = os.path.dirname(os.path.abspath(__file__)) + "/urdfs/cup/cup_green.urdf"

    urdf_links = {"URDF_table": URDF_table,
                  "URDF_cup_red" : URDF_cup_red,
                  "URDF_cup_green" : URDF_cup_green}
    z_table = 0.65*0.3
    table_position = [0, -1, 0.0]
    cup_red_position = [0, -1, z_table+0.05]
    cup_green_position = [0.1, -1, z_table+0.05]
    tableUid = pybullet.loadURDF(urdf_links["URDF_table"], basePosition=table_position,  globalScaling=0.3)
    cupUid = pybullet.loadURDF(urdf_links["URDF_cup_red"], basePosition=cup_red_position)
    cupUid = pybullet.loadURDF(urdf_links["URDF_cup_green"], basePosition=cup_green_position)


    # Forward kinematics for spheres
    robot_model = RobotModel(ROBOTTYPE, model_name=ROBOTMODEL)
    urdf_file = robot_model.get_urdf_path()
    chain = pk.build_serial_chain_from_urdf(open(urdf_file).read(), "arm_tool_frame")
    chain = chain.to(dtype=torch.float64, device="cpu")
    q_kinovas = torch.zeros((2, dof), dtype=torch.float64)
    collision_links = [
        "chassis_link",
        "arm_forearm_link",
        "arm_lower_wrist_link",
        "arm_upper_wrist_link",
        "arm_end_effector_link"
    ]

    rot_matrix = np.array([[-0.339, -0.784306, -0.51956],
                           [-0.0851341, 0.57557, -0.813309],
                           [0.936926, -0.23148, -0.261889]])
    x_goal_1_x = np.array([0.0, 0.0, 0.13])
    x_goal_2_z = np.array([0.0, 0.10, 0.00])
    p_orient_rot_x = rot_matrix @ x_goal_1_x
    p_orient_rot_z = rot_matrix @ x_goal_2_z
    weight_pose_goal = 1.0
    weight_orient_goal = 3.0
    
    for w in range(n_steps):
        ob_robot = ob['robot_0']
        ob_robot_2 = ob['robot_1']
        q_kinovas[0,:] = torch.as_tensor(ob_robot["joint_state"]["position"])
        q_kinovas[1,:] = torch.as_tensor(ob_robot_2["joint_state"]["position"])

        FK_W = chain.forward_kinematics(q_kinovas[:,:(dof-2)], end_only=False)
        obst_kinova_1, obst_kinova_2 = [], []
        for col_link in collision_links:
            obst_kinova_1.append( FK_W[col_link].get_matrix().numpy()[0,:3,3] )
            obst_kinova_2.append( FK_W[col_link].get_matrix().numpy()[1,:3,3] )

        arguments_dict_1 = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=cup_red_position,
            weight_goal_0= weight_pose_goal,
            x_goal_1 = p_orient_rot_x,
            weight_goal_1 =weight_orient_goal,
            x_goal_2 = p_orient_rot_z,
            weight_goal_2 = weight_orient_goal,
            x_obsts = obst_kinova_2,
            radius_obsts = [0.5, 0.1, 0.1, 0.1, 0.1],
            radius_body_chassis_link = 0.5,
            radius_body_arm_end_effector_link = 0.1,
            radius_body_arm_upper_wrist_link = 0.1,
            radius_body_arm_lower_wrist_link = 0.1,
            radius_body_arm_forearm_link=0.1,
            constraint_0=np.array([0, 0, 1, z_table])
        )

        arguments_dict_2 = dict(
            q=ob_robot_2["joint_state"]["position"],
            qdot=ob_robot_2["joint_state"]["velocity"],
            x_goal_0=cup_green_position,
            weight_goal_0= weight_pose_goal,
            x_goal_1 = p_orient_rot_x,
            weight_goal_1 = weight_orient_goal,
            x_goal_2 = p_orient_rot_z,
            weight_goal_2 = weight_orient_goal,
            x_obsts = obst_kinova_1,
            radius_obsts = [0.5, 0.1, 0.1, 0.1, 0.1],
            radius_body_chassis_link = 0.5,
            radius_body_arm_end_effector_link = 0.1,
            radius_body_arm_upper_wrist_link = 0.1,
            radius_body_arm_lower_wrist_link = 0.1,
            radius_body_arm_forearm_link=0.1,
            constraint_0=np.array([0, 0, 1, z_table])
        )
     
        action1 = planner_dinova_1.compute_action(**arguments_dict_1)
        action2 = planner_dinova_2.compute_action(**arguments_dict_2)

        action = np.concatenate((action1, action2), axis=None)
        action = np.clip(action, -2, 2)

        ob, *_ = env.step(action)

        
        
    env.close()
    return {}


if __name__ == "__main__":
    dof = 11
    res = run_kinova_example(n_steps=5000, dof=dof)


#TODO: [x] add clipping
#TODO: [x] add kinematic spheres for each robot
#TODO: [x] table fixed
#TODO: [x] add object
#TODO: [x] gripper
#TODO: [ ] dummy_axis for constraints
#TODO: [ ] create deadlocks
#TODO: does it make sense to use gpu for FK?
#TODO: increase damping