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
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

import copy
import pybullet
import pytorch_kinematics as pk
import torch
import time
from scipy.spatial.transform import Rotation as R
from grasp_planning import GOMP





# Scene urdf
current_script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_folder_path = os.path.join(current_script_dir, '..', 'urdfs')
URDF_FOLDER = os.path.normpath(urdf_folder_path)
# Robot urdf
# ROBOTTYPE = 'dingo_kinova'
# ROBOTMODEL = 'dingo_kinova'
# robot_model = RobotModel(ROBOTTYPE, model_name=ROBOTMODEL)
ROBOT_URDF_FILE = URDF_FOLDER + "/dinova/dinova.urdf"

HOME_JOINT_CONFIG = np.array([-0.75, 1, -np.pi/2, 0, 0, 1.54, 0, 0, 0, 0.9, -0.9])


def initalize_environment(render=True, nr_obst: int = 0):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
   
   
    robots = [
        GenericUrdfReacher(urdf=ROBOT_URDF_FILE, mode="acc"),
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
                        HOME_JOINT_CONFIG
                ])
    
    env.reset(pos=pos0)
    env.add_sensor(full_sensor_1, [0])
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
    
    collision_links = {
        pybullet_links_idx["chassis_link"] : 0.4,
        pybullet_links_idx["arm_forearm_link"] : 0.1,
        pybullet_links_idx["arm_lower_wrist_link"] : 0.1,
        pybullet_links_idx["arm_upper_wrist_link"] : 0.1,
        pybullet_links_idx["arm_end_effector_link"] : 0.1,
        # pybullet_links_idx["arm_gripper_base_link"] : 0.11,
    }

    for key, value in collision_links.items():
        env.add_collision_link(0, key, shape_type='sphere', size=[value])

    return (env, goal)

def create_scene():
    # Table
    URDF_table = URDF_FOLDER + "/table/table.urdf"
    URDF_cup_red = URDF_FOLDER + "/cup/cup_red.urdf"
    URDF_cup_green = URDF_FOLDER + "/cup/cup_green.urdf"
    print(URDF_table)

    urdf_links = {"URDF_table": URDF_table,
                  "URDF_cup_red" : URDF_cup_red,
                  "URDF_cup_green" : URDF_cup_green}
    z_table = 0.65*0.3
    scene_positions = {
        "z_table" : z_table,
        "table" : [0, -1, 0.0],
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

def set_planner(goal: GoalComposition, nr_obst: int = 0, degrees_of_freedom: int = 6, collision_links: list = []):
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
    with open(ROBOT_URDF_FILE, "r", encoding="utf-8") as file:
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
        number_plane_constraints=1,
        limits=dingo_kinova_limits,
    )
    planner.concretize()
    return planner

def create_grasp_model():
    return pybullet.createCollisionShape(shapeType=pybullet.GEOM_MESH,
                                                            flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
                                                            fileName= absolute_path + "/urdfs/grasp_v1.stl",
                                                            meshScale=[1, 1, 1])

def display_grasp_pybullet(T_W_Grasp, grasp_shape):
    # Create a multi-body and set its position and orientation
    multi_body_id = pybullet.createMultiBody(baseMass=1,
                                        baseCollisionShapeIndex=grasp_shape,
                                        basePosition= T_W_Grasp[:3,3],
                                        baseOrientation=R.from_matrix(T_W_Grasp[:3,:3]).as_quat())
    return multi_body_id
    
#Remove q_current since it does not make sense to have it there
def set_grasp_planner(q_current):
    n_waypoints = 3
    grasp_waypoint_ID = 2
    planner = GOMP(n_waypoints=n_waypoints, 
                    urdf=ROBOT_URDF_FILE, 
                    roll_obj_grasp=np.pi/2,
                    root_link='world', 
                    end_link='arm_tool_frame')
    
    planner.set_init_guess(q_current)
    planner.set_boundary_conditions(q_start=q_current)
    planner.add_acc_objective_function()
    # Create constraints for grasping
    planner.add_grasp_pos_constraint(name="g_grasp_pos_1",
                                    waypoint_ID=grasp_waypoint_ID,
                                    tolerance=0.0)
    planner.add_grasp_rot_dof_constraint(name="g_grasp_rot_1",
                                        waypoint_ID=grasp_waypoint_ID,
                                        theta = 0.0, #I want to align the axis
                                        axis="x")
    # Create constraints for collision avoidance
    g_collision_names = []
    # for i in range(n_waypoints):        
    #     name = "g_col_"+ "arm_upper_wrist_link" + "_" + str(i)
    #     g_collision_names.append(name)
    #     planner.add_collision_constraint(name=name,
    #                                     waypoint_ID=i, 
    #                                     child_link="arm_upper_wrist_link", 
    #                                     r_link=0.3,
    #                                     r_obst=0.3,
    #                                     tolerance=0.01)
        
    # Setup problem
    planner.setup_problem(verbose=False) 
    return (planner, g_collision_names)
    

def run_kinova_example(n_steps=5000, render=True, dof=9):
    nr_obst = 0
    nr_fingers = 2
    nr_robots = 1
    """
    1. Create environment
    """
    (env, goal) = initalize_environment(render, nr_obst=nr_obst)
    pybullet.setGravity(0,0,0)
    action = np.zeros(2*dof)
    ob, *_ = env.step(action)
    # grasp_shape = create_grasp_model()
    (objects_id, objects_position) = create_scene()

    """
    2. Specify collisions
    """
    # Forward kinematics for spheres
    chain = pk.build_serial_chain_from_urdf(open(ROBOT_URDF_FILE).read(), "arm_tool_frame")
    chain = chain.to(dtype=torch.float64, device="cpu")
    q_kinovas = torch.zeros((nr_robots, dof), dtype=torch.float64)
    collision_links = [
        "chassis_link",
        "arm_forearm_link",
        "arm_lower_wrist_link",
        "arm_upper_wrist_link",
        "arm_end_effector_link",
    ]
    collision_radius = [0.4, 0.1, 0.1, 0.1, 0.1]


    """
    3. Create fabrics
    """
    planner_dinova_1 = set_planner(goal, nr_obst, degrees_of_freedom=dof, collision_links=collision_links)
    weight_pose_goal = 0.5
    weight_orient_goal = 1.0
    rot_matrix = np.array([[-0.339, -0.784306, -0.51956],
                           [-0.0851341, 0.57557, -0.813309],
                           [0.936926, -0.23148, -0.261889]])
    x_goal_1_x = np.array([0.0, 0.0, 0.13])
    x_goal_2_z = np.array([0.0, 0.10, 0.00])
    p_orient_rot_x = rot_matrix @ x_goal_1_x
    p_orient_rot_z = rot_matrix @ x_goal_2_z
    objects_position["cup_red"][2] += 0.05
  

    """
    3. Create grasp planner
    """
    T_W_RedCup = np.eye(4)
    (grasp_planner_dinova_1, g_collision_names_d1) = set_grasp_planner(HOME_JOINT_CONFIG[:(dof-nr_fingers)])
    id_grasp_obj = None
  


    """
    4. Main loop
    """
    for w in range(n_steps):
        # Read current state
        ob_robot = ob['robot_0']
        q_robot = np.asarray(ob_robot["joint_state"]["position"])
        q_kinovas[0,:] = torch.as_tensor(q_robot)
        
        # Grasping part
        if w%10 == 0:
            # Dinova 1 ------------------------------
            redcup_pos, redcup_quat = pybullet.getBasePositionAndOrientation(objects_id["cup_red"])
            T_W_RedCup[:3,3] = np.asarray(redcup_pos)
            T_W_RedCup[2,3] += 0.03
            constraint_param_dict_1 = {
                "g_grasp_pos_1" : T_W_RedCup,
                "g_grasp_rot_1" : T_W_RedCup
            }
            # constraint_param_dict_1[g_collision_names_d1[0]] = obst_kinova_2[3]
            grasp_planner_dinova_1.set_boundary_conditions(q_start=q_robot[:(dof-nr_fingers)])
            grasp_planner_dinova_1.update_constraints_params(constraint_param_dict_1)
            x_d1, solver_flag = grasp_planner_dinova_1.solve()
            print(f"Solver 1: {solver_flag}")

            T_W_GraspRed = grasp_planner_dinova_1._robot_model.eval_fk(x_d1[:,-1])
            p_orient_rot_x_red = T_W_GraspRed[:3,:3] @ x_goal_1_x
            p_orient_rot_z_red = T_W_GraspRed[:3,:3] @ x_goal_2_z
            if id_grasp_obj is not None:
                pybullet.removeBody(id_grasp_obj)
            # id_grasp_obj = display_grasp_pybullet(T_W_GraspRed, grasp_shape)



        # Fabrics
        arguments_dict_1 = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=T_W_GraspRed[:3,3],
            # x_goal_0=[2., 2., 0.5],
            weight_goal_0= weight_pose_goal,
            x_goal_1 = p_orient_rot_x_red,
            weight_goal_1 =weight_orient_goal,
            x_goal_2 = p_orient_rot_z_red,
            weight_goal_2 = weight_orient_goal,
            # x_obsts = obst_kinova_2,
            radius_obsts = collision_radius,
            radius_body_chassis_link = collision_radius[0],
            radius_body_arm_end_effector_link = collision_radius[1],
            radius_body_arm_upper_wrist_link = collision_radius[2],
            radius_body_arm_lower_wrist_link = collision_radius[3],
            radius_body_arm_forearm_link = collision_radius[4],
            # radius_body_arm_gripper_base_link = collision_radius[5],
            constraint_0=np.array([0, 0, 1, objects_position["z_table"]])
        )

      
     
        action1 = planner_dinova_1.compute_action(**arguments_dict_1)
        action = action1
        action = np.clip(action, -3, 3)

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
#TODO: [x] dummy_axis for constraints
#TODO: [x] get link pose
#TODO: [x] create deadlocks
#TODO: [ ] grasp planner
#TODO: [ ] debug and videos


#TODO: does it make sense to use gpu for FK?
#TODO: increase damping