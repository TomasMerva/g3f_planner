import numpy as np
import copy
import yaml
import time
import os
from scipy.spatial.transform import Rotation as R
import threading

from fabrics_rollouts import GompSQP
from fabrics_rollouts import RolloutFabrics
from fabrics_rollouts import ReferenceTracker


class RGF_Planner():
    def __init__(self, fk_args, config_file_path) -> None:
        self._fk_args = fk_args

        self._CONFIG_FILE_PATH = config_file_path
        with open(config_file_path, 'r') as config_file:
            self._CONFIG = yaml.safe_load(config_file)

        self._dinova_vel_limits = np.asarray(self._CONFIG["problem"]["joint_limits"]["velocity"], dtype=np.float32)
        self._T_W_Obj, self._T_W_StaticGrasp = np.eye(4), np.eye(4)
        self.z_offset_grasping = 0.05
        self._roll_obj_grasp =  np.deg2rad(self._CONFIG["gomp"]["initial_grasp_roll_deg"])
        self.theta_preference = 0.0
        
        _goal_config = self._CONFIG['problem']["goal"]["goal_definition"]
        self._goal_weights_offline = [_goal_config["subgoal"+str(i)]["weight"] for i in range(len(_goal_config))]


        self.num_dofs = self._fk_args["num_dofs"]
        self.num_waypoints = self._CONFIG["gomp"]["n_waypoints"]
        self.num_obstacles = self._CONFIG["problem"]["environment"]["number_spheres"]["static"]
        self.num_optim_steps = self._CONFIG["gomp"]["n_optim_steps"]
        self._obst_definition = self._CONFIG["problem"]["environment"]["obstacle_definition"]
        self.r_obsts = [self._obst_definition[obst]["radius"] for obst in self._obst_definition]
        _robot_collision_config = self._CONFIG["problem"]["robot_representation"]["collision_links"]
        self._r_fabrics = [_robot_collision_config[link]["sphere"]["radius"] for link in _robot_collision_config] 

        self.establish_rollouts()
        self.establish_planner()

        self._q_coll_init, self._q_free_init = None, None
        self._position_error = 0.0

    def establish_rollouts(self) -> None:
        _current_script_dir = os.path.dirname(os.path.abspath(__file__))
        _fabrics_lib = os.path.normpath(os.path.join(_current_script_dir,  '../../../build/', self._CONFIG["gomp"]["fabrics_lib"]))
        
        self._init_rollouts_parameters()
        self._rollouts_planner = RolloutFabrics(fk_dict=self._fk_args,
                                                config_file=self._CONFIG_FILE_PATH,
                                                controller_lib=_fabrics_lib,
                                                vel_limits=self._dinova_vel_limits,
                                                dt=self._CONFIG["gomp"]["rollout_dt"],
                                                alpha_filter=self._CONFIG["gomp"]["alpha_filter"])
   
    def establish_planner(self):
        gomp_args = dict(
            urdf_file = self._fk_args["urdf_file"],
            root_link = self._fk_args["root_link"],
            end_link = self._fk_args["end_link"],
            num_waypoints = self._CONFIG["gomp"]["n_waypoints"],
            num_dim = self._fk_args["num_dofs"],
            joint_limits = self._CONFIG["problem"]["joint_limits"]
        )

        self._gomp_planner = GompSQP(gomp_args)

        self._gomp_planner.add_grasp_pos_constraint("g_grasp_pos", self.num_waypoints-1, np.array([0.25, 0.25, 0.0]))
        self._gomp_planner.add_grasp_rot_constraint("g_grasp_rot", self.num_waypoints-1, 0.0)
        
        self._establish_obstacles()
        self._establish_collision_constraint()

        self._gomp_planner.set_starting_state(np.zeros(self.num_dofs))
        self._gomp_planner.setup_problem(x0=np.ones((self.num_waypoints, self.num_dofs)),
                                         verbose = self._CONFIG["gomp"]["verbose"]
                                         )


    def _establish_obstacles(self):
        self.obstacles = {}
        for id_obst in range(self.num_obstacles):
            self.obstacles["obst_"+str(id_obst)] = self.r_obsts[id_obst]

    def _establish_collision_constraint(self):
        self.collision_links = {}
        self.collision_constraint_names = []
        for link_idx, link_name in enumerate(self._CONFIG["gomp"]["collision_links"]):
            self.collision_links[link_name] = self._CONFIG["gomp"]["collision_link_radius"][link_idx]
        
        for id_way in range(1, self.num_waypoints):
            for id_obst in range(self.num_obstacles):
                for link_name, link_radius in self.collision_links.items():
                    name = "g_col_" + "way" + str(id_way) + "_" + link_name + "_obst" + str(id_obst)
                    self._gomp_planner.add_collision_constraint(name= name, 
                                                                waypoint_ID=id_way, 
                                                                child_link=link_name,
                                                                r_link= link_radius, 
                                                                r_obst = self.obstacles["obst_"+str(id_obst)]
                                                                )
                    self.collision_constraint_names.append(name)
                    # print(f"{name} = {self.obstacles['obst_'+str(id_obst)]}")
                    self._gomp_planner.param_dict[name]["num_param"] = np.array([100, 100, 100])
           
    def update_gomp_parameters(self, q_current, T_W_Obj, obst_pos=None):
        self._T_W_Obj[:3,3] = T_W_Obj[:3,3]
        self.theta_preference = self.compute_theta_preference(q_current, self._T_W_Obj)
        self._T_W_StaticGrasp = self.compute_static_grasp(self._T_W_Obj)

        self._gomp_planner.param_dict["g_grasp_pos"]["num_param"] = self._T_W_StaticGrasp  
        self._gomp_planner.param_dict["g_grasp_rot"]["num_param"] = self._T_W_StaticGrasp
        
        if obst_pos is not None:
            x_obsts = []
            for id_obst, x_obst in enumerate(obst_pos):
                x_obsts.append(x_obst)
                for id_way in range(1, self.num_waypoints):
                    for link_name, _ in self.collision_links.items():
                        name = "g_col_" + "way" + str(id_way) + "_" + link_name + "_obst" + str(id_obst)
                        self._gomp_planner.param_dict[name]["num_param"] = x_obst


    def error(self, goal_pos:np.ndarray, q_current:np.ndarray) -> float:
        fk_current = self.compute_fk(q_current)[:3,3]
        return np.linalg.norm(goal_pos-fk_current)

    def update_rollouts_parameters(self, joint_state, T_W_Obj, obst_pos=None, obst_radius=None):
        x_goal_1_x = np.array([0.0, 0.0, 0.13])
        x_goal_2_z = np.array([0.0, 0.10, 0.00])
        # self.theta_preference = self.compute_theta_preference(joint_state[0], self._T_W_Obj)
        # self._T_W_StaticGrasp = self.compute_static_grasp(T_W_Obj)
        p_orient_rot_x = self._T_W_StaticGrasp[:3,:3] @ x_goal_1_x
        p_orient_rot_z = self._T_W_StaticGrasp[:3,:3] @ x_goal_2_z

        self._position_error = self.error(goal_pos=self._T_W_StaticGrasp[:3,3],
                                    q_current=joint_state[0])
        weight_goal_0, weight_goal_1, weight_goal_2, weight_goal_3 = self.set_runtime_weights(self._position_error)

        self._rollouts_args_dict["q"] = joint_state[0]
        self._rollouts_args_dict["qdot"] = joint_state[1]
        self._rollouts_args_dict["x_goal_0"] = self._T_W_StaticGrasp[:3, 3]
        self._rollouts_args_dict["weight_goal_0"] = weight_goal_0
        self._rollouts_args_dict["x_goal_1"] = p_orient_rot_x
        self._rollouts_args_dict["weight_goal_1"] = weight_goal_1
        self._rollouts_args_dict["x_goal_2"] = p_orient_rot_z
        self._rollouts_args_dict["weight_goal_2"] = weight_goal_2
        self._rollouts_args_dict["x_goal_3"] = [self.theta_preference]
        self._rollouts_args_dict["weight_goal_3"] = weight_goal_3
        if obst_pos is not None:
            self._rollouts_args_dict["x_obsts"] = obst_pos
        if obst_radius is not None:
            self._rollouts_args_dict["radius_obsts"] = obst_radius

        self._rollouts_args_dict["T_W_Goal"] = self._T_W_StaticGrasp



    def set_runtime_weights(self, error):
        # weight_goal_0 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * self._goal_weights_offline[0]
        weight_goal_0 = 0.4 * (np.tanh(4 * error - 0.5) + 2.0) * self._goal_weights_offline[0]
        weight_goal_1 = 0.4 * (np.tanh(-4 * error + 2.0) + 1.5) * self._goal_weights_offline[1]
        weight_goal_2 = 0.5 * (np.tanh(-4 * error + 2.0) + 1.0) * self._goal_weights_offline[2]
        weight_goal_3 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * self._goal_weights_offline[3]
        return weight_goal_0, weight_goal_1, weight_goal_2, weight_goal_3
    

    def _init_rollouts_parameters(self):
        x_goal_1_x = np.array([0.0, 0.0, 0.13])
        x_goal_2_z = np.array([0.0, 0.10, 0.00])
        p_orient_rot_x = self._T_W_StaticGrasp[:3,:3] @ x_goal_1_x
        p_orient_rot_z = self._T_W_StaticGrasp[:3,:3] @ x_goal_2_z

        self._rollouts_args_dict  = dict(
            q=np.zeros(self.num_dofs),
            qdot=np.zeros(self.num_dofs),
            x_goal_0= self._T_W_StaticGrasp[:3, 3],
            weight_goal_0=self._goal_weights_offline[0],
            x_goal_1=p_orient_rot_x,
            weight_goal_1=self._goal_weights_offline[1],
            x_goal_2=p_orient_rot_z,
            weight_goal_2=self._goal_weights_offline[2],
            x_goal_3=[0.0],
            weight_goal_3=self._goal_weights_offline[3],
            x_obsts=[np.array([20., 20., 20.]) for _ in range(self.num_obstacles)],
            radius_obsts=[0.1]*self.num_obstacles, #TODO: this could be read from yaml file
            radius_body_chassis_link=self._r_fabrics[0],
            radius_body_arm_upper_wrist_link=self._r_fabrics[1],
        )
        # self._rollouts_args_dict  = dict(
        #         q=np.zeros(self.num_dofs),
        #         qdot=np.zeros(self.num_dofs),
        #         x_goal_0= self._T_W_StaticGrasp[:3, 3],
        #         weight_goal_0=10.,
        #         x_goal_1=p_orient_rot_x,
        #         weight_goal_1=5.,
        #         x_goal_2=p_orient_rot_z,
        #         weight_goal_2=20.,
        #         x_goal_3=[0.0],
        #         weight_goal_3=30.,
        #         x_obsts=[np.array([20., 20., 20.]) for _ in range(self.num_obstacles)],
        #         radius_obsts=[0.1]*self.num_obstacles, #TODO: this could be read from yaml file
        #         radius_body_chassis_link=0.6,
        #         radius_body_arm_shoulder_link=0.3,
        #         radius_body_arm_end_effector_link=0.1,
        #         radius_body_arm_upper_wrist_link=0.3,
        #         radius_body_arm_lower_wrist_link=0.3,
        #         radius_body_arm_forearm_link=0.3,
        #     )
        
    
    def compute_theta_preference(self, q_current, T_W_Obj):
        position_diff = T_W_Obj[:2,3] - q_current[:2]
        theta_preference = np.arctan2(position_diff[1], position_diff[0])
        if theta_preference < -np.pi:
            theta_preference += 2 * np.pi
        if theta_preference > 1 * np.pi:
            theta_preference -= 2 * np.pi
        return theta_preference
    
    def compute_static_grasp(self, T_W_Obj):
        T_Obj_Grasp = np.eye(4)
        T_Obj_Grasp[:3,:3] = R.from_euler('xyz', [0, self._roll_obj_grasp, 0], degrees=False).as_matrix()
        T_Grasp_Theta = np.eye(4)
        T_Grasp_Theta[:3,:3] = R.from_euler('xyz', [-self.theta_preference, 0, 0], degrees=False).as_matrix()
 
        # Compute correct rotation
        T_W_Grasp = T_W_Obj @ T_Obj_Grasp @ T_Grasp_Theta
        # Compute offset
        
        T_Grasp_Offset = np.eye(4)
        T_Grasp_Offset[:3, 3] = [-self.z_offset_grasping, 0, -0.05 ]
        return T_W_Grasp @ T_Grasp_Offset



    def _compute_initial_guesses(self):
        # Collision-free
        q_coll_rollout = self._rollouts_planner.compute_rollout(
                                                timesteps=self._CONFIG["gomp"]["rollout_timesteps"],
                                                arg_dict=self._rollouts_args_dict,
                                                tolerance=self._CONFIG["gomp"]["rollout_tolerance"]
                                                )
        if len(q_coll_rollout) <= 2: 
            q_coll_rollout = np.linspace(q_coll_rollout[0], q_coll_rollout[-1], self.num_waypoints, axis=0)
        q_coll_guess = self._rollouts_planner.get_initial_guess(num_waypoints=self.num_waypoints,
                                                                rollout=q_coll_rollout)


        # Obstacle-free
        arguments_dicts_free = copy.deepcopy(self._rollouts_args_dict)
        arguments_dicts_free["x_obsts"] = [np.array([2000., 2000., 2000.]) for _ in range(self.num_obstacles)]
        q_free_rollout = self._rollouts_planner.compute_rollout(
                                                timesteps=self._CONFIG["gomp"]["rollout_timesteps"],
                                                arg_dict=arguments_dicts_free,
                                                tolerance=self._CONFIG["gomp"]["rollout_tolerance"]
                                                )
   
        if len(q_free_rollout) <= 2: 
            q_free_rollout = np.linspace(q_free_rollout[0], q_free_rollout[-1], self.num_waypoints, axis=0)
        q_free_guess = self._rollouts_planner.get_initial_guess(num_waypoints=self.num_waypoints,
                                                                rollout=q_free_rollout)
        
        return (q_coll_guess, q_free_guess)


    
    def _solve_QP(self, q_init):
        if q_init is not None:
            self._gomp_planner.change_linear_term(q_init)
            q_result_prev = copy.deepcopy(q_init)
            for _ in range(self.num_optim_steps):
                self._gomp_planner.change_fixed_point(x0=q_result_prev)
                q_result, solver_status = self._gomp_planner.solve(q_result_prev.reshape(-1,1))
                if solver_status =="solved":
                    q_result_prev = q_result
            if solver_status == "solved":
                f_q_result = self._gomp_planner.compute_cost(q_result_prev.reshape(-1,1))[0].item()
            else:
                f_q_result = np.nan
            
            return q_result_prev, f_q_result
        else:
            return None, np.nan

    def update_param_and_initial_guess(self, joint_state, T_W_Obj, x_obsts=None, r_obsts=None):
        self._gomp_planner.set_starting_state(q_start=joint_state[0])

        self.update_gomp_parameters(joint_state[0], T_W_Obj, x_obsts)
        self.update_rollouts_parameters(joint_state, T_W_Obj, x_obsts, r_obsts)

        (self._q_coll_init, self._q_free_init) = self._compute_initial_guesses()
    
    def solve(self, joint_state, T_W_Obj, x_obsts=None, r_obsts=None):
        self.update_param_and_initial_guess(joint_state, T_W_Obj, x_obsts, r_obsts)

        _q_result_coll, f_q_coll = self._solve_QP(q_init=self._q_coll_init)
        _q_result_free, f_q_free = self._solve_QP(q_init=self._q_free_init)

        f_q_coll += 1.0
        q_results = {
            f_q_coll: _q_result_coll,
            f_q_free : _q_result_free
        }
        f_results = np.array([f_q_coll, f_q_free], dtype=object)
        if all(isinstance(x, float) and np.isnan(x) for x in f_results):
            solver_flag = False
            joint_waypoints = []
            return self._return_solution(joint_waypoints, solver_flag)
        else:
            best_f = np.nanmin(f_results)
            solver_flag = True
            joint_waypoints = q_results[best_f] 
        
        return self._return_solution(joint_waypoints, solver_flag)
    
    def _return_solution(self, joint_waypoints, solver_flag):
        pose_waypoints = []
        if solver_flag == False:
            for _ in range(self.num_waypoints):
                pose_waypoints.append(self._T_W_StaticGrasp)
            return pose_waypoints, solver_flag
        else:
            for id_way in range(self.num_waypoints):
                T_W_EEF = self.compute_fk(joint_waypoints[id_way,:], self._fk_args["end_link"])
                pose_waypoints.append(T_W_EEF)
        return pose_waypoints, solver_flag
    
    def compute_fk(self, q, end_link=None):
        if end_link is None:
            return self._gomp_planner.compute_fk(q, self._fk_args["end_link"])
        else:
            return self._gomp_planner.compute_fk(q)

    
    def get_initial_guesses(self):
        assert self._q_coll_init is not None and self._q_free_init is not None, "Initial guesses have not been computed"
        return (self._q_coll_init, self._q_free_init)

    def get_velocity_average(self):
        return self._rollouts_planner.get_rollout_velocity_avg()