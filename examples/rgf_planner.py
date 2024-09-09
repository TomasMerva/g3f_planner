import numpy as np
import copy
import yaml
import time
import os
from scipy.spatial.transform import Rotation as R


from fabrics_rollouts import GompSQP
from fabrics_rollouts import RolloutFabrics
from fabrics_rollouts import ReferenceTracker


class RGF_Planner():
    def __init__(self, fk_args, config_file) -> None:
        self._fk_args = fk_args
        self._CONFIG = config_file
        self._dinova_vel_limits = np.asarray(self._CONFIG["problem"]["joint_limits"]["velocity"], dtype=np.float32)
        self._T_W_Obj, self._T_W_StaticGrasp = np.eye(4), np.eye(4)
        self.z_offset_grasping = 0.05
        self._roll_obj_grasp =  np.deg2rad(self._CONFIG["gomp"]["initial_grasp_roll_deg"])
        
        self.num_dofs = self._fk_args["num_dofs"]
        self.num_waypoints = self._CONFIG["gomp"]["n_waypoints"]
        self.num_obstacles = self._CONFIG["gomp"]["n_obstacles"]
        self.num_optim_steps = self._CONFIG["gomp"]["n_optim_steps"]


        self.establish_rollouts()
        self.establish_planner()

    def establish_rollouts(self) -> None:
        _current_script_dir = os.path.dirname(os.path.abspath(__file__))
        _fabrics_lib = os.path.normpath(os.path.join(_current_script_dir,  '../build/libfabrics_controller.so'))
        
        self._init_rollouts_parameters()
        self._rollouts_planner = RolloutFabrics(fk_dict=self._fk_args,
                                                config_file=self._CONFIG,
                                                controller_lib=_fabrics_lib,
                                                vel_limits=self._dinova_vel_limits,
                                                dt=self._CONFIG["gomp"]["rollout_dt"],
                                                alpha_filter=self._CONFIG["gomp"]["alpha_filter"])

    def establish_planner(self):
        gomp_args = dict(
            urdf_file = self._fk_args["urdf_file"],
            root_link = self._fk_args["world"],
            end_link = self._fk_args["arm_tool_frame"],
            num_waypoints = self._CONFIG["gomp"]["n_waypoints"],
            num_dim = self._fk_args["num_dofs"],
            joint_limits = self._CONFIG["problem"]["joint_limits"]
        )

        self._gomp_planner = GompSQP(gomp_args)

        self._gomp_planner.add_grasp_pos_constraint("g_grasp_pos", self.num_waypoints-1, np.zeros(3))
        self._gomp_planner.add_grasp_rot_constraint("g_grasp_rot", self.num_waypoints-1, np.zeros(3))
        
        self._establish_obstacles()
        self._establish_collision_constraint()

        self._gomp_planner.set_starting_state(np.zeros(self.num_dofs))
        self._gomp_planner.setup_problem(x0=np.ones((self.num_waypoints, self.num_dofs)),
                                         verbose = self._CONFIG["gomp"]["verbose"]
                                         )


    def _establish_obstacles(self):
        self.obstacles = {}
        obstacle_radius = self._CONFIG["gomp"]["obstacle_radius"]
        for id_obst in range(self.num_obstacles):
            self.obstacles["obst_"+str(id_obst)] = obstacle_radius[id_obst]

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
                    self._gomp_planner.param_dict[name]["num_param"] = np.array([100, 100, 100])
        
    def update_gomp_parameters(self, q_current, obst_pos, T_W_Obj):
        self._T_W_Obj[:3,3] = T_W_Obj[:3,3]
        self.theta_preference = self.compute_theta_preference(q_current, self._T_W_Obj)
        self._T_W_StaticGrasp = self.compute_static_grasp(self._T_W_Obj)


        self._gomp_planner.param_dict["g_grasp_pos"]["num_param"] = self._T_W_StaticGrasp
        self._gomp_planner.param_dict["g_grasp_rot"]["num_param"] = self._T_W_StaticGrasp
        
        x_obsts = []
        for id_obst, x_obst in enumerate(obst_pos):
            x_obsts.append(x_obst)
            for id_way in range(1, self.num_waypoints):
                for link_name, _ in self.collision_links.items():
                    name = "g_col_" + "way" + str(id_way) + "_" + link_name + "_obst" + str(id_obst)
                    self._gomp_planner.param_dict[name]["num_param"] = x_obst
        


    def update_rollouts_parameters(self, q_current, obst_pos, T_W_Obj):
        x_goal_1_x = np.array([0.0, 0.0, 0.13])
        x_goal_2_z = np.array([0.0, 0.10, 0.00])
        self._T_W_StaticGrasp = self.compute_static_grasp(T_W_Obj)
        p_orient_rot_x = self._T_W_StaticGrasp[:3,:3] @ x_goal_1_x
        p_orient_rot_z = self._T_W_StaticGrasp[:3,:3] @ x_goal_2_z

        self._rollouts_args_dict["q"] = q_current
        self._rollouts_args_dict["x_goal_0"] = self._T_W_StaticGrasp[:3, 3]
        self._rollouts_args_dict["x_goal_1"] = p_orient_rot_x
        self._rollouts_args_dict["x_goal_2"] = p_orient_rot_z
        self._rollouts_args_dict["x_goal_3"] = [self.theta_preference]

        self._rollouts_args_dict["x_obsts"] = obst_pos



    def _init_rollouts_parameters(self):
        x_goal_1_x = np.array([0.0, 0.0, 0.13])
        x_goal_2_z = np.array([0.0, 0.10, 0.00])
        p_orient_rot_x = self._T_W_StaticGrasp[:3,:3] @ x_goal_1_x
        p_orient_rot_z = self._T_W_StaticGrasp[:3,:3] @ x_goal_2_z

        self._rollouts_args_dict  = dict(
                q=np.zeros(self.num_dofs),
                qdot=np.zeros(self.num_dofs),
                x_goal_0= self._T_W_StaticGrasp[:3, 3],
                weight_goal_0=10.,
                x_goal_1=p_orient_rot_x,
                weight_goal_1=5.,
                x_goal_2=p_orient_rot_z,
                weight_goal_2=20.,
                x_goal_3=[0.0],
                weight_goal_3=30.,
                x_obsts=[np.array([20., 20., 20.]) for _ in range(self.num_obstacles)],
                radius_obsts=[0.1]*self.num_obstacles, #TODO: this could be read from yaml file
                radius_body_chassis_link=0.4,
                radius_body_arm_shoulder_link=0.1,
                radius_body_arm_end_effector_link=0.1,
                radius_body_arm_upper_wrist_link=0.1,
                radius_body_arm_lower_wrist_link=0.1,
                radius_body_arm_forearm_link=0.1,
            )
        
    
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
        T_Obj_Grasp[:3,:3] = R.from_euler('xyz', [0, self._roll_obj_grasp, 0], degrees=True).as_matrix()
        T_Grasp_Theta = np.eye(4)
        T_Grasp_Theta[:3,:3] = R.from_euler('xyz', [-self.theta_preference, 0, 0], degrees=False).as_matrix()
        T_W_Grasp = T_W_Obj @ T_Obj_Grasp @ T_Grasp_Theta
        T_W_Grasp[2,3] += self.z_offset_grasping
        return T_W_Grasp
        



    def compute_initial_guesses(self, q_current, x_obsts, T_W_Obj):
        # Collision-free
        self.update_rollouts_parameters(q_current=q_current,
                                        obst_pos=x_obsts,
                                        T_W_Obj=T_W_Obj)
        q_coll_rollout = self._rollouts_planner.compute_rollout(
                                                timesteps=self._CONFIG["gomp"]["rollout_timesteps"],
                                                arg_dict=self._rollouts_args_dict,
                                                tolerance=self._CONFIG["gomp"]["rollout_tolerance"]
                                                )
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
        q_free_guess = self._rollouts_planner.get_initial_guess(num_waypoints=self.num_waypoints,
                                                                rollout=q_free_rollout)
        
        return (q_coll_guess, q_free_guess)


    

    def _solve_QP(self, q_init):
        if len(q_init) <= 2:
            q_init = np.linspace(q_init[0], q_init[-1], self.num_waypoints, axis=0)
        
        if q_init is not None:
            self._gomp_planner.change_linear_term(q_init)
            q_result_prev = copy.deepcopy(q_init)
            for _ in range(self.num_optim_steps):
                self._gomp_planner.change_fixed_point(x0=q_result_prev)
                q_result, solver_status = self._gomp_planner.solve(q_result_prev.reshape(-1,1))
                if solver_status =="solved":
                    q_result_prev = q_result
            if solver_status == "solved":
                f_q_result = self._gomp_planner.compute_cost(q_result_prev.reshape(-1,1))
            else:
                f_q_result = None
            
            return q_result_prev, f_q_result
        else:
            return None, None
    

    def solve(self, q_current, x_obsts, T_W_Obj):
        self._gomp_planner.set_starting_state(q_start=q_current)

        self.update_gomp_parameters(q_current, x_obsts, T_W_Obj)
        self.update_rollouts_parameters(q_current, x_obsts, T_W_Obj)

        
        (q_coll_init, q_free_init) = self.compute_initial_guesses(q_current, x_obsts, T_W_Obj)

        q_result_coll, f_q_coll = self._solve_QP(q_init=q_coll_init)
        q_result_free, f_q_free = self._solve_QP(q_init=q_free_init)

        solver_flag = False
        # Take better solution
        if f_q_coll is not None and f_q_free is not None:
            if f_q_coll + 1 < f_q_free:
                joint_waypoints = q_result_coll
                solver_flag = True
            else:
                joint_waypoints = q_result_free
                solver_flag= True
        # Collision-free is better
        elif f_q_coll is not None and f_q_free is None:
            joint_waypoints = q_result_coll
            solver_flag = True
        # Obstacle free is better
        elif f_q_coll is None and f_q_free is not None:
            joint_waypoints = q_result_free
            solver_flag = True
        

        return self._return_solution(joint_waypoints, solver_flag)
    
    def _return_solution(self, joint_waypoints, solver_flag):
        pose_waypoints = []
        if solver_flag == False:
            for _ in range(self.num_waypoints):
                pose_waypoints.append(self._T_W_StaticGrasp)
            return pose_waypoints, solver_flag
        else:
            for id_way in range(self.num_waypoints):
                T_W_EEF = self._gomp_planner.compute_fk(joint_waypoints[id_way,:])
                pose_waypoints.append(T_W_EEF)
        return pose_waypoints, solver_flag


    
