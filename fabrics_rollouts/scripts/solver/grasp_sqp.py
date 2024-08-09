import osqp
import numpy as np
import casadi as ca
import sys
from scipy import sparse
import spatial_casadi as sc

from fabrics_rollouts.scripts.utils.robot_model import RobotKinematicModel
from fabrics_rollouts.scripts.constraints.constaint_template import * 
from fabrics_rollouts.scripts.constraints.euclidean_collision_constraint import * 
from fabrics_rollouts.scripts.constraints.grasp_position_constraint import * 
from fabrics_rollouts.scripts.constraints.grasp_rotation_constraint import * 


class GompSQP():
    def __init__(self, arg_dict) -> None:
        self._robot_model = RobotKinematicModel(arg_dict["urdf_file"], arg_dict["root_link"], arg_dict["end_link"])
        self._num_waypoints = arg_dict["num_waypoints"]
        self._num_dim = arg_dict["num_dim"]
        self._joint_limits = arg_dict["joint_limits"]

        self._x_init = np.zeros(( self._num_waypoints, self._num_dim), dtype=float).flatten()
        self._q_start = np.zeros(self._num_dim)
        self._P = self._create_P_matrix(self._num_waypoints, self._num_dim)

        self._x_ca = ca.SX.sym("x", (self._num_waypoints, self._num_dim))

        # self._T_W_EEF = self._robot_model.compute_fk_ca(self._x_ca)
        # self._Trpy_W_EEF = self._robot_model.compute_fk_rpy_ca(self._x_ca)
        # self._J = ca.jacobian(self._Trpy_W_EEF, self._x_ca)
        # self._fk_rpy_eval = ca.Function("fk_x", [self._x_ca], [self._Trpy_W_EEF])
        # self._J_eval = ca.Function("fk_x", [self._x_ca], [self._J])

        self._g_list = []
        self.param_ca_dict = {}

    
    def setup_problem(self, x0, max_iter=50):
        self._solver = osqp.OSQP()

        (A, l, u) = self._get_joint_limits()
        self._set_starting_boundary_con(l, u)

        for g_name, g_term in self._g_list:
            w_ID = self.param_ca_dict[g_name]["waypoint_ID"]
            (A_g, l_g, u_g) = self._linearize_constraint(g_term, x0[w_ID,:], self.param_ca_dict[g_name])
            A = sparse.vstack([A, A_g], format='csc')
            l = np.concatenate((l, l_g))
            u = np.concatenate((u, u_g))


        self._solver.setup(self._P, None, A, l, u, max_iter=50)

    def change_fixed_point(self, x0):
        (A, l, u) = self._get_joint_limits()
        self._set_starting_boundary_con(l, u)

        for g_name, g_term in self._g_list:
            w_ID = self.param_ca_dict[g_name]["waypoint_ID"]
            (A_g, l_g, u_g) = self._linearize_constraint(g_term, x0[w_ID,:], self.param_ca_dict[g_name])
            A = sparse.vstack([A, A_g], format='csc')
            l = np.concatenate((l, l_g))
            u = np.concatenate((u, u_g))

        self._solver.update(Ax=A.data,l=l ,u=u)

    def solve(self, x_init=None):
        if x_init is not None:
            self._solver.warm_start(x_init)
        else:
            self._solver.warm_start(self._x_init)
        res = self._solver.solve()
        
        return res.x.reshape((self._num_waypoints, self._num_dim))

   
    def _create_P_matrix(self, num_waypoints, num_dof):
        FD_matrix = np.zeros((num_waypoints, num_waypoints), dtype=float)
        FD_INDICES = [1., -2., 1.]
        for i in range(1, num_waypoints-1):
            FD_matrix[i, (i-1):(i-1)+3] = FD_INDICES
        FD_matrix[0,0]=1
        FD_matrix[-1,-1] = 1
        R = FD_matrix.T@FD_matrix
        return sparse.csc_matrix(sparse.kron(R, sparse.eye(num_dof)))
    

    def _compute_cost(self, x, P):
        return np.dot(np.dot(x.T, P.toarray()), x)

    def _linearize_constraint(self, g: ConstraintTemplate, x0, param_dict) -> tuple:
        idx_l = param_dict["waypoint_ID"]*self._num_dim
        idx_u = param_dict["waypoint_ID"]*self._num_dim + self._num_dim

        if "num_param" in param_dict:
            g_x0 = g.compute_constraint(x0, param_dict["num_param"])
            g_grad_x0 = g.compute_gradient(x0, param_dict["num_param"])
        else:
            g_x0 = g.compute_constraint(x0)
            g_grad_x0 = g.compute_gradient(x0)
        (lb, ub) = g.get_limits()

        lb = lb - g_x0 + np.dot(g_grad_x0, x0.T)
        ub = ub - g_x0 + np.dot(g_grad_x0, x0.T)
        
        A = np.zeros([lb.shape[0], self._num_waypoints*self._num_dim])
        for i in range(lb.shape[0]):
            A[i,idx_l:idx_u] = g_grad_x0[i, :].T

        return (A, lb, ub)

    
    def _get_joint_limits(self):
        A_limits = sparse.csc_matrix(np.eye(self._num_waypoints*self._num_dim))
        l_limits = np.full((self._num_waypoints*self._num_dim, 1), -2.61)
        u_limits = np.full((self._num_waypoints*self._num_dim, 1), 2.61)
        return (A_limits, l_limits, u_limits)
    

    def set_starting_state(self, q_start):
        self._q_start = q_start        

    def _set_starting_boundary_con(self, lb, ub):
        lb[:self._num_dim, 0] = self._q_start
        ub[:self._num_dim, 0] = self._q_start




    def add_grasp_pos_constraint(self, name, waypoint_ID, tolerance=0.0) -> None:
        self.param_ca_dict[name] =  {
            "waypoint_ID" : waypoint_ID,
            "sym_param" : ca.SX.sym(name, 4, 4),
            "num_param" : np.eye(4),
            "tolerance" : tolerance,
            "grasp" : True
        }

        # Tuple
        self._g_list.append((name, 
                             GraspPositionConstraint(robot_model = self._robot_model,
                                                    x_robot = self._x_ca[waypoint_ID, :self._num_dim],
                                                    param_T_W_Grasp = self.param_ca_dict[name]["sym_param"],
                                                    tolerance = tolerance)
                                                    )
                            )
        
    def add_grasp_rot_constraint(self, name, waypoint_ID, tolerance=0.0):
        self.param_ca_dict[name] =  {
            "waypoint_ID" : waypoint_ID,
            "sym_param" : ca.SX.sym(name, 4, 4),
            "num_param" : np.eye(4),
            "tolerance" : tolerance,
            "grasp" : True
            }
        self._g_list.append((name,
                             GraspRotationConstraint(robot_model = self._robot_model,
                                                     x_robot = self._x_ca[waypoint_ID, :self._num_dim],
                                                     param_T_W_Grasp = self.param_ca_dict[name]["sym_param"],
                                                     tolerance = tolerance)
                                                     )
                            )

        
    def add_collision_constraint(self, name: str, waypoint_ID: int, child_link:str, r_link:float, r_obst:float) -> None:
        # Define parameter sym variable
        self.param_ca_dict[name] =  {
            "waypoint_ID" : waypoint_ID,
            "sym_param" : ca.SX.sym(name, 3, 1),
            "num_param" : np.zeros((3,1)),
            "child_link" : child_link,
            "r_link" : r_link,
            "r_obst" : r_obst,
            "grasp" : False,
            }
        self._g_list.append((name,
                             EuclideanCollisionConstraint(robot_model = self._robot_model,
                                                          x_robot = self._x_ca[waypoint_ID, :self._num_dim],
                                                          x_obs = self.param_ca_dict[name]["sym_param"],
                                                          link_name = child_link,
                                                          r_link= r_link,
                                                          r_obst= r_obst,
                                                          tolerance=0.0)
                                                    ))
