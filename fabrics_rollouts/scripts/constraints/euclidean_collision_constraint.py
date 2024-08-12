from fabrics_rollouts.scripts.constraints.constaint_template import *


class EuclideanCollisionConstraint(ConstraintTemplate):
    def __init__(self, 
                 robot_model: RobotKinematicModel,
                 x_robot, 
                 x_obs,
                 link_name : str,
                 r_link : float,
                 r_obst : float,
                 tolerance=0.0) -> None:
        super().__init__()
        self._robot_model = robot_model
        self._link_name = link_name

        self._T_W_EEF = self._robot_model.compute_fk_ca(x_robot, self._link_name)

        self._g = ca.norm_2((x_obs - self._T_W_EEF[:3, 3]))
        self._gradient = ca.gradient(self._g,x_robot)
        self._eval_g = ca.Function("g_col" + self._link_name, [x_robot, x_obs], [self._g])
        self._eval_grad = ca.Function("dg_col" + self._link_name, [x_robot, x_obs], [self._gradient])
        self._lb = r_link + r_obst + tolerance
        self._ub = float("inf")

    # x_robot -> joint configuration for a given timesteps 1D array
    def express_linearization(self, x_robot, x_obs):
        g = self._g
        grad = self._gradient



        # idx_l = param_dict["waypoint_ID"]*self._num_dim
        # idx_u = param_dict["waypoint_ID"]*self._num_dim + self._num_dim

        # if "num_param" in param_dict:
        #     g_x0 = g.compute_constraint(x0, param_dict["num_param"])
        #     g_grad_x0 = g.compute_gradient(x0, param_dict["num_param"])
        # else:
        #     g_x0 = g.compute_constraint(x0)
        #     g_grad_x0 = g.compute_gradient(x0)
        # (lb, ub) = g.get_limits()

        lb = lb - g_x0 + np.dot(g_grad_x0, x0.T)
        ub = ub - g_x0 + np.dot(g_grad_x0, x0.T)
        
        A = np.zeros([lb.shape[0], self._num_waypoints*self._num_dim])
        for i in range(lb.shape[0]):
            A[i,idx_l:idx_u] = g_grad_x0[i, :].T

        return (A, lb, ub)
