from g3f_planner.scripts.constraints.constaint_template import *


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
        self._gradient = ca.gradient(self._g, x_robot)
        self._eval_g = ca.Function("g_col" + self._link_name, [x_robot, x_obs], [self._g])
        self._eval_grad = ca.Function("dg_col" + self._link_name, [x_robot, x_obs], [self._gradient])
        self._lb = r_link + r_obst + tolerance
        self._ub = float("inf")


        self._linear_lb, self._linear_ub = self.express_linearization(x_robot, x_obs)
        self._eval_linear_lb = ca.Function("g_col_lb" + self._link_name, [x_robot, x_obs], [self._linear_lb])
        self._eval_linear_ub = ca.Function("g_col_ub" + self._link_name, [x_robot, x_obs], [self._linear_ub])




    # x_robot -> joint configuration for a given timesteps 1D array
    def express_linearization(self, x_robot, x_obs):
        _linear_lb = self._lb - self._g + self._gradient@x_robot.T
        _linear_ub = self._ub - self._g + self._gradient@x_robot.T

        return (_linear_lb, _linear_ub)
    
    def eval_linearized_constraint(self, x0_robot, x_obs):
        g_grad_x0 = self._eval_grad(x0_robot, x_obs)
        lb = self._eval_linear_lb(x0_robot, x_obs)
        ub = self._eval_linear_ub(x0_robot, x_obs)
        return (g_grad_x0, lb, ub)

   