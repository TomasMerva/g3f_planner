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
