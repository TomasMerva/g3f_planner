from fabrics_rollouts.scripts.constraints.constaint_template import *


class GraspPositionConstraint(ConstraintTemplate):
    def __init__(self,
                 robot_model: RobotKinematicModel,
                 x_robot,
                 param_T_W_Grasp,
                 tolerance = 0.0) -> None:
        super().__init__()
        self._robot_model = robot_model
        _Trpy_W_EEF = self._robot_model.compute_fk_rpy_ca(x_robot)
        
        self._g = _Trpy_W_EEF[:3] - param_T_W_Grasp[:3, 3]
        self._gradient = ca.jacobian(self._g, x_robot)
        self._eval_g = ca.Function("g_grasp_pos", [x_robot, param_T_W_Grasp], [self._g])
        self._eval_grad = ca.Function("dg_grasp_pos",  [x_robot, param_T_W_Grasp], [self._gradient])

        # self._lb = np.zeros(3) - tolerance
        # self._ub = np.zeros(3) + tolerance
        self._lb = np.zeros(3) - tolerance
        self._ub = np.zeros(3) + tolerance
   