from fabrics_rollouts.scripts.constraints.constaint_template import *

class GraspRotationConstraint(ConstraintTemplate):
    def __init__(self,
                 robot_model: RobotKinematicModel,
                 x_robot,
                 param_T_W_Grasp,
                 tolerance = 0.0) -> None:
        super().__init__()
        self._robot_model = robot_model
        self.tolerance = tolerance

        T_W_EEF = self._robot_model.compute_fk_ca(x_robot)



        x_G = param_T_W_Grasp[:3,0] / ca.norm_2(param_T_W_Grasp[:3,0])
        x_EEF = T_W_EEF[:3,0] / ca.norm_2(T_W_EEF[:3,0])
        self._g = x_G.T @ np.eye(3)  @  x_EEF
        
        # self._g = ca.cross(x_G, x_EEF)
        self._gradient = ca.jacobian(self._g, x_robot)
        self._eval_g = ca.Function("g_grasp_rot", [x_robot, param_T_W_Grasp], [self._g])
        self._eval_grad = ca.Function("dg_grasp_rot",  [x_robot, param_T_W_Grasp], [self._gradient])

        # self._lb = ca.vertcat(-self.tolerance, -self.tolerance, -self.tolerance)
        # self._ub = ca.vertcat(self.tolerance, self.tolerance, self.tolerance)
        self._lb = np.cos(self.tolerance)
        self._ub = 100
