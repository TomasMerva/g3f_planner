import numpy as np
import casadi as ca

from .robot_model import RobotKinematicModel

class ConstraintTemplate():
    def __init__(self) -> None:
        pass
    
    def compute_gradient(self, x, param=None):
        if param is not None:
            return np.nan_to_num(self._eval_grad(x, param))
        else:
            return np.nan_to_num(self._eval_grad(x))
    
    def compute_constraint(self, x, param=None):
        if param is not None:
            return self._eval_g(x, param)
        else:
            return self._eval_g(x)
    
    def get_limits(self) -> tuple:
        return (self._lb, self._ub)
    
    def linearize(self) -> tuple:
        raise SyntaxError("not implemented") 
    
    def get_linearized_constraint(self, x0) -> tuple:
        raise SyntaxError("not implemented") 

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
        self._eval_g = ca.Function("g_col" + self._link_name, [x_obs, x_robot], [self._g])
        self._eval_grad = ca.Function("dg_col" + self._link_name, [x_obs, x_robot], [self._gradient])
        self._lb = r_link + r_obst + tolerance
        self._ub = float("inf")


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

        self._lb = np.zeros(3) - tolerance
        self._ub = np.zeros(3) + tolerance
