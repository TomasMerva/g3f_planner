import numpy as np
import casadi as ca
from g3f_planner.scripts.utils.robot_model import RobotKinematicModel


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
    
    def eval_linearized_constraint(self, x0)-> tuple:
        raise SyntaxError("not implemented") 