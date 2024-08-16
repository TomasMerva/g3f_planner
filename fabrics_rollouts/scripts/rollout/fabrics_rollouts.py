import numpy as np
import copy

from fabrics_rollouts.scripts.cpp_bridge.fabrics_driver import FabricsDriver
from fabrics_rollouts.scripts.utils.robot_model import RobotKinematicModel


class RolloutFabrics(FabricsDriver):
    def __init__(self,
                 fk_dict,
                 config_file = None,
                 controller_lib = None,
                 vel_limits = None,
                 dt=0.05,
                 alpha_filter=0.7,
                 ) -> None:
        super().__init__(config_file, controller_lib)
        self._dt = dt
        self._alpha_filter = alpha_filter
        self._robot_model = RobotKinematicModel(fk_dict["urdf_file"], fk_dict["root_link"], fk_dict["end_link"])
        if vel_limits is None:
            raise RuntimeError("Velocity limits are not specified")
        self.vel_limits = vel_limits
    
    def compute_error(self, x_goal, q_current):
        T_W_EEF = self._robot_model.compute_fk(q_current)
        position_error = np.linalg.norm(x_goal - T_W_EEF[:3,3])
        return position_error
    
    def apply_low_pass_filter(self, x, x_prev):
        return self._alpha_filter*x_prev + (1-self._alpha_filter)*x
    
    def clip_actions(self, action):
        #TODO: change it since it is tailored for dinova
        if np.linalg.norm(action[0:2]) > self.vel_limits[0]:
            action[0:2] = action[0:2] / np.linalg.norm(action[0:2]) * self.vel_limits[0]
        action[2:] = np.clip(action[2:], -1*self.vel_limits[2], self.vel_limits[2])
        return action
    
    def get_initial_guess(self, num_waypoints, rollout):
        if len(rollout) < 2:
            raise ValueError("Vector must have at least 2 elements.")
        
        indices = np.linspace(0, len(rollout) - 1, num_waypoints, dtype=int)
        waypoints = [rollout[i].tolist() for i in indices]
        return np.asarray(waypoints)

    def compute_rollout(self, timesteps, arg_dict, tolerance=0.15) -> list:
        rollout_arg_dict = copy.deepcopy(arg_dict)
        q = copy.deepcopy(rollout_arg_dict["q"])
        qdot = copy.deepcopy(rollout_arg_dict["qdot"])

        q_rollout_record = []
        for _ in range(timesteps):
            action = self.compute_action(**rollout_arg_dict)

            action = self.clip_actions(action)
        
            qdot = self.apply_low_pass_filter(x=action, x_prev=qdot)
            q = q + qdot*self._dt

            rollout_arg_dict["q"] = q
            rollout_arg_dict["qdot"] = qdot
            q_rollout_record.append(q)

            error = self.compute_error(rollout_arg_dict["x_goal_0"], q)
            if error <= tolerance:
                return q_rollout_record
        return q_rollout_record