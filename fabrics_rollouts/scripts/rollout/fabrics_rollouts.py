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
        self._end_link = fk_dict["end_link"]
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
            print("Cannot compute initial guess from a rollout. Vector must have at least 2 elements.")
            return None
        
        indices = np.linspace(0, len(rollout) - 1, num_waypoints, dtype=int)
        waypoints = [rollout[i].tolist() for i in indices]
        return np.asarray(waypoints)

    def compute_dynamic_weights(self, q, T_W_Goal):
        position_error = self.error(goal_pos=T_W_Goal[:3,3],
                                    q_current=q)
        self._weight_goal_0, self._weight_goal_1, self._weight_goal_2, self._weight_goal_3 = self.set_runtime_weights(position_error)

    def error(self, goal_pos:np.ndarray, q_current:np.ndarray) -> float:
        fk_current = self.compute_fk(q_current, self._end_link)[:3,3]
        return np.linalg.norm(goal_pos-fk_current)

    def set_runtime_weights(self, error):
        # weight_goal_0 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * self._goal_weights_offline[0]
        weight_goal_0 = 0.4 * (np.tanh(4 * error - 0.5) + 2.0) * self._goal_weights_offline[0]
        weight_goal_1 = 0.4 * (np.tanh(-4 * error + 2.0) + 1.5) * self._goal_weights_offline[1]
        weight_goal_2 = 0.5 * (np.tanh(-4 * error + 2.0) + 1.0) * self._goal_weights_offline[2]
        weight_goal_3 = 0.5 * (np.tanh(4 * error - 1.5) + 1.6) * self._goal_weights_offline[3]
        return weight_goal_0, weight_goal_1, weight_goal_2, weight_goal_3
    
    def compute_rollout(self, timesteps, arg_dict, tolerance=0.15) -> list:
        rollout_arg_dict = copy.deepcopy(arg_dict)
        q = copy.deepcopy(rollout_arg_dict["q"])
        qdot = copy.deepcopy(rollout_arg_dict["qdot"])
        self._goal_weights_offline = [0.0, 0.0, 0.0, 0.0]
        self._goal_weights_offline[0] = copy.deepcopy(rollout_arg_dict["weight_goal_0"])
        self._goal_weights_offline[1] = copy.deepcopy(rollout_arg_dict["weight_goal_1"])
        self._goal_weights_offline[2] = copy.deepcopy(rollout_arg_dict["weight_goal_2"])
        self._goal_weights_offline[3] = copy.deepcopy(rollout_arg_dict["weight_goal_3"])

        q_rollout_record = []
        self.qdot_rollout_record = []

        for _ in range(timesteps):
            action = self.compute_action(**rollout_arg_dict)

            action = self.clip_actions(action)
        
            qdot = self.apply_low_pass_filter(x=action, x_prev=qdot)
            q = q + qdot*self._dt


            self.compute_dynamic_weights(q=q, T_W_Goal=rollout_arg_dict["T_W_Goal"])
            rollout_arg_dict["q"] = q
            rollout_arg_dict["qdot"] = qdot
            rollout_arg_dict["weight_goal_0"] = self._weight_goal_0
            rollout_arg_dict["weight_goal_1"] = self._weight_goal_1
            rollout_arg_dict["weight_goal_2"] = self._weight_goal_2
            rollout_arg_dict["weight_goal_3"] = self._weight_goal_3

            q_rollout_record.append(q)
            self.qdot_rollout_record.append(qdot)

            error = self.compute_error(rollout_arg_dict["x_goal_0"], q)
            if error <= tolerance:
                return q_rollout_record
   
        return q_rollout_record

    def get_rollout_velocity_avg(self, horizon=2):
        timesteps = len(self.qdot_rollout_record)
        if timesteps < horizon:
            horizon= timesteps
        qdot_rollout_avg = np.mean((np.array(self.qdot_rollout_record[0:horizon]))**2)/horizon
        return qdot_rollout_avg
    
    def compute_fk(self, q, end_link):
        return self._robot_model.compute_fk(q, end_link)