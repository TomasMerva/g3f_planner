import os
import sys
import numpy as np
import time
from fabrics_rollouts import RolloutFabrics

current_script_dir = os.path.dirname(os.path.abspath(__file__))
fabrics_lib = os.path.normpath(os.path.join(current_script_dir, '../..', 'build', 'libfabrics_controller.so'))
config_path = os.path.normpath(os.path.join(current_script_dir, '../..', 'config', 'dinova_config_full.yaml'))
urdf_folder = os.path.normpath( os.path.join(current_script_dir, '..', 'urdfs'))
ROBOT_URDF_FILE = urdf_folder + "/dinova/dinova.urdf"

fk_args = dict(
    urdf_file = ROBOT_URDF_FILE,
    root_link = "world",
    end_link = "arm_end_effector_link"
)

n_dof = 9
vel_limits = [3]*n_dof
rollouts = RolloutFabrics(fk_dict=fk_args,
                          config_file=config_path,
                          controller_lib=fabrics_lib,
                          vel_limits=vel_limits,
                          dt=0.1,
                          alpha_filter=0.7)


arguments_dict = dict(
            q = np.zeros((n_dof), dtype=np.float64),
            qdot = np.zeros((n_dof), dtype=np.float64),
            x_goal_0 = np.array([1.0, 0, 0.5], dtype=np.float64),
            weight_goal_0= 1,
            x_goal_1= np.array([1.0, 0, 0.5], dtype=np.float64),
            weight_goal_1= 3,
            x_goal_2= np.array([1.0, 0, 0.5], dtype=np.float64),
            weight_goal_2= 3,
            x_goal_3=[0.5],
            weight_goal_3= 0.5,
            x_obsts=[np.array([10.0, 5.0, 0.5], dtype=np.float64),
                     np.array([10.0, 5.0, 0.5], dtype=np.float64)],
            radius_obsts=[ np.array([0.4], dtype=np.float64),
                           np.array([0.4], dtype=np.float64)],
            radius_body_chassis_link=0.4,
            radius_body_arm_shoulder_link=0.1,
            radius_body_arm_end_effector_link = 0.1,
            radius_body_arm_upper_wrist_link = 0.1,
            radius_body_arm_lower_wrist_link = 0.1,
            radius_body_arm_forearm_link=0.1,
        )

start_time = time.perf_counter()
q_rollout = rollouts.compute_rollout(timesteps=100, 
                                     arg_dict=arguments_dict,
                                     tolerance=0.15)
end_time = time.perf_counter()
print(f"Elapsed time: {end_time-start_time}s")
print(len(q_rollout))
# print(q_rollout)