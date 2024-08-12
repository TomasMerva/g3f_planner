import os
import sys
import numpy as np
from fabrics_rollouts import FabricsDriver
import time

current_script_dir = os.path.dirname(os.path.abspath(__file__))
fabrics_lib = os.path.normpath(os.path.join(current_script_dir, '../..', 'build', 'libfabrics_controller.so'))
config_file = os.path.normpath(os.path.join(current_script_dir, '../..', 'config', 'dinova_config_full.yaml'))

controller = FabricsDriver(config_file = config_file,
                            controller_lib = fabrics_lib)

arguments_dict = dict(
            q = np.zeros((9), dtype=np.float64),
            qdot = np.zeros((9), dtype=np.float64),
            x_goal_0 = np.array([1.0, 0, 0.5], dtype=np.float64),
            weight_goal_0= 0.5,
            x_goal_1= np.array([1.0, 0, 0.5], dtype=np.float64),
            weight_goal_1= 0.5,
            x_goal_2= np.array([1.0, 0, 0.5], dtype=np.float64),
            weight_goal_2= 0.5,
            x_goal_3=[0.5],
            weight_goal_3= 0.5,
            x_obsts=[np.array([1.0, 0, 0.5], dtype=np.float64),
                     np.array([1.0, 0, 0.5], dtype=np.float64)],
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
action = controller.compute_action(**arguments_dict)
end_time = time.perf_counter()
print(f"Elapsed time: {end_time-start_time}s")

print(controller.print_input_args())
