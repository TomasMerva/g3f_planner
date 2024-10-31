import numpy as np
import copy
import yaml
import time
import os
from scipy.spatial.transform import Rotation as R
import pybullet
from typing import Dict
from fabrics_rollouts import DeadlockPrevention
from dinovas_pybullet_env import Environment
from rgf_planner import RGF_Planner
from fabrics_planner import Fabrics
from fabrics_rollouts import ReferenceTracker
from tqdm import tqdm

from evaluation.record_data import RecordData

def transformation2dict(T : np.array) -> Dict:
    return {
        "position": T[:3,3],
        "orientation": R.from_matrix(T[:3,:3]).as_quat(),
    }

def dict2transformation(pose : dict) -> np.ndarray:
    T = np.eye(4)
    T[:3,3] = pose["position"]
    T[:3,:3] = R.from_quat(pose["orientation"]).as_matrix()
    return T






def run_dinova_example(n_steps, 
                       dof, 
                       n_robots, 
                       gomp_config_file,
                       env:Environment, 
                       nr_obst = 3, 
                       render=False, 
                       stopping_tolerance=0.05):
    RENDER = render
    NUM_ROBOTS = n_robots
    NUM_DOF = dof
    NUM_GRIPPER_FINGERS = 2
    NUM_TIMESTEPS = n_steps
    NUM_OBST = nr_obst
    PLANNER_PERIOD = 100
    CONFIG_FILE_PATH_GOMP = gomp_config_file

    robots_color = [ [0, 255, 0],
                    [0, 255, 255],
                    [0,0,255],
                    [255,255,0] ]


    sim = env.get_sim_handler()
    CONFIG_FILE_PATH = env.get_config_file_path()
    action = np.zeros(NUM_ROBOTS*NUM_DOF)
    ob, *_ = sim.step(action)

    # Deadlock resolution
    deadlock_prevention = DeadlockPrevention(dof=[NUM_DOF] * NUM_ROBOTS, n_robots=NUM_ROBOTS)
    qdot_rollout_avg = {f"robot_{i}": [] for i in range(n_robots)}
   
    #Modify obstacles based on number of robots
    obstacles = env.get_obstacles()
    x_obsts = [obstacles[i]["position"] for i in obstacles]
    r_obsts = [obstacles[i]["radius"] for i in obstacles]

    arguments_dicts = {f"robot_{i}": [] for i in range(n_robots)}
    
    # Planner
    fk_args = dict(
        urdf_file = env.ROBOT_URDF_FILE,
        root_link = "world",
        end_link = "arm_tool_frame",
        num_dofs = NUM_DOF-NUM_GRIPPER_FINGERS,
    )
    planner = RGF_Planner(fk_args=fk_args,
                          config_file_path=CONFIG_FILE_PATH_GOMP)
    
    
    # Fabrics
    fabrics = Fabrics(robot_urdf_path=env.ROBOT_URDF_FILE,
                      config_file_path=CONFIG_FILE_PATH,
                      degrees_of_freedom=NUM_DOF-NUM_GRIPPER_FINGERS)

    T_W_Goals, T_W_Objects = [], []
    for robot_id in range(NUM_ROBOTS):
        T_W_Object = np.eye(4)
        T_W_Object[:3,3], quat = env.get_cup(robot_id)
        T_W_Objects.append(T_W_Object)
        T_W_Goals.append(T_W_Object)    


    """
    Results metrics:
    """
    evaluation_data = RecordData()
    success_rate_per_robot = [0] * n_robots

    
    # Main loop
    print("Starting RF env")
    for timestep in tqdm(range(NUM_TIMESTEPS)):
        robot_states = [[ob["robot_"+str(i)]["joint_state"]["position"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)],
                         ob["robot_"+str(i)]["joint_state"]["velocity"][0:(NUM_DOF-NUM_GRIPPER_FINGERS)]]
                        for i in range(NUM_ROBOTS)]
        T_W_EEFs_current = [planner.compute_fk(robot_states[i][0]) for i in range(NUM_ROBOTS)]
        position_EEFs_current = [T_W_EEFs_current[i][:3, 3] for i in range(NUM_ROBOTS)]
        # Update collision spheres for GOMP & fabrics
        T_W_chassis_robots = [fabrics.compute_fk(robot_states[i][0], "chassis_link") for i in range(NUM_ROBOTS)]
        T_W_wrist_robots = [fabrics.compute_fk(robot_states[i][0], "arm_upper_wrist_link") for i in range(NUM_ROBOTS)]

        if timestep == 0:
            for robot_id in range(NUM_ROBOTS):
                theta = fabrics.get_theta_preference(q=robot_states[robot_id][0],
                                                     goal_position=T_W_Goals[robot_id][:3,3])
                T_W_Goals[robot_id] = fabrics.compute_static_grasp(T_W_Goals[robot_id], theta)



        
            

        # Rollout fabrics
        rollout_time = []
        for robot_id in range(NUM_ROBOTS):
            # Compute obstacles' poses
            counter = 0
            for i in range(NUM_ROBOTS):
                if i == robot_id:
                    continue
                else:
                    chassis_idx = counter 
                    wrist_idx = counter + 1
                    x_obsts[chassis_idx] = T_W_chassis_robots[i][:3,3].tolist()
                    x_obsts[wrist_idx] = T_W_wrist_robots[i][:3,3].tolist()
                    counter += 2
            
            # Compute rollouts
            if timestep%PLANNER_PERIOD == 0:
                start_time = time.perf_counter()
                if success_rate_per_robot[robot_id] == 0:
                    planner.update_param_and_initial_guess(joint_state=robot_states[robot_id],
                                                        T_W_Obj=T_W_Objects[robot_id],
                                                        x_obsts=x_obsts[:NUM_ROBOTS],
                                                        r_obsts=r_obsts[:NUM_ROBOTS],
                                                        )
                qdot_rollout_avg["robot_" + str(robot_id)] = planner.get_velocity_average()
                end_time = time.perf_counter()
                rollout_time.append(end_time-start_time)

            # Update weigths and arguments for fabrics
            fabrics.compute_dynamic_weights(q= robot_states[robot_id][0],
                                            T_W_Goal=T_W_Goals[robot_id])
            fabrics.update_arguments(joint_state= robot_states[robot_id],
                                    T_W_Goal=T_W_Goals[robot_id],
                                    obst_pos=x_obsts,
                                    obst_radius=r_obsts)
            arguments_dicts["robot_" + str(robot_id)] = copy.deepcopy(fabrics.get_arguments())
        
        # Record time but only when rollouts are performed
        if timestep%PLANNER_PERIOD == 0:
            evaluation_data.record_computational_time_qp(sum(rollout_time))
            
        deadlock_prevention.deadlock_checking(x_robots=position_EEFs_current,
                                                goals_final=[T_W_Goals[robot_id][:3,3] for robot_id in range(NUM_ROBOTS)],
                                                time_step=timestep,
                                                avg_sum=copy.deepcopy(sum(qdot_rollout_avg.values()) / NUM_ROBOTS))



        # for robot_id in range(NUM_ROBOTS):
        #     counter = 0
        #     for i in range(NUM_ROBOTS):
        #         if i == robot_id:
        #             continue
        #         else:
        #             chassis_idx = counter
        #             wrist_idx = counter + 1
        #             x_obsts[chassis_idx] = T_W_chassis_robots[i][:3,3].tolist()
        #             x_obsts[wrist_idx] = T_W_wrist_robots[i][:3,3].tolist()
        #             counter += 2

        #     if timestep%PLANNER_PERIOD == 0:
        #         if success_rate_per_robot[robot_id] == 0:
        #             planner.update_param_and_initial_guess(joint_state=robot_states[robot_id],
        #                                                 T_W_Obj=T_W_Objects[robot_id],
        #                                                 x_obsts=x_obsts[:NUM_ROBOTS],
        #                                                 r_obsts=r_obsts[:NUM_ROBOTS],
        #                                                 )

        #             qdot_rollout_avg["robot_" + str(robot_id)] = planner.get_velocity_average()

        #     start_time = time.perf_counter()
        #     fabrics.compute_dynamic_weights(q= robot_states[robot_id][0],
        #                                     T_W_Goal=T_W_Goals[robot_id])
        #     fabrics.update_arguments(joint_state= robot_states[robot_id],
        #                              T_W_Goal=T_W_Goals[robot_id],
        #                              obst_pos=x_obsts,
        #                              obst_radius=r_obsts)
        #     arguments_dicts["robot_" + str(robot_id)] = copy.deepcopy(fabrics.get_arguments())

        # # need information of GOMP for both robots for deadlock resolution:
        # deadlock_prevention.deadlock_checking(x_robots=position_EEFs_current,
        #                                       goals_final=[T_W_Goals[robot_id][:3,3] for robot_id in range(NUM_ROBOTS)],
        #                                       time_step=timestep,
        #                                       avg_sum=copy.deepcopy(sum(qdot_rollout_avg.values()) / NUM_ROBOTS))

        for robot_id in range(NUM_ROBOTS):
            start_time = time.perf_counter()
            goal_robots, goal_weights = deadlock_prevention.deadlock_adapt_goals_weights(x_robots=position_EEFs_current,
                                                                                         goal_robots=[arguments_dicts["robot_" + str(j_robot)]["x_goal_0"] for j_robot in range(NUM_ROBOTS)],
                                                                                         goal_weights=[arguments_dicts["robot_" + str(j_robot)]["weight_goal_0"]for j_robot in range(NUM_ROBOTS)])
            arguments_dicts["robot_" + str(robot_id)]["x_goal_0"] = goal_robots[robot_id]
            arguments_dicts["robot_" + str(robot_id)]["weight_goal_0"] = goal_weights[robot_id]
            fabrics.set_arguments(arguments_dicts["robot_" + str(robot_id)])
            action_unclipped = fabrics.compute_action()
            action[(robot_id*NUM_DOF): NUM_DOF*robot_id + (NUM_DOF-NUM_GRIPPER_FINGERS)] = fabrics.clip_action(action_unclipped)
            end_time = time.perf_counter()
            evaluation_data.record_computational_time(end_time - start_time)

            if fabrics.error(goal_pos=T_W_Goals[robot_id][:3, 3], q_current=robot_states[robot_id][0]) <= stopping_tolerance:
                success_rate_per_robot[robot_id] = 1

        collision_flag = env.check_collisions()
        if collision_flag == True:
            evaluation_data.record_success_rate(success=0.0)
            evaluation_data.record_collision_violation(collision_flag)
            print("RF failed")
            break

        if np.all(success_rate_per_robot):
            evaluation_data.record_success_rate(success=100.0)
            evaluation_data.record_time_to_goal(timestep, sim._dt)
            evaluation_data.record_collision_violation(collision_flag=False)
            print("RF succeeded")
            break         

        ob, *_ = sim.step(action)

    sim.close()

    return evaluation_data.get_result()

def main(render=True, timesteps=2000):
    RENDER = render
    NUM_ROBOTS = 2
    NUM_DOF = 11
    NUM_TIMESTEPS = timesteps

    #Read config file for Planner
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_script_dir, '..', 'config/dinova_config_rgf_3obst.yaml')
    CONFIG_FILE_PATH_GOMP = os.path.normpath(config_path)

    # Environment
    env = Environment()
    env.initialize(render=RENDER, nr_robots=NUM_ROBOTS)
    run_dinova_example(n_steps=NUM_TIMESTEPS,
                       dof=NUM_DOF,
                       n_robots=NUM_ROBOTS,
                       env=env,
                       render=RENDER,
                       gomp_config_file=CONFIG_FILE_PATH_GOMP
                       )
    return {}

if __name__=="__main__":
    main()

    