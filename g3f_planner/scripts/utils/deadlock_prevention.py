import numpy as np
import itertools

class DeadlockPrevention(object):

    def __init__(self, dof, n_robots, N_horizon=10):
        # params from inputs:
        self.dof = dof
        self.n_robots = n_robots
        self.N_horizon = N_horizon
        robot_nrs = list(range(self.n_robots))
        self.robot_combinations = list(itertools.combinations(robot_nrs, 2))

        # constants:
        self.i_leader = 0
        self.i_follower = 1
        self.avg_vel_constant = 0.01
        self.dist_constant = 0
        self.goal_weight_follower = 1
        self.goal_weight_leader = 3
        self.nr_goal_scale = 2

        # initial values
        self.deadlock = False
        self.i_robots_dead = [0, 1]
        self.time_in_deadlock = 1000
        self.time_wait = 300
        self.threshold_dist_endeff = 0.4
        self.goal_robot0 = np.array([0, 0, 0])
        self.deadlock_robots = [0]*self.n_robots
        self.deadlock_combinations = [0]*(len(self.robot_combinations))

    # def compute_velocity_average(self, q_dot_robots_N):
    #     """"
    #     Currently unused
    #     """
    #     avg_sum = 0
    #     for i_robot in range(self.n_robots):
    #         for df in range(self.dof[i_robot]):
    #             q_dot_N = q_dot_robots_N["robot_"+str(i_robot)][df]
    #             q_dot_N_squared = [np.sqrt(q_dot_N_i**2) for q_dot_N_i in q_dot_N]
    #             avg_sum = avg_sum + sum(q_dot_N_squared)/(self.N_horizon*self.dof[i_robot])
    #     return avg_sum

    def compute_distance_to_goal(self, x_robot, goal_robot):
        dist_to_goal = np.linalg.norm(x_robot - goal_robot)
        return dist_to_goal

    def deadlock_checking(self, x_robots, goals_final, time_step, avg_sum):
        # give priority to the one that is slightly closer to the goal, otherwise random.
        self.deadlock = False

        #check which robot combinations are in deadlock
        deadlock_distance = [100 for _ in range(len(self.robot_combinations))]

        for z, i_robots in enumerate(self.robot_combinations):
            dist_to_goal_sum = self.compute_distance_to_goal(x_robots[i_robots[0]], goals_final[i_robots[0]]) + self.compute_distance_to_goal(x_robots[i_robots[1]], goals_final[i_robots[1]])
            dist_endeff = np.linalg.norm(x_robots[i_robots[0]] - x_robots[i_robots[1]])
            check_dist_endeff = dist_endeff < self.threshold_dist_endeff
            #print("avg_sum < self.avg_vel_constant: ", avg_sum < self.avg_vel_constant )
            # print("avg_vel:", avg_sum, "avg_sum < self.avg_vel_constant: ", avg_sum < self.avg_vel_constant, "deadlock:", self.deadlock)
            if avg_sum < self.avg_vel_constant and dist_to_goal_sum>self.dist_constant and time_step>10 and check_dist_endeff:
                print("Deadlock detected!")
                for i_robot in i_robots:
                    self.deadlock_robots[i_robot] = self.deadlock_robots[i_robot] + 1
                    self.deadlock_combinations[z] = self.deadlock_combinations[z] + 1
                    deadlock_distance[z] = dist_endeff

                if len(self.deadlock_combinations) > 0:
                    self.deadlock = True
                    self.time_in_deadlock = 0

        self.deadlock_set_priorities(x_robots, goals_final, time_step)

    def deadlock_set_priorities(self, x_robots, goals_final, time_step):
        dist_robot_to_goal = [np.linalg.norm(x_robots[i_robot] - goals_final[i_robot]) for i_robot in range(self.n_robots)]

        # --- give priorities in deadlock ---#
        if self.deadlock == True and time_step>10:
            # give priority to the one that is slightly closer to the goal, otherwise robot 0 is the follower
            if dist_robot_to_goal[self.i_robots_dead[0]] > dist_robot_to_goal[self.i_robots_dead[1]]:
                self.i_leader = self.i_robots_dead[1]
                self.i_follower = self.i_robots_dead[0]
            else:
                self.i_leader = self.i_robots_dead[0]
                self.i_follower = self.i_robots_dead[1]

    def deadlock_adapt_goals_weights(self, x_robots, goal_robots, goal_weights):
        if self.deadlock or self.time_in_deadlock<self.time_wait:
            # --- adapt goal position ---#
            diff_robot_pos = x_robots[self.i_leader] - x_robots[self.i_follower]
            diff_goal = diff_robot_pos * self.nr_goal_scale
            self.goal_robot0 = x_robots[self.i_follower] - 1. / np.linalg.norm(diff_goal) * diff_goal
            if self.goal_robot0[2] < 0:
                # --- ensure goal in z direction is always positive ---#
                self.goal_robot0[2] = 0.1
            #goal_robots[self.i_follower] = self.goal_robot0

            # --- adapt goal weights ---#
            goal_weights[self.i_leader] = self.goal_weight_leader
            goal_weights[self.i_follower] = self.goal_weight_follower

            # --- update timer for being in the deadlock ---#
            self.time_in_deadlock = self.time_in_deadlock + 1

        return goal_robots, goal_weights