import yaml
import socket
import numpy as np
from enum import Enum
from dataclasses import dataclass
import time

@dataclass
class FabricsArgumentSize():
    GOAL_DIM : int = 3
    GOAL_WEIGTH_DIM : int = 1
    OBSTACLE_DIM : int = 1
    OBSTACLE_POS_DIM : int = 3
    COLLISION_LINK_DIM : int = 1
    PLANE_CONSTRAINT_DIM : int = 4

class FabricsClient():
    def __init__(self, config_file) -> None:
        self.desired_msg_lenght = 0
        self.config = self.read_config_file(config_file)
        self._compute_args_indices()
        self.print_args_indices()

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.config["server_address"],
                                    self.config["server_port"]))
    
    def compute_action(self, arguments_dict):
        msg = self._handle_argument_dict(arguments_dict)
        self._send_data(msg)
        
        action = self._recv_data()
        return action

    def disconnect(self):
        self.client_socket.close()

    
    def _send_data(self, data: str):
        self.client_socket.sendall(data.encode())

    def _recv_data(self):
        data = self.client_socket.recv(1024)
        return np.fromstring(data.decode(), dtype=float, sep=' ')
    

    def read_config_file(self, config_file):
        with open(config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)

        fabrics_config = {}
        # Server
        fabrics_config["server_address"] = config["server"]["ip_address"]
        fabrics_config["server_port"] = config["server"]["port"]
        # Fabrics
        fabrics_config["dim_states"] = 2
        fabrics_config["num_goals"] = len(config["problem"]["goal"]["goal_definition"])
        fabrics_config["num_goals_weights"] =  len(config["problem"]["goal"]["goal_definition"])
        fabrics_config["num_obstacles"] = config["problem"]["environment"]["number_spheres"]["dynamic"] \
                                         + config["problem"]["environment"]["number_spheres"]["static"]
        fabrics_config["num_collision_link"] = len(config["problem"]["robot_representation"]["collision_links"])
        fabrics_config["num_planes"]  = config["problem"]["environment"]["number_planes"]
        fabrics_config["num_dofs"]  = len(config["problem"]["joint_limits"]["lower_limits"])
        return fabrics_config

    def _compute_args_indices(self):
        start_idx = 0
        self._fabrics_args_idx = {}
        # plane constraint
        for i in range(self.config["num_planes"]):
            end_idx = start_idx + FabricsArgumentSize.PLANE_CONSTRAINT_DIM
            self._fabrics_args_idx["plane_"+str(i)] = (start_idx, end_idx)
            start_idx = end_idx
        # q and qdot
        for i in range(self.config["dim_states"]):
            end_idx = start_idx + self.config["num_dofs"]
            self._fabrics_args_idx["q_state_"+str(i)] = (start_idx, end_idx)
            start_idx = end_idx
        # radius body
        for i in range(self.config["num_collision_link"]):
            end_idx = start_idx + FabricsArgumentSize.COLLISION_LINK_DIM
            self._fabrics_args_idx["radius_body_"+str(i)] = (start_idx, end_idx)
            start_idx = end_idx
        # radius obstacle
        for i in range(self.config["num_obstacles"]):
            end_idx = start_idx + FabricsArgumentSize.OBSTACLE_DIM
            self._fabrics_args_idx["radius_obst_"+str(i)] = (start_idx, end_idx)
            start_idx = end_idx
        # weight goal
        for i in range(self.config["num_goals_weights"]):
            end_idx = start_idx + FabricsArgumentSize.GOAL_WEIGTH_DIM
            self._fabrics_args_idx["weight_goal_"+str(i)] = (start_idx, end_idx)
            start_idx = end_idx
        # goals
        for i in range(self.config["num_goals"]):
            end_idx = start_idx + FabricsArgumentSize.GOAL_DIM
            self._fabrics_args_idx["x_goal"+str(i)] = (start_idx, end_idx)
            start_idx = end_idx
        # obstacle pos
        for i in range(self.config["num_obstacles"]):
            end_idx = start_idx + FabricsArgumentSize.OBSTACLE_POS_DIM
            self._fabrics_args_idx["x_goal"+str(i)] = (start_idx, end_idx)
            start_idx = end_idx
        self.desired_msg_lenght = start_idx
    
    def print_args_indices(self):
        print("Fabrics args indices\n---")
        for keys, values in self._fabrics_args_idx.items():
            print(f"{keys}: {values}")
        print("---")

    def _handle_argument_dict(self, arg_dict : dict) -> str:
        data = []
        # Plane constraint
        if self.config["num_planes"] > 0:
            plane_constraints = [k for k, v in arg_dict.items() if k.startswith('constraint_')]
            for plane_g in plane_constraints:
                data.extend(arg_dict[plane_g].tolist())
                
        # q and qdot // np.array
        data.extend(arg_dict["q"].tolist())
        data.extend(arg_dict["qdot"].tolist())

        # radius body
        if self.config["num_collision_link"] > 0:
            radius_body = [v for k, v in arg_dict.items() if k.startswith('radius_body_')]
            data.extend(radius_body)

        # radius obst
        if self.config["num_obstacles"] > 0:
            radius_obsts = [v for k, v in arg_dict.items() if k.startswith('radius_obst')]
            data.extend(np.asarray(radius_obsts[0]).flatten().tolist())

        # weight goal
        if self.config["num_goals_weights"] > 0:
            weight_goal = [v for k, v in arg_dict.items() if k.startswith('weight_goal_')]
            data.extend(weight_goal[0].tolist())

        # x_goal (could be list)
        if self.config["num_goals"] > 0:
            x_goals = [k for k, v in arg_dict.items() if k.startswith('x_goal')]
            for x_goal in x_goals:
                data.extend(arg_dict[x_goal].tolist())
              
        # x_obst (could be list of arrays)
        if self.config["num_obstacles"] > 0:
            x_obsts = [v for k, v in arg_dict.items() if k.startswith('x_obst')]
            for x_obst in x_obsts[0]:
                data.extend(x_obst.tolist())
     
        assert self.desired_msg_lenght == len(data), f"Desired msg lenght {self.desired_msg_lenght} is not the same as actual msg lenght {len(data)}"

        data = np.asarray(data)
        msg = map(str, data)  
        msg = ' '.join(msg)  

        return msg
