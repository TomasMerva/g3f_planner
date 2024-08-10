import yaml
import numpy as np
from dataclasses import dataclass

@dataclass
class FabricsArgumentSize():
    GOAL_DIM : int = 3
    GOAL_WEIGTH_DIM : int = 1
    OBSTACLE_DIM : int = 1
    OBSTACLE_POS_DIM : int = 3
    COLLISION_LINK_DIM : int = 1
    PLANE_CONSTRAINT_DIM : int = 4