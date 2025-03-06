import numpy as np
import casadi as ca
import spatial_casadi as sc

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk


class RobotKinematicModel():
    def __init__(self, urdf_file, root_link, end_link):
        with open(urdf_file, "r") as file:
            self._urdf = file.read()

        self._robot_fk = GenericURDFFk(
                            self._urdf,
                            root_link = root_link,
                            end_links= end_link
                        )
        self._root_link = root_link
        self._end_link = end_link
        self.n_dofs = self._robot_fk.n()
        
    def compute_fk(self, q, end_link=None):
        if end_link is not None:
            return self._robot_fk.numpy(q, end_link, position_only=False)
        else:
            return self._robot_fk.numpy(q, self._end_link, position_only=False)
        
    def compute_fk_ca(self, q_ca, end_link=None):
        if end_link == None:
            return self._robot_fk.casadi(q_ca, self._end_link, position_only=False)
        else:
            return self._robot_fk.casadi(q_ca, child_link=end_link, position_only=False)

    def compute_fk_rpy_ca(self, q_ca, end_link=None):
        if end_link == None:
            fk_ca = self.compute_fk_ca(q_ca)
            R_ca_rpy = sc.Rotation.from_matrix(fk_ca[:3,:3]).as_euler("xyz") #convert rotation part of FK into rpy
            return ca.vertcat(fk_ca[:3,3], R_ca_rpy)
        else:
            fk_ca = self.compute_fk_ca(q_ca, end_link)
            R_ca_rpy = sc.Rotation.from_matrix(fk_ca[:3,:3]).as_euler("xyz") #convert rotation part of FK into rpy
            return ca.vertcat(fk_ca[:3,3], R_ca_rpy)
        
    def compute_jacobian_ca(self, q_ca):
        pass