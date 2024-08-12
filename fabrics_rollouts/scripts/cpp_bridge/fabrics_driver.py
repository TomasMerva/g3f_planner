import yaml
import numpy as np
from dataclasses import dataclass
import ctypes
import os 
import time

from fabrics_rollouts.scripts.cpp_bridge.fabrics_config import FabricsConfig

class FabricsDriver(FabricsConfig):
    def __init__(self, 
                 config_file = None,
                 controller_lib = None) -> None:
        super().__init__(config_file=config_file)
        if controller_lib is None:
            raise RuntimeError("No fabrics shared library loaded.")
        
        
        self._fabrics_lib = self.load_fabrics_lib(controller_lib)
        
    def load_fabrics_lib(self, controller_path):
        lib = ctypes.CDLL(controller_path)  

        # Define the casadi_f0 function signature
        lib.funs.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # const casadi_real** arg
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # casadi_real** res
            ctypes.POINTER(ctypes.c_int),                    # casadi_int* iw
            ctypes.POINTER(ctypes.c_double),                 # casadi_real* w
            ctypes.c_int                                     # int mem
        ]   

        lib.funs.restype = ctypes.c_int  # The return type

        self._setting_0 = np.array([0, 0], dtype=np.int32)
        self._setting_1 = np.array([0.0, 0.0], dtype=np.float64)
        self._setting_2 = 0

        self._setting_0 = self._setting_0.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._setting_1 = self._setting_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        return lib
    
    def compute_action(self, **arg_dict):
        arg_data = self.handle_input_arg_dict(arg_dict)
    
        arg = (ctypes.POINTER(ctypes.c_double) * len(arg_data))(
            *[a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for a in arg_data]
        )

        res_data = [np.zeros(9, dtype=np.float64)]
        res = (ctypes.POINTER(ctypes.c_double) * 1)(
            *[r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for r in res_data]
        )
        err_code = self._fabrics_lib.funs(arg, res, self._setting_0, self._setting_1, self._setting_2)
      
        return np.ctypeslib.as_array(res[0], shape=(9,)).astype(np.float32)
    

