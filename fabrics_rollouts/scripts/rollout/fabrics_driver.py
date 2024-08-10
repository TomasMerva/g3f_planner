import yaml
import numpy as np
from dataclasses import dataclass
import ctypes
import os 


class FabricsDriver():
    def __init__(self, 
                 config_file = None,
                 controller_lib = None) -> None:
        if config_file is None:
            raise RuntimeError("No config file loaded")
        if controller_lib is None:
            raise RuntimeError("No fabrics shared library loaded")
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
    
    def compute_action(self, arg_data):
        arg = (ctypes.POINTER(ctypes.c_double) * len(arg_data))(
            *[a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for a in arg_data]
        )

        res_data = [np.zeros(9, dtype=np.float64)]
        res = (ctypes.POINTER(ctypes.c_double) * 1)(
            *[r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for r in res_data]
        )
        err_code = self._fabrics_lib.funs(arg, res, self._setting_0, self._setting_1, self._setting_2)
        return np.ctypeslib.as_array(res[0], shape=(9,))



current_script_dir = os.path.dirname(os.path.abspath(__file__))
fabrics_path = os.path.normpath(os.path.join(current_script_dir, '../../..', 'build', 'libfabrics_controller.so'))

controller = FabricsDriver(fabrics_path)



i0 = np.zeros((9), dtype=np.float64)
i1 = np.zeros((9), dtype=np.float64)
i2 = np.array([0.2], dtype=np.float64)
i3 = np.array([0.2], dtype=np.float64)
i4 = np.array([0.2], dtype=np.float64)
i5 = np.array([0.2], dtype=np.float64)
i6 = np.array([0.2], dtype=np.float64)
i7 = np.array([0.2], dtype=np.float64)
i8 = np.array([0.2], dtype=np.float64)
i9 = np.array([0.2], dtype=np.float64)
i10 = np.array([0.2], dtype=np.float64)
i11 = np.array([0.2], dtype=np.float64)
i12 = np.array([0.2], dtype=np.float64)
i13 = np.array([0.2], dtype=np.float64)
i14 = np.array([1.0, 0, 0.5], dtype=np.float64)
i15 = np.array([1.0, 0, 0.5], dtype=np.float64)
i16 = np.array([1.0, 0, 0.5], dtype=np.float64)
i17 = np.array([5.0], dtype=np.float64)
i18 = np.array([1.0, 0, 0.5], dtype=np.float64)
i19 = np.array([1.0, 0, 0.5], dtype=np.float64)


arg_data = [
    i0, i1, i2, i3, i4, i5,
    i6, i7, i8, i9, i10, i11,
    i12, i13, i14, i15, i16, i17,
    i18, i19
]

action = controller.compute_action(arg_data)
print(action)


