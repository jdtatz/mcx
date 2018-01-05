import ctypes
import platform
import numpy as np

if platform.system() == 'Windows':
    _lib = ctypes.CDLL('mcx.dll')
else:
    _lib = ctypes.CDLL('mcx.so')

_config_size = ctypes.c_int.in_dll(_lib, 'SIZE_OF_CONFIG')

_set_field = _lib.mcx_set_field
_set_field.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p)]
_set_field.restype = ctypes.c_int

_get_field = _lib.mcx_get_field
_set_field.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p)]
_set_field.restype = ctypes.c_void_p

_initcfg = _lib.mcx_initcfg
_initcfg.argtypes = [ctypes.c_void_p]

_clearcfg = _lib.mcx_clearcfg
_clearcfg.argtypes = [ctypes.c_void_p]

_cleargpuinfo = _lib.mcx_cleargpuinfo
_cleargpuinfo.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

_validateconfig = _lib.mcx_validateconfig
_validateconfig.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
_validateconfig.restype = ctypes.c_int

_run_simulation = _lib.mcx_run_simulation
_run_simulation.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_list_gpu = _lib.mcx_list_gpu
_list_gpu.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_list_gpu.restype = ctypes.c_int

_flush = _lib.mcx_flush
_flush.argtypes = [ctypes.c_void_p]


def _converter(v):
    if v is True or v is False:
        return ctypes.c_char(v), 0, None
    elif isinstance(v, int):
        return ctypes.c_int(v), 0, None
    elif isinstance(v, float):
        return ctypes.c_float(v), 0, None
    elif isinstance(v, np.array):
        return v.ctypes, v.ndim, v.shape
    else:
        raise Exception("Only Booleans, Integers, Floats, and numpy Arrays may be passed.")
