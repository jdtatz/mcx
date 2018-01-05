import ctypes
import platform
import numpy as np

if platform.system() == 'Windows':
    _lib = ctypes.CDLL('mcx.dll')
    _libc = ctypes.cdll.msvcrt
else:
    _lib = ctypes.CDLL('mcx.so')
    _libc = CDLL("libc.so.6")

_malloc = _libc.malloc
_malloc.argtypes = [ctypes.c_size_t]
_malloc.restype = ctypes.c_void_p

_free = _libc.free
_free.argtypes = [ctypes.c_void_p]

_config_size = ctypes.c_int.in_dll(_lib, 'SIZE_OF_CONFIG')

_set_field = _lib.mcx_set_field
_set_field.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p)]
_set_field.restype = ctypes.c_int

_get_field = _lib.mcx_get_field
_get_field.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p)]
_get_field.restype = ctypes.c_void_p

_initcfg = _lib.mcx_initcfg
_initcfg.argtypes = [ctypes.c_void_p]

_init_output = _lib.initialize_output
_init_output.argtypes = [ctypes.c_void_p, ctypes.c_int]

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
        return ctypes.c_char(v), "char", 0, None
    elif isinstance(v, int):
        return ctypes.c_int(v), "int", 0, None
    elif isinstance(v, float):
        return ctypes.c_float(v), "float", 0, None
    elif isinstance(v, np.array):
        return v.ctypes, str(v.dtype), v.ndim, v.shape
    else:
        raise Exception("Only Booleans, Integers, Floats, and numpy Arrays may be passed.")

def run(opt, nout):
    cfg = _malloc(_config_size)
    gpuinfo = ctypes.c_void_p()
    _initcfg(cfg)
    err = ctypes.c_char_p()
    for key, val in opt:
        if _set_field(cfg, key, *_converter(val), ctypes.byref(err)):
            raise Exception(err)
    _flush(cfg)
    activedev = _list_gpu(cfg, ctypes.byref(gpuinfo))
    if not activedev:
        raise Eception("No active GPU device found")
    _init_output(cfg, nout)
    _validateconfig(cfg, err, 0, None, None)
    _run_simulation(cfg, gpuinfo)

    outs = []
    dtype, ndim, dims = ctypes.c_char_p(), c_int(), (ctypes.c_int*4)()
    if nout >= 1:
        temp = _get_field(cfg, "exportfield", dtype, ndim, dims, err)
        fleunce = np.ctypeslib.as_array(temp, dims).copy()
        outs.append(fleunce)
    elif nout >= 2:
        temp = _get_field(cfg, "exportdetected", dtype, ndim, dims, err)
        detphoton = np.ctypeslib.as_array(temp, dims).copy()
        outs.append(detphoton)

    _cleargpuinfo(gpuinfo)
    _clearcfg(cfg)
    return outs
