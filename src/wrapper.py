import ctypes
import platform
import numpy as np

if platform.system() == 'Windows':
    _lib = ctypes.WinDLL('mcx.dll')
else:
    _lib = ctypes.CDLL('./mcx.so')

_create_config = _lib.mcx_create_config
_create_config.restype = ctypes.c_void_p

_destroy_config = _lib.mcx_destroy_config
_destroy_config.argtypes = [ctypes.c_void_p]

_set_field = _lib.mcx_set_field
_set_field.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p)]
_set_field.restype = ctypes.c_int

_get_field = _lib.mcx_get_field
_get_field.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p)]
_get_field.restype = ctypes.c_void_p

_run_simulation = _lib.mcx_wrapped_run_simulation
_run_simulation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]


def _converter(v):
    def _dtyper(s):
        if s == 'float32':
            return 'float'
        elif s == 'int32':
            return 'int'
        elif s == 'uint32':
            return 'uint'
        else:
            return s
    if v is True or v is False:
        return ctypes.byref(ctypes.c_char(v)), b"char", 0, None
    elif isinstance(v, int):
        return ctypes.byref(ctypes.c_int(v)), b"int", 0, None
    elif isinstance(v, float):
        return ctypes.byref(ctypes.c_float(v)), b"float", 0, None
    elif isinstance(v, np.ndarray):
        return v.ctypes, _dtyper(str(v.dtype)).encode('ASCII'), v.ndim, (ctypes.c_int*len(v.shape))(*v.shape)
    else:
        raise Exception("Only Booleans, Integers, Floats, and numpy Arrays may be passed.")

def run(opt, nout):
    cfg = _create_config()
    err = ctypes.c_char_p()
    for key, val in opt.items():
        if _set_field(cfg, key.encode('ASCII'), *_converter(val), ctypes.byref(err)) != 0:
            print(key, val)
            raise Exception(err.value.decode('ASCII'))
    if _run_simulation(cfg, nout, ctypes.byref(err)) != 0:
        raise Exception(err.value.decode('ASCII'))
    outs = []
    dtype, ndim, dims = ctypes.c_char_p(), ctypes.c_int(), (ctypes.c_int*4)()
    if nout >= 1:
        temp = _get_field(cfg, "exportfield".encode('ASCII'), ctypes.byref(dtype), ctypes.byref(ndim), dims, ctypes.byref(err))
        ptr = ctypes.cast(temp, ctypes.POINTER(ctypes.c_float))
        shape = tuple(dims[i] for i in range(min(ndim.value, 4)))
        fleunce = np.ctypeslib.as_array(ptr, shape).copy()
        outs.append(fleunce)
    if nout >= 2:
        temp = _get_field(cfg, "exportdetected".encode('ASCII'), ctypes.byref(dtype), ctypes.byref(ndim), dims, ctypes.byref(err))
        ptr = ctypes.cast(temp, ctypes.POINTER(ctypes.c_float))
        shape = tuple(dims[i] for i in range(min(ndim.value, 4)))
        detphoton = np.ctypeslib.as_array(ptr, shape).copy()
        outs.append(detphoton)
    _destroy_config(cfg)
    return outs

if __name__ == "__main__":
    xdet = np.arange(100+5, 100+40, 5, np.float32)
    ydet = 101*np.ones(len(xdet), np.float32)
    zdet = np.ones(len(xdet), np.float32)
    raddet = np.ones(len(xdet), np.float32)
    detpos = np.stack((xdet, ydet, zdet, raddet)).T
    cfg = {"isrowmajor":True, "issrcfrom0": True, "nphoton": 3e6, "maxdetphoton": 1e6, "isreflect":False, "prop": np.array([[0,0,1,1.37],[0.01,10,0.9,1.37]], np.float32),
           "vol":np.ones((200,200,200), np.uint32), "srcpos": np.array([100, 100, 1], np.float32), "srcdir": np.array([0,0,1], np.float32),
           "detpos": detpos, "tstart": 0, "tend": 5e-9, "tstep": 1e-10, "autopilot": True, "gpuid": 1}
    print(run(cfg, 2))
