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


def _dtyper(s):
    if s == 'float32':
        return b'float'
    elif s == 'float64':
        return b'double'
    elif s == 'int32':
        return b'int'
    elif s == 'uint32':
        return b'uint'
    else:
        return s.encode('ASCII')


def _converter(v):
    if v is True or v is False:
        return ctypes.byref(ctypes.c_char(v)), b"char", 0, None
    elif isinstance(v, int):
        return ctypes.byref(ctypes.c_int(v)), b"int", 0, None
    elif isinstance(v, float):
        return ctypes.byref(ctypes.c_float(v)), b"float", 0, None
    elif isinstance(v, str):
        return ctypes.c_char_p(v.encode('ASCII')), b"string", 1, ctypes.pointer(ctypes.c_int(len(v)+1))
    elif isinstance(v, np.ndarray):
        return v.ctypes, _dtyper(str(v.dtype)), v.ndim, (ctypes.c_int*v.ndim)(*v.shape)
    else:
        raise Exception("Only Booleans, Integers, Floats, and numpy Arrays may be passed.")



class MCX:
    def __init__(self, **kws):
        super().__setattr__('_cfg', _create_config())
        for key, val in kws.items():
            setattr(self, key, val)

    def __setattr__(self, key, value):
        err = ctypes.c_char_p()
        if _set_field(self._cfg, key.encode('ASCII'), *_converter(value), ctypes.byref(err)) != 0:
            excep = "Issue with setting {} to {} with error {}.".format(key, value, err.value.decode('ASCII'))
            raise Exception(excep)

    def __getattr__(self, key):
        err = ctypes.c_char_p()
        dtype, ndim, dims = ctypes.c_char_p(), ctypes.c_int(), (ctypes.c_int*4)()
        temp = _get_field(cfg, "exportdetected".encode('ASCII'), ctypes.byref(dtype), ctypes.byref(ndim), dims, ctypes.byref(err))
        if temp is None:
            excep = "Issue with gettint {} with error {}.".format(key, err.value.decode('ASCII'))
            raise Exception(excep)
        dtype = dtype.value.decode('ASCII')
        if dtype == "int":
            ptr = ctypes.cast(temp, ctypes.POINTER(ctypes.c_int))
        elif dtype == "uint":
            ptr = ctypes.cast(temp, ctypes.POINTER(ctypes.c_uint))
        elif dtype == "float":
            ptr = ctypes.cast(temp, ctypes.POINTER(ctypes.c_float))
        elif dtype == "double":
            ptr = ctypes.cast(temp, ctypes.POINTER(ctypes.c_double))
        elif dtype == "uint8":
            ptr = ctypes.cast(temp, ctypes.POINTER(ctypes.c_uint8))
        else:
            raise Exception('Unknown type returned')
        ndim = ndim.value
        if ndim == 0:
            return ptr[0]
        else:
            shape = tuple(dims[i] for i in range(min(ndim, 4)))
            return np.ctypeslib.as_array(ptr, shape).copy()

    def run(self, nout):
        err = ctypes.c_char_p()
        if _run_simulation(self._cfg, nout, ctypes.byref(err)) != 0:
            excep = "RunTime error: {}".format(err.value.decode('ASCII'))
            raise Exception(excep)

    def __del__(self):
        _destroy_config(self._cfg)


if __name__ == "__main__":
    cfg = MCX(isrowmajor = True, issrcfrom0=True, isreflect=False, autopilot=True, gpuid=1)
    cfg.nphoton = 3e6
    cfg.maxdetphoton = 1e6
    cfg.tstart = 0
    cfg.tend = 5e-9
    cfg.tstep = 1e-10

    cfg.prop = np.array([[0,0,1,1.37],[0.01,10,0.9,1.37]], np.float32)
    cfg.vol = np.ones((200,200,200), np.uint32)
    cfg.srcpos = np.array([100, 100, 1], np.float32)
    cfg.srcdir = np.array([0,0,1], np.float32)
    
    xdet = np.arange(100+5, 100+40, 5, np.float32)
    ydet = 101*np.ones(len(xdet), np.float32)
    zdet = np.ones(len(xdet), np.float32)
    raddet = np.ones(len(xdet), np.float32)
    cfg.detpos = np.stack((xdet, ydet, zdet, raddet)).T

    cfg.run(2)
    fleunce, detphoton = cfg.exportfield, cfg.exportdetected
    print(fleunce, detphoton)
