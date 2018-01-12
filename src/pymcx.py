import ctypes
import os
import platform
import numpy as np

if platform.system() == 'Windows':
    _libname = 'libmcx.dll'
elif platform.system() == 'Darwin':
    _libname = 'libmcx.dylib'
else:
    _libname = 'libmcx.so'

_libpath = os.path.join(os.path.dirname(__file__), _libname)
_lib = ctypes.CDLL(_libpath)

_create_config = _lib.mcx_create_config
_create_config.restype = ctypes.c_void_p

_destroy_config = _lib.mcx_destroy_config
_destroy_config.argtypes = [ctypes.c_void_p]

_set_field = _lib.mcx_set_field
_set_field.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
_set_field.restype = ctypes.c_int

_get_field = _lib.mcx_get_field
_get_field.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p)]
_get_field.restype = ctypes.c_void_p

_run_simulation = _lib.mcx_wrapped_run_simulation
_run_simulation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]


def _converter(v):
    if v is True or v is False:
        return ctypes.byref(ctypes.c_char(v)), b"char", 0, None, b'C'
    elif isinstance(v, int):
        return ctypes.byref(ctypes.c_int(v)), b"int", 0, None, b'C'
    elif isinstance(v, float):
        return ctypes.byref(ctypes.c_float(v)), b"float", 0, None, b'C'
    elif isinstance(v, str):
        return ctypes.c_char_p(v.encode('ASCII')), b"string", 1, ctypes.pointer(ctypes.c_int(len(v)+1)), b'C'
    elif isinstance(v, np.ndarray):
        if not v.flags.f_contiguous and not v.flags.c_contiguous:
            raise Exception('Numpy arrays must be contiguous')
        return v.ctypes, str(v.dtype).encode('ASCII'), v.ndim, (ctypes.c_int*v.ndim)(*v.shape), v.flags.f_contiguous and b'F' or b'C'
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
            excep = 'Issue with setting "{}" to ({}) with error "{}".'.format(key, value, err.value.decode('ASCII') if err.value else "Unknown Error")
            raise Exception(excep)
        super().__setattr__(key, value)

    def __getattr__(self, key):
        err = ctypes.c_char_p()
        dtype, ndim, dims = ctypes.c_char_p(), ctypes.c_int(), (ctypes.c_int*4)()
        temp = _get_field(self._cfg, key.encode('ASCII'), ctypes.byref(dtype), ctypes.byref(ndim), dims, ctypes.byref(err))
        if temp is None:
            excep = 'Issue with getting "{}" with error "{}".'.format(key, err.value.decode('ASCII') if err.value else "Unknown Error")
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
        elif ndim == 1:
            return np.ctypeslib.as_array(ptr, (dims[0],)).copy()
        else:
            shape = tuple(dims[i] for i in range(ndim))
            size = np.prod(shape)
            return np.ctypeslib.as_array(ptr, (size,)).reshape(shape, order='F').copy()

    def run(self, nout):
        err = ctypes.c_char_p()
        if _run_simulation(self._cfg, nout, ctypes.byref(err)) != 0:
            excep = 'RunTime error: "{}"'.format(err.value.decode('ASCII') if err.value else "Unknown Error")
            raise Exception(excep)
        outs = [self.exportfield]
        if nout >= 5:
            outs.append(self.exportdebugdata)
        if nout >= 4:
            outs.append(self.seeddata)
        if nout >= 3:
            outs.append(self.vol)
        if nout >= 2:
            outs.append(self.exportdetected)
        return outs

    def __del__(self):
        _destroy_config(self._cfg)
