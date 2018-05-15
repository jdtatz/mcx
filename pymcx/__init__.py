from __future__ import unicode_literals
import ctypes
import os
import sys
import platform
import numpy as np
from contextlib import contextmanager


@contextmanager
def _grab_fstream(stream):
    fd = stream.fileno()
    pipe_out, pipe_in = os.pipe()
    copy = os.dup(fd)
    os.dup2(pipe_in, fd)
    hold = []
    yield hold
    end = b'\b'
    os.write(pipe_in, end)
    stream.flush()
    hold.append(b''.join(iter(lambda: (os.read(pipe_out, 1) or end), end)).decode())
    os.close(pipe_out)
    os.close(pipe_in)
    os.dup2(copy, fd)
    os.close(copy)


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



class MCX(object):
    def __init__(self, **kws):
        super(MCX, self).__setattr__('_cfg_ptr', _create_config())
        super(MCX, self).__setattr__('_config', dict())
        for key, val in kws.items():
            setattr(self, key, val)

    def __setattr__(self, key, v):
        ndim, dims, order = 0, None, b'C'
        if isinstance(v, bytes) or isinstance(v, str):
            val = v if isinstance(v, bytes) else (v.encode('ASCII'))
            ptr = ctypes.c_char_p(val)
            dtype = b"string"
            ndim = 1
            dims = ctypes.pointer(ctypes.c_int(len(val)+1))
        else:
            arr = np.asanyarray(v)
            if not arr.flags.f_contiguous and not arr.flags.c_contiguous:
                raise Exception('Numpy arrays must be contiguous')
            ptr = arr.ctypes
            dtype = str(arr.dtype).encode('ASCII')
            ndim = arr.ndim
            dims = (ctypes.c_int*ndim)(*arr.shape)
            order = arr.flags.f_contiguous and b'F' or b'C'
        err = ctypes.c_char_p()
        if _set_field(self._cfg_ptr, key.encode('ASCII'), ptr, dtype, ndim, dims, order, ctypes.byref(err)) != 0:
            excep = 'Issue with setting "{}" to ({}) with error "{}".'.format(key, v, err.value.decode('ASCII') if err.value else "Unknown Error")
            raise Exception(excep)
        super(MCX, self).__setattr__(key, v)
        self._config[key] = v

    def get_field(self, key):
        err = ctypes.c_char_p()
        dtype, ndim, dims = ctypes.c_char_p(), ctypes.c_int(), (ctypes.c_int*4)()
        temp = _get_field(self._cfg_ptr, key.encode('ASCII'), ctypes.byref(dtype), ctypes.byref(ndim), dims, ctypes.byref(err))
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
        elif dtype == "char":
            ptr = ctypes.cast(temp, ctypes.POINTER(ctypes.c_char))
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

    def __getattr__(self, key):
        return self.get_field(key)

    def run(self, nout=2):
        err = ctypes.c_char_p()
        with _grab_fstream(sys.__stderr__) as hold_stderr:
            with _grab_fstream(sys.__stdout__) as hold_stdout:
                flag = _run_simulation(self._cfg_ptr, nout, ctypes.byref(err))
        if flag != 0:
            msg = hold_stderr.pop() if flag > 0 else (err.value.decode('ASCII') if err.value else "Unknown Error")
            excep = 'RunTime error: "{}"'.format(msg)
            raise Exception(excep)
        basic_fields = ["runtime", "nphoton", "energytot", "energyabs", "normalizer", "workload"]
        result = {field: self.get_field(field) for field in basic_fields}
        if nout >= 1:
            result['fluence'] = self.get_field("exportfield")
        if nout >= 2:
            result['detphoton'] = self.get_field("exportdetected")
        if nout >= 3:
            result['vol'] = self.get_field("vol")
        if nout >= 4:
            result['seeds'] = self.get_field("seeddata")
        if nout >= 5:
            result['trajectory'] = self.get_field("exportdebugdata")
        result['stdout'] = hold_stdout.pop()
        return result

    def __del__(self):
        _destroy_config(self._cfg_ptr)

    def __getstate__(self):
        return self._config.copy()
    
    def __setstate__(self, state):
        self.__init__(**state)

