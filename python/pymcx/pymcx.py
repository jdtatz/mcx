# TODO Remove Python 3.6 support compat
# from __future__ import annotations
import os
import sys
from contextlib import contextmanager
from enum import IntEnum, IntFlag
from typing import NamedTuple, Optional

import numpy as np

from ._pymcx import ffi, lib
from .meta import Array, MetaStruct, field_annotation_docstrings


@contextmanager
def _grab_fstream(stream):
    fd = stream.fileno()
    pipe_out, pipe_in = os.pipe()
    copy = os.dup(fd)
    os.dup2(pipe_in, fd)
    hold = []
    yield hold
    end = b"\b"
    os.write(pipe_in, end)
    stream.flush()
    hold.append(b"".join(iter(lambda: (os.read(pipe_out, 1) or end), end)).decode())
    os.close(pipe_out)
    os.close(pipe_in)
    os.dup2(copy, fd)
    os.close(copy)


class SrcType(IntEnum):
    PENCIL = 0
    ISOTROPIC = 1
    CONE = 2
    GAUSSIAN = 3
    PLANAR = 4
    PATTERN = 5
    FOURIER = 6
    ARCSINE = 7
    DISK = 8
    FOURIERX = 9
    FOURIERX2D = 10
    ZGAUSSIAN = 11
    LINE = 12
    SLIT = 13
    PENCILARRAY = 14
    PATTERN3D = 15
    DISKARRAY = 16


class SaveFlags(IntFlag):
    DetectorId = 1
    NScatters = 2
    PartialPath = 4
    Momentum = 8
    ExitPosition = 16
    ExitDirection = 32
    InitialWeight = 64


class DetectedPhotons(NamedTuple):
    detector_id: Optional[np.ndarray] = None
    nscatters: Optional[np.ndarray] = None
    partial_path: Optional[np.ndarray] = None
    momentum: Optional[np.ndarray] = None
    exit_position: Optional[np.ndarray] = None
    exit_direction: Optional[np.ndarray] = None
    initial_weight: Optional[np.ndarray] = None


class Medium(NamedTuple):
    mua: float
    mus: float
    g: float
    n: float


Medium.dtype = np.dtype([(f, np.float32) for f in Medium._fields], align=True)


class uint3(NamedTuple):
    x: int = 0
    y: int = 0
    z: int = 0


uint3.dtype = np.dtype([(f, np.uint32) for f in uint3._fields], align=True)


class float3(NamedTuple):
    x: float = 0
    y: float = 0
    z: float = 0


float3.dtype = np.dtype([(f, np.float32) for f in float3._fields], align=True)


class float4(NamedTuple):
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 0


float4.dtype = np.dtype([(f, np.float32) for f in float4._fields], align=True)


def _to_structured(saveflags, nmedia, detp):
    fields = {}
    if SaveFlags.DetectorId in saveflags:
        fields["detector_id"] = detp[0]
        detp = detp[1:]
    if SaveFlags.NScatters in saveflags:
        fields["nscatters"] = detp[:nmedia]
        detp = detp[nmedia:]
    if SaveFlags.PartialPath in saveflags:
        fields["partial_path"] = detp[:nmedia]
        detp = detp[nmedia:]
    if SaveFlags.Momentum in saveflags:
        fields["momentum"] = detp[:nmedia]
        detp = detp[nmedia:]
    if SaveFlags.ExitPosition in saveflags:
        fields["exit_position"] = detp[:3]
        detp = detp[3:]
    if SaveFlags.ExitDirection in saveflags:
        fields["exit_direction"] = detp[:3]
        detp = detp[3:]
    if SaveFlags.InitialWeight in saveflags:
        fields["initial_weight"] = detp[:3]
        detp = detp[3:]
    return DetectedPhotons(**fields)


class MCXRunException(Exception):
    def __init__(self, output, error, flag):
        self.stdout = output
        self.stderr = error
        self.flag = flag
        super(MCXRunException, self).__init__(
            'RunTime error: "{}" with stderr\n{}'.format(flag, error)
        )


class MCXValidationError(Exception):
    pass


@field_annotation_docstrings
class MCX(metaclass=MetaStruct, ptr_field="_config", ctype=ffi.typeof("Config")):
    # _config: ffi.CData["Config *"]
    nphoton: int
    """Total simulated photon number"""
    seed: int
    """Integer to seed MCX's PRNG"""
    srcpos: float3  # Technically a float4, but mcx discards the fourth component 
    """Source position vector [grid unit]"""
    srcdir: float4
    """Source direction unit vector [grid unit]"""
    srctype: SrcType
    """Source type"""
    tstart: float
    """Start time [seconds]"""
    tstep: float
    """Time step [seconds]"""
    tend: float
    """End time [seconds]"""
    steps: float3
    dim: uint3
    maxdetphoton: int
    """Max number of detected photons that are saved"""
    detectedcount: int
    """Number of detected photons"""
    prop: Array[Medium, 1]
    """Medium properties {mua: [1/mm], mus: [1/mm], g: [unitless], n: [unitless]}"""
    detpos: Array[float4, 1]
    """Detector vector positions and radii [grid unit]"""
    vol: Array[np.uint32, 3, "F"]
    """Volume of property indices"""
    savedetflag: SaveFlags
    """Photon properties to save on detection"""
    issave2pt: bool
    """Save Fluence?"""
    issavedet: bool
    """Save detected photons?"""
    # TODO Totally change srcparam1, srcparam2, & srcpattern into diffrent
    # classes per enum to totally encapsulate the needed information
    srcparam1: float4
    srcparam2: float4
    # srcpattern: ????
    issrcfrom0: bool
    """Is the source position 0-indexed?"""
    isreflect: bool
    """Reflect at external boundaries?"""
    isrefint: bool
    """Reflect at internal boundaries?"""
    autopilot: bool
    """Optimally set `nblocksize` & `nthread`"""
    nblocksize: int
    """Thread block size"""
    nthread: int
    """Number of total threads, multiple of 128"""
    unitinmm: float
    """[grid unit] in [mm]"""
    # gpuid: int | Array[bool, 1]
    exportdetected: Array[np.float32, 2, "F"]
    exportfield: Array[np.float32, 4, "F"]

    def __init__(self, **kwargs):
        self._config = ffi.new("Config *")
        lib.mcx_initcfg(self._config)
        self._config.parentid = lib.mpPy
        self._config.mediabyte = 4
        self.fluence = None
        self.detphoton = None
        self.stdout = None
        self._gpuid = np.array([True])
        for k, v in kwargs.items():
            setattr(self, k, v)

    def validate(self):
        if self.vol is None:
            raise MCXValidationError("Simulation volume must be initialized")
        self.dim = self.vol.shape
        if self.prop is None:
            raise MCXValidationError("Simulation media properties must be initialized")
        self._config.medianum = len(self.prop)
        if self.tend < self.tstart:
            raise MCXValidationError("Simulation time end must be >= to time start")
        if self.tstep <= 0:
            raise MCXValidationError("Simulation time step must be > 0")

        if not self.issrcfrom0:
            self._config.srcpos.x -= 1
            self._config.srcpos.y -= 1
            self._config.srcpos.z -= 1
            self.issrcfrom0 = True
            if self.detpos is not None:
                for i in range(self.detpos.shape[0]):
                    self.detpos[i].x -= 1
                    self.detpos[i].y -= 1
                    self.detpos[i].z -= 1

        if self.unitinmm != 1:
            self.steps = self.unitinmm, self.unitinmm, self.unitinmm
            for i in range(self.prop.shape[0]):
                self.prop[i].mua *= self.unitinmm
                self.prop[i].mus *= self.unitinmm

        for i in range(self.prop.shape[0]):
            if self.prop[i].mus == 0:
                self.prop[i].mus = 1e-5

        if self.issave2pt:
            ntimegate = self._time_gates()
            self._config.maxgate = ntimegate
            self.fluence = np.zeros(
                (*self.vol.shape, ntimegate), dtype=np.float32, order="F"
            )
            self.exportfield = self.fluence
        if self.issavedet:
            if self.savedetflag == 0:
                raise MCXValidationError(
                    "Saving detectors is enabled, but no save flags have been set"
                )
            if self.detpos is None:
                raise MCXValidationError(
                    "Saving detectors is enabled, but detector positions is not initialized"
                )
            self._config.detnum = len(self.detpos)
            # buf len for media-specific data, copy from gpu to host
            partialdata = (self._config.medianum - 1) * (
                (1 if self.savedetflag & SaveFlags.NScatters else 0)
                + (1 if self.savedetflag & SaveFlags.PartialPath else 0)
                + (1 if self.savedetflag & SaveFlags.Momentum else 0)
            )
            # host-side det photon data buffer length
            hostdetreclen = (
                partialdata
                + (1 if self.savedetflag & SaveFlags.DetectorId else 0)
                + (3 if self.savedetflag & SaveFlags.ExitPosition else 0)
                + (3 if self.savedetflag & SaveFlags.ExitDirection else 0)
                + (1 if self.savedetflag & SaveFlags.InitialWeight else 0)
            )
            self.exportdetected = np.zeros(
                (hostdetreclen, self.maxdetphoton), dtype=np.float32, order="F"
            )
            # apply detector bit mask to volume
            lib.mcx_maskdet(self._config)

    def run(self):
        self.validate()
        flag = 0
        with _grab_fstream(sys.__stderr__) as hold_stderr:
            with _grab_fstream(sys.__stdout__) as hold_stdout:
                flag = lib.mcx_wrapped_run_simulation(self._config)
        if flag != 0:
            raise MCXRunException(hold_stdout.pop(), hold_stderr.pop(), flag)
        else:
            self.stdout = hold_stdout.pop()
            if self.exportdetected is not None:
                self.detphoton = _to_structured(
                    self.savedetflag,
                    len(self.prop) - 1,
                    self.exportdetected[:, : self.detectedcount],
                )
                # unmask volume
                self.vol &= 0x7FFFFFFF
            if self.unitinmm != 1:
                for i in range(len(self.prop)):
                    self.prop[i].mua /= self.unitinmm
                    self.prop[i].mus /= self.unitinmm
            # needed b/c mcx loves to mess with this
            self.gpuid = self._gpuid

    def _time_gates(self):
        return int(
            np.ceil((self._config.tend - self._config.tstart) / self._config.tstep)
        )

    @property
    def gpuid(self):
        return self._gpuid

    @gpuid.setter
    def gpuid(self, value):
        i, n = 0, 0
        if self._gpuid is not None:
            for i in range(self._gpuid.shape[0]):
                self._config.deviceid[i] = b"0"
        if isinstance(value, int):
            self._config.gpuid = value
            n = value
            self._gpuid = np.zeros(1 + n, dtype=np.bool_)
            self._gpuid[n] = True
        else:
            self._config.gpuid = 0
            self._gpuid = np.asarray(value, dtype=np.bool_)
        for i in range(self._gpuid.shape[0]):
            self._config.deviceid[i] = b"1" if self._gpuid[i] else b"0"

    def __getstate__(self):
        config = {k: getattr(self, k) for k in dir(self) if not k.startswith("_")}
        return {k: v for k, v in config.items() if not callable(v) and v is not None}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
