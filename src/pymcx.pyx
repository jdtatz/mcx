from libc.stdio cimport FILE, stderr, fprintf
from libc.setjmp cimport setjmp, jmp_buf
from openmp cimport omp_set_num_threads
from cython.parallel cimport parallel, prange
from cython cimport view
import os, sys
from contextlib import contextmanager
from enum import IntEnum, IntFlag
import numpy as np

__all__ = ['MCX', 'MCXRunException', 'SaveFlags', 'SrcType']


cdef extern from "<vector_types.h>":
    ctypedef struct uint3:
        unsigned int x
        unsigned int y
        unsigned int z

    ctypedef struct float3:
        float x
        float y
        float z

    ctypedef struct float4:
        float x
        float y
        float z
        float w


cdef extern from "mcx_utils.h":
    ctypedef struct Medium:
        float mua
        float mus
        float g
        float n

    ctypedef struct Config:
        size_t nphoton
        unsigned int nblocksize
        unsigned int nthread
        int seed
        float4 srcpos
        float4 srcdir
        float tstart
        float tstep
        float tend
        float3 steps
        uint3 dim
        unsigned int medianum
        unsigned int detnum
        unsigned int maxdetphoton
        float detradius
        float sradius
        Medium* prop
        float4* detpos
        int maxgate

        int gpuid
        unsigned int *vol

        char isreflect
        char isref3
        char isrefint
        char isnormalized
        char issavedet
        char issave2pt
        char isspecular
        char issrcfrom0
        char isdumpmask
        char autopilot
        char issaveseed
        char issaveexit
        char issaveref
        char ismomentum
        char isdumpjson
        char internalsrc
        char isgpuinfo

        char srctype

        char faststep
        float minenergy
        float unitinmm
        FILE *flog

        float *exportfield
        float *exportdetected
        unsigned long int detectedcount
        int maxvoidstep
        int voidtime
        float4 srcparam1
        float4 srcparam2
        unsigned int srcnum
        float* srcpattern

        void *seeddata
        int replaydet
        unsigned int debuglevel
        unsigned int savedetflag
        char* deviceid
        float* workload
        int parentid
        unsigned int runtime

        double energytot
        double energyabs
        double energyesc
        float normalizer
        unsigned int maxjumpdebug
        unsigned int debugdatalen
        unsigned int gscatter
        float *exportdebugdata

        char[8] bc

    ctypedef struct GPUInfo:
        pass

    void mcx_initcfg(Config *cfg) nogil
    void mcx_clearcfg(Config *cfg) nogil
    void mcx_set_error_handler(jmp_buf * bufp) nogil
    void mcx_cleargpuinfo(GPUInfo **gpuinfo) nogil

cdef extern from "mcx_core.h":
    int mcx_list_gpu(Config *cfg, GPUInfo **info) nogil
    void mcx_run_simulation(Config *cfg,GPUInfo *gpu) nogil


cdef int mcx_wrapped_run_simulation_inner(Config *cfg, GPUInfo *gpuinfo) nogil:
    cdef jmp_buf errHandler
    cdef int jmp_flag
    jmp_flag = setjmp(errHandler)
    if jmp_flag == 0:
        mcx_set_error_handler(&errHandler)
        mcx_run_simulation(cfg, gpuinfo)
    else:
        # error
        pass
    mcx_set_error_handler(NULL)
    return jmp_flag

cdef int mcx_wrapped_run_simulation(Config *cfg):
    cdef GPUInfo *gpuinfo
    cdef Py_ssize_t i
    cdef int activedev = mcx_list_gpu(cfg, &gpuinfo)
    cdef int[::1] errorflags

    if activedev == 0:
        fprintf(stderr, "\nMCX ERROR: No active GPU device found\n")
        mcx_cleargpuinfo(&gpuinfo)
        return -1
    errorflags = view.array(shape=(activedev,), itemsize=sizeof(int), format="i", mode="c", allocate_buffer=True)

    omp_set_num_threads(activedev)
    for i in prange(activedev, nogil=True, num_threads=activedev):
        errorflags[i] = mcx_wrapped_run_simulation_inner(cfg, gpuinfo)

    mcx_cleargpuinfo(&gpuinfo)
    for flag in errorflags:
        if flag != 0:
            return flag
    return 0


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


class MCXRunException(Exception):
    def __init__(self, stdout, stderr, flag):
        self.stdout, self.stderr, self.flag = stdout, stderr, flag
        super(MCXRunException, self).__init__('RunTime error: "{}" with stderr\n{}'.format(flag, stderr))


cdef class MCX:
    cdef Config config
    cdef float[:, ::1] prop
    cdef float[:, ::1] detpos
    cdef unsigned int[::1, :, :] volume
    cdef float[:, :] srcpattern
    cdef readonly object fluence
    cdef float[::1, :, :, :] exportfield
    cdef readonly object detphoton
    cdef float[::1, :] exportdetected
    # cdef readonly object seeds
    # TODO check shape & type
    # cdef float[::1, :] seeddata
    cdef readonly object stdout

    def __cinit__(self):
        mcx_initcfg(&self.config)

    def __init__(self):
        self.fluence = None
        self.detphoton = None
        # self.seeds = None
        self.stdout = None
        self.seed = np.random.rand()

    def validate(self):
        assert self.volume is not None
        assert self.prop is not None
        assert self.config.tend >= self.config.tstart and self.config.tstep > 0
        assert np.allclose((self.config.srcdir.x**2 + self.config.srcdir.y**2 + self.config.srcdir.z**2), 1)

        if not self.issrcfrom0:
            self.config.srcpos.x -= 1
            self.config.srcpos.y -= 1
            self.config.srcpos.z -= 1
            self.issrcfrom0 = True
            if self.detpos is not None:
                for i in range(self.detpos.shape[0]):
                    self.detpos[i, 0] -= 1
                    self.detpos[i, 1] -= 1
                    self.detpos[i, 2] -= 1

        if self.config.unitinmm != 1:
            self.config.steps.x = self.config.unitinmm
            self.config.steps.y = self.config.unitinmm
            self.config.steps.z = self.config.unitinmm
            for i in range(self.prop.shape[0]):
                self.prop[i, 0] *= self.config.unitinmm
                self.prop[i, 1] *= self.config.unitinmm

        for i in range(self.prop.shape[0]):
            if self.prop[i, 1] == 0:
                self.prop[i, 1] = 1e-5

        if self.config.issave2pt:
            ntimegate = self.time_gates
            self.config.maxgate = self.time_gates
            self.fluence = np.zeros((self.vol.shape[0], self.vol.shape[1], self.vol.shape[2], ntimegate), dtype=np.float32, order='F')
            self.exportfield = self.fluence
            self.config.exportfield = &self.exportfield[0,0,0,0]
        if self.config.issavedet or self.savedetflag:
            self.config.issavedet = True
            assert self.detpos is not None
            # buf len for media-specific data, copy from gpu to host
            partialdata = (self.config.medianum - 1) * (
                (1 if self.savedetflag & SaveFlags.NScatters else 0) +
                (1 if self.savedetflag & SaveFlags.PartialPath else 0) +
                (1 if self.savedetflag & SaveFlags.Momentum else 0)
            )
            # host-side det photon data buffer length
            hostdetreclen = (
                partialdata +
                (1 if self.savedetflag & SaveFlags.DetectorId else 0) +
                (3 if self.savedetflag & SaveFlags.ExitPosition else 0) +
                (3 if self.savedetflag & SaveFlags.ExitDirection else 0) +
                (1 if self.savedetflag & SaveFlags.InitialWeight else 0)
            )
            self.detphoton = np.zeros((hostdetreclen, self.maxdetphoton), dtype=np.float32, order='F')
            self.exportdetected = self.detphoton
            self.config.exportdetected = &self.exportdetected[0, 0]

    def run(self):
        self.validate()
        cdef int flag
        with _grab_fstream(sys.__stderr__) as hold_stderr:
            with _grab_fstream(sys.__stdout__) as hold_stdout:
                flag = mcx_wrapped_run_simulation(&self.config)
        if flag != 0:
            raise MCXRunException(hold_stdout.pop(), hold_stderr.pop(), flag)
        else:
            self.stdout = hold_stdout.pop()

    @property
    def nphoton(self):
        return self.config.nphoton

    @nphoton.setter
    def nphoton(self, value):
        self.config.nphoton = value

    @property
    def seed(self):
        return self.config.seed

    @seed.setter
    def seed(self, value):
        self.config.seed = value

    @property
    def maxdetphoton(self):
        return self.config.maxdetphoton

    @maxdetphoton.setter
    def maxdetphoton(self, value):
        self.config.maxdetphoton = value

    @property
    def srcpos(self):
        return self.config.srcpos.x, self.config.srcpos.y, self.config.srcpos.z

    @srcpos.setter
    def srcpos(self, value):
        self.config.srcpos.x, self.config.srcpos.y, self.config.srcpos.z = value
        self.config.srcpos.w = 0

    @property
    def srcdir(self):
        return self.config.srcdir.x, self.config.srcdir.y, self.config.srcdir.z

    @srcdir.setter
    def srcdir(self, value):
        self.config.srcdir.x, self.config.srcdir.y, self.config.srcdir.z = value
        self.config.srcdir.w = 0

    @property
    def srctype(self):
        return SrcType(self.config.srctype)

    @srctype.setter
    def srctype(self, value):
        self.config.srctype = value

    @property
    def tstart(self):
        return self.config.tstart

    @tstart.setter
    def tstart(self, value):
        self.config.tstart = value

    @property
    def tstep(self):
        return self.config.tstep

    @tstep.setter
    def tstep(self, value):
        self.config.tstep = value

    @property
    def tend(self):
        return self.config.tend

    @tend.setter
    def tend(self, value):
        self.config.tend = value

    @property
    def time_gates(self):
        return <int> np.ceil((self.config.tend - self.config.tstart) / self.config.tstep)

    @property
    def prop(self):
        return self.prop

    @prop.setter
    def prop(self, float[:, ::1] value not None):
        if value.shape[1] != 4:
            raise TypeError("Medium properties must be a C-continuous float array of shape [medianum x 4]")
        self.prop = value
        self.config.medianum = value.shape[0]
        self.config.prop = <Medium*> &self.prop[0, 0]

    @property
    def detpos(self):
        return self.detpos

    @detpos.setter
    def detpos(self, float[:, ::1] value not None):
        if value.shape[1] != 4:
            raise TypeError("Detector positions must be a C-continuous float array of shape [detnum x 4]")
        self.detpos = value
        self.config.detnum = value.shape[0]
        self.config.detpos = <float4*> &self.detpos[0, 0]

    @property
    def vol(self):
        return self.volume

    @vol.setter
    def vol(self, unsigned int[::1, :, :] value not None):
        self.volume = value
        self.config.dim.x = value.shape[0]
        self.config.dim.y = value.shape[1]
        self.config.dim.z = value.shape[2]
        self.config.vol = &self.volume[0,0,0]

    @property
    def savedetflag(self):
        return SaveFlags(self.config.savedetflag)

    @savedetflag.setter
    def savedetflag(self, value):
        self.config.savedetflag = value

    @property
    def issave2pt(self):
        return <bint> self.config.issave2pt

    @issave2pt.setter
    def issave2pt(self, bint value):
        self.config.issave2pt = value

    @property
    def issavedet(self):
        return <bint> self.config.issavedet

    @issavedet.setter
    def issavedet(self, bint value):
        self.config.issavedet = value

    # TODO impl issaveseed correctly
    # @property
    # def issaveseed(self):
    #     return <bint> self.config.issaveseed
    #
    # @issaveseed.setter
    # def issaveseed(self, bint value):
    #     self.config.issaveseed = value

    # TODO Totally change srcparam1, srcparam2, & srcpattern into diffrent
    # classes per enum to totally encapsulate the needed information

    @property
    def srcparam1(self):
        return self.config.srcparam1.x, self.config.srcparam1.y, self.config.srcparam1.z, self.config.srcparam1.w

    @srcparam1.setter
    def srcparam1(self, value):
        self.config.srcparam1.x, self.config.srcparam1.y, self.config.srcparam1.z, self.config.srcparam1.w = value

    @property
    def srcparam2(self):
        return self.config.srcparam2.x, self.config.srcparam2.y, self.config.srcparam2.z, self.config.srcparam2.w

    @srcparam2.setter
    def srcparam2(self, value):
        self.config.srcparam2.x, self.config.srcparam2.y, self.config.srcparam2.z, self.config.srcparam2.w = value

    @property
    def srcpattern(self):
        raise self.srcpattern

    @srcpattern.setter
    def srcpattern(self, float[:, :] value not None):
        self.srcpattern = value
        self.config.srcpattern = &self.srcpattern[0,0]

    @property
    def issrcfrom0(self):
        return <bint> self.config.issrcfrom0

    @issrcfrom0.setter
    def issrcfrom0(self, bint value):
        self.config.issrcfrom0 = value

    @property
    def isreflect(self):
        return <bint> self.config.isreflect

    @isreflect.setter
    def isreflect(self, bint value):
        self.config.isreflect = value

    @property
    def autopilot(self) -> bool:
        return <bint> self.config.autopilot

    @autopilot.setter
    def autopilot(self, bint value: bool):
        self.config.autopilot = value

    @property
    def unitinmm(self):
        return self.config.unitinmm

    @unitinmm.setter
    def unitinmm(self, value):
        self.config.unitinmm = value
