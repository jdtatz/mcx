from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source("_pymcx","""\
#include <vector_types.h>
#include "mcx_utils.h"
#include "mcx_core.h"
""", py_limited_api=True)

ffibuilder.cdef("""\
typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} uint3;

typedef struct {
    float x;
    float y;
    float z;
} float3;

typedef struct {
    float x;
    float y;
    float z;
    float w;
    ...;
} float4;

typedef struct {
    float mua;
    float mus;
    float g;
    float n;
} Medium;

typedef struct {
    size_t nphoton;
    unsigned int nblocksize;
	unsigned int nthread;
    int seed;

    float4 srcpos;
    float4 srcdir;
    float tstart;
    float tstep;
    float tend;
    float3 steps;

    uint3 dim;
    unsigned int medianum;
    unsigned int detnum;
    unsigned int maxdetphoton;

    Medium* prop;
    float4* detpos;
    int maxgate;

    int gpuid;
    unsigned int *vol;

    char session[...];
    unsigned char isreflect;
    unsigned char isrefint;
    unsigned char issavedet;
    unsigned char issave2pt;
    unsigned char issrcfrom0;
    unsigned char autopilot;
    unsigned char issaveseed;

    unsigned char srctype;

    float unitinmm;

    FILE *flog;

    float *exportfield;
    float *exportdetected;
    unsigned long int detectedcount;
    float4 srcparam1;
    float4 srcparam2;
    unsigned int srcnum;
    float* srcpattern;

    void *seeddata;
    unsigned int savedetflag;
    unsigned int mediabyte;
    char deviceid[...];
    float workload[...];
    int parentid;

    double energytot;
    double energyabs;
    double energyesc;
    float normalizer;
    float *exportdebugdata;

    char bc[8];

    ...;

} Config;

typedef ... GPUInfo;

const int mpPy;
void mcx_initcfg(Config *cfg);
void mcx_clearcfg(Config *cfg);
int mcx_wrapped_run_simulation(Config *cfg);
void mcx_cleargpuinfo(GPUInfo **gpuinfo);
void mcx_maskdet(Config *cfg);

int mcx_list_gpu(Config *cfg, GPUInfo **info);
void mcx_run_simulation(Config *cfg, GPUInfo *gpu);

""")

if __name__ == "__main__":
    ffibuilder.emit_c_code("_pymcx.c")
