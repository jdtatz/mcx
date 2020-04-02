#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "mcx_utils.h"
#include "mcx_core.h"
#include "mcx_shapes.h"
#include "mcx_const.h"
#include "mcx_ffi.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(USE_XORSHIFT128P_RAND)
#define RAND_WORD_LEN 4
#elif defined(USE_POSIX_RAND)
#define RAND_WORD_LEN 4
#elif defined(USE_MT_RAND)
#define RAND_WORD_LEN 0
#else
#define RAND_WORD_LEN 5
#endif


Config * mcx_create_config(){
    Config * cfg = malloc(sizeof(Config));
    mcx_initcfg(cfg);
    cfg->parentid = mpFFI;
    cfg->savedetflag = SET_SAVE_DETID(cfg->savedetflag);
    return cfg;
}

void mcx_destroy_config(Config * cfg){
    mcx_clearcfg(cfg);
    free(cfg);
}

#define SET_SCALAR_HELPER(DST, SRC) {DST = *SRC;}

#define SET_VECTOR_HELPER(DST, SRC) for(uint i=0; i < dims[0]; i++){DST[i] = SRC[i];}

#define INDEX_2D_C(i, j) (i*dims[1]+j)
#define INDEX_2D_F(i, j) (i+j*dims[0])
#define SET_MATRIX_LOOP(DST, SRC, DST_ORD, SRC_ORD) for(uint i=0; i < dims[0]; i++){for(uint j=0; j < dims[1]; j++){DST[INDEX_2D_##DST_ORD(i, j)]=SRC[INDEX_2D_##SRC_ORD(i, j)];}}
#define SET_MATRIX_2C(DST, SRC) if(*order=='C') {SET_MATRIX_LOOP(DST, SRC, C, C)} else {SET_MATRIX_LOOP(DST, SRC, C, F)}
#define SET_MATRIX_2F(DST, SRC) if(*order=='F') {SET_MATRIX_LOOP(DST, SRC, F, F)} else {SET_MATRIX_LOOP(DST, SRC, F, C)}

#define INDEX_3D_C(i, j, k) (i*dims[1]*dims[2]+j*dims[2]+k)
#define INDEX_3D_F(i, j, k) (i+j*dims[0]+k*dims[0]*dims[1])
#define SET_VOL_LOOP(DST, SRC, DST_ORD, SRC_ORD) for(uint i=0; i < dims[0]; i++){for(uint j=0; j < dims[1]; j++){for(uint k=0; k < dims[2]; k++){DST[INDEX_3D_##DST_ORD(i, j, k)]=SRC[INDEX_3D_##SRC_ORD(i, j, k)];}}}
#define SET_VOL_2C(DST, SRC) if(*order=='C') {SET_VOL_LOOP(DST, SRC, C, C)} else {SET_VOL_LOOP(DST, SRC, C, F)}
#define SET_VOL_2F(DST, SRC) if(*order=='F') {SET_VOL_LOOP(DST, SRC, F, F)} else {SET_VOL_LOOP(DST, SRC, F, C)}

#define TYPE_SAFE_PTR_SETTER(DST, SETTER) \
if (strcmp(dtype, "int") == 0) {\
SETTER(DST, ((int*)value))\
} else if (strcmp(dtype, "uint") == 0) {\
SETTER(DST, ((uint*)value))\
} else if (strcmp(dtype, "float") == 0 || strcmp(dtype, "float32") == 0 || strcmp(dtype, "single") == 0) {\
SETTER(DST, ((float*)value))\
} else if (strcmp(dtype, "double") == 0 || strcmp(dtype, "float64") == 0) {\
SETTER(DST, ((double*)value))\
} else if (strcmp(dtype, "int8") == 0) {\
SETTER(DST, ((int8_t*)value))\
} else if (strcmp(dtype, "uint8") == 0) {\
SETTER(DST, ((uint8_t*)value))\
} else if (strcmp(dtype, "int16") == 0) {\
SETTER(DST, ((int16_t*)value))\
} else if (strcmp(dtype, "uint16") == 0) {\
SETTER(DST, ((uint16_t*)value))\
} else if (strcmp(dtype, "int32") == 0) {\
SETTER(DST, ((int32_t*)value))\
} else if (strcmp(dtype, "uint32") == 0) {\
SETTER(DST, ((uint32_t*)value))\
} else if (strcmp(dtype, "int64") == 0) {\
SETTER(DST, ((int64_t*)value))\
} else if (strcmp(dtype, "uint64") == 0) {\
SETTER(DST, ((uint64_t*)value))\
} else if (strcmp(dtype, "char") == 0) {\
SETTER(DST, ((char*)value))\
} else if (strcmp(dtype, "bool") == 0) {\
SETTER(DST, ((_Bool*)value))\
} else {\
    *err = typeErr;\
    return -1;\
}

#define SET_SCALAR(DST) TYPE_SAFE_PTR_SETTER(DST, SET_SCALAR_HELPER)
#define SET_VECTOR(DST) TYPE_SAFE_PTR_SETTER(DST, SET_VECTOR_HELPER)
#define SET_MATRIX(DST, ORD) TYPE_SAFE_PTR_SETTER(DST, SET_MATRIX_2##ORD)
#define SET_VOL(DST, ORD) TYPE_SAFE_PTR_SETTER(DST, SET_VOL_2##ORD)

#define ELIF_SCALAR(NAME) else if (strcmp(key, #NAME) == 0) {if(ndim != 0) {*err=ndimErr; return -1;} SET_SCALAR(cfg->NAME);}
#define ELIF_VEC3(NAME, TYPE) else if (strcmp(key, #NAME) == 0) {if (ndim != 1) {*err = ndimErr;return -1;} else if (dims[0] != 3) {*err = dimsErr;    return -1;} SET_VECTOR(((TYPE*)&cfg->NAME));}
#define ELIF_VEC4(NAME, TYPE) else if (strcmp(key, #NAME) == 0) {if (ndim != 1) {*err = ndimErr;return -1;} else if (dims[0] != 4) {*err = dimsErr;    return -1;} SET_VECTOR(((TYPE*)&cfg->NAME));}
#define ELIF_VEC34(NAME, TYPE) else if (strcmp(key, #NAME) == 0) {if (ndim != 1) {*err = ndimErr;return -1;} else if (dims[0] != 3 && dims[0] != 4) {*err = dimsErr; return -1;} SET_VECTOR(((TYPE*)&cfg->NAME));}


int mcx_set_field(Config * cfg, const char *key, const void *value, const char * dtype, int ndim, const unsigned*dims, const char *order, const char**err) {
    static const char *typeErr = "Incorrect dtype given.";
    static const char *ndimErr = "Incorrect number of dimensions given.";
    static const char *dimsErr = "Incorrect shape given.";
    static const char *ordErr = "Order must be either 'C' or 'F'.";
    static const char * strLenErr = "Too long of a string";
    static int seedbyte; // temp workaround

    if (ndim >= 2 && (*order != 'C' && *order != 'F')) {
        *err = ordErr;
        return -1;
    }

    if (strcmp(key, "nphoton") == 0) {
        if (cfg->replay.seed != NULL)
            return 0;
        else if (ndim != 0) {
            *err = ndimErr;
            return -1;
        }
        SET_SCALAR(cfg->nphoton)
    }
    ELIF_SCALAR(nblocksize)
    ELIF_SCALAR(nthread)
    ELIF_SCALAR(tstart)
    ELIF_SCALAR(tstep)
    ELIF_SCALAR(tend)
    ELIF_SCALAR(maxdetphoton)
    ELIF_SCALAR(sradius)
    ELIF_SCALAR(maxgate)
    ELIF_SCALAR(respin)
    ELIF_SCALAR(isreflect)
    ELIF_SCALAR(isref3)
    ELIF_SCALAR(isrefint)
    ELIF_SCALAR(isnormalized)
    ELIF_SCALAR(issave2pt)
    ELIF_SCALAR(issavedet)
    ELIF_SCALAR(isspecular)
    ELIF_SCALAR(isgpuinfo)
    ELIF_SCALAR(issrcfrom0)
    ELIF_SCALAR(autopilot)
    ELIF_SCALAR(minenergy)
    ELIF_SCALAR(unitinmm)
    ELIF_SCALAR(printnum)
    ELIF_SCALAR(voidtime)
    ELIF_SCALAR(issaveseed)
    ELIF_SCALAR(issaveref)
    ELIF_SCALAR(issaveexit)
    ELIF_SCALAR(ismomentum)
    ELIF_SCALAR(internalsrc)
    ELIF_SCALAR(isrowmajor)
    ELIF_SCALAR(replaydet)
    ELIF_SCALAR(faststep)
    ELIF_SCALAR(maxvoidstep)
    ELIF_SCALAR(maxjumpdebug)
    ELIF_SCALAR(gscatter)
    ELIF_SCALAR(outputtype)
    ELIF_SCALAR(srcnum)
    ELIF_SCALAR(savedetflag)
    ELIF_VEC3(srcpos, float)
    ELIF_VEC34(srcdir, float)
    ELIF_VEC3(steps, float)
    ELIF_VEC3(dim, unsigned)
    ELIF_VEC3(crop0, unsigned)
    ELIF_VEC3(crop1, unsigned)
    ELIF_VEC4(srcparam1, float)
    ELIF_VEC4(srcparam2, float)
    else if (strcmp(key, "vol") == 0) {
        if (ndim != 3) {
            *err = ndimErr;
            return -1;
        }
        cfg->dim.x = dims[0];
        cfg->dim.y = dims[1];
        cfg->dim.z = dims[2];
        int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;
        if(cfg->vol) free(cfg->vol);
        cfg->vol = (unsigned int*)malloc(dimxyz * sizeof(unsigned int));
        SET_VOL(cfg->vol, F)
        cfg->mediabyte = sizeof(unsigned int);
        cfg->isrowmajor = 0;
    } else if(strcmp(key, "prop") == 0){
        if(ndim != 2){
            *err = ndimErr;
            return -1;
        } else if(dims[0] <= 0 || dims[1] != 4){
            *err = dimsErr;
            return -1;
        }
        cfg->medianum = dims[0];
        if(cfg->prop) free(cfg->prop);
        cfg->prop = malloc(cfg->medianum*sizeof(Medium));
        float * dst = (float*)cfg->prop;
        SET_MATRIX(dst, C);
    } else if(strcmp(key, "detpos")==0){
        if(ndim != 2){
            *err = ndimErr;
            return -1;
        } else if(dims[0] <= 0 || dims[1] != 4){
            *err = dimsErr;
            return -1;
        }
        cfg->detnum=dims[0];
        if(cfg->detpos) free(cfg->detpos);
        cfg->detpos = malloc(cfg->detnum*sizeof(float4));
        float *dst = (float*)cfg->detpos;
        SET_MATRIX(dst, C);
    } else if(strcmp(key,"session")==0) {
        if(strcmp(dtype, "string") != 0){
            *err = typeErr;
            return -1;
        } else if(dims[0] >= MAX_SESSION_LENGTH || dims[0] == 0){
            *err = strLenErr;
            return -1;
        }
        strncpy(cfg->session, value, MAX_SESSION_LENGTH);
        cfg->session[MAX_SESSION_LENGTH-1] = '\0';
    }else if(strcmp(key,"srctype")==0){
        if(strcmp(dtype, "string") != 0){
            *err = typeErr;
            return -1;
        } else if(dims[0] == 0){
            *err = strLenErr;
            return -1;
        }
        const char *srctypeid[]={"pencil","isotropic","cone","gaussian","planar","pattern","fourier","arcsine","disk","fourierx","fourierx2d","zgaussian","line","slit","pencilarray",""};
        int srctype = mcx_keylookup((char*)value,srctypeid);
        static char * srcTypErr = "the specified source type is not supported";
        if(srctype == -1){
            *err = srcTypErr;
            return -1;
        }
        cfg->srctype = srctype;
    }else if(strcmp(key,"debuglevel")==0){
        if(strcmp(dtype, "string") != 0){
            *err = typeErr;
            return -1;
        } else if(dims[0] == 0){
            *err = strLenErr;
            return -1;
        }
        const char debugflag[]={'R','M','P','\0'};
        cfg->debuglevel = mcx_parsedebugopt((char*)value,debugflag);
        static char * dbgLvlErr = "the specified debuglevel is not supported";
        if(cfg->debuglevel==0){
            *err = dbgLvlErr;
            return -1;
        }
    }
    else if(strcmp(key,"srcpattern")==0){
        if(ndim != 2){
            *err = ndimErr;
            return -1;
        }
        if(cfg->srcpattern) free(cfg->srcpattern);
        cfg->srcpattern = malloc(dims[0]*dims[1]*sizeof(float));
        SET_MATRIX(cfg->srcpattern, F)
    }else if(strcmp(key,"shapes")==0){
        if(strcmp(dtype, "string") != 0){
            *err = typeErr;
            return -1;
        } else if(dims[0] == 0){
            *err = strLenErr;
            return -1;
        }
        Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},0};
        if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
        if(mcx_parse_shapestring(&grid,(char*)value)){
            *err = mcx_last_shapeerror();
            return -1;
        }
    } else if(strcmp(key, "detphotons")==0){
        if (cfg->seed != SEED_FROM_FILE) {
            static char * replayErr = "Need cfg.seed for replay, before being given 'detphotons'";
            *err = replayErr;
            return -1;
        } else if (seedbyte == 0) {
            static char * seedErr = "the seed input is empty";
            *err = seedErr;
            return -1;
        } else if (strcmp(dtype, "float") != 0 && strcmp(dtype, "float32") != 0) {
            *err = typeErr;
            return -1;
        } else if (ndim != 2) {
            *err = ndimErr;
            return -1;
        } else if (cfg->nphoton != dims[1]) {
            *err = dimsErr;
            return -1;
        } else if (*order != 'F') {  // TODO: Really unsure of this
            *err = ordErr;
            return -1;
        }

        const float* detps = value;
        if(cfg->replay.weight) free(cfg->replay.weight);
        if(cfg->replay.tof) free(cfg->replay.tof);
        cfg->replay.weight = (float *)malloc(cfg->nphoton * sizeof(float));
        cfg->replay.tof = (float *)calloc(cfg->nphoton, sizeof(float));
        cfg->nphoton = 0;
        for (unsigned i = 0; i < dims[1]; i++) {
            if (cfg->replaydet == 0 || cfg->replaydet == (int)(detps[i * dims[0]])) {
                if (i != cfg->nphoton)
                    memcpy((char *)(cfg->replay.seed) + cfg->nphoton * seedbyte,
                        (char *)(cfg->replay.seed) + i * seedbyte, seedbyte);
                cfg->replay.weight[cfg->nphoton] = 1.f;
                cfg->replay.tof[cfg->nphoton] = 0.f;
                for (unsigned j = 2; j < cfg->medianum + 1; j++) {
                    cfg->replay.weight[cfg->nphoton] *= expf(-cfg->prop[j - 1].mua * detps[i * dims[0] + j] * cfg->unitinmm);
                    cfg->replay.tof[cfg->nphoton] += detps[i * dims[0] + j] * cfg->unitinmm * R_C0 * cfg->prop[j - 1].n;
                }
                if (cfg->replay.tof[cfg->nphoton] < cfg->tstart ||
                    cfg->replay.tof[cfg->nphoton] > cfg->tend) /*need to consider -g*/
                    continue;
                cfg->nphoton++;
            }
        }
    } else if(strcmp(key,"seed")==0){
        if(ndim == 0){
            SET_SCALAR(cfg->seed);
        } else if(ndim == 2){
            if(dims[0]!=sizeof(float)*RAND_WORD_LEN){
                static char * seedErr = "the row number of cfg.seed does not match RNG seed byte-length";
                *err = seedErr;
                return -1;
            }
            seedbyte = dims[0];
            if(cfg->replay.seed) free(cfg->replay.seed);
            cfg->replay.seed = malloc((dims[0]*dims[1]));
            int * dst = cfg->replay.seed;
            SET_MATRIX(dst, F)
            cfg->seed=SEED_FROM_FILE;
            cfg->nphoton=dims[1];
        } else {
            *err = ndimErr;
            return -1;
        }
    } else if(strcmp(key,"gpuid")==0){
        if(strcmp(dtype, "string") == 0){
            if(dims[0] > MAX_DEVICE || dims[0] == 0) {
                *err = strLenErr;
                return -1;
            }
            memcpy(cfg->deviceid, value, MAX_DEVICE);
        } else if(ndim == 0){
            cfg->gpuid = *((int*)value);
            memset(cfg->deviceid,0,MAX_DEVICE);
            static char * gpuidErr = "GPU id can not be more than 256";
            if(cfg->gpuid<MAX_DEVICE){
                memset(cfg->deviceid,'0',cfg->gpuid-1);
                cfg->deviceid[cfg->gpuid-1]='1';
            }else{
                *err = gpuidErr;
                return -1;
            }
        } else {
            *err = typeErr;
            return -1;
        }
        for(int i=0;i<MAX_DEVICE;i++)
            if(cfg->deviceid[i]=='0')
                cfg->deviceid[i]='\0';
    } else if(strcmp(key,"workload")==0){
        if(ndim != 1){
            *err = ndimErr;
            return -1;
        } else if(dims[0] == 0 || dims[0] >= MAX_DEVICE){
            *err = dimsErr;
            return -1;
        }
        SET_VECTOR(cfg->workload)
    } else if(strcmp(key,"flog")==0){
        if(strcmp(dtype, "file")==0){
            cfg->flog = (FILE*)value;
        } else {
            *err = typeErr;
            return -1;
        }
    } else {
        static char * unkErr = "Unknown Field Given";
        *err = unkErr;
        return -1;
    }
    return 0;
}

void initialize_output(Config *cfg, int nout) {
    if (nout >= 1) {
        if(cfg->exportfield) free(cfg->exportfield);
        int fieldlen = cfg->dim.x*cfg->dim.y*cfg->dim.z*(int)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);
        cfg->exportfield = (float*)calloc(fieldlen, sizeof(float));
    }
    if (nout >= 2) {
        if(cfg->exportdetected) free(cfg->exportdetected);
        cfg->exportdetected = (float*)malloc((cfg->medianum + 1 + (cfg->issaveexit > 0) * 6 + (cfg->ismomentum > 0)*(cfg->medianum - 1))*cfg->maxdetphoton*sizeof(float));
    }
    if (nout >= 4) {
        if(cfg->seeddata) free(cfg->seeddata);
        cfg->seeddata = malloc(cfg->maxdetphoton*sizeof(float)*RAND_WORD_LEN);
    }
    if (nout >= 5) {
        if(cfg->exportdebugdata) free(cfg->exportdebugdata);
        cfg->exportdebugdata = (float*)malloc(cfg->maxjumpdebug*sizeof(float)*MCX_DEBUG_REC_LEN);
    }
}


#define GET_SCALAR(NAME, TYPE) else if (strcmp(key, #NAME) == 0) {*dtype = TYPE##Type; *ndim=0; return &cfg->NAME; }
#define GET_VEC3(NAME, TYPE) else if (strcmp(key, #NAME) == 0) {*dtype = TYPE##Type; *ndim=1; dims[0]=3; return &cfg->NAME; }
#define GET_VEC4(NAME, TYPE) else if (strcmp(key, #NAME) == 0) {*dtype = TYPE##Type; *ndim=1; dims[0]=4; return &cfg->NAME; }
#define GET_VECTOR(NAME, TYPE, D0) else if (strcmp(key, #NAME) == 0) {*dtype = TYPE##Type; *ndim=1; dims[0]=D0; return cfg->NAME; }
#define GET_MATRIX(NAME, TYPE, D0, D1) else if (strcmp(key, #NAME) == 0) {*dtype = TYPE##Type; *ndim=2; dims[0]=D0; dims[1]=D1; return cfg->NAME; }
#define GET_CUBE(NAME, TYPE, D0, D1, D2) else if (strcmp(key, #NAME) == 0) {*dtype = TYPE##Type; *ndim=3; dims[0]=D0; dims[1]=D1; dims[2]=D2; return cfg->NAME; }

void* mcx_get_field(Config *cfg, const char *key, char** dtype, int* ndim, unsigned *dims, const char**err) {
    static char * intType = "int";
    static char * uintType = "uint";
    static char * floatType = "float";
    static char * doubleType = "double";
    static char * uint8Type = "uint8";
    static char * charType = "char";

    unsigned int partialdata=(cfg->medianum-1)*(SAVE_NSCAT(cfg->savedetflag)+SAVE_PPATH(cfg->savedetflag)+SAVE_MOM(cfg->savedetflag)); // buf len for media-specific data, copy from gpu to host
    unsigned int hostdetreclen=partialdata+SAVE_DETID(cfg->savedetflag)+3*(SAVE_PEXIT(cfg->savedetflag)+SAVE_VEXIT(cfg->savedetflag))+SAVE_W0(cfg->savedetflag); // host-side det photon data buffer length

    if (strcmp(key, "exportfield") == 0) {
        *dtype = floatType;
        *ndim = 4;
        dims[0] = cfg->dim.x;
        dims[1] = cfg->dim.y;
        dims[2] = cfg->dim.z;
        dims[3] = (cfg->tend - cfg->tstart) / cfg->tstep + 0.5;
        return cfg->exportfield;
    }
    GET_SCALAR(nphoton, int)
    GET_SCALAR(nblocksize, uint)
    GET_SCALAR(nthread, uint)
    GET_SCALAR(seed, int)
    GET_VEC4(srcpos, float)
    GET_VEC4(srcdir, float)
    GET_SCALAR(tstart, float)
    GET_SCALAR(tstep, float)
    GET_SCALAR(tend, float)
    GET_VEC3(steps, float)
    GET_VEC3(dim, uint)
    GET_VEC3(crop0, uint)
    GET_VEC3(crop1, uint)
    GET_SCALAR(medianum, uint)
    GET_SCALAR(detnum, uint)
    GET_SCALAR(maxdetphoton, uint)
    GET_SCALAR(detradius, float)
    GET_SCALAR(sradius, float)
    GET_MATRIX(prop, float, cfg->medianum, 4)
    GET_MATRIX(detpos, float, cfg->detnum, 4)
    GET_SCALAR(maxgate, uint)
    GET_SCALAR(respin, uint)
    GET_SCALAR(printnum, uint)
    GET_SCALAR(gpuid, int)
    GET_CUBE(vol, uint, cfg->dim.x, cfg->dim.y, cfg->dim.z)
    GET_VECTOR(session, char, MAX_SESSION_LENGTH)
    GET_SCALAR(isrowmajor, char)
    GET_SCALAR(isreflect, char)
    GET_SCALAR(isref3, char)
    GET_SCALAR(isrefint, char)
    GET_SCALAR(isnormalized, char)
    GET_SCALAR(issavedet, char)
    GET_SCALAR(issave2pt, char)
    GET_SCALAR(isspecular, char)
    GET_SCALAR(isgpuinfo, char)
    GET_SCALAR(issrcfrom0, char)
    GET_SCALAR(isdumpmask, char)
    GET_SCALAR(autopilot, char)
    GET_SCALAR(issaveseed, char)
    GET_SCALAR(issaveexit, char)
    GET_SCALAR(ismomentum, char)
    GET_SCALAR(savedetflag, uint)
    GET_SCALAR(internalsrc, char)
    GET_SCALAR(issaveref, char)
    GET_SCALAR(srctype, char)
    GET_SCALAR(outputtype, char)
    GET_SCALAR(outputformat, char)
    GET_SCALAR(faststep, char)
    GET_SCALAR(minenergy, float)
    GET_SCALAR(unitinmm, float)
    //GET_SCALAR(flog, FILE)
    //GET_SCALAR(his, History)
    GET_MATRIX(exportdetected, float, hostdetreclen, cfg->detectedcount)
    GET_SCALAR(detectedcount, uint)
    GET_VECTOR(rootpath, char, MAX_PATH_LENGTH)
    //GET_VECTOR(shapedata, char, 0)
    GET_SCALAR(maxvoidstep, int)
    GET_SCALAR(voidtime, int)
    GET_VEC4(srcparam1, float)
    GET_VEC4(srcparam2, float)
    //GET_MATRIX(srcpattern, float, 0, 0)
    //GET_SCALAR(replay, Replay)
    GET_MATRIX(seeddata, uint8, (cfg->issaveseed>0)*RAND_WORD_LEN*sizeof(float), cfg->detectedcount)
    GET_SCALAR(replaydet, int)
    GET_VECTOR(seedfile, char, MAX_PATH_LENGTH)
    GET_SCALAR(debuglevel, uint)
    GET_VECTOR(deviceid, char, MAX_DEVICE)
    GET_VECTOR(workload, float, MAX_DEVICE)
    GET_SCALAR(parentid, int)
    GET_SCALAR(runtime, uint)
    GET_SCALAR(energytot, double)
    GET_SCALAR(energyabs, double)
    GET_SCALAR(energyesc, double)
    GET_SCALAR(normalizer, float)
    GET_SCALAR(maxjumpdebug, uint)
    GET_SCALAR(debugdatalen, uint)
    GET_SCALAR(gscatter, uint)
    GET_MATRIX(exportdebugdata, float, MCX_DEBUG_REC_LEN, cfg->debugdatalen)
    GET_SCALAR(mediabyte, uint)
    static char * invalidErr = "Not yet implemented.";
    *err = invalidErr;
    return NULL;
}


int mcx_wrapped_run_simulation(Config *cfg, int nout, char**err) {
    char temp_gpu_workaround[MAX_DEVICE];
    memcpy(temp_gpu_workaround, cfg->deviceid, MAX_DEVICE);

    GPUInfo *gpuinfo;
    int errorflag = 0;
    int activedev = mcx_list_gpu(cfg, &gpuinfo);
    if(activedev == 0){
        static char * noGpuErr = "No active GPU device found";
        *err = noGpuErr;
        mcx_cleargpuinfo(&gpuinfo);
        memcpy(cfg->deviceid, temp_gpu_workaround, MAX_DEVICE);
        return -1;
    }
    cfg->issave2pt = (nout >= 1);  /** save fluence rate to the 1st output if present */
    cfg->issavedet = (nout >= 2);  /** save detected photon data to the 2nd output if present */
    cfg->issaveseed = (nout >= 4); /** save detected photon seeds to the 4th output if present */
#if defined(USE_MT_RAND)
    cfg->issaveseed = 0;
#endif
    if(mcx_validateconfig(cfg, err, 0, NULL, NULL)) {
        mcx_cleargpuinfo(&gpuinfo);
        memcpy(cfg->deviceid, temp_gpu_workaround, MAX_DEVICE);
        return -1;
    }
    initialize_output(cfg, nout);
#ifdef _OPENMP
    omp_set_num_threads(activedev);
#pragma omp parallel shared(errorflag)
    {
#endif
        jmp_buf errHandler;
        if (setjmp(errHandler) == 0) {
            mcx_set_error_handler(&errHandler);
            mcx_run_simulation(cfg, gpuinfo);
        } else {
            errorflag++;
        }
        mcx_set_error_handler(NULL);
#ifdef _OPENMP
    }
#endif
    mcx_cleargpuinfo(&gpuinfo);
    memcpy(cfg->deviceid, temp_gpu_workaround, MAX_DEVICE);
    return errorflag;
}


static float _my_abs_temp(float x) {
    return x >= 0 ? x : -x;
}

/**
 * @brief Validate all input fields, and warn incompatible inputs
 *
 * Perform self-checking and raise exceptions or warnings when input error is detected
 *
 * @param[in,out] cfg: the simulation configuration structure
 * @param[in,out] errmsg: the simulation configuration structure
 * @param[in] seedbyte: seed byte for replay functionality
 * @param[in,out] detps: buffer to receive data from cfg.detphotons field
 * @param[in] dimdetps: dimensions of the cfg.detphotons array
 * @return if valid, return 0, otherwise -1.
 */

int mcx_validateconfig(Config *cfg, char **errmsg, int seedbyte, float *detps, int *dimdetps){
    int i, gates, wassrcfrom0 = cfg->issrcfrom0;

    if(!cfg->issrcfrom0){
        cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index, grid center*/
        cfg->issrcfrom0 = 1;
    }
    /** One must define the domain and properties */
    if (cfg->vol == NULL || cfg->medianum == 0) {
        static char * domainErr = "You must define 'vol' and 'prop' field.";
        *errmsg = domainErr;
        return -1;
    }
    else if(cfg->tstart>cfg->tend || cfg->tstep==0.f){
        static char * timeErr = "incorrect time gate settings";
        *errmsg = timeErr;
        return -1;
    }
    else if(_my_abs_temp(cfg->srcdir.x*cfg->srcdir.x+cfg->srcdir.y*cfg->srcdir.y+cfg->srcdir.z*cfg->srcdir.z - 1.f)>1e-5){
        static char * unitaryErr = "field 'srcdir' must be a unitary vector";
        *errmsg = unitaryErr;
        return -1;
    }
    else if(cfg->steps.x==0.f || cfg->steps.y==0.f || cfg->steps.z==0.f){
        static char * stepsErr = "field 'steps' can not have zero elements";
        *errmsg = stepsErr;
        return -1;
    }
    else if(cfg->tend<=cfg->tstart){
        static char * tendErr = "field 'tend' must be greater than field 'tstart'";
        *errmsg = tendErr;
        return -1;
    }
    gates=(int)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
    if(cfg->maxgate>gates)
        cfg->maxgate=gates;
    if(cfg->sradius>0.f){
        cfg->crop0.x=MAX((int)(cfg->srcpos.x-cfg->sradius),0);
        cfg->crop0.y=MAX((int)(cfg->srcpos.y-cfg->sradius),0);
        cfg->crop0.z=MAX((int)(cfg->srcpos.z-cfg->sradius),0);
        cfg->crop1.x=MIN((int)(cfg->srcpos.x+cfg->sradius),cfg->dim.x-1);
        cfg->crop1.y=MIN((int)(cfg->srcpos.y+cfg->sradius),cfg->dim.y-1);
        cfg->crop1.z=MIN((int)(cfg->srcpos.z+cfg->sradius),cfg->dim.z-1);
    }else if(cfg->sradius==0.f){
        memset(&(cfg->crop0),0,sizeof(uint3));
        memset(&(cfg->crop1),0,sizeof(uint3));
    }else{
        /*
            if -R is followed by a negative radius, mcx uses crop0/crop1 to set the cachebox
        */
        if(!wassrcfrom0){
            cfg->crop0.x--;cfg->crop0.y--;cfg->crop0.z--;  /*convert to C index*/
            cfg->crop1.x--;cfg->crop1.y--;cfg->crop1.z--;
        }
    }
    if(cfg->medianum==0){
        static char * propErr = "you must define the 'prop' field in the input structure";
        *errmsg = propErr;
        return -1;
    }
    if(cfg->dim.x==0||cfg->dim.y==0||cfg->dim.z==0){
        static char * volErr = "the 'vol' field in the input structure can not be empty";
        *errmsg = volErr;
        return -1;
    }
    if(cfg->srctype==MCX_SRC_PATTERN && cfg->srcpattern==NULL){
        static char * srcErr = "the 'srcpattern' field can not be empty when your 'srctype' is 'pattern";
        *errmsg = srcErr;
        return -1;
    }
    if(cfg->steps.x!=1.f && cfg->unitinmm==1.f)
        cfg->unitinmm=cfg->steps.x;

    if(cfg->medianum){
        for(int i=0;i<cfg->medianum;i++)
            if(cfg->prop[i].mus==0.f)
                cfg->prop[i].mus=EPS;
    }
    if(cfg->unitinmm!=1.f){
        cfg->steps.x=cfg->unitinmm; cfg->steps.y=cfg->unitinmm; cfg->steps.z=cfg->unitinmm;
        for(i=1;i<cfg->medianum;i++){
            cfg->prop[i].mus*=cfg->unitinmm;
            cfg->prop[i].mua*=cfg->unitinmm;
        }
    }
    if(cfg->issavedet && cfg->detnum==0)
        cfg->issavedet=0;
    if(cfg->issavedet==0){
        cfg->issaveexit=0;
        cfg->ismomentum=0;
    }
    if(cfg->issavedet) cfg->savedetflag=SET_SAVE_PPATH(cfg->savedetflag);
    if(cfg->ismomentum) cfg->savedetflag=SET_SAVE_MOM(cfg->savedetflag);

    if(cfg->issaveexit){
        cfg->savedetflag=SET_SAVE_PEXIT(cfg->savedetflag);
	    cfg->savedetflag=SET_SAVE_VEXIT(cfg->savedetflag);
     }
    if(cfg->issavedet && cfg->savedetflag==0) cfg->savedetflag=0x5;

    if(cfg->mediabyte>=100){
	     cfg->savedetflag=UNSET_SAVE_NSCAT(cfg->savedetflag);
	     cfg->savedetflag=UNSET_SAVE_PPATH(cfg->savedetflag);
	     cfg->savedetflag=UNSET_SAVE_MOM(cfg->savedetflag);
    }

    if(cfg->seed<0 && cfg->seed!=SEED_FROM_FILE) cfg->seed=time(NULL);
    if((cfg->outputtype==otJacobian || cfg->outputtype==otWP) && cfg->seed!=SEED_FROM_FILE){
        static char * replyErr = "Jacobian output is only valid in the reply mode. Please define cfg.seed";
        *errmsg = replyErr;
        return -1;
    }
    for(i=0;i<cfg->detnum;i++){
        if(!wassrcfrom0){
            cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
        }
    }
    if(cfg->isrowmajor){
        /*from here on, the array is always col-major*/
        mcx_convertrow2col(&(cfg->vol), &(cfg->dim));
        cfg->isrowmajor=0;
    }
    if(cfg->issavedet)
        mcx_maskdet(cfg);
    if(cfg->seed==SEED_FROM_FILE){
        if(cfg->respin>1){
            cfg->respin=1;
            fprintf(stderr,"Warning: respin is disabled in the replay mode\n");
        }
    }
    cfg->his.maxmedia=cfg->medianum-1; /*skip medium 0*/
    cfg->his.detnum=cfg->detnum;
    cfg->his.colcount=cfg->medianum+1+(cfg->issaveexit > 0)*6+(cfg->ismomentum > 0)*(cfg->medianum - 1); /*column count=maxmedia+2*/

    /* mcx_replay_prep
     * Pre-computes the detected photon weight and time-of-fly from partial path input for replay
     * When detected photons are replayed, this function recalculates the detected photon
     * weight and their time-of-fly for the replay calculations.
     */
    if(cfg->seed==SEED_FROM_FILE && (detps == NULL || cfg->replay.weight == NULL || cfg->replay.tof == NULL)) {
        static char * replayErr = "you give cfg.seed for replay, but did not specify cfg.detphotons.\nPlease define it as the detphoton output from the baseline simulation";
        *errmsg = replayErr;
        return -1;
    }
    else if(detps!=NULL && cfg->seed==SEED_FROM_FILE) {
        if (cfg->nphoton != dimdetps[1]){
            static char * colErr = "the column numbers of detphotons and seed do not match";
            *errmsg = colErr;
            return -1;
        }
        else if (seedbyte == 0){
            static char * seedErr = "the seed input is empty";
            *errmsg = seedErr;
            return -1;
        }

        if(cfg->replay.weight) free(cfg->replay.weight);
        if(cfg->replay.tof) free(cfg->replay.tof);
        cfg->replay.weight = (float *) malloc(cfg->nphoton * sizeof(float));
        cfg->replay.tof = (float *) calloc(cfg->nphoton, sizeof(float));

        cfg->nphoton = 0;
        for (int i = 0; i < dimdetps[1]; i++) {
            if (cfg->replaydet == 0 || cfg->replaydet == (int) (detps[i * dimdetps[0]])) {
                if (i != cfg->nphoton)
                    memcpy((char *) (cfg->replay.seed) + cfg->nphoton * seedbyte,
                           (char *) (cfg->replay.seed) + i * seedbyte, seedbyte);
                cfg->replay.weight[cfg->nphoton] = 1.f;
                cfg->replay.tof[cfg->nphoton] = 0.f;
                for (int j = 2; j < cfg->medianum + 1; j++) {
                    cfg->replay.weight[cfg->nphoton] *= expf(
                            -cfg->prop[j - 1].mua * detps[i * dimdetps[0] + j] * cfg->unitinmm);
                    cfg->replay.tof[cfg->nphoton] +=
                            detps[i * dimdetps[0] + j] * cfg->unitinmm * R_C0 * cfg->prop[j - 1].n;
                }
                if (cfg->replay.tof[cfg->nphoton] < cfg->tstart ||
                    cfg->replay.tof[cfg->nphoton] > cfg->tend) /*need to consider -g*/
                    continue;
                cfg->nphoton++;
            }
        }
    }
    return 0;
}
