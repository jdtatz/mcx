#include <string.h>
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
    return cfg;
}

void mcx_destroy_config(Config * cfg){
    mcx_clearcfg(cfg);
    free(cfg);
}

#define SET_SCALAR_FIELD(NAME, TYPEVAL) if(ndim != 0) {*err=ndimErr; return -1;} cfg->NAME = TYPEVAL;

#define SET_VEC3_FIELD(NAME, TYPE) if(strcmp(dtype, #TYPE) != 0) {*err=typeErr; return -1;} \
if(ndim != 1){*err=ndimErr; return -1;} if(dims[0] != 3){*err=dimsErr; return -1;} \
 cfg->NAME.x = ((TYPE*)value)[0]; cfg->NAME.y = ((TYPE*)value)[1]; cfg->NAME.z = ((TYPE*)value)[2];

#define SET_VEC4_FIELD(NAME, TYPE) if(strcmp(dtype, #TYPE) != 0) {*err=typeErr; return -1;} \
if(ndim != 1){*err=ndimErr; return -1;} if(dims[0] != 4){*err=dimsErr; return -1;} \
 cfg->NAME.x = ((TYPE*)value)[0]; cfg->NAME.y = ((TYPE*)value)[1]; cfg->NAME.z = ((TYPE*)value)[2]; \
 cfg->NAME.w = ((TYPE*)value)[3];

#define SET_VEC34_FIELD(NAME, TYPE) if(strcmp(dtype, #TYPE) != 0) {*err=typeErr; return -1;} \
if(ndim != 1){*err=ndimErr; return -1;} if(dims[0] != 3 && dims[0] != 4){*err=dimsErr; return -1;} \
 cfg->NAME.x = ((TYPE*)value)[0]; cfg->NAME.y = ((TYPE*)value)[1]; cfg->NAME.z = ((TYPE*)value)[2]; \
 if(dims[0] == 4) {cfg->NAME.w = ((TYPE*)value)[3];}

#define IF_SCALAR_FIELD(NAME, TYPEVAL) if (strcmp(key, #NAME) == 0) {SET_SCALAR_FIELD(NAME, TYPEVAL);}
#define ELIF_SCALAR_FIELD(NAME, TYPEVAL) else IF_SCALAR_FIELD(NAME, TYPEVAL)
#define ELIF_VEC3_FIELD(NAME, TYPE) else if (strcmp(key, #NAME) == 0) {SET_VEC3_FIELD(NAME, TYPE);}
#define ELIF_VEC4_FIELD(NAME, TYPE) else if (strcmp(key, #NAME) == 0) {SET_VEC4_FIELD(NAME, TYPE);}
#define ELIF_VEC34_FIELD(NAME, TYPE) else if (strcmp(key, #NAME) == 0) {SET_VEC34_FIELD(NAME, TYPE);}


int mcx_set_field(Config * cfg, const char *key, const void *value, const char * dtype, int ndim, const unsigned*dims, const char**err) {
    static const char *typeErr = "Incorrect dtype given.";
    static const char *ndimErr = "Incorrect number of dimensions given.";
    static const char *dimsErr = "Incorrect shape given.";
    static const char * strLenErr = "Too long of a string";

	char charV;
	int intV;
	unsigned int uintV;
	float floatV;
	if (ndim == 0) {
		if (strcmp(dtype, "char") == 0) {
			charV = *((char*)value);
			intV = (int)charV;
			uintV = (unsigned int)charV;
			floatV = (float)charV;
		} else if (strcmp(dtype, "int") == 0) {
			intV = *((int*)value);
			charV = (char)intV;
			uintV = (unsigned int)intV;
			floatV = (float)intV;
		} else if (strcmp(dtype, "uint") == 0) {
			uintV = *((unsigned int*)value);
			charV = (char)uintV;
			intV = (int)uintV;
			floatV = (float)uintV;
		} else if (strcmp(dtype, "float") == 0) {
			floatV = *((float*)value);
			charV = (char)floatV;
			intV = (int)floatV;
			uintV = (unsigned int)floatV;
		} else {
			*err = typeErr;
			return -1;
		}
	}

    char *jsonshapes=NULL;
    /* // Unsure Why This Exists
    if(strcmp(name,"nphoton")==0 && cfg->replay.seed!=NULL)
        return 0;
    */

    IF_SCALAR_FIELD(nphoton, intV)
    ELIF_SCALAR_FIELD(nblocksize, uintV)
    ELIF_SCALAR_FIELD(nthread, uintV)
    ELIF_SCALAR_FIELD(tstart, floatV)
    ELIF_SCALAR_FIELD(tstep, floatV)
    ELIF_SCALAR_FIELD(tend, floatV)
    ELIF_SCALAR_FIELD(maxdetphoton, uintV)
    ELIF_SCALAR_FIELD(sradius, floatV)
    ELIF_SCALAR_FIELD(maxgate, uintV)
    ELIF_SCALAR_FIELD(respin, uintV)
    ELIF_SCALAR_FIELD(isreflect, charV)
    ELIF_SCALAR_FIELD(isref3, charV)
    ELIF_SCALAR_FIELD(isrefint, charV)
    ELIF_SCALAR_FIELD(isnormalized, charV)
    ELIF_SCALAR_FIELD(isgpuinfo, charV)
    ELIF_SCALAR_FIELD(issrcfrom0, charV)
    ELIF_SCALAR_FIELD(autopilot, charV)
    ELIF_SCALAR_FIELD(minenergy, floatV)
    ELIF_SCALAR_FIELD(unitinmm, floatV)
    ELIF_SCALAR_FIELD(reseedlimit, uintV)
    ELIF_SCALAR_FIELD(printnum, uintV)
    ELIF_SCALAR_FIELD(voidtime, intV)
    ELIF_SCALAR_FIELD(issaveseed, charV)
    ELIF_SCALAR_FIELD(issaveref, charV)
    ELIF_SCALAR_FIELD(issaveexit, charV)
    ELIF_SCALAR_FIELD(isrowmajor, charV)
    ELIF_SCALAR_FIELD(replaydet, intV)
    ELIF_SCALAR_FIELD(faststep, charV)
    ELIF_SCALAR_FIELD(maxvoidstep, intV)
    ELIF_SCALAR_FIELD(maxjumpdebug, uintV)
    ELIF_SCALAR_FIELD(gscatter, uintV)
    ELIF_VEC3_FIELD(srcpos, float)
    ELIF_VEC34_FIELD(srcdir, float)
    ELIF_VEC3_FIELD(steps, float)
    ELIF_VEC3_FIELD(crop0, unsigned)
    ELIF_VEC3_FIELD(crop1, unsigned)
    ELIF_VEC4_FIELD(srcparam1, float)
    ELIF_VEC4_FIELD(srcparam2, float)
    else if (strcmp(key, "vol") == 0) {
		if (strcmp(dtype, "uint") != 0) {
			*err = typeErr;
			return -1;
		} else if (ndim != 3) {
			*err = ndimErr;
			return -1;
		}
        cfg->dim.x = dims[0];
        cfg->dim.y = dims[1];
        cfg->dim.z = dims[2];
        int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;
        if(cfg->vol) free(cfg->vol);
        cfg->vol = (unsigned int*)malloc(dimxyz * sizeof(unsigned int));
        memcpy(cfg->vol, value, dimxyz * sizeof(unsigned int));
        cfg->mediabyte = sizeof(unsigned int);
    } else if(strcmp(key, "prop") == 0){
        if(strcmp(dtype, "float") != 0){
            *err = typeErr;
            return -1;
        } else if(ndim != 2){
            *err = ndimErr;
            return -1;
        } else if(dims[0] <= 0 || dims[1] != 4){
            *err = dimsErr;
            return -1;
        }
        cfg->medianum = dims[0];
        if(cfg->prop) free(cfg->prop);
        cfg->prop = (Medium*)malloc(cfg->medianum*sizeof(Medium));
        for(unsigned i=0;i<cfg->medianum;i++){
            cfg->prop[i].mua = ((float*)value)[i*4  ];
            cfg->prop[i].mus = ((float*)value)[i*4+1];
            cfg->prop[i].g   = ((float*)value)[i*4+2];
            cfg->prop[i].n   = ((float*)value)[i*4+3];
        }
    } else if(strcmp(key, "detpos")==0){
        if(strcmp(dtype, "float") != 0){
            *err = typeErr;
            return -1;
        } else if(ndim != 2){
            *err = ndimErr;
            return -1;
        } else if(dims[0] <= 0 || dims[1] != 4){
            *err = dimsErr;
            return -1;
        }
        cfg->detnum=dims[0];
        if(cfg->detpos) free(cfg->detpos);
        cfg->detpos = (float4*)malloc(cfg->detnum*sizeof(float4));
        for(unsigned i=0; i < cfg->detnum; i++){
            cfg->detpos[i].x = ((float *)value)[i*4];
            cfg->detpos[i].y = ((float *)value)[i*4+1];
            cfg->detpos[i].z = ((float *)value)[i*4+2];
            cfg->detpos[i].w = ((float *)value)[i*4+3];
        }
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
    } else if(strcmp(key,"outputtype")==0) {
        if(strcmp(dtype, "string") != 0){
            *err = typeErr;
            return -1;
        } else if(dims[0] == 0){
            *err = strLenErr;
            return -1;
        }
        const char *outputtype[]={"flux","fluence","energy","jacobian","nscat","wl","wp",""};
        int outtyp = mcx_keylookup((char*)value,outputtype);
        if(outtyp == 5 || outtyp == 6) // map wl to jacobian, wp to nscat
            cfg->outputtype = outtyp - 2;
        else
            cfg->outputtype = outtyp;
        static char * outTypErr = "the specified output type is not supported";
        if(outtyp == -1){
            *err = outTypErr;
            return -1;
        }
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
        if(strcmp(dtype, "float") != 0){
            *err = typeErr;
            return -1;
        } else if(ndim != 2){
            *err = ndimErr;
            return -1;
        }
        if(cfg->srcpattern) free(cfg->srcpattern);
        cfg->srcpattern = malloc(dims[0]*dims[1]*sizeof(float));
        memcpy(cfg->srcpattern, (float*)value, dims[0]*dims[1]*sizeof(float));
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
    }/*else if(strcmp(key,"detphotons")==0){ // TODO: Unknown option, is UnDocumented
        arraydim=mxGetDimensions(item);
        dimdetps[0]=arraydim[0];
        dimdetps[1]=arraydim[1];
        detps=malloc(arraydim[0]*arraydim[1]*sizeof(float));
        memcpy(detps,mxGetData(item),arraydim[0]*arraydim[1]*sizeof(float));
        printf("mcx.detphotons=[%d %d];\n",arraydim[0],arraydim[1]);
    }*/
    else if(strcmp(key,"seed")==0){
        if(strcmp(dtype, "int") != 0){
            *err = typeErr;
            return -1;
        }
        if(ndim == 0){
            cfg->seed = *((int*)value);
        } else if(ndim == 2){
            static char * seedErr = "the row number of cfg.seed does not match RNG seed byte-length";
            if(dims[0]!=sizeof(float)*RAND_WORD_LEN){
                *err = seedErr;
                return -1;
            }
            // seedbyte = dims[0];
            cfg->replay.seed = malloc((dims[0]*dims[1]));
            memcpy(cfg->replay.seed,value,dims[0]*dims[1]);
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
        } else if(strcmp(dtype, "int") == 0){
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
    }
    else if(strcmp(key,"workload")==0){
        if(strcmp(dtype, "float") != 0){
            *err = typeErr;
            return -1;
        } else if(ndim != 1){
            *err = ndimErr;
            return -1;
        } else if(dims[0] == 0 || dims[0]*dims[1] >= MAX_DEVICE){
            *err = dimsErr;
            return -1;
        }
        memcpy(cfg->workload, (float*)value, dims[0]*dims[1]* sizeof(float));
    }else{
        static char * unkErr = "Unknown Field Given";
        *err = unkErr;
        return -1;
    }
    return 0;
}

void initialize_output(Config *cfg, int nout) {
	cfg->issave2pt = (nout >= 1);  /** save fluence rate to the 1st output if present */
	cfg->issavedet = (nout >= 2);  /** save detected photon data to the 2nd output if present */
	cfg->issaveseed = (nout >= 4); /** save detected photon seeds to the 4th output if present */
#if defined(USE_MT_RAND)
	cfg->issaveseed = 0;
#endif
	if (nout >= 1) {
        if(cfg->exportfield) free(cfg->exportfield);
		int fieldlen = cfg->dim.x*cfg->dim.y*cfg->dim.z*(int)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);
		cfg->exportfield = (float*)calloc(fieldlen, sizeof(float));
	}
	if (nout >= 2) {
        if(cfg->exportdetected) free(cfg->exportdetected);
		cfg->exportdetected = (float*)malloc((cfg->medianum + 1 + cfg->issaveexit * 6)*cfg->maxdetphoton*sizeof(float));
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


void* mcx_get_field(Config *cfg, const char *key, char** dtype, int* ndim, unsigned *dims, const char**err) {
	static char * intType = "int";
	static char * uintType = "unsigned int";
	static char * floatType = "float";
	static char * doubleType = "double";
	static char * uint8Type = "uint8";

	if (strcmp(key, "exportfield") == 0) {
		*dtype = floatType;
		*ndim = 4;
		dims[0] = cfg->dim.x;
		dims[1] = cfg->dim.y;
		dims[2] = cfg->dim.z;
		dims[3] = (cfg->tend - cfg->tstart) / cfg->tstep + 0.5;
		return cfg->exportfield;
	} else if (strcmp(key, "exportdetected") == 0) {
		*dtype = floatType;
		*ndim = 2;
		dims[0] = cfg->medianum + 1 + cfg->issaveexit * 6;
		dims[1] = cfg->detectedcount;
		return cfg->exportdetected;
	} else if (strcmp(key, "seeddata")) {
		*dtype = uint8Type;
		*ndim = 2;
		dims[0] = (cfg->issaveseed>0)*RAND_WORD_LEN*sizeof(float);
		dims[1] = cfg->detectedcount;
		return cfg->seeddata;
	} else if (strcmp(key, "exportdebugdata")) {
		*dtype = floatType;
		*ndim = 2;
		dims[0] = MCX_DEBUG_REC_LEN;
		dims[1] = cfg->debugdatalen;
		return cfg->exportdebugdata;
	} else if (strcmp(key, "runtime")) {
		*dtype = uintType;
		*ndim = 0;
		return &cfg->runtime;
	} else if (strcmp(key, "nphoton")) {
		*dtype = intType;
		*ndim = 0;
		return &cfg->nphoton;
	} else if (strcmp(key, "energytot")) {
		*dtype = doubleType;
		*ndim = 0;
		return &cfg->energytot;
	} else if (strcmp(key, "energyabs")) {
		*dtype = doubleType;
		*ndim = 0;
		return &cfg->energyabs;
	} else if (strcmp(key, "energyabs")) {
		*dtype = floatType;
		*ndim = 0;
		return &cfg->normalizer;
	} else if (strcmp(key, "workload")) {
		*dtype = floatType;
		*ndim = 1;
		dims[0] = MAX_DEVICE;
		return &cfg->workload;
	}
	static const char * invalidErr = "Wanted key is not yet implemnted.";
	*err = invalidErr;
	return NULL;
}


int mcx_wrapped_run_simulation(Config *cfg, int nout, char**err) {
    GPUInfo *gpuinfo;
	static char * exceptionErr = "MCX Terminated due to an exception!";
	int threadid = 0, errorflag = 0;
    int activedev = mcx_list_gpu(cfg, &gpuinfo);
    if(activedev == 0){
        static char * noGpuErr = "No active GPU device found";
        *err = noGpuErr;
        mcx_cleargpuinfo(&gpuinfo);
        return -1;
    }
    if(mcx_validateconfig(cfg, err, 0, NULL, NULL)) {
        mcx_cleargpuinfo(&gpuinfo);
        return -1;
    }
    initialize_output(cfg, nout);
#ifdef _OPENMP
	omp_set_num_threads(activedev);
#pragma omp parallel shared(errorflag)
	{
		threadid = omp_get_thread_num();
#endif
		/** Enclose all simulation calls inside a try/catch construct for exception handling */
		//try {
			/** Call the main simulation host function to start the simulation */
			mcx_run_simulation(cfg, gpuinfo);
		/*}
		catch (const char *err) {
			printf("Error from thread (%d): %s\n", threadid, err);
			errorflag++;
		}
		catch (const std::exception &err) {
			printf("C++ Error from thread (%d): %s\n", threadid, err.what());
			errorflag++;
		}
		catch (...) {
			printf("Unknown Exception from thread (%d)", threadid);
			errorflag++;
		}*/
#ifdef _OPENMP
	}
#endif
	/** If error is detected, gracefully terminate the mex and return back */
    mcx_cleargpuinfo(&gpuinfo);
	if (errorflag){
		*err = exceptionErr;
		return -1;
	}
	return 0;
}
