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
	cfg->parentid = mpFFI;
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


int mcx_set_field(Config * cfg, const char *key, const void *value, const char * dtype, int ndim, const unsigned*dims, const char *order, const char**err) {
    static const char *typeErr = "Incorrect dtype given.";
    static const char *ndimErr = "Incorrect number of dimensions given.";
    static const char *dimsErr = "Incorrect shape given.";
	static const char *ordErr = "Order must be either 'C' or 'F'.";
    static const char * strLenErr = "Too long of a string";
	static int seedbyte; // temp workaround

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

	if (ndim >= 2 && (*order != 'C' && *order != 'F')) {
		*err = ordErr;
		return -1;
	}

	if (strcmp(key, "nphoton") == 0) {
		if (cfg->replay.seed != NULL)
			return 0;
		SET_SCALAR_FIELD(nphoton, intV)
	}
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
		cfg->isrowmajor = (*order) == 'C';
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
		if (*order == 'C') {
			for (unsigned i = 0;i < cfg->medianum;i++) {
				cfg->prop[i].mua = ((float*)value)[i * 4];
				cfg->prop[i].mus = ((float*)value)[i * 4 + 1];
				cfg->prop[i].g = ((float*)value)[i * 4 + 2];
				cfg->prop[i].n = ((float*)value)[i * 4 + 3];
			}
		}
		else {
			for (unsigned i = 0;i < cfg->medianum;i++) {
				cfg->prop[i].mua = ((float*)value)[i];
				cfg->prop[i].mus = ((float*)value)[cfg->medianum + i];
				cfg->prop[i].g = ((float*)value)[2*cfg->medianum + i];
				cfg->prop[i].n = ((float*)value)[3*cfg->medianum + i];
			}
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
		if (*order == 'C') {
			for (unsigned i = 0; i < cfg->detnum; i++) {
				cfg->detpos[i].x = ((float *)value)[i * 4];
				cfg->detpos[i].y = ((float *)value)[i * 4 + 1];
				cfg->detpos[i].z = ((float *)value)[i * 4 + 2];
				cfg->detpos[i].w = ((float *)value)[i * 4 + 3];
			}
		} else {
			for (unsigned i = 0; i < cfg->detnum; i++) {
				cfg->detpos[i].x = ((float *)value)[i];
				cfg->detpos[i].y = ((float *)value)[cfg->detnum + i];
				cfg->detpos[i].z = ((float *)value)[2*cfg->detnum + i];
				cfg->detpos[i].w = ((float *)value)[3*cfg->detnum + i];
			}
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
		if (*order == 'F') {
			memcpy(cfg->srcpattern, (float*)value, dims[0] * dims[1] * sizeof(float));
		} else {
			for (unsigned i = 0; i < dims[0]; i++) {
				for (unsigned j = 0; j < dims[1]; j++) {
					cfg->srcpattern[j*dims[1] + i] = ((float*)value)[i*dims[0]+j];
				}
			}
		}
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
		} else if (strcmp(dtype, "float") != 0) {
			*err = typeErr;
			return -1;
		} else if (ndim != 2) {
			*err = ndimErr;
			return -1;
		} else if (cfg->nphoton != dims[1]) {
			*err = dimsErr;
			return -1;
		}

		const float* detps = value;
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
            cfg->seed = intV;
        } else if(ndim == 2){
			if (strcmp(dtype, "int") != 0) {
				*err = typeErr;
				return -1;
			} else if(dims[0]!=sizeof(float)*RAND_WORD_LEN){
				static char * seedErr = "the row number of cfg.seed does not match RNG seed byte-length";
                *err = seedErr;
                return -1;
            }
            seedbyte = dims[0];
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
	static char * uintType = "uint";
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
	char temp_gpu_workaround[MAX_DEVICE];
	memcpy(temp_gpu_workaround, cfg->deviceid, MAX_DEVICE);

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
/*
#ifdef _OPENMP
	omp_set_num_threads(activedev);
#pragma omp parallel shared(errorflag)
	{
		threadid = omp_get_thread_num();
#endif
*/
	int errCode;
	jmp_buf errHandler;
	if ((errCode = setjmp(errHandler)) == 0) {
		mcx_set_error_handler(&errHandler);
		mcx_run_simulation(cfg, gpuinfo);
	} else {
		mcx_set_error_handler(NULL);
		static char * excepErr = "Exception occured while running";
		*err = excepErr;
		return -errCode;
	}
	mcx_set_error_handler(NULL);
/*
#ifdef _OPENMP
	}
#endif
*/
	mcx_cleargpuinfo(&gpuinfo);
	memcpy(cfg->deviceid, temp_gpu_workaround, MAX_DEVICE);
	return 0;
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
    if(cfg->issavedet==0)
        cfg->issaveexit=0;
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
    cfg->his.colcount=cfg->medianum+1+cfg->issaveexit*6; /*column count=maxmedia+2*/

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
