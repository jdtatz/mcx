#include <string>
#include "mcx_utils.h"
#include "mcx_shapes.h"
#include "mcx_const.h"
#include "mcx_ffi.h"

#define SET_SCALAR_FIELD(NAME, TYP, DTYP) if(dtype != (DTYP)) {*err=typeErr; return -1;} cfg->NAME = *((TYP*)value);

#define SET_VEC3_FIELD(NAME, TYP, DTYP) if(dtype != (DTYP)) {*err=typeErr; return -1;} \
if(ndim != 1){*err=ndimErr; return -1;} if(dims[0] != 3){*err=dimsErr; return -1;} \
 cfg->NAME.x = ((TYP*)value)[0]; cfg->NAME.y = ((TYP*)value)[1]; cfg->NAME.z = ((TYP*)value)[2];

#define SET_VEC4_FIELD(NAME, TYP, DTYP) if(dtype != (DTYP)) {*err=typeErr; return -1;} \
if(ndim != 1){*err=ndimErr; return -1;} if(dims[0] != 4){*err=dimsErr; return -1;} \
 cfg->NAME.x = ((TYP*)value)[0]; cfg->NAME.y = ((TYP*)value)[1]; cfg->NAME.z = ((TYP*)value)[2]; \
 cfg->NAME.w = ((TYP*)value)[3];

#define SET_VEC34_FIELD(NAME, TYP, DTYP) if(dtype != (DTYP)) {*err=typeErr; return -1;} \
if(ndim != 1){*err=ndimErr; return -1;} if(dims[0] != 3 && dims[0] != 4){*err=dimsErr; return -1;} \
 cfg->NAME.x = ((TYP*)value)[0]; cfg->NAME.y = ((TYP*)value)[1]; cfg->NAME.z = ((TYP*)value)[2]; \
 if(dims[0] == 4) {cfg->NAME.w = ((TYP*)value)[3];}


#define IF_SCALAR_FIELD(NAME, TYP, DTYP) if (strcmp(key, #NAME) == 0) {SET_SCALAR_FIELD(NAME, TYP, DTYP);}
#define ELIF_SCALAR_FIELD(NAME, TYP, DTYP) else if (strcmp(key, #NAME) == 0) {SET_SCALAR_FIELD(NAME, TYP, DTYP);}
#define ELIF_VEC3_FIELD(NAME, TYP, DTYP) else if (strcmp(key, #NAME) == 0) {SET_VEC3_FIELD(NAME, TYP, DTYP);}
#define ELIF_VEC4_FIELD(NAME, TYP, DTYP) else if (strcmp(key, #NAME) == 0) {SET_VEC4_FIELD(NAME, TYP, DTYP);}
#define ELIF_VEC34_FIELD(NAME, TYP, DTYP) else if (strcmp(key, #NAME) == 0) {SET_VEC34_FIELD(NAME, TYP, DTYP);}


int mcx_set_field(Config * cfg, const char *key, const void *value, int dtype, int ndim, const unsigned*dims, const char**err) {
    static const char *typeErr = "Incorrect dtype given.";
    static const char *ndimErr = "Incorrect number of dimensions given.";
    static const char *dimsErr = "Incorrect shape given.";
    static const char * strLenErr = "Too long of a string";

    char *jsonshapes=NULL;
    /* // Unsure Why This Exists
    if(strcmp(name,"nphoton")==0 && cfg->replay.seed!=NULL)
        return 0;
    */


    IF_SCALAR_FIELD(nphoton, int, MCX_INT)
    ELIF_SCALAR_FIELD(nblocksize, unsigned, MCX_UINT)
    ELIF_SCALAR_FIELD(nthread, unsigned, MCX_UINT)
    ELIF_SCALAR_FIELD(tstart, float, MCX_FLT32)
    ELIF_SCALAR_FIELD(tstep, float, MCX_FLT32)
    ELIF_SCALAR_FIELD(tend, float, MCX_FLT32)
    ELIF_SCALAR_FIELD(maxdetphoton, unsigned, MCX_UINT)
    ELIF_SCALAR_FIELD(sradius, float, MCX_FLT32)
    ELIF_SCALAR_FIELD(maxgate, unsigned, MCX_UINT)
    ELIF_SCALAR_FIELD(respin, unsigned, MCX_UINT)
    ELIF_SCALAR_FIELD(isreflect, char, MCX_CHR)
    ELIF_SCALAR_FIELD(isref3, char, MCX_CHR)
    ELIF_SCALAR_FIELD(isrefint, char, MCX_CHR)
    ELIF_SCALAR_FIELD(isnormalized, char, MCX_CHR)
    ELIF_SCALAR_FIELD(isgpuinfo, char, MCX_CHR)
    ELIF_SCALAR_FIELD(issrcfrom0, char, MCX_CHR)
    ELIF_SCALAR_FIELD(autopilot, char, MCX_CHR)
    ELIF_SCALAR_FIELD(minenergy, float, MCX_FLT32)
    ELIF_SCALAR_FIELD(unitinmm, float, MCX_FLT32)
    ELIF_SCALAR_FIELD(reseedlimit, unsigned, MCX_UINT)
    ELIF_SCALAR_FIELD(printnum, unsigned, MCX_UINT)
    ELIF_SCALAR_FIELD(voidtime, int, MCX_INT)
    ELIF_SCALAR_FIELD(issaveseed, char, MCX_CHR)
    ELIF_SCALAR_FIELD(issaveref, char, MCX_CHR)
    ELIF_SCALAR_FIELD(issaveexit, char, MCX_CHR)
    ELIF_SCALAR_FIELD(replaydet, int, MCX_INT)
    ELIF_SCALAR_FIELD(faststep, char, MCX_CHR)
    ELIF_SCALAR_FIELD(maxvoidstep, int, MCX_INT)
    ELIF_SCALAR_FIELD(maxjumpdebug, unsigned, MCX_UINT)
    ELIF_SCALAR_FIELD(gscatter, unsigned, MCX_UINT)
    ELIF_VEC3_FIELD(srcpos, float, MCX_FLT32)
    ELIF_VEC34_FIELD(srcdir, float, MCX_FLT32)
    ELIF_VEC3_FIELD(steps, float, MCX_FLT32)
    ELIF_VEC3_FIELD(crop0, float, MCX_FLT32)
    ELIF_VEC3_FIELD(crop1, float, MCX_FLT32)
    ELIF_VEC4_FIELD(srcparam1, float, MCX_FLT32)
    ELIF_VEC4_FIELD(srcparam2, float, MCX_FLT32)
    else if (strcmp(key, "vol") == 0) {
        cfg->mediabyte = 0;
        size_t size = 4;
        if (dtype == MCX_INT8 || dtype == MCX_UINT8) {
            cfg->mediabyte = 1;
            size = 1;
        } else if (dtype == MCX_INT16 || dtype == MCX_UINT16){
            cfg->mediabyte = 2;
            size = 2;
        } else if (dtype == MCX_INT32 || dtype == MCX_UINT32){
            cfg->mediabyte = 4;
        }else if(dtype == MCX_FLT64){
            cfg->mediabyte=8;
            size = 8;
        }else if(dtype == MCX_FLT32){
            cfg->mediabyte=14;
        }
        if(cfg->mediabyte==0 || ndim != 3 ) {
            static const char * volErr = "the 'vol' field must be a 3D integer array";
            *err = volErr;
            return -1;
        }
        cfg->dim.x = dims[0];
        cfg->dim.y = dims[1];
        cfg->dim.z = dims[2];
        int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;
        if(cfg->vol) free(cfg->vol);
        cfg->vol = malloc(dimxyz*sizeof(unsigned int));
        if(cfg->mediabyte==4) {
            memcpy(cfg->vol, value, dimxyz * size);
        } else{
            if(cfg->mediabyte==1){
                unsigned char *data = value;
                for(int i=0;i<dimxyz;i++)
                    cfg->vol[i] = (unsigned int) data[i];
            }else if(cfg->mediabyte==2){
                unsigned short *data = value;
                for(int i=0;i<dimxyz;i++)
                    cfg->vol[i] = (unsigned int) data[i];
            }else if(cfg->mediabyte==8){
                double *data = value;
                for(int i=0;i<dimxyz;i++)
                    cfg->vol[i]= (unsigned int) data[i];
            }else if(cfg->mediabyte==14){
                float *data = value;
                for(int i=0;i<dimxyz;i++)
                    cfg->vol[i]= (unsigned int) data[i];
                cfg->mediabyte=4;
            }
        }
    } else if(strcmp(key, "prop") == 0){
        if(dtype != MCX_FLT32){
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
        cfg->prop = malloc(cfg->medianum*sizeof(Medium));
        for(int i=0;i<cfg->medianum;i++){
            cfg->prop[i].mua = ((float*)value)[i*4  ];
            cfg->prop[i].mus = ((float*)value)[i*4+1];
            cfg->prop[i].g   = ((float*)value)[i*4+2];
            cfg->prop[i].n   = ((float*)value)[i*4+3];
        }
    } else if(strcmp(key, "detpos")==0){
        if(dtype != MCX_FLT32){
            *err = typeErr;
            return -1;
        } if(ndim != 2){
            *err = ndimErr;
            return -1;
        } else if(dims[0] <= 0 || dims[1] != 4){
            *err = dimsErr;
            return -1;
        }
        float *val = value;
        cfg->detnum=dims[0];
        if(cfg->detpos) free(cfg->detpos);
        cfg->detpos = malloc(cfg->detnum*sizeof(float4));
        for(int i=0; i < cfg->detnum; i++){
            cfg->detpos[i].x = val[i*4];
            cfg->detpos[i].y = val[i*4+1];
            cfg->detpos[i].z = val[i*4+2];
            cfg->detpos[i].w = val[i*4+3];
        }
    }
    //TODO: the rest of the options
    /*else if(strcmp(key,"session")==0) {
        if(dtype != MCX_STR){
            *err = typeErr;
            return -1;
        } else if(dims[0] > MAX_SESSION_LENGTH){
            *err = strLenErr;
            return -1;
        }
        strncpy(cfg->session, value, MAX_SESSION_LENGTH);
        cfg->session[MAX_SESSION_LENGTH] = '\0';
    }else if(strcmp(key,"srctype")==0){
        if(dtype != MCX_STR){
            *err = typeErr;
            return -1;
        } else if(dims[0] > MAX_SESSION_LENGTH){
            *err = strLenErr;
            return -1;
        }

        const char *srctypeid[]={"pencil","isotropic","cone","gaussian","planar","pattern","fourier","arcsine","disk","fourierx","fourierx2d","zgaussian","line","slit","pencilarray",""};
        char strtypestr[MAX_SESSION_LENGTH]={'\0'};

        if(!mxIsChar(item) || len==0)
            mexErrMsgTxt("the 'srctype' field must be a non-empty string");
        if(len>MAX_SESSION_LENGTH)
            mexErrMsgTxt("the 'srctype' field is too long");
        int status = mxGetString(item, strtypestr, MAX_SESSION_LENGTH);
        if (status != 0)
            mexWarnMsgTxt("not enough space. string is truncated.");
        cfg->srctype=mcx_keylookup(strtypestr,srctypeid);
        if(cfg->srctype==-1)
            mexErrMsgTxt("the specified source type is not supported");
        printf("mcx.srctype='%s';\n",strtypestr);
    }else if(strcmp(key,"outputtype")==0){
        int len=mxGetNumberOfElements(item);
        const char *outputtype[]={"flux","fluence","energy","jacobian","nscat","wl","wp",""};
        char outputstr[MAX_SESSION_LENGTH]={'\0'};

        if(!mxIsChar(item) || len==0)
            mexErrMsgTxt("the 'outputtype' field must be a non-empty string");
        if(len>MAX_SESSION_LENGTH)
            mexErrMsgTxt("the 'outputtype' field is too long");
        int status = mxGetString(item, outputstr, MAX_SESSION_LENGTH);
        if (status != 0)
            mexWarnMsgTxt("not enough space. string is truncated.");
        cfg->outputtype=mcx_keylookup(outputstr,outputtype);
        if(cfg->outputtype==5 || cfg->outputtype==6) // map wl to jacobian, wp to nscat
            cfg->outputtype-=2;
        if(cfg->outputtype==-1)
            mexErrMsgTxt("the specified output type is not supported");
        printf("mcx.outputtype='%s';\n",outputstr);
    }else if(strcmp(key,"debuglevel")==0){
        int len=mxGetNumberOfElements(item);
        const char debugflag[]={'R','M','P','\0'};
        char debuglevel[MAX_SESSION_LENGTH]={'\0'};

        if(!mxIsChar(item) || len==0)
            mexErrMsgTxt("the 'debuglevel' field must be a non-empty string");
        if(len>MAX_SESSION_LENGTH)
            mexErrMsgTxt("the 'debuglevel' field is too long");
        int status = mxGetString(item, debuglevel, MAX_SESSION_LENGTH);
        if (status != 0)
            mexWarnMsgTxt("not enough space. string is truncated.");
        cfg->debuglevel=mcx_parsedebugopt(debuglevel,debugflag);
        if(cfg->debuglevel==0)
            mexWarnMsgTxt("the specified debuglevel is not supported");
        printf("mcx.debuglevel='%d';\n",cfg->debuglevel);
    }
    else if(strcmp(key,"srcpattern")==0){
        arraydim=mxGetDimensions(item);
        double *val=mxGetPr(item);
        if(cfg->srcpattern) free(cfg->srcpattern);
        cfg->srcpattern=malloc(arraydim[0]*arraydim[1]*sizeof(float));
        for(i=0;i<arraydim[0]*arraydim[1];i++)
            cfg->srcpattern[i]=val[i];
        printf("mcx.srcpattern=[%d %d];\n",arraydim[0],arraydim[1]);
    }else if(strcmp(key,"shapes")==0){
        int len=mxGetNumberOfElements(item);
        if(!mxIsChar(item) || len==0)
            mexErrMsgTxt("the 'shapes' field must be a non-empty string");

        jsonshapes=new char[len+1];
        mxGetString(item, jsonshapes, len+1);
        jsonshapes[len]='\0';
    } else if(strcmp(key,"detphotons")==0){ // TODO: Unknown option, is UnDocumented
        arraydim=mxGetDimensions(item);
        dimdetps[0]=arraydim[0];
        dimdetps[1]=arraydim[1];
        detps=malloc(arraydim[0]*arraydim[1]*sizeof(float));
        memcpy(detps,mxGetData(item),arraydim[0]*arraydim[1]*sizeof(float));
        printf("mcx.detphotons=[%d %d];\n",arraydim[0],arraydim[1]);
    }
    else if(strcmp(key,"seed")==0){
        arraydim=mxGetDimensions(item);
        if(MAX(arraydim[0],arraydim[1])==0)
            mexErrMsgTxt("the 'seed' field can not be empty");
        if(!mxIsUint8(item)){
            double *val=mxGetPr(item);
            cfg->seed=val[0];
            printf("mcx.seed=%d;\n",cfg->seed);
        }else{
            seedbyte=arraydim[0];
            cfg->replay.seed=malloc(arraydim[0]*arraydim[1]);
            if(arraydim[0]!=sizeof(float)*RAND_WORD_LEN)
                mexErrMsgTxt("the row number of cfg.seed does not match RNG seed byte-length");
            memcpy(cfg->replay.seed,mxGetData(item),arraydim[0]*arraydim[1]);
            cfg->seed=SEED_FROM_FILE;
            cfg->nphoton=arraydim[1];
            printf("mcx.nphoton=%d;\n",cfg->nphoton);
        }
    }else if(strcmp(key,"gpuid")==0){
        int len=mxGetNumberOfElements(item);

        if(mxIsChar(item)){
            if(len==0)
                mexErrMsgTxt("the 'gpuid' field must be an integer or non-empty string");
            if(len>MAX_DEVICE)
                mexErrMsgTxt("the 'gpuid' field is too long");
            int status = mxGetString(item, cfg->deviceid, MAX_DEVICE);
            if (status != 0)
                mexWarnMsgTxt("not enough space. string is truncated.");

            printf("mcx.gpuid='%s';\n",cfg->deviceid);
        }else{
            double *val=mxGetPr(item);
            cfg->gpuid=val[0];
            memset(cfg->deviceid,0,MAX_DEVICE);
            if(cfg->gpuid<MAX_DEVICE){
                memset(cfg->deviceid,'0',cfg->gpuid-1);
                cfg->deviceid[cfg->gpuid-1]='1';
            }else
                mexErrMsgTxt("GPU id can not be more than 256");
            printf("mcx.gpuid=%d;\n",cfg->gpuid);
        }
        for(int i=0;i<MAX_DEVICE;i++)
            if(cfg->deviceid[i]=='0')
                cfg->deviceid[i]='\0';
    }
    else if(strcmp(key,"workload")==0){
        double *val=mxGetPr(item);
        arraydim=mxGetDimensions(item);
        if(arraydim[0]*arraydim[1]>MAX_DEVICE)
            mexErrMsgTxt("the workload list can not be longer than 256");
        for(i=0;i<arraydim[0]*arraydim[1];i++)
            cfg->workload[i]=val[i];
        printf("mcx.workload=<<%d>>;\n",arraydim[0]*arraydim[1]);
    }else{
        printf("WARNING: redundant field '%s'\n",name);
    }
    if(jsonshapes){
        Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},0};
        if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
        int status=mcx_parse_shapestring(&grid,jsonshapes);
        delete [] jsonshapes;
        if(status){
            mexErrMsgTxt(mcx_last_shapeerror());
        }
    }*/
    return 0;
}
