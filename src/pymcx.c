/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2018
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics, (in press) 2018.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/


/***************************************************************************//**
\file    pymcx.c

@brief   pyc function for PYMCX
*******************************************************************************/


/* TODO:
 * - Error Handling!!!
 * - More Returns
 * - More arguments accepted
 * - Documentation
 */

#include "Python.h"
#include "numpy/arrayobject.h"
#include <string>
#include "mcx_const.h"
#include "mcx_utils.h"
#include "mcx_core.h"
#include "mcx_shapes.h"

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


void pymcx_set_field(Config *cfg, PyObject *field, PyObject *val);

PyObject *pymcx_run(PyObject *self, PyObject *args);

static PyMethodDef PyMCXMethods[] = {
        {"simulate",  pymcx_run, METH_VARARGS, "Function Doc string."},
        {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef PyMCXModule = {
        PyModuleDef_HEAD_INIT,
        "pymcx",
        "Module Doc string",
        -1,
        PyMCXMethods
};

PyMODINIT_FUNC PyInit_pymcx() {
    return PyModule_Create(&PyMCXModule);
}

PyObject *pymcx_run(PyObject *self, PyObject *args) {
    Config cfg;
    GPUInfo *gpuinfo;
    float *detps;         //! buffer to receive data from cfg.detphotons field
    int    dimdetps[2]={0,0};  //! dimensions of the cfg.detphotons array
    int    seedbyte=0;
    //mxArray    *tmp;
    int        ifield, jstruct;
    int        ncfg, nfields;
    npy_intp   fielddim[4];
    int        activedev=0;
    int        errorflag=0;
    int        threadid=0;
    const char       *outputtag[]={"data"};
    const char       *datastruct[]={"data","stat","dref"};
    const char       *statstruct[]={"runtime","nphoton","energytot","energyabs","normalizer","workload"};
    const char       *gpuinfotag[]={"name","id","devcount","major","minor","globalmem",
                                    "constmem","sharedmem","regcount","clock","sm","core",
                                    "autoblock","autothread","maxgate"};

    PyObject *pycfg;
    if (!PyArg_ParseTuple(args, "O", &pycfg))
        return NULL;
    if(!PyDict_Check(pycfg))
        return NULL;
    mcx_initcfg(&cfg);
    cfg.isrowmajor = 1; // Python is row-major
    detps=NULL;

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(pycfg, &pos, &key, &value)){
        pymcx_set_field(&cfg, key, value);
    }
    mcx_flush(&cfg);

    cfg.issave2pt=1;  /** save fluence rate to the 1st output if present */
    cfg.issavedet=1;  /** save detected photon data to the 2nd output if present */
    cfg.issaveseed=0; /** save detected photon seeds to the 4th output if present */
#if defined(USE_MT_RAND)
    cfg.issaveseed=0;
#endif

    if(!(activedev=mcx_list_gpu(&cfg,&gpuinfo))){
        //mexErrMsgTxt("No active GPU device found");
        return NULL;
    }

    int fieldlen=cfg.dim.x*cfg.dim.y*cfg.dim.z*(int)((cfg.tend-cfg.tstart)/cfg.tstep+0.5);
    cfg.exportfield = (float*)calloc(fieldlen,sizeof(float));
    cfg.exportdetected = (float*)malloc((cfg.medianum+1+cfg.issaveexit*6)*cfg.maxdetphoton*sizeof(float));
    //cfg.seeddata=malloc(cfg.maxdetphoton*sizeof(float)*RAND_WORD_LEN);
    //cfg.exportdebugdata=(float*)malloc(cfg.maxjumpdebug*sizeof(float)*MCX_DEBUG_REC_LEN);

    char *errMsg;
    if(mcx_validateconfig(&cfg, &errMsg, seedbyte, detps, dimdetps)){
        return NULL;
    }

#ifdef _OPENMP
    omp_set_num_threads(activedev);
    #pragma omp parallel shared(errorflag)
    {
    threadid=omp_get_thread_num();
#endif
    /** Enclose all simulation calls inside a try/catch construct for exception handling */
    //try{
    /** Call the main simulation host function to start the simulation */
    mcx_run_simulation(&cfg,gpuinfo);
    /*}catch(const char *err){
        mexPrintf("Error from thread (%d): %s\n",threadid,err);
        errorflag++;
    }catch(const std::exception &err){
        mexPrintf("C++ Error from thread (%d): %s\n",threadid,err.what());
        errorflag++;
    }catch(...){
        mexPrintf("Unknown Exception from thread (%d)",threadid);
        errorflag++;
    }*/
#ifdef _OPENMP
    }
#endif


    // detphoton
    fielddim[0] = cfg.medianum+1+cfg.issaveexit*6;
    fielddim[1] = cfg.detectedcount;
    PyArrayObject * detphoton = PyArray_EMPTY(2, fielddim, NPY_FLOAT32, 0);
    if(cfg.detectedcount>0) {
        detphoton = PyArray_SimpleNewFromData(2, fielddim, NPY_FLOAT32, cfg.exportdetected);
        memcpy(PyArray_DATA(detphoton), cfg.exportdetected, fielddim[0]*fielddim[1]*sizeof(float));
    }
    free(cfg.exportdetected);
    cfg.exportdetected=NULL;

    // fluence
    fielddim[0]= cfg.dim.x;
    fielddim[1]= cfg.dim.y;
    fielddim[2]= cfg.dim.z;
    fielddim[3]= (cfg.tend-cfg.tstart)/cfg.tstep+0.5;
    PyArrayObject * fluence = PyArray_EMPTY(4, fielddim, NPY_FLOAT32, 0);
    fieldlen=fielddim[0]*fielddim[1]*fielddim[2]*fielddim[3];
    memcpy(PyArray_DATA(fluence),cfg.exportfield, fieldlen*sizeof(float));
    free(cfg.exportfield);
    cfg.exportfield=NULL;

    if(detps)
        free(detps);
    mcx_cleargpuinfo(&gpuinfo);
    mcx_clearcfg(&cfg);

    return PyTuple_Pack(2, fluence, detphoton);
}

void pymcx_set_field(Config *cfg, PyObject *field, PyObject *val){
    if(PyUnicode_CompareWithASCIIString(field, "nphoton") == 0){
        cfg->nphoton = PyLong_AsLong(val);
    } else if(PyUnicode_CompareWithASCIIString(field, "vol") == 0){
        int dtype = PyArray_TYPE(val);
        cfg->mediabyte=0;
        if(dtype == NPY_INT8 || dtype == NPY_UINT8)
            cfg->mediabyte=1;
        else if(dtype == NPY_INT16 || dtype == NPY_UINT16)
            cfg->mediabyte=2;
        else if(dtype == NPY_INT32 || dtype == NPY_UINT32)
            cfg->mediabyte=4;
        else if(dtype == NPY_FLOAT64)
            cfg->mediabyte=8;
        else if(dtype == NPY_FLOAT32)
            cfg->mediabyte=14;
        if(cfg->mediabyte==0 || PyArray_NDIM(val) != 3 ) mexErrMsgTxt("the 'vol' field must be a 3D integer array");
        npy_intp* arraydim = PyArray_DIMS(val);
        for(int i=0;i<3;i++) ((unsigned int *)(&cfg->dim))[i]=arraydim[i];
        int dimxyz=cfg->dim.x*cfg->dim.y*cfg->dim.z;
        if(cfg->vol) free(cfg->vol);
        cfg->vol=(unsigned int *)malloc(dimxyz*sizeof(unsigned int));
        if(cfg->mediabyte==4) {
            memcpy(cfg->vol, PyArray_DATA(val), dimxyz * sizeof(unsigned int));
        } else{
            if(cfg->mediabyte==1){
                unsigned char *data=(unsigned char *)PyArray_DATA(val);
                for(int i=0;i<dimxyz;i++)
                    cfg->vol[i]=data[i];
            }else if(cfg->mediabyte==2){
                unsigned short *data=(unsigned short *)PyArray_DATA(val);
                for(int i=0;i<dimxyz;i++)
                    cfg->vol[i]=data[i];
            }else if(cfg->mediabyte==8){
                double *data=(double *)PyArray_DATA(val);
                for(int i=0;i<dimxyz;i++)
                    cfg->vol[i]=data[i];
            }else if(cfg->mediabyte==14){
                float *data=(float *)PyArray_DATA(val);
                for(int i=0;i<dimxyz;i++)
                    cfg->vol[i]=data[i];
                cfg->mediabyte=4;
            }
        }
    } else if(PyUnicode_CompareWithASCIIString(field, "prop") == 0){
        npy_intp* arraydim = PyArray_DIMS(val);
        //if(arraydim[0]>0 && arraydim[1]!=4) mexErrMsgTxt("the 'prop' field must have 4 columns (mua,mus,g,n)");
        float *val=PyArray_DATA(val);
        cfg->medianum=arraydim[0];
        if(cfg->prop) free(cfg->prop);
        cfg->prop=(Medium *)malloc(cfg->medianum*sizeof(Medium));
        for(int i=0;i<cfg->medianum;i++){
            cfg->prop[i].mua = val[i*4  ];
            cfg->prop[i].mus = val[i*4+1];
            cfg->prop[i].g   = val[i*4+2];
            cfg->prop[i].n   = val[i*4+3];
        }
    } else if(PyUnicode_CompareWithASCIIString(field, "tstart") == 0){
        cfg->tstart = PyFloat_AsDouble(val);
    } else if(PyUnicode_CompareWithASCIIString(field, "tstep") == 0){
        cfg->tstep = PyFloat_AsDouble(val);
    } else if(PyUnicode_CompareWithASCIIString(field, "tend") == 0){
        cfg->tend = PyFloat_AsDouble(val);
    } else if(PyUnicode_CompareWithASCIIString(field, "srcpos") == 0){
        cfg->srcpos.x = PyFloat_AsDouble(PySequence_GetItem(val, 0));
        cfg->srcpos.y = PyFloat_AsDouble(PySequence_GetItem(val, 1));
        cfg->srcpos.z = PyFloat_AsDouble(PySequence_GetItem(val, 2));
    } else if(PyUnicode_CompareWithASCIIString(field, "srcdir") == 0){
        cfg->srcdir.x = PyFloat_AsDouble(PySequence_GetItem(val, 0));
        cfg->srcdir.y = PyFloat_AsDouble(PySequence_GetItem(val, 1));
        cfg->srcdir.z = PyFloat_AsDouble(PySequence_GetItem(val, 2));
        if(PySequence_Length(val) > 3)
            cfg->srcdir.w = PyFloat_AsDouble(PySequence_GetItem(val, 3));
    }
}
