#ifndef MCX_MCX_FFI_H
#define MCX_MCX_FFI_H

#include "mcx_utils.h"

#ifdef _WIN32
#define MCX_EXPORT  __declspec( dllexport )
#else
#define MCX_EXPORT
#endif

#ifdef	__cplusplus
extern "C" {
#endif

MCX_EXPORT Config * mcx_create_config();

MCX_EXPORT void mcx_destroy_config(Config * cfg);

MCX_EXPORT int mcx_set_field(Config *cfg, const char *key, const void *value, const char * dtype, int ndim, const unsigned*dims, const char *order, const char**err);

MCX_EXPORT void* mcx_get_field(Config *cfg, const char *key, char** dtype, int* ndim, unsigned* dims, const char**err);

MCX_EXPORT int mcx_wrapped_run_simulation(Config *cfg, int nout, char**err);

MCX_EXPORT int  mcx_validateconfig(Config *cfg, char **errmsg, int seedbyte, float *detps, int *dimdetps);

#ifdef	__cplusplus
}
#endif


#endif //MCX_MCX_FFI_H
