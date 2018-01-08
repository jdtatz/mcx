#ifndef MCX_MCX_FFI_H
#define MCX_MCX_FFI_H

#include "mcx_utils.h"

#ifdef	__cplusplus
extern "C" {
#endif

MCX_EXPORT int SIZE_OF_CONFIG = sizeof(Config);

MCX_EXPORT int mcx_set_field(Config *cfg, const char *key, const void *value, const char * dtype, int ndim, const unsigned*dims, const char**err);

MCX_EXPORT void* mcx_get_field(Config *cfg, const char *key, char** dtype, int* ndim, unsigned* dims, const char**err);

MCX_EXPORT void initialize_output(Config *cfg, int nout);

MCX_EXPORT int mcx_wrapped_run_simulation(Config *cfg, GPUInfo *gpu, const char**err);

#ifdef	__cplusplus
}
#endif


#endif //MCX_MCX_FFI_H
