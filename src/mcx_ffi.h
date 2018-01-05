#ifndef MCX_MCX_FFI_H
#define MCX_MCX_FFI_H

#include "mcx_utils.h"

#ifdef	__cplusplus
extern "C" {
#endif

int SIZE_OF_CONFIG = sizeof(Config);

int mcx_set_field(Config *cfg, const char *key, const void *value, const char * dtype, int ndim, const unsigned*dims, const char**err);

void* mcx_get_field(Config *cfg, const char *key, char** dtype, int* ndim, const unsigned* dims, const char**err);

void initialize_output(Config *cfg, int nout);

#ifdef	__cplusplus
}
#endif


#endif //MCX_MCX_FFI_H
