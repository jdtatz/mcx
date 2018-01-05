#ifndef MCX_MCX_FFI_H
#define MCX_MCX_FFI_H

#include "mcx_utils.h"

enum {
    MCX_INT8,
    MCX_UINT8,
    MCX_INT16,
    MCX_UINT16,
    MCX_INT32,
    MCX_UINT32,
    MCX_INT64,
    MCX_UINT64,
    MCX_FLT32,
    MCX_FLT64,
    MCX_CHR,
    MCX_SHORT,
    MCX_INT,
    MCX_UINT,
    MCX_LONG,
    MCX_STR
};

int mcx_set_field(Config *cfg, const char *key, const void *value, int dtype, int ndim, const unsigned*dims, const char**err);

#endif //MCX_MCX_FFI_H
