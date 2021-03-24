/****
 * Copyright (c) 2019, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 ****/

#ifndef NVSHMEM_UTIL_DEBUG_H_
#define NVSHMEM_UTIL_DEBUG_H_

#include "error_codes_internal.h"

static nvshmemResult_t getHostName(char* hostname, int maxlen) {
    if (gethostname(hostname, maxlen) != 0) {
        strncpy(hostname, "unknown", maxlen);
        return NVSHMEMI_SYSTEM_ERROR;
    }
    int i = 0;
    while ((hostname[i] != '.') && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
    hostname[i] = '\0';
    return NVSHMEMI_SUCCESS;
}
#endif
