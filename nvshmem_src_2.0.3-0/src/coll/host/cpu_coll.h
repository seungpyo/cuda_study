/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_COLL_CPU_H
#define NVSHMEMI_COLL_CPU_H 1

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "util.h"
#include "nvshmem_api.h"
#include "nvshmemx_api.h"
#include "nvshmem_internal.h"

#include "rdxn.h"
#include "rdxn_on_stream.h"
#include "alltoall.h"
#include "alltoall_on_stream.h"
#include "barrier.h"
#include "barrier_on_stream.h"
#include "broadcast.h"
#include "broadcast_on_stream.h"
#include "collect.h"
#include "collect_on_stream.h"

/* macro definitions */
#define NVSHMEMI_COLL_CPU_STATUS_SUCCESS 0
#define NVSHMEMI_COLL_CPU_STATUS_ERROR 1

/* function declarations */
int nvshmemi_coll_common_cpu_read_env();
int nvshmemi_coll_common_cpu_init();
int nvshmemi_coll_common_cpu_finalize();

#define NVSHMEMI_COLL_CPU_ERR_POP()                                                        \
    do {                                                                                   \
        fprintf(stderr, "[pe = %d] Error at %s:%d in %s\n", nvshmemi_state->mype, __FILE__, \
                __LINE__, __FUNCTION__);                                                   \
        fflush(stderr);                                                                    \
        goto fn_fail;                                                                      \
    } while (0)

#endif /* NVSHMEMI_COLL_CPU_H */
