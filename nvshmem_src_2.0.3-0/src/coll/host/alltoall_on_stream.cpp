/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "cpu_coll.h"

#define DEFN_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM(TYPENAME, TYPE)                                   \
    int nvshmemx_##TYPENAME##_alltoall_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *source, \
                                                  size_t nelems, cudaStream_t stream) {             \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                             \
        nvshmemi_team_t *teami = nvshmemi_team_pool[team];                                          \
        nvshmemx_barrier_on_stream(team, stream);                                                   \
        call_##TYPENAME##_alltoall_on_stream_kern(dest, source, nelems,                             \
                                                  teami->start, teami->stride, teami->size,         \
                                                  nvshmemi_team_get_psync(teami, ALLTOALL),         \
                                                  stream);                                          \
        return 0;                                                                                   \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM)
#undef DEFN_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM
