/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "cpu_coll.h"

#define DEFN_NVSHMEM_TYPENAME_COLLECT(TYPENAME, TYPE)                                                       \
    int nvshmem_##TYPENAME##_collect(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) {  \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                                     \
        nvshmemi_team_t * teami = nvshmemi_team_pool[team];                                                 \
        if (nvshmemi_use_nccl && NCCL_DT_##TYPENAME != -1) {                                                \
            NCCL_CHECK(nccl_ftable.AllGather(source, dest, nelems, (ncclDataType_t) NCCL_DT_##TYPENAME,     \
                          teami->nccl_comm, nvshmemi_state->my_stream));                                    \
        }                                                                                                   \
        else {                                                                                              \
            nvshmemx_barrier_on_stream(team, nvshmemi_state->my_stream);                                    \
            call_##TYPENAME##_collect_on_stream_kern(dest, source, nelems, teami->start, teami->stride,     \
                                                     teami->size, nvshmemi_team_get_psync(teami, COLLECT),  \
                                                     nvshmemi_state->my_stream);                            \
        }                                                                                                   \
        CUDA_CHECK(cuStreamSynchronize(nvshmemi_state->my_stream));                                         \
        return 0;                                                                                           \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEM_TYPENAME_COLLECT)
