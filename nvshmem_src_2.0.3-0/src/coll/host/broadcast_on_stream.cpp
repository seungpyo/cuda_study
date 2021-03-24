/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "cpu_coll.h"

#define DEFN_NVSHMEMX_BROADCAST_ON_STREAM(TYPENAME, TYPE)                                               \
    int nvshmemx_##TYPENAME##_broadcast_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *source,  \
                                                   size_t nelems, int PE_root, cudaStream_t stream) {   \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                                 \
        nvshmemi_team_t *teami = nvshmemi_team_pool[team];                                              \
        if (nvshmemi_use_nccl && NCCL_DT_##TYPENAME != -1) {                                            \
            NCCL_CHECK(nccl_ftable.Broadcast(source, dest, nelems, (ncclDataType_t) NCCL_DT_##TYPENAME, \
                          PE_root, teami->nccl_comm, stream));                                          \
        }                                                                                               \
        else {                                                                                          \
            nvshmemx_barrier_on_stream(team, stream);                                                   \
            call_##TYPENAME##_broadcast_on_stream_kern(dest, source, nelems,                            \
                                                       nvshmemi_team_translate_pe(teami, PE_root, &nvshmemi_team_world),\
                                                       teami->start, teami->stride, teami->size,        \
                                                       nvshmemi_team_get_psync(teami, BCAST), stream);  \
        }                                                                                               \
        return 0;                                                                                       \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEMX_BROADCAST_ON_STREAM)
#undef DEFN_NVSHMEMX_BROADCAST_ON_STREAM
