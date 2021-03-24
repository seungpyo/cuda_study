/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "cpu_coll.h"
#include "nvshmemi_coll.h"

#define DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM(TYPENAME, TYPE, OP)                          \
    int nvshmemx_##TYPENAME##_##OP##_reduce_on_stream(nvshmem_team_t team, TYPE *dest,          \
                                        const TYPE *source, int nreduce, cudaStream_t stream) { \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                         \
        nvshmemi_team_t * teami = nvshmemi_team_pool[team];                                     \
        if (nvshmemi_use_nccl && NCCL_REDOP_##OP != -1 && NCCL_DT_##TYPENAME != -1) {           \
            NCCL_CHECK(nccl_ftable.AllReduce(source, dest, nreduce, (ncclDataType_t) NCCL_DT_##TYPENAME,   \
                          (ncclRedOp_t)NCCL_REDOP_##OP, teami->nccl_comm, stream));             \
        }                                                                                       \
        else {                                                                                  \
            TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);                        \
            long *pSync = (long *)((long *)pWrk + NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE);            \
            nvshmemx_barrier_on_stream(team, stream);                                           \
            call_rdxn_##TYPENAME##_##OP##_on_stream_kern(dest, source, nreduce, teami->start,   \
                                                         teami->stride, teami->size, pWrk, pSync, stream);\
        }                                                                                       \
        return 0;                                                                               \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM, prod)
#undef DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_ON_STREAM
