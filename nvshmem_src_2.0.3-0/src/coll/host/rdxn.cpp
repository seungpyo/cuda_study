/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemi_coll.h"
#include "cpu_coll.h"

#define DEFN_NVSHMEMI_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP)                                     \
    void nvshmemi_##TYPENAME##_##OP##_reduce(TYPE *dest, const TYPE *source, int nreduce,        \
                                      int start, int stride, int size, TYPE *pWrk, long *pSync) { \
        call_rdxn_##TYPENAME##_##OP##_on_stream_kern(dest, source, nreduce, start,              \
                                                 stride, size, pWrk, pSync,                     \
                                                 nvshmemi_state->my_stream);                    \
        CUDA_CHECK(cuStreamSynchronize(nvshmemi_state->my_stream));                             \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEMI_TYPENAME_OP_REDUCE, prod)


#define DEFN_NVSHMEM_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP)                                                                 \
    int nvshmem_##TYPENAME##_##OP##_reduce(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {           \
        NVSHMEM_CHECK_STATE_AND_INIT();                                                                         \
        nvshmemi_team_t *teami = nvshmemi_team_pool[team];                                                                  \
        if (nvshmemi_use_nccl && NCCL_REDOP_##OP != -1 && NCCL_DT_##TYPENAME != -1) {                                       \
            NCCL_CHECK(nccl_ftable.AllReduce(source, dest, nreduce, (ncclDataType_t) NCCL_DT_##TYPENAME,                    \
                          (ncclRedOp_t)NCCL_REDOP_##OP, teami->nccl_comm, nvshmemi_state->my_stream));                      \
            CUDA_CHECK(cuStreamSynchronize(nvshmemi_state->my_stream));                                                     \
        }                                                                                                                   \
        else {                                                                                                              \
            TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);                                                    \
            long *pSync = (long *)((long *)pWrk + NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE);                                        \
            nvshmemx_barrier_on_stream(team, nvshmemi_state->my_stream);                                                    \
            nvshmemi_##TYPENAME##_##OP##_reduce(dest, source, nreduce, teami->start, teami->stride, teami->size, pWrk, pSync);\
        }                                                                                                                   \
        return 0;                                                                                                       \
    }

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DEFN_NVSHMEM_TYPENAME_OP_REDUCE, prod)
