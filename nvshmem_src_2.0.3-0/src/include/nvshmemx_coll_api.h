/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMX_COLL_API_H_
#define _NVSHMEMX_COLL_API_H_


#ifdef __cplusplus
extern "C" {
#endif
//==========================================
// nvshmem collective calls with stream param
//==========================================

// alltoall(s) collectives
#define DECL_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM(TYPENAME, TYPE)   \
    int nvshmemx_##TYPENAME##_alltoall_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *src, \
                                                  size_t nelem, cudaStream_t stream);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM)
#undef DECL_NVSHMEMX_TYPENAME_ALLTOALL_ON_STREAM

// barrier collectives
int nvshmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream);
void nvshmemx_barrier_all_on_stream(cudaStream_t stream);

// sync collectives
int nvshmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream);
void nvshmemx_sync_all_on_stream(cudaStream_t stream);

// broadcast collectives
#define DECL_NVSHMEMX_TYPENAME_BROADCAST_ON_STREAM(TYPENAME, TYPE)     \
    int nvshmemx_##TYPENAME##_broadcast_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *src, \
                                                   size_t nelem, int PE_root, cudaStream_t stream);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_NVSHMEMX_TYPENAME_BROADCAST_ON_STREAM)
#undef DECL_NVSHMEMX_TYPENAME_BROADCAST_ON_STREAM

// collect and fcollect collectives
#define DECL_NVSHMEMX_TYPENAME_COLLECT_ON_STREAM(TYPENAME, TYPE)        \
    int nvshmemx_##TYPENAME##_collect_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *src, \
                                                 size_t nelem, cudaStream_t stream);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_NVSHMEMX_TYPENAME_COLLECT_ON_STREAM)
#undef DECL_NVSHMEMX_TYPENAME_COLLECT_ON_STREAM

// reduction collectives
#define NVSHMEMI_DECL_REDUCE_ONSTREAM(NAME, TYPE, OP)                           \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmemx_##NAME##_##OP##_reduce_on_stream(  \
            nvshmem_team_t team, TYPE *dest, const TYPE *src, int nreduce,      \
            cudaStream_t stream);

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, prod)

#undef NVSHMEMI_DECL_REDUCE_ONSTREAM


//==========================================
// nvshmem collective calls on threadgroup
//==========================================

// alltoall(s) collectives
#define DECL_NVSHMEMX_TYPENAME_ALLTOALL_SCOPE(SCOPE, TYPENAME, TYPE) \
    __device__ int nvshmemx_##TYPENAME##_alltoall_##SCOPE(nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nelem);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_ALLTOALL_SCOPE, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_ALLTOALL_SCOPE, block)
#undef DECL_NVSHMEMX_TYPENAME_ALLTOALL_SCOPE

// barrier collectives
__device__ int nvshmemx_barrier_warp(nvshmem_team_t team);
__device__ void nvshmemx_barrier_all_warp();
__device__ int nvshmemx_barrier_block(nvshmem_team_t team);
__device__ void nvshmemx_barrier_all_block();

// sync collectives
__device__ int nvshmemx_team_sync_warp(nvshmem_team_t team);
__device__ void nvshmemx_sync_all_warp();
__device__ int nvshmemx_team_sync_block(nvshmem_team_t team);
__device__ void nvshmemx_sync_all_block();

// broadcast collectives
#define DECL_NVSHMEMX_TYPENAME_BROADCAST_SCOPE(SCOPE, TYPENAME, TYPE)   \
    __device__ int nvshmemx_##TYPENAME##_broadcast_##SCOPE(nvshmem_team_t team, TYPE *dest, const TYPE *src, \
                                                            size_t nelem, int PE_root);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_BROADCAST_SCOPE, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_BROADCAST_SCOPE, block)
#undef DECL_NVSHMEMX_TYPENAME_BROADCAST_SCOPE

// collect and fcollect collectives
#define DECL_NVSHMEMX_TYPENAME_COLLECT_SCOPE(SCOPE, TYPENAME, TYPE)   \
    __device__ int nvshmemx_##TYPENAME##_collect_##SCOPE(nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nelem);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_COLLECT_SCOPE, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_COLLECT_SCOPE, block)
#undef DECL_NVSHMEMX_TYPENAME_COLLECT_SCOPE

// reduction collectives
#define DECL_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP(SCOPE, TYPENAME, TYPE, OP) \
    NVSHMEMI_HOSTDEVICE_PREFIX int nvshmemx_##TYPENAME##_##OP##_reduce_##SCOPE(       \
            nvshmem_team_t team, TYPE *dest, const TYPE *src, int nreduce);

#define DECL_NVSHMEMX_TYPENAME_OP_REDUCE(SC)  \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, and)    \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, or)     \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, xor)    \
                                                                                                            \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, max)   \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, min)   \
                                                                                                            \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, sum)      \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, prod)

DECL_NVSHMEMX_TYPENAME_OP_REDUCE(warp);
DECL_NVSHMEMX_TYPENAME_OP_REDUCE(block);

#ifdef __cplusplus
}
#endif

#endif /* NVSHMEMX_COLL_H */
