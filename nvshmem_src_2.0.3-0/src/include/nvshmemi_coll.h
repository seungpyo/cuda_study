/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef NVSHMEMI_COLL_H
#define NVSHMEMI_COLL_H


#ifdef __cplusplus
extern "C" {
#endif

void nvshmemxi_barrier_on_stream(int PE_start, int PE_stride, int PE_size, long *pSync, cudaStream_t stream);

#define DECL_NVSHMEMI_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP)                                            \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemi_##TYPENAME##_##OP##_reduce(TYPE *dest, const TYPE *source, \
                                int nreduce, int start, int stride, int size, TYPE *pWrk, long *pSync);

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DECL_NVSHMEMI_TYPENAME_OP_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DECL_NVSHMEMI_TYPENAME_OP_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(DECL_NVSHMEMI_TYPENAME_OP_REDUCE, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DECL_NVSHMEMI_TYPENAME_OP_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(DECL_NVSHMEMI_TYPENAME_OP_REDUCE, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DECL_NVSHMEMI_TYPENAME_OP_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(DECL_NVSHMEMI_TYPENAME_OP_REDUCE, prod)
#undef DECL_NVSHMEMI_TYPENAME_OP_REDUCE

#define DECL_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP(SC, TYPENAME, TYPE, OP)                       \
    __device__ void nvshmemxi_##TYPENAME##_##OP##_reduce_##SC(TYPE *dest, const TYPE *source,       \
                            int nreduce, int start, int stride, int size, TYPE *pWrk, long *pSync);

#define DECL_NVSHMEMI_REDUCE_THREADGROUP(SC)                                                                 \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, and)    \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, or)     \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, xor)    \
                                                                                                             \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, max)   \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, min)   \
                                                                                                             \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, sum)      \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, prod)

DECL_NVSHMEMI_REDUCE_THREADGROUP(warp)
DECL_NVSHMEMI_REDUCE_THREADGROUP(block)
#undef DECL_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP
#undef DECL_NVSHMEMI_REDUCE_THREADGROUP

#define DECL_NVSHMEMXI_TYPENAME_BROADCAST_THREADGROUP(SC, TYPENAME, TYPE)           \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemxi_##TYPENAME##_broadcast_##SC(TYPE *dest, const TYPE * source, size_t nelems, \
                                               int PE_root, int PE_Start, int PE_stride, int PE_size, long *pSync); 

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_BROADCAST_THREADGROUP, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_BROADCAST_THREADGROUP, block)
#undef DECL_NVSHMEMXI_TYPENAME_BROADCAST_THREADGROUP

#define DECL_NVSHMEMXI_TYPENAME_COLLECT_THREADGROUP(SC, TYPENAME, TYPE)           \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemxi_##TYPENAME##_collect_##SC(TYPE *dest, const TYPE * source, size_t nelems, \
                                               int PE_Start, int PE_stride, int PE_size, long *pSync); 

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_COLLECT_THREADGROUP, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_COLLECT_THREADGROUP, block)
#undef DECL_NVSHMEMXI_TYPENAME_COLLECT_THREADGROUP

#define DECL_NVSHMEMXI_TYPENAME_ALLTOALL_THREADGROUP(SC, TYPENAME, TYPE)           \
    NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemxi_##TYPENAME##_alltoall_##SC(TYPE *dest, const TYPE * source, size_t nelems, \
                                               int PE_Start, int PE_stride, int PE_size, long *pSync); 

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_ALLTOALL_THREADGROUP, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(DECL_NVSHMEMXI_TYPENAME_ALLTOALL_THREADGROUP, block)
#undef DECL_NVSHMEMXI_TYPENAME_ALLTOALL_THREADGROUP

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemi_barrier(int PE_start, int PE_stride, int PE_size, long *pSync, long *counter);

#ifdef __CUDA_ARCH__
__device__ void nvshmemxi_barrier_warp(int PE_start, int PE_stride, int PE_size, long *pSync, long *counter);
__device__ void nvshmemxi_barrier_block(int PE_start, int PE_stride, int PE_size, long *pSync, long *counter);
__device__ void nvshmemi_sync(int PE_start, int PE_stride, int PE_size, long *pSync, long *counter);
__device__ void nvshmemxi_sync_warp(int PE_start, int PE_stride, int PE_size, long *pSync, long *counter);
__device__ void nvshmemxi_sync_block(int PE_start, int PE_stride, int PE_size, long *pSync, long *counter);
#endif

#ifdef __cplusplus
}
#endif

#endif /* NVSHMEMI_COLL_H */
