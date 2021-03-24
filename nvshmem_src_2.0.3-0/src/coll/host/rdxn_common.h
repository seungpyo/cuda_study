/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_RDXN_COMMON_CPU_H
#define NVSHMEMI_RDXN_COMMON_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

#if __cplusplus
extern "C" {
#endif

#define CALL_RDXN_ON_STREAM_KERN(TYPENAME, TYPE, OP)                                                        \
    extern "C" void call_rdxn_##TYPENAME##_##OP##_on_stream_kern(                                     \
        TYPE *dest, const TYPE *source, int nreduce, int PE_start, int PE_stride, int PE_size,    \
        TYPE *pWrk, long *pSync, cudaStream_t stream);

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(CALL_RDXN_ON_STREAM_KERN, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(CALL_RDXN_ON_STREAM_KERN, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(CALL_RDXN_ON_STREAM_KERN, xor)

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(CALL_RDXN_ON_STREAM_KERN, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(CALL_RDXN_ON_STREAM_KERN, min)

NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(CALL_RDXN_ON_STREAM_KERN, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(CALL_RDXN_ON_STREAM_KERN, prod)

#if __cplusplus
}
#endif

#endif /* NVSHMEMI_RDXN_COMMON_CPU_H */
