/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_ALLTOALL_COMMON_CPU_H
#define NVSHMEMI_ALLTOALL_COMMON_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

#if __cplusplus
extern "C" {
#endif

#define CALL_TYPENAME_ALLTOALL_ON_STREAM_KERN(TYPENAME, TYPE)                                \
    void call_##TYPENAME##_alltoall_on_stream_kern(TYPE *dest, const TYPE *source, size_t nelems, \
                                                   int PE_start, int PE_stride, int PE_size,   \
                                                   long *pSync, cudaStream_t stream);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(CALL_TYPENAME_ALLTOALL_ON_STREAM_KERN)
#undef CALL_TYPENAME_ALLTOALL_ON_STREAM_KERN

#if __cplusplus
}
#endif

#endif /* NVSHMEMI_ALLTOALL_COMMON_CPU_H */
