/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef NVSHMEMI_COLLECT_COMMON_CPU_H
#define NVSHMEMI_COLLECT_COMMON_CPU_H
#include <cuda.h>
#include <cuda_runtime.h>

#if __cplusplus
extern "C" {
#endif

#define CALL_COLLECT_ON_STREAM_KERN(TYPENAME, TYPE)                                          \
    void call_##TYPENAME##_collect_on_stream_kern(TYPE *dest, const TYPE *source, size_t nelems, \
                                                  int PE_start, int PE_stride, int PE_size,       \
                                                  long *pSync, cudaStream_t stream);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(CALL_COLLECT_ON_STREAM_KERN)

#if __cplusplus
}
#endif

#endif /* NVSHMEMI_COLLECT_COMMON_CPU_H */
